"""
Milvus 导入节点
===============
将向量化后的切片数据批量导入 Milvus，自动创建集合和索引（如果不存在）。

模块职责:
    - 校验并过滤待导入数据的完整性（拦截缺失向量的脏数据）
    - 按需创建 Milvus 集合（Schema + 索引）
    - 批量插入数据并回填自增主键 chunk_id
架构说明:
    ImportMilvusNode  ── 节点入口，继承 BaseNode
    ├── _MilvusSchemaBuilder   ── 集合 Schema 构建（纯策略，无副作用）
    ├── _MilvusIndexBuilder    ── 索引参数构建（纯策略，无副作用）
    └── _MilvusBatchInserter   ── 批量插入 + chunk_id 回填
"""
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Sequence

from pymilvus import DataType, MilvusClient

from knowledge.processor.import_process.base import BaseNode, T, setup_logging
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import MilvusError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.tools.embedding_utils import logger
from knowledge.tools.milvus_utils import get_milvus_client


@dataclass(frozen=True)
class ScalarFieldSpec:
    name: str
    datatype: DataType
    max_length: int = None


# 我是一个有序且只读的序列
_SCALAR_FIELDS: Sequence[ScalarFieldSpec] = (
    ScalarFieldSpec(name="content", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(name="title", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(name="parent_title", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(name="file_title", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(name="item_name", datatype=DataType.VARCHAR, max_length=65535),
)


class _MilvusIndexBuilder:
    @staticmethod
    def build(client: MilvusClient):
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name='dense_vector'
            , index_name='dense_vector_index'
            , index_type='AUTOINDEX'
            , metric_type='COSINE'
        )

        index_params.add_index(
            field_name='sparse_vector'
            , index_name='sparse_vector_index'
            , index_type='SPARSE_INVERTED_INDEX'
            , metric_type='IP'
        )
        return index_params


# -------------------------------
# Milvus的约束构建器（单一职责） 只构建索引
# -------------------------------

class _MilvusSchemaBuilder:

    @staticmethod
    def build(client: MilvusClient, dim: int):
        # 1. 创建约束
        schema = client.create_schema(enable_dynamic_field=True)
        # 2. 添加主键字段
        schema.add_field(field_name="chunk_id",
                         datatype=DataType.INT64,
                         is_primary=True,
                         auto_id=True
                         )
        # 3. 添加标量字段(避免大量重复代码创建配置类ScalarFieldSpec)
        for spec in _SCALAR_FIELDS:
            kwargs: Dict[str, Any] = {"field_name": spec.name, "datatype": spec.datatype}
            if spec.max_length is not None:
                kwargs['max_length'] = spec.max_length
            schema.add_field(**kwargs)
        # 4. 添加向量字段
        # 4.1 添加稠密字段
        schema.add_field(field_name="dense_vector",
                         datatype=DataType.FLOAT_VECTOR,
                         dim=dim)
        # 4.2 添加稀疏向量
        schema.add_field(field_name="sparse_vector",
                         datatype=DataType.SPARSE_FLOAT_VECTOR
                         )
        return schema


class _MilvusInserter:
    def __init__(self, client: MilvusClient, collection_name: str):
        self._client = client
        self._collection_name = collection_name

    def insert(self, chunks: List[Dict[str, Any]]):
        inserted_result = self._client.insert(self._collection_name, chunks)
        insert_count: int = inserted_result.get('insert_count', 0)
        logger.info(f"成功插入{insert_count}条数据到集合{self._collection_name}")
        inserted_ids = inserted_result.get('ids', [])
        self._fill_chunks_id(chunks, inserted_ids)
        return insert_count

    @staticmethod
    def _fill_chunks_id(chunks: List[Dict[str, Any]], inserted_ids: Optional[List]) -> None:
        if len(inserted_ids) != len(chunks):
            logger.warning("chunk_id回填失败：返回%d个ID，期望%d个", len(inserted_ids), len(chunks))
            return
        for chunk, id in zip(chunks, inserted_ids):
            chunk['chunk_id'] = str(id)


class ImportMilvusNode(BaseNode):
    name = "import_milvus_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 检查数据
        chunks = state.get('chunks')
        if not chunks:
            return state
        # 过滤脏数据
        valid_chunks, vector_dim = self._validate_get_inputs(chunks)
        config = get_config()
        # 获取集合名
        collection_name = getattr(config, 'chunks_collection', 'test_collection')
        try:
            client = get_milvus_client()
            # 确保集合存在
            self._ensure_collection(client, collection_name, vector_dim)
            # 批量插入+回填
            inserter = _MilvusInserter(client, collection_name)
            inserter.insert(valid_chunks)

            state['chunks'] = valid_chunks
        except Exception as exc:
            raise MilvusError(f"Milvus 操作失败: {exc}", node_name=self.name)
        return state

    def _validate_get_inputs(self, chunks: List[Dict[str, Any]]):
        validate_chunk = []
        for i, chunk in enumerate(chunks):
            # 只要有稠密和稀疏向量，就是合法数据
            if chunk.get('dense_vector') and chunk.get("sparse_vector"):
                # 缺少part存入的时候不会加入，有part的会被Milvus自动放进动态字段
                validate_chunk.append(chunk)
            else:
                self.logger.warning(f"发现脏数据：第{i + 1}条切片缺失向量，已拦截丢弃！")

        if not validate_chunk:
            raise MilvusError("所有切片均缺失向量数据，拦截入库", node_name=self.name)
        vector_dim = len(validate_chunk[0].get('dense_vector'))
        return validate_chunk, vector_dim

    def _ensure_collection(self, client, collection_name, vector_dim, delete_flag: bool = False):
        # 1. 安全删除逻辑：必须同时满足“开启了删除开关”且“集合确实存在”
        if delete_flag and client.has_collection(collection_name=collection_name):
            self.logger.warning(f"收到强行清理指令，正在清空旧集合！{collection_name}")
            client.drop_collection(collection_name=collection_name)

        # 2. 存在性检查：如果集合存在（且没被删除），直接放行
        if client.has_collection(collection_name=collection_name):
            self.logger.info(f"集合{collection_name}已存在，跳过创建")
            return
        # 3. 构建约束
        schema = _MilvusSchemaBuilder.build(client, vector_dim)

        index_params = _MilvusIndexBuilder.build(client)

        client.create_collection(
            collection_name=collection_name
            , schema=schema
            , index_params=index_params
        )
        self.logger.info(f"集合{collection_name}创建成功")


# ================================================================== #
#                        兼容 & 测试                                   #
# ================================================================== #

node_import_milvus = ImportMilvusNode()

if __name__ == "__main__":
    """
    独立测试 ImportMilvusNode

    测试流程：
    1. 从上一个节点的输出文件读取状态（带向量的切片数据）
    2. 执行 Milvus 导入
    3. 验证回填的 chunk_id
    4. 将结果保存到临时文件
    """

    setup_logging()

    # ----------------------------------------------------------------
    # Step 1: 配置路径
    # ----------------------------------------------------------------
    temp_dir = r"E:\rag\docretri_rag\knowledge\processor\import_process\import_temp_dir\output\万用表RS-12的使用\auto"

    # 输入：上一个向量化节点处理后的状态
    input_path = os.path.join(temp_dir, "chunks_item_name_vector.json")

    # 输出：导入 Milvus 后的状态（含 chunk_id）
    output_path = os.path.join(temp_dir, "chunks_item_name_vector_ids.json")

    # ----------------------------------------------------------------
    # Step 2: 读取输入数据
    # ----------------------------------------------------------------
    print(f"正在读取输入文件: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    chunks = content.get('chunks', [])
    print(f"读取到 {len(chunks)} 个切片")

    # ----------------------------------------------------------------
    # Step 3: 构建状态并执行处理
    # ----------------------------------------------------------------
    state = {
        "chunks": chunks
    }

    print("\n开始执行 Milvus 导入...")
    result_state = node_import_milvus.process(state)

    # ----------------------------------------------------------------
    # Step 4: 验证回填结果
    # ----------------------------------------------------------------
    output_chunks = result_state.get("chunks", [])
    print(f"\n处理完成，共 {len(output_chunks)} 个切片")

    # 检查 chunk_id 回填情况
    chunks_with_id = sum(1 for c in output_chunks if c.get("chunk_id"))
    chunks_without_id = len(output_chunks) - chunks_with_id

    print(f"\n回填统计:")
    print(f"  - 成功回填 chunk_id: {chunks_with_id} 个")
    print(f"  - 未回填 chunk_id: {chunks_without_id} 个")

    # 打印前 3 个切片的信息
    print("\n前 3 个切片信息:")
    for i, chunk in enumerate(output_chunks[:3]):
        print(f"\n  切片 {i + 1}:")
        print(f"    chunk_id: {chunk.get('chunk_id', '无')}")
        print(f"    title: {chunk.get('title', '')}")
        print(f"    item_name: {chunk.get('item_name', '')}")
        print(f"    content: {chunk.get('content', '')[:50]}...")

    # ----------------------------------------------------------------
    # Step 5: 保存输出文件
    # ----------------------------------------------------------------
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_state, f, ensure_ascii=False, indent=4)

    print(f"\n已保存到: {output_path}")
