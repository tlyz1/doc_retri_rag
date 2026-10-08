"""
Milvus 导入节点
===============
将向量化后的切片数据批量导入 Milvus，自动创建集合和索引（如果不存在）。

模块职责:
    - 校验并过滤待导入数据的完整性（拦截缺失向量的脏数据）
    - 按需创建 Milvus 集合（Schema + 索引）
    - 用确定性主键（chunk_id = sha1(doc_id:切片序号)）+ upsert 写入，保证重复导入幂等
架构说明:
    ImportMilvusNode  ── 节点入口，继承 BaseNode
    ├── _MilvusSchemaBuilder   ── 集合 Schema 构建（纯策略，无副作用）
    ├── _MilvusIndexBuilder    ── 索引参数构建（纯策略，无副作用）
    └── _MilvusInserter        ── 批量写入（先按 doc_id 清旧数据，再 upsert）
"""
import hashlib
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Sequence

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
    ScalarFieldSpec(name="doc_id", datatype=DataType.VARCHAR, max_length=64),
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
        # 2. 添加主键字段（确定性主键：值由节点本地按 doc_id + 切片序号算出来，不用自增）
        schema.add_field(field_name="chunk_id",
                         datatype=DataType.VARCHAR,
                         max_length=64,
                         is_primary=True,
                         auto_id=False
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

    def upsert(self, chunks: List[Dict[str, Any]], doc_id: str):
        """
        写入切片。

        chunk_id 是节点本地算出来的确定性主键，所以同一份文档重复写入 = 覆盖同一批行；
        写入前先按 doc_id 清一次旧数据，用来处理"重跑后切片数变少"留下的尾巴。
        """
        # 1. 先清理这份文档的旧切片
        self._delete_by_doc_id(doc_id)

        # 2. 整行覆盖写入
        self._client.upsert(self._collection_name, chunks)
        logger.info(f"成功写入{len(chunks)}条数据到集合{self._collection_name}（doc_id={doc_id}）")
        return len(chunks)

    def _delete_by_doc_id(self, doc_id: str) -> None:
        try:
            self._client.delete(
                collection_name=self._collection_name,
                filter=f'doc_id == "{doc_id}"',
            )
        except Exception as e:
            # 清理失败就不写，避免在旧数据上再叠一层
            raise MilvusError(f"清理集合{self._collection_name}中doc_id={doc_id}的旧数据失败: {e}")


class ImportMilvusNode(BaseNode):
    name = "import_milvus_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 检查数据
        chunks = state.get('chunks')
        if not chunks:
            return state
        # 文档身份：缺失就直接失败，不做静默兜底（否则幂等无从谈起）
        doc_id = str(state.get('doc_id', '')).strip()
        if not doc_id:
            raise MilvusError("缺少 doc_id，无法保证 Milvus 写入幂等", node_name=self.name)
        # 过滤脏数据
        valid_chunks, vector_dim = self._validate_get_inputs(chunks)
        # 给每个切片打上确定性主键（chunk_id）与文档身份（doc_id）
        self._assign_chunk_identity(valid_chunks, doc_id)
        config = get_config()
        # 获取集合名
        collection_name = getattr(config, 'chunks_collection', 'test_collection')
        try:
            client = get_milvus_client()
            # 确保集合存在
            self._ensure_collection(client, collection_name, vector_dim)
            # 批量写入（先按 doc_id 清旧数据，再 upsert 覆盖）
            inserter = _MilvusInserter(client, collection_name)
            inserter.upsert(valid_chunks, doc_id)

            state['chunks'] = valid_chunks
        except MilvusError:
            raise
        except Exception as exc:
            raise MilvusError(f"Milvus 操作失败: {exc}", node_name=self.name)
        return state

    @staticmethod
    def _assign_chunk_identity(chunks: List[Dict[str, Any]], doc_id: str) -> None:
        """
        给每个切片打上确定性主键与文档身份。

        chunk_id = sha1(doc_id:切片序号)[:32]

        同一份内容 + 同一个序号 → 永远是同一个 chunk_id，
        所以重复上传（或中断后重传）写的是同一行，天然覆盖而不是新增。
        """
        for index, chunk in enumerate(chunks):
            chunk["doc_id"] = doc_id
            chunk["chunk_id"] = hashlib.sha1(
                f"{doc_id}:{index}".encode("utf-8")
            ).hexdigest()[:32]

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
            self._assert_primary_key_compatible(client, collection_name)
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

    @staticmethod
    def _assert_primary_key_compatible(client, collection_name: str) -> None:
        """
        老集合的主键是 INT64 自增，与本次改造的确定性 VARCHAR 主键不兼容。
        这里提前给出明确提示，避免报一个看不懂的 upsert 错误。
        """
        try:
            description = client.describe_collection(collection_name=collection_name)
        except Exception as e:
            logger.warning(f"读取集合{collection_name}结构失败，跳过主键兼容性检查：{e}")
            return

        collection_auto_id = description.get("auto_id")
        for field in description.get("fields", []):
            if field.get("is_primary"):
                pk_type = field.get("type")
                pk_auto_id = field.get("auto_id", collection_auto_id)
                if pk_type != DataType.VARCHAR or pk_auto_id:
                    raise MilvusError(
                        f"集合{collection_name}的主键(auto_id={pk_auto_id}, 类型={pk_type})，"
                        f"与新的确定性主键（VARCHAR + 非自增）不兼容。请删除该集合后重新导入。"
                    )


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
    3. 验证生成的确定性 chunk_id（本地算，不再依赖服务端回填）
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
        "doc_id": "test_doc_id_0001",
        "chunks": chunks
    }

    print("\n开始执行 Milvus 导入...")
    result_state = node_import_milvus.process(state)

    # ----------------------------------------------------------------
    # Step 4: 验证主键生成结果
    # ----------------------------------------------------------------
    output_chunks = result_state.get("chunks", [])
    print(f"\n处理完成，共 {len(output_chunks)} 个切片")

    # 检查 chunk_id 生成情况（确定性主键，本地生成）
    chunks_with_id = sum(1 for c in output_chunks if c.get("chunk_id"))
    chunks_without_id = len(output_chunks) - chunks_with_id

    print(f"\n主键统计:")
    print(f"  - 已生成 chunk_id: {chunks_with_id} 个")
    print(f"  - 未生成 chunk_id: {chunks_without_id} 个")

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
