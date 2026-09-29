"""
BEG-M3向量化节点
为文档切分生成稠密和稀疏向量
"""
import json
import os
from distutils.command.config import config
from typing import List

from unicodedata import normalize

from knowledge.processor.import_process.base import BaseNode, T, setup_logging
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import EmbeddingError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.tools.embedding_utils import bge_m3_ef, get_bge_m3_embedding_model
from knowledge.tools.normalize_sparse_vector import normalize_sparse_vector


class BgeEmbeddingNode(BaseNode):
    """
    BGE-M3向量化节点
    为每个切片生成稠密向量和稀疏向量，用于后续的向量检索
    """
    name = 'bge-embedding'

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        执行向量化
        Args：
            state：图状态
        Returns：
            更新后的状态
        """
        config = get_config()
        # 获取切片
        chunks = state.get("chunks", [])
        if not isinstance(chunks, list) or not chunks:
            raise EmbeddingError("chunks为空或无效", node_name=self.name)
        self.log_step("step_1", f"开始为{len(chunks)}个切片生成向量")

        # 初始化BGE-M3
        try:
            bge_m3_ef = get_bge_m3_embedding_model()
        except Exception as e:
            raise EmbeddingError(f"初始化BGE-M3失败：{e}", node_name=self.name)
        # 批量处理
        output_data = []
        batch_size = config.embedding_batch_size

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_output = self._process_batch(bge_m3_ef, batch, i, len(chunks))
            output_data.extend(batch_output)

        self.log_step("step_2", f"向量化完成，共{len(output_data)}个切片")
        state['chunks'] = output_data
        return state

    def _process_batch(
            self,
            bge_m3_ef,
            batch: List[dict],
            start_idx: int,
            total: int
    ) -> List[dict]:
        """处理一个批次的切片"""
        try:
            # 构造输入文本：item_name+content
            texts = [
                (doc.get("item_name", "") or "") + "\n" + (doc.get("content", "") or "")
                for doc in batch
            ]

            # 批量生成向量
            embeddings = bge_m3_ef.encode_documents(texts)

            if not embeddings:
                self.logger.warning(f"批次{start_idx + 1}-{start_idx + len(batch)}未能生成向量")
                return batch
            output = []
            for j, doc in enumerate(batch):
                # 提取稠密向量
                dense_vector = embeddings['dense'][j].tolist()

                # 提取稀疏向量
                start = embeddings['sparse'].indptr[j]
                end = embeddings['sparse'].indptr[j + 1]
                token_ids = embeddings['sparse'].indices[start:end].tolist()
                weights = embeddings['sparse'].data[start:end].tolist()
                sparse_dict = dict(zip(token_ids, weights))
                sparse_vector = normalize_sparse_vector(sparse_dict)

                # 构建输出
                item = {
                    "content": doc.get("content", "")
                    , "title": doc.get("title")
                    , "parent_title": doc.get("parent_title", "")
                    , "part": doc.get("part", 0)
                    , "file_title": doc.get("file_title", "")
                    , "item_name": doc.get("item_name", "")
                    , "dense_vector": dense_vector
                    , "sparse_vector": sparse_vector
                }
                output.append(item)
            self.logger.info(f"成功处理批次{start_idx + 1}-{min(start_idx + len(batch), total)}/{total}")
            return output
        except Exception as e:
            self.logger.error(f"批次{start_idx + 1}-{start_idx + len(batch)}处理失败：{e}")

            return batch


# ================================================================== #
#                        兼容 & 测试                                   #
# ================================================================== #

# 兼容原有调用方式
node_bge_embedding = BgeEmbeddingNode()

if __name__ == '__main__':
    """
    独立测试 BgeEmbeddingNode

    测试流程：
    1. 从上一个节点的输出文件读取状态
    2. 执行向量化处理
    3. 将结果保存到临时文件
    4. 验证输出数据结构
    """

    setup_logging()

    # ----------------------------------------------------------------
    # Step 1: 配置路径
    # ----------------------------------------------------------------
    temp_dir = r"E:\rag\docretri_rag\knowledge\processor\import_process\import_temp_dir\output\万用表RS-12的使用\auto"

    # 输入：上一个商品识别节点处理后的状态
    input_path = os.path.join(temp_dir, "chunks_item_name.json")

    # 输出：向量化后的状态
    output_path = os.path.join(temp_dir, "chunks_item_name_vector.json")

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

    print("开始执行向量化...")
    result = node_bge_embedding.process(state)

    # ----------------------------------------------------------------
    # Step 4: 验证输出数据
    # ----------------------------------------------------------------
    output_chunks = result.get("chunks", [])
    print(f"\n处理完成，共 {len(output_chunks)} 个切片")

    # 检查第一个切片的向量
    if output_chunks:
        first_chunk = output_chunks[0]

        print("\n第一个切片数据结构:")
        print(f"  - content: {first_chunk.get('content', '')[:50]}...")
        print(f"  - title: {first_chunk.get('title', '')}")
        print(f"  - item_name: {first_chunk.get('item_name', '')}")

        dense_vec = first_chunk.get('dense_vector', [])
        sparse_vec = first_chunk.get('sparse_vector', {})

        print(f"\n向量信息:")
        print(f"  - dense_vector 维度: {len(dense_vec)}")
        print(f"  - dense_vector 前5维: {dense_vec[:5] if dense_vec else '无'}")
        print(f"  - sparse_vector 非零元素数: {len(sparse_vec)}")
        print(f"  - sparse_vector 前3项: {dict(list(sparse_vec.items())[:3]) if sparse_vec else '无'}")

        # 验证稀疏向量是否已归一化
        if sparse_vec:
            import numpy as np

            values = np.array(list(sparse_vec.values()))
            l2_norm = np.linalg.norm(values)
            print(f"  - sparse_vector L2范数: {l2_norm:.6f} (归一化后应接近1.0)")

    # ----------------------------------------------------------------
    # Step 5: 保存输出文件
    # ----------------------------------------------------------------
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print(f"\n已保存到: {output_path}")
