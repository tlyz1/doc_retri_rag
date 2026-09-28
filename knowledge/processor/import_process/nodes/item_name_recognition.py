"""
商品名称识别节点
从文档切片中识别商品/产品名称
"""
from operator import index

from langchain_core.messages import SystemMessage, HumanMessage
from typing import Tuple, List, Optional
from pymilvus import DataType

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.tools.embedding_utils import get_bge_m3_embedding_model
from knowledge.tools.llm_utils import get_llm_client
from knowledge.tools.milvus_utils import get_milvus_client


class ItemNameRecognitionNode(BaseNode):
    """
    商品名称识别节点
    处理流程：
    1. 接收输入验证
    2. 从前几个切片构造识别上下文
    3. 调用 LLM 识别商品名称
    4. 回填 item_name 到 state 和 chunks
    5. 生成商品名称的向量
    6. 保存到 Milvus
    """
    name = "item_name_recognition"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        config = get_config()

        # 输入验证
        file_title, chunks = self._validate_inputs(state)

        # 构造识别上下文
        context = self._build_context(chunks, config.item_name_chunk_k)

        # 调用LLM识别
        item_name = self._recognize_item_name(file_title, context, config)

        # 回填到state和chunks
        self.backfill_item_name(state, chunks, item_name)

        # 生成向量
        dense_vector, sparse_vector = self._generate_vectors(item_name)

        # 保存到 Milvus
        self._save_to_milvus(state, file_title, item_name, dense_vector, sparse_vector, config)

        return state

    def _validate_inputs(self, state: ImportGraphState) -> Tuple[str, List[dict]]:
        """验证输入"""
        self.log_step("step1", "验证输入")
        file_title = state.get("file_title", "")
        chunks = state.get("chunks", [])
        if not file_title:
            raise ValidationError(f"file_title为空", node_name=self.name)
        if not isinstance(chunks, list) or not chunks:
            raise ValidationError(f"chunks为空或者无效", node_name=self.name)

        self.logger.info(f"文件标题：{file_title},切片数:{len(chunks)}")
        return file_title, chunks

    def _build_context(self, chunks: List[dict], k: int, max_chars: int = 2500) -> str:
        """构造识别上下文"""
        self.log_step("step_2", "构造识别上下文")  # 日志：标记当前执行到第二步
        parts = []  # 存放每一条格式化后的文档片段
        total = 0  # 累计字符计数器
        # 遍历前k个chunk（检索返回的前k条结果）
        for i, chunk in enumerate(chunks[:k]):
            if not isinstance(chunk, dict):  # 如果chunk不是字典，跳过脏数据
                continue
            title = (chunk.get("title") or "").strip()  # 取出title，没有就为空字符串，去掉首尾空格
            content = (chunk.get("content") or "").strip()  # 取出content内容
            if not (title or content):  # 标题和内容都为空，直接跳过这条切片
                continue

            # 如果单条content超过800字符，截断，末尾加省略号
            if len(content) > 800:
                content = content[:800] + "..."

            # 格式化单条片段，带上序号、标题、内容
            piece = f"【切片{i + 1}】\n标题：{title}\n内容：{content}"
            parts.append(piece)
            total += len(piece)  # 累加这条片段的字符数
            if total >= max_chars:  # 累计字符到达上限，立刻停止循环，不再加入更多切片
                break
        # 所有片段用两个换行分隔拼接，最后再整体截断一次防止溢出，返回字符串
        return "\n\n".join(parts)[:max_chars]

    def _recognize_item_name(self, file_title: str, context: str, config) -> str:
        """调用LLM识别商品名称"""
        self.log_step("step_3", "调用LLM识别")
        prompt = f"""
请从已下信息中识别出商品名称与型号
文件名{file_title}
正文切片（用于辅助识别）：
{context}
要求：
1.返回内容为字符串形式，最好是带品牌、型号和名称的完整商品名称。比如：尼泊尔5000W大功率电磁炉；
2.返回结果要只包含商品名称，不要添加任何解释和其它内容；
3，如果无法识别商品名称，请返回空字符串
"""
        try:
            llm = get_llm_client(model_name=config.item_model, response_format=False)
            resp = llm.invoke([
                SystemMessage(content="你是商品识别专家，只输出字符串"),
                HumanMessage(content=prompt)
            ])
            item_name = getattr(resp, 'content', "").strip()
            if not item_name:
                self.logger.warning("llm未能识别商品名称，使用文件标题")
                item_name = file_title
            self.logger.info(f"识别结果：{item_name}")
            return item_name
        except Exception as e:
            self.logger.warning(f"LLM调用失败：{e},使用文件标题作为商品名称")
            return file_title

    def backfill_item_name(self, state: ImportGraphState, chunks: List[dict], item_name: str):
        self.log_step("step_4", '回填item_name')
        state['item_name'] = item_name
        for chunk in chunks:
            chunk["item_name"] = item_name
        state['chunks'] = chunks

    def _generate_vectors(self, item_name: str):
        self.log_step("step_5", "生成向量")
        try:
            bg3_m3_ef = get_bge_m3_embedding_model()
            vectors = bg3_m3_ef.encode_documents([item_name])
            if vectors:
                dense_vector = vectors["dense"][0].tolist()

                # 提取稀疏向量
                start_idx = vectors["sparse"].indptr[0]
                end_idx = vectors["sparse"].indptr[1]
                token_ids = vectors["sparse"].indices[start_idx:end_idx].tolist()
                weights = vectors["sparse"].data[start_idx:end_idx].tolist()
                sparse_vector = dict(zip(token_ids, weights))

                self.logger.info("向量生成成功")
                return dense_vector, sparse_vector
        except Exception as e:
            self.logger.warning(f"向量生成失败: {e}")
        return None, None

    def _save_to_milvus(
            self,
            state: ImportGraphState,
            file_title: str,
            item_name: str,
            dense_vector: Optional[List[float]],
            sparse_vector: Optional[dict],
            config
    ):
        """保存到 Milvus"""
        self.log_step("step_6", "保存到 Milvus")

        if not config.milvus_url or not config.item_name_collection:
            self.logger.warning("Milvus 配置不完整跳过保存")
            return
        try:
            client = get_milvus_client()
            collection_name = config.item_name_collection
            if not client.has_collection(collection_name):
                self._create_item_name_collection(client, collection_name)
            data = {
                "file_title": file_title,
                "item_name": item_name
            }
            if dense_vector is not None:
                data["dense_vector"] = dense_vector
            if sparse_vector is not None:
                data["sparse_vector"] = sparse_vector

            result = client.insert(collection_name=collection_name, data=[data])
            self.logger.info(f"已经保存到Milvus，ID:{result['ids'][0]}")
            state['item_name'] = item_name

        except Exception as e:
            self.logger.warning(f"Milvus保存失败：{str(e)}")

    def _create_item_name_collection(self, client, collection_name: str):
        """创建 item_name 集合"""
        self.logger.info(f"创建集合: {collection_name}")
        schema = client.create_schema(enable_dynamic_schema=True)
        # 定义字段
        schema.add_field(field_name="pk", datatype=DataType.VARCHAR,
                         is_primary=True, auto_id=True, max_length=100)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 创建索引
        index_params = client.prepare_index_params()
        index_params.add_index(field_name="dense_vector"
                               , index_name="dense_vector_index"
                               , index_type="AUTOINDEX"
                               , metric_type="COSINE")
        index_params.add_index(field_name="sparse_vector"
                               , index_name="sparse_vector_index"
                               , index_type="AUTOINDEX"
                               , metric_type="IP")

        client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
        self.logger.info(f"集合{collection_name}创建成功")


# ================================================================== #
#                        兼容 & 测试                                   #
# ================================================================== #

# 兼容原有调用方式
node_item_name_recognition = ItemNameRecognitionNode()

if __name__ == '__main__':
    """
    商品名识别节点测试

    测试不同场景下的商品名识别逻辑
    """
    import json
    import os

    from knowledge.processor.import_process.base import setup_logging
    from knowledge.processor.import_process.nodes.item_name_recognition import node_item_name_recognition

    # 1. 开启日志
    setup_logging()

    print("=" * 60)
    print("ItemNameRecognitionNode 节点测试")
    print("=" * 60)

    # -------------------- 测试用例 1: 从 chunks.json 加载 -------------------- #
    print("\n--- 测试用例 1: 从 chunks.json 加载并识别 ---")

    # 获取临时目录
    temp_dir = r"E:\rag\docretri_rag\knowledge\processor\import_process\import_temp_dir\output\万用表RS-12的使用\auto"
    chunk_json_input_path = os.path.join(temp_dir, "chunks.json")

    # 检查文件是否存在
    if os.path.exists(chunk_json_input_path):
        with open(chunk_json_input_path, "r", encoding="utf-8") as f:
            chunk_list = json.load(f)

        # 构建 state 状态
        state = {
            "file_title": "万用表的使用",
            "chunks": chunk_list
        }

        # 调用处理方法
        result = node_item_name_recognition.process(state)

        print(f"\n识别结果:")
        print(f"  item_name: {result.get('item_name', '未识别')}")
        print(f"  chunks 数量: {len(result.get('chunks', []))}")

        # 检查 chunks 是否已回填 item_name
        if result.get("chunks"):
            first_chunk = result["chunks"][0]
            print(f"  首个 chunk 的 item_name: {first_chunk.get('item_name', '未回填')}")

        # 备份结果
        os.makedirs(temp_dir, exist_ok=True)
        output_path = os.path.join(temp_dir, "chunks_item_name.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  已备份到: {output_path}")

    else:
        print(f"    chunks.json 文件不存在: {chunk_json_input_path}")
        print("  请先运行 document_split 节点生成 chunks.json")

    # # -------------------- 测试用例 2: 使用模拟数据 -------------------- #
    # print("\n\n--- 测试用例 2: 使用模拟数据 ---")
    #
    # mock_chunks = [
    #     {
    #         "title": "# 福禄克 15B+ 数字万用表",
    #         "content": "福禄克 15B+ 是一款专业级数字万用表，适用于电子工程师和技术人员。\n\n主要特点：\n- 自动量程\n- 高精度测量\n- 坚固耐用",
    #         "file_title": "万用表说明书"
    #     },
    #     {
    #         "title": "## 产品规格",
    #         "content": "直流电压：0.1mV - 600V\n交流电压：0.1mV - 600V\n电阻：0.1Ω - 40MΩ",
    #         "file_title": "万用表说明书"
    #     },
    #     {
    #         "title": "## 安全须知",
    #         "content": "使用前请仔细阅读本手册。不要测量超过额定值的电压。",
    #         "file_title": "万用表说明书"
    #     }
    # ]
    #
    # mock_state = {
    #     "file_title": "万用表说明书",
    #     "chunks": mock_chunks
    # }
    #
    # mock_result = node_item_name_recognition.process(mock_state)
    #
    # print(f"识别结果:")
    # print(f"  item_name: {mock_result.get('item_name', '未识别')}")
    #
    # # -------------------- 测试用例 3: 空 chunks -------------------- #
    # print("\n\n--- 测试用例 3: 空 chunks (预期抛出异常) ---")
    #
    # try:
    #     empty_state = {
    #         "file_title": "测试文件",
    #         "chunks": []
    #     }
    #     node_item_name_recognition.process(empty_state)
    # except Exception as e:
    #     print(f"捕获到预期异常: {e}")
    #
    # # -------------------- 测试用例 4: 缺少 file_title -------------------- #
    # print("\n\n--- 测试用例 4: 缺少 file_title (预期抛出异常) ---")
    #
    # try:
    #     no_title_state = {
    #         "file_title": "",
    #         "chunks": mock_chunks
    #     }
    #     node_item_name_recognition.process(no_title_state)
    # except Exception as e:
    #     print(f"捕获到预期异常: {e}")
    #
    # print("\n" + "=" * 60)
    # print("测试完成")
    # print("=" * 60)
    #
    #
    #
    #
    #
