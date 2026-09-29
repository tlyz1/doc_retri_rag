import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import field,dataclass
from json import JSONDecodeError
from typing import List, Tuple, Dict, Any, Optional, Set

from pymilvus import DataType, MilvusClient

from knowledge.processor.import_process.base import BaseNode, T
from knowledge.processor.import_process.exceptions import MilvusError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.tools.milvus_utils import milvus_client, get_milvus_client
from test.test import result

# ------------------------------------------
# 常量
# ------------------------------------------
MAX_ENTITY_NAME_LENGTH = 15
# ------------------------------------------
# 白名单
# ------------------------------------------
# 实体标签白名单
ALLOWED_ENTITY_LABELS: Set[str] = {
    "Device", "Part", "Operation", "Step",
    "Warning", "Condition", "Tool",
}
# 关系类型白名单
ALLOWED_RELATION_TYPES: Set[str] = ({
    "HAS_OPERATION", "HAS_PART", "HAS_STEP", "USES_TOOL",
    "HAS_WARNING", "NEXT_STEP", "AFFECTS", "REQUIRES",
    "MENTIONED_IN", "RELATED_TO",
})
DEFAULT_RELATION_TYPES = "RELATED_TO"
@dataclass
class ProcessingStats:
    """处理过程统计信息，用于日志和监控。"""

    total_chunks: int = 0
    processed_chunks: int = 0
    failed_chunks: int = 0
    total_entities: int = 0
    total_relations: int = 0
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"处理完成: {self.processed_chunks}/{self.total_chunks} 切片成功, "
            f"{self.failed_chunks} 失败, "
            f"共 {self.total_entities} 实体 / {self.total_relations} 关系"
        )

class KnowLedgeGraphNode(BaseNode):
    name = 'knowledge_graph_node'
    def process(self, state: ImportGraphState) -> ImportGraphState:
        #参数校验
        validate_chunks,item_name=self._validate_get_inputs(state)
        stats=ProcessingStats(total_chunks=len(validate_chunks))
        self.logger.info(f"开始构建知识图谱：{len(validate_chunks)}切片")
        #获取客户端
        milvus_client=get_milvus_client()
        #幂等性校验 （清理旧数据） 如果不做清理：重复导入同一个知识库，相同实体会多次插入 Milvus，检索的时候出现大量重复实体
        self._clear_existing_data(item_name,milvus_client)

        #并发处理每个切片
        self._process_chunks_concurrently(stats,validate_chunks,item_name,milvus_client)

    def _validate_get_inputs(self, state: ImportGraphState) -> Tuple[List[Dict[str, Any]], str]:
        self.log_step("step1", "知识图谱构建参数校验")
        chunks=state.get("chunks") or []
        global_item_name=str(state.get("item_name",'')).strip()

        if not chunks:
            raise ValueError(f"待提取图谱的chunks不存在，跳过图片构建")

        validated_chunks=[]
        for i,chunk in enumerate(chunks):
            if not isinstance(chunk,dict):
                self.logger.warning(f"第{i}个chunk不是字典类型，已抛弃")
                continue

            raw_id=chunk.get("chunk_id")
            chunk_id=str(raw_id).strip() if raw_id is not None else f"kg_chunk_temp_{i}"

            content=str(chunk.get("content",'')).strip()
            if not content:
                self.logger.warning(f"Chunk{chunk_id}缺少content，已抛弃")
                continue
            chunk_item=str(chunk.get("item_name",'')).strip() or global_item_name
            if not chunk_item:
                self.logger.warning(f"Chunk{chunk_id}缺少item_name，已抛弃")

            chunk['chunk_id'] = chunk_id
            chunk['content'] = content
            chunk['item_name'] = chunk_item
            validated_chunks.append(chunk)

        if not validated_chunks:
            raise ValueError(f"经过清洗后，没有任何有效的 chunk（{len(validated_chunks)}）可用于构建图谱。")

        self.logger.info(f"参数校验完成: 原始 {len(chunks)} 块 -> 有效 {len(validated_chunks)} 块。")

        return validated_chunks, global_item_name

    def _clear_existing_data(
            self,
            item_name: str,
            milvus_client: Optional[MilvusClient],
    ) -> None:
        """导入前清理该 item_name 下的所有旧数据（Milvus）。"""

        # 1. 清理 Milvus
        if not milvus_client:
            raise MilvusError("Milvus 客户端获取失败")

        collection_name = self.config.entity_collection
        try:
            if milvus_client.has_collection(collection_name):
                milvus_client.delete(
                    collection_name=collection_name,
                    filter=f'item_name == "{item_name}"',
                )
                self.logger.info(f"Milvus 旧数据已清理: item_name={item_name}")
        except Exception as e:
            raise MilvusError(f"Milvus 清理失败: {e}")

    def _process_chunks_concurrently(
            self,
            stats: ProcessingStats,
            validate_chunks: List[Dict[str, Any]],
            milvus_client,
    ) -> None:
        """使用线程池并发处理所有切片。"""
        with ThreadPoolExecutor(max_workers=4) as pool:
            #提交所有任务
            future_to_idx={}
            for i,chunk in enumerate(validate_chunks):
                content=chunk.get("content",'')
                chunk_id=str(chunk.get("chunk_id",'')).strip()
                chunk_item=chunk.get("item_name",'')
                future=pool.submit(
                    self._process_single_chunk
                    ,content,chunk_id,chunk_item,milvus_client
                )
                future_to_idx[future]=(i,chunk_id)

            #收集结果（按照顺序完成（

    def _process_single_chunk(
            self,
            content: str,
            chunk_id: str,
            item_name: str,
            milvus_client,
    ) -> Tuple[int, int]:
        """处理单个切片：LLM 提取 → 解析清洗 → 写入存储。"""

        # 1. LLM 提取结果（实体、关系）
        llm_response = self._llm_extract_graph_with_retry(content)

        # 2. 解析并清洗 LLM 结果
        graph_data = self._parse_and_clean(llm_response)

        # 3. 获取实体、关系
        entities = graph_data.get("entities")
        relations = graph_data.get("relations")

        # 4. 写入存储
        # 4.1 写入 Milvus
        if entities:
            self._milvus_writer.insert(milvus_client, entities, chunk_id, content, item_name)

        # 4.2 写入 Neo4j（第二天讲解）
        # TODO: neo4j 写入逻辑

        return len(entities), len(relations)



    def _llm_extract_graph_with_retry(self, content: str) -> str:
        """LLM 提取实体、关系，带重试（最多 3 次）。"""
        from knowledge.tools.llm_utils import get_llm_client
        from langchain_core.messages import SystemMessage, HumanMessage
        from knowledge.prompts.upload import import_prompt

        # 1. 获取 LLM 客户端
        llm_client = get_llm_client()
        last_error=None
        for attempt in range(1,4):
            try:
                llm_response=llm_client.invoke([
                    SystemMessage(content=import_prompt.KNOWLEDGE_GRAPH_SYSTEM_PROMPT)
                    ,HumanMessage(content=f"文本切分\n\n{content}")
                ])
                result=getattr(llm_response,'content','').strip()

                if result:
                    return result
            except Exception as e:
                last_error = e
                if attempt < 3:
                    delay = 0.5 * (2 ** (attempt - 1))
                    self.logger.warning(f"LLM 调用失败（第 {attempt} 次），{delay:.1f}s 后重试: {e}")
                    time.sleep(delay)

                # 3. 全部重试失败
            self.logger.error(f"LLM 提取最终失败（3 次）: {last_error}")
            return ""


    def _parse_and_clean(self, llm_response: str) -> Dict[str, Any]:
        """
        1.解析llm返回结果的json代码片段的围栏
        2.反序列化
        3.获取实体信息以及关系信息
        4.分别在清洗实体以及关系
        5. 清洗之后对应的实体和关系返回
        Args:
            llm_response: 模型的输出
        Returns:
             {
                "entities" :[{比较干净的实体名字:标签},{比较干净的实体名字:标签}]
                “relations” :[{比较干净的关系：“head”:"","tail":"","type":""},{比较干净的关系：“head”:"","tail":"","type":""}]
            }
        """
        # 1. 判断
        if not llm_response:
            raise ValueError(f"LLM提取chunk的图谱信息不存在")

        # 2. 清洗json代码块的围栏
        # 2.1 前面的7个非法字符踢掉```json
        # 2.2 后面的3个非法的字符踢掉```
        cleaned = re.sub(r"^```(?:json)?\s*", "", llm_response.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)

        # 3. 反序列化
        try:
            parsed_llm_response: Dict[str, Any] = json.loads(cleaned)
        except  JSONDecodeError as e:
            raise JSONDecodeError(f"反序列化失败 :{str(e)}")
        # 4. 获取信息
        # 4.1 获取实体信息
        entities = parsed_llm_response.get('entities', [])

        # 4.2 获取关系信息
        relations = parsed_llm_response.get('relations', [])

        # 5. 清洗实体
        cleaned_entities = self._clean_entities(entities)

        #获取清洗后的实体名
        cleaned_unique_entity_names={entity.get('name')
        for entity in cleaned_entities
            }
        # 7. 清洗关系
        cleaned_relations = self._clean_relations(cleaned_unique_entity_names, relations)

        # 8. 构建返回字典
        return {"entities": cleaned_entities, "relations": cleaned_relations}


    def _clean_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        1. 清洗无效实体（实体名没有）
        2. 阶段过长的实体名（实体名太长）
        3. 实体的标签是否在白名单中
        4，去重（同名同标签的实体只能存在一份）
        5. 返回
        Args:
            entities: LLM中提取的实体信息  处理同一个切片的
        Returns:
            合法干净的实体信息
        """
        unique_seen = set()
        clean_entities_result = []

        # 1. 遍历所有的实体信息
        for entity in entities:

            # 1.1 获取实体名
            entity_name = str(entity.get('name', '')).strip()

            # 1.2 校验名是否存在
            if not entity_name:
                continue

            #截取过长实体名
            if len(entity_name) > MAX_ENTITY_NAME_LENGTH:
                entity_name=entity_name[:MAX_ENTITY_NAME_LENGTH]

            #获取实体标签
            entity_label=str(entity.get('label', '')).strip()
            if entity_label not in ALLOWED_ENTITY_LABELS:
                continue
            #  去重
            unique_key=(entity_name, entity_label)
            if unique_key  in unique_seen:
                continue
            unique_seen.add(unique_key)

            #构建实体信息
            entity_info={"name": entity_name, "label": entity_label}

            description=str(entity.get('description', '')).strip()
            if description:
                entity_info['description'] = description
            clean_entities_result.append(entity_info)
        return clean_entities_result

    def _clean_relations(self, cleaned_unique_entity_names: Set[str], relations: List[Dict[str, Any]]) -> List[
        Dict[str, Any]]:
        """
        清洗关系:
        1. 清洗关系的头尾节点是否不存在
        2. 截取头尾实体名过长
        3. 校验头尾实体名是否有效（悬空关系处理）
        4. 校验每一个关系的类型是否关系类型的白名单
        5. 返回

        Args:
            cleaned_unique_entity_names: 所有唯一的实体名集合
            relations:  LLM中提取的关系信息

        Returns:
            List[Dict[str,Any]] 合法干净的关系信息

        """
        clean_relations_result = []
        # 1. 遍历所有的关系
        for relation in relations:

            # 1.1 提取头（head）实体名
            head_entity_name = str(relation.get('head', '')).strip()

            # 1.2 提取尾 (tail) 实体名
            tail_entity_name = str(relation.get('tail', '')).strip()

            # 1.3 判断头尾实体是否有任意一个不存在
            if not head_entity_name or not tail_entity_name:
                continue

            # 1.4 判断头尾实体名是否超过阈值
            if len(head_entity_name) > MAX_ENTITY_NAME_LENGTH:
                head_entity_name = head_entity_name[:MAX_ENTITY_NAME_LENGTH]

            if len(tail_entity_name) > MAX_ENTITY_NAME_LENGTH:
                tail_entity_name = tail_entity_name[:MAX_ENTITY_NAME_LENGTH]

            # 1.5 判断头尾实体名是否有效
            if head_entity_name not in cleaned_unique_entity_names or tail_entity_name not in cleaned_unique_entity_names:
                continue

            # 1.6 获取关系类型
            relation_type = str(relation.get('type', '')).strip()

            # 1.7 判断关系类型是否在关系类型的白名单中
            if relation_type not in ALLOWED_RELATION_TYPES:
                # TODO 思路：反哺白名单
                relation_type = DEFAULT_RELATION_TYPES

            # 1.8 构建最终关系链的数据结构
            cleaned_relation = {"head": head_entity_name, "tail": tail_entity_name, "type": relation_type}

            # 1.9 将清洗后最终的关系链放到最终的结果中
            clean_relations_result.append(cleaned_relation)

        return clean_relations_result
























































































