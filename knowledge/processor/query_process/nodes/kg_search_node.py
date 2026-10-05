"""
node_query_kg — 知识图谱查询节点。

类结构（与导入侧 kg_graph_node.py 的 Writer 模式对称）:
─────────────────────────────────────────────────────────
  _EntityExtractor    LLM 实体抽取 + JSON 解析
  _EntityAligner      Milvus ENTITY_NAME_COLLECTION 实体对齐
  _Neo4jGraphReader   Neo4j 种子节点 / 一跳关系 / chunk 反查
  _ChunkBackfiller    Milvus CHUNKS_COLLECTION chunk 回填
  KGQueryNode         主编排器（组装上述四个组件，执行 pipeline）
─────────────────────────────────────────────────────────
  node_query_kg()     LangGraph 节点入口函数（薄包装）
"""
import logging, re, json

from knowledge.processor.import_process.nodes.knowledge_graph import MAX_ENTITY_NAME_LENGTH
from knowledge.processor.query_process.config import get_config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from json import JSONDecodeError
from typing import List, Dict, Any, Tuple, Union
from pymilvus import MilvusClient
from langchain_core.messages import SystemMessage, HumanMessage
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.base import BaseNode, T
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.tools.llm_utils import get_llm_client
from knowledge.tools.embedding_utils import get_bge_m3_embedding_model, generate_hybrid_embeddings
from knowledge.tools.milvus_utils import get_milvus_client, create_hybrid_search_requests, execute_hybrid_search_query, \
    fetch_chunks_by_chunk_ids
from knowledge.prompts.query.query_prompt import ENTITY_EXTRACT_SYSTEM_PROMPT
from knowledge.tools.neo4j_util import get_neo4j_driver
# -------------------------------------------------
# Neo4J的信息
# -------------------------------------------------
ItemEntityPair = Dict[str, Any]
EntitySeedNode = Dict[str, Any]
OneHopRelation = Dict[str, Any]
#常量
_ENTITY_NAME_MAX_LENGTH = 15
_DEFAULT_ENTITY_NAME_ALIGN = 0.5

# Neo4j的Cypher语句
_CYPHER_EXACT_SEEDS = """
MATCH (n:Entity)
WHERE n.item_name=$item_name AND n.name=$name
RETURN  n.item_name as item_name,n.name as name
LIMIT 1
"""

# toLower()小写
_CYPHER_FUZZY_SEEDS = """
MATCH (n:Entity)
WHERE toLower(n.name) CONTAINS toLower($name)
      AND n.item_name = $item_name
RETURN n.name AS name, n.item_name AS item_name
LIMIT $limit
"""
# 查询种子节点的一跳关系
_CYPHER_ONE_HOP_RELATIONS = """

MATCH (seed:Entity {name:$name,item_name:$item_name})-[r]-(nbr:Entity)

WHERE type(r) <> 'MENTIONED_IN' AND nbr.item_name=$item_name

RETURN 
  CASE WHEN startNode(r)=seed  THEN  seed.name  ELSE nbr.name END AS head,
  type(r) as rel,
  CASE WHEN  startNode(r)=seed  THEN nbr.name ELSE seed.name END AS tail

limit $limit
"""
# -------------------------------------------------
# 工具函数 （服务各个组件、不会污染组件）
# -------------------------------------------------

def _item_name_filter_expr(item_names: List[str]) -> str:
    quoted = ", ".join(f"'{item_name}'" for item_name in item_names)
    return f"item_name in [{quoted}]"

def _clean_parse_llm_content(llm_response_content: str) -> List[str]:
    """
     职责：清洗以及解析LLM输出
    Args:
        llm_response_content:
    Returns:
        List[str]:清洗后的实体名
    """
    # 1. 判断LLM输出内容是否为空
    if not llm_response_content:
        return []
    # 2. 清洗json代码围栏
    text = re.sub(r"^```(?:json)?\s*", "", llm_response_content)
    re_sub = re.sub(r"\s*```$", "", text)

    # 3. 反序列解析
    try:
        deserialized_result: Dict[str, Any] = json.loads(re_sub)
    except JSONDecodeError as e:
        logging.error(f"JSON 反序列失败，原因: {str(e)}")
        return []
    entities_name=deserialized_result.get('entities_name','')
    if not entities_name:
        return []
    if not isinstance(entities_name, list):
        return []

    seen=set()
    entities_name_result=[]
    for entity_name in entities_name:
        if not entity_name:
            continue
        if not isinstance(entity_name, str):
            continue

        truncated_entity_name=truncate_entity_name_length(entity_name)
        # 去重保序【顺序：防御性】
        if truncated_entity_name not in seen:
            seen.add(truncated_entity_name)
            entities_name_result.append(truncated_entity_name)

    return entities_name_result
def truncate_entity_name_length(entity_name: str) -> str:
    name=entity_name.strip()
    return name[:_ENTITY_NAME_MAX_LENGTH] if len(name) > _ENTITY_NAME_MAX_LENGTH else name
def _clean_seed_rows(rows: List[Dict[str, Any]]) -> List[EntitySeedNode]:
    """
    职责：清洗查询种子节点的数据
    Args:
        rows:  查询到的结果记录
    Returns:
        干净的结果记录
    """
    if not rows:
        return []
    clean_seeds_result: List[EntitySeedNode] = []
    # 1. 遍历
    for row in rows:
        # 1.1 获取item_name
        item_name = row.get('item_name', '').strip()
        # 1.2 获取entity_name
        entity_name = row.get('name', '').strip()
        # 1.3 判断
        if not item_name or not entity_name:
            continue
        # 1.4 封装一下
        clean_seeds_result.append({
            "item_name": item_name,
            "entity_name": entity_name
        })
    # 2. 返回
    return clean_seeds_result


class _EntityExtractor:
    """
    实体提取器：
    责任： 利用LLM从查询问题中提取实体
    prompt:设计
    """

    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)
        config = get_config()
        self.model_name = config.default_model

    def extract(self, user_query: str) -> List[str]:
        """
         根据用户问题提取当前问题下的实体名
        Args:
            user_query:  用户问题
        Returns:
            List[str]: 提取后的实体名
        """
        llm_client = get_llm_client(model_name=self.model_name)

        entities_name_extract_system_prompt = ENTITY_EXTRACT_SYSTEM_PROMPT.format(
            MAX_ENTITY_NAME_LENGTH=MAX_ENTITY_NAME_LENGTH
        )

        try:
            llm_response = llm_client.invoke([
                SystemMessage(content=entities_name_extract_system_prompt),
                HumanMessage(content=f'用户问题：{user_query}'),
            ])
            llm_response_content = getattr(llm_response, 'content', '').strip()

            entities_name = _clean_parse_llm_content(llm_response_content)
            return entities_name
        except Exception as e:
            self._logger.error(f"LLM 调用失败:{str(e)}")
            return []

class _EntityAligner:
    """
     实体对齐器：
     责任： 根据LLM提取到的实体名 查询Milvus，获取真正的实体名（对齐后的实体名、能够查询neo4j(查询节点使用)）
    """
    def __init__(self, collection_name: str):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._collection_name = collection_name

    def align(self, entity_names: List[str], item_names: List[str]) -> Dict[str, Any]:
        """
        Args:
            entity_names:  LLM提取的实体名
            item_names: 商品名
        Returns:
         Dict[str,Any]:该字典准备封装两个key.
         第一个key:entities_aligned:[] 所有对齐后的实体名
         第二key:entity_elements[]: 所有对齐后的实体信息[source_id ,distance,origin,aligned,content]
        """
        fallback_result={'entities_aligned_name':[],'entity_aligned_elements':[]}
        if not entity_names:
            return fallback_result
        embedding_model=get_bge_m3_embedding_model()
        if embedding_model is None:
            self._logger.error('嵌入模型不存在')
            return fallback_result
        milvus_client = get_milvus_client()
        if milvus_client is None:
            self._logger.error('Milvus客户端不存在')
            return fallback_result
        embedding_result=generate_hybrid_embeddings(embedding_model=embedding_model,embedding_documents=entity_names)
        if embedding_result is None:
            self._logger.error('嵌入结果为空')
            return fallback_result
        #获取嵌入后的稠密+稀疏向量
        embedding_result_dense=embedding_result['dense']
        embedding_result_sparse=embedding_result['sparse']
        #获取item_name表达式
        item_name_filtered_expr=_item_name_filter_expr(item_names)
        #遍历所有的实体名字
        aligned_entities_name:List[str]=[]
        aligned_entity_elements:List[Dict[str,Any]]=[]

        seen=set()
        for index,entity_name in enumerate(entity_names):
            #对齐一个实体
            align_one_result:List[Dict[str,Any]]=self._align_one(milvus_client,
                                                                 self._collection_name,
                                                                 item_name_filtered_expr,
                                                                 embedding_result_dense,
                                                                 embedding_result_sparse,
                                                                 index,
                                                                 entity_name)
            #将商品对齐结果存储到最终结果中
            aligned_entity_elements.extend(align_one_result)
            #遍历商品下的最齐结果
            for detail in align_one_result:
                aligned_name=detail.get('aligned')
                item_name=detail.get('item_name')
                if aligned_name :
                    #去重 同名实体在不同商品下都保留
                    key=(item_name,aligned_name)
                    if key not in seen:
                        seen.add(key)
                        aligned_entities_name.append(aligned_name)
        self._logger.info(f"对齐后的实体个数 {len(aligned_entities_name)} 实体的名字：{aligned_entities_name}")

        return {
            "entities_aligned_name": aligned_entities_name,
            "entities_aligned_elements": aligned_entity_elements
        }


    def _align_one(self, milvus_client: MilvusClient,
                   _collection_name: str,
                   item_name_filtered_expr: str,
                   embedding_result_dense: List,
                   embedding_result_sparse: List,
                   index: int,
                   entity_name: str) -> List[Dict[str, Any]]:
        """
        对齐指定实体名
        Args:
            milvus_client:
            _collection_name:
            item_name_filtered_expr:
            embedding_result_dense:
            embedding_result_sparse:
            index:
        Returns:
        """
        dense_vector = embedding_result_dense[index]
        sparse_vector = embedding_result_sparse[index]
        # 1. 判断实体名的稠密和稀释向量
        if not dense_vector or not sparse_vector:
            return [{"original": entity_name, "aligned": "", "context": "", "reason": "vector values is not exist "}]

        # 2. 创建混合搜索请求
        hybrid_search_requests=create_hybrid_search_requests(dense_vector=dense_vector,
                                                             sparse_vector=sparse_vector,
                                                             expr=item_name_filtered_expr,
                                                             limit=5)
        #3，执行混合搜索
        reps=execute_hybrid_search_query(milvus_client=milvus_client,
                                         collection_name=_collection_name,
                                         search_requests=hybrid_search_requests,
                                         ranker_weights=(0.4,0.6),
                                         norm_score=True,
                                         limit=5,
                                         output_fields=['source_chunk_id','item_name','context','entity_name']
                                         )
        #解析结果
        hits=reps[0] if reps else []
        if not hits:
            if not hits:
                return [{"original": entity_name, "aligned": "", "score": "", "reason": "no_hit"}]
        #按照item_name 分组，每组获取最高分
        best_by_item:Dict[str,Any]={}
        for hit in hits:
            #获取实体
            entity=hit.get('entity')
            #从实体中获取
            item_name=entity.get('item_name').strip()
            #只保留每个item_name下的第一个（即最高分）
            if item_name not in best_by_item:
                best_by_item[item_name]=hit
            # 4.2 是否有最好的item_name
        if not best_by_item:
            return [{"original": entity_name, "aligned": "", "score": None, "reason": "no_valid_item_name"}]
        #item_name分组输出结果，过滤低于阈值的
        results:List[Dict[str, Any]]=[]
        for item_name,best in best_by_item.items():
            #获取最好的那个分数
            score=best.get('distance')
            if float(score) < float(_DEFAULT_ENTITY_NAME_ALIGN):
                continue
            ent=best.get('entity')
            #将不同商品下最好的实体名添加到结果集合中
            results.append({
                "original": entity_name,
                "aligned": ent.get("entity_name"),
                "score": score,
                "item_name": item_name,
                "source_chunk_id": ent.get("source_chunk_id"),
                "reason": "top1_per_item",
            })

        #  全部低于阈值时返回未命中
        if not results:
            return [{"original": entity_name, "aligned": "", "score": None, "reason": "all_below_threshold"}]

        return results

class _Neo4jGraphReader:
    """
    职责：所有对Neo4j的读操作
    1. 种子节点的查询（1.1 精确查询 1.2 降级走模糊查询兜底 ）
    2. 查询种子节点一跳关系（双向：种子节点指向另外的节点，以及另外的节点指向种子节点） 保留完整的关系
    3. 根据所有的节点（种子节点以及邻居节点）方向查询chunk(item_name，id)
    4. 根据所有的chunk_id 查询milvus得到所有的chunk
    """
    def __init__(self,database:str,
                 kg_max_total_seeds: int,
                 kg_max_seed_candidates,
                 kg_max_triples_per_seed: int,
                 ):
        self._database=database
        self._kg_max_seed_candidates=kg_max_seed_candidates
        self._kg_max_total_seeds=kg_max_total_seeds
        self._logger = logging.getLogger(self.__class__.__name__)
        self._kg_max_triples_per_seed = kg_max_triples_per_seed

    def _session(self):
        neo4j_driver = get_neo4j_driver()
        if neo4j_driver is None:
            raise RuntimeError(
                'Neo4j驱动获取失败'
            )
        return neo4j_driver.session(database=self._database)

    def find_seed_nodes(self, pairs: List[ItemEntityPair]) -> List[EntitySeedNode]:
        """
        职责：根据item_name 以及entity_name 查询种子节点
        策略：精确查询，只返回一条 模糊查询，返回三条
        Args:
            pairs:  _build_item_entity_pairs方法返回的商品名和实体名的pair对
        Returns:
            所有商品名下所有实体名对应的种子节点
        """
        # 1. pair对是否存在
        if not pairs:
            return []
        final_seeds_result:List[EntitySeedNode]=[]

        for pair in pairs:
            item_name=pair.get('item_name').strip()
            entity_name=pair.get('entity_name').strip()
            if not item_name or not entity_name:
                continue
            try:
                with self._session() as session:
                    #执行种子节点查询
                    candidates_seed_nodes=self._execute_seed_nodes(session,item_name,entity_name,
                                                                   self._kg_max_seed_candidates)
                    #将查询到的种子节点加入到最终列表中
                    final_seeds_result.extend(candidates_seed_nodes)
                    #截取种子节点个数，防止下游查询关系的时候性能太差（作用不大）
                    if len(final_seeds_result)>self._kg_max_total_seeds:
                        final_seeds_result=final_seeds_result[:self._kg_max_total_seeds]
                        break
            except Exception as e:
                self._logger.error(f"获取种子节点失败,原因 :{str(e)}")

            self._logger.info(f"获取种子节点 {len(final_seeds_result)} 个")
            return final_seeds_result

    def _execute_seed_nodes(self, session, item_name: str, entity_name: str, _kg_max_seed_candidates: int) -> List[EntitySeedNode]:

        """
         执行种子节点查询
        Args:
            session:  neo4j的驱动
            item_name: 商品名
            entity_name: 实体名
            _kg_max_seed_candidates: 单个商品留下的最大种子节点数
        Returns:
          List[EntitySeedNode] :找到的种子节点
        """
        # 1.精确查询
        exact_rows=session.execute_read(
            lambda tx:tx.run(_CYPHER_EXACT_SEEDS,item_name=item_name,name=entity_name).data()
        )
        if exact_rows:
            return _clean_seed_rows(exact_rows)
        #2.模糊查询
        fuzzy_rows=session.execute_read(
            lambda tx:tx.run(
                _CYPHER_FUZZY_SEEDS,item_name=item_name,name=entity_name,limit=_kg_max_seed_candidates
            ).data()
        )
        return _clean_seed_rows(fuzzy_rows)

    def find_one_hop_relations(self, seed_nodes: List[EntitySeedNode]) -> List[OneHopRelation]:
        """
        职责： 根据种子节点查询一跳的关系（双向），并且过滤掉MENTIONED_IN 关系的节点
        注意：1. 去重（不允许同一条边出现多次）只能出现一次。 2.图谱中存储的节点和关系结构是什么 查询的时候一定要和存储的我结构保证一致 3. 邻居节点可以是你在一跳范围内指向的节点也可以别人在一跳范围内指向你的节点
        比如：A->B(类型：认识) A->B(类型：认识) B->A(类型：认识)
        Args:
            seed_nodes: find_seed_nodes:所有种子节点（所有商品的种子节点）
        Returns:
            List[OneHopRelation]:item_name/head/rel/tail

        """
        # 1. 判断种子节点
        if not seed_nodes:
            return []
        seen = set()
        one_hop_relations_final_result=[]
        for seed_node in seed_nodes:
            item_name=seed_node.get('item_name','').strip()
            seed_name=seed_node.get('entity_name','').strip()
            if not item_name or not seed_name:
                continue

            try:
                with self._session() as session:
                    #查询所有种子节点的一跳关系
                    seed_one_hop_relations:List[OneHopRelation]=self._execute_one_hop_relations(session,item_name,seed_name,
                                                                                                self._kg_max_triples_per_seed)
                    if not seed_one_hop_relations:
                        continue

                        # b) 遍历种子节点所有的关系
                    for seed_one_hop_relation in seed_one_hop_relations:
                        # b.1 获取头
                        head = seed_one_hop_relation.get('head')
                        # b.2 获取rel
                        rel = seed_one_hop_relation.get('rel')
                        # b.3 获取tail
                        tail = seed_one_hop_relation.get('tail')
                        # b.4 获取item_name
                        item_name = seed_one_hop_relation.get('item_name')

                        # b.4 去重（同一条边不能重复出现）同一个商品下，不运行有重复的 不同商品下不能叫重复的边
                        # 场景：A节点是种子节点 令居也是种子节点（A节点作为种子查询邻居节点的时候已经把他们的关系查找到了）所以当在以邻居节点为种子查询的时候，就会出现重复的边。因此要过滤掉
                        # 去重key
                        key = (item_name, head, rel, tail)

                        if key not in seen:
                            seen.add(key)
                            one_hop_relations_final_result.append(seed_one_hop_relation)

                        # c) 截取 种子节点的关系，防止超过LLM窗口阈值
                    if len(one_hop_relations_final_result) > self._kg_max_total_triples:
                        one_hop_relations_final_result = one_hop_relations_final_result[:self._kg_max_total_triples]
                        break

                # d) 返回
            except Exception as e:
                self._logger.error(f"查询 {seed_name} 种子节点的一跳关系失败: {str(e)}")
                return []
        self._logger.info(f"查询 {len(seed_nodes)} 个种子节点对应的关系:{len(one_hop_relations_final_result)} 条")
        return one_hop_relations_final_result


def _execute_one_hop_relations(self, session, item_name: str, seed_name: str, kg_max_triples_per_seed: int) -> List[
        OneHopRelation]:
        """
        Args:
            session: neo4j驱动
            item_name: 商品名
            seed_name: 种子节点名字
            kg_max_triples_per_seed:种子节点最大的关系数
        Returns:
            List[OneHopRelation]:种子节点的关系
        """
        one_hop_relations=session.execute_read(
            lambda tx:tx.run(
                _CYPHER_ONE_HOP_RELATIONS,item_name=item_name,name=seed_name,limit=kg_max_triples_per_seed
            ).data()
        )
        if not one_hop_relations:
            return []
        one_hop_relations_result = []
        for one_hop_relation in one_hop_relations:
            # 1 提取head
            head = one_hop_relation.get('head', '').strip()
            # 2 提取rel
            rel = one_hop_relation.get('rel', '').strip()
            # 3 提取tail
            tail = one_hop_relation.get('tail', '').strip()

            # 4 判断是否存在关系链
            if not (head and rel and tail):
                continue

            # 3.5 将关系链添加到最终结果中
            one_hop_relations_result.append({
                "head": head,
                "rel": rel,
                "tail": tail,
                "item_name": item_name
            })
        return one_hop_relations_result




class KnowledgeGraphSearchNode(BaseNode):
    """
      知识图谱查询主编排器。

      职责：
      - 组装四个服务组件（Extractor / Aligner / GraphReader / Backfiller）
      - 按 pipeline 顺序编排调用

      Pipeline:
      ┌──────────┐   ┌──────────┐   ┌────────────┐   ┌──────────┐
         抽取实体  ──▶   对齐实体   ──▶    Neo4j查询   ──▶ 回填chunk
      └──────────┘   └──────────┘   └────────────┘   └──────────┘
      """

    name = "kg_search_node"

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:
        # 1. 参数校验
        validated_query, validated_item_names = self._validate_inputs(state)

        # 2. 执行流水线
        kg_result: Dict[str, Any] = self._run_pipeline(validated_query, validated_item_names)

    def _validate_inputs(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        # 1. 获取参数
        rewritten_query = state.get('rewritten_query', "")
        item_names = state.get('item_names', "")

        # 2. 校验
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name="rewritten_query", expected_type=str)

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name="item_names", expected_type=list)

        # 3. 从重写的问题中踢掉商品名(降噪以及无异议的查询)选择

        user_query = rewritten_query
        # 循环遍历 item_names 里面每一个名称 name
        for name in item_names:
            # 如果 name 是空字符串，跳过，不处理
            if not name:
                continue

            # ========== 重点：构建正则匹配模式 ==========
            # 1. name.replace(" ", "")：把当前名称内部所有空格删掉
            #    例：name = "A B C" → "ABC"
            # 2. re.escape(ch)：对每个字符转义，防止 . * + ? 这类正则特殊字符被当成正则语法
            # 3. r"\s*".join(...)：字符之间插入 \s*，含义【任意数量空白字符（可以0个）】
            #    例："ABC" → A\s*B\s*C
            #    效果：匹配 A B C、AB C、ABC、A  B  C，只要字符顺序对，中间随便多少空格都能命中
            pattern = r"\s*".join(re.escape(ch) for ch in name.replace(" ", ""))

            # ========== 正则替换 ==========
            # 在 user_query 中，把匹配到的 pattern 内容替换为空字符串（删掉）
            # flags=re.IGNORECASE：忽略大小写匹配
            user_query = re.sub(pattern, "", user_query, flags=re.IGNORECASE)

        # 清理结果：把连续多个空白（空格、制表符等）压缩成单个空格，首尾去空格
        user_query = " ".join(user_query.split()).strip()
        # 4. 返回
        return user_query, item_names

    def _run_pipeline(self, validated_query: str, validated_item_names: List[str]) -> Dict[str, Any]:
