"""查询流程主图

使用 LangGraph 构建知识库查询工作流。
"""

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from dotenv import load_dotenv
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.config import get_config

from knowledge.processor.query_process.nodes.answer_output_node import AnswerOutputNode
from knowledge.processor.query_process.nodes.item_name_confirm_node import ItemNameConfirmNode
from knowledge.processor.query_process.nodes.vector_search_node import VectorSearchNode
from knowledge.processor.query_process.nodes.hyde_search_node import HyDeSearchNode
from knowledge.processor.query_process.nodes.mcp_search_node import McpSearchNode
from knowledge.processor.query_process.nodes.kg_search_node import KnowledgeGraphSearchNode
from knowledge.processor.query_process.nodes.rrf_node import RrfNode
from knowledge.processor.query_process.nodes.rerank_node import RerankNode

# 加载环境变量
load_dotenv()


def route_after_item_confirm(state: QueryGraphState) -> bool:
    """商品名称确认后的路由逻辑。

    根据是否已有答案决定是否跳过搜索直接输出。

    Args:
        state: 查询图状态。

    Returns:
        True 表示已有答案需要跳过搜索，False 表示继续搜索流程。
    """
    if state.get("answer"):
        return True
    return False


def is_local_recall_sufficient(state: QueryGraphState) -> bool:
    """判断本地三路（向量 / HyDE / 知识图谱）召回是否足以支撑作答。

    判定依据：
        1. RRF 融合后的本地切片数是否达到 ``web_fallback_min_local_chunks``（默认 3 条）；
        2. 可选判据：本地最高归一化相似度是否达到 ``web_fallback_min_local_score``
           （默认 0，即不启用——实测本项目知识库的分数不具区分度，详见配置注释）。

    Args:
        state: 已经过 rrf 节点的查询图状态（此时 rrf_chunks / embedding_chunks 等已就绪）。

    Returns:
        True 表示本地召回充足（不需要联网兜底），False 表示需要网络检索补充。
    """
    config = get_config()

    # 判据一：本地融合后的切片数量
    local_chunks = state.get("rrf_chunks") or []
    if len(local_chunks) < config.web_fallback_min_local_chunks:
        return False

    # 判据二（可选）：本地最高归一化相似度
    min_score = config.web_fallback_min_local_score or 0
    if min_score > 0:
        scores = []
        for key in ("embedding_chunks", "hyde_embedding_chunks"):
            for hit in (state.get(key) or []):
                if not isinstance(hit, dict):
                    continue
                try:
                    scores.append(float(hit.get("distance") or 0))
                except (TypeError, ValueError):
                    continue
        if scores and max(scores) < min_score:
            return False

    return True


def route_after_rrf(state: QueryGraphState) -> str:
    """RRF 之后的路由：本地召回不足才走网络兜底检索。

    说明：网页检索原来与本地三路并行执行，并行超步里 MCP 节点看不到本地召回结果，
    无法判断"本地是否不足"；因此把它挪到 rrf 之后，由本函数做条件分支。

    Args:
        state: 已经过 rrf 节点的查询图状态。

    Returns:
        "web"：本地召回不足，先执行 MCP 网络检索再精排；
        "local"：本地召回充足，直接精排（不联网）。
    """
    config = get_config()
    if not config.web_fallback_enabled:
        return "local"
    return "local" if is_local_recall_sufficient(state) else "web"


def create_query_graph() -> CompiledStateGraph:
    """创建查询流程图。

    Returns:
        编译后的 StateGraph 实例。

    流程结构::

        item_name_confirm
              │
              ├── (已有答案) ─────────────────────────────────> answer_output ──> END
              │
              └── (无答案) ─> multi_search ─┬─> search_embedding        ┐
                                            ├─> search_embedding_hyde   ├─> join ─> rrf
                                            └─> query_kg                ┘            │
                                                                                     │
                     ┌─────────────────────(本地召回充足)────────────────────────────┤
                     │                                                               │
                     │ (本地召回不足：web_fallback_enabled 且本地切片数 < 阈值)         │
                     v                                                               v
              web_search_mcp ────────────────────────────────────────────────────> rerank
                                                                                     │
                                                                                     v
                                                                             answer_output ──> END
    """

    # 1. 定义LangGraph工作流
    workflow = StateGraph(QueryGraphState) # type:ignore

    # 2. 实例化节点
    nodes = {
        "item_name_confirm": ItemNameConfirmNode(),
        "multi_search": lambda x: x,   # 虚拟节点
        "search_embedding": VectorSearchNode(),
        "search_embedding_hyde": HyDeSearchNode(),
        "query_kg": KnowledgeGraphSearchNode(),
        "web_search_mcp": McpSearchNode(),
        "join": lambda x: {},  # 多路搜索汇合（虚节点）
        "rrf": RrfNode(),
        "rerank": RerankNode(),
        "answer_output": AnswerOutputNode()

    }

    # 3. 添加节点
    for name, node in nodes.items():
        workflow.add_node(name, node)  # type:ignore

    # 4. 设置入口点
    workflow.set_entry_point("item_name_confirm")

    # 5. 添加条件边：商品名称确认后根据是否有答案路由
    workflow.add_conditional_edges(
        "item_name_confirm",
        route_after_item_confirm,
        {
            False: "multi_search",
            True: "answer_output"
        }
    )

    # 6. 本地三路搜索分发（并行执行）
    #    注意：网页（MCP）检索不在这一步，改为 rrf 之后按"本地召回是否充足"条件触发
    workflow.add_edge("multi_search", "search_embedding")
    workflow.add_edge("multi_search", "search_embedding_hyde")
    workflow.add_edge("multi_search", "query_kg")

    # 7. 多路搜索汇合
    workflow.add_edge("search_embedding", "join")
    workflow.add_edge("search_embedding_hyde", "join")
    workflow.add_edge("query_kg", "join")

    # 8. 顺序边
    workflow.add_edge("join", "rrf")
    # 8.1 本地召回不足时才走网络兜底检索（并行执行时 MCP 看不到本地结果，故改为顺序条件分支）
    workflow.add_conditional_edges(
        "rrf",
        route_after_rrf,
        {
            "local": "rerank",
            "web": "web_search_mcp",
        }
    )
    workflow.add_edge("web_search_mcp", "rerank")
    workflow.add_edge("rerank", "answer_output")
    workflow.add_edge("answer_output", END)

    # 9. 返回可运行的状态
    return workflow.compile()


# 创建全局图实例
query_app = create_query_graph()


if __name__ == "__main__":
    from knowledge.processor.query_process.base import setup_logging
    import json

    setup_logging()

    print("=" * 60)
    print("开始测试: 查询流程主图 (main_graph)")
    print("=" * 60)

    # ---- 测试场景 1：商品名明确，走完整 pipeline ----
    print("\n【场景 1】: 商品名明确，走完整 pipeline")
    print("-" * 60)

    mock_state_1 = {
        "original_query": "RS-12 数字万用表如何测量直流电压？",
        "session_id": "test_session_main_graph",
        "task_id": "test_task_001",
        "is_stream": False,
    }

    print(f"  查询: {mock_state_1['original_query']}")
    print(f"  session_id: {mock_state_1['session_id']}")
    print(f"  is_stream: {mock_state_1['is_stream']}")

    result_1 = query_app.invoke(mock_state_1)

    print(f"\n  【结果】:")
    print(f"  商品名: {result_1.get('item_names')}")
    print(f"  重写查询: {result_1.get('rewritten_query')}")
    answer_1 = result_1.get("answer", "")
    print(f"  答案: {answer_1[:200]}..." if len(answer_1) > 200 else f"  答案: {answer_1}")

    # ---- 测试场景 2：商品名模糊，被拦截 ----
    print("\n\n【场景 2】: 商品名模糊，被拦截返回选项")
    # print("-" * 60)
    #
    # mock_state_2 = {
    #     "original_query": "万用表怎么测电压？",
    #     "session_id": "test_session_main_graph",
    #     "task_id": "test_task_002",
    #     "is_stream": False,
    # }
    #
    # print(f"  查询: {mock_state_2['original_query']}")
    #
    # result_2 = query_app.invoke(mock_state_2)
    #
    # print(f"\n  【结果】:")
    # print(f"  商品名: {result_2.get('item_names')}")
    # answer_2 = result_2.get("answer", "")
    # print(f"  答案: {answer_2}")
    #
    # print("\n" + "=" * 60)
    # print("全部测试完成")
