import asyncio
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import Dict, Any, List, Tuple, Union
from mcp import ClientSession  # mcp 2.x：官方 MCP SDK（venv 已装）
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
from langchain_core.messages import SystemMessage, HumanMessage
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.base import BaseNode, T
from knowledge.processor.query_process.exceptions import StateFieldError


class McpSearchNode(BaseNode):
    name = "mcp_search_node"
    """
     负责从网络查询当前的问题【整个知识库没有找到该问题，兜底的网络结果】
     mcp形式调用第三方的各种通用的搜索工具。
     百度：【电商】商品比价工具、商品搜索的工具、商品全维度对比工具、商品下单的工具 百度搜索工具 百度地图的工具..
     灵积服务平台:的通用搜索工具【bailian_web_search】
     mcp: 本质：就是各大平台把通用的功能，封装成了工具（函数） 然后通过mcp协议 客户端就可以直接调用它。【mcp客户端】---->【mcp服务端：任意选择某一个】

    """

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:

        # 1. 参数校验
        validated_rewritten_query, validated_item_names = self._validate_query_inputs(state)

        # 2. 创建mcp_client 并且让客户端执行工具
        mcp_result = asyncio.run(self._create_execute_web_search(validated_rewritten_query))

        if not mcp_result:
            return state

        # 3. 更新state web_search_docs

        # 4. 只更新state的web_search_docs
        return {"web_search_docs": mcp_result}

    def _validate_query_inputs(self, state: QueryGraphState) -> Tuple[str, List[str]]:

        # 1. 获取state的rewritten_query
        rewritten_query = state.get('rewritten_query', "")

        # 2. 获取state的item_names
        item_names = state.get('item_names', "")

        # 3. 校验
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name="rewritten_query", expected_type=str)

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name="item_names", expected_type=list)

        # 4. 返回
        return rewritten_query, item_names

    async def _create_execute_web_search(self, validated_rewritten_query: str) -> List[Dict[str, Any]]:
        """
        1. 创建mcp客户端
        2. 客户端执行对mcp服务端调用
        别人的接口：1. 发送请求调用别人
        使用客户端要调用服务端【1. 原生发送请求 2. mcp客户端发送请求 3. Agent【mcp客户端发送请求】】
        Agent:代理【代理对象就是程序员】
        Args:
            validated_rewritten_query:
        Returns:
        """

        # 1. 创建 http 客户端：鉴权信息（api_key）放在请求头里
        #    注意：mcp 2.x 的 streamable_http_client 没有 headers 参数，必须通过 http_client 传入
        http_client = create_mcp_http_client(
            headers={"Authorization": f"Bearer {self.config.mcp_dashscope_api_key}"}
        )

        # 2. 建立mcp连接：百炼服务地址是 Streamable HTTP 端点（.../mcps/{服务标识}/mcp），
        #    用 async with 管理生命周期，退出时自动关闭连接
        async with http_client:
            async with streamable_http_client(
                self.config.mcp_dashscope_base_url, http_client=http_client
            ) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    # 2.1 初始化会话（MCP 握手）
                    await session.initialize()

                    # 2.2 执行工具：EnhancedSearch 只提供 search_pro 一个工具，
                    #     且入参只支持 query（多传 count 会返回 PARAM_INVALID）
                    execute_tool_result = await session.call_tool(
                        "search_pro", {"query": validated_rewritten_query}
                    )

                    # 3. 解析工具执行完的结果
                    # 3.1 工具自身报错（此时仍是正常返回，只是 is_error=True）
                    if getattr(execute_tool_result, "is_error", False):
                        self.logger.error(f"MCP工具执行失败:{execute_tool_result.content}")
                        return []

                    # 3.2 获取对象的content属性
                    if not execute_tool_result.content:
                        return []
                    # 3.3 获取TextContent对象的text
                    text_content_text: str = getattr(execute_tool_result.content[0], "text", "")
                    if not text_content_text:
                        return []
                    # 3.4 反序列化
                    try:
                        payload: Dict[str, Any] = json.loads(text_content_text)

                        # a) 获取pages
                        pages = payload.get('pages', "")
                        if not pages:
                            return []
                        search_result = []
                        # b) 遍历得到每一个结果
                        for page in pages:
                            snippet = page.get('snippet', "").strip()
                            title = page.get('title', "").strip()
                            url = page.get('url', "").strip()
                            search_result.append({"snippet": snippet, "title": title, "url": url})
                        # c) 最终返回
                        return search_result
                    except Exception as e:
                        self.logger.error(f"反序列MCP结果失败:{e}")
                        return []


if __name__ == '__main__':
    state = {
        # "rewritten_query": "万用表如何测量电阻",
        "rewritten_query": "万用表如何测量电阻",
        "item_names": ["RS PRO 万用表 RS-12"]  # 对齐
    }

    mcp_search = McpSearchNode()

    result = mcp_search.process(state)

    for r in result.get('web_search_docs'):
        print(json.dumps(r, ensure_ascii=False, indent=2))
