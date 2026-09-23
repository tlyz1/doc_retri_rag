"""
文档切分节点
按 Markdown 标题切分文档，支持二次切分和短内容合并。
融合了原生层级追踪与 LangChain 递归切分算法。
"""
import re
from typing import Tuple, Optional, List

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.exceptions import DocumentSplitError


class DocumentSplitNode(BaseNode):
    """
    文档切分节点
    处理流程：
    1. 读取 MD 内容
    2. 按 Markdown 标题进行一级切分（title 与 body 分离，并发放 parent_title 身份证）
    3. 处理无标题情况兜底
    4. 对超长章节进行二次切分 (引入 LangChain Recursive Split)
    5. 合并过短的相邻章节 (基于 parent_title 同宗同源合并)
    6. 组装最终 content = title + body
    7. 备份与状态更新
    """
    name = 'document_split'

    # ------------------------------------------------------------------ #
    #                           主流程                                     #
    # ------------------------------------------------------------------ #
    def process(self, state: ImportGraphState) -> ImportGraphState:
        config = get_config()
        # 获取输入
        content, file_title, max_length = self._get_inputs(state, config)
        if not content:
            raise DocumentSplitError("md_content 为空", node_name=self.name)

        # 按照一级标题切(带层级追踪)
        sections, has_title = self._split_by_headings(content, file_title)

        return state

    # ------------------------------------------------------------------ #
    #                       Step 1: 获取输入                               #
    # ------------------------------------------------------------------ #
    def _get_inputs(self, state: ImportGraphState, config) -> Tuple[Optional[str], Optional[str], int]:
        # 获取本节点需要的md文本、文件标题、最大长度
        self.log_step("step_1", "获取输入")
        content = state.get('md_content', "")
        if content:
            # 统一换行符，避免正则匹配出Bug
            content = content.replace("\r\n", "\n").replace("\r", "\n")
        file_title = state.get('file_title', "")
        max_length = config.max_content_length
        return content, file_title, max_length

    # ------------------------------------------------------------------ #
    #                  Step 2: 按标题一级切分 (带层级追踪)                   #
    # ------------------------------------------------------------------ #
    def _split_by_headings(self, content: str, file_title: str) -> Tuple[List[str], bool]:
        """
        按 Markdown 标题行切分，title 与 body 分开存储。
        新增特性：向上追踪层级，寻找最近的高级标题作为 parent_title。
        """
        self.log_step("step_2", "按标题切分并追踪层级")
        # 使用括号分组，(#{1,6}) 捕获井号数量即层级，(.+) 捕获标题内容
        heading_re = re.compile(r"^\s*(#{1,6})\s+(.+)")
        lines = content.split("\n")

        sections: List[dict] = []
        current_title = ""
        current_level = 0
        body_lines: List[str] = []
        has_title = False
        in_fence = False  # 标记代码围栏
        hierarchy = [""] * 7 #记录1-6级标题的最新足迹（索引0不用）
        def _flush():
            """将当前积累的内容保存为一个 section，并计算 parent_title"""

        for line in lines:
            # 检测代码围栏（``` 或 ~~~），防止误切代码内部的注释 遇到 ```，如果本来开着就关上，本来关着就打开。


