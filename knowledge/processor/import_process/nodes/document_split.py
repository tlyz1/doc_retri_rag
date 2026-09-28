"""
文档切分节点
按 Markdown 标题切分文档，支持二次切分和短内容合并。
融合了原生层级追踪与 LangChain 递归切分算法。
"""
import json
import os
import re
from typing import Tuple, Optional, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.exceptions import DocumentSplitError
from knowledge.tools.markdown_utils import MarkdownTableLinearizer


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

        # 处理全文无标题的情况
        if not has_title:
            sections = [{
                "title": "无标题",
                "body": content,
                "file_title": file_title,
                "parent_title": file_title
            }]
            self.logger.info(f"全文无标题，作为单个chunk处理")

        # 二次切分+合并断章节
        sections = self._split_and_merge(sections, max_length, config.min_content_length)

        # 组装最终的content（title+body),清理内部字段
        sections = self._assemble_content(sections)

        # 日志统计
        self._log_summary(content, sections, max_length)
        # 备份
        state["chunks"] = sections
        self._backup_chunks(state, sections)
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
    def _split_by_headings(self, content: str, file_title: str) -> Tuple[List[dict], bool]:
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
        hierarchy = [""] * 7  # 记录1-6级标题的最新足迹（索引0不用）

        def _flush():
            """将当前积累的内容保存为一个 section，并计算 parent_title"""
            body = "\n".join(body_lines).strip()
            # 1.如果没有标题，直接就是正文 2.current_title,没有body
            if current_title or body:
                # 向上寻找最近的“长辈”作为parent_title
                parent_title = ""
                for lvl in range(current_level - 1, 0, -1):
                    # 当你倒序往上翻家谱的时候 必须用 if hierarchy[lvl]: 来睁大眼睛看一看：这个位置到底是有名字的真长辈，还是一个被清空的“空槽位”？
                    if hierarchy[lvl]:
                        parent_title = hierarchy[lvl]
                        break
                # 如果没找到长辈（自己就是H1，或者文档开头无标题段落），自己当家长
                if not parent_title:
                    parent_title = current_title if current_title else file_title
                sections.append({
                    "title": current_title,
                    "body": body,
                    "file_title": file_title,
                    "parent_title": parent_title,
                })

        for line in lines:
            # 检测代码围栏（``` 或 ~~~），防止误切代码内部的注释 遇到 ```，如果本来开着就关上，本来关着就打开。
            if line.strip().startswith("```") or line.strip().startswith("~~~"):
                in_fence = not in_fence
            match = heading_re.match(line) if not in_fence else None
            # 匹配的这一行标题可能是全文第一个标题或者匹配的这一行标题的前面没有标题或者匹配的这一行标题前面有标题
            if match:
                has_title = True
                _flush()
                # 获取当前标题的等级（1-6）
                level = len(match.group(1))
                current_level = level
                current_title = line.strip()
                hierarchy[level] = current_title

                # 登记当前层级的最新足迹(记录当前标题)
                # 重点：出现新的上级标题，其下属的子标题足迹全清空
                for i in range(level + 1, 7):
                    hierarchy[i] = ""
                body_lines = []
            else:
                body_lines.append(line)
        # 处理文档的最后一段
        _flush()
        return sections, has_title

    def _split_and_merge(self, sections: List[dict], max_length: int, min_content_length: int) -> List[dict]:
        self.log_step("step_4", "二次切分和合并")
        if max_length <= 0:
            return sections

        # 对超长章节做二次切分
        split_result: List[dict] = []
        for section in sections:
            split_result.extend(self._split_long_section(section, max_length))

        # 合并过短的相邻文章（仅限 parent_title 相同）
        return self._merge_short_sections(split_result, min_content_length)

    def _split_long_section(self, section: dict, max_length: int) -> List[dict]:
        """
        引入 LangChain 的 RecursiveCharacterTextSplitter 对超长 body 优雅降级切分
        """
        title = section.get("title", "")
        body = section.get("body", "")
        file_title = section.get("file_title", "")
        parent_title = section.get("parent_title", title)

        if "<table>" in body:
            self.logger.info(f"检查到了表格，进行表格切分")
            body = MarkdownTableLinearizer.process(body)

        # 把title计算入总长度
        title_prefix = f"{title}\n\n" if title else ""
        total = len(title_prefix) + len(body)
        # 如果总长度小于最大长度，直接返回
        if total <= max_length:
            return [section]
        # 计算给正文的实际可以字符数
        available = max_length - len(title_prefix)
        # 当遇到标题长度大于最长长度，直接返回
        if available <= 0:
            return [section]

        splitter = RecursiveCharacterTextSplitter(chunk_size=available
                                                  , chunk_overlap=0
                                                  # 优雅降级策略：优先按双换行切，再按单换行，最后按标点和空格
                                                  ,
                                                  separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "]
                                                  )
        pieces = splitter.split_text(body)
        # 如果切分结果小于等于1个，直接返回 没有切开的意思
        if len(pieces) <= 1:
            return [section]
        # 如果切分结果大于1个，则需要合并
        sub_sections: List[dict] = []
        for i, piece in enumerate(pieces):
            sub_sections.append({
                "title": f"{title}-{i + 1}" if title else f"chunk-{i + 1}",
                "body": piece.strip(),
                "file_title": file_title,
                "parent_title": parent_title,
                "part": i + 1,
            })
        return sub_sections

    def _merge_short_sections(
            self, sections: List[dict], min_length: int
    ) -> List[dict]:
        """
        合并过短的相邻子片段（仅限同一 parent_title 下的片段）。
        """
        if not sections:
            return []
        merged: List[dict] = []
        current = sections[0]
        for next_sec in sections[1:]:
            cur_body_len = len(current.get("body", ""))
            # 同宗同源检验（依赖 Step 2 发放的 parent_title）
            same_parent = (
                    current.get("parent_title")
                    and current["parent_title"] == next_sec.get("parent_title")
            )
            if cur_body_len < min_length and same_parent:
                # 合并: 将 next_sec 的 body 追加到 current
                current["body"] = (
                        current.get("body", "").rstrip()
                        + "\n\n"
                        + next_sec.get("body", "").lstrip()
                ).strip()
                # 标题回退为父标题（表示这是一个大综合块）
                current["title"] = current.get("parent_title", current.get("title", ""))
                # 更新 part 编号
                if "part" in next_sec:
                    current["part"] = next_sec["part"]
            else:
                merged.append(current)
                current = next_sec
        merged.append(current)
        return merged

    # ------------------------------------------------------------------ #
    #               Step 5: 组装最终 content                               #
    # ------------------------------------------------------------------ #
    def _assemble_content(self, sections: List[dict]) -> List[dict]:
        """
        将 title + body 组装为最终的 content 字段，
        清理内部临时字段 body，保留 parent_title 和 part 供下游使用。
        """
        self.log_step("step_5", "组装 content")
        result: List[dict] = []
        for sec in sections:
            title = sec.get("title", "")
            body = sec.get("body", "")
            # 组装：title+body
            if title and body:
                content = f"{title}\n\n{body}".strip()
            else:
                content = body or title
            chunk = {
                "title": title,
                "content": content.strip(),
                "file_title": sec.get("file_title", ""),
            }
            # 保留二次切分产生的字段，供下游合并/溯源使用
            if "parent_title" in sec:
                chunk["parent_title"] = sec["parent_title"]
            if "part" in sec:
                chunk["part"] = sec["part"]
            result.append(chunk)
        return result

    # ------------------------------------------------------------------ #
    #                       日志 & 备份                                    #
    # ------------------------------------------------------------------ #
    def _log_summary(self, raw_content: str, sections: List[dict], max_length: int):
        self.log_step("step_6", "输出统计")

        lines_count = raw_content.count("\n") + 1
        self.logger.info(f"原文档行数: {lines_count}")
        self.logger.info(f"最终切分章节数: {len(sections)}")
        self.logger.info(f"最大切片长度: {max_length}")

        if sections:
            self.logger.info("章节预览:")
            for i, sec in enumerate(sections[:5]):
                title = sec.get("title", "")[:50]
                self.logger.info(f"  {i + 1}. {title}...")
            if len(sections) > 5:
                self.logger.info(f"  ... 还有 {len(sections) - 5} 个章节")

    def _backup_chunks(self, state: ImportGraphState, sections: List[dict]):
        self.log_step("step_7", "备份切片")

        # 优先使用 file_dir，兼容 local_dir (避免因为字段命名引发写入失败)
        local_dir = state.get("file_dir", state.get("local_dir", ""))
        if not local_dir:
            self.logger.debug("未设置 file_dir/local_dir，跳过备份")
            return

        try:
            os.makedirs(local_dir, exist_ok=True)
            output_path = os.path.join(local_dir, "chunks.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(sections, f, ensure_ascii=False, indent=2)
            self.logger.info(f"已备份到: {output_path}")
        except Exception as e:
            self.logger.warning(f"备份失败: {e}")


# ================================================================== #
#                        兼容 & 测试                                   #
# ================================================================== #

# 实例化节点
node_document_split = DocumentSplitNode()

if __name__ == '__main__':
    setup_logging()
    # sample_document_path=r"D:\develop\develop\workspace\pycharm\usage\251020\shopkeeper_brain\knowledge\processor\import_process\import_temp_dir\万用表的使用\hybrid_auto\万用表的使用.md"
    # sample_document_path = r"D:\develop\develop\workspace\pycharm\usage\251020\shopkeeper_brain\knowledge\processor\import_process\import_temp_dir\test_hierarchy.md"
    sample_document_path = r"E:\rag\docretri_rag\knowledge\processor\import_process\import_temp_dir\output\万用表RS-12的使用\auto\万用表RS-12的使用_new.md"
    # 容错：如果找不到，就使用当前目录的测试
    with open(sample_document_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    # 构造状态字典
    state = {
        "file_title": "万用表的使用",
        "md_content": content,
        # 指向你想输出 JSON 的备份目录
        "file_dir": os.path.dirname(sample_document_path)
    }

    # 执行切分节点
    result_state = node_document_split.process(state)

    print("\n" + "=" * 50)
    print("切片执行完毕，最终状态字典概览：")
    print("=" * 50)

    # 仅打印前 2 个 chunk 作为预览，防止控制台刷屏
    preview_chunks = result_state.get("chunks", [])[:10]
    print(json.dumps(preview_chunks, ensure_ascii=False, indent=4))
    print(f"\n...... (共生成 {len(result_state.get('chunks', []))} 个 Chunks, 详情请查看 chunks.json)")
