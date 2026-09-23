import json
from pathlib import Path

from win32comext.taskscheduler.taskscheduler import TASK_FLAG_RUN_ONLY_IF_DOCKED

from knowledge.processor.import_process.base import BaseNode, T
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState


class EntryNode(BaseNode):
    """
        入口节点
        根据输入文件的扩展名设置相应的处理标志，
        决定后续流程走 PDF 转换分支还是直接处理 MD 分支。
        """
    name = 'entry'

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
       1. 获取输入文件路径
       2. 检测文件类型
       3. 设置相应的处理标志
       4. 提取文件标题
       Args:
           state: 图状态
       Returns:
           更新后的状态
       """

        self.log_step("step_1,获取文件路径")
        file_path = state.get("import_file_path", "")
        if not file_path:
            raise ValidationError(
                "import_file_path 不能为空",
                node_name=self.name
            )

        path = Path(file_path)
        suffix = path.suffix.lower()
        self.log_step("step2", f"检测到文本类型:{suffix}")
        if suffix == ".pdf":
            self.logger.info("启动PDF读取流程")
            state['is_pdf_read_enabled'] = True
            state['pdf_path'] = file_path
        elif suffix == ".md":
            self.logger.info("启动MD读取流程")
            state['is_md_read_enabled'] = True
            state['md_path'] = file_path
        # TODO:其它格式的文档
        else:
            self.logger.warning(f"不支持该文档格式:{suffix}")

        state['file_title'] = path.stem
        self.log_step("step3", f"文档标题:{state['file_title']}")
        return state


# ================================================================== #
#                        兼容 & 测试                                  #
# ================================================================== #

# 兼容原有调用方式
node_entry = EntryNode()

if __name__ == '__main__':
    """
    入口节点测试

    测试不同文件类型的处理逻辑
    """
    from knowledge.processor.import_process.base import setup_logging

    # 配置日志
    setup_logging()

    print("=" * 60)
    print("Entry 节点测试")
    print("=" * 60)

    # 测试用例 1: PDF 文件
    print("\n--- 测试用例 1: PDF 文件 ---")
    entry_node = EntryNode()
    state_pdf = {
        "import_file_path": r"E:\rag\docretri_rag\knowledge\processor\import_process\import_temp_dir\万用表RS-12的使用.pdf"
    }
    result_pdf = entry_node.process(state_pdf)
    print(json.dumps(result_pdf, indent=4, ensure_ascii=False))

    # # 测试用例 2: MD 文件
    # print("\n--- 测试用例 2: MD 文件 ---")
    # state_md = {
    #     "import_file_path": "D:/docs/使用指南.md"
    # }
    # result_md = entry_node.process(state_md)
    # print(json.dumps(result_md, indent=4, ensure_ascii=False))
    #
    # # 测试用例 3: 空路径（预期抛出异常）
    # print("\n--- 测试用例 3: 空路径 ---")
    # try:
    #     state_empty = {"import_file_path": ""}
    #     entry_node.process(state_empty)
    # except ValidationError as e:
    #     print(f"捕获到预期异常: {e}")

    # # 测试用例 4: 不支持的文件类型
    # print("\n--- 测试用例 4: 不支持的文件类型 ---")
    # state_other = {
    #     "import_file_path": "D:/docs/document.docx"
    # }
    # result_other = entry_node.process(state_other)
    # print(json.dumps(result_other, indent=4, ensure_ascii=False))
    # print(f"is_pdf_read_enabled: {result_other.get('is_pdf_read_enabled', False)}")
    # print(f"is_md_read_enabled: {result_other.get('is_md_read_enabled', False)}")
