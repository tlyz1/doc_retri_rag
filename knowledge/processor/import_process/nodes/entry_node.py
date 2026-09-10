from pathlib import Path

from knowledge.processor.import_process.base import BaseNode, T
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState


class EntryNode(BaseNode):
    name='entry'

    def process(self, state: ImportGraphState) -> ImportGraphState:
        self.log_step("step1","[获取文件路径]")
        import_file_path = state.get('import_file_path')
        file_dir = state.get('file_dir')

        self.log_step("step2","[检查文件路径]")
        if not file_dir or not import_file_path:
            raise ValidationError("文件目录或者文件不存在",self.name)

        path=Path(import_file_path)
        suffix=path.suffix.lower() # 获取文件后缀名

        if suffix == ".pdf":
            state["is_pdf_read_enabled"] = True
            state['pdf_path'] = import_file_path
        elif suffix == ".md":
            state["is_md_read_enabled"] = True
            state['md_path'] = import_file_path
        else:
            self.logger.debug(f"文件格式{suffix}不支持")
            raise ValidationError(f"文件格式{suffix}不支持")

        file_title =path.stem
        state["file_title"] = file_title

        return state