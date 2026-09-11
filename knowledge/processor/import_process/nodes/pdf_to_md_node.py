"""
PDF 转 Markdown 节点
使用 MinerU 将 PDF 文档转换为 Markdown 格式
"""
import os
import time
from pathlib import Path
import subprocess
from knowledge.processor.import_process.base import BaseNode, T
from knowledge.processor.import_process.exceptions import ValidationError, FileProcessingError, PdfConversionError
from knowledge.processor.import_process.state import ImportGraphState


class PdfToMdNode(BaseNode):
    name = 'pdf_to_md_node'

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 验证文件路径
        pdf_path_obj, output_path_obj = self._validate_paths(state)

        #执行mineru转换
        return_code=self._execute_mineru(pdf_path_obj, output_path_obj)
        if return_code != 0:
            raise PdfConversionError(
                "MinerU转换失败，请检查mineru日志",
                node_name=self.name
            )
        return state

    def _validate_paths(self, state: ImportGraphState) -> tuple:
        """
        验证 PDF 路径和输出目录
        Args:
            state: 图状态
        Returns:
            (pdf_path_obj, output_dir_obj) 元组
        Raises:
            FileProcessingError: 路径无效时抛出
        """
        self.log_step("step_1", "验证路径")
        # 双重判断
        pdf_path = state.get("pdf_path", "")
        if not pdf_path:  # 判断是否是空字符串
            raise FileProcessingError(f"pdf_path为空", node_name=self.name)
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.is_file():  # 判断文件是否存在磁盘
            raise FileProcessingError(f"PDF文件不存在:{pdf_path_obj}", node_name=self.name)

        # 获取输出路径
        output_dir = state.get("file_dir", "")
        if not output_dir:
            output_dir = str(pdf_path_obj.parent)
        output_dir_obj = Path(output_dir)
        self.logger.info(f"处理PDF:{pdf_path_obj}")

        return pdf_path_obj, output_dir_obj

    def _execute_mineru(self, pdf_path_obj: Path, output_path_obj: Path) -> int:
        """
        执行 MinerU 命令
        Args:
            pdf_path_obj: PDF 文件 Path 对象
            output_dir_obj: 输出目录 Path 对象
        Returns:
            命令返回码（0 表示成功）
        """
        self.logger.info("step_2", "执行MinerU转换")

        #构建命令
        cmd=["mineru",
             "-p",str(pdf_path_obj),
             "-o",str(output_path_obj),
             "--source","local"]
        self.logger.info(f"执行命令:{' '.join(cmd)}")
        start_ts=time.time()

        # 调用命令行工具
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,  # 捕获标准输出
            stderr=subprocess.STDOUT,  # 合并错误到 stdout
            text=True,  # 文本模式
            encoding="utf-8",
            errors="replace",  # 遇到乱码时替换
            env=os.environ.copy(),  # 传递环境变量
            bufsize=1,  # 行缓冲（实时输出）
        )

        #实时输出日志
        for line in proc.stdout:
            self.logger.debug(f"[mineru] {line.rstrip()}]")

        #等待命令完成
        return_code = proc.wait()

        elapsed=time.time() - start_ts
        if return_code == 0:
            self.logger.info(f"转换完成，耗时:{elapsed:.2f}秒")
        else:
            self.logger.error("转换失败")
        return return_code



if __name__ == '__main__':
    pd = PdfToMdNode()
    state = {
        "pdf_path": r"E:\rag\docretri_rag\knowledge\processor\import_process\import_temp_dir\万用表RS-12的使用.pdf"
    }
    pd.process(state)
