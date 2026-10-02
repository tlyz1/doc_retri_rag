"""
PDF 转 Markdown 节点
使用 MinerU 将 PDF 文档转换为 Markdown 格式
"""
import json
import os

import sys
import time
from pathlib import Path
import subprocess
from knowledge.processor.import_process.base import BaseNode, T, setup_logging
from knowledge.processor.import_process.exceptions import FileProcessingError, PdfConversionError
from knowledge.processor.import_process.state import ImportGraphState


class PdfToMdNode(BaseNode):
    name = 'pdf_to_md_node'

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 验证文件路径
        pdf_path_obj, output_dir_obj = self._validate_paths(state)

        # 执行mineru转换
        return_code = self._execute_mineru(pdf_path_obj, output_dir_obj)
        if return_code != 0:
            raise PdfConversionError(
                "MinerU转换失败，请检查mineru日志",
                node_name=self.name
            )
        # 获取结果路径
        state["md_path"] = self._get_output_path(pdf_path_obj, output_dir_obj)
        self.log_step("step_3", f"输出路径: {state['md_path']}")

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
        self.logger.info("step_2,执行MinerU转换")

        # 使用绝对路径而不是 "mineru" 短名：venv 调起子进程时 Python 标准库
        # 是从 D:\python11\Lib 加载的 subprocess.py，对短名 + 自定义 env 的解析
        # 在某些场景下会 CreateProcess 失败（WinError 2），用绝对路径绕过。
        mineru_exe = os.path.join(os.path.dirname(sys.executable), "mineru.exe")

        # 构建命令
        cmd = [
            mineru_exe,
            "-p", str(pdf_path_obj),
            "-o", str(output_path_obj),
            "-b", "pipeline",  # 没有使用cpu加速
            "-m", "auto",  # auto 模式同时提取文本与图像，输出 layout/tables/formula/images
            "--source", "local"
        ]
        self.logger.info(f"执行命令:{' '.join(cmd)}")
        start_ts = time.time()

        # 关键：把 venv 的 Scripts 目录放到 PATH 最前面，
        # 避免子进程被 D:\python11\Scripts\mineru.exe 抢占。
        # mineru 是个 package，不是 module，不能用 -m mineru 直接调用。
        env = os.environ.copy()
        venv_scripts = os.path.dirname(sys.executable)  # Windows: ...\.venv\Scripts
        env["PATH"] = venv_scripts + os.pathsep + env.get("PATH", "")

        # 限制 OpenBLAS/MKL 线程数与进程池规模，避免物理内存不足时 (RTX 3050 4GB + 16GB 共享内存)
        # OpenBLAS "Memory allocation still failed after 10 retries" 导致 mineru worker BrokenProcessPool
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        # 强制 mineru 用 CPU：单卡 4GB 共享给 BGE-M3(2GB+) + doclayout+Unimernet 易 OOM
        env.setdefault("MINERU_DEVICE_MODE", "cpu")
        # 关闭 safetensors 的 mmap，绕过 Windows ERROR_PARTIAL_COPY (1455) 在低内存下 mmap 失败
        env.setdefault("SAFETENSORS_FAST_MMAP", "0")
        env.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

        # 调用命令行工具
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,  # 捕获标准输出
            stderr=subprocess.STDOUT,  # 合并错误到 stdout
            text=True,  # 文本模式
            encoding="utf-8",
            errors="replace",  # 遇到乱码时替换
            env=env,  # 传递修正后的环境变量
            bufsize=1,  # 行缓冲（实时输出）
        )

        # 实时输出日志
        for line in proc.stdout:
            self.logger.info(f"[mineru] {line.rstrip()}]")

        # 等待命令完成
        return_code = proc.wait()

        elapsed = time.time() - start_ts
        if return_code == 0:
            self.logger.info(f"转换完成，耗时:{elapsed:.2f}秒")
        else:
            self.logger.error("转换失败")
        return return_code

    def _get_output_path(self, pdf_path_obj: Path, output_dir_obj: Path) -> str:
        """
        获取转换结果路径
        MinerU 输出目录结构:
        output_dir/
          └── 文件名/
               └── hybrid_auto/
                    ├── 文件名.md
                    └── images/
        Args:
            pdf_path_obj: PDF 文件 Path 对象
            output_dir_obj: 输出目录 Path 对象
        Returns:
            Markdown 文件路径字符串
        """
        file_stem = pdf_path_obj.stem
        md_path = output_dir_obj / file_stem / "auto" / f"{file_stem}.md"
        return str(md_path)


# ================================================================== #
#                        兼容 & 测试                                  #
# ================================================================== #

# 兼容原有调用方式
node_pdf_to_md = PdfToMdNode()

if __name__ == '__main__':
    """
    PDF 转 Markdown 节点测试

    注意：需要确保 MinerU 已安装且模型已下载
    """
    # 配置日志
    setup_logging()

    print("=" * 60)
    print("PDF to MD 节点测试")
    print("=" * 60)

    # 实例化节点
    pdf_to_md_node = PdfToMdNode()

    # 测试用例 1: 正常转换
    print("\n--- 测试用例 1: 正常 PDF 转换 ---")

    # 请修改为实际存在的 PDF 文件路径
    test_pdf_path = r"E:\rag\docretri_rag\knowledge\processor\import_process\import_temp_dir\万用表RS-12的使用.pdf"
    test_output_dir = r"E:\rag\docretri_rag\knowledge\processor\import_process\import_temp_dir\output"

    # 检查测试文件是否存在
    if not Path(test_pdf_path).exists():
        print(f"警告: 测试文件不存在: {test_pdf_path}")
        print("请修改 test_pdf_path 为有效的 PDF 文件路径")
    else:
        state = {
            "pdf_path": test_pdf_path,
            "file_dir": test_output_dir
        }

        try:
            result = pdf_to_md_node.process(state)
            print("转换成功!")
            print(json.dumps(result, indent=4, ensure_ascii=False))

            # 检查输出文件是否存在
            md_path = Path(result["md_path"])
            if md_path.exists():
                print(f"\n输出文件已生成: {md_path}")
                print(f"文件大小: {md_path.stat().st_size} 字节")
            else:
                print(f"警告: 输出文件不存在: {md_path}")

        except PdfConversionError as e:
            print(f"转换失败: {e}")
        except FileProcessingError as e:
            print(f"文件处理错误: {e}")

    # 测试用例 2: PDF 路径为空
    print("\n--- 测试用例 2: PDF 路径为空 ---")
    try:
        state_empty = {"pdf_path": ""}
        pdf_to_md_node.process(state_empty)
    except FileProcessingError as e:
        print(f"捕获到预期异常: {e}")

    # 测试用例 3: PDF 文件不存在
    print("\n--- 测试用例 3: PDF 文件不存在 ---")
    try:
        state_not_exist = {"pdf_path": "D:/not_exist/file.pdf"}
        pdf_to_md_node.process(state_not_exist)
    except FileProcessingError as e:
        print(f"捕获到预期异常: {e}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
