"""
Markdown 图片处理节点
处理 MD 文档中的图片：总结、上传 MinIO、替换链接
"""
import os
import re
from pathlib import Path
from typing import Tuple

from knowledge.processor.import_process.config import *
from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.exceptions import ImageProcessingError
from knowledge.processor.import_process.state import ImportGraphState


class MdImgNode(BaseNode):
    """
    Markdown 图片处理节点。
    该节点负责处理 Markdown 文档中的本地图片，主要流程包括：
    1. 读取 Markdown 内容，定位图片存储目录。
    2. 扫描并筛选需要处理的本地图片文件。
    3. 调用多模态大模型（VLM）生成图片的文本摘要。
    4. 将图片上传至 MinIO 对象存储，并替换 Markdown 中的本地路径为远程 URL。
    5. 保存替换后的 Markdown 内容到新文件。
    Attributes:
       name (str): 节点名称，标识为 "md_img"。
    """
    name = "md_img"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        执行图片处理流程。
        Args:
            state (ImportGraphState): 当前导入图的状态字典。
        Returns:
            ImportGraphState: 更新后的状态字典。
        """

        # 获取Markdown内容和相关路径
        md_content, md_path_obj, images_dir_obj = self._get_md_content_and_path(state)
        state["md_content"] = md_content
        if not images_dir_obj:
            self.logger.info(f"没有找到images目录，跳过图片处理流程")
            return state

        # 扫描并且筛选需要处理的图片 防止图片目录混进不是图片的格式
        target_images_info = self._scan_and_filter_images(images_dir_obj, config.image_extensions)

    def _get_md_content_and_path(self, state: ImportGraphState) -> Tuple[str, Path, Path]:
        """
        读取 Markdown 文件内容并获取相关路径对象。
        Args:
            state (ImportGraphState): 包含 'md_path' 的状态字典。
        Returns:
            Tuple[str, Path, Path]:
                - md_content: Markdown 文件的文本内容。
                - md_path_obj: Markdown 文件的 Path 对象。
                - images_dir_obj: 关联的 images 目录 Path 对象。
        Raises:
            ImageProcessingError: 当 'md_path' 为空时抛出。
        """
        self.log_step("step_1", "读取MD内容")
        md_file_path_str = state.get("md_path")
        if not md_file_path_str:
            raise ImageProcessingError("状态中md_path为空", node_name=self.name)
        md_path_obj = Path(md_file_path_str)
        if not md_path_obj.is_file():
            raise ImageProcessingError(f"文件{md_path_obj}不存在", node_name=self.name)
        try:
            with open(md_path_obj, "r", encoding='utf-8') as f:
                md_content = f.read()
        except IOError as e:
            raise ImageProcessingError(
                f"无法读取文件{md_path_obj}:{e}", node_name=self.name
            )

        # 获取图片路径
        images_dir_obj = md_path_obj.parent / "images"
        return md_content, md_path_obj, images_dir_obj

    def _scan_and_filter_images(self, md_content: str, images_dir_obj: Path, allowed_extensions: str):
        """
        扫描 images 目录，筛选出在 Markdown 内容中被引用的有效图片。
        Args:
            md_content (str): Markdown 文本内容。
            images_dir_obj (Path): 图片目录路径对象。
            allowed_extensions (set): 允许处理的图片扩展名集合。
        Returns:
            List[Tuple[str, str, Tuple[str, str, str]]]: 图片信息列表。
        """
        self.log_step("step_2", f"扫描图片目录{images_dir_obj}")
        # 便利图片目录下的所有文件并且查看是否在允许的扩展名列表中
        for image_filename in os.listdir(images_dir_obj):
            file_ext = os.path.splitext(image_filename)[1]
            if file_ext not in allowed_extensions:
                continue
            image_full_path = str(images_dir_obj / image_filename)
            contexts_list = self._find_image_contexts_in_md(md_content, image_filename)

    def _find_image_contexts_in_md(self, md_content, image_filename):
        """
        基于 Markdown 语义结构查找图片的上下文。
        策略：
        1. 向上查找最近的标题行（# 开头的行）作为 section 标题。
        2. 取标题到图片之间的完整段落作为上文。
        3. 向下取图片后的 1-2 个完整段落作为下文。
        4. 上文和下文分别不超过 max_chars 字符。
        Args:
           md_content (str): Markdown 文本内容。
           image_filename (str): 要查找的图片文件名。
           max_chars (int, optional): 上下文最大字符数。默认为 100。
        Returns:
           List[Tuple[str, str, str]]: 上下文列表。
        """
        #把md内容按照行进行切割
        lines = md_content.split("\n")
        # 构建正则匹配图片引用行
        image_pattern = re.compile(
            r"!\[.*?\]\(.*?" + re.escape(image_filename) + r".*?\)"
        )
        contexts_list = []
        for line_idx , line in enumerate(lines):
            if not image_pattern.search(line):
                continue


