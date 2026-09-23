"""
Markdown 图片处理节点
处理 MD 文档中的图片：总结、上传 MinIO、替换链接
"""
import base64
import json
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Tuple, List, Deque, Dict
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import ImageProcessingError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.tools.minio_utils import get_minio_client


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
        config = get_config()
        # 获取Markdown内容和相关路径
        md_content, md_path_obj, images_dir_obj = self._get_md_content_and_path(state)
        state["md_content"] = md_content
        if not images_dir_obj:
            self.logger.info(f"没有找到images目录，跳过图片处理流程")
            return state

        # 扫描并且筛选需要处理的图片 防止图片目录混进不是图片的格式 并且提取图片的上下文
        target_images_info = self._scan_and_filter_images(md_content, images_dir_obj, config.image_extensions)
        if not target_images_info:
            self.logger.info("没有找到需要处理的有效引用图片")
            return state

        minio_client = get_minio_client()

        # 生成图片总结
        image_summaries = self._generate_image_summaries(md_path_obj.stem, target_images_info,
                                                         config.requests_per_minute, config)

        # 上传图片到minio并且替换掉markdown中的链接
        new_md_content = self._upload_images_and_replace_links(minio_client, md_path_obj.stem
                                                               , target_images_info
                                                               , image_summaries
                                                               , md_content
                                                               , config)
        state['md_content'] = new_md_content

        # 备份生成新的markdown文件
        new_md_file_path = self._backup_new_md_file(state["md_path"], new_md_content)
        state['md_path'] = new_md_file_path
        return state

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

    def _scan_and_filter_images(self, md_content: str, images_dir_obj: Path, allowed_extensions: set[str]) -> List[
        Tuple[str, str, Tuple[str, str, str]]]:
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
        target_images = []
        # 便利图片目录下的所有文件并且查看是否在允许的扩展名列表中
        for image_filename in os.listdir(images_dir_obj):
            file_ext = os.path.splitext(image_filename)[1]
            if file_ext not in allowed_extensions:
                continue
            image_full_path = str(images_dir_obj / image_filename)
            # 在 Markdown 内容中查找该图片的引用上下文
            contexts_list = self._find_image_contexts_in_md(md_content, image_filename)
            if not contexts_list:
                self.logger.debug(f"图片 {image_filename} 未在文档中被引用，跳过。")
                continue
            # 取第一个引用处的上下文用于生成摘要
            # 注意：同一张图片如果在文档多处复用，仅使用【第一次出现位置】的上下文送入VLM生成图片摘要
            primary_context = contexts_list[0]
            target_images.append((image_filename, image_full_path, primary_context))

        self.logger.info(f"找到{len(target_images)}张需要处理的有效图片")
        return target_images

    def _find_image_contexts_in_md(self, md_content, image_filename, max_chars: int = 100) -> List[
        Tuple[str, str, str]]:
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
        # 把md内容按照行进行切割
        lines = md_content.split("\n")
        # 构建正则匹配图片引用行
        image_pattern = re.compile(
            r"!\[.*?\]\(.*?" + re.escape(image_filename) + r".*?\)"
        )
        contexts_list = []
        for line_idx, line in enumerate(lines):
            if not image_pattern.search(line):
                continue

            # 向上查找最近的标题
            section_heading = ""  # 保存找到的标题
            heading_line_idx = -1  # 保存标题的索引
            for i in range(line_idx - 1, -1, -1):
                if re.match(r"^#{1,6}\s+", lines[i]):
                    section_heading = lines[i].strip()
                    heading_line_idx = i
                    break

            # 提取上下文 heading_line_idx+1 是为了不取标题 如果小于0则表示没有标题
            pre_start = heading_line_idx + 1 if heading_line_idx >= 0 else 0
            pre_lines = lines[pre_start:line_idx]
            pre_paragraphs = self._extract_paragraphs_with_limit(pre_lines, max_chars, direction='backward')

            # 提取图片下文
            next_heading_idx = -1
            for i in range(line_idx + 1, len(lines)):
                if re.match(r"^#{1,6}\s+", lines[i]):
                    next_heading_idx = i
                    break
            post_lines = lines[line_idx + 1:next_heading_idx]
            post_paragraphs = self._extract_paragraphs_with_limit(post_lines, max_chars, direction='forward')

            contexts_list.append((section_heading, pre_paragraphs, post_paragraphs))
        return contexts_list

    def _extract_paragraphs_with_limit(self, lines: List[str], max_chars: int, direction: str) -> str:
        """
        从给定的行列表中提取完整段落，总字符数不超过 max_chars。
        Args:
            lines (List[str]): 待提取的行列表。
            max_chars (int): 最大字符数限制。
            direction (str): "forward" 从前往后，"backward" 从后往前。
        Returns:
            str: 拼接后的段落文本。
        """
        # 分割段落 遇到空行“” 或者遇到图片为一个段落
        paragraphs = []
        current_para = []
        for line in lines:
            stripped = line.strip()
            if stripped == "":
                if current_para:
                    paragraphs.append("\n".join(current_para))
                    current_para = []
            else:
                # 跳过图片行
                if re.match(r"^!\[.*?\]\(.*?\)$", stripped):
                    if current_para:
                        paragraphs.append("\n".join(current_para))
                        current_para = []
                    continue  # 跳过图片
                current_para.append(stripped)
        if current_para:
            paragraphs.append("\n".join(current_para))

        # 去除空行
        paragraphs = [p for p in paragraphs if p.strip()]
        if not paragraphs:
            return ""

        selected = []
        total_chars = 0

        # backward优先取靠图片的段落
        if direction == "backward":
            paragraphs = list(reversed(paragraphs))

        for para in paragraphs:
            parp_len = len(para)
            if total_chars + parp_len > max_chars and selected:
                break
            selected.append(para)
            total_chars += parp_len

        if direction == "backward":
            selected = list(reversed(selected))
        return "\n\n".join(selected)

    def _generate_image_summaries(self, document_stem: str,
                                  target_images_info: List[Tuple[str, str, Tuple[str, str, str]]],
                                  requests_per_minute: int, config):
        """
        调用多模态模型为图片生成内容摘要。
        Args:
            document_stem (str): 文档文件名（不含扩展名）。
            target_images_info (List): 待处理图片信息列表。
            requests_per_minute (int): API 请求速率限制。
            config: 全局配置对象。
        Returns:
            Dict[str, str]: 图片文件名到摘要的映射。
        """
        self.log_step("step3", "生成图片总结")
        image_summaries = {}
        request_timestamps: Deque[float] = deque()
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=config.openai_api_key,
                base_url=config.openai_api_base,
            )
        except ImportError:
            self.logger.error("未安装openai库，无法初始化客户端。")
            return image_summaries
        except Exception as e:
            self.logger.error(f"初始化VL客户端失败：{e}")
            return image_summaries
        for image_filename, image_full_path, context_tuple in target_images_info:
            # 限制速率
            self._enforce_rate_limit(request_timestamps, requests_per_minute)
            self.logger.debug(f"正在生成摘要:{image_filename}")
            summary_text = self._call_vlm_for_summary(client
                                                      , config.vl_model
                                                      , image_full_path
                                                      , document_stem
                                                      , context_tuple)
            image_summaries[image_filename] = summary_text
        return image_summaries

    def _enforce_rate_limit(self, request_timestamps: Deque[float], max_requests: int, window_seconds: int = 60):
        """
        强制执行 API 请求速率限制。
        Args:
            request_timestamps (Deque[float]): 请求时间戳队列。
            max_requests (int): 窗口内最大请求数。
            window_seconds (int, optional): 时间窗口大小（秒）。
        """
        current_time = time.time()

        # 移除窗口外的时间戳
        while request_timestamps and \
                current_time - request_timestamps[0] > window_seconds:
            request_timestamps.popleft()

        # 达到上限则等待
        if len(request_timestamps) >= max_requests:
            sleep_duration = window_seconds - (current_time - request_timestamps[0])
            if sleep_duration > 0:
                self.logger.info(f"达到速率限制，暂停{sleep_duration:.2f}秒...")
                time.sleep(sleep_duration)
            current_time = time.time()  # 时间等待后把已经过期的时间戳删除
            while request_timestamps and current_time - request_timestamps[0] > window_seconds:
                request_timestamps.popleft()
        return request_timestamps

    def _call_vlm_for_summary(self, client, model_name: str, image_path: str, doc_title: str,
                              context_tuple: Tuple[str, str, str]) -> str:
        """
        调用 VLM 接口生成图片描述。
        Args:
        client: OpenAI 客户端实例。
        model_name (str): 模型名称。
        image_path (str): 图片文件路径。
        doc_title (str): 文档标题。
        context_tuple (Tuple[str, str, str]): 上下文三元组。
        Returns:
        str: 生成的图片摘要。
        """
        # 读取并 Base64 编码
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        except IOError as e:
            self.logger.error(f"无法读取图片文件{image_path}:{e}")
            return "读取图片失败"
        section_heading, pre_text, post_text = context_tuple

        # 构建prompt
        context_parts = []
        if section_heading:
            context_parts.append(f"所属章节标题:{section_heading}")
        if pre_text:
            context_parts.append(f"图片上文为:{pre_text}")
        if post_text:
            context_parts.append(f"图片下文为:{post_text}")
        context_info = "\n".join(context_parts) if context_parts else "无上下文可用"

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""任务:为Markdown文档中的图片生成一个简短的中文标题。
背景信息:
1.所属文档标题:"{doc_title}"
2.图片上下文{context_info}
请结合图片视觉内容和上诉上下文信息，用中文简要总结这种图片的内容，
生成一个精准的中文标题（不要包含"图片"二字）。"""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=100,
                temperature=0.3
            )
            summary = response.choices[0].message.content.strip().replace("\n", " ")
            return summary
        except Exception as e:
            self.logger.warning(f"图片摘要生成失败{image_path}:{e}")
            return "图片描述"

    def _upload_images_and_replace_links(self, minio_client, document_stem: str,
                                         target_images_info: List[Tuple[str, str, Tuple[str, str, str]]],
                                         image_summaries: Dict[str, str]
                                         , md_content: str,
                                         config):
        """
        将图片上传至 MinIO 并替换 Markdown 中的本地链接。
        Args:
            minio_client: MinIO 客户端实例。
            document_stem (str): 文档文件名。
            target_images_info (List): 图片信息列表。
            image_summaries (Dict): 图片摘要字典。
            md_content (str): 原始 Markdown 内容。
            config: 配置对象。
        Returns:
            str: 替换链接后的 Markdown 内容。
        """
        self.log_step(f"step4", "上传图片并且替换链接")
        uploaded_urls = {}
        # 遍历上传图片
        for image_filename, image_full_path, _ in target_images_info:
            object_name = f"{document_stem}/{image_filename}"
            ext = os.path.splitext(image_filename)[1].lower()
            # 如果 `ext` 是以`.`开头（正常带后缀的文件，`.png`/`.jpg`）那么就把`.`去掉（`ext[1:]`），拼接成 `image/png`
            content_type = f"image/{ext[1:]}" if ext.startswith(".") else "application/octet-stream"
            if minio_client:
                try:
                    minio_client.fput_object(
                        config.minio_bucket,
                        object_name,
                        image_full_path,
                        content_type=content_type
                    )
                    # 拼接图片在 MinIO 上的可访问公网地址，把「本地图片文件名」→「远端图片 URL」存入字典
                    remote_url = f"{config.get_minio_base_url()}/{object_name}"
                    uploaded_urls[image_filename] = remote_url
                    self.logger.info(f"图片上传成功：{image_filename}->{remote_url}")
                except Exception as e:
                    self.logger.warning(f"图片上传失败{image_filename}:{e}")
            else:
                self.logger.warning(f"MinIO客户端未初始化，跳过实际上传")
                uploaded_urls[image_filename] = \
                    f"http://mock-minio/{document_stem}/{image_filename}"
        # 替换md中的图片链接
        new_md_content = md_content
        for image_filename, summary_text in image_summaries.items():
            remote_url = uploaded_urls.get(image_filename)
            if not remote_url:
                continue
            replace_pattern = re.compile(
                r"!\[(.*?)\]\((.*?" + re.escape(image_filename) + r".*?)\)",
                re.IGNORECASE
            )
            new_md_content = replace_pattern.sub(
                f"![{summary_text}]({remote_url})",
                new_md_content
            )
        self.logger.info(f"成功替换了{len(uploaded_urls)}张图片的链接")
        return new_md_content

    def _backup_new_md_file(self, original_md_path_str: str, new_md_content: str) -> str:
        """
        将处理后的 Markdown 内容写入新文件。
        Args:
            original_md_path_str (str): 原始文件路径。
            new_md_content (str): 新的 Markdown 内容。
        Returns:
            str: 新文件的绝对路径。
        """
        self.log_step(f"step_5", "备份新的文件")
        original_path = Path(original_md_path_str)
        new_file_path = original_path.with_name(
            f"{original_path.stem}_new{original_path.suffix}"
        )
        try:
            with open(new_file_path, "w", encoding='utf-8') as f:
                f.write(new_md_content)
            self.logger.info(f"处理后的文件已经备份至：{new_file_path}")
        except Exception as e:
            self.logger.error(f"写入新的文件失败{new_file_path}:{e}")
            raise ImageProcessingError(f"文件写入失败:{e}", node_name=self.name)
        return str(new_file_path)


# ================================================================== #
#                        兼容 & 测试                                   #
# ================================================================== #

# 兼容原有调用方式
node_md_img = MdImgNode()

if __name__ == '__main__':
    # 1. 开启日志
    setup_logging()

    # 2. 构建处理图片节点的状态
    img_state = {
        "md_path": r"E:\rag\docretri_rag\knowledge\processor\import_process\import_temp_dir\output\万用表RS-12的使用\auto\万用表RS-12的使用.md"
    }

    # 3. 处理 md 图片节点
    processed_img_result = node_md_img.process(img_state)

    # 4. 打印结果
    processed_img_result_str = json.dumps(
        processed_img_result,
        indent=4,
        ensure_ascii=False
    )
    print(processed_img_result_str)
