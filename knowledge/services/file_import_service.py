from dotenv import load_dotenv

load_dotenv()
import hashlib
import os.path
import re
import shutil
import uuid
from datetime import datetime
from typing import Tuple
from fastapi import UploadFile, HTTPException
from knowledge.core.paths import get_local_base_dir
from knowledge.tools.import_registry_util import (
    get_import_registry_tool,
    RESULT_DUPLICATE,
    RESULT_INCOMPLETE,
)
from knowledge.tools.minio_utils import get_minio_client
from knowledge.services.task_service import TaskService
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.main_graph import kb_import__graph_app


# 中日韩字符（含中文标点与全角字符），用于识别"这串字符本来是不是中文文件名"
_CJK_PATTERN = re.compile(r"[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]")


def restore_upload_filename(filename: str) -> str:
    """
    修复上传文件名的编码错乱。

    现象：`万用表RS-12的使用` 存进来变成 `ÍòÓÃ±íRS-12µÄÊ¹ÓÃ`
          （GBK 字节被按 cp1252/latin-1 解码后的结果）。
    成因：客户端按本地 ANSI 代码页（中文 Windows 上是 GBK）把文件名放进 multipart 头，
          服务端按 cp1252/latin-1 解码，于是每个汉字变成 1~2 个西文乱码字符。
    做法：把字符串按 cp1252/latin-1 编回字节，再按 GBK/GB18030 解码。

    防误伤：只处理"原文里没有任何中日韩字符"、且"还原结果里至少 2 个中日韩字符"的情况，
            避免把本来就正常的西文文件名（如 índice）改坏。
    """
    if not filename or _CJK_PATTERN.search(filename):
        return filename

    for encoding in ("cp1252", "latin-1"):
        try:
            raw = filename.encode(encoding)
        except UnicodeEncodeError:
            continue
        for decoding in ("gbk", "gb18030"):
            try:
                restored = raw.decode(decoding)
            except UnicodeDecodeError:
                continue
            # 还原结果必须真的像中文名，才认账
            if len(_CJK_PATTERN.findall(restored)) >= 2:
                return restored
    return filename


class ImportFileService:
    """

    文件导入的业务类

    1. 保存上传文件到本地
    2. 保存上传文件到MinIO
    3. 运行图谱的所有节点
    """

    def __init__(self, task_service: TaskService):
        self._task_service = task_service

    def get_date_dir(self) -> str:
        return os.path.join(get_local_base_dir(), datetime.now().strftime('%Y%m%d'))

    def save_upload_file_to_local(self, file: UploadFile, file_dir: str):
        """
         将上传的文件保存到本地，并顺带算出内容哈希 doc_id

         边写盘边做 sha256，不把文件再读一遍
        Args:
            file: 上传的文件
            file_dir: 文件的归档目录

        Returns:
            (import_file_path, doc_id)
        """

        # 1. 确保归档目录存在
        os.makedirs(file_dir, exist_ok=True)

        # 2. 构建上传文件的完整的path
        import_file_path = os.path.join(file_dir, file.filename)

        # 3. 分片写入本地，同时累计内容哈希
        hasher = hashlib.sha256()
        with open(import_file_path, 'wb') as f:
            while True:
                piece = file.file.read(1024 * 1024)
                if not piece:
                    break
                hasher.update(piece)
                f.write(piece)

        # 4. 文档身份 = 文件内容哈希前 32 位（与文件名、LLM 提取的商品名都无关）
        doc_id = hasher.hexdigest()[:32]

        # 5. 返回导入文件的path 与文档身份
        return import_file_path, doc_id

    def save_upload_file_to_minio(self, import_file_path: str, file: UploadFile, doc_id: str):
        """

        Args:
            import_file_path:
            file:
            doc_id: 文档身份（内容哈希）。写进归档路径后，"不同内容的同名文件"不会再互相覆盖

        Returns:

        """

        # 1. 获取minio客户端
        minio_client = get_minio_client()

        # 2. 判断minio客户端是否存在
        if not minio_client:
            raise HTTPException(status_code=500, detail="MinIO 服务不可用")

        # 3. 构建Minio客户端对象名（归档文件）
        #    MinIO 的对象名就是完整的 key，没有"目录"概念：两个请求算出同一个 key，
        #    后写的就会覆盖先写的。旧 key 只有"日期 + 文件名"，不含任何内容信息，
        #    所以不同内容的同名文件（如各产品的"说明书.pdf"、改过内容的同一份文件）会互相覆盖。
        #    中间插入 doc_id（内容哈希）后，key 里就带上了"内容"这个维度，不再撞车。
        #    注意边界：同一天 + 同一份内容 + 同一个文件名时 key 仍然相同，会重写同一个对象，
        #    但此时字节完全相同，属于幂等重写，不丢数据。
        minio_object_name = f"origin_files/{datetime.now().strftime('%Y%m%d')}/{doc_id}/{file.filename}"

        # 4. 获取桶名
        bucket_name = os.getenv("MINIO_BUCKET_NAME")
        # 5. 开始上传
        try:
            minio_client.fput_object(bucket_name, minio_object_name, import_file_path)
        except Exception as e:
            raise ValueError(f"{file.filename}文件上传失败 原因:{e}")

    def process_upload_file(self, file: UploadFile, overwrite: bool = False) ->Tuple[str,str,str,str]:
        """
        处理上传文件
        Returns:
        1. 标记当前文件上传节点（"upload_file"）为正在运行中
        2. 将上传的文件保存到本地，并算出内容哈希 doc_id
        3. 按 doc_id 查重：首次导入放行；上次中断/失败也放行；
           已经完整导入过则默认拒绝（带 overwrite=true 才覆盖）
        4. 将上传的文件保存到minio
        5. 标记当前文件上传节点（"upload_file"）为运行完毕
        6. 需要返回四部分信息（task_id file_dir import_file_path doc_id）
        """

        # 0. 修复客户端用 GBK 送过来的文件名
        #    （不修的话 本地目录名 / MinIO 归档名 / file_title 全是 "ÍòÓÃ±í..." 这种乱码）
        original_filename = file.filename or ""
        fixed_filename = restore_upload_filename(original_filename)
        if fixed_filename != original_filename:
            print(f"上传文件名编码修复: {original_filename!r} -> {fixed_filename!r}")
            file.filename = fixed_filename

        # 1. 构建时间日期的文件目录出来
        date_dir = self.get_date_dir()

        # 2. 生成一个任务id
        task_id = str(uuid.uuid4())

        # 3. 构建文件的最终归属目录
        file_dir = os.path.join(date_dir, task_id)

        self._task_service.mark_node_running(task_id, "upload_file")
        # 4. 将接收到的文件上传本地（顺带算出 doc_id）
        import_file_path, doc_id = self.save_upload_file_to_local(file, file_dir)

        # 5. 按 doc_id 查重（登记表不可用时直接拒绝上传，宁可挡住也不写脏数据）
        try:
            dup_result = get_import_registry_tool().reserve(
                doc_id=doc_id,
                file_title=os.path.splitext(file.filename or "")[0],
                task_id=task_id,
                force=overwrite,
            )
        except HTTPException:
            raise
        except Exception as e:
            self._task_service.update_task_status(task_id, "failed")
            raise HTTPException(
                status_code=503,
                detail=f"导入登记服务不可用，本次上传已拒绝（未产生任何入库）：{e}",
            )

        if dup_result == RESULT_DUPLICATE and not overwrite:
            # 已经完整导入过并且没有要求覆盖：拒绝，不写 MinIO、不跑图
            self._task_service.update_task_status(task_id, "failed")
            # 本次刚写下去的本地副本已经用不上了，清掉本次新建的这个任务目录（只删这一层）
            shutil.rmtree(file_dir, ignore_errors=True)
            print(f"[{task_id}] 重复上传被拒绝，doc_id={doc_id}，本次本地副本已清理")
            raise HTTPException(
                status_code=409,
                detail=(
                    f"该文档已经导入过（doc_id={doc_id}）。"
                    f"如需按当前内容覆盖，请用 overwrite=true 重新上传。"
                ),
            )

        if dup_result == RESULT_INCOMPLETE:
            print(f"[{task_id}] 检测到上次导入未完成，按覆盖方式重新导入，doc_id={doc_id}")

        # 6. 将本地磁盘的上传文件同步minio中
        self.save_upload_file_to_minio(import_file_path, file, doc_id)
        self._task_service.mark_node_done(task_id, "upload_file")

        # 7. 构建返回值
        return task_id, file_dir, import_file_path, doc_id

    def  run_import_graph(self,task_id:str,file_dir:str,import_file_path:str,doc_id:str):
        """
        运行导入graph的流程（跑节点）

        1. 构建初始状态
        graph.stream()之前调用update_task_status更新任务的状态为processing
        2. 运行（graph.stream()）
        2.1 在运行图的某个节点之前调用mark_node_running
        2.2 在运行图的某个节点结束调用mark_node_done
        graph.stream()执行完所有节点 update_task_status更新任务的状态为completed
        Args:
            task_id:
            file_dir:
            import_file_path:
            doc_id: 文档身份（内容哈希），贯穿整条链路用于幂等写入

        Returns:

        """
        try:
            # 1. 标记任务开始处理
            self._task_service.update_task_status(task_id, "processing")
            # 2. 构建 LangGraph 初始状态
            global_graph_init_status: ImportGraphState = {
                "task_id": task_id,
                "doc_id": doc_id,
                "file_dir": file_dir,
                "import_file_path": import_file_path
            }

            # 3. 流式执行整个导入流水线
            final_state: dict = {}
            for event in kb_import__graph_app.stream(global_graph_init_status):
                for key, value in event.items():
                    if isinstance(value, dict):
                        final_state.update(value)
                    print(f"[{task_id}] Completed Node: {key}")

            # 4. 标记任务完成
            self._task_service.update_task_status(task_id, "completed")

            # 5. 登记为 done：此后重复上传会被 409 拦住
            self._mark_doc_done(doc_id, final_state.get("item_name", ""))

        except Exception as e:
            self._task_service.update_task_status(task_id, "failed")
            self._mark_doc_failed(doc_id, e)
            print(f"[{task_id}] Error: {e}")

    @staticmethod
    def _mark_doc_done(doc_id: str, item_name: str) -> None:
        """登记导入成功；失败也不影响本次入库，只影响下次的查重结论"""
        try:
            get_import_registry_tool().mark_done(doc_id, item_name)
        except Exception as e:
            print(f"[{doc_id}] 导入登记写入失败（数据已入库，下次上传会按未完成处理）: {e}")

    @staticmethod
    def _mark_doc_failed(doc_id: str, error: Exception) -> None:
        """登记导入失败，下次重传按未完成放行"""
        try:
            get_import_registry_tool().mark_failed(doc_id, str(error))
        except Exception as e:
            print(f"[{doc_id}] 导入失败登记写入失败: {e}")
