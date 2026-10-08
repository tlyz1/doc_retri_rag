"""
上传相关 Schema 定义
"""
from pydantic import BaseModel, Field

class UploadResponse(BaseModel):
    """文件上传响应"""
    message: str = Field(..., description="响应消息")
    task_id: str = Field(..., description="任务ID")  # 改为单个字符串
    doc_id: str = Field("", description="文档身份：上传文件内容的 sha256 前 32 位")
    overwrite: bool = Field(False, description="本次是否为覆盖导入（对应 overwrite=true）")
