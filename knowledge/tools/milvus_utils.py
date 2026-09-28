import os
from dotenv import load_dotenv

from knowledge.tools.embedding_utils import logger

load_dotenv()
from typing import Optional
from pymilvus import MilvusClient

milvus_client: Optional[MilvusClient] = None


def get_milvus_client() -> Optional[MilvusClient]:
    global milvus_client

    if milvus_client is not None:
        return milvus_client

    try:
        milvus_url = os.getenv("MILVUS_URL", 'http://192.168.88.161:19530')
        milvus_client = MilvusClient(
            uri=milvus_url)
        return milvus_client
    except Exception as e:
        logger.error(f"MilVus客户端创建失败：{str(e)}")
        return None
