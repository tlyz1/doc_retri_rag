import os
from minio import Minio

MINIO_ENDPOINT = os.getenv('MINIO_EMDPOINT', "192.168.88.161:9000")
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', "minioadmin")
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', "minioadmin")
MINIO_BUCKET_NAME = os.getenv('MINIO_BUCKET_NAME', "knowledge-base")

try:
    minio_client = Minio(MINIO_ENDPOINT,
                         access_key=MINIO_ACCESS_KEY,
                         secret_key=MINIO_SECRET_KEY,
                         secure=False)
    if not minio_client.bucket_exists(MINIO_BUCKET_NAME):
        minio_client.make_bucket(MINIO_BUCKET_NAME)
except Exception as e:
    print(f"MinIO initialization failed : {e}")
    minio_client = None


def get_minio_client():
    return minio_client
