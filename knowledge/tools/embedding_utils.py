import logging
import gc
import os
import threading
from typing import Optional, List

# 优先使用外部环境变量，否则用默认值（国内镜像 + 自定义缓存路径，避免污染 ~/.cache）
os.environ.setdefault('HF_HUB_CACHE', r'E:\Milvues_models\huggingface_bgem3')

# 系统代理(127.0.0.1:7897 等)若挡住 huggingface.co，HEAD 验证会失败导致模型无法加载。
# 模型已经在本地缓存，强制离线模式跳过远端校验。
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

from dotenv import load_dotenv

load_dotenv()
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

# 加载.env环境变量


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 全局变量，保存BGE-M3嵌入模型实例，懒加载初始为None
bge_m3_ef: Optional[BGEM3EmbeddingFunction] = None
# 保护懒加载的多线程并发，详见 get_bge_m3_embedding_model 中的 double-check
_bge_init_lock = threading.Lock()


def get_bge_m3_embedding_model() -> Optional[BGEM3EmbeddingFunction]:
    """
    BGE-M3嵌入模型懒加载单例工厂函数
    一次加载全局复用，输出稠密向量+稀疏向量，用于Milvus混合检索
    :return: BGEM3EmbeddingFunction实例，加载失败返回None
    """
    global bge_m3_ef
    # 快速路径：已加载则直接返回
    if bge_m3_ef is not None:
        return bge_m3_ef
    # 双重检查锁，避免多个线程同时进入加载分支创建两份实例
    with _bge_init_lock:
        if bge_m3_ef is not None:
            return bge_m3_ef
        logger.info("开始加载BGE-M3嵌入模型...")
        model_name = os.getenv('BGE_M3_PATH', 'BAAI/bge-m3')
        # 优先 GPU：4GB 显存（RTX 3050）必须 fp16，否则 OOM
        try:
            import torch as _torch
            device = os.getenv('BGE_DEVICE', 'cuda' if _torch.cuda.is_available() else 'cpu')
        except ImportError:
            device = os.getenv('BGE_DEVICE', 'cpu')
        use_fp16_str = os.getenv('BGE_FP16', 'True' if device == 'cuda' else 'False')
        use_fp16 = use_fp16_str.lower() in ('true', '1', 'yes')

        # 加载前主动回收内存，避免与 PyCharm/Chrome 等抢内存导致 SegFault
        gc.collect()

        try:
            bge_m3_ef = BGEM3EmbeddingFunction(
                model_name=model_name,
                device=device,
                use_fp16=use_fp16
            )
            logger.info(f"BGE-M3模型加载成功，model={model_name}, device={device}, use_fp16={use_fp16}")
        except Exception as e:
            logger.error(f"BGE-M3模型加载失败: {str(e)}", exc_info=True)
            bge_m3_ef = None
        # 模型已加载，直接返回全局单例
        return bge_m3_ef
def generate_hybrid_embeddings(embedding_model: BGEM3EmbeddingFunction, embedding_documents: List[str]):
    """
    为文本生成向量嵌入
    :param embedding_model: 嵌入模型(这里使用BGEM3)
    :param embedding_documents: 要生成嵌入的文本列表
    :return: 包含dense和sparse向量的字典
    """
    try:
        # 1. 生成嵌入
        embedding_result = embedding_model.encode_documents(embedding_documents)

        processed_sparse_result = []
        # 2. 遍历每一个文档
        for index in range(len(embedding_documents)):
            # 2.1 解构csr矩阵&获取稀疏向量
            csr_array = embedding_result['sparse']
            # a) 行索引
            ind_ptr = csr_array.indptr

            # b) 获取行索引的起始值
            start_ind_ptr = ind_ptr[index]
            end_ind_ptr = ind_ptr[index + 1]

            # c) 获取token_id
            token_id = csr_array.indices[start_ind_ptr:end_ind_ptr].tolist()

            # d) 获取权重
            weight = csr_array.data[start_ind_ptr:end_ind_ptr].tolist()

            # 2.2 获取稀疏向量
            sparse_vector = dict(zip(token_id, weight))

            processed_sparse_result.append(sparse_vector)

        # 3. 返回
        return {
            "dense": [den.tolist() for den in embedding_result["dense"]],
            "sparse": processed_sparse_result
        }
    except Exception as e:
        return None

if __name__ == '__main__':
    embedding_model = get_bge_m3_embedding_model()
    query = "我喜欢Python语言"
    result = embedding_model.encode_queries([query])
    # 注意：不要直接 print(result)，里面包含 colbert_vecs 大列表，会触发访问冲突
    dense_list = result['dense']
    if isinstance(dense_list, list):
        # 新版本 pymilvus 返回 list[list[float]]
        print("dense type: list, outer len:", len(dense_list),
              "inner len:", len(dense_list[0]) if dense_list else 0)
        dense = dense_list[0]
    else:
        print("dense shape:", dense_list.shape)
        dense = dense_list[0].tolist()

    # 稀疏向量 CSR格式解析
    # CSR三大成员：indptr(行指针)、data(权重值)、indices(tokenId下标)
    start_index = result['sparse'].indptr[0]
    end_index = result['sparse'].indptr[1]
    print("end_index:", result['sparse'].indptr[1])

    weights = result['sparse'].data[start_index:end_index]
    tokenIds = result['sparse'].indices[start_index:end_index]

    print("权重weights：", weights)
    print("tokenIds：", tokenIds)
