import logging
from langchain_openai import ChatOpenAI
from knowledge.processor.import_process.config import get_config

# LLM客户端缓存字典，key为(模型名称,是否开启json输出)，value为ChatOpenAI实例
# 用于缓存不同参数的LLM实例，避免重复创建，复用http连接
cache_llm_client = {}


def get_llm_client(model_name: str = None, temperature: float = 0.0, response_format: bool = False):
    """
    LLM客户端工厂函数，支持缓存不同参数的ChatOpenAI实例
    场景：LangGraph不同节点可使用不同模型、不同响应格式；相同参数复用已有客户端

    :param mode_name: 模型名称，例如 Qwen/Qwen3-32B
    :param temperature: 模型温度，越小输出越稳定，NL2SQL场景推荐0.0~0.1
    :param response_format: 是否开启json_object强制JSON输出。Qwen3-32B在siliconflow上对此参数返回400 Bad Request；当前默认关闭，由 prompt 强制 JSON。
    :return: ChatOpenAI实例，创建失败返回None
    """
    # 读取项目配置文件，获取api_key、base_url等信息
    config = get_config()
    # 构造缓存key，模型名称+是否json输出，区分不同客户端实例
    cache_key = (model_name, response_format)

    # 如果缓存中已存在该实例，直接返回，不再重复创建
    if cache_key in cache_llm_client:
        return cache_llm_client[cache_key]

    # 初始化模型参数字典
    model_kwargs = {}

    # 兼容旧调用：若上游网关实际不支持 json_object，就关闭（硅流/Qwen3 实际拒绝此参数）
    if response_format:
        try:
            import os as _os
            if _os.getenv('LLM_ALLOW_RESPONSE_FORMAT', '0') == '1':
                model_kwargs['response_format'] = {"type": "json_object"}
        except Exception:
            pass

    try:
        # 实例化ChatOpenAI，兼容OpenAI协议的大模型服务
        client = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=config.openai_api_key,
            base_url=config.openai_api_base,
            # 扩展参数：关闭Qwen模型的思考链输出，根据上游API网关支持情况决定是否保留
            extra_body={"enable_thinking": False},
            model_kwargs=model_kwargs
        )
        # 将新建的客户端存入全局缓存
        cache_key = (model_name, response_format, bool(model_kwargs))
        cache_llm_client[cache_key] = client
        return client
    except Exception as e:
        # 捕获客户端实例化异常，记录错误日志
        logging.error(f"LLM客户端创建失败:{str(e)}")
        return None


if __name__ == "__main__":
    # 节点1：元数据筛选，普通文本输出
    llm_filter = get_llm_client("Qwen/Qwen3-32B", temperature=0.1, response_format=False)
    rs1 = llm_filter.invoke("你是豆包吗")
    print(rs1.content)
    # 节点2：生成SQL，强制输出JSON
    llm_sql_gen = get_llm_client("Qwen/Qwen3-32B", temperature=0.0, response_format=True)
    rs2 = llm_sql_gen.invoke("你是豆包吗")
    print(rs2.content)
