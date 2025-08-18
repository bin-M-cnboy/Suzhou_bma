import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm import openai_embedding
from lightrag.llm import openai_complete_if_cache



# 模型全局参数配置  根据自己的实际情况进行调整
OPENAI_API_BASE = "https://api.wlai.vip/v1"
OPENAI_CHAT_API_KEY = "sk-dUWW1jzueJ4lrDixWaPsq7nnyN5bCucMzvldpNJwfJlIvAcC"
OPENAI_CHAT_MODEL = "gpt-4o-mini"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"


# 检测并创建文件夹
WORKING_DIR = "./output"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


# 自定义Chat模型 配置类OpenAI
async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await openai_complete_if_cache(
        model=OPENAI_CHAT_MODEL,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=OPENAI_CHAT_API_KEY,
        base_url=OPENAI_API_BASE,
        **kwargs
    )


# 自定义Embedding模型 配置类OpenAI
async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embedding(
        texts,
        model=OPENAI_EMBEDDING_MODEL,
        api_key=OPENAI_CHAT_API_KEY,
        base_url=OPENAI_API_BASE,
    )


# 定义rag
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=1536,
        max_token_size=8192,
        func=embedding_func
    )
)


# 构建索引
with open("./input/book.txt", "r", encoding="utf-8") as f:
    rag.insert(f.read())


# # local检索
# print(
#     rag.query("这个故事的核心主题是什么？", param=QueryParam(mode="local"))
# )

# # global检索
# print(
#     rag.query("这个故事的核心主题是什么？", param=QueryParam(mode="global"))
# )

# # hybrid检索
# print(
#     rag.query("这个故事的核心主题是什么？", param=QueryParam(mode="hybrid"))
# )

# # naive检索
# print(
#     rag.query("这个故事的核心主题是什么？", param=QueryParam(mode="naive"))
# )
