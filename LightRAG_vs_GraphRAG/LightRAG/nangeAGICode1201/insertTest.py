import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path
import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm import openai_embedding
from lightrag.llm import openai_complete_if_cache




# gpt大模型相关配置根据自己的实际情况进行调整
OPENAI_API_BASE = "https://api.wlai.vip/v1"
OPENAI_CHAT_API_KEY = "sk-gdXw028PJ6JtobnBLeQiArQLnmqahdXUQSjIbyFgAhJdHb1Q"
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


# 构建索引  支持TXT, DOCX, PPTX, CSV, PDF等文件格式

# 1、按批次首次构建索引 1-5回
contents = []
current_dir = Path(__file__).parent
# 指定文件目录
files_dir = current_dir / "files/inputs"
for file_path in files_dir.glob("*.txt"):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    contents.append(content)
rag.insert(contents)
print("构建索引完成")


# # 2、按批次追加构建索引 6-7回
# contents = []
# current_dir = Path(__file__).parent
# # 指定文件目录
# files_dir = current_dir / "files/incremental_inputs"
# for file_path in files_dir.glob("*.txt"):
#     with open(file_path, "r", encoding="utf-8") as file:
#         content = file.read()
#     contents.append(content)
#
# rag.insert(contents)
# print("增量构建索引完成")


# # 3、按批次追加构建索引 8-9回
# import textract
# contents = []
# current_dir = Path(__file__).parent
# # 指定文件目录
# files_dir = current_dir / "files/incremental_inputs"
# for file_path in files_dir.glob("*.pdf"):
#     text_content = textract.process(str(file_path))
#     contents.append(text_content.decode('utf-8'))
#
# rag.insert(contents)
# print("增量构建索引完成")


