from openai import OpenAI

# 设置API基础地址和密钥，与配置代码中一致
client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="vllm-Qwen3-8B-Instruct-123456"
)

try:
    response = client.chat.completions.create(
        model="/data0/bma/models/Qwen3-8B",  # 模型地址
        messages=[{"role": "user", "content": "请介绍一下苹果公司"}]
    )
    print(response)
    # 检查响应中是否包含期望的字段，比如有生文本内容
    if response.choices and len(response.choices) > 0:
        print("请求成功，模型返回内容：", response.choices[0].message.content)
    else:
        print("请求成功，但响应内容格式异常")
except Exception as e:
    print(f"请求失败，错误信息：{e}")