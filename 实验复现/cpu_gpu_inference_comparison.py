import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# 禁用 Tokenizers 的并行性，以避免相关警告
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def measure_inference(model, tokenizer, text_inputs, device):
    # 确保text_inputs是列表，即使只有一个文本，也包装成列表
    if not isinstance(text_inputs, list):
        text_inputs = [text_inputs]

    inputs = tokenizer(text_inputs, return_tensors="pt", padding=True, truncation=True)
    # 检查input_ids是否为空
    if inputs["input_ids"].shape[1] == 0:
        print("警告: 分词后input_ids为空，跳过推理。")
        return [], 0.0, 0.0
    start_transfer = time.time()
    inputs = inputs.to(device)
    end_transfer = time.time()
    transfer_time = end_transfer - start_transfer

    start_infer = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    end_infer = time.time()
    infer_time = end_infer - start_infer

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded, transfer_time, infer_time


# 加载模型和分词器
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)


# 加载数据集
dataset = load_dataset("squad_v2", split="train")
# 取部分文本作为输入
sample_texts = [dataset[i]["context"] for i in range(min(100, len(dataset)))]


# CPU推理
model.to("cpu")
cpu_output, cpu_transfer_time, cpu_infer_time = measure_inference(model, tokenizer, sample_texts, "cpu")

# GPU推理（如果有GPU）
if torch.cuda.is_available():
    model.to("cuda")
    gpu_output, gpu_transfer_time, gpu_infer_time = measure_inference(model, tokenizer, sample_texts, "cuda")
else:
    gpu_output, gpu_transfer_time, gpu_infer_time = None, None, None

print(f"CPU推理时间: {cpu_infer_time:.4f}秒, 数据传输时间: {cpu_transfer_time:.4f}秒")
if gpu_output is not None:
    print(f"GPU推理时间: {gpu_infer_time:.4f}秒, 数据传输时间: {gpu_transfer_time:.4f}秒")
else:
    print("未检测到GPU，跳过GPU推理")
