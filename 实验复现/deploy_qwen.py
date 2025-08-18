import os
import subprocess
import time

class MA:
    def __init__(self, args):
        # 检测是否使用GPT模型（这里使用本地vllm）
        if "gpt" in args.llm:
            # GPT模型配置（保持不变）
            os.environ['base_url'] = ""
            os.environ['api_key'] = ""
        else:
            # 配置vllm服务的URL和API_KEY
            os.environ['base_url'] = "http://localhost:8001/v1"  # vllm默认API地址                  ###
            os.environ['api_key'] = "vllm-Qwen3-8B-Instruct-123456"  # 可自定义，需与启动参数一致      ###
            
            print("Preparing vllm ...")
            
            # 构建启动命令，指向本地Qwen模型
            cmd = [
                "/data0/bma/env/agentprune/bin/python", # 指定Python解释器路径                        ###
                "-m", "vllm.entrypoints.openai.api_server",
                "--model", "/data0/bma/models/Qwen3-8B",  # 你的本地模型路径                         ###
                "--dtype", "float16",  # 根据硬件能力选择float16或bfloat16
                "--tensor-parallel-size", "1",  # 单GPU部署
                "--api-key", os.environ['api_key'],  # 使用配置的API_KEY
                "--port", "8001"  # 明确指定端口，确保与base_url一致
            ]

            # 启动vllm服务进程
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )

            # 等待服务启动完成
            while True:
                output = process.stdout.readline()
                if 'Starting vLLM API server' in output:
                    print("vllm服务已启动，可通过以下信息访问：")
                    print(f"URL: {os.environ['base_url']}")
                    print(f"API_KEY: {os.environ['api_key']}")
                    time.sleep(5) # Add a 5-second delay to ensure the service is fully ready
                    break
class Args:
    def __init__(self, llm_type):
        self.llm = llm_type
        
args_vllm = Args(llm_type="vllm")
ma_instance_vllm = MA(args_vllm)



# import os
# import subprocess
# import time
# import threading
# from queue import Queue, Empty

# class MA:
#     def __init__(self, args):
#         # 检测是否使用GPT模型（这里使用本地vllm）
#         if "gpt" in args.llm:
#             # GPT模型配置
#             os.environ['base_url'] = ""
#             os.environ['api_key'] = ""
#         else:
#             # 配置vllm服务的URL和API_KEY
#             os.environ['base_url'] = "http://localhost:8001/v1"
#             os.environ['api_key'] = "vllm-Qwen3-8B-Instruct-123456"
            
#             print("Preparing vllm ...")
            
#             # 构建启动命令
#             cmd = [
#                 "/data0/bma/env/CausalRAG/bin/python",
#                 "-m", "vllm.entrypoints.openai.api_server",
#                 "--model", "/data0/bma/models/Qwen3-8B",
#                 "--dtype", "float16",
#                 "--tensor-parallel-size", "1",
#                 "--api-key", os.environ['api_key'],
#                 "--port", "8001"
#             ]

#             # 创建队列用于收集进程输出
#             output_queue = Queue()
            
#             # 启动vllm服务进程
#             self.process = subprocess.Popen(
#                 cmd,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.STDOUT,  # 将stderr合并到stdout
#                 universal_newlines=True,
#                 bufsize=1
#             )

#             # 启动线程收集输出
#             def enqueue_output(out, queue):
#                 for line in iter(out.readline, ''):
#                     queue.put(line)
#                 out.close()
            
#             threading.Thread(
#                 target=enqueue_output,
#                 args=(self.process.stdout, output_queue),
#                 daemon=True
#             ).start()

#             # 等待服务启动完成，增加超时机制
#             start_time = time.time()
#             timeout = 60  # 60秒超时
#             service_started = False
            
#             while time.time() - start_time < timeout:
#                 try:
#                     # 非阻塞方式读取队列
#                     output = output_queue.get_nowait()
#                     print(output.strip())  # 打印服务输出信息
                    
#                     if 'Starting vLLM API server' in output:
#                         print("\nvllm服务已启动，可通过以下信息访问：")
#                         print(f"URL: {os.environ['base_url']}")
#                         print(f"API_KEY: {os.environ['api_key']}")
#                         time.sleep(5)
#                         service_started = True
#                         break
                        
#                 except Empty:
#                     # 队列为空时短暂等待
#                     time.sleep(0.1)
#                     continue
            
#             if not service_started:
#                 # 超时或启动失败
#                 print("\nvllm服务启动超时或失败")
#                 self.process.terminate()  # 终止进程
#                 # 尝试获取剩余输出
#                 try:
#                     while True:
#                         output = output_queue.get_nowait()
#                         print(output.strip())
#                 except Empty:
#                     pass

# class Args:
#     def __init__(self, llm_type):
#         self.llm = llm_type
        
# if __name__ == "__main__":
#     args_vllm = Args(llm_type="vllm")
#     ma_instance_vllm = MA(args_vllm)
