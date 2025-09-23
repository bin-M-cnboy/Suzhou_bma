import sys
import os
import argparse # 添加命令行参数
import yaml
import json
import time
import asyncio # 异步函数
from pathlib import Path
import torch
import copy
from typing import List,Union,Literal
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from AgentPrune.graph.autogen_graph import GraphAutoGen
from AgentPrune.tools.reader.readers import JSONLReader
from AgentPrune.tools.coding.python_executor import PyExecutor
from AgentPrune.utils.globals import Time
from AgentPrune.utils.const import AgentPrune_ROOT
from AgentPrune.utils.globals import Cost, PromptTokens, CompletionTokens

def load_result(result_file):
    if not result_file.exists():
        with open(result_file, 'w',encoding='utf-8') as file:
            json.dump([], file)

    with open(result_file, 'r',encoding='utf-8') as file:
        data = json.load(file)
    return data

def dataloader(data_list, batch_size, i_batch):
    return data_list[i_batch*batch_size:i_batch*batch_size + batch_size]

def load_config(config_path):
    with open(config_path, 'r',encoding='utf-8') as file:
        return yaml.safe_load(file)
    
def parse_args():
    parser = argparse.ArgumentParser(description="AgentPrune Experiments on HumanEval") # 创建命令行参数解析器
    parser.add_argument("--dataset_json", type=str, default="/data0/bma/Lab2/dataset/humaneval/humaneval-py.jsonl")# default 默认值
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--llm_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument('--mode', type=str, default='Chain',
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain','Debate','Layered','Star'],
                        help="Mode of operation. Default is 'Chain'.")
    parser.add_argument('--lr', type=float, default=0.01,help="learning rate")
    parser.add_argument('--batch_size', type=int, default=4,help="batch size")
    parser.add_argument('--imp_per_iterations', type=int, default=5,help="Prune every few iterations. Default 5.")
    parser.add_argument('--num_rounds',type=int,default=2,help="Number of optimization/inference rounds for one query")
    parser.add_argument('--pruning_rate', type=float, default=0.25,help="The Rate of Pruning. Default 0.05.")
    parser.add_argument('--num_iterations', type=int, default = 10,help="The num of training iterations.")
    parser.add_argument('--domain', type=str, default="humaneval",help="Domain (the same as dataset name), default 'humaneval'")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['CodeWritingAG'],
                        help='Specify agent names as a list of strings')# nargs='+' 可接受多个值，解析为一个列表。
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[5],
                        help='Specify the number of agents for each name in agent_names')
    parser.add_argument('--decision_method', type=str, default='FinalWriteCodeAG',
                        help='The decison method of the agentprune')
    parser.add_argument('--optimized_spatial',action='store_true') # action='store_true' 出现此参数时设置为True
    parser.add_argument('--optimized_temporal',action='store_true')
    
    args = parser.parse_args()  # 传出命令行参数解析器
    result_path = AgentPrune_ROOT / "result"
    # AgentPrune_ROOT = Path(os.path.realpath(os.path.join(os.path.split(__file__)[0], "../..")))
    os.makedirs(result_path, exist_ok=True)# 没有目录递归创建目录
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")

    return args

async def main():
    args = parse_args()# 调用命令行参数解析器
    result_file = None
    dataset = JSONLReader.parse_file(args.dataset_json)
    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    result_dir = Path(f"{AgentPrune_ROOT}/result/eval")
    result_dir.mkdir(parents=True, exist_ok=True)# 创建所需的输出目录
    result_file = result_dir / f"{args.llm_name}_{current_time}.json"


    
    agent_names = [name for name,num in zip(args.agent_names,args.agent_nums) for _ in range(num)]
    decision_method = args.decision_method
    kwargs = get_kwargs(args.mode,len(agent_names)) 
    # kwargs = 根据程序的运行模式args.mode和代理数量len(agent_names)来获取一组配置参数 
    graph = GraphAutoGen(domain="humaneval",
                  llm_name=args.llm_name,
                  agent_names=agent_names,
                  decision_method=decision_method,
                  optimized_spatial=args.optimized_spatial,
                  optimized_temporal=args.optimized_temporal,
                  **kwargs)
    optimizer = torch.optim.Adam([graph.spatial_logits,graph.temporal_logits], lr=args.lr) # 创建Adam优化器，能自适应地调整每个参数的学习率
    # [模型参数], lr=步长

    num_batches = int(len(dataset)/args.batch_size)
    total_solved, total_executed = (0, 0)
    for i_batch in range(num_batches):          # 遍历每个batch
        print(f"Batch {i_batch}",80*'-')
        start_ts = time.time()
        answer_log_probs = []
        tests = []
        
        current_batch = dataloader(dataset,args.batch_size,i_batch)
        if current_batch is None:
            print("No more data available.")
            break
        
        for i_record, record in enumerate(current_batch):
            realized_graph = copy.deepcopy(graph)   # 深拷贝，递归地复制原始对象及其包含的所有子对象
            realized_graph.spatial_logits = graph.spatial_logits
            realized_graph.temporal_logits = graph.temporal_logits
            task = record["prompt"]
            test = record["test"]
            # 从 record 中解析出任务的描述(prompt)和对应的测试用例(test)
            tests.append(test)
            input_dict = {"task": task}
            answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict,args.num_rounds)))
        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs = zip(*raw_results)
        loss_list: List[torch.Tensor] = []
        utilities: List[float] = []
        data = load_result(result_file)
        
        for task, answer, log_prob, test in zip(current_batch, raw_answers, log_probs, tests):
            if not isinstance(answer,list):
                raise TypeError(f"Expected a list for the answer, but got {type(answer).__name__}")
            answer = answer[0].lstrip("```python\n").rstrip("\n```")# 移除其可能包含的特定格式标记
            is_solved, _, _ = PyExecutor().execute(answer, [test], timeout=100)
            total_solved = total_solved + is_solved
            total_executed = total_executed + 1
            accuracy = total_solved/ total_executed
            utility = is_solved
            utilities.append(utility)
            single_loss = -log_prob * utility
            loss_list.append(single_loss)
            updated_item = {                # 更新参数的小结
                "Question": task,
                "Tests": test,
                "Attempt answer": answer,
                "Solved": is_solved,
                "Solution": answer,
                "Total solved": total_solved,
                "Total executed": total_executed,
                "Accuracy": accuracy
            }
            data.append(updated_item)
        with open(result_file, 'w',encoding='utf-8') as file:
            json.dump(data, file, indent=4)# 将Python对象编码为JSON格式，并写入到文件。indent=缩进级别
        
        total_loss = torch.mean(torch.stack(loss_list))
        if args.optimized_spatial or args.optimized_temporal:
            optimizer.zero_grad()   # 反向传播之前模型参数的梯度清零
            total_loss.backward()   # 反向传播，计算梯度
            optimizer.step()        # 根据计算出的梯度更新模型参数
        spatial_probs = torch.sigmoid(graph.spatial_logits)
        temporal_probs = torch.sigmoid(graph.temporal_logits)
        # 将输入张量中的每个元素映射到 (0, 1) 的范围内
        
        print(f"Batch time {time.time() - start_ts:.3f}")
        print(f"Accuracy: {accuracy}")
        print("utilities:", utilities)

        if (i_batch+1)%args.imp_per_iterations == 0 and i_batch < args.num_iterations and (args.optimized_spatial or args.optimized_temporal):
            spatial_masks, temporal_masks = graph.update_masks(args.pruning_rate)
            # 总迭代次数完成之前，且空间或时间参数需要优化时，周期性地（每隔 args.imp_per_iterations 次迭代）更新模型的空间和时间掩码
        if i_batch+1 == args.num_iterations:    # 该batch所有迭代完成后
            args.optimized_spatial = False
            args.optimized_temporal = False


def get_kwargs(mode:Union[Literal['DirectAnswer'],Literal['FullConnected'],Literal['Random'],Literal['Chain'],Literal['Debate'],Literal['Layered'],Literal['Star']],
               N:int):
    initial_spatial_probability: float = 0.5
    fixed_spatial_masks:List[List[int]] = None
    initial_temporal_probability: float = 0.5
    fixed_temporal_masks:List[List[int]] = None
    node_kwargs = None
    
    def generate_layered_graph(N,layer_num=2):
        adj_matrix = [[0 for _ in range(N)] for _ in range(N)]  # 初始化一个 N x N 的二维列表
        base_size = N // layer_num
        remainder = N % layer_num
        layers = []
        for i in ranger(laye_num):
            size = base_size + (1 if i < remainder else 0)  # 将余数remainder分配给前面的层
            layers.extend([i] * size)   # 创建好layers[]，长度N，元素为i_laye_num。 如[0, 0, 0, 1, 1]
        random.shuffle(layers)          # 随机打乱layers
        # 构建邻接矩阵。相邻层间元素全连接
        for i in range(N):
            current_layer = layers[i]
            for j in range(N):
                if layers[j] == current_layer + 1:
                    adj_matrix[i][j] = 1
        return adj_matrix
    
    def generate_star_graph(n):
        matrix = [[0] * n for _ in range(n)]
        # 构建上三角矩阵，全图
        for i in range(0, n):
            for j in range(i+1,n):
                matrix[i][j] = 1
        return matrix

    if mode=='DirectAnswer':
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{'role':'Programming Expert'}]
    elif mode=='FullConnected':
        fixed_spatial_masks = [[1 if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode=='Random':
        fixed_spatial_masks = [[random.randint(0, 1)  if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
    elif mode=='Chain':
        fixed_spatial_masks = [[1 if i==j+1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i==0 and j==N-1 else 0 for i in range(N)] for j in range(N)]
    elif mode == 'Debate':
        fixed_spatial_masks = [[0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Layered':
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Star':
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    
    return {"initial_spatial_probability": initial_spatial_probability,
            "fixed_spatial_masks": fixed_spatial_masks,
            "initial_temporal_probability": initial_temporal_probability,
            "fixed_temporal_masks": fixed_temporal_masks,
            "node_kwargs":node_kwargs}    

if __name__ == '__main__':
    asyncio.run(main())