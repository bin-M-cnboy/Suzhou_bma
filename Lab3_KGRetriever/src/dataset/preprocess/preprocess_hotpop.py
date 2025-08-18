import os
import sys
sys.path.append("/data0/bma/Lab3_KG_Retriever")# 使用相对路径
# sys.path.append(os.path.dirname(os.path.dirname("__file__")))

import re
from src.utils.lm_modeling import load_model, load_text2embedding
import torch
from torch_geometric.data.data import Data
import pandas as pd
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer,util
from src.dataset import load_dataset
from torch.utils.data import DataLoader
import json
import pickle

path = '../HOTPOPQA'

nodeslist = []
edgeslist = []
deslist = []

'''
[1] 三元组文本行编码: 三元组文本行_文件 -> nodeslist,edgeslist

三元组文本行_文件 -> step_one(): 
for file: textualize_graph(i,line)
-> 三元组
分批存入 -> nodeslist,edgeslist, des_文件
'''
def textualize_graph(i,line):# 从文本行中提取三元组并构建图结构
    # 查找所有三元组(src;rel;dst) -> triples
    triples = re.findall(r'\((.*?)\)', line)
    nodes = {}
    edges = []
    des = {}
    # triples -> nodes,edges,des
    for tri in triples:
        tri = tri.split(';') # 按分号分割三元组
        if len(tri) != 3:# 移除错误的三元组（长度不为3）
            if len(tri) >= 3:
                tri[2] = ''.join(tri[2:len(tri)]) # 如果长度大于3，将多余部分合并到第三个元素
            if len(tri) < 3:
                continue # 如果长度小于3，跳过此三元组
        
        src, edge_attr, dst = tri[0].strip(),tri[1].strip(),tri[2].strip()# 提取src,rel,dst
        # 添加新节点scr,dst -> nodes{}
        if src not in nodes:
            nodes[src] = len(nodes)     # node值 = 加入次序, node位置 = src值
        if dst not in nodes:
            nodes[dst] = len(nodes)
        # 添加边信息 -> edges[]          # edge值 = {三元组}
        edges.append({'src': nodes[src], 'edge_attr': edge_attr.strip(), 'dst': nodes[dst], })
        # 添加描述 -> des{}
        # 描述结构：des[src_id,dst_id]=(src,rel,dst) 
        des[str(nodes[src])+','+ str(nodes[dst])] = '(' + src +',' + edge_attr + ',' + dst + ')'

    # 将节点和边信息转换为 Pandas DataFrame 格式
    nodes = pd.DataFrame(nodes.items(), columns=['node_attr', 'node_id'])
    edges = pd.DataFrame(edges)
    return nodes,edges,des

# 生成知识图谱和描述文件
def step_one():
    # 创建用于存储描述文件的目录
    os.makedirs(f'{path}/KG_QA_vRobert/des500_new', exist_ok=True)
    cnt = 1 # 计数器，用于des文件命名
    tmp_nodeslist = [] 
    tmp_edgeslist = [] 
    tmp_deslist = [] 
    # file处理
    with open(f'{path}/outputs(500).txt', 'r') as file:
        for index, line in enumerate(file):
                # 如果不是分隔符行
                if line != '------------------------------------------\n':
                    # 提取图结构和描述
                    nodes, edges,des = textualize_graph(index,line)
                    tmp_nodeslist.append(nodes)
                    tmp_edgeslist.append(edges)
                    tmp_deslist.append(des)
                else:
                    # 遇到分隔符，将临时列表添加到全局列表
                    nodeslist.append(tmp_nodeslist)
                    edgeslist.append(tmp_edgeslist)
                    # 将描述信息保存为 pickle 文件
                    with open(f'{path}/KG_QA_vRobert/des500_new/{cnt}.pkl', 'wb') as file:
                        pickle.dump(tmp_deslist, file)
                    cnt += 1 # 计数器递增
                    # 重置临时列表
                    tmp_nodeslist = []
                    tmp_edgeslist = []
                    tmp_deslist = []

'''
[2] 编码知识图谱: nodeslist,edgeslist -> 知识图谱_文件/图张量_文件

nodeslist,edgeslist -> step_two():
取出nodes,edges -.node_attr列名> node_des[],edge_des[]
-> 使用roberta_LLM:
    node_des -> 张量x
    edge_des -> 张量e
torch: [edges.src, edges.dst] -> 索引张量edge_index
-> data(节点特征x, 边索引edge_index, 边特征edge_attr=e, 节点数量num_nodes)
存入-> 知识图谱_文件/图张量_文件/{i}/{j}
'''
def step_two():
    model = SentenceTransformer('../../../all-roberta-large-v1',device='cuda:1')

    # 创建用于存储知识图谱和图数据的目录
    os.makedirs(f'{path}/KG_QA_vRobert', exist_ok=True)
    os.makedirs(f'{path}/KG_QA_vRobert/graph500_new', exist_ok=True)

    # 遍历nodeslist
    for i in range(len(nodeslist)):
        # 为每个知识图谱创建子目录
        os.makedirs(f'{path}/KG_QA_vRobert/graph500_new/{i}', exist_ok=True)
        for j in range(len(nodeslist[i])):  # 取出nodes,edges
            nodes = nodeslist[i][j]
            edges = edgeslist[i][j]
            if not len(nodes) or not len(edges):# 如果节点或边为空, 跳过
                continue

            # 提取节点和边的描述
            node_des = nodes.node_attr.tolist()
            edge_des = edges.edge_attr.tolist()

            # 使用模型编码节点和边的描述，并转换为 PyTorch 张量
            x = model.encode(node_des,convert_to_tensor = True)
            e = model.encode(edge_des,convert_to_tensor = True)
            # 构建边的索引张量
            edge_index = torch.LongTensor([edges.src, edges.dst])

            # 创建 PyTorch Geometric 的 Data 对象，包含节点特征、边索引、边特征和节点数量
            data = Data(x=x, edge_index=edge_index, edge_attr=e, num_nodes=len(nodes))

            # 保存图数据到文件
            torch.save(data, f'{path}/KG_QA_vRobert/graph500_new/{i}/{j}.pt')

'''
[3] 编码问题: 数据集_文件.question -> 问题张量_文件

数据集_文件 -> encode_question()
取出question
-> 使用roberta_LLM:
    question -> 张量question_emb
存入 -> 问题张量_文件/{i}
'''
def encode_question():
    model = SentenceTransformer('../../../all-roberta-large-v1')
    # 创建用于存储问题嵌入的目录
    os.makedirs(f'{path}/questions_emb_vRobert', exist_ok=True)

    # 定义 HOTPOPQA 开发集的数据路径
    data_path = '../HOTPOPQA/hotpot_dev_fullwiki_v1.json'
    # 加载 JSON 格式的数据
    with open(data_path, 'r') as file:
        data = json.load(file)

    # 遍历数据集中的每个问题
    for i in range(len(data)):
        question = data[i]["question"] # 提取问题文本
        # 编码问题文本并转换为 PyTorch 张量
        question_emb = model.encode(question, convert_to_tensor=True)
        # 保存问题嵌入到文件
        torch.save(question_emb, f'{path}/questions_emb_vRobert/{i}.pt')

'''
[3] 编码文档: 数据集_文件.context -> 知识图谱_文件/文档张量_文件

数据集_文件 -> encode_doc()
取出context -> docs
取前500个数据项 -> 使用roberta_LLM:
                    拼合过的passage -> 张量doc_emb
存入 ->
doc_emb -> doc_emb_list -> 知识图谱_文件/文档张量_文件/{500}
passage -> doc_text_list -> 知识图谱_文件/文档列表_二进制文件/{500}

'''
def encode_doc():
    model = SentenceTransformer('../../../all-roberta-large-v1',device='cuda:1')

    # 创建用于存储知识图谱、文档文本和文档嵌入的目录
    os.makedirs(f'{path}/KG_QA_vRobert', exist_ok=True)
    os.makedirs(f'{path}/KG_QA_vRobert/doc_text', exist_ok=True)
    os.makedirs(f'{path}/KG_QA_vRobert/doc_emb', exist_ok=True)

    # 定义 HOTPOPQA 开发集的数据路径
    data_path = '../HOTPOPQA/hotpot_dev_fullwiki_v1.json'
    # 加载 JSON 格式的数据
    with open(data_path, 'r') as file:
        data = json.load(file)

    doc_text_list = [] # 存储文档文本列表
    doc_emb_list = [] # 存储文档嵌入列表

    # 遍历前 500 个数据项（假设每个数据项包含多个文档）
    for i in range(500):
        docs = data[i]["context"] # 提取上下文文档
        for doc in docs:
            # 将文档中的段落拼接成一个完整的文本
            passage = ''.join(doc[1][j] for j in range(len(doc[1])))
            # 编码文档文本并转换为 PyTorch 张量
            doc_emb = model.encode(passage, convert_to_tensor=True)
            doc_emb_list.append(doc_emb)
            doc_text_list.append(passage)

    # 保存所有文档嵌入到一个文件中
    torch.save(torch.stack(doc_emb_list), f'{path}/KG_QA_vRobert/doc_emb/{500}.pt')
    # 保存所有文档文本到 pickle 文件中
    with open(f'{path}/KG_QA_vRobert/doc_text/{500}.pkl', 'wb') as file:
        pickle.dump(doc_text_list, file)

if __name__ == '__main__':
    step_one() # 生成知识图谱和描述
    print("--- step_one() DONE! ---")
    step_two() # 编码知识图谱
    print("--- step_two() DONE! ---")
    encode_question() # 编码问题
    print("--- encode_question() DONE! ---")
    encode_doc() # 编码文档
    print("--- encode_doc() DONE! ---")
    print("--- preprocess_hotpop.py DONE! ---")

'''
preprocess_hotpop.py

'{path}/outputs(500).txt'三元组 -> 知识图谱_文件/图张量_文件      # 得到 实体级层
数据集_文件 -> 知识图谱_文件/问题张量_文件, 文档张量_文件           # 得到 文档级层
'''
