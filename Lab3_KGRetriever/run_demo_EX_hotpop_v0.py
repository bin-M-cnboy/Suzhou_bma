import torch 
import gc
from src.config import parse_args_llama         
from src.model import load_model, llm_model_path
from src.dataset import load_dataset 
from torch.utils.data import DataLoader 
from src.retrieval import retrieval_func_two_stage,retrieval_func_via_doc
import os 
import pickle
import datetime 

def main(args):
    batch_size = 1
    n = 0 # 起始处理的数据索引
    break_p = XXX # 结束处理的数据索引，XXX是一个占位符，取前K条数据进行实验

    '''
    加载过程:
    模型，预处理数据，问题嵌入，知识图谱数据，文档嵌入
    '''
    # 加载模型
    args.llm_model_path = llm_model_path[args.llm_model_name]
    model = load_model[args.model_name](args=args)              
    # 加载数据
    test_dataset_path = './src/dataset/HOTPOPQA/hotpot_dev_fullwiki_v1.json'
    test_dataset = load_dataset['HOTPOPQA'](test_dataset_path,size1=n,size2=break_p)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, pin_memory=True, shuffle=False)

    # 加载问题嵌入
    questions_emb_list = []
    for i in range(break_p):
        question_emb = torch.load(f"./src/dataset/HOTPOPQA/questions_emb_vRobert/{i}.pt")
        questions_emb_list.append(question_emb)

    # 加载知识图谱数据
    KG_lists = []
    test_folder_path = "./src/dataset/HOTPOPQA/KG_QA_vRobert"
    for i in range(n,break_p):
        KG_path = f'{test_folder_path}/graph/{i}/'
        if os.path.exists(KG_path):
            for j in range(20):
                if os.path.exists(KG_path + f'{j}.pt'):
                    KG = torch.load(KG_path + f'{j}.pt')
                    KG_lists.append(KG)
                else:
                    break
        else: # 如果路径不存在，也尝试添加KG    # ???
            KG_lists.append(KG)

    # 加载文档嵌入张量
    doc_embed_tensor = torch.load(f"./src/dataset/HOTPOPQA/KG_QA_vRobert/doc_emb/{XXX}.pt")
    doc_embed_tensor = doc_embed_tensor[:len(KG_lists)] # 取前{len(KG_lists)}个

    '''
    计算过程    # ???自己跟自己找相似
    '''
    # 将文档嵌入作为子图列表张量
    SG_lists_tensor = doc_embed_tensor
    # 计算内积。内积越大，向量方向越接近
    inner_product = torch.matmul(SG_lists_tensor, SG_lists_tensor.T)
    # 计算L2范数。欧式距离
    l2_norm = torch.norm(SG_lists_tensor, p=2, dim=1, keepdim=True)
    # 归一化内积
    n_prizes = inner_product / (l2_norm * l2_norm.T)
    # 获取前3个最大值及索引(最相似的3个)
    topk_n_values, topk_n_indices = torch.topk(n_prizes, 3, largest=True)
    # 转置id，获取3个子图的边
    SG_edges = topk_n_indices.T

    # 加载描述信息des[]
    des = []
    for i in range(n,break_p):
        des_path = f'{test_folder_path}/des/' + f'{i+1}.pkl'
        f = open(des_path, 'rb')
        tmp_des = pickle.load(f)
        for j in range(len(tmp_des)):
            des.append(tmp_des[j])

    '''
    评估阶段
    v0.py --- retrieval_func_two_stage()
    v1.py --- retrieval_func_with_att()
    '''
    predictions = [] # 存储模型预测结果
    labels = [] # 存储真实标签

    current_time = datetime.datetime.now()
    print('Times:', current_time) # 打印当前时间
    for step, batch in enumerate(test_loader):
        if step * batch_size < n:
            continue

        with torch.no_grad():
            ids = batch['id'].tolist()
            questions_emb = [questions_emb_list[i] for i in ids]
            # 执行两阶段检索，获取额外知识、掩码和检索到的描述
            test_extra_knowledges, _, masks,retri_des = retrieval_func_two_stage(questions_emb, doc_embed_tensor, KG_lists, SG_edges,des,device=model.device(),topk_n=20,intervals=0)
            # 使用模型进行文本推理
            output = model.inference_text(batch,retri_des)
            for j in range(len(output['questions'])):
                predictions.append(output['pred'][j]) # 添加预测结果
                labels.append(output['answers'][j]) # 添加真实答案

    # 将预测结果保存到pkl文件
    with open('./answer_list/ours_hotpop.pkl', 'wb') as file2:
        pickle.dump(predictions,file2)

    current_time = datetime.datetime.now()
    print('Times:', current_time) # 再次打印当前时间

    # 计算精确匹配(Exact Match)率
    Truenums1 = 0
    for i in range(len(predictions)):
        if labels[i].lower() in predictions[i].lower():
            Truenums1 += 1
    print('EM: ',Truenums1/len(predictions))


if __name__ == "__main__":
    args = parse_args_llama()

    main(args)
    
    torch.cuda.empty_cache()                # 清空CUDA缓存
    torch.cuda.reset_max_memory_allocated() # 重置CUDA最大内存分配记录
    gc.collect()                            # 执行垃圾回收