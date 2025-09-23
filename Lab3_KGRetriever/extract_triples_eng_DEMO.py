from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch

device = "cuda:0" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "Qwen1.5-32B-Chat-AWQ",
    device_map=device
)
tokenizer = AutoTokenizer.from_pretrained("Qwen1.5-32B-Chat-AWQ")

path = './src/dataset/HOTPOPQA'

answer_list = []
question_list = []
docs_list = []
cnt = 0

with open(f'{path}/hotpot_dev_fullwiki_v1.json', 'r') as file:
    data = json.load(file)

    for i in range(len(data)):
        answer = data[i]['answer']
        question = data[i]['question']
        docs = data[i]['context']

        tmpd = []
        for j in range(len(docs)):
            tmpd.append(docs[j][1])

        answer_list.append(answer)
        question_list.append(question)
        docs_list.append(tmpd)


for docs in docs_list:
    ids_list = []
    cnt += 1
    if cnt > 186:   # 只处理第186条以后的数据
        for i in range(len(docs)):# 打包输入
            sentence = ''.join(docs[i][j] for j in range(len(docs[i])))# 将文档中的句子连接成完整文本

            messages = [{"role": "system",
                              "content": "You are an NLP assistant. Given a piece of text, you need to analyze its semantic information and generate a knowledge graph. Your output consists only of triples, considering only the text content, and avoiding newline characters. For example, (entity; relationship; entity),(entity; relationship; entity). The knowledge graph should be comprehensive, covering all information in the text."},
                        {"role": "user",
                              "content": 'Adam Collis is an American filmmaker and actor. He attended the Duke University from 1986 to 1990 and the University of California, Los Angeles from 2007 to 2010. He also studied cinema at the University of Southern California from 1991 to 1997. Collis first work was the assistant director for the Scott Derrickson\'s short "Love in the Ruins" (1995). In 1998, he played "Crankshaft" in Eric Koyanagi\'s "Hundred Percent".'},
                                  {"role": "assistant",
                              "content": '(Adam Collis; nationality; American),(Adam Collis; profession; filmmaker),(Adam Collis; profession; actor),(Adam Collis; education; Duke University),(Adam Collis; education; University of California, Los Angeles),(Adam Collis; education; University of Southern California),(Adam Collis; attended; Duke University),(Adam Collis; attended; University of California, Los Angeles),(Adam Collis; attended; University of Southern California),(Adam Collis; first work; assistant director),(Adam Collis; work; "Love in the Ruins"),("Love in the Ruins"; director; Scott Derrickson),(Adam Collis; work date; 1995),(Adam Collis; role; "Crankshaft"),("Hundred Percent"; director; Eric Koyanagi),(Adam Collis; work; "Hundred Percent"),(Adam Collis; work date; 1998)'},
                        {"role": "user",
                         "content": 'Tyler Bates (born June 5, 1965) is an American musician, music producer, and composer for films, television, and video games. Much of his work is in the action and horror film genres, with films like "Dawn of the Dead, 300, Sucker Punch," and "John Wick." He has collaborated with directors like Zack Snyder, Rob Zombie, Neil Marshall, William Friedkin, Scott Derrickson, and James Gunn. With Gunn, he has scored every one of the director\'s films; including "Guardians of the Galaxy", which became one of the highest grossing domestic movies of 2014, and its 2017 sequel. In addition, he is also the lead guitarist of the American rock band Marilyn Manson, and produced its albums "The Pale Emperor" and "Heaven Upside Down".'},
                        {"role": "assistant",
                         "content": '(Tyler Bates; birthdate; June 5, 1965),(Tyler Bates; nationality; American),(Tyler Bates; profession; musician),(Tyler Bates; profession; music producer),(Tyler Bates; profession; composer),(Tyler Bates; works in; films),(Tyler Bates; works in; television),(Tyler Bates; works in; video games),(Tyler Bates; specializes in; action and horror film genres),(Tyler Bates; notable films; "Dawn of the Dead"),(Tyler Bates; notable films; "300"),(Tyler Bates; notable films; "Sucker Punch"),(Tyler Bates; notable films; "John Wick"),(Tyler Bates; collaborations; Zack Snyder),(Tyler Bates; collaborations; Rob Zombie),(Tyler Bates; collaborations; Neil Marshall),(Tyler Bates; collaborations; William Friedkin),(Tyler Bates; collaborations; Scott Derrickson),(Tyler Bates; collaborations; James Gunn),(Tyler Bates; collaborations; James Gunn),(Tyler Bates; collaborations; James Gunn),(Tyler Bates; scored; "Guardians of the Galaxy"),("Guardians of the Galaxy"; release year; 2014),("Guardians of the Galaxy"; grossing; high),("Guardians of the Galaxy"; sequel; released in 2017),(Tyler Bates; lead guitarist; Marilyn Manson),(Marilyn Manson; music albums; "The Pale Emperor"),(Marilyn Manson; music albums; "Heaven Upside Down")'}
                        ]
            messages.append({"role": "user",
                            "content": sentence})

            text = tokenizer.apply_chat_template(  # 将messages输入
                messages,
                tokenize=False,
                add_generation_prompt=True  # 添加生成提示
            )
            # 将文本转换为模型输入张量
            model_inputs = tokenizer([text], return_tensors="pt").to(device)
            ids_list.append(model_inputs.input_ids.squeeze())   # 存储ID序列
        
        # 对所有id进行padding填充对齐
        max_length = max([x.shape[0] for x in ids_list])
        pad_id = tokenizer.pad_token_id     # 获取需填充的ID
        for i in range(len(ids_list)):
            pad_length = max_length - ids_list[i].shape[0]
            ids_list[i] = torch.concat([torch.tensor(pad_id, device=device).repeat(pad_length), ids_list[i]], 0)

        ids_tensor = torch.stack(ids_list) # 将ids_list列表转换为张量ids_tensor

        # 生成知识图谱三元组
        generated_ids = model.generate( # 输入ids_tensor, model输出结果generated_ids
            ids_tensor,
            max_new_tokens=2048 # 最大新生成的token数量
        )
        generated_ids = [   # generated_ids = [len(input_ids) -> output_ids] -> 使仅包含新生成部分的ID
            output_ids[len(input_ids):] for input_ids, output_ids in zip(ids_tensor, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)# 将生成的ID转换为文本

        for i in range(len(response)):
            print(response[i])
        print('------------------------------------------')

'''
extract_triples_eng_DEMO.py

数据集.json_文件 ->
取出answer,question,context -> answer_list,question_list,docs_list
doc -> 长文本sentence -> 输入张量model_inputs 
(仅用id ??? ) -> 输入张量的id列表 ids_list -> 对齐的id张量 ids_tensor
-> 使用Qwen_LLM -> generated_ids -> 解码为知识图谱三元组 response[]
人工整理为文件 -> 三元组文本行_文件
'''