# 科大苏州高研院学习总结

学生马斌，于大一暑期期间(2025.7.12 - 2025.8.24)有幸受徐宏力老师引荐，在科大苏州高研院进行学习交流。
期间备受许阳老师、姚志伟师兄、徐梓淮师兄等前辈的照顾，不胜感激。
为总结学习成果也为纪念此次难忘又感激的学习经历，特整理此文。

## 概述
本次学习收获大致可总结为三点：
1. 论文调研
2. 代码经验
3. 技能学习

## 1. 论文调研
调研论文主要围绕 **AI Agent** 展开。具体可分为以下方向：

### AIOS
以 *LLM as OS, Agents as Apps[^1-Report](SuZhou_Study/Report_PPTs/1-Report-20250712.pdf)* 为主线，对AIOS的整体架构及各个重要组件进行探索。
我根据AIOS有关论文，梳理了一条完整的技术发展路线[^2-Report](SuZhou_Study/Report_PPTs/2-Report-20250716.pdf)，并且具体分析了整个个架构[^4-Report](SuZhou_Study/Report_PPTs/4-Report-20250718.pdf)。

### Graph and Agent
以 *Graphs Meet AI Agents[^5-Report](SuZhou_Study/Report_PPTs/5-Report-20250721-图基智能体综述.pdf)* 为主线，探索图技术于智能体技术之间的交叉互利关系。
之后我针对于其中**拓扑优化多智能体通信框架**的方向进行了我的首次论文复现尝试，复现艰难进行，期间受师兄们指点，学习积累了宝贵的经验。
完成了两种实现方式的复现实验：
- Lab1 --- API接入openai大模型[^6-Report](SuZhou_Study/Report_PPTs/6-Report-20250723.pdf)
- Lab2 --- 通过vllm使用本地qwen小模型[^7-Report](SuZhou_Study/Report_PPTs/7-Report-20250725.pdf)

之后，我又对Graph-learnig Agents方向中的工具调用方向[^8-Report](SuZhou_Study/Report_PPTs/8-Report-20250801-工具调用方向.pdf)和记忆组织方向[^9-Report](SuZhou_Study/Report_PPTs/9-Report-20250804-记忆组织方向.pdf)进行更加细致的探索。
发现**GraphRAG**这一流行且充满潜力的研究方向。

### GraphRAG
检索增强生成(RAG)目前已被广泛应用，而基于图的检索增强生成(GraphRAG)则作为一个崭新的交叉领域备受瞩目。
我整理了近期的一些研究成果[^10-Report](SuZhou_Study/Report_PPTs/10-Report-20250810-GraphRAG.pdf)，在与师兄的交流后对基于因果图的检索增强生成(CausalRAG)进行进一步的调研[^11-Report](SuZhou_Study/Report_PPTs/11-Report-20250813-RAGandcCausalRAG.pdf)，发现此领域研究较少且富有研究价值，计划进行进一步的研究。

## 2. 代码经验
在复现实验过程收获到许多实用代码经验。

- 使用conda进行环境管理
- 通过huggingface查找并下载模型及数据集资源
- 使用vllm完成本地模型部署并使用
- 对服务器进程进行操作和管理
- 使用git&github管理代码
- 通过修改及添加系统路径，解决import异常问题

## 3. 技能学习
零碎地学习了不少使用技能，仅列举几项个人认为比较记忆深刻的技能。

- PPT更加深入的理解与应用
- Zotero及各种论文文献的查找及管理
- GPT、Llama框架学习
- Pytorch框架初步入门
- Manim科学动画入门
- Kaggle数据科学基本知识学习

最直接的感受就是**PPT制作更加熟练**和**英文文献阅读速度大幅提升**，与师兄们的交流和亲身的经历也让我思考问题和思考解决方案的方式有了较大的改变。

