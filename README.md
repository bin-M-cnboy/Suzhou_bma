# 学习总结💡💡💡

学生马斌，于大一暑期期间(2025.7.12 - 2025.8.24)有幸受徐宏力老师引荐，在<u>中科大苏州高研院</u>进行学习交流。
期间备受许阳老师、姚志伟师兄、徐梓淮师兄等前辈的照顾与指点，不胜感激。
为总结学习成果，也为纪念此次难忘又感激的学习经历，特整理此文。

## 概述
本次学习收获大致可总结为三点：
1. 论文调研📑
2. 代码经验💻
3. 技能学习🌏

## 1. 论文调研📑
调研论文主要围绕 **AI Agent** 展开。具体可分为以下方向：

### AIOS
以 *LLM as OS, Agents as Apps<sup>[1-Report](SuZhou_Study/Report_PPTs/1-Report-20250712.pdf)</sup>* 为主线，对 AIOS 的整体架构及各个重要组件进行探索。
我根据AIOS有关论文，梳理了一条完整的技术发展路线<sup>[2-Report](SuZhou_Study/Report_PPTs/2-Report-20250716.pdf)</sup>，并且具体分析了整个架构<sup>[4-Report](SuZhou_Study/Report_PPTs/4-Report-20250718.pdf)</sup>。

### Graph and Agent
以 *Graphs Meet AI Agents<sup>[5-Report](SuZhou_Study/Report_PPTs/5-Report-20250721-图基智能体综述.pdf)</sup>* 为主线，探索了**图技术**与**智能体技术**之间交叉领域的一些问题。
之后我针对于其中的**拓扑优化多智能体通信框架**方向进行了我的首次论文复现尝试，复现在艰难中进行，期间受师兄们指点，学习积累了宝贵的经验。
最后，我完成了两种不同实现方式的实验复现：
- Lab1 --- API接入openai大模型<sup>[6-Report](SuZhou_Study/Report_PPTs/6-Report-20250723.pdf)</sup>
- Lab2 --- 通过vllm使用本地qwen小模型<sup>[7-Report](SuZhou_Study/Report_PPTs/7-Report-20250725.pdf)</sup>

之后，我又对Graph-learnig Agents方向中的工具调用方向<sup>[8-Report](SuZhou_Study/Report_PPTs/8-Report-20250801-工具调用方向.pdf)</sup>和记忆组织方向<sup>[9-Report](SuZhou_Study/Report_PPTs/9-Report-20250804-记忆组织方向.pdf)</sup>进行更加细致的探索。
发现**GraphRAG**这一流行且充满潜力的研究方向。

### GraphRAG
**检索增强生成(RAG)** 目前已被广泛应用，而 **基于图的检索增强生成(GraphRAG)** 则作为一个崭新的交叉领域备受瞩目。
我整理了近期的一些研究成果<sup>[10-Report](SuZhou_Study/Report_PPTs/10-Report-20250810-GraphRAG.pdf)</sup>，在与徐师兄的交流后对基于因果图的检索增强生成(CausalRAG)进行进一步的调研<sup>[11-Report](SuZhou_Study/Report_PPTs/11-Report-20250813-RAGandcCausalRAG.pdf)</sup>，发现此领域现有研究较少且富有研究价值，并计划进行进一步的研究。

## 2. 代码经验💻
在复现实验和交流过程中，我收获到许多实用的代码经验。

- 使用conda进行环境管理
- 通过huggingface查找并下载模型及数据集资源
- 使用vllm完成本地模型部署并使用
- 对服务器进程进行操作和管理，在GPU上完成实验
- 使用git&github管理代码
- 通过修改及添加系统路径，解决import异常问题

## 3. 技能学习🌏
正常完成任务之余，我也或直接或间接地零碎地学习了不少实用技能，仅列举几项我个人认为颇具价值的技能。

- PPT更加深入的理解与应用
- Zotero及各种论文文献的查找及管理
- GPT、Llama框架学习
- Pytorch框架初步入门
- Manim科学动画入门
- Kaggle数据科学基本知识学习

虽然最直接的感受就是**PPT制作更加熟练**和**英文文献阅读速度大幅提升**，但是在与师兄们的交流和亲身的经历中，也让我认识问题和思考解决方案的方式有了较大的改变。

## 总结
这是一段十分难忘且宝贵的经历，给我后续的本科学业学习及课外科研工作注入了完全不同的新鲜活力。
最后，再次感谢徐宏力老师给我这个宝贵的学习机会，学生马斌感激不尽！


---
---
## 版权声明（Copyright Notice）
本仓库（SuZhou_Study）内所有文件（包括但不限于代码、文档、图片、数据等）的版权归作者 **马斌** 所有。

**未经作者书面允许，任何个人或组织不得擅自使用、复制、修改、分发、传播本仓库中的任何内容，包括但不限于用于商业用途、二次开发、公开传播等行为。**

---
*Last updated: 2025.8.19*


