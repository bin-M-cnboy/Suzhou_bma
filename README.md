<!-- GitHub 兼容的语言切换（使用锚点链接，无需 JavaScript） -->
[<kbd>🇨🇳 中文</kbd>](/workspaces/Suzhou_bma/README.md#zh-CN) · [<kbd>🇬🇧 English</kbd>](/workspaces/Suzhou_bma/README.md#en)

<a id="zh-CN"></a>

<!-- 中文内容 -->
<div class="tabcontent">
# 学习总结💡💡💡

学生马斌，于大一暑期期间(2025.7.12 - 2025.8.24)有幸受徐宏力老师引荐，到<u>中科大苏州高研院</u>进行学习交流。  
期间备受许杨老师、姚志伟师兄、徐梓淮师兄等前辈的照顾与指点，不胜感激。  
为总结学习成果，也为纪念此次难忘又满怀感激的学习经历，特整理此文。  

## 概述
本次学习的收获大致可总结为三点：  
1. 论文调研📑
2. 代码经验💻
3. 技能学习🌏

## 1. 论文调研📑
调研论文主要围绕 **AI Agent** 展开。具体为：  

### AIOS
以 *LLM as OS, Agents as Apps<sup>[1-Report](Report_PPTs/1-Report-20250712.pdf)</sup>* 为主线，对 AIOS 的整体架构及各个重要组件进行探索。    
我根据AIOS有关论文，梳理了一条完整的技术发展路线<sup>[2-Report](Report_PPTs/2-Report-20250716.pdf)</sup>，并且具体分析了整个架构<sup>[4-Report](Report_PPTs/4-Report-20250718.pdf)</sup>。

### Graph and Agent
以 *Graphs Meet AI Agents<sup>[5-Report](Report_PPTs/5-Report-20250721-图基智能体综述.pdf)</sup>* 为主线，探索了**图技术**与**智能体技术**之间交叉领域的一些问题。   
之后我针对于其中的**拓扑优化多智能体通信框架**方向进行了我的首次论文复现尝试，复现在艰难中进行，期间受师兄们指点，学习积累了宝贵的经验。  
最后，我完成了两种不同实现方式的实验复现：   
- Lab1 --- API接入openai大模型<sup>[6-Report](Report_PPTs/6-Report-20250723.pdf)</sup>
- Lab2 --- 通过vllm使用本地qwen小模型<sup>[7-Report](Report_PPTs/7-Report-20250725.pdf)</sup>

之后，我又对Graph-learnig Agents方向中的工具调用方向<sup>[8-Report](Report_PPTs/8-Report-20250801-工具调用方向.pdf)</sup>和记忆组织方向<sup>[9-Report](Report_PPTs/9-Report-20250804-记忆组织方向.pdf)</sup>进行更加细致的探索。
发现**GraphRAG**这一流行且充满潜力的研究方向。

### GraphRAG
**检索增强生成(RAG)** 目前已被广泛应用，而 **基于图的检索增强生成(GraphRAG)** 则作为一个崭新的交叉领域备受瞩目。  
我整理了近期的一些研究成果<sup>[10-Report](Report_PPTs/10-Report-20250810-GraphRAG.pdf)</sup>，在与徐师兄的交流后对基于因果图的检索增强生成(CausalRAG)进行进一步的调研<sup>[11-Report](Report_PPTs/11-Report-20250813-RAGandcCausalRAG.pdf)</sup>，发现此领域现有研究较少且富有研究价值，并计划进行进一步的研究。

### 总结工作
在离开苏州后，我又利用课余时间总结了前段科研过程中的学习收获和感悟。  

**AI infra** 和 **RAG**<sup>[RAG-NOTE](Report_PPTs/RAG基本框架 - NOTE.pdf)</sup> 作为现今最流行的AI可落地应用层技术，两者分别在硬件和软件设计上对大模型利用进行优化，其流行的原因归功于其符合多数AI落地场景下对**降低代价**和**个性化**的需求，其对于**准确性**的要求反而没那么大（其跨越性提升多归功于LLM本身能力的提升）。然而，其流行决定了其迭代极快，可能并不一定适合于本科生阶段科研，这给我带来不少困扰。

## 2. 代码经验💻
在复现实验和交流过程中，我收获了许多实用的代码经验。包括且不限于：

- conda 环境管理
- 利用 huggingface 平台，查找并下载模型及数据集资源
- vllm 本地模型部署
- 管理服务器进程，在 GPU 上完成实验
- 使用 git & github 管理代码
- 通过修改及添加系统路径，解决 import 异常问题

## 3. 技能学习🌏
正常完成任务之余，我也自主学习了不少实用技能，仅列举几项我个人认为颇具价值的技能。如：

- 熟练制作 PPT
- Zotero 文献管理，论文检索
- 经典的 GPT、Llama 框架学习
- Pytorch 框架初步入门
- Manim 科学动画入门
- Kaggle 数据科学基本知识学习
- Neo4j 图形数据库

虽然最直接的感受就是**PPT制作更加熟练**和**英文文献阅读速度大幅提升**，但是在与师兄们的交流和亲身的经历中，也让我认识问题和思考解决方案的方式有了较大的改变。

# 总结
这是一段十分难忘且宝贵的经历，给我后续的本科学业学习及课外科研工作注入了完全不同的新鲜活力。  
最后，再次感谢徐宏力老师给我这个宝贵的学习机会，学生马斌感激不尽！  

**关键词：RAG, Graph-based, AI agent.**

---
## 版权声明
本仓库（Suzhou_bma）内所有文件（包括但不限于代码、文档、图片、数据等）的版权归作者 **马斌** 所有。

**未经作者书面允许，任何个人或组织不得擅自使用、复制、修改、分发、传播本仓库中的任何内容，包括但不限于用于商业用途、二次开发、公开传播等行为。**

*Last updated: 2025.10.28*

</div>

<!-- 英文内容 -->
<a id="en"></a>
<div class="tabcontent">

# Study Summary 💡💡💡

Author: Ma Bin

This document summarizes my summer learning and research stay at the Suzhou Advanced Research Institute (USTC Suzhou) during 2025-07-12 to 2025-08-24, arranged by Professor Xu Hongli. I received generous guidance and help from senior students and teachers (many thanks to Xu Zihui, Yao Zhiwei, Xu Hongli, and others). This README records the outcomes, reflections, and technical materials I collected during the period.

## Overview
My main takeaways can be grouped into three areas:
1. Paper research 📑
2. Coding experience 💻
3. Practical skills 🌏

## 1. Paper Research 📑
My literature review focused on AI Agents and related infra; key directions are listed below.

### AIOS
Following the thread "LLM as OS, Agents as Apps"<sup>[1-Report](Report_PPTs/1-Report-20250712.pdf)</sup>, I explored the overall architecture of AIOS and its major components. I distilled a development roadmap from related papers<sup>[2-Report](Report_PPTs/2-Report-20250716.pdf)</sup> and performed a detailed architecture analysis<sup>[4-Report](Report_PPTs/4-Report-20250718.pdf)</sup>.

### Graph and Agent
Guided by the survey "Graphs Meet AI Agents"<sup>[5-Report](Report_PPTs/5-Report-20250721-图基智能体综述.pdf)</sup>, I investigated intersections between graph techniques and agent systems. I attempted a first paper reproduction on a topology-optimized multi-agent communication framework. The reproduction was challenging but highly educational thanks to mentorship from senior colleagues. I implemented two variants for experiments:
- Lab1 — Using OpenAI API for large model access<sup>[6-Report](Report_PPTs/6-Report-20250723.pdf)</sup>
- Lab2 — Using a local Qwen small model via vLLM<sup>[7-Report](Report_PPTs/7-Report-20250725.pdf)</sup>

I further explored tooling and memory organization directions in Graph-Learning Agents<sup>[8-Report](Report_PPTs/8-Report-20250801-工具调用方向.pdf)</sup><sup>[9-Report](Report_PPTs/9-Report-20250804-记忆组织方向.pdf)</sup>, and identified GraphRAG as a promising research area.

### GraphRAG
Retrieval-Augmented Generation (RAG) is widely adopted; Graph-based RAG (GraphRAG) is an emerging cross-disciplinary area. I summarized recent works<sup>[10-Report](Report_PPTs/10-Report-20250810-GraphRAG.pdf)</sup> and, after discussion with senior colleagues, conducted further investigation into causal-graph-based RAG (CausalRAG)<sup>[11-Report](Report_PPTs/11-Report-20250813-RAGandcCausalRAG.pdf)</sup>. This subfield appears under-explored and promising for future research.

### Summary of findings
After returning from Suzhou I compiled my insights and reflections. Both AI infrastructure and RAG are highly practical layers for applying large models: infrastructure optimizes hardware utilization, while RAG optimizes software-level retrieval and personalization. Their popularity stems from cost reduction and personalization benefits in many applied scenarios; improvements on accuracy often rely on advances in LLMs themselves. Because these areas evolve quickly, they can be fast-moving for undergraduate research, which I found challenging.

## 2. Coding Experience 💻
During experiments and reproduction attempts, I gained practical coding experience, including but not limited to:
- Managing Conda environments
- Using Hugging Face to find and download models/datasets
- Deploying local models with vLLM
- Managing server processes and running experiments on GPU
- Using Git & GitHub for code management
- Fixing import problems by adjusting system paths

## 3. Skills Learned 🌏
Beyond the required tasks, I self-studied several useful skills. Notable items include:
- Creating PPTs effectively
- Zotero for literature management and paper search
- Understanding classic GPT and Llama frameworks
- Getting started with PyTorch
- Intro to Manim for scientific animations
- Basic Kaggle/data-science knowledge
- Neo4j graph database basics

The most immediate gains were improved PPT skills and faster reading of English papers. Conversations with senior researchers also changed how I approach problems and design solutions.

## Conclusion
This was a memorable and valuable experience that energized my following undergraduate studies and extracurricular research. I’m grateful to Professor Xu Hongli for this opportunity.

Keywords: RAG, Graph-based, AI agent.

---
## Copyright
All files in this repository (including but not limited to code, documentation, images, and data) are copyrighted by the author, Ma Bin.

No part of this repository may be used, copied, modified, distributed, or otherwise shared by any individual or organization for commercial use, derivative works, or public distribution without the express written permission of the author.

*Last updated: 2025.10.28*

</div>


