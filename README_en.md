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
