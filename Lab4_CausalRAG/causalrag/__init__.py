# __init__.py
# 公开顶级导入以供外部使用

"""
CausalRAG: Causal Graph Enhanced Retrieval-Augmented Generation

这个包将因果推理与检索增强生成相结合，通过考虑概念之间的因果关系来提高生成答案的质量和准确性。
"""

__version__ = "0.1.0"
__author__ = "CausalRAG Team"
import sys
sys.path.append("/data0/bma/LAB_CausalRAG/causalrag/evaluation")

# Core components
from .pipeline import CausalRAGPipeline
from .causal_graph.builder import CausalGraphBuilder
from .causal_graph.retriever import CausalPathRetriever
from evaluator import EvaluationResult
from evaluator import CausalEvaluator

# Set default logging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Convenience function to create and configure a pipeline
def create_pipeline(
    model_name="gpt-4", 
    embedding_model="all-MiniLM-L6-v2",
    graph_path=None, 
    index_path=None,
    config_path=None
):
    """
    Create and configure a CausalRAG pipeline
    
    Args:
        model_name: Name of LLM model to use
        embedding_model: Name of embedding model for vector store
        graph_path: Optional path to pre-built causal graph
        index_path: Optional path to pre-built vector index
        config_path: Optional path to pipeline configuration
        
    Returns:
        Configured CausalRAGPipeline instance
    """
    return CausalRAGPipeline(
        model_name=model_name,
        embedding_model=embedding_model,
        graph_path=graph_path,
        index_path=index_path,
        config_path=config_path
    )

# Define what's available via import *
__all__ = [
    'CausalRAGPipeline',
    'CausalGraphBuilder',
    'CausalPathRetriever',
    'CausalEvaluator',
    'EvaluationResult',
    'create_pipeline',
]