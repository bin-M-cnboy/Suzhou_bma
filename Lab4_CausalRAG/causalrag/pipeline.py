# pipeline.py
# Top-level orchestration of CausalRAG pipeline

# # Added 
# import sys
# sys.path.append("/data0/bma/LAB_CausalRAG/causalrag")

from causalrag.causal_graph.builder import CausalGraphBuilder
from causalrag.causal_graph.retriever import CausalPathRetriever
from causalrag.reranker.causal_path import CausalPathReranker
from causalrag.retriever.vector_store import VectorStoreRetriever
from causalrag.retriever.hybrid import HybridRetriever
from causalrag.generator.prompt_builder import build_prompt
from causalrag.generator.llm_interface import LLMInterface

from sentence_transformers import SentenceTransformer   # new added

class CausalRAGPipeline:
    def __init__(self, 
                model_name="Qwen3-8B",#"gpt-4", 
                embedding_model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'),# "all-MiniLM-L6-v2",
                # graph_path=None, 
                index_path=None,
                config_path=None):
        """
        Initialize the CausalRAG pipeline with configurable components
        
        Args:
            model_name: Name of LLM model to use
            embedding_model: Name of embedding model for vector store
            graph_path: Optional path to pre-built causal graph
            index_path: Optional path to pre-built vector index
            config_path: Optional path to pipeline configuration
        """
        # Core components
        self.graph_builder = CausalGraphBuilder()#(graph_path=graph_path)
        self.vector_retriever = VectorStoreRetriever(
            # embedding_model=embedding_model, 
            # index_path=index_path
        )
        self.graph_retriever = CausalPathRetriever(self.graph_builder)
        self.hybrid_retriever = HybridRetriever(self.vector_retriever, self.graph_retriever)
        self.reranker = CausalPathReranker(self.graph_retriever)
        self.llm = LLMInterface()#(model_name=model_name)
        
        # Load configuration if provided
        if config_path:
            self._load_config(config_path)

    def _load_config(self, config_path):
        """Load configuration from file"""
        # Implementation for loading config
        pass

    def index(self, documents):
        """Build graph + vector index from documents"""
        self.graph_builder.index_documents(documents)
        self.vector_retriever.index_corpus(documents)

    def run(self, query: str, top_k: int = 5) -> str:
        """Query → Retrieval → Rerank → Prompt → Generate"""
        # Step 1: Hybrid retrieval
        candidates = self.hybrid_retriever.retrieve(query, top_k=top_k)

        # Step 2: Rerank via causal path
        reranked = self.reranker.rerank(query, candidates)

        # Step 3: Build prompt with causal context
        # causal_nodes = self.graph_retriever.retrieve_path_nodes(query)
        prompt = build_prompt(query, reranked[:top_k])#, causal_path=causal_nodes)

        # Step 4: Generate answer
        return self.llm.generate(prompt)
