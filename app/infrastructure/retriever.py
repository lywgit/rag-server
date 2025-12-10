from abc import ABC, abstractmethod
from numpy.typing import NDArray
from infrastructure.repositories.vector_store_interface import VectorStoreInterface
from infrastructure.repositories.keyword_store_interface import KeywordStoreInterface
from infrastructure.embeddings.embedder import EmbedderInterface
from domain.models import SearchResult, Document

class RetrieverInterface(ABC):
    @abstractmethod
    def build(self, documents:list[Document]) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def semantic_search(self, query: str, top_k: int) -> list[SearchResult]:
        raise NotImplementedError
    
    @abstractmethod
    def keyword_search(self, query: str, top_k: int) -> list[SearchResult]:
        raise NotImplementedError
    
    @abstractmethod
    def hybrid_search(self, query: str, top_k: int, alpha: float) -> list[SearchResult]:
        raise NotImplementedError
    


class FileRetriever(RetrieverInterface):
    def __init__(
            self, 
            embedder:EmbedderInterface, 
            vector_store: VectorStoreInterface,
            keyword_store: KeywordStoreInterface):
        self.embedder = embedder
        self.vector_store = vector_store  
        self.keyword_store = keyword_store 
    
    def build(self, documents):
        # build keyword index
        self.keyword_store.build_index(documents)
        # build vector index (not chunked)
        for doc in documents:
            embedding = self.embedder.embed(doc.content)
            self.vector_store.add(id=doc.id, vector=embedding, doc=doc)
        print(f"FileRetriever build: {len(self.vector_store)} documents")
            

    def semantic_search(self, query: str, top_k: int)  -> list[SearchResult]:
        query_vector = self.embedder.embed(query)
        return self.vector_store.search(query_vector, top_k)
    
    def keyword_search(self, query: str, top_k: int)  -> list[SearchResult]:
        return self.keyword_store.search(query, top_k)
    
    def hybrid_search(self, query: str, top_k: int, alpha: float) -> list[SearchResult]:
        semantic = self.semantic_search(query, top_k)
        keyword = self.keyword_search(query, top_k)
        return self._merge_results(semantic, keyword, alpha)
    
    def _merge_results(self, semantic_res, keyword_res, alpha) -> list[SearchResult]:
        raise NotImplementedError