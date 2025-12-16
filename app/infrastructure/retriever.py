import logging
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from collections import defaultdict

from app.infrastructure.repositories.vector_store_interface import VectorStoreInterface
from app.infrastructure.repositories.keyword_store_interface import KeywordStoreInterface
from app.infrastructure.embeddings.embedder import EmbedderInterface
from app.domain.models import SearchResult, Document
from app.core.config import HYBRID_RRF_K

logger = logging.getLogger(__name__)

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
    def hybrid_search(self, query: str, top_k: int) -> list[SearchResult]:
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
        logger.info(f"FileRetriever build complete: {len(self.vector_store)} documents indexed.")
            

    def semantic_search(self, query: str, top_k: int)  -> list[SearchResult]:
        logger.debug(f'Semantic search for query: "{query}" with top_k={top_k}')
        query_vector = self.embedder.embed(query)
        logger.debug(f'Semantic search for query embedding dim: {query_vector.shape}')
        return self.vector_store.search(query_vector, top_k)
    
    def keyword_search(self, query: str, top_k: int)  -> list[SearchResult]:
        return self.keyword_store.search(query, top_k)
    
    def hybrid_search(self, query: str, top_k: int) -> list[SearchResult]:
        extended_top_k = top_k * 5
        semantic_res = self.semantic_search(query, extended_top_k)
        keyword_res = self.keyword_search(query, extended_top_k)
        merged_results = self._rrf_merge_results(keyword_res, semantic_res)
        return merged_results[:top_k]

    def _rrf_merge_results(self, keyword_res:list[SearchResult], semantic_res:list[SearchResult]) -> list[SearchResult]:
        # Use reciprocal rank fusion (RRF)
        def rrf_score(rank:int, k:float = HYBRID_RRF_K):
            return 1 / (k + rank)
        
        doc_score_dic = defaultdict(float)
        doc_map = dict()  # doc_id -> Document

        for res in keyword_res:
            doc_id = res.document.id
            rank = res.rank
            doc_score_dic[doc_id] += rrf_score(rank)
            doc_map[doc_id] = res.document

        for res in semantic_res:
            rank = res.rank
            doc_id = res.document.id   
            doc_score_dic[doc_id] += rrf_score(rank)
            doc_map[doc_id] = res.document

        doc_scores:list[tuple] = [(doc_id, score) for doc_id, score in doc_score_dic.items()] # (doc_id, score)
        doc_scores.sort(key=lambda x:x[1], reverse=True)
        result = []
        for i, (doc_id, score) in enumerate(doc_scores):
            result.append(SearchResult(document=doc_map[doc_id], score=score, rank=i+1))
        return result

