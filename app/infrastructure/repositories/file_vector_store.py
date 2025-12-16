
import logging
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

from app.infrastructure.repositories.vector_store_interface import VectorStoreInterface
from app.core.config import FILE_CACHE_DIR
from app.domain.models import Document, SearchResult

logger = logging.getLogger(__name__)

# Expect future extension to use vector DB through VectorStoreInterface
# Here a simple file-based vector store is enough for current purpose

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

class FileVectorStore(VectorStoreInterface):
    """json document + numpy vector"""
    def __init__(self, base_path:str = FILE_CACHE_DIR):
        self.base_path = Path(base_path)
        self.doc_map = dict() # id->doc
        self.embedding_map = dict() # id->vector
    
    def add(self, id:str, vector:NDArray, doc:Document) -> str:
        """return ID"""
        self.doc_map[id] = doc
        self.embedding_map[id] = vector
        return id

    def search(self, query_vector:NDArray, limit:int) -> list[SearchResult]:
        logger.debug(f'FileVectorStore search called with query vector dim: {query_vector.shape} and limit: {limit}')
        score_list = []
        for id, doc_vector in self.embedding_map.items():
            score = cosine_similarity(query_vector, doc_vector)
            score_list.append((id, score))
        score_list.sort(key=lambda x: x[1], reverse=True)
        res = []
        for i, (id, score) in enumerate(score_list[:limit]):
            rank = i + 1
            doc = self.doc_map[id]
            res.append(SearchResult(document=doc, score=score, rank=rank))
        logger.debug(f'FileVectorStore search returning {len(res)} results')
        return res 

    def __len__(self) -> int:
        return len(self.doc_map)

    def retrieve_by_id(self, id) -> Document:
        raise

    def load_index(self):
        raise

    def save_index(self):
        raise
