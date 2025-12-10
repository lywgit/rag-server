from abc import ABC, abstractmethod
from numpy.typing import NDArray
from domain.models import Document,SearchResult

class VectorStoreInterface(ABC):
    # Core retrieval functions
    @abstractmethod
    def search(self, query_vector:NDArray, limit:int) -> list[SearchResult]:
        raise NotImplementedError

    @abstractmethod
    def retrieve_by_id(self, id) -> Document:
        raise NotImplementedError

    # CRUD operation (not used)
    @abstractmethod
    def add(self, id:str, vector:NDArray, doc:Document) -> str:
        """return id"""
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    # def update(self, id:str, vector:str, metadata:dict):
    #     raise NotImplementedError
        
    # def delete(self, id:str) -> None:
    #     raise NotImplementedError

    # File-based vector store methods
    @abstractmethod
    def save_index(self):
        raise NotImplementedError
    
    @abstractmethod
    def load_index(self):
        raise NotImplementedError
