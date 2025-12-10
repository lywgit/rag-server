from abc import ABC, abstractmethod
from domain.models import Document, SearchResult

class KeywordStoreInterface(ABC):
    # Core retrieval functions
    @abstractmethod
    def search(self, query:str, limit:int) -> list[SearchResult]:
        raise NotImplementedError

    @abstractmethod
    def retrieve_by_id(self, id) -> Document:
        raise NotImplementedError

    @abstractmethod
    def build_index(self, documents:list[Document]):
        raise NotImplementedError
    
    # persistent methods
    @abstractmethod
    def save_index(self):
        raise NotImplementedError
    
    @abstractmethod
    def load_index(self):
        raise NotImplementedError


class BM25Store(KeywordStoreInterface):
    def __init__(self) -> None:
        pass
        

    def build_index(self, documents:list[Document]):
        pass

    def search(self, query:str, limit:int) -> list[SearchResult]:
        raise NotImplementedError


    def retrieve_by_id(self, id) -> Document:
        raise NotImplementedError
    

    def save_index(self):
        raise NotImplementedError
    
    def load_index(self):
        raise NotImplementedError
