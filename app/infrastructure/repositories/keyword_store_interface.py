from abc import ABC, abstractmethod
from app.domain.models import Document, SearchResult

class TokenizerInterface(ABC):
    @abstractmethod
    def tokenize(self, text:str) -> list[str]:
        raise NotImplementedError


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

