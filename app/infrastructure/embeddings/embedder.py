# infrastructure/embeddings/embedder.py
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

class EmbedderInterface(ABC):
    @abstractmethod
    def embed(self, text:str) -> NDArray:
        raise NotImplementedError
    
    def embed_batch(self, texts: list[str]) -> NDArray:
        raise NotImplementedError


class SentenceTransformerEmbedder(EmbedderInterface):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    
    def embed(self, text: str) -> NDArray:
        return self.model.encode([text])[0]
    
    def embed_batch(self, texts: list[str]) -> NDArray:
        return self.model.encode(texts)

