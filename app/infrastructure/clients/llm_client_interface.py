
from abc import ABC, abstractmethod

class LLMClientInterface(ABC):
    @abstractmethod
    def ping(self) -> dict[str,str]:
        pass

    @abstractmethod
    def generate(self, prompt:str, history:list|None = None) -> str:
        pass

