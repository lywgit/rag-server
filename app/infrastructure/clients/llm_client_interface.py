
from abc import ABC, abstractmethod

class LLMClientInterface(ABC):
    @abstractmethod
    def hello(self) -> dict[str,str]:
        pass
    @abstractmethod
    def generate(self, prompt:str, history:list|None = None) -> str:
        pass

class AsyncLLMClientInterface(ABC):
    @abstractmethod
    async def hello(self) -> dict[str,str]:
        pass
    @abstractmethod
    async def generate(self, prompt:str, history:list|None = None) -> dict[str,str]:
        pass
