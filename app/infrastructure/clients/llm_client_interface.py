
from abc import ABC, abstractmethod

class LLMClientInterface(ABC):
    @abstractmethod
    def hello(self) -> dict:
        pass
    @abstractmethod
    def generate(self, prompt:str, history:list|None = None) -> dict:
        pass

class AsyncLLMClientInterface(ABC):
    @abstractmethod
    async def hello(self) -> dict:
        pass
    @abstractmethod
    async def generate(self, prompt:str, history:list|None = None) -> dict:
        pass
