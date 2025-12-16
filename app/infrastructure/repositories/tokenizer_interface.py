from abc import ABC, abstractmethod

class TokenizerInterface(ABC):
    @abstractmethod
    def tokenize(self, text:str) -> list[str]:
        raise NotImplementedError
