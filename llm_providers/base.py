# llm_providers/base.py
from abc import ABC, abstractmethod
from typing import List, Generator

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass

    @abstractmethod
    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        pass

    @abstractmethod
    def embed(self, text: str, **kwargs) -> List[float]:
        pass