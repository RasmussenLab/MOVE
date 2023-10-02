__all__ = ["Task"]

from abc import ABC, abstractmethod
from typing import Any

class Task(ABC):
    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        raise NotImplementedError()
