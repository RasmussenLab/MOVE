__all__ = ["Task", "SubTask"]

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from move.core.logging import get_logger


class Task(ABC):
    """A task with its own logger, input and output directory

    Args:
        input_path: Path where input files are read from
        output_path: Path where output files will be saved to
    """

    def __init__(self, input_path: Path, output_path: Path) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.input_path.mkdir(exist_ok=True, parents=True)
        self.output_path.mkdir(exist_ok=True, parents=True)
        self.logger = get_logger(__name__)

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        raise NotImplementedError()


class TestTask(Task):
    """Task used for testing"""

    def run(self) -> Any:
        pass


class SubTask(ABC):
    """A task that is the child of another task"""

    parent: Optional[Task] = None

    def set_parent(self, parent: Task) -> None:
        self.parent = parent
