__all__ = ["Task", "ParentTask", "SubTask", "CsvWriterMixin"]

import inspect
import logging
from abc import ABC, abstractmethod
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Optional, Sequence, Type, TypeVar, Union, cast

import hydra
from numpy.typing import NDArray
from omegaconf import OmegaConf, DictConfig

from move.core.qualname import get_fully_qualname
from move.core.exceptions import FILE_EXISTS_WARNING, UnsetProperty
from move.core.logging import get_logger
from move.core.typing import LoggingLevel, PathLike
from move.data.writer import CsvWriter


class InputDirMixin:
    """Mixin class for adding an input directory property to a class."""

    @property
    def input_dir(self) -> Path:
        if path := getattr(self, "_input_dir", None):
            return path
        raise UnsetProperty("Input directory")

    @input_dir.setter
    def input_dir(self, pathlike: PathLike) -> None:
        self._input_dir = Path(pathlike)
        self._input_dir.mkdir(parents=True, exist_ok=True)


class OutputDirMixin:
    """Mixin class for adding an output directory property to a class."""

    @property
    def output_dir(self) -> Path:
        if path := getattr(self, "_output_dir", None):
            return path
        raise UnsetProperty("Output directory")

    @output_dir.setter
    def output_dir(self, pathlike: PathLike) -> None:
        self._output_dir = Path(pathlike)
        self._output_dir.mkdir(parents=True, exist_ok=True)


class LoggerMixin:
    """Mixin class for logging."""

    @property
    def logger(self) -> logging.Logger:
        if issubclass(self.__class__, SubTaskMixin):
            task = cast(SubTaskMixin, self).parent
            if task:
                return task.logger
            raise UnsetProperty("Parent task")
        if getattr(self, "_logger", None) is None:
            self._logger = get_logger(self.__class__.__name__)
        return self._logger

    def log(self, message: str, level: LoggingLevel = "INFO") -> None:
        """Log a message.

        Args:
            message: logged message
            level: predefined logging level name or numeric value."""
        if isinstance(level, str):
            level_num = logging.getLevelName(level)
            if not isinstance(level_num, int):
                raise ValueError(f"Unexpected logging level: {level}")
        else:
            level_num = level
        self.logger.log(level_num, message)


T = TypeVar("T", bound="Task")


class Task(ABC, LoggerMixin):
    """Base class for a task"""

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        raise NotImplementedError()

    @classmethod
    def from_config(cls: Type[T], config: DictConfig) -> T:
        """Instantiate a task from its config file."""
        if not hasattr(config, "task"):
            raise UnsetProperty("Task configuration")
        target_qualname = config.task._target_
        current_qualname = get_fully_qualname(cls)
        if target_qualname == current_qualname:
            return hydra.utils.instantiate(config.task, _recursive_=False)
        raise ValueError(
            f"Received config for `{target_qualname}`, but current class is `{current_qualname}`"
        )

    def to_yaml(self, filepath: PathLike) -> None:
        """Save task config as YAML file."""
        signature = inspect.signature(self.__class__).parameters.keys()
        config = OmegaConf.create({name: getattr(self, name) for name in signature})
        with open(filepath, "w") as file:
            file.write(OmegaConf.to_yaml(config))


class ParentTask(InputDirMixin, OutputDirMixin, Task):
    """A simple task with an input and output directory. This task may have
    children (sub-tasks).

    Args:
        input_path: Path where input files are read from
        output_path: Path where output files will be saved to
    """

    def __init__(self, input_dir: PathLike, output_dir: PathLike) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir


class OutputDir(ParentTask):
    """Task used to set an output directory."""

    def __init__(self, output_dir: PathLike) -> None:
        super().__init__(Path.cwd(), output_dir)

    def run(self) -> None:
        raise NotImplementedError()


class TestTask(Task):
    """Task used for testing"""

    def run(self) -> Any:
        pass


class SubTaskMixin(LoggerMixin):
    """Mixin class to designate a task is child of another task."""

    @property
    def parent(self) -> Optional[ParentTask]:
        return getattr(self, "_parent", None)

    @parent.setter
    def parent(self, task: ParentTask) -> None:
        self._parent = task


class SubTask(SubTaskMixin, Task):
    """Base class for sub-tasks."""

    ...


CsvRow = dict[str, Any]


class CsvWriterMixin(LoggerMixin):
    """Mixin class to designate a sub-task that has its own CSV writer."""

    csv_filepath: Optional[Path] = None
    buffer_size = 1000

    @property
    def can_write(self) -> bool:
        return (
            getattr(self, "_csv_writer", None) is not None
            and getattr(self, "_csv_file", None) is not None
            and not self.csv_file.closed
        )

    @property
    def csv_file(self) -> TextIOWrapper:
        return getattr(self, "_csv_file")

    @csv_file.setter
    def csv_file(self, value: Optional[TextIOWrapper]) -> None:
        self._csv_file = value

    @property
    def csv_writer(self) -> CsvWriter:
        return getattr(self, "_csv_writer")

    @csv_writer.setter
    def csv_writer(self, value: Optional[CsvWriter]) -> None:
        self._csv_writer = value

    @property
    def row_buffer(self) -> list[CsvRow]:
        if getattr(self, "_buffer", None) is None:
            self._buffer: list[CsvRow] = []
        return self._buffer

    @property
    def output_dir(self) -> Path:
        if self.parent is None:
            raise UnsetProperty("Output directory")
        return self.parent.output_dir

    @output_dir.setter
    def output_dir(self, value: PathLike) -> None:
        self.parent = OutputDir(value)

    def init_csv_writer(self, filepath: Path, mode: str = "w", **writer_kwargs) -> None:
        """Initialize the CSV writer."""
        self.csv_filepath = filepath
        exists = self.csv_filepath.exists()
        if exists and mode == "w":  # Warn about overwriting
            self.log(FILE_EXISTS_WARNING.format(self.csv_filepath))
        self.csv_file = open(self.csv_filepath, mode, newline="")  # type: ignore
        self.csv_writer = CsvWriter(self.csv_file, **writer_kwargs)
        # Do not write header if file exists and appending
        if (not exists) or mode != "a":
            self.csv_writer.writeheader()

    def write_cols(self, cols: dict[str, Union[Sequence[Any], NDArray]]) -> None:
        """Directly write columns to CSV file.

        Args:
            cols: Column name to values dictionary."""
        if self.can_write:
            self.csv_writer.writecols(cols)

    def write_row(self, row: list[Any]) -> None:
        """Directly write a row to CSV file."""
        if self.can_write:
            self.csv_writer.writer.writerow(row)

    def add_row_to_buffer(self, csv_row: CsvRow) -> None:
        """Add row to buffer and flush buffer if it has reached its limit.

        Args:
            csv_row: Header names to values dictionary, representing a row
        """
        if self.can_write:
            self.row_buffer.append(csv_row)
            # Flush
            if len(self.row_buffer) >= self.buffer_size:
                self.csv_writer.writerows(self.row_buffer)
                self.row_buffer.clear()

    def close_csv_writer(self, clear: bool = False) -> None:
        """Close file and flush buffer.

        Args:
            clear: whether to nullify the writer and file object"""
        if self.can_write:
            if len(self.row_buffer) > 0:
                self.csv_writer.writerows(self.row_buffer)
                self.row_buffer.clear()
            self.csv_file.close()
        if clear:
            self.csv_file = None
            self.csv_filepath = None
            self.csv_writer = None
