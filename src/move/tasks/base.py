__all__ = ["Task", "MoveTask", "SubTaskMixin", "CsvWriterMixin"]

import csv
import logging
from abc import ABC, abstractmethod
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Optional, cast

import hydra
from torch import nn

from move.conf.schema import DataLoaderConfig, TrainingLoopConfig, VAEConfig
from move.core.exceptions import UnsetProperty
from move.core.logging import get_logger
from move.core.typing import LoggingLevel, PathLike
from move.data.dataloader import MoveDataLoader
from move.data.dataset import MoveDataset
from move.tasks.writer import CsvWriter
from move.training.loop import TrainingLoop


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


class Task(ABC):
    """Base class for a task"""

    @property
    def logger(self) -> logging.Logger:
        if issubclass(self.__class__, SubTaskMixin):
            task = cast(SubTaskMixin, self).parent
            if task:
                return task.logger
            raise UnsetProperty("Parent task")
        if getattr(self, "_logger", None) is None:
            self._logger = get_logger(__name__)
        return self._logger

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        raise NotImplementedError()


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


class MoveTask(ParentTask):
    """A task that can initialize a MOVE model, dataloader, and training loop"""

    def __init__(
        self,
        discrete_dataset_names: list[str],
        continuous_dataset_names: list[str],
        model_config: VAEConfig,
        training_dataloader_config: DataLoaderConfig,
        training_loop_config: TrainingLoopConfig,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.discrete_dataset_names = discrete_dataset_names
        self.continuous_dataset_names = continuous_dataset_names
        self.model_config = model_config
        self.training_dataloader_config = training_dataloader_config
        self.training_loop_config = training_loop_config

    def make_dataloader(self, **kwargs) -> MoveDataLoader:
        """Make a MOVE dataloader."""
        dataset = MoveDataset.load(
            self.input_dir, self.discrete_dataset_names, self.continuous_dataset_names
        )
        return hydra.utils.instantiate(
            self.training_dataloader_config, dataset=dataset, **kwargs
        )

    def init_model(self, dataloader: MoveDataLoader):
        """Initialize a MOVE model."""
        return hydra.utils.instantiate(
            self.model_config,
            discrete_shapes=dataloader.dataset.discrete_shapes,
            continuous_shapes=dataloader.dataset.continuous_shapes,
        )

    def init_training_loop(self) -> TrainingLoop:
        """Initialize a training loop."""
        training_loop: TrainingLoop = hydra.utils.instantiate(self.training_loop_config)
        training_loop.parent = self
        return training_loop


class TestTask(Task):
    """Task used for testing"""

    def run(self) -> Any:
        pass


class SubTaskMixin:
    """Mixin class to designate a task is child of another task."""

    @property
    def parent(self) -> Optional[ParentTask]:
        return getattr(self, "_parent", None)

    @parent.setter
    def parent(self, task: ParentTask) -> None:
        self._parent = task

    def log(self, message: str, level: LoggingLevel = "INFO") -> None:
        """Log a message using the parent task's logger.

        Args:
            message: logged message
            level: predefined logging level name or numeric value."""
        if self.parent is not None:
            if isinstance(level, str):
                level_num = logging.getLevelName(level)
                if not isinstance(level_num, int):
                    raise ValueError(f"Unexpected logging level: {level}")
            else:
                level_num = level
            self.parent.logger.log(level_num, message)


CsvRow = dict[str, float]


class CsvWriterMixin(SubTaskMixin):
    """Mixin class to designate a sub-task that has its own CSV writer."""

    csv_file: Optional[TextIOWrapper] = None
    csv_filepath: Optional[Path] = None
    csv_writer: Optional[CsvWriter] = None
    buffer_size = 1000

    @property
    def row_buffer(self) -> list[CsvRow]:
        if getattr(self, "_buffer", None) is None:
            self._buffer = []
        return self._buffer

    def init_csv_writer(self, filepath: Path, **writer_kwargs) -> None:
        """Initialize the CSV writer."""
        self.csv_filepath = filepath
        if self.csv_filepath.exists():
            self.log(f"File '{self.csv_filepath}' already exists. It be overwritten.")
        self.csv_file = open(self.csv_filepath, "w", newline="")
        self.csv_writer = CsvWriter(self.csv_file, **writer_kwargs)
        self.csv_writer.writeheader()

    def write_cols(self, cols: dict[str, list[float]]) -> None:
        """Directly write columns to CSV file.

        Args:
            cols: Column name to values dictionary."""
        if self.csv_file and self.csv_writer:
            self.csv_writer.writecols(cols)

    def add_row_to_buffer(self, csv_row: CsvRow) -> None:
        """Add row to buffer and flush buffer if it has reached its limit.

        Args:
            csv_row: Header names to values dictionary, representing a row
        """
        if self.csv_file and self.csv_writer:
            self.row_buffer.append(csv_row)
            # Flush
            if len(self.row_buffer) >= self.buffer_size:
                self.csv_writer.writerows(self.row_buffer)
                self.row_buffer.clear()

    def close_csv_writer(self) -> None:
        """Close file and flush"""
        if self.csv_file and self.csv_writer:
            if self.row_buffer:
                self.csv_writer.writerows(self.row_buffer)
                self.row_buffer.clear()
            self.csv_file.close()
