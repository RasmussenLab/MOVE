__all__ = ["Task", "SubTask"]

import csv
import logging
from abc import ABC, abstractmethod
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Optional, Union

import hydra
from torch import nn

from move.conf.schema import VAEConfig, DataLoaderConfig, TrainingLoopConfig
from move.core.typing import LoggingLevel
from move.core.logging import get_logger
from move.data.dataloader import MoveDataLoader
from move.data.dataset import MoveDataset
from move.tasks.writer import CsvWriter
from move.training.loop import TrainingLoop


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


class MoveTask(Task):
    """Task that can initialize a MOVE model, dataloader, and training loop"""

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
            self.input_path, self.discrete_dataset_names, self.continuous_dataset_names
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
        training_loop.set_parent(self)
        return training_loop


class TestTask(Task):
    """Task used for testing"""

    def run(self) -> Any:
        pass


class SubTask(ABC):
    """A task that is the child of another task"""

    parent: Optional[Task] = None

    def log(self, message: str, level: LoggingLevel = "INFO") -> None:
        """Log a message using the parent task's logger.

        Args:
            message: logged message
            level: predefined logging level name or numeric value."""
        if self.parent is None:
            return
        if isinstance(level, str):
            level_num = logging.getLevelName(level)
            if not isinstance(level_num, int):
                raise ValueError(f"Unexpected logging level: {level}")
        else:
            level_num = level
        self.parent.logger.log(level_num, message)

    def set_parent(self, parent: Task) -> None:
        self.parent = parent


class SubTaskWritesCsv(SubTask):
    """A sub-task that has a CSV writer."""

    csv_file: Optional[TextIOWrapper]
    csv_filepath: Optional[Path]
    csv_writer: Optional[CsvWriter]
    buffer: list[dict[str, float]]
    buffer_size: int = 1000

    def __init__(self) -> None:
        self.csv_filepath = None
        self.csv_file = None
        self.csv_writer = None
        self.buffer = []

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

    def add_row_to_buffer(self, csv_row: dict[str, float]) -> None:
        """Add row to buffer and flush buffer if it has reached its limit.

        Args:
            csv_row: Header names to values dictionary, representing a row
        """
        if self.csv_file and self.csv_writer:
            self.buffer.append(csv_row)
            # Flush
            if len(self.buffer) >= self.buffer_size:
                self.csv_writer.writerows(self.buffer)
                self.buffer.clear()

    def close_csv_writer(self) -> None:
        """Close file and flush"""
        if self.csv_file and self.csv_writer:
            if self.buffer:
                self.csv_writer.writerows(self.buffer)
                self.buffer.clear()
            self.csv_file.close()
