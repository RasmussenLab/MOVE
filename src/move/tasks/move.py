__all__ = ["MoveTask"]

from typing import Optional

import hydra
from torch import nn

from move.conf.models import ModelConfig
from move.conf.training import (
    DataLoaderConfig,
    TestDataLoaderConfig,
    TrainingLoopConfig,
)
from move.core.typing import Split
from move.data.dataloader import MoveDataLoader
from move.data.dataset import MoveDataset
from move.models.base import BaseVae
from move.tasks.base import ParentTask
from move.training.loop import TrainingLoop


class MoveTask(ParentTask):
    """A task that can initialize a MOVE model, dataloader, and training loop."""

    def __init__(
        self,
        discrete_dataset_names: list[str],
        continuous_dataset_names: list[str],
        model_config: ModelConfig,
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

    def make_dataloader(self, split: Split = "train") -> MoveDataLoader:
        """Make a MOVE dataloader. For the training split, data will be shuffled
        and the last batch will be dropped."""
        dataset = MoveDataset.load(
            self.input_dir, self.discrete_dataset_names, self.continuous_dataset_names
        )
        config = self.training_dataloader_config
        if split == "test":
            # Duplicate config, but set shuffle/drop_last to False
            batch_size = getattr(config, "batch_size")
            config = TestDataLoaderConfig(batch_size)
        return hydra.utils.instantiate(config, dataset=dataset)

    def init_model(self, dataloader: MoveDataLoader) -> BaseVae:
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
