__all__ = ["MoveTask"]

from typing import Optional

import hydra
from torch import nn

from move.conf.models import ModelConfig
from move.conf.training import (
    DataLoaderConfig,
    TestDataLoaderConfig,
    TrainingDataLoaderConfig,
    TrainingLoopConfig,
)
from move.core.exceptions import UnsetProperty
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
        model_config: Optional[ModelConfig],
        training_dataloader_config: DataLoaderConfig,  # TODO: make optional too
        training_loop_config: Optional[TrainingLoopConfig],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.discrete_dataset_names = discrete_dataset_names
        self.continuous_dataset_names = continuous_dataset_names
        self.model_config = model_config
        self.training_dataloader_config = training_dataloader_config
        self.training_loop_config = training_loop_config

    def make_dataloader(self, split: Split = "all") -> MoveDataLoader:
        """Make a MOVE dataloader. For the training split, data will be shuffled
        and the last batch will be dropped."""
        dataset = MoveDataset.load(
            self.input_dir,
            self.discrete_dataset_names,
            self.continuous_dataset_names,
            split,
        )
        config = self.training_dataloader_config
        batch_size = config.batch_size
        if split == "test" or split == "valid":
            # Duplicate config, but set shuffle/drop_last to False
            config = TestDataLoaderConfig(batch_size)
        else:
            # Duplicate config, but ensure shuffle/drop_last to True
            config = TrainingDataLoaderConfig(batch_size)
        return hydra.utils.instantiate(config, dataset=dataset)

    def init_model(self, dataloader: MoveDataLoader) -> BaseVae:
        """Initialize a MOVE model."""
        if self.model_config is None:
            raise UnsetProperty("Model config")
        return hydra.utils.instantiate(
            self.model_config,
            discrete_shapes=dataloader.dataset.discrete_shapes,
            continuous_shapes=dataloader.dataset.continuous_shapes,
        )

    def init_training_loop(self, set_parent: bool = True) -> TrainingLoop:
        """Initialize a training loop.

        Args:
            set_parent:
                Whether the training task is linked to a parent task. If
                orphaned, this task cannot use the logger to track its progress.
        """
        if self.training_loop_config is None:
            raise UnsetProperty("Training loop config")
        training_loop: TrainingLoop = hydra.utils.instantiate(
            self.training_loop_config, _recursive_=False
        )
        if set_parent:
            training_loop.parent = self
        else:
            # if orphan, cannot use logger
            training_loop.prog_every_n_epoch = None
        return training_loop
