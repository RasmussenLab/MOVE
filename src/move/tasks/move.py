__all__ = ["MoveTask"]

from typing import TYPE_CHECKING, Optional

import hydra
from torch import nn

from move.core.exceptions import UnsetProperty
from move.core.typing import Split
from move.data.dataloader import MoveDataLoader
from move.data.dataset import MoveDataset
from move.models.base import BaseVae
from move.tasks.base import ParentTask

if TYPE_CHECKING:
    from move.conf.models import ModelConfig
    from move.conf.training import TrainingLoopConfig
    from move.training.loop import TrainingLoop


class MoveTask(ParentTask):
    """A task that can initialize a MOVE model, dataloader, and training loop."""

    def __init__(
        self,
        discrete_dataset_names: list[str],
        continuous_dataset_names: list[str],
        batch_size: int,
        model_config: Optional["ModelConfig"],
        training_loop_config: Optional["TrainingLoopConfig"],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.discrete_dataset_names = discrete_dataset_names
        self.continuous_dataset_names = continuous_dataset_names
        self.model_config = model_config
        self.batch_size = batch_size
        self.training_loop_config = training_loop_config

    def make_dataloader(
        self, split: Split = "all", **dataloader_kwargs
    ) -> MoveDataLoader:
        """Make a MOVE dataloader. For the training split, data will be shuffled
        and the last batch will be dropped."""
        from move.conf.training import DataLoaderConfig

        dataset = MoveDataset.load(
            self.input_dir,
            self.discrete_dataset_names,
            self.continuous_dataset_names,
            split,
        )

        is_training = not (split == "test" or split == "valid")
        if "shuffle" not in dataloader_kwargs:
            dataloader_kwargs["shuffle"] = is_training

        if "drop_last" not in dataloader_kwargs:
            dataloader_kwargs["drop_last"] = is_training

        dataloader_kwargs["batch_size"] = self.batch_size
        config = DataLoaderConfig(**dataloader_kwargs)
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

    def init_training_loop(self, set_parent: bool = True) -> "TrainingLoop":
        """Initialize a training loop.

        Args:
            set_parent:
                Whether the training task is linked to a parent task. If
                orphaned, this task cannot use the logger to track its progress.
        """
        if self.training_loop_config is None:
            raise UnsetProperty("Training loop config")
        training_loop: "TrainingLoop" = hydra.utils.instantiate(
            self.training_loop_config, _recursive_=False
        )
        if set_parent:
            training_loop.parent = self
        else:
            # if orphan, cannot use logger
            training_loop.prog_every_n_epoch = None
        return training_loop
