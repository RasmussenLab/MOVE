__all__ = ["TrainModel"]

from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import hydra

from move.core.exceptions import FILE_EXISTS_WARNING
from move.core.typing import PathLike
from move.tasks.base import ParentTask

if TYPE_CHECKING:
    from move.conf.models import ModelConfig
    from move.conf.training import TrainingLoopConfig
    from move.data.dataloader import MoveDataLoader
    from move.models.base import BaseVae
    from move.training.loop import TrainingLoop


class TrainModel(ParentTask):
    """Train a single model."""

    model_filename: str = "model.pt"
    loop_filename: str = "loop.yaml"
    results_subdir: str = "train_model"

    def __init__(
        self,
        interim_data_path: PathLike,
        results_path: PathLike,
        discrete_dataset_names: list[str],
        continuous_dataset_names: list[str],
        batch_size: int,
        model_config: Union["ModelConfig", dict[str, Any]],
        training_loop_config: Union["TrainingLoopConfig", dict[str, Any]],
    ) -> None:
        super().__init__(
            input_dir=interim_data_path,
            output_dir=Path(results_path) / self.results_subdir,
        )
        self.discrete_dataset_names = discrete_dataset_names
        self.continuous_dataset_names = continuous_dataset_names
        self.batch_size = batch_size
        self.training_loop_config = training_loop_config
        self.model_config = model_config

    def make_dataloader(self, **kwargs) -> "MoveDataLoader":
        from move.data.dataloader import MoveDataLoader
        from move.data.dataset import MoveDataset

        dataset = MoveDataset.load(
            self.input_dir, self.discrete_dataset_names, self.continuous_dataset_names
        )
        return MoveDataLoader(dataset, **kwargs)

    def run(self) -> None:
        model_path = self.output_dir / self.model_filename
        if model_path.exists():
            self.logger.warning(FILE_EXISTS_WARNING.format(model_path))
        # Init data/model
        dataloader = self.make_dataloader(
            batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        model: "BaseVae" = hydra.utils.instantiate(
            self.model_config,
            discrete_shapes=dataloader.dataset.discrete_shapes,
            continuous_shapes=dataloader.dataset.continuous_shapes,
        )
        self.logger.info("Training model")
        # Train
        training_loop: "TrainingLoop" = hydra.utils.instantiate(
            self.training_loop_config, _recursive_=False
        )
        training_loop.parent = self
        training_loop.run(model, dataloader)
        training_loop.plot()
        self.logger.info("Training complete!")
        # Save model/config
        training_loop.to_yaml(self.output_dir / self.loop_filename)
        model.save(model_path)
