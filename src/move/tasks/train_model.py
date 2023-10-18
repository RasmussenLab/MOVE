__all__ = []

from pathlib import Path
from typing import Any

import hydra

from move.conf.schema import VAEConfig
from move.core.typing import PathLike
from move.data import MoveDataLoader, MoveDataset
from move.tasks.base import Task
from move.training.loop import TrainingLoop


class TrainModel(Task):
    results_subdir: str = "train_model"

    def __init__(
        self,
        interim_data_path: PathLike,
        results_path: PathLike,
        discrete_dataset_names: list[str],
        continuous_dataset_names: list[str],
        batch_size: int,
        model_config: VAEConfig,
        training_loop_config,
    ) -> None:
        super().__init__(
            input_path=Path(interim_data_path),
            output_path=Path(results_path) / self.results_subdir,
        )
        self.discrete_dataset_names = discrete_dataset_names
        self.continuous_dataset_names = continuous_dataset_names
        self.batch_size = batch_size
        self.training_loop_config = training_loop_config
        self.model_config = model_config

    def make_dataloader(self, **kwargs) -> MoveDataLoader:
        dataset = MoveDataset.load(
            self.input_path, self.discrete_dataset_names, self.continuous_dataset_names
        )
        return MoveDataLoader(dataset, **kwargs)

    def run(self) -> None:
        # Init data/model
        dataloader = self.make_dataloader(
            batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        model = hydra.utils.instantiate(
            self.model_config,
            discrete_shapes=dataloader.dataset.discrete_shapes,
            continuous_shapes=dataloader.dataset.continuous_shapes,
        )
        # Train
        training_loop: TrainingLoop = hydra.utils.instantiate(self.training_loop_config)
        training_loop.set_parent(self)
        training_loop.train(model, dataloader)
        training_loop.plot()
