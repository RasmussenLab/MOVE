__all__ = []

from pathlib import Path
from typing import Any

import torch

from move.core.typing import PathLike
from move.tasks.base import MoveTask
from move.tasks.feature_importance import FeatureImportance
from move.tasks.metrics import ComputeAccuracyMetrics


class LatentSpaceAnalysis(MoveTask):
    results_subdir: str = "latent_space"

    def __init__(
        self,
        interim_data_path: PathLike,
        results_path: PathLike,
        compute_accuracy_metrics: bool,
        compute_feature_importance: bool,
        **kwargs
    ) -> None:
        super().__init__(
            input_dir=interim_data_path,
            output_dir=Path(results_path) / self.results_subdir,
            **kwargs
        )
        self.compute_accuracy_metrics = compute_accuracy_metrics
        self.compute_feature_importance = compute_feature_importance

    def run(self) -> Any:
        train_dataloader = self.make_dataloader()

        model = self.init_model(train_dataloader)
        model_path = self.output_dir / "model.pt"

        if model_path.exists():
            self.logger.debug("Re-loading model")
            model.load_state_dict(torch.load(model_path))
        else:
            self.logger.debug("Training a new model")
            training_loop = self.init_training_loop()
            training_loop.train(model, train_dataloader)
            training_loop.plot()
            torch.save(model.state_dict(), model_path)

        model.eval()

        test_dataloader = self.make_dataloader(shuffle=False, drop_last=False)

        if self.compute_accuracy_metrics:
            subtask = ComputeAccuracyMetrics(self, model, test_dataloader)
            subtask.run()

        if self.compute_feature_importance:
            subtask = FeatureImportance(self, model, test_dataloader)
            subtask.run()
