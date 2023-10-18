__all__ = []

import csv
import math
from pathlib import Path
from typing import Optional, Literal, Any, cast

import hydra
import pandas as pd
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

import move.visualization as viz
from move.models.vae_new import Vae, LossDict
from move.data.dataloader import MoveDataLoader
from move.tasks.base import SubTask

AnnealingFunction = Literal["linear", "cosine", "sigmoid", "stairs"]
AnnealingSchedule = Literal["monotonic", "cyclical"]


class TrainingLoop(SubTask):
    max_steps: int
    global_step: int
    buffer: list[dict[str, float]]
    buffer_size: int = 1000
    csv_writer: csv.DictWriter

    def __init__(
        self,
        optimizer_config,
        lr_scheduler_config=None,
        max_epochs: int = 100,
        annealing_epochs: int = 20,
        annealing_function: AnnealingFunction = "linear",
        annealing_schedule: AnnealingSchedule = "monotonic",
    ):
        self.optimizer_config = optimizer_config
        self.lr_scheduler_config = lr_scheduler_config
        self.max_epochs = max_epochs
        self.annealing_epochs = annealing_epochs
        self.annealing_function = annealing_function
        self.annealing_schedule = annealing_schedule
        self.current_epoch = 0
        self.csv_file = None
        self.buffer = []

    def _repr_html_(self) -> str:
        return ""

    @property
    def annealing_factor(self) -> float:
        epoch = self.current_epoch
        if (
            self.annealing_schedule == "monotonic" and epoch < self.annealing_epochs
        ) or (self.annealing_schedule == "cyclical"):
            if self.annealing_function == "stairs":
                num_epochs_cyc = self.max_epochs / self.num_cycles
                # location in cycle: 0 (start) - 1 (end)
                loc = (epoch % math.ceil(num_epochs_cyc)) / num_epochs_cyc
                # first half of the cycle, KL weight is warmed up
                # second half, it is fixed
                if loc <= 0.5:
                    return loc * 2
            else:
                num_steps_cyc = self.max_steps / self.num_cycles
                step = self.global_step
                loc = (step % math.ceil(num_steps_cyc)) / num_steps_cyc
                if loc < 0.5:
                    if self.annealing_function == "linear":
                        return loc * 2
                    elif self.annealing_function == "sigmoid":
                        # ensure it reaches 0.5 at 1/4 of the cycle
                        shift = 0.25
                        slope = self.annealing_epochs
                        return 1 / (1 + math.exp(slope * (shift - loc)))
                    elif self.annealing_function == "cosine":
                        return math.cos((loc - 0.5) * math.pi)
        return 1.0

    @property
    def kl_weight(self) -> float:
        return self.annealing_factor * 1.0

    @property
    def num_cycles(self) -> float:
        return self.max_epochs / (self.annealing_epochs * 2)

    def _add_to_buffer(self, loss_dict: LossDict) -> None:
        """Add loss to buffer and flush buffer if it has reached its limit.

        Args:
            loss_dict: Dict containing ELBO loss components
        """
        if self.csv_file:
            csv_row: dict[str, float] = {
                "epoch": self.current_epoch,
                "step": self.global_step,
            }
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    csv_row[key] = value.item()
                else:
                    csv_row[key] = cast(float, value)
            self.buffer.append(csv_row)
            # Flush
            if len(self.buffer) >= self.buffer_size:
                self.csv_writer.writerows(self.buffer)
                self.buffer.clear()

    def _init_csv_writer(self) -> None:
        """Initialize the CSV writer if there is an output path."""
        if self.parent:
            self.csv_filepath = self.parent.output_path / "loss_curve.csv"
            if self.csv_filepath.exists() and self.parent:
                self.parent.logger.info(
                    f"File '{self.csv_filepath}' already exists. It be overwritten."
                )
            self.csv_file = open(self.csv_filepath, "w", newline="")
            colnames = ["epoch", "step"] + list(LossDict.__annotations__.keys())
            self.csv_writer = csv.DictWriter(self.csv_file, colnames)
            self.csv_writer.writeheader()

    def plot(self) -> None:
        if self.parent:
            data = pd.read_csv(self.csv_filepath)
            losses = [data[key].to_list() for key in LossDict.__annotations__.keys()]
            fig = viz.plot_loss_curves(losses, xlabel="Step")
            fig_path = str(self.parent.output_path / "loss_curve.png")
            fig.savefig(fig_path, bbox_inches="tight")

    def train(
        self,
        model: Vae,
        train_dataloader: MoveDataLoader,
    ):
        """Train a VAE model.

        Args:
            model: VAE model
            train_dataloader: Training data loader
        """
        num_batches = len(train_dataloader)
        self.max_steps = self.max_epochs * num_batches
        self.global_step = 0
        self._init_csv_writer()

        optimizer: Optimizer = hydra.utils.instantiate(
            self.optimizer_config, params=model.parameters()
        )
        if self.lr_scheduler_config:
            lr_scheduler: Optional[LRScheduler] = hydra.utils.instantiate(
                self.lr_scheduler_config, optimizer=optimizer
            )
        else:
            lr_scheduler = None

        for epoch in range(0, self.max_epochs):
            self.current_epoch = epoch

            model.train()

            for batch in train_dataloader:
                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                loss_dict = model.compute_loss(batch, self.annealing_factor)

                # Backward pass and optimize
                loss_dict["elbo"].backward()
                optimizer.step()

                self._add_to_buffer(loss_dict)
                self.global_step += 1

            """ if valid_dataloader is not None:
                model.eval()
                with torch.no_grad():
                    for batch in valid_dataloader:
                        loss_dict = model.compute_loss(batch, self.annealing_factor)
                        for key, value in loss_dict.items():
                            if isinstance(value, torch.Tensor):
                                epoch_loss[f"valid_{key}"] += value.item() / num_batches """

            if lr_scheduler:
                lr_scheduler.step()
            self.current_epoch += 1

        if self.csv_file:
            if self.buffer:
                self.csv_writer.writerows(self.buffer)
            self.csv_file.close()
