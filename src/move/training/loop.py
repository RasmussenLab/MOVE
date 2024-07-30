__all__ = []

import math
from typing import Literal, Optional, cast

import hydra
import pandas as pd
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

import move.visualization as viz
from move.conf.optim import LrSchedulerConfig, OptimizerConfig
from move.data.dataloader import MoveDataLoader
from move.models.base import BaseVae, LossDict
from move.tasks.base import CsvWriterMixin, SubTask

AnnealingFunction = Literal["linear", "cosine", "sigmoid", "stairs"]
AnnealingSchedule = Literal["monotonic", "cyclical"]


class TrainingLoop(CsvWriterMixin, SubTask):
    """Train a VAE model.

    Args:
        optimizer_config:
            Configuration for the optimizer.
        lr_scheduler_config:
            Configuration for the learning rate scheduler.
        max_epochs:
            Max training epochs, may be lower if early stopping is implemented.
        max_grad_norm:
            If other than none, clip gradient norm.
        annealing_epochs:
            Epochs required to fully warm KL divergence. Set to 0 and a
            `monotonic` schedue to turn off KL divergence annealing.
        annealing_function:
            Function to warm KL divergence.
        annealing_schedule:
            Whether KL divergence is warmed monotonically or cyclically.
        prog_every_n_epoch:
            Log progress every n-th epoch. Note this only controls a message
            displaying the current epoch. Loss and other metrics are logged at
            every step.
        log_grad:
            Whether gradients should be logged.
    """

    max_steps: int
    global_step: int

    def __init__(
        self,
        optimizer_config: OptimizerConfig,
        lr_scheduler_config: Optional[LrSchedulerConfig] = None,
        max_epochs: int = 100,
        max_grad_norm: Optional[float] = None,
        annealing_epochs: int = 20,
        annealing_function: AnnealingFunction = "linear",
        annealing_schedule: AnnealingSchedule = "monotonic",
        prog_every_n_epoch: Optional[int] = 10,
        log_grad: bool = False,
    ):
        if annealing_epochs < 0:
            raise ValueError("Annealing epochs must be a non-negative integer")
        if annealing_epochs == 0 and annealing_schedule == "cyclical":
            raise ValueError(
                "Annealing epochs must be a positive integer if schedule is cyclical"
            )
        self.optimizer_config = optimizer_config
        self.lr_scheduler_config = lr_scheduler_config
        self.max_epochs = max_epochs
        self.max_grad_norm = max_grad_norm
        self.annealing_epochs = annealing_epochs
        self.annealing_function = annealing_function
        self.annealing_schedule = annealing_schedule
        self.current_epoch = 0
        self.prog_every_n_epoch = prog_every_n_epoch
        self.log_grad = log_grad

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

    def plot(self) -> None:
        if self.parent is not None and self.csv_filepath is not None:
            data = pd.read_csv(self.csv_filepath)
            data["kl_div"] *= data["kl_weight"]
            fig = viz.plot_loss_curves(data)
            fig_path = str(self.parent.output_dir / "loss_curve.png")
            fig.savefig(fig_path, bbox_inches="tight")

    def run(self, model: BaseVae, train_dataloader: MoveDataLoader) -> None:
        return self.train(model, train_dataloader)

    def get_colnames(self, model: Optional[BaseVae] = None) -> list[str]:
        """Return the list of column names of the CSV being generated. If set
        to log gradients, a model would be required to obtain the names of its
        parameters.

        Args:
            model: a deep learning model"""
        colnames = ["epoch", "step"]
        colnames.extend(LossDict.__annotations__.keys())
        if self.log_grad and model is not None:
            for module_name, module in model.named_children():
                param_names = []
                for param_name, param in module.named_parameters(module_name):
                    if param.requires_grad:
                        param_names.append(param_name)
                if param_names:
                    colnames.append(module_name)
                    colnames.extend(param_names)
        return colnames

    def make_row(self, loss_dict: LossDict, model: BaseVae) -> dict[str, float]:
        """Format a loss dictionary and the model's gradients into a dictionary
        representing a CSV row.

        Args:
            loss_dict: dictionary with loss metrics
            model: deep-learning model"""
        csv_row: dict[str, float] = {
            "epoch": self.current_epoch,
            "step": self.global_step,
        }
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                csv_row[key] = value.item()
            else:
                csv_row[key] = cast(float, value)
        if self.log_grad and model is not None:
            for module_name, module in model.named_children():
                grads = []
                for param_name, param in module.named_parameters(module_name):
                    if param.grad is not None:
                        grad = torch.norm(param.grad.detach())
                        grads.append(grad)
                        csv_row[param_name] = grad.item()
                if len(grads) > 0:
                    csv_row[module_name] = torch.norm(torch.stack(grads)).item()
        return csv_row

    def train(
        self,
        model: BaseVae,
        train_dataloader: MoveDataLoader,
    ) -> None:
        """Train a VAE model.

        Args:
            model: VAE model
            train_dataloader: Training data loader
        """
        num_batches = len(train_dataloader)
        self.max_steps = self.max_epochs * num_batches
        self.global_step = 0
        if self.parent:
            self.init_csv_writer(
                self.parent.output_dir / "loss_curve.csv",
                fieldnames=self.get_colnames(model),
            )

        if train_dataloader.dataset.perturbation is not None:
            self.log("Dataset's perturbation will be removed", "WARNING")
            train_dataloader.dataset.perturbation = None

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

            for (batch,) in train_dataloader:
                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                try:
                    loss_dict = model.compute_loss(batch, self.annealing_factor)
                except (KeyboardInterrupt, ValueError) as exception:
                    self.close_csv_writer()
                    raise exception
                # Backward pass and optimize
                loss_dict["elbo"].backward()

                if self.max_grad_norm is not None:
                    clip_grad_norm_(model.parameters(), self.max_grad_norm)

                optimizer.step()

                csv_row = self.make_row(loss_dict, model)
                self.add_row_to_buffer(csv_row)

                self.global_step += 1

            """ if valid_dataloader is not None:
                model.eval()
                with torch.no_grad():
                    for (batch,) in valid_dataloader:
                        loss_dict = model.compute_loss(batch, self.annealing_factor)
                        for key, value in loss_dict.items():
                            if isinstance(value, torch.Tensor):
                                epoch_loss[f"valid_{key}"] += value.item() / num_batches """

            if lr_scheduler is not None:
                lr_scheduler.step()

            self.current_epoch += 1

            if (
                self.prog_every_n_epoch is not None
                and self.current_epoch % self.prog_every_n_epoch == 0
            ):
                num_zeros = int(math.log10(self.max_epochs)) + 1
                self.log(f"Epoch {self.current_epoch:0{num_zeros}}")

        model.freeze()
        self.close_csv_writer()
