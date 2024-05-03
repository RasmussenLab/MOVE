__all__ = ["Vae"]

import itertools
import operator
from typing import Optional, cast

import torch
import torch.optim
from torch import nn

from move.core.exceptions import CudaIsNotAvailable, ShapeAndWeightMismatch
from move.models.base import BaseVae, LossDict, VaeOutput
from move.models.layers.chunk import ContinuousData, SplitInput, SplitOutput
from move.models.layers.encoder_decoder import Decoder, Encoder


class Vae(BaseVae):
    """Variational autoencoder"""

    def __init__(
        self,
        discrete_shapes: list[tuple[int, int]],
        continuous_shapes: list[int],
        discrete_weights: Optional[list[float]] = None,
        continuous_weights: Optional[list[float]] = None,
        num_hidden: list[int] = [200, 200],
        num_latent: int = 20,
        kl_weight: float = 0.01,
        dropout_rate: float = 0.2,
        use_cuda: bool = False,
    ) -> None:
        super().__init__()

        # Validate and save arguments
        if sum(num_hidden) <= 0:
            raise ValueError(
                "Number of hidden units in encoder/decoder must be non-negative."
            )
        self.num_hidden = num_hidden

        if num_latent < 1:
            raise ValueError("Latent space size must be non-negative.")
        self.num_latent = num_latent

        if kl_weight <= 0:
            raise ValueError("KLD weight must be greater than zero.")
        self.kl_weight = kl_weight

        if not (0 <= dropout_rate < 1):
            raise ValueError("Dropout rate must be between [0, 1).")
        self.dropout_rate = dropout_rate

        if discrete_shapes is None and continuous_shapes is None:
            raise ValueError("Shapes of input datasets must be provided.")

        self.discrete_shapes = discrete_shapes
        self.disc_split_sizes = []
        self.num_disc_features = 0
        self.discrete_weights = [1.0] * len(self.discrete_shapes)
        if discrete_shapes is not None:
            (*shapes_1d,) = itertools.starmap(operator.mul, discrete_shapes)
            *self.disc_split_sizes, _ = itertools.accumulate(shapes_1d)
            self.num_disc_features = sum(shapes_1d)
            if discrete_weights is not None:
                if len(discrete_shapes) != len(discrete_weights):
                    raise ShapeAndWeightMismatch(
                        len(discrete_shapes), len(discrete_weights)
                    )
                self.discrete_weights = discrete_weights

        self.continuous_shapes = continuous_shapes
        self.cont_split_sizes = []
        self.num_cont_features = 0
        self.continuous_weights = [1.0] * len(self.continuous_shapes)
        if continuous_shapes is not None:
            *self.cont_split_sizes, _ = itertools.accumulate(
                [shape * self.output_args for shape in continuous_shapes]
            )
            self.num_cont_features = sum(continuous_shapes)
            if continuous_weights is not None:
                if len(continuous_shapes) != len(continuous_weights):
                    raise ShapeAndWeightMismatch(
                        len(continuous_shapes), len(continuous_weights)
                    )
                self.continuous_weights = continuous_weights

        self.in_features = self.num_disc_features + self.num_cont_features

        self.encoder = Encoder(
            self.in_features,
            num_hidden,
            num_latent,
            embedding_args=self.embedding_args,
            dropout_rate=dropout_rate,
        )
        self.decoder = Decoder(
            num_latent,
            num_hidden[::-1],
            self.in_features,
            dropout_rate=dropout_rate,
        )
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.split_input = SplitInput(self.discrete_shapes, self.continuous_shapes)
        self.split_output = SplitOutput(self.discrete_shapes, self.continuous_shapes)

        self.nll_loss = nn.NLLLoss(reduction="sum", ignore_index=-1)
        self.mse_loss = nn.MSELoss(reduction="sum")

        if use_cuda and not torch.cuda.is_available():
            raise CudaIsNotAvailable()
        self.use_cuda = use_cuda

        device = torch.device("cuda" if use_cuda else "cpu")
        self.to(device)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_loc, z_logvar, *_ = self.encoder(x)
        return z_loc, torch.exp(z_logvar * 0.5)

    def reparameterize(self, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(scale)
        return eps.mul(scale).add_(loc)

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.decoder(z)

    def project(self, batch: torch.Tensor) -> torch.Tensor:
        return self.encode(batch)[0]

    def reconstruct(self, batch: torch.Tensor, as_one: bool = False):
        out = self(batch)["x_recon"]
        if as_one:
            return out
        out_disc, out_cont = self.split_output(out)
        recon_disc = torch.cat(
            [logits.flatten(start_dim=1) for logits in out_disc], dim=1
        )
        recon_cont = torch.cat(cast(ContinuousData, out_cont), dim=1)
        return recon_disc, recon_cont

    def forward(self, x: torch.Tensor) -> VaeOutput:
        z_loc, z_scale = self.encode(x)
        z = self.reparameterize(z_loc, z_scale)
        x_recon, *_ = self.decode(z)
        return {
            "z_loc": z_loc,
            "z_scale": z_scale,
            "x_recon": x_recon,
        }

    def compute_loss(self, batch: torch.Tensor, annealing_factor: float) -> LossDict:
        # Split concatenated input
        batch_disc, batch_cont = self.split_input(batch)
        # Split concatenated output
        out = self(batch)
        out_disc, out_cont = self.split_output(out["x_recon"])

        # Compute discrete dataset losses
        disc_losses = []
        for disc_input, disc_logits, disc_wt in zip(
            batch_disc, out_disc, self.discrete_weights
        ):
            disc_recon = self.log_softmax(disc_logits).transpose(1, 2)
            disc_cats = disc_input.argmax(dim=-1)
            na_mask = disc_input.sum(dim=-1) == 0
            disc_cats[na_mask] = -1.0
            multiplier = disc_wt / operator.mul(*disc_input.shape[:-1])
            loss = self.nll_loss(disc_recon, disc_cats) * multiplier
            disc_losses.append(loss)
        disc_loss = torch.stack(disc_losses).sum()

        # Compute continuous dataset losses
        cont_losses = []
        for cont_input, cont_recon, cont_wt in zip(
            batch_cont, out_cont, self.continuous_weights
        ):
            na_mask = (cont_input == 0).logical_not().float()
            multiplier = cont_wt / operator.mul(*cont_input.shape)
            loss = self.mse_loss(na_mask * cont_recon, cont_input) * multiplier
            cont_losses.append(loss)
        cont_loss = torch.stack(cont_losses).sum()

        # Compute KL divergence
        z_loc, z_var = out["z_loc"], out["z_scale"] ** 2
        kl_div = (
            -0.5 * torch.sum(1 + z_var.log() - z_loc.pow(2) - z_var) / batch.size(0)
        )

        # Compute ELBO
        kl_weight = annealing_factor * self.kl_weight
        elbo = disc_loss + cont_loss + kl_div * kl_weight
        return {
            "elbo": elbo,
            "discrete_loss": disc_loss,
            "continuous_loss": cont_loss,
            "kl_div": kl_div,
            "kl_weight": kl_weight,
        }
