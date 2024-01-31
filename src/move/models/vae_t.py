__all__ = ["VaeT"]

import itertools
import operator
from typing import Optional, Type

import torch
import torch.optim
from torch import nn
from torch.distributions import Distribution, Categorical, Normal, StudentT
from torch.distributions import kl_divergence

from move.core.exceptions import CudaIsNotAvailable, ShapeAndWeightMismatch
from move.models.base import BaseVae, LossDict, VaeOutput
from move.models.layers.chunk import SplitInput, SplitOutput
from move.models.layers.encoder_decoder import Decoder, Encoder


class VaeT(BaseVae):
    """Variational autoencoder with a Student-t distribution on its decoder."""

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

        self.split_input = SplitInput(self.discrete_shapes, self.continuous_shapes)
        self.split_output = SplitOutput(
            self.discrete_shapes,
            self.continuous_shapes,
            "StudentT",
            # continuous_activation_name="Tanh",
        )

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
            self.split_output.num_expected_features,
            dropout_rate=dropout_rate,
        )

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

    def reconstruct(self, batch: torch.Tensor) -> torch.Tensor:
        out = self(batch)["x_recon"]
        out_disc, out_cont = self.split_output(out)
        recon = torch.cat(
            [logits.flatten(start_dim=1) for logits in out_disc]
            + [loc for (_, loc, _) in out_cont],
            dim=1,
        )
        return recon

    def forward(self, x: torch.Tensor) -> VaeOutput:
        z_loc, z_scale = self.encode(x)
        z = self.reparameterize(z_loc, z_scale)
        x_recon, *_ = self.decode(z)
        return {
            "z_loc": z_loc,
            "z_scale": z_scale,
            "x_recon": x_recon,
        }

    @staticmethod
    def compute_log_prob(dist: Type, x: torch.Tensor, **dist_args):
        """Compute the log of the probability density of the likelihood p(x|z)."""
        px: Distribution = dist(**dist_args)
        return torch.sum(px.log_prob(x), dim=-1).mean()

    @staticmethod
    def compute_kl_div(qz_loc: torch.Tensor, qz_scale: torch.Tensor):
        """Compute the KL divergence between posterior q(z|x) and prior p(z).
        The prior has a Normal(0, 1) distribution."""
        qz = Normal(qz_loc, qz_scale)
        pz = Normal(0.0, 1.0)
        return kl_divergence(qz, pz).sum(dim=-1).mean()

    def compute_loss(self, batch: torch.Tensor, annealing_factor: float) -> LossDict:
        # Split concatenated input
        batch_disc, batch_cont = self.split_input(batch)
        # Split concatenated output
        out = self(batch)
        out_disc, out_cont = self.split_output(out["x_recon"])

        # Compute discrete dataset losses
        disc_rec_loss = torch.tensor(0.0)
        for i, args in enumerate(out_disc):
            y = torch.argmax(batch_disc[i], dim=-1)
            disc_rec_loss -= self.compute_log_prob(Categorical, y, logits=args)

        # Compute continuous dataset losses
        cont_rec_loss = torch.tensor(0.0)
        for i, args in enumerate(out_cont):
            df, loc, logvar = args
            df = torch.pow(torch.exp(df * -0.5) * 27.5 + 2.5, -1)
            scale = torch.exp(logvar * 0.5)
            cont_rec_loss -= self.compute_log_prob(
                StudentT, batch_cont[i], df=df, loc=loc, scale=scale
            )

        # Calculate overall reconstruction and regularization loss
        rec_loss = disc_rec_loss + cont_rec_loss
        reg_loss = self.compute_kl_div(out["z_loc"], out["z_scale"])

        # Compute ELBO
        kl_weight = annealing_factor * self.kl_weight
        elbo = rec_loss + reg_loss * kl_weight
        return {
            "elbo": elbo,
            "discrete_loss": disc_rec_loss,
            "continuous_loss": cont_rec_loss,
            "kl_div": reg_loss,
            "kl_weight": kl_weight,
        }
