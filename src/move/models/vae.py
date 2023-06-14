__all__ = ["VAE"]

import logging
from typing import Optional, Callable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from move.core.typing import FloatArray, IntArray

logger = logging.getLogger("vae.py")


class VAE(nn.Module):
    """Variational autoencoder.

    Instantiate with:
        continuous_shapes: shape of the different continuous datasets if any
        categorical_shapes: shape of the different categorical datasets if any
        num_hidden: List of n_neurons in the hidden layers [[200, 200]]
        num_latent: Number of neurons in the latent layer [15]
        beta: Multiply KLD by the inverse of this value [0.0001]
        continuous_weights: list of weights for each continuous dataset
        categorical_weights: list of weights for each categorical dataset
        dropout: Probability of dropout on forward pass [0.2]
        cuda: Use CUDA (GPU accelerated training) [False]

    Raises:
        ValueError: Minimum 1 latent unit
        ValueError: Beta must be greater than zero.
        ValueError: Dropout must be between zero and one.
        ValueError: Shapes of the input data must be provided.
        ValueError: Number of continuous weights must be the same as number of
            continuous datasets
        ValueError: Number of categorical weights must be the same as number of
            categorical datasets
    """

    def __init__(
        self,
        categorical_shapes: Optional[list[tuple[int, ...]]] = None,
        continuous_shapes: Optional[list[int]] = None,
        categorical_weights: Optional[list[int]] = None,
        continuous_weights: Optional[list[int]] = None,
        num_hidden: list[int] = [200, 200],
        num_latent: int = 20,
        beta: float = 0.01,
        dropout: float = 0.2,
        cuda: bool = False,
    ):

        if num_latent < 1:
            raise ValueError(f"Minimum 1 latent unit. Input was {num_latent}.")

        if beta <= 0:
            raise ValueError("Beta must be greater than zero.")

        if not (0 <= dropout < 1):
            raise ValueError("Dropout must be between zero and one.")

        if continuous_shapes is None and categorical_shapes is None:
            raise ValueError("Shapes of the input data must be provided.")

        self.input_size = 0
        if continuous_shapes is not None:
            self.num_continuous = sum(continuous_shapes)
            self.input_size += self.num_continuous
            self.continuous_shapes = continuous_shapes

            if continuous_weights is not None:
                self.continuous_weights = continuous_weights
                if len(continuous_shapes) != len(continuous_weights):
                    raise ValueError(
                        "Number of continuous weights must be the same as"
                        " number of continuous datasets"
                    )
        else:
            self.num_continuous = 0

        if categorical_shapes is not None:
            self.num_categorical = sum(
                [int.__mul__(*shape) for shape in categorical_shapes]
            )
            self.input_size += self.num_categorical
            self.categorical_shapes = categorical_shapes

            if categorical_weights is not None:
                self.categorical_weights = categorical_weights
                if len(categorical_shapes) != len(categorical_weights):
                    raise ValueError(
                        "Number of categorical weights must be the same as"
                        " number of categorical datasets"
                    )
        else:
            self.num_categorical = 0

        super(VAE, self).__init__()

        # Initialize simple attributes
        self.beta = beta
        self.num_hidden = num_hidden
        self.num_latent = num_latent
        self.dropout = dropout

        self.device = torch.device("cuda" if cuda == True else "cpu")

        # Activation functions
        self.relu = nn.LeakyReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.dropoutlayer = nn.Dropout(p=self.dropout)

        # Initialize lists for holding hidden layers
        self.encoderlayers = nn.ModuleList()
        self.encodernorms = nn.ModuleList()
        self.decoderlayers = nn.ModuleList()
        self.decodernorms = nn.ModuleList()

        ### Layers
        # Hidden layers
        for nin, nout in zip([self.input_size] + self.num_hidden, self.num_hidden):
            self.encoderlayers.append(nn.Linear(nin, nout))
            self.encodernorms.append(nn.BatchNorm1d(nout))

        # Latent layers
        self.mu = nn.Linear(self.num_hidden[-1], self.num_latent)  # mu layer
        self.var = nn.Linear(self.num_hidden[-1], self.num_latent)  # logvariance layer

        # Decoding layers
        for nin, nout in zip(
            [self.num_latent] + self.num_hidden[::-1], self.num_hidden[::-1]
        ):
            self.decoderlayers.append(nn.Linear(nin, nout))
            self.decodernorms.append(nn.BatchNorm1d(nout))

        # Reconstruction - output layers
        self.out = nn.Linear(self.num_hidden[0], self.input_size)  # to output

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes the data in the data loader and returns the encoded matrix.

        Args:
            x: input data

        Returns:
            A tuple containing:
                mean latent vector
                log-variance latent vector
        """
        # Hidden layers
        for encoderlayer, encodernorm in zip(self.encoderlayers, self.encodernorms):
            x = encoderlayer(x)
            x = self.relu(x)
            x = self.dropoutlayer(x)
            x = encodernorm(x)

        return self.mu(x), self.var(x)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Performs reparametrization trick

        Args:
            mu: mean latent vector
            logvar: log-variance latent vector

        Returns:
            sample from latent space distribution
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps.mul(std).add_(mu)

    def decompose_categorical(self, reconstruction: torch.Tensor) -> list[torch.Tensor]:
        """
        Returns list of final reconstructions (after applying
        log-softmax to the outputs of decoder) of each categorical class

        Args:
            reconstruction: results of final layer of decoder

        Returns:
            final reconstructions of each categorical class
        """
        cat_tmp = reconstruction.narrow(1, 0, self.num_categorical)

        # handle soft max for each categorical dataset
        cat_out = []
        pos = 0
        for cat_shape in self.categorical_shapes:
            cat_dataset = cat_tmp[:, pos : (cat_shape[0] * cat_shape[1] + pos)]

            cat_out_tmp = cat_dataset.view(
                cat_dataset.shape[0], cat_shape[0], cat_shape[1]
            )
            cat_out_tmp = cat_out_tmp.transpose(1, 2)
            cat_out_tmp = self.log_softmax(cat_out_tmp)

            cat_out.append(cat_out_tmp)
            pos += cat_shape[0] * cat_shape[1]

        return cat_out

    def decode(
        self, x: torch.Tensor
    ) -> tuple[Optional[list[torch.Tensor]], Optional[torch.Tensor]]:
        """
        Decode to the input space from the latent space

        Args:
            x: sample from latent space distribution

        Returns:
            A tuple containing:
                cat_out:
                    list of reconstructions of every categorical data class
                con_out:
                    reconstruction of continuous data
        """
        for decoderlayer, decodernorm in zip(self.decoderlayers, self.decodernorms):
            x = decoderlayer(x)
            x = self.relu(x)
            x = self.dropoutlayer(x)
            x = decodernorm(x)

        reconstruction = self.out(x)

        # Decompose reconstruction to categorical and continuous variables
        # if both types are in the input
        cat_out, con_out = None, None
        if self.num_categorical > 0:
            cat_out = self.decompose_categorical(reconstruction)
        if self.num_continuous > 0:
            con_out = reconstruction.narrow(
                1, self.num_categorical, self.num_continuous
            )

        return cat_out, con_out

    def forward(
        self, tensor: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward propagate through the VAE network

        Args:
            tensor (torch.Tensor): input data

        Returns:
            (tuple): a tuple containing:
                cat_out (list): list of reconstructions of every categorical
                    data class
                con_out (torch.Tensor): reconstructions of continuous data
                mu (torch.Tensor): mean latent vector
                logvar (torch.Tensor): mean log-variance vector
        """
        mu, logvar = self.encode(tensor)
        z = self.reparameterize(mu, logvar)
        cat_out, con_out = self.decode(z)

        return cat_out, con_out, mu, logvar

    def calculate_cat_error(
        self,
        cat_in: torch.Tensor,
        cat_out: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Calculates errors (cross-entropy) for categorical data reconstructions

        Args:
            cat_in:
                input categorical data
            cat_out:
                list of reconstructions of every categorical data class

        Returns:
            torch.Tensor:
                Errors (cross-entropy) for categorical data reconstructions
        """
        batch_size = cat_in.shape[0]

        # calcualte target values for all cat datasets
        count = 0
        cat_errors = []
        pos = 0
        for cat_shape in self.categorical_shapes:
            cat_dataset = cat_in[:, pos : (cat_shape[0] * cat_shape[1] + pos)]

            cat_dataset = cat_dataset.view(cat_in.shape[0], cat_shape[0], cat_shape[1])
            cat_target = cat_dataset
            cat_target = cat_target.argmax(2)
            cat_target[cat_dataset.sum(dim=2) == 0] = -1
            cat_target = cat_target.to(self.device)

            # Cross entropy loss for categroical
            loss = nn.NLLLoss(reduction="sum", ignore_index=-1)
            cat_errors.append(
                loss(cat_out[count], cat_target) / (batch_size * cat_shape[0])
            )
            count += 1
            pos += cat_shape[0] * cat_shape[1]

        cat_errors = torch.stack(cat_errors)
        return cat_errors

    def calculate_con_error(
        self, con_in: torch.Tensor, con_out: torch.Tensor, loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """
        Calculates errors (MSE) for continuous data reconstructions

        Args:
            con_in: input continuous data
            con_out: reconstructions of continuous data
            loss: loss function

        Returns:
            MSE loss
        """
        batch_size = con_in.shape[0]
        total_shape = 0
        con_errors_list: list[torch.Tensor] = []
        for s in self.continuous_shapes:
            c_in = con_in[:, total_shape : (s + total_shape - 1)]
            c_re = con_out[:, total_shape : (s + total_shape - 1)]
            error = loss(c_re, c_in) / batch_size
            con_errors_list.append(error)
            total_shape += s

        con_errors = torch.stack(con_errors_list)
        con_errors = con_errors / torch.Tensor(self.continuous_shapes).to(self.device)
        MSE = torch.sum(
            con_errors * torch.Tensor(self.continuous_weights).to(self.device)
        )
        return MSE

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(
        self,
        cat_in: torch.Tensor,
        cat_out: list[torch.Tensor],
        con_in: torch.Tensor,
        con_out: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kld_w: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the loss for data reconstructions

        Args:
            cat_in: input categorical data
            cat_out: list of reconstructions of every categorical data class
            con_in: input continuous data
            con_out: reconstructions of continuous data
            mu: mean latent vector
            logvar: mean log-variance vector
            kld_w: kld weight

        Returns:
            (tuple): a tuple containing:
                total loss on train set during the training of the epoch
                BCE loss on train set during the training of the epoch
                SSE loss on train set during the training of the epoch
                KLD loss on train set during the training of the epoch
        """

        MSE = 0
        CE = 0
        # calculate loss for catecorical data if in the input
        if cat_out is not None:
            cat_errors = self.calculate_cat_error(cat_in, cat_out)
            if self.categorical_weights is not None:
                CE = torch.sum(
                    cat_errors * torch.Tensor(self.categorical_weights).to(self.device)
                )
            else:
                CE = torch.sum(cat_errors) / len(cat_errors)

        # calculate loss for continuous data if in the input
        if con_out is not None:
            batch_size = con_in.shape[0]
            # Mean square error loss for continauous
            loss = nn.MSELoss(reduction="sum")
            # set missing data to 0 to remove any loss these would provide
            con_out[con_in == 0] = 0

            # include different weights for each omics dataset
            if self.continuous_weights is not None:
                MSE = self.calculate_con_error(con_in, con_out, loss)
            else:
                MSE = loss(con_out, con_in) / (batch_size * self.num_continuous)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / (batch_size)

        KLD_weight = self.beta * kld_w
        loss = CE + MSE + KLD * KLD_weight

        return loss, CE, MSE, KLD * KLD_weight

    def encoding(
        self,
        train_loader: DataLoader,
        epoch: int,
        lrate: float,
        kld_w: float,
    ) -> tuple[float, float, float, float]:
        """
        One iteration of VAE

        Args:
            train_loader: Dataloader with train dataset
            epoch: the epoch
            lrate: learning rate for the model
            kld_w: float of KLD weight

        Returns:
            (tuple): a tuple containing:
                total loss on train set during the training of the epoch
                BCE loss on train set during the training of the epoch
                SSE loss on train set during the training of the epoch
                KLD loss on train set during the training of the epoch
        """
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lrate)

        epoch_loss = 0
        epoch_kldloss = 0
        epoch_sseloss = 0
        epoch_bceloss = 0

        for _, (cat, con) in enumerate(train_loader):
            # Move input to GPU if requested
            cat = cat.to(self.device)
            con = con.to(self.device)

            if self.num_categorical > 0 and self.num_continuous > 0:
                tensor = torch.cat((cat, con), 1)
            elif self.num_categorical > 0:
                tensor = cat
            elif self.num_continuous > 0:
                tensor = con
            else:
                assert False, "Must have at least 1 categorial or 1 continuous feature"

            optimizer.zero_grad()

            cat_out, con_out, mu, logvar = self(tensor)

            loss, bce, sse, kld = self.loss_function(
                cat, cat_out, con, con_out, mu, logvar, kld_w
            )
            loss.backward()

            epoch_loss += loss.data.item()
            epoch_kldloss += kld.data.item()

            if self.num_continuous > 0:
                epoch_sseloss += sse.data.item()

            if self.num_categorical > 0:
                epoch_bceloss += bce.data.item()

            optimizer.step()

        logger.info(
            "\tEpoch: {}\tLoss: {:.6f}\tCE: {:.7f}\tSSE: {:.6f}\t"
            "KLD: {:.4f}\tBatchsize: {}".format(
                epoch,
                epoch_loss / len(train_loader),
                epoch_bceloss / len(train_loader),
                epoch_sseloss / len(train_loader),
                epoch_kldloss / len(train_loader),
                train_loader.batch_size,
            )
        )
        return (
            epoch_loss / len(train_loader),
            epoch_bceloss / len(train_loader),
            epoch_sseloss / len(train_loader),
            epoch_kldloss / len(train_loader),
        )

    def make_cat_recon_out(self, length: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Initiate empty tensors for categorical data

        Args:
            length: number of samples

        Returns:
            (tuple): a tuple containing:
                cat_class: empty tensor for input categorical data
                cat_recon: empty tensor for reconstructed categorical data
                cat_total_shape: number of features of linearized one hot
                    categorical data
        """
        cat_total_shape = 0
        for cat_shape in self.categorical_shapes:
            cat_total_shape += cat_shape[0]

        cat_class = torch.empty((length, cat_total_shape)).int()
        cat_recon = torch.empty((length, cat_total_shape)).int()
        return cat_class, cat_recon, cat_total_shape

    def get_cat_recon(
        self, batch: int, cat_total_shape: int, cat: torch.Tensor, cat_out: torch.Tensor
    ) -> tuple[IntArray, IntArray]:
        """
        Generates reconstruction data of categorical data class

        Args:
            batch: number of samples in the batch
            cat_total_shape: number of features of linearized one hot
                categorical data
            cat: input categorical data
            cat_out: reconstructed categorical data

        Returns:
            (tuple): a tuple containing:
                cat_out_class: reconstructed categorical data
                cat_target: input categorical data
        """
        count = 0
        cat_out_class = torch.empty((batch, cat_total_shape)).int()
        cat_target = torch.empty((batch, cat_total_shape)).int()
        pos = 0
        shape_1 = 0
        for cat_shape in self.categorical_shapes:
            # Get input categorical data
            cat_in_tmp = cat[:, pos : (cat_shape[0] * cat_shape[1] + pos)]
            cat_in_tmp = cat_in_tmp.view(cat.shape[0], cat_shape[0], cat_shape[1])

            # Calculate target values for input
            cat_target_tmp = cat_in_tmp
            cat_target_tmp = torch.argmax(cat_target_tmp.detach(), dim=2)
            cat_target_tmp[cat_in_tmp.sum(dim=2) == 0] = -1
            cat_target[
                :, shape_1 : (cat_shape[0] + shape_1)
            ] = cat_target_tmp  # .numpy()

            # Get reconstructed categorical data
            cat_out_tmp = cat_out[count]
            cat_out_tmp = cat_out_tmp.transpose(1, 2)
            cat_out_class[:, shape_1 : (cat_shape[0] + shape_1)] = torch.argmax(
                cat_out_tmp, dim=2
            )  # .numpy()

            # make counts for next dataset
            pos += cat_shape[0] * cat_shape[1]
            shape_1 += cat_shape[0]
            count += 1

        cat_target = cat_target.numpy()
        cat_out_class = cat_out_class.numpy()

        return cat_out_class, cat_target

    def _validate_batch(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Returns the batch of categorical and continuous data if they are not
        None

        Args:
            batch: batches of categorical and continuous data

        Returns:
            a formed batch
        """
        cat, con = batch
        cat = cat.to(self.device)
        con = con.to(self.device)

        if self.num_categorical == 0:
            return con
        elif self.num_continuous == 0:
            return cat
        return torch.cat((cat, con), dim=1)

    @torch.no_grad()
    def project(self, dataloader: DataLoader) -> FloatArray:
        """Generates an embedding of the data contained in the DataLoader.

        Args:
            dataloader: A DataLoader with categorical or continuous data

        Returns:
            FloatArray: Embedding
        """
        self.eval()
        embedding = []
        for batch in dataloader:
            batch = self._validate_batch(batch)
            *_, mu, _ = self(batch)
            embedding.append(mu)
        embedding = torch.cat(embedding, dim=0).cpu().numpy()
        return embedding

    @torch.no_grad()
    def reconstruct(
        self, dataloader: DataLoader
    ) -> tuple[list[FloatArray], FloatArray]:
        """
        Generates a reconstruction of the data contained in the DataLoader.

        Args:
            dataloader: A DataLoader with categorical or continuous data

        Returns:
            A list of categorical reconstructions and the continuous
            reconstruction
        """
        self.eval()
        cat_recons = [[] for _ in range(len(self.categorical_shapes))]
        con_recons = []
        for batch in dataloader:
            batch = self._validate_batch(batch)
            cat_recon, con_recon, *_ = self(batch)
            if cat_recon is not None:
                for i, cat in enumerate(cat_recon):
                    cat_recons[i].append(torch.argmax(cat, dim=1))
            if con_recon is not None:
                con_recons.append(con_recon)
        if cat_recons:
            cat_recons = [torch.cat(cats, dim=0).cpu().numpy() for cats in cat_recons]
        if con_recons:
            con_recons = torch.cat(con_recons, dim=0).cpu().numpy()
        return cat_recons, con_recons

    @torch.no_grad()
    def latent(
        self, dataloader: DataLoader, kld_weight: float
    ) -> tuple[FloatArray, FloatArray, IntArray, IntArray, FloatArray, float, float]:
        """
        Iterate through validation or test dataset

        Args:
            dataloader: Dataloader with test dataset
            kld_weight: KLD weight

        Returns:
            (tuple): a tuple containing:
                latent: array of VAE latent space mean vectors values
                latent_var: array of VAE latent space logvar vectors values
                cat_recon: reconstructed categorical data
                cat_class: input categorical data
                con_recon: reconstructions of continuous data
                test_loss: total loss on test set
                test_likelihood: total likelihood on test set
        """

        self.eval()
        test_loss = 0
        test_likelihood = 0

        num_samples = dataloader.dataset.num_samples

        latent = torch.empty((num_samples, self.num_latent))
        latent_var = torch.empty((num_samples, self.num_latent))

        # reconstructed output
        if self.num_categorical > 0:
            cat_class, cat_recon, cat_total_shape = self.make_cat_recon_out(num_samples)
        else:
            cat_class = None
            cat_recon = None

        con_recon = (
            None
            if self.num_continuous == 0
            else torch.empty((num_samples, self.num_continuous))
        )

        row = 0
        for (cat, con) in dataloader:
            cat = cat.to(self.device)
            con = con.to(self.device)

            # get dataset
            if self.num_categorical > 0 and self.num_continuous > 0:
                tensor = torch.cat((cat, con), 1)
            elif self.num_categorical > 0:
                tensor = cat
            elif self.num_continuous > 0:
                tensor = con
            else:
                assert False, "Must have at least 1 categorial or 1 continuous feature"

            # Evaluate
            cat_out, con_out, mu, logvar = self(tensor)

            mu = mu.to(self.device)
            logvar = logvar.to(self.device)
            batch = len(mu)

            loss, bce, sse, _ = self.loss_function(
                cat, cat_out, con, con_out, mu, logvar, kld_weight
            )
            test_likelihood += bce + sse
            test_loss += loss.data.item()

            if self.num_categorical > 0:
                cat_out_class, cat_target = self.get_cat_recon(
                    batch, cat_total_shape, cat, cat_out
                )
                cat_recon[row : row + len(cat_out_class)] = torch.Tensor(cat_out_class)
                cat_class[row : row + len(cat_target)] = torch.Tensor(cat_target)

            if self.num_continuous > 0:
                con_recon[row : row + len(con_out)] = con_out

            latent_var[row : row + len(logvar)] = logvar
            latent[row : row + len(mu)] = mu
            row += len(mu)

        test_loss /= len(dataloader)
        logger.info("====> Test set loss: {:.4f}".format(test_loss))

        latent = latent.numpy()
        latent_var = latent_var.numpy()
        cat_recon = cat_recon.numpy()
        cat_class = cat_class.numpy()
        con_recon = con_recon.numpy()

        assert row == num_samples
        return (
            latent,
            latent_var,
            cat_recon,
            cat_class,
            con_recon,
            test_loss,
            test_likelihood,
        )

    def __repr__(self) -> str:
        return (
            f"VAE ({self.input_size} ⇄ {' ⇄ '.join(map(str, self.num_hidden))}"
            f" ⇄ {self.num_latent})"
        )
