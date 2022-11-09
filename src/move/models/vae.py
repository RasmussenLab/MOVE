__all__ = ["VAE"]

import logging
from typing import Optional

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from move.core.typing import FloatArray

logger = logging.getLogger("vae.py")


class VAE(nn.Module):
    """
    Variational autoencoder.

    Args:
        categorical_shapes (Optional[list[tuple]], optional): shape of the different continuous datasets if any. Defaults to None.
        continuous_shapes (Optional[list[tuple]], optional): shape of the different categorical datasets if any. Defaults to None.
        categorical_weights (Optional[list[int]], optional): list of weights for each categorical dataset. Defaults to None.
        continuous_weights (Optional[list[int]], optional): list of weights for each continuous dataset. Defaults to None.
        num_hidden (list[int], optional): List of n_neurons in the hidden layers. Defaults to [200, 200].
        num_latent (int, optional): Number of neurons in the latent layer. Defaults to 20.
        beta (float, optional): Multiply KLD by the inverse of this value. Defaults to 0.01.
        dropout (float, optional): Probability of dropout on forward pass. Defaults to 0.2.
        cuda (bool, optional): Use CUDA (GPU accelerated training). Defaults to False.

    Raises:
        ValueError: Minimum 1 latent unit
        ValueError: Beta must be greater than zero.
        ValueError: Dropout must be between zero and one.
        ValueError: Shapes of the input data must be provided.
        ValueError: Number of continuous weights must be the same as number of continuous datasets
        ValueError: Number of categorical weights must be the same as number of categorical datasets
    """    


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

    vae.trainmodel(dataloader, nepochs batchsteps, lrate, logfile, modelfile)
        Trains the model, returning None

    vae.encode(self, data_loader):
        Encodes the data in the data loader and returns the encoded matrix.
    """

    def __init__(
        self,
        categorical_shapes: Optional[list[tuple]] = None,
        continuous_shapes: Optional[list[tuple]] = None,
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

        num_categorical = sum([int.__mul__(*shape[1:]) for shape in categorical_shapes])
        num_continuous = sum(continuous_shapes)

        self.input_size = 0
        if not (num_continuous is None or continuous_shapes is None):
            self.num_continuous = num_continuous
            self.input_size += self.num_continuous
            self.continuous_shapes = continuous_shapes

            if not (continuous_weights is None):
                self.continuous_weights = continuous_weights
                if not len(continuous_shapes) == len(continuous_weights):
                    raise ValueError(
                        "Number of continuous weights must be the same as"
                        " number of continuous datasets"
                    )
        else:
            self.num_continuous = None

        if not (num_categorical is None or categorical_shapes is None):
            self.num_categorical = num_categorical
            self.input_size += self.num_categorical
            self.categorical_shapes = categorical_shapes

            if not (categorical_weights is None):
                self.categorical_weights = categorical_weights
                if not len(categorical_shapes) == len(categorical_weights):
                    raise ValueError(
                        "Number of categorical weights must be the same as"
                        " number of categorical datasets"
                    )
        else:
            self.num_categorical = None

        super(VAE, self).__init__()

        # Initialize simple attributes
        self.beta = beta
        self.num_hidden = num_hidden
        self.num_latent = num_latent
        self.dropout = dropout

        self.device = torch.device("cuda" if cuda == True else "cpu")

        # Activation functions
        self.relu = nn.LeakyReLU()
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
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

    def encode(self, x):
        """
        Encodes the data in the data loader and returns the encoded matrix.

        Args:
            x (torch.Tensor): input data

        Returns:
            (tuple): a tuple containing:
                (torch.Tensor): mean latent vector 
                (torch.Tensor): mean log-variance vector
        """        
        # Hidden layers
        for encoderlayer, encodernorm in zip(self.encoderlayers, self.encodernorms):
            x = encoderlayer(x)
            x = self.relu(x)
            x = self.dropoutlayer(x)
            x = encodernorm(x)

        return self.mu(x), self.var(x)

    def reparameterize(self, mu, logvar):
        """
        Performs reparametrization trick

        Args:
            mu (torch.Tensor): mean latent vector 
            logvar (torch.Tensor): mean log-variance vector

        Returns:
            torch.Tensor: sample from latent space distribution
        """        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps.mul(std).add_(mu)

    def decompose_categorical(self, reconstruction):
        """
        Returns list of final reconstructions (after applying 
        log-softmax to the outputs of decoder) of each categorical class

        Args:
            reconstruction (torch.Tensor): results of final layer of the decoder

        Returns:
            list: list of final reconstructions of each categorical class
        """        
        cat_tmp = reconstruction.narrow(1, 0, self.num_categorical)

        # handle soft max for each categorical dataset
        cat_out = []
        pos = 0
        for cat_shape in self.categorical_shapes:
            cat_dataset = cat_tmp[:, pos : (cat_shape[1] * cat_shape[2] + pos)]

            cat_out_tmp = cat_dataset.view(
                cat_dataset.shape[0], cat_shape[1], cat_shape[2]
            )
            cat_out_tmp = cat_out_tmp.transpose(1, 2)
            cat_out_tmp = self.log_softmax(cat_out_tmp)

            cat_out.append(cat_out_tmp)
            pos += cat_shape[1] * cat_shape[2]

        return cat_out

    def decode(self, x):
        """
        _summary_

        Args:
            x (torch.Tensor): sample from latent space distribution

        Returns:
            (tuple): a tuple containing:
                cat_out (list): list of reconstructions of every categorical data class
                con_out (torch.Tensor): reconstruction of continuous data 
        """        
        for decoderlayer, decodernorm in zip(self.decoderlayers, self.decodernorms):
            x = decoderlayer(x)
            x = self.relu(x)
            x = self.dropoutlayer(x)
            x = decodernorm(x)

        reconstruction = self.out(x)

        # Decompose reconstruction to categorical and continuous variables
        # if both types are in the input
        if not (self.num_categorical is None or self.num_continuous is None):
            cat_out = self.decompose_categorical(reconstruction)
            con_out = reconstruction.narrow(
                1, self.num_categorical, self.num_continuous
            )
        elif not (self.num_categorical is None):
            cat_out = self.decompose_categorical(reconstruction)
            con_out = None
        elif not (self.num_continuous is None):
            cat_out = None
            con_out = reconstruction.narrow(1, 0, self.num_continuous)

        return cat_out, con_out

    def forward(self, tensor):
        """
        Forward propagate through the VAE network

        Args:
            tensor (torch.Tensor): input data

        Returns:
            (tuple): a tuple containing:
                cat_out (list): list of reconstructions of every categorical data class
                con_out (torch.Tensor): reconstructions of continuous data  
                mu (torch.Tensor): mean latent vector 
                logvar (torch.Tensor): mean log-variance vector
        """        
        mu, logvar = self.encode(tensor)
        z = self.reparameterize(mu, logvar)
        cat_out, con_out = self.decode(z)

        return cat_out, con_out, mu, logvar

    def calculate_cat_error(self, cat_in, cat_out):
        """
        Calulates errors (cross-entropy) for categorical data reconstructions

        Args:
            cat_in (torch.Tensor): input categorical data
            cat_out (list): list of reconstructions of every categorical data class

        Returns:
            torch.Tensor: Errors (cross-entropy) for categorical data reconstructions
        """        
        batch_size = cat_in.shape[0]

        # calcualte target values for all cat datasets
        count = 0
        cat_errors = []
        pos = 0
        for cat_shape in self.categorical_shapes:
            cat_dataset = cat_in[:, pos : (cat_shape[1] * cat_shape[2] + pos)]

            cat_dataset = cat_dataset.view(cat_in.shape[0], cat_shape[1], cat_shape[2])
            cat_target = cat_dataset
            cat_target = cat_target.argmax(2)
            cat_target[cat_dataset.sum(dim=2) == 0] = -1
            cat_target = cat_target.to(self.device)

            # Cross entropy loss for categroical
            loss = nn.NLLLoss(reduction="sum", ignore_index=-1)
            cat_errors.append(
                loss(cat_out[count], cat_target) / (batch_size * cat_shape[1])
            )
            count += 1
            pos += cat_shape[1] * cat_shape[2]

        cat_errors = torch.stack(cat_errors)
        return cat_errors

    def calculate_con_error(self, con_in, con_out, loss):
        """
        Calulates errors (MSE) for continuous data reconstructions

        Args:
            con_in (torch.tensor): input continuous data
            con_out (torch.tensor): reconstructions of continuous data
            loss (torch.nn.modules.loss): loss function

        Returns:
            MSE (torch.Tensor): loss
        """        
        batch_size = con_in.shape[0]
        total_shape = 0
        con_errors = []
        for s in self.continuous_shapes:
            c_in = con_in[:, total_shape : (s + total_shape - 1)]
            c_re = con_out[:, total_shape : (s + total_shape - 1)]
            error = loss(c_re, c_in) / batch_size
            con_errors.append(error)
            total_shape += s

        con_errors = torch.stack(con_errors)
        con_errors = con_errors / torch.Tensor(self.continuous_shapes).to(self.device)
        MSE = torch.sum(
            con_errors * torch.Tensor(self.continuous_weights).to(self.device)
        )
        return MSE

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, cat_in, cat_out, con_in, con_out, mu, logvar, kld_w):
        """
        Calculates the loss for data reconstructions

        Args:
            cat_in (torch.Tensor): input categorical data
            cat_out (list): list of reconstructions of every categorical data class
            con_in (torch.Tensor): input continuous data
            con_out (torch.Tensor): reconstructions of continuous data
            mu (torch.Tensor): mean latent vector 
            logvar (torch.Tensor): mean log-variance vector
            kld_w (float): kld weight

        Returns:
            (tuple): a tuple containing:
                torch.Tensor: total loss on train set during the training of the epoch
                torch.Tensor: BCE loss on train set during the training of the epoch
                torch.Tensor: SSE loss on train set during the training of the epoch
                torch.Tensor: KLD loss on train set during the training of the epoch
        """       

        MSE = 0
        CE = 0
        # calculate loss for catecorical data if in the input
        if not (cat_out is None):
            cat_errors = self.calculate_cat_error(cat_in, cat_out)
            if not (self.categorical_weights is None):
                CE = torch.sum(
                    cat_errors * torch.Tensor(self.categorical_weights).to(self.device)
                )
            else:
                CE = torch.sum(cat_errors) / len(cat_errors)

        # calculate loss for continuous data if in the input
        if not (con_out is None):
            batch_size = con_in.shape[0]
            # Mean square error loss for continauous
            loss = nn.MSELoss(reduction="sum")
            # set missing data to 0 to remove any loss these would provide
            con_out[con_in == 0] == 0

            # include different weights for each omics dataset
            if not (self.continuous_weights is None):
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

    def encoding(self, train_loader, epoch, lrate, kld_w):
        """
        One iteration of VAE

        Args:
            train_loader (Dataloader): Dataloader with train dataset
            epoch (int): the epoch
            lrate (float): learning rate for the model
            kld_w (float): float of KLD weight

        Returns:
            (tuple): a tuple containing:
                torch.Tensor: total loss on train set during the training of the epoch
                torch.Tensor: BCE loss on train set during the training of the epoch
                torch.Tensor: SSE loss on train set during the training of the epoch
                torch.Tensor: KLD loss on train set during the training of the epoch
        """        
        self.train()
        train_loss = 0
        log_interval = 50

        optimizer = optim.Adam(self.parameters(), lr=lrate)

        epoch_loss = 0
        epoch_kldloss = 0
        epoch_sseloss = 0
        epoch_bceloss = 0

        for batch_idx, (cat, con) in enumerate(train_loader):
            # Move input to GPU if requested
            cat = cat.to(self.device)
            con = con.to(self.device)

            if not (self.num_categorical is None or self.num_continuous is None):
                tensor = torch.cat((cat, con), 1)
            elif not (self.num_categorical is None):
                tensor = cat
            elif not (self.num_continuous is None):
                tensor = con

            optimizer.zero_grad()

            cat_out, con_out, mu, logvar = self(tensor)

            loss, bce, sse, kld = self.loss_function(
                cat, cat_out, con, con_out, mu, logvar, kld_w
            )
            loss.backward()

            epoch_loss += loss.data.item()
            epoch_kldloss += kld.data.item()

            if not (self.num_continuous is None):
                epoch_sseloss += sse.data.item()

            if not (self.num_categorical is None):
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

    def make_cat_recon_out(self, length):
        """
        Initiate empty tensors for categorical data

        Args:
            length (int): number of samples

        Returns:
            (tuple): a tuple containing:
                cat_class (torch.Tensor): empty tensor for input categorical data 
                cat_recon (torch.Tensor): empty tensor for reconstructed categorical data
                cat_total_shape (int): number of features of linearized one hot categorical data    
        """        
        cat_total_shape = 0
        for cat_shape in self.categorical_shapes:
            cat_total_shape += cat_shape[1]

        cat_class = torch.empty((length, cat_total_shape)).int()
        cat_recon = torch.empty((length, cat_total_shape)).int()
        return cat_class, cat_recon, cat_total_shape

    def get_cat_recon(self, batch, cat_total_shape, cat, cat_out):
        """
        Generates categorical input and reconstruction data

        Args:
            batch (int): number of samples in the batch
            cat_total_shape (int): number of features of linearized one hot categorical data 
            cat (torch.Tensor): input categorical data
            cat_out (torch.Tensor): reconstructed categorical data

        Returns:
            (tuple): a tuple containing:
                cat_out_class (np.array): reconstructed categorical data 
                cat_target (np.array): input categorical data 
        """        
        count = 0
        cat_out_class = torch.empty((batch, cat_total_shape)).int()
        cat_target = torch.empty((batch, cat_total_shape)).int()
        pos = 0
        shape_1 = 0
        for cat_shape in self.categorical_shapes:
            # Get input categorical data
            cat_in_tmp = cat[:, pos : (cat_shape[1] * cat_shape[2] + pos)]
            cat_in_tmp = cat_in_tmp.view(cat.shape[0], cat_shape[1], cat_shape[2])

            # Calculate target values for input
            cat_target_tmp = cat_in_tmp
            cat_target_tmp = torch.argmax(cat_target_tmp.detach(), dim=2)
            cat_target_tmp[cat_in_tmp.sum(dim=2) == 0] = -1
            cat_target[
                :, shape_1 : (cat_shape[1] + shape_1)
            ] = cat_target_tmp  # .numpy()

            # Get reconstructed categorical data
            cat_out_tmp = cat_out[count]
            cat_out_tmp = cat_out_tmp.transpose(1, 2)
            cat_out_class[:, shape_1 : (cat_shape[1] + shape_1)] = torch.argmax(
                cat_out_tmp, dim=2
            )  # .numpy()

            # make counts for next dataset
            pos += cat_shape[1] * cat_shape[2]
            shape_1 += cat_shape[1]
            count += 1

        cat_target = cat_target.numpy()
        cat_out_class = cat_out_class.numpy()

        return cat_out_class, cat_target

    def _validate_batch(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Returns the batch of categorical and continuous data if they are not None

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): batches of categorical and continuous data 

        Returns:
            torch.Tensor: a formed batch 
        """        
        cat, con = batch
        if self.num_categorical is None:
            return con
        elif self.num_continuous is None:
            return cat
        return torch.cat(batch, dim=1)

    @torch.no_grad()
    def project(self, dataloader: DataLoader) -> FloatArray:
        """Generates an embedding of the data contained in the DataLoader.

        Args:
            dataloader (DataLoader): A DataLoader with categorical or continuous data

        Returns:
            FloatArray: Embedding
        """    
        self.eval()
        embedding = []
        for batch in dataloader:
            batch = self._validate_batch(batch)
            *_, mu, _ = self(batch)
            embedding.append(mu)
        embedding = torch.cat(embedding, dim=0).numpy()
        return embedding

    @torch.no_grad()
    def reconstruct(
        self, dataloader: DataLoader
    ) -> tuple[list[FloatArray], FloatArray]:
        """
        Generates a reconstruction of the data contained in the DataLoader.

        Args:
            dataloader (DataLoader): A DataLoader with categorical or continuous data

        Returns:
            tuple[list[FloatArray], FloatArray]: A list of categorical reconstructions and the continuous
            reconstruction
        """    
        self.eval()
        cat_recons = [[] for _ in range(len(self.categorical_shapes))]
        con_recons = []
        for batch in dataloader:
            batch = self._validate_batch(batch)
            cat_recon, con_recon, *_ = self(batch)
            for i, cat in enumerate(cat_recon):
                cat_recons[i].append(torch.argmax(cat, dim=1))
            con_recons.append(con_recon)
        cat_recons = [torch.cat(cats, dim=0).numpy() for cats in cat_recons]
        con_recons = torch.cat(con_recons, dim=0).numpy()
        return cat_recons, con_recons

    @torch.no_grad()
    def latent(self, dataloader: DataLoader, kld_weight: float):
        """
        Iterate through validation or test dataset

        Args:
            dataloader (DataLoader): Dataloader with test dataset
            kld_weight (float): KLD weight

        Returns:
            (tuple): a tuple containing:   
                latent (np.array):
                latent_var (np.array): array of VAE latent space mean vectors values
                cat_recon (np.array): array of VAE latent space logvar vectors values
                cat_class (np.array): input categorical data 
                con_recon (np.array): reconstructions of continuous data
                test_loss (float): total loss on test set
                test_likelihood (float): total likelihood on test set
        """        

        self.eval()
        test_loss = 0
        test_likelihood = 0

        num_samples = dataloader.dataset.num_samples

        latent = torch.empty((num_samples, self.num_latent))
        latent_var = torch.empty((num_samples, self.num_latent))

        # reconstructed output
        if not (self.num_categorical is None):
            cat_class, cat_recon, cat_total_shape = self.make_cat_recon_out(num_samples)
        else:
            cat_class = None
            cat_recon = None

        con_recon = (
            None
            if self.num_continuous is None
            else torch.empty((num_samples, self.num_continuous))
        )

        row = 0
        for (cat, con) in dataloader:
            cat = cat.to(self.device)
            con = con.to(self.device)

            # get dataset
            if not (self.num_categorical is None or self.num_continuous is None):
                tensor = torch.cat((cat, con), 1)
            elif not (self.num_categorical is None):
                tensor = cat
            elif not (self.num_continuous is None):
                tensor = con

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

            if not (self.num_categorical is None):
                cat_out_class, cat_target = self.get_cat_recon(
                    batch, cat_total_shape, cat, cat_out
                )
                cat_recon[row : row + len(cat_out_class)] = torch.Tensor(cat_out_class)
                cat_class[row : row + len(cat_target)] = torch.Tensor(cat_target)

            if not (self.num_continuous is None):
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
