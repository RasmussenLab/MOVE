from typing import Optional

from torch.utils.data import DataLoader

from move.models.vae import VAE

TrainingLoopOutput = tuple[list[float], list[float], list[float], list[float], float]


def dilate_batch(dataloader: DataLoader) -> DataLoader:
    """
    Increase the batch size of a dataloader.

    Args:
        dataloader (DataLoader): An object feeding data to the VAE

    Returns:
        DataLoader: An object feeding data to the VAE
    """
    assert dataloader.batch_size is not None
    dataset = dataloader.dataset
    batch_size = int(dataloader.batch_size * 1.5)
    return DataLoader(dataset, batch_size, shuffle=True, drop_last=True)


BATCH_DILATION_STEPS = []
KLD_WARMUP_STEPS = []


def training_loop(
    model: VAE,
    train_dataloader: DataLoader,
    valid_dataloader: Optional[DataLoader] = None,
    lr: float = 1e-4,
    num_epochs: int = 100,
    batch_dilation_steps: list[int] = BATCH_DILATION_STEPS,
    kld_warmup_steps: list[int] = KLD_WARMUP_STEPS,
    early_stopping: bool = False,
    patience: int = 0,
) -> TrainingLoopOutput:
    """
    Trains a VAE model with batch dilation and KLD warm-up. Optionally,
    enforce early stopping.

    Args:
        model (VAE): trained VAE model object
        train_dataloader (DataLoader):  An object feeding data to the VAE with training data
        valid_dataloader (Optional[DataLoader], optional): An object feeding data to the VAE with validation data.
                                                           Defaults to None.
        lr (float, optional): learning rate. Defaults to 1e-4.
        num_epochs (int, optional): number of epochs. Defaults to 100.
        batch_dilation_steps (list[int], optional): a list with integers corresponding to epochs when batch size is
                                                    increased. Defaults to [].
        kld_warmup_steps (list[int], optional):  a list with integers corresponding to epochs when kld is decreased by
                                                 the selected rate. Defaults to [].
        early_stopping (bool, optional):  boolean if use early stopping . Defaults to False.
        patience (int, optional): number of epochs to wait before early stop if no progress on the validation set.
                                  Defaults to 0.

    Returns:
        (tuple): a tuple containing:
            *outputs (*list): lists containing information of epoch loss, BCE loss, SSE loss, KLD loss
            kld_weight (float): final KLD after dilations during the training
    """

    outputs = [[] for _ in range(4)]
    min_likelihood = float("inf")
    counter = 0

    kld_weight = 0.0
    kld_rate = 20 / len(kld_warmup_steps)
    kld_multiplier = 1 + kld_rate

    for epoch in range(1, num_epochs + 1):
        if epoch in kld_warmup_steps:
            kld_weight = 0.05 * kld_multiplier
            kld_multiplier += kld_rate

        if epoch in batch_dilation_steps:
            train_dataloader = dilate_batch(train_dataloader)

        for i, output in enumerate(
            model.encoding(train_dataloader, epoch, lr, kld_weight)
        ):
            outputs[i].append(output)

        if early_stopping and valid_dataloader is not None:
            output = model.latent(valid_dataloader, kld_weight)
            valid_likelihood = output[-1]
            if valid_likelihood > min_likelihood and counter < patience:
                counter += 1
                if counter % 5 == 0:
                    lr *= 0.9
            elif counter == patience:
                break
            else:
                min_likelihood = valid_likelihood
                counter = 0

    return *outputs, kld_weight
