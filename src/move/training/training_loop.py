
from typing import Optional

from torch.utils.data import DataLoader

from move.models.vae import VAE

TrainingLoopOutput = tuple[list[float], list[float], list[float], list[float], float]


def dilate_batch(dataloader: DataLoader) -> DataLoader:
    """Increase the batch size of a dataloader."""
    assert dataloader.batch_size is not None
    dataset = dataloader.dataset
    batch_size = int(dataloader.batch_size * 1.5)
    return DataLoader(dataset, batch_size, shuffle=True, drop_last=True)


def training_loop(
    model: VAE,
    train_dataloader: DataLoader,
    valid_dataloader: Optional[DataLoader]=None,
    lr:float=1e-4,
    num_epochs: int=100,
    batch_dilation_steps: list[int]=[],
    kld_warmup_steps: list[int]=[],
    early_stopping: bool=False,
    patience: int=0
) -> TrainingLoopOutput:
    """Trains a VAE model with batch dilation and KLD warm-up. Optionally,
    enforce early stopping."""

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

        for i, output in enumerate(model.encoding(train_dataloader, epoch, lr, kld_weight)):
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
