__all__ = ["train_model"]

import copy

import hydra
import torch
from torch.utils.data import DataLoader

from move.conf.schema import MOVEConfig
from move.data.dataloaders import make_dataloader
from move.data.io import read_data
from move.models.vae import VAE
from move.training.seed import set_global_seed


@hydra.main(config_path="../conf", config_name="main")
def train_model(config: MOVEConfig):

    if config.seed is not None:
        set_global_seed(config.seed)

    device = torch.device("cuda" if config.training.cuda == True else "cpu")

    cat_list, _, con_list, _ = read_data(config)

    # Making the dataloader
    _, train_loader = make_dataloader(
        cat_list=cat_list, con_list=con_list, batchsize=10
    )  # Added drop_last

    # Make model
    model: VAE = hydra.utils.instantiate(
        config.model,
        continuous_shapes=train_loader.dataset.con_shapes,
        categorical_shapes=train_loader.dataset.cat_shapes,
        _convert_="partial",
    ).to(device)

    kld_w = 0.0
    r = 20 / len(config.training.kld_steps)
    update = 1 + r

    # Lists for saving the results
    losses = list()
    ce = list()
    sse = list()
    KLD = list()

    # Training the model
    for epoch in range(1, config.training.num_epochs + 1):

        if epoch in config.training.kld_steps:
            kld_w = 1 / 20 * update
            update += r

        if epoch in config.training.batch_steps:
            train_loader = DataLoader(
                dataset=train_loader.dataset,
                batch_size=int(
                    train_loader.batch_size * 1.25
                ),  # TODOs whhy train_loader bigger
                shuffle=True,
                drop_last=False,  # Added
                num_workers=train_loader.num_workers,
                pin_memory=train_loader.pin_memory,
            )

        l, c, s, k = model.encoding(train_loader, epoch, config.training.lr, kld_w)

        losses.append(l)
        ce.append(c)
        sse.append(s)
        KLD.append(k)

        best_model = copy.deepcopy(model)

    return best_model, losses, ce, sse, KLD


if __name__ == "__main__":
    train_model()
