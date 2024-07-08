from pathlib import Path
from typing import Literal, Union, cast

import hydra
import numpy as np
import torch
import torch.multiprocessing
from torch.multiprocessing import Pool
from torch.utils.data import DataLoader

from move.conf.schema import IdentifyAssociationsBayesConfig, MOVEConfig
from move.core.logging import get_logger
from move.core.typing import BoolArray, FloatArray, IntArray
from move.data.dataloaders import MOVEDataset
from move.data.perturbations import (
    ContinuousPerturbationType,
    perturb_continuous_data_extended_one,
)
from move.models.vae import VAE

# We can do three types of statistical tests. Multiprocessing is only implemented
# for bayes at the moment
TaskType = Literal["bayes", "ttest", "ks"]

# Possible values for continuous pertrubation
CONTINUOUS_TARGET_VALUE = ["minimum", "maximum", "plus_std", "minus_std"]

logger = get_logger(__name__)


def _bayes_approach_worker(args):
    """
    Worker function to calculate mean differences and Bayes factors for one feature.
    """
    # Set the number of threads available:
    # VERY IMPORTANT, TO AVOID CPU OVERSUBSCRIPTION
    torch.set_num_threads(1)

    # Unpack arguments.
    (
        config,
        task_config,
        baseline_dataloader,
        num_samples,
        num_continuous,
        i,
        models_path,
        continuous_shapes,
        categorical_shapes,
        nan_mask,
        feature_mask,
    ) = args
    # Initialize logging
    logger.debug(f"Inside the worker function for num_perturbed {i}")

    # Now we are inside the num_perturbed loop, we will do this for each of the
    # perturbed features
    # Now, mean_diff will not have a first dimension for num_perturbed, because we will
    # not store it for each perturbed feature
    # we will use it in each loop for calculating the bayes factors, and then overwrite
    # its content with a new perturbed feature
    # mean_diff will contain the differences between the baseline and the perturbed
    # reconstruction for feature i, taking into account
    # all refits (all refits have the same importance)
    # We also set up bayes_k, which has the same dimensions as mean_diff
    mean_diff = np.zeros((num_samples, num_continuous))
    # Set the normalizer
    # Divide by the number of refits. All the refits will have the same importance
    normalizer = 1 / task_config.num_refits

    # Create perturbed dataloader for the current feature (i)
    logger.debug(f"Creating perturbed dataloader for feature {i}")
    perturbed_dataloader = perturb_continuous_data_extended_one(
        baseline_dataloader=baseline_dataloader,
        con_dataset_names=config.data.categorical_names,  # ! error: continuous_names
        target_dataset_name=task_config.target_dataset,
        perturbation_type=cast(ContinuousPerturbationType, task_config.target_value),
        index_pert_feat=i,
    )
    logger.debug(f"created perturbed dataloader for feature {i}")

    # For each refit, reload baseline reconstruction (obtained in bayes_parallel
    # function). Also, get the reconstruction for the perturbed dataloader
    for j in range(task_config.num_refits):

        model_path = models_path / f"model_{task_config.model.num_latent}_{j}.pt"
        reconstruction_path = (
            models_path / f"baseline_recon_{task_config.model.num_latent}_{j}.pt"
        )
        if reconstruction_path.exists():
            logger.debug(f"Loading baseline reconstruction from {reconstruction_path}.")
            baseline_recon = torch.load(reconstruction_path)
        else:
            raise FileNotFoundError("Baseline reconstruction not found.")

        logger.debug(f"Loading model {model_path}, using load function")
        model: VAE = hydra.utils.instantiate(
            task_config.model,
            continuous_shapes=continuous_shapes,
            categorical_shapes=categorical_shapes,
        )
        device = torch.device("cuda" if task_config.model.cuda else "cpu")
        logger.debug(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path))
        logger.debug(f"Loaded model from {model_path}")
        model.to(device)
        model.eval()

        logger.debug(f"Reconstructing num_perturbed {i}, with model {model_path}")
        _, perturb_recon = model.reconstruct(
            perturbed_dataloader
        )  # Instead of dataloaders[i], create the perturbed one here and
        # use it only here
        logger.debug(
            f"Perturbed reconstruction succesful for feature {i}, model {model}"
        )

        # diff is a matrix with the same dimensions as perturb_recon and baseline_recon
        # (rows are samples and columns all the continuous features)
        # We calculate diff for each refit, and add it to mean_diff after dividing by
        # the number of refits
        logger.debug(f"Calculating diff for num_perturbed {i}, with model {model}")
        diff = perturb_recon - baseline_recon  # 2D: N x C
        logger.debug(
            f"Calculating mean_diff  for num_perturbed {i}, with model {model}"
        )
        mean_diff += diff * normalizer
        logger.debug(f"Deleting model {model_path}, to see if I can free up space?")
        del model
        logger.debug(f"Deleted model {model_path} in worker {i} to save some space")

    logger.debug(f"mean_diff for feature {i}, calculated, using all refits")
    mean_diff_shape = mean_diff.shape
    logger.debug(f"Returning mean_diff for feature {i}. Its shape is {mean_diff_shape}")

    # Apply nan_mask to the result in mean_diff
    mask = feature_mask | nan_mask  # 2D: N x C
    diff = np.ma.masked_array(mean_diff, mask=mask)
    diff_shape = diff.shape
    logger.debug(f"Calculated diff (masked) for feature {i}. Its shape is {diff_shape}")
    prob = np.ma.compressed(np.mean(diff > 1e-8, axis=0))
    logger.debug(f"prob calculated for feature {i}. Starting to calculate bayes_k")

    # Calculate bayes factor
    bayes_k = np.log(prob + 1e-8) - np.log(1 - prob + 1e-8)

    # Marc's masking approach (for subset of perturbed cont. features?)
    # difference for only perturbed feature?
    bayes_mask = np.zeros(np.shape(bayes_k.shape))
    if task_config.target_value in CONTINUOUS_TARGET_VALUE:
        bayes_mask = (
            baseline_dataloader.dataset.con_all[0, :]
            - perturbed_dataloader.dataset.con_all[0, :]
        )

    logger.debug(
        f"bayes factor calculated for feature {i}. Woker function {i} finished"
    )

    # Return bayes_k and the index of the feature
    return i, bayes_k, bayes_mask


def _bayes_approach_parallel(
    config: MOVEConfig,
    task_config: IdentifyAssociationsBayesConfig,
    train_dataloader: DataLoader,
    baseline_dataloader: DataLoader,
    models_path: Path,
    num_perturbed: int,
    num_samples: int,
    num_continuous: int,
    nan_mask: BoolArray,
    feature_mask: BoolArray,
) -> tuple[Union[IntArray, FloatArray], ...]:
    """
    Calculate Bayes factors for all perturbed features in parallel.

    First, I train or reload the models (number of refits), and save the baseline
    reconstruction. We train and get the reconstruction outside to make sure
    that we use the same model and use the same baseline reconstruction for all
    the worker functions.
    """
    logger.debug("Inside the bayes_parallel function")

    assert task_config.model is not None
    device = torch.device("cuda" if task_config.model.cuda else "cpu")

    # Train or reload models
    logger.info("Training or reloading models")
    # non-perturbed baseline dataset
    baseline_dataset = cast(MOVEDataset, baseline_dataloader.dataset)

    for j in range(task_config.num_refits):
        # We create as many models (refits) as indicated in the config file
        # For each j (number of refits) we train a different model, but on the same data
        # Initialize model
        model: VAE = hydra.utils.instantiate(
            task_config.model,
            continuous_shapes=baseline_dataset.con_shapes,
            categorical_shapes=baseline_dataset.cat_shapes,
        )
        if j == 0:
            # First, we see if the models are already created (if we trained them
            # before). for each j, we check if model number j has already been created.
            logger.debug(f"Model: {model}")

        # Define paths for the baseline reconstruction and for the model
        reconstruction_path = (
            models_path / f"baseline_recon_{task_config.model.num_latent}_{j}.pt"
        )
        model_path = models_path / f"model_{task_config.model.num_latent}_{j}.pt"

        if model_path.exists():
            # If the models were already created, we load them only if we need to get a
            # baseline reconstruction. Otherwise, nothing needs to be done at this point
            logger.debug(f"Model {model_path} already exists")
            if not reconstruction_path.exists():
                logger.debug(f"Re-loading refit {j + 1}/{task_config.num_refits}")
                model.load_state_dict(torch.load(model_path))
                model.to(device)
                logger.debug(f"Model {j} reloaded")
            else:
                logger.debug(
                    f"Baseline reconstruction for {reconstruction_path} already exists"
                    f", no need to load model {model_path} "
                )
        else:
            # If the models are not created yet, he have to train them, with the
            # parameters we indicated in the config file
            logger.debug(f"Training refit {j + 1}/{task_config.num_refits}")
            model.to(device)
            hydra.utils.call(
                task_config.training_loop,
                model=model,
                train_dataloader=train_dataloader,
            )
            # Save the refits, to use them later
            if task_config.save_refits:
                # pickle_protocol=4 is necessary for very big models
                torch.save(model.state_dict(), model_path, pickle_protocol=4)
        model.eval()

        # Calculate baseline reconstruction
        # For each model j, we get a different reconstruction for the baseline.
        # We haven't perturbed anything yet, we are just
        # getting the reconstruction for the baseline, to make sure that we get
        # the same reconstruction for each refit, we cannot
        # do it inside each process because the results might be different
        # ! here the logic is a bit off. If the reconstruction path exist, the model
        # ! does not to be loaded again.

        if reconstruction_path.exists():
            logger.debug(
                f"Loading baseline reconstruction from {reconstruction_path}, "
                "in the worker function"
            )
            # baseline_recon = torch.load(reconstruction_path)
        else:
            _, baseline_recon = model.reconstruct(baseline_dataloader)

            # Save the baseline reconstruction for each model
            logger.debug(f"Saving baseline reconstruction {j}")
            torch.save(baseline_recon, reconstruction_path, pickle_protocol=4)
            logger.debug(f"Saved baseline reconstruction {j}")
            del model

    # Calculate Bayes factors
    logger.info("Identifying significant features")

    # Define more arguments that are needed for the worker functions
    continuous_shapes = baseline_dataset.con_shapes
    categorical_shapes = baseline_dataset.cat_shapes

    logger.debug("Starting parallelization")

    # Define arguments for each worker, and iterate over models and perturbed features
    args = [
        (
            config,
            task_config,
            baseline_dataloader,
            num_samples,
            num_continuous,
            i,
            models_path,
            continuous_shapes,
            categorical_shapes,
            nan_mask,
            feature_mask[:, [i]],
        )
        for i in range(num_perturbed)
    ]

    with Pool(processes=torch.multiprocessing.cpu_count() - 1) as pool:
        logger.debug("Inside the pool loops")
        # Map worker function to arguments
        # We get the bayes_k matrix, filled for all the perturbed features
        results = pool.map(_bayes_approach_worker, args)

    logger.info("Pool multiprocess completed. Calculating bayes_abs and bayes_p")

    bayes_k = np.empty((num_perturbed, num_continuous))
    bayes_mask = np.zeros(np.shape(bayes_k))
    # Get results in the correct order
    for i, computed_bayes_k, mask_k in results:
        logger.debug(f"{i} has bayes_k worker {computed_bayes_k}")
        # computed_bayes_k: already normalized probability
        # (log differences, i.e. Bayes factors)
        bayes_k[i, :] = computed_bayes_k
        bayes_mask[i, :] = mask_k
    bayes_mask[bayes_mask != 0] = 1
    bayes_mask = np.array(bayes_mask, dtype=bool)

    # Calculate Bayes probabilities
    bayes_abs = np.abs(bayes_k)  # Dimensions are (num_perturbed, num_continuous)

    bayes_p = np.exp(bayes_abs) / (1 + np.exp(bayes_abs))  # 2D: N x C

    bayes_abs[bayes_mask] = np.min(
        bayes_abs
    )  # Bring feature_i feature_i associations to minimum
    # Get only the significant associations:
    # This will flatten the array, so we get all bayes_abs for all perturbed features
    # vs all continuous features in one 1D array
    # Then, we sort them, and get the indexes in the flattened array. So, we get an
    # list of sorted indexes in the flatenned array
    sort_ids = np.argsort(bayes_abs, axis=None)[::-1]  # 1D: N x C
    logger.debug(f"sort_ids are {sort_ids}")
    # bayes_p is the array from which elements will be taken.
    # sort_ids contains the indices that determine the order in which elements should
    # be taken from bayes_p.
    # This operation essentially rearranges the elements of bayes_p based on the
    # sorting order specified by sort_ids
    # np.take considers the input array as if it were flattened when extracting
    # elements using the provided indices.
    # So, even though sort_ids is obtained from a flattened version of bayes_abs,
    # np.take understands how to map these indices
    # correctly to the original shape of bayes_p.
    prob = np.take(bayes_p, sort_ids)  # 1D: N x C
    logger.debug(f"Bayes proba range: [{prob[-1]:.3f} {prob[0]:.3f}]")

    # Sort bayes_k in descending order, aligning with the sorted bayes_abs.
    bayes_k = np.take(bayes_k, sort_ids)  # 1D: N x C

    # Calculate FDR
    fdr = np.cumsum(1 - prob) / np.arange(1, prob.size + 1)  # 1D
    idx = np.argmin(np.abs(fdr - task_config.sig_threshold))
    # idx will contain the index of the element in fdr that is closest
    # to task_config.sig_threshold.
    # This line essentially finds the index where the False Discovery Rate (fdr) is
    # closest to the significance threshold
    # (task_config.sig_threshold).
    logger.debug(f"Index is {idx}")
    logger.debug(f"FDR range: [{fdr[0]:.3f} {fdr[-1]:.3f}]")

    # Return elements only up to idx. They will be the significant findings
    # sort_ids[:idx]: Indices of features sorted by significance.
    # prob[:idx]: Probabilities of significant associations for selected features.
    # fdr[:idx]: False Discovery Rate values for selected features.
    # bayes_k[:idx]: Bayes Factors indicating the strength of evidence for selected
    # associations.
    return sort_ids[:idx], prob[:idx], fdr[:idx], bayes_k[:idx]
