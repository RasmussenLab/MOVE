__all__ = ["identify_associations_multiprocess"]


from functools import reduce
from os.path import exists
from pathlib import Path
from typing import Literal, Sized, cast

import hydra
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
from omegaconf import OmegaConf
from torch.multiprocessing import Pool

from move.conf.schema import (
    IdentifyAssociationsBayesConfig,
    IdentifyAssociationsConfig,
    IdentifyAssociationsKSConfig,
    IdentifyAssociationsTTestConfig,
    MOVEConfig,
)
from move.core.logging import get_logger
from move.data import io
from move.data.dataloaders import MOVEDataset, make_dataloader

# Later, change this to import the _one options from these files,
# rather than defining them here
# CHANGE
from move.data.perturbations import (
    ContinuousPerturbationType,
    perturb_continuous_data_extended_one,
)
from move.models.vae import VAE
from move.visualization.dataset_distributions import (
    plot_feature_association_graph,
)

# from scipy.stats import ks_2samp, pearsonr  # type: ignore


# from move.visualization.dataset_distributions import plot_value_distributions


# from move.analysis.metrics import get_2nd_order_polynomial


# make_dataloader creates a dataloader with both continuous and categorical datasets


# We can do three types of statistical tests. Multiprocessing is only implemented
# for bayes at the moment
TaskType = Literal["bayes", "ttest", "ks"]

# Possible values for continuous pertrubation
CONTINUOUS_TARGET_VALUE = ["minimum", "maximum", "plus_std", "minus_std"]


def _get_task_type(
    task_config: IdentifyAssociationsConfig,
) -> TaskType:
    task_type = OmegaConf.get_type(task_config)
    if task_type is IdentifyAssociationsBayesConfig:
        return "bayes"
    if task_type is IdentifyAssociationsTTestConfig:
        return "ttest"
    if task_type is IdentifyAssociationsKSConfig:
        return "ks"
    raise ValueError("Unsupported type of task!")


def _validate_task_config(
    task_config: IdentifyAssociationsConfig, task_type: TaskType
) -> None:
    if not (0.0 <= task_config.sig_threshold <= 1.0):
        raise ValueError("Significance threshold must be within [0, 1].")
    if task_type == "ttest":
        task_config = cast(IdentifyAssociationsTTestConfig, task_config)
        if len(task_config.num_latent) != 4:
            raise ValueError("4 latent space dimensions required.")


# def _worker_process(queue, args):
#   result = _bayes_approach_worker(args)
#  queue.put(result)


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
    ) = args
    # Initialize logging
    logger = get_logger(__name__)
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
    bayes_k_worker = np.zeros(
        (num_continuous)
    )  # This will be what we will put in bayes_k[i,:]
    # Set the normalizer
    normalizer = (
        1 / task_config.num_refits
    )  # Divide by the number of refits. All the refits will have the same importance

    # Create perturbed dataloader for the current feature (i)
    logger.debug(f"Creating perturbed dataloader for feature {i}")
    perturbed_dataloader = perturb_continuous_data_extended_one(
        baseline_dataloader=baseline_dataloader,
        con_dataset_names=config.data.continuous_names,
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
            logger.debug(
                f"Loading baseline reconstruction from {reconstruction_path}, "
                "in the worker function"
            )
            baseline_recon = torch.load(reconstruction_path)

        logger.debug(f"Loading model {model_path}, using load function")
        model: VAE = hydra.utils.instantiate(
            task_config.model,
            continuous_shapes=continuous_shapes,
            categorical_shapes=categorical_shapes,
        )
        device = torch.device("cuda" if task_config.model.cuda == True else "cpu")
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
        diff = perturb_recon - baseline_recon
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
    diff_mask = np.ma.masked_array(mean_diff, mask=nan_mask)
    diff_mask_shape = diff_mask.shape
    logger.debug(
        f"Calculated diff_masked for feature {i}. Its shape is {diff_mask_shape}"
    )
    prob = np.ma.compressed(np.mean(diff_mask > 1e-8, axis=0))
    logger.debug(f"prob calculated for feature {i}. Starting to calculate bayes_k")

    # The code below is to save results, in case we want to compare results
    # data = diff_mask.data  # Extract the data from the masked array
    # mask = diff_mask.mask  # Extract the mask from the masked array
    # Replace masked values with a placeholder (e.g., np.nan)
    # data[mask] = np.nan
    # Define the file path to save the TSV file
    # output_path = Path(config.data.results_path
    # ) / "identify_associations_multiprocess"
    # file_path = output_path / "diff_multi.tsv"
    # Save the data to the TSV fil
    # logger.debug(f"Saving diff to {file_path}")
    # np.savetxt(file_path, diff_mask, delimiter='\t')

    # data = prob.data  # Extract the data from the masked array
    # mask = prob.mask  # Extract the mask from the masked array
    # Replace masked values with a placeholder (e.g., np.nan)
    # data[mask] = np.nan
    # Define the file path to save the TSV file
    # output_path = Path(config.data.results_path) / "identify_associations_selected"
    # file_path = output_path / "prob_multi_script.tsv"
    # Save the data to the TSV fil
    # logger.debug(f"Saving prob to {file_path}")
    # np.savetxt(file_path, prob, delimiter='\t')
    # logger.debug(f"prob is {prob}")

    # Calculate bayes factor
    bayes_k_worker = np.log(prob + 1e-8) - np.log(1 - prob + 1e-8)
    # Code below is to save file
    # file_path = output_path / f"bayes_k_multi_worker_{i}.tsv"
    # logger.debug(f"Saving bayes_k_worker {i} to {file_path}")
    # np.savetxt(file_path, bayes_k_worker, delimiter='\t')

    logger.debug(
        f"bayes factor calculated for feature {i}. Woker function {i} finished"
    )

    # Return bayes_k_worker and the index of the feature
    return i, bayes_k_worker


def _bayes_approach_parallel(
    config,
    task_config,
    train_dataloader,
    baseline_dataloader,
    num_perturbed,
    num_samples,
    num_continuous,
    models_path,
    # indexes,
):
    logger = get_logger(__name__)
    logger.debug("Inside the bayes_parallel function")

    # First, I train or reload the models (number of refits), and save the baseline reconstruction.
    # We train and get the reconstruction outside to make sure that we use the same model and use the same
    # baseline reconstruction for all the worker functions
    baseline_dataset = cast(MOVEDataset, baseline_dataloader.dataset)

    assert task_config.model is not None
    device = torch.device("cuda" if task_config.model.cuda == True else "cpu")
    logger.debug("Model moved to device in bayes_approach_parallel")

    # Train or reload models
    logger = get_logger(__name__)
    logger.info("Training or reloading models")

    for j in range(
        task_config.num_refits
    ):  # We create as many models (refits) as indicated in the config file
        # For each j (number of refits) we train a different model, but on the same data
        # Initialize model
        model: VAE = hydra.utils.instantiate(
            task_config.model,
            continuous_shapes=baseline_dataset.con_shapes,
            categorical_shapes=baseline_dataset.cat_shapes,
        )
        if (
            j == 0
        ):  # First, we see if the models are already created (if we trained them
            # before). for each j, we check if model number j has already been created.
            logger.debug(f"Model: {model}")

        # Define paths for the baseline reconstruction and for the model
        reconstruction_path = (
            models_path / f"baseline_recon_{task_config.model.num_latent}_{j}.pt"
        )
        model_path = models_path / f"model_{task_config.model.num_latent}_{j}.pt"

        if (
            model_path.exists()
        ):  # If the models were already created, we load them only if we need to get a
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
        else:  # If the models are not created yet, he have to train them, with the
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

        # Calculate baseline reconstruction
        # For each model j, we get a different reconstruction for the baseline.
        # We haven't perturbed anything yet, we are just
        # getting the reconstruction for the baseline, to make sure that we get
        # the same reconstruction for each refit, we cannot
        # do it inside each process because the results might be different

        if reconstruction_path.exists():
            logger.debug(f"Baseline reconstruction for model {j} already created")
        else:
            model.eval()
            _, baseline_recon = model.reconstruct(baseline_dataloader)

            # Save the baseline reconstruction for each model
            logger.debug(f"Saving baseline reconstruction {j}")
            torch.save(baseline_recon, reconstruction_path, pickle_protocol=4)
            logger.debug(f"Saved baseline reconstruction {j}")
            del model

    # Get NaN mask ouside the worker function, it is the same for all.
    logger.debug("Creating NaN mask in bayes_parallel")
    orig_con = (
        baseline_dataset.con_all
    )  # Original continuous data of the baseline dataset (all data)
    nan_mask = (
        orig_con == 0
    ).numpy()  # Creates a mask to identify NaN values in the original data.

    # Define more arguments that are needed for the worker functions
    continuous_shapes = baseline_dataset.con_shapes
    categorical_shapes = baseline_dataset.cat_shapes

    """
    Perform parallelized bayes approach.
    """
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
        )
        for i in range(num_perturbed)
    ]
    # for i in range(100)] # Possible to change to less perturbations if needed

    # logger.debug(f"Arguments for workers are {args}")

    # Create a Pool with multiprocessing.cpu_count() - 1 processes
    # with Pool(processes=torch.multiprocessing.cpu_count() - 1, maxtasksperchild=1
    # ) as pool:
    with Pool(processes=torch.multiprocessing.cpu_count() - 1) as pool:
        logger.debug("Inside the pool loops")
        # Map worker function to arguments
        # We get the bayes_k matrix, filled for all the perturbed features
        results = pool.map(_bayes_approach_worker, args)

    logger.info("Pool multiprocess completed. Calculating bayes_abs and bayes_p")

    bayes_k = np.empty((num_perturbed, num_continuous))

    # Get results in the correct order
    for i, computed_bayes_k in results:
        logger.debug(f"{i} has bayes_k worker {computed_bayes_k}")
        bayes_k[i, :] = computed_bayes_k

    # output_path = Path(config.data.results_path
    # ) / "identify_associations_multiprocess"
    # file_path = output_path / "bayes_k_multi_all.tsv"
    # ogger.debug(f"Saving bayes_k (all, not the worker) to {file_path}")
    # np.savetxt(file_path, bayes_k, delimiter='\t')

    # Once we have the Bayes factors for all features, we can calculate Bayes
    # probabilities
    bayes_abs = np.abs(bayes_k)  # Dimensions are (num_perturbed, num_continuous)
    bayes_max = np.max(bayes_abs)
    bayes_min = np.min(bayes_abs)
    bayes_abs_shape = bayes_abs.shape
    logger.debug(
        f"bayes_abs max is {bayes_max}. Bayes_abs min is {bayes_min}. "
        f"Bayes_abs shape is {bayes_abs_shape}"
    )
    # file_path = output_path / "bayes_abs_multi.tsv"
    # logger.debug(f"Saving bayes_abs to {file_path}")
    # np.savetxt(file_path, bayes_abs, delimiter='\t')

    bayes_p = np.exp(bayes_abs) / (
        1 + np.exp(bayes_abs)
    )  # 2D: N x C (perturbed features as rows, all continuous features as columns)
    bayes_p_shape = bayes_p.shape
    logger.debug(f"bayes_p shape is {bayes_p_shape}")
    # file_path = output_path / "bayes_p_multi.tsv"
    # logger.debug(f"Saving bayes_p to {file_path}")
    # np.savetxt(file_path, bayes_p, delimiter='\t')

    # NOTE_ : I AM SKIPPING THE MASK STEP, SO I WILL HAVE TO REMOVE
    # FEATURE I - FEATURE I ASSOCIATIONS LATER, in the results

    # Get only the significant associations:
    # This will flatten the array,so we get all bayes_abs for all perturbed features
    # vs all continuous features in one 1D array
    # Then, we sort them, and get the indexes in the flattened array. So, we get an
    # list of sorted indexes in the flatenned array
    sort_ids = np.argsort(bayes_abs, axis=None)[::-1]  # 1D: N x C
    # file_path = output_path / "sort_ids_multi.tsv"
    logger.debug(f"sort_ids are {sort_ids}")
    # logger.debug(f"Saving sort_ids to {file_path}")
    # np.savetxt(file_path, sort_ids, delimiter='\t')

    prob = np.take(bayes_p, sort_ids)  # 1D: N x C
    # file_path = output_path / "prob_final_multi.tsv"
    # logger.debug(f"Saving prob to {file_path}")
    # np.savetxt(file_path, prob, delimiter='\t')
    # bayes_p is the array from which elements will be taken.
    # sort_ids contains the indices that determine the order in which elements should
    # be taken from bayes_p.
    # This operation essentially rearranges the elements of bayes_p based on the
    # sorting order specified by sort_ids
    # np.take considers the input array as if it were flattened when extracting
    # elements using the provided indices.
    # So, even though sort_ids is obtained from a flattened version of bayes_abs,
    # np.take understands how to map these indices
    # correctly to the original shape of bayes_p. We get a flattened array?
    logger.debug(f"prob is {prob}")
    logger.debug(f"Bayes proba range: [{prob[-1]:.3f} {prob[0]:.3f}]")

    # Sort bayes_k in descending order, aligning with the sorted bayes_abs.
    bayes_k = np.take(bayes_k, sort_ids)  # 1D: N x C
    logger.debug(f"bayes k, after sorting with the ids, is {bayes_k}")
    # file_path = output_path / "sorted_bayes_k_multi.tsv"
    # logger.debug(f"Saving sorted_bayes_k to {file_path}")
    # np.savetxt(file_path, bayes_k, delimiter='\t')

    logger.debug("Calculating fdr")
    # Calculate FDR
    fdr = np.cumsum(1 - prob) / np.arange(1, prob.size + 1)  # 1D ???
    logger.debug(f"fdr is {fdr}")
    logger.debug(f"FDR range: [{fdr[0]:.3f} {fdr[-1]:.3f}]")
    # file_path = output_path / "fdr_multi.tsv"
    # logger.debug(f"Saving fdr to {file_path}")
    # np.savetxt(file_path, fdr, delimiter='\t')

    idx = np.argmin(np.abs(fdr - task_config.sig_threshold))
    # file_path = output_path / "idx_multi.tsv"
    # logger.debug(f"Saving idx to {file_path}")
    # np.savetxt(file_path, idx, delimiter='\t')
    logger.debug(f"Index is {idx}")
    # idx will contain the index of the element in fdr that is closest
    # to task_config.sig_threshold.
    # This line essentially finds the index where the False Discovery Rate (fdr) is
    # closest to the significance threshold
    # (task_config.sig_threshold).

    logger.debug(f"Sig ids sorted is {sort_ids}")
    logger.debug(f"Sigids[idx] is {sort_ids[:idx]}")

    # Return elements only up to idx. They will be the significant findings
    return sort_ids[:idx], prob[:idx], fdr[:idx], bayes_k[:idx]
    # sort_ids[:idx]: Indices of features sorted by significance.
    # prob[:idx]: Probabilities of significant associations for selected features.
    # fdr[:idx]: False Discovery Rate values for selected features.
    # bayes_k[:idx]: Bayes Factors indicating the strength of evidence for selected
    # associations.


def save_results(
    config: MOVEConfig,
    con_shapes: list[int],
    cat_names: list[list[str]],
    con_names: list[list[str]],
    output_path: Path,
    sig_ids,
    extra_cols,
    extra_colnames,
) -> None:
    """
    This function saves the obtained associations in a TSV file containing
    the following columns:
        feature_a_id
        feature_b_id
        feature_a_name
        feature_b_name
        feature_b_dataset
        proba/p_value: number quantifying the significance of the association

    Args:
        config: main config
        con_shapes: tuple with the number of features per continuous dataset
        cat_names: list of lists of names for the categorical features.
                   Each inner list corresponds to a separate dataset.
        con_names: list of lists of names for the continuous features.
                   Each inner list corresponds to a separate dataset.
        output_path: path where the results will be saved
        sig_ids: ids for the significat features
        extra_cols: extra data when calling the approach function
        extra_colnames: names for the extra data columns
    """
    logger = get_logger(__name__)
    logger.info(f"Significant hits found: {sig_ids.size}")
    task_config = cast(IdentifyAssociationsConfig, config.task)
    task_type = _get_task_type(task_config)
    pert_type = task_config.pert_type

    num_continuous = sum(con_shapes)  # C

    if sig_ids.size > 0:
        sig_ids = np.vstack((sig_ids // num_continuous, sig_ids % num_continuous)).T
        # sig_pairs = [np.vstack((sig_id // num_continuous, sig_id % num_continuous))
        # for sig_id in sig_ids]

        logger.info("Writing results")
        results = pd.DataFrame(sig_ids, columns=["feature_a_id", "feature_b_id"])

        # Check if the task is for continuous or categorical data
        if task_config.target_value in CONTINUOUS_TARGET_VALUE:
            target_dataset_idx = config.data.continuous_names.index(
                task_config.target_dataset
            )
            # This creates a DataFrame named a_df with one column
            # named "feature_a_name". The values in this column are
            # taken from con_names using the target_dataset_idx index.
            a_df = pd.DataFrame(dict(feature_a_name=con_names[target_dataset_idx]))
        else:
            target_dataset_idx = config.data.categorical_names.index(
                task_config.target_dataset
            )

            a_df = pd.DataFrame(dict(feature_a_name=cat_names[target_dataset_idx]))
        a_df.index.name = "feature_a_id"
        a_df.reset_index(inplace=True)
        feature_names = reduce(list.__add__, con_names)
        b_df = pd.DataFrame(dict(feature_b_name=feature_names))
        b_df.index.name = "feature_b_id"
        b_df.reset_index(inplace=True)
        results = results.merge(a_df, on="feature_a_id", how="left").merge(
            b_df, on="feature_b_id", how="left"
        )
        results["feature_b_dataset"] = pd.cut(
            results["feature_b_id"],
            bins=cast(list[int], np.cumsum([0] + con_shapes)),
            right=False,
            labels=config.data.continuous_names,
        )
        for col, colname in zip(extra_cols, extra_colnames):
            results[colname] = col
        results.to_csv(
            output_path
            / f"results_sig_assoc_{task_type}_multiprocess_all_{pert_type}.tsv",
            sep="\t",
            index=False,
        )


def identify_associations_multiprocess(config: MOVEConfig) -> None:
    """
    Leads to the execution of the appropriate association
    identification tasks. The function is organized in three
    blocks:
        1) Prepare the data and create the dataloaders with their masks.
        2) Evaluate associations using bayes or ttest approach.
        3) Save results.
    """
    #################### DATA PREPARATION ######################
    ####### Read original data and create perturbed datasets####

    logger = get_logger(__name__)
    task_config = cast(IdentifyAssociationsConfig, config.task)
    task_type = _get_task_type(task_config)
    _validate_task_config(task_config, task_type)

    interim_path = Path(config.data.interim_data_path)

    models_path = interim_path / "models"
    if task_config.save_refits:
        models_path.mkdir(exist_ok=True)

    output_path = Path(config.data.results_path) / "identify_associations_multiprocess"
    output_path.mkdir(exist_ok=True, parents=True)

    # Load datasets:
    cat_list, cat_names, con_list, con_names = io.load_preprocessed_data(
        interim_path,
        config.data.categorical_names,
        config.data.continuous_names,
    )

    logger.debug(
        "Making train dataloader in main function identify_associations_selected"
    )
    train_dataloader = make_dataloader(
        cat_list,  # List of categorical datasets
        con_list,  # List of continuous datasets
        shuffle=True,
        batch_size=task_config.batch_size,
        drop_last=True,
    )

    con_shapes = [con.shape[1] for con in con_list]

    num_samples = len(cast(Sized, train_dataloader.sampler))  # N
    num_continuous = sum(con_shapes)  # C
    logger.debug(f"# continuous features: {num_continuous}")

    # Creating the baseline dataloader:
    baseline_dataloader = make_dataloader(
        cat_list, con_list, shuffle=False, batch_size=task_config.batch_size
    )
    logger.debug(f"Baseline dataloader created")

    # Indentify associations between continuous features:
    logger.info(f"Perturbing dataset: '{task_config.target_dataset}'")
    if task_config.target_value in CONTINUOUS_TARGET_VALUE:
        logger.info(f"Beginning task: identify associations continuous ({task_type})")
        logger.info(f"Perturbation type: {task_config.target_value}")
        # output_subpath = Path(output_path) / "perturbation_visualization"
        # output_subpath.mkdir(exist_ok=True, parents=True)
    # Identify associations between categorical and continuous features:
    else:
        logger.info("Beginning task: identify associations categorical")
        (
            dataloaders,
            nan_mask,
            feature_mask,
        ) = prepare_for_categorical_perturbation(
            config, interim_path, baseline_dataloader, cat_list
        )

    baseline_dataset = baseline_dataloader.dataset
    con_shapes = [con.shape[1] for con in con_list]
    logger.debug(f"con_shapes is {con_shapes}")
    target_dataset_name = task_config.target_dataset
    con_dataset_names = config.data.continuous_names
    target_idx = con_dataset_names.index(target_dataset_name)
    logger.debug(f"Target idx is {target_idx}")

    num_perturbed = baseline_dataset.con_shapes[target_idx]
    logger.debug(f"# perturbed features: {num_perturbed}")

    ################# APPROACH EVALUATION ##########################

    if task_type == "bayes":
        task_config = cast(IdentifyAssociationsBayesConfig, task_config)
        sig_ids, *extra_cols = _bayes_approach_parallel(
            config,
            task_config,
            train_dataloader,
            baseline_dataloader,
            # dataloaders, I will create it inside
            num_perturbed,
            num_samples,
            num_continuous,
            models_path,
            # indexes,
        )
        logger.debug(
            "Completed bayes task (parallel function in main function "
            "(identify_associations_selected))"
        )

        extra_colnames = ["proba", "fdr", "bayes_k"]

    elif task_type == "ttest":
        task_config = cast(IdentifyAssociationsTTestConfig, task_config)
        sig_ids, *extra_cols = _ttest_approach(
            config,
            task_config,
            train_dataloader,
            baseline_dataloader,
            # dataloaders, I will create it inside
            models_path,
            interim_path,
            num_perturbed,
            num_samples,
            num_continuous,
            nan_mask,
            feature_mask,
        )

        extra_colnames = ["p_value"]

    elif task_type == "ks":
        task_config = cast(IdentifyAssociationsKSConfig, task_config)
        sig_ids, *extra_cols = _ks_approach(
            config,
            task_config,
            train_dataloader,
            baseline_dataloader,
            dataloaders,
            models_path,
            num_perturbed,
            num_samples,
            num_continuous,
            con_names,
            output_path,
        )

        extra_colnames = ["ks_distance"]

    else:
        raise ValueError()

    ###################### RESULTS ################################
    save_results(
        config,
        con_shapes,
        cat_names,
        con_names,
        output_path,
        sig_ids,
        extra_cols,
        extra_colnames,
    )
    logger.debug("Results tsv created")
    pert_type = task_config.pert_type
    logger.debug(f"pert type is {pert_type}")

    if exists(
        output_path / f"results_sig_assoc_{task_type}_multiprocess_all_{pert_type}.tsv"
    ):
        association_df = pd.read_csv(
            output_path
            / f"results_sig_assoc_{task_type}_multiprocess_all_{pert_type}.tsv",
            sep="\t",
        )
        plot_feature_association_graph(association_df, output_path)
        plot_feature_association_graph(association_df, output_path, layout="spring")
