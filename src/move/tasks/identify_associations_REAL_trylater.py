__all__ = ["identify_associations"]

from functools import reduce
from os.path import exists
from pathlib import Path
from typing import Literal, Sized, Union, cast, Optional
from move.data.preprocessing import feature_stats
from move.visualization.dataset_distributions import plot_value_distributions
ContinuousPerturbationType = Literal["minimum", "maximum", "plus_std", "minus_std"]


import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from scipy.stats import ks_2samp, pearsonr  # type: ignore
from torch.utils.data import DataLoader

from move.analysis.metrics import get_2nd_order_polynomial

from move.conf.schema import (
    IdentifyAssociationsBayesConfig,
    IdentifyAssociationsConfig,
    IdentifyAssociationsKSConfig,
    IdentifyAssociationsTTestConfig,
    MOVEConfig,
)
from move.core.logging import get_logger
from move.core.typing import BoolArray, FloatArray, IntArray
from move.data import io

from move.data.dataloaders import MOVEDataset, make_dataloader # Make dataloader with both continuous and categorical datasets

from move.data.perturbations import (
    ContinuousPerturbationType,
    perturb_categorical_data,
    perturb_continuous_data_extended,
) # We input functions for perturbing either categorical data or continuous data.
# perturb_continuous_data_extended takes as input:
#    - baseline_dataloader: Baseline dataloader. It is the whole original dataloader, with all the continuous and categorical features
#    - con_dataset_names: List of continuous dataset names
#    - target_dataset_name: Target continuous dataset to perturb
#    - perturbation_type: 'minimum', 'maximum', 'plus_std' or 'minus_std'.
#    - output_subpath: path where the figure showing the perturbation will be saved
# And returns:
#  - List of dataloaders containing all perturbed datasets
#  - Plot of the feature value distribution after the perturbation. All perturbations are collapsed into one single plot.

# Instead of this, We are going to use a function that returns only one perturbed dataloader, not all of them. It is going to be
# perturbed_continuous_extended_one :

'''
def perturb_continuous_data_extended_one( # We will keep the input almost the same, to make everything easier
    # However, I have to introduce a variable that allows me to index the specific dataloader I want to create (index_pert_feat)
    # And I eliminate the output directory, because I am not going to save any image
    baseline_dataloader: DataLoader,
    con_dataset_names: list[str],
    target_dataset_name: str,
    perturbation_type: ContinuousPerturbationType,
    index_pert_feat: int,
) -> DataLoader: # But we change the output from list[DataLoader] to just one DataLoader

    """Add perturbations to continuous data. For each feature in the target
    dataset, change the feature's value in all samples (in rows):
    1,2) substituting this feature in all samples by the feature's minimum/maximum value.
    3,4) Adding/Substracting one standard deviation to the sample's feature value.

    Args:
        baseline_dataloader: Baseline dataloader
        con_dataset_names: List of continuous dataset names
        target_dataset_name: Target continuous dataset to perturb
        perturbation_type: 'minimum', 'maximum', 'plus_std' or 'minus_std'.
        output_subpath: path where the figure showing the perturbation will be saved

    Returns:
        - Dataloader with the ith feature (index_pert_feat) perturbed.
        - Plot of the feature value distribution after the perturbation. Note that
          all perturbations are collapsed into one single plot.

    Note:
        This function was created so that it could generalize to non-normalized
        datasets. Scaling is done per dataset, not per feature -> slightly different stds
        feature to feature.
    """

    baseline_dataset = cast(MOVEDataset, baseline_dataloader.dataset)
    assert baseline_dataset.con_shapes is not None
    assert baseline_dataset.con_all is not None

    target_idx = con_dataset_names.index(target_dataset_name)  # dataset index
    splits = np.cumsum([0] + baseline_dataset.con_shapes)
    slice_ = slice(*splits[target_idx : target_idx + 2])

    num_features = baseline_dataset.con_shapes[target_idx]
    dataloaders = []
    perturbations_list = []
    # Change below.
    #num_features = 10

    # Now, instead of the for loop that iterates over all the features we want to perturb, we do it only for one feature, the one 
    # indicated in index_pert_feat

    #for i in range(num_features):
    perturbed_con = baseline_dataset.con_all.clone()
    target_dataset = perturbed_con[:, slice_]
    # Change the desired feature value by:
    min_feat_val_list, max_feat_val_list, std_feat_val_list = feature_stats(
        target_dataset
    )
    if perturbation_type == "minimum":
        target_dataset[:, index_pert_feat] = torch.FloatTensor([min_feat_val_list[index_pert_feat]])
    elif perturbation_type == "maximum":
        target_dataset[:, index_pert_feat] = torch.FloatTensor([max_feat_val_list[index_pert_feat]])
    elif perturbation_type == "plus_std":
        target_dataset[:, index_pert_feat] += torch.FloatTensor([std_feat_val_list[index_pert_feat]])
    elif perturbation_type == "minus_std":
        target_dataset[:, index_pert_feat] -= torch.FloatTensor([std_feat_val_list[index_pert_feat]])

    # We used this for a plot I have removed, so no need to use it
    # perturbations_list.append(target_dataset[:, i].numpy())

    perturbed_dataset = MOVEDataset(
        baseline_dataset.cat_all,
        perturbed_con,
        baseline_dataset.cat_shapes,
        baseline_dataset.con_shapes,
    )

    perturbed_dataloader = DataLoader(
        perturbed_dataset,
        shuffle=False,
        batch_size=baseline_dataloader.batch_size,
    )
    #dataloaders.append(perturbed_dataloader)

    return perturbed_dataloader

'''
def perturb_continuous_data_extended_one( # We will keep the input almost the same, to make everything easier
    # However, I have to introduce a variable that allows me to index the specific dataloader I want to create (index_pert_feat)
    # And I eliminate the output directory, because I am not going to save any image
    baseline_dataloader: DataLoader,
    con_dataset_names: list[str],
    target_dataset_name: str,
    perturbation_type: ContinuousPerturbationType,
    index_pert_feat: int,
) -> DataLoader: # But we change the output from list[DataLoader] to just one DataLoader
    logger = get_logger(__name__)
    """Add perturbations to continuous data. For each feature in the target
    dataset, change the feature's value in all samples (in rows):
    1,2) substituting this feature in all samples by the feature's minimum/maximum value.
    3,4) Adding/Substracting one standard deviation to the sample's feature value.

    Args:
        baseline_dataloader: Baseline dataloader
        con_dataset_names: List of continuous dataset names
        target_dataset_name: Target continuous dataset to perturb
        perturbation_type: 'minimum', 'maximum', 'plus_std' or 'minus_std'.
        #output_subpath: path where the figure showing the perturbation will be saved
        index_pert_feat: Index we want to perturb

    Returns:
        - Dataloader with the ith feature (index_pert_feat) perturbed.
        - Plot of the feature value distribution after the perturbation. Note that
          all perturbations are collapsed into one single plot.

    Note:
        This function was created so that it could generalize to non-normalized
        datasets. Scaling is done per dataset, not per feature -> slightly different stds
        feature to feature.
    """
    logger.debug(f"Inside perturb_continuous_data_extended_one for feature {index_pert_feat}")
    baseline_dataset = cast(MOVEDataset, baseline_dataloader.dataset)
    assert baseline_dataset.con_shapes is not None
    assert baseline_dataset.con_all is not None

    target_idx = con_dataset_names.index(target_dataset_name)  # dataset index
    splits = np.cumsum([0] + baseline_dataset.con_shapes)
    slice_ = slice(*splits[target_idx : target_idx + 2])

    # Use it only if we want to perturb all features in the target dataset
    num_features = baseline_dataset.con_shapes[target_idx] 
    # Change below.
    #num_features = 10

    # Now, instead of the for loop that iterates over all the features we want to perturb, we do it only for one feature, the one 
    # indicated in index_pert_feat

    #for i in range(num_features):
    logger.debug(f"Setting up perturbed_con for feature {index_pert_feat}")

    ###################################################################
    # WE MIGHT HAVE A PROBLEM HERE. UP TO HERE, IT RUNS WITHOUT PROBLEM
    ###################################################################
    
    perturbed_con = baseline_dataset.con_all.clone()
    target_dataset = perturbed_con[:, slice_]

    logger.debug(f"Changing to desired perturbation value for feature {index_pert_feat}")
    # Change the desired feature value by:
    min_feat_val_list, max_feat_val_list, std_feat_val_list = feature_stats(
        target_dataset
    )
    if perturbation_type == "minimum":
        target_dataset[:, index_pert_feat] = torch.FloatTensor([min_feat_val_list[index_pert_feat]])
    elif perturbation_type == "maximum":
        target_dataset[:, index_pert_feat] = torch.FloatTensor([max_feat_val_list[index_pert_feat]])
    elif perturbation_type == "plus_std":
        target_dataset[:, index_pert_feat] += torch.FloatTensor([std_feat_val_list[index_pert_feat]])
    elif perturbation_type == "minus_std":
        target_dataset[:, index_pert_feat] -= torch.FloatTensor([std_feat_val_list[index_pert_feat]])
    logger.debug(f"Perturbation succesful for feature {index_pert_feat}")
    # We used this for a plot I have removed, so no need to use it
    # perturbations_list.append(target_dataset[:, i].numpy())

    logger.debug(f"Creating perturbed dataset and dataloader for feature {index_pert_feat}")

    perturbed_dataset = MOVEDataset(
        baseline_dataset.cat_all,
        perturbed_con,
        baseline_dataset.cat_shapes,
        baseline_dataset.con_shapes,
    )

    perturbed_dataloader = DataLoader(
        perturbed_dataset,
        shuffle=False,
        batch_size=baseline_dataloader.batch_size,
    )
    #dataloaders.append(perturbed_dataloader)

    logger.debug(f"Finished perturb_continuous_data_extended_one for feature {index_pert_feat}")

    return perturbed_dataloader


from move.data.preprocessing import one_hot_encode_single
from move.models.vae import VAE
from move.visualization.dataset_distributions import (
    plot_correlations,
    plot_cumulative_distributions,
    plot_feature_association_graph,
    plot_reconstruction_movement,
)

# We can do three types of statistical tests
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


# I don't care about this, because I am not going to to categorical perturbation
def prepare_for_categorical_perturbation(
    config: MOVEConfig,
    interim_path: Path,
    baseline_dataloader: DataLoader,
    cat_list: list[FloatArray],
) -> tuple[list[DataLoader], BoolArray, BoolArray,]:
    """
    This function creates the required dataloaders and masks
    for further categorical association analysis.

    Args:
        config: main configuration file
        interim_path: path where the intermediate outputs are saved
        baseline_dataloader: reference dataloader that will be perturbed
        cat_list: list of arrays with categorical data

    Returns:
        dataloaders: all dataloaders, including baseline appended last.
        nan_mask: mask for Nans
        feature_mask: masks the column for the perturbed feature.
    """

    # Read original data and create perturbed datasets
    task_config = cast(IdentifyAssociationsConfig, config.task)
    logger = get_logger(__name__)

    # Loading mappings:
    mappings = io.load_mappings(interim_path / "mappings.json")
    target_mapping = mappings[task_config.target_dataset]
    target_value = one_hot_encode_single(target_mapping, task_config.target_value)
    logger.debug(
        f"Target value: {task_config.target_value} => {target_value.astype(int)[0]}"
    )

    dataloaders = perturb_categorical_data(
        baseline_dataloader,
        config.data.categorical_names,
        task_config.target_dataset,
        target_value,
    )
    dataloaders.append(baseline_dataloader) # If we only perturb 10 features, we will only have 10 dataloaders, plus the baseline

    baseline_dataset = cast(MOVEDataset, baseline_dataloader.dataset)

    assert baseline_dataset.con_all is not None
    orig_con = baseline_dataset.con_all
    nan_mask = (orig_con == 0).numpy()  # NaN values encoded as 0s
    logger.debug(f"# NaN values: {np.sum(nan_mask)}/{orig_con.numel()}")

    target_dataset_idx = config.data.categorical_names.index(task_config.target_dataset)
    target_dataset = cat_list[target_dataset_idx]
    feature_mask = np.all(target_dataset == target_value, axis=2)  # 2D: N x P
    feature_mask |= np.sum(target_dataset, axis=2) == 0

    return (
        dataloaders,
        nan_mask,
        feature_mask,
    )



# I will also have to change this, so that it only returns one dataloader, not all of them
def prepare_for_continuous_perturbation_one(
    config: MOVEConfig,
    # output_subpath: Path, Remove this
    baseline_dataloader: DataLoader,
    index_pert_feat,
) -> tuple[DataLoader, BoolArray, BoolArray,]: # change list[DataLoader] to only one dataloader
    """
    This function creates the required dataloaders and masks
    for further continuous association analysis.

    Args:
        config:
            main configuration file.
        output_subpath:
            path where the output plots for continuous analysis are saved.
        baseline_dataloader:
            reference dataloader that will be perturbed.

    Returns:
        dataloaders:
            list with all dataloaders, including baseline appended last.
        nan_mask:
            mask for NaNs
        feature_mask:
            same as `nan_mask`, in this case.
    """

    # Read original data and create perturbed datasets
    logger = get_logger(__name__)
    task_config = cast(IdentifyAssociationsConfig, config.task)

    # Change this to create only one perturbed dataloader
    #dataloaders = perturb_continuous_data_extended(
     #   baseline_dataloader,
      #  config.data.continuous_names,
       # task_config.target_dataset,
        #cast(ContinuousPerturbationType, task_config.target_value),
        #output_subpath,
    #)
    #dataloaders.append(baseline_dataloader) # Append the baseline to the list of perturbed dataloaders

    dataloaders = perturb_continuous_data_extended_one(
        baseline_dataloader,
        config.data.continuous_names,
        task_config.target_dataset,
        cast(ContinuousPerturbationType, task_config.target_value),
        index_pert_feat, # I will have to use this in a loop that iterates over all features in the target dataset
    ) # We will get only one dataloader for each iteration of the for loop
    #dataloaders.append(baseline_dataloader) # Add the baseline dataloader to the perturbed dataloader

    # Get the baseline dataset
    baseline_dataset = cast(MOVEDataset, baseline_dataloader.dataset)

    assert baseline_dataset.con_all is not None
    orig_con = baseline_dataset.con_all # Original continuous data of the baseline dataset (all data)
    nan_mask = (orig_con == 0).numpy()  # Creates a mask to identify NaN values in the original data. 
    # In this case, NaN values are encoded as 0s. The expression (orig_con == 0) creates a boolean mask where True indicates 
    # the presence of NaN values, and False indicates non-NaN values. .numpy() converts this boolean mask to a numpy array.
    logger.debug(f"# NaN values: {np.sum(nan_mask)}/{orig_con.numel()}")
    # np.sum(nan_mask) calculates the total number of NaN values using the nan_mask, and orig_con.numel() calculates the total number of elements in the original data.
    feature_mask = nan_mask

    return (dataloaders, nan_mask, feature_mask)


def _bayes_approach(
    config: MOVEConfig,
    task_config: IdentifyAssociationsBayesConfig,
    train_dataloader: DataLoader,
    baseline_dataloader: DataLoader,
    #dataloaders: list[DataLoader], I eliminate this, because I am not going to use it
    num_perturbed: int, # In this case, should it be only 1, or we keep the original? I should be the original
    num_samples: int,
    num_continuous: int,
    nan_mask: BoolArray,
    feature_mask: BoolArray,
    models_path: Path,
    ) -> tuple[Union[IntArray, FloatArray], ...]:
    
    logger = get_logger(__name__)
    logger.debug("Inside _bayes_approach function")

     # First, I train or reload the models (number of refits), and save the baseline reconstruction
    baseline_dataset = cast(MOVEDataset, baseline_dataloader.dataset)
    assert task_config.model is not None
    device = torch.device("cuda" if task_config.model.cuda == True else "cpu")
    logger.debug("Model moved to devide in bayes_approach_parallel")

    # Train models
    logger.info("Training models")

    for j in range(task_config.num_refits): # WE create as many models as indicated in the config file
        # For each j (number of refits) we train a different model, but on the same data
        # Initialize model
        model: VAE = hydra.utils.instantiate(
            task_config.model,
            continuous_shapes=baseline_dataset.con_shapes,
            categorical_shapes=baseline_dataset.cat_shapes,
        )
        if j == 0: # But first, we see if the models are already created (if we trained them before). for each j, we check if
        # model number j has already been created.
            logger.debug(f"Model: {model}")

        # Train/reload model
        model_path = models_path / f"model_{task_config.model.num_latent}_{j}.pt"
        
        if model_path.exists(): # If the models were already created, we load them
            logger.debug(f"Re-loading refit {j + 1}/{task_config.num_refits}")
            model.load_state_dict(torch.load(model_path))
            model.to(device)
            logger.debug(f"Model {j} reloaded")
        else: # Otherwise, he have to train them, with the parameters we indicated in the config file
            logger.debug(f"Training refit {j + 1}/{task_config.num_refits}")
            model.to(device)
            hydra.utils.call(
                task_config.training_loop,
                model=model,
                train_dataloader=train_dataloader,
            )
            if task_config.save_refits:
                torch.save(model.state_dict(), model_path, pickle_protocol=4)
        
        # We do this later, when we reload the models, no need to save this
        
        # Calculate baseline reconstruction
        # For each model j, we get a different reconstruction for the baseline. We haven't perturbed anything yet, we are just
        # getting the reconstruction for the baseline
        #CHANGE HERE, TO GO FASTER, SINCE I ALREADY HAVE THEM
        reconstruction_path = models_path / f"baseline_recon_{task_config.model.num_latent}_{j}.pt"
        if reconstruction_path.exists():
            logger.debug(f"We alredy have baseline reconstruction {reconstruction_path}")
        else:
            model.eval()
            _, baseline_recon = model.reconstruct(baseline_dataloader)

            # MAYBE CHANGE, IF WE CONTINUE INSIDE J LOOP, NO NEED TO SAVE IT AND RELOAD IT AGAIN
            # Save the reconstruction separately. Up to here, it works.
            logger.info(f"Saving baseline reconstruction {j}")
            reconstruction_path = models_path / f"baseline_recon_{task_config.model.num_latent}_{j}.pt"
            torch.save(baseline_recon, reconstruction_path, pickle_protocol=4)
            logger.debug(f"Saved baseline reconstruction {j}")

    
    # Here we will store the important results:
    # The first dimension refers to the index of the perturbed feature, the second to the 
    # number of continuous features
    #indexes = [9, 63, 76, 79, 101, 103, 109, 118, 126, 131]
    indexes = [1027, 1029] 
    num_perturbations = len(indexes)
    #bayes_k = np.empty((num_perturbed, num_continuous))
    bayes_k = np.empty((num_perturbed, num_continuous))

    
    #CHANGES HERE   
    #logger.info("Starting loop over num_perturbed")
    # This only for indexes
    h = 0
    for i in range(num_perturbed):
    #for i in(indexes):
        logger.debug(f"Setting up mean_diff and normalizer for feature {i}")
        # Now we are inside the num_perturbed loop, we will do this for each of the perturbed features
        # Now, mean_diff will not have a first dimension for num_perturbed, because we will not store it for each perturbed feature
        # we will use it in each loop for calculating the bayes factors, and then delete its content and refill it with a new perturbed feature
        mean_diff = np.zeros((num_samples, num_continuous))
        # Set the normalizer
        normalizer = 1 / task_config.num_refits # Divide by the number of refits. All the refits will have the same importance


        for j in range(task_config.num_refits):

            # First, we reload the model we trained and saved before
            model_path = models_path / f"model_{task_config.model.num_latent}_{j}.pt"
            if model_path.exists(): # If the models were already created, we load them
                logger.debug(f"Re-loading refit {j + 1}/{task_config.num_refits} for reconstructing perturbed reconstruction for {i}")
                model.load_state_dict(torch.load(model_path))
                model.to(device)
                model.eval()
                logger.debug(f"model_{task_config.model.num_latent}_{j}.pt loaded and set into evaluation mode")
            else:
                logger.info(f"Error loading model from models_path / model_{task_config.model.num_latent}_{j}.pt")

            # Then, we use that model to get the baseline reconstruction
            # Calculate baseline reconstruction
            # For each model j, we get a different reconstruction for the baseline. We haven't perturbed anything yet, we are just
            # getting the reconstruction for the baseline
            #_, baseline_recon = model.reconstruct(baseline_dataloader)

            # Better if we load the baseline reconstruction, so that it is the same for all perturbations
            reconstruction_path = models_path / f"baseline_recon_{task_config.model.num_latent}_{j}.pt"
            baseline_recon = torch.load(reconstruction_path)
            
            # Then, we create the perturbed dataset and get the reconstruction, using the same model
            logger.debug(f"Creating perturbed dataloader for feature {i}")
            # Now, we get the perturbed dataloader
            # MAYBE CHANGE, I DON'T NEAD TO CREATE NAN MASK HERE, IT IS THE SAME FOR ALL SO i CAN USE THE ONE
            # I CALCULATED BEFORE, AND HERE USE THE PERTURB_ONE FUNCTION, AND THIS WILL BE FASTER
            perturbed_dataloader, nan_mask, feature_mask = prepare_for_continuous_perturbation_one(
                config=config,
                baseline_dataloader = baseline_dataloader,
                index_pert_feat= i
            ) # Like this, I get only one perturbed dataloader, and the nan and feature masks
            logger.debug(f"created perturbed dataloader for feature {i}")

            logger.debug(f"Reconstructing num_perturbed {i}, with model model_{task_config.model.num_latent}_{j}.pt")
            _, perturb_recon = model.reconstruct(perturbed_dataloader) # Instead of dataloaders[i], create the perturbed one here and use it only here
            logger.debug(f"Perturbed reconstruction succesful for feature {i}, model model_{task_config.model.num_latent}_{j}.pt")

            # Finally, we get the mean difference for feature i, taking into account all refits
            logger.debug(f"Calculating diff for num_perturbed {i}, with model model_{task_config.model.num_latent}_{j}.pt")
            diff = perturb_recon - baseline_recon 
            logger.debug(f"Calculating mean_diff  for num_perturbed {i}, with model model_{task_config.model.num_latent}_{j}.pt")
            mean_diff += diff * normalizer
            mean_diff_shape = mean_diff.shape
            logger.debug(f"Returning mean_diff for feature {i}. Its shape is {mean_diff_shape}")
        
        # After the j loop finishes, we will have mean_diff for feature i complete
        # With this, we can calculate the prob
        # prob will be an array with as many elements as number of continuous features, and it contains the probability that that feautre is significant for ith feature (the one we are perturbating now)
        
        diff_mask = np.ma.masked_array(mean_diff, mask=nan_mask)
        diff_mask_shape = diff_mask.shape
        logger.debug(f"Calculated diff_masked for feature {i}. Its shape is {diff_mask_shape}")
        prob = np.ma.compressed(np.mean(diff_mask > 1e-8, axis=0))

        # Now, we calculate the bayes factor
        bayes_k[i, :] = np.log(prob + 1e-8) - np.log(1 - prob + 1e-8)
        h = h+1 # Count over the number of features we are perturbing

    
    # Once we have the Bayes factors for all features, we can calculate Bayes probabilities
    bayes_abs = np.abs(bayes_k)
    bayes_p = np.exp(bayes_abs) / (1 + np.exp(bayes_abs))  # 2D: N x C (perturbed features as rows, all continuous features as columns)

    # NOTE_ : I AM SKIPPING THE MASK STEP, SO I WILL HAVE TO REMOVE FEATURE I - FEATURE I ASSOCIATIONS LATER

    # Get only the significant associations:
    sort_ids = np.argsort(bayes_abs, axis=None)[::-1]  # 1D: N x C
    prob = np.take(bayes_p, sort_ids)  # 1D: N x C
    logger.debug(f"Bayes proba range: [{prob[-1]:.3f} {prob[0]:.3f}]")

    # Sort bayes_k in descending order, aligning with the sorted bayes_abs.
    bayes_k = np.take(bayes_k, sort_ids)  # 1D: N x C

    # Calculate FDR
    fdr = np.cumsum(1 - prob) / np.arange(1, prob.size + 1)  # 1D
    idx = np.argmin(np.abs(fdr - task_config.sig_threshold))
    logger.debug(f"FDR range: [{fdr[0]:.3f} {fdr[-1]:.3f}]")

    return sort_ids[:idx], prob[:idx], fdr[:idx], bayes_k[:idx]
    # sort_ids[:idx]: Indices of features sorted by significance.
    # prob[:idx]: Probabilities of significant associations for selected features.
    # fdr[:idx]: False Discovery Rate values for selected features.
    # bayes_k[:idx]: Bayes Factors indicating the strength of evidence for selected associations.




def _ttest_approach(
    config: MOVEConfig,
    task_config: IdentifyAssociationsTTestConfig,
    train_dataloader: DataLoader,
    baseline_dataloader: DataLoader,
    #dataloaders: list[DataLoader],
    models_path: Path,
    interim_path: Path,
    num_perturbed: int,
    num_samples: int,
    num_continuous: int,
    nan_mask: BoolArray,
    feature_mask: BoolArray,
) -> tuple[Union[IntArray, FloatArray], ...]:

    from scipy.stats import ttest_rel

    assert task_config.model is not None
    device = torch.device("cuda" if task_config.model.cuda == True else "cpu")

    # Train models
    logger = get_logger(__name__)
    logger.info("Training models")

    # The pvalues array will have 4 dimensions: the number of latent feature we are trying, the number of models to train (refits),
    # the number of perturbed features, and the number of all continuous features.
    pvalues = np.empty(
        (
            4, #len(task_config.num_latent),
            10, #task_config.num_refits,
            num_perturbed,
            num_continuous,
        )
    )

    # Last appended dataloader is the baseline
    baseline_dataset = cast(MOVEDataset, baseline_dataloader.dataset)

    for k, num_latent in enumerate(task_config.num_latent): # For all the latent spaces we try (ASK, can we try only one?)
        for j in range(task_config.num_refits): # Go over the different refits

            # As before, we train j models unless we have already trained them before, in which case we just load them
            # Initialize model
            model: VAE = hydra.utils.instantiate(
                task_config.model,
                continuous_shapes=baseline_dataset.con_shapes,
                categorical_shapes=baseline_dataset.cat_shapes,
                num_latent=num_latent,
            )
            if j == 0:
                logger.debug(f"Model: {model}")

            # Train model
            model_path = models_path / f"model_{num_latent}_{j}.pt"
            if model_path.exists():
                logger.debug(f"Re-loading refit {j + 1}/{task_config.num_refits}")
                model.load_state_dict(torch.load(model_path))
                model.to(device)
            else:
                logger.debug(f"Training refit {j + 1}/{task_config.num_refits}")
                model.to(device)
                hydra.utils.call(
                    task_config.training_loop,
                    model=model,
                    train_dataloader=train_dataloader,
                )
                if task_config.save_refits:
                    torch.save(model.state_dict(), model_path, pickle_protocol=4)
            
            # After training or loading the models, we go into evaluation mode.
            model.eval()

            # Get baseline reconstruction and baseline difference
            # First, reconstruct the baseline
            _, baseline_recon = model.reconstruct(baseline_dataloader)

            # I think, here CHANGE 10 for a variable we can change more easily. The first dimension is the number of times we reconstruct a baseline, the second
            # the number of samples, and the third dimension the continuous feature for which we are doing the comparison,
            # So, we get as many first dimensions as features we perturb. In each of them, we have the differences for all the samples and all the continuous features 
            baseline_diff = np.empty((10, num_samples, num_continuous))
            for i in range(10):
                _, recon = model.reconstruct(baseline_dataloader)
                baseline_diff[i, :, :] = recon - baseline_recon
            
            # Here, we get the mean difference between real baseline and baseline reconstruction for each of the features. We see how the reconstruction of the baseline is,
            # taking into account 10 different reconstructions that use the same model?
            baseline_diff = np.mean(baseline_diff, axis=0)  # 2D: N x C
            # And we mask the NaN values, to not use them in the future
            baseline_diff = np.where(nan_mask, np.nan, baseline_diff)

            # T-test between baseline and perturb difference
            # SAME HERE, ASK IF I CAN CREATE THE PERTURBED DATALOADER HERE, ONE AT A TIME, SO THAT I DON'T HAVE TO STORE THE LIST
            for i in range(num_perturbed): 
                #_, perturb_recon = model.reconstruct(dataloaders[i]) # Instead of doing dataloaders[i], I would actually create the dataloader here
                perturbed_dataloader, nan_mask, feature_mask = prepare_for_continuous_perturbation_one(
                    config=config,
                    baseline_dataloader = baseline_dataloader,
                    index_pert_feat= i)
                _, perturb_recon = model.reconstruct(perturbed_dataloader)
                perturb_diff = perturb_recon - baseline_recon
                mask = feature_mask[:, [i]] | nan_mask  # 2D: N x C

            # Before, we calculated the difference between baseline and baseline reconstruction. Now, difference between perturb reconstruction and baseline reconstruction.
            # We do this to make sure that the difference we see is due to perturbation, and not to reconstruction. If we see a difference between perturbed and baseline but
            # it is the same as baseline and baseline reconstruction, it won't be meaningful

                # k  es el number_latent que estamos probando. En el archivo por defecto, prueban con 4 valores distintos, así que aquí habría 4 dimensiones
                # j es el número del refit. En el ejemplo, entrenan 10 modelos para cada número de num_latent
                # i es el número de feature que estamos perturbando
                # La última dimensión corresponde a todos los continuous values
                # What we do is compare, for each perturbed feature, if the difference between perturb_diff and baseline_diff is significant
                _, pvalues[k, j, i, :] = ttest_rel(
                    a=np.where(mask, np.nan, perturb_diff),
                    b=np.where(mask, np.nan, baseline_diff),
                    axis=0,
                    nan_policy="omit",
                )
    # WE are out of all the loops. WE have completed all the info for all the latent spaces and all the models.
    # Correct p-values (Bonferroni)
    pvalues = np.minimum(pvalues * num_continuous, 1.0)
    np.save(interim_path / "pvals.npy", pvalues)

    # Find significant hits. I think we use all latent spaces to compare, and see if we find a certain significant difference
    # in enough of them, and not only in one.
    overlap_thres = task_config.num_refits // 2
    reject = pvalues <= task_config.sig_threshold  # 4D: L x R x P x C
    overlap = reject.sum(axis=1) >= overlap_thres  # 3D: L x P x C
    sig_ids = overlap.sum(axis=0) >= 3  # 2D: P x C
    sig_ids = np.flatnonzero(sig_ids)  # 1D

    # Report median p-value
    masked_pvalues = np.ma.masked_array(pvalues, mask=~reject)  # 4D
    masked_pvalues = np.ma.median(masked_pvalues, axis=1)  # 3D
    masked_pvalues = np.ma.median(masked_pvalues, axis=0)  # 2D
    sig_pvalues = np.ma.compressed(np.take(masked_pvalues, sig_ids))  # 1D

    return sig_ids, sig_pvalues # We get the significant ids and the significant p-values


def _ks_approach(
    config: MOVEConfig,
    task_config: IdentifyAssociationsKSConfig,
    train_dataloader: DataLoader,
    baseline_dataloader: DataLoader,
    dataloaders: list[DataLoader],
    models_path: Path,
    num_perturbed: int,
    num_samples: int,
    num_continuous: int,
    con_names: list[list[str]],
    output_path: Path,
) -> tuple[Union[IntArray, FloatArray], ...]:

    """
    Find associations between continuous features using Kolmogorov-Smirnov distances.
    When perturbing feature A, this function measures the shift of the reconstructed
    distribution for feature B (over samples) from 1) the baseline reconstruction to 2)
    the reconstruction when perturbing A.

    If A and B are related the perturbation of A in the input will lead to a change in
    feature B's reconstruction, that will be measured by KS distance.

    Associations are then ranked according to KS distance (absolute value).


    Args:
        config: MOVE main configuration.
        task_config: IdentifyAssociationsKSConfig configuration.
        train_dataloader: training DataLoader.
        baseline_dataloader: unperturbed DataLoader.
        dataloaders: list of DataLoaders where DataLoader[i] is obtained by perturbing feature i
                     in the target dataset.
        models_path: path to the models.
        num_perturbed: number of perturbed features.
        num_samples: total number of samples
        num_continuous: number of continuous features (all continuous datasets concatenated).
        con_names: list of lists where eah inner list contains the feature names of a specific continuous dataset
        output_path: path where QC summary metrics will be saved.

    Returns:
        sort_ids: list with flattened IDs of the associations above the significance threshold.
        ks_distance: Ordered list with signed KS scores. KS scores quantify the direction and
                     magnitude of the shift in feature B's reconstruction when perturbing feature A.


    !!! Note !!!:

    The sign of the KS score can be misleading: negative sign means positive shift.
    since the cumulative distribution starts growing later and is found below the reference
    (baseline). Hence:
    a) with plus_std, negative sign means a positive correlation.
    b) with minus_std, negative sign means a negative correlation.
    """

    assert task_config.model is not None
    device = torch.device("cuda" if task_config.model.cuda == True else "cpu")
    figure_path = output_path / "figures"
    figure_path.mkdir(exist_ok=True, parents=True)

    # Data containers
    stats = np.empty((task_config.num_refits, num_perturbed, num_continuous))
    stat_signs = np.empty_like(stats)
    rec_corr, slope = np.empty((task_config.num_refits, num_continuous)), np.empty(
        (task_config.num_refits, num_continuous)
    )
    ks_mask = np.zeros((num_perturbed, num_continuous))
    latent_matrix = np.empty(
        (num_samples, task_config.model.num_latent, len(dataloaders))
    )

    # Last appended dataloader is the baseline
    baseline_dataset = cast(MOVEDataset, baseline_dataloader.dataset)

    # Train models
    logger = get_logger(__name__)
    logger.info("Training models")

    target_dataset_idx = config.data.continuous_names.index(
        task_config.target_dataset
    )
    perturbed_names = con_names[target_dataset_idx]

    for j in range(task_config.num_refits):  # Train num_refits models

        # Initialize model
        model: VAE = hydra.utils.instantiate(
            task_config.model,
            continuous_shapes=baseline_dataset.con_shapes,
            categorical_shapes=baseline_dataset.cat_shapes,
        )
        if j == 0:
            logger.debug(f"Model: {model}")

        # Train/reload model
        model_path = models_path / f"model_{task_config.model.num_latent}_{j}.pt"
        if model_path.exists():
            logger.debug(f"Re-loading refit {j + 1}/{task_config.num_refits}")
            model.load_state_dict(torch.load(model_path))
            model.to(device)
        else:
            logger.debug(f"Training refit {j + 1}/{task_config.num_refits}")
            model.to(device)
            hydra.utils.call(
                task_config.training_loop,
                model=model,
                train_dataloader=train_dataloader,
            )
            if task_config.save_refits:
                torch.save(model.state_dict(), model_path, pickle_protocol=4)
        model.eval()

        # Calculate baseline reconstruction
        _, baseline_recon = model.reconstruct(baseline_dataloader)
        min_feat = np.zeros((num_perturbed, num_continuous))
        max_feat = np.zeros((num_perturbed, num_continuous))
        min_baseline = np.min(baseline_recon, axis=0)
        max_baseline = np.max(baseline_recon, axis=0)

        ############ QC of feature's reconstruction ##############################
        logger.debug("Calculating quality control of the feature reconstructions")
        # Correlation and slope for each feature's reconstruction
        feature_names = reduce(list.__add__, con_names)

        for k in range(num_continuous):
            x = baseline_dataloader.dataset.con_all.numpy()[:, k]  # baseline_recon[:,i]
            y = baseline_recon[:, k]
            x_pol, y_pol, (a2, a1, a) = get_2nd_order_polynomial(x, y)
            slope[j, k] = a1
            rec_corr[j, k] = pearsonr(x, y).statistic

            if (
                feature_names[k] in task_config.perturbed_feature_names
                or feature_names[k] in task_config.target_feature_names
            ):

                # Plot correlations
                fig = plot_correlations(x, y, x_pol, y_pol, a2, a1, a, k)
                fig.savefig(
                    figure_path
                    / f"Input_vs_reconstruction_correlation_feature_{k}_refit_{j}.png",
                    dpi=50,
                )

        ################## Calculate perturbed reconstruction and shifts #############################
        logger.debug("Computing KS scores")

        # Save original latent space for first refit:
        if j == 0:
            latent = model.project(baseline_dataloader)
            latent_matrix[:, :, -1] = latent

        for i, pert_feat in enumerate(perturbed_names):
            _, perturb_recon = model.reconstruct(dataloaders[i])
            min_perturb = np.min(perturb_recon, axis=0)
            max_perturb = np.max(perturb_recon, axis=0)
            min_feat[i, :] = np.min([min_baseline, min_perturb], axis=0)
            max_feat[i, :] = np.max([max_baseline, max_perturb], axis=0)

            # Save latent representation for perturbed samples
            if j == 0:
                latent_pert = model.project(dataloaders[i])
                latent_matrix[:, :, i] = latent_pert

            for k, targ_feat in enumerate(feature_names):
                # Calculate ks factors: measure distance between baseline and perturbed
                # reconstruction distributions per feature (k)
                res = ks_2samp(perturb_recon[:, k], baseline_recon[:, k])
                stats[j, i, k] = res.statistic
                stat_signs[j, i, k] = res.statistic_sign

                if (
                    pert_feat in task_config.perturbed_feature_names
                    and targ_feat in task_config.target_feature_names
                ):

                    # Plotting preliminary results:
                    n_bins = 50
                    hist_base, edges = np.histogram(
                        baseline_recon[:, k],
                        bins=np.linspace(min_feat[i, k], max_feat[i, k], n_bins),
                        density=True,
                    )
                    hist_pert, edges = np.histogram(
                        perturb_recon[:, k],
                        bins=np.linspace(min_feat[i, k], max_feat[i, k], n_bins),
                        density=True,
                    )

                    # Cumulative distribution:
                    fig = plot_cumulative_distributions(
                        edges,
                        hist_base,
                        hist_pert,
                        f"Cumulative_perturbed_{i}_measuring_{k}_stats_{stats[j,i,k]}",
                    )
                    fig.savefig(
                        figure_path
                        / f"Cumulative_refit_{j}_perturbed_{i}_measuring_{k}_stats_{stats[j,i,k]}.png"
                    )

                    # Feature changes:
                    fig = plot_reconstruction_movement(baseline_recon, perturb_recon, k)
                    fig.savefig(
                        figure_path / f"Changes_pert_{i}_on_feat_{k}_refit_{j}.png"
                    )

    # Save latent space matrix:
    np.save(output_path / "latent_location.npy", latent_matrix)
    np.save(output_path / "perturbed_features_list.npy", np.array(perturbed_names))

    # Creating a mask for self associations
    logger.debug("Creating self-association mask")
    for i in range(num_perturbed):
        if task_config.target_value in CONTINUOUS_TARGET_VALUE:
            ks_mask[i, :] = (
                baseline_dataloader.dataset.con_all[0, :]
                - dataloaders[i].dataset.con_all[0, :]
            )
    ks_mask[ks_mask != 0] = 1
    ks_mask = np.array(ks_mask, dtype=bool)

    # Take the median of KS values (with sign) over refits.
    final_stats = np.nanmedian(stats * stat_signs, axis=0)
    final_stats[ks_mask] = 0.0 # Zero all masked values, placing them at end of the ranking

    # KS-threshold:
    ks_thr = np.sqrt(-np.log(task_config.sig_threshold / 2) * 1 / (num_samples))
    logger.info(f"Suggested absolute KS threshold is: {ks_thr}")

    # Sort associations by absolute KS value
    sort_ids = np.argsort(abs(final_stats), axis=None)[::-1]  # 1D: N x C
    ks_distance = np.take(final_stats, sort_ids)  # 1D: N x C

    # Writing Quality control csv file. Mean slope and correlation over refits as qc metrics.
    logger.info("Writing QC file")
    qc_df = pd.DataFrame({"Feature names": feature_names})
    qc_df["slope"] = np.nanmean(slope, axis=0)
    qc_df["reconstruction_correlation"] = np.nanmean(rec_corr, axis=0)
    qc_df.to_csv(output_path / f"QC_summary_KS.tsv", sep="\t", index=False)

    # Return first idx associations: redefined for reasonable threshold

    return sort_ids[abs(ks_distance) >= ks_thr], ks_distance[abs(ks_distance) >= ks_thr]


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

    num_continuous = sum(con_shapes)  # C

    if sig_ids.size > 0:
        sig_ids = np.vstack((sig_ids // num_continuous, sig_ids % num_continuous)).T
        logger.info("Writing results")
        results = pd.DataFrame(sig_ids, columns=["feature_a_id", "feature_b_id"])

        # Check if the task is for continuous or categorical data
        if task_config.target_value in CONTINUOUS_TARGET_VALUE:
            target_dataset_idx = config.data.continuous_names.index(
                task_config.target_dataset
            )
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
            output_path / f"results_sig_assoc_{task_type}.tsv", sep="\t", index=False
        )


def identify_associations(config: MOVEConfig) -> None:
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

    output_path = Path(config.data.results_path) / "identify_associations"
    output_path.mkdir(exist_ok=True, parents=True)

    # Load datasets:
    cat_list, cat_names, con_list, con_names = io.load_preprocessed_data(
        interim_path,
        config.data.categorical_names,
        config.data.continuous_names,
    )

    train_dataloader = make_dataloader(
        cat_list,
        con_list,
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

    # POSSIBLE CHANGE_ GET NAN MASK HERE, IF IT IS THE SAME FOR ALL PERTURBED DATALOADERS. SEE MULTIPROCESS SCRIPT
    baseline_dataset = cast(MOVEDataset, baseline_dataloader.dataset)
    #cloned_dataset = baseline_dataset.con_all.clone()

    orig_con = baseline_dataset.con_all # Original continuous data of the baseline dataset (all data)
    nan_mask = (orig_con == 0).numpy()
    feature_mask = nan_mask

    # Indentify associations between continuous features:
    logger.info(f"Perturbing dataset: '{task_config.target_dataset}'")
    if task_config.target_value in CONTINUOUS_TARGET_VALUE:
        logger.info(f"Beginning task: identify associations continuous ({task_type})")
        logger.info(f"Perturbation type: {task_config.target_value}")
        output_subpath = Path(output_path) / "perturbation_visualization"
        output_subpath.mkdir(exist_ok=True, parents=True)
        #(dataloaders, nan_mask, feature_mask,) = prepare_for_continuous_perturbation_one(
            #config, baseline_dataloader, index_pert_feat=1 # I indicate 1 as index because I don't mind, I am only interested in
            # createing the masks, the dataloaders I will create inside def_bayes and def_ttest, one at a time
        #)

    # Identify associations between categorical and continuous features:
    else:
        logger.info("Beginning task: identify associations categorical")
        (dataloaders, nan_mask, feature_mask,) = prepare_for_categorical_perturbation(
            config, interim_path, baseline_dataloader, cat_list
        )

    num_perturbed = 3 # len(dataloaders) - 1  # P CHANGE THIS ACCORDINGLY VERY IMPORTANT. mAYBE TRY TO A
    # AUTOMATE IT LATER SO TAHT IT GETS THE LENGTH OF FEATURES IN THE TARGET DATASET. For now I only wnat to 
    # see it it works, así que voy a hacer un poco una chapuza y poner directamente el número. 


    # Perfect, no tengo que cambiar esto
    logger.debug(f"# perturbed features: {num_perturbed}")

    ################# APPROACH EVALUATION ##########################

    if task_type == "bayes":
        task_config = cast(IdentifyAssociationsBayesConfig, task_config)
        sig_ids, *extra_cols = _bayes_approach(
            config,
            task_config,
            train_dataloader,
            baseline_dataloader,
            #dataloaders, I will create it inside
            num_perturbed,
            num_samples,
            num_continuous,
            nan_mask,
            feature_mask,
            models_path,
        )

        extra_colnames = ["proba", "fdr", "bayes_k"]

    elif task_type == "ttest":
        task_config = cast(IdentifyAssociationsTTestConfig, task_config)
        sig_ids, *extra_cols = _ttest_approach(
            config,
            task_config,
            train_dataloader,
            baseline_dataloader,
            #dataloaders, I will create it inside
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

    if exists(output_path / f"results_sig_assoc_{task_type}.tsv"):
        association_df = pd.read_csv(
            output_path / f"results_sig_assoc_{task_type}.tsv", sep="\t"
        )
        plot_feature_association_graph(association_df, output_path)
        plot_feature_association_graph(
            association_df, output_path, layout="spring"
        )
