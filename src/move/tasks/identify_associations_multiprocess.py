__all__ = ["identify_associations_multiprocess"]

import faulthandler

from functools import reduce
from pathlib import Path
from typing import Literal, Sized, cast
from move.data.preprocessing import feature_stats
# from move.visualization.dataset_distributions import plot_value_distributions
ContinuousPerturbationType = Literal["minimum", "maximum", "plus_std", "minus_std"]

#from multiprocessing import Pool, Lock

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from scipy.stats import ks_2samp, pearsonr  # type: ignore
from torch.utils.data import DataLoader

# from move.analysis.metrics import get_2nd_order_polynomial

import multiprocessing
from multiprocessing import Pool, Process, Manager

from move.conf.schema import (
    IdentifyAssociationsBayesConfig,
    IdentifyAssociationsConfig,
    IdentifyAssociationsKSConfig,
    IdentifyAssociationsTTestConfig,
    MOVEConfig,
)
from move.core.logging import get_logger
from move.core.typing import BoolArray
from move.data import io

from move.data.dataloaders import MOVEDataset, make_dataloader 
# make_dataloader creates a dataloader with both continuous and categorical datasets

from move.data.perturbations import (
    ContinuousPerturbationType,
    perturb_categorical_data,
    # perturb_continuous_data_extended,
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


from move.data.preprocessing import one_hot_encode_single
from move.models.vae import VAE
# from move.visualization.dataset_distributions import (
#     plot_correlations,
#     plot_cumulative_distributions,
#     plot_feature_association_graph,
#     plot_reconstruction_movement,
# )

# NOT IMPORTANT NOW, use it once multiprocessing works and I change the script to get the real results
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



# perturb_continuous_data_extended_one adapts perturb_continuous_data_extended, so that instead of creating a list of perturbed
# dataloaders, it perturbs only one feature at a time.
########################################################################
### THIS FUNCTION WORKS IN MULTIPROCESSING ###
########################################################################
def perturb_continuous_data_extended_one( # We will keep the input almost the same,
    # but we to introduce a variable that allows to index the specific dataloader we want to create (index_pert_feat)
    # And no need for the output directory
    baseline_dataloader: DataLoader,
    cloned_dataset, # CHANGE for multiprocessing, import the already cloned dataset, as cloning inside the process does not work.
    con_dataset_names: list[str],
    target_dataset_name: str,
    perturbation_type: ContinuousPerturbationType,
    index_pert_feat: int,
    continuous_shapes,
    categorical_shapes,
    baseline_dataset_cat_all,
) -> DataLoader: # Ee change the output from list[DataLoader] to just one DataLoader
    logger = get_logger(__name__)
    """Add perturbations to continuous data. For each feature in the target
    dataset, change the feature's value in all samples (in rows):
    1,2) substituting this feature in all samples by the feature's minimum/maximum value.
    3,4) Adding/Substracting one standard deviation to the sample's feature value.

    Returns:
        - Dataloader with the ith feature (index_pert_feat) perturbed.
 
    """
    logger.debug(f"Inside perturb_continuous_data_extended_one for feature {index_pert_feat}")
    
    # Took the following part out and passed this as arguments, as trying to do it inside multiprocessing
    # does not work
    #baseline_dataset = cast(MOVEDataset, baseline_dataloader.dataset)
    #assert baseline_dataset.con_shapes is not None
    #assert baseline_dataset.con_all is not None

    target_idx = con_dataset_names.index(target_dataset_name)  # dataset index
    #logger.debug(f"target index: {target_idx}")
    splits = np.cumsum([0] + continuous_shapes)
    #logger.debug(f"splits: {splits}")
    slice_ = slice(*splits[target_idx : target_idx + 2])
    #logger.debug(f"slice: {slice}")

    # No need for this variable in the new script. Also, it would not work, because I need to operat on a pytorch object
    #num_features = baseline_dataset.con_shapes[target_idx] 


    # Now, instead of the for loop that iterates over all the features we want to perturb, we do it only for one feature, the one 
    # indicated in index_pert_feat
   
    logger.debug(f"Inside perturb_continuous_data_extended_one, using already cloned dataset for {index_pert_feat}")
    # Cloning inside multiprocessing does not work, so I clone outside the baseline dataset outside and pass it as an
    # argument
    #perturbed_con = baseline_dataset.con_all.clone()
    perturbed_con = cloned_dataset
    logger.debug(f"Already assigned the cloned dataset. Going into slice, for feature {index_pert_feat}")

    target_dataset = perturbed_con[:, slice_]
    logger.debug(f"target dataset created for feature {index_pert_feat}")

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

    logger.debug(f"Creating perturbed dataset and dataloader in pertrub_cont_data_extended_one for feature {index_pert_feat}")
    continuous_shapes = continuous_shapes
    categorical_shapes = categorical_shapes
    perturbed_dataset = MOVEDataset(
        baseline_dataset_cat_all,
        perturbed_con,
        continuous_shapes,
        categorical_shapes,
    )

    perturbed_dataloader = DataLoader(
        perturbed_dataset,
        shuffle=False,
        batch_size=baseline_dataloader.batch_size,
    )
    #dataloaders.append(perturbed_dataloader)

    logger.debug(f"Finished perturb_continuous_data_extended_one for feature {index_pert_feat}")

    return perturbed_dataloader




# Returns also NaN mask. May be redundant because now I create the NaN mask outside, 
# possible to join with the previous function into one, once everything works
def prepare_for_continuous_perturbation_one(
    config: MOVEConfig,
    # output_subpath: Path, Remove this
    baseline_dataloader: DataLoader,
    index_pert_feat,
    cloned_dataset,
    nan_mask,
    continuous_shapes,
    categorical_shapes,
    baseline_dataset_cat_all,) -> tuple[DataLoader, BoolArray, BoolArray,]:
    """
    This function creates the required dataloaders and masks
    for further continuous association analysis. 
    This function will probably be removed afterwards, and just use perturbed_continuous_data_extended_one
    Returns:
        dataloaders:
            One dataloader, correponding to the feature we are perturbing at the moment.
        nan_mask:
            mask for NaNs for that dataloader
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

    # Clone the baseline dataset to ensure it's not modified
    logger.debug(f"Passing the data in prepare_for_continuous_pertrubation_one to perturb_cont_data_extened_one for feature {index_pert_feat}")
    # First, I tried to do deepcopies when cloning, but it still did not work. So I just cloned the dataset ouside and
    # passed it as an argument
    #baseline_dataset = cast(MOVEDataset, baseline_dataloader.dataset)
    #cloned_dataset = copy.deepcopy(baseline_dataset)
    #logger.debug(f"Deep copy succesful for feature {index_pert_feat}")

    dataloaders = perturb_continuous_data_extended_one(
        baseline_dataloader,
        cloned_dataset,
        config.data.continuous_names,
        task_config.target_dataset,
        cast(ContinuousPerturbationType, task_config.target_value),
        index_pert_feat,
        continuous_shapes = continuous_shapes,
        categorical_shapes = categorical_shapes,
        baseline_dataset_cat_all = baseline_dataset_cat_all
          # I will have to use this in a loop that iterates over all features in the target dataset
    ) # We will get only one dataloader for each iteration of the for loop
    #dataloaders.append(baseline_dataloader) # Add the baseline dataloader to the perturbed dataloader


    # All this code is not necessary now, as I pass NaN mask for outside
    # Leave it to make sure that the new script is correct

    #logger.debug(f"Creating NaN mask in prepare_con for feature {index_pert_feat}")
    # Get the baseline dataset
    #baseline_dataset = cast(MOVEDataset, baseline_dataloader.dataset)
    #assert baseline_dataset.con_all is not None
    #orig_con = baseline_dataset.con_all # Original continuous data of the baseline dataset (all data)
    #nan_mask = (orig_con == 0).numpy()  # Creates a mask to identify NaN values in the original data. 
    # In this case, NaN values are encoded as 0s. The expression (orig_con == 0) creates a boolean mask where True indicates 
    # the presence of NaN values, and False indicates non-NaN values. .numpy() converts this boolean mask to a numpy array.
    #logger.debug(f"# NaN values: {np.sum(nan_mask)}/{orig_con.numel()}")
    # np.sum(nan_mask) calculates the total number of NaN values using the nan_mask, and orig_con.numel() calculates the total number of elements in the original data.
    feature_mask = nan_mask

    logger.debug(f"NaN mask for feature {index_pert_feat} copied.")

    return (dataloaders, nan_mask, feature_mask)


###################################################################################
#                       THINGS FOR MULTIPROCESSING 
###################################################################################

# Since training models does not work inside multiprocessing, I'll train the refits outside, and just load them
# inside the multiprocessing. I will use that each model to get the reconstruction of the perturbed dataset and 
# the baseline dataset, and compare them.
# The PROBLEM is that reconstruction does not work inside multiprocessing. The baseline I can fix by reconstructing it
# outside and loading it, but the reconstruction of the perturbed dataloader I need to do it inside each worker function
def load_model(models_path, task_config, continuous_shapes, categorical_shapes, j):
    logger = get_logger(__name__)
    """
    Load model from the given path and get corresponding reconstruction baseline
    """
    reconstruction_path = models_path / f"baseline_recon_{task_config.model.num_latent}_{j}.pt"
     # Load the reconstruction from the saved file
    if reconstruction_path.exists():
        baseline_recon = torch.load(reconstruction_path)

    model: VAE = hydra.utils.instantiate(
            task_config.model,
            continuous_shapes=continuous_shapes,
            categorical_shapes=categorical_shapes,
        )
        
    model_path = models_path / f"model_{task_config.model.num_latent}_{j}.pt"
    device = torch.device("cuda" if task_config.model.cuda == True else "cpu")

    #model = VAE()
    logger.debug(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path))
    logger.debug(f"Loaded model from {model_path} in the load_model function")
    model.to(device)
    model.eval()
    
    return model, baseline_recon

# Define a global lock
#global_lock = Lock()

def _bayes_approach_worker(args):
    """
    Worker function to calculate mean differences and Bayes factors for one feature.
    """
    #faulthandler.enable()

    # Unpack arguments, this step works. Later, get rid of the arguments that are not used
    (config, task_config, train_dataloader, baseline_dataloader,
     num_perturbed, num_samples, num_continuous, nan_mask, feature_mask, i, models_path, cloned_dataset,
     continuous_shapes, categorical_shapes, baseline_dataset_cat_all, model, baseline_recon) = args
    # Initialize logging
    logger = get_logger(__name__)
    logger.debug(f"Inside the worker function for num_perturbed {i}")    

    #assert task_config.model is not None
    #logger.debug("Moving model into devide in bayes_approach_worker")
    #device = torch.device("cuda" if task_config.model.cuda == True else "cpu")
    #logger.debug("Model moved to devide in bayes_approach_worker")
    #baseline_dataset = cast(MOVEDataset, baseline_dataloader.dataset)

    logger.debug(f"Setting up mean_diff and normalizer for feature {i}")
    # Now we are inside the num_perturbed loop, we will do this for each of the perturbed features
    # Now, mean_diff will not have a first dimension for num_perturbed, because we will not store it for each perturbed feature
    # we will use it in each loop for calculating the bayes factors, and then overwrite its content with a new perturbed feature
    mean_diff = np.zeros((num_samples, num_continuous))
    # Set the normalizer
    normalizer = 1 / task_config.num_refits # Divide by the number of refits. All the refits will have the same importance
 

    logger.debug(f"Creating perturbed dataloader for feature {i}")
    # Now, we get the perturbed dataloader
    perturbed_dataloader, nan_mask, feature_mask = prepare_for_continuous_perturbation_one(
        config=config,
        baseline_dataloader = baseline_dataloader,
        index_pert_feat= i,
        cloned_dataset = cloned_dataset,
        nan_mask = nan_mask,
        continuous_shapes = continuous_shapes,
        categorical_shapes = categorical_shapes,
        baseline_dataset_cat_all = baseline_dataset_cat_all
    ) # Like this, I get only one perturbed dataloader, and the nan and feature masks
    logger.debug(f"created perturbed dataloader for feature {i}")


    ######################################################################################
    # UP TO HERE, IT RUNS
    # Before, I got deadlocks when trying to train the models, that's why I trained the refits
    # outside and now I just load them
    ######################################################################################

    # For each perturbed feature, we will calculate the difference with the baseline reconstruction, taking into account all
    # refits
    
    #logger.debug(f"Entering in j loop for feature {i}")
    #for j in range(task_config.num_refits):
        # First, we need to reload the baseline reconstruction for the corresponding model
     #   logger.debug(f"Loading baseline_recon for refit {j}")
      #  reconstruction_path = models_path / f"baseline_recon_{task_config.model.num_latent}_{j}.pt"
        # Load the reconstruction from the saved file
       # if reconstruction_path.exists():
        #    baseline_recon = torch.load(reconstruction_path)
        #else:
         #   logger.debug(f"Reconstruction file for iteration {j} does not exist.")
        
        # Now, I get the perturbed reconstruction. First I load the model, then I use it to get the reconstruction

        
        #model: VAE = hydra.utils.instantiate(
         #   task_config.model,
          #  continuous_shapes=continuous_shapes,
           # categorical_shapes=categorical_shapes,
        #)
        
        #model_path = models_path / f"model_{task_config.model.num_latent}_{j}.pt"
        ####### THE ERROR IS HERE ###############

        #with global_lock:
         #   logger.debug(f"Re-loading refit in path {model_path}, refit {j + 1}/{task_config.num_refits} for perturbed recon")

          #  try:
           #     model.load_state_dict(torch.load(model_path))
            #except Exception as e:
             #   logger.error(f"Error loading model from {model_path}: {e}")

            #model.load_state_dict(torch.load(model_path))
            ##########################################
            #logger.debug(f"Loaded model inside lock, in {model_path}, passing it to device for feature {i}")
            #model.to(device)
            #logger.debug(f"Model {j} reloaded for perturbed recon")

            #model.eval()
        
    # Supposedly, we have passed that outside as an argument
    logger.debug(f"Reconstructing num_perturbed {i}, with model {model}")

    ########################################################################
    # NEW PROBLEM IS HERE, RECONSTRUCTION DOES NOT OCCUR
    ########################################################################
    _, perturb_recon = model.reconstruct(perturbed_dataloader) # Instead of dataloaders[i], create the perturbed one here and use it only here
    logger.debug(f"Perturbed reconstruction succesful for feature {i}, model {model}")
    # diff is a matrix with the same dimensions as perturb_recon and baseline_recon (rows are samples and columns all the continuous features)
        
    
    logger.debug(f"Calculating diff for num_perturbed {i}, with model {model}")
    diff = perturb_recon - baseline_recon 
    logger.debug(f"Calculating mean_diff  for num_perturbed {i}, with model {model}")
    mean_diff += diff * normalizer
    logger.debug(f"Returning mean_diff for feature {i}, worker function finished")

    return mean_diff # for the moment, return only this to see if multiprocessing works. Later we will calculate the real results


    #return sort_ids[:idx], prob[:idx], fdr[:idx], bayes_k[:idx] # RESULTS ONLY FOR ONE PERTURBED FEATURE




def _bayes_approach_parallel(
    config, task_config, train_dataloader, baseline_dataloader, num_perturbed,
    num_samples, num_continuous, nan_mask, feature_mask, models_path,
):
    logger = get_logger(__name__)
    logger.debug("Inside the bayes_parallel function")
    
    # First, I train or reload the models (number of refits), and save the baseline reconstruction
    baseline_dataset = cast(MOVEDataset, baseline_dataloader.dataset)

    assert task_config.model is not None
    device = torch.device("cuda" if task_config.model.cuda == True else "cpu")
    logger.debug("Model moved to device in bayes_approach_parallel")

    # Train models
    logger = get_logger(__name__)
    logger.info("Training or reloading models")

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
                torch.save(model.state_dict(), model_path)
        # Independently of whether we loaded the models or trained them, we go into evaluation mode        
        model.eval()

        # Calculate baseline reconstruction
        # For each model j, we get a different reconstruction for the baseline. We haven't perturbed anything yet, we are just
        # getting the reconstruction for the baseline, to make sure that we get the same reconstruction for each refit, we cannot
        # do it inside each process because the results might be different
        _, baseline_recon = model.reconstruct(baseline_dataloader)

        # Save the reconstruction separately. Up to here, it works.
        logger.debug(f"Saving baseline reconstruction {j}")
        reconstruction_path = models_path / f"baseline_recon_{task_config.model.num_latent}_{j}.pt"
        torch.save(baseline_recon, reconstruction_path)
        logger.debug(f"Saved baseline reconstruction {j}")


    # Clone the dataset outside the multiprocessing, because cloning it inside does not work
    cloned_dataset = baseline_dataset.con_all.clone()

    # Get NaN mask before, to see if it works, as I do not need dataloaders for anything to create it.

    logger.debug("Creating NaN mask in prepare_con for feature")
    orig_con = baseline_dataset.con_all # Original continuous data of the baseline dataset (all data)
    nan_mask = (orig_con == 0).numpy()  # Creates a mask to identify NaN values in the original data. 
    # In this case, NaN values are encoded as 0s. The expression (orig_con == 0)
    logger.debug(f"# NaN values: {np.sum(nan_mask)}/{orig_con.numel()}")
    # np.sum(nan_mask) calculates the total number of NaN values using the nan_mask, and orig_con.numel() calculates the total number of elements in the original data.
    feature_mask = nan_mask
    ####################################################
    # We are creating it, so prepare_for_continuous_perturbation_one is not necessary, right now it just
    # gets some argumetns and return them, so remove it in the future
    #####################################################

    logger.debug("NaN mask for feature created. Going into MULTIPROCESSING extended with no issues?")

    # I need to pass also these argumetns to each worker:
    continuous_shapes=baseline_dataset.con_shapes
    categorical_shapes=baseline_dataset.cat_shapes
    baseline_dataset_cat_all = baseline_dataset.cat_all

        
    """
    Perform parallelized bayes approach.
    """
    # Create a list to store loaded models
    models_list = []
    baseline_recon_list =[]

    # Iterate over models and load them. Since creating them inside each process does not work, I will store them in a list
    # and iterate over them
    for j in range(task_config.num_refits):
        model_path = models_path / f"model_{task_config.model.num_latent}_{j}.pt"
        logger.debug(f"Loading model {models_path} into models_list, using load function")
        model, baseline_recon = load_model(models_path, task_config, continuous_shapes, categorical_shapes, j)
        logger.debug(f"load_model succesful for {model_path}")
        models_list.append(model)
        baseline_recon_list.append(baseline_recon)

    logger.debug("Starting parallelization")
    # Define arguments for each worker, and iterate over models and perturbed features
    args = [(config, task_config, train_dataloader, baseline_dataloader,
               num_perturbed, num_samples, num_continuous, nan_mask, feature_mask, i, models_path, cloned_dataset,
               continuous_shapes, categorical_shapes, baseline_dataset_cat_all, model, baseline_recon)
               for model, baseline_recon in zip(models_list, baseline_recon_list)
               for i in range(num_perturbed)]
    
    logger.debug(f"Arguments for workers are {args}")
    
    
    # Create a Pool with multiprocessing.cpu_count() - 1 processes
    with Pool(processes=multiprocessing.cpu_count() - 1) as pool:
        logger.debug("Inside the pool loop?")
        # Map worker function to arguments
        results = pool.map(_bayes_approach_worker, args)
        
    
    logger.info("Pool multiprocess completed? in the bayes parallel function")
    logger.info(f"The results obtained (mean_diff) are {results}")

    return results



#########################################################
# LEAVE THIS FOR LATER, ONCE THE MULTIPROCESSING WORKS
#########################################################
    # # Unpack results from the workers
    # sig_ids, prob, fdr, bayes_k = zip(*results)

    # # Convert to numpy arrays
    # sig_ids = np.array(sig_ids)
    # prob = np.array(prob)
    # fdr = np.array(fdr)
    # bayes_k = np.array(bayes_k)

    # # I will get the results as tuples of arrays, where each array contains the indices, probs, fdr, or bayes factors 
    # # of significantly associated features for a specific perturbed feature. Each array corresponds to one perturbed feature, 
    # # and the order of arrays will follow the order of perturbed features.
    # return sig_ids, prob, fdr, bayes_k




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
    #models_path = Path("/home/qgh533/MOVE_cont/MOVE/isoforms_first_try/interim_data/models")
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

   
    baseline_dataset = cast(MOVEDataset, baseline_dataloader.dataset)

    orig_con = baseline_dataset.con_all # Original continuous data of the baseline dataset (all data)
    nan_mask = (orig_con == 0).numpy()  # Creates a mask to identify NaN values in the original data. 
    # In this case, NaN values are encoded as 0s. The expression (orig_con == 0) creates a boolean mask where True indicates 
    # the presence of NaN values, and False indicates non-NaN values. .numpy() converts this boolean mask to a numpy array.
    logger.debug(f"# NaN values: {np.sum(nan_mask)}/{orig_con.numel()}")
    # np.sum(nan_mask) calculates the total number of NaN values using the nan_mask, and orig_con.numel() calculates the total number of elements in the original data.
    feature_mask = nan_mask

    # Indentify associations between continuous features:
    logger.info(f"Perturbing dataset: '{task_config.target_dataset}'")
    if task_config.target_value in CONTINUOUS_TARGET_VALUE:
        logger.info(f"Beginning task: identify associations continuous ({task_type})")
        logger.info(f"Perturbation type: {task_config.target_value}")
        output_subpath = Path(output_path) / "perturbation_visualization"
        output_subpath.mkdir(exist_ok=True, parents=True)
        #(dataloaders, nan_mask, feature_mask,) = prepare_for_continuous_perturbation_one(
         #   config, baseline_dataloader, index_pert_feat=1, cloned_dataset=cloned_dataset, nan_mask=nan_mask,
          #  continuous_shapes= # I indicate 1 as index because I don't mind, I am only interested in
           # # createing the masks, the dataloaders I will create inside def_bayes and def_ttest, one at a time
        #) NO NEED TO DO THIS, I ALREADY HAVE THE MASK

    # Identify associations between categorical and continuous features:
    else:
        logger.info("Beginning task: identify associations categorical")
        (dataloaders, nan_mask, feature_mask,) = prepare_for_categorical_perturbation(
            config, interim_path, baseline_dataloader, cat_list,
        )

    num_perturbed = 50 # For the moment, 50 to see if it works. Later, change it to the number of features to perturb
    #, (which will be much higher) 


    logger.debug(f"# perturbed features: {num_perturbed}")

    ################# APPROACH EVALUATION ##########################

    # Call _bayes_approach_parallel instead of _bayes_approach
    if task_type == "bayes":
        sig_ids, *extra_cols = _bayes_approach_parallel(
            config,
            task_config,
            train_dataloader,
            baseline_dataloader,
            num_perturbed,
            num_samples,
            num_continuous,
            nan_mask,
            feature_mask,
            models_path,
        )

    # # Combine the results from all processes to get the final associations for all features
    # # Here, you can process the `sig_ids`, `prob`, `fdr`, and `bayes_k` arrays as needed.

    # elif task_type == "ttest":
    #     task_config = cast(IdentifyAssociationsTTestConfig, task_config)
    #     sig_ids, *extra_cols = _ttest_approach(
    #         config,
    #         task_config,
    #         train_dataloader,
    #         baseline_dataloader,
    #         #dataloaders, I will create it inside
    #         models_path,
    #         interim_path,
    #         num_perturbed,
    #         num_samples,
    #         num_continuous,
    #         nan_mask,
    #         feature_mask,
    #     )

    #     extra_colnames = ["p_value"]

    # elif task_type == "ks":
    #     task_config = cast(IdentifyAssociationsKSConfig, task_config)
    #     sig_ids, *extra_cols = _ks_approach(
    #         config,
    #         task_config,
    #         train_dataloader,
    #         baseline_dataloader,
    #         dataloaders,
    #         models_path,
    #         num_perturbed,
    #         num_samples,
    #         num_continuous,
    #         con_names,
    #         output_path,
    #     )

    #     extra_colnames = ["ks_distance"]

    # else:
    #     raise ValueError()

    # ###################### RESULTS ################################
    # save_results(
    #     config,
    #     con_shapes,
    #     cat_names,
    #     con_names,
    #     output_path,
    #     sig_ids,
    #     extra_cols,
    #     extra_colnames,
    # )

    # if exists(output_path / f"results_sig_assoc_{task_type}.tsv"):
    #     association_df = pd.read_csv(
    #         output_path / f"results_sig_assoc_{task_type}.tsv", sep="\t"
    #     )
    #     plot_feature_association_graph(association_df, output_path)
    #     plot_feature_association_graph(
    #         association_df, output_path, layout="spring"
    #     )
