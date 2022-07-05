# Load functions
import hydra 
from move.conf.schema import MOVEConfig

from move._utils.data_utils import get_data
from move._training.train import train_model_association
from move._utils.visualization_utils import visualize_indi_var, visualize_drug_similarity_across_all
from move._analysis.analysis import cal_reconstruction_change, overlapping_hits, identify_high_supported_hits, report_values, get_change_in_reconstruction, write_omics_results, make_files, get_inter_drug_variation, get_drug_similar_each_omics

import numpy as np #for some reason when I import numpy before move functions - stucks in running


@hydra.main(config_path="../conf", config_name="main")
def main(config: MOVEConfig): 
    
    #Get needed variables
    path = config.data.processed_data_path
    data_of_interest = config.data.data_of_interest
    version = config.data.version
    
    cuda = config.model.cuda
    nepochs = config.model.num_epochs

    lrate = config.model.lr
    kld_steps = config.model.kld_steps
    batch_steps = config.model.batch_steps
    categorical_names = config.model.categorical_names
    continuous_names = config.model.continuous_names

    nHiddens = config.training_final.num_hidden
    nLatents = config.training_final.num_latent
    nLayers = config.training_final.num_layers
    nDropout = config.training_final.dropout
    nBeta = config.training_final.beta
    batch_sizes = config.training_final.batch_sizes
    repeats = config.training_final.repeats
    
    types = [[1, 0]]

    
    cat_list, con_list, cat_names, con_names, headers_all, drug, drug_h = get_data(path, categorical_names, continuous_names, data_of_interest)
    
    #train model
    train_model_association(path, cuda, nepochs, nLatents, batch_sizes, nHiddens, nLayers, nBeta, nDropout, con_list, cat_list, version, repeats, kld_steps, batch_steps, lrate, drug, categorical_names, data_of_interest)
    
    # Load files 
    results = np.load(path + "results/results_" + version + ".npy", allow_pickle=True).item()
    recon_results = np.load(path + "results/results_recon_" + version + ".npy", allow_pickle=True).item()
    groups = np.load(path + "results/results_groups_" + version + ".npy", allow_pickle=True).item()
    
    mean_bas = np.load(path + "results/results_recon_mean_baseline_" + version + ".npy", allow_pickle=True).item()
    recon_results_1 = np.load(path + "results/results_recon_no_corr_" + version + ".npy", allow_pickle=True).item()
    cor_results = np.load(path + "wp2.2/sig_overlap/cor_results_" + version + ".npy", allow_pickle=True).item()
    
    # Start analysis
    recon_average = cal_reconstruction_change(recon_results, repeats)
    
    sig_hits, median_p_val = overlapping_hits(nLatents, cor_results, repeats, con_names, drug)
    
    all_hits, collected_overlap = identify_high_supported_hits(sig_hits, drug_h, version, path)
    print(collected_overlap)
    
    report_values(path, sig_hits, median_p_val, drug_h, all_hits, con_names)
    con_list_concat = np.concatenate(con_list, axis=-1)
    
    recon_average_corr_new_all, recon_average_corr_all_indi_new = get_change_in_reconstruction(recon_average, groups, drug, drug_h, con_names, collected_overlap, sig_hits, con_list_concat, version, path, types)
    
    recon_average_corr_new_all = np.load(path + "results/results_confidence_recon_all_" + version + ".npy", allow_pickle=True)
    
    recon_average_corr_all_indi_new = np.load(path + "results/results_confidence_recon_all_indi_" + version + ".npy", allow_pickle=True).item()

    up_down_list = ['baseline_target_metabolomics', 'baseline_untarget_metabolomics']         
    
    write_omics_results(path, up_down_list, collected_overlap, recon_average_corr_new_all, headers_all, continuous_names, data_of_interest)
    
    make_files(collected_overlap, groups, con_list_concat, path, recon_average_corr_all_indi_new, con_names, continuous_names, drug_h, drug, all_hits, types, version)
    
    
    df_indi_var = get_inter_drug_variation(con_names, drug_h, recon_average_corr_all_indi_new, 
                                           groups, collected_overlap, drug, con_list_concat, path, types)

    visualize_indi_var(df_indi_var, version, path)
    visualize_drug_similarity_across_all(recon_average_corr_new_all, drug_h, version, path)
    
    get_drug_similar_each_omics(con_names, continuous_names, all_hits, recon_average_corr_new_all, drug_h, version, path)

    
if __name__ == "__main__":
    main()