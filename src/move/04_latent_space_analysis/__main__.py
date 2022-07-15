# Load functions
import hydra 
from move.conf.schema import MOVEConfig

from move.training.train import train_model
from move.utils.data_utils import get_data, merge_configs
from move.utils.visualization_utils import embedding_plot_discrete, embedding_plot_float, visualize_training, plot_reconstruction_distribs, visualize_embedding, plot_categorical_importance, plot_continuous_importance
from move.utils.analysis import get_latents, calc_categorical_reconstruction_acc, calc_continuous_reconstruction_acc, get_embedding, get_pearsonr, get_feature_importance_categorical, get_feature_importance_continuous, save_feat_results, get_feat_importance_on_weights 

import numpy as np

@hydra.main(config_path="../conf", config_name="main")
def main(base_config: MOVEConfig): 
    
    # Overriding base_config with the user defined configs.
    cfg = merge_configs(base_config=base_config, 
                config_types=['data', 'model', 'training_latent'])
    
    #Getting the variables used in the notebook
    raw_data_path = cfg.data.raw_data_path
    interim_data_path = cfg.data.interim_data_path
    processed_data_path = cfg.data.processed_data_path 
    data_of_interest = cfg.data.data_of_interest
    categorical_names = cfg.data.categorical_names
    continuous_names = cfg.data.continuous_names
    categorical_weights = cfg.data.categorical_weights
    continuous_weights = cfg.data.continuous_weights  
    features_to_visualize = cfg.data.data_features_to_visualize_notebook4
    
    seed = cfg.model.seed
    cuda = cfg.model.cuda   
    lrate = cfg.model.lrate
    kld_steps = cfg.model.kld_steps
    batch_steps = cfg.model.batch_steps

    nHiddens = cfg.training_latent.num_hidden
    nLatents = cfg.training_latent.num_latent
    nLayers = cfg.training_latent.num_layers
    nDropout = cfg.training_latent.dropout
    nBeta = cfg.training_latent.beta
    batch_sizes = cfg.training_latent.batch_sizes
    nepochs = cfg.training_latent.tuned_num_epochs 
    
    epochs = range(1, nepochs + 1)    
    
    #Getting the data
    cat_list, con_list, cat_names, con_names, headers_all, drug, drug_h = get_data(raw_data_path, interim_data_path, categorical_names, continuous_names, data_of_interest)
    
    # Checking if all input features selected for visualization were in headers_all
    for feature in features_to_visualize:
        if feature not in np.concatenate(headers_all):
            raise ValueError(f"{feature} is not in the headers_all list. It could have been it was not among the features of the input dataset or was filtered out during data processing")
    
    # Training the model 
    print('Beginning training the model.\n')
    best_model, losses, ce, sse, KLD, train_loader, mask, kld_w, cat_shapes, con_shapes, best_epoch = train_model(cat_list, con_list, categorical_weights, continuous_weights, batch_sizes, nHiddens, nLayers, nLatents, nBeta, nDropout, cuda, kld_steps, batch_steps, nepochs, lrate, seed, test_loader=None, patience=None, early_stopping=False)
    print('\nFinished training the model.')
    
    # Visualizing the training
    visualize_training(processed_data_path, losses, ce, sse, KLD, epochs)
    
    # Getting the reconstruction results
    latent, latent_var, cat_recon, cat_class, con_recon, loss, likelihood = get_latents(best_model, train_loader, 1)
    cat_total_recon = calc_categorical_reconstruction_acc(cat_shapes, cat_class, cat_recon)
    all_values = calc_continuous_reconstruction_acc(con_shapes, con_recon, train_loader)
    
    # Plotting the reconstruction distributions   
    plot_reconstruction_distribs(processed_data_path, cat_total_recon, all_values)
    
    # Getting the embeddings
    print('\n Getting the embeddings.')    
    embedding = get_embedding(processed_data_path, latent)
    
    # Visualizing the embedding of three example features
    for feature in features_to_visualize:
        visualize_embedding(processed_data_path, feature, embedding, 
                            mask, cat_list, con_list, cat_names, con_names)
        
    # Getting pearson correlations of two example features
    for feature in features_to_visualize:
        spear_corr = get_pearsonr(feature, embedding, cat_list, con_list, cat_names, con_names)
        print(f"Pearson correlation for the 1st embedding dim of {feature}: {round(spear_corr[0][0], 3)}, p-value={round(spear_corr[0][1], 3)}")
        print(f"Pearson correlation for the 2nd embedding dim of {feature}: {round(spear_corr[1][0], 3)}, p-value={round(spear_corr[1][1], 3)}")
        
    # Getting features importance measures
    all_diffs, all_diffs_cat_np, sum_diffs_cat_np, sum_diffs_cat_abs_np,\
        total_diffs_cat_np = get_feature_importance_categorical(best_model, train_loader, latent)
    all_diffs_con_np, sum_diffs_con_np, sum_diffs_con_abs_np,\
            total_diffs_con_np = get_feature_importance_continuous(best_model, train_loader, mask, latent)

    # Saving features importance measure results 
    save_feat_results(processed_data_path, all_diffs_cat_np, sum_diffs_cat_np, sum_diffs_cat_abs_np, total_diffs_cat_np, 
                 all_diffs_con_np, sum_diffs_con_np,sum_diffs_con_abs_np, total_diffs_con_np)
    
    # Plotting categorical importance measures
    plot_categorical_importance(path=processed_data_path,
                            sum_diffs=sum_diffs_cat_np,
                            cat_list=cat_list,
                            feature_names=cat_names,
                            fig_name='importance_SHAP_cat')
    
    # Plotting continuous importance measures
    plot_continuous_importance(path=processed_data_path,
                           train_loader=train_loader,
                           sum_diffs=sum_diffs_con_np,
                           feature_names=con_names,
                           fig_name='importance_SHAP_con')
    
    # Getting feature importance on weights
    get_feat_importance_on_weights(processed_data_path, best_model, train_loader, cat_names, con_names)


if __name__ == "__main__":
    main()