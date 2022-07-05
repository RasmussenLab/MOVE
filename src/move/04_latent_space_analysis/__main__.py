# Load functions
import hydra 
from move.conf.schema import MOVEConfig

from move._utils.data_utils import get_data
from move._training.train import train_model
from move._utils.visualization_utils import embedding_plot_discrete, embedding_plot_float, visualize_training, plot_reconstruction_distribs, visualize_embedding, plot_categorical_importance, plot_continuous_importance
from move._analysis.analysis import get_latents, calc_categorical_reconstruction_acc, calc_continuous_reconstruction_acc, get_embedding, get_pearsonr, get_feature_importance_categorical, get_feature_importance_continuous, save_feat_results, get_feat_importance_on_weights 


@hydra.main(config_path="../conf", config_name="main")
def main(config: MOVEConfig): 
    # By cosine similirity. 
    
    #Get needed variables
    path = config.data.processed_data_path
    data_of_interest = config.data.data_of_interest
    
    cuda = config.model.cuda  
    nepochs = config.model.num_epochs  
    categorical_names = config.model.categorical_names
    continuous_names = config.model.continuous_names
    lrate = config.model.lr
    kld_steps = config.model.kld_steps
    batch_steps = config.model.batch_steps

    nHiddens = config.training_latent.num_hidden
    nLatents = config.training_latent.num_latent
    nLayers = config.training_latent.num_layers
    nDropout = config.training_latent.dropout
    nBeta = config.training_latent.beta
    batch_sizes = config.training_latent.batch_sizes

    epochs = range(1, nepochs + 1)    

    
    #Get the data
    cat_list, con_list, cat_names, con_names, headers_all, drug, drug_h = get_data(path, categorical_names, continuous_names, data_of_interest)
    
    best_model, losses, ce, sse, KLD, train_loader, mask, kld_w, cat_shapes, con_shapes, best_epoch = train_model(cat_list, con_list, batch_sizes, nHiddens, nLayers, nLatents, nBeta, nDropout, cuda, kld_steps, batch_steps, nepochs, lrate, test_loader=None, patience=None, early_stopping=False)
    
    visualize_training(path, losses, ce, sse, KLD, epochs)
    
    latent, latent_var, cat_recon, cat_class, con_recon, loss, likelihood = get_latents(best_model, train_loader, 1)
    
    cat_total_recon = calc_categorical_reconstruction_acc(cat_shapes, cat_class, cat_recon)
    
    all_values = calc_continuous_reconstruction_acc(con_shapes, con_recon, train_loader)
    
    plot_reconstruction_distribs(path, cat_total_recon, all_values)
    
    embedding = get_embedding(path, latent)
    
    visualize_embedding(path, 'categorical', "drug_1", embedding, 
                    mask, cat_list, con_list, cat_names, con_names)
#     visualize_embedding(path, 'continuous', "clinical_continuous_1", embedding, 
#                         mask, cat_list, con_list, cat_names, con_names) #Todo: clinical_continuous_1 does not exist??
    visualize_embedding(path, 'continuous', "clinical_continuous_2", embedding, 
                        mask, cat_list, con_list, cat_names, con_names)    
     
    
    
    get_pearsonr('categorical', "drug_1", embedding, cat_list, con_list, cat_names, con_names)
    get_pearsonr('continuous', "clinical_continuous_2", embedding, cat_list, con_list, cat_names, con_names)
    
    
    all_diffs, all_diffs_cat_np, sum_diffs_cat_np, sum_diffs_cat_abs_np,\
        total_diffs_cat_np = get_feature_importance_categorical(best_model, train_loader, latent)

    
    all_diffs_con_np, sum_diffs_con_np, sum_diffs_con_abs_np,\
            total_diffs_con_np = get_feature_importance_continuous(best_model, train_loader, mask, latent)

    save_feat_results(path, all_diffs_cat_np, sum_diffs_cat_np, sum_diffs_cat_abs_np, total_diffs_cat_np, 
                 all_diffs_con_np, sum_diffs_con_np,sum_diffs_con_abs_np, total_diffs_con_np)
    
    
    plot_categorical_importance(path=path,
                            sum_diffs=sum_diffs_cat_np,
                            cat_list=cat_list,
                            feature_names=cat_names,
                            fig_name='importance_SHAP_cat')
    
    
    plot_continuous_importance(path=path,
                           train_loader=train_loader,
                           sum_diffs=sum_diffs_con_np,
                           feature_names=con_names,
                           fig_name='importance_SHAP_con')
    
    get_feat_importance_on_weights(path, best_model, train_loader, cat_names, con_names)


if __name__ == "__main__":
    main()