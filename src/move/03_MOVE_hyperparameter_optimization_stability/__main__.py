# Import functions
import hydra 
from move.conf.schema import MOVEConfig
from move._utils.data_utils import get_data, get_list_value
from move._utils.visualization_utils import draw_boxplot
from move._training.train import optimize_stability
from move._analysis.analysis import get_top10_stability, calculate_latent

@hydra.main(config_path="../conf", config_name="main")
def main(config: MOVEConfig): 
    #Get needed variables
    path = config.data.processed_data_path   
    data_of_interest = config.data.data_of_interest
    
    cuda = config.model.cuda
    nepochs = config.model.num_epochs
    lrate = config.model.lr
    kld_steps = config.model.kld_steps
    batch_steps = config.model.batch_steps
#     patience = config.model.patience
    categorical_names = config.model.categorical_names
    continuous_names = config.model.continuous_names
    
    nHiddens = config.tuning_stability.num_hidden
    nLatents = config.tuning_stability.num_latent
    nLayers = config.tuning_stability.num_layers
    nDropout = config.tuning_stability.dropout
    nBeta = config.tuning_stability.beta
    batch_sizes = config.tuning_stability.batch_sizes
    repeat = config.tuning_stability.repeats    
    
    nLayers, batch_sizes = get_list_value(nLayers, batch_sizes)
    print(nBeta)
    if len(nBeta)==1:
        nBeta = nBeta[0]
    
    #Get the data
    cat_list, con_list, cat_names, con_names, headers_all, drug, drug_h = get_data(path, categorical_names, continuous_names, data_of_interest)

    
    embeddings, latents, con_recons, cat_recons, recon_acc = optimize_stability(nHiddens, nLatents, 
                                                                                nDropout, nBeta, repeat,
                                                                                nepochs, nLayers,
                                                                                batch_sizes, lrate, 
                                                                                kld_steps, batch_steps, 
                                                                                cuda, path, 
                                                                                con_list, cat_list)
    
    stability_top10 = get_top10_stability(nHiddens, nLatents, nDropout, repeat, nLayers, latents)
    
    stability_total, rand_index = calculate_latent(nHiddens, nLatents, nDropout, repeat, nLayers, latents) # Todo add priting or smth
    
    draw_boxplot(path=path,
                 df=stability_top10,
                 title_text='Difference across replicationes in cosine similarity of ten closest neighbours in first iteration',
                 y_label_text="Average change",
                 save_fig_name="stability_top10")
    
    
    draw_boxplot(df=stability_total,
                 path=path,
                 title_text='Difference across replicationes in cosine similarity compared to first iteration',
                 y_label_text="Average change",
                 save_fig_name="stability_all")
    
    draw_boxplot(df=rand_index,
                 path=path,
                 title_text='Rand index across replicationes compared to first iteration',
                 y_label_text="Rand index",
                 save_fig_name="rand_index_all")

if __name__ == "__main__":
    main()


    
    

