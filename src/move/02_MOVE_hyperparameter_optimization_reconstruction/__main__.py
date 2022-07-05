# Import functions
import hydra 
from move.conf.schema import MOVEConfig

from move._utils.data_utils import get_data
from move._training.train import optimize_reconstruction
from move._utils.visualization_utils import visualize_likelihood, visualize_recon_acc

@hydra.main(config_path="../conf", config_name="main")
def main(config: MOVEConfig): 
    
    #Get needed variables
    path = config.data.processed_data_path
    data_of_interest = config.data.data_of_interest
    
    cuda = config.model.cuda
    nepochs = config.model.num_epochs
    kld_steps = config.model.kld_steps
    batch_steps = config.model.batch_steps
    patience = config.model.patience
    lrate = config.model.lr
    categorical_names = config.model.categorical_names
    continuous_names = config.model.continuous_names
    
    nHiddens = config.tuning_reconstruction.num_hidden
    nLatents = config.tuning_reconstruction.num_latent
    nLayers = config.tuning_reconstruction.num_layers
    nDropout = config.tuning_reconstruction.dropout
    nBeta = config.tuning_reconstruction.beta
    batch_sizes = config.tuning_reconstruction.batch_sizes
    repeat = config.tuning_reconstruction.repeats
    
        
    #Get the data
    cat_list, con_list, cat_names, con_names, headers_all, drug, drug_h = get_data(path, categorical_names, continuous_names, data_of_interest)
    
    #Perform hyperparameter tuning
    likelihood_tests, recon_acc_tests, recon_acc = optimize_reconstruction(nHiddens, nLatents, 
                                                                            nLayers, nDropout, 
                                                                            nBeta, batch_sizes, 
                                                                            nepochs, repeat, 
                                                                            lrate, kld_steps, 
                                                                            batch_steps, patience, 
                                                                            cuda, path, 
                                                                            cat_list, con_list)

    #Visualize the data
    visualize_likelihood(path, nLayers, nHiddens, nDropout, nBeta, nLatents, likelihood_tests)
    visualize_recon_acc(path, nLayers, nHiddens, nDropout, nBeta, nLatents, recon_acc_tests, 'test')
    visualize_recon_acc(path, nLayers, nHiddens, nDropout, nBeta, nLatents, recon_acc, 'train')    


if __name__ == "__main__":
    main()
