import hydra 
from omegaconf import OmegaConf

from move.conf.schema import MOVEConfig

@hydra.main(config_path="conf", config_name="main")
def main(config: MOVEConfig) -> str:
    print('''
          To use MOVE pipeline, please run the following commands: 
          python -m move.01_encode_data
          python -m move.02_optimize_reconstruction
          python -m move.03_optimize_stability
          python -m move.04_analyze_latent
          python -m move.05_identify_drug_assosiation
          
          To override the hyperparameter values, please write user defined values in the working directory in the following files:
          data.yaml, model.yaml, tuning_reconstruction.yaml, tuning_stability, training_latent.yaml, training_association.yaml. 
          '''
         )
    
if __name__ == "__main__":
    main()

