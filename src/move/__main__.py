__all__ = []

import hydra
from omegaconf import OmegaConf

import move.tasks
from move.conf.schema import (
    AnalyzeLatentConfig,
    EncodeDataConfig,
    IdentifyAssociationsConfig,
    MOVEConfig,
)
from move.core.logging import get_logger


@hydra.main(config_path="conf", config_name="main")
def main(config: MOVEConfig) -> None:
    """Run MOVE.

    Example:
        $ python -m move experiment=random_small -cd=tutorial/config
    """
    if not hasattr(config, "task"):
        raise ValueError("No task defined.")
    task_type = OmegaConf.get_type(config.task)
    if task_type is None:
        logger = get_logger("move")
        logger.info("No task specified.")
    elif task_type is EncodeDataConfig:
        move.tasks.encode_data(config.data)
    elif task_type is AnalyzeLatentConfig:
        move.tasks.analyze_latent(config)
    elif issubclass(task_type, IdentifyAssociationsConfig):
        move.tasks.identify_associations(config)
    else:
        print(
            """
            To use MOVE pipeline, please run the following commands: 
            python -m move.01_encode_data
            python -m move.02_optimize_reconstruction
            python -m move.03_optimize_stability
            python -m move.04_analyze_latent
            python -m move.05_identify_associations
            
            To override the hyperparameter values, please write user defined values in the working directory in the following files:
            data.yaml, model.yaml, tuning_reconstruction.yaml, tuning_stability, training_latent.yaml, training_association.yaml. 
            """
        )

main()
