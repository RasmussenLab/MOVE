import click

from move.modules.encode_data import main as f_encode_data 
from move.modules.optimize_reconstruction import main as f_optimize_reconstruction
from move.modules.optimize_stability import main as f_optimize_stability
from move.modules.analyze_latent import main as f_analyze_latent
from move.modules.identify_associations import main as f_identify_associations
from move.utils.data_utils import merge_configs

###################################
# CLICK COMMAND GROUP DECLARATION
###################################

@click.group()
def cli():
    pass


@cli.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option('--data_cfg', required=True, type=click.Path(exists=True))
def encode_data(data_cfg):
    input_args = locals()
    cfgs_save = None
    
    config = merge_configs(cfgs_save, **input_args)
    f_encode_data(config)

@cli.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option('--data_cfg', required=True, type=click.Path(exists=True))
@click.option('--model_cfg', required=True, type=click.Path(exists=True))
@click.option('--reconstruction_cfg', required=True, type=click.Path(exists=True))
@click.option('--stability_cfg', required=True, type=click.Path())
def optimize_reconstruction(data_cfg, model_cfg, reconstruction_cfg, stability_cfg):
    input_args=locals()
    cfgs_save = (stability_cfg)
    
    config = merge_configs(Mcfgs_save, **input_args)
    f_optimize_reconstruction(config, cfgs_save)
    
    
@cli.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option('--data_cfg', required=True, type=click.Path(exists=True))
@click.option('--model_cfg', required=True, type=click.Path(exists=True))
@click.option('--stability_cfg', required=True, type=click.Path(exists=True))
@click.option('--latent_cfg', required=True, type=click.Path())
@click.option('--association_cfg', required=True, type=click.Path())
def optimize_stability(data_cfg, model_cfg, stability_cfg, latent_cfg, association_cfg):
    input_args=locals()
    cfgs_save = (latent_cfg, association_cfg)
    
    config = merge_configs(cfgs_save, **input_args)
    f_optimize_stability(config, cfgs_save)

    
@cli.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option('--data_cfg', required=True, type=click.Path(exists=True))
@click.option('--model_cfg', required=True,  type=click.Path(exists=True))
@click.option('--latent_cfg', required=True, type=click.Path(exists=True))
def analyze_latent(data_cfg, model_cfg, latent_cfg):
    input_args=locals()
    cfgs_save = None
    
    config = merge_configs(cfgs_save, **input_args)
    f_analyze_latent(config)

    
@cli.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option('--data_cfg', required=True, type=click.Path(exists=True))
@click.option('--model_cfg', required=True, type=click.Path(exists=True))
@click.option('--association_cfg', required=True, type=click.Path(exists=True))
def identify_associations(data_cfg, model_cfg, association_cfg):
    input_args=locals()
    cfgs_save = None
    
    config = merge_configs(cfgs_save=None, **input_args)
    f_identify_associations(config)    

    
if __name__ == "__main__":
    cli()