__all__ = []

from omegaconf import OmegaConf

from move.conf.tasks import InputConfig


def extract_weights(configs: list[InputConfig]) -> list[int]:
    """Extracts the weights from a list of input configs."""
    return [1 if not hasattr(item, "weight") else item.weight for item in configs]


def extract_names(configs: list[InputConfig]) -> list[str]:
    """Extracts the weights from a list of input configs."""
    return [item.name for item in configs]


# Register custom resolvers
OmegaConf.register_new_resolver("weights", extract_weights)
OmegaConf.register_new_resolver("names", extract_names)
