__all__ = ["config_store"]

from hydra.core.config_store import ConfigStore

config_store = ConfigStore.instance()
