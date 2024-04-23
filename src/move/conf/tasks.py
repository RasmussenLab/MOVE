__all__ = ["PcaConfig", "TsneConfig"]

from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from move.conf.schema import get_fully_qualname


@dataclass
class ReducerConfig:
    _target_: str
    n_components: int = 2


@dataclass
class PcaConfig(ReducerConfig):
    _target_: str = field(default=get_fully_qualname(PCA), init=False, repr=False)


@dataclass
class TsneConfig(ReducerConfig):
    _target_: str = field(default=get_fully_qualname(TSNE), init=False, repr=False)
    perplexity: float = 30.0


try:
    from umap import UMAP

    @dataclass
    class UmapConfig(ReducerConfig):
        _target_: str = field(default=get_fully_qualname(UMAP), init=False, repr=False)
        n_neighbors: int = 15

except (ModuleNotFoundError, SystemError, TypeError):
    pass


@dataclass
class PerturbationConfig:
    target_dataset_name: str
    target_feature_name: Optional[str]
    target_value: Union[float, int, str]
