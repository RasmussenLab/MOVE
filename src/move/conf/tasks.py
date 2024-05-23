__all__ = ["PcaConfig", "TsneConfig", "PerturbationConfig"]

from dataclasses import dataclass, field
from typing import Optional, Union

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from move.core.qualname import get_fully_qualname
from move.data.preprocessing import PreprocessingOpName


@dataclass
class InputConfig:
    name: str
    weight: int = 1
    preprocessing: PreprocessingOpName = "none"


@dataclass
class DiscreteInputConfig(InputConfig):
    preprocessing: PreprocessingOpName = "one_hot_encoding"


@dataclass
class ContinuousInputConfig(InputConfig):
    preprocessing: PreprocessingOpName = "standardization"


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
