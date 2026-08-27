__all__ = ["PcaConfig", "TsneConfig", "PerturbationConfig"]

from dataclasses import dataclass, field
from typing import Optional, Union

from omegaconf import MISSING
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from move.conf.config_store import config_store
from move.core.qualname import get_fully_qualname
from move.data.preprocessing import PreprocessingOpName


@dataclass
class InputConfig:
    """Configure a single dataset to be pre-processed by the ``EncodeData``
    task.

    Attributes:
        name: Name of the dataset
        weight: Weight given to this dataset's reconstruction loss
        preprocessing: Pre-processing operation to apply ('none',
            'one_hot_encode', or 'standardize')
    """

    name: str
    weight: int = 1
    preprocessing: PreprocessingOpName = "none"


@dataclass
class DiscreteInputConfig(InputConfig):
    """`InputConfig` defaulting to one-hot encoding, for discrete datasets."""

    preprocessing: PreprocessingOpName = "one_hot_encode"


@dataclass
class ContinuousInputConfig(InputConfig):
    """`InputConfig` defaulting to standardization, for continuous datasets."""

    preprocessing: PreprocessingOpName = "standardize"


@dataclass
class ReducerConfig:
    """Configure a dimensionality reduction algorithm used in latent space
    analysis.

    Attributes:
        _target_: Fully-qualified class name of the reducer to instantiate
        n_components: Number of dimensions to reduce the latent space to
    """

    _target_: str
    n_components: int = 2


@dataclass
class PcaConfig(ReducerConfig):
    """Configure a `sklearn.decomposition.PCA` reducer."""

    _target_: str = field(default=get_fully_qualname(PCA), init=False, repr=False)


@dataclass
class TsneConfig(ReducerConfig):
    """Configure a `sklearn.manifold.TSNE` reducer.

    Attributes:
        perplexity: Related to the number of nearest neighbors considered
    """

    _target_: str = field(default=get_fully_qualname(TSNE), init=False, repr=False)
    perplexity: float = 30.0


try:
    from umap import UMAP

    @dataclass
    class UmapConfig(ReducerConfig):
        """Configure a `umap.UMAP` reducer.

        Attributes:
            n_neighbors: Size of the local neighborhood used for manifold
                approximation
        """

        _target_: str = field(default=get_fully_qualname(UMAP), init=False, repr=False)
        n_neighbors: int = 15

except (ModuleNotFoundError, SystemError, TypeError):
    pass


@dataclass
class PerturbationConfig:
    """Configure a perturbation experiment used by the ``Associations`` task.

    Attributes:
        target_dataset_name: Name of dataset containing the target feature
        target_feature_name: Name of feature to perturb (all features in the
            dataset are perturbed if unset)
        target_value: Value the feature will be replaced with (only used
            when ``perturbation_type`` is ``"value"``)
        perturbation_type: How to perturb the target feature(s): ``"value"``
            (default), ``"minimum"``, ``"maximum"``, ``"plus_std"``, or
            ``"minus_std"`` — see :data:`move.data.dataset.PerturbationType`.
            ``"value"`` replaces it with ``target_value``, and works for both
            discrete and continuous datasets. The other four are
            continuous-only and ignore ``target_value``: ``"minimum"``/
            ``"maximum"`` replace the feature with its dataset-wide min/max;
            ``"plus_std"``/``"minus_std"`` add/subtract one standard
            deviation to/from each sample's own value instead of replacing
            it. 
    """

    target_dataset_name: str
    target_feature_name: Optional[str] = None
    target_value: Union[float, int, str] = MISSING
    perturbation_type: str = "value"


config_store.store(
    group="task/reducer_config",
    name="tsne",
    node=TsneConfig,
)
config_store.store(
    group="task/reducer_config",
    name="pca",
    node=PcaConfig,
)
config_store.store(
    group="task/perturbation_config",
    name="perturbation",
    node=PerturbationConfig,
)
