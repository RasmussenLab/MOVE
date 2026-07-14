Configuration schemas
=====================

MOVE is driven by `Hydra <https://hydra.cc/>`_ configuration files, whose
structure is validated against the dataclasses documented on this page.

Main configuration
-------------------

.. autoclass:: move.conf.schema.MOVEConfig
    :members:
    :undoc-members:

.. autoclass:: move.conf.schema.DataConfig
    :members:
    :undoc-members:

Task configurations
--------------------

.. autoclass:: move.conf.schema.TaskConfig
    :members:
    :undoc-members:

.. autoclass:: move.conf.schema.EncodeDataConfig
    :members:
    :undoc-members:

.. autoclass:: move.conf.schema.MoveTaskConfig
    :members:
    :undoc-members:

.. autoclass:: move.conf.schema.LatentSpaceAnalysisConfig
    :members:
    :undoc-members:

.. autoclass:: move.conf.schema.AssociationsConfig
    :members:
    :undoc-members:

.. autoclass:: move.conf.schema.TuningConfig
    :members:
    :undoc-members:

.. autoclass:: move.conf.schema.StabilityTuningConfig
    :members:
    :undoc-members:

Input configurations
----------------------

Used in ``categorical_inputs``/``continuous_inputs`` (``data`` config) to
describe each dataset that should be pre-processed by the ``EncodeData`` task.

.. autoclass:: move.conf.tasks.InputConfig
    :members:

.. autoclass:: move.conf.tasks.DiscreteInputConfig
    :members:
    :undoc-members:

.. autoclass:: move.conf.tasks.ContinuousInputConfig
    :members:
    :undoc-members:

Dimensionality-reduction configurations
-----------------------------------------

Used in ``reducer_config`` (latent space analysis task) to further reduce the
latent space to a plottable number of dimensions.

.. autoclass:: move.conf.tasks.ReducerConfig
    :members:

.. autoclass:: move.conf.tasks.PcaConfig
    :members:
    :undoc-members:

.. autoclass:: move.conf.tasks.TsneConfig
    :members:

.. note::
    If the optional ``umap-learn`` dependency is installed, a
    ``move.conf.tasks.UmapConfig`` (wrapping ``umap.UMAP``, with an
    ``n_neighbors`` attribute) is also registered under this group.

Perturbation configuration
----------------------------

Used in ``perturbation_config`` (associations task) to describe the *in
silico* perturbation experiment to run.

.. autoclass:: move.conf.tasks.PerturbationConfig
    :members:

Model configurations
----------------------

.. autoclass:: move.conf.models.ModelConfig
    :members:
    :undoc-members:

.. autoclass:: move.conf.models.VaeConfig
    :members:
    :undoc-members:

.. autoclass:: move.conf.models.VaeNormalConfig
    :members:
    :undoc-members:

.. autoclass:: move.conf.models.VaeTConfig
    :members:
    :undoc-members:

Training loop and dataloader configurations
----------------------------------------------

.. autoclass:: move.conf.training.DataLoaderConfig
    :members:
    :undoc-members:

.. autoclass:: move.conf.training.TrainingDataLoaderConfig
    :members:
    :undoc-members:

.. autoclass:: move.conf.training.TestDataLoaderConfig
    :members:
    :undoc-members:

.. autoclass:: move.conf.training.TrainingLoopConfig
    :members:
    :undoc-members:

Optimizer configurations
---------------------------

.. autoclass:: move.conf.optim.OptimizerConfig
    :members:
    :undoc-members:

.. autoclass:: move.conf.optim.AdamConfig
    :members:
    :undoc-members:

.. autoclass:: move.conf.optim.AdamWConfig
    :members:
    :undoc-members:

.. autoclass:: move.conf.optim.ProdigyConfig
    :members:
    :undoc-members:

.. autoclass:: move.conf.optim.SgdConfig
    :members:
    :undoc-members:

Learning-rate scheduler configurations
------------------------------------------

.. autoclass:: move.conf.optim.LrSchedulerConfig
    :members:
    :undoc-members:

.. autoclass:: move.conf.optim.ExponentialLrConfig
    :members:
    :undoc-members:

.. autoclass:: move.conf.optim.CosineAnnealingLrConfig
    :members:
    :undoc-members:

.. autoclass:: move.conf.optim.ReduceLrOnPlateauConfig
    :members:
    :undoc-members:

Resolvers
-----------

Custom `OmegaConf resolvers <https://omegaconf.readthedocs.io/en/2.3_branch/custom_resolvers.html>`_
used to derive values (e.g., dataset names/weights) from the ``data`` config's
``categorical_inputs``/``continuous_inputs`` lists.

.. autofunction:: move.conf.resolvers.register_resolvers

.. autofunction:: move.conf.resolvers.extract_names

.. autofunction:: move.conf.resolvers.extract_weights

Legacy configurations
------------------------

.. warning::
    The dataclasses in ``move.conf.legacy`` configure the pre-refactor
    function-based tasks (``move.tasks.tune_model``,
    ``move.tasks.identify_associations``, ``move.tasks.analyze_latent``) and
    the legacy ``move.models.vae_legacy.VAE`` model. New projects should use
    the configurations documented above instead.

.. automodule:: move.conf.legacy
    :members:
    :undoc-members:
    :show-inheritance:
