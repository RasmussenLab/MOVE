Tasks
=====

Base classes
--------------

.. autoclass:: move.tasks.base.Task
    :members:
    :undoc-members:

.. autoclass:: move.tasks.base.ParentTask
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: move.tasks.base.SubTask
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: move.tasks.base.InputDirMixin
    :members:
    :undoc-members:

.. autoclass:: move.tasks.base.OutputDirMixin
    :members:
    :undoc-members:

.. autoclass:: move.tasks.base.LoggerMixin
    :members:
    :undoc-members:

.. autoclass:: move.tasks.base.SubTaskMixin
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: move.tasks.base.CsvWriterMixin
    :members:
    :undoc-members:

.. autoclass:: move.tasks.base.OutputDir
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: move.tasks.move.MoveTask
    :members:
    :undoc-members:
    :show-inheritance:

Data encoding
---------------

.. autoclass:: move.tasks.encode_data.EncodeData
    :members:
    :undoc-members:
    :show-inheritance:

Training a model
-------------------

.. autoclass:: move.tasks.train_model.TrainModel
    :members:
    :undoc-members:
    :show-inheritance:

Hyperparameter tuning
------------------------

.. autoclass:: move.tasks.tuning.TuneModel
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: move.tasks.tuning.TuneStability
    :members:
    :undoc-members:
    :show-inheritance:

Association testing
-----------------------

.. autoclass:: move.tasks.associations.Associations
    :members:
    :undoc-members:
    :show-inheritance:

Latent space analysis
------------------------

.. autoclass:: move.tasks.latent_space_analysis.LatentSpaceAnalysis
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: move.tasks.latent_space_analysis.Project
    :members:
    :undoc-members:
    :show-inheritance:

Legacy tasks
--------------

.. warning::
    These function-based tasks predate the class-based tasks documented
    above and rely on ``move.conf.legacy`` and
    ``move.models.vae_legacy.VAE``. New projects should use
    :class:`~move.tasks.tuning.TuneModel`/:class:`~move.tasks.tuning.TuneStability`,
    :class:`~move.tasks.associations.Associations`, and
    :class:`~move.tasks.latent_space_analysis.LatentSpaceAnalysis` instead.

.. autofunction:: move.tasks.tune_model.tune_model

.. autofunction:: move.tasks.identify_associations.identify_associations

.. autofunction:: move.tasks.analyze_latent.analyze_latent

.. autofunction:: move.tasks.analyze_latent.find_feature_values
