Training
========

Training loop
---------------

.. autoclass:: move.training.loop.TrainingLoop
    :members:
    :undoc-members:
    :show-inheritance:

Optimizers
------------

.. autoclass:: move.training.optim.prodigy.Prodigy
    :members:
    :undoc-members:
    :show-inheritance:

Legacy training loop
-----------------------

.. warning::
    ``move.training.training_loop`` trains the legacy
    ``move.models.vae_legacy.VAE`` model. New projects should use
    :class:`~move.training.loop.TrainingLoop` instead.

.. automodule:: move.training.training_loop
    :members:
    :undoc-members:
