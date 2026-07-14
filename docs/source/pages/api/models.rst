Models
======

Base class
------------

.. autoclass:: move.models.base.BaseVae
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: move.models.base.VaeOutput
    :members:
    :undoc-members:

.. autoclass:: move.models.base.LossDict
    :members:
    :undoc-members:

.. autoclass:: move.models.base.SerializedModel
    :members:
    :undoc-members:

.. autofunction:: move.models.base.reload_vae

Variational autoencoders
---------------------------

.. autoclass:: move.models.vae.Vae
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: move.models.vae_distribution.VaeDistribution
    :members:
    :undoc-members:
    :show-inheritance:

.. note::
    ``move.models.vae_distribution.VaeNormal`` is an alias of
    :class:`~move.models.vae_distribution.VaeDistribution` (Normal-distributed
    continuous decoder).

.. autoclass:: move.models.vae_t.VaeT
    :members:
    :undoc-members:
    :show-inheritance:

Layers
--------

.. autofunction:: move.models.layers.encoder_decoder.build_network

.. autoclass:: move.models.layers.encoder_decoder.Encoder
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: move.models.layers.encoder_decoder.Decoder
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: move.models.layers.chunk.Chunk
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: move.models.layers.chunk.SplitOutput
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: move.models.layers.chunk.SplitInput
    :members:
    :undoc-members:
    :show-inheritance:

Legacy model
--------------

.. warning::
    ``move.models.vae_legacy.VAE`` is the pre-refactor VAE implementation
    (it does not derive from :class:`~move.models.base.BaseVae`). New
    projects should use :class:`~move.models.vae.Vae` or
    :class:`~move.models.vae_distribution.VaeDistribution` instead.

.. autoclass:: move.models.vae_legacy.VAE
    :members:
    :undoc-members:
