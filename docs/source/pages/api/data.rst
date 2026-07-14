Data
====

Datasets
----------

.. autoclass:: move.data.dataset.NamedDataset
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: move.data.dataset.DiscreteDataset
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: move.data.dataset.ContinuousDataset
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: move.data.dataset.MoveDataset
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: move.data.dataset.Perturbation
    :members:
    :undoc-members:

Dataloader
------------

.. autoclass:: move.data.dataloader.MoveDataLoader
    :members:
    :undoc-members:
    :show-inheritance:

Reading and writing files
----------------------------

.. autofunction:: move.data.io.read_config

.. autofunction:: move.data.io.read_names

.. autofunction:: move.data.io.read_tsv

.. autofunction:: move.data.io.dump_names

.. autofunction:: move.data.io.load_mappings

.. autofunction:: move.data.io.dump_mappings

.. autofunction:: move.data.io.sanitize_filename

.. autofunction:: move.data.io.load_preprocessed_data

.. note::
    :func:`~move.data.io.load_preprocessed_data` reads NumPy ``.npy`` files
    produced by an older pre-processing pipeline. Data encoded with the
    current :class:`~move.tasks.encode_data.EncodeData` task is saved as
    ``.pt`` files and should be loaded with
    :meth:`~move.data.dataset.MoveDataset.load` instead.

Pre-processing
----------------

.. autofunction:: move.data.preprocessing.one_hot_encode

.. autofunction:: move.data.preprocessing.one_hot_encode_single

.. autofunction:: move.data.preprocessing.standardize

.. autofunction:: move.data.preprocessing.log_n_standardize

.. autofunction:: move.data.preprocessing.fill

Splitting
-----------

.. autofunction:: move.data.splitting.split_samples

Writing CSV files
--------------------

.. autoclass:: move.data.writer.CsvWriter
    :members:
    :undoc-members:
    :show-inheritance:

Legacy data utilities
------------------------

.. warning::
    ``move.data.dataloaders`` and ``move.data.perturbations`` support the
    legacy, function-based tasks (``move.tasks.tune_model``,
    ``move.tasks.identify_associations``, ``move.tasks.analyze_latent``).
    New projects should use :class:`~move.data.dataset.MoveDataset` and
    :meth:`~move.data.dataset.MoveDataset.perturb` instead.

.. automodule:: move.data.dataloaders
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: move.data.perturbations
    :members:
    :undoc-members:

Reservoir sampling
--------------------

.. autoclass:: move.data.reservoir.Reservoir
    :members:
    :undoc-members:

.. autoclass:: move.data.reservoir.PairedReservoir
    :members:
    :undoc-members:
    :show-inheritance:
