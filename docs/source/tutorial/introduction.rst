Introduction to MOVE
====================

This guide can help you get started with MOVE. It walks through the whole
pipeline on a small simulated dataset, using the ``move-dl`` command. The same
steps are also available as Jupyter notebooks in the ``tutorial/notebooks``
folder of our GitHub repository.

About the MOVE pipeline
-----------------------

MOVE has the following four steps (also called tasks):

1. **Encoding data**: taking input data and converting it into a format MOVE
   can read.
2. **Tuning the hyperparameters of the VAE model**: training multiple models to
   find the set of hyperparameters that produces the best reconstructions or
   most stable latent space.
3. **Analyzing the latent space**: training a model and inspecting the latent
   representation it creates.
4. **Identifying associations**: using an ensemble of VAEs to find associations
   between the input datasets.

Simulated dataset
-----------------

For this short tutorial, we provide a `simulated dataset`_ (available from our
GitHub repository). This dataset consists of simulated proteomics and
metagenomics measurements for 500 fictitious individuals. We also report
whether these individuals have taken or not 20 imaginary drugs.

All values were randomly generated, but we have added 200
associations between different pairs of drugs and omics features. Let us find
these associations with MOVE!

.. _simulated dataset: https://download-directory.github.io/?url=https%3A%2F%2Fgithub.com%2FRasmussenLab%2FMOVE%2Ftree%2Fmain%2Ftutorial

Workspace structure
-------------------

First, we take a look at how to organize our data and configuration:::

    tutorial/
    │
    ├── data/
    │   ├── changes.small.txt              <- Ground-truth associations (200 links)
    │   ├── random.small.drugs.tsv         <- Drug dataset (20 drugs)
    │   ├── random.small.ids.txt           <- Sample IDs (500 samples)
    │   ├── random.small.proteomics.tsv    <- Proteomics dataset (200 proteins)
    │   └── random.small.metagenomics.tsv  <- Metagenomics dataset (1000 taxa)
    │
    └── config/                            <- Stores user configuration files
        ├── data/
        │   └── random_small.yaml          <- Configuration to read in the necessary
        │                                     data files.
        ├── experiment/                    <- Configuration for experiments (e.g.,
        │   │                                 for tuning hyperparameters).
        │   ├── random_small__tune_reconstruction.yaml
        │   └── random_small__tune_stability.yaml
        │
        └── task/                          <- Configuration for tasks: such as
            │                                 latent space or identify associations
            ├── random_small__id_assoc_bayes.yaml
            └── random_small__latent.yaml

The data directory
^^^^^^^^^^^^^^^^^^

All "raw" data files should be placed inside the same directory. These files
are TSVs (tab-separated value tables) containing discrete values (e.g., for
binary or categorical datasets) or continuous values. See the
:doc:`data preparation tutorial</tutorial/data_preparation>` for how to format
your own data.

Additionally, make sure each sample has an assigned ID and provide an ID
table containing a list of all valid IDs (must appear in every dataset).

The ``config`` directory
^^^^^^^^^^^^^^^^^^^^^^^^

Configuration is composed and managed by `Hydra`_.

User-defined configuration must be stored in a ``config`` folder. This folder
can contain a ``data``, ``task``, and ``experiment`` folder to store the
configuration files for your dataset, tasks, and hyperparameter tuning.

.. _`Hydra`: https://hydra.cc/

Data configuration
""""""""""""""""""

Let us take a look at the configuration for our dataset. It is a YAML file,
specifying: the directories to look for raw data and store intermediary and
final output files, and the list of categorical and continuous datasets we
have.

.. literalinclude:: /../../tutorial/config/data/random_small.yaml
    :language: yaml

Note that we do not recommend changing the ``defaults`` field, otherwise the
configuration file will not be properly recognized by MOVE.

Task configuration
""""""""""""""""""

Similarly, the ``task`` folder contains YAML files to configure the tasks of
MOVE. In this tutorial, we provide an example to perform latent space analysis
and an example to identify associations.

For example, to identify associations (``random_small__id_assoc_bayes.yaml``),
we define the following values: batch size, number of refits (models in the
ensemble), name of the dataset to perturb, value to perturb it to,
configuration for the VAE model, and configuration for the training loop.

.. literalinclude:: /../../tutorial/config/task/random_small__id_assoc_bayes.yaml
    :language: yaml

The ``defaults`` list at the top tells MOVE which task this config is for. All
settings that are not listed take MOVE's default values.

Config files are referenced by their file name, without the ``.yaml``
extension. For example, ``data=random_small task=random_small__latent`` reads
``config/data/random_small.yaml`` and ``config/task/random_small__latent.yaml``.

Changing settings from the command line
"""""""""""""""""""""""""""""""""""""""

Any setting can be changed for a single run by adding ``key=value`` to the
command, using dots to reach nested settings. For example, to train for more
epochs with a smaller latent space:

.. code-block:: bash

    >>> move-dl data=random_small task=random_small__latent \
            task.training_loop_config.max_epochs=100 \
            task.model_config.num_latent=50

|:bulb:| To check the final configuration without running anything, add
``--cfg job`` to a ``move-dl`` command. MOVE prints the complete configuration,
including all default values.

Running MOVE
------------

Encoding data
^^^^^^^^^^^^^

Make sure you are on the parent directory of the ``config`` folder (in this
example, it is the ``tutorial`` folder), and proceed to run:

.. code-block:: bash

    >>> cd tutorial
    >>> move-dl data=random_small task=encode_data

|:arrow_up:| This command will encode the datasets. The ``random.small.drugs``
dataset (defined in ``config/data/random_small.yaml``) will be one-hot encoded,
whereas the other two omics datasets will be standardized. Encoded data will
be placed in the intermediary folder defined in the
:ref:`data config<Data configuration>`.

|:loud_sound:| Every ``move-dl`` command will generate a ``logs`` folder to
store log files timestamping the program's current doings.

Tuning the model's hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once the data has been encoded, we can proceed with the first step of our
pipeline: tuning the hyperparameters of our deep learning model. This process
can be time-consuming, because several models will be trained and tested. For
this short tutorial, you may choose to skip it and proceed to
:ref:`analyze the latent space<Analyzing the latent space>`. See
:doc:`/tutorial/model_tuning` for how to tune the hyperparameters.

Analyzing the latent space
^^^^^^^^^^^^^^^^^^^^^^^^^^

Next, we will train a variational autoencoder and analyze how good it is at
reconstructing our input data and generating an informative latent space. Run:

.. code-block:: bash

    >>> move-dl data=random_small task=random_small__latent

|:arrow_up:| This command will create a ``latent_space`` directory in the
results folder defined in the :ref:`data config<Data configuration>`. This
folder will contain the following plots:

* **Loss curve** (``loss_curve.png``) shows the overall loss and each of its
  components over the training epochs: the Kullback-Leibler divergence (KLD)
  term, the reconstruction error of the categorical datasets (cross-entropy),
  and the reconstruction error of the continuous datasets (sum of squared
  errors).
* **Reconstruction metrics boxplot** (``reconstruction_metrics.png``) shows a
  score per sample (accuracy or cosine similarity for categorical and
  continuous datasets, respectively) for each reconstructed dataset.
* **Latent space scatterplots** (``latent_space_<feature>.png``) show a reduced
  representation of the latent space, with the samples colored by one of the
  features listed in ``features_to_plot``. To generate this visualization, the
  latent space is reduced to two dimensions using t-SNE (or PCA). Samples with a
  missing value for the feature are labeled "NaN" in the legend.
* **Feature importance swarmplots** (``feature_importance_<dataset>.png``)
  display the impact perturbing a feature has on the latent space.

Additionally, CSV files corresponding to each plot will be generated. These can
be used, for example, to re-create the plots manually or with different
styling. The trained model is saved as ``model.pt``.

The main settings of this task are:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Setting
     - Description
   * - ``features_to_plot``
     - List of feature names. For each one, a plot of the latent space is
       created with the samples colored by that feature.
   * - ``reducer_config``
     - Method used to reduce the latent space to two dimensions for plotting:
       ``tsne`` (default) or ``pca``. Select it in the ``defaults`` list, e.g.,
       ``- override reducer_config: pca``.
   * - ``compute_accuracy_metrics``
     - Whether to compute how well each dataset is reconstructed
   * - ``compute_feature_importance``
     - Whether to compute how much each feature affects the latent space

Identifying associations
^^^^^^^^^^^^^^^^^^^^^^^^

Next step is to find associations between the drugs taken by each individual
and the omics features. Run:

.. code-block:: bash

    >>> move-dl data=random_small task=random_small__id_assoc_bayes

|:arrow_up:| This command trains an ensemble of models, and uses them to
simulate giving each drug to every individual in the test set. For each drug,
it records which omics features consistently change in response. The results
are saved in ``associations/associations.csv`` in the results folder, listing
each significant pair of drug and omics feature. See
:ref:`Reading the association results` for what each column means.

|:warning:| Note that the value after ``task=`` matches the name of our
configuration file. We can create multiple configuration files (for example,
changing hyperparameters like learning rate) and call them by their name here.

|:stopwatch:| This command takes approximately 45 min to run on a work laptop
(Intel Core i7-10610U @ 1.80 GHz, 32 GB RAM). You can track the progress by
checking the corresponding log file created in the ``logs`` folder.

If you want to try the Bayesian approach instead, run:

.. code-block:: bash

    >>> move-dl data=random_small task=random_small__id_assoc_bayes

Again, it should generate similar results with over 120 associations known.

Take a look at the ``changes.small.txt`` file and compare your results against
it. The exact numbers vary from run to run. Did MOVE find any
false positives in your run?

The main settings of this task are:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Setting
     - Description
   * - ``num_refits``
     - Number of models in the ensemble. More models give more reliable
       results, but take longer to train.
   * - ``perturbation_config.target_dataset_name``
     - Dataset whose features are perturbed (e.g., the drug dataset)
   * - ``perturbation_config.target_feature_name``
     - A single feature to perturb. If not set, every feature of the dataset is
       perturbed, one at a time.
   * - ``perturbation_config.perturbation_type``
     - How features are perturbed. ``value`` (default) sets the feature to
       ``target_value``, and works for categorical and continuous datasets. For
       continuous datasets, there is also ``minimum`` and ``maximum`` (set to the
       feature's lowest or highest value in the data) and ``plus_std`` and
       ``minus_std`` (add or subtract one standard deviation).
   * - ``perturbation_config.target_value``
     - Value to set the feature to, when ``perturbation_type`` is ``value``.
       For categorical data, this is a category, e.g., ``1`` for "took the
       drug".
   * - ``sig_threshold``
     - False discovery rate (FDR) threshold for an association to be called
       significant (default 0.05)
   * - ``write_only_sig``
     - Whether to save only the significant associations (default ``true``) or
       all pairs of features

The trained models are saved as ``model_*.pt`` next to the results.

Reading the association results
"""""""""""""""""""""""""""""""

The ``associations.csv`` file has one row per pair of a perturbed feature and a
target feature (a feature of a continuous dataset), sorted from strongest to
weakest:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Column
     - Description
   * - ``perturbed_feature``
     - The feature that was perturbed (e.g., a drug)
   * - ``target_feature``
     - The continuous feature whose reconstruction was measured
   * - ``prob``
     - How consistently the perturbation moves the target feature in the same
       direction across the samples of the test set (using the change averaged
       over all models of the ensemble). Ranges from 0.5 (no consistent effect)
       to 1 (the same direction in every sample).
   * - ``bayes_k``
     - Bayes factor (log scale) of the effect. Its **sign** gives the
       direction: positive means the target feature increases when the feature
       is perturbed, negative means it decreases. Larger absolute values mean
       stronger evidence.
   * - ``fdr``
     - Estimated false discovery rate when calling this pair and all pairs above
       it significant. Pairs with an FDR below ``sig_threshold`` are
       significant.

Running MOVE from a notebook or script
--------------------------------------

The steps above can also be run from Python. There are two ways of doing this,
and both produce the same results as the command line.s

**Using the YAML config files.** The configuration is read with
``read_config`` and turned into a task, which is then run. Command-line style
overrides can be passed as extra arguments:

.. code-block:: python

    import hydra
    from move.data import io

    config = io.read_config(
        "random_small",
        "random_small__latent",
        "task.training_loop_config.max_epochs=100",
    )
    task = hydra.utils.instantiate(config.task, _recursive_=False)
    task.run()

This is what tutorial notebooks 01–05 do. Because the same config files are
used, you can switch between notebooks and the command line.

**From Python only, without config files.** Every task can also be built
directly from Python objects. This is handy for a self-contained script or a
quick run on a server. The ``Standalone pipeline`` notebook in the
``tutorial/notebooks`` folder shows the whole pipeline written this way.

In all cases, paths in the configuration (such as ``data/`` or ``results/``)
are relative to the **working directory**, i.e., the folder you run MOVE from.
This folder must also contain the ``config`` folder.

Re-running tasks
----------------

**Saved models are overwritten by default.** Re-running a task trains new
models, so the results always match the current data and settings. To save
time, you can reload the models of a previous run instead by setting
``reuse_models: true`` in the task config, or adding ``task.reuse_models=true``
to the command. Only do this if nothing has changed since the models were
trained:

Next steps
----------

* :doc:`/tutorial/data_preparation` shows how to format your own data.
* :doc:`/tutorial/model_tuning` explains how to find good hyperparameters.
