Tutorial
========

This guide can help you get started with MOVE.

Simulated dataset
-----------------

For this short tutorial, we provide `simulated dataset`_ (available from our
GitHub repository). This dataset consists of pretend proteomics and 
metagenomcis measurements for 500 fictitious individuals. We also report
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
    │   ├── random.small.ids.tsv           <- Sample IDs (500 samples)
    │   ├── random.small.proteomics.tsv    <- Proteomics dataset (200 proteins)
    │   └── random.small.metagenomics.tsv  <- Metagenomics dataset (1000 taxa)
    │
    └── config/                            <- Stores user configuration files
        ├── data/
        │   └── random_small.yaml          <- Configuration to read in the necessary
        │                                     data files.
        ├── experiment/                    <- Configuration for experiments (e.g.,
        │   └── random_small__tune.yaml       for tuning hyperparameters).
        │
        └── task/                          <- Configuration for tasks: such as
            |                                 latent space or identify associations
            │                                 using the t-test or Bayesian approach
            ├── random_small__id_assoc_bayes.yaml
            ├── random_small__id_assoc_ttest.yaml
            └── random_small__latent.yaml

The data directory
^^^^^^^^^^^^^^^^^^

All "raw" data files should be placed inside the same directory. These files
are TSVs (tab-separated value tables) containing discrete values (e.g., for
binary or categorical datasets) or continuous values.

Additionally, make sure each sample has an assigned ID and provide an ID
table containing a list of all valid IDs (must appear in every dataset).

The ``config`` directory
^^^^^^^^^^^^^^^^^^^^^^^^

Configuration is composed and managed by `Hydra`_.

User-defined configuration must be stored in a ``config`` folder. This folder
can contain a ``data`` and ``task`` folder to store the configuration files for
your dataset and tasks.

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
MOVE. In this tutorial, we provided two examples for running the method to
identify associations using our t-test and Bayesian approach, and an example to
perform latent space analysis.

For example, for the t-test approach (``random_small__id_assoc_ttest.yaml``),
we define the following values: batch size, number of refits, name of dataset to
perturb, target perturb value, configuration for VAE model, and configuration
for training loop.

.. literalinclude:: /../../tutorial/config/task/random_small__id_assoc_ttest.yaml
    :language: yaml

Note that the ``random_small__id_assoc_bayes.yaml`` looks pretty similar, but
declares a different ``defaults``. This tells MOVE which algorithm to use!

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
:ref:`analyze the latent space<Analyzing the latent space>`.

Analyzing the latent space
^^^^^^^^^^^^^^^^^^^^^^^^^^

Next, we will train a variational autoencoder and analyze how good it is at
reconstructing our input data and generating an informative latent space. Run:

.. code-block:: bash

    >>> move-dl data=random_small task=random_small__latent

|:arrow_up:| This command will create a ``latent_space`` directory in the 
results folder defined in the :ref:`data config<Data configuration>`. This
folder will contain the following plots:

* **Loss curve** shows the overall loss, KLD term, binary cross-entropy term,
  and sum of squared errors term over number of training epochs.
* **Reconstructions metrics boxplot** shows a score (accuracy or cosine
  similarity for categorical and continuous datasets, respectively) per
  reconstructed dataset.
* **Latent space scatterplot** shows a reduced representation of the latent 
  space. To generate this visualization, the latent space is reduced to two  
  dimensions using TSNE (or another user-defined algorithm, e.g., UMAP).
* **Feature importance swarmplot** displays the impact perturbing a feature has
  on the latent space.

Additionally, TSV files corresponding to each plot will be generated. These can
be used, for example, to re-create the plots manually.

Identifying associations
^^^^^^^^^^^^^^^^^^^^^^^^

Next step is to find associations between the drugs taken by each individual
and the omics features. Run:

.. code-block:: bash

    >>> move-dl data=random_small task=random_small__id_assoc_ttest

|:arrow_up:| This command will create a ``results_sig_assoc.tsv`` file, listing
each pair of associated features and the corresponding median p-value for such
association. There should be ~120 associations found. Due to the nature of the
method, this number may slightly fluctuate.

|:warning:| Note that the value after ``task=`` matches the name of our
configuration file. We can create multiple configuration files (for example,
changing hyperparameters like learning rate) and call them by their name here.

|:stopwatch:| This command takes approximately 45 min to run on a work laptop
(Intel Core i7-10610U @ 1.80 GHz, 32 MB RAM). You can track the progress by
checking the corresponding log file created in the ``logs`` folder.

If you want to run, the Bayesian approach instead. Run:

.. code-block:: bash

    >>> move-dl data=random_small task=random_small__id_assoc_bayes

Again, it should generate similar results with over 120 associations known.

Take a look at the ``changes.small.txt`` file and compare your results against
it. Did MOVE find any false positives?
