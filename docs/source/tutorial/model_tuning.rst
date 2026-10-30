Tuning models
=============

The second step of MOVE's pipeline consists of training multiple models with
different hyperparameters in order to determine which set is optimal, i.e.,
produces models that generate the most accurate reconstructions and/or the most
stable latent representations.

The hyperparameters can be anything from number of training epochs, to size of
samples per batch, to number and size of hidden layers in the encoder-decoder
architecture.

This step is also available as notebooks in the ``tutorial/notebooks`` folder:
``02 Tune model`` (reconstruction) and ``03 Tune model stability``.

The ``experiment`` config
-------------------------

To start with this step, we define an experiment configuration (please
first consult the :doc:`introductory</tutorial/introduction>` and
:doc:`data preparation tutorial</tutorial/data_preparation>` if you have not
set up your workspace and data). This type of config references a data
config, a task config, and the values of hyperparameters to test out.

The first lines of our config should look like:

.. literalinclude:: /../../tutorial/config/experiment/random_small__tune_reconstruction.yaml
    :language: yaml
    :lines: 1-8

The ``override`` directives indicate (1) the name of our data config (in this
example we reference the config of our simulated dataset, see tutorial for
more info about this dataset) and (2) the name of the tuning task. There are
two possible values for tuning task:

- ``tune_model_reconstruction``, which reports the reconstruction accuracy of
  models trained with different hyperparameter combinations; and
- ``tune_model_stability``, which reports the stability of the latent space of
  differently hyperparameterized models.

Next, we have to define the hyperparameters that we wish to test out. An
example would be:

.. literalinclude:: /../../tutorial/config/experiment/random_small__tune_reconstruction.yaml
    :language: yaml
    :lines: 15-21

The above config would result in 12 hyperparameter combinations (2 options of
batch size times 2 options of encoder-decoder architecture times 3 options of
training epochs).

Any parameter of the training loop, model, and task can be swept. However, do
note that the more options you provide, the more models that will be trained,
and the more resource-intensive this task will become.

Below is a list of hyperparameters that we recommend tuning:

.. list-table:: Tunable hyperparameters
   :width: 100
   :widths: 40 60
   :header-rows: 1

   * - Hyperparameter
     - Description
   * - ``task.batch_size``
     - Number of samples per training batch
   * - ``task.model_config.num_hidden``
     - Architecture of the encoder network
       (reversed for the decoder network), e.g., ``"[500],[1000]"`` to compare
       one hidden layer of 500 units with one of 1000 units
   * - ``task.model_config.num_latent``
     - Number of dimensions of the latent space
   * - ``task.model_config.kl_weight``
     - Weight of the KL divergence (KLD) term in the loss. Higher values give a
       smoother, more regular latent space, at the cost of reconstruction
       accuracy.
   * - ``task.model_config.dropout_rate``
     - Fraction of units randomly dropped during training (regularization)
   * - ``task.training_loop_config.max_epochs``
     - Number of training epochs
   * - ``task.training_loop_config.annealing_epochs``
     - Number of epochs over which the KLD weight is gradually increased from 0
       to ``kl_weight`` at the start of training
   * - ``task.training_loop_config.annealing_function``
     - Shape of the KLD weight increase: ``linear`` (default), ``cosine``,
       ``sigmoid``, or ``stairs`` (increased once per epoch rather than at
       every batch)
   * - ``task.training_loop_config.annealing_schedule``
     - ``monotonic`` (default): the KLD weight is increased once, over the
       first ``annealing_epochs``, and then kept at ``kl_weight``.
       ``cyclical``: the increase is repeated in cycles of
       ``2 × annealing_epochs`` epochs throughout training, which can help
       avoid a latent space that the decoder ignores. Requires
       ``annealing_epochs`` above 0.
   * - ``task.training_loop_config.optimizer_config.lr``
     - Learning rate of the optimizer

Tuning for reconstruction
-------------------------

Finally, to run the tuning:

.. code-block:: bash

    >>> cd tutorial
    >>> move-dl experiment=random_small__tune_reconstruction

This process may take a while (depending on the number of hyperparameter
combinations that will be trained and tested), and it will produce a CSV table
``metrics.csv`` in a ``{results_path}/tuning`` directory. For each
hyperparameter combination, dataset, and split (training or test set), it
summarizes the reconstruction scores of the samples: accuracy for categorical
datasets and cosine similarity for continuous datasets. The table contains the
mean, median (``med``), quartiles (``q1``, ``q3``), and whiskers (``whislo``,
``whishi``) of the scores, which can be plotted as boxplots (see notebook
``02 Tune model``).

Look for combinations with high scores on the **test set**: these models
reconstruct samples they were not trained on well. A model that scores much
higher on the training set than on the test set is overfitting.
