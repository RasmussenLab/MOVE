Tuning models
=============

The second of MOVE's pipeline consists of training multiple models with
different hyperparameters in order to determine which set is optimal, i.e.,
produces models that generate the most accurate reconstructions and/or the most
stable latent representations.

The hyperparameters can be anything from number of training epochs, to size of
samples per batch, to number and size of hidden layers in the encoder-decoder
architecture.

The ``experiment`` config
-------------------------

To start with this step, we define a experiment configuration (please
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
   * - ``task.model.num_hidden``
     - Architecture of the encoder network
       (reversed for the decoder network)
   * - ``task.model.num_latent``
     - Number of units of the latent space
   * - ``task.model.beta``
     - Weight applied to the KLD term in the loss function
   * - ``task.model.dropout``
     - Dropout
   * - ``task.training_loop.num_epochs``
     - Number of training epochs
   * - ``task.training_loop.lr``
     - Learning rate
   * - ``task.training_loop.kld_warmup_steps``
     - Epochs at which KLD is warmed
   * - ``task.training_loop.batch_dilation_steps``
     - Epochs at which batch size is increased
   * - ``task.training_loop.early_stopping``
     - Whether early stopping is triggered

Finally, to run the tuning:

.. code-block:: bash

    >>> cd tutorial
    >>> move-dl experiment=random_small__tune_reconstruction

This process may take a while (depending on the number of hyperparameter
combinations that will be trained and tested), and it will produce a TSV table
in a ``{results_path}/tune_model`` directory summarizing the metrics (either
reconstruction metrics like accuracy or stability). These metrics can be
plotted to visualize and select the optimal hyperparameter combination.
