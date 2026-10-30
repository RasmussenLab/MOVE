Install
=======

MOVE is distributed as ``move-dl``, a Python package.

It requires Python 3.11 (or later) and third-party libraries, such as `PyTorch`_
and `Hydra`_. These dependencies will be installed automatically when you
install with ``pip``.

Install the stable version
--------------------------

We recommend installing ``move-dl`` in a fresh virtual environment. If you wish
to learn how to create and manage virtual environments with Conda, please
follow `these instructions`_. For example:

.. code-block:: bash

    >>> conda create -n move python=3.11
    >>> conda activate move

The latest stable version of ``move-dl`` can be installed with ``pip``.

.. code-block:: bash

    >>> pip install move-dl

You can check that the installation worked by printing the installed version:

.. code-block:: bash

    >>> python -c "import move; print(move.__version__)"

Install the development version
-------------------------------

If you wish to install the latest development version of ``move-dl``, create a
new virtual environment, and do:

.. code-block:: bash

    >>> pip install git+https://github.com/RasmussenLab/MOVE@main

Alternatively, you can clone ``move-dl`` from `GitHub`_ and install by
running the following command from the top-level source directory:

.. code-block:: bash

    >>> git clone https://github.com/RasmussenLab/MOVE.git
    >>> cd MOVE
    >>> pip install -e .

The ``-e`` flag installs the project in "editable" mode, so you can update your
installation by pulling the latest changes from GitHub. Cloning the repository
also gives you the tutorial data, configuration files, and notebooks (see the
:doc:`tutorials</tutorial/index>`).

Using a GPU
-----------

MOVE runs fine on a CPU: the tutorial dataset can be analyzed on a standard
laptop. Training is faster on a GPU, but the ``pip`` installation does **not**
set up GPU support automatically. To use a GPU:

1. Install a CUDA-enabled PyTorch build that matches your system, following the
   instructions at `pytorch.org`_.
2. Enable it in the model configuration of your task by setting
   ``use_cuda: true`` under ``model_config``, or by adding
   ``task.model_config.use_cuda=true`` to the ``move-dl`` command.

.. _PyTorch: https://pytorch.org/
.. _Hydra: https://hydra.cc/
.. _GitHub: https://github.com/RasmussenLab/MOVE
.. _pytorch.org: https://pytorch.org/get-started/locally/

.. _these instructions: https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html
