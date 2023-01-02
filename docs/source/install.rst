Install
=======

MOVE is distributed as ``move-dl``, a Python package.

It requires Python 3.9 (or later) and third-party libraries, such as `PyTorch`_
and `Hydra`_. These dependencies will be installed automatically when you
install with ``pip``.

Install the stable version
--------------------------

We recommend installing ``move-dl`` in a fresh virtual environment. If you wish
to learn how to create and manage virtual environments with Conda, please
follow `these instructions`_.

The latest stable version of ``move-dl`` can be installed with ``pip``.

.. code-block:: bash

    >>> pip install move-dl

Install the development version
-------------------------------

If you wish to install the development of ``move-dl``, create a new virtual
environment, and do:

.. code-block:: bash

    >>> pip install git+https://github.com/RasmussenLab/MOVE@developer

Alternatively, you can clone ``move-dl`` from `GitHub`_ and install by
running the following command from the top-level source directory:

.. code-block:: bash

    >>> pip install -e .

The ``-e`` flag installs the project in "editable" mode, so you can follow the
development branch and update your installation by pulling from GitHub.

.. _PyTorch: https://pytorch.org/
.. _Hydra: https://hydra.cc/
.. _GitHub: https://github.com/RasmussenLab/MOVE

.. _these instructions: https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html
