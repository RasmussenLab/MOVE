Data preparation
================

In this tutorial, we explain how to make your data compatible with the
``move-dl`` commands.

For this tutorial we will work with a dataset taken from Walters et al. (2008)
[#]_. In their work, they report soil microbiome census data along with
environmental data (e.g., temperature and precipitation) of different cultivars
of maize.

We will start by downloading the files corresponding to their `OTU table`_
and `metadata`_.

Formatting omics data
---------------------

The ``move-dl`` pipeline requires continuous omics input to be formatted as a 
TSV file with one column per feature and one row per feature.

If we load the microbiome OTU table from the maize rhizosphere dataset, it will
look something like this:

.. table:: Original OTU table

    ========  ======================  ======================  ======================
    otuids    11116.C02A66.1194587    11116.C06A63.1195666    11116.C08A61.1197689
    ========  ======================  ======================  ======================
    4479944                      70                       8                      18
    513055                       2                      16                       1
    519510                      22                      15                      12
    810959                       5                       0                       3
    849092                       5                       2                       1
    ========  ======================  ======================  ======================

We have columns corresponding to samples and rows corresponding to features
(OTUs), so we need to **transpose** this table for MOVE.

.. table:: Transposed OTU table

    ====================  =========  ========  ========  ========  ========
    sampleids               4479944    513055    519510    810959    849092
    ====================  =========  ========  ========  ========  ========
    11116.C02A66.1194587         70         2        22         5         5
    11116.C06A63.1195666          8        16        15         0         2
    11116.C08A61.1197689         18         1        12         3         1
    ====================  =========  ========  ========  ========  ========

Now, we can save our table as a TSV and we are ready to go. No need to do any
further processing.

Formatting other continuous data
--------------------------------

Other non-omics continuous data is formatted in a similar way.

For this tutorial, we are going to extract some continuous data from the maize
metadata table. Let us load the table and take a peek:

.. table:: Original metadata table

    ====================  ====================  =========  ===============  ==============
    X.SampleID              Precipitation3Days  INBREDS    Maize_Line       Description1
    ====================  ====================  =========  ===============  ==============
    11116.C02A66.1194587                  0.14  Oh7B       Non_Stiff_Stalk  rhizosphere
    11116.C06A63.1195666                  0.14  P39        Sweet_Corn       rhizosphere
    11116.C08A61.1197689                  0.14  CML333     Tropical         rhizosphere
    11116.C08A63.1196825                  0.14  CML333     Tropical         rhizosphere
    11116.C12A64.1197667                  0.14  Il14H      Sweet_Corn       rhizosphere
    ====================  ====================  =========  ===============  ==============

The original metadata table contains both categorical (e.g., ``Maize_Line``)
and continuous data (e.g., ``Precipitation3Days``). We need to separate these
into different files.

In this example, we select three columns: ``age``, ``Precipitation3Days``, and
``Temperature``.

.. table:: Extracted continuous data

    ====================  =====  =============  ====================
    X.SampleID              age    Temperature    Precipitation3Days
    ====================  =====  =============  ====================
    11116.C02A66.1194587     12             76                  0.14
    11116.C06A63.1195666     12             76                  0.14
    11116.C08A61.1197689     12             76                  0.14
    11116.C08A63.1196825     12             76                  0.14
    11116.C12A64.1197667     12             76                  0.14
    ====================  =====  =============  ====================

Once again, we can save this table as a TSV, and we are ready to continue.

Formatting categorical data
---------------------------

Categorical data like binary variables (e.g., with/without treatment) or
discrete categories needs to be formatted in individual files.

The metadata table contains several discrete variables that can be useful for
classification, such as maize line, cultivar, and type of soil. For each one of
these, we need to create a separate TSV file that will look something like:

.. table:: Extracted maize line data

    ====================  ===============
    X.SampleID            Maize_Line
    ====================  ===============
    11116.C02A66.1194587  Non_Stiff_Stalk
    11116.C06A63.1195666  Sweet_Corn
    11116.C08A61.1197689  Tropical
    11116.C08A63.1196825  Tropical
    11116.C12A64.1197667  Sweet_Corn
    ====================  ===============

Creating a data config file
---------------------------

We are missing two components to make our data compatible with ``move-dl``.
First, we need to create an additional text file with all the sample IDs (one
ID per line, see example below). This file tells MOVE which samples to use, so
the IDs in this file must be present in all the other input files.

.. code-block:: text
    :caption: Maize sample IDs

    11116.C02A66.1194587
    11116.C06A63.1195666
    11116.C08A61.1197689
    11116.C08A63.1196825
    11116.C12A64.1197667

Finally, we need to create a data config YAML file. The purpose of this file is
to tell MOVE which files to load, where to find them, and where to save any
output files.

The data config file for this tutorial would look like this:

.. literalinclude:: /../../tutorial/config/data/maize.yaml
    :language: yaml

Here we break down the fields of this file:

* ``defaults`` indicates this file is a config file. It should be left intact.
* ``raw_data_path`` points to the raw data location (i.e., the files we
  created in this tutorial).
* ``interim_data_path`` points to the directory where intermediary files will
  be deposited.
* ``results_path`` points to the folder where results will be saved.
* ``sample_names`` is the file name of the file containing all valid sample
  IDs. This file must have a ``txt`` extension.
* ``categorical_inputs`` is a list of file names containing categorical data.
  Each element of the list should have a name ``name`` and may optionally have
  a ``weight``. All referenced files should have a ``tsv`` extension.
* ``continuous_inputs`` lists the continuous data files. Same format as
  ``categorical_inputs``.

The data config file can have any name, but it must be saved in ``config/data``
directory. The final workspace structure should look like this:::

    tutorial/
    │
    ├── maize/
    │   └── data/
    │       ├── maize_field.tsv       <- Type of soil data
    │       ├── maize_ids.txt         <- Sample IDs
    │       ├── maize_line.tsv        <- Maize line data
    │       ├── maize_metadata.tsv    <- Age, temperature, precipitation data
    │       ├── maize_microbiome.tsv  <- OTU table
    │       └── maize_variety.tsv     <- Maize variety data
    │
    └── config/
        └── data/
            └── maize.yaml            <- Data configuration file

With your data formatted and ready, we can continue to run MOVE and exploring
the associations between the different variables in your datasets. Have a look
at our :doc:`introductory tutorial</tutorial/introduction>` for more
information on this.

References
----------

.. [#] Walters WA, Jin Z, Youngblut N, Wallace JG, Sutter J, Zhang W, et al.
  Large-scale replicated field study of maize rhizosphere identifies heritable
  microbes. `Proc Natl Acad Sci U S A`. 2018; 115: 7368–7373.
  `doi:10.1073/pnas.1800918115`_

.. _`doi:10.1073/pnas.1800918115`: https://doi.org/10.1073/pnas.1800918115

.. _`OTU table`: https://github.com/jorgemf/DeepLatentMicrobiome/raw/91e384b7115978bb3cd0f61c7dd3d8ffc866efc3/Datasets/otu_table_all_80.csv
.. _`metadata`: https://github.com/jorgemf/DeepLatentMicrobiome/raw/91e384b7115978bb3cd0f61c7dd3d8ffc866efc3/Datasets/metadata_table_all_80.csv
