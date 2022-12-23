# MOVE (Multi-Omics Variational autoEncoder)

The code in this repository can be used to run our Multi-Omics Variational
autoEncoder (MOVE) framework for integration of omics and clinical variabels
spanning both categorial and continuous data. Our approach includes training
ensemble VAE models and using *in silico* perturbation experiments to identify
cross omics associations. The manuscript has been accepted and we will provide 
the link when it is published.

We developed the method based on a Type 2 Diabetes cohort from the IMI DIRECT
project containing 789 newly diagnosed T2D patients. The cohort and data
creation is described in
[Koivula et al.](https://dx.doi.org/10.1007%2Fs00125-019-4906-1) and
[Wesolowska-Andersen et al.](https://doi.org/10.1016/j.xcrm.2021.100477). For
the analysis we included the following data:

Multi-omics data sets:
```
Genomics
Transcriptomics
Proteomics
Metabolomics
Metagenomics
```

Other data sets:
```
Clinical data (blood measurements, imaging data, ...)
Questionnaire data (diet etc)
Accelerometer data
Medication data
```

# Installation

## Installing MOVE package

MOVE is written in Python and can therefore be installed using `pip`:

```bash
>>> pip install move-dl
```

## Requirements

MOVE should run on any environmnet where Python is available. The variational
autoencoder architecture is implemented in PyTorch.

The training of the VAEs can be done using CPUs only or GPU acceleration. If
you do not have powerful GPUs available, it is possible to run using only CPUs.
For instance, the tutorial data set consisting of simulated drug, metabolomics
and proteomics data for 500 individuals runs fine on a standard macbook.

> Note: The pip installation of `move-dl` does not setup your local GPU automatically

# The MOVE pipeline

MOVE has five-six steps:

```
01. Encode the data into a format that can be read by MOVE
02. Finding the right architecture of the network focusing on reconstruction accuracy
03. Finding the right architecture of the network focusing on stability of the model
04. Use model, determined from steps 02-03, to create and analyze the latent space
05. Identify associations between a categorical and continuous datasets
05a. Using an ensemble of VAEs with the t-test approach
05b. Using an ensemble of VAEs with the Bayesian decision theory approach
06. If both 5a and 5b were run select the overlap between them
```

## How to run MOVE

You can run the move-dl pipeline from the command line or within a Jupyter
notebook.

You can run MOVE as Python module with the following command. Details on how
to set up the configuration for the data and task can be found our
[tutorial](https://github.com/RasmussenLab/MOVE/tree/main/tutorial) folder.

```bash
>>> move-dl data=[name of data config] task=[name of task config]
```

Feel free to
[open an issue](https://github.com/RasmussenLab/MOVE/issues/new/choose) if you
need any help.

### How to use MOVE with your data

Your data files should be tab separated, include a header and the first column
should be the IDs of your samples. The configuration of MOVE is done using YAML
files that describe the input data and the task specification. These should be
placed in a `config` directory in the working directory. Please see the
[tutorial](https://github.com/RasmussenLab/MOVE/tree/main/tutorial)
for more information.


# Data sets

## DIRECT data set

The data used in notebooks are not available for testing due to the informed
consent given by study participants, the various national ethical approvals for
the study, and the European General Data Protection Regulation (GDPR).
Therefore, individual-level clinical and omics data cannot be transferred from
the centralized IMI-DIRECT repository. Requests for access to summary statistics
IMI-DIRECT data, including those presented here, can be made to
DIRECTdataaccess@Dundee.ac.uk. Requesters will be informed on how summary-level
data can be accessed via the DIRECT secure analysis platform following
submission of appropriate application. The IMI-DIRECT data access policy is
available [here](https://directdiabetes.org).

## Simulated and publicaly available data sets

We have therefore provided two datasets to test the workflow: a simulated 
dataset and a publicly-available maize rhizosphere microbiome data set.
