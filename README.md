# MOVE (Multi-Omics Variational autoEncoder)

The code in this repository can be used to run our Multi-Omics Variational autoEncoder (MOVE) framework for integration of omics and clinical variabels spanning both categorial and continuous data. Our approach includes training ensemble VAE models and using in-silico perturbation experiments to identify cross omics associations. The manuscript has been Accepted and we will link when it is published.

We developed the method based on a Type 2 Diabetes cohort from the IMI DIRECT project containing 789 newly diagnosed T2D patients. The cohort and data creation is described in [Koivula et al.](https://dx.doi.org/10.1007%2Fs00125-019-4906-1) and [Wesolowska-Andersen and Brorsson et al.](https://doi.org/10.1016/j.xcrm.2021.100477). For the analysis we included the following data:

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

MOVE is written in python and can therefore be installed using:

```
pip install move-dl
```

## Requirements

MOVE runs on Mac, Windows and Linux using python. The variational autoencoder framework is implemented in pytorch, but everything should be installed for you using pip. The only exception to that is that if you want to use the jupyter notebooks you have to install jupyter yourself.

The training of the VAEs can be done using CPUs only or GPU acceleration. If you dont have powerful GPUs available it is perfectly fine to run using only CPUs. For instance, the tutorial data set consisting of simulated drug, metabolomics and proteomics data for 500 individuals runs fine on a standard macbook.

# The MOVE pipeline

MOVE has five-six steps:

```
01. Encode the data into a format that can be read by MOVE
02. Finding the right architecture of the network focusing on reconstruction accuracy
03. Finding the right architecture of the network focusing on stability of the model
04. Use model, determined from steps 02-03, to create and analyze the latent space
05. Identify associations between a categorical and continuous datasets
05a. Using an ensemble of VAEs with the T-test approach
05b. Using an ensemble of VAEs with the Bayesian decision theory approach
06. If both 5a and 5b were run select the overlap between them
```

## How to run MOVE

You can run the move-dl pipeline using the command line or using Jupyter notebooks. Notebooks with explanations are in the [tutorial](https://github.com/RasmussenLab/MOVE/tree/developer/tutorial) folder. Feel free to open an issue for help.

You can run MOVE as Python module with the following commands: 
```
python -m move.01_encode_data 
python -m move.02_optimize_reconstruction
python -m move.03_optimize_stability
python -m move.04_analyze_latent
python -m move.05_identify_associations
```


## How to use MOVE with your data

Your data files should be tab separated, include a header and the first column should be the IDs of your samples. The configuration of MOVE is done using yaml files that describe the input data (data.yaml), the model (model.yaml) and files associated with each of the steps (tuning_reconstruction.yaml, tuning_stability.yaml, training_latent.yaml, training_association.yaml). These should be placed in the working directory. Please see the [tutorial](https://github.com/RasmussenLab/MOVE/tree/developer/tutorial) for more information.


# Data sets

## DIRECT data set
The data used in notebooks are not available for testing due to the informed consent given by study participants, the various national ethical approvals for the study, and the European General Data Protection Regulation (GDPR). Therefore, individual-level clinical and omics data cannot be transferred from the centralized IMI-DIRECT repository. Requests for access to summary statistics IMI-DIRECT data, including those presented here, can be made to DIRECTdataaccess@Dundee.ac.uk. Requesters will be informed on how summary-level data can be accessed via the DIRECT secure analysis platform following submission of appropriate application. The IMI-DIRECT data access policy is available at [here](https://directdiabetes.org).

## Simulated and publicaly available data set
We have therefore added a simulated data set that can be used for testing the workflow and a publicly available maize rhizosphere microbiome data set. We have also included a notebook that goes through a short [tutorial](https://github.com/RasmussenLab/MOVE/tree/developer/tutorial) with a publicly-available maize rhizosphere microbiome dataset.

