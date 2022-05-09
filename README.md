# MOVE (Multi-Omics Variational autoEncoder)

The code in this repository can be used to run our Multi-Omics Variational autoEncoder (MOVE) framework for integration of omics and clinical variabels spanning both categorial and continuous data. The manuscript is currently in review.

The framework includes five notebooks including:
1. Encode data and data formatting: encode_data.ipynb
2. Hyperparameter optimization part 1: MOVE_hyperparameter_optimization_reconstruction.ipynb
3. Hyperparameter optimization part 2: MOVE_hyperparameter_optimization_stability.ipynb
4. Create and analyse the MOVE latent space: latent_space_analysis.ipynb
5. Extract drug/categorical data assositions across all continuous datasets: identify_drug_assosiation.ipynb

We developed the method based on a Type 2 Diabetes cohort from the IMI DIRECT project containing 789 newly diagnosed T2D patients. The cohort and data creation is described in [Koivula et al.](https://dx.doi.org/10.1007%2Fs00125-019-4906-1) and [Wesolowska-Andersen and Brorsson et al.](https://doi.org/10.1016/j.xcrm.2021.100477). For the analysis we included the following data:

Omics:
```
Genomics
Transcriptomics
Proteomics
Metabolomics
Metagenomics
```

Other data:
```
Clinical data (blood measurements, imaging data, ...)
Questionnaire data (diet etc)
Accelerometer data
Medication data
```

The data used in our project is sensitive data and therefore not available for download - see below for the procedure to obtain access.

## Other datasets

We added simulated data that can be used for testing the workflow - note that these data are at the moment random data.

We have also included a notebook that goes through a short tutorial with a publicly-available maize rhizosphere microbiome dataset.

# Installation
MOVE is currently in the form of Jupyter Notebooks to ensure that one can modify the scripts according to the data available. Ie, we had five types of omics data together with clinical data - your dataset may differ. We are working on a script and pip package for easier access to the analysis.

### Requirements
We have only run MOVE in a Linux environment, however as it is depedent on python it should be possible to run on Macs and potentially Windows machines as well. We ran this with python3.8 from Anaconda 5.3.0 using the following packages:

```
jupyter
numpy
pytorch
sklearn
scipy
pandas
seaborn
matplotlib
```

For training on GPU:
```
pytorch with CUDA
CUDA drivers
```

All above packages can be installed using pip or conda, e.g. `pip install jupyterlab` or by installing [Anaconda](https://anaconda.org) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html). If you install Anaconda most packages should be installed by default or you can install using the conda approach `conda install packagename`. For installing pytorch go to the [website](https://pytorch.org/get-started/locally/) and chose the version that matches you OS and whether you want CUDA support or not.

The training of the VAEs can be done using GPU or CPU, as with all deep learning GPUs speed up the computations but it is perfectly doable to run on a server with multiple CPU cores. 

### How to run MOVE
Start with the Jupyter notebooks above from 01-05. Explanations are in the notebooks. Feel free to open an issue for help. **We are working on a script and pip package for easier access to the analysis, ETA April 2022.**

# DIRECT data access
The data used in notebooks are not available for testing due to the informed consent given by study participants, the various national ethical approvals for the study, and the European General Data Protection Regulation (GDPR). Therefore, individual-level clinical and omics data cannot be transferred from the centralized IMI-DIRECT repository. 

Requests for access to summary statistics IMI-DIRECT data, including those presented here, can be made to DIRECTdataaccess@Dundee.ac.uk. Requesters will be informed on how summary-level data can be accessed via the DIRECT secure analysis platform following submission of appropriate application. 

The IMI-DIRECT data access policy is available at [here](https://directdiabetes.org).
