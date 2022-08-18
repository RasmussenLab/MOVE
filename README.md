# MOVE (Multi-Omics Variational autoEncoder)

The code in this repository can be used to run our Multi-Omics Variational autoEncoder (MOVE) framework for integration of omics and clinical variabels spanning both categorial and continuous data. Our approach includes training ensemble VAE models and using adversarial attacks (ie. permutations) to identify cross omics associations. The manuscript is currently in review.

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
MOVE can be run both as a Python script and from the Jupyter notebook to ensure that one can run a pipeline conveniently and modify the scripts according to the available data. For instance, we had five types of omics data together with clinical data - your dataset may differ. 

### Installing MOVE package

To install the MOVE package, use: 
```
pip install move-dl
```

### Requirements for Jupyter notebook
To run scripts in the Jupyter notebook, install: 
```
move-dl
jupyter
```

We have only run MOVE in a Linux environment. However, as it depends on Python, it should be possible to run on Mac and Windows machines. We tested on an Anaconda (v5.3.0) environment with Python 3.8.

The training of the VAEs can be done using GPU or CPU. However, as with all deep learning, GPUs speed up the computations, but running on a server with multiple CPU cores is perfectly doable.


### How to run MOVE
You can run the move-dl pipeline on Jupyter notebooks from 01-05 located in tutorial/ folder. Explanations are in the notebooks. Feel free to open an issue for help.

You can run MOVE as Python scripts with the following commands: 
```
python -m move.01_encode_data 
python -m move.02_optimize_reconstruction
python -m move.03_optimize_stability
python -m move.04_analyze_latent
python -m move.05_identify_drug_assosiation
```

Python scripts have the same functions as notebooks. 
To override the default MOVE configurations, user-defined configurations should be defined in the working directory in the .yaml files (data.yaml, model.yaml, tuning_reconstruction.yaml, tuning_stability.yaml, training_latent.yaml, training_association.yaml).

# DIRECT data access
The data used in notebooks are not available for testing due to the informed consent given by study participants, the various national ethical approvals for the study, and the European General Data Protection Regulation (GDPR). Therefore, individual-level clinical and omics data cannot be transferred from the centralized IMI-DIRECT repository. 

Requests for access to summary statistics IMI-DIRECT data, including those presented here, can be made to DIRECTdataaccess@Dundee.ac.uk. Requesters will be informed on how summary-level data can be accessed via the DIRECT secure analysis platform following submission of appropriate application. 

The IMI-DIRECT data access policy is available at [here](https://directdiabetes.org).
