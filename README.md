# MOVE

The code in this repository can be used to run our Multi-Omics Variational autoEncoder (MOVE) framework for integration of omics and clinical variabels spanning both categorial and continuous data. The framework includes four notebooks including hyperparameter seletion based on two criteria, analysing the MOVE latent space integration and lastly using the trained VAE for extracting drug assosiations. The data used in the notebooks are not available so for usage on own data an initial guide for encoding can be found in the encode_data.ipynb notebook.

The notebooks should be used as follows:
1. Encode data and data formatting: encode_data.ipynb
2. Hyperparameter optimization part 1: MOVE_hyperparameter_optimization_reconstruction.ipynb
3. Hyperparameter optimization part 2: MOVE_hyperparameter_optimization_stability.ipynb
4. Create and analyse the MOVE latent space: latent_space_analysis.ipynb
5. Extract drug/categorical data assositions across all continuous datasets: identify_drug_assosiation.ipynb

The data used in notebooks are not availble for testing due to the informed consent given by study participants, the various national ethical approvals for the present study, and the European General Data Protection Regulation (GDPR). Therefore, individual-level clinical and omics data cannot be transferred from the centralized IMI-DIRECT repository. Requests for access to summary statistics IMI-DIRECT data, including those presented here, can be made to DIRECTdataaccess@Dundee.ac.uk. Requesters will be informed on how summary-level data can be accessed via the DIRECT secure analysis platform following submission of appropriate application. The IMI-DIRECT data access policy is available at [here](www.direct-diabetes.org).
