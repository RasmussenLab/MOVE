# MOVE

The code in this repository can be used to run our Multi-Omics Variational autoEncoder (MOVE) framework for integration of omics and clinical variabels spanning both categorial and continuous data. The framework includes four notebooks including hyperparameter seletion based on two criteria, analysing and comparing the MOVE latent space to other methods and lastly using the trained VAE for extracting drug assosiations. The data used in the notebooks are not available so for usages on own data an initial guide for encoding can be found in the encode_data.ipynb notebook.

The notebooks should be used as follows:
1. Encode data and data formatting: encode_data.ipynb
2. Hyperparamter optimization part 1: MOVE_hyperparameter_optimization_reconstruction.ipynb
3. Hyperparamter optimization part 2: MOVE_hyperparameter_optimization_stability.ipynb
4. Create and analyse the MOVE latent sapce: latent_space_analysis.ipynb
5. Extract drug/categorical data assositions across all continuous datasets: identify_drug_assosiation.ipynb
