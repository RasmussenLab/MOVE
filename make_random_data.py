import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy
import pandas as pd
import seaborn as sns
import re
import random
from collections import defaultdict

def write_cat(f_name, size, minimum, maximum, name):
    data = np.random.randint(minimum, maximum + 1, size = size)
    with open(f_name, 'w') as o:
        o.write("ID\t" + "\t".join([name + str(j + 1) for j in range(data.shape[1])])+ "\n")
        for i in range(data.shape[0]):
            o.write(str(i) + "\t" + "\t".join([str(j) for j in data[i,:]]) + "\n")
    
def write_con(f_name, size, m, s, zero, name):
    data = m + s * np.random.randn(size[0], size[1])
    indices = np.random.choice(data.shape[1]*data.shape[0], replace=False, size=int(data.shape[1]*data.shape[0]*zero))
    data[np.unravel_index(indices, data.shape)] = 0
    
    with open(f_name, 'w') as o:
        o.write("ID\t" + "\t".join([name + str(j + 1) for j in range(data.shape[1])]) + "\n")
        for i in range(data.shape[0]):
            o.write(str(i) + "\t" + "\t".join([str(round(np.abs(j))) for j in data[i,:]]) + "\n")


with open("data_for_git/baseline_ids.txt", "w") as f:
    f.write("\n".join([str(i) for i in range(789)]))

cat_f =  "data_for_git/baseline_categorical.tsv"
write_cat(cat_f, (789,42), 0, 4, "clinical_categorical_")

geno_f =  "data_for_git/diabetes_genotypes.tsv"
write_cat(geno_f, (789,393), 0, 2, "SNP_")

drug_f =  "data_for_git/baseline_drugs.tsv"
write_cat(drug_f, (789,20), 0, 1, "drug_")

# Generate continous data
clin_f = "data_for_git/baseline_continuous.tsv"
write_con(clin_f, [789, 76], 10, 3, 0.1, "clinical_continuous_")

diet_wearables_f = "data_for_git/baseline_diet_wearables.tsv"
write_con(diet_wearables_f, [789, 74], 100, 50, 0.2, "diet_wearables_")

pro_f = "data_for_git/baseline_proteomic_antibodies.tsv"
write_con(pro_f, [789, 373], 15, 7, 0.25, "protein_")

target_mata_f = "data_for_git/baseline_target_metabolomics.tsv"
write_con(target_mata_f, [789, 119], 10, 5, 0.2, "targeted_metabolomics_")

untarget_mata_f = "data_for_git/baseline_untarget_metabolomics.tsv"
write_con(untarget_mata_f, [789, 238], 10, 5, 0.3, "untargeted_metabolomics_")

trans_f = "data_for_git/baseline_transcriptomics.tsv"
write_con(trans_f, [789, 6018], 20, 15, 0.35, "transcriptomics_")

metagen_f = "data_for_git/baseline_metagenomics.tsv"
write_con(metagen_f, [789, 1463], 15, 10, 0.6, "metagenomics_")
