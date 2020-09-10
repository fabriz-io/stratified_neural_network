# %% Load modules.
import pandas as pd
import os

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(FILE_DIR, "data")

# Set how many genes to include.
NO_OF_FEATURES = 3000

# %% Load Data.
print("Loading data...")
full_data = pd.read_csv(os.path.join(DATA_DIR, "full_data_preprocessed.csv"))
print("Done.")

# %% Get patients with specified tumor types
print("Start preprocessing...")

# BRCA.
brca = (full_data.tumor_type == 'BRCA')

# GLIOMA
glioma = ((full_data.tumor_type == 'LGG') |
          (full_data.tumor_type == 'GBM'))

full_data.loc[glioma, 'tumor_type'] = 'GLIOMA'
glioma = (full_data.tumor_type == 'GLIOMA')

# KIPAN
kipan = ((full_data.tumor_type == 'KIRP') |
         (full_data.tumor_type == 'KICH') |
         (full_data.tumor_type == 'KIRC'))

full_data.loc[kipan, 'tumor_type'] = 'KIPAN'
kipan = (full_data.tumor_type == 'KIPAN')

# brca-kipan-glioma data.
brca_kipan_glioma_mask = (brca | glioma | kipan)
brca_kipan_glioma_data = full_data.loc[brca_kipan_glioma_mask, :]

# brca-kipan data.
brca_kipan_mask = (brca | kipan)
brca_kipan_data = full_data.loc[brca_kipan_mask, :]

# glioma data.
glioma_data = full_data.loc[glioma, :]

# Adjust data to be saved.
data = brca_kipan_glioma_data
data_name = "brca_kipan_glioma"

# %% Get Genes with highest Variance.

def get_genes_highest_variance(data, NO_OF_FEATURES):
    gene_counts = data.loc[:, '1/2-SBSRNA4':]
    variances = gene_counts.var()
    variances.sort_values(ascending=False, inplace=True)
    gene_names = list(variances.index)
    genes_to_keep = gene_names[:NO_OF_FEATURES]
    genes_to_keep.sort()
    return genes_to_keep


# Choose columns to keep.
genes_to_keep = get_genes_highest_variance(data, NO_OF_FEATURES)
final_columns = ["patient_id", "tumor_type", "time", "event"] + genes_to_keep

# Cast types.
data = data.astype({'time': 'int32', 'event': 'bool'})
data.reset_index(drop=True, inplace=True)
data = data.loc[:, final_columns]

# Normalize gene counts.
data.loc[:, genes_to_keep[0]:] = (data.loc[:, genes_to_keep[0]:] - data.loc[:, genes_to_keep[0]:].mean()) / data.loc[:, genes_to_keep[0]:].std()

print("Done.")
print("Saving data...")
data.to_csv(os.path.join(DATA_DIR, "{}_normalized_{}_features.csv".format(data_name, NO_OF_FEATURES)))
print("Done.")

