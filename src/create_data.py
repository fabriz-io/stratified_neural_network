
# %% Import Modules
import pandas as pd
import numpy as np
import requests
import os
import warnings
warnings.filterwarnings("ignore")


#######################################################
# Specify how many genes to include in final data frame.
NO_OF_FEATURES = 3000
#######################################################

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(FILE_DIR, "data")
RSUBREAD_FOLDER = os.path.join(FILE_DIR, "data", "rsubread")

if not os.path.exists(RSUBREAD_FOLDER):
    os.makedirs(RSUBREAD_FOLDER)

print("____ Downloading data ____ \n")
# %% Download and open clinical variables.
clinical_variables_url = r"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE62nnn/GSE62944/suppl/GSE62944%5F06%5F01%5F15%5FTCGA%5F24%5F548%5FClinical%5FVariables%5F9264%5FSamples%2Etxt%2Egz"

clinical_variables_path = os.path.join(RSUBREAD_FOLDER, "clinical_variables.txt.gz")

print("Started Download of Clinical Variables...")
r = requests.get(clinical_variables_url)
with open(clinical_variables_path, "wb") as f:
    f.write(r.content)

clinical_variables = pd.read_csv(clinical_variables_path, sep="\t", compression="gzip")

print("Done.")

# %% Dowload and open cancer types.
cancer_type_url = r"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE62nnn/GSE62944/suppl/GSE62944%5F06%5F01%5F15%5FTCGA%5F24%5FCancerType%5FSamples%2Etxt%2Egz"
cancer_type_path = os.path.join(RSUBREAD_FOLDER, "cancer_types.txt.gz")

print("Started Download of Cancer Types...")
r = requests.get(cancer_type_url)
with open(cancer_type_path, 'wb') as f:
    f.write(r.content)

cancer_types = pd.read_csv(cancer_type_path, sep="\t", header=0, names=["patient_id", "tumor_type"], compression="gzip")

print("Done.")

# %% Download and open rsubread gene counts.
rsubread_gene_counts_url = r"https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1536nnn/GSM1536837/suppl/GSM1536837%5F06%5F01%5F15%5FTCGA%5F24%2Etumor%5FRsubread%5FFeatureCounts%2Etxt%2Egz"
rsubread_gene_counts_path = os.path.join(RSUBREAD_FOLDER, "gene_counts.txt.gz")

print("Started Download of Gene Counts...")

r = requests.get(rsubread_gene_counts_url)
with open(rsubread_gene_counts_path, "wb") as f:
    f.write(r.content)

gene_counts = pd.read_csv(rsubread_gene_counts_path, sep="\t", compression="gzip")

print("Done.")


print("\n ___ Merging Data ___\n")
# %% Clinical Variables

print("Selecting variables of interest...")

clinical_variables = clinical_variables.drop(columns=["Unnamed: 1", "Unnamed: 2"])
clinical_variables.set_index('Unnamed: 0', inplace=True)
clinical_variables = clinical_variables.loc[["vital_status", "last_contact_days_to", "death_days_to"], :]
clinical_variables = clinical_variables.T
clinical_variables = clinical_variables.dropna(subset=["vital_status"])
clinical_variables = clinical_variables.dropna(subset=["last_contact_days_to", "death_days_to"])
clinical_variables = clinical_variables.loc[clinical_variables.vital_status != "[Not Available]", :]

clinical_variables["time"] = -1
mask = clinical_variables.vital_status == "Dead"
clinical_variables.time.loc[mask] = clinical_variables.death_days_to.loc[mask]

mask = clinical_variables.vital_status == "Alive"
clinical_variables.time.loc[mask] = clinical_variables.last_contact_days_to.loc[mask]

# Drop all not usable data points.
mask = (clinical_variables.time != "[Not Available]") & \
       (clinical_variables.time != "[Discrepancy]") & \
       (clinical_variables.time != "[Completed]")
clinical_variables = clinical_variables.loc[mask]

# Drop non-positive survival times.
clinical_variables.time = pd.to_numeric(clinical_variables.time)
clinical_variables = clinical_variables.loc[clinical_variables.time > 0]

# Set event indicator. Person died := event == True
clinical_variables["event"] = -1
clinical_variables.event[clinical_variables.vital_status == "Dead"] = True
clinical_variables.event[clinical_variables.vital_status == "Alive"] = False

clinical_variables = clinical_variables.loc[:, ["time", "event"]]
clinical_variables.reset_index(inplace=True)
clinical_variables.rename(columns={"index": "patient_id"}, inplace=True)

# %% Merge with cancer types.
patients = pd.merge(cancer_types, clinical_variables, on=["patient_id"])

# %% Merge with gene Counts.

gene_counts.set_index("Unnamed: 0", inplace=True)
gene_counts = gene_counts.T
gene_counts.reset_index(inplace=True)
gene_counts.rename(columns={"index": "patient_id"}, inplace=True)

# %% Data frame with all possible tumor types.
full_data = pd.merge(patients, gene_counts, on=["patient_id"])
print("Done.")

# %% Selecting Cancer types of interest.

print("Merging data of specified cancer types...")

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