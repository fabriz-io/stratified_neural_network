
# %% Import Modules
import pandas as pd
import numpy as np
import requests
import os
import warnings
warnings.filterwarnings("ignore")

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
RSUBREAD_FOLDER = os.path.join(FILE_DIR, "data", "rsubread")

if not os.path.exists(RSUBREAD_FOLDER):
    os.makedirs(RSUBREAD_FOLDER)

print("Starting Download of data.")
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


print("Started merging of Data...")
# %% Clinical Variables

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

# %% Save full data to file.
full_data = pd.merge(patients, gene_counts, on=["patient_id"])

print("Done.")
print("Saving to file...")
full_data.to_csv(os.path.join(FILE_DIR, "data", "full_data_preprocessed.csv"))
print("Done.")
