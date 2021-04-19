# %% Import Modules
import pandas as pd
import numpy as np
import requests
import os
import warnings
import sys

warnings.filterwarnings("ignore")


#######################################################################
# Specify amount of most important genes per tumor type to be included.
NO_OF_FEATURES = 2000
#######################################################################

# %% Check if tumor types specified exist in the data.

TUMOR_TYPES = [
    "BLCA",
    "BRCA",
    "CESC",
    "COAD",
    "DLBC",
    "GBM",
    "HNSC",
    "KICH",
    "KIRC",
    "KIRP",
    "LAML",
    "LGG",
    "LIHC",
    "LUAD",
    "LUSC",
    "OV",
    "PRAD",
    "READ",
    "SKCM",
    "STAD",
    "THCA",
    "UCEC",
]

TUMOR_TYPE_COMBINATION = sorted([x for x in sys.argv[1:]])

if "log" in TUMOR_TYPE_COMBINATION:
    log_scaling = True
    TUMOR_TYPE_COMBINATION.remove("log")
else:
    log_scaling = False

if not set(TUMOR_TYPE_COMBINATION).issubset(TUMOR_TYPES):
    raise ValueError(
        "Your Combination of Tumor Types seems not to exist: {}".format(TUMOR_TYPES)
    )

# %% Specify relative paths.

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(FILE_DIR, "data")
RSUBREAD_FOLDER = os.path.join(FILE_DIR, "data", "rsubread")

if not os.path.exists(RSUBREAD_FOLDER):
    os.makedirs(RSUBREAD_FOLDER)


# Specify paths.
clinical_variables_path = os.path.join(RSUBREAD_FOLDER, "clinical_variables.txt.gz")
cancer_type_path = os.path.join(RSUBREAD_FOLDER, "cancer_types.txt.gz")
rsubread_gene_counts_path = os.path.join(RSUBREAD_FOLDER, "gene_counts.txt.gz")

# %% Download data.
print("____ Downloading data ____ \n")

# Clinical Variables

if not os.path.exists(clinical_variables_path):
    print("Started Download of Clinical Variables...")
    clinical_variables_url = r"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE62nnn/GSE62944/suppl/GSE62944%5F06%5F01%5F15%5FTCGA%5F24%5F548%5FClinical%5FVariables%5F9264%5FSamples%2Etxt%2Egz"

    r = requests.get(clinical_variables_url)
    with open(clinical_variables_path, "wb") as f:
        f.write(r.content)

    print("Done.")
else:
    print("Raw data exists. Skipping Download.")

# Cancer types.
if not os.path.exists(cancer_type_path):
    print("Started Download of Cancer Types...")
    cancer_type_url = r"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE62nnn/GSE62944/suppl/GSE62944%5F06%5F01%5F15%5FTCGA%5F24%5FCancerType%5FSamples%2Etxt%2Egz"

    r = requests.get(cancer_type_url)
    with open(cancer_type_path, "wb") as f:
        f.write(r.content)

    print("Done.")

# Gene counts.
if not os.path.exists(rsubread_gene_counts_path):
    print("Started Download of Gene Counts...")
    rsubread_gene_counts_url = r"https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1536nnn/GSM1536837/suppl/GSM1536837%5F06%5F01%5F15%5FTCGA%5F24%2Etumor%5FRsubread%5FFeatureCounts%2Etxt%2Egz"

    r = requests.get(rsubread_gene_counts_url)
    with open(rsubread_gene_counts_path, "wb") as f:
        f.write(r.content)

    print("Done.")


# %% Open downloaded data.

print("Opening downloaded data...")

clinical_variables = pd.read_csv(clinical_variables_path, sep="\t", compression="gzip")
cancer_types = pd.read_csv(
    cancer_type_path,
    sep="\t",
    header=0,
    names=["patient_id", "tumor_type"],
    compression="gzip",
)
gene_counts = pd.read_csv(rsubread_gene_counts_path, sep="\t", compression="gzip")

print("Done.")


# %% Clinical Variables

print("\n ___ Merging Data ___\n")

try:
    print("Merged data exists, nothing to merge.")
    print("Loading merged data...")
    full_data = pd.read_csv(os.path.join(RSUBREAD_FOLDER, "complete_data_merged.csv"))
except:
    print("Selecting variables of interest...")

    clinical_variables = clinical_variables.drop(columns=["Unnamed: 1", "Unnamed: 2"])
    clinical_variables.set_index("Unnamed: 0", inplace=True)
    clinical_variables = clinical_variables.loc[
        ["vital_status", "last_contact_days_to", "death_days_to"], :
    ]
    clinical_variables = clinical_variables.T
    clinical_variables = clinical_variables.dropna(subset=["vital_status"])
    clinical_variables = clinical_variables.dropna(
        subset=["last_contact_days_to", "death_days_to"]
    )
    clinical_variables = clinical_variables.loc[
        clinical_variables.vital_status != "[Not Available]", :
    ]

    clinical_variables["time"] = -1
    mask = clinical_variables.vital_status == "Dead"
    clinical_variables.time.loc[mask] = clinical_variables.death_days_to.loc[mask]

    mask = clinical_variables.vital_status == "Alive"
    clinical_variables.time.loc[mask] = clinical_variables.last_contact_days_to.loc[
        mask
    ]

    # Drop all not usable data points.
    mask = (
        (clinical_variables.time != "[Not Available]")
        & (clinical_variables.time != "[Discrepancy]")
        & (clinical_variables.time != "[Completed]")
    )
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

    print("Done.")
    # Merge with cancer types.
    print("Merging with cancer types.")

    patients = pd.merge(cancer_types, clinical_variables, on=["patient_id"])

    print("Done.")

    # Merge with gene Counts.
    print("Merging with gene counts.")

    gene_counts.set_index("Unnamed: 0", inplace=True)
    gene_counts = gene_counts.T
    gene_counts.reset_index(inplace=True)
    gene_counts.rename(columns={"index": "patient_id"}, inplace=True)

    print("Done.")
    # Data frame with all possible tumor types.
    print("Merging all together.")

    full_data = pd.merge(patients, gene_counts, on=["patient_id"])
    print("Done.")

    print("Saving merged data...")
    full_data.to_csv(os.path.join(RSUBREAD_FOLDER, "complete_data_merged.csv"))

print("Done.")


# %% Get Genes with highest Variance for each tumor type.


def get_genes_highest_variance(data, num_of_features):
    gene_counts = data.loc[:, "1/2-SBSRNA4":]
    variances = gene_counts.var()
    variances.sort_values(ascending=False, inplace=True)
    gene_names = list(variances.index)
    genes_to_keep = gene_names[:num_of_features]
    genes_to_keep.sort()
    return genes_to_keep


def merge_genes_of_highest_variance(
    data, list_of_tumor_types, num_of_features, log_transform=True, standardization=True
):
    data = data.loc[full_data.tumor_type.isin(list_of_tumor_types), :]

    gene_names = set()

    # Get the highest variance genes for each tumor type.
    for tumor_type in list_of_tumor_types:
        gene_names.update(
            get_genes_highest_variance(
                data.loc[(data.tumor_type == tumor_type), :],
                num_of_features=num_of_features,
            )
        )

    # Final gene names list
    gene_names = sorted(list(gene_names))

    # gene count data to be transformed
    gene_count_data = data.loc[:, gene_names]

    # Log Transformation
    if log_transform:
        gene_count_data = gene_count_data.transform(lambda x: np.log(x + 1))

    # Standardization
    if standardization:
        gene_count_data = (
            gene_count_data - gene_count_data.mean()
        ) / gene_count_data.std()

    # Generate final dataframe
    columns = ["patient_id", "tumor_type", "time", "event"] + gene_names
    data = data.loc[:, columns]
    data.loc[:, gene_names] = gene_count_data
    data = data.astype({"time": "int", "event": "bool"})  # Cast types.
    data.reset_index(drop=True, inplace=True)

    return data


# %% Save data.

print("Saving data...")

data = merge_genes_of_highest_variance(
    full_data,
    TUMOR_TYPE_COMBINATION,
    num_of_features=NO_OF_FEATURES,
    log_transform=log_scaling,
)

if log_scaling:
    save_path = os.path.join(
        DATA_DIR,
        "{}_log_scaled.csv".format("_".join(TUMOR_TYPE_COMBINATION)),
    )
else:
    save_path = os.path.join(
        DATA_DIR,
        "{}_scaled.csv".format("_".join(TUMOR_TYPE_COMBINATION)),
    )


data.to_csv(save_path)

print("Done.")
