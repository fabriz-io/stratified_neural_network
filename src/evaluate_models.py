# %% Import Modules
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sksurv.metrics import brier_score
from lifelines import KaplanMeierFitter
import shap
import seaborn as sns

FILE_DIR = os.path.dirname(os.path.abspath(__file__))


# %% Load Data.

print("Loading Data...")

data_path = "data/brca_kipan_glioma_normalized_3000_features.csv"
data = pd.read_csv(data_path)
data.index = data.patient_id

gene_counts = data.iloc[:, 5:]
gene_counts_dim = gene_counts.shape[1]
gene_names = gene_counts.columns.to_numpy()
np.savetxt("summary/gene_names_3000.txt", gene_names, fmt="%s")
event_indicator = data.event.to_numpy(dtype=np.bool)
event_time = data.time.to_numpy(dtype=np.int16)
strata = data.tumor_type.to_numpy(dtype=np.str)

pec_summary_pl = pd.read_csv(FILE_DIR + "/summary/pec_pl.csv", index_col=0)
pec_summary_spl = pd.read_csv(FILE_DIR + "/summary/pec_spl.csv", index_col=0)
pec_summary_rl = pd.read_csv(FILE_DIR + "/summary/pec_rl.csv", index_col=0)
pec_summary_srl = pd.read_csv(FILE_DIR + "/summary/pec_srl.csv", index_col=0)

pec_summary_pl.columns = pec_summary_pl.columns.astype("int")
pec_summary_spl.columns = pec_summary_spl.columns.astype("int")
pec_summary_rl.columns = pec_summary_rl.columns.astype("int")
pec_summary_srl.columns = pec_summary_srl.columns.astype("int")

c_summary = pd.read_csv(FILE_DIR + "/summary/concordance_index.csv", index_col=0)

EVALUATION_TIMES_PEC = pec_summary_pl.columns.to_list()


# %% Concordance Index

# Create a figure instance
fig, ax = plt.subplots()

sns.violinplot(x="Model", y="C-Index", data=c_summary.melt(var_name='Model', value_name='C-Index'), ax=ax)

ax.set_xticklabels(['PL','SPL','RL', "SRL"])
plt.savefig("./plots/c_index_violin_plot.pdf", format="pdf", dpi=500)


# %% Fit Kaplan-Meier Curve on complete data set.
survival_data = np.zeros(event_indicator.shape[0],
    dtype={'names':('event_indicator', 'event_time'), 'formats':('bool', 'u2')})
survival_data['event_indicator'] = event_indicator
survival_data['event_time'] = event_time

kmf = KaplanMeierFitter()
kmf.fit(event_time, event_observed=event_indicator, timeline=pec_summary_pl.columns)
kmf_survival_func = np.array(kmf.survival_function_)
kaplan_preds = np.repeat(kmf_survival_func.T, len(event_indicator), axis=0)

eval_times, kmf_brier_scores = brier_score(survival_data, survival_data, kaplan_preds, EVALUATION_TIMES_PEC)


# %% Plot prediction error curves against baseline Kaplan-Meier plot.

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey=False, sharex=False, figsize=(10,5))

pec_summary_pl.T.plot.line(color="#C0C0C0", legend=False, ax=ax1, linewidth=0.1)
kmf_plot = ax1.plot(eval_times, kmf_brier_scores, color="black")
ax1.set_title("Partial Likelihood")
ax1.set_xlabel("Time")
ax1.set_ylabel("Prediction Error")

pec_summary_spl.T.plot.line(color="#C0C0C0", legend=False, ax=ax2, linewidth=0.1)
ax2.plot(eval_times, kmf_brier_scores, color="black")
ax2.set_title("Stratified Partial Likelihood")
ax2.set_xlabel("Time")
ax2.set_ylabel("Prediction Error")

pec_summary_rl.T.plot.line(color="#C0C0C0", legend=False, ax=ax3, linewidth=0.1, label="Ranking Loss")
ax3.plot(eval_times, kmf_brier_scores, color="black")
ax3.set_title("Ranking Loss")
ax3.set_xlabel("Time")
ax3.set_ylabel("Prediction Error")

pec_summary_srl.T.plot.line(color="#C0C0C0", legend=False, ax=ax4, linewidth=0.1, label="Stratified Ranking Loss")
ax4.plot(eval_times, kmf_brier_scores, color="black")
ax4.set_title("Stratified Ranking Loss")
ax4.set_xlabel("Time")
ax4.set_ylabel("Prediction Error")


fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.5)
plt.savefig("./plots/pecs_kmf_plot.pdf", format="pdf", dpi=200)


# %%  Integrated Prediction Error Curves"
print("\n\nMean and Median integrated prediction error curves:\n")

def calc_integrated_pec(pec_values_array):
    return np.trapz(y=pec_values_array, x=EVALUATION_TIMES_PEC)

integrated_pec_pl = pec_summary_pl.apply(calc_integrated_pec, axis=1)
integrated_pec_spl = pec_summary_spl.apply(calc_integrated_pec, axis=1)
integrated_pec_rl = pec_summary_rl.apply(calc_integrated_pec, axis=1)
integrated_pec_srl = pec_summary_srl.apply(calc_integrated_pec, axis=1)

integrated_pec_df = pd.DataFrame({
    "Partial Likelihood": integrated_pec_pl,
    "Stratified Partial Likelihood": integrated_pec_spl,
    "Ranking Loss": integrated_pec_rl,
    "Stratified Ranking Loss": integrated_pec_srl
})

ax = sns.violinplot(x="Model", y="Prediction Error", data=integrated_pec_df.melt(var_name='Model', value_name='Prediction Error'))

ax.set_xticklabels(['PL','SPL','RL', "SRL"])

plt.savefig("./plots/integrated_pec_violin.pdf", format="pdf", dpi=500)

# Mean integrated pec:
print()
print(integrated_pec_df.mean())

# Median integrated pec:
print()
print(integrated_pec_df.median())


# %% Calculate Mean Shap values.
# Skip this cell, if there are already mean values calculated and saved.

# Placeholders
shap_values_pl = pd.DataFrame(np.zeros(gene_counts.shape), columns=gene_counts.columns, index=gene_counts.index)
shap_values_spl = pd.DataFrame(np.zeros(gene_counts.shape), columns=gene_counts.columns, index=gene_counts.index)
shap_values_rl = pd.DataFrame(np.zeros(gene_counts.shape), columns=gene_counts.columns, index=gene_counts.index)
shap_values_srl = pd.DataFrame(np.zeros(gene_counts.shape), columns=gene_counts.columns, index=gene_counts.index)

i = 0

for direct, subdirects, files in os.walk('./summary'):
    for file_name in files:
        if "shap_pl" in file_name:
            print("Now reading file", os.path.join(direct, file_name))
            shap_values = pd.read_csv(os.path.join(direct, file_name), index_col=0)
            shap_values_pl += shap_values
        elif "shap_spl" in file_name:
            print("Now reading file", os.path.join(direct, file_name))
            shap_values = pd.read_csv(os.path.join(direct, file_name), index_col=0)
            shap_values_spl += shap_values
        elif "shap_rl" in file_name:
            print("Now reading file", os.path.join(direct, file_name))
            shap_values = pd.read_csv(os.path.join(direct, file_name), index_col=0)
            shap_values_rl += shap_values
        elif "shap_srl" in file_name:
            print("Now reading file", os.path.join(direct, file_name))
            shap_values = pd.read_csv(os.path.join(direct, file_name), index_col=0)
            shap_values_srl += shap_values

    i += 1

mean_shap_pl = shap_values_pl / i
mean_shap_spl = shap_values_spl / i
mean_shap_rl = shap_values_rl / i
mean_shap_srl = shap_values_srl / i

# %% Save mean shap values.

mean_shap_pl.to_csv("./summary/mean_shap_values_pl.csv")
mean_shap_spl.to_csv("./summary/mean_shap_values_spl.csv")
mean_shap_rl.to_csv("./summary/mean_shap_values_rl.csv")
mean_shap_srl.to_csv("./summary/mean_shap_values_srl.csv")

# %% SHAP evaluation.

mean_shap_pl = pd.read_csv("./summary/mean_shap_values_pl.csv", index_col="patient_id")
mean_shap_spl = pd.read_csv("./summary/mean_shap_values_spl.csv", index_col="patient_id")
mean_shap_rl = pd.read_csv("./summary/mean_shap_values_rl.csv", index_col="patient_id")
mean_shap_srl = pd.read_csv("./summary/mean_shap_values_srl.csv", index_col="patient_id")

# %% Get SHAP values for every strata.

glioma_patients = data.loc[data.tumor_type == "GLIOMA"].index
kipan_patients = data.loc[data.tumor_type == "KIPAN"].index
brca_patients = data.loc[data.tumor_type == "BRCA"].index

gene_counts_glioma = gene_counts.loc[glioma_patients].to_numpy()
gene_counts_kipan = gene_counts.loc[kipan_patients].to_numpy()
gene_counts_brca = gene_counts.loc[brca_patients].to_numpy()

glioma_mean_shap_values_pl = mean_shap_pl.loc[glioma_patients]
glioma_mean_shap_values_spl = mean_shap_spl.loc[glioma_patients]
glioma_mean_shap_values_rl = mean_shap_rl.loc[glioma_patients]
glioma_mean_shap_values_srl = mean_shap_srl.loc[glioma_patients]

kipan_mean_shap_values_pl = mean_shap_pl.loc[kipan_patients]
kipan_mean_shap_values_spl = mean_shap_spl.loc[kipan_patients]
kipan_mean_shap_values_rl = mean_shap_rl.loc[kipan_patients]
kipan_mean_shap_values_srl = mean_shap_srl.loc[kipan_patients]

brca_mean_shap_values_pl = mean_shap_pl.loc[brca_patients]
brca_mean_shap_values_spl = mean_shap_spl.loc[brca_patients]
brca_mean_shap_values_rl = mean_shap_rl.loc[brca_patients]
brca_mean_shap_values_srl = mean_shap_srl.loc[brca_patients]


# %% SHAP summary plots in .pdf format
shap.summary_plot(glioma_mean_shap_values_pl.astype('float64').to_numpy(), gene_counts_glioma, feature_names=gene_names, show=False, plot_size=(10, 5))
plt.savefig("./plots/shap_plot_glioma_pl.pdf", format="pdf", dpi=500)
plt.clf()

shap.summary_plot(glioma_mean_shap_values_spl.astype('float64').to_numpy(), gene_counts_glioma, feature_names=gene_names, show=False, plot_size=(10, 5))
plt.savefig("./plots/shap_plot_glioma_spl.pdf", format="pdf", dpi=500)
plt.clf()

shap.summary_plot(glioma_mean_shap_values_rl.astype('float64').to_numpy(), gene_counts_glioma, feature_names=gene_names, show=False, plot_size=(10, 5))
plt.savefig("./plots/shap_plot_glioma_rl.pdf", format="pdf", dpi=500)
plt.clf()

shap.summary_plot(glioma_mean_shap_values_srl.astype('float64').to_numpy(), gene_counts_glioma, feature_names=gene_names, show=False, plot_size=(10, 5))
plt.savefig("./plots/shap_plot_glioma_srl.pdf", format="pdf", dpi=500)
plt.clf()

# %%
shap.summary_plot(kipan_mean_shap_values_pl.astype('float64').to_numpy(), gene_counts_kipan, feature_names=gene_names, show=False, plot_size=(10, 5))
plt.savefig("./plots/shap_plot_kipan_pl.pdf", format="pdf", dpi=500)
plt.clf()

shap.summary_plot(kipan_mean_shap_values_spl.astype('float64').to_numpy(), gene_counts_kipan, feature_names=gene_names, show=False, plot_size=(10, 5))
plt.savefig("./plots/shap_plot_kipan_spl.pdf", format="pdf", dpi=500)
plt.clf()

shap.summary_plot(kipan_mean_shap_values_rl.astype('float64').to_numpy(), gene_counts_kipan, feature_names=gene_names, show=False, plot_size=(10, 5))
plt.savefig("./plots/shap_plot_kipan_rl.pdf", format="pdf", dpi=500)
plt.clf()

shap.summary_plot(kipan_mean_shap_values_srl.astype('float64').to_numpy(), gene_counts_kipan, feature_names=gene_names, show=False, plot_size=(10, 5))
plt.savefig("./plots/shap_plot_kipan_srl.pdf", format="pdf", dpi=500)
plt.clf()

# %%
shap.summary_plot(brca_mean_shap_values_pl.astype('float64').to_numpy(), gene_counts_brca, feature_names=gene_names, show=False, plot_size=(10, 5))
plt.savefig("./plots/shap_plot_brca_pl.pdf", format="pdf", dpi=500)
plt.clf()

shap.summary_plot(brca_mean_shap_values_spl.astype('float64').to_numpy(), gene_counts_brca, feature_names=gene_names, show=False, plot_size=(10, 5))
plt.savefig("./plots/shap_plot_brca_spl.pdf", format="pdf", dpi=500)
plt.clf()

shap.summary_plot(brca_mean_shap_values_rl.astype('float64').to_numpy(), gene_counts_brca, feature_names=gene_names, show=False, plot_size=(10, 5))
plt.savefig("./plots/shap_plot_brca_rl.pdf", format="pdf", dpi=500)
plt.clf()

shap.summary_plot(brca_mean_shap_values_srl.astype('float64').to_numpy(), gene_counts_brca, feature_names=gene_names, show=False, plot_size=(10, 5))
plt.savefig("./plots/shap_plot_brca_srl.pdf", format="pdf", dpi=500)
plt.clf()



# %% Add configs to summary files

dir_path = '/mnt/d/publication/summary'
complete_runs_path = [os.path.join(dir_path, config_folder) for config_folder in os.listdir(dir_path) if "[" in config_folder]

sorted_run_paths = sorted(complete_runs_path, key=os.path.getmtime)

# TODO: Insert configs into summary files

# %% Get Top genes based on SHAP values

# BRCA PL
shap_brca_pl_genes = [
                      "PSME1",
                      "XBP1",
                      "VAV3",
                      "KIAA1522",
                      "TSPAN15",
                      "ACADSB",
                      "BTG2",
                      "CLIC6",
                      "CFB",
                      "TLE3",
                      "MLPH",
                      "GREB1",
                      "PTGES3",
                      "RBM3",
                      "PALLD",
                      "FBP1",
                      "PDCD4",
                      "MYH14",
                      "FBLN2",
                      "HTRA3"
                      ]


# BRCA SPL
shap_brca_spl_genes = [
                       "TUBA1B",
                       "TFRC",
                       "PSMD2",
                       "CKAP4",
                       "ACADSB",
                       "TIMP3",
                       "BTG2",
                       "PDCD4",
                       "OGT",
                       "RAN",
                       "RPL3",
                       "RPS9",
                       "COPZ1",
                       "CLIC6",
                       "JUND",
                       "TCP1",
                       "MYH14",
                       "PAICS",
                       "S100A16",
                       "PTGES3"
                       ]

# BRCA RL
shap_brca_rl_genes = [
                    "S100A16",
                    "BTG2",
                    "CFB",
                    "TSPAN15",
                    "XBP1",
                    "TMC5",
                    "C1orf21",
                    "MUC1",
                    "AR",
                    "CLIC6",
                    "ACADSB",
                    "MYH14",
                    "FBLN2",
                    "FXYD3",
                    "RBM3",
                    "GREB1",
                    "PDCD4",
                    "PALLD",
                    "FBP1",
                    "HTRA3"
                    ]

# BRCA SRL

shap_brca_srl_genes = [
                       "SLC1A1",
                       "IARS",
                       "CKAP4",
                       "PSMD2",
                       "AR",
                       "RBM3",
                       "PALLD",
                       "BTG2",
                       "ACADSB",
                       "TCP1",
                       "PTGES3",
                       "PDCD4",
                       "ADAM15",
                       "S100A16",
                       "NCSTN",
                       "FBP1",
                       "CLIC6",
                       "GREB1",
                       "HTRA3",
                       "MYH14"
                       ]

shap_brca_pl_genes.reverse()
shap_brca_spl_genes.reverse()
shap_brca_rl_genes.reverse()
shap_brca_srl_genes.reverse()

result_genes_shap_brca = [shap_brca_pl_genes, shap_brca_spl_genes, shap_brca_rl_genes, shap_brca_srl_genes]
top_genes_brca = list(set().union(*result_genes_shap_brca))

brca_top_mean_shap_pl = brca_mean_shap_values_pl.loc[:, top_genes_brca]
brca_top_mean_shap_spl = brca_mean_shap_values_spl.loc[:, top_genes_brca]
brca_top_mean_shap_rl = brca_mean_shap_values_rl.loc[:, top_genes_brca]
brca_top_mean_shap_srl = brca_mean_shap_values_srl.loc[:, top_genes_brca]

# gene = "MYH14"
model_names = ["PL", "SPL", "RL", "SRL"]

for gene in top_genes_brca:

    temp_df = pd.concat([
        brca_mean_shap_values_pl.loc[:, gene],
        brca_mean_shap_values_spl.loc[:, gene],
        brca_mean_shap_values_rl.loc[:, gene],
        brca_mean_shap_values_srl.loc[:, gene]
        ], axis=1)

    temp_df.columns = model_names

    gene_counts_duplicated = pd.concat([
        gene_counts.loc[brca_patients, gene],
        gene_counts.loc[brca_patients, gene],
        gene_counts.loc[brca_patients, gene],
        gene_counts.loc[brca_patients, gene]
        ], axis=1)

    shap.summary_plot(temp_df.astype('float64').to_numpy(), gene_counts_duplicated, feature_names=model_names, show=False, plot_size=(10, 5))
    plt.savefig("./plots/shap_brca_top_genes_{}.png".format(gene), format="png", dpi=500)
    plt.clf()

# %% GLIOMA

# GLIOMA PL
shap_glioma_pl_genes = [
                      "PDPN",
                      "TUBB2A",
                      "LMF1",
                      "DKK3",
                      "ABLIM1",
                      "MAPK4",
                      "SLC1A3",
                      "TUBA1A",
                      "GPR37L1",
                      "CIRBP",
                      "GLUD1",
                      "LUZP2",
                      "TMSB4X",
                      "TJP2",
                      "DLL3",
                      "TUBA1B",
                      "GNA12",
                      "TUBB2B",
                      "TNK2",
                      "SHD"
                      ]

# GLIOMA SPL
shap_glioma_spl_genes = [
                       "HES6",
                       "GNA12",
                       "TJP2",
                       "ZDHHC22",
                       "TMSB4X",
                       "FLJ16779",
                       "ZCCHC24",
                       "SORBS1",
                       "CIRBP",
                       "LMF1",
                       "ABLIM1",
                       "ID4",
                       "GLUD1",
                       "LUZP2",
                       "TNK2",
                       "SPOCK2",
                       "TUBA1B",
                       "FXYD6",
                       "DLL3",
                       "SHD"
                       ]

# GLIOMA RL
shap_glioma_rl_genes = [
                    "PDPN",
                    "GLUD2",
                    "DGCR2",
                    "TMSB4X",
                    "CD99",
                    "MAPT",
                    "FLJ16779",
                    "GPR37L1",
                    "DKK3",
                    "TUBB2B",
                    "TUBA1A",
                    "GNA12",
                    "TUBA1B",
                    "SLC1A3",
                    "TJP2",
                    "GLUD1",
                    "LUZP2",
                    "TNK2",
                    "SHD",
                    "DLL3"
                    ]

# GLIOMA SRL
shap_glioma_srl_genes = [
                       "FLNC",
                       "HES6",
                       "FLJ16779",
                       "RAP2A",
                       "ADD3",
                       "LMF1",
                       "APOE",
                       "ZCCHC24",
                       "OLIG2",
                       "DKK3",
                       "FXYD6",
                       "TUBA1B",
                       "SORBS1",
                       "TNK2",
                       "ID4",
                       "TJP2",
                       "GLUD1",
                       "SHD",
                       "DLL3",
                       "LUZP2"
                       ]

shap_glioma_pl_genes.reverse()
shap_glioma_spl_genes.reverse()
shap_glioma_rl_genes.reverse()
shap_glioma_srl_genes.reverse()

result_genes_shap_glioma = [shap_glioma_pl_genes, shap_glioma_spl_genes, shap_glioma_rl_genes, shap_glioma_srl_genes]
top_genes_glioma = list(set().union(*result_genes_shap_glioma))

glioma_top_mean_shap_pl = glioma_mean_shap_values_pl.loc[:, top_genes_glioma]
glioma_top_mean_shap_spl = glioma_mean_shap_values_spl.loc[:, top_genes_glioma]
glioma_top_mean_shap_rl = glioma_mean_shap_values_rl.loc[:, top_genes_glioma]
glioma_top_mean_shap_srl = glioma_mean_shap_values_srl.loc[:, top_genes_glioma]

model_names = ["PL", "SPL", "RL", "SRL"]

for gene in top_genes_glioma:

    temp_df = pd.concat([
        glioma_mean_shap_values_pl.loc[:, gene],
        glioma_mean_shap_values_spl.loc[:, gene],
        glioma_mean_shap_values_rl.loc[:, gene],
        glioma_mean_shap_values_srl.loc[:, gene]
        ], axis=1)

    temp_df.columns = model_names

    gene_counts_duplicated = pd.concat([
        gene_counts.loc[glioma_patients, gene],
        gene_counts.loc[glioma_patients, gene],
        gene_counts.loc[glioma_patients, gene],
        gene_counts.loc[glioma_patients, gene]
        ], axis=1)

    shap.summary_plot(temp_df.astype('float64').to_numpy(), gene_counts_duplicated, feature_names=model_names, show=False, plot_size=(10, 5))
    plt.savefig("./plots/shap_glioma_top_genes_{}.png".format(gene), format="png", dpi=500)
    plt.clf()


# %% KIPAN

# KIPAN PL
shap_kipan_pl_genes = [
                      "THY1",
                      "CIT",
                      "NFKBIA",
                      "OGT",
                      "MALAT1",
                      "WSB1",
                      "UGT2A3",
                      "RHOB",
                      "NPR3",
                      "ARRDC3",
                      "NNMT",
                      "CUBN",
                      "RNASET2",
                      "VEGFA",
                      "FKBP10",
                      "TMEM27",
                      "LOC100132247",
                      "SHMT2",
                      "SLC16A12",
                      "ALPK2"
                      ]

# KIPAN SPL
shap_kipan_spl_genes = [
                       "C19orf77",
                       "UGT2B7",
                       "SLC22A2",
                       "CRYL1",
                       "VEGFA",
                       "RHOB",
                       "EPAS1",
                       "MALAT1",
                       "RNASET2",
                       "KCNJ15",
                       "NPR3",
                       "ARRDC3",
                       "UGT2A3",
                       "FKBP10",
                       "SHMT2",
                       "CUBN",
                       "TMEM27",
                       "LOC100132247",
                       "ALPK2",
                       "SLC16A12"
                       ]

# KIPAN RL
shap_kipan_rl_genes = [
                    "PILRB",
                    "IMPA2",
                    "SLC16A3",
                    "CIT",
                    "TSPAN1",
                    "FKBP10",
                    "UGT2A3",
                    "RHOB",
                    "KCNJ15",
                    "RNASET2",
                    "CUBN",
                    "VEGFA",
                    "NPR3",
                    "ARRDC3",
                    "LOC100132247",
                    "SLC16A12",
                    "RARRES2",
                    "ALPK2",
                    "TMEM27",
                    "SHMT2"
                    ]

# KIPAN SRL
shap_kipan_srl_genes = [
                       "COX4I1",
                       "FBXL5",
                       "MACC1",
                       "FKBP10",
                       "C19orf77",
                       "RHOB",
                       "TSPAN1",
                       "CYS1",
                       "EPAS1",
                       "RARRES2",
                       "UGT2A3",
                       "NPR3",
                       "KCNJ15",
                       "CUBN",
                       "ARRDC3",
                       "ALPK2",
                       "LOC100132247",
                       "SLC16A12",
                       "TMEM27",
                       "SHMT2",
                       ]

shap_kipan_pl_genes.reverse()
shap_kipan_spl_genes.reverse()
shap_kipan_rl_genes.reverse()
shap_kipan_srl_genes.reverse()

result_genes_shap_kipan = [shap_kipan_pl_genes, shap_kipan_spl_genes, shap_kipan_rl_genes, shap_kipan_srl_genes]
top_genes_kipan = list(set().union(*result_genes_shap_kipan))

kipan_top_mean_shap_pl = kipan_mean_shap_values_pl.loc[:, top_genes_kipan]
kipan_top_mean_shap_spl = kipan_mean_shap_values_spl.loc[:, top_genes_kipan]
kipan_top_mean_shap_rl = kipan_mean_shap_values_rl.loc[:, top_genes_kipan]
kipan_top_mean_shap_srl = kipan_mean_shap_values_srl.loc[:, top_genes_kipan]

model_names = ["PL", "SPL", "RL", "SRL"]

for gene in top_genes_kipan:

    temp_df = pd.concat([
        kipan_mean_shap_values_pl.loc[:, gene],
        kipan_mean_shap_values_spl.loc[:, gene],
        kipan_mean_shap_values_rl.loc[:, gene],
        kipan_mean_shap_values_srl.loc[:, gene]
        ], axis=1)

    temp_df.columns = model_names

    gene_counts_duplicated = pd.concat([
        gene_counts.loc[kipan_patients, gene],
        gene_counts.loc[kipan_patients, gene],
        gene_counts.loc[kipan_patients, gene],
        gene_counts.loc[kipan_patients, gene]
        ], axis=1)

    shap.summary_plot(temp_df.astype('float64').to_numpy(), gene_counts_duplicated, feature_names=model_names, show=False, plot_size=(10, 5))
    plt.savefig("./plots/shap_kipan_top_genes_{}.png".format(gene), format="png", dpi=500)
    plt.clf()
