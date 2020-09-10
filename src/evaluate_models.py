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
gene_names = gene_counts.columns
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


