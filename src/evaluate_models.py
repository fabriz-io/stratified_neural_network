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


# %% Violin Plot Concordance Index

fig, ax = plt.subplots()

sns.violinplot(x="Model", y="C-Index", data=c_summary.melt(var_name='Model', value_name='C-Index'), ax=ax)

ax.set_xticklabels(['PL','SPL','RL', "SRL"])
plt.savefig("./plots/c_index_violin_plot.pdf", format="pdf", dpi=500)


# %% Plot prediction error curves against baseline Kaplan-Meier plot.

# Specify maximum evaluation time in days
MAX_EVAL_TIME_PEC = 1500


EVALUATION_TIMES_PEC = pec_summary_pl.columns.to_numpy()
EVALUATION_TIMES_PEC = EVALUATION_TIMES_PEC[EVALUATION_TIMES_PEC <= MAX_EVAL_TIME_PEC]

# Fit Kaplan-Meier Curve on complete data set
survival_data = np.zeros(event_indicator.shape[0],
    dtype={'names':('event_indicator', 'event_time'), 'formats':('bool', 'u2')})
survival_data['event_indicator'] = event_indicator
survival_data['event_time'] = event_time

kmf = KaplanMeierFitter()
kmf.fit(event_time, event_observed=event_indicator, timeline=EVALUATION_TIMES_PEC)
kmf_survival_func = np.array(kmf.survival_function_)
kaplan_preds = np.repeat(kmf_survival_func.T, len(event_indicator), axis=0)

eval_times, kmf_brier_scores = brier_score(survival_data, survival_data, kaplan_preds, EVALUATION_TIMES_PEC)

# Plot prediction error curves
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey=True, sharex=True, figsize=(10,5))

pec_summary_pl.loc[:, EVALUATION_TIMES_PEC].T.plot.line(color="#C0C0C0", legend=False, ax=ax1, linewidth=0.1, label="Partial Likelihood")
kmf_plot = ax1.plot(eval_times, kmf_brier_scores, color="black")
ax1.set_title("Partial Likelihood")
ax1.set_xlabel("Time")
ax1.set_ylabel("Prediction Error")

pec_summary_spl.loc[:, EVALUATION_TIMES_PEC].T.plot.line(color="#C0C0C0", legend=False, ax=ax2, linewidth=0.1, label="Stratified Partial Likelihood")
ax2.plot(eval_times, kmf_brier_scores, color="black")
ax2.set_title("Stratified Partial Likelihood")
ax2.set_xlabel("Time")
ax2.set_ylabel("Prediction Error")

pec_summary_rl.loc[:, EVALUATION_TIMES_PEC].T.plot.line(color="#C0C0C0", legend=False, ax=ax3, linewidth=0.1, label="Ranking Loss")
ax3.plot(eval_times, kmf_brier_scores, color="black")
ax3.set_title("Ranking Loss")
ax3.set_xlabel("Time")
ax3.set_ylabel("Prediction Error")

pec_summary_srl.loc[:, EVALUATION_TIMES_PEC].T.plot.line(color="#C0C0C0", legend=False, ax=ax4, linewidth=0.1, label="Stratified Ranking Loss")
ax4.plot(eval_times, kmf_brier_scores, color="black")
ax4.set_title("Stratified Ranking Loss")
ax4.set_xlabel("Time")
ax4.set_ylabel("Prediction Error")


fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.5)
plt.savefig("./plots/pecs_kmf_plot_maxT_{}.pdf".format(MAX_EVAL_TIME_PEC), format="pdf", dpi=200)
plt.show()


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

# Save mean SHAP values.

mean_shap_pl.to_csv("./summary/mean_shap_values_pl.csv")
mean_shap_spl.to_csv("./summary/mean_shap_values_spl.csv")
mean_shap_rl.to_csv("./summary/mean_shap_values_rl.csv")
mean_shap_srl.to_csv("./summary/mean_shap_values_srl.csv")

# %% Load mean SHAP values.

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


# %% Order genes based on sum of absolute SHAP values

# BRCA top genes per model
feature_order_brca_pl = np.argsort(np.sum(np.abs(brca_mean_shap_values_pl), axis=0))
feature_order_brca_spl = np.argsort(np.sum(np.abs(brca_mean_shap_values_spl), axis=0))
feature_order_brca_rl = np.argsort(np.sum(np.abs(brca_mean_shap_values_rl), axis=0))
feature_order_brca_srl = np.argsort(np.sum(np.abs(brca_mean_shap_values_srl), axis=0))

shap_brca_pl_top_genes = brca_mean_shap_values_pl.iloc[:, feature_order_brca_pl[::-1]]
shap_brca_spl_top_genes = brca_mean_shap_values_spl.iloc[:, feature_order_brca_spl[::-1]]
shap_brca_rl_top_genes = brca_mean_shap_values_rl.iloc[:, feature_order_brca_rl[::-1]]
shap_brca_srl_top_genes = brca_mean_shap_values_srl.iloc[:, feature_order_brca_srl[::-1]]

# GLIOMA top genes
feature_order_glioma_pl = np.argsort(np.sum(np.abs(glioma_mean_shap_values_pl), axis=0))
feature_order_glioma_spl = np.argsort(np.sum(np.abs(glioma_mean_shap_values_spl), axis=0))
feature_order_glioma_rl = np.argsort(np.sum(np.abs(glioma_mean_shap_values_rl), axis=0))
feature_order_glioma_srl = np.argsort(np.sum(np.abs(glioma_mean_shap_values_srl), axis=0))

shap_glioma_pl_top_genes = glioma_mean_shap_values_pl.iloc[:, feature_order_glioma_pl[::-1]]
shap_glioma_spl_top_genes = glioma_mean_shap_values_spl.iloc[:, feature_order_glioma_spl[::-1]]
shap_glioma_rl_top_genes = glioma_mean_shap_values_rl.iloc[:, feature_order_glioma_rl[::-1]]
shap_glioma_srl_top_genes = glioma_mean_shap_values_srl.iloc[:, feature_order_glioma_srl[::-1]]

# KIPAN top genes
feature_order_kipan_pl = np.argsort(np.sum(np.abs(kipan_mean_shap_values_pl), axis=0))
feature_order_kipan_spl = np.argsort(np.sum(np.abs(kipan_mean_shap_values_spl), axis=0))
feature_order_kipan_rl = np.argsort(np.sum(np.abs(kipan_mean_shap_values_rl), axis=0))
feature_order_kipan_srl = np.argsort(np.sum(np.abs(kipan_mean_shap_values_srl), axis=0))

shap_kipan_pl_top_genes = kipan_mean_shap_values_pl.iloc[:, feature_order_kipan_pl[::-1]]
shap_kipan_spl_top_genes = kipan_mean_shap_values_spl.iloc[:, feature_order_kipan_spl[::-1]]
shap_kipan_rl_top_genes = kipan_mean_shap_values_rl.iloc[:, feature_order_kipan_rl[::-1]]
shap_kipan_srl_top_genes = kipan_mean_shap_values_srl.iloc[:, feature_order_kipan_srl[::-1]]


#%% Helper function for generating a subplot of SHAP summaries

def subplots_shap_summary(shap_dfs, patient_list, gene_list, nrows, ncols, fig_height=10, fig_width=10, subplot_height=5, subplot_width=8, wspace=0.2, hspace=0.6, subplot_title_size=20, save=False):
    model_names = ["PL", "SPL", "RL", "SRL"]

    # fig_height = 5.8 * fig_factor
    # fig_width = 8.3 * fig_factor

    # subplot_height = 5 * sub_factor
    # subplot_width = 8 * sub_factor

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, squeeze=False, figsize=(fig_width, fig_height))

    for gene, axis in zip(sorted(gene_list[:nrows*ncols]), ax.flatten()):

        # Concatenate SHAP values from all models for current gene
        temp_df = pd.concat([df.loc[:, gene] for df in shap_dfs], axis=1)

        gene_counts_duplicated = pd.concat([gene_counts.loc[patient_list, gene] for _ in range(len(shap_dfs))], axis=1)

        plt.sca(axis)
        plt.gca().set_title(gene, size=subplot_title_size)
        shap.summary_plot(temp_df.astype('float64').to_numpy(), gene_counts_duplicated, feature_names=model_names, show=False, color_bar=False, sort=False, plot_size=(subplot_width, subplot_height))

    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    # TODO: Look where to save figure
    if save:
        pass

    plt.show()


# %% Plot top N Genes GLIOMA based on SHAP values

N = 6

top_N_genes_glioma = [shap_glioma_pl_top_genes.columns[:N],
                    shap_glioma_spl_top_genes.columns[:N],
                    shap_glioma_rl_top_genes.columns[:N],
                    shap_glioma_srl_top_genes.columns[:N]]

top_N_genes_glioma_union_set = list(set().union(*top_N_genes_glioma))

print(len(top_N_genes_glioma_union_set))

subplots_shap_summary(
    [glioma_mean_shap_values_pl,
     glioma_mean_shap_values_spl,
     glioma_mean_shap_values_rl,
     glioma_mean_shap_values_srl], glioma_patients, sorted(top_N_genes_glioma_union_set), nrows=4, ncols=3, fig_height=100, fig_width=100, subplot_height=40, subplot_width=52, wspace=0.15, hspace=0.35, subplot_title_size=30)

# %% Plot top N Genes BRCA based on SHAP values

N = 5

top_N_genes_brca = [shap_brca_pl_top_genes.columns[:N],
                    shap_brca_spl_top_genes.columns[:N],
                    shap_brca_rl_top_genes.columns[:N],
                    shap_brca_srl_top_genes.columns[:N]]

top_N_genes_brca_union_set = list(set().union(*top_N_genes_brca))

print(len(top_N_genes_brca_union_set))

subplots_shap_summary(
    [brca_mean_shap_values_pl,
     brca_mean_shap_values_spl,
     brca_mean_shap_values_rl,
     brca_mean_shap_values_srl], brca_patients, sorted(top_N_genes_brca_union_set), nrows=4, ncols=3, fig_height=100, fig_width=100, subplot_height=40, subplot_width=52, wspace=0.15, hspace=0.35, subplot_title_size=30)


# %% Plot top N Genes KIPAN based on SHAP values

N = 7

top_N_genes_kipan = [shap_kipan_pl_top_genes.columns[:N],
                    shap_kipan_spl_top_genes.columns[:N],
                    shap_kipan_rl_top_genes.columns[:N],
                    shap_kipan_srl_top_genes.columns[:N]]

top_N_genes_kipan_union_set = list(set().union(*top_N_genes_kipan))

print(len(top_N_genes_kipan_union_set))

subplots_shap_summary(
    [kipan_mean_shap_values_pl,
     kipan_mean_shap_values_spl,
     kipan_mean_shap_values_rl,
     kipan_mean_shap_values_srl], kipan_patients, sorted(top_N_genes_kipan_union_set), nrows=5, ncols=2, fig_height=100, fig_width=100, subplot_height=40, subplot_width=32, wspace=0.15, hspace=0.4, subplot_title_size=30)


#%% Top different genes GLIOMA

no_of_difference_genes = 7

partial_likelihood_differences = np.abs((np.sum(np.abs(glioma_mean_shap_values_pl)) - np.sum(np.abs(glioma_mean_shap_values_spl)))).sort_values(ascending=False)
ranking_loss_differences = np.abs((np.sum(np.abs(glioma_mean_shap_values_rl)) - np.sum(np.abs(glioma_mean_shap_values_srl)))).sort_values(ascending=False)

top_difference_genes = [partial_likelihood_differences[:no_of_difference_genes].index,
                        ranking_loss_differences[:no_of_difference_genes].index]

top_difference_genes_set = list(set().union(*top_difference_genes))
print(len(top_difference_genes_set))

subplots_shap_summary(
    [glioma_mean_shap_values_pl,
     glioma_mean_shap_values_spl,
     glioma_mean_shap_values_rl,
     glioma_mean_shap_values_srl], glioma_patients, sorted(top_difference_genes_set), 4, 3, fig_height=100, fig_width=60, subplot_height=40, subplot_width=32, wspace=0.15, hspace=0.35, subplot_title_size=25)

#%% Top different genes BRCA

no_of_difference_genes = 7

partial_likelihood_differences = np.abs((np.sum(np.abs(brca_mean_shap_values_pl)) - np.sum(np.abs(brca_mean_shap_values_spl)))).sort_values(ascending=False)
ranking_loss_differences = np.abs((np.sum(np.abs(brca_mean_shap_values_rl)) - np.sum(np.abs(brca_mean_shap_values_srl)))).sort_values(ascending=False)

top_difference_genes = [partial_likelihood_differences[:no_of_difference_genes].index,
                        ranking_loss_differences[:no_of_difference_genes].index]

top_difference_genes_set = list(set().union(*top_difference_genes))

print(len(top_difference_genes_set))

subplots_shap_summary(
    [brca_mean_shap_values_pl,
     brca_mean_shap_values_spl,
     brca_mean_shap_values_rl,
     brca_mean_shap_values_srl], brca_patients, sorted(top_difference_genes_set), 4, 3, fig_height=100, fig_width=60, subplot_height=40, subplot_width=32, wspace=0.15, hspace=0.35, subplot_title_size=30)


#%% Top different genes KIPAN

no_of_difference_genes = 7

partial_likelihood_differences = np.abs((np.sum(np.abs(kipan_mean_shap_values_pl)) - np.sum(np.abs(kipan_mean_shap_values_spl)))).sort_values(ascending=False)
ranking_loss_differences = np.abs((np.sum(np.abs(kipan_mean_shap_values_rl)) - np.sum(np.abs(kipan_mean_shap_values_srl)))).sort_values(ascending=False)

top_difference_genes = [partial_likelihood_differences[:no_of_difference_genes].index,
                        ranking_loss_differences[:no_of_difference_genes].index]

top_difference_genes_set = list(set().union(*top_difference_genes))

print(len(top_difference_genes_set))

subplots_shap_summary(
    [kipan_mean_shap_values_pl,
     kipan_mean_shap_values_spl,
     kipan_mean_shap_values_rl,
     kipan_mean_shap_values_srl], kipan_patients, sorted(top_difference_genes_set), 4, 3, fig_height=100, fig_width=60, subplot_height=40, subplot_width=32, wspace=0.15, hspace=0.35, subplot_title_size=30)


# %% Bar Plots BRCA

tumor = "brca"

shap.summary_plot(brca_mean_shap_values_pl.astype('float64').to_numpy(), gene_counts_brca, feature_names=gene_names, show=False, plot_size=(10, 5), plot_type="bar")
plt.savefig("./plots/shap_bar_plot_{}_pl.png".format(tumor), format="png", dpi=500)
plt.show()

shap.summary_plot(brca_mean_shap_values_spl.astype('float64').to_numpy(), gene_counts_brca, feature_names=gene_names, show=False, plot_size=(10, 5), plot_type="bar")
plt.savefig("./plots/shap_bar_plot_{}_spl.png".format(tumor), format="png", dpi=500)
plt.show()

shap.summary_plot(brca_mean_shap_values_rl.astype('float64').to_numpy(), gene_counts_brca, feature_names=gene_names, show=False, plot_size=(10, 5), plot_type="bar")
plt.savefig("./plots/shap_bar_plot_{}_rl.png".format(tumor), format="png", dpi=500)
plt.show()

shap.summary_plot(brca_mean_shap_values_srl.astype('float64').to_numpy(), gene_counts_brca, feature_names=gene_names, show=False, plot_size=(10, 5), plot_type="bar")
plt.savefig("./plots/shap_bar_plot_{}_srl.png".format(tumor), format="png", dpi=500)
plt.show()

# %% Bar plots KIPAN

tumor = "kipan"

shap.summary_plot(kipan_mean_shap_values_pl.astype('float64').to_numpy(), gene_counts_kipan, feature_names=gene_names, show=False, plot_size=(10, 5), plot_type="bar")
plt.savefig("./plots/shap_bar_plot_{}_pl.png".format(tumor), format="png", dpi=500)
plt.show()

shap.summary_plot(kipan_mean_shap_values_spl.astype('float64').to_numpy(), gene_counts_kipan, feature_names=gene_names, show=False, plot_size=(10, 5), plot_type="bar")
plt.savefig("./plots/shap_bar_plot_{}_spl.png".format(tumor), format="png", dpi=500)
plt.show()

shap.summary_plot(kipan_mean_shap_values_rl.astype('float64').to_numpy(), gene_counts_kipan, feature_names=gene_names, show=False, plot_size=(10, 5), plot_type="bar")
plt.savefig("./plots/shap_bar_plot_{}_rl.png".format(tumor), format="png", dpi=500)
plt.show()

shap.summary_plot(kipan_mean_shap_values_srl.astype('float64').to_numpy(), gene_counts_kipan, feature_names=gene_names, show=False, plot_size=(10, 5), plot_type="bar")
plt.savefig("./plots/shap_bar_plot_{}_srl.png".format(tumor), format="png", dpi=500)
plt.show()


# %% Bar plots GLIOMA

tumor = "glioma"


shap.summary_plot(glioma_mean_shap_values_pl.astype('float64').to_numpy(), gene_counts_glioma, feature_names=gene_names, show=False, plot_size=(10, 5), plot_type="bar")
plt.savefig("./plots/shap_bar_plot_{}_pl.png".format(tumor), format="png", dpi=500)
plt.show()

shap.summary_plot(glioma_mean_shap_values_spl.astype('float64').to_numpy(), gene_counts_glioma, feature_names=gene_names, show=False, plot_size=(10, 5), plot_type="bar")
plt.savefig("./plots/shap_bar_plot_{}_spl.png".format(tumor), format="png", dpi=500)
plt.show()

shap.summary_plot(glioma_mean_shap_values_rl.astype('float64').to_numpy(), gene_counts_glioma, feature_names=gene_names, show=False, plot_size=(10, 5), plot_type="bar")
plt.savefig("./plots/shap_bar_plot_{}_rl.png".format(tumor), format="png", dpi=500)
plt.show()

shap.summary_plot(glioma_mean_shap_values_srl.astype('float64').to_numpy(), gene_counts_glioma, feature_names=gene_names, show=False, plot_size=(10, 5), plot_type="bar")
plt.savefig("./plots/shap_bar_plot_{}_srl.png".format(tumor), format="png", dpi=500)
plt.show()


# %% Boxplot SHAP values GLIOMA

# 75% quantile
fig, ax = plt.subplots()

df = pd.DataFrame({
    'Partial Likelihood': glioma_mean_shap_values_pl.quantile(0.75),
    'Stratified Partial Likelihood': glioma_mean_shap_values_spl.quantile(0.75),
    'Ranking Loss': glioma_mean_shap_values_rl.quantile(0.75),
    'Stratified Ranking Loss': glioma_mean_shap_values_srl.quantile(0.75)
})

sns.boxplot(x="Model", y='75% quantile SHAP values', data=df.melt(var_name='Model', value_name='75% quantile SHAP values'), ax=ax)
ax.set_xticklabels(['PL','SPL','RL', "SRL"])
plt.title("75% quantile of SHAP values across all genes")

plt.show()
plt.clf()

# %% Mean

fig, ax = plt.subplots()

df = pd.DataFrame({
    'Partial Likelihood': glioma_mean_shap_values_pl.mean(),
    'Stratified Partial Likelihood': glioma_mean_shap_values_spl.mean(),
    'Ranking Loss': glioma_mean_shap_values_rl.mean(),
    'Stratified Ranking Loss': glioma_mean_shap_values_srl.mean()
})

sns.boxplot(x="Model", y='Mean SHAP values', data=df.melt(var_name='Model', value_name='Mean SHAP values'), ax=ax)
ax.set_xticklabels(['PL','SPL','RL', "SRL"])
plt.title("Mean SHAP values across all genes")



# %% Mean 500

N = 500

top_N_genes_glioma = [shap_glioma_pl_top_genes.columns[:N],
                      shap_glioma_spl_top_genes.columns[:N],
                      shap_glioma_rl_top_genes.columns[:N],
                      shap_glioma_srl_top_genes.columns[:N]]

top_N_genes_glioma_union_set = list(set().union(*top_N_genes_glioma))

df = pd.DataFrame({
    'Partial Likelihood': glioma_mean_shap_values_pl.loc[:, top_N_genes_glioma_union_set].mean(),
    'Stratified Partial Likelihood': glioma_mean_shap_values_spl.loc[:, top_N_genes_glioma_union_set].mean(),
    'Ranking Loss': glioma_mean_shap_values_rl.loc[:, top_N_genes_glioma_union_set].mean(),
    'Stratified Ranking Loss': glioma_mean_shap_values_srl.loc[:, top_N_genes_glioma_union_set].mean()
})

fig, ax = plt.subplots()
sns.boxplot(x="Model", y='Mean SHAP values', data=df.melt(var_name='Model', value_name='Mean SHAP values'), ax=ax)
ax.set_xticklabels(['PL','SPL','RL', "SRL"])
plt.title("Mean SHAP values of top {} genes".format(N))

# %% Boxplot SHAP values BRCA ______________________________________________________________________________________
# __________________________________________________________________________________________________________________
# __________________________________________________________________________________________________________________

# Create a figure instance
fig, ax = plt.subplots()

df = pd.DataFrame({
    'Partial Likelihood': brca_mean_shap_values_pl.quantile(0.75),
    'Stratified Partial Likelihood': brca_mean_shap_values_spl.quantile(0.75),
    'Ranking Loss': brca_mean_shap_values_rl.quantile(0.75),
    'Stratified Ranking Loss': brca_mean_shap_values_srl.quantile(0.75)
})

sns.boxplot(x="Model", y='75% quantile SHAP values', data=df.melt(var_name='Model', value_name='75% quantile SHAP values'), ax=ax)
ax.set_xticklabels(['PL','SPL','RL', "SRL"])
plt.title("75% quantile of SHAP values across all genes")

plt.show()
plt.clf()

# %%
# Create a figure instance
fig, ax = plt.subplots()

df = pd.DataFrame({
    'Partial Likelihood': brca_mean_shap_values_pl.mean(),
    'Stratified Partial Likelihood': brca_mean_shap_values_spl.mean(),
    'Ranking Loss': brca_mean_shap_values_rl.mean(),
    'Stratified Ranking Loss': brca_mean_shap_values_srl.mean()
})

sns.boxplot(x="Model", y='Mean SHAP values', data=df.melt(var_name='Model', value_name='Mean SHAP values'), ax=ax)
ax.set_xticklabels(['PL','SPL','RL', "SRL"])
plt.title("Mean SHAP values across all genes")



# %%
N = 500

top_N_genes_brca = [shap_brca_pl_top_genes.columns[:N],
                      shap_brca_spl_top_genes.columns[:N],
                      shap_brca_rl_top_genes.columns[:N],
                      shap_brca_srl_top_genes.columns[:N]]

top_N_genes_brca_union_set = list(set().union(*top_N_genes_brca))

df = pd.DataFrame({
    'Partial Likelihood': brca_mean_shap_values_pl.loc[:, top_N_genes_brca_union_set].mean(),
    'Stratified Partial Likelihood': brca_mean_shap_values_spl.loc[:, top_N_genes_brca_union_set].mean(),
    'Ranking Loss': brca_mean_shap_values_rl.loc[:, top_N_genes_brca_union_set].mean(),
    'Stratified Ranking Loss': brca_mean_shap_values_srl.loc[:, top_N_genes_brca_union_set].mean()
})

fig, ax = plt.subplots()
sns.boxplot(x="Model", y='Mean SHAP values', data=df.melt(var_name='Model', value_name='Mean SHAP values'), ax=ax)
ax.set_xticklabels(['PL','SPL','RL', "SRL"])
plt.title("Mean SHAP values of top {} genes".format(N))


# %% Boxplot SHAP values KIPAN _____________________________________________________________________________________
# __________________________________________________________________________________________________________________
# __________________________________________________________________________________________________________________

# Create a figure instance
fig, ax = plt.subplots()

df = pd.DataFrame({
    'Partial Likelihood': kipan_mean_shap_values_pl.quantile(0.75),
    'Stratified Partial Likelihood': kipan_mean_shap_values_spl.quantile(0.75),
    'Ranking Loss': kipan_mean_shap_values_rl.quantile(0.75),
    'Stratified Ranking Loss': kipan_mean_shap_values_srl.quantile(0.75)
})

sns.boxplot(x="Model", y='75% quantile SHAP values', data=df.melt(var_name='Model', value_name='75% quantile SHAP values'), ax=ax)
ax.set_xticklabels(['PL','SPL','RL', "SRL"])
plt.title("75% quantile of SHAP values across all genes")

plt.show()
plt.clf()

# %%
# Create a figure instance
fig, ax = plt.subplots()

df = pd.DataFrame({
    'Partial Likelihood': kipan_mean_shap_values_pl.mean(),
    'Stratified Partial Likelihood': kipan_mean_shap_values_spl.mean(),
    'Ranking Loss': kipan_mean_shap_values_rl.mean(),
    'Stratified Ranking Loss': kipan_mean_shap_values_srl.mean()
})

sns.boxplot(x="Model", y='Mean SHAP values', data=df.melt(var_name='Model', value_name='Mean SHAP values'), ax=ax)
ax.set_xticklabels(['PL','SPL','RL', "SRL"])
plt.title("Mean SHAP values across all genes")



# %%
N = 500

top_N_genes_kipan = [shap_kipan_pl_top_genes.columns[:N],
                      shap_kipan_spl_top_genes.columns[:N],
                      shap_kipan_rl_top_genes.columns[:N],
                      shap_kipan_srl_top_genes.columns[:N]]

top_N_genes_kipan_union_set = list(set().union(*top_N_genes_kipan))

df = pd.DataFrame({
    'Partial Likelihood': kipan_mean_shap_values_pl.loc[:, top_N_genes_kipan_union_set].mean(),
    'Stratified Partial Likelihood': kipan_mean_shap_values_spl.loc[:, top_N_genes_kipan_union_set].mean(),
    'Ranking Loss': kipan_mean_shap_values_rl.loc[:, top_N_genes_kipan_union_set].mean(),
    'Stratified Ranking Loss': kipan_mean_shap_values_srl.loc[:, top_N_genes_kipan_union_set].mean()
})

fig, ax = plt.subplots()
sns.boxplot(x="Model", y='Mean SHAP values', data=df.melt(var_name='Model', value_name='Mean SHAP values'), ax=ax)
ax.set_xticklabels(['PL','SPL','RL', "SRL"])
plt.title("Mean SHAP values of top {} genes".format(N))



# # %% SHAP summary plots in .pdf format
# shap.summary_plot(glioma_mean_shap_values_pl.astype('float64').to_numpy(), gene_counts_glioma, feature_names=gene_names, show=False, plot_size=(10, 5))
# plt.savefig("./plots/shap_plot_glioma_pl.pdf", format="pdf", dpi=500)
# plt.clf()
# #%%
# shap.summary_plot(glioma_mean_shap_values_spl.astype('float64').to_numpy(), gene_counts_glioma, feature_names=gene_names, show=False, plot_size=(10, 5))
# plt.savefig("./plots/shap_plot_glioma_spl.pdf", format="pdf", dpi=500)
# plt.clf()

# shap.summary_plot(glioma_mean_shap_values_rl.astype('float64').to_numpy(), gene_counts_glioma, feature_names=gene_names, show=False, plot_size=(10, 5))
# plt.savefig("./plots/shap_plot_glioma_rl.pdf", format="pdf", dpi=500)
# plt.clf()

# shap.summary_plot(glioma_mean_shap_values_srl.astype('float64').to_numpy(), gene_counts_glioma, feature_names=gene_names, show=False, plot_size=(10, 5))
# plt.savefig("./plots/shap_plot_glioma_srl.pdf", format="pdf", dpi=500)
# plt.clf()

# # %%
# shap.summary_plot(kipan_mean_shap_values_pl.astype('float64').to_numpy(), gene_counts_kipan, feature_names=gene_names, show=False, plot_size=(10, 5))
# plt.savefig("./plots/shap_plot_kipan_pl.pdf", format="pdf", dpi=500)
# plt.clf()

# shap.summary_plot(kipan_mean_shap_values_spl.astype('float64').to_numpy(), gene_counts_kipan, feature_names=gene_names, show=False, plot_size=(10, 5))
# plt.savefig("./plots/shap_plot_kipan_spl.pdf", format="pdf", dpi=500)
# plt.clf()

# shap.summary_plot(kipan_mean_shap_values_rl.astype('float64').to_numpy(), gene_counts_kipan, feature_names=gene_names, show=False, plot_size=(10, 5))
# plt.savefig("./plots/shap_plot_kipan_rl.pdf", format="pdf", dpi=500)
# plt.clf()

# shap.summary_plot(kipan_mean_shap_values_srl.astype('float64').to_numpy(), gene_counts_kipan, feature_names=gene_names, show=False, plot_size=(10, 5))
# plt.savefig("./plots/shap_plot_kipan_srl.pdf", format="pdf", dpi=500)
# plt.clf()

# # %%
# shap.summary_plot(brca_mean_shap_values_pl.astype('float64').to_numpy(), gene_counts_brca, feature_names=gene_names, show=False, plot_size=(10, 5))
# plt.savefig("./plots/shap_plot_brca_pl.pdf", format="pdf", dpi=500)
# plt.clf()

# shap.summary_plot(brca_mean_shap_values_spl.astype('float64').to_numpy(), gene_counts_brca, feature_names=gene_names, show=False, plot_size=(10, 5))
# plt.savefig("./plots/shap_plot_brca_spl.pdf", format="pdf", dpi=500)
# plt.clf()

# shap.summary_plot(brca_mean_shap_values_rl.astype('float64').to_numpy(), gene_counts_brca, feature_names=gene_names, show=False, plot_size=(10, 5))
# plt.savefig("./plots/shap_plot_brca_rl.pdf", format="pdf", dpi=500)
# plt.clf()

# shap.summary_plot(brca_mean_shap_values_srl.astype('float64').to_numpy(), gene_counts_brca, feature_names=gene_names, show=False, plot_size=(10, 5))
# plt.savefig("./plots/shap_plot_brca_srl.pdf", format="pdf", dpi=500)
# plt.clf()



# %% Add configs to summary files

# TODO: Insert configs into summary files
dir_path = './summary'
complete_runs_path = [os.path.join(dir_path, config_folder) for config_folder in os.listdir(dir_path) if "[" in config_folder]

sorted_run_paths = sorted(complete_runs_path, key=os.path.getmtime)
