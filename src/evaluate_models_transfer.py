# %% Import Modules
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
from sksurv.metrics import brier_score
from sklearn.model_selection import train_test_split
from lifelines import KaplanMeierFitter
import seaborn as sns
import matplotlib.ticker as ticker
from mpl_toolkits.axisartist.parasite_axes import SubplotHost
import json
from matplotlib.ticker import AutoMinorLocator
from modules.evaluation_metrics import stratified_brier_score

# Wheter to save the figures to file or not.
save_plots = True

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
TUMOR_TYPE_COMBINATION = sorted([x for x in sys.argv[1:]])

# TUMOR_TYPE_COMBINATION = ["BRCA", "GBM"]

tumor_types_list = TUMOR_TYPE_COMBINATION
TUMOR_TYPE_COMBINATION = "_".join(TUMOR_TYPE_COMBINATION)
TUMOR_TYPE_COMBINATION_STRING_REPR = TUMOR_TYPE_COMBINATION + "_scaled"
summary_root_path = os.path.join(
    "summaries_transfer_learning", TUMOR_TYPE_COMBINATION_STRING_REPR
)

save_path = "./plots_transfer_learning"
data_path = "./data/{}.pickle".format(TUMOR_TYPE_COMBINATION_STRING_REPR)
data_path = "./data/{}.csv".format(TUMOR_TYPE_COMBINATION_STRING_REPR)

if not os.path.exists(save_path):
    os.makedirs(save_path)


# %% ##################### Prediction Error Curves ###########################
##############################################################################
##############################################################################
##############################################################################
##############################################################################

MIN_EVAL_TIME_PEC = 20
MAX_EVAL_TIME_PEC = 1500

eval_times_brier_score = np.arange(MIN_EVAL_TIME_PEC, MAX_EVAL_TIME_PEC, 20)

# data = pd.read_pickle(data_path)
data = pd.read_csv(data_path)
data.index = data.patient_id

event_indicator = data.event.to_numpy(dtype=bool)
event_time = data.time.to_numpy(dtype=np.int16)
strata = data.tumor_type.to_numpy(dtype=str)

# Structured arrays for Brier Score Evaluation.
survival_data = np.zeros(
    event_indicator.shape[0],
    dtype={
        "names": ("event_indicator", "event_time"),
        "formats": ("bool", "u2"),
    },
)

survival_data["event_indicator"] = event_indicator
survival_data["event_time"] = event_time

mean_integrated_pec_pl_1 = []
mean_integrated_pec_spl_1 = []
mean_integrated_pec_rl_1 = []
mean_integrated_pec_srl_1 = []

mean_integrated_pec_pl_2 = []
mean_integrated_pec_spl_2 = []
mean_integrated_pec_rl_2 = []
mean_integrated_pec_srl_2 = []

network_depth = []
network_width = []
network_size = []


for dir in os.listdir(summary_root_path):
    dir_path = os.path.join(summary_root_path, dir)

    if len(os.listdir(dir_path)) == 0:
        continue

    integrated_pec_pl_1 = []
    integrated_pec_pl_2 = []
    integrated_pec_spl_1 = []
    integrated_pec_spl_2 = []
    integrated_pec_rl_1 = []
    integrated_pec_rl_2 = []
    integrated_pec_srl_1 = []
    integrated_pec_srl_2 = []

    with open(os.path.join(dir_path, "hyperparameters.json"), "r") as f_path:
        hyp_params = json.load(f_path)
        depth = len(hyp_params["layers"])
        width = max(hyp_params["layers"])
        size = sum(hyp_params["layers"])

        network_depth.append(depth)
        network_width.append(width)
        network_size.append(size)

    for f in os.listdir(dir_path):
        if "pec" in f:
            pec_split = pd.read_csv(os.path.join(dir_path, f))

            brier_eval_times = pec_split.brier_eval_times
            brier_scores_1 = pec_split.loc[:, tumor_types_list[0]]
            brier_scores_2 = pec_split.loc[:, tumor_types_list[1]]

            brier_eval_times = brier_eval_times
            brier_scores_1 = brier_scores_1
            brier_scores_2 = brier_scores_2

            integrated_pec_1 = np.trapz(x=brier_eval_times, y=brier_scores_1)
            integrated_pec_2 = np.trapz(x=brier_eval_times, y=brier_scores_2)

        if "pec_PL" in f:
            integrated_pec_pl_1.append(integrated_pec_1)
            integrated_pec_pl_2.append(integrated_pec_2)
        elif "pec_SPL" in f:
            integrated_pec_spl_1.append(integrated_pec_1)
            integrated_pec_spl_2.append(integrated_pec_2)
        elif "pec_RL" in f:
            integrated_pec_rl_1.append(integrated_pec_1)
            integrated_pec_rl_2.append(integrated_pec_2)
        elif "pec_SRL" in f:
            integrated_pec_srl_1.append(integrated_pec_1)
            integrated_pec_srl_2.append(integrated_pec_2)

    mean_integrated_pec_pl_1.append(np.mean(integrated_pec_pl_1))
    mean_integrated_pec_pl_2.append(np.mean(integrated_pec_pl_2))
    mean_integrated_pec_spl_1.append(np.mean(integrated_pec_spl_1))
    mean_integrated_pec_spl_2.append(np.mean(integrated_pec_spl_2))
    mean_integrated_pec_rl_1.append(np.mean(integrated_pec_rl_1))
    mean_integrated_pec_rl_2.append(np.mean(integrated_pec_rl_2))
    mean_integrated_pec_srl_1.append(np.mean(integrated_pec_srl_1))
    mean_integrated_pec_srl_2.append(np.mean(integrated_pec_srl_2))


integrated_pec_df_1 = pd.DataFrame(
    {
        "PL": mean_integrated_pec_pl_1,
        "SPL": mean_integrated_pec_spl_1,
        "RL": mean_integrated_pec_rl_1,
        "SRL": mean_integrated_pec_srl_1,
        "depth": network_depth,
        "size": network_size,
        "width": network_width,
    }
)

integrated_pec_df_2 = pd.DataFrame(
    {
        "PL": mean_integrated_pec_pl_2,
        "SPL": mean_integrated_pec_spl_2,
        "RL": mean_integrated_pec_rl_2,
        "SRL": mean_integrated_pec_srl_2,
        "depth": network_depth,
        "size": network_size,
        "width": network_width,
    }
)


# %% iPEC Comparison single entities

single_entity_pecs = []

for tumor_type in tumor_types_list:
    i = 0
    if len(tumor_types_list) > 2:
        continue

    mean_integrated_single_tumor_pl = []
    mean_integrated_single_tumor_rl = []

    print(tumor_type)
    single_tumor_path = "./summaries/" + tumor_type + "_scaled"

    if not os.path.exists(single_tumor_path):
        raise ValueError("Path does not exist")

    for dir in os.listdir(single_tumor_path):
        i += 1
        dir_path = os.path.join(single_tumor_path, dir)

        if len(os.listdir(dir_path)) == 0:
            raise ValueError("Path is empty.")

        integrated_pec_pl = []
        integrated_pec_rl = []

        for f in os.listdir(dir_path):
            if "pec_SPL" in f or "pec_SRL" in f:
                continue

            elif "pec" in f:

                pec_split = pd.read_csv(os.path.join(dir_path, f))

                brier_eval_times = pec_split.brier_eval_times
                brier_scores = pec_split.brier_scores

                integrated_pec = np.trapz(x=brier_eval_times, y=brier_scores)

            if "pec_PL" in f:
                integrated_pec_pl.append(integrated_pec)

            elif "pec_RL" in f:
                integrated_pec_rl.append(integrated_pec)

        mean_integrated_single_tumor_pl.append(np.mean(integrated_pec_pl))
        mean_integrated_single_tumor_rl.append(np.mean(integrated_pec_rl))

    ipec_df_single_entity = pd.DataFrame(
        {
            "PL_{}".format(tumor_type): mean_integrated_single_tumor_pl,
            "RL_{}".format(tumor_type): mean_integrated_single_tumor_rl,
        }
    )

    single_entity_pecs.append(ipec_df_single_entity)

single_entity = pd.concat(single_entity_pecs, axis=1)


entities_df = [
    pd.concat([integrated_pec_df_1, single_entity_pecs[0]], axis=1),
    pd.concat([integrated_pec_df_2, single_entity_pecs[1]], axis=1),
]

# %% Calculate iPEC for Ridge Regression

# entity_combinations = [
#     ["BRCA", "GBM"],
#     ["BRCA", "KIRC"],
#     ["BRCA", "LGG"],
#     ["GBM", "LGG"],
#     ["GBM", "KIRC"],
# ]

mean_entity_ridge_ipecs = dict({
    tumor_types_list[0]: {
        "strat_augmented": -1,
        "single": -1,
    },
    tumor_types_list[1]: {
        "strat_augmented": -1,
        "single": -1,
    }
})

# for tumor_types_list in entity_combinations:
#     print("_____________________________________________")
#     print("Evaluating tumor combination {}".format("+".join(tumor_types_list)))

tumor_combination = "_".join(tumor_types_list)

for tumor_type in tumor_types_list:
    ridge_ipecs = []
    strat_ridge_ipecs = []
    single_ipecs = []

    for run in range(1, 11):
        for fold in range(1, 6):

            ridge_stratified = pd.read_csv(
                "baseline_regressions/210816_{}_tcga_linear_predictors_log_std_ridge_{}_{}.csv".format(tumor_combination, run, fold))

            single = pd.read_csv(
                "baseline_regressions/210816_{}_tcga_linear_predictors_log_std_ridge_{}_{}.csv".format(tumor_type, run, fold))

            # single_2 = pd.read_csv(
            #     "src/baseline_regressions/210811_{}_tcga_linear_predictors_log_std_ridge_{}.csv".format(tumor_types_list[1], fold))

            # ridge_train = ridge_stratified.loc[ridge_stratified.testtraining == 0]
            # ridge_test = ridge_stratified.loc[ridge_stratified.testtraining == 1]

            ridge_train = ridge_stratified.loc[(ridge_stratified.testtraining == 0) & (
                ridge_stratified.tumor_type == tumor_type)]
            ridge_test = ridge_stratified.loc[(ridge_stratified.testtraining == 1) & (
                ridge_stratified.tumor_type == tumor_type)]

            single_train = single.loc[single.testtraining == 0]
            single_test = single.loc[single.testtraining == 1]

            # single_2_train = single_2.loc[single_2.testtraining == 0]
            # single_2_test = single_2.loc[single_2.testtraining == 1]

            # Structured arrays for Brier Score Evaluation.
            survival_data_train = np.zeros(
                ridge_train.shape[0],
                dtype={
                    "names": ("event_indicator", "event_time"),
                    "formats": ("bool", "u2"),
                },
            )

            survival_data_test = np.zeros(
                ridge_test.shape[0],
                dtype={
                    "names": ("event_indicator", "event_time"),
                    "formats": ("bool", "u2"),
                },
            )

            survival_data_single_train = np.zeros(
                single_train.shape[0],
                dtype={
                    "names": ("event_indicator", "event_time"),
                    "formats": ("bool", "u2"),
                },
            )

            survival_data_single_test = np.zeros(
                single_test.shape[0],
                dtype={
                    "names": ("event_indicator", "event_time"),
                    "formats": ("bool", "u2"),
                },
            )

            # Combined Data.
            survival_data_train["event_indicator"] = ridge_train.event.to_numpy(
                dtype=bool)
            survival_data_train["event_time"] = ridge_train.time.to_numpy(
                dtype=np.int16)

            survival_data_test["event_indicator"] = ridge_test.event.to_numpy(
                dtype=bool)
            survival_data_test["event_time"] = ridge_test.time.to_numpy(
                dtype=np.int16)

            # Single Data.

            survival_data_single_train["event_indicator"] = single_train.event.to_numpy(
                dtype=bool)
            survival_data_single_train["event_time"] = single_train.time.to_numpy(
                dtype=np.int16)

            survival_data_single_test["event_indicator"] = single_test.event.to_numpy(
                dtype=bool)
            survival_data_single_test["event_time"] = single_test.time.to_numpy(
                dtype=np.int16)

            ridge_linear_predictor_train = ridge_train.linpred_nonstrat.to_numpy()
            ridge_linear_predictor_test = ridge_test.linpred_nonstrat.to_numpy()

            ridge_linear_predictor_strat_train = ridge_train.linpred_strat.to_numpy()
            ridge_linear_predictor_strat_test = ridge_test.linpred_strat.to_numpy()

            single_linear_predictor_train = single_train.linpred_nonstrat.to_numpy()
            single_linear_predictor_test = single_test.linpred_nonstrat.to_numpy()

            strata_train = ridge_train.tumor_type.to_numpy(dtype=str)
            strata_test = ridge_test.tumor_type.to_numpy(dtype=str)

            strata_single_train = single_train.tumor_type.to_numpy(
                dtype=str)
            strata_single_test = single_test.tumor_type.to_numpy(dtype=str)

            ridge_eval_times, ridge_brier_scores = stratified_brier_score(
                1500,
                survival_data_train,
                survival_data_test,
                ridge_linear_predictor_train,
                ridge_linear_predictor_test,
                strata_train=strata_train,
                strata_test=strata_test,
                stratified_fitted=False,
                minimum_brier_eval_time=20,
            )

            ridge_strat_eval_times, ridge_strat_brier_scores = stratified_brier_score(
                1500,
                survival_data_train,
                survival_data_test,
                ridge_linear_predictor_strat_train,
                ridge_linear_predictor_strat_test,
                strata_train=strata_train,
                strata_test=strata_test,
                stratified_fitted=True,
                minimum_brier_eval_time=20,
            )

            single_eval_times, single_brier_scores = stratified_brier_score(
                1500,
                survival_data_single_train,
                survival_data_single_test,
                single_linear_predictor_train,
                single_linear_predictor_test,
                strata_train=strata_single_train,
                strata_test=strata_single_test,
                stratified_fitted=False,
                minimum_brier_eval_time=20,
            )

            integrated_pec_ridge = np.trapz(
                x=ridge_eval_times, y=ridge_brier_scores)

            integrated_pec_ridge_strat = np.trapz(
                x=ridge_strat_eval_times, y=ridge_strat_brier_scores)

            integrated_pec_single = np.trapz(
                x=single_eval_times, y=single_brier_scores)

            ridge_ipecs.append(integrated_pec_ridge)
            strat_ridge_ipecs.append(integrated_pec_ridge_strat)
            single_ipecs.append(integrated_pec_single)

            # print("Integrated values: ")
            # print(integrated_pec_ridge)
            # print(integrated_pec_ridge_strat)
            # print(integrated_pec_single)
            # print(integrated_pec_single_2)

    mean_ridge_ipec = float(np.mean(ridge_ipecs))
    mean_strat_ridge_ipec = float(np.mean(strat_ridge_ipecs))
    mean_ipec_single = float(np.mean(single_ipecs))

    mean_entity_ridge_ipecs[tumor_type]["strat_augmented"] = mean_strat_ridge_ipec
    mean_entity_ridge_ipecs[tumor_type]["single"] = mean_ipec_single

    print()
    print("Mean Values {}: ".format(tumor_type))
    print("Non-Stratified Fitted combined data", mean_ridge_ipec)
    print("Stratified Fitted combined data", mean_strat_ridge_ipec)
    print("Non-Stratified Fitted single data", mean_ipec_single)
    # print(mean_ipec_single_2)


# %%

fig = plt.figure(figsize=(15, 5))

i = 1
for entity, tumor_type in zip(entities_df, tumor_types_list):

    ax = SubplotHost(fig, 1, 2, i)
    fig.add_subplot(ax)

    i += 1

    box_plot_df = entity.melt(
        id_vars=["depth", "size", "width"], var_name="Model", value_name="iPEC"
    )

    sns.boxplot(
        x="Model",
        y="iPEC",
        data=box_plot_df,
        ax=ax,
        color="darkgreen",
        width=0.4,
    )

    ax.set_xlabel("", labelpad=20)

    ax.set_xticklabels(
        [
            "PL",
            "SPL",
            "RL",
            "SRL",
            "PL",
            "RL",
        ]
    )

    ax.set_title("Evaluated on " + tumor_type)

    ax_twiny = ax.twiny()
    offset = 0, -50  # Position of the second axis
    new_axisline = ax_twiny.get_grid_helper().new_fixed_axis
    ax_twiny.axis["bottom"] = new_axisline(
        loc="bottom", axes=ax_twiny, offset=offset)
    ax_twiny.axis["top"].set_visible(False)

    ax_twiny.axis["bottom"].minor_ticks.set_ticksize(3)
    ax_twiny.axis["bottom"].major_ticks.set_ticksize(10)
    ax_twiny.set_xticks([0.0, 0.66, 1.0])
    ax_twiny.xaxis.set_major_formatter(ticker.NullFormatter())
    ax_twiny.xaxis.set_minor_locator(ticker.FixedLocator([0.33, 0.82]))
    ax_twiny.xaxis.set_minor_formatter(
        ticker.FixedFormatter(
            [
                "Fitted on {}+{}".format(tumor_types_list[0],
                                         tumor_types_list[1]),
                "Fitted on {}".format(tumor_type),
            ]
        )
    )

    ax.grid(
        b=True,
        zorder=0,
    )

tick_label_size = 8

fig.axes[0].tick_params(axis="x", which="major", labelsize=tick_label_size)
fig.axes[1].tick_params(axis="x", which="major", labelsize=tick_label_size)
fig.axes[0].tick_params(axis="y", which="major", labelsize=tick_label_size)
fig.axes[1].tick_params(axis="y", which="major", labelsize=tick_label_size)
fig.axes[0].xaxis.set_tick_params(which="minor", bottom=False)
fig.axes[0].set_ylabel("iPEC", labelpad=10, size=tick_label_size + 2)
fig.axes[1].set_ylabel("")

linewidth = 0.8
fig.axes[0].axhline(y=mean_entity_ridge_ipecs[tumor_types_list[0]]["strat_augmented"],
                    xmin=0, xmax=0.65, c="#1a75ff", linewidth=linewidth, zorder=0)
fig.axes[0].axhline(y=mean_entity_ridge_ipecs[tumor_types_list[0]]["single"],
                    xmin=0.67, xmax=1, c="#ff9900", linewidth=linewidth, zorder=0)
fig.axes[1].axhline(y=mean_entity_ridge_ipecs[tumor_types_list[1]]["strat_augmented"],
                    xmin=0, xmax=0.65, c="#1a75ff", linewidth=linewidth, zorder=0)
fig.axes[1].axhline(y=mean_entity_ridge_ipecs[tumor_types_list[1]]["single"],
                    xmin=0.67, xmax=1, c="#ff9900", linewidth=linewidth, zorder=0)

fig.subplots_adjust(wspace=0.3, left=0.15)

plt.tight_layout()

if save_plots:
    plt.savefig(
        os.path.join(
            save_path,
            TUMOR_TYPE_COMBINATION_STRING_REPR + "_transfer_learning_boxplot.pdf",
        ),
        format="pdf",
        dpi=500,
    )

plt.show()
