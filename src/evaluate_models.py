# %% Import Modules
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
from sksurv.metrics import brier_score
from lifelines import KaplanMeierFitter
import seaborn as sns
from modules.evaluation_metrics import stratified_brier_score
import json
from matplotlib.ticker import AutoMinorLocator
from sksurv.nonparametric import CensoringDistributionEstimator
from sklearn.model_selection import train_test_split

np.random.seed(42)

# Wheter to save the figures to file or not.
save_plots = True

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
TUMOR_TYPE_COMBINATION = sorted([x for x in sys.argv[1:]])

tumor_types_list = TUMOR_TYPE_COMBINATION
TUMOR_TYPE_COMBINATION = "_".join(TUMOR_TYPE_COMBINATION)
TUMOR_TYPE_COMBINATION_STRING_REPR = TUMOR_TYPE_COMBINATION + "_scaled"
summary_root_path = os.path.join("summaries", TUMOR_TYPE_COMBINATION_STRING_REPR)

save_path = "./plots/{}".format(TUMOR_TYPE_COMBINATION_STRING_REPR)
data_path = "./data/{}.pickle".format(TUMOR_TYPE_COMBINATION_STRING_REPR)

if TUMOR_TYPE_COMBINATION == "_".join(
    sorted(["BRCA", "GBM", "KIRC", "LGG", "KICH", "KIRP"])
):
    lasso_path = "./baseline_regressions/BRCA_GLIOMA_KIPAN_tcga_linear_predictors_log_std.csv"
    ridge_path = "./baseline_regressions/BRCA_GLIOMA_KIPAN_tcga_linear_predictors_log_std_ridge.csv"

if not os.path.exists(save_path):
    os.makedirs(save_path)

print(TUMOR_TYPE_COMBINATION_STRING_REPR)
# %% ################ Concordance Index Evaluation ###########################
mean_c_frames = []
run_no = 0
tick_label_size = 8

for dir in os.listdir(summary_root_path):
    dir_path = os.path.join(summary_root_path, dir)

    df = pd.read_csv(os.path.join(dir_path, "concordance_index.csv"))
    with open(os.path.join(dir_path, "hyperparameters.json"), "r") as f_path:
        hyp_params = json.load(f_path)
        depth = len(hyp_params["layers"])
        width = max(hyp_params["layers"])
        size = sum(hyp_params["layers"])

    df.insert(0, "run_no", run_no)
    df.insert(0, "depth", depth)
    df.insert(0, "width", width)
    df.insert(0, "size", size)

    mean_c_frames.append(df)

    run_no += 1

df_c = pd.concat(mean_c_frames, axis=0)

df_c_runs = (
    df_c.groupby(["run_no", "size", "depth", "width"])
    .agg(
        {
            "PL": ["mean"],
            "SPL": ["mean"],
            "RL": ["mean"],
            "SRL": ["mean"],
        }
    )
    .reset_index()
)

difference_col_name = ["SPL - PL", "SRL - RL"]

df_c_runs.insert(0, difference_col_name[0], df_c_runs.SPL - df_c_runs.PL)
df_c_runs.insert(0, difference_col_name[1], df_c_runs.SRL - df_c_runs.RL)

df_c_differences = df_c_runs.loc[:, ["run_no", "size", "depth"] + difference_col_name]

col = "size"
labels = ["20 - 200", "200 - 400", "400 - 2000"]
min_size = 20
min_range = 100
max_range = 1400
step = 100
labels = ["{} - 100".format(min_size)] + [
    "{0} - {1}".format(i, i + step) for i in range(min_range, max_range, step)
]
ranges = [min_size] + list(range(min_range, max_range + 5, step))

full_names = [
    "Partial Likelihood",
    "Stratified Partial Likelihood",
    "Ranking Loss",
    "Stratified Ranking Loss",
]


df_c_differences["size"] = pd.cut(
    df_c_differences["size"], ranges, right=False, labels=labels
)

df_c_differences = df_c_differences.drop(columns="depth")

df_c_runs_melt = df_c_differences.melt(
    id_vars=["run_no", "size"], var_name="model", value_name="c_index"
)

sns.set_style("darkgrid")

fig, ax = plt.subplots(figsize=(10, 5))

sns.boxplot(
    x="size",
    y="c_index",
    data=df_c_runs_melt,
    ax=ax,
    hue="model",
    palette="colorblind",
    width=0.6,
)

ax.set_ylabel("\u0394 C-Index", labelpad=10)
ax.set_xlabel("Network size", labelpad=10, size=tick_label_size + 2)

ax.tick_params(axis="x", labelsize=tick_label_size, rotation=60)

handles, _ = ax.get_legend_handles_labels()
ax.legend(handles, ["SPL - PL", "SRL - RL"])

plt.tight_layout()

if save_plots:
    plt.savefig(
        os.path.join(
            save_path,
            TUMOR_TYPE_COMBINATION_STRING_REPR + "_concordance_differences_boxplot.pdf",
        ),
        format="pdf",
        dpi=500,
    )

plt.show()

# %% Absolute Concordance based on architecture

model_names = ["PL", "SPL", "RL", "SRL"]

col = "size"
labels = ["20 - 200", "200 - 400", "400 - 2000"]
min_size = 20
min_range = 100
max_range = 1400
step = 100
labels = ["{} - 100".format(min_size)] + [
    "{0} - {1}".format(i, i + step) for i in range(min_range, max_range, step)
]
ranges = [min_size] + list(range(min_range, max_range + 5, step))

full_names = [
    "Partial Likelihood",
    "Stratified Partial Likelihood",
    "Ranking Loss",
    "Stratified Ranking Loss",
]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
    nrows=2, ncols=2, figsize=(15, 10), sharey=True, sharex=True
)

for ax, model, full_name in zip([ax1, ax2, ax3, ax4], model_names, full_names):
    i = 1

    print(model)

    i += 1

    df_c_nn = df_c_runs.loc[:, ["run_no", col, model]]

    df_c_nn[col] = pd.cut(df_c_nn[col], ranges, right=False, labels=labels)

    df_c_melt = df_c_nn.melt(
        id_vars=["run_no", col], var_name="model", value_name="c_index"
    )

    sns.boxplot(
        x=col,
        y="c_index",
        data=df_c_melt,
        ax=ax,
        hue="model",
        palette="colorblind",
        width=0.2,
    )

    ax.set_title(full_name)
    ax.get_legend().remove()

ax1.tick_params(axis="x")
ax1.tick_params(axis="y", labelsize=tick_label_size)
ax1.tick_params(axis="y")
ax2.tick_params(axis="x")
ax2.tick_params(axis="y")
ax3.tick_params(axis="x", labelsize=tick_label_size, rotation=60)
ax3.tick_params(axis="y", labelsize=tick_label_size)
ax4.tick_params(axis="x", labelsize=tick_label_size, rotation=60)
ax4.tick_params(axis="y")

ax1.set_xlabel("")
ax2.set_xlabel("")
ax2.set_ylabel("")
ax4.set_ylabel("")
ax1.set_ylabel("C-Index", labelpad=10, size=tick_label_size + 2)
ax3.set_ylabel("C-Index", labelpad=10, size=tick_label_size + 2)
ax3.set_xlabel("Network size", labelpad=10, size=tick_label_size + 2)
ax4.set_xlabel("Network size", labelpad=10, size=tick_label_size + 2)

fig.subplots_adjust(left=0.1, right=1.9, wspace=0.5, hspace=0.8)

plt.tight_layout()

if save_plots:
    plt.savefig(
        os.path.join(
            save_path,
            TUMOR_TYPE_COMBINATION_STRING_REPR
            + "_concordance_absolute_values_boxplot.pdf",
        ),
        format="pdf",
        dpi=500,
    )

plt.show()


##############################################################################
##############################################################################
# %% ##################### Prediction Error Curves ###########################
##############################################################################
##############################################################################


MIN_EVAL_TIME_PEC = 20
MAX_EVAL_TIME_PEC = 1500

sns.set_style("darkgrid")

eval_times_brier_score = np.arange(MIN_EVAL_TIME_PEC, MAX_EVAL_TIME_PEC, 20)

try:
    lasso_regression = pd.read_csv(lasso_path)
    lasso_regression = lasso_regression.rename(
        columns={
            "linpred_strat": "lasso_linpred_strat",
            "linpred_nonstrat": "lasso_linpred_nonstrat",
        },
    )

    ridge_regression = pd.read_csv(ridge_path)
    ridge_regression = ridge_regression.rename(
        columns={
            "linpred_strat": "ridge_linpred_strat",
            "linpred_nonstrat": "ridge_linpred_nonstrat",
        }
    )

    data = pd.merge(
        ridge_regression,
        lasso_regression,
        on=["patient_id", "time", "event", "stratum"],
    )

    event_indicator = data.event.to_numpy(dtype=bool)
    event_time = data.time.to_numpy(dtype=np.int16)
    strata = data.stratum.to_numpy(dtype=str)
    lasso_linear_predictor_strat = data.lasso_linpred_strat
    ridge_linear_predictor_strat = data.ridge_linpred_strat

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

    (
        survival_data_train,
        survival_data_test,
        lasso_linear_predictor_strat_train,
        lasso_linear_predictor_strat_test,
        ridge_linear_predictor_strat_train,
        ridge_linear_predictor_strat_test,
        strata_train,
        strata_test,
    ) = train_test_split(
        survival_data,
        lasso_linear_predictor_strat,
        ridge_linear_predictor_strat,
        strata,
        test_size=0.2,
        train_size=0.8,
        random_state=42,
        shuffle=True,
        stratify=strata,
    )

    event_time_train = survival_data_train["event_time"]
    event_indicator_train = survival_data_train["event_indicator"]
    event_time_test = survival_data_test["event_time"]
    event_indicator_test = survival_data_test["event_indicator"]

    # Calculating the brier scores at each time point
    kaplan_brier_scores = []
    kaplan_group_sizes = []

    lasso_brier_scores = []
    ridge_brier_scores = []

    kmf = KaplanMeierFitter()
    kmf.fit(
        event_time_train,
        event_observed=event_indicator_train,
        timeline=eval_times_brier_score,
    )

    for s in np.unique(strata):

        strata_train_dat = survival_data_train[strata_train == s]
        strata_test_dat = survival_data_test[strata_test == s]

        kaplan_preds = np.repeat(
            [kmf.predict(eval_times_brier_score).to_numpy()],
            strata_test_dat.shape[0],
            axis=0,
        )

        times, km_score = brier_score(
            survival_train=strata_train_dat,
            survival_test=strata_test_dat,
            estimate=kaplan_preds,
            times=eval_times_brier_score,
        )

        kaplan_brier_scores.append(km_score)
        kaplan_group_sizes.append(strata_test_dat.shape[0])

    kmf_brier_scores = np.average(
        np.stack(kaplan_brier_scores), weights=kaplan_group_sizes, axis=0
    )

    lasso_strat_eval_times, lasso_strat_brier_scores = stratified_brier_score(
        MAX_EVAL_TIME_PEC,
        survival_data_train,
        survival_data_test,
        lasso_linear_predictor_strat_train.to_numpy(),
        lasso_linear_predictor_strat_test.to_numpy(),
        strata_train=strata_train,
        strata_test=strata_test,
        stratified_fitted=True,
        minimum_brier_eval_time=20,
    )

    ridge_strat_eval_times, ridge_strat_brier_scores = stratified_brier_score(
        MAX_EVAL_TIME_PEC,
        survival_data_train,
        survival_data_test,
        ridge_linear_predictor_strat_train.to_numpy(),
        ridge_linear_predictor_strat_test.to_numpy(),
        strata_train=strata_train,
        strata_test=strata_test,
        stratified_fitted=True,
        minimum_brier_eval_time=20,
    )
except:
    pass

raw_integrated_pec_pl = []
raw_integrated_pec_spl = []
raw_integrated_pec_rl = []
raw_integrated_pec_srl = []

network_depth = []
network_width = []
network_size = []

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
    2, 2, sharey=True, sharex=True, figsize=(10, 5)
)


for dir in os.listdir(summary_root_path):
    dir_path = os.path.join(summary_root_path, dir)

    if len(os.listdir(dir_path)) == 0:
        continue

    integrated_pec_pl = []
    integrated_pec_spl = []
    integrated_pec_rl = []
    integrated_pec_srl = []

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
            brier_scores = pec_split.brier_scores

            mask_eval_times = (MIN_EVAL_TIME_PEC < brier_eval_times) & (
                brier_eval_times < MAX_EVAL_TIME_PEC
            )

            brier_eval_times = brier_eval_times[mask_eval_times]
            brier_scores = brier_scores[mask_eval_times]

            integrated_pec = np.trapz(x=brier_eval_times, y=brier_scores)

        if "pec_PL" in f:
            integrated_pec_pl.append(integrated_pec)
            ax1.plot(
                brier_eval_times,
                brier_scores,
                color="#C0C0C0",
                linewidth=0.2,
            )
        elif "pec_SPL" in f:
            integrated_pec_spl.append(integrated_pec)
            ax2.plot(
                brier_eval_times,
                brier_scores,
                color="#C0C0C0",
                linewidth=0.2,
            )

        elif "pec_RL" in f:
            integrated_pec_rl.append(integrated_pec)
            ax3.plot(
                brier_eval_times,
                brier_scores,
                color="#C0C0C0",
                linewidth=0.2,
            )

        elif "pec_SRL" in f:
            integrated_pec_srl.append(integrated_pec)
            ax4.plot(
                brier_eval_times,
                brier_scores,
                color="#C0C0C0",
                linewidth=0.2,
            )

    raw_integrated_pec_pl.append(integrated_pec_pl)
    raw_integrated_pec_spl.append(integrated_pec_spl)
    raw_integrated_pec_rl.append(integrated_pec_rl)
    raw_integrated_pec_srl.append(integrated_pec_srl)


for ax, loss_func in zip(
    [ax1, ax2, ax3, ax4],
    [
        "Partial Likelihood",
        "Stratified Partial Likelihood",
        "Ranking Loss",
        "Stratified Ranking Loss",
    ],
):

    ax.set_title(loss_func, size=8)
    try:
        ax.plot(
            lasso_strat_eval_times,
            lasso_strat_brier_scores,
            label="Lasso",
            color="blue",
        )
        ax.plot(
            ridge_strat_eval_times,
            ridge_strat_brier_scores,
            label="Ridge",
            color="green",
        )
        ax.plot(
            eval_times_brier_score,
            kmf_brier_scores,
            color="black",
            label="Kaplan-Meier",
        )
    except:
        pass

tick_label_size = 8
ax1.tick_params(axis="x")
ax1.tick_params(axis="y", labelsize=tick_label_size)
ax1.tick_params(axis="y")
ax2.tick_params(axis="x")
ax2.tick_params(axis="y")
ax3.tick_params(axis="x", labelsize=tick_label_size)
ax3.tick_params(axis="y", labelsize=tick_label_size)
ax4.tick_params(axis="x", labelsize=tick_label_size)
ax4.tick_params(axis="y")

ax1.set_xlabel("")
ax2.set_xlabel("")
ax2.set_ylabel("")
ax4.set_ylabel("")
ax1.set_ylabel("Prediction Error", labelpad=10, size=tick_label_size + 2)
ax3.set_ylabel("Prediction Error", labelpad=10, size=tick_label_size + 2)
ax3.set_xlabel("Time", labelpad=10, size=tick_label_size + 2)
ax4.set_xlabel("Time", labelpad=10, size=tick_label_size + 2)

fig.subplots_adjust(left=0.1, right=1.9, wspace=0.5, hspace=0.8)

ax2.legend(
    loc=(0.05, 0.8),
    framealpha=0.9,
    borderaxespad=0.25,
    facecolor="white",
    fontsize="xx-small",
)

plt.tight_layout()

if save_plots:
    plt.savefig(
        os.path.join(
            save_path,
            TUMOR_TYPE_COMBINATION_STRING_REPR + "_prediction_error_curves.pdf",
        ),
        format="pdf",
        dpi=500,
    )


plt.show()

integrated_pec_df_raw = pd.DataFrame(
    {
        "PL": np.array(raw_integrated_pec_pl).flatten(),
        "SPL": np.array(raw_integrated_pec_spl).flatten(),
        "RL": np.array(raw_integrated_pec_rl).flatten(),
        "SRL": np.array(raw_integrated_pec_srl).flatten(),
        "depth": np.repeat(network_depth, 5),
        "size": np.repeat(network_size, 5),
        "width": np.repeat(network_width, 5),
    }
)


# %% iPEC Absolute values over network size

model_names = ["PL", "SPL", "RL", "SRL"]

col = "size"
min_size = 20
min_range = 100
max_range = 1400
step = 100
labels = ["{} - 100".format(min_size)] + [
    "{0} - {1}".format(i, i + step) for i in range(min_range, max_range, step)
]
ranges = [min_size] + list(range(min_range, max_range + 5, step))

full_names = [
    "Partial Likelihood",
    "Stratified Partial Likelihood",
    "Ranking Loss",
    "Stratified Ranking Loss",
]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
    nrows=2, ncols=2, figsize=(15, 10), sharey=True, sharex=True
)


df = integrated_pec_df_raw.copy()

for ax, model, full_name in zip([ax1, ax2, ax3, ax4], model_names, full_names):
    i = 1

    print(model)
    i += 1

    df_pec_nn = df.loc[:, [col, model]]

    df_pec_nn[col] = pd.cut(df_pec_nn[col], ranges, right=False, labels=labels)

    df_pec_melt = df_pec_nn.melt(id_vars=[col], var_name="model", value_name="c_index")

    sns.boxplot(
        x=col,
        y="c_index",
        data=df_pec_melt,
        ax=ax,
        hue="model",
        palette="colorblind",
        width=0.2,
    )

    ax.set_title(full_name)
    ax.get_legend().remove()

tick_label_size = 8
ax1.tick_params(axis="x")
ax1.tick_params(axis="y", labelsize=tick_label_size)
ax1.tick_params(axis="y")
ax2.tick_params(axis="x")
ax2.tick_params(axis="y")
ax3.tick_params(axis="x", labelsize=tick_label_size, rotation=60)
ax3.tick_params(axis="y", labelsize=tick_label_size)
ax4.tick_params(axis="x", labelsize=tick_label_size, rotation=60)
ax4.tick_params(axis="y")

ax1.set_xlabel("")
ax2.set_xlabel("")
ax2.set_ylabel("")
ax4.set_ylabel("")
ax1.set_ylabel("iPEC", labelpad=10, size=tick_label_size + 2)
ax3.set_ylabel("iPEC", labelpad=10, size=tick_label_size + 2)
ax3.set_xlabel("Network size", labelpad=10, size=tick_label_size + 2)
ax4.set_xlabel("Network size", labelpad=10, size=tick_label_size + 2)

fig.subplots_adjust(left=0.1, right=1.9, wspace=0.5, hspace=0.8)

plt.tight_layout()

if save_plots:
    plt.savefig(
        os.path.join(
            save_path,
            TUMOR_TYPE_COMBINATION_STRING_REPR
            + "_ipec_absolute_values_boxplot_strat_eval.pdf",
        ),
        format="pdf",
        dpi=500,
    )

plt.show()


# %% Mean and Median iPEC values

print("Median iPEC", integrated_pec_df_raw.median())
try:
    lasso_ipec = np.trapz(x=lasso_strat_eval_times, y=lasso_strat_brier_scores)
    ridge_ipec = np.trapz(x=ridge_strat_eval_times, y=ridge_strat_brier_scores)
    kaplan_ipec = np.trapz(x=brier_eval_times, y=kmf_brier_scores)

    print()
    print("Lasso iPEC: ", lasso_ipec)
    print("Ridge iPEC: ", ridge_ipec)
    print("Kaplan-Meier iPEC: ", kaplan_ipec)
except:
    pass

# %% iPEC Difference Plot
print(TUMOR_TYPE_COMBINATION)

sns.set(style="darkgrid")

df = integrated_pec_df_raw.copy()

difference_col_name = ["SPL - PL", "SRL - RL"]

df.insert(0, difference_col_name[0], df.SPL - df.PL)
df.insert(0, difference_col_name[1], df.SRL - df.RL)

col = "size"
labels = ["20 - 200", "200 - 400", "400 - 2000"]
min_size = 20
min_range = 100
max_range = 1400
step = 100
labels = ["{} - 100".format(min_size)] + [
    "{0} - {1}".format(i, i + step) for i in range(min_range, max_range, step)
]
ranges = [min_size] + list(range(min_range, max_range + 5, step))

full_names = [
    "Partial Likelihood",
    "Stratified Partial Likelihood",
    "Ranking Loss",
    "Stratified Ranking Loss",
]


df["size"] = pd.cut(df["size"], ranges, right=False, labels=labels)


df = df.loc[:, ["SRL - RL", "SPL - PL", "size"]]

df_melt = df.loc[:, ["size"] + difference_col_name]

df_melt = df_melt.melt(id_vars=["size"], var_name="model", value_name="ipec")

fig, ax = plt.subplots(figsize=(12, 8))

sns.boxplot(
    x="size",
    y="ipec",
    data=df_melt,
    ax=ax,
    hue="model",
    palette="colorblind",
    width=0.6,
)

ax.set_xlabel("Network size", labelpad=10)
ax.set_ylabel("\u0394 iPEC", labelpad=10)

handles, _ = ax.get_legend_handles_labels()
ax.legend(handles, ["SPL - PL", "SRL - RL"])

plt.tight_layout()

ax.tick_params(axis="x", labelsize=tick_label_size, rotation=60)

if save_plots:
    plt.savefig(
        os.path.join(
            save_path,
            TUMOR_TYPE_COMBINATION_STRING_REPR + "_ipec_differences_boxplot.pdf",
        ),
        format="pdf",
        dpi=500,
    )

plt.show()

# %% iPEC Comparison single entities

sns.set(style="darkgrid")

single_entity_pecs = []

df = integrated_pec_df.copy()

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
        print(i)
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

    df = pd.concat([df, ipec_df_single_entity], axis=1)

single_entity = pd.concat(single_entity_pecs, axis=1)

cols_dict = {
    x: "{}_{}".format(x, TUMOR_TYPE_COMBINATION) for x in ["PL", "SPL", "RL", "SRL"]
}
df.rename(cols_dict, inplace=True, axis="columns")

fig, ax = plt.subplots(figsize=(8, 5))

box_plot_df = df.melt(
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
ax.set_ylabel("iPEC", labelpad=10)

tumor_type_sum = tumor_types_list[0] + "+" + tumor_types_list[1]

ax.set_xticklabels(
    [
        "PL: " + tumor_type_sum,
        "SPL: " + tumor_type_sum,
        "RL: " + tumor_type_sum,
        "SRL: " + tumor_type_sum,
        "PL: " + tumor_types_list[0],
        "RL: " + tumor_types_list[0],
        "PL: " + tumor_types_list[1],
        "RL: " + tumor_types_list[1],
    ]
)

ax.tick_params(axis="x", which="major", labelrotation=60)
ax.tick_params(axis="y")

plt.tight_layout()

if save_plots:
    plt.savefig(
        os.path.join(
            save_path,
            TUMOR_TYPE_COMBINATION_STRING_REPR + "_ipec_baseline_boxplot.pdf",
        ),
        format="pdf",
        dpi=500,
    )

plt.show()

# Differences in median iPEC
median_ipec_difference_pl = df.median()[0] - df.median()[1]
median_ipec_difference_rl = df.median()[2] - df.median()[3]

print("Median Differnece strat/non-strat Pl/SPL: ", median_ipec_difference_pl)
print("Median Differnece strat/non-strat Rl/SRL: ", median_ipec_difference_rl)

median_ipec_difference_pl = df.median()[0] - df.median()[1]
median_ipec_difference_rl = df.median()[2] - df.median()[3]

print("Median Differnece Pl/SPL: ", median_ipec_difference_pl)
print("Median Differnece Rl/SRL: ", median_ipec_difference_rl)