# %% Import Modules
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
from sksurv.metrics import brier_score
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
tumor_types_list = TUMOR_TYPE_COMBINATION
TUMOR_TYPE_COMBINATION = "_".join(TUMOR_TYPE_COMBINATION)
TUMOR_TYPE_COMBINATION_STRING_REPR = TUMOR_TYPE_COMBINATION + "_scaled"
summary_root_path = os.path.join(
    "summaries_transfer_learning", TUMOR_TYPE_COMBINATION_STRING_REPR
)

save_path = "./plots_transfer_learning"
data_path = "./data/{}.pickle".format(TUMOR_TYPE_COMBINATION_STRING_REPR)

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

data = pd.read_pickle(data_path)
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
    ax_twiny.axis["bottom"] = new_axisline(loc="bottom", axes=ax_twiny, offset=offset)
    ax_twiny.axis["top"].set_visible(False)

    ax_twiny.axis["bottom"].minor_ticks.set_ticksize(3)
    ax_twiny.axis["bottom"].major_ticks.set_ticksize(10)
    ax_twiny.set_xticks([0.0, 0.66, 1.0])
    ax_twiny.xaxis.set_major_formatter(ticker.NullFormatter())
    ax_twiny.xaxis.set_minor_locator(ticker.FixedLocator([0.33, 0.82]))
    ax_twiny.xaxis.set_minor_formatter(
        ticker.FixedFormatter(
            [
                "Fitted on {}+{}".format(tumor_types_list[0], tumor_types_list[1]),
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
fig.subplots_adjust(wspace=0.4)

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
