# %% Import modules.
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import collections
import os
import shap
import json
import sys

from modules.evaluation_metrics import (
    stratified_concordance_index,
    stratified_brier_score,
)
from modules.torch_models import (
    BaseFeedForwardNet,
    StratifiedPartialLikelihoodLoss,
    StratifiedRankingLoss,
)

np.random.seed(12)

# %% Hyperparameters.
with open("./optimization_configs.json", "r") as f:
    hyperparameters = json.load(f)

SAVE_MODELS = hyperparameters["SAVE_MODELS"]
SHAP_EVALUATION = hyperparameters["SHAP_EVALUATION"]
MAX_BRIER_EVAL_TIME = hyperparameters["MAX_BRIER_EVAL_TIME"]
LEARNING_RATE = hyperparameters["LEARNING_RATE"]
MAX_EPOCHS = hyperparameters["MAX_EPOCHS"]
NO_OF_K_SPLITS = hyperparameters["NO_OF_K_SPLITS"]
SHAP_BACKGROUND_SIZE = hyperparameters["SHAP_BACKGROUND_SIZE"]
MAX_RUNS = hyperparameters["MAX_RUNS"]

TUMOR_TYPE_COMBINATION = sorted([x for x in sys.argv[1:]])

print("Tumor type combination: ", TUMOR_TYPE_COMBINATION)

if not os.path.exists("./nn_hidden_layers.json"):
    os.system("python3 generate_nn_architectures.py")

with open("./nn_hidden_layers.json", "r") as f:
    HIDDEN_LAYERS = json.load(f)

# %% Load Data.
print("Loading Data...")

data_path = "./data/{}_scaled.pickle".format("_".join(TUMOR_TYPE_COMBINATION))

# Create data, if the combination does not exist.
if not os.path.exists(data_path):
    os.system("python3 create_data.py {}".format(
        " ".join(TUMOR_TYPE_COMBINATION)))

print(os.getcwd())
data = pd.read_pickle(data_path)
data.index = data.patient_id
gene_counts = data.iloc[:, 5:]
gene_counts_dim = gene_counts.shape[1]
gene_names = gene_counts.columns
event_indicator = data.event.to_numpy(dtype=bool)
event_time = data.time.to_numpy(dtype=np.int16)
strata = data.tumor_type.to_numpy(dtype="<U5")
patient_id = data.patient_id.to_numpy()

if TUMOR_TYPE_COMBINATION == sorted(["BRCA", "GBM", "KIRC", "LGG", "KICH", "KIRP"]):
    strata[strata == "GBM"] = "GLIOMA"
    strata[strata == "LGG"] = "GLIOMA"
    strata[strata == "KIRP"] = "KIPAN"
    strata[strata == "KICH"] = "KIPAN"
    strata[strata == "KIRC"] = "KIPAN"

print("Done.")

# %% Additional Neural Network settings.
print("Preparing Neural Network...")


def weights_init(m):
    try:
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.zeros_(m.bias.data)
    except:
        pass


# Note: Unstratified loss functions are only a special case and get selected
# through the functions arguments.
configs = {
    "PL": StratifiedPartialLikelihoodLoss,
    "SPL": StratifiedPartialLikelihoodLoss,
    "RL": StratifiedRankingLoss,
    "SRL": StratifiedRankingLoss,
}


# k-Fold cross-validation
kf = StratifiedKFold(n_splits=NO_OF_K_SPLITS, shuffle=True, random_state=42)

print("Done.")
# %% Fit models.
print("Start fitting models...")

for run_no, hidden_layers in enumerate(HIDDEN_LAYERS):
    if run_no == MAX_RUNS:
        break

    save_path = "./summaries/{}_scaled/{}".format(
        "_".join(TUMOR_TYPE_COMBINATION), str(hidden_layers)
    )

    if not os.path.exists(os.path.join(save_path, "hyperparameters.json")):
        try:
            os.makedirs(save_path)
        except:
            pass
    else:
        print("Skipping architecture, already present")
        continue

    hyperparameters["layers"] = hidden_layers

    # Placeholder for predictin error curves (PEC)
    pec_df_per_split = {key: [] for key in configs.keys()}

    # Placeholder for concordance index scores.
    concordance_index_summary = []

    # Placeholder for Shap values.
    shap_values_PartialLikelihood = pd.DataFrame(
        index=patient_id, columns=gene_names)
    shap_values_StratifiedPartialLikelihood = pd.DataFrame(
        index=patient_id, columns=gene_names
    )
    shap_values_RankingLoss = pd.DataFrame(
        index=patient_id, columns=gene_names)
    shap_values_StratifiedRankingLoss = pd.DataFrame(
        index=patient_id, columns=gene_names
    )

    split_no = 0

    for train_idx, test_idx in kf.split(gene_counts, y=strata):
        # Data and PyTorch Tensor initialization.
        gene_counts_train = gene_counts.iloc[train_idx]
        gene_counts_test = gene_counts.iloc[test_idx]

        patient_id_train = gene_counts_train.index
        patient_id_test = gene_counts_test.index

        event_indicator_train = event_indicator[train_idx]
        event_indicator_test = event_indicator[test_idx]

        event_time_train = event_time[train_idx]
        event_time_test = event_time[test_idx]

        strata_train = strata[train_idx]
        strata_test = strata[test_idx]

        X_train = torch.tensor(
            gene_counts_train.to_numpy(), requires_grad=False, dtype=torch.float32
        )
        X_test = torch.tensor(
            gene_counts_test.to_numpy(), requires_grad=False, dtype=torch.float32
        )

        if X_train.size()[0] < SHAP_BACKGROUND_SIZE:
            SHAP_BACKGROUND_SIZE = X_train.size()[0] - 1

        # Background samples for shap value estimation.
        shap_background = X_train[
            np.random.choice(
                X_train.size()[0], SHAP_BACKGROUND_SIZE, replace=False)
        ]

        # Structured arrays for Brier Score Evaluation.
        survival_data_train = np.zeros(
            event_indicator_train.shape[0],
            dtype={
                "names": ("event_indicator", "event_time"),
                "formats": ("bool", "u2"),
            },
        )
        survival_data_train["event_indicator"] = event_indicator_train
        survival_data_train["event_time"] = event_time_train

        survival_data_test = np.zeros(
            event_indicator_test.shape[0],
            dtype={
                "names": ("event_indicator", "event_time"),
                "formats": ("bool", "u2"),
            },
        )
        survival_data_test["event_indicator"] = event_indicator_test
        survival_data_test["event_time"] = event_time_test

        # Initialize model with given configs.
        current_concordance_index = []
        for loss_name, loss_function in configs.items():
            try:
                del net
            except:
                pass
            finally:
                net = BaseFeedForwardNet(
                    gene_counts_dim, 1, hidden_dims=hidden_layers)
                optimizer = torch.optim.Adam(
                    net.parameters(), lr=LEARNING_RATE)
                net.apply(weights_init)

            print("\n\n")
            print("________________________________________________________________")
            print("Run/Total Runs         : {}/{}".format(run_no + 1, MAX_RUNS))
            print("Split/Total Splits     : {}/{}".format(split_no + 1, NO_OF_K_SPLITS))
            print("Model                  : {}".format(loss_name))
            print("Tumor Type Combination : {}".format(TUMOR_TYPE_COMBINATION))
            print("Hidden Layers          : {}".format(hidden_layers))
            print("\n")

            loss_func = loss_function()

            # Monitor the last four brier scores.
            recent_brier_scores = collections.deque([1.0, 1.0, 1.0, 1.0])
            top_brier_score = 1.0

            # Stratificaion.
            if "S" in loss_name:
                stratified_fitted = True
            else:
                stratified_fitted = False
                strata_train = np.full(strata_train.shape[0], "UNSTRAT")
                strata_test = np.full(strata_test.shape[0], "UNSTRAT")

            # Start Training.
            for epoch in range(MAX_EPOCHS):

                # Training Phase. ____________________________________________
                net.train()
                optimizer.zero_grad()
                output_train = net(X_train)
                loss = loss_func(
                    output_train,
                    event_time_train,
                    event_indicator_train,
                    strata=strata_train,
                )
                loss.backward()
                optimizer.step()

                # Evaluation Phase ___________________________________________
                net.eval()

                with torch.no_grad():
                    numpy_output_train = torch.squeeze(
                        output_train.detach(), dim=1
                    ).numpy()
                    numpy_output_test = torch.squeeze(
                        net(X_test).detach(), dim=1
                    ).numpy()

                    unique_times, brier_scores = stratified_brier_score(
                        MAX_BRIER_EVAL_TIME,
                        survival_data_train,
                        survival_data_test,
                        numpy_output_train,
                        numpy_output_test,
                        strata_train=strata_train,
                        strata_test=strata_test,
                        stratified_fitted=stratified_fitted,
                    )

                    # Integrated Brier Score.
                    new_brier_score = np.trapz(
                        y=brier_scores,
                        x=unique_times,
                    )

                    c_index = stratified_concordance_index(
                        numpy_output_test,
                        event_indicator_test,
                        event_time_test,
                        strata_test,
                    )

                # Early stopping criterium.
                if (
                    ((new_brier_score < np.array(recent_brier_scores)).sum() <= 2)
                    and epoch > 10
                ) or epoch == MAX_EPOCHS - 1:

                    if SHAP_EVALUATION:
                        e = shap.DeepExplainer(net, shap_background)
                        shap_values = e.shap_values(X_test)

                    unique_times, brier_scores = stratified_brier_score(
                        MAX_BRIER_EVAL_TIME,
                        survival_data_train,
                        survival_data_test,
                        numpy_output_train,
                        numpy_output_test,
                        strata_train=strata_train,
                        strata_test=strata_test,
                        stratified_fitted=stratified_fitted,
                    )

                    pec_df_per_split[loss_name].append(
                        pd.DataFrame(
                            {
                                "brier_eval_times": unique_times,
                                "brier_scores": brier_scores,
                            }
                        )
                    )

                    if SHAP_EVALUATION:
                        if loss_name == "PL":
                            shap_values_PartialLikelihood.loc[
                                patient_id_test
                            ] = shap_values
                        elif loss_name == "SPL":
                            shap_values_StratifiedPartialLikelihood.loc[
                                patient_id_test
                            ] = shap_values
                        elif loss_name == "RL":
                            shap_values_RankingLoss.loc[patient_id_test] = shap_values
                        elif loss_name == "SRL":
                            shap_values_StratifiedRankingLoss.loc[
                                patient_id_test
                            ] = shap_values

                    # Concordance Index
                    current_concordance_index.append(c_index)

                    break

                elif new_brier_score < top_brier_score:
                    top_brier_score = new_brier_score

                recent_brier_scores.appendleft(float(new_brier_score))
                recent_brier_scores.pop()

                print(
                    "Epoch: {:3d} | Loss: {:.2f} | C-Index: {:.4f} | Brier Score: {:.4f}".format(
                        epoch, loss.item(), c_index, new_brier_score
                    )
                )

        concordance_index_summary.append(current_concordance_index)

        split_no += 1

    if SAVE_MODELS:

        # Save prediction error curves.
        for loss_name in pec_df_per_split.keys():
            for i, pec_df in enumerate(pec_df_per_split[loss_name]):
                pec_df.to_pickle(
                    os.path.join(
                        save_path, "pec_{}_split_{}.pickle".format(loss_name, i))
                )

        # Concordance Index
        concordance_index_summary = pd.DataFrame(
            concordance_index_summary,
            columns=configs.keys(),
            index=["split_{}".format(i) for i in range(1, NO_OF_K_SPLITS + 1)],
        )

        concordance_index_summary.to_pickle(
            os.path.join(save_path, "concordance_index.pickle")
        )

        if SHAP_EVALUATION:
            # Save shap values.
            shap_values_PartialLikelihood.astype("float32").to_pickle(
                os.path.join(save_path, "shap_pl.pickle")
            )
            shap_values_StratifiedPartialLikelihood.astype("float32").to_pickle(
                os.path.join(save_path, "shap_spl.pickle")
            )
            shap_values_RankingLoss.astype("float32").to_pickle(
                os.path.join(save_path, "shap_rl.pickle")
            )
            shap_values_StratifiedRankingLoss.astype("float32").to_pickle(
                os.path.join(save_path, "shap_srl.pickle")
            )

    # Save hyperparamater config.
    with open(os.path.join(save_path, "hyperparameters.json"), "w") as f:
        json.dump(hyperparameters, f, indent=2)
