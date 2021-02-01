# %% Import modules.
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import collections
import os
import shap

from modules.evaluation_metrics import stratified_concordance_index, stratified_brier_score
from modules.torch_models import BaseFeedForwardNet, StratifiedPartialLikelihoodLoss, StratifiedRankingLoss

# %% Hyperparameters.
BRIER_EVAL_TIME = 1500 # Evaluation time for Brier Score early stopping criterium.
LEARNING_RATE = 0.0001
MAX_EPOCHS = 100
MIN_WIDTH = 10  # Minimum layer width.
MAX_WIDTH = 200  # Maximum layer width.
MIN_DEPTH = 2 # Minimum network depth.
MAX_DEPTH = 10 # Maximum network depth.
NO_OF_K_SPLITS = 5
NO_OF_RUNS = 1
SHAP_BACKGROUND_SIZE = 200
EVALUATION_TIMES_PEC = list(range(10, 1500, 30)) # Evaluation points in time for prediction error curves.
SAVE_MODELS = True

hyperparameters = {
    "BRIER_EVAL_TIME": BRIER_EVAL_TIME,
    "LEARNING_RATE": LEARNING_RATE,
    "MAX_EPOCHS": MAX_EPOCHS,
    "MIN_WIDTH": MIN_WIDTH,
    "MAX_WIDTH": MAX_WIDTH,
    "MIN_DEPTH": MIN_DEPTH,
    "MAX_DEPTH": MAX_DEPTH,
    "NO_OF_K_SPLITS": NO_OF_K_SPLITS,
    "SHAP_BACKGROUND_SIZE": SHAP_BACKGROUND_SIZE,
    "EVALUATION_TIMES_PEC": EVALUATION_TIMES_PEC
}

# %% Load Data.
print("Loading Data...")

data_path = "data/brca_kipan_glioma_normalized_3000_features.csv"
data = pd.read_csv(data_path)
data.index = data.patient_id

gene_counts = data.iloc[:, 5:]
gene_counts_dim = gene_counts.shape[1]
gene_names = gene_counts.columns
event_indicator = data.event.to_numpy(dtype=bool)
event_time = data.time.to_numpy(dtype=np.int16)
strata = data.tumor_type.to_numpy(dtype=str)
patient_id = data.patient_id.to_numpy()

print("Done.")

#%% Additional Neural Network settings.
print("Preparing Neural Network...")

def weights_init(m):
    try:
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.zeros_(m.bias.data)
    except:
        pass

# Note: For unstratified models the argument "strata=None" is parsed later.
configs = {
    "Partial Likelihood": StratifiedPartialLikelihoodLoss,
    "Stratified Partial Likelihood": StratifiedPartialLikelihoodLoss,
    "Ranking Loss": StratifiedRankingLoss,
    "Stratified Ranking Loss": StratifiedRankingLoss
}

# k-Fold cross-validation
kf = StratifiedKFold(n_splits=NO_OF_K_SPLITS, shuffle=True, random_state=42)

print("Done.")

# %% Load summary files or create placeholders for evaluation metrics if they dont' exist.

try:
    prediction_error_curves_PartialLikelihood = pd.read_csv("./summary/pec_pl.csv", index_col=0)
    prediction_error_curves_StratifiedPartialLikelihood = pd.read_csv("./summary/pec_spl.csv", index_col=0)
    prediction_error_curves_RankingLoss = pd.read_csv("./summary/pec_rl.csv", index_col=0)
    prediction_error_curves_StratifiedRankingLoss = pd.read_csv("./summary/pec_srl.csv", index_col=0)

    concordance_index_summary_df = pd.read_csv("./summary/concordance_index.csv")

except:
    # Empty placeholder for PECs.
    prediction_error_curves_PartialLikelihood = pd.DataFrame(columns=[str(t) for t in EVALUATION_TIMES_PEC])
    prediction_error_curves_StratifiedPartialLikelihood = pd.DataFrame(columns=[str(t) for t in EVALUATION_TIMES_PEC])
    prediction_error_curves_RankingLoss = pd.DataFrame(columns=[str(t) for t in EVALUATION_TIMES_PEC])
    prediction_error_curves_StratifiedRankingLoss = pd.DataFrame(columns=[str(t) for t in EVALUATION_TIMES_PEC])

    # Placeholder for concordance index scores.
    concordance_index_summary_df = pd.DataFrame(columns=configs.keys())


# Placeholder for Shap values.
shap_values_PartialLikelihood = pd.DataFrame(index=patient_id, columns=gene_names)
shap_values_StratifiedPartialLikelihood = pd.DataFrame(index=patient_id, columns=gene_names)
shap_values_RankingLoss = pd.DataFrame(index=patient_id, columns=gene_names)
shap_values_StratifiedRankingLoss = pd.DataFrame(index=patient_id, columns=gene_names)


# %% Fit models.
print("Start fitting models...")

for run in range(NO_OF_RUNS):
    # Generate random layer depth and widths. Order widths descending.
    hidden_layers = np.random.randint(MIN_WIDTH, MAX_WIDTH, np.random.randint(MIN_DEPTH, MAX_DEPTH))
    HIDDEN_LAYERS = hidden_layers[np.argsort(-hidden_layers)]

    split_no = 1
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

        X_train = torch.tensor(gene_counts_train.to_numpy(), requires_grad=False, dtype=torch.float32)
        X_test = torch.tensor(gene_counts_test.to_numpy(), requires_grad=False, dtype=torch.float32)

        # Background samples for shap value estimation.
        shap_background = X_train[np.random.choice(X_train.size()[0], SHAP_BACKGROUND_SIZE, replace=False)]

        # Structured arrays for Brier Score Evaluation.
        survival_data_train = np.zeros(event_indicator_train.shape[0],
            dtype={'names':('event_indicator', 'event_time'), 'formats':('bool', 'u2')})
        survival_data_train['event_indicator'] = event_indicator_train
        survival_data_train['event_time'] = event_time_train

        survival_data_test = np.zeros(event_indicator_test.shape[0],
            dtype={'names':('event_indicator', 'event_time'), 'formats':('bool', 'u2')})
        survival_data_test['event_indicator'] = event_indicator_test
        survival_data_test['event_time'] = event_time_test

        # Initialize model with given configs.
        current_concordance_index = []
        for loss_name, loss_function in configs.items():           
            try:
                # Clean up if model exists.
                del net
            except:
                pass
            finally:
                net = BaseFeedForwardNet(gene_counts_dim, 1, hidden_dims=HIDDEN_LAYERS)
                optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
                net.apply(weights_init)

            print("\n\n")
            print("________________________________________________________________")
            print("Run/Total Runs    : {}/{}".format(run + 1, NO_OF_RUNS))
            print("Split/Total Splits: {}/{}".format(split_no, NO_OF_K_SPLITS))
            print("Model             : {}".format(loss_name))
            print("Hidden Layers     : {}".format(HIDDEN_LAYERS))
            print("\n")

            loss_func = loss_function()

            # Monitor the last four brier scores.
            recent_brier_scores = collections.deque([1.0, 1.0, 1.0, 1.0])
            top_brier_score = 1.0

            # Stratificaion.
            if "Stratified" in loss_name:
                strata_train = strata[train_idx]
                strata_test = strata[test_idx]
            else:
                strata_train = np.full(len(event_indicator_train), "NOT STRATIFIED")
                strata_test = np.full(len(event_indicator_test), "NOT STRATIFIED")

            strata_test_c_index = strata[test_idx]

            # Start Training.
            for epoch in range(MAX_EPOCHS):

                # Training Phase.
                net.train()
                optimizer.zero_grad()
                output_train = net(X_train)
                loss = loss_func(output_train,
                                event_time_train,
                                event_indicator_train,
                                strata=strata_train)
                loss.backward()
                optimizer.step()

                # Evaluation Phase ___________________________________________
                net.eval()

                with torch.no_grad():
                    numpy_output_train = torch.squeeze(output_train.detach(), dim=1).numpy()
                    numpy_output_test = torch.squeeze(net(X_test).detach(), dim=1).numpy()

                    new_brier_score = stratified_brier_score(BRIER_EVAL_TIME,
                                                             survival_data_train,
                                                             survival_data_test,
                                                             numpy_output_train,
                                                             numpy_output_test, 
                                                             strata_train=strata_train,
                                                             strata_test=strata_test)

                    c_index = stratified_concordance_index(numpy_output_test, event_indicator_test, event_time_test, strata_test_c_index) 

                # Early stopping criterium.
                if (new_brier_score > recent_brier_scores[-1] and epoch > 10) or epoch == MAX_EPOCHS - 1:
                    e = shap.DeepExplainer(net, shap_background)
                    shap_values = e.shap_values(X_test)

                    prediction_error_curve = pd.Series(
                        [stratified_brier_score(t,
                                                survival_data_train,
                                                survival_data_test,
                                                numpy_output_train,
                                                numpy_output_test, 
                                                strata_train=strata_train,
                                                strata_test=strata_test
                                                )
                            for t in EVALUATION_TIMES_PEC
                        ], index=[str(t) for t in EVALUATION_TIMES_PEC])

                    if loss_name == "Partial Likelihood":
                        prediction_error_curves_PartialLikelihood = \
                            prediction_error_curves_PartialLikelihood.append(prediction_error_curve, ignore_index=True)
                        
                        shap_values_PartialLikelihood.loc[patient_id_test] = shap_values

                    elif loss_name == "Stratified Partial Likelihood":
                        prediction_error_curves_StratifiedPartialLikelihood = \
                            prediction_error_curves_StratifiedPartialLikelihood.append(prediction_error_curve, ignore_index=True)

                        shap_values_StratifiedPartialLikelihood.loc[patient_id_test] = shap_values

                    elif loss_name == "Ranking Loss":
                        prediction_error_curves_RankingLoss = \
                            prediction_error_curves_RankingLoss.append(prediction_error_curve, ignore_index=True)

                        shap_values_RankingLoss.loc[patient_id_test] = shap_values

                    elif loss_name == "Stratified Ranking Loss":
                        prediction_error_curves_StratifiedRankingLoss = \
                            prediction_error_curves_StratifiedRankingLoss.append(prediction_error_curve, ignore_index=True)

                        shap_values_StratifiedRankingLoss.loc[patient_id_test] = shap_values

                    # Concordance Index
                    current_concordance_index.append(c_index)

                    break

                elif new_brier_score < top_brier_score:
                    top_brier_score = new_brier_score

                recent_brier_scores.appendleft(new_brier_score)
                recent_brier_scores.pop()

                print("Epoch: {:3d} | Loss: {:.2f} | C-Index: {:.4f} | Brier Score: {:.4f}".format(epoch, loss.item(), c_index, new_brier_score))


        concordance_index_summary_df = concordance_index_summary_df.append(pd.Series(current_concordance_index, index=configs.keys()), ignore_index=True)

        split_no += 1

    if SAVE_MODELS:
        save_path = "./summary/{}".format(str(HIDDEN_LAYERS))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Save shap values.
        shap_values_PartialLikelihood.astype('float64').to_csv(save_path + "/shap_pl.csv")
        shap_values_StratifiedPartialLikelihood.astype('float64').to_csv(save_path + "/shap_spl.csv")
        shap_values_RankingLoss.astype('float64').to_csv(save_path + "/shap_rl.csv")
        shap_values_StratifiedRankingLoss.astype('float64').to_csv(save_path + "/shap_srl.csv")

        # Save hyperparamater config.
        with open(os.path.join(save_path, "hyperparameters.txt"), 'w') as f:
            f.write(repr(hyperparameters))


#%% Save metrics.

if SAVE_MODELS:
    # Save prediction error curves.
    prediction_error_curves_PartialLikelihood.astype('float64').to_csv("./summary/pec_pl.csv")
    prediction_error_curves_StratifiedPartialLikelihood.astype('float64').to_csv("./summary/pec_spl.csv")
    prediction_error_curves_RankingLoss.astype('float64').to_csv("./summary/pec_rl.csv")
    prediction_error_curves_StratifiedRankingLoss.astype('float64').to_csv("./summary/pec_srl.csv")

    # Save concordance index.
    concordance_index_summary_df.to_csv("./summary/concordance_index.csv")
