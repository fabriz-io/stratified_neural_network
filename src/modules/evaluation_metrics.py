import numpy as np
from sksurv.nonparametric import _compute_counts
from sksurv.functions import StepFunction
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.metrics import brier_score, concordance_index_censored
from sksurv.linear_model.coxph import BreslowEstimator


def stratified_brier_score(eval_time, survival_data_train, survival_data_test, pred_train, pred_test, strata_train=None, strata_test=None):
    if strata_train is None and strata_test is None:
        strata_train = np.full(len(pred_train), 1)
        strata_test = np.full(len(pred_test), 1)

    breslow = BreslowEstimator()
    total_brier_scores = []
    group_sizes = []

    for strata in np.unique(strata_train):
        # Get predictions and survival times of strata.
        indices_strata_train = np.where(strata == strata_train)[0]
        indices_strata_test = np.where(strata == strata_test)[0]
        pred_strata_train = pred_train[indices_strata_train]
        pred_strata_test = pred_test[indices_strata_test]
        survival_strata_train = survival_data_train[indices_strata_train]
        survival_strata_test = survival_data_test[indices_strata_test]

        # Fit Breslow Estimator on Training Data.
        breslow.fit(pred_strata_train,
                    survival_strata_train['event_indicator'],
                    survival_strata_train['event_time'])

        # Predict Survival Probability on Test Data.
        surv_funcs = breslow.get_survival_function(pred_strata_test)
        prob_preds = [fn(eval_time) for fn in surv_funcs]

        # IPCW weights is included in imported function "brier_score"
        _, strata_brier_score = brier_score(survival_strata_train,
                                            survival_strata_test,
                                            prob_preds,
                                            eval_time)

        total_brier_scores.append(strata_brier_score[0])
        group_sizes.append(len(pred_strata_test))
    return np.average(total_brier_scores, weights=group_sizes)

#_____________________________________________________________________________
def stratified_concordance_index(output, event, time, strata=None):
    """" Calculated the stratified concordance index.

    If no strata is given or every individual belongs to the same strata,
    the unstratified concordance index is calculated.

    params:
    """
    concordant = 0
    discordant = 0

    if strata is None:
        strata = np.full(len(time), 1)

    for strat in np.unique(strata):
        # Get individuals of the strata.
        indices_strata = np.where(strata == strat)[0]

        # Calculate concordance index.
        c_index = concordance_index_censored(
            event_indicator=event[indices_strata],
            event_time=time[indices_strata],
            estimate=output[indices_strata])

        # Add up concordant and discordant pairs for stratified evaluation.
        concordant += c_index[1]
        discordant += c_index[2]
    return concordant / (concordant + discordant)
