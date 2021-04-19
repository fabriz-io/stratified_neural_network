# %%
import numpy as np
from sksurv.metrics import brier_score, concordance_index_censored
from sksurv.linear_model.coxph import BreslowEstimator

# %%
def stratified_brier_score(
    maximum_brier_eval_time,
    survival_data_train,
    survival_data_test,
    risk_score_train,
    risk_score_test,
    strata_train,
    strata_test,
    stratified_fitted=True,
    save_stratified_scores=True,
    minimum_brier_eval_time=None,
):

    event_time_train = survival_data_train["event_time"]
    event_time_test = survival_data_test["event_time"]

    # Assert values lie within the needed ranges
    min_strata_train = [
        np.min(event_time_train[strata_train == s]) for s in np.unique(strata_train)
    ]
    max_strata_train = [
        np.max(event_time_train[strata_train == s]) for s in np.unique(strata_train)
    ]

    # Get boolean indexer array for event times in test data, which are
    # smaller/greater than minimum/maximum event time in train data
    extends_strata_min = [
        (
            np.min(event_time_train[strata_train == s])
            > event_time_test[strata_test == s]
        )
        for s in np.unique(strata_test)
    ]
    extends_strata_max = [
        (
            np.max(event_time_train[strata_train == s])
            < event_time_test[strata_test == s]
        )
        for s in np.unique(strata_test)
    ]

    min_strata_test = []
    max_strata_test = []

    for s, e_min_mask, e_max_mask in zip(
        np.unique(strata_train), extends_strata_min, extends_strata_max
    ):
        if e_min_mask.any():
            min_strata_test.append(
                np.min(event_time_test[strata_test == s][~e_min_mask])
            )
        else:
            min_strata_test.append(np.min(event_time_test[strata_test == s]))
        if e_max_mask.any():
            max_strata_test.append(
                np.max(event_time_test[strata_test == s][~e_max_mask])
            )
        else:
            max_strata_test.append(np.max(event_time_test[strata_test == s]))

    # Choose the maximum of the minimal values within the strata
    event_time_strata_min = np.max(min_strata_train + min_strata_test)
    # Choose the minimum of maximal values within the strata
    event_time_strata_max = np.min(max_strata_train + max_strata_test)

    if event_time_strata_max < maximum_brier_eval_time:
        pec_largest_eval_time = event_time_strata_max
    else:
        pec_largest_eval_time = maximum_brier_eval_time

    if (
        minimum_brier_eval_time is not None
        and event_time_strata_min < minimum_brier_eval_time
    ):
        pec_smallest_eval_time = minimum_brier_eval_time
    else:
        pec_smallest_eval_time = event_time_strata_min

    # Final evaluation times for brier score
    eval_times_brier_score = np.arange(
        start=pec_smallest_eval_time, stop=pec_largest_eval_time - 1, step=20
    )

    survival_train_groups = []
    survival_test_groups = []

    risk_train_groups = []
    risk_test_groups = []

    strata_indicator_train_groups = []
    strata_indicator_test_groups = []

    for strata in np.unique(strata_train):

        risk_score_strata_train = risk_score_train[strata == strata_train]
        risk_score_strata_test = risk_score_test[strata == strata_test]

        survival_strata_train = survival_data_train[strata == strata_train]
        survival_strata_test = survival_data_test[strata == strata_test]

        strata_indicator_train = strata_train[strata == strata_train]
        strata_indicator_test = strata_test[strata == strata_test]

        # Check that testing times lie within range of training times.
        extends_train_min = (
            np.min(survival_strata_train["event_time"])
            > survival_strata_test["event_time"]
        )

        if extends_train_min.any():
            risk_score_strata_test = risk_score_strata_test[~extends_train_min]
            survival_strata_test = survival_strata_test[~extends_train_min]
            strata_indicator_test = strata_indicator_test[~extends_train_min]

        extends_train_max = (
            np.max(survival_strata_train["event_time"])
            < survival_strata_test["event_time"]
        )

        if extends_train_max.any():
            risk_score_strata_test = risk_score_strata_test[~extends_train_max]
            survival_strata_test = survival_strata_test[~extends_train_max]
            strata_indicator_test = strata_indicator_test[~extends_train_max]

        survival_train_groups.append(survival_strata_train)
        survival_test_groups.append(survival_strata_test)
        risk_train_groups.append(risk_score_strata_train)
        risk_test_groups.append(risk_score_strata_test)
        strata_indicator_train_groups.append(strata_indicator_train)
        strata_indicator_test_groups.append(strata_indicator_test)

    predictions = []

    if stratified_fitted:
        for train_data, train_risk, test_risk in zip(
            survival_train_groups,
            risk_train_groups,
            risk_test_groups,
        ):
            # Fit Breslow Estimator on Training Data.
            breslow = BreslowEstimator()
            breslow.fit(
                train_risk,
                train_data["event_indicator"],
                train_data["event_time"],
            )

            # Predict Survival Probability on Test Data.
            surv_funcs = breslow.get_survival_function(test_risk)
            prob_preds = [fn(eval_times_brier_score) for fn in surv_funcs]

            # Append stratified data.
            predictions.append(prob_preds)
    else:
        breslow = BreslowEstimator()
        breslow.fit(
            np.concatenate(risk_train_groups),
            np.concatenate(survival_train_groups)["event_indicator"],
            np.concatenate(survival_train_groups)["event_time"],
        )

        surv_funcs = breslow.get_survival_function(np.concatenate(risk_test_groups))
        prob_preds = [fn(eval_times_brier_score) for fn in surv_funcs]

        predictions.append(prob_preds)

    total_brier_scores = []
    group_sizes = []

    strata_train = np.concatenate(strata_indicator_train_groups)
    strata_test = np.concatenate(strata_indicator_test_groups)
    survival_data_train = np.concatenate(survival_train_groups)
    survival_data_test = np.concatenate(survival_test_groups)
    predictions = np.concatenate(predictions)

    for strata in np.unique(strata_train):
        train_dat = survival_data_train[strata_train == strata]
        test_dat = survival_data_test[strata_test == strata]
        preds = predictions[strata_test == strata]

        # IPCW weights are included in imported function "brier_score"
        _, strata_brier_score = brier_score(
            train_dat,
            test_dat,
            preds,
            eval_times_brier_score,
        )

        total_brier_scores.append(strata_brier_score)
        group_sizes.append(len(preds))

    if save_stratified_scores:
        return eval_times_brier_score, np.average(
            np.stack(total_brier_scores), weights=group_sizes, axis=0
        )
    else:
        return eval_times_brier_score, total_brier_scores, group_sizes


# _____________________________________________________________________________
def stratified_concordance_index(output, event, time, strata=None):
    """Calculates the stratified concordance index.

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
            estimate=output[indices_strata],
        )

        # Add up concordant and discordant pairs for stratified evaluation.
        concordant += c_index[1]
        discordant += c_index[2]
    return concordant / (concordant + discordant)
