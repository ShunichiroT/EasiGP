import pandas as pd
import numpy as np
import scipy.stats
from sklearn.metrics import mean_squared_error

from .ensemble_regularization import (
    analytic_simplex_weights, select_weights_with_floor, apply_naive_shrinkage,
)
from pipeline_utils import get_active_compute_resources


def _safe_row_normalize(df):
    """Row-wise L1-normalize (each row divided by its own sum of - here
    already non-negative - values, e.g. abs(marker effect)) - safely: a
    row whose sum is exactly 0 (a model that assigned literally zero
    effect to every marker - e.g. SVR/KNN, which aren't additive-effect
    models) is left as all-zero, rather than corrupting or crashing this
    whole weighted ensemble's combined marker-effect output. See
    models/Nelder_Mead.py::_safe_row_normalize()'s own docstring for the
    full root-cause explanation (object-dtype columns from the R-backed
    models via rpy2) this duplicated helper guards against - duplicated
    here rather than imported, matching how every other weight-
    optimisation method in this directory already keeps its own copy."""
    df = df.astype(float)
    row_sums = df.sum(axis=1).replace(0, np.nan)
    return df.div(row_sums, axis=0).fillna(0)


def Analytic_least_squares(data_train, data_valid, data_test, record, effect, weight, MODEL, HPARAMETERS_OPT):
    """Requirements.md item 5: solves directly (no iterative search) for
    the non-negative, sum-to-1 least-squares combination of the selected
    base models' validation predictions -
    ``models/ensemble_regularization.py::analytic_simplex_weights()`` -
    then applies that SAME weight vector to the train/valid/test splits.

    This is the exact algorithm previously only reachable as the other
    three weight-optimisation methods' opt-in ``W_OPT_ANALYTIC_SEED``
    seed/floor candidate, now offered as its own, independent method that
    a person can select on its own (or alongside the others) from the
    '4. Ensemble' tab - it no longer reads ``W_OPT_ANALYTIC_SEED`` at
    all, only the (also global, shared with every other weight-
    optimisation method) ``W_OPT_VALIDATION_FLOOR`` setting, which - when
    on - grades this method's own exact solution against equal weighting
    on realised validation MSE, exactly as ``Nelder_Mead.py``/
    ``Bayesian_optimisation.py`` already do for their own search winner.

    Parameters
    ----------
    HPARAMETERS_OPT : positional list -
        ``[0]`` ridge regularisation (float, default 1e-6) - forwarded to
            ``analytic_simplex_weights()``'s own ``ridge`` kwarg.
        ``[1]`` naive-shrinkage alpha (float, default 1.0) - forwarded to
            ``select_weights_with_floor()``/``apply_naive_shrinkage()``,
            exactly as every other weight-optimisation method's own
            trailing shrinkage field.
    """
    ridge = float(HPARAMETERS_OPT[0]) if len(HPARAMETERS_OPT) > 0 and HPARAMETERS_OPT[0] is not None else 1e-6
    shrinkage_alpha = float(HPARAMETERS_OPT[1]) if len(HPARAMETERS_OPT) > 1 and HPARAMETERS_OPT[1] is not None else 1.0

    model_selected = MODEL.copy()
    if 'ensemble' in MODEL:
        model_selected.remove('ensemble')

    # Force every model-prediction column this function's arithmetic
    # touches onto a clean float64 dtype before solving - see
    # models/Nelder_Mead.py's identical fix (same R-interop root cause:
    # marker-effect/prediction columns arriving via rpy2 can be
    # object-dtype, which turns an exact-zero division into a raised
    # ZeroDivisionError instead of a silent inf/nan).
    data_train = data_train.copy()
    data_valid = data_valid.copy()
    data_test = data_test.copy()
    for _df in (data_train, data_valid, data_test):
        _df[model_selected + ['actual']] = _df[model_selected + ['actual']].astype(float)

    pred_matrix_valid = data_valid[model_selected].to_numpy(dtype=float)
    actual_valid_arr = data_valid['actual'].to_numpy(dtype=float)

    # The exact, closed-form least-squares solve itself - the only
    # candidate this method ever considers (no search, no restarts).
    analytic_w = analytic_simplex_weights(pred_matrix_valid, actual_valid_arr, ridge=ridge)

    _resources = get_active_compute_resources()
    w_opt_validation_floor = bool(_resources.get('w_opt_validation_floor', False))

    if w_opt_validation_floor:
        # Grades the exact solution against equal weighting (and applies
        # naive shrinkage) on REALISED validation MSE - see
        # select_weights_with_floor()'s own docstring. Since this
        # method's only "candidate" IS the analytic solution itself, this
        # is exactly the "never worse than the naive ensemble on
        # validation" guarantee Nelder_Mead.py/Bayesian_optimisation.py
        # already give their own search winner.
        w_final, _label, _diag = select_weights_with_floor(
            analytic_w, pred_matrix_valid, actual_valid_arr, shrinkage_alpha=shrinkage_alpha,
        )
        print(f"[GP] Analytic least-squares weight solve: floor selected {_label!r} "
              f"(validation MSE - {_diag}).")
    else:
        k = len(model_selected)
        w = np.clip(np.asarray(analytic_w, dtype=float), 0.0, None)
        total = float(w.sum())
        w = (w / total) if (np.isfinite(total) and total > 0) else np.full(k, 1.0 / k)
        w_final = list(apply_naive_shrinkage(w, shrinkage_alpha))

    weight_extracted = pd.DataFrame([w_final])
    weight_extracted.columns = model_selected

    ## Apply the final weight vector to every split
    w_arr = np.asarray(w_final, dtype=float)
    pred_matrix_test = data_test[model_selected].to_numpy(dtype=float)
    pred_matrix_train = data_train[model_selected].to_numpy(dtype=float)

    predicted_test = pd.Series(pred_matrix_test @ w_arr).reset_index(drop=True)
    predicted_valid = pd.Series(pred_matrix_valid @ w_arr).reset_index(drop=True)
    predicted_train = pd.Series(pred_matrix_train @ w_arr).reset_index(drop=True)

    ## Calculate the metrics
    actual_test = data_test['actual'].tolist()
    mse = mean_squared_error(actual_test, predicted_test)
    r = scipy.stats.pearsonr(actual_test, predicted_test)[0]

    ## Store the metrics
    record = pd.concat([record, pd.DataFrame(record.iloc[-1, :]).T]).reset_index(drop=True)
    record.loc[record.shape[0] - 1, 'model'] = 'Analytic least-squares'
    record.loc[record.shape[0] - 1, 'Pearson correlation'] = r
    record.loc[record.shape[0] - 1, 'MSE'] = mse

    data_test['Analytic least-squares'] = predicted_test
    data_valid['Analytic least-squares'] = predicted_valid
    data_train['Analytic least-squares'] = predicted_train

    ## Extract weights
    weight_sample = pd.DataFrame(record.iloc[record.shape[0] - 1, :]).T.drop(['Pearson correlation', 'MSE'], axis=1)
    weight_sample['model'] = 'Analytic least-squares'
    weight_sample = pd.concat([weight_sample.reset_index(drop=True), weight_extracted.reset_index(drop=True)], axis=1)
    weight = pd.concat([weight, weight_sample])

    weight_extracted_normalised = _safe_row_normalize(weight_extracted)

    ## Calculate weighted effects
    for i in range(len(model_selected)):
        if effect[effect['model'] == model_selected[i]].shape[0] != 0:
            if i == 0:
                effect_weighted = _safe_row_normalize(
                    effect[effect['model'] == model_selected[i]].tail(1).iloc[:, 5:].abs().reset_index(drop=True)
                ).mul(weight_extracted_normalised[model_selected[i]], axis=0).reset_index(drop=True)
            else:
                effect_weighted += _safe_row_normalize(
                    effect[effect['model'] == model_selected[i]].tail(1).iloc[:, 5:].abs().reset_index(drop=True)
                ).mul(weight_extracted_normalised[model_selected[i]], axis=0).reset_index(drop=True)

    effect = pd.concat([effect, pd.DataFrame(effect.iloc[effect.shape[0] - 1, :]).T]).reset_index(drop=True)
    effect.loc[effect.shape[0] - 1, 'model'] = 'Analytic least-squares'
    effect.loc[effect.shape[0] - 1, list(effect_weighted.columns)] = effect_weighted.values

    return record, effect, data_test, data_valid, data_train, weight
