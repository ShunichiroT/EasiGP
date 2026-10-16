import pandas as pd
import numpy as np

from .ensemble_regularization import (
    analytic_simplex_weights, select_weights_with_floor, apply_naive_shrinkage,
)
from pipeline_utils import get_active_compute_resources, safe_regression_metrics
import blue_weights
from asreml_accuracy_metrics import compute_accuracy_metrics_row, ACCURACY_METRIC_COLUMNS


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


def Analytic_least_squares(data_train, data_valid, data_test, record, effect, weight, MODEL, HPARAMETERS_OPT,
                            asreml_accuracy_ctx=None, *, sample_weight_ctx=None):
    """Requirements.md item 5: solves directly (no iterative search) for
    the non-negative, sum-to-1 least-squares combination of the selected
    base models' validation predictions -
    ``models/ensemble_regularization.py::analytic_simplex_weights()`` -
    then applies that SAME weight vector to the train/valid/test splits.

    ver4-25, R8 (design blueprint SS2, Stage 6): ``sample_weight_ctx``
    (a ``blue_weights.BlueWeightContext``, or ``None`` - every pre-
    ver4-25 caller) applies the SAME shared BLUE-reliability weights
    every base model was itself fitted/scored with (Fairness Principle
    P8) to this method's own closed-form solve (a GLS least-squares
    combine instead of an ordinary one - see ``analytic_simplex_weights()
    ``'s own ``w=``/``W=`` parameters) and to its own final test-score
    row (the SAME central weighted-recompute pattern
    ``genomic_prediction.py``'s per-model dispatch loop already uses -
    appended ``(unweighted)`` columns + ``metric_weighting``). `None`
    (the default) is a strict no-op, byte-identical to before this
    parameter existed (I11) - `data_train`/`data_valid`/`data_test`'s own
    'id' columns are this task's `id_train`/`id_valid`/`id_test`, in the
    SAME order `sample_weight_ctx` is aligned to (architecture doc SS4.2
    positional-identity contract), so no id-based re-lookup is needed.

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

    # ver4-25, R8: the validation-side SCORING weight (diagonal `w` or
    # full `W`) - see `blue_weights.scoring_for()`'s own docstring for
    # why this is `{}` (unweighted no-op) whenever weighting is off or
    # this split has nothing to offer. This is a validation-prediction
    # COMBINE step (every base model is already fitted), not a fresh fit
    # - so the relevant shared weight here is the SCORING weight for the
    # 'valid' split, not `sample_weight_ctx.w_fit` (which is this task's
    # shared TRAINING weight, used only when a model is itself fitted).
    _valid_weight_kwargs = blue_weights.scoring_for(sample_weight_ctx, 'valid')

    # The exact, closed-form least-squares solve itself - the only
    # candidate this method ever considers (no search, no restarts).
    analytic_w = analytic_simplex_weights(pred_matrix_valid, actual_valid_arr, ridge=ridge,
                                           **_valid_weight_kwargs)

    _resources = get_active_compute_resources()
    w_opt_validation_floor = bool(_resources.get('w_opt_validation_floor', False))

    if w_opt_validation_floor:
        # Grades the exact solution against equal weighting (and applies
        # naive shrinkage) on REALISED validation MSE - see
        # select_weights_with_floor()'s own docstring. Since this
        # method's only "candidate" IS the analytic solution itself, this
        # is exactly the "never worse than the naive ensemble on
        # validation" guarantee Nelder_Mead.py/Bayesian_optimisation.py
        # already give their own search winner. ver4-25, R8: graded on
        # the SAME weighted metric (`mse_W`) whenever weighting is on.
        w_final, _label, _diag = select_weights_with_floor(
            analytic_w, pred_matrix_valid, actual_valid_arr, shrinkage_alpha=shrinkage_alpha,
            **_valid_weight_kwargs,
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
    # ver4-25, R8: replaces the previous direct mean_squared_error()/
    # pearsonr() pair with safe_regression_metrics() (never raises on a
    # NaN prediction column - R5's own fix, previously missing from this
    # particular file) AND, when weighting is on, reports the SAME
    # weighted Pearson/MSE every model's own row already does (via the
    # central recompute below), with unweighted copies kept alongside -
    # exactly `genomic_prediction.py`'s own Step 10 pattern (SS6.3).
    actual_test = np.asarray(data_test['actual'], dtype=float)
    predicted_test_arr = np.asarray(predicted_test, dtype=float)
    r_unweighted, mse_unweighted = safe_regression_metrics(actual_test, predicted_test_arr)
    _test_weight_kwargs = blue_weights.scoring_for(sample_weight_ctx, 'test')
    if _test_weight_kwargs:
        r, mse = safe_regression_metrics(actual_test, predicted_test_arr, **_test_weight_kwargs)
    else:
        r, mse = r_unweighted, mse_unweighted

    ## Store the metrics
    record = pd.concat([record, pd.DataFrame(record.iloc[-1, :]).T]).reset_index(drop=True)
    record.loc[record.shape[0] - 1, 'model'] = 'Analytic least-squares'
    record.loc[record.shape[0] - 1, 'Pearson correlation'] = r
    record.loc[record.shape[0] - 1, 'MSE'] = mse
    if sample_weight_ctx is not None:
        record.loc[record.shape[0] - 1, 'Pearson correlation (unweighted)'] = r_unweighted
        record.loc[record.shape[0] - 1, 'MSE (unweighted)'] = mse_unweighted
        record.loc[record.shape[0] - 1, 'metric_weighting'] = sample_weight_ctx.mode

    data_test['Analytic least-squares'] = predicted_test
    data_valid['Analytic least-squares'] = predicted_valid
    data_train['Analytic least-squares'] = predicted_train

    # Requirements.md (BLUEs/BLUPs accuracy-metrics phase) - see
    # models/Nelder_Mead.py's identical block for the full explanation.
    if asreml_accuracy_ctx:
        _new_metrics = compute_accuracy_metrics_row(
            list(predicted_test), data_test['id'].tolist() if 'id' in data_test.columns else [],
            r, asreml_accuracy_ctx,
        )
        for _col, _val in _new_metrics.items():
            record.loc[record.shape[0] - 1, _col] = _val

    ## Extract weights
    # Requirements.md (BLUEs/BLUPs accuracy-metrics phase) - see
    # models/Nelder_Mead.py's identical block for the full explanation.
    weight_sample = pd.DataFrame(record.iloc[record.shape[0] - 1, :]).T.drop(
        ['Pearson correlation', 'MSE'] + list(ACCURACY_METRIC_COLUMNS)
        + list(blue_weights.METRIC_WEIGHTING_EXTRA_COLUMNS),
        axis=1, errors='ignore'
    )
    weight_sample['model'] = 'Analytic least-squares'
    weight_sample = pd.concat([weight_sample.reset_index(drop=True), weight_extracted.reset_index(drop=True)], axis=1)
    weight = pd.concat([weight, weight_sample])

    weight_extracted_normalised = _safe_row_normalize(weight_extracted)

    ## Calculate weighted effects
    # F1 fix: the accumulator used to be created only when i == 0, so a first model without effect rows
    # (e.g. explainability off for it) raised UnboundLocalError on the next model, and no effect rows at all
    # raised it at the end. It now starts from the first model that HAS effect rows (identical result when
    # every model has them) and the ensemble effect row is skipped when no model has any.
    effect_weighted = None
    for i in range(len(model_selected)):
        if effect[effect['model'] == model_selected[i]].shape[0] != 0:
            if effect_weighted is None:
                effect_weighted = _safe_row_normalize(
                    effect[effect['model'] == model_selected[i]].tail(1).iloc[:, 5:].abs().reset_index(drop=True)
                ).mul(weight_extracted_normalised[model_selected[i]], axis=0).reset_index(drop=True)
            else:
                effect_weighted += _safe_row_normalize(
                    effect[effect['model'] == model_selected[i]].tail(1).iloc[:, 5:].abs().reset_index(drop=True)
                ).mul(weight_extracted_normalised[model_selected[i]], axis=0).reset_index(drop=True)

    if effect_weighted is not None:
        effect = pd.concat([effect, pd.DataFrame(effect.iloc[effect.shape[0] - 1, :]).T]).reset_index(drop=True)
        effect.loc[effect.shape[0] - 1, 'model'] = 'Analytic least-squares'
        effect.loc[effect.shape[0] - 1, list(effect_weighted.columns)] = effect_weighted.values

    return record, effect, data_test, data_valid, data_train, weight
