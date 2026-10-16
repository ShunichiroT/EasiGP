#Change here 
###============###
#DATA_PATH = r'C:\Users\uqstomur\OneDrive - The University of Queensland\Documents\Result\TeoNAM'

#MODEL = ['rrBLUP','BayesB','RKHS','RF','SVR','MLP']

#INPUT = 'Predicted_result_test_all.csv'

###============###


import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error
import scipy.stats

from pipeline_utils import safe_regression_metrics, safe_row_l1_normalize
from models.interaction_extraction import naive_ensemble_interactions, _INTERACTION_ENSEMBLE_SCHEMA
from asreml_accuracy_metrics import compute_accuracy_metrics_row


def _safe_row_normalize(df):
    """ver4-25, R5: thin alias for `pipeline_utils.safe_row_l1_normalize()`
    - promoted there as the single shared authority (imported by
    `scatter_plot.py` too, so its own per-task marker-effect
    normalisation can never numerically disagree with what this
    ensemble computes over the same rows). Kept as a module-level name
    here since it is called from more than one place within this file;
    see `pipeline_utils.safe_row_l1_normalize()`'s own docstring for the
    full rationale/history (originally a real production-crash bugfix)."""
    return safe_row_l1_normalize(df)


# Function to calculate the metrics
def _metric(data, asreml_accuracy_ctx_resolver=None, metric_directions=None):
    # ver4-4 R5 Fix 2: safe_regression_metrics() never raises, even when
    # an ensembled prediction column carries NaN (e.g. one of the
    # ensembled models diverged - see R5 design record). The direct
    # mean_squared_error()/pearsonr() pair this replaces is exactly what
    # used to raise ValueError('Input contains NaN') here, once that NaN
    # reaches the naive ensemble.
    #
    # ver4-25, R8 (design blueprint SS2, Stage 6; Stage 6 completion for
    # 'full' mode): when the resolved ctx (below) carries a non-empty
    # 'test_weights' dict (diagonal mode) or a 'test_weight_matrix' pair
    # (full mode), the naive ensemble's own Pearson/MSE are ALSO
    # weighted - the same fairness principle (P8) applied to every
    # per-model row already gets applied here.
    _ctx_for_weights = None
    if asreml_accuracy_ctx_resolver is not None:
        _first_row = data.iloc[0]
        _ids_for_scenario = data['id'].tolist() if 'id' in data.columns else None
        try:
            _ctx_for_weights = asreml_accuracy_ctx_resolver(
                _first_row.get('population'), _first_row.get('phenotype'),
                _first_row.get('ratio'), _first_row.get('sample'),
                ids=_ids_for_scenario,
            )
        except TypeError:
            # Backward compatibility (I11): a resolver built before the
            # 'full'-mode `ids=` extension existed (e.g. a hand-rolled
            # test double, or a resolver from an older code path) simply
            # doesn't accept the keyword at all - retry the original
            # 4-positional-argument call exactly as before this feature
            # existed, rather than losing 'diagonal'-mode weighting too.
            try:
                _ctx_for_weights = asreml_accuracy_ctx_resolver(
                    _first_row.get('population'), _first_row.get('phenotype'),
                    _first_row.get('ratio'), _first_row.get('sample'),
                )
            except Exception:
                _ctx_for_weights = None
        except Exception:
            _ctx_for_weights = None

    _test_weights = (_ctx_for_weights or {}).get('test_weights') or {}
    _test_weight_matrix = (_ctx_for_weights or {}).get('test_weight_matrix')

    r_unweighted, mse_unweighted = safe_regression_metrics(data.loc[:, 'actual'], data.loc[:, 'ensemble'])
    # A scenario group must hold each test individual exactly once. Repeated ids (a corrupted
    # `Prediction_result_test` table) make the 'full'-mode weight matrix impossible to rebuild,
    # so this row would silently be scored unweighted while every other row is weighted. Never
    # silent: say so (the values themselves are unchanged).
    if 'id' in data.columns and data['id'].duplicated().any():
        _f = data.iloc[0]
        print(f"[ensemble] WARNING: naive ensemble group (population={_f.get('population')!r}, "
              f"phenotype={_f.get('phenotype')!r}, ratio={_f.get('ratio')!r}, sample={_f.get('sample')!r}) "
              f"holds {int(data['id'].duplicated().sum())} repeated test id(s) out of {len(data)} rows - "
              f"the prediction table for this task is corrupted (duplicated/mislabelled rows); this row's "
              f"metrics are NOT comparable with the other models' rows. Re-run the task.")
    _row_metric_weighting = None
    if _test_weight_matrix is not None and 'id' in data.columns:
        _W, _mids = _test_weight_matrix
        # `_mids` is `data['id']`'s own order by construction (it is the
        # exact `ids` this call passed to the resolver above) - no
        # re-alignment needed.
        r, mse = safe_regression_metrics(data.loc[:, 'actual'], data.loc[:, 'ensemble'], W=_W)
        _row_metric_weighting = 'full'
    elif _test_weights and 'id' in data.columns:
        _w = np.array([_test_weights.get(str(_id), np.nan) for _id in data['id']], dtype=float)
        if np.all(np.isfinite(_w)):
            r, mse = safe_regression_metrics(data.loc[:, 'actual'], data.loc[:, 'ensemble'], w=_w)
            _row_metric_weighting = 'diagonal'
        else:
            # Every individual's own diagonal weight should always be
            # resolvable whenever 'test_weights' is non-empty at all
            # (the sidecar is written per-scenario, covering every test
            # individual in one shot) - a non-finite entry here means an
            # id genuinely wasn't found (e.g. a model-fan-out edge case),
            # never a "weighting is off" state, so this degrades to
            # unweighted for THIS row only, same as a per-model row's own
            # non-fatal degrade path, rather than fabricating a weight.
            r, mse = r_unweighted, mse_unweighted
            _row_metric_weighting = 'none'
    else:
        r, mse = r_unweighted, mse_unweighted

    row = dict(Pearson=r, MSE=mse)
    if _row_metric_weighting is not None:
        row['Pearson correlation (unweighted)'] = r_unweighted
        row['MSE (unweighted)'] = mse_unweighted
        row['metric_weighting'] = _row_metric_weighting
    # Requirements.md (BLUEs/BLUPs accuracy-metrics phase) - the naive
    # ensemble is scored across the WHOLE assembled result_test, one
    # `data` group per (population, ratio, phenotype, sample) - see
    # `ensemble()`'s own updated groupby just below, which now also keeps
    # the 'id' column this needs. `asreml_accuracy_ctx_resolver`
    # (`None` by default - a strict no-op, identical to before this
    # feature existed) rebuilds THIS group's own accuracy-metrics context
    # on demand (genomic_prediction.py's own `_asreml_accuracy_ctx_
    # resolver()` - see its docstring for the caching/refit rule).
    #
    # Requirements.md item 4 (multi-option update): Spearman correlation
    # and Top-k Coincidence are now ALWAYS computed (independent of
    # ASReml), from THIS group's own `actual` column - the very target
    # this row's Pearson correlation is scored against - so `_ctx_for_row`
    # below is built even when no ASReml adjusted-accuracy ctx resolved
    # (`enabled=False` then means "no Heritability/Adjusted accuracy
    # columns", never "no ranking columns"). `metric_directions`
    # (`{phenotype: 'higher'|'lower'}`, GP()'s own resolved dict) is the
    # SAME direction source the per-model rows use, so the naive
    # ensemble's Top-k rows can never disagree with them.
    if 'id' in data.columns:
        _ctx_for_row = dict(_ctx_for_weights) if _ctx_for_weights is not None else {'enabled': False}
        _phenotype = data.iloc[0].get('phenotype')
        _direction = (metric_directions or {}).get(
            _phenotype, _ctx_for_row.get('direction', 'higher'))
        _ctx_for_row['direction'] = _direction
        _ctx_for_row['actual_values'] = {
            str(_id): _val for _id, _val in zip(data['id'].tolist(), data['actual'].tolist())
        }
        row.update(compute_accuracy_metrics_row(
            data['ensemble'].tolist(), data['id'].tolist(), r, _ctx_for_row,
        ))
    elif _ctx_for_weights is not None:
        row.update(compute_accuracy_metrics_row(
            data['ensemble'].tolist(), [], r, _ctx_for_weights,
        ))
    return pd.Series(row)


# Backward-compatible alias - `metric` had no other in-tree caller before
# this feature existed, but is kept importable under its old name in case
# anything external imports it directly (mirrors the ver4-9
# leading-underscore alias precedent in metric_summary.py).
def metric(data):
    return _metric(data)


def ensemble(train, valid, test, effect, MODEL, interactions=None, asreml_accuracy_ctx_resolver=None,
             metric_directions=None):
    # Load prediction result from individual prediction models
    result_train = train
    result_valid = valid
    result_test = test
    
    model_selected = MODEL.copy()
    if 'ensemble' in MODEL:
        model_selected.remove('ensemble')
    
    # Arithmetic mean calculation & perfomrance metrics
    result_test['ensemble'] = result_test.loc[:,model_selected].mean(axis=1)
    if valid.shape[0] != 0:
        result_valid['ensemble'] = result_valid.loc[:,model_selected].mean(axis=1)
    result_train['ensemble'] = result_train.loc[:,model_selected].mean(axis=1)
    
    # Requirements.md (BLUEs/BLUPs accuracy-metrics phase): 'id' is now
    # ALSO selected (it was previously dropped before grouping) - needed
    # by `_metric()`'s own Spearman/Top-k computation, which must match
    # each individual's ensembled prediction to its own reference value.
    # `groupby(...)` itself is unaffected - still grouped by
    # `result_test.columns[1:5]` (population, ratio, phenotype, sample)
    # exactly as before.
    #
    # ver4-25, R8 bugfix: this used to be a single `.groupby(group_cols,
    # as_index=False).apply(lambda _g: _metric(_g, resolver))` call.
    # Newer pandas (2.2+, and unconditionally from pandas 3.0 on) no
    # longer includes the GROUPING columns themselves in the sub-frame
    # `_g` handed to the applied function (pandas' own `include_groups`
    # behaviour change - `include_groups=True` is rejected outright as of
    # pandas 3.0). Since `_metric()` reads this scenario's own
    # population/phenotype/ratio/sample straight off `_g.iloc[0]` (to
    # resolve its accuracy-metrics/BLUE-weighting context), those four
    # columns silently going missing meant EVERY new column this function
    # can add (Heritability/Spearman/Top-k, and the three BLUE-weighting
    # columns) resolved to nothing for the naive ensemble's own row on
    # such a pandas version - never raising, just silently degrading, the
    # exact failure mode this project always treats as worse than a loud
    # error. Iterating the groups explicitly and copying each group key
    # back onto its own sub-frame before calling `_metric()` makes this
    # independent of that pandas version behaviour (I11: identical output
    # shape/columns to the previous `as_index=False` result either way).
    _record_group_cols = list(result_test.columns[1:5])
    _record_rows = []
    for _group_key, _g in result_test.loc[:, ['id'] + list(result_test.columns[1:6]) + ['ensemble']].groupby(
        _record_group_cols, sort=False,
    ):
        _group_key_tuple = _group_key if isinstance(_group_key, tuple) else (_group_key,)
        _g = _g.copy()
        for _col, _val in zip(_record_group_cols, _group_key_tuple):
            # `[_val] * len(_g)` (never a bare scalar assignment) - `_val`
            # itself may be a tuple (the 'ratio' group key, e.g.
            # `(0.5, 0.25, 0.25)`), which pandas would otherwise try to
            # broadcast element-wise against `_g`'s own row count instead
            # of filling every row with that one tuple.
            _g[_col] = [_val] * len(_g)
        _row = dict(zip(_record_group_cols, _group_key_tuple))
        _row.update(_metric(_g, asreml_accuracy_ctx_resolver, metric_directions).to_dict())
        _record_rows.append(_row)
    record = pd.DataFrame(_record_rows).reset_index(drop=True)
    record = record.rename(columns={"Pearson": "Pearson correlation"})
    record['model'] = 'ensemble'
    
    effect = effect.reset_index(drop=True)

    # Extract genomic marker effects
    effect = pd.concat([effect.iloc[:,:5],
                        _safe_row_normalize(effect.iloc[:,5:].abs().reset_index(drop=True))
                       ], axis=1)
    
    effect_ensemble = pd.DataFrame()
    cnt = 0
    for kkk in range(len(model_selected)):
        selected = effect[effect['model']==model_selected[kkk]].reset_index(drop=True)
        if selected.shape[0] != 0:
            if effect_ensemble.shape[0] == 0:
                effect_ensemble = selected
                cnt += 1
            else:
                effect_ensemble.iloc[:,5:] += selected.iloc[:,5:] 
                cnt += 1
    
    if cnt > 0:
        effect_ensemble.iloc[:,5:] = effect_ensemble.iloc[:,5:] / cnt
        effect_ensemble['model'] = 'ensemble'

    # Requirement_patch3.md item 1: the naive-ensemble analogue of the
    # marker-EFFECT combination above, for marker-PAIR interactions -
    # normalises away each contributing model's own interaction-value
    # scale, then takes the equal-weight (naive) mean across whichever
    # selected models actually reported an interaction for each task -
    # see models/interaction_extraction.py::naive_ensemble_interactions()
    # for the full contract. `interactions` is optional (defaults to
    # None) purely so this signature stays callable exactly as before
    # for any caller that has no interaction data to offer; every actual
    # caller in this codebase (genomic_prediction.py's finalisation
    # block, intra_task_parallel.py's model-fan-out equivalent) always
    # passes its own accumulated `interactions` DataFrame.
    if interactions is not None and interactions.shape[0] != 0:
        interaction_ensemble = naive_ensemble_interactions(interactions, model_selected)
    else:
        interaction_ensemble = pd.DataFrame(columns=_INTERACTION_ENSEMBLE_SCHEMA)

    return result_train, result_valid, result_test, record, effect_ensemble, interaction_ensemble

