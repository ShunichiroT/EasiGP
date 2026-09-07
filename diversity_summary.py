"""
diversity_summary.py
=====================
Update ID ver4-9, R5 (design blueprint SS2.2).

Reports the four Diversity Prediction Theorem (DPT) quantities Tomura et
al. 2026 (`diag010.pdf` Eq. 1 p.7, Eq. 4 p.7) define for an ensemble
prediction, for EVERY ensemble label found in a run's own `Metric.csv` -
the naive (equal-weight) ensemble INCLUDED as a first-class row, not a
special case (see `resolve_ensemble_members()`'s own docstring for why).

Computed ENTIRELY from three files a run has already written to disk
(`Prediction_result_test.csv`, `Weight.csv`, `Metric.csv`) - `GP()` is not
modified, no model is re-fit, and the test split is not touched by any
fit a second time (I7): this module only reads predictions and weights
that were already made/computed.

Per scenario `(population, phenotype, ratio, sample)`, per ensemble
label, over the TEST split, with `w_i` normalised so `sum(w_i) == 1`,
`M_i` = model i's predicted vector, `V` = actual, `M_w = sum(w_i * M_i)`
(the WEIGHTED mean - see below for why), and `<.>` = mean over
individuals in the split:

    first_term  = <(M_w - V)^2>                    Many-Model error   (Eq. 1, term 1)
    second_term = sum_i w_i * <(M_i - V)^2>         Average-Model error (Eq. 1, term 2)
    third_term  = sum_i w_i * <(M_i - M_w)^2>       Diversity of model predictions (Eq. 1, term 3)
    ratio_term  = third_term / second_term          Eq. 4, p.7 (blank when second_term == 0)

Defining `M_w` as the WEIGHTED mean (not the naive, equal-weight mean
`models/Nelder_Mead.py::_raw_objective` uses internally) is the design
decision that makes `first_term == second_term - third_term` an EXACT
algebraic identity for weighted ensembles as well as naive ones - the
weighted bias/ambiguity decomposition. Setting `w_i = 1/N` reduces it to
Eq. 1 verbatim, and makes `first_term` identically equal to that
scenario's own naive-ensemble MSE (`Metric.csv`'s own MSE for the
`ensemble`/`ensemble__<algo>` model) - both are enforced as a FATAL
`ValueError` (the identity) and checked by Phase 3 (the naive-MSE
equality), never silently allowed to drift.

`models/Nelder_Mead.py`, `models/Bayesian_optimisation.py`,
`models/Linear_transformation.py` and every other weight-optimiser file
are NOT modified, and `_raw_objective` is NOT reused: it computes
DPT-shaped terms on the VALIDATION split, using the EQUAL-weight mean as
`M_w` even for unequal candidate weights, and divides by both `w_sum` and
`N` - so its own internal scalar is not on the scale of a realised test
MSE, and the numbers this module reports are NOT expected to equal it
(RK-5). Changing that normalisation would change which weights every
future run selects - i.e. would change users' results - which is out of
scope for a reporting-only requirement and would be a silent scientific
change.

Layout: the same block-stacked "summary" sheet layout R1-R4 introduced in
`metric_summary.py` (`pivot_data_rows`/`pivot_block_rows`/
`all_pivot_rows`/`write_pivot_worksheet`/`write_pivot_csv`, and the
`six_statistics()` estimator itself) is REUSED here, not reimplemented -
a lateral (I2), not upward, import: this module is a sibling of
`metric_summary.py` at the same layer, and imports only `pandas`,
`numpy`, `metric_summary` and `pipeline_utils` - never `main_app.py`,
`genomic_prediction.py`, or `streamlit`.

`build_dpt_terms()`/`build_dpt_summary()` are PURE (frames in, frames
out - no disk access, no printing beyond diagnostic `print()` calls used
by every other module in this codebase for the same purpose).
`write_dpt_summary()` is the only function here that touches disk, and
the only one that imports `openpyxl` - lazily, wrapped exactly like
`metric_summary.write_metric_summary()`, degrading to named CSVs with a
loud warning rather than a fatal error (I12).

`Diversity_prediction_theorem.xlsx` (or its CSV degrade) is deliberately
NOT added to `checkpoint_utils.RESULT_FILE_NAMES` - like
`Metric_summary.xlsx`, it is a reporting artefact derived from files
already on disk, not one of `GP()`'s own accumulators, and is fully
regenerable at any time (I8/I9).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from metric_summary import (
    PIVOT_STATISTICS, six_statistics, write_pivot_worksheet, write_pivot_csv,
)
from pipeline_utils import canonical_model_order, _is_naive_ensemble_label, _is_wopt_label

# ---------------------------------------------------------------------------
# Fixed vocabulary
# ---------------------------------------------------------------------------

def dpt_model_order(model_labels):
    """Requirements.md item 1: `terms` (as returned by `build_dpt_terms()`)
    only ever contains rows for ensemble labels - `resolve_ensemble_members()`
    is the sole gate, and it only ever admits naive-ensemble
    (`_is_naive_ensemble_label()`) and weighted-ensemble
    (`_is_wopt_label()`) labels, since a single-prediction model has no
    'other models' to diversify against and the terms are identically
    undefined for it. `build_dpt_summary()`'s own pivot construction,
    though, `reindex()`s every block's columns onto whatever `model_order`
    it is given - so passing it the SAME full `model_labels` list every
    other summary/plot call site uses (real models + naive ensemble +
    weighted methods, e.g. `cfg['MODEL'] + cfg['W_OPT']`) makes every
    single-prediction model reappear as its own column, filled entirely
    blank (there is no row for it in `terms` to reindex onto). Call this on
    that same `model_labels` list before handing it to `write_dpt_summary()`
    (as `model_order=`) to drop those always-blank columns while keeping
    the real models' own relative ordering among the ensemble labels intact
    - `build_dpt_summary()`'s own `canonical_model_order()` call still runs
    on the result exactly as before, so the naive ensemble still sorts
    ahead of the weighted methods regardless of `model_labels`' own input
    order.

    Never raises: `None`/empty in -> `[]` out, same as an all-single-model
    run with no ensemble label at all (`write_dpt_summary()` already
    handles that case with its own informational message).
    """
    if not model_labels:
        return []
    return [m for m in model_labels if _is_naive_ensemble_label(m) or _is_wopt_label(m)]


DPT_TERMS = ('first_term', 'second_term', 'third_term', 'ratio_term')

DPT_TERM_TITLES = {
    'first_term': 'Many-Model error (first term)',
    'second_term': 'Average-Model error (second term)',
    'third_term': 'Diversity of model predictions (third term)',
    'ratio_term': 'Diversity over average-model error (third/second term)',
}

# Same three aggregation levels R1-R4's build_metric_summary() reports at
# - 'population' is kept WHOLE for 'between' scenarios throughout this
# module (never split on '->'), matching metric_summary.py's own
# documented choice (see its build_metric_summary() docstring): splitting
# it would silently average across different training populations.
DPT_SHEET_NAMES = ('phenotype', 'phenotype_population', 'phenotype_population_ratio')
DPT_LEVEL_KEYS = (('phenotype',), ('phenotype', 'population'), ('phenotype', 'population', 'ratio'))

# The 24 block titles the DPT 'summary' sheet carries - DPT_TERMS outer
# (in DPT_TERM_TITLES' own human-readable form), PIVOT_STATISTICS inner
# (the SAME six statistics, same estimator, same rounding/blank rules
# R1-R4 define - imported from metric_summary, never re-derived).
DPT_BLOCK_TITLES = tuple(f'{DPT_TERM_TITLES[t]} {s}' for t in DPT_TERMS for s in PIVOT_STATISTICS)

# The three long-format sheets report a NARROWER subset of the six
# statistics (median, mean, and the mean's own standard error) per term -
# the wide 'summary' sheet above is where the full six-statistic,
# 24-block breakdown lives; the long sheets stay closer to
# build_metric_summary()'s own precedent (median/mean only, no SE at
# all) while still surfacing ONE dispersion figure per term.
_DPT_LONG_STAT_SUFFIXES = ('median', 'mean', 'mean standard error')

DPT_SUMMARY_NOTE = (
    "Diversity Prediction Theorem terms (Tomura et al. 2026, diag010.pdf Eq. 1 p.7 / Eq. 4 p.7), "
    "reported on the TEST split using each ensemble's own WEIGHTED mean prediction (naive "
    "ensembles use equal weight, w_i = 1/N, over their own contributing models - see "
    "resolve_ensemble_members()). These values are NOT on the same scale as "
    "models/Nelder_Mead.py's internal optimisation objective (_raw_objective), which is "
    "evaluated on the VALIDATION split using an EQUAL-weight mean regardless of the candidate "
    "weight vector - the two are intentionally different quantities and are not expected to "
    "agree. 'ratio_term' (third/second) is blank, not infinite, whenever the second term is "
    "exactly zero (every contributing model predicted perfectly for that scenario)."
)

# The five metadata columns of Weight.csv / Metric.csv that are never
# themselves a contributing-model column.
_WEIGHT_METADATA_COLS = ('population', 'phenotype', 'model', 'ratio', 'sample')
# Prediction_result_test.csv's own six leading metadata columns (see
# architecture doc SS16 / genomic_prediction.py's own Store-the-results
# section) - everything else is a per-model prediction column.
_PREDICTION_TEST_METADATA_COLS = ('id', 'population', 'ratio', 'phenotype', 'sample', 'actual')

# Update ID ver4-9: the PREDICTION-COLUMN name each weighted-ensemble
# method writes into Prediction_result_test.csv is, for two of the four
# methods, a DIFFERENT string than its own record/effect/weight 'model'
# label (genomic_prediction.py's own `_WOPT_LABELS` dict -
# 'Nelder Mead' writes column 'Nelder-Mead' (hyphenated); 'Bayesian
# optimisation' writes column 'Bayesian' - architecture doc SS17's
# "naming inconsistencies ... never corrected opportunistically").
# `pipeline_utils._is_wopt_label()` matches the LABEL form only, so it
# does not, and must not be expected to, catch these column names -
# without this second, column-specific exclusion list, a weighted
# method's own prediction column would be misread as a real contributing
# BASE model for an UNSUFFIXED naive label sharing the same run (the
# per-label cross-check in resolve_ensemble_members() step 3 catches
# this too, but only when a same-tuning-group weighted label happens to
# also be present - this list closes the gap unconditionally, at the
# source, rather than relying solely on that cross-check).
_WOPT_COLUMN_PREFIXES = ('Linear transformation', 'Nelder-Mead', 'Bayesian', 'Analytic least-squares')


def _is_wopt_column(name) -> bool:
    return isinstance(name, str) and any(
        name == p or name.startswith(p + '__') for p in _WOPT_COLUMN_PREFIXES
    )

_TERM_SCENARIO_KEYS = ('population', 'phenotype', 'ratio', 'sample')


def resolve_ensemble_members(
    metric: "pd.DataFrame", prediction_test: "pd.DataFrame", weight: "pd.DataFrame",
) -> "dict[str, dict]":
    """Update ID ver4-9, R5 (design blueprint SS2.2.3): for every ensemble
    label present in `metric['model']` (naive - `ensemble`/
    `ensemble__<algo>` - AND weighted - `Linear transformation`,
    `Nelder Mead`, `Bayesian optimisation`, `Analytic least-squares`, or
    their own `__<algo>` variants), resolve exactly which OTHER models
    contributed to it. The naive ensemble is a FIRST-CLASS entry here,
    not a special case handled separately - it goes through the identical
    dict shape as every weighted label, differing only in
    `is_naive=True` (which tells `build_dpt_terms()` to synthesise
    `w_i = 1/N` rather than look a weight row up).

    Resolution order (logged once per label):

    1. WEIGHTED labels: the `Weight.csv` columns for that label's own
       rows that are neither the five metadata columns
       (`population, phenotype, model, ratio, sample`) nor themselves an
       ensemble label, AND that carry at least one non-NaN value across
       that label's own rows (a column can be entirely NaN for one
       label's rows purely because `Weight.csv` is the OUTER UNION of
       every weighted method's own columns - see `models/Nelder_Mead.py`
       et al., each of which writes only its own contributing models'
       columns for its own rows).
    2. NAIVE labels: the `Prediction_result_test.csv` columns that are
       neither metadata (`id, population, ratio, phenotype, sample,
       actual`), nor any ensemble label, NOR a weighted method's own
       PREDICTION-COLUMN name (`_WOPT_COLUMN_PREFIXES` - two of the four
       methods write a column name that differs from their record/model
       label, e.g. `'Nelder Mead'`'s column is `'Nelder-Mead'`, hyphenated
       - see that constant's own comment), and - when the label carries
       an `__<algo>` suffix - that carry the SAME suffix (an unsuffixed
       naive label instead EXCLUDES every suffixed model column). This
       reproduces `models.hyperparameter_tuning.ensemble_groups()`'s own
       partition without importing it, because under
       `HP_TUNE_ENSEMBLE_MODE='per_method'` the naive ensemble is built
       once per tuning group (`genomic_prediction.py`'s own dispatch, see
       architecture doc SS10).
    3. If a WEIGHTED label from the SAME tuning group (same `__<algo>`
       suffix, or both unsuffixed) also resolved, its own set is used to
       CROSS-CHECK the naive set; a mismatch is logged as a warning and
       the weighted label's set wins - it is the authoritative record of
       what was actually ensembled (`models/Nelder_Mead.py` et al. write
       their OWN contributing columns directly; the naive reconstruction
       above is an inference from column presence).
    4. Fewer than 2 contributing models for a label -> that label is
       skipped (with a warning) - the diversity term is identically 0 for
       `N == 1` and the row would be meaningless.

    Returns
    -------
    dict, keyed by ensemble label, each value `{'models': list[str],
    'is_naive': bool}`. A label present in `metric` but with fewer than 2
    resolved contributing models is ABSENT from this dict (see point 4
    above) - never present with an empty/singleton list.
    """
    members: "dict[str, dict]" = {}
    if metric is None or metric.shape[0] == 0 or 'model' not in metric.columns:
        return members

    ensemble_labels = [m for m in pd.unique(metric['model']).tolist()
                        if _is_naive_ensemble_label(m) or _is_wopt_label(m)]
    if not ensemble_labels:
        return members

    def _suffix_of(label: str) -> "str | None":
        return label.split('__', 1)[1] if '__' in label else None

    weighted_sets: "dict[str, list[str]]" = {}
    if weight is not None and weight.shape[0] != 0 and 'model' in weight.columns:
        for label in ensemble_labels:
            if not _is_wopt_label(label):
                continue
            rows = weight[weight['model'] == label]
            if rows.shape[0] == 0:
                continue
            candidates = [
                c for c in rows.columns
                if c not in _WEIGHT_METADATA_COLS
                and not (_is_naive_ensemble_label(c) or _is_wopt_label(c))
                and rows[c].notna().any()
            ]
            weighted_sets[label] = candidates

    naive_sets: "dict[str, list[str]]" = {}
    if prediction_test is not None and prediction_test.shape[0] != 0:
        for label in ensemble_labels:
            if not _is_naive_ensemble_label(label):
                continue
            suffix = _suffix_of(label)
            candidates = [
                c for c in prediction_test.columns
                if c not in _PREDICTION_TEST_METADATA_COLS
                and not (_is_naive_ensemble_label(c) or _is_wopt_label(c) or _is_wopt_column(c))
            ]
            if suffix is not None:
                candidates = [c for c in candidates if c.endswith(f'__{suffix}')]
            else:
                candidates = [c for c in candidates if '__' not in c]
            naive_sets[label] = candidates

    for label in ensemble_labels:
        if _is_wopt_label(label):
            models = weighted_sets.get(label, [])
            source = 'weighted'
        else:
            models = naive_sets.get(label, [])
            source = 'naive'
            suffix = _suffix_of(label)
            for w_label, w_models in weighted_sets.items():
                if _suffix_of(w_label) == suffix:
                    if set(w_models) != set(models):
                        print(f"[diversity_summary] WARNING: naive label {label!r}'s own "
                              f"contributing-model set (from Prediction_result_test.csv: "
                              f"{sorted(models)}) disagrees with weighted label {w_label!r}'s own "
                              f"set (from Weight.csv: {sorted(w_models)}) for the same tuning "
                              f"group - using {w_label!r}'s set, the authoritative record of what "
                              f"was actually ensembled.")
                        models = w_models
                    break

        if len(models) < 2:
            print(f"[diversity_summary] NOTE: ensemble label {label!r} resolved to fewer than 2 "
                  f"contributing model(s) ({models!r}) - skipped (the diversity term is "
                  f"identically 0 for N=1 and the row would be meaningless).")
            continue

        members[label] = {'models': models, 'is_naive': source == 'naive'}
        print(f"[diversity_summary] Resolved ensemble label {label!r} ({source}): "
              f"{len(models)} contributing model(s): {models}.")

    return members


def build_dpt_terms(
    prediction_test: "pd.DataFrame", weight: "pd.DataFrame", metric: "pd.DataFrame",
) -> "pd.DataFrame":
    """PURE. One row per `(population, phenotype, ratio, sample, model)`
    for every ensemble label `resolve_ensemble_members()` resolves (naive
    included - it goes through this IDENTICAL path, not a separate
    branch: `w_i = 1/N` is simply what gets substituted for its weight
    vector). Columns: `population, phenotype, ratio, sample, model,
    n_models, first_term, second_term, third_term, ratio_term`. Empty
    frame in (either `prediction_test` or `metric`) -> empty frame out;
    never raises for that reason.

    Raises
    ------
    ValueError : if, for any scenario/label, the DPT identity
        `first_term == second_term - third_term` is violated by more than
        `1e-9 * max(1.0, abs(second_term))` - an ARITHMETIC INVARIANT, not
        a data condition (see the module docstring for why `M_w` is
        defined as the weighted mean specifically so this identity always
        holds). A violation means the contributing-model set or the
        weights were resolved incorrectly for that scenario/label, and
        silently writing the resulting number would put a wrong value in
        a published table - this fails loudly instead, naming the
        scenario and label, mirroring `GP()`'s own precedent of raising a
        diagnostic-rich `ValueError` on an effect-width mismatch
        (architecture doc SS4.3).
    """
    columns = ['population', 'phenotype', 'ratio', 'sample', 'model', 'n_models',
               'first_term', 'second_term', 'third_term', 'ratio_term']
    if prediction_test is None or prediction_test.shape[0] == 0 or metric is None or metric.shape[0] == 0:
        return pd.DataFrame(columns=columns)

    members = resolve_ensemble_members(metric, prediction_test, weight)
    if not members:
        return pd.DataFrame(columns=columns)

    weight_indexed = None
    if weight is not None and weight.shape[0] != 0:
        _index_cols = list(_TERM_SCENARIO_KEYS) + ['model']
        weight_indexed = weight.set_index(_index_cols)

    rows: "list[dict]" = []
    _zero_second_term_count = 0

    for scenario_key, group in prediction_test.groupby(list(_TERM_SCENARIO_KEYS), sort=False, dropna=False):
        pop, phen, rat, samp = scenario_key
        actual = group['actual'].to_numpy(dtype=float)

        for label, info in members.items():
            model_cols = info['models']
            is_naive = info['is_naive']
            n_models = len(model_cols)

            missing_cols = [c for c in model_cols if c not in group.columns]
            if missing_cols:
                print(f"[diversity_summary] WARNING: scenario (population={pop!r}, "
                      f"phenotype={phen!r}, ratio={rat!r}, sample={samp!r}) is missing "
                      f"Prediction_result_test.csv column(s) {missing_cols} for ensemble label "
                      f"{label!r} - skipping this scenario for this label.")
                continue
            preds = group[model_cols].to_numpy(dtype=float)  # (n_individuals, n_models)

            if is_naive:
                w = np.full(n_models, 1.0 / n_models)
            else:
                if weight_indexed is None:
                    continue
                try:
                    wrow = weight_indexed.loc[(pop, phen, rat, samp, label)]
                except KeyError:
                    continue
                if isinstance(wrow, pd.DataFrame):
                    # Defensive only: should not occur in practice (one
                    # weight row per scenario per label), kept safe rather
                    # than crashing on an unexpected duplicate.
                    wrow = wrow.iloc[-1]
                w_raw = wrow.reindex(model_cols).to_numpy(dtype=float)
                w_raw = np.nan_to_num(w_raw, nan=0.0)
                w_sum = float(w_raw.sum())
                if not np.isfinite(w_sum) or w_sum == 0:
                    print(f"[diversity_summary] WARNING: scenario (population={pop!r}, "
                          f"phenotype={phen!r}, ratio={rat!r}, sample={samp!r}), label {label!r}: "
                          f"weight row sums to {w_sum!r} - skipping this scenario for this label "
                          f"(mirrors models/ensemble.py::_safe_row_normalize()'s own defence "
                          f"against an exact-zero row sum).")
                    continue
                if abs(w_sum - 1.0) > 1e-6:
                    print(f"[diversity_summary] NOTE: scenario (population={pop!r}, "
                          f"phenotype={phen!r}, ratio={rat!r}, sample={samp!r}), label {label!r}: "
                          f"weight row summed to {w_sum:.6f}, not 1 - renormalised.")
                w = w_raw / w_sum

            weighted_mean = preds @ w  # (n_individuals,)
            first_term = float(np.mean((weighted_mean - actual) ** 2))
            per_model_mse = np.mean((preds - actual[:, None]) ** 2, axis=0)
            second_term = float(np.dot(w, per_model_mse))
            per_model_div = np.mean((preds - weighted_mean[:, None]) ** 2, axis=0)
            third_term = float(np.dot(w, per_model_div))

            residual = abs(first_term - (second_term - third_term))
            tolerance = 1e-9 * max(1.0, abs(second_term))
            if residual > tolerance:
                raise ValueError(
                    f"[diversity_summary] DPT identity violated for scenario "
                    f"population={pop!r}, phenotype={phen!r}, ratio={rat!r}, sample={samp!r}, "
                    f"label={label!r}: first_term={first_term!r}, second_term={second_term!r}, "
                    f"third_term={third_term!r}, |first-(second-third)|={residual!r} > "
                    f"tolerance={tolerance!r}. This is an arithmetic invariant, not a data "
                    f"condition - it means the contributing-model set or the weights were "
                    f"resolved incorrectly for this scenario/label; refusing to write a "
                    f"silently-wrong number."
                )

            if second_term == 0:
                ratio_term = None
                _zero_second_term_count += 1
            else:
                ratio_term = third_term / second_term

            rows.append({
                'population': pop, 'phenotype': phen, 'ratio': rat, 'sample': samp,
                'model': label, 'n_models': n_models,
                'first_term': first_term, 'second_term': second_term,
                'third_term': third_term, 'ratio_term': ratio_term,
            })

    if _zero_second_term_count:
        print(f"[diversity_summary] NOTE: {_zero_second_term_count} scenario/label row(s) had a "
              f"second_term of exactly 0 (every contributing model predicted perfectly) - "
              f"'ratio_term' left blank for those rows rather than reporting infinity.")

    return pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)


def build_dpt_summary(
    terms: "pd.DataFrame", model_order: "list[str] | None" = None,
) -> "tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]":
    """PURE. Returns `(long_sheets, pivot_blocks)`:

    `long_sheets` - keyed by `DPT_SHEET_NAMES`, one row per level-key x
        model, columns `<level keys...>, model, n`, then
        `f'{term} {stat}'` for every `term` in `DPT_TERMS` and `stat` in
        `_DPT_LONG_STAT_SUFFIXES` (median/mean/mean standard error) -
        `n` is the plain row count for that group (matching
        `metric_summary.build_metric_summary()`'s own `n` convention,
        never `n_eff`).
    `pivot_blocks` - keyed by `DPT_BLOCK_TITLES` (the 24 block titles),
        phenotype-indexed, ensemble-model-columned, using the SAME six
        statistics `metric_summary.six_statistics()` computes (identical
        estimator, identical rounding, identical blank rules - imported,
        not re-derived).

    `model_order` : the exact ensemble-label column order for
        `pivot_blocks` (and the model row order within each `long_sheets`
        level) - falls back to `pd.unique(terms['model'])` (order of
        first appearance) when not given, exactly like
        `build_metric_summary_pivot()`. `pipeline_utils.
        canonical_model_order()` is applied on top either way.

    Empty `terms` in -> `({}, {})` out.
    """
    long_sheets: "dict[str, pd.DataFrame]" = {}
    pivot_blocks: "dict[str, pd.DataFrame]" = {}
    if terms is None or terms.shape[0] == 0:
        return long_sheets, pivot_blocks

    if model_order is None:
        _order = pd.unique(terms['model']).tolist()
    else:
        _seen = set(model_order)
        _order = list(model_order) + [m for m in pd.unique(terms['model']) if m not in _seen]
    _order = canonical_model_order(_order)
    model_rank = {m: i for i, m in enumerate(_order)}

    phenotype_order = pd.unique(terms['phenotype']).tolist()

    for sheet_name, keys in zip(DPT_SHEET_NAMES, DPT_LEVEL_KEYS):
        group_cols = list(keys) + ['model']
        grouped_level = terms.groupby(group_cols, as_index=False, sort=False, dropna=False)
        wide = grouped_level.agg(n=('first_term', 'size'))

        _grouped_for_stats = terms.groupby(group_cols, sort=False, dropna=False)
        for term in DPT_TERMS:
            stats = six_statistics(_grouped_for_stats, term)
            for stat_name in _DPT_LONG_STAT_SUFFIXES:
                col_name = f'{term} {stat_name}'
                merge_frame = stats[stat_name].rename(col_name).reset_index()
                wide = wide.merge(merge_frame, on=group_cols, how='left')

        sort_keys = wide[list(keys)].astype(str).copy()
        sort_keys['_model_rank'] = wide['model'].map(model_rank)
        order = sort_keys.sort_values(by=list(keys) + ['_model_rank'], kind='mergesort').index
        wide = wide.loc[order].reset_index(drop=True)

        col_order = list(keys) + ['model', 'n'] + [
            f'{t} {s}' for t in DPT_TERMS for s in _DPT_LONG_STAT_SUFFIXES
        ]
        long_sheets[sheet_name] = wide[col_order]

    grouped_pm = terms.groupby(['phenotype', 'model'], sort=False, dropna=False)
    for term in DPT_TERMS:
        stats = six_statistics(grouped_pm, term)
        for stat in PIVOT_STATISTICS:
            series = stats[stat]
            pivot = series.unstack('model')
            pivot = pivot.reindex(index=phenotype_order, columns=_order)
            pivot.index.name = 'phenotype'
            pivot.columns.name = 'model'
            pivot_blocks[f'{DPT_TERM_TITLES[term]} {stat}'] = pivot

    return long_sheets, pivot_blocks


def write_dpt_summary(
    terms: "pd.DataFrame", RESULT_NAME: str, *, create: bool = True,
    model_order: "list[str] | None" = None,
) -> "list[str]":
    """Writes `Result/<RESULT_NAME>/Diversity_prediction_theorem.xlsx` -
    'summary' sheet FIRST (the 24 blocks, `DPT_BLOCK_TITLES`), then the
    three long sheets (`DPT_SHEET_NAMES`) - mirroring
    `metric_summary.write_metric_summary()`'s own layout exactly.
    Degrades to `Diversity_prediction_theorem.csv` (the pivot table) plus
    three named CSVs when `openpyxl` is unavailable. Returns the paths
    written; `[]` when nothing was.

    `create=False` (the `DPT_SUMMARY_CREATE` config key): nothing
    written, nothing logged - the same `*_CREATE` precedent
    `metric_summary.write_metric_summary()` already follows.

    `terms` empty (no ensemble label with >= 2 contributing models was
    found, or the source files had nothing to report): nothing written,
    one informational line logged - never fatal.

    This reports on the TEST split with the WEIGHTED mean, and is
    deliberately NOT on the same scale as `models/Nelder_Mead.py::
    _raw_objective`'s own internal value - see the module docstring and
    `DPT_SUMMARY_NOTE` (appended as a trailing row of the 'summary'
    sheet/CSV) for the full explanation.
    """
    if not create:
        return []
    if terms is None or terms.shape[0] == 0:
        print("[diversity_summary] No ensemble label with 2 or more contributing models was "
              "found in this run (or Prediction_result_test.csv/Weight.csv/Metric.csv had "
              "nothing to report) - no Diversity Prediction Theorem summary written.")
        return []

    long_sheets, pivot_blocks = build_dpt_summary(terms, model_order=model_order)
    _pivot_model_order = model_order if model_order is not None else pd.unique(terms['model']).tolist()
    _pivot_model_order = canonical_model_order(_pivot_model_order)
    result_dir = './Result/' + RESULT_NAME + '/'
    os.makedirs(result_dir, exist_ok=True)

    xlsx_path = result_dir + 'Diversity_prediction_theorem.xlsx'
    try:
        # Lazy import, same guarded pattern as metric_summary.py - see
        # that module's own docstring for why openpyxl==3.1.5 being
        # pinned in both environment_*.yml files still doesn't make this
        # import unconditional (I12).
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            for sheet_name in DPT_SHEET_NAMES:
                long_sheets[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)
            if pivot_blocks:
                write_pivot_worksheet(writer.book, pivot_blocks, _pivot_model_order, sheet_name='summary',
                                       block_titles=DPT_BLOCK_TITLES, note=DPT_SUMMARY_NOTE)
        print(f"[diversity_summary] Diversity Prediction Theorem summary written: {xlsx_path}")
        return [xlsx_path]
    except (ImportError, ValueError):
        csv_paths = []
        if pivot_blocks:
            pivot_path = f'{result_dir}Diversity_prediction_theorem.csv'
            write_pivot_csv(pivot_blocks, _pivot_model_order, pivot_path,
                             block_titles=DPT_BLOCK_TITLES, note=DPT_SUMMARY_NOTE)
            csv_paths.append(pivot_path)
        for sheet_name in DPT_SHEET_NAMES:
            path = f'{result_dir}Diversity_prediction_theorem_{sheet_name}.csv'
            long_sheets[sheet_name].to_csv(path, index=False)
            csv_paths.append(path)
        print("[diversity_summary] WARNING: openpyxl is not installed in this environment, so")
        print(f"[diversity_summary] {xlsx_path} could not be written. The same tables were")
        print("[diversity_summary] written as CSV instead: Diversity_prediction_theorem.csv (the "
              "pivot table),")
        print("[diversity_summary] Diversity_prediction_theorem_phenotype.csv,")
        print("[diversity_summary] Diversity_prediction_theorem_phenotype_population.csv,")
        print("[diversity_summary] Diversity_prediction_theorem_phenotype_population_ratio.csv")
        print("[diversity_summary] To get the Excel file, install openpyxl into this environment:")
        print("[diversity_summary]     conda install -c conda-forge openpyxl      (or: pip install openpyxl)")
        return csv_paths
