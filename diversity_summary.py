"""
diversity_summary.py
=====================
Update ID ver4-9, R5 (design blueprint SS2.2); Update ID ver4-11, R6
(Requirements.md items 1-2).

Reports the Diversity Prediction Theorem (DPT) quantities Tomura et al.
2026 (G3 16(6):jkag090, "Ensemble-based genomic prediction for maize
flowering time ...", the article shipped alongside this update - Eq. (4)
p.4 for the three terms, Eq. (5) p.4 for the coefficient of variation)
define for an ensemble prediction, for EVERY ensemble label found in a
run's own `Metric.csv` - the naive (equal-weight) ensemble INCLUDED as a
first-class row, not a special case (see `resolve_ensemble_members()`'s
own docstring for why).

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

    first_term  = <(M_w - V)^2>                    Many-Model error    (Eq. 4, term 1)
    second_term = sum_i w_i * <(M_i - V)^2>         Average-Model error (Eq. 4, term 2)
    third_term  = sum_i w_i * <(M_i - M_w)^2>       Diversity of model predictions (Eq. 4, term 3)
    ratio_term  = third_term / second_term          EasiGP-internal only - NOT in the article, see below

Defining `M_w` as the WEIGHTED mean (not the naive, equal-weight mean
`models/Nelder_Mead.py::_raw_objective` uses internally) is the design
decision that makes `first_term == second_term - third_term` an EXACT
algebraic identity for weighted ensembles as well as naive ones - the
weighted bias/ambiguity decomposition (Krogh and Vedelsby 1995), a strict
generalisation of Eq. (4) to an arbitrary weight vector that sums to 1.
Setting `w_i = 1/N` reduces it to Eq. (4) verbatim, and makes `first_term`
identically equal to that scenario's own naive-ensemble MSE (`Metric.csv`'s
own MSE for the `ensemble`/`ensemble__<algo>` model) - both are enforced
as a FATAL `ValueError` (the identity) and checked by Phase 3 (the
naive-MSE equality), never silently allowed to drift.

Requirements.md item 1 - investigation and resolution
-------------------------------------------------------
Item 1 asked whether the DPT calculation is "truly correct", flagging
that `ratio_term` (third/second) is HIGHER for the naive ensemble than
for a weighted ensemble, and higher even in scenarios where the
individual models are LESS diverse - both counterintuitive if
`ratio_term` is read as "the" diversity measure.

The three terms themselves ARE correct: the algebraic identity
`first_term == second_term - third_term` is derived directly from the
definition of `M_w` above (it holds for ANY weight vector summing to 1 -
verified analytically and enforced at runtime in `build_dpt_terms()`
below, never silently allowed to drift), and reduces exactly to Eq. (4)
when weights are equal. There is no bug in `first_term`/`second_term`/
`third_term`.

`ratio_term`, however, is NOT a quantity the article defines (it is an
EasiGP-internal addition, and the previous revision of this module
mis-cited it as being from the DPT equation itself) and it is
mathematically the wrong tool for the two comparisons item 1 makes,
for two independent reasons that were confirmed numerically:

1. A weighted ensemble's own `w_i` down-weights whichever models
   contribute most to `third_term` at the SAME time as it down-weights
   their contribution to `second_term`; concentrating weight onto a
   subset of models shrinks the weighted ambiguity term (`third_term`)
   from ANY underlying model set, diverse or not, simply because the
   models carrying the largest deviations from `M_w` are exactly the
   ones a good optimiser assigns the least weight. A naive ensemble
   cannot do this (`w_i` is fixed at `1/N`), so its `third_term` - and
   hence `ratio_term` - is systematically larger for the SAME
   underlying models. This alone explains the "naive > weighted"
   observation without any of the individual models being more or less
   diverse.
2. For models whose prediction errors are of similar magnitude,
   `third_term / second_term` under equal weighting tends toward
   `(N-1)/N` (a function of the CONTRIBUTING-MODEL COUNT `N` only),
   essentially independent of how large the disagreement between models
   actually is - so `ratio_term` alone cannot be trusted to rank
   "more diverse" against "less diverse" scenarios either.

Neither of these is a computational error - they are real properties of
a within-scenario RATIO of two already-averaged quantities. This is
precisely why the article does not use such a ratio to compare diversity
levels at all: it instead reports the MEAN and the COEFFICIENT OF
VARIATION (CV, Eq. (5) p.4) of each of the three raw terms, computed
across a run's own replicate prediction scenarios for one fixed
weighting scheme - never as a ratio, and never by comparing two
DIFFERENT weight vectors against each other. Requirements.md item 2's CV
addition (`DPT_CV_TERMS`, `cv_statistic()`, the `<term> CV` columns/
blocks below) is that metric, implemented verbatim from Eq. (5), and is
the scientifically-supported way to compare diversity levels - within
one ensemble label, across scenarios/phenotypes - that this module now
provides. `ratio_term` itself is left in place (it remains an
arithmetically valid quantity some readers may still find useful as a
per-scenario diagnostic) but is no longer mis-cited as being defined by
the DPT equation, and this docstring is the authoritative explanation for
why it should not be used, by itself, to compare diversity ACROSS
ensemble labels with different weight vectors.

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

ver4-95 - why a weighted label's terms are smaller than the naive ensemble's
-------------------------------------------------------------------------------
Reported again as "the naive ensemble's terms are on a different scale from the weighted
ones". Re-checked: the arithmetic is the same for every label (`w_i = 1/N` for the naive one,
the `Weight.csv` row for a weighted one, one code path), and with identical weights the two give
identical terms. The scale differs because the weights differ: a weighted ensemble (the Super
learner above all, whose cross-validated weights collapse onto the strongest members) sits close
to its heaviest members, so both `second_term` (the weighted average member error) and
`third_term` (the weighted spread of members around `M_w`) fall - `third_term` fastest - while
`first_term` stays near the naive value. The long sheets now report `n_models` and
`n_effective = 1 / sum(w_i^2)` next to the terms so this can be read off directly. Two things
CAN make the comparison unlike-for-like, and are now logged (nothing is recomputed): the two
labels combining different model sets (e.g. the Super learner leaves bio-prior models out), and a
label whose own prediction is not the plain weighted sum of the Step 9 member columns in
`Prediction_result_test.csv` (Super learner refitted on train+valid or pooled, Iterative
ensemble, Linear-transformation network) - for those `first_term` differs from the MSE of the
label's own column, and the NOTE says so.

Requirements.md item 2 - coefficient of variation (Eq. 5, p.4)
------------------------------------------------------------------
    CV_hat_termj = sigma_termj / mu_termj

where `sigma_termj`/`mu_termj` are the (sample, ddof=1) standard
deviation and mean of term `j` - `first_term`, `second_term` or
`third_term` (`DPT_CV_TERMS`; `ratio_term` is excluded, since it is not
one of the article's own three DPT terms - see the note on `ratio_term`
above) - taken across every `(population, phenotype, ratio, sample)`
prediction scenario within whatever group is being summarised (the SAME
rows `six_statistics()` already summarises with `median`/`mean` for that
group - `cv_statistic()` below is a sibling estimator over the identical
grouping, not a re-derivation). Matches the article's own usage exactly
at the phenotype-only level ("The CV was estimated per trait for each
dataset") - the wide 'summary' sheet's pivot blocks are grouped by
`['phenotype', 'model']` alone, precisely that level; the three
long-format sheets additionally report it at the finer
phenotype+population and phenotype+population+ratio levels this module
already breaks every other statistic out at, which is a superset of the
article's own reporting granularity, not a departure from it. A blank
(`NaN`) CV means fewer than 2 scenarios contributed, or the term's mean
was exactly 0 in that group (every contributing model predicted every
individual perfectly) - never a silently wrong number.

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
    apply_label_style_to_tables,
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

# Requirements.md item 2: the three terms Eq. (5) p.4 actually defines a
# CV for - the article's own equation is over "term j" of Eq. (4), which
# has exactly three terms. `ratio_term` is an EasiGP-internal addition
# (see the module docstring's "Requirements.md item 1" section for why it
# is not a safe cross-ensemble diversity measure) and is deliberately
# excluded here - a CV of a ratio-of-terms is not the quantity Eq. (5)
# defines and would invite exactly the same mis-citation this update
# fixes elsewhere in this module.
DPT_CV_TERMS = ('first_term', 'second_term', 'third_term')

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

# The first 24 of the DPT 'summary' sheet's block titles - DPT_TERMS
# outer (in DPT_TERM_TITLES' own human-readable form), PIVOT_STATISTICS
# inner (the SAME six statistics, same estimator, same rounding/blank
# rules R1-R4 define - imported from metric_summary, never re-derived).
# 3 more (Requirements.md item 2, Eq. (5) CV) are appended below, making
# 27 in total.
DPT_BLOCK_TITLES = tuple(f'{DPT_TERM_TITLES[t]} {s}' for t in DPT_TERMS for s in PIVOT_STATISTICS)

# Requirements.md item 2: three more blocks, ONE per DPT_CV_TERMS, each
# the Eq. (5) coefficient of variation (`cv_statistic()` below) - APPENDED
# after the original 24 (never inserted mid-list), so every pre-existing
# block keeps its own position - the same append-only discipline
# `HPARAMETERS` fields, `RESULT_FILE_NAMES`, etc. already follow elsewhere
# in this codebase (architecture doc SS4.4).
DPT_CV_BLOCK_TITLES = tuple(f'{DPT_TERM_TITLES[t]} CV' for t in DPT_CV_TERMS)
DPT_BLOCK_TITLES = DPT_BLOCK_TITLES + DPT_CV_BLOCK_TITLES

# The three long-format sheets report a NARROWER subset of the six
# statistics (median, mean, and the mean's own standard error) per term,
# PLUS (Requirements.md item 2) the Eq. (5) coefficient of variation for
# the three terms in DPT_CV_TERMS - the wide 'summary' sheet above is
# where the full six-statistic, 27-block breakdown lives; the long sheets
# stay closer to build_metric_summary()'s own precedent (median/mean only,
# no SE at all) while still surfacing ONE dispersion figure (mean SE) and
# ONE relative-diversity figure (CV) per term.
_DPT_LONG_STAT_SUFFIXES = ('median', 'mean', 'mean standard error')

# ver4-95: (terms column, aggregation) of the two trailing diagnostic columns of the long
# sheets - 'n_models median' and 'n_effective mean' (effective number of models, 1 / sum(w^2)).
_DPT_LONG_EXTRA_COLUMNS = (('n_models', 'median'), ('n_effective', 'mean'))

#DPT_SUMMARY_NOTE = (
#    "Diversity Prediction Theorem terms (Tomura et al. 2026, G3 16(6):jkag090, Eq. (4) p.4 for "
#    "the first/second/third term, Eq. (5) p.4 for the coefficient of variation 'CV' blocks/"
#    "columns), reported on the TEST split using each ensemble's own WEIGHTED mean prediction "
#    "(naive ensembles use equal weight, w_i = 1/N, over their own contributing models - see "
#    "resolve_ensemble_members()). These values are NOT on the same scale as "
#    "models/Nelder_Mead.py's internal optimisation objective (_raw_objective), which is "
#   "evaluated on the VALIDATION split using an EQUAL-weight mean regardless of the candidate "
#    "weight vector - the two are intentionally different quantities and are not expected to "
#    "agree. 'ratio_term' (third/second) is an EasiGP-internal addition, NOT defined by the "
#    "article, and is blank, not infinite, whenever the second term is exactly zero (every "
#    "contributing model predicted perfectly for that scenario); it should NOT, by itself, be "
#    "used to compare diversity levels across ensemble labels that use different weight vectors "
#    "(e.g. naive vs weighted) - see this module's own docstring ('Requirements.md item 1') for "
#   "why, and use the 'CV' blocks/columns instead, exactly as Eq. (5) defines them. A blank "
#    "'CV' cell means fewer than 2 contributing scenarios, or a mean of exactly 0, in that group."
#)

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
# ver4-26, R2/R_last item 6: 'Joint Bayesian' appended - the new
# JOINT_W_OPT combination method's own prediction-column name
# (genomic_prediction.py's new Step 11 `elif` branch writes
# `result_{test,valid,train}_sample['Joint Bayesian'] = ...`), distinct
# from its record/effect/weight 'model' LABEL ('Joint Bayesian
# optimisation' - see pipeline_utils._WOPT_LABEL_PREFIXES), matching the
# SAME label-vs-column distinction 'Bayesian optimisation'/'Bayesian'
# already establishes just above.
# ver4-57, R2: 'Iterative ensemble' appended - the opt-in iterative ensemble's prediction column
# carries the same name as its record/effect/weight label (genomic_prediction.py Step 11c).
# ver4-77, R2: 'Super learner' appended - the opt-in Super learner's prediction column carries the
# same name as its record/effect/weight label (genomic_prediction.py Step 11d).
_WOPT_COLUMN_PREFIXES = (
    'Linear transformation', 'Nelder-Mead', 'Bayesian', 'Analytic least-squares', 'Joint Bayesian',
    'Iterative ensemble', 'Super learner',
)


def _is_wopt_column(name) -> bool:
    return isinstance(name, str) and any(
        name == p or name.startswith(p + '__') for p in _WOPT_COLUMN_PREFIXES
    )


# ver4-95: record/weight 'model' LABEL -> Prediction_result_test.csv COLUMN, for the two weighted
# methods whose column name differs from their label (see _WOPT_COLUMN_PREFIXES above).
_WOPT_LABEL_TO_COLUMN = {
    'Nelder Mead': 'Nelder-Mead', 'Bayesian optimisation': 'Bayesian',
    'Joint Bayesian optimisation': 'Joint Bayesian',
}


def _label_prediction_column(label, columns):
    """The ``Prediction_result_test.csv`` column holding ensemble label ``label``'s OWN test
    prediction, or ``None`` when it is not there. Used only by the first-term consistency check
    in `build_dpt_terms()`."""
    if label in columns:
        return label
    base, sep, suffix = str(label).partition('__')
    mapped = _WOPT_LABEL_TO_COLUMN.get(base)
    if mapped is not None:
        candidate = mapped + (sep + suffix if sep else '')
        if candidate in columns:
            return candidate
    return None


def cv_statistic(grouped: "pd.core.groupby.DataFrameGroupBy", value_col: str) -> "pd.Series":
    """Requirements.md item 2 - Eq. (5), p.4:

        CV_hat_termj = sigma_termj / mu_termj

    A sibling estimator to `metric_summary.six_statistics()`, over the
    IDENTICAL grouping (same `grouped` object, same `value_col` - one of
    `DPT_CV_TERMS`): `sigma_termj` is the (sample, ddof=1) standard
    deviation of `value_col` within each group, `mu_termj` its mean - both
    computed across every `(population, phenotype, ratio, sample)`
    prediction scenario row that falls in that group (never across models,
    never within a single scenario - see the module docstring's
    "Requirements.md item 2" section for why this is the correct
    granularity, matching the article's own "per trait" usage exactly at
    the phenotype-only grouping).

    Blank (`NaN`), never `inf`/silently wrong, when:
      - fewer than 2 non-NaN observations contributed to the group
        (a CV, like a standard error, is undefined for `n_eff < 2`);
      - the group's mean is exactly 0 (division by zero - the DPT terms
        are sums/means of squared errors, so this only occurs when every
        contributing model predicted every individual in every scenario
        of the group perfectly).

    Rounded to 6 dp, matching `six_statistics()`'s own numeric statistics
    (never left for a caller to round, for the same reason that function
    documents).
    """
    count = grouped[value_col].count()
    mean = grouped[value_col].mean()
    std = grouped[value_col].std(ddof=1)
    _enough = (count >= 2) & mean.notna() & (mean != 0)
    return (std / mean).where(_enough).round(6)


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

    # ver4-95: terms of two labels are only on a common footing when they average over the SAME
    # models. The same-tuning-group cross-check above cannot see a Super learner row (its label
    # suffix is the weight METHOD, not a tuning group), and the Super learner leaves bio-prior
    # models out (never fitted out of fold), so say plainly when a naive label and a weighted
    # label do not share their member set - that alone moves every term.
    for naive_label, naive_info in members.items():
        if not naive_info['is_naive']:
            continue
        for weighted_label, weighted_info in members.items():
            if weighted_info['is_naive']:
                continue
            only_naive = sorted(set(naive_info['models']) - set(weighted_info['models']))
            only_weighted = sorted(set(weighted_info['models']) - set(naive_info['models']))
            if only_naive or only_weighted:
                print(f"[diversity_summary] NOTE: {naive_label!r} ({len(naive_info['models'])} "
                      f"models) and {weighted_label!r} ({len(weighted_info['models'])} models) "
                      f"do not combine the same models - only in {naive_label!r}: {only_naive}; "
                      f"only in {weighted_label!r}: {only_weighted}. Their DPT terms are over "
                      f"different model sets, so their scales are not directly comparable.")

    return members


def build_dpt_terms(
    prediction_test: "pd.DataFrame", weight: "pd.DataFrame", metric: "pd.DataFrame",
    *, sample_weight_resolver=None,
) -> "pd.DataFrame":
    """PURE. One row per `(population, phenotype, ratio, sample, model)`
    for every ensemble label `resolve_ensemble_members()` resolves (naive
    included - it goes through this IDENTICAL path, not a separate
    branch: `w_i = 1/N` is simply what gets substituted for its weight
    vector). Columns: `population, phenotype, ratio, sample, model,
    n_models, first_term, second_term, third_term, ratio_term,
    n_effective` (ver4-95: `n_effective = 1 / sum(w_i^2)`, the effective
    number of models; `n_models` for the naive ensemble). Empty
    frame in (either `prediction_test` or `metric`) -> empty frame out;
    never raises for that reason.

    sample_weight_resolver : ver4-25, R8 (design blueprint SS2, Stage 6;
        Stage 6 completion for 'full' mode) - optional callable, called
        per scenario as `sample_weight_resolver(population, phenotype,
        ratio, sample, ids=<this scenario's own group['id'] list>)`
        (mirroring `models/ensemble.py::_metric()`'s own
        `asreml_accuracy_ctx_resolver` call convention exactly, `ids=`
        included). Returns either `None` (nothing resolved), a ctx-shaped
        dict with `'test_weights'` (`{id: weight}`, diagonal mode) and/or
        `'test_weight_matrix'` (`(W, id_list)`, full mode) keys - or, for
        backward compatibility with a resolver written before the
        'full'-mode extension, a BARE `{id: weight}` dict (treated
        identically to a ctx dict's own `'test_weights'` entry). A
        resolver that raises `TypeError` on the `ids=` keyword (an even
        older-style resolver) is retried with the original 4-positional-
        argument call - see `models/ensemble.py::_metric()`'s identical
        fallback for the full rationale. `None` (the default - every
        pre-ver4-25 caller) is a byte-identical `np.mean(...)`-based
        computation throughout (I11).

        Turns every `<...>` mean-over-TEST-individuals in the three DPT
        terms into the SAME `mean_W(...)` blueprint SS2 R8 defines for
        every other weighted metric this run reports (diagonal: a 1-D
        `w`; full: a dense `W`, via `1'Wa / 1'W1`) - NOT to be confused
        with `w`, the per-scenario ENSEMBLE COMBINING weight vector
        (naive `1/N`, or a fitted weighted-ensemble method's own
        `Weight.csv` row) computed just below, which is a completely
        different, unrelated quantity. A scenario with no resolvable
        weight (weighting off, or a resolver miss) falls back to the
        SAME unweighted computation automatically - never fatal, never
        silently wrong.

    Raises
    ------
    ValueError : if, for any scenario/label, the DPT identity
        `first_term == second_term - third_term` is violated by more than
        `1e-9 * max(1.0, abs(second_term))` - an ARITHMETIC INVARIANT, not
        a data condition (see the module docstring for why `M_w` is
        defined as the weighted mean specifically so this identity always
        holds - this holds under ANY PSD per-individual weighting too,
        including the `mean_W` this function now optionally applies, per
        blueprint SS2 R8's own "the DPT identity holds exactly under any
        PSD W" note). A violation means the contributing-model set or the
        weights were resolved incorrectly for that scenario/label, and
        silently writing the resulting number would put a wrong value in
        a published table - this fails loudly instead, naming the
        scenario and label, mirroring `GP()`'s own precedent of raising a
        diagnostic-rich `ValueError` on an effect-width mismatch
        (architecture doc SS4.3).
    """
    columns = ['population', 'phenotype', 'ratio', 'sample', 'model', 'n_models',
               'first_term', 'second_term', 'third_term', 'ratio_term', 'n_effective']
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
    _own_mse_mismatch: "dict[str, list]" = {}     # label -> [scenarios checked, mismatching]
    _zero_second_term_count = 0
    _weighted_scenario_count = 0
    _full_weighted_scenario_count = 0

    def _resolve_scenario_weighting(pop, phen, rat, samp, ids):
        """`(w, W)` - exactly one of the two is non-`None` (or both
        `None`, unweighted) - see `sample_weight_resolver`'s own
        docstring above for every shape this tolerates."""
        if sample_weight_resolver is None:
            return None, None
        try:
            _res = sample_weight_resolver(pop, phen, rat, samp, ids=ids)
        except TypeError:
            try:
                _res = sample_weight_resolver(pop, phen, rat, samp)
            except Exception:
                return None, None
        except Exception:
            return None, None
        if not _res:
            return None, None
        if isinstance(_res, dict) and ('test_weights' in _res or 'test_weight_matrix' in _res):
            _matrix = _res.get('test_weight_matrix')
            if _matrix is not None:
                return None, _matrix
            return (_res.get('test_weights') or None), None
        # Backward compatibility: a bare {id: weight} dict, the
        # resolver's own pre-Stage-6-completion contract.
        return (_res or None), None

    for scenario_key, group in prediction_test.groupby(list(_TERM_SCENARIO_KEYS), sort=False, dropna=False):
        pop, phen, rat, samp = scenario_key
        actual = group['actual'].to_numpy(dtype=float)

        # ver4-25, R8: resolved ONCE per scenario (not per label - every
        # ensemble label sharing this scenario's own test individuals
        # uses the SAME BLUE-reliability weight, per Fairness Principle
        # P8). `None` whenever the resolver is absent, has nothing for
        # this scenario, or 'id' is missing from `prediction_test` (an
        # older run's own file shape) - falls back to unweighted below.
        blue_w = None
        blue_W = None
        if sample_weight_resolver is not None and 'id' in group.columns:
            _ids_for_scenario = [str(_id) for _id in group['id']]
            _w_by_id, _matrix_info = _resolve_scenario_weighting(pop, phen, rat, samp, _ids_for_scenario)
            if _matrix_info is not None:
                _Wfull, _mids = _matrix_info
                # `_mids` is `_ids_for_scenario`'s own order by
                # construction (the exact `ids` just passed to the
                # resolver above) - no re-alignment needed.
                if _Wfull is not None:
                    blue_W = np.asarray(_Wfull, dtype=float)
                    _weighted_scenario_count += 1
                    _full_weighted_scenario_count += 1
            elif _w_by_id:
                _w_arr = np.array([_w_by_id.get(str(_id), np.nan) for _id in group['id']], dtype=float)
                if np.all(np.isfinite(_w_arr)) and float(np.sum(_w_arr)) > 0:
                    blue_w = _w_arr
                    _weighted_scenario_count += 1

        def _wmean(values):
            """Scalar mean_W over TEST individuals - `values` is a
            (n_individuals,) array; byte-identical `np.mean()` when
            neither `blue_w` nor `blue_W` is set (I11). `blue_W` (dense,
            full mode) takes precedence over `blue_w` (diagonal) - the
            two are never both set for the same scenario by construction
            (`_resolve_scenario_weighting()` returns at most one)."""
            if blue_W is not None:
                ones = np.ones(len(values))
                denom = float(ones @ blue_W @ ones)
                if not np.isfinite(denom) or denom == 0:
                    return float(np.mean(values))
                return float(ones @ blue_W @ values) / denom
            if blue_w is None:
                return float(np.mean(values))
            return float(np.sum(blue_w * values)) / float(np.sum(blue_w))

        def _wmean_cols(matrix):
            """Per-column mean_W over TEST individuals - `matrix` is
            (n_individuals, n_models); byte-identical `np.mean(...,
            axis=0)` when neither `blue_w` nor `blue_W` is set (I11)."""
            if blue_W is not None:
                ones = np.ones(matrix.shape[0])
                denom = float(ones @ blue_W @ ones)
                if not np.isfinite(denom) or denom == 0:
                    return np.mean(matrix, axis=0)
                return (ones @ blue_W @ matrix) / denom
            if blue_w is None:
                return np.mean(matrix, axis=0)
            return (blue_w[:, None] * matrix).sum(axis=0) / np.sum(blue_w)

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
            first_term = _wmean((weighted_mean - actual) ** 2)
            per_model_mse = _wmean_cols((preds - actual[:, None]) ** 2)
            second_term = float(np.dot(w, per_model_mse))
            per_model_div = _wmean_cols((preds - weighted_mean[:, None]) ** 2)
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

            # ver4-95: effective number of models, 1 / sum(w_i^2) (= N for equal weights, 1 when
            # one model carries all the weight). The terms of a weighted label shrink with it
            # (a concentrated ensemble sits close to its heaviest members, so both the
            # average-model error and the diversity term fall) - this is what separates a
            # weighted label's scale from the naive one's on the SAME models.
            n_effective = float(1.0 / np.sum(w ** 2))

            # ver4-95: consistency check, log only. For an ensemble that is exactly the weighted
            # sum of the member columns in this file, first_term must equal the MSE of its OWN
            # prediction column. It does not for a Super learner refitted on train+valid, a pooled
            # Super learner row, an Iterative ensemble or a Linear-transformation network: their
            # weights apply to refitted / transformed predictions that are not in the file, so the
            # terms describe the Step 9 columns re-combined, not the ensemble's own prediction.
            # Skipped under BLUE weighting (the file's MSE is then a weighted one).
            if blue_w is None and blue_W is None:
                own_col = _label_prediction_column(label, group.columns)
                if own_col is not None:
                    own_mse = float(np.mean((group[own_col].to_numpy(dtype=float) - actual) ** 2))
                    seen = _own_mse_mismatch.setdefault(label, [0, 0])
                    seen[0] += 1
                    if abs(first_term - own_mse) > 1e-6 * max(1.0, abs(own_mse)):
                        seen[1] += 1

            rows.append({
                'population': pop, 'phenotype': phen, 'ratio': rat, 'sample': samp,
                'model': label, 'n_models': n_models,
                'first_term': first_term, 'second_term': second_term,
                'third_term': third_term, 'ratio_term': ratio_term,
                'n_effective': n_effective,
            })

    if _zero_second_term_count:
        print(f"[diversity_summary] NOTE: {_zero_second_term_count} scenario/label row(s) had a "
              f"second_term of exactly 0 (every contributing model predicted perfectly) - "
              f"'ratio_term' left blank for those rows rather than reporting infinity.")
    for _label, (_checked, _bad) in _own_mse_mismatch.items():
        if _bad:
            print(f"[diversity_summary] NOTE: for {_bad} of {_checked} scenario(s) of {_label!r} "
                  f"the first term (MSE of the member columns combined with this label's weights) "
                  f"differs from the MSE of the label's own prediction column. The label's "
                  f"prediction was not formed as a plain weighted sum of the Step 9 member "
                  f"columns in Prediction_result_test.csv (e.g. a Super learner refitted on "
                  f"train+valid, its pooled row, an Iterative ensemble or a Linear-"
                  f"transformation network), or the member set differs; its DPT terms describe "
                  f"those columns re-combined with its weights.")
    if sample_weight_resolver is not None:
        print(f"[diversity_summary] BLUE-reliability weighting resolver was given - "
              f"{_weighted_scenario_count} scenario(s) had a usable weight and were computed with "
              f"mean_W ({_full_weighted_scenario_count} of those via a dense 'full'-mode matrix, "
              f"the rest via a diagonal 'w' vector); every other scenario fell back to the "
              f"ordinary unweighted mean.")

    return pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)


def build_dpt_summary(
    terms: "pd.DataFrame", model_order: "list[str] | None" = None,
) -> "tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]":
    """PURE. Returns `(long_sheets, pivot_blocks)`:

    `long_sheets` - keyed by `DPT_SHEET_NAMES`, one row per level-key x
        model, columns `<level keys...>, model, n`, then
        `f'{term} {stat}'` for every `term` in `DPT_TERMS` and `stat` in
        `_DPT_LONG_STAT_SUFFIXES` (median/mean/mean standard error),
        THEN (Requirements.md item 2) `f'{term} CV'` for every `term` in
        `DPT_CV_TERMS` (Eq. (5), via `cv_statistic()` - APPENDED after the
        `_DPT_LONG_STAT_SUFFIXES` columns, never interleaved with them, so
        every pre-existing column keeps its own position) - `n` is the
        plain row count for that group (matching
        `metric_summary.build_metric_summary()`'s own `n` convention,
        never `n_eff`).
    `pivot_blocks` - keyed by `DPT_BLOCK_TITLES` (27 block titles: the
        original 24, PLUS 3 Eq. (5) CV blocks - Requirements.md item 2 -
        APPENDED after them), phenotype-indexed, ensemble-model-columned,
        using the SAME six statistics `metric_summary.six_statistics()`
        computes (identical estimator, identical rounding, identical
        blank rules - imported, not re-derived) plus `cv_statistic()`
        (this module's own sibling estimator, same grouping) for the 3
        CV blocks.

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

        # Requirements.md item 2: the Eq. (5) CV columns, ONE per
        # DPT_CV_TERMS - appended after every _DPT_LONG_STAT_SUFFIXES
        # column above (never interleaved), matching col_order below.
        for term in DPT_CV_TERMS:
            col_name = f'{term} CV'
            merge_frame = cv_statistic(_grouped_for_stats, term).rename(col_name).reset_index()
            wide = wide.merge(merge_frame, on=group_cols, how='left')

        # ver4-95: how many models the label combines, and how many of them effectively carry
        # the weight (1 / sum(w^2)): APPENDED after the CV columns, never interleaved. A weighted
        # label with n_effective well below n_models is close to its heaviest members, which is
        # why its second and third terms are smaller than the naive ensemble's on the same models.
        for extra_col, how in _DPT_LONG_EXTRA_COLUMNS:
            extra = getattr(_grouped_for_stats[extra_col], how)().round(3)
            wide = wide.merge(extra.rename(f'{extra_col} {how}').reset_index(),
                              on=group_cols, how='left')

        sort_keys = wide[list(keys)].astype(str).copy()
        sort_keys['_model_rank'] = wide['model'].map(model_rank)
        order = sort_keys.sort_values(by=list(keys) + ['_model_rank'], kind='mergesort').index
        wide = wide.loc[order].reset_index(drop=True)

        col_order = list(keys) + ['model', 'n'] + [
            f'{t} {s}' for t in DPT_TERMS for s in _DPT_LONG_STAT_SUFFIXES
        ] + [f'{t} CV' for t in DPT_CV_TERMS] + [f'{c} {h}' for c, h in _DPT_LONG_EXTRA_COLUMNS]
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

    # Requirements.md item 2: 3 more blocks, ONE per DPT_CV_TERMS, the
    # Eq. (5) coefficient of variation - grouped identically
    # (['phenotype', 'model']) to every block above, and to the article's
    # own "per trait" usage. Appended after the 24 blocks above so their
    # own dict insertion order - and hence DPT_BLOCK_TITLES' matching
    # order - is preserved.
    for term in DPT_CV_TERMS:
        cv_series = cv_statistic(grouped_pm, term)
        pivot = cv_series.unstack('model')
        pivot = pivot.reindex(index=phenotype_order, columns=_order)
        pivot.index.name = 'phenotype'
        pivot.columns.name = 'model'
        pivot_blocks[f'{DPT_TERM_TITLES[term]} CV'] = pivot

    return long_sheets, pivot_blocks


def write_dpt_summary(
    terms: "pd.DataFrame", RESULT_NAME: str, *, create: bool = True,
    model_order: "list[str] | None" = None,
    label_style: str = 'full',
) -> "list[str]":
    """Writes `Result/<RESULT_NAME>/Diversity_prediction_theorem.xlsx` -
    'summary' sheet FIRST (the 27 blocks, `DPT_BLOCK_TITLES`), then the
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

    `label_style` (ver4-95, config key `MODEL_LABEL_STYLE`): 'full' (default, unchanged) or 'short';
    'short' drops the ``'Super learner__'`` prefix from the model labels written to the sheets /
    CSVs (`pipeline_utils.display_label_map`; a label whose short text would clash keeps its full
    text). The terms themselves are never recomputed.
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
    # ver4-95: order / aggregation fixed on the full labels; only the written text is shortened
    long_sheets, pivot_blocks, _pivot_model_order = apply_label_style_to_tables(
        long_sheets, pivot_blocks, _pivot_model_order, label_style)
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
                                       block_titles=DPT_BLOCK_TITLES)#, note=DPT_SUMMARY_NOTE)
        print(f"[diversity_summary] Diversity Prediction Theorem summary written: {xlsx_path}")
        return [xlsx_path]
    except (ImportError, ValueError):
        csv_paths = []
        if pivot_blocks:
            pivot_path = f'{result_dir}Diversity_prediction_theorem.csv'
            write_pivot_csv(pivot_blocks, _pivot_model_order, pivot_path,
                             block_titles=DPT_BLOCK_TITLES)#, note=DPT_SUMMARY_NOTE)
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
