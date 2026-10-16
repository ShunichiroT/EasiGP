"""
metric_summary.py
==================

Update ID 3, R3: a tabular complement to `metric_plot()`'s violin plots -
median/mean Pearson correlation and MSE, at three levels of aggregation
(phenotype; phenotype x population; phenotype x population x ratio), so a
person doesn't have to build this by hand from Metric.csv every time.

Requirements.md item 8: ALSO builds a second, wide "pivot" table - one
block per metric (Pearson correlation, then MSE), phenotype as rows,
model as columns - matching the layout of a hand-built spreadsheet
summary exactly (see `build_metric_summary_pivot()`'s own docstring),
with row/column order guaranteed identical to `metric_plot()`'s own
violin plots (never independently re-derived - see that function's
`model_order`/`phenotype_order` parameters, and
`pipeline_utils.canonical_model_order()`, applied to every model order
this module resolves - passed-in or defaulted - so the naive ensemble
always lands right before the weighted-ensemble methods, matching
`metric_plot()`'s own convention). This is written ALONGSIDE
the original three-level breakdown above, not instead of it - the pivot
table is now the FIRST sheet of `Metric_summary.xlsx` (named 'summary'),
or a new standalone `Metric_summary.csv` when `openpyxl` is unavailable.

Update ID ver4-9, R1-R4 (findings F2/F3): the 'summary' sheet described
above SHIPPED as two blocks - one for Pearson correlation, one for MSE -
each reporting a single, UNLABELLED statistic that was actually the
MEAN, not the median the surrounding text called it (finding F2 - do not
assume there is any pre-existing "median" behaviour to preserve; there
never was one). It now reports twelve blocks: {median, mean} x {value,
standard error, value +/- standard error} for EACH of Pearson correlation
and MSE, each with its own explicit title row (`PIVOT_BLOCK_TITLES`,
Pearson block first, then MSE, matching `PIVOT_METRICS`'s own fixed
order). `build_metric_summary()` and its three long-format sheets
(`phenotype`, `phenotype_population`, `phenotype_population_ratio`) are
UNTOUCHED by this - they stay byte-identical; only the 'summary' pivot
sheet (and its CSV degrade) changed shape.

`n` vs `n_eff`: every long-format sheet's own `n` column (see
`build_metric_summary()`'s docstring) is the plain, UNFILTERED row count
for a cell - "how many replicates/tasks does this cell represent",
independent of whether Pearson correlation/MSE happen to be NaN for some
of them (`pipeline_utils.safe_regression_metrics()` can record a
divergent/constant-input fit as `(nan, mse)` - a finite MSE alongside a
NaN Pearson for the SAME row). The two new "standard error" statistics
below are, by contrast, always divided by `n_eff` - the NON-NaN count for
THAT metric in THAT cell (`pandas.Series.count()`) - never by `n`, because
a standard error divided by a count that includes rows the estimator
never actually saw would simply be wrong. `n` and `n_eff` can therefore
legitimately differ within one file; this module reports both concepts,
never conflates them, and this note is deliberately restated here rather
than left implicit.

Median standard-error estimator: `sqrt(pi/2) * SE(mean)` -
`MEDIAN_SE_FACTOR = 1.2533141373155003` - the large-sample, asymptotic-
normal approximation to the standard error of a sample median. This is a
DELIBERATE, documented choice, not the only one available (a seeded
bootstrap was considered and rejected - it would need a resample count
and a seed pinned deterministically across every execution mode for
reproducibility, adding runtime and a new source of nondeterminism for a
benefit no requirement currently asks for; see the ver4-9 design
blueprint SS2.1.2 for the full argument, preserved there for a later
update to revisit). Because MSE is right-skewed, this asymptotic-normal
estimate should be read as an approximation, not a robust interval - the
estimator's name is therefore printed both in the block title and in
`MEDIAN_SE_NOTE`, a note row appended below the last block of the
'summary' sheet, so no reader of the spreadsheet is left guessing what
was actually computed.

Combined ("+/- ") cells are built from the ROUNDED (3 dp) component
values, not from full precision, so the "+/-" block is always
arithmetically consistent with the two component blocks sitting in the
same sheet. They are STRINGS: Excel will not aggregate them and
`openpyxl` writes them as text. This is deliberate, not an oversight.

`openpyxl==3.1.5` is pinned in BOTH `environment_linux.yml` and
`environment_windows.yml` (finding F3) - an earlier draft of this
docstring, and the CSV-degrade warning text below, incorrectly implied it
was Windows-only/absent on Linux (the platform Parallel mode actually
runs on). That claim was wrong; it has been corrected here and in the
warning text. The CSV degrade path itself is UNCHANGED and still exists,
for any environment that genuinely lacks the package.

`build_metric_summary()`/`build_metric_summary_pivot()` are PURE (frame
in, dict of frames out - no disk access, no printing) so they're
directly unit-testable. `write_metric_summary()` is the only function in
this module that touches disk, and the only one that imports `openpyxl`
- lazily, inside itself, wrapped so a missing `openpyxl` degrades to
named CSVs with a loud warning, never a fatal error (I12 - no package is
added to any manifest for this).

Layout helpers (`pivot_data_rows`, `pivot_block_rows`, `all_pivot_rows`,
`write_pivot_worksheet`, `write_pivot_csv`) are PUBLIC as of ver4-9, so
`diversity_summary.py` (a lateral sibling module, Update ID ver4-9 R5)
can reuse the identical block-stacked layout for its own 24-block
Diversity Prediction Theorem summary, rather than maintaining a second,
independently-drifting copy. The old leading-underscore names are kept as
thin aliases so any existing caller of this module keeps working
unchanged.
"""

from __future__ import annotations

import csv
import math
import os

import pandas as pd

from pipeline_utils import canonical_model_order, display_label_map, normalise_label_style
from asreml_accuracy_metrics import ACCURACY_METRIC_COLUMNS, SCENARIO_CONSTANT_METRIC_COLUMNS

SHEET_NAMES = ('phenotype', 'phenotype_population', 'phenotype_population_ratio')
LEVEL_KEYS = (('phenotype',), ('phenotype', 'population'), ('phenotype', 'population', 'ratio'))

_METRIC_COLUMNS = ('Pearson correlation median', 'Pearson correlation mean', 'MSE median', 'MSE mean')

# Requirements.md item 8: the two metrics shown in the pivot table, in
# this FIXED display order (title-row order matches the attached
# example.csv: 'Pearson correlation' block first, then 'MSE') -
# deliberately not derived from any dict's own (insertion-ordered, but
# still incidental-looking) key order. UNCHANGED by ver4-9.
PIVOT_METRICS = ('Pearson correlation', 'MSE')

# Update ID ver4-9, R1-R4: the six statistics computed for EVERY metric x
# phenotype x model cell, in this FIXED order (median-family first, then
# mean-family, each as value/SE/value+/-SE) - matches the exact ordering
# R4 enumerates. `six_statistics()` below returns a dict keyed by exactly
# these six strings.
PIVOT_STATISTICS = (
    'median', 'mean',
    'median standard error', 'mean standard error',
    'median plus standard error', 'mean plus standard error',
)

# sqrt(pi/2) - the large-sample asymptotic-normal standard error of a
# sample median, expressed as a multiple of SE(mean). See the module
# docstring's "Median standard-error estimator" section for the full
# rationale and the rejected (seeded-bootstrap) alternative.
MEDIAN_SE_FACTOR = math.sqrt(math.pi / 2)  # 1.2533141373155003

# The 12 block titles the 'summary' sheet now carries, PEARSON BLOCK
# FIRST (outer loop over PIVOT_METRICS), then each metric's own six
# statistics in PIVOT_STATISTICS order (inner loop) - this nested
# comprehension order is what produces "all six Pearson blocks, then all
# six MSE blocks", exactly R4's own stated requirement. A fixed tuple,
# never derived from any dict's insertion order (the same discipline
# PIVOT_METRICS itself already follows).
PIVOT_BLOCK_TITLES = tuple(f'{m} {s}' for m in PIVOT_METRICS for s in PIVOT_STATISTICS)

# ver4-25, R10 (design blueprint stage 3, AC10.5): the three heritability-
# like "column families" the `variance_components` sheet reports, one per
# statistic `Preprocess/asreml_blup.py::fit_concurrent_variance_model()`
# computes on its own training-only, genotype-random-iid fit - see
# `build_varcomp_summary_pivot()`'s own docstring for the full six-blocks-
# per-family layout this produces.
VARCOMP_SUMMARY_COLUMNS = ('h2_plot', 'H2_classic', 'H2_cullis_train')
VARCOMP_SUMMARY_BLOCK_TITLES = tuple(
    f'{c} {s}' for c in VARCOMP_SUMMARY_COLUMNS for s in PIVOT_STATISTICS
)

# Printed as the final row of the 'summary' sheet (and its CSV degrade),
# below the last block - see the module docstring's "Median
# standard-error estimator" and "n vs n_eff" sections for what this is
# restating in spreadsheet-visible form.
MEDIAN_SE_NOTE = (
    "Note: every 'standard error' column above divides by n_eff (the number of NON-NaN "
    "values in that cell), never by the 'n' column shown on the other sheets (the plain, "
    "unfiltered row count) - blank when n_eff < 2. 'median standard error' is the "
    "large-sample asymptotic-normal estimate sqrt(pi/2) x SE(mean), an approximation (MSE "
    "is right-skewed), not a robust bootstrap interval. Every '... plus standard error' "
    "cell is a TEXT STRING built from the two ROUNDED (3 dp) values shown in the "
    "neighbouring blocks - Excel will not aggregate these cells as numbers. "
    "'Heritability (classic)' and 'Heritability' (Cullis) are per-scenario constants "
    "(population x phenotype x ratio x replicate), copied "
    "identically into every model's own row for a task, not per-model metrics like Pearson "
    "correlation/MSE - each one's two blocks above report the median/mean of that constant "
    "across scenarios, in a single column; no standard-error block is reported for them"
    "."
)


def build_metric_summary(
    metric: "pd.DataFrame", model_order: "list[str] | None" = None,
    extra_metrics: "tuple[str, ...]" = ACCURACY_METRIC_COLUMNS,
) -> "dict[str, pd.DataFrame]":
    """Pure: `metric` (Metric.csv's own frame, any source) in, a dict
    keyed by `SHEET_NAMES` out - one DataFrame per aggregation level, with
    columns `<level keys...>, model, n, Pearson correlation median,
    Pearson correlation mean, MSE median, MSE mean`, PLUS (follow-up
    requirement, BLUEs/BLUPs accuracy-metrics phase) a `'<metric> median'`/
    `'<metric> mean'` pair for every name in `extra_metrics` that is
    ACTUALLY a column of `metric` (i.e. whichever of the up-to-9 new
    accuracy-metric columns - `asreml_accuracy_metrics.
    ACCURACY_METRIC_COLUMNS`, the default - the run's own ASREML
    configuration produced; absent entirely, exactly as before, for any
    run/config that never produced them). "as done for the current
    metrics" (Requirements.md item 7) - i.e. the SAME median/mean
    aggregation already applied to Pearson correlation/MSE, generalised
    rather than re-derived.

    UNCHANGED by ver4-9 (R1-R4 only touch the 'summary' pivot sheet built
    by `build_metric_summary_pivot()` below) - kept byte-identical,
    including this docstring's own content, so any existing caller/test
    keeps working unmodified.

    Rows are sorted by the level's key columns ascending, then by `model`
    in ORDER OF FIRST APPEARANCE in `metric` (the same canonical model
    order `assemble()`/both runners already use via
    `pd.unique(metric['model'])`) - never alphabetically - UNLESS
    `model_order` is given, in which case that order is used instead (any
    model present in `metric` but missing from `model_order` is appended
    at the end, in its own order of first appearance, never dropped).
    Either way, `pipeline_utils.canonical_model_order()` is applied on
    top: it moves any naive-ensemble label ('ensemble', 'ensemble__<algo>')
    to sit immediately before the first weighted-ensemble label ('Nelder
    Mead', 'Linear transformation', 'Bayesian optimisation', 'Analytic
    least-squares', or their own '__<algo>' variants), since the naive
    ensemble's rows are always the LAST to first-appear in `metric` (GP()
    computes it once, in finalisation, strictly after the per-task loop
    that produces every weighted-ensemble row) - see that function's own
    docstring. Without
    this, plain order-of-first-appearance would always place the naive
    ensemble after every weighted-ensemble method, which is exactly the
    ordering `metric_plot()`'s violin plots (and this module's own pivot
    'summary' sheet, below) deliberately avoid. `n` is the
    number of contributing `metric` rows for that cell (a plain row
    count, independent of whether Pearson correlation/MSE happen to be
    NaN for some of them).

    `ratio` is read VERBATIM from `metric` - never through
    `pipeline_utils.restore_ratio()`, and never re-derived from any
    config. Re-deriving it would introduce a second source of truth that
    can disagree with the data actually in the file (e.g. a resumed run
    whose config was hand-edited between batches); this is a summary OF
    the file, so it reads the file. `population` is kept WHOLE for
    'between' scenarios (`'X->Y'`, or `'X->Y->Z'` when `W_OPT` is set) -
    splitting it, as `circos_plot()`/`metric_plot()` do for their own,
    different reasons, would silently average across different training
    populations. No `model` value is collapsed or normalised - `ensemble`,
    `ensemble__<algo>`, `<model>__<algo>`, `'Linear transformation'`,
    `'Nelder Mead'` (spaced), `'Bayesian optimisation'` and `'Analytic
    least-squares'` are all reported as their own distinct rows, exactly
    as they appear in `metric['model']`.

    The `-1` sentinel `ratio`/`sample` values 'between' scenarios use are
    never filtered - the third sheet is written even when every row's
    `ratio` is `-1` (so it ends up row-for-row identical to the second
    sheet; `write_metric_summary()` logs why, this function doesn't log
    anything at all).

    Sort key note: `ratio` can genuinely mix Python types within one
    `metric` frame (a float for a plain train/test ratio, a 3-tuple for a
    train/valid/test ratio, or the int `-1` sentinel for 'between') - as
    a real, in-config possibility ("floats and/or 3-tuples", architecture
    doc Appendix B). Comparing a tuple to a float raises in Python, so the
    ascending sort below compares each key column's STRING form
    (`.astype(str)`) rather than its raw value - a safe, deterministic,
    total order across mixed types. The DISPLAYED `ratio` value is never
    affected by this - only the row order is derived from the string
    form.
    """
    sheets: "dict[str, pd.DataFrame]" = {}
    if metric is None or metric.shape[0] == 0:
        return sheets

    if model_order is None:
        _order = pd.unique(metric['model']).tolist()
    else:
        _seen_m = set(model_order)
        _order = list(model_order) + [m for m in pd.unique(metric['model']) if m not in _seen_m]
    _order = canonical_model_order(_order)
    model_rank = {m: i for i, m in enumerate(_order)}

    # Follow-up requirement: whichever of `extra_metrics` is genuinely a
    # column of `metric` this run - NEVER a hardcoded list, so a run with
    # the new metrics off (or only some selected) produces exactly the
    # columns it has data for, nothing fabricated.
    _extra_present = [m for m in extra_metrics if m in metric.columns]

    for sheet_name, keys in zip(SHEET_NAMES, LEVEL_KEYS):
        group_cols = list(keys) + ['model']
        # ver4-4 R5/§2.5.4 decision (flagged since Stage 1, closed here in
        # Stage 3): the blueprint asks Phase 2 to "decide and document
        # whether the summary skips or propagates" NaN Pearson/MSE rows
        # (possible since R5's safe_regression_metrics() can now record a
        # divergent fit as NaN instead of crashing the batch), recommending
        # "skip". Decision: SKIP, and no code change was needed to
        # implement it - pandas' own `.agg(median=...)/(mean=...)` already
        # default to `skipna=True` (verified directly: a 3-row group with
        # one NaN Pearson correlation value produces a median/mean over
        # the other two rows only, not NaN-poisoned), so this aggregation
        # has always satisfied the "skip" recommendation, even before R5
        # existed. `n` (below) is DELIBERATELY the plain, unfiltered row
        # count for the cell - not a "how many were non-NaN" count - so it
        # answers "how many replicates/tasks does this cell represent"
        # (the more broadly useful number for a domain scientist sizing
        # their own confidence in the summary), not a per-metric
        # contributing-count that would need to differ between the
        # Pearson and MSE columns whenever one is NaN and the other isn't
        # (e.g. safe_regression_metrics()'s own constant-input case
        # returns `(nan, mse)` - a finite MSE alongside a NaN Pearson for
        # the very same row).
        grouped = metric.groupby(group_cols, as_index=False, sort=False, dropna=False).agg(
            n=('Pearson correlation', 'size'),
            pearson_median=('Pearson correlation', 'median'),
            pearson_mean=('Pearson correlation', 'mean'),
            mse_median=('MSE', 'median'),
            mse_mean=('MSE', 'mean'),
            **{f'extra{_idx}_median': (_m, 'median') for _idx, _m in enumerate(_extra_present)},
            **{f'extra{_idx}_mean': (_m, 'mean') for _idx, _m in enumerate(_extra_present)},
        )
        _rename = {
            'pearson_median': 'Pearson correlation median',
            'pearson_mean': 'Pearson correlation mean',
            'mse_median': 'MSE median',
            'mse_mean': 'MSE mean',
        }
        _extra_cols = []
        for _idx, _m in enumerate(_extra_present):
            _rename[f'extra{_idx}_median'] = f'{_m} median'
            _rename[f'extra{_idx}_mean'] = f'{_m} mean'
            _extra_cols += [f'{_m} median', f'{_m} mean']
        grouped = grouped.rename(columns=_rename)

        # Ascending by the level's own key columns (as strings, see the
        # docstring note above), then by `model` in order-of-first-
        # appearance in the INPUT (not per-group) - a stable sort so ties
        # never reorder for any other reason.
        sort_keys = grouped[list(keys)].astype(str).copy()
        sort_keys['_model_rank'] = grouped['model'].map(model_rank)
        order = sort_keys.sort_values(by=list(keys) + ['_model_rank'], kind='mergesort').index
        grouped = grouped.loc[order].reset_index(drop=True)

        sheets[sheet_name] = grouped[list(keys) + ['model', 'n', *_METRIC_COLUMNS, *_extra_cols]]

    return sheets


def six_statistics(grouped: "pd.core.groupby.DataFrameGroupBy", value_col: str) -> "dict[str, pd.Series]":
    """Update ID ver4-9, R1-R4/R5 (blueprint SS2.2.4): the SHARED estimator
    behind every "median/mean/SE" statistic this module (and
    `diversity_summary.py`, a lateral sibling - I2) reports - written
    ONCE so a future change to the SE estimator is automatically correct
    everywhere it's used, rather than needing to be re-applied in two
    places by hand.

    Parameters
    ----------
    grouped : the result of `some_df.groupby([...], sort=False,
        dropna=False)` (an un-materialised pandas GroupBy object, NOT yet
        aggregated) - whatever key columns the caller wants each
        statistic broken out by (e.g. `['phenotype', 'model']` for the
        wide pivot sheet, or a level's own key columns plus `'model'` for
        a long-format sheet).
    value_col : the column within each group to summarise (e.g.
        `'Pearson correlation'`, `'MSE'`, or one of
        `diversity_summary.DPT_TERMS`).

    Returns
    -------
    dict, keyed by exactly the six strings in `PIVOT_STATISTICS`, each
    value a `pandas.Series` indexed by `grouped`'s own group keys (a
    `MultiIndex` when grouped by more than one column) - ready either to
    `.unstack()` into a wide, phenotype-indexed/model-columned frame
    (this module's own pivot table), or to `.reset_index()` and merge
    into a long-format sheet (`diversity_summary.build_dpt_summary()`).

    Per group, over the group's own (possibly NaN-containing) values of
    `value_col`:
      'median'                      : `series.median()` (skipna=True)
      'mean'                        : `series.mean()` (skipna=True)
      'mean standard error'         : `series.std(ddof=1) / sqrt(n_eff)`
      'median standard error'       : `MEDIAN_SE_FACTOR * that`
      'median plus standard error'  : `f'{median:.3f}+/-{median_se:.3f}'`
      'mean plus standard error'    : `f'{mean:.3f}+/-{mean_se:.3f}'`

    `n_eff` is `series.count()` - the NON-NaN count for `value_col` in
    that group - NEVER the group's plain row count (see the module
    docstring's "n vs n_eff" section for why the two can legitimately
    differ). Both SE statistics, and both "+/-" combined statistics, are
    blank (`NaN`) whenever `n_eff < 2` - a standard error is undefined
    for fewer than two observations. `median`/`mean` are blank only when
    a group's `value_col` is ENTIRELY NaN (n_eff == 0), which
    `.median()`/`.mean()` with `skipna=True` already produce naturally,
    with no extra masking needed. The two "+/-" statistics are also blank
    whenever EITHER of their two components is blank - a group with
    exactly one non-NaN value has a real median/mean but no SE, and so no
    "+/-" cell either.

    The two "+/-" statistics are STRINGS, built from the ROUNDED (3 dp)
    component values (never full precision) - see the module docstring's
    "Combined cells" section. The four numeric statistics are also
    rounded to 3 dp here (not left for a later caller to round), so a
    caller can never accidentally combine a rounded "+/-" string with an
    unrounded numeric sibling from the same cell.
    """
    count = grouped[value_col].count()
    median = grouped[value_col].median().round(6)
    mean = grouped[value_col].mean().round(6)
    std = grouped[value_col].std(ddof=1)
    _enough = count >= 2
    mean_se = (std / count.astype(float).pow(0.5)).where(_enough).round(6)
    median_se = (mean_se * MEDIAN_SE_FACTOR).round(6)

    def _combine(value: "pd.Series", se: "pd.Series") -> "pd.Series":
        combined = (
            value.map(lambda v: f'{v:.8f}' if pd.notna(v) else None).astype('object')
            + '\u00b1'
            + se.map(lambda v: f'{v:.8f}' if pd.notna(v) else None).astype('object')
        )
        return combined.where(value.notna() & se.notna())

    return {
        'median': median,
        'mean': mean,
        'median standard error': median_se,
        'mean standard error': mean_se,
        'median plus standard error': _combine(median, median_se),
        'mean plus standard error': _combine(mean, mean_se),
    }


def _scenario_constant_metric_blocks(
    metric: "pd.DataFrame", column: str, phenotype_order: "list[str]",
) -> "dict[str, pd.DataFrame]":
    """ver4-25, R2 (root cause): `column` (currently only `Heritability` -
    see `asreml_accuracy_metrics.SCENARIO_CONSTANT_METRIC_COLUMNS`) is a
    property of the SCENARIO (population x phenotype x ratio x
    replicate), copied identically into every model's own row for that
    task by `asreml_accuracy_metrics.compute_accuracy_metrics_row()` -
    NOT a per-model metric the way Pearson correlation/MSE genuinely
    are. Pivoting it through `six_statistics()`'s per-`(phenotype,
    model)` treatment (the pre-ver4-25 behaviour) repeated the identical
    value under every model's own column and reported a "standard
    error" that has no real meaning for a constant (it either measured
    zero spread trivially, or - if models were fanned out unevenly
    across tasks, e.g. a hand-edited `MODEL_DISPATCH_FILTER` - a
    spurious apparent "difference between models" that was really just
    a different subset of tasks per model).

    This function instead:
      1. de-duplicates `metric` down to ONE row per
         `(population, phenotype, ratio, sample)` task, taking the
         first non-NaN value of `column` for that task (pandas'
         `GroupBy.first()` already skips NaN - every model's row for a
         well-formed task carries the identical value, so "first" is as
         good as "only");
      2. checks that assumption - if `column` genuinely disagrees across
         models for the same task (`max - min > 1e-9`), prints one
         `[metric_summary] NOTE:` naming the count of offending tasks
         (never silent - architecture doc §18's "log negative states
         unconditionally");
      3. returns ONLY `'median'` and `'mean'` (no standard-error blocks
         - a spread-across-scenarios statistic is a different, and here
         unrequested, quantity from Pearson/MSE's own per-model-per-
         phenotype SE), each a `phenotype_order`-indexed DataFrame with
         a SINGLE column named `column`.

    Args:
        metric: `Metric.csv`'s own frame (any source).
        column: the scenario-constant metric's column name (e.g.
            `'Heritability'`).
        phenotype_order: the exact row order to reindex onto (same list
            `build_metric_summary_pivot()` itself resolves).

    Returns:
        A dict with keys `'median'`/`'mean'`, each mapping to a single-
        column DataFrame - or an empty dict if `column` is absent from
        `metric`, or if `metric` lacks any of the four task-identity
        columns (`population`, `phenotype`, `ratio`, `sample`) needed to
        dedupe correctly (degrades to "no block" rather than fabricating
        one from an ambiguous dedupe key).

    Raises:
        Nothing - degrades to an empty dict on missing columns, matching
        this module's general "never fabricate, just skip" discipline.
    """
    if column not in metric.columns:
        return {}
    task_keys = ['population', 'phenotype', 'ratio', 'sample']
    if not all(k in metric.columns for k in task_keys):
        return {}

    task_grouped = metric.groupby(task_keys, sort=False, dropna=False)[column]
    spread = task_grouped.max() - task_grouped.min()
    n_offending = int((spread.fillna(0.0) > 1e-9).sum())
    if n_offending > 0:
        print(f"[metric_summary] NOTE: '{column}' disagrees across models for {n_offending} "
              f"task(s) (population x phenotype x ratio x sample) by more than 1e-9 - using "
              f"each task's first non-NaN value for the '{column} median'/'{column} mean' "
              f"summary blocks below. '{column}' is expected to be a per-scenario constant "
              f"(the same ASReml fit feeds every model in a task); a disagreement usually "
              f"means models were fanned out unevenly across tasks in this run (e.g. a "
              f"hand-edited MODEL_DISPATCH_FILTER).")
    per_task = task_grouped.first()

    phenotype_grouped = per_task.groupby(level='phenotype', sort=False, dropna=False)
    median = phenotype_grouped.median().round(6)
    mean = phenotype_grouped.mean().round(6)

    def _as_block(series: "pd.Series") -> "pd.DataFrame":
        frame = series.reindex(phenotype_order).to_frame(name=column)
        frame.index.name = 'phenotype'
        frame.columns.name = 'model'
        return frame

    return {'median': _as_block(median), 'mean': _as_block(mean)}


def build_metric_summary_pivot(
    metric: "pd.DataFrame", model_order: "list[str] | None" = None, phenotype_order: "list[str] | None" = None,
    extra_metrics: "tuple[str, ...]" = ACCURACY_METRIC_COLUMNS,
) -> "dict[str, pd.DataFrame]":
    """Requirements.md item 8; Update ID ver4-9, R1-R4 (findings F2, F3):
    a wide, "one block per metric x statistic" pivot table - phenotype as
    rows, model as columns - mirroring the layout a person would build by
    hand from Metric.csv in a spreadsheet (see the attached example.csv),
    rather than `build_metric_summary()`'s own long/tidy format above.
    Aggregated at the coarsest level (across every population/ratio/
    replicate), matching the attached example exactly, which shows no
    population/ratio breakdown at all. This is a separate, additional
    view - `build_metric_summary()` and its three aggregation levels are
    untouched, see `write_metric_summary()`'s own docstring for how the
    two now coexist in the same output.

    Follow-up requirement (BLUEs/BLUPs accuracy-metrics phase), item 7 -
    `extra_metrics` (default `asreml_accuracy_metrics.
    ACCURACY_METRIC_COLUMNS`, the up-to-9 new columns) extends
    `PIVOT_METRICS` with whichever of those names is ACTUALLY a column of
    `metric` this run - so the returned dict gains six more blocks per
    selected metric (the SAME `PIVOT_STATISTICS` six-statistic treatment
    already applied to Pearson correlation/MSE - "as done for the current
    metrics"), in insertion order right after the two original blocks.
    Callers should therefore use `tuple(pivots.keys())` as their own
    `block_titles` (never the fixed `PIVOT_BLOCK_TITLES` global) once
    `extra_metrics` might be non-empty - see `write_metric_summary()`.

    ver4-25, R2 CHANGE: any name in `asreml_accuracy_metrics.
    SCENARIO_CONSTANT_METRIC_COLUMNS` (currently the classic and Cullis heritabilities) is
    treated differently from every other metric here - it is a property
    of the SCENARIO, not of the model, so it gets only TWO blocks
    (`'<name> median'`, `'<name> mean'`), each a single-column frame,
    rather than the full six-statistic x per-model treatment. See
    `_scenario_constant_metric_blocks()` for the full rationale.

    ver4-9 CHANGE: previously reported a SINGLE, unlabelled statistic per
    cell (the mean, despite being described elsewhere as "the median" -
    finding F2). Now reports all SIX statistics in `PIVOT_STATISTICS`,
    for EACH metric in `PIVOT_METRICS` - twelve blocks total, each with
    its own explicit title row naming exactly what it holds. See
    `six_statistics()` for the shared estimator, and the module docstring
    for the median-SE estimator and the `n`-vs-`n_eff` distinction.

    Parameters
    ----------
    metric : pandas.DataFrame
        Metric.csv's own frame (any source) - needs at least
        'phenotype', 'model', 'Pearson correlation', 'MSE' columns.
    model_order : list of str or None
        The exact column order to use (and which models to include as
        columns at all) - normally the SAME list `metric_plot()` itself
        receives as its own `MODEL` argument (used there as
        `hue_order=MODEL`), so this table's model order can never
        disagree with the violin plots' own. Falls back to
        `pd.unique(metric['model'])` (order of first appearance) when
        not given. Either way, `pipeline_utils.canonical_model_order()`
        is applied on top - see `build_metric_summary()`'s docstring for
        why (in short: it moves the naive ensemble to sit right before
        the weighted-ensemble methods, rather than wherever it happened
        to first-appear/be listed).
    phenotype_order : list of str or None
        The exact row order to use. Falls back to
        `pd.unique(metric['phenotype'])` (order of first appearance)
        when not given - a DELIBERATE default, not just a placeholder:
        seaborn's `FacetGrid(col='phenotype')` (what `metric_plot()`'s
        violin plots use) resolves its own column-facet order the
        IDENTICAL way for a plain (non-categorical, non-numeric)
        'phenotype' column whenever no explicit `col_order` is passed
        (`metric_plot()` never passes one) - `values.unique()`, i.e.
        order of first appearance in the data. So this default already
        matches the violin plots' own phenotype order without needing
        the run's real `PHENOTYPE` config list threaded through at all.
        An explicitly-passed order that's missing a phenotype genuinely
        present in `metric` never silently drops it - any such
        stragglers are appended at the end, in their own order of first
        appearance, rather than lost.

    Returns
    -------
    dict, keyed by `'<metric> <statistic>'` strings - Pearson/MSE (and
    any other genuinely per-model `extra_metrics` entry) contribute the
    full six `PIVOT_STATISTICS` blocks each, `model_order`-columned;
    every name in `SCENARIO_CONSTANT_METRIC_COLUMNS` contributes only
    `'median'`/`'mean'`, each a SINGLE-column frame named after the
    metric itself (see `_scenario_constant_metric_blocks()`). Every
    block is `phenotype_order`-indexed. The per-model numeric-statistic
    frames hold `float` rounded to 3 dp (`NaN` where that phenotype x
    model combination has no rows in `metric` at all, or - for the two
    "standard error" frames - where it has fewer than 2 non-NaN values;
    never silently zero-filled, matching `build_metric_summary()`'s own
    "never fabricate a number" discipline); the two "... plus standard
    error" frames hold `str` (or `None`/`NaN` under the same blank
    rules). Empty dict if `metric` is empty/None.
    """
    if metric is None or metric.shape[0] == 0:
        return {}

    if phenotype_order is None:
        phenotype_order = pd.unique(metric['phenotype']).tolist()
    else:
        _seen_p = set(phenotype_order)
        phenotype_order = list(phenotype_order) + [
            p for p in pd.unique(metric['phenotype']) if p not in _seen_p
        ]

    if model_order is None:
        model_order = pd.unique(metric['model']).tolist()
    else:
        model_order = list(dict.fromkeys(model_order))  # de-dup, preserve order
    model_order = canonical_model_order(model_order)

    # ONE groupby, producing count/median/mean/std per (phenotype, model)
    # per metric - six_statistics() derives all six statistics from it in
    # a single pass; see that function's own docstring. Unused for any
    # metric handled by _scenario_constant_metric_blocks() instead (see
    # the loop below) - constructing the GroupBy object itself is cheap
    # (aggregation is lazy), so skipping it there costs nothing.
    grouped = metric.groupby(['phenotype', 'model'], sort=False, dropna=False)

    # Follow-up requirement: append whichever of `extra_metrics` is
    # genuinely present in `metric` - see this function's own docstring.
    _metrics_present = list(PIVOT_METRICS) + [m for m in extra_metrics if m in metric.columns]

    pivots: "dict[str, pd.DataFrame]" = {}
    for m in _metrics_present:
        # ver4-25, R2: scenario-constant metrics (Heritability) get their
        # own two-block, single-column treatment instead of the six-
        # statistic per-model pivot every other metric gets below.
        if m in SCENARIO_CONSTANT_METRIC_COLUMNS:
            for s, wide in _scenario_constant_metric_blocks(metric, m, phenotype_order).items():
                pivots[f'{m} {s}'] = wide
            continue
        stats = six_statistics(grouped, m)
        for s in PIVOT_STATISTICS:
            series = stats[s]
            wide = series.unstack('model')
            wide = wide.reindex(index=phenotype_order, columns=model_order)
            wide.index.name = 'phenotype'
            wide.columns.name = 'model'
            pivots[f'{m} {s}'] = wide

    return pivots


def pivot_data_rows(pivot_df: "pd.DataFrame", model_order: "list[str]") -> "list[list]":
    """One row per phenotype (`pivot_df.index`, already in the caller's
    desired order), each `[phenotype, value_for_model_1, value_for_model_2,
    ...]` in `model_order` - a missing model column, or a phenotype x
    model combination with no rows in the source data, renders as an
    empty string cell (never a fabricated 0 or a literal 'nan'). Also
    covers a STRING-valued cell (the "... plus standard error" blocks
    added in ver4-9) transparently - `pd.isna()` on a real string is
    always False, so a formatted "value+/-error" string passes through
    unchanged, and a blank ("+/-" undefined) cell renders as '' exactly
    like a blank numeric cell.

    Update ID ver4-9: PROMOTED from `_pivot_data_rows` to a public name
    (blueprint SS2.2.4) so `diversity_summary.py` can reuse this exact
    layout logic rather than re-implementing it - no change to this
    function's own behaviour. The old, leading-underscore name is kept as
    a thin alias below.
    """
    rows = []
    for phenotype in pivot_df.index:
        row = [phenotype]
        for model in model_order:
            value = pivot_df.loc[phenotype, model] if model in pivot_df.columns else None
            row.append('' if pd.isna(value) else value)
        rows.append(row)
    return rows


def pivot_block_rows(block_title: str, pivot_df: "pd.DataFrame", columns: "list[str]") -> "list[list]":
    """One title+blank+header+subheader+data-rows block for a single
    pivot block (Requirements.md item 8's own stacked-block layout) as a
    list of plain Python lists, one per row - ready for either
    `csv.writer.writerows()` or one worksheet `.append()` call per row.
    `columns` is taken as a separate argument (rather than always read
    from `pivot_df.columns`) so the header row is still correct even for
    an all-empty/all-NaN pivot. Used for EVERY block's own rendering (see
    `all_pivot_rows()`).

    ver4-25, R2: parameter RENAMED from `model_order` to `columns` (no
    behaviour change to this function itself) - as of ver4-25 the caller
    (`all_pivot_rows()`) no longer always passes the run's own model
    list here: a scenario-constant block (Heritability - see
    `asreml_accuracy_metrics.SCENARIO_CONSTANT_METRIC_COLUMNS`) has a
    SINGLE column named after the metric itself, and `all_pivot_rows()`
    passes THAT single-column list for those blocks instead, so the
    block's own header/data rows come out exactly as wide as the data
    genuinely is, rather than padded out to every model's own column
    with the same value repeated under each one.

    Update ID ver4-9: PROMOTED from `_pivot_block_rows` to a public name
    (blueprint SS2.2.4), and its first parameter RENAMED from
    `metric_name` to `block_title` for honesty - as of ver4-9 this is one
    of the 12 `'<metric> <statistic>'` strings in `PIVOT_BLOCK_TITLES`
    (or, for `diversity_summary.py`, one of its own 27 term/CV x
    statistic titles, Update ID ver4-11), not a bare metric name - no
    behavioural change. The old, leading-underscore name/parameter are
    kept as a thin alias below.

    Row shape (width = 1 + len(columns) throughout):
      1. [block_title, '', '', ...]               - title row
      2. ['', '', ...]                            - blank separator
      3. ['Phenotype', 'Model', '', ...]          - group header row
      4. ['', column_1, column_2, ...]            - column header row
      5..N. [phenotype_i, value_i1, value_i2, ...] - one row per phenotype
    """
    width = 1 + len(columns)
    rows = [
        [block_title] + [''] * (width - 1),
        [''] * width,
        ['Phenotype', 'Model'] + [''] * (width - 2),
        [''] + list(columns),
    ]
    rows.extend(pivot_data_rows(pivot_df, columns))
    return rows


def all_pivot_rows(
    pivots: "dict[str, pd.DataFrame]", model_order: "list[str]",
    block_titles: "tuple[str, ...]" = PIVOT_BLOCK_TITLES, note: "str | None" = None,
) -> "list[list]":
    """Every block's own rows (see `pivot_block_rows()` - title, blank,
    group header, column header, then one row per phenotype), stacked
    with exactly ONE additional blank separator row between consecutive
    blocks. Iterated in `block_titles` order (fixed - defaults to
    `PIVOT_BLOCK_TITLES`, this module's own 12 titles in
    'Pearson correlation' block-then-'MSE' block order, matching
    `metric_plot.py`'s own `metrics = ['Pearson correlation', 'MSE']`
    order), skipping any title absent from `pivots`.

    Requirements.md item 8's attached example.csv was missing the title
    row for its second block ('MSE') - confirmed by the person to be an
    oversight in how that example was put together, not the intended
    layout - so every block here, including the second and any further
    one, gets the exact same shape.

    ver4-25, R2: a block whose own `pivot_df` has exactly ONE column
    (a scenario-constant metric - see `_scenario_constant_metric_blocks()`
    in `build_metric_summary_pivot()`) is rendered using ITS OWN column
    (`pivot_df.columns`), not the global `model_order` - see
    `pivot_block_rows()`'s own docstring. Every other block (more than
    one column, or genuinely empty) is unaffected and still uses
    `model_order` exactly as before this update.

    Update ID ver4-9 (blueprint SS2.2.4): PROMOTED from `_all_pivot_rows`
    to a public name, and gained two new parameters:

    `block_titles` : lets `diversity_summary.py` pass its OWN 27-title
        catalogue (`DPT_TERMS` outer x `PIVOT_STATISTICS` inner, plus 3
        `DPT_CV_TERMS` x CV blocks appended, Update ID ver4-11) through
        this exact same stacking logic, rather than a second,
        independently-drifting copy of it - a lateral (I2), not upward,
        reuse.
    `note` : an optional single line of text appended, as its own row,
        after a blank separator following the LAST block - used for
        `MEDIAN_SE_NOTE` here, and for `diversity_summary.py`'s own
        first/second/third-term explanation there. `None` (the default)
        reproduces the pre-ver4-9 behaviour exactly (no trailing row at
        all).

    The old, leading-underscore name is kept as a thin alias below.
    """
    width = 1 + len(model_order)
    all_rows: "list[list]" = []
    first = True
    for block_title in block_titles:
        if block_title not in pivots:
            continue
        if not first:
            all_rows.append([''] * width)
        first = False
        _pivot_df = pivots[block_title]
        # ver4-25, R2: see this function's own docstring.
        _columns = list(_pivot_df.columns) if _pivot_df.shape[1] == 1 else model_order
        all_rows.extend(pivot_block_rows(block_title, _pivot_df, _columns))
    if note:
        all_rows.append([''] * width)
        all_rows.append([note] + [''] * (width - 1))
    return all_rows


def write_pivot_csv(
    pivots: "dict[str, pd.DataFrame]", model_order: "list[str]", path: str,
    block_titles: "tuple[str, ...]" = PIVOT_BLOCK_TITLES, note: "str | None" = None,
) -> None:
    """Writes `pivots` (see `build_metric_summary_pivot()`) to `path` in
    exactly the block-stacked layout Requirements.md item 8's attached
    example.csv shows. `utf-8-sig` (a leading BOM) plus the `csv` module's
    own default `'\\r\\n'` line terminator match that example file's own
    encoding/line-endings exactly - both are what a spreadsheet
    application typically writes for 'CSV UTF-8', which is almost
    certainly how the attached example was produced.

    Update ID ver4-9: PROMOTED from `write_metric_summary_pivot_csv` to
    this name (blueprint SS2.2.4), gaining the same `block_titles`/`note`
    passthrough as `all_pivot_rows()`. The old name is kept as a thin
    alias below.
    """
    rows = all_pivot_rows(pivots, model_order, block_titles=block_titles, note=note)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerows(rows)


def write_pivot_worksheet(
    workbook, pivots: "dict[str, pd.DataFrame]", model_order: "list[str]", sheet_name: str = 'summary',
    block_titles: "tuple[str, ...]" = PIVOT_BLOCK_TITLES, note: "str | None" = None,
):
    """Writes `pivots` into a NEW worksheet named `sheet_name`, inserted
    FIRST (index 0) in `workbook` - so opening `Metric_summary.xlsx`
    directly lands on a sheet laid out exactly like the attached
    example.csv (Excel's own default active sheet is the first one).
    Each block's own title cell (row 1 of its block) is bolded for
    readability; every other cell is left unstyled. `workbook` is the
    live `openpyxl.Workbook` object (`pd.ExcelWriter(..., engine=
    'openpyxl').book`) - called from inside the SAME `with pd.ExcelWriter
    (...) as writer:` block that writes the other sheets, so `openpyxl`
    is already guaranteed importable by the time this runs (no separate
    ImportError handling needed here).

    Update ID ver4-9: PROMOTED from `_write_pivot_worksheet` to a public
    name (blueprint SS2.2.4), gaining the same `block_titles`/`note`
    parameters as `all_pivot_rows()`. The bold-title membership test
    below now matches against the CALLER's own `block_titles` (previously
    hardcoded to the module-level `PIVOT_METRICS` tuple) - REQUIRED, not
    cosmetic: with the old membership test, none of the twelve new block
    titles (e.g. 'Pearson correlation median') would ever match plain
    'Pearson correlation'/'MSE', so not a single title would be bolded at
    all. The old, leading-underscore name is kept as a thin alias below.
    """
    from openpyxl.styles import Font
    ws = workbook.create_sheet(sheet_name, 0)
    bold = Font(bold=True)
    for row in all_pivot_rows(pivots, model_order, block_titles=block_titles, note=note):
        ws.append(row)
    # Bold every title-row cell - identified by an exact match against
    # the caller's OWN block_titles in column A, which a real phenotype
    # name essentially never collides with; a false-positive bold on a
    # same-named phenotype would be a harmless cosmetic edge case, not a
    # data bug.
    _title_set = set(block_titles)
    for row in ws.iter_rows(min_col=1, max_col=1):
        if row[0].value in _title_set:
            row[0].font = bold
    return ws


# ---------------------------------------------------------------------------
# ver4-9 backward-compatible aliases (blueprint RK-8): these leading-
# underscore/old-named functions had exactly one in-tree consumer each
# (this module itself) before ver4-9 - kept as thin aliases for one
# release in case any external caller imported them directly.
# ---------------------------------------------------------------------------
_pivot_data_rows = pivot_data_rows
_pivot_block_rows = pivot_block_rows
_all_pivot_rows = all_pivot_rows
_write_pivot_worksheet = write_pivot_worksheet
write_metric_summary_pivot_csv = write_pivot_csv


def build_varcomp_summary_pivot(
    varcomp: "pd.DataFrame", metric: "pd.DataFrame | None" = None,
    phenotype_order: "list[str] | None" = None,
) -> "dict[str, pd.DataFrame]":
    """ver4-25, R10 (design blueprint stage 3, AC10.5): the `Metric_
    summary.xlsx` `variance_components` sheet's own pivot blocks - SIX
    blocks (`PIVOT_STATISTICS`) per "column family", one family per
    heritability-like statistic `Preprocess/asreml_blup.py::fit_
    concurrent_variance_model()` computes (`h2_plot`, `H2_classic`,
    `H2_cullis_train`) - eighteen blocks total, phenotype rows, mirroring
    `build_metric_summary_pivot()`'s own layout exactly (this reuses the
    SAME `six_statistics()`/`pivot_block_rows()`/`all_pivot_rows()`
    machinery - lateral reuse, I2).

    Each of the three statistics is a property of the SCENARIO
    (population x phenotype x ratio x replicate), repeated identically
    across every `component` row `asreml_varcomp` has for that scenario
    (`fit_concurrent_variance_model()`'s own broadcast) - so, exactly
    like R2's `_scenario_constant_metric_blocks()`, this collapses to ONE
    row per scenario (`.groupby(task_keys)[...].first()`, which also
    transparently skips a scenario whose fit failed and left every
    column NaN) BEFORE summarising across replicates within a phenotype.
    Unlike `_scenario_constant_metric_blocks()`, there is no "per-model"
    dimension to worry about disagreeing here at all (one fit per task,
    never one per model), so the full six-statistic treatment
    (median/mean, both standard errors, both "+/-" combined strings) is
    reported, not just median/mean.

    Parameters
    ----------
    varcomp : `ASReml_variance_components.csv`'s own frame (any source,
        e.g. `batch_reader.ResultSet.asreml_varcomp()`).
    metric : optional - `Metric.csv`'s own frame, consulted ONLY to
        resolve a shared `phenotype_order` when the caller doesn't pass
        one explicitly (so this sheet's own row order matches the main
        `summary` sheet's without the caller needing to compute it
        twice) - never read for anything else.
    phenotype_order : as `build_metric_summary_pivot()`.

    Returns
    -------
    dict, keyed by `'<column> <statistic>'` strings (e.g. `'h2_plot
    median'`), each a single-column, `phenotype_order`-indexed
    DataFrame. Empty dict if `varcomp` is empty/None, or lacks the four
    task-identity columns.
    """
    if varcomp is None or varcomp.shape[0] == 0:
        return {}
    task_keys = ['population', 'phenotype', 'ratio', 'sample']
    if not all(k in varcomp.columns for k in task_keys):
        return {}

    if phenotype_order is None:
        if metric is not None and metric.shape[0] != 0 and 'phenotype' in metric.columns:
            phenotype_order = pd.unique(metric['phenotype']).tolist()
        else:
            phenotype_order = pd.unique(varcomp['phenotype']).tolist()
    else:
        _seen_p = set(phenotype_order)
        phenotype_order = list(phenotype_order) + [
            p for p in pd.unique(varcomp['phenotype']) if p not in _seen_p
        ]

    _columns_present = [c for c in VARCOMP_SUMMARY_COLUMNS if c in varcomp.columns]
    if not _columns_present:
        return {}
    per_task = varcomp.groupby(task_keys, sort=False, dropna=False)[_columns_present].first()
    grouped = per_task.groupby(level='phenotype', sort=False, dropna=False)

    pivots: "dict[str, pd.DataFrame]" = {}
    for col in _columns_present:
        stats = six_statistics(grouped, col)
        for s in PIVOT_STATISTICS:
            series = stats[s].reindex(phenotype_order)
            wide = series.to_frame(name=col)
            wide.index.name = 'phenotype'
            wide.columns.name = 'model'
            pivots[f'{col} {s}'] = wide
    return pivots


def apply_label_style_to_tables(sheets, pivots, model_order, label_style):
    """ver4-95 (`MODEL_LABEL_STYLE`): `(sheets, pivots, model_order)` with the model labels'
    TEXT shortened - the 'model' column of every long sheet, the model columns of every pivot block
    and the pivot column order list - AFTER everything was ordered and aggregated on the full
    labels (so no number, row or column moves). 'full' returns the inputs unchanged (same
    objects). Two labels that would collide keep their full text (`display_label_map`).
    Shared with `diversity_summary.write_dpt_summary`. Pure apart from one NOTE."""
    if normalise_label_style(label_style) != 'short':
        return sheets, pivots, model_order
    labels = list(model_order or [])
    for df in (sheets or {}).values():
        if hasattr(df, 'columns') and 'model' in df.columns:
            labels += list(pd.unique(df['model']))
    shown = display_label_map(labels, 'short')
    if not any(k != v for k, v in shown.items()):
        return sheets, pivots, model_order
    new_sheets = {}
    for name, df in (sheets or {}).items():
        if hasattr(df, 'columns') and 'model' in df.columns:
            df = df.copy()
            df['model'] = df['model'].map(lambda m: shown.get(m, m))
        new_sheets[name] = df
    new_pivots = {title: (df.rename(columns=shown) if hasattr(df, 'rename') else df)
                  for title, df in (pivots or {}).items()}
    new_order = [shown.get(m, m) for m in model_order] if model_order is not None else None
    return new_sheets, new_pivots, new_order


def run_dispersion_for_summary(cfg, load_prediction, log=print) -> "pd.DataFrame | None":
    """ver4-100 (O02): the table for the optional 'dispersion' sheet of Metric_summary.xlsx.

    `cfg['METRIC_SUMMARY_DISPERSION']` (default False) switches it on; off, nothing is read and
    ``None`` is returned, so `write_metric_summary()` behaves exactly as before. `load_prediction`
    is a zero-argument callable returning the combined test Prediction_result table (columns
    ``actual`` plus one per model, and the task columns). The calibration slope (regression of the
    actual value on the prediction) is computed per iteration with
    `tools.prediction_diagnostics.run()` and summarised over the iterations. Nothing is refitted.

    NEVER fatal: a missing/unreadable table or any computation error logs one line and returns
    ``None`` (the metric summary is then written without the sheet).

    Returns a DataFrame with the task columns (population / phenotype / ratio, as present), ``model``,
    ``n_iterations``, ``slope_mean``, ``slope_sd``, ``share_slope_below_0.9``,
    ``share_slope_above_1.1``, or ``None``."""
    if not bool((cfg or {}).get('METRIC_SUMMARY_DISPERSION', False)):
        return None
    try:
        pred = load_prediction()
        if pred is None or len(pred) == 0:
            log("Metric summary: the 'dispersion' sheet was skipped (no test predictions found).")
            return None
        from tools.prediction_diagnostics import run as _run_diag
        _, summary = _run_diag(pred)
        if summary is None or len(summary) == 0:
            log("Metric summary: the 'dispersion' sheet was skipped (no slope could be computed).")
            return None
        summary = summary.rename(columns={
            'n_tasks': 'n_iterations',
            'share_slope_below': 'share_slope_below_0.9',
            'share_slope_above': 'share_slope_above_1.1'})
        keep = [c for c in summary.columns if c not in ('pearson_mean',)]
        return summary[keep].reset_index(drop=True)
    except Exception as exc:  # reporting artefact: never stop a finished run
        log(f"Metric summary: the 'dispersion' sheet was skipped ({type(exc).__name__}: {exc}).")
        return None


def write_metric_summary(metric: "pd.DataFrame", RESULT_NAME: str, *, create: bool = True,
                          model_order: "list[str] | None" = None,
                          extra_metrics: "tuple[str, ...]" = ACCURACY_METRIC_COLUMNS,
                          asreml_varcomp: "pd.DataFrame | None" = None,
                          label_style: str = 'full',
                          dispersion: "pd.DataFrame | None" = None) -> "list[str]":
    """Write `build_metric_summary(metric)`'s three sheets, PLUS
    (Requirements.md item 8) `build_metric_summary_pivot(metric,
    model_order)`'s wide pivot table as a new FIRST sheet named
    'summary', to `Result/<RESULT_NAME>/Metric_summary.xlsx` - or, when
    `openpyxl` isn't importable in this environment, to CSVs: the three
    long-format ones named for their sheet (unchanged), plus a new
    `Metric_summary.csv` holding the pivot table in the exact
    block-stacked layout Requirements.md item 8's attached example.csv
    shows. Returns the list of paths actually written (empty when
    nothing was).

    `model_order` : the exact model column order/inclusion for the pivot
    table - normally the SAME `model_labels` list the caller's own
    `metric_plot()` call passes as `MODEL` (`hue_order=MODEL` there), so
    the pivot table's model order is GUARANTEED identical to the violin
    plots' own (Requirements.md item 8's own explicit requirement) - see
    `build_metric_summary_pivot()`'s own docstring for the fallback used
    when omitted, and for how the phenotype row order is derived (no
    separate parameter needed there - it already matches
    `metric_plot()`'s own default facet order without one).

    `create=False` (the `METRIC_SUMMARY_CREATE` config key, default
    `True`): nothing written, nothing logged - matches the existing
    `*_CREATE` flags' own precedent (`METRIC_PLOT_CREATE` et al. also log
    nothing when off) rather than this module inventing a new "feature
    disabled" line pattern.

    `metric` empty (every task in this run was skipped): nothing written,
    one informational line logged - the SAME legitimate-outcome treatment
    `assemble()` already gives an empty `Metric.csv`, never fatal.

    `Metric_summary.xlsx`/`Metric_summary.csv` are deliberately NOT added
    to `checkpoint_utils.RESULT_FILE_NAMES` - both are reporting
    artefacts derived from `Metric.csv`, not one of `GP()`'s own
    accumulators, and have no business in the checkpoint/resume naming
    authority.

    `asreml_varcomp` : ver4-25, R10 (design blueprint stage 3) -
        `ASReml_variance_components.csv`'s own frame (e.g. `batch_reader.
        ResultSet.asreml_varcomp()`), optional (`None` - the default -
        reproduces every pre-ver4-25 call exactly, I11). When given and
        non-empty, adds a further `variance_components` sheet (or
        `Metric_summary_variance_components.csv` degrade) - see
        `build_varcomp_summary_pivot()`.

    `label_style` : ver4-95 (config key `MODEL_LABEL_STYLE`) - 'full' (default; every pre-ver4-95
        call is unchanged) or 'short'. With 'short' the TEXT of the model labels in the written
        sheets / CSVs drops the ``'Super learner__'`` prefix (see
        `pipeline_utils.display_label_map`); Metric.csv itself is never touched, and two labels
        that would end up identical both keep their full text.

    `dispersion` : ver4-100 (O02) - the table returned by `run_dispersion_for_summary()` (``None`` or
        empty - the default - reproduces every earlier call exactly). When given and non-empty, adds a
        further sheet named 'dispersion' (or `Metric_summary_dispersion.csv` in the openpyxl-less
        degrade): per model and task group, the mean and SD over the iterations of the calibration
        slope and the share of iterations below 0.9 / above 1.1. Metric.csv is never touched.
    """
    if not create:
        return []
    if metric is None or metric.shape[0] == 0:
        print("[metric_summary] Metric.csv has no rows - no summary written.")
        return []

    sheets = build_metric_summary(metric, model_order=model_order, extra_metrics=extra_metrics)
    pivots = build_metric_summary_pivot(metric, model_order=model_order, extra_metrics=extra_metrics)
    # ver4-25, R10: the `variance_components` sheet's own blocks - see
    # `build_varcomp_summary_pivot()`'s own docstring. Empty dict (no
    # sheet/CSV written at all - see both write branches below) whenever
    # `asreml_varcomp` is `None`/empty, exactly like `pivots` itself
    # degrades to nothing when `metric` is empty.
    varcomp_pivots = build_varcomp_summary_pivot(asreml_varcomp, metric=metric) if asreml_varcomp is not None else {}
    # Follow-up requirement (BLUEs/BLUPs accuracy-metrics phase): `pivots`
    # may now hold MORE than the original 12 blocks (six more per
    # selected new metric) - `tuple(pivots.keys())` (insertion-ordered:
    # Pearson, MSE, then each present extra metric, each in
    # PIVOT_STATISTICS order) replaces the fixed `PIVOT_BLOCK_TITLES`
    # global at both write sites below, so a run with the new metrics on
    # gets every block written, not just the original 12; a run with them
    # off produces `tuple(pivots.keys()) == PIVOT_BLOCK_TITLES` exactly,
    # so this is a no-op change for every existing config.
    _block_titles = tuple(pivots.keys())
    # Same resolution `build_metric_summary_pivot()` used internally to
    # build `pivots` above (fall back to order-of-first-appearance, then
    # canonicalise) - recomputed here so the 'summary' worksheet's own
    # column order (written via `_pivot_model_order` below) is guaranteed
    # identical to what `pivots`' own columns actually ended up in.
    _pivot_model_order = model_order if model_order is not None else pd.unique(metric['model']).tolist()
    _pivot_model_order = canonical_model_order(_pivot_model_order)
    # ver4-95: the order above was fixed on the FULL labels (so it is identical to the violin plots'
    # and to every earlier run); only the text written to the file is shortened.
    sheets, pivots, _pivot_model_order = apply_label_style_to_tables(
        sheets, pivots, _pivot_model_order, label_style)
    if dispersion is not None and len(dispersion) and 'model' in dispersion.columns:
        _disp_map = display_label_map(dispersion['model'].tolist(), label_style)
        dispersion = dispersion.copy()
        dispersion['model'] = dispersion['model'].map(lambda m: _disp_map.get(m, m))
    else:
        dispersion = None
    result_dir = './Result/' + RESULT_NAME + '/'
    os.makedirs(result_dir, exist_ok=True)

    if 'ratio' in metric.columns:
        _ratio_numeric = pd.to_numeric(metric['ratio'], errors='coerce')
        if _ratio_numeric.notna().all() and (_ratio_numeric == -1).all():
            print("[metric_summary] SCENARIO='between': ratio is the -1 sentinel for every "
                  "task, so the 'phenotype_population_ratio' sheet is identical to "
                  "'phenotype_population'.")

    xlsx_path = result_dir + 'Metric_summary.xlsx'
    try:
        # Lazy on purpose (I2/I12): openpyxl==3.1.5 IS pinned in BOTH
        # environment_linux.yml and environment_windows.yml (finding F3 -
        # an earlier version of this comment/warning incorrectly implied
        # it was Windows-only) - but this stays a lazy, guarded import
        # regardless, so this whole module still degrades gracefully
        # (rather than failing to import at all) in any environment that
        # genuinely lacks it (e.g. a hand-built venv that skipped it).
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            for sheet_name in SHEET_NAMES:
                sheets[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)
            # ver4-25, R10: written BEFORE 'summary' below, so that
            # 'summary' (create_sheet(..., 0), always index 0) ends up
            # FIRST and this sheet lands at index 1, right after it -
            # write_pivot_worksheet() itself has no "insert at position N"
            # parameter, only "always index 0", so ordering these two
            # calls is what fixes their final positions.
            if dispersion is not None:
                dispersion.to_excel(writer, sheet_name='dispersion', index=False)
            if varcomp_pivots:
                write_pivot_worksheet(writer.book, varcomp_pivots, [], sheet_name='variance_components',
                                       block_titles=VARCOMP_SUMMARY_BLOCK_TITLES)
            # Requirements.md item 8: inserted FIRST (index 0) - see
            # write_pivot_worksheet()'s own docstring. Update ID ver4-9:
            # now 12 blocks (PIVOT_BLOCK_TITLES) with MEDIAN_SE_NOTE
            # appended as a trailing explanatory row.
            if pivots:
                write_pivot_worksheet(writer.book, pivots, _pivot_model_order, sheet_name='summary',
                                       block_titles=_block_titles, note=MEDIAN_SE_NOTE)
        print(f"[metric_summary] Metric summary workbook written: {xlsx_path}")
        return [xlsx_path]
    except (ImportError, ValueError):
        csv_paths = []
        if pivots:
            pivot_path = f'{result_dir}Metric_summary.csv'
            write_pivot_csv(pivots, _pivot_model_order, pivot_path,
                             block_titles=_block_titles, note=MEDIAN_SE_NOTE)
            csv_paths.append(pivot_path)
        # ver4-25, R10: the same CSV-degrade treatment as every other
        # sheet - block_titles-only columns (model_order=[] - every block
        # here has exactly one column, so all_pivot_rows() always takes
        # its own single-column special case; see that function's
        # docstring).
        if varcomp_pivots:
            varcomp_path = f'{result_dir}Metric_summary_variance_components.csv'
            write_pivot_csv(varcomp_pivots, [], varcomp_path, block_titles=VARCOMP_SUMMARY_BLOCK_TITLES)
            csv_paths.append(varcomp_path)
        if dispersion is not None:
            disp_path = f'{result_dir}Metric_summary_dispersion.csv'
            dispersion.to_csv(disp_path, index=False)
            csv_paths.append(disp_path)
        for sheet_name in SHEET_NAMES:
            path = f'{result_dir}Metric_summary_{sheet_name}.csv'
            sheets[sheet_name].to_csv(path, index=False)
            csv_paths.append(path)
        print("[metric_summary] WARNING: openpyxl is not installed in this environment, so")
        print(f"[metric_summary] {xlsx_path} could not be written. The same tables were")
        print("[metric_summary] written as CSV instead: Metric_summary.csv (the pivot table),")
        if varcomp_pivots:
            print("[metric_summary] Metric_summary_variance_components.csv,")
        if dispersion is not None:
            print("[metric_summary] Metric_summary_dispersion.csv,")
        print("[metric_summary] Metric_summary_phenotype.csv,")
        print("[metric_summary] Metric_summary_phenotype_population.csv,")
        print("[metric_summary] Metric_summary_phenotype_population_ratio.csv")
        print("[metric_summary] To get the Excel file, install openpyxl into this environment:")
        print("[metric_summary]     conda install -c conda-forge openpyxl      (or: pip install openpyxl)")
        return csv_paths
