"""
asreml_accuracy_metrics.py
============================
MULTI-OPTION / ALWAYS-ON RANKING METRICS UPDATE (Requirements.md items 3-5):
  * ``Spearman correlation`` and ``Top-k Coincidence`` are NO LONGER ASReml-
    only, and no longer compare a model against an independent BLUP
    "reference" fit. They are now computed exactly like ``Pearson
    correlation``/``MSE`` always were: predicted test values against the
    SAME test target that Pearson/MSE use for that row (the raw phenotype
    when ASReml is off; the BLUE / BLUP-without-GRM / BLUP-with-GRM target
    when the row belongs to an ASReml option) - see
    ``compute_ranking_metrics_row()``. ``compute_accuracy_metrics_row()``
    calls it whenever the caller's ctx carries ``actual_values``.
  * ``Adjusted prediction accuracy`` (and ``Heritability``) remain ASReml-
    only and opt-in (``ctx['enabled']``).
  * ``czekanowski_coefficient()`` is the new MODEL-vs-MODEL analogue of
    Top-k Coincidence used by the Czekanowski scatter-plot matrix.

Requirements.md (BLUEs/BLUPs accuracy-metrics phase), items 1-7 - a Layer
3d shared authority (zero project imports, mirroring the discipline
`model_registry.py` / `circos_geometry.py` / `asreml_syntax.py` already
follow - architecture doc SS2): the single place that computes the three
additional per-row accuracy metrics EasiGP reports whenever ASReml is
enabled AND the user has opted in:

  - **Adjusted prediction accuracy** = predictive ability / sqrt(H^2),
    converting a model's raw correlation with its own training target
    into an estimate of r(y_hat, true genetic value) on the breeder's own
    scale (BLUEs_vs_BLUPs.md, Q4 "Answer 4" - r(y_hat, BLUE) = r(y_hat, g)
    x sqrt(H^2)).
  - **Spearman correlation** - the rank-concordance between a model's own
    predicted test-set values and an INDEPENDENT BLUP-without-GRM
    reference value per individual (never the model's own training
    target, whatever that happens to be for this run's own strategy).
  - **Top-k Coincidence** (k = 5/10/15/20/25/30, a PERCENTAGE of the test
    set - confirmed against BLUEs_vs_BLUPs.md's own "top-k%" framing and
    Requirements.md item 8's own "Top-10%" bubble-size reference) - the
    share of the reference's own top-k% individuals that also appear in
    the model's own predicted top-k% (or bottom-k%, when the phenotype's
    own configured "direction" is 'lower').

Both `heritability` (H^2) and the Spearman/Top-k `reference_values` come
from a BLUP-without-GRM ASReml fit - either the run's own MAIN fit
(whenever `genotype_effect_strategy == 'blup_iid'`) or an extra
"companion" fit built alongside it (every other strategy) - see
`genomic_prediction.py`'s own `_asreml_get_accuracy_metrics_context()`
for exactly which model supplies what, per Answers.md items 1-2.

This module never touches ASReml/rpy2/R itself - it only consumes the
plain `AsremlResult.varcomp`/`.values` frames `Preprocess/asreml_blup.py`
already produces, and plain Python/NumPy arrays for the ranking metrics.
Imported by `genomic_prediction.py` (per-task/per-model computation),
`models/ensemble.py` and the four weighted-ensemble modules (their own
ensemble-level rows), and `metric_summary.py`/`accuracy_metrics_plot.py`
(column-name constants only - neither re-derives any of this module's
maths).
"""

from __future__ import annotations

import math
import warnings
from typing import Callable, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from scipy.stats import pearsonr, spearmanr
except Exception:  # pragma: no cover - scipy is a hard pinned dependency
    pearsonr = None
    spearmanr = None

# Requirements.md item 4: "the k needs to be calculated when k is 5, 10,
# 15, 20, 25 and 30" - confirmed (Answers.md item 4) to be a PERCENTAGE of
# the test set, not an absolute count.
TOPK_PERCENTAGES = (5, 10, 15, 20, 25, 30)

ADJUSTED_ACCURACY_COLUMN = 'Adjusted prediction accuracy'
SPEARMAN_COLUMN = 'Spearman correlation'
TOPK_COLUMNS = tuple(f'Top-{k} Coincidence' for k in TOPK_PERCENTAGES)
# Follow-up requirement (after the original Requirements.md phase): report
# the Cullis-approximate heritability itself in Metric.csv too, whenever
# the new metrics are selected - this is the SAME H^2
# `adjusted_prediction_accuracy()` divides by (see
# `cullis_heritability()`), surfaced directly rather than left implicit.
HERITABILITY_COLUMN = 'Heritability'
# Follow-up requirement (classic vs. Cullis comparison): the CLASSIC
# (entry-mean basis, `sigma_g^2 / (sigma_g^2 + sigma_e^2 / r_bar [+ ...])`)
# heritability, computed from the SAME ASReml reference fit the Cullis
# `Heritability` above comes from (see `Preprocess/asreml_blup.py::
# classic_heritability_from_fit()` and `AsremlResult.heritability_classic`),
# so the two are directly comparable. Reported in its own column placed
# IMMEDIATELY BEFORE `HERITABILITY_COLUMN` (see `ACCURACY_METRIC_COLUMNS`).
# `HERITABILITY_COLUMN`'s own name is deliberately left unchanged so every
# existing config/resume/consumer keeps working.
HERITABILITY_CLASSIC_COLUMN = 'Heritability (classic)'

# Requirement 7 ("eight new columns...in total" when all three original
# metrics are on: 1 adjusted-accuracy + 1 Spearman + 6 Top-k) PLUS the
# follow-up heritability column - nine in total - this tuple is the
# single authority for that full column set, in the FIXED display order
# every consumer (Metric.csv, metric_summary.py, accuracy_metrics_plot.py)
# renders them in.
ACCURACY_METRIC_COLUMNS = (
    HERITABILITY_CLASSIC_COLUMN, HERITABILITY_COLUMN, ADJUSTED_ACCURACY_COLUMN, SPEARMAN_COLUMN,
) + TOPK_COLUMNS

# ver4-25, R2 (root cause): `compute_accuracy_metrics_row()` copies the
# SAME `ctx['heritability']` value into every model's row for a given
# task (Heritability is a property of the SCENARIO - population x
# phenotype x ratio x replicate - not of any individual model), whereas
# every other name in ACCURACY_METRIC_COLUMNS above genuinely varies by
# model. `metric_summary.build_metric_summary_pivot()` used to treat
# Heritability exactly like a per-model metric anyway, producing a full
# six-statistic table with one (redundant, always-equal-within-a-task)
# column per model plus standard-error blocks that have no real meaning
# for a constant. This tuple is the single authority both
# `metric_summary.py` and any other reporting consumer must check before
# deciding whether to pivot a metric "per model" or "per scenario" -
# two-membership today (the classic and Cullis heritabilities are both
# scenario-constant), declared as a tuple so a future scenario-constant
# metric only needs adding here.
SCENARIO_CONSTANT_METRIC_COLUMNS = (HERITABILITY_CLASSIC_COLUMN, HERITABILITY_COLUMN)

DEFAULT_DIRECTION = 'higher'
VALID_DIRECTIONS = ('higher', 'lower')

# Requirements.md item 5 (follow-up) - "the adjusted prediction accuracy
# went beyond one. Need to think about how the results can be
# interpreted." `adjusted_prediction_accuracy()` divides a correlation
# by sqrt(heritability); since BOTH quantities are ESTIMATES (not the
# true population values the formula is derived for), the ratio has no
# hard ceiling in practice and a value > 1 is a real, expected,
# non-error outcome rather than a bug - see this constant's own
# docstring below and `_maybe_warn_adjusted_accuracy_above_one()` for
# where/how this is surfaced. No numerical change is made anywhere for
# this - the raw computed value is reported as-is; only its
# interpretation is clarified, in both the run log and the GUI
# (main_app.py's own 'Adjusted prediction accuracy' caption, Tab 2).
ADJUSTED_ACCURACY_ABOVE_ONE_EXPLANATION = (
    "Adjusted prediction accuracy = Pearson correlation / sqrt(Heritability) can legitimately "
    "come out ABOVE 1, even though a true correlation cannot. Both the numerator and the "
    "denominator here are ESTIMATES, not the true population quantities the formula assumes: "
    "the numerator is a correlation computed from one finite test set, and Heritability itself "
    "is an approximate (Cullis et al. 2006) estimate that can come out too LOW relative to that "
    "same correlation - e.g. with a small or unbalanced test partition, or a design with "
    "limited replication. Dividing by an under-estimated sqrt(Heritability) inflates the ratio "
    "past its own theoretical ceiling of 1. A value above (or noticeably close to) 1 should "
    "therefore be read as 'this model is doing about as well as this dataset's own estimated "
    "heritability can support' rather than literally 'a correlation greater than one' - look at "
    "it together with the reported Heritability column (a low Heritability alongside a high "
    "Adjusted prediction accuracy is exactly the signature of this effect) rather than in "
    "isolation, and treat a large excess over 1 as a cue to sanity-check the Heritability "
    "estimate (e.g. test-set size, design balance) rather than as a literal accuracy figure."
)

# Module-level (never reset mid-process) - printed at most ONCE per
# Python process, the FIRST time any row this process computes actually
# exceeds 1. Deliberately not printed on every occurrence: this metric
# is computed once per model per task (genomic_prediction.py's own
# dispatch loop) and again for every weighted/naive ensemble method, so
# an HPC run with many tasks could otherwise repeat an identical,
# already-understood explanation thousands of times in the log for no
# added information - the same "log the fact, not every instance of it"
# discipline `GP()` already applies to its other one-time NOTEs
# (architecture doc SS18, "Diagnostics as a design principle").
_warned_adjusted_accuracy_above_one = False


def _maybe_warn_adjusted_accuracy_above_one(value: float) -> None:
    """Prints `ADJUSTED_ACCURACY_ABOVE_ONE_EXPLANATION` to stdout (so it
    lands in the SAME run log `TimestampedWriter` already captures - see
    pipeline_utils.py) the first time `value` is a finite number greater
    than 1.0 in this process - a no-op every other time (including for
    `NaN`/`None`, and for every call after the first one that qualifies).
    Never raises; never changes `value` itself."""
    global _warned_adjusted_accuracy_above_one
    if _warned_adjusted_accuracy_above_one:
        return
    try:
        if value is not None and np.isfinite(value) and float(value) > 1.0:
            _warned_adjusted_accuracy_above_one = True
            print("[EasiGP NOTE] " + ADJUSTED_ACCURACY_ABOVE_ONE_EXPLANATION)
    except (TypeError, ValueError):
        pass


def cullis_heritability(
    varcomp: "object", values: "object", genotype_col: Optional[str],
) -> Optional[float]:
    """Approximate Cullis et al. (2006) generalised heritability,
    `H^2 = 1 - v_bar_delta / (2 * sigma_g^2)`, where `v_bar_delta` is the
    mean PAIRWISE prediction-error variance of two genotype BLUPs.
    Answers.md item 2 ("use the approximation approach to save
    computational time"): rather than requesting ASReml's own full
    average-SED matrix (`predict(..., sed = TRUE)` - expensive for any
    real genotype universe, and not currently requested anywhere in
    `Preprocess/asreml_fit.R`), this uses the standard practical
    approximation `v_bar_delta ~= 2 * mean(PEV_i)` (valid when pairwise
    prediction-error COVARIANCES are small relative to the marginal
    PEVs - the common case for a reasonably well-connected design), which
    simplifies the formula to::

        H^2 ~= 1 - mean(std_error_i ^ 2) / sigma_g^2

    Parameters
    ----------
    varcomp : the BLUP-without-GRM fit's own `AsremlResult.varcomp`
        (columns `component`, `estimate`, ...) - must have a genetic-
        variance row identifiable for `genotype_col`, using the exact
        same free-text component-name matching heuristic
        `Preprocess/asreml_blup.py::_estimate_heritability()` already
        uses (case-insensitive substring match against `genotype_col`).
    values : that SAME fit's own `AsremlResult.values` - must have a
        `std_error` column (NaN rows are dropped before averaging).
    genotype_col : the genotype/ID column name.

    Returns
    -------
    `float`, ordinarily in `(0, 1]` - or `None` (never raises, never
    fabricates a value) when the genetic-variance component cannot be
    identified, is non-positive, or no finite `std_error` is available.
    """
    if varcomp is None or values is None or not genotype_col:
        return None
    try:
        if varcomp.shape[0] == 0:
            return None
        components = varcomp['component'].astype(str)
        estimates = varcomp['estimate'].astype(float)
    except (KeyError, ValueError, AttributeError):
        return None

    genetic_mask = components.str.contains(genotype_col, case=False, regex=False)
    if not genetic_mask.any():
        return None
    try:
        genetic_variance = float(estimates[genetic_mask].iloc[0])
    except (ValueError, TypeError):
        return None
    if not np.isfinite(genetic_variance) or genetic_variance <= 0:
        return None

    try:
        std_error = values['std_error'].astype(float).to_numpy()
    except (KeyError, ValueError, AttributeError):
        return None
    std_error = std_error[np.isfinite(std_error)]
    if std_error.size == 0:
        return None
    mean_pev = float(np.mean(std_error ** 2))
    if not np.isfinite(mean_pev):
        return None

    heritability = 1.0 - (mean_pev / genetic_variance)
    return heritability if np.isfinite(heritability) else None


def adjusted_prediction_accuracy(
    pearson_r: Optional[float], heritability: Optional[float],
) -> float:
    """`pearson_r / sqrt(heritability)` - `NaN` (never raises) whenever
    either input is missing/non-finite or `heritability <= 0` (a
    non-positive heritability estimate makes the correction undefined,
    not merely small - see `cullis_heritability()`'s own docstring for
    why this can happen)."""
    if pearson_r is None or heritability is None:
        return float('nan')
    try:
        pearson_r = float(pearson_r)
        heritability = float(heritability)
    except (TypeError, ValueError):
        return float('nan')
    if not np.isfinite(pearson_r) or not np.isfinite(heritability) or heritability <= 0:
        return float('nan')
    return pearson_r / math.sqrt(heritability)


def _finite_pair_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.isfinite(a) & np.isfinite(b)


def spearman_correlation(predicted: Sequence[float], reference: Sequence[float]) -> float:
    """Spearman rank correlation between `predicted` and `reference`
    (paired by POSITION - the caller is responsible for aligning both
    arrays to the same individuals first). Pairs where either value is
    non-finite are dropped before ranking. `NaN` (never raises) with
    fewer than two usable pairs, or when either array is constant over
    the usable pairs (rank correlation is undefined)."""
    if spearmanr is None:
        return float('nan')
    predicted = np.asarray(predicted, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if predicted.shape[0] != reference.shape[0]:
        n = min(predicted.shape[0], reference.shape[0])
        predicted, reference = predicted[:n], reference[:n]
    mask = _finite_pair_mask(predicted, reference)
    if mask.sum() < 2:
        return float('nan')
    p, r = predicted[mask], reference[mask]
    if np.all(p == p[0]) or np.all(r == r[0]):
        return float('nan')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rho, _ = spearmanr(p, r)
    return float(rho) if np.isfinite(rho) else float('nan')


def topk_coincidence(
    predicted: Sequence[float], reference: Sequence[float], k_percent: float,
    direction: str = DEFAULT_DIRECTION,
) -> float:
    """Percentage (0-100) overlap between the `k_percent`% of individuals
    ranked highest (`direction='higher'`) or lowest (`'lower'`) by
    `predicted`, and the `k_percent`% ranked the same way by `reference`
    - "the share of the target's true top-k% captured in the predicted
    top-k%" (BLUEs_vs_BLUPs.md, Q4). Pairs where either value is
    non-finite are dropped first; `k_percent`'s own individual count is
    `round(n * k_percent / 100)`, floored at 1 and capped at `n`. `NaN`
    when no usable pair exists."""
    predicted = np.asarray(predicted, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if predicted.shape[0] != reference.shape[0]:
        n = min(predicted.shape[0], reference.shape[0])
        predicted, reference = predicted[:n], reference[:n]
    mask = _finite_pair_mask(predicted, reference)
    predicted, reference = predicted[mask], reference[mask]
    n = predicted.shape[0]
    if n == 0:
        return float('nan')
    k_n = int(round(n * float(k_percent) / 100.0))
    k_n = max(1, min(k_n, n))

    order_pred = np.argsort(predicted, kind='mergesort')
    order_ref = np.argsort(reference, kind='mergesort')
    if direction == 'lower':
        pred_set = set(order_pred[:k_n].tolist())
        ref_set = set(order_ref[:k_n].tolist())
    else:
        pred_set = set(order_pred[-k_n:].tolist())
        ref_set = set(order_ref[-k_n:].tolist())
    overlap = len(pred_set & ref_set)
    return 100.0 * overlap / k_n


def compute_ranking_metrics_row(
    predicted_test: Optional[Sequence[float]], actual_test: Optional[Sequence[float]],
    direction: str = DEFAULT_DIRECTION,
) -> Dict[str, float]:
    """`{SPEARMAN_COLUMN: rho, 'Top-k Coincidence' columns...}` for one
    model row - ALWAYS returns exactly `[SPEARMAN_COLUMN] + TOPK_COLUMNS`
    as keys (values `NaN` when the data is unusable), independent of
    ASReml. `predicted_test`/`actual_test` are paired by POSITION (the
    caller aligns them - every row-building site already has both from
    the SAME test frame Pearson/MSE are computed from), so the ranking
    metrics always describe the SAME target as that row's own Pearson
    correlation: the raw phenotype without ASReml, or - under an ASReml
    option - that option's own BLUE / BLUP target (Requirements.md item
    4: "Spearman correlation and Top-k Coincidence ... should be
    calculated using the same option").
    """
    if direction not in VALID_DIRECTIONS:
        direction = DEFAULT_DIRECTION
    predicted_arr = np.asarray([] if predicted_test is None else list(predicted_test), dtype=float)
    actual_arr = np.asarray([] if actual_test is None else list(actual_test), dtype=float)
    row: Dict[str, float] = {SPEARMAN_COLUMN: spearman_correlation(predicted_arr, actual_arr)}
    for k, col in zip(TOPK_PERCENTAGES, TOPK_COLUMNS):
        row[col] = topk_coincidence(predicted_arr, actual_arr, k, direction=direction)
    return row


def czekanowski_coefficient(
    pred_a: Sequence[float], pred_b: Sequence[float], k_percent: float,
    direction: str = DEFAULT_DIRECTION,
) -> float:
    """Czekanowski (Sorensen-Dice) similarity, in PERCENT (0-100), between
    the top-`k_percent`% sets two models select from the SAME individuals:
    `2 * |A & B| / (|A| + |B|) * 100`, where A/B are the sets of
    individuals ranked highest (`direction='higher'`) or lowest
    (`'lower'`) by `pred_a`/`pred_b`. It shares Top-k Coincidence's
    principle (overlap of top-k% sets) but compares TWO MODELS with each
    other instead of a model with the observed target. Both sets always
    hold `round(n * k/100)` (>= 1) individuals, so it equals the shared
    fraction of the selected set. Pairs where either value is non-finite
    are dropped first; `NaN` when no usable pair exists. Symmetric:
    `czekanowski_coefficient(a, b) == czekanowski_coefficient(b, a)`.
    """
    a = np.asarray(pred_a, dtype=float)
    b = np.asarray(pred_b, dtype=float)
    n = min(a.shape[0], b.shape[0])
    a, b = a[:n], b[:n]
    mask = _finite_pair_mask(a, b)
    a, b = a[mask], b[mask]
    n = a.shape[0]
    if n == 0:
        return float('nan')
    k_n = max(1, min(int(round(n * float(k_percent) / 100.0)), n))
    order_a = np.argsort(a, kind='mergesort')
    order_b = np.argsort(b, kind='mergesort')
    if direction == 'lower':
        set_a, set_b = set(order_a[:k_n].tolist()), set(order_b[:k_n].tolist())
    else:
        set_a, set_b = set(order_a[-k_n:].tolist()), set(order_b[-k_n:].tolist())
    return 100.0 * 2.0 * len(set_a & set_b) / (len(set_a) + len(set_b))


def compute_accuracy_metrics_row(
    predicted_test: Optional[Sequence[float]], id_test: Optional[Sequence],
    sample_pearson: Optional[float], ctx: Optional[Mapping],
) -> Dict[str, float]:
    """The single entry point every row-building site (the per-model
    dispatch, each weighted-ensemble method, and the naive ensemble) uses
    to get identical extra metric columns. Returns a dict with:

      * ALWAYS (whenever `ctx` carries `actual_values`, an `{id: test
        target}` lookup GP() builds for EVERY task, ASReml or not):
        `SPEARMAN_COLUMN` + `TOPK_COLUMNS`, from
        `compute_ranking_metrics_row()` against that SAME target;
      * ONLY when `ctx['enabled']` is truthy (ASReml "adjusted
        prediction accuracy" opted in): `HERITABILITY_CLASSIC_COLUMN`,
        `HERITABILITY_COLUMN` and `ADJUSTED_ACCURACY_COLUMN`.

    An EMPTY dict comes back when `ctx` is falsy, or neither block
    applies, so a caller that never resolved a ctx gains no new Metric.csv
    columns at all.

    Parameters
    ----------
    predicted_test : this row's own predicted test-set values.
    id_test : the individual id for each `predicted_test` entry, SAME
        length/order.
    sample_pearson : this row's own already-computed Pearson correlation
        against its OWN training target (reused as the adjusted-accuracy
        numerator when `ctx['accuracy_numerator_mode'] ==
        'reuse_pearson'` - i.e. whenever the option already IS BLUE, so
        the model's own target already is one).
    ctx : a dict with `enabled` (adjusted accuracy on/off), `direction`,
        `actual_values` (`{id: test target}`), and - only when `enabled` -
        `heritability`, `heritability_classic` (optional), `accuracy_numerator_mode`, `blue_values`
        (`{id: value}` or `None`). See `genomic_prediction.py`.
    """
    if not ctx:
        return {}

    predicted_test = [] if predicted_test is None else list(predicted_test)
    id_test = [] if id_test is None else [str(_id) for _id in id_test]
    n = min(len(predicted_test), len(id_test))
    predicted_arr = np.asarray(predicted_test[:n], dtype=float)
    id_test = id_test[:n]

    direction = ctx.get('direction', DEFAULT_DIRECTION)
    if direction not in VALID_DIRECTIONS:
        direction = DEFAULT_DIRECTION

    row: Dict[str, float] = {}

    if ctx.get('enabled'):
        # --- Heritability (follow-up requirement) ----------------------- #
        # Classic heritability FIRST, so it lands immediately before the
        # Cullis `Heritability` column in every consumer's column order.
        # `ctx.get(...)` - a ctx persisted/built before this column
        # existed simply has no such key -> NaN.
        heritability_classic = ctx.get('heritability_classic')
        row[HERITABILITY_CLASSIC_COLUMN] = float(heritability_classic) if (
            heritability_classic is not None and np.isfinite(heritability_classic)
        ) else float('nan')
        heritability = ctx.get('heritability')
        row[HERITABILITY_COLUMN] = float(heritability) if (
            heritability is not None and np.isfinite(heritability)
        ) else float('nan')

        # --- Adjusted prediction accuracy ------------------------------- #
        if ctx.get('accuracy_numerator_mode') == 'reuse_pearson':
            numerator = sample_pearson
        else:
            blue_values = ctx.get('blue_values') or {}
            blue_arr = np.array([blue_values.get(_id, np.nan) for _id in id_test], dtype=float)
            mask = _finite_pair_mask(predicted_arr, blue_arr)
            if mask.sum() >= 2 and pearsonr is not None and \
                    not np.all(predicted_arr[mask] == predicted_arr[mask][0]) and \
                    not np.all(blue_arr[mask] == blue_arr[mask][0]):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    numerator = float(pearsonr(predicted_arr[mask], blue_arr[mask])[0])
            else:
                numerator = float('nan')
        row[ADJUSTED_ACCURACY_COLUMN] = adjusted_prediction_accuracy(numerator, heritability)
        _maybe_warn_adjusted_accuracy_above_one(row[ADJUSTED_ACCURACY_COLUMN])

    # --- Spearman correlation + Top-k Coincidence (ALWAYS) ------------- #
    actual_values = ctx.get('actual_values')
    if actual_values is not None:
        actual_arr = np.array([actual_values.get(_id, np.nan) for _id in id_test], dtype=float)
        row.update(compute_ranking_metrics_row(predicted_arr, actual_arr, direction))

    return row


# ===========================================================================
# ver4-25, R3 (design blueprint stage 3) - persisting a task's own accuracy-
# metrics context, so the naive ensemble's own finalisation step
# (genomic_prediction.py, run once over the WHOLE assembled result_test
# after every task has already completed - architecture doc SS12.1) can
# resolve it from a plain accumulator lookup instead of a process-local,
# in-memory registry that has no entry for a task this exact process never
# itself dispatched (a resumed run's own pre-resume tasks, or a task
# finalised in a completely separate process by
# intra_task_parallel.finalise_task_from_merged()).
#
# The root cause (design blueprint E7): genomic_prediction.py used to store
# only enough state (mask_keys/target_populations) to RE-DERIVE a task's own
# ctx later, at finalisation time - which either found nothing (a resumed
# task) or, for a masked run, paid for a genuine fresh ASReml-R refit of
# something already correctly computed once, earlier, in that exact task's
# own dispatch loop. The fix here is simpler: persist the ALREADY-COMPUTED
# ctx dict itself (see genomic_prediction.py::_asreml_get_accuracy_metrics_
# context()'s own return shape) the instant it is first computed, so nothing
# downstream ever needs to recompute or refit anything.
# ===========================================================================

# Long-format role labels for ctx_to_rows()/ctx_resolver_from_frame() - a
# '__meta__' row (id='', value=NaN) carries the scenario-constant fields
# (direction/heritability/accuracy_numerator_mode) and, critically, is what
# lets a scenario with genuinely EMPTY blue_values/reference_values (a
# degraded context - see genomic_prediction.py::_asreml_degraded_accuracy_
# ctx()) still be told apart from a scenario this feature never processed
# at all (no row of any kind -> ctx_resolver_from_frame()'s own resolver
# returns None for it, exactly like "not enabled").
_CTX_META_ROLE = '__meta__'
_CTX_BLUE_ROLE = 'blue'
_CTX_REFERENCE_ROLE = 'reference'
# ver4-25, R8 (design blueprint SS2, Stage 6): a THIRD role, reusing this
# same long-format sidecar/accumulator rather than inventing a parallel
# one (deviation from the blueprint's own separate 'blue_weight' column -
# see genomic_prediction.py's own Step 6b call site and the ver4-25
# Change Summary Part 3 SS7 for the full rationale) - one row per test
# individual, `value` = that individual's own diagonal-mode BLUE-
# reliability scoring weight. Only ever written/read for `mode=
# 'diagonal'`; 'full' mode is not wired up this delivery (see
# genomic_prediction.py's own scope-decision note).
_CTX_TEST_WEIGHT_ROLE = 'test_weight'
# Classic-heritability follow-up: a FOURTH role on this same long-format
# sidecar - ONE row per scenario (`id=''`, `value` = the classic
# heritability), rather than a new column, so the persisted file's schema
# (`ASREML_ACCURACY_CTX_COLUMNS`, and checkpoint_utils' own header
# fallback copy of it) is UNCHANGED: a sidecar written before this
# existed still loads and resumes exactly as before (it simply has no
# such row -> classic heritability NaN for those scenarios), and an
# appended run never hits a header/column mismatch. Same "reuse the
# sidecar via a role" precedent as `_CTX_TEST_WEIGHT_ROLE`.
_CTX_H2_CLASSIC_ROLE = 'h2_classic'

# checkpoint_utils.RESULT_FILE_BASE_COLUMNS['asreml_accuracy_ctx'] keeps an
# independent copy of this same column list (used only as that module's own
# zero-row header fallback) - see that module's own comment for why the two
# cannot share one literal constant without violating either module's "zero
# project imports" discipline. Keep both in sync by hand if this changes.
ASREML_ACCURACY_CTX_COLUMNS = (
    'population', 'phenotype', 'ratio', 'sample', 'direction', 'heritability',
    'accuracy_numerator_mode', 'role', 'id', 'value',
)


def scenario_key(population, phenotype, ratio, sample) -> tuple:
    """Single authority for the `(population, phenotype, ratio, sample)`
    scenario-identity key every accuracy-metrics-context cache/sidecar
    uses - a plain 4-tuple of strings, so it works equally as an
    in-memory dict key (native Python types - a `ratio` 3-tuple included)
    and as a key reconstructed from a persisted CSV's own (already-
    stringified) columns (`ctx_resolver_from_frame()` below) - `str()` on
    a value already read back from CSV as a string is a no-op, so both
    directions agree.
    """
    return (str(population), str(phenotype), str(ratio), str(sample))


def ctx_to_rows(population, phenotype, ratio, sample, ctx: "Optional[Mapping]") -> "pd.DataFrame":
    """Long-format rows persisting ONE task's own accuracy-metrics
    context (see genomic_prediction.py::_asreml_get_accuracy_metrics_
    context() for `ctx`'s exact shape) to `ASReml_accuracy_context.csv
    [.gz]` - everything `compute_accuracy_metrics_row()` needs to
    reproduce every one of this scenario's metrics later, with no
    ASReml-R refit ever required again (ver4-25, R3).

    Always contributes exactly one `'__meta__'` row (`id=''`,
    `value=NaN`) whenever `ctx` is truthy, carrying the scenario-constant
    `direction`/`heritability`/`accuracy_numerator_mode` fields - this is
    what lets a scenario with genuinely EMPTY `blue_values`/
    `reference_values` (a degraded context) still be told apart, by
    `ctx_resolver_from_frame()`, from a scenario this feature never
    processed at all. Every `(id, value)` pair in `ctx['blue_values']`/
    `ctx['reference_values']` becomes one further row each, tagged
    `role='blue'`/`role='reference'` respectively. When the classic
    heritability is available (`ctx['heritability_classic']`, finite),
    ONE further `role='h2_classic'` row (`id=''`, `value` = that number)
    carries it - a role, not a column, so this file's schema is unchanged.

    Returns an EMPTY DataFrame (the caller's own accumulator therefore
    gains no rows for this task) when `ctx` is `None`/falsy - accuracy
    metrics disabled for this run, exactly matching `effect`/`hp_record`'s
    own "nothing to report" convention elsewhere in this codebase.
    """
    if not ctx:
        return pd.DataFrame(columns=ASREML_ACCURACY_CTX_COLUMNS)

    base = {
        'population': population, 'phenotype': phenotype, 'ratio': ratio, 'sample': sample,
        'direction': ctx.get('direction', DEFAULT_DIRECTION),
        'heritability': ctx.get('heritability'),
        'accuracy_numerator_mode': ctx.get('accuracy_numerator_mode'),
    }
    rows = [dict(base, role=_CTX_META_ROLE, id='', value=float('nan'))]
    _h2_classic = ctx.get('heritability_classic')
    try:
        _h2_classic_finite = _h2_classic is not None and np.isfinite(float(_h2_classic))
    except (TypeError, ValueError):
        _h2_classic_finite = False
    if _h2_classic_finite:
        rows.append(dict(base, role=_CTX_H2_CLASSIC_ROLE, id='', value=float(_h2_classic)))
    for role, values in ((_CTX_BLUE_ROLE, ctx.get('blue_values')),
                         (_CTX_REFERENCE_ROLE, ctx.get('reference_values'))):
        for _id, _value in (values or {}).items():
            rows.append(dict(base, role=role, id=str(_id), value=_value))
    return pd.DataFrame(rows, columns=list(ASREML_ACCURACY_CTX_COLUMNS))


def ctx_resolver_from_frame(
    frame: "Optional[pd.DataFrame]",
    test_weight_matrix_frame: "Optional[pd.DataFrame]" = None,
) -> "Callable[..., Optional[dict]]":
    """Builds a `(population, phenotype, ratio, sample, *, ids=None) ->
    ctx dict | None` resolver purely from a persisted/merged `ASReml_
    accuracy_context.csv[.gz]`-shaped DataFrame - the ver4-25 R3 fix's
    whole point: this replays numbers a task's own dispatch loop already
    computed once; it NEVER re-fits ASReml.

    The returned resolver always returns `None` (a legitimate "nothing to
    resolve for this scenario") when `frame` is `None`, empty, or missing
    the expected columns (e.g. an older run's accumulator, from before
    this file existed) - it never raises. The per-scenario lookup index
    is built ONCE, here, rather than re-scanning `frame` on every call -
    calling the resolver once per scenario at finalisation time is
    therefore cheap even for a many-task run.

    Returned ctx dicts always populate `blue_values` as a (possibly
    empty) dict, never `None` - `compute_accuracy_metrics_row()`'s own
    `ctx.get('blue_values') or {}` already treats the two identically, so
    no information is lost by not persisting the None/`{}` distinction.

    ver4-25, R8 Stage 6 completion: `test_weight_matrix_frame` (keyword-
    optional, default `None` - I11, every pre-existing call site is
    unaffected) is the persisted `ASReml_test_weight_matrix.csv[.gz]`
    long upper-triangle frame (`population, phenotype, ratio, sample,
    id1, id2, value` - see `checkpoint_utils.RESULT_FILE_NAMES
    ['asreml_test_weights']`'s own comment) that only a `'full'`-mode
    weighted run ever produces. Because a dense N x N matrix has no
    natural per-scenario ROW COUNT the way `test_weights` (one row per
    individual) does, resolving it needs the caller's own `ids` (the
    individual order the caller's `data['id']`/`Prediction_result_test`
    slice is already in) - so the returned resolver accepts an OPTIONAL,
    keyword-only `ids=` that pre-ver4-25-R8-Stage-6-completion callers
    never pass (defaulting to `None`, in which case `ctx['test_weight_
    matrix']` is always `None` too - a strict no-op, I11). When `ids` IS
    given and this scenario has persisted matrix rows covering every one
    of them, `ctx['test_weight_matrix']` is set to `(W, list(ids))` - a
    dense matrix aligned to `ids`' own order, ready to splat straight
    into `pipeline_utils.safe_regression_metrics(..., W=...)` - built via
    `blue_weights.weight_matrix_from_frame()`, the same reader this
    module's own "zero project imports outside of Layer 3d peers"
    discipline already permits (`blue_weights.py` itself has ZERO
    project imports, so this can never introduce an import cycle, unlike
    importing anything from a higher layer would).
    """
    import blue_weights

    if (frame is None or frame.shape[0] == 0
            or not set(ASREML_ACCURACY_CTX_COLUMNS).issubset(set(frame.columns))):
        index: Dict[tuple, dict] = {}
    else:
        index = {}
        # Force any lists in the 'ratio' column back to hashable tuples before grouping
        if 'ratio' in frame.columns:
            frame['ratio'] = frame['ratio'].apply(lambda x: tuple(x) if isinstance(x, list) else x)

        for key_values, group in frame.groupby(
            ['population', 'phenotype', 'ratio', 'sample'], sort=False, dropna=False,
        ):
            meta_rows = group[group['role'] == _CTX_META_ROLE]
            if meta_rows.empty:
                continue
            meta = meta_rows.iloc[0]
            direction = meta['direction'] if meta['direction'] in VALID_DIRECTIONS else DEFAULT_DIRECTION
            try:
                heritability = float(meta['heritability'])
                if not np.isfinite(heritability):
                    heritability = None
            except (TypeError, ValueError):
                heritability = None

            def _values_for(role: str) -> Dict[str, float]:
                role_rows = group[group['role'] == role]
                return {str(r['id']): float(r['value']) for _, r in role_rows.iterrows()}

            # Classic heritability (its own one-row role - see
            # `_CTX_H2_CLASSIC_ROLE`); absent row (older sidecar, or a
            # scenario whose classic value could not be computed) -> None.
            _h2c_rows = group[group['role'] == _CTX_H2_CLASSIC_ROLE]
            try:
                heritability_classic = float(_h2c_rows.iloc[0]['value']) if not _h2c_rows.empty else None
                if heritability_classic is not None and not np.isfinite(heritability_classic):
                    heritability_classic = None
            except (TypeError, ValueError):
                heritability_classic = None

            index[scenario_key(*key_values)] = {
                'enabled': True,
                'direction': direction,
                'heritability': heritability,
                'heritability_classic': heritability_classic,
                'accuracy_numerator_mode': meta['accuracy_numerator_mode'],
                'blue_values': _values_for(_CTX_BLUE_ROLE),
                'reference_values': _values_for(_CTX_REFERENCE_ROLE),
                # ver4-25, R8: diagonal-mode per-test-individual scoring
                # weight, `{}` whenever blue_weighting was off for this
                # scenario (no rows of this role were ever written) - see
                # genomic_prediction.py's own naive-ensemble finalisation
                # call site for how this is consumed.
                'test_weights': _values_for(_CTX_TEST_WEIGHT_ROLE),
            }

    # ver4-25, R8 Stage 6 completion: a SEPARATE per-scenario index over
    # the matrix frame (never merged into `index` above - a 'full'-mode
    # run writes NO rows at all to `frame`'s own 'test_weight' role, so a
    # scenario can appear in one index, the other, both, or neither,
    # depending on which mode - if any - this run's own ASREML
    # ['blue_weighting'] used).
    _matrix_index: Dict[tuple, "pd.DataFrame"] = {}
    if test_weight_matrix_frame is not None and test_weight_matrix_frame.shape[0] != 0:
        _mframe = test_weight_matrix_frame
        if 'ratio' in _mframe.columns:
            _mframe = _mframe.copy()
            _mframe['ratio'] = _mframe['ratio'].apply(lambda x: tuple(x) if isinstance(x, list) else x)
        for key_values, group in _mframe.groupby(
            ['population', 'phenotype', 'ratio', 'sample'], sort=False, dropna=False,
        ):
            _matrix_index[scenario_key(*key_values)] = group

    def _resolve(population, phenotype, ratio, sample,
                 *, ids: "Optional[Sequence[str]]" = None) -> "Optional[dict]":
        key = scenario_key(population, phenotype, ratio, sample)
        ctx = index.get(key)
        _sub = _matrix_index.get(key) if ids else None
        if ctx is None and _sub is None:
            return None
        # `enabled: False` here (NOT True) is deliberate: this branch only
        # ever fires when `index` (the accuracy-metrics sidecar's own
        # per-scenario data) has NOTHING for this scenario at all - e.g.
        # ASReml accuracy metrics were never enabled for this run, only
        # 'full'-mode weighting was. `compute_accuracy_metrics_row()`'s
        # own `if not ctx.get('enabled'): return {}` guard is what keeps
        # THAT run from gaining Heritability/Spearman/Top-k/Adjusted-
        # accuracy columns it never asked for, just because a weight
        # matrix happened to resolve - only 'test_weight_matrix' (read
        # directly by models/ensemble.py::_metric(), which never
        # consults 'enabled' at all) is meaningful in this fallback ctx.
        ctx = dict(ctx) if ctx is not None else {
            'enabled': False, 'direction': DEFAULT_DIRECTION, 'heritability': None,
            'heritability_classic': None,
            'accuracy_numerator_mode': None, 'blue_values': {}, 'reference_values': {},
            'test_weights': {},
        }
        ctx['test_weight_matrix'] = None
        if _sub is not None:
            W = blue_weights.weight_matrix_from_frame(_sub, key, ids)
            if W is not None:
                ctx['test_weight_matrix'] = (W, list(ids))
        return ctx

    return _resolve
