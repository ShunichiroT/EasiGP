"""candidate_reliability.py - ver4-98 (catalogue O04 out-of-domain flag + O03 per-candidate
prediction intervals).

For every test line of a finished run, two questions answered from the run's own files and the
genotype file, with no refit and no model import:

* O04, is this line inside the domain the training lines cover? ``k_max`` is the line's largest
  genomic relationship to the task's training + validation lines (``models/relatedness.py``,
  markers standardised on those reference lines only; ~1 = a near-duplicate, ~0 = unrelated).
  ``k_percentile`` is the share of the TRAINING lines whose own leave-one-out ``k`` is at or below
  the candidate's, so a low value means "less related to the training set than most training
  lines are to each other". ``out_of_domain`` is True when ``k_max`` is below the
  ``--ood-quantile`` (default 0.05) quantile of those leave-one-out values. No family label is
  needed and no phenotype is read.
* O03, how wide is a plausible interval for this line's prediction? Split-conformal intervals
  built from the held-out residuals of the SAME task's validation lines, per model column:
  ``prediction +- q``, where ``q`` is the finite-sample ``ceil((n+1)(1-alpha))/n`` quantile of the
  absolute validation residuals. The per-candidate part comes from relatedness: validation
  residuals are split into ``--k-bins`` equal-count strata by their ``k`` to the training lines,
  each test line takes the width of the stratum its own ``k`` falls in, and a stratum with fewer
  than ``--min-cal`` validation lines uses the pooled width instead. With ``--k-bins 1`` the
  width is the same for every line.

What the intervals do and do not promise. The coverage guarantee holds when the validation lines
and the test lines are exchangeable (random splits inside one population). Under across-family
or across-population deployment it does not, which is why the tool also reports the realised
coverage on the test lines (their actual values are in every EasiGP result file) per stratum and
per ``out_of_domain`` value. Read that table before trusting an interval, and note that member
disagreement is not used anywhere: it does not predict error (ver4-70).

Inputs: ``Prediction_result_test`` (required), ``_train`` and ``_valid`` (``.csv`` / ``.csv.gz`` /
per-batch files) from the run, plus ``--genotypes``: a CSV with an ``ID`` column, an optional
``population`` column and then numeric marker columns (0 / 1 / 2). PLINK input is not read here;
export a CSV first. Tasks without a validation split get the domain flag but no interval. A
missing genotype value is replaced by the column mean of the whole file.

Lines outside the calibration domain (ver4-101). The validation lines are a random sample of the
training population, so they are all in-domain and say nothing about the error of a line that is
far less related to the training lines: the intervals of lines flagged ``out_of_domain`` under-cover
(ver4-98 measured 0.81 against 0.96 for the in-domain lines in a two-family fixture). No refit of
the width from the validation lines can repair that (their ``k`` does not reach the flagged lines,
so a fitted error-versus-``k`` curve would only extrapolate), so the tool does not pretend to. It
instead (a) writes ``Candidate_inflation.csv`` - per model, the realised coverage of the in-domain
and the flagged lines and ``needed_factor``, the factor by which the flagged lines' half-widths
would have to be multiplied to reach the nominal coverage on THIS run's test lines - and (b) lets
you apply a factor with ``--ood-inflate F`` (default 1 = off, identical to ver4-98). Measure the
factor on a benchmark run whose test lines have actuals and apply it to a deployment run; a factor
applied to the run it was measured on is in-sample and says nothing about new data.

Output (in ``--out-dir``, default the result folder):

* ``Candidate_domain.csv``: task columns, ``id``, ``k_max``, ``k_percentile``, ``out_of_domain``.
* ``Candidate_interval.csv``: task columns, ``id``, ``model``, ``prediction``, ``lower``,
  ``upper``, ``stratum``, ``k_train``, ``actual``, ``covered``, and (ver4-101) ``out_of_domain``
  and ``inflation`` (the factor applied to the half-width, 1.0 unless ``--ood-inflate`` is set).
* ``Candidate_reliability_summary.csv``: per (population, phenotype, ratio, model) x
  ``stratum`` and x ``out_of_domain``: ``n``, ``coverage``, ``mean_width``,
  ``mean_abs_error``.
* ``Candidate_inflation.csv`` (ver4-101): per (population, phenotype, ratio, model), ``n_in`` /
  ``n_out`` test lines, ``coverage_in`` / ``coverage_out`` and ``needed_factor`` (see above;
  empty when fewer than ``--min-out`` flagged lines have an actual).
* ``Candidate_abstention.csv`` (opt-in, ``--abstain``; selective prediction, catalogue O04): per
  (population, phenotype, ratio, model) and ``rule``, what happens to the accuracy of the
  predictions that are KEPT when the least trustworthy lines are withheld. ``rule`` is
  ``k_train`` (lines ranked by relatedness to the training lines, most related kept first),
  ``interval_width`` (narrowest interval kept first) or ``out_of_domain`` (every flagged line
  withheld). Columns: ``fraction`` (share kept), ``n_kept``, ``n_total``, ``pearson``, ``mse``,
  ``mean_abs_error`` of the kept lines, and ``d_pearson`` / ``d_mse`` against keeping every line
  (``d_mse`` below 0 is an improvement). It only reads the table ``Candidate_interval.csv``
  holds (so only tasks with a validation split appear) and uses the test lines' own actual
  values: it measures how much abstaining would have helped on this run, it does not choose
  which line to withhold on new data without actuals - for that, apply the same rule
  (``out_of_domain`` or a ``k_train`` cut-off) to the new lines' domain table. Pearson and MSE
  pool the kept lines of all samples of a task group, as ``Candidate_reliability_summary.csv``
  does. Without ``--abstain`` nothing changes and no file is written.

Example::

    python tools/candidate_reliability.py --result Result/run1 --genotypes geno.csv \\
        --alpha 0.1 --models "Super learner,ensemble"

Exit codes: 0 = written, 1 = error, 3 = nothing to evaluate.
"""
import argparse
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from iterative_gate import read_result_table, resolve_result_dir  # noqa: E402
from prediction_diagnostics import TASK_COLUMNS, prediction_columns  # noqa: E402
from models.relatedness import max_kinship, standardise_on  # noqa: E402

EXIT_OK, EXIT_ERROR, EXIT_NOTHING = 0, 1, 3
DEFAULT_ALPHA = 0.1
DEFAULT_OOD_QUANTILE = 0.05
DEFAULT_K_BINS = 2
DEFAULT_MIN_CAL = 20
DEFAULT_MIN_OUT = 10
DEFAULT_ABSTAIN_FRACTIONS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
DEFAULT_ABSTAIN_MIN_KEPT = 10
MIN_REFERENCE_LINES = 10
DEFAULT_MODELS = ('Super learner', 'ensemble')


# ---------------------------------------------------------------------------------------------
# pure computation
# ---------------------------------------------------------------------------------------------
def domain_flags(G_ref: np.ndarray, G_query: np.ndarray,
                 quantile: float = DEFAULT_OOD_QUANTILE) -> Dict[str, np.ndarray]:
    """Relatedness of query lines to the reference lines.

    `G_ref` (n_ref x M) and `G_query` (n_q x M) are raw genotypes over the same markers.
    Markers are standardised on the reference rows only. Returns a dict with ``k_max`` (n_q),
    ``k_percentile`` (n_q, the share of reference lines whose leave-one-out k is <= the query's),
    ``out_of_domain`` (bool, k_max below the `quantile` of the leave-one-out values) and
    ``threshold`` (a 1-element array). Raises ValueError when fewer than MIN_REFERENCE_LINES
    reference lines exist or `quantile` is outside (0, 1)."""
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    G_ref = np.asarray(G_ref, dtype=float)
    if G_ref.shape[0] < MIN_REFERENCE_LINES:
        raise ValueError(f"only {G_ref.shape[0]} reference line(s); need at least "
                         f"{MIN_REFERENCE_LINES}")
    Z_ref, Z_q = standardise_on(G_ref, np.asarray(G_query, dtype=float))
    k_q = max_kinship(Z_q, Z_ref)
    k_loo = max_kinship(Z_ref, Z_ref, exclude_self=True)
    threshold = float(np.quantile(k_loo, quantile))
    sorted_loo = np.sort(k_loo)
    percentile = np.searchsorted(sorted_loo, k_q, side='right') / float(sorted_loo.size)
    return {'k_max': k_q, 'k_percentile': percentile, 'out_of_domain': k_q < threshold,
            'threshold': np.array([threshold])}


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The finite-sample split-conformal quantile ceil((n+1)(1-alpha))/n of `scores`; +inf when n
    is too small for that level."""
    n = scores.size
    if n == 0:
        return float('inf')
    level = math.ceil((n + 1) * (1.0 - alpha)) / n
    if level > 1.0:
        return float('inf')
    return float(np.quantile(scores, level, method='higher'))


def conformal_intervals(valid_pred: np.ndarray, valid_actual: np.ndarray, valid_k: np.ndarray,
                        test_pred: np.ndarray, test_k: np.ndarray,
                        alpha: float = DEFAULT_ALPHA, n_bins: int = DEFAULT_K_BINS,
                        min_cal: int = DEFAULT_MIN_CAL
                        ) -> Dict[str, np.ndarray]:
    """Relatedness-stratified split-conformal intervals for one model in one task.

    Validation lines with a finite prediction, actual and k define the calibration scores
    ``|actual - pred|``. Their k values give `n_bins` equal-count strata (edges at the validation
    k quantiles, outer strata open). A test line takes the quantile of its stratum when the
    stratum has at least `min_cal` calibration lines and a finite quantile, otherwise the pooled
    quantile (``stratum`` = -1). Returns ``lower``, ``upper``, ``stratum`` (int, 0 = least
    related) and ``half_width`` for every test line; lines with a non-finite prediction or k are
    NaN / -2. Raises ValueError when `alpha` is outside (0, 1) or no calibration line is usable."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    valid_pred = np.asarray(valid_pred, dtype=float)
    valid_actual = np.asarray(valid_actual, dtype=float)
    valid_k = np.asarray(valid_k, dtype=float)
    test_pred = np.asarray(test_pred, dtype=float)
    test_k = np.asarray(test_k, dtype=float)
    ok = np.isfinite(valid_pred) & np.isfinite(valid_actual) & np.isfinite(valid_k)
    if not ok.any():
        raise ValueError("no usable calibration line")
    scores = np.abs(valid_actual[ok] - valid_pred[ok])
    k_cal = valid_k[ok]
    n_bins = max(1, int(n_bins))
    edges = (np.quantile(k_cal, np.linspace(0, 1, n_bins + 1)[1:-1]) if n_bins > 1
             else np.array([]))
    cal_bin = np.searchsorted(edges, k_cal, side='right')
    pooled = _conformal_quantile(scores, alpha)
    width_of_bin = []
    for b in range(n_bins):
        s = scores[cal_bin == b]
        q = _conformal_quantile(s, alpha) if s.size >= min_cal else float('inf')
        width_of_bin.append(q)
    test_bin = np.where(np.isfinite(test_k), np.searchsorted(edges, test_k, side='right'), -2)
    half = np.full(test_pred.shape, np.nan)
    stratum = np.full(test_pred.shape, -2, dtype=int)
    for i in range(test_pred.size):
        if not (np.isfinite(test_pred[i]) and np.isfinite(test_k[i])):
            continue
        b = int(test_bin[i])
        q = width_of_bin[b]
        if np.isfinite(q) and n_bins > 1:
            stratum[i] = b
        else:
            q = pooled
            stratum[i] = -1 if n_bins > 1 else 0
        half[i] = q
    return {'lower': test_pred - half, 'upper': test_pred + half, 'stratum': stratum,
            'half_width': half}


# ---------------------------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------------------------
def load_genotypes(path: str) -> pd.DataFrame:
    """Genotype CSV -> float DataFrame indexed by ID (str); the marker columns only.

    The first column named ``ID`` (any case) is the index, a ``population`` column is dropped,
    every other column must be numeric. Missing values become the column mean. Raises ValueError
    on a missing ID column, duplicated IDs, or a non-numeric marker column."""
    table = pd.read_csv(path)
    id_col = next((c for c in table.columns if str(c).strip().lower() == 'id'), None)
    if id_col is None:
        raise ValueError(f"{path}: no 'ID' column")
    ids = table[id_col].astype(str).str.strip()
    if ids.duplicated().any():
        raise ValueError(f"{path}: duplicated IDs, e.g. {ids[ids.duplicated()].iloc[0]!r}")
    drop = [id_col] + [c for c in table.columns if str(c).strip().lower() == 'population']
    markers = table.drop(columns=drop)
    bad = [c for c in markers.columns if not pd.api.types.is_numeric_dtype(markers[c])]
    if bad:
        raise ValueError(f"{path}: non-numeric marker column(s), e.g. {bad[:3]}")
    markers = markers.astype(float)
    markers = markers.fillna(markers.mean())
    markers.index = ids.to_numpy()
    return markers


def _task_groups(frame: Optional[pd.DataFrame], cols: List[str]) -> Dict[Tuple[str, ...], pd.DataFrame]:
    """{task key (all values as str): rows} of a Prediction_result table."""
    if frame is None or frame.shape[0] == 0:
        return {}
    out = {}
    for key, part in frame.groupby(cols, sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        out[tuple(str(v) for v in key)] = part
    return out


def select_models(columns: Sequence[str], wanted: Optional[Sequence[str]]) -> List[str]:
    """The prediction columns to evaluate: those equal to one of `wanted`, or starting with
    ``<wanted>__`` (tuning / weighting suffixes), in table order. `wanted` None means the
    default (the naive ensemble and the Super learner rows); when that matches nothing, every
    column is used."""
    chosen = list(wanted) if wanted else list(DEFAULT_MODELS)
    hit = [c for c in columns if any(c == w or c.startswith(w + '__') for w in chosen)]
    if hit or wanted:
        return hit
    return list(columns)


def reliability(test: pd.DataFrame, train: Optional[pd.DataFrame], valid: Optional[pd.DataFrame],
                genotypes: pd.DataFrame, models: Optional[Sequence[str]] = None,
                alpha: float = DEFAULT_ALPHA, ood_quantile: float = DEFAULT_OOD_QUANTILE,
                n_bins: int = DEFAULT_K_BINS, min_cal: int = DEFAULT_MIN_CAL,
                notes: Optional[List[str]] = None, ood_inflate: float = 1.0
                ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Domain flags and intervals for every task of `test`. Returns ``(domain, interval)``.

    Task columns are the ones of ``TASK_COLUMNS`` present in `test`; `train` / `valid` are
    matched to a task by the same columns (as strings). A task is skipped with a NOTE (appended
    to `notes`, else printed) when it has fewer than MIN_REFERENCE_LINES reference lines with a
    genotype, or too few usable validation lines for an interval. `ood_inflate` (ver4-101, default
    1.0 = off) multiplies the half-width of every line flagged out of domain; it must be >= 1."""
    if not float(ood_inflate) >= 1.0:
        raise ValueError(f"ood_inflate must be at least 1, got {ood_inflate}")
    notes = notes if notes is not None else []
    task_cols = [c for c in TASK_COLUMNS if c in test.columns]
    pred_cols = select_models(prediction_columns(test), models)
    train_by = _task_groups(train, [c for c in task_cols if train is not None and c in train.columns])
    valid_by = _task_groups(valid, [c for c in task_cols if valid is not None and c in valid.columns])
    train_cols = [c for c in task_cols if train is not None and c in train.columns]
    valid_cols = [c for c in task_cols if valid is not None and c in valid.columns]
    marker_index = genotypes.index
    dom_rows, int_rows = [], []

    def geno(ids: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        ids = pd.Index([str(i).strip() for i in ids])
        have = ids.isin(marker_index)
        return genotypes.loc[ids[have]].to_numpy(dtype=float), have

    for key, part in test.groupby(task_cols, sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        skey = tuple(str(v) for v in key)
        label = dict(zip(task_cols, key))
        tr = train_by.get(tuple(skey[task_cols.index(c)] for c in train_cols)) if train_cols else None
        va = valid_by.get(tuple(skey[task_cols.index(c)] for c in valid_cols)) if valid_cols else None
        tr_ids = tr['id'].astype(str).str.strip().tolist() if tr is not None else []
        va_ids = va['id'].astype(str).str.strip().tolist() if va is not None else []
        te_ids = part['id'].astype(str).str.strip().tolist()
        G_te, have_te = geno(te_ids)
        if G_te.shape[0] == 0:
            notes.append(f"NOTE: no test line of task {label} has a genotype - skipped.")
            continue
        G_ref, _ = geno(tr_ids + va_ids)
        try:
            dom = domain_flags(G_ref, G_te, ood_quantile)
        except ValueError as exc:
            notes.append(f"NOTE: task {label} skipped: {exc}")
            continue
        te_kept = part.iloc[np.flatnonzero(have_te)]
        for j, i in enumerate(te_kept['id'].astype(str).str.strip()):
            dom_rows.append({**label, 'id': i, 'k_max': float(dom['k_max'][j]),
                             'k_percentile': float(dom['k_percentile'][j]),
                             'out_of_domain': bool(dom['out_of_domain'][j])})
        if va is None or len(va_ids) == 0 or len(tr_ids) == 0:
            notes.append(f"NOTE: task {label} has no validation split in the result files - "
                         f"domain flag only, no interval.")
            continue
        G_tr, _ = geno(tr_ids)
        va_part = va.copy()
        G_va, have_va = geno(va_ids)
        va_part = va_part.iloc[np.flatnonzero(have_va)]
        if G_tr.shape[0] < MIN_REFERENCE_LINES or G_va.shape[0] < 2:
            notes.append(f"NOTE: task {label}: too few training or validation lines with a "
                         f"genotype for an interval.")
            continue
        Z_tr, Z_va, Z_te = standardise_on(G_tr, G_va, G_te)
        k_va = max_kinship(Z_va, Z_tr)
        k_te = max_kinship(Z_te, Z_tr)
        actual_te = te_kept['actual'].to_numpy(dtype=float)
        for col in pred_cols:
            if col not in va_part.columns:
                continue
            try:
                res = conformal_intervals(va_part[col].to_numpy(dtype=float),
                                          va_part['actual'].to_numpy(dtype=float), k_va,
                                          te_kept[col].to_numpy(dtype=float), k_te, alpha,
                                          n_bins, min_cal)
            except ValueError as exc:
                notes.append(f"NOTE: task {label}, model {col!r}: {exc}")
                continue
            pred_te = te_kept[col].to_numpy(dtype=float)
            ids_te = te_kept['id'].astype(str).str.strip().to_numpy()
            for j in range(pred_te.size):
                if not np.isfinite(res['half_width'][j]):
                    continue
                flagged = bool(dom['out_of_domain'][j])
                factor = float(ood_inflate) if flagged else 1.0
                half = res['half_width'][j] * factor
                lower, upper = pred_te[j] - half, pred_te[j] + half
                covered = bool(lower <= actual_te[j] <= upper) \
                    if np.isfinite(actual_te[j]) else None
                int_rows.append({**label, 'id': ids_te[j], 'model': col,
                                 'prediction': float(pred_te[j]),
                                 'lower': float(lower), 'upper': float(upper),
                                 'stratum': int(res['stratum'][j]), 'k_train': float(k_te[j]),
                                 'actual': float(actual_te[j]), 'covered': covered,
                                 'out_of_domain': flagged, 'inflation': factor})
    domain = pd.DataFrame(dom_rows, columns=task_cols + ['id', 'k_max', 'k_percentile',
                                                           'out_of_domain'])
    interval = pd.DataFrame(int_rows, columns=task_cols + ['id', 'model', 'prediction', 'lower',
                                                             'upper', 'stratum', 'k_train',
                                                             'actual', 'covered', 'out_of_domain',
                                                             'inflation'])
    return domain, interval


def summarise(domain: pd.DataFrame, interval: pd.DataFrame) -> pd.DataFrame:
    """Realised coverage per (population, phenotype, ratio, model) x ``stratum`` and x
    ``out_of_domain`` over all samples. ``by`` says which split a row is."""
    if interval.empty:
        return pd.DataFrame(columns=['by', 'value', 'n', 'coverage', 'mean_width',
                                     'mean_abs_error'])
    work = interval.copy()
    if 'out_of_domain' in work.columns:          # ver4-101: the flag is already in the interval table
        pass
    elif not domain.empty:
        task_cols = [c for c in TASK_COLUMNS if c in domain.columns]
        work = work.merge(domain[task_cols + ['id', 'out_of_domain']], on=task_cols + ['id'],
                          how='left')
    else:
        work['out_of_domain'] = np.nan
    work['width'] = work['upper'] - work['lower']
    work['abs_error'] = (work['actual'] - work['prediction']).abs()
    work['covered_f'] = work['covered'].astype(float)
    keys = [c for c in ('asreml_option', 'population', 'phenotype', 'ratio') if c in work.columns]
    rows = []
    for by in ('stratum', 'out_of_domain'):
        for key, part in work.groupby(keys + ['model', by], sort=True, dropna=False):
            rows.append({**dict(zip(keys + ['model'], key[:-1])), 'by': by, 'value': key[-1],
                         'n': int(part.shape[0]), 'coverage': float(part['covered_f'].mean()),
                         'mean_width': float(part['width'].mean()),
                         'mean_abs_error': float(part['abs_error'].mean())})
    return pd.DataFrame(rows)


def inflation_table(interval: pd.DataFrame, alpha: float = DEFAULT_ALPHA,
                    min_out: int = DEFAULT_MIN_OUT) -> pd.DataFrame:
    """Per (population, phenotype, ratio, model): the realised coverage of the in-domain and the
    out-of-domain test lines and ``needed_factor``, the factor that would bring the flagged lines
    to the nominal ``1 - alpha`` coverage on these test lines.

    The factor is the finite-sample conformal quantile of ``|actual - prediction| / half_width``
    over the flagged lines (with the applied ``inflation`` divided out), floored at 1; it is NaN
    when fewer than `min_out` flagged lines have an actual or the sample is too small for the
    level. It is measured on the run's own test lines, so it is in-sample for this run."""
    columns = ['population', 'phenotype', 'ratio', 'model', 'n_in', 'n_out', 'coverage_in',
               'coverage_out', 'needed_factor']
    if interval.empty or 'out_of_domain' not in interval.columns:
        return pd.DataFrame(columns=columns)
    work = interval[np.isfinite(interval['actual'].to_numpy(dtype=float))].copy()
    base_half = (work['upper'] - work['lower']) / 2.0 / work['inflation']
    work['score'] = (work['actual'] - work['prediction']).abs() / base_half.where(base_half > 0)
    work['covered_f'] = work['covered'].astype(float)
    keys = [c for c in ('asreml_option', 'population', 'phenotype', 'ratio') if c in work.columns]
    rows = []
    for key, part in work.groupby(keys + ['model'], sort=True, dropna=False):
        flagged = part[part['out_of_domain'].astype(bool)]
        near = part[~part['out_of_domain'].astype(bool)]
        scores = flagged['score'].dropna().to_numpy(dtype=float)
        needed = float('nan')
        if scores.size >= min_out:
            q = _conformal_quantile(scores, alpha)
            if math.isfinite(q):
                needed = max(1.0, q)
        rows.append({**dict(zip(keys + ['model'], key)), 'n_in': int(near.shape[0]),
                     'n_out': int(flagged.shape[0]),
                     'coverage_in': float(near['covered_f'].mean()) if len(near) else float('nan'),
                     'coverage_out': float(flagged['covered_f'].mean()) if len(flagged)
                     else float('nan'), 'needed_factor': needed})
    return pd.DataFrame(rows)


def _kept_metrics(prediction: np.ndarray, actual: np.ndarray) -> Tuple[float, float, float]:
    """(Pearson, MSE, mean absolute error) of the lines kept; Pearson is NaN for fewer than 3
    lines or a constant column."""
    error = actual - prediction
    pearson = float('nan')
    if prediction.size >= 3 and prediction.std() > 0.0 and actual.std() > 0.0:
        pearson = float(np.corrcoef(prediction, actual)[0, 1])
    return pearson, float(np.mean(error ** 2)), float(np.mean(np.abs(error)))


def abstention_table(interval: pd.DataFrame,
                     fractions: Sequence[float] = DEFAULT_ABSTAIN_FRACTIONS,
                     min_kept: int = DEFAULT_ABSTAIN_MIN_KEPT) -> pd.DataFrame:
    """Selective prediction (catalogue O04): the accuracy of the kept lines when the least
    trustworthy ones are withheld.

    For every (option, population, phenotype, ratio, model) group of `interval` (lines with a
    finite prediction and actual), the rules ``k_train`` (most related to the training lines kept
    first) and ``interval_width`` (narrowest interval kept first) keep the top ``fraction`` of the
    lines for every fraction in `fractions` (each in (0, 1]; ties broken by table order), and the
    rule ``out_of_domain`` keeps every line not flagged (one row, only when some line is flagged).
    A row is written only when at least `min_kept` lines are kept. ``d_pearson`` and ``d_mse``
    are the difference to keeping every line of the group. Raises ValueError for a fraction
    outside (0, 1] or an empty `fractions`."""
    columns_out = ['population', 'phenotype', 'ratio', 'model', 'rule', 'fraction', 'n_kept',
                   'n_total', 'pearson', 'mse', 'mean_abs_error', 'd_pearson', 'd_mse']
    fractions = [float(f) for f in fractions]
    if not fractions or any(not 0.0 < f <= 1.0 for f in fractions):
        raise ValueError(f"fractions must be a non-empty list of values in (0, 1], got {fractions}")
    if interval is None or interval.empty:
        return pd.DataFrame(columns=columns_out)
    pred_all = interval['prediction'].to_numpy(dtype=float)
    act_all = interval['actual'].to_numpy(dtype=float)
    work = interval[np.isfinite(pred_all) & np.isfinite(act_all)].copy()
    work['width'] = work['upper'] - work['lower']
    keys = [c for c in ('asreml_option', 'population', 'phenotype', 'ratio') if c in work.columns]
    rows = []
    for key, part in work.groupby(keys + ['model'], sort=True, dropna=False):
        label = dict(zip(keys + ['model'], key))
        pred = part['prediction'].to_numpy(dtype=float)
        act = part['actual'].to_numpy(dtype=float)
        n_total = int(part.shape[0])
        if n_total < min_kept:
            continue
        base_r, base_mse, _ = _kept_metrics(pred, act)

        def emit(rule: str, keep: np.ndarray) -> None:
            if keep.size < min_kept:
                return
            r, mse, mae = _kept_metrics(pred[keep], act[keep])
            rows.append({**label, 'rule': rule, 'fraction': keep.size / n_total,
                         'n_kept': int(keep.size), 'n_total': n_total, 'pearson': r, 'mse': mse,
                         'mean_abs_error': mae, 'd_pearson': r - base_r, 'd_mse': mse - base_mse})

        order_k = np.argsort(-part['k_train'].to_numpy(dtype=float), kind='stable')
        order_w = np.argsort(part['width'].to_numpy(dtype=float), kind='stable')
        for fraction in sorted(set(fractions), reverse=True):
            n_keep = max(1, int(math.ceil(fraction * n_total - 1e-9)))
            emit('k_train', order_k[:n_keep])
            emit('interval_width', order_w[:n_keep])
        if 'out_of_domain' in part.columns:
            flagged = part['out_of_domain'].astype(bool).to_numpy()
            if flagged.any():
                emit('out_of_domain', np.flatnonzero(~flagged))
    return pd.DataFrame(rows, columns=keys + ['model', 'rule', 'fraction', 'n_kept', 'n_total',
                                              'pearson', 'mse', 'mean_abs_error', 'd_pearson',
                                              'd_mse'])


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Out-of-domain flag (O04) and per-candidate intervals (O03) for a run.")
    parser.add_argument('--result', required=True, help="Result folder (or its RESULT_NAME)")
    parser.add_argument('--genotypes', required=True,
                        help="genotype CSV: ID, optional population, then numeric markers")
    parser.add_argument('--out-dir', default=None, help="default: the result folder")
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA,
                        help=f"miscoverage level; 0.1 = 90%% intervals (default {DEFAULT_ALPHA})")
    parser.add_argument('--ood-quantile', type=float, default=DEFAULT_OOD_QUANTILE,
                        help=f"flag lines whose k_max is below this quantile of the training "
                             f"lines' leave-one-out k (default {DEFAULT_OOD_QUANTILE})")
    parser.add_argument('--models', default=None,
                        help="comma-separated prediction labels (a label also matches "
                             "'<label>__...'); default 'Super learner,ensemble'")
    parser.add_argument('--k-bins', type=int, default=DEFAULT_K_BINS,
                        help=f"relatedness strata for the interval width (default "
                             f"{DEFAULT_K_BINS}; 1 = one width for every line)")
    parser.add_argument('--min-cal', type=int, default=DEFAULT_MIN_CAL,
                        help=f"validation lines a stratum needs (default {DEFAULT_MIN_CAL})")
    parser.add_argument('--ood-inflate', type=float, default=1.0,
                        help="multiply the half-width of lines flagged out of domain by this "
                             "factor (>= 1; default 1 = off). Measure it first: see "
                             "Candidate_inflation.csv")
    parser.add_argument('--min-out', type=int, default=DEFAULT_MIN_OUT,
                        help=f"flagged test lines with an actual that a needed_factor requires "
                             f"(default {DEFAULT_MIN_OUT})")
    parser.add_argument('--abstain', action='store_true',
                        help="also write Candidate_abstention.csv: the accuracy of the kept "
                             "lines when the least related / widest-interval / flagged lines "
                             "are withheld (selective prediction). Default off.")
    parser.add_argument('--abstain-fractions', default=','.join(
        f"{f:g}" for f in DEFAULT_ABSTAIN_FRACTIONS),
        help="comma-separated shares of lines to keep for --abstain, each in (0, 1] "
             "(default %(default)s)")
    args = parser.parse_args(argv)
    try:
        abstain_fractions = [float(v) for v in args.abstain_fractions.split(',') if v.strip()]
        if args.abstain and (not abstain_fractions
                             or any(not 0.0 < f <= 1.0 for f in abstain_fractions)):
            raise ValueError
    except ValueError:
        print(f"[candidate_reliability] ERROR: --abstain-fractions must be comma-separated "
              f"numbers in (0, 1], got {args.abstain_fractions!r}.")
        return EXIT_ERROR
    if not args.ood_inflate >= 1.0:
        print(f"[candidate_reliability] ERROR: --ood-inflate must be at least 1, got "
              f"{args.ood_inflate}.")
        return EXIT_ERROR

    result_dir = resolve_result_dir(args.result)
    if not os.path.isdir(result_dir):
        print(f"[candidate_reliability] ERROR: {args.result!r} is not a result folder.")
        return EXIT_ERROR
    try:
        test = read_result_table(result_dir, 'Prediction_result_test')
        train = read_result_table(result_dir, 'Prediction_result_train')
        valid = read_result_table(result_dir, 'Prediction_result_valid')
        genotypes = load_genotypes(args.genotypes)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"[candidate_reliability] ERROR: {exc}")
        return EXIT_ERROR
    if test is None or test.shape[0] == 0 or 'id' not in test.columns:
        print("[candidate_reliability] Prediction_result_test not found, empty, or without an "
              "'id' column - nothing to evaluate.")
        return EXIT_NOTHING
    if train is None and valid is None:
        print("[candidate_reliability] neither Prediction_result_train nor _valid found - "
              "there are no reference lines.")
        return EXIT_NOTHING
    wanted = [m.strip() for m in args.models.split(',') if m.strip()] if args.models else None
    notes: List[str] = []
    try:
        domain, interval = reliability(test, train, valid, genotypes, wanted, args.alpha,
                                       args.ood_quantile, args.k_bins, args.min_cal, notes,
                                       args.ood_inflate)
    except ValueError as exc:
        print(f"[candidate_reliability] ERROR: {exc}")
        return EXIT_ERROR
    for message in dict.fromkeys(notes):
        print(f"[candidate_reliability] {message}")
    if domain.empty:
        print("[candidate_reliability] no task could be evaluated.")
        return EXIT_NOTHING
    summary = summarise(domain, interval)
    out_dir = args.out_dir or result_dir
    os.makedirs(out_dir, exist_ok=True)
    domain.to_csv(os.path.join(out_dir, 'Candidate_domain.csv'), index=False)
    interval.to_csv(os.path.join(out_dir, 'Candidate_interval.csv'), index=False)
    summary.to_csv(os.path.join(out_dir, 'Candidate_reliability_summary.csv'), index=False)
    inflation = inflation_table(interval, args.alpha, args.min_out)
    inflation.to_csv(os.path.join(out_dir, 'Candidate_inflation.csv'), index=False)
    if args.abstain:
        abstention = abstention_table(interval, abstain_fractions)
        abstention.to_csv(os.path.join(out_dir, 'Candidate_abstention.csv'), index=False)
        if abstention.empty:
            print("[candidate_reliability] NOTE: --abstain wrote an empty Candidate_abstention.csv "
                  "(no group has an interval table with enough lines with an actual).")
        else:
            half = abstention[(abstention['rule'] == 'k_train')
                              & (abstention['fraction'].between(0.45, 0.55))]
            if not half.empty:
                print(f"[candidate_reliability] abstaining on the least related half of the lines "
                      f"changes Pearson by {half['d_pearson'].mean():+.3f} and MSE by "
                      f"{half['d_mse'].mean():+.3f} on average over models and tasks "
                      f"(Candidate_abstention.csv).")
    share = float(domain['out_of_domain'].mean())
    print(f"[candidate_reliability] {domain.shape[0]} test lines, {100 * share:.1f}% flagged "
          f"out of domain (expected about {100 * args.ood_quantile:.0f}% for a random split).")
    if not summary.empty:
        with pd.option_context('display.width', 170, 'display.max_columns', 30,
                               'display.float_format', '{:.3f}'.format):
            print(summary.to_string(index=False))
        print(f"[candidate_reliability] nominal coverage is {100 * (1 - args.alpha):.0f}%. "
              f"Compare it with 'coverage' above before using an interval: the guarantee needs "
              f"validation and test lines to be exchangeable.")
    if not inflation.empty and inflation['n_out'].sum() > 0:
        gap = inflation.dropna(subset=['coverage_out'])
        if not gap.empty:
            print(f"[candidate_reliability] flagged lines: realised coverage "
                  f"{gap['coverage_out'].mean():.3f} against {gap['coverage_in'].mean():.3f} for "
                  f"the others (nominal {1 - args.alpha:.2f}); half-width factors applied: "
                  f"{args.ood_inflate:g}. needed_factor per model is in Candidate_inflation.csv "
                  f"(in-sample for this run - measure it on a benchmark, apply it to new data).")
    print(f"[candidate_reliability] wrote Candidate_domain.csv, Candidate_interval.csv, "
          f"Candidate_reliability_summary.csv and Candidate_inflation.csv in {out_dir}.")
    return EXIT_OK


if __name__ == '__main__':
    sys.exit(main())
