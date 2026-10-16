"""pairwise_diversity.py - ver4-102 (catalogue D13, pairwise diversity measures).

The Diversity Prediction Theorem workbook (``diversity_summary.py``) says how much diversity an
ENSEMBLE has. It does not say which PAIRS of members are near-duplicates, which is what a user
choosing a model list needs to know (ver4-69: BayesB 0.94 vs rrBLUP 0 at equal out-of-fold r).
This tool reads a finished run's own ``Prediction_result_test`` and, per task, reports for every
pair of member columns:

* ``pred_corr``: Pearson correlation of the two models' predictions.
* ``err_corr``: Pearson correlation of their errors (prediction - actual). Near 1 = the two make
  the same mistakes, so averaging them gains nothing; this is the pairwise quantity behind the
  ambiguity term of the Diversity Prediction Theorem.
* ``disagreement``, ``double_fault`` and ``q_stat``: the classification-ensemble pairwise
  measures (Kuncheva and Whitaker 2003) carried over to regression by calling a model's
  prediction on a line "bad" when its absolute error is above that model's own ``--bad-quantile``
  quantile (default 0.5, the median) of its absolute errors in the task. ``double_fault`` is the
  share of lines where both are bad (0.25 is independence at the median), ``disagreement`` the
  share where exactly one is bad (0.5 is independence), and ``q_stat`` Yule's Q over the 2 x 2
  good / bad table (+1 = the same lines are hard for both, 0 = independent, -1 = opposite).

Members are the prediction columns whose model is registered in ``model_registry.MODEL_CLASS``
(a tuning suffix such as ``RF__Grid`` is removed first); ensemble and weighted-ensemble columns
(``ensemble``, ``Super learner``, ``Nelder-Mead`` ...) are not members. ``--models`` replaces that
rule with an explicit comma-separated list of column names. A pair needs at least ``--min-lines``
test lines where both predictions and the actual value are finite, otherwise it gets no row.

Read-only; nothing is refitted, no model, R or GUI module is imported. Why a tool and not a sheet
of ``Diversity_prediction_theorem.xlsx`` (the catalogue's wording): the workbook is a ver4-9
module with a 24-block pivot machinery, and a diagnostic of a different object (pairs, not
ensembles) does not justify touching it. The tool writes ordinary CSV files instead.

Output (in ``--out-dir``, default the result folder):

* ``Pairwise_diversity.csv``: the task columns present, ``model_a``, ``model_b``, ``n``, and the
  five measures above.
* ``Pairwise_diversity_summary.csv``: the same over the tasks (samples) of each (option,
  population, phenotype, ratio, pair): ``n_tasks`` and the mean of every measure.

Example::

    python tools/pairwise_diversity.py --result Result/run1
    python tools/pairwise_diversity.py --result Result/run1 --models "RF,SVR,rrBLUP,KNN"

Exit codes: 0 = written, 1 = error, 3 = nothing to evaluate (no prediction file, fewer than two
members, no task with enough lines).
"""
import argparse
import itertools
import os
import sys
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from iterative_gate import read_result_table, resolve_result_dir  # noqa: E402
from prediction_diagnostics import TASK_COLUMNS, prediction_columns  # noqa: E402
from model_registry import model_class  # noqa: E402

EXIT_OK, EXIT_ERROR, EXIT_NOTHING = 0, 1, 3
DEFAULT_BAD_QUANTILE = 0.5
DEFAULT_MIN_LINES = 10
MEASURES = ('pred_corr', 'err_corr', 'disagreement', 'double_fault', 'q_stat')


def member_columns(columns: Sequence[str], wanted: Optional[Sequence[str]] = None) -> List[str]:
    """The member prediction columns, in table order. With `wanted`, the columns named there
    (exact match, or ``<name>__...`` for a tuning suffix); otherwise every column whose model
    class in ``model_registry.MODEL_CLASS`` is not ``'other'``."""
    if wanted:
        names = list(wanted)
        return [c for c in columns if any(c == w or c.startswith(w + '__') for w in names)]
    return [c for c in columns if model_class(c) != 'other']


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or a.std() == 0.0 or b.std() == 0.0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def pair_measures(pred_a: np.ndarray, pred_b: np.ndarray, actual: np.ndarray,
                  bad_quantile: float = DEFAULT_BAD_QUANTILE) -> dict:
    """The five pairwise measures of two prediction vectors against one actual vector (all the
    same length, all finite). Raises ValueError when `bad_quantile` is outside (0, 1).

    A line is "bad" for a model when its absolute error is strictly above the `bad_quantile`
    quantile of that model's own absolute errors. ``q_stat`` is NaN when the 2 x 2 table has a
    zero denominator."""
    if not 0.0 < bad_quantile < 1.0:
        raise ValueError(f"bad_quantile must be in (0, 1), got {bad_quantile}")
    err_a, err_b = pred_a - actual, pred_b - actual
    bad_a = np.abs(err_a) > np.quantile(np.abs(err_a), bad_quantile)
    bad_b = np.abs(err_b) > np.quantile(np.abs(err_b), bad_quantile)
    n = float(actual.size)
    n11 = float(np.sum(~bad_a & ~bad_b))    # both good
    n00 = float(np.sum(bad_a & bad_b))      # both bad
    n10 = float(np.sum(~bad_a & bad_b))
    n01 = float(np.sum(bad_a & ~bad_b))
    denom = n11 * n00 + n01 * n10
    return {'pred_corr': _pearson(pred_a, pred_b), 'err_corr': _pearson(err_a, err_b),
            'disagreement': (n01 + n10) / n, 'double_fault': n00 / n,
            'q_stat': (n11 * n00 - n01 * n10) / denom if denom > 0 else float('nan')}


def pairwise_table(test: pd.DataFrame, members: Sequence[str],
                   bad_quantile: float = DEFAULT_BAD_QUANTILE,
                   min_lines: int = DEFAULT_MIN_LINES) -> pd.DataFrame:
    """One row per task and unordered pair of `members` (see the module docstring)."""
    task_cols = [c for c in TASK_COLUMNS if c in test.columns]
    rows = []
    grouped = test.groupby(task_cols, sort=False, dropna=False) if task_cols else [((), test)]
    for key, part in grouped:
        key = key if isinstance(key, tuple) else (key,)
        label = dict(zip(task_cols, key))
        actual = part['actual'].to_numpy(dtype=float)
        values = {m: part[m].to_numpy(dtype=float) for m in members if m in part.columns}
        for a, b in itertools.combinations(list(values), 2):
            ok = np.isfinite(actual) & np.isfinite(values[a]) & np.isfinite(values[b])
            if int(ok.sum()) < min_lines:
                continue
            rows.append({**label, 'model_a': a, 'model_b': b, 'n': int(ok.sum()),
                         **pair_measures(values[a][ok], values[b][ok], actual[ok], bad_quantile)})
    return pd.DataFrame(rows, columns=task_cols + ['model_a', 'model_b', 'n', *MEASURES])


def summarise(table: pd.DataFrame) -> pd.DataFrame:
    """Mean of each measure over the samples of every (option, population, phenotype, ratio,
    pair); ``n_tasks`` counts the tasks that had a row for the pair."""
    keys = [c for c in ('asreml_option', 'population', 'phenotype', 'ratio') if c in table.columns]
    keys += ['model_a', 'model_b']
    out = []
    for key, part in table.groupby(keys, sort=True, dropna=False):
        row = dict(zip(keys, key))
        row['n_tasks'] = int(part.shape[0])
        for m in MEASURES:
            row[m] = float(part[m].mean()) if part[m].notna().any() else float('nan')
        out.append(row)
    return pd.DataFrame(out, columns=keys + ['n_tasks', *MEASURES])


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Pairwise diversity of the members of a run.")
    parser.add_argument('--result', required=True, help="Result folder (or its RESULT_NAME)")
    parser.add_argument('--out-dir', default=None, help="default: the result folder")
    parser.add_argument('--models', default=None,
                        help="comma-separated member columns (default: every registered model)")
    parser.add_argument('--bad-quantile', type=float, default=DEFAULT_BAD_QUANTILE,
                        help=f"a line is 'bad' for a model above this quantile of its own "
                             f"absolute errors (default {DEFAULT_BAD_QUANTILE})")
    parser.add_argument('--min-lines', type=int, default=DEFAULT_MIN_LINES,
                        help=f"lines a pair needs in a task (default {DEFAULT_MIN_LINES})")
    args = parser.parse_args(argv)
    if not 0.0 < args.bad_quantile < 1.0:
        print(f"[pairwise_diversity] ERROR: --bad-quantile must be in (0, 1), got "
              f"{args.bad_quantile}.")
        return EXIT_ERROR
    result_dir = resolve_result_dir(args.result)
    if not os.path.isdir(result_dir):
        print(f"[pairwise_diversity] ERROR: {args.result!r} is not a result folder.")
        return EXIT_ERROR
    try:
        test = read_result_table(result_dir, 'Prediction_result_test')
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"[pairwise_diversity] ERROR: could not read Prediction_result_test in "
              f"{result_dir}: {exc}")
        return EXIT_ERROR
    if test is None or test.shape[0] == 0 or 'actual' not in test.columns:
        print("[pairwise_diversity] Prediction_result_test not found, empty, or without an "
              "'actual' column - nothing to evaluate.")
        return EXIT_NOTHING
    wanted = [m.strip() for m in args.models.split(',') if m.strip()] if args.models else None
    members = member_columns(prediction_columns(test), wanted)
    if len(members) < 2:
        print(f"[pairwise_diversity] fewer than two member columns found ({members}); pass "
              f"--models to name them.")
        return EXIT_NOTHING
    table = pairwise_table(test, members, args.bad_quantile, args.min_lines)
    if table.empty:
        print("[pairwise_diversity] no task has a pair with enough lines.")
        return EXIT_NOTHING
    summary = summarise(table)
    out_dir = args.out_dir or result_dir
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(os.path.join(out_dir, 'Pairwise_diversity.csv'), index=False)
    summary.to_csv(os.path.join(out_dir, 'Pairwise_diversity_summary.csv'), index=False)
    with pd.option_context('display.width', 170, 'display.max_columns', 30,
                           'display.float_format', '{:.3f}'.format):
        print(summary.to_string(index=False))
    worst = summary.sort_values('err_corr', ascending=False).iloc[0]
    print(f"[pairwise_diversity] most redundant pair: {worst['model_a']} / {worst['model_b']} "
          f"(error correlation {worst['err_corr']:.3f}). Wrote Pairwise_diversity.csv and "
          f"Pairwise_diversity_summary.csv in {out_dir}.")
    return EXIT_OK


if __name__ == '__main__':
    sys.exit(main())
