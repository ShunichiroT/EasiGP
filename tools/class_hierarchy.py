"""class_hierarchy.py - ver4-102 (catalogue H07, the read-out half of the ver4-91 P3 class hierarchy).

A flat ensemble gives every member the same weight, so a model list with four tree models and one
kernel model is, in effect, a tree ensemble with a kernel garnish (the dilution problem, George
2010; ver4-69 saw BayesB take 0.94 of the weight at equal accuracy to rrBLUP). The ver4-91 P3
design averages in levels instead: tuning variants of one model -> that model -> the models of one
architecture class (``model_registry.MODEL_CLASS``) -> the ensemble, each level equal-weighted.
This tool computes that hierarchical equal-weight ensemble from a finished run's own
``Prediction_result_test`` and puts it next to the flat naive ensemble and the Super learner, so
the question "would this hierarchy have done better or worse here?" is answered from data, per
task and paired over the tasks, with nothing refitted.

What it is not. The ver4-91 design also had an engine row (``ensemble__family``) and a class-grouped
Super learner (``group_by='class'``). Neither is built: both would add accumulators and Metric rows
to the 10,000-line engine, and the design's own pre-registered gate (Q3: the class-grouped Super
learner within 0.003 r of the live one, plus a gain in how well the effect read-out recovers the
true loci) was never run. This tool needs none of that: it reads files, and its numbers on real
runs are the evidence a later build decision would need. No claim of an accuracy gain is made for
the hierarchy, and none was measured in simulation.

Rows for a task with no validation split are scored on the same test lines as every other column;
the equal weights use no phenotype, so there is nothing to leak. Pearson and MSE here are
unweighted (they equal ``Metric.csv`` unless BLUE weighting was on).

Members are the prediction columns whose model is in ``model_registry.MODEL_CLASS`` (a tuning
suffix ``RF__Grid`` is stripped for the class lookup and groups the column under its base model
``RF``); the ensemble and weighted-ensemble columns are references, not members. ``--models``
replaces the member rule with an explicit comma-separated list. A task is scored on the lines
where the actual value and every member prediction are finite, and needs ``--min-lines`` of them.

Output (in ``--out-dir``, default the result folder):

* ``Class_hierarchy_weights.csv``: one row per member: ``model``, ``base``, ``class``,
  ``weight_flat`` (1/K) and ``weight_hierarchy`` (its share under the four equal-weight levels).
* ``Class_hierarchy.csv``: per task and ``method`` (``flat_equal``, ``class_hierarchy``, and the
  reference columns that exist, e.g. ``ensemble``, ``Super learner``): ``n``, ``pearson``, ``mse``.
* ``Class_hierarchy_summary.csv``: per (option, population, phenotype, ratio, method): ``n_tasks``,
  mean ``pearson`` and ``mse``, and for ``class_hierarchy`` the paired mean difference to
  ``flat_equal`` (``d_pearson_vs_flat``, ``d_mse_vs_flat``) with its t statistic over the tasks
  (``t_vs_flat``, naive: tasks of one run share their genotypes, so read it as a rough guide, not
  a test; ``iterative_gate.py`` has the corrected one). ``d_pearson_vs_sl`` is the same against the
  Super learner when its column is present.

Example::

    python tools/class_hierarchy.py --result Result/run1

Exit codes: 0 = written, 1 = error, 3 = nothing to evaluate (no prediction file, fewer than two
members, or no task with enough lines).
"""
import argparse
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from iterative_gate import read_result_table, resolve_result_dir  # noqa: E402
from prediction_diagnostics import TASK_COLUMNS, prediction_columns  # noqa: E402
from model_registry import model_class  # noqa: E402

EXIT_OK, EXIT_ERROR, EXIT_NOTHING = 0, 1, 3
DEFAULT_MIN_LINES = 10
REFERENCE_LABELS = ('ensemble', 'Super learner')


def base_of_column(label: str) -> str:
    """The base model of a prediction column: the part before a tuning-algorithm suffix
    (``RF__Grid`` -> ``RF``). A bio-prior instance suffix (``..._2``) is kept, because two
    instances are two different models."""
    return str(label).split('__', 1)[0]


def hierarchy_weights(members: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """``(flat, hierarchy)`` weight vectors over `members`, each summing to 1.

    ``flat`` is 1/K. ``hierarchy`` is the product of the four equal-weight levels: the ensemble
    averages its classes, a class averages its base models, a base model averages its tuning
    variants. Raises ValueError for an empty `members`."""
    members = list(members)
    if not members:
        raise ValueError("no members")
    flat = np.full(len(members), 1.0 / len(members))
    bases = [base_of_column(m) for m in members]
    classes = {b: model_class(b) for b in bases}
    class_list = sorted(set(classes.values()))
    hier = np.zeros(len(members))
    for i, m in enumerate(members):
        cls = classes[bases[i]]
        bases_in_class = sorted({b for b in bases if classes[b] == cls})
        variants = sum(1 for b in bases if b == bases[i])
        hier[i] = (1.0 / len(class_list)) * (1.0 / len(bases_in_class)) * (1.0 / variants)
    return flat, hier


def _scores(pred: np.ndarray, actual: np.ndarray) -> Tuple[float, float]:
    mse = float(np.mean((pred - actual) ** 2))
    if pred.size < 3 or pred.std() == 0.0 or actual.std() == 0.0:
        return float('nan'), mse
    return float(np.corrcoef(pred, actual)[0, 1]), mse


def member_columns(columns: Sequence[str], wanted: Optional[Sequence[str]] = None) -> List[str]:
    """Member columns in table order: the names in `wanted` (exact or ``<name>__...``), else every
    column with a registered model class."""
    if wanted:
        names = list(wanted)
        return [c for c in columns if any(c == w or c.startswith(w + '__') for w in names)]
    return [c for c in columns if model_class(c) != 'other']


def hierarchy_table(test: pd.DataFrame, members: Sequence[str],
                    min_lines: int = DEFAULT_MIN_LINES) -> pd.DataFrame:
    """One row per task and method (see the module docstring)."""
    flat, hier = hierarchy_weights(members)
    task_cols = [c for c in TASK_COLUMNS if c in test.columns]
    references = [c for c in REFERENCE_LABELS if c in test.columns]
    rows = []
    grouped = test.groupby(task_cols, sort=False, dropna=False) if task_cols else [((), test)]
    for key, part in grouped:
        key = key if isinstance(key, tuple) else (key,)
        label = dict(zip(task_cols, key))
        actual = part['actual'].to_numpy(dtype=float)
        matrix = part[list(members)].to_numpy(dtype=float)
        ok = np.isfinite(actual) & np.isfinite(matrix).all(axis=1)
        if int(ok.sum()) < min_lines:
            continue
        a, m = actual[ok], matrix[ok]
        series: Dict[str, np.ndarray] = {'flat_equal': m @ flat, 'class_hierarchy': m @ hier}
        for ref in references:
            values = part[ref].to_numpy(dtype=float)[ok]
            if np.isfinite(values).all():
                series[ref] = values
        for method, pred in series.items():
            r, mse = _scores(pred, a)
            rows.append({**label, 'method': method, 'n': int(ok.sum()), 'pearson': r, 'mse': mse})
    return pd.DataFrame(rows, columns=task_cols + ['method', 'n', 'pearson', 'mse'])


def summarise(table: pd.DataFrame) -> pd.DataFrame:
    """Per (option, population, phenotype, ratio, method): mean Pearson / MSE over the tasks, and
    for ``class_hierarchy`` the paired differences to ``flat_equal`` and to the Super learner."""
    keys = [c for c in ('asreml_option', 'population', 'phenotype', 'ratio') if c in table.columns]
    sample_key = [c for c in ('sample',) if c in table.columns]
    out = []
    for gkey, part in table.groupby(keys, sort=True, dropna=False) if keys else [((), table)]:
        gkey = gkey if isinstance(gkey, tuple) else (gkey,)
        wide_r = part.pivot_table(index=sample_key, columns='method', values='pearson',
                                  aggfunc='first') if sample_key else None
        wide_m = part.pivot_table(index=sample_key, columns='method', values='mse',
                                  aggfunc='first') if sample_key else None
        for method, sub in part.groupby('method', sort=False):
            row = dict(zip(keys, gkey))
            row.update(method=method, n_tasks=int(sub.shape[0]),
                       pearson=float(sub['pearson'].mean()), mse=float(sub['mse'].mean()),
                       d_pearson_vs_flat=float('nan'), d_mse_vs_flat=float('nan'),
                       t_vs_flat=float('nan'), d_pearson_vs_sl=float('nan'))
            if method == 'class_hierarchy' and wide_r is not None and 'flat_equal' in wide_r:
                diff = (wide_r['class_hierarchy'] - wide_r['flat_equal']).dropna()
                dmse = (wide_m['class_hierarchy'] - wide_m['flat_equal']).dropna()
                if len(diff):
                    row['d_pearson_vs_flat'] = float(diff.mean())
                    row['d_mse_vs_flat'] = float(dmse.mean()) if len(dmse) else float('nan')
                    if len(diff) > 1 and diff.std(ddof=1) > 0:
                        row['t_vs_flat'] = float(diff.mean() / (diff.std(ddof=1) / math.sqrt(len(diff))))
                if 'Super learner' in wide_r:
                    gap = (wide_r['class_hierarchy'] - wide_r['Super learner']).dropna()
                    if len(gap):
                        row['d_pearson_vs_sl'] = float(gap.mean())
            out.append(row)
    return pd.DataFrame(out)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Hierarchical (variant -> model -> class) equal-weight ensemble, read-only.")
    parser.add_argument('--result', required=True, help="Result folder (or its RESULT_NAME)")
    parser.add_argument('--out-dir', default=None, help="default: the result folder")
    parser.add_argument('--models', default=None,
                        help="comma-separated member columns (default: every registered model)")
    parser.add_argument('--min-lines', type=int, default=DEFAULT_MIN_LINES,
                        help=f"lines a task needs (default {DEFAULT_MIN_LINES})")
    args = parser.parse_args(argv)
    result_dir = resolve_result_dir(args.result)
    if not os.path.isdir(result_dir):
        print(f"[class_hierarchy] ERROR: {args.result!r} is not a result folder.")
        return EXIT_ERROR
    try:
        test = read_result_table(result_dir, 'Prediction_result_test')
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"[class_hierarchy] ERROR: could not read Prediction_result_test in "
              f"{result_dir}: {exc}")
        return EXIT_ERROR
    if test is None or test.shape[0] == 0 or 'actual' not in test.columns:
        print("[class_hierarchy] Prediction_result_test not found, empty, or without an "
              "'actual' column - nothing to evaluate.")
        return EXIT_NOTHING
    wanted = [m.strip() for m in args.models.split(',') if m.strip()] if args.models else None
    members = member_columns(prediction_columns(test), wanted)
    if len(members) < 2:
        print(f"[class_hierarchy] fewer than two member columns found ({members}); pass "
              f"--models to name them.")
        return EXIT_NOTHING
    table = hierarchy_table(test, members, args.min_lines)
    if table.empty:
        print("[class_hierarchy] no task has enough lines with every member prediction.")
        return EXIT_NOTHING
    flat, hier = hierarchy_weights(members)
    weights = pd.DataFrame({'model': members, 'base': [base_of_column(m) for m in members],
                            'class': [model_class(m) for m in members],
                            'weight_flat': flat, 'weight_hierarchy': hier})
    summary = summarise(table)
    out_dir = args.out_dir or result_dir
    os.makedirs(out_dir, exist_ok=True)
    weights.to_csv(os.path.join(out_dir, 'Class_hierarchy_weights.csv'), index=False)
    table.to_csv(os.path.join(out_dir, 'Class_hierarchy.csv'), index=False)
    summary.to_csv(os.path.join(out_dir, 'Class_hierarchy_summary.csv'), index=False)
    with pd.option_context('display.width', 170, 'display.max_columns', 30,
                           'display.float_format', '{:.3f}'.format):
        print(weights.to_string(index=False))
        print(summary.to_string(index=False))
    hit = summary[summary['method'] == 'class_hierarchy']
    if not hit.empty and hit['d_pearson_vs_flat'].notna().any():
        print(f"[class_hierarchy] class hierarchy minus flat equal weighting: mean Pearson "
              f"{hit['d_pearson_vs_flat'].mean():+.4f} over {hit.shape[0]} group(s). A read-out of "
              f"this run, not a recommendation: see the module docstring.")
    print(f"[class_hierarchy] wrote Class_hierarchy_weights.csv, Class_hierarchy.csv and "
          f"Class_hierarchy_summary.csv in {out_dir}.")
    return EXIT_OK


if __name__ == '__main__':
    sys.exit(main())
