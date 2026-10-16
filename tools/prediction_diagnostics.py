"""prediction_diagnostics.py - ver4-98 (catalogue O02 + V06; design ver4-93 R4 / R5).

Two questions a finished run's own files can answer, read-only:

* O02, dispersion: are a model's (or ensemble's) test predictions too spread out or too
  compressed? ``slope`` is the least-squares slope of the actual value on the prediction,
  ``cov(actual, pred) / var(pred)``. 1 = well calibrated, below 1 = over-dispersed predictions
  (shrink them), above 1 = under-dispersed. A convex combination of members (the Super learner
  simplex) cannot rescale its own output, so this is where such a bias becomes visible.
* V06, within-group accuracy (needs ``--groups``): the overall r of a multi-family test set
  mixes "which family" with "which line inside the family". Breeders select within families, so
  the mean within-group r (and the r of the group means) are reported next to the overall r.

Why a tool and not a ``Metric.csv`` column: ``Metric.csv`` rows are written at four engine sites
plus checkpoint, merge and pivot code; a diagnostic does not justify that surface (ver4-93 R4).

Input: ``Prediction_result_test`` (or ``_valid`` with ``--split valid``), ``.csv`` / ``.csv.gz`` or
per-batch files, read like ``iterative_gate.py`` does. One row per test line, with the ID columns
``asreml_option`` (multi-option ASReml runs only), ``id``, ``population``, ``ratio``,
``phenotype``, ``sample``, ``actual`` and then one column per prediction label. The ID columns
are recognised BY NAME, never by position.

Output (in ``--out-dir``, default the result folder):

* ``Prediction_diagnostics.csv``: one row per task x prediction column: ``n``, ``slope``,
  ``intercept``, ``pearson``, ``pred_sd``, ``actual_sd`` and, with ``--groups``, ``within_r_mean``
  (unweighted over groups), ``within_r_wmean`` (weighted by group size), ``n_groups``,
  ``between_r``.
* ``Prediction_diagnostics_summary.csv``: per (option, population, phenotype, ratio, model) over
  its tasks: ``n_tasks``, mean / SD of ``slope``, the share of tasks with slope < 0.9 and > 1.1,
  mean ``pearson``, and the group columns' means.

``pearson`` here is computed on the rows of ``Prediction_result_test``. It equals the Pearson of
``Metric.csv`` for the same task and model when ``metric_weighting`` is ``none``; under BLUE
weighting ``Metric.csv`` may score with weights and the two can differ.

The tool imports no model, no R and nothing from the GUI. Example::

    python tools/prediction_diagnostics.py --result Result/run1
    python tools/prediction_diagnostics.py --result Result/run1 --groups families.csv

Exit codes: 0 = written, 1 = error, 3 = nothing to evaluate (no prediction file, no task).
"""
import argparse
import math
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from iterative_gate import read_result_table, resolve_result_dir  # noqa: E402

EXIT_OK, EXIT_ERROR, EXIT_NOTHING = 0, 1, 3

#: Columns of a Prediction_result_* table that identify a line or a task (everything else is a
#: prediction column). Recognised by name.
ID_COLUMNS = ('asreml_option', 'id', 'population', 'ratio', 'phenotype', 'sample', 'actual')
#: The subset that identifies one task (``id`` and ``actual`` identify a line).
TASK_COLUMNS = ('asreml_option', 'population', 'phenotype', 'ratio', 'sample')
DEFAULT_MIN_GROUP_SIZE = 5
SLOPE_LOW, SLOPE_HIGH = 0.9, 1.1


def prediction_columns(frame: pd.DataFrame) -> List[str]:
    """The prediction columns of a Prediction_result table: every non-ID column."""
    return [c for c in frame.columns if c not in ID_COLUMNS]


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float('nan')
    sa, sb = a.std(), b.std()
    if sa == 0.0 or sb == 0.0:
        return float('nan')
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def dispersion_rows(pred: pd.DataFrame, id_cols: List[str], pred_cols: List[str],
                    notes: Optional[List[str]] = None) -> pd.DataFrame:
    """One row per task (the groups of `id_cols`) and prediction column of `pred`, from the rows
    where both the actual value and the prediction are finite.

    Columns: the `id_cols`, ``model``, ``n``, ``slope`` (= cov(actual, pred) / var(pred), the
    regression of actual on prediction), ``intercept``, ``pearson``, ``pred_sd``,
    ``actual_sd``. A prediction column that is entirely missing in a task (a model that was not
    run there) gives no row. A constant column (variance 0) gives NaN slope and Pearson, and one
    NOTE per column is appended to `notes` (and printed when `notes` is None). Population
    variances (ddof=0) are used throughout; the slope is the same either way."""
    rows = []
    noted = set()
    grouped = pred.groupby(id_cols, sort=False, dropna=False) if id_cols else [((), pred)]
    for key, part in grouped:
        key = key if isinstance(key, tuple) else (key,)
        actual = part['actual'].to_numpy(dtype=float)
        for col in pred_cols:
            values = part[col].to_numpy(dtype=float)
            ok = np.isfinite(actual) & np.isfinite(values)
            if not ok.any():
                continue
            a, p = actual[ok], values[ok]
            var_p = float(p.var())
            row = dict(zip(id_cols, key))
            row.update(model=col, n=int(ok.sum()), pred_sd=math.sqrt(var_p),
                       actual_sd=float(a.std()))
            if ok.sum() < 3 or var_p == 0.0:
                if var_p == 0.0 and col not in noted:
                    noted.add(col)
                    message = (f"NOTE: the predictions of '{col}' are constant in at least one "
                               f"task - slope and Pearson are NaN there.")
                    if notes is None:
                        print(f"[prediction_diagnostics] {message}")
                    else:
                        notes.append(message)
                row.update(slope=float('nan'), intercept=float('nan'), pearson=float('nan'))
            else:
                slope = float(np.mean((a - a.mean()) * (p - p.mean())) / var_p)
                row.update(slope=slope, intercept=float(a.mean() - slope * p.mean()),
                           pearson=_pearson(a, p))
            rows.append(row)
    columns = list(id_cols) + ['model', 'n', 'slope', 'intercept', 'pearson', 'pred_sd',
                               'actual_sd']
    return pd.DataFrame(rows, columns=columns)


def read_groups(path: str) -> pd.Series:
    """A Series (index = ID as str, value = group as str) from a CSV whose first two columns are
    the ID and the group (by position, like ``unify_columns_by_position``).

    Raises ValueError naming the duplicated IDs when an ID appears twice, or when the file has
    fewer than two columns."""
    table = pd.read_csv(path, dtype=str)
    if table.shape[1] < 2:
        raise ValueError(f"{path}: the groups file needs two columns (ID, group)")
    ids = table.iloc[:, 0].astype(str).str.strip()
    groups = table.iloc[:, 1].astype(str).str.strip()
    duplicated = ids[ids.duplicated(keep=False)].unique().tolist()
    if duplicated:
        shown = ', '.join(duplicated[:10]) + (' ...' if len(duplicated) > 10 else '')
        raise ValueError(f"{path}: {len(duplicated)} duplicated ID(s): {shown}")
    return pd.Series(groups.to_numpy(), index=ids.to_numpy(), name='group')


def within_group_rows(pred: pd.DataFrame, groups: pd.Series, id_cols: List[str],
                      pred_cols: List[str], min_size: int = DEFAULT_MIN_GROUP_SIZE,
                      notes: Optional[List[str]] = None) -> pd.DataFrame:
    """Per task and prediction column, the accuracy inside the groups of `groups`.

    Only test lines whose ID has a group are used, and only groups with at least `min_size` of
    them (finite actual and prediction) AND a non-constant actual and prediction inside the
    group. Columns: the `id_cols`, ``model``, ``within_r_mean`` (mean of the per-group Pearson r,
    each group counted once), ``within_r_wmean`` (the same weighted by group size),
    ``n_groups`` (groups used) and ``between_r`` (Pearson r of the group means, needs at least 3
    groups). With fewer than 2 usable groups everything but ``n_groups`` is NaN and one NOTE per
    task is appended to `notes` (printed when `notes` is None). The per-group r of a small group
    is noisy; `min_size` trades that against coverage."""
    ids = pred['id'].astype(str).str.strip()
    group_of = ids.map(groups)
    rows = []
    grouped = pred.assign(_group=group_of.to_numpy()).groupby(id_cols, sort=False, dropna=False) \
        if id_cols else [((), pred.assign(_group=group_of.to_numpy()))]
    for key, part in grouped:
        key = key if isinstance(key, tuple) else (key,)
        actual = part['actual'].to_numpy(dtype=float)
        label = part['_group'].to_numpy(dtype=object)
        has_group = pd.notna(label)
        for col in pred_cols:
            values = part[col].to_numpy(dtype=float)
            ok = np.isfinite(actual) & np.isfinite(values) & has_group
            if not ok.any():
                continue
            r_list, size_list, mean_a, mean_p = [], [], [], []
            for g in pd.unique(label[ok]):
                sel = ok & (label == g)
                if sel.sum() < min_size:
                    continue
                a, p = actual[sel], values[sel]
                r = _pearson(a, p)
                if not np.isfinite(r):
                    continue
                r_list.append(r)
                size_list.append(int(sel.sum()))
                mean_a.append(a.mean())
                mean_p.append(p.mean())
            row = dict(zip(id_cols, key))
            row.update(model=col, n_groups=len(r_list), within_r_mean=float('nan'),
                       within_r_wmean=float('nan'), between_r=float('nan'))
            if len(r_list) >= 2:
                sizes = np.asarray(size_list, dtype=float)
                row['within_r_mean'] = float(np.mean(r_list))
                row['within_r_wmean'] = float(np.average(r_list, weights=sizes))
                if len(r_list) >= 3:
                    row['between_r'] = _pearson(np.asarray(mean_a), np.asarray(mean_p))
            elif col == pred_cols[0]:
                message = (f"NOTE: fewer than 2 groups with at least {min_size} usable test "
                           f"lines in task {dict(zip(id_cols, key))} - within-group columns are "
                           f"NaN there.")
                if notes is None:
                    print(f"[prediction_diagnostics] {message}")
                else:
                    notes.append(message)
            rows.append(row)
    columns = list(id_cols) + ['model', 'within_r_mean', 'within_r_wmean', 'n_groups',
                               'between_r']
    return pd.DataFrame(rows, columns=columns)


def summarise(per_task: pd.DataFrame, id_cols: List[str]) -> pd.DataFrame:
    """The per-model summary over tasks: group by every task column except ``sample``."""
    keys = [c for c in id_cols if c != 'sample'] + ['model']
    extra = [c for c in ('within_r_mean', 'within_r_wmean', 'between_r') if c in per_task.columns]
    out = []
    for key, part in per_task.groupby(keys, sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        slope = part['slope'].dropna()
        row = dict(zip(keys, key))
        row.update(
            n_tasks=int(part.shape[0]),
            slope_mean=float(slope.mean()) if len(slope) else float('nan'),
            slope_sd=float(slope.std(ddof=1)) if len(slope) > 1 else float('nan'),
            share_slope_below=float((slope < SLOPE_LOW).mean()) if len(slope) else float('nan'),
            share_slope_above=float((slope > SLOPE_HIGH).mean()) if len(slope) else float('nan'),
            pearson_mean=float(part['pearson'].mean()))
        for column in extra:
            row[column + '_mean'] = float(part[column].mean())
        out.append(row)
    return pd.DataFrame(out)


def run(pred: pd.DataFrame, groups: Optional[pd.Series] = None,
        min_group_size: int = DEFAULT_MIN_GROUP_SIZE, notes: Optional[List[str]] = None):
    """The whole computation on an in-memory Prediction_result table.

    Returns ``(per_task, summary)``. Raises ValueError when the table has no ``actual`` column or
    no prediction column."""
    if 'actual' not in pred.columns:
        raise ValueError("the prediction table has no 'actual' column")
    pred_cols = prediction_columns(pred)
    if not pred_cols:
        raise ValueError("the prediction table has no prediction column")
    id_cols = [c for c in TASK_COLUMNS if c in pred.columns]
    per_task = dispersion_rows(pred, id_cols, pred_cols, notes)
    if groups is not None:
        if 'id' not in pred.columns:
            raise ValueError("--groups needs an 'id' column in the prediction table")
        within = within_group_rows(pred, groups, id_cols, pred_cols, min_group_size, notes)
        per_task = per_task.merge(within, on=id_cols + ['model'], how='left')
    return per_task, summarise(per_task, id_cols)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibration slope (O02) and within-group accuracy (V06) of a finished run.")
    parser.add_argument('--result', required=True,
                        help="Result folder of the run (or its RESULT_NAME)")
    parser.add_argument('--split', choices=('test', 'valid'), default='test',
                        help="which prediction table to read (default test)")
    parser.add_argument('--out-dir', default=None,
                        help="where the two CSVs go (default: the result folder)")
    parser.add_argument('--groups', default=None,
                        help="CSV with columns ID, group (first two columns): adds the "
                             "within-group columns")
    parser.add_argument('--min-group-size', type=int, default=DEFAULT_MIN_GROUP_SIZE,
                        help=f"smallest group (test lines) that counts (default "
                             f"{DEFAULT_MIN_GROUP_SIZE})")
    args = parser.parse_args(argv)

    result_dir = resolve_result_dir(args.result)
    if not os.path.isdir(result_dir):
        print(f"[prediction_diagnostics] ERROR: {args.result!r} is not a result folder - "
              f"nothing read.")
        return EXIT_ERROR
    stem = f'Prediction_result_{args.split}'
    try:
        pred = read_result_table(result_dir, stem)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"[prediction_diagnostics] ERROR: could not read {stem} in {result_dir}: {exc}")
        return EXIT_ERROR
    if pred is None or pred.shape[0] == 0:
        print(f"[prediction_diagnostics] {stem}.csv (or per-batch files) not found or empty in "
              f"{result_dir} - nothing to evaluate.")
        return EXIT_NOTHING
    groups = None
    if args.groups:
        try:
            groups = read_groups(args.groups)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            print(f"[prediction_diagnostics] ERROR: {exc}")
            return EXIT_ERROR
        if 'id' in pred.columns:
            share = float(pred['id'].astype(str).str.strip().isin(groups.index).mean())
            print(f"[prediction_diagnostics] {100 * share:.1f}% of the {args.split} rows have a "
                  f"group; the others are left out of the within-group columns.")
    notes: List[str] = []
    try:
        per_task, summary = run(pred, groups, args.min_group_size, notes)
    except ValueError as exc:
        print(f"[prediction_diagnostics] ERROR: {exc}")
        return EXIT_ERROR
    for message in notes:
        print(f"[prediction_diagnostics] {message}")
    if per_task.empty:
        print("[prediction_diagnostics] no task has a finite prediction - nothing to evaluate.")
        return EXIT_NOTHING
    out_dir = args.out_dir or result_dir
    os.makedirs(out_dir, exist_ok=True)
    per_path = os.path.join(out_dir, 'Prediction_diagnostics.csv')
    sum_path = os.path.join(out_dir, 'Prediction_diagnostics_summary.csv')
    per_task.to_csv(per_path, index=False)
    summary.to_csv(sum_path, index=False)
    with pd.option_context('display.width', 160, 'display.max_columns', 30,
                           'display.float_format', '{:.3f}'.format):
        print(summary.to_string(index=False))
    print(f"[prediction_diagnostics] wrote {per_path} and {sum_path}. Reading guide: slope 1 = "
          f"calibrated; below {SLOPE_LOW} = predictions spread too widely (shrink); above "
          f"{SLOPE_HIGH} = too compressed.")
    return EXIT_OK


if __name__ == '__main__':
    sys.exit(main())
