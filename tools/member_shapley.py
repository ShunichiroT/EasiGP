"""member_shapley.py - Shapley value of each ensemble member (catalogue X04).

REBUILT 2026-10-15 (ver4-103 recovery patch): the ver4-100 original was not available; this is a
re-implementation from the recorded design (value function approved at ver4-100, exact up to 12
members, sampled above, values sum to the all-member r, class sums reported) and the interface
``tool_diagnostics.py`` calls (``main(argv) -> exit code``).

Question: how much of the ensemble's accuracy does each member account for, when members that
are near-duplicates must share the credit (raw ensemble weights split arbitrarily among them)?

Value function (the approved choice): ``v(S)`` = Pearson r, on the test lines of a task, between
the actual value and the UNWEIGHTED mean of the predictions of the members in S; ``v(empty) = 0``.
The Shapley value of member i is its average marginal contribution ``v(S + i) - v(S)`` over all
orders of adding the members, so the values of a task SUM to the all-member r (``total_r``). The
share is the value over that total.

Members = the prediction columns of ``Prediction_result_<split>`` that are base models: columns of
combined predictions (``ensemble``, ``Super learner``, ``Iterative ensemble``, the weighting
methods) are left out, otherwise the ensemble would be credited with itself. Rows with a missing
actual or member value are dropped (listwise) for that task.

Computation: Pearson of a mean equals Pearson of the sum, so only the member covariance matrix C
and the member-actual covariances c are needed: ``r(S) = 1'c / (sd_y * sqrt(1'C1))`` over S. Up to
``--exact-max`` (default 12) members all 2^n subsets are evaluated; above that
``--n-perm`` (default 200) random orders are sampled with ``--seed`` (default 0), and the
``method`` column says which was used (sampled values sum to the total exactly, since every order
telescopes, but individual values carry sampling error).

Output (in ``--out-dir``, default the result folder):

* ``Member_shapley.csv``: task columns, ``model`` (member), ``class`` (model_registry class),
  ``shapley``, ``share``, ``solo_r`` (the member's own r), ``total_r``, ``n_members``, ``n_lines``,
  ``method``.
* ``Member_shapley_summary.csv``: over the tasks of each (option, population, phenotype, ratio,
  member): mean and SD of ``shapley``, mean ``share`` and ``solo_r``.
* ``Member_shapley_class.csv``: the same per model class (sum of the members' values of the class
  in each task, then mean and SD over tasks).

A READ-OUT of a finished run. Do not use it to choose members on the same test lines it was
computed from; as ensemble weights it failed its gate (ice_sim42).

Example::

    python tools/member_shapley.py --result Result/run1 --split test

Exit codes: 0 = written, 1 = error, 3 = nothing to evaluate.
"""
import argparse
import os
import sys
from math import factorial
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from iterative_gate import read_result_table, resolve_result_dir  # noqa: E402
from model_registry import model_class  # noqa: E402
from prediction_diagnostics import ID_COLUMNS, TASK_COLUMNS  # noqa: E402
from stability_summary import is_combined_label  # noqa: E402

EXIT_OK, EXIT_ERROR, EXIT_NOTHING = 0, 1, 3
DEFAULT_EXACT_MAX = 12
DEFAULT_N_PERM = 200
MIN_LINES = 3


def member_columns(pred: pd.DataFrame) -> List[str]:
    """Base-model prediction columns: non-ID, numeric, not a combined prediction."""
    return [c for c in pred.columns if c not in ID_COLUMNS and not is_combined_label(c)
            and pd.api.types.is_numeric_dtype(pred[c])]


def _stats(P: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Member covariance matrix C, member-actual covariances c and the actual's SD."""
    Pc = P - P.mean(axis=0)
    yc = y - y.mean()
    n = P.shape[0]
    C = Pc.T @ Pc / (n - 1)
    c = Pc.T @ yc / (n - 1)
    return C, c, float(np.sqrt(yc @ yc / (n - 1)))


def _value(mask_rows: np.ndarray, C: np.ndarray, c: np.ndarray, sd_y: float) -> np.ndarray:
    """v(S) for every row of a 0/1 membership matrix (rows = subsets). Empty set and degenerate
    (zero-variance) sums are 0."""
    num = mask_rows @ c
    var = np.einsum('ij,jk,ik->i', mask_rows, C, mask_rows)
    out = np.zeros(mask_rows.shape[0])
    ok = (var > 1e-300) & (sd_y > 0)
    out[ok] = num[ok] / (sd_y * np.sqrt(var[ok]))
    return np.clip(out, -1.0, 1.0)


def shapley_exact(C: np.ndarray, c: np.ndarray, sd_y: float) -> np.ndarray:
    """All 2^n subsets. Returns the n Shapley values."""
    n = C.shape[0]
    masks = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(float)
    v = _value(masks, C, c, sd_y)
    size = masks.sum(axis=1).astype(int)
    weight = np.array([factorial(k) * factorial(n - k - 1) / factorial(n) for k in range(n)])
    phi = np.zeros(n)
    index = np.arange(2 ** n)
    for i in range(n):
        without = index[(index >> i) & 1 == 0]
        with_i = without | (1 << i)
        phi[i] = float(np.sum(weight[size[without]] * (v[with_i] - v[without])))
    return phi


def shapley_sampled(C: np.ndarray, c: np.ndarray, sd_y: float, n_perm: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Average marginal contribution over `n_perm` random orders (antithetic: each order and its
    reverse), exactly telescoping to v(all)."""
    n = C.shape[0]
    phi = np.zeros(n)
    count = 0
    for _ in range(max(1, n_perm // 2)):
        order = rng.permutation(n)
        for perm in (order, order[::-1]):
            masks = np.zeros((n + 1, n))
            for k in range(n):
                masks[k + 1:, perm[k]] = 1.0
            v = _value(masks, C, c, sd_y)
            phi[perm] += np.diff(v)
            count += 1
    return phi / count


def task_shapley(pred_task: pd.DataFrame, members: List[str], exact_max: int, n_perm: int,
                 seed: int) -> Optional[pd.DataFrame]:
    """Shapley rows of one task (None when it cannot be evaluated)."""
    data = pred_task[members + ['actual']].apply(pd.to_numeric, errors='coerce').dropna()
    if data.shape[0] < MIN_LINES:
        return None
    P = data[members].to_numpy(dtype=float)
    y = data['actual'].to_numpy(dtype=float)
    C, c, sd_y = _stats(P, y)
    n = len(members)
    if n <= exact_max:
        phi, method = shapley_exact(C, c, sd_y), 'exact'
    else:
        phi, method = shapley_sampled(C, c, sd_y, n_perm, np.random.default_rng(seed)), 'sampled'
    total = float(_value(np.ones((1, n)), C, c, sd_y)[0])
    solo = _value(np.eye(n), C, c, sd_y)
    share = phi / total if abs(total) > 1e-12 else np.full(n, np.nan)
    return pd.DataFrame({'model': members, 'class': [model_class(m) for m in members],
                         'shapley': phi, 'share': share, 'solo_r': solo, 'total_r': total,
                         'n_members': n, 'n_lines': data.shape[0], 'method': method})


def run(pred: pd.DataFrame, exact_max: int = DEFAULT_EXACT_MAX, n_perm: int = DEFAULT_N_PERM,
        seed: int = 0) -> pd.DataFrame:
    """Per-task Shapley table of a Prediction_result frame (empty when nothing evaluates)."""
    if 'actual' not in pred.columns:
        raise ValueError("the prediction table has no 'actual' column")
    members = member_columns(pred)
    if not members:
        return pd.DataFrame()
    keys = [c for c in TASK_COLUMNS if c in pred.columns]
    parts = []
    grouped = pred.groupby(keys, sort=False, dropna=False) if keys else [((), pred)]
    for t, (key, part) in enumerate(grouped):
        key = key if isinstance(key, tuple) else (key,)
        rows = task_shapley(part, members, exact_max, n_perm, seed + t)
        if rows is None:
            continue
        for k, v in zip(keys, key):
            rows.insert(keys.index(k), k, v)
        parts.append(rows)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def summarise(table: pd.DataFrame, by_class: bool = False) -> pd.DataFrame:
    keys = [c for c in ('asreml_option', 'population', 'phenotype', 'ratio') if c in table.columns]
    if by_class:
        task = [c for c in TASK_COLUMNS if c in table.columns]
        table = (table.groupby(task + ['class'], sort=False, dropna=False)
                 .agg(shapley=('shapley', 'sum'), share=('share', 'sum'), solo_r=('solo_r', 'max'),
                      total_r=('total_r', 'first'), n_members=('model', 'count'))
                 .reset_index())
        label = 'class'
    else:
        label = 'model'
    out = []
    for key, part in table.groupby(keys + [label], sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        out.append({**dict(zip(keys + [label], key)), 'n_tasks': int(part.shape[0]),
                    'shapley_mean': float(part['shapley'].mean()),
                    'shapley_sd': float(part['shapley'].std(ddof=1)) if part.shape[0] > 1
                    else float('nan'),
                    'share_mean': float(part['share'].mean()),
                    'solo_r_mean': float(part['solo_r'].mean()),
                    'total_r_mean': float(part['total_r'].mean())})
    return pd.DataFrame(out)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Shapley value of each ensemble member.")
    parser.add_argument('--result', required=True, help="Result folder (or its RESULT_NAME)")
    parser.add_argument('--out-dir', default=None, help="default: the result folder")
    parser.add_argument('--split', default='test', choices=('test', 'valid'))
    parser.add_argument('--exact-max', type=int, default=DEFAULT_EXACT_MAX,
                        help="exact enumeration up to this many members (default 12)")
    parser.add_argument('--n-perm', type=int, default=DEFAULT_N_PERM,
                        help="random orders above that (default 200)")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    if args.exact_max < 1 or args.exact_max > 16 or args.n_perm < 2:
        print("[member_shapley] ERROR: --exact-max must be 1..16 and --n-perm at least 2.")
        return EXIT_ERROR
    result_dir = resolve_result_dir(args.result)
    if not os.path.isdir(result_dir):
        print(f"[member_shapley] ERROR: {args.result!r} is not a result folder.")
        return EXIT_ERROR
    try:
        pred = read_result_table(result_dir, f'Prediction_result_{args.split}')
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"[member_shapley] ERROR: could not read the prediction table: {exc}")
        return EXIT_ERROR
    if pred is None or pred.shape[0] == 0:
        print(f"[member_shapley] Prediction_result_{args.split} not found or empty in {result_dir}.")
        return EXIT_NOTHING
    try:
        table = run(pred, args.exact_max, args.n_perm, args.seed)
    except ValueError as exc:
        print(f"[member_shapley] ERROR: {exc}")
        return EXIT_ERROR
    if table.empty:
        print("[member_shapley] no task had base-model prediction columns and enough lines.")
        return EXIT_NOTHING
    out_dir = args.out_dir or result_dir
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(os.path.join(out_dir, 'Member_shapley.csv'), index=False)
    summary = summarise(table)
    summary.to_csv(os.path.join(out_dir, 'Member_shapley_summary.csv'), index=False)
    summarise(table, by_class=True).to_csv(os.path.join(out_dir, 'Member_shapley_class.csv'),
                                           index=False)
    with pd.option_context('display.width', 170, 'display.max_columns', 30,
                           'display.float_format', '{:.3f}'.format):
        print(summary.to_string(index=False))
    if (table['method'] == 'sampled').any():
        print(f"[member_shapley] NOTE: more than {args.exact_max} members: values were sampled "
              f"({args.n_perm} orders, seed {args.seed}); they sum to the total exactly but "
              f"individual values carry sampling error.")
    print(f"[member_shapley] wrote Member_shapley.csv, Member_shapley_summary.csv and "
          f"Member_shapley_class.csv in {out_dir}.")
    return EXIT_OK


if __name__ == '__main__':
    sys.exit(main())
