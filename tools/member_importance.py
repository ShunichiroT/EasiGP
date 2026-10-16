"""member_importance.py - ver4-98 (catalogue X04, the class read-out of the ver4-91 P3 design).

Raw ensemble weights are an unstable way to say "which model matters" when several members are
near-duplicates: ver4-69 found BayesB at 0.94 and rrBLUP at 0 with identical out-of-fold r. The
information is in the architecture class, not in the individual column. This tool adds the
weights of a task's members up per class (``model_registry.MODEL_CLASS``: additive, sparse,
kernel, tree, local, neural, graph, bio_prior, other) and also reports how concentrated the
weight is inside each class, so a reader can tell "the kernel class matters" from "one arbitrary
kernel model won".

Read-only, from ``Weight.csv`` (``.csv`` / ``.csv.gz`` / per-batch files). One row of that file
is one task x weighting method (``model`` column). Member columns are every numeric column that
is not an ID column; a tuning suffix (``RF__Grid``) and a bio-prior instance suffix are removed
before the class lookup. Weights are renormalised to sum to 1 over the member columns of the
row (the raw sum is kept as ``weight_sum``); a missing weight counts as 0.

Output (in ``--out-dir``, default the result folder):

* ``Member_class_shares.csv``: task columns, ``model`` (the weighting method), ``class``,
  ``share`` (sum of the weights of that class), ``n_members``, ``top_member`` (the member with
  the largest weight in the class), ``top_within_class`` (its share of the class total) and
  ``effective_members`` (1 / sum of squared within-class shares: 1 = one member carries the
  class, ``n_members`` = an even split).
* ``Member_class_shares_summary.csv``: the same over the tasks (samples) of each (population,
  phenotype, ratio, method, class): mean and SD of ``share`` and the mean ``effective_members``
  and ``top_within_class``, plus ``top_member_consistency``, the share of tasks in which the
  most frequent top member was also that task's top member, among the tasks where the class got
  weight (low = an arbitrary pick among near-duplicates). A class with no weight in a task
  counts as share 0 there.

The class map is the ver4-91 P3 one and is the only place a class is defined. Shapley values of
members (a performance-based alternative) are deliberately not part of this tool.

Example::

    python tools/member_importance.py --result Result/run1

Exit codes: 0 = written, 1 = error, 3 = nothing to evaluate.
"""
import argparse
import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from iterative_gate import read_result_table, resolve_result_dir  # noqa: E402
from model_registry import model_class  # noqa: E402

EXIT_OK, EXIT_ERROR, EXIT_NOTHING = 0, 1, 3
ID_COLUMNS = ('asreml_option', 'population', 'phenotype', 'model', 'ratio', 'sample')


def member_columns(weight: pd.DataFrame) -> List[str]:
    """The numeric non-ID columns of a Weight table."""
    return [c for c in weight.columns
            if c not in ID_COLUMNS and pd.api.types.is_numeric_dtype(weight[c])]


def class_shares(weight: pd.DataFrame) -> pd.DataFrame:
    """One row per Weight.csv row and class present in it (see the module docstring)."""
    if 'model' not in weight.columns:
        raise ValueError("the weight table has no 'model' column")
    members = member_columns(weight)
    if not members:
        raise ValueError("the weight table has no member column")
    keys = [c for c in ID_COLUMNS if c in weight.columns]
    classes = {m: model_class(m) for m in members}
    rows = []
    values = weight[members].fillna(0.0).to_numpy(dtype=float)
    for pos in range(weight.shape[0]):
        w = values[pos]
        total = float(w.sum())
        if not np.isfinite(total) or total <= 0.0:
            continue
        w = w / total
        label = {c: weight.iloc[pos][c] for c in keys}
        for cls in sorted(set(classes.values())):
            idx = [j for j, m in enumerate(members) if classes[m] == cls]
            part = w[idx]
            class_total = float(part.sum())
            if class_total <= 0.0:
                # a class that got no weight is a real zero in the mean over tasks, not a gap
                rows.append({**label, 'weight_sum': total, 'class': cls, 'share': 0.0,
                             'n_members': len(idx), 'top_member': None,
                             'top_within_class': float('nan'),
                             'effective_members': float('nan')})
                continue
            within = part / class_total
            top = int(np.argmax(part))
            rows.append({**label, 'weight_sum': total, 'class': cls, 'share': class_total,
                         'n_members': len(idx), 'top_member': members[idx[top]],
                         'top_within_class': float(within[top]),
                         'effective_members': float(1.0 / np.sum(within ** 2))})
    return pd.DataFrame(rows)


def summarise(shares: pd.DataFrame) -> pd.DataFrame:
    """Mean over the samples of each (option, population, phenotype, ratio, method, class)."""
    keys = [c for c in ('asreml_option', 'population', 'phenotype', 'ratio', 'model')
            if c in shares.columns] + ['class']
    out = []
    for key, part in shares.groupby(keys, sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        counts = part['top_member'].dropna().value_counts()
        out.append({**dict(zip(keys, key)), 'n_tasks': int(part.shape[0]),
                    'share_mean': float(part['share'].mean()),
                    'share_sd': float(part['share'].std(ddof=1)) if part.shape[0] > 1
                    else float('nan'),
                    'effective_members_mean': float(part['effective_members'].mean()),
                    'top_within_class_mean': float(part['top_within_class'].mean()),
                    'top_member_modal': str(counts.index[0]) if len(counts) else '',
                    'top_member_consistency': float(counts.iloc[0] / counts.sum())
                    if len(counts) else float('nan')})
    return pd.DataFrame(out)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Per-class ensemble weight shares of a run.")
    parser.add_argument('--result', required=True, help="Result folder (or its RESULT_NAME)")
    parser.add_argument('--out-dir', default=None, help="default: the result folder")
    args = parser.parse_args(argv)
    result_dir = resolve_result_dir(args.result)
    if not os.path.isdir(result_dir):
        print(f"[member_importance] ERROR: {args.result!r} is not a result folder.")
        return EXIT_ERROR
    try:
        weight = read_result_table(result_dir, 'Weight')
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"[member_importance] ERROR: could not read Weight in {result_dir}: {exc}")
        return EXIT_ERROR
    if weight is None or weight.shape[0] == 0:
        print(f"[member_importance] Weight.csv not found or empty in {result_dir} - was a "
              f"weighted ensemble or the Super learner run?")
        return EXIT_NOTHING
    try:
        shares = class_shares(weight)
    except ValueError as exc:
        print(f"[member_importance] ERROR: {exc}")
        return EXIT_ERROR
    if shares.empty:
        print("[member_importance] no Weight row has a positive weight total.")
        return EXIT_NOTHING
    summary = summarise(shares)
    out_dir = args.out_dir or result_dir
    os.makedirs(out_dir, exist_ok=True)
    shares.to_csv(os.path.join(out_dir, 'Member_class_shares.csv'), index=False)
    summary.to_csv(os.path.join(out_dir, 'Member_class_shares_summary.csv'), index=False)
    with pd.option_context('display.width', 170, 'display.max_columns', 30,
                           'display.float_format', '{:.3f}'.format):
        print(summary.to_string(index=False))
    other = shares[shares['class'] == 'other']['share'].mean() if (shares['class'] == 'other').any() \
        else 0.0
    if other > 0:
        print(f"[member_importance] NOTE: members of an unrecognised model name carry class "
              f"'other' (mean share {other:.3f}); add them to model_registry.MODEL_CLASS.")
    print(f"[member_importance] wrote Member_class_shares.csv and "
          f"Member_class_shares_summary.csv in {out_dir}.")
    return EXIT_OK


if __name__ == '__main__':
    sys.exit(main())
