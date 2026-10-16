"""iterative_gate.py - ver4-57, R1 + R4: gate A-IE.1 for the opt-in Iterative ensemble.

Why: the iterative ensemble was only shown to be non-inferior to the naive ensemble on simulated
data (blueprint Appendix A: realised change -0.005 to +0.004 r, within 1 SE). Whether it holds on
real data is decided here, offline, from a finished run's own files. Read-only; it imports neither
``main_app`` nor R nor any model (I1).

Inputs (both ``.csv`` and ``.csv.gz`` are accepted; an assembled file wins over per-batch files
``<stem>_<idx>.csv[.gz]``, which are concatenated when no assembled file exists):

* ``Metric.csv`` - one row per (task, model) with ``Pearson correlation``. The ``Iterative
  ensemble`` row of a task is paired with the ``ensemble`` row of the SAME task (population,
  phenotype, ratio, sample and, for multi-option ASReml runs, ``asreml_option``). A tuning-method
  suffix is matched too (``Iterative ensemble__Grid`` with ``ensemble__Grid``).
* ``Iterative_ensemble_rounds.csv`` - per-round diagnostics, used for the hindsight headroom
  (best round's test Pearson minus round 0's, per task) and the mean number of refit rounds tried.
  Optional: without it those two columns are blank.

Output (AC4.5): per phenotype, and over all phenotypes, the number of paired tasks, the realised
delta r (iterative minus naive ensemble), its standard error, the share of tasks the iterative
ensemble won, the hindsight headroom and the mean refit rounds.

ver4-68 (R5): ``--baseline LABEL`` (default ``ensemble``) chooses the row the iterative ensemble is
paired with - e.g. ``--baseline "Analytic least-squares"`` pairs ``Iterative ensemble__X`` with
``Analytic least-squares__X`` for every tuning suffix ``__X``. Why: the two-phase mode is gated
twice (blueprint AC1.11): against the naive ensemble (the default) and against the best weighted
row present in the run. A baseline label with no row in ``Metric.csv`` exits with code 3 and names
the label. Without the flag the output and the exit code are exactly what they were.

ver4-77 (R2): ``--label LABEL`` (default ``Iterative ensemble``) chooses the row that is paired with
the baseline - ``--label "Super learner"`` gates the Super learner the same way (blueprint AC2.14:
``--baseline ensemble`` and ``--baseline "Analytic least-squares"``, mean delta r >= +0.010 is the
acceptance level the blueprint sets, passed with ``--threshold 0.010``). The per-round diagnostics
belong to the Iterative ensemble only, so with another label they are not read and those two
columns stay blank. Without the flag the output and the exit code are exactly what they were.

Gate A-IE.1: PASS when at least ``--min-replicates`` (default 20) paired tasks exist and the mean
realised delta r over all of them is >= ``--threshold`` (default -0.002); FAIL when enough tasks
exist and it is below; INSUFFICIENT when there are fewer tasks. A FAIL is not a rollback: the
checkbox stays labelled experimental and off by default.

Example::

    python tools/iterative_gate.py --result Result/run1 --min-replicates 20

Exit codes: 0 = PASS, 1 = error, 2 = FAIL, 3 = nothing to evaluate / INSUFFICIENT.
"""
import argparse
import glob
import math
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXIT_PASS, EXIT_ERROR, EXIT_FAIL, EXIT_NOTHING = 0, 1, 2, 3

ITER_LABEL = 'Iterative ensemble'
NAIVE_LABEL = 'ensemble'
R_COLUMN = 'Pearson correlation'
DEFAULT_THRESHOLD = -0.002
DEFAULT_MIN_REPLICATES = 20


def read_result_table(result_dir: str, stem: str) -> Optional[pd.DataFrame]:
    """The table ``<stem>.csv`` / ``<stem>.csv.gz`` in `result_dir`, or its per-batch files
    ``<stem>_<idx>.csv[.gz]`` concatenated in index order; None when nothing exists.

    Why both: a parallel run writes per-batch files and Step 2 assembles them, so a half-finished
    run (or one assembled elsewhere) may only have the batch files."""
    for suffix in ('.csv.gz', '.csv'):
        path = os.path.join(result_dir, stem + suffix)
        if os.path.isfile(path):
            return pd.read_csv(path)
    batch_paths = []
    for suffix in ('.csv.gz', '.csv'):
        batch_paths.extend(glob.glob(os.path.join(result_dir, f'{stem}_*{suffix}')))

    def batch_index(path: str) -> Tuple[int, str]:
        tail = os.path.basename(path)[len(stem) + 1:].split('.')[0]
        return (int(tail) if tail.isdigit() else 10 ** 9, path)

    batch_paths = sorted(set(batch_paths), key=batch_index)
    if not batch_paths:
        return None
    return pd.concat([pd.read_csv(p) for p in batch_paths], ignore_index=True)


def task_key_columns(frame: pd.DataFrame) -> List[str]:
    """The columns that identify one task (the optional ASReml option column leads)."""
    keys = [c for c in ('asreml_option', 'population', 'phenotype', 'ratio', 'sample')
            if c in frame.columns]
    return keys


def baseline_present(metric: pd.DataFrame, baseline: str = NAIVE_LABEL) -> bool:
    """True when `metric` has at least one row of the baseline label: ``baseline`` itself or
    ``baseline__<tuning suffix>``."""
    models = metric['model'].astype(str)
    return bool(((models == baseline) | models.str.startswith(baseline + '__')).any())


def paired_deltas(metric: pd.DataFrame, baseline: str = NAIVE_LABEL,
                  label: str = ITER_LABEL) -> pd.DataFrame:
    """One row per task and iterative label: ``delta`` = iterative r minus the matching `baseline`
    row's r (default: the naive ensemble; the tuning suffix of the iterative label is carried over
    to the baseline label). Rows whose baseline row is missing, or with a NaN r, are dropped.
    `label` (ver4-77) is the row being gated: 'Iterative ensemble' (default) or 'Super learner'."""
    keys = task_key_columns(metric)
    table = metric[keys + ['model', R_COLUMN]].copy()
    table['key'] = table[keys].astype(str).agg('|'.join, axis=1)
    iterative = table[table['model'].astype(str).str.startswith(label)].copy()
    if iterative.empty:
        return iterative.assign(delta=[])
    iterative['reference_model'] = iterative['model'].astype(str).str.replace(
        label, baseline, n=1, regex=False)
    reference = table.rename(columns={'model': 'reference_model', R_COLUMN: 'reference_r'})
    merged = iterative.merge(reference[['key', 'reference_model', 'reference_r']],
                             on=['key', 'reference_model'], how='inner')
    merged['delta'] = merged[R_COLUMN] - merged['reference_r']
    return merged.dropna(subset=['delta'])


def round_summary(rounds: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Per task and iterative label: hindsight headroom (best round's test Pearson minus round
    0's, never negative) and the number of refit rounds tried. None when `rounds` is absent."""
    if rounds is None or rounds.empty:
        return None
    keys = task_key_columns(rounds)
    rounds = rounds.copy()
    rounds['key'] = rounds[keys].astype(str).agg('|'.join, axis=1)
    out = []
    for (key, model), part in rounds.groupby(['key', 'model'], sort=False):
        base = part.loc[part['round'] == 0, 'test_pearson_diagnostic']
        best = part['test_pearson_diagnostic'].max()
        headroom = (float(best) - float(base.iloc[0])) if len(base) else float('nan')
        out.append({'key': key, 'model': model, 'headroom': headroom,
                    'rounds_tried': int((part['round'] > 0).sum())})
    return pd.DataFrame(out)


def summarise(deltas: pd.DataFrame, rounds: Optional[pd.DataFrame]) -> pd.DataFrame:
    """The AC4.5 table: one row per phenotype plus an ``ALL`` row."""
    extra = round_summary(rounds)
    if extra is not None:
        joined = deltas.merge(extra, on=['key', 'model'], how='left')
    else:
        joined = deltas.assign(headroom=np.nan, rounds_tried=np.nan)
    rows = []
    groups = [(str(name), part) for name, part in joined.groupby('phenotype', sort=True)]
    groups.append(('ALL', joined))
    for name, part in groups:
        n = int(part.shape[0])
        sd = float(part['delta'].std(ddof=1)) if n > 1 else float('nan')
        rows.append({
            'phenotype': name, 'n_tasks': n, 'delta_r': float(part['delta'].mean()),
            'se': sd / math.sqrt(n) if n > 1 else float('nan'),
            'win_share': float((part['delta'] > 0).mean()),
            'hindsight_headroom': float(part['headroom'].mean()),
            'mean_refit_rounds': float(part['rounds_tried'].mean())})
    return pd.DataFrame(rows)


def gate_verdict(all_row: pd.Series, threshold: float, min_replicates: int) -> Tuple[str, int]:
    """('PASS' | 'FAIL' | 'INSUFFICIENT', exit code) for the pooled delta r."""
    if int(all_row['n_tasks']) < min_replicates:
        return 'INSUFFICIENT', EXIT_NOTHING
    if float(all_row['delta_r']) >= threshold:
        return 'PASS', EXIT_PASS
    return 'FAIL', EXIT_FAIL


def resolve_result_dir(value: str) -> str:
    """`value` as a directory, else as a run name under the EasiGP Result folder."""
    if os.path.isdir(value):
        return value
    from pipeline_utils import result_dir_path   # lazy: only needed for a bare run name
    candidate = result_dir_path(value)
    return candidate if os.path.isdir(candidate) else value


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Gate A-IE.1 for the Iterative ensemble.")
    parser.add_argument('--result', required=True,
                        help="Result folder of the run (or its RESULT_NAME)")
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                        help=f"minimum pooled delta r to PASS (default {DEFAULT_THRESHOLD})")
    parser.add_argument('--min-replicates', type=int, default=DEFAULT_MIN_REPLICATES,
                        help=f"paired tasks needed to decide (default {DEFAULT_MIN_REPLICATES})")
    parser.add_argument('--out', default=None, help="optional CSV path for the table")
    parser.add_argument('--baseline', default=NAIVE_LABEL,
                        help=f"model label the Iterative ensemble is paired with (default "
                             f"'{NAIVE_LABEL}'); a tuning suffix '__X' is matched too")
    parser.add_argument('--label', default=ITER_LABEL,
                        help=f"model label that is gated (default '{ITER_LABEL}'; "
                             f"'Super learner'); a tuning suffix '__X' is matched too")
    args = parser.parse_args(argv)

    result_dir = resolve_result_dir(args.result)
    if not os.path.isdir(result_dir):
        print(f"[iterative_gate] ERROR: {args.result!r} is not a result folder - nothing read.")
        return EXIT_ERROR
    try:
        metric = read_result_table(result_dir, 'Metric')
        # the per-round diagnostics exist for the Iterative ensemble only (ver4-77)
        rounds = (read_result_table(result_dir, 'Iterative_ensemble_rounds')
                  if args.label == ITER_LABEL else None)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"[iterative_gate] ERROR: could not read the result tables in {result_dir}: {exc}")
        return EXIT_ERROR
    if metric is None or R_COLUMN not in metric.columns or 'model' not in metric.columns:
        print(f"[iterative_gate] Metric.csv (or per-batch Metric_<idx>.csv) with a "
              f"'{R_COLUMN}' column was not found in {result_dir} - nothing to evaluate.")
        return EXIT_NOTHING
    if rounds is None and args.label == ITER_LABEL:
        print("[iterative_gate] NOTE: Iterative_ensemble_rounds.csv not found - hindsight "
              "headroom and mean refit rounds are left blank (was write_rounds off?).")

    if args.baseline != NAIVE_LABEL and not baseline_present(metric, args.baseline):
        print(f"[iterative_gate] The baseline '{args.baseline}' has no row in Metric.csv of "
              f"{result_dir} - nothing to pair with (was that model or weighted ensemble run?).")
        return EXIT_NOTHING
    deltas = paired_deltas(metric, args.baseline, args.label)
    if deltas.empty:
        config_key = 'SUPER_LEARNER' if args.label.startswith('Super learner') \
            else 'ITERATIVE_ENSEMBLE'
        print(f"[iterative_gate] No task has both an '{args.label}' and an '{args.baseline}' row "
              f"in Metric.csv - was {config_key} on, with that baseline selected?")
        return EXIT_NOTHING
    if args.baseline != NAIVE_LABEL:
        print(f"[iterative_gate] Baseline: '{args.baseline}' (delta r = iterative minus baseline).")
    if args.label != ITER_LABEL:
        print(f"[iterative_gate] Gated row: '{args.label}' (delta r = that row minus the "
              f"baseline).")

    table = summarise(deltas, rounds)
    with pd.option_context('display.width', 140, 'display.float_format', '{:.4f}'.format):
        print(table.to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"[iterative_gate] table written to {args.out}")

    all_row = table[table['phenotype'] == 'ALL'].iloc[0]
    verdict, code = gate_verdict(all_row, args.threshold, args.min_replicates)
    print(f"[iterative_gate] A-IE.1 {verdict}: pooled delta r = {all_row['delta_r']:.4f} "
          f"(SE {all_row['se']:.4f}, n = {int(all_row['n_tasks'])} paired tasks; "
          f"threshold >= {args.threshold}, minimum tasks {args.min_replicates}).")
    if verdict == 'FAIL':
        config_key = 'SUPER_LEARNER' if args.label.startswith('Super learner') \
            else 'ITERATIVE_ENSEMBLE'
        print(f"[iterative_gate] The checkbox stays experimental and off by default; no rollback "
              f"is needed (set {config_key} to None to switch it off).")
    return code


if __name__ == '__main__':
    sys.exit(main())
