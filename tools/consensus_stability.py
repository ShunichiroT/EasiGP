"""consensus_stability.py - stability of the top markers (and pairs) of a finished run.

REBUILT 2026-10-15 (ver4-103 recovery patch): the ver4-99 original was not available; this is a
re-implementation from the recorded design. All the logic lives in ``stability_summary.py`` (read
its docstring for the definitions and limits); this file is the command line around it, the entry
point ``tool_diagnostics.py`` calls (``main(argv) -> exit code``).

Read-only, from ``Marker_effect`` (and with ``--pairs`` from ``Interaction``), each ``.csv`` /
``.csv.gz`` or per-batch files, read like the other tools do.

Output (in ``--out-dir``, default the result folder):

* ``Consensus_stability.csv``: one row per group (population, phenotype, ratio) and level
  (marker / pair): number of iterations, mean pairwise Jaccard of the top-K sets, markers
  selected at least once and at frequency >= 0.5 / >= 0.9 / = 1.
* ``Marker_selection_frequency.csv`` (with ``--frequencies``, the default): per group and marker
  the share of iterations in which it was in the top K.
* ``Pair_selection_frequency.csv`` (with ``--pairs`` and ``--frequencies``): the same for pairs.
* ``Interaction_selection_frequency.csv`` (with ``--interaction-frequency``, X03; implies
  ``--pairs``): every row of the run's ``Interaction.csv`` with the added columns
  ``pair_frequency``, ``marker1_frequency`` and ``marker2_frequency``. Needs a single
  ``Interaction.csv[.gz]`` (not per-batch files). ``Interaction.csv`` is never changed.

Options: ``--top-k N`` (default 50), ``--bin N`` (pool N consecutive marker columns into one unit,
markers only; default 1), ``--models A,B,...`` (default: every base model, combined rows left
out), ``--pairs`` (also the pair tables; needs Interaction), ``--no-frequencies`` (write only
Consensus_stability.csv), ``--chunksize N`` (rows per read of Interaction; changes memory, not
results).

Example::

    python tools/consensus_stability.py --result Result/run1 --top-k 50 --pairs

Exit codes: 0 = written, 1 = error, 3 = nothing to evaluate (no Marker_effect, or no task).
"""
import argparse
import os
import sys
from typing import List, Optional

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from iterative_gate import read_result_table, resolve_result_dir  # noqa: E402
import stability_summary as ss  # noqa: E402

EXIT_OK, EXIT_ERROR, EXIT_NOTHING = 0, 1, 3


def _interaction_source(result_dir: str):
    """Path of Interaction.csv[.gz] when one exists (a per-batch run is read via the table
    reader instead), else None."""
    for suffix in ('.csv.gz', '.csv'):
        path = os.path.join(result_dir, 'Interaction' + suffix)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            return path
    return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Stability of the top markers / pairs of a run.")
    parser.add_argument('--result', required=True, help="Result folder (or its RESULT_NAME)")
    parser.add_argument('--out-dir', default=None, help="default: the result folder")
    parser.add_argument('--top-k', type=int, default=ss.DEFAULT_TOP_K)
    parser.add_argument('--bin', type=int, default=1, dest='bin_size',
                        help="pool this many consecutive marker columns (default 1)")
    parser.add_argument('--models', default='',
                        help="comma-separated model labels (default: all base models)")
    parser.add_argument('--pairs', action='store_true', help="also the marker-pair tables")
    parser.add_argument('--frequencies', action=argparse.BooleanOptionalAction, default=True,
                        help="write the selection-frequency tables (default on)")
    parser.add_argument('--interaction-frequency', action='store_true',
                        help="X03: also write Interaction_selection_frequency.csv (implies --pairs)")
    parser.add_argument('--chunksize', type=int, default=ss.DEFAULT_CHUNKSIZE)
    args = parser.parse_args(argv)
    if args.interaction_frequency:
        args.pairs = True
    if args.top_k < 1 or args.bin_size < 1 or args.chunksize < 1:
        print("[consensus_stability] ERROR: --top-k, --bin and --chunksize must be at least 1.")
        return EXIT_ERROR
    result_dir = resolve_result_dir(args.result)
    if not os.path.isdir(result_dir):
        print(f"[consensus_stability] ERROR: {args.result!r} is not a result folder.")
        return EXIT_ERROR
    models = [m.strip() for m in args.models.split(',') if m.strip()] or None
    try:
        effect = read_result_table(result_dir, 'Marker_effect')
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"[consensus_stability] ERROR: could not read Marker_effect in {result_dir}: {exc}")
        return EXIT_ERROR
    if effect is None or effect.shape[0] == 0:
        print(f"[consensus_stability] Marker_effect not found or empty in {result_dir} - "
              f"nothing to evaluate.")
        return EXIT_NOTHING
    interactions = None
    if args.pairs:
        interactions = _interaction_source(result_dir)
        if interactions is None:
            try:
                interactions = read_result_table(result_dir, 'Interaction')
            except (OSError, ValueError, pd.errors.ParserError) as exc:
                print(f"[consensus_stability] ERROR: could not read Interaction: {exc}")
                return EXIT_ERROR
        if interactions is None or (isinstance(interactions, pd.DataFrame) and interactions.empty):
            print("[consensus_stability] NOTE: --pairs given but the run has no Interaction table; "
                  "markers only.")
            interactions = None
    try:
        report = ss.build_report(effect, interactions, top_k=args.top_k, models=models,
                                 bin_size=args.bin_size, pairs=args.pairs,
                                 chunksize=args.chunksize)
    except ValueError as exc:
        print(f"[consensus_stability] ERROR: {exc}")
        return EXIT_ERROR
    for note in report['notes']:
        print(f"[consensus_stability] NOTE: {note}")
    if report['consensus'].empty:
        print("[consensus_stability] nothing to evaluate.")
        return EXIT_NOTHING
    out_dir = args.out_dir or result_dir
    written = ss.write_report(report, out_dir, frequencies=args.frequencies)
    if args.interaction_frequency:
        source = _interaction_source(result_dir)
        if source is None or report['pair_frequency'].empty and report['marker_frequency'].empty:
            print("[consensus_stability] NOTE: --interaction-frequency needs a single "
                  "Interaction.csv[.gz] with rows; the file was not written.")
        else:
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, ss.INTERACTION_FREQUENCY_FILE)
            try:
                if ss.write_interaction_frequency(source, path, report, args.chunksize):
                    written.append(path)
            except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
                print(f"[consensus_stability] ERROR: could not write the interaction frequency "
                      f"file: {exc}")
                return EXIT_ERROR
    with pd.option_context('display.width', 170, 'display.max_columns', 30,
                           'display.float_format', '{:.3f}'.format):
        print(report['consensus'].to_string(index=False))
    print(f"[consensus_stability] wrote {', '.join(os.path.basename(p) for p in written)} "
          f"in {out_dir}.")
    return EXIT_OK


if __name__ == '__main__':
    sys.exit(main())
