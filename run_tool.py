#!/usr/bin/env python3
"""
EasiGP - headless, non-interactive runner for the "Tool" mode (ver4-54, R6):
cross-run performance comparison of FINISHED runs. It reads existing
`Result/<folder>/` outputs only - it never runs GP(), never imports R or rpy2,
and never writes into a run's own folder; everything goes to
`Result/<RESULT_NAME>/` (the Tool's own output folder).

Workflow
--------
1. In the Streamlit GUI choose "How do you want to run this pipeline?" = Tool,
   open "1. Performance comparison", select two or more results in a section,
   then on "2. Run tool" choose Option A and "Generate and save job files".
   This writes Result/<RESULT_NAME>/tool_config.json (plus its GUI-state
   sidecar) and a Slurm / PBS / NCI submission script.

2. Submit the script, or run this file directly (it is light: 1 CPU, a few GB
   of memory and a few minutes are usually enough):

       python run_tool.py --config Result/<RESULT_NAME>/tool_config.json

Exit codes
----------
0  at least one figure or workbook was written and nothing failed
1  the config is invalid, a section failed, or nothing could be written
2  the file is not a Tool config (RUN_KIND != 'tool') or cannot be read
3  the config requests nothing (no section has a plot or summary ticked)
"""

import argparse
import json
import sys

EXIT_OK, EXIT_FAILED, EXIT_NOT_TOOL, EXIT_NOTHING_REQUESTED = 0, 1, 2, 3


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--config',
        required=True,
        help='Path to the tool_config.json produced by the GUI ("Generate and save job files").',
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        with open(args.config, 'r', encoding='utf-8-sig') as f:
            cfg = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"[run_tool] ERROR: cannot read {args.config!r} as JSON ({exc}).")
        return EXIT_NOT_TOOL
    if not isinstance(cfg, dict) or cfg.get('RUN_KIND') != 'tool':
        print(
            f"[run_tool] ERROR: {args.config!r} is not a Tool config (RUN_KIND must be 'tool'). "
            "Pipeline configs are run with run_sequential.py, run_step1_batch.py or "
            "run_step2_assemble.py."
        )
        return EXIT_NOT_TOOL

    # Light imports only: no genomic_prediction, no rpy2, no R (I1/I2).
    import matplotlib

    matplotlib.use('Agg')
    import pipeline_utils
    import result_comparison
    from pipeline_utils import TimestampedWriter, make_run_log_path

    # Validate BEFORE creating any folder, so a bad config (e.g. an output
    # folder equal to a finished run's folder) never writes a log into it.
    problems = result_comparison.validate_tool_config(cfg)
    if problems:
        for p in problems:
            print(f"[run_tool] ERROR: {p}")
        return EXIT_FAILED

    log_file_path = make_run_log_path(cfg['RESULT_NAME'], 'tool')
    console = sys.stdout
    try:
        with open(log_file_path, 'w', encoding='utf-8') as log_file:
            sys.stdout = TimestampedWriter(console, log_file)
            print(f'[run_tool] Logging to {log_file_path}')
            print(
                f"[run_tool] engine {pipeline_utils.ENGINE_VERSION} (config exported by "
                f"{cfg.get('ENGINE_VERSION', 'unknown')})."
            )
            result = result_comparison.run_tool(cfg)
    finally:
        sys.stdout = console  # the log file is closed now; later prints go to the console only

    if result['errors']:
        print(f"[run_tool] FAILED with {len(result['errors'])} error(s) - see the lines above.")
        return EXIT_FAILED
    if not result['requested']:
        print(
            "[run_tool] nothing was requested: tick violin plots, the summary workbook, the metric comparison plots "
            "and/or the per-run diagnostics in at least one section."
        )
        return EXIT_NOTHING_REQUESTED
    if not result['written']:
        print(
            "[run_tool] FAILED: every requested section was skipped, so nothing was written "
            f"({result['skipped']})."
        )
        return EXIT_FAILED
    return EXIT_OK


if __name__ == '__main__':
    sys.exit(main())
