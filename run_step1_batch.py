#!/usr/bin/env python3
"""
EasiGP - headless, non-interactive runner for a single Step 1 (parallel
model-fitting) batch. This script is meant to be launched directly by an
HPC job scheduler (Slurm/PBS) as the command for each task in an array job -
NOT through Streamlit - so that submitting an array of thousands of jobs
never requires configuring the GUI more than once.

Typical workflow
-----------------
1. In the Streamlit GUI (streamlit_app.py), go to Parallel > Step 1,
   configure everything as usual (data paths, models, hyperparameters,
   batch size, total number of batches, etc.), then click
   "Save configuration for HPC array job". This writes a JSON file
   (by default: Result/<RESULT_NAME>/step1_config.json) capturing every
   setting EXCEPT the batch ID, which is different for every task and is
   therefore resolved at run time instead (see below). The GUI also
   generates ready-to-submit slurm_step1_array.sh / pbs_step1_array.sh
   scripts for you.

2. Submit the array job, e.g.:

       Slurm:  sbatch slurm_step1_array.sh
       PBS:    qsub  pbs_step1_array.sh

   Each task in the array automatically calls:

       python run_step1_batch.py --config Result/<RESULT_NAME>/step1_config.json

   and this script resolves its own batch ID from whichever scheduler
   environment variable is set (SLURM_ARRAY_TASK_ID, or
   PBS_ARRAY_INDEX / PBS_ARRAYID) - no manual input, no GUI, per task.

3. You can also run (or re-run) a single batch manually, overriding the
   batch ID explicitly:

       python run_step1_batch.py --config Result/<RESULT_NAME>/step1_config.json --batch-id 0
"""

import argparse
import json
import sys
import time

from pipeline_utils import (
    configure_r_environment, init_rpy2_conversion,
    resolve_batch_id_from_env, restore_ratio, TimestampedWriter,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--config', required=True,
        help='Path to the step1_config.json produced by the GUI ("Generate and save job files").'
    )
    parser.add_argument(
        '--batch-id', type=int, default=None,
        help=(
            'Explicit batch ID for this run. If omitted (the normal case inside an '
            'array job), the batch ID is read automatically from SLURM_ARRAY_TASK_ID '
            'or PBS_ARRAY_INDEX/PBS_ARRAYID.'
        ),
    )
    return parser.parse_args()


def main():
    # Wrap stdout so every line this process prints gets a timestamp prefix
    # automatically - the scheduler captures this process's stdout to the
    # array task's log file, so this is what makes that file timestamped.
    sys.stdout = TimestampedWriter(sys.stdout)

    args = parse_args()

    with open(args.config, 'r') as f:
        cfg = json.load(f)

    batch_id = args.batch_id
    if batch_id is None:
        try:
            batch_id = resolve_batch_id_from_env()
        except RuntimeError as exc:
            raise SystemExit(str(exc))
    batch_size = cfg['PARALLEL']['batch_size']
    parallel = {'batch_id': batch_id, 'batch_size': batch_size}

    configure_r_environment(cfg.get('R_PATH'))
    init_rpy2_conversion()

    # Imported after R/rpy2 setup so R_HOME/PATH are already correct.
    from genomic_prediction import GP

    ratio = restore_ratio(cfg['RATIO'], cfg['W_OPT'])

    print(f"[run_step1_batch] RESULT_NAME={cfg['RESULT_NAME']} "
          f"batch_id={batch_id} batch_size={batch_size} models={cfg['MODEL']}")

    _t0 = time.time()
    GP(
        cfg['GENOTYPE_FILE_NAME'], cfg['PHENOTYPE_FILE_NAME'], cfg['MODEL'],
        cfg['PHENOTYPE'], ratio, cfg['ITER_NUM'], cfg['HPARAMETERS'],
        cfg['R_PATH'], cfg['W_OPT'], cfg['RESULT_NAME'], cfg['HYPERPARAMETERS_OPT'],
        cfg['SCENARIO'], parallel, LD_prune=cfg.get('LD_PRUNE')
    )

    print(f'[run_step1_batch] Batch {batch_id} finished (took {time.time() - _t0:.1f}s). '
          f'See the [GP] lines above for LD-pruning and per-model detail.')


if __name__ == '__main__':
    main()
