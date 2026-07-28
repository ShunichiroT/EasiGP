#!/usr/bin/env python3
"""
EasiGP - headless, non-interactive runner for Step 2 (assemble all Step 1
batches, then generate metric/scatter/circos plots) of the Parallel
workflow. This script is meant to be launched directly by an HPC job
scheduler (Slurm/PBS) as a normal (non-array) job - NOT through Streamlit -
so a single GUI configuration step is all that's ever needed.

Workflow
--------
1. In the Streamlit GUI (streamlit_app.py), choose Parallel > Step 2,
   configure everything, then under 'Run pipeline' generate a script draft,
   edit it if needed, and save it into this project's folder. This also
   writes Result/<RESULT_NAME>/step2_config.json.

2. Submit the saved script, e.g.:

       Slurm:  sbatch EasiGP_<RESULT_NAME>_step2.sh
       PBS:    qsub  EasiGP_<RESULT_NAME>_step2.sh

   The script simply calls:

       python run_step2_assemble.py --config Result/<RESULT_NAME>/step2_config.json

3. You can also run it directly for a local test:

       python run_step2_assemble.py --config Result/<RESULT_NAME>/step2_config.json
"""

import argparse
import json
import sys
import time

from pipeline_utils import configure_r_environment, init_rpy2_conversion, TimestampedWriter


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--config', required=True,
        help='Path to the step2_config.json produced by the GUI ("Generate and save job files").'
    )
    return parser.parse_args()


def main():
    # Wrap stdout so every line this process prints gets a timestamp prefix
    # automatically - the scheduler captures this process's stdout to the
    # job's log file, so this is what makes that file timestamped.
    sys.stdout = TimestampedWriter(sys.stdout)

    args = parse_args()

    with open(args.config, 'r') as f:
        cfg = json.load(f)

    configure_r_environment(cfg.get('R_PATH'))
    init_rpy2_conversion()

    # Imported after R/rpy2 setup so R_HOME/PATH are already correct.
    from assemble import assemble, load_assembled
    from metric_plot import metric_plot
    from scatter_plot import scatter_plot
    from circos_plot import circos_plot
    from attention_histogram import attention_distribution

    print(f"[run_step2_assemble] RESULT_NAME={cfg['RESULT_NAME']}")

    if cfg.get('SKIP_ASSEMBLE'):
        _t0 = time.time()
        (metrics, predicted_result_train, predicted_result_test, effect,
         interactions, attention, population, phenotype, assembled_model) = load_assembled(cfg['RESULT_NAME'])
        print(f'[run_step2_assemble] Skipped assemble - reloaded previously assembled results for models '
              f'{assembled_model} (took {time.time() - _t0:.1f}s).')
    else:
        _t0 = time.time()
        (metrics, predicted_result_train, predicted_result_test, effect,
         interactions, attention, population, phenotype, assembled_model) = assemble(cfg['RESULT_NAME'])
        print(f'[run_step2_assemble] Assembled results from all batches for models '
              f'{assembled_model} (took {time.time() - _t0:.1f}s).')

    if 'GAT_fully_connected' in assembled_model or 'GAT_prior_knowledge' in assembled_model:
        _t0 = time.time()
        attention_distribution(attention, cfg['RESULT_NAME'], 10)
        print(f'[run_step2_assemble] Attention distribution plots generated (took {time.time() - _t0:.1f}s).')

    model_labels = assembled_model + cfg['W_OPT'] if cfg['W_OPT'] is not None else assembled_model
    _t0 = time.time()
    metric_plot(metrics.copy(), model_labels, cfg['RESULT_NAME'], cfg['SCENARIO'])
    print(f'[run_step2_assemble] Metric plots generated (took {time.time() - _t0:.1f}s).')

    if cfg['SCATTER_CREATE']:
        _t0 = time.time()
        scatter_plot(assembled_model, phenotype, predicted_result_test, effect,
                     cfg['QTL'], cfg['SCATTER_CONFIG'], cfg['RESULT_NAME'])
        print(f'[run_step2_assemble] Scatter plot matrix generated (took {time.time() - _t0:.1f}s).')

    _t0 = time.time()
    circos_plot(effect, interactions, cfg['MARKER_INFO'], cfg['CHROMOSOME_INFO'],
                cfg['GENE_INFO'], population, phenotype, cfg['CIRCOS_CONFIG'],
                cfg['END_ADJUST'], cfg['WINDOW'], cfg['CYTOBAND_COLORMAP'],
                cfg['RESULT_NAME'], attention, cfg['SCENARIO'], cfg['ASCENDING'])
    print(f'[run_step2_assemble] Circos plot generated (took {time.time() - _t0:.1f}s).')
    print('[run_step2_assemble] Pipeline completed successfully.')


if __name__ == '__main__':
    main()
