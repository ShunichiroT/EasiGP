#!/usr/bin/env python3
"""
EasiGP - headless, non-interactive runner for the full Sequential pipeline
(data setup -> model fitting -> ensemble weighting -> scatter plot ->
circos plot) in a single pass. This script is meant to be launched directly
by an HPC job scheduler (Slurm/PBS) as a normal (non-array) batch job - NOT
through Streamlit - so a single GUI configuration step is all that's ever
needed, with no browser session required to stay open on the cluster.

Workflow
--------
1. In the Streamlit GUI (streamlit_app.py), choose Sequential mode,
   configure everything, then under 'Run pipeline' generate a script draft,
   edit it if needed, and save it into this project's folder. This also
   writes Result/<RESULT_NAME>/sequential_config.json.

2. Submit the saved script, e.g.:

       Slurm:  sbatch EasiGP_<RESULT_NAME>_sequential.sh
       PBS:    qsub  EasiGP_<RESULT_NAME>_sequential.sh

   The script simply calls:

       python run_sequential.py --config Result/<RESULT_NAME>/sequential_config.json

3. You can also run it directly for a local test:

       python run_sequential.py --config Result/<RESULT_NAME>/sequential_config.json
"""

import argparse
import json
import sys
import time

from pipeline_utils import configure_r_environment, init_rpy2_conversion, restore_ratio, TimestampedWriter


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--config', required=True,
        help='Path to the sequential_config.json produced by the GUI ("Generate and save job files").'
    )
    return parser.parse_args()


def main():
    # Wrap stdout so every line this process prints (from this script, from
    # genomic_prediction.py's own progress output, etc.) gets a timestamp
    # prefix automatically - the scheduler captures this process's stdout to
    # the job's log file, so this is what makes that file timestamped.
    sys.stdout = TimestampedWriter(sys.stdout)

    args = parse_args()

    with open(args.config, 'r') as f:
        cfg = json.load(f)

    configure_r_environment(cfg.get('R_PATH'))
    init_rpy2_conversion()

    # Imported after R/rpy2 setup so R_HOME/PATH are already correct.
    from genomic_prediction import GP
    from metric_plot import metric_plot
    from scatter_plot import scatter_plot
    from circos_plot import circos_plot
    from attention_histogram import attention_distribution

    ratio = restore_ratio(cfg['RATIO'], cfg['SCENARIO'])

    print(f"[run_sequential] RESULT_NAME={cfg['RESULT_NAME']} models={cfg['MODEL']}")

    _t0 = time.time()
    (metrics, predicted_result_train, predicted_result_test, effect,
     interactions, population, phenotype, attention) = GP(
        cfg['GENOTYPE_FILE_NAME'], cfg['PHENOTYPE_FILE_NAME'], cfg['MODEL'],
        cfg['PHENOTYPE'], ratio, cfg['ITER_NUM'], cfg['HPARAMETERS'],
        cfg['R_PATH'], cfg['W_OPT'], cfg['RESULT_NAME'], cfg['HYPERPARAMETERS_OPT'],
        cfg['SCENARIO'], LD_prune=cfg.get('LD_PRUNE')
    )
    print(f'[run_sequential] Genomic prediction finished (took {time.time() - _t0:.1f}s). '
          f'See the [GP] lines above for LD-pruning and per-model detail.')

    if 'GAT_fully_connected' in cfg['MODEL'] or 'GAT_prior_knowledge' in cfg['MODEL']:
        _t0 = time.time()
        attention_distribution(attention, cfg['RESULT_NAME'], 10)
        print(f'[run_sequential] Attention distribution plots generated (took {time.time() - _t0:.1f}s).')

    model_labels = cfg['MODEL'] + cfg['W_OPT'] if cfg['W_OPT'] is not None else cfg['MODEL']
    _t0 = time.time()
    metric_plot(metrics.copy(), model_labels, cfg['RESULT_NAME'], cfg['SCENARIO'])
    print(f'[run_sequential] Metric plots generated (took {time.time() - _t0:.1f}s).')

    if cfg['SCATTER_CREATE']:
        _t0 = time.time()
        scatter_plot(cfg['MODEL'], phenotype, predicted_result_test, effect,
                     cfg['QTL'], cfg['SCATTER_CONFIG'], cfg['RESULT_NAME'])
        print(f'[run_sequential] Scatter plot matrix generated (took {time.time() - _t0:.1f}s).')

    _t0 = time.time()
    circos_plot(effect, interactions, cfg['MARKER_INFO'], cfg['CHROMOSOME_INFO'],
                cfg['GENE_INFO'], population, phenotype, cfg['CIRCOS_CONFIG'],
                cfg['END_ADJUST'], cfg['WINDOW'], cfg['CYTOBAND_COLORMAP'],
                cfg['RESULT_NAME'], attention, cfg['SCENARIO'], cfg['ASCENDING'])
    print(f'[run_sequential] Circos plot generated (took {time.time() - _t0:.1f}s).')
    print('[run_sequential] Pipeline completed successfully.')


if __name__ == '__main__':
    main()
