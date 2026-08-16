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

import pandas as pd

from pipeline_utils import (
    configure_r_environment, init_rpy2_conversion, restore_ratio,
    TimestampedWriter, make_run_log_path,
)


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
    args = parse_args()

    with open(args.config, 'r', encoding='utf-8-sig') as f:
        cfg = json.load(f)

    # Wrap stdout so every line this process prints (from this script, from
    # genomic_prediction.py's own progress output, etc.) gets a timestamp
    # prefix automatically, AND save it to Result/<RESULT_NAME>/logs/ - the
    # same place the GUI's own local-run option already saves a log (see
    # pipeline_utils.make_run_log_path) - in addition to whatever the job
    # scheduler itself captures to its own job-output file, so the log
    # travels with this run's other output either way.
    log_file_path = make_run_log_path(cfg['RESULT_NAME'], 'sequential')
    log_file = open(log_file_path, 'w', encoding='utf-8')
    sys.stdout = TimestampedWriter(sys.stdout, log_file)
    print(f'[run_sequential] Logging to {log_file_path}')

    configure_r_environment(cfg.get('R_PATH'))
    init_rpy2_conversion()

    # Imported after R/rpy2 setup so R_HOME/PATH are already correct.
    from genomic_prediction import GP
    from metric_plot import metric_plot
    from scatter_plot import scatter_plot
    from circos_plot import circos_plot, _broadcast_population_info, _clean_population_label
    from attention_histogram import attention_distribution
    from Preprocess.LD_decay_plot import average_and_plot_ld_decay

    ratio = restore_ratio(cfg['RATIO'], cfg['SCENARIO'])

    print(f"[run_sequential] RESULT_NAME={cfg['RESULT_NAME']} models={cfg['MODEL']}")

    _t0 = time.time()
    (metrics, predicted_result_train, predicted_result_test, effect,
     interactions, population, phenotype, attention) = GP(
        cfg['GENOTYPE_FILE_NAME'], cfg['PHENOTYPE_FILE_NAME'], cfg['MODEL'],
        cfg['PHENOTYPE'], ratio, cfg['ITER_NUM'], cfg['HPARAMETERS'],
        cfg['R_PATH'], cfg['W_OPT'], cfg['RESULT_NAME'], cfg['HYPERPARAMETERS_OPT'],
        cfg['SCENARIO'], LD_prune=cfg.get('LD_PRUNE'), RF_filter=cfg.get('RF_FILTER'),
        GENOTYPE_FORMAT=cfg.get('GENOTYPE_FORMAT', 'csv'), GENOTYPE_PLINK_PATH=cfg.get('GENOTYPE_PLINK_PATH', 'plink2'),
        OTHER_MODELS_MARKER_SOURCE=cfg.get('OTHER_MODELS_MARKER_SOURCE', 'full_or_filtered'),
        HP_TUNE=cfg.get('HP_TUNE'), HP_TUNE_ENSEMBLE_MODE=cfg.get('HP_TUNE_ENSEMBLE_MODE', 'per_method'),
        MIN_DATA_POINTS=cfg.get('MIN_DATA_POINTS', 100),
    )
    print(f'[run_sequential] Genomic prediction finished (took {time.time() - _t0:.1f}s). '
          f'See the [GP] lines above for LD-pruning and per-model detail.')

    # Requirement 6: once every prediction scenario has finished, average
    # every per-scenario LD decay CSV (written above, inside GP(), if the
    # LD decay plot feature was enabled) into one plot per population x
    # phenotype combination. Safe to call unconditionally - a no-op (with
    # an informational log line) whenever the feature wasn't used.
    _t0 = time.time()
    _n_ld_decay_plots = average_and_plot_ld_decay(cfg['RESULT_NAME'])
    if _n_ld_decay_plots:
        print(f'[run_sequential] {_n_ld_decay_plots} average LD decay plot(s) generated '
              f'(took {time.time() - _t0:.1f}s).')

    if 'GAT_fully_connected' in cfg['MODEL'] or 'GAT_prior_knowledge' in cfg['MODEL'] or any(m.startswith('GAT_biological_prior_knowledge') for m in cfg['MODEL']):
        _t0 = time.time()
        attention_distribution(attention, cfg['RESULT_NAME'], 10)
        print(f'[run_sequential] Attention distribution plots generated (took {time.time() - _t0:.1f}s).')

    # model_labels is what metric_plot() groups/filters by. Ordinarily this
    # is just every selected model name plus every weighted-ensemble method
    # name (cfg['MODEL'] never gets rewritten with tuning-algorithm suffixes -
    # that expansion only ever happens inside GP() itself). But when HP_TUNE
    # is in use, the 'model' column metrics actually contains can include
    # suffixed variants (e.g. 'RF__Grid', 'RF__Bayesian') and/or per-method
    # ensemble labels (e.g. 'ensemble__Grid', 'Nelder Mead__Bayesian') that
    # this reconstruction from cfg alone can't predict - so in that case,
    # read the real label set directly out of metrics itself (the actual
    # source of truth for what GP() produced) instead of guessing.
    if cfg.get('HP_TUNE'):
        model_labels = pd.unique(metrics['model']).tolist()
    else:
        model_labels = cfg['MODEL'] + cfg['W_OPT'] if cfg['W_OPT'] is not None else cfg['MODEL']
    if cfg.get('METRIC_PLOT_CREATE', True):
        _t0 = time.time()
        metric_plot(metrics.copy(), model_labels, cfg['RESULT_NAME'], cfg['SCENARIO'], cfg.get('METRIC_PLOT_CONFIG'))
        print(f'[run_sequential] Metric plots generated (took {time.time() - _t0:.1f}s).')

    if cfg.get('SCATTER_CREATE', True):
        _t0 = time.time()
        # scatter_plot() selects columns out of predicted_result_test/effect
        # by name (see models/ensemble.py-style 'model_selected' pattern),
        # so it needs the same real, possibly-suffixed names as
        # model_labels above - NOT cfg['MODEL'] (never suffixed) - whenever
        # HP_TUNE is in play. Unlike metric_plot, scatter_plot was never
        # given weighted-ensemble (W_OPT) method labels even before this
        # change, so those are filtered back out here to match that existing
        # scope exactly.
        if cfg.get('HP_TUNE'):
            _wopt_prefixes = ('Linear transformation', 'Nelder Mead', 'Bayesian optimisation')
            scatter_models = [m for m in model_labels
                               if not any(m == p or m.startswith(p + '__') for p in _wopt_prefixes)]
        else:
            scatter_models = cfg['MODEL']
        scatter_plot(scatter_models, phenotype, predicted_result_test, effect,
                     cfg['QTL'], cfg['SCATTER_CONFIG'], cfg['RESULT_NAME'])
        print(f'[run_sequential] Scatter plot matrix generated (took {time.time() - _t0:.1f}s).')

    if cfg.get('CIRCOS_CREATE', True):
        _t0 = time.time()
        # Requirement (bugfix - see run_step2_assemble.py's identical
        # comment): this script never had the 'Chromosome/gene lengths
        # are the same for every population' broadcast logic
        # main_app.py's own interactive GUI code has - ticking that
        # checkbox in the GUI never actually took effect for a
        # Sequential job submitted this way.
        _chrom_info_path, _gene_info_path = cfg['CHROMOSOME_INFO'], cfg['GENE_INFO']
        print(f"[run_sequential] Broadcast checkbox: {cfg.get('CIRCOS_BROADCAST_POPULATION')} | "
              f"population from results: {list(population)} | gene info path set: {bool(_gene_info_path)}")
        if cfg.get('CIRCOS_BROADCAST_POPULATION'):
            _target_pop_source = population
            if cfg['SCENARIO'] == 'between':
                _target_pop_source = [
                    p.split('->')[-1] if isinstance(p, str) and '->' in p else p
                    for p in population
                ]
            _target_pops = [_clean_population_label(p) for p in _target_pop_source] + ['all']
            _chrom_info_path = _broadcast_population_info(_chrom_info_path, _target_pops, 'chrom')
            if _gene_info_path:
                _gene_info_path = _broadcast_population_info(_gene_info_path, _target_pops, 'gene')
            print(f"[run_sequential] Broadcast target populations: {_target_pops} | "
                  f"broadcast chrom file: {_chrom_info_path} | broadcast gene file: {_gene_info_path}")
        circos_plot(effect, interactions, cfg['MARKER_INFO'], _chrom_info_path,
                    _gene_info_path, population, phenotype, cfg['CIRCOS_CONFIG'],
                    cfg['END_ADJUST'], cfg['WINDOW'], cfg['CYTOBAND_COLORMAP'],
                    cfg['RESULT_NAME'], attention, cfg['SCENARIO'], cfg['ASCENDING'],
                    gene_adjust=cfg.get('GENE_ADJUST', 0))
        print(f'[run_sequential] Circos plot generated (took {time.time() - _t0:.1f}s).')
    print('[run_sequential] Pipeline completed successfully.')


if __name__ == '__main__':
    main()
