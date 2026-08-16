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
import os
import sys
import time

import pandas as pd

from pipeline_utils import configure_r_environment, init_rpy2_conversion, TimestampedWriter, make_run_log_path


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
    args = parse_args()

    with open(args.config, 'r', encoding='utf-8-sig') as f:
        cfg = json.load(f)

    # Wrap stdout so every line this process prints gets a timestamp prefix
    # automatically, AND save it to Result/<RESULT_NAME>/logs/ - the same
    # place the GUI's own local-run option already saves a log (see
    # pipeline_utils.make_run_log_path) - in addition to whatever the job
    # scheduler itself captures to its own job-output file.
    log_file_path = make_run_log_path(cfg['RESULT_NAME'], 'step2')
    log_file = open(log_file_path, 'w', encoding='utf-8')
    sys.stdout = TimestampedWriter(sys.stdout, log_file)
    print(f'[run_step2_assemble] Logging to {log_file_path}')

    configure_r_environment(cfg.get('R_PATH'))
    init_rpy2_conversion()

    # Imported after R/rpy2 setup so R_HOME/PATH are already correct.
    from assemble import assemble, load_assembled
    from metric_plot import metric_plot
    from scatter_plot import scatter_plot
    from circos_plot import circos_plot, _broadcast_population_info, _clean_population_label
    from attention_histogram import attention_distribution
    from models.hyperparameter_tuning import base_of
    from Preprocess.LD_decay_plot import average_and_plot_ld_decay
    from checkpoint_utils import describe_incomplete_batch, format_batch_id_list

    print(f"[run_step2_assemble] RESULT_NAME={cfg['RESULT_NAME']}")

    if cfg.get('SKIP_ASSEMBLE'):
        _t0 = time.time()
        (metrics, predicted_result_train, predicted_result_test, effect,
         interactions, attention, population, phenotype, assembled_model,
         missing_batches, incomplete_batches) = load_assembled(cfg['RESULT_NAME'])
        print(f'[run_step2_assemble] Skipped assemble - reloaded previously assembled results for models '
              f'{assembled_model} (took {time.time() - _t0:.1f}s).')
    else:
        _t0 = time.time()
        (metrics, predicted_result_train, predicted_result_test, effect,
         interactions, attention, population, phenotype, assembled_model,
         missing_batches, incomplete_batches) = assemble(cfg['RESULT_NAME'], expected_batches=cfg.get('EXPECTED_BATCHES'))
        print(f'[run_step2_assemble] Assembled results from all batches for models '
              f'{assembled_model} (took {time.time() - _t0:.1f}s).')
        if missing_batches:
            print(f'[run_step2_assemble] Missing batch ID(s) to re-run from scratch: '
                  f'{format_batch_id_list(missing_batches)}')

    # Requirement: make it easy to notice, at a glance, exactly which batch(es)
    # started but did not finish - printed unconditionally, right after the
    # assemble/load-assembled step (whichever ran), regardless of whether
    # assembly itself succeeded, so this is never buried by later output.
    if incomplete_batches:
        print(f'[run_step2_assemble] {len(incomplete_batches)} batch(es) did NOT finish and were '
              f'excluded from the results above - re-submit these to complete the job:')
        for _b in incomplete_batches:
            print(f'[run_step2_assemble]   - {describe_incomplete_batch(_b)}')

    if not assembled_model:
        raise SystemExit(
            '[run_step2_assemble] No usable batch output was found to assemble - see the '
            'warning above for which batch(es) to re-run, then re-run Step 2.'
        )

    # Requirement 6: once every batch's prediction scenarios have been
    # assembled, average every per-scenario LD decay CSV (written by each
    # Step 1 batch's own GP() call, into this same shared Result folder, if
    # the LD decay plot feature was enabled) into one plot per population x
    # phenotype combination. Safe to call unconditionally - a no-op (with
    # an informational log line) whenever the feature wasn't used. Unlike
    # every other plotting call below, this doesn't depend on `metrics`/
    # `assembled_model` at all - it reads directly from the LD decay data
    # files every batch already wrote to disk.
    _t0 = time.time()
    _n_ld_decay_plots = average_and_plot_ld_decay(cfg['RESULT_NAME'])
    if _n_ld_decay_plots:
        print(f'[run_step2_assemble] {_n_ld_decay_plots} average LD decay plot(s) generated '
              f'(took {time.time() - _t0:.1f}s).')

    if any(base_of(m) in ('GAT_fully_connected', 'GAT_prior_knowledge') or m.startswith('GAT_biological_prior_knowledge') for m in assembled_model):
        _t0 = time.time()
        attention_distribution(attention, cfg['RESULT_NAME'], 10)
        print(f'[run_step2_assemble] Attention distribution plots generated (took {time.time() - _t0:.1f}s).')

    # Ordinarily this is just assembled_model (already the real, possibly
    # tuning-suffixed model names - see assemble.py/load_assembled(), which
    # strip only weighted-ensemble method labels) plus cfg['W_OPT']'s plain
    # method names added back in for metric_plot's benefit. But
    # step2_config.json never carries HP_TUNE (Step 2 doesn't select models
    # itself - it only assembles Step 1's output), so unlike
    # run_sequential.py this can't branch on cfg.get('HP_TUNE') directly.
    # Instead, detect it from the assembled data itself: hyperparameter.csv
    # only exists if HP_TUNE was actually used in Step 1 (see
    # genomic_prediction.py/assemble.py). When it was, a weighted-ensemble
    # method's real label can be per-method-suffixed (e.g.
    # 'Nelder Mead__Grid') rather than the plain cfg['W_OPT'] name - read the
    # real, complete label set directly out of metrics itself in that case,
    # exactly as run_sequential.py now does for the same reason.
    hp_tune_was_used = os.path.isfile(f"./Result/{cfg['RESULT_NAME']}/hyperparameter.csv")
    if hp_tune_was_used:
        model_labels = pd.unique(metrics['model']).tolist()
    else:
        model_labels = assembled_model + cfg['W_OPT'] if cfg['W_OPT'] is not None else assembled_model
    if cfg.get('METRIC_PLOT_CREATE', True):
        _t0 = time.time()
        metric_plot(metrics.copy(), model_labels, cfg['RESULT_NAME'], cfg['SCENARIO'], cfg.get('METRIC_PLOT_CONFIG'))
        print(f'[run_step2_assemble] Metric plots generated (took {time.time() - _t0:.1f}s).')

    if cfg.get('SCATTER_CREATE', True):
        _t0 = time.time()
        scatter_plot(assembled_model, phenotype, predicted_result_test, effect,
                     cfg['QTL'], cfg['SCATTER_CONFIG'], cfg['RESULT_NAME'])
        print(f'[run_step2_assemble] Scatter plot matrix generated (took {time.time() - _t0:.1f}s).')

    if cfg.get('CIRCOS_CREATE', True):
        _t0 = time.time()
        # Requirement (bugfix - the actual root cause of gene/chromosome
        # rings being missing for real populations on headless/HPC runs):
        # this script used to call circos_plot() with the raw
        # cfg['CHROMOSOME_INFO']/cfg['GENE_INFO'] paths directly - it never
        # had the 'Chromosome/gene lengths are the same for every
        # population' broadcast logic main_app.py's own interactive GUI
        # code has, so ticking that checkbox in the GUI never actually
        # took effect for a Parallel Step 2 job submitted this way (the
        # GUI only writes cfg to step2_config.json and hands off to THIS
        # script to do the real work - it doesn't run the broadcast
        # itself). See _broadcast_population_info()'s own docstring
        # (circos_plot.py) for why this logic now lives in one shared
        # place instead of being duplicated (and, as happened here,
        # silently drifting out of sync) between this script and
        # main_app.py.
        _chrom_info_path, _gene_info_path = cfg['CHROMOSOME_INFO'], cfg['GENE_INFO']
        print(f"[run_step2_assemble] Broadcast checkbox: {cfg.get('CIRCOS_BROADCAST_POPULATION')} | "
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
            print(f"[run_step2_assemble] Broadcast target populations: {_target_pops} | "
                  f"broadcast chrom file: {_chrom_info_path} | broadcast gene file: {_gene_info_path}")
        circos_plot(effect, interactions, cfg['MARKER_INFO'], _chrom_info_path,
                    _gene_info_path, population, phenotype, cfg['CIRCOS_CONFIG'],
                    cfg['END_ADJUST'], cfg['WINDOW'], cfg['CYTOBAND_COLORMAP'],
                    cfg['RESULT_NAME'], attention, cfg['SCENARIO'], cfg['ASCENDING'],
                    gene_adjust=cfg.get('GENE_ADJUST', 0))
        print(f'[run_step2_assemble] Circos plot generated (took {time.time() - _t0:.1f}s).')
    print('[run_step2_assemble] Pipeline completed successfully.')


if __name__ == '__main__':
    main()
