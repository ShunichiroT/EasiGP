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

from pipeline_utils import (
    configure_r_environment, init_rpy2_conversion, TimestampedWriter, make_run_log_path,
    result_dir_path, apply_scratch_tmp_dir,
)


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

    # PATCH_NOTES (worker-crash fix): see pipeline_utils.apply_scratch_tmp_dir()'s
    # own docstring - a no-op unless EASIGP_SCRATCH_TMPDIR (or this config's
    # own 'SCRATCH_TMP_DIR' key) is set.
    apply_scratch_tmp_dir(result_name=cfg['RESULT_NAME'], explicit=cfg.get('SCRATCH_TMP_DIR'))

    configure_r_environment(cfg.get('R_PATH'), r_max_ppsize=cfg.get('R_MAX_PPSIZE', 500000))
    init_rpy2_conversion()

    # Imported after R/rpy2 setup so R_HOME/PATH are already correct.
    from assemble import load_for_step2
    from metric_summary import write_metric_summary
    from metric_plot import metric_plot
    from scatter_plot import scatter_plot
    from circos_plot import circos_plot, _broadcast_population_info, _clean_population_label
    from attention_histogram import attention_distribution
    from models.hyperparameter_tuning import base_of
    from Preprocess.LD_decay_plot import average_and_plot_ld_decay
    from checkpoint_utils import describe_incomplete_batch, format_batch_id_list
    # Update ID ver4-9, R5.
    from diversity_summary import build_dpt_terms, write_dpt_summary, resolve_ensemble_members, dpt_model_order
    # Update ID ver4-9, R6.
    from weight_plot import weight_plot

    print(f"[run_step2_assemble] RESULT_NAME={cfg['RESULT_NAME']}")

    # ver4-4 R7.g/R7.h: resolved once, here, and reused at every plotting
    # call below - PLOT_EFFECT_FLOAT32 (default True - halves the widest
    # frame in the plotting path; Marker_effect.csv itself is unaffected,
    # see batch_reader.ResultSet.effect()'s own docstring) and PLOT_DPI.
    # Logged unconditionally so a person comparing output across
    # runs/versions can see what was actually used without having to
    # read the config file.
    #
    # R7.h follow-up: DPI is now THREE INDEPENDENT config keys - one per
    # plot section (METRIC_PLOT_DPI/SCATTER_PLOT_DPI/CIRCOS_PLOT_DPI) -
    # rather than one shared 'PLOT_DPI' - see run_sequential.py's
    # identical note. Each falls back to the legacy shared 'PLOT_DPI'
    # (then to 300) so a *_config.json written before this split still
    # reproduces its own exact behaviour unchanged.
    _plot_effect_float32 = bool(cfg.get('PLOT_EFFECT_FLOAT32', True))
    _legacy_plot_dpi = cfg.get('PLOT_DPI', 300)
    _metric_plot_dpi = int(cfg.get('METRIC_PLOT_DPI', _legacy_plot_dpi))
    _scatter_plot_dpi = int(cfg.get('SCATTER_PLOT_DPI', _legacy_plot_dpi))
    _circos_plot_dpi = int(cfg.get('CIRCOS_PLOT_DPI', _legacy_plot_dpi))
    print(f"[run_step2_assemble] PLOT_EFFECT_FLOAT32={_plot_effect_float32}, "
          f"METRIC_PLOT_DPI={_metric_plot_dpi}, SCATTER_PLOT_DPI={_scatter_plot_dpi}, "
          f"CIRCOS_PLOT_DPI={_circos_plot_dpi}")

    # Patch 3, Requirement 1: the single shared entry point that resolves
    # cfg['ASSEMBLE_MODE'] ('assemble' / 'use_preassembled' / 'no_assemble')
    # into the right assemble()/load_assembled() call - see
    # assemble.load_for_step2()'s own docstring for the full decision
    # table, including the legacy SKIP_ASSEMBLE-only fallback for any
    # config written before ASSEMBLE_MODE existed. main_app.py's own
    # in-process Step 2 block calls the exact same function, so the two
    # execution paths can never disagree about what a given config means.
    result, _assemble_message = load_for_step2(cfg['RESULT_NAME'], cfg)
    print(f'[run_step2_assemble] {_assemble_message}')
    if getattr(result, 'missing_batches', None):
        print(f'[run_step2_assemble] Missing batch ID(s) to re-run from scratch: '
              f'{format_batch_id_list(result.missing_batches)}')

    # Requirement: make it easy to notice, at a glance, exactly which batch(es)
    # started but did not finish - printed unconditionally, right after the
    # assemble/load-assembled step (whichever ran), regardless of whether
    # assembly itself succeeded, so this is never buried by later output.
    if result.incomplete_batches:
        print(f'[run_step2_assemble] {len(result.incomplete_batches)} batch(es) did NOT finish and were '
              f'excluded from the results above - re-submit these to complete the job:')
        for _b in result.incomplete_batches:
            print(f'[run_step2_assemble]   - {describe_incomplete_batch(_b)}')

    if not result.models:
        raise SystemExit(
            '[run_step2_assemble] No usable batch output was found to assemble - see the '
            'warning above for which batch(es) to re-run, then re-run Step 2.'
        )

    # Update ID 3, R3: the Metric_summary.xlsx workbook (or its CSV
    # degrade), written immediately once the results are available and
    # before the first plot - same position in every one of R3's four call
    # sites (blueprint §R3.9).
    #
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
    #
    # Requirements.md item 8: computed here (moved up from just before
    # metric_plot() below) and passed into write_metric_summary() too, so
    # the metric summary's own model column order always matches the
    # violin plots' own model order exactly.
    hp_tune_was_used = os.path.isfile(os.path.join(result_dir_path(cfg['RESULT_NAME']), 'hyperparameter.csv'))
    if hp_tune_was_used:
        model_labels = pd.unique(result.metric['model']).tolist()
    else:
        model_labels = result.models + cfg['W_OPT'] if cfg['W_OPT'] is not None else result.models

    _t0 = time.time()
    _summary_paths = write_metric_summary(
        result.metric, cfg['RESULT_NAME'], create=cfg.get('METRIC_SUMMARY_CREATE', True),
        model_order=model_labels,
    )
    if _summary_paths:
        print(f"[run_step2_assemble] Metric summary written: {', '.join(_summary_paths)} "
              f"(took {time.time() - _t0:.1f}s).")

    # Update ID ver4-9, R5: the Diversity Prediction Theorem summary,
    # written right after the metric summary (same position as every
    # other R3/R5 call site), reusing the already-assembled `result`
    # (source='combined' or 'batches', whichever load_for_step2() chose)
    # rather than re-reading anything from a hardcoded path.
    _t0 = time.time()
    _dpt_terms = build_dpt_terms(result.prediction('test'), result.weight(), result.metric)
    _dpt_paths = write_dpt_summary(
        _dpt_terms, cfg['RESULT_NAME'], create=cfg.get('DPT_SUMMARY_CREATE', True),
        # Requirements.md item 1: ensemble labels only - see
        # diversity_summary.dpt_model_order()'s own docstring for why the
        # full model_labels list (as used for the metric summary/violin
        # plots) would otherwise reintroduce a permanently-blank column
        # per single-prediction model.
        model_order=dpt_model_order(model_labels),
    )
    if _dpt_paths:
        print(f"[run_step2_assemble] Diversity Prediction Theorem summary written: "
              f"{', '.join(_dpt_paths)} (took {time.time() - _t0:.1f}s).")

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

    if any(base_of(m) in ('GAT_fully_connected', 'GAT_prior_knowledge') or m.startswith('GAT_biological_prior_knowledge') for m in result.models):
        _t0 = time.time()
        attention_distribution(result.attention(), cfg['RESULT_NAME'], 10)
        print(f'[run_step2_assemble] Attention distribution plots generated (took {time.time() - _t0:.1f}s).')
        # Update ID 3, R1: free the cached attention frame before the next
        # stage - it's read fresh again, if needed, by circos_plot() below
        # (R1.4's fix for allocation A3: the Step-2 peak is one stage's
        # working set, not the sum of every stage's).
        result.release('attention')

    if cfg.get('METRIC_PLOT_CREATE', True):
        _t0 = time.time()
        metric_plot(result.metric.copy(), model_labels, cfg['RESULT_NAME'], cfg['SCENARIO'], cfg.get('METRIC_PLOT_CONFIG'), PLOT_DPI=_metric_plot_dpi)
        print(f'[run_step2_assemble] Metric plots generated (took {time.time() - _t0:.1f}s).')

    # Update ID ver4-9, R6: see run_sequential.py's own identical block for
    # the default-False rationale (I11) and naive_models resolution.
    if cfg.get('WEIGHT_PLOT_CREATE', False):
        _t0 = time.time()
        _weight_for_plot = result.weight()
        _members_for_plot = resolve_ensemble_members(result.metric, result.prediction('test'), _weight_for_plot)
        _naive_for_plot = next((info['models'] for info in _members_for_plot.values() if info['is_naive']), None)
        _weight_plot_paths = weight_plot(
            _weight_for_plot, model_labels, cfg['RESULT_NAME'], cfg['SCENARIO'],
            cfg.get('WEIGHT_PLOT_CONFIG'), PLOT_DPI=cfg.get('WEIGHT_PLOT_DPI', 300),
            naive_models=_naive_for_plot,
        )
        if _weight_plot_paths:
            print(f"[run_step2_assemble] Weight plot(s) generated: {', '.join(_weight_plot_paths)} "
                  f"(took {time.time() - _t0:.1f}s).")

    if cfg.get('SCATTER_CREATE', True):
        _t0 = time.time()
        # ver4-4 R6: see run_sequential.py's identical note - cfg.get(...)
        # defaults reproduce a pre-ver4-4 step2_config.json's exact
        # behaviour (exact-name matching only, or no QTL highlighting at
        # all if QTL is unset).
        scatter_plot(result.models, result.phenotype, result.prediction('test'), result.effect(float32=_plot_effect_float32),
                     cfg['QTL'], cfg['SCATTER_CONFIG'], cfg['RESULT_NAME'],
                     MARKER_INFO=cfg.get('MARKER_INFO'),
                     QTL_WINDOW=cfg.get('QTL_WINDOW', 0.0),
                     QTL_WINDOW_MODE=cfg.get('QTL_WINDOW_MODE', 'all_in_window'),
                     PLOT_DPI=_scatter_plot_dpi)
        print(f'[run_step2_assemble] Scatter plot matrix generated (took {time.time() - _t0:.1f}s).')
        # Update ID 3, R1: the test-prediction frame isn't needed again;
        # `effect` IS (circos_plot below), so it's kept resident.
        result.release('prediction_test')

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
              f"population from results: {list(result.population)} | gene info path set: {bool(_gene_info_path)}")
        if cfg.get('CIRCOS_BROADCAST_POPULATION'):
            _target_pop_source = result.population
            if cfg['SCENARIO'] == 'between':
                _target_pop_source = [
                    p.split('->')[-1] if isinstance(p, str) and '->' in p else p
                    for p in result.population
                ]
            _target_pops = [_clean_population_label(p) for p in _target_pop_source] + ['all']
            _chrom_info_path = _broadcast_population_info(_chrom_info_path, _target_pops, 'chrom')
            if _gene_info_path:
                _gene_info_path = _broadcast_population_info(_gene_info_path, _target_pops, 'gene')
            print(f"[run_step2_assemble] Broadcast target populations: {_target_pops} | "
                  f"broadcast chrom file: {_chrom_info_path} | broadcast gene file: {_gene_info_path}")
        # OOM fix (large Interaction.csv/Attention.csv files): circos_plot()'s
        # interaction()/attention branches have always reduced these two
        # tables down to one row per (population, model, phenotype,
        # marker1, marker2) combination before doing anything else with
        # them (a `.groupby(...).mean()` over every raw row). Reading the
        # FULL raw tables via result.interactions()/result.attention() just
        # to immediately throw away everything but that mean is what was
        # exhausting memory on a large HPC array job's combined files - so
        # the circos-plot call specifically uses the memory-bounded,
        # already-collapsed equivalents instead (see batch_reader.
        # aggregate_marker_pair_sums()/ResultSet.interactions_grouped()/
        # attention_grouped() for why this produces IDENTICAL numbers, not
        # an approximation). attention_distribution() above still uses the
        # raw result.attention() - it genuinely needs the individual
        # values for its histogram, unlike this call.
        circos_plot(result.effect(float32=_plot_effect_float32), result.interactions_grouped(), cfg['MARKER_INFO'], _chrom_info_path,
                    _gene_info_path, result.population, result.phenotype, cfg['CIRCOS_CONFIG'],
                    cfg['END_ADJUST'], cfg['WINDOW'], cfg['CYTOBAND_COLORMAP'],
                    cfg['RESULT_NAME'], result.attention_grouped(), cfg['SCENARIO'], cfg['ASCENDING'],
                    gene_adjust=cfg.get('GENE_ADJUST', 0), plot_dpi=_circos_plot_dpi,
                    # Performance (multi-CPU circos-plot rendering): see
                    # run_sequential.py's own identical call for the
                    # rationale - same config key, same fully-serial
                    # default for every config that predates it.
                    n_workers=cfg.get('CIRCOS_PLOT_WORKERS', 1))
        print(f'[run_step2_assemble] Circos plot generated (took {time.time() - _t0:.1f}s).')
        result.release()
    print('[run_step2_assemble] Pipeline completed successfully.')


if __name__ == '__main__':
    main()
