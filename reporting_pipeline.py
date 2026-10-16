"""
reporting_pipeline.py
======================
Requirements.md item 3 (multi-option update) - the reporting chain
(summaries + plots) for a run whose result files carry the leading
`asreml_option` column (two or more of BLUEs / BLUPs without GRM /
BLUPs with GRM were selected).

The four report drivers (`run_sequential.py`, `run_step2_assemble.py`, and
the two in-process blocks of `main_app.py`) keep their existing, unchanged
single-option chains (I11 - a one-option run is byte-identical to before)
and hand a multi-option run to `run_multi_option_reports()` below instead.
It reproduces that same chain once PER OPTION, over that option's own
rows (`batch_reader.ResultSet.view(option)` - exactly the frames a single-
option run would have produced, so `metric_plot`, `scatter_plot`,
`circos_plot`, `weight_plot`, `attention_distribution` and the two
summary writers need no change at all), writing into that option's own
folder `Result/<RESULT_NAME>/<option tag>/`; then the plots that compare
options (and the LD-decay average, which is option-independent) once into
`Result/<RESULT_NAME>/`.

Every heavy import is lazy - this module can be imported before the R /
rpy2 environment is configured.
"""

from __future__ import annotations

import os
import plot_style as _plot_style   # ver4-31 R15
import time
from typing import Callable, Optional

import pandas as pd

import asreml_syntax as _syn
import strategy_results as _sr


def is_multi_option_run(cfg: "Optional[dict]") -> bool:
    """True when `cfg` selects more than one genotype-effect option."""
    return _syn.is_multi_strategy((cfg or {}).get('ASREML'))


def _has_attention(models) -> bool:
    return any(
        m in ('GAT_fully_connected', 'GAT_prior_knowledge') or str(m).startswith('GAT_biological_prior_knowledge')
        for m in models
    )


def phase_count(cfg: dict, n_options: int, models) -> int:
    """Number of progress phases `run_multi_option_reports()` will
    announce through its `advance` callback - lets the GUI divide the
    post-run progress bar evenly (`main_app._set_post_phase_count`)."""
    from accuracy_metrics_plot import any_plot_requested
    per_option = (
        (1 if _has_attention(models) else 0)
        + (1 if cfg.get('METRIC_PLOT_CREATE', True) else 0)
        + (1 if cfg.get('WEIGHT_PLOT_CREATE', False) else 0)
        + (1 if cfg.get('SCATTER_CREATE', True) else 0)
        + (1 if cfg.get('CIRCOS_CREATE', True) else 0)
    )
    ld = cfg.get('LD_PRUNE') or {}
    has_ld = bool((ld.get('decay_plot') or {}).get('enabled')) if isinstance(ld, dict) else False
    return per_option * max(n_options, 1) + (1 if has_ld else 0) + (1 if any_plot_requested(cfg) else 0)


def _resolve_dpi(cfg: dict):
    legacy = cfg.get('PLOT_DPI', 300)
    return (int(cfg.get('METRIC_PLOT_DPI', legacy)),
            int(cfg.get('SCATTER_PLOT_DPI', legacy)),
            int(cfg.get('CIRCOS_PLOT_DPI', legacy)))


def run_multi_option_reports(cfg: dict, result, *, log: Callable[[str], None] = print,
                             advance: Optional[Callable[[str], None]] = None,
                             circos_workers: Optional[int] = None) -> None:
    """Run the reporting chain for a multi-option `result` (a
    `batch_reader.ResultSet`, any source).

    Parameters
    ----------
    cfg     : the run config (same keys the single-option chains read).
    result  : ResultSet over the merged result files (leading option column).
    log     : one-line logger (print in the headless runners, the GUI's own
              `_log` in-process).
    advance : optional `advance(message)` progress hook, called once per
              plot phase (the GUI's `_advance_progress`).
    circos_workers : override for `CIRCOS_PLOT_WORKERS` - the in-process GUI
              passes 1 (a `spawn`ed worker would re-import Streamlit's
              bootstrap); the headless runners leave it `None` (= config).
    """
    # Lazy imports: R/rpy2 must be configured before these load.
    from metric_summary import write_metric_summary
    from metric_plot import metric_plot
    from accuracy_metrics_plot import accuracy_metrics_plot, any_plot_requested
    from scatter_plot import scatter_plot
    from circos_plot import circos_plot, _broadcast_population_info, _clean_population_label
    from attention_histogram import attention_distribution
    from Preprocess.LD_decay_plot import average_and_plot_ld_decay
    from diversity_summary import (build_dpt_terms, write_dpt_summary, resolve_ensemble_members,
                                   dpt_model_order)
    from weight_plot import weight_plot
    from pipeline_utils import result_dir_path

    def _adv(msg):
        if advance is not None:
            advance(msg)

    name = cfg['RESULT_NAME']
    options = _sr.canonical_option_order(result.options)
    if not options:
        raise ValueError("run_multi_option_reports() needs a ResultSet with an option column.")
    log(f"Multi-option run - reporting per option: {', '.join(options)}")

    metric_dpi, scatter_dpi, circos_dpi = _resolve_dpi(cfg)
    float32 = bool(cfg.get('PLOT_EFFECT_FLOAT32', True))
    hp_tune_used = os.path.isfile(os.path.join(result_dir_path(name), 'hyperparameter.csv'))

    def _model_labels(view):
        if hp_tune_used or cfg.get('HP_TUNE'):
            return pd.unique(view.metric['model']).tolist()
        w_opt = cfg.get('W_OPT')
        return view.models + w_opt if w_opt is not None else view.models

    # ---- option-independent: LD decay average (reads the shared folder) ---- #
    ld_cfg = cfg.get('LD_PRUNE') or {}
    if isinstance(ld_cfg, dict) and (ld_cfg.get('decay_plot') or {}).get('enabled'):
        _adv('Generating average LD decay plots...')
        _t0 = time.time()
        n = average_and_plot_ld_decay(name)
        log(f'{n} average LD decay plot(s) generated (took {time.time() - _t0:.1f}s).')

    # ---- circos broadcast paths, resolved ONCE (population set is shared) ---- #
    chrom_info_path = cfg.get('CHROMOSOME_INFO')
    gene_info_path = cfg.get('GENE_INFO')
    if cfg.get('CIRCOS_CREATE', True) and cfg.get('CIRCOS_BROADCAST_POPULATION'):
        pop_source = result.population
        if cfg.get('SCENARIO') == 'between':
            pop_source = [p.split('->')[-1] if isinstance(p, str) and '->' in p else p for p in pop_source]
        target_pops = [_clean_population_label(p) for p in pop_source] + ['all']
        chrom_info_path = _broadcast_population_info(chrom_info_path, target_pops, 'chrom')
        if gene_info_path:
            gene_info_path = _broadcast_population_info(gene_info_path, target_pops, 'gene')
        log(f"[circos] Broadcast target populations: {target_pops}")

    # ---- per option ---- #
    for label in options:
        view = result.view(label)
        out_name = _sr.option_result_name(name, label)
        os.makedirs(os.path.join('.', 'Result', out_name), exist_ok=True)
        tag = f'[{label}]'
        model_labels = _model_labels(view)

        _t0 = time.time()
        paths = write_metric_summary(
            view.metric, out_name, create=cfg.get('METRIC_SUMMARY_CREATE', True),
            model_order=model_labels, asreml_varcomp=view.asreml_varcomp(),
            label_style=cfg.get('MODEL_LABEL_STYLE', 'full'))
        if paths:
            log(f"{tag} Metric summary written: {', '.join(paths)} (took {time.time() - _t0:.1f}s).")

        _t0 = time.time()
        ctx_frame = view.asreml_accuracy_ctx()
        tw_frame = view.asreml_test_weights()
        resolver = None
        if ((ctx_frame is not None and ctx_frame.shape[0] != 0)
                or (tw_frame is not None and tw_frame.shape[0] != 0)):
            import asreml_accuracy_metrics as _aam
            resolver = _aam.ctx_resolver_from_frame(ctx_frame, tw_frame)
        weight_frame = view.weight()
        terms = build_dpt_terms(view.prediction('test'), weight_frame, view.metric,
                                sample_weight_resolver=resolver)
        paths = write_dpt_summary(terms, out_name, create=cfg.get('DPT_SUMMARY_CREATE', True),
                                  model_order=dpt_model_order(model_labels),
                                  label_style=cfg.get('MODEL_LABEL_STYLE', 'full'))
        if paths:
            log(f"{tag} Diversity Prediction Theorem summary written: {', '.join(paths)} "
                f"(took {time.time() - _t0:.1f}s).")

        # ver4-99 (R2) / ver4-103 (X03): marker / pair stability across iterations. Off unless
        # STABILITY_SUMMARY_CREATE is true; never fatal (the call logs and returns on any failure).
        from stability_summary import run_stability_report as _run_stability_report
        _run_stability_report(cfg, out_name, lambda: view.effect(), log)

        if _has_attention(view.models):
            _adv(f'{tag} Generating attention distribution plots...')
            _t0 = time.time()
            attention_distribution(view.attention(), out_name, 10)
            log(f'{tag} Attention distribution plots generated (took {time.time() - _t0:.1f}s).')
            view.release('attention')

        if cfg.get('METRIC_PLOT_CREATE', True):
            _adv(f'{tag} Generating metric plots...')
            _t0 = time.time()
            metric_plot(view.metric.copy(), model_labels, out_name, cfg['SCENARIO'],
                        _plot_style.with_extra_formats(cfg.get('METRIC_PLOT_CONFIG'), cfg), PLOT_DPI=metric_dpi)
            log(f'{tag} Metric plots generated (took {time.time() - _t0:.1f}s).')

        if cfg.get('WEIGHT_PLOT_CREATE', False):
            _adv(f'{tag} Generating weight plots...')
            _t0 = time.time()
            members = resolve_ensemble_members(view.metric, view.prediction('test'), weight_frame)
            naive = next((info['models'] for info in members.values() if info['is_naive']), None)
            paths = weight_plot(weight_frame, model_labels, out_name, cfg['SCENARIO'],
                                _plot_style.with_extra_formats(cfg.get('WEIGHT_PLOT_CONFIG'), cfg), PLOT_DPI=cfg.get('WEIGHT_PLOT_DPI', 300),
                                naive_models=naive)
            if paths:
                log(f"{tag} Weight plot(s) generated: {', '.join(paths)} (took {time.time() - _t0:.1f}s).")

        if cfg.get('SCATTER_CREATE', True):
            _adv(f'{tag} Generating scatter plot matrix...')
            _t0 = time.time()
            scatter_plot(view.models, view.phenotype, view.prediction('test'),
                         view.effect(float32=float32), cfg['QTL'], _plot_style.with_extra_formats(cfg['SCATTER_CONFIG'], cfg), out_name,
                         MARKER_INFO=cfg.get('MARKER_INFO'),
                         QTL_WINDOW=cfg.get('QTL_WINDOW', 0.0),
                         QTL_WINDOW_MODE=cfg.get('QTL_WINDOW_MODE', 'all_in_window'),
                         PLOT_DPI=scatter_dpi,
                         COLOR_BY_POPULATION=cfg.get('SCATTER_COLOR_BY_POPULATION', False),
                         NORMALIZE_MARKER_EFFECT=cfg.get('SCATTER_NORMALIZE_MARKER_EFFECT', True))
            log(f'{tag} Scatter plot matrix generated (took {time.time() - _t0:.1f}s).')
            view.release('prediction_test')

        if cfg.get('CIRCOS_CREATE', True):
            _adv(f'{tag} Generating circos plot...')
            _t0 = time.time()
            circos_plot(view.effect(float32=float32), view.interactions_grouped(), cfg['MARKER_INFO'],
                        chrom_info_path, gene_info_path, view.population, view.phenotype,
                        _plot_style.with_extra_formats(cfg['CIRCOS_CONFIG'], cfg), cfg['END_ADJUST'], cfg['WINDOW'], cfg['CYTOBAND_COLORMAP'],
                        out_name, view.attention_grouped(), cfg['SCENARIO'], cfg['ASCENDING'],
                        gene_adjust=cfg.get('GENE_ADJUST', 0), plot_dpi=circos_dpi,
                        interactions_hok=view.interactions_hok_grouped(),
                        interaction_orders=cfg.get('CIRCOS_INTERACTION_ORDERS', (2,)),
                        higher_order_layout=cfg.get('CIRCOS_HIGHER_ORDER_LAYOUT', 'per_order'),
                        n_workers=(cfg.get('CIRCOS_PLOT_WORKERS') if circos_workers is None else circos_workers),
                        # N_JOBS fix: only consulted when n_workers is None
                        # (the headless runners); the in-process GUI's
                        # explicit circos_workers=1 stays serial.
                        n_jobs=cfg.get('N_JOBS'))
            log(f'{tag} Circos plot generated (took {time.time() - _t0:.1f}s).')
        view.release()

    # ---- accuracy-metric plots: per-option families + the cross-option one ---- #
    if any_plot_requested(cfg):
        _adv('Generating accuracy-metric plots...')
        _t0 = time.time()
        all_labels = pd.unique(result.metric['model']).tolist()
        accuracy_metrics_plot(
            result.metric.copy(), all_labels, name, cfg['SCENARIO'],
            cfg.get('ACCURACY_METRICS_PLOT_CONFIG'),
            ACCURACY_BUBBLE_PLOT_CREATE=cfg.get('ACCURACY_BUBBLE_PLOT_CREATE', False),
            prediction_test=result.prediction('test'), cfg=cfg)
        log(f'Accuracy-metric plots generated (took {time.time() - _t0:.1f}s).')
        result.release('prediction_test')

    result.release()
    log('Pipeline completed successfully.')
