"""
accuracy_metrics_plot.py
==========================
Tab 5 visualisations built on `Metric.csv`'s `record` frame (and, for the
Czekanowski matrix, `Prediction_result_test.csv`), reorganised by
Requirements.md items 4-7 (multi-option update). Every plot family is an
INDEPENDENT subsection with its own create / font / figure-size / DPI
fields (`ACCURACY_METRICS_PLOT_CONFIG[name]`, name in `PLOT_NAMES`):

  violin_accuracy   Adjusted prediction accuracy violin (ASReml adjusted
                    accuracy; produced only when that column exists).
  violin_spearman   Spearman correlation violin (always available - it is
                    computed for every run, ASReml or not).
  coincidence       Top-k Coincidence curve (always available).
  czekanowski       NEW (item 5): per-trait scatter-plot MATRIX comparing
                    models - upper triangle = dots (the two models' test
                    predictions), lower triangle = the Czekanowski
                    coefficient of that model pair at the chosen top-k%
                    threshold. One matrix per trait per selected threshold.
  pairwise_metrics  NEW (item 6): pairwise scatter of metrics (Pearson vs
                    Adjusted Pearson vs Spearman) WITHIN one option.
  across_options    NEW (item 7): each metric compared ACROSS options
                    (e.g. Pearson under BLUEs vs BLUPs without GRM vs BLUPs
                    with GRM); replaces the removed combined
                    "Accuracy metrics" subsection.

Multi-option runs (`Metric.csv` carries the leading `asreml_option`
column): every per-option plot is written into that option's own
sub-folder `Result/<RESULT_NAME>/<option tag>/` from that option's rows
only (so e.g. three Pearson-style violin sets are produced for three
options), and the cross-option plots go into `Result/<RESULT_NAME>/`. A
run without the column behaves exactly as a single-option run: plots land
in `Result/<RESULT_NAME>/` from all rows.

Legacy plot functions (`coincidence_curve_plot`, `pairwise_accuracy_
scatter_plot`, `accuracy_bubble_plot`) keep their signatures; every
function independently checks for the columns it needs and is a silent
no-op when they are absent.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import rankdata as _rankdata

# ver4-25, R4: set_identity_axes() is the shared authority (also used by
# scatter_plot.py, R5) for giving a comparison panel identical x/y limits
# and tick positions plus a corner-to-corner 1:1 line - see that
# function's own docstring in pipeline_utils.py.
from pipeline_utils import canonical_model_order, set_identity_axes
import plot_style as _ps
from asreml_accuracy_metrics import (
    TOPK_PERCENTAGES, TOPK_COLUMNS, ADJUSTED_ACCURACY_COLUMN, SPEARMAN_COLUMN,
)
from metric_plot import metric_plot as _violin_metric_plot

_DEFAULT_SUB_CONFIG = {'font_size': 1, 'fig_size': 5, 'dpi': 300}

# Requirements.md item 8: "six different line types...colourblind
# friendly" - a fixed, cycling set of dash patterns, paired with
# seaborn's own 'colorblind' categorical palette (the SAME palette
# `metric_plot.py`'s violins already use), so overlapping lines/markers
# remain distinguishable without relying on colour alone.
_LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]
_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']


# Keys `_sub_config` never copies through: the two it already returns
# explicitly (`font_size`, `fig_size`), the DPI it returns as the second
# tuple element, and the plot family's own on/off flag.
_SUB_CONFIG_CORE_KEYS = ('font_size', 'fig_size', 'dpi', 'create')


def _sub_config(plot_config, name):
    """`plot_config` is the run's own `ACCURACY_METRICS_PLOT_CONFIG` dict
    (five keys - `violin_accuracy`, `violin_spearman`, `coincidence`,
    `pairwise`, `bubble` - each `{'font_size', 'fig_size', 'dpi'}`, GUI-
    populated on Tab 5). Returns `(config_dict, dpi)` for sub-function
    `name`, degrading to `_DEFAULT_SUB_CONFIG` field-by-field for any
    missing key/sub-dict, so a hand-written/partial config still works.

    ver4-31 (R1/R2/R8/R9/R15): every OTHER key of the sub-config (the new
    violin style knobs, the coincidence baseline / CI / band options, ...)
    is copied through unchanged, plus the run-level `_extra_formats`
    marker `accuracy_metrics_plot()` injects for `PLOT_EXTRA_FORMATS`. A
    sub-config with only the old keys therefore returns exactly the old
    two-key dict."""
    d = (plot_config or {}).get(name)
    if not d and name == 'pairwise_metrics':
        d = (plot_config or {}).get('pairwise')     # pre-multi-option config key
    d = d or {}
    cfg = {
        'font_size': d.get('font_size', _DEFAULT_SUB_CONFIG['font_size']),
        'fig_size': d.get('fig_size', _DEFAULT_SUB_CONFIG['fig_size']),
    }
    for k, v in d.items():
        if k not in _SUB_CONFIG_CORE_KEYS:
            cfg[k] = v
    if (plot_config or {}).get(_ps.EXTRA_FORMATS_KEY):
        cfg[_ps.EXTRA_FORMATS_KEY] = plot_config[_ps.EXTRA_FORMATS_KEY]
    if (plot_config or {}).get(_ps.LABEL_STYLE_KEY):        # ver4-95: MODEL_LABEL_STYLE
        cfg[_ps.LABEL_STYLE_KEY] = plot_config[_ps.LABEL_STYLE_KEY]
    return cfg, d.get('dpi', _DEFAULT_SUB_CONFIG['dpi'])


def _model_style_map(models, all_models=None, mode='legacy'):
    """`{model: {'color', 'linestyle', 'marker'}}` for `models`.

    ver4-31 R0: delegates to the shared `plot_style.model_style_map()`.
    `all_models` is the run's FULL model list (`MODEL`): styles are keyed on
    a model's position in THAT list, so a model has the same colour, line
    style and marker whichever subset a plot draws. With `all_models=None`
    the styles are keyed on `models` itself - the historical behaviour
    (and, for a plot that draws every model, identical to it)."""
    full = list(all_models) if all_models is not None else list(models)
    styles = _ps.model_style_map(full, mode=mode)
    fallback = None
    out = {}
    for m in models:
        if m not in styles:
            if fallback is None:
                fallback = _ps.model_style_map(list(dict.fromkeys(list(full) + list(models))), mode=mode)
            out[m] = fallback[m]
        else:
            out[m] = styles[m]
    return out


def _population_facets(df, SCENARIO, between_label='test'):
    """Requirement 8: "generated per population and across populations,
    as done for the other plots" - mirrors `metric_plot()`'s own
    'between'-scenario collapse (keep only the TEST half of 'X->Y')
    before faceting, so a 'between' run's population axis reads the same
    way here as it already does in the existing violin/scatter plots."""
    df = df.copy()
    # ver4-31 R3: `between_label='pair'` keeps the full 'X->Y' label (one facet
    # per distinct pair); the default 'test' is the historical collapse.
    if SCENARIO == 'between' and 'population' in df.columns and between_label != 'pair':
        df['population'] = df['population'].astype(str).str.rsplit('->', n=1).str[-1]
    return df


# --------------------------------------------------------------------- #
# 3. Top-k Coincidence coincidence curve
# --------------------------------------------------------------------- #

def _coincidence_long(df):
    present = [(k, c) for k, c in zip(TOPK_PERCENTAGES, TOPK_COLUMNS) if c in df.columns]
    if not present:
        return pd.DataFrame(columns=['population', 'phenotype', 'model', 'k', 'percentage'])
    frames = []
    for k, col in present:
        sub = df[['population', 'phenotype', 'model', col]].rename(columns={col: 'percentage'})
        sub = sub.copy()
        sub['k'] = k
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


_COINCIDENCE_CI_MODES = ('z95', 't95', 'sd', 'none')
_COINCIDENCE_BAND_MODES = ('fill', 'errorbar', 'auto')
_COINCIDENCE_Y_MODES = ('percentage', 'enrichment')
_BAND_AUTO_MAX_MODELS = 6         # 'auto' band: errorbar above this many models
_GREYED_COLOUR = '0.75'           # highlight_top: colour of the de-emphasised lines


def _coincidence_opts(config):
    """Validated ver4-31 (R8/R9) coincidence settings from the sub-config.
    Every key is optional; an absent / unknown value yields the legacy
    behaviour (I11)."""
    c = config or {}

    def _pick(key, allowed, default):
        v = str(c.get(key, default)).strip().lower()
        return v if v in allowed else default

    try:
        top = max(int(c.get('highlight_top', 0) or 0), 0)
    except (TypeError, ValueError):
        top = 0
    return {
        'baseline': bool(c.get('baseline', False)),
        'y': _pick('y', _COINCIDENCE_Y_MODES, 'percentage'),
        'ci': _pick('ci', _COINCIDENCE_CI_MODES, 'z95'),
        'band': _pick('band', _COINCIDENCE_BAND_MODES, 'fill'),
        'highlight_top': top,
        'legend_sort': bool(c.get('legend_sort', False)),
        'write_summary': bool(c.get('write_summary', False)),
        'palette': _pick('palette', ('legacy', 'auto', 'group'), 'legacy'),
    }


def _coincidence_agg(sub, ci='z95'):
    """Per-k summary of one model's rows within one facet, in PERCENT units:
    DataFrame indexed by `TOPK_PERCENTAGES` with `mean`, `lower`, `upper`,
    `n`. `ci`: 'z95' = 1.96*SEM (legacy; a single observation gives a zero
    half-width), 't95' = t(0.975, n-1)*SEM (NaN when n < 2), 'sd' = one
    standard deviation, 'none' = no interval (lower = upper = NaN). Bounds
    are clipped to the [0, 100] % range of the metric."""
    agg = sub.groupby('k')['percentage'].agg(['mean', 'std', 'count']).reindex(TOPK_PERCENTAGES)
    n = agg['count'].fillna(0)
    sem = agg['std'] / np.sqrt(agg['count'].clip(lower=1))
    if ci == 'z95':
        half = 1.96 * sem.fillna(0)
    elif ci == 't95':
        from scipy.stats import t as _t
        dof = (n - 1).clip(lower=1)
        tq = pd.Series(_t.ppf(0.975, dof.to_numpy(dtype=float)), index=agg.index)
        half = (tq * sem).where(n >= 2)
    elif ci == 'sd':
        half = agg['std']
    else:
        half = pd.Series(np.nan, index=agg.index)
    out = pd.DataFrame({
        'mean': agg['mean'],
        'lower': (agg['mean'] - half).clip(lower=0),
        'upper': (agg['mean'] + half).clip(upper=100),
        'n': n.astype(int),
    })
    return out


def _direction_word(direction):
    """'Bottom' for a lower-is-better phenotype, else 'Top' (R8)."""
    return 'Bottom' if str(direction).lower() == 'lower' else 'Top'


def _model_auc(long_df, models, enrich=False):
    """`{model: mean over k of the model's mean coincidence (%)}` pooled over
    every facet of `long_df` - the area-under-the-curve ranking used by
    `highlight_top` and the legend (R9). `enrich=True` divides each k's mean
    by k first (the enrichment scale)."""
    out = {}
    for m in models:
        sub = long_df[long_df['model'] == m]
        if sub.empty:
            continue
        per_k = sub.groupby('k')['percentage'].mean()
        if enrich:
            per_k = per_k / per_k.index.to_numpy(dtype=float)
        out[m] = float(per_k.mean()) if len(per_k) else float('nan')
    return out


def _highlight_set(models, auc, top):
    """The models that keep their colour under `highlight_top=top`: the `top`
    best NON-ensemble models by `auc`, plus every naive / weighted ensemble
    label. `None` when highlighting is off (`top` <= 0)."""
    if top <= 0:
        return None
    from pipeline_utils import is_naive_ensemble_label, is_wopt_label
    ens = {m for m in models if is_naive_ensemble_label(m) or is_wopt_label(m)}
    rest = [m for m in models if m not in ens and m in auc and np.isfinite(auc[m])]
    rest.sort(key=lambda m: (-auc[m], models.index(m)))
    return set(rest[:top]) | ens


def _plot_coincidence_ax(ax, facet, models, styles, opts=None, direction='higher',
                          highlight=None, labels=None):
    """Draw one facet of the Top-k Coincidence curve.

    `opts` (from `_coincidence_opts`, `None` = legacy) adds: a chance
    baseline (R8), the enrichment scale (R8), the CI type / band drawing
    (R9) and grey-out highlighting (R9). `labels` maps model -> legend
    text; `direction` selects the 'Top'/'Bottom' axis word. With `opts=None`
    the drawing is the historical one."""
    o = opts or _coincidence_opts(None)
    ks = np.array(TOPK_PERCENTAGES, dtype=float)
    enrich = o['y'] == 'enrichment'
    scale = (1.0 / ks) if enrich else np.ones_like(ks)
    band = o['band']
    if band == 'auto':
        band = 'errorbar' if len(models) > _BAND_AUTO_MAX_MODELS else 'fill'
    n = max(len(models), 1)
    for i, m in enumerate(models):
        sub = facet[facet['model'] == m]
        if sub.empty:
            continue
        agg = _coincidence_agg(sub, o['ci'])
        style = styles[m]
        greyed = highlight is not None and m not in highlight
        mean = np.asarray(agg['mean'], dtype=float) * scale
        lower = np.asarray(agg['lower'], dtype=float) * scale
        upper = np.asarray(agg['upper'], dtype=float) * scale
        label = (labels or {}).get(m, m)
        # A small per-model horizontal jitter (Requirement 8: "be careful
        # with overlaps of dots when there are multiple prediction
        # models") - the six k-values are always the same for every
        # model, so unjittered markers would sit exactly on top of one
        # another. In 'errorbar' mode the LINE keeps the true k and only the
        # error bars are dodged (R9).
        x = ks + (i - (n - 1) / 2.0) * (0.6 / n)
        if greyed:
            ax.plot(ks if band == 'errorbar' else x, mean, label=label, color=_GREYED_COLOUR,
                    linestyle=style['linestyle'], marker=style['marker'], markersize=3,
                    linewidth=0.8, zorder=1.5)
            continue
        if band == 'errorbar':
            ax.plot(ks, mean, label=label, color=style['color'], linestyle=style['linestyle'],
                    marker=style['marker'], markersize=5, linewidth=1.6)
            if o['ci'] != 'none':
                ok = np.isfinite(lower) & np.isfinite(upper) & np.isfinite(mean)
                if ok.any():
                    ax.errorbar(x[ok], mean[ok],
                                yerr=np.vstack([mean[ok] - lower[ok], upper[ok] - mean[ok]]),
                                fmt='none', ecolor=style['color'], elinewidth=0.9,
                                capsize=2, capthick=0.9, alpha=0.8)
        else:
            ax.plot(x, mean, label=label, color=style['color'], linestyle=style['linestyle'],
                    marker=style['marker'], markersize=5, linewidth=1.6)
            if o['ci'] != 'none':
                ax.fill_between(x, lower, upper, color=style['color'], alpha=0.15, linewidth=0)
    if o['baseline']:
        ax.plot(ks, ks * scale, color='0.4', linestyle=(0, (4, 3)),
                linewidth=1.0, label='random selection', zorder=1)
    word = _direction_word(direction)
    ax.set_xlabel('k (%)')
    if enrich:
        ax.set_ylabel('Enrichment over random (×)')
    else:
        ax.set_ylabel(f'{word}-k Coincidence (%)')
    ax.set_xticks(list(TOPK_PERCENTAGES))
    if enrich:
        ax.set_ylim(bottom=0)
    else:
        ax.set_ylim(-2, 102)


def _add_outside_legend(fig, axes_flat, legend_kwargs=None, order=None):
    """Requirement 1: the legend must NEVER overlap the plot area - the
    same guarantee `metric_plot.py`'s violin plots already get for free
    from seaborn's `FacetGrid.add_legend()` (which reserves its own
    margin on the right before drawing). These hand-built (non-FacetGrid)
    figures need the equivalent done manually: the legend is anchored
    just OUTSIDE the figure's own right edge (`bbox_to_anchor=(1.0, 0.5)`
    in figure-fraction coordinates, i.e. never inside any axes' bounding
    box no matter how many rows/columns are faceted), and the figure is
    saved with `bbox_inches='tight'` so the canvas is grown to actually
    include it rather than clipping it - the previous code instead placed
    the legend INSIDE the bottom-right axes' own corner (`loc='lower
    right'` with no reserved space), which is what let legend text sit
    directly on top of data/tick labels whenever that corner wasn't
    empty. Returns the handles found (empty if none), so callers can
    detect "nothing to draw a legend for".

    ver4-31 (R9): `order`, when given, is a list of legend labels; the
    entries are listed in that order (labels not in it follow, in their
    original order).
    """
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        if order:
            ranked = [l for l in order if l in by_label]
            ranked += [l for l in by_label if l not in ranked]
            by_label = {l: by_label[l] for l in ranked}
        fig.legend(by_label.values(), by_label.keys(), loc='center left',
                   bbox_to_anchor=(1.0, 0.5), borderaxespad=0.5,
                   frameon=False, **(legend_kwargs or {}))
    return handles


def _faceted_figure(phenotypes, populations, fig_size, plot_fn, savepath, dpi, legend_kwargs=None,
                     extra_formats=(), legend_order=None, label_map=None):
    n_rows, n_cols = max(len(populations), 1), max(len(phenotypes), 1)
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False,
                              figsize=(fig_size * n_cols, fig_size * n_rows))
    last = None
    for r, pop in enumerate(populations or [None]):
        for c, ph in enumerate(phenotypes or [None]):
            ax = axes[r][c]
            out = plot_fn(ax, pop, ph)
            last = out if out is not None else last
            if r == 0:
                ax.set_title(str(ph))
            if c == 0 and pop is not None:
                ax.set_ylabel(f"{pop}\n{ax.get_ylabel()}")
    _ps.relabel_figure(fig, label_map)           # ver4-95: shortened model text, before the layout
    fig.tight_layout()
    _add_outside_legend(fig, axes.flat, legend_kwargs, order=legend_order)
    _ps.relabel_figure(fig, label_map)           # ... and the legend entries just added
    # bbox_inches='tight' (rather than a hard-coded subplots_adjust
    # fraction) grows the saved canvas to whatever the legend actually
    # measures - correct regardless of how many models/how long their
    # names are, so a run with many selected models can never push the
    # legend back over the rightmost axis the way a fixed margin would.
    _ps.save_figure(fig, savepath, dpi=dpi, bbox_inches='tight', extra_formats=extra_formats)
    plt.close(fig)
    return last


def _coincidence_summary_frame(long_df, models, ci):
    """Rows (phenotype, population, model, k, mean, lower, upper, n) - one
    block per population plus a pooled `population='all'` block - in raw
    percent units (independent of the plotted `y` scale)."""
    rows = []

    def _emit(sub_all, pop_label):
        for ph in pd.unique(sub_all['phenotype']):
            sub_ph = sub_all[sub_all['phenotype'] == ph]
            for m in models:
                sub = sub_ph[sub_ph['model'] == m]
                if sub.empty:
                    continue
                agg = _coincidence_agg(sub, ci)
                for k in TOPK_PERCENTAGES:
                    r = agg.loc[k]
                    rows.append({'phenotype': ph, 'population': pop_label, 'model': m, 'k': k,
                                 'mean': r['mean'], 'lower': r['lower'], 'upper': r['upper'],
                                 'n': int(r['n'])})

    for pop in pd.unique(long_df['population']):
        _emit(long_df[long_df['population'] == pop], pop)
    _emit(long_df, 'all')
    return pd.DataFrame(rows, columns=['phenotype', 'population', 'model', 'k', 'mean',
                                       'lower', 'upper', 'n'])


@_ps.themed('ticks', lambda a: (a['config'] or {}).get('font_size', _DEFAULT_SUB_CONFIG['font_size']), lambda a: a['dpi'])
def coincidence_curve_plot(record, MODEL, RESULT_NAME, SCENARIO, config=None, dpi=300,
                            directions=None):
    """Top-k Coincidence curve (per population and pooled).

    ver4-31 (R8/R9): `config` may carry `baseline`, `y`, `ci`, `band`,
    `highlight_top`, `legend_sort`, `write_summary` (and `palette`) - see
    `_coincidence_opts`; absent keys give the legacy figure. `directions`
    (`{phenotype: 'higher'|'lower'}`, the run's `METRIC_DIRECTIONS`) makes
    the y-label read 'Bottom-k' for a lower-is-better phenotype."""
    long_df = _coincidence_long(_population_facets(
        record, SCENARIO, str((config or {}).get('between_label', 'test')).strip().lower()))
    if long_df.empty:
        return
    full_order = canonical_model_order(MODEL)
    models_present = [m for m in full_order if m in set(long_df['model'])]
    if not models_present:
        return
    opts = _coincidence_opts(config)
    directions = directions or {}
    # R0: styles are keyed on the run's FULL model list, so a model keeps its
    # colour / line style / marker whichever subset a plot draws.
    styles = _model_style_map(models_present, all_models=full_order, mode=opts['palette'])
    fig_size = (config or {}).get('fig_size', _DEFAULT_SUB_CONFIG['fig_size'])
    font_size = (config or {}).get('font_size', _DEFAULT_SUB_CONFIG['font_size'])
    extra_formats = _ps.extra_formats_of(config)
    # ver4-31 R16: the ticks theme is applied by the @_ps.themed decorator (scoped to this call).

    phenotypes = list(pd.unique(long_df['phenotype']))
    populations = list(pd.unique(long_df['population']))

    auc = _model_auc(long_df, models_present)
    highlight = _highlight_set(models_present, auc, opts['highlight_top'])
    labels = None
    legend_order = None
    if opts['legend_sort']:
        shown = _model_auc(long_df, models_present, enrich=True) if opts['y'] == 'enrichment' else auc
        fmt = (lambda v: f"{v:.2f}×") if opts['y'] == 'enrichment' else (lambda v: f"{v:.1f} %")
        labels = {m: (f"{m} (mean {fmt(shown[m])})" if np.isfinite(shown.get(m, float('nan'))) else m)
                  for m in models_present}
        ranked = sorted(models_present,
                        key=lambda m: (-(shown[m] if np.isfinite(shown.get(m, float('nan')))
                                         else float('-inf')), models_present.index(m)))
        legend_order = [labels[m] for m in ranked]

    def _plot_per_pop(ax, pop, ph):
        facet = long_df[(long_df['population'] == pop) & (long_df['phenotype'] == ph)]
        _plot_coincidence_ax(ax, facet, models_present, styles, opts, directions.get(ph, 'higher'),
                             highlight, labels)

    def _plot_pooled(ax, pop, ph):
        facet = long_df[long_df['phenotype'] == ph]
        _plot_coincidence_ax(ax, facet, models_present, styles, opts, directions.get(ph, 'higher'),
                             highlight, labels)

    label_map = _ps.shown_labels(models_present, config)        # ver4-95 (display only)
    _faceted_figure(phenotypes, populations, fig_size, _plot_per_pop,
                     f'./Result/{RESULT_NAME}/Coincidence_curve.png', dpi,
                     extra_formats=extra_formats, legend_order=legend_order, label_map=label_map)
    _faceted_figure(phenotypes, [], fig_size, _plot_pooled,
                     f'./Result/{RESULT_NAME}/Coincidence_curve_total.png', dpi,
                     extra_formats=extra_formats, legend_order=legend_order, label_map=label_map)
    if opts['write_summary']:
        _coincidence_summary_frame(long_df, models_present, opts['ci']).to_csv(
            f'./Result/{RESULT_NAME}/Coincidence_curve_summary.csv', index=False)
    _ps.log_resolved('coincidence', **{k: v for k, v in opts.items()})


# --------------------------------------------------------------------- #
# 4. Pairwise comparison scatter (with confidence ellipses)
# --------------------------------------------------------------------- #

_METRIC_LABELS = ('Pearson correlation', ADJUSTED_ACCURACY_COLUMN, SPEARMAN_COLUMN)
_METRIC_FILE_TAGS = {
    'Pearson correlation': 'Pearson',
    ADJUSTED_ACCURACY_COLUMN: 'AdjustedAccuracy',
    SPEARMAN_COLUMN: 'Spearman',
}


def _available_metric_columns(df):
    """The metrics of `_METRIC_LABELS` present in `df` with at least one
    finite value (adjusted accuracy is absent/all-NaN unless it was
    requested)."""
    out = []
    for col in _METRIC_LABELS:
        if col in df.columns and pd.to_numeric(df[col], errors='coerce').notna().any():
            out.append(col)
    return out


def _metric_pairs(df):
    """Every unordered pair of the available metrics, in the fixed
    Pearson -> Adjusted -> Spearman order, as `(xcol, ycol, file_tag)`.
    Requirements.md item 6: "Pearson, adjusted Pearson (if selected),
    Spearman" - so without the adjusted metric only Pearson-vs-Spearman
    remains."""
    cols = _available_metric_columns(df)
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], f'{_METRIC_FILE_TAGS[cols[i]]}_vs_{_METRIC_FILE_TAGS[cols[j]]}'))
    return pairs


def _plot_pairwise_ax(ax, facet, models, styles, xcol, ycol):
    """ver4-25, R4: one CENTROID point per model - mean `xcol`/`ycol`
    across that model's own replicates/tasks in this facet (NaN-skipping,
    pairwise-complete rows: a row only counts if BOTH columns are
    finite) - with +/-1 standard-error bars on both axes. A model with
    fewer than 2 finite pairs still gets a point, just without error
    bars. `pipeline_utils.set_identity_axes()` then gives the panel
    identical x/y limits and tick positions plus a 1:1 line."""
    x_extent = []
    y_extent = []
    for m in models:
        sub = facet[facet['model'] == m]
        x = pd.to_numeric(sub[xcol], errors='coerce').to_numpy()
        y = pd.to_numeric(sub[ycol], errors='coerce').to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if x.size == 0:
            continue
        style = styles[m]
        x_mean, y_mean = float(np.mean(x)), float(np.mean(y))
        n_eff = x.size
        if n_eff >= 2:
            x_se = float(np.std(x, ddof=1) / np.sqrt(n_eff))
            y_se = float(np.std(y, ddof=1) / np.sqrt(n_eff))
        else:
            x_se = y_se = None
        ax.errorbar(
            x_mean, y_mean, xerr=x_se, yerr=y_se,
            fmt=style['marker'], color=style['color'], capsize=3, elinewidth=1.2,
            markersize=7, label=m,
        )
        x_extent.append(x_mean - (x_se or 0.0))
        x_extent.append(x_mean + (x_se or 0.0))
        y_extent.append(y_mean - (y_se or 0.0))
        y_extent.append(y_mean + (y_se or 0.0))
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    if x_extent and y_extent:
        set_identity_axes(ax, x_extent, y_extent)


def _pairwise_scatter_figures(df, models_present, fig_size, dpi, pairs, out_dir, prefix, extra_formats=(),
                               label_map=None):
    """Shared driver for `pairwise_accuracy_scatter_plot()` (within-option
    metric pairs) and `metric_across_options_plot()` (option pairs of one
    metric): `pairs` is `[(xcol, ycol, file_tag, xlabel, ylabel), ...]`
    over columns of `df`; each pair yields one per-population and one
    pooled ('_total') faceted figure, `<prefix><file_tag>[_total].png`."""
    styles = _model_style_map(models_present)
    phenotypes = list(pd.unique(df['phenotype']))
    populations = list(pd.unique(df['population']))
    for xcol, ycol, fname, xlabel, ylabel in pairs:
        def _draw(ax, facet, xcol=xcol, ycol=ycol, xlabel=xlabel, ylabel=ylabel):
            _plot_pairwise_ax(ax, facet, models_present, styles, xcol, ycol)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        def _plot_per_pop(ax, pop, ph, _draw=_draw):
            _draw(ax, df[(df['population'] == pop) & (df['phenotype'] == ph)])

        def _plot_pooled(ax, pop, ph, _draw=_draw):
            _draw(ax, df[df['phenotype'] == ph])

        _faceted_figure(phenotypes, populations, fig_size, _plot_per_pop,
                         f'{out_dir}/{prefix}{fname}.png', dpi, extra_formats=extra_formats,
                         label_map=label_map)
        _faceted_figure(phenotypes, [], fig_size, _plot_pooled,
                         f'{out_dir}/{prefix}{fname}_total.png', dpi, extra_formats=extra_formats,
                         label_map=label_map)


@_ps.themed('ticks', lambda a: (a['config'] or {}).get('font_size', _DEFAULT_SUB_CONFIG['font_size']), lambda a: a['dpi'])
def pairwise_accuracy_scatter_plot(record, MODEL, RESULT_NAME, SCENARIO, config=None, dpi=300):
    """Requirements.md item 6: pairwise comparison of the metrics (Pearson
    correlation, adjusted Pearson correlation if present, Spearman
    correlation) WITHIN the same option - `record` must already be one
    option's rows (`strategy_results.option_view()`); `RESULT_NAME` is
    that option's output folder name. Files: `Scatter_<x>_vs_<y>[_total].png`.
    A silent no-op with fewer than two available metrics."""
    df = _population_facets(record, SCENARIO)
    pairs = _metric_pairs(df)
    if not pairs:
        return
    models_present = [m for m in canonical_model_order(MODEL) if m in set(df['model'])]
    if not models_present:
        return
    fig_size = (config or {}).get('fig_size', _DEFAULT_SUB_CONFIG['fig_size'])
    font_size = (config or {}).get('font_size', _DEFAULT_SUB_CONFIG['font_size'])
    # ver4-31 R16: the ticks theme is applied by the @_ps.themed decorator (scoped to this call).
    _pairwise_scatter_figures(
        df, models_present, fig_size, dpi,
        [(x, y, tag, x, y) for x, y, tag in pairs],
        f'./Result/{RESULT_NAME}', 'Scatter_', extra_formats=_ps.extra_formats_of(config),
        label_map=_ps.shown_labels(models_present, config))
    return True


@_ps.themed('ticks', lambda a: (a['config'] or {}).get('font_size', _DEFAULT_SUB_CONFIG['font_size']), lambda a: a['dpi'])
def metric_across_options_plot(record, MODEL, RESULT_NAME, SCENARIO, config=None, dpi=300):
    """Requirements.md item 7: each metric compared ACROSS the options of
    a multi-option run, one metric at a time (Pearson correlation among
    BLUEs / BLUPs without GRM / BLUPs with GRM, and likewise adjusted
    Pearson and Spearman when present). `record` is the FULL Metric frame
    (leading option column). For every metric and every option pair, the
    same task (population, phenotype, ratio, sample, model) is matched
    across the two options, and the panel is the familiar centroid +/-1 SE
    scatter with identical axes and a 1:1 line (x = the first option, y =
    the second), so points above the line favour the second option.
    Files: `Across_options_<metric>_<optA>_vs_<optB>[_total].png` in
    `Result/<RESULT_NAME>/`. A silent no-op unless >= 2 options exist."""
    import strategy_results as _sr
    options = _sr.options_in_frame(record)
    if len(options) < 2:
        return False
    df = _population_facets(record, SCENARIO)
    models_present = [m for m in canonical_model_order(MODEL) if m in set(df['model'])]
    if not models_present:
        return False
    fig_size = (config or {}).get('fig_size', _DEFAULT_SUB_CONFIG['fig_size'])
    font_size = (config or {}).get('font_size', _DEFAULT_SUB_CONFIG['font_size'])
    # ver4-31 R16: the ticks theme is applied by the @_ps.themed decorator (scoped to this call).

    key_cols = ['population', 'phenotype', 'model'] + [c for c in ('ratio', 'sample') if c in df.columns]
    wrote = False
    for metric in _available_metric_columns(df):
        sub = df[[_sr.OPTION_COLUMN] + key_cols + [metric]].copy()
        sub[metric] = pd.to_numeric(sub[metric], errors='coerce')
        # ratio can be a tuple/list-like value; a string key is always safe to pivot on
        for c in ('ratio',):
            if c in sub.columns:
                sub[c] = sub[c].astype(str)
        wide = sub.pivot_table(index=key_cols, columns=_sr.OPTION_COLUMN, values=metric,
                               aggfunc='mean', dropna=False).reset_index()
        present_opts = [o for o in options if o in wide.columns]
        pairs = []
        for i in range(len(present_opts)):
            for j in range(i + 1, len(present_opts)):
                a, b = present_opts[i], present_opts[j]
                if not (wide[a].notna() & wide[b].notna()).any():
                    continue
                tag = f'{_METRIC_FILE_TAGS[metric]}_{_sr.option_tag(a)}_vs_{_sr.option_tag(b)}'
                pairs.append((a, b, tag, f'{metric}\n({a})', f'{metric}\n({b})'))
        if pairs:
            _pairwise_scatter_figures(wide, models_present, fig_size, dpi, pairs,
                                       f'./Result/{RESULT_NAME}', 'Across_options_',
                                       extra_formats=_ps.extra_formats_of(config),
                                       label_map=_ps.shown_labels(models_present, config))
            wrote = True
    return wrote


def _unique_file_tags(labels):
    """`{label: filesystem-safe tag}` for `labels`, unique even when two different
    labels sanitise to the same text (a numeric suffix is appended to the later one)."""
    tags, used = {}, set()
    for label in labels:
        base = _safe_filename(label)
        tag, n = base, 2
        while tag in used:
            tag = f'{base}_{n}'
            n += 1
        used.add(tag)
        tags[label] = tag
    return tags


@_ps.themed('ticks', lambda a: (a['config'] or {}).get('font_size', _DEFAULT_SUB_CONFIG['font_size']), lambda a: a['dpi'])
def metric_across_results_plot(record, MODEL, out_dir, SCENARIO, results, metrics=None,
                               config=None, dpi=300, prefix='Across_results_', log=print):
    """ver4-60 (Tool mode): each metric compared BETWEEN finished results (for
    example runs kept in different `Result/` sub-folders) - the cross-run twin of
    `metric_across_options_plot()`, drawn with the very same panel code
    (`_pairwise_scatter_figures` -> `_plot_pairwise_ax` -> `set_identity_axes`).

    `record` is the stacked Metric frame `result_comparison.build_comparison_record()`
    returns: every source's rows plus a `result` column holding that source's
    (unique) display name. `results` is the ordered list of names to compare (the
    selection order; x = the earlier, y = the later one, so points ABOVE the 1:1 line
    favour the later result). For every metric of `metrics` (default: Pearson
    correlation, adjusted prediction accuracy, Spearman correlation - the same three
    Tab 5 offers; MSE is not on a comparable scale) and every pair of results, the
    same task - (population, phenotype, model, ratio, sample) - is matched across the
    two results (`sample` is the train/test-split seed, so a matched pair is the same
    split of the same individuals) and the panel shows one centroid per model with
    +/-1 SE bars on both axes. Unmatched tasks are ignored (pairwise-complete).

    Files, in `out_dir`: `<prefix><metric>_<resultA>_vs_<resultB>[_total].png`
    (+ svg/pdf when the config carries `_extra_formats`). Returns the paths written
    (`[]` when nothing could be drawn - fewer than two results with rows, or no task
    shared by any pair; the reasons are logged through `log`).

    The existing Tab-5 function is deliberately untouched (I11): it keys on the
    leading `asreml_option` column, which a stack of independent runs does not have."""
    if 'result' not in record.columns:
        return []
    df = _population_facets(record, SCENARIO)
    df = df.assign(result=df['result'].astype(str), model=df['model'].astype(str))
    present = set(df['result'])
    results = [r for r in dict.fromkeys(str(r) for r in results) if r in present]
    if len(results) < 2:
        log("[tool] NOTE: fewer than two results have rows - nothing to compare.")
        return []
    models_present = [m for m in canonical_model_order(MODEL) if m in set(df['model'])]
    if not models_present:
        return []
    fig_size = (config or {}).get('fig_size', _DEFAULT_SUB_CONFIG['fig_size'])
    extra = _ps.extra_formats_of(config)
    key_cols = ['population', 'phenotype', 'model'] + [c for c in ('ratio', 'sample') if c in df.columns]
    # Internal, collision-proof column names: a result called 'model' or 'phenotype' must
    # not clash with the key columns once the results become columns of the wide table.
    col_of = {r: f'__result_{i}' for i, r in enumerate(results)}
    tag_of = _unique_file_tags(results)
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for metric in (metrics or _METRIC_LABELS):
        if metric not in df.columns:
            log(f"[tool] NOTE: no result has a {metric!r} column - skipped.")
            continue
        sub = df.loc[df['result'].isin(col_of), ['result'] + key_cols + [metric]].copy()
        sub[metric] = pd.to_numeric(sub[metric], errors='coerce')
        if not sub[metric].notna().any():
            log(f"[tool] NOTE: {metric!r} has no finite value in any result - skipped.")
            continue
        if 'ratio' in sub.columns:
            sub['ratio'] = sub['ratio'].astype(str)  # may be a tuple-like value; a string always groups
        wide = (sub.groupby(key_cols + ['result'])[metric].mean().unstack('result')
                   .rename(columns=col_of).reset_index())
        usable = [r for r in results if col_of[r] in wide.columns]
        pairs, file_stems = [], []
        for i in range(len(usable)):
            for j in range(i + 1, len(usable)):
                a, b = usable[i], usable[j]
                n_matched = int((wide[col_of[a]].notna() & wide[col_of[b]].notna()).sum())
                if n_matched == 0:
                    log(f"[tool] NOTE: {metric!r}: no task is shared by {a!r} and {b!r} "
                        f"(or one lacks the metric) - that pair is skipped.")
                    continue
                log(f"[tool] {metric!r}: {a!r} vs {b!r} - {n_matched} matched task(s).")
                stem = f'{_METRIC_FILE_TAGS.get(metric, _safe_filename(metric))}_{tag_of[a]}_vs_{tag_of[b]}'
                pairs.append((col_of[a], col_of[b], stem, f'{metric}\n({a})', f'{metric}\n({b})'))
                file_stems.append(stem)
        if not pairs:
            continue
        _pairwise_scatter_figures(wide, models_present, fig_size, dpi, pairs,
                                  out_dir, prefix, extra_formats=extra,
                                  label_map=_ps.shown_labels(models_present, config))
        for stem in file_stems:
            for suffix in ('', '_total'):
                base = os.path.join(out_dir, f'{prefix}{stem}{suffix}')
                for ext in ('png',) + tuple(extra):
                    if os.path.isfile(f'{base}.{ext}'):
                        written.append(f'{base}.{ext}')
    return written


# --------------------------------------------------------------------- #
# 6. Czekanowski coefficient scatter-plot matrix (Requirements.md item 5)
# --------------------------------------------------------------------- #

CZEKANOWSKI_THRESHOLDS = tuple(TOPK_PERCENTAGES)
_TEST_META_COLUMNS = ('id', 'population', 'ratio', 'phenotype', 'sample', 'actual')
_CZEK_MAX_DOTS = 4000          # per panel; deterministic subsample beyond this


def _norm_model_name(name):
    s = str(name).replace('-', ' ').replace('_', ' ').strip().lower()
    # Prediction_result_test names the weighted method 'Bayesian' while
    # Metric/Marker_effect/Weight label it 'Bayesian optimisation'.
    return 'bayesian' if s == 'bayesian optimisation' else s


def czekanowski_model_columns(prediction_test, MODEL=None):
    """The prediction columns of `prediction_test` to compare: every
    non-metadata column, restricted to `MODEL` (matching tolerant of the
    'Nelder Mead' vs 'Nelder-Mead' spelling difference between Metric.csv
    labels and prediction-column names) when that leaves >= 2 columns,
    otherwise all of them. Order follows `canonical_model_order()`."""
    import strategy_results as _sr
    cols = [c for c in prediction_test.columns if c not in _TEST_META_COLUMNS and c != _sr.OPTION_COLUMN]
    if MODEL:
        wanted = {_norm_model_name(m) for m in MODEL}
        picked = [c for c in cols if _norm_model_name(c) in wanted]
        if len(picked) >= 2:
            cols = picked
    ordered = canonical_model_order(cols)
    return [c for c in ordered if c in cols] + [c for c in cols if c not in ordered]


def czekanowski_table(prediction_test, MODEL=None, thresholds=CZEKANOWSKI_THRESHOLDS,
                      directions=None):
    """Long table of Czekanowski coefficients (percent) for every model
    pair, threshold and scenario: columns `population, phenotype, ratio,
    sample, model_1, model_2, threshold, coefficient`. A scenario is one
    (population, phenotype, ratio, sample) test split; both models' test
    predictions for the SAME individuals are compared
    (`asreml_accuracy_metrics.czekanowski_coefficient`). `directions` is
    the optional `{phenotype: 'higher'|'lower'}` map (default higher).
    Empty (correct columns) when fewer than two models or no data."""
    from asreml_accuracy_metrics import czekanowski_coefficient
    cols_out = ['population', 'phenotype', 'ratio', 'sample', 'model_1', 'model_2',
                'threshold', 'coefficient']
    if prediction_test is None or prediction_test.shape[0] == 0:
        return pd.DataFrame(columns=cols_out)
    models = czekanowski_model_columns(prediction_test, MODEL)
    if len(models) < 2:
        return pd.DataFrame(columns=cols_out)
    group_cols = [c for c in ('population', 'phenotype', 'ratio', 'sample') if c in prediction_test.columns]
    if 'phenotype' not in group_cols or 'population' not in group_cols:
        return pd.DataFrame(columns=cols_out)
    directions = directions or {}
    rows = []
    for keys, g in prediction_test.groupby(group_cols, sort=False, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        meta = dict(zip(group_cols, keys))
        direction = directions.get(meta.get('phenotype'), 'higher')
        arrays = {m: pd.to_numeric(g[m], errors='coerce').to_numpy(dtype=float) for m in models}
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                for k in thresholds:
                    rows.append({
                        'population': meta.get('population'), 'phenotype': meta.get('phenotype'),
                        'ratio': meta.get('ratio', np.nan), 'sample': meta.get('sample', np.nan),
                        'model_1': models[i], 'model_2': models[j], 'threshold': k,
                        'coefficient': czekanowski_coefficient(arrays[models[i]], arrays[models[j]],
                                                               k, direction),
                    })
    return pd.DataFrame(rows, columns=cols_out)


def _safe_filename(text):
    import re as _re
    return _re.sub(r'[^A-Za-z0-9_.-]+', '_', str(text)) or 'trait'


# ---- Czekanowski scatter-plot matrix: layout / palette -------------------- #
#
# Redesign (this session). The previous matrix filled its lower triangle with
# a Blues-coloured tile carrying default-size text, so dark tiles (high
# coefficients - exactly the values of interest) made the number hard to read,
# and every font was a fixed ~10 pt inside panels that were ~4 in wide, so
# the text looked tiny next to the figure. It also never showed marker
# effects at all. The matrix now follows scatter_plot.py's layout:
#
#   upper triangle : predicted phenotype, model column (x) vs model row (y)
#   lower triangle : marker effect (|effect|, L1-normalised per model and
#                    replicate - the same scale scatter_plot.py uses)
#   diagonal       : the model name
#
# Each off-diagonal panel carries that pair's mean +/- SE Czekanowski
# coefficient (%) in a large white-boxed label (SE = SD / sqrt(number of
# replicate scenarios), the convention metric_summary.py uses).
#
# Dots in BOTH triangles are coloured by WHICH of the two models put the
# point in its top-k% set, so the number can be checked against what is drawn:
#
#   vermillion : in the top-k% of BOTH models
#   sky blue   : in the top-k% of ONE model only
#   grey       : in neither
#
# The lower (marker-effect) triangle is drawn on a common 0-1 scale
# (`_CZEK_EFFECT_SCALE`: |effect| / the model's largest |effect| in that
# replicate, or percentile rank), so x and y share identical limits in every
# panel. L1-normalised effects of a tree model and of rrBLUP still differ by
# orders of magnitude, so on shared axes the colour classes were unreadable.

# Fonts are absolute point sizes chosen for the panel size (times the GUI's
# `font_size` multiplier), not a seaborn font_scale on a ~10 pt base.

_CZEK_MAX_DOTS = 4000              # predicted-phenotype dots per panel
_CZEK_MAX_EFFECT_DOTS = 8000       # marker-effect dots per panel
_CZEK_PANEL_IN_PER_FIG_SIZE = 0.8  # inches of panel per unit of `fig_size`
_CZEK_PANEL_IN_MIN, _CZEK_PANEL_IN_MAX = 3.0, 6.0
# Scale of the marker-effect (lower-triangle) axes. Both options give every
# panel identical 0-1 x and y limits:
#   'max'  : |effect| divided by that model's largest |effect| in the same
#            replicate (min-max scaling; |effect| >= 0, so min is 0). Keeps
#            the shape/magnitude information of the effects.
#   'rank' : percentile rank of |effect| (uniform 0-1), with dotted top-k%
#            cut-off lines.
_CZEK_EFFECT_SCALE = 'max'
_CZEK_COLOURS = {'both': '#d55e00', 'one': '#56b4e9', 'none': '#b3b3b3'}
_EFFECT_META_COLUMNS = ('population', 'phenotype', 'model', 'ratio', 'sample')


def _topk_n(n, k_percent):
    """Size of a top-k% set drawn from `n` items - the SAME rule
    `asreml_accuracy_metrics.czekanowski_coefficient` uses."""
    return max(1, min(int(round(n * float(k_percent) / 100.0)), n))


def _select_top(values, k_percent, direction='higher', magnitude_only=False):
    """Indices (into the finite array `values`) of the top-k% set. Same
    stable-sort selection as `czekanowski_coefficient`. `magnitude_only`
    (marker effects) additionally drops exact zeros: several models
    (SVR/KNN/RKHS...) only score a shortlist of markers and leave every other
    marker at exactly 0, and a top-k% cut that reached into that tied block
    would 'select' arbitrary markers and create spurious overlap."""
    n = values.shape[0]
    k_n = _topk_n(n, k_percent)
    order = np.argsort(values, kind='mergesort')
    idx = order[:k_n] if direction == 'lower' else order[-k_n:]
    if magnitude_only:
        idx = idx[values[idx] > 0]
    return idx


def _czek_coefficient_from_flags(flag_a, flag_b):
    denom = int(flag_a.sum()) + int(flag_b.sum())
    if denom == 0:
        return float('nan')
    return 100.0 * 2.0 * int((flag_a & flag_b).sum()) / denom


def _sample_dots(x, y, flag, budget, rng, uniform=False):
    """Thin a dot cloud to at most ~`budget` points (highlighted points are
    kept preferentially; `uniform=True` thins every point equally). ver4-31
    R4: the body moved to `plot_style.thin_rows` so the scatter matrix can
    share it; this wrapper keeps the historical name and semantics."""
    return _ps.thin_rows(x, y, flag, budget, rng, uniform)


def _concat_dots(parts, budget, rng, uniform=False):
    if not parts:
        return None
    x = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    f = np.concatenate([p[2] for p in parts])
    if x.size == 0:
        return None
    x, y, f = _sample_dots(x, y, f, budget, rng, uniform)
    return {'x': x, 'y': y, 'flag': f}


def _match_effect_label(column, effect_labels):
    """Marker_effect.csv `model` label matching a Prediction_result_test
    column name (tolerant of 'Nelder Mead'/'Nelder-Mead' and
    'Bayesian'/'Bayesian optimisation'), or None."""
    wanted = _norm_model_name(column)
    for lab in effect_labels:
        if _norm_model_name(lab) == wanted:
            return lab
    return None


def _czek_prediction_dots(pred_ph, models, k_percent, direction, rng):
    """Upper-triangle data for one phenotype: `{(i, j): dots}` for i < j,
    x = models[i], y = models[j], each dot flagged 0/1/2 = in the top-k% set
    of neither/one/both models. Selection is per scenario (population x
    ratio x replicate) exactly as in `czekanowski_table`, so the flags agree
    with the printed coefficient."""
    group_cols = [c for c in ('population', 'ratio', 'sample') if c in pred_ph.columns]
    groups = (list(pred_ph.groupby(group_cols, sort=False, dropna=False)) if group_cols
              else [((), pred_ph)])
    per = max(50, int(np.ceil(_CZEK_MAX_DOTS / max(len(groups), 1))))
    parts = {}
    for _keys, g in groups:
        arrays = {m: pd.to_numeric(g[m], errors='coerce').to_numpy(dtype=float) for m in models}
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                a, b = arrays[models[i]], arrays[models[j]]
                mask = np.isfinite(a) & np.isfinite(b)
                x, y = a[mask], b[mask]
                if x.size == 0:
                    continue
                fa = np.zeros(x.size, dtype=bool)
                fb = np.zeros(x.size, dtype=bool)
                fa[_select_top(x, k_percent, direction)] = True
                fb[_select_top(y, k_percent, direction)] = True
                flag = fa.astype(np.int8) + fb.astype(np.int8)
                parts.setdefault((i, j), []).append(_sample_dots(x, y, flag, per, rng))
    return {pair: d for pair, p in parts.items()
            if (d := _concat_dots(p, _CZEK_MAX_DOTS, rng)) is not None}


def _czek_effect_analysis(effect_ph, models, thresholds, plot_ks, rng):
    """Marker-effect side of the matrix for ONE phenotype.

    Returns `(table, dots_by_k)` or `(None, {})` when fewer than two of the
    compared models have marker-effect rows. `table` has the same columns as
    `czekanowski_table` (coefficients in percent, one row per model pair x
    threshold x scenario); `dots_by_k[k]` is `{(i, j): dots}` for the lower
    triangle (x = models[i], y = models[j], i < j).

    The 'top-k%' set of a model in a scenario is its k% largest |effect|
    markers (`_select_top(..., magnitude_only=True)`). Dots are plotted on a
    common 0-1 scale per model and replicate (`_CZEK_EFFECT_SCALE`: |effect|
    / max |effect|, or percentile rank), so every marker-effect panel has
    identical x and y scales; coefficients are computed from the raw |effect|
    ranks and are unaffected by the plotting scale."""
    cols_out = ['population', 'phenotype', 'ratio', 'sample', 'model_1', 'model_2',
                'threshold', 'coefficient']
    marker_cols = [c for c in effect_ph.columns
                   if c not in _EFFECT_META_COLUMNS and c != 'asreml_option']
    if effect_ph.shape[0] == 0 or not marker_cols:
        return None, {}
    present_labels = pd.unique(effect_ph['model']).tolist()
    labels = {m: _match_effect_label(m, present_labels) for m in models}
    usable = [i for i, m in enumerate(models) if labels[m] is not None]
    if len(usable) < 2:
        return None, {}

    mat = np.abs(effect_ph[marker_cols].to_numpy(dtype=np.float32, na_value=np.nan))
    row_sum = np.nansum(mat, axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        mat = np.where(row_sum > 0, mat / row_sum, np.nan).astype(np.float32)

    scen_cols = [c for c in ('population', 'ratio', 'sample') if c in effect_ph.columns]
    if scen_cols:
        keys = list(map(tuple, effect_ph[scen_cols].astype(str).to_numpy()))
    else:
        keys = [()] * effect_ph.shape[0]
    scen_rows, scen_first = {}, {}
    for pos, (key, mdl) in enumerate(zip(keys, effect_ph['model'].tolist())):
        scen_rows.setdefault(key, {}).setdefault(mdl, pos)
        scen_first.setdefault(key, pos)

    phenotype = effect_ph['phenotype'].iloc[0]
    per = max(100, int(np.ceil(_CZEK_MAX_EFFECT_DOTS / max(len(scen_rows), 1))))
    all_ks = sorted(set(thresholds) | set(plot_ks))
    rows, parts = [], {k: {} for k in plot_ks}
    for key, rowmap in scen_rows.items():
        sel = {}
        for i in usable:
            pos = rowmap.get(labels[models[i]])
            if pos is None:
                continue
            row = mat[pos]
            valid = np.flatnonzero(np.isfinite(row))
            if valid.size == 0:
                continue
            order = valid[np.argsort(row[valid], kind='mergesort')]
            # Plot coordinate on a common 0-1 scale (see _CZEK_EFFECT_SCALE):
            # L1 normalisation alone does not unify the axes, because a tree
            # importance and a shrunken rrBLUP effect are L1-normalised to
            # very different ranges (RF's largest marker can be 100x
            # rrBLUP's), so shared axes squash one model against an edge.
            pct = np.full(row.size, np.nan, dtype=np.float32)
            if _CZEK_EFFECT_SCALE == 'rank':
                pct[valid] = _rankdata(row[valid], method='average') / valid.size
            else:
                row_max = float(row[valid].max())
                if row_max <= 0:
                    continue
                pct[valid] = row[valid] / row_max
            flags = {}
            for k in all_ks:
                top = order[-_topk_n(valid.size, k):]
                top = top[row[top] > 0]
                f = np.zeros(row.size, dtype=bool)
                f[top] = True
                flags[k] = f
            sel[i] = (pct, flags)
        first = effect_ph.iloc[scen_first[key]]
        for i in sorted(sel):
            for j in sorted(sel):
                if j <= i:
                    continue
                for k in thresholds:
                    rows.append({
                        'population': first.get('population'), 'phenotype': phenotype,
                        'ratio': first.get('ratio', np.nan), 'sample': first.get('sample', np.nan),
                        'model_1': models[i], 'model_2': models[j], 'threshold': k,
                        'coefficient': _czek_coefficient_from_flags(sel[i][1][k], sel[j][1][k]),
                    })
                both_finite = np.isfinite(sel[i][0]) & np.isfinite(sel[j][0])
                if not both_finite.any():
                    continue
                for k in plot_ks:
                    flag = (sel[i][1][k][both_finite].astype(np.int8)
                            + sel[j][1][k][both_finite].astype(np.int8))
                    parts[k].setdefault((i, j), []).append(_sample_dots(
                        sel[i][0][both_finite], sel[j][0][both_finite], flag, per, rng,
                        uniform=True))
    if not rows:
        return None, {}
    dots_by_k = {k: {pair: d for pair, p in parts[k].items()
                     if (d := _concat_dots(p, _CZEK_MAX_EFFECT_DOTS, rng, uniform=True)) is not None}
                 for k in plot_ks}
    return pd.DataFrame(rows, columns=cols_out), dots_by_k


# ver4-31 R0: the diagonal-label helpers live in plot_style.py now (one copy -
# scatter_plot.py used to hold a verbatim duplicate "to avoid a circular
# import"). Same names kept here so the Czekanowski code below is untouched.
_wrap_diag_label = _ps.wrap_diag_label
_fit_label_pt = _ps.fit_label_pt
_wrap_for_width = _ps.wrap_for_width


def _annotate_coefficient(ax, stats, big_pt):
    """Large, white-boxed coefficient label in the panel's upper-left corner
    (the empty corner of a scatter whose points hug the 1:1 diagonal)."""
    stroke = [path_effects.withStroke(linewidth=3, foreground='white')]
    if stats is None:
        ax.annotate('NA', xy=(0.04, 0.96), xycoords='axes fraction', va='top', ha='left',
                    fontsize=big_pt, fontweight='bold', color='0.35', zorder=10)
        return
    mean, se, count = stats
    ax.annotate(f'{mean:.1f}%', xy=(0.04, 0.96), xycoords='axes fraction', va='top', ha='left',
                fontsize=big_pt, fontweight='bold', color='#111111', zorder=10,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='0.35',
                          linewidth=0.8, alpha=0.93))
    if np.isfinite(se) and count > 1:
        ax.annotate(f'±{se:.1f}', xy=(0.04, 0.96), xycoords='axes fraction',
                    xytext=(2, -big_pt * 1.75), textcoords='offset points', va='top', ha='left',
                    fontsize=big_pt * 0.62, color='#222222', zorder=10, path_effects=stroke)


def _draw_dots(ax, dots, dot_s, size_mult=1.0):
    for flag, z, mult, edge in ((0, 2, 1.0, 'none'), (1, 3, 1.25, 'none'), (2, 4, 1.6, '#3a1a00')):
        sel = dots['flag'] == flag
        if not sel.any():
            continue
        colour = _CZEK_COLOURS[('none', 'one', 'both')[flag]]
        ax.scatter(dots['x'][sel], dots['y'][sel], s=dot_s * mult * size_mult, color=colour,
                   alpha=0.5 if flag == 0 else 0.85, edgecolors=edge,
                   linewidths=0.3 if flag == 2 else 0, zorder=z,
                   rasterized=dots['x'].size > 2000)


def _style_panel_axes(ax, x_name, y_name, tick_pt, label_pt, panel_in):
    fmt = mticker.FuncFormatter(lambda v, _pos: f'{v:g}')
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.tick_params(labelsize=tick_pt, length=4, width=1)
    for lab in ax.get_xticklabels():
        lab.set_rotation(45)
        lab.set_ha('right')
        lab.set_rotation_mode('anchor')
    ax.set_xlabel(x_name, fontsize=_fit_label_pt(x_name, label_pt, panel_in * 0.8), labelpad=4)
    ax.set_ylabel(y_name, fontsize=_fit_label_pt(y_name, label_pt, panel_in * 0.8), labelpad=4)
    for sp in ax.spines.values():
        sp.set_linewidth(1.0)
        sp.set_color('0.25')


def _set_panel_limits(ax, dots, is_effect, threshold=None, extent=None):
    """Shared-limit 1:1 panel (the scatter_plot.py convention,
    `pipeline_utils.set_identity_axes`). Marker-effect panels are drawn on a
    0-1 scale (`_CZEK_EFFECT_SCALE`), so both axes are exactly 0-1. On the
    'rank' scale, dotted lines also mark the top-k% cut-off (1 - k/100) on
    each axis (the upper-right square = markers both models put in their
    top-k%); on the 'max' scale the cut-off differs per replicate, so no
    line is drawn. Predicted-phenotype panels use `extent` (lo, hi) - one
    range shared by EVERY upper panel and by both of its axes - when given,
    else the panel's own data range."""
    x, y = dots['x'], dots['y']
    if is_effect:
        set_identity_axes(ax, [0.0, 1.0], [0.0, 1.0], zero_min=True, draw_line=True)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        if threshold is not None and _CZEK_EFFECT_SCALE == 'rank':
            cut = 1.0 - float(threshold) / 100.0
            ax.axvline(cut, color='0.25', linestyle=':', linewidth=1.1, zorder=1)
            ax.axhline(cut, color='0.25', linestyle=':', linewidth=1.1, zorder=1)
    else:
        lo, hi = extent if extent is not None else (
            float(min(np.nanmin(x), np.nanmin(y))), float(max(np.nanmax(x), np.nanmax(y))))
        set_identity_axes(ax, [lo, hi], [lo, hi], zero_min=False, draw_line=True)


def _czekanowski_matrix_figure(pred_pheno, table_pheno, models, phenotype, threshold, fig_size,
                                dpi, savepath, *, font_size=1.0, direction='higher',
                                effect_table=None, effect_dots=None, pred_dots=None, extra_formats=(),
                                label_map=None):
    """One trait x one threshold: the scatter-plot matrix described at the top
    of this section. `effect_table` / `effect_dots` come from
    `_czek_effect_analysis`; without them (no marker-effect data for this
    trait) the lower triangle falls back to a pale tile carrying the
    predicted-phenotype coefficient in large dark text - never a dark fill
    that hides the number."""
    n = len(models)
    panel = min(max(fig_size * _CZEK_PANEL_IN_PER_FIG_SIZE, _CZEK_PANEL_IN_MIN), _CZEK_PANEL_IN_MAX)
    fs = float(font_size) if font_size else 1.0
    ann_pt, diag_pt, tick_pt, label_pt = 22 * fs, 24 * fs, 12 * fs, 14 * fs
    title_pt, sub_pt, leg_pt = 24 * fs, 14 * fs, 15 * fs
    dot_s = 14.0 * (panel / 4.0) ** 2

    rng = np.random.RandomState(0)
    if pred_dots is None:
        pred_dots = _czek_prediction_dots(pred_pheno, models, threshold, direction, rng)
    has_effect = bool(effect_dots)
    # One predicted-phenotype range for every upper panel, both axes.
    _ext = [(float(min(np.nanmin(d['x']), np.nanmin(d['y']))),
             float(max(np.nanmax(d['x']), np.nanmax(d['y'])))) for d in (pred_dots or {}).values()]
    pheno_extent = (min(e[0] for e in _ext), max(e[1] for e in _ext)) if _ext else None

    def _coef(table):
        if table is None or table.empty:
            return None
        return (table[table['threshold'] == threshold]
                .groupby(['model_1', 'model_2'])['coefficient'].agg(['mean', 'std', 'count']))
    pred_coef, eff_coef = _coef(table_pheno), _coef(effect_table)

    def _stats(coef, lo, hi):
        if coef is None or (models[lo], models[hi]) not in coef.index:
            return None
        s = coef.loc[(models[lo], models[hi])]
        if not np.isfinite(s['mean']):
            return None
        count = int(s['count'])
        # Standard error of the mean over replicate scenarios.
        se = float(s['std']) / np.sqrt(count) if count > 1 and np.isfinite(s['std']) else float('nan')
        return (float(s['mean']), se, count)

    fig_w = fig_h_panels = n * panel
    top_txt = ('Upper triangle: predicted phenotype   |   Lower triangle: marker effect '
               '(' + ('percentile rank of |effect|' if _CZEK_EFFECT_SCALE == 'rank'
                                                       else '|effect| / largest |effect|') + ', 0\u20131 on both axes)   |   Value in panel: '
               'mean ± SE Czekanowski coefficient (%) over replicates') if has_effect else (
               'Upper triangle: predicted phenotype   |   Lower triangle: Czekanowski '
               'coefficient (%), mean ± SE over replicates (no marker-effect data)')
    title = f'{phenotype} – Czekanowski coefficient, top {threshold}%'
    title_w = _wrap_for_width(title, fig_w - 0.6, title_pt)
    sub_w = _wrap_for_width(top_txt, fig_w - 0.6, sub_pt)
    head_in = (title_w.count('\n') + 1) * title_pt * 1.25 / 72 + (sub_w.count('\n') + 1) * sub_pt * 1.35 / 72 + 0.45
    foot_in = leg_pt * 2.6 / 72 + 0.3
    fig_h = fig_h_panels + head_in + foot_in

    fig, axes = plt.subplots(n, n, squeeze=False, figsize=(fig_w, fig_h))
    for r in range(n):
        for c in range(n):
            ax = axes[r][c]
            if r == c:
                max_chars = max(8, int(panel * 72 * 0.86 / (0.62 * diag_pt)))
                text = _wrap_diag_label(models[r], max_chars)
                ax.text(0.5, 0.5, text, ha='center', va='center', fontweight='bold',
                        fontsize=_fit_label_pt(text, diag_pt, panel * 0.86, floor_pt=8.0),
                        transform=ax.transAxes, linespacing=1.1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_facecolor('#eef1f5')
                ax.set_box_aspect(1)
                for sp in ax.spines.values():
                    sp.set_linewidth(1.0)
                    sp.set_color('0.25')
                continue
            lo, hi = min(r, c), max(r, c)
            is_effect = r > c
            if is_effect and not has_effect:
                # Fallback tile: pale fill + big dark number (readable at any value).
                stats = _stats(pred_coef, lo, hi)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_box_aspect(1)
                if stats is not None:
                    ax.set_facecolor(plt.get_cmap('Blues')(0.06 + 0.24 * min(max(stats[0] / 100.0, 0), 1)))
                    ax.text(0.5, 0.56, f'{stats[0]:.1f}%', ha='center', va='center', fontsize=diag_pt,
                            fontweight='bold', color='#111111', transform=ax.transAxes)
                    if np.isfinite(stats[1]) and stats[2] > 1:
                        ax.text(0.5, 0.36, f'±{stats[1]:.1f}', ha='center', va='center',
                                fontsize=diag_pt * 0.6, color='#222222', transform=ax.transAxes)
                else:
                    ax.text(0.5, 0.5, 'NA', ha='center', va='center', fontsize=diag_pt * 0.8,
                            color='0.35', transform=ax.transAxes)
                for sp in ax.spines.values():
                    sp.set_linewidth(1.0)
                    sp.set_color('0.25')
                continue

            source = effect_dots if is_effect else pred_dots
            stored = source.get((lo, hi)) if source else None
            if stored is not None and not is_effect:
                # stored: x = models[lo], y = models[hi]; upper panel wants x = column (hi).
                dots = {'x': stored['y'], 'y': stored['x'], 'flag': stored['flag']}
            elif stored is not None:
                dots = stored          # lower: x = column (lo), y = row (hi) - as stored
            else:
                dots = None
            if dots is None:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center', fontsize=label_pt,
                        color='0.4', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_box_aspect(1)
                continue
            _draw_dots(ax, dots, dot_s, size_mult=0.6 if is_effect else 1.0)
            _set_panel_limits(ax, dots, is_effect, threshold, extent=pheno_extent)
            _style_panel_axes(ax, models[c], models[r], tick_pt, label_pt, panel)
            _annotate_coefficient(ax, _stats(eff_coef if is_effect else pred_coef, lo, hi), ann_pt)

    what = 'lowest' if direction == 'lower' else 'top'
    handles = [
        Line2D([0], [0], marker='o', linestyle='None', markersize=10, markerfacecolor=_CZEK_COLOURS['both'],
               markeredgecolor='#3a1a00', label=f'In {what} {threshold}% of both models'),
        Line2D([0], [0], marker='o', linestyle='None', markersize=9, markerfacecolor=_CZEK_COLOURS['one'],
               markeredgecolor='none', label=f'In {what} {threshold}% of one model only'),
        Line2D([0], [0], marker='o', linestyle='None', markersize=9, markerfacecolor=_CZEK_COLOURS['none'],
               markeredgecolor='none', label=f'In neither {what} {threshold}% set'),
    ]
    if has_effect and _CZEK_EFFECT_SCALE == 'rank':
        handles.append(Line2D([0], [0], color='0.25', linestyle=':', linewidth=1.5,
                              label=f'{what.capitalize()} {threshold}% cut-off (lower triangle)'))
    fig.text(0.5, 1 - 0.18 / fig_h, title_w, ha='center', va='top', fontsize=title_pt, fontweight='bold')
    fig.text(0.5, 1 - (0.18 + (title_w.count('\n') + 1) * title_pt * 1.25 / 72) / fig_h, sub_w,
             ha='center', va='top', fontsize=sub_pt, color='0.25')
    _ps.relabel_figure(fig, label_map)           # ver4-95: shortened model text, before the layout
    fig.tight_layout(rect=(0, foot_in / fig_h, 1, 1 - head_in / fig_h))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.1 / fig_h), ncol=len(handles),
               fontsize=leg_pt, frameon=True, framealpha=0.95, columnspacing=1.8, handletextpad=0.5)
    _ps.save_figure(fig, savepath, dpi=dpi, extra_formats=extra_formats)
    plt.close(fig)


@_ps.themed('ticks', lambda a: 1.0, lambda a: a['dpi'])
def czekanowski_matrix_plot(prediction_test, MODEL, RESULT_NAME, SCENARIO, config=None, dpi=300,
                             thresholds=(10,), directions=None, option_label=None, effect=None):
    """Requirements.md item 5: one scatter-plot matrix per trait per selected
    threshold (`thresholds`, a subset of `CZEKANOWSKI_THRESHOLDS`). Upper
    triangle = predicted-phenotype scatter of every model pair, lower
    triangle = marker-effect scatter of the same pair (needs `effect`, a
    `Marker_effect.csv`-shaped frame already restricted to this option - see
    `accuracy_metrics_plot`), each panel labelled with that pair's mean +/-
    SE Czekanowski coefficient at the threshold. Dots in both triangles are
    coloured by top-k% membership; marker effects are plotted on a common
    0-1 scale per model (`_CZEK_EFFECT_SCALE`). Without `effect` (or without marker-effect rows for a
    trait) the lower triangle shows the predicted-phenotype coefficient as a
    pale tile instead.

    Writes `Czekanowski_coefficients.csv` (predicted phenotypes, ALL
    thresholds - unchanged) and, when marker effects were available,
    `Czekanowski_coefficients_marker_effect.csv` (same columns) next to the
    plots, each with the leading option column when `option_label` is given.
    Returns the list of files written (empty for a silent no-op)."""
    import strategy_results as _sr
    if prediction_test is None or prediction_test.shape[0] == 0:
        return []
    table = czekanowski_table(prediction_test, MODEL, CZEKANOWSKI_THRESHOLDS, directions)
    if table.empty:
        return []
    fig_size = (config or {}).get('fig_size', _DEFAULT_SUB_CONFIG['fig_size'])
    font_size = (config or {}).get('font_size', _DEFAULT_SUB_CONFIG['font_size'])
    # ver4-31 R16: the ticks theme is applied by the @_ps.themed decorator (scoped to this call).
    out_dir = f'./Result/{RESULT_NAME}'
    os.makedirs(out_dir, exist_ok=True)
    written = []
    csv_path = f'{out_dir}/Czekanowski_coefficients.csv'
    (_sr.add_option_column(table, option_label) if option_label else table).to_csv(csv_path, index=False)
    written.append(csv_path)
    models = czekanowski_model_columns(prediction_test, MODEL)
    pred = _population_facets(prediction_test, SCENARIO)
    directions = directions or {}
    plot_ks = [k for k in thresholds if k in CZEKANOWSKI_THRESHOLDS]
    effect_tables = []
    for phenotype in pd.unique(pred['phenotype']):
        pred_ph = pred[pred['phenotype'] == phenotype]
        table_ph = table[table['phenotype'] == phenotype]
        if table_ph.empty:
            continue
        direction = directions.get(phenotype, 'higher')
        eff_table_ph, eff_dots_by_k = None, {}
        if effect is not None and effect.shape[0] and 'phenotype' in effect.columns:
            eff_ph = effect[effect['phenotype'] == phenotype]
            try:
                eff_table_ph, eff_dots_by_k = _czek_effect_analysis(
                    eff_ph, models, CZEKANOWSKI_THRESHOLDS, plot_ks, np.random.RandomState(0))
            except Exception as exc:       # never let the marker-effect half kill the plot
                print(f"[accuracy_metrics_plot] Czekanowski marker-effect panels for {phenotype!r} "
                      f"skipped ({type(exc).__name__}: {exc}); lower triangle falls back to tiles.")
                eff_table_ph, eff_dots_by_k = None, {}
        if eff_table_ph is not None:
            effect_tables.append(eff_table_ph)
        for k in plot_ks:
            path = f'{out_dir}/Czekanowski_{_safe_filename(phenotype)}_top{k}.png'
            _czekanowski_matrix_figure(
                pred_ph, table_ph, models, phenotype, k, fig_size, dpi, path,
                font_size=font_size, direction=direction,
                effect_table=eff_table_ph, effect_dots=eff_dots_by_k.get(k),
                extra_formats=_ps.extra_formats_of(config),
                label_map=_ps.shown_labels(models, config))
            written.append(path)
    if effect_tables:
        eff_all = pd.concat(effect_tables, ignore_index=True)
        eff_csv = f'{out_dir}/Czekanowski_coefficients_marker_effect.csv'
        (_sr.add_option_column(eff_all, option_label) if option_label else eff_all).to_csv(
            eff_csv, index=False)
        written.append(eff_csv)
    return written


# --------------------------------------------------------------------- #
# 5. Accuracy vs Spearman bubble plot
# --------------------------------------------------------------------- #

# Requirements.md item 2 - bubble AREA (in points^2, matplotlib's own
# `s=` unit for `scatter()`) is confined to this fixed, deliberately
# narrow range regardless of the raw size-column's own spread, so one
# unusually large value can no longer blow up every bubble's relative
# scale (the previous `size * 5.0` had no ceiling at all) and two
# similarly-sized models are still visually distinguishable rather than
# both saturating to the same enormous or tiny dot.
_BUBBLE_AREA_MIN = 90.0
_BUBBLE_AREA_MAX = 420.0
_BUBBLE_ALPHA = 0.72


def _scaled_bubble_areas(raw_values, area_min=_BUBBLE_AREA_MIN, area_max=_BUBBLE_AREA_MAX):
    """Min-max scales `raw_values` (any array-like) onto `[area_min,
    area_max]` - computed ONCE, globally across every facet of a figure
    (never per-panel), so the same underlying size value always buys the
    same bubble area everywhere it appears, and a single shared size
    legend (see `_draw_bubble_size_legend()`) is valid for the whole
    figure. Falls back to the midpoint area for a constant/degenerate
    column (never divides by zero)."""
    raw = np.asarray(raw_values, dtype=float)
    finite = raw[np.isfinite(raw)]
    if finite.size == 0 or np.nanmax(finite) == np.nanmin(finite):
        return np.full_like(raw, (area_min + area_max) / 2.0), (
            float(finite[0]) if finite.size else float('nan')
        ), (float(finite[0]) if finite.size else float('nan'))
    rmin, rmax = float(np.nanmin(finite)), float(np.nanmax(finite))
    frac = np.clip((raw - rmin) / (rmax - rmin), 0.0, 1.0)
    return area_min + frac * (area_max - area_min), rmin, rmax


def _perceptually_centred_blue_norm(values):
    """Requirement 2: "a perceptually centred colour scale" paired with
    "dot colour should be based on blue, colour darkness reflecting the
    level of Pearson correlation" - `matplotlib.colors.TwoSlopeNorm`
    piecewise-linearly maps `[vmin, median, vmax] -> [0, 0.5, 1]` of
    whatever colormap it is given, so the sequential 'Blues' colormap's
    own middle shade (rather than one tail of it) lands on the bulk of
    the data's central tendency - this is what keeps a tightly-clustered
    set of Pearson values (e.g. 0.835-0.846) visually spread across the
    colour range instead of all landing within a few percent of the same
    shade the way a plain linear `Normalize(min, max)` easily can once
    combined with a sequential colormap's own non-uniform perceptual
    spacing. Falls back to a plain `Normalize` whenever the data can't
    support a strict `vmin < vcenter < vmax` (too few points, a constant
    column, or a median sitting exactly at one extreme) - `TwoSlopeNorm`
    raises `ValueError` in that case, so this is deliberately checked
    up-front rather than caught after the fact.
    """
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)
    vmin, vmax = float(np.min(finite)), float(np.max(finite))
    if vmin == vmax:
        return Normalize(vmin=vmin - 0.5, vmax=vmax + 0.5)
    vcenter = float(np.median(finite))
    if vmin < vcenter < vmax:
        return TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    return Normalize(vmin=vmin, vmax=vmax)


# Follow-up fix - "not clear which bubble corresponds to which model".
# The original implementation fanned each label's angle purely by that
# row's position within `facet` (i.e. whatever order `groupby()` happened
# to emit - alphabetical by model name), which has no relationship to
# which bubbles actually sit close together on screen. Two bubbles
# drawn almost on top of each other but far apart in that arbitrary row
# order got very different fan angles/distances (fine by luck), while
# bubbles that were close together AND adjacent in that order still
# ended up with near-identical, overlapping labels - and an isolated
# point near an axis edge (e.g. "RF" sitting at the very bottom-left of
# its panel) could still get fanned leftward into the y-axis tick
# labels themselves.
#
# A first attempt at fixing this tried a general-purpose, iterative
# "measure the rendered text boxes and push apart whatever overlaps"
# decluttering pass (in the spirit of the `adjustText` package). That
# approach is powerful but genuinely unstable for a handful of points
# whose stacks compete for the same vertical space: testing this exact
# plot showed pairwise pushes oscillating and one label drifting further
# away every iteration, never converging - a well-known failure mode of
# naive greedy repulsion, not a tuning problem this file should try to
# paper over with damping constants.
#
# The fix that IS robust is to stop resolving collisions after the fact
# and make them structurally impossible instead: every label in a facet
# is assigned a place in ONE global, strictly top-to-bottom ordered
# stack (ranked by that label's own bubble's y - Requirement 2's own
# "Spearman correlation" axis), each row separated from the next by a
# FIXED point-based step. Two labels can never land in the same row, so
# they can never overlap vertically, by construction - no iteration, no
# possibility of non-convergence, regardless of how many bubbles a given
# run selects or how tightly they cluster. Bubbles that sit close
# together on screen (the original bug report) still get a shared,
# nearby horizontal anchor for compactness (see the clustering step
# below), but that only ever affects HOW CLOSE a label sits to its
# neighbours horizontally, never whether it collides with them
# vertically - the one axis where overlap actually happens for stacked
# text.
_LABEL_CLUSTER_TOL = 0.09        # fraction of the facet's own x/y range,
                                  # each axis normalised independently so
                                  # the very different units/scales of
                                  # "Adjusted prediction accuracy" and
                                  # "Spearman correlation" contribute
                                  # comparably to "closeness" - used ONLY
                                  # to decide which labels share a
                                  # horizontal anchor for compactness,
                                  # never to decide vertical placement
# Anchor padding and row spacing are specified in POINTS (the same
# device-independent unit font sizes use), not data units or axis-range
# fractions - a fraction-of-range guess is exactly what under-shot the
# actual rendered text height in earlier testing (stacked 8.5pt bold
# labels visibly overlapping) once the real axes size, DPI and this
# figure's own font_scale were all accounted for. `_points_to_data()`
# converts these into the correct data-unit offsets for a SPECIFIC axes,
# using that axes' own current (post-layout) pixel scale, so the
# conversion is exact regardless of fig_size/dpi/row or column count.
_LABEL_LEFT_PAD_PTS = 10.0
_LABEL_TOP_PAD_PTS = 10.0
_LABEL_ROW_STEP_PTS = 16.0       # comfortably clears an 8.5pt bold line
                                  # plus its white halo stroke (linewidth
                                  # 2.2) at font_size=1.0, with headroom
                                  # to spare - verified by rendering the
                                  # original bug-report configuration;
                                  # scales with font_size like the label
                                  # text itself


def _points_to_data(ax, x_pts, y_pts):
    """Converts a `(x_pts, y_pts)` OFFSET, in points, into the
    equivalent `(dx, dy)` offset in THIS axes' own data units - using
    its actual current pixel scale (`ax.transData`), which is only
    correct once the axes has its FINAL position/size (i.e. after
    `fig.tight_layout()` has run; calling this any earlier would bake in
    whatever provisional, equally-sized grid `plt.subplots()` started
    with, which is wrong the moment a colourbar/legend or a differently
    shaped grid changes any axes' final size). `transData` is affine for
    a standard linear (non-log) axis, so the scale is the same
    everywhere - sampling it once from the origin is exact, not an
    approximation.
    """
    dpi = ax.figure.dpi
    px_x = x_pts * dpi / 72.0
    px_y = y_pts * dpi / 72.0
    x0, y0 = ax.transData.transform((0.0, 0.0))
    x1, _ = ax.transData.transform((1.0, 0.0))
    _, y1 = ax.transData.transform((0.0, 1.0))
    px_per_data_x = abs(x1 - x0) or 1.0
    px_per_data_y = abs(y1 - y0) or 1.0
    return px_x / px_per_data_x, px_y / px_per_data_y


def _cluster_bubble_indices(xs, ys, tol=_LABEL_CLUSTER_TOL):
    """Union-find clustering of the points in one facet, in NORMALISED
    per-axis units (so x and y contribute comparably to "closeness"
    regardless of their raw scales). Two points are linked whenever
    their normalised Euclidean distance is below `tol`; clustering is
    transitive (A-B close and B-C close puts all three in one cluster
    even when A and C are not directly close), which is the right
    behaviour for a chain of near-identical points such as several
    closely-tuned weighted-ensemble methods. Returns a list of
    index-lists, in no particular order. A degenerate (zero-range) axis
    falls back to a range of 1.0 so it never causes a divide-by-zero or
    forces every point together.

    This is used ONLY to choose a shared horizontal anchor for
    compactness (see `_place_facet_bubble_labels()`) - it has no bearing
    on vertical placement, which is resolved globally and cannot overlap
    regardless of how clustering turns out.
    """
    n = len(xs)
    if n == 0:
        return []
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    x_range = float(np.nanmax(xs) - np.nanmin(xs)) or 1.0
    y_range = float(np.nanmax(ys) - np.nanmin(ys)) or 1.0
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            dx = (xs[i] - xs[j]) / x_range
            dy = (ys[i] - ys[j]) / y_range
            if np.hypot(dx, dy) < tol:
                union(i, j)

    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return list(clusters.values())


def _place_facet_bubble_labels(ax, rows, font_size=1.0):
    """Lays out every bubble's label in one facet. `rows` is a list of
    `(x, y, text, area_pts2)` tuples, one per plotted bubble (`area_pts2`
    is that bubble's own scaled area in points^2, exactly what `s=` was
    given, so its on-screen radius can be measured in the same points
    unit `arrowprops`' `shrinkB` below uses).

    Vertical placement (the axis where overlapping text is actually a
    problem) is a single GLOBAL rank ordering across every bubble in
    this facet, by y descending: rank 0 (the visually topmost bubble)
    gets the topmost row, rank 1 the next row down, and so on, each row
    separated from its neighbours by a fixed `_LABEL_ROW_STEP_PTS`
    (converted to this axes' own data units). Because every label's row
    is unique and the step is fixed, no two labels can ever occupy
    overlapping vertical space, by construction - regardless of how many
    bubbles are selected, how tightly several of them cluster, or how
    the clustering below groups them. This is what replaced an earlier,
    iterative "measure and nudge apart" approach that could oscillate
    instead of converging (see the module-level comment above this
    function for why that was abandoned).

    Horizontal placement is local and only affects visual compactness:
    bubbles closer together than `_LABEL_CLUSTER_TOL` (see
    `_cluster_bubble_indices()`) share ONE horizontal anchor, positioned
    left of that cluster's own leftmost bubble - so labels for a tight
    group of near-identical models still sit near that group rather than
    strung out across the whole width of the panel, while an isolated
    bubble (a cluster of size one) simply gets its own anchor at its own
    edge. Every anchor stays left of its cluster (never rightward) - the
    shared Pearson colourbar sits just past the whole figure's right
    edge (Requirement 1), so a label that grows rightward risks growing
    into it.

    Every label is connected to its own bubble by a thin leader line
    (`arrowprops`, `shrinkB` = that bubble's own radius in points plus a
    small pad, so the line starts exactly at its edge regardless of how
    big or small that particular bubble was scaled) - this is the actual
    fix for "not clear which bubble corresponds to which model": even
    when the global ranking places a label well away from its bubble
    (e.g. a bubble that is spatially isolated but ranks in the middle of
    the facet's y-order still gets a row roughly there, not necessarily
    right beside a visually near neighbour), the line makes the
    association explicit rather than relying on proximity alone.
    `clip_on=False` means a label that ends up outside the axes' own
    current view (common near the top of a tall stack) is still drawn in
    full rather than cropped - `bbox_inches="tight"` at save time then
    grows the canvas to include it, exactly as it already does for the
    colourbar and size legend.

    Must be called AFTER `fig.tight_layout()` (see `_points_to_data()`'s
    own docstring for why) - `accuracy_bubble_plot()`'s `_render()` calls
    this in a second pass over all facets, once every axes has its final
    size.
    """
    if not rows:
        return
    xs = np.array([r[0] for r in rows], dtype=float)
    ys = np.array([r[1] for r in rows], dtype=float)
    left_pad_x, _ = _points_to_data(ax, _LABEL_LEFT_PAD_PTS * font_size, 0.0)
    _, top_pad_y = _points_to_data(ax, 0.0, _LABEL_TOP_PAD_PTS * font_size)
    _, row_step_y = _points_to_data(ax, 0.0, _LABEL_ROW_STEP_PTS * font_size)

    # Local horizontal anchor per cluster (compactness only).
    anchor_x = np.empty(len(rows))
    for member_idx in _cluster_bubble_indices(xs, ys):
        cluster_x = xs[member_idx].min() - left_pad_x
        for i in member_idx:
            anchor_x[i] = cluster_x

    # Global vertical rank (the actual overlap-prevention mechanism).
    order = np.argsort(-ys)
    anchor_y_top = float(ys.max()) + top_pad_y
    for rank, i in enumerate(order):
        x, y, text, area_pts2 = rows[i]
        radius_pts = np.sqrt(max(area_pts2, 1.0) / np.pi)
        label_y = anchor_y_top - rank * row_step_y
        ax.annotate(
            text, xy=(x, y), xycoords='data',
            xytext=(anchor_x[i], label_y), 
            textcoords='data',
            ha='right', va='center', fontsize=8.5 * font_size, fontweight='bold',
            color='#08306b', clip_on=False,
            path_effects=[path_effects.withStroke(linewidth=2.2, foreground='white')],
            arrowprops=dict(arrowstyle='-', color='#08306b', linewidth=0.7, alpha=0.75,
                             shrinkA=3.0, shrinkB=radius_pts + 2.0),
        )


def _draw_bubble_size_legend(fig, size_column, area_min, area_max, raw_min, raw_max, font_size=1.0):
    """A compact, three-point reference legend (`min`/median/`max` of the
    raw size column, drawn at the SAME scaled areas `_scaled_bubble_areas()`
    used for the real data) explaining what a bubble's size means - the
    thing a viewer loses once the per-model legend is replaced by on-
    point labels (Requirement 2). Placed BELOW the whole figure (never
    to the right, where the Pearson colourbar already lives - see
    `accuracy_bubble_plot()`'s own `_render()`), so the two auxiliary
    elements can never compete for the same margin, and drawn with
    genuine dummy `Line2D` handles (not a second `scatter()` call) so it
    never becomes part of any axes' own data and cannot be mistaken for
    a real point. A no-op when the raw range is degenerate (nothing
    meaningful to show three reference points for)."""
    if not (np.isfinite(raw_min) and np.isfinite(raw_max)):
        return
    if raw_min == raw_max:
        ref_raw = [raw_min]
    else:
        ref_raw = [raw_min, (raw_min + raw_max) / 2.0, raw_max]
    ref_areas, _, _ = _scaled_bubble_areas(np.array(ref_raw), area_min, area_max)
    handles = [
        Line2D([0], [0], marker='o', linestyle='none', markerfacecolor='#6baed6',
               markeredgecolor='#08306b', markeredgewidth=0.8, alpha=_BUBBLE_ALPHA,
               markersize=np.sqrt(a) , label=f'{v:.3g}')
        for a, v in zip(ref_areas, ref_raw)
    ]
    fig.legend(
        handles=handles, title=f'Bubble size = mean {size_column}', loc='upper center',
        bbox_to_anchor=(0.5, 0.0), ncol=len(handles), frameon=False, fontsize=8.5 * font_size,
        title_fontsize=9 * font_size, handletextpad=1.2, columnspacing=1.6,
    )


@_ps.themed('ticks', lambda a: (a['config'] or {}).get('font_size', _DEFAULT_SUB_CONFIG['font_size']), lambda a: a['dpi'])
def accuracy_bubble_plot(record, MODEL, RESULT_NAME, SCENARIO, config=None, dpi=300,
                          size_column='Top-10 Coincidence'):
    df = _population_facets(record, SCENARIO)
    needed = {'Pearson correlation', ADJUSTED_ACCURACY_COLUMN, SPEARMAN_COLUMN, size_column}
    if not needed.issubset(df.columns):
        return
    models_present = [m for m in canonical_model_order(MODEL) if m in set(df['model'])]
    if not models_present:
        return
    fig_size = (config or {}).get('fig_size', _DEFAULT_SUB_CONFIG['fig_size'])
    font_size = (config or {}).get('font_size', _DEFAULT_SUB_CONFIG['font_size'])
    # ver4-31 R16: the ticks theme is applied by the @_ps.themed decorator (scoped to this call).

    agg = df.groupby(['population', 'phenotype', 'model'], as_index=False).agg(
        x=(ADJUSTED_ACCURACY_COLUMN, 'mean'), y=(SPEARMAN_COLUMN, 'mean'),
        size=(size_column, 'mean'), colour=('Pearson correlation', 'mean'),
    )
    # Requirement 2: bubble AREA and colour NORM are both resolved once,
    # globally across the whole figure's data - see each helper's own
    # docstring for why a per-facet resolution would make bubbles/colours
    # incomparable across panels.
    areas_all, size_min, size_max = _scaled_bubble_areas(agg['size'].to_numpy())
    agg = agg.assign(_area=areas_all)
    norm = _perceptually_centred_blue_norm(agg['colour'].to_numpy())
    # Requirement 2: "the dot colour should be based on blue" - a
    # sequential, colourblind-safe Blues colormap (darker = higher
    # Pearson correlation / predictive ability), replacing the previous
    # categorical marker-shape-per-model encoding entirely now that
    # model identity is carried by the on-point labels instead.
    cmap = plt.get_cmap('Blues')

    phenotypes = list(pd.unique(agg['phenotype']))
    populations = list(pd.unique(agg['population']))

    label_map = _ps.shown_labels(models_present, config)        # ver4-95 (display only)

    def _draw(ax, pop, ph):
        facet = agg if pop is None else agg[agg['population'] == pop]
        facet = facet[facet['phenotype'] == ph].reset_index(drop=True)
        last_sc = None
        label_rows = []
        for i, row in facet.iterrows():
            last_sc = ax.scatter(
                row['x'], row['y'], s=row['_area'],
                c=[row['colour']], cmap=cmap, norm=norm,
                marker='o', edgecolors='#08306b', linewidths=0.7,
                alpha=_BUBBLE_ALPHA,
            )
            label_rows.append((row['x'], row['y'], label_map.get(str(row['model']), str(row['model'])),
                               row['_area']))
        ax.set_xlabel('Adjusted prediction accuracy')
        ax.set_ylabel('Spearman correlation')
        # Headroom for the (possibly stacked, up-and-left) callout
        # labels, which can sit well above/left of a bubble near an axis
        # edge - autoscale alone doesn't know about annotate()'s own
        # data-space xytext. clip_on=False plus bbox_inches='tight' at
        # save time means a label is never actually cropped even when it
        # lands outside these margins; this just keeps the common case
        # tidy without relying on that fallback.
        ax.margins(0.22)
        # NOTE: labels are deliberately NOT placed here. Requirement 2
        # follow-up fix: `_place_facet_bubble_labels()` sizes its
        # anchor/row spacing in POINTS, converted to this axes' own data
        # units via its CURRENT pixel scale (`_points_to_data()`) - which
        # is only correct once every axes has its final post-layout
        # size. `_render()` below calls it in a second pass, after
        # `fig.tight_layout()`, for exactly this reason.
        return last_sc, label_rows

    def _render(pop_list, savepath):
        n_rows, n_cols = max(len(pop_list), 1), max(len(phenotypes), 1)
        fig, axes = plt.subplots(n_rows, n_cols, squeeze=False,
                                  figsize=(fig_size * n_cols, fig_size * n_rows))
        last_sc = None
        facet_label_rows = {}
        for r, pop in enumerate(pop_list or [None]):
            for c, ph in enumerate(phenotypes or [None]):
                ax = axes[r][c]
                sc, label_rows = _draw(ax, pop, ph)
                last_sc = sc if sc is not None else last_sc
                facet_label_rows[(r, c)] = label_rows
                if r == 0:
                    ax.set_title(str(ph))
                if c == 0 and pop is not None:
                    ax.set_ylabel(f"{pop}\n{ax.get_ylabel()}")
        fig.tight_layout()
        # Second pass: every axes now has its FINAL size, so label
        # placement can convert point-based offsets into accurate data
        # units per facet (see `_place_facet_bubble_labels()`'s own
        # docstring for the "not clear which bubble corresponds to which
        # model" bug this replaced).
        for (r, c), label_rows in facet_label_rows.items():
            _place_facet_bubble_labels(axes[r][c], label_rows, font_size=font_size)
        # Requirement 1/2: colourbar (right margin) and the bubble-size
        # legend (bottom margin) live on two DIFFERENT edges of the
        # figure specifically so they can never compete for the same
        # space, or for the space either the old per-model legend or the
        # on-point labels use - `bbox_inches='tight'` at save time then
        # grows the canvas to include both rather than clipping either.
        from matplotlib.ticker import FixedLocator, FormatStrFormatter
        if last_sc is not None:
            cbar = fig.colorbar(last_sc, ax=axes, shrink=0.7, pad=0.02,
                         label='Pearson correlation (predictive ability)')
            # Follow-up fix - "the ticks on the colourbar are uneven".
            # `norm` (`_perceptually_centred_blue_norm()`) is a
            # piecewise-linear TwoSlopeNorm whenever the data supports
            # it, mapping [vmin, vcenter, vmax] -> [0, 0.5, 1] of the bar
            # rather than a single straight line - deliberately, so a
            # tightly clustered set of Pearson values still spans the
            # colour range (see that function's own docstring). The
            # PREVIOUS code picked 6 tick VALUES evenly spaced in raw
            # data units (`np.linspace(vmin, vmax, 6)`) and handed them
            # straight to the bar, which draws each tick at the POSITION
            # its value maps to under `norm` - so evenly-spaced values
            # produced visibly bunched-up positions whenever the median
            # sat far from the midpoint of [vmin, vmax] (exactly what
            # happened in the reported example: one isolated tick near
            # the top, the other five crowded into the bottom third).
            # The fix does the inverse: pick 6 tick POSITIONS evenly
            # spaced along the bar (`[0, 1]` in normalised units) and
            # invert them through the SAME norm to get the data values
            # to label them with - now honestly positioned rather than
            # evenly numbered. This is also a no-op change for a plain
            # linear `Normalize` (its `inverse()` is just `vmin + t *
            # (vmax - vmin)`, identical to the old linspace), so a
            # dataset whose median already sits near the middle looks
            # exactly as before.
            ticks = np.asarray(norm.inverse(np.linspace(0.0, 1.0, 6)), dtype=float)
            cbar.locator = FixedLocator(ticks)
            # Decimal places scale with how tightly the data is
            # clustered, so labels stay distinguishable even for the
            # narrow Pearson-correlation spreads this norm exists to
            # handle - a fixed 3 d.p. can otherwise print the same
            # rounded value twice when vmax - vmin is only a few
            # thousandths.
            span = float(norm.vmax) - float(norm.vmin)
            decimals = 3 if span >= 0.01 else (4 if span >= 0.001 else 5)
            cbar.ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{decimals}f'))
            cbar.update_ticks()
            _draw_bubble_size_legend(fig, size_column, _BUBBLE_AREA_MIN, _BUBBLE_AREA_MAX,
                                     size_min, size_max, font_size=font_size)
        _ps.save_figure(fig, savepath, dpi=dpi, bbox_inches='tight', extra_formats=_ps.extra_formats_of(config))
        plt.close(fig)

    _render(populations, f'./Result/{RESULT_NAME}/Accuracy_bubble.png')
    _render([], f'./Result/{RESULT_NAME}/Accuracy_bubble_total.png')


# --------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------- #

PLOT_NAMES = ('violin_accuracy', 'violin_spearman', 'coincidence', 'czekanowski',
              'pairwise_metrics', 'across_options')

# Configs written before the multi-option update carried ONE master
# checkbox (`ACCURACY_METRICS_PLOT_CREATE`) and four sub-configs; those
# keep producing exactly what they produced (I11): the master flag enables
# these legacy families, the two new ones stay off unless asked for.
_LEGACY_FAMILIES = {
    'violin_accuracy': 'violin_accuracy', 'violin_spearman': 'violin_spearman',
    'coincidence': 'coincidence', 'pairwise_metrics': 'pairwise',
}


def plot_enabled(cfg, name):
    """Is plot family `name` requested by run config `cfg`? A sub-config's
    own `'create'` wins; otherwise the legacy master flag decides for the
    legacy families only."""
    sub = ((cfg or {}).get('ACCURACY_METRICS_PLOT_CONFIG') or {})
    if name in sub and isinstance(sub[name], dict) and 'create' in sub[name]:
        return bool(sub[name]['create'])
    legacy = _LEGACY_FAMILIES.get(name)
    if legacy is not None:
        old = sub.get(legacy)
        if isinstance(old, dict) and 'create' in old:
            return bool(old['create'])
        return bool((cfg or {}).get('ACCURACY_METRICS_PLOT_CREATE', False))
    return False


def any_plot_requested(cfg):
    """True when at least one Tab 5 accuracy-metric plot family is on."""
    return any(plot_enabled(cfg, n) for n in PLOT_NAMES)


def _czek_thresholds(cfg):
    sub = (((cfg or {}).get('ACCURACY_METRICS_PLOT_CONFIG') or {}).get('czekanowski') or {})
    raw = sub.get('thresholds', (10,))
    if isinstance(raw, (int, float)):
        raw = [raw]
    out = []
    for v in raw or []:
        try:
            v = int(v)
        except (TypeError, ValueError):
            continue
        if v in CZEKANOWSKI_THRESHOLDS and v not in out:
            out.append(v)
    return tuple(out) or (10,)


def accuracy_metrics_plot(record, MODEL, RESULT_NAME, SCENARIO, ACCURACY_METRICS_PLOT_CONFIG=None,
                           ACCURACY_BUBBLE_PLOT_CREATE: bool = False, *, prediction_test=None,
                           cfg=None, effect=None):
    """Single entry point `run_sequential.py`/`run_step2_assemble.py`/
    `main_app.py` call for the Tab 5 accuracy-metric plots.

    `record` is the Metric frame (leading `asreml_option` column in a
    multi-option run), `prediction_test` the matching Prediction_result_
    test frame and `effect` the matching Marker_effect frame (both needed
    only by the Czekanowski matrix; without `effect` its lower triangle
    falls back to coefficient tiles). Which families run
    is decided by `plot_enabled()` over `cfg` (the run config; when
    omitted, one is synthesised from `ACCURACY_METRICS_PLOT_CONFIG` and
    the legacy master flag = True, i.e. the pre-multi-option behaviour of
    calling this function at all).

    Multi-option: per-option families (adjusted-accuracy/Spearman violins,
    Top-k curve, Czekanowski matrix, within-option metric pairs) run once
    per option into `Result/<RESULT_NAME>/<option tag>/`; the cross-option
    family runs once into `Result/<RESULT_NAME>/`.
    """
    import strategy_results as _sr
    if cfg is None:
        cfg = {'ACCURACY_METRICS_PLOT_CONFIG': ACCURACY_METRICS_PLOT_CONFIG or {},
               'ACCURACY_METRICS_PLOT_CREATE': True}
    else:
        cfg = dict(cfg)
        cfg.setdefault('ACCURACY_METRICS_PLOT_CONFIG', ACCURACY_METRICS_PLOT_CONFIG or {})
    plot_config = _ps.with_extra_formats(cfg.get('ACCURACY_METRICS_PLOT_CONFIG') or {}, cfg)      # ver4-31 R15
    directions = cfg.get('METRIC_DIRECTIONS') or {}

    if record is None or record.shape[0] == 0:
        print("[accuracy_metrics_plot] Metric.csv has no rows - no accuracy-metric plots written.")
        return
    if not any_plot_requested(cfg):
        return

    options = _sr.options_in_frame(record)
    targets = ([(None, RESULT_NAME)] if not options else
               [(lab, _sr.option_result_name(RESULT_NAME, lab)) for lab in options])
    written_any = False

    for label, out_name in targets:
        rec = _sr.option_view(record, label)
        pred = _sr.option_view(prediction_test, label) if prediction_test is not None else None
        eff = _sr.option_view(effect, label) if effect is not None else None
        tag = f" [{label}]" if label else ''
        os.makedirs(f'./Result/{out_name}', exist_ok=True)

        if plot_enabled(cfg, 'violin_accuracy'):
            if ADJUSTED_ACCURACY_COLUMN in rec.columns and pd.to_numeric(
                    rec[ADJUSTED_ACCURACY_COLUMN], errors='coerce').notna().any():
                c, dpi = _sub_config(plot_config, 'violin_accuracy')
                _violin_metric_plot(rec.copy(), MODEL, out_name, SCENARIO, c, PLOT_DPI=dpi,
                                     metrics=[ADJUSTED_ACCURACY_COLUMN])
                written_any = True
            else:
                print(f"[accuracy_metrics_plot]{tag} no adjusted prediction accuracy values - "
                      f"'Additional accuracy metrics' was off (or every ASReml fit failed); "
                      f"adjusted-accuracy violin skipped.")
        if plot_enabled(cfg, 'violin_spearman') and SPEARMAN_COLUMN in rec.columns:
            c, dpi = _sub_config(plot_config, 'violin_spearman')
            _violin_metric_plot(rec.copy(), MODEL, out_name, SCENARIO, c, PLOT_DPI=dpi,
                                 metrics=[SPEARMAN_COLUMN])
            written_any = True
        if plot_enabled(cfg, 'coincidence'):
            c, dpi = _sub_config(plot_config, 'coincidence')
            coincidence_curve_plot(rec, MODEL, out_name, SCENARIO, c, dpi, directions=directions)
            written_any = True
        if plot_enabled(cfg, 'czekanowski'):
            if pred is None or pred.shape[0] == 0:
                print(f"[accuracy_metrics_plot]{tag} no test-prediction frame available - "
                      f"Czekanowski matrix skipped.")
            else:
                c, dpi = _sub_config(plot_config, 'czekanowski')
                files = czekanowski_matrix_plot(pred, MODEL, out_name, SCENARIO, c, dpi,
                                                 thresholds=_czek_thresholds(cfg), directions=directions,
                                                 option_label=label, effect=eff)
                written_any = written_any or bool(files)
        if plot_enabled(cfg, 'pairwise_metrics'):
            c, dpi = _sub_config(plot_config, 'pairwise_metrics')
            if pairwise_accuracy_scatter_plot(rec, MODEL, out_name, SCENARIO, c, dpi):
                written_any = True
        if label is None and ACCURACY_BUBBLE_PLOT_CREATE:
            c, dpi = _sub_config(plot_config, 'bubble')
            accuracy_bubble_plot(rec, MODEL, out_name, SCENARIO, c, dpi)

    if options and plot_enabled(cfg, 'across_options'):
        c, dpi = _sub_config(plot_config, 'across_options')
        os.makedirs(f'./Result/{RESULT_NAME}', exist_ok=True)
        if metric_across_options_plot(record, MODEL, RESULT_NAME, SCENARIO, c, dpi):
            written_any = True
    elif plot_enabled(cfg, 'across_options') and not options:
        print("[accuracy_metrics_plot] 'Metrics across options' needs a run with more than one "
              "genotype-effect option (BLUEs / BLUPs without GRM / BLUPs with GRM) - skipped.")

    if written_any:
        print(f"[accuracy_metrics_plot] Accuracy-metric plots written under ./Result/{RESULT_NAME}/")
