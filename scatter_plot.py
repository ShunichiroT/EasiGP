import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd

from pipeline_utils import unify_columns_by_position, markers_in_windows, safe_row_l1_normalize, set_identity_axes
from model_registry import model_family
import plot_style as _ps


# Requirements.md (this session) item 3: promoted out of
# _render_scatter_matrix() to module level so BOTH that function and
# scatter_plot()'s own population_palette construction (via
# _distinct_population_palette(), below) read the fixed QTL highlight
# colour from the SAME single authority - rather than scatter_plot()
# hard-coding a second '#ff7f00' literal that could silently drift out of
# sync with the one _render_scatter_matrix() actually draws with.
_LEVEL_PALETTE = {'non-QTL': '#377eb8', 'QTL': '#ff7f00', 'phenotype': 'g'}


def _distinct_population_palette(populations, *, avoid_hex=_LEVEL_PALETTE['QTL'], min_dist=0.35):
    """Build a `{population: colour}` mapping - one distinct colour per
    population - that steers clear of the fixed QTL highlight colour
    (`avoid_hex`, `_LEVEL_PALETTE['QTL']` by default).

    Requirements.md (this session) item 3: with COLOR_BY_POPULATION on,
    the previous plain `sns.color_palette('tab10', ...)` cycle could (and,
    with as few as 2 populations, routinely did - tab10's own index 1)
    hand back the SAME orange as the fixed QTL highlight colour. Whichever
    population then landed on that slot rendered in a colour visually
    indistinguishable from a QTL marker, defeating the very highlight the
    population-colouring path already goes out of its way to keep legible
    (Requirements_2.md item 1 - the QTL layer's colour is always this
    fixed orange, never the population colour). This is the root cause of
    "it becomes difficult to see QTL regions" when population colouring is
    on. Colours within `min_dist` (Euclidean RGB distance, 0-1 scale) of
    `avoid_hex` are skipped when building the cycle; 'tab10' is extended
    with 'tab20' (de-duplicated) if more distinct colours are needed than
    'tab10' has left after the exclusion, so this never runs out even for
    many populations."""
    avoid_rgb = np.array(mcolors.to_rgb(avoid_hex))
    candidates = list(sns.color_palette('tab10')) + list(sns.color_palette('tab20'))
    seen = set()
    usable = []
    for c in candidates:
        key = tuple(round(v, 4) for v in c)
        if key in seen:
            continue
        seen.add(key)
        if np.linalg.norm(np.array(c) - avoid_rgb) >= min_dist:
            usable.append(c)
    if not usable:
        # Pathological avoid_hex/min_dist combination that excludes every
        # candidate colour - fall back to the unfiltered cycle rather than
        # raising (a population palette that occasionally shares the QTL
        # hue is still far better than crashing the whole plot).
        usable = candidates
    return {pop: usable[idx % len(usable)] for idx, pop in enumerate(populations)}


def _save_scatter_legend(save_path, *, dpi=300, color_by_population=False, actual_ranges=None,
                          extra_formats=()):
    """Requirements.md item 3: a standalone legend PNG for the scatter-plot
    matrix, generated alongside the Scatter_plot_<phenotype>.png files -
    mirrors circos_plot.py's own '_legend.png' pattern (a separate,
    dedicated legend image, since the matrix's own per-subplot axes are
    already crowded and each subplot has its own legend removed - see the
    `axes[i,j].get_legend().remove()` call below).

    Three fixed entries, matching this module's own `markers`/`_LEVEL_
    PALETTE` dicts (used for the real scatter points) exactly - content
    never varies between calls, so ONE legend file is generated per
    scatter_plot() call, shared across every phenotype's own matrix,
    rather than one per phenotype:
      - dot       : predicted phenotype (the 'phenotype' level -
        predicted vs. actual)
      - rectangle : marker effect, non-QTL markers
      - triangle  : marker effect, markers flagged as QTL

    Requirements.md (this session) item 3 - `color_by_population`: when
    `True`, every dot's actual COLOUR in the rendered matrix encodes its
    own population (see `_render_scatter_matrix()`'s own docstring), not
    'predicted phenotype' / 'non-QTL' / 'QTL' - that mapping is shown
    separately in `_save_population_legend()`'s own PNG. Before this fix,
    this legend always showed the fixed green/blue/orange colours
    regardless of `color_by_population`, so a viewer comparing it against
    the actual (population-coloured) plot saw two of its three entries'
    colours contradicted by what was actually drawn - the reported "the
    legend does not reflect this setting". `False` (the default)
    reproduces this function's exact prior output.

    Requirements.md (this session, later) - QTL dot colour under
    population colouring: the QTL entry now ALSO switches to the neutral
    grey face when `color_by_population` is True (previously it kept its
    true, fixed orange unconditionally, because QTL markers used to always
    be drawn in that exact colour regardless of `color_by_population` -
    Requirements_2.md item 1). Now that a QTL marker's own colour means
    "population" just like every other dot when this mode is on (see
    `_render_scatter_matrix()`'s own docstring), showing a fixed orange
    swatch here would misrepresent what's actually drawn. The triangle
    shape (plus the on-top/full-opacity/larger-size/thicker-edge emphasis
    applied to every real QTL dot) still identifies it as a QTL point
    regardless of colour - the edge is drawn a little heavier here too, so
    the shape itself reads clearly even in a flat neutral grey.
    """
    _neutral = '#bdbdbd'
    _phenotype_colour = _neutral if color_by_population else _LEVEL_PALETTE['phenotype']
    _non_qtl_colour = _neutral if color_by_population else _LEVEL_PALETTE['non-QTL']
    _qtl_colour = _neutral if color_by_population else _LEVEL_PALETTE['QTL']
    handles = [
        mlines.Line2D([0], [0], marker='o', linestyle='None', markerfacecolor=_phenotype_colour,
                      markeredgecolor='black', markersize=9, label='Predicted phenotype'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor=_non_qtl_colour, edgecolor='black',
                            linewidth=0.6, label='Marker effect'),
        mlines.Line2D([0], [0], marker='v', linestyle='None', markerfacecolor=_qtl_colour,
                      markeredgecolor='black', markeredgewidth=1.4 if color_by_population else 1.0,
                      markersize=9, label='Marker effect (QTL)'),
    ]
    _fig_h = 1.9 if color_by_population else 1.6
    # ver4-31 R7: `actual_ranges` = {phenotype: (min, max)} when the upper
    # panels are coloured by the observed phenotype - one horizontal viridis
    # colourbar per phenotype, titled with its name, is added under the shapes.
    _bar_h = 0.85
    _n_bars = len(actual_ranges) if actual_ranges else 0
    _total_h = _fig_h + _bar_h * _n_bars
    legend_fig = plt.figure(figsize=(3.2, _total_h))
    legend_fig.legend(handles=handles, loc='upper center', fontsize=9, title='Legend (shape)',
                       title_fontsize=10, frameon=True,
                       bbox_to_anchor=(0.5, 1.0 if color_by_population else 0.98))
    for _k, (_ph, (_lo, _hi)) in enumerate((actual_ranges or {}).items()):
        _y0 = (_bar_h * (_n_bars - 1 - _k) + 0.30) / _total_h
        _cax = legend_fig.add_axes([0.12, _y0, 0.76, 0.16 / _total_h * 1.0 + 0.0])
        _cb = legend_fig.colorbar(plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin=_lo, vmax=_hi),
                                                          cmap='viridis'),
                                   cax=_cax, orientation='horizontal')
        _cb.set_label(f'Observed phenotype: {_ph}', fontsize=8)
        _cb.ax.tick_params(labelsize=7)
    if color_by_population:
        legend_fig.text(0.5, 0.04,
                         'Colour = population\n(see Scatter_plot_population_legend.png)',
                         ha='center', va='bottom', fontsize=7, style='italic')
    _ps.save_figure(legend_fig, save_path, dpi=dpi, bbox_inches='tight', extra_formats=extra_formats)
    plt.close(legend_fig)


def _save_population_legend(save_path, population_palette, *, dpi=300, extra_formats=()):
    """Requirements.md item 4: a standalone legend PNG naming which colour
    represents which population - generated once per `scatter_plot()` call
    whenever `COLOR_BY_POPULATION` is on, mirroring `_save_scatter_legend()`
    's own one-legend-per-call pattern (Requirements.md item 3). The
    population-to-colour mapping is fixed for the whole call (built once,
    over every population present across every phenotype - see
    `scatter_plot()`'s own construction of `population_palette`), so one
    legend file covers every phenotype's own matrix rather than one per
    phenotype."""
    handles = [
        mlines.Line2D([0], [0], marker='o', linestyle='None', markerfacecolor=color,
                      markeredgecolor='black', markersize=9, label=str(pop))
        for pop, color in population_palette.items()
    ]
    legend_fig = plt.figure(figsize=(3.2, max(1.6, 0.35 * len(handles) + 0.6)))
    legend_fig.legend(handles=handles, loc='center', fontsize=9, title='Population',
                       title_fontsize=10, frameon=True)
    _ps.save_figure(legend_fig, save_path, dpi=dpi, bbox_inches='tight', extra_formats=extra_formats)
    plt.close(legend_fig)


# ---------------------------------------------------------------------------
# Requirements.md item 1: conventional-vs-machine-learning (conv_vs_ML)
# comparison. A tuning-algorithm suffix ('RKHS__Grid' - see
# models.hyperparameter_tuning.tuned_model_name()) must be stripped before
# model_registry.model_family() can recognise a conventional model that
# happens to have been tuned with more than one algorithm. Duplicated here
# as a single string split rather than importing
# models.hyperparameter_tuning.base_of() itself - that module also pulls
# in scipy.optimize and the batch-Bayesian search engine, considerably
# heavier than anything else this plotting file needs for one inverse-of-
# a-string-format helper. This mirrors the same "duplicate a one-line
# helper rather than add a cross-branch import" choice pipeline_utils.py's
# own _WOPT_LABEL_PREFIXES docstring already makes for the analogous
# W_OPT label set.
# ---------------------------------------------------------------------------
_TUNE_SUFFIX_SEP = '__'


def _base_model_name(name):
    """Inverse of models.hyperparameter_tuning.tuned_model_name(): strips a
    '__<Algorithm>' hyperparameter-tuning suffix, if present, so the
    remaining name can be looked up via model_registry.py's model_family().
    A plain, unsuffixed name is returned unchanged."""
    return str(name).split(_TUNE_SUFFIX_SEP, 1)[0]


def _classify_conv_vs_ML_members(candidate_models):
    """Split `candidate_models` (real column/model names - possibly
    tuning-suffixed, never 'ensemble', which callers already exclude) into
    `(conventional, machine_learning)` lists, each in its own original
    relative order, via model_registry.model_family() - the SAME single
    authority weight_plot.py's own Blues/Greens stacked-bar palette
    already reads from (architecture doc S3's Layer 3d table), rather than
    a second, independently maintained list of the four conventional model
    names. Every name model_family() has never seen before is,
    definitionally, 'machine_learning' (see that function's own
    docstring) - so this never raises and never drops a model silently."""
    conventional = [m for m in candidate_models if model_family(_base_model_name(m)) == 'conventional']
    machine_learning = [m for m in candidate_models if model_family(_base_model_name(m)) == 'machine_learning']
    return conventional, machine_learning


def _conv_vs_ML_masked_mean(df):
    """Row-wise mean across `df`'s (already L1-normalised) columns,
    treating an exact 0.0 as 'this model did not report a value for this
    marker' rather than a genuine zero effect.

    Requirements.md item 1: RKHS/SVR/KNN (architecture doc S9.1/S9.2 -
    Kernel SHAP / surrogate-TreeSHAP effect sources) only ever populate a
    shortlisted top-N% of markers, filling every other marker with an
    exact 0.0 (e.g. SVR.py's own `effect_full = pd.Series(0.0, ...)` then
    `.loc[top_features] = effect_scores`). Averaging those exact zeros in
    on equal footing with a model that scores every marker would dilute a
    conv_vs_ML group's mean in direct proportion to how many of its
    members happen to shortlist - the "difference in the total number of
    marker effects considered" the requirement calls out. Masking them
    out of the mean (rather than out of the denominator) means each
    marker's conv_vs_ML mean reflects only the members that actually
    considered it. A marker with no reporting member at all in this
    conv_vs_ML group (every column exactly 0) comes back as 0.0, not
    NaN."""
    return df.replace(0.0, np.nan).mean(axis=1, skipna=True).fillna(0.0)


def _scenario_codes(frame, cols=('population', 'ratio', 'sample')):
    """Integer scenario id per row of `frame` (one scenario = one
    population x ratio x replicate task); every row of a `Prediction_result`
    / `Marker_effect` frame belongs to exactly one."""
    present = [c for c in cols if c in frame.columns]
    if not present or frame.shape[0] == 0:
        return np.zeros(frame.shape[0], dtype=int)
    key = frame[present].astype(str).agg('|'.join, axis=1)
    return pd.factorize(key)[0]


def _pairwise_pearson(values, names, codes, min_n=3):
    """Pearson correlation between every pair of columns `names` of `values`,
    computed WITHIN each scenario (`codes`, one integer per row) and then
    summarised over scenarios as `(mean r, SE, n_scenarios)` with
    SE = SD(r) / sqrt(n_scenarios). Returned as `{(i, j): stats}` keyed by
    positions in `names` (i < j); SE is NaN with fewer than two scenarios
    (a single replicate has no spread to estimate it from). Non-finite pairs
    are dropped within each scenario (pairwise-complete, `min_n` needed) and
    a scenario where a model is constant (r undefined) is skipped for that
    pair. The SE treats replicates as independent, like the SEs in
    metric_summary.py; replicates that share individuals or markers are not,
    so it is a descriptive measure of replicate-to-replicate spread."""
    present = [n for n in names if n in values.columns]
    if len(present) < 2 or values.shape[0] == 0:
        return {}
    pos = {n: k for k, n in enumerate(names)}
    data = values[present].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    codes = np.asarray(codes)
    order = np.argsort(codes, kind='stable')
    groups = np.split(order, np.flatnonzero(np.diff(codes[order])) + 1)
    collected = {}
    for idx in groups:
        if idx.size < min_n:
            continue
        corr = pd.DataFrame(data[idx], columns=range(len(present))).corr(min_periods=min_n).to_numpy()
        for a in range(len(present)):
            for b in range(a + 1, len(present)):
                r = corr[a, b]
                if np.isfinite(r):
                    collected.setdefault((pos[present[a]], pos[present[b]]), []).append(float(r))
    out = {}
    for pair, rs in collected.items():
        rs = np.asarray(rs)
        se = float(rs.std(ddof=1) / np.sqrt(rs.size)) if rs.size > 1 else float('nan')
        out[pair] = (float(rs.mean()), se, int(rs.size))
    return out


def _annotate_pearson(ax, stats, big_pt, se_ratio=0.68):
    """Pearson r +/- SE label in the panel's upper-left corner (the empty
    corner of a scatter whose points follow the 1:1 diagonal) - same style
    as the Czekanowski matrix (accuracy_metrics_plot.py)."""
    if not stats:
        return
    mean, se, _n = stats
    stroke = [path_effects.withStroke(linewidth=3, foreground='white')]
    ax.annotate(f'r = {mean:.3f}', xy=(0.04, 0.96), xycoords='axes fraction', va='top', ha='left',
                fontsize=big_pt, fontweight='bold', color='#111111', zorder=10,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='0.35',
                          linewidth=0.8, alpha=0.93))
    if np.isfinite(se):
        ax.annotate((f'\u00b1{se:.3f} SE' if se >= 0.0005 else f'\u00b1{se:.1e} SE' if se > 0 else '\u00b10 SE'), xy=(0.04, 0.96), xycoords='axes fraction',
                    xytext=(2, -big_pt * 1.75), textcoords='offset points', va='top', ha='left',
                    fontsize=big_pt * se_ratio, color='#222222', zorder=10, path_effects=stroke)


def _pearson_rows(phenotype, matrix, level, stats, names):
    """Long-table rows for `Scatter_plot_pearson.csv`."""
    return [{'phenotype': phenotype, 'matrix': matrix, 'level': level,
             'model_1': names[i], 'model_2': names[j], 'mean_pearson_r': m, 'se_pearson_r': se,
             'n_scenarios': n} for (i, j), (m, se, n) in sorted(stats.items())]


# ver4-31 R0: the diagonal-label helpers are imported from plot_style.py (the
# verbatim copies that used to live here "to avoid a circular import" are
# gone - plot_style imports no plot module, so there is no cycle).
_wrap_diag_label = _ps.wrap_diag_label
_fit_label_pt = _ps.fit_label_pt


_EFFECT_TICKS = [0, 0.25, 0.5, 0.75, 1.0]


def _row_max_scale(frame):
    """Scale every row (one model's marker-effect vector for one scenario)
    to 0-1 by its own largest |effect| - the scale the Czekanowski matrix
    uses (accuracy_metrics_plot._CZEK_EFFECT_SCALE = 'max'). L1 normalisation
    alone leaves models such as RF (importances) and rrBLUP (shrunken
    effects) orders of magnitude apart, so shared axes would squash one of
    them. Rows that are all zero stay 0."""
    vals = frame.abs().astype(float)
    mx = vals.max(axis=1).replace(0.0, np.nan)
    return vals.div(mx, axis=0).fillna(0.0)


# --------------------------------------------------------------------------- #
# ver4-31 (R4 / R5 / R6 / R7): optional scatter-matrix settings, all read from
# `SCATTER_CONFIG`; every key is optional and an absent key is the legacy
# behaviour (I11).
# --------------------------------------------------------------------------- #

_SIZE_MODES = ('total', 'per_panel')
_DIAG_MODES = ('name', 'name_hist')
_DENSITY_MODES = ('off', 'hexbin', 'auto')
_EFFECT_SCALES = ('max', 'rank', 'sqrt')
_PHENO_HUES = ('level', 'actual')
_PER_PANEL_IN_PER_FIG_SIZE = 0.8      # mirrors the Czekanowski matrix
_PER_PANEL_IN_MIN, _PER_PANEL_IN_MAX = 2.5, 6.0
_EFFECT_SCALE_CAPTION = {'max': '', 'rank': ' (rank-scaled)', 'sqrt': ' (sqrt-scaled)'}


def _scatter_options(cfg):
    """Validated ver4-31 scatter settings from a `SCATTER_CONFIG` dict."""
    c = cfg or {}

    def _pick(key, allowed, default):
        v = c.get(key, default)
        v = default if v is None else str(v).strip().lower()
        return v if v in allowed else default

    mp = c.get('max_points')
    try:
        mp = int(mp) if mp not in (None, '', 0, False) else None
        mp = mp if mp is not None and mp > 0 else None
    except (TypeError, ValueError):
        mp = None
    return {
        'max_points': mp,
        'density': _pick('density', _DENSITY_MODES, 'off'),
        'size_mode': _pick('size_mode', _SIZE_MODES, 'total'),
        'outer_labels_only': bool(c.get('outer_labels_only', False)),
        'diag': _pick('diag', _DIAG_MODES, 'name'),
        'effect_scale': _pick('effect_scale', _EFFECT_SCALES, 'max'),
        'pheno_hue': _pick('pheno_hue', _PHENO_HUES, 'level'),
    }


def _panel_size_in(fig_size, n, mode):
    """Total figure edge (inches) of an n x n matrix: `fig_size` (legacy
    'total') or n panels of clip(fig_size*0.8, 2.5, 6) inches ('per_panel')."""
    if mode == 'per_panel':
        return float(n) * float(min(max(float(fig_size) * _PER_PANEL_IN_PER_FIG_SIZE,
                                        _PER_PANEL_IN_MIN), _PER_PANEL_IN_MAX))
    return fig_size


def _display_effect_scale(values, codes, mode):
    """Display-only rescaling of ONE model's max-scaled |effect| column (0-1).
    'sqrt' = square root; 'rank' = percentile rank among the non-zero values of
    each scenario (`codes`, one id per row; `None` = the whole column), so the
    plotted non-zero values are uniform on (0, 1]; zeros stay 0, NaN stay NaN.
    The Pearson labels never see this transform (they are computed first)."""
    v = np.asarray(values, dtype=float)
    if mode == 'sqrt':
        return np.sqrt(np.clip(v, 0.0, None))
    if mode != 'rank':
        return v
    from scipy.stats import rankdata as _rankdata
    out = np.where(np.isfinite(v), 0.0, np.nan)
    groups = [np.arange(v.size)] if codes is None else [
        np.flatnonzero(codes == c) for c in np.unique(codes)]
    for idx in groups:
        sub = v[idx]
        nz = np.isfinite(sub) & (sub > 0)
        if nz.any():
            r = np.zeros(sub.size)
            r[nz] = _rankdata(sub[nz]) / float(nz.sum())
            r[~np.isfinite(sub)] = np.nan
            out[idx] = r
    return out


def _with_display_effect_scale(frame, cols, codes, mode):
    """Copy of `frame` whose model columns `cols` carry the display-scaled
    effects (`mode` 'rank' / 'sqrt'); `frame` itself is untouched."""
    out = frame.copy()
    use_codes = codes if (codes is not None and len(codes) == len(out)) else None
    for c in cols:
        if c in out.columns:
            out[c] = _display_effect_scale(out[c].to_numpy(dtype=float), use_codes, mode)
    return out


def _render_scatter_matrix(data_scatter, names, markers, save_path, fig_size_total, PLOT_DPI,
                            color_by_population=False, population_palette=None,
                            normalize_effect=True, pheno_r=None, effect_r=None, font_scale=1.0,
                            opts=None, pheno_actual=None, pheno_label=None, actual_range=None,
                            effect_caption='', extra_formats=()):
    """Render one N x N scatter-plot matrix - predicted-phenotype-level
    scatter above the diagonal, marker-effect-level scatter (QTL-
    highlighted) below it - and save it to `save_path`.

    Factored out of the original single-purpose loop so that BOTH the
    per-model matrix (`Scatter_plot_<phenotype>.png`) and the new
    Requirements.md item 1 conv_vs_ML comparison matrix
    (`Scatter_plot_conv_vs_ML_<phenotype>.png`) are drawn by exactly one
    code path and can never visually diverge - the same "single
    authority" discipline the rest of this codebase applies to
    cross-cutting decisions (architecture doc S19.2). `names` supplies
    both the column names read out of `data_scatter` and the axis labels:
    for the per-model matrix this is `model_selected`; for the conv_vs_ML
    comparison it is the two synthesised conv_vs_ML-mean column names.

    Pearson label (this session): when `pheno_r` / `effect_r` are given
    (`{(i, j): (mean r, SE, n)}` keyed by positions in `names`, from
    `_pairwise_pearson`), each predicted-phenotype (upper) / marker-effect
    (lower) panel is labelled with that pair's mean Pearson r and its
    standard error over replicate scenarios. Colours are untouched.

    ver4-25, R5: every off-diagonal panel additionally gets
    `pipeline_utils.set_identity_axes()` - identical x/y limits and tick
    positions plus a dashed corner-to-corner 1:1 line - applied to
    WHATEVER data that specific panel actually plots (the phenotype-level
    frame above the diagonal, the marker-effect-level frame below it), so
    a panel is never visually distorted by independently-scaled axes.
    This applies identically to both matrices this function draws (the
    per-model one and the conv_vs_ML one), since both go through this one
    code path.

    Requirements.md item 4 - `color_by_population`: when `True` (and
    `data_scatter` actually carries a `'population'` column - the
    conv_vs_ML comparison matrix never does, so it always keeps the
    original level-based colouring regardless of this flag), every NON-QTL
    dot's COLOUR encodes its own population (via `population_palette`, a
    `{population: colour}` mapping built once for the whole
    `scatter_plot()` call - see that function's own docstring) instead of
    'predicted phenotype' / 'non-QTL'. On the marker-effect (lower-
    triangle) panels, dot SHAPE still separates non-QTL from QTL markers
    (`style=`), so which marker is a QTL remains visible even while colour
    now means population; the phenotype (upper-triangle) panels have only
    one style ('phenotype') and so are coloured by population alone. When
    `False` (the default - unchanged from before this feature), colouring
    is exactly as before: by 'level'.

    Requirements.md item 5 - regardless of `color_by_population`, a
    marker-effect panel's QTL dots are now drawn in a SEPARATE, LATER
    `sns.scatterplot()` call than the non-QTL dots: non-QTL dots are drawn
    first, at reduced opacity (`alpha=0.6`); QTL dots are then drawn ON
    TOP (matplotlib draws later calls over earlier ones), at full opacity,
    with a larger marker size and a thin black edge - so a QTL marker
    stands out from the non-QTL background regardless of which colour
    scheme (level-based or population-based) is active, rather than being
    just another same-opacity dot in z-order determined by row position.

    Requirements_2.md item 1 (superseded when population colouring is on -
    see the next paragraph) - when `color_by_population` is False, the QTL
    layer's colour is always the fixed `_level_palette['QTL']` orange,
    matching the standalone scatter legend (`_save_scatter_legend()`),
    which always promises an orange triangle for 'Marker effect (QTL)' in
    that mode. Shape (`style=`) and the item-5 on-top/opacity/size/edge
    emphasis above already single out QTL markers independently of
    colour, so this fixed colour costs nothing else here.

    Requirements.md (this session) - QTL dot colour under population
    colouring: when `color_by_population` is True, the QTL layer's colour
    instead follows `population_palette`, exactly like every other dot on
    the panel - a QTL dot's colour means "population" consistently with
    the rest of the plot, rather than always being a fixed orange that
    contradicts its neighbours. A QTL point still needs to be readily
    identifiable AS a QTL point in this mode, so that job shifts entirely
    onto the non-colour cues already in place: drawn last/on top of the
    non-QTL dots (`zorder=5`), full opacity (vs. 0.6 for non-QTL), a
    larger marker, a thicker black edge, and, on marker-effect panels, a
    distinct triangle shape (`style=`) - population is still fully
    recoverable from every dot, QTL or not.

    Requirements.md item 5 (this session) - `normalize_effect`: when True
    (the default), marker-effect panels are treated as the existing
    non-negative, L1-normalised value range they've always been - pinned
    to start at 0 (`zero_min=True`) and drawn with the dashed 1:1
    reference line, exactly as before this feature existed (I11). When
    False, `data_scatter`'s marker-effect columns instead hold RAW,
    un-normalised marker-effect values (see `scatter_plot()`'s own
    `NORMALIZE_MARKER_EFFECT` parameter) - which can be negative and are
    not on any shared scale across models - so marker-effect panels keep
    data-derived (unforced) axis limits (`zero_min=False`) and skip the
    1:1 reference line (`draw_line=False`); only the marker-effect
    (lower-triangle) panels are affected, the phenotype (upper-triangle)
    panels always keep their identity line regardless of this flag.

    ver4-31 (`opts` from `_scatter_options()`; `None` = legacy): `max_points` /
    `density` thin or bin the non-QTL layer of each panel (QTL dots are never
    thinned; Pearson labels come from `pheno_r` / `effect_r`, computed on the
    full data), `outer_labels_only` clears the inner axis labels, `diag=
    'name_hist'` adds a histogram of that model's predicted phenotype under
    its name, `pheno_hue='actual'` colours the upper-triangle dots by the
    observed phenotype (`pheno_actual`, row-aligned with the phenotype-level
    rows of `data_scatter`; `actual_range` = its (min, max)), and
    `effect_caption` is appended to the marker-effect axis captions.
    """
    o = opts or _scatter_options(None)
    n = len(names)
    fig, axes = plt.subplots(n, n, figsize=(fig_size_total, fig_size_total), squeeze=False)
    # Pearson label size follows the PANEL size (fig_size_total / n), times
    # the GUI font_size multiplier.
    _ann_pt = min(max(5.0 * fig_size_total / max(n, 1), 11.0), 26.0) * (font_scale or 1.0)
    _se_ratio = 0.68
    if o['size_mode'] == 'per_panel':
        # Per-panel sizing: the text sizes of the Czekanowski matrix (22 / 24 pt
        # x font size; the SE line at 0.62 of the coefficient), which read
        # better than the panel-scaled legacy sizes on a 6 in panel.
        _ann_pt, _se_ratio = 22.0 * (font_scale or 1.0), 0.62

    # Determine which subplots show marker effects
    lower = np.arange(0, n * n).reshape(n, n)
    lower = list(lower[np.tril_indices(n, k=-1)])

    # Requirements.md (this session) item 3: reads the SAME module-level
    # _LEVEL_PALETTE that _distinct_population_palette() steers the
    # population palette away from, rather than a second, independently
    # maintained copy of these three colours that could silently drift out
    # of sync with it.
    _level_palette = _LEVEL_PALETTE
    _use_population_colour = bool(color_by_population) and 'population' in data_scatter.columns
    # ver4-31 R7: observed-phenotype hue for the upper panels (population wins).
    _use_actual = (o['pheno_hue'] == 'actual' and pheno_actual is not None and not _use_population_colour
                   and actual_range is not None)
    _actual_norm = (mcolors.Normalize(vmin=actual_range[0], vmax=actual_range[1])
                    if _use_actual else None)

    # One predicted-phenotype range shared by EVERY upper panel and both of
    # its axes (same x and y scale in every sub-plot, as in the Czekanowski
    # matrix).
    _pheno_extent = None
    try:
        _pv = data_scatter.loc[data_scatter['level'] == 'phenotype', list(names)].to_numpy(dtype=float)
        _pv = _pv[np.isfinite(_pv)]
        if _pv.size:
            _pheno_extent = (float(_pv.min()), float(_pv.max()))
    except Exception:
        _pheno_extent = None
    _tick_fmt = mticker.FuncFormatter(lambda v, _pos: f'{v:g}')
    _diag_pt = min(max(5.5 * fig_size_total / max(n, 1), 12.0), 28.0) * (font_scale or 1.0)
    if o['size_mode'] == 'per_panel':
        _diag_pt = 24.0 * (font_scale or 1.0)
    _panel_in = fig_size_total / max(n, 1)

    # Generate a subplot per model (or conv_vs_ML group) combination in both types
    for i in range(n):
        for j in range(n):
            if i == j:
                # Diagonal: the model name (as in the Czekanowski matrix).
                _max_chars = max(8, int(_panel_in * 72 * 0.86 / (0.62 * _diag_pt)))
                _text = _wrap_diag_label(names[i], _max_chars)
                _with_hist = o['diag'] == 'name_hist'
                axes[i, j].text(0.5, 0.80 if _with_hist else 0.5, _text, ha='center', va='center',
                                fontweight='bold',
                                fontsize=_fit_label_pt(_text, _diag_pt, _panel_in * 0.86, floor_pt=8.0),
                                transform=axes[i, j].transAxes, linespacing=1.1)
                if _with_hist:
                    # ver4-31 R6: name in the top ~40 %, histogram of this
                    # model's predicted phenotype below, its x range aligned
                    # with the shared `_pheno_extent` of the upper panels.
                    try:
                        _hv = data_scatter.loc[data_scatter['level'] == 'phenotype', names[i]].to_numpy(dtype=float)
                        _hv = _hv[np.isfinite(_hv)]
                        if _hv.size:
                            _hax = axes[i, j].inset_axes([0.10, 0.06, 0.80, 0.50])
                            _hax.hist(_hv, bins=20, range=_pheno_extent, color='#8f9bab', edgecolor='white',
                                      linewidth=0.3)
                            if _pheno_extent is not None and _pheno_extent[1] > _pheno_extent[0]:
                                _hax.set_xlim(*_pheno_extent)
                            _hax.set_yticks([])
                            _hax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=3))
                            _hax.xaxis.set_major_formatter(_tick_fmt)
                            _hax.tick_params(axis='x', labelsize=max(5.0, _diag_pt * 0.45), length=2, pad=1)
                            _hax.grid(False)
                            _hax.patch.set_alpha(0.0)
                            for _sp in ('top', 'right', 'left'):
                                _hax.spines[_sp].set_visible(False)
                    except Exception:
                        pass
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
                axes[i, j].set_facecolor('#eef1f5')
                axes[i, j].set_box_aspect(1)
                continue
            elif (n * i) + j in lower:
                _is_effect_panel = True
                extracted = data_scatter[data_scatter['level'] != 'phenotype']
            else:
                _is_effect_panel = False
                extracted = data_scatter[data_scatter['level'] == 'phenotype']

            _hue_col = 'population' if _use_population_colour else 'level'
            _palette = population_palette if _use_population_colour else _level_palette

            # Requirements.md item 5: non-QTL dots first (lower opacity,
            # behind), QTL dots last (full opacity, larger, on top) - a
            # no-op split for the phenotype panels, whose 'level' is
            # always 'phenotype' (never 'QTL'), so `_qtl` there is always
            # empty and only the first call ever draws anything.
            _non_qtl = extracted[extracted['level'] != 'QTL']
            _qtl = extracted[extracted['level'] == 'QTL']
            _style_col = extracted['level'] if _is_effect_panel else None
            _style_markers = markers if _is_effect_panel else None

            _actual_here = None
            if _use_actual and not _is_effect_panel and len(pheno_actual) == len(_non_qtl):
                _actual_here = np.asarray(pheno_actual, dtype=float)
            if not _non_qtl.empty:
                _drawn = False
                _n_nq = len(_non_qtl)
                _want_hex = (o['density'] == 'hexbin'
                             or (o['density'] == 'auto' and o['max_points'] is not None
                                 and _n_nq > o['max_points']))
                if _want_hex and not _use_population_colour and _actual_here is None:
                    # ver4-31 R4: the non-QTL layer as a log-count hexbin (QTL
                    # dots are drawn on top, unchanged). < 2 finite points
                    # cannot form a density - fall back to the scatter below.
                    _hx = _non_qtl[names[j]].to_numpy(dtype=float)
                    _hy = _non_qtl[names[i]].to_numpy(dtype=float)
                    _ok = np.isfinite(_hx) & np.isfinite(_hy)
                    if _ok.sum() >= 2:
                        _extent = (0.0, 1.0, 0.0, 1.0) if (_is_effect_panel and normalize_effect) else None
                        axes[i, j].hexbin(_hx[_ok], _hy[_ok], gridsize=40, mincnt=1, cmap='Greys',
                                           bins='log', extent=_extent, zorder=2, linewidths=0.1)
                        axes[i, j].set_xlabel(names[j])
                        axes[i, j].set_ylabel(names[i])
                        _drawn = True
                if not _drawn:
                    _keep = _ps.thin_indices(_n_nq, o['max_points'])
                    if _keep.size < _n_nq:
                        _non_qtl = _non_qtl.iloc[_keep]
                        if _actual_here is not None:
                            _actual_here = _actual_here[_keep]
                    if _actual_here is not None:
                        # ver4-31 R7: upper-triangle dots coloured by the observed phenotype.
                        axes[i, j].scatter(
                            _non_qtl[names[j]].to_numpy(dtype=float), _non_qtl[names[i]].to_numpy(dtype=float),
                            c=_actual_here, cmap='viridis', norm=_actual_norm, s=plt.rcParams['lines.markersize'] ** 2,
                            alpha=0.85, zorder=2, edgecolors='none')
                        axes[i, j].set_xlabel(names[j])
                        axes[i, j].set_ylabel(names[i])
                    else:
                        sns.scatterplot(
                            ax=axes[i, j], data=_non_qtl, x=names[j], y=names[i],
                            hue=_non_qtl[_hue_col],
                            style=_non_qtl['level'] if _is_effect_panel else None,
                            markers=_style_markers, palette=_palette,
                            alpha=0.6 if _is_effect_panel else 0.85, zorder=2,
                        )
            if not _qtl.empty:
                # Requirements.md (this session) - QTL dot colour under
                # population colouring: Requirements_2.md item 1 used to
                # force the QTL layer to the fixed QTL orange
                # (`_level_palette['QTL']`) even when `color_by_population`
                # was on, so a QTL dot's colour never matched its own
                # population - contradicting every OTHER dot on the same
                # panel and making it impossible to tell which population a
                # QTL point belongs to. That fix is now narrowed to when
                # population colouring is OFF (`_use_population_colour`
                # False) - there, colour is still the only thing separating
                # 'phenotype'/'non-QTL'/'QTL', so the fixed orange (matching
                # the standalone scatter legend) is kept exactly as before
                # (I11). When population colouring IS on, the QTL layer
                # instead takes `_hue_col`/`_palette` - the SAME per-
                # population colour as the non-QTL call just above - so a
                # QTL dot's colour always means "population", consistently
                # with every other dot on the panel. It still needs to
                # stand out as a QTL point regardless of colour, so that
                # job is carried entirely by the non-colour cues already in
                # place: drawn last/on top (`zorder=5`), full opacity, a
                # larger marker, a thicker black edge, and (on effect
                # panels) a distinct triangle shape via `style=`.
                sns.scatterplot(
                    ax=axes[i, j], data=_qtl, x=names[j], y=names[i],
                    hue=_qtl[_hue_col] if _use_population_colour else None,
                    color=None if _use_population_colour else _level_palette['QTL'],
                    palette=_palette if _use_population_colour else None,
                    style=_qtl['level'] if _is_effect_panel else None,
                    markers=_style_markers,
                    alpha=1.0, s=90, edgecolor='black', linewidth=1.0, zorder=5,
                )
            # ver4-25, R5: identity axes for THIS panel's own plotted data
            # (see this function's own docstring). A panel with no finite
            # data at all (e.g. an empty phenotype/model combination)
            # falls back to set_identity_axes()'s own empty-input default
            # rather than raising.
            # Requirements3.md item 3: the marker-effect (lower-triangle)
            # half of this matrix plots ONLY non-negative values (this
            # function's own `effect_selected` is L1-normalised on
            # `.abs()` input before ever reaching here - see
            # scatter_plot()'s own docstring/comment above the
            # safe_row_l1_normalize() call), so its axes are pinned to
            # start at exactly 0 (`zero_min=True`) - a marker effect can
            # never be negative, and a padded-below-0 axis previously
            # implied otherwise. The phenotype (upper-triangle) half
            # keeps unforced, data-derived limits (Q2's own decision -
            # a predicted/actual phenotype can genuinely be negative).
            # Requirements.md item 5 (this session): a marker-effect panel
            # only gets the "pinned at 0" axis and the 1:1 reference line
            # when marker effects are actually normalised (see this
            # function's own docstring) - raw marker-effect values are
            # neither guaranteed non-negative nor on a comparable scale
            # across models. Phenotype panels (`_is_effect_panel=False`)
            # are unaffected either way (I11).
            _fixed_effect_axes = _is_effect_panel and normalize_effect
            if _fixed_effect_axes:
                # Marker effects are on a 0-1 scale (|effect| / largest
                # |effect|), so every lower panel has identical 0-1 axes.
                set_identity_axes(axes[i, j], [0.0, 1.0], [0.0, 1.0], zero_min=True, draw_line=True)
                axes[i, j].set_xlim(0, 1.0)
                axes[i, j].set_ylim(0, 1.0)
            elif (not _is_effect_panel) and _pheno_extent is not None:
                set_identity_axes(axes[i, j], list(_pheno_extent), list(_pheno_extent),
                                   zero_min=False, draw_line=True)
            else:
                set_identity_axes(axes[i, j], extracted[names[j]].to_numpy(), extracted[names[i]].to_numpy(),
                                   zero_min=_is_effect_panel and normalize_effect,
                                   draw_line=normalize_effect or not _is_effect_panel)
            # Requirements.md (this session) item 1: the x-axis tick labels
            # were heavily overlapping - set_identity_axes() targets up to
            # 6 tick positions per axis (its own default `nbins`), which,
            # left horizontal in a compact N x N grid of square (equal-
            # aspect) panels, routinely collide into an unreadable smear
            # once there are more than a handful of models/phenotypes.
            # Fixed here, specific to this matrix, rather than by lowering
            # set_identity_axes()'s own default `nbins` - that helper is
            # shared with accuracy_metrics_plot.py's single-panel plot,
            # which has no such crowding problem and shouldn't lose tick
            # resolution over a change made for this grid. Thinning to at
            # most 4 ticks (`MaxNLocator(nbins=4)`) first reduces how many
            # labels have to fit at all; rotating the survivors 45 degrees
            # and right-aligning them is the same standard remedy
            # matplotlib itself documents for crowded categorical/numeric
            # tick labels - together the two keep every label legible
            # without needing to touch the y-axis (only the x-axis was
            # reported as overlapping) or the shared identity-axis limits
            # set_identity_axes() already established just above.
            _panel_stats = (effect_r if _is_effect_panel else pheno_r)
            if _panel_stats:
                _annotate_pearson(axes[i, j], _panel_stats.get((min(i, j), max(i, j))), _ann_pt, _se_ratio)
            if _fixed_effect_axes:
                axes[i, j].set_xticks(_EFFECT_TICKS)
                axes[i, j].set_yticks(_EFFECT_TICKS)
            else:
                axes[i, j].xaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
                if not _is_effect_panel:
                    axes[i, j].yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
            # Same compact number format as the Czekanowski matrix (no 1e+03 style).
            axes[i, j].xaxis.set_major_formatter(_tick_fmt)
            axes[i, j].yaxis.set_major_formatter(_tick_fmt)
            for _tick_label in axes[i, j].get_xticklabels():
                _tick_label.set_rotation(45)
                _tick_label.set_ha('right')
                _tick_label.set_rotation_mode('anchor')
            if _is_effect_panel and effect_caption:
                axes[i, j].set_xlabel(names[j] + effect_caption)
                axes[i, j].set_ylabel(names[i] + effect_caption)
            try:
                axes[i, j].get_legend().remove()
            except Exception:
                continue
    if o['outer_labels_only']:
        # ver4-31 R6: axis captions only on the bottom row (x) and the left
        # column (y); every other panel's caption repeats one of those.
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if i != n - 1:
                    axes[i, j].set_xlabel('')
                if j != 0:
                    axes[i, j].set_ylabel('')
    plt.tight_layout()
    if _use_actual and o['outer_labels_only']:
        # The colourbar lives in the figure only when the panels' own labels are
        # gone (the standalone legend always carries one).
        fig.subplots_adjust(right=0.92)
        _cax = fig.add_axes([0.945, 0.35, 0.012, 0.30])
        _cb = fig.colorbar(plt.cm.ScalarMappable(norm=_actual_norm, cmap='viridis'), cax=_cax)
        _cb.set_label(str(pheno_label or 'observed phenotype'))
    _ps.save_figure(fig, save_path, dpi=PLOT_DPI, extra_formats=extra_formats)
    plt.close(fig)


@_ps.themed('whitegrid', lambda a: a['SCATTER_CONFIG']['font_size'], lambda a: a['PLOT_DPI'])
def scatter_plot(MODEL, PHENOTYPE, predicted_result_test, effect, QTL, SCATTER_CONFIG, RESULT_NAME,
                  MARKER_INFO=None, QTL_WINDOW=0.0, QTL_WINDOW_MODE='all_in_window', PLOT_DPI=300,
                  COLOR_BY_POPULATION=False, NORMALIZE_MARKER_EFFECT=True):
    """Draw one scatter-plot matrix per phenotype, comparing every pair of
    selected models at both the predicted-phenotype and marker-effect
    levels, and (optionally) highlighting markers known/believed to be
    near a real QTL.

    ver4-4 R6: the QTL file can now be given in EITHER of two schemas,
    detected automatically by column count:

      - **Legacy (2 columns): 'phenotype', 'marker'.** Exact marker-NAME
        matching, byte-identical to ver4-3 - a marker is flagged 'QTL'
        only if its name appears verbatim in this file for this
        phenotype. `MARKER_INFO`/`QTL_WINDOW`/`QTL_WINDOW_MODE` are
        accepted but have no effect in this mode.
      - **Coordinate (>=5 columns).** So one file can serve both circos
        gene rings and scatter highlighting (ver4-4 Design Blueprint
        S2.6.2), TWO column layouts are accepted here, dispatched by
        column count (bugfix, this session - see the code's own comment
        for the concrete real-world symptom this fixes):
          - **5-7 columns:** 'chromosome', 'start', 'end', 'name',
            'phenotype' [, 'colour'] [, one further ignored column] - a
            QTL-only file, never meant to double as GENE_INFO.
          - **8+ columns:** 'chromosome', 'start', 'end', 'name',
            'colour', 'source', 'phenotype', 'population' [, ...] - the
            REAL `gene_info.csv` layout `circos_plot.py`'s own
            `data_conversion()` uses (its `unify_columns_by_position()`
            call is the single authority this mirrors); pass the exact
            same file already used for `GENE_INFO` here as `QTL` to
            highlight the same regions in the scatter-plot matrix.
        Either way, a marker is flagged 'QTL' if it falls within
        `QTL_WINDOW` of a QTL region for this phenotype
        (`QTL_WINDOW_MODE='all_in_window'`) or is the single nearest
        marker to it (`QTL_WINDOW_MODE='nearest'`) - see
        `pipeline_utils.markers_in_windows()`, which this delegates to
        rather than re-implementing the interval-overlap test (I6).
        Requires `MARKER_INFO` (a marker_info.csv path - already a
        required config key for circos, so no new user burden).

    This was hidden (commented out in the GUI) in ver4-3 because exact-
    name matching essentially never fires in practice - a genotyped
    marker rarely coincides exactly with a reported QTL position - not
    because the underlying styling/highlighting machinery was broken.
    ver4-4 R6 fixes the ROOT CAUSE (exact-name-only matching) rather than
    only re-exposing the same, largely-inert, feature.

    Parameters
    ----------
    MODEL : list of str
        Model labels to include (matches columns already present in
        `predicted_result_test`/`effect`).
    PHENOTYPE : list of str
        Phenotypes to plot - one scatter-plot matrix PNG per entry.
    predicted_result_test : pandas.DataFrame
        `Prediction_result_test.csv`-shaped frame (or the in-memory
        equivalent).
    effect : pandas.DataFrame
        `Marker_effect.csv`-shaped frame (or the in-memory equivalent).
    QTL : str or None
        Path to the QTL file (either schema above), or `None` - `None`
        reproduces today's behaviour exactly (no QTL highlighting at
        all).
    SCATTER_CONFIG : dict
        `{'font_size': int, 'fig_size': int}`.
    RESULT_NAME : str
        Output folder name under `./Result/`.
    MARKER_INFO : str or None, default None
        Path to `marker_info.csv` (`chromosome, name, start, end`).
        Required (raises `ValueError` otherwise) only when `QTL` is in
        coordinate mode; ignored entirely for legacy-mode QTL files or
        when `QTL` is `None`.
    QTL_WINDOW : float, default 0.0
        Symmetric window width (same units as `MARKER_INFO`'s
        `start`/`end` columns) used for coordinate-mode QTL matching.
        `0.0` reproduces exact-overlap-only matching. Ignored in legacy
        mode.
    QTL_WINDOW_MODE : {'all_in_window', 'nearest'}, default
        'all_in_window'. Selects which `pipeline_utils.markers_in_windows`
        assignment strategy is used for coordinate-mode QTL matching.
        Ignored in legacy mode.
    PLOT_DPI : int, default 300
        Resolution (dots per inch) used for both the per-phenotype
        scatter-plot-matrix PNGs and the shared legend PNG. ver4-4 R7.h
        gave `metric_plot()`/`circos_plot()` this same caller-supplied
        `PLOT_DPI` knob (replacing their own previously hard-coded
        values); this brings `scatter_plot()`'s own previously
        hard-coded `dpi=300` (both the `sns.set_theme` rc values and the
        legend save) in line with them, so a single `cfg['PLOT_DPI']`
        controls every raster plot a run produces. `300` is kept as this
        function's own default so a caller that omits the argument keeps
        today's exact behaviour.
    COLOR_BY_POPULATION : bool, default False
        Requirements.md item 4. `False` (the default) reproduces today's
        exact colouring - every dot coloured by 'level' (predicted
        phenotype / non-QTL marker effect / QTL marker effect), via the
        fixed green/blue/orange palette. `True` colours every dot by its
        own `population` column instead (a distinct colour per population
        present in `predicted_result_test`/`effect`, chosen once for the
        whole call and shared across every phenotype's own matrix - see
        `_save_population_legend()`), while marker-effect panels keep
        distinguishing non-QTL from QTL markers by dot SHAPE. Only ever
        applied to the per-model matrix - the conv_vs_ML comparison
        matrix (Requirements.md item 1) has no per-individual/per-marker
        population to colour by (its two 'models' are already
        Conventional/Machine-learning group MEANS) and always keeps the
        level-based colouring regardless of this flag. See
        `_render_scatter_matrix()`'s own docstring for exactly how this
        interacts with Requirements.md item 5's QTL-emphasis change.
    NORMALIZE_MARKER_EFFECT : bool, default True
        Requirements.md item 5 (this session). `True` (the default)
        reproduces today's exact behaviour: each task-row's marker-effect
        vector is L1-normalised (on `.abs()` input) before plotting, via
        `pipeline_utils.safe_row_l1_normalize()` - putting every model's
        effects on the same 0-1 scale, with marker-effect panels pinned to
        a zero-anchored axis and a dashed 1:1 reference line, exactly as
        `ver4-25, R5` established. `False` uses the RAW marker-effect
        values instead (no `.abs()`, no L1 normalisation) - marker-effect
        panels then get data-derived (unforced) axis limits and no 1:1
        reference line is drawn for them (see
        `_render_scatter_matrix()`'s own docstring); the predicted-
        phenotype panels are unaffected either way. Only the per-model
        matrix and the conv_vs_ML comparison matrix's marker-effect halves
        are affected - both read from the same underlying `effect`
        argument.

    Raises
    ------
    ValueError
        If the QTL file's column count matches neither accepted schema,
        or if it's in coordinate mode but `MARKER_INFO` is not provided.
    """

    #if 'Linear transformation' in list(predicted_result_test.columns):
    #     predicted_result_test = predicted_result_test.drop('Linear transformation', axis=1)
    #elif 'Nelder_Mead' in list(predicted_result_test.columns):
    #    predicted_result_test = predicted_result_test.drop('Nelder Mead', axis=1)
    #elif 'Bayesian optimisation' in list(predicted_result_test.columns):
    #    predicted_result_test = predicted_result_test.drop('Bayesian optimisation', axis=1)
    
    #if 'Linear transformation' in effect['type'].values:
    #     effect = effect[effect['type']!='Linear transformation'].reset_index(drop=True)
    #elif 'Nelder_Mead' in effect['type'].values:
    #     effect = effect[effect['type']!='Nelder_Mead'].reset_index(drop=True)
    #elif 'Bayesian optimisation' in effect['type'].values:
    #     effect = effect[effect['type']!='Bayesian optimisation'].reset_index(drop=True) 
    
    model_selected = MODEL.copy()
    if 'ensemble' in MODEL:
        model_selected.remove('ensemble')
        
    # Setting
    # ver4-31 R16: the whitegrid theme is applied by the @_ps.themed decorator (scoped to this call).
    markers = {"non-QTL": "s", "QTL": "v", "phenotype": "o"}
    figsize = SCATTER_CONFIG['fig_size']

    # ver4-31 (R4-R7, R15): optional settings, all absent = legacy.
    opts = _scatter_options(SCATTER_CONFIG)
    extra_formats = _ps.extra_formats_of(SCATTER_CONFIG)
    if opts['effect_scale'] != 'max' and not NORMALIZE_MARKER_EFFECT:
        print("[scatter_plot] effect_scale is ignored: NORMALIZE_MARKER_EFFECT is off "
              "(raw marker effects are not on a shared 0-1 scale).")
        opts['effect_scale'] = 'max'
    if opts['pheno_hue'] == 'actual' and COLOR_BY_POPULATION:
        print("[scatter_plot] pheno_hue='actual' is ignored: COLOR_BY_POPULATION is on "
              "(colouring by population takes precedence).")
        opts['pheno_hue'] = 'level'
    _effect_caption = _EFFECT_SCALE_CAPTION[opts['effect_scale']]
    if any(opts[k] != d for k, d in (('max_points', None), ('density', 'off'), ('size_mode', 'total'),
                                     ('outer_labels_only', False), ('diag', 'name'),
                                     ('effect_scale', 'max'), ('pheno_hue', 'level'))):
        _ps.log_resolved('scatter_plot', **opts)
    # Observed-phenotype colour range per phenotype (R7), from the test rows.
    _actual_ranges = {}
    if opts['pheno_hue'] == 'actual':
        if 'actual' not in predicted_result_test.columns:
            print("[scatter_plot] pheno_hue='actual' is ignored: the test predictions carry no 'actual' column.")
            opts['pheno_hue'] = 'level'
        else:
            for _ph in PHENOTYPE:
                _av = pd.to_numeric(predicted_result_test.loc[predicted_result_test['phenotype'] == _ph, 'actual'],
                                     errors='coerce').to_numpy(dtype=float)
                _av = _av[np.isfinite(_av)]
                if _av.size:
                    _actual_ranges[_ph] = (float(_av.min()), float(_av.max()))

    # Requirements.md item 3: one shared legend PNG per scatter_plot() call
    # (see _save_scatter_legend()'s own docstring for why this isn't
    # generated once per phenotype - the three entries never change).
    # Requirements.md (this session) item 3: color_by_population is now
    # threaded through so this legend's colours match what's actually
    # drawn when population colouring is on (see that function's own
    # docstring for why).
    _save_scatter_legend('./Result/'+RESULT_NAME+'/Scatter_plot_legend.png', dpi=PLOT_DPI,
                          color_by_population=COLOR_BY_POPULATION,
                          actual_ranges=_actual_ranges or None, extra_formats=extra_formats)

    # Requirements.md item 4: the population->colour mapping is built ONCE,
    # over every population present in EITHER source frame across every
    # phenotype (not just this call's own MODEL/PHENOTYPE subset) - so a
    # given population is always the same colour on every phenotype's own
    # matrix, mirroring the fixed, call-wide 'level' palette this feature
    # replaces. `population_palette` stays `None` (a strict no-op, I11)
    # when `COLOR_BY_POPULATION` is off, so nothing below this point
    # behaves any differently than before this feature existed.
    population_palette = None
    if COLOR_BY_POPULATION:
        _all_populations = sorted(
            set(predicted_result_test['population'].astype(str).unique().tolist())
            | set(effect['population'].astype(str).unique().tolist())
        )
        # Requirements.md (this session) item 3: was a plain
        # `sns.color_palette('tab10', ...)` cycle, which could (and with as
        # few as 2 populations, routinely did) assign a population the same
        # orange used to highlight QTL markers, making QTL regions hard to
        # see. _distinct_population_palette() builds the same kind of
        # cycle but steers clear of that fixed QTL colour - see its own
        # docstring.
        population_palette = _distinct_population_palette(_all_populations)
        _save_population_legend(
            './Result/'+RESULT_NAME+'/Scatter_plot_population_legend.png',
            population_palette, dpi=PLOT_DPI, extra_formats=extra_formats,
        )

    # ver4-4 R6: dual-schema QTL file read - detected by column count,
    # exactly as the Design Blueprint (S2.6.2) specifies, rather than a
    # config flag the user has to set correctly themselves. Read and
    # schema-detected ONCE here, outside the phenotype loop below (the
    # same "don't re-read per iteration" discipline ver4-4 R7.b/f already
    # applied to circos_plot.py's own per-model/per-population re-reads).
    QTL_info = None
    _qtl_mode = None
    marker_info_df = None
    if QTL != None:
        QTL_raw = pd.read_csv(QTL)
        _n_qtl_cols = QTL_raw.shape[1]
        if _n_qtl_cols >= 5:
            _qtl_mode = 'coordinate'
            # Requirement 8 / I6: unify by position - the QTL file's
            # STRUCTURAL column headers (the fixed schema position), not
            # the QTL/marker names themselves (the VALUES, matched against
            # real marker names elsewhere via markers_in_windows() and
            # left completely untouched).
            #
            # Bugfix (this session, part 1): a previous version of this
            # block followed the rename with `QTL_info.iloc[:,-3] =
            # QTL_info.iloc[:,-1]` then `QTL_info = QTL_info.iloc[:, :-2]`
            # - negative-position slicing that dropped/corrupted 'end' and
            # 'phenotype' for the ordinary 5/6-column case, crashing every
            # coordinate-mode QTL run outright. Removed; nothing downstream
            # reads a 'colour' column from QTL_info (its only other use is
            # the phenotype filter below), so an extra trailing column
            # simply rides along unused.
            #
            # Bugfix (this session, part 2 - the actual remaining "QTL
            # never shows up" cause): this file is explicitly meant to
            # double as GENE_INFO for circos (this function's own
            # docstring: "the gene_info.csv column set, so one file can
            # serve both circos gene rings and scatter highlighting"), and
            # a real user config does exactly that (QTL and GENE_INFO set
            # to the SAME path). But circos_plot.py's own gene_info.csv
            # schema (see its `data_conversion()`) is EIGHT columns -
            # 'chromosome', 'start', 'end', 'name', 'colour', 'source',
            # 'phenotype', 'population' - with 'phenotype' at position 7,
            # NOT position 5 as this block used to assume unconditionally.
            # Reading such a file with the old 5-column assumption
            # silently renamed 'colour' (hex strings like '#d92f27') to
            # 'phenotype', so `QTL_info[QTL_info['phenotype']==
            # PHENOTYPE[k]]` below never matched any real phenotype name -
            # QTL_info_selected came back empty for every phenotype, so
            # QTL regions/markers were never flagged at all, silently,
            # regardless of QTL_WINDOW/QTL_WINDOW_MODE/
            # COLOR_BY_POPULATION - exactly the "QTL doesn't seem to be
            # represented" symptom, for any run sharing one file between
            # GENE_INFO and QTL. Now detected explicitly by column count,
            # the same dispatch principle this whole dual-schema read
            # already uses: >=8 columns reads the real gene_info.csv
            # layout (extracting 'phenotype' from its true position 7,
            # ignoring 'colour'/'source'/'population', which this
            # function has no use for); 5-7 columns keeps the original,
            # QTL-only minimal layout ('chromosome', 'start', 'end',
            # 'name', 'phenotype'[, 'colour'][, one further ignored
            # column]) for a file that was never meant to double as
            # GENE_INFO.
            if _n_qtl_cols >= 8:
                _gene_info_schema = unify_columns_by_position(
                    QTL_raw,
                    ['chromosome', 'start', 'end', 'name', 'colour', 'source', 'phenotype', 'population'],
                    'QTL file (gene_info.csv-shaped: also used as GENE_INFO)', first_n=8)
                QTL_info = _gene_info_schema[['chromosome', 'start', 'end', 'name', 'phenotype']]
            else:
                QTL_info = unify_columns_by_position(
                    QTL_raw, ['chromosome', 'start', 'end', 'name', 'phenotype'],
                    'QTL file', first_n=5)
        elif _n_qtl_cols == 2:
            _qtl_mode = 'legacy'
            # Requirement 8: unify by position (phenotype, marker) -
            # 'marker' here is this file's marker-identifying column
            # HEADER (a fixed, structural field per the GUI's own
            # documented legacy schema: 'phenotype|marker name identified
            # as QTL'), not the marker names themselves (the VALUES,
            # matched against real marker names elsewhere and left
            # completely untouched).
            QTL_info = unify_columns_by_position(QTL_raw, ['phenotype', 'marker'], 'QTL file')
        else:
            raise ValueError(
                f"[scatter_plot] QTL file '{QTL}' has {_n_qtl_cols} column(s); expected either "
                f"exactly 2 ('phenotype', 'marker' - legacy exact marker-name matching) or at "
                f"least 5 ('chromosome', 'start', 'end', 'name', 'phenotype'[, 'colour'] - "
                f"coordinate-mode window matching). Got columns: {list(QTL_raw.columns)}."
            )

        if _qtl_mode == 'coordinate':
            if not MARKER_INFO:
                raise ValueError(
                    "[scatter_plot] QTL file is in coordinate mode (>=5 columns: chromosome, "
                    "start, end, name, phenotype[, colour]), which requires MARKER_INFO (the "
                    "same marker_info.csv path already used for the circos gene/marker rings - "
                    "see the architecture doc's Appendix B) so QTL regions can be mapped to "
                    "nearby markers by genomic position. MARKER_INFO was not provided."
                )
            marker_info_df = pd.read_csv(MARKER_INFO)
            # Requirement 8: same reasoning as circos_plot.py's own
            # marker_info.csv load - 'chromosome', 'name', 'start', 'end'
            # are this file's structural column headers, not marker
            # identities.
            marker_info_df = unify_columns_by_position(
                marker_info_df, ['chromosome', 'name', 'start', 'end'], 'marker info file')
            print(
                f"[scatter_plot] QTL file detected as COORDINATE mode ({_n_qtl_cols} columns) - "
                f"markers within QTL_WINDOW={QTL_WINDOW} (QTL_WINDOW_MODE={QTL_WINDOW_MODE!r}) of "
                f"a QTL region will be flagged, rather than requiring an exact marker-name match."
            )
        else:
            print(
                "[scatter_plot] QTL file detected as LEGACY mode (2 columns: phenotype, marker) "
                "- exact marker-name matching. QTL_WINDOW/QTL_WINDOW_MODE have no "
                "effect in this mode."
            )
    
    pearson_rows = []      # -> Scatter_plot_pearson.csv (one row per model pair)
    # Generate a scatter plot per phenotype
    for k in range(len(PHENOTYPE)):
        
        # Change the format of the data for scatter plot matrix
        _phenotype_rows = predicted_result_test[predicted_result_test['phenotype']==PHENOTYPE[k]].reset_index(drop=True)
        predicted_test_formatted = _phenotype_rows.iloc[:,6:].reset_index(drop=True)
        if COLOR_BY_POPULATION:
            # Requirements.md item 4: each individual's own real
            # population, carried alongside its prediction columns so it
            # survives into `data_scatter` below for use as the hue.
            predicted_test_formatted['population'] = _phenotype_rows['population'].astype(str).values
        effect_selected = effect[effect['phenotype']==PHENOTYPE[k]].reset_index(drop=True)

        # ver4-25, R5: per-TASK-ROW L1 normalisation of marker effects -
        # each row (one population x model x ratio x replicate task's
        # own full marker-effect vector) divided by its own sum of
        # magnitudes, via the SAME `safe_row_l1_normalize()` the naive
        # ensemble already applies to its own combined marker effects
        # (models/ensemble.py::ensemble(), architecture doc S12.1) -
        # putting every model's effects on the same scale BEFORE they
        # are melted into the per-marker table below, rather than only
        # taking a magnitude (`.abs()`) as before. Replaces the separate,
        # later per-model-COLUMN `.abs()` loop entirely (removed below -
        # a row already L1-normalised from non-negative inputs is
        # already non-negative, so re-abs()-ing it afterwards was, and
        # remains, a no-op).
        # Requirements.md item 5 (this session): normalisation is now
        # opt-out via NORMALIZE_MARKER_EFFECT (default True, I11 - see
        # this function's own docstring). When False, the raw marker-
        # effect values are left exactly as read (no `.abs()`, no L1
        # normalisation), and `_render_scatter_matrix()` is told
        # (`normalize_effect=False`, passed at both call sites below) not
        # to pin marker-effect panels to a zero-anchored axis or draw a
        # 1:1 reference line for them.
        if effect_selected.shape[0] != 0 and NORMALIZE_MARKER_EFFECT:
            effect_selected.iloc[:, 5:] = safe_row_l1_normalize(effect_selected.iloc[:, 5:].abs())
            # Then |effect| / that row's largest |effect| -> every model and
            # scenario on the same 0-1 scale, so all lower panels share
            # identical x and y axes (Pearson r is scale-free, unchanged).
            effect_selected.iloc[:, 5:] = _row_max_scale(effect_selected.iloc[:, 5:])

        effect_formatted = pd.DataFrame()
        _eff_codes = None      # scenario id per melted marker-effect row (see below)
        for jj in range(len(MODEL)):
            if MODEL[jj] == 'ensemble':
                continue
            else:
                tmp = effect_selected[effect_selected['model']==MODEL[jj]].reset_index(drop=True)
                if tmp.shape[0] != 0:
                    if effect_formatted.shape[0] == 0:
                        # `melt()` below stacks marker columns one after the
                        # other (marker-major): melted row index = marker *
                        # n_tasks + task, so the scenario id of every melted
                        # row is the per-task id tiled once per marker.
                        _eff_codes = np.tile(_scenario_codes(tmp), max(tmp.shape[1] - 5, 1))
                    if COLOR_BY_POPULATION and effect_formatted.shape[0] == 0:
                        # Requirements.md item 4: keep the row-aligned
                        # 'population' column through the SAME melt that
                        # already flattens every (population, ratio,
                        # replicate) task-row's marker columns into one
                        # long per-marker series - `id_vars=['population']`
                        # changes nothing about the row/column ordering
                        # `pd.DataFrame.melt()` already produces (still one
                        # block per marker column, rows in original order
                        # within each block), so every OTHER model's own
                        # plain melt below still lines up column-wise with
                        # this one. Only captured from the FIRST
                        # contributing model - the task order (and so the
                        # population sequence) is shared across every
                        # model, so one 'population' column already
                        # describes every model's own column in this frame.
                        tmp = tmp[['population'] + list(tmp.columns[5:])].melt(id_vars=['population'])
                        tmp.columns = ['population', 'marker', MODEL[jj]]
                        tmp = tmp[['marker', 'population', MODEL[jj]]]
                    else:
                        tmp = tmp.iloc[:,5:].melt()
                        tmp.columns = ['marker',MODEL[jj]]
                    if effect_formatted.shape[0] == 0:
                         effect_formatted = tmp
                    else:
                        effect_formatted = pd.concat([effect_formatted,
                                                      tmp[[MODEL[jj]]]],axis=1)

        # Requirements.md item 1: classify this phenotype's actually-
        # present models - only those that really contributed a column to
        # predicted_test_formatted/effect_formatted above, not the nominal
        # MODEL list, since a model can be silently absent for one
        # phenotype (e.g. `tmp.shape[0] == 0` above) - into the two
        # groups model_registry.py's model_family() defines, ready for
        # the conv_vs_ML comparison matrix built after the QTL section
        # below (marker-effect conv_vs_ML means need `qtl_markers`,
        # computed there). 'population' (Requirements.md item 4, only
        # present when COLOR_BY_POPULATION) is excluded from both column
        # lists here - it names no real model and model_family() would
        # otherwise (harmlessly, but wrongly) classify it as
        # 'machine_learning' (its own documented "never seen before"
        # fallback).
        _pred_model_cols = [c for c in predicted_test_formatted.columns if c != 'population']
        _conventional_pred_cols, _ml_pred_cols = _classify_conv_vs_ML_members(_pred_model_cols)
        _effect_model_cols = [c for c in effect_formatted.columns if c not in ('marker', 'population')]
        _conventional_effect_cols, _ml_effect_cols = _classify_conv_vs_ML_members(_effect_model_cols)

        conv_vs_ML_predicted_formatted = None
        if _conventional_pred_cols and _ml_pred_cols:
            # No scale normalisation needed here (unlike marker effects,
            # below): every model's prediction column is already the same
            # phenotype in the same units, so a plain conv_vs_ML mean is
            # directly comparable.
            conv_vs_ML_predicted_formatted = pd.DataFrame({
                'Conventional': predicted_test_formatted[_conventional_pred_cols].mean(axis=1),
                'Machine learning': predicted_test_formatted[_ml_pred_cols].mean(axis=1),
            })
        
        # Add the information of level
        predicted_test_formatted['level'] = 'phenotype'
        effect_formatted['level'] = 'non-QTL'
        if conv_vs_ML_predicted_formatted is not None:
            conv_vs_ML_predicted_formatted['level'] = 'phenotype'
        
        if QTL:
            QTL_info = QTL_info.loc[:,~QTL_info.columns.duplicated(keep='last')]
            QTL_info_selected = QTL_info[QTL_info['phenotype']==PHENOTYPE[k]]
            if _qtl_mode == 'legacy':
                qtl_markers = set(QTL_info_selected['marker'].tolist())
            else:
                # ver4-4 R6: coordinate-mode window matching. `available_markers`
                # is restricted to the markers actually present in THIS
                # phenotype's effect table (mirroring
                # _map_genes_to_markers's own "the genotype table actually
                # being modelled for this task" restriction) - not every
                # marker on this file's chromosome, and not markers from a
                # different phenotype/model combination.
                available_markers = effect_formatted['marker'].tolist()
                region_to_markers = markers_in_windows(
                    QTL_info_selected, marker_info_df, available_markers,
                    window=QTL_WINDOW, mode=QTL_WINDOW_MODE)
                qtl_markers = set()
                for _markers in region_to_markers:
                    qtl_markers.update(_markers)
            effect_formatted.loc[effect_formatted['marker'].isin(qtl_markers), 'level'] = 'QTL'
            # Requirement (diagnostic, matches the "log negative/informative
            # states unconditionally" convention this codebase already
            # follows elsewhere - architecture doc S17): how many markers
            # ended up flagged, out of how many considered, so a QTL_WINDOW
            # set far too large (flagging almost everything) or far too
            # small (flagging nothing) is immediately visible in the log,
            # not just guessable from the rendered plot.
            _n_flagged = int((effect_formatted['level'] == 'QTL').sum())
            print(
                f"[scatter_plot] phenotype={PHENOTYPE[k]!r}: {_n_flagged} of "
                f"{effect_formatted.shape[0]} marker(s) flagged as QTL "
                f"({QTL_info_selected.shape[0]} QTL region(s)/row(s) considered)."
            )
        
        # Requirements.md item 1: conv_vs_ML (conventional vs machine-
        # learning) marker-effect comparison, built from the SAME
        # effect_formatted table already assembled above for the
        # individual-model matrix - never re-reads or re-derives anything
        # from Marker_effect.csv itself. Placed after (not before) the QTL
        # section above so the SAME `qtl_markers` set already resolved
        # there (legacy or coordinate mode) can be reused for this frame's
        # own QTL flagging, rather than recomputed a second time.
        conv_vs_ML_effect_formatted = None
        if _conventional_effect_cols and _ml_effect_cols:
            # ver4-25, R5: `effect_formatted` is already PER-TASK-ROW
            # L1-normalised (see the `safe_row_l1_normalize()` call
            # applied to `effect_selected` before melting, above) -
            # exactly the same normalisation the naive ensemble itself
            # applies (models/ensemble.py::ensemble(), architecture doc
            # S12.1). The separate, additional per-model-COLUMN
            # normalisation this used to do here (`_safe_col_l1_normalize()`,
            # now removed) would have normalised an already-normalised
            # value a second time, on a different axis - built straight
            # from `effect_formatted` directly instead.
            conv_vs_ML_effect_formatted = pd.DataFrame({'marker': effect_formatted['marker']})
            # Marker-count normalisation (the "different number of markers
            # considered" half of Requirements.md item 1):
            # `_conv_vs_ML_masked_mean()` averages each marker only over
            # the conv_vs_ML group members that actually reported a
            # (post-normalisation, still non-zero) value for it - see that
            # function's own docstring for why RKHS/SVR/KNN's top-N%
            # shortlisting makes this necessary.
            conv_vs_ML_effect_formatted['Conventional'] = _conv_vs_ML_masked_mean(
                effect_formatted[_conventional_effect_cols])
            conv_vs_ML_effect_formatted['Machine learning'] = _conv_vs_ML_masked_mean(
                effect_formatted[_ml_effect_cols])
            conv_vs_ML_effect_formatted['level'] = 'non-QTL'
            if QTL:
                conv_vs_ML_effect_formatted.loc[
                    conv_vs_ML_effect_formatted['marker'].isin(qtl_markers), 'level'] = 'QTL'
        elif _effect_model_cols:
            # Requirements.md item 1 / architecture doc S18 ("log negative
            # states unconditionally"): the comparison needs at least one
            # model from EACH conv_vs_ML group to mean anything - a run
            # selecting only conventional (or only machine-learning)
            # models still gets its normal scatter-plot matrix below, just
            # no conv_vs_ML comparison figure, with an explicit reason
            # logged rather than a silently missing output file.
            _only_conv_vs_ML_group = 'conventional' if not _ml_effect_cols else 'machine-learning'
            print(
                f"[scatter_plot] phenotype={PHENOTYPE[k]!r}: conv_vs_ML comparison skipped - "
                f"every selected model with marker-effect data for this phenotype is "
                f"{_only_conv_vs_ML_group} (need at least one conventional AND one "
                f"machine-learning model to compare)."
            )
        
        # Combine both predicted phenotype and marker effect information.
        # ver4-25, R5: fillna(0) is scoped to each LEVEL's own genuinely
        # missing columns, rather than applied as one blanket fillna(0)
        # over the whole concatenated frame. The two source frames here
        # normally share the identical set of model columns, but a model
        # can legitimately be absent from one of them for this phenotype
        # (e.g. `tmp.shape[0] == 0` above - no marker-effect rows at all
        # for one model) - concat() then leaves that column entirely NaN
        # for every row of the OTHER frame, which is a genuine "nothing
        # to report" structural gap and belongs at 0. A NaN that already
        # existed WITHIN either frame's own native data (e.g. a diverged
        # model's NaN prediction) is left as NaN instead of being
        # silently forced to 0 - `sns.scatterplot`/`set_identity_axes`
        # already skip non-finite values, so it is correctly excluded
        # from that model's own plotted points/axis range rather than
        # being plotted as a fabricated point at zero (V5.1 confirmed the
        # pre-ver4-25 blanket fillna(0) affected only this same class of
        # cross-level filler cell in practice; this removes the risk of
        # that ever silently changing rather than relying on it).
        _predicted_cols = [c for c in predicted_test_formatted.columns if c not in ('level', 'population')]
        _predicted_only_cols = [c for c in _predicted_cols if c not in _effect_model_cols]
        _effect_only_cols = [c for c in _effect_model_cols if c not in _predicted_cols]
        # ver4-31 R5: display-only rescaling of the marker effects ('rank' /
        # 'sqrt'). Pearson r below is computed from `effect_formatted`, i.e.
        # BEFORE this transform, so the labels and Scatter_plot_pearson.csv are
        # independent of `effect_scale`.
        _effect_for_scatter = (effect_formatted if opts['effect_scale'] == 'max' else
                               _with_display_effect_scale(effect_formatted, _effect_model_cols,
                                                          _eff_codes, opts['effect_scale']))
        data_scatter = pd.concat([predicted_test_formatted, _effect_for_scatter.iloc[:, 1:]])
        if _predicted_only_cols:
            _is_effect_row = data_scatter['level'] != 'phenotype'
            data_scatter.loc[_is_effect_row, _predicted_only_cols] = \
                data_scatter.loc[_is_effect_row, _predicted_only_cols].fillna(0)
        if _effect_only_cols:
            _is_phenotype_row = data_scatter['level'] == 'phenotype'
            data_scatter.loc[_is_phenotype_row, _effect_only_cols] = \
                data_scatter.loc[_is_phenotype_row, _effect_only_cols].fillna(0)
        
        #if 'ensemble' in list(data_scatter.columns):
        #    data_scatter = data_scatter.drop('ensemble',axis=1)
        
        # Pearson r +/- SE per model pair, within scenario then over
        # scenarios (see _pairwise_pearson). Phenotype level: across the
        # individuals of each scenario; marker-effect level: across the
        # markers of each scenario, on the same (normalised or raw) values
        # the panels plot.
        _pheno_codes = _scenario_codes(_phenotype_rows)
        pheno_r = _pairwise_pearson(predicted_test_formatted, model_selected, _pheno_codes)
        effect_r = (_pairwise_pearson(effect_formatted, model_selected, _eff_codes)
                    if _eff_codes is not None and effect_formatted.shape[0] == _eff_codes.size else {})
        pearson_rows.extend(_pearson_rows(PHENOTYPE[k], 'per_model', 'phenotype', pheno_r, model_selected))
        pearson_rows.extend(_pearson_rows(PHENOTYPE[k], 'per_model', 'marker_effect', effect_r, model_selected))

        # Set the matrix size, render and save - via the shared
        # _render_scatter_matrix() helper (Requirements.md item 1), so
        # this matrix and the new conv_vs_ML comparison matrix below can
        # never visually diverge. Byte-identical to the original inline
        # loop.
        _figsize_matrix = _panel_size_in(figsize, len(model_selected), opts['size_mode'])
        _pheno_actual = None
        if opts['pheno_hue'] == 'actual' and PHENOTYPE[k] in _actual_ranges:
            _pheno_actual = pd.to_numeric(_phenotype_rows['actual'], errors='coerce').to_numpy(dtype=float)
        _render_scatter_matrix(
            data_scatter, model_selected, markers,
            './Result/'+RESULT_NAME+'/Scatter_plot_'+PHENOTYPE[k]+'.png',
            _figsize_matrix, PLOT_DPI,
            color_by_population=COLOR_BY_POPULATION, population_palette=population_palette,
            normalize_effect=NORMALIZE_MARKER_EFFECT, pheno_r=pheno_r, effect_r=effect_r,
            font_scale=SCATTER_CONFIG['font_size'],
            opts=opts, pheno_actual=_pheno_actual, pheno_label=PHENOTYPE[k],
            actual_range=_actual_ranges.get(PHENOTYPE[k]), effect_caption=_effect_caption,
            extra_formats=extra_formats)

        # Requirements.md item 1: the conv_vs_ML comparison matrix - 2x2,
        # since there are exactly two groups - comparing the mean
        # predicted phenotype and mean (normalised) marker effect between
        # the conventional models and the machine-learning models,
        # alongside the current per-model matrix above. Only drawn when
        # both halves (predicted-phenotype AND marker-effect) actually
        # have at least one model from EACH group for this phenotype;
        # either half being unavailable already logged its own reason
        # above.
        if conv_vs_ML_predicted_formatted is not None and conv_vs_ML_effect_formatted is not None:
            _conv_vs_ML_names = ['Conventional', 'Machine learning']
            _conv_effect_for_scatter = (conv_vs_ML_effect_formatted if opts['effect_scale'] == 'max' else
                                        _with_display_effect_scale(conv_vs_ML_effect_formatted, _conv_vs_ML_names,
                                                                   _eff_codes, opts['effect_scale']))
            data_scatter_conv_vs_ML = pd.concat([
                conv_vs_ML_predicted_formatted[_conv_vs_ML_names + ['level']],
                _conv_effect_for_scatter[_conv_vs_ML_names + ['level']],
            ]).fillna(0)
            # Panel-size parity with the per-model matrix above: each
            # subplot in THIS 2x2 figure is rendered at the same size as
            # each subplot in the (len(model_selected) x len(model_selected))
            # matrix above, floored at 6in so a run with many individual
            # models doesn't shrink the conv_vs_ML comparison into
            # illegibility, and capped at the original `figsize` so it
            # never renders LARGER than the matrix it accompanies.
            _panel_size = figsize / max(len(model_selected), 1)
            _conv_vs_ML_figsize = min(max(_panel_size * 2, 6), figsize)
            if opts['size_mode'] == 'per_panel':
                _conv_vs_ML_figsize = 2 * (_figsize_matrix / max(len(model_selected), 1))
            _render_scatter_matrix(
                data_scatter_conv_vs_ML, _conv_vs_ML_names, markers,
                './Result/'+RESULT_NAME+'/Scatter_plot_conv_vs_ML_'+PHENOTYPE[k]+'.png',
                _conv_vs_ML_figsize, PLOT_DPI,
                normalize_effect=NORMALIZE_MARKER_EFFECT,
                pheno_r=_pairwise_pearson(conv_vs_ML_predicted_formatted, _conv_vs_ML_names, _pheno_codes),
                effect_r=(_pairwise_pearson(conv_vs_ML_effect_formatted, _conv_vs_ML_names, _eff_codes)
                          if _eff_codes is not None and conv_vs_ML_effect_formatted.shape[0] == _eff_codes.size
                          else {}),
                font_scale=SCATTER_CONFIG['font_size'],
                opts=opts, pheno_actual=_pheno_actual, pheno_label=PHENOTYPE[k],
                actual_range=_actual_ranges.get(PHENOTYPE[k]), effect_caption=_effect_caption,
                extra_formats=extra_formats)
        

    if pearson_rows:
        pd.DataFrame(pearson_rows).to_csv('./Result/'+RESULT_NAME+'/Scatter_plot_pearson.csv', index=False)
