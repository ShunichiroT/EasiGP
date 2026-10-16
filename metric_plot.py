import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as _pe
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch

from pipeline_utils import canonical_model_order, is_naive_ensemble_label
import plot_style as _ps

# Requirement: give the violin plots the same font/figure-size adjustment
# scatter_plot() already offers, for consistency between the two. Defaults
# match this function's previous hard-coded behaviour exactly (font_scale=1,
# FacetGrid's own default height=5), so any caller that doesn't pass
# METRIC_PLOT_CONFIG (e.g. an older/unreviewed script) keeps working
# unchanged.
_DEFAULT_METRIC_PLOT_CONFIG = {'font_size': 1, 'fig_size': 5}

# ver4-4 R7.h: PLOT_DPI replaces the previous hard-coded dpi=600 for both
# 'figure.dpi' and 'savefig.dpi' below. Default kept at 600 here (this
# function's own, legacy-preserving default if a caller omits the
# argument) - callers thread the actual PLOT_DPI config value through
# (default 300 there, per the blueprint's §4a reproducibility policy).



# --------------------------------------------------------------------------- #
# ver4-31 (R1 / R2 / R3): optional violin styles. EVERY key below is optional;
# an absent key gives the legacy figure (I11) - `style='legacy'` runs the
# literal historical `g.map(sns.violinplot, ...)` call.
# --------------------------------------------------------------------------- #

_STYLES = ('legacy', 'violin_points', 'box_points', 'auto')
_COLOUR_BY = ('model', 'group', 'auto')
_LEGEND_MODES = ('on', 'auto', 'off')
_BETWEEN_LABELS = ('test', 'pair')
_AUTO_WIDTH_PER_MODEL = 0.38      # inches of facet width per model (auto_width)


def _resolve_metric_options(cfg):
    """Validated ver4-31 options from a `METRIC_PLOT_CONFIG`-style dict."""
    c = cfg or {}

    def _pick(key, allowed, default):
        v = c.get(key, default)
        v = default if v is None else str(v).strip().lower()
        return v if v in allowed else default

    try:
        min_n = max(int(c.get('min_n_violin', 5)), 1)
    except (TypeError, ValueError):
        min_n = 5
    ref = c.get('reference')
    ref = str(ref).strip().lower() if ref else None
    return {
        'style': _pick('style', _STYLES, 'legacy'),
        'min_n_violin': min_n,
        'show_mean_se': bool(c.get('show_mean_se', False)),
        'colour_by': _pick('colour_by', _COLOUR_BY, 'model'),
        'auto_width': bool(c.get('auto_width', False)),
        'group_separators': bool(c.get('group_separators', False)),
        'reference': ref if ref == 'ensemble' else None,
        'legend': _pick('legend', _LEGEND_MODES, 'on'),
        'between_label': _pick('between_label', _BETWEEN_LABELS, 'test'),
        'cut': _cut_of(c.get('cut', 0)),
        # Median value printed above every violin (off = the historical figure, I11).
        'show_median': bool(c.get('show_median', False)),
        'median_decimals': _decimals_of(c.get('median_decimals', 'auto')),
    }


def _cut_of(v):
    """Violin tail extension in bandwidths (seaborn's `cut`): 0 clips each violin
    at its data range, 2 is seaborn's classic look. Bad values -> 0."""
    try:
        return min(max(float(v), 0.0), 5.0)
    except (TypeError, ValueError):
        return 0.0


def _decimals_of(v):
    """Decimals of the printed median: ``None`` = choose from the data ('auto'),
    otherwise an integer 0-6. Bad values -> auto."""
    if v is None or (isinstance(v, str) and v.strip().lower() in ('', 'auto')):
        return None
    try:
        return min(max(int(v), 0), 6)
    except (TypeError, ValueError):
        return None


def _auto_decimals(medians):
    """Decimals that resolve the differences between the plotted medians: about 30
    distinguishable steps across their spread (0.17 -> 3 decimals, 0.3 -> 2, 5 -> 1,
    100 -> 0), clipped to 0-4. One value (or none) falls back on 10% of its size."""
    m = np.asarray([x for x in medians if np.isfinite(x)], dtype=float)
    if m.size == 0:
        return 2
    spread = float(m.max() - m.min())
    if spread <= 0:
        spread = max(abs(float(m[0])) * 0.1, 1e-3)
    return int(min(max(np.ceil(np.log10(30.0 / spread)), 0), 4))


def _violin_tops(ax):
    """``{x position: highest point of the violin drawn there}`` read from the axis'
    violin bodies (seaborn draws each one as a PolyCollection), so a label can sit above a
    tail that runs past the data (``cut`` > 0), not only above the largest value."""
    tops = {}
    for coll in ax.collections:
        if not isinstance(coll, PolyCollection):
            continue
        for path in coll.get_paths():
            v = np.asarray(path.vertices, dtype=float)
            if v.size == 0 or not np.isfinite(v).all():
                continue
            pos = int(round(float((v[:, 0].min() + v[:, 0].max()) / 2.0)))
            tops[pos] = max(tops.get(pos, -np.inf), float(v[:, 1].max()))
    return tops


def _annotate_medians(g, metric, order, opts):
    """Print each violin's median just above it (``show_median``).

    Why it is laid out this way: the value sits above the violin's own top (tail included),
    in a small dark font with a white halo so it stays readable over neighbouring violins,
    gridlines and the points; its decimals follow the spread of the medians (or
    ``median_decimals``); when a label would be wider than its slot the font shrinks to 6 pt
    and then the label is turned vertical, so neighbours never overprint; the upper y limit is
    raised so no label is clipped. Medians are taken over exactly the values drawn in that
    panel. Never raises: a panel it cannot read is left without labels."""
    pos_of = {m: i for i, m in enumerate(order)}
    panels = []
    for (ri, ci, _hi), fdata in g.facet_data():
        if not len(fdata):
            continue
        d = fdata[['x', metric]].copy()
        d[metric] = pd.to_numeric(d[metric], errors='coerce')
        d = d[np.isfinite(d[metric].to_numpy(dtype=float))]
        if d.empty:
            continue
        meds = d.groupby('x')[metric].median()
        mx = d.groupby('x')[metric].max()
        panels.append((g.axes[ri, ci], meds, mx))
    if not panels:
        return
    decimals = opts.get('median_decimals')
    if decimals is None:
        decimals = _auto_decimals([v for _, meds, _ in panels for v in meds.to_numpy(dtype=float)])
    fig = g.figure
    base_fs = float(min(max(plt.rcParams.get('font.size', 10.0) * 0.9, 7.0), 11.0))
    halo = [_pe.withStroke(linewidth=2.4, foreground='white')]
    for ax, meds, mx in panels:
        try:
            tops = _violin_tops(ax)
            n_slots = max(len(order), 1)
            width_in = ax.get_position().width * fig.get_size_inches()[0]
            slot_in = width_in / n_slots
            texts = {m: f"{v:.{decimals}f}" for m, v in meds.items() if m in pos_of}
            longest = max((len(t) for t in texts.values()), default=1)
            fs, vertical = base_fs, False
            # an average digit is about 0.6 em wide; leave 8% of the slot as a gap
            while fs - 0.5 >= 6.0 and longest * fs * 0.6 / 72.0 > slot_in * 0.92:
                fs -= 0.5
            if longest * fs * 0.6 / 72.0 > slot_in * 0.92:
                vertical = True
            anchor_top = -np.inf
            ymin = ax.get_ylim()[0]
            for m, text in texts.items():
                pos = pos_of[m]
                top = max(float(mx[m]), tops.get(pos, -np.inf))
                anchor_top = max(anchor_top, top)
                extra = {'rotation': 90, 'rotation_mode': 'anchor'} if vertical else {}
                ax.annotate(
                    text, xy=(pos, top), xytext=(0, 2.5), textcoords='offset points',
                    ha='center', va='bottom', fontsize=fs, color='0.1', path_effects=halo,
                    zorder=9, annotation_clip=False, **extra)
            # headroom: the tallest label (a vertical one is as tall as it is long)
            label_in = (longest * fs * 0.6 / 72.0 if vertical else fs * 1.25 / 72.0) + 2.5 / 72.0
            axis_h_in = max(ax.get_position().height * fig.get_size_inches()[1], 1e-6)
            frac = min(label_in / axis_h_in, 0.4)
            if np.isfinite(anchor_top) and anchor_top > ymin:
                needed = ymin + (anchor_top - ymin) / (1.0 - frac)
                cur_lo, cur_hi = ax.get_ylim()
                if needed > cur_hi:
                    ax.set_ylim(cur_lo, needed)
        except Exception as exc:  # noqa: BLE001 - a label must never cost the figure
            print(f"[metric_plot] {metric}: median labels skipped for one panel "
                  f"({type(exc).__name__}: {exc}).")


def _mean_se(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float('nan'), float('nan')
    se = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0
    return float(v.mean()), se


def _draw_metric_cell(xname, yname, hname, *, data, order, palette, style, min_n, show_mean_se,
                       skipped, cut=0.0, color=None, label=None):
    """One facet cell of the improved violin figure (R1 / R2), drawn via
    `FacetGrid.map_dataframe`. `xname == hname == 'model'`-valued columns as in
    the legacy call, so positions follow `order` exactly.

    Per model: `auto` draws the CLASSIC violin (seaborn's area-normalised
    violin with its inner box - the look of the legacy figure; `cut` sets how far
    the tails extend past the data, 0 = clipped) for every model with at least
    `min_n` values, and a box + the raw points for the others - no points are
    overlaid on a violin. `violin_points` = violin (no inner box) + jittered
    points + a white-edged median marker; `box_points` = box (no fliers) +
    points. `show_mean_se` adds a mean +- SE bar."""
    ax = plt.gca()
    d = data[[xname, yname, hname]].copy()
    d[yname] = pd.to_numeric(d[yname], errors='coerce')
    d = d[np.isfinite(d[yname].to_numpy(dtype=float))]
    if d.empty:
        skipped.append(1)
        return
    counts = d.groupby(xname)[yname].count()
    if style == 'violin_points':
        violin_models = list(counts.index)
    elif style == 'auto':
        violin_models = [m for m in counts.index if counts[m] >= min_n]
    else:
        violin_models = []
    box_models = [m for m in counts.index if m not in violin_models]
    common = dict(x=xname, y=yname, hue=hname, order=order, hue_order=order, palette=palette,
                  ax=ax, legend=False)
    if violin_models:
        sub = d[d[xname].isin(violin_models)]
        if style == 'auto':
            sns.violinplot(data=sub, cut=cut, **common)      # classic look (inner box, area-normalised)
        else:
            sns.violinplot(data=sub, cut=cut, inner=None, density_norm='width', **common)
    if box_models:
        sub = d[d[xname].isin(box_models)]
        sns.boxplot(data=sub, showfliers=False, width=0.6, **common)
    points_for = list(counts.index) if style != 'auto' else box_models
    if points_for:
        sns.stripplot(data=d[d[xname].isin(points_for)], x=xname, y=yname, order=order, jitter=0.15,
                      size=3, color='0.2', alpha=0.7, ax=ax, legend=False)
    pos = {m: i for i, m in enumerate(order)}
    if style != 'auto':
        for m in violin_models:
            vals = d.loc[d[xname] == m, yname].to_numpy(dtype=float)
            ax.scatter([pos[m]], [np.median(vals)], s=26, c='white', edgecolors='0.1',
                       linewidths=0.9, zorder=6)
    if show_mean_se:
        for m in counts.index:
            mean, se = _mean_se(d.loc[d[xname] == m, yname])
            if np.isfinite(mean):
                ax.errorbar([pos[m]], [mean], yerr=[se], fmt='D', color='0.05', markersize=3.5,
                            elinewidth=1.2, capsize=2.5, zorder=7)


def _decorate_axis(ax, data, xname, yname, separators, reference_label):
    """Group separators (dotted verticals at display-group boundaries) and the
    dashed naive-ensemble-median reference line (R2), for either style."""
    if separators:
        for b in separators:
            ax.axvline(b, color='0.55', linestyle=':', linewidth=0.9, zorder=0.5)
    if reference_label is not None:
        ref = pd.to_numeric(data.loc[data[xname] == reference_label, yname], errors='coerce').dropna()
        if len(ref):
            ax.axhline(float(ref.median()), color='0.35', linestyle='--', linewidth=0.9, zorder=0.5)


def _metric_palette(order, colour_by):
    """`{model: colour}` per `colour_by` ('model' keeps the seaborn
    'colorblind' colours of the legacy figure, 'group' / 'auto' use the
    display-group ramps of `plot_style`)."""
    mode = {'model': 'legacy', 'group': 'group', 'auto': 'auto'}[colour_by]
    styles = _ps.model_style_map(order, mode=mode)
    return {m: styles[m]['color'] for m in order}


def _legend_handles(order, palette, legend, colour_by):
    """Legend entries for the improved styles; `None` = draw no legend."""
    if legend == 'off':
        return None
    if legend == 'on':
        return {m: Patch(facecolor=palette[m], edgecolor='0.2', linewidth=0.5) for m in order}
    grouped = colour_by == 'group' or (colour_by == 'auto' and len(order) > _ps.LEGACY_MAX_LABELS)
    if not grouped:
        return None            # 'auto': the x ticks already name every model
    if grouped:
        # Legend swatch = the group's middle member's colour (violins use
        # shades of the group hue, so one swatch stands for the ramp).
        out = {}
        for g in _ps.groups_present(order):
            members = [m for m in order if _ps.display_group(m) == g]
            out[_ps.GROUP_LABELS.get(g, g) + ' (shades = models)'] = Patch(
                facecolor=palette[members[len(members) // 2]], edgecolor='0.2', linewidth=0.5)
        return out
    return None


def _render_grid(record, MODEL, metric, ax_share, fig_size, path, *, row, opts, palette=None,
                  separators=None, reference_label=None, aspect=None, axis_hook=None,
                  legend_handles=None):
    """Build, decorate and save ONE FacetGrid figure (`row='population'` for
    the per-population figure, `None` for the pooled `_total` one). With
    `opts['style'] == 'legacy'` the sequence of calls is the historical one.

    ver4-54 (R2): two optional hooks let `result_comparison.py` (Tool mode,
    cross-run comparison) reuse this exact drawing path instead of copying it.
    `axis_hook(axis)` is called once per axis after the tick rotation and
    before `tight_layout` (it draws the second x-axis tier / replaces the tick
    text). A non-`None` `legend_handles` (`{label: Patch}`) replaces BOTH
    legend branches below (an empty dict draws no legend). With both left at
    `None` - every existing caller - the sequence of calls is unchanged (I11)."""
    kw = {}
    if row is not None:
        kw['row'] = row
    if aspect is not None:
        kw['aspect'] = aspect
    g = sns.FacetGrid(record, col="phenotype", sharey=ax_share, height=fig_size, **kw)

    for axis in g.axes.flat:
        axis.tick_params(labelleft=True)

    if opts['style'] == 'legacy':
        # Requirements.md item 6: `order=MODEL` places each model's own
        # violin at its own x position in the SAME canonical order
        # `hue_order=MODEL` already colours them in, so x-axis position
        # and colour/legend never disagree on model order.
        g.map(sns.violinplot, 'x', metric, 'model',
              palette='colorblind' if opts['colour_by'] == 'model' else palette,
              hue_order=MODEL, order=MODEL)
    else:
        skipped = []
        g.map_dataframe(_draw_metric_cell, 'x', metric, 'model', order=list(MODEL), palette=palette,
                        style=opts['style'], min_n=opts['min_n_violin'],
                        show_mean_se=opts['show_mean_se'], skipped=skipped, cut=opts['cut'])
        if skipped:
            print(f"[metric_plot] {metric}: {len(skipped)} empty cell(s) (no finite values) skipped "
                  f"in {path}.")
    if separators or reference_label is not None:
        for (ri, ci, _hi), fdata in g.facet_data():
            if len(fdata):
                _decorate_axis(g.axes[ri, ci], fdata, 'x', metric, separators, reference_label)

    shown = opts.get('_labels') or {}      # ver4-95: {label: text to draw}; {} = unchanged
    for axis in g.axes.flat:
        axis.set_ylabel(metric)
        axis.set_xlabel("")
        # Requirements.md item 6: each violin's own x tick now reads
        # the model's real name (record['x'] = record['model'], set
        # above) instead of the previous single shared 'Model' tick -
        # rotated so several/long model names don't overlap.
        # Requirements_2.md item 2: `rotation_mode='anchor'` (paired
        # with `ha='right'`) keeps labels of different lengths at an even
        # offset from their own tick; `rotation=45` widens the gap.
        plt.setp(axis.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        if shown and any(k != v for k, v in shown.items()):
            # ver4-95: MODEL_LABEL_STYLE - the tick TEXT only; grouping, colours and order above
            # all used the full label.
            _txt = [shown.get(t.get_text(), t.get_text()) for t in axis.get_xticklabels()]
            axis.set_xticks(axis.get_xticks())       # fix the locator so set_xticklabels is exact
            axis.set_xticklabels(_txt,
                                 rotation=45, ha='right', rotation_mode='anchor')
        if axis_hook is not None:
            axis_hook(axis)

    plt.tight_layout()
    if legend_handles is not None:
        if legend_handles:
            g.add_legend(legend_data=legend_handles, loc='lower right')
    elif opts['legend'] == 'on' and opts['style'] == 'legacy':
        g.add_legend(loc='lower right')
    else:
        handles = _legend_handles(list(MODEL), palette, opts['legend'], opts['colour_by'])
        if handles:
            g.add_legend(legend_data=handles, loc='lower right')
    if shown and getattr(g, '_legend', None) is not None:
        _ps.relabel_texts(g._legend.get_texts(), shown)
    if opts.get('show_median'):
        # after the legend and tight_layout, so the panels have their final width (the label
        # size is chosen from the width of one violin's slot)
        _annotate_medians(g, metric, list(MODEL), opts)
    _ps.save_figure(g.figure, path, bbox_inches='tight',      # = FacetGrid.savefig's own default
                    extra_formats=_ps.extra_formats_of(opts.get('_cfg')))
    # Multi-option runs call this once per option; close the figure so open
    # figures never accumulate (matplotlib warns beyond 20).
    plt.close(g.figure)


@_ps.themed('ticks', lambda a: (a['METRIC_PLOT_CONFIG'] or _DEFAULT_METRIC_PLOT_CONFIG).get('font_size', _DEFAULT_METRIC_PLOT_CONFIG['font_size']),
            lambda a: a['PLOT_DPI'])
def metric_plot(record, MODEL, RESULT_NAME, SCENARIO, METRIC_PLOT_CONFIG=None, PLOT_DPI=600,
                 metrics=None):

    # Requirements.md (BLUEs/BLUPs accuracy-metrics phase), item 8 -
    # `metrics` (default `None` -> the original, hardcoded
    # `['Pearson correlation', 'MSE']`, byte-identical to every existing
    # caller) lets `accuracy_metrics_plot.py` reuse this EXACT same
    # violin-plotting strategy ("Apply the same strategy used to
    # visualise the current Pearson correlation and MSE, providing the
    # same GUI field options") for 'Adjusted prediction accuracy' and
    # 'Spearman correlation' too - one call per metric, each with its own
    # METRIC_PLOT_CONFIG/PLOT_DPI, producing '<metric>.png'/
    # '<metric>_total.png' exactly as this function already does.

    # The naive ensemble ('ensemble', or 'ensemble__<algo>' when tuned
    # per-method) is computed once, in GP()'s finalisation step, strictly
    # AFTER the per-task loop that produces every weighted-ensemble
    # ('Nelder Mead' etc.) row - so any caller deriving MODEL straight
    # from Metric.csv's own row order (e.g. the HP_TUNE fallback both
    # headless runners use) would otherwise place it after every
    # weighted-ensemble method's violin instead of before, as intended.
    # Applied here too - not just at the call sites - so this function's
    # own `hue_order` is always correct regardless of how the caller built
    # MODEL. A no-op whenever MODEL doesn't contain both an ensemble-family
    # and a weighted-ensemble-family label - see canonical_model_order()'s
    # own docstring.
    MODEL = canonical_model_order(MODEL)

    if METRIC_PLOT_CONFIG is None:
        METRIC_PLOT_CONFIG = _DEFAULT_METRIC_PLOT_CONFIG
    font_size = METRIC_PLOT_CONFIG.get('font_size', _DEFAULT_METRIC_PLOT_CONFIG['font_size'])
    fig_size = METRIC_PLOT_CONFIG.get('fig_size', _DEFAULT_METRIC_PLOT_CONFIG['fig_size'])
    opts = _resolve_metric_options(METRIC_PLOT_CONFIG)
    opts['_cfg'] = METRIC_PLOT_CONFIG          # carries the run's `_extra_formats` (R15)
    opts['_labels'] = _ps.shown_labels(list(MODEL), METRIC_PLOT_CONFIG)   # ver4-95 (display only)

    record = record.reset_index(drop=True)
    metrics = ['Pearson correlation', 'MSE'] if metrics is None else list(metrics)
    # Requirements.md item 6: the x-axis used to be one constant label
    # ('Model') shared by every violin, with the actual model identified
    # only via colour/the legend. Using each row's own model name as the
    # x position instead gives every violin its own labelled tick - e.g.
    # rrBLUP/RF/Bayesian weighted ensemble each get their own name printed
    # at the bottom of their own violin - while `order=MODEL`/`hue_order=
    # MODEL` below keep them left-to-right in the same canonical order the
    # legend already used, and `hue='model'` keeps each violin coloured
    # exactly as before (one colour per model, same legend).
    record['x'] = record['model']

    if SCENARIO == 'between':
        # ver4-31 R3: `between_label='test'` (default) keeps only the TEST
        # population of 'X->Y' (so 'A->C' and 'B->C' share the facet 'C');
        # 'pair' keeps the full 'X->Y' label - one facet per distinct pair.
        if opts['between_label'] == 'test':
            record['population'] = record['population'].str.split('->', expand=True).iloc[:,-1]

    # ver4-31 R16: the ticks theme is applied by the @_ps.themed decorator (scoped to this call).

    palette = separators = reference_label = aspect = None
    if opts['style'] != 'legacy' or opts['colour_by'] != 'model' or opts['auto_width'] \
            or opts['group_separators'] or opts['reference'] or opts['legend'] != 'on' \
            or opts['between_label'] != 'test':
        # Improved-figure settings resolved ONCE per call (and logged, so the
        # run log shows whether the legacy or the improved figure was drawn).
        palette = _metric_palette(list(MODEL), opts['colour_by'])
        if opts['group_separators']:
            separators = _ps.group_breaks(MODEL) or None
        if opts['reference'] == 'ensemble':
            naive = [m for m in MODEL if is_naive_ensemble_label(m)]
            if naive:
                reference_label = naive[0]
            else:
                print("[metric_plot] reference='ensemble' requested but MODEL has no naive "
                      "ensemble - no reference line drawn.")
        if opts['auto_width']:
            aspect = max(1.0, len(MODEL) * _AUTO_WIDTH_PER_MODEL / float(fig_size))
        _ps.log_resolved('metric_plot', **{k: v for k, v in opts.items() if not k.startswith('_')})

    for i in range(len(metrics)):
        ax_share = False if metrics[i] == 'MSE' else True
        base = './Result/' + RESULT_NAME + '/' + metrics[i]
        _render_grid(record, MODEL, metrics[i], ax_share, fig_size, base + '.png', row='population',
                     opts=opts, palette=palette, separators=separators,
                     reference_label=reference_label, aspect=aspect)
        #======#
        _render_grid(record, MODEL, metrics[i], ax_share, fig_size, base + '_total.png', row=None,
                     opts=opts, palette=palette, separators=separators,
                     reference_label=reference_label, aspect=aspect)
