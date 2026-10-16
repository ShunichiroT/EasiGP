"""
plot_style.py
=============
Update ID ver4-31-VIS, R0 (Design Blueprint `EasiGP_ver4-31_Design_Blueprint.md`)
- the shared PLOT-STYLE AUTHORITY for the four visualisation families
(violin `metric_plot.py`, Top-k coincidence curve and Czekanowski matrix in
`accuracy_metrics_plot.py`, scatter matrix `scatter_plot.py`, circos
`circos_plot.py`).

Layer 3d: a pure, import-light module that sits BELOW every plot module and
imports none of them (invariant I2 - no upward import; an import-order test
in the ver4-31 checks pins this). It owns the facts the plot modules used to
re-derive independently and let drift:

  * which DISPLAY GROUP a model label belongs to (conventional /
    machine-learning / graph / naive ensemble / weighted ensemble);
  * the colour, line style and marker a model gets - keyed on the model's
    position in the run's FULL, canonically-ordered `MODEL` list, never on
    whichever subset a particular plot happens to draw, so the SAME model
    has the SAME colour in the violin, the coincidence curve and every other
    plot of a run;
  * the figure-save routine (PNG exactly as before, plus optional vector
    formats - R15);
  * the diagonal-label fitting helpers `scatter_plot.py` used to hold a
    verbatim copy of "to avoid a circular import" (R0);
  * the row-thinning helper (`thin_rows`, R4) `accuracy_metrics_plot.py`'s
    Czekanowski matrix and the scatter matrix now share;
  * a scoped seaborn/matplotlib theme context (R16).

Everything here degrades rather than raises on an unrecognised label: an
unknown model is a `'machine_learning'` model (the same contract as
`model_registry.model_family`).

Nothing in this module changes a legacy render: every function has a
`mode='legacy'`/no-op default that reproduces the ver4-30 behaviour exactly
(I11); the plot modules only reach for the new behaviour when their own
config keys ask for it.
"""

from __future__ import annotations

import os
import re
import textwrap
from contextlib import contextmanager
from typing import Sequence

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from pipeline_utils import (canonical_model_order, is_naive_ensemble_label, is_wopt_label,
                            display_label_map, normalise_label_style)
from model_registry import model_family


# --------------------------------------------------------------------------- #
# Display groups
# --------------------------------------------------------------------------- #

GROUP_ORDER = ('conventional', 'machine_learning', 'graph', 'naive_ensemble', 'weighted_ensemble')

GROUP_LABELS = {
    'conventional': 'Conventional',
    'machine_learning': 'Machine learning',
    'graph': 'Graph (GAT)',
    'naive_ensemble': 'Naive ensemble',
    'weighted_ensemble': 'Weighted ensemble',
}

# One colour per display group (Okabe-Ito, colour-vision-deficiency safe) -
# used when a plot colours by GROUP rather than by model.
GROUP_COLOURS = {
    'conventional': '#0072B2',
    'machine_learning': '#009E73',
    'graph': '#CC79A7',
    'naive_ensemble': '#000000',
    'weighted_ensemble': '#D55E00',
}

# Per-model ramp colour maps, sampled at evenly spaced lightness within each
# group (the `weight_plot.py` precedent) - used when there are more model
# labels than the legacy 10-colour palette can keep distinct.
_GROUP_CMAPS = {
    'conventional': 'Blues',
    'machine_learning': 'Greens',
    'graph': 'Purples',
    'naive_ensemble': 'Greys',
    'weighted_ensemble': 'Oranges',
}

# Prediction-column spellings of two weighted-ensemble methods (see the
# 'Nelder-Mead' vs 'Nelder Mead' note in the architecture doc, S12.2). Plot
# code that works from Prediction_result_test.csv column names, rather than
# from Metric.csv `model` labels, can hand these to `display_group()` too.
_WOPT_COLUMN_ALIASES = frozenset({'Nelder-Mead', 'Bayesian'})

_TUNE_SUFFIX_SEP = '__'
_BIO_PRIOR_PREFIX = 'GAT_biological_prior_knowledge_'

LEGACY_MAX_LABELS = 10       # seaborn 'colorblind' has 10 colours

_LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]
_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']


def _base_name(label: str) -> str:
    """`models.hyperparameter_tuning.base_of()` + `schema_key_of()` semantics,
    re-implemented as plain string rules so this module never imports the
    `models` package (I2; the same "duplicate a one-line helper rather than
    add a cross-branch import" choice `scatter_plot._base_model_name` made).
    A parity test in the ver4-31 checks compares this against the originals."""
    base = str(label).split(_TUNE_SUFFIX_SEP, 1)[0]
    if base.startswith(_BIO_PRIOR_PREFIX) and base[len(_BIO_PRIOR_PREFIX):].isdigit():
        return 'GAT_biological_prior_knowledge'
    return base


def display_group(label) -> str:
    """One of `GROUP_ORDER` for a model label as it appears in `Metric.csv`
    (`'RF'`, `'RF__Grid'`, `'ensemble__Bayesian'`, `'Nelder Mead'`,
    `'Joint Bayesian optimisation'`, `'GAT_biological_prior_knowledge_2'`).
    Never raises: an unknown label is `'machine_learning'`."""
    s = str(label)
    if is_naive_ensemble_label(s):
        return 'naive_ensemble'
    if is_wopt_label(s) or s in _WOPT_COLUMN_ALIASES:
        return 'weighted_ensemble'
    base = _base_name(s)
    if base.startswith('GAT_'):
        return 'graph'
    if model_family(base) == 'conventional':
        return 'conventional'
    return 'machine_learning'


def group_breaks(labels: Sequence[str]) -> list:
    """Positions (`i - 0.5`) between adjacent labels that belong to different
    display groups - where a plot draws its group separator."""
    labels = list(labels)
    out = []
    for i in range(1, len(labels)):
        if display_group(labels[i]) != display_group(labels[i - 1]):
            out.append(i - 0.5)
    return out


def groups_present(labels: Sequence[str]) -> list:
    """The display groups actually present in `labels`, in `GROUP_ORDER`."""
    present = {display_group(m) for m in labels}
    return [g for g in GROUP_ORDER if g in present]


def result_colour_map(labels: Sequence[str]) -> dict:
    """`{label: (r, g, b)}` giving every RESULT a colour - ver4-54, R2 ("Tool"
    mode's cross-run comparison colours each violin by the result it came from,
    not by its model).

    Why it lives here: this module is the single authority for plot colours
    (I3); `result_comparison.py` must not choose its own palette.

    Up to `LEGACY_MAX_LABELS` (10) results use seaborn `'colorblind'` - the same
    palette every other EasiGP violin uses; beyond that the 10 colours would
    repeat, so a `'husl'` palette with one evenly spaced hue per result is used
    so all colours stay pairwise distinct. Colours follow the INPUT order and
    never depend on anything else (deterministic). Duplicate labels share one
    colour; an empty input gives `{}`."""
    unique = list(dict.fromkeys(labels))
    n = len(unique)
    if n == 0:
        return {}
    if n <= LEGACY_MAX_LABELS:
        palette = sns.color_palette('colorblind', n_colors=n)
    else:
        palette = sns.color_palette('husl', n_colors=n)
    return {label: tuple(palette[i]) for i, label in enumerate(unique)}


# --------------------------------------------------------------------------- #
# Model style map (colour / line style / marker)
# --------------------------------------------------------------------------- #

def _ramp_colours(labels: Sequence[str]) -> dict:
    by_group = {g: [] for g in GROUP_ORDER}
    for m in labels:
        by_group[display_group(m)].append(m)
    out = {}
    for g, members in by_group.items():
        if not members:
            continue
        cmap = matplotlib.colormaps[_GROUP_CMAPS[g]]
        k = len(members)
        if g == 'naive_ensemble':
            positions = [1.0] if k == 1 else np.linspace(1.0, 0.55, k)
        elif k == 1:
            positions = [0.65]
        else:
            positions = np.linspace(0.45, 0.95, k)
        for m, pos in zip(members, positions):
            out[m] = tuple(cmap(float(pos))[:3])
    return out


def model_style_map(models: Sequence[str], *, mode: str = 'legacy') -> dict:
    """`{label: {'color', 'linestyle', 'marker', 'group'}}` for the FULL list
    of model labels of a run.

    `models` is passed through `canonical_model_order` (the ordering
    authority every plot and summary already defers to), which places the
    naive-ensemble labels just before the weighted-ensemble ones and keeps
    the caller's order otherwise - so the styles depend on the run's model
    list in its usual order (the legacy palette has always been positional).
    What matters is that the caller passes the run's FULL list, not the
    subset a particular plot draws, and then simply indexes the result with
    the labels it needs - that is what makes a model's colour, line style
    and marker identical in every plot of a run (R0).

    mode
      'legacy'  seaborn `'colorblind'`, `n_colors=len(models)`, indexed by
                position (cycling past 10) - the exact palette
                `metric_plot.py` and `accuracy_metrics_plot._model_style_map`
                have always used, so a run with <= 10 labels is unchanged.
      'auto'    'legacy' for <= `LEGACY_MAX_LABELS` labels, else 'group'.
      'group'   per-model ramps inside each display group (conventional
                Blues, machine-learning Greens, graph Purples, naive
                ensemble black/greys, weighted ensemble Oranges), each ramp
                sampled at evenly spaced lightness - no two groups share a
                hue, and every label keeps its own shade.

    Line style and marker cycle by position in every mode, so identity never
    rests on colour alone."""
    if mode not in ('legacy', 'auto', 'group'):
        raise ValueError(f"model_style_map: mode must be 'legacy', 'auto' or 'group' (got {mode!r}).")
    labels = list(dict.fromkeys(canonical_model_order(list(models))))
    n = len(labels)
    use_ramp = mode == 'group' or (mode == 'auto' and n > LEGACY_MAX_LABELS)
    if use_ramp:
        colours = _ramp_colours(labels)
    else:
        palette = sns.color_palette('colorblind', n_colors=max(n, 1))
        colours = {m: palette[i % len(palette)] for i, m in enumerate(labels)}
    return {
        m: {
            'color': colours[m],
            'linestyle': _LINESTYLES[i % len(_LINESTYLES)],
            'marker': _MARKERS[i % len(_MARKERS)],
            'group': display_group(m),
        }
        for i, m in enumerate(labels)
    }


# --------------------------------------------------------------------------- #
# Extra vector formats (R15) and figure saving
# --------------------------------------------------------------------------- #

ALLOWED_EXTRA_FORMATS = ('svg', 'pdf')
EXTRA_FORMATS_KEY = '_extra_formats'      # injected into a plot config dict


def normalise_extra_formats(value) -> tuple:
    """Validated tuple of `ALLOWED_EXTRA_FORMATS` from a config value (list,
    tuple, comma string, `None`); an unknown entry is dropped, never raised
    on (a hand-edited config must not abort a finished run's plotting)."""
    if not value:
        return ()
    if isinstance(value, str):
        value = [v for v in re.split(r'[,\s]+', value) if v]
    out = []
    for v in value:
        f = str(v).strip().lower().lstrip('.')
        if f in ALLOWED_EXTRA_FORMATS and f not in out:
            out.append(f)
    return tuple(out)


LABEL_STYLE_KEY = '_label_style'          # injected into a plot config dict (ver4-95)


def with_extra_formats(plot_config, cfg):
    """A plot-config dict carrying the run-level plot options - `PLOT_EXTRA_FORMATS` under
    `_extra_formats` and (ver4-95) `MODEL_LABEL_STYLE` under `_label_style` - so no plot function
    signature has to change. When the run asks for neither (no extra format, label style 'full')
    the ORIGINAL object is returned untouched (`None` stays `None`) - the legacy call is then
    literally unchanged."""
    fmts = normalise_extra_formats((cfg or {}).get('PLOT_EXTRA_FORMATS'))
    style = normalise_label_style((cfg or {}).get('MODEL_LABEL_STYLE'))
    if not fmts and style == 'full':
        return plot_config
    out = dict(plot_config or {})
    if fmts:
        out[EXTRA_FORMATS_KEY] = list(fmts)
    if style != 'full':
        out[LABEL_STYLE_KEY] = style
    return out


def extra_formats_of(plot_config) -> tuple:
    """The extra formats a plot config carries (`()` for none / `None`)."""
    return normalise_extra_formats((plot_config or {}).get(EXTRA_FORMATS_KEY))


def label_style_of(plot_config) -> str:
    """The model-label style a plot config carries ('full' for none / `None`)."""
    return normalise_label_style((plot_config or {}).get(LABEL_STYLE_KEY))


def shown_labels(labels, plot_config=None, style=None) -> dict:
    """`{label: text to draw}` for `labels` under the plot config's label style (or an explicit
    `style`). Logic (grouping, colours, order) always keeps the full label; only the text that is
    drawn goes through this map."""
    return display_label_map(labels, label_style_of(plot_config) if style is None else style)


def relabel_texts(texts, mapping) -> None:
    """Replace the string of each matplotlib Text in `texts` that is a key of `mapping` (others are
    left alone). Used on legend entries, which the plotting libraries build from the raw labels."""
    for t in texts or ():
        try:
            cur = t.get_text()
            if cur in mapping and mapping[cur] != cur:
                t.set_text(mapping[cur])
        except Exception:                                  # pragma: no cover - defensive
            pass


def relabel_figure(fig, mapping) -> int:
    """Replace the SHORTENED model labels in everything `fig` already shows (ver4-95).

    `mapping` is a `{full label: shown text}` dict (see `shown_labels`); only entries whose text
    differs matter. Every Text artist (titles, legend entries, annotations, axis labels) has each
    full label found inside it replaced (longest label first, so ``'Super learner__X__Grid'`` is
    handled before ``'Super learner__X'``); tick labels are rewritten through `set_ticklabels`
    because matplotlib regenerates a tick Text from its formatter on every draw. Call it BEFORE
    `tight_layout` so the layout sees the final, shorter text. Returns the number of strings
    changed; 0 (and no work) when nothing is shortened. Never raises."""
    pairs = sorted(((k, v) for k, v in (mapping or {}).items() if k != v),
                   key=lambda kv: -len(kv[0]))
    if not pairs:
        return 0

    def fix(text):
        for k, v in pairs:
            if k in text:
                text = text.replace(k, v)
        return text

    changed = 0
    try:
        from matplotlib.text import Text
        from matplotlib.ticker import FixedLocator
        for ax in list(fig.axes):
            for axis in (ax.xaxis, ax.yaxis):
                try:
                    if not isinstance(axis.get_major_locator(), FixedLocator):
                        continue
                    ticks = axis.get_ticklabels()
                    old = [t.get_text() for t in ticks]
                    new = [fix(o) for o in old]
                    if new != old:
                        kw = {}
                        if ticks:
                            kw = {'rotation': ticks[0].get_rotation(),
                                  'ha': ticks[0].get_ha(), 'va': ticks[0].get_va(),
                                  'rotation_mode': ticks[0].get_rotation_mode()}
                        axis.set_ticklabels(new, **kw)
                        changed += sum(a != b for a, b in zip(old, new))
                except Exception:                          # pragma: no cover - defensive
                    pass
        tick_texts = set()
        for ax in list(fig.axes):
            for axis in (ax.xaxis, ax.yaxis):
                tick_texts.update(id(t) for t in axis.get_ticklabels())
        for t in fig.findobj(Text):
            if id(t) in tick_texts:
                continue
            cur = t.get_text()
            if cur:
                new = fix(cur)
                if new != cur:
                    t.set_text(new)
                    changed += 1
    except Exception:                                      # pragma: no cover - defensive
        pass
    return changed


_DENSE_POINTS = 2000       # scatter PathCollection with more points is rasterised
_DENSE_PATHS = 500         # PatchCollection / PolyCollection with more paths is


def _rasterise_dense_layers(fig, min_points=_DENSE_POINTS, min_paths=_DENSE_PATHS):
    """Mark dense collections (thousands of scatter points, hexbin cells,
    circos link patches) as rasterised so an SVG/PDF stays a reasonable size;
    text, axes and sparse artists stay vector. Returns the list of artists
    changed so the caller can restore them."""
    from matplotlib.collections import PathCollection, PatchCollection, PolyCollection, QuadMesh
    changed = []
    for ax in fig.axes:
        for coll in list(ax.collections):
            try:
                if isinstance(coll, PathCollection):
                    dense = len(coll.get_offsets()) > min_points
                elif isinstance(coll, QuadMesh):
                    dense = True
                elif isinstance(coll, (PatchCollection, PolyCollection)):
                    dense = len(coll.get_paths()) > min_paths
                else:
                    dense = False
            except Exception:
                dense = False
            if dense and not coll.get_rasterized():
                coll.set_rasterized(True)
                changed.append(coll)
    return changed


def save_figure(fig, path_png: str, *, dpi=None, extra_formats: Sequence[str] = (), saver=None,
                **savefig_kw) -> list:
    """Save `fig` to `path_png` EXACTLY as the plot modules always did, then
    to `<stem>.svg` / `<stem>.pdf` for each entry of `extra_formats` (R15).

    The PNG call receives only what the caller gives it (`dpi=None` means the
    `dpi` argument is not passed at all, so a figure whose resolution comes
    from `rcParams['savefig.dpi']` - the seaborn FacetGrid figures - is saved
    byte-identically); `saver`, when given, replaces `fig.savefig` for the
    PNG (circos passes its retrying `_safe_savefig`). Vector saves reuse the
    same `savefig_kw` (so `bbox_inches='tight'` carries over) after dense
    layers have been rasterised; the rasterisation is undone afterwards.

    Returns every path written, PNG first. A failure of a vector save is
    logged and skipped - it never costs the PNG that was already written."""
    kw = dict(savefig_kw)
    if dpi is not None:
        kw['dpi'] = dpi
    if saver is not None:
        saver(fig, path_png, **kw)
    else:
        fig.savefig(path_png, **kw)
    written = [path_png]
    fmts = normalise_extra_formats(extra_formats)
    if not fmts:
        return written
    stem = os.path.splitext(path_png)[0]
    changed = _rasterise_dense_layers(fig)
    try:
        for fmt in fmts:
            target = f'{stem}.{fmt}'
            try:
                fig.savefig(target, format=fmt, **kw)
                written.append(target)
            except Exception as exc:                     # pragma: no cover - defensive
                print(f"[plot_style] NOTE: could not write {target!r} ({exc!r}); the PNG was written.")
    finally:
        for coll in changed:
            try:
                coll.set_rasterized(False)
            except Exception:
                pass
    return written


# --------------------------------------------------------------------------- #
# Theme scope (R16)
# --------------------------------------------------------------------------- #

@contextmanager
def theme_context(style='ticks', font_scale=1.0, dpi=300, rc=None, palette='deep'):
    """Apply `sns.set_theme(style, font_scale, rc={figure.dpi, savefig.dpi})`
    for the duration of the block only, then restore every matplotlib rcParam
    - so a plot's font scale / style can no longer leak into whichever plot
    the runner draws next (R16). Same arguments, same call order as the
    plot modules' own `sns.set_theme(...)` lines they replace."""
    merged = {'figure.dpi': dpi, 'savefig.dpi': dpi}
    if rc:
        merged.update(rc)
    with matplotlib.rc_context():
        sns.set_theme(style=style, font_scale=font_scale, rc=merged, palette=palette)
        yield


def themed(style='ticks', font_scale_of=lambda a: 1.0, dpi_of=lambda a: 300):
    """Decorator form of `theme_context` for a plot function whose theme
    depends on its own arguments: `font_scale_of(bound)` / `dpi_of(bound)` get
    the call's bound-argument dict (defaults applied) and return the values
    the function used to hand to `sns.set_theme`. The whole call then runs
    inside the context, so nothing it sets leaks into the next plot (R16)."""
    import functools
    import inspect

    def deco(fn):
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            with theme_context(style, font_scale_of(bound.arguments), dpi_of(bound.arguments)):
                return fn(*args, **kwargs)
        return wrapper
    return deco


# --------------------------------------------------------------------------- #
# Diagonal-label helpers (moved verbatim from accuracy_metrics_plot.py; the
# copy in scatter_plot.py is deleted - R0)
# --------------------------------------------------------------------------- #

def wrap_diag_label(name, max_chars):
    """Break a model name at '_', '-' or ' ' so it fits a diagonal panel."""
    parts = [p for p in re.split(r'(?<=[_\- ])', str(name)) if p]
    lines, cur = [], ''
    for p in parts:
        if cur and len(cur) + len(p.rstrip()) > max_chars:
            lines.append(cur.rstrip())
            cur = p
        else:
            cur += p
    if cur:
        lines.append(cur.rstrip())
    return '\n'.join(lines) if lines else str(name)


def fit_label_pt(text, base_pt, width_in, floor_pt=7.0):
    """Shrink `base_pt` so `text` (~0.6 em per character) fits `width_in`."""
    longest = max((len(t) for t in str(text).split('\n')), default=1)
    return max(floor_pt, min(base_pt, width_in * 72.0 / (0.6 * max(longest, 1))))


def wrap_for_width(text, width_in, pt):
    return '\n'.join(textwrap.wrap(text, width=max(int(width_in * 72.0 / (0.55 * pt)), 20)))


# --------------------------------------------------------------------------- #
# Row thinning (R4)
# --------------------------------------------------------------------------- #

def thin_rows(x, y, flag, budget, rng, uniform=False):
    """Thin a dot cloud to at most ~`budget` points. Highlighted points
    (flag > 0, i.e. in somebody's top-k% set) are kept preferentially (up to
    60% of the budget) because they ARE the analysis; the grey background
    is thinned uniformly at random. `uniform=True` thins every point equally
    (used for the marker-effect panels, which are drawn without highlighting,
    so density must not be distorted).

    Body moved verbatim from `accuracy_metrics_plot._sample_dots` (which now
    delegates here) so the Czekanowski matrix draws the same dots, consuming
    the random generator identically."""
    if x.size <= budget:
        return x, y, flag
    if uniform:
        keep = rng.choice(x.size, budget, replace=False)
        return x[keep], y[keep], flag[keep]
    hi = np.flatnonzero(flag > 0)
    lo = np.flatnonzero(flag == 0)
    n_hi = min(hi.size, int(budget * 0.6))
    sel_hi = hi if hi.size <= n_hi else rng.choice(hi, n_hi, replace=False)
    n_lo = min(lo.size, max(budget - sel_hi.size, 0))
    sel_lo = lo if lo.size <= n_lo else rng.choice(lo, n_lo, replace=False)
    keep = np.concatenate([sel_lo, sel_hi])
    return x[keep], y[keep], flag[keep]


def thin_indices(n_rows, budget, protect=None, seed=0):
    """Sorted positional indices of a deterministic, uniform thinning of
    `n_rows` rows to at most `budget` NON-protected rows. Every index in
    `protect` (a boolean mask or an index array - the QTL rows) is ALWAYS
    kept, on top of the budget. `budget=None` keeps everything. The same
    `(n_rows, budget, protect, seed)` always returns the same rows."""
    idx = np.arange(int(n_rows))
    if budget is None or n_rows <= int(budget):
        return idx
    protect_idx = np.array([], dtype=int)
    if protect is not None:
        protect = np.asarray(protect)
        protect_idx = np.flatnonzero(protect) if protect.dtype == bool else protect.astype(int)
    mask = np.ones(int(n_rows), dtype=bool)
    mask[protect_idx] = False
    rest = idx[mask]
    if rest.size > int(budget):
        rng = np.random.default_rng(seed)
        rest = np.sort(rng.choice(rest, int(budget), replace=False))
    return np.sort(np.concatenate([rest, protect_idx]))


# --------------------------------------------------------------------------- #
# One resolved-settings log line per family (I11)
# --------------------------------------------------------------------------- #

def log_resolved(family: str, **settings) -> None:
    """`[family] key=value ...` - the resolved plot settings, logged
    unconditionally so a person reading a run log can tell, without opening
    the config JSON, whether a plot used the legacy or the improved options."""
    body = ' '.join(f'{k}={v!r}' for k, v in settings.items())
    print(f'[{family}] {body}')
