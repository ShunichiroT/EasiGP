"""
weight_plot.py
===============
Update ID ver4-9, R6 (design blueprint SS2.3).

Stacked bar plots of the MEAN optimised ensemble weight each contributing
model received (`diag010.pdf` Fig. 3's own "comparison of mean weights
allocated to the individual genomic prediction models"), a sibling of
`metric_plot.py` at the same layer and built to the same two-output shape:

    Weight.png       - row='population', col='phenotype'
    Weight_total.png - col='phenotype' only (aggregated across population)

Each subplot's x axis is one ENSEMBLE METHOD (the naive/equal-weight
ensemble, if present, then every weighted method actually used -
`Linear transformation`, `Nelder Mead`, `Bayesian optimisation`,
`Analytic least-squares`, or their own `__<algo>` variants), in
`pipeline_utils.canonical_model_order()` order - which already places the
naive ensemble immediately before the weighted methods, giving the
"equal-weight bar at the very left" R6 asks for from the SAME ordering
authority `metric_plot()`/`metric_summary.py` already use, rather than a
second, independently-derived rule. Each bar is a STACK of its own
contributing (base) models, coloured by `model_registry.model_family()`
- `matplotlib.cm.Blues` for the four conventional models, `Greens` for
everything else, each family's own members sampled at evenly spaced
lightness in `[0.35, 0.95]` in THEIR OWN canonical order, so a given base
model is drawn in the identical colour in every bar of the figure.

The naive ensemble writes no `Weight.csv` row at all (finding F5 - its
equal weighting is never "optimised", so nothing is fit or saved for it).
Its bar is SYNTHESISED here, not looked up: every segment at
`1 / len(naive_models)`, where `naive_models` is the SAME contributing-
model list `diversity_summary.resolve_ensemble_members()` already
resolves for R5 - one resolution rule, two consumers (the
`_map_genes_to_markers` precedent, architecture doc SS8). `naive_models`
is `None` (the default) when no naive-ensemble label was actually
selected for this run - no naive bar is drawn in that case.

This module imports only `pandas`, `numpy`, `matplotlib`,
`pipeline_utils.canonical_model_order` and `model_registry.model_family`
- never `main_app.py`, `genomic_prediction.py`, `diversity_summary.py`,
or `streamlit` (I2). `weight_plot()` never raises on empty/partial input.
"""

from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_utils import canonical_model_order, _is_naive_ensemble_label, _is_wopt_label
from model_registry import model_family

# Mirrors metric_plot.py's own METRIC_PLOT_CONFIG default exactly
# (font_scale=1, a 5-inch-tall panel per subplot) - a caller that omits
# WEIGHT_PLOT_CONFIG gets the same look-and-feel as the violin plots
# sitting beside this figure in the same result folder.
_DEFAULT_WEIGHT_PLOT_CONFIG = {'font_size': 1, 'fig_size': 5}

_NAIVE_LABEL = 'ensemble'
_WEIGHT_METADATA_COLS = ('population', 'phenotype', 'model', 'ratio', 'sample')

# ColorBrewer sequential maps (luminance-monotonic, so they stay legible
# under every CVD simulation main_app.py implements - see the module
# docstring's own note on why this fixed, non-user-editable palette needs
# no live CVD check).
_FAMILY_CMAPS = {'conventional': matplotlib.cm.Blues, 'machine_learning': matplotlib.cm.Greens}


def _family_colours(member_order: "list[str]") -> "dict[str, tuple]":
    """One RGBA colour per name in `member_order`, grouped by
    `model_registry.model_family()` and sampled at evenly spaced values in
    `[0.35, 0.95]` WITHIN each family, in that family's own subsequence of
    `member_order` (i.e. `canonical_model_order`'s own relative order,
    already applied by the caller). Logs a legibility warning (not an
    error) whenever a family has more than 10 members - the ramp still
    produces a colour for every one of them, just increasingly hard to
    tell apart by eye."""
    colours: "dict[str, tuple]" = {}
    for family, cmap in _FAMILY_CMAPS.items():
        members = [m for m in member_order if model_family(m) == family]
        if not members:
            continue
        if len(members) > 10:
            print(f"[weight_plot] NOTE: {len(members)} '{family}' model(s) share one figure - "
                  f"the colour ramp compresses above 10 members and segments may become hard to "
                  f"tell apart by eye. Reading order stays stable (canonical_model_order), so "
                  f"this is a legibility note, not an error.")
        positions = [0.65] if len(members) == 1 else np.linspace(0.35, 0.95, len(members))
        for name, pos in zip(members, positions):
            colours[name] = cmap(pos)
    return colours


def _mean_weight_long(weight: "pd.DataFrame", member_order: "list[str]", group_cols: "list[str]") -> "pd.DataFrame":
    """`weight` (already population-split for 'between', see caller) -> a
    tidy long frame, one row per `(*group_cols, x_label, contributing_model)`,
    `mean_weight` = the mean of that model's own weight column across every
    scenario row in that group - Fig. 3's own statistic. Rows for a
    contributing model absent from `member_order` (should not occur, by
    construction) are dropped rather than silently invented. Empty
    DataFrame in (or `weight` has no data) -> empty DataFrame out."""
    columns = [*group_cols, 'x_label', 'contributing_model', 'mean_weight']
    if weight is None or weight.shape[0] == 0 or 'model' not in weight.columns:
        return pd.DataFrame(columns=columns)
    model_cols = [c for c in weight.columns if c not in _WEIGHT_METADATA_COLS and c in member_order]
    if not model_cols:
        return pd.DataFrame(columns=columns)
    rows = []
    group_by_cols = [*group_cols, 'model']
    for key, sub in weight.groupby(group_by_cols, sort=False, dropna=False):
        if len(group_by_cols) == 1:
            key = (key,)
        *group_key, method = key
        for m in model_cols:
            vals = sub[m].dropna()
            if vals.shape[0] == 0:
                continue
            rows.append({**dict(zip(group_cols, group_key)), 'x_label': method,
                         'contributing_model': m, 'mean_weight': float(vals.mean())})
    return pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)


def _add_naive_rows(long_df: "pd.DataFrame", group_cols: "list[str]", naive_models: "list[str]") -> "pd.DataFrame":
    """Append the synthesised `1/N` naive-ensemble bar into EVERY facet
    cell already present in `long_df` (or, when `long_df` is empty - no
    real weighted-method data exists at all, only a naive ensemble was
    selected - a single placeholder cell, so at least one bar is still
    drawn, per the module's own "naive-only figure" failure-mode
    handling)."""
    equal_w = 1.0 / len(naive_models)
    if long_df.shape[0] == 0:
        cells = [tuple('all' for _ in group_cols)]
    else:
        cells = long_df[group_cols].drop_duplicates().itertuples(index=False, name=None)
    rows = []
    for cell in cells:
        for m in naive_models:
            rows.append({**dict(zip(group_cols, cell)), 'x_label': _NAIVE_LABEL,
                         'contributing_model': m, 'mean_weight': equal_w})
    naive_df = pd.DataFrame(rows, columns=[*group_cols, 'x_label', 'contributing_model', 'mean_weight'])
    return pd.concat([long_df, naive_df], ignore_index=True) if long_df.shape[0] else naive_df


def _draw_stacked_bar(ax, cell: "pd.DataFrame", x_order: "list[str]", member_order: "list[str]",
                       colours: "dict[str, tuple]") -> None:
    """Draw one stacked bar per entry in `x_order` onto `ax` - a missing
    `(x_label, contributing_model)` combination (that model contributed
    nothing in that facet cell) renders as a zero-height segment, never a
    gap or a fabricated value."""
    x_positions = np.arange(len(x_order))
    bottoms = np.zeros(len(x_order))
    for m in member_order:
        heights = np.array([
            float(cell.loc[(cell['x_label'] == x) & (cell['contributing_model'] == m), 'mean_weight'].sum())
            for x in x_order
        ])
        if not heights.any():
            continue
        ax.bar(x_positions, heights, bottom=bottoms, width=0.6, color=colours.get(m, '#999999'), label=m)
        bottoms = bottoms + heights
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_order, rotation=30, ha='right')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Weight')


def _draw_figure(long_df: "pd.DataFrame", x_order: "list[str]", member_order: "list[str]",
                  colours: "dict[str, tuple]", fig_size: int, row_facet: bool, path: str) -> "str | None":
    """One figure: `col='phenotype'`, and `row='population'` when
    `row_facet` is True (`Weight.png`) or a single unfaceted row when it
    is False (`Weight_total.png`, aggregated across population - the
    caller passes an already population-aggregated `long_df` for that
    case, see `weight_plot()`). Returns `path` if a figure was written,
    `None` if there was nothing to draw (never raises either way)."""
    if long_df.shape[0] == 0:
        return None
    phenotypes = list(pd.unique(long_df['phenotype']))
    populations = list(pd.unique(long_df['population'])) if row_facet else [None]
    n_rows, n_cols = max(len(populations), 1), max(len(phenotypes), 1)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_size * n_cols, fig_size * n_rows), squeeze=False)
    for r, pop in enumerate(populations):
        for c, phen in enumerate(phenotypes):
            ax = axes[r][c]
            cell = long_df[long_df['phenotype'] == phen]
            if pop is not None:
                cell = cell[cell['population'] == pop]
            _draw_stacked_bar(ax, cell, x_order, member_order, colours)
            ax.set_title(str(phen) if pop is None else f'{pop} | {phen}')

    handles = [plt.Rectangle((0, 0), 1, 1, color=colours.get(m, '#999999')) for m in member_order]
    if handles:
        fig.legend(handles, member_order, loc='lower right')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def weight_plot(
    weight: "pd.DataFrame",
    MODEL: "list[str]",
    RESULT_NAME: str,
    SCENARIO: str,
    WEIGHT_PLOT_CONFIG: "dict | None" = None,
    PLOT_DPI: int = 300,
    naive_models: "list[str] | None" = None,
) -> "list[str]":
    """Writes `Result/<RESULT_NAME>/Weight.png` and `Weight_total.png`;
    returns the paths actually written (`[]` when `weight` is empty AND
    `naive_models` is falsy - nothing to plot at all).

    Parameters
    ----------
    weight : `Weight.csv`'s own frame (any source, e.g.
        `batch_reader.ResultSet.weight()` / `batch_reader.load_combined
        (result_name, 'weight')`) - `population, phenotype, model, ratio,
        sample, <one column per contributing model>`. May be empty or
        `None`.
    MODEL : the run's own model list (base models are fine to include -
        anything that isn't a genuine contributing-model column, or isn't
        in `naive_models`, is silently excluded) - used ONLY to fix the
        STACKING/COLOUR order of the segments WITHIN each bar (via
        `pipeline_utils.canonical_model_order()`), so a given base model
        is drawn in the identical colour/position in every bar of the
        figure, regardless of which ensemble method's bar it appears in.
    naive_models : the naive ensemble's own resolved contributing-model
        list (see `diversity_summary.resolve_ensemble_members()`) -
        `None` (default) draws no naive bar at all. When given, a bar
        labelled `'ensemble'` is synthesised with every segment at
        `1 / len(naive_models)`.

    `SCENARIO == 'between'` splits `weight['population']` on `'->'` and
    keeps the LAST element, identically to `metric_plot.py`'s own split -
    so the two figures facet on the same population labels.
    """
    if WEIGHT_PLOT_CONFIG is None:
        WEIGHT_PLOT_CONFIG = _DEFAULT_WEIGHT_PLOT_CONFIG
    font_size = WEIGHT_PLOT_CONFIG.get('font_size', _DEFAULT_WEIGHT_PLOT_CONFIG['font_size'])
    fig_size = WEIGHT_PLOT_CONFIG.get('fig_size', _DEFAULT_WEIGHT_PLOT_CONFIG['fig_size'])

    has_weight_data = weight is not None and weight.shape[0] != 0 and 'model' in weight.columns
    if not has_weight_data and not naive_models:
        print("[weight_plot] Weight.csv is empty/absent and no naive-ensemble contributing "
              "models were given - nothing to plot (see W_OPT/naive-ensemble selection).")
        return []

    weight = weight.copy() if has_weight_data else pd.DataFrame(columns=list(_WEIGHT_METADATA_COLS))
    if has_weight_data and SCENARIO == 'between':
        weight['population'] = weight['population'].astype(str).str.split('->', expand=True).iloc[:, -1]

    weighted_labels_present = pd.unique(weight['model']).tolist() if has_weight_data else []
    for label in weighted_labels_present:
        if not _is_wopt_label(label):
            print(f"[weight_plot] NOTE: Weight.csv contains a 'model' value {label!r} that is "
                  f"not a recognised weighted-ensemble method label - plotted as-is.")
    x_order = canonical_model_order(([_NAIVE_LABEL] if naive_models else []) + weighted_labels_present)

    member_order = canonical_model_order(list(dict.fromkeys(MODEL)))
    member_order = [m for m in member_order if not (_is_naive_ensemble_label(m) or _is_wopt_label(m))]
    if has_weight_data:
        _model_cols_in_data = [c for c in weight.columns if c not in _WEIGHT_METADATA_COLS]
        for m in _model_cols_in_data:
            if m not in member_order:
                member_order.append(m)
    if naive_models:
        for m in naive_models:
            if m not in member_order:
                member_order.append(m)

    colours = _family_colours(member_order)

    per_pop_long = _mean_weight_long(weight, member_order, ['population', 'phenotype'])
    total_long = _mean_weight_long(weight, member_order, ['phenotype'])
    if naive_models:
        per_pop_long = _add_naive_rows(per_pop_long, ['population', 'phenotype'], naive_models)
        total_long = _add_naive_rows(total_long, ['phenotype'], naive_models)

    matplotlib.rcParams.update({'font.size': 10 * font_size, 'figure.dpi': PLOT_DPI, 'savefig.dpi': PLOT_DPI})

    result_dir = './Result/' + RESULT_NAME + '/'
    os.makedirs(result_dir, exist_ok=True)
    paths = []
    p1 = _draw_figure(per_pop_long, x_order, member_order, colours, fig_size, True, result_dir + 'Weight.png')
    if p1:
        paths.append(p1)
    p2 = _draw_figure(total_long, x_order, member_order, colours, fig_size, False, result_dir + 'Weight_total.png')
    if p2:
        paths.append(p2)

    if not paths:
        print("[weight_plot] Nothing to plot after resolving contributing models/ensemble "
              "labels - no figure written.")
    return paths
