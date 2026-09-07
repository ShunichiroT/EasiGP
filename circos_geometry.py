"""
circos_geometry.py
===================
Update ID ver4-6, R2 (blueprint §3.2/§3.5).

Single authority on circos ring radii, points<->r-unit conversion, text
footprint and seam geometry - imported by BOTH `main_app.py` (Defence 1:
GUI suggestions that try to make the common case look right) and
`circos_plot.py` (Defence 2: a renderer-side guard that makes ring-label
overlap impossible regardless of whether the prediction was right, a
config predates this update, or a user hand-edited the angles).

WHY THIS MODULE EXISTS (root cause, blueprint §3.1 RC-5)
--------------------------------------------------------------------------
Before this module existed, `main_app.py`'s own seam-gap/scale/window
suggestion functions re-derived a handful of rendering constants that
only ever really lived in `circos_plot.py` (the ring stride of 3 r-units,
the outer radius of 100, the chromosome-name radius of 108, the
hard-coded ring-label font size of 8pt, pycirclize's own 8-inch default
figure size) - and got some of them wrong, because it was measuring the
CHROMOSOME NAME's own footprint while the thing that actually overlaps on
a many-ring plot is the RING LABEL, drawn at a different radius that
shrinks by 3 r-units per ring. Any future change to one of these numbers
in `circos_plot.py` would silently drift out of sync with `main_app.py`'s
own suggestion - exactly the class of defect that sharing
`_map_genes_to_markers()` across module boundaries (architecture document
§8) was already designed to prevent for marker-pool routing. This module
applies the same discipline here: one authority, two importers, never a
second implementation.

This module is deliberately PURE: no Streamlit import, no project import
beyond the standard library and `matplotlib` (already a hard dependency
of `circos_plot.py`/`metric_plot.py`/`scatter_plot.py` - invariant I12 is
unaffected). It sits beside `circos_plot.py` in Layer 3c of the
architecture document's layering, so `main_app.py` (Layer 6) importing it
is a downward import (I2), and `circos_plot.py` importing it is lateral
within Layer 3c.

GEOMETRY MODEL
--------------------------------------------------------------------------
pycirclize renders a circos plot inside a square matplotlib figure of
side `figsize_in` inches (pycirclize's own default: 8 inches), with the
plot's own radial coordinate system spanning `r=0` (centre) to
`r=r_max` (pycirclize's own default: 100, "r-units") at the edge of the
polar axes. One r-unit therefore equals

    points_per_r_unit = (figsize_in / 2) * 72 / r_max

points on the rendered page (72 points per inch is matplotlib's own
fixed unit, `figsize_in / 2` because the axes radius is half the figure's
side length). At the pinned defaults this is `(8 / 2) * 72 / 100 = 2.88`
points per r-unit - verified directly against a real render during Phase
1 (blueprint Appendix B).

A ring label sitting at radius `r` (in r-units) occupies an angular width
that grows as its own text footprint (in points, converted to r-units)
divided by `r` - so the SAME label text needs MORE angular room the
CLOSER it sits to the centre (the innermost rings, after many outer rings
have been stacked). This inverse relationship is what makes a single,
one-size-fits-all seam gap unable to serve every ring simultaneously
(blueprint §3.1 RC-1) and is why Defence 2 (a per-ring, per-radius guard
inside `circos_plot.py::plot()`) is mandatory, not merely cosmetic.
"""

from __future__ import annotations

import math

# Logged once per process, not once per call - `text_width_points()` can be
# called many times per plot (once per ring label, once per tick label per
# chromosome); repeating the same "TextPath unavailable" note on every one
# of those calls would be noise, not diagnosis. Matches the codebase's own
# "log the resolved/negative state unconditionally, but don't spam an
# unconditional line per micro-call" convention used elsewhere (e.g. the
# H-statistic degenerate-pair skip, logged once per call, not once per pair).
_TEXTPATH_FALLBACK_LOGGED = False

# The only base font sizes this module ever rounds a suggestion to -
# matches `main_app.py::_nice_round_number()`'s own step set, but this
# module NEVER calls or modifies that function (see `nice_round_down()`'s
# own docstring for why a second, floor-only variant lives here instead).
_NICE_MULTIPLIERS = (1.0, 2.0, 5.0)

# Update ID ver4-6, R3 (bugfix - Defence 2 rendered with zero headroom):
# the SAME safety margin `seam_gap_for_labels()` has always applied to its
# own GUI-time PREDICTION (Defence 1, RK-7: "pycirclize's own text
# placement can differ slightly from the TextPath model used here -
# kerning, font fallback - and this margin absorbs that") - now shared,
# via `label_fits()` below, with the render-time GUARD (Defence 2:
# `fit_label_size()`/`truncate_label()`, called from
# `circos_plot.py::plot()`'s own `_resolve_ring_label()`), which
# previously compared the raw, zero-margin analytic requirement directly
# against the available gap.
#
# Root cause (found against a real render, RESULT_NAME='MaizeNAM',
# PHENOTYPE='days2anthesis'): with `CIRCOS_CONFIG['start']=6.2` /
# `['end']=353.8` (a symmetric, correctly-centred 12.4 degree seam), the
# gene-source ring label 'wisser_et_al' analytically needed 11.82 degrees
# at its configured 8pt - a bare 4.9% margin. Defence 2's own raw
# comparison judged this "fits" and rendered it at full size with no real
# headroom, so the actual renderer's own glyph metrics - which RK-7
# already documents can differ slightly from this module's analytic
# `TextPath` estimate - were enough on their own to visibly overlap the
# neighbouring chromosome sector. The label WAS correctly centred on the
# seam the whole time (confirmed against the render's own true geometric
# centre, not the seam's naive on-page midpoint) - centring was never the
# defect; the defect was Defence 2 trusting an exact-fit calculation with
# no room for the same rendering slop Defence 1 already budgets for.
#
# This is a general robustness gap, not specific to that one plot: ANY
# config whose widest ring label lands within ~15% of its own seam
# budget is equally exposed, on any genome, any model list, any gene
# source list - hence a single shared constant here rather than a
# one-off tweak in `circos_plot.py` for this specific label.
RENDER_SAFETY = 1.15


def label_fits(text: str, size_pt: float, r_centre: float, gap_deg: float, *,
                safety: float = RENDER_SAFETY, figsize_in: float = 8.0, r_max: float = 100.0) -> bool:
    """The single, authoritative "does this label actually fit" test -
    used by BOTH halves of Defence 2 (`fit_label_size()`'s shrink search
    and `truncate_label()`'s truncation search) so they can never disagree
    with each other, and callable directly by `circos_plot.py` itself
    (see `_resolve_ring_label()`) so its own final overflow check uses
    the exact same standard those two functions already used internally,
    rather than a third, separately-maintained comparison.

    `text`, rendered at `size_pt` and centred at radius `r_centre`, fits
    within `gap_deg` of seam gap if its own analytic requirement
    (`seam_gap_deg_for_label()`), inflated by `safety`, does not exceed
    `gap_deg` - i.e. the same margin `seam_gap_for_labels()` already
    applies when PREDICTING a seam gap (Defence 1) is now required here
    too, when actually DECIDING whether a label fits the gap that was
    built (Defence 2), so the two never operate to two different
    standards of "fits" (see `RENDER_SAFETY`'s own module-level note for
    the real render this was found against).

    `gap_deg <= 0` never fits (a zero or negative gap has no room for
    any label, however small) - returned directly rather than dividing
    by/comparing against a non-positive budget.
    """
    if gap_deg <= 0:
        return False
    required = seam_gap_deg_for_label(text, size_pt, r_centre, figsize_in=figsize_in, r_max=r_max)
    return required * safety <= gap_deg


def points_per_r_unit(*, figsize_in: float = 8.0, r_max: float = 100.0) -> float:
    """How many rendered points correspond to one radial "r-unit" of a
    pycirclize plot, at a given (square) figure size and radial extent.

    Derived from pycirclize's own `plotfig()` behaviour: the polar axes
    occupy the full `figsize_in` x `figsize_in` inch figure, so the axes
    RADIUS is `figsize_in / 2` inches; multiplying by 72 points/inch and
    dividing by `r_max` (the radial coordinate value pycirclize treats as
    the axes edge - 100 by this codebase's own convention throughout
    `circos_plot.py`) gives points per r-unit. At the pinned defaults
    (`figsize_in=8.0`, `r_max=100.0`) this is exactly 2.88 - verified
    directly against a real pycirclize render (blueprint Appendix B).
    """
    if figsize_in <= 0 or r_max <= 0:
        raise ValueError(
            f"points_per_r_unit: figsize_in and r_max must both be positive, "
            f"got figsize_in={figsize_in!r}, r_max={r_max!r}."
        )
    return (figsize_in / 2.0) * 72.0 / r_max


def text_width_points(text: str, size_pt: float, *, font_family: str | None = None) -> float:
    """Rendered width, in points, of `text` at font size `size_pt` -
    deterministic, analytic (no live renderer or figure needed), via
    `matplotlib.textpath.TextPath`'s own glyph metrics.

    `font_family` is forwarded to `matplotlib.font_manager.FontProperties`
    when given; `None` (the default) uses matplotlib's own default font,
    matching what pycirclize itself renders with unless a caller has
    configured otherwise.

    Never raises. Falls back to a crude character-count estimate
    (`len(text) * size_pt * 0.55`, matplotlib's own commonly-cited average
    glyph-width-to-size ratio for a typical sans-serif font) on ANY
    exception - e.g. a missing font on a minimal HPC image - logging the
    fallback exactly ONCE per process (see the module-level
    `_TEXTPATH_FALLBACK_LOGGED` flag), never per call.
    """
    global _TEXTPATH_FALLBACK_LOGGED
    if not text:
        return 0.0
    try:
        from matplotlib.font_manager import FontProperties
        from matplotlib.textpath import TextPath

        prop = FontProperties(family=font_family) if font_family else None
        path = TextPath((0, 0), text, size=size_pt, prop=prop)
        return float(path.get_extents().width)
    except Exception as exc:
        if not _TEXTPATH_FALLBACK_LOGGED:
            print(f"[circos_geometry] NOTE: matplotlib.textpath.TextPath metrics are "
                  f"unavailable in this environment ({exc!r}) - falling back to a "
                  f"character-count width estimate (len(text) * size_pt * 0.55) for every "
                  f"text-width calculation this run. This is less precise than real font "
                  f"metrics but never fatal; ring-label/tick-label sizing may be slightly "
                  f"more conservative than necessary as a result.")
            _TEXTPATH_FALLBACK_LOGGED = True
        return len(text) * size_pt * 0.55


def ring_geometry(n_rings: int, *, layout: str = 'legacy', r_outer: float = 100.0,
                   thickness: float = 3.0, r_inner_min: float = 25.0
                   ) -> list[tuple[float, float, float]]:
    """The single authority on ring radii - returns a list of
    `(r_lo, r_hi, r_centre)` tuples, one per ring, in OUTER-TO-INNER draw
    order (ring 0 is the outermost, matching `circos_plot.py::plot()`'s
    own `cnt` loop variable).

    Parameters
    ----------
    n_rings : int
        Total number of rings this plot will draw (model marker-effect
        rings plus gene-source rings combined - `circos_plot.py::plot()`
        computes this BEFORE its own ring-drawing loop starts, precisely
        so `layout='fit'` below can thin the stride to the true total,
        not just whatever has been drawn so far).
    layout : {'legacy', 'fit'}, default 'legacy'
        - ``'legacy'`` - `thickness` is used unmodified, reproducing
          `circos_plot.py`'s own historical, hard-coded
          `(97-(3*cnt), 100-(3*cnt))` literals EXACTLY when
          `thickness=3.0`/`r_outer=100.0` (the defaults) - this is what
          keeps an absent-config-key render byte-identical (I11, AC2.5).
          Unbounded: at `n_rings >= 33` (with the default `thickness=3.0`,
          `r_outer=100.0`), `r_lo` becomes non-positive - reproduced
          deliberately, since 'legacy' exists ONLY to match old output,
          never to protect a new run from a defect it was never subject
          to.
        - ``'fit'`` - thins `thickness` down to
          `min(thickness, (r_outer - r_inner_min) / n_rings)` (never UP -
          a plot with few rings still gets the same, familiar 3-unit
          stride) so that `n_rings` rings always fit between `r_outer`
          and `r_inner_min` without ever producing a non-positive radius,
          however many rings are requested (AC2.7: a 40-ring fixture
          renders without exception).
    r_outer : float, default 100.0
        Outer radius of the outermost ring, in r-units - matches
        `circos_plot.py`'s own `r=100` convention throughout.
    thickness : float, default 3.0
        Radial thickness of one ring under `layout='legacy'`, and the
        NEVER-EXCEEDED ceiling on thickness under `layout='fit'`.
    r_inner_min : float, default 25.0
        Under `layout='fit'` only: the innermost radius rings are never
        allowed to shrink past - leaves room, inside it, for whatever
        pycirclize draws at the very centre of the plot.

    Returns
    -------
    list of (r_lo, r_hi, r_centre) - length `n_rings`. With every default
    argument, `ring_geometry(n, layout='legacy')[cnt] ==
    (97 - 3*cnt, 100 - 3*cnt, 98.5 - 3*cnt)` for every `cnt` in
    `range(n)` - the exact triple `circos_plot.py`'s own pre-ver4-6
    literals produced.

    Raises
    ------
    ValueError
        If `n_rings < 0`, or `layout` is not `'legacy'`/`'fit'`.
    """
    if n_rings < 0:
        raise ValueError(f"ring_geometry: n_rings must be >= 0, got {n_rings!r}.")
    if layout == 'legacy':
        eff_thickness = thickness
    elif layout == 'fit':
        eff_thickness = thickness if n_rings <= 0 else min(thickness, (r_outer - r_inner_min) / n_rings)
    else:
        raise ValueError(f"ring_geometry: layout must be 'legacy' or 'fit', got {layout!r}.")

    geometry = []
    for cnt in range(n_rings):
        r_hi = r_outer - eff_thickness * cnt
        r_lo = r_hi - eff_thickness
        r_centre = r_hi - eff_thickness / 2.0
        geometry.append((r_lo, r_hi, r_centre))
    return geometry


def _arc_length_points(deg: float, r: float, *, figsize_in: float = 8.0,
                        r_max: float = 100.0) -> float:
    """Rendered arc LENGTH, in points, of a `deg`-degree sweep at radius
    `r` (r-units) - `r * points_per_r_unit(...) * radians(deg)`. Private:
    every public function below that needs this derives it internally,
    so a caller never has to reason about r-units vs. points itself."""
    return r * points_per_r_unit(figsize_in=figsize_in, r_max=r_max) * math.radians(deg)


def seam_gap_deg_for_label(text: str, size_pt: float, r_centre: float, *,
                            figsize_in: float = 8.0, r_max: float = 100.0) -> float:
    """The seam-gap angle (in degrees) `text`, rendered at `size_pt` and
    centred on the seam at radius `r_centre`, actually needs to avoid
    overlapping itself across the seam.

    Derivation: half of the text's own rendered width (points), converted
    to r-units via `points_per_r_unit()`, is the half-chord length at
    radius `r_centre` the text occupies; `2 * atan2(half_width_r_units,
    r_centre)` is the angular width that half-chord subtends. Verified
    directly against every worked example in the ver4-6 blueprint's own
    §3.1 table (4.2 degrees for a chromosome name at r=108, 56.1 degrees
    for `GAT_biological_prior_knowledge` at ring 19's r=41.5, 71.3 degrees
    for `GAT_biological_prior_knowledge__Bayesian` at the same ring) - all
    reproduce to the precision the blueprint itself reports.
    """
    width_pt = text_width_points(text, size_pt)
    half_width_r_units = (width_pt / 2.0) / points_per_r_unit(figsize_in=figsize_in, r_max=r_max)
    return 2.0 * math.degrees(math.atan2(half_width_r_units, r_centre))


def seam_gap_for_labels(labels, size_pt: float, radii, *, safety: float = RENDER_SAFETY,
                         cap_deg: float = 40.0, figsize_in: float = 8.0,
                         r_max: float = 100.0) -> tuple[float, bool]:
    """The seam gap (degrees) that would avoid overlap for EVERY label in
    `labels`, each rendered at its own corresponding radius in `radii`
    (same length and order - typically each ring's own `r_centre` from
    `ring_geometry()`), with a `safety` multiplier applied on top of the
    analytic worst case (RK-7: pycirclize's own text placement can differ
    slightly from the `TextPath` model used here - kerning, font
    fallback - and this margin absorbs that), then capped at `cap_deg`.

    Returns
    -------
    (gap_deg, was_capped) - `gap_deg` is rounded to 1 decimal place and
    floored at 8.0 (this module's own minimum sensible seam, matching the
    pre-ver4-6 formula's own floor); `was_capped` is True whenever the
    analytically-required gap (after `safety`) exceeded `cap_deg` - the
    caller (`main_app.py`) uses this to add an honest note ("the renderer
    will shrink/truncate ring labels") rather than silently returning a
    gap that will not, on its own, prevent overlap for the worst label.
    """
    labels = list(labels)
    radii = list(radii)
    if not labels or not radii:
        return 8.0, False
    if len(labels) != len(radii):
        raise ValueError(
            f"seam_gap_for_labels: labels ({len(labels)}) and radii ({len(radii)}) must be "
            f"the same length - one radius per label."
        )
    required = max(
        seam_gap_deg_for_label(lbl, size_pt, r, figsize_in=figsize_in, r_max=r_max)
        for lbl, r in zip(labels, radii)
    )
    gap = required * safety
    was_capped = gap > cap_deg
    gap = min(gap, cap_deg)
    gap = max(8.0, gap)
    return round(gap, 1), was_capped


def fit_label_size(text: str, r_centre: float, gap_deg: float, *, size_pt: float,
                    min_size_pt: float = 4.0, safety: float = RENDER_SAFETY,
                    figsize_in: float = 8.0, r_max: float = 100.0) -> float:
    """The largest font size, no larger than `size_pt` and no smaller than
    `min_size_pt`, at which `text` (rendered at radius `r_centre`) fits
    within `gap_deg` of seam gap per `label_fits()` - i.e. the shrink half
    of `circos_plot.py`'s "shrink, then truncate" Defence 2 guard.

    `safety` (Update ID ver4-6, R3): forwarded to every `label_fits()`
    call below, so a label is only judged to fit at a given size with the
    SAME margin `seam_gap_for_labels()` already applies to its own
    GUI-time prediction - see `RENDER_SAFETY`'s own module-level note for
    why this now matters here too, not only there.

    `label_fits()` is monotonically non-decreasing in `size_pt` (a larger
    font is never narrower), so this is a well-posed bisection over
    `[min_size_pt, size_pt]`. Returns `size_pt` unchanged if it already
    fits (the common case - most rings, most labels, need no shrinking at
    all); returns `min_size_pt` if even the SMALLEST allowed size does
    not fit at `gap_deg` - the caller is responsible for checking whether
    that floor size actually fits (via `label_fits(text, min_size_pt,
    r_centre, gap_deg, safety=safety)`) and switching to
    `truncate_label()` if not; this function itself never truncates
    text, only resizes it.
    """
    if size_pt <= min_size_pt:
        return size_pt
    if label_fits(text, size_pt, r_centre, gap_deg, safety=safety, figsize_in=figsize_in, r_max=r_max):
        return size_pt
    if not label_fits(text, min_size_pt, r_centre, gap_deg, safety=safety, figsize_in=figsize_in, r_max=r_max):
        # Does not fit even at the floor size - return the floor; the
        # caller decides whether to truncate on top of it.
        return min_size_pt

    lo, hi = min_size_pt, size_pt
    for _ in range(24):  # 24 halvings is comfortably sub-0.001pt precision
        mid = (lo + hi) / 2.0
        if label_fits(text, mid, r_centre, gap_deg, safety=safety, figsize_in=figsize_in, r_max=r_max):
            lo = mid
        else:
            hi = mid
    return round(lo, 2)


def truncate_label(text: str, r_centre: float, gap_deg: float, size_pt: float, *,
                    safety: float = RENDER_SAFETY, figsize_in: float = 8.0, r_max: float = 100.0,
                    ellipsis: str = '\u2026') -> str:
    """Shortens `text` (appending `ellipsis`) until it fits within
    `gap_deg` at font size `size_pt` and radius `r_centre`, per
    `label_fits()` - the truncate half of Defence 2's "shrink, then
    truncate" guard, used ONLY once `fit_label_size()` has already
    reduced the font to its floor and the label STILL does not fit there.

    `safety` (Update ID ver4-6, R3): forwarded to every `label_fits()`
    call below - see `fit_label_size()`'s own note and `RENDER_SAFETY`'s
    module-level note for why a truncation that only just clears the raw
    analytic requirement is not enough on its own.

    Returns `text` unchanged if it already fits. Never raises: if even a
    single character plus the ellipsis does not fit, returns the bare
    ellipsis - overlap is not possible either way (a caller-side legend
    entry, built from the `{truncated: full}` map every truncating caller
    is expected to keep, is what preserves the full name for the reader).
    """
    if label_fits(text, size_pt, r_centre, gap_deg, safety=safety, figsize_in=figsize_in, r_max=r_max):
        return text
    for n in range(len(text) - 1, 0, -1):
        candidate = text[:n].rstrip() + ellipsis
        if label_fits(candidate, size_pt, r_centre, gap_deg, safety=safety,
                       figsize_in=figsize_in, r_max=r_max):
            return candidate
    return ellipsis


def _nice_candidates(low: float, high: float) -> list:
    """Every `{1, 2, 5} x 10**n` value in `[low, high]`, ascending. Shared
    by `suggest_scale()`'s candidate sweep and `nice_round_down()`."""
    if low <= 0 or high <= 0 or low > high:
        return []
    n_low = int(math.floor(math.log10(low))) - 1
    n_high = int(math.ceil(math.log10(high))) + 1
    candidates = set()
    for n in range(n_low, n_high + 1):
        for m in _NICE_MULTIPLIERS:
            v = m * (10.0 ** n)
            if low <= v <= high:
                candidates.add(v)
    return sorted(candidates)


def nice_round_down(x: float, ceiling: float) -> float:
    """The largest `{1, 2, 5} x 10**n` value `<= min(x, ceiling)`.

    Deliberately a SEPARATE, floor-only function from
    `main_app.py::_nice_round_number()` (nearest-rounding, used elsewhere
    for unrelated suggestions this update does not touch) rather than a
    modification of it - blueprint §3.4 is explicit that
    `_nice_round_number` must not change, since other callers depend on
    its nearest-rounding behaviour. This is what fixes RC-4 (blueprint
    §3.1): the OLD window suggestion applied nearest-rounding AFTER its
    own clamp, so a clamped value of `3.6e6` against a `4.0e6` ceiling
    rounded UP to `5.0e6` - silently exceeding the very ceiling meant to
    bound it. `nice_round_down(3.6e6, 4.0e6)` instead returns `2.0e6` -
    the largest nice value that does not exceed `min(3.6e6, 4.0e6)` -
    never above the ceiling, by construction.

    Returns the raw value `min(x, ceiling)` unchanged (not rounded to any
    nice number) if no `{1, 2, 5} x 10**n` value lies at or below it -
    only possible for a non-positive `x`/`ceiling`, since a positive real
    number always has SOME nice value at or below it for a small enough
    exponent.
    """
    clamped = min(x, ceiling)
    if clamped <= 0:
        return clamped
    candidates = [c for c in _nice_candidates(clamped * 1e-9, clamped) if c <= clamped]
    if not candidates:
        return clamped
    return max(candidates)


def suggest_scale(chrom_lengths: dict, *, start: float, end: float, space: float,
                   tick_label_size_pt: float, packing: float = 1.30,
                   figsize_in: float = 8.0, r_max: float = 100.0) -> float:
    """The smallest `{1, 2, 5} x 10**n` tick interval such that, for
    EVERY chromosome (not only the longest - RC-2's own root cause was
    checking only the longest chromosome's own arc, blueprint §3.1),
    `length / scale + 1` tick labels - at their actual rendered width and
    a `packing` safety factor - fit within that chromosome's own sector
    arc length.

    Parameters
    ----------
    chrom_lengths : dict {chromosome name: length}
    start, end : float
        The circos plot's own start/end angle (degrees) - `360 - (end -
        start)` is the seam gap consumed before allocating the remaining
        circle across every chromosome's own sector.
    space : float
        Angular space (degrees) pycirclize inserts BETWEEN adjacent
        sectors - consumed once per chromosome, same as `circos_plot.py`
        itself already passes to `Circos.initialize_from_bed(space=...)`.
    tick_label_size_pt : float
        The ACTUAL rendered tick-label font size (already resolved
        through the same `max(5.0, min(8.0, label_size * 1.8))` transform
        `circos_plot.py` itself applies - callers pass the resolved
        value, not a raw GUI `label_size`).
    packing : float, default 1.30
        Safety multiplier on top of the analytic tick-label footprint
        (kerning/inter-label spacing pycirclize itself adds that a bare
        glyph-width sum does not capture).

    Returns
    -------
    float - the smallest fitting `{1, 2, 5} x 10**n` scale, so ticks are
    as dense (informative) as the available arc genuinely allows. Falls
    back to the LARGEST candidate considered (least dense, most likely to
    fit) if nothing in the swept range fits every chromosome - a
    best-effort answer rather than a raised exception, since a scale
    suggestion must never block a run.
    """
    lengths = {name: float(length) for name, length in chrom_lengths.items() if length and length > 0}
    if not lengths:
        return 1.0

    n_chrom = len(lengths)
    total_length = sum(lengths.values())
    gap_deg = 360.0 - (end - start)
    usable_deg = max(0.0, 360.0 - gap_deg - n_chrom * space)
    max_len = max(lengths.values())

    def _fits(scale: float) -> bool:
        for length in lengths.values():
            sector_deg = usable_deg * (length / total_length) if total_length > 0 else 0.0
            sector_pt = _arc_length_points(sector_deg, r_max, figsize_in=figsize_in, r_max=r_max)
            n_labels = math.floor(length / scale) + 1
            widest_tick_text = str(int(math.floor(length / scale)))
            label_width_pt = text_width_points(widest_tick_text, tick_label_size_pt)
            needed_pt = n_labels * label_width_pt * packing
            if needed_pt > sector_pt:
                return False
        return True

    candidates = _nice_candidates(max_len / 2000.0, max_len * 2.0)
    for candidate in candidates:  # ascending -> smallest (densest) fitting scale wins
        if _fits(candidate):
            return candidate
    # Nothing swept fit every chromosome - fall back to the coarsest
    # candidate available (or the ceiling itself if the sweep was empty).
    return candidates[-1] if candidates else max_len


def _median_inter_marker_gap(marker_positions) -> float | None:
    """Median consecutive-position gap, pooled across every chromosome.

    `marker_positions` may be either a `{chromosome: [positions, ...]}`
    dict (preferred - gaps are only meaningful WITHIN a chromosome, never
    across one) or a flat iterable of positions (treated as a single
    pseudo-chromosome). Returns `None` if fewer than 2 positions are
    available anywhere - the caller falls back to a length-based default
    in that case rather than dividing by a non-existent gap.
    """
    if marker_positions is None:
        return None
    if isinstance(marker_positions, dict):
        chrom_groups = list(marker_positions.values())
    else:
        chrom_groups = [list(marker_positions)]

    gaps = []
    for positions in chrom_groups:
        positions = sorted(float(p) for p in positions)
        for a, b in zip(positions[:-1], positions[1:]):
            gap = b - a
            if gap > 0:
                gaps.append(gap)
    if not gaps:
        return None
    gaps.sort()
    mid = len(gaps) // 2
    if len(gaps) % 2 == 1:
        return gaps[mid]
    return (gaps[mid - 1] + gaps[mid]) / 2.0


def suggest_window(chrom_lengths: dict, marker_positions=None, *, min_bins_shortest: int = 8,
                    max_bins_longest: int = 150, target_markers_per_bin: int = 5
                    ) -> tuple[float, str]:
    """The circos-track window size (bp or cM, whatever unit
    `chrom_lengths` is already in), bounded so that:

      - the LONGEST chromosome never exceeds `max_bins_longest` bins
        (drawability and `circos_plot.quantile_conversion()`'s own
        per-bin loop cost - the pre-ver4-6 `floor`), and
      - the SHORTEST chromosome always has at least `min_bins_shortest`
        bins (RC-3, blueprint §3.1: the pre-ver4-6 `ceiling` was anchored
        on the LONGEST chromosome too, so a chromosome a quarter that
        length received only ~1 bin at the ceiling - this is the fix:
        the ceiling is now the SHORTEST chromosome's own bound).

    The raw candidate (`median inter-marker gap * target_markers_per_bin`
    when `marker_positions` is supplied and has at least 2 positions
    somewhere, else the midpoint of `[floor, ceiling]`) is clamped into
    `[floor, ceiling]`, then rounded DOWN (never up - RC-4) to the nearest
    `{1, 2, 5} x 10**n` value via `nice_round_down()`.

    Returns
    -------
    (window, note) - `note` explains the resolution, including the
    RC-3-style degenerate case where the chromosome-length spread is so
    wide that `ceiling < floor` (both bounds cannot be satisfied
    simultaneously) - handled deterministically by setting
    `ceiling = floor` and naming both chromosomes, never by raising.
    """
    lengths = {name: float(length) for name, length in chrom_lengths.items() if length and length > 0}
    if not lengths:
        return 0.0, "no positive chromosome lengths were supplied - window defaults to 0."

    longest_name = max(lengths, key=lengths.get)
    shortest_name = min(lengths, key=lengths.get)
    l_max = lengths[longest_name]
    l_min = lengths[shortest_name]

    floor = l_max / max_bins_longest
    ceiling = l_min / min_bins_shortest
    note_parts = []
    if ceiling < floor:
        note_parts.append(
            f"chromosome lengths span too wide a range ({longest_name}={l_max:g} vs. "
            f"{shortest_name}={l_min:g}) for one window to satisfy both the "
            f"{max_bins_longest}-bin drawability ceiling on the longest chromosome and the "
            f"{min_bins_shortest}-bin floor on the shortest - using ceiling = floor "
            f"({floor:g})."
        )
        ceiling = floor

    median_gap = _median_inter_marker_gap(marker_positions)
    if median_gap:
        raw = median_gap * target_markers_per_bin
        note_parts.append(
            f"raw window = median inter-marker gap ({median_gap:g}) x "
            f"target_markers_per_bin ({target_markers_per_bin}) = {raw:g}."
        )
    else:
        raw = (floor + ceiling) / 2.0
        note_parts.append(
            "no usable marker positions were supplied - raw window defaulted to the "
            f"midpoint of [floor, ceiling] ({raw:g})."
        )

    clamped = min(max(raw, floor), ceiling)
    window = nice_round_down(clamped, ceiling)
    note_parts.append(f"clamped to [{floor:g}, {ceiling:g}] then rounded down to a nice value: {window:g}.")
    return window, " ".join(note_parts)


def suggest_chrom_label_size(chrom_names, chrom_lengths: dict, *, start: float, end: float,
                              space: float, lo: float = 4.0, hi: float = 9.0,
                              figsize_in: float = 8.0, r_max: float = 100.0) -> float:
    """The largest `label_size` in `[lo, hi]` such that the LONGEST
    chromosome NAME, rendered at `max(5.0, min(8.0, label_size * 1.8))`
    (the SAME transform `circos_plot.py` itself applies to derive its own
    `_tick_label_size` - kept in lockstep deliberately, see that
    variable's own comment there), fits within the NARROWEST sector's arc
    at `r=108` (the chromosome-name radius) with 10% headroom.

    Replaces the pre-ver4-6 `150 / n_chrom` heuristic, which read only
    chromosome COUNT, never actual name length or actual sector width.

    Returns
    -------
    float - `label_size` (NOT the rendered `_tick_label_size` - callers
    apply that same transform themselves, exactly as `circos_plot.py`
    already does). Falls back to `lo` if nothing in `[lo, hi]` fits (a
    best-effort minimum, never an exception - a label-size suggestion
    must never block a run).
    """
    names = [str(n) for n in chrom_names]
    lengths = {name: float(length) for name, length in chrom_lengths.items() if length and length > 0}
    if not names or not lengths:
        return lo

    longest_name = max(names, key=len)
    n_chrom = len(lengths)
    total_length = sum(lengths.values())
    gap_deg = 360.0 - (end - start)
    usable_deg = max(0.0, 360.0 - gap_deg - n_chrom * space)
    narrowest_deg = min(
        (usable_deg * (length / total_length) if total_length > 0 else 0.0)
        for length in lengths.values()
    )
    narrowest_arc_pt = _arc_length_points(narrowest_deg, 108.0, figsize_in=figsize_in, r_max=r_max)

    size = hi
    step = 0.1
    while size >= lo - 1e-9:
        rendered_size = max(5.0, min(8.0, size * 1.8))
        width_pt = text_width_points(longest_name, rendered_size)
        if width_pt * 1.10 <= narrowest_arc_pt:
            return round(size, 1)
        size -= step
    return lo
