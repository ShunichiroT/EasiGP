from pycirclize import Circos
from pycirclize.parser import Bed
from pycirclize import config as _pycirclize_config
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.legend_handler as mlegend_handler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
import tempfile
import uuid
import math
import re
import time
import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED

from pipeline_utils import unify_columns_by_position, is_naive_ensemble_label, is_wopt_label
import plot_style as _ps
# Memory-bounded (dictionary-coded, streaming, spill-capable) reduction of the
# marker-pair interaction/attention tables - see interaction_reduce.py. Pure
# NumPy/pandas with no project imports, so this is a safe downward import.
from interaction_reduce import PairSumStore, store_from_frame

# ---------------------------------------------------------------------------
# Performance (multi-CPU circos-plot rendering): circos_plot()'s own
# (PHENOTYPE x POPULATION) loop below - see that function - draws and
# saves ONE independent circos PNG per iteration (plus, optionally, a
# per-model interaction chord diagram). Each iteration's own cost is
# dominated by pycirclize/matplotlib rendering and a 600 dpi (by default)
# `_safe_savefig()`, not by anything already sped up by the interaction-
# assembly fixes elsewhere in this module - and, critically, one
# iteration's work is fully independent of every other's (each reads the
# SAME precomputed, read-only lookup tables built once before the loop,
# and writes its OWN distinct output file, never touching another
# iteration's). That combination - independent units, each individually
# expensive, sharing only read-only inputs - is exactly what a process
# pool is for.
#
# `multiprocessing.get_context('spawn')` explicitly, never the platform
# default (`fork` on Linux) - mirrors intra_batch_parallel.py's/
# intra_task_parallel.py's own `_MP_CONTEXT`, and that module's own
# comment on why (see `_MP_CONTEXT` there): a spawned worker starts from
# a genuinely fresh interpreter, inheriting this process's environment
# but never its already-initialised native library/thread state - this
# module has no rpy2/R dependency to protect the way that one does, but
# `spawn` is also the ONLY start method Windows supports at all, so using
# it unconditionally here keeps this feature's behaviour identical on
# every OS this codebase runs on, rather than "usually fine on Linux,
# untested on Windows".
_MP_CONTEXT = multiprocessing.get_context('spawn')

# Set by `_init_circos_plot_worker()` once per spawned worker process -
# see that function's own docstring for why a module-level global (rather
# than a closure) is what a `spawn`-started worker needs here.
_CIRCOS_WORKER_CTX = None
# Update ID ver4-6, R2 (blueprint §3.2/§3.4): the single shared authority
# on ring radii, text footprint and seam geometry - see that module's own
# docstring for why it is a separate, pure sibling of this file rather
# than living in pipeline_utils.py (which would pull matplotlib into
# every model process) or being re-derived here a second time (the
# defect R2 exists to fix in the first place).
import circos_geometry
# Region highlighting (links coming from chosen genome regions) - pure helpers
# shared with main_app.py's GUI table/preview; see that module's docstring.
import circos_highlight as _hl

# ver4-4 R7.b - module-level cache for _load_combined_marker_info() below.
# Cleared at the top of every circos_plot() call (see that function) so it
# never serves a stale entry across separate runs/RESULT_NAMEs within one
# long-lived process (e.g. main_app.py's in-process 'Option B: run now').
_MARKER_INFO_CACHE = {}


# ---------------------------------------------------------------------------
# Bugfix (see log_step2_local - a Windows run failed with
# "OSError: [Errno 22] Invalid argument: './Result/MaizeNAM/
# circos_days2anthesis_1.png'" from inside PIL's own `Image.save()`, at
# the literal `open(filename, "w+b")` call - i.e. the OS itself refused
# to open a path that LOOKS unremarkable as plain text. On Windows,
# `CreateFile` can reject an apparently-ordinary path for reasons that
# never show up by eye in a log line: a component that is (or, after
# Windows' own trailing-character stripping, resolves to) a reserved
# device name (CON, PRN, AUX, NUL, COM1-9, LPT1-9), a component ending in
# a space or a period, or a handful of characters that are reserved on
# Windows but legal on the OS this codebase is normally developed and
# tested on (: < > " | ? *) - and PHENOTYPE/POPULATION/model names here
# are exactly the kind of free-text, user-authored values (a phenotype
# column header typed by a scientist, e.g. containing '/', ':', or a
# trailing space copied out of a spreadsheet) that can carry exactly one
# of those without ever being validated as a filename anywhere upstream.
#
# Two independent, complementary fixes follow:
#   1. sanitise every free-text value that becomes part of a saved
#      filename here, so a value that is perfectly fine as DATA (a
#      population/phenotype/model label) can never itself make the
#      resulting path invalid for the OS actually running this - see
#      `_sanitize_path_component()`;
#   2. save through `_safe_savefig()`, a small wrapper that also retries
#      briefly on any OSError - the same symptom is also a well-known,
#      genuinely transient nuisance on Windows when the target folder is
#      inside an actively-syncing cloud folder (OneDrive/Dropbox) or
#      briefly held by antivirus/indexing right after creation, cases a
#      sanitised filename alone would not fix.
# Neither fix is destructive: a filename with no reserved characters at
# all is returned byte-for-byte unchanged by `_sanitize_path_component()`,
# and `_safe_savefig()` only ever retries - it never silently drops or
# renames a plot the person didn't ask it to.
# ---------------------------------------------------------------------------
_WINDOWS_RESERVED_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED_NAMES = {
    'CON', 'PRN', 'AUX', 'NUL',
    *{f'COM{i}' for i in range(1, 10)}, *{f'LPT{i}' for i in range(1, 10)},
}


def _sanitize_path_component(value) -> str:
    """Makes an arbitrary, free-text value (a phenotype name, population
    label, or model name) SAFE to use as one component of a saved
    filename on every OS this codebase runs on - never touches the
    actual DATA (marker/trait/model identifiers used as join keys
    elsewhere in the pipeline are untouched; only the string used here,
    for a file NAME, is affected). Replaces any character Windows
    forbids in a filename with '_', strips trailing spaces/periods
    (silently dropped by Windows but a well-documented cause of
    'Invalid argument' from other tools/APIs that don't drop them first),
    and appends '_' to a bare Windows-reserved device name (CON, LPT1,
    ...) so it no longer collides with one. A value that was already
    clean is returned completely unchanged."""
    text = str(value)
    cleaned = _WINDOWS_RESERVED_CHARS.sub('_', text).strip().rstrip('. ')
    if not cleaned:
        cleaned = '_'
    if cleaned.upper() in _WINDOWS_RESERVED_NAMES:
        cleaned = cleaned + '_'
    return cleaned


def _safe_savefig(fig, path, *, dpi=600, attempts=3, base_delay=0.5, **kwargs):
    """`fig.savefig(path, dpi=dpi, **kwargs)`, with two defensive
    additions (see the module-level note above): the parent directory is
    (re-)created immediately before writing (`exist_ok=True` - a no-op if
    it's already there), and a transient `OSError` (the target briefly
    locked by a sync client/antivirus/indexer, or - see the CPython
    tracker's own long-running discussion of Windows' 'Invalid argument'
    - a momentary filesystem hiccup unrelated to the path's own text) is
    retried a few times with a short backoff before finally propagating,
    with a clear, actionable message rather than the bare, unhelpful one
    that would otherwise surface deep inside PIL/matplotlib."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            fig.savefig(path, dpi=dpi, **kwargs)
            return
        except OSError as exc:
            last_exc = exc
            if attempt == attempts:
                break
            time.sleep(base_delay * attempt)
    raise OSError(
        f"[circos_plot] Could not write '{path}' after {attempts} attempt(s): {last_exc!r}. "
        f"This is usually a transient OS-level issue (the file briefly locked by a sync client, "
        f"antivirus, or file indexer - especially if this folder is inside a OneDrive/Dropbox-"
        f"synced location) rather than a problem with the plot itself - re-running the export "
        f"usually succeeds. If it keeps failing, check that '{os.path.dirname(path) or '.'}' is "
        f"writable and not inside a cloud-sync folder that is currently paused/offline."
    ) from last_exc


def _visible_edge_color(fill_color, min_contrast=90.0, min_factor=0.15):
    """The border colour to use for a given fill colour.

    Requirement (does the border colour match the marker colour?):
    returns `fill_color` COMPLETELY UNCHANGED whenever it's already
    visible enough against a white plot background on its own - which,
    for EasiGP's own real marker-effect/gene-source palette
    (DEFAULT_CYTOBAND_COLORMAP, main_app.py), is the MAJORITY of colours
    (every mid-to-high quantile level, e.g. blue4 upward, red3 upward,
    and any other reasonably saturated custom colour someone adds).
    Only darkens - and only by the SMALLEST amount that actually fixes
    it - for colours too close to white to be visible as a border at all
    on their own (EasiGP's own deliberately pale LOWEST quantile levels,
    e.g. blue0 = #def2ff, red0 = #fce1a7, used to visually de-emphasise
    low-effect markers - see this file's data_conversion()/
    quantile_conversion() - plus plain white itself). Preserves hue in
    both cases: an untouched colour is obviously identical, and a
    darkened one keeps every RGB channel's relative ratio (scaling all
    three by the same factor never shifts which channel dominates), so
    even an adjusted border stays recognisably in the same colour family
    as its own fill - still bluish for a blue-family marker, still
    reddish for a red-family one - never replaced by an unrelated fixed
    colour like plain black, which would lose that information entirely.

    Uses matplotlib's own colour parser (matplotlib.colors.to_rgb),
    which accepts every colour format matplotlib itself understands -
    '#rrggbb', '#rgb', named CSS colours ('steelblue', 'orange', ...),
    etc. - so a NEWLY ADDED custom colour in ANY of those formats is
    handled exactly the same way, not just the specific '#rrggbb' shape
    EasiGP's own built-in palette happens to use. A colour matplotlib
    can't parse at all is returned unchanged (the same colour a caller
    would already have seen fail when trying to use it as a FILL, so
    this never needs to raise its own separate error over it).

    Parameters
    ----------
    min_contrast : the minimum acceptable distance (0-255 scale) between
        a colour's perceptual luminance and white - anything at or above
        this is left completely untouched; anything below is darkened by
        just enough to reach it (never further).
    min_factor : a floor on how much darkening is ever applied (0.15 =
        never darker than 15% of the original brightness), so a colour
        that's ALREADY very close to pure black (luminance near 0, which
        would otherwise need an enormous - or undefined - darkening
        factor to reach min_contrast) doesn't get pushed to solid black
        and lose its own hue entirely.
    """
    try:
        r, g, b = mcolors.to_rgb(fill_color)
    except ValueError:
        return fill_color

    luminance = 255.0 * (0.299 * r + 0.587 * g + 0.114 * b)
    contrast = 255.0 - luminance
    if contrast >= min_contrast or luminance <= 0:
        return fill_color

    factor = max(min_factor, min(1.0, (255.0 - min_contrast) / luminance))
    return mcolors.to_hex((r * factor, g * factor, b * factor))


def _add_cytoband_tracks_with_border(circos, r_lim, cytoband_file, track_name, cytoband_cmap,
                                      edge_width=0.4):
    """A drop-in replacement for Circos.add_cytoband_tracks() (pycirclize's
    own built-in method, used elsewhere in this file for the gene/model
    tracks) that ALSO draws a thin, visible border around every region -
    pycirclize's own version calls track.rect(start, end, fc=color) with
    no way to pass through an edge colour/width at all, filling the
    region only.

    Requirement (marker/gene regions too small to actually see on the
    plot): a filled region's rendered width scales with its DATA-SPACE
    size, so an unusually short marker/gene, or one on an unusually long
    chromosome, can end up thinner than a single pixel and simply not
    render at all - this is exactly what the GENE_ADJUST/END_ADJUST
    widening (see this file's data_conversion()) and the WINDOW-size
    suggestion (main_app.py) already fix at the DATA level, by making the
    region itself bigger before it's ever drawn. A border LINE WIDTH,
    however, is a fixed on-screen thickness (matplotlib's `lw`, in
    points) that does NOT scale down with the region's data-space size -
    so even a region that still ends up sub-pixel-thin after widening
    (e.g. an extreme genome scale, or a value smaller than intended) gets
    outlined by a border that stays visible regardless, as a second,
    complementary safety net on top of the data-level fix rather than a
    replacement for it.

    The border colour is DERIVED from each region's own intended fill
    colour via _visible_edge_color() (an EXACT match whenever the fill is
    already visible enough on its own - the majority of colours in
    practice - and only ever the smallest necessary darkening otherwise,
    never a single fixed colour for everything) - for a normal-sized
    (visible) region this usually means the border and fill are
    literally the same colour; for a sub-pixel-width one, this border IS
    the only thing that ends up visible at all, so it needs to still
    convey which quantile level / gene source that region actually
    represents, not a uniform, information-losing colour like plain
    black would.

    Mirrors pycirclize's own add_cytoband_tracks() implementation
    exactly otherwise (same per-sector track creation, same axis() call,
    same cytoband_cmap colour lookup) - see that method's source (in the
    installed pycirclize.circos module) for the reference this is kept
    in sync with. Tooltips (pycirclize's own hover-text feature, only
    relevant for interactive/HTML output) are deliberately omitted here,
    since this codebase only ever saves static PNGs.

    ver4-4 R7.d: the per-sector, per-record scan used to be a plain
    O(sectors x records) double loop over EVERY record for EVERY sector,
    checking `sector.name == rec.chr` record-by-record - wasteful once a
    genome has many chromosomes and/or many marker/gene records.
    `cytoband_records` is now bucketed by `rec.chr` ONCE, up front, into
    a dict - turning the scan into O(records) total (each record visited
    exactly once, then looked up by its own sector directly) plus
    O(sectors) for track creation, rather than re-scanning the full
    record list once per sector.

    Also returns the set of `str(rec.score)` values that were actually
    drawn onto a real sector (i.e. `rec.score`, which is this codebase's
    own 'colour' value - see `data_conversion()`'s own note on why gene/
    marker colour values are written into a BED file's 'score' column)
    - this is exactly the same set `plot()` used to recover by
    RE-READING the same `cytoband_file` a second time via a plain
    `pd.read_csv(...)['colour'].unique()` call, purely to know which
    colours need a legend entry (Requirement 13's own "only colours that
    ACTUALLY appear on this plot" rule). Returning it here as a
    byproduct of the drawing loop this function was already doing lets
    `plot()` drop that second read entirely - see that function's own
    R7.d note. (A record whose chromosome isn't among `circos.sectors`
    at all was never drawn in the FIRST implementation either - its
    colour simply never matched any `sector.name == rec.chr` comparison
    - so it is correctly excluded here too, and doing so is at worst a
    tightening of Requirement 13's own "only actually-drawn colours"
    rule, not a behaviour change to anything that could ever have been
    visibly rendered.)

    Returns
    -------
    set[str]
        Every `str(rec.score)` value actually drawn onto some sector.
    """
    if cytoband_cmap is None:
        cytoband_cmap = _pycirclize_config.CYTOBAND_COLORMAP
    cytoband_records = Bed(cytoband_file).records
    _records_by_chr = {}
    for rec in cytoband_records:
        _records_by_chr.setdefault(rec.chr, []).append(rec)

    used_colours = set()
    for sector in circos.sectors:
        track = sector.add_track(r_lim, name=track_name)
        track.axis()
        for rec in _records_by_chr.get(sector.name, []):
            color_key = str(rec.score)
            color = cytoband_cmap.get(color_key, 'white')
            used_colours.add(color_key)
            # Requirement 3 (bugfix - ValueError: x=... is invalid
            # range of '...' sector): the widened start/end written by
            # quantile_conversion()/data_conversion() are ALREADY meant
            # to be clamped to their own chromosome's bounds at write
            # time (see those functions' own per-chromosome clamping
            # loops) - but that clamp depends on a SEPARATE lookup
            # against a chrom_info file read independently of what
            # circos itself was actually initialized with, and can
            # silently skip a chromosome entirely (falls through via
            # `continue`) if that lookup ever fails to find a matching
            # row (e.g. a chromosome present in marker/gene info but
            # missing - or named slightly differently - in the
            # chrom_info file). Clamping AGAIN here, directly against
            # THIS sector's own authoritative bounds (exactly what
            # circos was actually initialized with, so this can never
            # itself be wrong) guarantees pycirclize is never handed an
            # out-of-range coordinate, regardless of whether the
            # earlier, separate clamp ran into that edge case - a
            # defensive second layer, not a replacement for fixing the
            # data itself to be correctly widened in the first place.
            _start = max(sector.start, min(rec.start, sector.end))
            _end = max(sector.start, min(rec.end, sector.end))
            track.rect(_start, _end, fc=color, ec=_visible_edge_color(color), lw=edge_width)
    return used_colours


def _load_combined_marker_info(marker_info, RESULT_NAME, PHENOTYPE):
    """Load marker_info.csv (SNP-level coordinates: chromosome, name, start,
    end) and transparently merge in any per-model GENE-level coordinate
    table(s) for this phenotype, so a single 'name' lookup resolves both SNP
    names (the usual case) and gene names (e.g.
    models/GAT_biological_prior_knowledge.py's own attention output, whose
    marker1/marker2 values are gene names, not SNP names, since its graph
    nodes are genes - see that file's RESULT_NAME/PHENOTYPE_NAME docstring
    entry for the side file this picks up:
    './Result/<RESULT_NAME>/<model>_gene_coordinates_<PHENOTYPE>.csv').

    Matched by filename pattern (`*_gene_coordinates_<PHENOTYPE>.csv`) so
    this stays usable for any future gene-node-based model that writes the
    same shape of side file, without circos_plot.py needing to hardcode a
    specific model's name. If no such file exists for this phenotype (the
    common case - every non-gene-level model), this is exactly equivalent
    to the plain `pd.read_csv(marker_info)` it replaces: zero behaviour
    change for SNP-only results.

    ver4-4 R7.b: this used to re-read marker_info.csv AND re-run
    glob.glob() AND re-read every matching gene-coordinate side file on
    EVERY call - and this function is called once per model per phenotype
    per population from plot()/quantile_conversion()/interaction()
    combined (measured: ~480 reads for a modest 5-phenotype x
    6-population x 12-model run, ~22ms/read, ~10s of pure re-parsing on
    top of everything else). Now cached at module level, keyed by
    `(marker_info, mtime, RESULT_NAME, PHENOTYPE)` - the SAME
    "path + mtime" pattern `pipeline_utils.phenotype_file_mtime_key()`
    already establishes elsewhere in this codebase, so a marker_info.csv
    edited BETWEEN runs (or mid-run, however unlikely) is still re-read
    rather than silently served stale. The cache is cleared at the top of
    every `circos_plot()` call (see that function), so it never persists
    stale entries across separate runs/RESULT_NAMEs within one long-lived
    process (e.g. main_app.py's in-process 'Option B: run now' path).

    The returned DataFrame is served directly from the cache on a hit
    (no defensive copy) - safe because every current caller only ever
    reads from it via `pd.merge(...)`, never mutates it in place; adding
    a copy here would spend back part of the very cost this cache exists
    to remove.
    """
    try:
        _mtime = os.path.getmtime(marker_info)
    except OSError:
        _mtime = -1.0
    _cache_key = (marker_info, _mtime, RESULT_NAME, PHENOTYPE)
    _cached = _MARKER_INFO_CACHE.get(_cache_key)
    if _cached is not None:
        return _cached

    marker = pd.read_csv(marker_info)
    # Requirement 8: unify by position (chromosome, name, start, end) -
    # 'name' here is the MARKER's identifying column HEADER (a fixed,
    # structural part of this file's schema), not the marker names
    # themselves (the VALUES in that column, which are of course left
    # completely untouched) - so this is safe to rename regardless of
    # what header text the file actually uses.
    marker = unify_columns_by_position(marker, ['chromosome', 'name', 'start', 'end'], 'marker info file')

    pattern = os.path.join('.', 'Result', RESULT_NAME, f'*_gene_coordinates_{PHENOTYPE}.csv')
    gene_coord_files = glob.glob(pattern)
    if not gene_coord_files and os.path.dirname(RESULT_NAME):
        # Multi-option runs (BLUEs / BLUPs ...) plot into
        # 'Result/<RESULT_NAME>/<option tag>/' but the side artefacts a
        # model wrote (gene-coordinate tables) live in the parent run folder.
        gene_coord_files = glob.glob(os.path.join(
            '.', 'Result', os.path.dirname(RESULT_NAME), f'*_gene_coordinates_{PHENOTYPE}.csv'))
    if not gene_coord_files:
        combined = marker
    else:
        gene_tables = [pd.read_csv(f) for f in gene_coord_files]
        combined = pd.concat([marker] + gene_tables, ignore_index=True)
        # A name should resolve to exactly one location; if a gene coordinate
        # table and marker_info somehow both define the same name, keep
        # marker_info's own entry (read first, so kept by keep='first').
        combined = combined.drop_duplicates(subset=['name'], keep='first')

    _MARKER_INFO_CACHE[_cache_key] = combined
    return combined


def _circos_intermediate_dir(RESULT_NAME):
    """Requirement 2: the chrom_*.bed / gene_info_*.tsv / marker_effect_*.tsv
    files this module writes and reads back are purely intermediate
    working files for pycirclize's own Bed-file-based track API - never
    meant to be inspected directly the way the REAL result files (
    Metric.csv, Marker_effect.csv, the final circos_*.png images
    themselves, etc.) are. Keeping them in this subfolder (created here,
    on demand) instead of dumped directly into
    './Result/<RESULT_NAME>/' alongside the real output keeps that
    top-level result folder readable - the final circos_*.png plots
    still save directly there, unaffected; only these intermediate
    per-model/per-population working files move."""
    path = os.path.join('.', 'Result', RESULT_NAME, 'circos_intermediate')
    os.makedirs(path, exist_ok=True)
    return path


def data_conversion(chrom_info, gene_info, PHENOTYPE, RESULT_NAME, gene_adjust=0):
    chromosome = pd.read_csv(chrom_info)
    # Requirement 8: unify by position - every column in this file is a
    # fixed structural field (chromosome ID, start, end, population),
    # never a user-chosen identifier used as a lookup key elsewhere, so
    # positional unification is always safe here.
    chromosome = unify_columns_by_position(
        chromosome, ['chromosome', 'start', 'end', 'population'], 'chromosome info file'
    )
    # Requirement (bugfix): explicit str cast, not left to pandas' own
    # CSV type inference - a 'population' column mixing numeric
    # populations with the literal string 'all' (which circos_plot()
    # itself always adds - see its own POPULATION = ('all',) +
    # tuple(POPULATION)) usually gets inferred as all-string by pandas
    # anyway, but that inference is CONTEXT-DEPENDENT on the exact file
    # contents (whether 'all' rows happen to be present, their exact
    # position, etc.) - relying on it left chromosome_population's
    # actual dtype (str vs int) genuinely unpredictable from one file to
    # the next, and the SAME risk applied separately to gene_population
    # below, for the SAME reason. Two independently-unpredictable dtypes
    # being compared/indexed against each other (see the fix a few lines
    # down) is exactly the kind of thing that can silently work in a
    # simple case and silently break in a different one.
    # ver4-75: _clean_population_label() (not a bare astype(str)) so a file
    # that writes the population as 2, 2.0 or '2' all become the SAME '2'
    # the effect/POPULATION side uses.
    chromosome['population'] = _remap_population_column(chromosome['population'], _clean_population_label)
    chromosome_population = pd.unique(chromosome['population'])
    
    # ver4-4 R7.a (blueprint §2.7.1/§2.7.2, root-cause finding R7.a):
    # replaces a Python-level list comprehension that called int(round(...))
    # once per row with the equivalent vectorised pandas op. Benchmarked at
    # 219x faster on 200k rows with byte-identical output (Series.round()
    # uses the same round-half-to-even convention as the builtin round()
    # this replaces) - see the R7 design record.
    chromosome['start'] = chromosome['start'].round().astype('int64')
    chromosome['end'] = chromosome['end'].round().astype('int64')
    chromosome['chromosome'] = chromosome['chromosome'].astype(str)
    
    for i in range(len(chromosome_population)):
        chromosome_selected = chromosome[chromosome['population']==chromosome_population[i]]
        chromosome_selected = chromosome_selected.drop(['population'],axis=1)
        chromosome_selected.to_csv(_circos_intermediate_dir(RESULT_NAME)+'/chrom_'+str(chromosome_population[i])+'.bed', sep='\t', index=False)
    
    if gene_info is not None:
        gene = pd.read_csv(gene_info)
        # Requirement 8: same reasoning as marker_info.csv above - 'name'
        # here is this file's gene-identifying column HEADER, a fixed
        # structural field, not the gene names themselves (which stay
        # untouched as VALUES).
        gene = unify_columns_by_position(
            gene, ['chromosome', 'start', 'end', 'name', 'colour', 'source', 'phenotype', 'population'],
            'gene info file'
        )
        # Requirement (bugfix): same explicit str cast as chromosome
        # above, and for the same reason - see that comment.
        gene['population'] = _remap_population_column(gene['population'], _clean_population_label)

        # Requirement: gene regions are often too small to actually see
        # on the plot (a modest-sized gene against a whole chromosome's
        # rendered arc) - gene_adjust (in the same units as the
        # chromosome/gene info files, e.g. bp) widens each one the same
        # way END_ADJUST already widens marker regions: subtracted from
        # start, added to end, then clamped to stay within that gene's
        # OWN chromosome (never a hardcoded 0, and never another
        # chromosome's bounds) - see the per-chromosome loop just below.
        gene['chromosome'] = gene['chromosome'].astype(str)
        # Requirement (diagnostic): check the RAW, un-widened positions
        # against the chromosome file BEFORE gene_adjust is applied below -
        # see _warn_if_raw_position_exceeds_chromosome()'s own docstring.
        _warn_if_raw_position_exceeds_chromosome(gene, chromosome, 'gene info file')
        # Requirement (bugfix - GENE_ADJUST could overflow into a wildly
        # wrong number for a large coordinate): int(...) used to be
        # applied AFTER the subtraction/addition below, not before -
        # gene.loc[k, 'start'] is a numpy scalar (whatever dtype the
        # 'start' column happens to be, e.g. int32 in some pandas/numpy
        # version or code path), and numpy arithmetic between a
        # fixed-width integer type and a plain Python int STAYS in that
        # same fixed-width type rather than safely upcasting (confirmed
        # directly: np.int32(2_147_483_600) + 100 silently wraps around
        # to a large NEGATIVE number, with only a RuntimeWarning, not an
        # exception) - genomic coordinates for a large genome can
        # realistically approach or exceed that range. Converting each
        # coordinate to a plain Python int FIRST, before doing the
        # widening arithmetic, guarantees the arithmetic itself happens
        # in Python's own arbitrary-precision integers - which cannot
        # overflow at all, regardless of how large the numbers are -
        # rather than in whatever fixed-width numpy type the column
        # happened to be, whether that's ever actually int32 in practice
        # or not. gene_adjust itself is also explicitly rounded to a
        # plain int first, for the same reason (a non-integer widening
        # amount doesn't correspond to a real base-pair distance anyway).
        _gene_adjust_int = int(round(gene_adjust))
        # ver4-4 R7.a - see the chromosome start/end vectorisation above.
        # NOTE: unlike the chromosome conversion above, this one never
        # rounded - it truncated via plain int(...) - so this uses
        # .astype('int64') (which truncates toward zero, exactly like
        # Python's int() on a float) rather than .round().astype('int64'),
        # to reproduce that exact (documented, deliberate - see the
        # overflow-safety comment above) truncating behaviour.
        gene['start'] = gene['start'].astype('int64') - _gene_adjust_int
        gene['end'] = gene['end'].astype('int64') + _gene_adjust_int

        chromosome_total = pd.unique(gene['chromosome'])
        _unmatched_chromosomes = [c for c in chromosome_total if chromosome.loc[chromosome['chromosome'] == c].shape[0] == 0]
        if _unmatched_chromosomes:
            # Requirement (diagnostic - a real, confirmed cause of gene
            # regions being invisible, found by checking a real user's
            # actual chrom_info/gene_info files directly): if a gene's
            # chromosome name never matches ANY row in the chromosome info
            # file, its widened start/end below never get clamped at all
            # (the loop simply skips it), and - more fundamentally -
            # plot()'s own rendering match (`sector.name == rec.chr`)
            # will never find that chromosome either, so the gene never
            # gets drawn at all, however large GENE_ADJUST is. A common,
            # easy-to-make cause: the two files using different naming
            # conventions for the same chromosomes (e.g. chrom_info using
            # 'A10' while gene_info uses '10A' for the exact same
            # chromosome - confirmed to happen in practice, not a
            # hypothetical). Printed as a clear, specific warning rather
            # than left to be silently invisible - this is exactly the
            # kind of problem that looks identical to 'GENE_ADJUST is too
            # small' from the rendered plot alone, but no amount of
            # widening fixes it.
            _all_chrom_names = sorted(pd.unique(chromosome['chromosome']).tolist())
            print(
                f"[circos] WARNING: {len(_unmatched_chromosomes)} chromosome name(s) in the gene info "
                f"file have NO match at all in the chromosome info file, so genes on them can never "
                f"be drawn (this is a common cause of a gene ring looking empty, regardless of "
                f"GENE_ADJUST): {_unmatched_chromosomes[:10]}{'...' if len(_unmatched_chromosomes) > 10 else ''}\n"
                f"[circos]   Chromosome names in the chromosome info file: "
                f"{_all_chrom_names[:10]}{'...' if len(_all_chrom_names) > 10 else ''}\n"
                f"[circos]   Check whether the two files are using different naming conventions for "
                f"the same chromosomes (e.g. 'A10' vs '10A')."
            )
        for k in range(len(chromosome_total)):
            _chrom_mask = gene['chromosome'] == chromosome_total[k]
            _chrom_row = chromosome.loc[chromosome['chromosome'] == chromosome_total[k]]
            if _chrom_row.shape[0] == 0:
                continue
            _chrom_start = int(_chrom_row['start'].values[0])
            _chrom_end = int(_chrom_row['end'].values[0])
            _clamp_region_to_chromosome(gene, _chrom_mask, _chrom_start, _chrom_end)

        gene_population = pd.unique(gene['population'])
        gene_source = pd.unique(gene['source'])
        print(f"[circos] data_conversion(): gene file's own distinct population values "
              f"(what gene rings will actually be written for): {list(gene_population)}")

        for i in range(len(PHENOTYPE)):
            for j in range(len(gene_population)):
                for k in range(len(gene_source)):
                    gene_selected = gene[(gene['population']==gene_population[j]) & (gene['phenotype']==PHENOTYPE[i]) & (gene['source']==gene_source[k])]
                    gene_selected = gene_selected.drop(['source','population','phenotype'],axis=1)
                    if gene_selected.shape[0] != 0:
                        # Requirement (bugfix): this MUST be
                        # gene_population[j] (the loop's own index
                        # variable, i.e. the population this specific
                        # gene_selected block actually belongs to) - it
                        # was chromosome_population[j] before, a
                        # DIFFERENT array from an entirely different
                        # file, coincidentally the same length/order in
                        # simple cases (masking the bug) but never
                        # guaranteed to be, and genuinely NOT
                        # guaranteed to be once chromosome_population's
                        # dtype could differ unpredictably from
                        # gene_population's (see the two explicit str
                        # casts added above) - a mismatch here silently
                        # writes a gene track under the WRONG
                        # population's filename, which plot() would
                        # then never find when it goes looking for the
                        # CORRECT population's own gene file, exactly
                        # matching a 'gene ring silently missing for
                        # some population' symptom.
                        gene_selected.to_csv(_circos_intermediate_dir(RESULT_NAME)+'/gene_info_'+str(PHENOTYPE[i])+'_'+str(gene_source[k])+'_'+str(gene_population[j])+'.tsv', sep='\t', index=False)
        
        pop_source = gene.loc[:,['phenotype','population','source']].drop_duplicates()
    
    else:
        pop_source = None
        
    return pop_source

def _window_division(window, chrom_end_max):
    """Bin edges ``[window*0, window*1, ..., window*k]`` for
    ``quantile_conversion()``'s ``pd.cut(...)`` - performance fix.

    The previous implementation built this list with a Python ``for`` loop
    that ran once per bin, and ``quantile_conversion()`` calls it for EVERY
    (phenotype, population) pair. Its iteration bound was
    ``int(chromosome['end'].max())`` (one iteration per coordinate unit
    when ``window`` is small), so a chromosome length expressed in bp with
    a modest ``WINDOW`` meant hundreds of millions of interpreter-level
    iterations per call. This computes the identical edges with one
    ``numpy`` expression.

    Semantics reproduced exactly from the loop it replaces: append
    ``window*cnt`` for ``cnt = 0, 1, 2, ...``; stop right after appending
    the first ``window*cnt`` whose successor ``window*(cnt+1)`` exceeds
    ``chrom_end_max``; never append more than ``chrom_end_max`` values.
    A non-positive ``window`` is handed to the original loop unchanged.
    """
    if window <= 0:
        division = []
        cnt = 0
        for n in range(int(chrom_end_max)):
            division += [window*cnt]
            cnt += 1
            if window*cnt > int(chrom_end_max):
                break
        return division
    k = int(np.floor(chrom_end_max / window))
    while window * (k + 1) <= chrom_end_max:
        k += 1
    while k > 0 and window * k > chrom_end_max:
        k -= 1
    n_edges = min(k + 1, max(int(chrom_end_max), 0))
    return (window * np.arange(n_edges)).tolist()


# ---------------------------------------------------------------------------
# ver4-31, R10 (bug fix): which HUE ('red' = ensemble ring, 'blue' = single
# model ring) a marker-effect ring gets. This used to be an exact-name list
# in two places (quantile_conversion() and plot()) that missed every
# 'ensemble__<algo>' label (per-method HP_TUNE grouping) and 'Joint Bayesian
# optimisation' (ver4-26), colouring those ensemble rings blue. The label
# predicates below are the SAME authority pipeline_utils / diversity_summary
# / weight_plot use to recognise an ensemble label (invariant I3).
# `ensemble_hue_rule='legacy_list'` (CIRCOS_CONFIG) restores the old list
# exactly, as a rollback.
# ---------------------------------------------------------------------------
_LEGACY_ENSEMBLE_HUE_LABELS = ('ensemble', 'Linear transformation', 'Nelder Mead',
                               'Bayesian optimisation', 'Analytic least-squares')
ENSEMBLE_HUE_RULES = ('predicates', 'legacy_list')


def _ring_hue(label, rule='predicates'):
    """`'red'` for an ensemble ring, `'blue'` for a single-model ring."""
    if rule == 'legacy_list':
        return 'red' if label in _LEGACY_ENSEMBLE_HUE_LABELS else 'blue'
    return 'red' if (is_naive_ensemble_label(label) or is_wopt_label(label)) else 'blue'


def _ensemble_hue_rule(circos_config):
    rule = (circos_config or {}).get('ensemble_hue_rule', 'predicates')
    return rule if rule in ENSEMBLE_HUE_RULES else 'predicates'


# --------------------------------------------------------------------------- #
# ver4-31 R14: label the top-N markers (or the genes they fall in) of one model
# outside the outer ring. `quantile_conversion()` writes the sidecar table
# (it is the only place the continuous effects are still around); `plot()`
# reads it back and draws the labels.
# --------------------------------------------------------------------------- #

LABEL_SOURCE_CONSENSUS_NAMES = ('consensus', 'consensus_ring')   # ver4-45: `label_source` values meaning "the consensus ring"
_LABEL_POOL_MAX = 3000       # ver4-45: cap on the candidate pool written beside the top-N label table


def _label_options(circos_config):
    """`{'top', 'source', 'min_sep'}` from `CIRCOS_CONFIG.label_top_markers`
    (0 = off), `.label_source` ('ensemble' or a model label) and
    `.label_min_sep_deg`."""
    c = circos_config or {}
    try:
        top = max(int(c.get('label_top_markers', 0) or 0), 0)
    except (TypeError, ValueError):
        top = 0
    try:
        sep = float(c.get('label_min_sep_deg', 2.0))
    except (TypeError, ValueError):
        sep = 2.0
    _src = str(c.get('label_source', 'ensemble') or 'ensemble').strip() or 'ensemble'
    return {'top': top, 'source': _src,
            'consensus': _src.lower() in LABEL_SOURCE_CONSENSUS_NAMES,      # ver4-45: label the consensus ring's top windows
            'min_sep': sep if sep > 0 else 2.0}


def _label_model(MODEL, source):
    """Which model's effects are labelled: `source` when it names a model in
    `MODEL`; for 'ensemble' the naive ensemble ('ensemble' /
    'ensemble__<algo>') when present; otherwise the first model."""
    models = list(MODEL)
    if not models:
        return None
    if source in models:
        return source
    if source == 'ensemble':
        for m in models:
            if is_naive_ensemble_label(m):
                return m
    return models[0]


def _label_table_path(RESULT_NAME, PHENOTYPE, POPULATION):
    return _circos_intermediate_dir(RESULT_NAME)+'/marker_effect_top_'+str(PHENOTYPE)+'_'+str(POPULATION)+'.tsv'


def _atomic_to_csv(frame, path, **kw):
    """`frame.to_csv(path)` via a per-process temp file + `os.replace`, so a
    sibling worker process (circos renders one (phenotype, population) across
    several processes, each rebuilding the same consensus/label sidecars) never
    reads a half-written file."""
    tmp = f"{path}.tmp{os.getpid()}"
    frame.to_csv(tmp, **kw)
    os.replace(tmp, path)


def _label_pool_path(label_tsv):
    """Sidecar holding a LONGER, ranked candidate list for the same label table
    (ver4-45). The N strongest rows are what the user-facing table lists; the
    pool is what `_add_top_marker_labels()` draws from, so that when several of
    the top-N hits collapse onto one gene label (or fall on a chromosome that is
    not plotted) the next-strongest candidates take their place and exactly N
    distinct labels are drawn."""
    return label_tsv[:-4] + '_pool.tsv' if label_tsv.endswith('.tsv') else label_tsv + '_pool'


def _write_label_table(path, frame, top, pool_n=None):
    """Write the `top` largest-effect rows of `frame` (columns chromosome, start,
    end, name, value[, tiebreak]) as chromosome / pos / name / value, and (ver4-45)
    a ranked candidate pool of up to `pool_n` rows beside it (`_label_pool_path`).
    `tiebreak`, when present, orders rows whose `value` is equal (the consensus
    ring's integer counts tie constantly)."""
    f = frame.copy()
    f['value'] = pd.to_numeric(f['value'], errors='coerce')
    f = f[np.isfinite(f['value'].to_numpy(dtype=float))]
    sort_cols, ascending = ['value'], [False]
    if 'tiebreak' in f.columns:
        f['tiebreak'] = pd.to_numeric(f['tiebreak'], errors='coerce').fillna(0.0)
        sort_cols, ascending = ['value', 'tiebreak'], [False, False]
    f = f.sort_values(sort_cols, ascending=ascending, kind='mergesort')
    top = int(top)
    if pool_n is None:
        pool_n = min(max(top * 10, top + 50), _LABEL_POOL_MAX)

    def _shape(g):
        return pd.DataFrame({'chromosome': g['chromosome'].astype(str),
                             'pos': (pd.to_numeric(g['start']) + pd.to_numeric(g['end'])) / 2.0,
                             'name': g['name'].astype(str), 'value': g['value']})

    _atomic_to_csv(_shape(f.head(top)), path, sep='\t', index=False)
    _atomic_to_csv(_shape(f.head(max(int(pool_n), top))), _label_pool_path(path), sep='\t', index=False)


_GENE_LABEL_UNAVAILABLE_LOGGED = False


def _gene_names_for_markers(label_frame, gene_tsvs):
    """`{marker name: gene name}` for the labelled markers that fall inside a
    gene-ring region (interval overlap via the shared `pipeline_utils.
    markers_in_windows`, I6). Empty when there is no gene ring or the helper
    cannot run (logged once)."""
    global _GENE_LABEL_UNAVAILABLE_LOGGED
    frames = []
    for path in gene_tsvs:
        try:
            g = pd.read_csv(path, sep='\t', dtype={'chromosome': str})
        except Exception:
            continue
        if {'chromosome', 'start', 'end'}.issubset(g.columns) and g.shape[0]:
            name_col = 'name' if 'name' in g.columns else ('index' if 'index' in g.columns else None)
            if name_col is not None:
                frames.append(g[['chromosome', 'start', 'end', name_col]].rename(columns={name_col: 'name'}))
    if not frames:
        return {}
    regions = pd.concat(frames, ignore_index=True)
    marker_df = pd.DataFrame({'chromosome': label_frame['chromosome'].astype(str),
                              'name': label_frame['name'].astype(str),
                              'start': label_frame['pos'], 'end': label_frame['pos']})
    try:
        from pipeline_utils import markers_in_windows
        hits = markers_in_windows(regions, marker_df, marker_df['name'].tolist(), window=0.0,
                                  mode='all_in_window')
    except Exception as exc:
        if not _GENE_LABEL_UNAVAILABLE_LOGGED:
            print(f"[circos_plot] NOTE: gene names for the labelled markers unavailable ({exc!r}); "
                  f"marker names are used instead.")
            _GENE_LABEL_UNAVAILABLE_LOGGED = True
        return {}
    out = {}
    for gene_name, members in zip(regions['name'].astype(str).tolist(), hits):
        for m in members:
            out.setdefault(str(m), gene_name)
    return out


def quantile_conversion(effect_grouped_all, effect_grouped_pop, marker_info, chrom_info, PHENOTYPE, MODEL, end_adjust, POPULATION, WINDOW, RESULT_NAME, ASCENDING, ensemble_hue_rule='predicates', label_cfg=None):
    """Assign each marker a 10-level colour bucket ('<hue><0-9>') from its
    mean effect for one (PHENOTYPE, POPULATION) pair, per model, and write
    one `marker_effect_<model>_<phenotype>_<population>.tsv` per surviving
    model. Returns `MODEL` with any model that had no data for this
    (PHENOTYPE, POPULATION) pair removed (`REMOVE`, below) - the caller
    (`circos_plot()`) keeps this filtered result PER ITERATION rather than
    rebinding its own loop-invariant `MODEL` (ver4-4 R7, the latent
    MODEL-rebinding bug fixed in Stage 1).

    ver4-4 R7.f: `effect_grouped_all`/`effect_grouped_pop` are the
    ALREADY-``abs()``ed, ALREADY-grouped-and-averaged frames
    `circos_plot()` computes exactly ONCE, before its own PHENOTYPE x
    POPULATION loop (see that function) - `effect_grouped_all` for
    `POPULATION == 'all'` (grouped by phenotype+model only) and
    `effect_grouped_pop` for every other, real population (grouped by
    population+phenotype+model). This function used to take the RAW
    `effect` frame and redo BOTH the `.abs()` cast AND the full
    `.groupby().mean()` itself, on EVERY call - i.e. once per (phenotype,
    population) pair, even though neither computation's result actually
    varies across that loop. Worse, the old `effect.iloc[:,5:] =
    effect.iloc[:,5:].abs().astype(float)` line mutated the CALLER's own
    `effect` object in place (a correctness bug in its own right - see
    the R7.f note in `circos_plot()`, which now hands this function an
    already-abs'd COPY instead of raw data it could mutate).
    """
    chromosome = pd.read_csv(_circos_intermediate_dir(RESULT_NAME)+'/chrom_'+str(POPULATION)+'.bed', delimiter='\t')
    chromosome['chromosome'] = chromosome['chromosome'].astype(str)
    
    if WINDOW != 0:
        division = _window_division(WINDOW, int(chromosome['end'].max()))

    effect_grouped = effect_grouped_all if POPULATION == 'all' else effect_grouped_pop
    
    REMOVE = []

    _lopts = _label_options(label_cfg)
    _label_model_name = (_label_model(MODEL, _lopts['source'])
                         if (_lopts['top'] > 0 and not _lopts['consensus']) else None)   # ver4-45: consensus labels are built in plot()
    
    for iii in range(len(MODEL)):
        colour = _ring_hue(MODEL[iii], ensemble_hue_rule)      # ver4-31 R10
        if POPULATION == 'all':
            effect_selected = effect_grouped[(effect_grouped['model']==MODEL[iii]) & (effect_grouped['phenotype']==PHENOTYPE)].iloc[:,3:].T
        else:
            # Requirement 7 (bugfix, found while fixing the all-zero-
            # effect crash): iloc[:,3:] here only skips 3 columns
            # (assuming population/phenotype/model precede the marker
            # columns), but this branch's own groupby(['population',
            # 'phenotype','model']).mean().reset_index() actually
            # produces population, phenotype, model, sample, <markers...>
            # - the replicate number ('sample') sits between 'model' and
            # the first real marker column, so iloc[:,3:] was silently
            # including it as if it were itself a marker's effect value
            # (only ever discarded much later, once merged against
            # marker_info.csv - by which point it had already
            # contaminated the quantile-threshold computation every
            # marker's colour level is chosen from). iloc[:,4:] correctly
            # skips all 4 metadata columns. The POPULATION == 'all'
            # branch above groups by phenotype/model only (population
            # was already dropped before that groupby), so its own
            # reset_index() only ever re-adds 2 columns - its iloc[:,3:]
            # already lands on the first real marker column correctly,
            # and is intentionally left alone.
            effect_selected = effect_grouped[(effect_grouped['model']==MODEL[iii]) & (effect_grouped['phenotype']==PHENOTYPE) & (effect_grouped['population']==str(POPULATION))].iloc[:,4:].T
           
        if effect_selected.shape[1] != 0:
            # Requirement 7 (bugfix): if EVERY marker's effect for this
            # model is exactly 0, the quantile-threshold approach below
            # breaks down in two different ways depending on WINDOW:
            #   - WINDOW == 0: every np.quantile(...) breakpoint also
            #     collapses to 0, so every '>= threshold' comparison
            #     matches EVERY marker at EVERY level in turn, and the
            #     LAST assignment (colour9 - the highest/darkest level)
            #     silently overwrites all the earlier ones - exactly
            #     backwards from what an all-zero effect means.
            #   - WINDOW != 0: rows get filtered to effect > 0 further
            #     down BEFORE any quantile is computed - with every
            #     effect at exactly 0, that filter empties the
            #     DataFrame entirely, and np.quantile() on an empty
            #     array raises an exception outright.
            # Detected once, up front, for both branches: skip the
            # quantile computation entirely and assign colour0 (the
            # lowest/faintest level) to every marker directly instead -
            # correct either way, and never touches np.quantile() on
            # data that can't support it.
            #
            # Requirement (SECOND correction - confirmed against a real
            # crash report): this used to check '== 0' specifically -
            # but the actual filter a few lines below (WINDOW != 0
            # branch) is 'effect > 0', not 'effect != 0'. Those are NOT
            # the same condition: a genomic marker's effect on a trait
            # can legitimately be NEGATIVE (decreases the trait, not
            # just 'has no effect'), and a set of effects that are all
            # zero-or-negative (no marker strictly positive, but not
            # literally every single one exactly 0 either) still passes
            # '== 0).all()' as False - so the ORIGINAL fix's own
            # detection silently failed to catch this close cousin of
            # the exact case it was written for, and the SAME empty-
            # DataFrame crash it was meant to prevent still happened.
            # Checking '<= 0' instead matches the real filter condition
            # exactly, catching every input that filter would empty out
            # - not just the narrower all-exactly-zero case.
            _all_zero_effect = bool((effect_selected.to_numpy(dtype=float) <= 0).all())

            if WINDOW == 0:
                effect_selected_copy = effect_selected.copy().astype(object)

                if _all_zero_effect:
                    effect_selected_copy.iloc[:,0] = colour+'0'
                else:
                    # ver4-4 R7.e - vectorised equivalent of the previous
                    # nine separate np.quantile(...) calls (each
                    # re-flattening and re-scanning the SAME array) plus
                    # nine separate boolean-mask '>= threshold' passes.
                    # Computed ONCE as a single ascending array of the 9
                    # decile thresholds, then np.searchsorted(...,
                    # side='right') assigns each value's level directly:
                    # side='right' counts every threshold a value is
                    # >= to (ties included, matching '>=' exactly),
                    # reproducing the ORIGINAL cumulative overwrite
                    # chain's own "highest threshold this value meets or
                    # exceeds" semantics exactly, level-for-level and
                    # tie-for-tie - verified directly against the
                    # original nine-call chain on a fixture with ties and
                    # exact-threshold values (R7.7 acceptance criterion).
                    _values = effect_selected.to_numpy(dtype=float).flatten()
                    _thresholds = np.quantile(_values, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
                    _levels = np.searchsorted(_thresholds, _values, side='right')
                    effect_selected_copy.iloc[:, 0] = [f"{colour}{lvl}" for lvl in _levels]
                
                effect_selected_copy.columns = ['colour']
                if ASCENDING is not None:
                    effect_selected_copy = effect_selected_copy.sort_values(by=['colour'], ascending=ASCENDING)
                effect_selected_copy = effect_selected_copy.reset_index(drop=False)
                
                marker = _load_combined_marker_info(marker_info, RESULT_NAME, PHENOTYPE)
                merged = pd.merge(effect_selected_copy, marker, left_on=['index'], right_on=['name'])
                if _label_model_name is not None and MODEL[iii] == _label_model_name:
                    # ver4-31 R14: the labelled model's top markers (marker centre, effect).
                    _lt = merged[['chromosome', 'start', 'end', 'index']].rename(columns={'index': 'name'})
                    _lt['value'] = _lt['name'].map(effect_selected.iloc[:, 0].astype(float))
                    _write_label_table(_label_table_path(RESULT_NAME, PHENOTYPE, POPULATION), _lt, _lopts['top'])
                merged = merged.loc[:,['chromosome','start','end','index','colour']]
                # Requirement (bugfix - END_ADJUST could overflow into a
                # wildly wrong number for a large coordinate): explicit
                # .astype('int64') BEFORE the widening arithmetic below,
                # not left at whatever dtype 'start'/'end' happened to
                # arrive in from the merge above - a narrower fixed-width
                # integer type (e.g. int32) silently wraps around to a
                # large negative/wrong number once it overflows, rather
                # than raising an error (confirmed directly:
                # np.int32(2_147_483_600) + 100 wraps to a large negative
                # number, with only a RuntimeWarning) - genomic
                # coordinates for a large genome can realistically
                # approach or exceed int32's range. int64's own range
                # (+-~9.2 quintillion) is, by contrast, far beyond any
                # real genomic coordinate (even the largest known genome
                # is under 150 billion bp in total), so this is a safe,
                # practically-unlimited ceiling for this domain without
                # needing Python's fully arbitrary-precision int (which
                # isn't available for a vectorised pandas column the way
                # it is for the scalar per-row case in data_conversion()
                # above).
                merged['start'] = merged['start'].astype('int64')
                merged['end'] = merged['end'].astype('int64')
                _end_adjust_int = int(round(end_adjust))
                merged['start'] = merged['start'] - _end_adjust_int
                merged['end'] = merged['end'] + _end_adjust_int
                merged['chromosome'] = merged['chromosome'].astype(str)

                chromosome_total = pd.unique(merged['chromosome'])

                # Requirement (bugfix): clamp BOTH the widened start and
                # end to THIS SPECIFIC chromosome's own bounds - not a
                # hardcoded 0 for start (chromosomes don't always start
                # at coordinate 0 in every coordinate system a chrom_info
                # file might use), and not another chromosome's bounds by
                # mistake. Both sides are handled together, per
                # chromosome, so a widened marker region can never spill
                # outside where its own chromosome actually starts/ends.
                for k in range(len(chromosome_total)):
                    _chrom_mask = merged['chromosome'] == chromosome_total[k]
                    _chrom_row = chromosome.loc[chromosome['chromosome'] == chromosome_total[k]]
                    if _chrom_row.shape[0] == 0:
                        continue
                    _chrom_start = int(_chrom_row['start'].values[0])
                    _chrom_end = int(_chrom_row['end'].values[0])
                    _clamp_region_to_chromosome(merged, _chrom_mask, _chrom_start, _chrom_end)
                merged.to_csv(_circos_intermediate_dir(RESULT_NAME)+'/marker_effect_'+str(MODEL[iii])+'_'+str(PHENOTYPE)+'_'+str(POPULATION)+'.tsv', sep ='\t',index=False)
            else:
                effect_selected.columns = ['effect']
                effect_selected = effect_selected.reset_index(drop=False)
                marker = _load_combined_marker_info(marker_info, RESULT_NAME, PHENOTYPE)
                effect_selected = pd.merge(effect_selected, marker, left_on=['index'], right_on=['name'])
                effect_selected = effect_selected.loc[:,['chromosome','start','end','index','effect']]

                effect_selected['chromosome'] = effect_selected['chromosome'].astype(str)
                
                effect_selected['range'] = (effect_selected['start'] + effect_selected['end'])/2
                
                chromosome_total = pd.unique(effect_selected['chromosome'])
                
                for k in range(len(chromosome_total)):
                    effect_selected.loc[(effect_selected['chromosome']==chromosome_total[k]) & 
                               (effect_selected['range'] > chromosome.loc[chromosome['chromosome']==chromosome_total[k],'end'].values[0]),'range'] = int(chromosome.loc[chromosome['chromosome']==chromosome_total[k], 'end'].values[0])

                effect_selected = effect_selected.groupby(['chromosome',pd.cut((effect_selected['range']), bins=division)]).sum().drop(['start','end', 'range'],axis=1).reset_index(drop=False)
                effect_selected = effect_selected.rename(columns={'range':'interval'})
                # ver4-4 R7.a - vectorised equivalent of the previous
                # per-row [int(round(iv.left)) for k in range(...)] /
                # [int(iv.right) for k in range(...)] loops over this
                # column's pandas Interval objects (from the pd.cut(...)
                # bins used to group effect_selected just above).
                # IntervalArray.left/.right already return the per-bin
                # edges as a single vectorised array - this only adds the
                # SAME int(round(...)) / int(...) casts the loops applied,
                # preserving the deliberate asymmetry between the two
                # (start is rounded, end is truncated - unchanged here).
                _interval_arr = effect_selected['interval'].array
                effect_selected['start'] = np.round(np.asarray(_interval_arr.categories.left[_interval_arr.codes], dtype=float)).astype('int64')
                effect_selected['end'] = np.asarray(_interval_arr.categories.right[_interval_arr.codes], dtype=float).astype('int64')
                
                effect_selected = effect_selected.drop('interval',axis=1)
                # Requirement (bugfix): clamp to each bin's own
                # chromosome start, not a hardcoded 0 - matching the same
                # fix already applied elsewhere in this file (chromosomes
                # don't always start at coordinate 0 in every coordinate
                # system a chrom_info file might use).
                for k in range(len(chromosome_total)):
                    _chrom_row = chromosome.loc[chromosome['chromosome'] == chromosome_total[k]]
                    if _chrom_row.shape[0] == 0:
                        continue
                    _chrom_start = int(_chrom_row['start'].values[0])
                    effect_selected.loc[
                        (effect_selected['chromosome'] == chromosome_total[k]) &
                        (effect_selected['start'] < _chrom_start), 'start'
                    ] = _chrom_start
                if not _all_zero_effect:
                    _positive_effect_selected = effect_selected[effect_selected['effect'] > 0].reset_index(drop=True)
                    if _positive_effect_selected.shape[0] == 0:
                        # Requirement (bugfix - crash report, ver4-11
                        # circos IndexError): binning GROUPS several
                        # markers together and SUMS their effects
                        # (the groupby(...).sum() a few lines above)
                        # before this filter ever runs. `_all_zero_effect`
                        # above is computed from the RAW, pre-binning
                        # per-marker values, so "not every raw marker's
                        # effect is <= 0" does NOT guarantee "at least
                        # one bin's SUMMED effect is > 0" - a few small
                        # positive markers can still net negative once
                        # summed into the same bin as larger negative
                        # neighbours. When every bin nets <= 0 this way,
                        # the '> 0' filter below used to empty
                        # effect_selected entirely even though
                        # _all_zero_effect was False, and np.quantile()
                        # on the resulting empty array raised
                        # 'IndexError: index -1 is out of bounds for
                        # axis 0 with size 0' a few lines down. Falling
                        # through to the SAME colour0-for-every-row
                        # treatment the genuine all-zero-effect case
                        # already gets (the 'else' branch just below) is
                        # correct here too: there is no positively-net
                        # bin left to build a decile scale from, so
                        # every bin is coloured the lowest/faintest
                        # level instead of crashing.
                        _all_zero_effect = True
                    else:
                        effect_selected = _positive_effect_selected
                # else: keep every row (would otherwise be filtered down
                # to 0 rows, since every effect is exactly 0) - all get
                # assigned colour0 directly below instead.


                effect_selected_copy = effect_selected.copy()
                effect_selected_copy = effect_selected_copy.astype({effect_selected_copy.columns[2]: object})
                effect_selected_copy.iloc[:,2] = colour+'0'
                if not _all_zero_effect:
                    # ver4-4 R7.e - same vectorisation as the WINDOW==0
                    # branch above: one ascending array of 9 decile
                    # thresholds, one np.searchsorted(..., side='right')
                    # call assigns every row's level directly, reproducing
                    # the original nine-call cumulative-overwrite chain's
                    # "highest threshold this value meets or exceeds"
                    # semantics exactly (see the WINDOW==0 branch's own
                    # comment for the full boundary-semantics argument).
                    _values = effect_selected['effect'].to_numpy(dtype=float)
                    _thresholds = np.quantile(_values, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
                    _levels = np.searchsorted(_thresholds, _values, side='right')
                    effect_selected_copy.iloc[:, 2] = [f"{colour}{lvl}" for lvl in _levels]
                
                if _label_model_name is not None and MODEL[iii] == _label_model_name:
                    # ver4-31 R14: the labelled model's top windows (window centre, summed effect).
                    _lt = effect_selected[['chromosome', 'start', 'end']].copy()
                    _lt['name'] = [f"{c}:{int((a + b) / 2)}" for c, a, b in
                                   zip(_lt['chromosome'], _lt['start'], _lt['end'])]
                    _lt['value'] = effect_selected['effect'].to_numpy(dtype=float)
                    _write_label_table(_label_table_path(RESULT_NAME, PHENOTYPE, POPULATION), _lt, _lopts['top'])
                merged = effect_selected_copy.loc[:,['chromosome', 'start', 'end', 'index', 'effect']]
                merged = merged.rename(columns={'effect':'colour'})
                
                # Requirement (bugfix): clamp BOTH start and end to each
                # bin's own chromosome, matching the same fix already
                # applied in the WINDOW==0 branch above and in
                # data_conversion() - this branch previously only ever
                # clamped 'end' here (never 'start'; that used a
                # hardcoded 0 further up instead of a chromosome's own
                # start - see effect_selected['start'] a few lines above)
                # and did so with a genuine crash-causing bug of its own:
                # int(...values) without an index on a (possibly multi-
                # element, or on newer numpy, even single-element) array
                # raises TypeError rather than converting - int(...
                # values[0]) is the correct scalar extraction, matching
                # every other clamp in this file.
                for k in range(len(chromosome_total)):
                    _chrom_mask = merged['chromosome'] == chromosome_total[k]
                    _chrom_row = chromosome.loc[chromosome['chromosome'] == chromosome_total[k]]
                    if _chrom_row.shape[0] == 0:
                        continue
                    _chrom_start = int(_chrom_row['start'].values[0])
                    _chrom_end = int(_chrom_row['end'].values[0])
                    _clamp_region_to_chromosome(merged, _chrom_mask, _chrom_start, _chrom_end)

                merged.to_csv(_circos_intermediate_dir(RESULT_NAME)+'/marker_effect_'+str(MODEL[iii])+'_'+str(PHENOTYPE)+'_'+str(POPULATION)+'.tsv', sep ='\t',index=False)
        else:
            REMOVE += [MODEL[iii]]
    
    MODEL = [e for e in MODEL if e not in REMOVE]
    
    return MODEL

def _select_top_interactions(df, circos_config):
    """Requirements.md item 4: select which marker-pair interactions are
    strong enough to draw as links on the circos plot - either the top
    N% by value (this module's original, still-default behaviour) or
    the top M (an absolute count), whichever
    `circos_config['interaction_top_mode']` selects.

    Shared by both branches of interaction() below (the RF branch and
    the GAT-attention branch) - previously each duplicated its own
    identical quantile-filtering line independently; centralising the
    decision here means the two branches can't drift apart on HOW 'top'
    is defined, only on which 'value'-bearing rows are being ranked.

    Parameters
    ----------
    df : pandas.DataFrame
        Already phenotype-filtered, grouped rows with a 'value' column
        (higher = a stronger interaction/attention weight) - the exact
        shape both call sites already build before calling this.
    circos_config : dict
        'interaction_top_mode' : 'percentage' (default - every existing
            config that predates this feature has no such key, and
            reproduces EXACTLY today's behaviour) or 'count'.
          - 'percentage': keeps a RANK-based top `interaction_top`% of
            `df` (`round(df.shape[0] * interaction_top / 100)` rows, by
            `.nlargest`), restricted to rows with a genuinely nonzero
            'value' - see "Bug fix" below.
          - 'count': keeps exactly the top `interaction_top_count` rows
            by `value` (fewer than that if `df` has fewer rows to begin
            with) - a literal top-M selection, no quantile involved.
            Unaffected by the bug fix below.

    Returns
    -------
    A filtered, index-reset copy of `df` - never mutates the input.

    Bug fix (quantile-threshold collapse under tied/zero values)
    --------------------------------------------------------------
    The original 'percentage' implementation (`df['value'] >=
    quantile(df['value'], 1 - interaction_top/100)`) silently breaks
    whenever a large share of `df['value']` is TIED at (or below) the
    computed quantile - most commonly at EXACTLY 0.0. This is not rare:
    every marker-pair-interaction extractor this codebase has
    (`models/interaction_extraction.py`) writes `value=0.0` for every
    pair it did NOT genuinely evaluate - most consequentially, every
    UNSCREENED pair whenever that module's own `h_statistic_interactions
    (screen=...)` pre-filter is enabled (`screen_keep` defaults to a
    mere 2%). Every 'value' column this function ever receives is
    non-negative by construction (SHAP-interaction magnitudes, Friedman's
    H^2, GAT attention weights, EBM/NID importances - see
    models.interaction_extraction.normalize_task_interactions()'s own
    docstring), so once that large zero-tied mass exceeds
    `100 - interaction_top` percent of `df`, `np.quantile(...)` itself
    resolves to `0.0` and `value >= 0.0` becomes true for EVERY row -
    confirmed directly: with a 98%-zero-tied reproduction (pre-screen at
    its own 2% default), this function jumped from a correctly-
    thresholded selection at `interaction_top=1` to keeping 100% of the
    table (every row, including every never-evaluated pair) at
    `interaction_top` in {5, 10, 50}. This is exactly the reported
    symptom: a circos interaction ring for one model/setting rendering a
    dense web of many more links than another model's ring at the
    IDENTICAL "top N%" setting, purely because that model's own
    interaction-value distribution happens to have a large tied-at-zero
    mass and the other's does not.

    The fix ranks by value instead of thresholding by quantile - `m =
    round(df.shape[0] * interaction_top / 100)` rows, taken by
    `.nlargest`, is always approximately `interaction_top`% of `df`
    regardless of how many rows tie at any given value. It is further
    capped to `df`'s own count of rows with `value > 0`, so a pair that
    was never actually evaluated (screened-out or degenerate, `value ==
    0.0` by construction) is never drawn as a link purely to pad the
    requested count out to `m` - drawing a link with no evidence behind
    it is worse than drawing fewer links than requested. This mirrors
    (and must be kept in sync with) the identical fix in
    `models.interaction_extraction.top_select()`'s own 'percentage'
    mode - see that function's own "Bug fix" note for the full
    reproduction.

    Bug fix (round-to-zero starvation on small candidate tables)
    --------------------------------------------------------------
    `_m = round(df.shape[0] * interaction_top / 100)` rounds DOWN to
    ``0`` whenever `df` has fewer than roughly ``50 / interaction_top``
    rows - e.g. under 50,000 rows at this GUI's own default
    `interaction_top=0.001` - even when `interaction_top` is genuinely
    positive and `nonzero` contains real, scored interactions. Confirmed
    directly: a 66-row shapiq_tree interaction table, entirely nonzero,
    renders zero links at this default (`_m` rounds to 0) while the
    identical table renders every link once `interaction_top=100`. This
    was invisible for the exhaustive `pairwise_shap`/`friedman_h` routes
    (whose shortlist-scan candidate tables comfortably exceed 50,000
    rows at the default 500-marker shortlist) but is the reliable,
    reported failure mode for every shapiq-based route (`shapiq_tree`,
    `shapiq`, `surrogate_shapiq`), whose budget-bounded or depth-capped
    output is inherently far smaller - "extracted interactions are not
    shown in circos plot when using shapiq". The fix floors `_m` at 1
    whenever a positive `interaction_top` was requested and at least one
    genuinely-scored (nonzero) row exists, so "some positive percentage
    of a nonempty, real result" never rounds down to "nothing shown".
    Mirrors (and must be kept in sync with) the identical fix in
    `models.interaction_extraction.top_select()`.
    """
    mode = circos_config.get('interaction_top_mode', 'percentage')
    if mode == 'count':
        _m = circos_config.get('interaction_top_count', 0) or 0
        _m = max(0, min(int(_m), df.shape[0]))
        return df.nlargest(_m, 'value').reset_index(drop=True)
    # 'percentage' (default) - rank-based top-N%, not a quantile
    # threshold. See this function's own "Bug fix" note above.
    nonzero = df[df['value'] > 0]
    _interaction_top = circos_config['interaction_top']
    _m = int(round(df.shape[0] * (_interaction_top / 100)))
    if _m == 0 and _interaction_top > 0 and nonzero.shape[0] > 0:
        # Bug fix (round-to-zero starvation) - see docstring above.
        _m = 1
    _m = max(0, min(_m, nonzero.shape[0]))
    return nonzero.nlargest(_m, 'value').reset_index(drop=True)


def _ensure_marker_pair_sum_count(df):
    """OOM fix (large Interaction.csv/Attention.csv files): normalise
    whatever `df` was handed in to a uniform 'value_sum'/'value_count'
    shape, so every caller below can always finish with one single
    `sum -> divide` step (`_regroup_and_finalize_mean()`), regardless of
    whether `df` is:

      - the ORIGINAL shape this module has always accepted: one row per
        raw (population, model, phenotype, marker1, marker2) OBSERVATION,
        carrying a single 'value' column - run_sequential.py and
        main_app.py's in-process 'run now' path still hand this in
        completely unchanged (GP()'s own in-memory accumulators were
        never the source of the OOM this fixes, only reading a large
        on-disk Interaction.csv/Attention.csv back in for Step-2
        plotting was - see batch_reader.aggregate_marker_pair_sums()'s
        own docstring); or
      - the NEW, memory-bounded shape `batch_reader.
        aggregate_marker_pair_sums()` produces for exactly that Step-2
        case: already collapsed to one row per (population, model,
        phenotype, marker1, marker2) COMBINATION, carrying 'value_sum'/
        'value_count' instead of a single 'value'.

    Either way, callers apply the row-level 'factor' sentinel filter
    BEFORE this function, so a 'factor' row from either shape is dropped
    identically before it can contribute to any sum."""
    if 'value_sum' in df.columns and 'value_count' in df.columns:
        return df
    df = df.copy()
    df['value_sum'] = df['value']
    df['value_count'] = 1
    return df


def _regroup_and_finalize_mean(df, group_cols):
    """The single `sum -> divide` step every caller finishes with, after
    `_ensure_marker_pair_sum_count()` above: sums 'value_sum'/
    'value_count' across whatever rows still share the same `group_cols`
    key - correctly RE-combining, e.g., two originally-different
    'between'-scenario population labels that `circos_plot()`'s own
    `str.split('->')` step upstream can leave mapped to the identical
    population string (see that call site's own comment) - then divides
    ONCE to produce the final per-key mean.

    Replaces every `.groupby(keys, as_index=False).mean(numeric_only=True)`
    call this module used to make directly on a raw-row 'value' column.
    For a `df` that started as one row per raw observation (i.e. came
    through `_ensure_marker_pair_sum_count()`'s pass-through branch,
    'value_count' all 1s), summing N ones and dividing the summed values
    by N is arithmetically the exact same number pandas' own `.mean()`
    would have produced - nothing changes for that caller except the
    intermediate column name.

    Update ID ver4-10 (performance fix): `interaction()`'s own two
    branches no longer call this function directly - see
    `_precompute_interaction_groups()`, which performs this EXACT same
    sum-then-divide reduction, but as ONE combined `.groupby(...).sum()`
    across every key at once rather than one call per (population, model,
    phenotype) key (hundreds of separate calls' worth of avoidable
    per-call overhead on a many-population/many-phenotype run - see that
    function's own docstring). This function is kept, unchanged, as the
    single documented reference for what that reduction IS and why it
    must happen this way (`batch_reader.py` still points here by name to
    explain why `aggregate_marker_pair_sums()` returns sum/count rather
    than a pre-finalised mean) - it is no longer on the hot path itself,
    but the invariant it documents still is.
    """
    grouped = df.groupby(list(group_cols), as_index=False)[['value_sum', 'value_count']].sum()
    grouped['value'] = grouped['value_sum'] / grouped['value_count']
    return grouped.drop(columns=['value_sum', 'value_count'])


def _precompute_interaction_groups(df, *, scope_model_order_by_population, circos_config=None, population_map=None, restrict=None):
    """Requirement (performance fix - circos plot generation "taking
    forever" once more than one interaction-emitting model is selected,
    e.g. RKHS + RF + SVR + KNN together):

    Before this function existed, `interaction()` below re-derived its
    per-(POPULATION, model, PHENOTYPE) marker-pair table from the FULL
    `interactions`/`attention` DataFrame from scratch, EVERY time it was
    called - and `circos_plot()` calls it once per (PHENOTYPE[i],
    POPULATION[j]) pair, i.e. `len(PHENOTYPE) * len(POPULATION)` times in
    total. Each of those calls re-ran the SAME 'factor'-row filter,
    `_ensure_marker_pair_sum_count()` normalisation, and
    `.groupby(...).mean()` reduction over the ENTIRE table (every model,
    every phenotype, every population at once), even though only a tiny
    slice of it was actually used by that one call. This was already
    wasteful with a single interaction-emitting model (RF, historically
    the only one); it now scales directly with how many models are
    selected, since `model_registry.py` wires up rrBLUP/GBLUP/BayesB/
    RKHS/RF/SVR/KNN/MLP as additional interaction emitters - four models
    selected together means a table roughly 4x the size, re-scanned in
    full the exact same `len(PHENOTYPE) * len(POPULATION)` number of
    times, for a roughly 4x-and-growing wall-clock cost with no change in
    what actually gets drawn.

    This mirrors the ver4-4 R7.f treatment already given to the `effect`
    frames in `circos_plot()` (`_effect_grouped_all`/`_effect_grouped_pop`,
    computed ONCE before that function's own (PHENOTYPE, POPULATION) loop
    rather than once per iteration inside `quantile_conversion()`) -
    `interaction()`'s two branches (the RF-family/interaction branch and
    the GAT-attention branch) never received the same treatment until
    now, which is why this symptom is specific to the marker-pair
    interaction/attention rings and not the per-marker effect rings.

    Runs the EXACT SAME reduction as before - the row-level 'factor'
    filter, then `_ensure_marker_pair_sum_count()`, then a
    `.groupby(...).mean()` - just ONCE up front over the whole table,
    keyed by every (population-or-'all', model, phenotype) combination
    actually present, instead of once per loop iteration. Numerically
    identical output to the pre-fix code; only WHEN the work happens
    changes. Accepts either the raw one-row-per-observation shape
    (Sequential/in-process "Option B: run now" - see `circos_plot.py`
    call sites) or the pre-aggregated 'value_sum'/'value_count' shape
    `batch_reader.aggregate_marker_pair_sums()`/`ResultSet.
    interactions_grouped()`/`attention_grouped()` produce (Parallel Step
    2 assemble, and the GUI's own "load an assembled result" path) -
    `_ensure_marker_pair_sum_count()` already normalises either shape
    identically, so this one function is what makes the fix apply the
    same way regardless of which of the three ways a run's interactions/
    attention data reached `circos_plot()` in the first place.

    Implementation note (found while benchmarking this fix against a
    many-population, many-phenotype run): the first version of this
    function still called `_regroup_and_finalize_mean()` - itself one
    `.groupby(...).sum()` call - once PER (population, model, phenotype)
    key, i.e. still hundreds of separate pandas groupby invocations, just
    no longer `len(PHENOTYPE) * len(POPULATION)` times each. Each
    invocation's fixed per-call overhead (hashing/sorting the grouping
    keys from scratch) dominated at that key-count, so this does the
    sum-and-divide reduction with exactly TWO combined `.groupby(...)
    .sum()` calls total (one for the 'all' cross-population aggregate,
    one for every named population at once) and only THEN splits each
    already-small reduced result into its per-key slices - splitting a
    `.groupby()` object that has already computed its group index is
    cheap per key; computing that index from scratch hundreds of times
    was not.

    Parameters
    ----------
    df : pandas.DataFrame
        The full `interactions` or `attention_original` table handed to
        `circos_plot()` - every model, every phenotype, every population
        at once (or empty).
    scope_model_order_by_population : bool
        Preserves a pre-existing, historical INCONSISTENCY between the
        two branches of `interaction()` exactly as it always behaved,
        rather than opportunistically "fixing" it as a side effect of
        this performance change: the interaction/RF-family branch always
        derived its per-model iteration order (`.unique()`) from the
        table ALREADY FILTERED to one POPULATION; the GAT-attention
        branch always derived its own model order from the FULL,
        population-UNFILTERED table. True reproduces the former
        (`interactions`); False reproduces the latter (`attention`).

    Returns
    -------
    (groups, model_order) :
        groups : dict {(population_key, model_name, phenotype_value):
            DataFrame[marker1, marker2, value]} - the fully reduced,
            per-key marker-pair table `interaction()` used to derive
            inline. `population_key` is either the literal 'all' (the
            cross-population aggregate `POPULATION == 'all'` has always
            used - population dropped entirely, never filtered) or the
            population's own `str(...)` label (matches a specific,
            already-`_clean_population_label()`-normalised
            `POPULATION[j]`).
        model_order : dict {population_key: [model_name, ...]} - the
            same first-appearance `.unique()` order the pre-fix code
            derived inline, computed once instead of once per
            (PHENOTYPE, POPULATION) call. Always has exactly one key
            ('all') when `scope_model_order_by_population` is False.

    Update ID circos-oom-fix (OOM fix - "memory error while creating the circos
    plot even with 900 GB", reproduced with four interaction-emitting
    models and ``interaction_top='all'``): the reduction above used to run
    as plain pandas ``groupby`` calls over five *object* (Python string)
    columns, i.e. ~150-300 bytes per (population, model, phenotype,
    marker1, marker2) group, made three more full-size copies of that table
    (`work`, `_all_reduced`, `_pop_reduced`), kept every per-key slice
    alive, and then PICKLED the whole dict into every worker process of the
    parallel fan-out. G (the distinct group count) scales with
    pairs x models x phenotypes x populations, so with several models and
    a large interaction shortlist it reaches hundreds of millions.

    It now goes through `interaction_reduce.PairSumStore`: names are
    dictionary-coded once (24 bytes per group), the reduction is a
    hash-factorize + bincount, the running total spills to disk past a
    resident-row budget, and - the key point - the "top N% / top M"
    selection circos applies anyway (`_select_top_interactions()`) is done
    HERE, per (population, model, phenotype) key, straight on the numeric
    arrays, so only the rows that will actually be drawn are ever decoded to
    strings or kept in `groups`. The numbers (means, which rows are kept,
    their order, tie-breaks) are identical to the old path; see
    `interaction_reduce.select_top_indices()`.

    Extra parameters
    ----------------
    df may also be a `PairSumStore` (what `ResultSet.interactions_grouped()`
    / `attention_grouped()` now return - already streamed and coded, never a
    materialised DataFrame).
    circos_config : when given, `groups` holds only the already-selected
        rows (``df.attrs['top_selected']`` is set so
        `_build_interaction_ring()` does not select a second time). None
        keeps the legacy "full per-key table" behaviour.
    population_map : optional ``str -> str`` population-label rewrite (the
        'between'-scenario ``'train->test'`` -> ``'test'`` split); labels
        that collide are merged by summing sum/count before the mean.
    restrict : optional ``{'mask_for', 'match'}`` from `_highlight_restrict()` - keep only the
        pairs involving the highlighted regions BEFORE the top-N%/top-M cut (see
        `PairSumStore.reduce_groups`). None (default) = legacy behaviour.
    """
    if isinstance(df, PairSumStore):
        store = df
    else:
        if df is None or df.shape[0] == 0:
            return {}, {}
        # in-memory tables keep the legacy first-appearance model order
        store = store_from_frame(df, order_mode='appearance')
    return store.reduce_groups(
        population_map=population_map,
        select=_selection_spec(circos_config),
        scope_model_order_by_population=scope_model_order_by_population,
        restrict=restrict,
    )


def _selection_spec(circos_config):
    """`interaction_reduce.PairSumStore.reduce_groups(select=...)` spec for
    `_select_top_interactions()`'s settings, or None when no config was
    given (legacy: keep every row, select later in `_build_interaction_ring`)."""
    if circos_config is None:
        return None
    return {
        'mode': circos_config.get('interaction_top_mode', 'percentage'),
        'top_percent': circos_config.get('interaction_top', 0),
        'top_count': circos_config.get('interaction_top_count', 0),
    }


# ---------------------------------------------------------------------------
# Update ID ver4-25, R12 (blueprint "Glyph geometry"/"Data flow"): k>=3
# interaction-hyperedge support - Option A ("one circos per k; hyperedges as
# hub-and-spoke glyphs", chosen over chord-decomposition (ambiguous - a
# triple reads identically to 3 independent pairs) and filled-polygon
# ribbons (occlusion) - see the blueprint's own design-space table). Every
# function in this block is ADDITIVE: none of it runs, and none of the
# existing k=2 chord code above is touched, unless `interactions_hok`
# actually has rows for the (PHENOTYPE, POPULATION, model, order) being
# rendered - which is only possible when a shapiq-based interaction method
# was selected with an order>=3 request (checkpoint_utils.
# _model_requests_higher_order_interactions()) - so a run that never
# touches this feature renders byte-identically to before it existed
# (AC12.3).
# ---------------------------------------------------------------------------

# Colourblind-safe qualitative palette (Okabe-Ito) used ONLY by the
# 'combined' higher-order layout's own "colour = k" scheme (blueprint's own
# wording) - the existing red/blue intra-/inter-chromosome convention used
# everywhere else in this module (every k=2 chord, and every k>=3 hub glyph
# under the DEFAULT 'per_order' layout) is completely untouched by this
# palette; it only ever appears on a 'combined'-layout render.
_OKABE_ITO_PALETTE = ('#E69F00', '#56B4E9', '#009E73', '#F0E442',
                      '#0072B2', '#D55E00', '#CC79A7', '#000000')


def _okabe_ito_colour_for_order(order: int) -> str:
    """`_OKABE_ITO_PALETTE`, cycled starting at k=2 - deterministic across
    every 'combined'-layout render (the SAME order always gets the SAME
    colour, run to run, plot to plot), which is what makes a `combined`
    figure's own per-order legend entries meaningful rather than
    arbitrary."""
    idx = max(0, int(order) - 2) % len(_OKABE_ITO_PALETTE)
    return _OKABE_ITO_PALETTE[idx]


LINK_SORTS = ('none', 'ascending')
LINK_SCALES = ('minmax', 'rank')
LINK_COLOUR_MODES = ('chrom', 'chrom_cvd', 'value')
# 'chrom_cvd' link colours: Okabe-Ito vermillion (intra) and a violet (inter).
# The blueprint proposed Okabe-Ito bluish-green / reddish-purple, but measured
# against blue0-9 / red0-9 of the default cytoband map with the Machado
# deuteranopia simulation and CIELab distance (main_app's own checker) those
# sit at Delta-E 20.1 / 14.7 - the reddish-purple misses the required 20.
# Vermillion / violet measure 33.2 / 25.8 under deuteranopia (and >= 25.8 in the
# worst case over all three dichromacies) and are 67 apart from each other.
_CVD_INTRA_COLOUR = '#D55E00'
_CVD_INTER_COLOUR = '#7B2CBF'
_LINK_VALUE_CMAP = 'cividis'


def _link_options(circos_config):
    """Validated ver4-31 (R11 / R12) link settings from `CIRCOS_CONFIG`; every
    key is optional and an absent / unknown value means the legacy behaviour
    (I11): `link_sort` ('none' | 'ascending'), `link_scale` ('minmax' |
    'rank'), `link_lw_range` (None | (lo, hi) line widths) and
    `link_colour_mode` ('chrom' | 'chrom_cvd' | 'value')."""
    c = circos_config or {}

    def _pick(key, allowed, default):
        v = str(c.get(key, default) if c.get(key, default) is not None else default).strip().lower()
        return v if v in allowed else default

    lw = c.get('link_lw_range')
    try:
        lo, hi = (float(lw[0]), float(lw[1])) if lw is not None else (None, None)
        lw = (lo, hi) if lo is not None and 0 < lo <= hi else None
    except (TypeError, ValueError, IndexError):
        lw = None
    return {
        'sort': _pick('link_sort', LINK_SORTS, 'none'),
        'scale': _pick('link_scale', LINK_SCALES, 'minmax'),
        'lw_range': lw,
        'colour_mode': _pick('link_colour_mode', LINK_COLOUR_MODES, 'chrom'),
    }


def _link_strength(values, circos_config, default_lw, alpha_min=None):
    """`(alpha, lw, s)` Series (index = `values`' index) for a set of link /
    hyperedge values: THE shared strength mapping of the k=2 chords and the
    k>=3 hub glyphs (R11). `s` in [0, 1] is the scaled strength - min-max
    within the displayed set (legacy) or its rank (`link_scale='rank'`, robust
    to one huge outlier); `alpha = alpha_min + s*(1 - alpha_min)`; `lw` is
    `default_lw` unless `link_lw_range=(lo, hi)` maps `s` onto a width."""
    opts = _link_options(circos_config)
    if alpha_min is None:
        alpha_min = (circos_config or {}).get('link_alpha_min', 0.15)
    v = pd.Series(values, dtype=float)
    n = len(v)
    if n == 0:
        return v.copy(), v.copy(), v.copy()
    if opts['scale'] == 'rank':
        from scipy.stats import rankdata as _rd
        s = pd.Series((_rd(v.to_numpy()) - 1.0) / (n - 1.0) if n > 1 else np.ones(1), index=v.index)
        constant = n <= 1 or float(v.max()) == float(v.min())
    else:
        v_min, v_max = v.min(), v.max()
        constant = not (v_max > v_min)
        s = (v - v_min) / (v_max - v_min) if not constant else pd.Series(1.0, index=v.index)
    if constant:
        # Every displayed link has the same value (including the common case
        # of a single link) - nothing to contrast against: fully opaque.
        alpha = pd.Series(1.0, index=v.index)
        s = pd.Series(1.0, index=v.index)
    else:
        alpha = alpha_min + s * (1.0 - alpha_min)
    if opts['lw_range'] is not None:
        lo, hi = opts['lw_range']
        lw = lo + s * (hi - lo)
    else:
        lw = pd.Series(float(default_lw), index=v.index)
    return alpha, lw, s


def _link_colour(mode, is_intra, s=1.0):
    """Colour of one link / hyperedge under `link_colour_mode` `mode`.
    'chrom' is the legacy red (intra) / blue (inter) pair."""
    if mode == 'chrom_cvd':
        return _CVD_INTRA_COLOUR if is_intra else _CVD_INTER_COLOUR
    if mode == 'value':
        import matplotlib as _mpl
        return _mpl.colors.to_hex(_mpl.colormaps[_LINK_VALUE_CMAP](0.08 + 0.87 * float(s)))
    return 'red' if is_intra else 'blue'


def _sort_links(df, circos_config):
    """`link_sort='ascending'`: weakest first, so the strongest link is drawn
    last and sits on top (stable sort, index reset). Otherwise `df` as is."""
    if _link_options(circos_config)['sort'] == 'ascending' and df.shape[0] > 1:
        return df.sort_values('value', kind='mergesort').reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Region highlighting: draw-side helpers. The decisions (which links match a
# region, the styling numbers, the config contract) live in circos_highlight.py;
# this block only applies them to pyCirclize. Everything here is inert unless
# `CIRCOS_CONFIG['highlight_regions']` resolves to at least one region on the
# plot being drawn, so a run that never asks for it renders exactly as before.
# ---------------------------------------------------------------------------
_HL_WARNED = set()


def _hl_warn_once(message):
    """Print `message` once per process (every model of every phenotype/population repeats the
    same region problems otherwise)."""
    if message not in _HL_WARNED:
        _HL_WARNED.add(message)
        print(f"[circos_plot] highlight: {message}")


def _resolve_highlight(circos, circos_config, neutral_colours=False):
    """`(resolved_regions, options)` for one Circos object: the configured regions matched to the
    sectors actually on this plot (clamped to them, unknown chromosomes dropped with a one-time
    warning). `resolved_regions == []` means "do not highlight anything".

    `neutral_colours` (the 'combined' higher-order layout, where link colour already encodes k
    with the very palette the regions would otherwise draw from): every region mark is drawn in
    one dark neutral instead, so a band can never be mistaken for a k colour."""
    opts = _hl.highlight_options(circos_config)
    if neutral_colours:
        opts = dict(opts, colour_by='single', colour=_hl.NEUTRAL_MARK)
    for problem in opts['problems']:
        _hl_warn_once(problem)
    if not opts['regions']:
        return [], opts
    ranges = {sec.name: (sec.start, sec.end) for sec in circos.sectors}
    resolved, problems = _hl.resolve_regions(opts['regions'], ranges, opts)
    for problem in problems:
        _hl_warn_once(problem)
    if not resolved:
        _hl_warn_once("none of the requested regions could be placed on this plot - drawing it without highlighting.")
    return resolved, opts


def _highlight_chrom_ranges(chrom_info):
    """`{chromosome name: (start, end)}` from the chromosome info file - the same ranges the plot's
    sectors get, so a region resolves here exactly as it will when drawn. `{}` if unreadable."""
    try:
        chrom = pd.read_csv(chrom_info)
        chrom = unify_columns_by_position(chrom, ['chromosome', 'start', 'end'], 'chromosome info file')
        chrom = chrom.drop_duplicates(subset=['chromosome'])
        return {str(c): (float(a), float(b)) for c, a, b in zip(chrom['chromosome'], chrom['start'], chrom['end'])}
    except Exception:
        return {}


def _highlight_restrict(circos_config, chrom_info, marker_info, RESULT_NAME):
    """`(store_spec, hok_spec)` when `highlight_only_relevant` is on and at least one region can be
    placed, else `(None, None)` (= legacy selection, untouched).

    `store_spec` is `PairSumStore.reduce_groups(restrict=...)`: for each phenotype, a boolean mask
    over the marker dictionary saying which markers lie in a highlighted region. `hok_spec` carries
    the resolved regions for the k >= 3 selection (`_build_higher_order_ring`)."""
    opts = _hl.highlight_options(circos_config)
    if not opts['only_relevant'] or not opts['regions']:
        return None, None
    ranges = _highlight_chrom_ranges(chrom_info)
    resolved, problems = _hl.resolve_regions(opts['regions'], ranges, opts) if ranges else ([], [])
    for problem in problems:
        _hl_warn_once(problem)
    if not resolved:
        _hl_warn_once("'only links involving the highlighted regions' is on but no region could be placed - "
                      "the top interactions are chosen from all links as usual.")
        return None, None
    cache = {}

    def mask_for(phenotype, names):
        key = str(phenotype)
        if key not in cache:
            cache[key] = _marker_region_mask(_load_combined_marker_info(marker_info, RESULT_NAME, phenotype),
                                             names, resolved)
        return cache[key]

    print(f"[circos_plot] highlight: top interactions are chosen among the links that involve "
          f"{', '.join(r['tag'] for r in resolved)} only (match={opts['match']!r}).")
    return ({'mask_for': mask_for, 'match': opts['match']},
            {'regions': resolved, 'match': opts['match']})


def _marker_region_mask(loc_info, names, resolved):
    """Boolean array over `names` (marker names, any dtype): is the marker inside a region?
    Markers absent from `loc_info` are False."""
    names = np.asarray(names, dtype=object)
    chrom = loc_info['chromosome'].astype(str).to_numpy()
    start = np.round(loc_info['start'].astype(float).to_numpy())
    end = np.round(loc_info['end'].astype(float).to_numpy())
    inside = _hl.marker_membership(chrom, start, end, resolved)
    kind = loc_info['name'].dtype.kind
    if kind in 'iuf':
        key = pd.to_numeric(pd.Series(names), errors='coerce').to_numpy()
        known = pd.Series(inside, index=loc_info['name'].to_numpy())
    else:
        key = names.astype(str)
        known = pd.Series(inside, index=loc_info['name'].astype(str).to_numpy())
    known = known[~known.index.duplicated(keep='first')]
    return known.reindex(key).fillna(False).to_numpy(dtype=bool)


def _highlight_legend_regions(resolved):
    """`{region index: {'colour', 'legend'}}`, the form the session legend accumulator stores."""
    return {r['idx']: {'colour': r['colour'], 'legend': r['legend']} for r in resolved}


def _prepare_links_with_highlight(interactions, circos_config, link_lw, regions, hl_opts):
    """The highlight-aware counterpart of the legacy `_sort_links()` + `_link_strength()` pair
    `plot()` runs on one model's k=2 links. Returns `(interactions, lws, strength, style)`:

      interactions  rows in DRAW order (non-highlighted first, so highlighted links sit on top;
                    within each group the usual `link_sort` order is kept), with `_alpha` set to
                    the final opacity. In mode 'only' the non-highlighted rows are gone.
      lws, strength Series aligned to `interactions` (line width; 0-1 strength for colour mode).
      style         {'hit': bool array, 'region': int array (index into `regions`, -1 = none),
                    'n_total': links before 'only' filtering, 'n_hit': highlighted links}.

    Strength (opacity rank) is computed over the links that will be drawn, exactly as in the
    legacy path, so turning highlighting on never changes how strong a link looks relative to its
    neighbours - only whether it is emphasised or faded.
    """
    interactions = _sort_links(interactions, circos_config)
    c = interactions
    hit, region = _hl.classify_pairs(c.iloc[:, 0], c.iloc[:, 1], c.iloc[:, 2],
                                     c.iloc[:, 3], c.iloc[:, 4], c.iloc[:, 5], regions, hl_opts['match'])
    n_total, n_hit = int(hit.shape[0]), int(hit.sum())
    if hl_opts['mode'] == 'only':
        interactions = interactions.loc[hit].reset_index(drop=True)
        region, hit = region[hit], hit[hit]
    style = {'hit': hit, 'region': region, 'n_total': n_total, 'n_hit': n_hit}
    if interactions.shape[0] == 0:
        empty = pd.Series(dtype=float)
        return interactions, empty, empty, style
    alpha, lws, strength = _link_strength(interactions['value'], circos_config, link_lw)
    a_min = circos_config.get('link_alpha_min', 0.15)
    a_out, lw_out = _hl.style_alpha_lw(alpha.to_numpy(), lws.to_numpy(), hit, hl_opts, a_min)
    order = _hl.draw_order(hit)
    interactions = interactions.iloc[order].reset_index(drop=True)
    interactions['_alpha'] = a_out[order]
    style = dict(style, hit=hit[order], region=region[order])
    return (interactions, pd.Series(lw_out[order]),
            strength.iloc[order].reset_index(drop=True), style)


def _highlight_counts_text(style, regions):
    """'12 of 75 links touch the highlighted region(s) (R1=7, R2=5)' for the run log."""
    per = [f"{regions[p]['tag']}={int((style['region'] == p).sum())}" for p in range(len(regions))]
    return (f"{style['n_hit']} of {style['n_total']} links touch the highlighted region(s) "
            f"({', '.join(per)})")


def _write_highlight_links_table(png_path, interactions, style, regions):
    """Sidecar `<plot>_links.tsv` listing the highlighted k=2 links (coordinates, strength, the
    region each belongs to) - the figure shows them, this lets a person read them off. Written
    only when at least one link is highlighted."""
    hit = np.asarray(style['hit'], dtype=bool)
    if not hit.any() or interactions.shape[0] == 0:
        return
    d = interactions.loc[hit]
    reg = [regions[int(p)] for p in np.asarray(style['region'])[hit]]
    out = pd.DataFrame({
        'chromosome_1': d.iloc[:, 0].astype(str).to_numpy(), 'start_1': d.iloc[:, 1].to_numpy(), 'end_1': d.iloc[:, 2].to_numpy(),
        'chromosome_2': d.iloc[:, 3].astype(str).to_numpy(), 'start_2': d.iloc[:, 4].to_numpy(), 'end_2': d.iloc[:, 5].to_numpy(),
        'value': d['value'].to_numpy(),
        'link_type': np.where(d.iloc[:, 0].astype(str).to_numpy() == d.iloc[:, 3].astype(str).to_numpy(),
                              'within_chromosome', 'between_chromosomes'),
        'region': [r['tag'] for r in reg], 'region_description': [r['desc'] for r in reg],
    })
    _atomic_to_csv(out.sort_values('value', ascending=False, kind='mergesort'),
                   re.sub(r'\.png$', '', png_path) + '_links.tsv', sep='\t', index=False)


def _draw_highlight_regions(circos, regions, opts, circos_config):
    """Mark every resolved region on the plot (loop-invariant, so called once before the base
    figure is frozen): a coloured BAND just inside the innermost ring - exactly where the links
    land - a faint WEDGE of the same colour through the link area so the eye can follow a region's
    links to its end, and a short radial TAG. A region narrower than `MIN_BAND_DEG` is widened on
    the page (never in the data) so a 50 kb window on a 100 Mb chromosome is still visible."""
    if not regions or not (opts['band'] or opts['shade'] or opts['label']):
        return
    import matplotlib.patheffects as _pe
    sectors = {sec.name: sec for sec in circos.sectors}
    r_in = min(sec.get_lowest_r() for sec in circos.sectors)
    band_w = 3.0
    shade_depth = 18.0
    r_band_lo = max(r_in - band_w, 1.0)
    figsize = float(circos_config.get('figsize') or 8.0)
    ppu = circos_geometry.points_per_r_unit(figsize_in=figsize)
    tags = []
    for reg in regions:
        sec = sectors.get(reg['chrom'])
        if sec is None:
            continue
        s, e = float(reg['start']), float(reg['end'])
        span = max(float(sec.end) - float(sec.start), 1e-9)
        deg_per_x = math.degrees(sec.x_to_rad(sec.end) - sec.x_to_rad(sec.start)) / span
        min_x = _hl.MIN_BAND_DEG / max(deg_per_x, 1e-12)
        if e - s < min_x:
            mid = (s + e) / 2.0
            s, e = max(float(sec.start), mid - min_x / 2.0), min(float(sec.end), mid + min_x / 2.0)
        if opts['shade'] and r_in > 2.0:
            # a shallow tinted strip (not a wedge to the centre: a whole-chromosome region would
            # otherwise tint a large part of the disc) under the links
            sec.rect(s, e, r_lim=(max(r_in - shade_depth, 0.5), r_in), fc=reg['colour'], ec='none', lw=0,
                     alpha=0.14, zorder=0.5)
        if opts['band']:
            sec.rect(s, e, r_lim=(r_band_lo, r_in), fc=reg['colour'], ec='black', lw=0.4)
        tags.append((reg['tag'], math.degrees(sec.x_to_rad((s + e) / 2.0, ignore_range_error=True))))
    if not (opts['label'] and tags):
        return
    size = max(5.0, min(8.0, float(circos_config.get('label_size', 6)) * 1.3))
    r_anchor = r_band_lo - 1.5                 # the tag's outer end; it reads inward from here
    room = max(r_anchor - 3.0, 4.0)
    placed = []
    for text, deg in tags:
        s_pt = size
        length = circos_geometry.text_width_points(text, s_pt) / ppu
        if length > room:
            s_pt = max(4.5, s_pt * room / length)
        placed.append((text, deg, s_pt))
    sep = max(2.0, math.degrees(1.5 * size / ppu / max(r_anchor - 5.0, 5.0)))
    lo = min(math.degrees(sec.rad_lim[0]) for sec in circos.sectors)
    hi = max(math.degrees(sec.rad_lim[1]) for sec in circos.sectors)
    angles = circos_geometry.spread_labels([p[1] for p in placed], sep, lo=lo, hi=hi)
    halo = [_pe.withStroke(linewidth=2.2, foreground='white')]
    for (text, _deg, s_pt), ang in zip(placed, angles):
        # Radial text anchored at its OUTER end and reading toward the centre, upright on both
        # halves of the circle (what pyCirclize's adjust_rotation does for an outside label,
        # mirrored: its 'outer' style would run the text outward, over the band).
        d = ang % 360.0
        right = d <= 180.0
        circos.text(text, r=r_anchor, deg=ang, size=s_pt, color='black', fontweight='bold',
                    rotation=(90.0 - d) if right else (270.0 - d), rotation_mode='anchor',
                    ha='right' if right else 'left', va='center_baseline', path_effects=halo)


# Logged once per process, not once per hyperedge - matches this codebase's
# own established convention for a would-be-noisy per-call warning (e.g.
# circos_geometry.text_width_points()'s own TextPath-fallback flag).
_X_TO_RAD_UNAVAILABLE_LOGGED = False


def _sector_rad_for_position(circos, chromosome, position):
    """The angle (radians, in the whole-circle coordinate system
    pycirclize's own polar axes use) of one genomic `position` within
    `chromosome`'s own sector of `circos` - the single conversion every
    hub-and-spoke glyph member needs (blueprint: "hub angle = circular
    mean of member angles").

    Uses pycirclize's own public `Sector.x_to_rad(x)` (`circos.get_sector
    (name)` returns that chromosome's `Sector` instance) - `name` here is
    the SAME sector-name string this module already hands `circos.link()`
    directly as a region's first element (see `_build_interaction_ring()`
    's own 'chromosome_marker1'/'chromosome_marker2' columns), and `x` is
    the SAME BED start/end coordinate space those region tuples already
    use - i.e. this is the documented, public conversion pycirclize
    itself relies on internally for exactly this purpose.

    VERIFIED (Update ID ver4-25, R12): unlike the shapiq/R-side code
    elsewhere in this update - written and syntax-checked only, never
    executed, for lack of an install in that sandbox - this specific
    conversion WAS exercised end-to-end against a real, pinned
    `pycirclize==1.10.1` install: `Circos.get_sector(name).x_to_rad(x,
    ignore_range_error=True)` confirmed to exist with exactly this
    signature; `fig.axes[0]` confirmed to be a genuine
    `matplotlib.projections.polar.PolarAxes` whose data coordinates are
    literally `(theta_radians, r)`; and a full hub-and-spoke glyph -
    three spokes converging on one hub via `circos_geometry.
    hyperedge_hub()`/`bezier_spoke_path()`, drawn with
    `ax.add_patch(PathPatch(...))` exactly as `_draw_higher_order_
    glyphs()` below does - rendered and saved to a real PNG with the
    expected shape (hub pulled toward centre for a spread-out cluster,
    smooth curved spokes, no exception). This confirms the blueprint's
    own "V12.1" pre-check claim directly rather than merely trusting it.
    Still never fatal on its own regardless: any exception here (e.g. a
    genuinely unresolvable chromosome name for THIS specific run's own
    BED sectors) is caught by every caller (`_draw_higher_order_glyphs()`
    below) and turns into "skip drawing this one hyperedge's glyph",
    never an aborted render - every existing ring/chord on the same plot
    is completely unaffected.
    """
    global _X_TO_RAD_UNAVAILABLE_LOGGED
    try:
        # ignore_range_error=True: a marker's own midpoint((start,end))
        # position should always fall inside its chromosome's own sector
        # range, but a rounding-edge position sitting a fraction outside
        # (float midpoint vs. int BED bounds) would otherwise raise
        # ValueError and drop an otherwise-perfectly-good hyperedge for no
        # real reason - pycirclize's own `x_to_rad()` docstring offers
        # exactly this escape hatch for that case (verified directly
        # against the pinned pycirclize==1.10.1 source: `Sector.x_to_rad
        # (self, x, ignore_range_error=False)`, confirmed via
        # `circos.get_sector(name)` + `Sector.x_to_rad()` both existing
        # with this exact signature - the one part of this module that
        # WAS verified end-to-end against a real install, unlike the
        # DISCLOSURE note below for the rest of this function's own
        # reasoning trail).
        return circos.get_sector(str(chromosome)).x_to_rad(float(position), ignore_range_error=True)
    except Exception as exc:
        if not _X_TO_RAD_UNAVAILABLE_LOGGED:
            print(f"[circos_plot] NOTE: could not resolve a genomic position to a plot angle "
                  f"via pycirclize's Sector.x_to_rad() ({exc!r}) - k>=3 interaction hub-and-spoke "
                  f"glyphs will be skipped for the rest of this run (every other ring/chord on "
                  f"the plot is unaffected).")
            _X_TO_RAD_UNAVAILABLE_LOGGED = True
        raise


def _draw_higher_order_glyphs(ax, circos, hok_selected, r_link, *, colourise, alpha_min=0.15,
                               marker_size_min=6.0, marker_size_max=18.0, circos_config=None,
                               hl=None, hl_state=None):
    """Draws every hyperedge in `hok_selected` (columns: 'members' - a
    list of `(chromosome, position)` tuples, already resolved by
    `_build_higher_order_ring()` - and 'value', already normalised to sum
    to 1 within this one (model, order) render) as one hub-and-spoke
    glyph each, directly onto `ax` (`fig.axes[0]`, the polar axes
    `circos.plotfig()` just produced for THIS render - see
    `_sector_rad_for_position()`'s own docstring for why its data
    coordinates are `(theta, r)` in the same r-unit scale
    `circos_geometry`'s formulas already use).

    `colourise(is_intra: bool) -> str` picks each hyperedge's own patch
    colour - the caller decides the scheme: `plot()`'s 'per_order' call
    site reuses the EXISTING red/blue intra-/inter-chromosome convention
    every k=2 chord already uses; its 'combined' call site instead colours
    by k (`_okabe_ito_colour_for_order()`) and lets LINESTYLE (solid/
    dashed) carry the intra-/inter-chromosome distinction here.

    A hyperedge is "intra" iff EVERY member resolved to the SAME
    chromosome (blueprint: "existing intra-chromosome (all members same
    chromosome) / inter-chromosome palette") - any two members on
    different chromosomes makes the whole hyperedge "inter", exactly
    mirroring the k=2 chord loop's own `chrom1 != chrom2` test.

    Returns `(has_intra, has_inter)` - whether ANY drawn hyperedge was
    intra-/inter-chromosome - for the caller to fold into the shared
    legend accumulator exactly like the k=2 loop's own
    has_intra_chr_link_for_legend/has_inter_chr_link_for_legend.

    A hyperedge whose own `_sector_rad_for_position()` call fails is
    silently skipped (not drawn, does not affect `has_intra`/`has_inter`,
    does not abort the loop) - one bad/unresolvable member never prevents
    the other, otherwise-drawable, hyperedges on the same ring from being
    drawn.

    Region highlighting (optional): `hl` is `{'regions': resolved regions, 'opts': highlight
    options, 'recolour': bool}` (None = off, the legacy behaviour). A hyperedge is highlighted
    when a member (or, with `match='both'`, every member) lies in a region. Highlighted
    hyperedges are drawn last, thicker and more opaque, in their region's colour when `recolour`
    is true (the 'per_order' layout; the 'combined' layout colours by k, so it keeps that colour
    and only emphasises); the others are faded (grey when `recolour` and `dim == 'grey'`). In
    mode 'only' the others are not drawn. `hl_state`, when given, is filled with `n_total`,
    `n_hit`, `hit_intra`, `hit_inter`, `dimmed` and `region` (per hyperedge) for the caller's
    log/legend; the returned `(has_intra, has_inter)` then covers only hyperedges drawn in their
    ordinary colours."""
    has_intra = False
    has_inter = False
    if hl_state is not None:
        hl_state.update(n_total=0, n_hit=0, hit_intra=False, hit_inter=False, dimmed=False, region=[])
    if hok_selected is None or hok_selected.shape[0] == 0:
        return has_intra, has_inter

    # ver4-31 R11/R12: the same `link_sort` / `link_scale` / `link_lw_range` /
    # `link_colour_mode` settings as the k=2 chords (no `circos_config` = the
    # historical min-max alpha, fixed 1.2 spoke width, file order).
    _lcfg = dict(circos_config or {})
    _lcfg['link_alpha_min'] = alpha_min      # glyphs always used the function default, not CIRCOS_CONFIG's
    hok_selected = _sort_links(hok_selected.reset_index(drop=True), _lcfg).reset_index(drop=True)
    _hl_on = bool(hl and hl.get('regions'))
    _hl_hit = _hl_region = None
    if _hl_on:
        _hl_hit, _hl_region = _hl.classify_members(list(hok_selected['members']), hl['regions'], hl['opts']['match'])
        if hl_state is not None:
            hl_state.update(n_total=int(_hl_hit.shape[0]), n_hit=int(_hl_hit.sum()))
        if hl['opts']['mode'] == 'only':
            hok_selected = hok_selected.loc[_hl_hit].reset_index(drop=True)
            _hl_region, _hl_hit = _hl_region[_hl_hit], _hl_hit[_hl_hit]
            if hok_selected.shape[0] == 0:
                return has_intra, has_inter
    alphas, spoke_lws, strengths = _link_strength(hok_selected['value'], _lcfg, 1.2)
    if _hl_on:
        _a, _lw = _hl.style_alpha_lw(alphas.to_numpy(), spoke_lws.to_numpy(), _hl_hit, hl['opts'], alpha_min)
        _order = _hl.draw_order(_hl_hit)               # highlighted hyperedges last = on top
        hok_selected = hok_selected.iloc[_order].reset_index(drop=True)
        alphas, spoke_lws = pd.Series(_a[_order]), pd.Series(_lw[_order])
        strengths = strengths.iloc[_order].reset_index(drop=True)
        _hl_hit, _hl_region = _hl_hit[_order], _hl_region[_order]
        if hl_state is not None:
            hl_state['region'] = [int(r) for r in _hl_region]
    _mode = _link_options(_lcfg)['colour_mode']
    ranks = hok_selected['value'].rank(ascending=False, method='min')
    n_hyperedges = hok_selected.shape[0]

    for idx, row in hok_selected.iterrows():
        members = row['members']
        try:
            rads = [_sector_rad_for_position(circos, chrom, pos) for chrom, pos in members]
        except Exception:
            continue  # already logged once by _sector_rad_for_position()

        is_intra = len({chrom for chrom, _ in members}) == 1
        _is_hit = bool(_hl_on and _hl_hit[idx])
        _recoloured = False
        if _hl_on and hl.get('recolour') and _is_hit:
            colour = hl['regions'][int(_hl_region[idx])]['colour']
            _recoloured = True
            if hl_state is not None:
                hl_state['hit_intra' if is_intra else 'hit_inter'] = True
        elif _hl_on and hl.get('recolour') and hl['opts']['dim'] == 'grey':
            colour = _hl.DIM_GREY
            _recoloured = True
            if hl_state is not None:
                hl_state['dimmed'] = True
        else:
            colour = _link_colour(_mode, is_intra, strengths.iloc[idx]) if _mode != 'chrom' else colourise(is_intra)
        if not _recoloured:
            if is_intra:
                has_intra = True
            else:
                has_inter = True
        linestyle = 'solid' if is_intra else 'dashed'
        if _recoloured and not _is_hit:
            linestyle = 'solid'
        alpha = float(alphas.iloc[idx])

        hub_rad, r_hub = circos_geometry.hyperedge_hub(rads, r_link)
        for member_rad in rads:
            path = circos_geometry.bezier_spoke_path(member_rad, r_link, hub_rad, r_hub)
            ax.add_patch(mpatches.PathPatch(
                path, facecolor='none', edgecolor=colour, linestyle=linestyle,
                linewidth=float(spoke_lws.iloc[idx]), alpha=alpha,
            ))

        # Hub marker size ∝ rank (blueprint) - the strongest hyperedge
        # (rank 1) on THIS render gets the largest hub marker, tapering
        # down to `marker_size_min` for the weakest; a single surviving
        # hyperedge always gets the maximum size (nothing to rank it
        # against).
        rank = float(ranks.iloc[idx])
        size = marker_size_max if n_hyperedges <= 1 else (
            marker_size_max - (rank - 1.0) / (n_hyperedges - 1.0) * (marker_size_max - marker_size_min)
        )
        if _hl_on and not _is_hit:
            size = max(marker_size_min * 0.7, size * 0.7)     # faded hubs stay out of the way
        ax.plot([hub_rad], [r_hub], marker='o', markersize=size, color=colour, alpha=alpha,
                markeredgecolor='black', markeredgewidth=0.3, linestyle='none')

    return has_intra, has_inter


def _precompute_higher_order_groups(df, *, scope_model_order_by_population):
    """The k>=3 analog of `_precompute_interaction_groups()` above -
    identical rationale (avoid re-deriving a per-(population, model,
    phenotype, order) slice of the FULL `interactions_hok` table from
    scratch on every one of `circos_plot()`'s `len(PHENOTYPE) *
    len(POPULATION)` calls) and identical dual-shape handling (the raw
    one-row-per-observation shape `ResultSet.interactions_hok()`/
    Sequential's own in-memory accumulator produce, OR the pre-aggregated
    'value_sum'/'value_count' shape `batch_reader.
    aggregate_marker_set_sums()`/`ResultSet.interactions_hok_grouped()`
    produce - `_ensure_marker_pair_sum_count()` already normalises either
    shape identically, reused here unchanged), just keyed by `order` +
    the ';'-joined 'markers' string in place of `marker1`/`marker2`.

    Unlike `_precompute_interaction_groups()`, no 'factor'-sentinel-row
    filter is applied - see `batch_reader.aggregate_marker_set_sums()`'s
    own docstring for why that filter has no analog here (nothing in the
    k>=3 accumulation path ever emits one).

    Returns
    -------
    (groups, model_order) :
        groups : dict {(population_key, model_name, phenotype_value,
            order): DataFrame[markers, value]} - 'population_key' is
            either 'all' (the cross-population aggregate) or the
            population's own string label, exactly as
            `_precompute_interaction_groups()`'s own groups dict.
        model_order : dict {population_key: [model_name, ...]} - the
            first-appearance order every model with ANY k>=3 row
            (regardless of which order) appears in, for THIS
            population_key - `higher_order_interaction()` below iterates
            this list and tries every requested order per model, so a
            model missing a SPECIFIC order simply contributes nothing for
            that order (see `_build_higher_order_ring()`'s own
            None-return "nothing to draw" case), rather than needing a
            second, order-specific membership index here.
    """
    if df is None or df.shape[0] == 0:
        return {}, {}

    _value_cols = [c for c in ('value', 'value_sum', 'value_count') if c in df.columns]
    has_population = 'population' in df.columns

    keep_cols = (['population'] if has_population else []) + ['model', 'phenotype', 'order', 'markers'] + _value_cols
    work = df.loc[:, keep_cols]
    work = _ensure_marker_pair_sum_count(work)
    if has_population:
        work = work.copy()
        work['population'] = work['population'].astype(str)

    groups = {}

    _all_reduced = work.groupby(['model', 'phenotype', 'order', 'markers'], as_index=False)[['value_sum', 'value_count']].sum()
    _all_reduced['value'] = _all_reduced['value_sum'] / _all_reduced['value_count']
    for (model_name, phenotype_value, order_value), sub in _all_reduced.groupby(['model', 'phenotype', 'order'], sort=False):
        groups[('all', model_name, phenotype_value, order_value)] = sub[['markers', 'value']]
    model_order = {'all': work['model'].drop_duplicates().tolist()}

    if has_population:
        _pop_reduced = work.groupby(['population', 'model', 'phenotype', 'order', 'markers'], as_index=False)[['value_sum', 'value_count']].sum()
        _pop_reduced['value'] = _pop_reduced['value_sum'] / _pop_reduced['value_count']
        for (population_value, model_name, phenotype_value, order_value), sub in _pop_reduced.groupby(['population', 'model', 'phenotype', 'order'], sort=False):
            groups[(population_value, model_name, phenotype_value, order_value)] = sub[['markers', 'value']]
        if scope_model_order_by_population:
            for population_value, sub in work.groupby('population', sort=False):
                model_order[population_value] = sub['model'].drop_duplicates().tolist()

    return groups, model_order


def _build_higher_order_ring(selected, circos_config, marker_info, PHENOTYPE, RESULT_NAME, model_name, order):
    """The k>=3 analog of `_build_interaction_ring()` above: per-order top
    selection (via the SAME `_select_top_interactions()` every k=2 ring
    already uses - `selected` is already scoped to one `order` by
    `_precompute_higher_order_groups()`'s own per-key grouping, so this
    naturally implements the blueprint's "per-order top selection" with
    no new selection function needed), sum-to-1 normalisation (within
    this one (model, order), matching `_build_interaction_ring()`'s own
    per-(model) normalisation), and marker-name -> (chromosome, position)
    resolution for EVERY member of each surviving hyperedge (the
    variable-length generalisation of `_build_interaction_ring()`'s own
    fixed marker1/marker2 resolution).

    Returns `None` when there is nothing to draw (no data for this key,
    `_select_top_interactions()` narrowed it to zero rows, or every
    surviving hyperedge had an unresolvable member) - callers `continue`
    on that, mirroring `_build_interaction_ring()`'s own contract
    exactly.

    A hyperedge with an unresolvable member (a marker name absent from
    `marker_info`/the gene-coordinate side files) is dropped WHOLE - the
    same "no resolvable coordinate -> silently drop this row" behaviour
    `_build_interaction_ring()`'s own `pd.merge(..., 'inner', ...)`
    already gives a k=2 pair, applied here at the whole-hyperedge
    granularity, since a glyph with fewer than 2 resolvable members has
    nothing left worth drawing."""
    if selected is None or selected.shape[0] == 0:
        return None
    _restrict = (circos_config or {}).get('_hl_restrict')
    if _restrict:
        # 'only links involving the highlighted regions': drop the hyperedges with no member in a
        # region BEFORE the top-N% cut, so the cut is taken over the relevant ones.
        _loc = _load_combined_marker_info(marker_info, RESULT_NAME, PHENOTYPE)
        _all_names = pd.unique(np.concatenate([str(m).split(';') for m in selected['markers']])) if len(selected) else []
        _in = dict(zip(_all_names, _marker_region_mask(_loc, np.asarray(_all_names, dtype=object), _restrict['regions'])))
        _flags = [[_in.get(n, False) for n in str(m).split(';') if n] for m in selected['markers']]
        _agg = all if _restrict['match'] == 'both' else any
        selected = selected.loc[[bool(f) and _agg(f) for f in _flags]]
        if selected.shape[0] == 0:
            return None
    selected = _select_top_interactions(selected, circos_config)
    if selected.shape[0] == 0:
        return None
    selected = selected.copy()
    selected['value'] = selected['value'] / selected['value'].sum()

    loc_info = _load_combined_marker_info(marker_info, RESULT_NAME, PHENOTYPE)
    loc_lookup = loc_info.drop_duplicates(subset=['name'], keep='first').set_index('name')

    rows = []
    for _, row in selected.iterrows():
        member_names = [m for m in str(row['markers']).split(';') if m]
        members = []
        unresolved = False
        for name in member_names:
            if name not in loc_lookup.index:
                unresolved = True
                break
            loc = loc_lookup.loc[name]
            chromosome = str(loc['chromosome'])
            position = int(round((float(loc['start']) + float(loc['end'])) / 2.0))
            members.append((chromosome, position))
        if unresolved or len(members) < 2:
            continue
        rows.append({'model': model_name, 'order': order, 'members': members, 'value': float(row['value'])})

    if not rows:
        return None
    return pd.DataFrame(rows)


def higher_order_interaction(higher_order_groups, higher_order_model_order, marker_info, PHENOTYPE,
                              circos_config, POPULATION, RESULT_NAME, interaction_orders):
    """The k>=3 analog of `interaction()` below - returns a small
    DataFrame, one row per SURVIVING (top-selected, fully-resolved)
    hyperedge, with columns `['model', 'order', 'members', 'value']`
    ('members': a list of `(chromosome, position)` tuples) - completely
    SEPARATE from `interaction()`'s own return value, never merged into
    it, so the existing k=2 chord path stays exactly what it always was
    (AC12.3: `CIRCOS_INTERACTION_ORDERS=[2]` renders byte-identically to
    a run that never called this function at all, since `interaction_
    orders` would then contain no value >= 3 and this returns an empty,
    correctly-columned frame).

    Only orders `>= 3` in `interaction_orders` are ever resolved here -
    `2` is silently skipped (that stays `interaction()`'s own, unchanged
    job)."""
    total = []
    models = higher_order_model_order.get(POPULATION, [])
    for model_name in models:
        for order in interaction_orders:
            try:
                order = int(order)
            except (TypeError, ValueError):
                continue
            if order < 3:
                continue
            selected = higher_order_groups.get((POPULATION, model_name, PHENOTYPE, order))
            ring = _build_higher_order_ring(selected, circos_config, marker_info, PHENOTYPE, RESULT_NAME, model_name, order)
            if ring is None:
                continue
            total.append(ring)
    if not total:
        return pd.DataFrame(columns=['model', 'order', 'members', 'value'])
    return pd.concat(total, ignore_index=True)


def _build_interaction_ring(selected, circos_config, marker_info, PHENOTYPE, RESULT_NAME, model_name):
    """The shared tail end of both branches below, factored out
    unchanged (byte-for-byte the same operations, same order) so the
    performance fix's new dict-lookup front end doesn't have to keep two
    copies of it in sync: top-N/top-% selection, sum-to-1 normalisation,
    marker-name -> chromosome/position resolution via `loc_info`, and the
    final `[chromosome_marker1, start, end, chromosome_marker2, start,
    end, value]` shape `plot()` expects, tagged with `model_name`.

    `selected` is already the small, single-(population, model,
    phenotype) slice `_precompute_interaction_groups()` produced -
    exactly the `interaction_model[interaction_model['phenotype'] ==
    PHENOTYPE]` / post-`_regroup_and_finalize_mean()` frame the pre-fix
    code built inline at this point, just arrived at without re-scanning
    the full table to get here. Returns `None` when there is nothing to
    draw (no data for this key, or `_select_top_interactions()` narrowed
    it to zero rows) - callers `continue` on that, matching the pre-fix
    code's own early-`continue` checks.
    """
    if selected is None or selected.shape[0] == 0:
        return None
    # Update ID circos-oom-fix: `_precompute_interaction_groups(circos_config=...)`
    # already applied `_select_top_interactions()`'s exact rule per key (on
    # the numeric arrays, before any strings were built) and flagged the
    # frame - selecting AGAIN would apply "top N% of the already-selected
    # rows". Legacy (unflagged) frames still select here, unchanged.
    if not selected.attrs.get('top_selected'):
        selected = _select_top_interactions(selected, circos_config)
        if selected.shape[0] == 0:
            return None
    selected = selected.copy()
    selected['value'] = selected['value'] / selected['value'].sum()

    loc_info = _load_combined_marker_info(marker_info, RESULT_NAME, PHENOTYPE)
    # marker names are decoded as the strings they were streamed in as; if
    # marker_info.csv's own name column was type-inferred as numeric (all-
    # numeric SNP ids) align the merge key instead of failing on object vs int.
    _name_kind = loc_info['name'].dtype.kind
    _m1, _m2 = selected['marker1'], selected['marker2']
    if _name_kind in 'iuf':
        _m1, _m2 = pd.to_numeric(_m1, errors='coerce'), pd.to_numeric(_m2, errors='coerce')
    elif _name_kind in 'OUS' or str(loc_info['name'].dtype).startswith('str'):
        _m1, _m2 = _m1.astype(str), _m2.astype(str)
    start = pd.merge(_m1, loc_info, 'inner', left_on='marker1', right_on='name')
    end = pd.merge(_m2, loc_info, 'inner', left_on='marker2', right_on='name')

    # ver4-4 R7.a - see the vectorisation note in data_conversion()
    # above; identical int(round(...)) row-loop pattern, replaced the
    # same way.
    start['start'] = start['start'].round().astype('int64')
    start['end'] = start['end'].round().astype('int64')
    end['start'] = end['start'].round().astype('int64')
    end['end'] = end['end'].round().astype('int64')

    chrom_start = start['chromosome'].astype(str)
    chrom_end = end['chromosome'].astype(str)

    selected = pd.concat([chrom_start, start.loc[:, ['start', 'end']],
                           chrom_end, end.loc[:, ['start', 'end']],
                           selected['value']], axis=1)
    selected.columns = ['chromosome_marker1', 'start', 'end', 'chromosome_marker2', 'start', 'end', 'value']
    selected['model'] = model_name
    return selected


def interaction(interaction_groups, interaction_model_order, marker_info, PHENOTYPE, circos_config, POPULATION, RESULT_NAME, attention_groups, attention_model_order):
    """Requirement 2 (Update ID ver4-5): `interaction` can now hold rows
    from MORE THAN ONE model (model_registry.emits_interactions() decides
    which models are allowed to write to Interaction.csv - previously
    RF-only). This function mirrors the attention branch immediately
    below it, which already did the right thing for multiple GAT
    variants: one independent ring per model, never averaged together.

    Previously this branch dropped the 'model' column BEFORE grouping by
    (phenotype, marker1, marker2) and relabelled the combined result 'RF'
    unconditionally - harmless with exactly one emitter selected (the
    only configuration that existed before this update), but silently
    AVERAGED two different models' values for the same pair into one ring
    whenever a second interaction-emitting model was selected, and
    mislabelled the result as RF's regardless. An RF-only run takes
    exactly the same code path as before (one iteration of the loop
    below, with model_selected==['RF']) and produces byte-identical
    output - see EasiGP_ver4-5_Change_Summary.md §10 for the fixture this
    was checked against.

    Requirement (performance fix, Update ID ver4-10): this function used
    to take the FULL `interactions`/`attention_original` tables and
    re-derive its own per-model slice from scratch on every call -
    `circos_plot()` calls it once per (PHENOTYPE, POPULATION) pair, so
    that full-table work happened `len(PHENOTYPE) * len(POPULATION)`
    times over. With more than one interaction-emitting model selected
    (RKHS/RF/SVR/KNN, etc. - see `model_registry.py`), that full table is
    several times larger than the RF-only case this was originally
    written for, which is what made circos-plot generation "take
    forever". `circos_plot()` now calls `_precompute_interaction_groups()`
    ONCE, before its (PHENOTYPE, POPULATION) loop, for `interactions` and
    for `attention` independently, and hands this function the resulting
    small lookup dicts instead of the raw tables - see that function's
    own docstring for the full rationale and for why this applies
    identically regardless of whether the run was Sequential, an
    in-process GUI "Option B: run now", or a Parallel Step 2 assemble
    (raw vs pre-aggregated 'value_sum'/'value_count' shape - both were,
    and still are, handled transparently). This function's own
    observable output - which rings get drawn, with which values - is
    unchanged; only how expensively it gets there.

    `interaction_model_order`/`attention_model_order` reproduce the
    pre-fix code's own per-branch model-iteration order exactly (see
    `_precompute_interaction_groups()`'s `scope_model_order_by_population`
    parameter for the historical inconsistency between the two branches
    that this preserves rather than "fixes").
    """
    interaction_selected_total = pd.DataFrame()

    models_interaction = interaction_model_order.get(POPULATION, [])
    for model_name in models_interaction:
        selected = interaction_groups.get((POPULATION, model_name, PHENOTYPE))
        ring = _build_interaction_ring(selected, circos_config, marker_info, PHENOTYPE, RESULT_NAME, model_name)
        if ring is None:
            continue
        interaction_selected_total = pd.concat([interaction_selected_total, ring])

    # The GAT-attention branch's model order has never been scoped by
    # POPULATION (see `_precompute_interaction_groups()`'s docstring) -
    # always the single 'all' key, regardless of which POPULATION this
    # call is for.
    models_GAT = attention_model_order.get('all', [])
    for model_name in models_GAT:
        selected = attention_groups.get((POPULATION, model_name, PHENOTYPE))
        ring = _build_interaction_ring(selected, circos_config, marker_info, PHENOTYPE, RESULT_NAME, model_name)
        if ring is None:
            continue
        interaction_selected_total = pd.concat([interaction_selected_total, ring])

    return interaction_selected_total

def _circos_axis_unit_label(circos_config):
    """Requirement 2: a human-readable label for what the tick numbers
    around the plot actually mean - shown once, at the centre of the
    plot (see plot()'s own circos.text(...,r=0,...) call), since
    xticks_by_interval()'s own label_formatter only ever prints the raw
    (scaled) number with no unit attached at all.

    Requirement (THIRD correction): the previous version showed
    circos_config['scale'] itself, exactly - technically accurate (a
    tick step of '1' really is 'scale' raw units), but confirmed
    directly against a real report to look wrong in practice: for
    scale=200 (chosen by the auto-suggestion formula's own 1/2/5 x 10^n
    'nice number' rule - 200 = 2 x 10^2), the reported IDEAL label was
    '100 cM', not '200 cM'; separately, a bp scale of 5,000,000 (5 x
    10^6) labelled as '5 Mb' was confirmed to look 'unnatural' - the
    ideal was apparently just 'Mb' (10^6), with no multiplier at all.

    Both examples agree on the same fix, which is also what was
    suggested directly: describe the axis using only the LARGEST POWER
    OF TEN that 'scale' doesn't exceed - i.e. keep only scale's own
    trailing zeros, drop its leading significant digit(s) (1, 2, or 5,
    from the nice-number rule). 200 -> 10^2 = 100. 5,000,000 -> 10^6 =
    1,000,000, which is EXACTLY 1 Mb, so no multiplier is needed at
    all - matching both reports at once. This describes the axis at a
    clean, round order of magnitude a reader can immediately relate to
    (matching how bp/kb/Mb/Gb are conventionally used as approximate
    magnitude indicators in genomics generally, not as an exact
    per-tick distance), rather than the exact, possibly-2x-or-5x 'nice
    number' scale value used for the real tick spacing underneath it -
    which is unchanged by this and still exactly what
    xticks_by_interval()/its label_formatter actually use for placing
    and numbering ticks; only how it's DESCRIBED in this one caption
    changes."""
    unit = circos_config.get('unit', 'bp')
    scale = circos_config.get('scale', 1) or 1
    power_of_ten = 10 ** math.floor(math.log10(scale)) if scale > 0 else 1

    if unit == 'cM':
        if power_of_ten == 1:
            return 'Position (cM)'
        return f'Position ({power_of_ten:g} cM)'

    if power_of_ten >= 1_000_000_000:
        unit_name, unit_size = 'Gb', 1_000_000_000
    elif power_of_ten >= 1_000_000:
        unit_name, unit_size = 'Mb', 1_000_000
    elif power_of_ten >= 1_000:
        unit_name, unit_size = 'kb', 1_000
    else:
        unit_name, unit_size = 'bp', 1
    multiplier = power_of_ten / unit_size
    if multiplier == 1:
        return f'Position ({unit_name})'
    return f'Position ({multiplier:g} {unit_name})'


class _GradientSwatchHandler(mlegend_handler.HandlerBase):
    """Requirement 1 (correction, matching the attached 'example.png'
    reference format): draws a smooth light-to-dark colour gradient
    rectangle for a legend entry, instead of matplotlib's usual single
    flat-colour swatch - the marker-effect quantile scale is a
    continuous range, not two discrete levels, so the legend should look
    like one now, exactly like the reference image's own 'Genomic marker
    effect' gradient bars."""
    def __init__(self, low_hex, high_hex, **kwargs):
        self.low_hex = low_hex
        self.high_hex = high_hex
        super().__init__(**kwargs)

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        n = 24
        low_rgb = np.array(mcolors.to_rgb(self.low_hex))
        high_rgb = np.array(mcolors.to_rgb(self.high_hex))
        patches = []
        for i in range(n):
            frac = i / (n - 1)
            color = tuple(low_rgb + (high_rgb - low_rgb) * frac)
            rect = mpatches.Rectangle(
                (xdescent + frac * width, ydescent), width / n * 1.08, height,
                facecolor=color, edgecolor='none', transform=trans,
            )
            patches.append(rect)
        return patches


def _build_circos_legend_handles(colours_used_by_hue, CYTOBAND_COLORMAP, gene_colours_used, has_inter_chr_link, has_intra_chr_link, truncated_labels=None, has_hub_glyph=False, combined_orders_used=None, link_colour_mode='chrom',
                                 highlight_regions=None, highlight_intra=False, highlight_inter=False, highlight_dimmed=False):
    """Requirement 13 (and its Requirement 1 correction): build the
    legend entries for the current plot - ONLY for colours/categories
    that actually appear on THIS specific rendering, never every colour
    the underlying scheme could theoretically produce.

    Returns (handles, handler_map) - `handler_map` pairs each gradient-bar
    handle with a _GradientSwatchHandler instance (see plot()'s own
    fig.legend(..., handler_map=handler_map) call, which is REQUIRED for
    the gradient rendering to actually happen - a plain fig.legend(handles)
    call would fall back to treating each handle as an ordinary flat-colour
    patch instead).

    `colours_used_by_hue` is a {hue: {colour strings actually found in
    that hue's marker_effect_*.tsv file(s), e.g. 'blue3'}} mapping - built
    by plot() itself from the SAME tsv file(s) it just rendered, so this
    reflects the real, current data rather than an assumption. For each
    hue actually present, this shows a GRADIENT bar spanning the LOWEST
    to HIGHEST quantile level ACTUALLY FOUND (not hardcoded 0-9) -
    labelled 'Marker effect (single model)' for blue and 'Marker effect
    (ensemble model)' for red, matching what these two hues actually
    represent (see quantile_conversion()'s own colour = 'red' if MODEL in
    [ensemble/meta-model combinations] else 'blue') - NOT a meaningless
    'low effect'/'high effect' per-swatch label, which is what this
    looked like before this correction.

    `gene_colours_used` is the set of DISTINCT 'colour' (pathway/category)
    values actually found in the gene tsv file(s) rendered this plot -
    NOT the 'source' values (e.g. 'leaf'/'SAM'/'QTL'/'wisser_et_al'),
    which are just RING LABELS identifying which data source a gene
    annotation came from, never colour-coded at all (gene_info.csv's
    'colour' column - a pathway/category name like 'photoperiod' - is
    what actually determines the fill colour, per data_conversion()'s own
    gene_selected.drop(['source','population','phenotype'],...) call,
    which leaves 'colour' as BED's 5th/'score' column,
    _add_cytoband_tracks_with_border's own colour lookup key). Using
    'source' here (the original bug) meant looking up 'leaf'/'SAM'/etc.
    in CYTOBAND_COLORMAP, which are never actual keys in it at all - every
    such legend swatch silently fell back to plain white.

    `has_inter_chr_link`/`has_intra_chr_link` are checked independently
    (not a single 'has_links' flag) - a plot could have EITHER only
    within-chromosome or only between-chromosome interactions among
    whatever's actually displayed, and showing both colours regardless
    would list one that never appears as a line anywhere on the plot.

    `truncated_labels` (Update ID ver4-6, R2, blueprint §3.2 Defence 2):
    optional `{truncated ring-label text: full original label}` dict -
    one invisible-swatch legend row per entry, e.g.
    'GAT_biological_pri... = GAT_biological_prior_knowledge__Bayesian',
    so a ring label shortened for space never loses the reader's ability
    to look up what it actually stands for. `None`/empty adds nothing -
    the common case, since most configs never trigger a truncation.

    `has_hub_glyph` (Update ID ver4-25, R12): True whenever ANY k>=3
    interaction hub-and-spoke glyph was actually drawn anywhere this
    session (`'per_order'` layout only - see `plot()`'s own call site) -
    adds ONE marker-only legend row so a reader who has never seen this
    glyph before knows what the filled circles mean, without needing a
    colour scheme of its own (the glyph's spokes/hub already reuse the
    existing 'inter-chr link'/'intra-chr link' red/blue rows above,
    exactly like every k=2 chord already does).

    `combined_orders_used` (Update ID ver4-25, R12): optional set of
    order values (k>=3) actually drawn on a `'combined'`-layout render
    this session - one legend row per order, in ascending order, coloured
    with that order's own `_okabe_ito_colour_for_order()` (see
    `plot()`'s combined-layout call site for why THAT layout colours by
    k instead of red/blue - both k=2 chords and k>=3 hub glyphs share
    this same per-order colour scheme on a combined render, so one shared
    set of legend rows here covers both).

    Region highlighting: `highlight_regions` ({index: {'colour', 'legend'}}, as
    `_highlight_legend_regions()` builds it) adds one coloured row per highlighted region;
    `highlight_intra` / `highlight_inter` add the solid / dashed line rows that tell a
    highlighted within- from a between-chromosome link (their colour is the region's, so the
    ordinary red/blue rows cannot); `highlight_dimmed` adds the grey 'other links' row."""
    handles = []
    handler_map = {}
    if CYTOBAND_COLORMAP is None:
        CYTOBAND_COLORMAP = _pycirclize_config.CYTOBAND_COLORMAP

    _hue_label = {'blue': 'Marker effect (single model)', 'red': 'Marker effect (ensemble model)',
                  'consensus': 'Consensus (models in top level)'}
    for hue in sorted(colours_used_by_hue):
        _levels = []
        for c in colours_used_by_hue[hue]:
            if isinstance(c, str) and c.startswith(hue) and c[len(hue):].isdigit():
                _levels.append(int(c[len(hue):]))
        if not _levels:
            continue
        _low_level, _high_level = min(_levels), max(_levels)
        low_c = CYTOBAND_COLORMAP.get(f'{hue}{_low_level}', '#ffffff')
        high_c = CYTOBAND_COLORMAP.get(f'{hue}{_high_level}', low_c)
        _label = _hue_label.get(hue, f'Marker effect ({hue})')
        if _high_level == _low_level:
            # Every marker actually rendered landed on the exact same
            # quantile level (e.g. the all-zero-effect case) - a flat
            # swatch, not a (visually identical) gradient, is the
            # honest representation here.
            handles.append(mpatches.Patch(facecolor=low_c, edgecolor='black', linewidth=0.3, label=f'{_label} (uniform)'))
        else:
            _proxy = mpatches.Patch(facecolor=low_c, edgecolor='black', linewidth=0.3, label=f'{_label}  weaker \u2192 stronger')
            handles.append(_proxy)
            handler_map[_proxy] = _GradientSwatchHandler(low_c, high_c)

    for colour_value in gene_colours_used:
        color = CYTOBAND_COLORMAP.get(str(colour_value), '#ffffff')
        handles.append(mpatches.Patch(facecolor=color, edgecolor='black', linewidth=0.3, label=str(colour_value)))

    if highlight_regions:
        for _idx in sorted(highlight_regions):
            _r = highlight_regions[_idx]
            handles.append(mpatches.Patch(facecolor=_r['colour'], edgecolor='black', linewidth=0.3,
                                           label=f"Highlighted: {_r['legend']}"))
        if highlight_intra:
            handles.append(mlines.Line2D([0], [0], color='0.25', lw=2, linestyle='solid',
                                          label='Highlighted link, within chromosome'))
        if highlight_inter:
            handles.append(mlines.Line2D([0], [0], color='0.25', lw=2, linestyle='dashed',
                                          label='Highlighted link, between chromosomes'))
        if highlight_dimmed:
            handles.append(mlines.Line2D([0], [0], color=_hl.DIM_GREY, lw=1, alpha=0.6,
                                          label='Other links (faded)'))

    # ver4-31 R12: the link rows follow `CIRCOS_CONFIG.link_colour_mode`.
    #   'chrom' (legacy)  blue inter / red intra
    #   'chrom_cvd'       Okabe-Ito reddish-purple inter / bluish-green intra
    #   'value'           colour = strength (cividis gradient swatch), intra /
    #                     inter told apart by line style (solid / dashed)
    if link_colour_mode == 'value':
        if has_intra_chr_link:
            handles.append(mlines.Line2D([0], [0], color='0.3', lw=2, linestyle='solid', label='intra-chr link'))
        if has_inter_chr_link:
            handles.append(mlines.Line2D([0], [0], color='0.3', lw=2, linestyle='dashed', label='inter-chr link'))
        if has_intra_chr_link or has_inter_chr_link:
            import matplotlib as _mpl
            _cm = _mpl.colormaps[_LINK_VALUE_CMAP]
            _lo_c, _hi_c = _mpl.colors.to_hex(_cm(0.08)), _mpl.colors.to_hex(_cm(0.95))
            _lp = mpatches.Patch(facecolor=_lo_c, edgecolor='black', linewidth=0.3,
                                  label='Link strength  weaker \u2192 stronger')
            handles.append(_lp)
            handler_map[_lp] = _GradientSwatchHandler(_lo_c, _hi_c)
    else:
        _inter_c, _intra_c = ((_CVD_INTER_COLOUR, _CVD_INTRA_COLOUR) if link_colour_mode == 'chrom_cvd'
                              else ('blue', 'red'))
        if has_inter_chr_link:
            handles.append(mlines.Line2D([0], [0], color=_inter_c, lw=2, label='inter-chr link'))
        if has_intra_chr_link:
            handles.append(mlines.Line2D([0], [0], color=_intra_c, lw=2, label='intra-chr link'))

    # Update ID ver4-25, R12: one marker-only row identifying what a
    # filled-circle hub glyph means, whenever 'per_order' layout actually
    # drew at least one this session - its spokes/hub otherwise share the
    # SAME red/blue rows immediately above (no separate colour legend of
    # its own needed for that layout).
    if has_hub_glyph:
        handles.append(mlines.Line2D([0], [0], marker='o', linestyle='none', color='black',
                                      markerfacecolor='none', markersize=7,
                                      label='k≥3 interaction hub (per-order layout)'))

    # Update ID ver4-25, R12: one coloured row per k>=3 order actually
    # drawn on a 'combined'-layout render this session (ascending order) -
    # the SAME Okabe-Ito colour `plot()`'s combined call site used for
    # both that order's own k=2-style chords (when order==2 happens to be
    # part of a combined render) and its hub glyphs.
    if combined_orders_used:
        for _order in sorted(combined_orders_used):
            handles.append(mlines.Line2D([0], [0], color=_okabe_ito_colour_for_order(_order), lw=2,
                                          label=f'k={_order} interaction (combined layout)'))

    # Update ID ver4-6, R2 (blueprint §3.2 Defence 2): one invisible-swatch
    # row per shortened ring label, so the full name a truncated label
    # stands for is always still readable somewhere on the page.
    if truncated_labels:
        for _truncated, _full in sorted(truncated_labels.items()):
            handles.append(mpatches.Patch(facecolor='none', edgecolor='none',
                                           label=f'{_truncated}  =  {_full}'))

    return handles, handler_map


def _new_legend_accumulator():
    """ver4-5 (single session-wide legend): the mutable, shared record of
    every legend-relevant thing actually drawn across ALL circos plots
    produced by one `circos_plot()` call - one instance is created at
    the top of that function and threaded through every `plot()` call
    made during its (PHENOTYPE x POPULATION) loop, so the SAME shape
    `_build_circos_legend_handles()` already expects for a single plot
    (a {hue: {colour strings}} dict, a set of gene colours, and two
    booleans for the two link types) is simply accumulated - unioned -
    across every one of those calls instead of being rebuilt fresh, and
    thrown away, for each one. Mirrors exactly what a single plot() call
    used to compute for itself right before saving its own
    '<...>_legend.png' - see `_merge_into_legend_accumulator()`."""
    return {
        'colours_used_by_hue': {},
        'gene_colours_used': set(),
        'has_inter_chr_link': False,
        'has_intra_chr_link': False,
        # Update ID ver4-6, R2 (blueprint §3.2 Defence 2): {truncated
        # ring-label text -> full original label}, unioned across every
        # plot() call this circos_plot() invocation makes - empty unless
        # `ring_label_fit` actually had to shorten something anywhere in
        # the whole run. See `_build_circos_legend_handles()`'s own new
        # parameter for how this becomes legend rows.
        'truncated_labels': {},
        # Update ID ver4-25, R12: True the first time ANY k>=3 hub-and-
        # spoke glyph is drawn anywhere this session ('per_order' layout
        # only) - empty/False for every run that never selects an
        # order>=3 shapiq method, so the combined legend gains no new row
        # at all in that (overwhelmingly common, pre-existing-config)
        # case.
        'has_hub_glyph': False,
        # Update ID ver4-25, R12: the set of k>=3 orders actually drawn
        # on a 'combined'-layout render anywhere this session - empty
        # unless that layout was both selected AND actually had order>=3
        # data to draw.
        'combined_orders_used': set(),
        # Region highlighting: {region index: {'colour', 'legend'}} plus whether any highlighted
        # within-/between-chromosome link, or any faded 'other' link, was drawn. All empty/False
        # when no region was configured, so the legend gains no row in that case.
        'highlight_regions': {},
        'highlight_intra': False,
        'highlight_inter': False,
        'highlight_dimmed': False,
    }


def _merge_into_legend_accumulator(accumulator, *, colours_used_by_hue=None, gene_colours_used=None,
                                    has_inter_chr_link=None, has_intra_chr_link=None,
                                    truncated_labels=None, has_hub_glyph=None, combined_orders_used=None,
                                    highlight_regions=None, highlight_intra=None, highlight_inter=None,
                                    highlight_dimmed=None):
    """Folds one plot() call's own legend-relevant findings into the
    shared, session-wide `accumulator` (see `_new_legend_accumulator()`).
    Every argument is optional and merged only if provided, since
    `plot()` calls this twice per render - once for colours/gene-colours
    (computed once per call, before its own per-model loop) and once per
    model for the two link-type flags (which DO vary per model - see
    plot()'s own has_inter_chr_link_for_legend/has_intra_chr_link_for_legend)
    - so a single call site would otherwise have to pass placeholder
    values for whichever half it doesn't yet have. Colours are UNIONED
    (a level seen on any one plot in the session belongs in the combined
    legend), and the link-type flags are OR'd (the combined legend shows
    a link type if it appeared on ANY plot this session, not only the
    last one)."""
    if colours_used_by_hue is not None:
        for hue, levels in colours_used_by_hue.items():
            accumulator['colours_used_by_hue'].setdefault(hue, set()).update(levels)
    if gene_colours_used is not None:
        accumulator['gene_colours_used'].update(gene_colours_used)
    if has_inter_chr_link is not None:
        accumulator['has_inter_chr_link'] = accumulator['has_inter_chr_link'] or has_inter_chr_link
    if has_intra_chr_link is not None:
        accumulator['has_intra_chr_link'] = accumulator['has_intra_chr_link'] or has_intra_chr_link
    if truncated_labels:
        accumulator['truncated_labels'].update(truncated_labels)
    if has_hub_glyph is not None:
        accumulator['has_hub_glyph'] = accumulator['has_hub_glyph'] or has_hub_glyph
    if combined_orders_used:
        accumulator['combined_orders_used'].update(combined_orders_used)
    if highlight_regions:
        accumulator['highlight_regions'].update(highlight_regions)
    for _key, _val in (('highlight_intra', highlight_intra), ('highlight_inter', highlight_inter),
                       ('highlight_dimmed', highlight_dimmed)):
        if _val is not None:
            accumulator[_key] = accumulator[_key] or bool(_val)


def _save_circos_legend(handles, handler_map, save_path, *, dpi=600, extra_formats=()):
    """Requirement 1 (correction - separate legend file): builds and saves
    a small, STANDALONE figure containing ONLY the legend - no circos plot
    at all - to its own PNG file.

    ver4-5: called exactly ONCE per `circos_plot()` call now, from
    `circos_plot()` itself after its whole (PHENOTYPE x POPULATION) loop
    has finished calling `plot()` for every combination - not once per
    individual circos plot image as before. `handles`/`handler_map` are
    built from the SESSION-WIDE `legend_accumulator` (every colour/link
    type used on ANY plot generated this call), so the single PNG this
    writes is the combined legend for the whole run, covering every
    component that appears anywhere in `Result/<RESULT_NAME>/`'s circos
    plots - not just one of them. This function's own drawing logic is
    otherwise unchanged: it still does nothing (no file written) if
    there's nothing to show a legend for, and is still sized to the
    actual number of legend rows it's asked to draw.

    Replaces the earlier approach of widening the main plot's own figure
    and shifting its axes to carve out room for the legend inside the SAME
    image - a legend can end up needing anywhere from a couple of rows to
    dozens (Requirement 1's own fix reads every colour genuinely used in
    the data, so a richly-annotated gene file needs a correspondingly
    long one), and getting a one-shot margin estimate right for that whole
    range, on top of an unpredictable chromosome count/sizing, proved
    fragile in practice. A completely separate file sidesteps needing
    that estimate at all: there is nothing for the legend to ever overlap,
    on any plot, at any size or row count.

    The main plot's own figure IS still adjusted slightly (see plot()'s
    own comment on its unit-label placement) - but only by a small, FIXED
    amount for that one line of caption text, not a variable amount
    depending on legend content, which is what made this approach workable
    for the label but not for the legend itself.

    Sized to the ACTUAL number of legend rows about to be drawn - a short
    legend gets a small image, a long one (dozens of gene-pathway colours)
    gets a taller one - rather than a fixed guess either way. Does nothing
    (no file written) if there's nothing to show a legend for.

    ver4-4 R7.h: `dpi` (default 600, this function's own legacy-preserving
    value) replaces what used to be a hard-coded `dpi=600` in the
    `_safe_savefig()` call below - callers thread the run's actual
    `PLOT_DPI` config value through (default 300, per the blueprint's
    §4a reproducibility policy)."""
    if not handles:
        return
    n_rows = len(handles)
    fig_height_in = max(1.2, 0.28 * n_rows + 0.6)
    legend_fig = plt.figure(figsize=(3.2, fig_height_in))
    legend_fig.legend(handles=handles, handler_map=handler_map, loc='center left',
                       fontsize=8, title='Legend', title_fontsize=9, frameon=True)
    _ps.save_figure(legend_fig, save_path, dpi=dpi, saver=_safe_savefig, bbox_inches='tight',
                    extra_formats=extra_formats)
    plt.close(legend_fig)


def _attach_link_patches(ax, link_patches):
    """Attach `link_patches` (pycirclize link patches, as accumulated in
    ``circos._patches``) to a base figure's axes exactly the way
    ``Circos.plotfig()`` attaches its default-zorder patches: deep-copied,
    ``clip_on=False``, batched into one ``PatchCollection(match_original=
    True)`` - so links are rasterised by the same code path (and come out
    pixel-identical) as when ``plotfig()`` drew them itself.

    ``plotfig()`` draws links BEFORE the ring patches (they are first in
    its shared collection); this collection is added after the base one,
    so its zorder is set just below the base collection's default 1 to keep
    that stacking. Returns the added artists (for callers that want to
    ``.remove()`` them).
    """
    from copy import deepcopy
    from matplotlib.collections import PatchCollection
    patches = []
    others = []
    for patch in link_patches:
        p = deepcopy(patch)
        p.set_clip_on(False)
        if p.get_zorder() == 1 and p.get_hatch() is None:
            patches.append(p)
        else:
            ax.add_patch(p)
            others.append(p)
    added = list(others)
    if patches:
        coll = PatchCollection(patches, match_original=True, clip_on=False)
        coll.set_zorder(0.99)
        ax.add_collection(coll)
        added.append(coll)
    return added


_LABEL_MARGIN_IN = 1.0     # blank border added around the circle when top-marker labels are drawn


def _add_top_marker_labels(circos, circos_config, lopts, label_tsv, gene_tsvs, *,
                            r_marker=101.0, r_label=112.0):
    """Draw the labelled markers' (or genes') names radially outside the outer
    ring, each joined to its marker by a short leader line (R14). Label angles
    are spread with `circos_geometry.spread_labels()` so neighbours stay at
    least `label_min_sep_deg` apart. Returns the number of labels drawn."""
    if not os.path.exists(label_tsv):
        return 0
    # ver4-45: read the ranked candidate pool (falls back to the plain top-N table
    # for results written by an older version).
    _pool = _label_pool_path(label_tsv)
    df = pd.read_csv(_pool if os.path.exists(_pool) else label_tsv, sep='\t', dtype={'chromosome': str})
    if df.empty:
        return 0
    want = int(lopts['top'])
    genes = _gene_names_for_markers(df, gene_tsvs)
    sectors = {sec.name: sec for sec in circos.sectors}
    items = []
    seen = set()
    for _, row in df.iterrows():
        if len(items) >= want:                 # exactly `want` distinct labels (the pool is ranked)
            break
        sec = sectors.get(str(row['chromosome']))
        if sec is None:
            continue
        text = genes.get(str(row['name']), str(row['name']))
        if text in seen:                       # two hits in one gene: label it once, take the next-strongest candidate
            continue
        seen.add(text)
        items.append((text, sec.x_to_rad(float(row['pos']), ignore_range_error=True)))
    if len(items) < want:
        print(f"[circos_plot] label_top_markers: asked for {want} labels but only {len(items)} distinct "
              f"label(s) are available (candidate pool of {df.shape[0]} exhausted).")
    if not items:
        return 0
    angles = [math.degrees(r) for _, r in items]
    lo = min(math.degrees(sec.rad_lim[0]) for sec in circos.sectors)
    hi = max(math.degrees(sec.rad_lim[1]) for sec in circos.sectors)
    spread = circos_geometry.spread_labels(angles, lopts['min_sep'], lo=lo, hi=hi)
    size = max(5.0, min(8.0, float(circos_config.get('label_size', 6)) * 1.5))

    def _leader(rad0, rad1):
        def draw(ax):
            ax.plot([rad0, rad1], [r_marker, r_label - 1.0], color='0.35', linewidth=0.5,
                    solid_capstyle='butt', clip_on=False)
        return draw

    for (text, rad0), ang in zip(items, spread):
        circos.text(text, r=r_label, deg=ang, size=size, adjust_rotation=True,
                    orientation='vertical', color='black')
        circos._plot_funcs.append(_leader(rad0, math.radians(ang)))
    return len(items)


def _finish_circos_figure(fig, circos_config):
    """The shared tail end of every `plot()` render (k=2's own historical
    path, and both new k>=3 render paths Update ID ver4-25/R12 adds) -
    opens the same small, fixed margin for the axis-unit caption line
    every render has always had. Extracted verbatim, byte-for-byte, from
    what used to be `plot()`'s own inline code (see AC12.3: this is a
    SOURCE-level refactor only - the exact same computation, on the exact
    same `fig`, still runs for k=2, so k=2's own OUTPUT is unaffected)
    purely so the new k>=3 render paths don't have to duplicate it a
    second and third time.

    Requirement (bugfix - unit label overlapping a chromosome name): a
    fixed-position fig.text(0.5, 0.02, ...) worked for a modest
    chromosome count, but for a genome with MANY chromosomes (e.g. a real
    26-sub-genome cotton assembly, A01-A13/D01-D13) some chromosome's own
    name label inevitably ends up positioned right at the bottom of the
    circle, where the unit label was also fixed - colliding with it
    (confirmed with a real render at that scale during development).
    Rather than trying to predict WHICH chromosome that will be
    (unknowable in general - it depends on chromosome count, sizes, and
    the start/end angles), this opens a small, fixed, DEDICATED margin
    below the circle instead - unlike the legend (Requirement 13's own
    correction, a separate file), a single line of caption text doesn't
    need a variable-sized allowance, so growing the figure by a small
    fixed amount and shifting the polar axes up within it by exactly that
    much (so the circle's own on-page size is unchanged) reliably clears
    space for it - verified empirically against a real 26-chromosome
    render, not just assumed. Mutates `fig` in place; returns nothing."""
    if _label_options(circos_config)['top'] > 0:
        # ver4-31 R14: room for the radial marker / gene labels around the
        # circle (the circle itself keeps its on-page size).
        _m = _LABEL_MARGIN_IN
        _w0, _h0 = fig.get_size_inches()
        _p0 = fig.axes[0].get_position()
        fig.set_size_inches(_w0 + 2 * _m, _h0 + 2 * _m)
        fig.axes[0].set_position([(_p0.x0 * _w0 + _m) / (_w0 + 2 * _m), (_p0.y0 * _h0 + _m) / (_h0 + 2 * _m),
                                   _p0.width * _w0 / (_w0 + 2 * _m), _p0.height * _h0 / (_h0 + 2 * _m)])
    _extra_height_in = 0.6
    _fig_width_in, _fig_height_in = fig.get_size_inches()
    _polar_ax = fig.axes[0]
    _old_pos = _polar_ax.get_position()
    _old_y0_in = _old_pos.y0 * _fig_height_in
    _old_height_in = _old_pos.height * _fig_height_in
    _new_fig_height_in = _fig_height_in + _extra_height_in
    fig.set_size_inches(_fig_width_in, _new_fig_height_in)
    _polar_ax.set_position([
        _old_pos.x0,
        (_old_y0_in + _extra_height_in) / _new_fig_height_in,
        _old_pos.width,
        _old_height_in / _new_fig_height_in,
    ])
    fig.text(0.5, 0.02, _circos_axis_unit_label(circos_config), ha='center', va='bottom',
             fontsize=9, color='black')


def _circos_models_to_render(interactions_original, higher_order_ring_frame):
    """The ordered list of model names ``plot()`` renders one interaction
    figure for, for one (PHENOTYPE, POPULATION): every model with k=2 rows
    (first-appearance order), then any model that has k>=3 rows only; the
    sentinel ``['not_returned']`` when there is nothing at all (one plain
    circos, no interaction links).

    Factored out of ``plot()`` unchanged so ``circos_plot()``'s parallel
    scheduler can enumerate the same render units ``plot()`` will iterate,
    without duplicating (and eventually drifting from) that logic.
    """
    if higher_order_ring_frame is None:
        higher_order_ring_frame = pd.DataFrame(columns=['model', 'order', 'members', 'value'])
    if interactions_original.shape[0] != 0:
        model_selected = interactions_original['model'].unique().tolist()
    else:
        model_selected = ['not_returned']
    _hok_models = higher_order_ring_frame['model'].unique().tolist() if higher_order_ring_frame.shape[0] else []
    if model_selected == ['not_returned'] and _hok_models:
        model_selected = list(_hok_models)
    else:
        for _m in _hok_models:
            if _m not in model_selected:
                model_selected.append(_m)
    return model_selected


# --------------------------------------------------------------------------- #
# ver4-31 R13: consensus ring - how many models put a region in their top
# quantile level. A pure reduction over the per-model marker_effect_*.tsv files
# `quantile_conversion()` already wrote; no new data path.
# --------------------------------------------------------------------------- #

CONSENSUS_SCOPES = ('base_models', 'all_models')
_CONSENSUS_CMAP = 'Purples'
_CONSENSUS_TRACK = '_consensus'


def _consensus_options(circos_config):
    """`{'on', 'level', 'scope'}` from `CIRCOS_CONFIG.consensus_ring` /
    `.consensus_level` (0-9, default 9) / `.consensus_scope`."""
    c = circos_config or {}
    try:
        level = min(max(int(c.get('consensus_level', 9)), 0), 9)
    except (TypeError, ValueError):
        level = 9
    scope = str(c.get('consensus_scope', 'base_models') or 'base_models').strip().lower()
    return {'on': bool(c.get('consensus_ring', False)), 'level': level,
            'scope': scope if scope in CONSENSUS_SCOPES else 'base_models'}


def _cytoband_cmap_with_consensus(cmap, n_max):
    """A copy of `cmap` (or of pycirclize's default when `None`) that also
    holds `consensus0 .. consensus<n_max>` - a light-to-dark `Purples` ramp -
    for any of those keys it does not already define (a user-supplied entry
    wins)."""
    import matplotlib as _mpl
    out = dict(cmap if cmap is not None else _pycirclize_config.CYTOBAND_COLORMAP)
    ramp = _mpl.colormaps[_CONSENSUS_CMAP]
    n_max = max(int(n_max), 1)
    for k in range(n_max + 1):
        out.setdefault(f'consensus{k}', _mpl.colors.to_hex(ramp(0.08 + 0.87 * k / n_max)))
    return out


def _colour_level(colour):
    """Trailing quantile level of a colour key ('blue7' -> 7), else None."""
    import re as _re
    m = _re.search(r'(\d+)$', str(colour))
    return int(m.group(1)) if m else None


def _consensus_models(MODEL, scope):
    """The models a consensus counts: every one (`all_models`) or only the base
    models (`base_models`) - ensembles are functions of the base models, so
    counting them would inflate the agreement."""
    if scope == 'all_models':
        return list(MODEL)
    return [m for m in MODEL if not (is_naive_ensemble_label(m) or is_wopt_label(m))]


def _build_consensus_tsv(model_tsvs, level, out_path, label_table=None, label_top=0):
    """Reduce per-model `marker_effect_*.tsv` files (chromosome, start, end,
    index, colour) to one consensus file at `out_path`: for every
    (chromosome, start, end) region, the number of models whose quantile level
    is >= `level`, written as the colour key `consensus<count>`. A region a
    model does not list counts as "not in its top". Returns
    `(out_path, n_models_counted)`.

    ver4-45: when `label_table` is given (and `label_top` > 0) the regions are
    also ranked - by consensus count, ties broken by the summed quantile level
    across the counted models - and written as the top-marker label table (+
    candidate pool) at that path, so 'Label source = consensus ring' costs no
    extra file read."""
    counts = {}
    level_sums = {}
    n_counted = 0
    for path in model_tsvs:
        df = pd.read_csv(path, sep='\t', dtype={'chromosome': str})
        if df.empty:
            continue
        n_counted += 1
        lv = df['colour'].map(_colour_level).fillna(-1).astype(int)
        key_frame = pd.DataFrame({'chromosome': df['chromosome'].astype(str), 'start': df['start'],
                                   'end': df['end'], 'top': (lv >= level).astype(int),
                                   'lv': lv.clip(lower=0)})
        per_file = key_frame.groupby(['chromosome', 'start', 'end'], sort=False)[['top', 'lv']].max()
        for key, top, lvl in zip(per_file.index, per_file['top'].tolist(), per_file['lv'].tolist()):
            counts[key] = counts.get(key, 0) + int(top)
            level_sums[key] = level_sums.get(key, 0) + int(lvl)
    rows = [{'chromosome': c, 'start': a, 'end': b, 'index': f'{c}:{a}-{b}', 'colour': f'consensus{n}'}
            for (c, a, b), n in counts.items()]
    _atomic_to_csv(pd.DataFrame(rows, columns=['chromosome', 'start', 'end', 'index', 'colour']),
                   out_path, sep='\t', index=False)
    if label_table is not None and int(label_top) > 0 and counts:
        _lt = pd.DataFrame({
            'chromosome': [k[0] for k in counts],
            'start': [k[1] for k in counts],
            'end': [k[2] for k in counts],
            'name': [f"{k[0]}:{int((float(k[1]) + float(k[2])) / 2)}" for k in counts],
            'value': [counts[k] for k in counts],
            'tiebreak': [level_sums[k] for k in counts]})
        _write_label_table(label_table, _lt, int(label_top))
    return out_path, n_counted


def plot(interactions_original, chrom_info, gene_info, pop_source, PHENOTYPE, MODEL, circos_config, CYTOBAND_COLORMAP, POPULATION, RESULT_NAME, plot_dpi=600, legend_accumulator=None,
         higher_order_ring_frame=None, interaction_orders=(2,), higher_order_layout='per_order', only_models=None):
    """ver4-5 (single session-wide legend): `legend_accumulator`, when
    provided, is a mutable dict shared across EVERY plot() call made
    during one circos_plot() invocation - see that function's own
    `_new_legend_accumulator()`/`_merge_into_legend_accumulator()`. Each
    call to plot() now merges the colours/link-types it actually drew
    into that shared dict INSTEAD of building and saving its own,
    per-plot '<...>_legend.png' file - circos_plot() builds and saves
    the ONE combined legend, covering everything used across every
    phenotype/population/model rendered in that call, only once, after
    its own (i, j) loop over PHENOTYPE x POPULATION has finished calling
    plot() for every combination. `legend_accumulator=None` (the
    default) is kept only so plot() still has a well-defined, harmless
    behaviour (accumulate nothing, save nothing) if ever called without
    one; circos_plot() itself always passes one.

    Update ID ver4-25, R12 (new, additive parameters - I11/AC12.3):
    `higher_order_ring_frame` - the DataFrame `higher_order_interaction()`
    returns (columns `['model', 'order', 'members', 'value']`), already
    resolved to genomic coordinates and normalised, for THIS (PHENOTYPE,
    POPULATION) - `None`/empty (the default, and what every caller that
    predates this feature passes implicitly) means there is nothing to
    draw beyond k=2, and the ENTIRE per-model loop below runs exactly the
    historical code path with exactly the historical output.
    `interaction_orders` - which k values to actually draw this call,
    default `(2,)` (today's only order, unconditionally, for a caller
    that never passes this at all). `higher_order_layout` - `'per_order'`
    (default, recommended) draws one file per (model, order), with k=2
    keeping its historical filename; `'combined'` draws ONE file per
    model spanning every requested order (k=2 chords plus every k>=3
    hub glyph together), coloured by k (Okabe-Ito) rather than the usual
    red/blue intra-/inter-chromosome split, with linestyle (solid/dashed)
    carrying that intra-/inter- distinction instead.

    Performance update (circos speed-up): `only_models`, when given (an
    iterable of model names), restricts the per-model render loop below to
    those models - the loop-invariant base (every marker-effect/gene ring,
    ticks) is still built, once, but only for this subset's figures. It is
    how ``circos_plot()`` hands DISJOINT slices of one (PHENOTYPE,
    POPULATION)'s models to different worker processes. `None` (default)
    renders every model exactly as before.

    The k=2 renders no longer call ``circos.plotfig()`` once per model:
    that call deep-copies and re-adds every ring's patches, and dominated
    the cost of a many-model plot (cost ~ models x rings). The link-free
    base figure is now built ONCE per plot() call and only each model's
    own link patches are attached, saved and detached again - the same
    pixels (links are drawn beneath the ring patches, as before), a
    fraction of the work.
    """
    if higher_order_ring_frame is None:
        higher_order_ring_frame = pd.DataFrame(columns=['model', 'order', 'members', 'value'])
    _requested_orders = sorted({int(o) for o in (interaction_orders or (2,))})
    _only_models = None if only_models is None else set(only_models)

    # (ver4-25, R12: also includes models that only have k>=3 rows - see
    # `_circos_models_to_render()`.)
    model_selected = _circos_models_to_render(interactions_original, higher_order_ring_frame)

    # ver4-4 R7.c (blueprint §2.7.2, PC-3): Circos.initialize_from_bed(...)
    # and every marker-effect / gene-region / tick track added below do
    # NOT depend on `n` (model_selected[n]) - only the chord LINKS added
    # further down, inside the per-model loop, do. This whole block used
    # to be rebuilt from scratch, in full, once PER MODEL (i.e. once per
    # element of model_selected, re-reading and re-drawing every marker-
    # effect/gene track from disk each time) - it now runs exactly ONCE
    # for this whole (PHENOTYPE, POPULATION) plot.
    #
    # Verified via a REAL pre-check (blueprint's PC-3), not just reasoned
    # about: run directly against pycirclize 1.10.1 (the tree pins
    # 1.9.1 on Linux / 1.10.0 on Windows - close enough for this specific
    # internal-API behaviour to be trusted). A single Circos object CAN
    # be rendered to MORE THAN ONE figure via repeated .plotfig() calls -
    # each call creates a fresh Figure/PolarAxes (ax=None) and reads
    # (never mutates) the object's own patch/plot-function lists via an
    # internal deepcopy (pycirclize's own Circos._get_all_patches()), so
    # calling .plotfig() a second time does not consume or alter
    # anything an earlier call already drew.
    #
    # The ONE thing that DOES accumulate across repeated .plotfig() calls
    # on the same object - because pycirclize exposes no public "clear
    # links" method - is exactly what THIS function adds via
    # circos.link() (the chord links, added per-model inside the loop
    # below): Circos.link() appends its own patch to the private
    # `circos._patches` list (confirmed directly from pycirclize's own
    # source). `_base_patch_count`, recorded once here before the loop
    # ever adds a link, is truncated back to at the START of every
    # iteration (`del circos._patches[_base_patch_count:]`) so each
    # model's render starts from the same clean, link-free state rather
    # than accumulating every previously-rendered model's links onto
    # every subsequent one. This exact sequence (add a link, render,
    # truncate, add a DIFFERENT link, render again) was run directly
    # against a real pycirclize Circos object during this pre-check: the
    # two renders' link geometry differed exactly as expected and never
    # accumulated the earlier call's link.
    circos = Circos.initialize_from_bed(_circos_intermediate_dir(RESULT_NAME)+'/chrom_'+str(POPULATION)+".bed", space=circos_config['space'], start=circos_config['start']+2+3, end=circos_config['end']-3)

    # Update ID ver4-6, R2 (blueprint §3.2 Defence 2): `gene_source` is
    # hoisted ABOVE the model ring-drawing loop (it used to be computed
    # only once the "Add known gene regions" block itself started) so the
    # TOTAL ring count for this specific (PHENOTYPE, POPULATION) plot -
    # model rings plus gene-source rings - is known BEFORE any ring is
    # drawn. `ring_layout='fit'` (see circos_geometry.ring_geometry())
    # needs that total up front to thin the per-ring stride so every ring
    # fits; `ring_layout='legacy'` (the absent-key default) does not
    # depend on it at all (I11, AC2.5) - this hoist is a no-op for that
    # path. `gene_source` itself is unchanged: the same
    # `pd.unique(pop_source.loc[...])` read that already existed here,
    # just moved earlier - it depends only on `pop_source`/`POPULATION`/
    # `PHENOTYPE`, none of which the model loop below touches.
    if len(MODEL) == 0:
        _warn_no_model_data(PHENOTYPE, POPULATION)
        return
    if gene_info is not None:
        gene_source = pd.unique(pop_source.loc[(pop_source['population'].astype(str)==str(POPULATION)) & (pop_source['phenotype']==str(PHENOTYPE)),'source'])
    else:
        gene_source = np.array([], dtype=object)
    # ver4-31 R13: optional consensus ring (outermost), built from the per-model
    # marker_effect_*.tsv files quantile_conversion() has already written.
    _cons_opts = _consensus_options(circos_config)
    _cons_path, _cons_n = None, 0
    _lopts_cons = _label_options(circos_config)
    _cons_for_labels = _lopts_cons['top'] > 0 and _lopts_cons['consensus']     # ver4-45: label source = consensus ring
    if _cons_opts['on'] or _cons_for_labels:
        _cons_models = _consensus_models(list(MODEL), _cons_opts['scope'])
        if len(_cons_models) >= 2:
            _cons_path, _cons_n = _build_consensus_tsv(
                [_circos_intermediate_dir(RESULT_NAME)+'/marker_effect_'+m+'_'+PHENOTYPE+'_'+str(POPULATION)+'.tsv'
                 for m in _cons_models],
                _cons_opts['level'],
                _circos_intermediate_dir(RESULT_NAME)+'/marker_effect_consensus_'+PHENOTYPE+'_'+str(POPULATION)+'.tsv',
                label_table=_label_table_path(RESULT_NAME, PHENOTYPE, POPULATION) if _cons_for_labels else None,
                label_top=_lopts_cons['top'] if _cons_for_labels else 0)
            if not _cons_opts['on']:
                _cons_path = None          # built for the labels only - the ring itself is not drawn
        else:
            print(f"[circos_plot] consensus_ring: only {len(_cons_models)} model(s) in scope "
                  f"'{_cons_opts['scope']}' for {PHENOTYPE}/{POPULATION} - consensus ring"
                  f"{' and consensus-based marker labels' if _cons_for_labels else ''} skipped.")
            if _cons_for_labels:           # never label from a stale table of an earlier run
                _stale = _label_table_path(RESULT_NAME, PHENOTYPE, POPULATION)
                for _f in (_stale, _label_pool_path(_stale)):
                    try:
                        os.remove(_f)
                    except OSError:
                        pass
    _n_rings_total = len(MODEL) + len(gene_source) + (1 if _cons_path else 0)

    # Update ID ver4-6, R2: `circos_config.get(..., <legacy default>)`
    # throughout this block is what makes an absent-key (pre-ver4-6)
    # config render byte-identically (I11, AC2.5) - 'legacy'/'off'/8.0
    # reproduce the exact pre-ver4-6 literals this block used to hard-code.
    _ring_layout = circos_config.get('ring_layout', 'legacy')
    _ring_label_size_cfg = circos_config.get('ring_label_size', 8.0)
    _ring_label_fit = circos_config.get('ring_label_fit', 'off')
    _ring_label_max_chars = int(circos_config.get('ring_label_max_chars', 0) or 0)
    _ring_geometries = circos_geometry.ring_geometry(_n_rings_total, layout=_ring_layout)
    # The seam gap this specific render is ACTUALLY using - not a
    # suggestion, the real, resolved `start`/`end` this call was given -
    # so Defence 2 guards against whatever gap is really on the page,
    # including an old config's hand-set angles (blueprint §3.8: "old
    # config with start/end hand-set to an overlapping pair" - I11 wins;
    # `ring_label_fit='off'`, the absent-key default, renders exactly as
    # before regardless of this calculation ever running).
    _seam_gap_deg = 360.0 - (circos_config['end'] - circos_config['start'])
    # {truncated label -> full original label}, collected across both
    # ring-drawing loops below and folded into the shared legend
    # accumulator (alongside colours/gene-colours) so no information is
    # lost when a label is shortened for space - see
    # `_build_circos_legend_handles()`'s own new parameter.
    _truncated_ring_labels = {}
    # ver4-95: MODEL_LABEL_STYLE ('full' default = unchanged). Display only - the ring's data, track
    # name, colour and file names keep the full label; only the TEXT written on the ring is shortened,
    # so a 'Super learner__Bayesian optimisation' ring is labelled 'Bayesian optimisation' (and
    # fits a smaller seam gap). Labels that would clash keep their full text.
    _ring_label_text_of = _ps.shown_labels(list(MODEL), circos_config)

    def _resolve_ring_label(raw_label, r_centre):
        """Defence 2 (blueprint §3.2): the renderer-side guard that makes
        ring-label overlap impossible regardless of whether Defence 1's
        own GUI-time PREDICTION of the ring-label list was right, a
        config predates this update, or a person hand-edited the seam
        angles. Returns (label_text, size_pt) to actually draw.

        `ring_label_fit='off'` (the absent-key legacy default) returns
        `raw_label` and the configured/legacy size UNCHANGED - this is
        the path that keeps a pre-ver4-6 config byte-identical (AC2.5).
        """
        label_text = raw_label
        if _ring_label_max_chars > 0 and len(label_text) > _ring_label_max_chars:
            _hard_capped = label_text[:_ring_label_max_chars].rstrip() + '\u2026'
            _truncated_ring_labels[_hard_capped] = raw_label
            label_text = _hard_capped

        if _ring_label_fit == 'off':
            return label_text, _ring_label_size_cfg

        size_pt = circos_geometry.fit_label_size(
            label_text, r_centre, _seam_gap_deg, size_pt=_ring_label_size_cfg, min_size_pt=4.0,
        )
        # Update ID ver4-6, R3 (bugfix - Defence 2 rendered with zero
        # headroom): this used to compare the raw, zero-margin
        # `seam_gap_deg_for_label()` requirement directly against
        # `_seam_gap_deg` - a THIRD, separately-maintained "does it fit"
        # check that could (and, on a real render, did - see
        # `circos_geometry.RENDER_SAFETY`'s own note) disagree with the
        # margin `fit_label_size()` itself just used to pick `size_pt`.
        # `label_fits()` is the same single authority both of those
        # already call internally, so this final check can never drift
        # out of sync with them again.
        _still_overflows = not circos_geometry.label_fits(
            label_text, size_pt, r_centre, _seam_gap_deg,
        )
        if _still_overflows and 'truncate' in _ring_label_fit:
            _truncated = circos_geometry.truncate_label(label_text, r_centre, _seam_gap_deg, 4.0)
            _truncated_ring_labels[_truncated] = raw_label
            return _truncated, 4.0
        return label_text, size_pt

    cnt = 0
    # Add genomic marker effects
    _colours_used_by_hue = {}  # e.g. {'blue': {'blue0', 'blue3', 'blue7'}, ...}
    if _cons_path:
        _r_lo, _r_hi, _r_centre = _ring_geometries[cnt]
        _cons_colours = _add_cytoband_tracks_with_border(circos, (_r_lo, _r_hi), _cons_path,
                                                          track_name=_CONSENSUS_TRACK, cytoband_cmap=CYTOBAND_COLORMAP)
        _label_text, _label_size = _resolve_ring_label(f'Consensus (n/{_cons_n})', circos.tracks[-1].r_center - 1)
        circos.text(_label_text, r=circos.tracks[-1].r_center-1, deg=0, size=_label_size, color="black")
        cnt += 1
        _colours_used_by_hue.setdefault('consensus', set()).update(_cons_colours)
    for i in range(len(MODEL)):
         _tsv_path = _circos_intermediate_dir(RESULT_NAME)+'/marker_effect_'+MODEL[i]+'_'+PHENOTYPE+'_'+str(POPULATION)+'.tsv'
         _r_lo, _r_hi, _r_centre = _ring_geometries[cnt]
         # ver4-4 R7.d: _add_cytoband_tracks_with_border() now RETURNS the
         # set of colour values it actually drew (a byproduct of its own,
         # now-bucketed, drawing loop) - replaces the second, dedicated
         # pd.read_csv(_tsv_path, ...)['colour'].unique() read this line
         # used to do purely to recover the same information (Requirement
         # 13's "only colours that actually appear on this plot" rule).
         _tsv_colours = _add_cytoband_tracks_with_border(circos, (_r_lo, _r_hi), _tsv_path, track_name=MODEL[i], cytoband_cmap=CYTOBAND_COLORMAP)
         _label_text, _label_size = _resolve_ring_label(_ring_label_text_of.get(MODEL[i], MODEL[i]), circos.tracks[-1].r_center - 1)
         circos.text(_label_text, r=circos.tracks[-1].r_center-1, deg=0, size=_label_size, color="black")
         cnt+=1
         _hue = _ring_hue(MODEL[i], _ensemble_hue_rule(circos_config))      # ver4-31 R10
         _colours_used_by_hue.setdefault(_hue, set()).update(_tsv_colours)
    
    # Add known gene regions
    gene_colours_used = set()
    if gene_info is not None:
        for i in range(len(gene_source)):    
            _gene_tsv_path = _circos_intermediate_dir(RESULT_NAME)+'/gene_info_'+str(PHENOTYPE)+'_'+str(gene_source[i])+'_'+str(POPULATION)+'.tsv'
            _r_lo, _r_hi, _r_centre = _ring_geometries[cnt]
            # ver4-4 R7.d: _add_cytoband_tracks_with_border() now RETURNS
            # the set of colour values it actually drew (a byproduct of
            # its own, now-bucketed, drawing loop) - replaces this
            # block's own second, dedicated
            # pd.read_csv(_gene_tsv_path, ...)['colour'].unique() read
            # that used to run purely to recover the same information.
            # 'source' (leaf/SAM/QTL/wisser_et_al, etc.) is only ever a
            # RING LABEL identifying which data source a gene annotation
            # came from - it's never itself a colour-coded category; the
            # actual fill colour comes from gene_info.csv's own 'colour'
            # column (a pathway/category name, e.g. 'photoperiod'),
            # which data_conversion() writes into this tsv's 'score'
            # column (see _add_cytoband_tracks_with_border()'s own note
            # on why 'colour' lives in a BED file's 'score' column).
            _gene_colours = _add_cytoband_tracks_with_border(circos, (_r_lo, _r_hi), _gene_tsv_path, track_name=gene_source[i], cytoband_cmap=CYTOBAND_COLORMAP)
            _label_text, _label_size = _resolve_ring_label(gene_source[i], circos.tracks[-1].r_center - 1)
            circos.text(_label_text, r=circos.tracks[-1].r_center-1, deg=0, size=_label_size, color="black")
            cnt+=1
            gene_colours_used.update(_gene_colours)

    # ver4-5: fold this plot's own colour usage into the session-wide
    # accumulator immediately (rather than at the bottom of this
    # function, alongside the link-type flags) - it doesn't depend on
    # `n`/model_selected at all (see the comment above `circos =
    # Circos.initialize_from_bed(...)`), so it's only computed once per
    # plot() call, same as `_colours_used_by_hue`/`gene_colours_used`
    # themselves. Update ID ver4-6, R2: `truncated_labels` (empty unless
    # `ring_label_fit` actually shortened something) is folded in
    # alongside them, for the exact same reason.
    if legend_accumulator is not None:
        _merge_into_legend_accumulator(
            legend_accumulator, colours_used_by_hue=_colours_used_by_hue,
            gene_colours_used=gene_colours_used, truncated_labels=_truncated_ring_labels,
        )

    # Add ticks to the outermost ring
    for sector in circos.sectors:
        # Requirement 6: a bit more radial space between the tick
        # numbers and the chromosome name text above them - label_margin
        # (pycirclize's own gap between a tick and its number label,
        # default 0.5) pushes the tick numbers a little further out,
        # and the chromosome name's own radius is nudged out to match
        # (105 -> 108) so the two don't end up crowding each other -
        # modest increases on both sides rather than a single large
        # jump, so the gap grows without leaving an awkwardly empty
        # ring between them.
        #
        # Requirement 1 (bugfix - the actual root cause of labels still
        # overlapping ring/tick data on a many-chromosome genome, e.g.
        # a real 26-chromosome cotton assembly): the chromosome NAME
        # text below used to be size=10, hardcoded - completely
        # unaffected by circos_config['label_size'] (the GUI's own
        # 'Label font size' field, and everything the Requirement 1
        # start/end-angle fix was calibrated against). The chromosome
        # name is the WIDER of the two texts (e.g. 'A01', vs a tick's
        # 1-3 digit number), so it was always the real driver of
        # overlap on a genome with many, densely-packed sectors - and
        # a fixed size=10 never shrank no matter how small
        # 'Label font size' was suggested/set to, which is exactly why
        # a real render still showed overlap despite that suggestion
        # already being smaller for many chromosomes.
        #
        # Tick numbers get their OWN, separately-derived size instead
        # of directly sharing circos_config['label_size'] - short
        # numeric ticks don't need to shrink nearly as aggressively as
        # a multi-character chromosome name does to avoid the same
        # overlap, and forcing them down to the (usually smaller)
        # chromosome-name size made them harder to read for no
        # actual benefit (a real, reported regression). Floored at
        # 5pt and capped at 8pt regardless of how small the
        # chromosome name itself gets.
        # Requirement (chromosome name now matches tick size): the
        # chromosome name used to render at circos_config['label_size']
        # directly (the smaller of the two sizes, deliberately kept
        # small so it wouldn't overlap neighbouring ticks/data on a
        # many-chromosome genome) while tick numbers got their own,
        # separately-derived _tick_label_size (see the comment above)
        # - by explicit request, the chromosome name now renders at
        # THAT SAME, larger size instead, to read more consistently
        # with the ticks around it. Since the chromosome name is the
        # WIDER of the two texts to begin with (e.g. 'A01', vs a
        # tick's 1-3 digit number), this makes it the same size AND
        # still the wider text - so it remains the real driver of how
        # much seam-gap room is needed, now more so than before. See
        # main_app.py's own _circos_suggest_start_end_angle() - it
        # applies this SAME size transform to whatever label size it's
        # given before calculating the gap, specifically so the two
        # stay in sync and this size increase doesn't reintroduce the
        # overlap it was originally calibrated against.
        _tick_label_size = max(5.0, min(8.0, circos_config['label_size'] * 1.8))
        sector.text(sector.name, r=108, size=_tick_label_size)
        sector.get_track(MODEL[0]).xticks_by_interval(
            circos_config['scale'],
            label_size=_tick_label_size,
            label_orientation="vertical",
            label_margin=1.5,
            label_formatter=lambda v: f"{v / circos_config['scale']:.0f}",
        )

    # ver4-31 R14: top-marker / gene labels (loop-invariant, so added before the base
    # figure is frozen; nothing happens when `label_top_markers` is 0).
    _lopts_plot = _label_options(circos_config)
    if _lopts_plot['top'] > 0:
        _n_labelled = _add_top_marker_labels(
            circos, circos_config, _lopts_plot, _label_table_path(RESULT_NAME, PHENOTYPE, POPULATION),
            [_circos_intermediate_dir(RESULT_NAME)+'/gene_info_'+str(PHENOTYPE)+'_'+str(g)+'_'+str(POPULATION)+'.tsv'
             for g in gene_source])
        if _n_labelled == 0:
            print(f"[circos_plot] label_top_markers: no labels for {PHENOTYPE}/{POPULATION} "
                  f"(the label model has no data for it).")

    # Region highlighting (optional, off unless CIRCOS_CONFIG['highlight_regions'] resolves to a
    # region on this plot): the band / wedge / tags are loop-invariant like every ring above, so
    # they are drawn here, once, before the base is frozen. `_hl_sfx` keeps a highlighted figure
    # from overwriting the plain one of an earlier run.
    _hl_regions, _hl_opts = _resolve_highlight(
        circos, circos_config,
        neutral_colours=(higher_order_layout == 'combined' and len(_requested_orders) >= 2
                         and max(_requested_orders) >= 3))
    _hl_sfx = '_highlight' if _hl_regions else ''
    if _hl_regions:
        _draw_highlight_regions(circos, _hl_regions, _hl_opts, circos_config)
        if legend_accumulator is not None:
            _merge_into_legend_accumulator(legend_accumulator, highlight_regions=_highlight_legend_regions(_hl_regions))

    # ver4-4 R7.c - see the note above `circos = Circos.
    # initialize_from_bed(...)` for the full rationale/verification
    # behind this truncation technique. Recorded ONCE, after every
    # loop-invariant track has been added and BEFORE the per-model loop
    # below ever calls circos.link() for the first time.
    _base_patch_count = len(circos._patches)

    # Update ID ver4-25, R12: 'combined' layout only actually combines
    # anything when there are 2+ distinct requested orders, at least one
    # >= 3 - otherwise (the overwhelming majority of runs, including
    # EVERY run predating this feature, where `_requested_orders ==
    # [2]` always) this is False and every model falls through to the
    # SAME code path 'per_order' layout takes for k=2 below - so AC12.3
    # (byte-identical output whenever only k=2 is requested) holds
    # regardless of which layout radio button happens to be selected in
    # the GUI.
    _use_combined_layout = (
        higher_order_layout == 'combined'
        and len(_requested_orders) >= 2
        and max(_requested_orders) >= 3
    )

    # Frozen (pickled), link-free, undrawn base figure for the k=2 renders,
    # built lazily on the first k=2 render - see plot()'s docstring.
    _shared_k2 = {'blob': None, 'ok': True}

    for n in range(len(model_selected)):
        model_name = model_selected[n]
        if _only_models is not None and model_name not in _only_models:
            continue

        if _use_combined_layout:
            # ================================================================
            # Update ID ver4-25, R12: 'combined' layout - ONE render per
            # model, spanning every requested order together. k=2's own
            # chords and every k>=3 hub glyph share ONE colour scheme here
            # (Okabe-Ito, keyed by k) instead of the usual red/blue
            # intra-/inter-chromosome split - intra-/inter- is instead
            # carried by LINESTYLE (solid/dashed), per the blueprint's own
            # spec.
            # ================================================================
            del circos._patches[_base_patch_count:]
            _combined_has_data = False
            _combined_has_intra = False
            _combined_has_inter = False
            _combined_orders_drawn = set()

            if 2 in _requested_orders and interactions_original.shape[0] != 0:
                interactions = interactions_original[interactions_original['model'] == model_name].dropna().reset_index(drop=True)
                if interactions.shape[0] != 0:
                    link_alpha_min = circos_config.get('link_alpha_min', 0.15)
                    link_lw = 1.5
                    if _hl_regions:
                        # colour already encodes k here, so highlighting only emphasises / fades
                        interactions, _link_lws, _, _hl_style = _prepare_links_with_highlight(
                            interactions, circos_config, link_lw, _hl_regions, _hl_opts)
                        print(f"[circos_plot] highlight {PHENOTYPE}/{POPULATION}/{model_name} k=2: "
                              f"{_highlight_counts_text(_hl_style, _hl_regions)}")
                    else:
                        interactions = _sort_links(interactions, circos_config)
                        interactions['_alpha'], _link_lws, _ = _link_strength(interactions['value'], circos_config, link_lw)
                    _k2_colour = _okabe_ito_colour_for_order(2)
                    for ii in range(interactions.shape[0]):
                        region1 = (interactions.iloc[ii, 0], interactions.iloc[ii, 1], interactions.iloc[ii, 2])
                        region2 = (interactions.iloc[ii, 3], interactions.iloc[ii, 4], interactions.iloc[ii, 5])
                        _is_intra = interactions.iloc[ii, 0] == interactions.iloc[ii, 3]
                        if _is_intra:
                            _combined_has_intra = True
                        else:
                            _combined_has_inter = True
                        circos.link(region1, region2, lw=float(_link_lws.iloc[ii]), alpha=float(interactions.loc[ii, '_alpha']),
                                    color=_k2_colour, linestyle='solid' if _is_intra else 'dashed')
                    _combined_has_data = True
                    _combined_orders_drawn.add(2)

            _combined_hok_by_order = {}
            for order in _requested_orders:
                if order < 3:
                    continue
                hok_selected = higher_order_ring_frame[
                    (higher_order_ring_frame['model'] == model_name) & (higher_order_ring_frame['order'] == order)
                ] if higher_order_ring_frame.shape[0] else pd.DataFrame()
                if hok_selected.shape[0]:
                    _combined_hok_by_order[order] = hok_selected
                    _combined_has_data = True
                    _combined_orders_drawn.add(order)

            if not _combined_has_data:
                # Nothing at all (no k=2 rows, no k>=3 hyperedges) for this
                # model under ANY requested order - skip entirely, no file
                # written, mirroring 'per_order' layout's own per-(model,
                # order) skip below.
                continue

            _figsize_cfg = circos_config.get('figsize')
            fig = circos.plotfig(figsize=(_figsize_cfg, _figsize_cfg)) if _figsize_cfg else circos.plotfig()
            _ax = fig.axes[0]
            # r=100.0: the outermost ring edge - the one radius every
            # chromosome sector shares in common regardless of which
            # model's own marker-effect ring a hyperedge's individual
            # members happen to sit closest to (unlike a k=2 chord, whose
            # own r1/r2 pycirclize resolves per-sector via `get_lowest_r()`
            # - not applicable here since a single hub glyph's spokes can
            # originate from several different models' rings at once).
            for order, hok_selected in _combined_hok_by_order.items():
                _hl_state = {}
                _has_intra, _has_inter = _draw_higher_order_glyphs(
                    _ax, circos, hok_selected, 100.0,
                    colourise=lambda is_intra, _order=order: _okabe_ito_colour_for_order(_order),
                    # combined layout colours by k, so `link_colour_mode` does not apply
                    circos_config=dict(circos_config, link_colour_mode='chrom'),
                    hl=({'regions': _hl_regions, 'opts': _hl_opts, 'recolour': False} if _hl_regions else None),
                    hl_state=_hl_state,
                )
                if _hl_regions:
                    print(f"[circos_plot] highlight {PHENOTYPE}/{POPULATION}/{model_name} k={order}: "
                          f"{_hl_state.get('n_hit', 0)} of {_hl_state.get('n_total', 0)} hyperedges touch the highlighted region(s)")
                _combined_has_intra = _combined_has_intra or _has_intra
                _combined_has_inter = _combined_has_inter or _has_inter

            _finish_circos_figure(fig, circos_config)

            if legend_accumulator is not None:
                _merge_into_legend_accumulator(
                    legend_accumulator,
                    has_inter_chr_link=_combined_has_inter,
                    has_intra_chr_link=_combined_has_intra,
                    combined_orders_used=_combined_orders_drawn,
                )

            _phenotype_fs = _sanitize_path_component(PHENOTYPE)
            _population_fs = _sanitize_path_component(POPULATION)
            _model_fs = _sanitize_path_component(model_name)
            _k_sorted = sorted(_combined_orders_drawn)
            _k_suffix = f"k{_k_sorted[0]}-{_k_sorted[-1]}" if len(_k_sorted) > 1 else f"k{_k_sorted[0]}"
            _plot_path = './Result/' + RESULT_NAME + '/circos_' + _phenotype_fs + '_' + _population_fs + '_interaction_' + _model_fs + '_' + _k_suffix + _hl_sfx + '.png'
            _ps.save_figure(fig, _plot_path, dpi=plot_dpi, saver=_safe_savefig,
                            extra_formats=_ps.extra_formats_of(circos_config))
            plt.close(fig)
            continue  # 'combined' layout fully handled this model

        # ====================================================================
        # 'per_order' layout (default, and every pre-ver4-25 caller's only
        # option) - one render per (model, order), k=2 keeping its
        # historical, unchanged filename and code path.
        # ====================================================================
        for order in _requested_orders:
            # Remove any chord links/hub patches a PREVIOUS iteration of
            # this SAME loop added, so this iteration starts from the same
            # clean (no-link) state every time - a no-op on the very first
            # iteration, since nothing has been added past
            # _base_patch_count yet.
            del circos._patches[_base_patch_count:]

            if order == 2:
                # ============================================================
                # EXACTLY today's historical k=2 code path, byte-for-byte
                # unchanged (AC12.3) - see git history / this function's
                # own pre-ver4-25 form for provenance. `_finish_circos_
                # figure()` (extracted, see its own docstring) replaces
                # what used to be this block's own inline "extra height /
                # unit label" code with an IDENTICAL computation - a
                # source-level refactor only, never a behavioural one.
                # ============================================================
                has_inter_chr_link_for_legend = False
                has_intra_chr_link_for_legend = False
                _hl_style = None
                _hl_hit_intra = _hl_hit_inter = _hl_dimmed = False
                if interactions_original.shape[0] != 0:
                    interactions = interactions_original[interactions_original['model']==model_selected[n]].dropna().reset_index(drop=True)
                    # Requirement: link OPACITY (not width) encodes each link's
                    # relative importance ('value') - a higher value means a more
                    # solid/opaque (visually "thicker-looking") link, while every
                    # link is drawn at the same, fixed line width. 'value' itself
                    # is normalized to sum to 1 across whatever's currently
                    # displayed (see the `value / value.sum()` step above), so its
                    # absolute scale shifts with how many links are shown - using
                    # it as alpha directly (or via a fixed multiplier, as the old
                    # width-based version did) would make the same link look very
                    # different across plots with different link counts. Instead,
                    # min-max normalize WITHIN this plot's own displayed link set,
                    # into [link_alpha_min, 1.0] - the strongest link on any given
                    # plot is always fully opaque, the weakest is always still at
                    # least link_alpha_min visible (never fully invisible), and
                    # everything in between scales smoothly - regardless of how
                    # many links happen to be shown.
                    link_alpha_min = circos_config.get('link_alpha_min', 0.15)
                    # Fixed for every link - not user-configurable, since only
                    # opacity (above) is meant to vary by strength; a separate
                    # width control would just uniformly scale every line at once
                    # (no diagnostic value) and was removed from the GUI.
                    link_lw = 1.5
                    if interactions.shape[0] != 0:
                        if _hl_regions:
                            # Region highlighting: non-highlighted links first (faded, or
                            # dropped in mode 'only'), highlighted ones last = on top.
                            interactions, _link_lws, _link_s, _hl_style = _prepare_links_with_highlight(
                                interactions, circos_config, link_lw, _hl_regions, _hl_opts)
                            print(f"[circos_plot] highlight {PHENOTYPE}/{POPULATION}/{model_selected[n]}: "
                                  f"{_highlight_counts_text(_hl_style, _hl_regions)}")
                            if _hl_style['n_hit'] == 0:
                                print(f"[circos_plot] highlight {PHENOTYPE}/{POPULATION}/{model_selected[n]}: no link among the "
                                      f"{_hl_style['n_total']} displayed touches the requested region(s) - raise the number of "
                                      f"interactions shown, or widen the region.")
                        else:
                            # ver4-31 R11: `link_sort` / `link_scale` / `link_lw_range`
                            # (all absent by default = exactly the historical
                            # min-max alpha, fixed width, file order) via the
                            # shared `_link_strength()`.
                            interactions = _sort_links(interactions, circos_config)
                            interactions['_alpha'], _link_lws, _link_s = _link_strength(
                                interactions['value'], circos_config, link_lw)
                    _link_mode = _link_options(circos_config)['colour_mode']
                    for ii in range(interactions.shape[0]):
                        region1 = (interactions.iloc[ii,0], interactions.iloc[ii,1], interactions.iloc[ii,2])
                        region2 = (interactions.iloc[ii,3], interactions.iloc[ii,4], interactions.iloc[ii,5])
                        _intra = interactions.iloc[ii,0] == interactions.iloc[ii,3]
                        if _hl_style is not None and _hl_style['hit'][ii]:
                            # highlighted: the region's colour; line style (not colour) tells
                            # within- from between-chromosome links.
                            if _intra:
                                _hl_hit_intra = True
                            else:
                                _hl_hit_inter = True
                            circos.link(region1, region2, lw=float(_link_lws.iloc[ii]), alpha=float(interactions.loc[ii,'_alpha']),
                                        color=_hl_regions[int(_hl_style['region'][ii])]['colour'],
                                        linestyle='solid' if _intra else _hl.HIGHLIGHT_DASH)
                            continue
                        if _hl_style is not None and _hl_opts['dim'] == 'grey':
                            _hl_dimmed = True
                            circos.link(region1, region2, lw=float(_link_lws.iloc[ii]), alpha=float(interactions.loc[ii,'_alpha']),
                                        color=_hl.DIM_GREY)
                            continue
                        if not _intra:   #within chromosome or between chromosome
                            has_inter_chr_link_for_legend = True
                        else:
                            has_intra_chr_link_for_legend = True
                        colour = _link_colour(_link_mode, _intra, _link_s.iloc[ii])
                        _extra = {'linestyle': 'solid' if _intra else 'dashed'} if _link_mode == 'value' else {}
                        circos.link(region1, region2, lw=float(_link_lws.iloc[ii]), alpha=float(interactions.loc[ii,'_alpha']),
                                    color=colour, **_extra)

                # Store the circos plot
                # Update ID ver4-6, R2 (blueprint §3.4/S3): `circos_config.get(
                # 'figsize')` is absent (None) for every pre-ver4-6 config, in
                # which case `circos.plotfig()` is called exactly as before this
                # option existed (pycirclize's own (8, 8)-inch default) - I11,
                # AC2.5. A configured figsize (e.g. widened for a many-ring plot)
                # is forwarded as a square `(f, f)` tuple, matching pycirclize's
                # own `figsize: tuple[float, float]` parameter.
                fig = None
                _k2_link_artists = []
                if _shared_k2['ok']:
                    if _shared_k2['blob'] is None:
                        # Build the link-free base ONCE: temporarily set this
                        # model's freshly-added link patches aside so plotfig()
                        # sees only the loop-invariant patches, then freeze the
                        # still-UNDRAWN figure as a pickle blob. (The figure is
                        # created with tight_layout=True, which re-runs on every
                        # draw and is not idempotent, so re-saving ONE figure
                        # object would yield slightly different image sizes for
                        # the 2nd+ model - cloning a pristine, undrawn copy per
                        # model reproduces the one-draw-per-figure result of
                        # the original per-model plotfig() exactly.)
                        _pending_links = circos._patches[_base_patch_count:]
                        del circos._patches[_base_patch_count:]
                        _figsize_cfg = circos_config.get('figsize')
                        _base_fig = circos.plotfig(figsize=(_figsize_cfg, _figsize_cfg)) if _figsize_cfg else circos.plotfig()
                        _finish_circos_figure(_base_fig, circos_config)
                        circos._patches.extend(_pending_links)
                        try:
                            _shared_k2['blob'] = pickle.dumps(_base_fig, protocol=pickle.HIGHEST_PROTOCOL)
                        except Exception as _exc:
                            print(f"[circos_plot] NOTE: could not freeze the base figure for reuse ({_exc!r}); "
                                  f"falling back to one full plotfig() per model (slower, same output).")
                            _shared_k2['ok'] = False
                        plt.close(_base_fig)
                    if _shared_k2['ok']:
                        fig = pickle.loads(_shared_k2['blob'])
                        _k2_link_artists = _attach_link_patches(fig.axes[0], circos._patches[_base_patch_count:])
                if fig is None:
                    # Fallback / historical path: one full plotfig() per model.
                    _figsize_cfg = circos_config.get('figsize')
                    if _figsize_cfg:
                        fig = circos.plotfig(figsize=(_figsize_cfg, _figsize_cfg))
                    else:
                        fig = circos.plotfig()
                    _finish_circos_figure(fig, circos_config)

                # Requirement 13 (correction - separate legend file): an earlier
                # version tried to fit the legend INTO this same image, widening
                # the figure and shrinking the polar axes' own reported bounding
                # box to make room - this looked correct in isolated testing, but
                # broke down on real plots: chromosome NAME labels are drawn well
                # outside the polar axes' own nominal r-max (see the sector.text(
                # ..., r=108, ...) call above - deliberately larger than the
                # marker-effect rings' own r=100 so labels sit clearly outside
                # them), so matplotlib's own ax.get_position() (a plain rectangle)
                # never actually captured where those labels really render on the
                # page - repositioning based on it could shrink the DECLARED axes
                # bounds without preventing labels drawn OUTSIDE those bounds from
                # still overlapping the legend, and (for a genuinely large,
                # densely-labelled plot) the plot itself could still collide with
                # a tall legend regardless of how the two were arranged on one
                # shared canvas.
                #
                # ver4-5 (single session-wide legend, supersedes the earlier
                # "separate PNG per plot" correction described above): rather
                # than building this render's own legend handles and saving them
                # to a dedicated '<...>_legend.png' right here, only the RAW
                # ingredients that would have gone into that legend - which
                # link-type colours actually appear on THIS render - are folded
                # into the shared, session-wide `legend_accumulator` (colours/
                # gene-colours were already merged in above, before this n-loop,
                # since they don't vary with `n`). circos_plot() itself builds
                # and saves the ONE combined legend, covering every phenotype/
                # population/model plotted during its call, after its own loop
                # over all of them has finished - see
                # `_merge_into_legend_accumulator()` and the end of
                # circos_plot(). The circos plot image itself is completely
                # unaffected: `fig = circos.plotfig()` and the unit label
                # applied by `_finish_circos_figure()` above are unchanged from
                # before the legend feature existed, exactly as when each
                # render still saved its own legend file.
                if legend_accumulator is not None:
                    _merge_into_legend_accumulator(
                        legend_accumulator,
                        has_inter_chr_link=has_inter_chr_link_for_legend,
                        has_intra_chr_link=has_intra_chr_link_for_legend,
                    )
                    if _hl_regions:
                        _merge_into_legend_accumulator(
                            legend_accumulator, highlight_intra=_hl_hit_intra,
                            highlight_inter=_hl_hit_inter, highlight_dimmed=_hl_dimmed)
                # Bugfix (see module-level note above `_sanitize_path_component`):
                # PHENOTYPE/POPULATION/model_selected[n] are free-text values that
                # were never validated as filename-safe anywhere upstream - sanitise
                # them here, immediately before they become part of a saved path,
                # and save through `_safe_savefig()` rather than `fig.savefig()`
                # directly.
                _phenotype_fs = _sanitize_path_component(PHENOTYPE)
                _population_fs = _sanitize_path_component(POPULATION)
                if model_selected[n] == 'not_returned':
                    _plot_path = './Result/'+RESULT_NAME+'/circos_'+_phenotype_fs+'_'+_population_fs+_hl_sfx+'.png'
                    _ps.save_figure(fig, _plot_path, dpi=plot_dpi, saver=_safe_savefig,
                            extra_formats=_ps.extra_formats_of(circos_config))
                else:
                    _model_fs = _sanitize_path_component(model_selected[n])
                    _plot_path = './Result/'+RESULT_NAME+'/circos_'+_phenotype_fs+'_'+_population_fs+'_interaction_'+_model_fs+_hl_sfx+'.png'
                    _ps.save_figure(fig, _plot_path, dpi=plot_dpi, saver=_safe_savefig,
                            extra_formats=_ps.extra_formats_of(circos_config))
                    if _hl_style is not None:
                        _write_highlight_links_table(_plot_path, interactions, _hl_style, _hl_regions)
                plt.close(fig)
                # ============================================================
                # end unchanged k=2 path
                # ============================================================
                continue

            # ----------------------------------------------------------------
            # Update ID ver4-25, R12: order >= 3 under 'per_order' layout -
            # one hub-and-spoke-only render, its own new filename suffix.
            # Skipped ENTIRELY (no file written) whenever this specific
            # (model, order) combination has no surviving hyperedge - unlike
            # k=2 above, there is no historical "always render, even if
            # empty" precedent to preserve for a k that never existed before
            # this feature.
            # ----------------------------------------------------------------
            if model_name == 'not_returned':
                continue  # the sentinel only ever applies to the (empty) k=2 case
            hok_selected = higher_order_ring_frame[
                (higher_order_ring_frame['model'] == model_name) & (higher_order_ring_frame['order'] == order)
            ] if higher_order_ring_frame.shape[0] else pd.DataFrame()
            if hok_selected.shape[0] == 0:
                continue

            _figsize_cfg = circos_config.get('figsize')
            fig = circos.plotfig(figsize=(_figsize_cfg, _figsize_cfg)) if _figsize_cfg else circos.plotfig()
            _ax = fig.axes[0]
            # 'per_order' layout keeps the SAME red/blue intra-/inter-
            # chromosome convention every k=2 chord already uses (blueprint:
            # "existing intra-chromosome ... / inter-chromosome palette") -
            # only 'combined' layout colours by k instead (see above).
            _hl_state = {}
            _has_intra, _has_inter = _draw_higher_order_glyphs(
                _ax, circos, hok_selected, 100.0,
                colourise=lambda is_intra: 'red' if is_intra else 'blue',
                circos_config=circos_config,
                hl=({'regions': _hl_regions, 'opts': _hl_opts, 'recolour': True} if _hl_regions else None),
                hl_state=_hl_state,
            )
            _finish_circos_figure(fig, circos_config)
            if _hl_regions:
                print(f"[circos_plot] highlight {PHENOTYPE}/{POPULATION}/{model_name} k={order}: "
                      f"{_hl_state.get('n_hit', 0)} of {_hl_state.get('n_total', 0)} hyperedges touch the highlighted region(s)")

            if legend_accumulator is not None:
                _merge_into_legend_accumulator(
                    legend_accumulator,
                    has_inter_chr_link=_has_inter,
                    has_intra_chr_link=_has_intra,
                    has_hub_glyph=True,
                )
                if _hl_regions:
                    _merge_into_legend_accumulator(
                        legend_accumulator, highlight_intra=_hl_state.get('hit_intra'),
                        highlight_inter=_hl_state.get('hit_inter'), highlight_dimmed=_hl_state.get('dimmed'))

            _phenotype_fs = _sanitize_path_component(PHENOTYPE)
            _population_fs = _sanitize_path_component(POPULATION)
            _model_fs = _sanitize_path_component(model_name)
            _plot_path = './Result/'+RESULT_NAME+'/circos_'+_phenotype_fs+'_'+_population_fs+'_interaction_'+_model_fs+f'_k{order}'+_hl_sfx+'.png'
            _ps.save_figure(fig, _plot_path, dpi=plot_dpi, saver=_safe_savefig,
                            extra_formats=_ps.extra_formats_of(circos_config))
            plt.close(fig)


def _clamp_region_to_chromosome(df, mask, chrom_start, chrom_end):
    """Requirement (bugfix - a widened marker/gene position could come out
    excessively large or small, not matching a simple GENE_ADJUST/
    END_ADJUST addition or subtraction): every call site that widens a
    region and then clamps it to its own chromosome's bounds used to
    clamp 'start' only from BELOW (raise it if under chrom_start) and
    'end' only from ABOVE (lower it if over chrom_end) - correct on its
    own for the ordinary case, but it silently left 'start' completely
    unclamped whenever it was ALSO too large (above chrom_end), and
    likewise left 'end' unclamped whenever it was ALSO too small (below
    chrom_start).

    Confirmed against real data this actually happens: a gene's own
    reported position can already extend past its chromosome's stated
    end even BEFORE any widening is applied at all (e.g. a coordinate
    system mismatch between a gene_info file and its chrom_info file) -
    widening such a position pushes it even further out, 'end' gets
    correctly clamped down to chrom_end by the old one-sided check, but
    'start' - despite being even larger than the now-clamped 'end' -
    never gets touched, producing a nonsensical start > end region
    that's neither the widened value NOR a sensible clamp of it.

    The fix: clamp 'start' and 'end' fully independently, each to the
    complete [chrom_start, chrom_end] range (both directions, not just
    one) - this is provably safe (never introduces a start > end that
    wasn't already true beforehand): clamping is a monotonic operation,
    so if start <= end before clamping (always true here - a widening
    step only ever moves start down and end up from an originally valid
    start <= end, by the same amount in each direction), then
    clamp(start) <= clamp(end) afterward too, for any shared [lo, hi]
    bounds - regardless of by how much either one originally overshot.

    Modifies df in place (matching every call site's own prior in-place
    .loc[...] = ... usage); returns nothing."""
    df.loc[mask & (df['start'] < chrom_start), 'start'] = chrom_start
    df.loc[mask & (df['start'] > chrom_end), 'start'] = chrom_end
    df.loc[mask & (df['end'] > chrom_end), 'end'] = chrom_end
    df.loc[mask & (df['end'] < chrom_start), 'end'] = chrom_start


def _warn_if_raw_position_exceeds_chromosome(df, chromosome, file_description, result_name_for_log=None):
    """Requirement (diagnostic - companion to the clamping fix above):
    the clamp fix guarantees a widened region can never come out
    nonsensical (start > end) any more, but a region whose RAW,
    un-widened position ALREADY exceeded its own chromosome's stated
    bounds - confirmed to happen in practice, see
    _clamp_region_to_chromosome()'s own docstring - still gets silently
    squashed down to a single point at that chromosome's very edge,
    which is technically valid but doesn't reflect where the position
    actually was. Surfaced here as an explicit, specific warning (the
    same pattern as the chromosome-name-mismatch warning above) rather
    than left for a person to notice only as an oddly-clustered handful
    of markers/genes sitting right at a chromosome's edge with no
    obvious explanation. Call this BEFORE any widening is applied, on
    the raw start/end columns."""
    _n_bad = 0
    _examples = []
    for _chrom_name, _group in df.groupby('chromosome'):
        _chrom_row = chromosome.loc[chromosome['chromosome'] == _chrom_name]
        if _chrom_row.shape[0] == 0:
            continue
        _c_start = int(_chrom_row['start'].values[0])
        _c_end = int(_chrom_row['end'].values[0])
        _out_of_range = _group[(_group['start'] < _c_start) | (_group['start'] > _c_end) |
                                (_group['end'] < _c_start) | (_group['end'] > _c_end)]
        if _out_of_range.shape[0] > 0:
            _n_bad += _out_of_range.shape[0]
            for _, _row in _out_of_range.head(3 - len(_examples) if len(_examples) < 3 else 0).iterrows():
                _examples.append(f"{_chrom_name} {int(_row['start'])}-{int(_row['end'])} (chromosome is {_c_start}-{_c_end})")
    if _n_bad > 0:
        print(
            f"[circos] WARNING: {_n_bad} row(s) in the {file_description} have a position that already "
            f"falls outside their own chromosome's stated start/end - BEFORE any widening is applied. "
            f"These get clamped to sit exactly at their chromosome's edge instead of crashing or coming "
            f"out reversed, but that edge position won't reflect where the row actually is. This usually "
            f"means the {file_description} and the chromosome info file don't agree on that chromosome's "
            f"coordinate system (e.g. a different genome assembly version). Example(s): {_examples}"
        )


def _clean_population_label(pop):
    """Requirement (bugfix - gene ring silently missing for real
    populations, only ever present for 'all'): normalizes a population
    VALUE to a consistent, 'clean' string form, so the SAME population is
    never accidentally treated as two different ones depending on which
    code path it happened to pass through on its way here.

    Concretely: Parallel mode's own assemble() (main_app.py) computes
    population = pd.unique(metric['population']).tolist() from the
    combined Metric_*.csv across every batch - and pandas silently
    promotes an entire int column to float64 the moment even ONE value
    in it is missing/NaN anywhere across those concatenated batches (a
    well-known, common pandas behaviour, not specific to any one
    dataset - e.g. one incomplete/placeholder row in a single batch's
    own Metric_N.csv is enough). Once that happens, EVERY population
    value becomes e.g. 1.0 instead of 1, and str(1.0) == '1.0', not '1' -
    which then fails to match a gene/chromosome file correctly written
    under the 'clean' label '1' (by data_conversion(), from the
    broadcast file's own str-cast population column), or fails
    pop_source's own population-column lookup in plot(). The literal
    string 'all' is never affected by this (it was never a number to
    begin with), which is exactly why this symptom shows up as 'only
    ever works for the all population' - every real, numeric population
    silently fails the same string comparison in the same way.

    Converts a whole-number float (1.0, 2.0, ...), a numpy integer/float
    scalar, AND the string form of a whole-number float ('1.0', '2.0',
    ...) to its clean integer string ('1', '2', ...); anything else
    (already a clean int/str, or a non-numeric string like 'all') is
    returned as plain str(pop) unchanged.

    The string form ('2.0') matters for SCENARIO='between': its
    population labels are built by string concatenation ('1.0->2.0') from
    the phenotype file's population values, so a float-promoted column
    leaves the TEST half as the STRING '2.0' after the '->' split - which
    the float-only check above never normalised, while the population
    list handed in by the caller (raw numeric values, via
    `_clean_population_label()`) WAS normalised to '2'. The two sides of
    the very same comparison then disagreed ('2.0' vs '2'), every model
    was dropped for that population, and plot() died on `MODEL[0]` (list
    index out of range). Every population label - effect/interaction/
    attention tables, the POPULATION list, the chromosome/gene info files
    and the broadcast step - now goes through this ONE function, so they
    can never disagree again."""
    if isinstance(pop, (bool, np.bool_)):
        return str(pop)
    if isinstance(pop, (int, np.integer)):
        return str(int(pop))
    if isinstance(pop, (float, np.floating)):
        return str(int(pop)) if float(pop).is_integer() else str(pop)
    _s = str(pop)
    if re.fullmatch(r'-?\d+\.0+', _s):
        return _s.split('.')[0]
    return _s


def _warn_no_model_data(phenotype, population):
    """Log (never raise) that no model has marker-effect data for this
    (phenotype, population), so its circos plot is skipped - the previous
    behaviour was an `IndexError: list index out of range` from
    `MODEL[0]` inside plot(), which aborted every remaining plot too."""
    print(f"[circos_plot] WARNING: no model has marker-effect data for phenotype={phenotype!r}, "
          f"population={population!r} - this circos plot is skipped. (Usually a population label "
          f"mismatch between the effect table and the population list, or a population that is "
          f"never a test population in this run.)")


def _between_test_population_label(pop):
    """SCENARIO='between': the clean TEST-population label of a combined
    'train->test' (or 'train->valid->test') population value - the last
    '->' part, passed through `_clean_population_label()`. Safe for a value
    with no '->' at all (returned cleaned, unsplit) and for non-string
    values (an int/float population is first turned into its string)."""
    return _clean_population_label(str(pop).split('->')[-1])


def _remap_population_column(series, label_fn):
    """`series.map(label_fn)` computed once per DISTINCT value (the
    population column of a big effect/interaction table has a handful of
    distinct values over millions of rows), returning a plain-string
    Series on the original index."""
    _mapping = {v: label_fn(v) for v in pd.unique(series)}
    return series.map(_mapping).astype(str)


def _broadcast_population_info(path, target_populations, description):
    """Requirement 8: expands a chromosome/gene info file's rows across
    every population circos actually needs, when the person has said (via
    the 'Chromosome/gene lengths are the same for every population'
    checkbox) that they're all identical - lets them provide the
    coordinates ONCE instead of manually duplicating every row once per
    population plus once more for 'all' (which circos_plot() itself
    always adds on top of the real populations - see its own
    `POPULATION = ('all',) + tuple(POPULATION)`).

    Lives here in circos_plot.py (moved from main_app.py) rather than
    duplicated separately in every place that needs it - this exact
    kind of duplication (a near-identical copy living in the Parallel/
    Sequential HEADLESS scripts, run_step2_assemble.py/run_sequential.py,
    never updated when this function or the broadcast feature as a whole
    changed in main_app.py) was a real, confirmed cause of the broadcast
    step silently never running at all for headless/HPC runs - both of
    those scripts call circos_plot() directly with the raw, un-broadcast
    cfg['CHROMOSOME_INFO']/cfg['GENE_INFO'] paths, with no broadcast logic
    of their own. A single shared implementation, imported by every
    caller (main_app.py's own GUI code included) from this one place,
    means there is only ever one version of this logic to keep correct
    and in sync - the caller (see run_step2_assemble.py/run_sequential.py/
    main_app.py, all updated to call this the same way) is still
    responsible for checking whether broadcasting was actually requested
    at all (a config flag) and for the SCENARIO='between' population-label
    split beforehand (that needs SCENARIO, which isn't a parameter here).

    Any existing 'population' column is dropped first (its values don't
    matter in this mode - only that a value CAN be written back in,
    correctly, per target population) and a fresh one appended at the
    end, matching where circos_plot.py's own unify_columns_by_position()
    calls expect it (chromosome/gene info files both have 'population' as
    their LAST expected column).

    Writes a new, fully-expanded TEMPORARY CSV - never touches or
    overwrites the person's own original file - and returns its path;
    the caller uses this path for the rest of the circos-plotting
    pipeline, exactly as if the duplication had been done by hand.
    Raises ValueError (surfaced to the person, not silently swallowed)
    if the source file can't be read at all, since silently falling back
    to the original (non-broadcast) file here would produce a confusing
    'file not found'/schema error much later instead.
    """
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Could not read {description} at {path!r} to broadcast it across populations: {exc}")

    if 'population' in df.columns:
        df = df.drop(columns=['population'])

    frames = []
    for pop in target_populations:
        _block = df.copy()
        _block['population'] = pop
        frames.append(_block)
    expanded = pd.concat(frames, ignore_index=True)

    tmp_path = os.path.join(tempfile.gettempdir(), f"easigp_broadcast_{description}_{uuid.uuid4().hex[:8]}.csv")
    expanded.to_csv(tmp_path, index=False)
    return tmp_path


def _init_circos_plot_worker(ctx):
    """Spawned-worker initializer for circos_plot()'s optional multi-process
    (phenotype, population) fan-out (``CIRCOS_PLOT_WORKERS`` / `n_workers`
    below). Stashes the read-only, per-run context every worker needs
    (the precomputed effect/interaction/attention lookup tables, config
    dicts, file paths, etc. that circos_plot() itself computes ONCE before
    its loop) in a module-level global exactly ONCE per worker process,
    via ``ProcessPoolExecutor(initializer=..., initargs=(ctx,))`` - rather
    than re-pickling that same, potentially large, context onto every one
    of what can be dozens-to-hundreds of individual (phenotype, population)
    tasks the way passing it as a per-call argument would.

    A `spawn`-started worker (see `_MP_CONTEXT`) is a fresh interpreter
    that re-imports this module but does NOT share the parent's memory, so
    this global is genuinely private to the one worker process that set
    it - never shared or raced between workers, and never visible back in
    the parent process either (see `_render_one_circos_plot()`'s own
    return value for how a worker's findings get back to the parent).

    Also forces the non-interactive 'Agg' matplotlib backend inside the
    worker: a spawned process has no display to attach to, and leaving
    backend selection to whatever matplotlib would otherwise auto-detect
    risks it trying (and failing, or behaving inconsistently across
    platforms) to initialise a GUI toolkit that was never actually needed
    for saving PNGs to disk.
    """
    global _CIRCOS_WORKER_CTX
    _CIRCOS_WORKER_CTX = ctx
    import matplotlib
    matplotlib.use('Agg', force=True)


def _render_one_circos_plot(phenotype_value, population_value):
    """One (phenotype, population) unit of work for circos_plot()'s
    parallel fan-out - performs exactly what one iteration of that
    function's own serial loop does (`quantile_conversion()` ->
    `interaction()` -> `plot()`), just reading its shared, read-only
    inputs from `_CIRCOS_WORKER_CTX` (set once per worker by
    `_init_circos_plot_worker()`) instead of from the enclosing
    function's closure, since a `spawn`-started worker process cannot see
    the parent's closure state at all - only what was explicitly passed
    through `initargs`.

    Returns this ONE plot's own LOCAL legend accumulator, rather than
    mutating circos_plot()'s own session-wide `_legend_accumulator` the
    serial loop mutates directly - a separate process can never see or
    modify the parent's Python objects, so there is nothing for it to
    mutate here; the parent instead merges every worker's returned
    accumulator into its own after all tasks complete (see
    `circos_plot()`'s own merge loop). Merge order does not matter:
    `_merge_into_legend_accumulator()` is a pure, commutative,
    associative union/OR over sets and booleans, so the combined result
    is identical regardless of which (phenotype, population) task
    happens to finish first.
    """
    ctx = _CIRCOS_WORKER_CTX
    model_for_plot = quantile_conversion(
        ctx['effect_grouped_all'], ctx['effect_grouped_pop'], ctx['marker_info'], ctx['chrom_info'],
        phenotype_value, ctx['MODEL'], ctx['end_adjust'], population_value, ctx['WINDOW'],
        ctx['RESULT_NAME'], ctx['ASCENDING'],
        ensemble_hue_rule=_ensemble_hue_rule(ctx['circos_config']),
        label_cfg=ctx['circos_config'],
    )
    interaction_selected = interaction(
        ctx['interaction_groups'], ctx['interaction_model_order'], ctx['marker_info'], phenotype_value,
        ctx['circos_config'], population_value, ctx['RESULT_NAME'], ctx['attention_groups'], ctx['attention_model_order'],
    )
    # Update ID ver4-25, R12: the k>=3 analog of `interaction_selected`
    # above - see `higher_order_interaction()`'s own docstring. Empty
    # (correctly-columned) whenever `ctx['higher_order_groups']` is empty
    # - i.e. every run this feature is not selected for.
    higher_order_selected = higher_order_interaction(
        ctx['higher_order_groups'], ctx['higher_order_model_order'], ctx['marker_info'], phenotype_value,
        ctx['circos_config'], population_value, ctx['RESULT_NAME'], ctx['interaction_orders'],
    )
    local_legend_accumulator = _new_legend_accumulator()
    plot(
        interaction_selected, ctx['chrom_info'], ctx['gene_info'], ctx['pop_source'], phenotype_value, model_for_plot,
        ctx['circos_config'], ctx['CYTOBAND_COLORMAP'], population_value, ctx['RESULT_NAME'],
        plot_dpi=ctx['plot_dpi'], legend_accumulator=local_legend_accumulator,
        higher_order_ring_frame=higher_order_selected, interaction_orders=ctx['interaction_orders'],
        higher_order_layout=ctx['higher_order_layout'],
    )
    return local_legend_accumulator


def _prepare_one_circos_unit(phenotype_value, population_value):
    """Stage 1 of circos_plot()'s parallel fan-out for one (phenotype,
    population): the cheap, ordering-sensitive preparation that used to sit
    at the top of each serial-loop iteration - writes this unit's
    marker-effect `.tsv` files (`quantile_conversion()`) and resolves its
    link tables (`interaction()`/`higher_order_interaction()`). Returns
    those three small results so the parent can slice the unit's models
    into render chunks. Runs exactly once per unit, so no two workers ever
    write the same `.tsv`."""
    ctx = _CIRCOS_WORKER_CTX
    model_for_plot = quantile_conversion(
        ctx['effect_grouped_all'], ctx['effect_grouped_pop'], ctx['marker_info'], ctx['chrom_info'],
        phenotype_value, ctx['MODEL'], ctx['end_adjust'], population_value, ctx['WINDOW'],
        ctx['RESULT_NAME'], ctx['ASCENDING'],
        ensemble_hue_rule=_ensemble_hue_rule(ctx['circos_config']),
        label_cfg=ctx['circos_config'],
    )
    interaction_selected = interaction(
        ctx['interaction_groups'], ctx['interaction_model_order'], ctx['marker_info'], phenotype_value,
        ctx['circos_config'], population_value, ctx['RESULT_NAME'], ctx['attention_groups'], ctx['attention_model_order'],
    )
    higher_order_selected = higher_order_interaction(
        ctx['higher_order_groups'], ctx['higher_order_model_order'], ctx['marker_info'], phenotype_value,
        ctx['circos_config'], population_value, ctx['RESULT_NAME'], ctx['interaction_orders'],
    )
    return model_for_plot, interaction_selected, higher_order_selected


def _render_one_circos_chunk(phenotype_value, population_value, model_for_plot, interaction_selected,
                             higher_order_selected, only_models):
    """Stage 2 of circos_plot()'s parallel fan-out: render the figures for
    `only_models` (a slice of one (phenotype, population)'s models) from
    the tables `_prepare_one_circos_unit()` produced. Splitting a unit's
    models across workers is what lets a run with few phenotypes/populations
    but many interaction-emitting models still use every core. Returns this
    chunk's LOCAL legend accumulator (merged by the parent - see
    `_render_one_circos_plot()`'s docstring for why it is returned rather
    than shared)."""
    ctx = _CIRCOS_WORKER_CTX
    local_legend_accumulator = _new_legend_accumulator()
    plot(
        interaction_selected, ctx['chrom_info'], ctx['gene_info'], ctx['pop_source'], phenotype_value, model_for_plot,
        ctx['circos_config'], ctx['CYTOBAND_COLORMAP'], population_value, ctx['RESULT_NAME'],
        plot_dpi=ctx['plot_dpi'], legend_accumulator=local_legend_accumulator,
        higher_order_ring_frame=higher_order_selected, interaction_orders=ctx['interaction_orders'],
        higher_order_layout=ctx['higher_order_layout'], only_models=only_models,
    )
    return local_legend_accumulator


def _available_cpu_count():
    """CPUs this process may actually run on: the scheduler's affinity mask
    when the OS exposes one (PBS/Slurm cgroup-limited jobs - where
    ``os.cpu_count()`` reports the WHOLE node and would badly oversubscribe
    the allocation), else ``os.cpu_count()``."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _resolve_circos_plot_workers(requested, n_jobs=None):
    """Effective worker-process count for circos_plot()'s optional parallel
    fan-out - daemon-safe, mirroring
    `pipeline_utils.dataloader_num_workers()`/`nested_safe_n_jobs()`'s own
    fresh-per-call daemon check (never cached, since the correct answer
    depends on whichever process is actually about to call
    `ProcessPoolExecutor()`, not on whatever process originally read the
    config).

    Where the count comes from (fix: ``N_JOBS`` was previously ignored by
    the circos step entirely, so a run configured with ``N_JOBS=6`` still
    rendered every circos figure serially):

      1. ``requested`` (``CIRCOS_PLOT_WORKERS``) when it is given - an
         explicit circos-specific setting always wins; ``<= 1`` means
         serial, exactly as before.
      2. otherwise ``n_jobs`` (the run's ``N_JOBS``), using scikit-learn's
         convention: a positive value is a worker count, ``-1`` means "every
         CPU available to this process", ``0``/``None`` means serial.
      3. otherwise ``1`` - fully serial, today's behaviour for a config
         that has neither key.

    Returns ``1`` unconditionally when the CALLING process is itself an
    already-daemonic `multiprocessing` worker (e.g. circos_plot() invoked
    from inside an intra_batch_parallel/intra_task_parallel worker) - Python
    forbids a daemonic process from spawning children at all.

    The result is capped at the CPUs actually available to this process
    (`_available_cpu_count()`): more worker processes than cores cannot make
    rendering faster and only adds process-startup overhead.
    """
    try:
        if multiprocessing.current_process().daemon:
            return 1
    except Exception:
        pass
    _ncpu = _available_cpu_count()
    if requested is not None and requested != '':
        try:
            _requested = int(requested)
        except (TypeError, ValueError):
            return 1
        if _requested <= 1:
            return 1
        return max(1, min(_requested, _ncpu))
    if n_jobs is None:
        return 1
    try:
        _n_jobs = int(n_jobs)
    except (TypeError, ValueError):
        return 1
    if _n_jobs == -1 or _n_jobs < -1:
        # sklearn convention: -1 = all CPUs, -2 = all but one, ...
        _n_jobs = max(1, _ncpu + 1 + _n_jobs)
    if _n_jobs <= 1:
        return 1
    return max(1, min(_n_jobs, _ncpu))


def circos_plot(effect, interactions, marker_info, chrom_info, gene_info, POPULATION, PHENOTYPE, circos_config, end_adjust, WINDOW, CYTOBAND_COLORMAP, RESULT_NAME, attention, SCENARIO, ASCENDING, gene_adjust=0, plot_dpi=600, n_workers=1,
                interactions_hok=None, interaction_orders=(2,), higher_order_layout='per_order',
                n_jobs=None):
    """Update ID ver4-25, R12 (new, additive parameters - I11/AC12.3):
    `interactions_hok` - Interaction_higher_order.csv's own table (or its
    batch-merged/grouped equivalent - `ResultSet.interactions_hok()`/
    `interactions_hok_grouped()`, mirroring exactly how `interactions`
    itself is handed in either raw or pre-aggregated), `None`/empty (the
    default, and what every caller that predates this feature passes
    implicitly) meaning there is no k>=3 data anywhere in this run.
    `interaction_orders` - which k values to actually draw, default
    `(2,)` (today's only order, unconditionally). `higher_order_layout` -
    `'per_order'` (default) or `'combined'` - see `plot()`'s own
    docstring for what each means. Every one of these three is a no-op
    (byte-identical output, AC12.3) whenever `interactions_hok` is empty
    and/or `interaction_orders == [2]`, which is every run this feature
    is not selected for.

    Parallelism (`n_workers` / `n_jobs`): `n_workers` is the circos-specific
    setting (``CIRCOS_PLOT_WORKERS``); `n_jobs` is the run's general
    ``N_JOBS``. An explicit `n_workers` wins; when it is ``None`` the worker
    count is taken from `n_jobs` (previously ``N_JOBS`` was ignored here and
    circos always rendered serially). See `_resolve_circos_plot_workers()`.
    The work is split per (phenotype, population) AND per model, so a run
    with few traits/populations but many interaction-emitting models still
    keeps every worker busy."""
    if interactions_hok is None:
        interactions_hok = pd.DataFrame()

    # ver4-4 R7.b - start every circos_plot() call from a clean
    # _load_combined_marker_info() cache: this run's own marker_info.csv/
    # gene-coordinate side files are the only things that cache should
    # ever ever hand back, never a stale entry left over from an earlier
    # RESULT_NAME/run inside the same long-lived process (e.g. main_app.py's
    # in-process 'Option B: run now' path, which can call circos_plot()
    # for more than one RESULT_NAME across a single Streamlit session).
    _MARKER_INFO_CACHE.clear()

    # Update ID ver4-6, R2 (blueprint §3.4/S3, I11): log every R2 config
    # key's RESOLVED value unconditionally, on every run, naming "absent
    # from config, using legacy default" whenever the key itself is
    # missing - the codebase's own "log the negative/resolved state
    # unconditionally" convention (architecture document §17), applied
    # here so a person looking at a run's log can always tell, without
    # opening the config JSON, whether this render used the old, fixed
    # geometry or the new, fitted one.
    def _resolved_note(key):
        return 'from config' if key in circos_config else 'absent from config, using legacy default'
    print(f"[circos_plot] R2 geometry config resolved for RESULT_NAME={RESULT_NAME!r}: "
          f"ring_layout={circos_config.get('ring_layout', 'legacy')!r} ({_resolved_note('ring_layout')}), "
          f"ring_label_fit={circos_config.get('ring_label_fit', 'off')!r} ({_resolved_note('ring_label_fit')}), "
          f"ring_label_size={circos_config.get('ring_label_size', 8.0)!r} ({_resolved_note('ring_label_size')}), "
          f"ring_label_max_chars={circos_config.get('ring_label_max_chars', 0)!r} "
          f"({_resolved_note('ring_label_max_chars')}), "
          f"figsize={circos_config.get('figsize') or '8.0 (pycirclize default)'} "
          f"({_resolved_note('figsize') if circos_config.get('figsize') else 'absent from config, using pycirclize default'}).")

    # Region highlighting: log the resolved state unconditionally (same convention as the R2 line
    # above), so a run's log always says whether this render highlights anything and why not.
    _hl_cfg = _hl.highlight_options(circos_config)
    if _hl_cfg['regions']:
        print(f"[circos_plot] highlight: {len(_hl_cfg['regions'])} region(s) "
              f"({'; '.join(r['chrom'] + ('' if r['start'] is None and r['end'] is None else ':' + _hl.format_position(r['start'] or 0, _hl_cfg['unit']) + '-' + (_hl.format_position(r['end'], _hl_cfg['unit']) if r['end'] is not None else 'end')) for r in _hl_cfg['regions'])}), "
              f"mode={_hl_cfg['mode']!r}, match={_hl_cfg['match']!r}, colour_by={_hl_cfg['colour_by']!r}; "
              f"plots that place at least one region are saved with the suffix '_highlight'.")
    else:
        print("[circos_plot] highlight: no regions configured (absent from config) - plots are not highlighted.")

    # 'Only show links involving the highlighted regions': a selection-stage filter, so it must be
    # known before the top-N% cut below. (None, None) - and circos_config untouched - otherwise.
    _restrict_store, _restrict_hok = _highlight_restrict(circos_config, chrom_info, marker_info, RESULT_NAME)
    if _restrict_hok is not None:
        circos_config = dict(circos_config, _hl_restrict=_restrict_hok)

    pop_source =  data_conversion(chrom_info, gene_info, PHENOTYPE, RESULT_NAME, gene_adjust=gene_adjust)
    
    # Update ID circos-oom-fix: the 'between'-scenario 'train->test' -> 'test' label
    # split for interactions/attention is no longer an in-place rewrite of a
    # (potentially enormous) 'population' column - the table is reduced by
    # `interaction_reduce.PairSumStore`, which applies this label map to its
    # tiny population dictionary and merges colliding labels by summing
    # sum/count before the mean (see `_precompute_interaction_groups()`).
    # (This also stops mutating the caller's own frame.)
    # ver4-75: every label (between: the test half of 'train->test'; within: the
    # label itself) goes through _clean_population_label(), so '2.0'/2/2.0 all
    # become '2' - the same form POPULATION and the effect table use below.
    _between_population_map = (_between_test_population_label if SCENARIO == 'between'
                               else _clean_population_label)
    # Update ID ver4-25, R12: the SAME 'between'-scenario population-label
    # split every other interaction/attention/effect table above already
    # gets - `interactions_hok` needs it for the identical reason (its own
    # 'population' column is otherwise still the combined train->test
    # label, which would never match a gene/chromosome file written under
    # the clean, split label).
    # ver4-75 (bugfix - "model[0] ... out of index" for between-population
    # circos plots): the effect / k>=3 tables' 'population' column and the
    # POPULATION list must end up as the SAME clean string for the same
    # population, whatever type it arrived as (int, float-promoted 2.0, the
    # string '2.0' left over from a float-promoted 'train->test' label, or a
    # plain name). They used to be normalised by DIFFERENT code on each side
    # (`.str.split('->', expand=True).iloc[:, -1]` with no clean-up on the
    # tables; `_clean_population_label()` on the list), so a population whose
    # label was '2.0' on one side and '2' on the other matched no effect row,
    # quantile_conversion() dropped every model for it, and plot() then
    # crashed on `MODEL[0]`. One function now does it for both sides, and
    # the caller's own frames are no longer mutated (shallow copy below).
    _population_label_fn = _between_test_population_label if SCENARIO == 'between' else _clean_population_label
    if interactions_hok.shape[0] != 0:
        interactions_hok = interactions_hok.copy(deep=False)
        interactions_hok['population'] = _remap_population_column(interactions_hok['population'], _population_label_fn)
    if effect.shape[0] != 0:
        effect = effect.copy(deep=False)
        effect['population'] = _remap_population_column(effect['population'], _population_label_fn)

    # POPULATION (this function's own parameter, used for every file name /
    # comparison in quantile_conversion()/interaction()/plot() below, and by
    # the broadcast step in the callers before this function is called) gets
    # the identical treatment: for SCENARIO='between' a combined 'Historical->2014'
    # label is reduced to its test half '2014' (see the long-standing note
    # on the gene ring silently missing for this scenario), and every value
    # is cleaned to its canonical string form. Several combined labels
    # ('A->2014', 'B->2014') reduce to the same test population, so
    # duplicates are dropped (order kept) - a duplicate would otherwise
    # render, and in parallel mode WRITE, the same files twice.
    POPULATION = list(dict.fromkeys(_population_label_fn(p) for p in POPULATION))

    # Sequential mode hands in GP()'s raw population list (every population
    # named in the phenotype file), which can name a population that is never
    # a test population in this run (e.g. every 'X->that' combination
    # excluded) - no effect rows exist for it, so there is nothing to draw.
    # Say so and skip it instead of crashing on an empty model list later.
    if effect.shape[0] != 0:
        _populations_with_effect = set(pd.unique(effect['population']))
        _skipped_populations = [p for p in POPULATION if p not in _populations_with_effect]
        if _skipped_populations:
            print(f"[circos_plot] WARNING: no marker-effect rows for population(s) {_skipped_populations} "
                  f"(populations present in the effect table: {sorted(_populations_with_effect)[:20]}); "
                  f"no circos plot is generated for them.")
            POPULATION = [p for p in POPULATION if p in _populations_with_effect]

    POPULATION = ('all',) + tuple(POPULATION)

    MODEL = pd.unique(effect['model'])

    # ver4-31 R13: the consensus ring's colour keys (consensus0..N) must be in
    # the colour map that plot() AND the session legend read.
    if _consensus_options(circos_config)['on']:
        CYTOBAND_COLORMAP = _cytoband_cmap_with_consensus(CYTOBAND_COLORMAP, len(MODEL))

    # ver4-4 R7.f - compute the abs()'d, grouped-and-averaged marker-effect
    # frames used by quantile_conversion() exactly ONCE here, rather than
    # once per (phenotype, population) pair inside the loop below (neither
    # computation's result actually depends on which specific phenotype/
    # population is being rendered at that moment - only the SELECTION
    # of which precomputed frame to read, and which rows/columns to filter
    # out of it, varies per iteration; that filtering still happens inside
    # quantile_conversion() itself, unchanged).
    #
    # Also FIXES a correctness bug in the same motion: quantile_conversion()
    # used to run `effect.iloc[:,5:] = effect.iloc[:,5:].abs().astype(float)`
    # directly on the CALLER's own `effect` object, mutating it in place -
    # e.g. run_step2_assemble.py's own cached ResultSet frame, already used
    # once by scatter_plot() earlier in the very same call, silently
    # modified before circos_plot() itself was even done reading it.
    # `.copy()` here means every downstream read of the ORIGINAL `effect`
    # argument (there are none left in this function after this point, but
    # a future caller/change might reasonably still hold a reference to
    # it) is never affected by what circos_plot() does internally.
    _effect_abs = effect.copy()
    _effect_abs.iloc[:, 5:] = _effect_abs.iloc[:, 5:].abs().astype(float)
    _effect_abs = _effect_abs.drop('ratio', axis=1)
    _effect_abs['population'] = _effect_abs['population'].astype(str)
    _effect_grouped_all = _effect_abs.iloc[:, 1:].groupby(['phenotype', 'model']).mean()
    _effect_grouped_all = _effect_grouped_all.reset_index(drop=False)
    _effect_grouped_pop = _effect_abs.groupby(['population', 'phenotype', 'model']).mean()
    _effect_grouped_pop = _effect_grouped_pop.reset_index(drop=False)
    _effect_grouped_pop['population'] = _effect_grouped_pop['population'].astype(str)

    # Requirement (performance fix, Update ID ver4-10): apply the exact
    # same "compute once, outside the (PHENOTYPE, POPULATION) loop"
    # treatment ver4-4 R7.f already gave `_effect_grouped_all`/
    # `_effect_grouped_pop` above to the marker-pair interaction/
    # attention tables too - see `_precompute_interaction_groups()`'s own
    # docstring for the full rationale (circos-plot generation "taking
    # forever" once more than one interaction-emitting model - RKHS, RF,
    # SVR, KNN, etc. - is selected together). `interaction()` below now
    # does a cheap dict lookup per (PHENOTYPE[i], POPULATION[j]) call
    # instead of re-scanning the full `interactions`/`attention` table
    # from scratch every time - this one change is what fixes the
    # slowdown identically for all three ways a run's interactions/
    # attention data reaches this function (Sequential, in-process GUI
    # "Option B: run now", and Parallel Step 2 assemble), since all three
    # converge on this same shared `circos_plot()` call.
    # Update ID circos-oom-fix: also performs the top-N%/top-M selection per key
    # (circos_config=...), so `_interaction_groups`/`_attention_groups` hold
    # only the rows that will be drawn - small enough to hand to every worker
    # process of the fan-out below without multiplying memory.
    _interaction_groups, _interaction_model_order = _precompute_interaction_groups(
        interactions, scope_model_order_by_population=True,
        circos_config=circos_config, population_map=_between_population_map, restrict=_restrict_store)
    _attention_groups, _attention_model_order = _precompute_interaction_groups(
        attention, scope_model_order_by_population=False,
        circos_config=circos_config, population_map=_between_population_map, restrict=_restrict_store)
    # Update ID ver4-25, R12: the k>=3 analog of the two precomputes
    # above - see `_precompute_higher_order_groups()`'s own docstring.
    # `{}, {}` whenever `interactions_hok` is empty (every run this
    # feature is not selected for), which makes `higher_order_
    # interaction()` below return its own empty, correctly-columned frame
    # for every (PHENOTYPE, POPULATION) call - i.e. AC12.3 holds all the
    # way through this precompute too.
    _higher_order_groups, _higher_order_model_order = _precompute_higher_order_groups(
        interactions_hok, scope_model_order_by_population=True)

    # ver4-5 (single session-wide legend, replaces one '<...>_legend.png'
    # per circos plot image): one shared accumulator, created here and
    # passed into every plot() call below, collects the union of every
    # colour/link-type actually drawn across the WHOLE (PHENOTYPE x
    # POPULATION) loop - i.e. across every circos plot this one
    # circos_plot() call produces. See `_new_legend_accumulator()`/
    # `_merge_into_legend_accumulator()` and the single
    # `_save_circos_legend()` call after the loop below.
    _legend_accumulator = _new_legend_accumulator()

    # Performance (multi-CPU circos-plot rendering): `_effective_workers`
    # resolves to 1 (the loop below then behaves EXACTLY as before this
    # feature existed - same call order, same shared, mutated-in-place
    # `_legend_accumulator`) for every config/caller that predates
    # `n_workers`/`CIRCOS_PLOT_WORKERS`, or that explicitly asks for 1, or
    # that is itself already running inside a daemonic worker process -
    # see `_resolve_circos_plot_workers()`'s own docstring. Only a
    # request for 2+ workers from a non-daemonic process takes the
    # parallel branch.
    _effective_workers = _resolve_circos_plot_workers(n_workers, n_jobs)

    if _effective_workers <= 1:
        for i in range(len(PHENOTYPE)):
            for j in range(len(POPULATION)):
                # ver4-4 R7 latent-bug fix (blueprint §2.7.2, found while
                # implementing R7.a): quantile_conversion() returns MODEL minus
                # any model with no data for THIS (phenotype, population) pair
                # (its own REMOVE list - see that function's docstring/L586,
                # L753). The ORIGINAL code here rebound the shared,
                # loop-INVARIANT `MODEL` variable itself from that return
                # value, so a model dropped for one (phenotype, population)
                # combination stayed missing from EVERY SUBSEQUENT iteration's
                # plot too - even for a phenotype/population where it DOES
                # have data - because the loop accumulated removals across
                # iterations instead of recomputing them fresh each time.
                # Fixed by always passing the ORIGINAL, never-rebound `MODEL`
                # list into quantile_conversion() and keeping its filtered
                # return in a per-iteration local (`model_for_plot`) instead -
                # quantile_conversion()'s own contract/return shape is
                # unchanged.
                model_for_plot = quantile_conversion(_effect_grouped_all, _effect_grouped_pop, marker_info, chrom_info, PHENOTYPE[i], MODEL, end_adjust, POPULATION[j], WINDOW, RESULT_NAME, ASCENDING,
                                                    ensemble_hue_rule=_ensemble_hue_rule(circos_config),
                                                    label_cfg=circos_config)
                if len(model_for_plot) == 0:
                    # ver4-75: nothing to draw for this (phenotype, population) - see
                    # _warn_no_model_data(); plot() would otherwise die on MODEL[0].
                    _warn_no_model_data(PHENOTYPE[i], POPULATION[j])
                    continue
                interaction_selected = interaction(_interaction_groups, _interaction_model_order, marker_info, PHENOTYPE[i], circos_config, POPULATION[j], RESULT_NAME, _attention_groups, _attention_model_order)
                higher_order_selected = higher_order_interaction(_higher_order_groups, _higher_order_model_order, marker_info, PHENOTYPE[i], circos_config, POPULATION[j], RESULT_NAME, interaction_orders)
                plot(interaction_selected, chrom_info, gene_info, pop_source, PHENOTYPE[i], model_for_plot, circos_config, CYTOBAND_COLORMAP, POPULATION[j],RESULT_NAME, plot_dpi=plot_dpi, legend_accumulator=_legend_accumulator,
                     higher_order_ring_frame=higher_order_selected, interaction_orders=interaction_orders, higher_order_layout=higher_order_layout)
    else:
        # Every (phenotype, population) pair reads the SAME read-only
        # inputs (only the SELECTION of which phenotype/population varies
        # per task - see the ver4-4 R7.f comment above these are computed
        # once for exactly this reason) and writes its OWN distinct output
        # file, so the whole `PHENOTYPE x POPULATION` unit-of-work space
        # can be hashed out across a process pool with no coordination
        # needed beyond merging each task's own small legend-accumulator
        # return value afterwards (see `_render_one_circos_plot()`'s own
        # docstring for why that return-and-merge shape, rather than a
        # "shared" mutable accumulator, is what a process pool requires).
        print(f"[circos_plot] Rendering {len(PHENOTYPE) * len(POPULATION)} "
              f"(phenotype, population) circos plot set(s), split per model, across "
              f"{_effective_workers} worker processes "
              f"(n_workers={n_workers!r}, n_jobs={n_jobs!r}).")
        _ctx = {
            'effect_grouped_all': _effect_grouped_all, 'effect_grouped_pop': _effect_grouped_pop,
            'marker_info': marker_info, 'chrom_info': chrom_info, 'gene_info': gene_info,
            'MODEL': MODEL, 'end_adjust': end_adjust, 'WINDOW': WINDOW, 'RESULT_NAME': RESULT_NAME,
            'ASCENDING': ASCENDING, 'interaction_groups': _interaction_groups,
            'interaction_model_order': _interaction_model_order, 'circos_config': circos_config,
            'attention_groups': _attention_groups, 'attention_model_order': _attention_model_order,
            'pop_source': pop_source, 'CYTOBAND_COLORMAP': CYTOBAND_COLORMAP, 'plot_dpi': plot_dpi,
            # Update ID ver4-25, R12: the k>=3 analog of 'interaction_groups'/
            # 'interaction_model_order' above, plus the two small config
            # values `_render_one_circos_plot()` needs to call
            # `higher_order_interaction()`/`plot()` the same way the serial
            # loop above does.
            'higher_order_groups': _higher_order_groups, 'higher_order_model_order': _higher_order_model_order,
            'interaction_orders': interaction_orders, 'higher_order_layout': higher_order_layout,
        }
        _n_units = len(PHENOTYPE) * len(POPULATION)
        # How many model-slices to cut each unit into so that, in total,
        # there are at least as many render tasks as workers. With many
        # units (>= workers) one slice per unit is right - the per-slice
        # setup cost (building the ring tracks) is paid once per unit.
        _slices_per_unit = max(1, -(-_effective_workers // max(1, _n_units)))
        with ProcessPoolExecutor(
            max_workers=_effective_workers, mp_context=_MP_CONTEXT,
            initializer=_init_circos_plot_worker, initargs=(_ctx,),
        ) as _executor:
            # Two-stage pipeline. Stage 1 (per unit): write the marker-effect
            # tsv files and resolve the link tables. Stage 2 (per unit, per
            # slice of its models): render and save the figures. Stage-2 tasks
            # are submitted the moment their unit's stage 1 finishes, so
            # rendering starts while later units are still being prepared.
            _pending = {}
            for i in range(len(PHENOTYPE)):
                for j in range(len(POPULATION)):
                    _fut = _executor.submit(_prepare_one_circos_unit, PHENOTYPE[i], POPULATION[j])
                    _pending[_fut] = ('prepare', PHENOTYPE[i], POPULATION[j])
            try:
                while _pending:
                    _done, _ = wait(list(_pending), return_when=FIRST_COMPLETED)
                    for _future in _done:
                        _stage, _phenotype_value, _population_value = _pending.pop(_future)
                        try:
                            _result = _future.result()
                        except Exception:
                            print(f"[circos_plot] FAILED while {'preparing' if _stage == 'prepare' else 'rendering'} "
                                  f"phenotype={_phenotype_value!r} population={_population_value!r} "
                                  f"(see traceback below).")
                            raise
                        if _stage == 'prepare':
                            _model_for_plot, _interaction_selected, _higher_order_selected = _result
                            if len(_model_for_plot) == 0:
                                _warn_no_model_data(_phenotype_value, _population_value)
                                continue
                            _models = _circos_models_to_render(_interaction_selected, _higher_order_selected)
                            _k = max(1, min(_slices_per_unit, len(_models)))
                            for _s in range(_k):
                                _slice = _models[_s::_k]
                                _rf = _executor.submit(
                                    _render_one_circos_chunk, _phenotype_value, _population_value,
                                    _model_for_plot, _interaction_selected, _higher_order_selected, _slice)
                                _pending[_rf] = ('render', _phenotype_value, _population_value)
                        else:
                            _merge_into_legend_accumulator(
                                _legend_accumulator,
                                colours_used_by_hue=_result['colours_used_by_hue'],
                                gene_colours_used=_result['gene_colours_used'],
                                has_inter_chr_link=_result['has_inter_chr_link'],
                                has_intra_chr_link=_result['has_intra_chr_link'],
                                truncated_labels=_result['truncated_labels'],
                                has_hub_glyph=_result['has_hub_glyph'],
                                combined_orders_used=_result['combined_orders_used'],
                                highlight_regions=_result.get('highlight_regions'),
                                highlight_intra=_result.get('highlight_intra'),
                                highlight_inter=_result.get('highlight_inter'),
                                highlight_dimmed=_result.get('highlight_dimmed'),
                            )
            except BaseException:
                # Fail fast like the serial loop: don't sit through every
                # still-queued render before surfacing the error.
                _executor.shutdown(wait=False, cancel_futures=True)
                raise

    # ver4-5 (single session-wide legend): build and save the ONE
    # combined legend now that every plot() call above has finished
    # merging its own findings into `_legend_accumulator` - covers every
    # colour/link-type used anywhere across this circos_plot() call,
    # rather than a separate, plot-specific file for each one.
    # `_build_circos_legend_handles()` itself is unchanged - it already
    # only ever draws what it's given, and a session-wide accumulator is
    # exactly the same shape a single plot's own (now-removed) local
    # variables used to be, just unioned across more than one render.
    _legend_handles, _legend_handler_map = _build_circos_legend_handles(
        _legend_accumulator['colours_used_by_hue'], CYTOBAND_COLORMAP, _legend_accumulator['gene_colours_used'],
        _legend_accumulator['has_inter_chr_link'], _legend_accumulator['has_intra_chr_link'],
        truncated_labels=_legend_accumulator['truncated_labels'],
        has_hub_glyph=_legend_accumulator['has_hub_glyph'],
        combined_orders_used=_legend_accumulator['combined_orders_used'],
        link_colour_mode=_link_options(circos_config)['colour_mode'],
        highlight_regions=_legend_accumulator['highlight_regions'],
        highlight_intra=_legend_accumulator['highlight_intra'],
        highlight_inter=_legend_accumulator['highlight_inter'],
        highlight_dimmed=_legend_accumulator['highlight_dimmed'],
    )
    _save_circos_legend(_legend_handles, _legend_handler_map,
                         './Result/'+RESULT_NAME+'/circos_legend.png',
                         dpi=plot_dpi, extra_formats=_ps.extra_formats_of(circos_config))
