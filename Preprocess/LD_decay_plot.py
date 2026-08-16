"""
Preprocess/LD_decay_plot.py
============================

Optional LD-decay diagnostic add-on for the LD pruning pre-processing step
(Preprocess/LD_pruning.py). This module is entirely self-contained - it
never imports anything from LD_pruning.py, and LD_pruning.py never imports
anything from here. genomic_prediction.py's GP() is the only place the two
are wired together (see its per-task LD-pruning blocks), so nothing about
the existing LD *pruning* flow (what markers survive, what gets passed to
the models) is touched by this feature existing, whether it's turned on or
off.

WHAT AN "LD DECAY PLOT" SHOWS
------------------------------
For pairs of markers, LD (measured here as the unphased hardcall r^2
between two markers' genotype dosages - exactly the statistic
Preprocess.LD_pruning.LD_pruning() already prunes on) typically falls off
as the distance between the pair increases. Plotting mean r^2 against
distance (binned into distance windows) is the standard diagnostic for
choosing an LD-pruning window/r^2 threshold: it shows, empirically, how
far apart two markers need to be before they stop being redundant for
this dataset/population.

DISTANCE UNITS ('window_unit')
--------------------------------------------------------------------------
This module supports the exact same three distance units
Preprocess.LD_pruning.ld_prune_snps() itself supports, and a decay curve
can be requested for any subset of them (kb only, or kb+cm+variants all
at once, etc. - see genomic_prediction.py's `_decay_window_units`):

    'kb'       - physical distance in kilobases. Markers are grouped by
                 chromosome and ordered by physical position. Needs a real
                 SNP info file with 'CHR'/'POS' columns.
    'cm'       - genetic distance in centimorgans. Same idea, ordered by
                 genetic position instead. Needs a real SNP info file with
                 'CHR'/'CM' columns.
    'variants' - distance measured in "markers apart" - i.e. purely by
                 column order/index, exactly mirroring
                 Preprocess.LD_pruning's own window_unit='variants' path
                 (which likewise never needs a real map). This is the one
                 unit that works with NO SNP info file at all.

WORKFLOW (see genomic_prediction.py's GP() for how these are actually
wired together)
--------------------------------------------------------------------------
1. compute_ld_decay_data(...)      - one binned decay curve, for ONE
   prediction scenario and ONE window_unit, computed from that scenario's
   own PRE-pruning training-set genotypes (same data LD_pruning() itself
   prunes on for that scenario).
2. save_ld_decay_data(...)         - that curve, plus scenario metadata
   (including which window_unit it's in), to its own CSV file (one row
   per distance bin).
3. plot_ld_decay(...)              - that same curve, as a PNG.
4. average_ld_decay_data(...) /
   average_and_plot_ld_decay(...)  - once every prediction scenario has
   finished, combine every per-scenario CSV written by (2), SEPARATELY
   for each (population, window_unit) combination, into one n_pairs-
   weighted average curve per combination, and plot each one as its own
   PNG - the "average across all prediction scenarios" plot(s).

NOT split by phenotype, at either step 1-3 or step 4: LD decay reflects
which INDIVIDUALS end up in a scenario's training set, which never
depends on phenotype - genomic_prediction.py's train/test split is keyed
only on population/ratio/replicate (see GP()'s train_test_split calls) -
so two scenarios that only differ by phenotype share the exact same
training-set individuals and would produce an identical decay curve.
GP() therefore samples decay data (steps 1-3) at most ONCE per
(population, ratio, replicate) combination, regardless of how many
phenotypes share it (see GP()'s _ld_decay_due()) - both reducing the
amount of data stored (Requirement) and making step 4's averaging
correctly per-POPULATION only (never diluted, or arbitrarily split, by a
phenotype axis that carries no real LD-decay information). A kb-based
curve and a cm-based curve for the same population are still never mixed
together (window_unit is still part of both the sampling key and the
averaging group).

Steps 1-3 are called from INSIDE GP()'s main task loop, so they run
identically whether GP() is invoked once (Sequential) or many times, once
per HPC batch (Parallel / Step 1) - each call only ever writes its own
uniquely-named files (see genomic_prediction.py's filename de-
duplication), so many concurrent GP() processes writing into the very
same shared subfolder is safe. Step 4 is called once, after EVERY
prediction scenario across every batch has finished (Sequential: directly
after GP() returns; Parallel: from Step 2, once every batch's files
already sit in the same shared folder) - see run_sequential.py /
run_step2_assemble.py / main_app.py.

WHERE FILES ARE WRITTEN
--------------------------------------------------------------------------
Result/<RESULT_NAME>/LD_decay_plots/                                    <- the subfolder
Result/<RESULT_NAME>/LD_decay_plots/.keep_log_state                     <- tiny marker recording this run's keep_log setting (see write_keep_log_marker)
Result/<RESULT_NAME>/LD_decay_plots/LDdecay_<unit>_<population>_<sample>.png  <- per-(population, ratio, replicate) plot (ALWAYS kept)
Result/<RESULT_NAME>/LD_decay_plots/data/LDdecay_<unit>_<population>_<sample>.csv  <- matching data ("the log" - deleted after averaging if keep_log=False)
Result/<RESULT_NAME>/LD_decay_plots/LD_decay_average_<unit>_<population>.png  <- Step 4's plots (one per unit x population combination, ALWAYS kept)
Result/<RESULT_NAME>/LD_decay_plots/LD_decay_average_<unit>_<population>.csv  <- Step 4's data (one per combination, ALWAYS kept)
Result/<RESULT_NAME>/LD_decay_plots/LD_decay_average_all.csv            <- every combination's averaged rows, stacked together, for convenience (ALWAYS kept)

KEEPING OR DISCARDING THE PER-SCENARIO "LOG"
--------------------------------------------------------------------------
LD_prune['decay_plot']['keep_log'] (default True) controls only the
per-scenario data/<...>.csv files above - the raw numeric bins behind
each individual scenario's plot. Everything else (every PNG plot, at
both the per-scenario and the population x phenotype average level, plus
the average CSVs) is ALWAYS produced and kept, regardless of this
setting - see genomic_prediction.py's GP() and average_and_plot_ld_decay()
below.

Setting keep_log=False does NOT skip writing those per-scenario CSVs
during the run - they're still needed on disk, transiently, as the only
way average_and_plot_ld_decay() can combine data from many scenarios
(and, for the Parallel workflow, from many separate GP() processes/HPC
batches that share no memory - see 'WORKFLOW' above). What it controls is
whether they're deleted once the average has been computed from them:
average_and_plot_ld_decay() removes every file under data/ as its very
last step when keep_log is False, after every average PNG/CSV has already
been written - freeing disk space for runs with many prediction scenarios
without losing any of the actual deliverables (the plots, at every
level).

The subfolder name ('LD_decay_plots') is a fixed default rather than a
user-configurable GUI setting - deliberately, since the averaging step
(Step 4 above) runs from a completely separate process/step for the
Parallel workflow (Step 2, which never sees Step 1's LD_PRUNE config at
all - see run_step2_assemble.py) and needs to agree on the same folder
name without any config plumbing between the two steps. The same reasoning
is why keep_log itself is recorded to a small marker file
(.keep_log_state, written once by every GP() call - see
write_keep_log_marker/read_keep_log_marker) rather than only living in
the LD_PRUNE config: Step 2 needs to know its value too, without Step 1's
config ever being passed to it.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

from pipeline_utils import unify_columns_by_position

# NOTE: deliberately no `logging` module usage anywhere in this file. This
# codebase never calls logging.basicConfig() (or attaches any handler)
# anywhere - genomic_prediction.py, assemble.py, and Preprocess/plink_io.py
# all report progress/diagnostics via plain print() instead. Under Python's
# default logging configuration, logger.info(...) messages are silently
# DROPPED (the root logger's effective level is WARNING), so using the
# logging module here would make exactly the messages a user most needs -
# "why did this produce 0 plots?" - invisible in both the console and the
# saved run log (pipeline_utils.TimestampedWriter only ever wraps stdout).
# print() is used everywhere below instead, prefixed with
# '[LD_decay_plot]' to match the '[GP]'/'[assemble]'/'[plink_io]' style
# already used throughout the rest of the pipeline.

WINDOW_UNITS = ('kb', 'cm', 'variants')

# Sane, documented defaults - every one of these is overridable via the
# 'decay_plot' sub-dict inside an LD_prune config (see
# genomic_prediction.py / main_app.py's build_ld_decay_plot_config()).
# Units for max_distance/bin_width depend on window_unit: kb -> kilobases,
# cm -> centimorgans, variants -> number of markers apart (matching
# Preprocess.LD_pruning.ld_prune_snps' own three window_unit modes).
DEFAULT_SUBFOLDER = 'LD_decay_plots'
DEFAULT_DATA_DIRNAME = 'data'
DEFAULT_MAX_PAIRS_PER_CHR = 20000
DEFAULT_RANDOM_STATE = 0

DEFAULT_MAX_DISTANCE = {'kb': 1000.0, 'cm': 50.0, 'variants': 50.0}
DEFAULT_BIN_WIDTH = {'kb': 10.0, 'cm': 1.0, 'variants': 1.0}

# Axis label / unit-name shown on plots and in log/filename text, per window_unit.
_UNIT_LABEL = {'kb': 'kb', 'cm': 'cM', 'variants': 'markers apart'}
_AXIS_LABEL = {
    'kb': 'Physical distance between markers (kb)',
    'cm': 'Genetic distance between markers (cM)',
    'variants': 'Markers apart (variant count)',
}

# Above this many mapped markers in a single group (a chromosome for
# 'kb'/'cm', or - for 'variants' with no CHR available at all - the whole
# marker set), computing r^2 via one full (n_group x n_group) correlation
# matrix (see _decay_pairs_for_group's vectorised fast path) would use too
# much memory (O(n^2)) - fall back to the slower, but memory-bounded,
# per-pair computation instead (still capped by max_pairs_per_chr either
# way).
_VECTORIZED_MAX_GROUP_MARKERS = 4000

_DECAY_COLUMNS = ['bin_start', 'bin_end', 'bin_mid', 'mean_r2', 'n_pairs']
# Requirement: LD decay is a property of the genotypes (which individuals
# end up in a scenario's training set), not of the phenotype being
# predicted - genomic_prediction.py's train/test split is keyed only on
# population/ratio/replicate (see GP()'s train_test_split calls), never
# on phenotype, so two scenarios that only differ by phenotype share the
# exact same training-set individuals and would produce the same decay
# curve. Averaging is therefore done per (window_unit, population) only -
# never split further by phenotype - and genomic_prediction.py's own
# per-scenario SAMPLING is deduplicated the same way (see GP()'s
# _ld_decay_due()): once a (population, ratio, replicate) combination has
# been sampled (or skipped for this run's configured frequency), no
# further phenotype sharing that same combination samples it again.
_GROUP_COLUMNS = ['window_unit', 'population']


class LDDecayPlotError(ValueError):
    """Raised for bad configuration (never for 'no data to plot' - that's
    a normal, silently-handled outcome, not an error - see the empty-
    DataFrame returns throughout this module)."""


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _pairwise_r2(x: np.ndarray, y: np.ndarray) -> float:
    """Unphased hardcall r^2 between two genotype dosage vectors, ignoring
    samples missing in either - the exact same statistic (and the same
    handling of missing/monomorphic markers) as
    Preprocess.LD_pruning.LD_pruning's own internal _pairwise_r2,
    duplicated here (rather than imported) so this module has zero
    dependency on LD_pruning.py's internals and can't be broken by a
    future refactor there (see module docstring)."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return np.nan
    xm, ym = x[mask], y[mask]
    if xm.std() == 0 or ym.std() == 0:
        return np.nan
    r = np.corrcoef(xm, ym)[0, 1]
    return np.nan if np.isnan(r) else r * r


def sanitize_for_filename(value) -> str:
    """Turn an arbitrary population/phenotype/sample/unit label into
    something safe to use as (part of) a filename - collapses any run of
    characters that isn't alphanumeric/dot/dash/underscore into a single
    '_' (e.g. 'PopA->PopB' -> 'PopA_PopB'), so a 'between'-scenario
    population label (joined with '->'), or a phenotype name containing
    spaces/punctuation, can never introduce a path separator or other
    filesystem-unsafe character into the output filename."""
    text = re.sub(r'[^A-Za-z0-9._-]+', '_', str(value)).strip('_')
    return text or 'NA'


def unique_path(path: str) -> str:
    """Return `path` unchanged, or - if it already exists on disk - the
    same path with an incrementing numeric suffix inserted before the
    extension, until a free name is found (e.g. 'foo.png' -> 'foo_v2.png').

    Guards against the (uncommon but real) case where window_unit +
    population + phenotype + sample doesn't uniquely identify a single
    scenario on its own - e.g. the 'within' scenario with more than one
    split RATIO configured re-uses the same replicate numbers
    (1..SAMPLE_NUM) for every ratio - rather than silently overwriting an
    earlier scenario's plot/data with a later one that happens to share
    the same label.
    """
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    n = 2
    while os.path.exists(f'{base}_v{n}{ext}'):
        n += 1
    return f'{base}_v{n}{ext}'


# Filename for the tiny marker file that records a run's "keep the
# per-scenario LD decay data CSVs after averaging?" setting (see
# write_keep_log_marker/read_keep_log_marker below).
_KEEP_LOG_MARKER_FILENAME = '.keep_log_state'


def write_keep_log_marker(plot_dir: str, keep_log: bool) -> None:
    """Persist this run's 'keep the per-scenario LD decay data CSVs after
    averaging?' setting to a tiny marker file inside plot_dir, so
    average_and_plot_ld_decay() - which for the Parallel workflow runs
    from Step 2, a completely separate process that never sees Step 1's
    LD_PRUNE config at all (see module docstring) - can look the setting
    up without needing that config passed to it explicitly. Every GP()
    call in a given run writes the same value (every batch shares one
    LD_PRUNE config), so the repeated overwrites across parallel batches
    are harmless - whichever batch runs first 'wins', and every batch
    would write the same thing anyway."""
    os.makedirs(plot_dir, exist_ok=True)
    with open(os.path.join(plot_dir, _KEEP_LOG_MARKER_FILENAME), 'w') as f:
        f.write('1' if keep_log else '0')


def read_keep_log_marker(plot_dir: str) -> "bool | None":
    """Read back what write_keep_log_marker() wrote for this RESULT_NAME,
    or None if it was never written - e.g. the LD decay plot feature
    wasn't enabled for this run at all, or `plot_dir` is left over from a
    version of this module that predates the keep_log option. Callers
    should treat None as 'keep everything' (average_and_plot_ld_decay()
    does exactly this) - the safe default is to never delete data unless
    a run explicitly opted out of keeping it."""
    path = os.path.join(plot_dir, _KEEP_LOG_MARKER_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return f.read().strip() == '1'
    except OSError:
        return None


def _clear_directory(dir_path: str) -> int:
    """Delete every regular file directly inside dir_path (data_dir is
    always flat - no subdirectories to recurse into), then remove
    dir_path itself if it ends up empty. Returns how many files were
    removed. Never raises - a file that can't be removed (e.g. a
    permissions issue) is reported with a warning and simply left behind
    rather than aborting the whole cleanup, since this cleanup is a
    disk-space nicety, not something the rest of the pipeline depends on."""
    if not os.path.isdir(dir_path):
        return 0
    removed = 0
    for name in os.listdir(dir_path):
        full = os.path.join(dir_path, name)
        if os.path.isfile(full):
            try:
                os.remove(full)
                removed += 1
            except OSError as exc:
                print(f"[LD_decay_plot] WARNING: could not remove '{full}': {exc!r}")
    try:
        os.rmdir(dir_path)
    except OSError:
        pass  # not empty (a removal above failed), or already gone - fine either way
    return removed


def resolve_snp_info_for_decay(ld_config: dict) -> "pd.DataFrame | None":
    """Load the same SNP map Preprocess.LD_pruning.LD_pruning() itself
    would load from `ld_config['snp_info']`, for the decay plot's
    kb/cm axes. Returns None if no snp_info file is configured at all.

    This is fine (not an error) if only window_unit='variants' is
    requested: that mode's x-axis is "number of markers apart" along
    genotype_df's own column order, exactly like
    Preprocess.LD_pruning's own window_unit='variants' path (see its
    module docstring), which likewise never needs a real map. For
    window_unit in ('kb', 'cm') a real map IS required (a synthetic,
    evenly-spaced placeholder position - see
    Preprocess.LD_pruning.make_placeholder_snp_info - would produce a
    decay curve that looks meaningful but isn't) - the caller
    (genomic_prediction.py's GP()) is responsible for dropping those two
    window units from the requested set when this returns None."""
    snp_info_path = (ld_config or {}).get('snp_info')
    if not snp_info_path:
        return None
    snp_info = pd.read_csv(snp_info_path, index_col=0)
    # Requirement 8: same positional column unification as
    # Preprocess.LD_pruning.LD_pruning() itself applies to this exact same
    # file (CHR, POS, then optionally CM, A1, A2, in that order) - kept
    # consistent between the two readers of this file so a decay plot's
    # idea of which column is CHR/POS/CM can never disagree with LD
    # pruning's own.
    _snp_info_canonical_order = ['CHR', 'POS', 'CM', 'A1', 'A2']
    return unify_columns_by_position(
        snp_info, _snp_info_canonical_order[:min(snp_info.shape[1], 5)], 'SNP info file'
    )


def result_paths(result_name: str, subfolder: str = DEFAULT_SUBFOLDER,
                  data_dirname: str = DEFAULT_DATA_DIRNAME):
    """(plot_dir, data_dir) for a given RESULT_NAME - the single source of
    truth for this feature's folder layout, used by every other function
    in this module and by genomic_prediction.py."""
    plot_dir = os.path.join('.', 'Result', result_name, subfolder)
    data_dir = os.path.join(plot_dir, data_dirname)
    return plot_dir, data_dir


def ld_decay_data_exists(result_name: str, subfolder: str = DEFAULT_SUBFOLDER,
                          data_dirname: str = DEFAULT_DATA_DIRNAME) -> bool:
    """True if at least one per-scenario LD decay data CSV has already
    been written for this RESULT_NAME. Used by callers (main_app.py's
    local-run progress bar, run_step2_assemble.py) that want to know
    whether the averaging step has anything to do, without needing their
    own copy of the LD_PRUNE config (Step 2 of the Parallel workflow never
    sees it - see module docstring)."""
    _, data_dir = result_paths(result_name, subfolder, data_dirname)
    return os.path.isdir(data_dir) and bool(glob.glob(os.path.join(data_dir, '*.csv')))


# --------------------------------------------------------------------------
# Step 1: per-scenario decay data
# --------------------------------------------------------------------------

def _decay_pairs_for_group(geno_ordered: np.ndarray, metric_sorted: np.ndarray,
                            max_distance: float, max_pairs: int, rng: np.random.Generator):
    """Shared core for ALL THREE window units: given one group's genotype
    columns already ordered ascending by whatever distance metric applies
    (kb for 'kb', centimorgans for 'cm', or plain marker-index for
    'variants' - metric_sorted, same order/length as geno_ordered's
    columns), find every pair within `max_distance` of each other and
    return their (distance, r^2) arrays.

    This is the exact same two-pointer sweep + vectorised-correlation-
    matrix-with-memory-bounded-fallback approach for every window unit -
    only the *meaning* of 'distance' changes between calls, never the
    algorithm - so kb/cm/variants get identical performance
    characteristics and the same max_pairs guarantee.
    """
    n = len(metric_sorted)
    if n < 2:
        return np.array([]), np.array([])

    # Two-pointer / binary-search sweep over an ALREADY ascending-sorted
    # metric: for each anchor a, every partner within the window is a
    # contiguous run of later indices (np.searchsorted finds where that
    # run ends) - O(n log n) total, never an O(n^2) scan of pairs that are
    # obviously too far apart to ever matter.
    pair_a_parts, pair_b_parts, pair_dist_parts = [], [], []
    for a in range(n - 1):
        hi = np.searchsorted(metric_sorted, metric_sorted[a] + max_distance, side='right')
        if hi > a + 1:
            js = np.arange(a + 1, hi)
            pair_a_parts.append(np.full(js.shape, a))
            pair_b_parts.append(js)
            pair_dist_parts.append(metric_sorted[js] - metric_sorted[a])

    if not pair_a_parts:
        return np.array([]), np.array([])

    pair_a = np.concatenate(pair_a_parts)
    pair_b = np.concatenate(pair_b_parts)
    pair_dist = np.concatenate(pair_dist_parts)

    if len(pair_a) > max_pairs:
        keep = rng.choice(len(pair_a), size=max_pairs, replace=False)
        pair_a, pair_b, pair_dist = pair_a[keep], pair_b[keep], pair_dist[keep]

    if n <= _VECTORIZED_MAX_GROUP_MARKERS:
        # Fast path: one correlation-matrix computation for the whole
        # group, instead of one numpy call per candidate pair - this is
        # what keeps compute_ld_decay_data usably fast even for many
        # thousands of candidate pairs. Missing genotype calls are
        # imputed to each marker's own mean ONLY for this correlation
        # matrix (a diagnostic-plot-only approximation - np.corrcoef
        # requires complete data; the exact, pairwise-deletion r^2
        # _pairwise_r2 computes is what actually decides which markers
        # survive real LD pruning, never this plot, so this approximation
        # never affects prediction results). Monomorphic (zero-variance)
        # markers correctly contribute nothing to the curve rather than
        # corrupting it, exactly like _pairwise_r2's own std==0 guard.
        col_mean = np.nanmean(geno_ordered, axis=0)
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
        filled = np.where(np.isnan(geno_ordered), col_mean, geno_ordered)
        col_std = filled.std(axis=0)

        with np.errstate(invalid='ignore', divide='ignore'):
            corr = np.corrcoef(filled, rowvar=False)
        pair_r2 = np.asarray(corr)[pair_a, pair_b] ** 2
        zero_var = (col_std[pair_a] == 0) | (col_std[pair_b] == 0)
        valid = ~np.isnan(pair_r2) & ~zero_var
        return pair_dist[valid], pair_r2[valid]

    # Memory-bounded fallback for a very marker-dense group - exact
    # pairwise-deletion r^2, one pair at a time, capped at max_pairs pairs
    # (already enforced above).
    out_dist, out_r2 = [], []
    for a_idx, b_idx, d in zip(pair_a, pair_b, pair_dist):
        r2 = _pairwise_r2(geno_ordered[:, a_idx], geno_ordered[:, b_idx])
        if not np.isnan(r2):
            out_dist.append(d)
            out_r2.append(r2)
    return np.asarray(out_dist), np.asarray(out_r2)


def compute_ld_decay_data(
    genotype_df: pd.DataFrame,
    snp_info: "pd.DataFrame | None" = None,
    window_unit: str = 'kb',
    max_distance: "float | None" = None,
    bin_width: "float | None" = None,
    max_pairs_per_chr: int = DEFAULT_MAX_PAIRS_PER_CHR,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """
    Bin SNP-pair LD (r^2) by distance. Supports the same three distance
    units Preprocess.LD_pruning.ld_prune_snps() itself supports - see the
    module docstring's 'DISTANCE UNITS' section for what each one means
    and what it needs from `snp_info`.

    Parameters
    ----------
    genotype_df : samples x SNPs, values in [0, 2] (NaN allowed for
        missing) - same orientation/encoding
        Preprocess.LD_pruning.ld_prune_snps() expects. Pass the
        PRE-pruning genotype matrix: the whole point of a decay plot is to
        show the redundancy LD pruning is about to remove, so pruned data
        would defeat its purpose.
    snp_info : DataFrame indexed by SNP ID. Required (with the columns
        noted in the module docstring) for window_unit in ('kb', 'cm');
        optional for 'variants'. Markers not present in snp_info's index,
        or with a missing/non-numeric required column, are silently
        excluded from that unit's grouping (there's no position to place
        them at) - except for 'variants' with snp_info=None, where every
        marker in genotype_df is used.
    window_unit : 'kb', 'cm', or 'variants' - see module docstring.
    max_distance : only marker pairs within this distance of each other
        are considered ("the window shown on the x-axis"), in the units
        implied by window_unit (kb / cM / markers). Defaults to
        DEFAULT_MAX_DISTANCE[window_unit] if not given.
    bin_width : width of each distance bin along the x-axis, same units as
        max_distance. Defaults to DEFAULT_BIN_WIDTH[window_unit] if not
        given.
    max_pairs_per_chr : candidate pairs are randomly subsampled down to at
        most this many (per chromosome for 'kb'/'cm'; per the single
        pooled group for 'variants' with no CHR available) before r^2 is
        actually computed, so runtime stays bounded on marker-dense
        groups. This only affects how many points make up the curve's
        precision - it never removes any marker from the actual
        prediction pipeline.
    random_state : seed for the subsampling above, so a given scenario's
        decay curve is reproducible run to run.

    Returns
    -------
    DataFrame with columns ['bin_start', 'bin_end', 'bin_mid', 'mean_r2',
    'n_pairs'] (values in whatever unit window_unit implies), one row per
    non-empty distance bin, sorted by distance. Empty (0 rows, same
    columns) whenever there's nothing meaningful to plot - fewer than 2
    usable markers, fewer than 3 samples, or no marker pair falls within
    max_distance of another in the same group - rather than raising, since
    this is always an optional diagnostic and "nothing to show" is a
    normal outcome (e.g. a very small task-level training set).

    Notes
    -----
    Runtime is roughly linear in the number of (marker, group) pairs that
    fall within max_distance of each other, capped per group by
    max_pairs_per_chr - for very dense, genome-wide marker panels
    (hundreds of thousands of SNPs), consider a smaller max_distance
    and/or max_pairs_per_chr to keep this fast, exactly as
    Preprocess.LD_pruning's own cM-based fallback pruning recommends for
    its comparable NumPy-only computation.
    """
    if window_unit not in WINDOW_UNITS:
        raise LDDecayPlotError(
            f"window_unit must be one of {WINDOW_UNITS} (got {window_unit!r})"
        )
    if max_distance is None:
        max_distance = DEFAULT_MAX_DISTANCE[window_unit]
    if bin_width is None:
        bin_width = DEFAULT_BIN_WIDTH[window_unit]

    empty = pd.DataFrame(columns=_DECAY_COLUMNS)

    if genotype_df.shape[0] < 3 or genotype_df.shape[1] < 2:
        return empty
    if max_distance <= 0 or bin_width <= 0:
        raise LDDecayPlotError(
            f"max_distance and bin_width must both be > 0 for window_unit={window_unit!r} "
            f"(got max_distance={max_distance}, bin_width={bin_width})."
        )

    columns = genotype_df.columns
    n_markers = len(columns)

    if window_unit == 'variants':
        # No real map needed - distance is measured in "markers apart",
        # along genotype_df's own column order, exactly mirroring
        # Preprocess.LD_pruning's own window_unit='variants' path (see
        # that module's docstring: "Only depends on marker order").
        metric = pd.Series(np.arange(n_markers, dtype=float), index=columns)
        if snp_info is not None and 'CHR' in snp_info.columns:
            chr_series = snp_info.reindex(columns)['CHR']
            # Markers absent from snp_info (or with a blank CHR) still get
            # a group here (rather than being dropped) - unlike kb/cm,
            # 'variants' has no map dependency to fail on, so there's no
            # reason to exclude them; they're simply pooled together in
            # their own group, keeping their relative column order.
            chr_series = chr_series.fillna('__unmapped__')
        else:
            chr_series = pd.Series('__all__', index=columns)
    else:
        if snp_info is None:
            raise LDDecayPlotError(
                f"window_unit={window_unit!r} requires snp_info with a 'CHR' column and a "
                f"{'POS' if window_unit == 'kb' else 'CM'} column."
            )
        required_col = 'POS' if window_unit == 'kb' else 'CM'
        missing_cols = [c for c in ('CHR', required_col) if c not in snp_info.columns]
        if missing_cols:
            raise LDDecayPlotError(
                f"snp_info is missing required column(s) for window_unit={window_unit!r}: {missing_cols}"
            )
        info = snp_info.reindex(columns).copy()
        info[required_col] = pd.to_numeric(info[required_col], errors='coerce')
        chr_series = info['CHR']
        metric = info[required_col]
        if window_unit == 'kb':
            metric = metric / 1000.0  # bp -> kb

    mapped = chr_series.notna() & metric.notna()
    if mapped.sum() < 2:
        # Loud (print, not logging - see module header) since this is
        # exactly the kind of silent-no-op that's otherwise very hard to
        # diagnose: it usually means snp_info's index didn't line up with
        # genotype_df's marker names for (almost) any marker - typically a
        # SNP-ID-naming mismatch between the SNP info CSV and the
        # genotype file, or CHR/POS/CM columns that are blank/non-numeric.
        required_desc = {'kb': 'CHR/POS', 'cm': 'CHR/CM', 'variants': 'CHR'}[window_unit]
        print(
            f"[LD_decay_plot] compute_ld_decay_data: only {int(mapped.sum())} / {n_markers} "
            f"marker(s) have a real, numeric {required_desc} (after matching snp_info's index "
            f"against genotype_df's column names) - need at least 2 to plot anything on a "
            f"{window_unit} distance axis. If this is unexpected, check that the SNP info "
            f"file's ID column values match the genotype file's marker column names exactly."
        )
        return empty

    geno_all = genotype_df.to_numpy(dtype=float)
    col_index = {c: i for i, c in enumerate(columns)}

    rng = np.random.default_rng(random_state)
    distances = []
    r2_values = []

    grouping = pd.DataFrame({'CHR': chr_series, 'METRIC': metric})[mapped]
    for chrom, group in grouping.groupby('CHR'):
        if group.shape[0] < 2:
            continue
        group_sorted = group.sort_values('METRIC', kind='mergesort')
        marker_ids = group_sorted.index
        col_idx = np.array([col_index[m] for m in marker_ids])
        metric_sorted = group_sorted['METRIC'].to_numpy(dtype=float)
        geno_group = geno_all[:, col_idx]

        d, r2 = _decay_pairs_for_group(geno_group, metric_sorted, max_distance, max_pairs_per_chr, rng)
        if d.size:
            distances.append(d)
            r2_values.append(r2)

    if not distances:
        return empty

    distances = np.concatenate(distances)
    r2_values = np.concatenate(r2_values)

    bin_edges = np.arange(0.0, max_distance + bin_width, bin_width)
    bin_idx = np.clip(np.digitize(distances, bin_edges, right=False) - 1, 0, len(bin_edges) - 2)

    pairs_df = pd.DataFrame({'bin_idx': bin_idx, 'r2': r2_values})
    grouped = pairs_df.groupby('bin_idx')['r2'].agg(['mean', 'count']).reset_index()
    grouped['bin_start'] = bin_edges[grouped['bin_idx'].to_numpy()]
    grouped['bin_end'] = bin_edges[grouped['bin_idx'].to_numpy() + 1]
    grouped['bin_mid'] = (grouped['bin_start'] + grouped['bin_end']) / 2.0
    grouped = grouped.rename(columns={'mean': 'mean_r2', 'count': 'n_pairs'})
    grouped = grouped[_DECAY_COLUMNS].sort_values('bin_start').reset_index(drop=True)
    return grouped


def save_ld_decay_data(decay_df: pd.DataFrame, csv_path: str, metadata: "dict | None" = None) -> None:
    """Write one scenario's binned decay curve to its own CSV (Requirement
    5), with `metadata` attached as constant columns so the CSV is
    self-describing on its own, and so average_ld_decay_data() (Step 4)
    can group scenarios correctly. `metadata` MUST include 'window_unit',
    'population', and 'phenotype' keys for this file to be picked up by
    the averaging step (see average_ld_decay_data's own docstring) -
    genomic_prediction.py always supplies all three (plus 'ratio'/'sample'
    for completeness) when it calls this. Writes a header-only (0-row) CSV
    when decay_df is empty, rather than skipping the file entirely, so the
    'one data file per plot attempt' contract holds even for a scenario
    with nothing plottable."""
    out_dir = os.path.dirname(csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out = decay_df.copy()
    if metadata:
        for key, value in metadata.items():
            out[key] = value
    out.to_csv(csv_path, index=False)


# --------------------------------------------------------------------------
# Step 1 (continued): plotting a single decay curve
# --------------------------------------------------------------------------

def plot_ld_decay(decay_df: pd.DataFrame, png_path: str, title: "str | None" = None,
                   window_unit: str = 'kb') -> bool:
    """Draw one LD-decay curve (mean r^2 vs. binned distance) and save it
    as a PNG. `window_unit` only controls the x-axis label/units shown
    ('kb'/'cm'/'variants' - see module docstring) - decay_df's own
    'bin_mid' values are assumed to already be in that unit.

    Returns False (no file written) if decay_df is empty - e.g. no mapped
    marker pair fell within the configured distance window for this
    scenario - rather than raising, or silently writing a blank/misleading
    plot.
    """
    if decay_df is None or decay_df.shape[0] == 0:
        print(f"[LD_decay_plot] plot_ld_decay: no decay data to plot for '{png_path}' - skipping.")
        return False

    import matplotlib
    matplotlib.use('Agg')  # headless-safe - no dependency on a display existing (HPC nodes, CI, etc.)
    import matplotlib.pyplot as plt

    out_dir = os.path.dirname(png_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(decay_df['bin_mid'], decay_df['mean_r2'], marker='o', markersize=3,
             linewidth=1.5, color='#1f4e79')
    ax.set_xlabel(_AXIS_LABEL.get(window_unit, 'Distance between markers'))
    ax.set_ylabel(r'Mean LD ($r^2$)')
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.set_title(title or 'LD decay')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    # Explicitly closed (rather than relying on garbage collection) since
    # this is called from inside a loop that may run many times per
    # prediction pipeline run - leaving figures open would otherwise
    # accumulate unbounded memory over a long run.
    plt.close(fig)
    return True


# --------------------------------------------------------------------------
# Step 4: averaging every scenario's data into one final plot PER
# (window_unit, population) COMBINATION
# --------------------------------------------------------------------------

def average_ld_decay_data(data_dir: str) -> pd.DataFrame:
    """Combine every per-scenario decay CSV written by save_ld_decay_data()
    under `data_dir`, GROUPED BY (window_unit, population) - i.e. averaged
    across every replicate, split ratio, AND phenotype for that
    combination (LD decay carries no phenotype-specific information - see
    module docstring), but never averaged across different window units
    or different populations, which would mix curves from incompatible
    units or genuinely different genotype pools into one meaningless
    number.

    Within each group, bins are combined with an n_pairs-weighted average
    (a bin with more underlying marker pairs behind it counts more - a
    replicate with only a handful of markers surviving to a given
    distance bin shouldn't move the group average as much as one with
    thousands), approximating the curve you would get by pooling every
    replicate's raw marker pairs together, without needing to keep every
    raw pair from every replicate in memory at once.

    Every CSV under `data_dir` must carry 'window_unit' and 'population'
    columns to be included here - exactly what save_ld_decay_data()'s
    `metadata` argument attaches when genomic_prediction.py calls it (see
    GP()'s per-task LD-pruning blocks) - CSVs missing either are skipped
    with a warning, since there is no group to place them in.

    Returns
    -------
    DataFrame with columns ['window_unit', 'population', 'bin_start',
    'bin_end', 'bin_mid', 'mean_r2', 'n_pairs'], one row per non-empty
    distance bin per (window_unit, population) group. Empty (0 rows, same
    columns) if `data_dir` doesn't exist, contains no CSV files, or every
    CSV found is empty/malformed/missing the group columns.
    """
    empty = pd.DataFrame(columns=_GROUP_COLUMNS + _DECAY_COLUMNS)
    if not os.path.isdir(data_dir):
        print(f"[LD_decay_plot] average_ld_decay_data: '{data_dir}' doesn't exist - "
              f"nothing to average.")
        return empty

    csv_paths = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    if not csv_paths:
        print(f"[LD_decay_plot] average_ld_decay_data: '{data_dir}' exists but contains no "
              f"CSV files - nothing to average.")
        return empty

    required = set(_GROUP_COLUMNS) | set(_DECAY_COLUMNS)
    frames = []
    n_empty = 0
    n_missing_cols = 0
    n_legacy = 0
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            n_empty += 1
            continue
        if df.shape[0] == 0:
            # A real, expected outcome for a scenario where
            # compute_ld_decay_data() itself found nothing to plot (e.g. no
            # marker pair fell within the configured distance window, or
            # too few markers had a real CHR/POS/CM) - save_ld_decay_data()
            # still writes a header-only CSV for the record (see its own
            # docstring), so this is NOT malformed data, just an empty
            # scenario. Counted separately from n_missing_cols below so the
            # summary at the end of this function can tell the two apart.
            n_empty += 1
            continue
        if 'window_unit' not in df.columns and {'bin_start_kb', 'bin_end_kb', 'bin_mid_kb'}.issubset(df.columns):
            # A per-scenario CSV written by a pre-multi-unit version of
            # this module (kb-only, no 'window_unit' metadata column) -
            # can't be safely mixed with the current, unit-aware format,
            # so it's skipped rather than guessed-at. Re-run this
            # scenario to get a current-format CSV.
            n_legacy += 1
            continue
        if not required.issubset(df.columns):
            n_missing_cols += 1
            print(f"[LD_decay_plot] average_ld_decay_data: skipping '{path}' - missing "
                  f"'window_unit'/'population' metadata column(s) needed to group it correctly.")
            continue
        frames.append(df[_GROUP_COLUMNS + _DECAY_COLUMNS])

    if not frames:
        print(
            f"[LD_decay_plot] average_ld_decay_data: found {len(csv_paths)} CSV file(s) under "
            f"'{data_dir}', but none had usable data ({n_empty} empty/header-only, "
            f"{n_missing_cols} missing window_unit/population columns, {n_legacy} "
            f"in an older pre-multi-unit format) - nothing to average. An empty per-scenario "
            f"CSV usually means no marker pair fell within the configured distance window, or "
            f"fewer than 2 markers matched a real CHR/POS/CM in the SNP info file for that "
            f"scenario - see the '[LD_decay_plot] compute_ld_decay_data: ...' / '[GP] LD decay "
            f"plot skipped for this scenario ...' lines earlier in this run's log for the "
            f"specific reason."
        )
        return empty

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=['mean_r2', 'n_pairs'])
    combined = combined[combined['n_pairs'] > 0]
    if combined.shape[0] == 0:
        return empty

    combined['_weighted_r2'] = combined['mean_r2'] * combined['n_pairs']
    group_cols = _GROUP_COLUMNS + ['bin_start', 'bin_end', 'bin_mid']
    grouped = combined.groupby(group_cols, as_index=False, dropna=False).agg(
        _weighted_r2_sum=('_weighted_r2', 'sum'),
        n_pairs=('n_pairs', 'sum'),
    )
    grouped['mean_r2'] = grouped['_weighted_r2_sum'] / grouped['n_pairs']
    grouped = grouped[_GROUP_COLUMNS + _DECAY_COLUMNS]
    grouped = grouped.sort_values(_GROUP_COLUMNS + ['bin_start']).reset_index(drop=True)
    return grouped


def average_and_plot_ld_decay(result_name: str, subfolder: str = DEFAULT_SUBFOLDER,
                               data_dirname: str = DEFAULT_DATA_DIRNAME,
                               keep_log: "bool | None" = None) -> int:
    """Requirement 6: once every prediction scenario in a run has
    finished, average every per-scenario decay CSV under
    Result/<result_name>/<subfolder>/<data_dirname>/, separately for each
    (window_unit, population) combination present, and plot each group's
    curve as its own PNG:

        Result/<result_name>/<subfolder>/LD_decay_average_<unit>_<population>.png
        Result/<result_name>/<subfolder>/LD_decay_average_<unit>_<population>.csv

    (the .csv is the same "data used to create the plot" Requirement 5
    already asks for at the per-scenario level, just applied here too).
    NOT split further by phenotype - LD decay reflects which individuals
    are in a scenario's training set, which never depends on phenotype
    (see genomic_prediction.py's GP(), whose train/test split is keyed
    only on population/ratio/replicate) - only population, which DOES
    have a real, biologically meaningful effect on LD decay. In addition,
    a single combined Result/<result_name>/<subfolder>/LD_decay_average_all.csv
    is written with every group's averaged rows stacked together
    (window_unit/population columns included), purely for convenience -
    nothing reads it back in.

    keep_log : whether to leave the per-scenario decay data CSVs (under
        <subfolder>/<data_dirname>/) on disk after averaging them, or
        delete them now that they've served their purpose - freeing disk
        space for runs with many scenarios. The per-scenario PNG PLOTS
        under <subfolder>/ (population/replicate-level) are NEVER deleted
        by this, regardless of keep_log - only the underlying numeric CSV
        "log" data is optional; the per-population average (both PNG and
        CSV) is also never deleted, since it's the small, already-
        summarised final result, not the per-scenario "log". If None (the
        default), the setting each GP() call recorded via
        Preprocess.LD_decay_plot.write_keep_log_marker() is looked up
        automatically (falling back to True - keep everything - if no
        such marker exists, e.g. the feature wasn't enabled for this run
        at all) - this is what every caller in this codebase relies on
        (run_sequential.py / run_step2_assemble.py / main_app.py never
        pass this explicitly), since it lets a single GUI setting control
        the outcome for both the Sequential and Parallel (Step 2 - which
        never sees Step 1's LD_PRUNE config directly) workflows without
        any extra config plumbing between them.

    Safe to call unconditionally, every run, regardless of whether the LD
    decay plot feature was ever enabled (Requirement 7) - see
    run_sequential.py / run_step2_assemble.py / main_app.py, which all
    call this after every run/assemble step without checking first. Logs
    an informational message and returns 0 (nothing to do) rather than
    raising when no per-scenario data exists.

    Returns
    -------
    int : the number of (window_unit, population) average plots actually
    written (0 if none).
    """
    plot_dir, data_dir = result_paths(result_name, subfolder, data_dirname)

    averaged = average_ld_decay_data(data_dir)
    if averaged.shape[0] == 0:
        # average_ld_decay_data() itself already prints the specific reason
        # (no directory / no CSVs / every CSV empty or malformed) - nothing
        # further to add here.
        return 0

    os.makedirs(plot_dir, exist_ok=True)
    averaged.to_csv(os.path.join(plot_dir, 'LD_decay_average_all.csv'), index=False)

    n_plotted = 0
    for (window_unit, population), group_df in averaged.groupby(_GROUP_COLUMNS, dropna=False):
        group_decay = group_df[_DECAY_COLUMNS].sort_values('bin_start').reset_index(drop=True)
        name_stub = (
            f"LD_decay_average_{sanitize_for_filename(window_unit)}_{sanitize_for_filename(population)}"
        )
        csv_path = os.path.join(plot_dir, f'{name_stub}.csv')
        png_path = os.path.join(plot_dir, f'{name_stub}.png')
        group_decay.to_csv(csv_path, index=False)
        plotted = plot_ld_decay(
            group_decay, png_path,
            title=(f'LD decay ({_UNIT_LABEL.get(window_unit, window_unit)}) - average across '
                   f'replicates and phenotypes\npopulation={population}'),
            window_unit=window_unit,
        )
        if plotted:
            n_plotted += 1

    if keep_log is None:
        keep_log = read_keep_log_marker(plot_dir)
        if keep_log is None:
            keep_log = True  # safe default: never delete data unless explicitly told to
    if not keep_log:
        n_removed = _clear_directory(data_dir)
        print(f"[LD_decay_plot] keep_log is False - removed {n_removed} per-scenario LD decay "
              f"data CSV file(s) under '{data_dir}' now that the average(s) above have been "
              f"computed from them. The per-scenario PNG plots under '{plot_dir}' (population/"
              f"replicate-level) and the per-population average (PNG + CSV) were kept either way.")

    return n_plotted
