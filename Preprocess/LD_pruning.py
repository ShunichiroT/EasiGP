"""
ld_pruning.py
=============

LD-based SNP pruning for 0/1/2-coded genotype matrices, using PLINK2 for the
heavy lifting.

--------------------------------------------------------------------------
WHY THIS DESIGN
--------------------------------------------------------------------------
PLINK2's `--indep-pairwise` natively supports two window types:

    * variant-count windows :  --indep-pairwise 50 5 0.2
    * physical (kb) windows :  --indep-pairwise 500kb 1 0.2

It does NOT support a centimorgan-based window for `--indep-pairwise`
(verified against the current plink2 manual -- only the 'kb' modifier
exists). Because 1 cM does not correspond to a fixed number of base pairs
(recombination rate varies a lot across the genome, e.g. centromeres),
faking a cM window with a generous bp window is unsafe -- it can silently
miss pairs in low-recombination regions.

So this module gives you two real, honest code paths:

    window_unit="kb"       -> full PLINK2 pipeline (fast, scales to millions
                               of variants, exactly what you sketched).
    window_unit="variants" -> full PLINK2 pipeline, variant-count window.
    window_unit="cm"       -> PLINK2 still builds the bed/bim/fam, but the
                               actual pruning is done with a small greedy
                               sliding-window algorithm in NumPy, using the
                               genetic-distance (cM) column you supply. This
                               is a simplified re-implementation of PLINK's
                               greedy pruning logic -- not a PLINK internal
                               call -- and is only practical for up to
                               ~100-200k variants at a time. If you need cM
                               pruning on genome-scale data, precompute a
                               kb-equivalent window per chromosome from your
                               genetic map instead.

--------------------------------------------------------------------------
EXPECTED INPUTS
--------------------------------------------------------------------------
genotype_df : pandas.DataFrame
    Samples x SNPs. Index = sample IDs. Columns = SNP IDs.
    Values in [0, 2], NaN allowed for missing. 0/2 = homozygous, 1 =
    heterozygous. Values strictly between the integers (e.g. 0.5, 1.5) are
    tolerated -- this is common in RIL/NAM-style populations where calls
    are expressed as a dosage/probability rather than a hard genotype --
    and are rounded to the nearest hardcall before PLINK ever sees them
    (see `round_dosage` on ld_prune_snps). If you'd rather PLINK never see
    rounded data, round/impute upstream yourself and pass round_dosage=False.

snp_info : pandas.DataFrame
    Indexed by SNP ID (must contain every column in genotype_df.columns).
    Required columns:
        CHR : chromosome code (int or str)
        POS : base-pair position (int)
    Optional columns:
        CM  : genetic position in centimorgans (required if window_unit="cm")
        A1  : reference/major allele (str, single character recommended)
        A2  : alternate/minor allele (str, single character recommended)
    If A1/A2 are omitted, placeholder alleles "A"/"T" are used for every
    SNP. That's fine for LD pruning (only genotype *state*, not allele
    identity, drives r^2) but do NOT reuse the intermediate PED/BED files
    this function writes for anything allele-identity-sensitive (merging
    with other datasets, strand checks, etc.) unless you supply real
    alleles.

--------------------------------------------------------------------------
OUTPUT
--------------------------------------------------------------------------
A DataFrame with the same index (samples), same dtypes, and the same
column ORDER as genotype_df, restricted to the columns that survived
pruning. No PED/MAP round-trip is used to build this -- it's a direct
pandas column selection driven by the SNP IDs PLINK (or the cM fallback)
decided to keep, so there is no risk of allele re-encoding or sample
reordering creeping into your data.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pipeline_utils import unify_columns_by_position, get_active_compute_resources
from Preprocess import ld_kernels

logger = logging.getLogger(__name__)


def _resolve_plink_threads(plink_threads):
    """Resolve the `--threads N` value every plink2 subprocess call below
    should use - mirrors `Preprocess/plink_io.py::_resolve_plink_threads()`
    exactly (kept as a local copy rather than imported, since the two
    modules have no other dependency on each other and this keeps each
    independently usable).

    An explicit `plink_threads` argument always wins; otherwise falls back
    to the run's shared compute-resource settings (`PLINK_THREADS` config
    key via `get_active_compute_resources()`).

    Without this, every plink2 invocation below left `--threads` unset
    entirely, so plink2 fell back to its own hardware auto-detection -
    which reads the number of processors VISIBLE to the machine, not the
    number actually reserved for this job/cgroup. On a node where only a
    handful of CPUs were reserved (e.g. a GPU job, which typically
    requests far fewer CPUs than a CPU-only job of the same size), plink2
    would still try to spawn threads sized to the whole node, causing
    heavy oversubscription/context-switch thrashing instead of a
    genuine speed-up - the root cause of LD pruning taking far longer on
    a GPU allocation than on an equivalently-sized CPU-only one, even
    though LD pruning itself never touches the GPU.
    """
    if plink_threads is not None:
        return max(1, int(plink_threads))
    return get_active_compute_resources()['plink_threads']


class PlinkError(RuntimeError):
    """Raised when a PLINK2 subprocess call returns a non-zero exit code."""


class LDPruneInputError(ValueError):
    """Raised when genotype_df / snp_info fail validation."""


# --------------------------------------------------------------------------
# PATCH_NOTES (performance fix): pre-flight cost guard for a 'kb'-window
# --indep-pairwise call, run BEFORE plink2 is ever invoked.
#
# WHY THIS EXISTS: --indep-pairwise's own cost scales with how many OTHER
# markers fall inside each marker's window, not with marker count alone -
# a "kb" window over a dense, genome-wide marker set can silently imply an
# enormous per-marker comparison count (a wide window over millions of
# dense markers can mean hundreds of thousands of neighbours per window),
# turning what looks like an ordinary LD-pruning config into a
# multi-hour-to-multi-day plink2 call with NO warning beforehand - the
# exact failure mode tracked down from a production run whose LD pruning
# step alone ran for ~12 hours (window=10000kb over ~3.2M dense markers),
# consuming the run's entire HPC walltime allocation before a single model
# had finished tuning.
#
# This is a closed-form, DENSITY-based estimate (marker count / genomic
# span, per chromosome) - it costs a fraction of a second (the .bim /
# snp_info file is already being read for this call regardless) and needs
# no plink2 subprocess of its own, unlike an actual calibration run. It is
# deliberately NOT a runtime prediction (marker density varies a lot
# locally - clustering, centromeres, structural variation - and this
# codebase has no reliable, hardware/PLINK-version-independent throughput
# constant to convert a comparison count into a wall-clock estimate without
# risking a confidently wrong number) - it reports the "average markers
# per window" figure itself, which is directly, monotonically responsible
# for the cost, and lets the person judge against typical practice.
# --------------------------------------------------------------------------

# Typical published LD-pruning configs (array/GWAS-density data) keep the
# realised "markers per window" figure in the tens to low hundreds - this
# default trip-wire is set an order of magnitude above that, so it only
# fires for configurations that are genuinely far outside normal practice
# (informational only; raise/lower via the warn_avg_markers_per_window
# parameter, or ld_config['warn_avg_markers_per_window']).
DEFAULT_WARN_AVG_MARKERS_PER_WINDOW = 5000


def estimate_kb_window_marker_load(chrom, pos, window):
    """Closed-form, density-based estimate of how many OTHER markers fall,
    on average, inside a `window`-kb --indep-pairwise window, from marker
    positions alone - no plink2 subprocess involved.

    Parameters
    ----------
    chrom : array-like
        Per-marker chromosome labels (any hashable type - compared as str).
    pos : array-like
        Per-marker base-pair positions, same length/order as `chrom`.
    window : float
        The 'kb' window size (as passed to --indep-pairwise's own
        <window>kb argument).

    Returns
    -------
    dict, or None if there isn't enough position information to compute a
    density (fewer than 2 positioned markers on any chromosome - e.g. an
    all-NaN/placeholder map), with keys:
        n_markers, n_chromosomes         : int
        avg_markers_per_window           : float, marker-count-weighted
                                            mean across chromosomes
        max_markers_per_window           : float, the single densest
                                            chromosome's own figure
        est_total_comparisons            : float, n_markers *
                                            avg_markers_per_window - an
                                            ORDER-OF-MAGNITUDE indicator
                                            of total pairwise work, not a
                                            literal operation count.
    Never raises - a caller with degenerate input (e.g. every position
    identical or missing) just gets None back, same as "nothing to warn
    about" from this function's point of view.
    """
    df = pd.DataFrame({
        'chrom': pd.Series(chrom).astype(str).reset_index(drop=True),
        'pos': pd.to_numeric(pd.Series(pos).reset_index(drop=True), errors='coerce'),
    }).dropna(subset=['pos'])
    if df.empty:
        return None

    window_bp = float(window) * 1000.0
    per_chrom = df.groupby('chrom')['pos'].agg(n='count', lo='min', hi='max')
    per_chrom = per_chrom[per_chrom['n'] >= 2]
    if per_chrom.empty:
        return None

    span_bp = (per_chrom['hi'] - per_chrom['lo']).clip(lower=1.0)
    density_per_bp = per_chrom['n'] / span_bp
    markers_per_window = density_per_bp * window_bp

    n_markers = int(df.shape[0])
    weights = per_chrom['n'] / n_markers
    avg_markers_per_window = float((markers_per_window * weights).sum())

    return {
        'n_markers': n_markers,
        'n_chromosomes': int(per_chrom.shape[0]),
        'avg_markers_per_window': avg_markers_per_window,
        'max_markers_per_window': float(markers_per_window.max()),
        'est_total_comparisons': avg_markers_per_window * n_markers,
    }


def check_kb_window_cost(chrom, pos, window, *, threads=1,
                          max_avg_markers_per_window=None,
                          warn_avg_markers_per_window=DEFAULT_WARN_AVG_MARKERS_PER_WINDOW,
                          context=''):
    """Print a loud, actionable diagnostic (see this module's own "Diagnostics
    as a design principle" convention elsewhere in this codebase) - and,
    if `max_avg_markers_per_window` is set and exceeded, raise BEFORE
    plink2 is ever invoked - when a 'kb'-window --indep-pairwise call is
    about to run against a marker density that makes it likely to be
    extremely slow. See `estimate_kb_window_marker_load()`'s own docstring
    for exactly what is estimated and why no wall-clock time is quoted.

    A no-op (returns immediately) whenever the load can't be estimated
    (see `estimate_kb_window_marker_load`) - this is a best-effort safety
    net, not a required precondition.

    Parameters
    ----------
    chrom, pos : see estimate_kb_window_marker_load.
    window : float, the 'kb' window size in effect.
    threads : int, purely for the printed message (how many plink2
        --threads are already in play - multi-threading helps, but does
        not change the fundamental per-marker workload being reported).
    max_avg_markers_per_window : float, optional
        HARD cap (opt-in, default None = no cap, i.e. no behaviour change
        for any existing config). When set and the estimated average
        exceeds it, raises `LDPruneInputError`/`PlinkFilesetError`-style
        (a plain ValueError here - callers with their own exception type,
        e.g. Preprocess/plink_io.py, catch and re-raise as their own)
        BEFORE starting plink2, so a misconfigured HPC job fails in
        seconds rather than after exhausting a walltime allocation.
    warn_avg_markers_per_window : float
        Trip-wire for the (non-fatal) warning printout - see
        DEFAULT_WARN_AVG_MARKERS_PER_WINDOW's own docstring comment.
    context : str, optional
        Short caller-identifying string (e.g. " [GAT_biological_prior_knowledge
        data-driven merge]") inserted into the printed/raised message, so
        multiple LD-pruning call sites in one run (the 'other models' pool
        vs. a bio-prior instance's own data-driven-merge pool, say) are
        distinguishable in the log.

    Returns
    -------
    dict or None
        Whatever `estimate_kb_window_marker_load()` returned (so a caller
        can log/reuse it further), or None if it couldn't be computed.
    """
    load = estimate_kb_window_marker_load(chrom, pos, window)
    if load is None:
        return None

    avg = load['avg_markers_per_window']
    if max_avg_markers_per_window is not None and avg > max_avg_markers_per_window:
        raise ValueError(
            f"[LD-prune cost guard]{context} refusing to start --indep-pairwise: a {window}kb "
            f"window against this marker density implies an estimated average of "
            f"~{avg:,.0f} OTHER markers inside every marker's window (out of {load['n_markers']:,} "
            f"markers total across {load['n_chromosomes']} chromosome(s), densest chromosome "
            f"~{load['max_markers_per_window']:,.0f} markers/window) - on the order of "
            f"{load['est_total_comparisons']:.2e} pairwise comparisons overall. This exceeds the "
            f"configured safety cap of {max_avg_markers_per_window:,.0f} average markers/window "
            f"(ld_config['max_avg_markers_per_window']). Typical LD-pruning windows keep this figure "
            f"in the tens to low hundreds. Likely fixes: reduce 'window' (e.g. to a few hundred kb "
            f"or less), switch window_unit to 'variants' (which bounds the per-window cost directly, "
            f"regardless of physical marker density), or raise/remove max_avg_markers_per_window if "
            f"this cost is genuinely intended."
        )

    if avg > warn_avg_markers_per_window:
        print(
            f"[LD_pruning] LD-PRUNE COST WARNING{context}: a {window}kb window against this marker "
            f"density implies an estimated average of ~{avg:,.0f} OTHER markers inside every "
            f"marker's window (out of {load['n_markers']:,} markers total across "
            f"{load['n_chromosomes']} chromosome(s), densest chromosome ~{load['max_markers_per_window']:,.0f} "
            f"markers/window) - on the order of {load['est_total_comparisons']:.2e} pairwise "
            f"comparisons overall. This is far outside typical LD-pruning practice (usually tens to "
            f"low hundreds of markers per window) and can take many hours to complete even "
            f"multi-threaded ('--threads {threads}' is already applied here). If this is not "
            f"intended: reduce 'window', or switch window_unit to 'variants' (bounds the per-window "
            f"cost directly, regardless of physical marker density). To turn this into a hard, "
            f"fail-fast error next time instead of burning compute, set "
            f"ld_config['max_avg_markers_per_window'] (e.g. 5000-20000)."
        )
    return load


# --------------------------------------------------------------------------
# Optional MAF (minor allele frequency) filtering - a separate, simpler
# pre-filter that can run alongside LD pruning. Pure NumPy/pandas (no
# PLINK dependency), computed directly from dosage genotypes, so it works
# identically regardless of window_unit ('kb'/'cm'/'variants'). Applied
# inside ld_prune_snps() BEFORE the plink2-dependent pruning step itself
# runs, but if that later step then fails (e.g. plink2 isn't installed -
# see PlinkError), the whole call still raises rather than returning a
# MAF-filtered-but-not-LD-pruned partial result - see LD_pruning()'s own
# comment on why a plink2 failure must stop the run, not silently return
# something other than what was actually requested.
# --------------------------------------------------------------------------

def compute_maf(genotype_df: pd.DataFrame) -> pd.Series:
    """Minor allele frequency per marker (column), computed directly from
    0/1/2(.x) dosage genotypes: allele frequency = mean(dosage) / 2,
    folded to the minor allele (so the result is always <= 0.5) via
    min(p, 1-p). Missing calls (NaN) are ignored when computing the mean
    (pandas' default skipna behaviour) - a marker with EVERY sample
    missing has no defined frequency and gets NaN here, not 0."""
    p = genotype_df.mean(axis=0, skipna=True) / 2.0
    return p.where(p <= 0.5, 1.0 - p)


def maf_filter(genotype_df: pd.DataFrame, maf_threshold: float) -> list:
    """Marker (column) names in genotype_df that SURVIVE MAF filtering -
    i.e. minor allele frequency >= maf_threshold. `maf_threshold` <= 0
    disables filtering entirely (every column is kept, without even
    computing MAF) - the same "0/None means off" convention as
    r2_threshold's own use elsewhere in this module.

    Markers with an undefined MAF (every sample missing for that column -
    see compute_maf) are KEPT, not dropped: "we don't know its frequency"
    is a data-completeness problem, not evidence the marker is rare, and
    conflating the two would silently discard markers for the wrong
    reason. Missing-data handling elsewhere in the pipeline (dosage
    imputation, model-level NaN handling) is where that should be
    addressed instead.
    """
    if maf_threshold is None or maf_threshold <= 0:
        return list(genotype_df.columns)
    maf = compute_maf(genotype_df)
    keep = maf.isna() | (maf >= maf_threshold)
    return list(genotype_df.columns[keep.to_numpy()])


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def _validate_inputs(
    genotype_df: pd.DataFrame,
    snp_info: pd.DataFrame,
    window_unit: str,
) -> pd.DataFrame:
    if window_unit not in ("kb", "variants", "cm"):
        raise LDPruneInputError(
            f"window_unit must be one of 'kb', 'variants', 'cm' (got {window_unit!r})"
        )
    if snp_info is not None:
        snp_info = snp_info.copy()
    else:
        snp_info = pd.DataFrame()
    for col in ("CHR", "POS", "CM"):
        if col not in snp_info.columns:
            snp_info[col] = np.nan
    
    missing_snps = set(genotype_df.columns) - set(snp_info.index)
    if missing_snps:
        logger.info(
            "%d SNP(s) have no entry in snp_info at all (e.g. %s) -- treating them as "
            "unmapped rather than erroring.",
            len(missing_snps), list(missing_snps)[:5],
        )
        filler = pd.DataFrame(index=list(missing_snps), columns=snp_info.columns)
        snp_info = pd.concat([snp_info, filler])

    # Reindex snp_info to genotype_df's column order so everything downstream
    # can assume aligned, 1:1 ordering.
    snp_info = snp_info.loc[genotype_df.columns].copy()

    if "A1" not in snp_info.columns:
        snp_info["A1"] = np.nan
    if "A2" not in snp_info.columns:
        snp_info["A2"] = np.nan
    n_missing_allele = int(snp_info["A1"].isna().sum())
    if n_missing_allele:
        logger.warning(
            "%d SNP(s) have no A1/A2 alleles -- using placeholder alleles 'A'/'T' for them. "
            "Fine for LD pruning, but don't reuse the intermediate PED/BED files for anything "
            "allele-identity-sensitive unless you supply real alleles.",
            n_missing_allele,
        )
    snp_info["A1"] = snp_info["A1"].fillna("A")
    snp_info["A2"] = snp_info["A2"].fillna("T")

    stacked = genotype_df.stack().dropna()
    if not stacked.empty and not ((stacked >= 0) & (stacked <= 2)).all():
        raise LDPruneInputError(
            "genotype_df must contain only values in [0, 2] (NaN allowed for missing). "
            f"Found values outside that range, e.g. min={stacked.min()}, max={stacked.max()}."
        )

    return snp_info


# --------------------------------------------------------------------------
# Step 1: 0/1/2 matrix -> PED/MAP
# --------------------------------------------------------------------------

def _write_ped_map(
    genotype_df: pd.DataFrame,
    snp_info: pd.DataFrame,
    out_prefix: Path,
    round_dosage: bool = True,
) -> None:
    """Write PLINK1 text-format .ped and .map files from a 0/1/2(.x) matrix.

    PED only understands hard genotype calls (0/1/2). If round_dosage=True
    (default), any non-integer dosage values (e.g. 0.5, 1.5 -- common in
    RIL/NAM populations with uncertain calls) are rounded to the nearest
    integer with banker's rounding (0.5->0, 1.5->2) before being written.
    This is a lossy step -- it discards the uncertainty in those calls --
    so if that matters for your analysis, resolve it upstream (e.g. by
    imputing or hard-calling with your own rule) and pass round_dosage=False
    to make this function reject non-integer input instead of silently
    rounding it.
    """
    n_samples, n_snps = genotype_df.shape

    # MAP: CHR  SNP_ID  CM  POS   (4-column format; CM=0 if not supplied)
    cm_col = snp_info["CM"] if "CM" in snp_info.columns else 0
    map_df = pd.DataFrame(
        {
            "CHR": snp_info["CHR"].values,
            "SNP_ID": snp_info.index.values,
            "CM": cm_col if isinstance(cm_col, int) else cm_col.values,
            "POS": snp_info["POS"].values,
        }
    )
    map_df.to_csv(f"{out_prefix}.map", sep="\t", header=False, index=False)
    
    # PED genotype columns, vectorized.
    geno = genotype_df.to_numpy(dtype=float)  # shape (n_samples, n_snps), NaN-capable

    non_integer = np.nansum(geno != np.round(geno))
    if non_integer > 0:
        if not round_dosage:
            raise LDPruneInputError(
                f"genotype_df contains {int(non_integer)} non-integer dosage value(s) "
                "(e.g. 0.5, 1.5) and round_dosage=False. Either pass round_dosage=True "
                "or hard-call/round the data yourself before calling this function."
            )
        logger.warning(
            "%d / %d genotype calls are non-integer dosages (e.g. 0.5, 1.5) -- "
            "rounding to the nearest hardcall before writing PED. This discards "
            "call-uncertainty information; see the round_dosage docstring.",
            int(non_integer), geno.size,
        )
        geno = np.round(geno)  # banker's rounding: 0.5->0, 1.5->2
    a1 = snp_info["A1"].to_numpy().astype(object)
    a2 = snp_info["A2"].to_numpy().astype(object)
    a1_b = np.broadcast_to(a1, geno.shape)
    a2_b = np.broadcast_to(a2, geno.shape)

    allele1 = np.where(geno == 2, a2_b, a1_b).astype(object)
    allele2 = np.where(geno == 0, a1_b, a2_b).astype(object)
    is_missing = np.isnan(geno)
    allele1[is_missing] = "0"
    allele2[is_missing] = "0"

    geno_block = np.empty((n_samples, n_snps * 2), dtype=object)
    geno_block[:, 0::2] = allele1
    geno_block[:, 1::2] = allele2

    fid = (genotype_df.index.to_numpy() + 1).astype(str)
    iid = fid
    # PERFORMANCE FIX: these four "family-file" columns are always the
    # same constant values (0/0/-9/-9) for every sample and every call --
    # write them as the literal PED-format STRINGS directly, rather than
    # as bare Python ints (0, -9) that then had to be converted to text
    # later. Doing that conversion here, once, for four narrow columns is
    # nothing; doing it below, per SAMPLE ROW, for the *entire* row width
    # (6 + 2*n_snps columns) was pure repeated work -- see the write loop
    # below for why that mattered.
    pat = np.full(n_samples, "0", dtype=object)
    mat = np.full(n_samples, "0", dtype=object)
    sex = np.full(n_samples, "-9", dtype=object)
    pheno = np.full(n_samples, "-9", dtype=object)

    ped_left = np.column_stack([fid, iid, pat, mat, sex, pheno]).astype(object)
    ped_full = np.hstack([ped_left, geno_block])

    # PERFORMANCE FIX: every element of ped_full is now already a genuine
    # Python `str` (allele1/allele2/pat/mat/sex/pheno were all built as
    # strings above; fid/iid via .astype(str)/.astype(object)) -- so the
    # per-row `row.astype(str)` this loop used to do before joining was
    # pure wasted work: a full elementwise dtype conversion, repeated for
    # EVERY sample, over the entire (6 + 2*n_snps)-wide row, that would
    # have been a no-op even if it had been skipped entirely. Building the
    # whole text blob once and writing it in a single call (instead of one
    # `fh.write()` per sample) also cuts the per-row Python/IO overhead
    # that dominates when n_samples is large. The on-disk content is
    # byte-for-byte identical to the previous implementation.
    with open(f"{out_prefix}.ped", "w") as fh:
        fh.write("\n".join(" ".join(row) for row in ped_full))
        if n_samples:
            fh.write("\n")

    logger.info("Wrote %s.ped (%d samples x %d SNPs) and %s.map", out_prefix, n_samples, n_snps, out_prefix)


# --------------------------------------------------------------------------
# PLINK2 subprocess wrapper
# --------------------------------------------------------------------------

def _run_plink(cmd: list, log_prefix: Path, heartbeat_seconds: Optional[float] = None,
               heartbeat_label: Optional[str] = None) -> subprocess.CompletedProcess:
    """PATCH_NOTES (visibility fix): `heartbeat_seconds`/`heartbeat_label`
    mirror `Preprocess/plink_io.py::_run_plink()`'s own parameters of the
    same name (kept as a local copy rather than imported - see
    `_resolve_plink_threads()`'s own docstring note on why these two
    modules duplicate small helpers rather than depend on each other) -
    print a "still running" line every `heartbeat_seconds` while a
    potentially long, silent `--indep-pairwise` call is in flight, so it
    reads as "slow but alive" rather than "hung" (see that other
    `_run_plink()`'s own docstring for the full rationale)."""
    logger.info("Running: %s", " ".join(cmd))
    if heartbeat_seconds is None:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError as e:
            raise PlinkError(
                f"Could not find/execute {cmd[0]!r}. Make sure plink2 is installed and on "
                f"PATH, or pass plink_path='/full/path/to/plink2'. ({e})"
            ) from e
    else:
        label = heartbeat_label or os.path.basename(str(cmd[0]))
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except FileNotFoundError as e:
            raise PlinkError(
                f"Could not find/execute {cmd[0]!r}. Make sure plink2 is installed and on "
                f"PATH, or pass plink_path='/full/path/to/plink2'. ({e})"
            ) from e
        start = time.time()
        while True:
            try:
                stdout, stderr = proc.communicate(timeout=heartbeat_seconds)
                break
            except subprocess.TimeoutExpired:
                elapsed_min = (time.time() - start) / 60.0
                print(
                    f"[LD_pruning] ... still running {label} (elapsed {elapsed_min:.1f} min, PID "
                    f"{proc.pid}) - this can be expected for a large window/dense marker set (see "
                    f"any 'LD-PRUNE COST WARNING' printed above); no action needed unless this "
                    f"keeps growing far beyond what that warning implied."
                )
        result = subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
    with open(f"{log_prefix}.pylog", "w") as fh:
        fh.write("CMD: " + " ".join(cmd) + "\n\nSTDOUT:\n" + result.stdout + "\n\nSTDERR:\n" + result.stderr)
    if result.returncode != 0:
        tail = "\n".join(result.stderr.splitlines()[-25:])
        raise PlinkError(
            f"Command failed (exit {result.returncode}): {' '.join(cmd)}\n"
            f"--- stderr tail ---\n{tail}\n"
            f"(full log: {log_prefix}.pylog)"
        )
    return result


# --------------------------------------------------------------------------
# cM-based fallback pruning (pure NumPy, PLINK-style greedy sliding window)
# --------------------------------------------------------------------------

def _pairwise_r2(x: np.ndarray, y: np.ndarray) -> float:
    """r^2 between two genotype vectors, ignoring samples missing in either.

    ver4-4 R4.g: delegates to the shared ``ld_kernels.pairwise_r2()`` core
    (promoted out of this file, and out of
    ``Preprocess/LD_decay_plot.py``'s own byte-for-byte duplicate of it) -
    ``degenerate=0.0`` preserves THIS module's own existing sentinel for a
    degenerate pair exactly (``LD_decay_plot.py``'s own thin wrapper uses
    ``degenerate=float('nan')`` instead - the two modules' conventions are
    deliberately NOT unified; see ``ld_kernels.pairwise_r2()``'s own
    docstring)."""
    return ld_kernels.pairwise_r2(x, y, degenerate=0.0)


def _cm_prune(
    genotype_df: pd.DataFrame,
    snp_info: pd.DataFrame,
    window_cm: float,
    r2_threshold: float,
    n_jobs: Optional[int] = None,
    device: Optional[str] = None,
) -> list:
    """
    Greedy left-to-right pruning within a sliding cM window, per chromosome.

    Unlike the PLINK/PED path, this uses genotype_df's raw values directly
    (Pearson r^2 handles continuous dosage values like 0.5/1.5 fine), so no
    rounding is applied here -- dosage information isn't lost in this path.

    For each retained SNP i (in cM order), compare it against subsequent
    SNPs within `window_cm`. Any SNP whose r^2 with SNP i exceeds the
    threshold is dropped. This mirrors the *spirit* of PLINK's
    --indep-pairwise (greedy, left-to-right, single retained "anchor" wins
    each comparison) but is not bit-identical to PLINK's internal
    implementation.

    ver4-4 R3.e (blueprint §2.3.2, invariant I7 - the closest invariant in
    this whole update): UNCHANGED semantics from the description above.
    Two internal changes only:

      1. Per chromosome, WHEN that chromosome's genotype block has no
         missing (``NaN``) calls at all, the within-window r^2 values are
         precomputed ONCE as a banded matrix
         (``ld_kernels.pairwise_r2_matrix(block, pairwise_complete=True)``
         - the SAME shared kernel R4.g introduces for GPU LD r^2), and the
         greedy left-to-right decision loop below simply looks values up
         in it instead of calling ``_pairwise_r2()`` per pair. This is a
         byte-for-byte-equivalent (verified to float-rounding tolerance,
         not merely reasoned about - see the ver4-4 Stage 5 design
         record's own test evidence) substitution: the DECISION SEQUENCE
         itself - which SNP is compared against which, in which order,
         and which threshold each comparison is checked against - is
         completely unchanged; only WHERE each r^2 value comes from
         changes (a precomputed matrix lookup instead of a fresh
         ``np.corrcoef`` call).
      2. WHEN a chromosome's genotype block contains ANY missing call,
         this falls back to the ORIGINAL, exact per-pair ``_pairwise_r2``
         path for that one chromosome (option (ii) from the blueprint's
         own R3.e chosen-approach text: "fall back to today's per-pair
         path whenever the chromosome block contains any NaN... the
         recommended first implementation") - a banded PRECOMPUTED matrix
         would otherwise require reproducing ``_pairwise_r2``'s own
         pairwise-DELETION missing-data handling per pair, which
         ``ld_kernels.pairwise_r2_matrix(..., pairwise_complete=True)``
         DOES support (see that function's own docstring) but at the cost
         of 4 full (n_markers x n_markers) matrix products per
         chromosome regardless of how sparse the actual within-window
         comparisons are - for a real, mostly-complete genotype matrix
         (the overwhelmingly common case) the no-NaN fast path above
         already captures the real win; this fallback keeps the NaN case
         simple and unambiguously correct rather than chasing a marginal
         extra speed-up on already-imperfect data.
      3. The ``for chrom, grp in snp_info.groupby("CHR")`` loop fans out
         across ``n_jobs`` PROCESSES (``joblib.Parallel``, the SAME
         backend/fallback-on-any-exception convention as
         ``pipeline_utils.parallel_shap_values`` - falls back to serial
         execution rather than raising) when ``n_jobs > 1`` and there is
         more than one chromosome group to fan out across. ``keep_ids``
         is then concatenated in the SAME chromosome order
         ``snp_info.groupby("CHR")`` itself produces (each chromosome's
         own worker call - serial or parallel - returns its OWN keep list
         in ORIGINAL within-chromosome order; ``joblib.Parallel`` always
         returns results in call-submission order, never completion
         order, so parallel fan-out cannot reorder the outer,
         cross-chromosome sequence either) - the returned list is
         therefore order-identical to ``n_jobs=1`` for any ``n_jobs``.

    Parallelising *within* a chromosome is explicitly forbidden - the
    greedy anchor order is load-bearing and a different order yields a
    different surviving marker set, silently changing every downstream
    result (I7). Every worker below processes one WHOLE chromosome
    independently, start to finish, with no shared mutable state.

    ``device`` (ver4-4 R4.g, added during the Stage 5 completion pass):
    forwarded to the no-NaN fast path's own
    ``ld_kernels.pairwise_r2_matrix(..., device=...)`` call - ``None``
    (default) resolves from the run's shared compute-resource settings,
    gated on ``GPU_LD_R2`` exactly like ``LD_decay_plot.compute_ld_decay_data``
    gates it (``'cpu'`` when ``GPU_LD_R2`` is off, even on a CUDA-visible
    node) - see that function's own comment for why the gate has to live
    at the point ``device`` is resolved, not deeper in the call chain.
    Never changes the RESULT (the banded matrix is verified
    device-invariant to floating-point-rounding tolerance - see
    ``ld_kernels.pairwise_r2_matrix``'s own docstring and the R4.g
    acceptance criterion "`_cm_prune` returns an identical keep-list with
    `GPU_LD_R2` on and off"), only which hardware computes it.
    """
    geno_all = genotype_df.to_numpy(dtype=float)
    col_index = {snp_id: i for i, snp_id in enumerate(genotype_df.columns)}
    groups = list(snp_info.groupby("CHR"))

    def _process_one_chromosome(chrom, grp):
        grp_sorted = grp.sort_values("CM")
        ids = grp_sorted.index.to_numpy()
        cms = grp_sorted["CM"].to_numpy()
        n = len(ids)
        keep_mask = np.ones(n, dtype=bool)

        col_positions = np.array([col_index[snp_id] for snp_id in ids])
        block = geno_all[:, col_positions]  # this chromosome only, already in cM order
        block_has_nan = bool(np.isnan(block).any())
        if block_has_nan:
            r2_matrix = None
        else:
            r2_matrix = ld_kernels.pairwise_r2_matrix(block, device=_resolved_device, pairwise_complete=True)

        for i in range(n):
            if not keep_mask[i]:
                continue
            gi = None if r2_matrix is not None else geno_all[:, col_index[ids[i]]]
            j = i + 1
            while j < n and (cms[j] - cms[i]) <= window_cm:
                if keep_mask[j]:
                    if r2_matrix is not None:
                        r2 = r2_matrix[i, j]
                    else:
                        gj = geno_all[:, col_index[ids[j]]]
                        r2 = _pairwise_r2(gi, gj)
                    if r2 > r2_threshold:
                        keep_mask[j] = False
                j += 1

        n_dropped = int((~keep_mask).sum())
        logger.info("cM-prune chr %s: kept %d / %d SNPs (dropped %d)", chrom, keep_mask.sum(), n, n_dropped)
        if block_has_nan:
            print(f"[LD_pruning] NOTE: cM-prune chr {chrom}: missing genotype call(s) found "
                  f"among this chromosome's {n} marker(s) - used the exact per-pair pruning "
                  f"path for this chromosome (the banded precompute requires a complete block, "
                  f"per invariant I7 - see _cm_prune's own docstring).")
        return chrom, ids[keep_mask].tolist()

    _active_resources = get_active_compute_resources()
    _n_jobs = n_jobs if n_jobs is not None else _active_resources['n_jobs']
    # ver4-4 R4.g completion fix: resolved ONCE here (outside
    # _process_one_chromosome, mirroring _n_jobs immediately above) so
    # every chromosome - serial or process-fanned-out - uses the SAME
    # device string; gated on GPU_LD_R2 exactly like
    # LD_decay_plot.compute_ld_decay_data (see that function's own
    # comment for why the gate belongs at THIS resolution point, not
    # deeper in the call chain, or GPU_LD_R2=false would never actually
    # disable the CUDA path on a CUDA-visible node).
    _resolved_device = device if device is not None else (
        _active_resources['device'] if _active_resources.get('gpu_ld_r2', True) else 'cpu'
    )
    if _n_jobs and _n_jobs != 1 and len(groups) > 1:
        try:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=_n_jobs)(
                delayed(_process_one_chromosome)(chrom, grp) for chrom, grp in groups
            )
        except Exception as exc:
            print(f"[LD_pruning] NOTE: parallel cM-prune fan-out across chromosomes failed "
                  f"({exc}) - falling back to serial execution (per-chromosome results are "
                  f"unaffected, only wall-clock time).")
            results = [_process_one_chromosome(chrom, grp) for chrom, grp in groups]
    else:
        results = [_process_one_chromosome(chrom, grp) for chrom, grp in groups]

    keep_ids: list = []
    for _chrom, ids_kept in results:
        keep_ids.extend(ids_kept)
    return keep_ids


# --------------------------------------------------------------------------
# Loader for NAM-style genotype CSVs
# --------------------------------------------------------------------------

def load_nam_genotype_csv(
    path: str,
    id_col: str = "ID",
    metadata_cols: tuple = ("population",),
    marker_prefix: Optional[str] = "i",
) -> tuple:
    """
    Load a NAM-style genotype CSV: one row per sample, sample ID + a few
    non-SNP metadata columns (e.g. 'population'), followed by marker
    columns (e.g. i0, i1, ..., i1105).

    This is already in the samples x SNPs orientation ld_prune_snps wants
    -- no transposition needed -- it just also carries metadata columns
    that have to be split off first, and its marker columns commonly hold
    dosage values (0, 0.5, 1, 1.5, 2) rather than pure hardcalls.

    Parameters
    ----------
    path          : path to the CSV file.
    id_col        : column to use as the sample index.
    metadata_cols : non-SNP columns to split off (kept in the returned
                    metadata DataFrame, dropped from genotype_df). Any
                    names not present in the file are ignored.
    marker_prefix : if given, only columns starting with this prefix are
                    treated as SNPs (guards against stray non-marker
                    columns being swept into the genotype matrix). Set to
                    None to instead treat every column that isn't id_col
                    or in metadata_cols as a SNP.

    Returns
    -------
    (genotype_df, metadata_df) : genotype_df is samples x SNPs, values in
    [0, 2] (dosages allowed); metadata_df holds the split-off non-SNP
    columns, indexed the same way.
    """
    df = pd.read_csv(path)
    if id_col not in df.columns:
        raise LDPruneInputError(f"id_col={id_col!r} not found in columns: {list(df.columns)[:10]}...")
    df = df.set_index(id_col)

    meta_present = [c for c in metadata_cols if c in df.columns]
    metadata_df = df[meta_present].copy()

    marker_cols = [c for c in df.columns if c not in meta_present]
    if marker_prefix is not None:
        marker_cols = [c for c in marker_cols if str(c).startswith(marker_prefix)]

    genotype_df = df[marker_cols].apply(pd.to_numeric, errors="coerce")
    logger.info(
        "Loaded %s: %d samples x %d markers (metadata columns: %s)",
        path, genotype_df.shape[0], genotype_df.shape[1], meta_present or "none",
    )
    return genotype_df, metadata_df


def make_placeholder_snp_info(
    genotype_df: pd.DataFrame,
    chrom=1,
    spacing_bp: int = 1000,
    cm_per_mb: float = 1.0,
) -> pd.DataFrame:
    """
    Build a minimal snp_info table when you don't have a real genetic/
    physical map for these markers -- e.g. the NAM CSV only gives you
    marker labels like i0, i1, ... with no CHR/POS/CM.

    THIS IS A PLACEHOLDER, not real genomic coordinates. It assumes the
    marker columns are already in genomic order (true for most NAM/RIL
    marker sets, which are typically exported in genetic-map order) and
    lays them out evenly spaced on one synthetic chromosome so PLINK's
    file formats are satisfied.

    Consequence for window_unit:
        * "variants" -- meaningful. Only depends on marker order, which
          this placeholder preserves faithfully from genotype_df.columns.
        * "kb" / "cm" -- NOT biologically meaningful with this placeholder,
          since the spacing/cM-per-Mb here is made up, not measured. Only
          use these once you've plugged in a real map (real CHR/POS, and
          real CM from a genetic map for this population).

    If you have the real NAM marker map (chromosome, bp position, cM),
    build snp_info from that instead of this function.
    """
    n = genotype_df.shape[1]
    pos = np.arange(n, dtype=np.int64) * spacing_bp + 1
    cm = pos / 1e6 * cm_per_mb
    logger.warning(
        "make_placeholder_snp_info: using SYNTHETIC positions (chrom=%s, %dbp spacing) "
        "for %d markers -- only window_unit='variants' is biologically meaningful here. "
        "Supply a real map for kb/cm-based pruning.",
        chrom, spacing_bp, n,
    )
    return pd.DataFrame(
        {"CHR": chrom, "POS": pos, "CM": cm, "A1": "A", "A2": "T"},
        index=genotype_df.columns,
    )


# --------------------------------------------------------------------------
# One pruning pass over a single partition of SNPs (all mapped, or all
# treated as variant-count-only) -- reused for the mapped/unmapped split
# --------------------------------------------------------------------------

def _prune_partition(
    genotype_df: pd.DataFrame,
    snp_info: pd.DataFrame,
    window: float,
    window_unit: str,
    step: int,
    r2_threshold: float,
    plink_path: str,
    common_flags: list,
    workdir: Path,
    label: str,
    round_dosage: bool,
    n_jobs: Optional[int] = None,
    device: Optional[str] = None,
    materialize_pruned_bed: bool = True,
    max_avg_markers_per_window: Optional[float] = None,
    warn_avg_markers_per_window: Optional[float] = None,
) -> list:
    """Run one LD-pruning pass (PLINK2 kb/variants, or the Python cm
    fallback) over a single partition of SNPs. Returns the list of SNP IDs
    that survived.

    ``n_jobs``/``device`` (ver4-4 R3.e/R4.g) are forwarded to
    ``_cm_prune()`` only - the PLINK2 kb/variants path below is already
    externally multi-threaded via ``common_flags``' own ``--threads``
    (see ``_resolve_plink_threads()``), which is a different
    (PLINK-internal, CPU-only) parallelism mechanism entirely - it has no
    GPU path for ``device`` to select.

    PERFORMANCE FIX: ``materialize_pruned_bed`` gates the third plink2
    subprocess call below (``--extract <prune.in> --make-bed``), which
    re-reads the whole (pre-pruning) bed/bim/fam fileset just written and
    writes out a SECOND, pruned copy of it. That pruned fileset is pure
    provenance -- ``ld_prune_snps`` builds its own returned DataFrame
    directly from ``prune_in_file`` via a plain pandas column selection
    (see this module's docstring), never from these files. Whenever the
    caller (``ld_prune_snps``) is about to delete the whole working
    directory anyway (``work_dir`` not given and ``keep_intermediate`` is
    False -- the common case, and this project's own default), running
    this extra plink2 call only to immediately throw its output away was
    pure wasted wall-clock time: one whole additional read-modify-write
    pass over the genotype fileset, on every single LD-pruning call, for
    a file nothing ever reads. ``ld_prune_snps`` now only asks for it when
    the intermediate files are actually going to survive the call (an
    explicit ``work_dir``, or ``keep_intermediate=True``) -- i.e. exactly
    when a person could plausibly want the pruned bed fileset on disk
    afterwards."""
    if genotype_df.shape[1] == 0:
        return []

    if window_unit == "cm":
        return _cm_prune(genotype_df, snp_info, window_cm=window, r2_threshold=r2_threshold,
                          n_jobs=n_jobs, device=device)

    # PLINK2's PED/MAP -> BED conversion (--make-bed, below) requires every
    # variant belonging to the same chromosome to be CONTIGUOUS in the file
    # -- it errors out ("has a split chromosome") the moment a chromosome
    # code reappears after a different one was seen in between. snp_info's
    # row order at this point is whatever genotype_df's column order was
    # (see _validate_inputs's "reindex to genotype_df's column order"
    # comment) -- i.e. the order columns happened to arrive in from the
    # genotype file / upstream marker-pool routing, which is NOT guaranteed
    # to be chromosome-contiguous (e.g. PLINK --extract, gene-window
    # restriction, or RF/SHAP-importance ranking all reorder markers).
    #
    # For a real (mapped) partition this reordering is free: kb-window
    # pruning is defined by actual CHR/POS values, not by row order, so
    # sorting here changes nothing about which markers get compared -- it
    # only satisfies PLINK2's file-format requirement. We do NOT do this for
    # window_unit="variants", where the variant-count window is explicitly
    # defined BY column order (see ld_prune_snps's docstring) -- reordering
    # there would silently change which markers get windowed together.
    if window_unit != "variants":
        sort_cols = ["CHR", "POS"] if "POS" in snp_info.columns else ["CHR"]
        sort_order = snp_info[sort_cols].reset_index().sort_values(sort_cols, kind="mergesort").index
        snp_info = snp_info.iloc[sort_order]
        genotype_df = genotype_df.iloc[:, sort_order]

    raw_prefix = workdir / f"{label}_raw"
    _write_ped_map(genotype_df, snp_info, raw_prefix, round_dosage=round_dosage)

    bed_prefix = workdir / f"{label}_data"
    _run_plink(
        [plink_path, "--pedmap", str(raw_prefix), *common_flags, "--make-bed", "--out", str(bed_prefix)],
        bed_prefix,
    )

    prune_prefix = workdir / f"{label}_out"
    window_arg = f"{window}kb" if window_unit == "kb" else str(int(window))

    # PATCH_NOTES (performance fix): pre-flight cost estimate/guard, BEFORE
    # the (potentially very slow) --indep-pairwise call below ever starts -
    # see check_kb_window_cost()'s own docstring. Only meaningful for 'kb'
    # windows; 'variants' windows are already bounded by construction.
    if window_unit == "kb":
        try:
            check_kb_window_cost(
                snp_info["CHR"], snp_info["POS"], window,
                threads=(common_flags[common_flags.index("--threads") + 1] if "--threads" in common_flags else 1),
                max_avg_markers_per_window=max_avg_markers_per_window,
                warn_avg_markers_per_window=(
                    warn_avg_markers_per_window if warn_avg_markers_per_window is not None
                    else DEFAULT_WARN_AVG_MARKERS_PER_WINDOW
                ),
                context=f" [{label} partition, --indep-pairwise on '{bed_prefix}']",
            )
        except ValueError as exc:
            raise PlinkError(str(exc)) from exc

    _run_plink(
        [
            plink_path, "--bfile", str(bed_prefix), *common_flags,
            "--indep-pairwise", window_arg, str(step), str(r2_threshold),
            "--out", str(prune_prefix),
        ],
        prune_prefix,
        heartbeat_seconds=300,
        heartbeat_label=f"--indep-pairwise {window_arg} {step} {r2_threshold} ({label} partition)",
    )
    prune_in_file = Path(f"{prune_prefix}.prune.in")
    with open(prune_in_file) as fh:
        keep_ids = [line.strip() for line in fh if line.strip()]

    # Optional: also materialize the pruned BED fileset, purely for
    # provenance / downstream PLINK use -- not needed for the DataFrame
    # ld_prune_snps returns. Skipped whenever the caller isn't keeping the
    # working directory around to look at it (see materialize_pruned_bed's
    # own docstring entry above) -- this is the expensive, entirely
    # optional half of this function's plink2 work.
    if materialize_pruned_bed:
        pruned_prefix = workdir / f"{label}_pruned"
        _run_plink(
            [
                plink_path, "--bfile", str(bed_prefix), *common_flags,
                "--extract", str(prune_in_file), "--make-bed", "--out", str(pruned_prefix),
            ],
            pruned_prefix,
        )
    return keep_ids


def _fill_unmapped_positions(genotype_sub: pd.DataFrame, snp_info_sub: pd.DataFrame, spacing_bp: int) -> pd.DataFrame:
    """Fill missing CHR/POS (NaN) with synthetic placeholder coordinates,
    just enough for PLINK's MAP format to be satisfied for a variant-count
    pruning pass. A1/A2 are already guaranteed non-null by _validate_inputs."""
    placeholder = make_placeholder_snp_info(genotype_sub, chrom="UNMAPPED", spacing_bp=spacing_bp)
    filled = snp_info_sub.copy()
    filled["CHR"] = filled["CHR"].fillna(placeholder["CHR"])
    filled["POS"] = filled["POS"].fillna(placeholder["POS"])
    return filled


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def ld_prune_snps(
    genotype_df: pd.DataFrame,
    genotype_df_valid: pd.DataFrame,
    genotype_df_test: pd.DataFrame,
    snp_info: pd.DataFrame,
    window: float = 50,
    window_unit: str = "kb",
    step: int = 5,
    r2_threshold: float = 0.2,
    plink_path: str = None,
    allow_extra_chr: bool = False,
    chr_set: Optional[int] = None,
    work_dir: Optional[str] = None,
    keep_intermediate: bool = False,
    round_dosage: bool = True,
    unmapped_strategy: str = "variant_count",
    unmapped_window: int = 50,
    unmapped_step: int = 5,
    maf_threshold: float = 0.0,
    plink_threads: Optional[int] = None,
    n_jobs: Optional[int] = None,
    device: Optional[str] = None,
    max_avg_markers_per_window: Optional[float] = None,
    warn_avg_markers_per_window: Optional[float] = None,
) -> pd.DataFrame:
    """
    LD-prune a 0/1/2(.x) genotype matrix and return it restricted to the
    surviving SNPs, in the original sample order and column order.

    --------------------------------------------------------------------
    MIXED MAPPED / UNMAPPED SNPs
    --------------------------------------------------------------------
    In practice you often have a real genetic/physical map (CHR/POS/CM)
    for some markers but not others -- e.g. new markers added after the
    reference map was built. This function handles that automatically:
    it does NOT require every SNP in snp_info to have a value.

    For window_unit="kb" or "cm", a SNP counts as "mapped" if it has a
    non-null CHR and POS (and, for "cm", a non-null CM); everything else
    (including SNPs entirely absent from snp_info's index) is "unmapped".
    The two groups are pruned in separate passes:

        * mapped SNPs   -> pruned using your requested window/window_unit
                           against their real coordinates, exactly as if
                           you'd only passed the fully-mapped subset.
        * unmapped SNPs -> handled per `unmapped_strategy` (see below),
                           since kb/cm windows are meaningless without
                           real coordinates for them.

    For window_unit="variants", there's nothing to split -- variant-count
    windows never depend on CHR/POS/CM, so ALL SNPs are pruned together
    in original column order regardless of map completeness.

    Parameters
    ----------
    genotype_df : DataFrame, samples x SNPs, values in [0, 2] (NaN = missing) for a validation set.
    genotype_df_valid : DataFrame, samples x SNPs, values in [0, 2] (NaN = missing) for a validation set.
    genotype_df_test : DataFrame, samples x SNPs, values in [0, 2] (NaN = missing) for a validation set.
    snp_info    : DataFrame indexed by SNP ID; see module docstring. Rows
                  or columns (CHR/POS/CM) may be missing/NaN for SNPs
                  without a real map -- see above.
    window      : window size for the *mapped* partition. Units depend on
                  window_unit: "variants" -> # variants, "kb" -> kilobases,
                  "cm" -> centimorgans.
    window_unit : "kb" (default), "variants", or "cm".
    step        : step size (variant count) for the mapped partition.
    r2_threshold: unphased hardcall r^2 threshold above which a variant is
                  pruned (same meaning as PLINK's --indep-pairwise).
    plink_path  : path to the plink2 executable.
    allow_extra_chr : pass --allow-extra-chr to PLINK2 (non-standard
                  chromosome names, e.g. scaffolds).
    chr_set     : if working with a non-human genome, pass the haploid
                  chromosome count for --chr-set N.
    work_dir    : directory for intermediate files. A temp dir is created
                  and cleaned up automatically if not given.
    keep_intermediate : if True (or work_dir was explicitly given), the
                  PED/MAP/BED/log files are left on disk for inspection.
    round_dosage : PED only stores hard genotype calls. If your data has
                  fractional dosage values (0.5/1.5), round_dosage=True
                  (default) rounds them to the nearest hardcall before
                  writing PED, with a warning. Set False to raise instead.
                  Doesn't affect the cm path, which uses raw dosages.
    unmapped_strategy : what to do with SNPs that lack a real map, only
                  relevant when window_unit is "kb" or "cm":
                    "variant_count" (default) -- prune them separately
                        using a variant-count window (unmapped_window/
                        unmapped_step below), i.e. based only on their
                        column order in genotype_df. Reasonable when that
                        order is genomic/genetic-map order (true for most
                        exported marker sets) but not a substitute for a
                        real map.
                    "skip"  -- keep every unmapped SNP, untouched, no
                        pruning decision made for them at all.
                    "drop"  -- discard every unmapped SNP outright, since
                        no reliable LD-pruning decision can be made
                        without a map.
    unmapped_window, unmapped_step : window/step (variant counts) used for
                  the unmapped partition when unmapped_strategy="variant_count".
    maf_threshold : minor allele frequency (MAF) filtering, applied ONCE
                  up front - before LD pruning itself - using the TRAIN
                  genotype (genotype_df) to decide which markers survive,
                  exactly like LD pruning's own r2 decisions are made from
                  train and then applied to valid/test too. 0 (default)
                  disables this entirely - no behaviour change from before
                  this parameter existed. A marker with an undefined MAF
                  (every train sample missing for it) is always kept -
                  see maf_filter()'s own docstring for why. This is a pure
                  NumPy/pandas computation with no PLINK dependency, so it
                  applies identically regardless of window_unit and even
                  when plink2 itself isn't available.
    plink_threads : forwarded as every plink2 subprocess call's own
                  `--threads` flag. `None` (default) falls back to the
                  run's shared compute-resource settings (`PLINK_THREADS`
                  config key, via `get_active_compute_resources()`) -
                  see `_resolve_plink_threads()`'s own docstring for why
                  this matters (GPU allocations typically reserve far
                  fewer CPUs than a CPU-only run, and plink2's own
                  hardware auto-detection is not cgroup-aware).
    n_jobs      : ver4-4 R3.e - process fan-out width for the ``cm``
                  window-unit path's own per-chromosome pruning
                  (``_cm_prune()``); has NO effect on ``"kb"``/
                  ``"variants"`` (PLINK2-driven) pruning, which is
                  parallelised separately via `plink_threads` above.
                  `None` (default) falls back to the run's shared
                  compute-resource settings (`N_JOBS`, via
                  `get_active_compute_resources()`); `1` forces today's
                  exact serial per-chromosome loop.
    device      : ver4-4 R4.g - forwarded to `_cm_prune()`'s own
                  `device` (its no-NaN fast path's
                  `ld_kernels.pairwise_r2_matrix()` call). Has NO effect
                  on `"kb"`/`"variants"` (PLINK2-driven) pruning, which
                  has no GPU path. `None` (default) falls back to the
                  run's shared compute-resource settings, gated on
                  `GPU_LD_R2` (see `_cm_prune()`'s own docstring for why
                  the gate lives at the resolution point).
    max_avg_markers_per_window : float, optional
                  PATCH_NOTES (performance fix): HARD cap (opt-in, default
                  None = no cap, no behaviour change for any existing
                  caller) on the estimated average number of OTHER markers
                  falling inside each 'kb'-window --indep-pairwise window -
                  see `check_kb_window_cost()`'s own docstring. When set
                  and exceeded, this call raises immediately, BEFORE
                  plink2 ever runs, instead of silently taking a very long
                  time. Has no effect for `window_unit != "kb"`.
    warn_avg_markers_per_window : float, optional
                  Trip-wire for the (non-fatal) cost warning printed for
                  'kb' windows - defaults to `DEFAULT_WARN_AVG_MARKERS_PER_WINDOW`
                  when not given.

    Returns
    -------
    DataFrame, same index and column order as genotype_df, restricted to
    the SNPs that survived pruning.
    """
    if unmapped_strategy not in ("variant_count", "skip", "drop"):
        raise LDPruneInputError(
            f"unmapped_strategy must be one of 'variant_count', 'skip', 'drop' (got {unmapped_strategy!r})"
        )
    # Requirement (prevent this from happening silently): PLINK's own
    # --indep-pairwise rejects a kb-based window whenever the step size
    # (variant count) isn't exactly 1 - see _prune_partition's window_arg/
    # step usage below, and main_app.py's build_ld_prune_config() for the
    # matching GUI-level check. Validated here too (not just in the GUI) so
    # a headless/HPC run (run_sequential.py etc., which never goes through
    # main_app.py at all) gets this same clear, immediate error instead of
    # a raw PLINK failure partway through _prune_partition().
    if window_unit == "kb" and step != 1:
        raise LDPruneInputError(
            f"PLINK's --indep-pairwise requires step to be exactly 1 when window_unit='kb' "
            f"(got step={step}) - PLINK itself errors out on any other value for a kb-based "
            f"window. Either set step=1, or use window_unit='variants' (where a step other "
            f"than 1 is meaningful and supported)."
        )

    
    #genotype_df_pheno = genotype_df.iloc[:,-1]
    #genotype_df = genotype_df.iloc[:,:-1]

    #genotype_df_valid_pheno = genotype_df_valid.iloc[:,-1] if genotype_df_valid.shape[1] != 0 else pd.DataFrame()
    #genotype_df_test_pheno = genotype_df_test.iloc[:,-1]
    
    snp_info = _validate_inputs(genotype_df, snp_info, window_unit)

    # MAF filtering (see maf_threshold's own docstring above) - deliberately
    # BEFORE LD pruning, so a low-frequency marker never gets to "win" an
    # LD-pruning comparison against a more common one it happens to be in
    # LD with, only to survive here anyway; filtering it out first removes
    # it from consideration entirely, exactly like a real PLINK
    # `--maf X --indep-pairwise ...` pipeline would.
    if maf_threshold and maf_threshold > 0:
        _n_before_maf = genotype_df.shape[1]
        _maf_keep_cols = maf_filter(genotype_df, maf_threshold)
        genotype_df = genotype_df[_maf_keep_cols]
        if genotype_df_valid.shape[1] != 0:
            genotype_df_valid = genotype_df_valid[_maf_keep_cols]
        genotype_df_test = genotype_df_test[_maf_keep_cols]
        snp_info = snp_info.loc[_maf_keep_cols]
        # print(), not logger.info() - this module's existing logger.info()
        # calls are silently dropped under this codebase's default logging
        # configuration (nothing anywhere calls logging.basicConfig() - see
        # Preprocess/LD_decay_plot.py's module docstring for the full
        # explanation of the same issue there). A brand new, user-facing
        # "did my MAF filter actually do anything?" message should not
        # inherit that same invisibility.
        print(
            f"[LD_pruning] MAF filtering: {_n_before_maf} -> {genotype_df.shape[1]} SNPs "
            f"(maf >= {maf_threshold}), before LD pruning."
        )
        if genotype_df.shape[1] == 0:
            print("[LD_pruning] MAF filtering removed every marker - nothing left for LD pruning to do.")

    cleanup = work_dir is None and not keep_intermediate
    workdir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="ld_prune_"))
    workdir.mkdir(parents=True, exist_ok=True)
    logger.info("Working directory: %s", workdir)

    threads = _resolve_plink_threads(plink_threads)

    common_flags = []
    if allow_extra_chr:
        common_flags.append("--allow-extra-chr")
    if chr_set is not None:
        common_flags += ["--chr-set", str(chr_set)]
    common_flags += ["--threads", str(threads)]

    # PERFORMANCE FIX: only ask _prune_partition() to materialize the
    # (entirely optional, provenance-only) pruned BED fileset when this
    # workdir is actually going to survive past this call -- see
    # _prune_partition()'s own materialize_pruned_bed docstring entry for
    # the full rationale. `cleanup` (computed just above) already encodes
    # exactly that: True means work_dir was None and keep_intermediate was
    # False, i.e. the finally-block below is about to shutil.rmtree()
    # this entire directory regardless.
    _materialize_pruned_bed = not cleanup

    try:
        if window_unit == "variants":
            # Variant-count windows never need real coordinates -- no split.
            keep_ids = _prune_partition(
                genotype_df, snp_info, window, window_unit, step, r2_threshold,
                plink_path, common_flags, workdir, "all", round_dosage, n_jobs=n_jobs, device=device,
                materialize_pruned_bed=_materialize_pruned_bed,
                max_avg_markers_per_window=max_avg_markers_per_window,
                warn_avg_markers_per_window=warn_avg_markers_per_window,
            )
        else:
            required = ["CHR", "POS"] + (["CM"] if window_unit == "cm" else [])
            has_map = snp_info[required].notna().all(axis=1)
            mapped_ids = snp_info.index[has_map]
            unmapped_ids = snp_info.index[~has_map]

            logger.info(
                "%d / %d SNPs have a complete %s map; %d are unmapped (strategy=%r).",
                len(mapped_ids), snp_info.shape[0], required, len(unmapped_ids), unmapped_strategy,
            )

            keep_ids = []
            if len(mapped_ids):
                keep_ids += _prune_partition(
                    genotype_df[mapped_ids], snp_info.loc[mapped_ids], window, window_unit,
                    step, r2_threshold, plink_path, common_flags, workdir, "mapped", round_dosage,
                    n_jobs=n_jobs, device=device, materialize_pruned_bed=_materialize_pruned_bed,
                    max_avg_markers_per_window=max_avg_markers_per_window,
                    warn_avg_markers_per_window=warn_avg_markers_per_window,
                )

            if len(unmapped_ids):
                if unmapped_strategy == "skip":
                    logger.info("Keeping all %d unmapped SNP(s) untouched (unmapped_strategy='skip').", len(unmapped_ids))
                    keep_ids += list(unmapped_ids)
                elif unmapped_strategy == "drop":
                    logger.info("Dropping all %d unmapped SNP(s) (unmapped_strategy='drop').", len(unmapped_ids))
                else:  # variant_count
                    unmapped_snp_info = _fill_unmapped_positions(
                        genotype_df[unmapped_ids], snp_info.loc[unmapped_ids], spacing_bp=1000
                    )
                    keep_ids += _prune_partition(
                        genotype_df[unmapped_ids], unmapped_snp_info, unmapped_window, "variants",
                        unmapped_step, r2_threshold, plink_path, common_flags, workdir, "unmapped", round_dosage,
                        n_jobs=n_jobs, device=device, materialize_pruned_bed=_materialize_pruned_bed,
                    )

        n_before, n_after = genotype_df.shape[1], len(keep_ids)
        logger.info("LD pruning: %d -> %d SNPs (window=%s%s, r2>%s)", n_before, n_after, window, window_unit, r2_threshold)

        keep_set = set(keep_ids)
        kept_columns = [c for c in genotype_df.columns if c in keep_set]  # preserves original order

        if genotype_df_valid.shape[0] != 0:
            return genotype_df[kept_columns].copy(), genotype_df_valid[kept_columns].copy(), genotype_df_test[kept_columns].copy()
        else:
            return genotype_df[kept_columns].copy(), genotype_df_valid, genotype_df_test[kept_columns].copy()
    finally:
        if cleanup:
            shutil.rmtree(workdir, ignore_errors=True)
        else:
            logger.info("Intermediate PLINK files kept at: %s", workdir)


# --------------------------------------------------------------------------
# Example usage
# --------------------------------------------------------------------------


def LD_pruning(train, valid, test, ld_config):
    genotype_df = train
    genotype_df_valid = valid
    genotype_df_test = test
    snp_info = ld_config['snp_info']
    window = ld_config['window']
    window_unit = ld_config['window_unit']
    step = ld_config['step']
    r2_threshold = ld_config['r2_threshold']
    plink_path = ld_config['plink_path']
    allow_extra_chr = ld_config['allow_extra_chr']
    chr_set = ld_config['chr_set']
    work_dir = ld_config['work_dir']
    keep_intermediate = ld_config['keep_intermediate']
    round_dosage = ld_config['round_dosage']
    unmapped_strategy = ld_config['unmapped_strategy']
    # Only present in ld_config when unmapped_strategy=='variant_count' (see
    # streamlit_app_ver7.py's gather_config) - fall back to ld_prune_snps'
    # own defaults otherwise rather than raising a KeyError.
    unmapped_window = ld_config.get('unmapped_window', 50)
    unmapped_step = ld_config.get('unmapped_step', 5)
    # Optional MAF pre-filter (see ld_prune_snps' own maf_threshold
    # docstring) - defaults to 0 (disabled) so an older saved LD_PRUNE
    # config with no 'maf_threshold' key behaves exactly as before this
    # option existed.
    maf_threshold = ld_config.get('maf_threshold', 0.0)
    # Optional explicit override (mirrors Preprocess/plink_io.py's own
    # 'plink_threads' parameter pattern) - falls back to the run's shared
    # compute-resource settings when absent (see
    # _resolve_plink_threads()'s docstring above for why this matters).
    plink_threads = ld_config.get('plink_threads')
    # ver4-4 R3.e: explicit override for the 'cm' window-unit path's own
    # per-chromosome process fan-out (_cm_prune) - falls back to the run's
    # shared N_JOBS setting when absent, exactly like plink_threads above.
    # Has no effect on 'kb'/'variants' pruning (PLINK-internal threading,
    # via plink_threads instead).
    n_jobs = ld_config.get('n_jobs')
    # ver4-4 R4.g: explicit override for the same 'cm' window-unit path's
    # no-NaN fast path device (_cm_prune -> ld_kernels.pairwise_r2_matrix)
    # - falls back to the run's shared TORCH_DEVICE/GPU_LD_R2 settings when
    # absent, exactly like n_jobs above. Has no effect on 'kb'/'variants'
    # pruning (no GPU path).
    device = ld_config.get('device')
    # PATCH_NOTES (performance fix): optional pre-flight cost guard for a
    # 'kb'-window --indep-pairwise call - see check_kb_window_cost()'s own
    # docstring. Both default to "off"/"module default" respectively, so
    # an existing ld_config with neither key behaves exactly as before.
    max_avg_markers_per_window = ld_config.get('max_avg_markers_per_window')
    warn_avg_markers_per_window = ld_config.get('warn_avg_markers_per_window')

    # Build the SNP map. window_unit='variants' doesn't strictly need real
    # CHR/POS/CM, so fall back to a placeholder map only when the caller
    # didn't supply one - if a real (or partial) map was provided, honour it
    # instead of silently discarding it.
    if snp_info is None:
        if window_unit != 'variants':
            raise LDPruneInputError(
                "LD pruning: a SNP info file is required when window_unit is 'kb' or 'cm'."
            )
        snp_info = make_placeholder_snp_info(genotype_df)
    else:
        snp_info = pd.read_csv(snp_info, index_col=0)
        # Requirement 8: the SNP info file's columns are documented as
        # CHR, POS, then optionally CM, A1, A2 - IN THAT ORDER (see this
        # function's own docstring) - unify by position so a file that
        # follows the order but uses different header text (e.g.
        # 'Chromosome'/'Position' instead of 'CHR'/'POS') still works,
        # rather than those columns being silently treated as absent (see
        # _validate_inputs' own 'if col not in snp_info.columns' checks).
        # The index (SNP ID - a marker name, meaningful/user-chosen) is
        # never touched by this.
        _snp_info_canonical_order = ['CHR', 'POS', 'CM', 'A1', 'A2']
        snp_info = unify_columns_by_position(
            snp_info, _snp_info_canonical_order[:min(snp_info.shape[1], 5)], 'SNP info file'
        )

    # Requirement: LD pruning was explicitly requested (this function was
    # called at all), so a PlinkError here - whether because plink2 itself
    # isn't installed/on PATH, or because a plink2 command it ran failed -
    # must STOP the run rather than silently falling back to the unpruned
    # markers. Continuing silently would give the person results that
    # look normal but were never actually LD-pruned, with nothing in the
    # final output to reveal that - a far worse outcome than a clear,
    # immediate failure they can see and fix (install/locate plink2, or
    # fix whatever made the plink2 command itself fail - see the
    # PlinkError's own message for which). genomic_prediction.py's
    # per-task checkpoint/rollback (see checkpoint_utils.py) already
    # handles this exactly like any other task failure: whatever
    # scenarios completed before this one are saved, and the run can be
    # resumed once the underlying problem is fixed.
    pruned = ld_prune_snps(
        genotype_df, genotype_df_valid, genotype_df_test, snp_info,
        window=window, window_unit=window_unit,
        step=step, r2_threshold=r2_threshold,
        plink_path = plink_path,
        allow_extra_chr = allow_extra_chr,
        chr_set = chr_set,
        work_dir = work_dir,
        keep_intermediate = keep_intermediate,
        round_dosage = round_dosage,
        unmapped_strategy = unmapped_strategy,
        unmapped_window = unmapped_window,
        unmapped_step = unmapped_step,
        maf_threshold = maf_threshold,
        plink_threads = plink_threads,
        n_jobs = n_jobs,
        device = device,
        max_avg_markers_per_window = max_avg_markers_per_window,
        warn_avg_markers_per_window = warn_avg_markers_per_window,
    )
    print(f"Pruned: {genotype_df.shape[1]} -> {pruned[0].shape[1]} markers")

    return pruned[0], pruned[1], pruned[2]
