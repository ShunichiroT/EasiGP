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
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PlinkError(RuntimeError):
    """Raised when a PLINK2 subprocess call returns a non-zero exit code."""


class LDPruneInputError(ValueError):
    """Raised when genotype_df / snp_info fail validation."""


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
    pat = np.zeros(n_samples, dtype=object)
    mat = np.zeros(n_samples, dtype=object)
    sex = np.full(n_samples, -9)
    pheno = np.full(n_samples, -9)
    
    ped_left = np.column_stack([fid, iid, pat, mat, sex, pheno]).astype(object)
    ped_full = np.hstack([ped_left, geno_block])

    with open(f"{out_prefix}.ped", "w") as fh:
        for row in ped_full:
            fh.write(" ".join(row.astype(str)) + "\n")

    logger.info("Wrote %s.ped (%d samples x %d SNPs) and %s.map", out_prefix, n_samples, n_snps, out_prefix)


# --------------------------------------------------------------------------
# PLINK2 subprocess wrapper
# --------------------------------------------------------------------------

def _run_plink(cmd: list, log_prefix: Path) -> subprocess.CompletedProcess:
    logger.info("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise PlinkError(
            f"Could not find/execute {cmd[0]!r}. Make sure plink2 is installed and on "
            f"PATH, or pass plink_path='/full/path/to/plink2'. ({e})"
        ) from e
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
    """r^2 between two genotype vectors, ignoring samples missing in either."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return 0.0
    xm, ym = x[mask], y[mask]
    if xm.std() == 0 or ym.std() == 0:
        return 0.0
    r = np.corrcoef(xm, ym)[0, 1]
    return 0.0 if np.isnan(r) else r * r


def _cm_prune(
    genotype_df: pd.DataFrame,
    snp_info: pd.DataFrame,
    window_cm: float,
    r2_threshold: float,
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
    """
    keep_ids: list = []
    geno_all = genotype_df.to_numpy(dtype=float)
    col_index = {snp_id: i for i, snp_id in enumerate(genotype_df.columns)}

    for chrom, grp in snp_info.groupby("CHR"):
        grp_sorted = grp.sort_values("CM")
        ids = grp_sorted.index.to_numpy()
        cms = grp_sorted["CM"].to_numpy()
        n = len(ids)
        keep_mask = np.ones(n, dtype=bool)

        for i in range(n):
            if not keep_mask[i]:
                continue
            gi = geno_all[:, col_index[ids[i]]]
            j = i + 1
            while j < n and (cms[j] - cms[i]) <= window_cm:
                if keep_mask[j]:
                    gj = geno_all[:, col_index[ids[j]]]
                    if _pairwise_r2(gi, gj) > r2_threshold:
                        keep_mask[j] = False
                j += 1

        n_dropped = int((~keep_mask).sum())
        logger.info("cM-prune chr %s: kept %d / %d SNPs (dropped %d)", chrom, keep_mask.sum(), n, n_dropped)
        keep_ids.extend(ids[keep_mask].tolist())

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
) -> list:
    """Run one LD-pruning pass (PLINK2 kb/variants, or the Python cm
    fallback) over a single partition of SNPs. Returns the list of SNP IDs
    that survived."""
    if genotype_df.shape[1] == 0:
        return []

    if window_unit == "cm":
        return _cm_prune(genotype_df, snp_info, window_cm=window, r2_threshold=r2_threshold)

    raw_prefix = workdir / f"{label}_raw"
    _write_ped_map(genotype_df, snp_info, raw_prefix, round_dosage=round_dosage)

    bed_prefix = workdir / f"{label}_data"
    _run_plink(
        [plink_path, "--pedmap", str(raw_prefix), *common_flags, "--make-bed", "--out", str(bed_prefix)],
        bed_prefix,
    )

    prune_prefix = workdir / f"{label}_out"
    window_arg = f"{window}kb" if window_unit == "kb" else str(int(window))
    _run_plink(
        [
            plink_path, "--bfile", str(bed_prefix), *common_flags,
            "--indep-pairwise", window_arg, str(step), str(r2_threshold),
            "--out", str(prune_prefix),
        ],
        prune_prefix,
    )
    prune_in_file = Path(f"{prune_prefix}.prune.in")
    with open(prune_in_file) as fh:
        keep_ids = [line.strip() for line in fh if line.strip()]

    # Optional: also materialize the pruned BED fileset, purely for
    # provenance / downstream PLINK use -- not needed for the DataFrame
    # ld_prune_snps returns.
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

    Returns
    -------
    DataFrame, same index and column order as genotype_df, restricted to
    the SNPs that survived pruning.
    """
    if unmapped_strategy not in ("variant_count", "skip", "drop"):
        raise LDPruneInputError(
            f"unmapped_strategy must be one of 'variant_count', 'skip', 'drop' (got {unmapped_strategy!r})"
        )
    
    #genotype_df_pheno = genotype_df.iloc[:,-1]
    #genotype_df = genotype_df.iloc[:,:-1]

    #genotype_df_valid_pheno = genotype_df_valid.iloc[:,-1] if genotype_df_valid.shape[1] != 0 else pd.DataFrame()
    #genotype_df_test_pheno = genotype_df_test.iloc[:,-1]
    
    snp_info = _validate_inputs(genotype_df, snp_info, window_unit)

    cleanup = work_dir is None and not keep_intermediate
    workdir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="ld_prune_"))
    workdir.mkdir(parents=True, exist_ok=True)
    logger.info("Working directory: %s", workdir)

    common_flags = []
    if allow_extra_chr:
        common_flags.append("--allow-extra-chr")
    if chr_set is not None:
        common_flags += ["--chr-set", str(chr_set)]

    try:
        if window_unit == "variants":
            # Variant-count windows never need real coordinates -- no split.
            keep_ids = _prune_partition(
                genotype_df, snp_info, window, window_unit, step, r2_threshold,
                plink_path, common_flags, workdir, "all", round_dosage,
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

    try:
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
        )
        print(f"Pruned: {genotype_df.shape[1]} -> {pruned[0].shape[1]} markers")
        
        return pruned[0], pruned[1], pruned[2]
    
    except PlinkError as e:
        print(f"plink2 not available in this environment ({e}); "
              f"skipping LD pruning and continuing with the unpruned markers.")
        return genotype_df, genotype_df_valid, genotype_df_test
