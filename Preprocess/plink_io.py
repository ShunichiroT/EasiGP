"""
Preprocess/plink_io.py
========================

Support for PLINK1 binary genotype filesets (a .bed/.bim/.fam trio sharing
one file stem, e.g. 'mydata.bed'/'mydata.bim'/'mydata.fam') as an alternative
to a genotype CSV. Everything downstream of this module (LD pruning, RF
marker filtering, every prediction model) keeps working exactly as before,
unmodified - this module's job is entirely to get a bed/bim/fam fileset into
the exact same shape the rest of the pipeline already expects: a DataFrame
with columns ['ID', 'population', <marker columns...>], identical in
structure to a hand-supplied genotype CSV.

Two things this module deliberately does NOT do, by design (see
genomic_prediction.py's orchestration logic for where these decisions are
actually made):
  - It never decides *when* to convert a bed/bim/fam fileset (e.g. whether
    to convert immediately, or only after LD pruning, or only for a
    gene-window-restricted subset of markers) - it just provides the
    building blocks (read .bim, run a PLINK export, merge with phenotype
    metadata) that the caller composes as needed.
  - 'ID' and 'population' are NEVER invented from the .fam file (PLINK has
    no native "population" concept, and re-purposing FID for it would be a
    guess). They always come from the phenotype file instead - the .fam
    file's IID column is only used as the join key to attach them, exactly
    mirroring how the existing genotype-CSV workflow already expects a
    phenotype file with the SAME 'ID'/'population' values.

Requires a working `plink2` executable on PATH (or an explicit path via
`plink_path=`) - the same dependency `Preprocess/LD_pruning.py` already has.
"""

import os
import re
import subprocess
import tempfile

import pandas as pd


class PlinkFilesetError(Exception):
    """Raised for a malformed/incomplete PLINK fileset, or a PLINK2
    subprocess failure."""


def validate_plink_fileset(bfile_stem):
    """Check that '<bfile_stem>.bed', '.bim', and '.fam' all exist (PLINK's
    own convention: all three files share one stem, differing only in
    extension - the "file names must be identical" requirement). Returns
    (bed_path, bim_path, fam_path).
    """
    bed_path = f"{bfile_stem}.bed"
    bim_path = f"{bfile_stem}.bim"
    fam_path = f"{bfile_stem}.fam"
    missing = [p for p in (bed_path, bim_path, fam_path) if not os.path.isfile(p)]
    if missing:
        raise PlinkFilesetError(
            f"Incomplete PLINK fileset for stem '{bfile_stem}': missing {missing}. "
            f"All three of <stem>.bed, <stem>.bim, <stem>.fam must exist, sharing the "
            f"exact same stem/prefix."
        )
    return bed_path, bim_path, fam_path


def read_bim_marker_info(bim_path):
    """Read a .bim file's marker positions only (chromosome, SNP ID, genetic
    distance, bp position, allele 1, allele 2 - PLINK1's standard 6-column,
    whitespace-separated, no-header format) into the same
    ['chromosome', 'name', 'start', 'end'] schema every marker_info.csv in
    this codebase already uses - so it's a drop-in replacement wherever a
    marker_info.csv would otherwise be read, e.g. for
    GAT_biological_prior_knowledge's gene<->SNP mapping, without needing to
    touch the (potentially far larger) genotype matrix in the .bed file at
    all.

    A SNP's own position is a single base pair, not an interval, so `start`
    and `end` are both set to the .bim BP column (a zero-width point) -
    this is exactly equivalent for every interval-overlap test this
    codebase does against marker_info (`marker.start <= gene.end and
    marker.end >= gene.start`), and for LD-pruning window logic, while
    staying simpler and more literal than inventing bin boundaries that
    the .bim file itself doesn't provide.
    """
    bim = pd.read_csv(
        bim_path, sep=r'\s+', header=None,
        names=['chromosome', 'name', 'cm', 'position', 'a1', 'a2'],
        dtype={'chromosome': str, 'name': str},
        encoding='utf-8-sig',
    )
    return pd.DataFrame({
        'chromosome': bim['chromosome'],
        'name': bim['name'],
        'start': bim['position'].astype(float),
        'end': bim['position'].astype(float),
    })


def read_fam_iids(fam_path):
    """Read a .fam file's IID column (2nd of its 6 whitespace-separated
    columns: FID, IID, father IID, mother IID, sex, phenotype) - the sample
    identifiers PLINK itself uses, and the join key back to the phenotype
    file's own 'ID' column (see module docstring)."""
    fam = pd.read_csv(fam_path, sep=r'\s+', header=None,
                       names=['fid', 'iid', 'pat', 'mat', 'sex', 'pheno'],
                       dtype={'iid': str}, encoding='utf-8-sig')
    return fam['iid'].tolist()


def _run_plink(cmd, log_prefix, plink_path):
    """Run a plink2 subprocess, raising a clear PlinkFilesetError (with the
    command and plink2's own stderr/log tail) on any non-zero exit, rather
    than letting a cryptic CalledProcessError surface."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise PlinkFilesetError(
            f"Could not run '{plink_path}' - is PLINK2 installed and on PATH, or pass "
            f"plink_path='/full/path/to/plink2'? ({e})"
        )
    if result.returncode != 0:
        log_path = f"{log_prefix}.log"
        log_tail = ''
        if os.path.isfile(log_path):
            with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                log_tail = f.read()[-2000:]
        raise PlinkFilesetError(
            f"PLINK2 command failed (exit code {result.returncode}): {' '.join(cmd)}\n"
            f"--- stderr ---\n{result.stderr}\n--- log tail ({log_path}) ---\n{log_tail}"
        )
    return result


def ld_prune_plink_native(bfile_stem, phenotype_df, keep_iid_list, ld_config, work_dir=None):
    """LD-prune a PLINK bed/bim/fam fileset natively - running PLINK2's own
    `--indep-pairwise` directly on a `--keep`-restricted copy of the
    fileset - for `ld_config['window_unit']` in ('kb', 'variants'). This
    skips converting the (potentially genome-wide, very large) unpruned
    genotype matrix to a DataFrame/CSV entirely; only the much smaller
    PRUNED result is ever materialised as a DataFrame (via
    plink_to_genotype_df), right at the end.

    A .bim file always has a real CHR/POS for every SNP (it's inherent to
    the PLINK format), so - unlike Preprocess.LD_pruning.LD_pruning(),
    which has to handle a possibly incomplete/gappy snp_info map for a
    hand-supplied genotype CSV - there's no "unmapped SNP" partition to
    handle here at all; every SNP is pruned together in one native PLINK2
    pass.

    `window_unit == 'cm'` has no PLINK2-native equivalent (PLINK2 doesn't do
    cM-based pruning; the existing 'cm' path in Preprocess/LD_pruning.py is
    a pure Python/numpy correlation calculation that needs real dosage
    values in memory regardless of input format) - that case is handled by
    converting `keep_iid_list`'s genotypes to a DataFrame first (via
    plink_to_genotype_df, with no marker restriction, since pruning hasn't
    happened yet), then delegating to the existing, unmodified
    Preprocess.LD_pruning.LD_pruning() exactly as the CSV-input workflow
    already does.

    Parameters
    ----------
    bfile_stem : str
    phenotype_df : pd.DataFrame with 'ID'/'population' columns (see
        plink_to_genotype_df).
    keep_iid_list : list of str
        The current task's training-set sample IIDs only - LD pruning
        (exactly like RF marker filtering) is decided from the training set
        alone, never validation/test.
    ld_config : dict
        Same schema Preprocess.LD_pruning.LD_pruning() already accepts
        (window, window_unit, step, r2_threshold, plink_path,
        allow_extra_chr, chr_set, keep_intermediate, round_dosage,
        maf_threshold, ...) - the unmapped_strategy/unmapped_window/
        unmapped_step keys are simply unused for the kb/variants native
        path (nothing is ever "unmapped"), and irrelevant for the cm
        fallback for the same reason. maf_threshold (0/None = disabled) is
        applied natively too, as PLINK2's own `--maf` flag alongside
        `--indep-pairwise` in the same call (kb/variants), or passed
        through unchanged to Preprocess.LD_pruning.ld_prune_snps()'s own
        pure-Python MAF filter (cm).
    work_dir : str, optional

    Returns
    -------
    pd.DataFrame with columns ['ID', 'population', <surviving markers>],
    restricted to `keep_iid_list`'s samples - the exact same shape
    LD_pruning() already returns for the CSV-input workflow.
    """
    validate_plink_fileset(bfile_stem)
    window_unit = ld_config.get('window_unit', 'kb')
    plink_path = ld_config.get('plink_path') or 'plink2'

    # Requirement (prevent this from happening silently): PLINK's own
    # --indep-pairwise rejects a kb-based window whenever the step size
    # (variant count) isn't exactly 1 - this native path builds and runs
    # that exact command itself (see window_arg/step in _do_prune below),
    # so it needs its own copy of the same check
    # Preprocess.LD_pruning.ld_prune_snps() applies for the CSV-input path,
    # and main_app.py's build_ld_prune_config() applies in the GUI - all
    # three call sites validate independently since each can be reached on
    # its own (this one specifically whenever GENOTYPE_FORMAT=='plink').
    if window_unit == 'kb' and int(ld_config.get('step', 5)) != 1:
        raise PlinkFilesetError(
            f"PLINK's --indep-pairwise requires step to be exactly 1 when window_unit='kb' "
            f"(got step={ld_config.get('step', 5)}) - PLINK itself errors out on any other "
            f"value for a kb-based window. Either set step=1, or use window_unit='variants' "
            f"(where a step other than 1 is meaningful and supported)."
        )

    if window_unit == 'cm':
        # No PLINK2-native cM pruning - fall back to the existing DataFrame
        # path, on the train-restricted (but not yet marker-restricted)
        # genotypes only. valid is genuinely unused here (empty, matching
        # the "no validation split" convention used throughout this
        # codebase); test is filled with the same train data as a
        # placeholder purely because ld_prune_snps() doesn't need a
        # meaningful test set to decide which markers to prune - only the
        # pruned *column set* (from the train return value) is used below.
        unpruned = plink_to_genotype_df(bfile_stem, phenotype_df, keep_iid_list=keep_iid_list,
                                         plink_path=plink_path, work_dir=work_dir)
        from Preprocess.LD_pruning import LD_pruning
        train_x = unpruned.iloc[:, 2:]
        pruned_train_x, _, _ = LD_pruning(train_x, pd.DataFrame(), train_x, ld_config)
        pruned_cols = list(pruned_train_x.columns)
        return unpruned[['ID', 'population'] + pruned_cols]

    def _do_prune(tmp_dir):
        fam_df = pd.read_csv(
            f"{bfile_stem}.fam", sep=r'\s+', header=None,
            names=['fid', 'iid', 'pat', 'mat', 'sex', 'pheno'],
            dtype={'fid': str, 'iid': str}, encoding='utf-8-sig',
        )
        fid_by_iid = dict(zip(fam_df['iid'], fam_df['fid']))
        missing_iids = [iid for iid in keep_iid_list if iid not in fid_by_iid]
        if missing_iids:
            raise PlinkFilesetError(
                f"keep_iid_list contains {len(missing_iids)} IID(s) not present in "
                f"'{bfile_stem}.fam': {missing_iids[:10]}{', ...' if len(missing_iids) > 10 else ''}"
            )
        keep_path = os.path.join(tmp_dir, 'keep.txt')
        with open(keep_path, 'w', encoding='utf-8') as f:
            for iid in keep_iid_list:
                f.write(f"{fid_by_iid[iid]}\t{iid}\n")

        common_flags = []
        if ld_config.get('allow_extra_chr'):
            common_flags.append('--allow-extra-chr')
        if ld_config.get('chr_set'):
            common_flags += ['--chr-set', str(ld_config['chr_set'])]
        # Optional MAF (minor allele frequency) pre-filter, applied by
        # PLINK2 itself as part of the SAME --indep-pairwise call below -
        # variants below this frequency are excluded from consideration
        # entirely (never appear in .prune.in), exactly like a real PLINK
        # `--maf X --indep-pairwise ...` pipeline. 0/None (default) means
        # "no filter", matching Preprocess.LD_pruning.ld_prune_snps()'s own
        # maf_threshold convention for the non-native (CSV-input) path.
        maf_threshold = ld_config.get('maf_threshold', 0.0)
        if maf_threshold and maf_threshold > 0:
            common_flags += ['--maf', str(maf_threshold)]

        window = ld_config.get('window', 50)
        step = ld_config.get('step', 5)
        r2_threshold = ld_config.get('r2_threshold', 0.2)
        window_arg = f"{window}kb" if window_unit == 'kb' else str(int(window))

        prune_prefix = os.path.join(tmp_dir, 'prune_out')
        _run_plink(
            [plink_path, '--bfile', bfile_stem, '--keep', keep_path, *common_flags,
             '--indep-pairwise', window_arg, str(step), str(r2_threshold),
             '--out', prune_prefix],
            prune_prefix, plink_path,
        )
        prune_in_path = f"{prune_prefix}.prune.in"
        with open(prune_in_path, 'r', encoding='utf-8') as f:
            pruned_snps = [line.strip() for line in f if line.strip()]

        if not pruned_snps:
            raise PlinkFilesetError(
                f"LD pruning removed every marker (0 survived --indep-pairwise "
                f"{window_arg} {step} {r2_threshold}) - check the window/step/r2 "
                f"threshold settings."
            )
        return pruned_snps

    if work_dir is not None:
        os.makedirs(work_dir, exist_ok=True)
        pruned_snps = _do_prune(work_dir)
    else:
        with tempfile.TemporaryDirectory(prefix='ld_prune_native_') as tmp_dir:
            pruned_snps = _do_prune(tmp_dir)

    print(f"[plink_io] Native PLINK LD pruning finished: {len(pruned_snps)} marker(s) survived.")

    return plink_to_genotype_df(bfile_stem, phenotype_df, extract_snp_list=pruned_snps,
                                 keep_iid_list=keep_iid_list, plink_path=plink_path, work_dir=work_dir)


def plink_to_genotype_df(bfile_stem, phenotype_df, extract_snp_list=None, keep_iid_list=None,
                          plink_path='plink2', work_dir=None):
    """The main entry point: convert a PLINK bed/bim/fam fileset into a
    genotype DataFrame structurally IDENTICAL to a hand-supplied genotype
    CSV - columns ['ID', 'population', <markers, in .bim order>] - by
    running PLINK2's own additive (0/1/2) dosage export and then attaching
    'ID'/'population' from `phenotype_df` (joined on sample ID; see module
    docstring for why those never come from the .fam file itself).

    Parameters
    ----------
    bfile_stem : str
        Path prefix shared by <bfile_stem>.bed/.bim/.fam.
    phenotype_df : pd.DataFrame
        Must have 'ID' and 'population' columns (the same phenotype file
        already used everywhere else in this pipeline). Only samples whose
        IID (the .fam file's own sample identifier) matches a value in
        phenotype_df['ID'] are kept - samples present in the PLINK fileset
        but absent from the phenotype file are silently dropped (they have
        no phenotype to predict), and are reported via a printed warning.
    extract_snp_list : list of str, optional
        If given, only these SNP IDs (matched against the .bim file's own
        SNP ID column) are exported - passed straight to PLINK2's
        `--extract`. This is what lets a caller convert only a small,
        gene-window-restricted subset of markers instead of the whole
        genome-wide fileset (see genomic_prediction.py's orchestration for
        GAT_biological_prior_knowledge).
    keep_iid_list : list of str, optional
        If given, only these sample IIDs are exported - passed straight to
        PLINK2's `--keep`. This is what lets a caller restrict PLINK's own
        LD-pruning-adjacent work to one task's training-set individuals
        only, without ever materialising the full genotype matrix as a
        DataFrame first.
    plink_path : str
        Name/path of the PLINK2 executable.
    work_dir : str, optional
        Directory for PLINK2's intermediate files. A fresh temporary
        directory is created (and cleaned up automatically) if not given.

    Returns
    -------
    pd.DataFrame with columns ['ID', 'population', <markers>], row order
    matching phenotype_df's own row order (restricted to samples actually
    present in the PLINK fileset), column order matching the (possibly
    --extract-restricted) .bim file's own marker order.
    """
    validate_plink_fileset(bfile_stem)

    if 'ID' not in phenotype_df.columns or 'population' not in phenotype_df.columns:
        raise ValueError("phenotype_df must have 'ID' and 'population' columns.")

    def _do_convert(tmp_dir):
        cmd = [plink_path, '--bfile', bfile_stem, '--allow-extra-chr']

        extract_path = None
        if extract_snp_list is not None:
            extract_path = os.path.join(tmp_dir, 'extract.txt')
            with open(extract_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(extract_snp_list) + '\n')
            cmd += ['--extract', extract_path]

        keep_path = None
        if keep_iid_list is not None:
            # plink2's --keep matches on the real (FID, IID) pair as it
            # appears in the .fam file - a placeholder FID (e.g. '0') simply
            # matches nothing and silently keeps zero samples. Look up each
            # requested IID's actual FID from the .fam file itself.
            fam_df = pd.read_csv(
                f"{bfile_stem}.fam", sep=r'\s+', header=None,
                names=['fid', 'iid', 'pat', 'mat', 'sex', 'pheno'],
                dtype={'fid': str, 'iid': str}, encoding='utf-8-sig',
            )
            fid_by_iid = dict(zip(fam_df['iid'], fam_df['fid']))
            missing_iids = [iid for iid in keep_iid_list if iid not in fid_by_iid]
            if missing_iids:
                raise PlinkFilesetError(
                    f"keep_iid_list contains {len(missing_iids)} IID(s) not present in "
                    f"'{bfile_stem}.fam': {missing_iids[:10]}"
                    f"{', ...' if len(missing_iids) > 10 else ''}"
                )
            keep_path = os.path.join(tmp_dir, 'keep.txt')
            with open(keep_path, 'w', encoding='utf-8') as f:
                for iid in keep_iid_list:
                    f.write(f"{fid_by_iid[iid]}\t{iid}\n")
            cmd += ['--keep', keep_path]

        out_prefix = os.path.join(tmp_dir, 'export')
        cmd += ['--export', 'A', '--write-snplist', '--out', out_prefix]
        _run_plink(cmd, out_prefix, plink_path)

        raw_path = f"{out_prefix}.raw"
        if not os.path.isfile(raw_path):
            raise PlinkFilesetError(
                f"PLINK2 reported success but '{raw_path}' was not created - "
                f"was extract_snp_list/keep_iid_list empty, or did every marker/sample "
                f"get filtered out?"
            )

        # --export A alone doesn't also write a .bim - '--write-snplist'
        # gives the authoritative post-filter marker ID list, in the same
        # order --export A itself used, without needing a full .bim (which
        # would need a separate --make-bed run to regenerate).
        snplist_path = f"{out_prefix}.snplist"
        with open(snplist_path, 'r', encoding='utf-8') as f:
            expected_names = [line.strip() for line in f if line.strip()]

        raw = pd.read_csv(raw_path, sep=r'\s+', encoding='utf-8-sig')
        marker_cols = [c for c in raw.columns if c not in ('FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE')]

        if len(marker_cols) != len(expected_names):
            raise PlinkFilesetError(
                f"Internal error: PLINK2's --export A produced {len(marker_cols)} marker "
                f"column(s), but --write-snplist reported {len(expected_names)}. This "
                f"should not happen - please report it."
            )

        # --export A appends the counted allele to each column name (e.g.
        # 'SNP0_T'); positional order matches the .bim file (verified above
        # on count, and here per-column against the expected SNP ID as a
        # defensive check rather than trusting order blindly).
        rename_map = {}
        for col, expected in zip(marker_cols, expected_names):
            stripped = re.sub(r'_[^_]+$', '', col)
            if stripped != expected:
                raise PlinkFilesetError(
                    f"Internal error: expected marker column for SNP '{expected}' at this "
                    f"position, but found '{col}' (stripped: '{stripped}') instead - PLINK2's "
                    f"--export A column order may not match the .bim file order in this PLINK2 "
                    f"version. Please report this."
                )
            rename_map[col] = expected

        raw = raw.rename(columns=rename_map)
        raw['IID'] = raw['IID'].astype(str)
        genotype = raw[['IID'] + expected_names]

        return genotype

    if work_dir is not None:
        os.makedirs(work_dir, exist_ok=True)
        genotype = _do_convert(work_dir)
    else:
        with tempfile.TemporaryDirectory(prefix='plink_io_') as tmp_dir:
            genotype = _do_convert(tmp_dir)

    phenotype_ids = phenotype_df.copy()
    phenotype_ids['ID'] = phenotype_ids['ID'].astype(str)

    merged = phenotype_ids[['ID', 'population']].merge(
        genotype, left_on='ID', right_on='IID', how='inner'
    ).drop(columns=['IID'])

    n_fam_only = len(set(genotype['IID']) - set(phenotype_ids['ID']))
    n_pheno_only = len(set(phenotype_ids['ID']) - set(genotype['IID']))
    if n_fam_only:
        print(f"[plink_io] {n_fam_only} sample(s) in the PLINK fileset have no matching "
              f"'ID' in the phenotype file and were dropped.")
    if n_pheno_only:
        print(f"[plink_io] {n_pheno_only} sample(s) in the phenotype file have no matching "
              f"IID in the PLINK fileset and were dropped.")
    if merged.shape[0] == 0:
        raise PlinkFilesetError(
            "No samples matched between the PLINK fileset's IID column and the phenotype "
            "file's 'ID' column - check that they use the same sample identifiers."
        )

    return merged.reset_index(drop=True)
