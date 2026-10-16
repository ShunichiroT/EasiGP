"""
Preprocess/plink_bed.py
========================

ver4-33, R1.b - a direct NumPy reader for PLINK1 binary genotype filesets
(``.bed`` / ``.bim`` / ``.fam``), used behind
``Preprocess.plink_io.plink_to_genotype_df``.

Why this exists
---------------
``plink_to_genotype_df`` used to run ``plink2 --export A`` for every call
(about three times per split-set per task, times phenotypes, times model
groups), write a text ``.raw`` file and parse it with pandas. That
subprocess + text round trip is a large share of the wall time of a PLINK
run (design blueprint R1.b). A ``.bed`` file is just a 2-bit-packed
variant-major matrix, so the same numbers can be gathered straight from a
read-only memory map and decoded with a 256-entry lookup table - no
subprocess, no text, no copy of the file, and every process on the node
shares the page cache.

What this module guarantees (and what it does not)
--------------------------------------------------
* The frame returned by :meth:`BedReader.genotype_frame` is meant to be
  identical to what the legacy ``--export A`` path produced: columns
  ``['IID', <snps in .bim order>]``, rows in ``.fam`` order restricted to
  the kept samples, columns with no missing call ``int64``, columns with any
  missing call ``float64`` with ``NaN`` (what ``read_csv`` yields).
* Which allele ``--export A`` counts depends on the plink2 build and
  options, so the decoder is NEVER trusted blindly: :meth:`BedReader.probe`
  runs the legacy export on a few variants x samples, learns whether the
  counted allele is ``.bim`` column 5 (A1) or column 6 (A2), and checks the
  decoded block equals the legacy block. A mismatch disables the fast path
  (the caller falls back to plink2, or raises when the user forced ``'on'``).
* Known, disclosed difference (blueprint D7): the legacy parser read the
  ``IID`` column without ``dtype=str``, so a fileset whose IIDs are ALL
  numeric-looking (``001``) had them coerced (``1``) before the phenotype
  join. The direct reader keeps the ``.fam`` text. For non-numeric IIDs the
  two are identical; for numeric ones the direct reader is the correct one.
  The count of affected IIDs is logged at open time.

Only NumPy and pandas are used (I12). The ``.bed`` is opened read-only (I10).
"""

import os
import subprocess
import tempfile
import io
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from Preprocess.plink_io import PlinkFilesetError, validate_plink_fileset


class BedFormatUnsupported(Exception):
    """The fileset cannot be read by the direct reader (not an error in the
    data - the caller falls back to ``plink2 --export A``). The message
    says why."""


# .bed header: magic number 0x6C 0x1B, then 0x01 for SNP-major ("variant
# major") order - the only layout this reader supports.
_BED_MAGIC = bytes([0x6C, 0x1B, 0x01])

# Max variants decoded at once: bounds the `lut[rows]` temporary to
# chunk * ceil(N/4) * 4 bytes (e.g. 50 000 x 5 000 samples = 250 MB).
_CHUNK_VARIANTS = 50_000


def _build_lut(counts_a1: bool) -> np.ndarray:
    """256 x 4 int8 table: row = packed byte, column = sample slot in that
    byte (slot 0 = lowest two bits), value = allele count or -1 for missing.

    The 2-bit codes in a .bed are 00 = homozygous for the first allele in
    the .bim (A1), 01 = missing, 10 = heterozygous, 11 = homozygous for the
    second allele (A2). ``counts_a1`` selects which allele the count refers
    to: A1 gives 2/-1/1/0, A2 gives 0/-1/1/2.
    """
    code_to_count = [2, -1, 1, 0] if counts_a1 else [0, -1, 1, 2]
    table = np.empty((256, 4), dtype=np.int8)
    for byte in range(256):
        for slot in range(4):
            table[byte, slot] = code_to_count[(byte >> (2 * slot)) & 3]
    return table


_LUT_A1 = _build_lut(True)
_LUT_A2 = _build_lut(False)


def _file_signature(paths: List[str]) -> Tuple:
    """(size, mtime_ns) of each path - the cache key part that changes when
    a file is rewritten."""
    sig = []
    for path in paths:
        st = os.stat(path)
        sig.append((st.st_size, st.st_mtime_ns))
    return tuple(sig)


# Process-level reader cache: (absolute stem, file signature) -> BedReader.
_READERS: Dict[Tuple, "BedReader"] = {}
_NOTED_ONCE = set()   # (kind, stem) pairs already NOTE-logged in this process


def note_once(kind: str, stem: str, message: str) -> None:
    """Print `message` once per process per (kind, stem)."""
    key = (kind, os.path.abspath(stem))
    if key in _NOTED_ONCE:
        return
    _NOTED_ONCE.add(key)
    print(message)


class BedReader:
    """Read-only, memory-mapped view of one PLINK1 fileset.

    Build it with :meth:`open` (process-level cached); never pickle it - it
    holds an open memory map, and every process should open its own.
    """

    def __init__(self, stem: str, snp_ids: np.ndarray, fam: pd.DataFrame,
                 bed: np.memmap, bytes_per_variant: int, signature: Tuple) -> None:
        self.stem = stem
        self.snp_ids = snp_ids                      # (M,) object array of str, .bim order
        self.fam = fam                              # columns fid, iid (str), .fam order
        self._bed = bed                             # (M, bytes_per_variant) uint8, read-only
        self.bytes_per_variant = bytes_per_variant
        self.n_samples = int(fam.shape[0])
        self.n_variants = int(snp_ids.shape[0])
        self.signature = signature
        self._index = pd.Index(snp_ids)             # unique by construction (checked in open)
        self._lut: Optional[np.ndarray] = None      # set by a passed probe
        self._counts_label = ''                     # 'A1' / 'A2', for logging
        self._probe_result: Optional[bool] = None   # None = not probed yet
        self.probe_detail = ''

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    @classmethod
    def open(cls, bfile_stem: str) -> "BedReader":
        """Open (or fetch from the process cache) the reader for
        ``<bfile_stem>.bed/.bim/.fam``.

        Raises
        ------
        BedFormatUnsupported
            Bad magic number / not SNP-major, duplicate SNP IDs in the
            .bim, ``.bed`` size not equal to ``3 + M * ceil(N/4)``, or any
            read error. The caller falls back to ``plink2 --export A``.
        PlinkFilesetError
            One of the three files is missing (same error as the legacy path).
        """
        bed_path, bim_path, fam_path = validate_plink_fileset(bfile_stem)
        try:
            signature = _file_signature([bed_path, bim_path, fam_path])
        except OSError as exc:
            raise BedFormatUnsupported(f"cannot stat the fileset ({exc})") from exc
        key = (os.path.abspath(bfile_stem), signature)
        cached = _READERS.get(key)
        if cached is not None:
            return cached

        try:
            with open(bed_path, 'rb') as handle:
                header = handle.read(3)
            if header != _BED_MAGIC:
                raise BedFormatUnsupported(
                    f"'{bed_path}' header is {header.hex()} - expected 6c1b01 (SNP-major PLINK1 "
                    f"binary); sample-major or non-PLINK1 files are not supported"
                )
            bim_ids = pd.read_csv(
                bim_path, sep=r'\s+', header=None, usecols=[1], dtype=str, encoding='utf-8-sig',
            ).iloc[:, 0].to_numpy(dtype=object)
            fam = pd.read_csv(
                fam_path, sep=r'\s+', header=None, usecols=[0, 1], names=['fid', 'iid'],
                dtype=str, encoding='utf-8-sig',
            )
        except BedFormatUnsupported:
            raise
        except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise BedFormatUnsupported(f"could not read the fileset ({type(exc).__name__}: {exc})") from exc

        n_variants, n_samples = int(bim_ids.shape[0]), int(fam.shape[0])
        if n_variants == 0 or n_samples == 0:
            raise BedFormatUnsupported(f".bim has {n_variants} variant(s) and .fam has {n_samples} sample(s)")
        if not pd.Index(bim_ids).is_unique:
            raise BedFormatUnsupported(
                "the .bim contains duplicate SNP IDs, which the legacy export path resolves in a "
                "plink2-specific way"
            )
        bytes_per_variant = (n_samples + 3) // 4
        expected_size = 3 + n_variants * bytes_per_variant
        actual_size = os.path.getsize(bed_path)
        if actual_size != expected_size:
            raise BedFormatUnsupported(
                f"'{bed_path}' is {actual_size} bytes but {n_variants} variants x {n_samples} samples "
                f"needs exactly {expected_size}"
            )
        try:
            bed = np.memmap(bed_path, dtype=np.uint8, mode='r', offset=3,
                            shape=(n_variants, bytes_per_variant))
        except (OSError, ValueError) as exc:
            raise BedFormatUnsupported(f"could not memory-map '{bed_path}' ({exc})") from exc

        reader = cls(bfile_stem, bim_ids, fam, bed, bytes_per_variant, signature)
        # Drop readers of an older version of the same stem (files rewritten).
        for stale in [k for k in _READERS if k[0] == key[0]]:
            _READERS.pop(stale, None)
        _READERS[key] = reader
        reader._log_iid_coercion_difference()
        return reader

    def _log_iid_coercion_difference(self) -> None:
        """Blueprint D7 / risk register: say how many IIDs the legacy
        parser would have altered (``001`` -> ``1``)."""
        try:
            iids = self.fam['iid'].tolist()
            coerced = pd.read_csv(io.StringIO('\n'.join(iids) + '\n'), header=None,
                                  encoding='utf-8-sig').iloc[:, 0].astype(str).tolist()
            n_changed = sum(1 for a, b in zip(iids, coerced) if a != b)
        except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
            return
        if n_changed:
            print(f"[plink_io] NOTE: {n_changed} of {len(iids)} .fam IID(s) in '{self.stem}' are "
                  f"numeric-looking and the legacy plink2 text parser would have altered them "
                  f"(e.g. '{next(a for a, b in zip(iids, coerced) if a != b)}' -> "
                  f"'{next(b for a, b in zip(iids, coerced) if a != b)}'). The direct .bed reader "
                  f"keeps the .fam text, so the phenotype file's 'ID' column must carry the same "
                  f"text (blueprint D7).")

    # ------------------------------------------------------------------ #
    # decoding
    # ------------------------------------------------------------------ #
    def _rows_for_samples(self, keep_iid_list: Optional[List[str]]) -> np.ndarray:
        """Indices (into .fam order) of the samples to keep.

        Mirrors the legacy path: each requested IID is resolved to the FID
        of its LAST .fam row (``dict(zip(iid, fid))``), plink2 then keeps
        the .fam rows whose (FID, IID) pair is in that set.
        """
        if keep_iid_list is None:
            return np.arange(self.n_samples, dtype=np.int64)
        fid_by_iid = dict(zip(self.fam['iid'], self.fam['fid']))
        missing = [iid for iid in keep_iid_list if iid not in fid_by_iid]
        if missing:
            raise PlinkFilesetError(
                f"keep_iid_list contains {len(missing)} IID(s) not present in "
                f"'{self.stem}.fam': {missing[:10]}{', ...' if len(missing) > 10 else ''}"
            )
        wanted = {(fid_by_iid[iid], iid) for iid in keep_iid_list}
        pairs = zip(self.fam['fid'].tolist(), self.fam['iid'].tolist())
        mask = np.fromiter((pair in wanted for pair in pairs), dtype=bool, count=self.n_samples)
        rows = np.flatnonzero(mask)
        if rows.size == 0:
            raise PlinkFilesetError(
                f"keep_iid_list matched no sample of '{self.stem}.fam' (requested {len(wanted)})."
            )
        return rows

    def _variants_for_snps(self, extract_snp_list: Optional[List[str]]) -> np.ndarray:
        """Sorted (.bim order) variant indices for the requested SNP IDs;
        IDs absent from the .bim are ignored, none left -> PlinkFilesetError."""
        if extract_snp_list is None:
            return np.arange(self.n_variants, dtype=np.int64)
        positions = self._index.get_indexer(pd.Index(list(extract_snp_list)))
        positions = np.unique(positions[positions >= 0])
        if positions.size == 0:
            raise PlinkFilesetError(
                f"None of the {len(extract_snp_list)} requested SNP ID(s) exist in "
                f"'{self.stem}.bim' - was extract_snp_list empty, or did every marker get "
                f"filtered out?"
            )
        return positions.astype(np.int64)

    def _decode(self, var_idx: np.ndarray, samp_rows: np.ndarray, lut: np.ndarray) -> np.ndarray:
        """(n_keep, k) int8 matrix of allele counts, -1 = missing, in
        `samp_rows` x `var_idx` order. Chunked over variants."""
        n_keep = int(samp_rows.shape[0])
        out = np.empty((n_keep, int(var_idx.shape[0])), dtype=np.int8)
        take_all_samples = (n_keep == self.n_samples)
        for lo in range(0, int(var_idx.shape[0]), _CHUNK_VARIANTS):
            idx = var_idx[lo:lo + _CHUNK_VARIANTS]
            if idx[-1] - idx[0] + 1 == idx.shape[0]:
                packed = np.asarray(self._bed[idx[0]:idx[-1] + 1])      # contiguous: a slice, no gather
            else:
                packed = np.asarray(self._bed[idx])                    # gather rows -> private copy
            decoded = lut[packed].reshape(idx.shape[0], -1)[:, :self.n_samples]
            if not take_all_samples:
                decoded = decoded[:, samp_rows]
            out[:, lo:lo + idx.shape[0]] = decoded.T
        return out

    def _frame(self, var_idx: np.ndarray, samp_rows: np.ndarray, lut: np.ndarray) -> pd.DataFrame:
        """DataFrame ['IID', *snps] with the legacy dtypes: no-missing
        columns int64, columns with any missing call float64 + NaN."""
        counts = self._decode(var_idx, samp_rows, lut)
        names = self.snp_ids[var_idx].tolist()
        has_missing = (counts == -1).any(axis=0)
        iid_col = pd.Series(self.fam['iid'].to_numpy(dtype=object)[samp_rows], name='IID')

        if not has_missing.any():
            body = pd.DataFrame(counts.astype(np.int64), columns=names)
        elif has_missing.all():
            as_float = counts.astype(np.float64)
            as_float[counts == -1] = np.nan
            body = pd.DataFrame(as_float, columns=names)
        else:
            ok_pos = np.flatnonzero(~has_missing)
            miss_pos = np.flatnonzero(has_missing)
            ints = pd.DataFrame(counts[:, ok_pos].astype(np.int64), columns=[names[i] for i in ok_pos])
            floats = counts[:, miss_pos].astype(np.float64)
            floats[counts[:, miss_pos] == -1] = np.nan
            floats = pd.DataFrame(floats, columns=[names[i] for i in miss_pos])
            joined = pd.concat([ints, floats], axis=1)
            # restore .bim order: position of each wanted column inside `joined`
            order = np.empty(len(names), dtype=np.int64)
            order[ok_pos] = np.arange(ok_pos.shape[0])
            order[miss_pos] = ok_pos.shape[0] + np.arange(miss_pos.shape[0])
            body = joined.iloc[:, order]
        body.insert(0, 'IID', iid_col.to_numpy(dtype=object))
        return body

    def genotype_frame(self, keep_iid_list: Optional[List[str]] = None,
                       extract_snp_list: Optional[List[str]] = None) -> pd.DataFrame:
        """The direct equivalent of the legacy ``_do_convert`` result.

        Parameters
        ----------
        keep_iid_list : IIDs to keep (``--keep``), or None for everyone.
        extract_snp_list : SNP IDs to keep (``--extract``), or None for all.

        Returns
        -------
        DataFrame with columns ``['IID', *snps]`` (``IID`` is ``str``),
        rows in ``.fam`` order, columns in ``.bim`` order.

        Raises
        ------
        RuntimeError
            If :meth:`probe` has not passed on this reader (the allele
            coding is unverified - the caller must probe first).
        PlinkFilesetError
            Unknown IIDs in `keep_iid_list`; no sample / no SNP left.
        """
        if self._lut is None:
            raise RuntimeError("BedReader.genotype_frame() called before a passing probe(); "
                               "the counted allele is unverified.")
        samp_rows = self._rows_for_samples(keep_iid_list)
        var_idx = self._variants_for_snps(extract_snp_list)
        return self._frame(var_idx, samp_rows, self._lut)

    # ------------------------------------------------------------------ #
    # counted-allele parity probe
    # ------------------------------------------------------------------ #
    def probe(self, plink_path: str = 'plink2', threads: int = 1,
              n_variants: int = 16, n_samples: int = 64, timeout_s: float = 900.0) -> bool:
        """Verify the decoder against the legacy ``plink2 --export A`` on a
        small block, once per process per fileset (the answer is cached).

        Steps: pick up to `n_variants` variants and `n_samples` samples
        spread evenly over the file; run the legacy export on exactly those;
        read the counted allele from each ``<SNP>_<allele>`` header and
        compare it with .bim A1/A2 to choose the lookup table; decode the
        same block and require ``DataFrame.equals`` with the legacy block.

        Returns True when the fast path is safe. On any problem (plink2
        missing/failing, mixed counted alleles, value mismatch) returns
        False and leaves the reason in ``self.probe_detail`` - this method
        never raises, because "cannot verify" simply means "do not use the
        fast path".
        """
        if self._probe_result is not None:
            return self._probe_result
        self._probe_result = False
        try:
            self._probe_result = self._run_probe(plink_path, threads, n_variants, n_samples, timeout_s)
        except (OSError, ValueError, KeyError, subprocess.SubprocessError, pd.errors.ParserError,
                pd.errors.EmptyDataError, PlinkFilesetError) as exc:
            self.probe_detail = f"probe could not complete ({type(exc).__name__}: {exc})"
            self._probe_result = False
        return bool(self._probe_result)

    def _run_probe(self, plink_path: str, threads: int, n_variants: int, n_samples: int,
                   timeout_s: float) -> bool:
        v_idx = np.unique(np.linspace(0, self.n_variants - 1, num=min(n_variants, self.n_variants)).astype(np.int64))
        s_rows = np.unique(np.linspace(0, self.n_samples - 1, num=min(n_samples, self.n_samples)).astype(np.int64))
        snps = self.snp_ids[v_idx].tolist()
        bim_alleles = pd.read_csv(
            f"{self.stem}.bim", sep=r'\s+', header=None, usecols=[4, 5], dtype=str,
            encoding='utf-8-sig',
        ).iloc[v_idx].to_numpy(dtype=object)
        fids = self.fam['fid'].to_numpy(dtype=object)[s_rows].tolist()
        iids = self.fam['iid'].to_numpy(dtype=object)[s_rows].tolist()

        with tempfile.TemporaryDirectory(prefix='plink_bed_probe_') as tmp:
            extract_path = os.path.join(tmp, 'extract.txt')
            keep_path = os.path.join(tmp, 'keep.txt')
            with open(extract_path, 'w', encoding='utf-8') as handle:
                handle.write('\n'.join(snps) + '\n')
            with open(keep_path, 'w', encoding='utf-8') as handle:
                for fid, iid in zip(fids, iids):
                    handle.write(f"{fid}\t{iid}\n")
            prefix = os.path.join(tmp, 'probe')
            cmd = [plink_path, '--bfile', self.stem, '--allow-extra-chr', '--threads', str(max(1, int(threads))),
                   '--extract', extract_path, '--keep', keep_path, '--export', 'A', '--out', prefix]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
            except FileNotFoundError as exc:
                self.probe_detail = f"'{plink_path}' not found ({exc})"
                return False
            except subprocess.TimeoutExpired:
                self.probe_detail = f"legacy export timed out after {timeout_s:.0f}s"
                return False
            if result.returncode != 0 or not os.path.isfile(f"{prefix}.raw"):
                self.probe_detail = f"legacy export failed (exit {result.returncode}): {result.stderr[-300:]}"
                return False
            raw = pd.read_csv(f"{prefix}.raw", sep=r'\s+', encoding='utf-8-sig')

        marker_cols = [c for c in raw.columns if c not in ('FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE')]
        if len(marker_cols) != len(snps):
            self.probe_detail = (f"legacy export returned {len(marker_cols)} marker column(s) for "
                                 f"{len(snps)} requested variant(s)")
            return False

        # Which allele does the legacy export count? Learn it from the headers.
        votes = set()
        for col, snp, (a1, a2) in zip(marker_cols, snps, bim_alleles):
            m = re.match(r'^(.*)_([^_]+)$', col)
            if m is None or m.group(1) != snp:
                self.probe_detail = f"unexpected export header '{col}' for SNP '{snp}'"
                return False
            counted = m.group(2)
            if counted == a1 and counted != a2:
                votes.add('A1')
            elif counted == a2 and counted != a1:
                votes.add('A2')
            elif counted != a1 and counted != a2:
                self.probe_detail = (f"export counted allele '{counted}' of SNP '{snp}' is neither .bim "
                                     f"A1 '{a1}' nor A2 '{a2}'")
                return False
            # counted == a1 == a2 (monomorphic): ambiguous, no vote
        if len(votes) > 1:
            self.probe_detail = ("the legacy export counts A1 for some probed variants and A2 for "
                                 "others (per-variant allele choice) - one lookup table cannot match it")
            return False
        label = votes.pop() if votes else 'A1'
        lut = _LUT_A1 if label == 'A1' else _LUT_A2

        ours = self._frame(v_idx, s_rows, lut)
        # Compare the marker values/dtypes only: the legacy IID column is
        # subject to the D7 numeric coercion, which is a deliberate,
        # logged difference and says nothing about the allele decoding.
        legacy = raw.rename(columns=dict(zip(marker_cols, snps)))[snps].reset_index(drop=True)
        ours_markers = ours[snps].reset_index(drop=True)
        if not ours_markers.equals(legacy):
            bad = [s for s in snps if not ours_markers[s].equals(legacy[s])]
            self.probe_detail = (f"decoded block differs from the legacy export for {len(bad)} of "
                                 f"{len(snps)} probed variant(s), e.g. {bad[:3]}")
            return False

        self._lut = lut
        self._counts_label = label
        self.probe_detail = (f"counted allele = .bim {label}; decoded block identical to the legacy "
                             f"export on {len(snps)} variant(s) x {len(iids)} sample(s)")
        return True
