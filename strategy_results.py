"""
strategy_results.py
====================
Requirements.md items 1-3 (multi-option update) - the single authority for
how the per-OPTION result files of a multi-option ASReml run (BLUEs / BLUPs
without GRM / BLUPs with GRM, any subset) are combined into ONE set of
result files carrying an option-distinguishing column, and for how a
reader takes one option's rows back out again.

Why a separate module
---------------------
`genomic_prediction.GP()` is a ~10k-line, strongly single-strategy engine
(every per-task cache, ASReml fit, accumulator and checkpoint assumes
exactly one genotype-effect strategy). Rather than thread a strategy
dimension through all of it, `GP()` is now a thin wrapper that runs that
UNCHANGED engine once per selected option, each into its own scratch
RESULT_NAME, and calls `merge_option_results()` below to write the usual
result files (`Metric.csv`, `Marker_effect.csv`, ...) with the extra
`OPTION_COLUMN` column FIRST. Everything downstream (assemble.py,
batch_reader.ResultSet, the reporting pipeline) then only has to know
"if this column exists, filter on it".

Design rules
------------
* Zero project imports except `checkpoint_utils` (paths/keys) and
  `asreml_syntax` (labels): this module sits at the same layer as
  `batch_reader.py` and never imports a plotting or model module.
* The column is `asreml_syntax.STRATEGY_COLUMN` ('asreml_option'), values
  are the human-readable labels ('BLUEs', 'BLUPs without GRM', 'BLUPs
  with GRM'). It is written as the FIRST column of every file so that
  filtering it out (`option_view()`) restores the original column order -
  the positional contract (`Marker_effect.csv`'s 5 leading metadata
  columns) that scatter_plot/circos_plot rely on is never disturbed.
* A run with a single option NEVER gets the column (byte-identical to
  before, I11) - `merge_option_results()` is only ever called for >= 2.
"""

from __future__ import annotations

import csv
import gzip
import os
import shutil
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

import checkpoint_utils as _ckpt
import asreml_syntax as _syn

OPTION_COLUMN = _syn.STRATEGY_COLUMN


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _open_text(path: str, mode: str = 'r'):
    if path.endswith('.gz'):
        return gzip.open(path, mode + 't', newline='', encoding='utf-8')
    return open(path, mode, newline='', encoding='utf-8')


def option_tag(label: str) -> str:
    """Filesystem-safe folder/file tag for an option label ('BLUEs',
    'BLUPs_without_GRM', 'BLUPs_with_GRM'); unknown labels are sanitised."""
    for key, lab in _syn.STRATEGY_LABELS.items():
        if lab == label:
            return _syn.STRATEGY_FILE_TAGS[key]
    import re as _re
    return _re.sub(r'[^A-Za-z0-9_.-]+', '_', str(label)) or 'option'


def option_result_name(result_name: str, label: str) -> str:
    """RESULT_NAME of one option's OUTPUT sub-folder
    (`Result/<result_name>/<tag>/`) - where the per-option plots/summaries
    written by the reporting pipeline live."""
    return os.path.join(result_name, option_tag(label))


def canonical_option_order(labels: Iterable[str]) -> List[str]:
    """`labels` (unique, any order) sorted into BLUEs, BLUPs without GRM,
    BLUPs with GRM order; unknown labels follow, in first-seen order."""
    labels = list(dict.fromkeys(labels))
    canonical = [_syn.STRATEGY_LABELS[k] for k in _syn.GENOTYPE_EFFECT_STRATEGIES]
    return [l for l in canonical if l in labels] + [l for l in labels if l not in canonical]


def options_in_frame(df: "Optional[pd.DataFrame]") -> List[str]:
    """The option labels present in `df` (canonical order), or `[]` when
    the frame has no option column (a single-option / non-ASReml run)."""
    if df is None or OPTION_COLUMN not in getattr(df, 'columns', []):
        return []
    return canonical_option_order(pd.unique(df[OPTION_COLUMN].dropna()).tolist())


def add_option_column(df: "pd.DataFrame", label: str) -> "pd.DataFrame":
    """Copy of `df` with `OPTION_COLUMN`=label inserted as the FIRST column.
    A frame with no columns at all is returned unchanged (nothing to tag)."""
    if df is None or df.shape[1] == 0:
        return df
    out = df.copy()
    if OPTION_COLUMN in out.columns:
        out = out.drop(columns=[OPTION_COLUMN])
    out.insert(0, OPTION_COLUMN, label)
    return out


def option_view(df: "Optional[pd.DataFrame]", label: Optional[str]) -> "Optional[pd.DataFrame]":
    """Rows of `df` belonging to option `label`, with `OPTION_COLUMN`
    DROPPED and the index reset - i.e. exactly the frame a single-option
    run would have produced, so every existing consumer (metric_plot,
    scatter_plot, circos_plot, ...) works on it unchanged.

    `label=None` or a frame without the column -> `df` returned as-is
    (never a copy, never an error), which is what lets the same call site
    serve single-option and multi-option runs."""
    if df is None or label is None or OPTION_COLUMN not in df.columns:
        return df
    out = df.loc[df[OPTION_COLUMN] == label].drop(columns=[OPTION_COLUMN])
    return out.reset_index(drop=True)


def concat_with_option(frames: "Sequence[Tuple[str, pd.DataFrame]]") -> "pd.DataFrame":
    """In-memory counterpart of `merge_option_results()`: `[(label, frame),
    ...]` -> one frame with the option column first. Empty/columnless
    frames are skipped; an all-empty input returns an empty DataFrame."""
    parts = [add_option_column(f, lab) for lab, f in frames if f is not None and f.shape[1] != 0]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True, sort=False)
    ordered = [OPTION_COLUMN] + [c for c in out.columns if c != OPTION_COLUMN]
    return out[ordered]


# --------------------------------------------------------------------------- #
# On-disk merge of the per-option result files
# --------------------------------------------------------------------------- #

def _read_header(path: str) -> "Optional[List[str]]":
    with _open_text(path, 'r') as f:
        first = f.readline()
    if not first:
        return None                      # zero-byte file: no header at all
    return next(csv.reader([first]), [])


def _merge_one_key(sources: "Sequence[Tuple[str, str]]", out_path: str, gzip_out: bool) -> int:
    """Stream `sources` (`[(label, path), ...]`, existing files only) into
    `out_path` with `OPTION_COLUMN` first. Column union in first-seen
    order (the exact policy `batch_reader.header_union()` uses), missing
    cells left empty. Written to a temp file and atomically renamed, so a
    crash never leaves a half-written parent file that a later resume
    could mistake for a complete one. Returns the data rows written."""
    headers: Dict[str, List[str]] = {}
    union: List[str] = []
    seen = set()
    for label, path in sources:
        header = _read_header(path)
        if header is None:
            continue
        header = [c for c in header if c != OPTION_COLUMN]
        headers[path] = header
        for col in header:
            if col not in seen:
                seen.add(col)
                union.append(col)
    if not union:
        return 0                         # nothing usable at all for this key

    union_index = {c: i for i, c in enumerate(union)}
    n_union = len(union)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    tmp_path = out_path + '.tmp_options'
    rows_written = 0
    opener = gzip.open if gzip_out else open
    mode = 'wt' if gzip_out else 'w'
    with opener(tmp_path, mode, newline='', encoding='utf-8') as out_f:
        writer = csv.writer(out_f)
        writer.writerow([OPTION_COLUMN] + union)
        for label, path in sources:
            header = headers.get(path)
            if header is None:
                continue
            with _open_text(path, 'r') as in_f:
                reader = csv.reader(in_f)
                try:
                    raw_header = next(reader)
                except StopIteration:
                    continue
                keep = [i for i, c in enumerate(raw_header) if c != OPTION_COLUMN]
                fill_pairs = [(union_index[raw_header[i]], i) for i in keep]
                same = (header == union)
                for row in reader:
                    if len(row) != len(raw_header):
                        raise ValueError(
                            f"[strategy_results] Ragged row in the '{label}' option's file "
                            f"'{path}': header has {len(raw_header)} field(s), a data row "
                            f"(around line {reader.line_num}) has {len(row)}."
                        )
                    if same and len(keep) == len(raw_header):
                        writer.writerow([label] + row)
                    else:
                        out_row = [''] * n_union
                        for dest_i, src_i in fill_pairs:
                            out_row[dest_i] = row[src_i]
                        writer.writerow([label] + out_row)
                    rows_written += 1
    os.replace(tmp_path, out_path)
    return rows_written


def merge_option_results(parent_result_name: str, idx: int, parallel: bool,
                         nested: "Sequence[Tuple[str, str]]", *,
                         compression: Optional[str] = None) -> Dict[str, int]:
    """Write the parent's standard result files (every key of
    `checkpoint_utils.RESULT_FILE_NAMES`) as the option-tagged union of the
    per-option nested runs.

    Parameters
    ----------
    parent_result_name : the RESULT_NAME the caller of `GP()` asked for.
    idx, parallel      : the SAME batch id / parallel flag `GP()` itself
        resolved (`checkpoint_utils.result_file_paths()`'s own arguments),
        so the parent files land exactly where a single-option run's would.
    nested             : `[(option_label, nested_result_name), ...]`, in
        canonical option order.
    compression        : the run's RESULT_COMPRESSION ('gzip'/'none'/None).

    Returns `{key: rows_written}` for every key that produced a file.
    Nested files are located with `resolve_result_path()` (gzip or plain,
    whichever exists), so this is independent of the compression setting
    the nested runs happened to use."""
    out_paths = _ckpt.result_file_paths(parent_result_name, idx, parallel, compression=compression)
    written: Dict[str, int] = {}
    for key in _ckpt.RESULT_FILE_NAMES:
        sources = []
        for label, nested_name in nested:
            path = _ckpt.resolve_result_path(nested_name, idx, parallel, key)
            if path is not None:
                sources.append((label, path))
        if not sources:
            continue
        out_path = out_paths[key]
        gzip_out = out_path.endswith('.gz')
        # A stale file under the OTHER compression extension (a re-run
        # after RESULT_COMPRESSION changed) must not linger and shadow
        # the fresh one - `resolve_result_path()` prefers `.gz`.
        other = out_path[:-3] if gzip_out else out_path + '.gz'
        if os.path.isfile(other):
            try:
                os.remove(other)
            except OSError:
                pass
        rows = _merge_one_key(sources, out_path, gzip_out)
        if rows or os.path.isfile(out_path):
            written[key] = rows
    return written


def read_option_frame(result_name: str, idx: int, parallel: bool, key: str) -> "pd.DataFrame":
    """One result file (whichever compression exists) as a DataFrame, empty
    when absent/unreadable - used by `GP()`'s wrapper to rebuild in-memory
    return frames for options that were satisfied from disk."""
    path = _ckpt.resolve_result_path(result_name, idx, parallel, key)
    if path is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, EOFError, gzip.BadGzipFile):
        return pd.DataFrame()


def remove_nested_run(nested_result_name: str) -> None:
    """Delete one option's scratch folder (after its results were merged
    and any side artefacts hoisted). Never raises."""
    from pipeline_utils import result_dir_path
    path = result_dir_path(nested_result_name)
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
    except OSError as exc:
        print(f"[strategy_results] WARNING: could not remove option scratch folder '{path}': "
              f"{exc!r} - safe to delete manually; it does not affect any result.")
