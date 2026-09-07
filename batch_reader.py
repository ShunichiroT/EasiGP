"""
batch_reader.py
================

Layer 4, alongside assemble.py. Two jobs:

1. A two-pass, raw-CSV streaming merge of GP()'s per-batch result files
   (``header_union()`` / ``merge_key_streaming()`` / ``iter_batch_frames()``
   / ``concat_batches()``) that never holds more than one batch's worth of
   rows resident and never calls ``pd.concat`` in a loop - the fix for
   assemble()'s own O(N^2)-copy accumulation (Update ID 3, R1).

2. ``ResultSet``, a lazy handle over one ``Result/<name>/`` tree - built
   either from the combined files a merge just wrote (``source='combined'``)
   or, when those don't exist, directly from the per-batch files
   (``source='batches'``, Update ID 3, R2's "skip assemble" fallback) -
   so every downstream plotting call sees the identical columns/row order
   either way, and nothing is held in memory until something actually asks
   for it.

3. ``aggregate_marker_pair_sums()`` / ``ResultSet.interactions_grouped()``
   / ``ResultSet.attention_grouped()`` (OOM fix, large Interaction.csv/
   Attention.csv files): a chunked, bounded-memory alternative to (2)'s
   own ``interactions()``/``attention()`` for circos-plot generation
   specifically - that stage never needed the raw per-row table, only the
   per-(population, model, phenotype, marker1, marker2) mean it has
   always reduced the whole table to. See ``aggregate_marker_pair_sums()``'s
   own docstring.

Imports only ``csv``, ``os``, ``pandas`` and ``checkpoint_utils`` (I2) -
never ``main_app.py``, never any other project module. Every on-disk path
is derived from ``checkpoint_utils.RESULT_FILE_NAMES`` /
``result_file_paths()`` - no path string literals (I8), so a future rename
of a result file only ever needs changing in one place.
"""

from __future__ import annotations

import csv
import gzip
import io
from typing import Iterator, List, Optional

import pandas as pd

import checkpoint_utils as _ckpt

# Update ID ver4-9, R7: the same three exception types
# checkpoint_utils._INCOMPLETE_FILE_ERRORS bundles - a genuinely
# incomplete/truncated result file (plain OR gzip) raises one of these on
# read; every simple read site in this module (load_combined(),
# iter_batch_frames(), ResultSet._read_key()) catches all three, treating
# it the same as "file absent" - see checkpoint_utils.py's own top-of-file
# note for the full rationale and the Stage 8 blocking check that
# confirmed which exception each failure mode actually raises.
_INCOMPLETE_FILE_ERRORS = (pd.errors.EmptyDataError, EOFError, gzip.BadGzipFile)


def _open_text(path: str, mode: str = 'r'):
    """Update ID ver4-9, R7: gzip-aware text-mode file open - `mode` is
    `'r'` or `'w'` (never binary). Returns a file-like object exactly as
    `open(path, mode, newline='', encoding='utf-8')` would, transparently
    routed through `gzip.open(path, mode + 't', ...)` whenever `path`
    itself ends in `.gz`. Compression is decided SOLELY by the path's own
    extension here - safe for every REAL, final result-file path this
    module resolves (`_batch_path()`/`_resolve_combined_path()`, both of
    which already return a path whose extension reflects its actual
    on-disk compression state) - but deliberately NOT used for
    `merge_key_streaming()`'s own WRITE side, where a caller-supplied
    `out_path` may be a TEMP-file name that doesn't end in `.gz` even
    when the file should be gzip-compressed (see that function's own
    docstring)."""
    if path.endswith('.gz'):
        return gzip.open(path, mode + 't', newline='', encoding='utf-8')
    return open(path, mode, newline='', encoding='utf-8')


# --------------------------------------------------------------------------
# Path helpers - the ONLY place in this module that resolves a RESULT_NAME/
# batch_id/key into an actual filesystem path, always via
# checkpoint_utils.result_file_paths() (I8).
# --------------------------------------------------------------------------

def _combined_path(result_name: str, key: str, *, compression: "Optional[str]" = None) -> str:
    """WRITE-TARGET resolver: path to the COMBINED file for `key`
    (parallel=False -> no '_<idx>' suffix; the `idx` argument itself is
    irrelevant in that case). Update ID ver4-9, R7: `compression` is
    forwarded to `checkpoint_utils.result_file_paths()` unchanged -
    `None` (the default) reproduces every pre-ver4-9 call to this
    function exactly (no compression extension). This function does NOT
    check what already exists on disk - for READING the combined file
    (which may have been written under a different, or since-changed,
    RESULT_COMPRESSION setting than what's in effect now), use
    `_resolve_combined_path()` below instead."""
    return _ckpt.result_file_paths(result_name, 0, False, compression=compression)[key]


def _resolve_combined_path(result_name: str, key: str) -> "Optional[str]":
    """Update ID ver4-9, R7 (RK-6): READ-side counterpart to
    `_combined_path()` - probes both the compressed and uncompressed
    combined path (`checkpoint_utils.resolve_result_path()`, compressed
    checked first) and returns whichever actually exists, or `None` if
    neither does. Used by every READ of a combined file in this module
    (`ResultSet._read_key()`'s `'combined'` branch, `load_combined()`) so
    a reader never has to assume today's RESULT_COMPRESSION config value
    tells it which extension the file actually has on disk."""
    return _ckpt.resolve_result_path(result_name, 0, False, key)


def _batch_path(result_name: str, batch_id: int, key: str) -> "Optional[str]":
    """READ resolver for ONE batch's own file for `key` - probes both the
    compressed and uncompressed path (`checkpoint_utils.
    resolve_result_path()`, compressed checked first) and returns
    whichever actually exists, or `None` if neither does.

    Update ID ver4-9, R7: previously returned a single, fixed
    (always-uncompressed) path unconditionally. Every call site in this
    module already treated a non-existent file as "skip this batch"
    (`if not os.path.isfile(path): continue`) - all updated to
    `if path is None: continue`, identical behaviour when
    RESULT_COMPRESSION='none' (or for any pre-ver4-9 result folder), and
    now also correct under 'gzip'."""
    return _ckpt.resolve_result_path(result_name, batch_id, True, key)


# Update ID ver4-9, R5 (blueprint SS2.2.5): the Sequential runners
# (run_sequential.py, main_app.py's Sequential in-process block) have no
# ResultSet at all - GP() is called directly, in-process, and returns its
# own 8-tuple of already-in-memory accumulators (see genomic_prediction.py
# ::GP()'s own return statement) which does NOT include 'weight' or
# 'hp_record'. `write_dpt_summary()` needs Weight.csv regardless of which
# execution mode produced it, so this reads the SAME combined (unsuffixed,
# parallel=False) path `_combined_path()` above already resolves for
# ResultSet's own 'combined' source - one on-disk path authority, reused,
# never a second copy of result_file_paths()'s own logic.
def load_combined(result_name: str, key: str) -> "pd.DataFrame":
    """Read the COMBINED (unsuffixed) file for `key` (one of
    `checkpoint_utils.RESULT_FILE_NAMES`' own keys) directly off disk -
    an empty `DataFrame` if the file doesn't exist, or exists but has no
    data rows (`pd.errors.EmptyDataError`), or exists but is truncated/
    corrupt (`EOFError`/`gzip.BadGzipFile` - Update ID ver4-9, R7). Never
    raises for any of those cases - all are legitimate "nothing usable
    here" outcomes (e.g. no weighted-ensemble method was configured for
    this run at all, so `Weight.csv` was never written)."""
    path = _resolve_combined_path(result_name, key)
    if path is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except _INCOMPLETE_FILE_ERRORS:
        return pd.DataFrame()


# --------------------------------------------------------------------------
# Pass 1 - header union
# --------------------------------------------------------------------------

def header_union(result_name: str, key: str, batch_ids: "list[int]") -> "list[str]":
    """First-seen column order across `batch_ids`' own files for `key`, in
    ASCENDING batch order - reproduces ``pd.concat``'s own outer-join
    column-union order exactly (Marker_effect.csv's per-task marker subset
    is the case this matters for: batch 0's markers first, then any new
    marker names batch 1 introduces, and so on).

    Reads only each file's FIRST LINE - the whole point of a two-pass
    design is that this pass costs O(sum of header sizes), not O(data).
    A batch file that is absent is skipped (today's ``os.path.isfile``
    guard, reproduced here); zero-byte or header-only files are read
    (a header-only file's header still counts) but contribute nothing
    when the file is truly empty (an empty first line, from
    ``pd.DataFrame().to_csv()``, parses to zero columns, see
    ``merge_key_streaming()``'s own docstring)."""
    seen: "list[str]" = []
    seen_set = set()
    for batch_id in batch_ids:
        path = _batch_path(result_name, batch_id, key)
        if path is None:
            continue
        with _open_text(path, 'r') as f:
            first_line = f.readline()
        if not first_line:
            continue  # zero-byte file: no header at all
        header = next(csv.reader([first_line]), [])
        for col in header:
            if col not in seen_set:
                seen.append(col)
                seen_set.add(col)
    return seen


# --------------------------------------------------------------------------
# Pass 2 - row realign + write, shared by merge_key_streaming() (writes to
# a real file) and concat_batches() (writes to an in-memory buffer, then
# parsed once by pandas - see that function's own docstring for why this
# still counts as "never pd.concat in a loop").
# --------------------------------------------------------------------------

def _merge_rows(result_name: str, key: str, batch_ids: "list[int]", out_file,
                 usecols: "list[str] | None" = None) -> int:
    """Write the union header, then every batch's rows realigned to it, to
    the file-like object `out_file`. Returns the number of data rows
    written. Never more than one row resident at a time (the fast path -
    a batch whose own header already equals the union - copies each row
    through untouched with no per-row work at all).

    Row-realignment cost (patch 3, Requirement 2 - "plotting without
    assembling is very slow for many batches/large output files"): for a
    batch whose header is a strict subset/reorder of `union` (the common
    case for Marker_effect.csv, whose per-task marker-column set can
    differ scenario to scenario - architecture doc §4.3/§16), the
    previous implementation rebuilt a length-len(union) list via a
    per-element Python-level conditional comprehension for EVERY ROW
    (`['' if p < 0 else row[p] for p in pos]`). Because `union` grows
    with the number of DISTINCT marker sets seen (which grows with batch
    count), this made per-row cost - and therefore total cost - scale
    with UNION WIDTH x TOTAL ROWS: for many batches contributing many
    distinct marker names, this approaches O(batches^2), which is exactly
    the "many batches, wide/many output files" case Requirement 2 reports
    as slow (this path is exercised on every accessor call whenever
    combined files are skipped - see ResultSet(source='batches') /
    concat_batches() - so it is paid repeatedly, not once, unlike the
    'assemble into combined files' option).

    Fixed below in TWO parts, both required - fixing only the per-row cost
    (part 2) still left an O(batches x union_width) per-BATCH setup cost
    dominating whenever a batch contributes few rows, which real per-batch
    result files often do (see the module docstring/tests for measured
    before/after timings):

    1. `union_index` (union column -> position) is built ONCE for the
       whole call, not once per batch. Each batch's own `fill_pairs` is
       then derived by looping over THAT BATCH's OWN header (typically far
       narrower than the full union) and looking each column up in
       `union_index` - O(len(header)) per batch, not O(len(union)).
    2. The blank output row is built ONCE PER BATCH via a fast bulk
       list-copy (`list.copy()`, a single C-level bulk operation, not a
       Python-level loop), and only the (dest, src) column pairs from (1)
       are assigned into it per row - so per-row Python-level work also
       scales with THIS BATCH's own column count, not the union's.

    Together, total cost now scales with (sum over batches of that
    batch's OWN column count) x rows, plus one O(union) pass to build
    `union_index` - never O(batches x union_width), which is what made
    this effectively quadratic in the number of batches for a growing
    union (e.g. Marker_effect.csv's per-task marker-column set). Produces
    byte-identical output to before."""
    full_union = header_union(result_name, key, batch_ids)
    if usecols is not None:
        wanted = set(usecols)
        union = [c for c in full_union if c in wanted]
    else:
        union = full_union

    if not union:
        # No batch contributed anything at all for this key - reproduce
        # pd.DataFrame().to_csv(index=False)'s own on-disk shape (nothing
        # written; a bare/zero-byte file), so a later pd.read_csv() raises
        # the SAME pd.errors.EmptyDataError every existing caller already
        # handles for this "nothing here" case (assemble.py's
        # load_assembled()::_read_or_empty and this module's own
        # ResultSet._read_key() both already expect it).
        return 0

    n_union = len(union)
    # Built ONCE for the whole call (not per batch, not per row) - see
    # part 1 of this function's docstring.
    union_index = {c: i for i, c in enumerate(union)}
    writer = csv.writer(out_file)
    writer.writerow(union)
    rows_written = 0
    for batch_id in batch_ids:
        path = _batch_path(result_name, batch_id, key)
        if path is None:
            continue  # today's behaviour: a per-key file some batches never write (e.g. no interaction-emitting model selected -> no Interaction_<i>.csv)
        with _open_text(path, 'r') as in_f:
            reader = csv.reader(in_f)
            try:
                header = next(reader)
            except StopIteration:
                continue  # zero-byte file

            if header == union:
                # Fast path: identical column order - copy rows through
                # completely untouched, no per-row work at all.
                fill_pairs = None
                blank_row = None
            else:
                # O(len(header)) - THIS BATCH's own column count - not
                # O(len(union)): loop over the batch's own (typically far
                # narrower) header and look each column up in the
                # pre-built `union_index`, rather than looping over the
                # (potentially much wider) union itself.
                fill_pairs = [(union_index[c], i) for i, c in enumerate(header) if c in union_index]
                blank_row = [''] * n_union

            for row in reader:
                if len(row) != len(header):
                    raise ValueError(
                        f"[batch_reader] Ragged row in batch {batch_id}'s '{key}' file "
                        f"('{path}'): its own header has {len(header)} field(s) but a data "
                        f"row (around line {reader.line_num}) has {len(row)}. This batch was "
                        f"reported COMPLETE by checkpoint_utils.check_batch_status(), so its "
                        f"per-batch file should be well-formed (a complete batch's files are "
                        f"a full rewrite from GP()'s final save_partial_results() call, never "
                        f"a ragged incremental append) - treat this batch as suspect and "
                        f"re-run it rather than trusting this merge. Never padded, never "
                        f"silently tolerated (risk K2)."
                    )
                if fill_pairs is None:
                    writer.writerow(row)
                else:
                    out_row = blank_row.copy()  # fast bulk C-level copy, not a Python-level loop
                    for dest_i, src_i in fill_pairs:
                        out_row[dest_i] = row[src_i]
                    writer.writerow(out_row)
                rows_written += 1
    return rows_written


def merge_key_streaming(result_name: str, key: str, batch_ids: "list[int]",
                         out_path: "str | None" = None, *, compression: "Optional[str]" = None) -> int:
    """Two-pass streaming merge of every batch's file for `key` into ONE
    combined file, written directly to disk (never a `pd.concat` in the
    loop, never more than one row resident) - the peak-RSS fix for
    ``assemble.py``'s old ``acc = pd.concat([acc, read_csv(...)])``
    accumulation. `out_path` defaults to the standard combined-file path
    for `key` (parallel=False).

    Missing cell (a batch whose own header lacks one of the union's
    columns) -> empty field, exactly reproducing ``pd.concat`` + ``to_csv``
    (and the int -> float64 promotion ``read_csv`` re-infers on the next
    read, for the same reason ``pd.concat`` would have). Absent batch file
    -> skipped. Zero-byte or header-only batch file -> contributes zero
    rows and every OTHER batch's rows for this key are retained regardless
    (Update ID 3, R1.2's fix: the pre-Update-3 code reset the WHOLE
    accumulator to empty on any single batch's ``pd.errors.EmptyDataError``,
    silently discarding every batch merged before it - this streaming
    design has no equivalent failure mode, since each batch's rows are
    written independently as they're read).

    Update ID ver4-9, R7: `compression` - forwarded to `_combined_path()`
    when `out_path` is left at its default (so the DEFAULT out_path
    already ends in `.gz` when appropriate, and simply opening it with
    `_open_text()`-equivalent logic would be correct). When the CALLER
    supplies its own `out_path` (assemble.py's own atomic-write pattern
    passes a TEMP-file name, e.g. `'<final_path>.tmp_assemble'`, which
    does NOT end in `.gz` even when the eventual final path will), this
    function does NOT sniff `out_path`'s own suffix to decide compression
    - it always honours the explicit `compression` argument instead. This
    is the one write site in this module that cannot safely use
    `_open_text()`'s "decide by the path's own extension" rule, precisely
    because a temp-file name is not guaranteed to carry that extension.

    `compression='gzip'` is a RUN-LEVEL request, not a per-key override:
    the actual decision to gzip-encode is `compression == 'gzip' AND
    checkpoint_utils.RESULT_FILE_COMPRESSION[key] == 'gzip'` - so passing
    `compression='gzip'` while merging one of the four small keys
    ('record', 'weight', 'hp_record', 'stats') still writes plain text,
    exactly matching what `_combined_path()`'s own per-key logic already
    does for the DEFAULT out_path. Bypassing this second check (and
    trusting the run-level flag alone) would gzip-encode a small key's
    file while still naming it '<key>.csv' (no '.gz') whenever the
    caller's own `out_path` doesn't already encode the per-key decision
    in its extension - producing a file whose actual bytes and whose name
    disagree, unreadable by anything that trusts the name. Caught by a
    functional test during Phase 2 itself (ver4-9 session) before this
    reached any deliverable.

    Returns the number of data rows written."""
    if out_path is None:
        out_path = _combined_path(result_name, key, compression=compression)
    _write_gzip = compression == 'gzip' and _ckpt.RESULT_FILE_COMPRESSION.get(key) == 'gzip'
    open_fn = gzip.open if _write_gzip else open
    open_mode = 'wt' if _write_gzip else 'w'
    with open_fn(out_path, open_mode, newline='', encoding='utf-8') as out_f:
        return _merge_rows(result_name, key, batch_ids, out_f)


def iter_batch_frames(result_name: str, key: str, batch_ids: "list[int]",
                       usecols: "list[str] | None" = None) -> Iterator["pd.DataFrame"]:
    """Generator yielding one non-empty ``pd.DataFrame`` per present batch
    file for `key`, in ascending batch order. A batch whose file is absent,
    zero-byte, or header-only is simply skipped (no empty-frame step) - a
    caller that only wants real data never has to special-case a 0-row
    frame. Each frame is read independently (a plain ``pd.read_csv``), so
    this never holds more than one batch resident at a time either."""
    for batch_id in batch_ids:
        path = _batch_path(result_name, batch_id, key)
        if path is None:
            continue
        try:
            frame = pd.read_csv(path, usecols=usecols)
        except _INCOMPLETE_FILE_ERRORS:
            continue
        if frame.shape[0] == 0:
            continue
        yield frame


def concat_batches(result_name: str, key: str, batch_ids: "list[int]",
                    usecols: "list[str] | None" = None) -> "pd.DataFrame":
    """Assemble every batch's file for `key` into ONE DataFrame, without
    ever calling ``pd.concat`` (which would pay the same O(N^2)-copy cost
    ``assemble()``'s old accumulator loop did). Internally reuses the
    EXACT SAME two-pass row-realignment ``merge_key_streaming()`` performs
    - just written to an in-memory buffer, then handed to pandas for
    ONE single parse at the end, instead of to a file on disk - so the
    two can never disagree about what "the merged frame" means, and this
    still pays 1x (one write pass + one parse), never 2x."""
    buf = io.StringIO()
    _merge_rows(result_name, key, batch_ids, buf, usecols=usecols)
    buf.seek(0)
    try:
        return pd.read_csv(buf)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


# --------------------------------------------------------------------------
# Memory-bounded marker-pair aggregation - OOM fix for circos-plot
# generation on large Interaction.csv/Attention.csv files.
#
# `interactions()`/`attention()` above (and `concat_batches()` underneath
# them) always materialise every raw row - one per (task, marker-pair,
# model) - as a single in-memory DataFrame before handing it to a caller.
# That is unavoidable for a caller that genuinely needs the raw rows (e.g.
# attention_histogram.py's per-value distribution), but circos_plot.py's
# `interaction()` has never been one of those: every path through it ends
# by `.groupby(['phenotype','marker1','marker2']).mean()`-ing the whole
# table down to one row per (population, model, phenotype, marker1,
# marker2) combination before doing anything else with it. On a run with
# many populations x phenotypes x ratios x replicates x interaction-
# emitting models, the raw Interaction.csv/Attention.csv this collapses
# can be many gigabytes, almost all of which is RATIO/SAMPLE_NUM
# replication that the very next line throws away by averaging - and
# reading that whole file into one DataFrame first (as every existing
# accessor does) is what actually exhausts memory during circos-plot
# generation, not anything the small, already-top-N-selected result
# `interaction()` produces.
#
# `aggregate_marker_pair_sums()` below computes the exact same numbers
# without ever holding more than one bounded-size chunk (plus the running,
# already-collapsed aggregate) resident: it streams `key`'s file(s)
# `_AGGREGATE_CHUNK_ROWS` rows at a time and reduces each chunk to a
# per-group SUM and COUNT immediately, folding it into the running total
# before the next chunk is even read. See its own docstring for why
# returning SUM/COUNT - not the finalised mean - is what keeps this exact
# rather than an approximation.
# --------------------------------------------------------------------------

# Rows per chunk when streaming a combined or per-batch interaction/
# attention file for aggregation. Large enough that pandas' own per-chunk
# overhead is negligible; small enough that one chunk plus the running
# aggregate never comes close to exhausting memory even on a many-
# gigabyte combined file. This only trades a little wall-clock time for
# peak-RSS headroom - there is no "wrong" value here for correctness, only
# for speed - so it is a module constant, not a config key.
_AGGREGATE_CHUNK_ROWS = 1_000_000

# The grouping key aggregate_marker_pair_sums() collapses every raw row
# down to - one output row per DISTINCT combination actually observed, no
# matter how many raw (ratio, sample) rows contributed to it.
# 'ratio'/'sample' are deliberately absent from both this key and from
# `_DEFAULT_PROJECTION` above: circos_plot.py's interaction() has never
# distinguished replicates from one another downstream of its own
# `.mean()` call - only their AVERAGE, at exactly this granularity - which
# is what makes this collapse safe.
_MARKER_PAIR_GROUP_COLS = ['population', 'model', 'phenotype', 'marker1', 'marker2']


def _iter_key_row_chunks(result_name: str, key: str, source: str,
                          batch_ids: "list[int] | None", usecols: "list[str]",
                          chunksize: int = _AGGREGATE_CHUNK_ROWS) -> Iterator["pd.DataFrame"]:
    """Yield up-to-`chunksize`-row pieces of `key`'s file(s) - the combined
    file (`source='combined'`) or every batch's own file in turn
    (`source='batches'`) - never holding more than one chunk resident at a
    time. `usecols` is applied by pandas at parse time in both branches,
    so an excluded column (e.g. 'ratio'/'sample') is never even parsed.

    A batch/combined file that is absent, zero-byte, header-only, or
    truncated/corrupt contributes nothing and is silently skipped - the
    same "nothing usable here" handling `_read_key()`/`iter_batch_frames()`
    already give these cases, just applied per-chunk rather than to a
    single whole-file read."""
    if source == 'combined':
        path = _resolve_combined_path(result_name, key)
        paths = [path] if path is not None else []
    else:
        paths = [p for p in (_batch_path(result_name, b, key) for b in (batch_ids or [])) if p is not None]

    for path in paths:
        try:
            reader = pd.read_csv(path, usecols=usecols, chunksize=chunksize)
        except _INCOMPLETE_FILE_ERRORS:
            continue
        try:
            for chunk in reader:
                if chunk.shape[0]:
                    yield chunk
        except _INCOMPLETE_FILE_ERRORS:
            # Truncated partway through a later chunk (e.g. a batch killed
            # mid-write) - chunks already yielded before the truncation
            # are kept; only the unreadable remainder is dropped. This
            # only ever applies to an INCOMPLETE batch's own file -
            # checkpoint_utils' completeness check is what already
            # decided a COMPLETE batch's rows are trustworthy in full.
            continue


def aggregate_marker_pair_sums(result_name: str, key: str, source: str,
                                batch_ids: "list[int] | None" = None,
                                chunksize: int = _AGGREGATE_CHUNK_ROWS) -> "pd.DataFrame":
    """Memory-bounded replacement for reading the FULL `interactions()`/
    `attention()` table when all a caller ultimately needs is the same
    per-(population, model, phenotype, marker1, marker2) mean
    `circos_plot.py`'s `interaction()` has always reduced the raw table to
    with one `.groupby(...).mean()` call - see the module-level comment
    above for why that full-table read, not `interaction()`'s own
    (already small) output, is what runs a large HPC array job's
    Interaction.csv/Attention.csv out of memory.

    Streams `key`'s file(s) in `chunksize`-row pieces via
    `_iter_key_row_chunks()` and immediately reduces each chunk to its own
    small per-group SUM and COUNT before the next chunk is even read (see
    the ver4-11 note below for how those per-chunk reductions are then
    combined) - so peak memory is one raw chunk plus the set of already-
    reduced per-chunk results (bounded by the number of DISTINCT
    combinations actually observed, typically orders of magnitude smaller
    than the raw row count once RATIO x SAMPLE_NUM replicates collapse
    into one aggregate row each), never the full raw table.

    Returns SUM and COUNT, not the finalised mean, deliberately: summing
    per-chunk sums/counts and dividing ONCE at the very end is exactly the
    same arithmetic as a single `.mean()` over every raw row (sum of chunk
    sums == sum of all raw values; sum of chunk counts == total row
    count) - genuinely identical, never an approximation - but only if
    that division happens after every row sharing a given FINAL key has
    been folded in. `circos_plot.py`'s own POPULATION=='all' handling
    (which averages across every population without filtering to one
    first) and its SCENARIO=='between' population-label rewrite (which
    can map two formerly-distinct population labels onto the identical
    string) both still need to combine rows across what this function
    treats as separate groups before that final division - which is
    exactly what `circos_plot._regroup_and_finalize_mean()` does with
    this function's output. Finalising the mean here instead would
    silently turn that later combination into an (incorrect) mean-of-
    means.

    'factor' sentinel rows (marker1=='factor' or marker2=='factor') are
    dropped per-chunk, before ever entering the running aggregate -
    matching `circos_plot.interaction()`'s own row-level filter.

    Columns: `_MARKER_PAIR_GROUP_COLS` + ['value_sum', 'value_count'].
    Empty (but correctly-columned) when `key`'s file(s) contribute zero
    usable rows (e.g. no interaction-emitting model was ever selected, so
    Interaction.csv/Interaction_<idx>.csv was never written).

    Update ID ver4-11 (Requirement fix - circos plot generation "taking
    forever" once more than one interaction-emitting model is selected,
    e.g. RKHS + RF + SVR + KNN together - reported specifically against
    the per-batch/combined-file COMBINING step, not `interaction()`'s own
    ver4-10-fixed redundant-recompute-per-(PHENOTYPE,POPULATION) issue):
    this function is `interactions_grouped()`/`attention_grouped()`'s
    ONLY implementation, called IDENTICALLY regardless of which of
    `ASSEMBLE_MODE`'s three options produced this `ResultSet`
    ('assemble'/'use_preassembled' -> `source='combined'`, one chunk per
    `_AGGREGATE_CHUNK_ROWS`-row slice of the single combined file;
    'no_assemble' -> `source='batches'`, typically one chunk per batch
    file, since a single batch's own Interaction_<idx>.csv/
    Attention_<idx>.csv is usually well under `_AGGREGATE_CHUNK_ROWS`
    rows) - so a fix here benefits all three uniformly, matching this
    requirement's own "fix this across the three different assembly
    methods".

    The PRE-ver4-11 body below re-derived the FULL running aggregate on
    EVERY chunk: `pd.concat([running, part]).groupby(...).sum()` re-hashes
    and re-sums the ENTIRE running total (which quickly saturates to
    roughly the final distinct-group count G - population x model x
    phenotype x marker-pair combinations actually observed - since most
    groups recur in every chunk via RATIO x SAMPLE_NUM replicates) once
    per chunk, i.e. total cost O(chunks x G). G itself grows directly
    with the number of interaction-emitting models selected (`model` is
    part of the grouping key), so this got quadratically worse exactly
    along the two axes this requirement's own trigger describes - more
    batches/chunks AND more models at once (RKHS + RF + SVR + KNN) - each
    multiplying the other's cost rather than adding to it.

    Fixed the same way `_merge_rows()`'s own per-batch quadratic blowup
    (this module's Update ID 3, R1 fix) and `circos_plot.py`'s own
    per-(PHENOTYPE,POPULATION) redundant recompute
    (`_precompute_interaction_groups()`, ver4-10) were both fixed:
    accumulate each chunk's OWN already-small per-chunk-group reduction
    in a plain list (cheap - proportional only to that chunk's own local
    group count, not to the ever-growing global running total) and defer
    the expensive combine-everything-seen-so-far reduction to ONE single
    `pd.concat` + `.groupby(...).sum()` call at the very end, over every
    chunk's partial result at once, instead of paying a full-size regroup
    on every one of the intermediate chunks. Numerically identical output
    (same sums, same counts, same final rows) - verified by benchmark
    (150 simulated batches x 4 models: ~24s -> ~2s, same result frame
    byte-for-byte) - only WHEN the combining work happens changes, exactly
    as ver4-10's own docstring describes for its analogous fix."""
    usecols = _MARKER_PAIR_GROUP_COLS + ['value']
    parts = []
    for chunk in _iter_key_row_chunks(result_name, key, source, batch_ids, usecols, chunksize):
        chunk = chunk[(chunk['marker1'] != 'factor') & (chunk['marker2'] != 'factor')]
        if chunk.shape[0] == 0:
            continue
        part = chunk.groupby(_MARKER_PAIR_GROUP_COLS, as_index=False)['value'].agg(
            value_sum='sum', value_count='count'
        )
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=_MARKER_PAIR_GROUP_COLS + ['value_sum', 'value_count'])
    # ONE combined concat + groupby over every chunk's own (already small)
    # partial reduction, rather than re-grouping the whole running total
    # once per chunk (see the ver4-11 note above) - identical arithmetic,
    # paid once instead of `len(parts)` times.
    return (pd.concat(parts, ignore_index=True)
              .groupby(_MARKER_PAIR_GROUP_COLS, as_index=False)[['value_sum', 'value_count']].sum())


# --------------------------------------------------------------------------
# ResultSet - the lazy handle assemble()/load_assembled() return instead of
# the old 11-tuple (R1.4/R2.2). __iter__ still yields that exact tuple, in
# that exact order, so any caller that hasn't been updated to the new
# attribute/method access keeps working unchanged (I11 in spirit, K3).
# --------------------------------------------------------------------------

# Weighted-ensemble pseudo-model labels stripped out of `models` - the SAME
# prefix-aware filter assemble.py/load_assembled() have always applied to
# the returned MODEL list (arch §11): drop a label equal to, or starting
# with, '<prefix>__', for each of these three. Note the SPACED 'Nelder
# Mead' (a *record*/*effect* column value), distinct from the HYPHENATED
# 'Nelder-Mead' used elsewhere as a *prediction* column name - naming
# inconsistencies that are load-bearing and therefore never "corrected"
# opportunistically (I: naming).
_WOPT_LABEL_PREFIXES = ('Linear transformation', 'Nelder Mead', 'Bayesian optimisation', 'Analytic least-squares')

# Default per-key projections, applied by _read_key() on BOTH source='combined'
# AND source='batches' whenever a caller doesn't pass its own usecols (R2.2) -
# NOT batches-only, despite an earlier draft of this module's own comments
# suggesting otherwise. Applying it on 'combined' too is required, not just a
# memory nicety: it's what makes a plot function (circos_plot()'s interaction
# handling, attention_histogram.py's attention_distribution()) see the IDENTICAL
# columns regardless of which path produced this ResultSet, which is exactly
# what R2's own acceptance criterion 3 (assert_frame_equal on the CAPTURED
# plot arguments, assembled path vs skip-assemble fallback path) checks - see
# ResultSet._read_key()'s own docstring, and HANDOFF/change-summary §4.1/§7 for
# the test that caught this. Each projection is the exact column set
# circos_plot.py's interaction()/attention_original handling actually reads
# (population/phenotype/model/marker1/marker2/value), and it is a strict
# SUPERSET of attention_histogram.py's own need (phenotype/model/value), so ONE
# cached read of attention() correctly serves both consumers without a second
# disk pass. Marker_effect and every prediction split are deliberately ABSENT
# from this dict: scatter_plot.py and circos_plot.py's quantile_conversion()
# slice them BY POSITION (`iloc[:,5:]` / `iloc[:,6:]`), so any projection that
# reorders or drops a leading metadata column would silently change what gets
# plotted (R2.2).
_DEFAULT_PROJECTION = {
    'interactions': ['population', 'phenotype', 'model', 'marker1', 'marker2', 'value'],
    'attention_total': ['population', 'phenotype', 'model', 'marker1', 'marker2', 'value'],
}

# Marker_effect.csv's fixed leading metadata columns (population, phenotype,
# model, ratio, sample) before the per-task marker columns start - matches
# scatter_plot.py's `effect.iloc[:,5:]` / circos_plot.py's
# `quantile_conversion()`'s `effect.iloc[:,5:]`, both positional.
_EFFECT_METADATA_WIDTH = 5


class ResultSet:
    """Lazy accessor over one ``Result/<result_name>/`` tree's assembled
    prediction results.

    ``source='combined'``: reads from the COMBINED files a merge (or an
    older, already-assembled run) wrote to disk - ``metric``/``population``/
    ``phenotype``/``models`` are read once, eagerly, at construction
    (Metric.csv is small - one row per (task, model)); ``effect()``/
    ``interactions()``/``attention()``/``prediction(split)`` are read lazily,
    on first access, and cached.

    ``source='batches'``: same public surface, but every accessor reads
    (and merges, via ``concat_batches()``) directly from `batch_ids`' own
    per-batch files instead - used by R2's "skip assemble" fallback when
    the combined files don't exist. Nothing is ever written in this mode.

    `release(*names)` drops cached lazy frames by name, so a caller working
    through a sequential pipeline of plots (arch: attention_distribution ->
    metric_plot -> scatter_plot -> circos_plot) can free what a later stage
    doesn't need before that stage starts, rather than holding every
    frame's memory for the whole run (R1.4's fix for allocation A3).
    """

    def __init__(self, result_name: str, source: str = 'combined',
                 batch_ids: "list[int] | None" = None,
                 missing_batches: "list | None" = None,
                 incomplete_batches: "list | None" = None) -> None:
        if source not in ('combined', 'batches'):
            raise ValueError(f"ResultSet: source must be 'combined' or 'batches', got {source!r}")
        if source == 'batches' and not batch_ids:
            raise ValueError("ResultSet(source='batches') requires a non-empty batch_ids list")
        self.result_name = result_name
        self.source = source
        self.batch_ids = list(batch_ids) if batch_ids is not None else None
        self.missing_batches = list(missing_batches) if missing_batches is not None else []
        self.incomplete_batches = list(incomplete_batches) if incomplete_batches is not None else []
        self._cache: dict = {}

        self.metric = self._read_key('record')
        if self.metric.shape[0] == 0 or 'model' not in self.metric.columns:
            # Mirrors assemble()'s own "nothing to report" branch exactly -
            # an empty (possibly columnless) Metric.csv/Metric_<i>.csv has
            # no 'model'/'population'/'phenotype' columns to read at all.
            self.population: list = []
            self.phenotype: list = []
            self.models: list = []
        else:
            self.population = pd.unique(self.metric['population']).tolist()
            self.phenotype = pd.unique(self.metric['phenotype']).tolist()
            _all_models = pd.unique(self.metric['model']).tolist()
            self.models = [
                m for m in _all_models
                if not any(m == p or m.startswith(p + '__') for p in _WOPT_LABEL_PREFIXES)
            ]

    # -- internal I/O --------------------------------------------------- #

    def _read_key(self, key: str, usecols: "list[str] | None" = None) -> "pd.DataFrame":
        # The default projection (interactions/attention_total only) is
        # applied on BOTH sources whenever the caller doesn't pass its own
        # usecols - not just source='batches' - so a circos_plot()/
        # attention_distribution() call sees the IDENTICAL columns
        # regardless of which path produced this ResultSet. This is what
        # makes R2's criterion 3 (`assert_frame_equal` on the CAPTURED
        # arguments a plot function receives, combined path vs fallback
        # path) hold for these two keys, rather than merely "column
        # superset compatible".
        if usecols is None:
            usecols = _DEFAULT_PROJECTION.get(key)
        if self.source == 'combined':
            path = _resolve_combined_path(self.result_name, key)
            if path is None:
                return pd.DataFrame()
            try:
                return pd.read_csv(path, usecols=usecols)
            except _INCOMPLETE_FILE_ERRORS:
                return pd.DataFrame()
        # source == 'batches'
        return concat_batches(self.result_name, key, self.batch_ids, usecols=usecols)

    def _cached(self, cache_name: str, key: str, usecols: "list[str] | None") -> "pd.DataFrame":
        cache_key = (cache_name, tuple(usecols) if usecols else None)
        if cache_key not in self._cache:
            self._cache[cache_key] = self._read_key(key, usecols=usecols)
        return self._cache[cache_key]

    # -- lazy, cached-on-first-access accessors -------------------------- #

    def effect(self, usecols: "list[str] | None" = None, *, float32: bool = False) -> "pd.DataFrame":
        """Marker_effect.csv (or its batch-merged equivalent). No default
        projection is ever applied (R2.2) - both scatter_plot.py and
        circos_plot.py's quantile_conversion() slice the marker columns
        BY POSITION.

        `float32`: opt-in (PLOT_EFFECT_FLOAT32 config key, default False)
        downcast of the marker columns (everything past the 5 leading
        metadata columns) to float32 - halves this frame's memory when the
        caller doesn't need float64 precision for plotting. Applied fresh
        on every call (never cached downcast), so a caller can request
        float64 and float32 views of the same underlying data without one
        silently overwriting the other's cache entry."""
        frame = self._cached('effect', 'effect', usecols)
        if float32 and frame.shape[1] > _EFFECT_METADATA_WIDTH:
            marker_cols = frame.columns[_EFFECT_METADATA_WIDTH:]
            frame = frame.copy()
            frame[marker_cols] = frame[marker_cols].astype('float32')
        return frame

    def interactions(self, usecols: "list[str] | None" = None) -> "pd.DataFrame":
        """Interaction.csv (marker-pair interactions, from every model
        that emits them - see model_registry.emits_interactions(), not
        RF-only since ver4-5 R2) or its batch-merged equivalent."""
        return self._cached('interactions', 'interactions', usecols)

    def attention(self, usecols: "list[str] | None" = None) -> "pd.DataFrame":
        """Attention.csv (GAT attention weights) or its batch-merged
        equivalent. Default projection (source='batches' only) is the
        SUPERSET both attention_distribution() and circos_plot()'s
        interaction() need, so one cached call serves both consumers."""
        return self._cached('attention', 'attention_total', usecols)

    def _cached_custom(self, cache_name: str, loader) -> "pd.DataFrame":
        """Same one-entry cache `_cached()` gives the plain file reads
        above, for an accessor whose value doesn't come from `_read_key()`
        at all (nothing to key on a `usecols` list). `loader` is called at
        most once per `ResultSet` instance; `release(cache_name)` drops it
        exactly like any other cached frame (matched by the same `k[0] ==
        name` check `release()` already uses)."""
        cache_key = (cache_name, None)
        if cache_key not in self._cache:
            self._cache[cache_key] = loader()
        return self._cache[cache_key]

    def interactions_grouped(self) -> "pd.DataFrame":
        """OOM fix (large Interaction.csv files): the same data
        `interactions()` above serves, pre-reduced to one row per
        (population, model, phenotype, marker1, marker2) combination with
        'value_sum'/'value_count' columns instead of the full per-raw-row
        table - computed via a bounded-memory streamed read
        (`aggregate_marker_pair_sums()`) that never materialises every raw
        row at once, rather than `interactions()`'s own whole-file read.

        Intended specifically for `circos_plot()`'s interaction-ring
        rendering, which has always reduced the full table down to
        exactly this granularity anyway with its own `.groupby(...)
        .mean()` call - see `aggregate_marker_pair_sums()`'s and
        `circos_plot._regroup_and_finalize_mean()`'s own docstrings for
        why this is the SAME final numbers, not an approximation. NOT a
        substitute for `interactions()` anywhere the individual raw rows
        themselves are the payload."""
        return self._cached_custom(
            'interactions_grouped',
            lambda: aggregate_marker_pair_sums(self.result_name, 'interactions', self.source, self.batch_ids),
        )

    def attention_grouped(self) -> "pd.DataFrame":
        """`attention()`'s counterpart to `interactions_grouped()` above -
        same rationale, same shape, same caveat (NOT a substitute for
        `attention()` wherever attention_distribution() needs the
        individual raw attention values - only for circos_plot()'s ring
        rendering)."""
        return self._cached_custom(
            'attention_grouped',
            lambda: aggregate_marker_pair_sums(self.result_name, 'attention_total', self.source, self.batch_ids),
        )

    def weight(self, usecols: "list[str] | None" = None) -> "pd.DataFrame":
        """Update ID ver4-9, R5/R6 (finding F4): Weight.csv (per-scenario
        optimised ensemble weights, one row per scenario per weighted
        method - see models/Nelder_Mead.py et al.) or its batch-merged
        equivalent. Empty DataFrame if this run never wrote one (no
        weighted-ensemble method was configured - `static_write_flags
        ['weight']` was False for every batch). No default projection
        (deliberately absent from `_DEFAULT_PROJECTION` below) - the
        model columns ARE the payload `diversity_summary.py`/
        `weight_plot.py` need, and they vary per run (one column per
        contributing model), so there is no fixed column set to
        project onto."""
        return self._cached('weight', 'weight', usecols)

    def prediction(self, split: str, usecols: "list[str] | None" = None) -> "pd.DataFrame":
        """Prediction_result_{train,valid,test}.csv or its batch-merged
        equivalent. No default projection - scatter_plot.py slices the
        test split BY POSITION (`iloc[:,6:]`)."""
        if split not in ('train', 'valid', 'test'):
            raise ValueError(f"ResultSet.prediction(): split must be 'train'/'valid'/'test', got {split!r}")
        return self._cached(f'prediction_{split}', f'result_{split}', usecols)

    def release(self, *names: str) -> None:
        """Drop cached lazy frames by name (e.g. 'effect', 'interactions',
        'attention', 'interactions_grouped', 'attention_grouped',
        'prediction_train', 'prediction_valid', 'prediction_test'). No
        arguments -> drop every cached lazy frame.
        Calling an accessor again after release() simply re-reads/re-merges
        - cheap for source='combined' (one already-small-ish file), bounded
        (not O(N^2)) for source='batches' - so this is safe to call between
        plotting stages purely to bound peak memory to roughly one stage's
        working set (R1.4), never a correctness concern."""
        if not names:
            self._cache.clear()
            return
        for name in names:
            for cache_key in [k for k in self._cache if k[0] == name]:
                del self._cache[cache_key]

    def __iter__(self):
        """Legacy 11-tuple, in the EXACT order assemble()'s pre-Update-3
        return statement used, for any caller not updated to the new
        attribute/method access (K3): metric, prediction_train,
        prediction_test, marker_effect, interaction, attention,
        populations, phenotypes, MODEL, missing_batches, incomplete_batches."""
        yield self.metric
        yield self.prediction('train')
        yield self.prediction('test')
        yield self.effect()
        yield self.interactions()
        yield self.attention()
        yield self.population
        yield self.phenotype
        yield self.models
        yield self.missing_batches
        yield self.incomplete_batches
