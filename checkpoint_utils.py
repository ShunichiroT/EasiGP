"""
checkpoint_utils.py
====================

Checkpoint/resume support for GP()'s main per-scenario task loop
(genomic_prediction.py). If GP() raises partway through processing its
list of (population, phenotype, ratio, replicate) scenarios - a bad
hyperparameter combination, an out-of-memory model fit, a transient I/O
error, etc. - every scenario that already finished successfully is saved
to the SAME result files GP() would have produced on a full, successful
run (see save_partial_results() / RESULT_FILE_NAMES below), along with a
small checkpoint file recording exactly which scenario to resume from.
The next time the same job is submitted (same RESULT_NAME, same
population/phenotype/ratio/replicate list, same PARALLEL batch, if any),
GP() loads that checkpoint automatically, skips every already-completed
scenario, and continues from the one that failed - see
genomic_prediction.py's GP() for how this is actually wired in.

Checkpointing is per (RESULT_NAME, PARALLEL batch_id) - Sequential runs
(PARALLEL=None) and each Parallel batch each get their own independent
checkpoint file and result files (exactly mirroring the existing
Metric.csv vs Metric_<idx>.csv split - see result_file_paths()), so
fixing and resuming ONE failed Parallel batch never touches any other
batch's already-completed, already-saved results.

WHAT COUNTS AS "COMPLETED"
--------------------------------------------------------------------------
A scenario only counts as completed - and only ever appears in the saved
result files / advances the checkpoint - once EVERY model configured for
it has finished successfully. If scenario i's first model succeeds but
its second model raises, scenario i's checkpoint state is rolled back to
'not completed' (see genomic_prediction.py's per-task snapshot/rollback)
and none of its partial rows are saved - so a resumed run always cleanly
re-does the ENTIRE failed scenario from scratch, never a half-finished
one, which would otherwise risk duplicate rows for whichever model(s) did
complete before the failure.

WHAT THIS DOES NOT PROTECT AGAINST
--------------------------------------------------------------------------
This only saves progress when GP() catches a Python exception raised
while processing a single scenario. It does NOT protect against the
process being killed outright (SIGKILL, an out-of-memory kill from the
OS/scheduler rather than a catchable MemoryError, a power loss, etc.) -
the same result files/checkpoint that would ordinarily be written on a
caught error are, by definition, never reached in that case either.
Progress is only as safe as whichever save point - a previous caught
error, or a previous fully-successful completion - most recently wrote
these files.
"""

from __future__ import annotations

import glob
import gzip
import hashlib
import json
import os
import re
from typing import Dict, List, Optional, Sequence

import pandas as pd

from models.hyperparameter_tuning import base_of, schema_key_of
from model_registry import emits_interactions, emits_attention
from pipeline_utils import result_dir_path

# Keys for GP()'s 10 accumulator DataFrames, and the (unsuffixed) output
# filename each one is written to - PARALLEL runs get '_<idx>' inserted
# before the extension (see result_file_paths()), exactly matching GP()'s
# original, pre-checkpointing file-naming convention. (Finding F8: this
# comment previously said "9" - it has always been 10 keys; fixed here,
# no behaviour change.)
RESULT_FILE_NAMES = {
    'record': 'Metric.csv',
    'result_train': 'Prediction_result_train.csv',
    'result_valid': 'Prediction_result_valid.csv',
    'result_test': 'Prediction_result_test.csv',
    'effect': 'Marker_effect.csv',
    'interactions': 'Interaction.csv',
    'attention_total': 'Attention.csv',
    'weight': 'Weight.csv',
    'hp_record': 'hyperparameter.csv',
    'stats': 'Basic_stats.csv',
}

# Update ID ver4-9, R7 (design blueprint SS2.4.2): a sibling of
# RESULT_FILE_NAMES (same module, same keys - the same "never let two
# dicts name the same accumulator differently" discipline
# MODEL_GROUP_MERGE_SPEC below already follows, I8) - which compression
# scheme, when RESULT_COMPRESSION='gzip' is in effect, applies to EACH
# accumulator. The six large, per-individual/per-marker/per-pair files
# compress well and dominate a Result folder's disk usage
# (Prediction_result_{train,valid,test}.csv, Marker_effect.csv,
# Interaction.csv, Attention.csv); the four small, per-scenario-or-
# coarser files (Metric.csv, Weight.csv, hyperparameter.csv,
# Basic_stats.csv) are left as plain CSV UNCONDITIONALLY - gzip's own
# per-file overhead isn't worth paying for files that are already small,
# and leaving them uncompressed keeps a quick `head`/`cat`/spreadsheet-
# double-click workflow available for exactly the files a person is most
# likely to want to eyeball directly.
RESULT_FILE_COMPRESSION = {
    'record': None,
    'result_train': 'gzip',
    'result_valid': 'gzip',
    'result_test': 'gzip',
    'effect': 'gzip',
    'interactions': 'gzip',
    'attention_total': 'gzip',
    'weight': None,
    'hp_record': None,
    'stats': None,
}

_COMPRESSION_EXT = {None: '', 'none': '', 'gzip': '.gz'}

# Update ID ver4-9, R7: the three exception types a read of a possibly-
# gzip-compressed result file can legitimately raise for a genuinely
# INCOMPLETE (not corrupt-forever) file - a batch killed mid-append can
# leave a truncated '.gz' behind, exactly as a killed batch could always
# leave a truncated plain CSV behind (pandas' own pd.errors.EmptyDataError
# for a zero-byte file). Bundled as one tuple so every read site below
# (and in batch_reader.py/assemble.py) catches the SAME three types,
# never just a subset - verified directly (Stage 8 blocking check, ver4-9
# Phase 2 session) that a truncated multi-member gzip file raises
# EOFError, and a corrupted/non-gzip header raises gzip.BadGzipFile (an
# OSError subclass, NOT an EOFError subclass) - both must be listed
# explicitly.
_INCOMPLETE_FILE_ERRORS = (pd.errors.EmptyDataError, EOFError, gzip.BadGzipFile)

# These are always written by save_partial_results() (even if empty) -
# matches GP()'s own original, unconditional writes for the first four.
# 'stats' (Requirement: basic per-scenario statistics - individual counts
# per split, marker count) is written unconditionally alongside them,
# since a row is added to it at the exact same point a scenario's own
# 'record' row(s) become possible (right before its per-model dispatch
# loop - see genomic_prediction.py's GP()), so the two always have the
# same "has this run produced anything yet" status.
# 'effect'/'hp_record' are written only when non-empty (a DATA condition,
# re-checked fresh every call - see save_partial_results); 'interactions'/
# 'attention_total'/'weight' depend on the run's model selection, not the
# data itself, so the caller (GP()) computes and passes those in as
# `static_write_flags` - see that function's own docstring for why this
# split exists.
ALWAYS_WRITTEN_KEYS = ('record', 'result_train', 'result_valid', 'result_test', 'stats')


# ---------------------------------------------------------------------------
# Update ID 2, R1 - tier-1 (model-group -> task) merge semantics.
#
# Phase 2's own merge tier (intra_batch_parallel.py's
# _merge_isolated_results_into_batch()) combines DIFFERENT TASKS' result
# files, for which pure row-concat is always correct (every task
# contributes wholly disjoint rows). R1 (intra_task_parallel.py) adds a
# tier BELOW that one, merging DIFFERENT MODEL GROUPS' result files for
# the SAME task - and at that tier, three different merge semantics are
# needed depending on the file:
#   - 'rows'       : each group contributes wholly separate rows (record,
#                     effect, interactions, attention_total, hp_record,
#                     weight) - plain concat, ordered back to MODEL_RUN
#                     order afterwards so output matches a serial run.
#   - 'dedup_rows' : every group emits the IDENTICAL row (stats - one row
#                     per TASK, not per model, since Steps 1-8 pool
#                     construction is byte-identical in every group - see
#                     architecture doc §7 Step 8) - concat-then-dedup, or
#                     Basic_stats.csv would gain G copies of the same row.
#   - 'columns'    : each group contributes only ITS OWN model's
#                     prediction column(s) (result_train/valid/test -
#                     architecture doc §7 Step 10, "later models concat a
#                     single prediction COLUMN, axis=1") - an outer merge
#                     on the row identity columns, not a concat.
#
# Kept as a sibling of RESULT_FILE_NAMES (same module, same keys) rather
# than a separate authority, so the two can never name a file differently
# for the same accumulator (I8).
# ---------------------------------------------------------------------------
MODEL_GROUP_MERGE_SPEC: Dict[str, Dict[str, object]] = {
    'record':          {'axis': 'rows', 'key': ('population', 'phenotype', 'model', 'ratio', 'sample')},
    'effect':          {'axis': 'rows', 'key': ('population', 'phenotype', 'model', 'ratio', 'sample')},
    'interactions':    {'axis': 'rows', 'key': ('population', 'phenotype', 'model', 'ratio', 'sample',
                                                 'marker1', 'marker2')},
    'attention_total': {'axis': 'rows', 'key': ('population', 'phenotype', 'model', 'ratio', 'sample',
                                                 'marker1', 'marker2')},
    'hp_record':       {'axis': 'rows', 'key': ('population', 'phenotype', 'model', 'ratio', 'sample',
                                                 'algorithm')},
    # Never actually populated by a model-group unit in this delivery (W_OPT
    # tasks are ineligible for fan-out - see intra_task_parallel.py's
    # eligibility gate / blueprint decision D1) - specified anyway so a
    # future relaxation of that scope decision has a ready merge rule and
    # so this dict stays a complete, single source of truth for all ten
    # RESULT_FILE_NAMES keys, not nine.
    'weight':          {'axis': 'rows', 'key': ('population', 'phenotype', 'model', 'ratio', 'sample')},
    'stats':           {'axis': 'dedup_rows', 'key': ('population', 'phenotype', 'ratio', 'sample')},
    'result_train':    {'axis': 'columns', 'key': ('id', 'population', 'ratio', 'phenotype', 'sample', 'actual')},
    'result_valid':    {'axis': 'columns', 'key': ('id', 'population', 'ratio', 'phenotype', 'sample', 'actual')},
    'result_test':     {'axis': 'columns', 'key': ('id', 'population', 'ratio', 'phenotype', 'sample', 'actual')},
}


def _order_rows_by_model_run(df: "pd.DataFrame", model_run: Sequence[str]) -> "pd.DataFrame":
    """Stable categorical sort of `df`'s 'model' column against
    `model_run`'s order, so a `rows`-axis tier-1 merge produces the same
    ROW ORDER a fully-serial `GP()` call would have (needed for the
    `DataFrame.equals()` acceptance check, since cost-balanced bin-packing
    can place model groups in an order that doesn't match `MODEL_RUN`).
    A `kind='stable'` sort keeps every group's own internal row order
    (e.g. multiple marker1/marker2 rows for one model) unchanged. Rows
    whose 'model' isn't in `model_run` (should not happen, by
    construction) sort last rather than raising, since this is an
    ordering nicety, not a correctness gate."""
    if 'model' not in df.columns or df.shape[0] == 0:
        return df
    order = {name: position for position, name in enumerate(model_run)}
    df = df.copy()
    df['_model_order'] = df['model'].map(order)
    df['_model_order'] = df['_model_order'].fillna(len(order))
    df = df.sort_values('_model_order', kind='stable').drop(columns='_model_order').reset_index(drop=True)
    return df


def _order_prediction_columns(df: "pd.DataFrame", join_cols: Sequence[str],
                               model_run: Sequence[str]) -> "pd.DataFrame":
    """Restore `[*join_cols, *MODEL_RUN-ordered model columns]` after a
    `columns`-axis outer merge, so a tier-1-merged
    `Prediction_result_*.csv`'s column order matches a fully-serial
    `GP()` call's exactly. Any merged column not named after a
    `model_run` entry (should not occur, by construction - every
    prediction column a group contributes is named after one of its own
    dispatched models) is appended at the end rather than silently
    dropped, so a genuine surprise is visible rather than hidden."""
    model_cols = [m for m in model_run if m in df.columns]
    other_cols = [c for c in df.columns if c not in join_cols and c not in model_cols]
    return df[[*join_cols, *model_cols, *other_cols]]


def merge_model_group_frames(key: str, frames: List["pd.DataFrame"], model_run: Sequence[str]) -> "pd.DataFrame":
    """Pure function (no I/O): merge one `RESULT_FILE_NAMES` key's worth
    of per-model-group `DataFrame`s, from the SAME task, into the single
    `DataFrame` a fully-serial `GP()` call would have produced for that
    task - see `MODEL_GROUP_MERGE_SPEC` above for which of the three
    semantics `key` uses.

    Parameters
    ----------
    key : one of `RESULT_FILE_NAMES`' keys (`'record'`, `'effect'`, ...).
    frames : that key's `DataFrame` from every model group that
        dispatched at least one model this task, in any order - group
        order does not need to match `MODEL_RUN` order; ordering is
        restored internally. Empty (0-row) and `None` entries are
        ignored (matching every group's own "only written when there's
        something to write" convention - see `ALWAYS_WRITTEN_KEYS`).
    model_run : the task's own `MODEL_RUN` (post hyperparameter-tuning-
        expansion model list), used purely to restore the same row/
        column order a serial run would have.

    Returns
    -------
    The merged `DataFrame` - empty (0 rows, 0 columns) if every entry in
    `frames` was empty/`None` (e.g. every group skipped this task for the
    same `MIN_DATA_POINTS` reason Steps 1-8 already agree on).

    Raises
    ------
    ValueError if `key` isn't in `MODEL_GROUP_MERGE_SPEC`, or (for a
    `columns`-axis key) if no join column is present in every frame - a
    genuine contract break (I4) that must surface loudly here rather than
    silently merge on the wrong (or no) key.
    """
    if key not in MODEL_GROUP_MERGE_SPEC:
        raise ValueError(f"merge_model_group_frames: no MODEL_GROUP_MERGE_SPEC entry for key {key!r}.")

    usable_frames = [f for f in frames if f is not None and f.shape[0] != 0]
    if not usable_frames:
        return pd.DataFrame()

    spec = MODEL_GROUP_MERGE_SPEC[key]
    axis = spec['axis']

    if axis == 'rows':
        merged = pd.concat(usable_frames, ignore_index=True)
        return _order_rows_by_model_run(merged, model_run)

    if axis == 'dedup_rows':
        merged = pd.concat(usable_frames, ignore_index=True)
        dedup_cols = [c for c in spec['key'] if c in merged.columns]
        if dedup_cols:
            merged = merged.drop_duplicates(subset=dedup_cols, keep='first').reset_index(drop=True)
        return merged

    if axis == 'columns':
        join_cols = [c for c in spec['key'] if all(c in f.columns for f in usable_frames)]
        if not join_cols:
            raise ValueError(
                f"merge_model_group_frames: cannot column-merge {key!r} - none of the expected "
                f"join columns {list(spec['key'])} are present in every group's frame (columns "
                f"seen: {[list(f.columns) for f in usable_frames]}). This indicates a Steps-1-8 "
                f"pool-construction mismatch across groups (I4/I6 violation) rather than a "
                f"harmless naming difference - do not paper over it by relaxing the join key."
            )
        merged = usable_frames[0]
        for frame in usable_frames[1:]:
            merged = merged.merge(frame, on=join_cols, how='outer', validate='one_to_one')
        return _order_prediction_columns(merged, join_cols, model_run)

    raise ValueError(f"merge_model_group_frames: unknown merge axis {axis!r} for key {key!r}.")


def model_group_fingerprint(members: Sequence[str]) -> str:
    """Short (8 hex char), deterministic, ORDER-INDEPENDENT fingerprint of
    a model group's member list - a group with the same members is the
    SAME group regardless of what order `build_model_groups()` happened
    to list them in. Mirrors `sample_fingerprint()`'s role at the task
    tier (I8/I9): used to name each group's scratch folder
    (`group_<g>_<fp8>`), so a resubmission under a DIFFERENT
    `N_MODEL_WORKERS`/`MODEL_GROUPING` (and therefore different group
    composition) creates new folder names rather than misreading a stale
    group's already-complete files as this run's own."""
    payload = ','.join(sorted(str(m) for m in members))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:8]


def _is_bio_prior_model(name: str) -> bool:
    """Duplicated, one-line-for-one-line, from genomic_prediction.py's own
    _is_bio_prior_model() - kept as an independent copy (rather than
    imported) so this module has no dependency on genomic_prediction.py
    itself, avoiding any risk of a circular import (genomic_prediction.py
    is the one importing FROM this module)."""
    return name == 'GAT_biological_prior_knowledge' or name.startswith('GAT_biological_prior_knowledge_')


def static_write_flags_for(model_run: list, w_opt) -> dict:
    """The 3 model-selection-driven (not data-driven) 'should this
    conditional result file be written at all' flags, computed with the
    EXACT SAME expressions GP()'s original 'Store the results' section
    used before checkpointing existed - kept here, as a single source of
    truth, so a partial (on-error) save and the final (on-success) save
    can never disagree about which files a given run is expected to
    produce. `model_run` is GP()'s own MODEL_RUN list; `w_opt` is GP()'s
    own W_OPT parameter (weight.csv is written whenever a weighted-
    ensemble method was configured at all, regardless of whether any
    weight rows exist yet - matching the original unconditional-on-W_OPT
    check, not a non-emptiness check)."""
    # Update ID ver4-5, R2 (blueprint §4.4, invariant I14): both flags now
    # read model_registry.py's single source of truth instead of a
    # hardcoded 'RF'/two-name-tuple comparison - 'interactions' used to be
    # RF-only (meaning a run selecting ONLY a second interaction-emitting
    # model, e.g. GAT_prior_knowledge with emit_interaction=True, would
    # never have written Interaction.csv at all). schema_key_of() collapses
    # a bio-prior instance suffix ('GAT_biological_prior_knowledge_2') down
    # to the registry's own base key before lookup, exactly as HPARAM_SPECS/
    # DIAGNOSTIC_FLAG_FIELDS already require - this also means the previous
    # separate `_is_bio_prior_model(...)` check for the attention flag is
    # now folded into the same registry lookup (still verified equivalent:
    # every bio-prior instance schema-keys to 'GAT_biological_prior_
    # knowledge', which the registry marks 'attention').
    return {
        'interactions': any(emits_interactions(schema_key_of(base_of(m))) for m in model_run),
        'attention_total': any(emits_attention(schema_key_of(base_of(m))) for m in model_run),
        'weight': w_opt is not None,
    }


def result_file_paths(result_name: str, idx: int, parallel: bool, *, compression: "Optional[str]" = None) -> dict:
    """Path for each of GP()'s 10 result files, for this RESULT_NAME and
    (if `parallel`) batch index - the single source of truth both the
    checkpoint save/load helpers below AND genomic_prediction.py's own
    final 'Store the results' section use, so the two can never drift
    apart (see module docstring).

    Update ID ver4-9, R7: `compression` - `None`/`'none'` (the DEFAULT,
    UNCHANGED from every ver4-8-and-earlier call) never appends a
    compression extension to any of the 10 paths, regardless of
    RESULT_FILE_COMPRESSION. This default is deliberately preserved so
    every call site that is NOT part of R7's own explicit touch-point
    list (this module's own save/load/append/clear helpers below,
    assemble.py, and genomic_prediction.py::GP()'s own final 'Store the
    results' section) - e.g. intra_task_parallel.py/
    intra_batch_parallel.py's own intermediate per-task/per-model-group
    merge files, which R7 does not touch - keeps producing byte-identical
    paths to before, with no risk of silently starting to read/write
    gzip where a caller's own raw-text logic still assumes plain CSV.

    Pass `compression='gzip'` (normally the run's own `RESULT_COMPRESSION`
    config value) to append `.gz` to the SIX keys RESULT_FILE_COMPRESSION
    marks `'gzip'`; the four small files are UNAFFECTED either way - never
    compressed, regardless of what is passed here. The compression
    extension is appended AFTER the real extension and AFTER the
    `'_<idx>'` batch suffix - e.g. `Prediction_result_test_0.csv.gz`,
    NEVER `Prediction_result_test.csv_0.gz` - so a directory listing still
    sorts and reads exactly the way a person expects.
    """
    base_dir = result_dir_path(result_name)
    suffix = f'_{idx}' if parallel else ''
    paths = {}
    for key, filename in RESULT_FILE_NAMES.items():
        stem, ext = os.path.splitext(filename)
        file_compression = RESULT_FILE_COMPRESSION[key] if compression == 'gzip' else None
        paths[key] = os.path.join(base_dir, f'{stem}{suffix}{ext}{_COMPRESSION_EXT[file_compression]}')
    return paths


def resolve_result_path(result_name: str, idx: int, parallel: bool, key: str) -> "Optional[str]":
    """Update ID ver4-9, R7 (RK-6): for `key` (one of RESULT_FILE_NAMES'
    own keys), returns whichever of the compressed (`.gz`) or
    uncompressed path actually EXISTS on disk right now - compressed
    checked first - or `None` if NEITHER does. The read-side counterpart
    to `result_file_paths(..., compression=...)`'s write-side choice: a
    caller that needs to READ a file that may have been written under
    EITHER RESULT_COMPRESSION setting (e.g. a run resumed after the
    config's own RESULT_COMPRESSION value was changed between the
    original attempt and the resume) uses this rather than assuming
    today's config value tells it which extension the file actually has
    on disk. Callers MUST branch on `None` - this never fabricates a
    path, and never guesses; it only ever reports what is actually
    present."""
    gzip_path = result_file_paths(result_name, idx, parallel, compression='gzip')[key]
    if os.path.isfile(gzip_path):
        return gzip_path
    plain_path = result_file_paths(result_name, idx, parallel, compression=None)[key]
    if os.path.isfile(plain_path):
        return plain_path
    return None


def _checkpoint_path(result_name: str, idx: int, parallel: bool) -> str:
    base_dir = result_dir_path(result_name)
    suffix = f'_{idx}' if parallel else ''
    return os.path.join(base_dir, f'.checkpoint{suffix}.json')


def sample_fingerprint(sample_df: "pd.DataFrame", extra: "Optional[dict]" = None) -> str:
    """Deterministic hash of the scenario list (population/phenotype/
    ratio/replicate combinations) GP() is about to process, used to
    detect a STALE checkpoint - one saved by a previous run with a
    different PHENOTYPE/POPULATION/RATIO/SAMPLE_NUM/SCENARIO config that
    happens to reuse the same RESULT_NAME. Resuming against a mismatched
    scenario list would silently skip the wrong tasks or misinterpret
    saved rows, so load_checkpoint() below refuses to use a checkpoint
    whose fingerprint doesn't match and starts fresh instead (with a
    clear warning printed).

    Parameters
    ----------
    extra : optional, ver4-6 R4.6 (blueprint §5/§7 RK-5). An arbitrary
        JSON-serialisable payload folded into this SAME hash alongside
        the scenario list. genomic_prediction.py passes HP_TUNE plus the
        run's own frozen HPARAMETERS_BASELINE (the tuning anchor - see
        GP()'s own R3.2a fix) here, so that editing WHICH models are
        tuned, their algorithm/budget, or their own untuned baseline
        values - none of which changes `sample_df` itself - now ALSO
        correctly invalidates a stale checkpoint, rather than silently
        resuming a changed config against tuned values computed under
        the OLD one.

        This intentionally invalidates every IN-FLIGHT checkpoint the
        first time a tree upgrades to include this fix (disclosed in the
        ver4-6 Change Summary, risk RK-5) - a one-time, expected, correct
        durability fix, not a regression: a resubmitted job simply
        restarts that batch's task loop from scratch, exactly as if its
        checkpoint had never existed. `None` (the default) reproduces the
        exact pre-ver4-6 fingerprint, unchanged - used only where a
        caller genuinely has no such payload to fold in.

        Falls back to `str(extra)` (rather than raising) if `extra`
        contains something `json.dumps` can't serialise - a fingerprint
        mismatch is the WORST that can happen from an imperfect
        serialisation (an unnecessary but harmless fresh start), so this
        never blocks a run over a durability nicety.
    """
    cols = [c for c in ('population', 'phenotype', 'ratio', 'sample') if c in sample_df.columns]
    payload = sample_df[cols].astype(str).to_csv(index=False)
    if extra is not None:
        try:
            extra_payload = json.dumps(extra, sort_keys=True, default=str)
        except Exception:
            extra_payload = str(extra)
        payload = payload + '\n---EXTRA---\n' + extra_payload
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def save_checkpoint(result_name: str, idx: int, parallel: bool, sample_fp: str,
                     last_completed_i: int, total_tasks: int) -> None:
    """Record that the first `last_completed_i + 1` scenarios IN THIS BATCH
    (a 0-based index into this batch's own task list, e.g. 2 means "the
    1st, 2nd, and 3rd scenarios of this batch are done" - NOT a row index
    into GP()'s full `sample` DataFrame, which for a Parallel batch starts
    partway through `sample`) have been fully processed and saved via
    save_partial_results(). -1 means nothing has completed yet (only ever
    written this way from an error on the very first scenario of a run)."""
    base_dir = result_dir_path(result_name)
    os.makedirs(base_dir, exist_ok=True)
    payload = {
        'last_completed_i': last_completed_i,
        'total_tasks': total_tasks,
        'sample_fingerprint': sample_fp,
    }
    path = _checkpoint_path(result_name, idx, parallel)
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(payload, f)
    os.replace(tmp_path, path)  # atomic on POSIX/Windows - never leaves a half-written checkpoint behind


def load_checkpoint(result_name: str, idx: int, parallel: bool, sample_fp: str,
                     total_tasks: int) -> "int | None":
    """Return the last_completed_i recorded by a previous, still-valid
    checkpoint for this (result_name, batch), or None if there's no
    checkpoint, or the one on disk doesn't match this run's scenario list
    (see sample_fingerprint()) - in which case it's ignored (with a clear
    warning printed) and GP() starts this batch fresh, exactly as if no
    checkpoint file existed at all.

    Bug fix: this used to also ignore (return None for) a checkpoint
    recording last_completed_i >= total_tasks - 1, on the reasoning that
    "every task is done, so there's nothing left to resume". That's only
    true if the run that wrote it went on to finish EVERYTHING, including
    the naive/weighted-ensemble finalisation step that runs after every
    task completes - and clear_checkpoint() is only ever called once ALL
    of that has succeeded (see genomic_prediction.py's GP()). So if this
    checkpoint file still exists on disk at all, clear_checkpoint() was
    NEVER reached, meaning the run that wrote it did NOT fully finish -
    regardless of what last_completed_i says. Discarding it in exactly
    the "every task done" case silently threw away the MOST valuable
    checkpoint state to resume from: a run whose (expensive, hours-long)
    per-task model tuning had entirely finished and was safely saved to
    disk, but which then crashed during the (comparatively fast)
    finalisation step - e.g. the naive-ensemble marker-effect combination
    - forcing a resume to blindly redo every task's tuning from scratch,
    discarding results that were already safely on disk the whole time.
    A checkpoint this function does return is always safe to resume from
    exactly where it says: genomic_prediction.py's finalisation step is
    itself now wrapped in the same snapshot/rollback-on-error pattern as
    the main per-task loop (see GP()), so the partial results this
    checkpoint points at can never include a partially-completed
    finalisation attempt to begin with."""
    path = _checkpoint_path(result_name, idx, parallel)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[checkpoint] WARNING: could not read checkpoint '{path}' ({exc!r}) - "
              f"ignoring it and starting this batch from the beginning.")
        return None

    if payload.get('sample_fingerprint') != sample_fp:
        print(f"[checkpoint] Found a checkpoint at '{path}', but it doesn't match this run's "
              f"population/phenotype/ratio/replicate list (the configuration changed since "
              f"that checkpoint was saved) - ignoring it and starting this batch from the "
              f"beginning, to avoid resuming into the wrong scenario.")
        return None

    last_completed_i = payload.get('last_completed_i')
    if not isinstance(last_completed_i, int):
        return None
    return min(last_completed_i, total_tasks - 1)


def clear_checkpoint(result_name: str, idx: int, parallel: bool) -> None:
    """Delete this (result_name, batch)'s checkpoint file, once every
    scenario in it has finished successfully - there is nothing left to
    resume, and leaving a stale checkpoint around could otherwise make an
    unrelated FUTURE run (that happens to reuse this RESULT_NAME) think
    it should skip scenarios it hasn't actually run yet."""
    path = _checkpoint_path(result_name, idx, parallel)
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError as exc:
        print(f"[checkpoint] WARNING: could not remove checkpoint '{path}': {exc!r}.")


def clear_result_files(result_name: str, idx: int, parallel: bool) -> None:
    """Delete every result file this (result_name, batch) could produce
    (see result_file_paths()) - called by GP() whenever it's about to
    start a batch FRESH (no valid checkpoint to resume from - see
    load_checkpoint()).

    Bug fix: append_partial_results() (used after every single task
    succeeds, precisely so a kill mid-batch never loses an already-
    finished task - see that function's own docstring) decides whether to
    write a CSV header purely from whether a file already exists at that
    path: `header=not os.path.isfile(path)`. That's exactly right when
    resuming a genuinely in-progress batch (task 1's own earlier-written
    file is what task 2's append should land in without repeating the
    header) - but if this is actually a FRESH start (no checkpoint, so
    nothing from THIS run has written anything yet) and a file happens to
    already exist at that path anyway - a previous, unrelated run that
    reused the same RESULT_NAME, or a previous run of this exact batch
    that finished successfully and had its checkpoint cleared - the very
    first task's own new rows would silently get appended AFTER whatever
    old content is already sitting there, permanently mixing stale data
    into what's supposed to be this run's own fresh output, with no error
    or warning of any kind.

    Also matters for the conditional files (Interaction.csv, Weight.csv,
    etc.), which are only ever written at all when THIS run's own model
    selection calls for them (see static_write_flags_for()): if an OLDER
    run used RF (and so has a leftover Interaction.csv) but this new run
    doesn't, that stale file would otherwise linger and look exactly like
    this run's own genuine output.

    Deletes every file unconditionally (whether or not this run will end
    up writing to it) - simpler and safer than trying to predict in
    advance exactly which files this run will touch, and no output file
    for this batch should exist at all until this run itself produces it.

    Update ID ver4-9, R7 (RK-7): deletes BOTH the compressed (`.gz`) AND
    the plain-CSV path for every key, unconditionally - not just whichever
    one today's RESULT_COMPRESSION setting would produce. Without this, a
    fresh start (RESULT_COMPRESSION='gzip' today) would leave an older,
    UNCOMPRESSED Prediction_result_test.csv (written by a previous run of
    this same batch under RESULT_COMPRESSION='none', or from before ver4-9
    existed at all) sitting right next to the new '.csv.gz' - and every
    read site in this codebase that doesn't know to check for both would
    silently read the STALE, wrong one. Same "delete unconditionally, not
    predictively" reasoning as the rest of this function, just extended to
    both possible extensions."""
    for compression in ('gzip', None):
        for path in result_file_paths(result_name, idx, parallel, compression=compression).values():
            try:
                if os.path.isfile(path):
                    os.remove(path)
            except OSError as exc:
                print(f"[checkpoint] WARNING: could not remove stale result file '{path}': {exc!r}.")


def append_partial_results(result_name: str, idx: int, parallel: bool, delta_frames: dict,
                            static_write_flags: dict, *, compression: "Optional[str]" = None) -> None:
    """Requirement (bugfix - "Jobs pick up where they left off" losing
    everything to an OOM/walltime kill): save_partial_results() (below)
    was, and still is, the only thing that ever wrote a task's results to
    disk - and it's only ever CALLED on a caught Python exception, or once
    the whole batch finishes. A SIGKILL (the OS's own out-of-memory
    killer) or a scheduler's walltime kill is NOT a catchable Python
    exception - genuinely, unconditionally uncatchable, by design, at the
    OS level - so neither of those trigger points is ever reached when
    that's what ends the job. Confirmed directly against a real report:
    every task that had actually finished before the kill was lost too,
    because nothing had saved them yet - not because save_partial_results()
    itself did anything wrong, but because it was simply never called in
    time. GP()'s own main loop now calls THIS function after every SINGLE
    task succeeds (see genomic_prediction.py) - not just on error - so a
    kill with literally zero warning still only ever loses the ONE task
    that was actually running at that moment, never any that already
    finished.

    Deliberately NOT the same approach as save_partial_results(), which
    re-writes each file's ENTIRE accumulated content from scratch every
    time it's called - calling THAT after every single task would cost
    O(total rows completed so far) on EACH task, making a whole batch's
    worth of per-task saves O(n^2) overall, a real and measurable
    slowdown exactly proportional to how many tasks a batch has. This
    instead APPENDS only `delta_frames` - the rows THIS ONE task just
    added (the caller computes this as the difference between its own
    pre- and post-task DataFrame snapshots) - so each task's own save
    cost is proportional only to that task's own row count, not the
    running total, keeping the TOTAL cost across a whole batch O(n) - the
    same order of growth the underlying computation already has, not a
    new, worse one.

    delta_frames : keyed exactly like RESULT_FILE_NAMES, but containing
        ONLY the NEW rows this task contributed (not the full running
        total save_partial_results() expects) - a DataFrame with 0 rows
        for any accumulator this task didn't touch.
    static_write_flags : from static_write_flags_for() - same meaning as
        in save_partial_results(); a conditional file is only ever
        appended to (or created) when this run's model selection makes
        it relevant at all AND this task's own delta is non-empty.
    compression : Update ID ver4-9, R7 - forwarded to result_file_paths()
        unchanged (see that function's own docstring for the default).
        Each individual `to_csv()` call below still derives its OWN
        `compression=` argument from the resolved PATH's actual extension
        (`.gz` or not), not from this parameter directly - Stage 8's own
        blocking check (ver4-9 Phase 2 session) confirmed that repeatedly
        appending to the SAME gzip path with `mode='a', compression=
        'gzip'` produces a valid multi-member gzip stream that
        `pd.read_csv()` transparently reads back as one continuous file,
        exactly like a plain-CSV append does - this is what makes the
        existing O(n) per-task append strategy above still valid under
        gzip, unchanged.
    """
    paths = result_file_paths(result_name, idx, parallel, compression=compression)
    base_dir = result_dir_path(result_name)
    os.makedirs(base_dir, exist_ok=True)

    def _append(key: str) -> None:
        df = delta_frames[key]
        if df.shape[0] == 0:
            return
        path = paths[key]
        _file_exists = os.path.isfile(path)
        df.to_csv(path, mode='a', header=not _file_exists, index=False,
                  compression='gzip' if path.endswith('.gz') else None)

    # ALWAYS_WRITTEN_KEYS: unlike save_partial_results() (which writes
    # these unconditionally, even when empty, purely to guarantee the
    # file exists), an APPEND of an empty delta is already a correct
    # no-op on its own - nothing gets added either way - so the same
    # _append() (which already skips empty deltas) is used for these
    # too, rather than a separate unconditional branch. The file simply
    # comes into existence the first time any task actually contributes
    # a row to it, exactly matching when save_partial_results() would
    # first have produced non-empty content for it anyway.
    for key in ALWAYS_WRITTEN_KEYS:
        _append(key)

    _append('effect')
    _append('hp_record')
    if static_write_flags.get('interactions', False):
        _append('interactions')
    if static_write_flags.get('attention_total', False):
        _append('attention_total')
    if static_write_flags.get('weight', False):
        _append('weight')


def save_partial_results(result_name: str, idx: int, parallel: bool, frames: dict,
                          static_write_flags: dict, *, compression: "Optional[str]" = None) -> None:
    """Write every one of GP()'s 10 accumulator DataFrames (`frames`,
    keyed exactly like RESULT_FILE_NAMES) to their designated result
    files (see result_file_paths()) - used BOTH mid-run, when a scenario
    fails and everything completed before it needs to be saved, AND at
    the true end of a fully successful run, so the two code paths can
    never produce different-looking output for the same underlying data
    (see genomic_prediction.py's GP(), which calls this from both
    places).

    static_write_flags : from static_write_flags_for() - whether
        'interactions'/'attention_total'/'weight' are ever relevant for
        this run's model selection at all (independent of whether their
        DataFrame happens to be empty at THIS particular save point).
        'effect'/'hp_record', by contrast, are written whenever they're
        currently non-empty, re-checked fresh on every call - exactly
        matching GP()'s original, pre-checkpointing behaviour for those
        two (a purely data-driven condition, not a model-selection one).
    compression : Update ID ver4-9, R7 - forwarded to result_file_paths()
        unchanged; see append_partial_results()'s own docstring for why
        each individual to_csv() call still derives its own compression
        argument from the resolved path's actual extension.
    """
    def _compression_for(path: str) -> "Optional[str]":
        return 'gzip' if path.endswith('.gz') else None

    paths = result_file_paths(result_name, idx, parallel, compression=compression)
    os.makedirs(result_dir_path(result_name), exist_ok=True)

    for key in ALWAYS_WRITTEN_KEYS:
        frames[key].to_csv(paths[key], index=False, compression=_compression_for(paths[key]))

    if frames['effect'].shape[0] != 0:
        frames['effect'].to_csv(paths['effect'], index=False, compression=_compression_for(paths['effect']))
    if static_write_flags.get('interactions', False):
        frames['interactions'].to_csv(paths['interactions'], index=False,
                                       compression=_compression_for(paths['interactions']))
    if static_write_flags.get('attention_total', False):
        frames['attention_total'].to_csv(paths['attention_total'], index=False,
                                          compression=_compression_for(paths['attention_total']))
    if static_write_flags.get('weight', False):
        frames['weight'].to_csv(paths['weight'], index=False, compression=_compression_for(paths['weight']))
    if frames['hp_record'].shape[0] != 0:
        frames['hp_record'].to_csv(paths['hp_record'], index=False)


def load_partial_results(result_name: str, idx: int, parallel: bool) -> dict:
    """The inverse of save_partial_results(): read back whatever result
    files already exist on disk for this (result_name, batch) into a
    dict of DataFrames keyed like RESULT_FILE_NAMES (an empty DataFrame
    for any file that doesn't exist) - used to re-populate GP()'s 10
    accumulators when resuming from a checkpoint, so already-completed
    scenarios' rows are carried forward into the eventual final output
    instead of being lost.

    Update ID ver4-9, R7 (RK-6): takes NO `compression` parameter, and
    deliberately does not consult the run's own current RESULT_COMPRESSION
    config value at all - each key's path is resolved independently via
    resolve_result_path() (compressed-then-uncompressed existence probe),
    since a batch being resumed may have been STARTED under a different
    RESULT_COMPRESSION setting than the one in effect for the resume
    (e.g. the person changed the config between attempts, or is resuming
    a pre-ver4-9 run under a ver4-9 codebase) - reading based on what is
    actually on disk, not on what today's config claims, is the only way
    this stays correct in that situation. `pd.read_csv()` infers gzip vs
    plain transparently from the resolved path's own extension either
    way, so no explicit `compression=` argument is needed here.

    A file that exists but is EMPTY (zero-byte plain CSV) or TRUNCATED
    (a `.gz` killed mid-append, before its trailing end-of-stream marker
    was written) is treated identically to "file absent" - an empty
    DataFrame, not a crash - exactly matching this function's own
    pre-ver4-9 handling of a zero-byte plain CSV
    (`pd.errors.EmptyDataError`), just extended to the two additional
    exception types a truncated/corrupt gzip stream can raise
    (`_INCOMPLETE_FILE_ERRORS`, see this module's own top-of-file note and
    Stage 8's blocking check, ver4-9 Phase 2 session). This is the correct
    outcome for a resume: the batch's own checkpoint file (a separate,
    NEVER-compressed JSON, unaffected by any of this) is the authority on
    how far the batch actually got, not the result files' own byte
    content - a truncated result file simply means "this key's own rows
    for the in-flight task at the moment of the kill are gone", which is
    exactly what should happen (the SAME task is about to be re-run from
    scratch on resume anyway, per this module's own "what counts as
    completed" discipline, module docstring)."""
    frames = {}
    for key in RESULT_FILE_NAMES:
        path = resolve_result_path(result_name, idx, parallel, key)
        if path is None:
            frames[key] = pd.DataFrame()
            continue
        try:
            frames[key] = pd.read_csv(path)
        except _INCOMPLETE_FILE_ERRORS:
            frames[key] = pd.DataFrame()
    return frames


# --------------------------------------------------------------------------
# Making incomplete Parallel batches easy to notice (not just resumable)
# --------------------------------------------------------------------------
#
# Before checkpointing existed, a batch that failed partway through left
# NO result files behind at all - so "Metric_<idx>.csv is missing" was a
# perfectly reliable signal that batch never finished (see assemble.py's
# original missing-batch detection). Checkpointing changes that: a batch
# that fails now DOES leave a Metric_<idx>.csv (and friends) behind - just
# a PARTIAL one, containing only the scenarios that finished before the
# error - so file-presence alone can no longer tell "finished" apart from
# "started, then interrupted". find_incomplete_batches() (and its
# Sequential-run counterpart, sequential_run_status()) closes that gap:
# a batch/run only ever has its checkpoint file removed once EVERY one of
# its scenarios has completed (see clear_checkpoint()), so the checkpoint
# file's mere continued EXISTENCE is itself the "still incomplete" signal
# - regardless of what its (possibly quite complete-looking) result files
# contain. assemble.py uses this to keep any such batch OUT of the final
# assembled results (rather than silently folding partial data in as if
# it were whole), and reports exactly which batch(es) and how far each
# one got, so re-submitting the right batch ID(s) to finish the job is a
# one-line lookup rather than a guessing game.

def find_incomplete_batches(result_name: str) -> list:
    """Every Parallel batch for `result_name` that has a checkpoint file
    still on disk (see module note above) - i.e. STARTED, but has not yet
    finished ALL of its scenarios. Batches with no checkpoint file at all
    (never run, OR already finished successfully) are not included.

    Returns a list of dicts, sorted by batch_id:
        {'batch_id': int, 'completed': int, 'total_tasks': int,
         'resume_from_task': int}
    ('completed'/'total_tasks'/'resume_from_task' are all 1-based counts,
    ready to show directly to a person - e.g. "3/5 done, resumes at task
    4" - rather than the 0-based indices used internally.)
    """
    base_dir = result_dir_path(result_name)
    if not os.path.isdir(base_dir):
        return []
    incomplete = []
    for path in sorted(glob.glob(os.path.join(base_dir, '.checkpoint_*.json'))):
        match = re.match(r'\.checkpoint_(\d+)\.json$', os.path.basename(path))
        if not match:
            continue
        batch_id = int(match.group(1))
        try:
            with open(path) as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            # Corrupt/unreadable checkpoint - still report the batch as
            # incomplete (we genuinely don't know its state), just
            # without progress detail, rather than silently omitting it.
            incomplete.append({'batch_id': batch_id, 'completed': None,
                                'total_tasks': None, 'resume_from_task': None})
            continue
        last_completed_i = payload.get('last_completed_i')
        total_tasks = payload.get('total_tasks')
        if not isinstance(last_completed_i, int) or not isinstance(total_tasks, int):
            incomplete.append({'batch_id': batch_id, 'completed': None,
                                'total_tasks': None, 'resume_from_task': None})
            continue
        # Bug fix: last_completed_i can legitimately equal total_tasks - 1
        # (every task finished) while this checkpoint file still exists -
        # meaning the run's finalisation step (naive/weighted-ensemble
        # combination, which runs after every task) hadn't succeeded yet
        # when it was written (see load_checkpoint()'s own comment for why
        # that no longer means "nothing to resume"). 'resume_from_task' is
        # None in exactly that case rather than the nonsensical
        # total_tasks + 1 (e.g. "resumes at task 6/5") the plain formula
        # below would otherwise produce - see describe_incomplete_batch().
        last_completed_i = min(last_completed_i, total_tasks - 1)
        _all_tasks_done = last_completed_i >= total_tasks - 1
        incomplete.append({
            'batch_id': batch_id,
            'completed': last_completed_i + 1,
            'total_tasks': total_tasks,
            'resume_from_task': None if _all_tasks_done else last_completed_i + 2,
        })
    return incomplete


def sequential_run_status(result_name: str) -> "dict | None":
    """The Sequential-run (PARALLEL=None) counterpart to
    find_incomplete_batches() - returns the same kind of progress dict
    ({'completed', 'total_tasks', 'resume_from_task'}, all 1-based) if
    that run's own '.checkpoint.json' still exists (meaning it hasn't
    finished all of its scenarios yet), or None if it has finished (or
    never started)."""
    path = os.path.join(result_dir_path(result_name), '.checkpoint.json')
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {'completed': None, 'total_tasks': None, 'resume_from_task': None}
    last_completed_i = payload.get('last_completed_i')
    total_tasks = payload.get('total_tasks')
    if not isinstance(last_completed_i, int) or not isinstance(total_tasks, int):
        return {'completed': None, 'total_tasks': None, 'resume_from_task': None}
    last_completed_i = min(last_completed_i, total_tasks - 1)
    _all_tasks_done = last_completed_i >= total_tasks - 1
    return {'completed': last_completed_i + 1, 'total_tasks': total_tasks,
            'resume_from_task': None if _all_tasks_done else last_completed_i + 2}


def format_batch_id_list(batch_ids) -> str:
    """Comma-joined batch IDs with NO space after the comma (e.g.
    '3,7,12', not '3, 7, 12') - deliberately, since these lists are meant
    to be copy-pasted directly into a scheduler directive (e.g. PBS's
    `-J 3,7,12` for resubmitting a specific subset of failed/missing
    batches) - a space in that position makes PBS treat what follows as a
    separate, malformed directive and fail with a 'directive not found'
    error. Every batch-ID-list display in this codebase (main_app.py,
    run_step2_assemble.py, assemble.py) uses this, rather than Python's
    default list repr/str (which inserts ', ') or an ad-hoc ', '.join(),
    so the formatting can never drift back to being unsafe to paste."""
    return ','.join(str(b) for b in batch_ids)


def describe_incomplete_batch(batch: dict) -> str:
    """One human-readable line for a dict from find_incomplete_batches(),
    e.g. 'batch 3: 2/5 task(s) completed - resumes automatically at task
    3/5 if re-submitted with the same config and batch_id' - factored out
    since assemble.py, run_step2_assemble.py, and main_app.py all print/
    display this same message."""
    if batch['completed'] is None:
        return (f"batch {batch['batch_id']}: checkpoint found but unreadable/corrupt - "
                f"re-submit this batch from scratch (delete "
                f"'.checkpoint_{batch['batch_id']}.json' first if it persists) to be safe.")
    if batch['resume_from_task'] is None:
        # All tasks finished, but the finalisation step (naive/weighted-
        # ensemble combination) hadn't succeeded yet - see
        # load_checkpoint()'s own comment. Nothing about the (expensive)
        # per-task model tuning needs to be re-done here.
        return (f"batch {batch['batch_id']}: all {batch['total_tasks']} task(s) completed - "
                f"resumes automatically by re-running just the final ensemble/aggregation "
                f"step if re-submitted with the same config and batch_id.")
    return (f"batch {batch['batch_id']}: {batch['completed']}/{batch['total_tasks']} task(s) "
            f"completed - resumes automatically at task {batch['resume_from_task']}/"
            f"{batch['total_tasks']} if re-submitted with the same config and batch_id.")


# --------------------------------------------------------------------------
# Fast, read-only batch status check - the discovery half of assemble.py's
# own logic, factored out here so it can be called on its own (e.g. from a
# GUI "check status" button) WITHOUT paying the cost of actually reading
# and concatenating every batch's Metric_<idx>.csv/Prediction_result_*.csv
# etc. - for a run with many large batches, that data-reading step is the
# slow part; the status itself (which batch IDs are complete/incomplete/
# missing) only ever needs file-existence checks and small checkpoint-file
# reads, both of which are effectively instant regardless of batch size.
# assemble.py's own assemble() calls this same function for its own
# batch-discovery step, so the two can never disagree about a given run's
# status.
# --------------------------------------------------------------------------

def check_batch_status(result_name: str, expected_batches: "int | None" = None) -> dict:
    """Fast, read-only status check for a Parallel run's batches.

    Parameters
    ----------
    result_name : the RESULT_NAME to check.
    expected_batches : optional total number of batches submitted (e.g.
        Array end index + 1 from Step 1's HPC export) - see assemble()'s
        own docstring for exactly what this changes (a batch missing at
        the very end, beyond the highest batch ID actually found on disk,
        is only ever reported when this is given).

    Returns
    -------
    dict with:
        'complete'   : sorted list of batch IDs that finished successfully
                       (a Metric_<idx>.csv exists AND no checkpoint file
                       remains for that batch).
        'incomplete' : list of dicts from find_incomplete_batches() -
                       batches that started but have not yet finished
                       every scenario (a checkpoint file still exists).
        'missing'    : sorted list of batch IDs with no output at all -
                       never started, or crashed before completing even
                       one scenario.
    """
    base_dir = result_dir_path(result_name)
    batch_ids_with_metric = sorted(
        int(m.group(1)) for f in glob.glob(os.path.join(base_dir, 'Metric_*.csv'))
        if (m := re.match(r'Metric_(\d+)\.csv$', os.path.basename(f)))
    )

    incomplete = find_incomplete_batches(result_name)
    incomplete_ids = {b['batch_id'] for b in incomplete}
    complete_ids = [b for b in batch_ids_with_metric if b not in incomplete_ids]

    if complete_ids or incomplete_ids:
        lo = min(complete_ids + list(incomplete_ids))
        hi = max(complete_ids + list(incomplete_ids))
        expected_ids = set(range(lo, hi + 1))
    else:
        expected_ids = set()
    if expected_batches:
        expected_ids |= set(range(0, expected_batches))
    missing_ids = sorted(expected_ids - set(complete_ids) - incomplete_ids)

    return {'complete': complete_ids, 'incomplete': incomplete, 'missing': missing_ids}


