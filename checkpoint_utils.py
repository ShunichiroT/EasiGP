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
import hashlib
import json
import os
import re

import pandas as pd

from models.hyperparameter_tuning import base_of

# Keys for GP()'s 9 accumulator DataFrames, and the (unsuffixed) output
# filename each one is written to - PARALLEL runs get '_<idx>' inserted
# before the extension (see result_file_paths()), exactly matching GP()'s
# original, pre-checkpointing file-naming convention.
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
    return {
        'interactions': any(base_of(m) == 'RF' for m in model_run),
        'attention_total': any(
            base_of(m) in ('GAT_fully_connected', 'GAT_prior_knowledge') or _is_bio_prior_model(base_of(m))
            for m in model_run
        ),
        'weight': w_opt is not None,
    }


def result_file_paths(result_name: str, idx: int, parallel: bool) -> dict:
    """Path for each of GP()'s 9 result files, for this RESULT_NAME and
    (if `parallel`) batch index - the single source of truth both the
    checkpoint save/load helpers below AND genomic_prediction.py's own
    final 'Store the results' section use, so the two can never drift
    apart (see module docstring)."""
    base_dir = os.path.join('.', 'Result', result_name)
    suffix = f'_{idx}' if parallel else ''
    paths = {}
    for key, filename in RESULT_FILE_NAMES.items():
        stem, ext = os.path.splitext(filename)
        paths[key] = os.path.join(base_dir, f'{stem}{suffix}{ext}')
    return paths


def _checkpoint_path(result_name: str, idx: int, parallel: bool) -> str:
    base_dir = os.path.join('.', 'Result', result_name)
    suffix = f'_{idx}' if parallel else ''
    return os.path.join(base_dir, f'.checkpoint{suffix}.json')


def sample_fingerprint(sample_df: "pd.DataFrame") -> str:
    """Deterministic hash of the scenario list (population/phenotype/
    ratio/replicate combinations) GP() is about to process, used to
    detect a STALE checkpoint - one saved by a previous run with a
    different PHENOTYPE/POPULATION/RATIO/SAMPLE_NUM/SCENARIO config that
    happens to reuse the same RESULT_NAME. Resuming against a mismatched
    scenario list would silently skip the wrong tasks or misinterpret
    saved rows, so load_checkpoint() below refuses to use a checkpoint
    whose fingerprint doesn't match and starts fresh instead (with a
    clear warning printed)."""
    cols = [c for c in ('population', 'phenotype', 'ratio', 'sample') if c in sample_df.columns]
    payload = sample_df[cols].astype(str).to_csv(index=False)
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
    base_dir = os.path.join('.', 'Result', result_name)
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
    checkpoint file existed at all. A checkpoint recording
    last_completed_i >= total_tasks - 1 (every scenario in this batch
    already finished - e.g. the previous run actually succeeded and this
    is an unrelated re-submission reusing the same RESULT_NAME) is
    likewise ignored, since there is nothing left to resume."""
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
    if last_completed_i >= total_tasks - 1:
        return None
    return last_completed_i


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


def append_partial_results(result_name: str, idx: int, parallel: bool, delta_frames: dict,
                            static_write_flags: dict) -> None:
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
    """
    paths = result_file_paths(result_name, idx, parallel)
    base_dir = os.path.join('.', 'Result', result_name)
    os.makedirs(base_dir, exist_ok=True)

    def _append(key: str) -> None:
        df = delta_frames[key]
        if df.shape[0] == 0:
            return
        path = paths[key]
        _file_exists = os.path.isfile(path)
        df.to_csv(path, mode='a', header=not _file_exists, index=False)

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
                          static_write_flags: dict) -> None:
    """Write every one of GP()'s 9 accumulator DataFrames (`frames`,
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
    """
    paths = result_file_paths(result_name, idx, parallel)
    os.makedirs(os.path.join('.', 'Result', result_name), exist_ok=True)

    for key in ALWAYS_WRITTEN_KEYS:
        frames[key].to_csv(paths[key], index=False)

    if frames['effect'].shape[0] != 0:
        frames['effect'].to_csv(paths['effect'], index=False)
    if static_write_flags.get('interactions', False):
        frames['interactions'].to_csv(paths['interactions'], index=False)
    if static_write_flags.get('attention_total', False):
        frames['attention_total'].to_csv(paths['attention_total'], index=False)
    if static_write_flags.get('weight', False):
        frames['weight'].to_csv(paths['weight'], index=False)
    if frames['hp_record'].shape[0] != 0:
        frames['hp_record'].to_csv(paths['hp_record'], index=False)


def load_partial_results(result_name: str, idx: int, parallel: bool) -> dict:
    """The inverse of save_partial_results(): read back whatever result
    files already exist on disk for this (result_name, batch) into a
    dict of DataFrames keyed like RESULT_FILE_NAMES (an empty DataFrame
    for any file that doesn't exist) - used to re-populate GP()'s 9
    accumulators when resuming from a checkpoint, so already-completed
    scenarios' rows are carried forward into the eventual final output
    instead of being lost."""
    paths = result_file_paths(result_name, idx, parallel)
    frames = {}
    for key, path in paths.items():
        if os.path.isfile(path):
            try:
                frames[key] = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                frames[key] = pd.DataFrame()
        else:
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
    base_dir = os.path.join('.', 'Result', result_name)
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
        incomplete.append({
            'batch_id': batch_id,
            'completed': last_completed_i + 1,
            'total_tasks': total_tasks,
            'resume_from_task': last_completed_i + 2,
        })
    return incomplete


def sequential_run_status(result_name: str) -> "dict | None":
    """The Sequential-run (PARALLEL=None) counterpart to
    find_incomplete_batches() - returns the same kind of progress dict
    ({'completed', 'total_tasks', 'resume_from_task'}, all 1-based) if
    that run's own '.checkpoint.json' still exists (meaning it hasn't
    finished all of its scenarios yet), or None if it has finished (or
    never started)."""
    path = os.path.join('.', 'Result', result_name, '.checkpoint.json')
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
    return {'completed': last_completed_i + 1, 'total_tasks': total_tasks,
            'resume_from_task': last_completed_i + 2}


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
    base_dir = os.path.join('.', 'Result', result_name)
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


