"""
intra_batch_parallel.py
=========================

Phase 2, Requirement 7 - parallelise across the TASKS inside one Parallel-
mode array-job batch, using multiple CPU processes (and, for GPU-capable
models, a bounded number of concurrent GPU slots) available to that single
PBS/Slurm job - rather than the existing, strictly serial
``_process_one_task(i)`` driving loop inside ``genomic_prediction.py::GP()``.

Design (see ``EasiGP_Phase2_Design_Blueprint.md`` §7 for the full
rationale)
--------------------------------------------------------------------------
``GP()``'s per-task logic (``_process_one_task``) is a deeply-nested
closure over dozens of outer-scope variables (HPARAMETERS, MODEL_RUN, the
R model functions bound into the R global environment, PLINK working-
directory factories, the bio-prior merge cache, ...). Extracting a
picklable ``TaskContext`` out of that closure - the textbook way to fan a
function like this out across worker PROCESSES - is a large, invasive
refactor of ``GP()`` itself (flagged as the #1 architectural tech-debt item
independently of this requirement - see architecture doc §18).

This module takes a different, much lower-risk route that requires ZERO
changes to ``GP()``'s internals, by reusing a mechanism the codebase
*already* relies on for inter-batch parallelism: ``GP()`` is deterministic
and re-buildable per array-job task purely from its own arguments (the
``sample`` scenario table is rebuilt identically by every task - see
architecture doc §6, "every array-job task independently reconstructs the
identical table and simply takes a different slice"). A ``PARALLEL={
'batch_id': i, 'batch_size': 1}`` call therefore already means "process
EXACTLY task index ``i`` of the full scenario table, and nothing else" -
this is precisely a single-task unit of work, addressable without touching
one line of ``GP()``.

This module fans OUT across the ``interval`` (outer ``batch_size``) task
indices belonging to ONE outer Parallel batch, running each index's own
``GP(..., PARALLEL={'batch_id': i, 'batch_size': 1})`` call in its own
worker PROCESS (never a thread - see the constraints below), each writing
into an ISOLATED, per-task result folder (a nested ``RESULT_NAME``) so
these fine-grained sub-runs can NEVER collide with - or be mistaken for -
one of the outer, real Parallel batches that
``checkpoint_utils.check_batch_status()``/``assemble.py`` already know how
to discover (those only ever glob directly inside
``Result/<RESULT_NAME>/``, never a subfolder). Once every task index has
succeeded, this module merges the per-task isolated result files into the
outer batch's OWN standard result files
(``checkpoint_utils.result_file_paths(RESULT_NAME, batch_id, parallel=True)``)
- so from ``run_step2_assemble.py``'s point of view, an intra-batch-
parallel batch's output is byte-for-byte indistinguishable in SHAPE from
one produced by the original, fully serial code path.

Constraints honoured (from the design blueprint)
--------------------------------------------------------------------------
- **Process-based parallelism only, never threads**: each worker process
  calls ``configure_r_environment()``/``init_rpy2_conversion()`` and
  sources the ``.R`` model files itself (inside its own fresh call to
  ``GP()``) - R/rpy2 is not thread-safe and not safely shareable across
  threads within one process.
- **Each worker gets its own PLINK temp dir**: already true by default
  (``work_dir=None`` in every ``Preprocess/plink_io.py`` call site) since
  each worker is a genuinely separate OS process.
- **GPU contention**: a ``multiprocessing.Semaphore``-like GPU-slot limiter
  (``pipeline_utils.install_gpu_semaphore``/``gpu_slot()``) is installed in
  every worker process, bounding how many GPU-dispatched model calls
  (MLP/GAT) run concurrently across all workers sharing one physical GPU.
- **Concurrent-safe persistence**: no worker ever writes to a file another
  worker (or the parent) could also be writing to at the same time - each
  worker's isolated ``RESULT_NAME`` is unique to its own task index, and
  the merge step (reading, never writing, those files) only ever runs
  AFTER every worker has finished.
- **Failure isolation**: a worker process crash only loses the ONE task
  index it was processing - every other task index's already-succeeded,
  isolated result files are untouched, and are skipped (not re-run) on a
  later retry of the same outer batch, via the completion check in
  ``_task_already_complete()``.
- **Minimum-tasks-per-batch threshold**: a batch with too few tasks to be
  worth the worker-pool startup/R-sourcing overhead falls straight back to
  a single, ordinary serial ``GP()`` call, unchanged.
"""

from __future__ import annotations

import errno
import glob
import multiprocessing
import os
import random
import re
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import pandas as pd

import checkpoint_utils as _ckpt
from pipeline_utils import configure_r_environment, init_rpy2_conversion, install_gpu_semaphore, result_dir_path

# ---------------------------------------------------------------------------
# Update ID 2, Defect D2 fix (see PATCH_NOTES_D2.md).
#
# Every ProcessPoolExecutor/Manager this module creates explicitly SPAWNS
# its worker processes, rather than relying on the platform default (fork,
# on Linux) - the well-documented, rpy2-recommended fix for "forking a
# process pool out of a parent that has R (or a threaded BLAS/OpenMP
# library R is linked against) already initialised is unsafe": the child
# can inherit an inconsistent copy of a native library's internal state
# (locks held by threads that don't exist post-fork, partially-initialised
# allocator arenas, ...), which can then surface unpredictably - sometimes
# immediately, sometimes only hours into a long BGLR/MCMC run - as a
# SIGBUS, a nonsensical ImportError, or worse. This is exactly what a
# production N_CPU_WORKERS=4 batch hit (see the linked patch notes).
#
# `_worker_init()`/`_run_isolated_task()` are both already module-level,
# picklable functions, and every `gp_kwargs` dict passed to a worker is
# already required to be fully picklable (`_sub_kwargs_for()`'s own
# `progress_callback` removal, pre-existing, is for exactly this reason) -
# so spawn-compatibility was already an implicit design constraint here,
# just not the actual start method in use. A spawned worker starts from a
# genuinely fresh interpreter - inheriting this process's environment
# variables (including R_HOME/BLAS thread counts already exported by
# configure_r_environment(), same as before) but NOT its memory or thread
# state - which is what makes each worker's own `_worker_init()` call safe
# to initialise R in, regardless of whatever this (parent) process has or
# hasn't imported/initialised itself.
# ---------------------------------------------------------------------------
_MP_CONTEXT = multiprocessing.get_context('spawn')

# ---------------------------------------------------------------------------
# Defect D4 fix (production HPC failure - see the linked batch-90 log:
# every one of a 10-task batch failed near-simultaneously with
# BrokenPipeError(108, 'Cannot send after transport endpoint shutdown')
# while a freshly-spawned worker was reading genomic_prediction.py/
# Preprocess/plink_io.py/the models package off disk).
#
# errno 108 (ESHUTDOWN) is a SOCKET error, not an ordinary local-disk
# error - on Linux it is only ever raised for a transport that is
# actually socket-backed. `/scratch/...` on an HPC cluster is routinely a
# network filesystem (Lustre/BeeGFS/GPFS/NFS) whose reads are internally
# implemented over exactly such a transport, so a brief server-side or
# network hiccup surfaces to Python as a plain BrokenPipeError raised
# from deep inside `importlib` while the interpreter is simply trying to
# read a .py file's bytes - nothing to do with the file's own content or
# this module's own logic.
#
# The observed timing (every one of the 4 initially-spawned workers'
# imports failing inside the same ~1s window, several minutes into the
# batch) is consistent with a thundering-herd cause: `_worker_init()`
# below has every worker source the R/BGLR environment via
# `configure_r_environment()`/`init_rpy2_conversion()` - itself
# filesystem-heavy (R package files) - essentially in lockstep, since
# every worker process starts at almost the same instant; by the time
# they all finish and immediately move on to importing
# `genomic_prediction` (the very next thing each worker does), they are
# once again hitting the same network mount at almost the same instant.
# Two independent, complementary mitigations follow from that:
#   1. absorb a brief, genuinely transient mount hiccup via a bounded
#      retry-with-backoff around exactly the operations that hit the
#      mount at worker-startup time (`_worker_init()`'s R/rpy2 setup, and
#      `_run_isolated_task()`'s own deferred `genomic_prediction` import)
#      - never around `GP()` itself, which may already have done
#      substantial, non-idempotent-feeling work and has its own
#      checkpointing story;
#   2. reduce how synchronised that startup burst is in the first place
#      via a small random jitter at the very start of `_worker_init()`.
# Only a narrow allow-list of errno values genuinely associated with
# transient network-transport failures is treated as retryable - a real,
# reproducible fault (bad config, missing file, genuine code bug) must
# still fail immediately and visibly, not be silently retried into a
# longer, more confusing failure.
# ---------------------------------------------------------------------------
_TRANSIENT_IMPORT_ERRNOS = frozenset({
    errno.EPIPE,       # 32  - broken pipe
    errno.ESHUTDOWN,   # 108 - "Cannot send after transport endpoint shutdown"
    errno.ECONNRESET,  # 104 - connection reset by peer
    errno.ENOTCONN,    # 107 - transport endpoint is not connected
    errno.ETIMEDOUT,   # 110 - connection timed out
    errno.EAGAIN,      # 11  - resource temporarily unavailable (NFS retry-able)
})


def _is_transient_mount_error(exc: BaseException) -> bool:
    """True only for OS-level errors on the narrow allow-list above -
    i.e. errno values that are never a legitimate 'this data/config/code
    is wrong' signal, only ever a flaky underlying transport. Deliberately
    narrow: this is a targeted workaround for the specific, observed
    class of failure (see the note above `_TRANSIENT_IMPORT_ERRNOS`), not
    a general-purpose catch-and-hope-for-the-best."""
    return isinstance(exc, OSError) and getattr(exc, 'errno', None) in _TRANSIENT_IMPORT_ERRNOS


# ---------------------------------------------------------------------------
# Additional Requirements 9 (production defect - see batch-6 log,
# 2026-08-25: a 29-task, N_CPU_WORKERS=20 batch with
# ['rrBLUP','BayesB','RKHS','RF','SVR','KNN','ensemble'] selected had ALL
# 29 tasks die within the same second, ~34s after startup, every one with
# `BrokenProcessPool('A process in the process pool was terminated
# abruptly...')` - then EVERY SINGLE retry (already fully serial - one
# task at a time, each in its own fresh worker - see
# `run_batch_with_intra_batch_parallelism()`'s own retry loop) failed
# again too, each roughly 2.3 minutes into that task's own attempt).
#
# `BrokenProcessPool` (unlike Defect D4's `BrokenPipeError`/errno-108
# mount hiccup, a DIFFERENT failure class entirely - see
# `_is_transient_mount_error()` above) is `concurrent.futures`' own
# signal that a WORKER PROCESS ITSELF was killed out from under a running
# task - on an HPC node this is overwhelmingly the OS/cgroup OOM killer
# terminating a process whose memory usage crossed the job's requested
# `mem` limit. Both symptoms here point at the same root cause: this
# batch's model roster fits FOUR R/BGLR MCMC models (`rrBLUP`, `BayesB`,
# `RKHS` - the highest `peak_mem_gb_per_worker` figures in
# `resource_profiles.MODEL_COST_PROFILE`) plus RF/SVR/KNN, ALL sequentially
# within EVERY SINGLE task (architecture doc §7 Step 9 - one task fits
# every selected model, one after another, in the SAME persistent rpy2/R
# session, which does not reliably release memory back to the OS between
# successive `source()`-based model calls) - so even ONE worker's own
# single-task peak footprint can be substantial, and N_CPU_WORKERS=20
# concurrent, fully independent copies of that (memory is NEVER shared
# across separate OS processes) multiplies it further still. Retrying at
# the SAME N_CPU_WORKERS/requested-memory configuration cannot fix this -
# which is exactly what the log shows (every retry failing too) - it only
# burns walltime (a full `max_task_retries` sweep over every failed task,
# serially, at ~2+ minutes each, can run to hours before finally raising).
#
# `resource_profiles.estimate_worker_memory()`/`estimate_resources()`'s
# new `coordinated_mem_gb` (Additional Requirements 9, R2/R3) is the
# GUI-side fix - a memory recommendation that already accounts for
# N_CPU_WORKERS_TASK concurrent workers, each running this run's own full
# model roster - so a person who re-derives their resource request from
# it should not hit this again. The two helpers below are this module's
# OWN, runtime-side half of the same fix: detect the "most/all of this
# batch's tasks died together, worker-process-killed rather than a normal
# exception" signature EARLY (after the first pass, and again after the
# first retry attempt) and (a) say so plainly, pointing at the actual
# remedy, and (b) stop burning further retry attempts against a
# configuration that has already twice demonstrated it cannot succeed -
# rather than silently grinding through the rest of `max_task_retries`.
# ---------------------------------------------------------------------------
_WORKER_DEATH_SIGNATURES = ('brokenprocesspool', 'terminated abruptly')

# What fraction of a pass's attempted tasks must die with a worker-death
# signature (see above) before that pass is treated as SYSTEMIC (OOM/node-
# level) rather than one or two isolated, plausibly-transient failures -
# deliberately majority-based (not "any failure at all"), since a single
# unlucky task failing this way among many successes is still worth a
# normal retry rather than an early, alarming abort.
_SYSTEMIC_FAILURE_FRACTION = 0.5


def _is_worker_death_error(error_text: Optional[str]) -> bool:
    """Whether `error_text` (a failed task's already-stringified `error` -
    either `_run_isolated_task()`'s own message, or this module's own
    `f"worker process error: {exc!r}"` / `f"worker process error on
    retry: {exc!r}"` wrapper around a future/subprocess failure) looks
    like the WORKER PROCESS ITSELF was killed out from under a running
    task (`BrokenProcessPool`, or its own message text), rather than an
    ordinary Python exception raised from inside `GP()` (a data problem,
    a config error, ...). Text-matched, since by the time an error
    reaches this module it has already been converted to a string."""
    if not error_text:
        return False
    text = error_text.lower()
    return any(sig in text for sig in _WORKER_DEATH_SIGNATURES)


def _systemic_failure_note(
    failed: Dict[int, str], attempted_count: int, batch_id: int, n_cpu_workers: int, *, context: str,
) -> Optional[str]:
    """Returns a clear, actionable diagnostic string when `failed` (this
    pass's own failures - a subset, by task index, of what was just
    attempted) looks SYSTEMIC per `_SYSTEMIC_FAILURE_FRACTION` above -
    i.e. this pass, as a whole, looks like an OOM/node-level event rather
    than one or two unlucky, plausibly-transient task failures - or
    `None` otherwise (nothing worth saying beyond the existing per-task
    FAILED log lines). `context` names which pass this is ("initial
    attempt" / "retry attempt N/M") purely for the printed message."""
    if attempted_count <= 0 or not failed:
        return None
    worker_death_count = sum(1 for err in failed.values() if _is_worker_death_error(err))
    if worker_death_count == 0 or worker_death_count < attempted_count * _SYSTEMIC_FAILURE_FRACTION:
        return None
    return (
        f"[intra_batch_parallel] Batch {batch_id}: \u26a0\ufe0f {worker_death_count} of "
        f"{attempted_count} task(s) in this {context} died with a worker-process-killed error "
        f"(BrokenProcessPool / 'terminated abruptly') rather than a normal exception - this is the "
        f"signature of the OS/cgroup OOM killer terminating worker processes, almost always because "
        f"this batch's requested memory was sized for a SINGLE process while "
        f"N_CPU_WORKERS={n_cpu_workers} runs that many independent, full-memory-footprint copies "
        f"concurrently (each worker fits EVERY selected model for its own task, one after another, "
        f"in its own process - memory is never shared across them, and R/BGLR sessions in particular "
        f"tend not to release memory back to the OS between successive models within the same task). "
        f"Retrying at the SAME configuration is unlikely to help. Before resubmitting: reduce "
        f"N_CPU_WORKERS_TASK, or increase this batch's requested memory to match - the GUI's "
        f"'Suggested compute resources' panel now recommends a memory figure "
        f"(resource_profiles.estimate_resources()'s 'coordinated_mem_gb') that already accounts for "
        f"N_CPU_WORKERS_TASK running this many concurrent workers."
    )


def _retry_transient_mount_error(fn, *, attempts: int = 4, base_delay: float = 1.0,
                                  max_delay: float = 10.0, what: str = ""):
    """Call ``fn()`` and return its result. If it raises an OSError whose
    errno is on the transient-network-mount allow-list (see
    ``_is_transient_mount_error``), retry with exponential backoff plus
    jitter, up to ``attempts`` times total, before finally letting the
    last such failure propagate. Any other exception (including an
    OSError with a different errno) propagates immediately, unretried -
    only the specific failure mode this was written for is ever masked
    by a retry.

    ``what`` is a short, human-readable label used only in the
    progress-print between attempts (e.g. ``'configure_r_environment'``),
    so a person watching the log can see this is a known, handled retry
    rather than a silent hang.
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - re-raised below unless transient
            if not _is_transient_mount_error(exc) or attempt == attempts:
                raise
            last_exc = exc
            delay = min(max_delay, base_delay * (2 ** (attempt - 1))) * (0.5 + random.random())
            print(f"[intra_batch_parallel] transient mount error on {what or 'operation'} "
                  f"(attempt {attempt}/{attempts}): {exc!r} - retrying in {delay:.1f}s.")
            time.sleep(delay)
    raise last_exc  # pragma: no cover - unreachable (loop above always returns or raises)


def _startup_jitter(max_seconds: float = 2.5) -> None:
    """A small, random sleep called once at the very start of
    `_worker_init()` (see Defect D4 fix note above `_MP_CONTEXT`) -
    purely to desynchronise otherwise near-simultaneous, filesystem-heavy
    worker startup (R/rpy2 environment sourcing, then the
    `genomic_prediction` import) across however many worker processes
    were just spawned together, so they don't all hit a network scratch
    mount at literally the same instant. Bounded and small: this trades a
    few seconds of wall time for materially lower odds of a repeat of the
    Defect D4 failure, never enough to meaningfully slow a batch down."""
    time.sleep(random.uniform(0, max_seconds))


# Update ID 2, Appendix B finding F1 / V8: the blueprint's own working
# assumption was that `unique_path()` lives in `pipeline_utils.py`. It does
# NOT - it is defined in `Preprocess/LD_decay_plot.py` and is imported from
# there by `genomic_prediction.py` itself (see that module's own import
# block). Imported the same way here, rather than re-implementing a second
# collision-safe-path helper, so relocate_side_artefacts() below reuses the
# EXACT SAME collision behaviour LD-decay output already relies on. This
# module-level import is safe (no R/rpy2 dependency - LD_decay_plot.py is a
# pure Python/matplotlib/pandas module), unlike genomic_prediction.py's own
# deferred, inside-the-worker import of GP() itself.
from Preprocess.LD_decay_plot import unique_path

# Result files are merged in this fixed key order purely for deterministic,
# reproducible output - checkpoint_utils.RESULT_FILE_NAMES itself is an
# ordinary dict (insertion-ordered in modern Python, but not part of its
# documented contract), so this module pins its own explicit order rather
# than relying on that incidentally.
_MERGE_KEY_ORDER = (
    'record', 'result_train', 'result_valid', 'result_test', 'effect',
    'interactions', 'attention_total', 'weight', 'hp_record', 'stats',
)


def _isolated_result_name(result_name: str, batch_id: int, task_index: int) -> str:
    """A RESULT_NAME nested under the real one, unique to a single task
    index within a single outer batch - guarantees zero collision with
    the real, outer-batch-level result files
    (``Result/<RESULT_NAME>/Metric_<batch_id>.csv``) that
    ``checkpoint_utils.check_batch_status()`` globs for directly inside
    ``Result/<RESULT_NAME>/`` (never recursively), and zero collision
    between two different task indices' own isolated output."""
    return os.path.join(result_name, '.intra_batch_parallel', f'batch_{batch_id}', f'task_{task_index}')


def _task_already_complete(result_name: str, batch_id: int, task_index: int) -> bool:
    """True if this task index's isolated sub-run already finished
    successfully in a PREVIOUS attempt at this same outer batch (its own
    Metric_<task_index>.csv exists and its own checkpoint has already been
    cleared - exactly GP()'s own definition of "this (isolated) run
    finished") - in which case it's skipped entirely rather than redone,
    so retrying an outer batch that partially failed never redoes
    already-successful task-level work.

    Note: the isolated sub-run is always called with
    ``PARALLEL={'batch_id': task_index, 'batch_size': 1}`` (never
    ``PARALLEL=None``), so GP() treats it as a Parallel run and writes
    `_<task_index>`-suffixed files (`checkpoint_utils.result_file_paths(...,
    idx=task_index, parallel=True)`), not plain, unsuffixed ones."""
    isolated_name = _isolated_result_name(result_name, batch_id, task_index)
    paths = _ckpt.result_file_paths(isolated_name, task_index, True)
    metric_path = paths['record']
    checkpoint_path = os.path.join(result_dir_path(isolated_name), f'.checkpoint_{task_index}.json')
    return os.path.isfile(metric_path) and not os.path.isfile(checkpoint_path)


def _worker_init(r_path: Optional[str], r_blas_threads: Optional[int], gpu_semaphore) -> None:
    """Runs once per worker PROCESS, before it handles any task -
    initialises this process's own R environment/rpy2 conversion context
    (required: R/rpy2 state is per-process, never safely shared - see
    module docstring) and installs the shared GPU-slot semaphore (a no-op
    for CPU-only models; see pipeline_utils.gpu_slot()).

    Defect D4 fix: a small random jitter runs first (see
    `_startup_jitter()`), and the R/rpy2 setup calls are each wrapped in
    `_retry_transient_mount_error()` - both aimed at the same production
    failure (every worker's startup import hitting a network scratch
    mount in the same ~1s window - see the note above `_MP_CONTEXT`)."""
    _startup_jitter()
    _retry_transient_mount_error(
        lambda: configure_r_environment(r_path, r_blas_threads=r_blas_threads),
        what="configure_r_environment",
    )
    _retry_transient_mount_error(init_rpy2_conversion, what="init_rpy2_conversion")
    if gpu_semaphore is not None:
        install_gpu_semaphore(gpu_semaphore)


def _run_isolated_task(gp_kwargs: dict, task_index: int) -> Tuple[int, bool, Optional[str]]:
    """Worker-process entry point: run GP() for EXACTLY task index
    `task_index` (PARALLEL={'batch_id': task_index, 'batch_size': 1}),
    writing into that task's own isolated RESULT_NAME. Never raises back
    to the pool - failures are reported in the return tuple so one
    worker's crash can't take down the whole
    ProcessPoolExecutor.map()/as_completed() loop (failure isolation, per
    the design blueprint).

    Returns (task_index, success, error_message_or_None).
    """
    try:
        # Update ID 2, Defect D3 fix (see PATCH_NOTES_D2.md): this deferred
        # import used to sit OUTSIDE the try/except below - an import-time
        # failure (e.g. a transient hiccup loading a compiled extension
        # under heavy concurrent multi-worker load) therefore propagated
        # straight out of this function uncaught, instead of being
        # reported through the same (task_index, success, error) contract
        # every OTHER failure mode here already uses ("never raises back
        # to the pool" - see this function's own docstring above). Moved
        # inside the try so an import failure is handled identically to a
        # failure inside GP() itself.
        #
        # Defect D4 fix: the import itself (which reads genomic_prediction.py
        # and, transitively, Preprocess/plink_io.py and the models package
        # off disk) is wrapped in a bounded retry for the narrow class of
        # transient network-mount errors observed in production (see the
        # note above `_MP_CONTEXT`) - a real import error (syntax error,
        # missing dependency, etc.) is not on that allow-list and still
        # fails immediately, on the first attempt.
        def _import_gp():
            from genomic_prediction import GP
            return GP
        GP = _retry_transient_mount_error(_import_gp, what=f"import genomic_prediction (task {task_index})")
        GP(**gp_kwargs)
        return task_index, True, None
    except Exception as exc:  # noqa: BLE001 - deliberately broad: report, never crash the pool
        return task_index, False, f"{exc!r}\n{traceback.format_exc()}"


def _run_in_fresh_worker(fn, args: tuple, r_path: Optional[str], r_blas_threads: Optional[int],
                          gpu_semaphore) -> tuple:
    """Run ``fn(*args)`` in a single, FRESH, disposable worker PROCESS -
    never inline in the caller's own process - and return whatever ``fn``
    itself returns on success, or a synthesized failure tuple (shaped
    exactly like one of ``fn``'s own failure returns) if that worker
    process itself died unexpectedly - e.g. the same underlying fault
    that failed the original attempt recurring.

    Update ID 2, Defect D2/D3 fix (see PATCH_NOTES_D2.md): used for RETRY
    attempts specifically, by both this module's own
    ``run_batch_with_intra_batch_parallelism()`` and
    ``intra_task_parallel.run_batch_with_model_level_parallelism()``. A
    first attempt already runs isolated inside a ``ProcessPoolExecutor``
    worker; retrying inline in the parent process (as this codebase
    originally did) defeats that same isolation for the retry path
    specifically - if the underlying cause of the original failure
    recurs, it would kill the entire orchestrating process outright
    (losing every other in-flight/queued unit and any chance to report
    the failure cleanly), instead of just failing that one retry attempt,
    exactly as a first-attempt failure already does.

    ``fn`` must be a module-level (picklable) callable whose first
    positional argument is a ``gp_kwargs`` dict and whose return value is
    ``(*identifier_fields, success, error_message_or_None)`` - i.e.
    ``_run_isolated_task`` (``identifier_fields = (task_index,)``) or
    ``intra_task_parallel._run_unit`` (``identifier_fields = (task_index,
    group_index)``) - both already follow this shape. ``args`` is the
    full positional-argument tuple ``fn`` itself expects (``gp_kwargs``
    first, then its own identifier field(s))."""
    try:
        with ProcessPoolExecutor(
            max_workers=1, mp_context=_MP_CONTEXT,
            initializer=_worker_init, initargs=(r_path, r_blas_threads, gpu_semaphore),
        ) as pool:
            future = pool.submit(fn, *args)
            return future.result()
    except Exception as exc:  # noqa: BLE001 - the retry worker process itself died unexpectedly
        # `args[0]` is always fn's own gp_kwargs argument; every argument
        # AFTER it is one of fn's own leading identifier field(s), which a
        # SUCCESSFUL call to fn always echoes back before its trailing
        # (success, error) pair - mirroring that same shape here lets
        # every caller treat this exactly like an ordinary failure
        # result, without needing to know which of the two worker entry
        # points it called.
        return (*args[1:], False, f"worker process error on retry: {exc!r}")


def _read_csv_if_exists(path: str) -> Optional["pd.DataFrame"]:
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    if df.shape[0] == 0:
        return None
    return df


def _merge_isolated_results_into_batch(
    result_name: str, batch_id: int, task_indices: List[int], parallel: bool = True,
) -> None:
    """Concatenate every successfully-completed task index's own isolated
    result CSVs (in task-index order, for reproducibility) into the outer
    batch's real, standard result files - the shape
    ``run_step2_assemble.py``/``assemble.py`` already expect, unmodified.
    Only ever called once EVERY task index in `task_indices` has already
    been confirmed complete (see `_task_already_complete`).

    Parameters
    ----------
    parallel : bool, default True (ver4-4 Stage 6 / R3.g)
        Mirrors ``checkpoint_utils.result_file_paths()``'s own ``parallel``
        flag exactly, for the OUTER merge target only (never for the
        per-task isolated files this reads FROM, which are always written
        ``parallel=True`` regardless - see ``_sub_kwargs_for()``'s own
        unconditional per-task ``PARALLEL`` override, unrelated to this
        parameter). ``True`` (every existing caller before Stage 6) writes
        ``Metric_<batch_id>.csv``-style suffixed files - byte-for-byte
        this function's pre-Stage-6 behaviour, for a real Parallel-mode
        array-job batch. ``False`` (new, Stage 6 - used only by
        ``run_sequential.py``'s own ``SEQUENTIAL_INTRA_BATCH`` route, via
        ``run_batch_with_intra_batch_parallelism(..., sequential_mode=
        True)``) writes plain, UNSUFFIXED files instead - the naming
        ``batch_reader.ResultSet(RESULT_NAME, source='combined')`` reads
        (``_combined_path()`` always resolves ``parallel=False`` - see
        that module), and clears the plain, unsuffixed checkpoint to
        match, rather than a ``_<batch_id>``-suffixed one that nothing
        else in a Sequential run would ever look for.
    """
    out_paths = _ckpt.result_file_paths(result_name, batch_id, parallel)

    for key in _MERGE_KEY_ORDER:
        out_path = out_paths[key]
        frames = []
        for task_index in task_indices:
            isolated_name = _isolated_result_name(result_name, batch_id, task_index)
            in_path = _ckpt.result_file_paths(isolated_name, task_index, True)[key]
            df = _read_csv_if_exists(in_path)
            if df is not None:
                frames.append(df)
        if not frames:
            # Nothing to write for this key from any task in this batch -
            # matches GP()'s own "only ever written when non-empty/model-
            # selection-dependent" behaviour for the conditional files
            # (effect/interactions/attention_total/weight/hp_record); the
            # always-written keys (record/result_train/result_valid/
            # result_test/stats) simply end up with 0 total tasks having
            # contributed rows, which only happens for an outer batch that
            # itself covers 0 real scenarios (e.g. entirely beyond the end
            # of the sample table) - in which case an empty/absent file is
            # also exactly what the original serial code path would leave.
            continue
        merged = pd.concat(frames, ignore_index=True)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        merged.to_csv(out_path, index=False)

    # Success: no checkpoint should remain for the outer batch id, so
    # checkpoint_utils.check_batch_status() reports it as complete -
    # matches GP()'s own clear_checkpoint() call on a fully successful run.
    # `parallel` here matches the CSV naming above for the exact same
    # reason - a plain Sequential run's own resume logic (if any is ever
    # added on top of this route) would look for an unsuffixed checkpoint,
    # never a `_<batch_id>`-suffixed one.
    _ckpt.clear_checkpoint(result_name, batch_id, parallel)


def _cleanup_isolated_folders(result_name: str, batch_id: int) -> None:
    """Remove every isolated per-task folder for this outer batch, once
    its results have been merged into the real batch-level files - these
    are pure scratch space, never referenced again afterward."""
    root = os.path.join(result_dir_path(result_name), '.intra_batch_parallel', f'batch_{batch_id}')
    try:
        if os.path.isdir(root):
            shutil.rmtree(root)
    except OSError as exc:
        print(f"[intra_batch_parallel] WARNING: could not remove scratch folder '{root}': {exc!r}. "
              f"This does not affect correctness (it will simply be skipped/overwritten on any "
              f"future re-run of this batch) - safe to delete manually.")


# ---------------------------------------------------------------------------
# Update ID 2, R1, T9 - public aliases.
#
# intra_task_parallel.py (Update ID 2's new model-level fan-out module)
# reuses this module's isolated-result-folder layout, completion check,
# merge and cleanup logic VERBATIM for its own, finer-grained (task,
# model-group) tier, rather than re-implementing them - the same "reuse,
# don't re-derive" precedent architecture doc §8 already sets for
# `_map_genes_to_markers`. The leading-underscore originals are kept
# exactly as they were so every call site inside THIS module (all written
# before this alias existed) needs no changes at all.
# ---------------------------------------------------------------------------
isolated_result_name = _isolated_result_name
task_already_complete = _task_already_complete
merge_isolated_results_into_batch = _merge_isolated_results_into_batch
cleanup_isolated_folders = _cleanup_isolated_folders
worker_init = _worker_init
# Update ID 2, Defect D2/D3 fix - reused verbatim by
# intra_task_parallel.py's own retry loop, for the same reasons (see
# _run_in_fresh_worker()'s own docstring and _MP_CONTEXT's module-level
# note above).
run_in_fresh_worker = _run_in_fresh_worker
MP_CONTEXT = _MP_CONTEXT
# Defect D4 fix - reused verbatim by intra_task_parallel.py's own
# `_run_unit()` deferred import, for the same reason (see the note above
# `_MP_CONTEXT`, and `_retry_transient_mount_error()`'s own docstring).
retry_transient_mount_error = _retry_transient_mount_error


# ---------------------------------------------------------------------------
# Update ID 2, Appendix B finding F1 (defect fix, independent of R1/R2).
#
# _merge_isolated_results_into_batch() above only ever copies the ten
# RESULT_FILE_NAMES-keyed CSVs out of an isolated task folder. Everything
# else a task's own isolated GP() call may have written under that same
# folder - LD_decay_plots/ (and its data/ subfolder), BGLR_output/,
# _plink_bim_marker_info.csv, <MODEL_NAME>_gene_coordinates_<phenotype>.csv
# (architecture doc §16) - was, until this fix, simply destroyed the moment
# _cleanup_isolated_folders() rmtree's the whole scratch tree, since nothing
# ever moved it out first. That silently starved run_step2_assemble.py's
# average_and_plot_ld_decay() of any data to average after ANY intra-batch-
# parallel run, with no error or warning anywhere.
#
# Fixed by relocating - not copying - every non-result-file entry out of
# each task's isolated folder into the REAL Result/<RESULT_NAME>/ (the same
# top-level location a Sequential/single-process run would have written it
# to directly), before that isolated folder is ever removed. Called once
# per task from run_batch_with_intra_batch_parallelism() below, and again
# (per model-group) from intra_task_parallel.py - see that module's own
# call site - since R1's model-level fan-out multiplies the number of
# isolated folders (and therefore the blast radius of this defect) by the
# model-group count on top of Phase 2's task count.
# ---------------------------------------------------------------------------
_PROTECTED_RESULT_FILE_RE = re.compile(
    r'^(' + '|'.join(re.escape(os.path.splitext(name)[0]) for name in _ckpt.RESULT_FILE_NAMES.values())
    + r')(_\d+)?\.[A-Za-z0-9]+$'
)
_PROTECTED_CHECKPOINT_RE = re.compile(r'^\.checkpoint(_\d+)?\.json(\.tmp)?$')
_SCRATCH_TIER_DIR_NAMES = ('.intra_batch_parallel', '.intra_task_parallel')


def _relocate_file(src_path: str, dst_dir: str) -> bool:
    """Move a single side-artefact FILE from `src_path` into `dst_dir`,
    collision-safe via `unique_path()` (several tasks/groups can
    independently produce a file with the same name, e.g.
    `_plink_bim_marker_info.csv`, since each starts from its own empty
    scratch folder). Returns True on success; logs and returns False on
    failure rather than raising, since losing a diagnostic side artefact
    must never fail the run that produced the actual result it documents."""
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = unique_path(os.path.join(dst_dir, os.path.basename(src_path)))
    try:
        shutil.move(src_path, dst_path)
        return True
    except OSError as exc:
        print(f"[intra_batch_parallel] WARNING: could not relocate side artefact "
              f"'{src_path}' -> '{dst_path}': {exc!r}. It will be permanently lost when the "
              f"scratch folder is cleaned up - this does not affect any Metric/Marker_effect/"
              f"Prediction result, only diagnostic output (LD-decay data, BGLR traces, ...).")
        return False


def _relocate_dir(src_dir_path: str, dst_root: str, top_name: str) -> bool:
    """Recursively move every FILE under the side-artefact directory
    `src_dir_path` into the matching relative path under
    `dst_root/<top_name>/...`, applying `unique_path()` collision
    avoidance to each individual FILE (never to a whole directory) -
    required because several different tasks/model-groups each start
    from their own EMPTY copy of a folder like `LD_decay_plots/data/` and
    so can easily pick the same filename independently; moving whole
    directories on top of each other would silently overwrite one
    task's/group's data with another's instead of preserving both."""
    moved_any = False
    for dirpath, _dirnames, filenames in os.walk(src_dir_path):
        rel_dir = os.path.relpath(dirpath, src_dir_path)
        dst_subdir = os.path.join(dst_root, top_name) if rel_dir == '.' else os.path.join(dst_root, top_name, rel_dir)
        for filename in filenames:
            if _relocate_file(os.path.join(dirpath, filename), dst_subdir):
                moved_any = True
    return moved_any


def relocate_side_artefacts(src_result_name: str, dst_result_name: str) -> None:
    """F1 fix: move every non-result-file artefact out of an isolated
    (task- or model-group-level) scratch RESULT_NAME into the real,
    top-level `dst_result_name`, before that scratch folder is removed.

    "Non-result-file" is determined by EXCLUSION, not an enumerated
    allow-list: any top-level entry under `Result/<src_result_name>/`
    that is neither one of the ten `checkpoint_utils.RESULT_FILE_NAMES`
    CSVs (in either their plain or `_<idx>`-suffixed form) nor a
    `.checkpoint[_<idx>].json[.tmp]` file is treated as a side artefact
    and relocated. This deliberately covers whatever the LD-decay
    subfolder happens to be named (it is user-configurable via
    `LD_prune['decay_plot']['subfolder']`, default `'LD_decay_plots'` -
    see genomic_prediction.py), `BGLR_output/`,
    `_plink_bim_marker_info.csv`, and `<MODEL_NAME>_gene_coordinates_
    <phenotype>.csv` without hard-coding any of those names, and would
    also catch any future side-artefact type a model file starts writing,
    without needing this function updated in lockstep.

    A no-op (nothing to log) if `src_result_name` doesn't exist on disk at
    all, or contains nothing but the protected result/checkpoint files.
    Never raises - a failure to relocate one artefact (see
    `_relocate_file`) is logged and skipped, never allowed to fail the
    batch whose actual scientific output has already been safely merged
    by the time this is called.
    """
    src_dir = result_dir_path(src_result_name)
    if not os.path.isdir(src_dir):
        return
    dst_dir = result_dir_path(dst_result_name)

    moved_any = False
    for entry in sorted(os.listdir(src_dir)):
        if _PROTECTED_RESULT_FILE_RE.match(entry) or _PROTECTED_CHECKPOINT_RE.match(entry):
            continue
        if entry in _SCRATCH_TIER_DIR_NAMES:
            # Defensive only - a leaf task/group folder should never
            # itself contain another tier's scratch directory; skip
            # rather than recurse into (and possibly relocate out from
            # under) a tier that manages its own lifecycle.
            continue
        src_path = os.path.join(src_dir, entry)
        if os.path.isdir(src_path):
            if _relocate_dir(src_path, dst_dir, entry):
                moved_any = True
        else:
            if _relocate_file(src_path, dst_dir):
                moved_any = True

    if moved_any:
        print(f"[intra_batch_parallel] Relocated side artefact(s) (LD-decay data, BGLR MCMC "
              f"traces, PLINK-derived marker info, gene-coordinate CSVs, or similar) from "
              f"'{src_dir}' into '{dst_dir}' before discarding the scratch folder (F1 fix - "
              f"see the ver4-3 design blueprint Appendix B).")


def run_batch_with_intra_batch_parallelism(
    gp_kwargs: dict,
    n_cpu_workers: int,
    n_gpu_slots: int = 0,
    min_tasks_for_parallel: int = 2,
    max_task_retries: int = 2,
    sequential_mode: bool = False,
) -> None:
    """Process one Parallel-mode array-job batch's tasks using up to
    `n_cpu_workers` worker processes instead of GP()'s own fully serial
    driving loop - or, whenever that would not be worthwhile, falls back
    to a single ordinary serial `GP(**gp_kwargs)` call, UNCHANGED from
    this pipeline's original behaviour.

    Parameters
    ----------
    gp_kwargs : every keyword argument `genomic_prediction.GP()` accepts
        for this batch, exactly as `run_step1_batch.py` would otherwise
        pass to a single direct `GP(**gp_kwargs)` call - including
        `gp_kwargs['PARALLEL'] = {'batch_id': <outer batch id>,
        'batch_size': <outer batch size>}`. `gp_kwargs['progress_callback']`
        (not reliably picklable, and generally unused by headless HPC
        runs) is dropped for the per-task sub-calls if present; each
        isolated sub-run's own console/log output is exactly what GP()
        would already print for that task.

        ver4-4 Stage 6 (R3.g) / PC-2 finding: `gp_kwargs['PARALLEL']`
        MUST be a real dict here, never `None` - `PARALLEL=None` hits the
        early no-op branch below (`parallel_cfg is None`) and this
        function degenerates to a single, ordinary, un-fanned-out
        `GP(**gp_kwargs)` call, exactly as if this function had never
        been called at all. This is true regardless of `sequential_mode`.
        A `run_sequential.py` caller that wants genuine intra-batch
        fan-out under `SEQUENTIAL_INTRA_BATCH=True` must therefore build
        its OWN synthetic outer `PARALLEL={'batch_id': 0, 'batch_size':
        <the run's own total task count>}` before calling this function -
        see that script's own `SEQUENTIAL_INTRA_BATCH` branch for exactly
        this construction. (This is the blueprint's own §10 PC-2
        pre-check, resolved during Stage 6 implementation: calling this
        function with `PARALLEL=None`, as the blueprint's §2.3.2 diagram
        literally shows, does NOT reach the fan-out/merge code at all,
        so the "does it write Metric.csv or Metric_0.csv" question the
        blueprint posed does not arise in that literal case - it writes
        neither, because GP() itself runs the whole table in one process,
        exactly like today's un-modified Sequential mode. The naming
        question only arises once a caller constructs a real, single-outer
        -batch `PARALLEL` dict to force fan-out to actually happen -
        see `sequential_mode` below for how that case is handled.)
    n_cpu_workers : number of worker PROCESSES to use. <= 1 is a strict
        no-op - falls straight back to one ordinary serial GP() call.
    n_gpu_slots : maximum number of GPU-dispatched model calls (MLP/GAT)
        allowed to run CONCURRENTLY across every worker sharing one
        physical GPU - 0 disables the limiter (no cap; only sensible for
        CPU-only runs, or a GPU with enough memory for every worker at
        once). See `pipeline_utils.gpu_slot()`.
    min_tasks_for_parallel : if this outer batch's own `batch_size` is
        smaller than this, the worker-pool startup/R-sourcing overhead
        isn't worth paying - falls back to serial, unchanged.
    max_task_retries : how many additional serial retry attempts a task
        index gets (in the PARENT process) if its worker-process attempt
        failed, before this function gives up and re-raises - keeps a
        single transient failure (e.g. a momentary filesystem hiccup) from
        failing an entire batch outright. Defect D4 fix: default raised
        1 -> 2, and each retry attempt now waits out a short backoff
        first (see below) - a production batch was observed failing ALL
        of its tasks in the same ~1s window from a transient network-
        mount error (see the note above `_MP_CONTEXT`), which a single,
        immediate retry is not reliably enough time to recover from if
        the underlying mount hiccup is still ongoing. Additional
        Requirements 9: this is now a CEILING, not a guarantee - a pass
        (initial attempt, or a retry attempt) whose OWN failures look
        SYSTEMIC (`_systemic_failure_note()` - most/all of that pass died
        worker-process-killed, an OOM/node-level signature no amount of
        same-configuration retrying can fix) skips any REMAINING retry
        attempts and raises immediately instead, rather than grinding
        through the rest of `max_task_retries` at ~minutes per task for a
        failure it has already twice demonstrated cannot succeed.
    sequential_mode : bool, default False (ver4-4 Stage 6 / R3.g)
        When True, the outer batch's MERGED result files are written
        UNSUFFIXED (`Metric.csv`, not `Metric_<batch_id>.csv`) - the
        naming `run_sequential.py`'s own plotting chain expects, via
        `batch_reader.ResultSet(RESULT_NAME, source='combined')` (see
        `_merge_isolated_results_into_batch()`'s own `parallel` parameter,
        which this simply inverts and forwards). Every OTHER behaviour of
        this function - the per-task isolation, retry logic, side-artefact
        relocation, isolated-folder cleanup - is completely unchanged;
        only the FINAL merge target's naming differs. False (every
        existing caller: `run_step1_batch.py`'s own real Parallel-mode
        array-job batches) reproduces this function's exact pre-Stage-6
        behaviour.

    Raises
    ------
    RuntimeError if, after all retries, at least one task index still
    failed - exactly like an uncaught exception from the original serial
    `GP()` call would propagate to `run_step1_batch.py`'s caller (the
    scheduler sees a non-zero exit and can resubmit; a resubmission of the
    SAME batch skips every already-succeeded task index automatically -
    see `_task_already_complete`).
    """
    parallel_cfg = gp_kwargs.get('PARALLEL')
    if n_cpu_workers <= 1 or parallel_cfg is None or int(parallel_cfg.get('batch_size', 1)) < min_tasks_for_parallel:
        # Req 2 fix (2026-09): this branch calls GP() directly, IN THIS
        # (the caller's) PROCESS - no ProcessPoolExecutor is ever created
        # on this path (the function returns right after), so it is
        # exactly as fork-safe as run_step1_batch.py's own plain-serial
        # route, which already calls init_rpy2_conversion() before its
        # own direct GP() call for precisely that reason (see that
        # script's own comment beside its `configure_r_environment()`
        # call). Before this fix, a caller reaching this fallback via
        # run_step1_batch.py's N_CPU_WORKERS>1 route (which deliberately
        # withholds init_rpy2_conversion() from the parent, ONLY correct
        # for the case where this function actually forks/spawns workers
        # below) ran GP() with NO rpy2 conversion context ever
        # initialised in this process at all - every rrBLUP/GBLUP/BayesB/
        # RKHS call then returned its R result UNCONVERTED (confirmed:
        # `result[name]` raises `TypeError: Indices must be integers or
        # slices, not <class 'str'>` on every single access in that
        # state - see pipeline_utils.r_list_get()'s own Req 2 fix note).
        # A `PARALLEL.batch_size` of exactly 1 - the single most common
        # shape for a one-task-per-array-element HPC job - reaches this
        # exact branch every time N_CPU_WORKERS/N_MODEL_WORKERS>1 is also
        # set, which is why this reproduced so reliably in production.
        # init_rpy2_conversion() is idempotent/cheap when R/rpy2 hasn't
        # been touched yet in this process (the normal case here, since
        # run_step1_batch.py's own N_CPU_WORKERS>1 route never calls it
        # first) and a safe no-op if something upstream already did.
        init_rpy2_conversion()
        from genomic_prediction import GP
        _fallback_kwargs = dict(gp_kwargs)

        # Requirement.md item 1 fix - reclaim the CPU share this batch
        # will never actually use via concurrent task-level workers.
        #
        # gp_kwargs['N_JOBS']/['PLINK_THREADS']/['R_BLAS_THREADS'] arrive
        # here ALREADY divided down by N_CPU_WORKERS (main_app.py's own
        # HPC export handler writes N_JOBS = PLINK_THREADS =
        # cpus_per_task // N_CPU_WORKERS at config-write time - see that
        # module's 'Pipeline compute settings' export block), on the
        # assumption that N_CPU_WORKERS concurrent worker PROCESSES will
        # each independently claim their own equal share of this batch's
        # CPU allocation. Reaching THIS branch means that assumption just
        # failed for one of three reasons (n_cpu_workers<=1, no PARALLEL,
        # or - the common case - too few tasks in this batch to be worth
        # fanning out at all): no sibling worker process is going to run,
        # so nothing else will ever claim the other (n_cpu_workers - 1)
        # shares of the CPU budget those settings were divided by. Left
        # un-reclaimed, they simply sit idle for the rest of this call.
        #
        # This is exactly what starved LD pruning (PLINK2) and RF
        # filtering (scikit-learn) in a GPU + GAT_biological_prior_
        # knowledge run reported as "taking too much time": a one-task-
        # per-array-job GPU submission (batch_size=1 - the natural shape
        # when each task needs its own GPU) with N_CPU_WORKERS sized for
        # a many-task batch divided N_JOBS/PLINK_THREADS down to 1 at
        # config-write time, then landed in this exact serial fallback
        # (batch_size=1 < min_tasks_for_parallel) - so the FULL marker-
        # pool LD pruning + RF filtering GAT_biological_prior_knowledge's
        # own data-driven merge performs (architecture doc §12.4) ran on
        # a single CPU core while every other reserved core sat idle.
        #
        # Multiplying back by n_cpu_workers is safe precisely BECAUSE this
        # branch guarantees no concurrent sibling GP() call is competing
        # for the same budget (unlike the genuine fan-out path below,
        # where each of up to n_cpu_workers concurrent workers legitimately
        # needs to keep its own already-divided share). N_JOBS' `-1`
        # ("use every core") sentinel is left untouched - it is already
        # maximal, and `-1 * n_cpu_workers` is not a meaningful value.
        # Reclaimed values are still capped to the CPUs this process can
        # actually see (os.cpu_count()), mirroring the same conservative,
        # zero-configuration signal pipeline_utils.resolve_compute_
        # resources() already uses for its own N_MODEL_WORKERS clamp.
        if n_cpu_workers > 1:
            _detected_cpus = os.cpu_count() or 1
            _configured_n_jobs = int(_fallback_kwargs.get('N_JOBS', 1) or 1)
            _configured_plink_threads = max(1, int(_fallback_kwargs.get('PLINK_THREADS', 1) or 1))
            _configured_r_blas = _fallback_kwargs.get('R_BLAS_THREADS')
            _reclaimed_n_jobs = _configured_n_jobs
            if _configured_n_jobs > 0:
                _reclaimed_n_jobs = min(_configured_n_jobs * n_cpu_workers, _detected_cpus)
                _fallback_kwargs['N_JOBS'] = _reclaimed_n_jobs
            _reclaimed_plink_threads = min(_configured_plink_threads * n_cpu_workers, _detected_cpus)
            _fallback_kwargs['PLINK_THREADS'] = _reclaimed_plink_threads
            if _configured_r_blas:
                _fallback_kwargs['R_BLAS_THREADS'] = min(int(_configured_r_blas) * n_cpu_workers, _detected_cpus)
            _r_blas_note = (
                f", R_BLAS_THREADS {_configured_r_blas}->{_fallback_kwargs['R_BLAS_THREADS']}"
                if _configured_r_blas else ""
            )
            print(f"[intra_batch_parallel] Falling back to a single serial GP() call for this "
                  f"batch (too few tasks to fan out across N_CPU_WORKERS={n_cpu_workers} worker "
                  f"process(es)) - reclaiming the unused task-level CPU share: "
                  f"N_JOBS {_configured_n_jobs}->{_fallback_kwargs['N_JOBS']}, "
                  f"PLINK_THREADS {_configured_plink_threads}->{_fallback_kwargs['PLINK_THREADS']}"
                  f"{_r_blas_note} - so CPU-bound preprocessing (LD pruning/RF filtering) actually "
                  f"uses this batch's full CPU allocation instead of leaving most of it idle.")

        if sequential_mode and parallel_cfg is not None:
            # ver4-4 Stage 6 (R3.g): a Sequential caller builds a SYNTHETIC
            # outer PARALLEL={'batch_id': 0, 'batch_size': total_tasks}
            # purely to make the fan-out branch below reachable at all
            # (see this function's own PARALLEL docstring note above) - it
            # is not a real Parallel-mode marker and must never reach
            # GP() as one. Whenever fan-out turns out not to be worth
            # doing after all (too few tasks / N_CPU_WORKERS<=1) and this
            # function is about to fall back to one direct GP() call, that
            # synthetic PARALLEL is stripped back to None here so GP()
            # takes its ordinary, un-modified Sequential code path
            # (unsuffixed Metric.csv, the whole table in one process) -
            # covering the EXACT same scope (batch_size was already the
            # run's own total task count) rather than writing
            # Parallel-suffixed files a Sequential caller's own plotting
            # chain would never look for. Mutates the SAME _fallback_kwargs
            # built above (never a fresh `dict(gp_kwargs)` copy here) so the
            # CPU-reclaim adjustment just applied is preserved rather than
            # silently discarded.
            _fallback_kwargs['PARALLEL'] = None
        GP(**_fallback_kwargs)
        return

    result_name = gp_kwargs['RESULT_NAME']
    batch_id = int(parallel_cfg['batch_id'])
    interval = int(parallel_cfg['batch_size'])
    task_indices = list(range(batch_id * interval, batch_id * interval + interval))

    print(f"[intra_batch_parallel] Batch {batch_id}: fanning out {len(task_indices)} task index(es) "
          f"across up to {n_cpu_workers} worker process(es) (n_gpu_slots={n_gpu_slots}"
          f"{', sequential_mode=True (merged output will be UNSUFFIXED)' if sequential_mode else ''}).")

    to_run = [i for i in task_indices if not _task_already_complete(result_name, batch_id, i)]
    already_done = [i for i in task_indices if i not in to_run]
    if already_done:
        print(f"[intra_batch_parallel] Batch {batch_id}: {len(already_done)} task index(es) already "
              f"completed in a previous attempt - skipping: {already_done}")

    # Update ID 2, Defect D2 fix: the Manager's own background server
    # process is spawned via the same explicit _MP_CONTEXT as every other
    # worker process this module creates, for the same fork-safety reason
    # (see the module-level note beside _MP_CONTEXT's own definition).
    gpu_semaphore = _MP_CONTEXT.Manager().Semaphore(n_gpu_slots) if n_gpu_slots > 0 else None
    r_path = gp_kwargs.get('R_PATH')
    r_blas_threads = gp_kwargs.get('R_BLAS_THREADS')

    failed: Dict[int, str] = {}

    def _sub_kwargs_for(task_index: int) -> dict:
        sub_kwargs = dict(gp_kwargs)
        sub_kwargs['RESULT_NAME'] = _isolated_result_name(result_name, batch_id, task_index)
        sub_kwargs['PARALLEL'] = {'batch_id': task_index, 'batch_size': 1}
        sub_kwargs.pop('progress_callback', None)
        # Update ID ver4-9, R7 (correctness fix, found during Phase 2
        # itself): intra_task_parallel.py/intra_batch_parallel.py are OUT
        # OF R7's own touch-point scope - _merge_isolated_results_into_
        # batch() below still reads each isolated file's path via a plain
        # checkpoint_utils.result_file_paths() call with NO `compression=`
        # argument (i.e. it always assumes PLAIN, uncompressed isolated
        # files). Without this override, an isolated worker's own GP()
        # call would otherwise silently inherit whatever RESULT_COMPRESSION
        # the OUTER run configured (since `dict(gp_kwargs)` above copies
        # it verbatim) - defaulting to 'gzip' - and write GZIP-compressed
        # isolated files that the unmodified merge-read logic below would
        # then fail to find at all, silently discarding every task's
        # contribution to the 6 large result files. Forcing 'none' here,
        # unconditionally, keeps every isolated file exactly the plain CSV
        # this merge step has always expected, regardless of what the
        # outer run's own RESULT_COMPRESSION is - the OUTER, merged batch
        # file (out_paths above) is correspondingly also written as plain
        # CSV for a batch produced via this parallelism route (a
        # documented, narrow gap - see the ver4-9 Change Summary - rather
        # than a correctness risk).
        sub_kwargs['RESULT_COMPRESSION'] = 'none'
        return sub_kwargs

    if to_run:
        with ProcessPoolExecutor(
            max_workers=min(n_cpu_workers, len(to_run)), mp_context=_MP_CONTEXT,
            initializer=_worker_init, initargs=(r_path, r_blas_threads, gpu_semaphore),
        ) as pool:
            futures = {
                pool.submit(_run_isolated_task, _sub_kwargs_for(i), i): i for i in to_run
            }
            for future in as_completed(futures):
                task_index = futures[future]
                try:
                    _, success, error = future.result()
                except Exception as exc:  # noqa: BLE001 - the worker process itself died unexpectedly
                    success, error = False, f"worker process error: {exc!r}"
                if success:
                    print(f"[intra_batch_parallel] Batch {batch_id}: task {task_index} finished.")
                else:
                    print(f"[intra_batch_parallel] Batch {batch_id}: task {task_index} FAILED: {error}")
                    failed[task_index] = error

    # Additional Requirements 9 - a SYSTEMIC-looking initial pass (see the
    # module-level banner above `_WORKER_DEATH_SIGNATURES`) gets a loud,
    # actionable diagnostic printed immediately, BEFORE any retry is even
    # attempted - a person tailing the log should not have to wait for the
    # eventual RuntimeError (which, at up to `max_task_retries` serial
    # attempts per failed task, could be a long time away) to learn why.
    # This is purely additive logging here - it does not change whether a
    # retry is attempted (that decision is still "were there any
    # failures at all", unchanged) - only the RETRY loop below short-
    # circuits early once a systemic signature has been seen TWICE
    # (initial pass AND at least one retry pass).
    _initial_systemic_note = _systemic_failure_note(
        failed, len(to_run), batch_id, n_cpu_workers, context="initial attempt",
    )
    if _initial_systemic_note:
        print(_initial_systemic_note)

    # Requirement 7 (failure isolation): retry only the task indices that
    # actually failed - each retry attempt gets its own FRESH, disposable
    # worker process (via _run_in_fresh_worker(), Update ID 2 Defect D2/D3
    # fix - see PATCH_NOTES_D2.md), never inline in THIS (parent) process,
    # so a single transient failure can't take the whole batch (or the
    # rest of this retry loop) down with it, without masking a genuinely
    # reproducible one (which will simply fail again here and be
    # reported).
    _abort_early = False
    _attempts_run = 0
    for attempt in range(1, max_task_retries + 1):
        if not failed:
            break
        _attempts_run = attempt
        retry_indices = list(failed.keys())
        # Defect D4 fix: a short backoff (with jitter) before each retry
        # attempt, so a transient network-mount hiccup (see the note
        # above `_MP_CONTEXT`) actually has time to clear before the same
        # tasks are attempted again - the original code retried
        # immediately, in the very same second the first attempt failed.
        _backoff = min(15.0, 2.0 * (2 ** (attempt - 1))) * (0.5 + random.random())
        print(f"[intra_batch_parallel] Batch {batch_id}: retrying {len(retry_indices)} failed task "
              f"index(es) in {_backoff:.1f}s, each in a fresh worker process (attempt "
              f"{attempt}/{max_task_retries}): {retry_indices}")
        time.sleep(_backoff)
        failed = {}
        for task_index in retry_indices:
            _, success, error = _run_in_fresh_worker(
                _run_isolated_task, (_sub_kwargs_for(task_index), task_index),
                r_path, r_blas_threads, gpu_semaphore,
            )
            if success:
                print(f"[intra_batch_parallel] Batch {batch_id}: task {task_index} succeeded on retry.")
            else:
                print(f"[intra_batch_parallel] Batch {batch_id}: task {task_index} FAILED again on retry: {error}")
                failed[task_index] = error

        # Additional Requirements 9 - if THIS retry pass ALSO looks
        # systemic (not just the initial attempt), further retries at the
        # exact same N_CPU_WORKERS/memory configuration have now twice
        # demonstrated they cannot succeed - stop here rather than
        # continuing to serially burn through the remaining
        # `max_task_retries` attempts (each task in this batch's log took
        # ~2+ minutes to fail again; a full sweep can run to hours before
        # finally raising). Only short-circuits when the INITIAL pass was
        # ALSO systemic - a retry pass that happens to look systemic on
        # its own, following a non-systemic initial pass, is more likely
        # a coincidence of which few tasks remained than a confirmed
        # repeat, so the ordinary retry budget still applies in that case.
        _retry_systemic_note = _systemic_failure_note(
            failed, len(retry_indices), batch_id, n_cpu_workers, context=f"retry attempt {attempt}",
        )
        if _initial_systemic_note and _retry_systemic_note:
            print(_retry_systemic_note)
            print(f"[intra_batch_parallel] Batch {batch_id}: this configuration has now failed "
                  f"systemically twice (initial attempt and retry attempt {attempt}/{max_task_retries}) "
                  f"- stopping further retries early rather than continuing to burn walltime against an "
                  f"unwinnable configuration.")
            _abort_early = True
            break

    if failed:
        _attempts_made = (
            f"initial attempt + {_attempts_run} retry attempt(s) "
            f"({'stopped early - see above' if _abort_early else f'of {max_task_retries} allowed'})"
            if _attempts_run > 0 else "initial attempt (no retries configured)"
        )
        raise RuntimeError(
            f"[intra_batch_parallel] Batch {batch_id}: {len(failed)} task index(es) failed after "
            f"{_attempts_made}: {sorted(failed.keys())}. Re-submit this SAME batch "
            f"(same RESULT_NAME, same batch_id) to resume - every already-succeeded task index will be "
            f"skipped automatically. First failure detail: {next(iter(failed.values()))}"
        )


    _merge_isolated_results_into_batch(result_name, batch_id, task_indices, parallel=not sequential_mode)
    # F1 fix: relocate every task's side artefacts (LD-decay data, BGLR
    # traces, PLINK/gene-coordinate CSVs, ...) into the real, top-level
    # RESULT_NAME - the same place a Sequential/single-process run would
    # have written them directly - BEFORE the isolated scratch tree below
    # is removed. Must run per task index (each isolated GP() call wrote
    # its own, independent copy of these paths).
    for task_index in task_indices:
        relocate_side_artefacts(_isolated_result_name(result_name, batch_id, task_index), result_name)
    _cleanup_isolated_folders(result_name, batch_id)
    print(f"[intra_batch_parallel] Batch {batch_id}: all {len(task_indices)} task index(es) complete "
          f"and merged into the standard batch-level result files.")
