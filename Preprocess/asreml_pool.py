"""Preprocess/asreml_pool.py - ver4-25, R9-S2/S3 (design blueprint stage 5).

A small, self-contained process pool for running MULTIPLE independent
ASReml-R fits concurrently, on top of the exact same rpy2/R initialisation
pattern `intra_batch_parallel.py`/`intra_task_parallel.py` already use for
their own worker processes (`configure_r_environment()` + `init_rpy2_
conversion()`, called once per worker process, then reused for every unit
that worker goes on to handle).

Why this exists (design blueprint R9, "Speed up ASReml-related work"):
today, every ASReml-R fit `genomic_prediction.py::GP()` needs - the main
per-trait adjustment, its accuracy-metrics companions, R7's step (i) - runs
IN-PROCESS, ONE AT A TIME, via `rpy2` (a single R session cannot run two
fits concurrently - "Rejected" list, R9). When a run has several
independent phenotype traits, each needing its own main fit (and,
depending on configuration, one or two companions and/or an R7 step (i)
fit), those fits are all mutually independent and could in principle run
in parallel, one per worker process, each with its own R session.

`N_JOBS` is this pool's OWN sizing input (never `N_CPU_WORKERS`/
`N_MODEL_WORKERS` - design blueprint R9, "Relation to R14": ASReml
parallelism is a property of how many independent trait-level fits a run
has, completely orthogonal to the array-job/task-level/model-level fan-out
widths architecture doc SS14 documents).

SCOPE OF THIS MODULE (ver4-25). This module implements the pool
infrastructure itself - spawning workers, initialising each one's own R/
rpy2 session exactly once, and a submit/collect API over that pool - as
real, usable, standalone code. It does NOT, in this update, replace
`genomic_prediction.py::GP()`'s own existing per-trait lazy fit/cache
closures (`_asreml_get_unmasked_result()` et al.) with calls through this
pool. That integration - enumerating every hoistable fit for a run BEFORE
the task loop starts, submitting them here, and having the existing
closures serve their result from the S1 fit cache instead of fitting
in-process - is recorded as the concrete next step in the ver4-25 Change
Summary's own Known Limitations section, not attempted here: it touches
the per-task dispatch loop's own caching contract in a way that cannot be
verified without a licensed ASReml-R install and a real multi-trait run to
compare against (design blueprint AC9.3 requires bit-identical output at
pool size 1 vs. N - a claim this environment has no way to check). Wiring
untested concurrency into the run's own numeric results is a worse outcome
than shipping the tested-as-far-as-possible infrastructure on its own.

Design (mirrors intra_batch_parallel.py's own worker-process discipline):

    - `spawn` multiprocessing context ONLY (never `fork`) - rpy2/R must
      never inherit a parent process's own live R session state (the same
      reasoning intra_batch_parallel.py's own `_MP_CONTEXT` comment gives).
    - Each worker process initialises its OWN R session exactly once
      (`_ensure_worker_r_ready()`, a module-level, per-process "already
      initialised" guard - safe because a `ProcessPoolExecutor` worker
      process is reused for many submitted units, and `configure_r_
      environment()`/`init_rpy2_conversion()` are themselves idempotent,
      but re-sourcing `asreml_fit.R` and re-checking the licence on every
      single submitted fit would be wasteful).
    - A GRM (or any other large, fitting-relevant matrix) is passed to a
      worker as a `.npy` file PATH, never as an in-memory array pickled
      through the pool's own IPC channel - large genomic relationship
      matrices are exactly the kind of payload multiprocessing's default
      pickling handles poorly (design blueprint S2's own "GRM inverses and
      V passed as `.npy` paths").
"""

from __future__ import annotations

import multiprocessing
import os
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

# `spawn`, never `fork` - see this module's own top-of-file docstring.
_MP_CONTEXT = multiprocessing.get_context('spawn')

# Set to True inside a worker process the FIRST time it initialises its own
# R/rpy2 session (`_ensure_worker_r_ready()`) - a fresh, empty `False` in
# every new worker process (module-level state is NOT shared across
# processes; each spawned worker gets its own copy), so this correctly
# guards "once per worker process", not "once per pool" or "once ever".
_worker_r_ready = False


@dataclass
class AsremlFitJob:
    """One fit request submitted to the pool.

    Attributes
    ----------
    job_id : caller-assigned identifier (e.g. a `(role_signature, trait,
        scope_hash)` tuple - design blueprint S1's own cache key) - echoed
        back unchanged in the matching `AsremlFitJobResult`, so the caller
        can match results back to requests regardless of completion order
        (`as_completed()` never preserves submission order).
    fit_kind : which Preprocess.asreml_blup function to call - one of
        `'fit_unmasked_adjustment'`, `'fit_concurrent_variance_model'`,
        `'fit_blue_refinement_step1'` (the three hoistable fit kinds R9-S2
        enumerates before the task loop). Dispatched by name (not a
        pickled function reference) because a bound closure from `GP()`'s
        own nested-function scope cannot cross a process boundary at all
        (architecture doc SS18, "Every worker entry point is a MODULE-
        level function ... because a closure is not picklable").
    raw_records : the exact DataFrame the chosen fit function's own first
        positional argument expects - crosses the pool's own IPC channel
        as an ordinary pickled DataFrame (unlike a GRM, a phenotype raw-
        records table is not large enough to warrant `.npy` staging).
    kwargs : every OTHER keyword argument the chosen fit function expects,
        EXCEPT `grm_inv` - passed here as ordinary picklable values
        (strings, numbers, lists, small dicts).
    grm_path : optional path to a `.npy` file holding the GRM (or its
        inverse) this fit needs, staged by the CALLER (see
        `stage_grm_npy()`) before submission - loaded back into memory
        inside the worker process, right before the fit call, never
        pickled through the pool's own IPC channel.
    """
    job_id: Any
    fit_kind: str
    raw_records: "Any"
    kwargs: Dict[str, Any] = field(default_factory=dict)
    grm_path: Optional[str] = None


@dataclass
class AsremlFitJobResult:
    """One completed (or failed) job's own outcome.

    `result` is whatever the chosen `Preprocess.asreml_blup` function
    itself returns (an `AsremlResult`, or a `Dict[str, float]` for
    `fit_blue_refinement_step1`) when `error` is `None`; `None` otherwise.
    `error` is the formatted exception (never a raw exception object -
    those are not reliably picklable either) when the fit failed in the
    worker - the caller is responsible for deciding whether that is fatal
    for the run (mirrors `intra_task_parallel.py::_run_unit()`'s own
    "never raise back to the pool - failures are reported in the return
    value" discipline).
    """
    job_id: Any
    result: Optional[Any] = None
    error: Optional[str] = None


def stage_grm_npy(grm_inv: "Optional[np.ndarray]", staging_dir: str, name: str) -> Optional[str]:
    """Writes `grm_inv` to `<staging_dir>/<name>.npy` and returns that
    path, or returns `None` unchanged when `grm_inv` is `None` (no GRM
    configured for this fit) - the caller's own job then carries
    `grm_path=None`, and the worker passes `grm_inv=None` straight through
    to the fit function, exactly reproducing "no GRM" behaviour.

    `staging_dir` is the CALLER's responsibility to create and clean up
    (ordinarily a `tempfile.mkdtemp()` living for exactly this one pool's
    own lifetime - see `AsremlProcessPool`'s own docstring) - this
    function only ever writes into it, never creates or removes it.
    """
    if grm_inv is None:
        return None
    os.makedirs(staging_dir, exist_ok=True)
    path = os.path.join(staging_dir, f'{name}.npy')
    np.save(path, np.asarray(grm_inv))
    return path


def _ensure_worker_r_ready(r_path: Optional[str], r_blas_threads: Optional[int]) -> None:
    """Initialises THIS worker process's own R/rpy2 session exactly once -
    see this module's own top-of-file docstring. Safe to call before every
    single job this worker ever handles; the module-level `_worker_r_ready`
    guard makes every call after the first one a no-op.
    """
    global _worker_r_ready
    if _worker_r_ready:
        return
    # Deferred import (I12: rpy2/R is an environment-dependent dependency,
    # never imported at module scope here, exactly like every other
    # ASReml-adjacent module in this codebase).
    from pipeline_utils import configure_r_environment, init_rpy2_conversion
    configure_r_environment(r_path, r_blas_threads=r_blas_threads)
    init_rpy2_conversion()
    _worker_r_ready = True


def _run_job(job: AsremlFitJob, r_path: Optional[str], r_blas_threads: Optional[int]) -> AsremlFitJobResult:
    """The pool's own worker-process entry point - a plain, module-level
    function (never a closure - see `AsremlFitJob.fit_kind`'s own
    docstring on why), so it is picklable and safe to hand to
    `ProcessPoolExecutor.submit()`. Never raises back to the pool - any
    exception, including one from R/rpy2 initialisation itself, is caught
    and reported in the returned `AsremlFitJobResult.error` instead
    (mirrors `intra_task_parallel.py::_run_unit()`'s identical discipline).
    """
    try:
        _ensure_worker_r_ready(r_path, r_blas_threads)
        # Deferred import: only needed once R/rpy2 is already initialised
        # in THIS process (see `_ensure_worker_r_ready()` just above) -
        # importing Preprocess.asreml_blup at module scope in the PARENT
        # process is harmless (it already guards its own rpy2 import
        # lazily), but importing it here, inside the worker, keeps this
        # module's own import graph honest about what actually needs R.
        from Preprocess import asreml_blup

        fit_fn: Callable = getattr(asreml_blup, job.fit_kind)
        kwargs = dict(job.kwargs)
        if job.grm_path is not None:
            kwargs['grm_inv'] = np.load(job.grm_path)
        result = fit_fn(job.raw_records, **kwargs)
        return AsremlFitJobResult(job_id=job.job_id, result=result)
    except Exception:
        return AsremlFitJobResult(job_id=job.job_id, error=traceback.format_exc())


class AsremlProcessPool:
    """A thin, explicit-lifetime wrapper around one `ProcessPoolExecutor`
    of ASReml-R fit workers - construct it once per `GP()` call (or not at
    all, when `n_processes <= 1` - see `resolve_pool_size()`), submit every
    hoistable fit for this run, collect the results, then close it.

    Usage
    -----
        pool = AsremlProcessPool(n_processes, r_path=..., r_blas_threads=...)
        try:
            pending = {pool.submit(job): job.job_id for job in jobs}
            for future in as_completed(pending):
                job_result = future.result()
                ...
        finally:
            pool.close()

    Or, for the common "submit everything, then wait for everything" case,
    `run_all()` below does exactly this in one call.
    """

    def __init__(self, n_processes: int, *, r_path: Optional[str] = None,
                 r_blas_threads: Optional[int] = None):
        if n_processes < 1:
            raise ValueError(f"AsremlProcessPool: n_processes must be >= 1, got {n_processes}")
        self.n_processes = n_processes
        self._r_path = r_path
        self._r_blas_threads = r_blas_threads
        self._executor = ProcessPoolExecutor(max_workers=n_processes, mp_context=_MP_CONTEXT)

    def submit(self, job: AsremlFitJob):
        """Submits ONE job; returns the `concurrent.futures.Future` (whose
        own `.result()` is an `AsremlFitJobResult`, never a raw exception -
        `_run_job()` never raises)."""
        return self._executor.submit(_run_job, job, self._r_path, self._r_blas_threads)

    def run_all(self, jobs: Sequence[AsremlFitJob]) -> Dict[Any, AsremlFitJobResult]:
        """Submits every job in `jobs`, waits for all of them, and returns
        a `{job_id: AsremlFitJobResult}` dict - the common case for R9-
        S2's own "before the task loop, submit every hoistable fit" usage.
        Order of completion does not matter; every result is matched back
        to its own `job_id`."""
        from concurrent.futures import as_completed
        futures = {self.submit(job): job.job_id for job in jobs}
        results: Dict[Any, AsremlFitJobResult] = {}
        for future in as_completed(futures):
            job_result = future.result()
            results[job_result.job_id] = job_result
        return results

    def close(self) -> None:
        """Shuts the pool down - idempotent (a second call is a harmless
        no-op, matching `ProcessPoolExecutor.shutdown()`'s own contract)."""
        self._executor.shutdown(wait=True, cancel_futures=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def resolve_pool_size(n_jobs: int, n_pending_fits: int,
                       license_max_processes: "Optional[int]" = None) -> int:
    """Design blueprint S2's own sizing formula for `ASREML_N_PROCESSES=
    'auto'`: `min(N_JOBS, n_pending_fits, ASREML_LICENSE_MAX_PROCESSES)` -
    never more workers than there are independent fits to run (a worker
    with nothing to do is pure overhead), never more than `N_JOBS` (R9's
    own "sized from N_JOBS only" rule - architecture doc SS14's `N_CPU_
    WORKERS`/`N_MODEL_WORKERS` play no part here), and never more than the
    site's own ASReml-R licence permits (V9.3 - unresolved in this
    environment; `license_max_processes=None`, the design blueprint's own
    documented default, means "no cap" and is NOT itself a verified
    statement that the site's licence permits `N_JOBS` concurrent R
    sessions - see the module docstring's own Known Limitations note).

    Returns at least 1 (never 0, even if `n_pending_fits` is 0 - the
    caller simply has nothing to submit in that case and never
    constructs a pool at all)."""
    candidates = [max(1, int(n_jobs)), max(1, int(n_pending_fits))]
    if license_max_processes is not None:
        candidates.append(max(1, int(license_max_processes)))
    return max(1, min(candidates))
