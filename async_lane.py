"""Asynchronous model lane (ver4-35, design blueprint R1.f).

Why this exists
---------------
A task fits its models strictly one after the other, so the CPU models
(rrBLUP, GBLUP, RF, ...) and a torch model (the bio-prior GAT) never work at
the same time: the CPUs idle while the GAT trains and the GPU idles while the
CPU models run (blueprint E1/D7). The *lane* is one persistent spawned worker
process that fits lane-capable models (``model_registry.LANE_CAPABLE``) while
the parent keeps fitting the CPU models. The parent collects each lane result
at that model's own turn in ``MODEL_RUN``, so Step 10 (accumulation) and Step 11
(ensembles) run unchanged and in the same order.

What stays exact
----------------
A serial run feeds the process-global random streams (torch / numpy /
``random``) to the models in ``MODEL_RUN`` order. The lane reproduces that
order when no model run in the PARENT draws from a stream a lane model also
draws from (:func:`schedule_lane` closes a *chain* at every such model):

* the first job of a chain carries the parent's stream states at the moment of
  submission (:func:`capture_rng`);
* the worker carries them from job to job;
* every job returns the states it ends with, and the parent restores them
  (:func:`restore_rng`) when it collects that job - so the parent's streams
  are exactly where the serial run would have left them before any later
  model, before Step 11 and before the next task.

CUDA generators are not handed off (a GPU fit is not bit-reproducible
anyway); ``ASYNC_LANE_RNG='per_job_seed'`` is the schedule-independent mode
for multi-GPU nodes: each job is reseeded from ``crc32(f"{task}|{model}")``
and jobs spread over all lane slots. It is deterministic but NOT identical to
a lane-off run, and is disclosed as such.

Failure handling: a job exception (or a dead worker) is never fatal. The parent
cancels what is still queued, keeps the stream states of the last good job, and
re-runs the failed model and the rest of its chain inline through the unchanged
main-dispatch branch (with a NOTE). A dead worker also switches the lane off
for the rest of the run.

Nothing in here touches a result accumulator (I8): the lane only returns the
same result dict ``ModelTrialRunner`` returns, and the parent writes it into
the accumulators exactly as it does for an inline fit.
"""

import concurrent.futures as cf
import hashlib
import multiprocessing
import os
import pickle
import random
import time
import zlib
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from pipeline_utils import current_stage_timer, timed_stage

#: Every global stream the lane knows how to hand off.
ALL_STREAMS = frozenset({'torch', 'numpy', 'random'})

#: Keys the worker adds to a result dict; the parent removes them before use.
_PRIVATE_KEYS = ('_secs', '_stages', '_rng_out', '_rng_census')


# ---------------------------------------------------------------------------
# Random-stream hand-off
# ---------------------------------------------------------------------------

def capture_rng(streams: Iterable[str]) -> Dict[str, object]:
    """Snapshot of the named global random streams of THIS process (picklable
    plain data). ``'torch'`` is the CPU generator (as ``bytes``), ``'numpy'``
    the legacy global ``RandomState``, ``'random'`` the stdlib generator. A
    stream whose library is not installed is left out."""
    wanted = set(streams)
    states: Dict[str, object] = {}
    if 'torch' in wanted:
        try:
            import torch
            states['torch'] = bytes(torch.get_rng_state().numpy().tobytes())
        except ImportError:
            pass
    if 'numpy' in wanted:
        states['numpy'] = np.random.get_state()
    if 'random' in wanted:
        states['random'] = random.getstate()
    return states


def restore_rng(states: Mapping[str, object]) -> None:
    """Install snapshots taken by :func:`capture_rng` (only the streams present
    in `states`)."""
    if 'torch' in states:
        import torch
        torch.set_rng_state(torch.frombuffer(bytearray(states['torch']), dtype=torch.uint8))
    if 'numpy' in states:
        np.random.set_state(states['numpy'])
    if 'random' in states:
        random.setstate(states['random'])


def rng_digests() -> Dict[str, str]:
    """Short digest of each global stream's current state - used to notice
    that a lane job drew from a stream the model registry did not declare
    (blueprint check P6): equal digests before and after mean "untouched"."""
    digests: Dict[str, str] = {}
    for stream, state in capture_rng(ALL_STREAMS).items():
        if stream == 'torch':
            raw = state
        elif stream == 'numpy':
            raw = state[1].tobytes() + repr(state[2:]).encode('utf-8')
        else:
            raw = repr(state).encode('utf-8')
        digests[stream] = hashlib.blake2b(raw, digest_size=8).hexdigest()
    return digests


def lane_job_seed(task_index: int, model_run: str) -> int:
    """Seed of one ``'per_job_seed'`` job: a pure function of the task index
    and the model's name, so it does not depend on which slot ran the job or
    in which order jobs finished (I9)."""
    return zlib.crc32(f"{int(task_index)}|{model_run}".encode('utf-8')) & 0xFFFFFFFF


def payload_bytes(obj, _depth: int = 0) -> int:
    """Approximate pickled size of `obj`: DataFrames, ndarrays and whatever
    dicts / lists / tuples hold them. A cheap upper-level estimate used only
    to keep a lane job under ``ASYNC_LANE_MAX_PAYLOAD_MB``."""
    if _depth > 6 or obj is None:
        return 0
    if hasattr(obj, 'memory_usage') and hasattr(obj, 'shape'):      # pandas
        try:
            usage = obj.memory_usage(deep=False)
            return int(usage.sum() if hasattr(usage, 'sum') else usage)
        except (TypeError, ValueError):
            return 0
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj, Mapping):
        return sum(payload_bytes(v, _depth + 1) for v in obj.values())
    if isinstance(obj, (list, tuple, set, frozenset)):
        return sum(payload_bytes(v, _depth + 1) for v in obj)
    return 0


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

@dataclass
class LaneChain:
    """One chain of lane models that run in the lane worker back to back.

    ``release_index``: the ``MODEL_RUN`` index at which the parent submits it
    (just after the last parent-run model that draws from a shared stream);
    ``overlap``: the parent-run models that run while the chain does (the
    reason the chain is worth running in the lane at all)."""
    models: List[str]
    release_index: int
    overlap: List[str] = field(default_factory=list)


def _streams_of(name: str, rng_streams: Mapping[str, Optional[frozenset]]) -> frozenset:
    """Declared streams of `name`; an unknown model counts as drawing from all."""
    declared = rng_streams.get(name)
    return ALL_STREAMS if declared is None else frozenset(declared)


def schedule_lane(model_run: Sequence[str], eligible: Set[str],
                  rng_streams: Mapping[str, Optional[frozenset]],
                  per_job_seed: bool = False, parallel_slots: int = 1) -> List[LaneChain]:
    """Split the lane-eligible models of `model_run` into chains and say when
    each is released.

    ``'exact'`` mode (default): a parent-run model whose declared streams
    intersect the streams the lane models draw from closes the current chain
    (the next lane model starts a new one, released only after that model has
    finished and its stream states are the parent's). An unknown model
    (``rng_streams`` value ``None`` or missing) is treated as drawing from every
    stream. ``per_job_seed`` mode: every lane model is its own chain, released
    at index 0 (jobs are reseeded, so no ordering constraint exists).

    A chain with nothing to overlap (no parent-run model runs while it does) is
    dropped - running it in the lane would only add a process and a payload -
    unless ``per_job_seed`` with several slots lets lane jobs overlap each
    other. Returns the chains in ``MODEL_RUN`` order."""
    names = [m for m in model_run if m != 'ensemble']
    lane_models = [m for m in names if m in eligible]
    if not lane_models:
        return []
    lane_streams = frozenset().union(*(_streams_of(m, rng_streams) for m in lane_models))

    raw: List[Tuple[List[str], int]] = []
    if per_job_seed:
        raw = [([m], 0) for m in lane_models]
    else:
        current: List[str] = []
        release = 0
        for index, name in enumerate(names):
            if name in eligible:
                current.append(name)
                continue
            if _streams_of(name, rng_streams) & lane_streams:
                if current:
                    raw.append((current, release))
                    current = []
                release = index + 1
        if current:
            raw.append((current, release))

    chains: List[LaneChain] = []
    for models, release in raw:
        tail = names.index(models[-1])
        overlap = [m for m in names[release:tail] if m not in eligible]
        keep = bool(overlap) or (per_job_seed and parallel_slots > 1 and len(lane_models) > 1)
        if keep:
            chains.append(LaneChain(models=models, release_index=release, overlap=overlap))
    return chains


def plan_lane_chains(model_run: Sequence[str], eligible: Set[str],
                     rng_streams: Mapping[str, Optional[frozenset]]) -> List[List[str]]:
    """The chains of :func:`schedule_lane` (``'exact'`` mode) as plain lists of
    model names."""
    return [chain.models for chain in schedule_lane(model_run, eligible, rng_streams)]


# ---------------------------------------------------------------------------
# Worker side (module-level: spawn must be able to import and pickle all of it)
# ---------------------------------------------------------------------------

_WORKER_STATE: Dict[str, object] = {'pools': {}}


@dataclass
class LaneJob:
    """One model fit for the lane worker - plain data only (I13).

    model_run     : the name the result is reported under (``MODEL_RUN[jj]``)
    base_model    : the model's base name
    params        : the model's parameter list, already resolved per phenotype
    runner_init   : keyword arguments of ``ModelTrialRunner`` (result_name,
                    r_path, compute_resources, genotype_format,
                    sample_weight_ctx); ``bio_prior_ctx`` comes from the pool
    frames_ref    : ``(token, key)`` of a pool previously sent with
                    :meth:`ModelLane.put_pool`, or None
    rng_in        : stream states to install before the fit (first job of an
                    ``'exact'`` chain), or None to continue from the worker's
    want_rng_out  : return the stream states after the fit
    streams       : streams the model is declared to draw from
    seed          : ``'per_job_seed'`` seed, or None
    """
    model_run: str
    base_model: str
    params: list
    runner_init: dict
    frames_ref: Optional[Tuple[str, str]] = None
    rng_in: Optional[dict] = None
    want_rng_out: bool = False
    streams: Tuple[str, ...] = ()
    seed: Optional[int] = None


def _lane_worker_init(init_args) -> None:
    """Initialiser of the lane worker process. Runs the SAME restore steps as a
    hyperparameter-tuning trial worker (R environment, rpy2, the resolved
    compute resources, the GPU semaphore and - when
    ``ACCELERATION['worker_thread_limits']`` is on - the native thread-pool
    cap; see ``genomic_prediction._trial_worker_init``). Imports
    ``genomic_prediction`` lazily: it imports this module, so a module-level
    import here would be circular."""
    import genomic_prediction
    genomic_prediction._trial_worker_init(*init_args)
    print(f"[lane] worker PID {os.getpid()} ready.")


def _lane_put_pool(token: str, key: str, payload) -> bool:
    """Store a shared payload (frames / bio-prior context) in the worker."""
    _WORKER_STATE['pools'][(token, key)] = payload
    return True


def _lane_drop_pool(token: str) -> bool:
    """Forget every payload stored under `token` (a finished task)."""
    pools = _WORKER_STATE['pools']
    for ref in [ref for ref in pools if ref[0] == token]:
        del pools[ref]
    return True


def _lane_run_job(job: LaneJob) -> dict:
    """Fit one model in the worker through ``ModelTrialRunner`` - the canonical
    "(model, params, frames) -> result dict" path - and return its result dict
    plus ``'_secs'`` (wall seconds), ``'_stages'`` (the fit's own stage
    seconds, e.g. ``bio:train``) and, when asked, ``'_rng_out'`` (the stream
    states afterwards) and ``'_rng_census'`` (which global streams the fit
    changed, for the parent's declared-stream check)."""
    import genomic_prediction
    from pipeline_utils import StageTimer, set_current_stage_timer

    payload = None
    if job.frames_ref is not None:
        payload = _WORKER_STATE['pools'].get(tuple(job.frames_ref))
        if payload is None:
            raise RuntimeError(f"lane pool {job.frames_ref!r} is not present in the worker "
                               f"(it was dropped or never sent)")
    frames = payload.get('frames') if isinstance(payload, dict) else None
    bio_prior_ctx = payload.get('bio_prior_ctx') if isinstance(payload, dict) else None

    if job.rng_in is not None:
        restore_rng(job.rng_in)
    if job.seed is not None:
        random.seed(job.seed)
        np.random.seed(job.seed % (2 ** 32))
        try:
            import torch
            torch.manual_seed(job.seed)
        except ImportError:
            pass
    before = rng_digests()

    runner = genomic_prediction.ModelTrialRunner(bio_prior_ctx=bio_prior_ctx, **job.runner_init)
    train, valid, test = frames if frames is not None else (None, None, None)
    timer = StageTimer()
    set_current_stage_timer(timer)
    started = time.perf_counter()
    try:
        # A lane job is an untuned MAIN-LOOP fit, i.e. a final fit: GAT models may run their
        # seed ensemble here exactly as they do in the in-process main loop
        # (models/gat_stability.py, F2).
        from models.gat_stability import final_fit_role
        with final_fit_role():
            result = runner(job.model_run, train, valid, test, list(job.params),
                            job.runner_init['result_name'])
    finally:
        set_current_stage_timer(None)
    out = dict(result)
    out['_secs'] = time.perf_counter() - started
    out['_stages'] = timer.snapshot()
    after = rng_digests()
    out['_rng_census'] = {stream: before.get(stream) != after.get(stream) for stream in after}
    if job.want_rng_out:
        out['_rng_out'] = capture_rng(job.streams)
    return out


# ---------------------------------------------------------------------------
# Parent side
# ---------------------------------------------------------------------------

class ModelLane:
    """Persistent lane: ``n_slots`` single-worker spawn executors, opened
    lazily on first use and kept for the whole ``GP()`` call, so the worker's
    imports and its CUDA context are paid once.

    ``init_args`` is the plain-data tuple of ``genomic_prediction.
    _trial_worker_init`` (result name, R path, resources dict, GPU semaphore).
    """

    def __init__(self, init_args, n_slots: int = 1):
        self._init_args = init_args
        self._n_slots = max(1, int(n_slots))
        self._executors: List[Optional[cf.ProcessPoolExecutor]] = [None] * self._n_slots
        self._disabled_reason: Optional[str] = None
        self._closed = False

    @property
    def n_slots(self) -> int:
        return self._n_slots

    @property
    def usable(self) -> bool:
        """False once the lane was closed or marked broken."""
        return not self._closed and self._disabled_reason is None

    @property
    def disabled_reason(self) -> Optional[str]:
        return self._disabled_reason

    def disable(self, reason: str) -> None:
        """Stop using the lane for the rest of the run (a dead worker)."""
        if self._disabled_reason is None:
            self._disabled_reason = reason
            print(f"[lane] NOTE: the asynchronous model lane is switched off for the rest of this "
                  f"run ({reason}) - every model runs inline from now on.")

    def _executor(self, slot: int) -> cf.ProcessPoolExecutor:
        if self._closed:
            raise RuntimeError("the model lane is closed")
        executor = self._executors[slot]
        if executor is None:
            executor = cf.ProcessPoolExecutor(
                max_workers=1, mp_context=multiprocessing.get_context('spawn'),
                initializer=_lane_worker_init, initargs=(self._init_args,))
            self._executors[slot] = executor
        return executor

    def put_pool(self, token: str, key: str, payload) -> List[cf.Future]:
        """Send `payload` to every slot, once, under ``(token, key)``."""
        return [self._executor(slot).submit(_lane_put_pool, token, key, payload)
                for slot in range(self._n_slots)]

    def drop_pool(self, token: str) -> None:
        """Ask every opened worker to forget `token` (fire and forget)."""
        for executor in self._executors:
            if executor is not None:
                try:
                    executor.submit(_lane_drop_pool, token)
                except (RuntimeError, BrokenProcessPool):
                    pass

    def submit_job(self, job: LaneJob, slot: int = 0) -> cf.Future:
        """Queue `job` FIFO on slot `slot`; returns its future."""
        return self._executor(slot % self._n_slots).submit(_lane_run_job, job)

    def close(self) -> None:
        """Shut the workers down without waiting for running jobs; queued
        jobs are cancelled."""
        self._closed = True
        for index, executor in enumerate(self._executors):
            if executor is not None:
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except (RuntimeError, OSError) as exc:
                    print(f"[lane] NOTE: lane slot {index} did not shut down cleanly "
                          f"({type(exc).__name__}: {exc}).")
                self._executors[index] = None


@dataclass
class _JobState:
    job: Optional[LaneJob]
    future: Optional[cf.Future]
    chain: int
    consumed: bool = False


#: Task lanes with futures outstanding in this process - so a task that raises
#: can cancel them from the rollback handler without holding the object.
_ACTIVE_TASK_LANES: List['TaskLane'] = []


def cancel_active_task_lanes() -> None:
    """Cancel every queued future of every task lane still open in this
    process (running ones are simply never collected). Called by the task
    loop's rollback handler (I8: nothing from the lane reaches an accumulator
    until the parent writes it)."""
    for task_lane in list(_ACTIVE_TASK_LANES):
        task_lane.abort()


class TaskLane:
    """One task's lane plan: releases chains at their release index, hands the
    parent each result at that model's turn, and falls back to inline on any
    failure.

    Parameters
    ----------
    lane          : the persistent :class:`ModelLane`
    task_label    : text for log lines (``'Task 3/12'``)
    chains        : :func:`schedule_lane` output
    build_job     : ``build_job(model_run) -> (LaneJob, pool_key, pool_payload)``;
                    ``pool_payload`` is sent once per ``pool_key`` per task
    token         : unique task token for the worker-side pools
    rng_mode      : ``'exact'`` or ``'per_job_seed'``
    streams_of    : ``model_run -> declared streams`` (restore / census)
    task_index    : the task's row index (``per_job_seed`` seeding)
    """

    def __init__(self, lane: ModelLane, task_label: str, chains: Sequence[LaneChain],
                 build_job: Callable[[str], Tuple[LaneJob, str, object]], token: str,
                 rng_mode: str, streams_of: Callable[[str], Iterable[str]], task_index: int):
        self._lane = lane
        self._label = task_label
        self._chains = list(chains)
        self._build_job = build_job
        self._token = token
        self._rng_mode = rng_mode
        self._streams_of = streams_of
        self._task_index = int(task_index)
        self._released = [False] * len(self._chains)
        self._failed = [False] * len(self._chains)
        self._jobs: Dict[str, _JobState] = {}
        self._pools_sent: Set[str] = set()
        self._member_chain = {m: c for c, chain in enumerate(self._chains) for m in chain.models}
        self._closed = False
        _ACTIVE_TASK_LANES.append(self)

    def covers(self, model_run: str) -> bool:
        """True while `model_run` belongs to a chain that has not failed."""
        chain = self._member_chain.get(model_run)
        return chain is not None and not self._failed[chain] and self._lane.usable

    def release_due(self, index: int) -> None:
        """Submit every not-yet-released chain whose release index is `index`
        or earlier (call at the top of each ``MODEL_RUN`` iteration)."""
        for c, chain in enumerate(self._chains):
            if self._released[c] or chain.release_index > index:
                continue
            self._released[c] = True
            if not self._lane.usable:
                self._failed[c] = True
                continue
            self._submit_chain(c)

    def _submit_chain(self, c: int) -> None:
        chain = self._chains[c]
        exact = self._rng_mode == 'exact'
        streams_union = sorted(frozenset().union(*(frozenset(self._streams_of(m))
                                                   for m in chain.models)))
        try:
            for position, model_run in enumerate(chain.models):
                job, pool_key, pool_payload = self._build_job(model_run)
                if pool_key is not None and pool_key not in self._pools_sent:
                    self._lane.put_pool(self._token, pool_key, pool_payload)
                    self._pools_sent.add(pool_key)
                if pool_key is not None:
                    job.frames_ref = (self._token, pool_key)
                job.streams = tuple(sorted(frozenset(self._streams_of(model_run))))
                job.want_rng_out = exact
                if exact and position == 0:
                    job.rng_in = capture_rng(streams_union)
                if not exact:
                    job.seed = lane_job_seed(self._task_index, model_run)
                slot = c % self._lane.n_slots if not exact else 0
                self._jobs[model_run] = _JobState(job, self._lane.submit_job(job, slot), c)
        except Exception as exc:  # noqa: BLE001 - whatever stops a submit must degrade to inline
            # (a missing HPARAMETERS entry, an oversized payload, a payload
            # that cannot be pickled, a worker that cannot start): the inline
            # branch then raises or succeeds exactly as it always did.
            self._fail_chain(c, f"could not submit ({type(exc).__name__}: {exc})",
                             broken=isinstance(exc, BrokenProcessPool))

    def _fail_chain(self, c: int, reason: str, broken: bool = False) -> None:
        """Mark chain `c` failed: cancel what is still queued, say so, and let
        the parent run its remaining models inline."""
        if self._failed[c]:
            return
        self._failed[c] = True
        remaining = [m for m in self._chains[c].models
                     if m in self._jobs and not self._jobs[m].consumed]
        for model_run in remaining:
            future = self._jobs[model_run].future
            if future is not None:
                future.cancel()
        print(f"[lane] NOTE: {self._label}: lane chain {self._chains[c].models} failed "
              f"({reason}) - the failed model and the rest of its chain run inline on the unchanged legacy "
              f"path with the random streams of the last good lane job restored; results are "
              f"unaffected, only the overlap is lost.")
        if broken:
            self._lane.disable(reason)

    def consume(self, model_run: str) -> Optional[dict]:
        """Wait for `model_run`'s lane result and return its result dict (the
        ``ModelTrialRunner`` dict, private keys removed), or ``None`` when the
        model must be fitted inline (its chain failed, was never released, or
        this very job failed). On success the parent's random streams are
        advanced to where the lane left them (``'exact'`` mode)."""
        c = self._member_chain.get(model_run)
        state = self._jobs.get(model_run)
        if c is None or state is None or self._failed[c] or state.future is None:
            return None
        waited_from = time.perf_counter()
        try:
            with timed_stage('lane_wait'):
                result = state.future.result()
        except BrokenProcessPool as exc:
            self._fail_chain(c, f"the lane worker died: {exc!r}", broken=True)
            return None
        except cf.CancelledError:
            self._fail_chain(c, "the job was cancelled")
            return None
        except Exception as exc:  # noqa: BLE001 - any model error must degrade to the inline path
            self._fail_chain(c, f"{type(exc).__name__}: {exc}")
            return None
        state.consumed = True
        waited = time.perf_counter() - waited_from

        if self._rng_mode == 'exact' and '_rng_out' in result:
            restore_rng(result['_rng_out'])
        self._check_census(model_run, result.get('_rng_census') or {})
        timer = current_stage_timer()
        stages = result.get('_stages') or {}
        detail = " ".join(f"{k}={v:.1f}s" for k, v in sorted(stages.items()))
        secs = float(result.get('_secs', 0.0))
        if timer is not None:
            timer.info(f"lane:{model_run}", f"{secs:.1f}s" + (f" [{detail}]" if detail else ""))
        print(f"[lane] {self._label}: {model_run} fitted in the lane in {secs:.1f}s "
              f"(parent waited {waited:.1f}s).")
        for key in _PRIVATE_KEYS:
            result.pop(key, None)
        return result

    def _check_census(self, model_run: str, census: Mapping[str, bool]) -> None:
        declared = frozenset(self._streams_of(model_run))
        undeclared = sorted(s for s, changed in census.items() if changed and s not in declared)
        if undeclared:
            print(f"[lane] WARNING: {self._label}: {model_run} drew from the global random "
                  f"stream(s) {undeclared} that model_registry.GLOBAL_RNG_STREAMS does not declare "
                  f"for it - a lane run may then differ from a lane-off run. Add the stream(s) to "
                  f"GLOBAL_RNG_STREAMS (blueprint check P6).")

    def abort(self) -> None:
        """Cancel everything still queued, ask the worker to forget this
        task's data and stop tracking it (a task that raised, or the normal
        end of the task's loop). Running jobs are never collected."""
        if self._closed:
            return
        for state in self._jobs.values():
            if state.future is not None and not state.consumed:
                state.future.cancel()
        if self._pools_sent and self._lane.usable:
            self._lane.drop_pool(self._token)
        self._detach()

    def finish(self) -> None:
        """End of the task's Step 9 loop: same as :meth:`abort` - any job that
        was never collected is cancelled, the worker-side pools are dropped."""
        self.abort()

    def _detach(self) -> None:
        if not self._closed:
            self._closed = True
            if self in _ACTIVE_TASK_LANES:
                _ACTIVE_TASK_LANES.remove(self)


# ---------------------------------------------------------------------------
# One lane per GP() call
# ---------------------------------------------------------------------------

_LANE_CACHE: Dict[Tuple, ModelLane] = {}


def acquire_lane(init_args, n_slots: int = 1) -> ModelLane:
    """The persistent lane for these (plain-data, hashable-by-pickle) worker
    init arguments, created on first request. The key is the pickled init
    tuple so a changed resource dict or R path gets its own worker."""
    try:
        key = (hashlib.blake2b(pickle.dumps(init_args), digest_size=16).hexdigest(), int(n_slots))
    except (pickle.PicklingError, TypeError, AttributeError):
        key = (repr(init_args), int(n_slots))
    lane = _LANE_CACHE.get(key)
    if lane is None or not lane.usable:
        lane = ModelLane(init_args, n_slots=n_slots)
        _LANE_CACHE[key] = lane
    return lane


def release_lanes() -> None:
    """Close every cached lane (outermost ``GP()`` exit)."""
    for key in list(_LANE_CACHE):
        _LANE_CACHE.pop(key).close()
