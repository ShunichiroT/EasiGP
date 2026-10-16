"""
EasiGP - resource telemetry (ver4-56, blueprint R1).

Purpose
-------
Record what each model, task and job ACTUALLY used (seconds, peak memory,
data size, hyper-parameters, allocation) as structured JSON lines, so that
``resource_calibration.py`` can fit a per-model cost model and
``resource_profiles.estimate_resources_calibrated()`` can suggest HPC
resources from measurements instead of provisional constants.

Why a new module (blueprint R1): the only measurement before this was the
``[timing]`` log line (``pipeline_utils.StageTimer.line``): seconds only, free
text, no memory, no data size, no hyper-parameters. That cannot calibrate
anything (defects E1-E9 in the blueprint, section 0.1).

Contract
--------
* LOG-ONLY. Nothing in this module changes a result, a random stream, a
  checkpoint or a file the pipeline reads (AC1.6). It uses ``secrets`` (OS
  entropy) for its file name so the global ``random`` state is untouched.
* NEVER FATAL. Every public function swallows its own exceptions; a failure
  is reported once per process as ``[telemetry] what - why - what next`` and
  telemetry then goes quiet. A broken disk must not fail a 40-hour run.
* One file per process per ``configure()`` call:
  ``<dir>/<host>_<pid>_<rand8>.jsonl`` (I10: no shared file, so concurrent
  worker processes never interleave lines).
* No EasiGP imports (I2). ``pipeline_utils`` owns the directory environment
  variable helpers and imports the two constants below.

Record schema v1 (one JSON object per line, ``"v": 1``)
-------------------------------------------------------
Every record also carries ``ts`` (unix seconds) and ``pool_sig`` (ver4-56 addition
to the blueprint schema: lets the calibration separate runs whose LD / RF-filter /
genotype-format switches change the per-task pool cost).
``kind='model'`` - one per (task, model): the common header (``run_id``,
``engine``, ``machine_key``, ``host``, ``job``, ``task``) plus ``model``,
``base``, ``sig_inputs`` fields (``params``, ``hp_tune``, ``tuned``,
``device``), sizes (``n_train``, ``n_valid``, ``n_test``, ``p``, ``p_raw``),
``seconds``, ``gpu_wait_s``, ``rss_tree_peak_gb``, ``rss_self_peak_gb``,
``rss_baseline_gb``, ``gpu_peak_gb``, ``workers``, ``n_jobs``,
``blas_threads``, ``torch_threads`` and ``status``.
``kind='task'`` - header plus ``n_task``, ``p_raw``, ``stages`` (the
``StageTimer.snapshot()``), ``total_s``, ``rss_tree_peak_gb``, ``status``.
``kind='job'`` - header plus ``wall_s``, ``startup_s``, ``tasks_total_s``,
``tail_s`` (post-loop: ensemble finalisation, saving), ``cgroup_peak_gb``,
``ru_maxrss_children_gb``, ``n_tasks_done``.
"""

import json
import math
import os
import platform
import secrets
import socket
import sys
import threading
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCHEMA_VERSION = 1

# Same string as ``pipeline_utils.TELEMETRY_DIR_ENV`` (that module imports
# these two constants from here, so there is one definition).
TELEMETRY_DIR_ENV = 'EASIGP_TELEMETRY_DIR'
# Sub-folder of ``Result/<name>/logs/`` that holds the JSONL files.
TELEMETRY_SUBDIR = 'resource_telemetry'
# Set to '1' by the pilot job (run_step1_batch.py --pilot) so its records are
# tagged ``task.pilot = true``.
TELEMETRY_PILOT_ENV = 'EASIGP_TELEMETRY_PILOT'

_GB = 1024.0 ** 3
_MAX_LIST_ITEMS = 256
_MAX_REPR_CHARS = 200


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------

def _num(value: Any, digits: int = 4) -> Optional[float]:
    """A finite float rounded to ``digits`` places, else None (JSON-safe)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _jsonable(value: Any, depth: int = 0) -> Any:
    """Convert ``value`` to something ``json.dumps`` accepts, deterministically.

    NumPy scalars become Python numbers, tuples/lists become lists (very long
    ones are replaced by a ``<list len=N>`` marker), dict keys become strings,
    non-finite floats become None and anything unknown becomes a short
    ``repr``. Never raises.
    """
    try:
        if value is None or isinstance(value, (bool, int, str)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if depth > 6:
            return f"<{type(value).__name__}>"
        if isinstance(value, Mapping):
            return {str(k): _jsonable(v, depth + 1) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            if len(value) > _MAX_LIST_ITEMS:
                return f"<list len={len(value)}>"
            return [_jsonable(v, depth + 1) for v in value]
        if hasattr(value, 'item') and callable(value.item):
            try:
                return _jsonable(value.item(), depth + 1)
            except (ValueError, TypeError):
                pass
        return repr(value)[:_MAX_REPR_CHARS]
    except Exception:
        return '<unserialisable>'


def _read_proc_rss_gb(pid: str = 'self') -> Optional[float]:
    """Resident set size of one process in GB from ``/proc`` (None if absent)."""
    try:
        with open(f'/proc/{pid}/statm', 'r') as handle:
            fields = handle.read().split()
        pages = int(fields[1])
        return pages * os.sysconf('SC_PAGE_SIZE') / _GB
    except (OSError, ValueError, IndexError):
        return None


def _read_proc_vmhwm_gb() -> Optional[float]:
    """This process's exact peak resident size (``VmHWM``) in GB, or None."""
    try:
        with open('/proc/self/status', 'r') as handle:
            for line in handle:
                if line.startswith('VmHWM:'):
                    return int(line.split()[1]) / (1024.0 ** 2)
    except (OSError, ValueError, IndexError):
        return None
    return None


def _proc_children_map() -> Dict[int, List[int]]:
    """``{ppid: [pid, ...]}`` built from ``/proc/<pid>/stat`` (fallback for a
    missing psutil). Linux only; empty on any failure."""
    children: Dict[int, List[int]] = {}
    try:
        for name in os.listdir('/proc'):
            if not name.isdigit():
                continue
            try:
                with open(f'/proc/{name}/stat', 'r') as handle:
                    text = handle.read()
                # The command name is in parentheses and may contain spaces.
                after = text[text.rindex(')') + 2:].split()
                children.setdefault(int(after[1]), []).append(int(name))
            except (OSError, ValueError, IndexError):
                continue
    except OSError:
        return {}
    return children


def machine_key() -> str:
    """A 10-character id of this machine class: sha1 of
    ``cpu model name | logical cores | gpu name or '-'``.

    The GPU name comes from ``/proc/driver/nvidia`` (not from torch), so the
    key is the same whether or not torch happens to be imported yet.
    Same CPU model + core count + GPU => same key, on any node.
    """
    import hashlib
    cpu_name = ''
    try:
        with open('/proc/cpuinfo', 'r') as handle:
            for line in handle:
                if line.lower().startswith('model name'):
                    cpu_name = line.split(':', 1)[1].strip()
                    break
    except OSError:
        cpu_name = ''
    if not cpu_name:
        cpu_name = platform.processor() or platform.machine() or 'unknown-cpu'
    gpu_name = '-'
    try:
        gpu_dir = '/proc/driver/nvidia/gpus'
        for entry in sorted(os.listdir(gpu_dir)):
            with open(os.path.join(gpu_dir, entry, 'information'), 'r') as handle:
                for line in handle:
                    if line.startswith('Model:'):
                        gpu_name = line.split(':', 1)[1].strip()
                        break
            if gpu_name != '-':
                break
    except OSError:
        gpu_name = '-'
    text = f"{cpu_name}|{os.cpu_count() or 0}|{gpu_name}"
    return hashlib.sha1(text.encode('utf-8')).hexdigest()[:10]


def _env_int(*names: str) -> Optional[int]:
    for name in names:
        raw = os.environ.get(name)
        if raw is None or raw == '':
            continue
        try:
            return int(float(raw))
        except ValueError:
            continue
    return None


def alloc_from_env() -> Dict[str, Any]:
    """What the scheduler allocated to this job, read from its environment.

    Returns ``{'sched', 'job_id', 'array_idx', 'alloc_ncpus', 'alloc_mem_gb',
    'alloc_ngpus'}``; unknown fields are None. ``sched`` is ``'pbs'``,
    ``'slurm'`` or ``'local'``.
    """
    info: Dict[str, Any] = {'sched': 'local', 'job_id': None, 'array_idx': None,
                            'alloc_ncpus': None, 'alloc_mem_gb': None, 'alloc_ngpus': None}
    try:
        if os.environ.get('SLURM_JOB_ID'):
            info['sched'] = 'slurm'
            info['job_id'] = os.environ.get('SLURM_JOB_ID')
            info['array_idx'] = os.environ.get('SLURM_ARRAY_TASK_ID')
            info['alloc_ncpus'] = _env_int('SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE')
            mem_node_mb = _env_int('SLURM_MEM_PER_NODE')
            mem_cpu_mb = _env_int('SLURM_MEM_PER_CPU')
            if mem_node_mb is not None:
                info['alloc_mem_gb'] = _num(mem_node_mb / 1024.0)
            elif mem_cpu_mb is not None and info['alloc_ncpus']:
                info['alloc_mem_gb'] = _num(mem_cpu_mb * info['alloc_ncpus'] / 1024.0)
            info['alloc_ngpus'] = _env_int('SLURM_GPUS_ON_NODE', 'SLURM_GPUS')
        elif os.environ.get('PBS_JOBID'):
            info['sched'] = 'pbs'
            info['job_id'] = os.environ.get('PBS_JOBID')
            info['array_idx'] = os.environ.get('PBS_ARRAY_INDEX')
            info['alloc_ncpus'] = _env_int('PBS_NCPUS', 'NCPUS')
            vmem_bytes = _env_int('PBS_VMEM')
            if vmem_bytes is not None:
                info['alloc_mem_gb'] = _num(vmem_bytes / _GB)
            info['alloc_ngpus'] = _env_int('PBS_NGPUS')
        else:
            try:
                info['alloc_ncpus'] = len(os.sched_getaffinity(0))
            except (AttributeError, OSError):
                info['alloc_ncpus'] = os.cpu_count()
    except Exception:
        pass
    return info


def _cgroup_peak_gb() -> Optional[float]:
    """Best-effort peak memory of this job's cgroup (blueprint P3).

    cgroup v2 (Slurm/Bunya): ``memory.peak``; cgroup v1 (PBS/Gadi):
    ``memory.max_usage_in_bytes``. The path comes from ``/proc/self/cgroup``.
    Returns None when neither file can be read.
    """
    try:
        with open('/proc/self/cgroup', 'r') as handle:
            lines = handle.read().splitlines()
    except OSError:
        return None
    candidates: List[str] = []
    for line in lines:
        parts = line.split(':', 2)
        if len(parts) != 3:
            continue
        _, controllers, path = parts
        if controllers == '':
            candidates.append(os.path.join('/sys/fs/cgroup', path.lstrip('/'), 'memory.peak'))
        elif 'memory' in controllers.split(','):
            candidates.append(os.path.join('/sys/fs/cgroup/memory', path.lstrip('/'),
                                           'memory.max_usage_in_bytes'))
    candidates.append('/sys/fs/cgroup/memory.peak')
    candidates.append('/sys/fs/cgroup/memory/memory.max_usage_in_bytes')
    for path in candidates:
        try:
            with open(path, 'r') as handle:
                value = int(handle.read().strip())
            if value > 0:
                return value / _GB
        except (OSError, ValueError):
            continue
    return None


def _children_maxrss_gb() -> Optional[float]:
    """Largest resident set of any waited-for child process (``ru_maxrss``)."""
    try:
        import resource
        kilobytes = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        return kilobytes / (1024.0 ** 2) if sys.platform.startswith('linux') else kilobytes / _GB
    except (ImportError, OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# The sampler thread
# ---------------------------------------------------------------------------

class _Sampler:
    """Polls the process-tree and own-process RSS every ``interval_s`` seconds
    and keeps running maxima for the open task and the open model.

    The tree is this process plus every descendant (loky/spawn workers, R
    helpers, PLINK). Uses psutil when installed (blueprint I12), else walks
    ``/proc``. RSS can double count pages shared between processes; that errs
    toward over-requesting memory, which is the safe direction.
    """

    def __init__(self, interval_s: float) -> None:
        self.interval_s = max(0.05, float(interval_s))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._psutil = None
        self._psutil_tried = False
        self.task_tree_peak = 0.0
        self.task_self_peak = 0.0
        self.model_tree_peak = 0.0
        self.model_self_peak = 0.0
        self.tree_supported = True

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name='easigp-telemetry-sampler',
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    def reset_task(self) -> None:
        with self._lock:
            self.task_tree_peak = 0.0
            self.task_self_peak = 0.0

    def reset_model(self) -> None:
        with self._lock:
            self.model_tree_peak = 0.0
            self.model_self_peak = 0.0

    def current(self) -> Tuple[Optional[float], Optional[float]]:
        """``(tree_rss_gb, self_rss_gb)`` right now; either may be None."""
        own = _read_proc_rss_gb('self')
        tree = self._tree_rss_gb(own)
        return tree, own

    def tick(self) -> None:
        """One synchronous sample (also called at model/task boundaries so a
        short model is never missed)."""
        tree, own = self.current()
        with self._lock:
            if own is not None:
                self.task_self_peak = max(self.task_self_peak, own)
                self.model_self_peak = max(self.model_self_peak, own)
            if tree is not None:
                self.task_tree_peak = max(self.task_tree_peak, tree)
                self.model_tree_peak = max(self.model_tree_peak, tree)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.tick()
            except Exception:
                # A sampling error must never reach the pipeline; the next
                # tick simply tries again.
                continue

    def _tree_rss_gb(self, own: Optional[float]) -> Optional[float]:
        if not self._psutil_tried:
            self._psutil_tried = True
            try:
                import psutil
                self._psutil = psutil
            except ImportError:
                self._psutil = None
        if self._psutil is not None:
            try:
                proc = self._psutil.Process()
                total = proc.memory_info().rss
                for child in proc.children(recursive=True):
                    try:
                        total += child.memory_info().rss
                    except (self._psutil.NoSuchProcess, self._psutil.AccessDenied):
                        continue
                return total / _GB
            except Exception:
                return own
        children = _proc_children_map()
        if not children:
            return own
        total = own or 0.0
        pending = list(children.get(os.getpid(), []))
        seen = set()
        while pending:
            pid = pending.pop()
            if pid in seen:
                continue
            seen.add(pid)
            rss = _read_proc_rss_gb(str(pid))
            if rss is not None:
                total += rss
            pending.extend(children.get(pid, []))
        return total


# ---------------------------------------------------------------------------
# The per-process recorder
# ---------------------------------------------------------------------------

class Telemetry:
    """Per-process recorder. Use the module-level functions, not this class
    directly; one instance lives in ``_STATE`` and is re-armed by
    :func:`configure` at the start of every ``GP()`` call."""

    def __init__(self) -> None:
        self.enabled = False
        self._warned = False
        self._path: Optional[str] = None
        self._handle = None
        self._lock = threading.Lock()
        self._sampler: Optional[_Sampler] = None
        self._ctx: Dict[str, Any] = {}
        self._job_t0 = time.perf_counter()
        self._first_task_t: Optional[float] = None
        self._task: Optional[Dict[str, Any]] = None
        self._task_t0 = 0.0
        self._task_dims: Dict[str, Any] = {}
        self._model: Optional[Dict[str, Any]] = None
        self._n_tasks_done = 0
        self._tasks_total_s = 0.0
        self._machine_key = ''
        self._alloc: Dict[str, Any] = {}
        self._run_id = ''
        self._host = ''

    # -- lifecycle ---------------------------------------------------------

    def configure(self, enabled: bool = True, interval_s: float = 0.5,
                  engine: str = '', result_name: str = '', scenario: str = '',
                  context: Optional[Mapping] = None) -> None:
        """Arm telemetry for one ``GP()`` call. ``context`` carries the
        concurrency layout written into every model record (``workers``,
        ``n_jobs``, ``blas_threads``, ``torch_threads``, ``device``)."""
        self._shutdown_quietly()
        self.enabled = False
        self._warned = False
        self._ctx = {'engine': str(engine), 'result_name': str(result_name),
                     'scenario': str(scenario)}
        self._ctx.update(dict(context or {}))
        self._job_t0 = time.perf_counter()
        self._first_task_t = None
        self._task = None
        self._model = None
        self._task_dims = {}
        self._n_tasks_done = 0
        self._tasks_total_s = 0.0
        if not enabled:
            return
        directory = os.environ.get(TELEMETRY_DIR_ENV, '')
        if not directory:
            self._warn_once('no telemetry directory is set (EASIGP_TELEMETRY_DIR is empty)',
                            'this run records no resource usage')
            return
        try:
            os.makedirs(directory, exist_ok=True)
            host = socket.gethostname().split('.')[0] or 'host'
            stem = f"{host}_{os.getpid()}_{secrets.token_hex(4)}"
            self._path = os.path.join(directory, stem + '.jsonl')
            self._handle = open(self._path, 'a', buffering=1, encoding='utf-8')
        except OSError as exc:
            self._warn_once(f"cannot open a file in '{directory}' ({type(exc).__name__}: {exc})",
                            'this run records no resource usage')
            self._handle = None
            return
        self._host = host
        self._run_id = stem
        self._machine_key = machine_key()
        self._alloc = alloc_from_env()
        self._sampler = _Sampler(float(interval_s))
        self.enabled = True

    def set_context(self, context: Mapping) -> None:
        """Merge more concurrency facts (resolved later than ``configure``)."""
        if not self.enabled:
            return
        self._ctx.update(dict(context or {}))

    def _warn_once(self, why: str, consequence: str) -> None:
        if self._warned:
            return
        self._warned = True
        print(f"[telemetry] WARNING: {why} - {consequence}.")

    def _disable(self, why: str) -> None:
        self._warn_once(why, 'telemetry is switched off for the rest of this process')
        self.enabled = False

    def _shutdown_quietly(self) -> None:
        try:
            if self._sampler is not None:
                self._sampler.stop()
        except Exception:
            pass
        self._sampler = None
        try:
            if self._handle is not None:
                self._handle.close()
        except OSError:
            pass
        self._handle = None

    # -- writing -----------------------------------------------------------

    def _header(self, kind: str) -> Dict[str, Any]:
        task = dict(self._task) if self._task else None
        return {
            'v': SCHEMA_VERSION, 'kind': kind, 'run_id': self._run_id,
            'ts': round(time.time(), 1),
            'engine': self._ctx.get('engine', ''), 'machine_key': self._machine_key,
            'host': self._host, 'job': dict(self._alloc), 'task': task,
            # Short hash of the config switches that change the per-task pool cost
            # (LD pruning, RF filtering, genotype format, ...); set by GP().
            'pool_sig': self._ctx.get('pool_sig'),
        }

    def _write(self, record: Dict[str, Any]) -> None:
        if not self.enabled or self._handle is None:
            return
        try:
            line = json.dumps(_jsonable(record), separators=(',', ':'))
            with self._lock:
                self._handle.write(line + '\n')
        except (OSError, ValueError, TypeError) as exc:
            self._disable(f"writing '{self._path}' failed ({type(exc).__name__}: {exc})")

    # -- task --------------------------------------------------------------

    def task_begin(self, row: int, population: Any, phenotype: Any, ratio: Any,
                   sample: Any, option: Any = None) -> None:
        if not self.enabled:
            return
        if self._first_task_t is None:
            self._first_task_t = time.perf_counter()
            if self._sampler is not None:
                self._sampler.start()
        self._task = {
            'row': int(row), 'population': str(population), 'phenotype': str(phenotype),
            'ratio': _jsonable(ratio), 'sample': _jsonable(sample),
            'option': _jsonable(option),
            'pilot': os.environ.get(TELEMETRY_PILOT_ENV, '') == '1',
        }
        self._task_t0 = time.perf_counter()
        self._task_dims = {}
        self._model = None
        if self._sampler is not None:
            self._sampler.reset_task()
            self._sampler.tick()

    def task_dims(self, n_task: int, p_raw: int) -> None:
        if not self.enabled:
            return
        self._task_dims = {'n_task': int(n_task), 'p_raw': int(p_raw)}

    def task_end(self, status: str = 'ok',
                 stages: Optional[Mapping[str, float]] = None) -> None:
        if not self.enabled or self._task is None:
            return
        if self._sampler is not None:
            self._sampler.tick()
        # A task that failed while a model was running leaves that model open:
        # record it as a failed model (its memory is a lower bound, AC2.4).
        if status != 'ok' and self._model is not None:
            self._emit_model(status='failed')
        record = self._header('task')
        record.update({
            'workers': self._ctx.get('workers'),
            'n_task': self._task_dims.get('n_task'),
            'p_raw': self._task_dims.get('p_raw'),
            'stages': {str(k): _num(v, 3) for k, v in (stages or {}).items()},
            'total_s': _num(time.perf_counter() - self._task_t0, 3),
            'rss_tree_peak_gb': _num(self._sampler.task_tree_peak) if self._sampler else None,
            'status': str(status),
        })
        if record['rss_tree_peak_gb'] == 0.0:
            record['rss_tree_peak_gb'] = None
        self._write(record)
        self._tasks_total_s += max(0.0, time.perf_counter() - self._task_t0)
        if status == 'ok':
            self._n_tasks_done += 1
        self._task = None
        self._model = None

    # -- model -------------------------------------------------------------

    def model_begin(self, model: str, base: Optional[str] = None) -> None:
        if not self.enabled:
            return
        tree, own = (None, None)
        if self._sampler is not None:
            tree, own = self._sampler.current()
            self._sampler.reset_model()
            self._sampler.tick()
        gpu_before = self._reset_gpu_peak()
        self._model = {
            'model': str(model), 'base': str(base if base is not None else model),
            't0': time.perf_counter(), 'baseline_tree': tree, 'baseline_self': own,
            'hwm_begin': _read_proc_vmhwm_gb(), 'gpu_reset': gpu_before, 'context': None,
        }

    def model_context(self, params: Optional[Sequence], hp_tune: Optional[Mapping],
                      train_shape: Tuple[int, int], valid_rows: int, test_rows: int,
                      p_raw: Optional[int], tuned: bool) -> None:
        """Store what is known about the open model now, so that a model that
        later fails can still be recorded with its sizes and parameters."""
        if not self.enabled or self._model is None:
            return
        self._model['context'] = {
            'params': params, 'hp_tune': hp_tune, 'train_shape': tuple(train_shape),
            'valid_rows': int(valid_rows), 'test_rows': int(test_rows),
            'p_raw': p_raw, 'tuned': bool(tuned),
        }

    def model_end(self, model: str, params: Optional[Sequence], hp_tune: Optional[Mapping],
                  train_shape: Tuple[int, int], valid_rows: int, test_rows: int,
                  p_raw: Optional[int], tuned: bool, status: str = 'ok',
                  gpu_wait_s: Optional[float] = None) -> None:
        if not self.enabled or self._model is None:
            return
        self.model_context(params, hp_tune, train_shape, valid_rows, test_rows, p_raw, tuned)
        self._emit_model(status=status, gpu_wait_s=gpu_wait_s)
        self._model = None

    def _emit_model(self, status: str, gpu_wait_s: Optional[float] = None) -> None:
        state = self._model
        if state is None:
            return
        if self._sampler is not None:
            self._sampler.tick()
        context = state.get('context') or {}
        shape = context.get('train_shape') or (None, None)
        seconds = time.perf_counter() - state['t0']
        tree_peak = self._sampler.model_tree_peak if self._sampler else 0.0
        self_peak = self._sampler.model_self_peak if self._sampler else 0.0
        # VmHWM is exact for this process: when it rose during the model, the
        # new value is a true peak and beats any sampled maximum.
        hwm_now = _read_proc_vmhwm_gb()
        hwm_begin = state.get('hwm_begin')
        if hwm_now is not None and (hwm_begin is None or hwm_now > hwm_begin + 1e-6):
            self_peak = max(self_peak, hwm_now)
            tree_peak = max(tree_peak, hwm_now)
        record = self._header('model')
        record.update({
            'model': state['model'], 'base': state['base'],
            'params': context.get('params'), 'hp_tune': context.get('hp_tune'),
            'tuned': context.get('tuned'),
            'device': self._ctx.get('device'),
            'n_train': shape[0], 'n_valid': context.get('valid_rows'),
            'n_test': context.get('test_rows'), 'p': shape[1], 'p_raw': context.get('p_raw'),
            'seconds': _num(seconds, 3), 'gpu_wait_s': _num(gpu_wait_s, 3),
            'rss_tree_peak_gb': _num(tree_peak) if tree_peak > 0 else None,
            'rss_self_peak_gb': _num(self_peak) if self_peak > 0 else None,
            'rss_baseline_gb': _num(state.get('baseline_tree') or state.get('baseline_self')),
            'gpu_peak_gb': self._read_gpu_peak(),
            'workers': self._ctx.get('workers'), 'n_jobs': self._ctx.get('n_jobs'),
            'blas_threads': self._ctx.get('blas_threads'),
            'torch_threads': self._ctx.get('torch_threads'),
            'scenario': self._ctx.get('scenario'), 'status': str(status),
        })
        self._write(record)

    @staticmethod
    def _reset_gpu_peak() -> bool:
        """Reset torch's peak-memory counter if torch is already imported and
        CUDA is initialised (never imports torch itself)."""
        torch = sys.modules.get('torch')
        if torch is None:
            return False
        try:
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                torch.cuda.reset_peak_memory_stats()
                return True
        except Exception:
            return False
        return False

    @staticmethod
    def _read_gpu_peak() -> Optional[float]:
        torch = sys.modules.get('torch')
        if torch is None:
            return None
        try:
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                return _num(torch.cuda.max_memory_allocated() / _GB)
        except Exception:
            return None
        return None

    # -- job ---------------------------------------------------------------

    def job_end(self) -> None:
        """Write the one ``job`` record and stop the sampler."""
        if not self.enabled:
            self._shutdown_quietly()
            return
        try:
            if self._task is not None:
                self.task_end(status='failed')
            now = time.perf_counter()
            record = self._header('job')
            first = self._first_task_t
            record.update({
                'wall_s': _num(now - self._job_t0, 3),
                'startup_s': _num((first - self._job_t0) if first is not None
                                  else (now - self._job_t0), 3),
                'cgroup_peak_gb': _num(_cgroup_peak_gb()),
                'ru_maxrss_children_gb': _num(_children_maxrss_gb()),
                'n_tasks_done': self._n_tasks_done,
                # Everything after the last task (ensemble finalisation, result
                # saving): wall time minus start-up minus the sum of task times.
                'tasks_total_s': _num(self._tasks_total_s, 3),
                'tail_s': _num(max(0.0, (now - self._job_t0)
                                   - ((first - self._job_t0) if first is not None else 0.0)
                                   - self._tasks_total_s), 3),
            })
            self._write(record)
        except Exception as exc:
            self._warn_once(f"writing the job record failed ({type(exc).__name__}: {exc})",
                            'this job has no job-level record')
        finally:
            self.enabled = False
            self._shutdown_quietly()


# ---------------------------------------------------------------------------
# Module-level API (what genomic_prediction.py calls)
# ---------------------------------------------------------------------------

_STATE = Telemetry()


def _safe(callable_obj, *args, **kwargs) -> None:
    """Run one recorder method; telemetry must never raise into the pipeline."""
    try:
        callable_obj(*args, **kwargs)
    except Exception as exc:
        _STATE._warn_once(f"internal error in {getattr(callable_obj, '__name__', '?')} "
                          f"({type(exc).__name__}: {exc})", 'continuing without this record')


def get_state() -> Telemetry:
    """The process-wide recorder (for tests)."""
    return _STATE


def configure(enabled: bool = True, interval_s: float = 0.5, engine: str = '',
              result_name: str = '', scenario: str = '',
              context: Optional[Mapping] = None) -> None:
    """Arm telemetry for one ``GP()`` call (see :meth:`Telemetry.configure`)."""
    _safe(_STATE.configure, enabled, interval_s, engine, result_name, scenario, context)


def set_context(context: Mapping) -> None:
    """Add concurrency facts (``workers``, ``n_jobs``, ``blas_threads``,
    ``torch_threads``, ``device``) once ``GP()`` has resolved them."""
    _safe(_STATE.set_context, context)


def is_enabled() -> bool:
    """True while records are being written."""
    return bool(_STATE.enabled)


def task_begin(row: int, population: Any, phenotype: Any, ratio: Any, sample: Any,
               option: Any = None) -> None:
    """Open a task (``row`` is the absolute row of the ``sample`` table)."""
    _safe(_STATE.task_begin, row, population, phenotype, ratio, sample, option)


def task_dims(n_task: int, p_raw: int) -> None:
    """Record the task's individual count and raw marker count."""
    _safe(_STATE.task_dims, n_task, p_raw)


def task_end(status: str = 'ok', stages: Optional[Mapping[str, float]] = None) -> None:
    """Close the open task and write its record (``status`` 'ok' or 'failed')."""
    _safe(_STATE.task_end, status, stages)


def model_begin(model: str, base: Optional[str] = None) -> None:
    """Open a model: reset the per-model memory peak, note the baseline."""
    _safe(_STATE.model_begin, model, base)


def model_context(params: Optional[Sequence], hp_tune: Optional[Mapping],
                  train_shape: Tuple[int, int], valid_rows: int, test_rows: int,
                  p_raw: Optional[int], tuned: bool) -> None:
    """Store the open model's sizes/parameters so a later failure keeps them."""
    _safe(_STATE.model_context, params, hp_tune, train_shape, valid_rows, test_rows,
          p_raw, tuned)


def model_end(model: str, params: Optional[Sequence], hp_tune: Optional[Mapping],
              train_shape: Tuple[int, int], valid_rows: int, test_rows: int,
              p_raw: Optional[int], tuned: bool, status: str = 'ok',
              gpu_wait_s: Optional[float] = None) -> None:
    """Close the open model and write its ``model`` record.

    ``train_shape`` is ``(n_train, p)``: the training rows and the number of
    MARKER columns the model saw (not counting the phenotype column).
    """
    _safe(_STATE.model_end, model, params, hp_tune, train_shape, valid_rows, test_rows,
          p_raw, tuned, status, gpu_wait_s)


def job_end() -> None:
    """Write the ``job`` record and stop the sampler (call in a ``finally``)."""
    _safe(_STATE.job_end)
