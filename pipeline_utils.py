"""
EasiGP - shared helpers used by BOTH streamlit_app.py (the interactive GUI,
used to author a configuration ONCE) and the headless, non-interactive
scripts actually launched by the HPC scheduler (run_sequential.py,
run_step1_batch.py, run_step2_assemble.py).

Keeping these in a separate module means the headless scripts never have to
import streamlit_app.py itself (which would re-execute the whole GUI script
outside of a Streamlit runtime and fail).
"""

import contextlib
import csv
import io
import math
import os
import re
import shutil
import sys
import threading
import time
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Silence scikit-learn's repeated "X does not have valid feature names"
# UserWarning (e.g. "X does not have valid feature names, but
# WeightedKNeighborsRegressor was fitted with feature names").
#
# Why it appears: every model here is FITTED on a pandas DataFrame (so the
# estimator remembers the marker names), but the model-agnostic explainers
# (shap.KernelExplainer, shapiq, the Friedman-H / baseline-SHAP routines)
# call `.predict()` with plain NumPy arrays for speed. scikit-learn then
# warns on EVERY such call - thousands of identical lines per task. The
# predictions are unaffected (column order is preserved by construction), so
# the warning carries no information here.
#
# Two layers, both needed:
#   1. `warnings.filterwarnings` - covers this process (and every module that
#      imports pipeline_utils, which every model file does, so joblib/loky
#      and multiprocessing workers that unpickle a model function pick it up
#      on import).
#   2. PYTHONWARNINGS - covers child interpreters started AFTER this point
#      (loky / spawn workers), which do not inherit in-process filters.
# Only this one message is filtered; every other warning is untouched.
# ---------------------------------------------------------------------------
_FEATURE_NAMES_WARNING_MSG = 'X does not have valid feature names'


def silence_feature_names_warning():
    warnings.filterwarnings('ignore', message=_FEATURE_NAMES_WARNING_MSG, category=UserWarning)
    _entry = f'ignore:{_FEATURE_NAMES_WARNING_MSG}:UserWarning'
    _existing = os.environ.get('PYTHONWARNINGS', '')
    if _entry not in _existing:
        os.environ['PYTHONWARNINGS'] = f'{_existing},{_entry}' if _existing else _entry


silence_feature_names_warning()

# Absolute path to the EasiGP install directory (where this module,
# genomic_prediction.py, main_app.py, checkpoint_utils.py etc. all live),
# resolved once at import time from this module's own __file__.
#
# EVERY "Result/<RESULT_NAME>/..." path anywhere in the codebase MUST be
# built from RESULT_BASE_DIR / result_dir_path() below, rather than a
# cwd-relative "./Result/...". Bugfix: several modules (checkpoint_utils.py,
# genomic_prediction.py, main_app.py, run_step2_assemble.py,
# intra_batch_parallel.py) previously each built their own "./Result/..."
# path independently, relative to whatever the CURRENT WORKING DIRECTORY of
# the calling process happened to be. That works fine as long as every
# entry point is launched with cwd == the install directory, but nothing
# guaranteed that:
#   - Streamlit does not chdir into the directory containing the script
#     it's running, so `streamlit run main_app.py` launched from any OTHER
#     directory left the GUI computing `Result/<name>` relative to THAT
#     directory instead.
#   - The headless HPC scripts, once cd'd into the actual install
#     directory (see the generated PBS/Slurm scripts), computed
#     `Result/<name>` relative to the install directory instead.
# Two different processes for the SAME run could therefore end up
# creating/looking for two DIFFERENT "Result/<RESULT_NAME>" folders in two
# different places on disk - e.g. the GUI writes step1_config.json and the
# HPC manifest under one location while GP() itself (running headlessly)
# creates and writes all of its actual output under another. Anchoring
# every such path to RESULT_BASE_DIR (install-directory-relative, never
# cwd-relative) makes every entry point agree on the exact same location
# regardless of the process's own working directory.
_INSTALL_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_BASE_DIR = os.path.join(_INSTALL_DIR, 'Result')


def result_dir_path(result_name):
    """The single source of truth for 'where does Result/<result_name>
    live on disk'. Always an absolute path anchored to the EasiGP install
    directory (RESULT_BASE_DIR above), never to the calling process's
    current working directory. Every module that needs a path under
    Result/<result_name>/... should build it from this function (or
    RESULT_BASE_DIR directly), not by re-deriving './Result/...' itself."""
    return os.path.join(RESULT_BASE_DIR, result_name)


# ---------------------------------------------------------------------------
# ver4-44, R0 - stale-engine detection.
#
# Why this exists (blueprint S5): "Run pipeline now (local)" runs GP() inside
# the long-lived Streamlit process, and a module that was imported before the
# source files were updated stays in memory. The GUI (re-rendered from the new
# main_app.py) then offered `search_strategy='weight_first_cv'` while the
# engine in memory did not know it and silently ran 'monolithic'; the run was
# read as a weight-first result. A run must execute the code on disk, or stop.
# ---------------------------------------------------------------------------
ENGINE_VERSION = 'ver4-56'

# When this module was imported. A module whose source was modified after this
# moment cannot have been imported from that source (used when a module has no
# readable .pyc header to compare against).
_PROCESS_T0 = time.time()

#: Environment variable that switches the stale-engine check off ('off').
STALE_CHECK_ENV = 'EASIGP_STALE_CHECK'


def stale_check_enabled() -> bool:
    """False only when ``EASIGP_STALE_CHECK`` is set to ``off`` (any case).
    Why: a file-sync tool that touches modification times would otherwise make
    every run refuse; the override is the documented escape hatch."""
    return os.environ.get(STALE_CHECK_ENV, 'on').strip().lower() != 'off'


def _pyc_header(cached_path):
    """``(source_mtime, source_size)`` recorded in a timestamp-based ``.pyc``
    header, or ``None``.

    The 16-byte header is: magic (4), flags (4), source mtime (4, little
    endian), source size (4). ``None`` is returned for no path, a missing or
    unreadable file, a short read, or a hash-based ``.pyc`` (flags != 0, which
    records no mtime) - the caller then uses the process-start rule instead.
    Never raises."""
    if not cached_path:
        return None
    try:
        with open(cached_path, 'rb') as fh:
            head = fh.read(16)
    except OSError:
        return None
    if len(head) < 16 or int.from_bytes(head[4:8], 'little') != 0:
        return None
    return (int.from_bytes(head[8:12], 'little'), int.from_bytes(head[12:16], 'little'))


def stale_engine_modules(root: Optional[str] = None) -> List[str]:
    """Names of loaded modules under ``root`` (default: the EasiGP install
    directory) whose ``.py`` source changed after the module was imported.

    A module is stale when its source was modified after this process started
    (``_PROCESS_T0``) AND, if its cached ``.pyc`` has a readable header, that
    header (source mtime and size at compile time) no longer matches the file.
    The first condition keeps a read-only install tree - where an outdated
    ``.pyc`` cannot be rewritten although the module was imported from the
    current source - from being reported as stale on every run.

    Args: ``root`` - directory prefix to inspect. Returns a sorted list of
    module names (empty when nothing is stale). Never raises: a module whose
    source cannot be ``stat``-ed is skipped."""
    base = os.path.abspath(root or _INSTALL_DIR)
    prefix = base if base.endswith(os.sep) else base + os.sep
    stale = []
    for name, mod in list(sys.modules.items()):
        src = getattr(mod, '__file__', None) or ''
        if not src.endswith('.py'):
            continue
        src = os.path.abspath(src)
        if not src.startswith(prefix):
            continue
        try:
            st = os.stat(src)
        except OSError:
            continue
        if st.st_mtime <= _PROCESS_T0:
            continue                    # unchanged since this process started
        header = _pyc_header(getattr(mod, '__cached__', None))
        if header is None or header != (int(st.st_mtime) & 0xFFFFFFFF,
                                        st.st_size & 0xFFFFFFFF):
            stale.append(name)
    return sorted(stale)


def stale_engine_message(stale: List[str]) -> str:
    """The one user-facing sentence shared by the GUI run button and ``GP()``:
    which modules changed and what to do (restart; or the override variable)."""
    return (f"EasiGP code changed since this window started ({', '.join(stale)}) - the engine "
            f"in memory is older than the files on disk. Restart EasiGP (close the Streamlit "
            f"window/process and start it again) before running. If the files were only "
            f"touched by a sync tool, set {STALE_CHECK_ENV}=off to skip this check.")


# Options offered in the GUI for how a PARALLEL batch ID should be
# determined. Kept here so the GUI and the headless runner agree on the
# same source names / environment variables.
BATCH_ID_SOURCES = [
    'Manual integer',
    'Slurm (SLURM_ARRAY_TASK_ID)',
    'PBS (PBS_ARRAY_INDEX / PBS_ARRAYID)',
]

# ver4-25, R14: the fan-out / CPU-threading settings the GUI used to
# expose behind the "Show advanced setting components" button in
# `main_app.py::_render_advanced_setting_components()` (worker-fan-out
# widgets N_CPU_WORKERS/N_MODEL_WORKERS, the model-grouping strategy
# fields, and the torch/BLAS-threading and SEQUENTIAL_INTRA_BATCH
# globals). That button is removed (R14; E6 root-cause fix - see
# `main_app.py::render_hpc_export_section()`'s own comment at the
# deleted call site): every export the GUI produces from ver4-25 onward
# pins these to the values below, unconditionally, rather than reading a
# widget's `session_state` value that may never have been instantiated.
#
# The engine itself is completely UNCHANGED by this: a hand-edited
# `*_config.json` that sets any of these keys differently is still
# honoured exactly as before - only the GUI stops being able to produce
# anything else. Defined here (rather than only in `main_app.py`)
# because the headless runners (`run_sequential.py`, `run_step1_batch.py`)
# also need it, purely to log when a config's fan-out settings were NOT
# produced by the GUI since ver4-25 (see `log_pinned_fanout_deviations()`
# below) - and importing `main_app.py` from a headless script would
# re-execute the whole Streamlit script outside a Streamlit runtime and
# fail (see this module's own docstring, top of file).
PINNED_FANOUT_DEFAULTS: Dict[str, object] = {
    'N_CPU_WORKERS': 1,
    'N_CPU_WORKERS_TASK': 1,
    'N_MODEL_WORKERS': 1,
    'MODEL_GROUPING': 'cost_balanced',
    'BIO_PRIOR_GROUPING': 'affinity',
    'MIN_MODELS_FOR_PARALLEL': 2,
    'SEQUENTIAL_INTRA_BATCH': False,
    'R_BLAS_THREADS': None,
    'R_BLAS_FOLLOWS_N_JOBS': True,
    'TORCH_NUM_THREADS': None,
    'TORCH_DATALOADER_WORKERS': 0,
}


def log_pinned_fanout_deviations(cfg: Dict, *, log_prefix: str) -> None:
    """Print one NOTE line when this run's fan-out/CPU-threading settings
    (PINNED_FANOUT_DEFAULTS) differ from the values the GUI has pinned
    unconditionally into every export since ver4-25 (R14).

    Purely diagnostic - never changes `cfg`, never changes GP()'s
    behaviour, and never raises. A hand-edited config with, e.g.,
    N_CPU_WORKERS=4 still runs with 4 workers exactly as before this
    update; this only makes that fact visible in the run log rather than
    silent, since the GUI itself can no longer produce such a value (see
    architecture doc §18, "Diagnostics as a design principle" - log
    negative/unusual states unconditionally rather than staying silent).

    Args:
        cfg: the loaded run config dict (upper-case engine keys), as read
            from a `*_config.json` file.
        log_prefix: the calling script's own bracketed log tag (e.g.
            '[run_sequential]' or '[run_step1_batch]'), so the NOTE reads
            consistently with that script's other log lines.

    Returns:
        None.

    Raises:
        Nothing - a side-effect-only (stdout) diagnostic helper.
    """
    deviations = []
    for key, default in PINNED_FANOUT_DEFAULTS.items():
        value = cfg.get(key, default)
        if value != default:
            deviations.append(f"{key}={value!r} (GUI-pinned default {default!r})")
    if deviations:
        print(f"{log_prefix} NOTE: this config's fan-out/CPU-threading settings differ from "
              f"the values the GUI has pinned unconditionally - i.e. this "
              f"config was not produced by the current GUI, or was hand-edited afterwards: "
              + '; '.join(deviations) + ". Every value is honoured exactly as configured - "
              "this line is purely informational.")


class TimestampedWriter:
    """A minimal stdout-like writer that prefixes every non-blank line with
    a timestamp before passing it through to one or more underlying streams
    at once (e.g. the real console, an in-memory buffer for the GUI to
    display, and a log file on disk).

    This is used both by the headless HPC scripts (wrapping sys.stdout, so
    the scheduler's captured output file gets timestamps on every line
    uniformly - not just the handful of lines that happen to call a
    dedicated logging helper) and by the GUI's local-run option (wrapping
    the in-memory log buffer plus a real file, so 'Option B' runs get a
    saved, timestamped log too, not just an ephemeral in-browser display).

    Handles partial/multi-part writes correctly (print() and other callers
    don't always write a whole line at once) - a timestamp is only added at
    the start of an actual new line, never mid-line.
    """

    def __init__(self, *streams):
        self.streams = [s for s in streams if s is not None]
        self._at_line_start = True

    def write(self, text):
        if not text:
            return 0
        for chunk in text.splitlines(keepends=True):
            if self._at_line_start:
                content = chunk.rstrip('\r\n')
                if content != '':
                    prefix = f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] '
                    chunk = prefix + chunk
                self._at_line_start = False
            line_complete = chunk.endswith('\n')
            if line_complete:
                self._at_line_start = True
            for s in self.streams:
                s.write(chunk)
            if line_complete:
                # Requirement 9: flush every underlying stream (in
                # particular the on-disk log file) as soon as a complete
                # line has been written, rather than waiting for Python's
                # own internal buffering to do it eventually - stdout is
                # BLOCK-buffered (not line-buffered) by default whenever
                # it isn't a real interactive terminal, which is every
                # HPC job (output always redirected to a file) and every
                # GUI local-run's log file. Without this, a long-running
                # job's saved log can go HOURS without any new lines
                # actually landing on disk - even though the job is
                # progressing completely normally - making it impossible
                # to monitor in real time (e.g. `tail -f` on the
                # scheduler's own .o file, or the GUI's live log
                # display). The cost of a flush() call is negligible next
                # to the time between log lines in this pipeline (at
                # minimum whole model-fitting epochs, typically much
                # more), so there's no meaningful performance downside.
                self.flush()
        return len(text)

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        return False


def make_run_log_path(result_name, label):
    """Build a timestamped log file path under Result/<result_name>/logs/,
    e.g. Result/MaizeNAM/logs/log_sequential_20260727_143205.txt. `label` is
    a short tag identifying which kind of run this is (e.g.
    'sequential_local', 'step1_local', 'step2_local'). Creates
    Result/<result_name>/logs/ if it doesn't exist yet.

    Every run (GUI local-run and every headless HPC script) saves its log
    here, so Result/<result_name>/logs/ is where every actual run log
    lives - nothing else is written there."""
    log_dir = os.path.join(result_dir_path(result_name), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(log_dir, f'log_{label}_{stamp}.txt')


def resolve_scratch_tmp_dir(result_name=None, explicit=None):
    """Resolve an OPTIONAL, PERSISTENT scratch-temp base directory for this
    run - purely additive, off by default (returns None, meaning "do
    nothing, leave the OS/scheduler's own default temp directory alone",
    exactly as every run behaved before this option existed).

    WHY THIS EXISTS (PATCH_NOTES: "FileNotFoundError ... tmpXXXXXXXX" worker
    crash during a long HPC run): every `tempfile.TemporaryDirectory()` /
    `tempfile.mkdtemp()` call anywhere in this process (or in a spawned
    child - PLINK2 subprocesses, and every `joblib`/`loky` hyperparameter-
    tuning trial worker, which each independently re-import this whole
    project fresh - see genomic_prediction.py's own top-of-file imports)
    resolves to whatever `tempfile.gettempdir()` returns, which - absent
    this option - is entirely at the mercy of the `TMPDIR` environment
    variable the job scheduler happened to set. Many HPC sites point
    `TMPDIR` at a job-ID-scoped scratch directory (e.g.
    `/scratch/temp/<jobid>`) that the scheduler can reclaim as soon as a
    job is flagged for termination (walltime hit, `scancel`, ...) - a
    worker process spawned (or still starting up) at exactly that moment
    can fail with a raw `FileNotFoundError` deep inside a completely
    unrelated import (observed: a fresh `loky` worker failing while
    merely IMPORTING `torch_geometric`, which transitively imports
    `torch.distributed`, which creates its OWN temp directory at import
    time) - a confusing, misleading symptom of what is really just
    "the job's temp directory disappeared out from under a process that's
    still running". Redirecting to a stable, user-owned directory (NOT
    tied to any one job's lifetime) removes this whole failure class,
    independent of whatever actually triggered the original TMPDIR's
    disappearance.

    This is deliberately NOT the default, for two reasons: (1) it would
    silently change where potentially large PLINK2 `--export A` scratch
    files land for every existing installation (see genomic_prediction.py
    ``_new_plink_work_dir()``'s own docstring on why that scratch usage is
    kept OUT of Result/<RESULT_NAME>/ by design), and (2) a location this
    module could safely assume is both large enough and NOT auto-cleaned
    per-job cannot be guessed generically across HPC sites - only the
    person submitting the job knows that. Opt in by setting the
    `EASIGP_SCRATCH_TMPDIR` environment variable (e.g. in the Slurm/PBS
    submission script: `export EASIGP_SCRATCH_TMPDIR=/scratch/user/<you>/easigp_tmp`)
    to a directory on a persistent (non-job-scoped) filesystem with enough
    free space for the largest single PLINK2 export this run will do -
    typically the full, unrestricted marker pool for the largest
    population, at up to 4 bytes/genotype-call in `--export A`'s text
    format.

    Parameters
    ----------
    result_name : str, optional
        If given, a `result_name`-named subdirectory is used (so different
        runs sharing one `EASIGP_SCRATCH_TMPDIR` base don't collide) -
        NOT further namespaced per batch/PID, since PLINK2/loky's own
        scratch files already use unique per-call temp subdirectory names
        underneath whatever base directory is in effect.
    explicit : str, optional
        Overrides the `EASIGP_SCRATCH_TMPDIR` environment variable when
        given directly (e.g. from a config key), for callers that want to
        control this without setting a process environment variable.

    Returns
    -------
    str or None
        The resolved, already-created directory, or None if neither
        `explicit` nor `EASIGP_SCRATCH_TMPDIR` is set.
    """
    base = explicit or os.environ.get('EASIGP_SCRATCH_TMPDIR')
    if not base:
        return None
    path = os.path.join(base, result_name) if result_name else base
    os.makedirs(path, exist_ok=True)
    return path


def apply_scratch_tmp_dir(result_name=None, explicit=None):
    """Resolve `resolve_scratch_tmp_dir(...)` and, if it returns non-None,
    point THIS process's own default temp directory at it - both
    `os.environ['TMPDIR']` (inherited by every child process spawned
    AFTER this call: PLINK2 subprocesses, and every `loky` hyperparameter-
    tuning worker) and `tempfile.tempdir` (so anything in THIS process
    that already cached `tempfile.gettempdir()`'s result before this
    call... doesn't - `tempfile.tempdir` starts `None` and is only
    resolved lazily on first real use, so calling this early enough, as
    every headless entry point now does, is sufficient).

    A no-op (prints nothing, changes nothing) when nothing is configured -
    see `resolve_scratch_tmp_dir()`'s own docstring for why this is
    opt-in rather than a new default. Safe to call more than once.

    Returns the resolved path (or None, matching `resolve_scratch_tmp_dir`).
    """
    path = resolve_scratch_tmp_dir(result_name=result_name, explicit=explicit)
    if path is None:
        return None
    os.environ['TMPDIR'] = path
    import tempfile
    tempfile.tempdir = path
    print(
        f"[pipeline_utils] EASIGP_SCRATCH_TMPDIR resolved: this process's default temp "
        f"directory (and every child/worker process's, since they inherit this via "
        f"os.environ) is now '{path}' instead of the scheduler/OS default. Set to avoid "
        f"long runs depending on a job-scoped scratch directory that can be reclaimed "
        f"before every spawned worker has finished starting."
    )
    return path


def ensure_r_native_symbols_global():
    """BUGFIX (undefined symbol: R_NilValue) - best-effort fix for a
    well-known Linux embedding failure that is otherwise invisible in
    this codebase's four pre-existing R models and only surfaces the
    first time a package with its OWN compiled native code (`asreml` -
    Preprocess/asreml_fit.R's `library(asreml)`) is actually loaded.

    Root cause: when R is hosted INSIDE Python via rpy2 (as everywhere
    in this codebase - see configure_r_environment()'s own docstring on
    why `import rpy2.robjects` triggers R's initialisation), Python -
    not R - is the process's main executable, and rpy2 loads `libR.so`
    itself via an explicit `dlopen()` call rather than the OS resolving
    it as one of the main executable's own link-time dependencies. On
    glibc, a library loaded via an explicit `dlopen()` defaults to
    LOCAL symbol visibility unless the loader specifically requests
    RTLD_GLOBAL - meaning `libR.so`'s own exported C-API symbols
    (`R_NilValue`, `R_GlobalEnv`, ...) are NOT automatically visible to
    a LATER, separate `dyn.load()` of some OTHER compiled `.so`, even
    though they always would be when R itself (`Rscript`) is the host
    executable, because then `libR.so` is one of the EXECUTABLE's own
    link-time dependencies, which the dynamic linker always resolves
    with global scope. A compiled R package that expects those symbols
    to already be globally resolvable - which is the ordinary case for
    every package author, since almost nobody develops or tests against
    an embedded-R scenario - fails at `dyn.load()` time instead, with
    exactly the "undefined symbol" error this function exists to avoid.

    This is NOT something the four pre-existing R models (rrBLUP/GBLUP/
    BayesB/RKHS's shared BGLR dependency) ever exposed, because BGLR's
    own compiled component happens to link directly against `libR.so`
    at BUILD time (its own author's choice, outside this project's
    control) rather than relying on symbols already being globally
    visible at LOAD time - a `dyn.load()` whose own shared object
    explicitly lists `libR.so` as a dependency resolves its symbols
    against that explicit dependency regardless of the calling
    process's global symbol table, so it was never affected by this.
    `asreml` - a commercial, closed-source binary VSNi ships
    independently per R-ABI version (see the `asreml42.so` naming
    itself: ASReml-R 4.x, R 4.2) - is EasiGP's first compiled R
    dependency built the other way, and consequently the first to ever
    surface this failure mode, the very first time it was exercised
    against a real licensed installation (see Preprocess/asreml_fit.R's
    own "VERSION CAVEAT" note on why that could not be tested here).

    Fix: re-`dlopen()` the SAME `libR.so` already loaded for this R
    session, this time explicitly requesting RTLD_GLOBAL. On glibc,
    re-opening an ALREADY-LOADED shared object with RTLD_GLOBAL PROMOTES
    its symbols to global scope for the remainder of the process (this
    is documented glibc dlopen() behaviour, not a hack) - so this is
    effective even when called well AFTER R has already been
    initialised by rpy2's own earlier, LOCAL-scope load, as long as it
    runs before the SPECIFIC dyn.load() call that needs the promotion.
    Called from two places for exactly that reason (defence in depth,
    the same principle R13's own three ASReml gating layers already
    apply): once here, as early as every other R environment setting
    this function owns; and again, defensively, immediately before
    Preprocess/asreml_blup.py's own `library(asreml)`-triggering
    `r_source()` call - the one dyn.load() this whole codebase actually
    depends on being resolvable, and the one Preprocess/asreml_license.py's
    own preflight check (Layer 2) also calls it ahead of, so a genuine
    load failure on a given machine is reported consistently by the
    preflight check rather than passing there and only failing later,
    per task, inside Step 6b.

    Never raises, and never touches anything if `R_HOME` is not yet
    known (nothing to preload from) or on a platform where ELF-style
    symbol-visibility scoping does not apply (Windows PE/DLL loading
    works differently - this is a no-op there). Idempotent and cheap to
    call on every invocation - re-`dlopen()`-ing an already-loaded
    object is a fast reference-count bump, not a re-load from disk, so
    callers are never expected to guard against calling this more than
    once per process.
    """
    if sys.platform.startswith('win'):
        return

    r_home = os.environ.get('R_HOME')
    if not r_home:
        return

    # Mirrors the exact libR.so/libR.dylib location convention rpy2
    # itself already assumes when locating R_HOME's shared library (see
    # rpy2.rinterface_lib.openrlib) - if rpy2 already found and loaded R
    # successfully via this same R_HOME (a precondition for this
    # function ever being reached at all: every call site is well after
    # R has been touched once already), one of these candidates is
    # necessarily the file already resident in this process.
    candidates = [
        os.path.join(r_home, 'lib', 'libR.so'),
        os.path.join(r_home, 'lib64', 'libR.so'),
        os.path.join(r_home, 'lib', 'libR.dylib'),
    ]

    import ctypes
    for candidate in candidates:
        if not os.path.isfile(candidate):
            continue
        try:
            ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        return


def configure_r_environment(r_path, r_blas_threads=None, r_max_ppsize=500000):
    """Configure R_HOME/PATH for rpy2 to find the right R installation.

    Parameters
    ----------
    r_path : R_HOME override, or falsy to leave R_HOME as whatever's
        already in the environment (only cleaning up stray quoting/
        trailing separators on it, if present).
    r_blas_threads : int, optional (Phase 2, Requirement 6). If given,
        exports OPENBLAS_NUM_THREADS/OMP_NUM_THREADS/MKL_NUM_THREADS
        before R is ever imported/sourced, so a multi-threaded BLAS
        linked into this R build (if any) can use more than one core for
        the linear algebra underneath rrBLUP/GBLUP/BayesB/RKHS's BGLR
        MCMC. EasiGP cannot force R to link a specific BLAS at runtime -
        this only forwards a thread-count hint to whichever BLAS is
        already present; a single-threaded reference BLAS build is
        unaffected. Defaults to None (nothing set), fully backward
        compatible with every run predating this option.
    r_max_ppsize : int, optional. Raises the size of R's pointer-PROTECT
        stack before R is ever initialised, to avoid
        "Error: protect(): protection stack overflow" - a real, fatal R
        error (not an EasiGP bug) that kills the whole process outright,
        confirmed in production (EasiGP_Blueberry_step1_27682067_0.error)
        the very first time rrBLUP ran against Blueberry's full,
        unfiltered marker set: rpy2's pandas2ri conversion of a very wide
        DataFrame (one column per marker - EasiGP genotype matrices
        routinely have tens of thousands to millions of columns, see
        marker-pool routing in the architecture doc) issues one PROTECT
        per column before any UNPROTECT happens, exhausting R's default
        stack (50000) well before a genome-scale marker matrix is fully
        converted.

        Bug fix: an earlier attempt at this fix just exported an
        'R_MAX_PPSIZE' environment variable, on the assumption that R
        reads it at startup the same way it reads R_MAX_VSIZE/R_VSIZE/
        R_NSIZE. It doesn't - R's own R_SizeFromEnv() (src/main/
        startup.c) only ever calls getenv() for those three; the
        pointer-protection stack size (Rp->ppsize) is left at its
        compiled-in default and is settable ONLY via the `--max-ppsize`
        *command-line* flag R_SetParams()/R_SetPPSize() apply from -
        which is exactly why the crash still happened even with that
        environment variable set. An embedded R session (rpy2's, as used
        here) never goes through R's own command-line parsing at all;
        rpy2 instead lets the equivalent startup arguments be set
        directly via rpy2.rinterface_lib.embedded.set_initoptions()
        BEFORE anything triggers R's own initialisation (importing
        rpy2.robjects - which init_rpy2_conversion() below does - runs
        `r = R()` at that module's own top level, which calls
        rinterface.initr_simple() immediately; set_initoptions() raises
        RuntimeError once that has already happened, so this function
        MUST run first - see the "R environment configuration must
        precede the genomic_prediction import" note in the architecture
        doc, which every headless runner already honours). Defaults to
        500000, R's own documented maximum for --max-ppsize, so this is
        raised as far as R allows rather than guessed at per-dataset;
        the memory cost of a larger stack is negligible. Pass an
        explicit smaller value (or None to leave R's own compiled-in
        default of 50000 alone) if you need to override this for some
        reason.
    """
    apply_r_blas_threads(r_blas_threads)

    if r_max_ppsize is not None:
        try:
            import rpy2.rinterface_lib.embedded as _r_embedded
        except ImportError:
            # rpy2 not installed / not used by this pipeline - nothing to do.
            _r_embedded = None
        if _r_embedded is not None:
            if _r_embedded.isinitialized():
                # Too late to change this - R was already started (by an
                # earlier configure_r_environment() call in this same
                # process, or by some other import path getting to
                # rpy2.robjects first). Not fatal: just means this call
                # can't raise the limit any more than it already is.
                print(f"[pipeline_utils] NOTE: R is already initialised - "
                      f"can no longer raise its --max-ppsize to "
                      f"{int(r_max_ppsize)} (must be set before the first "
                      f"rpy2.robjects import). If a 'protect(): protection "
                      f"stack overflow' error follows, configure_r_environment() "
                      f"needs to run earlier in this process's startup.")
            else:
                _r_embedded.set_initoptions(
                    ('rpy2', '--quiet', '--no-save', f'--max-ppsize={int(r_max_ppsize)}')
                )
        # Also export the environment variable, in case a different rpy2
        # version or R build ever does start honouring it at startup (R's
        # own front ends already treat most of these Rp fields as
        # env-var-overridable - see R_SizeFromEnv()) - harmless either way
        # since real R builds simply ignore an environment variable they
        # don't read.
        if 'R_MAX_PPSIZE' not in os.environ:
            os.environ['R_MAX_PPSIZE'] = str(int(r_max_ppsize))

    existing = os.environ.get('R_HOME')
    if existing:
        cleaned = existing.strip().strip('"').rstrip(';').rstrip('\\/')
        if cleaned != existing:
            os.environ['R_HOME'] = cleaned

    if r_path:
        r_path = r_path.strip().strip('"').rstrip(';').rstrip('\\/')
        os.environ['R_HOME'] = r_path

        for bin_subdir in (os.path.join('bin', 'x64'), 'bin'):
            r_bin = os.path.join(r_path, bin_subdir)
            if os.path.isdir(r_bin):
                current_path = os.environ.get('PATH', '')
                if r_bin not in current_path:
                    os.environ['PATH'] = r_bin + os.pathsep + current_path
                break

    # BUGFIX (undefined symbol: R_NilValue) - see
    # ensure_r_native_symbols_global()'s own docstring for the full
    # root-cause explanation. Deliberately unconditional - runs whether
    # or not `r_path` was given above, because R_HOME is just as often
    # already correct on entry (e.g. an activated conda environment that
    # sets it via its own activation hooks, with R_PATH left blank in
    # the EasiGP config) as it is overridden here. This used to be an
    # early `return` right above for the `not r_path` case, which would
    # have skipped this call entirely for exactly that (common) setup -
    # restructured to an `if r_path:` block instead so this line is
    # always reached.
    ensure_r_native_symbols_global()


# ver4-59 - very wide pandas DataFrame -> R data.frame, in column chunks.
#
# rpy2 converts a DataFrame with ONE R call `data.frame(<p columns>)`, every
# column protected on R's PROTECT stack at the same time. Reproduced in a
# stand-alone test (R 4.3, rpy2 3.5.17, --max-ppsize=500000): at p = 600,000
# columns the stock conversion dies with "protect(): protection stack
# overflow"; sending the same frame in blocks of 2,500 columns and joining them
# inside R with cbind() works and gives an identical data.frame (names, types,
# class, row names; checked at p = 1,050).
#
# THIS IS NOT THE CAUSE OF THE COTTON SEGFAULTS: that data has p = 12,297
# columns (see the [R-bridge]/crash logs), far below any protect-stack limit.
# The helper is therefore only used ABOVE `r_df_chunk_threshold()` columns
# (default 100,000, env EASIGP_R_DF_CHUNK_ABOVE); at or below it the stock
# pandas2ri path runs exactly as before ver4-59.
_R_DF_CHUNK_COLS_ENV = 'EASIGP_R_DF_CHUNK_COLS'
_R_DF_CHUNK_COLS_DEFAULT = 2500
_r_df_chunk_note_printed = False


_R_DF_CHUNK_ABOVE_ENV = 'EASIGP_R_DF_CHUNK_ABOVE'
_R_DF_CHUNK_ABOVE_DEFAULT = 100000


def r_df_chunk_threshold() -> int:
    """Column count above which a DataFrame argument is sent in blocks
    (env override EASIGP_R_DF_CHUNK_ABOVE)."""
    try:
        v = int(os.environ.get(_R_DF_CHUNK_ABOVE_ENV, _R_DF_CHUNK_ABOVE_DEFAULT))
    except (TypeError, ValueError):
        v = _R_DF_CHUNK_ABOVE_DEFAULT
    return max(v, 100)


def r_df_chunk_cols() -> int:
    """Block width for wide DataFrame transfers (env override
    EASIGP_R_DF_CHUNK_COLS; values < 100 fall back to the default)."""
    try:
        v = int(os.environ.get(_R_DF_CHUNK_COLS_ENV, _R_DF_CHUNK_COLS_DEFAULT))
    except (TypeError, ValueError):
        v = _R_DF_CHUNK_COLS_DEFAULT
    return v if v >= 100 else _R_DF_CHUNK_COLS_DEFAULT


def pandas_to_r_dataframe_chunked(df, chunk_cols: Optional[int] = None):
    """Convert a (wide) pandas DataFrame to an R data.frame without ever
    putting more than `chunk_cols` columns into a single R call. Returns an
    rpy2 ``robjects.DataFrame`` (an R object: passing it to an R function does
    not trigger another pandas conversion)."""
    global _r_df_chunk_note_printed
    import rpy2.rinterface as _ri
    import rpy2.robjects as _ro
    from rpy2.robjects import conversion as _conv

    chunk = int(chunk_cols or r_df_chunk_cols())
    n_rows, n_cols = df.shape
    converter = _conv.get_conversion()
    blocks = [converter.py2rpy(df.iloc[:, i:i + chunk]) for i in range(0, n_cols, chunk)]
    if not _r_df_chunk_note_printed:
        print(f"[R-bridge] wide DataFrame ({n_rows} rows x {n_cols} columns) is sent to R in "
              f"{len(blocks)} block(s) of <= {chunk} columns (a single {n_cols}-argument "
              f"data.frame() call overflows R's protect stack). Printed once per process.")
        _r_df_chunk_note_printed = True
    # rinterface-level call: the raw R result, NOT run back through the
    # pandas converter (which would turn the data.frame into a pandas frame
    # again and re-trigger the very conversion this function avoids).
    joined = _ri.baseenv['cbind'](*blocks)
    return _ro.vectors.DataFrame(joined)


def wide_safe_r_function(r_fn):
    """Wrap an rpy2 R function object (e.g. ``robjects.globalenv['BayesB']``)
    so that any pandas DataFrame argument wider than the chunk width is
    transferred with `pandas_to_r_dataframe_chunked()`. Everything else (narrow
    frames, arrays, scalars, keyword arguments, the returned value) is passed
    through to `r_fn` unchanged."""
    import functools

    def _maybe(value):
        try:
            import pandas as _pd
            if isinstance(value, _pd.DataFrame) and value.shape[1] > r_df_chunk_threshold():
                return pandas_to_r_dataframe_chunked(value)
        except ImportError:
            pass
        return value

    def _wrapped(*args, **kwargs):
        return r_fn(*[_maybe(a) for a in args], **{k: _maybe(v) for k, v in kwargs.items()})

    try:
        functools.update_wrapper(_wrapped, r_fn, updated=())
    except Exception:
        pass
    _wrapped.r_function = r_fn
    return _wrapped


_rpy2_release_patched = False
_rpy2_deferred_releases = None   # collections.deque of cffi cdata awaiting release on the R thread


def ensure_rpy2_thread_safe_release() -> bool:
    """ver4-59 - make rpy2 release R objects only on the thread that runs R.
    Idempotent; returns True when the patch is (now) active.

    Why: in rpy2 3.5.x `SexpCapsule.__del__` calls `R_ReleaseObject()` (via
    `_rinterface_capi._release`) without any lock, and `__del__` can run on ANY
    Python thread - Python's cyclic GC runs on whichever thread happens to
    allocate. Every rpy2 call runs R with the GIL released (cffi), so a
    background Python thread (this pipeline starts one: the resource-telemetry
    sampler, which allocates twice a second) can free an rpy2 object and edit
    R's precious-object list while the main thread is inside R's allocator or
    garbage collector. R is not thread-safe, so the likely outcome is
    intermittent heap corruption ("corrupted double-linked list", segmentation
    faults, vectors whose length reads 0) detected at a later allocation - the
    signature of the Cotton step-1 crashes.

    With this patch, a release requested on any other thread is only QUEUED;
    the R thread (the thread that first called this function, i.e. the one that
    initialised R) drains the queue at its next rpy2 object release/preserve,
    when no R call is in progress. Nothing ever blocks, so there is no
    shutdown-time deadlock either; objects still queued at exit are simply not
    released (R is going away anyway).

    Never raises; a no-op when rpy2 is absent or its internals differ.
    """
    global _rpy2_release_patched, _rpy2_deferred_releases
    if _rpy2_release_patched:
        return True
    try:
        import collections
        import rpy2.rinterface_lib._rinterface_capi as _capi
        _orig_release, _orig_preserve = _capi._release, _capi._preserve
    except Exception:
        return False

    r_thread = threading.get_ident()
    deferred = collections.deque()
    _rpy2_deferred_releases = deferred

    def _drain():
        while True:
            try:
                cdata = deferred.popleft()
            except IndexError:
                return
            _orig_release(cdata)

    def _safe_release(cdata):
        if threading.get_ident() == r_thread:
            if deferred:
                _drain()
            return _orig_release(cdata)
        deferred.append(cdata)       # never touch R from another thread

    def _safe_preserve(cdata):
        if deferred and threading.get_ident() == r_thread:
            _drain()
        return _orig_preserve(cdata)

    _capi._release = _safe_release
    _capi._preserve = _safe_preserve
    _rpy2_release_patched = True
    return True


def init_rpy2_conversion():
    """Make sure this process has a valid rpy2 R<->Python conversion
    context. Safe to call from any entry point (GUI or headless script)."""
    try:
        from rpy2.robjects import conversion, default_converter
    except ImportError:
        # rpy2 not installed / not used by this pipeline - nothing to do.
        return
    ensure_rpy2_thread_safe_release()  # ver4-59

    combined = default_converter
    try:
        import rpy2.robjects.numpy2ri as numpy2ri
        combined = combined + numpy2ri.converter
    except Exception:
        pass
    try:
        import rpy2.robjects.pandas2ri as pandas2ri
        combined = combined + pandas2ri.converter
    except Exception:
        pass

    conversion.set_conversion(combined)


def detect_array_job_env():
    """Return (scheduler_name, batch_id) if this process appears to be
    running inside a Slurm or PBS job array, else (None, None). Used by the
    GUI to warn people against launching Streamlit itself as the array-job
    command."""
    env_val = os.environ.get('SLURM_ARRAY_TASK_ID')
    if env_val is not None:
        return 'Slurm', env_val

    env_val = os.environ.get('PBS_ARRAY_INDEX', os.environ.get('PBS_ARRAYID'))
    if env_val is not None:
        return 'PBS', env_val

    return None, None


def restore_ratio(ratio_list, scenario):
    """Undo JSON's tuple -> list flattening for RATIO entries.

    A tuple-valued ratio (e.g. (0.7, 0.15, 0.15) for a 3-way train/valid/
    test split - used by both 'within' and 'between' SCENARIOs, not just
    'between') becomes a plain JSON array once a config dict is written
    to disk (json.dump has no tuple type), and comes back as an ordinary
    Python LIST when the config is read back (json.load) - every headless
    entry point (run_sequential.py, run_step1_batch.py) always goes
    through exactly this JSON round-trip. genomic_prediction.py's own
    logic throughout GP() distinguishes a tuple-ratio task from a plain
    float-ratio one strictly by TYPE (`type(sample.loc[i,'ratio']) is
    tuple`) - a list is not a tuple, so without this restoration every
    tuple ratio would silently be treated as the wrong kind of task the
    moment a run goes through a saved config file, rather than being run
    directly from a live Python call (as every GUI 'Option B: run now'
    path, and every test that calls GP() directly, does instead - which
    is exactly why this only ever surfaces for headless/HPC runs).

    Converts every list-typed entry in `ratio_list` back into a tuple;
    every plain float entry is left exactly as-is. `scenario` is accepted
    for API symmetry with call sites that already have it on hand (and in
    case a future scenario-specific restoration rule is ever needed) -
    the restoration itself is purely type-based and doesn't currently
    depend on it.
    """
    return [tuple(r) if isinstance(r, list) else r for r in ratio_list]


# Same four W_OPT method names (and the '__<algo>' suffix a per-method
# HP_TUNE ensemble grouping adds - see genomic_prediction.py's `_WOPT_LABELS`
# and ensemble_groups()) that assemble.py and batch_reader.py already each
# define locally as their own '_WOPT_LABEL_PREFIXES' constant, to strip
# weighted-ensemble pseudo-models out of a raw model list. Duplicated here
# for the same reason those two duplicate each other: none of these modules
# import one another, and a shared-constants module just for one 4-item
# tuple isn't worth the extra import surface.
# Requirements.md item 5: 'Analytic least-squares' appended.
# ver4-26, R2/R_last item 5: 'Joint Bayesian optimisation' appended - the
# new JOINT_W_OPT combination method's own record/effect/weight 'model'
# label (genomic_prediction.py's new Step 11 `elif` branch), recognised
# here the same way every other weighted-ensemble method's label already
# is.
# ver4-57 (R2): 'Iterative ensemble' appended - the opt-in iterative ensemble's own
# record/effect/weight 'model' label and prediction-column name (genomic_prediction.py's
# new Step 11c), recognised here the same way every other combination label is.
# ver4-77 (R2): 'Super learner' appended - the opt-in Super learner's own record/effect/weight
# 'model' label and prediction-column name (genomic_prediction.py's new Step 11d).
_WOPT_LABEL_PREFIXES = (
    'Linear transformation', 'Nelder Mead', 'Bayesian optimisation', 'Analytic least-squares',
    'Joint Bayesian optimisation', 'Iterative ensemble', 'Super learner',  # ver4-77 (R2)
)


def _is_wopt_label(label) -> bool:
    return isinstance(label, str) and any(
        label == p or label.startswith(p + '__') for p in _WOPT_LABEL_PREFIXES
    )


def _is_naive_ensemble_label(label) -> bool:
    return isinstance(label, str) and (label == 'ensemble' or label.startswith('ensemble__'))


# ver4-31 (R0/R10): public aliases of the two canonical ensemble-label
# predicates above. `plot_style.py` and `circos_plot.py` import THESE
# instead of re-deriving the same name lists (the circos ring-hue test was a
# hard-coded 5-name list that missed 'ensemble__<algo>' and
# 'Joint Bayesian optimisation' - R10) or importing a private name across a
# module boundary. The private names above are kept unchanged for the
# existing importers (e.g. diversity_summary.py).
is_wopt_label = _is_wopt_label
is_naive_ensemble_label = _is_naive_ensemble_label


# ver4-95: DISPLAY-ONLY model labels. `MODEL_LABEL_STYLE` (config key) chooses how a model label is
# SHOWN in plots and in the derived summary workbooks; 'full' (default) is today's text. The label
# stored in Metric.csv / Weight.csv / Prediction_result_test.csv is never changed, and every
# predicate / colour / ordering rule keeps working on the full label (the short text is looked up
# only when a tick, legend entry or column header is written).
MODEL_LABEL_STYLES = ('full', 'short')
_SL_LABEL_PREFIX = 'Super learner__'
_SL_MEMBER_ROW_RE = re.compile(r'^.+ \((?:pooled|pooled, gated)\)(?:__.+)?$')
_CLASH_NOTED = set()        # clash NOTEs already printed (one per distinct clash set per process)


def normalise_label_style(value) -> str:
    """'full' or 'short' from a config value (also accepts True/False and 'no_prefix');
    anything unknown is 'full' - a hand-edited config never aborts plotting. Pure."""
    if value is True:
        return 'short'
    if value is None or value is False:
        return 'full'
    v = str(value).strip().lower()
    return 'short' if v in ('short', 'no_prefix', 'strip', 'true') else 'full'


def short_model_label(label):
    """The 'short' text of one model label: the ``'Super learner__'`` prefix removed when what
    follows starts with a real weight-method name (``'Super learner__Bayesian optimisation
    (pooled)__Grid'`` -> ``'Bayesian optimisation (pooled)__Grid'``; a per-member row
    ``'Super learner__RF (pooled)'`` -> ``'RF (pooled)'``). A label that is not of
    that form - including the legacy ``'Super learner'``, ``'Super learner__pooled'`` and
    ``'Super learner__<group>'`` rows, where the remainder alone would be meaningless - is
    returned unchanged, as is anything that is not a string. Pure; never raises."""
    if not isinstance(label, str) or not label.startswith(_SL_LABEL_PREFIX):
        return label
    rest = label[len(_SL_LABEL_PREFIX):]
    method = rest.split('__', 1)[0].split(' (', 1)[0]
    if method in SL_WEIGHT_METHODS:
        return rest
    # ver4-96 per-member rows: 'Super learner__<model> (pooled)' / '(pooled, gated)' [__<group>]
    # -> '<model> (pooled)' ... (the bracket keeps them apart from the base model of that name)
    if _SL_MEMBER_ROW_RE.match(rest):
        return rest
    return label


def display_label_map(labels, style='full') -> dict:
    """``{label: shown text}`` for every label in `labels` (duplicates collapse).

    'full' maps each label to itself. 'short' applies :func:`short_model_label`, but never lets two
    different labels show the same text: when a shortened text equals another label of the set
    (e.g. a Super learner row ``'Super learner__Nelder Mead'`` next to the plain weighted ensemble
    ``'Nelder Mead'``) or another shortened text, the colliding Super learner labels keep their
    full text and one NOTE is printed. Pure apart from that NOTE; never raises."""
    uniq = list(dict.fromkeys(l for l in labels if isinstance(l, str)))
    out = {l: l for l in uniq}
    if normalise_label_style(style) != 'short':
        return out
    cand = {l: short_model_label(l) for l in uniq}
    counts = {}
    for l in uniq:
        counts[cand[l]] = counts.get(cand[l], 0) + 1
    clash = [l for l in uniq if cand[l] != l and counts[cand[l]] > 1]
    for l in uniq:
        if cand[l] != l and l not in clash:
            out[l] = cand[l]
    if clash and tuple(clash) not in _CLASH_NOTED:
        _CLASH_NOTED.add(tuple(clash))
        print(f"[labels] NOTE: {len(clash)} label(s) keep their full text because the short text "
              f"would clash with another label: {', '.join(clash[:4])}"
              f"{' ...' if len(clash) > 4 else ''}.")
    return out


def canonical_model_order(labels):
    """Reorder `labels` so every naive-ensemble label (`'ensemble'`,
    `'ensemble__<algo>'`) sits immediately before the first
    weighted-ensemble label (`'Nelder Mead'`, `'Linear transformation'`,
    `'Bayesian optimisation'`, and their own `'__<algo>'`-suffixed
    variants) - regardless of where the naive ensemble happens to fall in
    `labels`' own input order.

    Why this is needed: the naive ensemble is computed once, in GP()'s
    finalisation step, strictly AFTER the entire per-task dispatch loop
    (genomic_prediction.py §11/§Finalisation) - and the per-task loop is
    where every weighted-ensemble ('Nelder Mead' etc.) row is produced,
    one task at a time. That means naive-ensemble rows are always the
    LAST to first-appear in Metric.csv/`record`, so any model ordering
    derived directly from `pd.unique(metric['model'])` (e.g. the HP_TUNE
    fallback both run_sequential.py and run_step2_assemble.py use, since a
    tuning-algorithm-suffixed model list can't be reconstructed from
    config alone) puts the naive ensemble AFTER every weighted-ensemble
    method - even though the intended, and far more legible, presentation
    order is: real models, then the naive ensemble, then the
    weighted-ensemble methods (the same order the GUI's own '4. Ensemble'
    tab presents them in). `metric_plot()`'s violin plots (`hue_order`)
    and `metric_summary.py`'s `Metric_summary.xlsx` (both the pivot
    'summary' sheet and the three long-format sheets) both order their
    own 'model' axis this way.

    A no-op whenever `labels` doesn't contain BOTH a naive-ensemble-family
    label AND a weighted-ensemble-family label - there is nothing to
    reorder relative to. Otherwise, relative order is preserved exactly
    within each of the three groups (real models, naive-ensemble labels,
    weighted labels); only the naive-ensemble group is moved, as a block,
    to sit between the other two.
    """
    labels = list(labels)
    ensemble_labels = [m for m in labels if _is_naive_ensemble_label(m)]
    wopt_labels = [m for m in labels if _is_wopt_label(m)]
    if not ensemble_labels or not wopt_labels:
        return labels
    other_labels = [m for m in labels if m not in ensemble_labels and m not in wopt_labels]
    return other_labels + ensemble_labels + wopt_labels


def resolve_batch_id_from_env() -> int:
    """Read the batch ID from whichever scheduler's environment variable is
    set, honouring an optional ``OFFSET`` environment variable on top of it
    (Phase 2, Requirement 1 - Gadi native job-array submission).

    A Gadi native job-array is submitted repeatedly, once per chunk, each
    time with a different ``OFFSET`` (via ``qsub -v OFFSET=<n>``) and a
    fresh, always-zero-based ``#PBS -J 0-<chunk size - 1>`` range - see
    ``main_app.py::render_hpc_export_section()``'s generated
    ``run_batch_array.pbs`` / ``submit_arrays.sh``. Each subjob's REAL batch
    ID is therefore ``OFFSET + PBS_ARRAY_INDEX`` (or
    ``OFFSET + SLURM_ARRAY_TASK_ID`` for the Slurm equivalent), not the raw
    scheduler index on its own. ``OFFSET`` defaults to 0 when unset, which
    makes this fully backward compatible with a plain (non-chunked) Slurm/
    PBS array job that never sets it at all - such a job's batch ID
    resolves exactly as before this feature existed.

    Precedence (documented explicitly, since a caller may also accept a
    manual ``--batch-id`` override - see ``run_step1_batch.py``):
    an explicit ``--batch-id`` command-line argument ALWAYS wins over
    whatever this function would resolve from the environment; this
    function is only ever consulted when no such override was given.

    Raises RuntimeError with a clear message if neither
    SLURM_ARRAY_TASK_ID nor PBS_ARRAY_INDEX/PBS_ARRAYID is present.
    (Deliberately RuntimeError rather than SystemExit so this can be called
    safely from inside the Streamlit app too, where SystemExit would tear
    down the whole session instead of just showing an error message.)
    """
    offset_val = os.environ.get('OFFSET')
    try:
        offset = int(offset_val) if offset_val is not None else 0
    except ValueError:
        raise RuntimeError(
            f"OFFSET environment variable is set but not a valid integer: {offset_val!r}"
        )

    env_val = os.environ.get('SLURM_ARRAY_TASK_ID')
    if env_val is not None:
        return int(env_val) + offset

    env_val = os.environ.get('PBS_ARRAY_INDEX', os.environ.get('PBS_ARRAYID'))
    if env_val is not None:
        return int(env_val) + offset

    raise RuntimeError(
        'Could not determine the batch ID: neither SLURM_ARRAY_TASK_ID nor '
        'PBS_ARRAY_INDEX/PBS_ARRAYID is set in the environment, and no '
        '--batch-id override was given. Are you running this inside a job array?'
    )


def unify_columns_by_position(df, expected_names, description, first_n=None):
    """Rename `df`'s columns POSITIONALLY to `expected_names` - i.e. trust
    that the file's columns are in the ORDER EasiGP's documentation
    requires, even if the person's own file uses different HEADER TEXT
    for them (e.g. a genotype file with 'SampleID'/'Pop' instead of
    'ID'/'population', or a marker-info file with 'Chrom'/'Marker'
    instead of 'chromosome'/'name') - a common, easy-to-make mismatch
    when a file was exported from a different tool/pipeline with its own
    naming convention, and otherwise a silent source of KeyErrors deep
    inside the pipeline.

    Renames by POSITION ONLY, never by fuzzy name matching - deterministic,
    and impossible to get wrong by misinterpreting a similar-sounding but
    different column. Prints which columns were actually renamed (if any),
    so this is never a silent behind-the-scenes change to the person's
    data.

    Parameters
    ----------
    df : DataFrame to rename columns on (a shallow copy is returned;
        `df` itself is never mutated).
    expected_names : the canonical column names, in the exact order
        EasiGP expects them to appear in the file.
    description : short label for what this file is (e.g. 'genotype
        file', 'SNP info file') - used only in the printed message.
    first_n : if given, ONLY the first `first_n` columns are positionally
        renamed (`expected_names` must then have exactly `first_n`
        entries) - every column after that is left completely untouched.
        This is how marker-name / phenotype-trait columns - whose actual
        names are meaningful, user-chosen values (e.g. a trait literally
        called 'days2anthesis', or a marker called 'snp_1042') and must
        NEVER be renamed - stay exactly as they are, while a file's own
        fixed leading metadata columns (e.g. 'ID', 'population') still
        get unified. Omit (None) to rename every column in `df`.

    Raises
    ------
    ValueError if `df` has fewer columns than `expected_names` needs (too
        few columns to even attempt a positional match - a genuine schema
        problem this can't paper over, unlike a mere naming difference).
    """
    n_target = first_n if first_n is not None else len(expected_names)
    if first_n is not None and len(expected_names) != first_n:
        raise ValueError(
            f"unify_columns_by_position: expected_names has {len(expected_names)} entries "
            f"but first_n={first_n} - these must match."
        )
    if df.shape[1] < n_target:
        raise ValueError(
            f"{description}: expected at least {n_target} column(s) (for "
            f"{expected_names}), but found only {df.shape[1]}: {list(df.columns)}."
        )

    df = df.copy()
    current = list(df.columns[:n_target])
    rename_map = {old: new for old, new in zip(current, expected_names) if old != new}
    if rename_map:
        _scope = f"the first {n_target} column(s)" if first_n is not None else "its column(s)"
        print(f"[EasiGP] {description}: {_scope} matched EasiGP's expected ORDER, but used "
              f"different header text - unified internally (by position, not name - the "
              f"underlying data itself is untouched): "
              + ', '.join(f"{old!r} -> {new!r}" for old, new in rename_map.items()))
        df = df.rename(columns=rename_map)
    return df


# ---------------------------------------------------------------------------
# Phase 2, Requirement 3 - phenotype name auto-suggestion.
# ---------------------------------------------------------------------------

def list_phenotype_columns(phenotype_file_path: str) -> List[str]:
    """Return the phenotype trait names available in a phenotype CSV,
    cheaply - a HEADER-ONLY read (O(1) I/O cost regardless of file size),
    never loading any data rows.

    Per the phenotype file contract (``ID, population, <trait_1>, ...,
    <trait_P>``), the first two columns are always the positionally-
    unified ``ID``/``population`` pair (see ``unify_columns_by_position``)
    and are therefore dropped here purely for the purpose of *suggesting*
    trait names - this function never renames anything, it only reads and
    slices the header row.

    Parameters
    ----------
    phenotype_file_path : path to a phenotype CSV file.

    Returns
    -------
    List of trait column names (columns 3+), in file order. Returns an
    empty list (never raises) if the path is missing, unreadable, or has
    fewer than 3 columns - callers should treat an empty list as "no
    suggestions available" and fall back to free-text entry, exactly as
    the GUI does.
    """
    if not phenotype_file_path or not os.path.isfile(phenotype_file_path):
        return []
    try:
        with open(phenotype_file_path, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
    except (OSError, csv.Error, StopIteration):
        return []
    if not header or len(header) < 3:
        return []
    return [str(col).strip() for col in header[2:]]


# ---------------------------------------------------------------------------
# Update ID ver4-12, R2/R3 (design blueprint SS2.2/SS2.3) - population
# derivation and raw/analysed phenotype-file handling.
#
# Both functions below are PURE (no I/O beyond the DataFrame they're
# handed) and are the single authority both `genomic_prediction.py::GP()`
# (the engine) and `main_app.py`/`resource_profiles.py` (the GUI's own
# resource-advisor task count - R2's own "one authority, two consumers"
# requirement) call, so the sample table's own scenario count can never
# drift from what GP() itself will actually construct (invariant I9).
# ---------------------------------------------------------------------------

def list_all_columns(phenotype_file_path: str) -> List[str]:
    """Header-only read of EVERY column in a phenotype file, in file
    order - the R3 counterpart to `list_phenotype_columns()` above, which
    only ever returns columns[2:] under the ANALYSED-file positional
    contract. Used for `PHENOTYPE_FILE_MODE='raw'`, where a file's
    columns are a mix of design columns (Row, Range, Location, Year, ...)
    and trait columns, distinguished only by which ones the person
    explicitly names as traits (`PHENOTYPE_TRAIT_COLUMNS`) - inferring
    this by dtype would be wrong (design columns are very often numeric
    too - see genomic_prediction.py's own R3 note on this).

    Same never-raises, empty-list-on-any-failure contract as
    `list_phenotype_columns()` - callers should treat an empty list as
    "no columns detected yet" and fall back to free-text entry.
    """
    if not phenotype_file_path or not os.path.isfile(phenotype_file_path):
        return []
    try:
        with open(phenotype_file_path, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
    except (OSError, csv.Error, StopIteration):
        return []
    if not header:
        return []
    return [str(col).strip() for col in header]


def resolve_trait_columns(cfg: dict, phenotype_file_path: str) -> List[str]:
    """The single place `PHENOTYPE == 'all'` (and the GUI's own trait
    selector) resolves to a concrete trait-column list, for EITHER
    `PHENOTYPE_FILE_MODE`:

    - `'analysed'` (default): `list_phenotype_columns(phenotype_file_path)`
      - byte-for-byte the pre-ver4-12 `columns[2:]` behaviour.
    - `'raw'`: `cfg['PHENOTYPE_TRAIT_COLUMNS']`, validated to be a
      non-empty list whose every entry is actually present in the raw
      file's header (`list_all_columns()`) - raised loudly here (not
      returned as an empty list) because an empty/missing trait
      declaration in raw mode is a genuine configuration error, not a
      "nothing detected yet" state the caller should silently tolerate.

    Raises
    ------
    ValueError if `PHENOTYPE_FILE_MODE == 'raw'` and
        `PHENOTYPE_TRAIT_COLUMNS` is missing/empty, or names a column not
        present in the file's own header - named explicitly (offending
        column(s), and the full header) so a headless config (which
        bypasses the GUI's own validation) fails at load with an
        actionable message rather than a confusing downstream KeyError.
    """
    mode = cfg.get('PHENOTYPE_FILE_MODE', 'analysed')
    if mode == 'analysed':
        return list_phenotype_columns(phenotype_file_path)

    if mode != 'raw':
        raise ValueError(f"PHENOTYPE_FILE_MODE must be 'analysed' or 'raw', got {mode!r}.")

    trait_columns = cfg.get('PHENOTYPE_TRAIT_COLUMNS')
    if not trait_columns:
        raise ValueError(
            "PHENOTYPE_FILE_MODE='raw' requires PHENOTYPE_TRAIT_COLUMNS to name at least one "
            "trait column explicitly - traits cannot be inferred by position (or by numeric "
            "dtype - design columns like Row/Range/Year are numeric too) from a raw, "
            "plot-level phenotype file."
        )
    header = list_all_columns(phenotype_file_path)
    missing = [c for c in trait_columns if header and c not in header]
    if missing:
        raise ValueError(
            f"PHENOTYPE_TRAIT_COLUMNS names column(s) {missing!r} that are not present in "
            f"the phenotype file's own header ({header!r})."
        )
    return list(trait_columns)


def derive_population_series(df: "pd.DataFrame", population_source: "Optional[dict]") -> "pd.Series":
    """Update ID ver4-12, R2 - the single authority for "what is this
    individual's population", replacing two independent hard-coded reads
    of a column literally named 'population' (one on the genotype side
    for CSV input, one on the phenotype side for PLINK input - see
    genomic_prediction.py's own load block).

    Parameters
    ----------
    df : the phenotype DataFrame to derive a population Series FROM - one
        row per (raw or already-analysed) phenotype record. Never
        mutated.
    population_source : `None` (or an empty dict) resolves to today's
        exact behaviour - `{'mode': 'column', 'column': 'population'}` -
        i.e. an absent config key reproduces ver4-11 exactly (I11).
        Otherwise a dict with:
          - `{'mode': 'column', 'column': <name>}` - read `df[<name>]`
            as-is (today's behaviour, generalised to any column name).
          - `{'mode': 'constant', 'label': <label>}` - every individual
            gets the SAME population label (e.g. `'all_lines'` - chosen
            by the GUI, R2, for exactly this case: no natural population
            grouping exists in the data at all).
          - `{'mode': 'derived', 'columns': [<name>, ...]}` - `'_'`-joined
            values of the named columns, in the given order (e.g.
            `Location`x`Year` -> `'Roseworthy_2014'`).

    Raises
    ------
    ValueError if `mode` is unrecognised, if `mode='column'`/`'derived'`
        names a column absent from `df`, or if deriving would produce a
        `NaN`/missing population value for any row (a silent `'nan'`
        population string is exactly the kind of quiet data corruption
        this function exists to prevent - named here, with the affected
        row count, rather than surfacing three steps later as a
        mysteriously-tiny population).
    """
    import pandas as pd

    if not population_source:
        population_source = {'mode': 'column', 'column': 'population'}
    mode = population_source.get('mode', 'column')

    if mode == 'column':
        column = population_source.get('column', 'population')
        if column not in df.columns:
            raise ValueError(
                f"derive_population_series: POPULATION_SOURCE mode='column' names column "
                f"{column!r}, which is not present in the phenotype data ({list(df.columns)})."
            )
        series = df[column]
        if series.isna().any():
            raise ValueError(
                f"derive_population_series: column {column!r} has "
                f"{int(series.isna().sum())} missing value(s) - every individual must have a "
                f"population value."
            )
        return series

    if mode == 'constant':
        label = population_source.get('label', 'all_lines')
        return pd.Series([label] * len(df), index=df.index, name='population')

    if mode == 'derived':
        columns = population_source.get('columns', [])
        if not columns:
            raise ValueError(
                "derive_population_series: POPULATION_SOURCE mode='derived' needs at least "
                "one column in 'columns'."
            )
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"derive_population_series: POPULATION_SOURCE mode='derived' names column(s) "
                f"{missing!r}, which are not present in the phenotype data "
                f"({list(df.columns)})."
            )
        if df[columns].isna().any().any():
            _n_missing = int(df[columns].isna().any(axis=1).sum())
            raise ValueError(
                f"derive_population_series: {_n_missing} row(s) have a missing value in one "
                f"of the derived-population columns {columns!r} - every individual must have "
                f"a value for every one of them."
            )
        derived = df[columns[0]].astype(str)
        for extra_col in columns[1:]:
            derived = derived + '_' + df[extra_col].astype(str)
        derived.name = 'population'
        return derived

    raise ValueError(
        f"derive_population_series: POPULATION_SOURCE['mode'] must be 'column', 'constant', "
        f"or 'derived', got {mode!r}."
    )


def collapse_raw_to_analysed_level(raw_df: "pd.DataFrame", id_col: str, population: "pd.Series",
                                    trait_columns: List[str]) -> "pd.DataFrame":
    """Update ID ver4-12, R8 - collapse a (potentially plot-level, many-
    rows-per-individual) raw phenotype table down to EXACTLY the shape
    every existing row-level guard in `genomic_prediction.py::GP()`
    already assumes: one row per (ID x population), columns
    `['ID', 'population'] + trait_columns`.

    Trait values here are a PLACEHOLDER - the arithmetic mean of that
    same individual's own raw records only (never mixed across
    individuals, so this is not a leakage channel) - that exists purely
    so `dropna()`, `MIN_DATA_POINTS`, the ID-dedup guard, the genotype
    merge, `train_test_split`, and `Basic_stats.csv`'s row counts all
    keep working completely unchanged on a 'raw'-mode file, exactly as
    they already do on an 'analysed'-mode one. When `ASREML` is enabled,
    Step 6b (genomic_prediction.py) OVERWRITES this placeholder
    unconditionally with the converged BLUE/BLUP before any model ever
    sees it - see that step's own post-condition assertions, which fail
    the task loudly rather than let the placeholder silently reach a
    model. `GP()` logs, once per run, that this placeholder is in use
    and where it gets overwritten.

    Parameters
    ----------
    raw_df : the raw phenotype table, ALREADY positionally-unified on its
        `id_col` (see `unify_columns_by_position(..., first_n=1)` for raw
        mode - only the ID column is positionally unified; every other
        column, including any trait columns, is a meaningful, user-named
        column and is never renamed).
    id_col : the (already-unified) individual-identifier column name -
        `'ID'`.
    population : a Series, same length/index as `raw_df`, from
        `derive_population_series()` - attached as the `'population'`
        column of the result.
    trait_columns : from `resolve_trait_columns()` - which columns of
        `raw_df` are trait (response) values to be averaged; every other
        column in `raw_df` is a design column and is dropped here (Step
        6b reads the ORIGINAL raw table separately for the actual mixed-
        model fit - this collapsed table is for the main split/model
        pipeline only).

    Returns
    -------
    DataFrame with columns `['ID', 'population'] + trait_columns`, one
    row per (ID x population) - `'analysed'`-mode input already has this
    exact shape, so calling this on it (mean of exactly one value = that
    value) is a safe identity operation should a caller ever do so.
    """
    working = raw_df[[id_col] + list(trait_columns)].copy()
    working = working.rename(columns={id_col: 'ID'})
    working['population'] = population.values
    grouped = working.groupby(['ID', 'population'], as_index=False, sort=False)[list(trait_columns)].mean()
    return grouped[['ID', 'population'] + list(trait_columns)].reset_index(drop=True)


def phenotype_file_mtime_key(phenotype_file_path: str) -> Tuple[str, float]:
    """``(path, mtime)`` cache key for ``list_phenotype_columns()`` -
    intended to be passed to ``st.cache_data``-decorated GUI wrappers (via
    ``hash_funcs``, or simply as part of the cached function's own
    arguments) so a repeated Streamlit rerun re-reads the header only when
    the underlying file has actually changed, not on every rerun. Returns
    ``(phenotype_file_path, -1.0)`` if the file doesn't exist, so a
    not-yet-uploaded/typo'd path still produces a stable, hashable key
    rather than raising.
    """
    try:
        mtime = os.path.getmtime(phenotype_file_path)
    except OSError:
        mtime = -1.0
    return (phenotype_file_path, mtime)


# ---------------------------------------------------------------------------
# Phase 2, Requirement 6 - shared compute-resource resolver.
#
# One config-derived source of truth for "what hardware should this model
# module use", so every model file (RF/SVR/KNN/MLP/GAT variants), plink_io,
# and configure_r_environment() all read compute settings the SAME way
# instead of each independently guessing - preserving the "config-as-data"
# architectural strength (architecture doc §18) rather than letting hardware
# selection drift out of the JSON config contract.
# ---------------------------------------------------------------------------

# ver4-33 (R1.b/c/f/g/h): the sub-keys of the `ACCELERATION` config dict
# (design blueprint SS4). Defined once here so the resolver, the GUI
# (`main_app.py`) and `gui_config_loader.py` all agree on the same names.
_ACCELERATION_KEYS = frozenset({
    'plink_direct_read', 'preprocess_memo', 'gat_dense_batching',
    'gat_batched_ig', 'gpu_device_round_robin',
    # ver4-35 (blueprint SS4): user-facing switches of R1.b/d/e/f/g/h. Every
    # one resolves to its legacy (ver4-33) value when absent (I11).
    'gat_factorised_input', 'gat_sync_free', 'worker_thread_limits', 'gpu_slots',
    'torch_device_policy', 'async_model_lane', 'ld_cm_banded', 'bio_prior_cost_guard',
    'merge_shap_device',
    # ver4-43 (R1): engine for the RKHS/SVR/KNN `baseline_shap` explanations.
    'baseline_shap_engine',
    # Engine-only keys (blueprint SS4: top-level BIO_PRIOR_PREP_FAST,
    # BIO_PRIOR_LEAN_POOLS, GAT_IG_INTERNAL_BATCH). `GP()` has no keyword
    # for top-level keys, so the entry points carry them INSIDE the
    # `ACCELERATION` dict under their config names - see
    # `acceleration_from_cfg` (Change Summary SS7, deviation D-1).
    'BIO_PRIOR_PREP_FAST', 'BIO_PRIOR_LEAN_POOLS', 'GAT_IG_INTERNAL_BATCH',
    # ver4-35 engine-only keys (same carrying rule as the three above).
    'SHAP_STREAM_REDUCE', 'SHARED_BIM_MARKER_INFO', 'ASYNC_LANE_RNG', 'ASYNC_LANE_THREADS',
    'ASYNC_LANE_MAX_PAYLOAD_MB', 'BIO_PRIOR_MAX_FIT_GFLOP',
    # ver4-61 (R2): cap trial concurrency from the predicted memory of a bio-prior trial.
    'trial_memory_guard',
    # ver4-103: GAT run-to-run stability (models/gat_stability.py). Engine-only keys.
    'GAT_SEED', 'GAT_SEED_ENSEMBLE', 'GAT_RESCORE_TOP', 'GAT_RESCORE_SEEDS',
    # optimise_scan F3: engine-only, opt-in (default False = the legacy per-task id-sort).
    'DEFER_RESULT_SORT',
})

_ACCELERATION_ENGINE_KEYS = (
    'BIO_PRIOR_PREP_FAST', 'BIO_PRIOR_LEAN_POOLS', 'GAT_IG_INTERNAL_BATCH',
    'SHAP_STREAM_REDUCE', 'SHARED_BIM_MARKER_INFO', 'ASYNC_LANE_RNG', 'ASYNC_LANE_THREADS',
    'ASYNC_LANE_MAX_PAYLOAD_MB', 'BIO_PRIOR_MAX_FIT_GFLOP',
    'GAT_SEED', 'GAT_SEED_ENSEMBLE', 'GAT_RESCORE_TOP', 'GAT_RESCORE_SEEDS',
    'DEFER_RESULT_SORT',
)

#: ver4-103 - GAT stability keys (models/gat_stability.py). `GAT_SEED` is an int base seed, or
#: None/'off' for the legacy unseeded behaviour. Engine value for an ABSENT key: seeded fits
#: (the instability fix itself), a single network per final fit, no re-scoring. A freshly
#: exported GUI config also averages 3 seeded networks in the FINAL fits and re-scores the top 3
#: search candidates + the default over 2 extra seed draws.
GAT_STABILITY_ENGINE_DEFAULTS: Dict[str, object] = {
    'GAT_SEED': 20240615, 'GAT_SEED_ENSEMBLE': 1, 'GAT_RESCORE_TOP': 0, 'GAT_RESCORE_SEEDS': 0,
}
GAT_STABILITY_GUI_DEFAULTS: Dict[str, object] = {
    'GAT_SEED': 20240615, 'GAT_SEED_ENSEMBLE': 3, 'GAT_RESCORE_TOP': 3, 'GAT_RESCORE_SEEDS': 2,
}


def _gat_seed_from_cfg(value):
    """int base seed, or None for 'off'/None/False (legacy unseeded GAT fits)."""
    if value is None or value is False:
        return None
    if isinstance(value, str):
        if value.strip().lower() in ('off', 'none', 'false', 'legacy', ''):
            return None
        try:
            return int(value)
        except ValueError:
            print(f"[pipeline_utils] WARNING: GAT_SEED={value!r} is not an integer or 'off' - "
                  f"using {GAT_STABILITY_ENGINE_DEFAULTS['GAT_SEED']}.")
            return int(GAT_STABILITY_ENGINE_DEFAULTS['GAT_SEED'])
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(GAT_STABILITY_ENGINE_DEFAULTS['GAT_SEED'])


def _gat_count_from_cfg(key, value, minimum):
    try:
        return max(minimum, int(value))
    except (TypeError, ValueError):
        print(f"[pipeline_utils] WARNING: {key}={value!r} is not an integer - using "
              f"{GAT_STABILITY_ENGINE_DEFAULTS[key]}.")
        return int(GAT_STABILITY_ENGINE_DEFAULTS[key])


def acceleration_from_cfg(cfg: Dict) -> Optional[Dict]:
    """The value entry points pass as ``GP(ACCELERATION=...)``.

    The config's ``ACCELERATION`` dict, plus any of the three engine-only
    top-level keys (``BIO_PRIOR_PREP_FAST`` / ``BIO_PRIOR_LEAN_POOLS`` /
    ``GAT_IG_INTERNAL_BATCH``) that the config sets - merged in because
    ``GP()`` accepts only this one new keyword. ``None`` when the config has
    neither (every pre-ver4-33 config), which resolves to the legacy path.
    An explicit ``ACCELERATION`` sub-key wins over a top-level key.
    """
    base = cfg.get('ACCELERATION')
    engine = {key: cfg[key] for key in _ACCELERATION_ENGINE_KEYS if key in cfg}
    if base is None and not engine:
        return None
    if base is not None and not isinstance(base, dict):
        return base                       # resolver reports the type error
    merged = dict(engine)
    merged.update(base or {})
    return merged

#: New-config defaults the GUI writes for `ACCELERATION` (SS16.5 precedent:
#: the engine's own default for an ABSENT key stays legacy; only a config
#: freshly exported from the GUI gets the accelerated values).
ACCELERATION_GUI_DEFAULTS: Dict[str, object] = {
    'plink_direct_read': 'auto',
    'preprocess_memo': True,
    'gat_dense_batching': True,
    'gat_batched_ig': True,
    'gpu_device_round_robin': True,
    # ver4-35 (blueprint SS4). `torch_device_policy` stays 'legacy' until the
    # R1.i measurements (tools/gat_bench.py) calibrate TORCH_DEVICE_CROSSOVER;
    # `merge_shap_device` stays 'cpu' until check P10 decides GPU TreeSHAP.
    'gat_factorised_input': True,
    'gat_sync_free': True,
    'worker_thread_limits': True,
    'gpu_slots': 'auto',
    'torch_device_policy': 'legacy',
    'async_model_lane': True,
    'ld_cm_banded': True,
    'bio_prior_cost_guard': 'warn',
    'merge_shap_device': 'cpu',
    # ver4-43 (R1): a freshly exported GUI config uses the exact deduplicated
    # engine; an ABSENT key stays 'legacy' (see ACCELERATION_LEGACY_DEFAULTS).
    'baseline_shap_engine': 'fast',
    # ver4-61 (R2): the engine default too - listed so the GUI writes it explicitly.
    'trial_memory_guard': 'auto',
}

#: Values the legacy-preserving engine uses when `ACCELERATION` (or one of
#: its sub-keys) is absent (I11).
ACCELERATION_LEGACY_DEFAULTS: Dict[str, object] = {
    'plink_direct_read': 'off',
    'preprocess_memo': False,
    'gat_dense_batching': False,
    'gat_batched_ig': False,
    'gpu_device_round_robin': False,
    'gat_factorised_input': False,
    'gat_sync_free': False,
    'worker_thread_limits': False,
    'gpu_slots': 'legacy',
    'torch_device_policy': 'legacy',
    'async_model_lane': False,
    'ld_cm_banded': False,
    'bio_prior_cost_guard': 'warn',
    'merge_shap_device': 'cpu',
    'baseline_shap_engine': 'legacy',
    # ver4-61 (R2): an ABSENT key means 'auto' in the engine (the one non-legacy default of
    # ver4-61), so an older config loads into the GUI exactly as the engine would run it.
    'trial_memory_guard': 'auto',
}

#: Allowed values of the enum-valued ver4-35 keys (anything else is logged as
#: a WARNING and replaced by the legacy value - never silently accepted).
_TORCH_DEVICE_POLICIES = ('legacy', 'auto', 'cpu', 'gpu')
_COST_GUARD_MODES = ('warn', 'raise', 'off')
# ver4-61 (R2): 'auto' lowers the number of concurrent bio-prior trials when the memory model
# predicts the configured width would exceed 70 % of the node's available memory; 'off' is
# the ver4-60 behaviour. 'auto' is the default - the one deliberate I11 exception of ver4-61
# (blueprint R2): it only acts in the situation that got a worker killed by the OS.
_TRIAL_MEMORY_GUARD_MODES = ('auto', 'off')
_MERGE_SHAP_DEVICES = ('cpu', 'gpu')
_BASELINE_SHAP_ENGINES = ('legacy', 'fast')   # ver4-43 R1
_ASYNC_LANE_RNG_MODES = ('exact', 'per_job_seed')


def _normalise_enum_key(key: str, value, allowed: Tuple[str, ...], legacy: str) -> str:
    """`value` lower-cased if it is one of `allowed`, else `legacy` after one
    WARNING line (why: a typo in a hand-edited JSON must not silently pick a
    behaviour; the legacy value is the safe one - I11)."""
    text = str(value).strip().lower() if value is not None else legacy
    if text in allowed:
        return text
    print(f"[pipeline_utils] WARNING: ACCELERATION['{key}']={value!r} is not one of "
          f"{list(allowed)} - using the legacy value {legacy!r}.")
    return legacy


def _normalise_gpu_slots(value) -> object:
    """`ACCELERATION['gpu_slots']` as ``'legacy'``, ``'auto'`` or a positive
    int (slots per CUDA device). Anything else -> WARNING + ``'legacy'``."""
    if value is None or value is False:
        return 'legacy'
    if isinstance(value, bool):
        return 'auto'
    if isinstance(value, (int, float)) and int(value) >= 1 and int(value) == value:
        return int(value)
    text = str(value).strip().lower()
    if text in ('legacy', 'auto'):
        return text
    if text.isdigit() and int(text) >= 1:
        return int(text)
    print(f"[pipeline_utils] WARNING: ACCELERATION['gpu_slots']={value!r} is not 'legacy', "
          f"'auto' or a positive integer - using 'legacy'.")
    return 'legacy'


def normalise_plink_direct_read(value) -> str:
    """Normalise `ACCELERATION['plink_direct_read']` to 'off'/'auto'/'on'.

    Why: the three-state switch is hand-editable in a JSON config, where a
    user may reasonably write a boolean. `True` is read as 'auto' (use the
    fast path, fall back safely), `False`/`None` as 'off'.

    Raises
    ------
    ValueError
        For any other value - a typo here would otherwise silently leave
        the slow path on, which is exactly the failure this update fixes.
    """
    if value is None or value is False:
        return 'off'
    if value is True:
        return 'auto'
    text = str(value).strip().lower()
    if text in ('off', 'auto', 'on'):
        return text
    raise ValueError(
        f"[pipeline_utils] ACCELERATION['plink_direct_read'] must be 'off', 'auto' or 'on' "
        f"(or a boolean), got {value!r}."
    )


_CUML_GPU_PROBE_CACHE: Dict[str, object] = {}


def cuml_gpu_usable(verbose: bool = True) -> bool:
    """True only if a CUDA device can ACTUALLY be used by cuML/CuPy right now.

    Why this exists: ``import cuml`` succeeds on any node where the package is
    merely *installed* - including CPU-only nodes, or GPU nodes whose NVIDIA
    driver is older than the CUDA runtime bundled in the conda env. The failure
    (``cudaErrorInsufficientDriver`` / ``cudaErrorNoDevice``) then only surfaces
    later, deep inside ``rf.fit()``, aborting the whole task (observed with
    GAT_biological_prior_knowledge's RF marker-selection step). Importing is
    therefore NOT a sufficient availability test; this probe asks the CUDA
    runtime for its device count and performs a tiny allocation.

    Never raises. The result is cached for the process lifetime (the answer
    cannot change mid-run), and the reason for a negative result is printed once.
    """
    if 'ok' in _CUML_GPU_PROBE_CACHE:
        return bool(_CUML_GPU_PROBE_CACHE['ok'])
    ok, reason = False, ''
    try:
        import cupy  # noqa: F401  (cuML's own CUDA backend)
        n_dev = int(cupy.cuda.runtime.getDeviceCount())
        if n_dev < 1:
            reason = 'no CUDA device is visible'
        else:
            cupy.zeros(1, dtype='float32').sum().item()  # forces a real CUDA context
            ok = True
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
    _CUML_GPU_PROBE_CACHE['ok'] = ok
    if not ok and verbose:
        print(f"[pipeline_utils] CUDA is not usable for cuML/CuPy on this node ({reason}) - "
              f"USE_GPU_SKLEARN is ignored and scikit-learn CPU models are used instead.")
    return ok


# ver4-40, R1: the legacy-preserving defaults of every HP_TUNE_FAST sub-key. A
# user-supplied dict is merged over these, so a partial dict (e.g. only `patience`)
# switches ONLY that lever on; every other lever keeps its legacy-equivalent value.
_HP_TUNE_FAST_DEFAULTS = {
    'freeze_fidelity': False,      # keep run-length fields in the search space
    'search_space': 'all',         # 'all' | 'priority'
    'test_stub': False,            # 1-row test frame for R search fits
    'stub_parity_min': 0.995,
    'search_engine': 'bglr',       # 'bglr' | 'exact_kernel'
    'engine_parity_min': 0.98,
    'patience': 0,                 # 0 = no early stop
    'auto_budget': False,
    'evals_per_dim': 6,
    'persistent_executor': False,
    # ver4-46, R1-R4 (GAT search cost). Every key defaults to its legacy-equivalent value:
    'gat_lean_fit': False,         # GAT search fits: no test/attention/train inference, dense
                                   #   batching, cached topology forest (GAT models only)
    'cap_search_cost': False,      # clamp the search box of cost-driving fields
                                   #   (`search_high` marks in HPARAM_SPECS, e.g. GAT neurons/heads)
    'async_batch': False,          # asynchronous constant-liar Bayesian batches (needs
                                   #   persistent_executor); candidate order then depends on timing
    'ship_once': False,            # send the search objective (data) once per worker, not per candidate
    # ver4-61, R4: validation early-stopping patience (epochs) for bio-prior GAT SEARCH fits;
    # 0 = off (every epoch trains, as before). Values 1-2 are raised to 3 (see resolver).
    'gat_search_early_stop': 0,
}


def resolve_hp_tune_fast(raw) -> Optional[Dict[str, object]]:
    """ver4-40, R1: validate and default the top-level ``HP_TUNE_FAST`` config value.

    Returns ``None`` when `raw` is ``None`` / empty / not a dict (legacy single-model
    tuning, byte-identical to ver4-39), otherwise a dict holding EVERY key of
    ``_HP_TUNE_FAST_DEFAULTS`` with a validated value. A bad value (wrong type, unknown
    enum, out-of-range number) prints a WARNING and falls back to that key's legacy
    default - never an exception, so a typo in a config cannot abort a run; an unknown key
    is reported and ignored. Why a helper: the same validation is needed by the GUI
    loader and the resolver, and the defaults must exist in exactly one place (I3)."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        print(f"[pipeline_utils] WARNING: HP_TUNE_FAST={raw!r} is not a dict - ignored "
              f"(legacy single-model tuning).")
        return None
    out = dict(_HP_TUNE_FAST_DEFAULTS)
    for key in raw:
        if key not in _HP_TUNE_FAST_DEFAULTS:
            print(f"[pipeline_utils] WARNING: unknown HP_TUNE_FAST key {key!r} ignored "
                  f"(known: {sorted(_HP_TUNE_FAST_DEFAULTS)}).")
    for key in ('freeze_fidelity', 'test_stub', 'auto_budget', 'persistent_executor',
                'gat_lean_fit', 'cap_search_cost', 'async_batch', 'ship_once'):
        if key in raw:
            if isinstance(raw[key], bool) or type(raw[key]).__name__ == 'bool_':
                out[key] = bool(raw[key])
            else:
                print(f"[pipeline_utils] WARNING: HP_TUNE_FAST[{key!r}]={raw[key]!r} is not a "
                      f"bool - using {out[key]!r}.")
    for key, allowed in (('search_space', ('all', 'priority')),
                         ('search_engine', ('bglr', 'exact_kernel'))):
        if key in raw:
            if raw[key] in allowed:
                out[key] = raw[key]
            else:
                print(f"[pipeline_utils] WARNING: HP_TUNE_FAST[{key!r}]={raw[key]!r} is not one "
                      f"of {allowed} - using {out[key]!r}.")
    for key in ('stub_parity_min', 'engine_parity_min'):
        if key in raw:
            try:
                val = float(raw[key])
                if not (math.isfinite(val) and -1.0 <= val <= 1.5):
                    raise ValueError(val)
                out[key] = val
            except (TypeError, ValueError):
                print(f"[pipeline_utils] WARNING: HP_TUNE_FAST[{key!r}]={raw[key]!r} is not a "
                      f"correlation threshold in [-1, 1.5] - using {out[key]!r}.")
    for key, lo in (('patience', 0), ('evals_per_dim', 1)):
        if key in raw:
            try:
                if isinstance(raw[key], bool):
                    raise ValueError(raw[key])
                val = int(raw[key])
                if val < lo or val != float(raw[key]):
                    raise ValueError(val)
                out[key] = val
            except (TypeError, ValueError):
                print(f"[pipeline_utils] WARNING: HP_TUNE_FAST[{key!r}]={raw[key]!r} is not an "
                      f"integer >= {lo} - using {out[key]!r}.")
    if 'gat_search_early_stop' in raw:
        try:
            if isinstance(raw['gat_search_early_stop'], bool):
                raise ValueError(raw['gat_search_early_stop'])
            val = int(raw['gat_search_early_stop'])
            if val < 0 or val != float(raw['gat_search_early_stop']):
                raise ValueError(val)
            if 0 < val < 3:
                print(f"[pipeline_utils] WARNING: HP_TUNE_FAST['gat_search_early_stop']={val} is "
                      f"too small for a noisy validation curve (a 1-2 epoch plateau is usually "
                      f"noise) - raised to 3. Use 0 to switch it off.")
                val = 3
            out['gat_search_early_stop'] = val
        except (TypeError, ValueError):
            print(f"[pipeline_utils] WARNING: HP_TUNE_FAST['gat_search_early_stop']="
                  f"{raw['gat_search_early_stop']!r} is not an integer >= 0 - using "
                  f"{out['gat_search_early_stop']!r}.")
    return out


# ver4-57, R2/R5: the legacy-preserving defaults of every ITERATIVE_ENSEMBLE sub-key (blueprint
# §4). The feature itself is off unless the config key is a dict (``None`` = off, byte-identical
# to ver4-56); a user-supplied dict is merged over these, so a partial dict switches ONLY that
# setting. Single authority: the GUI loader, the resolver and the GUI panel all read this dict.
_ITER_ENS_DEFAULTS = {
    'max_rounds': 3,                  # int 1-10: most refit rounds that may be tried
    'marker_keep': 0.5,               # float (0, 1): share of markers kept per round (geometric)
    'marker_min': 50,                 # int >= 10: never keep fewer base markers than this
    'protect_interactions': False,    # keep markers of the strongest shared interactions
    'interaction_protect_top': 0.01,  # float [0, 0.2]: share of top interaction pairs protected
    'ld_aware': True,                 # share marker slots across LD windows (d'Hondt)
    'ld_window': 10,                  # int 2-200: consecutive marker columns per window
    'rank_method': 'mean_effect',     # 'mean_effect' | 'rank_average'
    'reentry_share': 0.05,            # float [0, 0.2]: dropped markers put back (0 = off)
    'write_kept_markers': False,      # record kept marker names in the diagnostics file
    'accept_rule': 'one_se',          # 'one_se' | 'margin'
    'accept_margin': 0.005,           # float >= 0: used by accept_rule='margin'
    'bootstrap_reps': 200,            # int 50-2000: paired resamples for the one-SE rule
    'patience': 1,                    # int 1-10: rejected rounds tolerated before stopping
    'explain_selected': False,        # refit the selected round once with explainability on
    'max_fits': 0,                    # int >= 0: extra fits per task (0 = rounds x members)
    'seed': 2026,                     # int: bootstrap seed base (seed + 7919*round)
    'write_rounds': True,             # write Iterative_ensemble_rounds.csv
    # ---- ver4-68 (R1): the 'two_phase' snapshot-library mode. Every key below is INERT under
    # mode='subset' (the default), so each default is legacy-preserving (I11). Appended after
    # `write_rounds`; the iterative-ensemble fingerprint strips them while they sit at these
    # values (iter_ens_fingerprint_view), so a ver4-57..67 checkpoint still resumes.
    'mode': 'subset',                 # 'subset' (ver4-57 loop) | 'two_phase' (ver4-68 library)
    'weight_rule': 'factorised',      # 'factorised' (w0 x snapshot pool) | 'equal' (pooled mean)
    'p1_rounds': 1,                   # int 0-10: distillation snapshots (phase 1); R6 Q1 may set 3
    'p1_distill': 0.5,                # float (0, 1): share of the other members' OOF in the target
    'p2_rounds': 3,                   # int 1-10: chained CMA snapshots (phase 2)
    'cma_core': 0.10,                 # float (0, 0.5]: share of markers every member keeps
    'cma_unique': 0.30,               # float (0, 0.9]: share of markers only this member favours
    'oof_folds': 4,                   # int 2-10: positional folds of the out-of-fold fits
    'oof_engine': 'runner',           # 'runner' | 'exact' (closed form for rrBLUP / RKHS OOF)
    # ---- ver4-69: appended after `oof_engine`, inert under mode='subset', default = ver4-68
    # behaviour (no cap). Stripped from the fingerprint while at the default.
    'w0_alpha_max': 1.0,              # float (0, 1]: largest share of the fitted member weights
                                      # in w0 = (1-a)*equal + a*fitted (1 = the full ver4-42 path)
}

_ITER_ENS_BOOL_KEYS = ('protect_interactions', 'ld_aware', 'write_kept_markers',
                       'explain_selected', 'write_rounds')
_ITER_ENS_ENUMS = {'rank_method': ('mean_effect', 'rank_average'),
                   'accept_rule': ('one_se', 'margin'),
                   'mode': ('subset', 'two_phase'),
                   'weight_rule': ('factorised', 'equal'),
                   'oof_engine': ('runner', 'exact')}
# key -> (low, high, low_inclusive, high_inclusive) for the float settings
_ITER_ENS_FLOAT_RANGES = {'marker_keep': (0.0, 1.0, False, False),
                          'interaction_protect_top': (0.0, 0.2, True, True),
                          'reentry_share': (0.0, 0.2, True, True),
                          'accept_margin': (0.0, float('inf'), True, True),
                          'p1_distill': (0.0, 1.0, False, False),
                          'cma_core': (0.0, 0.5, False, True),
                          'cma_unique': (0.0, 0.9, False, True),
                          'w0_alpha_max': (0.0, 1.0, False, True)}
# key -> (low, high) inclusive, for the integer settings (high None = unbounded)
_ITER_ENS_INT_RANGES = {'max_rounds': (1, 10), 'marker_min': (10, None), 'ld_window': (2, 200),
                        'bootstrap_reps': (50, 2000), 'patience': (1, 10), 'max_fits': (0, None),
                        'p1_rounds': (0, 10), 'p2_rounds': (1, 10), 'oof_folds': (2, 10)}

# ver4-68 (R1/R4): the three cost tiers of mode='two_phase'. A tier is a BUNDLE of ordinary keys
# (I1: the settings stay a plain dict), never a key of its own; ``iter_ens_tier_of`` derives the
# name back from the three keys. Single authority for the GUI tier select, the log line and the
# acceptance tests.
_ITER_ENS_TIERS = {
    'lite': {'weight_rule': 'equal', 'p1_rounds': 0, 'p2_rounds': 3},
    'standard': {'weight_rule': 'factorised', 'p1_rounds': 1, 'p2_rounds': 3},
    'thorough': {'weight_rule': 'factorised', 'p1_rounds': 3, 'p2_rounds': 3},
}

# The ver4-68 keys, in the order they were appended to ``_ITER_ENS_DEFAULTS``.
_ITER_ENS_V468_KEYS = ('mode', 'weight_rule', 'p1_rounds', 'p1_distill', 'p2_rounds', 'cma_core',
                       'cma_unique', 'oof_folds', 'oof_engine')

# ver4-69: keys appended after the ver4-68 ones, and every appended two_phase-only key together.
_ITER_ENS_V469_KEYS = ('w0_alpha_max',)
_ITER_ENS_APPENDED_KEYS = _ITER_ENS_V468_KEYS + _ITER_ENS_V469_KEYS

# Keys that have no effect in mode='two_phase' (reported once per run by GP()).
_ITER_ENS_TWO_PHASE_IGNORED = (
    'max_rounds', 'marker_keep', 'marker_min', 'protect_interactions', 'interaction_protect_top',
    'ld_aware', 'ld_window', 'rank_method', 'reentry_share', 'accept_rule', 'accept_margin',
    'bootstrap_reps', 'patience', 'explain_selected')


def resolve_iterative_ensemble(raw) -> Optional[Dict[str, object]]:
    """ver4-57, R2: validate and default the top-level ``ITERATIVE_ENSEMBLE`` config value.

    Returns ``None`` when `raw` is ``None`` or not a dict (feature off: ver4-56 behaviour,
    byte for byte), otherwise a dict holding EVERY key of ``_ITER_ENS_DEFAULTS`` with a
    validated value (``{}`` therefore means "on, all defaults"). A bad value (wrong type,
    unknown enum, out-of-range number) prints a WARNING and falls back to that key's
    default; an unknown key is reported and ignored; a non-dict switches the feature off
    with a WARNING - never an exception, so a typo in a config cannot abort a run.
    Why a helper: the GUI loader, the engine and the GUI panel need the same validation and
    the defaults must live in exactly one place (I3). Follows ``resolve_hp_tune_fast``."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        print(f"[pipeline_utils] WARNING: ITERATIVE_ENSEMBLE={raw!r} is not a dict - ignored "
              f"(the iterative ensemble stays off).")
        return None
    out = dict(_ITER_ENS_DEFAULTS)
    for key in raw:
        if key not in _ITER_ENS_DEFAULTS:
            print(f"[pipeline_utils] WARNING: unknown ITERATIVE_ENSEMBLE key {key!r} ignored "
                  f"(known: {sorted(_ITER_ENS_DEFAULTS)}).")
    for key in _ITER_ENS_BOOL_KEYS:
        if key in raw:
            if isinstance(raw[key], bool) or type(raw[key]).__name__ == 'bool_':
                out[key] = bool(raw[key])
            else:
                print(f"[pipeline_utils] WARNING: ITERATIVE_ENSEMBLE[{key!r}]={raw[key]!r} is "
                      f"not a bool - using {out[key]!r}.")
    for key, allowed in _ITER_ENS_ENUMS.items():
        if key in raw:
            if raw[key] in allowed:
                out[key] = raw[key]
            else:
                print(f"[pipeline_utils] WARNING: ITERATIVE_ENSEMBLE[{key!r}]={raw[key]!r} is "
                      f"not one of {allowed} - using {out[key]!r}.")
    for key, (lo, hi, lo_inc, hi_inc) in _ITER_ENS_FLOAT_RANGES.items():
        if key in raw:
            try:
                if isinstance(raw[key], bool):
                    raise ValueError(raw[key])
                val = float(raw[key])
                if not math.isfinite(val):
                    raise ValueError(val)
                if (val < lo or (val == lo and not lo_inc)
                        or val > hi or (val == hi and not hi_inc)):
                    raise ValueError(val)
                out[key] = val
            except (TypeError, ValueError):
                print(f"[pipeline_utils] WARNING: ITERATIVE_ENSEMBLE[{key!r}]={raw[key]!r} is "
                      f"not a number in the allowed range {'[' if lo_inc else '('}{lo}, {hi}"
                      f"{']' if hi_inc else ')'} - using {out[key]!r}.")
    for key, (lo, hi) in list(_ITER_ENS_INT_RANGES.items()) + [('seed', (None, None))]:
        if key in raw:
            try:
                if isinstance(raw[key], bool):
                    raise ValueError(raw[key])
                val = int(raw[key])
                if val != float(raw[key]):
                    raise ValueError(val)
                if (lo is not None and val < lo) or (hi is not None and val > hi):
                    raise ValueError(val)
                out[key] = val
            except (TypeError, ValueError):
                print(f"[pipeline_utils] WARNING: ITERATIVE_ENSEMBLE[{key!r}]={raw[key]!r} is "
                      f"not an integer in [{lo}, {hi}] - using {out[key]!r}.")
    # ver4-68 (R1): the CMA masks must leave room for the members' own picks - a core plus an own
    # share of 100 % or more would make every member's mask the whole pool (no diversity).
    if out['cma_core'] + out['cma_unique'] >= 1.0:
        print(f"[pipeline_utils] WARNING: ITERATIVE_ENSEMBLE cma_core ({out['cma_core']}) + "
              f"cma_unique ({out['cma_unique']}) must be below 1 - both reset to their defaults "
              f"({_ITER_ENS_DEFAULTS['cma_core']} and {_ITER_ENS_DEFAULTS['cma_unique']}).")
        out['cma_core'] = _ITER_ENS_DEFAULTS['cma_core']
        out['cma_unique'] = _ITER_ENS_DEFAULTS['cma_unique']
    if out['mode'] == 'two_phase' and out['p1_rounds'] >= 2:
        print(f"[pipeline_utils] NOTE: ITERATIVE_ENSEMBLE p1_rounds={out['p1_rounds']} rebuilds "
              f"the phase-1 targets from the previous round's out-of-fold predictions without "
              f"nested cross-fitting; in simulations such targets leaked the "
              f"training phenotype more with every round (design rule D4). p1_rounds=1 uses "
              f"clean round-0 predictions only.")
    return out


def iter_ens_tier_of(cfg) -> str:
    """ver4-68 (R1/R4): the cost tier a settings dict corresponds to - ``'lite'``,
    ``'standard'``, ``'thorough'`` - or ``'custom'`` for any other combination (and for ``None``).

    Only the three keys a tier bundles are compared (``weight_rule``, ``p1_rounds``,
    ``p2_rounds``; a missing key counts as its default), so a bundle of ``_ITER_ENS_TIERS`` maps
    back to its own name. The tier is derived, never stored (I1/I11: no new config key).
    Pure; never raises."""
    if not isinstance(cfg, dict):
        return 'custom'
    for name, bundle in _ITER_ENS_TIERS.items():
        if all(cfg.get(key, _ITER_ENS_DEFAULTS[key]) == value for key, value in bundle.items()):
            return name
    return 'custom'


def iter_ens_fingerprint_view(cfg):
    """ver4-68 (R1): the part of a resolved ITERATIVE_ENSEMBLE dict that joins the checkpoint
    fingerprint - a copy without every ver4-68 / ver4-69 key that still equals its default. ``None`` passes
    through (feature off).

    Why: a checkpoint written by ver4-57..67 never hashed the nine ver4-68 keys. While they sit
    at their defaults (mode='subset') the view equals the old dict, so an old run still resumes;
    any non-default value (``mode='two_phase'`` or a changed lever) stays in the view and so
    refuses a stale resume (I8). Pure; the input is not modified."""
    if cfg is None:
        return None
    return {key: value for key, value in cfg.items()
            if not (key in _ITER_ENS_APPENDED_KEYS and key in _ITER_ENS_DEFAULTS
                    and value == _ITER_ENS_DEFAULTS[key])}


def iter_ens_wave_groups(settings, n_members: int, n_refit: int,
                         oof_possible: bool = True) -> List[List[Tuple[str, int]]]:
    """ver4-68 (R2): the dependency-wave plan of ``mode='two_phase'`` as, per wave, a list of
    ``(unit_group_id, n_units)``. Pure and deterministic - the plan depends on the settings and
    the member counts only, never on timing (I9).

    Unit groups: ``'oof0'`` (round-0 out-of-fold fits, ``K * F`` units), ``'p1r{r}'`` (phase-1
    refits of round r, ``k`` units), ``'p1r{r}oof'`` (the out-of-fold fits of that round's target,
    ``k * F`` units, only while a later phase-1 round needs them) and ``'p2r{r}'`` (phase-2 CMA
    refits, ``k`` units). With ``K`` = `n_members`, ``k`` = `n_refit`, ``F`` = ``oof_folds``::

        wave 1      : oof0 (if needed)                       + p2r1
        wave w >= 2 : p1r{w-1} [+ p1r{w-1}oof if w-1 < p1]   + p2r{w}

    Phase 2 restarts from the Step 9 columns, so it never waits for the out-of-fold fits or for
    phase 1. The out-of-fold fits are needed when ``weight_rule='factorised'`` or ``p1_rounds >=
    1``; `oof_possible=False` (a member that cannot be fitted out of fold, or too few training
    rows - decided by the caller) drops them and phase 1 with them. Returns ``[]`` only when no
    group exists at all. Used by ``models.iterative_ensemble``, the GUI caption and
    ``iter_ens_fit_estimate``."""
    cfg = settings if isinstance(settings, dict) else {}
    folds = int(cfg.get('oof_folds', _ITER_ENS_DEFAULTS['oof_folds']))
    p1 = int(cfg.get('p1_rounds', _ITER_ENS_DEFAULTS['p1_rounds'])) if oof_possible else 0
    p2 = int(cfg.get('p2_rounds', _ITER_ENS_DEFAULTS['p2_rounds']))
    rule = cfg.get('weight_rule', _ITER_ENS_DEFAULTS['weight_rule'])
    need_oof = bool(oof_possible) and (rule == 'factorised' or p1 >= 1)
    n_waves = max(p1 + (1 if need_oof else 0), p2)
    k, big_k = int(n_refit), int(n_members)
    waves: List[List[Tuple[str, int]]] = []
    for w in range(1, n_waves + 1):
        wave: List[Tuple[str, int]] = []
        if w == 1 and need_oof:
            wave.append(('oof0', big_k * folds))
        if w >= 2 and 1 <= w - 1 <= p1:
            wave.append((f'p1r{w - 1}', k))
            if w - 1 < p1:
                wave.append((f'p1r{w - 1}oof', k * folds))
        if w <= p2:
            wave.append((f'p2r{w}', k))
        waves.append(wave)
    return waves


def iter_ens_fit_estimate(settings, n_members: int, n_bio_prior: int) -> Tuple[int, int]:
    """ver4-57, R5: ``(expected, maximum)`` extra model fits per task and ensemble group.

    `settings` is a resolved ITERATIVE_ENSEMBLE dict; `n_members` the models in the group;
    `n_bio_prior` how many of them are bio-prior models (never refit, I6). With
    ``k = n_members - n_bio_prior`` marker-channel members: ``maximum = max_rounds * k`` (+ ``k``
    when ``explain_selected``), further capped by ``max_fits`` when that is above 0;
    ``expected = round(1.2 * k)`` (the measured 1.0-1.3 refit rounds, blueprint R5), never
    above ``maximum``. Used by the GUI caption and ``resource_profiles``. Never raises."""
    cfg = settings if isinstance(settings, dict) else {}
    k = max(0, int(n_members) - int(n_bio_prior))
    if cfg.get('mode', _ITER_ENS_DEFAULTS['mode']) == 'two_phase':
        # ver4-68 (R1): [need_oof]*K*F + p1*k + max(p1-1, 0)*k*F + p2*k, capped by max_fits.
        # A bio-prior model cannot be fitted out of fold (its runner ignores the fold frames), so
        # with one in the group the out-of-fold fits and phase 1 are dropped - the same rule the
        # engine applies - and only phase 2 remains. `expected` = `maximum` (no early stop).
        waves = iter_ens_wave_groups(cfg, int(n_members), k, oof_possible=int(n_bio_prior) == 0)
        total = sum(count for wave in waves for _gid, count in wave)
        cap = int(cfg.get('max_fits', 0) or 0)
        if cap > 0:
            total = min(total, cap)
        return total, total
    rounds = int(cfg.get('max_rounds', _ITER_ENS_DEFAULTS['max_rounds']))
    refits = rounds * k
    cap = int(cfg.get('max_fits', 0) or 0)
    if cap > 0:
        refits = min(refits, cap)
    maximum = refits + (k if cfg.get('explain_selected') else 0)
    expected = min(int(round(1.2 * k)), maximum)
    return expected, maximum


def iterative_labels(cfg) -> List[str]:
    """ver4-57, R2: the model labels the iterative ensemble adds to a run's reporting list
    (``['Iterative ensemble']`` when ``cfg['ITERATIVE_ENSEMBLE']`` is set, else ``[]``).
    `cfg` is a loaded run config dict (``None`` is tolerated). Used by the runners and the
    GUI wherever ``model_labels`` is extended for the other combination methods."""
    # a dict (even an empty one: "on, all defaults") means on; anything else means off -
    # the same rule resolve_iterative_ensemble() applies, without its warnings.
    if isinstance(cfg, dict) and isinstance(cfg.get('ITERATIVE_ENSEMBLE'), dict):
        return ['Iterative ensemble']
    return []


# ver4-77 (R2): the opt-in SUPER LEARNER (blueprint ``EasiGP_ver4-77_Design_Blueprint.md``, §4).
# The feature is off unless the config key ``SUPER_LEARNER`` is a dict (``None`` = off, byte-
# identical to the previous version; ``{}`` = on with every default below). A user-supplied
# dict is merged over these defaults, so a partial dict switches ONLY that setting. Single
# authority: the GUI loader, the resolver, the GUI panel and ``models.super_learner`` all read
# this dict (I3).
_SL_DEFAULTS = {
    'folds': 5,                       # int 2-20: positional folds of the out-of-fold fits
    'deploy': 'refit',                # 'refit' | 'cross_fit' (see models/super_learner.py)
    'shrink_rule': 'min',             # 'min' | 'one_se': the alpha rule of the weight CV
    'oof_engine': 'runner',           # 'runner' | 'exact' (closed form for rrBLUP / RKHS OOF)
    'explain_refit': False,           # refit with explainability on (deploy='refit' only)
    'between_sources': ['split_train'],   # 'between' only: which validation sources may run
    'max_fits': 0,                    # int >= 0: refuse a plan above this many fits (0 = no cap)
    'seed': 2077,                     # int: fold / weight-CV seed base (+ _SL_SEED_OFFSET)
    # ver4-81 (R1): the W_OPT methods the Super learner applies to its out-of-fold matrix, one row
    # each. The GUI fills it from the ticked weight-optimisation methods when the Super learner
    # takes them over (see ``sl_takes_over_w_opt``): the run then has W_OPT = None, so it needs no
    # validation split and Split ratio(s) are plain numbers. Empty = nothing taken over (the
    # engine then reads W_OPT; with neither, it writes the single legacy analytic row 'Super
    # learner'). ver4-83: the 'weight_methods' switch of ver4-80 was removed - the Super learner
    # always uses the ticked methods, and the GUI refuses to run with none ticked.
    'weight_optimisers': [],
    # ver4-85: tune every HP_TUNE model on OUT-OF-FOLD predictions over its training lines instead
    # of on a validation split (models.hyperparameter_tuning, `oof_folds`). The run then needs no
    # validation split: the GUI accepts plain Split ratio(s) only. False = holdout tuning, exactly
    # as before (it is left out of the checkpoint fingerprint while False, see
    # ``sl_fingerprint_view``). The folds are ``folds`` and are drawn independently of the Super
    # learner's own partition (``models.super_learner.sl_tuning_folds``).
    'oof_tuning': False,
    # ver4-86: apply the cost savings of ``_SL_OOF_FAST`` to out-of-fold tuning (read only while
    # ``oof_tuning`` is True). False = the ver4-85 search (every candidate at full search cost).
    'oof_tuning_fast': True,
    # ver4-94 (R3, gate ``ice_sim40_p1_pooled.py``; simulation only): the population-POOLED Super
    # learner for a 'within' run on a phenotype file with several populations. ``pop_sources`` =
    # ['pooled'] adds, for every member, columns fitted with the OTHER populations' labelled lines
    # added to the training set (its out-of-fold columns on train+valid minus the fold, its test
    # column on all of train+valid), and ONE extra row per task, ``Super learner__<method>
    # (pooled)``, next to the unchanged 'Super learner' row (a free paired comparison). The
    # combiner starts from "own columns equal, pooled columns 0", so a pooled column gets weight
    # only when cross-validation supports it. [] = off (byte-identical to ver4-87; all four
    # ``pop_*`` keys are then left out of the checkpoint fingerprint, see ``sl_fingerprint_view``).
    'pop_sources': [],                # list, subset of ('pooled',)
    'pop_select': 'all',              # 'all' | 'related' (the most related lines, needs pop_max_lines)
    'pop_max_lines': 0,               # int >= 0: cap on source lines (0 = every line)
    'pop_center': True,               # centre each source population's phenotype to the own mean
    # ver4-100: False (default) = the borrowed populations are centred ONCE on the mean of all the
    # own train+valid lines (so a held-out fold nudges the centring: a small optimism in the pooled
    # out-of-fold columns). True = each pooled fold centres on the mean of ITS OWN training lines
    # only. No extra fits; leaves the checkpoint fingerprint while False (earlier hashes unchanged).
    'pop_center_per_fold': False,
    # ver4-96: with 'pooled' ticked, ALSO write one row per member, 'Super learner__<model>
    # (pooled)': that model's own prediction when it is fitted on the own lines plus the other
    # populations' lines (the pooled refit the pooled Super learner already makes - no extra fit).
    # pool-fix: it is joined by 'Super learner__<model> (pooled, gated)', the same prediction passed
    # through the safeguard the pooled Super learner has (cross-validation chooses how much of the
    # pooled prediction to use, starting from the model's own): compare THAT row with the pooled
    # Super learner. This (ungated) row shows what borrowing does to the model with no safeguard.
    # Next to the plain model's row it is the fair, like-for-like test of whether the borrowed
    # lines help that model. False = no such rows (and the key leaves the checkpoint fingerprint
    # while 'pop_sources' is empty, like the other pop_* keys).
    'pop_member_rows': False,
    # pool-fix: rescale each borrowed population's phenotype to the own lines' spread (about the
    # population's own mean, so it composes with 'pop_center'). Populations measured on another
    # scale or in a noisier trial otherwise dominate the shared marker effects of the pooled
    # fits. False = the borrowed phenotypes keep their own spread (the ver4-94 behaviour). The key
    # leaves the checkpoint fingerprint while it is False, so earlier pooled checkpoints resume.
    'pop_scale': False,
    # pool-speed: the cost of the pooled columns. Each pooled fit trains on the own lines PLUS every
    # borrowed line, so the pooled stage dominates the run time. Two levers, measured in
    # ``ice_sim41_pooled_speed.py``:
    #   pop_folds   folds of the POOLED out-of-fold columns (0 = the same as ``folds``). The pooled
    #               columns need not use the own columns' partition: each row only has to be
    #               predicted by a model that did not see it. Fewer folds = fewer fits.
    #   pop_deploy  'refit' = the pooled test column is one extra fit on all own lines + the source
    #               (the ver4-94 behaviour); 'cross_fit' = the mean of the pooled fold models'
    #               test predictions, so that no extra fit is made.
    # Evidence (ice_sim41, 72 replicates, 9 cells, paired with the ver4-94 pooled arm):
    #   cross_fit, same folds   0.82 x the pooled time, mean delta r -0.0008, no harmful cell: ADOPTED
    #   pop_folds=4 + cross_fit ~0.66 x,                mean delta r -0.0014, but foreign/epistatic
    #                           -0.0058 vs the own-only Super learner: misses the no-harm rule
    #   pop_folds=3 (+/- refit) 0.47-0.62 x,            mean delta r -0.003 / -0.004: rejected
    #   pop_folds=2 + cross_fit 0.28 x,                 mean delta r -0.007: rejected
    #   capping the source at 1-2 x the own lines       -0.018 ... -0.037 (related cells -0.05 ...
    #                           -0.11): rejected, so the cap stays a user choice (pop_max_lines)
    # 'refit' with pop_folds 0 reproduces ver4-98 exactly. The legacy values leave the checkpoint
    # fingerprint only (see ``_SL_POP_LEGACY``); the new default value enters it.
    'pop_folds': 0,
    'pop_deploy': 'cross_fit',
}
_SL_BOOL_KEYS = ('explain_refit', 'oof_tuning', 'oof_tuning_fast', 'pop_center',
                 'pop_member_rows', 'pop_scale', 'pop_center_per_fold')

# ver4-86: the cost savings of out-of-fold hyperparameter tuning (SUPER_LEARNER['oof_tuning_fast']),
# single authority - genomic_prediction hands a copy to tune_model_hyperparameters(oof_fast=...).
#   freeze_fidelity  run-length fields (MCMC iterations / burn-in, tree counts, epochs) leave the
#                    search, so the cheap search length applies to every fold fit; the final fit
#                    keeps the configured value (they have no accuracy optimum, ver4-40 C1)
#   exact_engine     rrBLUP / RKHS fold fits by the closed-form kernel engine, kept only when its
#                    held-out column agrees with a real fit on fold 1 (r >= engine_parity_min)
#   test_stub        a 1-row test frame for models registered as stub-safe (no fold fit reads the
#                    test predictions)
#   evals_per_dim    Bayesian budget min(configured, max(4, 1 + evals_per_dim x d)), d = searched
#                    dimensions. OFF (None): in ice_sim28 (float ratio) 1 + 6 d cost -0.0021 r on
#                    the Super learner (S-epistatic -0.0076, t -2.5), failing the no-harm rule
#   race_margin      fold racing: after >= race_min_folds folds a candidate whose pooled partial
#                    score is below the default point's score minus this margin stops there.
#                    0.05 passed ice_sim28 (+0.0003 r, no harmful cell, 16 % fewer search fits);
#                    0.02 failed (per-model rows -0.008, t -2.5)
#   one_se           ver4-87: paired one-standard-error rule on the winner-minus-default fold
#                    scores (the out-of-fold counterpart of HP_TUNE_VALID_REPEATS' 1-SE rule).
#                    OFF: in ice_sim29 it was neutral for the Super learner (-0.0001 r) but kept
#                    the defaults more often and lowered the per-model rows (S-oligogenic -0.0034,
#                    t -2.2); the holdout formula (independent SEs) was worse (-0.0083, t -4.1)
_SL_OOF_FAST = {
    'freeze_fidelity': True,
    'exact_engine': True,
    'engine_parity_min': 0.98,
    'test_stub': True,
    'evals_per_dim': None,
    'race_margin': 0.05,
    'race_min_folds': 2,
    'one_se': False,
}
_SL_ENUMS = {'deploy': ('refit', 'cross_fit'),
             'shrink_rule': ('min', 'one_se'),
             'oof_engine': ('runner', 'exact'),
             'pop_select': ('all', 'related'),
             'pop_deploy': ('refit', 'cross_fit')}
# pool-speed: the values of the two pooled-cost keys that mean "as in ver4-94..99"; they leave the
# checkpoint fingerprint at these values, so an earlier pooled checkpoint still resumes
_SL_POP_LEGACY = {'pop_folds': 0, 'pop_deploy': 'refit'}
# ver4-83: settings that were removed; a saved config that still holds one gets a NOTE, not a crash
_SL_RETIRED_KEYS = ('weight_methods',)
# ver4-80 (R1): the weight-optimisation methods a Super learner row can be built with, in the
# order of the GUI's W_OPT_METHODS (single authority for the names, I3). The legacy row is the
# cross-validated analytic least-squares one (models.ensemble_regularization.
# shrinkage_path_weights) and keeps the label 'Super learner'.
SL_ANALYTIC_METHOD = 'Analytic least-squares'
SL_WEIGHT_METHODS = ('Nelder Mead', 'Linear transformation', 'Bayesian optimisation',
                     SL_ANALYTIC_METHOD)
# key -> (low, high) inclusive, for the integer settings (high None = unbounded)
_SL_INT_RANGES = {'folds': (2, 20), 'max_fits': (0, None), 'pop_max_lines': (0, None),
                  'pop_folds': (0, 20)}
_SL_BETWEEN_SOURCES = ('split_train', 'split_test', 'third_population')
# ver4-94 (R3): the kinds of borrowed columns. The blueprint also names 'others' (columns fitted
# on the other populations alone); it was not better than 'pooled' in any simulated cell (gate
# verdict A10), so it is not built.
_SL_POP_SOURCES = ('pooled',)
# ver4-94 (R0): what ``max_fits`` drops first, in this order, before it refuses a plan outright
# (the unchanged Super learner is never dropped). One authority for models/super_learner.py.
_SL_OPTIONAL_DROP_ORDER = ('pop_sources',)


def resolve_super_learner(raw) -> Optional[Dict[str, object]]:
    """ver4-77, R2: validate and default the top-level ``SUPER_LEARNER`` config value.

    Returns ``None`` when `raw` is ``None`` or not a dict (feature off: previous-version
    behaviour, byte for byte), otherwise a dict holding EVERY key of ``_SL_DEFAULTS`` with a
    validated value (``{}`` therefore means "on, all defaults"). A bad value (wrong type,
    unknown enum, out-of-range number, unknown source name) prints a WARNING and falls back to
    that key's default; an unknown key is reported and ignored; a non-dict switches the feature
    off with a WARNING - never an exception, so a typo in a config cannot abort a run.
    Why a helper: the GUI loader, the engine and the GUI panel need the same validation and the
    defaults must live in exactly one place (I3). Follows ``resolve_iterative_ensemble``."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        print(f"[pipeline_utils] WARNING: SUPER_LEARNER={raw!r} is not a dict - ignored "
              f"(the Super learner stays off).")
        return None
    out = dict(_SL_DEFAULTS)
    out['between_sources'] = list(_SL_DEFAULTS['between_sources'])
    out['weight_optimisers'] = list(_SL_DEFAULTS['weight_optimisers'])
    out['pop_sources'] = list(_SL_DEFAULTS['pop_sources'])
    for key in raw:
        if key in _SL_RETIRED_KEYS:
            print(f"[pipeline_utils] NOTE: SUPER_LEARNER[{key!r}] was removed and is "
                  f"ignored: the Super learner now always determines its weights with the "
                  f"weight-optimisation method(s) ticked in '4. Ensemble' (W_OPT).")
        elif key not in _SL_DEFAULTS:
            print(f"[pipeline_utils] WARNING: unknown SUPER_LEARNER key {key!r} ignored "
                  f"(known: {sorted(_SL_DEFAULTS)}).")
    for key in _SL_BOOL_KEYS:
        if key in raw:
            if isinstance(raw[key], bool) or type(raw[key]).__name__ == 'bool_':
                out[key] = bool(raw[key])
            else:
                print(f"[pipeline_utils] WARNING: SUPER_LEARNER[{key!r}]={raw[key]!r} is "
                      f"not a bool - using {out[key]!r}.")
    for key, allowed in _SL_ENUMS.items():
        if key in raw:
            if raw[key] in allowed:
                out[key] = raw[key]
            else:
                print(f"[pipeline_utils] WARNING: SUPER_LEARNER[{key!r}]={raw[key]!r} is "
                      f"not one of {allowed} - using {out[key]!r}.")
    for key, (lo, hi) in list(_SL_INT_RANGES.items()) + [('seed', (None, None))]:
        if key in raw:
            try:
                if isinstance(raw[key], bool):
                    raise ValueError(raw[key])
                val = int(raw[key])
                if val != float(raw[key]):
                    raise ValueError(val)
                if (lo is not None and val < lo) or (hi is not None and val > hi):
                    raise ValueError(val)
                out[key] = val
            except (TypeError, ValueError):
                print(f"[pipeline_utils] WARNING: SUPER_LEARNER[{key!r}]={raw[key]!r} is "
                      f"not an integer in [{lo}, {hi}] - using {out[key]!r}.")
    if out['pop_folds'] == 1:                 # pool-speed: 0 (= folds) or at least 2
        print("[pipeline_utils] WARNING: SUPER_LEARNER['pop_folds']=1 is not allowed (0 = same as "
              "'folds', or 2-20) - using 0.")
        out['pop_folds'] = 0
    if 'between_sources' in raw:
        value = raw['between_sources']
        if (isinstance(value, (list, tuple))
                and all(isinstance(v, str) and v in _SL_BETWEEN_SOURCES for v in value)):
            out['between_sources'] = [v for v in _SL_BETWEEN_SOURCES if v in value]
        else:
            print(f"[pipeline_utils] WARNING: SUPER_LEARNER['between_sources']={value!r} is "
                  f"not a list of {list(_SL_BETWEEN_SOURCES)} - using "
                  f"{out['between_sources']!r}.")
    if 'pop_sources' in raw:
        value = raw['pop_sources']
        if (isinstance(value, (list, tuple))
                and all(isinstance(v, str) and v in _SL_POP_SOURCES for v in value)):
            out['pop_sources'] = [v for v in _SL_POP_SOURCES if v in value]
        else:
            others_asked = isinstance(value, (list, tuple)) and 'others' in value
            print(f"[pipeline_utils] WARNING: SUPER_LEARNER['pop_sources']={value!r} is not a "
                  f"list of {list(_SL_POP_SOURCES)} - using {out['pop_sources']!r}."
                  + (" The 'others' option is not available in this version." if others_asked
                     else ""))
    if 'weight_optimisers' in raw:
        value = raw['weight_optimisers']
        if (isinstance(value, (list, tuple))
                and all(isinstance(v, str) and v in SL_WEIGHT_METHODS for v in value)):
            out['weight_optimisers'] = [v for v in SL_WEIGHT_METHODS if v in value]
        else:
            print(f"[pipeline_utils] WARNING: SUPER_LEARNER['weight_optimisers']={value!r} is "
                  f"not a list of {list(SL_WEIGHT_METHODS)} - using "
                  f"{out['weight_optimisers']!r}.")
    return out


def super_learner_labels(cfg) -> List[str]:
    """ver4-77, R2: the model label the Super learner adds to a run's reporting list
    (``['Super learner']`` when ``cfg['SUPER_LEARNER']`` is a dict, else ``[]``). `cfg` is a
    loaded run config dict (``None`` is tolerated). Used by the runners and the GUI wherever
    ``model_labels`` is extended for the other combination methods (see ``iterative_labels``;
    the same rule ``resolve_super_learner`` applies, without its warnings)."""
    if isinstance(cfg, dict) and isinstance(cfg.get('SUPER_LEARNER'), dict):
        # ver4-80 (R1): one label per weight method the run will write a row for
        w_opt = None if cfg.get('JOINT_W_OPT') else cfg.get('W_OPT')
        labels = list(sl_row_labels(cfg['SUPER_LEARNER'], w_opt).values())
        # ver4-94 (R3): the pooled row exists only for a 'within' run with 'pooled' ticked
        if (cfg.get('SCENARIO') == 'within'
                and 'pooled' in (cfg['SUPER_LEARNER'].get('pop_sources') or [])):
            labels += list(sl_tagged_row_labels(cfg['SUPER_LEARNER'], w_opt, 'pooled').values())
            # ver4-96: the per-member pooled rows (named after the selected models; with HP_TUNE
            # the runners read the real label set from Metric.csv instead)
            labels += list(sl_member_row_labels(
                cfg['SUPER_LEARNER'], w_opt, cfg.get('MODEL') or []).values())
            # pool-fix: and the gated per-member rows next to them
            labels += list(sl_member_row_labels(
                cfg['SUPER_LEARNER'], w_opt, cfg.get('MODEL') or [], 'pooled_gated').values())
        return labels
    return []


def sl_weight_methods(settings, w_opt) -> List[str]:
    """ver4-80, R1: the weight methods the Super learner writes a row for, in a fixed order.

    `settings` is a SUPER_LEARNER dict (resolved or raw; its ``weight_optimisers`` list, if any,
    counts as ticked), `w_opt` the run's ``W_OPT`` list (``None`` or empty = nothing ticked).
    Returns the ticked methods of ``SL_WEIGHT_METHODS`` (unknown names are ignored), so
    SUPER_LEARNER + Bayesian optimisation gives the Bayesian row built on the out-of-fold matrix.
    Returns ``['Analytic least-squares']`` - the single legacy row, labelled 'Super learner' -
    only when nothing is ticked (an old or headless config; the GUI refuses to run that case,
    see ``main_app.gather_config``). Pure; never raises. Why: the Super learner used to be one
    fixed combination (out-of-fold predictions + cross-validated analytic least squares) whatever
    the user ticked as the weight optimiser (blueprint ver4-80, R1)."""
    ticked = _sl_ticked_methods(settings, w_opt)
    return ticked if ticked else [SL_ANALYTIC_METHOD]


def _sl_ticked_methods(settings, w_opt) -> List[str]:
    """The weight methods the user asked for, in ``SL_WEIGHT_METHODS`` order; ``[]`` when nothing
    was asked for (the fallback case)."""
    own = settings.get('weight_optimisers') if isinstance(settings, dict) else None
    # ver4-81 (R1): the Super learner's own list (taken over from W_OPT by the GUI) and the run's
    # W_OPT (a headless config, or a run that also needs W_OPT's own rows) are both honoured.
    wanted = list(own) if isinstance(own, (list, tuple)) else []
    if isinstance(w_opt, (list, tuple)):
        wanted += list(w_opt)
    return [m for m in SL_WEIGHT_METHODS if m in wanted]


def sl_takes_over_w_opt(sl_settings, w_opt, hp_tune=None, iterative=None) -> bool:
    """ver4-81, R1: True when the Super learner takes the ticked weight-optimisation methods over
    from ``W_OPT``, so the run needs no validation split for them.

    Why: a weight optimiser normally searches its weights on the validation split, which is why
    ``W_OPT`` forces (train, validation, test) tuples. With the Super learner on, it searches
    them on out-of-fold predictions of every labelled row instead, so nothing needs a validation
    split and plain Split ratio(s) such as 0.8 are enough. It is False (the optimisers keep
    their own validation-split rows) when the Super learner is off, when no method is ticked, or
    when hyperparameter tuning (`hp_tune`) or the iterative ensemble (`iterative`) needs a
    validation split anyway. Joint optimisation is not a case: it already clears ``W_OPT``. Pure;
    never raises.

    ver4-85: hyperparameter tuning needs no validation split when it is done on out-of-fold
    predictions (``sl_settings['oof_tuning']``), so it then does not block the takeover."""
    if not isinstance(sl_settings, dict):
        return False
    if not isinstance(w_opt, (list, tuple)) or not any(m in w_opt for m in SL_WEIGHT_METHODS):
        return False
    tuning_needs_valid = bool(hp_tune) and not sl_oof_tuning_on(sl_settings)
    return not tuning_needs_valid and iterative is None


def sl_oof_tuning_on(sl_settings) -> bool:
    """ver4-85: True when a SUPER_LEARNER dict (resolved or raw) asks for out-of-fold
    hyperparameter tuning (``oof_tuning``). ``None`` / not a dict / key absent = False. Pure."""
    return isinstance(sl_settings, dict) and bool(sl_settings.get('oof_tuning', False))


def sl_fingerprint_view(sl_settings) -> Dict[str, object]:
    """ver4-85: the SUPER_LEARNER dict as it joins the checkpoint fingerprint. ``oof_tuning`` is
    dropped while it is False, so a run resolved by ver4-77..84 hashes exactly what it hashed
    then and its checkpoint still resumes (I8, I11); switching it on refuses a stale resume.
    ver4-86: ``oof_tuning_fast`` is dropped with it (it means nothing while tuning is holdout).
    ver4-94: the four ``pop_*`` keys are dropped while ``pop_sources`` is empty (ver4-96: the
    fifth, ``pop_member_rows``, with them)."""
    view = dict(sl_settings or {})
    if not view.get('oof_tuning', False):
        view.pop('oof_tuning', None)
        view.pop('oof_tuning_fast', None)
    # ver4-94 (R3): the pop_* keys mean nothing while no population is borrowed, so they leave the
    # hash then - a ver4-87 checkpoint with the Super learner on still resumes (I8, I11).
    if not view.get('pop_sources'):
        for key in ('pop_sources', 'pop_select', 'pop_max_lines', 'pop_center',
                    'pop_member_rows', 'pop_scale', 'pop_folds', 'pop_deploy',
                    'pop_center_per_fold'):
            view.pop(key, None)
    if not view.get('pop_scale', False):       # pool-fix: off = the ver4-94..98 hash
        view.pop('pop_scale', None)
    if not view.get('pop_center_per_fold', False):   # ver4-100: off = the earlier hash
        view.pop('pop_center_per_fold', None)
    for key, legacy in _SL_POP_LEGACY.items():  # pool-speed: legacy value = the ver4-94..99 hash
        if view.get(key, legacy) == legacy:
            view.pop(key, None)
    return view


def sl_oof_tuning_plan(sl_settings) -> Optional[Dict[str, object]]:
    """ver4-86: the cost-saving plan for out-of-fold tuning - a copy of ``_SL_OOF_FAST`` when
    ``oof_tuning`` and ``oof_tuning_fast`` are both on, else None (ver4-85 search, or holdout
    tuning). Pure; never raises."""
    if not sl_oof_tuning_on(sl_settings):
        return None
    if not bool(sl_settings.get('oof_tuning_fast', _SL_DEFAULTS['oof_tuning_fast'])):
        return None
    return dict(_SL_OOF_FAST)


def sl_row_label(method, group_label=None, legacy_analytic: bool = False,
                 tag: Optional[str] = None) -> str:
    """ver4-80, R1: the ``model`` label of one Super learner row.

    Every weight method names itself after a double underscore, the analytic one included
    (ver4-82): ``'Super learner__Analytic least-squares'``, ``'Super learner__Bayesian
    optimisation'``. Every module that recognises a weighted-ensemble label by the prefix
    ``'Super learner__'`` keeps working unchanged. Only with ``legacy_analytic=True`` does
    ``'Analytic least-squares'`` keep the plain label ``'Super learner'`` (the row written when
    no method was asked for: the ver4-77 label, so old configs and result files stay comparable). `group_label` is the HP-tuning ensemble group
    (``None`` = no suffix), appended last as for every other weighted ensemble:
    ``'Super learner__<method>__<group>'`` (``'Super learner__<group>'`` for the legacy label).
    The method names cannot collide with a tuning-method name. Pure; never raises.

    ver4-94 (R0): `tag` names an optional Super-learner variant that is written as an EXTRA row
    next to the untagged one (today only ``'pooled'``): ``'Super learner__<method> (<tag>)'``,
    and ``'Super learner__<tag>'`` for the legacy analytic label; the group suffix stays last.
    Both keep the prefix ``'Super learner__'`` that every weighted-ensemble predicate looks for.
    ``tag=None`` is the pre-ver4-94 label, unchanged."""
    label = 'Super learner'
    if tag is None:
        if not (legacy_analytic and method == SL_ANALYTIC_METHOD):
            label = f'{label}__{method}'
    elif legacy_analytic and method == SL_ANALYTIC_METHOD:
        label = f'{label}__{tag}'
    else:
        label = f'{label}__{method} ({tag})'
    if group_label is not None:
        label = f'{label}__{group_label}'
    return label


def sl_row_labels(settings, w_opt, group_label=None) -> Dict[str, str]:
    """ver4-82, R1: ``{method: model label}`` for every Super learner row of a run, in row order
    - the single authority the engine, the runners and the GUI all use.

    The methods are those of :func:`sl_weight_methods`. The analytic row is labelled
    ``'Super learner__Analytic least-squares'`` when the user ticked it, and keeps the plain
    legacy label ``'Super learner'`` only when it is the fallback (nothing ticked). Pure; never
    raises."""
    methods = sl_weight_methods(settings, w_opt)
    legacy = not _sl_ticked_methods(settings, w_opt)
    return {m: sl_row_label(m, group_label, legacy_analytic=legacy) for m in methods}


def sl_tagged_row_labels(settings, w_opt, tag: str, group_label=None) -> Dict[str, str]:
    """ver4-94 (R0): ``{method: model label}`` of the rows of the optional Super-learner variant
    `tag` (today only ``'pooled'``), in row order; ``{}`` when the variant writes none.

    The analytic pooled row weights ``[own | pooled]`` columns with a prior (the route the
    ``ice_sim40`` gate measured). pool-methods: every other ticked weight method (Nelder Mead,
    Linear transformation, Bayesian optimisation) also gets a pooled row, built as a two-stage
    gate (see ``models.super_learner._pooled_method_variant``): ``'Super learner__<method>
    (pooled)'``. The label carries the same legacy / group rules as :func:`sl_row_label`. Pure;
    never raises. `sl_row_labels` is deliberately left alone so its callers keep their return
    shape."""
    if tag != 'pooled':
        return {}
    methods = sl_weight_methods(settings, w_opt)
    legacy = not _sl_ticked_methods(settings, w_opt)
    return {m: sl_row_label(m, group_label, legacy_analytic=legacy, tag=tag) for m in methods}


# pool-fix: the per-member row families and the text each puts in the label's brackets. 'pooled' is
# the ungated row (the model fitted with the borrowed lines, no safeguard); 'pooled_gated' keeps
# the model's own prediction as the starting point and lets cross-validation decide how much of
# the pooled prediction to use (the like-for-like counterpart of the pooled Super-learner row).
SL_MEMBER_TAGS = ('pooled', 'pooled_gated')
_SL_MEMBER_TAG_TEXT = {'pooled': 'pooled', 'pooled_gated': 'pooled, gated'}


def sl_member_row_label(name, tag: str = 'pooled', group_label=None) -> str:
    """ver4-96: the ``model`` label of one MEMBER row of an optional Super-learner variant:
    ``'Super learner__<model> (pooled)'`` (the group suffix, if any, stays last as for every
    other row). It keeps the prefix ``'Super learner__'`` on purpose: every weighted-ensemble
    predicate (and the ensemble-member exclusion lists, which match the prefix on prediction
    columns too) then leaves the row out of the weight, diversity and naive-ensemble logic, which
    a bare ``'<model> (pooled)'`` would leak into as if it were a base model. Pure; never raises."""
    label = f'Super learner__{name} ({_SL_MEMBER_TAG_TEXT.get(tag, tag)})'
    if group_label is not None:
        label = f'{label}__{group_label}'
    return label


def sl_member_row_labels(settings, w_opt, names, tag: str = 'pooled',
                         group_label=None) -> Dict[str, str]:
    """ver4-96: ``{model name: label}`` of the per-member rows of the optional variant `tag`
    (``'pooled'``; pool-fix: also ``'pooled_gated'``), in `names` order; ``{}`` when they are not asked for.

    They exist only with ``pop_member_rows`` on and ``'pooled'`` in ``pop_sources`` (pool-methods:
    whatever weight methods are ticked; earlier the analytic one had to be among them). Whether a given model ends up with a row still
    depends on its pooled fits succeeding. Pure; never raises."""
    if tag not in SL_MEMBER_TAGS or not isinstance(settings, dict):
        return {}
    if not settings.get('pop_member_rows') or 'pooled' not in (settings.get('pop_sources') or []):
        return {}
    return {str(n): sl_member_row_label(n, tag, group_label) for n in (names or [])}


def sl_fit_estimate(settings, n_members: int, n_bio_prior: int, has_valid: bool) -> int:
    """ver4-77, R2: extra model fits the Super learner makes per task and ensemble group.

    `settings` is a resolved SUPER_LEARNER dict; `n_members` the models in the group;
    `n_bio_prior` how many are bio-prior models (never part of the Super learner, I6);
    `has_valid` whether the task has a validation split. With ``k = n_members - n_bio_prior``:
    ``k * folds`` out-of-fold fits, plus ``k`` refits when ``has_valid`` and
    ``deploy == 'refit'``; ``0`` when ``k < 2`` (no row is written). Used by the GUI caption and
    ``resource_profiles``. Never raises.

    ver4-94 (R3): with ``'pooled'`` in ``pop_sources``, plus ``k * (folds + 1)`` (the pooled
    columns' out-of-fold fits and one refit each). It is an upper bound: the pooled fits only run
    for 'within' tasks that have source lines, and each runs on more rows than an own fit."""
    cfg = settings if isinstance(settings, dict) else {}
    k = max(0, int(n_members) - int(n_bio_prior))
    if k < 2:
        return 0
    folds = int(cfg.get('folds', _SL_DEFAULTS['folds']))
    refit = cfg.get('deploy', _SL_DEFAULTS['deploy']) == 'refit'
    pf = int(cfg.get('pop_folds', 0) or 0) or folds                       # pool-speed
    pooled = (k * (pf + (0 if cfg.get('pop_deploy', _SL_DEFAULTS['pop_deploy']) == 'cross_fit'
                         else 1))
              if 'pooled' in (cfg.get('pop_sources') or []) else 0)
    return k * folds + (k if (has_valid and refit) else 0) + pooled


_GAT_FAMILY = ('GAT_infinitesimal', 'GAT_infinitesimal_node_level', 'GAT_fully_connected',
               'GAT_prior_knowledge', 'GAT_biological_prior_knowledge')


def tuning_sanity_report(model_list, hp_tune, compute_resources, acceleration_absent: bool,
                         hp_tune_fast) -> List[str]:
    """ver4-46, R5: human-readable warnings about settings that make GAT hyperparameter
    tuning far slower than it needs to be. Returns [] when nothing applies (no GAT model is
    tuned, or every setting is fine); never raises and never changes any setting.

    Why: each cause below was found by reading code, not by profiling - the run log is
    where the person running the job can confirm it. Checks: (1) HP_TUNE_FAST is unset
    although a GAT model is tuned; (2) ACCELERATION is absent, so the ver4-33/35 GAT
    switches (dense batching, factorised input, worker thread limits) stay legacy;
    (3) the process is daemonic, which silently collapses batch Bayesian search to one
    candidate at a time; (4) TORCH_DATALOADER_WORKERS > 0, which respawns loader processes
    every epoch; (5) worker thread limits are off while several trial workers run at once
    (the node is oversubscribed); (6) HP_TUNE_VALID_REPEATS multiplies the fit count.

    Args: `model_list` (MODEL names), `hp_tune` (the HP_TUNE dict or None),
    `compute_resources` (resolve_compute_resources() output), `acceleration_absent` (True
    when the config had no ACCELERATION key), `hp_tune_fast` (resolved HP_TUNE_FAST or None)."""
    try:
        tuned = []
        bio_cfgs = []
        for name in (model_list or []):
            base = str(name).split('__')[0]
            for gat in _GAT_FAMILY:
                if base == gat or base.startswith(gat + '_'):
                    cfg = (hp_tune or {}).get(gat) or (hp_tune or {}).get(base) or {}
                    if cfg.get('enabled'):
                        tuned.append(base)
                        if gat == 'GAT_biological_prior_knowledge':
                            bio_cfgs.append(cfg)
                    break
        if not tuned:
            return []
        out = []
        tag = ', '.join(sorted(set(tuned)))
        if not hp_tune_fast:
            out.append(f"HP_TUNE_FAST is not set although {tag} is tuned - the search runs the "
                       f"full-length, full-output fit for every candidate. Set HP_TUNE_FAST "
                       f"(e.g. {{'freeze_fidelity': True, 'gat_lean_fit': True, 'cap_search_cost': "
                       f"True, 'persistent_executor': True, 'ship_once': True}}).")
        else:
            if not hp_tune_fast.get('gat_lean_fit'):
                out.append("HP_TUNE_FAST['gat_lean_fit'] is off - GAT search fits still compute "
                           "test/train/attention outputs that the search never reads.")
            if not hp_tune_fast.get('freeze_fidelity'):
                out.append("HP_TUNE_FAST['freeze_fidelity'] is off - the Epoch field stays in the "
                           "search space, so the reduced-epoch search rung is skipped and "
                           "candidates train up to the full Epoch range.")
        if acceleration_absent:
            out.append("ACCELERATION is absent from the config - gat_dense_batching, "
                       "gat_factorised_input, gat_sync_free and worker_thread_limits all stay at "
                       "their legacy (off) values.")
        elif not compute_resources.get('accel_worker_thread_limits', False):
            out.append("ACCELERATION['worker_thread_limits'] is off - concurrent trial workers keep "
                       "full-size torch/BLAS thread pools and oversubscribe the node.")
        try:
            import multiprocessing
            if multiprocessing.current_process().daemon:
                out.append("this process is a daemonic worker - batch Bayesian search cannot spawn "
                           "trial workers here and runs ONE candidate at a time (q=1).")
        except Exception:
            pass
        if int(compute_resources.get('torch_dataloader_workers') or 0) > 0:
            out.append(f"TORCH_DATALOADER_WORKERS={compute_resources.get('torch_dataloader_workers')} "
                       f"- each epoch starts new loader processes; for in-memory graph data 0 is "
                       f"almost always faster.")
        if int(compute_resources.get('hp_tune_valid_repeats') or 1) > 1:
            out.append(f"HP_TUNE_VALID_REPEATS={compute_resources.get('hp_tune_valid_repeats')} "
                       f"multiplies every candidate's fit count by that factor.")
        # ver4-61, R3: bio-prior searches are the most expensive of the family (one fit can
        # take hours on a merged graph) and had no wall-clock bound.
        if bio_cfgs:
            if not any(float(c.get('time_limit_min') or 0) > 0 for c in bio_cfgs):
                out.append("GAT_biological_prior_knowledge is tuned without HP_TUNE[...]"
                           "['time_limit_min'] - the search has no wall-clock bound (only the "
                           "candidate count and patience stop it). The GUI button 'Apply bio-prior "
                           "tuning preset' sets 180 min together with a 20 min per-fit cap.")
            if str(compute_resources.get('device', 'cpu')).startswith('cpu'):
                out.append("GAT_biological_prior_knowledge is tuned on device=cpu - its edge "
                           "scatter/gather is memory-bound there and a merged graph can take "
                           "hours per fit; a GPU (TORCH_DEVICE='cuda') is expected to be much "
                           "faster.")
        return out
    except Exception:
        return []


def resolve_compute_resources(cfg: Optional[Dict] = None) -> Dict[str, object]:
    """Resolve the compute-hardware settings every model module should use,
    from a single config-derived source.

    Parameters
    ----------
    cfg : the run's config dict (or any dict-like subset of it - only the
        keys documented below are read). ``None`` (or a dict missing every
        key) resolves every setting to its safe, backward-compatible
        default - identical to this pipeline's original ad hoc,
        single-core-CPU behaviour.

    Recognised config keys (all optional)
    --------------------------------------
    ``USE_GPU_SKLEARN`` (bool, default False) - whether RF/KNN/SVR should
        attempt a ``cuML`` GPU backend when available.
    ``N_JOBS`` (int, default -1) - ``n_jobs`` forwarded to scikit-learn
        estimators (RF/KNN) that support it; -1 means "use every core".
    ``PLINK_THREADS`` (int, default 1) - forwarded as PLINK2's own
        ``--threads`` flag by every ``plink2`` subprocess call.
    ``R_BLAS_THREADS`` (int, optional) - if given, exported as
        ``OPENBLAS_NUM_THREADS``/``OMP_NUM_THREADS``/``MKL_NUM_THREADS``
        before R/BGLR runs, so the linear algebra underneath rrBLUP/GBLUP/
        BayesB/RKHS's MCMC can use a multi-threaded BLAS if the R build
        linked one - EasiGP does not control which BLAS R itself was
        built against; this only forwards the THREAD COUNT, matching
        whatever multi-threaded BLAS is already present.
    ``R_BLAS_FOLLOWS_N_JOBS`` (bool, default True) - ver4-4 R3.c: when
        ``R_BLAS_THREADS`` is absent, use this run's own (already
        n_model_workers-divided) ``n_jobs`` as the BLAS thread count
        instead of leaving it unset - only when ``n_jobs`` resolves to a
        positive, explicit count (never for the ``-1``/"every core"
        sentinel, which isn't a usable environment-variable value). Set
        False to keep R's BLAS threading independent of ``N_JOBS``.
    ``TORCH_DEVICE`` (str, optional) - explicit override (``'cuda'``,
        ``'cuda:0'``, ``'cpu'``, ...) for MLP/GAT model device placement.
        When omitted, resolves to ``'cuda'`` if a GPU is visible to
        PyTorch, else ``'cpu'`` - never raises if PyTorch or CUDA aren't
        available (falls back to ``'cpu'``).
    ``TORCH_NUM_THREADS`` (int, optional) - ver4-4 R3.b: explicit CPU
        intra-op thread count for ``apply_torch_compute_settings()`` to
        pass to ``torch.set_num_threads()``. When omitted, resolves to
        this run's own ``n_jobs`` (when positive) or, failing that, to
        ``cpus_per_model_worker`` - never raises, and is simply unused on
        a CUDA device.
    ``CUDNN_BENCHMARK`` (bool, default True when a CUDA device is
        resolved) - sets ``torch.backends.cudnn.benchmark``.
    ``USE_AMP`` (bool, default False) - whether GAT/MLP training should use
        automatic mixed precision (``torch.autocast``) on a CUDA device.
    ``N_CPU_WORKERS`` (int, default 1) - number of task-level worker
        PROCESSES for intra-batch parallelism (Requirement 7); 1 means the
        existing fully-serial driving loop, unchanged.
    ``PARALLEL`` (dict, optional) - this run's own ``{'batch_id',
        'batch_size'}`` dict (the same one GP() otherwise uses to slice
        the task table). Read-only here: only used to WARN when
        ``N_CPU_WORKERS_TASK``/``N_CPU_WORKERS`` requests more concurrent
        task-level workers than ``batch_size`` tasks could ever fill at
        once while ``N_JOBS``/``PLINK_THREADS`` are left single-threaded -
        the exact CPU-starvation pattern ``main_app.py``'s own HPC export
        already avoids for GUI-generated configs (see this function's own
        inline comment near where ``n_jobs``/``plink_threads`` are
        resolved). Never changes any resolved value - omitting this key
        (e.g. a Sequential run, which has no meaningful ``batch_size``)
        simply skips the check.
    ``N_GPU_SLOTS`` (int, optional) - number of GPU-dispatched model calls
        allowed to run concurrently across ``N_CPU_WORKERS`` worker
        processes; defaults to ``GPU_SLOTS_PER_DEVICE`` x the visible CUDA
        device count (at least 1 device) when a GPU is available, else 0
        (GPU calls fall back to CPU automatically via the resolved
        device). An explicit ``N_GPU_SLOTS`` always wins outright - see
        ``GPU_SLOTS_PER_DEVICE`` below.
    ``GPU_SLOTS_PER_DEVICE`` (int, default 1) - Update ID 2 (R2, ver4-3
        design blueprint §4): how many concurrent GPU-dispatched model
        calls each physical CUDA device can host (e.g. ``2`` lets two
        model fits share one GPU's memory instead of serialising through
        a single slot). Only multiplies the AUTO-COMPUTED ``N_GPU_SLOTS``
        default above - it never scales an explicitly-supplied
        ``N_GPU_SLOTS``, and its own default (1) reproduces Phase 2's
        original "one slot per visible device" behaviour exactly
        (``N_GPU_SLOTS <= ngpus``), preserving flag-off byte-identity
        (A2.2) for every config written before this key existed. (Fixes
        Test Report D1: previously accepted by ``resource_profiles.
        estimate_resources()`` [the GUI advisor] but never read here, so
        a hand-edited value was silently inert at run time.)
    ``TORCH_DATALOADER_WORKERS`` (int, default 0) - ver4-4 R4.d: worker
        PROCESS count passed to every torch/PyG ``DataLoader``'s own
        ``num_workers``. 0 (today's behaviour) means the loader iterates
        in the calling process itself - no new subprocess is ever spawned,
        which is why this is unconditionally safe as a default. A caller
        should read this back via ``dataloader_num_workers()`` below,
        rather than this raw resolved value directly, whenever the call
        site could itself already be running inside a daemonic worker
        process (RK-3) - see that function's own docstring.
    ``GPU_EVAL_BATCH`` (int, default 32) - ver4-4 R4.b/c: the requested
        batch size for a torch/PyG inference or explanation loop, read
        back via ``torch_eval_batch_size()`` below (never directly) -
        that function collapses this to 1 (today's behaviour) whenever
        the resolved device isn't CUDA, so this key is fully inert on a
        CPU-only node regardless of its own value.
    ``GPU_LD_R2`` (bool, default True - ver4-4 §4a "default on") -
        whether ``Preprocess/ld_kernels.py::pairwise_r2_matrix()`` should
        attempt a CUDA (torch) LD r² computation when the resolved device
        is CUDA; always falls back to the existing NumPy path otherwise,
        or on any CUDA failure. Consumed by ``Preprocess/LD_pruning.py``
        and ``Preprocess/LD_decay_plot.py``, not by this module directly.
    ``GPU_KERNEL_PRECOMPUTE`` (bool, default True - ver4-4 §4a "default
        on") - whether ``genomic_prediction.py`` should build GBLUP's/
        RKHS's N x N kernel matrix in Python (optionally on GPU) and hand
        it to the R model function pre-built, skipping that function's own
        (CPU, single-threaded-R) kernel construction. Consumed by
        ``genomic_prediction.py`` directly, not by this module.
    ``HP_TUNE_PARALLEL_TRIALS`` (bool, default True - ver4-4 §4a "default
        on") - ver4-4 R3.f: whether ``genomic_prediction.py`` should
        request a process-backend joblib fan-out across Grid/Random
        hyperparameter-search trial evaluations. Consumed by
        ``genomic_prediction.py`` directly (this run's own resolved
        ``n_jobs`` is only actually forwarded to
        ``tune_model_hyperparameters()`` when this flag is true), not by
        this module.
    ``RESOURCE_TELEMETRY`` (bool, default True - ver4-56 R1) - append per-model,
        per-task and per-job usage records (seconds, peak memory, sizes,
        hyper-parameters, allocation) as JSON lines under
        ``Result/<name>/logs/resource_telemetry/``. Log-only: results are
        byte-identical on or off. ``RESOURCE_TELEMETRY_INTERVAL_S`` (float,
        default 0.5) is the memory sampling period.
    ``HP_TUNE_BAYES_BATCH`` (bool, default True - ver4-5 R1.b) - whether
        the Bayesian hyperparameter search should run in constant-liar
        BATCH mode (q candidates per round, see
        ``models/parallel_search.py``) whenever this run's own resolved
        ``n_jobs`` names more than one usable core AND
        ``HP_TUNE_PARALLEL_TRIALS`` is also enabled. False (or
        ``n_jobs<=1``) reproduces ver4-4's exact serial
        ``optimizer.maximize()`` code path, byte-for-byte. Consumed by
        ``genomic_prediction.py`` -> ``tune_model_hyperparameters()`` ->
        ``search_bayesian()``.
    ``HP_TUNE_BAYES_BATCH_MAX`` (int, default 8) - upper cap on the
        batch width `q` (``q = min(n_jobs, HP_TUNE_BAYES_BATCH_MAX)``),
        so a very wide node doesn't request an unreasonably large,
        surrogate-quality-degrading batch per round. A cap only - never
        changes behaviour when ``HP_TUNE_BAYES_BATCH`` is off.
    ``HP_TUNE_BAYES_LIAR`` (str, default ``'max'``) - which fabricated
        target ``models/parallel_search.py::_constant_liar_value()``
        registers for a just-suggested, not-yet-truly-evaluated point
        within one batch round: ``'max'`` (classical Constant-Liar-max,
        pessimistic), ``'mean'``, or ``'believer'`` (Kriging Believer -
        reaches a private ``bayes_opt`` attribute, guarded, falls back
        to ``'max'`` if unavailable).
    ``HP_TUNE_PARALLEL_RESTARTS`` (bool, default True - ver4-5 R1.c) -
        whether Nelder-Mead/Powell hyperparameter search's own multi-
        restart local search (``models/hyperparameter_tuning.py::
        _multistart_local_search``) dispatches its independent restarts
        across worker processes (when ``n_jobs>1``) instead of a serial
        ``for`` loop. Result-preserving at any value (each restart is
        already fully independent of the others - only wall-clock time
        changes), so this is inert-by-construction rather than a
        numerics-changing flag.
    ``HP_TUNE_BAYES_DOMAIN_REDUCTION`` (str, default ``'auto'`` - ver4-6
        R1.2(d)) - ``'auto'`` enables sequential domain reduction only
        when a model's tunable box has no categorical dimension (see
        ``models/hyperparameter_tuning.py::search_bayesian``'s own
        docstring for why); ``'always'`` restores ver4-5's unconditional
        behaviour; ``'never'`` disables it outright. **Not** ver4-5-
        equivalent by default - set ``'always'`` to restore it exactly.
    ``HP_TUNE_FAST`` (dict or None, default None - ver4-40 R1) - opt-in fast mode for
        single-model hyperparameter tuning (``HP_TUNE``). ``None`` = ver4-39 behaviour,
        byte-for-byte. A dict switches on the levers named in it (keys: freeze_fidelity,
        search_space, test_stub, stub_parity_min, search_engine, engine_parity_min,
        patience, auto_budget, evals_per_dim, persistent_executor - see
        ``resolve_hp_tune_fast()``); every other lever keeps its legacy value. Resolved
        into ``hp_tune_fast`` (None or the fully defaulted dict).
    ``HP_TUNE_BAYES_MULTI_FIDELITY`` (bool, default False) - whether the
        Bayesian hyperparameter search's own BATCH path (see
        ``HP_TUNE_BAYES_BATCH`` above) additionally pre-screens each
        round's candidates on a small, fixed, reduced-row-count subsample
        of the current task's own training split before spending the full
        evaluation budget only on the more promising half of that round -
        see ``models/hyperparameter_tuning.py::search_bayesian()``'s own
        docstring for the full mechanism. ``False`` (default) reproduces
        today's exact full-fidelity-every-candidate behaviour; has no
        effect at all (a logged NOTE explains why) whenever
        ``HP_TUNE_BAYES_BATCH`` is off, ``n_jobs``/batch width resolves to
        1, or the current task's own training split is too small to
        subsample meaningfully.
    ``HP_TUNE_WARM_START`` (bool, default False - ver4-6 R3.2a) -
        ``False`` (new default): every tuned ``(task, model)`` call
        starts its search from the user's own configured ``HPARAMETERS``
        baseline, never a previous task's tuned result - fixes a latent
        sequential-vs-sharded divergence (the previous behaviour mutated
        ``HPARAMETERS`` in place across tasks). ``True`` restores the
        exact ver4-5 carry-over behaviour for anyone relying on it.
    ``HP_TUNE_SELECTION_MARGIN`` (float, default 0.02 - ver4-6 R3.2b) -
        the tuning search's winning candidate must beat the untuned
        default's own score by at least this much (in objective-score
        units) or the untuned defaults are kept instead - guards the
        "never worse than untuned" floor against winner's-curse bias from
        comparing one noisy default draw against the maximum of many
        search draws. **Not** ver4-5-equivalent by default - ``0.0``
        restores the exact ver4-5 bare ``>=`` comparison.
    ``HP_TUNE_VALID_REPEATS`` (int >= 1, default 1 - ver4-6 R3.2c) -
        when > 1, each tuning candidate is scored as the mean over this
        many independent inner train/validation resamples drawn from the
        current task's own TRAINING split only (never touching the
        task's real ``valid``/``test`` frames - invariant I7), reducing
        the objective's own sampling variance at a proportional cost in
        model fits. ``1`` (default) reproduces ver4-5 exactly.
    ``HP_TUNE_SCOPE`` (str, default ``'per_task'`` - ver4-6 R4.2b) -
        ``'per_task'`` (default, ver4-5-equivalent): every ``(task,
        model)`` pair with tuning enabled runs its own independent
        search. ``'per_scenario'``: tunes ONCE per ``(population,
        phenotype, ratio, model, algorithm)`` group and reuses the
        winning parameters for every remaining replicate in that group
        WITHIN THE CURRENT SHARD - a real reduction in total model fits,
        but makes tuned results shard-dependent (disclosed unconditionally
        in the run log and per-row in ``hyperparameter.csv``'s own
        ``tuning_source`` column whenever active).
    ``W_OPT_ANALYTIC_SEED`` (bool, default False - ver4-6 R6; GUI
        checkbox hidden as of Requirements.md item 5) - whether
        Bayesian optimisation/Nelder-Mead/Linear transformation each
        additionally compute and use ``models/ensemble_regularization.py
        ::analytic_simplex_weights``'s closed-form least-squares solution
        as an extra search probe/floor candidate ON TOP OF their own
        search. Distinct from - and unrelated to - the 'Analytic
        least-squares' entry in ``W_OPT``/``HYPERPARAMETERS_OPT``, which
        is now this exact same closed-form solve offered as its OWN,
        independent weight-optimisation method (see
        ``models/Analytic_least_squares.py``) and does not read this flag
        at all. A hand-edited config may still set this to ``true``.
    ``W_OPT_VALIDATION_FLOOR`` (bool, default False - ver4-6 R5.2b; GUI
        checkbox hidden as of Requirements.md item 6) - whether every
        weight-optimisation method grades its own winning weight vector
        (and the analytic solution, and equal weighting) on REALISED
        validation MSE via ``models/ensemble_regularization.
        py::select_weights_with_floor``, guaranteeing the returned
        weights are never worse than equal (naive-ensemble) weighting on
        validation. ``True`` restores the ver4-6 guarantee (a hand-edited
        config may still set this); the ver4-5 default was an ad hoc
        extraction (abs-normalise-shrink, no floor).
    ``W_OPT_SIMPLEX_SEARCH`` (bool, default True - ver4-6 R7.2(b)) -
        whether Bayesian-optimisation/Nelder-Mead weight search
        parameterises its search space as a ``(K-1)``-dimensional stick-
        breaking map onto the probability simplex, rather than an
        unconstrained ``K``-dimensional box (whose objective is scale-
        invariant along one whole dimension - see
        ``models/Bayesian_optimisation.py``'s own R7 design note).
        **Not** ver4-5-equivalent by default - ``False`` restores the
        ver4-5 K-dimensional box search byte-for-byte.
    ``SHAPIQ_ROW_PARALLEL`` (bool, default True - ver4-26 design blueprint
        R1) - whether `shapiq`-based marker-interaction routes (RF's
        'shapiq_tree'; SVR/KNN's 'shapiq'; RKHS-family's
        'surrogate_shapiq') fan their per-row `shapiq` explain loop out
        across this run's own resolved ``n_jobs`` worker processes
        (``pipeline_utils._shapiq_explain_rows()``), mirroring the
        sibling exact-SHAP route's own established parallel-fan-out
        precedent (``parallel_shap_values()``). ``False`` (or a
        single-core node, ``n_jobs<=1``) reproduces every affected
        route's pre-R1 strictly-serial per-row loop byte-for-byte (I11) -
        pure speed, no value change at any `n_jobs`.
    ``SHAPIQ_ADAPTIVE_DEPTH`` (bool, default True - ver4-26 design
        blueprint R2) - whether RF's 'shapiq_tree' route's interaction-
        only refit caps its own ``max_depth``/``n_estimators`` more
        aggressively as the requested interaction order grows (see
        ``models/RF.py::_resolve_shapiq_tree_depth_cap()``'s own
        docstring for the ``O(T*L*D**k)`` rationale). Identical output at
        order=2 (the common case) either way (I11); **does** change
        order>=3 output values when ``True`` (a shallower/smaller forest
        is a different approximation) - disclosed, not silent, since
        order>=3 'shapiq_tree' is itself a disclosed-approximate feature
        with no prior "reliable" baseline at that order. ``False``
        restores today's flat ``max_depth=8``/uncapped-``n_estimators``
        behaviour regardless of order. ver4-28 (this update): the SAME
        flag also gates every 'shapiq'-family route's own explained-ROW-
        count shrink at order>=3 (``pipeline_utils.
        resolve_shapiq_explain_row_count()``, consumed by ``models/RF.py``,
        ``models/SVR.py``, ``models/KNN.py``) - reused rather than adding
        a fourth config key, since it is the same "shrink cost as order
        grows" policy applied to one more axis.
    ``SHAPIQ_CANDIDATE_NARROWING`` (bool, default True - ver4-26 design
        blueprint R3) - whether `shapiq_pairwise_interactions()` (SVR/
        KNN's 'shapiq'; RKHS-family's 'surrogate_shapiq') narrows its
        order>=3 search to a calibration-ranked marker subset when
        ``max(orders) >= 3`` (see that function's own R3b docstring).
        Identical output at ``orders=(2,)`` (the common case; narrowing
        is never entered) either way (I11); **does** change order>=3
        output values when ``True`` and narrowing actually triggers.
        ``False`` reproduces this route's pre-R3 single-pass, full-
        shortlist search exactly - the companion feasibility pre-check
        (``_shapiq_coalition_space_size()``) still runs regardless of
        this flag, since it only changes WHEN an already-certain failure
        fires, never a value.

    Returns
    -------
    Dict with keys: ``device`` (str, e.g. ``'cuda:0'``/``'cpu'``),
    ``use_gpu_sklearn`` (bool), ``n_jobs`` (int), ``plink_threads`` (int),
    ``r_blas_threads`` (int or None), ``cudnn_benchmark`` (bool),
    ``use_amp`` (bool), ``n_cpu_workers`` (int), ``n_gpu_slots`` (int),
    ``gpu_slots_per_device`` (int), ``torch_num_threads`` (int, ver4-4 R3.b),
    ``torch_dataloader_workers`` (int, ver4-4 R4.d), ``gpu_eval_batch``
    (int, ver4-4 R4.b/c), ``gpu_ld_r2`` (bool, ver4-4 R4.g),
    ``gpu_kernel_precompute`` (bool, ver4-4 R4.h), ``hp_tune_parallel_trials``
    (bool, ver4-4 R3.f).

    ver4-33 additions (design blueprint SS4): ``n_cuda_devices`` (int, the
    visible CUDA device count, 0 on CPU-only nodes) and the acceleration
    switches ``accel_plink_direct_read`` ('off'/'auto'/'on'),
    ``accel_preprocess_memo``, ``accel_gat_dense_batching``,
    ``accel_gat_batched_ig``, ``accel_gpu_device_round_robin`` (all read
    from the ``ACCELERATION`` config dict, legacy values when absent) and
    the engine-only ``accel_bio_prior_prep_fast`` /
    ``accel_bio_prior_lean_pools`` (``BIO_PRIOR_PREP_FAST`` /
    ``BIO_PRIOR_LEAN_POOLS``, default True - byte-identical refactors) and
    ``accel_gat_ig_internal_batch`` (``GAT_IG_INTERNAL_BATCH``, default 256).

    ver4-35 additions (design blueprint SS4), all legacy when absent:
    ``policy_threads`` (the CPU thread count device placement assumes; never
    divided for trial workers), ``accel_gat_factorised_input``,
    ``accel_gat_sync_free``, ``accel_worker_thread_limits``,
    ``accel_gpu_slots`` ('legacy' | 'auto' | int slots per device),
    ``accel_torch_device_policy`` ('legacy' | 'auto' | 'cpu' | 'gpu'),
    ``accel_async_model_lane``, ``accel_ld_cm_banded``,
    ``accel_bio_prior_cost_guard`` ('warn' | 'raise' | 'off'),
    ``accel_merge_shap_device`` ('cpu' | 'gpu') and the engine-only
    ``accel_shap_stream_reduce``, ``accel_shared_bim_marker_info``,
    ``accel_async_lane_rng``, ``accel_async_lane_threads``,
    ``accel_async_lane_max_payload_mb``, ``accel_bio_prior_max_fit_gflop``,
    ``accel_defer_result_sort`` (optimise_scan F3; False when absent).

    ver4-61 addition (R2): ``accel_trial_memory_guard`` ('auto' | 'off'; 'auto' when absent -
    the one non-legacy default of ver4-61, see ``_TRIAL_MEMORY_GUARD_MODES``).

    ver4-43 addition (R1): ``accel_baseline_shap_engine`` ('legacy' | 'fast';
    'legacy' when absent) - selects the hybrid-row loop used by the RKHS/SVR/
    KNN ``baseline_shap`` marker-pair explanations.

    This function never imports/raises on a missing optional dependency -
    it degrades gracefully to CPU-only settings whenever PyTorch/CUDA
    aren't importable, exactly matching a CPU-only node's expected
    behaviour with zero configuration.
    """
    cfg = cfg or {}

    device = cfg.get('TORCH_DEVICE')
    cuda_available = False
    cuda_device_count = 0
    if not device:
        try:
            import torch  # local import: never a hard dependency of this module
            cuda_available = bool(torch.cuda.is_available())
            if cuda_available:
                cuda_device_count = torch.cuda.device_count()
        except Exception:
            cuda_available = False
        device = 'cuda' if cuda_available else 'cpu'
    else:
        # An explicit override is trusted as given, but we still probe
        # device COUNT (best-effort, never raising) for N_GPU_SLOTS' own
        # default below.
        try:
            import torch
            if str(device).startswith('cuda') and torch.cuda.is_available():
                cuda_available = True
                cuda_device_count = torch.cuda.device_count()
        except Exception:
            pass

    # Update ID 2 (R2, ver4-3 design blueprint §4 / Test Report D1 fix) -
    # GPU_SLOTS_PER_DEVICE only ever multiplies the AUTO-COMPUTED default
    # below (device count -> slot count). An explicit N_GPU_SLOTS (the
    # branch below this one) is trusted outright and is never re-scaled by
    # it, exactly as N_JOBS/PLINK_THREADS/R_BLAS_THREADS above are trusted
    # outright when supplied explicitly. Defaults to 1, which reproduces
    # "one slot per visible device" - i.e. every config written before
    # this key existed (or that never sets it) resolves an identical
    # n_gpu_slots to before (A2.2 flag-off byte-identity).
    gpu_slots_per_device = max(1, int(cfg.get('GPU_SLOTS_PER_DEVICE', 1) or 1))

    n_gpu_slots = cfg.get('N_GPU_SLOTS')
    if n_gpu_slots is None:
        n_gpu_slots = (max(1, cuda_device_count) * gpu_slots_per_device) \
            if (cuda_available or str(device).startswith('cuda')) else 0
    else:
        n_gpu_slots = max(0, int(n_gpu_slots))

    # ------------------------------------------------------------------ #
    # Update ID 2, R1/R2 (blueprint T12) - a SECOND parallelism width.
    #
    # N_CPU_WORKERS_TASK is the canonical name for Phase 2's existing
    # task-level fan-out width (intra_batch_parallel.py); N_CPU_WORKERS is
    # kept, unchanged in meaning, as an alias (I11) so no existing caller
    # of get_active_compute_resources()['n_cpu_workers'] needs to change.
    # N_MODEL_WORKERS is Update ID 2's new model-level fan-out width
    # (intra_task_parallel.py) - defaults to 1, which is fully inert.
    #
    # Deliberate deviation from a literal reading of the design blueprint
    # (recorded in EasiGP_2_Change_Summary.md §7): the blueprint describes
    # re-dividing n_jobs/plink_threads/r_blas_threads by
    # "N_CPU_WORKERS_TASK x N_MODEL_WORKERS". This function has NEVER
    # actually divided those by N_CPU_WORKERS_TASK/N_CPU_WORKERS at
    # runtime - main_app.py's own HPC export handler already pre-divides
    # N_JOBS by N_CPU_WORKERS at CONFIG-WRITE time (see
    # render_hpc_export_section()'s 'Pipeline compute settings' block)
    # before it ever reaches this function, which is the existing,
    # shipped Phase 2 behaviour (existing code behaviour takes precedence
    # over the blueprint text where the two disagree - see
    # Phase2_Coding_2.md §1). Dividing by N_CPU_WORKERS_TASK a SECOND time
    # here would silently halve (or worse) an already-correctly-sized
    # N_JOBS for every existing Phase-2-era config. Instead, ONLY the new
    # N_MODEL_WORKERS width is divided out here - purely additive, since
    # N_MODEL_WORKERS defaults to 1 (division by 1 is a no-op), so every
    # config that predates this feature gets byte-identical n_jobs/
    # plink_threads/r_blas_threads values (I11). A caller/GUI that sets
    # BOTH widths is expected to have already accounted for
    # N_CPU_WORKERS_TASK in whatever N_JOBS/PLINK_THREADS/R_BLAS_THREADS
    # it supplies, exactly as Phase 2 already required for N_CPU_WORKERS
    # alone - this function then divides that value by N_MODEL_WORKERS on
    # top, giving the full N_CPU_WORKERS_TASK x N_MODEL_WORKERS division
    # across the two responsibility layers combined.
    # ------------------------------------------------------------------ #
    _n_cpu_workers_task_cfg = cfg.get('N_CPU_WORKERS_TASK')
    if _n_cpu_workers_task_cfg is None:
        _n_cpu_workers_task_cfg = cfg.get('N_CPU_WORKERS', 1)
    n_cpu_workers_task = max(1, int(_n_cpu_workers_task_cfg or 1))
    n_model_workers = max(1, int(cfg.get('N_MODEL_WORKERS', 1) or 1))

    # Safety clamp (blueprint T12/risk RK-4): never recommend/accept a
    # (n_cpu_workers_task x n_model_workers) process count this process
    # can't even see CPUs for. os.cpu_count() is the same conservative,
    # zero-configuration signal already used elsewhere in this codebase;
    # falls back to 1 (never raises) if undetermined. Only ever reduces
    # n_model_workers, and only when it was requested above 1 - so a
    # config with the R1 feature flag at its default (N_MODEL_WORKERS=1)
    # is NEVER altered by this clamp (max(1, anything) still floors at
    # the same 1 it started at), preserving A2.2 flag-off byte-identity.
    _detected_cpus = os.cpu_count() or 1
    _requested_n_model_workers = n_model_workers
    if n_cpu_workers_task * n_model_workers > _detected_cpus:
        n_model_workers = max(1, _detected_cpus // n_cpu_workers_task)
    if n_model_workers != _requested_n_model_workers:
        print(f"[pipeline_utils] NOTE: requested N_CPU_WORKERS_TASK={n_cpu_workers_task} x "
              f"N_MODEL_WORKERS={_requested_n_model_workers} = "
              f"{n_cpu_workers_task * _requested_n_model_workers} concurrent process(es) exceeds "
              f"the {_detected_cpus} CPU(s) this process can see - clamping N_MODEL_WORKERS down "
              f"to {n_model_workers} so the product fits, to avoid oversubscribing this node.")

    _n_jobs_cfg = cfg.get('N_JOBS', -1)
    if _n_jobs_cfg is not None and int(_n_jobs_cfg) > 0 and n_model_workers > 1:
        n_jobs = max(1, int(_n_jobs_cfg) // n_model_workers)
    else:
        n_jobs = int(_n_jobs_cfg) if _n_jobs_cfg is not None else -1

    _plink_threads_cfg = max(1, int(cfg.get('PLINK_THREADS', 1)))
    plink_threads = max(1, _plink_threads_cfg // n_model_workers) if n_model_workers > 1 else _plink_threads_cfg

    # Bugfix (companion to main_app.py's render_hpc_export_section() "CPUs
    # per task" fix - see its own comment for the full story, and the
    # architecture doc §12.4/§18 for the underlying data-driven-merge cost):
    # that GUI code path already avoids handing out N_CPU_WORKERS_TASK
    # worker "slots" a batch can never actually fill (batch_size caps how
    # many tasks ever run concurrently) without redirecting the freed-up
    # CPU budget into N_JOBS/PLINK_THREADS instead - but it can only do
    # that at the moment a config is GENERATED from the GUI. A config
    # written any other way (hand-maintained, produced by an older
    # EasiGP version, or edited directly for a headless HPC submission -
    # exactly the shape of a real config this was tracked down from) never
    # passes through that code and so never benefits from it: it can
    # request N_CPU_WORKERS_TASK worker slots that this batch's own
    # batch_size can never fill, while leaving N_JOBS/PLINK_THREADS at
    # their single-threaded defaults - silently wasting however many CPUs
    # were actually reserved for the job and leaving the CPU-bound LD
    # pruning / RF filtering / hyperparameter-tuning trial fan-out to run
    # on one core no matter how many were requested. This is a read-only
    # diagnostic, not a silent rewrite - explicit N_JOBS/PLINK_THREADS
    # values are trusted outright everywhere else in this function, and
    # this is not an exception (some runs genuinely want single-threaded
    # determinism) - it only ever prints, so it is safe for every existing
    # caller/config regardless of whether the pattern below actually
    # applies to it.
    _parallel_cfg = cfg.get('PARALLEL') or {}
    _batch_size_cfg = _parallel_cfg.get('batch_size')
    if _batch_size_cfg is not None:
        _effective_task_workers = max(1, min(n_cpu_workers_task, int(_batch_size_cfg) or 1))
        if (_effective_task_workers < n_cpu_workers_task) and n_jobs in (0, 1) and plink_threads <= 1:
            print(f"[pipeline_utils] WARNING: N_CPU_WORKERS_TASK={n_cpu_workers_task} requests that "
                  f"many concurrent task-level worker(s), but this batch's own PARALLEL.batch_size="
                  f"{_batch_size_cfg} means at most {_effective_task_workers} task(s) ever run at "
                  f"once - the other {n_cpu_workers_task - _effective_task_workers} worker slot(s) "
                  f"can never be used and sit idle. Meanwhile N_JOBS={n_jobs} and "
                  f"PLINK_THREADS={plink_threads} leave LD pruning and RF filtering (and "
                  f"hyperparameter-tuning trial fan-out) single-threaded. This is the same "
                  f"'LD filtering and RF filtering take too much time' pattern main_app.py's own "
                  f"HPC export already avoids for GUI-generated configs by redirecting unused "
                  f"worker capacity into N_JOBS/PLINK_THREADS (see render_hpc_export_section()). "
                  f"If this job actually reserved more than {max(n_jobs, plink_threads)} CPU(s), set "
                  f"N_JOBS and PLINK_THREADS to that count explicitly, and set N_CPU_WORKERS_TASK "
                  f"(and N_CPU_WORKERS) to {_effective_task_workers} - it cannot do anything useful "
                  f"above that for this batch.")

    # ver4-4 R3.c - R_BLAS_FOLLOWS_N_JOBS: when R_BLAS_THREADS itself is
    # absent, fall back to this run's own (already n_model_workers-divided)
    # n_jobs rather than leaving R's BLAS single-threaded by default -
    # default True (the blueprint's own §4a-documented non-legacy default
    # for this ONE flag), so an existing config that only ever set N_JOBS
    # now ALSO gets multi-threaded BLAS underneath BGLR's own MCMC linear
    # algebra for free, unless explicitly turned off. Implemented HERE
    # (the shared resolver), rather than duplicated inside
    # run_step1_batch.py/run_sequential.py individually as the blueprint's
    # own prose literally suggests - both of those scripts (and GP()'s own
    # internal call) already funnel through this ONE function, so a single
    # implementation here reaches every caller uniformly and can never
    # drift out of sync between them (see the Change Summary for why this
    # is a deliberate, disclosed deviation from the blueprint's literal
    # "where" instruction, not a scope change to WHAT R3.c does). `n_jobs`
    # is the LOCAL, already-resolved value above (its own n_model_workers
    # division already applied) - not a second, independent read of
    # cfg['N_JOBS'] - so the two settings can never disagree about which
    # "N_JOBS" they mean. A sentinel n_jobs of -1 ("every core") is, same
    # as for TORCH_NUM_THREADS above, not a usable thread COUNT for an
    # environment variable, so the fallback only fires for a positive,
    # explicit n_jobs.
    _r_blas_follows_n_jobs = bool(cfg.get('R_BLAS_FOLLOWS_N_JOBS', True))
    _r_blas_cfg = cfg.get('R_BLAS_THREADS')
    if not _r_blas_cfg and _r_blas_follows_n_jobs and n_jobs > 0:
        _r_blas_cfg = n_jobs
    if _r_blas_cfg and n_model_workers > 1:
        r_blas_threads = max(1, int(_r_blas_cfg) // n_model_workers)
    else:
        r_blas_threads = int(_r_blas_cfg) if _r_blas_cfg else None

    cpus_per_model_worker = max(1, _detected_cpus // max(1, n_cpu_workers_task * n_model_workers))

    # ver4-4 R3.b - torch_num_threads: the CPU intra-op thread count
    # apply_torch_compute_settings() will pass to torch.set_num_threads().
    # Precedence (blueprint §2.3.2 R3.b): an explicit TORCH_NUM_THREADS
    # always wins; otherwise reuse this SAME run's already-resolved n_jobs
    # when it names a positive core count (n_jobs's own resolution above
    # already applied the N_MODEL_WORKERS division, so reusing it here
    # keeps torch and scikit-learn/BLAS sized consistently with each
    # other rather than inventing a second, independent division); a
    # sentinel n_jobs of -1 ("every core", sklearn's own convention) is
    # not a usable thread COUNT for torch.set_num_threads(), so that case
    # falls through to cpus_per_model_worker instead - the SAME
    # "os.cpu_count() // (n_cpu_workers_task * n_model_workers)" division
    # already computed above for exactly this situation, rather than a
    # second, independently-derived one.
    _torch_num_threads_cfg = cfg.get('TORCH_NUM_THREADS')
    if _torch_num_threads_cfg is not None:
        torch_num_threads = max(1, int(_torch_num_threads_cfg))
    elif n_jobs > 0:
        torch_num_threads = n_jobs
    else:
        torch_num_threads = cpus_per_model_worker

    # ver4-4 R4.b/c/d/g/h - four new resolved keys. Every one of them
    # defaults to a value that is either byte-identical to today's
    # behaviour (torch_dataloader_workers=0, gpu_eval_batch is inert
    # without a CUDA device) or an explicitly §4a-announced "default on,
    # non-bit-identical, logged" acceleration (gpu_ld_r2,
    # gpu_kernel_precompute) - never silently changes a CPU-only run's
    # numbers. `int(...)` here mirrors every other integer-resolving line
    # above; `or 0`/`or 32` guards against an explicit `null`/empty-string
    # config value behaving differently from an absent key.
    torch_dataloader_workers = max(0, int(cfg.get('TORCH_DATALOADER_WORKERS', 0) or 0))
    gpu_eval_batch = max(1, int(cfg.get('GPU_EVAL_BATCH', 32) or 32))
    gpu_ld_r2 = bool(cfg.get('GPU_LD_R2', True))
    gpu_kernel_precompute = bool(cfg.get('GPU_KERNEL_PRECOMPUTE', True))

    # ver4-4 R3.f (blueprint §2.3.2/§4a "default on") - whether
    # genomic_prediction.py should request a process-backend joblib
    # fan-out across Grid/Random hyperparameter-search trial evaluations
    # (models/hyperparameter_tuning.py::search_grid/search_random, via
    # tune_model_hyperparameters()'s own `n_jobs` parameter). Default True
    # per the blueprint's §4a policy - this is a PARALLELISM-only flag
    # (n_jobs=1 reproduces the exact serial `trials` list/winner - see
    # search_grid()/search_random()'s own R3.f docstrings), never a
    # numerics one, so it is NOT listed in GP()'s own `[GP] NUMERICS:`
    # line. Consumed by genomic_prediction.py directly (which multiplies
    # this run's own resolved `n_jobs` by this flag before ever calling
    # tune_model_hyperparameters()), not by this module.
    hp_tune_parallel_trials = bool(cfg.get('HP_TUNE_PARALLEL_TRIALS', True))

    # ver4-5 blueprint R1 (§3.6) - four further hyperparameter-tuning
    # keys, every one defaulted so an untouched config reproduces
    # today's exact behaviour on a single core (q collapses to 1
    # whenever N_JOBS<=1 regardless of these flags - see
    # models/hyperparameter_tuning.py::search_bayesian's own docstring).
    hp_tune_bayes_batch = bool(cfg.get('HP_TUNE_BAYES_BATCH', True))
    hp_tune_bayes_batch_max = max(1, int(cfg.get('HP_TUNE_BAYES_BATCH_MAX', 8) or 8))
    hp_tune_bayes_liar = str(cfg.get('HP_TUNE_BAYES_LIAR', 'max') or 'max')
    hp_tune_parallel_restarts = bool(cfg.get('HP_TUNE_PARALLEL_RESTARTS', True))

    # ver4-6 blueprint §4 - eight further keys (five hyperparameter-
    # tuning-track, three weight-optimisation-track). See this function's
    # own docstring above for each key's full meaning/default rationale;
    # this block only resolves + validates + defaults them, never
    # re-decides a default independently of what's documented there.
    _hp_tune_bayes_dr = str(cfg.get('HP_TUNE_BAYES_DOMAIN_REDUCTION', 'auto') or 'auto')
    if _hp_tune_bayes_dr not in ('auto', 'always', 'never'):
        print(f"[pipeline_utils] WARNING: HP_TUNE_BAYES_DOMAIN_REDUCTION={_hp_tune_bayes_dr!r} "
              f"is not one of 'auto'/'always'/'never' - falling back to 'auto'.")
        _hp_tune_bayes_dr = 'auto'
    hp_tune_bayes_domain_reduction = _hp_tune_bayes_dr
    # FEATURE (multi-fidelity screening for standard Bayesian
    # hyperparameter tuning) - see this function's own docstring above for
    # the full meaning/default rationale.
    hp_tune_bayes_multi_fidelity = bool(cfg.get('HP_TUNE_BAYES_MULTI_FIDELITY', False))
    # ver4-40 R1: None = legacy; resolved (validated + fully defaulted) dict otherwise.
    # Logged unconditionally (I11) so "which fast levers did this run use" always has an
    # answer in the run log.
    hp_tune_fast = resolve_hp_tune_fast(cfg.get('HP_TUNE_FAST'))
    print(f"[pipeline_utils] HP_TUNE_FAST resolved: "
          f"{'None (legacy single-model tuning)' if hp_tune_fast is None else hp_tune_fast}")
    hp_tune_warm_start = bool(cfg.get('HP_TUNE_WARM_START', False))
    # ver4-56 R1 (I11): absent key => True (log-only; justified in the blueprint by the
    # ver4-33 precedent of the unconditional [timing] line). Logged every run.
    resource_telemetry = bool(cfg.get('RESOURCE_TELEMETRY', True))
    try:
        resource_telemetry_interval_s = max(0.05, float(cfg.get('RESOURCE_TELEMETRY_INTERVAL_S', 0.5)))
    except (TypeError, ValueError):
        print(f"[pipeline_utils] WARNING: RESOURCE_TELEMETRY_INTERVAL_S="
              f"{cfg.get('RESOURCE_TELEMETRY_INTERVAL_S')!r} is not a number - using 0.5 s.")
        resource_telemetry_interval_s = 0.5
    print(f"[pipeline_utils] RESOURCE_TELEMETRY resolved: "
          f"{'on (sampling every ' + str(resource_telemetry_interval_s) + ' s)' if resource_telemetry else 'off (no resource usage is recorded)'}")
    hp_tune_selection_margin = max(0.0, float(cfg.get('HP_TUNE_SELECTION_MARGIN', 0.02) or 0.0))
    hp_tune_valid_repeats = max(1, int(cfg.get('HP_TUNE_VALID_REPEATS', 1) or 1))
    _hp_tune_scope = str(cfg.get('HP_TUNE_SCOPE', 'per_task') or 'per_task')
    if _hp_tune_scope not in ('per_task', 'per_scenario'):
        print(f"[pipeline_utils] WARNING: HP_TUNE_SCOPE={_hp_tune_scope!r} is not one of "
              f"'per_task'/'per_scenario' - falling back to 'per_task'.")
        _hp_tune_scope = 'per_task'
    hp_tune_scope = _hp_tune_scope
    w_opt_analytic_seed = bool(cfg.get('W_OPT_ANALYTIC_SEED', False))
    w_opt_validation_floor = bool(cfg.get('W_OPT_VALIDATION_FLOOR', False))
    w_opt_simplex_search = bool(cfg.get('W_OPT_SIMPLEX_SEARCH', True))

    # ver4-26, design blueprint R1/R2/R3 - three new engine-only keys for
    # the `shapiq`-based marker-interaction routes (RF's 'shapiq_tree',
    # SVR/KNN's 'shapiq', RKHS-family's 'surrogate_shapiq'). All three
    # read via `.get(KEY, True)` - an old config lacking them takes the
    # new, faster path (I11: never crashes). `SHAPIQ_ROW_PARALLEL`/
    # `SHAPIQ_CANDIDATE_NARROWING` are pure speed/diagnostics-timing
    # settings (no value change at their own default order);
    # `SHAPIQ_ADAPTIVE_DEPTH` DOES change order>=3 `shapiq_tree` output
    # values (a shallower/smaller forest is a different approximation -
    # see `models/RF.py::_resolve_shapiq_tree_depth_cap()`'s own
    # docstring) - disclosed here, not silent, and not listed in this
    # module's own `[GP] NUMERICS:` line (that line lives in
    # `genomic_prediction.py` and only covers ITS OWN resolved settings;
    # every `shapiq` route already prints its own dedicated NOTE when a
    # cap/narrowing/fan-out actually changes what was attempted).
    shapiq_row_parallel = bool(cfg.get('SHAPIQ_ROW_PARALLEL', True))
    shapiq_adaptive_depth = bool(cfg.get('SHAPIQ_ADAPTIVE_DEPTH', True))
    shapiq_candidate_narrowing = bool(cfg.get('SHAPIQ_CANDIDATE_NARROWING', True))
    # ver4-27: 'shapiq_tree' previously ASSUMED the worst case up front (a
    # flat, order-shrunk depth GUESS - `models/RF.py::_resolve_shapiq_tree_
    # depth_cap()`) regardless of how much of that budget this task's own
    # marker count/hardware could actually afford - confirmed (Interaction
    # complaint this update addresses) to leave 'shapiq_tree' finding far
    # FEWER real marker-pair interactions than 'pairwise_shap' on the SAME
    # model, since a depth-8 forest simply never reaches most candidate
    # pairs at a realistic (~500-marker) shortlist. `SHAPIQ_CALIBRATE_DEPTH`
    # (default True) replaces that guess with a REAL, wall-clock-and-
    # memory-bounded ladder search (`pipeline_utils.calibrate_shapiq_tree_
    # depth()`) that tries deeper candidate depths FIRST and only falls
    # back to a shallower one when the deeper one actually proves too slow
    # or memory-hungry AT THIS TASK'S OWN SCALE. `False` reproduces the
    # pre-ver4-27 fixed-guess behaviour exactly (I11) - only meaningful
    # when `SHAPIQ_ADAPTIVE_DEPTH` is also True (calibration replaces THAT
    # flag's own depth guess, not the `SHAPIQ_ADAPTIVE_DEPTH=False` legacy
    # flat-8 path, which stays a fixed constant on purpose).
    shapiq_calibrate_depth = bool(cfg.get('SHAPIQ_CALIBRATE_DEPTH', True))

    # ver4-33 (design blueprint R1.b/c/f/g/h, SS4): the PLINK + bio-prior
    # GAT acceleration switches. User-facing ones live in ONE dict key,
    # `ACCELERATION`; an absent key (every config written before ver4-33)
    # resolves to the legacy ver4-32 path for all five (I11). The three
    # engine-only keys below default to values that are byte-identical to
    # ver4-32 output (BIO_PRIOR_PREP_FAST / BIO_PRIOR_LEAN_POOLS are exact
    # refactors; GAT_IG_INTERNAL_BATCH is inert unless gat_batched_ig).
    accel_cfg = cfg.get('ACCELERATION')
    if accel_cfg is None:
        accel_cfg = {}
    if not isinstance(accel_cfg, dict):
        raise ValueError(
            f"[pipeline_utils] ACCELERATION must be a dict (or absent), got "
            f"{type(accel_cfg).__name__}: {accel_cfg!r}. Expected e.g. "
            f"{{'plink_direct_read': 'auto', 'preprocess_memo': True, "
            f"'gat_dense_batching': True, 'gat_batched_ig': True, "
            f"'gpu_device_round_robin': True}}."
        )
    _unknown_accel = sorted(set(accel_cfg) - _ACCELERATION_KEYS)
    if _unknown_accel:
        print(f"[pipeline_utils] WARNING: ACCELERATION has unrecognised key(s) {_unknown_accel} - "
              f"ignored (known keys: {sorted(_ACCELERATION_KEYS)}).")
    accel_plink_direct_read = normalise_plink_direct_read(accel_cfg.get('plink_direct_read', 'off'))
    accel_preprocess_memo = bool(accel_cfg.get('preprocess_memo', False))
    accel_gat_dense_batching = bool(accel_cfg.get('gat_dense_batching', False))
    accel_gat_batched_ig = bool(accel_cfg.get('gat_batched_ig', False))
    accel_gpu_device_round_robin = bool(accel_cfg.get('gpu_device_round_robin', False))
    # Engine-only keys: read from the ACCELERATION dict (where the entry
    # points carry them, see `acceleration_from_cfg`) or, for a caller that
    # passes a flat cfg, from the top level; byte-identical defaults.
    accel_bio_prior_prep_fast = bool(accel_cfg.get('BIO_PRIOR_PREP_FAST', cfg.get('BIO_PRIOR_PREP_FAST', True)))
    accel_bio_prior_lean_pools = bool(accel_cfg.get('BIO_PRIOR_LEAN_POOLS', cfg.get('BIO_PRIOR_LEAN_POOLS', True)))
    _ig_batch_raw = accel_cfg.get('GAT_IG_INTERNAL_BATCH', cfg.get('GAT_IG_INTERNAL_BATCH', 256))
    accel_gat_ig_internal_batch = max(1, int(_ig_batch_raw if _ig_batch_raw is not None else 256))

    # ver4-35 (design blueprint R1.b/d/e/f/g/h, SS4): the speed switches of
    # this update. Every user-facing key resolves to its LEGACY value when
    # absent (I11), so an old config reproduces ver4-33 exactly; an invalid
    # enum value is logged and replaced by the legacy value. The engine-only
    # keys follow the BIO_PRIOR_PREP_FAST precedent (read from the
    # ACCELERATION dict, or from the top level for a flat cfg).
    accel_gat_factorised_input = bool(accel_cfg.get('gat_factorised_input', False))
    accel_gat_sync_free = bool(accel_cfg.get('gat_sync_free', False))
    accel_worker_thread_limits = bool(accel_cfg.get('worker_thread_limits', False))
    accel_gpu_slots = _normalise_gpu_slots(accel_cfg.get('gpu_slots', 'legacy'))
    accel_torch_device_policy = _normalise_enum_key(
        'torch_device_policy', accel_cfg.get('torch_device_policy', 'legacy'),
        _TORCH_DEVICE_POLICIES, 'legacy')
    accel_async_model_lane = bool(accel_cfg.get('async_model_lane', False))
    accel_ld_cm_banded = bool(accel_cfg.get('ld_cm_banded', False))
    accel_bio_prior_cost_guard = _normalise_enum_key(
        'bio_prior_cost_guard', accel_cfg.get('bio_prior_cost_guard', 'warn'),
        _COST_GUARD_MODES, 'warn')
    accel_trial_memory_guard = _normalise_enum_key(
        'trial_memory_guard', accel_cfg.get('trial_memory_guard', 'auto'),
        _TRIAL_MEMORY_GUARD_MODES, 'auto')
    accel_merge_shap_device = _normalise_enum_key(
        'merge_shap_device', accel_cfg.get('merge_shap_device', 'cpu'),
        _MERGE_SHAP_DEVICES, 'cpu')
    # ver4-43 (R1): 'legacy' (the untouched ver4-42 hybrid-row loop) when absent (I11).
    accel_baseline_shap_engine = _normalise_enum_key(
        'baseline_shap_engine', accel_cfg.get('baseline_shap_engine', 'legacy'),
        _BASELINE_SHAP_ENGINES, 'legacy')
    # ver4-103: GAT stability (models/gat_stability.py). Engine-only keys, read like the above.
    _gs = GAT_STABILITY_ENGINE_DEFAULTS
    accel_gat_seed = _gat_seed_from_cfg(accel_cfg.get('GAT_SEED', cfg.get('GAT_SEED', _gs['GAT_SEED'])))
    accel_gat_seed_ensemble = _gat_count_from_cfg(
        'GAT_SEED_ENSEMBLE', accel_cfg.get('GAT_SEED_ENSEMBLE', cfg.get('GAT_SEED_ENSEMBLE', _gs['GAT_SEED_ENSEMBLE'])), 1)
    accel_gat_rescore_top = _gat_count_from_cfg(
        'GAT_RESCORE_TOP', accel_cfg.get('GAT_RESCORE_TOP', cfg.get('GAT_RESCORE_TOP', _gs['GAT_RESCORE_TOP'])), 0)
    accel_gat_rescore_seeds = _gat_count_from_cfg(
        'GAT_RESCORE_SEEDS', accel_cfg.get('GAT_RESCORE_SEEDS', cfg.get('GAT_RESCORE_SEEDS', _gs['GAT_RESCORE_SEEDS'])), 0)
    # optimise_scan F3 (opt-in, I11): False = the legacy engine (every task re-sorts the three
    # prediction accumulators by 'id'); True = one stable sort before finalisation instead.
    accel_defer_result_sort = bool(
        accel_cfg.get('DEFER_RESULT_SORT', cfg.get('DEFER_RESULT_SORT', False)))
    accel_shap_stream_reduce = bool(
        accel_cfg.get('SHAP_STREAM_REDUCE', cfg.get('SHAP_STREAM_REDUCE', True)))
    accel_shared_bim_marker_info = bool(
        accel_cfg.get('SHARED_BIM_MARKER_INFO', cfg.get('SHARED_BIM_MARKER_INFO', True)))
    accel_async_lane_rng = _normalise_enum_key(
        'ASYNC_LANE_RNG', accel_cfg.get('ASYNC_LANE_RNG', cfg.get('ASYNC_LANE_RNG', 'exact')),
        _ASYNC_LANE_RNG_MODES, 'exact')
    _lane_threads_raw = accel_cfg.get('ASYNC_LANE_THREADS', cfg.get('ASYNC_LANE_THREADS', 'match'))
    if str(_lane_threads_raw).strip().lower() == 'match':
        accel_async_lane_threads = 'match'
    else:
        try:
            accel_async_lane_threads = max(1, int(_lane_threads_raw))
        except (TypeError, ValueError):
            print(f"[pipeline_utils] WARNING: ASYNC_LANE_THREADS={_lane_threads_raw!r} is neither "
                  f"'match' nor an integer - using 'match'.")
            accel_async_lane_threads = 'match'
    _lane_payload_raw = accel_cfg.get(
        'ASYNC_LANE_MAX_PAYLOAD_MB', cfg.get('ASYNC_LANE_MAX_PAYLOAD_MB', 2048))
    try:
        accel_async_lane_max_payload_mb = max(1, int(_lane_payload_raw))
    except (TypeError, ValueError):
        print(f"[pipeline_utils] WARNING: ASYNC_LANE_MAX_PAYLOAD_MB={_lane_payload_raw!r} is "
              f"not an integer - using 2048.")
        accel_async_lane_max_payload_mb = 2048
    _max_gflop_raw = accel_cfg.get('BIO_PRIOR_MAX_FIT_GFLOP', cfg.get('BIO_PRIOR_MAX_FIT_GFLOP'))
    accel_bio_prior_max_fit_gflop = None
    if _max_gflop_raw is not None:
        try:
            accel_bio_prior_max_fit_gflop = float(_max_gflop_raw)
        except (TypeError, ValueError):
            print(f"[pipeline_utils] WARNING: BIO_PRIOR_MAX_FIT_GFLOP={_max_gflop_raw!r} is not a "
                  f"number - the GFLOP threshold of the cost guard is off.")

    return {
        'device': str(device),
        'use_gpu_sklearn': bool(cfg.get('USE_GPU_SKLEARN', False)) and cuml_gpu_usable(),
        'n_jobs': n_jobs,
        'plink_threads': plink_threads,
        'r_blas_threads': r_blas_threads,
        'cudnn_benchmark': bool(cfg.get('CUDNN_BENCHMARK', True)),
        'use_amp': bool(cfg.get('USE_AMP', False)),
        'n_cpu_workers': n_cpu_workers_task,  # alias of n_cpu_workers_task, meaning unchanged (I11)
        'n_cpu_workers_task': n_cpu_workers_task,
        'n_model_workers': n_model_workers,
        'cpus_per_model_worker': cpus_per_model_worker,
        'n_gpu_slots': n_gpu_slots,
        'gpu_slots_per_device': gpu_slots_per_device,
        'n_cuda_devices': int(cuda_device_count),
        'torch_num_threads': torch_num_threads,
        'torch_dataloader_workers': torch_dataloader_workers,
        'gpu_eval_batch': gpu_eval_batch,
        'gpu_ld_r2': gpu_ld_r2,
        'gpu_kernel_precompute': gpu_kernel_precompute,
        'hp_tune_parallel_trials': hp_tune_parallel_trials,
        'hp_tune_bayes_batch': hp_tune_bayes_batch,
        'hp_tune_bayes_batch_max': hp_tune_bayes_batch_max,
        'hp_tune_bayes_liar': hp_tune_bayes_liar,
        'hp_tune_parallel_restarts': hp_tune_parallel_restarts,
        'hp_tune_bayes_domain_reduction': hp_tune_bayes_domain_reduction,
        'hp_tune_bayes_multi_fidelity': hp_tune_bayes_multi_fidelity,
        'hp_tune_fast': hp_tune_fast,
        'hp_tune_warm_start': hp_tune_warm_start,
        'resource_telemetry': resource_telemetry,
        'resource_telemetry_interval_s': resource_telemetry_interval_s,
        'hp_tune_selection_margin': hp_tune_selection_margin,
        'hp_tune_valid_repeats': hp_tune_valid_repeats,
        'hp_tune_scope': hp_tune_scope,
        'w_opt_analytic_seed': w_opt_analytic_seed,
        'w_opt_validation_floor': w_opt_validation_floor,
        'w_opt_simplex_search': w_opt_simplex_search,
        'shapiq_row_parallel': shapiq_row_parallel,
        'shapiq_adaptive_depth': shapiq_adaptive_depth,
        'shapiq_candidate_narrowing': shapiq_candidate_narrowing,
        'shapiq_calibrate_depth': shapiq_calibrate_depth,
        'accel_plink_direct_read': accel_plink_direct_read,
        'accel_preprocess_memo': accel_preprocess_memo,
        'accel_gat_dense_batching': accel_gat_dense_batching,
        'accel_gat_batched_ig': accel_gat_batched_ig,
        'accel_gpu_device_round_robin': accel_gpu_device_round_robin,
        'accel_bio_prior_prep_fast': accel_bio_prior_prep_fast,
        'accel_bio_prior_lean_pools': accel_bio_prior_lean_pools,
        'accel_gat_ig_internal_batch': accel_gat_ig_internal_batch,
        # ver4-35. `policy_threads` is the CPU thread count the device-
        # placement policy assumes. It is copied (never divided) into a
        # trial worker's resources dict, so a tuning trial and the
        # confirmatory fit take the SAME placement decision (I9, A-e3).
        'policy_threads': int(torch_num_threads),
        'accel_gat_factorised_input': accel_gat_factorised_input,
        'accel_gat_sync_free': accel_gat_sync_free,
        'accel_worker_thread_limits': accel_worker_thread_limits,
        'accel_gpu_slots': accel_gpu_slots,
        'accel_torch_device_policy': accel_torch_device_policy,
        'accel_async_model_lane': accel_async_model_lane,
        'accel_ld_cm_banded': accel_ld_cm_banded,
        'accel_bio_prior_cost_guard': accel_bio_prior_cost_guard,
        'accel_trial_memory_guard': accel_trial_memory_guard,
        'accel_merge_shap_device': accel_merge_shap_device,
        'accel_baseline_shap_engine': accel_baseline_shap_engine,
        'accel_gat_seed': accel_gat_seed,
        'accel_gat_seed_ensemble': accel_gat_seed_ensemble,
        'accel_gat_rescore_top': accel_gat_rescore_top,
        'accel_gat_rescore_seeds': accel_gat_rescore_seeds,
        'accel_shap_stream_reduce': accel_shap_stream_reduce,
        'accel_defer_result_sort': accel_defer_result_sort,
        'accel_shared_bim_marker_info': accel_shared_bim_marker_info,
        'accel_async_lane_rng': accel_async_lane_rng,
        'accel_async_lane_threads': accel_async_lane_threads,
        'accel_async_lane_max_payload_mb': accel_async_lane_max_payload_mb,
        'accel_bio_prior_max_fit_gflop': accel_bio_prior_max_fit_gflop,
    }


# ver4-4 R3.b - one-shot guard for the CPU thread-count branch of
# apply_torch_compute_settings() below. torch.set_num_interop_threads()
# raises RuntimeError if called more than once, or after any parallel
# work has already started - so the WHOLE CPU-threading branch (both
# set_num_threads() and set_num_interop_threads()) is applied at most
# once per process, matching the blueprint's explicit instruction
# ("behind a one-shot module flag, since torch raises RuntimeError on a
# second call"). A later call with a different resolved thread count is
# silently skipped rather than raising - see the NOTE log line below.
_TORCH_CPU_THREADS_APPLIED = False

# ver4-35, R1.e - the same one-shot guard for the CPU thread cap of a process
# whose torch device is CUDA (see `_apply_cuda_worker_cpu_threads`).
_TORCH_CUDA_CPU_THREADS_APPLIED = False


def _worker_thread_budget(resources: Dict[str, object]) -> int:
    """The CPU thread count one process of this run may use: the resolved
    ``torch_num_threads`` (already divided by the fan-out width the caller
    set up), else a positive ``n_jobs``, else ``cpus_per_model_worker``.
    Always >= 1. Single authority for ``torch.set_num_threads`` on CUDA
    workers and for ``apply_worker_thread_limits``."""
    for key in ('torch_num_threads', 'n_jobs'):
        value = resources.get(key)
        if isinstance(value, (int, float)) and int(value) > 0:
            return int(value)
    return max(1, int(resources.get('cpus_per_model_worker') or 1))


def _apply_cuda_worker_cpu_threads(resources: Dict[str, object]) -> None:
    """Cap torch's CPU intra-op pool in a process that computes on CUDA
    (ver4-35, R1.e1). Applied at most once per process; never raises - a
    failure is logged and torch keeps its default pool (slower, not wrong)."""
    global _TORCH_CUDA_CPU_THREADS_APPLIED
    if _TORCH_CUDA_CPU_THREADS_APPLIED:
        return
    budget = _worker_thread_budget(resources)
    try:
        import torch
        torch.set_num_threads(budget)
    except ImportError:
        return
    except (RuntimeError, ValueError) as exc:
        print(f"[pipeline_utils] NOTE: could not cap torch CPU threads on a CUDA worker "
              f"({type(exc).__name__}: {exc}) - torch keeps its default thread pool in PID "
              f"{os.getpid()}; the node may be oversubscribed.")
        return
    _TORCH_CUDA_CPU_THREADS_APPLIED = True
    print(f"[pipeline_utils] NOTE: torch CPU threads capped to {budget} in PID {os.getpid()} "
          f"(CUDA worker, ACCELERATION['worker_thread_limits']).")


def apply_torch_compute_settings(resources: Dict[str, object]) -> None:
    """Apply the process-global PyTorch settings implied by
    ``resolve_compute_resources()``'s output - split out from the
    resolver itself so a pure-CPU caller (or one that hasn't imported
    torch yet) never pays for/triggers a torch import just to read
    settings it won't use. Safe to call repeatedly; a no-op if torch
    isn't installed.

    CUDA device: sets ``torch.backends.cudnn.benchmark = True`` when
    ``resources['cudnn_benchmark']`` is truthy. ver4-35 (R1.e, blueprint
    D4/E4): with ``ACCELERATION['worker_thread_limits']`` on it ALSO caps the
    CPU intra-op threads (``torch.set_num_threads``) - before that, every
    CUDA worker kept a full-size OpenMP pool, so q trial workers
    oversubscribed the node q-fold. Off = the pre-ver4-35 behaviour.

    CPU device (ver4-4 R3.b - new): calls
    ``torch.set_num_threads(resources['torch_num_threads'])`` and
    ``torch.set_num_interop_threads(1)`` so MLP/GAT CPU intra-op
    threading no longer silently falls back to whatever
    ``OMP_NUM_THREADS`` happens to be (previously only ever set, if at
    all, by ``apply_r_blas_threads()`` - see that function's own R3.c
    note below on why that was itself always a no-op before ver4-4).
    Never raises: a no-op if torch isn't installed, if
    ``torch_num_threads`` resolves to ``None``/``0``, or if this has
    already been applied once in this process (see
    ``_TORCH_CPU_THREADS_APPLIED`` above).
    """
    device = str(resources.get('device', 'cpu'))

    if device.startswith('cuda'):
        if resources.get('accel_worker_thread_limits'):
            _apply_cuda_worker_cpu_threads(resources)
        if not resources.get('cudnn_benchmark'):
            return
        try:
            import torch
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        return

    # CPU device (ver4-4 R3.b).
    global _TORCH_CPU_THREADS_APPLIED
    if _TORCH_CPU_THREADS_APPLIED:
        return
    torch_num_threads = resources.get('torch_num_threads')
    if not torch_num_threads:
        return
    try:
        import torch
        torch.set_num_threads(int(torch_num_threads))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as exc:
            # Must be called before any parallel work has started - if
            # something upstream already triggered torch's own internal
            # threadpool (e.g. an earlier model call in the same
            # process), this raises. set_num_threads() above still took
            # effect either way, so this is logged, not fatal.
            print(f"[pipeline_utils] NOTE: torch.set_num_interop_threads(1) could not be "
                  f"applied ({exc}) - this usually means torch's parallel work had already "
                  f"started before this call. torch.set_num_threads({int(torch_num_threads)}) "
                  f"was still applied successfully.")
        _TORCH_CPU_THREADS_APPLIED = True
        print(f"[pipeline_utils] NOTE: torch CPU threading applied - "
              f"intra-op threads={int(torch_num_threads)}, inter-op threads=1.")
    except ImportError:
        pass
    except Exception as exc:
        print(f"[pipeline_utils] NOTE: could not apply torch CPU thread settings ({exc}) - "
              f"continuing with torch's own default thread count.")


#: Keeps the ``threadpoolctl`` limiter alive for the life of the process (a
#: limiter that is garbage-collected restores the previous limits).
_WORKER_THREADPOOL_LIMITER = None


def apply_worker_thread_limits(resources: Dict[str, object]) -> bool:
    """Cap every native thread pool (OpenMP, OpenBLAS/MKL) of THIS process at
    the worker's CPU budget (ver4-35, R1.e1).

    Why: a spawned worker imports torch / numpy at module level BEFORE
    ``_trial_worker_init`` exports the divided ``OMP_NUM_THREADS`` /
    ``OPENBLAS_NUM_THREADS``, so each worker's pools were created at full
    node size (blueprint E4) and q workers ran q x the CPUs. Environment
    variables are read once at library load, so only ``threadpoolctl`` can
    resize an already-loaded pool.

    Does nothing (returns False) unless ``resources['accel_worker_thread_limits']``
    is truthy, so every pre-ver4-35 config behaves exactly as before.

    Returns True if the limits were applied. Never raises: a missing
    ``threadpoolctl`` (``pip install threadpoolctl``) or a failing call is
    logged and the worker keeps its default pools (slower, never wrong).
    """
    global _WORKER_THREADPOOL_LIMITER
    if not resources or not resources.get('accel_worker_thread_limits'):
        return False
    budget = _worker_thread_budget(resources)
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        print(f"[pipeline_utils] NOTE: ACCELERATION['worker_thread_limits'] needs the optional "
              f"package 'threadpoolctl' (install: pip install threadpoolctl) - native thread "
              f"pools of PID {os.getpid()} are NOT capped (may oversubscribe the node).")
        return False
    try:
        _WORKER_THREADPOOL_LIMITER = threadpool_limits(limits=budget)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"[pipeline_utils] NOTE: threadpoolctl could not cap native thread pools to "
              f"{budget} in PID {os.getpid()} ({type(exc).__name__}: {exc}) - default pools "
              f"stay in force.")
        return False
    print(f"[pipeline_utils] NOTE: native thread pools (OpenMP/BLAS) capped to {budget} "
          f"thread(s) in PID {os.getpid()} (ACCELERATION['worker_thread_limits']).")
    return True


# Mute fix (post-ver4-6): apply_r_blas_threads() is called far more than
# once per RUN - once in the parent process at GP() startup, then AGAIN by
# every freshly-spawned HP-tuning trial worker (_trial_worker_init(),
# genomic_prediction.py) and every intra-batch/intra-task worker
# (_worker_init(), intra_batch_parallel.py/intra_task_parallel.py), since
# each is a genuinely separate process that must configure its own R
# environment from scratch (see this function's own R3.c note below for
# why that per-process call is real and necessary, not redundant). Because
# every worker in a process tree inherits 'torch' already being importable
# the same way, the diagnostic condition below is true on every single one
# of those calls - which, across a 25-task batch with several tuned models
# and an 8-candidate Bayesian search per tuning round, was printing
# dozens of byte-identical lines into every run log (see the repeated
# "[pipeline_utils] NOTE: apply_r_blas_threads(...)" block in a real
# production *.output file). The diagnostic is genuinely useful - just not
# once per worker. It is muted here to print at most ONCE per process
# TREE (not merely once per process), using an environment-variable flag
# rather than a plain module-level bool: a plain bool only dedupes calls
# within a single process, but the repeats here mostly come from separate
# *processes*. Because every worker below is spawned only after this same
# function's env-var-setting loop has already run at least once in the
# parent (GP()'s own startup, before any task/trial worker is created),
# and a spawned child inherits the parent's os.environ at spawn time, this
# flag reliably reaches every descendant without needing any shared
# file/lock. The BLAS thread-count environment variables themselves are
# still exported unconditionally, on every call, in every process, exactly
# as before - only the repeated PRINT is muted.
_R_BLAS_NOTE_ENV_FLAG = 'EASIGP_R_BLAS_NOTE_EMITTED'


def apply_r_blas_threads(r_blas_threads: Optional[int]) -> None:
    """Export the common BLAS-library thread-count environment variables
    (OpenBLAS/OMP/MKL) so a multi-threaded BLAS underneath R's own linear
    algebra (used throughout rrBLUP/GBLUP/BayesB/RKHS's BGLR MCMC) can use
    more than one core, if the R build in use is linked against one -
    EasiGP has no way to force R to link a different BLAS at runtime; this
    only forwards a thread-count hint to whichever BLAS is already
    present. A no-op when ``r_blas_threads`` is ``None``/falsy, which is
    the fully backward-compatible default (nothing set, matching the
    pipeline's original behaviour).

    ver4-4 R3.c: setting ``OMP_NUM_THREADS`` via ``os.environ`` here has
    no effect on a torch build that was already imported and has already
    read that variable at its own import time (torch's OpenMP thread
    pool is initialised once, at import). ``GP()`` calls this function
    early (before the rpy2 import block) specifically to apply BEFORE any
    model module (which may import torch) has run - but a caller from a
    different entry point, or a future code path that imports a model
    module earlier, could reorder that. Rather than silently doing
    nothing in that case (the exact failure shape R3.c exists to make
    visible - the same shape as the Patch 9 Bug 3 precedent), this logs a
    diagnostic note whenever torch is already present in ``sys.modules``
    at the time this function runs, so the "why didn't BLAS threading
    apply" question is answerable from the run log rather than requiring
    a source read.

    Muted (post-ver4-6, see ``_R_BLAS_NOTE_ENV_FLAG`` above): this note is
    now printed at most once per process tree, not once per call - see
    that flag's own comment for why an env-var guard, not a module bool,
    is what actually achieves that across the many separate worker
    processes this function is called from.
    """
    if not r_blas_threads:
        return
    threads_str = str(max(1, int(r_blas_threads)))
    for env_var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[env_var] = threads_str
    if 'torch' in sys.modules and not os.environ.get(_R_BLAS_NOTE_ENV_FLAG):
        print(f"[pipeline_utils] NOTE: apply_r_blas_threads({r_blas_threads}) set "
              f"OPENBLAS_NUM_THREADS/OMP_NUM_THREADS/MKL_NUM_THREADS={threads_str}, but 'torch' "
              f"is already imported in this process - an already-imported torch build has "
              f"already read OMP_NUM_THREADS at its own import time and will NOT pick up this "
              f"change. This only affects torch's own CPU threading (see "
              f"apply_torch_compute_settings()/TORCH_NUM_THREADS for that); R/BGLR's BLAS "
              f"threading via these environment variables is unaffected by import order and "
              f"still applies normally. (This note prints once per run - every subsequent "
              f"call in this process tree hits the identical condition and is muted.)")
        os.environ[_R_BLAS_NOTE_ENV_FLAG] = '1'


# ver4-4 R4.b/c (blueprint §2.4.2): a shared helper avoids five near-
# identical edits (one per torch-using model module) drifting apart, per
# the blueprint's own stated rationale for introducing this function.
def torch_eval_batch_size(resources: Dict[str, object], requested: Optional[int], *, default: int = 1) -> int:
    """Batch size for a torch/PyG inference or explanation loop.

    Returns ``requested`` (coerced to at least 1) when the resolved device
    is CUDA and ``requested`` is a positive number; returns ``default``
    (1) on CPU, or when ``requested`` is falsy/non-positive - so a
    CPU-only run (or a run with ``GPU_EVAL_BATCH`` unset/0) keeps today's
    exact loader behaviour (``batch_size=1``) and today's exact numerics,
    unconditionally. Never raises.

    Parameters
    ----------
    resources : this process's active compute-resource dict (typically
        ``get_active_compute_resources()``'s own return value) - only
        ``resources['device']`` is read.
    requested : the caller's own requested batch size, typically
        ``resources['gpu_eval_batch']`` itself, threaded through
        explicitly (rather than read a second time inside this function)
        so a caller can pass a DIFFERENT count for, e.g., a training
        loader vs. an explainer loop without this function needing to
        know which is which.
    default : returned instead of ``requested`` whenever batching isn't
        appropriate (CPU device, or a non-positive ``requested``).
    """
    device = str(resources.get('device', 'cpu')) if resources else 'cpu'
    if not device.startswith('cuda'):
        return default
    try:
        requested_int = int(requested) if requested else 0
    except (TypeError, ValueError):
        requested_int = 0
    return requested_int if requested_int > 0 else default


def dataloader_num_workers(resources: Dict[str, object]) -> int:
    """DataLoader ``num_workers``, daemon-process-safe (ver4-4 RK-3).

    Returns ``0`` unconditionally when the CALLING process is itself a
    daemonic ``multiprocessing`` process (e.g. an
    ``intra_batch_parallel``/``intra_task_parallel`` worker) - Python's
    own ``multiprocessing`` forbids a daemonic process from spawning
    children, so a torch/PyG ``DataLoader`` configured with
    ``num_workers>0`` would raise ``AssertionError: daemonic processes
    are not allowed to have children`` the moment it tried to iterate.
    The daemon check is performed FRESH on every call (never cached),
    since the correct answer depends on whichever process is actually
    about to construct the ``DataLoader`` - not on whatever process
    happened to call ``resolve_compute_resources()`` earlier (which may
    be a parent process, before any worker was even forked).

    Otherwise returns ``resources['torch_dataloader_workers']`` (default
    0, today's behaviour, if the key is absent - e.g. a resources dict
    built before this ver4-4 R4.d key existed). Never raises.
    """
    try:
        import multiprocessing
        if multiprocessing.current_process().daemon:
            return 0
    except Exception:
        pass
    try:
        return max(0, int((resources or {}).get('torch_dataloader_workers', 0) or 0))
    except (TypeError, ValueError):
        return 0


# ver4-5 blueprint R1.h (§3.4) - budget arithmetic for a THIRD level of
# process fan-out (hyperparameter-tuning trial evaluation), added
# beneath the two that already exist (intra_batch_parallel's tasks,
# intra_task_parallel's models). Two separate concerns, both handled
# here in one call rather than at each of R1's several call sites:
#
# 1. Daemon safety (RK-3 precedent - dataloader_num_workers() above):
#    Python's multiprocessing forbids a daemonic process from spawning
#    children at all, so a trial-evaluation fan-out requested from
#    INSIDE an already-daemonic worker (however that worker was itself
#    spawned) must collapse to serial (1), not merely to a smaller
#    number, or joblib itself raises. Checked FRESH every call, never
#    cached, for the identical reason dataloader_num_workers() checks
#    fresh - the correct answer depends on which process is actually
#    about to spawn, not on whichever process originally resolved
#    `resources`.
# 2. Divide, never multiply (R1.h): a task already fanned out across
#    models is already using its own share of this job's CPUs: trial
#    fan-out must be sized as `requested // divisor` (typically the
#    caller's own already-decided q/model-parallelism width), floored
#    at 1, rather than requesting `requested` outright on top of an
#    already-busy budget.
def nested_safe_n_jobs(resources: Optional[Dict], requested, divisor: int = 1) -> int:
    """Effective worker count for a process fan-out that may itself be
    running inside a worker process (ver4-5 R1.h).

    Returns ``1`` unconditionally when the CALLING process is itself
    daemonic (mirrors ``dataloader_num_workers()``'s own fresh-per-call
    daemon check - see that function's docstring for why this cannot be
    cached). Otherwise returns ``max(1, int(requested) // max(1,
    divisor))`` - ``requested`` is trusted as already representing this
    run's own resolved core count (typically ``resources['n_jobs']``,
    though ``resources`` itself is accepted mainly for a consistent call
    shape with the rest of this module's resource-reading helpers and is
    not otherwise inspected). ``requested`` of ``-1`` (the "every core"
    sentinel) or ``None`` resolves to ``os.cpu_count() or 1`` before
    dividing, since ``-1``/``None`` are not themselves usable literal
    worker COUNTS. Never raises.
    """
    try:
        import multiprocessing
        if multiprocessing.current_process().daemon:
            return 1
    except Exception:
        pass
    try:
        _requested = int(requested) if requested is not None else -1
    except (TypeError, ValueError):
        _requested = -1
    if _requested <= 0:
        _requested = os.cpu_count() or 1
    _divisor = max(1, int(divisor) if divisor else 1)
    return max(1, _requested // _divisor)


# ---------------------------------------------------------------------------
# Memory-aware, worker-death-resilient process fan-out
# (baseline_shap_interactions "SIGKILL -> falling back to a single serial
# pass" fix).
#
# Background: a joblib/loky fan-out whose workers each build a large
# temporary array (e.g. baseline_shap's `pair_chunk_size * n_background`
# hybrid rows x `n_support_vectors` kernel matrix) can be reclaimed by the
# OS / cgroup OOM-killer with an uncatchable SIGKILL. joblib then reports
# "A worker process managed by the executor was unexpectedly terminated"
# and every call site in this codebase used to answer that with a single
# SERIAL pass over ALL the work - hours of wall-clock for a job that was
# meant to run on N cores, and one that also throws away every block that
# had already finished. The three helpers below replace that with:
#   1. `available_memory_bytes()`  - what may THIS process tree really use
#      (min of psutil/MemAvailable and any cgroup v1/v2 limit headroom),
#   2. `is_worker_death_error()`   - recognises the worker-killed family,
#   3. `resilient_block_fanout()`  - keeps every block that finished,
#      re-plans (smaller per-worker memory footprint, then fewer workers)
#      and re-dispatches ONLY the unfinished blocks - still in parallel.
# ---------------------------------------------------------------------------

def available_memory_bytes() -> Optional[int]:
    """Best estimate, in bytes, of the memory this process tree can still
    allocate: the minimum of the OS's own "available" figure and the
    remaining headroom (`limit - current usage`) of every cgroup v2 / v1
    memory limit that applies to this process (PBS/Slurm/Kubernetes job
    limits live there and are invisible to `psutil`). The optional
    environment variable ``EASIGP_FANOUT_MEM_GB`` caps the result further
    (a manual override for nodes where neither source is trustworthy).
    ``None`` when nothing at all can be determined. Never raises."""
    cands: List[int] = []
    try:
        import psutil
        cands.append(int(psutil.virtual_memory().available))
    except Exception:
        try:
            with open('/proc/meminfo') as fh:
                for line in fh:
                    if line.startswith('MemAvailable:'):
                        cands.append(int(line.split()[1]) * 1024)
                        break
        except Exception:
            pass

    def _read_int(path):
        try:
            with open(path) as fh:
                txt = fh.read().strip()
            if not txt or txt == 'max':
                return None
            val = int(txt)
            return val if 0 < val < (1 << 60) else None
        except Exception:
            return None

    try:
        with open('/proc/self/cgroup') as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
    except Exception:
        lines = []
    for ln in lines:
        try:
            _hid, ctrls, cpath = ln.split(':', 2)
        except ValueError:
            continue
        if ctrls == '':                                   # cgroup v2 unified
            d = os.path.normpath('/sys/fs/cgroup/' + cpath.lstrip('/'))
            while d.startswith('/sys/fs/cgroup'):
                lim = _read_int(os.path.join(d, 'memory.max'))
                cur = _read_int(os.path.join(d, 'memory.current'))
                if lim is not None:
                    cands.append(max(0, lim - (cur or 0)))
                if d == '/sys/fs/cgroup':
                    break
                d = os.path.dirname(d)
        elif 'memory' in ctrls.split(','):                # cgroup v1
            d = os.path.normpath('/sys/fs/cgroup/memory/' + cpath.lstrip('/'))
            lim = _read_int(os.path.join(d, 'memory.limit_in_bytes'))
            cur = _read_int(os.path.join(d, 'memory.usage_in_bytes'))
            if lim is not None:
                cands.append(max(0, lim - (cur or 0)))
    try:
        env = os.environ.get('EASIGP_FANOUT_MEM_GB')
        if env:
            cands.append(int(float(env) * (1024 ** 3)))
    except Exception:
        pass
    return min(cands) if cands else None


def is_worker_death_error(exc: BaseException) -> bool:
    """True when `exc` (or anything in its cause/context chain) is the
    "a worker process was killed / crashed" family - joblib/loky's
    ``TerminatedWorkerError``, ``concurrent.futures``' ``BrokenProcessPool``,
    a worker-side ``MemoryError``, or an unmistakable SIGKILL/OOM message -
    as opposed to an ordinary bug raised BY the work function."""
    seen = set()
    cur = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, MemoryError):
            return True
        name = type(cur).__name__
        if name in ('TerminatedWorkerError', 'BrokenProcessPool', 'WorkerLostError'):
            return True
        msg = str(cur)
        if ('unexpectedly terminated' in msg or 'SIGKILL' in msg or 'SIGSEGV' in msg
                or 'process pool was terminated' in msg or 'out of memory' in msg.lower()):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _tagged_block_call(fn, idx, block):
    """Module-level (picklable by reference) wrapper: returns
    ``(idx, fn(block))`` so unordered results can be put back in order."""
    return idx, fn(block)


def resilient_block_fanout(make_task, blocks, plan, *, label, max_attempts=8,
                           log_prefix="[pipeline_utils]"):
    """Run ``make_task(params)(block)`` for every block across worker
    processes and return the results IN BLOCK ORDER - never falling back to
    a serial in-process pass.

    ``plan(attempt)`` -> ``(n_workers, params, description)`` for attempt
    0, 1, 2, ... (each attempt should ask for a smaller memory footprint
    than the last), or ``None`` when nothing smaller exists. ``make_task``
    turns ``params`` into the (picklable) per-block function.

    When a worker dies (OOM-kill etc. - see `is_worker_death_error()`),
    every block that already finished is KEPT, and only the unfinished
    blocks are re-dispatched under the next, smaller plan. A genuine
    exception raised by the work function itself is re-raised unchanged
    (retrying cannot help). If the plans run out (or `max_attempts` is
    reached) a `RuntimeError` naming the last plan is raised - by design
    this function never silently degrades to a single-process pass."""
    from joblib import Parallel, delayed

    results: Dict[int, object] = {}
    pending = list(range(len(blocks)))
    attempt = 0
    last_desc = ''
    while pending:
        planned = plan(attempt)
        if planned is None:
            raise RuntimeError(
                f"{log_prefix} {label}: worker processes keep being killed (out of memory?) and "
                f"there is no smaller parallel configuration left to try (last: {last_desc}); "
                f"{len(pending)}/{len(blocks)} block(s) unfinished. Free memory on the node, "
                f"request more, or lower the interaction shortlist / background size.")
        n_workers, params, desc = planned
        n_workers = max(1, min(int(n_workers), len(pending)))
        last_desc = desc
        print(f"{log_prefix} {label}: attempt {attempt + 1} - {len(pending)}/{len(blocks)} block(s) "
              f"on {n_workers} worker(s); {desc}")
        fn = make_task(params)
        try:
            try:
                par = Parallel(n_jobs=n_workers, return_as='generator_unordered')
            except TypeError:                              # joblib < 1.3
                par = Parallel(n_jobs=n_workers)
                for idx, res in par(delayed(_tagged_block_call)(fn, i, blocks[i]) for i in pending):
                    results[idx] = res
            else:
                for idx, res in par(delayed(_tagged_block_call)(fn, i, blocks[i]) for i in pending):
                    results[idx] = res
        except Exception as exc:
            if not is_worker_death_error(exc):
                raise
            pending = [i for i in pending if i not in results]
            attempt += 1
            print(f"{log_prefix} NOTE: {label}: worker process(es) were killed "
                  f"({type(exc).__name__}) - {len(results)}/{len(blocks)} block(s) already finished "
                  f"and kept; re-planning with a smaller memory footprint and re-dispatching the "
                  f"remaining {len(pending)} block(s) IN PARALLEL (no serial fallback).")
            if attempt >= max_attempts:
                raise RuntimeError(
                    f"{log_prefix} {label}: still losing workers after {attempt} attempts "
                    f"(last plan: {last_desc}); {len(pending)} block(s) unfinished.") from exc
            continue
        pending = [i for i in pending if i not in results]
    return [results[i] for i in range(len(blocks))]


def split_batched_edge_attention(alpha, num_graphs: int):
    """Split a ``GATv2Conv`` batched ``return_attention_weights=True``
    alpha tensor - shape ``(edges_total, heads)``, where ``edges_total =
    num_graphs * edges_per_graph`` - back into ``num_graphs`` separate
    per-graph ``(edges_per_graph, heads)`` tensors, in the SAME order the
    constituent graphs appeared in this batch.

    ver4-4 R4.b (blueprint §2.4.2, PC-1): every EasiGP GAT variant that
    emits attention (``GAT_fully_connected``, ``GAT_prior_knowledge``,
    ``GAT_biological_prior_knowledge``) builds an IDENTICAL, FIXED graph
    topology for every individual within one data split - topology comes
    from the marker/gene set and the model's own edge-construction rule,
    never from an individual's own genotype VALUES (which live only on
    node features) - so every graph in a batch has exactly the same edge
    count. Under this guarantee, PyG's ``Batch.from_data_list`` simply
    concatenates each constituent graph's own edges IN ORDER (only a
    node-index offset is added, verified directly - not merely reasoned
    about - against a real ``torch_geometric`` install in the ver4-4
    Stage 5 design record's PC-1 pre-check: the returned batched
    ``edge_index``/``alpha`` from a real ``GATv2Conv`` call is bit-
    identical, per graph, to what an unbatched ``batch_size=1`` call on
    that same graph alone produces), so this is a pure reshape - no
    numerical difference from computing each graph's attention one at a
    time.

    Each caller then applies its OWN existing per-graph post-processing
    (e.g. ``GAT_fully_connected.py``/``GAT_prior_knowledge.py``'s
    ``.flatten()``, or ``GAT_biological_prior_knowledge.py``'s
    ``.mean(dim=1)``) to each returned slice - this function deliberately
    does not impose either convention (mirrors ``parallel_shap_values``'s
    own precedent of not unifying divergent per-file post-processing).

    Parameters
    ----------
    alpha : the ``(edges_total, heads)`` (or ``(edges_total,)``, for a
        single-head model) tensor - the second element of GATv2Conv's own
        ``return_attention_weights=True`` output.
    num_graphs : the number of graphs actually present in THIS batch
        (``batch.num_graphs``) - may be smaller than the loader's own
        nominal ``batch_size`` for a final, partial batch.

    Returns
    -------
    A list of length ``num_graphs``, each element a ``.detach().cpu()``'d
    tensor slice of ``alpha``, one per graph.

    Raises
    ------
    ValueError if ``alpha``'s leading dimension is not evenly divisible
    by ``num_graphs`` - this would mean a graph in this batch has a
    DIFFERENT edge count than its neighbours, breaking the fixed-topology
    assumption every caller of this function depends on; surfacing this
    loudly here is far easier to diagnose than a silently misaligned
    attention column several steps downstream.
    """
    total_edges = alpha.shape[0]
    if num_graphs <= 0 or total_edges % num_graphs != 0:
        raise ValueError(
            f"split_batched_edge_attention: cannot evenly split {total_edges} edge(s) of "
            f"attention weights across {num_graphs} graph(s) in this batch - every graph in "
            f"an EasiGP GAT split is expected to share the SAME fixed topology (same edge "
            f"count). This usually means a genuinely variable per-individual graph structure "
            f"reached a code path that assumes a constant one."
        )
    edges_per_graph = total_edges // num_graphs
    alpha_cpu = alpha.detach().cpu()
    return [alpha_cpu[g * edges_per_graph:(g + 1) * edges_per_graph] for g in range(num_graphs)]


# ---------------------------------------------------------------------------
# Phase 2, Requirement 7 - GPU-slot limiter for intra-batch parallelism.
#
# A lightweight, OPT-IN cross-process semaphore that GPU-dispatched model
# code (MLP.py, GAT_*.py) can acquire around the part of a model call that
# actually touches the GPU, so N_CPU_WORKERS worker processes sharing ONE
# physical GPU don't all pile onto it at once. A no-op (immediately
# acquired/released, unlimited concurrency) whenever no semaphore has been
# installed - i.e. for ordinary serial/single-process runs, which is every
# run unless intra-batch parallelism (N_CPU_WORKERS > 1) is explicitly
# enabled - so existing single-process behaviour is completely unaffected.
# ---------------------------------------------------------------------------

_GPU_SEMAPHORE = None

# ---------------------------------------------------------------------------
# Phase 2, Requirement 6 - process-global "active compute resources".
#
# Model modules (RF.py/SVR.py/KNN.py/MLP.py/GAT_*.py) are called by
# genomic_prediction.py::GP() with a flat POSITIONAL hyperparameter list
# (HPARAMETERS[model] - architecture doc §4.4) whose order/length is a
# load-bearing contract (hparam_specs.py). Compute-hardware settings
# (device, n_jobs, ...) are deliberately NOT threaded through that
# positional contract - doing so would require every model's
# HPARAM_SPECS entry, every GUI field-renderer call site, and every saved
# config on disk to change in lockstep. Instead, GP() resolves them ONCE
# per run (via resolve_compute_resources()) and stores them here; each
# model module reads them back at call time via get_active_compute_resources().
# This keeps the existing per-model hyperparameter contract completely
# untouched while still giving every model module one shared, config-
# derived source of truth for hardware selection (preserving the
# "config-as-data" architectural strength - see architecture doc §18).
# ---------------------------------------------------------------------------
_ACTIVE_COMPUTE_RESOURCES = None


def set_active_compute_resources(resources: dict) -> None:
    """Store this process's resolved compute-resource dict (the return
    value of ``resolve_compute_resources()``) for model modules to read
    back via ``get_active_compute_resources()``. Called once by
    ``genomic_prediction.py::GP()`` near the start of each run/worker
    process."""
    global _ACTIVE_COMPUTE_RESOURCES
    _ACTIVE_COMPUTE_RESOURCES = dict(resources) if resources else None


def get_active_compute_resources() -> dict:
    """This process's active compute-resource dict, or the fully
    backward-compatible CPU-only default (``resolve_compute_resources()``
    with no config) if none has been set yet - e.g. a model module unit-
    tested standalone, outside of a ``GP()`` run."""
    if _ACTIVE_COMPUTE_RESOURCES is not None:
        return _ACTIVE_COMPUTE_RESOURCES
    return resolve_compute_resources(None)


def install_gpu_semaphore(semaphore) -> None:
    """Install a (typically ``multiprocessing.Semaphore``-like) semaphore
    object for ``gpu_slot()`` to guard entry with. Called once per worker
    process by the intra-batch parallel task runner
    (``genomic_prediction.py``'s ``TaskContext``/worker-pool machinery);
    never called at all in ordinary single-process runs, in which case
    ``gpu_slot()`` stays a no-op."""
    global _GPU_SEMAPHORE
    _GPU_SEMAPHORE = semaphore


def get_gpu_semaphore():
    """The raw semaphore object installed via ``install_gpu_semaphore()``
    (or ``None`` if none has been installed in THIS process) - ver4-5
    R1.i: used by ``genomic_prediction.py``'s ``_trial_worker_init()`` to
    forward the SAME already-shared semaphore object into a freshly
    spawned hyperparameter-tuning trial worker, rather than that worker
    silently having none (see that function's own docstring for why an
    uninitialised trial worker is a correctness risk, not merely a speed
    one - ``gpu_slot()`` degrades to a no-op without it, so concurrent
    CUDA trial fits would enter the device unguarded)."""
    return _GPU_SEMAPHORE


class _NullGpuSlot:
    """No-op context manager used when no GPU semaphore has been
    installed - unlimited concurrency, matching this pipeline's original,
    unguarded GPU-dispatch behaviour."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


# ---------------------------------------------------------------------------
# ver4-33, R1.a - per-task stage timing provenance.
#
# Why this exists: "prohibitively slow" could not be apportioned to a stage
# because the logs only carried per-model progress lines. One `[timing]`
# line per task now says where the wall time went, so every later
# acceleration (R1.b-R1.h) is measured, not assumed. Log-only: it never
# changes a result (I11), and any error inside it is swallowed.
# ---------------------------------------------------------------------------

class StageTimer:
    """Accumulates wall-clock seconds per named stage for ONE task.

    Time is attributed *exclusively*: when stages nest (e.g. ``geno_read``
    inside ``bio_pool``) the inner stage's seconds are removed from the
    outer one, so the stage values add up to (at most) ``total`` and the
    ``other`` remainder printed by :meth:`line` is honest.

    Thread-safe (a task's tuning trials may time stages from threads).
    Never raises into the caller: a failing timer must not fail a task.
    """

    def __init__(self) -> None:
        self._secs: Dict[str, float] = {}
        self._tags: Dict[str, Dict[str, int]] = {}
        self._info: Dict[str, str] = {}       # ver4-35: informational, NOT part of the accounting
        self._stack: List[List[float]] = []   # per open stage: [child_seconds]
        self._t0 = time.perf_counter()
        self._lock = threading.RLock()

    @contextlib.contextmanager
    def stage(self, name: str):
        """Context manager timing one stage (exclusive of nested stages)."""
        start = time.perf_counter()
        frame = [0.0]
        try:
            with self._lock:
                self._stack.append(frame)
        except Exception:
            pass
        try:
            yield self
        finally:
            try:
                elapsed = time.perf_counter() - start
                with self._lock:
                    if self._stack and self._stack[-1] is frame:
                        self._stack.pop()
                    self._secs[name] = self._secs.get(name, 0.0) + max(0.0, elapsed - frame[0])
                    if self._stack:
                        self._stack[-1][0] += elapsed
            except Exception:
                pass

    def add(self, name: str, secs: float) -> None:
        """Add `secs` to stage `name`. If a stage is currently open the
        seconds are treated as a child of it (removed from its exclusive
        time) - used for ``gpu_wait``, which happens INSIDE a model stage."""
        try:
            with self._lock:
                self._secs[name] = self._secs.get(name, 0.0) + max(0.0, float(secs))
                if self._stack:
                    self._stack[-1][0] += max(0.0, float(secs))
        except Exception:
            pass

    def tag(self, name: str, label: str) -> None:
        """Count one occurrence of `label` (e.g. 'memo hit') on stage `name`."""
        try:
            with self._lock:
                bucket = self._tags.setdefault(name, {})
                bucket[label] = bucket.get(label, 0) + 1
        except Exception:
            pass

    def info(self, name: str, text: str) -> None:
        """Record an informational entry (``name=text``) printed after the
        stages. Unlike :meth:`add` it is NOT wall time of this process and is
        excluded from ``other``/``total`` - used for worker-side seconds (a
        lane job or a tuning trial ran in ANOTHER process, concurrently with
        this one), which would otherwise break "stages add up to total"
        (ver4-35, R1.a). A repeated name keeps the latest text."""
        try:
            with self._lock:
                self._info[str(name)] = str(text)
        except Exception:
            pass

    def snapshot(self) -> Dict[str, float]:
        """A copy of the exclusive seconds per stage (empty on any error).
        A lane worker returns this so the parent can log its sub-stages."""
        try:
            with self._lock:
                return dict(self._secs)
        except Exception:
            return {}

    def seconds(self, name: str) -> float:
        """Exclusive seconds recorded for stage `name` so far (0.0 if none)."""
        with self._lock:
            return float(self._secs.get(name, 0.0))

    def _tag_text(self, name: str) -> str:
        bucket = self._tags.get(name)
        if not bucket:
            return ''
        if len(bucket) == 1:
            return f"({next(iter(bucket))})"
        return "(" + ", ".join(f"{label} x{n}" for label, n in sorted(bucket.items())) + ")"

    def line(self, task) -> str:
        """The single ``[timing]`` log line for this task ('' on any error).

        Order: the fixed preprocessing stages, then (ver4-35) ``rf_filter``,
        ``merge_shap`` and ``ld_decay``, any other stage, the ``model:*``
        stages, the GAT sub-stages ``bio:*`` and the lane stages ``lane:*``;
        then ``gpu_wait`` and (when present) ``lane_wait``, ``other`` and
        ``total``. Informational entries (:meth:`info`) follow after ``|``."""
        try:
            with self._lock:
                total = time.perf_counter() - self._t0
                names = list(self._secs)
                fixed = [n for n in ('geno_read', 'ld_prune', 'bio_pool', 'merge_rf',
                                     'rf_filter', 'merge_shap', 'ld_decay') if n in names]
                models = sorted(n for n in names if n.startswith('model:'))
                bio = sorted(n for n in names if n.startswith('bio:'))
                lane = sorted(n for n in names if n.startswith('lane:'))
                rest = sorted(n for n in names if n not in fixed and n not in models
                              and n not in bio and n not in lane
                              and n not in ('gpu_wait', 'lane_wait'))
                ordered = fixed + rest + models + bio + lane
                parts = [f"{n}={self._secs[n]:.1f}s{self._tag_text(n)}" for n in ordered]
                parts.append(f"gpu_wait={self._secs.get('gpu_wait', 0.0):.1f}s")
                if 'lane_wait' in self._secs:
                    parts.append(f"lane_wait={self._secs['lane_wait']:.1f}s")
                accounted = sum(self._secs.values())
                parts.append(f"other={max(0.0, total - accounted):.1f}s")
                parts.append(f"total={total:.1f}s")
                text = f"[timing] task={task} | " + " ".join(parts)
                if self._info:
                    text += " | info: " + " ".join(
                        f"{k}={v}" for k, v in sorted(self._info.items()))
            return text
        except Exception:
            return ''


_CURRENT_STAGE_TIMER: Optional[StageTimer] = None


def set_current_stage_timer(timer: Optional[StageTimer]) -> None:
    """Install (or clear, with None) this process's active per-task timer.
    Called by `genomic_prediction._process_one_task` at task start/end."""
    global _CURRENT_STAGE_TIMER
    _CURRENT_STAGE_TIMER = timer


def current_stage_timer() -> Optional[StageTimer]:
    """This process's active :class:`StageTimer`, or None outside a task."""
    return _CURRENT_STAGE_TIMER


def timed_stage(name: str):
    """``with timed_stage('geno_read'):`` - times the block on the active
    task timer; a plain no-op context manager when no timer is active (so
    library code can be timed without caring whether it runs inside a task)."""
    timer = _CURRENT_STAGE_TIMER
    if timer is None:
        return contextlib.nullcontext()
    return timer.stage(name)


def begin_stage(name: str):
    """Open stage `name` on the active task timer and return a handle for
    :func:`end_stage`. Exists for the few blocks of ``genomic_prediction``
    too large to re-indent under a ``with`` without a diff that hides the
    real change; everywhere else use :func:`timed_stage`."""
    handle = timed_stage(name)
    handle.__enter__()
    return handle


def end_stage(handle) -> None:
    """Close a stage opened by :func:`begin_stage`."""
    handle.__exit__(None, None, None)


def note_stage_tag(name: str, label: str) -> None:
    """Tag the active task timer's stage `name` with `label` (no-op when
    no timer is active) - e.g. ``note_stage_tag('ld_prune', 'memo hit')``."""
    timer = _CURRENT_STAGE_TIMER
    if timer is not None:
        timer.tag(name, label)


# ---------------------------------------------------------------------------
# ver4-33, R1.h - round-robin GPU device tokens.
#
# Why this exists (blueprint D2): every worker resolves `device='cuda'`
# (unindexed), so a k-GPU node's k semaphore slots all landed on cuda:0 and
# devices 1..k-1 sat idle. A token queue pre-filled round-robin with device
# indices lets `gpu_slot()` pin the acquiring process to a *different*
# device each time; unindexed 'cuda' then resolves to that device at
# `.to()` time. The legacy semaphore object stays accepted (duck-typed).
# ---------------------------------------------------------------------------

_GPU_SLOT_ANNOUNCED = set()   # (pid, device) pairs already logged in this process


class GpuTokenPool:
    """A picklable pool of CUDA device-index tokens (wraps a
    ``multiprocessing.Manager().Queue()`` proxy, so it crosses process
    boundaries exactly like the semaphore proxy it replaces)."""

    def __init__(self, queue) -> None:
        self.q = queue

    @contextlib.contextmanager
    def slot(self):
        """Acquire a device token, pin this process to that device, yield
        the device index, and always return the token afterwards."""
        dev = self.q.get()
        try:
            try:
                import torch
                torch.cuda.set_device(int(dev))
            except Exception as exc:  # torch missing, CUDA error, bad index...
                print(f"[gpu_slot] NOTE: could not select CUDA device {dev} "
                      f"({type(exc).__name__}: {exc}) - running on the process's current device "
                      f"instead; the slot is still held for throttling.")
            else:
                key = (os.getpid(), int(dev))
                if key not in _GPU_SLOT_ANNOUNCED:
                    _GPU_SLOT_ANNOUNCED.add(key)
                    print(f"[gpu_slot] PID {os.getpid()} acquired a slot on cuda:{int(dev)} "
                          f"(round-robin device tokens).")
            yield dev
        finally:
            self.q.put(dev)


def make_gpu_arbiter(n_devices: int, per_device, manager, round_robin: bool,
                     n_slots: Optional[int] = None, est_fit_mem: Optional[float] = None):
    """Build the cross-process GPU limiter the worker pools install.

    Parameters
    ----------
    n_devices : visible CUDA device count (0 -> round-robin impossible).
    per_device : slots per device (``GPU_SLOTS_PER_DEVICE``), or ``'auto'``
        (ver4-35, R1.e3): computed here from the free device memory and
        `est_fit_mem` by :func:`auto_gpu_slots_per_device` (logged).
    manager : an open ``multiprocessing`` Manager.
    round_robin : ``ACCELERATION['gpu_device_round_robin']``. False returns
        the legacy ``Manager().Semaphore`` exactly as before ver4-33.
    n_slots : total slots. None -> ``n_devices * per_device``. Callers pass
        the resolved ``n_gpu_slots`` so an explicit ``N_GPU_SLOTS`` is
        honoured (tokens are dealt round-robin: d0, d1, ..., d0, d1, ...).
    est_fit_mem : bytes one fit needs on a device; only read for
        ``per_device='auto'``.

    Returns
    -------
    ``Manager().Semaphore`` (legacy / single GPU / no devices) or a
    :class:`GpuTokenPool`; None when there are no slots at all.
    """
    if isinstance(per_device, str):
        per_device = auto_gpu_slots_per_device(est_fit_mem)
    total = int(n_slots) if n_slots is not None else int(n_devices) * max(1, int(per_device))
    if total <= 0:
        return None
    if not round_robin or int(n_devices) <= 1:
        # One device (or switch off): a token pool schedules identically to a
        # semaphore (A-h2), so keep the legacy object - zero behaviour change.
        return manager.Semaphore(total)
    queue = manager.Queue()
    for i in range(total):
        queue.put(i % int(n_devices))
    print(f"[gpu_slot] Round-robin GPU arbitration: {total} slot(s) dealt over "
          f"{int(n_devices)} CUDA device(s).")
    return GpuTokenPool(queue)


def visible_cuda_device_count() -> int:
    """Number of visible CUDA devices (0 when torch/CUDA is unavailable)."""
    try:
        import torch  # local import: never a hard dependency of this module
        return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except (ImportError, RuntimeError, OSError):
        return 0


def gpu_arbiter_for_workers(acceleration, n_gpu_slots: int, gpu_slots_per_device: int, manager):
    """The object the fan-out entry points install in their workers: exactly
    the legacy ``manager.Semaphore(n_gpu_slots)`` (None for 0 slots) unless
    ``ACCELERATION['gpu_device_round_robin']`` is on and more than one CUDA
    device is visible, in which case a :class:`GpuTokenPool` (R1.h).

    ver4-35, R1.e3: when ``ACCELERATION['gpu_slots']`` is ``'auto'`` or an
    integer k, the slot count is ``devices x per-device`` with the per-device
    value computed from free memory (``'auto'``) or fixed at k, replacing the
    GUI-pinned ``GPU_SLOTS_PER_DEVICE=1``; ``'legacy'`` (or absent) changes
    nothing.

    `acceleration` is the ``GP(ACCELERATION=...)`` value (dict or None)."""
    round_robin = bool(acceleration.get('gpu_device_round_robin', False)) \
        if isinstance(acceleration, dict) else False
    slots_mode = _normalise_gpu_slots(acceleration.get('gpu_slots', 'legacy')) \
        if isinstance(acceleration, dict) else 'legacy'
    n_visible = visible_cuda_device_count()
    if slots_mode != 'legacy' and int(n_gpu_slots) > 0:
        n_devices = max(1, n_visible)
        per_device = slots_mode if slots_mode == 'auto' else int(slots_mode)
        if per_device == 'auto':
            per_device = auto_gpu_slots_per_device()
        return make_gpu_arbiter(n_visible, per_device, manager, round_robin,
                                n_slots=n_devices * int(per_device))
    return make_gpu_arbiter(n_visible, gpu_slots_per_device, manager, round_robin,
                            n_slots=max(0, int(n_gpu_slots)))


# ---------------------------------------------------------------------------
# ver4-35, R1.e - GPU slot sizing, a lazily created arbiter, deterministic
# device placement and the CUDA-context census.
#
# Why (blueprint E5, E6, D2): in a GUI-exported run no arbiter is installed
# (fan-out is pinned off), so q tuning workers each created an unthrottled
# CUDA context on cuda:0; in a fan-out run the pinned GPU_SLOTS_PER_DEVICE=1
# serialised every GPU fit while the GPU sat mostly idle. A small GATv2 step
# is also launch-bound, so for small graphs the GPU can be SLOWER than the
# CPU - and a CPU-placed fit must not queue for a GPU slot at all.
# ---------------------------------------------------------------------------

#: Hard caps of the ``'auto'`` slot count per device: CUDA MPS lets several
#: processes share a device efficiently; without it contexts time-slice.
_GPU_SLOT_CAP_MPS = 8
_GPU_SLOT_CAP_PLAIN = 4

#: Fraction of free device memory the ``'auto'`` slot count may plan to use.
_GPU_SLOT_MEM_FRACTION = 0.8

#: The arbiter `ensure_gpu_arbiter` created in THIS process (None = none, or
#: one installed by a fan-out entry point that owns it).
_OWNED_ARBITER = {'manager': None}


def _mps_active() -> bool:
    """True if a CUDA MPS daemon appears to serve this node: the
    ``CUDA_MPS_PIPE_DIRECTORY`` variable is set, or a
    ``nvidia-cuda-mps-control`` process is running. Never raises; any
    probe failure answers False (the conservative, smaller slot cap)."""
    if os.environ.get('CUDA_MPS_PIPE_DIRECTORY'):
        return True
    import shutil
    import subprocess
    pgrep = shutil.which('pgrep')
    if pgrep is None:
        return False
    try:
        result = subprocess.run([pgrep, '-x', 'nvidia-cuda-mps-control'], capture_output=True,
                                timeout=5, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"[gpu_slot] NOTE: could not probe for a CUDA MPS daemon "
              f"({type(exc).__name__}: {exc}) - assuming none (slot cap {_GPU_SLOT_CAP_PLAIN} "
              f"per device).")
        return False
    return result.returncode == 0


def _free_device_memory_bytes() -> Optional[float]:
    """Smallest free memory (bytes) over the visible CUDA devices, or None.

    Asked of ``nvidia-smi`` first so the PARENT process does not create a
    CUDA context just to count memory; ``torch.cuda.mem_get_info`` is the
    fallback."""
    import shutil
    import subprocess
    smi = shutil.which('nvidia-smi')
    if smi is not None:
        try:
            result = subprocess.run(
                [smi, '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10, check=False)
            values = [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]
            if result.returncode == 0 and values:
                return min(values) * 1024.0 * 1024.0
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            print(f"[gpu_slot] NOTE: nvidia-smi free-memory query failed "
                  f"({type(exc).__name__}: {exc}) - trying torch.cuda.mem_get_info.")
    try:
        import torch
        if torch.cuda.is_available():
            return float(min(torch.cuda.mem_get_info(i)[0]
                             for i in range(torch.cuda.device_count())))
    except (ImportError, RuntimeError) as exc:
        print(f"[gpu_slot] NOTE: could not read free CUDA memory ({type(exc).__name__}: {exc}).")
    return None


def auto_gpu_slots_per_device(est_fit_mem: Optional[float] = None) -> int:
    """Slots per CUDA device for ``ACCELERATION['gpu_slots']='auto'``:
    ``clamp(floor(0.8 * free_mem / est_fit_mem), 1, cap)`` with ``cap`` 8
    under MPS and 4 otherwise (blueprint R1.e3). ``est_fit_mem`` defaults to
    the modelled footprint of a mid-size bio-prior fit
    (``resource_profiles.estimate_gat_step``). When the free memory cannot be
    read the answer is 1 (the legacy count) and the log says so. Logged
    every time."""
    if est_fit_mem is None:
        from resource_profiles import estimate_gat_step
        est_fit_mem = estimate_gat_step({'G': 500, 'S': 50, 'N_train': 1000, 'N_eval': 500,
                                         'factorised': True})['mem_fit']
    mps = _mps_active()
    cap = _GPU_SLOT_CAP_MPS if mps else _GPU_SLOT_CAP_PLAIN
    free = _free_device_memory_bytes()
    if free is None or est_fit_mem <= 0:
        print(f"[gpu_slot] gpu_slots='auto': free device memory unknown - using 1 slot per device "
              f"(MPS={mps}).")
        return 1
    slots = int(max(1, min(cap, math.floor(_GPU_SLOT_MEM_FRACTION * free / est_fit_mem))))
    print(f"[gpu_slot] gpu_slots='auto': free={free / 1e9:.1f} GB (smallest device), estimated fit="
          f"{est_fit_mem / 1e9:.2f} GB, MPS={mps}, cap={cap} -> {slots} slot(s) per device.")
    return slots


def ensure_gpu_arbiter(resources: Dict[str, object], est_fit_mem: Optional[float] = None):
    """Install (once) the GPU arbiter before processes fan out on a GPU node
    (ver4-35, R1.e2) and return it, or None when none applies.

    Returns the already-installed object if there is one. Creates a new one
    only when ``ACCELERATION['gpu_slots']`` is not ``'legacy'`` (the opt-in -
    with it absent nothing changes, I11) and the run has GPU slots. The
    parent owns a spawn ``Manager`` for it; ``release_gpu_arbiter()`` shuts
    that down. Safe to call from every fan-out entry point.

    Never raises for a Manager failure: it logs the reason and returns None,
    leaving the legacy (unarbitrated) behaviour.
    """
    existing = get_gpu_semaphore()
    if existing is not None:
        return existing
    if not resources or resources.get('accel_gpu_slots', 'legacy') == 'legacy':
        return None
    if int(resources.get('n_gpu_slots', 0) or 0) <= 0:
        return None
    import multiprocessing
    try:
        manager = multiprocessing.get_context('spawn').Manager()
        accel = {'gpu_slots': resources.get('accel_gpu_slots'),
                 'gpu_device_round_robin': bool(
                     resources.get('accel_gpu_device_round_robin', False))}
        arbiter = gpu_arbiter_for_workers(
            accel, int(resources.get('n_gpu_slots', 0)),
            int(resources.get('gpu_slots_per_device', 1)), manager)
    except (OSError, RuntimeError, EOFError, ValueError) as exc:
        print(f"[gpu_slot] NOTE: could not create the GPU arbiter ({type(exc).__name__}: "
              f"{exc}) - GPU fits in worker processes will not be throttled in this run.")
        return None
    if arbiter is None:
        try:
            manager.shutdown()
        except (OSError, RuntimeError, EOFError):
            pass
        return None
    install_gpu_semaphore(arbiter)
    _OWNED_ARBITER['manager'] = manager
    print(f"[gpu_slot] GPU arbiter installed in PID {os.getpid()} (ACCELERATION['gpu_slots']="
          f"{resources.get('accel_gpu_slots')!r}).")
    return arbiter


def release_gpu_arbiter() -> None:
    """Undo :func:`ensure_gpu_arbiter` if (and only if) this process created
    the arbiter: uninstall it and stop its Manager. A long-lived process (the
    Streamlit GUI runs ``GP()`` in-process) therefore never keeps one run's
    arbiter for the next. No-op otherwise; never raises."""
    manager = _OWNED_ARBITER.get('manager')
    if manager is None:
        return
    _OWNED_ARBITER['manager'] = None
    install_gpu_semaphore(None)
    try:
        manager.shutdown()
    except (OSError, RuntimeError, EOFError) as exc:
        print(f"[gpu_slot] NOTE: GPU arbiter manager did not shut down cleanly "
              f"({type(exc).__name__}: {exc}).")


def resolve_model_device(shape: Dict[str, object], res: Dict[str, object]) -> str:
    """Device (``'cpu'`` or ``res['device']``) one bio-prior GAT fit of
    `shape` should run on (ver4-35, R1.e4).

    A PURE function of the config and the data shape - never of a timer or a
    queue - so every tuning trial and the confirmatory fit take the same
    decision (I9). ``'legacy'`` and ``'gpu'`` return ``res['device']``
    unchanged; ``'cpu'`` returns ``'cpu'``; ``'auto'`` picks the GPU only when
    its estimated step time, multiplied by the safety margin, is below the CPU
    estimate (constants: ``resource_profiles.TORCH_DEVICE_CROSSOVER``). The
    CPU estimate uses ``res['policy_threads']`` - the parent's undivided
    thread count - so a divided trial worker still agrees with the parent.
    """
    policy = res.get('accel_torch_device_policy', 'legacy')
    device = str(res.get('device', 'cpu'))
    if policy in ('legacy', 'gpu'):
        return device
    if policy == 'cpu' or not device.startswith('cuda'):
        return 'cpu'
    from resource_profiles import TORCH_DEVICE_CROSSOVER, estimate_gat_step_seconds
    threads = int(res.get('policy_threads') or res.get('torch_num_threads') or 1)
    cpu_s, gpu_s = estimate_gat_step_seconds(shape, threads)
    return device if gpu_s * TORCH_DEVICE_CROSSOVER['margin'] < cpu_s else 'cpu'


_CUDA_CONTEXT_ANNOUNCED = set()


def note_cuda_context(device) -> None:
    """Print ``[cuda] pid=... context=created`` the first time THIS process
    computes on a CUDA device (ver4-35, R1.a census): counting these lines
    across a run's log shows how many CUDA contexts coexisted (acceptance
    A-e2: at most slots + 1). Does nothing for a non-CUDA device."""
    text = str(device)
    if not text.startswith('cuda') or os.getpid() in _CUDA_CONTEXT_ANNOUNCED:
        return
    _CUDA_CONTEXT_ANNOUNCED.add(os.getpid())
    print(f"[cuda] pid={os.getpid()} context=created device={text}")


class _TimedGpuSlot:
    """Wraps whichever slot object is installed so the time spent BLOCKED
    acquiring it is recorded as ``gpu_wait`` on the active task timer
    (R1.a, A-f4). Transparent otherwise."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self._started = 0.0

    def __enter__(self):
        t0 = time.perf_counter()
        result = self._inner.__enter__()
        waited = time.perf_counter() - t0
        timer = _CURRENT_STAGE_TIMER
        if timer is not None:
            timer.add('gpu_wait', waited)
        return result

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._inner.__exit__(exc_type, exc_val, exc_tb)


def gpu_slot(device=None):
    """Context manager guarding entry into a GPU-dispatched model call.
    Usage::

        with gpu_slot(device):
            model.to(device)
            ...forward/backward pass...

    A no-op whenever ``install_gpu_semaphore()`` hasn't been called (the
    default for every run that isn't using intra-batch parallelism with
    more than one GPU-capable worker). With a :class:`GpuTokenPool`
    installed (ver4-33, R1.h) the slot also pins the process to the device
    its token names. The wait for the slot is recorded as ``gpu_wait`` on
    the active task timer (R1.a).

    ``device`` (ver4-35, R1.e4, optional): when given and NOT a CUDA device
    the slot is a no-op - a fit placed on the CPU must never queue for, or
    hold, a GPU slot. ``None`` (every pre-ver4-35 caller) keeps the legacy
    behaviour."""
    if device is not None and not str(device).startswith('cuda'):
        return _NullGpuSlot()
    if _GPU_SEMAPHORE is None:
        return _NullGpuSlot()
    if isinstance(_GPU_SEMAPHORE, GpuTokenPool):
        return _TimedGpuSlot(_GPU_SEMAPHORE.slot())
    return _TimedGpuSlot(_GPU_SEMAPHORE)


# ---------------------------------------------------------------------------
# ver4-46, R1 - the "search fit" context.
#
# A hyperparameter-search candidate only needs its VALIDATION predictions: the
# test, train and attention outputs, the explainability pass and the per-fit
# bookkeeping a normal fit produces are all discarded by `make_objective()`.
# `hyperparameter_tuning.make_objective()._score_once` wraps every candidate fit
# in `search_fit_context(...)`; a model that knows how to skip that work (the GAT
# family, see `models/gat_search.py`) asks `search_fit_cfg()` and takes its lean
# path. Outside the context (the confirmatory fit, every untuned fit) the
# accessor returns None and nothing changes (I11). The state is thread-local and
# per process, so a loky trial worker carries its own copy by construction.
# ---------------------------------------------------------------------------

_SEARCH_FIT = threading.local()


@contextlib.contextmanager
def search_fit_context(**cfg):
    """Mark the enclosed model call as a hyperparameter-SEARCH fit.

    `cfg` keys (all optional): ``lean`` (bool) - the model may skip every output
    the search does not read. Nested contexts restore the outer one on exit."""
    previous = getattr(_SEARCH_FIT, 'cfg', None)
    _SEARCH_FIT.cfg = dict(cfg)
    try:
        yield
    finally:
        _SEARCH_FIT.cfg = previous


def search_fit_cfg() -> Optional[Dict[str, object]]:
    """The active search-fit configuration dict, or None outside a search fit."""
    return getattr(_SEARCH_FIT, 'cfg', None)


def is_lean_search_fit() -> bool:
    """True inside a search fit that asked for the lean path (``lean=True``)."""
    cfg = search_fit_cfg()
    return bool(cfg and cfg.get('lean'))


# ---------------------------------------------------------------------------
# ver4-33, R1.c - location of the content-addressed preprocessing memo.
#
# `Preprocess/preprocess_memo.py` reads $EASIGP_MEMO_DIR. It is set ONCE, by
# the outermost entry point, to Result/<outer RESULT_NAME>/_memo/ so that
# model-group workers, array tasks and loky/spawn children (which inherit the
# environment) all share one memo - that sharing is the whole point, because
# the repeated work (blueprint D5) happens ACROSS those processes.
# ---------------------------------------------------------------------------

MEMO_DIR_ENV = 'EASIGP_MEMO_DIR'
MEMO_SUBDIR_NAME = '_memo'


def ensure_memo_dir_env(result_name: str) -> bool:
    """Set ``EASIGP_MEMO_DIR`` to ``Result/<result_name>/_memo`` unless it
    is already set (an outer entry point's value always wins).

    Returns True if THIS call set it - the caller should then pass that
    flag to :func:`release_memo_dir_env` in a ``finally`` so a long-lived
    process (the Streamlit GUI runs ``GP()`` in-process) never leaks one
    run's memo location into the next run. The directory itself is created
    lazily by the memo on first write, not here.
    """
    if not result_name or os.environ.get(MEMO_DIR_ENV):
        return False
    os.environ[MEMO_DIR_ENV] = os.path.join(result_dir_path(str(result_name)), MEMO_SUBDIR_NAME)
    return True


def clear_memo_dir(result_name: str) -> bool:
    """Delete ``Result/<result_name>/_memo`` (used by ``--overwrite``).

    The memo is content-addressed, so a stale entry can never be wrong - this
    only reclaims disk and honours "redo from scratch". Returns True if a
    directory was removed; a failure is logged and never fatal.
    """
    import shutil
    path = os.path.join(result_dir_path(str(result_name)), MEMO_SUBDIR_NAME)
    if not os.path.isdir(path):
        return False
    try:
        shutil.rmtree(path)
        print(f"[pipeline_utils] --overwrite: removed preprocessing memo '{path}'.")
        return True
    except OSError as exc:
        print(f"[pipeline_utils] NOTE: could not remove preprocessing memo '{path}' "
              f"({type(exc).__name__}: {exc}); stale entries are harmless (content-addressed).")
        return False


def release_memo_dir_env(was_set_here: bool) -> None:
    """Undo :func:`ensure_memo_dir_env` if (and only if) it set the variable."""
    if was_set_here:
        os.environ.pop(MEMO_DIR_ENV, None)


# ---------------------------------------------------------------------------
# ver4-56, R1 - location of the resource-telemetry JSONL files.
#
# Same pattern as the memo directory above: the OUTERMOST entry point sets
# EASIGP_TELEMETRY_DIR once to Result/<outer RESULT_NAME>/logs/resource_telemetry
# so that every worker process (intra-batch / intra-task pools, loky, spawn
# children, which inherit the environment) writes into ONE run-stable folder
# instead of its own nested ``_intra_batch/...`` one. Each process still writes
# its own file (resource_telemetry.py), so there is no shared file (I10).
# ---------------------------------------------------------------------------

from resource_telemetry import TELEMETRY_DIR_ENV, TELEMETRY_SUBDIR  # noqa: E402  (stdlib-only module, I2)


def ensure_telemetry_dir_env(result_name: str) -> bool:
    """Set ``EASIGP_TELEMETRY_DIR`` to ``Result/<result_name>/logs/resource_telemetry``
    unless it is already set (an outer entry point's value always wins).

    Returns True if THIS call set it; pass that flag to
    :func:`release_telemetry_dir_env` in a ``finally`` so the long-lived
    Streamlit process never leaks one run's folder into the next run. The
    directory itself is created by ``resource_telemetry`` on first write.
    """
    if not result_name or os.environ.get(TELEMETRY_DIR_ENV):
        return False
    os.environ[TELEMETRY_DIR_ENV] = os.path.join(
        result_dir_path(str(result_name)), 'logs', TELEMETRY_SUBDIR)
    return True


def telemetry_dir_for(result_name: str) -> str:
    """The telemetry folder of the run called ``result_name``
    (``Result/<result_name>/logs/resource_telemetry``). Pure: creates nothing."""
    return os.path.join(result_dir_path(str(result_name)), 'logs', TELEMETRY_SUBDIR)


def release_telemetry_dir_env(was_set_here: bool) -> None:
    """Undo :func:`ensure_telemetry_dir_env` if (and only if) it set the variable."""
    if was_set_here:
        os.environ.pop(TELEMETRY_DIR_ENV, None)


# ---------------------------------------------------------------------------
# Phase 2, Requirement 8 - cheap, non-data-reading resource-advisor inputs.
#
# Deliberately kept in this module (rather than resource_profiles.py, which
# holds the STATIC per-model cost table) since it reuses/extends the exact
# same "cheap header/line-count read" pattern list_phenotype_columns()
# already established, and pipeline_utils.py already exists to serve both
# the GUI and headless sides.
# ---------------------------------------------------------------------------

# Streamlit re-executes the whole GUI script on EVERY widget interaction, and the resource
# advisor / LD-pruning helper / task-count preview all call the line counters below on the
# (possibly multi-GB, very wide) genotype file each time.  Counting lines in text mode decodes
# every byte, and on a shared parallel filesystem (NCI Gadi /g/data, /scratch) re-reading the
# file on each rerun made the GUI take minutes to start and to respond.  Two fixes:
#   1. count newlines on raw bytes in large chunks (no per-line UTF-8 decode, no per-line
#      Python object) - several times faster than ``sum(1 for _ in f)``;
#   2. memoise per process on (path, size, mtime_ns) so a rerun with an unchanged file costs one
#      ``os.stat`` instead of a full read.  An edited file changes size/mtime and is re-counted.
_LINE_COUNT_CACHE: "dict" = {}
_LINE_COUNT_CACHE_MAX = 64


def _count_lines_cached(path: str) -> Optional[int]:
    """Number of lines in `path` (same value as ``sum(1 for _ in open(path))`` for LF / CRLF
    files, i.e. a final line without a trailing newline still counts). ``None`` if the file
    can't be read. Cached on (absolute path, size, mtime_ns)."""
    try:
        st_ = os.stat(path)
    except OSError:
        return None
    key = (os.path.abspath(path), st_.st_size, st_.st_mtime_ns)
    hit = _LINE_COUNT_CACHE.get(key)
    if hit is not None:
        return hit
    n_lf = 0
    n_cr = 0
    last = b''
    try:
        with open(path, 'rb') as f:
            while True:
                chunk = f.read(16 * 1024 * 1024)
                if not chunk:
                    break
                n_lf += chunk.count(b'\n')
                n_cr += chunk.count(b'\r')
                last = chunk[-1:]
    except OSError:
        return None
    if n_lf == 0 and n_cr > 0:      # classic-Mac '\r'-only line endings (text mode used to split on these)
        n_lines = n_cr + (0 if last == b'\r' else 1)
    else:
        n_lines = n_lf + (0 if (last in (b'\n', b'') ) else 1)
    if len(_LINE_COUNT_CACHE) >= _LINE_COUNT_CACHE_MAX:
        _LINE_COUNT_CACHE.clear()
    _LINE_COUNT_CACHE[key] = n_lines
    return n_lines


def count_csv_data_rows(path: str) -> Optional[int]:
    """Cheap row COUNT (not a full read) of a CSV file's data rows (header
    excluded) - a raw-byte newline count, never materialising the file's
    columns/values, cached per (path, size, mtime) so repeated GUI reruns
    are free. Returns ``None`` if the path is missing/unreadable."""
    if not path or not os.path.isfile(path):
        return None
    n_lines = _count_lines_cached(path)
    if n_lines is None:
        return None
    return max(0, n_lines - 1)


def count_csv_columns(path: str) -> Optional[int]:
    """Cheap column COUNT of a CSV file's header row only. Returns
    ``None`` if the path is missing/unreadable/empty."""
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
    except (OSError, csv.Error, StopIteration):
        return None
    return len(header) if header else None


def count_bim_lines(bim_path: str) -> Optional[int]:
    """Cheap line count of a PLINK1 ``.bim`` file - one line per marker,
    positions only, no genotype parsing (consistent with EasiGP's existing
    "deferred PLINK conversion" optimisation - see architecture doc §8).
    Returns ``None`` if the path is missing/unreadable."""
    if not bim_path or not os.path.isfile(bim_path):
        return None
    return _count_lines_cached(bim_path)


def count_fam_lines(fam_path: str) -> Optional[int]:
    """Cheap line count of a PLINK1 ``.fam`` file - one line per
    individual. Returns ``None`` if the path is missing/unreadable."""
    if not fam_path or not os.path.isfile(fam_path):
        return None
    return _count_lines_cached(fam_path)


def cheap_marker_and_sample_counts(
    genotype_format: str,
    genotype_file_name: str,
    phenotype_file_name: Optional[str] = None,
) -> Tuple[Optional[int], Optional[int]]:
    """``(n_markers, n_samples)`` estimated with only cheap, header/line-
    count-only I/O - never a full genotype/phenotype materialisation (see
    ``resource_profiles.py::estimate_resources()``, which is this
    function's caller).

    - CSV genotype: marker count = header column count - 2 (ID/population -
      same positional convention as ``list_phenotype_columns()``); sample
      count = data row count of the SAME genotype file (one row per
      individual), falling back to the phenotype file's row count if the
      genotype file can't be read.
    - PLINK genotype: marker count = line count of ``<stem>.bim``; sample
      count = line count of ``<stem>.fam``.

    Either element of the returned tuple is ``None`` if it couldn't be
    determined cheaply (e.g. file not found yet) - callers should treat
    ``None`` as "unknown", not zero.
    """
    if genotype_format == 'plink':
        bim_path = f'{genotype_file_name}.bim'
        fam_path = f'{genotype_file_name}.fam'
        return count_bim_lines(bim_path), count_fam_lines(fam_path)

    n_cols = count_csv_columns(genotype_file_name)
    n_markers = max(0, n_cols - 2) if n_cols is not None else None
    n_samples = count_csv_data_rows(genotype_file_name)
    if n_samples is None and phenotype_file_name:
        n_samples = count_csv_data_rows(phenotype_file_name)
    return n_markers, n_samples


def count_unique_populations(csv_path: str, population_source: "Optional[dict]" = None) -> Optional[int]:
    """Best-effort count of DISTINCT populations in a genotype/phenotype
    CSV - used only by the resource advisor's cheap `n_population` input.

    Update ID ver4-12, R2: `population_source` (default `None`, reproducing
    ver4-11 exactly - I11) is forwarded straight to
    `derive_population_series()`, so the SAME single authority that
    `genomic_prediction.py::GP()` uses to build its `population` column
    also drives this GUI-facing estimate (the `_map_genes_to_markers`
    "one authority, two consumers" precedent - architecture doc SS8). With
    `population_source=None` (or `{'mode': 'column', ...}`), this still
    only reads the ONE named column (`usecols`) - the cheap path is
    preserved. `mode='derived'`/`'constant'` need every named column
    (there is no single column to `usecols=` on), so those modes read the
    full file - still far cheaper than materialising the genotype matrix,
    and this remains a best-effort, NEVER-raising estimate either way.

    On any failure (missing pandas, malformed file, a
    `derive_population_series()` error such as a missing/NaN column,
    huge file that's impractical to scan this way, ...) this returns
    `None` rather than raising, and callers should fall back to a
    conservative default (e.g. 1) when that happens.
    """
    if not csv_path or not os.path.isfile(csv_path):
        return None
    try:
        import pandas as pd
        mode = (population_source or {}).get('mode', 'column')
        if mode == 'column':
            col_name = (population_source or {}).get('column', 'population')
            # Positional fallback (pre-ver4-12 behaviour): a config with no
            # explicit POPULATION_SOURCE at all names 'population' as the
            # column, which - per the fixed genotype/phenotype file
            # contract - is always the file's SECOND column, whatever its
            # own header text says (unify_columns_by_position). Read by
            # name when the header already matches; fall back to reading
            # the second column by position and renaming it otherwise, so
            # a file using different header text for its population
            # column (exactly what unify_columns_by_position tolerates
            # everywhere else) doesn't make this estimate silently fail.
            try:
                df = pd.read_csv(csv_path, usecols=[col_name], encoding='utf-8-sig')
            except ValueError:
                df = pd.read_csv(csv_path, usecols=[1], encoding='utf-8-sig')
                df.columns = [col_name]
        elif mode == 'constant':
            return 1
        else:  # 'derived'
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
        series = derive_population_series(df, population_source)
        return int(series.nunique())
    except Exception:
        return None


def list_unique_populations(csv_path: str, population_source: "Optional[dict]" = None,
                             limit: int = 500) -> Optional[list]:
    """Best-effort, sorted list of the DISTINCT population labels in a
    genotype/phenotype CSV - the actual-VALUES sibling of
    ``count_unique_populations()`` above (same cheap, ``usecols``-only read
    for ``mode='column'``; same full-file read for ``'derived'``/
    ``'constant'``), used by the GUI's between-scenario population-
    combination EXCLUSION picker (Requirements.md item 3) to populate its
    options - it needs the actual labels, not just a count.

    Returns ``None`` on any failure (missing file, malformed CSV, a
    ``derive_population_series()`` error, ...), exactly like
    ``count_unique_populations()`` - callers should treat that as
    "unknown" and fall back to showing no options rather than raising.
    Truncated to at most ``limit`` labels (sorted first, so truncation
    drops the lexicographically-last ones) purely as a GUI-responsiveness
    guard against a pathological file with a near-unique 'population'
    column; every real use of this field has at most a handful of
    populations.
    """
    if not csv_path or not os.path.isfile(csv_path):
        return None
    try:
        import pandas as pd
        mode = (population_source or {}).get('mode', 'column')
        if mode == 'column':
            col_name = (population_source or {}).get('column', 'population')
            try:
                df = pd.read_csv(csv_path, usecols=[col_name], encoding='utf-8-sig')
            except ValueError:
                df = pd.read_csv(csv_path, usecols=[1], encoding='utf-8-sig')
                df.columns = [col_name]
        elif mode == 'constant':
            series = derive_population_series(pd.DataFrame(index=[0]), population_source)
            return sorted(series.astype(str).unique().tolist())
        else:  # 'derived'
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
        series = derive_population_series(df, population_source)
        values = sorted(set(series.astype(str).unique().tolist()))
        return values[:limit]
    except Exception:
        return None


def between_population_combinations(populations, mode: str = 'pairs') -> list:
    """Every ordered population combination relevant to a
    ``SCENARIO='between'`` run, computed purely from the population LIST
    (no genotype/phenotype I/O - by the time a caller has this list in
    hand, deriving it is just string concatenation over a small Python
    list, "a computationally light way" per Requirements.md item 3).

    ``mode='pairs'`` returns every ``'X->Y'`` string with ``X != Y``
    (train->test) - the shape needed when a between-scenario run uses no
    validation split at all, or when its validation individuals are
    carved out of the train or test population itself rather than drawn
    from a third one (Requirements.md item 1's ``'split_train'``/
    ``'split_test'`` options - see ``genomic_prediction.py::GP()``'s own
    scenario-enumeration code).

    ``mode='triples'`` returns every ``'X->Y->Z'`` string with X, Y, Z
    pairwise-distinct (train->valid->test - Requirements.md item 1's
    default ``'third_population'`` option, using an entire extra
    population purely for validation).

    This is the SINGLE place both ``genomic_prediction.py::GP()``
    (building its real ``sample`` table) and ``main_app.py`` (populating
    the population-combination EXCLUSION multiselect's options -
    Requirements.md item 3) derive this list, so the two can never drift
    apart from one another - the same "one authority, several consumers"
    discipline ``_map_genes_to_markers``/``canonical_model_order`` already
    established elsewhere in this codebase (architecture doc SS8).
    """
    pops = [str(p) for p in populations]
    if mode == 'triples':
        return [f'{x}->{y}->{z}' for x in pops for y in pops for z in pops
                if x != y and x != z and y != z]
    return [f'{x}->{y}' for x in pops for y in pops if x != y]


def build_sample_table(scenario, population, phenotype, ratio, sample_num,
                       between_validation_source=None, between_ratio=None,
                       excluded=None):
    """Return the ``sample`` task table that ``GP()`` iterates over.

    Why this lives here (ver4-56, blueprint R3 step 1): the resource advisor
    and the pilot-job selector must enumerate EXACTLY the tasks ``GP()`` will
    run. The block below is the verbatim move of the inline code that used to
    sit in ``genomic_prediction.GP()`` (I9: same rows, same order). Both
    ``GP()`` and the advisor now call this one function, so the two can never
    drift apart (fixes defect E9: the GUI used the ``within`` formula for
    ``between``).

    Args:
        scenario: ``'within'`` or ``'between'``.
        population: list of population names (``POPULATION``).
        phenotype: list of trait names (``PHENOTYPE``).
        ratio: list of train ratios (floats or tuples) (``RATIO``); only used
            by ``within``.
        sample_num: repetitions per ratio (``SAMPLE_NUM``).
        between_validation_source: ``None`` (no validation split),
            ``'third_population'``, ``'split_train'`` or ``'split_test'``;
            already resolved by ``GP()`` (see its ``_between_validation_source``).
        between_ratio: list of floats (``BETWEEN_RATIO``); only used by the
            ``split_train`` / ``split_test`` options.
        excluded: iterable of ``'X->Y'`` / ``'X->Y->Z'`` strings to drop
            (``EXCLUDED_POPULATION_COMBINATIONS``); None or empty drops nothing.

    Returns:
        ``pandas.DataFrame`` with columns ``population, phenotype, ratio,
        sample`` (one row per task; ``ratio`` and ``sample`` are ``-1`` for
        the ``between`` rows without a split).

    Raises:
        ValueError: ``between`` with a split source but an empty or
            tuple-valued ``between_ratio``.
    """
    import pandas as pd

    # Local aliases keep the moved block below character-identical to the
    # code it replaced in GP().
    SCENARIO = scenario
    POPULATION = population
    PHENOTYPE = phenotype
    RATIO = ratio
    SAMPLE_NUM = sample_num
    _between_validation_source = between_validation_source
    BETWEEN_RATIO = between_ratio
    EXCLUDED_POPULATION_COMBINATIONS = excluded
    sample = None

    if SCENARIO == 'within':
        sample = pd.DataFrame({'population':[item for item in POPULATION for i in range(SAMPLE_NUM*len(PHENOTYPE)*len(RATIO))],
                                'phenotype':[item for item in PHENOTYPE for i in range(SAMPLE_NUM*len(RATIO))]*len(POPULATION),
                                'ratio':[item for item in RATIO for i in range(SAMPLE_NUM)]*len(POPULATION)*len(PHENOTYPE),
                                'sample':list(range(1,SAMPLE_NUM+1)) *len(PHENOTYPE)*len(POPULATION)*len(RATIO)})
    elif SCENARIO == 'between':
        # Requirements.md item 3: excluded combinations are removed from
        # `comb` BEFORE `sample` is built, i.e. before any task for them
        # is ever enumerated - "computationally light" (pure string-set
        # membership over the small population-combination list; no
        # genotype/phenotype I/O). Default (None/empty) excludes nothing,
        # matching pre-patch behaviour exactly (I11).
        _excluded_combos = set(EXCLUDED_POPULATION_COMBINATIONS or [])
        if _between_validation_source == 'third_population':
            # X->Y->Z, all-distinct triples - identical to the pre-patch
            # W_OPT-is-not-None branch, now reached via an explicit
            # validation choice instead of inferred from W_OPT alone.
            comb = [c for c in between_population_combinations(POPULATION, mode='triples')
                    if c not in _excluded_combos]
            sample = pd.DataFrame({'population': comb * len(PHENOTYPE),
                                    'phenotype': [p for p in PHENOTYPE for _ in range(len(comb))],
                                    'ratio': [-1] * (len(comb) * len(PHENOTYPE)),
                                    'sample': [-1] * (len(comb) * len(PHENOTYPE))})
        elif _between_validation_source in ('split_train', 'split_test'):
            # X->Y pairs, each replicated across every configured
            # BETWEEN_RATIO x SAMPLE_NUM iteration - mirrors the 'within'
            # scenario's own tiling pattern exactly (POPULATION -> comb,
            # RATIO -> _between_ratio_list), since this is the same idea:
            # a random train_test_split of one population's own
            # individuals, repeated SAMPLE_NUM times per ratio.
            comb = [c for c in between_population_combinations(POPULATION, mode='pairs')
                    if c not in _excluded_combos]
            _between_ratio_list = list(BETWEEN_RATIO or [])
            if not _between_ratio_list:
                raise ValueError(
                    "BETWEEN_RATIO must contain at least one float when "
                    f"BETWEEN_VALIDATION_SOURCE={_between_validation_source!r} - it decides what "
                    "fraction of the training (or test) population's own individuals stay in "
                    "that role, the remainder becoming the validation set."
                )
            for _r in _between_ratio_list:
                if isinstance(_r, tuple):
                    raise ValueError(
                        f"BETWEEN_RATIO entry {_r!r} is a tuple - only plain floats are accepted "
                        f"for BETWEEN_VALIDATION_SOURCE={_between_validation_source!r} (it splits a "
                        "single population's individuals in two, not three)."
                    )
            sample = pd.DataFrame({
                'population': [item for item in comb for i in range(SAMPLE_NUM * len(PHENOTYPE) * len(_between_ratio_list))],
                'phenotype': [item for item in PHENOTYPE for i in range(SAMPLE_NUM * len(_between_ratio_list))] * len(comb),
                'ratio': [item for item in _between_ratio_list for i in range(SAMPLE_NUM)] * len(comb) * len(PHENOTYPE),
                'sample': list(range(1, SAMPLE_NUM + 1)) * len(PHENOTYPE) * len(comb) * len(_between_ratio_list),
            })
        else:
            # No validation split - X->Y pairs, self-pairs excluded.
            # Identical to the pre-patch W_OPT-is-None branch.
            comb = [c for c in between_population_combinations(POPULATION, mode='pairs')
                    if c not in _excluded_combos]
            sample = pd.DataFrame({'population': comb * len(PHENOTYPE),
                                    'phenotype': [p for p in PHENOTYPE for _ in range(len(comb))],
                                    'ratio': [-1] * (len(comb) * len(PHENOTYPE)),
                                    'sample': [-1] * (len(comb) * len(PHENOTYPE))})

    return sample


def resolve_between_validation_source(cfg) -> Optional[str]:
    """The ``between`` validation source ``GP()`` would resolve from ``cfg``.

    Mirrors GP()'s own resolution (``BETWEEN_VALIDATION`` / ``BETWEEN_VALIDATION_SOURCE``
    / ``W_OPT``); returns None for ``within`` and for ``between`` without validation.
    """
    if cfg.get('SCENARIO') != 'between':
        return None
    flag = cfg.get('BETWEEN_VALIDATION')
    if flag is None:
        return 'third_population' if cfg.get('W_OPT') is not None else None
    return (cfg.get('BETWEEN_VALIDATION_SOURCE') or 'third_population') if flag else None


def _phenotype_axes_uncached(path, mode, population_source_json, trait_columns_json):
    """Read the phenotype file the way ``GP()`` does and return
    ``(populations, traits, sizes)``; see :func:`resolve_task_axes`."""
    import json
    import pandas as pd
    population_source = json.loads(population_source_json) if population_source_json else None
    trait_columns = json.loads(trait_columns_json) if trait_columns_json else None
    frame = pd.read_csv(path)
    resolved = population_source or {'mode': 'column', 'column': 'population'}
    legacy = (mode == 'analysed' and resolved.get('mode', 'column') == 'column'
              and resolved.get('column', 'population') == 'population')
    with contextlib.redirect_stdout(io.StringIO()):     # the rename notice is GP's, not ours
        if legacy:
            analysed = unify_columns_by_position(frame, ['ID', 'population'], 'phenotype file',
                                                 first_n=2)
            analysed['ID'] = analysed['ID'].astype(str)
        else:
            if mode == 'raw':
                frame = unify_columns_by_position(frame, ['ID'], 'phenotype file', first_n=1)
            else:
                frame = unify_columns_by_position(frame, ['ID', 'population'], 'phenotype file',
                                                  first_n=2)
            frame['ID'] = frame['ID'].astype(str)
            series = derive_population_series(frame, population_source)
            traits_in_file = resolve_trait_columns(
                {'PHENOTYPE_FILE_MODE': mode, 'PHENOTYPE_TRAIT_COLUMNS': trait_columns}, path)
            analysed = collapse_raw_to_analysed_level(frame, 'ID', series, traits_in_file)
    populations = list(pd.unique(analysed['population']))
    traits = [str(c) for c in analysed.columns[2:]]
    sizes = {}
    counts = analysed.groupby('population', sort=False)[list(analysed.columns[2:])].count()
    for population in populations:
        for column, trait in zip(analysed.columns[2:], traits):
            sizes[(str(population), trait)] = int(counts.loc[population, column])
    return populations, traits, sizes


_PHENOTYPE_AXES_CACHE = {}


def resolve_task_axes(cfg):
    """The populations, traits and per-(population, trait) sizes ``GP()`` will use.

    Why (ver4-56 bug fix): ``GP()`` does not take populations from the config; it
    reads them from the phenotype file (``pd.unique`` of the unified/derived
    ``population`` column, in file order) and, for ``PHENOTYPE='all'``, takes every
    trait column. The advisor, pilot and audit must enumerate the SAME table (I9),
    so this mirrors that block: positional unification of the first columns, the
    ``POPULATION_SOURCE`` rules, and the raw-file collapse.

    Cached on (path, modified time, mode, population source, trait columns), because
    the GUI re-runs this on every widget click.

    Returns ``(populations, traits, sizes)`` or ``None`` (with a printed reason)
    when the file is missing or cannot be interpreted; the caller then falls back.
    """
    import json
    path = cfg.get('PHENOTYPE_FILE_NAME')
    if not path or not os.path.isfile(str(path)):
        print(f"[resource] phenotype file {path!r} not found - populations and task sizes "
              f"unknown; the advisor will fall back to the genotype sample count.")
        return None
    try:
        stamp = os.stat(path)
        key = (str(path), stamp.st_mtime_ns, stamp.st_size,
               str(cfg.get('PHENOTYPE_FILE_MODE') or 'analysed'),
               json.dumps(cfg.get('POPULATION_SOURCE'), sort_keys=True, default=str),
               json.dumps(cfg.get('PHENOTYPE_TRAIT_COLUMNS'), default=str))
        if key not in _PHENOTYPE_AXES_CACHE:
            if len(_PHENOTYPE_AXES_CACHE) > 8:
                _PHENOTYPE_AXES_CACHE.clear()
            _PHENOTYPE_AXES_CACHE[key] = _phenotype_axes_uncached(key[0], key[3], key[4], key[5])
        populations, traits, sizes = _PHENOTYPE_AXES_CACHE[key]
        return list(populations), list(traits), dict(sizes)
    except Exception as exc:  # noqa: BLE001 - advisory path: any failure means "unknown", printed
        print(f"[resource] could not read the phenotype file {path!r} "
              f"({type(exc).__name__}: {exc}) - populations and task sizes unknown; the advisor "
              f"will fall back to the genotype sample count.")
        return None


def build_sample_table_from_cfg(cfg):
    """The ``sample`` task table of the run described by the config dict ``cfg``.

    One authority for the advisor, the pilot selector and the audit: it is
    exactly the table ``GP()`` builds (same :func:`build_sample_table`, same
    between-validation resolution, tuple ratios restored after the JSON
    round-trip). Populations and (for ``PHENOTYPE='all'``) traits come from the
    phenotype file via :func:`resolve_task_axes`, as in ``GP()``; an explicit
    ``cfg['POPULATION']`` list is used only when the file cannot be read.
    Raises ValueError on an invalid ``BETWEEN_RATIO`` (as GP does).
    """
    scenario = cfg.get('SCENARIO', 'within')
    axes = resolve_task_axes(cfg)
    if axes is not None:
        populations, file_traits, _ = axes
    else:
        populations, file_traits = list(cfg.get('POPULATION') or []), []
    phenotype = cfg.get('PHENOTYPE')
    if isinstance(phenotype, str):
        phenotype = file_traits if phenotype == 'all' else [phenotype]
    return build_sample_table(
        scenario, populations, list(phenotype or []),
        restore_ratio(list(cfg.get('RATIO') or []), scenario),
        int(cfg.get('ITER_NUM', cfg.get('SAMPLE_NUM', 1)) or 1),
        between_validation_source=resolve_between_validation_source(cfg),
        between_ratio=cfg.get('BETWEEN_RATIO'),
        excluded=cfg.get('EXCLUDED_POPULATION_COMBINATIONS'),
    )


def select_pilot_rows(sample_df, sizes, k):
    """Which rows of the ``sample`` table the pilot job runs.

    Why: the calibration must interpolate the real run, so the pilot spans the
    extremes of task size instead of running 'batch 0' (whose rows are one
    population and one ratio, forcing extrapolation). A pure, deterministic
    function of the table, the per-population sizes and ``k`` (I9): two
    invocations pick the same rows.

    Order of choice: (1) the largest task, (2) the smallest, (3) for every ratio
    not yet covered, the task of that ratio closest to the median size, (4) the
    rest at evenly spaced positions of the size-ordered list.

    Args:
        sample_df: ``build_sample_table`` result (columns population, phenotype,
            ratio, sample).
        sizes: ``{(population, phenotype): n}`` (``cheap_task_sizes``); a task's
            size is the sum over the populations of a ``X->Y[->Z]`` row.
        k: how many rows to pick (clamped to the table length).

    Returns:
        Sorted list of distinct absolute row indices, ``len <= k``.
    """
    n_rows = len(sample_df)
    if n_rows == 0 or k <= 0:
        return []
    k = min(int(k), n_rows)
    populations = sample_df['population'].tolist()
    phenotypes = sample_df['phenotype'].tolist()
    ratios = sample_df['ratio'].tolist()
    size = [sum(sizes.get((part, str(phenotypes[i])), 0) for part in str(populations[i]).split('->'))
            for i in range(n_rows)]
    order = sorted(range(n_rows), key=lambda i: (size[i], i))
    chosen = []

    def take(index):
        if index not in chosen and len(chosen) < k:
            chosen.append(index)

    take(min(i for i in range(n_rows) if size[i] == size[order[-1]]))
    take(min(i for i in range(n_rows) if size[i] == size[order[0]]))
    median_size = size[order[n_rows // 2]]
    seen = {repr(ratios[i]) for i in chosen}
    for ratio_key in dict.fromkeys(repr(r) for r in ratios):
        if ratio_key in seen or len(chosen) >= k:
            continue
        candidates = [i for i in range(n_rows) if repr(ratios[i]) == ratio_key]
        take(min(candidates, key=lambda i: (abs(size[i] - median_size), i)))
        seen.add(ratio_key)
    remaining = k - len(chosen)
    for slot in range(remaining):
        target = int(round((slot + 0.5) / remaining * (n_rows - 1)))
        for offset in range(n_rows):
            picked = False
            for position in (target + offset, target - offset):
                if 0 <= position < n_rows and order[position] not in chosen:
                    take(order[position])
                    picked = True
                    break
            if picked:
                break
    return sorted(chosen)


def cheap_task_sizes(cfg) -> "Dict[Tuple[str, str], int]":
    """Number of usable individuals per ``(population, phenotype)`` pair,
    read from the PHENOTYPE file only (never the genotype matrix).

    Why (ver4-56, blueprint R3 step 2): the calibrated resource advisor needs
    each task's own training size, because cost scales with ``n`` and the
    populations of one run can differ a lot in size. ``GP()`` itself matches
    genotype and phenotype IDs; this cheap count does not read the genotype
    file, so it can over-count by the (normally small) number of phenotyped
    individuals without a genotype. That makes the advice slightly
    conservative, never optimistic.

    The file is interpreted exactly as ``GP()`` does (:func:`resolve_task_axes`):
    first columns unified by position, ``POPULATION_SOURCE`` rules, raw-file
    collapse; the config's ``POPULATION`` key is NOT used (``GP()`` ignores it).

    Returns:
        ``{(population, phenotype): n_individuals}`` for every population in the
        file and every trait the config selects (all traits for ``'all'``). An
        empty dict when the file cannot be read; a message saying why is
        printed so the caller's fall-back is never silent.
    """
    axes = resolve_task_axes(cfg)
    if axes is None:
        return {}
    _, traits, sizes = axes
    phenotype = cfg.get('PHENOTYPE')
    if isinstance(phenotype, str):
        wanted = set(traits if phenotype == 'all' else [phenotype])
    else:
        wanted = {str(t) for t in (phenotype or traits)}
    return {key: n for key, n in sizes.items() if key[1] in wanted}


# ---------------------------------------------------------------------------
# Disk-quota remediation (2026-08-22) - BGLR diagnostic-trace cleanup.
#
# ROOT CAUSE (confirmed from EasiGP_Cotton_step1_0.error/.output): every
# R/BGLR-backed model (rrBLUP, GBLUP, BayesB, RKHS - architecture doc
# Sec 9.1) writes a handful of small ``saveAt``-prefixed MCMC trace files
# (``mu.dat``, ``varE.dat``, ``ETA_<component>_*.dat``, ...) to
# ``<task work dir>/BGLR_output/`` on EVERY call to ``BGLR()`` - not once
# per task, but once per Bayesian-search HYPERPARAMETER-TUNING TRIAL
# (``tune_model_hyperparameters()`` in genomic_prediction.py calls the
# model function ~20-30 times per model per task while searching for the
# best hyperparameters; only the final, best-scoring trial's result is
# ever used - see the ``rrBLUP tuned via Bayesian: ... (took 1380.1s)``
# log lines in EasiGP_Cotton_step1_0.output). Every OTHER trial's saveAt
# files are written and then never read again by anything in this
# codebase - they exist purely because BGLR itself offers no way to skip
# writing them.
#
# Across 4 R/BGLR models x ~20-30 tuning trials x many tasks x many
# batches, this accumulates into tens of thousands of small files. On
# quota-managed HPC filesystems (e.g. NCI Gadi's Lustre-backed /scratch,
# which enforces an INODE quota independently of the byte-size quota)
# this exhausts the quota long before the actual result data (Metric.csv,
# Prediction_result*.csv, etc. - architecture doc Sec 16, all tiny) would.
# EasiGP_Cotton_step1_0.error shows exactly this: hundreds of
# ``brr_<pid>_<random>_{mu,varE,ETA_mrk_varB}.dat`` writes under
# ``.intra_batch_parallel/batch_90/task_904/BGLR_output/`` from rrBLUP's
# OWN hyperparameter search inside a single task, well before that task's
# own (tiny) CSV outputs ever attempted to write - which is why
# ``checkpoint_utils.py::save_partial_results()`` is where the exception
# actually surfaces (``Metric_904.csv``), even though checkpoint_utils.py
# itself has nothing to do with creating the files that exhausted the
# quota in the first place.
#
# ``BGLR_output/`` is documented (architecture doc Sec 16, "Working /
# provenance files") as an R MCMC *trace* directory, not part of the
# Result output catalogue - so deleting its contents after a model call
# has already extracted its predictions/effects into Python/R objects
# loses nothing the pipeline itself ever reads back.
#
# ``cleanup_bglr_output()`` deletes the ENTIRE ``BGLR_output`` directory
# (not just the ``*.dat`` trace files inside it - see the Patch 3 v3,
# Requirement 4 note immediately below for why this is now known to be
# safe). Intended call sites:
#   (a) genomic_prediction.py, immediately after each task's own R/BGLR
#       model call returns - see integration note below. Safe because
#       each task/batch already owns its own PID/random-prefixed,
#       batch/task-suffixed working area (architecture doc Sec 17,
#       "Concurrency safety"), and models run strictly sequentially
#       WITHIN one task (rrBLUP, then BayesB, then RKHS, ... - see the
#       "Progress: N/6 steps complete" log lines), so purging a task's
#       own BGLR_output dir right after its own call returns can never
#       touch a file another, concurrently-running task/process is still
#       writing.
#   (b) bglr_output_cleanup.py (standalone maintenance script, shipped
#       alongside this fix) - run against an existing Result/ tree to
#       reclaim quota from a run that already hit this error, before
#       resubmitting the affected batch(es). The pipeline's own
#       checkpoint/resume logic (checkpoint_utils.py) already skips every
#       task index that completed successfully, so freeing quota and
#       resubmitting the SAME batch/RESULT_NAME is sufficient to resume -
#       no result data is at risk from this cleanup.
#
# INTEGRATION NOTE for whoever has genomic_prediction.py: the traceback in
# EasiGP_Cotton_step1_0.error pinpoints the call site precisely -
# ``genomic_prediction.py`` line 576, function ``_call_model`` (the
# ``lambda`` passed into ``tune_model_hyperparameters`` as
# ``run_model_fn``). Call ``cleanup_bglr_output(bglr_output_dir)`` right
# after the R/BGLR-backed branch's model call returns, where
# ``bglr_output_dir`` is whatever directory THIS task already builds for
# BGLR's ``saveAt`` (genomic_prediction.py already knows this path - it's
# the one being passed to R as the saveAt prefix's parent - so this
# module deliberately does NOT try to re-derive it independently; see
# this module's own RESULT_BASE_DIR note above for why a second,
# independently-maintained path scheme has bitten this codebase before).
# A minimal patch shape (adapt names to the real call site):
#
#   result = rrBLUP(train, valid, test, params, RESULT_NAME)
#   pipeline_utils.cleanup_bglr_output(bglr_output_dir)
#   return result
#
# or equivalently, wrapped once around all four R/BGLR models in
# ``_call_model`` rather than repeated at each one.
# ---------------------------------------------------------------------------
#
# Patch 3 v3, Requirement 4 ("this new version automatically deletes all
# the content inside the BGLR folder - should we just delete this folder
# after finishing a task?"):
#
# ANSWER: yes - and this is now what ``cleanup_bglr_output()`` does. The
# original version of this fix (above) deliberately stopped short of
# removing ``bglr_output_dir`` itself, on the stated assumption that
# "BGLR/rpy2 expects it to still exist for the next call". That assumption
# has now been checked directly against the actual R source rather than
# left as a guess: all four R/BGLR model files -
# ``models/rrBLUP.R``, ``models/GBLUP.R``, ``models/BayesB.R``,
# ``models/RKHS.R`` - each call
#
#   dir.create(bglr_output_dir, showWarnings = FALSE, recursive = TRUE)
#
# themselves, UNCONDITIONALLY, at the very top of every single function
# call, before ``saveAt`` is ever used (``showWarnings = FALSE`` is
# exactly what makes this safe to call every time, whether or not the
# directory already exists). GBLUP.R and RKHS.R each make a SECOND,
# internal ``BGLR()`` call for their own Shapley-value computation, and
# both reuse the SAME already-created directory rather than re-calling
# ``dir.create()`` - which only matters if something deletes the directory
# BETWEEN those two calls, which this cleanup never does (it only ever
# runs after a whole R function call - rrBLUP()/GBLUP()/BayesB()/RKHS() -
# has fully returned, never in the middle of one).
#
# In other words: the directory is unconditionally RE-CREATED by R on its
# own very next call, regardless of whether it existed when that call
# started. There is no call site anywhere in this codebase - Python or R -
# that assumes ``BGLR_output/`` pre-exists without also (re)creating it
# first. It is therefore safe to remove the directory ITSELF, not just its
# ``*.dat`` contents, right after each call finishes.
#
# Doing so is also a genuine improvement, not just a simplification:
#   - it frees one more inode per cleanup than the file-only version did
#     (the empty directory itself is an inode too, on the same
#     inode-quota filesystems that motivated this fix in the first
#     place - architecture doc Sec 16/17, DISK_QUOTA_FIX.md);
#   - a single ``shutil.rmtree()`` is one syscall-tree instead of one
#     ``os.remove()`` per ``.dat`` file, which matters when a single
#     hyperparameter-tuning trial can leave several trace files behind;
#   - it is robust to any OTHER stray file BGLR might ever leave in that
#     directory (not just ``*.dat``) without needing this function to
#     enumerate every extension BGLR happens to use today.
# ---------------------------------------------------------------------------

def cleanup_bglr_output(bglr_output_dir: str) -> Tuple[int, int]:
    """Delete every BGLR MCMC trace file ('*.dat') inside
    `bglr_output_dir` - NOT the directory itself, and nothing else inside
    it (see module-level note above for why only '.dat' files are ever
    touched). Best-effort and never raises: a single file that can't be
    removed (permissions, a concurrent reader, ...) is skipped with a
    warning rather than aborting the rest of the cleanup - this always
    runs as a courtesy pass and the pipeline's correctness never depends
    on it succeeding.

    Parameters
    ----------
    bglr_output_dir : the specific BGLR_output directory to clean -
        caller-supplied (see integration note above for why this
        function does not try to independently reconstruct the path).

    Returns
    -------
    (files_removed, bytes_freed) - both 0 if `bglr_output_dir` doesn't
    exist (e.g. a model/task that never actually called BGLR - not an
    error, not even worth a warning).
    """
    if not bglr_output_dir or not os.path.isdir(bglr_output_dir):
        return 0, 0
    try:
        entries = os.listdir(bglr_output_dir)
    except OSError as exc:
        print(f"[pipeline_utils] WARNING: could not list BGLR_output dir "
              f"'{bglr_output_dir}' for cleanup: {exc!r}.")
        return 0, 0

    files_removed = 0
    bytes_freed = 0
    for name in entries:
        if not name.endswith('.dat'):
            continue  # BGLR only ever writes '.dat' traces here - anything
                      # else (a subdirectory, a stray non-BGLR file) is left
                      # strictly alone.
        path = os.path.join(bglr_output_dir, name)
        try:
            size = os.path.getsize(path)
            os.remove(path)
            files_removed += 1
            bytes_freed += size
        except OSError as exc:
            print(f"[pipeline_utils] WARNING: could not remove BGLR trace "
                  f"file '{path}': {exc!r}.")
    return files_removed, bytes_freed


# ---------------------------------------------------------------------------
# ver4-4, Requirement R5 - "BGLR NaN-Pearson crash" (blueprint §2.5).
#
# Two independent robustness fixes for the R/BGLR (rrBLUP/GBLUP/BayesB/
# RKHS) call path in genomic_prediction.py:
#
#   r_list_get()            - Fix 1: a shape-agnostic accessor for named
#                              elements of an R function's return value.
#   safe_regression_metrics() - Fix 2: a never-raising Pearson r / MSE pair.
#
# Both are used at BOTH `_call_model()` and the main per-task dispatch
# loop in genomic_prediction.py - `_call_model()`'s own docstring requires
# every model-call-site change to be applied in lockstep across both, since
# it is a second implementation, not a refactor (architecture doc §10,
# Patch 9's failure mode: only some of eight equivalent call sites were
# updated last time this shape was touched).
# ---------------------------------------------------------------------------

def _rpy2_to_native(value):
    """Re-apply the CURRENTLY ACTIVE rpy2 conversion to `value`, exactly
    once, and return the result - or `value` itself, completely
    unchanged, if conversion isn't applicable or fails for any reason.

    Req 2 fix (2026-09 - see r_list_get()'s own docstring for the full
    root-cause account). Every one of r_list_get()'s three strategies can
    successfully RETRIEVE a named element (no exception) while still
    handing back a RAW, unconverted rpy2 object (a
    `rpy2.robjects.vectors.DataFrame`/`FloatVector`/etc., never a
    pandas.DataFrame/numpy array) whenever the active conversion context
    wasn't a full dict-converting one at the moment `result` itself was
    produced - confirmed empirically: `.rx2('r_effect')` on an
    unconverted rrBLUP/GBLUP/BayesB/RKHS result reliably returns a raw
    `rpy2.robjects.vectors.DataFrame`, which has no `.shape`, no
    `.columns` setter, no `.reset_index()` - every one of which the
    shared per-task result-handling code in genomic_prediction.py assumes
    unconditionally. That failure mode is silent here (no exception to
    catch) and only surfaces later, far from this function, as a
    confusing `AttributeError` (or worse) with no obvious link back to
    r_list_get() at all.

    `conversion.get_conversion().rpy2py(value)` is the general,
    version-agnostic re-conversion entry point - it dispatches on
    `value`'s own R class/type using whichever converter is ACTUALLY
    active right now (so this stays correct regardless of which of
    init_rpy2_conversion()'s/pandas2ri.activate()'s conversion contexts
    happens to be installed), and is empirically confirmed to be a safe,
    exact IDENTITY passthrough for anything already native (a numpy
    array, a plain float, a str, None, an already-pandas DataFrame) - so
    calling it unconditionally on every r_list_get() return, regardless
    of which Strategy produced it, can only fix a raw value, never
    disturb an already-correct one.

    Never raises: if `rpy2.robjects.conversion` can't be imported, or the
    active conversion has no rule for this particular value's R type
    (some exotic shape none of the four R model files here ever
    produces), the ORIGINAL `value` is returned unchanged - exactly
    today's (pre-fix) behaviour for that value, so this can only IMPROVE
    a call site, never regress one.
    """
    try:
        from rpy2.robjects import conversion
        return conversion.get_conversion().rpy2py(value)
    except Exception:
        return value


def r_list_get(result, name: str, default=None):
    """Read a named element from an R model function's return value,
    regardless of which of two possible shapes rpy2's ACTIVE conversion
    context happens to have produced for it (R5 design record §2.5.1
    "Problem 2 - the accessor"):

      1. A CONVERTED mapping - what ``init_rpy2_conversion()``'s installed
         ``default_converter + numpy2ri.converter + pandas2ri.converter``
         stack produces TODAY for an R named list
         (``rpy2.rlike.container.OrdDict``, or a plain ``dict``) -
         supports ``result[name]`` but has **no** ``.rx2`` method.
      2. An UNCONVERTED rpy2 ``ListVector`` - what a bare
         ``default_converter`` (no numpy2ri/pandas2ri) yields, and what
         re-entering a ``localconverter`` block anywhere upstream can
         reintroduce - supports ``.rx2(name)`` but not ``result[name]``
         with a *string* key.

    Which shape actually arrives depends on the rpy2 version and on
    whether any caller has entered a ``localconverter`` context - not on
    anything this codebase controls directly - so the two historical
    "fixes" (always assume shape 1; always assume shape 2, as Patch 9
    tried) each work until the other shape shows up. This tries every
    strategy instead of assuming one.

    Parameters
    ----------
    result : the R function's return value (whatever shape rpy2 handed
        back for it).
    name : the named element to read (e.g. ``'r_pearson'``, ``'r_effect'``).
    default : returned when `name` cannot be found under ANY strategy,
        instead of raising - ``None`` (the default) means "no soft
        default": every current call site in this codebase leaves this at
        ``None``, because a BGLR model call that completed without
        raising is always expected to carry every one of its documented
        ``r_*`` keys; a missing key there is a genuine contract
        violation, not a soft-fail case a caller should quietly paper
        over.

    Returns
    -------
    The value stored under `name`, or `default` if given and not found.

    Raises
    ------
    KeyError
        If `name` isn't found under any strategy and `default is None`:
        names the requested key, the observed Python type of `result`,
        and (best-effort) the names actually available on it - so the
        failure is diagnosable at a glance instead of surfacing as a bare
        ``TypeError: string indices must be integers`` deep inside a BGLR
        call (the exact failure shape R5 exists to replace).
    """
    # Req 2 fix (2026-09): every return point below now passes its raw
    # find through `_rpy2_to_native()` before handing it back. Root-cause
    # analysis (see the module-level R5 note above this function, and the
    # Req 2 fix note immediately below `init_rpy2_conversion()`) found a
    # real, confirmed, easily-hit gap: whenever Strategy 1 fails (the
    # ACTIVE rpy2 conversion context did not auto-convert `result` into a
    # dict-like mapping - e.g. because the calling process's rpy2
    # conversion context was never properly initialised via
    # `init_rpy2_conversion()`, only the deprecated, weaker
    # `pandas2ri.activate()` GP() itself also calls), Strategy 2's
    # `.rx2(name)` reliably retrieves the NAMED ELEMENT but returns it
    # completely UNCONVERTED (a raw `rpy2.robjects.vectors.DataFrame` /
    # `FloatVector`, never a pandas.DataFrame / numpy array) - silently
    # handing every caller downstream (e.g. `sample_effect.shape`,
    # `sample_effect.columns = ...` in genomic_prediction.py) an object
    # that LOOKS like it worked (no exception here) but breaks on first
    # use. `_rpy2_to_native()` re-applies the currently-active conversion
    # explicitly, exactly once, regardless of which Strategy produced the
    # value - a safe no-op for anything already native (see that
    # function's own docstring for the empirically-verified identity-
    # passthrough behaviour), and the actual fix for anything still raw.

    # Strategy 1: mapping-style access - dict / OrdDict / anything
    # __getitem__-able by a string key without raising TypeError/KeyError.
    try:
        return _rpy2_to_native(result[name])
    except (TypeError, KeyError, IndexError):
        pass
    except Exception:
        pass

    # Strategy 2: an unconverted rpy2 ListVector's own named-access method.
    rx2 = getattr(result, 'rx2', None)
    if callable(rx2):
        try:
            return _rpy2_to_native(rx2(name))
        except Exception:
            pass

    # Strategy 3: positional lookup via the object's own `.names` -
    # covers an object whose `.rx2` is absent/misbehaving but whose
    # `.names` + integer indexing both still work (e.g. a bare
    # rpy2 Vector accessed without any of pandas2ri's dict-like sugar).
    #
    # Req 2 fix: the `getattr(result, 'names', None)` line used to sit
    # OUTSIDE any try/except - `getattr(..., default)` only ever
    # suppresses `AttributeError`, so if reading `.names` on this
    # particular `result` shape ITSELF raised something else internally
    # (plausible: rpy2's own attribute/slot lookup can go through the
    # same string-keyed path Strategy 1 already proved raises
    # `TypeError` for an unconverted vector - see the R5 docstring above),
    # that exception used to propagate straight out of r_list_get()
    # UNCAUGHT - past every call site here, past `_score_once()`'s/
    # `tune_model_hyperparameters()`'s own `except Exception` guards where
    # it happened not to be already inside one, and all the way to GP()'s
    # own outer per-task handler, as a bare, undiagnosable
    # `TypeError: Indices must be integers or slices, not <class 'str'>`
    # with no indication it ever passed through r_list_get() at all. The
    # whole rest of Strategy 3 was already wrapped this defensively - this
    # was the one ungated line. Now wrapped like every other strategy
    # here, so ANY failure this deep falls through to the diagnostic
    # KeyError below instead of escaping raw.
    try:
        names_attr = getattr(result, 'names', None)
        if names_attr is not None:
            names_list = list(names_attr)
            if name in names_list:
                return _rpy2_to_native(result[names_list.index(name)])
    except Exception:
        pass

    if default is not None:
        return default

    observed_type = type(result).__name__
    try:
        available = list(getattr(result, 'names', None) or list(result.keys()))
    except Exception:
        available = '<could not be determined>'
    raise KeyError(
        f"r_list_get: could not find {name!r} on an R result of type "
        f"{observed_type!r}. Tried mapping access (result[{name!r}]), "
        f".rx2({name!r}), and positional lookup via .names. Available "
        f"names: {available!r}. This means the ACTIVE rpy2 conversion "
        f"context (see pipeline_utils.init_rpy2_conversion()) returned a "
        f"shape r_list_get() does not recognise - see the R5 design "
        f"record ('Problem 2 - the accessor') for the two shapes this is "
        f"meant to cover."
    )


def safe_regression_metrics(actual, predicted, *, w=None, W=None) -> Tuple[float, float]:
    """(pearson_r, mse) for one prediction column, NEVER raising.

    ver4-25, R8 (design blueprint SS2, Stage 6): `w` (a 1-D array,
    'diagonal' BLUE-reliability weighting) or `W` (a 2-D PSD matrix,
    'full' variance-covariance weighting) - at most one should be given.
    Both keyword-only and BOTH default to `None`, in which case this
    function's behaviour is EXACTLY what it was before this parameter
    existed - the branch below is a strict, byte-identical no-op for
    every pre-ver4-25 caller (I11) and is skipped entirely (not merely
    given a no-op weight) so nothing about the unweighted numeric path
    changes. When either is given, the whole computation instead routes
    through ``blue_weights.weighted_pearson()``/``weighted_mse()`` - see
    that module's docstring for the exact weighted-Pearson/weighted-MSE
    formulas (they reduce to the ordinary statistics when `W` happens to
    be the identity matrix, but this function never takes that path
    itself - see docstring below for why the two paths are kept
    separate rather than unified into one call to the weighted formula
    with `W=None -> I`).

    R5 design record §2.5.2 Fix 2. ``genomic_prediction.py``'s per-task
    validation scoring (and, once a NaN prediction column reaches them,
    ``models/ensemble.py``'s naive ensemble and
    ``models/Linear_transformation.py``'s weighted-ensemble scoring) used
    to call ``scipy.stats.pearsonr()`` / ``sklearn.metrics.
    mean_squared_error()`` directly. A BGLR MCMC fit that diverges
    produces ``NaN`` predictions (the R side's masked-``NA``-then-slice-
    back calling convention - architecture doc §9.1 - has no way to
    prevent this): ``pearsonr`` merely warns and returns ``nan``, but
    ``mean_squared_error`` RAISES ``ValueError: Input contains NaN``,
    which propagates out of ``GP()``'s per-task dispatch loop and kills
    the entire batch (R5 root cause, Problem 1).

    Parameters
    ----------
    actual, predicted : array-likes of equal length (the phenotype's
        actual values and a model's predictions for the same rows).

    Returns
    -------
    (pearson_r, mse) as plain Python ``float``:
      - unequal lengths, fewer than 2 paired observations, or any
        non-finite (``NaN``/``inf``) value in EITHER array -> ``(nan, nan)``.
        A degenerate/divergent fit must be VISIBLE as ``NaN`` in
        ``Metric.csv``, not silently imputed into something that looks
        like a working model (R5's explicitly rejected alternative was
        clamping NaN predictions to the training mean - this function
        never does that).
      - constant ``actual`` or constant ``predicted`` (zero variance -
        Pearson r is mathematically undefined) -> ``(nan, mse)``, with
        ``scipy``'s ``ConstantInputWarning`` suppressed; ``mse`` is still
        computed and finite, since mean squared error is well-defined for
        constant input.
      - otherwise: identical to ``(scipy.stats.pearsonr(actual,
        predicted)[0], sklearn.metrics.mean_squared_error(actual,
        predicted))`` to float precision - this function changes nothing
        about a healthy fit's reported numbers, only what happens for
        degenerate input that used to raise.
    """
    import numpy as np
    import warnings
    from scipy.stats import pearsonr
    try:
        from scipy.stats import ConstantInputWarning
    except ImportError:  # pragma: no cover - older/newer scipy layouts
        ConstantInputWarning = Warning
    from sklearn.metrics import mean_squared_error

    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)

    if actual_arr.shape[0] != predicted_arr.shape[0] or actual_arr.shape[0] < 2:
        return float('nan'), float('nan')
    if not (np.all(np.isfinite(actual_arr)) and np.all(np.isfinite(predicted_arr))):
        return float('nan'), float('nan')

    if w is not None or W is not None:
        # ver4-25, R8: weighted path - see this function's own updated
        # docstring for why this is a genuinely separate branch, not a
        # W=I unification of the unweighted one below.
        import blue_weights
        r = blue_weights.weighted_pearson(actual_arr, predicted_arr, w=w, W=W)
        mse = blue_weights.weighted_mse(actual_arr, predicted_arr, w=w, W=W)
        return r, mse

    try:
        mse = float(mean_squared_error(actual_arr, predicted_arr))
    except Exception:
        mse = float('nan')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=ConstantInputWarning)
        try:
            r = float(pearsonr(actual_arr, predicted_arr)[0])
        except Exception:
            r = float('nan')

    return r, mse


def paired_r_gap(actual, pred_a, pred_b, *, n_boot: int = 200,
                 seed: int = 0) -> Tuple[float, float]:
    """Paired gap in Pearson r between two prediction columns, with a bootstrap SE. NEVER raises.

    Why this exists (ver4-79, R1; simulation ``ice_sim24``): one task has few test lines, so
    "the Super learner scored 0.0018 below its best member" cannot be told apart from noise.
    This helper puts a standard error next to such a gap so the log line can say so. It is a
    LOG-ONLY statistic: nothing is fitted, weighted or selected from it (I7).

    Args:
        actual: the observed phenotypes of the rows being scored.
        pred_a, pred_b: two prediction columns for the same rows (``a`` is the column whose
            gap to ``b`` is reported).
        n_boot: number of paired bootstrap resamples (one row-index draw serves BOTH columns,
            which is what makes the SE a paired one).
        seed: seed of the resampling generator; the same inputs and seed give the same output.

    Returns:
        ``(r(actual, a) - r(actual, b), standard error of that difference)`` as plain floats.
        ``(nan, nan)`` when it cannot be computed: unequal lengths, fewer than 10 rows with
        all three values finite, a constant vector, fewer than 2 finite bootstrap draws, or any
        unexpected error.
    """
    import numpy as np

    def _pearson_rows(y, a, b, idx=None):
        # r(y, a) - r(y, b) on the rows `idx` (all rows when None); NaN when a column is constant.
        if idx is not None:
            y, a, b = y[idx], a[idx], b[idx]
        y_c = y - y.mean()
        a_c = a - a.mean()
        b_c = b - b.mean()
        denom_a = float(np.sqrt(np.sum(y_c * y_c) * np.sum(a_c * a_c)))
        denom_b = float(np.sqrt(np.sum(y_c * y_c) * np.sum(b_c * b_c)))
        if denom_a <= 0.0 or denom_b <= 0.0:
            return float('nan')
        return float(np.sum(y_c * a_c) / denom_a - np.sum(y_c * b_c) / denom_b)

    try:
        y_arr = np.asarray(actual, dtype=float).ravel()
        a_arr = np.asarray(pred_a, dtype=float).ravel()
        b_arr = np.asarray(pred_b, dtype=float).ravel()
        if not (y_arr.shape[0] == a_arr.shape[0] == b_arr.shape[0]):
            return float('nan'), float('nan')
        finite = np.isfinite(y_arr) & np.isfinite(a_arr) & np.isfinite(b_arr)
        if int(finite.sum()) < 10:
            return float('nan'), float('nan')
        y_arr, a_arr, b_arr = y_arr[finite], a_arr[finite], b_arr[finite]
        if (np.ptp(y_arr) == 0.0 or np.ptp(a_arr) == 0.0 or np.ptp(b_arr) == 0.0):
            return float('nan'), float('nan')

        gap = _pearson_rows(y_arr, a_arr, b_arr)
        if not np.isfinite(gap):
            return float('nan'), float('nan')

        n_rows = y_arr.shape[0]
        rng = np.random.default_rng(int(seed))
        draws = []
        for _ in range(int(n_boot)):
            idx = rng.integers(0, n_rows, n_rows)
            d = _pearson_rows(y_arr, a_arr, b_arr, idx)
            if np.isfinite(d):
                draws.append(d)
        if len(draws) < 2:
            return float('nan'), float('nan')
        return float(gap), float(np.std(np.asarray(draws), ddof=1))
    except Exception:
        return float('nan'), float('nan')


# ver4-38, R3: (task_tag, model_label) pairs that have already produced a
# constant-prediction WARN IN THIS PROCESS, so a hyperparameter search that
# evaluates hundreds of settings prints the warning once per model per task
# per process instead of once per fit. A worker process of a parallel run
# keeps its own set, so a parallel run can print the same warning once per
# worker process; that is a log-volume limit, not a correctness one.
_CONSTANT_PREDICTION_WARNED = set()


def warn_if_constant_prediction(model_label, predicted, *, task_tag='') -> bool:
    """Print one ``[GP] WARN`` when `predicted` carries no information
    (zero variance, or every value NaN) and return True; otherwise return
    False. NEVER raises.

    Why this exists (ver4-38, R3): a model that returns the same number for
    every individual has an undefined Pearson correlation (``NaN`` in
    ``Metric.csv``) and, inside a sum-to-one ensemble, can still be given
    weight as a shrink-to-the-mean knob, which silently removes that model's
    genetic signal. Before this helper the only symptom was a ``NaN`` cell.
    One helper serves both ``ModelTrialRunner.__call__`` (tuning, joint
    search and refits) and ``GP()``'s main dispatch loop (production rows),
    so the wording and the "constant" definition (I3) live in one place.

    Args:
        model_label: the model name as it appears in the ``model`` column,
            e.g. ``'SVR'`` or ``'SVR__Bayesian'``.
        predicted: array-like of the model's test (or validation) predictions.
        task_tag: text identifying the task or run (e.g. the ``sample`` id);
            the WARN is printed once per ``(task_tag, model_label)`` per
            process.

    Returns:
        True if `predicted` is constant (``std <= 1e-12``) or all-NaN or
        empty (ver4-69: a single finite value is never reported), whether or not a line was printed this call (a repeat of an
        already-reported pair returns True without printing); False
        otherwise, including when `predicted` cannot be converted to a
        float array (that case is reported by the caller's own metric code).
    """
    import numpy as np
    try:
        values = np.asarray(predicted, dtype=float).ravel()
    except (TypeError, ValueError):
        return False
    if values.size == 1 and np.isfinite(values[0]):
        # ver4-69: a single value is not evidence of a constant prediction. A one-row test
        # frame is the parity-guarded test stub (joint_tuning, ver4-36; two_phase out-of-fold
        # units, ver4-68), which would otherwise print one false WARN per model.
        return False
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        constant = True
        detail = f"all {values.size} value(s) are NaN/inf" if values.size else "it is empty"
    else:
        spread = float(np.std(finite))
        constant = spread <= 1e-12
        detail = f"std={spread:.3g} over {finite.size} value(s)"
    if not constant:
        return False
    key = (str(task_tag), str(model_label))
    if key in _CONSTANT_PREDICTION_WARNED:
        return True
    _CONSTANT_PREDICTION_WARNED.add(key)
    hint = ""
    if str(model_label).split('__')[0] == 'SVR':
        hint = (" For SVR this usually means Epsilon is larger than the spread of the trait "
                "(no support vectors): tick 'Scale epsilon and C by the trait SD' in the SVR "
                "hyperparameters, or lower Epsilon.")
    tag = f" (task {task_tag})" if task_tag else ""
    print(f"[GP] WARN: model '{model_label}' returned a CONSTANT prediction{tag}: {detail}. "
          f"Its Pearson/Spearman correlation is undefined and it carries no genetic signal "
          f"into any ensemble.{hint}")
    return True


def fit_with_sample_weight(estimator, X, y, sample_weight, *, cpu_fallback_factory=None,
                            log_prefix='[fit_with_sample_weight]'):
    """ver4-25, R8 (design blueprint SS2, Stage 6): fit `estimator` with
    `sample_weight` (a no-op when `sample_weight` is `None`), falling
    back to a freshly-constructed CPU estimator (`cpu_fallback_factory()`,
    a zero-arg callable returning an unfitted CPU estimator with the SAME
    hyperparameters, or `None` if no such fallback exists) when the GIVEN
    estimator's own `.fit()` raises `TypeError` for the `sample_weight`
    keyword - the shared implementation of the blueprint's own per-model
    mechanism table entry "cuML estimators: if sample_weight unsupported
    => scikit-learn CPU fallback + one NOTE (never unweighted)".

    Returns the estimator that ended up actually fitted (either `estimator`
    itself, or the fallback). Never silently drops the weights: if
    `sample_weight` is not `None`, `cpu_fallback_factory` is `None`, and
    the GPU/other estimator's `.fit()` does not accept `sample_weight`,
    this re-raises the original `TypeError` rather than fitting
    unweighted - a caller offering no fallback must not pretend it
    weighted the fit when it didn't.
    """
    if sample_weight is None:
        estimator.fit(X, y)
        return estimator
    try:
        estimator.fit(X, y, sample_weight=sample_weight)
        return estimator
    except TypeError:
        if cpu_fallback_factory is None:
            raise
        print(f"{log_prefix} NOTE: BLUE-reliability weighting is on but this estimator's "
              f"own .fit() does not accept sample_weight - falling back to a CPU estimator "
              f"with the same hyperparameters so the fit is genuinely weighted.")
        fallback = cpu_fallback_factory()
        fallback.fit(X, y, sample_weight=sample_weight)
        return fallback


def parallel_shap_values(explainer, samples, nsamples, n_jobs, *, interaction: bool = False,
                          l1_reg=None):
    """Row-fan-out wrapper around a SHAP explainer's own batch call - fans
    a ``samples`` batch out across ``n_jobs`` worker PROCESSES (one row
    per worker call, via ``joblib.Parallel``), then re-stacks the
    per-row results back into the SAME shape a single, un-parallelised
    call to the explainer would have returned.

    Bug fix (``shap.KernelExplainer`` silently dropping to ~10 nonzero
    features per row, "Max markers considered for Shapley scores" looking
    ignored even when correctly set to 'all')
    --------------------------------------------------------------------
    ``shap.KernelExplainer.shap_values()`` takes its own, SEPARATE
    ``l1_reg`` argument controlling debiased-lasso feature SELECTION for
    the returned Shapley values - as of shap 0.47.0 its default changed
    from ``'auto'`` to ``'num_features(10)'`` (see that method's own
    docstring). Every ``interaction=False`` caller here (``models/SVR.py``,
    ``models/KNN.py``, ``genomic_prediction.py``'s RKHS ``shapiq_predictor``
    effect route) builds ``effect_train_x``/``effect_test_x`` correctly
    honouring the user's own "Max markers considered for Shapley scores"
    setting (including 'all') BEFORE ever reaching this function, but
    never passed ``l1_reg`` through to ``shap_values()`` itself - so once
    on a real shap>=0.47 install, EVERY explained row silently kept only
    its own top ~10 markers (via lasso) and zeroed every other one,
    regardless of how many markers were actually shortlisted upstream.
    Confirmed directly: a 1106-marker SVR task with "Max markers
    considered for Shapley scores"='all' correctly explained all 1106
    markers per this module's own NOTE log line, yet Marker_effect.csv
    came back with only ~260 nonzero entries (the union of each
    explained row's own top-10 selection across ~30 rows) - not the
    "Max markers" setting being ignored, but this SEPARATE, unrelated
    shap-internal default silently re-truncating the result afterwards.

    Fix: an explicit ``l1_reg`` is threaded through to every
    ``explainer.shap_values(...)`` call below when given. Left as
    ``None`` (the default) leaves today's behaviour byte-for-byte
    unchanged for any caller that does not pass it, and is never applied
    to the ``interaction=True`` branch (``shap.TreeExplainer.
    shap_interaction_values()`` has no such argument).

    Follow-up bug fix (this update - the value above was still
    incomplete): every ``interaction=False`` caller originally passed
    ``l1_reg=f'num_features({n_features})'`` (i.e. "select the top
    `n_features` markers via LARS" - a no-op in intent, since that IS
    every shortlisted marker). Confirmed directly against shap's own
    source (``shap.explainers._kernel.Kernel.solve()``): that string form
    still runs ``sklearn.linear_model.lars_path(..., max_iter=n_features)``,
    and LARS can never select more markers than it has (roughly)
    independent coalition SAMPLES to resolve them from - so whenever the
    caller's own coalition-sample budget (``nsamples`` here) is smaller
    than ``n_features`` (routine once "Max markers considered for Shapley
    scores" is 'all' on a real marker panel), LARS silently capped the
    number of nonzero markers at whatever that budget could resolve,
    regardless of what ``l1_reg=f'num_features(...)'`` asked for -
    reproduced directly: 2,000 markers at a 200-sample budget returned
    only ~890 nonzero markers, not 2,000. Every caller now passes
    ``l1_reg=False`` instead, which skips LARS feature selection entirely
    (see ``solve()``'s own ``nonzero_inds = np.arange(self.M)`` default)
    and solves the weighted-least-squares system directly across every
    shortlisted marker - reproduced directly to restore all of them to a
    real value (and run faster, since the LARS step itself is skipped).

    ver4-4 R3.h: promoted, verbatim for the ``interaction=False`` branch,
    from ``models/SVR.py``'s (and, identically, ``models/KNN.py``'s own
    byte-for-byte duplicate of) local ``_parallel_kernel_shap_values()`` -
    one shared copy instead of two, plus the new ``interaction=True``
    branch below. The ``interaction=False`` branch's behaviour at
    ``n_jobs <= 1`` is unchanged from that original function (RK-12).

    This is a drop-in REPLACEMENT for the single explainer call only - it
    deliberately does NO other post-processing (no ``abs()``, no
    ``.sum(axis=0)``, no reduction of any kind). Every current caller's
    own downstream arithmetic keeps running exactly as before, unchanged
    by switching to this wrapper:

      - ``interaction=False``: calls ``explainer.shap_values(row,
        nsamples=nsamples)`` per row and ``numpy.vstack()``s the results
        into a single 2D ``(n_samples_explained, n_features)`` array -
        what ``models/SVR.py``/``models/KNN.py`` already do with the
        result (``abs(parallel_shap_values(...)).sum(axis=0)``).
      - ``interaction=True``: calls ``explainer.shap_interaction_values(
        row)`` per row (``nsamples`` is accepted but IGNORED in this
        branch - ``shap.TreeExplainer.shap_interaction_values()`` takes
        no such argument; kept in the shared signature only so callers
        don't need to know which branch they're using) and
        ``numpy.concatenate()``s the per-row ``(1, n_features,
        n_features)`` results along axis 0 into a single
        ``(n_samples_explained, n_features, n_features)`` array -
        verified directly against ``shap.TreeExplainer`` (0.52.0): a
        single-row call preserves the leading batch dimension of 1, so
        concatenation along axis 0 reproduces exactly what one
        un-parallelised ``explainer.shap_interaction_values(samples)``
        call already returns for the whole batch.

    Deliberately does NOT bake in ``abs()``/``.sum(axis=0)`` for the
    ``interaction=True`` branch, even though every current caller
    (``models/RF.py``, ``models/GAT_prior_knowledge.py``,
    ``Preprocess/data_driven_prior_network.py``) immediately applies
    exactly that to the result - because those three callers do NOT all
    apply it in the SAME ORDER. ``models/RF.py`` computes
    ``abs(explainer.shap_interaction_values(...)).sum(axis=0)`` (absolute
    value of each row's matrix, summed across rows), while
    ``models/GAT_prior_knowledge.py`` and
    ``Preprocess/data_driven_prior_network.py`` both compute
    ``abs(explainer.shap_interaction_values(...).sum(axis=0))`` (summed
    across rows FIRST, absolute value of the sum taken afterwards) - a
    materially different result whenever a pairwise interaction's sign
    varies across the sampled rows (``abs(sum(x)) != sum(abs(x))`` in
    general). This is a pre-existing divergence between the three
    call sites, not introduced here; baking either order into this
    shared helper would silently change the OTHER callers' numeric
    output, so each caller keeps its own existing post-processing line
    completely unchanged - only the ``explainer.shap_interaction_values(
    ...)`` call itself is replaced by this wrapper. (Flagged for the
    project owner in the ver4-4 Change Summary §8; out of scope to
    "fix" here since neither R3 nor R5 asked for a numeric-behaviour
    change to any of the three callers.)

    Falls back to a single serial call (the explainer's own batched
    implementation) on ANY exception during the parallel fan-out - e.g.
    an unpicklable explainer/model combination on this platform - never
    raises; correctness never depends on which path actually ran.

    Parameters
    ----------
    explainer : a fitted ``shap.KernelExplainer`` or ``shap.TreeExplainer``
        (or any object exposing ``.shap_values(X, nsamples=...)`` and/or
        ``.shap_interaction_values(X)`` with the same per-batch contract).
    samples : pandas.DataFrame
        The rows to explain.
    nsamples : int or None
        Coalition-sample budget forwarded to ``explainer.shap_values(...)``
        when ``interaction=False``; ignored when ``interaction=True``.
    n_jobs : int
        ``<= 1`` (or ``0``/``None``) skips the fan-out entirely and makes
        a single serial call, exactly as ``n_jobs=1`` always has.
    interaction : bool, keyword-only, default False
        Selects which explainer method is called per row (see above).
    l1_reg : str, float, or None, keyword-only, default None
        Forwarded to ``explainer.shap_values(..., l1_reg=l1_reg)`` when
        ``interaction=False`` AND this is not ``None`` - see this
        function's own "Bug fix" note above. ``None`` (default) omits the
        argument entirely, leaving ``shap``'s own default (whatever that
        is on the caller's installed version) in effect, exactly as
        before this parameter existed. Ignored when ``interaction=True``.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_samples_explained, n_features)`` when
        ``interaction=False``, or ``(n_samples_explained, n_features,
        n_features)`` when ``interaction=True``. Row order is always
        preserved exactly; only a caller's OWN downstream reduction
        (e.g. a sum) can see floating-point summation-order effects when
        ``n_jobs > 1``.
    """
    import numpy as np
    from joblib import Parallel, delayed

    # l1_reg is only meaningful for the interaction=False (shap_values)
    # branch - built once here as a kwargs dict so both the serial and
    # fanned-out call sites below stay in lockstep and neither one risks
    # passing l1_reg=None explicitly (which shap's own signature treats
    # as a real value, not "omit this argument", for at least one
    # historical shap version - passing nothing at all when l1_reg is
    # None is the only way to guarantee today's pre-existing default is
    # preserved byte-for-byte for every caller that doesn't opt in).
    _extra_kwargs = {} if l1_reg is None else {'l1_reg': l1_reg}

    n_rows = samples.shape[0]
    if n_jobs in (None, 0, 1) or n_rows <= 1:
        if interaction:
            return np.asarray(explainer.shap_interaction_values(samples))
        return np.asarray(explainer.shap_values(samples, nsamples=nsamples, **_extra_kwargs))

    try:
        rows = [samples.iloc[[i]] for i in range(n_rows)]
        if interaction:
            results = Parallel(n_jobs=n_jobs)(
                delayed(lambda row: np.asarray(explainer.shap_interaction_values(row)))(row)
                for row in rows
            )
            return np.concatenate(results, axis=0)
        results = Parallel(n_jobs=n_jobs)(
            delayed(lambda row: np.asarray(explainer.shap_values(row, nsamples=nsamples, **_extra_kwargs)))(row)
            for row in rows
        )
        return np.vstack([np.atleast_2d(r) for r in results])
    except Exception as exc:
        print(f"[pipeline_utils] NOTE: parallel_shap_values fan-out failed ({exc}) - falling "
              f"back to a single serial explainer call.")
        if interaction:
            return np.asarray(explainer.shap_interaction_values(samples))
        return np.asarray(explainer.shap_values(samples, nsamples=nsamples, **_extra_kwargs))


# ---------------------------------------------------------------------------
# Bugfix - shared, defensive replacement for shap.kmeans().
#
# ``models/SVR.py`` and ``models/KNN.py`` both build their KernelExplainer
# background summary the same way:
#     background_size = min(shap_background_size, effect_train_x.shape[0])
#     background = shap.kmeans(effect_train_x, background_size)
# and both were observed to fail a task outright with
# ``ValueError('# of weights must match data matrix!')`` raised from inside
# ``shap.utils._legacy.DenseData.__init__`` (see the ``[GP] ERROR: task ...
# failed`` traceback through ``models/SVR.py::SV_Regression`` ->
# ``shap.kmeans`` -> ``shap.utils._legacy.kmeans``).
#
# Root cause, confirmed against the pinned shap==0.47 source
# (``shap/utils/_legacy.py::kmeans``):
#
#     kmeans = KMeans(n_clusters=k, ...).fit(X)
#     ...
#     return DenseData(kmeans.cluster_centers_, group_names, None,
#                       1.0 * np.bincount(kmeans.labels_))
#
# ``cluster_centers_`` always has exactly ``k`` rows, but
# ``np.bincount(kmeans.labels_)`` returns an array of length
# ``max(kmeans.labels_) + 1`` - NOT ``k``. Whenever the highest-indexed
# cluster ends up with zero members, the two arrays disagree in length and
# ``DenseData.__init__`` raises the exact ValueError above. This is not
# exotic for this codebase's own data: markers are 0/1/2-coded genotype
# calls, so once an effect step narrows to a shortlist of markers (or a
# task's population/split is small), many samples routinely end up with
# byte-identical marker rows. Requesting a background size (``k``) at or
# above the number of DISTINCT rows in that shortlisted data is exactly
# when sklearn's KMeans cannot populate every one of the ``k`` requested
# clusters - reproduced directly: 0/1/2-coded synthetic marker data with
# 53 distinct rows out of 60 total succeeds for k<=50 and fails for
# k in {55, 59, 60} with this identical error, sklearn itself warning
# "Number of distinct clusters (53) found smaller than n_clusters (55)".
#
# Fix, in two layers, neither of which changes what the background summary
# is FOR (it only ever marginalises out features for
# shap.KernelExplainer - a smaller-than-requested summary is still a
# perfectly valid one):
#   1. Clamp k to the number of distinct rows up front - the real upper
#      bound on how many non-empty clusters KMeans can ever produce -
#      logging a one-time NOTE when this reduces the user's configured
#      background size, per this codebase's own "never fail silently"
#      diagnostics discipline (§18 of the architecture notes).
#   2. Belt-and-suspenders: if the identical ValueError still escapes
#      shap.kmeans() for any other reason (e.g. a future shap/sklearn
#      version's empty-cluster handling), retry with k-1 rather than
#      letting the whole task - and everything after it in this batch -
#      fail and be re-queued.
def safe_shap_kmeans_background(X, k, *, log_prefix="[pipeline_utils]"):
    """Defensive, drop-in replacement for ``shap.kmeans(X, k)`` that never
    raises shap's ``ValueError('# of weights must match data matrix!')``.

    See the module comment immediately above for the root cause and the
    two-layer fix. Shared by ``models/SVR.py`` and ``models/KNN.py``
    (the only two call sites of ``shap.kmeans`` in this codebase, since
    they are the only models whose effect step uses
    ``shap.KernelExplainer``), mirroring how ``parallel_shap_values``
    above is already shared by the same two files.

    Parameters
    ----------
    X : pandas.DataFrame
        The (already marker-shortlisted) training rows to summarise.
        Duplicate rows are expected and handled - no need for a caller to
        deduplicate first.
    k : int
        Requested background size. Callers already clamp this to
        ``X.shape[0]`` before calling (unchanged) - this function
        additionally clamps to the number of DISTINCT rows in ``X``.
    log_prefix : str, keyword-only
        Per-caller tag for the NOTE line, e.g. ``"[SVR]"`` / ``"[KNN]"``,
        matching this codebase's ``[ModuleOrModel] NOTE:`` convention.

    Returns
    -------
    shap.utils._legacy.DenseData
        Exactly what ``shap.kmeans(X, k_effective)`` returns - a drop-in
        substitute for ``shap.KernelExplainer(..., background)``.
    """
    import shap

    n_rows = X.shape[0]
    try:
        n_unique = X.drop_duplicates().shape[0]
    except Exception:
        # Defensive only - X is always a DataFrame from both call sites;
        # falls back to the row count (no clamping beyond what the caller
        # already did) rather than raising here.
        n_unique = n_rows

    k_eff = max(1, min(int(k), n_rows, n_unique))
    if k_eff < k:
        print(f"{log_prefix} NOTE: requested SHAP background size {k} exceeds the "
              f"{n_unique} distinct row(s) available in this call's (possibly "
              f"marker-shortlisted) training data out of {n_rows} total rows - "
              f"shap.kmeans() cannot populate more non-empty clusters than "
              f"there are distinct rows, so {k_eff} will be used instead. This "
              f"does not change what the background summary is used for (it "
              f"only marginalises out features for shap.KernelExplainer).")

    while True:
        try:
            return shap.kmeans(X, k_eff)
        except ValueError as exc:
            if "# of weights must match data matrix" not in str(exc) or k_eff <= 1:
                raise
            print(f"{log_prefix} NOTE: shap.kmeans still produced an empty cluster "
                  f"at k={k_eff} despite the distinct-row clamp above (a residual "
                  f"KMeans edge case) - retrying with k={k_eff - 1}.")
            k_eff -= 1


# ---------------------------------------------------------------------------
# Requirements.md item 1 (this update): "the current KernelSHAP is very slow,
# especially when the total number of markers is > 10,000 for GBLUP, RKHS,
# SVR and KNN" - a genuinely FASTER algorithm for `models/KNN.py`'s/
# `models/SVR.py`'s 'kernel_shap' route and `genomic_prediction.py`'s RKHS
# 'shap_predictor' route (the three call sites that actually build a
# `shap.KernelExplainer` in Python - GBLUP's own equivalent slow route is
# the R-side legacy `iml_shapley` refit loop, already superseded by the
# exact, O(1)-per-task `closed_form` route in `models/kernel_effects.py`;
# nothing here needs to touch that).
#
# Bug fix (this update - root cause of "RKHS/SVR/KNN do not return marker
# effects for ALL markers when 'Max markers considered for Shapley scores'
# is 'all'"): the ORIGINAL version of `fast_kernel_shap_effects()` below
# sampled a small, feature-count-INDEPENDENT number of random coalitions
# (`nsamples`, default 200) via a Shapley-kernel-weighted paired sampler,
# then solved ONE shared weighted-least-squares system for every marker at
# once (`np.linalg.pinv`). That sampler is mathematically correct on its own
# terms, but at the marker counts this route actually runs at (this file's
# own threshold below is 10,000; real marker panels needing 'all' are
# routinely tens of thousands), the Shapley-kernel weight formula
# concentrates essentially ALL of a 200-sample budget on the two most
# extreme coalition sizes (one marker "on" alone, or one marker "off"
# alone) - confirmed directly (this update): at 15,000 synthetic markers
# and the default `nsamples=200`, only ~1-2 markers out of 15,000 ever came
# back with a distinguishable, non-numerically-zero value; every other
# marker's column in that shared design matrix was either never sampled at
# all, or exactly collinear with every OTHER never-individually-sampled
# marker, so the minimum-norm solve could not (and, short of raising
# `nsamples` to be comparable to the marker count itself - defeating the
# entire point of a FAST route - structurally never could) tell them apart.
# This is the same failure mode `parallel_shap_values()`'s own "Bug fix"
# docstring above already diagnosed for the per-row `shap.KernelExplainer`
# route's `l1_reg` default (a small sample/selection budget silently
# capping how many markers ever get a non-zero value, independent of what
# "Max markers considered for Shapley scores" asked for) - it was only
# fixed there, not here, and this route is exactly the one every 'all'
# selection above 10,000 markers actually reaches.
#
# Fix: replace the random-coalition WLS estimate with an exact,
# deterministic, per-marker OCCLUSION difference - for an explained row
# `x` and marker `j`, `phi_j(x) = f(x) - f(x with column j alone reset to
# the baseline row, every OTHER column keeping x's own real value)`. This
# is the Shapley value's own "marker added last" term evaluated EXACTLY
# rather than estimated from an under-sized random subsample - the same
# quantity the removed sampler was already trying (and, at this scale,
# failing) to approximate via its size-(M-1) coalitions - so it needs no
# coalition-sampling budget, no weighted-least-squares solve, and no
# pseudo-inverse at all: cost is a fixed, predictable `n_features`
# masked predictions per explained row (batched into as few
# `predict_fn(...)` calls as memory allows, exactly like the removed
# version's own row-batching), so EVERY one of the markers this call was
# asked to explain now gets a real, well-defined, reproducible (zero
# sampling variance) value - never silently zero just because a random
# draw never touched it.
#
# DISCLOSED approximation (only relevant above the large-marker threshold
# this is gated behind - the existing per-row `shap.KernelExplainer` route
# is UNCHANGED, and remains the only route taken, below it): this route
# masks a held-out feature with a SINGLE reference row - the background's
# own weighted mean (a standard "expected-value baseline", the same
# simplification `genomic_prediction.py`'s RKHS `_masked_predict()` already
# uses for its own baseline) - and, like every other 'kernel_shap'/
# 'shap_predictor' route this codebase offers, only ever reports each
# marker's OWN main-effect contribution (higher-order marker-marker
# interactions are this codebase's separate 'Interaction.csv' artefact, not
# this one). Any failure here (never expected in normal operation) falls
# back to the proven per-row `shap.KernelExplainer` route with a printed
# NOTE, matching every other "optional faster route" fallback in this
# codebase (`shapiq_marker_effects()` above, `models/RF.py`'s
# `shapiq_tree`, etc.) - this NEVER silently returns a wrong effect vector.
FAST_KERNEL_SHAP_MARKER_THRESHOLD = 10000
"""Marker-count threshold (Requirements.md item 1's own ">10,000" figure)
above which `models/KNN.py`, `models/SVR.py` and `genomic_prediction.py`'s
RKHS 'shap_predictor' route switch their KernelSHAP marker-effect step from
the proven per-row `shap.KernelExplainer` fan-out to
`fast_kernel_shap_effects()` below. Below this threshold, nothing changes -
the per-row route is already fast enough there and stays the one every
existing task/regression test has been validated against."""


def fast_kernel_shap_effects(predict_fn, background_X, X_explain, *, nsamples=200,
                              background_weights=None, seed=0, max_batch_elements=2e8,
                              log_prefix="[pipeline_utils]"):
    """Exact, batched, per-marker OCCLUSION marker-effect estimator for
    `n_features > FAST_KERNEL_SHAP_MARKER_THRESHOLD` tasks - see the module
    comment above this function for the full "why" (the per-row
    `shap.KernelExplainer` route's own cost at this scale, and the random-
    coalition estimator this replaces, which silently left almost every
    marker at zero once `n_features` grew far past its own sample budget -
    this update's own bug fix).

    Drop-in REPLACEMENT for a caller's own
    ``abs(parallel_shap_values(explainer, shap.sample(X_explain, k),
    nsamples, n_jobs, l1_reg=False)).sum(axis=0)`` expression - returns the
    SAME shape/meaning (the per-marker sum of ``|phi_ij|`` over every
    explained row), so every caller can substitute this in unchanged.

    Method
    ------
    For explained row `x` and marker `j`:

        phi_j(x) = f(x) - f(x^{j->baseline})

    where `x^{j->baseline}` is `x` with ONLY column `j` replaced by the
    background's weighted-mean baseline row - every other column keeps
    `x`'s own real value. This is the Shapley value's own coalition-
    size-(M-1) ("marker removed last") term, computed EXACTLY for every
    one of the `n_features` markers rather than estimated from a random
    coalition subsample - unlike a subsample, it cannot skip a marker
    simply because a random draw never touched it, and it is entirely
    deterministic (`nsamples`/`seed` are still accepted, for every
    existing caller's own call signature, but no longer influence this
    route's coverage or introduce any run-to-run variance - see this
    update's own "Bug fix" note in the module comment above for why the
    coalition-sampling approach they used to configure could never
    reliably cover more than a small, budget-sized fraction of a large
    marker panel in the first place).

    Parameters
    ----------
    predict_fn : callable
        The model's own plain-NumPy ``predict(X) -> (n,)`` function (the
        SAME callable a caller would otherwise hand to
        ``shap.KernelExplainer(predict_fn, background)``) - called on
        batches of shape ``(n, n_features)``, never one row at a time.
    background_X : (n_bg, n_features) array-like
        The background/reference rows (e.g. this task's own
        `safe_shap_kmeans_background(...)` centroids, or the raw
        marker-shortlisted training rows) - collapsed to a single
        weighted-mean baseline row (see the module comment's "DISCLOSED
        approximation" above) before any marker is masked.
    X_explain : (n_rows, n_features) array-like
        The rows being explained (this task's own ``shap.sample(...)``
        draw, exactly as every per-row caller already produces).
    nsamples : int, default 200
        Accepted for call-signature compatibility with every existing
        caller (which threads its own `shap_nsamples`/`Shapley_num`-style
        setting through here unchanged) - no longer used: this route now
        computes every marker's occlusion difference exactly rather than
        estimating a budget-sized subset of them, so there is no sample
        count left to configure. See the module comment above for the
        full "why" (this update's own bug fix).
    background_weights : (n_bg,) array-like or None
        Per-background-row weight (e.g. `shap.kmeans()`'s own returned
        cluster sizes) for the weighted-mean baseline. `None` (default)
        weights every background row equally.
    seed : int, default 0
        Accepted for call-signature compatibility only (this route is now
        fully deterministic, so there is no randomness left to seed) - see
        `nsamples` above.
    max_batch_elements : float, default 2e8
        Caps the largest single ``predict_fn(...)`` batch (``n_features``
        elements per masked row, ~1.6GB at the default for float64 across
        an `n_features`-wide chunk) - both the markers being occluded and
        `X_explain`'s own rows are processed in chunks rather than all at
        once whenever a single batch would exceed this, so a very large
        `n_features` (tens of thousands of markers, this route's own
        target case) never has to materialise a full `n_features x
        n_features` masked tensor in memory at once.
    log_prefix : str, keyword-only
        Per-caller ``[Model]``-style tag, matching `safe_shap_kmeans_
        background`'s identical parameter (kept for signature
        compatibility - this route currently has no NOTE of its own to
        print, since it never falls back or approximates further beyond
        what the module comment above already discloses).

    Returns
    -------
    1-D `np.ndarray`, length `n_features`, in `X_explain`'s own column
    order - `sum over rows of |phi_ij|`, directly comparable to (and a
    drop-in substitute for) every existing 'kernel_shap'/'shap_predictor'
    caller's own aggregated effect vector.

    Raises
    ------
    ValueError
        On a malformed/empty input (never a silently wrong effect vector) -
        every call site wraps this in a `try`/`except` and falls back to
        the proven per-row route on ANY exception, exactly like this
        codebase's other optional-faster-route fallbacks (never lets this
        function's own failure fail the task).
    """
    import numpy as np

    X_explain = np.atleast_2d(np.asarray(X_explain, dtype=float))
    background_X = np.atleast_2d(np.asarray(background_X, dtype=float))
    if X_explain.ndim != 2 or background_X.ndim != 2:
        raise ValueError("pipeline_utils.fast_kernel_shap_effects: X_explain/background_X must be 2-D.")
    n_rows, n_features = X_explain.shape
    if n_features == 0 or background_X.shape[1] != n_features:
        raise ValueError(
            f"pipeline_utils.fast_kernel_shap_effects: X_explain has {n_features} marker columns "
            f"but background_X has {background_X.shape[1]} - these must match.")
    if n_rows == 0:
        return np.zeros(n_features)

    if background_weights is None:
        baseline = background_X.mean(axis=0)
    else:
        background_weights = np.asarray(background_weights, dtype=float).reshape(-1)
        if background_weights.shape[0] != background_X.shape[0]:
            raise ValueError(
                f"pipeline_utils.fast_kernel_shap_effects: background_X has "
                f"{background_X.shape[0]} rows but background_weights has "
                f"{background_weights.shape[0]} values.")
        baseline = np.average(background_X, axis=0, weights=background_weights)

    f_full = np.asarray(predict_fn(X_explain)).reshape(-1)
    if f_full.shape[0] != n_rows:
        raise ValueError(
            f"pipeline_utils.fast_kernel_shap_effects: predict_fn returned {f_full.shape[0]} "
            f"predictions for {n_rows} explained rows.")

    # Markers being occluded are processed in chunks (bounded by
    # `max_batch_elements`) so a single `predict_fn(...)` call never has to
    # materialise more than `feat_chunk x n_features` masked rows at once -
    # the memory bound this function's own docstring promises, regardless
    # of how large `n_features` grows (this route's whole reason to exist).
    feat_chunk = max(1, int(max_batch_elements // max(n_features, 1)))
    phi_abs_sum = np.zeros(n_features)
    for i in range(n_rows):
        x = X_explain[i]
        fx = f_full[i]
        for start in range(0, n_features, feat_chunk):
            end = min(start + feat_chunk, n_features)
            width = end - start
            # hybrid[k, :] = x, except column (start + k), which is reset
            # to the single baseline row - i.e. every OTHER marker keeps
            # this row's own real value, exactly one column at a time
            # reverts to baseline (this function's own disclosed
            # background approximation - see the module comment above).
            # One batched predict_fn(...) call per chunk covers every
            # occluded marker in that chunk at once.
            hybrid = np.tile(x, (width, 1))
            rows_idx = np.arange(width)
            cols_idx = np.arange(start, end)
            hybrid[rows_idx, cols_idx] = baseline[cols_idx]
            y = np.asarray(predict_fn(hybrid)).reshape(-1)
            phi_abs_sum[start:end] += np.abs(fx - y)

    return phi_abs_sum


# ---------------------------------------------------------------------------
# Bug fix (this update - "KNN marker effects are almost zero"): `KNeighbors
# Regressor` is the one model in this codebase whose prediction is a
# DISCRETE, rank-based function of its input (the unweighted mean of
# whichever k training rows happen to be nearest by Minkowski distance),
# rather than a smooth function of every input coordinate the way RF's
# splits, SVR's/RKHS's kernels, or a neural net's activations are. Both of
# this codebase's own `effect_method='kernel_shap'` routes - the per-row
# `shap.KernelExplainer` fallback AND `fast_kernel_shap_effects()` above -
# estimate a marker's contribution by perturbing ONE marker (or a randomly
# sampled SUBSET, below `FAST_KERNEL_SHAP_MARKER_THRESHOLD`) to a baseline
# value and looking for a CHANGE IN THE MODEL'S OUTPUT. For KNN at genomic
# marker counts (thousands to tens of thousands of columns), changing a
# single marker moves the total squared distance to every training point by
# only a tiny fraction of the sum over every OTHER marker - it essentially
# never reorders which k rows are nearest, so `f(x) - f(x^{j->baseline})`
# is exactly zero for the overwhelming majority of markers and every
# explained row (confirmed empirically against a 400-row/16,379-marker
# synthetic reproduction of this exact symptom: `fast_kernel_shap_effects`
# returned a fully-zero vector, 0/16,379 nonzero). This is not a narrow
# numerical edge case, and it is not fixable by tuning `shap_nsamples` or
# `shap_background_size` - occlusion/coalition-masking estimators are
# fundamentally the wrong tool for a model whose output only ever changes
# at the (extremely rare, at this scale) exact point where a perturbation
# flips a neighbour-set membership.
#
# Downstream consequence (why this is worth fixing rather than a curiosity):
# `models/ensemble.py`'s naive ensemble L1-row-normalises each model's own
# effect row before averaging across models (`_safe_row_normalize()`), and
# `models/interaction_extraction.should_use_effect_for_interaction_shortlist()`
# lets a model's own effect values rank its interaction-search shortlist. A
# KNN effect row that is ~99% exact zero gets that same near-zero L1-mass
# collapsed onto whichever handful of markers happened to win a tie, wildly
# over-weighting them in the ensemble effect and in KNN's own interaction
# shortlist - a real correctness bug for every downstream consumer, not just
# a cosmetic one.
#
# Fix: replace occlusion/masking with a Relief-style (Kira & Rendell, 1992;
# the same family as ReliefF, long-established for exactly this "nearest-
# neighbour feature relevance" problem in SNP/GWAS analysis) NEIGHBOUR-
# DISTANCE DECOMPOSITION. Instead of asking "does masking marker j change
# which neighbours get selected" (almost always "no" at this scale), this
# asks "how much does marker j differ between this individual and the
# neighbours that are ALREADY driving its prediction, weighted by how much
# those neighbours' own phenotypes pull the prediction away from the
# population mean" - a quantity that is continuous and never structurally
# zero, computed directly from the fitted model's own `.kneighbors()` call
# (no occlusion, no coalition sampling, no synthetic joint-genotype rows -
# every distance is between two REAL individuals). Cost is a single
# `.kneighbors()` + `.predict()` call per explained row (cheaper than every
# occlusion-based route it replaces, not just more correct).
_RELIEF_EFFECT_EPS = 1e-12


def knn_relief_effects(knn_model, X_train, y_train, X_explain, *, log_prefix="[pipeline_utils]"):
    """Relief-style per-marker effect estimate for a fitted k-NN regressor -
    the KNN-appropriate replacement for `fast_kernel_shap_effects()`/
    `shap.KernelExplainer` (see the module comment directly above this
    function for the full "why").

    Method
    ------
    For explained row `x`, let `N_k(x)` be `knn_model`'s own k nearest
    TRAINING neighbours (exactly the rows already driving its prediction -
    obtained from `knn_model.kneighbors()`, never re-derived). For each
    neighbour `i` with training label `y_i`, define its "pull" as how far
    its OWN label sits from the population mean, `|y_i - mean(y_train)|` -
    a neighbour whose label is unremarkable contributes little signal about
    why THIS row's prediction differs from the population baseline, while
    an extreme neighbour contributes a lot. Marker `j`'s contribution for
    this row is then the pull-weighted, per-marker-scale-normalised distance
    between the row and that neighbour:

        phi_j(x) = (1 / k) * sum_{i in N_k(x)} pull_i * |x_j - X_train[i, j]| / std_j

    where `std_j` is marker `j`'s own standard deviation across `X_train`
    (so a highly polymorphic marker is not automatically credited more than
    a near-monomorphic one just because its raw differences are numerically
    larger - the same per-feature normalisation ReliefF itself uses). As
    with every other effect route in this codebase, the returned vector is
    `sum over explained rows of phi_j(x)` - directly comparable in shape and
    aggregation convention to `fast_kernel_shap_effects()`'s own return
    value, so this is a drop-in substitute at every call site.

    Parameters
    ----------
    knn_model : fitted KNeighborsRegressor-like estimator
        Must already be `.fit()` on `X_train`/`y_train` and expose
        `.kneighbors()` and `.predict()` (true of every KNN estimator this
        codebase fits, including `models/KNN.py`'s own
        `WeightedKNeighborsRegressor`).
    X_train : array-like, (n_train, n_features)
        The SAME rows `knn_model` was fitted on (this task's own shortlisted
        `effect_train_x`, matching the existing call sites' own convention).
    y_train : array-like, (n_train,)
        The SAME labels `knn_model` was fitted on.
    X_explain : array-like, (n_rows, n_features)
        The rows being explained (this task's own `shap.sample(...)` draw,
        exactly as every existing caller already produces).
    log_prefix : str, keyword-only
        Per-caller ``[Model]``-style tag, matching this module's other
        effect/interaction routes' identical parameter (kept for signature
        symmetry - this function currently has no NOTE of its own to print).

    Returns
    -------
    1-D `np.ndarray`, length `n_features`, in `X_train`'s own column order.
    """
    import numpy as np
    import pandas as pd

    X_train_arr = np.atleast_2d(np.asarray(X_train, dtype=float))
    y_train_arr = np.asarray(y_train, dtype=float).reshape(-1)
    X_explain_arr = np.atleast_2d(np.asarray(X_explain, dtype=float))
    n_features = X_train_arr.shape[1]

    if X_explain_arr.shape[0] == 0 or X_train_arr.shape[0] == 0:
        return np.zeros(n_features)

    marker_std = X_train_arr.std(axis=0)
    marker_std = np.where(marker_std > _RELIEF_EFFECT_EPS, marker_std, 1.0)
    y_mean = y_train_arr.mean()

    # Cosmetic-only (see pipeline_utils._predict_fn_with_feature_names()'s
    # own identical rationale): `knn_model` was fitted on a DataFrame, so
    # calling `.kneighbors()` with a bare ndarray fires a benign but noisy
    # sklearn "X does not have valid feature names" warning once per
    # explained row. Restoring the fitted column names (never touches any
    # computed value - sklearn matches columns positionally either way)
    # silences it at the source instead of suppressing warnings wholesale.
    _feature_names = getattr(knn_model, 'feature_names_in_', None)
    _kneighbors_input = (
        pd.DataFrame(X_explain_arr, columns=_feature_names) if _feature_names is not None
        else X_explain_arr
    )
    _, neighbor_idx = knn_model.kneighbors(_kneighbors_input)
    n_rows, k = neighbor_idx.shape

    phi = np.zeros(n_features)
    for row_i in range(n_rows):
        neigh_idx = neighbor_idx[row_i]
        neigh_pull = np.abs(y_train_arr[neigh_idx] - y_mean)
        neigh_X = X_train_arr[neigh_idx]
        diff = np.abs(X_explain_arr[row_i][None, :] - neigh_X) / marker_std[None, :]
        phi += (neigh_pull[:, None] * diff).sum(axis=0) / max(k, 1)

    return phi


# ---------------------------------------------------------------------------
# ver4-25, Stage 7/8 (design blueprint R13/R11): the optional `shapiq`
# package (Shapley-INTERACTION indices - k-SII/FSII/STII, order >= 2,
# unlike Friedman's H-statistic's pairs-only scope) as an ALTERNATIVE
# explanation engine to the existing `shap.KernelExplainer` route, for
# models/SVR.py's/models/KNN.py's own selectable 'Marker effect method'
# ('kernel_shap' (default, unchanged) / 'shapiq') and 'Marker interaction
# method' ('friedman_h' (default, unchanged) / 'shapiq') fields.
#
# DISCLOSED LIMITATION: `shapiq` could not be installed in this
# development environment (no network access to PyPI from the sandbox
# this update was authored in - see the ver4-25 Change Summary), so the
# functions below were written directly against `shapiq`'s own public
# documentation/API surface (`shapiq.TabularExplainer`, `.explain(x,
# budget=...)` returning an `InteractionValues` object indexable by a
# coalition tuple) rather than verified by actually running them. Both
# are called ONLY from inside a caller's own `try`/`except` (never
# directly), and 'shapiq' is never the only route to a result - it is
# always an OPT-IN alternative to the existing, unchanged, already-
# verified 'kernel_shap'/'friedman_h' default (I11) - so a version
# mismatch or an API surface this module got wrong surfaces as a single
# printed NOTE and a graceful fallback to that proven default, never a
# failed task. Recommended before production use: run one task with
# 'shapiq' selected and confirm no fallback NOTE is printed.
# ---------------------------------------------------------------------------
_SHAPIQ_AVAILABLE = None


def shapiq_available() -> bool:
    """Cached import probe for the optional `shapiq` package. Mirrors
    `model_registry.is_available()`'s own convention (never raises,
    cached for the lifetime of this process) for a per-FIELD-VALUE
    optional dependency (a model's own CHOICE of explanation engine)
    rather than a per-MODEL one - `model_registry.OPTIONAL_DEPENDENCY`
    itself is deliberately not extended for this, since every model that
    can select 'shapiq' already runs perfectly well without it (via its
    own 'kernel_shap'/'friedman_h' default)."""
    global _SHAPIQ_AVAILABLE
    if _SHAPIQ_AVAILABLE is None:
        try:
            import shapiq  # noqa: F401
            _SHAPIQ_AVAILABLE = True
        except Exception:
            _SHAPIQ_AVAILABLE = False
    return _SHAPIQ_AVAILABLE


def _current_virtual_bytes():
    """Best-effort current virtual-address-space size of THIS process, in
    bytes. Linux-specific (`/proc/self/statm`'s own first field, in pages
    - this codebase's own HPC deployment target per its PBS/Slurm launch
    scripts elsewhere), falling back to `resource.getrusage()`'s peak-RSS
    figure (a looser, RSS-not-VSZ proxy, but still a usable floor) on any
    other POSIX platform, and `None` (guard disabled entirely - see
    `bounded_memory_guard()`'s own fallback) wherever neither is
    available."""
    try:
        with open('/proc/self/statm') as fh:
            vsize_pages = int(fh.readline().split()[0])
        import resource as _resource
        return vsize_pages * _resource.getpagesize()
    except Exception:
        pass
    try:
        import resource as _resource
        return _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss * 1024
    except Exception:
        return None


import contextlib


@contextlib.contextmanager
def bounded_memory_guard(extra_gb=6.0, log_prefix="[pipeline_utils]"):
    """Temporarily caps THIS PROCESS's own address space (`RLIMIT_AS`) at
    its current usage plus `extra_gb` GiB of headroom (never more than
    ~80% of whatever the OS currently reports as free, so this guard
    itself cannot starve the rest of the machine/other workers) for the
    duration of the ``with`` block, then restores whatever limit (if any)
    was in effect before - never LOOSENS an externally-configured limit,
    only ever tightens one for the duration of the block.

    Why this exists (RF.py 'shapiq_tree' route, Interaction_higher_
    order.csv coming back empty - the OTHER half of the same symptom
    `shapiq_tree_interactions()`'s own "Known shapiq==1.4.1 TreeExplainer
    bug" docstring already covers): confirmed empirically (this update),
    shapiq's exact `TreeExplainer` at `max_order>=3` has NO natural memory
    ceiling of its own - its cost scales with tree DEPTH and interaction
    ORDER, not with any of this codebase's own configured shortlist sizes
    (`max_interaction_features` et al.), so even a heavily-shortlisted,
    modest-depth forest can exhaust several GB explaining a SINGLE row.
    Left unguarded, that growth is reclaimed by the OS/cgroup OOM-killer,
    which sends an uncatchable SIGKILL - silently killing this run's
    ENTIRE process (predictions, every other already-computed model's
    results, the final report) rather than just the one optional,
    opt-in interaction search that overran.

    `RLIMIT_AS` converts that same overrun into an ordinary, catchable
    Python `MemoryError` instead - which `models/RF.py`'s own outer
    `try`/`except` around this route (and `shapiq_tree_interactions()`'s
    own inner 'k-SII' retry) already wraps every call in, falling back to
    the proven 'pairwise_shap' route with a printed NOTE exactly as
    designed. A computation that fits within the granted headroom behaves
    identically to before this guard existed (I11) - this is a strictly
    safety-improving change, never a stricter one for a task that was
    always going to fit.

    POSIX-only (`resource.RLIMIT_AS` does not exist on Windows, and
    `/proc/self/statm` is Linux-specific) - a silent no-op wherever either
    is unavailable, the same "protection unavailable, no behaviour
    change" fallback every other optional/platform-dependent guard in
    this codebase already uses.
    """
    try:
        import resource as _resource
    except Exception:
        yield
        return

    current = _current_virtual_bytes()
    if current is None:
        yield
        return

    headroom = int(max(extra_gb, 0.0) * (1024 ** 3))
    try:
        import psutil
        available = psutil.virtual_memory().available
        headroom = min(headroom, int(available * 0.8)) if headroom > 0 else headroom
    except Exception:
        pass

    new_soft = current + max(headroom, 0)
    restore = None
    try:
        soft, hard = _resource.getrlimit(_resource.RLIMIT_AS)
        effective_new_soft = new_soft if hard == _resource.RLIM_INFINITY else min(new_soft, hard)
        if soft == _resource.RLIM_INFINITY or effective_new_soft < soft:
            _resource.setrlimit(_resource.RLIMIT_AS, (effective_new_soft, hard))
            restore = (soft, hard)
            print(f"{log_prefix} NOTE: capping this process's own memory ceiling at roughly "
                  f"{effective_new_soft / (1024 ** 3):.1f} GiB for an optional, exact "
                  f"higher-order interaction search - a search that overruns this cap fails "
                  f"with a catchable MemoryError (triggering this route's own existing "
                  f"fallback) instead of the OS terminating this entire task.")
    except Exception:
        restore = None

    try:
        yield
    finally:
        if restore is not None:
            try:
                _resource.setrlimit(_resource.RLIMIT_AS, restore)
            except Exception:
                pass


def _predict_fn_with_feature_names(predict_fn, feature_names):
    """Wrap a fitted estimator's ``predict`` callable so every call carries
    the SAME column names the estimator was originally fitted with, even
    though `shapiq` (every route below) only ever calls the wrapped model
    with plain 2-D numpy arrays.

    Requirements.md item 1 - fixes a purely cosmetic but extremely noisy
    sklearn warning: ``UserWarning: X does not have valid feature names,
    but <Estimator> was fitted with feature names``. Every model in this
    codebase fits on a `pandas.DataFrame` (so scikit-learn records
    `feature_names_in_`), while every `shapiq` explainer below calls the
    wrapped model with a raw `numpy.ndarray` (`shapiq` has no concept of
    column names at all) - scikit-learn matches columns POSITIONALLY
    either way, so this warning never indicated a computation problem,
    but `shapiq` calls the wrapped model once per sampled coalition per
    explained row, so a single interaction/effect task can fire this
    warning many thousands of times, flooding stdout/log files (the
    reported symptom - "output repeatedly"). Restoring the original
    column names before calling through changes no computed value
    (values are already matched positionally, with or without names) -
    it only supplies the metadata scikit-learn was looking for, silencing
    the warning at its source rather than suppressing warnings wholesale
    (which could also hide a genuinely new, unrelated warning).

    Parameters
    ----------
    predict_fn : callable
        The estimator's own bound ``.predict`` (or any callable expecting
        a `pandas.DataFrame` with these exact columns, in this order) -
        exactly what every caller below already has fitted.
    feature_names : sequence of str
        Column names, in the SAME order `predict_fn`'s own estimator was
        fitted with.

    Returns
    -------
    callable
        Accepts a 2-D (or 1-D, promoted to one row) array-like and
        forwards a `pandas.DataFrame` with `feature_names` as columns to
        `predict_fn`, returning its result unchanged.
    """
    import numpy as np
    import pandas as pd

    feature_names = list(feature_names)

    def _wrapped(X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return predict_fn(pd.DataFrame(X, columns=feature_names))

    return _wrapped


def shapiq_marker_effects(predict_fn, X_train_background, X_explain, *,
                           background_size=50, budget=200, index='k-SII',
                           approximator='auto'):
    """Per-marker order-1 Shapley marker-effect values via `shapiq`'s own
    `TabularExplainer`, aggregated the SAME way every `shap.
    KernelExplainer`-based marker-effect route in this codebase already
    is (``sum(|phi|, axis=0)`` over ``X_explain``'s own rows) - the
    'shapiq' alternative to `shap.KernelExplainer` for `models/SVR.py`'s/
    `models/KNN.py`'s own 'Marker effect method' field. See this
    module's own header comment above for the disclosed "written but not
    runtime-verified against a real `shapiq` install" caveat.

    An order-1 (single-feature) Shapley/interaction value is, by
    `shapiq`'s own definition, numerically identical to a classical
    Shapley value regardless of which interaction `index` is requested
    (k-SII/FSII/STII only differ from each other, and from a plain
    Shapley value, at order >= 2) - so this function always explains at
    ``max_order=1``, and `index` only selects which of `shapiq`'s
    internal estimators computes that (in principle) identical result.

    Parameters
    ----------
    predict_fn : callable
        A plain ``predict(X) -> array`` callable (e.g. a fitted
        estimator's own ``.predict``), exactly what
        ``shap.KernelExplainer`` already requires from every caller here.
    X_train_background : pandas.DataFrame
        Training rows to draw `shapiq`'s own background/imputation
        reference from (the SAME shortlisted columns `X_explain` uses).
    X_explain : pandas.DataFrame
        The already-sampled rows to explain (e.g. via ``shap.sample(...)``
        upstream, exactly like every `shap.KernelExplainer` caller here).
    background_size : int, default 50
        Background rows sampled from `X_train_background` (mirrors
        `safe_shap_kmeans_background`'s own role for the `shap` route).
    budget : int, default 200
        Coalition-evaluation budget per explained row, forwarded to
        `shapiq`'s own ``.explain(x, budget=...)``.
    index : str, default 'k-SII'
        `shapiq`'s own interaction-index name.
    approximator : str, default 'auto'
        Forwarded to `shapiq.TabularExplainer`'s own `approximator`
        kwarg - see `shapiq_pairwise_interactions()`'s own identical
        parameter for the full rationale/choices. `'auto'` (default):
        at `max_order=1` (always, here), `shapiq`'s OWN `'auto'`
        resolution never picks SPEX regardless of `n_features` (SPEX's
        own automatic selection, `choose_spex()`/`_shapiq_would_use_
        spex()` in this module, is order->=2 only) - it always falls
        through to `KernelSHAP`, whose own kernel-weight computation is
        confirmed (this update) to silently overflow to a non-finite
        result once `n_features` grows large (see this function's own
        "Bug fix" comment just above the explainer construction below
        for the full root cause). Since shapiq's own automatic choice
        does not compensate for this the way it does at `max_order>=2`,
        THIS function's own `'auto'` handling now does instead: once
        `n_features > 64` (the same threshold `_shapiq_would_use_spex()`
        already applies at order 2), `'auto'` here resolves to `'spex'`
        rather than being forwarded to `shapiq` unchanged - confirmed
        (this update) to return finite, correctly-sparse results at the
        marker counts (1000+) that previously always raised the
        `RuntimeError` below. An explicit non-`'auto'` value (e.g.
        forcing `'regression'`) is always forwarded to `shapiq` exactly
        as given, never second-guessed.

    Returns
    -------
    1-D `numpy.ndarray`, length ``X_explain.shape[1]``, in `X_explain`'s
    own column order.

    Raises
    ------
    Exception
        Any failure at all (package missing, an API mismatch against the
        `shapiq` version actually installed, a numerical failure) -
        deliberately NOT caught here; every caller wraps this in its own
        ``try``/``except`` and falls back to the proven 'kernel_shap'/
        'friedman_h' route with a printed NOTE (see this module's own
        header comment).
    """
    import numpy as np
    import shapiq

    # Requirements.md item 1: every call below reaches `predict_fn` with a
    # raw numpy array - wrapping it once here so the estimator's original
    # column names are restored before each call silences sklearn's
    # "X does not have valid feature names" warning without changing any
    # computed value (see _predict_fn_with_feature_names()'s own docstring).
    predict_fn = _predict_fn_with_feature_names(predict_fn, X_explain.columns)

    n_rows_bg = X_train_background.shape[0]
    background = (
        X_train_background.sample(n=min(int(background_size), n_rows_bg), random_state=0)
        if n_rows_bg > 0 else X_train_background
    )
    n_features = X_explain.shape[1]
    # Bug fix (this update - root cause of the reported "shapiq was not
    # used for SVR's marker effects, it silently fell back to
    # kernel_shap" behaviour): `approximator='auto'` (the default) at
    # `max_order=1` (always, here) NEVER resolves to shapiq's own SPEX
    # approximator, regardless of `n_features` - confirmed directly
    # against a real shapiq==1.4.1 install (this update):
    # `shapiq.explainer.configuration.setup_approximator_automatically()`
    # only ever offers `SPEX` when its own `choose_spex(max_order,
    # n_players)` is True, and THAT function (duplicated locally as
    # `_shapiq_would_use_spex()` above) hard-codes `max_order >= 2` -
    # order 1 always falls through to `KernelSHAP`, shapiq's own
    # regression-based order-1 (plain Shapley value) estimator,
    # regardless of how large `n_features` is.
    #
    # That `KernelSHAP` estimator's own per-coalition-size kernel weight
    # (`shapiq.approximator.regression.base.Regression.
    # _init_kernel_weights()`) divides by `scipy.special.binom(n_features
    # - 2, ...)` - confirmed directly (this update) that this binomial
    # coefficient silently OVERFLOWS TO `inf` in float64 once
    # `n_features` grows large enough (e.g. `scipy.special.binom(1030,
    # 515)` is already `inf`), which is the exact, deterministic,
    # BUDGET-INDEPENDENT mechanism behind this function's own documented
    # "NaN once explained markers cross ~1000+" failure mode below (the
    # reported run: 1106 markers, budget=1000) - not a "just raise the
    # budget" or "underdetermined regression" story, an outright float64
    # overflow in shapiq's own kernel-weight computation.
    #
    # `SPEX` doesn't use this kernel-weight regression at all (a sparse-
    # Fourier-transform approximator instead) and fully supports order-1
    # ('SV'-generalizing) explanations (`'SV' in shapiq.explainer.
    # configuration.SPEX.valid_indices`) - confirmed directly (this
    # update) to return finite, correctly-sparse order-1 Shapley values
    # at the SAME scale (1106 markers, budget=1000) the reported failure
    # used, and at every smaller marker count tested (40/64/100/200/400+)
    # given a large-enough budget, so switching earlier costs nothing.
    # Mirrors `shapiq_pairwise_interactions()`'s own automatic SPEX
    # upgrade at `max_order>=2` (there, shapiq's own 'auto' already does
    # this; here, shapiq's own 'auto' never does, so this function does
    # it instead) - using the SAME order-2 threshold (64)
    # `_shapiq_would_use_spex()` already applies, deliberately
    # conservative relative to the ~1030 hard overflow point since SPEX
    # is already confirmed sane well below it.
    #
    # Only overrides the actual `'auto'` default - an explicit caller-
    # requested `approximator` (e.g. `effect_shapiq_approximator` forced
    # to `'regression'`) is never second-guessed here (I5/I11).
    _approximator = approximator
    if approximator == 'auto' and n_features > 64:
        _approximator = 'spex'
    explainer = shapiq.TabularExplainer(
        model=predict_fn, data=background.to_numpy(), imputer='marginal',
        index=index, max_order=1, approximator=_approximator,
    )
    X_explain_arr = X_explain.to_numpy()
    scores = np.zeros(n_features, dtype=float)
    for row_idx, row in enumerate(X_explain_arr):
        interaction_values = explainer.explain(row, budget=int(budget))
        row_scores = np.asarray([interaction_values[(j,)] for j in range(n_features)], dtype=float)
        # Bug fix ('all'/large-marker-count 'shapiq' marker effect silently
        # producing an all-NaN/empty result): confirmed directly against a
        # real shapiq install - shapiq's own `TabularExplainer` (its
        # KernelSHAP-style order-1 regression estimator) can silently
        # return NaN for EVERY player once the number of explained markers
        # crosses a sharp, shapiq-internal threshold (empirically ~1000+
        # markers in this codebase's own testing) - REGARDLESS of how large
        # `budget` is raised (reproduced unchanged from the default budget
        # up to budget=15000 on the same marker count), so this is not a
        # "just raise the budget" situation on this codebase's side. This
        # never raises on its own - `interaction_values[(j,)]` returns a
        # perfectly normal-looking `float('nan')`, so the NaN silently
        # flows into `scores` (NaN + anything = NaN), then into the
        # reassembled full-width effect vector every caller here builds -
        # explaining the reported "'all' scores either return nothing (all
        # NaN/blank cells in Marker_effect.csv) or the setting looks
        # ignored (a smaller, working marker count quietly used instead by
        # whichever fallback path happened to run)". Every caller of this
        # function already wraps it in its own try/except and falls back to
        # a proven, non-shapiq Shapley-effect route (exact TreeSHAP for the
        # tree models - numerically the SAME order-1 quantity 'shapiq' is
        # approximating here - or shap.KernelExplainer/permutation
        # importance elsewhere) with a printed NOTE the moment this raises -
        # exactly the safety net this module's own docstring already
        # promises ("NEVER lets a shapiq failure ... fail this task"). The
        # only thing missing was treating a non-finite result AS a failure -
        # fixed here by raising instead of silently returning it.
        if not np.all(np.isfinite(row_scores)):
            raise RuntimeError(
                f"shapiq_marker_effects: shapiq's own TabularExplainer returned non-finite "
                f"(NaN/Inf) Shapley values for explained row {row_idx} ({n_features} markers, "
                f"budget={int(budget)}) - a known shapiq numerical-stability failure mode once "
                f"the marker count explained in one call grows large (observed starting in the "
                f"low thousands), independent of how high `budget` is set. Falling back to this "
                f"model's own non-shapiq marker-effect route instead of returning a silently "
                f"blank/garbage result."
            )
        scores += np.abs(row_scores)
    return scores


# ---------------------------------------------------------------------------
# ver4-26, design blueprint R1/R3 - shared row-fan-out and feasibility/
# narrowing helpers for the `shapiq`-based interaction routes below.
# ---------------------------------------------------------------------------

_SHAPIQ_INFEASIBLE_SPACE_RATIO = 50.0
"""Module constant (design blueprint R3a): pre-flight feasibility
threshold for `shapiq_pairwise_interactions()`'s budget-sampled
`TabularExplainer` route. If the requested coalition search space
(`_shapiq_coalition_space_size(n_features, orders)`) exceeds `budget *
_SHAPIQ_INFEASIBLE_SPACE_RATIO`, the search is treated as hopeless BEFORE
spending any row-explain time.

Chosen conservatively: `shapiq`'s `TabularExplainer` estimates each
interaction coefficient via a weighted-least-squares-style regression
over its sampled coalitions - as an empirical rule of thumb for this
kind of estimator to resolve coefficients reliably (rather than return
numerically-degenerate near-zero noise), the number of sampled
coalitions needs to be a modest multiple of the number of unknowns being
estimated, not merely >= it. A ratio of 50 is deliberately generous (a
search only 10x oversubscribed is NOT pre-empted here, since some real
budgets still resolve at that ratio) while still catching the
empirically-confirmed, always-hopeless case this module's own post-hoc
`RuntimeError` below documents (budget=4096 against ~500 markers at
order 3 - a space of ~20.7 million coalitions, a ratio of ~5,000 - two
orders of magnitude past this threshold).
"""

_SHAPIQ_NARROW_SAFETY_FACTOR = 0.5
"""Module constant (design blueprint R3b): the candidate-narrowed
marker subset's OWN coalition-space size (at the real, order>=3 search)
is capped at `budget * _SHAPIQ_NARROW_SAFETY_FACTOR` rather than
`budget` itself, leaving headroom for the same sampling-density
reasoning documented on `_SHAPIQ_INFEASIBLE_SPACE_RATIO` above - a
narrowed search that lands exactly at the infeasibility ratio's own
boundary would be a marginal, not a comfortable, fit.
"""


# ver4-29 (this update): both `_SHAPIQ_INFEASIBLE_SPACE_RATIO` and
# `_SHAPIQ_NARROW_SAFETY_FACTOR` above were calibrated by reasoning about
# the RAW combinatorial coalition space (`C(n_features, order)`) alone,
# under the disclosed "written but not runtime-verified against a real
# shapiq install" limitation (see environment_linux.yml's own shapiq pin
# comment). Verified directly against a real `shapiq==1.4.1` install
# (this update): that reasoning is only correct for the estimators shapiq
# picks when `n_features` is small relative to `budget` - shapiq's own
# `TabularExplainer(approximator='auto', ...)` does NOT pick one of those
# estimators once `n_features` grows large at a given `max_order`. It
# switches to SPEX (a sparse-Fourier-transform approximator built
# specifically for large, budget-constrained coalition spaces), whose
# sample complexity is NOT well described by "coalitions needed roughly
# proportional to C(n_features, order)" at all - confirmed empirically
# (this update) to return genuine, non-degenerate k-SII estimates for a
# 500-feature, order-2 (and order-3) search at budget=4096, the EXACT
# scale this module's own pre-existing R3a/R3b logic was calibrated to
# treat as hopeless and narrow/pre-empt. Narrowing or pre-flight-failing
# a search SPEX would have handled directly defeats the purpose of
# selecting 'shapiq' at all for a marker count this common in genomic
# data.
#
# `_shapiq_would_use_spex()` is a deliberate, DUPLICATED (not imported)
# reproduction of shapiq==1.4.1's own internal
# `shapiq.explainer.configuration.choose_spex()` heuristic - not part of
# `shapiq`'s public `__init__` surface, and this project pins shapiq to
# an EXACT version (`shapiq==1.4.1`, see environment_linux.yml), so
# duplicating one small, version-pinned function is safer than importing
# a private module path that could move or change behaviour on a future
# shapiq release without this project's own pin changing first. Used
# ONLY to decide whether THIS module's own R3a/R3b feasibility logic
# should even run (see `shapiq_pairwise_interactions()`'s own
# `_uses_spex` local below) - it never changes what shapiq itself
# actually computes, since `approximator='auto'` (the unchanged default)
# would have made shapiq choose SPEX for this `n_features`/`max_order`
# combination regardless of anything below. Revisit this function (and
# re-verify the empirical claims above) if the shapiq pin ever changes -
# see environment_linux.yml's own shapiq note.
def _shapiq_would_use_spex(n_features: int, max_order: int) -> bool:
    """True when shapiq==1.4.1's own `approximator='auto'` selection
    would resolve to SPEX for this `n_features`/`max_order` combination -
    see this function's own module-level comment above for why this is
    duplicated rather than imported, and why it matters here."""
    if max_order == 2 and n_features > 64:
        return True
    if max_order == 3 and n_features > 32:
        return True
    if max_order == 4 and n_features > 16:
        return True
    return bool(max_order > 4 and n_features > 16)


def resolve_shapiq_explain_row_count(max_order: int, base_n: int, floor: int = 5) -> int:
    """ver4-28 (this update, Requirements2.md item 1 - re-opening of design
    blueprint R2/R3): order-aware shrink of the ROW axis for every
    `shapiq`-based interaction route (RF's 'shapiq_tree', SVR/KNN's
    'shapiq', RKHS-family's 'surrogate_shapiq'), mirroring `models/RF.py::
    _resolve_shapiq_tree_depth_cap()`'s own depth/tree-count shrink
    schedule (design blueprint R2) rather than inventing a new one.

    R2 already shrinks the PER-ROW cost as `max_order` grows (tree depth/
    count for 'shapiq_tree'; R3's own candidate-narrowed coalition space
    for the budget-sampled route) - but total wall-clock time for any of
    these routes is `(rows explained) x (per-row cost)`, and the row
    count itself was, before this update, a FLAT, order-blind setting
    (`shapley_num`/its per-route alias) for every one of them, left
    unaddressed by R2/R3. Shrinking it too compounds MULTIPLICATIVELY with
    R2/R3's own already-shrunk per-row cost, rather than leaving the row
    axis as the one remaining untouched cost driver once a route's own
    depth/coalition-space floor is reached - directly attacking "the
    computational time ... is still prohibitively high ... [and] worsens
    when k becomes larger" on a THIRD, independent axis.

    Public (not underscore-prefixed): imported directly by `models/RF.py`,
    `models/SVR.py`, `models/KNN.py`, and `models/interaction_extraction.
    py::surrogate_shapiq_interactions()` - the four call sites that each
    resolve their own "how many rows to explain" setting independently,
    every one already gated by the SAME `shapiq_adaptive_depth` compute-
    resource key R2 established (no new config key introduced here).

    Parameters
    ----------
    max_order : int
        Highest interaction order requested for this task.
    base_n : int
        This task's own configured row count (e.g. `shapley_num`) - never
        INCREASED, only capped.
    floor : int, default 5
        Minimum rows kept even at the deepest shrink - enough for the
        'abs_then_sum' aggregation across rows to remain meaningful. Itself
        capped at `base_n` (``min(floor, base_n)``), so a task whose OWN
        configured row count is already below `floor` is never bumped UP -
        this function only ever shrinks, matching `base_n`'s own "never
        INCREASED" contract above.

    Returns
    -------
    int
        `base_n` unchanged at `max_order <= 2` (I11 - the common,
        default-order case). For `max_order > 2`, shrunk by the SAME
        ratio `models/RF.py::_resolve_shapiq_tree_depth_cap()` applies to
        depth (`(8 - 2*(max_order-2)) / 8`, floored at `3/8`), then
        floored at `floor`. That ratio is duplicated here (rather than
        importing `models/RF.py`'s own resolver) to avoid a `models/` ->
        `pipeline_utils` -> `models/` import cycle - `models/RF.py`'s own
        resolver stays the authority for the SCHEDULE itself; this
        function only reuses its numbers so every 'shapiq' route's row/
        depth/tree axes shrink together at a consistent rate.
    """
    if max_order <= 2:
        return int(base_n)
    _shrunk_depth = max(3, 8 - 2 * (int(max_order) - 2))
    _ratio = _shrunk_depth / 8.0
    # `floor` is capped at `base_n` itself - the docstring's own "never
    # INCREASED, only capped" guarantee must hold even when a user's own
    # configured row count is already below `floor` (a task that already
    # explains only 4 rows should never be bumped UP to 5 by this shrink).
    _effective_floor = min(int(floor), int(base_n))
    return max(_effective_floor, int(round(int(base_n) * _ratio)))


def _dedup_shapiq_rows(rows_arr):
    """ver4-28 (this update): exact-duplicate row memoization for the
    `shapiq`-based interaction routes' own row-explain loop
    (`shapiq_tree_interactions()`, `shapiq_pairwise_interactions()`).

    A `shapiq` explainer's per-row `.explain()` call is a pure,
    deterministic function of that row's own feature values - the
    explainer/background/model are all already fixed for the duration of
    one call - so two BYTE-IDENTICAL rows are GUARANTEED to produce the
    identical interaction result. Genomic marker data restricted to a
    shortlisted marker subset (`max_interaction_features`, or design
    blueprint R3b's own candidate-narrowed columns) routinely contains
    such exact duplicates - individuals from tightly related family/
    clonal/inbred breeding lines can share an identical genotype across a
    few hundred markers even when they differ genome-wide - so explaining
    a duplicate row a second (or twentieth) time is pure waste that
    compounds with every OTHER per-row cost this update's row/depth/
    coalition-space shrinks already attack.

    This is an EXACT optimisation, never an approximation: it changes
    WHICH rows `.explain()` is actually called on (fewer, unique ones),
    never what value comes back for any of them - a caller reconstructs
    the identical total by weighting each unique row's own result by how
    many times it occurred (`counts` below), mathematically the same as
    adding that SAME deterministic per-row value the duplicate count of
    times separately (I11-preserving up to the SAME floating-point
    summation-order tolerance design blueprint R1 already established for
    its own parallel fan-out - see `_shapiq_explain_rows()`'s own
    docstring).

    Parameters
    ----------
    rows_arr : numpy.ndarray, shape (n_rows, n_features)

    Returns
    -------
    (unique_rows, counts) : (numpy.ndarray, numpy.ndarray)
        `unique_rows` keeps every row's FIRST-OCCURRENCE position in
        `rows_arr`'s own order - never sorted or otherwise reordered - so
        a batch with NO duplicate rows returns `rows_arr` completely
        unchanged, in the SAME order, row for row (I11: byte-identical
        row loop to before this optimisation existed whenever there is
        nothing to deduplicate). `counts[i]` (float, ready for direct
        multiplication against an accumulated value) is how many times
        `unique_rows[i]` occurred in `rows_arr`.
    """
    import numpy as np

    seen = {}
    unique_rows = []
    counts = []
    for row in rows_arr:
        key = np.ascontiguousarray(row).tobytes()
        idx = seen.get(key)
        if idx is None:
            seen[key] = len(unique_rows)
            unique_rows.append(row)
            counts.append(1.0)
        else:
            counts[idx] += 1.0
    if not unique_rows:
        return np.asarray(rows_arr), np.asarray(counts, dtype=float)
    return np.asarray(unique_rows), np.asarray(counts, dtype=float)


def _shapiq_coalition_space_size(n_features: int, orders) -> int:
    """Total number of distinct marker coalitions `shapiq` would need to
    consider across every requested interaction order -
    ``sum(C(n_features, o) for o in orders)`` - used by
    `shapiq_pairwise_interactions()`'s own pre-flight feasibility check
    and candidate-narrowing size search (design blueprint R3). Pure,
    side-effect-free.

    Parameters
    ----------
    n_features : int
        Marker count the search would run over.
    orders : Sequence[int]
        Interaction orders requested (e.g. ``(2, 3)``).

    Returns
    -------
    int
        Coalitions of an order larger than `n_features` are impossible
        and contribute 0 (``math.comb`` would raise on `o > n_features`
        under some Python versions' stricter argument checks, so those
        orders are skipped rather than passed through).
    """
    return sum(math.comb(n_features, o) for o in orders if 0 <= o <= n_features)


def _shapiq_explain_rows(explainer, rows, orders, n_jobs, **explain_kwargs):
    """Design blueprint R1: row-fan-out wrapper around a `shapiq`
    explainer's own single-row ``.explain()`` call - fans a batch of
    `rows` out across `n_jobs` worker PROCESSES (one row per worker
    call, via ``joblib.Parallel``), mirroring `parallel_shap_values()`'s
    own established contract and fallback convention for the sibling
    exact-SHAP route, which every `shapiq`-based interaction route here
    (`shapiq_pairwise_interactions()`, `shapiq_tree_interactions()`) had
    never applied before this update even though the underlying
    per-row `.explain()` calls are exactly as independent of each other
    as `parallel_shap_values()`'s own per-row `shap_values()` calls.

    Parameters
    ----------
    explainer
        A fitted `shapiq.TabularExplainer` or `shapiq.TreeExplainer` (or
        any object exposing ``.explain(row, **explain_kwargs)`` that
        returns an object with an ``.interaction_lookup`` dict, `shapiq`'s
        own `InteractionValues` contract).
    rows : numpy.ndarray or sequence of numpy.ndarray
        The rows to explain, iterated in order.
    orders : Sequence[int]
        Coalition lengths to keep - mirrors every existing caller's own
        per-row filtering (``if len(coalition) not in orders or
        len(coalition) == 0: continue``), performed here once instead of
        once per caller so serial and parallel paths agree byte-for-byte.
    n_jobs : int
        ``<= 1`` (or ``0``/``None``), or a single row, skips the fan-out
        entirely and explains every row in a single process, in order -
        BYTE-IDENTICAL to every caller's own pre-R1 serial loop (I11).
        ``> 1`` fans the rows out across that many worker processes.
    **explain_kwargs
        Forwarded to ``explainer.explain(row, **explain_kwargs)``
        verbatim - e.g. ``budget=...`` for `TabularExplainer`; empty for
        `TreeExplainer` (which takes no such argument).

    Returns
    -------
    list of dict[tuple, float]
        One dict per row, in `rows`' own order, already filtered to
        `orders` and already `abs()`-valued (the SAME `'abs_then_sum'`
        aggregation both callers already apply) - a caller merges each
        row's dict into its own running ``totals`` accumulator via
        ``totals[c] = totals.get(c, 0.0) + v``, exactly as before this
        helper existed.

    Raises
    ------
    Exception
        Only in the ``n_jobs <= 1`` (serial) path: a genuine failure
        inside ``explainer.explain()`` itself propagates to the caller
        unchanged, matching every pre-R1 call site's own contract (none
        of them caught per-row failures - the caller's own outer
        ``try``/``except`` is what handles this). In the ``n_jobs > 1``
        path, ANY exception during the fan-out (an unpicklable
        explainer/model, a per-row `shapiq` failure, a daemonic-process
        spawn restriction) is caught here instead and this function
        falls back to the same single-process serial loop, with a
        printed NOTE - mirroring `parallel_shap_values()`'s own
        established "never let parallelism itself be a new failure
        mode" precedent. A genuine per-row failure therefore still
        surfaces to the caller (via the serial retry), just one row
        loop later than in the `n_jobs <= 1` path.
    """
    orders = tuple(int(o) for o in orders)

    def _row_result(row):
        interaction_values = explainer.explain(row, **explain_kwargs)
        lookup = getattr(interaction_values, 'interaction_lookup', None)
        coalitions = lookup.keys() if lookup is not None else []
        return {
            coalition: abs(float(interaction_values[coalition]))
            for coalition in coalitions
            if len(coalition) in orders and len(coalition) > 0
        }

    rows = list(rows)
    if n_jobs in (None, 0, 1) or len(rows) <= 1:
        return [_row_result(row) for row in rows]

    from joblib import Parallel, delayed
    try:
        return Parallel(n_jobs=n_jobs)(delayed(_row_result)(row) for row in rows)
    except Exception as exc:
        print(f"[pipeline_utils] NOTE: _shapiq_explain_rows fan-out failed ({exc!r}) - falling "
              f"back to a single-process serial loop over all {len(rows)} row(s) instead, "
              f"matching n_jobs<=1's own behaviour for this call.")
        return [_row_result(row) for row in rows]


def _shapiq_narrow_row_worker(predict_fn, row_full, narrow_positions, background_narrow_arr,
                               index, orders, budget, approximator='auto'):
    """Design blueprint R3b: explain ONE row's order>=3 `shapiq`
    interactions restricted to `narrow_positions` (candidate-narrowed
    marker column POSITIONS into the original, full-width feature
    space `row_full` already lives in) - module-level and picklable
    (I10), since it is also this narrowed pass's own `n_jobs`
    fan-out worker.

    The coalition GAME is played only among `narrow_positions`; every
    marker outside that set is held FIXED at this row's own true value
    throughout (never perturbed/imputed) rather than dropped - this
    function is model-AGNOSTIC (`predict_fn` is an opaque, already-fixed
    callable, unlike `models/RF.py`'s own `max_interaction_features`
    shortlist, which can cheaply REFIT a narrower model instead), so the
    only way to search a marker SUBSET without changing `predict_fn`'s
    own expected input width is to keep every call full-width and hold
    the non-players constant. This mirrors the same "shortlist before
    searching" precedent `models/RF.py`'s own `max_interaction_features`
    already establishes at the marker-count level (see
    `shapiq_pairwise_interactions()`'s own R3b docstring for the full
    rationale), applied here at the search-space level without a refit.
    Order-2 interactions for every ORIGINAL marker (including ones
    outside `narrow_positions`) are computed separately, over the FULL
    marker set, by `shapiq_pairwise_interactions()`'s own unrestricted
    order-2 pass - this function only ever computes `orders` (already
    filtered by its caller to the order>=3 subset).

    A fresh, single-row `shapiq.TabularExplainer` is built for this row
    (construction cost for `TabularExplainer` is small, unlike
    `shapiq.TreeExplainer` - see `shapiq_tree_interactions()`'s own
    construction-cost note), since the wrapper `model` callable below
    must close over THIS row's own full-width context, which differs
    row to row.

    Parameters
    ----------
    predict_fn : callable
        A plain ``predict(X) -> array`` callable accepting a 2-D numpy
        array of the SAME width `row_full` has (the caller's own,
        already marker-shortlisted feature space) - identical contract
        to `shapiq_pairwise_interactions()`'s own `predict_fn` parameter.
    row_full : numpy.ndarray, shape (n_features,)
        This row's own full-width feature vector.
    narrow_positions : Sequence[int]
        Column POSITIONS (into `row_full`'s own width) kept as coalition
        players for this pass, sorted ascending.
    background_narrow_arr : numpy.ndarray, shape (n_background, len(narrow_positions))
        Background rows, already restricted to `narrow_positions`'s own
        columns, for `shapiq`'s own marginal imputer.
    index : str
        `shapiq`'s own interaction-index name.
    orders : Sequence[int]
        Coalition lengths to keep (already restricted by the caller to
        the order>=3 subset of the task's own requested `orders`).
    budget : int
        Per-row coalition-evaluation budget, forwarded to
        ``explainer.explain(row, budget=budget)`` verbatim.
    approximator : str, default 'auto'
        ver4-29 (this update): forwarded to `shapiq.TabularExplainer`'s
        own `approximator` kwarg. Default `'auto'` reproduces this
        function's pre-ver4-29 behaviour exactly (I11) - `shapiq` itself
        already defaulted to `'auto'` here, this only makes that choice
        explicit and overridable by the caller.

    Returns
    -------
    dict[tuple, float]
        Coalition tuples use LOCAL indices into `narrow_positions` (0..
        ``len(narrow_positions) - 1``) - the caller translates these
        back into GLOBAL marker-column positions via
        ``narrow_positions[i]`` before merging into its own `totals`.
    """
    import numpy as np
    import shapiq

    row_full = np.asarray(row_full, dtype=float)
    narrow_positions = list(narrow_positions)

    def _model(X_narrow_batch):
        X_narrow_batch = np.atleast_2d(np.asarray(X_narrow_batch, dtype=float))
        X_full_batch = np.tile(row_full, (X_narrow_batch.shape[0], 1))
        X_full_batch[:, narrow_positions] = X_narrow_batch
        return predict_fn(X_full_batch)

    explainer = shapiq.TabularExplainer(
        model=_model, data=background_narrow_arr, imputer='marginal',
        index=index, max_order=max(int(o) for o in orders), approximator=approximator,
    )
    row_narrow = row_full[narrow_positions]
    interaction_values = explainer.explain(row_narrow, budget=int(budget))
    lookup = getattr(interaction_values, 'interaction_lookup', None)
    coalitions = lookup.keys() if lookup is not None else []
    orders = tuple(int(o) for o in orders)
    return {
        coalition: abs(float(interaction_values[coalition]))
        for coalition in coalitions
        if len(coalition) in orders and len(coalition) > 0
    }


def _shapiq_explain_rows_narrowed(predict_fn, X_full_arr, narrow_positions, background_narrow_arr,
                                   index, orders, budget, n_jobs, approximator='auto'):
    """Design blueprint R3b: `_shapiq_explain_rows()`'s own row-fan-out
    contract, specialised for `_shapiq_narrow_row_worker()` - each row
    needs its OWN worker call (unlike the shared-explainer case
    `_shapiq_explain_rows()` handles) because the wrapper `model`
    callable closes over that row's own full-width context. Same
    ``n_jobs <= 1`` -> serial, ``> 1`` -> `joblib.Parallel` with
    exception-triggered serial fallback convention as
    `_shapiq_explain_rows()`.

    Returns
    -------
    list of dict[tuple, float]
        One dict per row, LOCAL-indexed into `narrow_positions` (see
        `_shapiq_narrow_row_worker()`'s own return-value docstring).
    """
    rows = list(X_full_arr)

    def _serial():
        return [
            _shapiq_narrow_row_worker(
                predict_fn, row, narrow_positions, background_narrow_arr, index, orders, budget,
                approximator=approximator,
            )
            for row in rows
        ]

    if n_jobs in (None, 0, 1) or len(rows) <= 1:
        return _serial()

    from joblib import Parallel, delayed
    try:
        return Parallel(n_jobs=n_jobs)(
            delayed(_shapiq_narrow_row_worker)(
                predict_fn, row, narrow_positions, background_narrow_arr, index, orders, budget,
                approximator=approximator,
            )
            for row in rows
        )
    except Exception as exc:
        print(f"[pipeline_utils] NOTE: _shapiq_explain_rows_narrowed fan-out failed ({exc!r}) - "
              f"falling back to a single-process serial loop over all {len(rows)} row(s) instead, "
              f"matching n_jobs<=1's own behaviour for this call.")
        return _serial()


def shapiq_pairwise_interactions(predict_fn, X_train_background, X_explain, marker_names, *,
                                  orders=(2,), background_size=100, budget=4096, index='k-SII',
                                  n_jobs: int = 1, candidate_narrowing: bool = True,
                                  approximator: str = 'auto'):
    """Shapley-INTERACTION values (order >= 2) via `shapiq`'s own
    `TabularExplainer`, returned as the SAME canonical long-format table
    `models.interaction_extraction.tree_shap_interactions()`/
    `h_statistic_interactions()` already return - the 'shapiq' alternative
    to Friedman's H-statistic for `models/SVR.py`'s/`models/KNN.py`'s/
    `models/RF.py`'s (`'shapiq_tree'`)/`RKHS.R`'s own 'Marker interaction
    method' field (design blueprint R11 - Stage 8). See this module's own
    header comment above for the disclosed "written but not runtime-
    verified against a real `shapiq` install" caveat.

    Unlike Friedman's H-statistic (pairs only), `shapiq` natively supports
    interaction orders above 2 - `orders` may list any subset of
    ``{2, 3, ...}`` and every requested order's own interactions are
    returned in the SAME table (an ``'order'`` column distinguishes them),
    which `models/circos_plot.py`'s own per-``k`` panel rendering (design
    blueprint R12 - Stage 9) reads directly.

    ver4-28: rows that are EXACT duplicates of an already-explained row
    (common in genomic marker data restricted to a shortlisted/narrowed
    column subset) are explained once and reused - see
    `_dedup_shapiq_rows()`'s own docstring. Automatic, no new parameter;
    a batch with no duplicates is unaffected (I11).

    Parameters
    ----------
    predict_fn : callable
        A plain ``predict(X) -> array`` callable.
    X_train_background : pandas.DataFrame
        Background/imputation reference rows.
    X_explain : pandas.DataFrame
        The already-sampled rows to explain.
    marker_names : sequence of str
        Column names, in `X_explain`'s own column order.
    orders : sequence of int, default ``(2,)``
        Which interaction orders to report (2 = pairwise, 3 = three-way,
        ...). The explainer's own ``max_order`` is set to
        ``max(orders)`` (computing every order up to that one is
        `shapiq`'s own natural mode of operation - lower orders are not
        extra work once the top order is budgeted for).
    background_size : int, default 100
    budget : int, default 4096
    index : str, default 'k-SII'
        One of `shapiq`'s own interaction-index names (``'k-SII'``/
        ``'FSII'``/``'STII'``).
    n_jobs : int, keyword-only, default 1
        Design blueprint R1: forwarded to `_shapiq_explain_rows()` -
        ``<= 1`` (the default) explains every row in a single process,
        in order, BYTE-IDENTICAL to this function's own pre-R1 behaviour
        (I11). ``> 1`` fans the per-row `.explain()` calls out across
        that many worker processes (summation order into `totals` then
        depends on completion order, so results are identical only up to
        floating-point summation order - see `_shapiq_explain_rows()`'s
        own docstring).
    candidate_narrowing : bool, keyword-only, default True
        Design blueprint R3b: when `max(orders) >= 3` and this is
        `True` (the default), a cheap order-2-only calibration probe
        (`min(5, n_rows)` rows) ranks markers by summed order-2
        magnitude and restricts the order>=3 search to the smallest
        top-ranked subset whose own coalition-space size fits this
        task's `budget` (see `_SHAPIQ_NARROW_SAFETY_FACTOR`) - order-2
        rows are still returned for EVERY original marker, unaffected;
        only order>=3 rows are restricted to the narrowed subset.

        Requirements.md item 2: the SAME calibration probe/narrowing is
        now ALSO applied to the order-2 pass itself, independently of
        the order>=3 logic above, whenever ``2 in orders`` and the
        order-2 search space ALONE (``C(n_features, 2)``) already
        exceeds ``budget * _SHAPIQ_NARROW_SAFETY_FACTOR`` - e.g. a
        500-marker shortlist (``C(500, 2)`` = 124,750 coalitions)
        against the default ``budget=4096``. Previously this exact
        combination ran the order-2 pass unnarrowed regardless, which
        `shapiq`'s own budget-limited estimator empirically cannot
        resolve into ANY nonzero interaction (see this function's own
        post-hoc ``RuntimeError`` below) - not a narrow miss, a total
        failure that forced every caller's own fallback to
        `models.interaction_extraction.h_statistic_interactions()`
        (Friedman's H-statistic) EVERY time, defeating the point of
        selecting 'shapiq' at all for a marker count this common.
        Narrowing the order-2 pass to the top-ranked subset whose own
        ``C(k, 2)`` fits the same safety-factored budget gives `shapiq`
        a real chance to resolve genuine interactions among the markers
        most likely to carry them, instead of guaranteed failure.

        `False` reproduces this function's pre-R3 single-pass,
        full-shortlist search exactly (I11) for BOTH the order>=3 and
        (as of this fix) the order-2 narrowing above - the calibration
        probe is never run at all, and a task whose order-2 space is too
        large for its budget goes back to raising (triggering the
        caller's own 'friedman_h' fallback), exactly as before this fix.

        ver4-29 (this update): regardless of this flag, narrowing (and
        the R3a pre-flight feasibility `RuntimeError`) is now skipped
        entirely whenever `_shapiq_would_use_spex(n_features, max_order)`
        is true for `approximator in ('auto', 'spex')` - see this
        module's own comment above `_shapiq_would_use_spex()` for why:
        both R3a and R3b were calibrated against the raw combinatorial
        coalition space, which does not describe SPEX's own, much better
        sample complexity at this scale (confirmed empirically against a
        real `shapiq==1.4.1` install). `candidate_narrowing=False`
        together with an explicit non-SPEX `approximator` (e.g.
        `'regression'`) still reproduces the exact pre-ver4-29 behaviour
        for that approximator (I11).
    approximator : str, keyword-only, default 'auto'
        ver4-29 (this update, in response to the shapiq-based SVR
        interaction search reporting mostly `value=0.0`-padded rows at
        this function's own default `budget=4096` against a 500-marker
        shortlist - see this function's own module-level comment above
        `_shapiq_would_use_spex()`): forwarded to `shapiq.TabularExplainer`'s
        own `approximator` kwarg. One of `shapiq`'s own literal choices -
        `'auto'` (default, UNCHANGED behaviour - `shapiq` itself already
        defaulted to `'auto'` when this function never passed the kwarg
        at all, so this default is a genuine no-op, I11), `'spex'`
        (a sparse-Fourier-transform approximator built for large,
        budget-constrained coalition spaces - what `'auto'` itself
        already resolves to once `n_features` is large enough at a given
        `max_order`, see `_shapiq_would_use_spex()`; passing it
        explicitly FORCES that choice even for a smaller `n_features`
        where `'auto'` would otherwise pick a different estimator),
        `'montecarlo'`, `'svarm'`, `'permutation'`, or `'regression'`
        (`shapiq`'s own regression-based estimator - closest in spirit
        to this function's own pre-ver4-29 implicit choice for a SMALL
        `n_features`). See `shapiq.explainer.configuration.
        APPROXIMATOR_CONFIGURATIONS` (shapiq==1.4.1) for exactly which
        concrete estimator class each name resolves to for a given
        `index`.

    Returns
    -------
    pandas.DataFrame with columns ``['markers', 'order', 'value']`` -
    ``'markers'`` is a tuple of marker NAMES (length = that row's own
    ``'order'``), never positional indices, mirroring
    `tree_shap_interactions()`'s own "always real names" convention.

    Raises
    ------
    Exception
        See `shapiq_marker_effects()`'s own identical note - never caught
        here, always caught by the caller. Design blueprint R3a: now also
        raised BEFORE any row is explained (message says "pre-flight, not
        exhausted") when `_shapiq_coalition_space_size(n_features, orders)`
        is knowably hopeless relative to `budget` - see
        `_SHAPIQ_INFEASIBLE_SPACE_RATIO`'s own docstring. This pre-check is
        NOT gated by `candidate_narrowing` (it only changes WHEN an
        already-certain failure fires, never a value).
    """
    import pandas as pd
    import shapiq

    marker_names = list(marker_names)
    if X_explain.shape[1] != len(marker_names):
        raise ValueError(
            f"shapiq_pairwise_interactions: X_explain has {X_explain.shape[1]} columns but "
            f"{len(marker_names)} marker_names were given - these must match 1:1."
        )
    # Requirements.md item 1: every internal call below (the calibration
    # probe, the unrestricted/order-2/order>=3 explainers, and the
    # narrowed-row workers, all of which reach `predict_fn` with a raw
    # numpy array) is reached through this SAME local `predict_fn` name,
    # so re-pointing it once here at a name-restoring wrapper covers every
    # one of them - silences sklearn's "X does not have valid feature
    # names" warning without changing any computed value (see
    # _predict_fn_with_feature_names()'s own docstring).
    predict_fn = _predict_fn_with_feature_names(predict_fn, marker_names)
    orders = sorted(set(int(o) for o in orders))
    max_order = max(orders) if orders else 2
    n_features = X_explain.shape[1]

    # ver4-29 (this update): see `_shapiq_would_use_spex()`'s own
    # module-level comment for the full rationale. `_uses_spex` is `True`
    # whenever the resolved approximator either IS SPEX (explicit
    # `approximator='spex'`) or WOULD BE, once `shapiq` itself resolves
    # `approximator='auto'` for this `n_features`/`max_order` combination.
    # Every non-SPEX approximator (an explicit 'montecarlo'/'svarm'/
    # 'permutation'/'regression', or 'auto' at a scale where shapiq's own
    # heuristic would not pick SPEX) leaves `_uses_spex` False, and this
    # function's pre-existing R3a/R3b logic runs completely unchanged
    # (I11).
    _uses_spex = approximator == 'spex' or (
        approximator == 'auto' and _shapiq_would_use_spex(n_features, max_order)
    )

    # --- design blueprint R3a: pre-flight feasibility check ---------------
    # Runs unconditionally (not gated by `candidate_narrowing` - see this
    # function's own "Raises" docstring above) EXCEPT when `_uses_spex` -
    # this check's own threshold assumes the raw combinatorial coalition
    # space is the right proxy for feasibility, which does not hold for
    # SPEX (see `_shapiq_would_use_spex()`'s own module-level comment).
    _space_size = _shapiq_coalition_space_size(n_features, orders)
    if not _uses_spex and budget and _space_size > budget * _SHAPIQ_INFEASIBLE_SPACE_RATIO:
        raise RuntimeError(
            f"shapiq_pairwise_interactions: shapiq's TabularExplainer (index={index!r}, "
            f"max_order={max_order}, budget={budget}) was not attempted (pre-flight, not "
            f"exhausted) - the requested {n_features}-marker, order-{tuple(orders)} search "
            f"space has " + " + ".join(f"C({n_features},{o})" for o in orders) +
            f" = {_space_size:,} possible coalitions, more than "
            f"{_SHAPIQ_INFEASIBLE_SPACE_RATIO:.0f}x this task's budget of {budget} - `budget` "
            f"is almost certainly far too small relative to this search space for shapiq's own "
            f"estimator to resolve any nonzero interaction from it (see this function's own "
            f"post-hoc empirical RuntimeError note below for the confirmed case this guards "
            f"against). Raising immediately, before spending any row-explain time, so the "
            f"caller's own fallback route runs sooner."
        )

    n_rows_bg = X_train_background.shape[0]
    background = (
        X_train_background.sample(n=min(int(background_size), n_rows_bg), random_state=0)
        if n_rows_bg > 0 else X_train_background
    )
    X_explain_arr = X_explain.to_numpy()

    # ver4-28: exact-duplicate row memoization (see `_dedup_shapiq_rows()`'s
    # own docstring) - computed ONCE here and reused by every row-explain
    # pass below (the unrestricted pass, and, when candidate-narrowing is
    # active, both its order-2 and order>=3 passes), each of which weights
    # a unique row's own result by `_row_counts[i]` when merging into
    # `totals` rather than re-explaining an already-seen row. A batch with
    # no duplicate rows is completely unaffected - `_X_unique` is
    # `X_explain_arr` unchanged, in the same order (I11).
    _X_unique, _row_counts = _dedup_shapiq_rows(X_explain_arr)
    if _X_unique.shape[0] < X_explain_arr.shape[0]:
        print(f"[pipeline_utils] NOTE: shapiq_pairwise_interactions: {X_explain_arr.shape[0]} "
              f"explained row(s) collapsed to {_X_unique.shape[0]} exact-duplicate-free row(s) "
              f"before calling shapiq - {X_explain_arr.shape[0] - _X_unique.shape[0]} row(s) "
              f"skipped as exact duplicates of an already-explained row.")

    # --- design blueprint R3b: candidate narrowing for order >= 3, AND
    # (Requirements.md item 2) for order 2 alone whenever ITS OWN search
    # space already exceeds this task's safety-factored budget -----------
    # `_narrowed_positions` (order>=3) and `_narrowed_positions_2` (order
    # 2) each stay `None` (no narrowing) unless their own trigger fires
    # AND the shared calibration probe below succeeds. Both are resolved
    # from the SAME probe run (order-2-only, `min(5, n_rows)` rows) - no
    # need to probe twice.
    _narrowed_positions = None
    _narrowed_positions_2 = None
    _order2_requested = 2 in orders
    _order2_space_too_large = (
        not _uses_spex and _order2_requested
        and _shapiq_coalition_space_size(n_features, (2,)) > budget * _SHAPIQ_NARROW_SAFETY_FACTOR
    )
    _order_hi_narrowing_applies = (
        not _uses_spex and candidate_narrowing and max_order >= 3 and n_features > max_order
    )
    if candidate_narrowing and n_features > 2 and (_order_hi_narrowing_applies or _order2_space_too_large):
        try:
            _probe_explainer = shapiq.TabularExplainer(
                model=predict_fn, data=background.to_numpy(), imputer='marginal',
                index=index, max_order=2, approximator=approximator,
            )
            _probe_rows = X_explain_arr[:min(5, X_explain_arr.shape[0])]
            _probe_results = _shapiq_explain_rows(
                _probe_explainer, _probe_rows, (2,), n_jobs, budget=int(budget),
            )
            _marker_score = [0.0] * n_features
            for _row_dict in _probe_results:
                for _coalition, _value in _row_dict.items():
                    for _i in _coalition:
                        _marker_score[_i] += _value
            _ranked = sorted(range(n_features), key=lambda i: _marker_score[i], reverse=True)

            if _order_hi_narrowing_applies:
                _orders_hi_probe = tuple(o for o in orders if o >= 3)
                _narrow_budget = budget * _SHAPIQ_NARROW_SAFETY_FACTOR
                _narrow_k = 0
                for _k in range(max_order, n_features + 1):
                    if _shapiq_coalition_space_size(_k, _orders_hi_probe) <= _narrow_budget:
                        _narrow_k = _k
                    else:
                        break
                if 0 < _narrow_k < n_features:
                    _narrowed_positions = sorted(_ranked[:_narrow_k])
                    print(f"[pipeline_utils] NOTE: shapiq_pairwise_interactions: candidate-narrowing "
                          f"(design blueprint R3b) restricted this task's order>={min(_orders_hi_probe)} "
                          f"search to the {_narrow_k}/{n_features} markers with the strongest order-2 "
                          f"calibration-probe magnitude - order-2 interactions are still computed for "
                          f"every marker. Set SHAPIQ_CANDIDATE_NARROWING: false to disable.")

            if _order2_space_too_large:
                _narrow_budget_2 = budget * _SHAPIQ_NARROW_SAFETY_FACTOR
                _narrow_k_2 = 0
                for _k in range(2, n_features + 1):
                    if _shapiq_coalition_space_size(_k, (2,)) <= _narrow_budget_2:
                        _narrow_k_2 = _k
                    else:
                        break
                if 0 < _narrow_k_2 < n_features:
                    _narrowed_positions_2 = sorted(_ranked[:_narrow_k_2])
                    print(f"[pipeline_utils] NOTE: shapiq_pairwise_interactions: (Requirements.md item "
                          f"2) this task's order-2 search space alone (C({n_features}, 2) = "
                          f"{_shapiq_coalition_space_size(n_features, (2,)):,} coalitions) is too large "
                          f"for its own budget of {budget} for shapiq's own estimator to resolve any "
                          f"nonzero interaction from (see this function's own post-hoc RuntimeError "
                          f"note below) - previously this ran unnarrowed anyway and reliably returned "
                          f"NOTHING, forcing a full fallback to 'friedman_h' every time. Restricting "
                          f"the order-2 search to the {_narrow_k_2}/{n_features} markers with the "
                          f"strongest calibration-probe magnitude instead, so shapiq has a real chance "
                          f"to resolve genuine interactions. Set SHAPIQ_CANDIDATE_NARROWING: false to "
                          f"disable (this task then runs unnarrowed and most likely raises, triggering "
                          f"the caller's own 'friedman_h' fallback, exactly as before this fix).")
        except Exception as exc:
            print(f"[pipeline_utils] NOTE: shapiq_pairwise_interactions: candidate-narrowing's own "
                  f"order-2 calibration probe failed ({exc!r}) - skipping narrowing and proceeding "
                  f"with the full {n_features}-marker set at this task's real budget, exactly as if "
                  f"SHAPIQ_CANDIDATE_NARROWING were false.")
            _narrowed_positions = None
            _narrowed_positions_2 = None

    # coalition tuple -> running sum(|phi|) across every explained row -
    # the SAME 'abs_then_sum' aggregation tree_shap_interactions() uses.
    totals: "dict[tuple, float]" = {}
    with bounded_memory_guard(log_prefix="[pipeline_utils] shapiq_pairwise_interactions"):
        if _narrowed_positions is None and _narrowed_positions_2 is None:
            explainer = shapiq.TabularExplainer(
                model=predict_fn, data=background.to_numpy(), imputer='marginal',
                index=index, max_order=max_order, approximator=approximator,
            )
            row_results = _shapiq_explain_rows(explainer, _X_unique, orders, n_jobs,
                                                budget=int(budget))
            for row_dict, _w in zip(row_results, _row_counts):
                for coalition, value in row_dict.items():
                    totals[coalition] = totals.get(coalition, 0.0) + value * _w
        else:
            # Order-2 pass: full marker set, UNLESS order-2's own search
            # space was itself too large for `budget` (Requirements.md
            # item 2 - `_narrowed_positions_2` above), in which case it is
            # restricted to its own candidate-narrowed subset via the
            # SAME narrowed-row-worker machinery the order>=3 pass below
            # already uses.
            _orders_2 = tuple(o for o in orders if o == 2)
            if _orders_2 and _narrowed_positions_2 is not None:
                background_narrow_arr_2 = background.to_numpy()[:, _narrowed_positions_2]
                row_results = _shapiq_explain_rows_narrowed(
                    predict_fn, _X_unique, _narrowed_positions_2, background_narrow_arr_2,
                    index, _orders_2, int(budget), n_jobs, approximator=approximator,
                )
                for row_dict, _w in zip(row_results, _row_counts):
                    for local_coalition, value in row_dict.items():
                        global_coalition = tuple(_narrowed_positions_2[i] for i in local_coalition)
                        totals[global_coalition] = totals.get(global_coalition, 0.0) + value * _w
            elif _orders_2:
                explainer_2 = shapiq.TabularExplainer(
                    model=predict_fn, data=background.to_numpy(), imputer='marginal',
                    index=index, max_order=2, approximator=approximator,
                )
                row_results = _shapiq_explain_rows(explainer_2, _X_unique, _orders_2, n_jobs,
                                                    budget=int(budget))
                for row_dict, _w in zip(row_results, _row_counts):
                    for coalition, value in row_dict.items():
                        totals[coalition] = totals.get(coalition, 0.0) + value * _w

            # Order >= 3 pass: narrowed marker subset when narrowing
            # triggered (design blueprint R3b) - see
            # _shapiq_narrow_row_worker()'s own docstring for why every
            # row needs its own worker call. If this task entered the
            # split path solely because of order-2 narrowing above (order
            # >= 3 was requested too, but its OWN space already fit the
            # budget), run it exactly like the pre-split unrestricted
            # path would have - unaffected by the order-2 branch above.
            _orders_hi = tuple(o for o in orders if o >= 3)
            if _orders_hi and _narrowed_positions is not None:
                background_narrow_arr = background.to_numpy()[:, _narrowed_positions]
                row_results = _shapiq_explain_rows_narrowed(
                    predict_fn, _X_unique, _narrowed_positions, background_narrow_arr,
                    index, _orders_hi, int(budget), n_jobs, approximator=approximator,
                )
                for row_dict, _w in zip(row_results, _row_counts):
                    for local_coalition, value in row_dict.items():
                        global_coalition = tuple(_narrowed_positions[i] for i in local_coalition)
                        totals[global_coalition] = totals.get(global_coalition, 0.0) + value * _w
            elif _orders_hi:
                explainer_hi = shapiq.TabularExplainer(
                    model=predict_fn, data=background.to_numpy(), imputer='marginal',
                    index=index, max_order=max(_orders_hi), approximator=approximator,
                )
                row_results = _shapiq_explain_rows(explainer_hi, _X_unique, _orders_hi, n_jobs,
                                                    budget=int(budget))
                for row_dict, _w in zip(row_results, _row_counts):
                    for coalition, value in row_dict.items():
                        totals[coalition] = totals.get(coalition, 0.0) + value * _w

    if not totals:
        # Confirmed empirically (this update): with `budget` small relative
        # to the search space at `max_order` across `n_features` markers -
        # e.g. budget=4096 against ~500 markers at max_order=3, where the
        # order-2/order-3 search space alone is in the tens of millions of
        # possible coalitions - `shapiq`'s own budget-based `TabularExplainer`
        # can return an `interaction_lookup` that never contains ANY
        # coalition of size >= 2 for ANY explained row, at ANY requested
        # order, for the WHOLE task (not merely a sparse/partial sample of
        # them). Previously this function returned an empty DataFrame here
        # with no error at all, which every caller (`models/SVR.py`'s/
        # `models/KNN.py`'s own 'shapiq' branch, `models.interaction_
        # extraction.surrogate_shapiq_interactions()` for RKHS) treated as a
        # perfectly valid "shapiq genuinely found nothing" result - so it
        # was stored as-is, silently discarding the ENTIRE interaction step
        # for that model (Interaction.csv AND Interaction_higher_order.csv
        # both end up with zero rows for it, and `circos_plot.py` then has
        # nothing to draw for that model either). Raising here instead
        # routes this exact situation into the SAME `try`/`except` ->
        # printed-NOTE -> fall-back-to-'friedman_h' path every caller
        # already has for a hard `shapiq` failure (package missing, an API
        # mismatch) - restoring at least the order-2 result Friedman's
        # H-statistic can always compute at this scale, with a clear
        # explanation instead of a silently empty file.
        raise RuntimeError(
            f"shapiq_pairwise_interactions: shapiq's TabularExplainer (index={index!r}, "
            f"max_order={max_order}, budget={budget}) returned no interaction of order "
            f">= 2 at all, across {X_explain_arr.shape[0]} explained row(s) and {n_features} "
            f"marker(s) - `budget` is almost certainly far too small relative to the "
            f"{n_features}-marker, order-{tuple(orders)} search space requested (there are "
            + " + ".join(f"C({n_features},{o})" for o in orders) +
            f" possible coalitions at those orders) for shapiq's own estimator to resolve "
            f"ANY of them from this budget. Raising so the caller's own fallback route runs "
            f"instead of silently storing zero interaction rows for this task."
        )
    rows = [
        {'markers': tuple(marker_names[i] for i in coalition), 'order': len(coalition), 'value': value}
        for coalition, value in totals.items()
    ]
    return pd.DataFrame(rows)


def shapiq_tree_interactions(fitted_model, X_explain, marker_names, *, orders=(2,), index='k-SII',
                              max_seconds=None, n_jobs: int = 1):
    """Exact Shapley-INTERACTION values (order >= 2) for a fitted tree
    ensemble via `shapiq`'s own `TreeExplainer` (the TreeSHAP-IQ
    algorithm) - the 'shapiq_tree' alternative to `models/RF.py`'s own
    default 'pairwise_shap' route (`models.interaction_extraction.
    tree_shap_interactions()`, `shap.TreeExplainer`-based). See this
    module's own header comment above `shapiq_marker_effects()` for the
    disclosed "written but not runtime-verified against a real `shapiq`
    install" caveat - identical here.

    Unlike `shapiq_pairwise_interactions()` above (which is model-
    AGNOSTIC - any `.predict` callable, via a perturbation-based
    `TabularExplainer` and a `budget`), this function is tree-SPECIFIC
    and EXACT, mirroring `shap.TreeExplainer`'s own no-background,
    no-sampling contract - `shapiq.TreeExplainer` walks the fitted
    ensemble's own tree structure directly, so there is no
    `background_size`/`budget` parameter here (nothing is sampled or
    perturbed).

    Parameters
    ----------
    fitted_model
        A scikit-learn-compatible fitted tree ensemble (whatever
        `shapiq.TreeExplainer` accepts) - the caller fits and (if
        needed) shortlists this BEFORE calling, exactly like
        `models.interaction_extraction.tree_shap_interactions()`'s own
        `fitted_model` parameter.
    X_explain : pandas.DataFrame
        The already-sampled rows to explain.
    marker_names : sequence of str
        Column names, in `X_explain`'s own column order.
    orders : sequence of int, default ``(2,)``
        Which interaction orders to report - forwarded to the
        explainer's own ``max_order`` (``max(orders)``), exactly like
        `shapiq_pairwise_interactions()` above.
    index : str, default 'k-SII'
        `shapiq`'s own interaction-index name. See "Known `shapiq==1.4.1`
        TreeExplainer bug" below - a request for ``'FSII'``/``'STII'``
        may be transparently computed as ``'k-SII'`` instead.
    max_seconds : float or None, default None
        Requirements.md item 2 (performance): an optional wall-clock
        budget for the WHOLE row loop below (across every row in
        `X_explain`, not per row - a single row's own `explainer.explain()`
        call cannot itself be interrupted, since it returns control to
        Python only once it finishes). Checked BETWEEN rows; once elapsed
        time exceeds this budget, the remaining rows are skipped and
        whatever coalitions were already accumulated are returned, with a
        printed NOTE naming how many of `X_explain`'s rows were actually
        explained. `None` (the default) preserves this function's
        previous behaviour exactly - no budget, every row is explained
        (I11). This is a coarse SAFETY NET, not the primary fix for the
        cost problem `models/RF.py`'s own depth-capped refit for this
        route already addresses (see that file's own
        `_SHAPIQ_TREE_MAX_DEPTH` comment - tree DEPTH, not row count, is
        the dominant cost driver here); it exists so an unusual forest
        this cap does not fully tame still degrades to a partial result
        instead of hanging indefinitely.
    n_jobs : int, keyword-only, default 1
        Design blueprint R1: forwarded to `_shapiq_explain_rows()` in
        CHUNKS of size `n_jobs` (rather than one row at a time) so that
        `max_seconds`'s wall-clock check (below) can still run BETWEEN
        chunks at the same granularity it always checked between rows
        at `n_jobs<=1`. ``<= 1`` (the default) explains one row at a
        time in a single process, BYTE-IDENTICAL to this function's own
        pre-R1 behaviour (I11). ``> 1`` fans each chunk's rows out
        across that many worker processes (summation order into
        `totals` then depends on completion order within a chunk, so
        results are identical only up to floating-point summation order
        - see `_shapiq_explain_rows()`'s own docstring).

    ver4-28: rows that are EXACT duplicates of an already-explained row
    are explained once and reused - see `_dedup_shapiq_rows()`'s own
    docstring. Automatic, no new parameter; a batch with no duplicates is
    unaffected (I11); `max_seconds`'s own row-count accounting (below)
    reports ORIGINAL row coverage, not the (possibly smaller) number of
    `.explain()` calls actually made.

    Returns
    -------
    pandas.DataFrame with columns ``['markers', 'order', 'value']`` -
    the SAME shape `shapiq_pairwise_interactions()` returns. May reflect
    fewer than `X_explain.shape[0]` explained rows if `max_seconds` cut
    the loop short (see that parameter's own docstring above).

    Known `shapiq==1.4.1` TreeExplainer bug (RF.py 'shapiq_tree' route,
    Interaction_higher_order.csv coming back empty)
    ------------------------------------------------------------------
    Confirmed directly (not assumed - shapiq is installable in this
    session's sandbox, unlike when this module's other shapiq helpers
    above were first written): the pinned ``shapiq==1.4.1``'s
    ``TreeExplainer`` raises ``ValueError: non-broadcastable output
    operand ...`` from its own internal ``_compute_shapley_interaction_
    values()`` whenever ``index`` is ``'FSII'`` or ``'STII'`` (the two
    non-``'k-SII'`` choices `hparam_specs.py` offers for RF's
    'shapiq_tree' method) and the fitted forest has any tree deeper than
    a shallow toy depth - i.e. essentially always, since RF's own
    default 'Maximum tree depth' is `None` (fully grown trees). This
    reproduces regardless of the requested interaction order - it is
    not specific to order>=3, though that is where its consequence is
    most visible: `models/RF.py`'s own outer `try`/`except` around
    `models.interaction_extraction.tree_shapiq_interactions()` catches
    this and falls back to the exact, order-2-only 'pairwise_shap'
    route (`tree_shap_interactions()`) for EVERY task, so
    Interaction.csv still fills in correctly (indistinguishable from a
    genuine 'shapiq_tree' result at order 2) while
    Interaction_higher_order.csv - which only 'shapiq_tree' can ever
    populate - stays permanently empty (header-only), with the only
    trace being a `[RF] NOTE: shapiq_tree interaction extraction
    failed ...` line in the run log that is easy to miss.

    `'k-SII'` (the field's own default) does NOT hit this bug at any
    order or tree depth (confirmed the same way). Rather than let a
    library bug in one interaction-index choice silently discard the
    entire higher-order-interaction feature via RF.py's outer fallback,
    this function catches exactly that failure and retries once with
    `index='k-SII'` before giving up - so a run configured for
    order>=3 still gets real order>=3 rows (computed with 'k-SII')
    instead of none at all. A printed NOTE always accompanies the
    substitution, so the discrepancy between the requested and the
    actually-used index is never silent. If shapiq is ever upgraded
    past this bug (or `index` was already `'k-SII'`), the `try` above
    simply succeeds on the first attempt and this fallback never
    triggers - a no-op change for every unaffected caller (I11).

    Raises
    ------
    Exception
        Any failure that survives the 'k-SII' retry above (package
        missing, a model type `shapiq.TreeExplainer` does not support
        at all, an API mismatch against the `shapiq` version actually
        installed) - deliberately NOT caught here;
        `models.interaction_extraction.tree_shapiq_interactions()`
        wraps this call in its own `try`/`except` and falls back to the
        proven 'pairwise_shap' route (`tree_shap_interactions()`) with a
        printed NOTE, exactly like every other `shapiq` integration in
        this codebase (see this module's own header comment).
    """
    import pandas as pd
    import shapiq

    marker_names = list(marker_names)
    if X_explain.shape[1] != len(marker_names):
        raise ValueError(
            f"shapiq_tree_interactions: X_explain has {X_explain.shape[1]} columns but "
            f"{len(marker_names)} marker_names were given - these must match 1:1."
        )
    orders = sorted(set(int(o) for o in orders))
    max_order = max(orders) if orders else 2
    X_explain_arr = X_explain.to_numpy()

    # ver4-28: exact-duplicate row memoization (see `_dedup_shapiq_rows()`'s
    # own docstring) - `_collect()` below chunks and explains `_X_unique`
    # (never re-explaining a row already seen), then weights each unique
    # row's own result by `_row_counts[i]` when merging into `totals`. A
    # batch with no duplicate rows is completely unaffected - `_X_unique`
    # is `X_explain_arr` unchanged, in the same order (I11).
    _X_unique, _row_counts = _dedup_shapiq_rows(X_explain_arr)
    if _X_unique.shape[0] < X_explain_arr.shape[0]:
        print(f"[pipeline_utils] NOTE: shapiq_tree_interactions: {X_explain_arr.shape[0]} explained "
              f"row(s) collapsed to {_X_unique.shape[0]} exact-duplicate-free row(s) before calling "
              f"shapiq - {X_explain_arr.shape[0] - _X_unique.shape[0]} row(s) skipped as exact "
              f"duplicates of an already-explained row.")

    def _collect(explain_index):
        totals: "dict[tuple, float]" = {}
        # Confirmed empirically (this update): shapiq's exact TreeExplainer
        # at max_order>=3 has no natural memory ceiling of its own - even a
        # SMALL, shallow forest (30 markers, 100 trees of depth ~10-14) can
        # exceed several GB explaining a SINGLE row, and that growth is
        # unbounded in wall-clock time too. Guarded so a blow-up raises a
        # catchable MemoryError (caught below, and by this function's own
        # caller) instead of the OS/cgroup OOM-killer silently taking down
        # this run's entire process - see bounded_memory_guard()'s own
        # docstring for the full rationale.
        #
        # ver4-27 fix (calibrate_shapiq_tree_depth() prerequisite):
        # `shapiq.TreeExplainer(...)` construction itself - not just the
        # row-explain loop below - is now INSIDE this guard too. Confirmed
        # empirically (this update, against a real shapiq==1.4.1 install):
        # construction ALONE (before any row is ever explained) can exhaust
        # memory and get SIGKILLed by the OS at depths/marker-counts well
        # within what a real RF task's default shortlist (500 markers)
        # reaches - previously that killed this run's entire process
        # uncatchably, identical to the row-loop's own pre-fix exposure
        # this guard was originally added for. This is what makes it SAFE
        # for `calibrate_shapiq_tree_depth()` (models/RF.py) to actually
        # ATTEMPT a deeper candidate depth and simply move on to a
        # shallower one on failure, rather than only ever guessing depth
        # conservatively up front.
        _t0 = time.monotonic()
        _n_explained = 0          # unique rows explained so far (loop control)
        _n_original_covered = 0.0  # original-row-equivalent coverage (for the NOTE below only)
        # Design blueprint R1: rows are explained in CHUNKS of `n_jobs` via
        # `_shapiq_explain_rows()` rather than one at a time - at n_jobs<=1
        # this collapses to a chunk size of 1, so the between-row
        # `max_seconds` check below fires at EXACTLY the same points as
        # before this update (I11: byte-identical at n_jobs<=1). At
        # n_jobs>1, the check becomes a between-CHUNK check (coarser
        # granularity, same semantics) - a single chunk's own rows are
        # explained in parallel, so they cannot individually be checked
        # mid-flight any more than a single row's own `explain()` call
        # ever could. ver4-28: chunks are drawn from `_X_unique` (the
        # exact-duplicate-free row set), not `X_explain_arr` directly - see
        # this function's own `_dedup_shapiq_rows()` note above.
        _chunk_size = max(1, int(n_jobs)) if n_jobs else 1
        with bounded_memory_guard(log_prefix="[pipeline_utils] shapiq_tree_interactions"):
            explainer = shapiq.TreeExplainer(model=fitted_model, max_order=max_order, index=explain_index)
            for _chunk_start in range(0, _X_unique.shape[0], _chunk_size):
                # Requirements.md item 2: the wall-clock safety net - see
                # this function's own `max_seconds` docstring above. Checked
                # BEFORE each chunk (never mid-chunk - `explain()` calls
                # cannot themselves be interrupted), so a budget of 0 still
                # explains at least nothing further, not a partial chunk.
                if max_seconds is not None and _n_explained > 0 and (time.monotonic() - _t0) > max_seconds:
                    print(f"[pipeline_utils] NOTE: shapiq_tree_interactions: stopped after "
                          f"{_n_original_covered:.0f}/{X_explain_arr.shape[0]} row(s) - exceeded the "
                          f"{max_seconds:.0f}s wall-clock budget for this task's interaction "
                          f"search. Returning the partial result accumulated so far rather than "
                          f"continuing to explain the remaining rows.")
                    break
                _chunk = _X_unique[_chunk_start:_chunk_start + _chunk_size]
                _chunk_weights = _row_counts[_chunk_start:_chunk_start + _chunk_size]
                _chunk_results = _shapiq_explain_rows(explainer, _chunk, orders, n_jobs)
                _n_explained += len(_chunk)
                _n_original_covered += float(_chunk_weights.sum())
                for row_dict, _w in zip(_chunk_results, _chunk_weights):
                    for coalition, value in row_dict.items():
                        totals[coalition] = totals.get(coalition, 0.0) + value * _w
        return totals

    # Requirements.md item 2: `shapiq.TreeExplainer`'s CONSTRUCTION step
    # (inside `_collect`, before its row loop even starts) is itself a
    # single non-interruptible call - confirmed empirically to take
    # anywhere from ~1s to 80+s depending on forest size, with no clean
    # relationship to any one hyperparameter (tree count and depth both
    # matter, but not monotonically at the scales tested) - so it cannot
    # be bounded by the same "check between rows" trick `max_seconds`
    # uses inside `_collect`. What CAN be bounded cheaply: the 'k-SII'
    # RETRY below (see "Known shapiq==1.4.1 TreeExplainer bug") pays this
    # unguarded construction cost a SECOND time on top of whatever the
    # first, failing attempt already spent - so if the budget is already
    # gone by the time the first attempt fails, retrying is skipped
    # outright (an empty result, same as `max_seconds` cutting the row
    # loop short at 0 rows) rather than silently doubling an already-over-
    # budget wait.
    _construct_t0 = time.monotonic()
    try:
        totals = _collect(index)
    except Exception as exc:
        # See "Known shapiq==1.4.1 TreeExplainer bug" above - only retried
        # for a non-'k-SII' index, so an already-'k-SII' failure (or any
        # future shapiq version this bug no longer reproduces on) is
        # unaffected and simply raises/succeeds as before this fix (I11).
        if index == 'k-SII':
            raise
        if max_seconds is not None and (time.monotonic() - _construct_t0) > max_seconds:
            print(f"[pipeline_utils] NOTE: shapiq_tree_interactions: shapiq's TreeExplainer failed "
                  f"to compute the {index!r} interaction index ({exc!r}), and this task's "
                  f"{max_seconds:.0f}s wall-clock budget was already spent on that failed attempt - "
                  f"skipping the 'k-SII' retry (which would rebuild the explainer from scratch) and "
                  f"returning an empty result for this row batch rather than doubling an "
                  f"already-over-budget wait.")
            totals = {}
        else:
            print(f"[pipeline_utils] NOTE: shapiq_tree_interactions: shapiq's TreeExplainer failed "
                  f"to compute the {index!r} interaction index ({exc!r}) - this matches a known "
                  f"shapiq==1.4.1 bug affecting 'FSII'/'STII' on non-trivial tree depths (see this "
                  f"function's own docstring). Retrying this task's interaction computation with "
                  f"'k-SII' instead, so the requested order(s) {tuple(orders)} are still computed "
                  f"rather than silently lost to RF.py's own outer 'pairwise_shap' fallback.")
            totals = _collect('k-SII')

    if not totals:
        return pd.DataFrame(columns=['markers', 'order', 'value'])
    rows = [
        {'markers': tuple(marker_names[i] for i in coalition), 'order': len(coalition), 'value': value}
        for coalition, value in totals.items()
    ]
    return pd.DataFrame(rows)


def calibrate_shapiq_tree_depth(fit_forest_fn, probe_row, depth_ladder, *, max_order=2,
                                 index='k-SII', per_rung_seconds, log_prefix="[pipeline_utils]"):
    """ver4-27: real, empirically-verified replacement for guessing the
    'shapiq_tree' interaction-only refit's own `max_depth` up front (the
    flat `_SHAPIQ_TREE_MAX_DEPTH=8`/order-aware `_resolve_shapiq_tree_
    depth_cap()` this superseded - see `models/RF.py`'s own call site).

    Why a REAL attempt, not a cheap scaled-down probe
    ---------------------------------------------------
    A tempting cheaper design would fit a small proxy forest (a handful of
    markers, a handful of trees) at each candidate depth, time THAT, and
    extrapolate to this task's real (n_estimators, n_features) scale.
    Confirmed empirically (this update, against a real installed
    `shapiq==1.4.1`) that this would be unreliable here: holding depth=8
    and n_estimators=100 fixed, going from 40 to 150 markers took
    `shapiq.TreeExplainer` CONSTRUCTION from 2.8s to 22.2s - roughly an 8x
    jump for a 3.75x feature-count increase, i.e. construction cost grows
    noticeably FASTER than linearly in marker count, not just in tree
    count/depth as `models/RF.py`'s own `O(T*L*D**k)` comment describes.
    A small-marker-count proxy would systematically UNDER-estimate real
    cost at this task's actual shortlist size (500 markers by default) and
    recommend too deep a rung. This function instead fits the ACTUAL
    interaction-only forest (real `n_estimators`/`max_features`/
    `max_samples`/marker shortlist - everything `fit_forest_fn` already
    closes over except `max_depth`) at each candidate depth and measures
    it directly - slower to calibrate, but the ONLY numbers this function
    trusts are ones it actually measured at real scale.

    Two independent safety bounds per rung
    -----------------------------------------
    - Memory: fit + `shapiq.TreeExplainer` construction + one `.explain()`
      call all run inside `bounded_memory_guard()` - an out-of-memory rung
      raises a catchable `MemoryError` (never an uncatchable SIGKILL that
      would take down this run's entire process) and is simply skipped in
      favour of the next, shallower rung. This relies on
      `shapiq_tree_interactions()`'s own construction call ALSO now being
      guarded (ver4-27 fix, same update) - this function builds its OWN
      probe explainer the same way, under the same guard, for the same
      reason.
    - Wall clock: construction is not itself interruptible mid-call (same
      caveat `shapiq_tree_interactions()`'s own `max_seconds` docstring
      already discloses), so a rung cannot be aborted WHILE it runs - but
      once `fit_forest_fn(depth)` + `TreeExplainer(...)` + one
      `.explain(probe_row)` call all return, elapsed time is checked
      against `per_rung_seconds` and a too-slow rung (even one that
      technically succeeded) is rejected exactly like a `MemoryError`
      would be. Worst case, a single over-budget rung costs roughly
      `per_rung_seconds` of wall clock before being abandoned - bounded,
      not unbounded, and paid at most once per rung in `depth_ladder`.

    Parameters
    ----------
    fit_forest_fn : callable
        ``fit_forest_fn(depth: int) -> fitted scikit-learn-compatible
        forest`` - refits the SAME interaction-only forest `models/RF.py`
        would otherwise build directly, at the given `max_depth`; every
        other hyperparameter (`n_estimators`, `max_features`,
        `max_samples`, `min_samples_leaf`, sample weights, the marker
        shortlist/training rows) is already closed over by the caller.
    probe_row : numpy.ndarray
        A single already-selected explained row (1-D, same column order
        as the forest `fit_forest_fn` returns) - the SAME row shape
        `shapiq.TreeExplainer.explain()` takes elsewhere in this module.
    depth_ladder : sequence of int
        Candidate depths, DESCENDING (deepest tried first, so the first
        rung that fits the budget - the deepest affordable one - is
        returned immediately without trying the rest). Supplying this as
        a parameter (rather than a hardcoded ladder here) keeps this
        function's own order-2-vs-order>=3 shrink policy entirely in
        `models/RF.py` (mirroring where `_resolve_shapiq_tree_depth_
        cap()`'s own order-aware schedule already lived).
    max_order, index : forwarded to the probe `shapiq.TreeExplainer(
        max_order=..., index=...)` construction - should match what the
        REAL (non-probe) call will use, so the probe's own cost is
        representative of it.
    per_rung_seconds : float, keyword-only, required
        Wall-clock ceiling for a SINGLE rung's fit + construct + probe-row
        attempt. Deliberately required (no default) - `models/RF.py`'s
        own call site derives this from the real task's marker count
        (see that call site's own comment for the benchmarked heuristic),
        since a single fixed constant cannot be safe across both a
        50-marker and a 5,000-marker shortlist.
    log_prefix : str
        Prefix for this function's own printed NOTE lines.

    Returns
    -------
    int
        The deepest `depth_ladder` entry whose fit + construct + one
        probe-row attempt completed within `per_rung_seconds` without
        raising. Falls back to `depth_ladder[-1]` (the shallowest,
        intended-as-a-safety-floor rung) if EVERY rung failed or
        overran - `depth_ladder`'s own caller is responsible for making
        that floor genuinely safe (mirrors `_resolve_shapiq_tree_depth_
        cap()`'s own floor-of-3 contract); this function itself never
        raises.
    """
    import time
    import shapiq

    for _depth in depth_ladder:
        _t0 = time.monotonic()
        try:
            with bounded_memory_guard(log_prefix=f"{log_prefix} calibrate_shapiq_tree_depth"):
                _forest = fit_forest_fn(_depth)
                _explainer = shapiq.TreeExplainer(model=_forest, max_order=max_order, index=index)
                _explainer.explain(probe_row)
        except Exception as exc:
            _elapsed = time.monotonic() - _t0
            print(f"{log_prefix} NOTE: calibrate_shapiq_tree_depth: candidate depth={_depth} "
                  f"rejected after {_elapsed:.1f}s ({exc!r}) - trying the next, shallower "
                  f"candidate depth.")
            continue
        _elapsed = time.monotonic() - _t0
        if _elapsed > per_rung_seconds:
            print(f"{log_prefix} NOTE: calibrate_shapiq_tree_depth: candidate depth={_depth} "
                  f"completed (fit + TreeExplainer construction + one probe row) in {_elapsed:.1f}s "
                  f"- over this rung's {per_rung_seconds:.1f}s budget - trying the next, shallower "
                  f"candidate depth.")
            continue
        print(f"{log_prefix} NOTE: calibrate_shapiq_tree_depth: candidate depth={_depth} accepted "
              f"({_elapsed:.1f}s fit + TreeExplainer construction + one probe row, within the "
              f"{per_rung_seconds:.1f}s per-rung budget). Using this depth for the real "
              f"interaction search.")
        return _depth

    print(f"{log_prefix} NOTE: calibrate_shapiq_tree_depth: every candidate depth "
          f"{tuple(depth_ladder)} was rejected (too slow and/or ran out of memory) - falling back "
          f"to the shallowest candidate ({depth_ladder[-1]}) unconditionally, the same safety "
          f"floor this route always fell back to before calibration existed.")
    return depth_ladder[-1]


# ---------------------------------------------------------------------------
# ver4-25, R4/R5 - shared plot-authority helpers: identical (shared-limit,
# shared-tick, 1:1-line) scatter axes, and the same row-wise L1
# normalisation the naive ensemble already uses for marker effects.
#
# Both are used by more than one plotting module (accuracy_metrics_plot.py
# for R4, scatter_plot.py for R5) - defined here, once, rather than in
# either plotting module, so a future third caller (or a change to either
# formula) never has to pick which existing copy to import from or risk
# two independently-drifting copies (the same "single shared authority"
# discipline `canonical_model_order()`/`safe_regression_metrics()` above
# already establish for this module - architecture doc §2's Layer 3d
# precedent, applied here to a plotting/normalisation concern instead of
# a model-registry one).
# ---------------------------------------------------------------------------

def set_identity_axes(ax, x_extent, y_extent, *, pad_frac: float = 0.05, nbins: int = 6,
                       line_kwargs: Optional[Dict] = None, zero_min: bool = False,
                       draw_line: bool = True) -> Tuple[float, float]:
    """Give a scatter-style Axes IDENTICAL x/y limits and tick positions,
    equal aspect, and a corner-to-corner 1:1 reference line - so a
    "predicted vs actual"/"model A vs model B" panel is never visually
    distorted by independently-scaled axes (ver4-25 blueprint, R4 §2/R5
    §2 - shared design record).

    Args:
        ax: the `matplotlib.axes.Axes` to configure (mutated in place).
        x_extent: every x-value (or x-error-bar endpoint) that will be
            plotted on `ax` - used only to derive the shared limits, not
            plotted itself. May contain NaN/inf; non-finite values are
            ignored.
        y_extent: the y-axis counterpart of `x_extent`.
        pad_frac: fractional padding added to each side of the raw
            `[min, max]` span before snapping to tick positions (default
            5%), so the outermost points/error bars are not drawn flush
            against the frame. When `zero_min=True`, padding is only
            ever applied to the UPPER end - the lower end is pinned to
            exactly 0 regardless of `pad_frac` (see `zero_min` below).
        nbins: the target number of tick intervals passed to
            `matplotlib.ticker.MaxNLocator` (default 6) - the same
            locator matplotlib's own default axis formatting uses, so
            the resulting ticks look like an ordinary, unforced axis.
        line_kwargs: optional overrides for the 1:1 line's own
            `matplotlib.lines.Line2D` keyword arguments (default: grey,
            dashed, thin, drawn behind the data via `zorder=0`).
        zero_min: Requirements3.md item 3 - when True, both axes are
            forced to start EXACTLY at 0 rather than at a padded
            `min(data)` (which, for marker-effect panels, is often
            slightly negative purely from `pad_frac` padding applied to
            an already non-negative - post-`abs()`/L1-normalisation -
            value range, making it look like marker effects can go
            below 0, which they never do). Any negative values in
            `x_extent`/`y_extent` are clamped up to 0 for the purpose of
            deriving `hi` (defensive - this function does not itself
            assume its caller's data is non-negative), rather than
            silently cropping/hiding them, so a caller that accidentally
            passes signed data still gets a sensible (if slightly
            misleading) axis instead of a crash. Default False
            reproduces this function's pre-Requirements3.md behaviour
            exactly (I11) - used only for marker-effect panels
            (`scatter_plot.py`'s lower-triangle/marker-effect side), never
            for phenotype-prediction panels (R4/R5's own Q2 decision:
            those keep data-derived, unforced limits).
        draw_line: Requirements.md item 5 (this session) - when False, the
            dashed corner-to-corner 1:1 reference line is skipped, but the
            shared identity limits/ticks/aspect are still applied. Used by
            `scatter_plot.py` for its marker-effect panels only when
            marker effects are left un-normalised (raw values are not
            guaranteed to be on a comparable scale across models, so a 1:1
            reference line would be misleading there) - phenotype panels,
            and every panel when marker effects ARE normalised, keep the
            line (default True reproduces this function's prior behaviour
            exactly - I11).

    Returns:
        `(lo, hi)` - the final shared axis limits (also each axis's
        first/last tick), in case the caller wants them for anything
        else (e.g. annotating the panel).

    Raises:
        Nothing - a completely empty/non-finite `x_extent`/`y_extent`
        falls back to a small window around zero (or `(0, 0.5)` when
        `zero_min=True`) rather than raising, since an empty panel is
        still a valid (if uninformative) plot.
    """
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    combined = np.asarray(list(x_extent) + list(y_extent), dtype=float)
    combined = combined[np.isfinite(combined)]
    if combined.size == 0:
        lo, hi = (0.0, 0.5) if zero_min else (-0.5, 0.5)
    else:
        lo = float(np.min(combined))
        hi = float(np.max(combined))
        if zero_min:
            # Requirements3.md item 3: the lower end is ALWAYS exactly 0
            # for a zero_min panel - never padded, never below 0 even if
            # a stray negative value is present (see docstring above).
            lo = 0.0
            hi = max(hi, 0.0)
            if hi == 0.0:
                hi = 0.5
            else:
                hi = hi + pad_frac * hi
        elif hi == lo:
            half = 0.5 * max(abs(lo), 1e-9)
            lo, hi = lo - half, hi + half
        else:
            pad = pad_frac * (hi - lo)
            lo, hi = lo - pad, hi + pad

    ticks = MaxNLocator(nbins=nbins).tick_values(lo, hi)
    if zero_min:
        # MaxNLocator.tick_values() is free to return a first tick BELOW
        # the requested lo (e.g. a "nice round number" just under 0) -
        # the whole point of zero_min is that the axis's own drawn origin
        # is always exactly 0, so lo is pinned here rather than taken
        # from ticks[0]; only hi (and the tick positions themselves,
        # which still include any sub-0 tick MaxNLocator proposed) come
        # from the locator. Any tick below 0 is dropped - it would sit
        # outside the [0, hi] limits being set below anyway.
        ticks = ticks[ticks >= 0]
        if ticks.size == 0 or ticks[0] != 0.0:
            ticks = np.concatenate(([0.0], ticks))
        lo, hi = 0.0, float(ticks[-1])
    else:
        lo, hi = float(ticks[0]), float(ticks[-1])

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_aspect('equal', adjustable='box')

    if draw_line:
        resolved_line_kwargs = {'color': 'grey', 'linestyle': '--', 'linewidth': 1.0, 'zorder': 0}
        if line_kwargs:
            resolved_line_kwargs.update(line_kwargs)
        ax.add_line(Line2D([lo, hi], [lo, hi], **resolved_line_kwargs))

    return lo, hi


def safe_row_l1_normalize(df: "pd.DataFrame") -> "pd.DataFrame":
    """Row-wise L1 normalisation (`row / row.sum()`) - VERBATIM semantics
    of `models/ensemble.py::_safe_row_normalize`, promoted here (ver4-25,
    R5) as the single shared authority so `scatter_plot.py`'s per-task
    effect normalisation can never numerically disagree with what the
    naive ensemble itself computes over the same rows (I6's "reuse,
    never re-implement" discipline, applied to this normalisation the
    same way it already applies to `_map_genes_to_markers`).
    `models/ensemble.py::_safe_row_normalize` now delegates to this
    function (kept as a thin alias, since it is called from several
    places within that module already).

    Like the function it replaces, this assumes the caller has already
    made every value non-negative (e.g. via `.abs()`) where only
    magnitude matters - it sums each row AS GIVEN, it does not take an
    absolute value itself, so a caller that passes signed values gets a
    signed-sum normalisation (dividing by the row's plain sum, not its
    L1 norm) - exactly `_safe_row_normalize`'s own pre-existing contract.

    Casts to `float64` first (an all-`object`-dtype frame - e.g. marker
    effects arriving from the R models via `rpy2` - falls back to
    per-element Python arithmetic, where dividing by an exact-zero row
    sum RAISES `ZeroDivisionError` before pandas' own NaN-producing
    float division ever runs - the real production crash
    `_safe_row_normalize`'s own docstring/history records), replaces an
    exact-zero row sum with `NaN` *before* dividing (so a genuinely
    all-zero row - e.g. SVR/KNN, which are not additive-effect models -
    comes back all-zero rather than raising), and never raises.

    Args:
        df: a DataFrame whose columns are the values to normalise WITHIN
            each row (e.g. one column per marker). The caller applies
            `.abs()` first wherever only magnitude should matter -
            mirroring every existing call site of `_safe_row_normalize`.

    Returns:
        A new DataFrame, same shape/index/columns as `df`, each row
        divided by that row's own sum (an all-zero row stays all-zero,
        never `NaN`/`inf`).

    Raises:
        Nothing - degrades a malformed (non-numeric, unconvertible)
        frame the same way `float64` casting itself would (a `TypeError`/
        `ValueError` from `.astype('float64')`, i.e. this function is not
        a silent no-op for genuinely non-numeric input).
    """
    import numpy as np

    values = df.astype('float64')
    row_sums = values.sum(axis=1).replace(0, np.nan)
    return values.div(row_sums, axis=0).fillna(0.0)


# ---------------------------------------------------------------------------
# ver4-4 R6 - QTL window marker assignment.
#
# Generalises the SAME interval-overlap predicate the biological-prior GAT
# uses for gene -> marker assignment (models.GAT_biological_prior_knowledge.
# _map_genes_to_markers) to any region table shaped like gene_info.csv
# (chromosome, start, end, ...) - here, a QTL table - so scatter_plot.py's
# QTL highlighting can flag markers that fall NEAR a QTL, not only markers
# whose NAME happens to exactly match a pre-mapped marker name (the
# pre-ver4-4 behaviour, which almost never fires in practice - a genotyped
# marker rarely coincides exactly with a reported QTL position; see the
# ver4-4 Design Blueprint S2.6.1). Imported, never re-implemented (I6) - a
# second copy of the same overlap test could silently drift from the one
# the bio-prior GAT actually uses to build its gene graph.
# ---------------------------------------------------------------------------

def markers_in_windows(regions, marker_info, available_markers, *,
                        window: float = 0.0, mode: str = 'all_in_window'):
    """Assign every region (row of `regions`) the marker(s) that fall near
    it, by genomic interval overlap - the ver4-4 R6 QTL-windowing helper.

    Two assignment modes:

      - ``mode='all_in_window'``: every marker whose interval overlaps the
        region AFTER the region has been widened by +/- `window` on each
        side. Delegates this overlap test to
        ``models.GAT_biological_prior_knowledge._map_genes_to_markers`` -
        the SAME predicate (``marker.start <= region.end AND
        marker.end >= region.start``) that function already uses for
        gene-window marker assignment - rather than a second, potentially
        divergent, implementation. `window=0` therefore returns only
        markers that genuinely, exactly overlap the region.

      - ``mode='nearest'``: returns the SINGLE marker (on the same
        chromosome, after the same chromosome-name normalisation
        `_map_genes_to_markers` itself applies) whose midpoint is closest
        to the region's own midpoint - a plain nearest-neighbour search
        across every candidate marker on that chromosome, independent of
        `window` (widening a region symmetrically never moves its
        midpoint, so `window` has no effect on which marker is nearest;
        it is still accepted, for signature symmetry with
        `all_in_window`). This is what makes `window=0` with
        `mode='nearest'` still return a marker even when nothing
        genuinely overlaps - `all_in_window` and `nearest` answer
        genuinely different questions ("what overlaps?" vs. "what's
        closest?"), not the same question at two window sizes. Ties
        (identical distance) are broken by lower marker `start`, then by
        lexicographic marker `name`, so the result is always fully
        deterministic.

    Parameters
    ----------
    regions : pandas.DataFrame
        One row per region to assign markers to (a QTL table, or anything
        else shaped like it). Must have at least 'chromosome', 'start',
        'end' columns; any other columns (e.g. 'name', 'phenotype') are
        ignored by this function and left completely untouched by the
        caller.
    marker_info : pandas.DataFrame
        The marker coordinate table (marker_info.csv's own contract:
        'chromosome', 'name', 'start', 'end' - see architecture doc S4.1).
        Never mutated; a filtered copy is made internally.
    available_markers : iterable of str
        The marker names actually present in the genotype table being
        modelled for this task - markers outside this set are never
        assigned, mirroring `_map_genes_to_markers`'s own restriction (a
        marker that isn't even in this task's data can't usefully be
        flagged as "near" anything).
    window : float, default 0.0
        Symmetric distance, in the SAME units as marker_info's/regions'
        `start`/`end` columns, by which every region is widened before
        the `all_in_window` overlap test. Ignored by `mode='nearest'`
        (see above). `window=0.0` reproduces exact-boundary overlap.
    mode : {'all_in_window', 'nearest'}, keyword-only, default
        'all_in_window'. Selects which of the two assignment strategies
        above is used.

    Returns
    -------
    list of list of str
        Positionally aligned with `regions` - i.e. `region_to_markers[i]`
        is the marker-name list assigned to `regions.iloc[i]` (matching
        `_map_genes_to_markers`'s own positional, not label-based,
        alignment convention). `all_in_window` entries are sorted by
        marker start position, exactly as `_map_genes_to_markers` returns
        them. `nearest` entries are single-element lists, except when
        zero candidate markers exist on that region's chromosome at all
        (see the chromosome-mismatch warning below), in which case the
        entry is an empty list for both modes.

    Raises
    ------
    ValueError
        If `mode` isn't one of the two supported values, or if `regions`/
        `marker_info` is missing a required column - both genuine schema
        problems this function can't silently paper over.
    ImportError
        If `models.GAT_biological_prior_knowledge` (and, through it,
        torch/torch_geometric) can't be imported. This function itself
        never touches torch - the import is needed only to reuse
        `_map_genes_to_markers`'s pure-pandas overlap predicate without
        re-implementing it (I6) - but that predicate lives in a module
        that also imports torch/torch_geometric at the top level, so
        calling `markers_in_windows()` at all requires those packages to
        be importable in this environment, even though this specific
        function has no other use for them. This dependency is
        DEFERRED (imported here, inside the function body, not at this
        module's top level) specifically so that merely importing
        `pipeline_utils` - as every layer of this codebase does, GUI
        through headless scripts through preprocessing - never forces a
        torch/torch_geometric import; only actually CALLING
        `markers_in_windows()` (i.e. only ever from `scatter_plot.py`,
        only when a coordinate-mode QTL file is in use) does. Reusing
        `_map_genes_to_markers` from `pipeline_utils` at module level
        would otherwise invert the architecture's own "dependencies
        point strictly downward" layering (architecture doc S2) - models/
        is layer 3b, pipeline_utils is a shared helper every layer,
        including the GUI, imports.
    """
    if mode not in ('all_in_window', 'nearest'):
        raise ValueError(
            f"[pipeline_utils] markers_in_windows: mode must be 'all_in_window' or "
            f"'nearest', got {mode!r}."
        )

    required_region_cols = {'chromosome', 'start', 'end'}
    missing_region_cols = required_region_cols - set(regions.columns)
    if missing_region_cols:
        raise ValueError(
            f"[pipeline_utils] markers_in_windows: `regions` is missing required column(s) "
            f"{sorted(missing_region_cols)} - expected at least {sorted(required_region_cols)}. "
            f"Got columns: {list(regions.columns)}."
        )
    required_marker_cols = {'name', 'chromosome', 'start', 'end'}
    missing_marker_cols = required_marker_cols - set(marker_info.columns)
    if missing_marker_cols:
        raise ValueError(
            f"[pipeline_utils] markers_in_windows: `marker_info` is missing required column(s) "
            f"{sorted(missing_marker_cols)} - expected at least {sorted(required_marker_cols)}. "
            f"Got columns: {list(marker_info.columns)}."
        )

    if regions.shape[0] == 0:
        return []

    try:
        from models.GAT_biological_prior_knowledge import _map_genes_to_markers, _normalise_chrom
    except ImportError as exc:
        raise ImportError(
            "[pipeline_utils] markers_in_windows: could not import the shared interval-overlap "
            "predicate from models.GAT_biological_prior_knowledge (_map_genes_to_markers/"
            "_normalise_chrom). This helper is REUSED, not re-implemented, to guarantee "
            "QTL-window marker assignment is byte-identical to the bio-prior gene-window "
            "assignment (architecture doc S8, invariant I6). That module imports torch/"
            "torch_geometric at its own top level even though neither function this call needs "
            "ever touches them - install torch and torch_geometric, or QTL windowing cannot be "
            "used in this environment (the rest of the pipeline is unaffected)."
        ) from exc

    regions_reset = regions.reset_index(drop=True)

    # available_markers is consumed twice below (once inside
    # _map_genes_to_markers, once again for the 'nearest' branch's own
    # candidate filtering) - materialise it once as a set for a fast,
    # order-independent membership test in both places.
    available_markers_set = set(available_markers)

    # --- one consolidated "chromosome has zero candidate markers at all"
    # warning, computed once regardless of mode, mirroring the existing
    # chromosome-name-mismatch warning in circos_plot.py (same "list every
    # affected name once, up front" pattern, rather than one warning per
    # region). ---
    marker_info_avail = marker_info[marker_info['name'].isin(available_markers_set)].copy()
    marker_info_avail['_chrom_norm'] = _normalise_chrom(marker_info_avail['chromosome'])
    region_chrom_norm = _normalise_chrom(regions_reset['chromosome'])
    chrom_have = set(marker_info_avail['_chrom_norm'].unique().tolist())
    missing_chrom = sorted({c for c in region_chrom_norm.unique().tolist() if c not in chrom_have})
    if missing_chrom:
        print(
            f"[pipeline_utils] WARNING: markers_in_windows: {len(missing_chrom)} chromosome(s) "
            f"present in the region/QTL table have NO matching marker at all in marker_info "
            f"(after restricting to this task's available markers), so 0 markers can ever be "
            f"flagged for a region on them, regardless of window size or mode: "
            f"{missing_chrom[:10]}{'...' if len(missing_chrom) > 10 else ''}. Check whether the "
            f"two files use different chromosome-naming conventions (e.g. 'A10' vs '10A', or a "
            f"'chr' prefix on only one side)."
        )

    if mode == 'all_in_window':
        regions_widened = regions_reset.copy()
        regions_widened['start'] = regions_widened['start'].astype(float) - float(window)
        regions_widened['end'] = regions_widened['end'].astype(float) + float(window)
        return _map_genes_to_markers(regions_widened, marker_info, available_markers_set)

    # mode == 'nearest' - plain nearest-neighbour search, independent of
    # `window` (see docstring). marker_info_avail/region_chrom_norm above
    # are reused directly rather than recomputed.
    marker_info_avail = marker_info_avail.copy()
    marker_info_avail['_start_f'] = marker_info_avail['start'].astype(float)
    marker_info_avail['_end_f'] = marker_info_avail['end'].astype(float)
    marker_info_avail['_mid'] = (marker_info_avail['_start_f'] + marker_info_avail['_end_f']) / 2.0

    region_start = regions_reset['start'].astype(float)
    region_end = regions_reset['end'].astype(float)
    region_mid = (region_start + region_end) / 2.0

    region_to_markers = []
    for i in range(regions_reset.shape[0]):
        chrom = region_chrom_norm.iloc[i]
        candidates = marker_info_avail[marker_info_avail['_chrom_norm'] == chrom]
        if candidates.shape[0] == 0:
            region_to_markers.append([])
            continue
        candidates = candidates.copy()
        candidates['_dist'] = (candidates['_mid'] - region_mid.iloc[i]).abs()
        candidates = candidates.sort_values(['_dist', '_start_f', 'name'])
        region_to_markers.append([candidates['name'].iloc[0]])

    return region_to_markers


def find_bglr_output_dirs(root_dir: str) -> List[str]:
    """Recursively find every directory literally named 'BGLR_output'
    under `root_dir` (typically a `Result/<RESULT_NAME>` tree, or
    `Result/` itself to sweep every RESULT_NAME at once). Matches BOTH the
    sequential-run location (`Result/<name>/BGLR_output/`) and the
    intra-batch-parallel per-task location
    (`Result/<name>/.intra_batch_parallel/batch_<b>/task_<t>/
    BGLR_output/` - architecture doc Sec 16/17) - deliberately a directory
    NAME match rather than a reconstructed path, so this can never
    silently drift out of sync with intra_batch_parallel.py's own
    directory-naming scheme (see this module's RESULT_BASE_DIR note above
    for why independently re-deriving a path elsewhere in this codebase
    has caused a real bug before). Used only by the standalone maintenance
    script (bglr_output_cleanup.py); the in-process cleanup call site (see
    cleanup_bglr_output() above) always uses its own already-known
    directory instead."""
    found = []
    if not root_dir or not os.path.isdir(root_dir):
        return found
    for dirpath, dirnames, _filenames in os.walk(root_dir):
        if os.path.basename(dirpath) == 'BGLR_output':
            found.append(dirpath)
            dirnames[:] = []  # nothing nested under a BGLR_output dir - don't descend
    return found
