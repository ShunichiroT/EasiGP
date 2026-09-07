"""
EasiGP - shared helpers used by BOTH streamlit_app.py (the interactive GUI,
used to author a configuration ONCE) and the headless, non-interactive
scripts actually launched by the HPC scheduler (run_sequential.py,
run_step1_batch.py, run_step2_assemble.py).

Keeping these in a separate module means the headless scripts never have to
import streamlit_app.py itself (which would re-execute the whole GUI script
outside of a Streamlit runtime and fail).
"""

import csv
import os
import shutil
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

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


# Options offered in the GUI for how a PARALLEL batch ID should be
# determined. Kept here so the GUI and the headless runner agree on the
# same source names / environment variables.
BATCH_ID_SOURCES = [
    'Manual integer',
    'Slurm (SLURM_ARRAY_TASK_ID)',
    'PBS (PBS_ARRAY_INDEX / PBS_ARRAYID)',
]


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

    if not r_path:
        return

    r_path = r_path.strip().strip('"').rstrip(';').rstrip('\\/')
    os.environ['R_HOME'] = r_path

    for bin_subdir in (os.path.join('bin', 'x64'), 'bin'):
        r_bin = os.path.join(r_path, bin_subdir)
        if os.path.isdir(r_bin):
            current_path = os.environ.get('PATH', '')
            if r_bin not in current_path:
                os.environ['PATH'] = r_bin + os.pathsep + current_path
            break


def init_rpy2_conversion():
    """Make sure this process has a valid rpy2 R<->Python conversion
    context. Safe to call from any entry point (GUI or headless script)."""
    try:
        from rpy2.robjects import conversion, default_converter
    except ImportError:
        # rpy2 not installed / not used by this pipeline - nothing to do.
        return

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
_WOPT_LABEL_PREFIXES = ('Linear transformation', 'Nelder Mead', 'Bayesian optimisation', 'Analytic least-squares')


def _is_wopt_label(label) -> bool:
    return isinstance(label, str) and any(
        label == p or label.startswith(p + '__') for p in _WOPT_LABEL_PREFIXES
    )


def _is_naive_ensemble_label(label) -> bool:
    return isinstance(label, str) and (label == 'ensemble' or label.startswith('ensemble__'))


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
    hp_tune_warm_start = bool(cfg.get('HP_TUNE_WARM_START', False))
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

    return {
        'device': str(device),
        'use_gpu_sklearn': bool(cfg.get('USE_GPU_SKLEARN', False)),
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
        'hp_tune_warm_start': hp_tune_warm_start,
        'hp_tune_selection_margin': hp_tune_selection_margin,
        'hp_tune_valid_repeats': hp_tune_valid_repeats,
        'hp_tune_scope': hp_tune_scope,
        'w_opt_analytic_seed': w_opt_analytic_seed,
        'w_opt_validation_floor': w_opt_validation_floor,
        'w_opt_simplex_search': w_opt_simplex_search,
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


def apply_torch_compute_settings(resources: Dict[str, object]) -> None:
    """Apply the process-global PyTorch settings implied by
    ``resolve_compute_resources()``'s output - split out from the
    resolver itself so a pure-CPU caller (or one that hasn't imported
    torch yet) never pays for/triggers a torch import just to read
    settings it won't use. Safe to call repeatedly; a no-op if torch
    isn't installed.

    CUDA device: UNCHANGED behaviour from before ver4-4 - sets
    ``torch.backends.cudnn.benchmark = True`` when
    ``resources['cudnn_benchmark']`` is truthy, else does nothing.

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


def gpu_slot():
    """Context manager guarding entry into a GPU-dispatched model call.
    Usage::

        with gpu_slot():
            model.to(device)
            ...forward/backward pass...

    A no-op whenever ``install_gpu_semaphore()`` hasn't been called (the
    default for every run that isn't using intra-batch parallelism with
    more than one GPU-capable worker)."""
    if _GPU_SEMAPHORE is None:
        return _NullGpuSlot()
    return _GPU_SEMAPHORE


# ---------------------------------------------------------------------------
# Phase 2, Requirement 8 - cheap, non-data-reading resource-advisor inputs.
#
# Deliberately kept in this module (rather than resource_profiles.py, which
# holds the STATIC per-model cost table) since it reuses/extends the exact
# same "cheap header/line-count read" pattern list_phenotype_columns()
# already established, and pipeline_utils.py already exists to serve both
# the GUI and headless sides.
# ---------------------------------------------------------------------------

def count_csv_data_rows(path: str) -> Optional[int]:
    """Cheap row COUNT (not a full read) of a CSV file's data rows (header
    excluded) - a fast line-iteration, never materialising the file's
    columns/values. Returns ``None`` if the path is missing/unreadable."""
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8-sig', newline='') as f:
            n_lines = sum(1 for _ in f)
    except OSError:
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
    try:
        with open(bim_path, 'r', encoding='utf-8-sig') as f:
            return sum(1 for _ in f)
    except OSError:
        return None


def count_fam_lines(fam_path: str) -> Optional[int]:
    """Cheap line count of a PLINK1 ``.fam`` file - one line per
    individual. Returns ``None`` if the path is missing/unreadable."""
    if not fam_path or not os.path.isfile(fam_path):
        return None
    try:
        with open(fam_path, 'r', encoding='utf-8-sig') as f:
            return sum(1 for _ in f)
    except OSError:
        return None


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


def count_unique_populations(csv_path: str) -> Optional[int]:
    """Best-effort count of DISTINCT values in a genotype/phenotype CSV's
    'population' column (its second column, per the fixed positional
    contract - see ``unify_columns_by_position``) - used only by the
    Phase 2, Requirement 8 resource advisor's cheap `n_population` input.

    Unlike this module's other cheap-read helpers, this genuinely reads
    every data row (there is no way to know how many distinct populations
    exist without doing so) - it only reads ONE column (`usecols`), not
    the full row, which is still far cheaper than materialising the whole
    genotype matrix, and is bounded by this function's own best-effort,
    never-raising contract: on any failure (missing pandas, malformed
    file, huge file that's impractical to scan this way, ...) this
    returns ``None`` rather than raising, and callers should fall back to
    a conservative default (e.g. 1) when that happens.
    """
    if not csv_path or not os.path.isfile(csv_path):
        return None
    try:
        import pandas as pd
        col = pd.read_csv(csv_path, usecols=[1], encoding='utf-8-sig')
        return int(col.iloc[:, 0].nunique())
    except Exception:
        return None


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


def safe_regression_metrics(actual, predicted) -> Tuple[float, float]:
    """(pearson_r, mse) for one prediction column, NEVER raising.

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


def parallel_shap_values(explainer, samples, nsamples, n_jobs, *, interaction: bool = False):
    """Row-fan-out wrapper around a SHAP explainer's own batch call - fans
    a ``samples`` batch out across ``n_jobs`` worker PROCESSES (one row
    per worker call, via ``joblib.Parallel``), then re-stacks the
    per-row results back into the SAME shape a single, un-parallelised
    call to the explainer would have returned.

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

    n_rows = samples.shape[0]
    if n_jobs in (None, 0, 1) or n_rows <= 1:
        if interaction:
            return np.asarray(explainer.shap_interaction_values(samples))
        return np.asarray(explainer.shap_values(samples, nsamples=nsamples))

    try:
        rows = [samples.iloc[[i]] for i in range(n_rows)]
        if interaction:
            results = Parallel(n_jobs=n_jobs)(
                delayed(lambda row: np.asarray(explainer.shap_interaction_values(row)))(row)
                for row in rows
            )
            return np.concatenate(results, axis=0)
        results = Parallel(n_jobs=n_jobs)(
            delayed(lambda row: np.asarray(explainer.shap_values(row, nsamples=nsamples)))(row)
            for row in rows
        )
        return np.vstack([np.atleast_2d(r) for r in results])
    except Exception as exc:
        print(f"[pipeline_utils] NOTE: parallel_shap_values fan-out failed ({exc}) - falling "
              f"back to a single serial explainer call.")
        if interaction:
            return np.asarray(explainer.shap_interaction_values(samples))
        return np.asarray(explainer.shap_values(samples, nsamples=nsamples))


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
