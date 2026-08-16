"""
EasiGP - shared helpers used by BOTH streamlit_app.py (the interactive GUI,
used to author a configuration ONCE) and the headless, non-interactive
scripts actually launched by the HPC scheduler (run_sequential.py,
run_step1_batch.py, run_step2_assemble.py).

Keeping these in a separate module means the headless scripts never have to
import streamlit_app.py itself (which would re-execute the whole GUI script
outside of a Streamlit runtime and fail).
"""

import os
from datetime import datetime

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
    log_dir = os.path.join('.', 'Result', result_name, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(log_dir, f'log_{label}_{stamp}.txt')


def configure_r_environment(r_path):
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


def resolve_batch_id_from_env():
    """Read the batch ID from whichever scheduler's environment variable is
    set. Raises RuntimeError with a clear message if neither is present.
    (Deliberately RuntimeError rather than SystemExit so this can be called
    safely from inside the Streamlit app too, where SystemExit would tear
    down the whole session instead of just showing an error message.)"""
    env_val = os.environ.get('SLURM_ARRAY_TASK_ID')
    if env_val is not None:
        return int(env_val)

    env_val = os.environ.get('PBS_ARRAY_INDEX', os.environ.get('PBS_ARRAYID'))
    if env_val is not None:
        return int(env_val)

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



    """JSON turns (train, valid, test) tuples into plain lists; restore
    tuples when the 'between' scenario needs them. No-op otherwise."""
    #if w_opt is not None:
    if len(ratio[0]) != 0:
        return [tuple(item) if isinstance(item, list) else item for item in ratio]
    return ratio