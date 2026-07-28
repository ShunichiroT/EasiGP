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
            if chunk.endswith('\n'):
                self._at_line_start = True
            for s in self.streams:
                s.write(chunk)
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
    """Build a timestamped log file path under Result/<result_name>/, e.g.
    Result/MaizeNAM/log_sequential_20260727_143205.txt. `label` is a short
    tag identifying which kind of run this is (e.g. 'sequential_local',
    'step1_local', 'step2_local'). Creates the Result/<result_name>/ folder
    if it doesn't exist yet."""
    result_dir = os.path.join('.', 'Result', result_name)
    os.makedirs(result_dir, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(result_dir, f'log_{label}_{stamp}.txt')


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


def restore_ratio(ratio, scenario):
    """JSON turns (train, valid, test) tuples into plain lists; restore
    tuples when the 'between' scenario needs them. No-op otherwise."""
    if scenario == 'between':
        return [tuple(item) if isinstance(item, list) else item for item in ratio]
    return ratio
