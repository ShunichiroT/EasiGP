"""
Preprocess/preprocess_memo.py
==============================

ver4-33, R1.c - a content-addressed, on-disk memo for the two expensive,
PURE preprocessing steps that every model group, every array task and every
phenotype sharing a training set used to repeat (design blueprint D5):

* the native PLINK LD-prune marker list (``plink2 --indep-pairwise`` on the
  full fileset), and
* the data-driven-merge Random-Forest marker selection
  (``select_markers_for_data_driven_network``).

Both are pure functions of their inputs, so a result stored under a key
that contains EVERY input is exactly what recomputation would return - this
is a cache, not shared mutable state, and it does not weaken task
independence (I9): every task still builds its own ``sample`` table and
fits on its own training IIDs only (I7 - the keys contain the training IIDs
/ training matrix, never validation or test).

Location and safety
-------------------
Entries live under ``$EASIGP_MEMO_DIR`` (set once by the outermost entry
point to ``Result/<RESULT_NAME>/_memo/`` - see
``pipeline_utils.ensure_memo_dir_env``), so spawned and loky children, and
SLURM array tasks on a shared filesystem, all see the same memo. Writes are
atomic (temp file in the same directory + ``os.replace``), so a race only
duplicates identical work and a reader never sees a half-written file (I10).
A corrupt/unreadable entry is recomputed and overwritten (logged); an
unwritable directory disables the memo for this process (logged once).

Disclosed exception: with ``USE_GPU_SKLEARN`` (cuML, possibly
non-deterministic) the memo freezes the FIRST selection - which removes an
existing cross-model-group inconsistency rather than adding one.
"""

import contextlib
import hashlib
import json
import os
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pipeline_utils import (
    MEMO_DIR_ENV, get_active_compute_resources, note_stage_tag, timed_stage,
)

# Bump when the stored format or a key recipe changes: old entries are then
# simply never matched.
_MEMO_SCHEMA = 1

# Whether the number of PLINK threads is part of the LD key. The design
# blueprint's pre-check P4 (is `.prune.in` identical for --threads 1 vs 8?)
# could not be executed when this was written, so the conservative choice is
# made: threads ARE in the key. Within one run every worker uses the same
# PLINK_THREADS, so this costs no hits in practice; set False once P4 passes.
LD_KEY_INCLUDES_THREADS = True

_DISABLED_REASON: Optional[str] = None      # set when the memo dir proved unusable
_PLINK_VERSION_CACHE: Dict[str, Optional[str]] = {}


# ---------------------------------------------------------------------- #
# switches and paths
# ---------------------------------------------------------------------- #

def memo_enabled() -> bool:
    """True when ``ACCELERATION['preprocess_memo']`` is on AND a usable
    ``EASIGP_MEMO_DIR`` exists for this process (see module docstring)."""
    if _DISABLED_REASON is not None:
        return False
    if not bool(get_active_compute_resources().get('accel_preprocess_memo', False)):
        return False
    return bool(os.environ.get(MEMO_DIR_ENV))


def _disable(reason: str) -> None:
    """Disable the memo for the rest of this process and say why, once."""
    global _DISABLED_REASON
    if _DISABLED_REASON is None:
        _DISABLED_REASON = reason
        print(f"[preprocess_memo] NOTE: memo disabled for this process - {reason}. "
              f"Preprocessing is recomputed every time (results are unaffected).")


def _entry_path(kind: str, key: str) -> str:
    return os.path.join(os.environ[MEMO_DIR_ENV], f"{kind}_{key}.json")


# ---------------------------------------------------------------------- #
# keys
# ---------------------------------------------------------------------- #

def _canonical_json(obj) -> str:
    """Deterministic JSON text for hashing (sorted keys, no whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), default=str)


def _digest(*parts) -> str:
    """32-hex-character blake2b digest of the given bytes/str parts."""
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(str(_MEMO_SCHEMA).encode('ascii'))
    for part in parts:
        data = part if isinstance(part, bytes) else str(part).encode('utf-8')
        hasher.update(len(data).to_bytes(8, 'little'))
        hasher.update(data)
    return hasher.hexdigest()


def plink_version(plink_path: str) -> Optional[str]:
    """First line of ``<plink_path> --version`` (cached per process), or
    None if it cannot be determined - in which case the LD memo is skipped,
    because a result keyed without the tool version could go stale."""
    if plink_path in _PLINK_VERSION_CACHE:
        return _PLINK_VERSION_CACHE[plink_path]
    version = None
    try:
        out = subprocess.run([plink_path, '--version'], capture_output=True, text=True, timeout=60)
        text = (out.stdout or out.stderr or '').strip().splitlines()
        version = text[0].strip() if text else None
    except (OSError, subprocess.SubprocessError):
        version = None
    _PLINK_VERSION_CACHE[plink_path] = version
    return version


def ld_prune_key(bfile_stem: str, keep_iid_list, ld_config: dict, plink_path: str,
                 threads: int) -> Optional[str]:
    """Key for the native LD-prune marker list, or None if unkeyable.

    Contains: size + mtime_ns of .bed/.bim/.fam, the plink2 version, the
    sorted unique training IIDs, and every ``ld_config`` field that reaches
    the plink2 command line (window, window_unit, step, r2_threshold,
    maf_threshold, allow_extra_chr, chr_set) plus the thread count (see
    ``LD_KEY_INCLUDES_THREADS``).
    """
    version = plink_version(plink_path)
    if version is None:
        return None
    try:
        stats = [(os.stat(f"{bfile_stem}{ext}").st_size, os.stat(f"{bfile_stem}{ext}").st_mtime_ns)
                 for ext in ('.bed', '.bim', '.fam')]
    except OSError:
        return None
    cfg_part = {name: ld_config.get(name) for name in (
        'window', 'window_unit', 'step', 'r2_threshold', 'maf_threshold',
        'allow_extra_chr', 'chr_set',
    )}
    if LD_KEY_INCLUDES_THREADS:
        cfg_part['threads'] = int(threads)
    return _digest('ld', stats, version, '\n'.join(sorted(set(map(str, keep_iid_list)))),
                   _canonical_json(cfg_part))


def rf_select_key(train_x: pd.DataFrame, train_y, rf_config: dict, sample_weight,
                  use_gpu_sklearn: bool) -> str:
    """Key for the merge RF marker selection: the training matrix content
    (row hashes), its column names, ``y``, ``sample_weight``, the full
    ``rf_config``, the cuML switch and the scikit-learn version."""
    import sklearn
    row_hashes = pd.util.hash_pandas_object(train_x, index=False).to_numpy(dtype=np.uint64)
    y_bytes = np.ascontiguousarray(np.asarray(train_y, dtype=np.float64)).tobytes()
    w_bytes = (b'' if sample_weight is None
               else np.ascontiguousarray(np.asarray(sample_weight, dtype=np.float64)).tobytes())
    return _digest('rf', row_hashes.tobytes(), '\x1f'.join(map(str, train_x.columns)),
                   tuple(train_x.shape), y_bytes, w_bytes, _canonical_json(rf_config),
                   bool(use_gpu_sklearn), sklearn.__version__)


# ---------------------------------------------------------------------- #
# storage
# ---------------------------------------------------------------------- #

def memo_get(kind: str, key: str):
    """The stored payload for (kind, key), or None on a miss.

    A corrupt or mismatching file is treated as a miss and logged (the
    caller recomputes and ``memo_put`` overwrites it)."""
    path = _entry_path(kind, key)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            record = json.load(handle)
        if record.get('schema') != _MEMO_SCHEMA or record.get('key') != key or record.get('kind') != kind:
            raise ValueError("record header does not match its file name")
        return record['payload']
    except (OSError, ValueError, KeyError) as exc:
        print(f"[preprocess_memo] NOTE: memo entry '{os.path.basename(path)}' is unreadable "
              f"({type(exc).__name__}: {exc}) - recomputing and overwriting it.")
        return None


def memo_put(kind: str, key: str, payload) -> bool:
    """Atomically store `payload` (JSON-serialisable) under (kind, key).
    Returns False (and disables the memo for this process) if the memo
    directory cannot be written."""
    directory = os.environ.get(MEMO_DIR_ENV)
    if not directory:
        return False
    tmp_path = None
    try:
        os.makedirs(directory, exist_ok=True)
        handle = tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=directory, prefix=f".{kind}_",
                                             suffix='.tmp', delete=False)
        tmp_path = handle.name
        with handle:
            json.dump({'schema': _MEMO_SCHEMA, 'kind': kind, 'key': key, 'payload': payload}, handle)
        os.replace(tmp_path, _entry_path(kind, key))
        return True
    except OSError as exc:
        if tmp_path and os.path.exists(tmp_path):
            with contextlib.suppress(OSError):
                os.remove(tmp_path)
        _disable(f"cannot write to '{directory}' ({type(exc).__name__}: {exc})")
        return False


# ---------------------------------------------------------------------- #
# RF marker selection
# ---------------------------------------------------------------------- #

def memo_select_markers(train_x: pd.DataFrame, train_y, rf_config: dict,
                        sample_weight=None) -> Tuple[List[str], Optional[object]]:
    """Memoised ``select_markers_for_data_driven_network``.

    Same arguments and the same ``(selected_markers, fitted_rf)`` return as
    that function, with ONE documented difference: on a memo hit
    ``fitted_rf`` is ``None`` (no forest is fitted). Every caller in this
    code base discards it (verified when ver4-33 was written).

    When the memo is off or unusable this is a plain pass-through.
    """
    from Preprocess.data_driven_prior_network import select_markers_for_data_driven_network

    with timed_stage('merge_rf'):
        if not memo_enabled():
            note_stage_tag('merge_rf', 'memo off')
            return select_markers_for_data_driven_network(
                train_x, train_y, rf_config, sample_weight=sample_weight)

        use_gpu = bool(get_active_compute_resources().get('use_gpu_sklearn', False))
        key = rf_select_key(train_x, train_y, rf_config, sample_weight, use_gpu)
        cached = memo_get('rf', key)
        if cached is not None and isinstance(cached.get('selected'), list):
            note_stage_tag('merge_rf', 'memo hit')
            print(f"[preprocess_memo] RF marker selection: memo hit ({len(cached['selected'])} "
                  f"marker(s), key {key[:8]}) - forest fit skipped.")
            return list(cached['selected']), None

        note_stage_tag('merge_rf', 'memo miss')
        selected, fitted_rf = select_markers_for_data_driven_network(
            train_x, train_y, rf_config, sample_weight=sample_weight)
        memo_put('rf', key, {'selected': list(selected)})
        return selected, fitted_rf
