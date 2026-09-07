"""
Preprocess/RF_marker_filtering.py
====================================

Optional preprocessing step: keep only the top markers by Random Forest
feature importance - either the top N% (percentage mode) or a fixed count
of M markers (count mode; requirement 1 of the "biological prior network
data-driven upgrade" spec). Meant to run *after* LD pruning when both are
enabled - LD pruning removes markers in high linkage disequilibrium first
(a genotype-only, unsupervised criterion), and even after that the marker
count can still be too large for some models; this step further narrows the
(possibly already LD-pruned) marker set down to the top-importance ones by
a supervised criterion (how much each marker actually helps predict the
phenotype, per a fitted Random Forest).

Exactly like Preprocess.LD_pruning.LD_pruning(), the Random Forest here is
fit ONLY on the training set of the specific (population, phenotype, ratio,
replicate) task it is called for - genomic_prediction.py's GP() calls this
once per task, on that task's own train/valid/test, never on pooled or
test-set data - so the derived marker subset never leaks information from
the validation/test folds.

This module also exposes select_rf_markers(), the "fit a forest on the
training set and rank markers by importance" core shared by two call
sites:
  1. RF_marker_filtering() itself (this file's original, public entry
     point - a straight preprocessing step for whichever model(s) use the
     full/filtered marker pool).
  2. Preprocess/data_driven_prior_network.py's side pipeline for
     models/GAT_biological_prior_knowledge.py's data-driven prior-network
     feature (requirement 2), which reuses the exact same "which markers
     survive RF filtering" decision - same config, same ranking - rather
     than re-implementing it, so "keep the top M/Y% markers" always means
     the same thing everywhere in this codebase.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from pipeline_utils import get_active_compute_resources


def _random_forest_regressor_cls(use_gpu):
    """Return (RandomForestRegressor class, is_gpu) - cuML's GPU-backed
    RandomForestRegressor when `use_gpu` is True AND cuML is importable,
    else scikit-learn's own CPU implementation. Never raises: any failure
    importing cuML (not installed, no compatible GPU/driver, etc.) falls
    back to CPU silently, so a CPU-only node (or a config that never sets
    USE_GPU_SKLEARN) behaves exactly as before this option existed.

    This is an EXACT mirror of models/RF.py's own
    `_random_forest_regressor_cls()` (Phase 2, Requirement 6) - kept as an
    independent per-module copy rather than a shared import, matching this
    codebase's established pattern for small, self-contained helpers (e.g.
    Preprocess/LD_pruning.py's own `_resolve_plink_threads()` docstring:
    "kept as a local copy rather than imported, since the two modules have
    no other dependency on each other").

    Patch 3, Requirement 3 ("GPU+CPU combined should be significantly
    faster than CPU alone, especially when combining filtering with the
    GAT biological prior knowledge model"): this RF fit
    (`select_rf_markers()` below) is exactly the "RF importance filtering"
    step named in the reported slow run - it drives BOTH the general
    RF_marker_filtering() preprocessing step AND (via
    Preprocess/data_driven_prior_network.py's
    select_markers_for_data_driven_network(), which calls
    select_rf_markers() directly) GAT_biological_prior_knowledge's own
    data-driven-merge side pipeline. Before this fix, only models/RF.py's
    own model-fitting RF (a DIFFERENT RandomForestRegressor instance -
    the one that produces predictions/marker effects, not the one that
    picks which markers survive filtering) could use cuML at all; this
    preprocessing-layer RF was hardcoded to scikit-learn/CPU regardless of
    USE_GPU_SKLEARN. Wiring it into the SAME, already-proven mechanism
    (rather than a new one) is the safer fix: it reuses a code path this
    codebase already ships, tests, and falls back from safely.
    """
    if use_gpu:
        try:
            from cuml.ensemble import RandomForestRegressor as CumlRF
            return CumlRF, True
        except Exception as exc:
            print(f"[RF_marker_filtering] USE_GPU_SKLEARN requested but cuML is unavailable "
                  f"({exc}) - falling back to scikit-learn's CPU RandomForestRegressor.")
    return RandomForestRegressor, False


def _resolve_n_jobs(rf_config):
    """Resolve the `n_jobs` value the Random Forest fit below should use.

    An explicit `rf_config['n_jobs']` always wins; otherwise falls back to
    the run's shared compute-resource settings (`N_JOBS` config key, via
    `get_active_compute_resources()`) - the same source
    `Preprocess/plink_io.py` and `Preprocess/LD_pruning.py` already use for
    their own thread counts.

    Previously this was hardcoded to `-1` ("use every core"), regardless
    of how many CPUs were actually reserved for the job. `n_jobs=-1`
    resolves the core count via the OS (`os.cpu_count()` under the hood),
    which - like plink2's own hardware auto-detection - is not cgroup-
    aware: on a node where only a handful of CPUs were reserved (e.g. a
    GPU job, which typically requests far fewer CPUs than a CPU-only job
    of the same size), joblib would still spawn worker processes/threads
    sized to the whole node, causing heavy oversubscription/context-switch
    thrashing rather than a genuine speed-up. This is the RF-filtering
    half of the same root cause that made LD pruning slow under GPU
    allocations (see Preprocess/LD_pruning.py's own docstring for the
    PLINK2 side of it).
    """
    n_jobs = rf_config.get('n_jobs')
    if n_jobs is not None:
        return int(n_jobs)
    return get_active_compute_resources()['n_jobs']


def _resolve_n_keep(rf_config, n_markers):
    """How many markers `rf_config` says to keep, out of `n_markers` total.

    Parameters
    ----------
    rf_config : dict
        - 'mode' : 'percent' (default) or 'count'.
        - 'top_ratio' : float in (0, 1] - required when mode == 'percent'.
          Fraction of markers to keep (e.g. 0.2 keeps the top 20%).
          Converting a user-facing percentage (e.g. "20") to this ratio
          (0.2) is the GUI's job, not this function's.
        - 'top_n' : int >= 1 - required when mode == 'count'. Absolute
          number of markers to keep (requirement 1's "M markers" option).
          Clamped to n_markers if larger (keeping everything is a valid,
          harmless outcome, not an error - mirrors 'top_ratio' == 1).

    Returns
    -------
    int, the number of markers to keep (at least 1, at most n_markers).
    """
    mode = rf_config.get('mode', 'percent')
    if mode == 'percent':
        top_ratio = rf_config.get('top_ratio')
        if top_ratio is None:
            raise ValueError("rf_config['top_ratio'] is required when rf_config['mode'] == 'percent'.")
        if not (0 < top_ratio <= 1):
            raise ValueError(f"rf_config['top_ratio'] must be in (0, 1], got {top_ratio!r}")
        return min(n_markers, max(1, int(np.ceil(n_markers * top_ratio))))
    elif mode == 'count':
        top_n = rf_config.get('top_n')
        if top_n is None:
            raise ValueError("rf_config['top_n'] is required when rf_config['mode'] == 'count'.")
        top_n = int(top_n)
        if top_n < 1:
            raise ValueError(f"rf_config['top_n'] must be >= 1, got {top_n!r}")
        return min(n_markers, top_n)
    else:
        raise ValueError(f"rf_config['mode'] must be 'percent' or 'count', got {mode!r}")


def select_rf_markers(train_x, train_y, rf_config):
    """Fit a Random Forest on (train_x, train_y) and return the top markers
    by importance, per `rf_config` (see `_resolve_n_keep` for the
    percent/count mode choice), plus the fitted model itself so a caller
    (e.g. the data-driven prior-network side pipeline) can reuse its
    hyperparameter configuration - and, when the caller genuinely wants the
    exact same feature space, the fitted model - without re-fitting from
    scratch on the same data.

    Note: when `n_keep >= n_markers` (nothing to actually filter), a forest
    is still fit here so `fitted_rf` is always a real, usable model for
    callers that need one (unlike RF_marker_filtering() below, which skips
    fitting entirely in that case since it has no model-reuse need of its
    own).

    Returns
    -------
    (top_markers, fitted_rf, n_keep) :
        top_markers : list of str, in ORIGINAL column order (not sorted by
            importance) - a stable, position-based subset.
        fitted_rf : sklearn.ensemble.RandomForestRegressor (or, when
            USE_GPU_SKLEARN resolves True and cuML is available,
            cuml.ensemble.RandomForestRegressor - see
            _random_forest_regressor_cls() above), fit on (train_x,
            train_y) with every marker as a feature. Every caller in this
            codebase currently uses this only for its hyperparameters
            (never for scikit-learn-specific attributes/methods such as
            passing it into shap.TreeExplainer, which cuML's forest does
            not support - see Preprocess/data_driven_prior_network.py's
            compute_data_driven_interactions(), which always fits its OWN,
            separate, CPU-only forest for exactly this reason), so this
            substitution is safe; a caller with a NEW use that genuinely
            needs scikit-learn-specific internals should fit its own
            forest rather than assume this one's type.
        n_keep : int, how many markers were kept (== len(top_markers)).
    """
    n_markers = train_x.shape[1]
    if n_markers == 0:
        raise ValueError("select_rf_markers: train_x has no marker columns.")
    n_keep = _resolve_n_keep(rf_config, n_markers)

    # Patch 3, Requirement 3: an optional cuML GPU backend, resolved from
    # the run's shared compute-resource settings - the SAME
    # USE_GPU_SKLEARN switch models/RF.py's own model-fitting forest
    # already uses (see _random_forest_regressor_cls()'s docstring above).
    # A run that never sets USE_GPU_SKLEARN (or has no GPU/no cuML)
    # resolves use_gpu=False here and gets EXACTLY today's scikit-learn
    # CPU behaviour - no change for that case.
    _resources = get_active_compute_resources()
    _rf_cls, _is_gpu = _random_forest_regressor_cls(_resources.get('use_gpu_sklearn', False))

    if _is_gpu:
        # cuML's RandomForestRegressor has a narrower constructor surface
        # than scikit-learn's (no n_jobs - it's GPU-resident, not
        # thread-parallel) - mirrors models/RF.py's own GPU branch exactly.
        rf = _rf_cls(
            n_estimators=rf_config.get('n_estimators', 200),
            max_depth=rf_config.get('max_depth', None),
            max_features=rf_config.get('max_features', 'sqrt'),
            min_samples_leaf=rf_config.get('min_samples_leaf', 1),
            random_state=rf_config.get('random_state', 0),
        )
    else:
        rf = _rf_cls(
            n_estimators=rf_config.get('n_estimators', 200),
            max_depth=rf_config.get('max_depth', None),
            max_features=rf_config.get('max_features', 'sqrt'),
            min_samples_leaf=rf_config.get('min_samples_leaf', 1),
            random_state=rf_config.get('random_state', 0),
            n_jobs=_resolve_n_jobs(rf_config),
        )
    rf.fit(train_x, train_y)

    # cuML's `.feature_importances_` returns a cuDF/cupy array rather than
    # a plain numpy one - np.argsort/np.asarray both coerce it correctly
    # via cuML's own __array__ interop, so this line is unchanged for
    # either backend.
    top_idx = np.argsort(np.asarray(rf.feature_importances_))[::-1][:n_keep]
    top_marker_set = set(train_x.columns[top_idx])
    # Preserve original column order among the kept markers (see docstring).
    top_markers = [c for c in train_x.columns if c in top_marker_set]

    return top_markers, rf, n_keep


def RF_marker_filtering(train, valid, test, rf_config, return_model=False):
    """Keep the top markers - by importance from a Random Forest fit on
    `train` only - in `train`, `valid`, and `test` alike.

    Parameters
    ----------
    train, valid, test : pd.DataFrame
        Same convention as every model/preprocessing function in this
        codebase: all columns except the last are genomic markers, the last
        column is the phenotype. `valid` may be an empty DataFrame (no
        validation split requested for this task) - passed through
        unfiltered in that case, since there's nothing to subset.
    rf_config : dict
        See `_resolve_n_keep` for 'mode'/'top_ratio'/'top_n'. Also:
        - 'n_estimators' : int, number of trees (default 200)
        - 'max_depth' : int or None (default None)
        - 'max_features' : passed straight to RandomForestRegressor
          (default 'sqrt')
        - 'min_samples_leaf' : int (default 1)
        - 'random_state' : int, for reproducibility (default 0)
        - 'n_jobs' : int, optional. Forwarded to RandomForestRegressor's
          own `n_jobs`. Defaults to the run's shared compute-resource
          setting (`N_JOBS` config key) rather than unconditionally `-1` -
          see `_resolve_n_jobs()`'s own docstring for why.
    return_model : bool, default False
        If True, additionally return the fitted RandomForestRegressor and
        the list of kept marker names as a 5-tuple, instead of the plain
        3-tuple every existing caller in this codebase already expects.
        Every existing call site passes this as False (the default), so
        this is purely additive - see Preprocess/data_driven_prior_network.py
        for the one caller that sets it True, to fit-once-reuse-twice
        rather than re-fitting an equivalent forest a second time.

    Returns
    -------
    (train, valid, test) : pd.DataFrame
        (default; return_model=False) Same row counts as the inputs,
        narrowed to the top-importance marker columns plus the original
        phenotype column (in that order). Column order among the kept
        markers matches their ORIGINAL order (not sorted by importance).
    (train, valid, test, fitted_rf, top_markers) :
        (return_model=True) As above, plus the fitted RandomForestRegressor
        (fit on every marker, i.e. BEFORE narrowing to top_markers - since
        that is what "the trained RF used in RF filtering" actually refers
        to) and the list of kept marker names. `fitted_rf` is None and
        `top_markers` is every column of `train` when nothing needed
        filtering (n_keep >= n_markers) - see the note on that case below.
    """
    train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1]
    n_markers = train_x.shape[1]
    if n_markers == 0:
        raise ValueError("RF_marker_filtering: train has no marker columns to filter.")
    n_keep = _resolve_n_keep(rf_config, n_markers)

    if n_keep >= n_markers:
        # Nothing to filter (e.g. top_ratio == 1, top_n >= n_markers, or so
        # few markers remain after LD pruning that "top N%"/"top M" already
        # covers all of them) - skip fitting a forest entirely, exactly as
        # before. return_model callers still get a well-formed 5-tuple: no
        # forest was fit (nothing to reuse), and every marker "survived".
        train_out = train.reset_index(drop=True)
        valid_out = valid.reset_index(drop=True) if valid.shape[0] != 0 else valid
        test_out = test.reset_index(drop=True)
        if return_model:
            return train_out, valid_out, test_out, None, list(train_x.columns)
        return train_out, valid_out, test_out

    top_markers, rf, _ = select_rf_markers(train_x, train_y, rf_config)

    phenotype_col = train.columns[-1]
    keep_cols = top_markers + [phenotype_col]

    train_out = train[keep_cols].reset_index(drop=True)
    test_out = test[keep_cols].reset_index(drop=True)
    valid_out = valid[keep_cols].reset_index(drop=True) if valid.shape[0] != 0 else valid

    if return_model:
        return train_out, valid_out, test_out, rf, top_markers
    return train_out, valid_out, test_out
