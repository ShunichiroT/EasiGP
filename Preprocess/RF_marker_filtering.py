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
        fitted_rf : sklearn.ensemble.RandomForestRegressor, fit on
            (train_x, train_y) with every marker as a feature.
        n_keep : int, how many markers were kept (== len(top_markers)).
    """
    n_markers = train_x.shape[1]
    if n_markers == 0:
        raise ValueError("select_rf_markers: train_x has no marker columns.")
    n_keep = _resolve_n_keep(rf_config, n_markers)

    rf = RandomForestRegressor(
        n_estimators=rf_config.get('n_estimators', 200),
        max_depth=rf_config.get('max_depth', None),
        max_features=rf_config.get('max_features', 'sqrt'),
        min_samples_leaf=rf_config.get('min_samples_leaf', 1),
        random_state=rf_config.get('random_state', 0),
        n_jobs=-1,
    )
    rf.fit(train_x, train_y)

    top_idx = np.argsort(rf.feature_importances_)[::-1][:n_keep]
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
