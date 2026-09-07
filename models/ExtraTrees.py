"""
models/ExtraTrees.py
=====================
Update ID ver4-5, R2 Stage 8 (blueprint §4.2 Layer 3, "Tier 1" - invariant
I12: scikit-learn is already a pinned hard dependency, so this model adds
NO new dependency).

An ``ExtraTreesRegressor`` sibling of ``models/RF.py`` - same 7-tuple
return contract (invariant I3), same TreeSHAP interaction route via the
shared ``models.interaction_extraction`` module. The only algorithmic
difference from RF is what scikit-learn's own docs describe as Extremely
Randomised Trees: split THRESHOLDS are drawn at random (rather than
searched for the locally-optimal split, as RF does) and, by default, NO
bootstrap resampling is used (every tree sees the full training set,
relying entirely on the random split thresholds - not the resampled rows
- for tree-to-tree diversity). This is a genuinely different bias on
high-dimensional genotype data (blueprint §4.2 Layer 3 table), not merely
a relabelled RF.

`sample_max` (params[2]) keeps the SAME positional meaning RF.py's own
field has, for a consistent HPARAMETERS shape across the tree-ensemble
family (I5-adjacent consistency, not itself an invariant requirement),
but is applied differently here: ExtraTreesRegressor's natural, textbook
form uses NO bootstrap at all (``bootstrap=False``, scikit-learn's own
default for this estimator) - passing a `max_samples` value alongside
`bootstrap=False` raises inside scikit-learn, since the two are
incompatible. So `sample_max=None` (the default) leaves ExtraTrees in
its natural, textbook, no-bootstrap form; setting `sample_max` to any
other value explicitly opts INTO bootstrap resampling at that fraction,
for anyone who wants that comparison instead. This is documented in the
GUI help text (hparam_specs.py) so the difference is visible, not silent.
"""

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import shap

from pipeline_utils import get_active_compute_resources
from models.interaction_extraction import tree_shap_interactions, top_select


def ExtraTrees(train, valid, test, params):

    estimators = params[0]
    features_max = params[1]
    # None (the default) = ExtraTrees' own natural, no-bootstrap form (see
    # module docstring). Any other value opts into bootstrap resampling at
    # that fraction, mirroring RF.py's own field.
    sample_max = params[2]
    max_depth = int(params[3]) if params[3] is not None else None
    min_samples_leaf = params[4]
    get_interaction = params[5]
    shapley_num = params[6]
    threshold = params[7]
    max_interaction_features = params[8]

    # Split the data sets into x and y, exactly as every other model file does.
    train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1]
    if valid.shape[0] != 0:
        valid_x, valid_y = valid.iloc[:, :-1], valid.iloc[:, -1]
    test_x, test_y = test.iloc[:, :-1], test.iloc[:, -1]

    # Requirement 6 precedent (RF.py/SVR.py/KNN.py): n_jobs is resolved
    # from the run's shared compute-resource settings, never hardcoded.
    # ExtraTrees has no cuML GPU-backed equivalent in this codebase (RF.py's
    # own USE_GPU_SKLEARN path is RF-specific), so this model is CPU-only
    # (multi-threaded via n_jobs) regardless of USE_GPU_SKLEARN - stated
    # explicitly rather than left to look like an oversight.
    _resources = get_active_compute_resources()

    _use_bootstrap = sample_max is not None
    if _use_bootstrap:
        et = ExtraTreesRegressor(n_estimators=estimators, random_state=0, bootstrap=True,
                                  max_samples=sample_max, max_features=features_max,
                                  max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                  n_jobs=_resources['n_jobs'])
    else:
        et = ExtraTreesRegressor(n_estimators=estimators, random_state=0, bootstrap=False,
                                  max_features=features_max, max_depth=max_depth,
                                  min_samples_leaf=min_samples_leaf, n_jobs=_resources['n_jobs'])
    et.fit(train_x, train_y)

    predicted = np.asarray(et.predict(test_x)).ravel()
    if valid.shape[0] != 0:
        predicted_valid = np.asarray(et.predict(valid_x)).ravel()
    else:
        predicted_valid = []
    predicted_train = np.asarray(et.predict(train_x)).ravel()
    _feature_importances = np.asarray(et.feature_importances_).ravel()

    # Calculate the metrics
    actual_test = test_y.values.tolist()
    mse = mean_squared_error(actual_test, predicted)
    r = pearsonr(actual_test, predicted)[0]

    # Extract interactions - same two-stage strategy as RF.py (shortlist by
    # main-effect importance first, then pairwise TreeSHAP only among the
    # shortlist) for exactly the same reason: an unshortlisted pairwise SHAP
    # matrix is O(M^2) and infeasible at real marker counts.
    if get_interaction == True:
        shapley_num = min(shapley_num, test_x.shape[0])

        n_features = train_x.shape[1]
        if max_interaction_features != 'all' and n_features > max_interaction_features:
            top_features = train_x.columns[
                np.argsort(_feature_importances)[::-1][:max_interaction_features]
            ]
            if _use_bootstrap:
                et_interaction = ExtraTreesRegressor(n_estimators=estimators, random_state=0,
                                                      bootstrap=True, max_samples=sample_max,
                                                      max_features=features_max, max_depth=max_depth,
                                                      min_samples_leaf=min_samples_leaf,
                                                      n_jobs=_resources['n_jobs'])
            else:
                et_interaction = ExtraTreesRegressor(n_estimators=estimators, random_state=0,
                                                      bootstrap=False, max_features=features_max,
                                                      max_depth=max_depth,
                                                      min_samples_leaf=min_samples_leaf,
                                                      n_jobs=_resources['n_jobs'])
            et_interaction.fit(train_x[top_features], train_y)
            interaction_test_x = test_x[top_features]
        else:
            # 'all' - use every marker, no cap.
            et_interaction = et
            interaction_test_x = test_x

        interaction_sample = tree_shap_interactions(
            et_interaction, shap.sample(interaction_test_x, shapley_num),
            interaction_test_x.columns, n_jobs=_resources['n_jobs'], reduce='abs_then_sum',
        )
        interaction_sample = top_select(interaction_sample, 'percentage', threshold)
    else:
        interaction_sample = pd.DataFrame()

    return r, mse, pd.DataFrame(_feature_importances).T, interaction_sample, predicted, predicted_valid, predicted_train
