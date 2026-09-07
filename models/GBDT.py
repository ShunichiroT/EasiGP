"""
models/GBDT.py
================
Update ID ver4-5, R2 Stage 8 (blueprint §4.2 Layer 3, "Tier 1" - invariant
I12: scikit-learn is already a pinned hard dependency).

``HistGradientBoostingRegressor`` - verified directly (not assumed) to
support ``shap.TreeExplainer.shap_interaction_values`` on this codebase's
pinned-equivalent ``shap``/``sklearn`` versions during Phase 2 preparation
(blueprint Appendix B1/RK-4's own pre-check). A genuinely different
inductive bias from RF/ExtraTrees: sequential, error-correcting boosted
trees rather than an averaged bagged/randomised ensemble.

Two structural differences from ``models/RF.py``/``models/ExtraTrees.py``,
both because ``HistGradientBoostingRegressor`` has NO ``feature_
importances_`` attribute (verified directly - scikit-learn simply does
not expose one for this estimator, unlike every other tree ensemble in
this codebase):

  1. Its HYPERPARAMETER SURFACE below is HistGradientBoostingRegressor's
     OWN natural knobs (``max_iter``/``learning_rate``/``max_leaf_nodes``/
     ``l2_regularization``, ...), not a re-shaped copy of RF's (there is
     no bagging/column-subsampling concept to expose here - boosting, not
     bagging).
  2. Its marker EFFECT (main-effect) output cannot come from a free,
     already-fitted attribute the way RF's/ExtraTrees' can. It is instead
     read from ``sklearn.inspection.permutation_importance`` on a
     lightweight refit restricted to a correlation-shortlisted set of
     markers (capped at `_EFFECT_SHORTLIST_CAP`, not user-configurable
     this pass - a disclosed scope simplification, see the ver4-5 Change
     Summary §7) - mirroring RF.py's own "refit a lightweight model on a
     shortlist" pattern, applied to the effect step instead of (only) the
     interaction step. Interactions still use the SAME shared TreeSHAP
     route (``models.interaction_extraction``) every other tree model
     here uses.
"""

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import shap

from pipeline_utils import get_active_compute_resources
from models.interaction_extraction import tree_shap_interactions, top_select

# Not user-configurable this pass (disclosed scope simplification, ver4-5
# Change Summary §7) - bounds the cost of the permutation-importance effect
# step on datasets with thousands of markers, mirroring the SAME order-of-
# magnitude cap RF.py/ExtraTrees.py already default their OWN interaction
# shortlist to (500).
_EFFECT_SHORTLIST_CAP = 500
_EFFECT_PERMUTATION_REPEATS = 5


def GBDT(train, valid, test, params):

    max_iter = params[0]
    learning_rate = params[1]
    max_depth = int(params[2]) if params[2] is not None else None
    max_leaf_nodes = int(params[3]) if params[3] is not None else None
    min_samples_leaf = params[4]
    l2_regularization = params[5]
    get_interaction = params[6]
    shapley_num = params[7]
    threshold = params[8]
    max_interaction_features = params[9]

    train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1]
    if valid.shape[0] != 0:
        valid_x, valid_y = valid.iloc[:, :-1], valid.iloc[:, -1]
    test_x, test_y = test.iloc[:, :-1], test.iloc[:, -1]

    # Requirement 6 precedent: n_jobs (HistGradientBoostingRegressor
    # threads its own histogram-building step) resolved from the run's
    # shared compute-resource settings. No cuML GPU-backed equivalent
    # exists for this estimator in this codebase - CPU-only, stated
    # explicitly rather than left to look like an oversight.
    _resources = get_active_compute_resources()

    gbdt = HistGradientBoostingRegressor(
        max_iter=max_iter, learning_rate=learning_rate, max_depth=max_depth,
        max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization, random_state=0,
    )
    gbdt.fit(train_x, train_y)

    predicted = np.asarray(gbdt.predict(test_x)).ravel()
    if valid.shape[0] != 0:
        predicted_valid = np.asarray(gbdt.predict(valid_x)).ravel()
    else:
        predicted_valid = []
    predicted_train = np.asarray(gbdt.predict(train_x)).ravel()

    # Calculate the metrics
    actual_test = test_y.values.tolist()
    mse = mean_squared_error(actual_test, predicted)
    r = pearsonr(actual_test, predicted)[0]

    # Marker effect - see module docstring: HistGradientBoostingRegressor
    # has no free feature_importances_, so this is read from permutation
    # importance on a lightweight refit restricted to a correlation-
    # shortlisted marker set, then reassembled into a full-width vector
    # (0 for every marker outside the shortlist) - the SAME reassembly
    # convention models/SVR.py/models/KNN.py already use for their own
    # shortlisted Shapley effect output, since genomic_prediction.py
    # assigns column names positionally from the FULL marker list
    # regardless of how many markers were actually scored.
    n_features = train_x.shape[1]
    if n_features > _EFFECT_SHORTLIST_CAP:
        correlations = train_x.corrwith(train_y).abs().fillna(0)
        effect_features = correlations.sort_values(ascending=False).index[:_EFFECT_SHORTLIST_CAP]
        gbdt_effect = HistGradientBoostingRegressor(
            max_iter=max_iter, learning_rate=learning_rate, max_depth=max_depth,
            max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization, random_state=0,
        )
        gbdt_effect.fit(train_x[effect_features], train_y)
        effect_test_x = test_x[effect_features]
    else:
        gbdt_effect = gbdt
        effect_features = train_x.columns
        effect_test_x = test_x

    _perm = permutation_importance(
        gbdt_effect, effect_test_x, test_y, n_repeats=_EFFECT_PERMUTATION_REPEATS,
        random_state=0, n_jobs=_resources['n_jobs'],
    )
    effect_full = pd.Series(0.0, index=train_x.columns)
    effect_full.loc[effect_features] = _perm.importances_mean
    effect = pd.DataFrame(effect_full).T

    # Extract interactions - same shared TreeSHAP route as RF.py/ExtraTrees.py.
    if get_interaction == True:
        shapley_num = min(shapley_num, test_x.shape[0])

        _feature_importances = _perm.importances_mean  # aligned with effect_features
        if max_interaction_features != 'all' and len(effect_features) > max_interaction_features:
            top_features = pd.Index(effect_features)[
                np.argsort(_feature_importances)[::-1][:max_interaction_features]
            ]
            gbdt_interaction = HistGradientBoostingRegressor(
                max_iter=max_iter, learning_rate=learning_rate, max_depth=max_depth,
                max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                l2_regularization=l2_regularization, random_state=0,
            )
            gbdt_interaction.fit(train_x[top_features], train_y)
            interaction_test_x = test_x[top_features]
        elif max_interaction_features != 'all' and n_features > max_interaction_features:
            # More markers overall than max_interaction_features, but the
            # effect shortlist above was ALSO capped below that same
            # number (_EFFECT_SHORTLIST_CAP < max_interaction_features) -
            # re-rank from the full marker set via the same cheap
            # correlation proxy RF.py/ExtraTrees.py's own importance-based
            # shortlist mirrors, since gbdt_effect's own shortlist is too
            # narrow to reuse here.
            correlations = train_x.corrwith(train_y).abs().fillna(0)
            top_features = correlations.sort_values(ascending=False).index[:max_interaction_features]
            gbdt_interaction = HistGradientBoostingRegressor(
                max_iter=max_iter, learning_rate=learning_rate, max_depth=max_depth,
                max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                l2_regularization=l2_regularization, random_state=0,
            )
            gbdt_interaction.fit(train_x[top_features], train_y)
            interaction_test_x = test_x[top_features]
        else:
            # 'all' - use every marker already in the effect shortlist
            # (which itself is 'every marker' whenever n_features <=
            # _EFFECT_SHORTLIST_CAP).
            gbdt_interaction = gbdt_effect
            interaction_test_x = effect_test_x

        interaction_sample = tree_shap_interactions(
            gbdt_interaction, shap.sample(interaction_test_x, shapley_num),
            interaction_test_x.columns, n_jobs=_resources['n_jobs'], reduce='abs_then_sum',
        )
        interaction_sample = top_select(interaction_sample, 'percentage', threshold)
    else:
        interaction_sample = pd.DataFrame()

    return r, mse, effect, interaction_sample, predicted, predicted_valid, predicted_train
