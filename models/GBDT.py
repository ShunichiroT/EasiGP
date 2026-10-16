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

from pipeline_utils import get_active_compute_resources, shapiq_marker_effects
from models.interaction_extraction import tree_shap_interactions, top_select

# Not user-configurable this pass (disclosed scope simplification, ver4-5
# Change Summary §7) - bounds the cost of the permutation-importance effect
# step on datasets with thousands of markers, mirroring the SAME order-of-
# magnitude cap RF.py/ExtraTrees.py already default their OWN interaction
# shortlist to (500).
_EFFECT_SHORTLIST_CAP = 500
_EFFECT_PERMUTATION_REPEATS = 5
# Requirements.md item 2 (this update): how many test individuals the new
# 'shapiq' marker-effect route (below) explains - not user-configurable,
# same disclosed-scope-simplification convention as the two constants
# above (GBDT is engine-only/hidden from the GUI - see hparam_specs.py's
# own comment on this model's new 'Marker effect method' field).
_EFFECT_SHAPIQ_ROW_COUNT = 30


def GBDT(train, valid, test, params, *, sample_weight_ctx=None):

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
    # Requirements.md item 2 (this update) - see models/RF.py's identical
    # fields (and hparam_specs.py's own comment on this model's new
    # 'Marker effect method' field for why there is no master 'Return
    # marker effect?' toggle here, unlike RF/ExtraTrees/XGBoost). Appended,
    # read defensively (I5) - a config predating these four fields keeps
    # this model's previous, permutation-importance-only behaviour exactly.
    effect_method = params[10] if len(params) > 10 else 'permutation_importance'
    effect_max_shapiq_features = params[11] if len(params) > 11 else 500
    effect_shapiq_background = params[12] if len(params) > 12 else 50
    effect_shapiq_budget = params[13] if len(params) > 13 else 200

    train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1]
    if valid.shape[0] != 0:
        valid_x, valid_y = valid.iloc[:, :-1], valid.iloc[:, -1]
    test_x, test_y = test.iloc[:, :-1], test.iloc[:, -1]
    # ver4-25, R8 - see models/RF.py's identical comment.
    _w_fit = sample_weight_ctx.w_fit if sample_weight_ctx is not None else None

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
    gbdt.fit(train_x, train_y, sample_weight=_w_fit)

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
        gbdt_effect.fit(train_x[effect_features], train_y, sample_weight=_w_fit)
        effect_test_x = test_x[effect_features]
    else:
        gbdt_effect = gbdt
        effect_features = train_x.columns
        effect_test_x = test_x

    _perm = permutation_importance(
        gbdt_effect, effect_test_x, test_y, n_repeats=_EFFECT_PERMUTATION_REPEATS,
        random_state=0, n_jobs=_resources['n_jobs'],
    )

    def _permutation_importance_effect():
        effect_full = pd.Series(0.0, index=train_x.columns)
        effect_full.loc[effect_features] = _perm.importances_mean
        return pd.DataFrame(effect_full).T

    # Requirements.md item 2 (this update): 'shapiq' - a model-agnostic
    # order-1 Shapley marker effect (pipeline_utils.shapiq_marker_
    # effects()), a genuinely DIFFERENT quantity from permutation
    # importance (not merely an exact/approximate variant of the same
    # thing, unlike RF/ExtraTrees/XGBoost's own 'shap' vs 'shapiq', which
    # are index-equivalent) - optionally replaces the RETURNED effect
    # vector below. The permutation-importance step above is ALWAYS
    # computed regardless of this choice: the interaction section further
    # down reuses its own shortlist/ranking (_perm.importances_mean,
    # effect_features, gbdt_effect) as its importance proxy, exactly as
    # before this field existed (I11) - only the vector actually RETURNED
    # as this task's marker effect changes. Falls back to permutation
    # importance (with a printed NOTE) on any shapiq failure, never
    # letting an optional-dependency issue fail this task.
    if effect_method == 'shapiq':
        try:
            n_shapiq_features = len(effect_features)
            if effect_max_shapiq_features != 'all' and n_shapiq_features > effect_max_shapiq_features:
                shapiq_features = pd.Index(effect_features)[
                    np.argsort(_perm.importances_mean)[::-1][:effect_max_shapiq_features]
                ]
                gbdt_shapiq_effect = HistGradientBoostingRegressor(
                    max_iter=max_iter, learning_rate=learning_rate, max_depth=max_depth,
                    max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                    l2_regularization=l2_regularization, random_state=0,
                )
                gbdt_shapiq_effect.fit(train_x[shapiq_features], train_y, sample_weight=_w_fit)
                shapiq_train_x = train_x[shapiq_features]
                shapiq_test_x = test_x[shapiq_features]
            else:
                # Reuse the already-fitted permutation-importance model/
                # shortlist directly - no extra refit needed.
                gbdt_shapiq_effect = gbdt_effect
                shapiq_features = effect_features
                shapiq_train_x = train_x[effect_features]
                shapiq_test_x = effect_test_x

            effect_scores = shapiq_marker_effects(
                gbdt_shapiq_effect.predict, shapiq_train_x,
                shap.sample(shapiq_test_x, min(_EFFECT_SHAPIQ_ROW_COUNT, shapiq_test_x.shape[0])),
                background_size=min(effect_shapiq_background, shapiq_train_x.shape[0]),
                budget=effect_shapiq_budget,
            )
            effect_full = pd.Series(0.0, index=train_x.columns)
            effect_full.loc[shapiq_features] = effect_scores
            effect = pd.DataFrame(effect_full).T
        except Exception as exc:
            print(f"[GBDT] NOTE: shapiq marker-effect computation failed ({exc!r}) - falling "
                  f"back to this model's own permutation-importance route for this task instead. "
                  f"('shapiq' is an optional dependency - 'pip install shapiq' if not yet "
                  f"installed; this may also mean this shapiq version's API differs from what "
                  f"this codebase expects.)")
            effect = _permutation_importance_effect()
    else:
        effect = _permutation_importance_effect()

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
            gbdt_interaction.fit(train_x[top_features], train_y, sample_weight=_w_fit)
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
            gbdt_interaction.fit(train_x[top_features], train_y, sample_weight=_w_fit)
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
