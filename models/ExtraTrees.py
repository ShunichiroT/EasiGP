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

from pipeline_utils import get_active_compute_resources, shapiq_marker_effects
from models.interaction_extraction import (
    tree_shap_interactions, tree_shap_main_effects, top_select,
    should_use_effect_for_interaction_shortlist,
)
from models.tree_interaction_routes import tree_model_interactions


def ExtraTrees(train, valid, test, params, *, sample_weight_ctx=None):

    estimators = params[0]
    features_max = params[1]
    # None (the default) = ExtraTrees' own natural, no-bootstrap form (see
    # module docstring). Any other value opts into bootstrap resampling at
    # that fraction, mirroring RF.py's own field.
    sample_max = params[2]
    max_depth = int(params[3]) if params[3] is not None else None
    params_max_depth = max_depth
    min_samples_leaf = params[4]
    get_interaction = params[5]
    shapley_num = params[6]
    threshold = params[7]
    max_interaction_features = params[8]
    # Requirements3.md item 2 - see models/RF.py's identical field.
    effect_method = params[9] if len(params) > 9 else 'feature_importance'
    effect_shapley_num = params[10] if len(params) > 10 else 30
    # Requirements.md items 2/3 (this update) - see models/RF.py's
    # identical fields for the full rationale. Appended, read defensively
    # (I5) - a config predating these four fields keeps this model's
    # previous, always-on marker-effect behaviour exactly.
    get_effect = params[11] if len(params) > 11 else True
    effect_max_shapiq_features = params[12] if len(params) > 12 else 500
    effect_shapiq_background = params[13] if len(params) > 13 else 50
    effect_shapiq_budget = params[14] if len(params) > 14 else 200
    # This update - see models/RF.py's identical field for the full
    # rationale. Appended, read defensively (I5).
    use_effect_for_interaction_shortlist = params[15] if len(params) > 15 else True
    # RF-style 'Marker interaction method' block (hparam_specs.
    # _append_rf_style_interaction_fields) - appended, read defensively
    # (I5): a config predating these eight fields resolves to
    # 'pairwise_shap', i.e. this model's previous exact-TreeSHAP route,
    # unchanged (I11). Same meanings as models/RF.py's params[9..15, 23].
    interaction_method = params[16] if len(params) > 16 else 'pairwise_shap'
    interaction_screen = params[17] if len(params) > 17 else 'off'
    interaction_screen_top = params[18] if len(params) > 18 else 2.0
    interaction_grid_resolution = params[19] if len(params) > 19 else 3
    interaction_h_background = params[20] if len(params) > 20 else None
    interaction_shapiq_orders = params[21] if len(params) > 21 else '2'
    interaction_shapiq_index = params[22] if len(params) > 22 else 'k-SII'
    interaction_shapiq_budget = params[23] if len(params) > 23 else 4096

    # Split the data sets into x and y, exactly as every other model file does.
    train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1]
    if valid.shape[0] != 0:
        valid_x, valid_y = valid.iloc[:, :-1], valid.iloc[:, -1]
    test_x, test_y = test.iloc[:, :-1], test.iloc[:, -1]
    # ver4-25, R8 - see models/RF.py's identical comment.
    _w_fit = sample_weight_ctx.w_fit if sample_weight_ctx is not None else None

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
    et.fit(train_x, train_y, sample_weight=_w_fit)

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

    # This update: marker-EFFECT computation (below) is hoisted ahead of
    # the interaction-shortlist step - see models/RF.py's identical
    # comment for the full rationale (pure reordering, I11 either way).
    def _tree_shap_effect():
        _effect_num = min(effect_shapley_num, test_x.shape[0])
        effect_sample = shap.sample(test_x, _effect_num) if _effect_num > 0 else test_x.iloc[:0]
        effect_scores = tree_shap_main_effects(et, effect_sample, train_x.columns)
        return pd.DataFrame([effect_scores], columns=train_x.columns)

    # Requirements.md item 3 - see models/RF.py's identical gating.
    if not get_effect:
        effect = pd.DataFrame()
    elif effect_method == 'shap':
        try:
            effect = _tree_shap_effect()
        except Exception as exc:
            print(f"[ExtraTrees] NOTE: Shapley marker-effect computation failed ({exc!r}) - "
                  f"falling back to this model's own feature_importances_ for this task instead.")
            effect = pd.DataFrame(_feature_importances).T
    elif effect_method == 'shapiq':
        # Requirements.md item 2 - see models/RF.py's identical 'shapiq'
        # branch for the full rationale. No GPU variant exists for this
        # model, so - unlike RF.py's version of this branch - there is no
        # cuML refit consideration either way.
        try:
            n_features = train_x.shape[1]
            if effect_max_shapiq_features != 'all' and n_features > effect_max_shapiq_features:
                top_features = train_x.columns[
                    np.argsort(_feature_importances)[::-1][:effect_max_shapiq_features]
                ]
                if _use_bootstrap:
                    et_shapiq_effect = ExtraTreesRegressor(
                        n_estimators=estimators, random_state=0, bootstrap=True,
                        max_samples=sample_max, max_features=features_max, max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf, n_jobs=_resources['n_jobs'],
                    )
                else:
                    et_shapiq_effect = ExtraTreesRegressor(
                        n_estimators=estimators, random_state=0, bootstrap=False,
                        max_features=features_max, max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf, n_jobs=_resources['n_jobs'],
                    )
                et_shapiq_effect.fit(train_x[top_features], train_y, sample_weight=_w_fit)
                effect_train_x = train_x[top_features]
                effect_test_x = test_x[top_features]
            else:
                et_shapiq_effect = et
                top_features = train_x.columns
                effect_train_x = train_x
                effect_test_x = test_x

            _effect_num = min(effect_shapley_num, effect_test_x.shape[0])
            effect_scores = shapiq_marker_effects(
                et_shapiq_effect.predict, effect_train_x, shap.sample(effect_test_x, _effect_num),
                background_size=min(effect_shapiq_background, effect_train_x.shape[0]),
                budget=effect_shapiq_budget,
            )
            effect_full = pd.Series(0.0, index=train_x.columns)
            effect_full.loc[top_features] = effect_scores
            effect = pd.DataFrame(effect_full).T
        except Exception as exc:
            print(f"[ExtraTrees] NOTE: shapiq marker-effect computation failed ({exc!r}) - "
                  f"falling back to this model's own exact TreeSHAP ('shap') route for this task "
                  f"instead. ('shapiq' is an optional dependency - 'pip install shapiq' if not "
                  f"yet installed; this may also mean this shapiq version's API differs from "
                  f"what this codebase expects.)")
            try:
                effect = _tree_shap_effect()
            except Exception as exc2:
                print(f"[ExtraTrees] NOTE: TreeSHAP fallback also failed ({exc2!r}) - falling "
                      f"back to this model's own feature_importances_ for this task instead.")
                effect = pd.DataFrame(_feature_importances).T
    else:
        effect = pd.DataFrame(_feature_importances).T

    # This update: how many markers this task's own marker-effect step
    # above actually covers - see models/RF.py's identical comment for
    # the full rationale.
    if not get_effect:
        _effect_marker_cap = None
    elif effect_method == 'shapiq':
        _effect_marker_cap = effect_max_shapiq_features
    else:
        _effect_marker_cap = 'all'
    _use_effect_shortlist = should_use_effect_for_interaction_shortlist(
        use_effect_for_interaction_shortlist, max_interaction_features, _effect_marker_cap,
    )
    _interaction_importance = (
        np.asarray(effect.iloc[0]).ravel() if _use_effect_shortlist else _feature_importances
    )

    # Extract interactions - same two-stage strategy as RF.py (shortlist by
    # main-effect importance first, then explain only the shortlist) for
    # exactly the same reason: an unshortlisted pairwise matrix is O(M^2)
    # and infeasible at real marker counts. The three selectable routes
    # (pairwise_shap / friedman_h / shapiq_tree) now mirror RF.py's own
    # 'Marker interaction method' field; they live in
    # models/tree_interaction_routes.py, shared with XGBoost.py, and the
    # default 'pairwise_shap' reproduces this model's previous behaviour
    # exactly.
    if get_interaction == True:
        shapley_num = min(shapley_num, test_x.shape[0])

        def _refit_extratrees(columns, max_depth=None, n_estimators=None):
            _kw = dict(
                n_estimators=estimators if n_estimators is None else n_estimators,
                random_state=0, max_features=features_max,
                max_depth=(params_max_depth if max_depth is None else max_depth),
                min_samples_leaf=min_samples_leaf, n_jobs=_resources['n_jobs'],
            )
            if _use_bootstrap:
                _m = ExtraTreesRegressor(bootstrap=True, max_samples=sample_max, **_kw)
            else:
                _m = ExtraTreesRegressor(bootstrap=False, **_kw)
            _m.fit(train_x[columns], train_y, sample_weight=_w_fit)
            return _m

        interaction_sample = tree_model_interactions(
            tag='ExtraTrees', fitted_model=et, refit_fn=_refit_extratrees,
            train_x=train_x, test_x=test_x, importance=_interaction_importance,
            interaction_method=interaction_method, shapley_num=shapley_num,
            threshold=threshold, max_interaction_features=max_interaction_features,
            screen=interaction_screen, screen_top=interaction_screen_top,
            grid_resolution=interaction_grid_resolution, h_background=interaction_h_background,
            shapiq_orders=interaction_shapiq_orders, shapiq_index=interaction_shapiq_index,
            shapiq_budget=interaction_shapiq_budget,
            n_estimators=estimators, effective_max_depth=params_max_depth, resources=_resources,
        )
    else:
        interaction_sample = pd.DataFrame()

    # 'effect' was already computed above (hoisted ahead of the
    # interaction-shortlist step) - nothing left to do here but return it.

    return r, mse, effect, interaction_sample, predicted, predicted_valid, predicted_train
