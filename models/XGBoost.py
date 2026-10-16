"""
models/XGBoost.py
===================
Update ID ver4-5, R2 Stage 11 (blueprint §4.2 Layer 3, "Tier 2" - decision
D2: gated behind an availability probe, never a hard import at module
scope, invariant I12).

``xgboost`` is an OPTIONAL dependency. This module is only ever REACHED
by genomic_prediction.py's dispatch after ``model_registry.is_available
('XGBoost')`` has already confirmed the package imports - see that
module's own ``is_available()``/``optional_dependency_for()`` and
``GP()``'s config-validation-time fail-fast check (``MODEL_AVAILABILITY_
STRICT``). The import below is still wrapped defensively (never a bare
top-of-file ``import xgboost``), so this file can be imported (e.g. by
static analysis, or a headless script that imports every models/*.py
file) even on a machine without ``xgboost`` installed, without itself
raising - only actually CALLING ``XGBoost()`` without the package
installed raises, with an explicit, actionable message.

Interactions reuse the SAME shared TreeSHAP route
(``models.interaction_extraction.tree_shap_interactions``) every other
tree model in this codebase uses - ``shap.TreeExplainer`` supports
``xgboost.XGBRegressor`` natively (verified directly during Phase 2
preparation, not assumed - see the ver4-5 Change Summary §10), so no
XGBoost-specific native-``pred_interactions``-argument code path is
needed here; the blueprint's own "native pred_interactions=True" framing
(§4.2 Layer 3 table) describes what XGBoost computes UNDER THE HOOD to
answer a TreeSHAP query, not a separate API this file needs to call.
"""

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


def _xgb_regressor_cls():
    """Lazily import xgboost.XGBRegressor. Raises a clear, actionable
    ImportError (never a bare xgboost ImportError with no install
    instructions) if the package is not installed - this should never
    actually fire in practice, since GP()'s own config-validation-time
    check (MODEL_AVAILABILITY_STRICT) is meant to catch this BEFORE any
    model call, but this file must still be safe to import on a machine
    without xgboost (see module docstring), so the import itself is
    deferred to here, not module scope."""
    try:
        from xgboost import XGBRegressor
        return XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "models/XGBoost.py: the 'XGBoost' model was selected but the optional "
            "'xgboost' package is not installed in this environment. Install it with "
            "'pip install xgboost' (or 'conda install -c conda-forge xgboost'), or remove "
            "'XGBoost' from MODEL. This should have been caught earlier by GP()'s own "
            "config-validation check (MODEL_AVAILABILITY_STRICT) - seeing this error "
            "directly means that check was bypassed."
        ) from exc


def XGBoost(train, valid, test, params, *, sample_weight_ctx=None):

    n_estimators = params[0]
    max_depth = int(params[1]) if params[1] is not None else None
    # Configured values, under names the interaction refit closure below can
    # reference without being shadowed by its own override parameters.
    n_estimators_cfg = n_estimators
    max_depth_cfg = max_depth
    learning_rate = params[2]
    subsample = params[3]
    colsample_bytree = params[4]
    reg_lambda = params[5]
    get_interaction = params[6]
    shapley_num = params[7]
    threshold = params[8]
    max_interaction_features = params[9]
    # Requirements3.md item 2 - see models/RF.py's identical field.
    effect_method = params[10] if len(params) > 10 else 'feature_importance'
    effect_shapley_num = params[11] if len(params) > 11 else 30
    # Requirements.md items 2/3 (this update) - see models/RF.py's
    # identical fields for the full rationale. Appended, read defensively
    # (I5) - a config predating these four fields keeps this model's
    # previous, always-on marker-effect behaviour exactly.
    get_effect = params[12] if len(params) > 12 else True
    effect_max_shapiq_features = params[13] if len(params) > 13 else 500
    effect_shapiq_background = params[14] if len(params) > 14 else 50
    effect_shapiq_budget = params[15] if len(params) > 15 else 200
    # This update - see models/RF.py's identical field for the full
    # rationale. Appended, read defensively (I5).
    use_effect_for_interaction_shortlist = params[16] if len(params) > 16 else True
    # RF-style 'Marker interaction method' block - see models/ExtraTrees.py
    # / models/RF.py's identical fields. Appended, read defensively (I5):
    # an older config resolves to 'pairwise_shap' (previous behaviour, I11).
    interaction_method = params[17] if len(params) > 17 else 'pairwise_shap'
    interaction_screen = params[18] if len(params) > 18 else 'off'
    interaction_screen_top = params[19] if len(params) > 19 else 2.0
    interaction_grid_resolution = params[20] if len(params) > 20 else 3
    interaction_h_background = params[21] if len(params) > 21 else None
    interaction_shapiq_orders = params[22] if len(params) > 22 else '2'
    interaction_shapiq_index = params[23] if len(params) > 23 else 'k-SII'
    interaction_shapiq_budget = params[24] if len(params) > 24 else 4096

    train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1]
    if valid.shape[0] != 0:
        valid_x, valid_y = valid.iloc[:, :-1], valid.iloc[:, -1]
    test_x, test_y = test.iloc[:, :-1], test.iloc[:, -1]
    # ver4-25, R8 - see models/RF.py's identical comment.
    _w_fit = sample_weight_ctx.w_fit if sample_weight_ctx is not None else None

    _resources = get_active_compute_resources()
    _xgb_cls = _xgb_regressor_cls()

    # XGBoost has its own native GPU training path ('gpu_hist'/'cuda'
    # device string in recent versions) - kept deliberately CPU-only here
    # for this first delivery (disclosed scope simplification, ver4-5
    # Change Summary §7), matching ExtraTrees.py/GBDT.py's own CPU-only
    # scope decision for the same reason (no existing precedent in this
    # codebase's own compute-resource plumbing for a THIRD GPU backend
    # beyond cuML/torch).
    xgb_model = _xgb_cls(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        subsample=subsample, colsample_bytree=colsample_bytree, reg_lambda=reg_lambda,
        random_state=0, n_jobs=_resources['n_jobs'],
    )
    xgb_model.fit(train_x, train_y, sample_weight=_w_fit)

    predicted = np.asarray(xgb_model.predict(test_x)).ravel()
    if valid.shape[0] != 0:
        predicted_valid = np.asarray(xgb_model.predict(valid_x)).ravel()
    else:
        predicted_valid = []
    predicted_train = np.asarray(xgb_model.predict(train_x)).ravel()
    _feature_importances = np.asarray(xgb_model.feature_importances_).ravel()

    actual_test = test_y.values.tolist()
    mse = mean_squared_error(actual_test, predicted)
    r = pearsonr(actual_test, predicted)[0]

    # This update: marker-EFFECT computation (below) is hoisted ahead of
    # the interaction-shortlist step - see models/RF.py's identical
    # comment for the full rationale (pure reordering, I11 either way).
    # tree_shap_main_effects() routes XGBoost through its own native
    # pred_contribs=True path automatically (duck-typed via get_booster) -
    # no XGBoost-specific branching needed here.
    def _tree_shap_effect():
        _effect_num = min(effect_shapley_num, test_x.shape[0])
        effect_sample = shap.sample(test_x, _effect_num) if _effect_num > 0 else test_x.iloc[:0]
        effect_scores = tree_shap_main_effects(xgb_model, effect_sample, train_x.columns)
        return pd.DataFrame([effect_scores], columns=train_x.columns)

    # Requirements.md item 3 - see models/RF.py's identical gating.
    if not get_effect:
        effect = pd.DataFrame()
    elif effect_method == 'shap':
        try:
            effect = _tree_shap_effect()
        except Exception as exc:
            print(f"[XGBoost] NOTE: Shapley marker-effect computation failed ({exc!r}) - "
                  f"falling back to this model's own feature_importances_ for this task instead.")
            effect = pd.DataFrame(_feature_importances).T
    elif effect_method == 'shapiq':
        # Requirements.md item 2 - see models/RF.py's identical 'shapiq'
        # branch for the full rationale. XGBoost is kept CPU-only
        # throughout this file (module docstring), so there is no GPU
        # refit consideration here either.
        try:
            n_features = train_x.shape[1]
            if effect_max_shapiq_features != 'all' and n_features > effect_max_shapiq_features:
                top_features = train_x.columns[
                    np.argsort(_feature_importances)[::-1][:effect_max_shapiq_features]
                ]
                xgb_shapiq_effect = _xgb_cls(
                    n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                    subsample=subsample, colsample_bytree=colsample_bytree, reg_lambda=reg_lambda,
                    random_state=0, n_jobs=_resources['n_jobs'],
                )
                xgb_shapiq_effect.fit(train_x[top_features], train_y, sample_weight=_w_fit)
                effect_train_x = train_x[top_features]
                effect_test_x = test_x[top_features]
            else:
                xgb_shapiq_effect = xgb_model
                top_features = train_x.columns
                effect_train_x = train_x
                effect_test_x = test_x

            _effect_num = min(effect_shapley_num, effect_test_x.shape[0])
            effect_scores = shapiq_marker_effects(
                xgb_shapiq_effect.predict, effect_train_x, shap.sample(effect_test_x, _effect_num),
                background_size=min(effect_shapiq_background, effect_train_x.shape[0]),
                budget=effect_shapiq_budget,
            )
            effect_full = pd.Series(0.0, index=train_x.columns)
            effect_full.loc[top_features] = effect_scores
            effect = pd.DataFrame(effect_full).T
        except Exception as exc:
            print(f"[XGBoost] NOTE: shapiq marker-effect computation failed ({exc!r}) - falling "
                  f"back to this model's own exact TreeSHAP ('shap') route for this task instead. "
                  f"('shapiq' is an optional dependency - 'pip install shapiq' if not yet "
                  f"installed; this may also mean this shapiq version's API differs from what "
                  f"this codebase expects.)")
            try:
                effect = _tree_shap_effect()
            except Exception as exc2:
                print(f"[XGBoost] NOTE: TreeSHAP fallback also failed ({exc2!r}) - falling back "
                      f"to this model's own feature_importances_ for this task instead.")
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

    if get_interaction == True:
        shapley_num = min(shapley_num, test_x.shape[0])

        def _refit_xgb(columns, max_depth=None, n_estimators=None):
            _m = _xgb_cls(
                n_estimators=n_estimators_cfg if n_estimators is None else n_estimators,
                max_depth=(max_depth_cfg if max_depth is None else max_depth),
                learning_rate=learning_rate, subsample=subsample,
                colsample_bytree=colsample_bytree, reg_lambda=reg_lambda,
                random_state=0, n_jobs=_resources['n_jobs'],
            )
            _m.fit(train_x[columns], train_y, sample_weight=_w_fit)
            return _m

        interaction_sample = tree_model_interactions(
            tag='XGBoost', fitted_model=xgb_model, refit_fn=_refit_xgb,
            train_x=train_x, test_x=test_x, importance=_interaction_importance,
            interaction_method=interaction_method, shapley_num=shapley_num,
            threshold=threshold, max_interaction_features=max_interaction_features,
            screen=interaction_screen, screen_top=interaction_screen_top,
            grid_resolution=interaction_grid_resolution, h_background=interaction_h_background,
            shapiq_orders=interaction_shapiq_orders, shapiq_index=interaction_shapiq_index,
            shapiq_budget=interaction_shapiq_budget,
            n_estimators=n_estimators_cfg,
            # XGBoost trains at depth 6 when max_depth is None, so that - not
            # None ("unlimited") - is the depth the shapiq_tree cap compares to.
            effective_max_depth=(max_depth_cfg if max_depth_cfg is not None else 6),
            resources=_resources,
        )
    else:
        interaction_sample = pd.DataFrame()

    # 'effect' was already computed above (hoisted ahead of the
    # interaction-shortlist step) - nothing left to do here but return it.
    return r, mse, effect, interaction_sample, predicted, predicted_valid, predicted_train
