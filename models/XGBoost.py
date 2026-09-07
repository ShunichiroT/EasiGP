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

from pipeline_utils import get_active_compute_resources
from models.interaction_extraction import tree_shap_interactions, top_select


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


def XGBoost(train, valid, test, params):

    n_estimators = params[0]
    max_depth = int(params[1]) if params[1] is not None else None
    learning_rate = params[2]
    subsample = params[3]
    colsample_bytree = params[4]
    reg_lambda = params[5]
    get_interaction = params[6]
    shapley_num = params[7]
    threshold = params[8]
    max_interaction_features = params[9]

    train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1]
    if valid.shape[0] != 0:
        valid_x, valid_y = valid.iloc[:, :-1], valid.iloc[:, -1]
    test_x, test_y = test.iloc[:, :-1], test.iloc[:, -1]

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
    xgb_model.fit(train_x, train_y)

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

    if get_interaction == True:
        shapley_num = min(shapley_num, test_x.shape[0])

        n_features = train_x.shape[1]
        if max_interaction_features != 'all' and n_features > max_interaction_features:
            top_features = train_x.columns[
                np.argsort(_feature_importances)[::-1][:max_interaction_features]
            ]
            xgb_interaction = _xgb_cls(
                n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                subsample=subsample, colsample_bytree=colsample_bytree, reg_lambda=reg_lambda,
                random_state=0, n_jobs=_resources['n_jobs'],
            )
            xgb_interaction.fit(train_x[top_features], train_y)
            interaction_test_x = test_x[top_features]
        else:
            xgb_interaction = xgb_model
            interaction_test_x = test_x

        interaction_sample = tree_shap_interactions(
            xgb_interaction, shap.sample(interaction_test_x, shapley_num),
            interaction_test_x.columns, n_jobs=_resources['n_jobs'], reduce='abs_then_sum',
        )
        interaction_sample = top_select(interaction_sample, 'percentage', threshold)
    else:
        interaction_sample = pd.DataFrame()

    return r, mse, pd.DataFrame(_feature_importances).T, interaction_sample, predicted, predicted_valid, predicted_train
