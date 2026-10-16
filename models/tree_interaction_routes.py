"""
models/tree_interaction_routes.py
=================================
Shared implementation of RF's selectable 'Marker interaction method'
(``pairwise_shap`` / ``friedman_h`` / ``shapiq_tree``) for the OTHER tree
models that now expose the same field block - ``ExtraTrees`` and
``XGBoost`` (see ``hparam_specs._append_rf_style_interaction_fields``).

Why a shared function instead of a third copy of RF's block: RF's version
is ~300 lines inside ``models/RF.py`` (deliberately left untouched, so RF's
own results stay byte-identical - I11). ExtraTrees and XGBoost differ from
RF ONLY in how an estimator is (re)fitted, so that single difference is
passed in as ``refit_fn`` and everything else - the marker shortlist, the
three routes, the top-N% filter, the depth/estimator calibration for
``shapiq_tree`` - is written once here and follows RF's block step for
step (same helpers, imported from ``models.RF``, same fall-back-to-
pairwise_shap-on-any-shapiq-failure behaviour, same padding).

With ``interaction_method == 'pairwise_shap'`` (the default, and what
every config written before these fields existed resolves to) this
reproduces the previous ExtraTrees/XGBoost code path exactly: shortlist by
importance, refit on the shortlist, ``tree_shap_interactions(...,
reduce='abs_then_sum')``, ``top_select(..., 'percentage', threshold)``.
"""

import numpy as np
import pandas as pd
import shap

from pipeline_utils import (
    calibrate_shapiq_tree_depth, resolve_shapiq_explain_row_count,
    _predict_fn_with_feature_names,
)
from models.interaction_extraction import (
    tree_shap_interactions, tree_shapiq_interactions, h_statistic_interactions,
    top_select, top_select_by_order, pad_shapiq_pairs_to_shap_parity,
)
# RF's own order-aware depth / tree-count / calibration helpers - reused (not
# re-implemented) so the three tree models' 'shapiq_tree' cost controls can
# never drift apart. Same precedent as `_map_genes_to_markers` (see the
# architecture doc, section 8).
from models.RF import (
    _resolve_shapiq_tree_depth_cap, _resolve_shapiq_tree_estimator_cap,
    _resolve_shapiq_tree_depth_ladder, _resolve_shapiq_calibration_seconds,
    _SHAPIQ_TREE_MAX_DEPTH,
)


def tree_model_interactions(
    *, tag, fitted_model, refit_fn, train_x, test_x, importance,
    interaction_method, shapley_num, threshold, max_interaction_features,
    screen, screen_top, grid_resolution, h_background,
    shapiq_orders, shapiq_index, shapiq_budget,
    n_estimators, effective_max_depth, resources,
):
    """Marker-pair (and, for ``shapiq_tree``, higher-order) interactions
    for a fitted tree model. Returns the same frame RF's block produces:
    ``marker1, marker2, value`` for a pairwise route, plus ``markers`` /
    ``order`` columns when ``shapiq_tree`` searched an order above 2.

    Parameters
    ----------
    tag : str
        Log prefix, e.g. ``'ExtraTrees'``.
    fitted_model
        The task's already-fitted estimator (used directly when no marker
        shortlist is needed).
    refit_fn : callable ``(columns, max_depth=None, n_estimators=None) ->
        fitted estimator``
        Refits THIS model, with its own configured hyperparameters, on
        ``train_x[columns]``. ``max_depth`` / ``n_estimators``, when not
        None, override the configured values (used only to build the
        depth-capped ``shapiq_tree`` proxy). Sample weights are the
        caller's business (closed over inside ``refit_fn``).
    train_x, test_x : DataFrame
        Full marker frames for this task.
    importance : array-like
        One importance per marker (aligned to ``train_x`` columns) used to
        rank the shortlist - ``feature_importances_`` or, when eligible,
        the already-computed marker-effect values.
    interaction_method : str
        'pairwise_shap' | 'friedman_h' | 'shapiq_tree'.
    shapley_num : int
        Rows explained (already clamped/resolved by the caller).
    threshold : 'all' | number
        Top-N% output filter.
    max_interaction_features : 'all' | int
        Marker-shortlist cap.
    screen, screen_top, grid_resolution, h_background
        'friedman_h' route settings (ignored by the other routes).
    shapiq_orders, shapiq_index, shapiq_budget
        'shapiq_tree' route settings (ignored by the other routes).
    n_estimators : int
        Configured tree count (upper bound for the capped proxy refit).
    effective_max_depth : int or None
        The tree depth the model is ACTUALLY fitted with (None = no limit).
        ExtraTrees passes its own ``max_depth`` unchanged; XGBoost passes
        6 when its ``max_depth`` is None, since that is the depth XGBoost
        then trains at - the cap/refit logic below compares against the
        real depth, not a placeholder.
    resources : dict
        ``get_active_compute_resources()``.
    """
    n_features = train_x.shape[1]
    if max_interaction_features != 'all' and n_features > max_interaction_features:
        top_features = train_x.columns[
            np.argsort(np.abs(importance))[::-1][:max_interaction_features]
        ]
        model_i = refit_fn(top_features)
        interaction_test_x = test_x[top_features]
        interaction_train_x = train_x[top_features]
    else:
        model_i = fitted_model
        interaction_test_x = test_x
        interaction_train_x = train_x

    if interaction_method == 'friedman_h':
        columns = list(interaction_test_x.columns)
        pairs = [(a, b) for a in range(len(columns)) for b in range(a + 1, len(columns))]
        _h_bg = shapley_num if h_background is None else h_background
        interaction_sample = h_statistic_interactions(
            _predict_fn_with_feature_names(model_i.predict, columns),
            interaction_test_x, columns,
            pairs=pairs, n_background=min(_h_bg, interaction_test_x.shape[0]),
            n_jobs=resources['n_jobs'],
            screen=(screen if screen != 'off' else None),
            screen_keep=(float(screen_top) / 100.0 if screen_top != 'all' else 1.0),
            grid_resolution=grid_resolution,
        )

    elif interaction_method == 'shapiq_tree':
        try:
            _orders = tuple(
                int(_o.strip()) for _o in str(shapiq_orders).split(',') if _o.strip()
            ) or (2,)
            _max_order = max(_orders)
            if _max_order > 2:
                # k>2: budget-bounded, model-agnostic route on the task's
                # own fitted model (exact TreeSHAP-IQ is exponential in k) -
                # see models/RF.py's own comment for the full rationale.
                model_shapiq = model_i
                print(f"[{tag}] NOTE: 'shapiq_tree' interaction search requested order(s) "
                      f"{_orders} (k>2) - shapiq's exact TreeExplainer algorithm is "
                      f"exponentially more expensive per extra order. Using shapiq's "
                      f"model-agnostic, budget-bounded TabularExplainer directly on this "
                      f"task's own fitted model instead (budget={shapiq_budget}) for a "
                      f"bounded, predictable wall-clock cost. Order-2-only requests are "
                      f"unaffected and still use the exact TreeExplainer route.")
            else:
                if resources.get('shapiq_adaptive_depth', True):
                    _estimator_cap = _resolve_shapiq_tree_estimator_cap(_max_order, n_estimators)
                    if resources.get('shapiq_calibrate_depth', True):
                        _ladder_full = _resolve_shapiq_tree_depth_ladder(_max_order)
                        _ladder = tuple(
                            d for d in _ladder_full
                            if effective_max_depth is None or d <= effective_max_depth
                        )
                        if not _ladder:
                            _ladder = (effective_max_depth,)
                        _per_rung_seconds = _resolve_shapiq_calibration_seconds(
                            len(interaction_test_x.columns))
                        _probe_row = interaction_test_x.to_numpy()[0]
                        _cols = interaction_train_x.columns

                        def _fit_probe(_depth, _n_est=_estimator_cap):
                            return refit_fn(_cols, max_depth=_depth, n_estimators=_n_est)

                        try:
                            _depth_cap = calibrate_shapiq_tree_depth(
                                _fit_probe, _probe_row, _ladder,
                                max_order=_max_order, index=shapiq_index,
                                per_rung_seconds=_per_rung_seconds, log_prefix=f"[{tag}]",
                            )
                        except Exception as _cal_exc:
                            _depth_cap = _resolve_shapiq_tree_depth_cap(_max_order)
                            print(f"[{tag}] NOTE: 'shapiq_tree' depth calibration itself failed "
                                  f"({_cal_exc!r}) - falling back to the fixed, order-aware depth "
                                  f"guess ({_depth_cap}) instead.")
                    else:
                        _depth_cap = _resolve_shapiq_tree_depth_cap(_max_order)
                else:
                    _depth_cap = _SHAPIQ_TREE_MAX_DEPTH
                    _estimator_cap = n_estimators
                _refit_depth = (
                    _depth_cap if (effective_max_depth is None or effective_max_depth > _depth_cap)
                    else effective_max_depth
                )
                _refit_needed = (_refit_depth != effective_max_depth) or (_estimator_cap < n_estimators)
                if _refit_needed:
                    model_shapiq = refit_fn(
                        interaction_train_x.columns, max_depth=_refit_depth, n_estimators=_estimator_cap)
                    print(f"[{tag}] NOTE: 'shapiq_tree' interaction search is refitting a separate, "
                          f"capped (max_depth={_refit_depth}, n_estimators={_estimator_cap}) "
                          f"model for this step only, for order-{_max_order} search - shapiq's "
                          f"exact TreeSHAP-IQ algorithm becomes prohibitively slow on this "
                          f"model's own configured depth "
                          f"({'unlimited' if effective_max_depth is None else effective_max_depth})"
                          f"/tree count ({n_estimators}). Predictions, marker effects, and every "
                          f"other interaction route are computed from the model exactly as "
                          f"configured; only this optional, approximate interaction search uses "
                          f"the capped model.")
                else:
                    model_shapiq = model_i

            _shapiq_row_n = (
                resolve_shapiq_explain_row_count(_max_order, shapley_num)
                if resources.get('shapiq_adaptive_depth', True) else shapley_num
            )
            _shapley_explain_n = min(_shapiq_row_n, interaction_test_x.shape[0])
            _shapiq_n_jobs = resources['n_jobs'] if resources.get('shapiq_row_parallel', True) else 1
            interaction_sample = tree_shapiq_interactions(
                model_shapiq, shap.sample(interaction_test_x, _shapiq_row_n),
                interaction_test_x.columns, orders=_orders, index=shapiq_index,
                max_seconds=max(60.0, 20.0 * max(_shapley_explain_n, 1)),
                n_jobs=_shapiq_n_jobs,
                X_train_background=interaction_train_x,
                budget=shapiq_budget,
            )
        except Exception as exc:
            print(f"[{tag}] NOTE: shapiq_tree interaction extraction failed ({exc!r}) - "
                  f"falling back to 'pairwise_shap' (exact shap.TreeExplainer) for this "
                  f"task instead.")
            interaction_sample = tree_shap_interactions(
                model_i, shap.sample(interaction_test_x, shapley_num),
                interaction_test_x.columns, n_jobs=resources['n_jobs'], reduce='abs_then_sum',
            )

    else:
        # 'pairwise_shap' (default) - the route ExtraTrees/XGBoost always used.
        interaction_sample = tree_shap_interactions(
            model_i, shap.sample(interaction_test_x, shapley_num),
            interaction_test_x.columns, n_jobs=resources['n_jobs'], reduce='abs_then_sum',
        )

    if interaction_method in ('friedman_h', 'shapiq_tree'):
        # RF's own filter for these two routes: order-aware, and (for the
        # shapiq route) padded with explicit zero rows to the row count the
        # exact pairwise route would produce - see models/RF.py.
        interaction_sample = top_select_by_order(interaction_sample, 'percentage', threshold)
        if interaction_method == 'shapiq_tree':
            interaction_sample = pad_shapiq_pairs_to_shap_parity(
                interaction_sample, list(interaction_test_x.columns), threshold,
            )
    else:
        # Byte-identical to the pre-existing ExtraTrees/XGBoost behaviour.
        interaction_sample = top_select(interaction_sample, 'percentage', threshold)
    return interaction_sample
