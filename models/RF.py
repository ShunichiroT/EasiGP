from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import shap

from pipeline_utils import (
    cuml_gpu_usable, get_active_compute_resources, fit_with_sample_weight, calibrate_shapiq_tree_depth,
    resolve_shapiq_explain_row_count, shapiq_marker_effects,
)
from models.interaction_extraction import (
    tree_shap_interactions, tree_shap_main_effects, tree_shapiq_interactions,
    h_statistic_interactions, top_select_by_order, pad_shapiq_pairs_to_shap_parity,
    should_use_effect_for_interaction_shortlist,
)


def _random_forest_regressor_cls(use_gpu):
    """Return (RandomForestRegressor class, is_gpu) - cuML's GPU-backed
    RandomForestRegressor when `use_gpu` is True AND cuML is importable,
    else scikit-learn's own CPU implementation. Never raises: any failure
    importing cuML (not installed, no compatible GPU/driver, etc.) falls
    back to CPU silently, so a CPU-only node behaves exactly as before
    this option existed (Phase 2, Requirement 6)."""
    if use_gpu and cuml_gpu_usable():
        try:
            from cuml.ensemble import RandomForestRegressor as CumlRF
            return CumlRF, True
        except Exception as exc:
            print(f"[RF] USE_GPU_SKLEARN requested but cuML is unavailable "
                  f"({exc}) - falling back to scikit-learn's CPU RandomForestRegressor.")
    return RandomForestRegressor, False

# Exact pairwise SHAP interaction values require building an M x M matrix
# (M = number of markers) for every sampled test row, and the triu-mask /
# stack() step afterwards is itself an O(M^2) operation. With ~8,000 markers
# that is a 64-million-cell matrix and a ~32-million-row table - this is what
# turns the interaction step into a multi-day run.
#
# Since only the top `threshold`% of interactions are ever kept anyway, by
# default we restrict the *computation* itself to the top-importance markers
# (from the already-fitted forest) before running SHAP, rather than every
# marker in the dataset. This is the standard two-stage strategy for
# large-scale epistasis screening: filter candidates by main-effect
# importance first, then test pairwise interactions only among the
# shortlist. Predictions and per-marker effects (returned separately below)
# are unaffected - they still use the full model trained on every marker;
# only the interaction search is narrowed. The cap itself is user-configurable
# via params[8] (GUI: "Max markers considered for interaction search").

# Requirements.md item 2: the marker shortlist above bounds the 'shapiq_tree'
# route's cost along the MARKER axis, exactly as it already does for the
# exact 'pairwise_shap' route - but shapiq's own TreeSHAP-IQ algorithm
# (models.interaction_extraction.tree_shapiq_interactions(), used only by
# 'shapiq_tree') has a SECOND, much steeper cost axis that 'pairwise_shap'
# does not share: tree DEPTH. Confirmed empirically (benchmarked directly
# against a real shapiq==1.4.1 install, order-2, k-SII, 100 trees): a
# depth-4 forest explains one row in ~0.4s, depth-6 in ~1.6s, depth-8 in
# ~3.7s, and RF's own DEFAULT 'Maximum tree depth' (None - fully grown
# trees, routinely depth 10-25+ on a real training set) took 5-100+ SECONDS
# for a SINGLE row even at only 50 markers - this is what makes the
# 'shapiq_tree' route "prohibitively computationally expensive" in
# practice, and why the issue is "more apparent when using RF" (RF is the
# only model with this route at all - see hparam_specs.py's own note that
# ExtraTrees/XGBoost have no shapiq-interaction option). Unlike
# 'pairwise_shap'/'friedman_h', 'shapiq_tree' is already a disclosed,
# opt-in APPROXIMATION of this model's own exact interaction structure (it
# exists specifically to reach interaction orders above 2, which the exact
# route cannot) - so refitting a SEPARATE, depth-capped forest purely for
# this explanation step (never used for prediction, and never substituted
# into the exact 'pairwise_shap'/'friedman_h' routes) keeps it tractable
# without changing this model's own configured 'Maximum tree depth' at
# all. The cap below is only ever applied when it would actually SHRINK the
# tree relative to what the user configured (a user-set depth already at or
# below the cap is left alone, and the existing `rf_interaction` object is
# reused with no extra refit) - never widens a user's own smaller setting.
_SHAPIQ_TREE_MAX_DEPTH = 8

# ver4-26, design blueprint R2: the flat cap above was benchmarked ONLY at
# order-2 (see the comment block above it - "order-2, k-SII, 100 trees").
# TreeSHAP-IQ's own published complexity is O(T*L*D^k) (T=trees, L=leaves,
# D=depth, k=requested interaction order) - the depth term is EXPONENTIAL
# in k, so a cap sized for k=2 (D=8 => D**2=64) becomes D**3=512 at k=3
# with the SAME cap - an 8x jump from one extra order, independent of
# dataset size, and the most direct, falsifiable explanation available
# from the live tree for "cost worsens sharply as k grows". The two
# resolvers below make BOTH the depth cap and the interaction-only refit's
# `n_estimators` (T also appears in the complexity term, previously
# uncapped) shrink as the requested order grows, gated by the new
# `SHAPIQ_ADAPTIVE_DEPTH` compute-resource key (default True).


def _resolve_shapiq_tree_depth_cap(max_order: int, base_cap: int = 8) -> int:
    """Design blueprint R2: order-aware replacement for the flat
    `_SHAPIQ_TREE_MAX_DEPTH` constant - shrinks the depth cap as the
    requested interaction order grows, attacking TreeSHAP-IQ's own
    exponential-in-``k`` cost driver (``O(T*L*D**k)``) at its root
    instead of relying solely on the wall-clock safety net
    (`pipeline_utils.shapiq_tree_interactions()`'s own `max_seconds`).

    Parameters
    ----------
    max_order : int
        The highest interaction order requested for this task
        (``max(_orders)`` at this function's own call site).
    base_cap : int, default 8
        The order-2 cap - unchanged from `_SHAPIQ_TREE_MAX_DEPTH`'s own
        original, benchmarked value (I11: `max_order <= 2` reproduces
        today's exact constant).

    Returns
    -------
    int
        `base_cap` at `max_order <= 2` (unchanged, I11); shrinks by 2 per
        extra order above 2 (``base_cap - 2*(max_order-2)``), floored at
        3 - below which TreeSHAP-IQ degenerates to near-stump trees with
        little signal left to explain.
    """
    if max_order <= 2:
        return base_cap
    return max(3, base_cap - 2 * (max_order - 2))


def _resolve_shapiq_tree_estimator_cap(max_order: int, base_n: int, cap: int = 200) -> int:
    """Design blueprint R2: caps the 'shapiq_tree' interaction-only
    refit's own `n_estimators` the same way `_resolve_shapiq_tree_depth_
    cap()` caps `max_depth` - tree COUNT (T) also appears in TreeSHAP-IQ's
    ``O(T*L*D**k)`` complexity term and was previously left uncapped
    entirely, regardless of requested order.

    Parameters
    ----------
    max_order : int
        The highest interaction order requested for this task.
    base_n : int
        This task's own configured 'Number of trees' (`estimators`) -
        the refit never INCREASES tree count above what the user asked
        for, only caps it.
    cap : int, default 200
        The order-2 ceiling - a no-op for the overwhelmingly common
        configuration (few models are tuned to more than 200 trees for
        an interaction-only refit); shrinks further for order >= 3 by
        the SAME ratio `_resolve_shapiq_tree_depth_cap()` applies to
        depth, so the two axes of the complexity term shrink together.

    Returns
    -------
    int
        ``min(base_n, cap)`` at `max_order <= 2` (I11: a no-op for any
        `base_n` already at or below `cap`); shrinks further, floored at
        10 (enough trees for TreeSHAP-IQ's own per-tree averaging to
        remain meaningful), for `max_order > 2`.
    """
    _capped = min(int(base_n), int(cap))
    if max_order <= 2:
        return _capped
    # Ratio computed against the STANDARD depth base (8), not against
    # `cap` itself - `_resolve_shapiq_tree_depth_cap()`'s own shrink
    # schedule (`base_cap - 2*(max_order-2)`, floored at 3) is defined
    # relative to its own default `base_cap=8`; reusing that same ratio
    # here (rather than recomputing the schedule against `cap`) is what
    # keeps the two complexity-term axes (T and D) shrinking together at
    # the SAME relative rate, regardless of what `cap`/`base_n` are.
    _ratio = _resolve_shapiq_tree_depth_cap(max_order, base_cap=8) / 8.0
    return max(10, int(round(_capped * _ratio)))


# ver4-27: 'shapiq_tree' finding SIGNIFICANTLY fewer marker-pair
# interactions than 'pairwise_shap' on the identical fitted forest, traced
# to `_resolve_shapiq_tree_depth_cap()`'s own flat, unconditional depth-8
# guess above. Confirmed directly (this update, against a real installed
# `shapiq==1.4.1`, `RandomForestRegressor`): a depth-capped forest does not
# merely compute SMALL interaction values for markers it never split on -
# those pairs are ABSENT from `shapiq.TreeExplainer`'s own
# `interaction_lookup` entirely (not "evaluated as zero"), and the
# fraction of C(M,2) candidate pairs actually present climbs steeply with
# depth: at a fixed 100 trees, going from depth 4 -> 6 -> 8 -> 10 covered
# 8.9% -> 39.5% -> 84.3% -> 98.1% of C(150,2)=11,175 pairs at 150 markers
# (a smaller shortlist than this route's own 500-marker default) - i.e.
# depth 8 alone was already discarding ~16% of the candidate space at
# 150 markers, and construction cost grows FASTER than depth alone
# predicts as marker count grows too (2.8s -> 22.2s construction going
# from 40 to 150 markers at the SAME depth=8/100 trees - an ~8x jump for
# a 3.75x feature increase), so the gap versus 'pairwise_shap' (which
# explains the model's own, typically unlimited-depth forest directly)
# only widens at this route's real 500-marker default. Also confirmed:
# `shapiq.TreeExplainer` CONSTRUCTION at depth=12/150 markers/100 trees
# was killed by the OS's OOM killer outright (uncatchable SIGKILL) - this
# is exactly why `pipeline_utils.shapiq_tree_interactions()` now wraps
# construction (not just the row-explain loop) inside
# `bounded_memory_guard()` too (ver4-27, same update): a MemoryError from
# an over-deep calibration attempt below must be catchable, or trying
# deeper candidate depths at all would be unsafe.
#
# `_SHAPIQ_TREE_DEPTH_LADDER` raises the CEILING this route will ever
# attempt (12, up from the old flat 8) and `calibrate_shapiq_tree_depth()`
# (pipeline_utils.py) - called from the 'shapiq_tree' branch below - picks
# the deepest rung this task's OWN real scale can actually afford within
# a wall-clock/memory budget, trying deeper rungs FIRST and falling back
# only as needed, rather than assuming the worst case (a flat 8) up front
# regardless of how small/fast this particular task actually is.
_SHAPIQ_TREE_DEPTH_LADDER = (12, 10, 9, 8, 6, 4, 3)

# Per-rung calibration wall-clock budget: `max(120s, 0.3s * n_markers)`.
# Grounded in the SAME benchmark cited above - depth=8 construction alone
# took ~22s at 150 markers; extrapolating construction's own observed
# super-linear growth in marker count to this route's 500-marker default
# lands well over 120s, hence the per-marker term (0.3 * 500 = 150s) - not
# a tight bound, deliberately generous, since the ONLY failure mode of a
# too-generous budget is spending longer discovering that a rung is too
# slow, never an incorrect result (`calibrate_shapiq_tree_depth()` always
# still falls back to a safe, working depth). A tighter, task-specific
# value can be tuned directly here if a given deployment's own hardware
# warrants it.
_SHAPIQ_CALIBRATION_MIN_SECONDS = 120.0
_SHAPIQ_CALIBRATION_SECONDS_PER_MARKER = 0.3


def _resolve_shapiq_tree_depth_ladder(max_order: int) -> tuple:
    """ver4-27: candidate depths to calibrate against (descending - the
    deepest affordable one wins), order-2's own `_SHAPIQ_TREE_DEPTH_
    LADDER` with the SAME per-order shrink `_resolve_shapiq_tree_depth_
    cap()` already applies, so a higher-order search still calibrates
    over a correspondingly shallower ladder (I11 spirit: this function's
    own `max_order<=2` case is simply the unmodified ladder constant).
    """
    if max_order <= 2:
        return _SHAPIQ_TREE_DEPTH_LADDER
    _shrunk = sorted({max(3, d - 2 * (max_order - 2)) for d in _SHAPIQ_TREE_DEPTH_LADDER}, reverse=True)
    return tuple(_shrunk)


def _resolve_shapiq_calibration_seconds(n_features: int) -> float:
    """ver4-27: per-rung calibration wall-clock budget - see
    `_SHAPIQ_CALIBRATION_MIN_SECONDS`'s own module-level comment above for
    the benchmarked rationale."""
    return max(_SHAPIQ_CALIBRATION_MIN_SECONDS, _SHAPIQ_CALIBRATION_SECONDS_PER_MARKER * max(int(n_features), 0))


def RF(train, valid, test, params, *, sample_weight_ctx=None):
    
    estimators = params[0]
    features_max = params[1]
    sample_max = params[2]
    # Tree-complexity controls. max_depth is coerced to int (or None) since
    # the GUI field it comes from resolves to a float, but sklearn requires
    # max_depth to be an int or None specifically.
    max_depth = int(params[3]) if params[3] is not None else None
    min_samples_leaf = params[4]
    get_interaction = params[5]
    shapley_num = params[6]
    threshold = params[7]
    max_interaction_features = params[8]
    # Requirements.md item 3: selectable between the original pairwise-SHAP
    # route (default) and Friedman's H-statistic (model-agnostic - the same
    # route SVR/KNN already use via models.interaction_extraction.
    # h_statistic_interactions()). Appended-only, read defensively so a
    # pre-existing, shorter HPARAMETERS['RF'] list keeps reproducing this
    # file's original behaviour unchanged.
    interaction_method = params[9] if len(params) > 9 else 'pairwise_shap'
    # Update ID ver4-6, R1/R1b (blueprint §2.4/§2.10.4): appended, read
    # defensively (I5) - only actually used in the 'friedman_h' branch
    # below (see that branch's own comment); the default 'pairwise_shap'
    # route is unaffected by these three fields entirely.
    interaction_screen = params[10] if len(params) > 10 else 'off'
    interaction_screen_top = params[11] if len(params) > 11 else 2.0
    interaction_grid_resolution = params[12] if len(params) > 12 else 3
    # Requirements.md item 2 (cross-model H-index result diversity): a
    # DEDICATED background-sample-size field for the 'friedman_h' route,
    # independent of `shapley_num` above (which is sized for the exact
    # 'pairwise_shap' route's own explained-row count, not for an
    # averaged H-statistic - see hparam_specs.py's own new field for the
    # full rationale). Read defensively (I5) - a config predating this
    # field falls back to reusing `shapley_num`, reproducing this file's
    # previous 'friedman_h' behaviour exactly.
    interaction_h_background = params[13] if len(params) > 13 else None
    # Requirements3.md item 2: selectable between the original
    # feature_importances_-based effect (default, unchanged) and an
    # exact TreeSHAP additive-attribution effect - appended-only (I5),
    # read defensively so a pre-existing, shorter HPARAMETERS['RF'] list
    # keeps reproducing this file's original behaviour unchanged.
    effect_method = params[16] if len(params) > 16 else 'feature_importance'
    effect_shapley_num = params[17] if len(params) > 17 else 30
    # ver4-25, Stage 8 (design blueprint R11): only meaningful for the
    # 'shapiq_tree' interaction_method branch below - read defensively
    # (I5), a config predating these two fields simply never reaches
    # that branch (interaction_method itself would already be missing).
    interaction_shapiq_orders = params[14] if len(params) > 14 else '2'
    interaction_shapiq_index = params[15] if len(params) > 15 else 'k-SII'
    # Requirements.md items 2/3 (this update): appended, read defensively
    # (I5) - a config predating these four fields keeps this model's
    # previous, always-on marker-effect behaviour exactly (`get_effect`
    # defaults to True, matching hparam_specs.py's own default for the
    # new 'Return marker effect?' toggle). The three 'shapiq'-only fields
    # are only ever read inside the new 'shapiq' branch below - see
    # models/SVR.py's own identical 'shapiq' effect branch for the
    # template this follows.
    get_effect = params[18] if len(params) > 18 else True
    effect_max_shapiq_features = params[19] if len(params) > 19 else 500
    effect_shapiq_background = params[20] if len(params) > 20 else 50
    effect_shapiq_budget = params[21] if len(params) > 21 else 200
    # This update: new checkbox - when the interaction shortlist (field 8)
    # is no wider than this task's own marker-effect step already covers
    # (see should_use_effect_for_interaction_shortlist()'s own docstring),
    # reuse the ALREADY-COMPUTED marker-effect values as the interaction
    # shortlist's own ranking instead of feature_importances_. Appended,
    # read defensively (I5); default True only actually changes anything
    # when the GUI has also determined (and shown) that the size condition
    # holds - a config predating this field keeps today's behaviour
    # unchanged either way (I11).
    use_effect_for_interaction_shortlist = params[22] if len(params) > 22 else True
    # ver4-30: coalition-sample budget for the NEW model-agnostic,
    # budget-bounded 'shapiq_tree' route taken whenever a higher-order
    # (k>2) search is requested (see the 'shapiq_tree' branch below) -
    # meaningless (never read) for an order-2-only search, which still
    # takes the exact TreeExplainer route unaffected by this field.
    # Appended, read defensively (I5); default (4096) matches
    # `pipeline_utils.shapiq_pairwise_interactions()`'s own default
    # budget, the same routine RKHS's 'surrogate_shapiq'/SVR's 'shapiq'
    # routes already use, so a config predating this field gets that
    # same well-exercised default rather than an untested new one.
    interaction_shapiq_budget = params[23] if len(params) > 23 else 4096

    #Split the data sets into x and y here as specified in the original code
    train_x, train_y = train.iloc[:,:-1], train.iloc[:,-1]    
    # ver4-25, R8 (design blueprint SS2, Stage 6): the shared BLUE-
    # reliability fitting weight, aligned to train_x's own row order -
    # `None` (a strict sklearn no-op) whenever weighting is off, so this
    # is byte-identical to before this parameter existed (I11).
    _w_fit = sample_weight_ctx.w_fit if sample_weight_ctx is not None else None
    if valid.shape[0] != 0:
        valid_x, valid_y = valid.iloc[:,:-1], valid.iloc[:,-1]
    test_x, test_y = test.iloc[:,:-1], test.iloc[:,-1]

    # Phase 2, Requirement 6: n_jobs and an optional cuML GPU backend are
    # resolved from the run's shared compute-resource settings
    # (pipeline_utils.resolve_compute_resources()), not hardcoded - a
    # CPU-only node is completely unaffected (falls back to scikit-learn's
    # own multi-threaded n_jobs=-1, exactly as before this option existed).
    _resources = get_active_compute_resources()
    _rf_cls, _is_gpu = _random_forest_regressor_cls(_resources['use_gpu_sklearn'])

    #Develop & evaluate a model here as specified in the original code
    if _is_gpu:
        # cuML's RandomForestRegressor has a narrower constructor surface
        # than scikit-learn's - unsupported CPU-only knobs are simply
        # omitted rather than raising.
        rf = _rf_cls(n_estimators=estimators, random_state=0,
                     max_features=features_max, max_depth=max_depth,
                     min_samples_leaf=min_samples_leaf)
    else:
        rf = _rf_cls(n_estimators=estimators, random_state=0, max_samples=sample_max,
                     max_features=features_max, max_depth=max_depth,
                     min_samples_leaf=min_samples_leaf, n_jobs=_resources['n_jobs'])
    # ver4-25, R8: falls back to a CPU forest (same hyperparameters) if
    # the GPU forest's own .fit() doesn't accept sample_weight - see
    # pipeline_utils.fit_with_sample_weight()'s own docstring.
    rf = fit_with_sample_weight(
        rf, train_x, train_y, _w_fit, log_prefix='[RF]',
        cpu_fallback_factory=(lambda: RandomForestRegressor(
            n_estimators=estimators, random_state=0, max_samples=sample_max,
            max_features=features_max, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, n_jobs=_resources['n_jobs'],
        )) if _is_gpu else None,
    )
    
    predicted = np.asarray(rf.predict(test_x)).ravel()
    if valid.shape[0] != 0:
        predicted_valid = np.asarray(rf.predict(valid_x)).ravel()
    else:
        predicted_valid = []
    predicted_train = np.asarray(rf.predict(train_x)).ravel()
    _feature_importances = np.asarray(rf.feature_importances_).ravel()

    #Calculate the metrics
    actual_test = test_y.values.tolist()
    mse = mean_squared_error(actual_test, predicted)
    r = pearsonr(actual_test, predicted)[0]

    # Requirements3.md item 2 / this update: marker-EFFECT computation is
    # hoisted here, ahead of the interaction-shortlist step below - a pure
    # reordering (nothing computed here depends on the interaction step,
    # and the interaction step did not depend on 'effect' before this
    # update either). Doing this first lets the interaction step below
    # optionally reuse these SAME already-computed values as its own
    # shortlist ranking (new 'Use marker effects to select markers for
    # interaction search?' checkbox) instead of paying for a second,
    # redundant computation. See should_use_effect_for_interaction_
    # shortlist()'s own docstring (models/interaction_extraction.py) for
    # the exact eligibility rule this feeds into, just below.
    #
    # Requirements3.md item 2: 'shap' computes an exact TreeSHAP additive-
    # attribution effect instead of the forest's own feature_importances_
    # (the SAME sum(|phi|) aggregation convention every other Shapley-
    # based effect in this codebase already uses) - 'feature_importance'
    # (default) keeps today's exact behaviour unchanged (I11).
    def _tree_shap_effect():
        _effect_num = min(effect_shapley_num, test_x.shape[0])
        # shap.TreeExplainer needs a CPU scikit-learn forest - reuse
        # the SAME cuML->CPU refit convention the interaction step
        # above already uses (a cuML GPU forest is not TreeExplainer-
        # compatible). Independent of get_interaction/interaction_method
        # above - always explains every marker (never a shortlist),
        # matching feature_importances_'s own always-full-width shape.
        if _is_gpu:
            rf_effect = RandomForestRegressor(
                n_estimators=estimators, random_state=0, max_samples=sample_max,
                max_features=features_max, max_depth=max_depth,
                min_samples_leaf=min_samples_leaf, n_jobs=_resources['n_jobs'],
            )
            rf_effect.fit(train_x, train_y, sample_weight=_w_fit)
        else:
            rf_effect = rf
        effect_sample = shap.sample(test_x, _effect_num) if _effect_num > 0 else test_x.iloc[:0]
        effect_scores = tree_shap_main_effects(rf_effect, effect_sample, train_x.columns)
        return pd.DataFrame([effect_scores], columns=train_x.columns)

    # Requirements.md item 3: 'Return marker effect?' (field 18) now
    # gates this whole step - a config predating this field has
    # `get_effect=True` (see its own defensive read above), so this stays
    # a strict no-op for every existing saved config (I11).
    if not get_effect:
        effect = pd.DataFrame()
    elif effect_method == 'shap':
        try:
            effect = _tree_shap_effect()
        except Exception as exc:
            print(f"[RF] NOTE: Shapley marker-effect computation failed ({exc!r}) - falling "
                  f"back to this model's own feature_importances_ for this task instead.")
            effect = pd.DataFrame(_feature_importances).T
    elif effect_method == 'shapiq':
        # Requirements.md item 2 (this update): model-agnostic order-1
        # Shapley marker effect via the optional shapiq package
        # (pipeline_utils.shapiq_marker_effects()) - the SAME helper
        # SVR/KNN/RKHS's own 'shapiq' effect route already uses. Offered
        # here mainly for tooling/consistency parity with those models:
        # an order-1 Shapley value is index-independent, so this model's
        # own exact 'shap' (TreeSHAP) route above already computes the
        # mathematically-identical quantity, exactly rather than by
        # sampling - 'shap' is usually the faster, more precise choice
        # for a tree model. shapiq_marker_effects() only needs a plain
        # .predict callable, so - unlike the 'shap' (shap.TreeExplainer)
        # route above - no CPU refit is needed even when the main fit
        # used a GPU (cuML) backend. NEVER lets a shapiq failure (missing
        # package, an API mismatch) fail this task - falls back to the
        # exact 'shap' route (and, if that also fails, to
        # feature_importances_) with a printed NOTE, exactly like RF's
        # own 'shapiq_tree' interaction route falls back to
        # 'pairwise_shap' above.
        try:
            n_features = train_x.shape[1]
            if effect_max_shapiq_features != 'all' and n_features > effect_max_shapiq_features:
                # Same top-importance shortlist strategy as the
                # interaction step below (field 19, "Max markers
                # considered for marker effect (shapiq)") - a SEPARATE
                # cap from the interaction section's own shortlist
                # (field 8), per this project's "respective fields for
                # both marker effects and interactions" convention.
                top_features = train_x.columns[
                    np.argsort(_feature_importances)[::-1][:effect_max_shapiq_features]
                ]
                rf_shapiq_effect = RandomForestRegressor(
                    n_estimators=estimators, random_state=0, max_samples=sample_max,
                    max_features=features_max, max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf, n_jobs=_resources['n_jobs'],
                )
                rf_shapiq_effect.fit(train_x[top_features], train_y, sample_weight=_w_fit)
                effect_train_x = train_x[top_features]
                effect_test_x = test_x[top_features]
            else:
                # 'all' - use the already-fitted forest directly (GPU or
                # CPU, shapiq_marker_effects() only needs .predict).
                rf_shapiq_effect = rf
                top_features = train_x.columns
                effect_train_x = train_x
                effect_test_x = test_x

            _effect_num = min(effect_shapley_num, effect_test_x.shape[0])
            effect_scores = shapiq_marker_effects(
                rf_shapiq_effect.predict, effect_train_x, shap.sample(effect_test_x, _effect_num),
                background_size=min(effect_shapiq_background, effect_train_x.shape[0]),
                budget=effect_shapiq_budget,
            )
            # Reassemble into a full-width vector (one entry per marker,
            # 0 for any marker outside the shortlist) - same convention
            # as SVR.py's own 'shapiq'/'kernel_shap' effect reassembly.
            effect_full = pd.Series(0.0, index=train_x.columns)
            effect_full.loc[top_features] = effect_scores
            effect = pd.DataFrame(effect_full).T
        except Exception as exc:
            print(f"[RF] NOTE: shapiq marker-effect computation failed ({exc!r}) - falling back "
                  f"to this model's own exact TreeSHAP ('shap') route for this task instead. "
                  f"('shapiq' is an optional dependency - 'pip install shapiq' if not yet "
                  f"installed; this may also mean this shapiq version's API differs from what "
                  f"this codebase expects.)")
            try:
                effect = _tree_shap_effect()
            except Exception as exc2:
                print(f"[RF] NOTE: TreeSHAP fallback also failed ({exc2!r}) - falling back to "
                      f"this model's own feature_importances_ for this task instead.")
                effect = pd.DataFrame(_feature_importances).T
    else:
        effect = pd.DataFrame(_feature_importances).T

    # This update: how many markers this task's own marker-effect step
    # above actually covers - 'all' for every native/uncapped route, an
    # int for a capped sampling route, None when effects were not computed
    # at all (get_effect is False) - feeds the shared eligibility check
    # below. Kept in sync with hparam_specs.py's own field ordering by
    # hand (I5) - see main_app.py's dynamic show/hide for this same rule
    # applied to the checkbox's own visibility.
    if not get_effect:
        _effect_marker_cap = None
    elif effect_method == 'shapiq':
        _effect_marker_cap = effect_max_shapiq_features
    else:
        _effect_marker_cap = 'all'
    _use_effect_shortlist = should_use_effect_for_interaction_shortlist(
        use_effect_for_interaction_shortlist, max_interaction_features, _effect_marker_cap,
    )
    # Reuse this task's own already-computed, full-width marker-effect
    # values (aligned to train_x's own column order) as the interaction
    # shortlist's ranking when eligible; otherwise this model's own
    # feature_importances_, exactly as before this checkbox existed (I11).
    _interaction_importance = (
        np.asarray(effect.iloc[0]).ravel() if _use_effect_shortlist else _feature_importances
    )

    #Extract interactions - the SHAP interaction search always uses a
    # CPU (scikit-learn) forest, regardless of whether the main fit above
    # used a GPU backend: shap.TreeExplainer requires a scikit-learn-
    # compatible tree structure that cuML's GPU forest does not expose.
    if get_interaction == True:
        # Defensive clamp: avoid shap.sample() erroring if shapley_num exceeds
        # the number of available test rows.
        shapley_num = min(shapley_num, test_x.shape[0])

        n_features = train_x.shape[1]
        if max_interaction_features != 'all' and n_features > max_interaction_features:
            # Refit a lightweight forest on just the top-importance markers so
            # the SHAP interaction matrix is max_interaction_features x
            # max_interaction_features instead of n_features x n_features.
            # This update: ranked by this task's own already-computed
            # marker-effect values instead of feature_importances_ when
            # '_use_effect_shortlist' is True (see the eligibility check
            # above, right after 'effect' was computed) - otherwise
            # '_interaction_importance' IS 'feature_importances_' and this
            # line is byte-identical to before this checkbox existed (I11).
            top_features = train_x.columns[
                np.argsort(np.abs(_interaction_importance))[::-1][:max_interaction_features]
            ]
            rf_interaction = RandomForestRegressor(n_estimators = estimators, random_state = 0,
                                                     max_samples=sample_max, max_features=features_max,
                                                     max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                                     n_jobs=_resources['n_jobs'])
            rf_interaction.fit(train_x[top_features], train_y, sample_weight=_w_fit)
            interaction_test_x = test_x[top_features]
            interaction_train_x = train_x[top_features]
        elif _is_gpu and interaction_method != 'friedman_h':
            # 'all' markers, but the main fit was GPU-backed - still need a
            # CPU forest for shap.TreeExplainer; refit on every marker.
            # Requirements.md item 3: Friedman's H-statistic is fully
            # model-agnostic (needs only a plain .predict callable - see
            # h_statistic_interactions()'s own docstring), so it has no such
            # requirement and this refit is skipped for it below.
            rf_interaction = RandomForestRegressor(n_estimators=estimators, random_state=0,
                                                    max_samples=sample_max, max_features=features_max,
                                                    max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                                    n_jobs=_resources['n_jobs'])
            rf_interaction.fit(train_x, train_y, sample_weight=_w_fit)
            interaction_test_x = test_x
            interaction_train_x = train_x
        else:
            # 'all' - use every marker, no cap
            rf_interaction = rf
            interaction_test_x = test_x
            interaction_train_x = train_x

        if interaction_method == 'friedman_h':
            # Requirements.md item 3: Friedman's H-statistic (Friedman &
            # Popescu, 2008) via models.interaction_extraction.
            # h_statistic_interactions() - the same model-agnostic route
            # SVR/KNN already use for their own interaction search, reusing
            # RF's own already-fitted forest and importance-based shortlist
            # above (RF has a real importance measure, unlike SVR/KNN, which
            # fall back to a correlation-based proxy).
            interaction_columns = list(interaction_test_x.columns)
            interaction_pairs = [
                (a, b) for a in range(len(interaction_columns)) for b in range(a + 1, len(interaction_columns))
            ]
            # Requirements.md item 2: use the DEDICATED
            # `interaction_h_background` field when this config has it
            # (properly sized to match SVR/KNN/RKHS's own background,
            # default 100 - see hparam_specs.py's own new field), rather
            # than always reusing `shapley_num` (30 by default - tuned
            # for the 'pairwise_shap' route's exact per-row explanation,
            # not for H^2's own averaged estimate). A config predating
            # this field (`interaction_h_background is None`) keeps the
            # exact previous behaviour.
            _h_background = (
                shapley_num if interaction_h_background is None else interaction_h_background
            )
            interaction_sample = h_statistic_interactions(
                rf_interaction.predict, interaction_test_x, interaction_columns,
                pairs=interaction_pairs, n_background=min(_h_background, interaction_test_x.shape[0]),
                n_jobs=_resources['n_jobs'],
                # Update ID ver4-6, R1/R1b: forwarded ONLY here, the
                # 'friedman_h' branch - 'pairwise_shap' below never calls
                # h_statistic_interactions() at all, so these three
                # fields have no meaning for it. See models/SVR.py's own
                # identical note for the percentage -> fraction conversion.
                screen=(interaction_screen if interaction_screen != 'off' else None),
                screen_keep=(float(interaction_screen_top) / 100.0 if interaction_screen_top != 'all' else 1.0),
                grid_resolution=interaction_grid_resolution,
            )
        elif interaction_method == 'shapiq_tree':
            # ver4-25, Stage 8 (design blueprint R11): exact Shapley-
            # interaction values straight off the fitted forest via
            # shapiq's own TreeExplainer (models.interaction_extraction.
            # tree_shapiq_interactions()) - like 'pairwise_shap', no
            # surrogate, no sampling, but can also search orders above 2
            # (only order-2 rows are kept for now - see that function's
            # own docstring). Wrapped in its own try/except, separate
            # from the exact 'pairwise_shap' route below, so a shapiq-
            # specific failure (package missing, API mismatch) falls
            # back to 'pairwise_shap' for this task with a printed NOTE
            # rather than losing the interaction step entirely.
            try:
                _orders = tuple(
                    int(_o.strip()) for _o in str(interaction_shapiq_orders).split(',') if _o.strip()
                ) or (2,)
                _max_order = max(_orders)
                # ver4-30: exact TreeSHAP-IQ's own published complexity is
                # O(T*L*D**k) - EXPONENTIAL in the requested order k. Every
                # mitigation below (order-aware depth/estimator shrinking -
                # design blueprint R2 -, a real calibrated depth ladder -
                # ver4-27 -, order-aware row-count shrinking - ver4-28 -)
                # only ever shrinks the INPUTS to that exponent, and was
                # confirmed (this update) to still leave a genuine k>2
                # search "extremely large"/"cannot be used" in practice on
                # a realistically-sized forest/shortlist: even the deepest
                # floor this schedule ever reaches (depth=3) still means
                # D**3=27 versus D**2=4 at the SAME depth, a >6x jump from
                # one extra order, before T (tree count) and the marker
                # shortlist's own effect on per-tree feature count are even
                # considered. Rather than continue chasing an EXACT
                # algorithm whose cost is inherently combinatorial in k,
                # k>2 now skips the depth-capped refit entirely and instead
                # explains this task's own REAL, unmodified fitted forest
                # through shapiq's model-AGNOSTIC, BUDGET-bounded
                # TabularExplainer (models.interaction_extraction.
                # tree_shapiq_interactions()'s own new `X_train_background`/
                # `budget` parameters below) - the SAME bounded-cost
                # mechanism this codebase already relies on for RKHS's
                # 'surrogate_shapiq' and SVR/KNN's 'shapiq' routes. Wall-
                # clock cost is then governed by `interaction_shapiq_budget`
                # (a fixed, user-configurable ceiling), not by tree depth,
                # tree count, or order - this is what actually bounds the
                # previously-unbounded-in-practice cost, at the price of
                # trading shapiq's own exact tree-walk for an approximate,
                # sampling-based estimate (already the practical reality of
                # the pre-ver4-30 route too, once a k>2 request forced a
                # severely depth/estimator-capped proxy forest - this is
                # arguably a MORE faithful approximation, since it explains
                # the real, user-configured forest rather than a crippled
                # stand-in). `_max_order <= 2` is completely unaffected
                # (I11) - it still takes the exact, calibrated-depth-capped
                # TreeExplainer route below, since order-2 was never the
                # reported problem and remains fast and exact there.
                if _max_order > 2:
                    rf_shapiq = rf_interaction
                    print(f"[RF] NOTE: 'shapiq_tree' interaction search requested order(s) "
                          f"{_orders} (k>2) - shapiq's exact TreeExplainer algorithm is "
                          f"exponentially more expensive per extra order and was confirmed "
                          f"prohibitively slow here even after depth/estimator/row-count "
                          f"adaptive shrinking. Using shapiq's model-agnostic, budget-bounded "
                          f"TabularExplainer directly on this task's own fitted forest instead "
                          f"(budget={interaction_shapiq_budget}) for a bounded, predictable "
                          f"wall-clock cost governed by this budget rather than tree "
                          f"depth/count. Order-2-only requests are unaffected and still use "
                          f"the exact TreeExplainer route.")
                else:
                    # Design blueprint R2: the depth/tree-count cap now SHRINKS
                    # as the requested interaction order grows (see
                    # `_resolve_shapiq_tree_depth_cap()`'s own module-level
                    # comment above for the O(T*L*D**k) rationale), gated by
                    # the new `shapiq_adaptive_depth` compute-resource key
                    # (`SHAPIQ_ADAPTIVE_DEPTH`, default True). `False`
                    # reproduces today's flat `_SHAPIQ_TREE_MAX_DEPTH`/
                    # uncapped-`n_estimators` behaviour regardless of order
                    # (I11) - the two resolver functions themselves stay pure
                    # (order-based only, no flag knowledge), so the flag is
                    # applied here, at the one call site that ever needs it.
                    if _resources.get('shapiq_adaptive_depth', True):
                        _estimator_cap = _resolve_shapiq_tree_estimator_cap(_max_order, estimators)
                        # ver4-27: replace the flat/order-shrunk depth GUESS
                        # (_resolve_shapiq_tree_depth_cap()) with a real,
                        # calibrated ladder search when `SHAPIQ_CALIBRATE_DEPTH`
                        # is on (default True) - see `_SHAPIQ_TREE_DEPTH_
                        # LADDER`'s own module-level comment above for why this
                        # was needed (depth-8 alone was confirmed to leave most
                        # of a realistic marker shortlist's candidate pairs
                        # completely unconsidered, not just small-valued).
                        if _resources.get('shapiq_calibrate_depth', True):
                            _n_features_shapiq = len(interaction_test_x.columns)
                            _ladder_full = _resolve_shapiq_tree_depth_ladder(_max_order)
                            # Never CALIBRATE past a user's own smaller
                            # 'Maximum tree depth' either - matches the
                            # existing "never widens a user's own smaller
                            # setting" invariant this whole route already
                            # follows (see the `_refit_depth` line below), and
                            # avoids wasting calibration time on rungs that
                            # could never be used anyway.
                            _ladder = tuple(d for d in _ladder_full if max_depth is None or d <= max_depth)
                            if not _ladder:
                                _ladder = (max_depth,)
                            _per_rung_seconds = _resolve_shapiq_calibration_seconds(_n_features_shapiq)
                            _probe_row = interaction_test_x.to_numpy()[0]

                            def _fit_shapiq_probe_forest(_depth, _n_est=_estimator_cap):
                                _forest = RandomForestRegressor(
                                    n_estimators=_n_est, random_state=0, max_samples=sample_max,
                                    max_features=features_max, max_depth=_depth,
                                    min_samples_leaf=min_samples_leaf, n_jobs=_resources['n_jobs'],
                                )
                                _forest.fit(interaction_train_x, train_y, sample_weight=_w_fit)
                                return _forest

                            try:
                                _depth_cap = calibrate_shapiq_tree_depth(
                                    _fit_shapiq_probe_forest, _probe_row, _ladder,
                                    max_order=_max_order, index=interaction_shapiq_index,
                                    per_rung_seconds=_per_rung_seconds, log_prefix="[RF]",
                                )
                            except Exception as _cal_exc:
                                _depth_cap = _resolve_shapiq_tree_depth_cap(_max_order)
                                print(f"[RF] NOTE: 'shapiq_tree' depth calibration itself failed "
                                      f"({_cal_exc!r}) - falling back to the fixed, order-aware depth "
                                      f"guess ({_depth_cap}) instead.")
                        else:
                            _depth_cap = _resolve_shapiq_tree_depth_cap(_max_order)
                    else:
                        _depth_cap = _SHAPIQ_TREE_MAX_DEPTH
                        _estimator_cap = estimators
                    # Requirements.md item 2: refit a SEPARATE, capped forest
                    # for this route only when the model's own configured depth
                    # and/or tree count would actually make shapiq's TreeSHAP-IQ
                    # algorithm prohibitively expensive at this task's requested
                    # order (see _resolve_shapiq_tree_depth_cap()'s own
                    # module-level comment above for the benchmarked rationale).
                    # A user who already set a shallow-enough 'Maximum tree
                    # depth' AND a small-enough 'Number of trees' pays no extra
                    # refit at all - `rf_shapiq` is simply `rf_interaction` in
                    # that case (I11: byte-identical at order=2, the common
                    # case, for any config that predates R2).
                    _refit_depth = _depth_cap if (max_depth is None or max_depth > _depth_cap) else max_depth
                    _refit_needed = (_refit_depth != max_depth) or (_estimator_cap < estimators)
                    if _refit_needed:
                        rf_shapiq = RandomForestRegressor(
                            n_estimators=_estimator_cap, random_state=0, max_samples=sample_max,
                            max_features=features_max, max_depth=_refit_depth,
                            min_samples_leaf=min_samples_leaf, n_jobs=_resources['n_jobs'],
                        )
                        rf_shapiq.fit(interaction_train_x, train_y, sample_weight=_w_fit)
                        print(f"[RF] NOTE: 'shapiq_tree' interaction search is refitting a separate, "
                              f"capped (max_depth={_refit_depth}, n_estimators={_estimator_cap}) "
                              f"forest for this step only, for order-{_max_order} search - shapiq's "
                              f"own exact TreeSHAP-IQ algorithm becomes prohibitively slow on this "
                              f"model's own configured depth "
                              f"({'unlimited' if max_depth is None else max_depth})/tree count "
                              f"({estimators}) at this order. Predictions, marker effects, and every "
                              f"other interaction route are computed from the model exactly as "
                              f"configured; only this optional, approximate higher-order interaction "
                              f"search uses the capped forest.")
                    else:
                        rf_shapiq = rf_interaction
                # Requirements.md item 2: a wall-clock safety net on top of
                # the depth cap above - see shapiq_tree_interactions()'s own
                # `max_seconds` docstring (pipeline_utils.py). 20s/row is
                # generously above the ~3.7s/row this cap's own depth (8)
                # benchmarked at, so a normally-behaving task never comes
                # close to it - this exists purely to bound the tail case
                # (an unusually bushy capped tree, many estimators) that the
                # depth cap alone does not fully rule out.
                # ver4-28 (this update): order-aware shrink of the ROW axis
                # too, mirroring the depth/estimator shrink just above -
                # total wall-clock is (rows explained) x (per-row cost), so
                # a flat, order-blind `shapley_num` left this axis as the
                # one remaining untouched cost driver once the depth/tree
                # floor is reached. Gated by the SAME `shapiq_adaptive_depth`
                # flag (no new config key) - `_max_order <= 2` (I11) leaves
                # `shapley_num` completely unchanged.
                _shapiq_row_n = (
                    resolve_shapiq_explain_row_count(_max_order, shapley_num)
                    if _resources.get('shapiq_adaptive_depth', True) else shapley_num
                )
                _shapley_explain_n = min(_shapiq_row_n, interaction_test_x.shape[0])
                # Design blueprint R1: fan the per-row shapiq explain loop
                # out across this task's own resolved worker count, gated
                # by `shapiq_row_parallel` (`SHAPIQ_ROW_PARALLEL`, default
                # True) - `n_jobs=1` (the flag off, or a single-core node)
                # is byte-identical to this route's pre-R1 serial loop (I11).
                _shapiq_n_jobs = (
                    _resources['n_jobs'] if _resources.get('shapiq_row_parallel', True) else 1
                )
                interaction_sample = tree_shapiq_interactions(
                    rf_shapiq, shap.sample(interaction_test_x, _shapiq_row_n),
                    interaction_test_x.columns, orders=_orders, index=interaction_shapiq_index,
                    max_seconds=max(60.0, 20.0 * max(_shapley_explain_n, 1)),
                    n_jobs=_shapiq_n_jobs,
                    # ver4-30: only ever consumed by tree_shapiq_interactions()
                    # itself when `_max_order > 2` - passing these
                    # unconditionally is a no-op for an order-2-only search
                    # (I11), which still ignores both and takes the exact
                    # TreeExplainer route above.
                    X_train_background=interaction_train_x,
                    budget=interaction_shapiq_budget,
                )
            except Exception as exc:
                print(f"[RF] NOTE: shapiq_tree interaction extraction failed ({exc!r}) - "
                      f"falling back to 'pairwise_shap' (exact shap.TreeExplainer) for this "
                      f"task instead.")
                interaction_sample = tree_shap_interactions(
                    rf_interaction, shap.sample(interaction_test_x, shapley_num),
                    interaction_test_x.columns, n_jobs=_resources['n_jobs'], reduce='abs_then_sum',
                )
        else:
            # Update ID ver4-5, R2 (blueprint §4.4): the fit/explain/reduce/
            # triangle-select recipe below used to be implemented inline here
            # (and, separately, in models/GAT_prior_knowledge.py and
            # Preprocess/data_driven_prior_network.py) - all three now share
            # one implementation, models.interaction_extraction.
            # tree_shap_interactions(). reduce='abs_then_sum' is this file's
            # OWN pre-existing reduction order, passed explicitly so this
            # redirection is a numeric no-op (see that function's own
            # docstring). The row-fan-out (pipeline_utils.parallel_shap_values)
            # is unchanged - it is called FROM inside tree_shap_interactions()
            # now, not from here directly.
            interaction_sample = tree_shap_interactions(
                rf_interaction, shap.sample(interaction_test_x, shapley_num),
                interaction_test_x.columns, n_jobs=_resources['n_jobs'], reduce='abs_then_sum',
            )

        # RF's own pre-existing top-N% filter - moved, unchanged, into the
        # shared top_select() helper (models.interaction_extraction) so
        # its 'percentage' semantics are defined in exactly one place;
        # this call reproduces the original inline
        # `interaction_sample[interaction_sample['value'] >
        # interaction_sample['value'].quantile((1-(threshold/100)))]` line
        # exactly (strict '>', 'all' = no filtering).
        interaction_sample = top_select_by_order(interaction_sample, 'percentage', threshold)

        # ver4-27: pad 'shapiq_tree''s own order-2 rows with explicit
        # value=0.0 entries for every candidate pair (from the SAME
        # top-importance marker shortlist `interaction_test_x.columns`
        # 'pairwise_shap' searches) that this task did not end up
        # selecting - up to however many rows 'pairwise_shap' would
        # select from that IDENTICAL C(M,2) candidate pool at this SAME
        # `threshold`%. 'pairwise_shap' explains an exact, (generically)
        # dense M x M SHAP interaction matrix, so its own Interaction.csv
        # row count for a given `threshold`% is effectively
        # round(C(M,2) * threshold/100) already; 'shapiq_tree' - even
        # after the depth-calibration fix above - can still genuinely
        # find fewer real (nonzero) pairs than that at this task's scale.
        # Padding here makes the two routes' Interaction.csv row counts
        # DIRECTLY comparable (same candidate universe, same target
        # count) without touching `top_select_by_order()`'s own selection
        # math above, which stays exactly as strict about never treating
        # an unconsidered pair as a genuine top-N% finding (Requirements3.md
        # item 3's own fix - see that function's docstring) - the padded
        # rows are added AFTER selection, honestly marked value=0.0, so a
        # reader filtering `value > 0` recovers the exact set
        # `top_select_by_order()` actually selected.
        if interaction_method == 'shapiq_tree':
            interaction_sample = pad_shapiq_pairs_to_shap_parity(
                interaction_sample, list(interaction_test_x.columns), threshold,
            )
    else:
        interaction_sample = pd.DataFrame()

    # 'effect' was already computed above (hoisted ahead of the
    # interaction-shortlist step - see this update's own comment there),
    # so there is nothing left to do here but return it.

    return r, mse, effect, interaction_sample, predicted, predicted_valid, predicted_train

    
