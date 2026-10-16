from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import shap

from pipeline_utils import (
    cuml_gpu_usable, get_active_compute_resources, shapiq_marker_effects, shapiq_pairwise_interactions,
    resolve_shapiq_explain_row_count, knn_relief_effects, timed_stage,
    available_memory_bytes,
)
from models.knn_shapley import knn_shapley_effects, KNNBaselineShapFastEval
from models.interaction_extraction import (
    h_statistic_interactions, top_select_by_order, canonicalize_shapiq_table,
    pad_shapiq_pairs_to_shap_parity, should_use_effect_for_interaction_shortlist,
    # This update (user feedback, backed by this project's own experimental
    # results: "friedman-h was not reliable ... did not capture correct
    # interactions" and "shapiq ... too computationally expensive"): a
    # THIRD marker-pair interaction route, directly comparable in units/
    # aggregation to models/RF.py's own TreeSHAP interaction table (same
    # `abs_then_sum` reduction) and cheaper than shapiq at equal
    # background-size settings - see baseline_shap_interactions()'s own
    # module-level docstring (models/interaction_extraction.py) for the
    # full rationale (LD-safe: never evaluates a synthetic, independence-
    # assuming joint genotype combination the way Friedman's H's own grid
    # does; no coalition-sampling budget the way shapiq needs).
    baseline_shap_interactions,
)


class WeightedKNeighborsRegressor(KNeighborsRegressor):
    """ver4-25, R8 (design blueprint SS2, Stage 6): ordinary K-nearest-
    neighbours has no notion of a WEIGHTED fit - `.fit()` just stores the
    training rows verbatim, so BLUE-reliability weighting for KNN has to
    live in `.predict()` instead, by folding the shared fitting weight
    `w_fit` into the neighbour average alongside whatever `weights=`
    ('uniform'/'distance') scheme was already configured (blueprint's own
    per-model mechanism table: "kneighbors() then np.average(y_nb,
    weights=neighbour_weight x w_fit[nb])").

    `fit(X, y, sample_weight=None)` stores `sample_weight` (aligned to
    `X`'s own row order, i.e. this task's `id_train` order) alongside the
    ordinary scikit-learn fit; `predict()` is a strict, byte-identical
    pass-through to `KNeighborsRegressor.predict()` whenever no
    `sample_weight` was given at fit time (I11) - the weighted branch
    only ever runs when this task's BLUE weighting is actually on.
    """

    def fit(self, X, y, sample_weight=None):
        self._blue_sample_weight = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self._blue_fit_y = np.asarray(y, dtype=float)
        return super().fit(X, y)

    def predict(self, X):
        if getattr(self, '_blue_sample_weight', None) is None:
            return super().predict(X)
        neigh_dist, neigh_ind = self.kneighbors(X)
        y_train = self._blue_fit_y
        if self.weights == 'distance':
            with np.errstate(divide='ignore'):
                inv_dist = 1.0 / neigh_dist
            exact_hit = ~np.isfinite(inv_dist)  # a query exactly at a training point
            row_has_exact_hit = np.any(exact_hit, axis=1)
            dist_w = np.where(np.isfinite(inv_dist), inv_dist, 0.0)
            if np.any(row_has_exact_hit):
                # sklearn's own convention: a row with an exact-distance-0
                # neighbour uses ONLY that neighbour (else 0/0 elsewhere).
                dist_w[row_has_exact_hit] = exact_hit[row_has_exact_hit].astype(float)
        else:
            dist_w = np.ones_like(neigh_dist)
        neighbour_sample_w = self._blue_sample_weight[neigh_ind]
        combined = dist_w * neighbour_sample_w
        numerator = np.sum(combined * y_train[neigh_ind], axis=1)
        denominator = np.sum(combined, axis=1)
        fallback = float(np.mean(y_train)) if y_train.size else 0.0
        return np.where(denominator > 0, numerator / np.where(denominator > 0, denominator, 1.0), fallback)

#: Upper bound on the number of (query row x training row) distances held at
#: once by `DeterministicKNNPredict.__call__` (~32 MB of float64 per work array).
_DET_KNN_BLOCK_ELEMENTS = 4_000_000


class DeterministicKNNPredict:
    """ver4-43 (R1, defect S5): a K-nearest-neighbours predictor whose output
    for a row never depends on which other rows share its batch, for the
    `baseline_shap` explanations.

    Why: scikit-learn's `KNeighborsRegressor.predict` breaks distance ties at
    the k-th neighbour in an order that depends on how the query batch is
    blocked. 0/1/2 genotype coding makes such ties common (measured: ~1.5% of
    hybrid rows got a different prediction when predicted in a batch than when
    predicted alone), so the four terms of a `baseline_shap` pair difference -
    which are evaluated in different batches - carried tie noise and could
    produce spurious non-zero pairs. Here ties at the k-th distance go to the
    LOWER training-row index, whatever else is in the batch.

    It mirrors `WeightedKNeighborsRegressor.predict` for the same `n_neighbors`,
    `weights` ('uniform' | 'distance', with sklearn's exact-hit rule: a query
    at distance 0 from a training row uses only those rows), Minkowski `p` and
    optional BLUE fit weights `sample_weight` (folded into the neighbour
    average; a non-positive denominator returns mean(y)). It equals sklearn
    wherever sklearn has no tie ambiguity, up to floating-point summation order.

    Exactness note: for p=2 the squared distances come from a matrix product;
    with integer genotype coding every intermediate is an exactly representable
    integer, so equal distances compare exactly equal (the point of the class).
    With continuous coding exact ties are measure-zero.

    Parameters
    ----------
    X_ref : (n_ref, M) array-like - training rows (the shortlist columns).
    y_ref : (n_ref,) array-like - training targets.
    n_neighbors : int - k (must be <= n_ref).
    weights : {'uniform', 'distance'}.
    p : float - Minkowski power (2 = Euclidean, 1 = Manhattan).
    sample_weight : (n_ref,) array-like or None - BLUE fit weights.

    Raises
    ------
    ValueError
        On inconsistent shapes, k outside [1, n_ref], an unknown `weights`
        value or a non-positive `p`.
    """

    def __init__(self, X_ref, y_ref, n_neighbors, weights, p, sample_weight=None):
        X_ref = np.ascontiguousarray(np.asarray(X_ref, dtype=float))
        y_ref = np.asarray(y_ref, dtype=float).reshape(-1)
        if X_ref.ndim != 2 or X_ref.shape[0] != y_ref.shape[0]:
            raise ValueError(
                f"DeterministicKNNPredict: X_ref {X_ref.shape} and y_ref {y_ref.shape} must "
                f"describe the same training rows.")
        if not 1 <= int(n_neighbors) <= X_ref.shape[0]:
            raise ValueError(
                f"DeterministicKNNPredict: n_neighbors={n_neighbors} must be between 1 and the "
                f"number of training rows ({X_ref.shape[0]}).")
        if weights not in ('uniform', 'distance'):
            raise ValueError(
                f"DeterministicKNNPredict: weights must be 'uniform' or 'distance', got {weights!r}.")
        if not float(p) > 0:
            raise ValueError(f"DeterministicKNNPredict: p must be positive, got {p!r}.")
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float).reshape(-1)
            if sample_weight.shape[0] != X_ref.shape[0]:
                raise ValueError(
                    f"DeterministicKNNPredict: sample_weight has {sample_weight.shape[0]} entries "
                    f"for {X_ref.shape[0]} training rows.")
        self.X_ref = X_ref
        self.y_ref = y_ref
        self.k = int(n_neighbors)
        self.weights = weights
        self.p = float(p)
        self.sample_weight = sample_weight
        self._ref_sq = np.einsum('ij,ij->i', X_ref, X_ref)

    def _rank_distances(self, X):
        """(m, n_ref) distances to rank by - squared Euclidean for p=2, else
        the p-th-power sum (monotone in the true distance either way)."""
        if self.p == 2.0:
            D = (np.einsum('ij,ij->i', X, X)[:, np.newaxis] - 2.0 * (X @ self.X_ref.T)
                 + self._ref_sq[np.newaxis, :])
            np.clip(D, 0.0, None, out=D)
            return D
        D = np.empty((X.shape[0], self.X_ref.shape[0]), dtype=float)
        step = max(1, 2_000_000 // max(1, self.X_ref.shape[0] * self.X_ref.shape[1]))
        for start in range(0, X.shape[0], step):
            diff = np.abs(X[start:start + step, np.newaxis, :] - self.X_ref[np.newaxis, :, :])
            D[start:start + step] = (diff ** self.p).sum(axis=2)
        return D

    def _true_distance(self, D_selected):
        return np.sqrt(D_selected) if self.p == 2.0 else D_selected ** (1.0 / self.p)

    def _predict_block(self, X):
        D = self._rank_distances(X)
        kth = np.partition(D, self.k - 1, axis=1)[:, self.k - 1]
        less = D < kth[:, np.newaxis]
        tie = D == kth[:, np.newaxis]
        need = self.k - less.sum(axis=1)
        take_tie = tie & (np.cumsum(tie, axis=1) <= need[:, np.newaxis])   # lowest indices first
        chosen = less | take_tie
        rows, cols = np.nonzero(chosen)
        neighbours = cols.reshape(X.shape[0], self.k)
        dist = self._true_distance(D[rows, cols].reshape(X.shape[0], self.k))
        if self.weights == 'distance':
            with np.errstate(divide='ignore'):
                inv = 1.0 / dist
            exact = ~np.isfinite(inv)
            has_exact = exact.any(axis=1)
            w = np.where(np.isfinite(inv), inv, 0.0)
            if has_exact.any():
                w[has_exact] = exact[has_exact].astype(float)
        else:
            w = np.ones_like(dist)
        if self.sample_weight is not None:
            w = w * self.sample_weight[neighbours]
        numerator = (w * self.y_ref[neighbours]).sum(axis=1)
        denominator = w.sum(axis=1)
        fallback = float(self.y_ref.mean()) if self.y_ref.size else 0.0
        safe = np.where(denominator > 0, denominator, 1.0)
        return np.where(denominator > 0, numerator / safe, fallback)

    def __call__(self, X):
        """Predictions for `X` (m, M) or a single row (M,); 1-D array (m,)."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.X_ref.shape[1]:
            raise ValueError(
                f"DeterministicKNNPredict: expected X with {self.X_ref.shape[1]} column(s), "
                f"got shape {X.shape}.")
        out = np.empty(X.shape[0], dtype=float)
        step = max(1, _DET_KNN_BLOCK_ELEMENTS // self.X_ref.shape[0])
        for start in range(0, X.shape[0], step):
            out[start:start + step] = self._predict_block(X[start:start + step])
        return out


# ver4-4 R3.h: the row-fan-out SHAP helper that used to live here as
# `_parallel_kernel_shap_values()` - byte-for-byte identical to `models/
# SVR.py`'s own copy of the same function - has been promoted, verbatim
# for this module's own `interaction=False` use, into `pipeline_utils.
# parallel_shap_values()`. One shared copy instead of two duplicated
# ones. Call sites in this file are unchanged other than the function
# name itself (see below).


def _knn_regressor_cls(use_gpu):
    """Return (KNeighborsRegressor class, is_gpu) - cuML's GPU-backed
    KNeighborsRegressor when `use_gpu` is True AND cuML is importable,
    else scikit-learn's own CPU implementation. Never raises - falls back
    to CPU on any import/compatibility failure (Phase 2, Requirement 6)."""
    if use_gpu and cuml_gpu_usable():
        try:
            from cuml.neighbors import KNeighborsRegressor as CumlKNN
            return CumlKNN, True
        except Exception as exc:
            print(f"[KNN] USE_GPU_SKLEARN requested but cuML is unavailable "
                  f"({exc}) - falling back to scikit-learn's CPU KNeighborsRegressor.")
    return KNeighborsRegressor, False

# shap.KernelExplainer is model-agnostic - KNN has no fast, structure-exploiting
# explainer the way tree models do - so its cost is driven entirely by how many
# knn.predict() calls it needs, which multiply together as:
#     (# background samples) x (# coalition samples per explanation) x (# explained samples)
# This is the exact same issue fixed in SVR.py: the original implementation used
# `shapley_num` for BOTH the background size and the explained-sample count, and
# left the coalition count (`nsamples`) at SHAP's default, which scales as
# roughly 2*n_features+2048. With thousands of markers that is tens of
# thousands of coalitions per explained sample - see SVR.py for the full
# explanation and measurements (there it turned an 8,000-marker run into
# 14+ days).
#
# Fixes applied below (identical strategy to SVR.py):
#  - a small, fixed-size background summary (shap.kmeans) - KernelExplainer
#    only needs this to marginalise out features, it does not need to be
#    anywhere near as large as the explained-sample count
#  - a fixed, modest `nsamples` budget instead of SHAP's feature-count-scaling
#    default
#  - restricting the explanation itself to the top-importance markers (by
#    absolute correlation with the trait - KNN has no built-in importance
#    measure either), refitting a lightweight KNN on just those
# All three are user-configurable via params[5:8] (GUI: "Max markers considered
# for Shapley scores", "Background sample size for Shapley scores", "Number of
# coalition samples for Shapley scores").

# Update ID ver4-5, R2 Stage 9 (blueprint §4.2 Layer 3): identical
# rationale to models/SVR.py's own note - KNN has no native pairwise-
# interaction API either, so interactions come from the same shared
# Friedman H-statistic route (models.interaction_extraction.
# h_statistic_interactions), shortlisted and background-sampled the same
# way. See SVR.py's own note for the full explanation.


# ver4-67: tuning of models/knn_shapley.knn_shapley_effects for the
# 'baseline_shap' marker-effect route (hierarchical players: 256 genome-order
# blocks, then up to ~1000 extra single-marker slots, split only where the
# block's Shapley mass is statistically significant). n_players is raised to
# 'all' (exact marker-level game) automatically when M is small.
_KNN_SHAPLEY_DEFAULTS = dict(max_walks=200, n_players=256, refine_markers=1000)


def KNN(train, valid, test, params, *, sample_weight_ctx=None):
    
    n_neighbours = params[0]
    # The two standard KNN tuning knobs beyond n_neighbours: weights
    # ('uniform' treats every neighbour equally; 'distance' weights closer
    # neighbours more heavily) and p (Minkowski distance power - 1 is
    # Manhattan, 2 is Euclidean).
    weights = params[1]
    p = params[2]
    get_effect = params[3]
    shapley_num = params[4]
    max_shap_features = params[5]
    shap_background_size = params[6]
    shap_nsamples = params[7]
    # Update ID ver4-5, R2 Stage 9: appended, read defensively (I5) - see
    # models/SVR.py's own identical note.
    get_interaction = params[8] if len(params) > 8 else False
    max_interaction_features = params[9] if len(params) > 9 else 500
    interaction_background = params[10] if len(params) > 10 else 100
    interaction_top = params[11] if len(params) > 11 else 'all'
    # Update ID ver4-6, R1/R1b (blueprint §2.4/§2.10.4): appended, read
    # defensively (I5) - see models/SVR.py's own identical note.
    interaction_screen = params[12] if len(params) > 12 else 'off'
    interaction_screen_top = params[13] if len(params) > 13 else 2.0
    interaction_grid_resolution = params[14] if len(params) > 14 else 3
    # ver4-25, Stage 8 (design blueprint R11) - see models/SVR.py's own
    # identical field.
    interaction_method = params[15] if len(params) > 15 else 'friedman_h'
    interaction_shapiq_orders = params[16] if len(params) > 16 else '2'
    interaction_shapiq_index = params[17] if len(params) > 17 else 'k-SII'
    interaction_shapiq_budget = params[18] if len(params) > 18 else 4096
    # ver4-25, Stage 7 (design blueprint R13) - see models/SVR.py's own
    # identical field.
    effect_method = params[19] if len(params) > 19 else 'kernel_shap'
    # This update - see models/RF.py's identical field for the full
    # rationale. Appended, read defensively (I5).
    use_effect_for_interaction_shortlist = params[20] if len(params) > 20 else True

    #Split the data sets into x and y here as specified in the original code
    train_x, train_y = train.iloc[:,:-1], train.iloc[:,-1]
    if valid.shape[0] != 0:
        valid_x, valid_y = valid.iloc[:,:-1], valid.iloc[:,-1]
    test_x, test_y = test.iloc[:,:-1], test.iloc[:,-1]
    # ver4-25, R8 - see models/RF.py's identical comment. KNN has no
    # native weighted-fit concept, so this feeds WeightedKNeighbors
    # Regressor's own predict()-time neighbour average instead - see
    # that class's own docstring above.
    _w_fit = sample_weight_ctx.w_fit if sample_weight_ctx is not None else None
    
    # Phase 2, Requirement 6: n_jobs (neighbor search) and an optional
    # cuML GPU backend are resolved from the run's shared compute-resource
    # settings - a CPU-only node falls back to scikit-learn's own
    # multi-threaded neighbor search, exactly as before this option existed.
    _resources = get_active_compute_resources()
    _knn_cls, _is_gpu = _knn_regressor_cls(_resources['use_gpu_sklearn'])

    #Develop & evaluate a model here as specified in the original code
    if _is_gpu and _w_fit is not None:
        # ver4-25, R8: cuML's KNeighborsRegressor has no weighted-fit
        # equivalent at all - fall back to the CPU WeightedKNeighbors
        # Regressor so the fit is genuinely weighted (never silently
        # unweighted - blueprint's own per-model mechanism table).
        print("[KNN] NOTE: BLUE-reliability weighting is on - using the CPU "
              "WeightedKNeighborsRegressor instead of cuML's GPU KNeighborsRegressor "
              "for this task.")
        knn = WeightedKNeighborsRegressor(n_neighbors=n_neighbours, weights=weights, p=p,
                                           n_jobs=_resources['n_jobs'])
        _is_gpu = False
    elif _is_gpu:
        knn = _knn_cls(n_neighbors=n_neighbours, weights=weights)
    else:
        knn = WeightedKNeighborsRegressor(n_neighbors=n_neighbours, weights=weights, p=p,
                                           n_jobs=_resources['n_jobs'])
    knn.fit(train_x, train_y, sample_weight=_w_fit) if not _is_gpu else knn.fit(train_x, train_y)
    
    predicted = np.asarray(knn.predict(test_x)).ravel()
    if valid.shape[0] != 0:
        predicted_valid = np.asarray(knn.predict(valid_x)).ravel()
    else:
        predicted_valid = []
    predicted_train = np.asarray(knn.predict(train_x)).ravel()

    ## Calculate the metrics
    actual_test = test_y.values.tolist()
    mse = mean_squared_error(actual_test, predicted)
    r = pearsonr(actual_test, predicted)[0]
    
    with timed_stage('xai_effect:KNN'):
        if get_effect == True:
            # Defensive clamp: avoid shap.sample() erroring if shapley_num exceeds
            # the number of available test rows.
            shapley_num = min(shapley_num, test_x.shape[0])

            n_features = train_x.shape[1]
            if max_shap_features != 'all' and n_features > max_shap_features:
                # Cheap, model-agnostic importance proxy: absolute correlation
                # with the trait (O(N*M), no extra model fitting needed to rank
                # candidates).
                correlations = train_x.corrwith(train_y).abs().fillna(0)
                top_features = correlations.sort_values(ascending=False).index[:max_shap_features]

                knn_effect = WeightedKNeighborsRegressor(n_neighbors=n_neighbours, weights=weights, p=p,
                                                          n_jobs=_resources['n_jobs'])
                knn_effect.fit(train_x[top_features], train_y, sample_weight=_w_fit)
                effect_train_x = train_x[top_features]
                effect_test_x = test_x[top_features]
            else:
                # 'all' - use every marker, no shortlist. KernelExplainer needs a
                # plain CPU .predict() callable, so a GPU-fitted cuML model is
                # refit on CPU here for the explanation step.
                if _is_gpu:
                    knn_effect = WeightedKNeighborsRegressor(n_neighbors=n_neighbours, weights=weights, p=p,
                                                              n_jobs=_resources['n_jobs'])
                    knn_effect.fit(train_x, train_y, sample_weight=_w_fit)
                else:
                    knn_effect = knn
                top_features = train_x.columns
                effect_train_x = train_x
                effect_test_x = test_x

            # Bugfix: shap.kmeans(effect_train_x, background_size) used to be
            # called directly here and could raise
            # ValueError('# of weights must match data matrix!') whenever
            # background_size met or exceeded the number of DISTINCT rows in
            # effect_train_x - routine for 0/1/2-coded marker data once
            # narrowed to a shortlist. See pipeline_utils.
            # safe_shap_kmeans_background's docstring (in models/SVR.py's
            # identical call site) for the full root cause; this wrapper
            # clamps to the distinct-row count instead of raising.
            # Bug fix (this update - "KNN marker effects come back almost
            # entirely zero"): every occlusion/coalition-masking route below
            # this comment used to be tried FIRST for KNN, exactly like every
            # other 'kernel_shap' caller in this codebase (SVR, RKHS). That is
            # wrong specifically for KNN - see pipeline_utils.knn_relief_effects()'s
            # own module-level comment for the full "why" (KNN's prediction is a
            # discrete function of which k training rows are nearest, so masking
            # a single marker to a baseline value essentially never changes it
            # once the marker panel is more than a few dozen columns wide,
            # confirmed empirically against a 16,379-marker reproduction of this
            # exact symptom returning an EXACTLY zero effect vector). KNN's own
            # effect step now uses the Relief-style neighbour-distance
            # decomposition unconditionally for 'kernel_shap' - it is both
            # correct (never structurally zero) and cheaper (one `.kneighbors()`
            # + one `.predict()` call per explained row, vs. up to
            # `n_features / feat_chunk` `.predict()` calls under the old route).
            def _knn_relief_effect():
                return knn_relief_effects(
                    knn_effect, effect_train_x, train_y, shap.sample(effect_test_x, shapley_num),
                    log_prefix="[KNN]",
                )

            if effect_method == 'baseline_shap':
                # ver4-67: TRUE interventional Shapley values for KNN marker
                # effects (same game - real training-row background,
                # v_x(S)=E_z f(x_S, z_rest) - that this file's own
                # 'baseline_shap' INTERACTION route already uses), computed by
                # models/knn_shapley.py: KNN's prediction depends on the markers
                # only through additive per-training-row distances, so a
                # permutation walk costs O(n_train) per step, independent of
                # the marker count. Hierarchical, significance-driven players
                # (genome-order blocks -> sub-blocks -> single markers) keep the
                # cost bounded for M > 10,000. Any failure falls back to the
                # Relief-style route (never a new failure mode).
                try:
                    _cols = list(train_x.columns)
                    _pos = train_x.columns.get_indexer(top_features)
                    # `top_features` may be in correlation order; blocks must be
                    # contiguous in GENOME (column) order.
                    _marker_order = np.argsort(_pos, kind='stable')
                    _X_ex = np.asarray(shap.sample(effect_test_x, shapley_num), dtype=float)
                    _bg_n = int(min(shap_background_size, effect_train_x.shape[0])) \
                        if isinstance(shap_background_size, (int, np.integer)) else int(effect_train_x.shape[0])
                    _bg_n = max(1, _bg_n)
                    _bg_idx = np.random.default_rng(0).choice(effect_train_x.shape[0], _bg_n, replace=False)
                    _X_bg = np.asarray(effect_train_x, dtype=float)[_bg_idx]
                    _mem = available_memory_bytes()
                    effect_scores, _det = knn_shapley_effects(
                        np.asarray(effect_train_x, dtype=float), np.asarray(train_y, dtype=float),
                        _X_ex, _X_bg, n_neighbors=n_neighbours, weights=weights, p=p,
                        sample_weight=_w_fit, marker_order=_marker_order,
                        max_bytes=(0.5 * _mem if _mem else 2.0e9), log_prefix="[KNN]",
                        return_details=True,
                        **_KNN_SHAPLEY_DEFAULTS,
                    )
                    print(f"[KNN] NOTE: 'baseline_shap' marker effects: true interventional Shapley "
                          f"values for {len(effect_scores)} marker(s) from {_X_ex.shape[0]} explained "
                          f"row(s) x {_X_bg.shape[0]} real background row(s) in {_det.get('seconds', 0):.1f}s "
                          f"({int(np.sum(_det['resolved']))} marker(s) individually resolved; the rest "
                          f"carry their block's Shapley share). Not numerically comparable with "
                          f"'kernel_shap' (Relief) effects.")
                except Exception as exc:
                    print(f"[KNN] NOTE: Shapley marker-effect computation failed ({exc!r}) - falling "
                          f"back to the Relief-style neighbour-distance route for this task instead.")
                    effect_scores = _knn_relief_effect()
            elif effect_method == 'shapiq':
                # ver4-25, Stage 7 (R13) - see models/SVR.py's own identical
                # block and pipeline_utils.shapiq_marker_effects()'s own
                # module-level header comment for the disclosed "written but
                # not runtime-verified against a real shapiq install" caveat.
                # shapiq's own model-agnostic TabularExplainer is itself a
                # coalition-sampling estimator, so it is not immune to the SAME
                # "single/subset-feature masking rarely flips KNN's neighbour
                # set" failure mode above - falling back to the Relief-style
                # route (rather than the old, equally-affected
                # shap.KernelExplainer path) on any failure keeps this branch
                # from silently landing back on a degenerate result.
                try:
                    effect_scores = shapiq_marker_effects(
                        knn_effect.predict, effect_train_x, shap.sample(effect_test_x, shapley_num),
                        background_size=shap_background_size, budget=shap_nsamples,
                    )
                except Exception as exc:
                    print(f"[KNN] NOTE: shapiq marker-effect computation failed ({exc!r}) - falling "
                          f"back to the Relief-style neighbour-distance route (this codebase's own "
                          f"'kernel_shap' default for KNN) for this task instead. ('shapiq' is an "
                          f"optional dependency - 'pip install shapiq' if not yet installed; this may "
                          f"also mean this shapiq version's API differs from what this codebase "
                          f"expects.)")
                    effect_scores = _knn_relief_effect()
            else:
                effect_scores = _knn_relief_effect()

            # Reassemble into a full-width vector (one entry per marker, in the
            # original train_x column order, with 0 for any marker that wasn't
            # in the shortlist above) - genomic_prediction.py assigns column
            # names positionally from the full marker list, so the returned
            # DataFrame must always have exactly n_features columns regardless
            # of how many markers were actually explained.
            effect_full = pd.Series(0.0, index=train_x.columns)
            effect_full.loc[top_features] = effect_scores
            effect = pd.DataFrame(effect_full).T
        else:
            effect = pd.DataFrame()

    # This update - see models/SVR.py's identical block for the full
    # rationale. 'Max markers considered for Shapley scores' (field 5) is
    # this model's own effect-shortlist cap, shared by both 'kernel_shap'
    # and 'shapiq'.
    _effect_marker_cap = max_shap_features if get_effect else None
    _use_effect_shortlist = should_use_effect_for_interaction_shortlist(
        use_effect_for_interaction_shortlist, max_interaction_features, _effect_marker_cap,
    )
    # Optimisation (behaviour-preserving): the |correlation| ranking was evaluated
    # eagerly on EVERY fit - including hyper-parameter tuning trials and
    # Super-Learner cross-fit units, which never reach the interaction step - but
    # it is only read inside the `get_interaction` shortlist branch below. It is
    # now computed there, on demand; the value used is unchanged.
    def _interaction_importance_fn():
        return (
            effect.iloc[0].abs() if _use_effect_shortlist
            else train_x.corrwith(train_y).abs().fillna(0)
        )

    with timed_stage('xai_inter:KNN'):
        if get_interaction == True:
            n_features = train_x.shape[1]
            if max_interaction_features != 'all' and n_features > max_interaction_features:
                # Ranked by this task's own already-computed marker-effect
                # values instead of |correlation| when eligible (see
                # '_use_effect_shortlist' above) - otherwise byte-identical
                # to before this checkbox existed (I11).
                interaction_features = _interaction_importance_fn().sort_values(ascending=False).index[:max_interaction_features]
            else:
                interaction_features = train_x.columns

            knn_interaction = WeightedKNeighborsRegressor(n_neighbors=n_neighbours, weights=weights, p=p,
                                                           n_jobs=_resources['n_jobs'])
            knn_interaction.fit(train_x[interaction_features], train_y, sample_weight=_w_fit)

            # Bugfix: see models/SVR.py's own identical note - shapiq's
            # TabularExplainer bills a fixed `interaction_shapiq_budget` cost
            # PER EXPLAINED ROW, so the 'shapiq' branch's own row count needs
            # to be bounded the same way models/RF.py's/RKHS's own shapiq
            # interaction routes already are. Reuses this file's own
            # `shapley_num` field (already used for this file's own effect
            # step above) rather than adding a new hyperparameter.
            _interaction_shapley_num = min(shapley_num, test_x.shape[0])

            def _h_statistic_interaction_sample():
                interaction_pairs = [
                    (a, b) for a in range(len(interaction_features)) for b in range(a + 1, len(interaction_features))
                ]
                # Bugfix: n_jobs was previously never forwarded here, so this step
                # ran single-threaded regardless of the run's compute-resource
                # settings, unlike this file's own neighbour-search fit and SHAP
                # marker-effect step (both already use _resources['n_jobs']) - see
                # models.interaction_extraction.h_statistic_interactions()'s own
                # n_jobs docstring for what this now does. Mirrors how models/RF.py
                # forwards n_jobs into tree_shap_interactions().
                return h_statistic_interactions(
                    knn_interaction.predict, test_x[interaction_features], list(interaction_features),
                    pairs=interaction_pairs, n_background=min(interaction_background, test_x.shape[0]),
                    n_jobs=_resources['n_jobs'],
                    # Update ID ver4-6, R1/R1b - see models/SVR.py's own identical note.
                    screen=(interaction_screen if interaction_screen != 'off' else None),
                    screen_keep=(float(interaction_screen_top) / 100.0 if interaction_screen_top != 'all' else 1.0),
                    grid_resolution=interaction_grid_resolution,
                )

            # ver4-25, Stage 8/9 (design blueprint R11/R12) - see models/SVR.py's
            # own identical 'shapiq' branch: every requested order's own rows
            # are returned, canonicalised; genomic_prediction.py's Step 10
            # splits k=2 (Interaction.csv) from k>=3 (Interaction_higher_
            # order.csv).
            def _baseline_shap_interaction_sample():
                # NEW route (user feedback, backed by this project's own
                # experimental results - see the baseline_shap_interactions
                # import comment above): background rows are REAL training
                # individuals (never a synthetic grid), so linkage
                # disequilibrium between markers is preserved exactly, and
                # cost is fixed (no coalition-sampling budget) - see that
                # function's own Cost section.
                interaction_pairs = [
                    (a, b) for a in range(len(interaction_features)) for b in range(a + 1, len(interaction_features))
                ]
                _bg_size = min(interaction_background, train_x.shape[0])
                X_background = np.asarray(shap.sample(train_x[interaction_features], _bg_size), dtype=float)
                X_explain = np.asarray(
                    shap.sample(test_x[interaction_features], _interaction_shapley_num), dtype=float,
                )
                # ver4-43 (R1, defect S5): under ACCELERATION['baseline_shap_engine']
                # == 'fast', the hybrid rows are scored by a tie-deterministic
                # predictor (ties at the k-th distance -> lower training-row
                # index) instead of scikit-learn's batch-composition-dependent
                # one, so the four terms of a pair difference no longer carry
                # tie noise. 'legacy' (the default for an absent key) keeps
                # `knn_interaction.predict` exactly as before (I11).
                _predict_fn = knn_interaction.predict
                _engine = _resources.get('accel_baseline_shap_engine', 'legacy')
                if _engine == 'fast':
                    try:
                        _predict_fn = DeterministicKNNPredict(
                            train_x[interaction_features].to_numpy(dtype=float),
                            train_y.to_numpy(dtype=float), n_neighbours, weights, p, _w_fit,
                        )
                    except Exception as exc:
                        print(f"[KNN] NOTE: DeterministicKNNPredict could not be built ({exc!r}) - "
                              f"baseline_shap uses scikit-learn's own predict() for this task "
                              f"instead (distance ties may add noise to the pair values).")
                # ver4-67: closed-form O(n_train)-per-hybrid-row evaluator
                # (models/knn_shapley.KNNBaselineShapFastEval), tie-identical
                # to DeterministicKNNPredict. Only under the 'fast' engine; any
                # failure keeps the predict_fn route (I11).
                _fast_eval = None
                if _engine == 'fast':
                    try:
                        _fast_eval = KNNBaselineShapFastEval(
                            train_x[interaction_features].to_numpy(dtype=float),
                            train_y.to_numpy(dtype=float), X_background, n_neighbours,
                            weights=weights, p=p, sample_weight=_w_fit,
                        )
                    except Exception as exc:
                        print(f"[KNN] NOTE: KNNBaselineShapFastEval could not be built ({exc!r}) - "
                              f"baseline_shap evaluates the predictor directly for this task instead.")
                return baseline_shap_interactions(
                    _predict_fn, X_explain, X_background, list(interaction_features),
                    pairs=interaction_pairs, n_jobs=_resources['n_jobs'],
                    kernel_fast_eval=_fast_eval,
                    # Update (perf): reuses the SAME pre-existing 'Marker
                    # interaction screen' fields the 'friedman_h' branch
                    # above already reads (interaction_screen/
                    # interaction_screen_top) - no new hyperparameter. 'off'
                    # (the field's own default) resolves to screen=None,
                    # byte-identical to this parameter not existing (I11) -
                    # a no-op for every existing config, and a real lever
                    # (cuts the O(pairs) cost directly) for anyone who turns
                    # screening on because the unscreened cost was
                    # prohibitive on a large marker shortlist.
                    screen=(interaction_screen if interaction_screen != 'off' else None),
                    screen_keep=(float(interaction_screen_top) / 100.0 if interaction_screen_top != 'all' else 1.0),
                )

            if interaction_method == 'baseline_shap':
                try:
                    interaction_sample = _baseline_shap_interaction_sample()
                    print(f"[KNN] NOTE: baseline-referenced pairwise interaction extraction produced "
                          f"{interaction_sample.shape[0]} pair(s) from a {len(interaction_features)}-"
                          f"marker shortlist - directly comparable (same units/reduction) to RF's own "
                          f"TreeSHAP interaction table; no coalition sampling, no synthetic joint-"
                          f"genotype grid.")
                except Exception as exc:
                    print(f"[KNN] NOTE: baseline_shap interaction computation failed ({exc!r}) - "
                          f"falling back to 'friedman_h' (Friedman's H-statistic) for this task "
                          f"instead.")
                    interaction_sample = _h_statistic_interaction_sample()
            elif interaction_method == 'shapiq':
                try:
                    _orders = tuple(
                        int(_o.strip()) for _o in str(interaction_shapiq_orders).split(',') if _o.strip()
                    ) or (2,)
                    # Design blueprint R1: fan the per-row shapiq explain loop
                    # out across this task's own resolved worker count, gated
                    # by `shapiq_row_parallel` (`SHAPIQ_ROW_PARALLEL`, default
                    # True) - `n_jobs=1` (the flag off, or a single-core node)
                    # is byte-identical to this route's pre-R1 serial loop (I11).
                    _shapiq_n_jobs = (
                        _resources['n_jobs'] if _resources.get('shapiq_row_parallel', True) else 1
                    )
                    # ver4-28: order-aware shrink of the explained-ROW count too
                    # - see models/SVR.py's own identical note. Gated by the
                    # SAME `shapiq_adaptive_depth` flag (no new config key);
                    # `max(_orders) <= 2` (I11) leaves `_interaction_shapley_num`
                    # completely unchanged.
                    _shapiq_explain_n = (
                        resolve_shapiq_explain_row_count(max(_orders), _interaction_shapley_num)
                        if _resources.get('shapiq_adaptive_depth', True) else _interaction_shapley_num
                    )
                    raw = shapiq_pairwise_interactions(
                        knn_interaction.predict, train_x[interaction_features],
                        shap.sample(test_x[interaction_features], _shapiq_explain_n),
                        list(interaction_features),
                        orders=_orders, background_size=min(interaction_background, train_x.shape[0]),
                        budget=interaction_shapiq_budget, index=interaction_shapiq_index,
                        n_jobs=_shapiq_n_jobs,
                    )
                    interaction_sample = canonicalize_shapiq_table(raw)
                except Exception as exc:
                    print(f"[KNN] NOTE: shapiq interaction computation failed ({exc!r}) - falling "
                          f"back to 'friedman_h' (Friedman's H-statistic) for this task instead.")
                    interaction_sample = _h_statistic_interaction_sample()
            else:
                interaction_sample = _h_statistic_interaction_sample()
            interaction_sample = top_select_by_order(interaction_sample, 'percentage', interaction_top)

            # Requirements3.md item 4 - see models/SVR.py's own identical
            # note: 'shapiq' (KNN, like SVR, has no native pairwise-SHAP API)
            # genuinely finds fewer nonzero pairs than a dense SHAP-based
            # route at the same task scale, so its own Interaction.csv row
            # count needs the SAME parity padding RF.py's 'shapiq_tree'
            # already applies, against the SAME candidate universe this
            # branch itself searched (interaction_features) at the SAME
            # threshold (interaction_top) - never applied to 'friedman_h'
            # (I11).
            if interaction_method == 'shapiq':
                interaction_sample = pad_shapiq_pairs_to_shap_parity(
                    interaction_sample, list(interaction_features), interaction_top,
                )
        else:
            interaction_sample = pd.DataFrame()

    return r, mse, effect, interaction_sample, predicted, predicted_valid, predicted_train
