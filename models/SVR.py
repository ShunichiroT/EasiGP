from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import shap

from pipeline_utils import (cuml_gpu_usable, get_active_compute_resources, parallel_shap_values,
                            safe_shap_kmeans_background, fit_with_sample_weight,
                            shapiq_marker_effects, shapiq_pairwise_interactions,
                            resolve_shapiq_explain_row_count,
                            fast_kernel_shap_effects, FAST_KERNEL_SHAP_MARKER_THRESHOLD,
                            timed_stage)
from models.interaction_extraction import (
    h_statistic_interactions, top_select_by_order, canonicalize_shapiq_table,
    pad_shapiq_pairs_to_shap_parity, should_use_effect_for_interaction_shortlist,
    # ver4-43 (R2): the per-marker 'baseline_shap' EFFECT - the single-marker
    # term of the interaction estimator just below (same real background
    # rows, same exact deduplication).
    baseline_shap_effects,
    # This update (user feedback, backed by this project's own experimental
    # results: "friedman-h was not reliable ... did not capture correct
    # interactions" and "shapiq ... too computationally expensive"): a
    # FOURTH marker-pair interaction route (alongside 'kernel_hessian'
    # above), directly comparable in units/aggregation to models/RF.py's
    # own TreeSHAP interaction table (same `abs_then_sum` reduction), and
    # cheaper than shapiq at equal background-size settings - see
    # baseline_shap_interactions()'s own module-level docstring (models/
    # interaction_extraction.py) for the full rationale (LD-safe: never
    # evaluates a synthetic, independence-assuming joint genotype
    # combination the way Friedman's H's own grid does; no coalition-
    # sampling budget the way shapiq needs).
    baseline_shap_interactions,
)
# This update (user feedback: "shapiq is unreliable and should not rely
# on this"): a THIRD, fully closed-form/deterministic marker-pair
# interaction route - originally 'rbf' only, see
# rbf_kernel_hessian_interactions()'s own module-level docstring (models/
# kernel_effects.py) for that derivation; a later update ("minimise the
# computational time of marker effects and interactions for other kernel
# cases: linear, poly, or sigmoid") extended the SAME closed-form idea to
# every other kernel this file supports - see kernel_effects.py's own
# module-level comment directly above linear_kernel_occlusion_effects()
# for that derivation. Unlike 'shapiq' (coalition sampling) it has zero
# run-to-run variance, and unlike 'friedman_h' (finite-difference over a
# background sample) it is the model's own EXACT analytic second
# derivative, not an estimate of one.
from models import kernel_effects

# Requirement 6 (design note): libsvm (scikit-learn's SVR backend) is not
# internally multi-threaded - there is no CPU n_jobs knob to set here, only
# two speed-up avenues: (a) Intel's `sklearnex` patch, which accelerates
# scikit-learn's own CPU SVR transparently via `sklearnex.patch_sklearn()`
# (no code-level model swap needed - it monkeypatches sklearn.svm.SVR in
# place), and (b) cuML's GPU-backed SVR. Both are entirely optional and
# fall back to plain scikit-learn if unavailable.
_SKLEARNEX_PATCHED = False

# ver4-4 R3.h: the row-fan-out SHAP helper that used to live here as
# `_parallel_kernel_shap_values()` (and, identically, as `models/KNN.py`'s
# own byte-for-byte duplicate of it) has been promoted, verbatim for this
# module's own `interaction=False` use, into `pipeline_utils.
# parallel_shap_values()` - one shared copy instead of two, plus a new
# `interaction=True` branch that `models/RF.py`, `models/
# GAT_prior_knowledge.py` and `Preprocess/data_driven_prior_network.py`
# now also use. Call sites in this file are unchanged other than the
# function name itself (see below).

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

def _maybe_patch_sklearnex():
    """Best-effort, one-time `sklearnex.patch_sklearn()` call - transparently
    accelerates scikit-learn's own CPU SVR (and other estimators) via
    Intel's oneDAL backend when the `sklearnex` package is installed. A
    silent no-op otherwise. Idempotent (patches at most once per process)."""
    global _SKLEARNEX_PATCHED
    if _SKLEARNEX_PATCHED:
        return
    try:
        from sklearnex import patch_sklearn
        patch_sklearn(['SVR'])
    except Exception:
        pass
    finally:
        _SKLEARNEX_PATCHED = True


def _resolve_svr_gamma(svr_model, gam, X_ref):
    """Resolve this SVR fit's own actual kernel bandwidth ('gamma') -
    factored out (this update, SVR non-'rbf'-kernel speedup) from what
    used to be an inline block duplicated once for the 'kernel_shap'
    marker-effect route and once more for the 'kernel_hessian' interaction
    route (both 'rbf'-only at the time) - now shared by every kernel that
    actually uses gamma ('rbf', 'poly', 'sigmoid' - 'linear' never calls
    this, since sklearn's own 'linear' kernel does not use gamma at all).
    Byte-identical logic to both original inline blocks (pure refactor,
    no behaviour change for 'rbf').

    Prefers sklearn's own already-resolved `_gamma` attribute (present
    regardless of whether the fit's own `gamma` constructor argument was a
    plain float or the 'scale'/'auto' string) - the SAME bandwidth
    `svr_model.predict()` itself actually uses, so this is never a second,
    possibly-inconsistent gamma. Falls back to resolving `gam` (this
    file's own hyperparameter, params[4]) directly only if `_gamma` is
    unexpectedly absent (defensive only - every real sklearn SVR fit sets
    it for a gamma-using kernel).
    """
    _svr_gamma = getattr(svr_model, '_gamma', None)
    if _svr_gamma is not None:
        return float(_svr_gamma)
    _n_gamma_features = max(1, np.asarray(X_ref).shape[1])
    if isinstance(gam, (int, float)):
        return float(gam)
    if gam == 'auto':
        return 1.0 / _n_gamma_features
    _var = np.asarray(X_ref, dtype=float).var()
    return 1.0 / (_n_gamma_features * _var) if _var > 0 else 1.0 / _n_gamma_features


def _svr_cls(use_gpu):
    """Return (SVR class, is_gpu) - cuML's GPU-backed SVR when `use_gpu`
    is True AND cuML is importable, else scikit-learn's own CPU SVR
    (optionally sklearnex-accelerated - see _maybe_patch_sklearnex()).
    Never raises - falls back to plain CPU scikit-learn on any import/
    compatibility failure (Phase 2, Requirement 6)."""
    if use_gpu and cuml_gpu_usable():
        try:
            from cuml.svm import SVR as CumlSVR
            return CumlSVR, True
        except Exception as exc:
            print(f"[SVR] USE_GPU_SKLEARN requested but cuML is unavailable "
                  f"({exc}) - falling back to (sklearnex-accelerated, if available) CPU SVR.")
    _maybe_patch_sklearnex()
    return SVR, False


def _resolve_eps_c(scale_eps_c, train_y, eps, con):
    """Return ``(eps_eff, con_eff, sd, warn)``: the epsilon and C that are
    actually handed to scikit-learn's / cuML's SVR.

    Why this exists (ver4-38, R3): SVR's epsilon and C act on the RAW
    phenotype scale. On a trait whose SD is far smaller than epsilon (the
    spec default is 0.5; a typical yield-ratio trait has SD ~0.05) every
    training point lies inside the epsilon-tube, so no support vector is
    created and the model predicts one constant for everybody. Multiplying
    both by ``sd = std(train_y)`` is exactly the same optimisation as fitting
    the SVR on ``y / sd`` and multiplying the predictions back by ``sd``:
    ``0.5*|w|^2 + C*sd * sum(max(0, |y - f| - eps*sd))
    = sd**2 * [0.5*|w'|^2 + C * sum(max(0, |y' - f'| - eps))]``
    with ``y' = y / sd``, ``f' = f / sd``, ``w' = w / sd``. The fitted
    object, ``dual_coef_``, ``support_vectors_`` and every explainer built on
    them therefore keep their meaning, and ``sample_weight`` (which scales C
    per row) is unaffected.

    Args:
        scale_eps_c: ``params[26]`` (False when the list predates ver4-38).
        train_y: the CURRENT task's training phenotype only (I7).
        eps, con: the user's / tuner's raw epsilon and C.

    Returns:
        ``(eps_eff, con_eff, sd, warn)``. When ``scale_eps_c`` is falsy the
        inputs are returned unchanged with ``sd = 1.0`` and ``warn = ''``
        (the legacy path). ``warn`` is a non-empty message when the SD was
        unusable (zero, NaN, inf, or fewer than 2 rows) and ``sd = 1.0`` was
        used instead; the caller prints it. Never raises.
    """
    if not scale_eps_c:
        return eps, con, 1.0, ''
    y_arr = np.asarray(train_y, dtype=float).ravel()
    y_arr = y_arr[np.isfinite(y_arr)]
    sd = float(np.std(y_arr)) if y_arr.size > 1 else 0.0
    if not np.isfinite(sd) or sd <= 0.0:
        return eps, con, 1.0, (
            f"training phenotype SD is {sd!r} over {y_arr.size} finite value(s), so "
            f"'Scale epsilon and C by the trait SD' cannot be applied - using SD = 1.0 "
            f"(epsilon and C unchanged)")
    return float(eps) * sd, float(con) * sd, sd, ''

# shap.KernelExplainer is model-agnostic - SVR has no fast, structure-exploiting
# explainer the way tree models do - so its cost is driven entirely by how many
# svr.predict() calls it needs, which multiply together as:
#     (# background samples) x (# coalition samples per explanation) x (# explained samples)
# The previous implementation used `shapley_num` for BOTH the background size
# and the explained-sample count, and left the coalition count (`nsamples`) at
# SHAP's default, which scales as roughly 2*n_features+2048. With ~8,000
# markers that is ~18,000 coalitions, each evaluated against every background
# sample, for every explained sample - tens of billions of predict() calls in
# total, which is what turns this into a multi-day run.
#
# Fixes applied below:
#  - a small, fixed-size background summary (shap.kmeans) - KernelExplainer
#    only needs this to marginalise out features, it does not need to be
#    anywhere near as large as the explained-sample count
#  - a fixed, modest `nsamples` budget instead of SHAP's feature-count-scaling
#    default
#  - restricting the explanation itself to the top-importance markers (by
#    absolute correlation with the trait - the standard cheap, model-agnostic
#    proxy, since SVR has no built-in importance measure like a forest does),
#    refitting a lightweight SVR on just those. This mirrors the same
#    strategy used for RF.py's interaction search.
# Together these give roughly a 3,000-4,000x reduction in predict() calls for
# an 8,000-marker, multi-thousand-sample dataset. All three are user-configurable
# via params[8:11] (GUI: "Max markers considered for Shapley scores", "Background
# sample size for Shapley scores", "Number of coalition samples for Shapley
# scores").

# Update ID ver4-5, R2 Stage 9 (blueprint §4.2 Layer 3): SVR has no native
# pairwise-interaction API of any kind (shap.KernelExplainer offers no
# shap_interaction_values equivalent - confirmed by introspection, blueprint
# Appendix B2), so pairwise interactions come from Friedman's H-statistic
# (models.interaction_extraction.h_statistic_interactions) - model-agnostic,
# needing only svr.predict itself. Restricted to a shortlist of markers
# (params[12]) for exactly the same O(M^2) reason RF.py's own interaction
# search is shortlisted, and evaluated against a small background sample
# (params[13]) rather than the whole training set. All four new fields are
# appended, never inserted (I5), and default to no-op (get_interaction=False)
# so a ver4-4 config's shorter params list keeps working unchanged.


def SV_Regression(train, valid, test, params, *, sample_weight_ctx=None):
    
    ker = params[0]
    eps = params[1]
    con = params[2]
    deg = params[3]
    gam = params[4]
    # Independent term in the kernel function, only used by the 'poly' and
    # 'sigmoid' kernels (ignored otherwise).
    coef0 = params[5]
    get_effect = params[6]
    shapley_num = params[7]
    max_shap_features = params[8]
    shap_background_size = params[9]
    shap_nsamples = params[10]
    # Update ID ver4-5, R2 Stage 9: appended, read defensively (I5) - a
    # ver4-4 (or Stages 1-8) config's shorter params list keeps working,
    # with interactions simply off (matching every other model's own
    # default-False convention for a NEWLY added interaction toggle).
    get_interaction = params[11] if len(params) > 11 else False
    max_interaction_features = params[12] if len(params) > 12 else 500
    interaction_background = params[13] if len(params) > 13 else 100
    interaction_top = params[14] if len(params) > 14 else 'all'
    # Update ID ver4-6, R1/R1b (blueprint §2.4/§2.10.4): appended, read
    # defensively (I5) - a config predating this update has interaction
    # pre-screening off and the default grid_resolution (3), reproducing
    # ver4-5 numbers exactly (AC1.7). 'screen_top' is a percentage
    # (0-100, or 'all') exactly like every other 'top_pct' field in this
    # schema - converted to h_statistic_interactions()'s own (0, 1]
    # fraction convention (or an exact count) immediately below, so this
    # file never has to reason about that distinction itself.
    interaction_screen = params[15] if len(params) > 15 else 'off'
    interaction_screen_top = params[16] if len(params) > 16 else 2.0
    interaction_grid_resolution = params[17] if len(params) > 17 else 3
    # ver4-25, Stage 8 (design blueprint R11): selectable between the
    # existing Friedman's H-statistic interaction route (default,
    # unchanged) and the optional shapiq package's own Shapley
    # interaction indices, explaining this model's own `.predict`
    # directly (no surrogate) - appended, read defensively (I5) so a
    # pre-existing, shorter HPARAMETERS['SVR'] list keeps reproducing
    # this file's original behaviour unchanged. This update adds a THIRD
    # choice, 'kernel_hessian' (see the kernel_effects import above) -
    # read the SAME field, no new params index needed.
    interaction_method = params[18] if len(params) > 18 else 'friedman_h'
    interaction_shapiq_orders = params[19] if len(params) > 19 else '2'
    interaction_shapiq_index = params[20] if len(params) > 20 else 'k-SII'
    interaction_shapiq_budget = params[21] if len(params) > 21 else 4096
    # ver4-25, Stage 7 (design blueprint R13): selectable between the
    # existing shap.KernelExplainer route (default, unchanged) and the
    # optional shapiq package's own explainer - appended, read
    # defensively (I5) so a pre-existing, shorter HPARAMETERS['SVR'] list
    # keeps reproducing this file's original behaviour unchanged.
    effect_method = params[22] if len(params) > 22 else 'kernel_shap'
    # ver4-29 (this update): which of shapiq's own approximators the
    # 'shapiq' interaction/effect routes below use - appended, read
    # defensively (I5) exactly like every field above. Default 'auto' is
    # a genuine no-op (shapiq itself already defaulted to 'auto' when
    # neither pipeline_utils.shapiq_pairwise_interactions() nor
    # shapiq_marker_effects() passed this kwarg at all), so a
    # pre-existing, shorter HPARAMETERS['SVR'] list keeps reproducing
    # this file's original behaviour unchanged. See pipeline_utils.
    # shapiq_pairwise_interactions()'s own `approximator` docstring for
    # why this field exists: at this file's own default budget (4096)
    # against a several-hundred-marker shortlist, 'auto' already
    # resolves to shapiq's SPEX approximator internally (built for
    # exactly this large-marker-count/budget-constrained regime), which
    # this update's own pipeline_utils.py changes now let run directly
    # instead of being pre-narrowed away - see that module's own
    # `_shapiq_would_use_spex()` comment for the full rationale. Exposed
    # as a field (rather than hardcoded) so a run can force a specific
    # approximator instead of relying on shapiq's own automatic choice.
    interaction_shapiq_approximator = params[23] if len(params) > 23 else 'auto'
    effect_shapiq_approximator = params[24] if len(params) > 24 else 'auto'
    # This update - see models/RF.py's identical field for the full
    # rationale. Appended, read defensively (I5).
    use_effect_for_interaction_shortlist = params[25] if len(params) > 25 else True
    # ver4-38, R3: appended, read defensively (I5/I11) - a list predating
    # this update has no index 26 and keeps the raw-scale epsilon / C
    # (False), so old configs reproduce their old numbers.
    scale_eps_c = bool(params[26]) if len(params) > 26 else False

    #Split the data sets into x and y here as specified in the original code
    train_x, train_y = train.iloc[:,:-1], train.iloc[:,-1]
    if valid.shape[0] != 0:
        valid_x, valid_y = valid.iloc[:,:-1], valid.iloc[:,-1]
    test_x, test_y = test.iloc[:,:-1], test.iloc[:,-1]
    # ver4-25, R8 - see models/RF.py's identical comment.
    _w_fit = sample_weight_ctx.w_fit if sample_weight_ctx is not None else None

    # ver4-38, R3: epsilon / C actually used by every SVR construction below
    # (unchanged raw values when scale_eps_c is False). Logged once per call
    # when the scaling is on, so the resolved value is never invisible (I11).
    eps_eff, con_eff, _y_sd, _scale_warn = _resolve_eps_c(scale_eps_c, train_y, eps, con)
    if _scale_warn:
        print(f"[SVR] WARN: {_scale_warn}.")
    elif scale_eps_c:
        print(f"[SVR] epsilon/C scaled by sd(train y)={_y_sd:.6g} : "
              f"eps_eff={eps_eff:.6g} (raw {eps}), C_eff={con_eff:.6g} (raw {con}).")

    # Phase 2, Requirement 6: an optional cuML GPU backend (else
    # sklearnex-accelerated CPU, else plain scikit-learn) is resolved from
    # the run's shared compute-resource settings - a CPU-only node without
    # sklearnex falls back to plain scikit-learn SVR, exactly as before
    # this option existed.
    _resources = get_active_compute_resources()
    _svr_regressor_cls, _is_gpu = _svr_cls(_resources['use_gpu_sklearn'])

    #Develop & evaluate a model here as specified in the original code
    svr = _svr_regressor_cls(kernel=ker, epsilon=eps_eff, C=con_eff, degree=deg,
                            gamma=gam, coef0=coef0)
    svr = fit_with_sample_weight(
        svr, train_x, train_y, _w_fit, log_prefix='[SVR]',
        cpu_fallback_factory=(lambda: SVR(kernel=ker, epsilon=eps_eff, C=con_eff, degree=deg,
                                           gamma=gam, coef0=coef0)) if _is_gpu else None,
    )
    
    predicted = np.asarray(svr.predict(test_x)).ravel()
    if valid.shape[0] != 0:
        predicted_valid = np.asarray(svr.predict(valid_x)).ravel()
    else:
        predicted_valid = []
    predicted_train = np.asarray(svr.predict(train_x)).ravel()

    ## Calculate the metrics
    actual_test = test_y.values.tolist()
    mse = mean_squared_error(actual_test, predicted)
    r = pearsonr(actual_test, predicted)[0]
    
    with timed_stage('xai_effect:SVR'):
        if get_effect == True:
            # Defensive clamp: avoid shap.sample() erroring if shapley_num exceeds
            # the number of available test rows.
            shapley_num = min(shapley_num, test_x.shape[0])

            n_features = train_x.shape[1]
            if max_shap_features != 'all' and n_features > max_shap_features:
                # Cheap, kernel-agnostic importance proxy: absolute correlation
                # with the trait (O(N*M), no extra model fitting needed to rank
                # candidates).
                correlations = train_x.corrwith(train_y).abs().fillna(0)
                top_features = correlations.sort_values(ascending=False).index[:max_shap_features]

                svr_effect = SVR(kernel=ker, epsilon=eps_eff, C=con_eff, degree=deg, gamma=gam,
                                        coef0=coef0)
                svr_effect.fit(train_x[top_features], train_y, sample_weight=_w_fit)
                effect_train_x = train_x[top_features]
                effect_test_x = test_x[top_features]
            else:
                # 'all' - use every marker, no shortlist. KernelExplainer needs a
                # plain CPU .predict() callable, so a GPU-fitted cuML model is
                # refit on CPU here for the explanation step.
                if _is_gpu:
                    svr_effect = SVR(kernel=ker, epsilon=eps_eff, C=con_eff, degree=deg, gamma=gam,
                                        coef0=coef0)
                    svr_effect.fit(train_x, train_y, sample_weight=_w_fit)
                else:
                    svr_effect = svr
                top_features = train_x.columns
                effect_train_x = train_x
                effect_test_x = test_x

            # Bugfix: shap.kmeans(effect_train_x, background_size) used to be
            # called directly here and could raise
            # ValueError('# of weights must match data matrix!') whenever
            # background_size met or exceeded the number of DISTINCT rows in
            # effect_train_x - routine for 0/1/2-coded marker data once
            # narrowed to a shortlist. See pipeline_utils.
            # safe_shap_kmeans_background's docstring for the full root cause;
            # this wrapper clamps to the distinct-row count instead of raising.
            def _kernel_shap_effect():
                background_size = min(shap_background_size, effect_train_x.shape[0])
                background = safe_shap_kmeans_background(effect_train_x, background_size, log_prefix="[SVR]")

                # Requirements.md item 1 (this update): "the current KernelSHAP
                # is very slow ... when the total number of markers is >
                # 10,000" - above that threshold, replace the per-row
                # shap.KernelExplainer fan-out below with
                # pipeline_utils.fast_kernel_shap_effects()'s shared-coalition,
                # fully-batched estimator (one coalition sample + one WLS
                # factorisation for the WHOLE task, reused across every
                # explained row, instead of once per row - see that function's
                # own module-level comment for the full algorithmic rationale).
                # NEVER lets a failure here fail the task - falls back to the
                # proven per-row route below with a printed NOTE, exactly like
                # this file's own 'shapiq' fallback just above.
                if effect_train_x.shape[1] > FAST_KERNEL_SHAP_MARKER_THRESHOLD:
                    _explain_rows = shap.sample(effect_test_x, shapley_num)
                    # Perf fix ("the same problem exists for SVR ... when using
                    # KernelSHAP ... minimise the computational cost when the
                    # total number of markers is > 20,000"): for kernel='rbf'
                    # specifically, `svr_effect`'s own dual decision function is
                    # an exact Gaussian/RBF-kernel function of x (the SAME fact
                    # 'kernel_hessian' already exploits for interactions, just
                    # below), so `fast_kernel_shap_effects()`'s own generic,
                    # black-box occlusion loop (`O(n_sv * n_features^2)`) can be
                    # replaced with `kernel_effects.rbf_kernel_occlusion_effects()`'s
                    # closed-form, bit-identical `O(n_sv * n_features)`
                    # computation - see that function's own module-level comment
                    # for the full derivation.
                    #
                    # This update (previously 'rbf' only - "minimise the
                    # computational time of marker effects and interactions for
                    # other kernel cases: linear, poly, or sigmoid"): every OTHER
                    # SVR kernel this file supports also has an exact closed form
                    # (`kernel_effects.linear_kernel_occlusion_effects()`/
                    # `poly_kernel_occlusion_effects()`/
                    # `sigmoid_kernel_occlusion_effects()` - see that module's own
                    # header comment for the shared derivation), so the SAME
                    # "try the closed form first, fall back to
                    # `fast_kernel_shap_effects()` on any failure" strategy now
                    # covers all four kernels, not only 'rbf'. Never a new
                    # failure mode for a task that used to work: any failure or
                    # an unrecognised kernel still falls through to
                    # `fast_kernel_shap_effects()`, which itself still falls
                    # through to `shap.KernelExplainer` below.
                    try:
                        _occlusion_baseline = np.average(
                            background.data, axis=0, weights=background.weights,
                        )
                    except Exception:
                        _occlusion_baseline = None
                    if _occlusion_baseline is not None:
                        if ker == 'rbf':
                            try:
                                _svr_gamma = _resolve_svr_gamma(svr_effect, gam, effect_train_x)
                                return kernel_effects.rbf_kernel_occlusion_effects(
                                    svr_effect.dual_coef_.ravel(), svr_effect.support_vectors_,
                                    _explain_rows, _occlusion_baseline, _svr_gamma,
                                )
                            except Exception as exc:
                                print(f"[SVR] NOTE: closed-form rbf_kernel_occlusion_effects failed "
                                      f"({exc!r}) - falling back to fast_kernel_shap_effects for this "
                                      f"task instead.")
                        elif ker == 'linear':
                            try:
                                _w = np.asarray(svr_effect.coef_, dtype=float).ravel()
                                return kernel_effects.linear_kernel_occlusion_effects(
                                    _w, _explain_rows, _occlusion_baseline,
                                )
                            except Exception as exc:
                                print(f"[SVR] NOTE: closed-form linear_kernel_occlusion_effects failed "
                                      f"({exc!r}) - falling back to fast_kernel_shap_effects for this "
                                      f"task instead.")
                        elif ker in ('poly', 'sigmoid'):
                            try:
                                _svr_gamma = _resolve_svr_gamma(svr_effect, gam, effect_train_x)
                                _coef0 = float(getattr(svr_effect, 'coef0', coef0))
                                if ker == 'poly':
                                    _degree = int(getattr(svr_effect, 'degree', deg))
                                    _effect_scores = kernel_effects.poly_kernel_occlusion_effects(
                                        svr_effect.dual_coef_.ravel(), svr_effect.support_vectors_,
                                        _explain_rows, _occlusion_baseline, _svr_gamma, _coef0, _degree,
                                    )
                                else:
                                    _effect_scores = kernel_effects.sigmoid_kernel_occlusion_effects(
                                        svr_effect.dual_coef_.ravel(), svr_effect.support_vectors_,
                                        _explain_rows, _occlusion_baseline, _svr_gamma, _coef0,
                                    )
                                return _effect_scores
                            except Exception as exc:
                                print(f"[SVR] NOTE: closed-form {ker}_kernel_occlusion_effects failed "
                                      f"({exc!r}) - falling back to fast_kernel_shap_effects for this "
                                      f"task instead.")
                    try:
                        return fast_kernel_shap_effects(
                            svr_effect.predict, background.data, _explain_rows,
                            nsamples=shap_nsamples, background_weights=background.weights,
                            log_prefix="[SVR]",
                        )
                    except Exception as exc:
                        print(f"[SVR] NOTE: fast_kernel_shap_effects failed ({exc!r}) - falling back "
                              f"to the per-row shap.KernelExplainer route for this task instead.")

                explainer = shap.KernelExplainer(svr_effect.predict, background)
                # Bug fix ("Max markers considered for Shapley scores" looking
                # ignored - reported: 'all' correctly explains every shortlisted
                # marker per this task's own log line, yet Marker_effect.csv
                # comes back with only ~10 nonzero markers per explained row):
                # shap.KernelExplainer.shap_values()'s own `l1_reg` argument
                # defaults (shap>=0.47.0) to 'num_features(10)' - a SEPARATE,
                # shap-internal lasso feature-selection step that silently
                # zeroes every marker beyond its own top ~10 per row, completely
                # independent of how many markers `effect_train_x`/`top_features`
                # above actually shortlisted.
                #
                # Bug fix (this update - the first fix above was still
                # incomplete): explicitly requesting `l1_reg=f'num_features(M)'`
                # (M = every shortlisted marker) does NOT actually restore all M
                # markers whenever M is large relative to `shap_nsamples` -
                # confirmed directly against shap's own source
                # (`shap.explainers._kernel.Kernel.solve()`): `'num_features(M)'`
                # still runs `sklearn.linear_model.lars_path(..., max_iter=M)`,
                # and LARS can never select more markers than it has (roughly)
                # independent coalition SAMPLES to resolve them from - so once
                # `shap_nsamples` (the coalition budget, NOT 'max_shap_features')
                # is smaller than M, LARS silently caps the number of nonzero
                # markers at whatever it manages to resolve from that budget
                # (reproduced directly: 2,000 markers at the default
                # `shap_nsamples=200` returned only ~890 nonzero markers, not
                # 2,000). Passing `l1_reg=False` instead skips that LARS
                # feature-selection step entirely (see `solve()`'s own
                # `nonzero_inds = np.arange(self.M)` default) and solves the
                # weighted-least-squares system directly across every one of
                # THIS task's own shortlisted markers - reproduced directly to
                # give all M markers a real value (and run FASTER, since the
                # LARS step itself is skipped) at the same M=2,000/
                # `shap_nsamples`=200 case above.
                return abs(parallel_shap_values(
                    explainer, shap.sample(effect_test_x, shapley_num), shap_nsamples, _resources['n_jobs'],
                    l1_reg=False,
                )).sum(axis=0)

            # ver4-43 (R2): opt-in 'baseline_shap' marker effect. phi_q(x) =
            # mean over REAL training background rows z of [f(x) - f(x^{q->z})],
            # reported as sum_x |phi_q(x)| - the single-marker term of this
            # file's own 'baseline_shap' interaction route. Computed for EVERY
            # shortlisted marker at any marker count (no 10,000-marker threshold,
            # no coalition sampling) through the exact deduplicated engine and
            # this fit's closed-form kernel evaluator. Not numerically
            # comparable with 'kernel_shap' values (different baseline). Any
            # failure falls back to the proven 'kernel_shap' route, never
            # fails the task.
            def _baseline_shap_effect():
                _bg_n = min(shap_background_size, effect_train_x.shape[0])
                _X_bg = np.asarray(shap.sample(effect_train_x, _bg_n), dtype=float)
                _X_ex = np.asarray(shap.sample(effect_test_x, shapley_num), dtype=float)
                if ker == 'linear':
                    _scores = kernel_effects.linear_baseline_shap_effects(
                        np.asarray(svr_effect.coef_, dtype=float).ravel(), _X_ex, _X_bg,
                    )
                    _how = "closed-form linear"
                else:
                    _svr_gamma = _resolve_svr_gamma(svr_effect, gam, effect_train_x)
                    _fast = kernel_effects.SVRBaselineShapFastEval(
                        ker, svr_effect.dual_coef_.ravel(), svr_effect.support_vectors_, _X_bg,
                        intercept=float(np.asarray(svr_effect.intercept_).reshape(-1)[0]),
                        gamma=_svr_gamma, coef0=float(getattr(svr_effect, 'coef0', coef0)),
                        degree=int(getattr(svr_effect, 'degree', deg)),
                    )
                    _scores = baseline_shap_effects(
                        svr_effect.predict, _X_ex, _X_bg, kernel_fast_eval=_fast,
                        n_jobs=_resources['n_jobs'], log_prefix="[SVR]",
                    )
                    _how = f"closed-form {ker!r}-kernel evaluator"
                print(f"[SVR] NOTE: 'baseline_shap' marker effects computed for "
                      f"{len(_scores)} marker(s) from {_X_ex.shape[0]} explained row(s) and "
                      f"{_X_bg.shape[0]} real background row(s) via the {_how} (not numerically "
                      f"comparable with 'kernel_shap' effects - different baseline).")
                return _scores

            if effect_method == 'baseline_shap':
                try:
                    effect_scores = _baseline_shap_effect()
                except Exception as exc:
                    print(f"[SVR] NOTE: baseline_shap marker-effect computation failed ({exc!r}) - "
                          f"falling back to the 'kernel_shap' route for this task instead.")
                    effect_scores = _kernel_shap_effect()
            elif effect_method == 'shapiq':
                # ver4-25, Stage 7 (R13) - see pipeline_utils.shapiq_marker_
                # effects()'s own module-level header comment for the
                # disclosed "written but not runtime-verified against a real
                # shapiq install" caveat. NEVER lets a shapiq failure (missing
                # package, an API mismatch) fail this task - falls back to
                # the proven 'kernel_shap' route with a printed NOTE.
                try:
                    effect_scores = shapiq_marker_effects(
                        svr_effect.predict, effect_train_x, shap.sample(effect_test_x, shapley_num),
                        background_size=shap_background_size, budget=shap_nsamples,
                        approximator=effect_shapiq_approximator,
                    )
                except Exception as exc:
                    print(f"[SVR] NOTE: shapiq marker-effect computation failed ({exc!r}) - falling "
                          f"back to the 'kernel_shap' route (shap.KernelExplainer) for this task "
                          f"instead. ('shapiq' is an optional dependency - 'pip install shapiq' if "
                          f"not yet installed; this may also mean this shapiq version's API differs "
                          f"from what this codebase expects.)")
                    effect_scores = _kernel_shap_effect()
            else:
                effect_scores = _kernel_shap_effect()

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

    # This update: how many markers this task's own marker-effect step
    # above actually covers ('Max markers considered for Shapley scores',
    # field 8 - both 'kernel_shap' and 'shapiq' share that same shortlist
    # here, unlike RF/ExtraTrees/XGBoost's split between a capped and an
    # uncapped effect route) - None when effects were not computed at all
    # this task. Feeds should_use_effect_for_interaction_shortlist() (see
    # its own docstring, models/interaction_extraction.py) for the shared
    # eligibility rule main_app.py's dynamic checkbox visibility mirrors.
    _effect_marker_cap = max_shap_features if get_effect else None
    _use_effect_shortlist = should_use_effect_for_interaction_shortlist(
        use_effect_for_interaction_shortlist, max_interaction_features, _effect_marker_cap,
    )
    # Reuse this task's own already-computed, full-width marker-effect
    # values (aligned to train_x's own column order) as the interaction
    # shortlist's ranking when eligible; otherwise the existing |correlation|
    # proxy, exactly as before this checkbox existed (I11).
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

    with timed_stage('xai_inter:SVR'):
        if get_interaction == True:
            n_features = train_x.shape[1]
            if max_interaction_features != 'all' and n_features > max_interaction_features:
                # Ranked by this task's own already-computed marker-effect
                # values instead of |correlation| when eligible (see
                # '_use_effect_shortlist' above) - otherwise a SEPARATE
                # fit/shortlist from the effect step above, since the
                # interaction shortlist size (max_interaction_features) need
                # not match the effect shortlist size (max_shap_features) -
                # byte-identical to before this checkbox existed (I11).
                interaction_features = _interaction_importance_fn().sort_values(ascending=False).index[:max_interaction_features]
            else:
                interaction_features = train_x.columns

            svr_interaction = SVR(kernel=ker, epsilon=eps_eff, C=con_eff, degree=deg, gamma=gam,
                                 coef0=coef0)
            svr_interaction.fit(train_x[interaction_features], train_y, sample_weight=_w_fit)

            # Bugfix: shapiq's own TabularExplainer bills a fixed
            # `interaction_shapiq_budget` cost PER EXPLAINED ROW (unlike
            # Friedman's H-statistic just below, which is a cheap, grid-based
            # per-pair computation whose cost does not scale with row count -
            # see _h_statistic_interaction_sample()'s own docstring), so the
            # 'shapiq' branch's own explained-row count needs to be BOUNDED,
            # exactly like every OTHER per-row Shapley-based route in this
            # codebase already is (models/RF.py's own `shap.sample(
            # interaction_test_x, shapley_num)` for its 'shapiq_tree' route;
            # `models.interaction_extraction.surrogate_shapiq_interactions()`'s
            # own `min(30, ...)` for RKHS's 'surrogate_shapiq' route). Reuses
            # this file's own `shapley_num` field (already used for the SAME
            # "how many rows to explain" purpose by this file's own effect
            # step above) rather than adding a new hyperparameter. Previously
            # this branch explained the FULL, un-sampled test split - cost
            # scaled directly and unboundedly with the number of test
            # individuals, unlike every other Shapley-based route here.
            _interaction_shapley_num = min(shapley_num, test_x.shape[0])

            def _h_statistic_interaction_sample():
                interaction_pairs = [
                    (a, b) for a in range(len(interaction_features)) for b in range(a + 1, len(interaction_features))
                ]
                # Bugfix: n_jobs was previously never forwarded here, so this step
                # ran single-threaded regardless of the run's compute-resource
                # settings, unlike this file's own SHAP marker-effect step (which
                # already uses _resources['n_jobs']) - see models.interaction_
                # extraction.h_statistic_interactions()'s own n_jobs docstring for
                # what this now does. Mirrors how models/RF.py forwards n_jobs into
                # tree_shap_interactions().
                return h_statistic_interactions(
                    svr_interaction.predict, test_x[interaction_features], list(interaction_features),
                    pairs=interaction_pairs, n_background=min(interaction_background, test_x.shape[0]),
                    n_jobs=_resources['n_jobs'],
                    # Update ID ver4-6, R1/R1b: `screen=None` (the 'off' default)
                    # and `grid_resolution=3` reproduce ver4-5's own per-pair,
                    # unscreened, [:3]-observed-grid numbers exactly (AC1.7) -
                    # h_statistic_interactions() itself resolves grid='auto' by
                    # default (R1b's own correctness fix), independent of this
                    # file's own settings.
                    screen=(interaction_screen if interaction_screen != 'off' else None),
                    screen_keep=(float(interaction_screen_top) / 100.0 if interaction_screen_top != 'all' else 1.0),
                    grid_resolution=interaction_grid_resolution,
                )

            # ver4-25, Stage 8 (design blueprint R11): 'shapiq' explains this
            # SVR fit's own `.predict` directly (no surrogate) with shapiq's
            # Shapley interaction indices, and can search orders above 2 -
            # only order-2 rows are kept for now (Interaction.csv/circos
            # storage - ver4-25, Stage 9 (design blueprint R12) - k=2 rows go
            # to Interaction.csv, k>=3 rows to Interaction_higher_order.csv,
            # split by genomic_prediction.py's own Step 10; this file just
            # returns every requested order's own rows, canonicalised. Falls
            # back to 'friedman_h' with a printed NOTE on any failure.
            if interaction_method == 'kernel_hessian':
                # NEW route (user feedback: "shapiq is unreliable and should
                # not rely on this") - the SVR dual decision function
                # `f(x) = intercept + sum_i dual_coef_i * K(x, sv_i)` is an
                # exact, twice-differentiable function of x for EVERY kernel
                # this file supports, not only 'rbf' - see kernel_effects.
                # rbf_kernel_hessian_interactions()'s own docstring for the
                # 'rbf' derivation, and this update's own module-level
                # comment in kernel_effects.py (directly above
                # linear_kernel_occlusion_effects()) for 'linear'/'poly'/
                # 'sigmoid''s own closed forms. No sampling, no coalitions, no
                # `shapiq` dependency at all, for any of the four kernels.
                try:
                    _explain_rows = np.asarray(
                        shap.sample(test_x[interaction_features], _interaction_shapley_num), dtype=float,
                    )
                    _sv = svr_interaction.dual_coef_.ravel()
                    _X_ref = svr_interaction.support_vectors_
                    if ker == 'rbf':
                        _svr_gamma = _resolve_svr_gamma(svr_interaction, gam, train_x[interaction_features])
                        interaction_sample = kernel_effects.rbf_kernel_hessian_interactions(
                            _sv, _X_ref, _explain_rows, _svr_gamma, list(interaction_features),
                        )
                    elif ker == 'linear':
                        # An exactly linear model has zero curvature - answered
                        # with no per-row computation at all (see kernel_effects.
                        # poly_kernel_hessian_interactions()'s own degree<2
                        # shortcut).
                        interaction_sample = kernel_effects.poly_kernel_hessian_interactions(
                            _sv, _X_ref, _explain_rows, 1.0, 0.0, 1, list(interaction_features),
                        )
                    elif ker == 'poly':
                        _svr_gamma = _resolve_svr_gamma(svr_interaction, gam, train_x[interaction_features])
                        _coef0 = float(getattr(svr_interaction, 'coef0', coef0))
                        _degree = int(getattr(svr_interaction, 'degree', deg))
                        interaction_sample = kernel_effects.poly_kernel_hessian_interactions(
                            _sv, _X_ref, _explain_rows, _svr_gamma, _coef0, _degree, list(interaction_features),
                        )
                    elif ker == 'sigmoid':
                        _svr_gamma = _resolve_svr_gamma(svr_interaction, gam, train_x[interaction_features])
                        _coef0 = float(getattr(svr_interaction, 'coef0', coef0))
                        interaction_sample = kernel_effects.sigmoid_kernel_hessian_interactions(
                            _sv, _X_ref, _explain_rows, _svr_gamma, _coef0, list(interaction_features),
                        )
                    else:
                        raise ValueError(f"unrecognised SVR kernel {ker!r}")
                    print(f"[SVR] NOTE: analytic kernel-Hessian interaction extraction produced "
                          f"{interaction_sample.shape[0]} pair(s) from a "
                          f"{len(interaction_features)}-marker shortlist - a closed-form, "
                          f"deterministic computation on this fit's OWN {ker!r} kernel; no sampling, "
                          f"no 'shapiq' dependency.")
                except Exception as exc:
                    print(f"[SVR] NOTE: kernel_hessian interaction computation failed ({exc!r}) - "
                          f"falling back to 'friedman_h' (Friedman's H-statistic) for this task "
                          f"instead.")
                    interaction_sample = _h_statistic_interaction_sample()
            elif interaction_method == 'baseline_shap':
                # NEW route (user feedback, backed by this project's own
                # experimental results - see the baseline_shap_interactions
                # import comment above): works for ANY kernel - unlike
                # 'kernel_hessian' above, it needs no per-kernel analytic
                # Hessian formula, since it evaluates the model's own real
                # predict() at hybrid rows rather than differentiating it.
                # Background rows are REAL training individuals (never a
                # synthetic grid), so linkage disequilibrium between markers
                # is preserved exactly, and cost is fixed (no coalition-
                # sampling budget) - see that function's own Cost section.
                #
                # This update ("the computational minimisation should be
                # integrated into ... baseline_shap for marker interactions"):
                # every hybrid row this route evaluates still differs from the
                # explained row `x` in only one or two columns, so the SAME
                # closed-form idea 'kernel_hessian' uses (this file's own
                # 'linear'/'poly'/'sigmoid'/'rbf' branches above) applies here
                # too - see `kernel_effects.SVRBaselineShapFastEval`'s own
                # module-level comment for the full derivation. `kernel='linear'`
                # is a further special case: the four-term discrete difference
                # this route computes is EXACTLY zero for every pair for an
                # exactly linear model (same reason 'kernel_hessian' already
                # reports zero for it), so it is answered directly, with no
                # per-row evaluation of any kind. Any failure to construct or
                # use the closed-form accelerator falls back to the ORIGINAL,
                # proven `predict()`-based route below (still 'baseline_shap',
                # just unaccelerated) before this whole branch's own outer
                # `except` would fall all the way back to 'friedman_h' -
                # never a new failure mode for a task that used to work.
                try:
                    interaction_pairs = [
                        (a, b)
                        for a in range(len(interaction_features)) for b in range(a + 1, len(interaction_features))
                    ]
                    if ker == 'linear':
                        interaction_sample = kernel_effects.zero_pair_interaction_table(list(interaction_features))
                        print(f"[SVR] NOTE: baseline_shap interaction extraction for kernel='linear' is "
                              f"EXACTLY zero for every one of this task's {interaction_sample.shape[0]} "
                              f"pair(s) (an exactly linear model has no pairwise feature interactions by "
                              f"construction) - reported with no per-row evaluation at all.")
                    else:
                        _bg_size = min(interaction_background, train_x.shape[0])
                        _X_background = np.asarray(shap.sample(train_x[interaction_features], _bg_size), dtype=float)
                        _X_explain = np.asarray(
                            shap.sample(test_x[interaction_features], _interaction_shapley_num), dtype=float,
                        )
                        _fast_eval = None
                        try:
                            _svr_gamma = _resolve_svr_gamma(svr_interaction, gam, train_x[interaction_features])
                            _coef0 = float(getattr(svr_interaction, 'coef0', coef0))
                            _degree = int(getattr(svr_interaction, 'degree', deg))
                            _intercept = float(np.asarray(svr_interaction.intercept_).reshape(-1)[0])
                            _fast_eval = kernel_effects.SVRBaselineShapFastEval(
                                ker, svr_interaction.dual_coef_.ravel(), svr_interaction.support_vectors_,
                                _X_background, intercept=_intercept, gamma=_svr_gamma, coef0=_coef0, degree=_degree,
                            )
                        except Exception as exc:
                            print(f"[SVR] NOTE: closed-form SVRBaselineShapFastEval construction failed "
                                  f"({exc!r}) - baseline_shap will evaluate this fit's own predict() "
                                  f"directly for this task instead (slower, still exact).")
                        interaction_sample = baseline_shap_interactions(
                            svr_interaction.predict, _X_explain, _X_background, list(interaction_features),
                            pairs=interaction_pairs, n_jobs=_resources['n_jobs'],
                            # Update (perf): reuses the SAME pre-existing 'Marker
                            # interaction screen' fields the 'friedman_h' branch
                            # above already reads (interaction_screen/
                            # interaction_screen_top) - no new hyperparameter.
                            # 'off' (the field's own default) resolves to
                            # screen=None, which is byte-identical to this
                            # parameter not existing (I11) - so this is a no-op
                            # for every existing config, and a real lever (cuts
                            # the O(pairs) cost directly) for anyone who turns
                            # screening on because the unscreened cost was
                            # prohibitive on a large marker shortlist.
                            screen=(interaction_screen if interaction_screen != 'off' else None),
                            screen_keep=(float(interaction_screen_top) / 100.0 if interaction_screen_top != 'all' else 1.0),
                            kernel_fast_eval=_fast_eval,
                        )
                        print(f"[SVR] NOTE: baseline-referenced pairwise interaction extraction produced "
                              f"{interaction_sample.shape[0]} pair(s) from a {len(interaction_features)}-"
                              f"marker shortlist - directly comparable (same units/reduction) to RF's own "
                              f"TreeSHAP interaction table; no coalition sampling, no synthetic joint-"
                              f"genotype grid"
                              + (f"; hybrid rows evaluated via this fit's OWN closed-form {ker!r}-kernel "
                                 f"correction, not a re-run of predict()." if _fast_eval is not None else "."))
                except Exception as exc:
                    print(f"[SVR] NOTE: baseline_shap interaction computation failed ({exc!r}) - "
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
                    # (mirrors models/RF.py's own depth/tree-count shrink at
                    # order>=3, design blueprint R2, applied here to the ROW
                    # axis) - gated by the SAME `shapiq_adaptive_depth` flag
                    # (no new config key). `max(_orders) <= 2` (I11) leaves
                    # `_interaction_shapley_num` completely unchanged.
                    _shapiq_explain_n = (
                        resolve_shapiq_explain_row_count(max(_orders), _interaction_shapley_num)
                        if _resources.get('shapiq_adaptive_depth', True) else _interaction_shapley_num
                    )
                    raw = shapiq_pairwise_interactions(
                        svr_interaction.predict, train_x[interaction_features],
                        shap.sample(test_x[interaction_features], _shapiq_explain_n),
                        list(interaction_features),
                        orders=_orders, background_size=min(interaction_background, train_x.shape[0]),
                        budget=interaction_shapiq_budget, index=interaction_shapiq_index,
                        n_jobs=_shapiq_n_jobs, approximator=interaction_shapiq_approximator,
                    )
                    interaction_sample = canonicalize_shapiq_table(raw)
                except Exception as exc:
                    print(f"[SVR] NOTE: shapiq interaction computation failed ({exc!r}) - falling "
                          f"back to 'friedman_h' (Friedman's H-statistic) for this task instead.")
                    interaction_sample = _h_statistic_interaction_sample()
            else:
                interaction_sample = _h_statistic_interaction_sample()
            interaction_sample = top_select_by_order(interaction_sample, 'percentage', interaction_top)

            # Requirements3.md item 4: 'shapiq' (shap.KernelExplainer has no
            # interaction API for SVR, so this is the SAME model-agnostic
            # shapiq route models/RF.py's own 'shapiq_tree' uses for a tree
            # model) was reporting a much smaller "total number of extracted
            # interactions" in Interaction.csv than RF's own SHAP-based
            # ('pairwise_shap'/'shapiq_tree', both already at parity - see
            # models/RF.py's own identical padding call and models.
            # interaction_extraction.pad_shapiq_pairs_to_shap_parity()'s own
            # docstring) route at the SAME top-N% setting - not a routing
            # bug, but the same "shapiq genuinely finds fewer nonzero pairs
            # than a dense SHAP matrix at this task's scale" effect already
            # documented for RF's 'shapiq_tree'. Padding here with the SAME
            # helper, against the SAME candidate universe this branch itself
            # searched (interaction_features) at the SAME threshold
            # (interaction_top), makes SVR's 'shapiq' row count directly
            # comparable to any SHAP-based model's, exactly like RF's own
            # 'shapiq_tree' already is - never applied to 'friedman_h' (that
            # route's own top_select_by_order() output is left exactly as
            # before, I11).
            if interaction_method == 'shapiq':
                interaction_sample = pad_shapiq_pairs_to_shap_parity(
                    interaction_sample, list(interaction_features), interaction_top,
                )
        else:
            interaction_sample = pd.DataFrame()

    return r, mse, effect, interaction_sample, predicted, predicted_valid, predicted_train
