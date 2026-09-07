from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import shap

from pipeline_utils import get_active_compute_resources, parallel_shap_values
from models.interaction_extraction import h_statistic_interactions, top_select

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


def _svr_cls(use_gpu):
    """Return (SVR class, is_gpu) - cuML's GPU-backed SVR when `use_gpu`
    is True AND cuML is importable, else scikit-learn's own CPU SVR
    (optionally sklearnex-accelerated - see _maybe_patch_sklearnex()).
    Never raises - falls back to plain CPU scikit-learn on any import/
    compatibility failure (Phase 2, Requirement 6)."""
    if use_gpu:
        try:
            from cuml.svm import SVR as CumlSVR
            return CumlSVR, True
        except Exception as exc:
            print(f"[SVR] USE_GPU_SKLEARN requested but cuML is unavailable "
                  f"({exc}) - falling back to (sklearnex-accelerated, if available) CPU SVR.")
    _maybe_patch_sklearnex()
    return SVR, False

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


def SV_Regression(train, valid, test, params):
    
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
    
    #Split the data sets into x and y here as specified in the original code
    train_x, train_y = train.iloc[:,:-1], train.iloc[:,-1]
    if valid.shape[0] != 0:
        valid_x, valid_y = valid.iloc[:,:-1], valid.iloc[:,-1]
    test_x, test_y = test.iloc[:,:-1], test.iloc[:,-1]
    
    # Phase 2, Requirement 6: an optional cuML GPU backend (else
    # sklearnex-accelerated CPU, else plain scikit-learn) is resolved from
    # the run's shared compute-resource settings - a CPU-only node without
    # sklearnex falls back to plain scikit-learn SVR, exactly as before
    # this option existed.
    _resources = get_active_compute_resources()
    _svr_regressor_cls, _is_gpu = _svr_cls(_resources['use_gpu_sklearn'])

    #Develop & evaluate a model here as specified in the original code
    svr = _svr_regressor_cls(kernel=ker, epsilon=eps, C=con, degree=deg, gamma=gam, coef0=coef0)
    svr.fit(train_x, train_y)
    
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

            svr_effect = SVR(kernel=ker, epsilon=eps, C=con, degree=deg, gamma=gam, coef0=coef0)
            svr_effect.fit(train_x[top_features], train_y)
            effect_train_x = train_x[top_features]
            effect_test_x = test_x[top_features]
        else:
            # 'all' - use every marker, no shortlist. KernelExplainer needs a
            # plain CPU .predict() callable, so a GPU-fitted cuML model is
            # refit on CPU here for the explanation step.
            if _is_gpu:
                svr_effect = SVR(kernel=ker, epsilon=eps, C=con, degree=deg, gamma=gam, coef0=coef0)
                svr_effect.fit(train_x, train_y)
            else:
                svr_effect = svr
            top_features = train_x.columns
            effect_train_x = train_x
            effect_test_x = test_x

        background_size = min(shap_background_size, effect_train_x.shape[0])
        background = shap.kmeans(effect_train_x, background_size)
        explainer = shap.KernelExplainer(svr_effect.predict, background)
        effect_scores = abs(parallel_shap_values(
            explainer, shap.sample(effect_test_x, shapley_num), shap_nsamples, _resources['n_jobs']
        )).sum(axis=0)

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

    if get_interaction == True:
        n_features = train_x.shape[1]
        if max_interaction_features != 'all' and n_features > max_interaction_features:
            # Same cheap, kernel-agnostic |correlation| shortlist as the
            # effect step above - a SEPARATE fit/shortlist, since the
            # interaction shortlist size (max_interaction_features) need
            # not match the effect shortlist size (max_shap_features).
            correlations = train_x.corrwith(train_y).abs().fillna(0)
            interaction_features = correlations.sort_values(ascending=False).index[:max_interaction_features]
        else:
            interaction_features = train_x.columns

        svr_interaction = SVR(kernel=ker, epsilon=eps, C=con, degree=deg, gamma=gam, coef0=coef0)
        svr_interaction.fit(train_x[interaction_features], train_y)

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
        interaction_sample = h_statistic_interactions(
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
        interaction_sample = top_select(interaction_sample, 'percentage', interaction_top)
    else:
        interaction_sample = pd.DataFrame()

    return r, mse, effect, interaction_sample, predicted, predicted_valid, predicted_train
