from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import shap

from pipeline_utils import get_active_compute_resources, parallel_shap_values
from models.interaction_extraction import h_statistic_interactions, top_select

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
    if use_gpu:
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


def KNN(train, valid, test, params):
    
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
    
    #Split the data sets into x and y here as specified in the original code
    train_x, train_y = train.iloc[:,:-1], train.iloc[:,-1]
    if valid.shape[0] != 0:
        valid_x, valid_y = valid.iloc[:,:-1], valid.iloc[:,-1]
    test_x, test_y = test.iloc[:,:-1], test.iloc[:,-1]
    
    # Phase 2, Requirement 6: n_jobs (neighbor search) and an optional
    # cuML GPU backend are resolved from the run's shared compute-resource
    # settings - a CPU-only node falls back to scikit-learn's own
    # multi-threaded neighbor search, exactly as before this option existed.
    _resources = get_active_compute_resources()
    _knn_cls, _is_gpu = _knn_regressor_cls(_resources['use_gpu_sklearn'])

    #Develop & evaluate a model here as specified in the original code
    if _is_gpu:
        knn = _knn_cls(n_neighbors=n_neighbours, weights=weights)
    else:
        knn = _knn_cls(n_neighbors=n_neighbours, weights=weights, p=p, n_jobs=_resources['n_jobs'])
    knn.fit(train_x, train_y)
    
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

            knn_effect = KNeighborsRegressor(n_neighbors=n_neighbours, weights=weights, p=p,
                                              n_jobs=_resources['n_jobs'])
            knn_effect.fit(train_x[top_features], train_y)
            effect_train_x = train_x[top_features]
            effect_test_x = test_x[top_features]
        else:
            # 'all' - use every marker, no shortlist. KernelExplainer needs a
            # plain CPU .predict() callable, so a GPU-fitted cuML model is
            # refit on CPU here for the explanation step.
            if _is_gpu:
                knn_effect = KNeighborsRegressor(n_neighbors=n_neighbours, weights=weights, p=p,
                                                  n_jobs=_resources['n_jobs'])
                knn_effect.fit(train_x, train_y)
            else:
                knn_effect = knn
            top_features = train_x.columns
            effect_train_x = train_x
            effect_test_x = test_x

        background_size = min(shap_background_size, effect_train_x.shape[0])
        background = shap.kmeans(effect_train_x, background_size)
        explainer = shap.KernelExplainer(knn_effect.predict, background)
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
            correlations = train_x.corrwith(train_y).abs().fillna(0)
            interaction_features = correlations.sort_values(ascending=False).index[:max_interaction_features]
        else:
            interaction_features = train_x.columns

        knn_interaction = KNeighborsRegressor(n_neighbors=n_neighbours, weights=weights, p=p,
                                               n_jobs=_resources['n_jobs'])
        knn_interaction.fit(train_x[interaction_features], train_y)

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
        interaction_sample = h_statistic_interactions(
            knn_interaction.predict, test_x[interaction_features], list(interaction_features),
            pairs=interaction_pairs, n_background=min(interaction_background, test_x.shape[0]),
            n_jobs=_resources['n_jobs'],
            # Update ID ver4-6, R1/R1b - see models/SVR.py's own identical note.
            screen=(interaction_screen if interaction_screen != 'off' else None),
            screen_keep=(float(interaction_screen_top) / 100.0 if interaction_screen_top != 'all' else 1.0),
            grid_resolution=interaction_grid_resolution,
        )
        interaction_sample = top_select(interaction_sample, 'percentage', interaction_top)
    else:
        interaction_sample = pd.DataFrame()

    return r, mse, effect, interaction_sample, predicted, predicted_valid, predicted_train
