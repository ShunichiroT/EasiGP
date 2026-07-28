from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import shap

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
    
    #Split the data sets into x and y here as specified in the original code
    train_x, train_y = train.iloc[:,:-1], train.iloc[:,-1]
    if valid.shape[0] != 0:
        valid_x, valid_y = valid.iloc[:,:-1], valid.iloc[:,-1]
    test_x, test_y = test.iloc[:,:-1], test.iloc[:,-1]
    
    #Develop & evaluate a model here as specified in the original code
    knn = KNeighborsRegressor(n_neighbours, weights=weights, p=p)
    knn.fit(train_x, train_y)
    
    predicted = knn.predict(test_x)
    if valid.shape[0] != 0:
        predicted_valid = knn.predict(valid_x)
    else:
        predicted_valid = []
    predicted_train = knn.predict(train_x)

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

            knn_effect = KNeighborsRegressor(n_neighbours, weights=weights, p=p)
            knn_effect.fit(train_x[top_features], train_y)
            effect_train_x = train_x[top_features]
            effect_test_x = test_x[top_features]
        else:
            # 'all' - use every marker, no shortlist
            knn_effect = knn
            top_features = train_x.columns
            effect_train_x = train_x
            effect_test_x = test_x

        background_size = min(shap_background_size, effect_train_x.shape[0])
        background = shap.kmeans(effect_train_x, background_size)
        explainer = shap.KernelExplainer(knn_effect.predict, background)
        effect_scores = abs(explainer.shap_values(
            shap.sample(effect_test_x, shapley_num), nsamples=shap_nsamples
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
    
    return r, mse, effect, predicted, predicted_valid, predicted_train
