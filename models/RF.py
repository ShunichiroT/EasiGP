from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import shap

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


def RF(train, valid, test, params):
    
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
    
    #Split the data sets into x and y here as specified in the original code
    train_x, train_y = train.iloc[:,:-1], train.iloc[:,-1]    
    if valid.shape[0] != 0:
        valid_x, valid_y = valid.iloc[:,:-1], valid.iloc[:,-1]
    test_x, test_y = test.iloc[:,:-1], test.iloc[:,-1]
    
    #Develop & evaluate a model here as specified in the original code
    rf = RandomForestRegressor(n_estimators = estimators, random_state = 0, max_samples=sample_max,
                                max_features=features_max, max_depth=max_depth,
                                min_samples_leaf=min_samples_leaf, n_jobs=-1)
    rf.fit(train_x, train_y)
    
    predicted = rf.predict(test_x)
    if valid.shape[0] != 0:
        predicted_valid = rf.predict(valid_x)
    else:
        predicted_valid = []
    predicted_train = rf.predict(train_x)

    #Calculate the metrics
    actual_test = test_y.values.tolist()
    mse = mean_squared_error(actual_test, predicted)
    r = pearsonr(actual_test, predicted)[0]
    
    #Extract interactions 
    if get_interaction == True:
        # Defensive clamp: avoid shap.sample() erroring if shapley_num exceeds
        # the number of available test rows.
        shapley_num = min(shapley_num, test_x.shape[0])

        n_features = train_x.shape[1]
        if max_interaction_features != 'all' and n_features > max_interaction_features:
            # Refit a lightweight forest on just the top-importance markers so
            # the SHAP interaction matrix is max_interaction_features x
            # max_interaction_features instead of n_features x n_features.
            top_features = train_x.columns[
                np.argsort(rf.feature_importances_)[::-1][:max_interaction_features]
            ]
            rf_interaction = RandomForestRegressor(n_estimators = estimators, random_state = 0,
                                                     max_samples=sample_max, max_features=features_max,
                                                     max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                                     n_jobs=-1)
            rf_interaction.fit(train_x[top_features], train_y)
            interaction_test_x = test_x[top_features]
        else:
            # 'all' - use every marker, no cap
            rf_interaction = rf
            interaction_test_x = test_x

        explainer = shap.TreeExplainer(rf_interaction)
        interaction_sample = pd.DataFrame(abs(explainer.shap_interaction_values(shap.sample(interaction_test_x, shapley_num))).sum(axis=0))
        interaction_sample = interaction_sample.where(np.triu(np.ones(interaction_sample.shape)).astype(bool))
        np.fill_diagonal(interaction_sample.values, np.nan)
        
        interaction_sample.index = interaction_sample.columns = interaction_test_x.columns
        interaction_sample = interaction_sample.stack(dropna=True).reset_index(drop=False)
        interaction_sample.columns = ['marker1','marker2','value']
        
        if threshold != 'all':
            interaction_sample = interaction_sample[interaction_sample['value'] > interaction_sample['value'].quantile((1-(threshold/100)))]
    else:
        interaction_sample = pd.DataFrame()
    
    return r, mse, pd.DataFrame(rf.feature_importances_).T, interaction_sample, predicted, predicted_valid, predicted_train

    
