import pandas as pd
import numpy as np
import scipy.stats
from sklearn.metrics import mean_squared_error
from scipy.optimize import Bounds, minimize


def _safe_row_normalize(df):
    """Row-wise L1-normalize (each row divided by its own sum of - here
    already non-negative - values, e.g. abs(marker effect)) - safely: a
    row whose sum is exactly 0 (a model that assigned literally zero
    effect to every marker) is left as all-zero, rather than becoming
    all-NaN via an unguarded 0/0 division, which would otherwise silently
    corrupt this weighted ensemble's entire combined marker-effect output
    - NaN + anything is NaN."""
    row_sums = df.sum(axis=1)
    return df.div(row_sums, axis=0).fillna(0)


def Nelder_Mead(data_train, data_valid, data_test, record, effect, weight, MODEL, HPARAMETERS_OPT):

    initial_value = HPARAMETERS_OPT[0]
    minimum_boundary = HPARAMETERS_OPT[1]
    maximum_boundary = HPARAMETERS_OPT[2]
    fatol = HPARAMETERS_OPT[3]
    xatol = HPARAMETERS_OPT[4]
    adaptive = HPARAMETERS_OPT[5]
    
    model_selected = MODEL.copy()
    if 'ensemble' in MODEL:
        model_selected.remove('ensemble')

    ## Define the objective function to minimise based on the Diversity Prediction Theorem
    def optimisation(weights):
       
       second_term = data_valid.loc[:,model_selected+['actual']]
       third_term, mean_value = second_term.copy(), second_term.iloc[:,:-1].mean(axis=1)
       
       for i in range(len(model_selected)):
           second_term.iloc[:,i] = (second_term.iloc[:,i] - second_term['actual']).pow(2)*weights[i]
           third_term.iloc[:,i] = (third_term.iloc[:,i] - mean_value).pow(2)*weights[i]
           
       second_term = second_term.iloc[:,:-1].div(weights.sum(),axis=1).mean(axis=1)
       third_term = third_term.iloc[:,:-1].div(weights.sum(),axis=1).mean(axis=1)
       first_term = (second_term - third_term).mean()
        
       return first_term
    
    ## Define the range of weights
    x0 = np.array([initial_value]*len(model_selected))
    bounds = Bounds([minimum_boundary]*len(model_selected), [maximum_boundary]*len(model_selected))
    
    ## Implement the minimisation fuciton
    pos = minimize(optimisation, x0, method='nelder-mead', 
           options={'disp': True,
                    'fatol': fatol, 'xatol': xatol, "adaptive": adaptive}, 
           bounds=bounds,
           )
    
    ## Weight extraction
    weight_extracted = pd.DataFrame(pos.x).T
    weight_extracted.columns = model_selected 
    
    # Requirement (safeguard): fall back to equal weighting if the
    # minimiser converged to all-zero weights (sum(pos.x)==0) - rather
    # than raising ZeroDivisionError on the plain-Python division below.
    _abs_x = [abs(x) for x in pos.x]
    _sum_abs_x = sum(_abs_x)
    if _sum_abs_x != 0:
        pos.x = [x / _sum_abs_x for x in _abs_x]
    else:
        pos.x = [1.0 / len(_abs_x)] * len(_abs_x)
    
    ## Weight the predicted phenotypes for the test set
    data_test_selected = data_test.loc[:,model_selected+['actual']]
    for i in range(len(data_test_selected.columns)-1):
        data_test_selected.iloc[:,i] = data_test_selected.iloc[:,i] * pos.x[i]
    predicted_test =  data_test_selected.iloc[:,:-1].sum(axis=1).reset_index(drop=True)
    actual_test =  data_test_selected.iloc[:,-1].tolist()
    
    ## calculate the metrics
    mse = mean_squared_error(actual_test, predicted_test)
    r = scipy.stats.pearsonr(actual_test, predicted_test)[0]
    
    ## Store the metrics
    record = pd.concat([record, pd.DataFrame(record.iloc[-1,:]).T]).reset_index(drop=True)
    record.loc[record.shape[0]-1,'model'] = 'Nelder Mead'
    record.loc[record.shape[0]-1,'Pearson correlation'] = r
    record.loc[record.shape[0]-1,'MSE'] = mse

    ## Weight the predicted phenotypes for the validation set
    predicted_valid = []
    data_valid_selected = data_valid.loc[:,model_selected+['actual']]
    for i in range(len(data_valid_selected.columns)-1):
        data_valid_selected.iloc[:,i] = data_valid_selected.iloc[:,i] * pos.x[i]
    
    ## Calculate the weighted average
    predicted_valid =  data_valid_selected.iloc[:,:-1].sum(axis=1).reset_index(drop=True)
    #actual_valid =  data_valid_selected.iloc[:,-1].tolist()
    
    ## Weight the predicted phenotypes for the training set
    predicted_train = []
    data_train_selected = data_train.loc[:,model_selected+['actual']]
    for i in range(len(data_train_selected.columns)-1):
        data_train_selected.iloc[:,i] = data_train_selected.iloc[:,i] * pos.x[i]
    predicted_train =  data_train_selected.iloc[:,:-1].sum(axis=1).reset_index(drop=True)
    #actual_train =  data_train_selected.iloc[:,-1].tolist()
    data_test['Nelder-Mead'] = predicted_test
    data_valid['Nelder-Mead'] = predicted_valid
    data_train['Nelder-Mead'] = predicted_train

    ## Extract weights
    weight_sample = pd.DataFrame(record.iloc[record.shape[0]-1,:]).T.drop(['Pearson correlation', 'MSE'],axis=1)
    weight_sample['model'] = 'Nelder Mead' 
    
    weight_sample = pd.concat([weight_sample.reset_index(drop=True), weight_extracted.reset_index(drop=True)], axis=1)
    weight = pd.concat([weight, weight_sample])
    
    weight_extracted_normalised = _safe_row_normalize(weight_extracted)
    
    ## Calculate weighted effects
    for i in range(len(model_selected)):
        if effect[effect['model']==model_selected[i]].shape[0] != 0:
            if i == 0:
                effect_weighted = _safe_row_normalize(effect[effect['model']==model_selected[i]].tail(1).iloc[:,5:].abs().reset_index(drop=True)).mul(weight_extracted_normalised[model_selected[i]], axis=0).reset_index(drop=True)
            else:
                effect_weighted += _safe_row_normalize(effect[effect['model']==model_selected[i]].tail(1).iloc[:,5:].abs().reset_index(drop=True)).mul(weight_extracted_normalised[model_selected[i]], axis=0).reset_index(drop=True)

    effect = pd.concat([effect, pd.DataFrame(effect.iloc[effect.shape[0]-1,:]).T]).reset_index(drop=True)
    effect.loc[effect.shape[0]-1,'model'] = 'Nelder Mead'
    
    effect.loc[effect.shape[0]-1,list(effect_weighted.columns)] = effect_weighted.values

    return record, effect, data_test, data_valid, data_train, weight