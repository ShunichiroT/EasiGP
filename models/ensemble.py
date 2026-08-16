#Change here 
###============###
#DATA_PATH = r'C:\Users\uqstomur\OneDrive - The University of Queensland\Documents\Result\TeoNAM'

#MODEL = ['rrBLUP','BayesB','RKHS','RF','SVR','MLP']

#INPUT = 'Predicted_result_test_all.csv'

###============###


import pandas as pd
import os
from sklearn.metrics import mean_squared_error
import scipy.stats


def _safe_row_normalize(df):
    """Row-wise L1-normalize (each row divided by its own sum of - here
    already non-negative - values, e.g. abs(marker effect)) - safely: a
    row whose sum is exactly 0 (a model that assigned literally zero
    effect to every marker) is left as all-zero, rather than becoming
    all-NaN via an unguarded 0/0 division. A single such row would
    otherwise silently corrupt this WHOLE ensemble's combined marker
    effects once summed with every other (valid) model's row below -
    NaN + anything is NaN, so one degenerate model poisons the average
    for every marker, not just its own."""
    row_sums = df.sum(axis=1)
    return df.div(row_sums, axis=0).fillna(0)


# Function to calculate the metrics
def metric(data):
    mse = mean_squared_error(data.loc[:,'actual'],data.loc[:,'ensemble'])
    r = scipy.stats.pearsonr(data.loc[:,'actual'], data.loc[:,'ensemble'])[0]
    
    return pd.Series(dict(Pearson = r, MSE = mse))

def ensemble(train, valid, test, effect, MODEL):
    # Load prediction result from individual prediction models
    result_train = train
    result_valid = valid
    result_test = test
    
    model_selected = MODEL.copy()
    if 'ensemble' in MODEL:
        model_selected.remove('ensemble')
    
    # Arithmetic mean calculation & perfomrance metrics
    result_test['ensemble'] = result_test.loc[:,model_selected].mean(axis=1)
    if valid.shape[0] != 0:
        result_valid['ensemble'] = result_valid.loc[:,model_selected].mean(axis=1)
    result_train['ensemble'] = result_train.loc[:,model_selected].mean(axis=1)
    
    record = result_test.loc[:,list(result_test.columns[1:6])+['ensemble']].groupby(list(result_test.columns[1:5]), as_index=False).apply(metric).reset_index(drop=True)
    record = record.rename(columns={"Pearson": "Pearson correlation"})
    record['model'] = 'ensemble'
    
    effect = effect.reset_index(drop=True)

    # Extract genomic marker effects
    effect = pd.concat([effect.iloc[:,:5],
                        _safe_row_normalize(effect.iloc[:,5:].abs().reset_index(drop=True))
                       ], axis=1)
    
    effect_ensemble = pd.DataFrame()
    cnt = 0
    for kkk in range(len(model_selected)):
        selected = effect[effect['model']==model_selected[kkk]].reset_index(drop=True)
        if selected.shape[0] != 0:
            if effect_ensemble.shape[0] == 0:
                effect_ensemble = selected
                cnt += 1
            else:
                effect_ensemble.iloc[:,5:] += selected.iloc[:,5:] 
                cnt += 1
    
    if cnt > 0:
        effect_ensemble.iloc[:,5:] = effect_ensemble.iloc[:,5:] / cnt
        effect_ensemble['model'] = 'ensemble'

    return result_train, result_valid, result_test, record, effect_ensemble

