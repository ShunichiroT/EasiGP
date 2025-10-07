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
    record['type'] = 'ensemble'
    
    effect = effect.reset_index(drop=True)

    # Extract genomic marker effects
    effect = pd.concat([effect.iloc[:,:5],
                        effect.iloc[:,5:].abs().reset_index(drop=True).div(effect.iloc[:,5:].abs().sum(axis=1).reset_index(drop=True),axis=0)
                       ], axis=1)
    
    effect_ensemble = pd.DataFrame()
    cnt = 0
    for kkk in range(len(model_selected)):
        selected = effect[effect['type']==model_selected[kkk]].reset_index(drop=True)
        if selected.shape[0] != 0:
            if effect_ensemble.shape[0] == 0:
                effect_ensemble = selected
                cnt += 1
            else:
                effect_ensemble.iloc[:,5:] += selected.iloc[:,5:] 
                cnt += 1
    
    effect_ensemble.iloc[:,5:] = effect_ensemble.iloc[:,5:] / cnt
    effect_ensemble['type'] = 'ensemble'
        
    return result_train, result_valid, result_test, record, effect_ensemble

