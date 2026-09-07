#Change here 
###============###
#DATA_PATH = r'C:\Users\uqstomur\OneDrive - The University of Queensland\Documents\Result\TeoNAM'

#MODEL = ['rrBLUP','BayesB','RKHS','RF','SVR','MLP']

#INPUT = 'Predicted_result_test_all.csv'

###============###


import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error
import scipy.stats

from pipeline_utils import safe_regression_metrics
from models.interaction_extraction import naive_ensemble_interactions, _INTERACTION_ENSEMBLE_SCHEMA


def _safe_row_normalize(df):
    """Row-wise L1-normalize (each row divided by its own sum of - here
    already non-negative - values, e.g. abs(marker effect)) - safely: a
    row whose sum is exactly 0 (a model that assigned literally zero
    effect to every marker - e.g. SVR/KNN, which aren't additive-effect
    models and so never report per-marker effects at all) is left as
    all-zero, rather than corrupting or crashing this whole ensemble's
    combined marker effects.

    Bug fix: this used to just be `df.div(df.sum(axis=1), axis=0).
    fillna(0)`, on the assumption that dividing by exactly 0 always
    produces NaN, which .fillna(0) then mops up. That's only true when df
    is a clean float64 DataFrame. In production this df's columns can
    come back as pandas 'object' dtype (marker-effect values assembled
    from the R-backed models - rrBLUP/BayesB/RKHS - via rpy2), and on an
    object-dtype column pandas' .div() falls back to plain Python
    arithmetic per element instead of numpy's vectorised path - and plain
    Python division by exactly 0.0 RAISES ZeroDivisionError rather than
    returning NaN, so .fillna(0) never gets a chance to run at all.
    Confirmed as the root cause of a real production crash: hours of
    hyperparameter tuning across every model completed successfully, then
    this single division crashed the process during the final marker-
    effect combination step, before any output file was ever written -
    which is exactly why a run can finish all its logged steps and still
    produce no output files at all.

    Fixed by never attempting to divide by an exact 0 in the first place
    (replacing it with NaN before dividing, not after) and by forcing a
    clean float64 dtype regardless of what the caller passed in - either
    change alone would have prevented the crash; both together make this
    safe no matter what dtype the input arrives in."""
    df = df.astype(float)
    row_sums = df.sum(axis=1).replace(0, np.nan)
    return df.div(row_sums, axis=0).fillna(0)


# Function to calculate the metrics
def metric(data):
    # ver4-4 R5 Fix 2: safe_regression_metrics() never raises, even when
    # an ensembled prediction column carries NaN (e.g. one of the
    # ensembled models diverged - see R5 design record). The direct
    # mean_squared_error()/pearsonr() pair this replaces is exactly what
    # used to raise ValueError('Input contains NaN') here, once that NaN
    # reaches the naive ensemble.
    r, mse = safe_regression_metrics(data.loc[:, 'actual'], data.loc[:, 'ensemble'])

    return pd.Series(dict(Pearson = r, MSE = mse))

def ensemble(train, valid, test, effect, MODEL, interactions=None):
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

    # Requirement_patch3.md item 1: the naive-ensemble analogue of the
    # marker-EFFECT combination above, for marker-PAIR interactions -
    # normalises away each contributing model's own interaction-value
    # scale, then takes the equal-weight (naive) mean across whichever
    # selected models actually reported an interaction for each task -
    # see models/interaction_extraction.py::naive_ensemble_interactions()
    # for the full contract. `interactions` is optional (defaults to
    # None) purely so this signature stays callable exactly as before
    # for any caller that has no interaction data to offer; every actual
    # caller in this codebase (genomic_prediction.py's finalisation
    # block, intra_task_parallel.py's model-fan-out equivalent) always
    # passes its own accumulated `interactions` DataFrame.
    if interactions is not None and interactions.shape[0] != 0:
        interaction_ensemble = naive_ensemble_interactions(interactions, model_selected)
    else:
        interaction_ensemble = pd.DataFrame(columns=_INTERACTION_ENSEMBLE_SCHEMA)

    return result_train, result_valid, result_test, record, effect_ensemble, interaction_ensemble

