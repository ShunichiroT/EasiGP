import pandas as pd
import os
import glob
import re


# Combine prediction results from all batches
def assemble(RESULT_NAME):
    
    prediction_train = pd.DataFrame()
    prediction_valid = pd.DataFrame()
    prediction_test = pd.DataFrame()
    metric = pd.DataFrame()
    marker_effect = pd.DataFrame()
    interaction = pd.DataFrame()
    weight = pd.DataFrame()
    attention = pd.DataFrame()
    
    # Discover the actual batch IDs present on disk rather than assuming a
    # contiguous 0..N-1 range. Array jobs can have gaps (a batch failed and
    # is resubmitted on its own, or only a subset of failed batches is
    # rerun), so counting files and looping range(count) can silently skip
    # existing higher-numbered batches while re-checking indices that never
    # existed.
    batch_ids = sorted(
        int(match.group(1))
        for f in glob.glob1('./Result/'+RESULT_NAME+'/', 'Metric_*.csv')
        if (match := re.match(r'Metric_(\d+)\.csv$', f))
    )

    for i in batch_ids:
        if os.path.isfile('./Result/'+RESULT_NAME+'/Prediction_result_train_'+str(i)+'.csv'):
            prediction_train = pd.concat([prediction_train,
                                          pd.read_csv('./Result/'+RESULT_NAME+'/Prediction_result_train_'+str(i)+'.csv')])
        if os.path.isfile('./Result/'+RESULT_NAME+'/Prediction_result_valid_'+str(i)+'.csv'):
            try:
                prediction_valid = pd.concat([prediction_valid,
                                              pd.read_csv('./Result/'+RESULT_NAME+'/Prediction_result_valid_'+str(i)+'.csv')])
            except pd.errors.EmptyDataError:
                prediction_valid = pd.DataFrame()
        if os.path.isfile('./Result/'+RESULT_NAME+'/Prediction_result_test_'+str(i)+'.csv'):
            prediction_test = pd.concat([prediction_test,
                                         pd.read_csv('./Result/'+RESULT_NAME+'/Prediction_result_test_'+str(i)+'.csv')])
        if os.path.isfile('./Result/'+RESULT_NAME+'/Metric_'+str(i)+'.csv'):
            metric = pd.concat([metric,
                                pd.read_csv('./Result/'+RESULT_NAME+'/Metric_'+str(i)+'.csv')])
        if os.path.isfile('./Result/'+RESULT_NAME+'/Marker_effect_'+str(i)+'.csv'):
            marker_effect = pd.concat([marker_effect,
                                       pd.read_csv('./Result/'+RESULT_NAME+'/Marker_effect_'+str(i)+'.csv')])
        if os.path.isfile('./Result/'+RESULT_NAME+'/Interaction_'+str(i)+'.csv'):
            interaction = pd.concat([interaction,
                                     pd.read_csv('./Result/'+RESULT_NAME+'/Interaction_'+str(i)+'.csv')])
        if os.path.isfile('./Result/'+RESULT_NAME+'/Weight_'+str(i)+'.csv'):
            try:
                weight = pd.concat([weight,
                                    pd.read_csv('./Result/'+RESULT_NAME+'/Weight_'+str(i)+'.csv')])
            except pd.errors.EmptyDataError:
                weight = pd.DataFrame()
        if os.path.isfile('./Result/'+RESULT_NAME+'/Attention_'+str(i)+'.csv'):
            try:
                attention = pd.concat([attention,
                                    pd.read_csv('./Result/'+RESULT_NAME+'/Attention_'+str(i)+'.csv')])
            except pd.errors.EmptyDataError:
                attention = pd.DataFrame()

    prediction_train.to_csv('./Result/'+RESULT_NAME+'/Prediction_result_train.csv',index=False)
    prediction_valid.to_csv('./Result/'+RESULT_NAME+'/Prediction_result_valid.csv',index=False)
    prediction_test.to_csv('./Result/'+RESULT_NAME+'/Prediction_result_test.csv',index=False)
    metric.to_csv('./Result/'+RESULT_NAME+'/Metric.csv',index=False)
    marker_effect.to_csv('./Result/'+RESULT_NAME+'/Marker_effect.csv',index=False)
    interaction.to_csv('./Result/'+RESULT_NAME+'/Interaction.csv',index=False)
    
    if weight.shape[0] != 0: 
        weight.to_csv('./Result/'+RESULT_NAME+'/Weight.csv',index=False)
    if attention.shape[0] != 0:
        attention.to_csv('./Result/'+RESULT_NAME+'/Attention.csv',index=False)
    
    MODEL = pd.unique(metric['model']).tolist() 
    if 'Linear transformation' in MODEL:
        MODEL.remove('Linear transformation')
    if 'Nelder Mead' in MODEL:
        MODEL.remove('Nelder Mead')
    if 'Bayesian optimisation' in MODEL:
        MODEL.remove('Bayesian optimisation')
    
    return metric, prediction_train, prediction_test, marker_effect, interaction, attention,\
            pd.unique(metric['population']).tolist(), pd.unique(metric['phenotype']).tolist(), MODEL  


# Skip re-combining every batch and instead load the already-assembled combined
# CSVs that a previous assemble() call wrote to disk. Useful when Step 2 has
# already been run successfully once and only the scatter/circos plotting
# step needs retrying (e.g. after fixing a QTL file or circos config issue) -
# re-running assemble() on every retry is wasteful, and for many batches,
# slow, when the combined files are already sitting there correctly.
def load_assembled(RESULT_NAME):

    result_dir = './Result/'+RESULT_NAME+'/'
    required = ['Metric.csv', 'Prediction_result_train.csv', 'Prediction_result_test.csv',
                'Marker_effect.csv', 'Interaction.csv']
    missing = [f for f in required if not os.path.isfile(result_dir+f)]
    if missing:
        raise FileNotFoundError(
            "Cannot skip assemble: the following previously-assembled file(s) are "
            f"missing from '{result_dir}': {', '.join(missing)}. Run Step 2 at least "
            "once with 'Skip assemble' unchecked first."
        )

    metric = pd.read_csv(result_dir+'Metric.csv')
    prediction_train = pd.read_csv(result_dir+'Prediction_result_train.csv')
    prediction_test = pd.read_csv(result_dir+'Prediction_result_test.csv')
    marker_effect = pd.read_csv(result_dir+'Marker_effect.csv')
    interaction = pd.read_csv(result_dir+'Interaction.csv')

    attention = pd.DataFrame()
    if os.path.isfile(result_dir+'Attention.csv'):
        try:
            attention = pd.read_csv(result_dir+'Attention.csv')
        except pd.errors.EmptyDataError:
            attention = pd.DataFrame()

    MODEL = pd.unique(metric['model']).tolist()
    if 'Linear transformation' in MODEL:
        MODEL.remove('Linear transformation')
    if 'Nelder Mead' in MODEL:
        MODEL.remove('Nelder Mead')
    if 'Bayesian optimisation' in MODEL:
        MODEL.remove('Bayesian optimisation')

    return metric, prediction_train, prediction_test, marker_effect, interaction, attention,\
            pd.unique(metric['population']).tolist(), pd.unique(metric['phenotype']).tolist(), MODEL
