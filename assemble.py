import pandas as pd
import os
import glob


# Combine prediction results from all batches
def assemble():
    
    prediction_train = pd.DataFrame()
    prediction_valid = pd.DataFrame()
    prediction_test = pd.DataFrame()
    metric = pd.DataFrame()
    marker_effect = pd.DataFrame()
    interaction = pd.DataFrame()
    weight = pd.DataFrame()
    
    for i in range(len(glob.glob1('./Result',"Metric_*.csv"))):
        if os.path.isfile('./Result/Prediction_result_train_'+str(i)+'.csv'):
            prediction_train = pd.concat([prediction_train,
                                          pd.read_csv('./Result/Prediction_result_train_'+str(i)+'.csv')])
        if os.path.isfile('./Result/Prediction_result_valid_'+str(i)+'.csv'):
            prediction_valid = pd.concat([prediction_valid,
                                          pd.read_csv('./Result/Prediction_result_valid_'+str(i)+'.csv')])
        if os.path.isfile('./Result/Prediction_result_test_'+str(i)+'.csv'):
            prediction_test = pd.concat([prediction_test,
                                         pd.read_csv('./Result/Prediction_result_test_'+str(i)+'.csv')])
        if os.path.isfile('./Result/Metric_'+str(i)+'.csv'):
            metric = pd.concat([metric,
                                pd.read_csv('./Result/Metric_'+str(i)+'.csv')])
        if os.path.isfile('./Result/Marker_effect_'+str(i)+'.csv'):
            marker_effect = pd.concat([marker_effect,
                                       pd.read_csv('./Result/Marker_effect_'+str(i)+'.csv')])
        if os.path.isfile('./Result/Interaction_'+str(i)+'.csv'):
            interaction = pd.concat([interaction,
                                     pd.read_csv('./Result/Interaction_'+str(i)+'.csv')])
        if os.path.isfile('./Result/Weight_'+str(i)+'.csv'):
            weight = pd.concat([weight,
                                pd.read_csv('./Result/Weight_'+str(i)+'.csv')])

    prediction_train.to_csv('./Result/Prediction_result_train.csv',index=False)
    prediction_valid.to_csv('./Result/Prediction_result_valid.csv',index=False)
    prediction_test.to_csv('./Result/Prediction_result_test.csv',index=False)
    metric.to_csv('./Result/Metric.csv',index=False)
    marker_effect.to_csv('./Result/Marker_effect.csv',index=False)
    interaction.to_csv('./Result/Interaction.csv',index=False)
    weight.to_csv('./Result/Weight.csv',index=False)
    
    MODEL = pd.unique(metric['type']).tolist() 
    if 'Linear transformation' in MODEL:
        MODEL.remove('Linear transformation')
    if 'Nelder Mead' in MODEL:
        MODEL.remove('Nelder Mead')
    if 'Bayesian optimisation' in MODEL:
        MODEL.remove('Bayesian optimisation')
    
    return metric, prediction_train, prediction_test, marker_effect, interaction, \
            pd.unique(metric['population']).tolist(), pd.unique(metric['phenotype']).tolist(), MODEL  