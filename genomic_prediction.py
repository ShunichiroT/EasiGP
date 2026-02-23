import pandas as pd
import os
from itertools import product
from sklearn.model_selection import train_test_split

from models.RF import *
from models.SVR import *
from models.MLP import *
from models.GAT_infinitesimal_node_level import *
from models.GAT_infinitesimal import *
from models.GAT_fully_connected import *
from models.GAT_prior_knowledge import *
from models.ensemble import *

from models.Linear_transformation import *
from models.Nelder_Mead import *
from models.Bayesian_optimisation import *
    
def GP(GENOTYPE_FILE_NAME, PHENOTYPE_FILE_NAME, MODEL, PHENOTYPE, RATIO, SAMPLE_NUM, HPARAMETERS, R_PATH, W_OPT, RESULT_NAME, HYPERPARAMETERS_OPT, SCENARIO, PARALLEL=None):
    
    # Import R modules
    if R_PATH != None:
        os.environ['R_HOME'] = R_PATH
    
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    
    r_source = robjects.r['source']
    r_source('./models/rrBLUP.R')
    rrBLUP = robjects.globalenv['rrBLUP']
    r_source('./models/GBLUP.R')
    GBLUP = robjects.globalenv['GBLUP']
    r_source('./models/BayesB.R')
    BayesB = robjects.globalenv['BayesB']
    r_source('./models/RKHS.R')
    RKHS = robjects.globalenv['RKHS']
    
    # Read genotype and phenotype data
    data_genotype_original = pd.read_csv(GENOTYPE_FILE_NAME)
    data_phenotype_original = pd.read_csv(PHENOTYPE_FILE_NAME)
    
    POPULATION = pd.unique(data_genotype_original['population'])
    
    if (type(PHENOTYPE) is not list) and (PHENOTYPE == 'all'):
        PHENOTYPE = list(data_phenotype_original.columns[2:])
    
    # Create the total number of combinations of prediction scenarios
    if SCENARIO == 'within':
        sample = pd.DataFrame({'population':[item for item in POPULATION for i in range(SAMPLE_NUM*len(PHENOTYPE)*len(RATIO))],
                                'phenotype':[item for item in PHENOTYPE for i in range(SAMPLE_NUM*len(RATIO))]*len(POPULATION),
                                'ratio':[item for item in RATIO for i in range(SAMPLE_NUM)]*len(POPULATION)*len(PHENOTYPE),
                                'sample':list(range(1,SAMPLE_NUM+1)) *len(PHENOTYPE)*len(POPULATION)*len(RATIO)})
    elif SCENARIO == 'between':
        if W_OPT is None:
            comb = [str(x)+'->'+str(y) for x in POPULATION for y in POPULATION]
            sample = pd.DataFrame({'population':[item for item in comb] *len(PHENOTYPE),
                                    'phenotype':[item for item in PHENOTYPE for i in range(len(POPULATION))]*len(POPULATION),
                                    'ratio':[-1]*len(POPULATION)*len(PHENOTYPE)*len(POPULATION),
                                    'sample':[-1] *len(PHENOTYPE)*len(POPULATION)*len(POPULATION)})
            tmp = sample['population'].str.split('->',expand=True)
            sample = sample[tmp[0]!=tmp[1]].reset_index(drop=True)
        else:
            comb = [str(x)+'->'+str(y)+'->'+str(z) for x in POPULATION for y in POPULATION for z in POPULATION]
            sample = pd.DataFrame({'population':[item for item in comb] *len(PHENOTYPE),
                                    'phenotype':[item for item in PHENOTYPE for i in range(len(POPULATION))]*len(POPULATION)*len(POPULATION),
                                    'ratio':[-1]*len(POPULATION)*len(PHENOTYPE)*len(POPULATION)*len(POPULATION),
                                    'sample':[-1] *len(PHENOTYPE)*len(POPULATION)*len(POPULATION)*len(POPULATION)})
            tmp = sample['population'].str.split('->',expand=True)
            sample = sample[(tmp[0]!=tmp[1]) & (tmp[0]!=tmp[2]) & (tmp[1]!=tmp[2])].reset_index(drop=True)

    record = pd.DataFrame()          #store performance metrics
    result_train = pd.DataFrame()    #store predicted phenotypes for train set
    result_valid = pd.DataFrame()    #store predicted phenotypes for validation set
    result_test = pd.DataFrame()     #store predicted phenotypes for test set
    effect = pd.DataFrame()          #store genomic marker effects
    interactions = pd.DataFrame()    #store marker interaction effects
    weight = pd.DataFrame()          #store weight values for weight optimisation
    attention_total = pd.DataFrame() #store attention values from GAT models
    
    if PARALLEL is not None:
        idx = PARALLEL['batch_id']
        interval = PARALLEL['batch_size']
    else:
        idx = 0
        interval = sample.shape[0]
        
    for i in range(idx*interval, idx*interval+interval):
        
        if i >= sample.shape[0]:
            break
        
        # Convert the data structure
        if SCENARIO == 'within':
            data_genotype = data_genotype_original[data_genotype_original['population']==sample.loc[i,'population']].reset_index(drop=True)
            data_phenotype = data_phenotype_original.loc[data_phenotype_original['population']==sample.loc[i,'population'], ['ID','population',sample.loc[i,'phenotype']]].reset_index(drop=True)
        elif SCENARIO == 'between':
            if W_OPT is None:
                data_genotype = data_genotype_original[(data_genotype_original['population'].astype(str) == sample.loc[i, 'population'].split('->')[0]) | 
                                                       (data_genotype_original['population'].astype(str) == sample.loc[i, 'population'].split('->')[1])].reset_index(drop=True)
                data_phenotype = data_phenotype_original.loc[(data_phenotype_original['population'].astype(str) ==sample.loc[i,'population'].split('->')[0]) | 
                                                             (data_phenotype_original['population'].astype(str) ==sample.loc[i,'population'].split('->')[1]), 
                                                             ['ID','population',sample.loc[i,'phenotype']]].reset_index(drop=True)
            else:
                data_genotype = data_genotype_original[(data_genotype_original['population'].astype(str) == sample.loc[i, 'population'].split('->')[0]) | 
                                                       (data_genotype_original['population'].astype(str) == sample.loc[i, 'population'].split('->')[1]) |
                                                       (data_genotype_original['population'].astype(str) == sample.loc[i, 'population'].split('->')[2])].reset_index(drop=True)
                data_phenotype = data_phenotype_original.loc[(data_phenotype_original['population'].astype(str) ==sample.loc[i,'population'].split('->')[0]) | 
                                                             (data_phenotype_original['population'].astype(str) ==sample.loc[i,'population'].split('->')[1]) |
                                                             (data_phenotype_original['population'].astype(str) ==sample.loc[i,'population'].split('->')[2]), 
                                                             ['ID','population',sample.loc[i,'phenotype']]].reset_index(drop=True)
        
        data = data_genotype.merge(data_phenotype, on=['ID','population']).dropna().reset_index(drop=True)

        if SCENARIO =='within':
            if type(sample.loc[i,'ratio']) is not tuple:
                train, test = train_test_split(data,train_size=sample.loc[i,'ratio'], random_state=sample.loc[i,'sample'])
                train, test = train.reset_index(drop=True), test.reset_index(drop=True)
                id_train, id_valid, id_test = train.iloc[:,0], pd.DataFrame(), test.iloc[:,0]
                train, valid, test = train.iloc[:,2:], pd.DataFrame(), test.iloc[:,2:]
            elif type(sample.loc[i,'ratio']) is tuple:
                train, valid = train_test_split(data,train_size=sample.loc[i,'ratio'][0], random_state=sample.loc[i,'sample'])
                valid, test = train_test_split(valid,train_size=sample.loc[i,'ratio'][1]/(sample.loc[i,'ratio'][2]+sample.loc[i,'ratio'][1]), random_state=sample.loc[i,'sample'])
                train, valid, test = train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)
                id_train, id_valid, id_test = train.iloc[:,0], valid.iloc[:,0], test.iloc[:,0]
                train, valid, test = train.iloc[:,2:], valid.iloc[:,2:], test.iloc[:,2:]
        elif SCENARIO =='between':
            tmp = sample.loc[i, 'population'].split('->')
            if W_OPT is None:
                train, test = data[data['population'].astype(str)==tmp[0]], data[data['population'].astype(str)==tmp[1]]
                train, test = train.reset_index(drop=True), test.reset_index(drop=True)
                id_train, id_valid, id_test = train.iloc[:,0], pd.DataFrame(), test.iloc[:,0]
                train, valid, test = train.iloc[:,2:], pd.DataFrame(), test.iloc[:,2:] 
            elif W_OPT is not None:
                train, valid, test = data[data['population'].astype(str)==tmp[0]], data[data['population'].astype(str)==tmp[1]], data[data['population'].astype(str)==tmp[2]]
                train, valid, test = train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)
                id_train, id_valid, id_test = train.iloc[:,0], valid.iloc[:,0], test.iloc[:,0]
                train, valid, test = train.iloc[:,2:], valid.iloc[:,2:], test.iloc[:,2:]
        
        result_train_sample = pd.DataFrame()
        result_valid_sample = pd.DataFrame()
        result_test_sample = pd.DataFrame()
        
        # Prediction model implementation
        for jj in range(len(MODEL)):
            if MODEL[jj] == 'ensemble':
                continue
            elif MODEL[jj] == 'rrBLUP':
                sample_pearson, sample_mse, sample_effect, predicted_test, predicted_valid, predicted_train = rrBLUP(train, valid, test, HPARAMETERS[MODEL[jj]], RESULT_NAME)
                sample_pearson, sample_mse = sample_pearson[0], sample_mse[0]
                sample_effect = robjects.conversion.rpy2py(sample_effect)
                predicted_test = robjects.conversion.rpy2py(predicted_test)
                predicted_valid = robjects.conversion.rpy2py(predicted_valid)
                predicted_train = robjects.conversion.rpy2py(predicted_train)
            elif MODEL[jj] == 'GBLUP':
                sample_pearson, sample_mse, sample_effect, predicted_test, predicted_valid, predicted_train = GBLUP(train, valid, test, HPARAMETERS[MODEL[jj]], RESULT_NAME)
                sample_pearson, sample_mse = sample_pearson[0], sample_mse[0]
                sample_effect = robjects.conversion.rpy2py(sample_effect)
                predicted_test = robjects.conversion.rpy2py(predicted_test)
                predicted_valid = robjects.conversion.rpy2py(predicted_valid)
                predicted_train = robjects.conversion.rpy2py(predicted_train)
            elif MODEL[jj]  == 'BayesB':
                sample_pearson, sample_mse, sample_effect, predicted_test, predicted_valid, predicted_train = BayesB(train, valid, test, HPARAMETERS[MODEL[jj]],RESULT_NAME)
                sample_pearson, sample_mse = sample_pearson[0], sample_mse[0]
                sample_effect = robjects.conversion.rpy2py(sample_effect)
                predicted_test = robjects.conversion.rpy2py(predicted_test)
                predicted_valid = robjects.conversion.rpy2py(predicted_valid)
                predicted_train = robjects.conversion.rpy2py(predicted_train)
            elif MODEL[jj]  == 'RKHS':
                if HPARAMETERS[MODEL[jj]][-2] == 'all':
                    HPARAMETERS[MODEL[jj]][-2] = test.shape[0]       
                sample_pearson, sample_mse, sample_effect, predicted_test, predicted_valid, predicted_train = RKHS(train, valid, test, HPARAMETERS[MODEL[jj]],RESULT_NAME)
                sample_pearson, sample_mse = sample_pearson[0], sample_mse[0]
                sample_effect = robjects.conversion.rpy2py(sample_effect)
                predicted_test = robjects.conversion.rpy2py(predicted_test)
                predicted_valid = robjects.conversion.rpy2py(predicted_valid)
                predicted_train = robjects.conversion.rpy2py(predicted_train)
            elif MODEL[jj]  == 'RF':
                if HPARAMETERS[MODEL[jj]][-2] == 'all':
                    HPARAMETERS[MODEL[jj]][-2] = test.shape[0] 
                sample_pearson, sample_mse, sample_effect, sample_interaction, predicted_test, predicted_valid, predicted_train = RF(train, valid, test, HPARAMETERS[MODEL[jj]])
            elif MODEL[jj]  == 'SVR':
                if HPARAMETERS[MODEL[jj]][-2] == 'all':
                    HPARAMETERS[MODEL[jj]][-2] = test.shape[0] 
                sample_pearson, sample_mse, sample_effect, predicted_test, predicted_valid, predicted_train = SV_Regression(train, valid, test, HPARAMETERS[MODEL[jj]])
            elif MODEL[jj]  == 'MLP':
                if HPARAMETERS[MODEL[jj]][-1] == 'all':
                    HPARAMETERS[MODEL[jj]][-1] = test.shape[0] 
                sample_pearson, sample_mse, sample_effect, predicted_test, predicted_valid, predicted_train = ML_Perceptron(train, valid, test, HPARAMETERS[MODEL[jj]])
            elif MODEL[jj]  == 'GAT_infinitesimal_node_level':
                if HPARAMETERS[MODEL[jj]][-1] == 'all':
                    HPARAMETERS[MODEL[jj]][-1] = test.shape[0] 
                sample_pearson, sample_mse, sample_effect, predicted_test, predicted_valid, predicted_train = GAT_infinitesimal_node_level(train, valid, test, HPARAMETERS[MODEL[jj]])
            elif MODEL[jj]  == 'GAT_infinitesimal':
                if HPARAMETERS[MODEL[jj]][-1] == 'all':
                    HPARAMETERS[MODEL[jj]][-1] = test.shape[0] 
                sample_pearson, sample_mse, sample_effect, predicted_test, predicted_valid, predicted_train = GAT_infinitesimal(train, valid, test, HPARAMETERS[MODEL[jj]])
            elif MODEL[jj]  == 'GAT_fully_connected':
                if HPARAMETERS[MODEL[jj]][-1] == 'all':
                    HPARAMETERS[MODEL[jj]][-1] = test.shape[0] 
                sample_pearson, sample_mse, sample_effect, predicted_test, predicted_valid, predicted_train, sample_attention = GAT_fully_connected(train, valid, test, HPARAMETERS[MODEL[jj]])
            elif MODEL[jj]  == 'GAT_prior_knowledge':
                if HPARAMETERS[MODEL[jj]][-1] == 'all':
                    HPARAMETERS[MODEL[jj]][-1] = test.shape[0] 
                sample_pearson, sample_mse, sample_effect, predicted_test, predicted_valid, predicted_train, sample_attention = GAT_prior_knowledge(train, valid, test, HPARAMETERS[MODEL[jj]])
            
            # Store prediction results
            record_sample = pd.DataFrame([{'population': sample.loc[i,'population'],
                                           'phenotype': sample.loc[i,'phenotype'],
                                           'model': MODEL[jj],
                                           'ratio': sample.loc[i,'ratio'],
                                           'sample': sample.loc[i,'sample'],
                                           'Pearson correlation': sample_pearson,
                                           'MSE': sample_mse}
                                          ])
            record = pd.concat([record, record_sample])
            
            if result_test_sample.shape[0]==0:
                result_test_sample = pd.DataFrame({'id': id_test,
                                                   'population': [sample.loc[i,'population']] * len(id_test),
                                                   'ratio': [sample.loc[i,'ratio']] * len(id_test),
                                                   'phenotype': [sample.loc[i,'phenotype']] * len(id_test),
                                                   'sample': [sample.loc[i,'sample']] * len(id_test),
                                                   'actual':test.iloc[:,-1],
                                                   MODEL[jj]:predicted_test
                                                  })
            else:
                result_test_sample = pd.concat([result_test_sample,
                                                pd.DataFrame({MODEL[jj]:predicted_test})
                                              ], axis=1)
           
            if (result_valid_sample.shape[0]==0 and type(sample.loc[i,'ratio']) is tuple) or (result_valid_sample.shape[0]==0 and W_OPT is not None and SCENARIO == 'between'):
                result_valid_sample = pd.DataFrame({'id': id_valid,
                                                   'population': [sample.loc[i,'population']] * len(id_valid),
                                                   'ratio': [sample.loc[i,'ratio']] * len(id_valid),
                                                   'phenotype': [sample.loc[i,'phenotype']] * len(id_valid),
                                                   'sample': [sample.loc[i,'sample']] * len(id_valid),
                                                   'actual':valid.iloc[:,-1],
                                                   MODEL[jj]:predicted_valid
                                                  })
            elif (result_valid_sample.shape[0]!=0 and type(sample.loc[i,'ratio']) is tuple) or (W_OPT is not None and SCENARIO == 'between'):
                result_valid_sample = pd.concat([result_valid_sample,
                                                pd.DataFrame({MODEL[jj]:predicted_valid})
                                              ], axis=1) 
            else:
                result_valid_sample = pd.DataFrame()
            
            if result_train_sample.shape[0]==0:
                result_train_sample = pd.DataFrame({'id': id_train,
                                                   'population': [sample.loc[i,'population']] * len(id_train),
                                                   'ratio': [sample.loc[i,'ratio']] * len(id_train),
                                                   'phenotype': [sample.loc[i,'phenotype']] * len(id_train),
                                                   'sample': [sample.loc[i,'sample']] * len(id_train),
                                                   'actual':train.iloc[:,-1],
                                                   MODEL[jj]:predicted_train
                                                  })
            else:
                result_train_sample = pd.concat([result_train_sample,
                                                pd.DataFrame({MODEL[jj]:predicted_train})
                                              ], axis=1)  
            
            if sample_effect.shape[0] != 0:
                sample_effect.columns = train.columns.tolist()[:-1]
                effect_sample = pd.DataFrame([{'population': sample.loc[i,'population'],
                                               'phenotype': sample.loc[i,'phenotype'],
                                               'model': MODEL[jj],
                                               'ratio': sample.loc[i,'ratio'],
                                               'sample': sample.loc[i,'sample'],
                                               }])
                effect_sample = pd.concat([effect_sample,
                                           sample_effect.reset_index(drop=True),
                                          ],axis=1)
                effect = pd.concat([effect, effect_sample])
           
            if MODEL[jj]  == 'RF' and sample_interaction.shape[0] != 0:
                sample_interaction['population'] = sample.loc[i,'population']
                sample_interaction['phenotype'] = sample.loc[i,'phenotype']
                sample_interaction['model'] = MODEL[jj]
                sample_interaction['ratio'] = str(sample.loc[i,'ratio'])
                sample_interaction['sample'] = sample.loc[i,'sample'] 
                
                interactions = pd.concat([interactions, sample_interaction])   
            
            if MODEL[jj] == 'GAT_fully_connected' or MODEL[jj] == 'GAT_prior_knowledge':
                sample_attention['population'] = sample.loc[i,'population']
                sample_attention['model'] = MODEL[jj]
                sample_attention['ratio'] = str(sample.loc[i,'ratio'])
                sample_attention['phenotype'] = sample.loc[i,'phenotype']
                sample_attention['sample'] = sample.loc[i,'sample'] 
                sample_attention.columns = ['from','to','value','population','model','ratio','phenotype','sample']
                sample_attention = sample_attention.loc[:,['population','phenotype','model','ratio','sample', 'from', 'to', 'value']]
                sample_attention = pd.concat([attention_total, sample_attention],axis=0)
                
                attention_total = pd.concat([attention_total, sample_attention], axis=0)

        # Weight optimisation 
        if W_OPT is not None and (type(sample.loc[i,'ratio']) is tuple or SCENARIO=='between'):            
            for kk in range(len(W_OPT)):
                if W_OPT[kk]  == 'Linear transformation':
                    record, effect, predicted_test_sample, predicted_valid_sample, predicted_train_sample, weight = Linear_transformation(result_train_sample, result_valid_sample, result_test_sample, record, effect, weight, MODEL, HYPERPARAMETERS_OPT['Linear transformation'])
                elif W_OPT[kk] == 'Nelder Mead':
                    record, effect, predicted_test_sample, predicted_valid_sample, predicted_train_sample, weight = Nelder_Mead(result_train_sample, result_valid_sample, result_test_sample, record, effect, weight, MODEL, HYPERPARAMETERS_OPT['Nelder Mead'])
                elif W_OPT[kk] == 'Bayesian optimisation':
                    record, effect, predicted_test_sample, predicted_valid_sample, predicted_train_sample, weight = Bayesian(result_train_sample, result_valid_sample, result_test_sample, record, effect, weight, MODEL, HYPERPARAMETERS_OPT['Bayesian optimisation'])
            
        result_test = pd.concat([result_test, result_test_sample],axis=0)
        result_valid = pd.concat([result_valid, result_valid_sample],axis=0)
        result_train = pd.concat([result_train, result_train_sample],axis=0)
        
        result_train = result_train.sort_values(['id']).reset_index(drop=True)
        
        if type(sample.loc[i,'ratio']) is tuple:
            result_valid = result_valid.sort_values(['id']).reset_index(drop=True)
        result_test = result_test.sort_values(['id']).reset_index(drop=True)
        
    # Run the ensemble model in the end
    if 'ensemble' in MODEL:
        result_train, result_valid, result_test, sample_record, sample_effect = ensemble(result_train, result_valid, result_test, effect, MODEL) 
        record = pd.concat([record, sample_record])
        effect = pd.concat([effect, sample_effect])
    
    # Store the results
    if PARALLEL is None:
        record.to_csv('./Result/'+RESULT_NAME+'/Metric.csv', index=False)
        result_train.to_csv('./Result/'+RESULT_NAME+'/Prediction_result_train.csv', index=False)
        result_valid.to_csv('./Result/'+RESULT_NAME+'/Prediction_result_valid.csv', index=False)
        result_test.to_csv('./Result/'+RESULT_NAME+'/Prediction_result_test.csv', index=False)
    else:
        record.to_csv('./Result/'+RESULT_NAME+'/Metric_'+str(idx)+'.csv', index=False)
        result_train.to_csv('./Result/'+RESULT_NAME+'/Prediction_result_train_'+str(idx)+'.csv', index=False)
        result_valid.to_csv('./Result/'+RESULT_NAME+'/Prediction_result_valid_'+str(idx)+'.csv', index=False)
        result_test.to_csv('./Result/'+RESULT_NAME+'/Prediction_result_test_'+str(idx)+'.csv', index=False)
    
    if effect.shape[0] != 0:
        if PARALLEL is None:
            effect.to_csv('./Result/'+RESULT_NAME+'/Marker_effect.csv', index=False)
        else:
            effect.to_csv('./Result/'+RESULT_NAME+'/Marker_effect_'+str(idx)+'.csv', index=False)
    if 'RF' in MODEL:
        if PARALLEL is None:
            interactions.to_csv('./Result/'+RESULT_NAME+'/Interaction.csv', index=False)
        else:
            interactions.to_csv('./Result/'+RESULT_NAME+'/Interaction_'+str(idx)+'.csv', index=False)
    if 'GAT_fully_connected' in MODEL or 'GAT_prior_knowledge' in MODEL:
        if PARALLEL is None:
            attention_total.to_csv('./Result/'+RESULT_NAME+'/Attention.csv', index=False)
        else:
            attention_total.to_csv('./Result/'+RESULT_NAME+'/Attention_'+str(idx)+'.csv', index=False)
    if W_OPT is not None:
        if PARALLEL is None:
            weight.to_csv('./Result/'+RESULT_NAME+'/Weight.csv', index=False)
        else:
            weight.to_csv('./Result/'+RESULT_NAME+'/Weight_'+str(idx)+'.csv', index=False)
        
    return record, result_train, result_test, effect, interactions, POPULATION, PHENOTYPE, attention_total
