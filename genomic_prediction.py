import pandas as pd
import os
from sklearn.model_selection import train_test_split

from models.RF import *
from models.SVR import *
from models.MLP import *
from models.GAT import *
from models.ensemble import *

def metric_plot(record, MODEL):
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    metrics = ['Pearson correlation', 'MSE']
    record['models'] = 'models'
    sns.set_theme(style="ticks",font_scale = 1, rc={"figure.dpi":300, 'savefig.dpi':300})
    
    for i in range(len(metrics)):
        ax_share = False if metrics[i] == 'MSE' else True
        
        g = sns.FacetGrid(record, col="phenotype", row='population', sharey=ax_share)
        
        for axis in g.axes.flat:
            axis.tick_params(labelleft=True) 
                    
        g.map(sns.violinplot,'models', metrics[i], 'type', palette='colorblind', hue_order=MODEL)    
        
        for axis in g.axes.flat:
            axis.set_ylabel(metrics[i])  
            axis.set_xlabel("")  
        
        plt.tight_layout()
        g.add_legend()
        g.savefig("./Result/"+metrics[i]+".png") 

    
def GP(DATA_NAME, MODEL, PHENOTYPE, POPULATION, RATIO, SAMPLE_NUM, HPARAMETERS, TOTAL_PHENOTYPE, R_PATH):
    
    # Import R modules
    if R_PATH != None:
        os.environ['R_HOME'] = R_PATH
    
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    
    r_source = robjects.r['source']
    r_source('./models/rrBLUP.R')
    rrBLUP = robjects.globalenv['rrBLUP']
    r_source('./models/BayesB.R')
    BayesB = robjects.globalenv['BayesB']
    r_source('./models/RKHS.R')
    RKHS = robjects.globalenv['RKHS']
    
    # Read genotype and phenotype data
    data_original = pd.read_csv(DATA_NAME+'.csv')
    
    # Create the total number of combinations of prediction scenarios
    sample = pd.DataFrame({'population':[item for item in POPULATION for i in range(SAMPLE_NUM*len(PHENOTYPE)*len(RATIO))],
                            'phenotype':[item for item in PHENOTYPE for i in range(SAMPLE_NUM*len(RATIO))]*len(POPULATION),
                            'ratio':[item for item in RATIO for i in range(SAMPLE_NUM)]*len(POPULATION)*len(PHENOTYPE),
                            'sample':list(range(1,SAMPLE_NUM+1)) *len(PHENOTYPE)*len(POPULATION)*len(RATIO)})
        
    record = pd.DataFrame()  #store performance metrics
    result_train = pd.DataFrame() #store predicted phenotypes for train set
    result_test = pd.DataFrame() #store predicted phenotypes for test set
    effect = pd.DataFrame() #store genomic marker effects
    interactions = pd.DataFrame() #store marker interaction effects

    for i in range(sample.shape[0]):
        # Convert the data structure
        data = pd.concat([data.iloc[:,:-TOTAL_PHENOTYPE], data[sample.loc[i,'phenotype']]],axis=1).dropna()
        train, test = train_test_split(data,train_size=sample.loc[i,'ratio'], random_state=sample.loc[i,'sample'])
        train, test = train.reset_index(drop=True), test.reset_index(drop=True)
        id_train, id_test = train.iloc[:,0], test.iloc[:,0]
        train, test = train.iloc[:,2:], test.iloc[:,2:]
        
        result_test_sample = pd.DataFrame()
        result_train_sample = pd.DataFrame()
        
        # Prediction model implementation
        for jj in range(len(MODEL)):
            if MODEL[jj] == 'ensemble':
                continue
            elif MODEL[jj] == 'rrBLUP':
                sample_pearson, sample_mse, sample_effect, predicted_test, predicted_train = rrBLUP(train, test, HPARAMETERS[MODEL[jj]])
                sample_pearson, sample_mse = sample_pearson[0], sample_mse[0]
                sample_effect = robjects.conversion.rpy2py(sample_effect)
                predicted_test = robjects.conversion.rpy2py(predicted_test)
                predicted_train = robjects.conversion.rpy2py(predicted_train)
            elif MODEL[jj]  == 'BayesB':
                sample_pearson, sample_mse, sample_effect, predicted_test, predicted_train = BayesB(train, test, HPARAMETERS[MODEL[jj]])
                sample_pearson, sample_mse = sample_pearson[0], sample_mse[0]
                sample_effect = robjects.conversion.rpy2py(sample_effect)
                predicted_test = robjects.conversion.rpy2py(predicted_test)
                predicted_train = robjects.conversion.rpy2py(predicted_train)
            elif MODEL[jj]  == 'RKHS':
                if HPARAMETERS[MODEL[jj]][-2] == 'all':
                    HPARAMETERS[MODEL[jj]][-2] = test.shape[0]       
                sample_pearson, sample_mse, sample_effect, predicted_test, predicted_train = RKHS(train, test, HPARAMETERS[MODEL[jj]])
                sample_pearson, sample_mse = sample_pearson[0], sample_mse[0]
                sample_effect = robjects.conversion.rpy2py(sample_effect)
                predicted_test = robjects.conversion.rpy2py(predicted_test)
                predicted_train = robjects.conversion.rpy2py(predicted_train)
            elif MODEL[jj]  == 'RF':
                if HPARAMETERS[MODEL[jj]][-2] == 'all':
                    HPARAMETERS[MODEL[jj]][-2] = test.shape[0] 
                sample_pearson, sample_mse, sample_effect, sample_interaction, predicted_test, predicted_train = RF(train, test, HPARAMETERS[MODEL[jj]])
            elif MODEL[jj]  == 'SVR':
                if HPARAMETERS[MODEL[jj]][-2] == 'all':
                    HPARAMETERS[MODEL[jj]][-2] = test.shape[0] 
                sample_pearson, sample_mse, sample_effect, predicted_test, predicted_train = SV_Regression(train, test, HPARAMETERS[MODEL[jj]])
            elif MODEL[jj]  == 'MLP':
                if HPARAMETERS[MODEL[jj]][-1] == 'all':
                    HPARAMETERS[MODEL[jj]][-1] = test.shape[0] 
                sample_pearson, sample_mse, sample_effect, predicted_test, predicted_train = ML_Perceptron(train, test, HPARAMETERS[MODEL[jj]])
            elif MODEL[jj]  == 'GAT':
                if HPARAMETERS[MODEL[jj]][-1] == 'all':
                    HPARAMETERS[MODEL[jj]][-1] = test.shape[0] 
                sample_pearson, sample_mse, sample_effect, predicted_test, predicted_train = GAT_model(train, test, HPARAMETERS[MODEL[jj]])
            
            # Store prediction results
            record_sample = pd.DataFrame([{'population': sample.loc[i,'population'],
                                           'phenotype': sample.loc[i,'phenotype'],
                                           'type': MODEL[jj],
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
                                               'type': MODEL[jj],
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
                sample_interaction['type'] = MODEL[jj]
                sample_interaction['ratio'] = sample.loc[i,'ratio']
                sample_interaction['sample'] = sample.loc[i,'sample'] 
                
                interactions = pd.concat([interactions, sample_interaction])     
    
        result_test = pd.concat([result_test, result_test_sample],axis=0)
        result_train = pd.concat([result_train, result_train_sample],axis=0)
        
        result_train = result_train.sort_values(['id']).reset_index(drop=True)
        result_test = result_test.sort_values(['id']).reset_index(drop=True)
        
    # Run the ensemble model in the end
    if 'ensemble' in MODEL:
        result_train, result_test, sample_record, sample_effect = ensemble(result_train, result_test, effect, MODEL) 
        record = pd.concat([record, sample_record])
        effect = pd.concat([effect, sample_effect])
    
    # Store the results
    record.to_csv('./Result/Metric.csv', index=False)
    result_train.to_csv('./Result/Prediction_result_train.csv', index=False)
    result_test.to_csv('./Result/Prediction_result_test.csv', index=False)
    
    if effect.shape[0] != 0:
        effect.to_csv('./Result/Marker_effect.csv', index=False)
    if 'RF' in MODEL:
        interactions.to_csv('./Result/Interaction.csv', index=False)
    
    # Store violin plots
    metric_plot(record.copy(), MODEL)
    
    return record, result_train, result_test, effect, interactions    
    
    

