import pandas as pd
import numpy as np
import scipy.stats
from sklearn.metrics import mean_squared_error
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer#,UtilityFunction
from bayes_opt.util import ensure_rng
from bayes_opt.parameter import BayesParameter

## Customise inout parameters (weights) for the Bayesian optimisation
class ParameterSetting(BayesParameter):
    def __init__(self, name: str, bounds, dimension) -> None:
        super().__init__(name, bounds)
        self.dimension = dimension

    @property
    def is_continuous(self):
        return True

    def random_sample(self, n_samples: int, random_state):
        random_state = ensure_rng(random_state)
        samples = []
        while len(samples) < n_samples:
            samples_ = random_state.dirichlet(np.ones(self.dimension), n_samples)
            samples_ = samples_[np.all((self.bounds[:, 0] <= samples_) & (samples_ <= self.bounds[:, 1]), axis=-1)]
            samples.extend(np.atleast_2d(samples_))
        samples = np.array(samples[:n_samples])
        return samples

    def to_float(self, value):
        return value

    def to_param(self, value):
        return value 

    def kernel_transform(self, value):
        return value

    def to_string(self, value, str_len: int) -> str:
        str_ = '|'.join([f"{float(np.round(value[i], 6)):.5f}"[:str_len] for i in range(self.dimension)])
        return str_.ljust(str_len)

    @property
    def dim(self):
        return self.dimension 

def Bayesian(data_train, data_valid, data_test, record, effect, weight, MODEL, HPARAMETERS_OPT):
    
    minimum_boundary = HPARAMETERS_OPT[0]
    maximum_boundary = HPARAMETERS_OPT[1]
    iteration = HPARAMETERS_OPT[2]
    point_num = HPARAMETERS_OPT[3]
    duplicate_points = HPARAMETERS_OPT[4]
 
    model_selected = MODEL.copy()
    if 'ensemble' in MODEL:
        model_selected.remove('ensemble')

    ## Define the objective function to maximise based on the Diversity Prediction Theorem
    def optimisation(weights):
       
       second_term = data_valid.loc[:,model_selected+['actual']]
       third_term, mean_value = second_term.copy(), second_term.iloc[:,:-1].mean(axis=1)
       
       for i in range(len(model_selected)):
           second_term.iloc[:,i] = (second_term.iloc[:,i]*weights[i] - second_term['actual']).pow(2)
           third_term.iloc[:,i] = (third_term.iloc[:,i]*weights[i] - mean_value).pow(2)
           
       second_term = second_term.iloc[:,:-1].mean(axis=1)
       third_term = third_term.iloc[:,:-1].mean(axis=1)
       first_term = (second_term - third_term).mean()
        
       return 1/first_term

    ## Define the parameters for the Bayesian optimisation
    #bounds_transformer = SequentialDomainReductionTransformer(minimum_window=0.2)
    
    param = ParameterSetting(
        name='weights',
        bounds=np.array([[minimum_boundary, maximum_boundary]]*len(model_selected)),
        dimension=len(model_selected)
    )
    
    pbounds = {'weights': param}
    optimizer = BayesianOptimization(
        optimisation,
        pbounds,
        random_state=1,
        allow_duplicate_points=duplicate_points,
        #bounds_transformer=bounds_transformer
    )

    #acquisition_function = UtilityFunction(kind="ei", kappa=1e-6) 

    ## Implement the Bayesian optimisation method
    optimizer.maximize(
        init_points= point_num, 
        n_iter=iteration,
        #acquisition_function=acquisition_function
    )
    
    best_target, best_first_weight = optimizer.max['target'], optimizer.max['params']['weights']
    res = optimizer.res
    for i in range(len(res)):
        if res[i]['target'] == best_target and res[i]['params']['weights'][0] == best_first_weight:
            w = res[i]['params']['weights']
    
    ## Weight extraction
    weight_extracted = pd.DataFrame(w).T
    weight_extracted.columns = model_selected 
    
    w =  [x / sum(w) for x in [abs(x) for x in w]]
    
    ## Weight the predicted phenotypes for the test set
    data_test_selected = data_test.loc[:,model_selected+['actual']]
    for i in range(len(data_test_selected.columns)-1):
        data_test_selected.iloc[:,i] = data_test_selected.iloc[:,i] * w[i]
    predicted_test =  data_test_selected.iloc[:,:-1].sum(axis=1).reset_index(drop=True)
    actual_test =  data_test_selected.iloc[:,-1].tolist()
    
    ## Calculate the metrics
    mse = mean_squared_error(actual_test, predicted_test)
    r = scipy.stats.pearsonr(actual_test, predicted_test)[0]
    
    ## Store the metrics
    record = pd.concat([record, pd.DataFrame(record.iloc[-1,:]).T]).reset_index(drop=True)
    record.loc[record.shape[0]-1,'type'] = 'Bayesian optimisation'
    record.loc[record.shape[0]-1,'Pearson correlation'] = r
    record.loc[record.shape[0]-1,'MSE'] = mse

    ## Weight the predicted phenotypes for the validation set
    predicted_valid = []
    data_valid_selected = data_valid.loc[:,model_selected+['actual']]
    for i in range(len(data_valid_selected.columns)-1):
        data_valid_selected.iloc[:,i] = data_valid_selected.iloc[:,i] * w[i]
    predicted_valid =  data_valid_selected.iloc[:,:-1].sum(axis=1).reset_index(drop=True)
    #actual_valid =  data_valid_selected.iloc[:,-1].tolist()
    
    ## Weight the predicted phenotypes for the training set
    predicted_train = []
    data_train_selected = data_train.loc[:,model_selected+['actual']]
    for i in range(len(data_train_selected.columns)-1):
        data_train_selected.iloc[:,i] = data_train_selected.iloc[:,i] * w[i]
    predicted_train =  data_train_selected.iloc[:,:-1].sum(axis=1).reset_index(drop=True)
    #actual_train =  data_train_selected.iloc[:,-1].tolist()
    
    data_test['Bayesian'] = predicted_test
    data_valid['Bayesian'] = predicted_valid
    data_train['Bayesian'] = predicted_train

    ## Extract weights
    weight_sample = pd.DataFrame(record.iloc[record.shape[0]-1,:]).T.drop(['Pearson correlation', 'MSE'],axis=1)
    weight_sample['type'] = 'Bayesian optimisation' 
    
    weight_sample = pd.concat([weight_sample.reset_index(drop=True), weight_extracted.reset_index(drop=True)], axis=1)
    weight = pd.concat([weight, weight_sample])
    
    weight_extracted_normalised = weight_extracted.div(weight_extracted.sum(axis=1),axis=0)
    
    ## Calculate weighted effects
    for i in range(len(model_selected)):
        if effect[effect['type']==model_selected[i]].shape[0] != 0:
            if i == 0:
                effect_weighted = effect[effect['type']==model_selected[i]].tail(1).iloc[:,5:].abs().reset_index(drop=True).div(effect[effect['type']==model_selected[i]].tail(1).iloc[:,5:].abs().sum(axis=1).reset_index(drop=True), axis=0).mul(weight_extracted_normalised[model_selected[i]], axis=0).reset_index(drop=True)
            else:
                effect_weighted += effect[effect['type']==model_selected[i]].tail(1).iloc[:,5:].abs().reset_index(drop=True).div(effect[effect['type']==model_selected[i]].tail(1).iloc[:,5:].abs().sum(axis=1).reset_index(drop=True), axis=0).mul(weight_extracted_normalised[model_selected[i]], axis=0).reset_index(drop=True)
 
    effect = pd.concat([effect, pd.DataFrame(effect.iloc[effect.shape[0]-1,:]).T]).reset_index(drop=True)
    effect.loc[effect.shape[0]-1,'type'] = 'Bayesian optimisation'
    
    effect.loc[effect.shape[0]-1,list(effect_weighted.columns)] = effect_weighted.values

    return record, effect, data_test, data_valid, data_train, weight