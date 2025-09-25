import pandas as pd
import torch
import shap
import torch.nn.functional as F
import copy
import scipy.stats
from torch.utils.data import Dataset, DataLoader
from torch.nn import Linear, Module
from sklearn.metrics import mean_squared_error

## Define the class converting the data into a specific format
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, data):
        # store the inputs and outputs
        X = data.iloc[:,1:]
        y = data.iloc[:,0]
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).reshape(-1, 1)
        
    def __len__(self):
        return len(self.X)
 
    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]
    
class MLP_linear(Module):
    def __init__(self, n_inputs):
        super(MLP_linear, self).__init__()
        self.hidden1 = Linear(n_inputs,1, bias=False)
 
    def forward(self, X):
        X = self.hidden1(X)
        return X

def Linear_transformation(data_train, data_valid, data_test, record, effect, weight, MODEL, HPARAMETERS_OPT):
    
    # Hyperparameter setting
    learning_rate = HPARAMETERS_OPT[0]
    epochs = HPARAMETERS_OPT[1]
    decay = HPARAMETERS_OPT[2]
    batch_s = HPARAMETERS_OPT[3]
    pat = HPARAMETERS_OPT[4]
    num_models = HPARAMETERS_OPT[5]
    
    model_selected = MODEL.copy()
    if 'ensemble' in MODEL:
        model_selected.remove('ensemble')

    ## Change the data structure
    train = CSVDataset(data_train.loc[:,['actual']+model_selected])
    valid = CSVDataset(data_valid.loc[:,['actual']+model_selected])
    test = CSVDataset(data_test.loc[:,['actual']+model_selected])

    ## Create data loaders
    train_loader = DataLoader(train, batch_size=batch_s, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=batch_s, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_s, shuffle=False)
    
    ## Implement weight optimisation 
    best_loss_final = float('inf')
    best_model_weights = None
    for i in range(num_models):
        model = MLP_linear(data_train.loc[:,model_selected].shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
        patience = pat
        best_loss = float('inf')  
        
        ## Each MLP is trained epoch times 
        for epoch in range(epochs):
            loss_train_sum = loss_valid_sum = 0
            batch_size = len(train_loader)
            batch_size_valid = len(valid_loader)
            
            ## MLP training
            for inputs, targets in train_loader:
                model.train()
                optimizer.zero_grad()
                yhat = model(inputs)
                loss = F.mse_loss(yhat, targets)
                loss_train_sum += loss
                loss.backward()
                optimizer.step()
            
            ## MLP validation
            for inputs, targets in valid_loader:
                model.eval()
                with torch.no_grad(): 
                    yhat = model(inputs)
                    loss = F.mse_loss(yhat, targets)                            
                
                loss_valid_sum += loss 
            
            ## Evaluate MLP per epoch & store the best one with the early stopping
            if (loss_valid_sum/batch_size_valid) < best_loss:
                patience = pat
                best_loss = loss_valid_sum/batch_size_valid
                if best_loss_final > best_loss:
                    best_loss_final = best_loss
                    best_model_weights = copy.deepcopy(model.state_dict())   
            else:
                patience -= 1
                if patience == 0:
                    break
                
            print(f'Epoch {epoch:>3} | Train Loss: {loss_train_sum/batch_size:.5f} | Valid Loss: {loss_valid_sum/batch_size_valid:.5f}')
        
    ## Load the trained weights of the best model
    model.load_state_dict(best_model_weights)
    
    ## Evaluate model
    model.eval()
    
    predicted_test = []
    actual_test = []
    for inputs, targets in test_loader:
        yhat = model(inputs)    
        predicted_test += yhat.tolist()
        actual_test += targets.tolist()
    predicted_test = [item for sublist in predicted_test for item in sublist]
    actual_test = [item for sublist in actual_test for item in sublist]
    
    ## Calculate metrics
    mse = mean_squared_error(actual_test, predicted_test)
    r = scipy.stats.pearsonr(actual_test, predicted_test)[0]
    
    ## Store the metrics
    record = pd.concat([record, pd.DataFrame(record.iloc[-1,:]).T]).reset_index(drop=True)
    record.loc[record.shape[0]-1,'type'] = 'Linear transformation'
    record.loc[record.shape[0]-1,'Pearson correlation'] = r
    record.loc[record.shape[0]-1,'MSE'] = mse

    ## Extract predicted and observed values for the validation set
    predicted_valid = []
    #actual_valid = []
    valid_loader = DataLoader(valid, batch_size=batch_s, shuffle=False)
    for inputs, targets in valid_loader:
        yhat = model(inputs)    
        predicted_valid += yhat.tolist()
        #actual_valid += targets.tolist()
    predicted_valid = [item for sublist in predicted_valid for item in sublist]
    #actual_valid = [item for sublist in actual_valid for item in sublist]       
    
    ## Extract predicted and observed values for the train set
    train_loader = DataLoader(train, batch_size=batch_s, shuffle=False)
    predicted_train = []
    #actual_train = []
    for inputs, targets in train_loader:
        yhat = model(inputs)    
        predicted_train += yhat.tolist()
        #actual_train += targets.tolist()
    predicted_train = [item for sublist in predicted_train for item in sublist]
    #actual_train = [item for sublist in actual_train for item in sublist]                 
    
    data_test['Linear transformation'] = predicted_test
    data_valid['Linear transformation'] = predicted_valid
    data_train['Linear transformation'] = predicted_train
    
    ## Extract weights
    data_train_selected, data_test_selected = data_train.loc[:,model_selected], data_test.loc[:,model_selected]
    
    d_train = torch.tensor(data_train_selected.values, dtype=torch.float32)
    d_test = torch.tensor(data_test_selected.values, dtype=torch.float32)

    weight_sample = pd.DataFrame(record.iloc[record.shape[0]-1,:]).T.drop(['Pearson correlation', 'MSE'],axis=1)
    weight_sample['type'] = 'Linear transformation'
    
    explainer = shap.DeepExplainer(model, shap.sample(d_train, 50))    
    weight_extracted = pd.DataFrame(abs(explainer.shap_values(shap.sample(d_test, 50),check_additivity=False)).sum(axis=0)).T
    weight_extracted = pd.DataFrame(weight_extracted)
    weight_extracted.columns = model_selected    
    weight_sample = pd.concat([weight_sample.reset_index(drop=True), weight_extracted.reset_index(drop=True)], axis=1)
    weight = pd.concat([weight, weight_sample])
    
    ## Calculate weighted effects    
    weight_extracted_normalised = weight_extracted.div(weight_extracted.sum(axis=1),axis=0)
    
    for i in range(len(model_selected)):
        if effect[effect['type']==model_selected[i]].shape[0] != 0:
            if i == 0:
                effect_weighted = effect[effect['type']==model_selected[i]].tail(1).iloc[:,5:].abs().reset_index(drop=True).div(effect[effect['type']==model_selected[i]].tail(1).iloc[:,5:].abs().sum(axis=1).reset_index(drop=True), axis=0).mul(weight_extracted_normalised[model_selected[i]], axis=0).reset_index(drop=True)
            else:
                effect_weighted += effect[effect['type']==model_selected[i]].tail(1).iloc[:,5:].abs().reset_index(drop=True).div(effect[effect['type']==model_selected[i]].tail(1).iloc[:,5:].abs().sum(axis=1).reset_index(drop=True), axis=0).mul(weight_extracted_normalised[model_selected[i]], axis=0).reset_index(drop=True)

    effect = pd.concat([effect, pd.DataFrame(effect.iloc[effect.shape[0]-1,:]).T]).reset_index(drop=True)
    effect.loc[effect.shape[0]-1,'type'] = 'Linear transformation'

    effect.loc[effect.shape[0]-1,list(effect_weighted.columns)] = effect_weighted.values

    return record, effect, data_test, data_valid, data_train, weight
