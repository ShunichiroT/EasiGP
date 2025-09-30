import pandas as pd
#from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import r2_score, mean_squared_error
from torch.nn import Linear, Module, Dropout
import torch.nn.functional as F
import scipy.stats
import shap
import os


class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, data):
        # store the inputs and outputs
        X = data.iloc[:,:-1]
        y = data.iloc[:,-1]
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).reshape(-1, 1)
        
    def __len__(self):
        return len(self.X)
 
    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

class MLP(Module):
    # define model elements
    def __init__(self, n_inputs, neurons, dout):
        super(MLP, self).__init__()

        self.hidden1 = Linear(n_inputs, neurons)
        self.dropout = Dropout(dout) 
        self.hidden2 = Linear(neurons,1)
 
    # forward propaMLPe input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.dropout(X)
        X = F.relu(X)
        X = self.hidden2(X)

        return X
    

def ML_Perceptron(train, valid, test, params):
    
    neurons = int(params[0])
    dout = params[1]
    lrate = params[2]
    decay = params[3]
    ep = int(params[4])
    bsize = int(params[5])
    shapley_num = params[6]
    
    train_data = CSVDataset(train)
    if valid.shape[0] != 0:
        valid_data = CSVDataset(valid)
    test_data = CSVDataset(test)
                     
    train_loader = DataLoader(train_data, batch_size=bsize, shuffle=True)
    if valid.shape[0] != 0:
        valid_loader = DataLoader(valid_data, batch_size=bsize, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=bsize, shuffle=False)
    
    model = MLP(train.shape[1]-1, neurons, dout)
   
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate, weight_decay=decay) #0.0005
   
    for epoch in range(ep):
         loss_train_sum = 0
         for inputs, targets in train_loader:
             optimizer.zero_grad()
             #inputs = inputs.to(device)
             yhat = model(inputs)
             loss = F.mse_loss(yhat, targets)
             loss_train_sum += loss
             loss.backward()
             optimizer.step()
         print(f'Epoch {epoch:>3} | Train Loss: {loss_train_sum/len(train_loader):.5f}')
         
    predicted = []
    actuals = []
    for inputs, targets in test_loader:
         #inputs = inputs.to(device)
         yhat = model(inputs)
         yhat = yhat.detach().tolist()
         actual = targets.detach().tolist()
         predicted.append([item for sublist in yhat for item in sublist])
         actuals.append([item for sublist in actual for item in sublist])
    predicted = [item for sublist in predicted for item in sublist]
    actuals = [item for sublist in actuals for item in sublist]
     
    mse = mean_squared_error(actuals, predicted)
    r = scipy.stats.pearsonr(actuals, predicted)[0]
    
    train_loader = DataLoader(train_data, batch_size=bsize, shuffle=False)
    predicted_train = []
    for inputs, targets in train_loader:
         yhat = model(inputs)
         yhat = yhat.detach().tolist()
         predicted_train.append([item for sublist in yhat for item in sublist])
    predicted_train = [item for sublist in predicted_train for item in sublist]
    
    predicted_valid = []
    if valid.shape[0] != 0:
        for inputs, targets in valid_loader:
             yhat = model(inputs)
             yhat = yhat.detach().tolist()
             predicted_valid.append([item for sublist in yhat for item in sublist])
        predicted_valid = [item for sublist in predicted_valid for item in sublist] 
    
    d_train = torch.tensor(train.iloc[:,:-1].values, dtype=torch.float32)
    #d_valid = torch.tensor(valid.iloc[:,:-1].values, dtype=torch.float32)
    d_test = torch.tensor(test.iloc[:,:-1].values, dtype=torch.float32)
    
    explainer = shap.DeepExplainer(model,shap.sample(d_train, shapley_num))
    effect = abs(explainer.shap_values(shap.sample(d_test, shapley_num),check_additivity=False)).sum(axis=0)

    return r, mse, pd.DataFrame(effect).T, predicted, predicted_valid, predicted_train