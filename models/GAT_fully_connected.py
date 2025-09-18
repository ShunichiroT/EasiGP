import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.explain import Explainer, CaptumExplainer


def GAT_fully_connected(data_train, data_test, params):
 
    neuron = params[0]
    dropout = params[1]
    lrate = params[2]
    decay = params[3]
    epoch = params[4]
    bsize = params[5]
    heads = params[6]
    samples = params[7]
    marker_effect = params[8]
    
    ## Preprocess the data so that it caon be converted into a graph format    
    data_QTL_train, data_QTL_test = data_train.iloc[:,:-1].reset_index(drop=True),data_test.iloc[:,:-1].reset_index(drop=True)
    data_QTL_melt_train, data_QTL_melt_test = data_QTL_train.T.melt(), data_QTL_test.T.melt()
    data_pheno_train, data_pheno_test = data_train.iloc[:,-1].reset_index(drop=True), data_test.iloc[:,-1].reset_index(drop=True)     

    ## Change the data structure to create graphs
    edges_from_train = [ele for ele in range(data_QTL_melt_train.shape[0]) for i in range(data_QTL_train.shape[1])]
    edges_to_train = []
    for ii in range(0,int(len(edges_from_train)/data_QTL_train.shape[1]),data_QTL_train.shape[1]):
        for kk in range(data_QTL_train.shape[1]):
            for jj in range(ii,ii+data_QTL_train.shape[1]):
                edges_to_train += [jj]
    edges_from_train, edges_to_train = np.array(edges_from_train), np.array(edges_to_train)
    
    edges_from_test = [ele for ele in range(data_QTL_melt_test.shape[0]) for i in range(data_QTL_test.shape[1])]
    edges_to_test = []
    for ii in range(0,int(len(edges_from_test)/data_QTL_test.shape[1]),data_QTL_test.shape[1]):
        for kk in range(data_QTL_test.shape[1]):
            for jj in range(ii,ii+data_QTL_test.shape[1]):
                edges_to_test += [jj]
    edges_from_test, edges_to_test = np.array(edges_from_test), np.array(edges_to_test)
    
    dummy = pd.get_dummies(pd.DataFrame(list(range(data_QTL_train.shape[1]))), columns=[0])
    dummy[dummy==False]=0
    dummy[dummy==True]=1
    data_QTL_melt_train = pd.concat([data_QTL_melt_train,pd.concat([dummy]*int(data_QTL_melt_train.shape[0]/dummy.shape[0])).reset_index(drop=True)],axis=1)
    data_QTL_melt_test = pd.concat([data_QTL_melt_test,pd.concat([dummy]*int(data_QTL_melt_test.shape[0]/dummy.shape[0])).reset_index(drop=True)],axis=1)
 
    ## Create graphs
    data_train = []
    for kk in range(data_pheno_train.shape[0]):
        tmp = Data()
        data_QTL_melt_train_tmp = data_QTL_melt_train.loc[data_QTL_melt_train['variable']==kk].iloc[:,1:]#.reset_index(drop=True)
        data_pheno_train_tmp = np.expand_dims(np.array(data_pheno_train[kk]),axis=0)
        edges_from_train_tmp = edges_from_train[:data_QTL_train.shape[1]*data_QTL_train.shape[1]]
        edges_to_train_tmp = edges_to_train[:data_QTL_train.shape[1]*data_QTL_train.shape[1]]
        tmp.x = torch.from_numpy(data_QTL_melt_train_tmp.to_numpy(dtype=float)).to(torch.float)
        tmp.y = torch.from_numpy(data_pheno_train_tmp).to(torch.float)
        tmp.edge_index = torch.stack([torch.from_numpy(edges_from_train_tmp).to(torch.long),torch.from_numpy(edges_to_train_tmp).to(torch.long)], dim=0)
        data_train += [tmp]
    
    data_test = []
    for kk in range(data_pheno_test.shape[0]):
        tmp = Data()
        data_QTL_melt_test_tmp = data_QTL_melt_test.loc[data_QTL_melt_test['variable']==kk].iloc[:,1:]#.reset_index(drop=True)
        data_pheno_test_tmp = np.expand_dims(np.array(data_pheno_test[kk]),axis=0)
        edges_from_test_tmp = edges_from_test[:data_QTL_test.shape[1]*data_QTL_test.shape[1]]
        edges_to_test_tmp = edges_to_test[:data_QTL_test.shape[1]*data_QTL_test.shape[1]]
        tmp.x = torch.from_numpy(data_QTL_melt_test_tmp.to_numpy(dtype=float)).to(torch.float)
        tmp.y = torch.from_numpy(data_pheno_test_tmp).to(torch.float)
        tmp.edge_index = torch.stack([torch.from_numpy(edges_from_test_tmp).to(torch.long),torch.from_numpy(edges_to_test_tmp).to(torch.long)], dim=0)
        data_test += [tmp]
        
    ## Create GAT    
    class GAT(torch.nn.Module):
        def __init__(self, hidden_channels, out_channels, dpout):
            super().__init__()
            
            self.conv1 = GATv2Conv((-1,-1), hidden_channels, add_self_loops=False, heads=heads, concat=True, dropout=dpout)
            self.conv2 = GATv2Conv((-1,-1), hidden_channels, add_self_loops=False, heads=heads, concat=False, dropout=dpout)
            self.lin1 = torch.nn.Linear(hidden_channels, out_channels)
    
        def forward(self, x,edge_index,batch):
            x, edge_index, batch = x, edge_index, batch
            x = self.conv1(x, edge_index)
            x = F.elu(x)
            x = self.conv2(x, edge_index)
            x = F.elu(x)
            x = global_mean_pool(x, batch)
            x = self.lin1(x)
    
            return x
                
    model = GAT(hidden_channels=neuron, out_channels=1, dpout=dropout)
    
    train_loader = DataLoader(data_train, 
                             shuffle=True,
                             batch_size=bsize)
    
    test_loader = DataLoader(data_test, 
                             batch_size=bsize)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    ## Train GAT
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate, weight_decay=decay)
    
    for epoch in range(epoch): 
        loss_train_sum = 0
        batch_size = len(train_loader)
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x,batch.edge_index,batch.batch)
            loss = F.mse_loss(torch.squeeze(out), batch.y)                            
            loss.backward()
            optimizer.step()
            loss_train_sum += loss 
        
        print(f'Epoch {epoch:>3} | Train Loss: {loss_train_sum/batch_size:.5f}')
    
    ## Predict phenotypes for the test data
    model.eval()
        
    predicted_test = []
    actual_test = []
    for test in test_loader:
        result = model(test.x,test.edge_index,test.batch)
        predicted_test += result.tolist()
        actual_test += test.y.tolist()
               
    predicted_test = [item for sublist in predicted_test for item in sublist]     
    
    ## Calculate the metrics
    mse = mean_squared_error(actual_test,predicted_test)
    r = pearsonr(actual_test, predicted_test)[0]
    
    ## Predict phenotypes for the train data
    train_loader = DataLoader(data_train, 
                             shuffle=False,
                             batch_size=bsize)
    predicted_train = []
    #actual_train = []
    for train in train_loader:
        result = model(train.x,train.edge_index,train.batch)
        predicted_train += result.tolist()
        #actual_train += train.y.tolist()
    
    predicted_train = [k for i in predicted_train for k in i]

    ## Extract genomic marker effects
    if marker_effect == True:
        explainer = Explainer(
            model = model,
            algorithm=CaptumExplainer('IntegratedGradients'),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type=None, # do not change here
            model_config = dict(
                mode='regression',
                task_level='node',
                return_type='raw',
                ),
        )
        
        train_loader = DataLoader(data_train, 
                                shuffle=True,
                                batch_size=1)
        
        explanation = pd.DataFrame()
        cnt = 0
        for batch in train_loader:
            t = explainer(
                batch.x,
                batch.edge_index,
                batch=batch.batch
            )
            t = pd.DataFrame(t['node_mask'].squeeze().detach()).sum(axis=1)
            if explanation.shape[0] == 0:
                explanation = t
            else:
                explanation += t
            cnt += 1
            
            if cnt == samples:
                break
        
        effect = pd.DataFrame(explanation/cnt).T
        effect.columns = list(data_QTL_train.columns)
    else:
        effect = pd.DataFrame()
        
    return r, mse, effect, predicted_test, predicted_train