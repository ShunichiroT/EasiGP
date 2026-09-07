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

from pipeline_utils import (
    get_active_compute_resources, apply_torch_compute_settings, gpu_slot,
    dataloader_num_workers, torch_eval_batch_size, split_batched_edge_attention,
)


def GAT_fully_connected(data_train, data_valid, data_test, params):
 
    if data_valid.shape[0] != 0:
        VALID = True
    else:
        VALID = False
        
    data_test_columns = pd.Series(data_test.columns)
        
    neuron = params[0]
    dropout = params[1]
    lrate = params[2]
    decay = params[3]
    epoch = params[4]
    bsize = params[5]
    heads = params[6]
    marker_effect = params[7]
    samples = params[8]
    
    ## Preprocess the data so that it can be converted into a graph format    
    if VALID:
        data_QTL_train,data_QTL_valid, data_QTL_test = data_train.iloc[:,:-1].reset_index(drop=True), data_valid.iloc[:,:-1].reset_index(drop=True), data_test.iloc[:,:-1].reset_index(drop=True)
        data_QTL_melt_train, data_QTL_melt_valid, data_QTL_melt_test = data_QTL_train.T.melt(), data_QTL_valid.T.melt(), data_QTL_test.T.melt()
        data_pheno_train, data_pheno_valid, data_pheno_test = data_train.iloc[:,-1].reset_index(drop=True), data_valid.iloc[:,-1].reset_index(drop=True), data_test.iloc[:,-1].reset_index(drop=True)
    else:
        data_QTL_train,data_QTL_test = data_train.iloc[:,:-1].reset_index(drop=True), data_test.iloc[:,:-1].reset_index(drop=True)
        data_QTL_melt_train, data_QTL_melt_test = data_QTL_train.T.melt(), data_QTL_test.T.melt()
        data_pheno_train, data_pheno_test = data_train.iloc[:,-1].reset_index(drop=True), data_test.iloc[:,-1].reset_index(drop=True)

    ## Change the data structure to create graphs
    #
    # ver4-4 R8 (blueprint §2.8): the fully-connected marker graph is
    # IDENTICAL across every individual in a split - every marker attends
    # to every other marker, regardless of that individual's own genotype
    # values, which live on the node features (`tmp.x` below), not on the
    # edges. The ORIGINAL code below built `edges_from_train`/
    # `edges_to_train` as Python lists of length N*M^2 (N = individuals in
    # this split, M = markers) via a triple-nested loop, then every
    # per-individual Data object sliced out only the FIRST M^2 elements
    # (`edges_from_train[:M*M]`) - identical for every individual. The
    # remaining (N-1)*M^2 elements were built and then never read at all.
    # For N=200, M=1,000 that is 2x10^8 discarded list elements per
    # direction (~3.2 GB of Python object pointers) and minutes of pure
    # interpreter time, before any GPU work even starts - and is what
    # makes this model infeasible at realistic marker counts in the first
    # place (architecture doc §9.3, root-cause finding R8).
    #
    # Fix: build the O(M^2) fully-connected edge index ONCE per split,
    # vectorised (`np.repeat`/`np.tile` reproduce the exact same edge
    # ordering the discarded O(N*M^2) construction's first M^2 elements
    # produced - verified algebraically in the R8 design record and
    # covered by this update's regression fixture), and give every
    # individual's Data object its own `.clone()` of that ONE base
    # tensor. `.clone()` is deliberately kept (rather than sharing the
    # exact same tensor object across every Data instance) because
    # confirming that PyG's DataLoader collation never mutates a shared
    # `edge_index` in place for the installed torch-geometric version is
    # PC-4 in the ver4-4 blueprint's Phase-2 hand-off checklist, and that
    # pre-check could not be executed in this build environment (no GPU/
    # torch_geometric available) - see the ver4-4 Change Summary §7/§8.
    # `.clone()` still eliminates the entire O(N*M^2) construction cost
    # above (the actual, measured waste), it only forgoes the smaller,
    # additional gain of sharing one tensor object outright; per RK-11's
    # own documented mitigation, this is the safe default until PC-4 is
    # confirmed, and switching `.clone()` -> a bare shared reference is a
    # one-line follow-up once it is.
    def _fully_connected_edge_index(n_markers: int) -> torch.Tensor:
        """The (2, n_markers^2) edge_index for a fully-connected graph over
        `n_markers` nodes, self-loops included - byte-identical to what
        the original O(N*M^2) construction's first M^2 elements produced
        (edges_from[:M*M] == repeat(arange(M), M), edges_to[:M*M] ==
        tile(arange(M), M) - see the R8 design record for the derivation)."""
        edges_from = np.repeat(np.arange(n_markers), n_markers)
        edges_to = np.tile(np.arange(n_markers), n_markers)
        return torch.stack(
            [torch.from_numpy(edges_from).to(torch.long), torch.from_numpy(edges_to).to(torch.long)],
            dim=0,
        )

    edge_index_train_base = _fully_connected_edge_index(data_QTL_train.shape[1])
    if VALID:
        edge_index_valid_base = _fully_connected_edge_index(data_QTL_valid.shape[1])
    edge_index_test_base = _fully_connected_edge_index(data_QTL_test.shape[1])
    
    dummy = pd.get_dummies(pd.DataFrame(list(range(data_QTL_train.shape[1]))), columns=[0])
    dummy[dummy==False]=0
    dummy[dummy==True]=1
    data_QTL_melt_train = pd.concat([data_QTL_melt_train,pd.concat([dummy]*int(data_QTL_melt_train.shape[0]/dummy.shape[0])).reset_index(drop=True)],axis=1)
    if data_valid.shape[0] != 0:
        data_QTL_melt_valid = pd.concat([data_QTL_melt_valid,pd.concat([dummy]*int(data_QTL_melt_valid.shape[0]/dummy.shape[0])).reset_index(drop=True)],axis=1)
    data_QTL_melt_test = pd.concat([data_QTL_melt_test,pd.concat([dummy]*int(data_QTL_melt_test.shape[0]/dummy.shape[0])).reset_index(drop=True)],axis=1)
 
    ## Create graphs
    ## Group once by sample id instead of re-scanning the full melted dataframe with a
    ## boolean mask on every loop iteration (O(n) instead of O(n^2) over samples).
    train_groups = dict(tuple(data_QTL_melt_train.groupby('variable')))
    if VALID:
        valid_groups = dict(tuple(data_QTL_melt_valid.groupby('variable')))
    test_groups = dict(tuple(data_QTL_melt_test.groupby('variable')))

    data_train = []
    for kk in range(data_pheno_train.shape[0]):
        tmp = Data()
        data_QTL_melt_train_tmp = train_groups[kk].iloc[:,1:]
        data_pheno_train_tmp = np.expand_dims(np.array(data_pheno_train[kk]),axis=0)
        tmp.x = torch.from_numpy(data_QTL_melt_train_tmp.to_numpy(dtype=float)).to(torch.float)
        tmp.y = torch.from_numpy(data_pheno_train_tmp).to(torch.float)
        tmp.edge_index = edge_index_train_base.clone()
        data_train += [tmp]
    
    if VALID:
        data_valid = []
        for kk in range(data_pheno_valid.shape[0]):
            tmp = Data()
            data_QTL_melt_valid_tmp = valid_groups[kk].iloc[:,1:]
            data_pheno_valid_tmp = np.expand_dims(np.array(data_pheno_valid[kk]),axis=0)
            tmp.x = torch.from_numpy(data_QTL_melt_valid_tmp.to_numpy(dtype=float)).to(torch.float)
            tmp.y = torch.from_numpy(data_pheno_valid_tmp).to(torch.float)
            tmp.edge_index = edge_index_valid_base.clone()
            data_valid += [tmp]
    
    data_test = []
    for kk in range(data_pheno_test.shape[0]):
        tmp = Data()
        data_QTL_melt_test_tmp = test_groups[kk].iloc[:,1:]
        data_pheno_test_tmp = np.expand_dims(np.array(data_pheno_test[kk]),axis=0)
        tmp.x = torch.from_numpy(data_QTL_melt_test_tmp.to_numpy(dtype=float)).to(torch.float)
        tmp.y = torch.from_numpy(data_pheno_test_tmp).to(torch.float)
        tmp.edge_index = edge_index_test_base.clone()
        data_test += [tmp]
    
    edge_name_from = list(data_test_columns[data_test[1].edge_index[0].tolist()])   
    edge_name_to = list(data_test_columns[data_test[1].edge_index[1].tolist()])  
        
    ## Create GAT    
    class GAT(torch.nn.Module):
        def __init__(self, hidden_channels, out_channels, dpout):
            super().__init__()
            
            self.conv1 = GATv2Conv((-1,-1), hidden_channels, add_self_loops=False, heads=heads, concat=True, dropout=dpout)
            self.conv2 = GATv2Conv((-1,-1), hidden_channels, add_self_loops=False, heads=heads, concat=False, dropout=dpout)
            self.lin1 = torch.nn.Linear(hidden_channels, out_channels)
    
        def forward(self, x,edge_index,batch, return_attention):
            x, edge_index, batch = x, edge_index, batch
            x = self.conv1(x, edge_index)
            x = F.elu(x)
            if return_attention:
                x, attention = self.conv2(x, edge_index,return_attention_weights=return_attention)
            else:
                x = self.conv2(x, edge_index,return_attention_weights=return_attention)
            x = F.elu(x)
            x = global_mean_pool(x, batch)
            x = self.lin1(x)
    
            if return_attention:
                return x, attention
            else:
                return x
                
    model = GAT(hidden_channels=neuron, out_channels=1, dpout=dropout)

    # ver4-4 R4.b/c/d/e (blueprint §2.4.2): device/AMP/loader-hygiene
    # settings resolved BEFORE the loaders (moved up from their original
    # position after the loaders) so every DataLoader construction below
    # can use them. use_amp/scaler mirror MLP.py's own already-proven
    # pattern exactly - both are inert (False / a disabled no-op scaler)
    # whenever the resolved device isn't CUDA, so a CPU-only run's
    # numerics and behaviour are completely unaffected.
    _resources = get_active_compute_resources()
    device = torch.device(_resources['device'])
    apply_torch_compute_settings(_resources)
    use_amp = bool(_resources['use_amp']) and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    _num_workers = dataloader_num_workers(_resources)
    _pin_memory = device.type == 'cuda'
    # ver4-4 R4.b (PC-1, verified - see pipeline_utils.split_batched_edge_attention's
    # own docstring): the test loader can now be genuinely batched on a
    # CUDA device (returns 1, today's exact behaviour, on CPU or when
    # GPU_EVAL_BATCH is unset) - every individual in this split shares the
    # identical, fixed edge_index topology, so PyG's own batch collation
    # keeps each graph's attention exactly recoverable afterwards.
    _eval_batch = torch_eval_batch_size(_resources, _resources.get('gpu_eval_batch'))

    train_loader = DataLoader(data_train, 
                             shuffle=True,
                             batch_size=bsize,
                             num_workers=_num_workers, pin_memory=_pin_memory)
    if VALID:
        valid_loader = DataLoader(data_valid, 
                                 batch_size=bsize,
                                 num_workers=_num_workers, pin_memory=_pin_memory)
    test_loader = DataLoader(data_test, 
                             batch_size=_eval_batch,
                             num_workers=_num_workers, pin_memory=_pin_memory)

    with gpu_slot():
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
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    out = model(batch.x,batch.edge_index,batch.batch,None)
                    loss = F.mse_loss(torch.squeeze(out), batch.y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_train_sum += loss.detach()

            print(f'Epoch {epoch:>3} | Train Loss: {loss_train_sum/batch_size:.5f}')

        ## Predict phenotypes for the test data
        model.eval()

        predicted_test = []
        actual_test = []
        attention = []
        with torch.no_grad():
            for test in test_loader:
                test = test.to(device)
                result, att = model(test.x,test.edge_index,test.batch,True)
                predicted_test += result.cpu().tolist()
                actual_test += test.y.cpu().tolist()
                # ver4-4 R4.b: att[1] covers every graph in THIS batch at
                # once when _eval_batch > 1 - split it back into one
                # per-graph attention vector each, exactly matching what
                # this SAME "for test in test_loader: ... .flatten()"
                # loop produced one graph at a time before this change
                # (verified directly against a real torch_geometric
                # install - see pipeline_utils.split_batched_edge_attention).
                for alpha_g in split_batched_edge_attention(att[1], test.num_graphs):
                    attention += [alpha_g.flatten().tolist()]

        predicted_test = [item for sublist in predicted_test for item in sublist]     

        ## Calculate the metrics
        mse = mean_squared_error(actual_test,predicted_test)
        r = pearsonr(actual_test, predicted_test)[0]

        ## Predict phenotypes for the train data
        train_loader = DataLoader(data_train, 
                                 shuffle=False,
                                 batch_size=bsize,
                                 num_workers=_num_workers, pin_memory=_pin_memory)
        predicted_train = []
        #actual_train = []
        with torch.no_grad():
            for train in train_loader:
                train = train.to(device)
                result = model(train.x,train.edge_index,train.batch,None)
                predicted_train += result.cpu().tolist()
                #actual_train += train.y.tolist()

        predicted_train = [k for i in predicted_train for k in i]

        predicted_valid = []
        #actual_valid = []
        if VALID:
            with torch.no_grad():
                for valid in valid_loader:
                    valid = valid.to(device)
                    result = model(valid.x,valid.edge_index,valid.batch,None)
                    predicted_valid += result.cpu().tolist()
                    #actual_valid += train.y.tolist()
            predicted_valid = [item for sublist in predicted_valid for item in sublist] 

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

            test_loader = DataLoader(data_test, 
                                    shuffle=True,
                                    batch_size=1,
                                    num_workers=_num_workers, pin_memory=_pin_memory)

            explanation = pd.DataFrame()
            cnt = 0
            for batch in test_loader:
                batch = batch.to(device)
                t = explainer(
                    batch.x,
                    batch.edge_index,
                    batch=batch.batch,
                    return_attention=None
                )
                t = pd.DataFrame(t['node_mask'].squeeze().detach().cpu()).sum(axis=1)
                if explanation.shape[0] == 0:
                    explanation = t
                else:
                    explanation += t
                cnt += 1

                if cnt == samples:
                    break

            effect = pd.DataFrame(explanation/cnt).T
            effect.columns = list(data_QTL_test.columns)
        else:
            effect = pd.DataFrame()

        attention = pd.concat([pd.DataFrame(edge_name_from),
                               pd.DataFrame(edge_name_to),
                               pd.DataFrame(attention).mean().T
                               ],axis=1)
        
    return r, mse, effect, predicted_test, predicted_valid, predicted_train, attention
