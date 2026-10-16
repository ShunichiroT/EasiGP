import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.explain import Explainer, CaptumExplainer

from pipeline_utils import (
    get_active_compute_resources, apply_torch_compute_settings, gpu_slot, dataloader_num_workers,
    is_lean_search_fit,
)
from models.gat_search import lean_search_fit, undirected_edge_index
from models.gat_stability import gat_stable


@gat_stable('GAT_infinitesimal')
def GAT_infinitesimal(data_train, data_valid, data_test, params, *, sample_weight_ctx=None):
    # ver4-25, R8 (design blueprint SS2, Stage 6): `sample_weight_ctx.
    # w_fit` (a plain per-individual vector, aligned to `data_train`'s
    # own row order - see architecture doc SS4.2's positional-identity
    # contract) is attached as a per-graph scalar attribute (`tmp.w`,
    # built EXACTLY like `tmp.y` just below) and used for a
    # `Sum(w_i*r_i^2)/Sum(w_i)` training loss - the SAME mechanism
    # `models/MLP.py` already ships. TEST/VALID scoring for this model
    # is unaffected either way (computed centrally by the caller,
    # regardless of how the model itself was fit). `None` (every pre-
    # ver4-25 caller) is a byte-identical `F.mse_loss(...)` - I11.
    _w_fit = sample_weight_ctx.w_fit if sample_weight_ctx is not None else None

    
    if data_valid.shape[0] != 0:
        VALID = True
    else:
        VALID = False
    
    neuron = params[0]
    dropout = params[1]
    lrate = params[2]
    decay = params[3]
    epoch = params[4]
    bsize = params[5]
    heads = params[6]
    marker_effect = params[7]
    samples = params[8]

    # ver4-46, R1: lean validation-only fit inside a hyperparameter-SEARCH fit - see
    # models/GAT_fully_connected.py's identical block and models/gat_search.py.
    # Every node attends only to itself here, so the shared edge index is the
    # (de-duplicated) self-loop set the legacy body builds per individual.
    if VALID and is_lean_search_fit():
        _m = data_train.shape[1] - 1
        _loops = np.arange(_m)
        _predicted_valid = lean_search_fit(
            geno_train=data_train.iloc[:, :-1], y_train=data_train.iloc[:, -1].to_numpy(),
            geno_valid=data_valid.iloc[:, :-1], w_fit=_w_fit,
            edge_index=undirected_edge_index(_loops, _loops, _m), n_markers=_m,
            neuron=neuron, dropout=dropout, lrate=lrate, decay=decay, epochs=epoch,
            bsize=bsize, heads=heads, attention_arg=False)
        return (float('nan'), float('nan'), pd.DataFrame(), [], _predicted_valid, [])

    ## Preprocess the data so that it can be converted into a graph format
    if VALID:
        data_QTL_train,data_QTL_valid, data_QTL_test = data_train.iloc[:,:-1].reset_index(drop=True), data_valid.iloc[:,:-1].reset_index(drop=True), data_test.iloc[:,:-1].reset_index(drop=True)
        data_QTL_melt_train, data_QTL_melt_valid, data_QTL_melt_test = data_QTL_train.T.melt(), data_QTL_valid.T.melt(), data_QTL_test.T.melt()
        data_pheno_train, data_pheno_valid, data_pheno_test = data_train.iloc[:,-1].reset_index(drop=True), data_valid.iloc[:,-1].reset_index(drop=True), data_test.iloc[:,-1].reset_index(drop=True)
    else:
        data_QTL_train,data_QTL_test = data_train.iloc[:,:-1].reset_index(drop=True), data_test.iloc[:,:-1].reset_index(drop=True)
        data_QTL_melt_train, data_QTL_melt_test = data_QTL_train.T.melt(), data_QTL_test.T.melt()
        data_pheno_train, data_pheno_test = data_train.iloc[:,-1].reset_index(drop=True), data_test.iloc[:,-1].reset_index(drop=True)
            
    ## Create graphs
    dummy = pd.get_dummies(pd.DataFrame(list(range(data_QTL_train.shape[1]))), columns=[0])
    dummy[dummy==False]=0
    dummy[dummy==True]=1
    data_QTL_melt_train = pd.concat([data_QTL_melt_train,pd.concat([dummy]*int(data_QTL_melt_train.shape[0]/dummy.shape[0])).reset_index(drop=True)],axis=1)
    if VALID:
        data_QTL_melt_valid = pd.concat([data_QTL_melt_valid,pd.concat([dummy]*int(data_QTL_melt_valid.shape[0]/dummy.shape[0])).reset_index(drop=True)],axis=1)
    data_QTL_melt_test = pd.concat([data_QTL_melt_test,pd.concat([dummy]*int(data_QTL_melt_test.shape[0]/dummy.shape[0])).reset_index(drop=True)],axis=1)
    
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
        edges_from_train_tmp = edges_to_train_tmp = np.array(range(0,data_QTL_melt_train_tmp.shape[0]))
        tmp.edge_index = torch.stack([torch.from_numpy(edges_from_train_tmp).to(torch.long),torch.from_numpy(edges_to_train_tmp).to(torch.long)], dim=0)
        tmp.x = torch.from_numpy(data_QTL_melt_train_tmp.to_numpy(dtype=float)).to(torch.float)
        tmp.y = torch.from_numpy(data_pheno_train_tmp).to(torch.float)
        if _w_fit is not None:
            # ver4-25, R8: built exactly like `tmp.y` just above (a
            # shape-(1,) tensor per graph) so PyG's default batch
            # collation concatenates it along dim 0 the SAME way it
            # already does for `y` - `batch.w` therefore comes out
            # shape (batch_size,), matching `batch.y` element-for-element.
            tmp.w = torch.from_numpy(np.expand_dims(np.array(_w_fit[kk], dtype=float), axis=0)).to(torch.float)
        tmp = T.ToUndirected()(tmp)
        data_train += [tmp]
    
    if VALID:
        data_valid = []
        for kk in range(data_pheno_valid.shape[0]):
            tmp = Data()
            data_QTL_melt_valid_tmp = valid_groups[kk].iloc[:,1:]
            data_pheno_valid_tmp = np.expand_dims(np.array(data_pheno_valid[kk]),axis=0)
            edges_from_valid_tmp = edges_to_valid_tmp = np.array(range(0,data_QTL_melt_valid_tmp.shape[0]))
            tmp.edge_index = torch.stack([torch.from_numpy(edges_from_valid_tmp).to(torch.long),torch.from_numpy(edges_to_valid_tmp).to(torch.long)], dim=0)
            tmp.x = torch.from_numpy(data_QTL_melt_valid_tmp.to_numpy(dtype=float)).to(torch.float)
            tmp.y = torch.from_numpy(data_pheno_valid_tmp).to(torch.float)
            tmp = T.ToUndirected()(tmp)
            data_valid += [tmp]
    
    data_test = []
    for kk in range(data_pheno_test.shape[0]):
        tmp = Data()
        data_QTL_melt_test_tmp = test_groups[kk].iloc[:,1:]
        data_pheno_test_tmp = np.expand_dims(np.array(data_pheno_test[kk]),axis=0)
        edges_from_test_tmp = edges_to_test_tmp = np.array(range(0,data_QTL_melt_test_tmp.shape[0]))
        tmp.edge_index = torch.stack([torch.from_numpy(edges_from_test_tmp).to(torch.long),torch.from_numpy(edges_to_test_tmp).to(torch.long)], dim=0)
        tmp.x = torch.from_numpy(data_QTL_melt_test_tmp.to_numpy(dtype=float)).to(torch.float)
        tmp.y = torch.from_numpy(data_pheno_test_tmp).to(torch.float)
        tmp = T.ToUndirected()(tmp)
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

    # Phase 2, Requirement 6: device dispatch, cudnn.benchmark, and
    # optional mixed precision are resolved from the run's shared
    # compute-resource settings (pipeline_utils.resolve_compute_resources())
    # rather than a bare "cuda:0 if available" check - this keeps every
    # model module consistent with a single, config-derived source, and
    # lets a run explicitly pin a device (TORCH_DEVICE) when more than one
    # GPU is visible. Resolved BEFORE the loaders below (moved up from its
    # original position) so ver4-4 R4.d/e's pin_memory/num_workers can use
    # it immediately.
    _resources = get_active_compute_resources()
    device = torch.device(_resources['device'])
    apply_torch_compute_settings(_resources)
    use_amp = bool(_resources['use_amp']) and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ver4-4 R4.d/e: num_workers from the shared, config-derived
    # TORCH_DATALOADER_WORKERS setting (daemon-process-safe - RK-3), and
    # pin_memory only on a CUDA device. Both default to 0/False,
    # reproducing today's exact loader behaviour.
    _num_workers = dataloader_num_workers(_resources)
    _pin_memory = device.type == 'cuda'

    train_loader = DataLoader(data_train, 
                             shuffle=True,
                             batch_size=bsize,
                             num_workers=_num_workers, pin_memory=_pin_memory)
    if VALID:
        valid_loader = DataLoader(data_valid, 
                                 batch_size=bsize,
                                 num_workers=_num_workers, pin_memory=_pin_memory)
    test_loader = DataLoader(data_test, 
                             batch_size=bsize,
                             num_workers=_num_workers, pin_memory=_pin_memory)

    # Requirement 6: guard entry into GPU-dispatched work with the shared
    # GPU-slot semaphore, so N_CPU_WORKERS worker processes sharing one
    # physical GPU (intra-batch parallelism, Requirement 7) don't all pile
    # onto it at once - a no-op unless that feature is actually enabled.
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
                    out = model(batch.x, batch.edge_index, batch.batch)
                    if _w_fit is None:
                        loss = F.mse_loss(torch.squeeze(out), batch.y)
                    else:
                        # ver4-25, R8: Sum(w_i*r_i^2)/Sum(w_i) - the SAME
                        # weighted-loss formula models/MLP.py already
                        # ships (blueprint SS2 R8's own mechanism table).
                        _sq_err = (torch.squeeze(out) - batch.y) ** 2
                        loss = (_sq_err * batch.w).sum() / batch.w.sum().clamp_min(1e-12)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_train_sum += loss.detach()

            print(f'Epoch {epoch:>3} | Train Loss: {loss_train_sum/batch_size:.5f}')

        ## Predict phenotypes for the test data
        model.eval()

        predicted_test = []
        actual_test = []
        with torch.no_grad():
            for test in test_loader:
                test = test.to(device)
                result = model(test.x, test.edge_index, test.batch)
                predicted_test += result.cpu().tolist()
                actual_test += test.y.cpu().tolist()

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
                result = model(train.x, train.edge_index, train.batch)
                predicted_train += result.cpu().tolist()
                #actual_train += train.y.tolist()

        predicted_train = [k for i in predicted_train for k in i]

        predicted_valid = []
        #actual_valid = []
        if VALID:
            with torch.no_grad():
                for valid in valid_loader:
                    valid = valid.to(device)
                    result = model(valid.x, valid.edge_index, valid.batch)
                    predicted_valid += result.cpu().tolist()
                    #actual_valid += train.y.tolist()

            predicted_valid = [k for i in predicted_valid for k in i]

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
                # Requirement 6: the explainer must run on the SAME device
                # as the model it's explaining - CaptumExplainer silently
                # falls back to CPU (or raises a device-mismatch error)
                # otherwise, so every tensor handed to it is explicitly
                # moved here.
                batch = batch.to(device)
                t = explainer(
                    batch.x,
                    batch.edge_index,
                    batch=batch.batch
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

    return r, mse, effect, predicted_test, predicted_valid, predicted_train
