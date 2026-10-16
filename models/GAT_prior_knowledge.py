import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import os
import shap
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.explain import Explainer, CaptumExplainer

from pipeline_utils import (
    get_active_compute_resources, apply_torch_compute_settings, gpu_slot,
    dataloader_num_workers, torch_eval_batch_size, split_batched_edge_attention,
    is_lean_search_fit,
)
from models.interaction_extraction import tree_shap_interactions
from models.gat_search import lean_search_fit, topology_for, undirected_edge_index
from models.gat_stability import gat_stable

# ver4-4 R3.a (blueprint §2.3.2, root-cause finding R3.a): n_estimators and
# the SHAP-interaction sample count below used to be inline literals. Lifted
# into named module constants purely to make the divergence from
# Preprocess/data_driven_prior_network.py (which takes both of these values
# from config, not a hard-coded literal) at least VISIBLE here - R3 does not
# require making these configurable (that would be a HPARAM_SPECS/I5 change
# and is out of scope for this requirement).
_RF_N_ESTIMATORS = 500
_SHAP_INTERACTION_SAMPLE_SIZE = 50


@gat_stable('GAT_prior_knowledge')
def GAT_prior_knowledge(data_train, data_valid, data_test, params, *, sample_weight_ctx=None):
    # ver4-25, R8 (design blueprint SS2, Stage 6): the internal
    # topology-deciding RF (below) AND this model's own GAT training
    # loop (further down - see `tmp.w`/the weighted loss at its own
    # training loop) are BOTH weighted with the SAME shared `_w_fit`.
   
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
    top_rate = params[7]
    marker_effect = params[8]
    samples = params[9]
    # Update ID ver4-5, R2 (blueprint §4.2/§4.4): appended, not inserted -
    # I5. Read defensively (a ver4-4 config's shorter params list keeps
    # working, exactly as it did the day this field didn't exist).
    # Default False preserves ver4-4 behaviour: no run writes rows to
    # Interaction.csv for this model unless this is explicitly turned on.
    emit_interaction = params[10] if len(params) > 10 else False

    ## Train RF & extract interactions (-> the graph topology)

    train_x, train_y = data_train.iloc[:,:-1], data_train.iloc[:,-1]

    # ver4-4 R3.a: sklearn defaults n_jobs to 1 (one core) when not passed explicitly -
    # this forest fits on the FULL marker set and its SHAP interactions define this
    # model's entire graph topology (architecture doc §9.3), so it is threaded.
    _resources = get_active_compute_resources()
    # ver4-25, R8: the topology forest is weighted with the SAME shared fitting weight
    # every other model uses (fairness principle P8).
    _w_fit = sample_weight_ctx.w_fit if sample_weight_ctx is not None else None

    # ver4-46, R3: the forest + pairwise TreeSHAP + top-`top_rate` selection moved,
    # unchanged, into models/gat_search.py::compute_topology(). Its inputs are the
    # training genotypes/phenotype/weights and `top_rate` (not a tunable field), so
    # every candidate of a hyperparameter search recomputed the identical result;
    # `topology_for()` returns it from an exact content-addressed cache inside a
    # tuning call (search fits and the confirmatory fit) and is a plain call of
    # compute_topology() everywhere else (I11). Still the shared extractor of ver4-5 R2
    # (`tree_shap_interactions`, reduce='sum_then_abs' - this file's own reduction order
    # - governs the model's PREDICTIONS, so it is kept byte-for-byte; same quantile,
    # same '>=' operator).
    _topology = topology_for(
        train_x, train_y, _w_fit, top_rate,
        n_jobs=_resources['n_jobs'], rf_trees=_RF_N_ESTIMATORS,
        shap_samples=_SHAP_INTERACTION_SAMPLE_SIZE,
    )

    # Requirement 2 (near-zero marginal cost): when this model is asked to EMIT its
    # interaction pairs, this IS that same, already-computed, already-filtered pair set -
    # with real marker names, ready to write to Interaction.csv (gated by
    # model_registry.emits_interactions()).
    sample_interaction = _topology.as_named_frame(train_x.columns) if emit_interaction else pd.DataFrame()

    # The torch graph construction below needs POSITIONAL indices; `pair` keeps its
    # historical shape (an unnamed two-column frame read positionally by
    # edges_from/edges_to below). int64 is guaranteed even for an EMPTY selection.
    pair = pd.DataFrame({0: _topology.pos1, 1: _topology.pos2}).astype(np.int64)

    # ver4-46, R1: lean validation-only fit inside a hyperparameter-SEARCH fit - see
    # models/GAT_fully_connected.py's identical block and models/gat_search.py. The
    # undirected edge index is computed once (the topology is shared by every individual).
    if VALID and is_lean_search_fit():
        _m = train_x.shape[1]
        _predicted_valid = lean_search_fit(
            geno_train=train_x, y_train=train_y.to_numpy(),
            geno_valid=data_valid.iloc[:, :-1], w_fit=_w_fit,
            edge_index=undirected_edge_index(pair.iloc[:, 0].values, pair.iloc[:, 1].values, _m),
            n_markers=_m, neuron=neuron, dropout=dropout, lrate=lrate, decay=decay,
            epochs=epoch, bsize=bsize, heads=heads, attention_arg=True)
        return (float('nan'), float('nan'), pd.DataFrame(), [], _predicted_valid, [],
                pd.DataFrame(), pd.DataFrame())

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
    dummy = pd.get_dummies(pd.DataFrame(list(range(data_QTL_train.shape[1]))), columns=[0])
    dummy[dummy==False]=0
    dummy[dummy==True]=1
    data_QTL_melt_train = pd.concat([data_QTL_melt_train,pd.concat([dummy]*int(data_QTL_melt_train.shape[0]/dummy.shape[0])).reset_index(drop=True)],axis=1)
    if VALID:
        data_QTL_melt_valid = pd.concat([data_QTL_melt_valid,pd.concat([dummy]*int(data_QTL_melt_valid.shape[0]/dummy.shape[0])).reset_index(drop=True)],axis=1)
    data_QTL_melt_test = pd.concat([data_QTL_melt_test,pd.concat([dummy]*int(data_QTL_melt_test.shape[0]/dummy.shape[0])).reset_index(drop=True)],axis=1)
 
    ## Create graphs 
    edges_from = np.array(pair.iloc[:,0].values)
    edges_to = np.array(pair.iloc[:,1].values)
    
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
        edges_from_train_tmp, edges_to_train_tmp = edges_from, edges_to
        tmp.x = torch.from_numpy(data_QTL_melt_train_tmp.to_numpy(dtype=float)).to(torch.float)
        tmp.y = torch.from_numpy(data_pheno_train_tmp).to(torch.float)
        if _w_fit is not None:
            # ver4-25, R8: see models/GAT_infinitesimal.py's identical
            # comment - built exactly like `tmp.y` above.
            tmp.w = torch.from_numpy(np.expand_dims(np.array(_w_fit[kk], dtype=float), axis=0)).to(torch.float)
        tmp.edge_index = torch.stack([torch.from_numpy(edges_from_train_tmp).to(torch.long),torch.from_numpy(edges_to_train_tmp).to(torch.long)], dim=0)
        tmp = T.ToUndirected()(tmp)
        data_train += [tmp]
    
    if VALID:
        data_valid = []
        for kk in range(data_pheno_valid.shape[0]):
            tmp = Data()
            data_QTL_melt_valid_tmp = valid_groups[kk].iloc[:,1:]
            data_pheno_valid_tmp = np.expand_dims(np.array(data_pheno_valid[kk]),axis=0)
            edges_from_valid_tmp = edges_from
            edges_to_valid_tmp = edges_to
            tmp.x = torch.from_numpy(data_QTL_melt_valid_tmp.to_numpy(dtype=float)).to(torch.float)
            tmp.y = torch.from_numpy(data_pheno_valid_tmp).to(torch.float)
            tmp.edge_index = torch.stack([torch.from_numpy(edges_from_valid_tmp).to(torch.long),torch.from_numpy(edges_to_valid_tmp).to(torch.long)], dim=0)
            tmp = T.ToUndirected()(tmp)
            data_valid += [tmp]
        
    data_test = []
    for kk in range(data_pheno_test.shape[0]):
        tmp = Data()
        data_QTL_melt_test_tmp = test_groups[kk].iloc[:,1:]
        data_pheno_test_tmp = np.expand_dims(np.array(data_pheno_test[kk]),axis=0)
        edges_from_test_tmp = edges_from
        edges_to_test_tmp = edges_to
        tmp.x = torch.from_numpy(data_QTL_melt_test_tmp.to_numpy(dtype=float)).to(torch.float)
        tmp.y = torch.from_numpy(data_pheno_test_tmp).to(torch.float)
        tmp.edge_index = torch.stack([torch.from_numpy(edges_from_test_tmp).to(torch.long),torch.from_numpy(edges_to_test_tmp).to(torch.long)], dim=0)
        tmp = T.ToUndirected()(tmp)
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

    # ver4-4 R4.b/c/d/e: device/AMP/loader-hygiene settings resolved
    # BEFORE the loaders (moved up from their original position) - see
    # GAT_fully_connected.py's identical note for the full rationale.
    _resources = get_active_compute_resources()
    device = torch.device(_resources['device'])
    apply_torch_compute_settings(_resources)
    use_amp = bool(_resources['use_amp']) and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    _num_workers = dataloader_num_workers(_resources)
    _pin_memory = device.type == 'cuda'
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
                    if _w_fit is None:
                        loss = F.mse_loss(torch.squeeze(out), batch.y)
                    else:
                        # ver4-25, R8: Sum(w_i*r_i^2)/Sum(w_i) - see
                        # models/GAT_infinitesimal.py's identical comment.
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
        attention = []
        with torch.no_grad():
            for test in test_loader:
                test = test.to(device)
                result, att = model(test.x,test.edge_index,test.batch,True)
                predicted_test += result.cpu().tolist()
                actual_test += test.y.cpu().tolist()
                # ver4-4 R4.b - see GAT_fully_connected.py's identical note.
                for alpha_g in split_batched_edge_attention(att[1], test.num_graphs):
                    attention += [alpha_g.flatten().tolist()]

        predicted_test = [item for sublist in predicted_test for item in sublist]     

        ## Calculate the metrics
        mse = mean_squared_error(actual_test,predicted_test)
        r = pearsonr(actual_test, predicted_test)[0]

        ## Predict phenotypes for the validation data
        predicted_valid = []
        actual_valid = []
        if VALID:
            with torch.no_grad():
                for valid in valid_loader:
                    valid = valid.to(device)
                    result = model(valid.x,valid.edge_index,valid.batch,None)
                    predicted_valid += result.cpu().tolist()
                    actual_valid += valid.y.cpu().tolist()

            predicted_valid = [item for sublist in predicted_valid for item in sublist]   

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

    # Update ID ver4-5, R2: sample_interaction (empty unless
    # emit_interaction is True - see params unpacking above) is appended
    # as an 8th return value. Every caller (_call_model() and the main
    # per-task dispatch loop in genomic_prediction.py) was updated in
    # lockstep (I3) to unpack it.
    return r, mse, effect, predicted_test, predicted_valid, predicted_train, attention, sample_interaction
