
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, Linear, to_hetero_with_bases, to_hetero
from torch_geometric.loader import HGTLoader
from torch_geometric.explain import Explainer, CaptumExplainer

from pipeline_utils import (
    get_active_compute_resources, apply_torch_compute_settings, gpu_slot, dataloader_num_workers,
    is_lean_search_fit,
)
from models.gat_stability import gat_stable


@gat_stable('GAT_infinitesimal_node_level')
def GAT_infinitesimal_node_level(data_train, data_valid, data_test, params, *, sample_weight_ctx=None):
    # ver4-25, R8 (design blueprint SS2, Stage 6): unlike the other four
    # GAT variants, this model builds ONE graph for the whole task (train
    # + valid + test as a single HeteroData object, split via boolean
    # masks - see the mask construction below), so `_w_fit` cannot be
    # attached per-graph the same way. Instead it becomes a per-NODE
    # attribute on the 'pheno' node type (`data['pheno'].w`), aligned to
    # the SAME `data_train`+`data_valid`+`data_test` concatenation order
    # every mask below already uses - train positions carry the real
    # weight, valid/test positions carry a neutral placeholder (1.0) that
    # is NEVER read (the training loss only ever indexes by
    # `train_mask`). `None` (every pre-ver4-25 caller) is a byte-
    # identical `F.mse_loss(...)` - I11.
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
    samples = params[7]
    marker_effect = params[8]

    # ver4-46, R1: inside a hyperparameter-SEARCH fit that asked for the lean path only the
    # validation predictions are read. Every pheno node is connected ONLY to its own
    # marker nodes (the 'affect' / 'self' edges above), so test individuals never touch
    # the train/valid nodes: leaving them out changes no train or validation output, and
    # saves building their node types, the HGT test sampling and the full-graph train
    # pass below. (Not bit-identical to a full-graph fit: the HGT sampler's own random
    # draws and its neighbour cap `max_neighbors` depend on the node count.)
    _lean = bool(VALID and is_lean_search_fit())
    if _lean:
        data_test = data_test.iloc[:0]

    ## Preprocess the data so that it can be converted into a graph format
    data = pd.concat([data_train, data_valid, data_test])
    data_QTL, data_pheno = data.iloc[:,:-1].reset_index(drop=True), data.iloc[:,-1].reset_index(drop=True)
    
    x_qtl = []
    for i in range(len(data_QTL.iloc[0,:])):
        x_qtl.append(list(map(lambda x: [x], data_QTL.iloc[:,i].to_list())))
    x_qtl = np.array(x_qtl)
    
    x_pheno = [[0]] * len(data)    
    y_pheno = np.array(list(map(lambda x: [x], data_pheno.to_list())), dtype='float32')
    
    edges = np.array(range(0,len(data_QTL.iloc[:,0])))
    
    ## Convert the data into a graph
    data = HeteroData()
    data['pheno'].x = torch.tensor(x_pheno, dtype=torch.float)
    data['pheno'].y = torch.from_numpy(y_pheno)
     
    for i in range(len(x_qtl)):
        data['qtl_'+str(i+1)].x = torch.tensor(x_qtl[i], dtype=torch.float)  
        data['qtl_'+str(i+1),'affect','pheno'].edge_index = torch.stack([torch.from_numpy(edges).to(torch.long),torch.from_numpy(edges).to(torch.long)], dim=0)
        data['qtl_'+str(i+1),'self','qtl_'+str(i+1)].edge_index = torch.stack([torch.from_numpy(edges).to(torch.long),torch.from_numpy(edges).to(torch.long)], dim=0)
    
    ## Add masks to distinguish between train, validation and test data
    ## data was built as concat([data_train, data_valid, data_test]), so the three
    ## groups are always contiguous blocks - build the masks directly instead of
    ## looping over every sample with an O(n) 'in list' check each time.
    n_train, n_valid, n_test = data_train.shape[0], data_valid.shape[0], data_test.shape[0]

    mask_train = np.array([True] * n_train + [False] * n_valid + [False] * n_test)
    mask_valid = np.array([False] * n_train + [True] * n_valid + [False] * n_test)
    mask_test = np.array([False] * n_train + [False] * n_valid + [True] * n_test)
    
    data['pheno'].train_mask = torch.from_numpy(mask_train).to(torch.bool)
    data['pheno'].valid_mask = torch.from_numpy(mask_valid).to(torch.bool)
    data['pheno'].test_mask = torch.from_numpy(mask_test).to(torch.bool)

    if _w_fit is not None:
        # ver4-25, R8: real weight for the n_train training nodes,
        # neutral placeholder (1.0, never read - see this function's own
        # docstring) for valid/test nodes, in the SAME row order the
        # masks above already use (train, then valid, then test).
        w_all = np.concatenate([
            np.asarray(_w_fit, dtype=float), np.ones(n_valid + n_test, dtype=float),
        ])
        data['pheno'].w = torch.from_numpy(w_all).to(torch.float)
    
    ## Define GAT class
    class GAT(torch.nn.Module):
        def __init__(self, neurons, out_channels, dpout):
            super().__init__()
            self.conv1 = GATv2Conv((-1,-1), neurons, add_self_loops=False, heads=heads, concat=True, dropout=dpout)
            self.lin1 = Linear(-1, neurons*heads)
            self.conv2 = GATv2Conv((-1,-1), neurons, add_self_loops=False,heads=heads, concat=True, dropout=dpout)
            self.lin2 = Linear(-1, neurons*heads)
            self.conv3 = GATv2Conv((-1,-1), out_channels, add_self_loops=False, heads=heads, concat=False, dropout=dpout)
            self.lin3 = Linear(-1, out_channels)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index) + self.lin1(x)
            x = F.elu(x)
            x = self.conv2(x, edge_index) + self.lin2(x)
            x = F.elu(x)
            x = self.conv3(x, edge_index) + self.lin3(x)
            return x
        
    class NodeDecoder(torch.nn.Module):
        def __init__(self, hidden_channels):
            super().__init__()

        def forward(self, z_dict):
            return z_dict['pheno'].mean(axis=1)

    class Model(torch.nn.Module):
        def __init__(self, neurons, out_channels, dpout):
            super().__init__()
            self.encoder = GAT(neurons, out_channels,dpout)
            self.encoder = to_hetero(self.encoder, data.metadata())
            self.decoder = NodeDecoder(neurons)

        def forward(self, x_dict, edge_index_dict):
            z_dict = self.encoder(x_dict, edge_index_dict)

            return self.decoder(z_dict)

    ## Define training pipeline
    def train():
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lrate, weight_decay=decay)
        
        for ep in range(epoch):        
            loss_train_sum = 0
            for batch in train_loader:
                batch = batch.to(device)
                model.train()
                optimizer.zero_grad()
                # ver4-4 R4.b/c: mirrors MLP.py's own GradScaler/autocast
                # pattern exactly - a no-op (use_amp resolves False)
                # whenever the resolved device isn't CUDA, so a CPU-only
                # run's numerics are completely unaffected.
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    out = model(batch.x_dict, batch.edge_index_dict)
                    mask = batch['pheno'].train_mask
                    if _w_fit is None:
                        loss = F.mse_loss(out[mask].unsqueeze(-1), batch['pheno'].y[mask])
                    else:
                        # ver4-25, R8: Sum(w_i*r_i^2)/Sum(w_i) over the
                        # sampled subgraph's own train-masked nodes -
                        # HGTLoader carries `data['pheno'].w` forward
                        # through its own node sampling/relabelling the
                        # SAME way it already does for `y`/`train_mask`
                        # (both node-level attributes on the identical
                        # node type), so `batch['pheno'].w[mask]` is
                        # aligned to `batch['pheno'].y[mask]` element-
                        # for-element.
                        _w_batch = batch['pheno'].w[mask].unsqueeze(-1)
                        _sq_err = (out[mask].unsqueeze(-1) - batch['pheno'].y[mask]) ** 2
                        loss = (_sq_err * _w_batch).sum() / _w_batch.sum().clamp_min(1e-12)
                loss_train_sum += loss.detach()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            print(f'Epoch {ep:>3} | Train Loss: {loss_train_sum/len(train_loader):.5f}')
        
        return model
    
    ## Develop a GAT model
    model = Model(neurons=neuron, out_channels=1, dpout=dropout)
    _resources = get_active_compute_resources()
    device = torch.device(_resources['device'])
    apply_torch_compute_settings(_resources)
    use_amp = bool(_resources['use_amp']) and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ver4-4 R4.d/e: num_workers from the shared, config-derived
    # TORCH_DATALOADER_WORKERS setting (daemon-process-safe - RK-3), and
    # pin_memory only on a CUDA device. Both default to 0/False,
    # reproducing today's exact loader behaviour. HGTLoader forwards
    # **kwargs to the underlying torch DataLoader it wraps, so both
    # keywords are accepted here exactly as on every other loader in
    # this codebase (verified directly against the installed
    # torch_geometric's own HGTLoader.__init__ signature).
    _num_workers = dataloader_num_workers(_resources)
    _pin_memory = device.type == 'cuda'

    with gpu_slot():
        model.to(device)

        with torch.no_grad():
            out = model(data.x_dict, data.edge_index_dict)

        ## Convert the data into mini-batches
        ## Sample enough neighbors per hop to always cover every marker/sample node, rather
        ## than a fixed count tuned for a specific dataset size (which would silently drop
        ## markers on larger datasets).
        max_neighbors = max(data_QTL.shape[1], data_pheno.shape[0])
        train_loader = HGTLoader(data, 
                                num_samples={key:[max_neighbors] * 4 for key in data.node_types},
                                shuffle=True,
                                batch_size=bsize,
                                input_nodes=('pheno', data['pheno'].train_mask),
                                num_workers=_num_workers, pin_memory=_pin_memory)
        if VALID:
            valid_loader = HGTLoader(data, 
                                     num_samples={key:[max_neighbors] * 4 for key in data.node_types},
                                     batch_size=bsize,shuffle=False,
                                     input_nodes=('pheno', data['pheno'].valid_mask),
                                     num_workers=_num_workers, pin_memory=_pin_memory)
        if not _lean:
            test_loader = HGTLoader(data, 
                                     num_samples={key:[max_neighbors] * 4 for key in data.node_types},
                                     batch_size=bsize,shuffle=False,
                                     input_nodes=('pheno', data['pheno'].test_mask),
                                     num_workers=_num_workers, pin_memory=_pin_memory)
        ## Train a model
        model = train()

        ## Predict phenotypes for the test data
        model.eval()
        predicted_test = []
        actual_test = []
        if not _lean:
            with torch.no_grad():
                for test in test_loader:
                    test = test.to(device)
                    result = model(test.x_dict, test.edge_index_dict)
                    predicted_test.append(result[test['pheno'].test_mask].cpu().tolist())
                    actual_test.append([item for sublist in test['pheno'].y[test['pheno'].test_mask].cpu().tolist() for item in sublist])
            predicted_test = [item for sublist in predicted_test for item in sublist]
            actual_test = [item for sublist in actual_test for item in sublist]

            ## Calculate the metrics
            mse = mean_squared_error(actual_test,predicted_test)
            r = pearsonr(actual_test, predicted_test)[0]
        else:
            mse = r = float('nan')   # not read by a search fit

        ## Store prediction result for the validation data
        predicted_valid = []
        actual_valid = []
        if VALID:
            with torch.no_grad():
                for valid in valid_loader:
                    valid = valid.to(device)
                    result = model(valid.x_dict, valid.edge_index_dict)
                    predicted_valid.append(result[valid['pheno'].valid_mask].cpu().tolist())
                    actual_valid.append([item for sublist in valid['pheno'].y[valid['pheno'].valid_mask].cpu().tolist() for item in sublist])
            predicted_valid = [item for sublist in predicted_valid for item in sublist]
            actual_valid = [item for sublist in actual_valid for item in sublist] 

        ## Store prediction result for the train data
        predicted_train = []
        actual_train = []
        if not _lean:
            with torch.no_grad():
                result = model(data.x_dict, data.edge_index_dict)
            predicted_train.append(result[data['pheno'].train_mask].cpu().tolist())
            actual_train.append([item for sublist in data['pheno'].y[data['pheno'].train_mask].cpu().tolist() for item in sublist])
            predicted_train = [k for i in predicted_train for k in i]
            actual_train = [k for i in actual_train for k in i]

        ## Extract marker effects
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
                    )
            )

            hetero_explanation = explainer(
                data.x_dict,
                data.edge_index_dict,
            )

            effect = pd.DataFrame()
            for ii in range(1,len(data.x_dict)):
                effect = pd.concat([effect, pd.DataFrame(hetero_explanation['qtl_'+str(ii)]['node_mask'].squeeze().cpu().tolist())],axis=1)
            effect = effect[mask_test==1]
            if effect.shape[0] > samples:
                effect = effect.sample(n=samples, random_state=1)
            effect = pd.DataFrame(effect.sum()).T/effect.shape[0]
            effect.columns = list(data_QTL.columns) 

        else:
            effect = pd.DataFrame()
        
    return r, mse, effect, predicted_test, predicted_valid, predicted_train
 