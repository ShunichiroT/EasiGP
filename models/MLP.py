import pandas as pd
import numpy as np
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

from pipeline_utils import (
    get_active_compute_resources, apply_torch_compute_settings, gpu_slot, dataloader_num_workers,
)
from models.interaction_extraction import nid_interactions, shortlist_markers, top_select


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
    def __init__(self, n_inputs, neurons, dout, neurons2=None):
        super(MLP, self).__init__()

        self.hidden1 = Linear(n_inputs, neurons)
        self.dropout = Dropout(dout)
        # Optional second hidden layer - off by default (neurons2=None/0),
        # which reproduces the original single-hidden-layer architecture
        # exactly. Set neurons2 to a positive number of units to add a
        # second Linear+Dropout+ReLU block before the output layer.
        if neurons2:
            self.hidden_extra = Linear(neurons, int(neurons2))
            self.hidden2 = Linear(int(neurons2), 1)
        else:
            self.hidden_extra = None
            self.hidden2 = Linear(neurons, 1)
 
    # forward propaMLPe input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.dropout(X)
        X = F.relu(X)
        if self.hidden_extra is not None:
            X = self.hidden_extra(X)
            X = self.dropout(X)
            X = F.relu(X)
        X = self.hidden2(X)

        return X

    def downstream_influence(self):
        """Per-hidden-unit (of the FIRST hidden layer) aggregate absolute
        weight path to the scalar output - Neural Interaction Detection's
        (Tsang, Cheng & Liu, 2018) own definition of how much a hidden
        unit influences the network's final prediction, used to weight
        that unit's contribution to every marker pair's interaction
        strength (models.interaction_extraction.nid_interactions).

        Two architectures this model can have (see __init__):
          - single hidden layer: hidden1 -> hidden2 (Linear(neurons, 1)) -
            influence is simply |hidden2.weight| for each of the
            `neurons` units, flattened to shape (neurons,).
          - two hidden layers: hidden1 -> hidden_extra -> hidden2 - the
            influence of a hidden1 unit is the sum, over every
            hidden_extra unit, of |hidden_extra.weight| into it times
            |hidden2.weight| out of it (a standard "sum of absolute
            weighted paths" aggregation through the extra layer) -
            shape (neurons,), matrix product of the two weight matrices'
            absolute values.
        """
        with torch.no_grad():
            if self.hidden_extra is None:
                return self.hidden2.weight.detach().cpu().numpy().reshape(-1).copy()
            w_extra = self.hidden_extra.weight.detach().cpu().numpy()  # (neurons2, neurons)
            w_out = self.hidden2.weight.detach().cpu().numpy()        # (1, neurons2)
            path = np.abs(w_out) @ np.abs(w_extra)                     # (1, neurons)
            return path.reshape(-1)


def ML_Perceptron(train, valid, test, params):
    
    neurons = int(params[0])
    dout = params[1]
    lrate = params[2]
    decay = params[3]
    ep = int(params[4])
    bsize = int(params[5])
    # Size of an additional second hidden layer. None/0 keeps the original
    # single-hidden-layer architecture unchanged; a positive number adds a
    # second Linear+Dropout+ReLU block.
    neurons2 = params[6]
    shapley_num = params[7]
    # Update ID ver4-5, R2 Stage 9 (blueprint §4.2 Layer 3): appended,
    # read defensively (I5) - a ver4-4 (or Stages 1-8) config's shorter
    # params list keeps working, with interactions simply off. MLP's
    # interactions use Neural Interaction Detection (NID) - unlike
    # SVR/KNN's Friedman H-statistic, NID needs no background sample or
    # forward passes at all (see nid_interactions()'s own docstring),
    # so there is no interaction_background field here - only a
    # marker-count shortlist (to bound the OUTPUT table size, not
    # compute cost - NID's own cost is independent of how many pairs are
    # reported) and the usual top-N% output filter.
    get_interaction = params[8] if len(params) > 8 else False
    max_interaction_features = params[9] if len(params) > 9 else 500
    interaction_top = params[10] if len(params) > 10 else 'all'

    # Phase 2, Requirement 6: device dispatch (CPU/GPU), cudnn.benchmark,
    # and optional mixed precision are all resolved from the run's shared
    # compute-resource settings (pipeline_utils.resolve_compute_resources())
    # rather than hardcoded - a CPU-only node (the pre-existing default)
    # is completely unaffected.
    _resources = get_active_compute_resources()
    device = torch.device(_resources['device'])
    apply_torch_compute_settings(_resources)
    use_amp = bool(_resources['use_amp']) and device.type == 'cuda'

    # ver4-4 R4.d/e: num_workers from the shared, config-derived
    # TORCH_DATALOADER_WORKERS setting (daemon-process-safe - see
    # dataloader_num_workers()'s own docstring for RK-3), and
    # pin_memory only on a CUDA device (pinning on CPU wastes memory
    # for no benefit and is explicitly guarded against). Both default
    # to 0/False, reproducing today's exact loader behaviour.
    _num_workers = dataloader_num_workers(_resources)
    _pin_memory = device.type == 'cuda'

    train_data = CSVDataset(train)
    if valid.shape[0] != 0:
        valid_data = CSVDataset(valid)
    test_data = CSVDataset(test)
                     
    train_loader = DataLoader(train_data, batch_size=bsize, shuffle=True,
                               num_workers=_num_workers, pin_memory=_pin_memory)
    if valid.shape[0] != 0:
        valid_loader = DataLoader(valid_data, batch_size=bsize, shuffle=False,
                                   num_workers=_num_workers, pin_memory=_pin_memory)
    test_loader = DataLoader(test_data, batch_size=bsize, shuffle=False,
                              num_workers=_num_workers, pin_memory=_pin_memory)
    
    model = MLP(train.shape[1]-1, neurons, dout, neurons2)
    # Requirement 6: guard entry into GPU-dispatched work with the shared
    # GPU-slot semaphore, so N_CPU_WORKERS worker processes sharing one
    # physical GPU (intra-batch parallelism, Requirement 7) don't all pile
    # onto it at once - a no-op unless that feature is actually enabled.
    with gpu_slot():
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lrate, weight_decay=decay) #0.0005
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        for epoch in range(ep):
             loss_train_sum = 0
             for inputs, targets in train_loader:
                 inputs = inputs.to(device)
                 targets = targets.to(device)
                 optimizer.zero_grad()
                 with torch.autocast(device_type=device.type, enabled=use_amp):
                     yhat = model(inputs)
                     loss = F.mse_loss(yhat, targets)
                 loss_train_sum += loss.detach()
                 scaler.scale(loss).backward()
                 scaler.step(optimizer)
                 scaler.update()
             print(f'Epoch {epoch:>3} | Train Loss: {loss_train_sum/len(train_loader):.5f}')

        predicted = []
        actuals = []
        model.eval()
        with torch.no_grad():
            for inputs, targets in test_loader:
                 inputs = inputs.to(device)
                 yhat = model(inputs)
                 yhat = yhat.detach().cpu().tolist()
                 actual = targets.detach().cpu().tolist()
                 predicted.append([item for sublist in yhat for item in sublist])
                 actuals.append([item for sublist in actual for item in sublist])
        predicted = [item for sublist in predicted for item in sublist]
        actuals = [item for sublist in actuals for item in sublist]

        mse = mean_squared_error(actuals, predicted)
        r = scipy.stats.pearsonr(actuals, predicted)[0]

        train_loader = DataLoader(train_data, batch_size=bsize, shuffle=False,
                                   num_workers=_num_workers, pin_memory=_pin_memory)
        predicted_train = []
        with torch.no_grad():
            for inputs, targets in train_loader:
                 inputs = inputs.to(device)
                 yhat = model(inputs)
                 yhat = yhat.detach().cpu().tolist()
                 predicted_train.append([item for sublist in yhat for item in sublist])
        predicted_train = [item for sublist in predicted_train for item in sublist]

        predicted_valid = []
        if valid.shape[0] != 0:
            with torch.no_grad():
                for inputs, targets in valid_loader:
                     inputs = inputs.to(device)
                     yhat = model(inputs)
                     yhat = yhat.detach().cpu().tolist()
                     predicted_valid.append([item for sublist in yhat for item in sublist])
            predicted_valid = [item for sublist in predicted_valid for item in sublist]

        model.train()

        d_train = torch.tensor(train.iloc[:,:-1].values, dtype=torch.float32).to(device)
        d_test = torch.tensor(test.iloc[:,:-1].values, dtype=torch.float32).to(device)

        # Requirement 6: 'Explainer'/'CaptumExplainer' (here, SHAP's
        # DeepExplainer) may silently run on CPU regardless of the model's
        # own device unless every tensor handed to it is explicitly moved
        # first - both the model (already .to(device) above) and its
        # background/explained samples (d_train/d_test, moved above) are
        # aligned to the SAME device here, so the explainer genuinely runs
        # on the GPU when one is in use rather than falling back to CPU.
        explainer = shap.DeepExplainer(model, shap.sample(d_train, shapley_num))
        effect = abs(explainer.shap_values(shap.sample(d_test, shapley_num), check_additivity=False)).sum(axis=0)
        effect_flat = np.asarray(effect).ravel()

        # Update ID ver4-5, R2 Stage 9 (blueprint §4.2 Layer 3): Neural
        # Interaction Detection (NID) is MLP's DEFAULT interaction method
        # - pure matrix algebra over the already-trained first-layer
        # weights and this model's own downstream_influence() (see the
        # MLP class above), no forward passes at all. Shortlisted by the
        # SAME effect values just computed above (already the model's own
        # best per-marker importance measure - no separate importance
        # computation needed, unlike SVR.py/KNN.py's own correlation
        # proxy), purely to bound the OUTPUT table size for very large
        # marker counts (NID's own compute cost does not depend on how
        # many pairs are reported).
        if get_interaction == True:
            marker_names = list(train.iloc[:, :-1].columns)
            shortlist_idx = shortlist_markers(effect_flat, max_interaction_features)
            shortlist_names = [marker_names[idx] for idx in shortlist_idx]
            w1_full = model.hidden1.weight.detach().cpu().numpy()  # (neurons, n_markers)
            w1_shortlist = w1_full[:, shortlist_idx]
            influence = model.downstream_influence()

            interaction_sample = nid_interactions(w1_shortlist, influence, shortlist_names)
            interaction_sample = top_select(interaction_sample, 'percentage', interaction_top)
        else:
            interaction_sample = pd.DataFrame()

    return r, mse, pd.DataFrame(effect).T, interaction_sample, predicted, predicted_valid, predicted_train