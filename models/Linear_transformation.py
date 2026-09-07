import pandas as pd
import numpy as np
import torch
import shap
import torch.nn.functional as F
import copy
import scipy.stats
from torch.utils.data import Dataset, DataLoader
from torch.nn import Linear, Module
from sklearn.metrics import mean_squared_error

from pipeline_utils import safe_regression_metrics, get_active_compute_resources, apply_torch_compute_settings, gpu_slot
from .ensemble_regularization import apply_naive_shrinkage, analytic_simplex_weights


def _safe_row_normalize(df):
    """Row-wise L1-normalize (each row divided by its own sum of - here
    already non-negative - values, e.g. abs(marker effect) or SHAP-based
    importance) - safely: a row whose sum is exactly 0 (a model that
    assigned literally zero effect/importance to everything) is left as
    all-zero, rather than becoming all-NaN via an unguarded 0/0 division,
    which would otherwise silently corrupt this weighted ensemble's
    entire combined marker-effect output - NaN + anything is NaN."""
    row_sums = df.sum(axis=1)
    return df.div(row_sums, axis=0).fillna(0)


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


class MLP_linear_simplex(Module):
    """Phase 2, Requirement 4.4 (audit follow-up): the plain `MLP_linear`
    above has no sign constraint at all on its combining weights - an
    audited, real risk flagged in the Phase 2 design blueprint
    ('unconstrained weights admit negative/extrapolating combinations').
    This variant reparameterises the same single linear combination as
    `softmax(z)` over a raw, unconstrained parameter vector `z`, so the
    EFFECTIVE combining weights are always non-negative and sum to 1 by
    construction - a constrained-simplex ensemble, still trained by plain
    gradient descent on the same MSE loss. Opt-in via
    HPARAMETERS_OPT[7] (`use_simplex_weights`) - off by default, so this
    pipeline's original (pre-Phase-2), unconstrained behaviour is
    unchanged unless explicitly requested.
    """
    def __init__(self, n_inputs):
        super(MLP_linear_simplex, self).__init__()
        self.raw_weights = torch.nn.Parameter(torch.zeros(n_inputs))

    def combining_weights(self):
        return torch.softmax(self.raw_weights, dim=0)

    def forward(self, X):
        w = self.combining_weights()
        return (X * w).sum(dim=1, keepdim=True)

def Linear_transformation(data_train, data_valid, data_test, record, effect, weight, MODEL, HPARAMETERS_OPT):
    
    # Hyperparameter setting
    learning_rate = HPARAMETERS_OPT[0]
    epochs = HPARAMETERS_OPT[1]
    decay = HPARAMETERS_OPT[2]
    batch_s = HPARAMETERS_OPT[3]
    pat = HPARAMETERS_OPT[4]
    num_models = HPARAMETERS_OPT[5]

    # Phase 2, Requirement 4.4 (audit follow-up) / Requirement 5: unlike
    # Nelder_Mead.py/Bayesian_optimisation.py, this method's objective is
    # ALREADY the quadratic, naturally interior-optimised
    # `sum_s(sum_i wi*Mi(s) - V(s))^2` form (plain MSE of the trained
    # linear combination's own prediction) - not the linear-fractional
    # DPT-ratio form those two audited files reduce to (see
    # ensemble_regularization.py's module docstring for that derivation).
    # It is therefore NOT susceptible to the same vertex/one-hot-collapse
    # degeneracy, and no diversity_penalty is structurally REQUIRED here.
    # What the audit DID flag as a genuine, distinct risk (§4.4): the
    # combining weights (`model.hidden1.weight`) are completely
    # UNCONSTRAINED - no non-negativity, no sum-to-1 - which admits
    # negative/extrapolating combinations. Three opt-in, appended-only
    # settings address this and bring this method's safety guarantees in
    # line with the other two weighted-ensemble methods; every one
    # defaults to this method's exact original (pre-Phase-2) behaviour
    # when absent:
    #   [6] diversity_penalty (float, default 0.0) - an ADDITIONAL ridge-
    #       to-uniform term on the (softmax-normalised, for a stable
    #       comparison regardless of raw weight scale) combining weights,
    #       added to the training loss - optional extra regularisation,
    #       not a structural requirement here.
    #   [7] use_simplex_weights (bool, default False) - use
    #       MLP_linear_simplex (softmax-reparameterised, non-negative,
    #       sum-to-1 combining weights) instead of the original
    #       unconstrained MLP_linear.
    #   [8] alpha (float, default 1.0) - naive-shrinkage blend of the
    #       trained model's own predictions toward the naive equal-weight
    #       combination (see apply_naive_shrinkage()'s docstring);
    #       combined with a never-worse-than-naive fallback guard below.
    diversity_penalty = float(HPARAMETERS_OPT[6]) if len(HPARAMETERS_OPT) > 6 and HPARAMETERS_OPT[6] else 0.0
    use_simplex_weights = bool(HPARAMETERS_OPT[7]) if len(HPARAMETERS_OPT) > 7 else False
    shrinkage_alpha = float(HPARAMETERS_OPT[8]) if len(HPARAMETERS_OPT) > 8 and HPARAMETERS_OPT[8] is not None else 1.0
    _model_cls = MLP_linear_simplex if use_simplex_weights else MLP_linear

    model_selected = MODEL.copy()
    if 'ensemble' in MODEL:
        model_selected.remove('ensemble')

    # ver4-4 R4.f (blueprint §2.4.2): the same three lines every other
    # torch model module already has - Linear_transformation was the one
    # weighted-ensemble method still hardcoded to whatever device torch
    # itself defaults to (CPU). Resolved from the SAME shared, config-
    # derived source (pipeline_utils.resolve_compute_resources()) as
    # MLP.py/the GAT_*.py files, so a run with TORCH_DEVICE/USE_AMP/
    # CUDNN_BENCHMARK set reaches this method too, not just the
    # per-marker-effect models.
    _resources = get_active_compute_resources()
    device = torch.device(_resources['device'])
    apply_torch_compute_settings(_resources)

    ## Change the data structure
    train = CSVDataset(data_train.loc[:,['actual']+model_selected])
    valid = CSVDataset(data_valid.loc[:,['actual']+model_selected])
    test = CSVDataset(data_test.loc[:,['actual']+model_selected])

    ## Create data loaders
    train_loader = DataLoader(train, batch_size=batch_s, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=batch_s, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_s, shuffle=False)

    def _diversity_term(m):
        """Ridge-to-uniform penalty on `m`'s own combining weights,
        normalised (L1, on absolute value) to a comparable scale
        regardless of which model class is in use - 0.0 whenever
        diversity_penalty is unset (the default), exactly reproducing
        this method's original loss."""
        if not diversity_penalty:
            return 0.0
        n_inputs = len(model_selected)
        if use_simplex_weights:
            shares = m.combining_weights()
        else:
            raw = m.hidden1.weight.flatten()
            denom = raw.abs().sum().clamp_min(1e-12)
            shares = raw.abs() / denom
        uniform = 1.0 / n_inputs
        return diversity_penalty * torch.sum((shares - uniform) ** 2)

    # ver4-4 R4.f: guard entry into GPU-dispatched work with the shared
    # GPU-slot semaphore (a no-op unless intra-batch parallelism with more
    # than one GPU-capable worker is actually enabled), mirroring every
    # other torch model module - wraps the whole ensemble-of-`num_models`
    # training loop plus the evaluation/SHAP-explainer work below it,
    # since a NEW model is created and `.to(device)`'d on every `i`.
    with gpu_slot():
        ## Implement weight optimisation
        best_loss_final = float('inf')
        best_model_weights = None
        for i in range(num_models):
            model = _model_cls(data_train.loc[:,model_selected].shape[1]).to(device)
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
                    inputs, targets = inputs.to(device), targets.to(device)
                    model.train()
                    optimizer.zero_grad()
                    yhat = model(inputs)
                    loss = F.mse_loss(yhat, targets) + _diversity_term(model)
                    loss_train_sum += loss
                    loss.backward()
                    optimizer.step()

                ## MLP validation
                for inputs, targets in valid_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
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

        ## Load the trained weights of the best model (fall back to the last trained model's
        ## weights if no validation improvement was ever recorded, e.g. all-NaN losses)
        if best_model_weights is not None:
            model.load_state_dict(best_model_weights)

        ## Evaluate model
        model.eval()

        predicted_test = []
        actual_test = []
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            yhat = model(inputs)
            predicted_test += yhat.detach().cpu().tolist()
            actual_test += targets.detach().cpu().tolist()
        predicted_test = [item for sublist in predicted_test for item in sublist]
        actual_test = [item for sublist in actual_test for item in sublist]
    
        ## Calculate metrics
        # ver4-4 R5 Fix 2: safe_regression_metrics() never raises, even when
        # this torch model's test-set predictions carry NaN (e.g. training
        # diverged) - the direct mean_squared_error()/pearsonr() pair this
        # replaces is exactly what would raise ValueError('Input contains
        # NaN') here, before any output file for this scenario is written.
        # NOTE (ver4-4 scope note, recorded in the Change Summary §7): the R5
        # design record's touch-point table for this file only lists the
        # L262-263/272-273/287-288 sites below; this initial computation has
        # identical NaN exposure (it is the very column those three sites
        # either reuse or recompute) and is included here too so the fix is
        # not defeated by an unguarded first call.
        r, mse = safe_regression_metrics(actual_test, predicted_test)

        ## Store the metrics
        record = pd.concat([record, pd.DataFrame(record.iloc[-1,:]).T]).reset_index(drop=True)
        record.loc[record.shape[0]-1,'model'] = 'Linear transformation'
        record.loc[record.shape[0]-1,'Pearson correlation'] = r
        record.loc[record.shape[0]-1,'MSE'] = mse

        ## Extract predicted and observed values for the validation set
        predicted_valid = []
        #actual_valid = []
        valid_loader = DataLoader(valid, batch_size=batch_s, shuffle=False)
        for inputs, targets in valid_loader:
            inputs = inputs.to(device)
            yhat = model(inputs)
            predicted_valid += yhat.detach().cpu().tolist()
            #actual_valid += targets.tolist()
        predicted_valid = [item for sublist in predicted_valid for item in sublist]
        #actual_valid = [item for sublist in actual_valid for item in sublist]

        ## Extract predicted and observed values for the train set
        train_loader = DataLoader(train, batch_size=batch_s, shuffle=False)
        predicted_train = []
        #actual_train = []
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            yhat = model(inputs)
            predicted_train += yhat.detach().cpu().tolist()
            #actual_train += targets.tolist()
        predicted_train = [item for sublist in predicted_train for item in sublist]
        #actual_train = [item for sublist in actual_train for item in sublist]

        # Phase 2, Requirement 5 (never-worse-than-naive guard + optional
        # naive-shrinkage blend) - applied here, at the PREDICTION level,
        # since this method (unlike Nelder_Mead.py/Bayesian_optimisation.py)
        # has no explicit weight vector for its actual predictions, only a
        # trained model. Computed on the ORIGINAL model_selected columns
        # (before they're overwritten below), on the validation split (never
        # test), mirroring the other two methods' own validation-only
        # never-worse-than-naive comparison.
        _naive_valid_pred = data_valid[model_selected].astype(float).mean(axis=1).to_numpy()
        _actual_valid_arr = data_valid['actual'].astype(float).to_numpy()
        # ver4-4 R5 Fix 2: safe_regression_metrics() never raises on a NaN
        # predicted_valid (e.g. this model's own fit diverged) - see the note
        # above this function's first metric computation. Only the MSE half
        # of the returned pair is used here; this guard has always compared
        # MSE only, never Pearson r.
        _, _naive_valid_mse = safe_regression_metrics(_actual_valid_arr, _naive_valid_pred)
        _, _trained_valid_mse = safe_regression_metrics(_actual_valid_arr, predicted_valid)

        # Update ID ver4-6, R6 (blueprint touch point: Linear_transformation
        # gets the analytic solution as an "extra_candidates entry only").
        # Unlike Nelder_Mead.py/Bayesian_optimisation.py, this method's own
        # floor ALREADY grades candidates on realised validation MSE
        # directly (the never-worse-than-naive comparison below predates
        # ver4-6 and already satisfies R5's structural requirement) - so
        # rather than routing through ensemble_regularization.py::
        # select_weights_with_floor()'s weight-VECTOR-shaped API (which
        # does not fit a TRAINED model's own predictions), the closed-form
        # analytic solution is added here as a THIRD candidate PREDICTION,
        # compared using the SAME MSE comparison this file already
        # performs. This is a disclosed, deliberate deviation from the
        # blueprint's literal architecture (preserving this file's own,
        # already-correct existing floor mechanism - see this update's
        # Change Summary §7) while still delivering R6's actual intent:
        # the analytic optimum genuinely participates as a floor candidate.
        _resources_for_wopt = get_active_compute_resources()
        w_opt_analytic_seed = bool(_resources_for_wopt.get('w_opt_analytic_seed', False))
        _pred_matrix_valid = data_valid[model_selected].astype(float).to_numpy()
        _analytic_valid_mse = np.inf
        _analytic_w = None
        if w_opt_analytic_seed:
            _analytic_w = analytic_simplex_weights(_pred_matrix_valid, _actual_valid_arr)
            _analytic_valid_pred = _pred_matrix_valid @ _analytic_w
            _, _analytic_valid_mse = safe_regression_metrics(_actual_valid_arr, _analytic_valid_pred)

        # A non-finite _trained_valid_mse (this model's OWN validation
        # predictions are NaN) must trip the never-worse-than-naive guard
        # below exactly like a genuinely worse-than-naive MSE would -
        # `float('nan') > _naive_valid_mse` is always False in plain
        # Python/NumPy comparison semantics, which would otherwise let a
        # diverged model's predictions through completely untouched instead
        # of falling back to the naive combination. Treated as "worse than
        # naive" explicitly rather than relying on NaN comparison semantics.
        _trained_beats_naive = np.isfinite(_trained_valid_mse) and _trained_valid_mse <= _naive_valid_mse
        _analytic_beats_naive = np.isfinite(_analytic_valid_mse) and _analytic_valid_mse < _naive_valid_mse
        _trained_beats_analytic = (not _analytic_beats_naive) or (_trained_valid_mse <= _analytic_valid_mse)

        if _trained_beats_naive and _trained_beats_analytic:
            # Trained model wins (or ties) both comparisons - keep its own
            # predictions, subject to the existing optional naive-shrinkage
            # blend further below.
            _wopt_winner = 'trained'
        elif _analytic_beats_naive:
            print(f"[Linear_transformation] The analytic least-squares solution's own "
                  f"validation MSE ({_analytic_valid_mse:.6g}) beat both the trained ensemble's "
                  f"({_trained_valid_mse:.6g}) and naive/equal weighting's ({_naive_valid_mse:.6g}) "
                  f"- using it for this scenario's predictions instead (ver4-6 R6 floor).")
            _pred_matrix_test = data_test[model_selected].astype(float).to_numpy()
            _pred_matrix_train = data_train[model_selected].astype(float).to_numpy()
            predicted_test = (_pred_matrix_test @ _analytic_w).tolist()
            predicted_valid = _analytic_valid_pred.tolist()
            predicted_train = (_pred_matrix_train @ _analytic_w).tolist()
            r, mse = safe_regression_metrics(actual_test, predicted_test)
            record.loc[record.shape[0]-1,'Pearson correlation'] = r
            record.loc[record.shape[0]-1,'MSE'] = mse
            _wopt_winner = 'analytic'
        else:
            print(f"[Linear_transformation] Trained ensemble's own validation MSE "
                  f"({_trained_valid_mse:.6g}) did not beat naive/equal-weight validation MSE "
                  f"({_naive_valid_mse:.6g}) - falling back to the naive equal-weight combination "
                  f"for this scenario's predictions (never-worse-than-naive guard).")
            predicted_test = data_test[model_selected].astype(float).mean(axis=1).to_numpy().tolist()
            predicted_valid = _naive_valid_pred.tolist()
            predicted_train = data_train[model_selected].astype(float).mean(axis=1).to_numpy().tolist()
            # ver4-4 R5 Fix 2 - see the initial metric computation's note above.
            r, mse = safe_regression_metrics(actual_test, predicted_test)
            record.loc[record.shape[0]-1,'Pearson correlation'] = r
            record.loc[record.shape[0]-1,'MSE'] = mse
            _wopt_winner = 'naive'

        if _wopt_winner == 'trained' and shrinkage_alpha < 1.0:
            # Requirement 5 (secondary mitigation): optional naive-shrinkage
            # blend of the (already naive-beating) trained predictions toward
            # the naive combination - a no-op when shrinkage_alpha is 1.0
            # (unset/default). Only applies when the TRAINED model itself
            # was actually selected above - shrinking an already-closed-form
            # analytic solution, or the naive fallback itself, toward naive
            # has no equivalent rationale.
            alpha = max(0.0, min(1.0, shrinkage_alpha))
            predicted_test = (alpha * pd.Series(predicted_test).to_numpy()
                               + (1 - alpha) * data_test[model_selected].astype(float).mean(axis=1).to_numpy()).tolist()
            predicted_valid = (alpha * pd.Series(predicted_valid).to_numpy() + (1 - alpha) * _naive_valid_pred).tolist()
            predicted_train = (alpha * pd.Series(predicted_train).to_numpy()
                                + (1 - alpha) * data_train[model_selected].astype(float).mean(axis=1).to_numpy()).tolist()
            # ver4-4 R5 Fix 2 - see the initial metric computation's note above.
            r, mse = safe_regression_metrics(actual_test, predicted_test)
            record.loc[record.shape[0]-1,'Pearson correlation'] = r
            record.loc[record.shape[0]-1,'MSE'] = mse

        data_test['Linear transformation'] = predicted_test
        data_valid['Linear transformation'] = predicted_valid
        data_train['Linear transformation'] = predicted_train

        ## Extract weights
        data_train_selected, data_test_selected = data_train.loc[:,model_selected], data_test.loc[:,model_selected]

        # ver4-4 R4.f: the SHAP background/explained samples must be on the
        # SAME device as the model being explained, or DeepExplainer
        # silently falls back to CPU (or raises a device-mismatch error) -
        # see MLP.py's identical note at its own DeepExplainer call.
        d_train = torch.tensor(data_train_selected.values, dtype=torch.float32).to(device)
        d_test = torch.tensor(data_test_selected.values, dtype=torch.float32).to(device)

        weight_sample = pd.DataFrame(record.iloc[record.shape[0]-1,:]).T.drop(['Pearson correlation', 'MSE'],axis=1)
        weight_sample['model'] = 'Linear transformation'

        explainer = shap.DeepExplainer(model, shap.sample(d_train, 50))
        # ver4-4 R4.f (I3): shap.DeepExplainer's own PyTorch backend already
        # materialises its return value as a plain NumPy array (it calls
        # .cpu().numpy() internally regardless of the explained model's
        # device), so no additional .cpu() is needed on this line itself -
        # matching MLP.py's own unchanged DeepExplainer call, which relies
        # on the exact same SHAP behaviour.
        weight_extracted = pd.DataFrame(abs(explainer.shap_values(shap.sample(d_test, 50),check_additivity=False)).sum(axis=0)).T
        weight_extracted = pd.DataFrame(weight_extracted)
        weight_extracted.columns = model_selected
        weight_sample = pd.concat([weight_sample.reset_index(drop=True), weight_extracted.reset_index(drop=True)], axis=1)
        weight = pd.concat([weight, weight_sample])

        ## Calculate weighted effects
        weight_extracted_normalised = _safe_row_normalize(weight_extracted)

        for i in range(len(model_selected)):
            if effect[effect['model']==model_selected[i]].shape[0] != 0:
                if i == 0:
                    effect_weighted = _safe_row_normalize(effect[effect['model']==model_selected[i]].tail(1).iloc[:,5:].abs().reset_index(drop=True)).mul(weight_extracted_normalised[model_selected[i]], axis=0).reset_index(drop=True)
                else:
                    effect_weighted += _safe_row_normalize(effect[effect['model']==model_selected[i]].tail(1).iloc[:,5:].abs().reset_index(drop=True)).mul(weight_extracted_normalised[model_selected[i]], axis=0).reset_index(drop=True)

        effect = pd.concat([effect, pd.DataFrame(effect.iloc[effect.shape[0]-1,:]).T]).reset_index(drop=True)
        effect.loc[effect.shape[0]-1,'model'] = 'Linear transformation'

        effect.loc[effect.shape[0]-1,list(effect_weighted.columns)] = effect_weighted.values

    return record, effect, data_test, data_valid, data_train, weight
