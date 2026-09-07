import pandas as pd
import numpy as np
import scipy.stats
from sklearn.metrics import mean_squared_error
from scipy.optimize import Bounds, minimize

from .ensemble_regularization import (
    diversity_regularizer, ensemble_mse_objective, apply_naive_shrinkage, read_regularization_settings,
    analytic_simplex_weights, select_weights_with_floor, stick_breaking_to_simplex,
    simplex_to_stick_breaking,
)
from pipeline_utils import get_active_compute_resources


def _safe_row_normalize(df):
    """Row-wise L1-normalize (each row divided by its own sum of - here
    already non-negative - values, e.g. abs(marker effect)) - safely: a
    row whose sum is exactly 0 (a model that assigned literally zero
    effect to every marker - e.g. SVR/KNN, which aren't additive-effect
    models and so never report per-marker effects at all) is left as
    all-zero, rather than corrupting or crashing this whole weighted
    ensemble's combined marker-effect output.

    Bug fix: this used to just be `df.div(df.sum(axis=1), axis=0).
    fillna(0)`, on the assumption that dividing by exactly 0 always
    produces NaN, which .fillna(0) then mops up. That's only true when df
    is a clean float64 DataFrame. In production this df's columns can
    come back as pandas 'object' dtype (marker-effect values assembled
    from the R-backed models - rrBLUP/BayesB/RKHS - via rpy2), and on an
    object-dtype column pandas' .div() falls back to plain Python
    arithmetic per element instead of numpy's vectorised path - and plain
    Python division by exactly 0.0 RAISES ZeroDivisionError rather than
    returning NaN, so .fillna(0) never gets a chance to run at all.
    Confirmed as the root cause of a real production crash in this exact
    function's twin in models/ensemble.py: hours of hyperparameter tuning
    across every model completed successfully, then this single division
    crashed the process during the final marker-effect combination step,
    before any output file was ever written - which is exactly why a run
    can finish all its logged steps and still produce no output files at
    all. This module's own weight-vector normalisation further down
    already guards its own, separate zero-sum case (see its own
    comments) - this is the analogous fix for the marker-effect side.

    Fixed by never attempting to divide by an exact 0 in the first place
    (replacing it with NaN before dividing, not after) and by forcing a
    clean float64 dtype regardless of what the caller passed in - either
    change alone would have prevented the crash; both together make this
    safe no matter what dtype the input arrives in."""
    df = df.astype(float)
    row_sums = df.sum(axis=1).replace(0, np.nan)
    return df.div(row_sums, axis=0).fillna(0)


def Nelder_Mead(data_train, data_valid, data_test, record, effect, weight, MODEL, HPARAMETERS_OPT):

    initial_value = HPARAMETERS_OPT[0]
    minimum_boundary = HPARAMETERS_OPT[1]
    maximum_boundary = HPARAMETERS_OPT[2]
    fatol = HPARAMETERS_OPT[3]
    xatol = HPARAMETERS_OPT[4]
    adaptive = HPARAMETERS_OPT[5]

    # Phase 2, Requirements 4/5: the published DPT-ratio objective (Eq.2)
    # reduces to a linear-fractional function of the weight vector, which
    # is always vertex-optimised over a box-constrained domain - this is
    # what produces near-one-hot weight collapse and validation-noise
    # overfitting (see ensemble_regularization.py's own module docstring
    # for the full derivation). `diversity_penalty` (a strictly-convex
    # ridge-to-uniform/entropy regulariser, REQUIRED to guarantee an
    # interior optimum for any lambda > 0) and the optional alternate
    # `objective_mode='ensemble_mse'` (a naturally interior-optimised,
    # quadratic-in-w objective) are both opt-in, appended-only entries at
    # the END of HYPERPARAMETERS_OPT - absent entirely, both default to
    # this method's exact original (pre-Phase-2) behaviour.
    _reg = read_regularization_settings(HPARAMETERS_OPT, start_index=6)
    diversity_penalty = _reg['diversity_penalty']
    objective_mode = _reg['objective_mode']
    diversity_method = _reg['diversity_method']
    shrinkage_alpha = _reg['alpha']

    # ver4-5 R1.d (blueprint §3.4) - appended-only, index 10 (regularisation
    # occupies 6-9, via read_regularization_settings(start_index=6) above -
    # see that call's own comment), read defensively (I5) so a pre-ver4-5
    # config (HYPERPARAMETERS_OPT 10 elements long, or shorter) keeps
    # working unchanged. No restarts/batch parallelism for this algorithm
    # (Nelder-Mead's own multi-restart loop below stays exactly as it is,
    # serial - see this file's own module-level note in the blueprint's
    # touch-point table: "R1.d only. No parallelism.").
    w_opt_vectorised_objective = bool(HPARAMETERS_OPT[10]) if len(HPARAMETERS_OPT) > 10 else True
    
    model_selected = MODEL.copy()
    if 'ensemble' in MODEL:
        model_selected.remove('ensemble')

    # Bug fix: force every model-prediction column this function's arithmetic
    # touches onto a clean float64 dtype before the weight search begins -
    # see models/Bayesian_optimisation.py's identical fix (same function
    # shape, same R-interop root cause) for the full explanation of why an
    # object-dtype column turns an all-zero-weights candidate's division
    # into a raised ZeroDivisionError instead of a silent inf/nan.
    data_train = data_train.copy()
    data_valid = data_valid.copy()
    data_test = data_test.copy()
    for _df in (data_train, data_valid, data_test):
        _df[model_selected + ['actual']] = _df[model_selected + ['actual']].astype(float)

    # ver4-5 R1.d precompute (vectorised objective, default on) - see
    # models/Bayesian_optimisation.py::_bayesian_single_search()'s own
    # module-level docstring for the full algebraic derivation (this
    # method's objective is the SAME Diversity Prediction Theorem ratio,
    # just minimised directly here instead of maximised via a reciprocal
    # target). An inert precompute (built but never read) when
    # w_opt_vectorised_objective is False.
    pred_matrix = data_valid[model_selected].to_numpy(dtype=float)
    actual_valid_arr = data_valid['actual'].to_numpy(dtype=float)
    _row_mean_pred = pred_matrix.mean(axis=1)
    e_minus_d = (pred_matrix - actual_valid_arr[:, None]) ** 2 - (pred_matrix - _row_mean_pred[:, None]) ** 2
    _K = len(model_selected)

    # ver4-4 §4a / ver4-5 R1.d precedent - see
    # models/Bayesian_optimisation.py's identical announcement for why
    # this default-on, numerics-changing acceleration is logged
    # unconditionally (I11) from here rather than GP()'s own shared
    # "[GP] NUMERICS:" block (this key lives in this method's own
    # positional HYPERPARAMETERS_OPT list, not in resolve_compute_
    # resources()'s _compute_cfg).
    if w_opt_vectorised_objective:
        print("[GP] NUMERICS: Nelder-Mead weight-optimisation objective is VECTORISED "
              "(W_OPT_VECTORISED_OBJECTIVE=True, default) - agrees with the original "
              "loop-based computation to 1e-12 relative, but floating-point summation "
              "order differs. Set W_OPT_VECTORISED_OBJECTIVE=false to restore the exact "
              "pre-ver4-5 computation.")

    # Update ID ver4-6 (R5/R6/R7) - three further top-level config flags,
    # read via the SAME process-global compute-resources dict
    # models/Bayesian_optimisation.py already reads from - NOT a
    # positional HYPERPARAMETERS_OPT entry, deliberately, so all three
    # weight-optimisation methods share ONE switch each. See
    # pipeline_utils.resolve_compute_resources()'s own docstring for each
    # key's default/rationale.
    _resources = get_active_compute_resources()
    w_opt_analytic_seed = bool(_resources.get('w_opt_analytic_seed', False))
    w_opt_validation_floor = bool(_resources.get('w_opt_validation_floor', False))
    w_opt_simplex_search = bool(_resources.get('w_opt_simplex_search', True))

    _objective_source = 'explicit HYPERPARAMETERS_OPT entry' if len(HPARAMETERS_OPT) > 7 else \
        'default (ver4-6: ensemble_mse fallback - see read_regularization_settings)'
    print(f"[GP] Nelder Mead weight search: objective_mode={objective_mode!r} "
          f"({_objective_source}), w_opt_analytic_seed={w_opt_analytic_seed}, "
          f"w_opt_validation_floor={w_opt_validation_floor}, "
          f"w_opt_simplex_search={w_opt_simplex_search}.")

    K = len(model_selected)
    # ver4-6 R7.2(b): search K-1 dimensions on the unit cube and map onto
    # the K-dimensional probability simplex via a stick-breaking
    # construction, eliminating the raw K-dimensional box's own scale-
    # invariant flat direction entirely - see
    # models/Bayesian_optimisation.py::_bayesian_single_search's own
    # module-level docstring for the full rationale (shared verbatim by
    # both methods). Falls back to False whenever there are fewer than 2
    # base models (nothing to stick-break).
    use_simplex = bool(w_opt_simplex_search) and K >= 2
    search_dims = (K - 1) if use_simplex else K

    if use_simplex and minimum_boundary < 0:
        print(f"[GP] NOTE: Nelder Mead weight search - Minimum boundary={minimum_boundary} "
              f"is negative, but W_OPT_SIMPLEX_SEARCH=True means every candidate weight is "
              f"non-negative by construction (a stick-breaking map onto the probability "
              f"simplex - ver4-6 R7.2(b)); negative weights are unreachable regardless of "
              f"this setting. Set W_OPT_SIMPLEX_SEARCH=False to restore the literal "
              f"per-weight box semantics.")

    # ver4-6 R6.2: the closed-form least-squares solution, computed ONCE
    # (deterministic) - used as one restart's own starting point below
    # (R6.4) and handed to the validation floor as an extra candidate.
    analytic_w = analytic_simplex_weights(pred_matrix, actual_valid_arr) if w_opt_analytic_seed else None

    # Update ID ver4-6, R2.2: WORST_TERM is now a FINITE-VALUE CLIP only
    # (dropped from 1e6 to 100.0 - the live objective range for both
    # `dpt_ratio` and `ensemble_mse` is a small, bounded quantity, so
    # 100.0 stays far outside anything real while no longer being six
    # orders of magnitude off). A failed/degenerate candidate is instead
    # scored via FAILURE_CEILING/_failure_score() below - the MIRROR
    # IMAGE, for a MINIMISED objective, of models/hyperparameter_tuning.
    # py's own worst-finite-relative failure scoring (see that module's
    # own R2.2 design note for the full rationale): "worse" here means
    # LARGER, not smaller, and a direct (non-surrogate) search like
    # Nelder-Mead is less vulnerable to a poisoned GP, but a fixed
    # extreme sentinel still creates a cliff the simplex can walk into
    # and contract around, which the SAME relative scoring removes.
    WORST_TERM = 100.0
    FAILURE_CEILING = 10.0
    FAILURE_MARGIN = 0.05
    _state = {'worst_finite': None}

    def _failure_score():
        w = _state['worst_finite']
        if w is None:
            return FAILURE_CEILING
        return w + max(FAILURE_MARGIN, 0.10 * abs(w))

    def _clip_shares(shares):
        """ver4-6 R7.2(b): Minimum/Maximum boundary reinterpreted as a
        post-hoc clip on the resulting shares - only reached under
        `use_simplex`."""
        clipped = np.clip(shares, minimum_boundary, maximum_boundary)
        total = float(clipped.sum())
        if total > 0 and np.isfinite(total):
            return clipped / total
        return shares

    ## Define the objective function to minimise based on the Diversity Prediction Theorem
    def _raw_objective(weights):

       # Bug fix: guard the weight vector BEFORE it's used as a divisor -
       # a candidate can legitimately land on (or effectively at, given
       # floating-point precision) all-zero weights - e.g. every dimension
       # pinned near 'Minimum boundary' - and weights.sum() dividing by
       # exactly 0 is the direct trigger for this function's
       # ZeroDivisionError in practice (see the dtype-coercion comment
       # above for why object-dtype columns turn that into a raised
       # exception rather than a silent inf/nan). Scored as a failure
       # (see _failure_score() above) and skipped, exactly like any other
       # degenerate candidate.
       w_sum = weights.sum()
       if not np.isfinite(w_sum) or w_sum == 0:
           return _failure_score()

       if objective_mode == 'ensemble_mse':
           # Requirement 5 (§4.3b): alternate, naturally interior-optimised
           # objective - the ensemble prediction's own realised MSE against
           # the actual validation target, quadratic in `w`.
           first_term = ensemble_mse_objective(
               weights, w_sum, data_valid, model_selected,
               pred_matrix=(pred_matrix if w_opt_vectorised_objective else None),
               actual=(actual_valid_arr if w_opt_vectorised_objective else None),
           )
       elif w_opt_vectorised_objective:
           # ver4-5 R1.d - single float64 matrix-vector product + a mean,
           # replacing the loop-based computation below. Verified to
           # agree with it to 1e-12 relative (see this update's Change
           # Summary §10).
           first_term = float(np.mean(e_minus_d @ weights) / (_K * w_sum))
       else:
           second_term = data_valid.loc[:,model_selected+['actual']]
           third_term, mean_value = second_term.copy(), second_term.iloc[:,:-1].mean(axis=1)

           for i in range(len(model_selected)):
               second_term.iloc[:,i] = (second_term.iloc[:,i] - second_term['actual']).pow(2)*weights[i]
               third_term.iloc[:,i] = (third_term.iloc[:,i] - mean_value).pow(2)*weights[i]

           second_term = second_term.iloc[:,:-1].div(w_sum,axis=1).mean(axis=1)
           third_term = third_term.iloc[:,:-1].div(w_sum,axis=1).mean(axis=1)
           first_term = (second_term - third_term).mean()

       # Requirement 5 (§4.3a, REQUIRED structural fix, not optional
       # tuning): add the strictly-convex diversity/entropy regulariser
       # BEFORE returning - this is what breaks the linear-fractional
       # objective's vertex-only-optimum shape and guarantees an interior
       # optimum exists for any diversity_penalty > 0. A no-op (adds
       # exactly 0.0) when diversity_penalty is 0/unset, so every existing
       # config's behaviour is completely unchanged unless this is
       # explicitly turned on.
       first_term = first_term + diversity_regularizer(weights, w_sum, diversity_penalty, diversity_method)

       if not np.isfinite(first_term):
           score = _failure_score()
       else:
           score = float(np.clip(first_term, -WORST_TERM, WORST_TERM))
           _state['worst_finite'] = (score if _state['worst_finite'] is None
                                      else max(_state['worst_finite'], score))
       return score

    def optimisation(x):
       x = np.asarray(x, dtype=float)
       if use_simplex:
           shares = _clip_shares(stick_breaking_to_simplex(x))
           return _raw_objective(shares)
       return _raw_objective(x)

    ## Define the range of weights
    if use_simplex:
        bounds = Bounds([0.0] * search_dims, [1.0] * search_dims)
    else:
        bounds = Bounds([minimum_boundary]*search_dims, [maximum_boundary]*search_dims)

    # Bug fix (weighted ensembles failing to outperform - or even match -
    # simple equal weighting), two changes:
    #
    # 1. Multi-restart, mirroring models/hyperparameter_tuning.py's own
    #    _multistart_local_search() for its Nelder-Mead/Powell model-
    #    hyperparameter searches: a single Nelder-Mead run from one
    #    starting point can converge to a local optimum it never escapes,
    #    simply because of where it happened to start - restarting from
    #    several different points and keeping the best mitigates this,
    #    and costs little here since evaluating a weight candidate is
    #    cheap (it re-weights already-computed base-model predictions, no
    #    model refitting involved).
    # 2. One restart is explicitly anchored at equal weighting (every
    #    model contributing the same share). Under the pre-ver4-6
    #    (non-simplex) box search this is the SAME point
    #    x0=[initial_value]*n already started from exclusively before
    #    this fix (the objective is invariant to uniformly scaling the
    #    whole weight vector - see the ZeroDivisionError-guard comments
    #    above for why - so 'Initial value' already meant "start from
    #    equal weighting" whether or not that was the original intent).
    #    Clipped into bounds in case 'Initial value' sits outside
    #    [minimum_boundary, maximum_boundary]. Under ver4-6 R7.2(b)
    #    simplex search, the equal-weighting restart is instead the
    #    stick-breaking encoding of the TRUE 1/K share vector.
    _equal_weight_val = float(np.clip(initial_value, minimum_boundary, maximum_boundary))
    _equal_weights = np.array([_equal_weight_val] * K)  # non-simplex restart anchor (byte-identical to ver4-5)
    _equal_shares = np.full(K, 1.0 / K)                  # TRUE equal weighting - used for scoring/the floor

    if use_simplex:
        _equal_x0 = simplex_to_stick_breaking(_equal_shares)
    else:
        _equal_x0 = _equal_weights

    N_RESTARTS = 5
    rng = np.random.default_rng(0)
    best_res, best_score = None, np.inf
    for _i in range(N_RESTARTS):
        if _i == 0:
            x0 = _equal_x0
        elif _i == 1 and analytic_w is not None:
            # ver4-6 R6.4: one restart explicitly anchored at the
            # closed-form least-squares solution, on the same footing as
            # the equal-weighting anchor above.
            x0 = simplex_to_stick_breaking(np.asarray(analytic_w, dtype=float)) if use_simplex \
                else np.asarray(analytic_w, dtype=float)
        elif use_simplex:
            x0 = rng.uniform(0.0, 1.0, size=search_dims)
        else:
            x0 = rng.uniform(minimum_boundary, maximum_boundary, size=search_dims)
        res = minimize(optimisation, x0, method='nelder-mead',
               options={'disp': False,
                        'fatol': fatol, 'xatol': xatol, "adaptive": adaptive},
               bounds=bounds,
               )
        if res.fun < best_score:
            best_score, best_res = res.fun, res
    pos = best_res

    # Decode the winning restart's own search-space point back into a
    # plain, K-length weight-SHARE vector - from here on, every downstream
    # step (the floor, the legacy abs/normalise/shrink rollback path, and
    # weight_extracted itself) operates on this same, consistent shape
    # regardless of which search space (K-dim box, or ver4-6's K-1-dim
    # stick-breaking simplex) actually produced it.
    pos_weights = _clip_shares(stick_breaking_to_simplex(pos.x)) if use_simplex \
        else np.asarray(pos.x, dtype=float)

    ## Weight extraction (ver4-6 R5.2b/R6.2/R7.2(c)).
    if w_opt_validation_floor:
        # Every candidate - the search's own winner (`pos_weights`),
        # equal weighting, and (when W_OPT_ANALYTIC_SEED) the R6 analytic
        # solution - is normalised and graded on its OWN realised
        # validation MSE via select_weights_with_floor(), which is ALSO
        # where naive shrinkage (Requirement 5) is applied. This makes
        # "never worse than the naive ensemble on validation" a
        # construction-level guarantee, replacing the old
        # `equal_score <= best_score` + separate abs()/normalise/shrink
        # block entirely (R7.2(c): grading and applying - and now
        # recording into Weight.csv - the SAME vector removes the sign/
        # normalisation mismatch R7.1(c) named).
        _extra_candidates = [('analytic', analytic_w)] if analytic_w is not None else None
        pos.x, _label, _diag = select_weights_with_floor(
            pos_weights, pred_matrix, actual_valid_arr, shrinkage_alpha=shrinkage_alpha,
            extra_candidates=_extra_candidates,
        )
        print(f"[GP] Nelder Mead weight search: floor selected {_label!r} (validation MSE - {_diag}).")
        weight_extracted = pd.DataFrame(pos.x).T
        weight_extracted.columns = model_selected
    else:
        # ver4-5-equivalent path (rollback: W_OPT_VALIDATION_FLOOR=False)
        # - byte-for-byte the pre-ver4-6 code (when W_OPT_SIMPLEX_SEARCH
        # is ALSO False - see §8 of the ver4-6 Change Summary for the
        # full rollback config), including weight_extracted being built
        # from the RAW pre-normalisation search winner.
        #
        # Explicitly compare the multi-restart search's own winner
        # against equal weighting's own score (evaluated fresh, exactly
        # as any other candidate would be - not just relying on it
        # having been ONE of the restarts above), and fall back to equal
        # weighting if the search's winner didn't actually beat it. This
        # GUARANTEES the result below is never worse than plain equal
        # weighting on THIS objective (though see R5's own finding for
        # why that guarantee is weaker than it sounds under
        # `dpt_ratio` - the floor above is the structural fix).
        equal_score = _raw_objective(_equal_shares if use_simplex else _equal_weights)
        if equal_score <= best_score:
            pos.x = _equal_weights if not use_simplex else list(_equal_shares)
        else:
            pos.x = list(pos_weights)

        weight_extracted = pd.DataFrame(pos.x).T
        weight_extracted.columns = model_selected

        # Requirement (safeguard): fall back to equal weighting if the
        # minimiser converged to all-zero weights (sum(pos.x)==0) - rather
        # than raising ZeroDivisionError on the plain-Python division below.
        # `> 0` (not `!= 0`) also catches a NaN sum - `math.nan != 0` is True
        # in Python, so the old check let a NaN _sum_abs_x straight through
        # into `x / _sum_abs_x`, silently producing all-NaN weights instead of
        # falling back to equal weighting the way a zero sum already did.
        _abs_x = [abs(x) for x in pos.x]
        _sum_abs_x = sum(_abs_x)
        if _sum_abs_x > 0 and np.isfinite(_sum_abs_x):
            pos.x = [x / _sum_abs_x for x in _abs_x]
        else:
            pos.x = [1.0 / len(_abs_x)] * len(_abs_x)

        # Requirement 5 (secondary mitigation): optional naive-shrinkage blend
        # toward equal weighting, applied on top of (not instead of) the
        # diversity regulariser above - a no-op when shrinkage_alpha is 1.0
        # (unset/default), so existing behaviour is unaffected unless this is
        # explicitly configured.
        pos.x = apply_naive_shrinkage(pos.x, shrinkage_alpha)
    
    ## Weight the predicted phenotypes for the test set
    data_test_selected = data_test.loc[:,model_selected+['actual']]
    for i in range(len(data_test_selected.columns)-1):
        data_test_selected.iloc[:,i] = data_test_selected.iloc[:,i] * pos.x[i]
    predicted_test =  data_test_selected.iloc[:,:-1].sum(axis=1).reset_index(drop=True)
    actual_test =  data_test_selected.iloc[:,-1].tolist()
    
    ## calculate the metrics
    mse = mean_squared_error(actual_test, predicted_test)
    r = scipy.stats.pearsonr(actual_test, predicted_test)[0]
    
    ## Store the metrics
    record = pd.concat([record, pd.DataFrame(record.iloc[-1,:]).T]).reset_index(drop=True)
    record.loc[record.shape[0]-1,'model'] = 'Nelder Mead'
    record.loc[record.shape[0]-1,'Pearson correlation'] = r
    record.loc[record.shape[0]-1,'MSE'] = mse

    ## Weight the predicted phenotypes for the validation set
    predicted_valid = []
    data_valid_selected = data_valid.loc[:,model_selected+['actual']]
    for i in range(len(data_valid_selected.columns)-1):
        data_valid_selected.iloc[:,i] = data_valid_selected.iloc[:,i] * pos.x[i]
    
    ## Calculate the weighted average
    predicted_valid =  data_valid_selected.iloc[:,:-1].sum(axis=1).reset_index(drop=True)
    #actual_valid =  data_valid_selected.iloc[:,-1].tolist()
    
    ## Weight the predicted phenotypes for the training set
    predicted_train = []
    data_train_selected = data_train.loc[:,model_selected+['actual']]
    for i in range(len(data_train_selected.columns)-1):
        data_train_selected.iloc[:,i] = data_train_selected.iloc[:,i] * pos.x[i]
    predicted_train =  data_train_selected.iloc[:,:-1].sum(axis=1).reset_index(drop=True)
    #actual_train =  data_train_selected.iloc[:,-1].tolist()
    data_test['Nelder-Mead'] = predicted_test
    data_valid['Nelder-Mead'] = predicted_valid
    data_train['Nelder-Mead'] = predicted_train

    ## Extract weights
    weight_sample = pd.DataFrame(record.iloc[record.shape[0]-1,:]).T.drop(['Pearson correlation', 'MSE'],axis=1)
    weight_sample['model'] = 'Nelder Mead' 
    
    weight_sample = pd.concat([weight_sample.reset_index(drop=True), weight_extracted.reset_index(drop=True)], axis=1)
    weight = pd.concat([weight, weight_sample])
    
    weight_extracted_normalised = _safe_row_normalize(weight_extracted)
    
    ## Calculate weighted effects
    for i in range(len(model_selected)):
        if effect[effect['model']==model_selected[i]].shape[0] != 0:
            if i == 0:
                effect_weighted = _safe_row_normalize(effect[effect['model']==model_selected[i]].tail(1).iloc[:,5:].abs().reset_index(drop=True)).mul(weight_extracted_normalised[model_selected[i]], axis=0).reset_index(drop=True)
            else:
                effect_weighted += _safe_row_normalize(effect[effect['model']==model_selected[i]].tail(1).iloc[:,5:].abs().reset_index(drop=True)).mul(weight_extracted_normalised[model_selected[i]], axis=0).reset_index(drop=True)

    effect = pd.concat([effect, pd.DataFrame(effect.iloc[effect.shape[0]-1,:]).T]).reset_index(drop=True)
    effect.loc[effect.shape[0]-1,'model'] = 'Nelder Mead'
    
    effect.loc[effect.shape[0]-1,list(effect_weighted.columns)] = effect_weighted.values

    return record, effect, data_test, data_valid, data_train, weight