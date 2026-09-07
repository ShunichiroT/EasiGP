import pandas as pd
import numpy as np
import scipy.stats
from sklearn.metrics import mean_squared_error
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer

from .ensemble_regularization import (
    diversity_regularizer, ensemble_mse_objective, apply_naive_shrinkage, read_regularization_settings,
    analytic_simplex_weights, select_weights_with_floor, stick_breaking_to_simplex,
    simplex_to_stick_breaking,
)
from pipeline_utils import get_active_compute_resources, nested_safe_n_jobs

try:
    from bayes_opt.acquisition import ExpectedImprovement
except ImportError:  # pragma: no cover - already a project dependency
    ExpectedImprovement = None


def _safe_row_normalize(df):
    """Row-wise L1-normalize (each row divided by its own sum of - here
    already non-negative - values, e.g. abs(marker effect) or SHAP-based
    importance) - safely: a row whose sum is exactly 0 (a model that
    assigned literally zero effect/importance to everything - e.g. SVR/
    KNN, which aren't additive-effect models and so never report per-
    marker effects at all) is left as all-zero, rather than corrupting or
    crashing this whole weighted ensemble's combined marker-effect output.

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


# Update ID ver4-6, R7.2(a): the objective below no longer computes
# `target = 1/first_term` (an unbounded, pole-bearing landscape whenever
# first_term is near 0 - see R2's own note in
# models/hyperparameter_tuning.py for why that shape is exactly E2's
# GP-flattening failure mode) - it now computes `target = -first_term`
# directly: monotone in the same direction (still MAXIMISED by the
# search), bounded, no pole, far better-conditioned for a Gaussian-
# process surrogate. TARGET_CAP is retasked, purely as a FINITE-VALUE
# CLIP on this now-naturally-small-magnitude target (dropped from 1e6 to
# 1e3 - still far outside anything a real target can reach, but no
# longer six orders of magnitude off). It is NEVER written to
# `Weight.csv`, `Metric.csv`, or `hyperparameter.csv` - purely an
# internal search-time value.
TARGET_CAP = 1e3

# Update ID ver4-6, R2.2: mirrors models/hyperparameter_tuning.py's own
# relative, worst-finite-score failure scoring exactly (see that
# module's own FAILURE_FLOOR/FAILURE_MARGIN comment for the full
# rationale) - a failed/degenerate candidate here (w_sum invalid, or a
# non-finite first_term) is scored strictly worse than every FINITE
# target actually observed so far THIS restart, never as a fixed extreme
# constant, so it can never distort this search's own Gaussian-process
# surrogate the way an unconditional `-TARGET_CAP` sentinel could (E2's
# failure mode, mirrored here for the weight-search objective).
FAILURE_FLOOR = -10.0
FAILURE_MARGIN = 0.05


## Customise inout parameters (weights) for the Bayesian optimisation
#
# HISTORY / WHY THIS IS NO LONGER A CUSTOM MULTI-DIMENSIONAL BayesParameter:
# this used to be one custom parameter ('weights', a single bayes_opt
# BayesParameter subclass covering all len(model_selected) dimensions at
# once, with its own Dirichlet-based random_sample()). It's been replaced
# below by len(model_selected) independent, ordinary FloatParameter
# dimensions (bayes_opt's own standard 'name: (low, high)' pbounds entries)
# so that Sequential Domain Reduction (bounds_transformer) can be enabled -
# bayes_opt's SequentialDomainReductionTransformer explicitly rejects any
# non-FloatParameter ("Domain reduction is only supported for
# all-FloatParameter optimization"), and that rejection turns out to be
# load-bearing, not overly cautious: its internal _create_bounds() zips
# target_space.keys (length 1 for the old single 'weights' parameter)
# against target_space.bounds's rows (length len(model_selected), since
# TargetSpace flattens a multi-dimensional custom parameter's bounds to
# one row per underlying dimension but keeps only ONE key for it) - a
# fundamental one-key-to-many-rows mismatch that only produces the
# correct, fully-shaped result when there's truly one FloatParameter per
# bounds row, which the old design never satisfied. There would be no
# reliable way to enable domain reduction correctly without this change.
#
# The one behavioural difference this carries: the len(model_selected)
# random init_points draws immediately below now come from bayes_opt's
# own standard per-dimension independent uniform sampling instead of the
# old Dirichlet-simplex sampling. Since the objective is invariant to
# uniformly scaling the whole weight vector (see optimisation()'s
# ZeroDivisionError-guard comments for why), what actually matters for
# exploration is the RATIO between models' weights, not their absolute
# scale - independent uniform sampling still explores a full range of
# ratios, just via evenly-spread hypercube sampling rather than a
# distribution that (by construction) concentrates more probability
# toward sparse, one-or-two-model-dominated corners of the simplex. The
# explicit equal-weighting probe below (unaffected by this change) is
# what actually guarantees a balanced starting point is always evaluated,
# regardless of how init_points happen to land.


# ---------------------------------------------------------------------------
# ver4-5 blueprint R1.d/e/f (§3.4) - one complete, independent weight-
# search restart. MODULE-LEVEL (picklable - invariant I13) so it can be
# dispatched to a separate worker PROCESS by Bayesian() below when
# W_OPT_BAYES_RESTARTS > 1 (R1.e); built from PLAIN DATA arguments only
# (floats, strings, a DataFrame, numpy arrays) - no closure over
# anything Bayesian() itself only has locally.
#
# R1.d (vectorised objective): the published Diversity Prediction
# Theorem ratio objective (Eq.2, see ensemble_regularization.py's own
# module docstring for the full derivation) reduces algebraically to
#     first_term = mean_over_rows( ((E - D) @ w) / (K * w.sum()) )
# where E[n,k] = (pred[n,k] - actual[n])**2 and
# D[n,k] = (pred[n,k] - rowmean_pred[n])**2 - both PRECOMPUTED ONCE by
# Bayesian() below (they depend only on the validation predictions,
# never on the candidate weight vector `w`), so each candidate's own
# evaluation becomes a single float64 matrix-vector product plus a
# mean, replacing the original per-model Python `for` loop over pandas
# columns. Algebraically identical to the original formulation (each
# model's contribution is scaled by w_i/w_sum, then averaged over the K
# models, then averaged over rows) - verified to agree with the
# original loop-based computation to 1e-12 relative (see this update's
# Change Summary §10). `vectorised=False` keeps the ORIGINAL loop-based
# computation, byte-for-byte, as an explicit escape hatch
# (W_OPT_VECTORISED_OBJECTIVE=False) for anyone who wants the exact
# pre-ver4-5 floating-point summation order.
#
# R1.f (batch suggestion): when `use_batch` is True, this restart's own
# search reuses models/parallel_search.py::batch_bayesian_maximise()
# (the SAME constant-liar machinery R1.b uses for hyperparameter
# tuning) instead of bayes_opt's own strictly-sequential
# optimizer.maximize(). Unlike R1.b, this candidate BATCH is evaluated
# SERIALLY, in plain Python, never via a joblib process fan-out - this
# objective is pandas/numpy arithmetic on ALREADY-FITTED predictions
# (microseconds per candidate - see the blueprint's own §3.1 cost
# table), so a process pool would add pure overhead for zero benefit
# (Option H in the blueprint's own option survey, §3.3). The value of
# batch mode here is entirely in REDUCING THE NUMBER OF GAUSSIAN-
# PROCESS SURROGATE REFITS needed to reach the same evaluation budget
# (GP refit cost grows superlinearly with the number of observations) -
# not in using more CPU cores. Because evaluation never leaves this
# process, R1.f composes FREELY with R1.e's own process-level restart
# fan-out with no additional oversubscription risk (unlike R1.b/
# hyperparameter tuning, where each trial IS an expensive model fit and
# the two levels genuinely compete for cores - see R1.h). The batch
# width is a fixed q=4 (not a separate configurable key - the blueprint
# lists W_OPT_BAYES_BATCH as boolean only, §3.6), matching
# HP_TUNE_BAYES_BATCH_MAX's own default.
# ---------------------------------------------------------------------------
def _bayesian_single_search(seed, model_selected, minimum_boundary, maximum_boundary,
                             point_num, iteration, duplicate_points,
                             objective_mode, diversity_penalty, diversity_method,
                             vectorised, data_valid, pred_matrix, actual_valid_arr,
                             e_minus_d, equal_weight_val, use_batch, liar,
                             simplex_search=True, analytic_probe=None):
    """Returns (best_target, best_weights) - a plain float and a 1-D
    numpy array of length ``len(model_selected)`` ALWAYS expressed in raw,
    len(model_selected)-dimensional weight-SHARE space (regardless of
    whether the search itself ran in that space directly, or via the
    ver4-6 R7.2(b) stick-breaking reparameterisation below - this
    function's own RETURN CONTRACT is unchanged either way), both
    trivially picklable back to the parent process.

    Parameters
    ----------
    simplex_search : bool, default True (ver4-6 R7.2(b), gated by
        config key W_OPT_SIMPLEX_SEARCH). True searches K-1 dimensions on
        the unit cube and maps every candidate onto the K-dimensional
        probability simplex via a stick-breaking construction
        (ensemble_regularization.stick_breaking_to_simplex) - symmetric,
        and eliminates the raw K-dimensional box's own scale-invariant
        flat direction entirely (R7.1(b): the objective depends only on
        weights/sum(weights), so one whole dimension of the raw box is a
        flat manifold the surrogate must waste capacity modelling, for a
        direction that can never matter). `minimum_boundary`/
        `maximum_boundary` are reinterpreted as a POST-HOC CLIP applied to
        the resulting shares (renormalised afterwards), not literal
        per-weight bounds - see `_clip_shares()` below. False restores
        the pre-ver4-6 K-dimensional box search byte-for-byte. Silently
        treated as False whenever there are fewer than 2 base models
        (nothing to stick-break).
    analytic_probe : optional (K,)-length raw share vector (ver4-6 R6.2's
        `analytic_simplex_weights(...)` result) - probed alongside the
        equal-weighting point, on the SAME footing, so the search always
        evaluates the closed-form least-squares optimum too, not merely
        uses it as a floor after the fact (see Bayesian()'s own R5/R6
        integration below).
    """
    K = len(model_selected)
    use_simplex = bool(simplex_search) and K >= 2
    search_dims = (K - 1) if use_simplex else K
    names = [f'w{i}' for i in range(search_dims)]

    # Tracks the best weight vector as the optimiser evaluates it -
    # identical in spirit to the pre-ver4-5 Bayesian()'s own closure-
    # level `best` dict, just scoped to THIS restart (each restart, in
    # whichever process it runs, tracks its own independently).
    best = {'target': -np.inf, 'weights': None}

    # Update ID ver4-6, R2.2: mirrors models/hyperparameter_tuning.py's
    # own worst-finite-relative failure scoring exactly (see this
    # module's own FAILURE_FLOOR/FAILURE_MARGIN comment above for the
    # full rationale).
    state = {'worst_finite': None}

    def _failure_score():
        w = state['worst_finite']
        if w is None:
            return FAILURE_FLOOR
        return w - max(FAILURE_MARGIN, 0.10 * abs(w))

    def _clip_shares(shares):
        """ver4-6 R7.2(b): Minimum/Maximum boundary reinterpreted as a
        post-hoc clip on the resulting shares (only reached under
        `use_simplex` - the non-simplex box path clips via bayes_opt's
        own pbounds instead, exactly as before)."""
        clipped = np.clip(shares, minimum_boundary, maximum_boundary)
        total = float(clipped.sum())
        if total > 0 and np.isfinite(total):
            return clipped / total
        return shares

    def optimisation(weights):
        w_sum = weights.sum()
        if not np.isfinite(w_sum) or w_sum == 0:
            target = _failure_score()
            if target > best['target']:
                best['target'] = target
                best['weights'] = np.asarray(weights, dtype=float).copy()
            return target

        if objective_mode == 'ensemble_mse':
            first_term = ensemble_mse_objective(
                weights, w_sum, data_valid, model_selected,
                pred_matrix=(pred_matrix if vectorised else None),
                actual=(actual_valid_arr if vectorised else None),
            )
        elif vectorised:
            # R1.d - see this function's own module-level note above for
            # the full derivation.
            first_term = float(np.mean(e_minus_d @ weights) / (K * w_sum))
        else:
            # Original (pre-ver4-5), byte-for-byte loop-based computation.
            second_term = data_valid.loc[:, model_selected + ['actual']]
            third_term, mean_value = second_term.copy(), second_term.iloc[:, :-1].mean(axis=1)
            for i in range(K):
                second_term.iloc[:, i] = (second_term.iloc[:, i] - second_term['actual']).pow(2) * weights[i]
                third_term.iloc[:, i] = (third_term.iloc[:, i] - mean_value).pow(2) * weights[i]
            second_term = second_term.iloc[:, :-1].div(w_sum, axis=1).mean(axis=1)
            third_term = third_term.iloc[:, :-1].div(w_sum, axis=1).mean(axis=1)
            first_term = (second_term - third_term).mean()

        first_term = first_term + diversity_regularizer(weights, w_sum, diversity_penalty, diversity_method)

        if not np.isfinite(first_term):
            target = _failure_score()
        else:
            # Update ID ver4-6, R7.2(a): maximise -first_term (bounded,
            # no pole) instead of the old 1/first_term - see this
            # module's own TARGET_CAP comment above for why the
            # reciprocal was E2's GP-flattening failure mode all over
            # again. `first_term == 0` is now simply a valid, finite
            # target (0.0) rather than a special-cased failure - the
            # entire "divide by exactly 0" pathology this guarded against
            # no longer exists once the reciprocal itself is gone.
            target = float(np.clip(-first_term, -TARGET_CAP, TARGET_CAP))
            state['worst_finite'] = (target if state['worst_finite'] is None
                                      else min(state['worst_finite'], target))

        if target > best['target']:
            best['target'] = target
            best['weights'] = np.asarray(weights, dtype=float).copy()
        return target

    def objective_1d(x):
        x = np.asarray(x, dtype=float)
        if use_simplex:
            shares = _clip_shares(stick_breaking_to_simplex(x))
            return optimisation(shares)
        return optimisation(x)

    if use_simplex:
        bounds_list = [(0.0, 1.0)] * search_dims
    else:
        bounds_list = [(minimum_boundary, maximum_boundary)] * search_dims

    _window_frac = 0.02

    def _make_bounds_transformer():
        if use_simplex:
            return SequentialDomainReductionTransformer(minimum_window=[_window_frac] * search_dims)
        return SequentialDomainReductionTransformer(
            minimum_window=_window_frac * (maximum_boundary - minimum_boundary))

    def _make_acquisition():
        return ExpectedImprovement(xi=0.01) if ExpectedImprovement is not None else None

    # ver4-6 R6.2/R6.4: the equal-weighting probe, PLUS - when supplied -
    # the analytic least-squares solution, on the same footing, encoded
    # into whichever search space this restart is actually using.
    equal_shares = np.full(K, 1.0 / K)
    if use_simplex:
        probe_points = [simplex_to_stick_breaking(equal_shares)]
        if analytic_probe is not None:
            probe_points.append(simplex_to_stick_breaking(np.asarray(analytic_probe, dtype=float)))
    else:
        probe_points = [np.full(K, equal_weight_val)]
        if analytic_probe is not None:
            probe_points.append(np.asarray(analytic_probe, dtype=float))

    if use_batch:
        from .parallel_search import batch_bayesian_maximise

        def objective_batch(xs):
            return [objective_1d(x) for x in xs]

        try:
            batch_bayesian_maximise(
                objective_batch, bounds_list,
                n_iter=iteration, init_points=point_num, seed=seed, q=4,
                probe_points=probe_points,
                acquisition_factory=_make_acquisition,
                bounds_transformer_factory=_make_bounds_transformer,
                liar=liar,
            )
            return best['target'], best['weights']
        except Exception as exc:
            print(f"[Bayesian_optimisation] NOTE: batch weight-search failed "
                  f"({type(exc).__name__}: {exc}) - falling back to the serial weight "
                  f"search for this restart (seed={seed}). This search's own result is "
                  f"unaffected, only wall-clock time.")
            best['target'], best['weights'] = -np.inf, None  # restart cleanly for serial below

    # Serial path - byte-for-byte the original (pre-ver4-5) search body,
    # generalised to `search_dims`/`bounds_list` and multiple probe points
    # (ver4-6 R6/R7).
    pbounds = {n: b for n, b in zip(names, bounds_list)}

    def wrapped(**kwargs):
        x = np.array([kwargs[n] for n in names])
        return objective_1d(x)

    optimizer = BayesianOptimization(
        wrapped, pbounds, random_state=seed, allow_duplicate_points=duplicate_points,
        acquisition_function=_make_acquisition(), bounds_transformer=_make_bounds_transformer(),
    )
    try:
        optimizer._gp.set_params(alpha=1e-3)
    except Exception:
        pass
    for _probe in probe_points:
        try:
            optimizer.probe(params=dict(zip(names, _probe)), lazy=True)
        except Exception:
            pass  # a probe point outside narrowed/degenerate bounds is skipped, never fatal
    optimizer.maximize(init_points=point_num, n_iter=iteration)
    return best['target'], best['weights']


def Bayesian(data_train, data_valid, data_test, record, effect, weight, MODEL, HPARAMETERS_OPT):
    
    minimum_boundary = HPARAMETERS_OPT[0]
    maximum_boundary = HPARAMETERS_OPT[1]
    iteration = HPARAMETERS_OPT[2]
    point_num = HPARAMETERS_OPT[3]
    duplicate_points = HPARAMETERS_OPT[4]

    # Phase 2, Requirements 4/5: see Nelder_Mead.py's identical comment
    # (same underlying linear-fractional-objective degeneracy, same fix) -
    # both are opt-in, appended-only entries at the END of
    # HPARAMETERS_OPT, defaulting to this method's exact original
    # (pre-Phase-2) behaviour when absent.
    _reg = read_regularization_settings(HPARAMETERS_OPT, start_index=5)
    diversity_penalty = _reg['diversity_penalty']
    objective_mode = _reg['objective_mode']
    diversity_method = _reg['diversity_method']
    shrinkage_alpha = _reg['alpha']
 
    model_selected = MODEL.copy()
    if 'ensemble' in MODEL:
        model_selected.remove('ensemble')

    # Bug fix: force every model-prediction column this function's arithmetic
    # touches onto a clean float64 dtype before any weight search begins.
    # Columns produced by the R-backed models (rrBLUP/BayesB/RKHS/GBLUP, via
    # rpy2) can come back as pandas 'object' dtype rather than float64 -
    # concatenating them alongside the sklearn/PyTorch-backed models'
    # ordinary float64 columns then upcasts the WHOLE frame's numeric columns
    # to 'object' too. Every pandas arithmetic op below (.div, .mean, .pow)
    # still "works" on an object-dtype column, but silently falls back to
    # element-by-element PLAIN PYTHON arithmetic instead of numpy's
    # vectorised fast path - and plain Python float division by exactly 0.0
    # raises ZeroDivisionError (unlike the numpy/pandas fast path, which
    # quietly produces inf/nan instead). That mismatch is exactly why a
    # weight vector collapsing to all-zero can raise ZeroDivisionError here:
    # not because the division itself is unguarded (see the explicit checks
    # added below), but because an all-zero-weights candidate is exactly the
    # one guaranteed to divide by 0 wherever it occurs, and object dtype is
    # what turns that into a raised exception instead of a silent inf/nan.
    data_train = data_train.copy()
    data_valid = data_valid.copy()
    data_test = data_test.copy()
    for _df in (data_train, data_valid, data_test):
        _df[model_selected + ['actual']] = _df[model_selected + ['actual']].astype(float)

    # ver4-5 R1.d/e/f (blueprint §3.4) - appended-only, indices 9-12,
    # read defensively (I5) so a pre-ver4-5 config (HYPERPARAMETERS_OPT 9
    # elements long, or shorter) keeps working with every new field at
    # its own documented, behaviour-preserving default.
    def _wopt_get(idx, default):
        return HPARAMETERS_OPT[idx] if len(HPARAMETERS_OPT) > idx else default

    w_opt_bayes_restarts = max(1, int(_wopt_get(9, 1) or 1))
    w_opt_bayes_batch = bool(_wopt_get(10, True))
    w_opt_bayes_liar = _wopt_get(11, 'max') or 'max'
    w_opt_vectorised_objective = bool(_wopt_get(12, True))

    K = len(model_selected)
    # The objective is invariant to uniformly scaling the whole weight
    # vector (both the numerator and the weights.sum() denominator scale
    # by the same factor and cancel), so any single positive constant,
    # clipped into [minimum_boundary, maximum_boundary], represents equal
    # weighting equally validly here.
    _equal_weight_val = float(np.clip(1.0 / K, minimum_boundary, maximum_boundary))

    # R1.d precompute (vectorised objective, default on) - see
    # _bayesian_single_search()'s own module-level docstring above for
    # the full derivation. Computed ONCE here (never re-derived per
    # candidate, and never per restart either - every restart/process
    # receives the SAME already-built arrays) since it depends only on
    # the validation predictions, never on a candidate weight vector.
    # An inert precompute (built but never read) when
    # w_opt_vectorised_objective is False.
    pred_matrix = data_valid[model_selected].to_numpy(dtype=float)
    actual_valid_arr = data_valid['actual'].to_numpy(dtype=float)
    _row_mean_pred = pred_matrix.mean(axis=1)
    e_minus_d = (pred_matrix - actual_valid_arr[:, None]) ** 2 - (pred_matrix - _row_mean_pred[:, None]) ** 2

    # ver4-4 §4a / ver4-5 R1.d precedent - default-on accelerations that
    # change a run's RESULTS at floating-point-rounding level (summation
    # order), not merely wall-clock time, are announced unconditionally,
    # every run (I11), exactly like GP()'s own top-level "[GP] NUMERICS:"
    # block does for its own such flags. W_OPT_VECTORISED_OBJECTIVE lives
    # in this method's own positional HYPERPARAMETERS_OPT list (Tab 4),
    # not in GP()'s resolve_compute_resources()-sourced _compute_cfg, so
    # it is announced HERE rather than folded into that shared block.
    if w_opt_vectorised_objective:
        print("[GP] NUMERICS: Bayesian weight-optimisation objective is VECTORISED "
              "(W_OPT_VECTORISED_OBJECTIVE=True, default) - agrees with the original "
              "loop-based computation to 1e-12 relative, but floating-point summation "
              "order differs. Set W_OPT_VECTORISED_OBJECTIVE=false to restore the exact "
              "pre-ver4-5 computation.")

    # Update ID ver4-6 (R5/R6/R7) - three further top-level config flags,
    # read via the SAME process-global compute-resources dict this
    # function already reads (for _restart_n_jobs) further below - NOT a
    # positional HYPERPARAMETERS_OPT entry, deliberately, so all three
    # weight-optimisation methods (this module, Nelder_Mead.py,
    # Linear_transformation.py) share ONE switch each rather than three
    # independently-drifting copies. See resolve_compute_resources()'s
    # own docstring for each key's default/rationale.
    _resources = get_active_compute_resources()
    w_opt_analytic_seed = bool(_resources.get('w_opt_analytic_seed', False))
    w_opt_validation_floor = bool(_resources.get('w_opt_validation_floor', False))
    w_opt_simplex_search = bool(_resources.get('w_opt_simplex_search', True))

    # ver4-6 R6.2: the closed-form least-squares solution, computed ONCE
    # here (deterministic, outside the restart fan-out - every restart
    # receives the SAME probe/floor candidate) - probed alongside equal
    # weighting in every restart (R6.4), and handed to the validation
    # floor below as an extra candidate.
    analytic_w = analytic_simplex_weights(pred_matrix, actual_valid_arr) if w_opt_analytic_seed else None

    # ver4-6 R7.2(c)/AC-R7.3: negative weights are structurally
    # unreachable under a stick-breaking simplex map (every share is
    # non-negative by construction) - printed once here (not per-
    # candidate/per-restart) whenever a negative Minimum boundary is
    # configured alongside simplex search, rather than silently ignoring
    # the setting.
    if w_opt_simplex_search and minimum_boundary < 0:
        print(f"[GP] NOTE: Bayesian optimisation weight search - Minimum boundary="
              f"{minimum_boundary} is negative, but W_OPT_SIMPLEX_SEARCH=True means every "
              f"candidate weight is non-negative by construction (a stick-breaking map onto "
              f"the probability simplex - ver4-6 R7.2(b)); negative weights are unreachable "
              f"regardless of this setting. Set W_OPT_SIMPLEX_SEARCH=False to restore the "
              f"literal per-weight box semantics.")

    # ver4-6 R5.2a/AC-R5.5: the resolved objective and its source,
    # printed unconditionally (I11) for this weight-optimisation method.
    _objective_source = 'explicit HYPERPARAMETERS_OPT entry' if len(HPARAMETERS_OPT) > 6 else \
        'default (ver4-6: ensemble_mse fallback - see read_regularization_settings)'
    print(f"[GP] Bayesian optimisation weight search: objective_mode={objective_mode!r} "
          f"({_objective_source}), w_opt_analytic_seed={w_opt_analytic_seed}, "
          f"w_opt_validation_floor={w_opt_validation_floor}, "
          f"w_opt_simplex_search={w_opt_simplex_search}.")

    _search_kwargs = dict(
        model_selected=model_selected, minimum_boundary=minimum_boundary,
        maximum_boundary=maximum_boundary, point_num=point_num, iteration=iteration,
        duplicate_points=duplicate_points, objective_mode=objective_mode,
        diversity_penalty=diversity_penalty, diversity_method=diversity_method,
        vectorised=w_opt_vectorised_objective, data_valid=data_valid, pred_matrix=pred_matrix,
        actual_valid_arr=actual_valid_arr, e_minus_d=e_minus_d, equal_weight_val=_equal_weight_val,
        use_batch=w_opt_bayes_batch, liar=w_opt_bayes_liar,
        # ver4-6 R6/R7: threaded through to every restart identically.
        simplex_search=w_opt_simplex_search, analytic_probe=analytic_w,
    )

    if w_opt_bayes_restarts <= 1:
        # seed=1 - preserves today's exact single-run seed
        # (random_state=1) unconditionally; W_OPT_BAYES_RESTARTS=1 is
        # this method's default, so an untouched config reaches exactly
        # this branch.
        best_target, best_weights = _bayesian_single_search(seed=1, **_search_kwargs)
    else:
        # ver4-5 R1.e (blueprint §3.4) - R independent, complete
        # weight-search restarts, seeds 1..R (seed 1 FIRST, preserving
        # today's exact single-run search as one of the restarts), the
        # highest-target restart wins with first-write-wins on ties in
        # seed order. This is the mechanism that genuinely uses
        # multiple CPUs for weight optimisation (unlike R1.f - see this
        # module's own module-level note above _bayesian_single_search).
        _resources = get_active_compute_resources()
        _restart_n_jobs = nested_safe_n_jobs(_resources, _resources.get('n_jobs'))
        _restart_n_jobs = min(_restart_n_jobs, w_opt_bayes_restarts)
        seeds = list(range(1, w_opt_bayes_restarts + 1))
        results = None
        if _restart_n_jobs > 1:
            try:
                from joblib import Parallel, delayed
                results = Parallel(n_jobs=_restart_n_jobs, backend='loky')(
                    delayed(_bayesian_single_search)(seed=s, **_search_kwargs) for s in seeds
                )
            except Exception as exc:
                print(f"[Bayesian_optimisation] NOTE: parallel weight-search restart "
                      f"fan-out failed ({type(exc).__name__}: {exc}) - falling back to "
                      f"serial restarts (W_OPT_BAYES_RESTARTS={w_opt_bayes_restarts} "
                      f"requested). This search's own result is unaffected, only "
                      f"wall-clock time.")
                results = None
        if results is None:
            results = [_bayesian_single_search(seed=s, **_search_kwargs) for s in seeds]

        best_target, best_weights = -np.inf, None
        for _target, _weights in results:
            if _target is not None and _target > best_target:
                best_target, best_weights = _target, _weights

    # w is the weight vector the winning restart's own optimisation()
    # closure captured at the point it achieved the best target -
    # guaranteed to be the correct, full-length array (the equal-
    # weighting probe below is always evaluated first, in every
    # restart, so this is never None in practice).
    w = best_weights

    ## Weight extraction (ver4-6 R5.2b/R6.2/R7.2(c)).
    if w_opt_validation_floor:
        # Every candidate - the search's own winner (`w`), equal
        # weighting, and (when W_OPT_ANALYTIC_SEED) the R6 analytic
        # solution - is normalised and graded on its OWN realised
        # validation MSE via select_weights_with_floor(), which is ALSO
        # where naive shrinkage (Requirement 5) is applied. This makes
        # "never worse than the naive ensemble on validation" a
        # construction-level guarantee, replacing the old ad hoc
        # abs()/normalise/shrink block entirely.
        #
        # R7.2(c): weight_extracted (recorded to Weight.csv, and used
        # below to weight marker effects) is now built from THIS SAME,
        # already-normalised final `w` - previously it was built from the
        # RAW, pre-normalisation search winner, while predictions used
        # the separately abs()-normalised vector - two different vectors
        # for "the weights this run used", which is exactly the sign/
        # normalisation mismatch R7.1(c) named. Grading and applying (and
        # now recording) the SAME vector removes that mismatch outright.
        _extra_candidates = [('analytic', analytic_w)] if analytic_w is not None else None
        w, _wopt_chosen_label, _wopt_diag = select_weights_with_floor(
            w, pred_matrix, actual_valid_arr, shrinkage_alpha=shrinkage_alpha,
            extra_candidates=_extra_candidates,
        )
        print(f"[GP] Bayesian optimisation weight search: floor selected {_wopt_chosen_label!r} "
              f"(validation MSE - {_wopt_diag}).")
        weight_extracted = pd.DataFrame(w).T
        weight_extracted.columns = model_selected
    else:
        # ver4-5-equivalent path (rollback: W_OPT_VALIDATION_FLOOR=False)
        # - byte-for-byte the pre-ver4-6 code, INCLUDING weight_extracted
        # being built from the RAW pre-normalisation search winner
        # (AC-R6.3 rollback verification requires bit-identical output).
        weight_extracted = pd.DataFrame(w).T
        weight_extracted.columns = model_selected

        # Requirement (safeguard): fall back to equal weighting if the
        # optimiser converged to all-zero weights (sum(w)==0) - degenerate,
        # but possible depending on the configured bounds - rather than
        # raising ZeroDivisionError on the plain-Python division below.
        # `> 0` (not `!= 0`) also catches a NaN sum - `math.nan != 0` is True
        # in Python, so the old check let a NaN _sum_abs_w straight through
        # into `x / _sum_abs_w`, silently producing all-NaN weights instead of
        # falling back to equal weighting the way a zero sum already did.
        _abs_w = [abs(x) for x in w]
        _sum_abs_w = sum(_abs_w)
        if _sum_abs_w > 0 and np.isfinite(_sum_abs_w):
            w = [x / _sum_abs_w for x in _abs_w]
        else:
            w = [1.0 / len(_abs_w)] * len(_abs_w)

        # Requirement 5 (secondary mitigation): optional naive-shrinkage blend
        # toward equal weighting - a no-op when shrinkage_alpha is 1.0
        # (unset/default).
        w = apply_naive_shrinkage(w, shrinkage_alpha)
    
    ## Weight the predicted phenotypes for the test set
    data_test_selected = data_test.loc[:,model_selected+['actual']]
    for i in range(len(data_test_selected.columns)-1):
        data_test_selected.iloc[:,i] = data_test_selected.iloc[:,i] * w[i]
    predicted_test =  data_test_selected.iloc[:,:-1].sum(axis=1).reset_index(drop=True)
    actual_test =  data_test_selected.iloc[:,-1].tolist()
    
    ## Calculate the metrics
    mse = mean_squared_error(actual_test, predicted_test)
    r = scipy.stats.pearsonr(actual_test, predicted_test)[0]
    
    ## Store the metrics
    record = pd.concat([record, pd.DataFrame(record.iloc[-1,:]).T]).reset_index(drop=True)
    record.loc[record.shape[0]-1,'model'] = 'Bayesian optimisation'
    record.loc[record.shape[0]-1,'Pearson correlation'] = r
    record.loc[record.shape[0]-1,'MSE'] = mse

    ## Weight the predicted phenotypes for the validation set
    predicted_valid = []
    data_valid_selected = data_valid.loc[:,model_selected+['actual']]
    for i in range(len(data_valid_selected.columns)-1):
        data_valid_selected.iloc[:,i] = data_valid_selected.iloc[:,i] * w[i]
    predicted_valid =  data_valid_selected.iloc[:,:-1].sum(axis=1).reset_index(drop=True)
    #actual_valid =  data_valid_selected.iloc[:,-1].tolist()
    
    ## Weight the predicted phenotypes for the training set
    predicted_train = []
    data_train_selected = data_train.loc[:,model_selected+['actual']]
    for i in range(len(data_train_selected.columns)-1):
        data_train_selected.iloc[:,i] = data_train_selected.iloc[:,i] * w[i]
    predicted_train =  data_train_selected.iloc[:,:-1].sum(axis=1).reset_index(drop=True)
    #actual_train =  data_train_selected.iloc[:,-1].tolist()
    
    data_test['Bayesian'] = predicted_test
    data_valid['Bayesian'] = predicted_valid
    data_train['Bayesian'] = predicted_train

    ## Extract weights
    weight_sample = pd.DataFrame(record.iloc[record.shape[0]-1,:]).T.drop(['Pearson correlation', 'MSE'],axis=1)
    weight_sample['model'] = 'Bayesian optimisation' 
    
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
    effect.loc[effect.shape[0]-1,'model'] = 'Bayesian optimisation'
    
    effect.loc[effect.shape[0]-1,list(effect_weighted.columns)] = effect_weighted.values

    return record, effect, data_test, data_valid, data_train, weight