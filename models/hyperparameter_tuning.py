"""
EasiGP - models/hyperparameter_tuning.py
--------------------------------------------------------------------------
Generic, model-agnostic hyperparameter search for EasiGP prediction models.

SCOPE / GUARANTEES
--------------------------------------------------------------------------
- Only ever runs when a real train/valid/test split exists for the current
  task (caller must check `valid.shape[0] != 0` before calling in here -
  this module does not re-derive that itself, to keep it a pure function of
  whatever data it's handed).
- Selects hyperparameters by maximising Pearson r / MSE on the VALIDATION
  set only. The test set is only ever touched once, for the single
  confirmatory fit performed after the winning hyperparameters are chosen -
  identical in spirit to every other (untuned) model call already in
  genomic_prediction.py.
- "Maximise Pearson r / MSE" and "minimise MSE / Pearson r" are the same
  optimum (one is the reciprocal of the other). This module implements one
  canonical score and lets minimiser-style algorithms (Nelder-Mead, Powell)
  work on its negation internally - there's no behavioural difference for
  the two framings to expose as separate options.
- Nothing in here imports Streamlit, main_app.py, or genomic_prediction.py,
  so it can be imported from headless runners exactly like every other
  models/*.py file.

HOW A CALLER USES THIS MODULE
--------------------------------------------------------------------------
    from hparam_specs import HPARAM_SPECS
    from models.hyperparameter_tuning import tune_model_hyperparameters

    final_params, final_result, best_score, elapsed = tune_model_hyperparameters(
        model_name='RF',
        base_params=HPARAMETERS['RF'],       # the user's current positional list
        train=train, valid=valid, test=test,
        run_model_fn=_call_model,            # see INTEGRATION_GUIDE.md
        RESULT_NAME=RESULT_NAME,
        algorithm='Bayesian',
        budget_kwargs={'n_iter': 30, 'init_points': 5},
        hparam_specs=HPARAM_SPECS,
        reduced_cost_search=True,
    )

`run_model_fn` must have the signature
    run_model_fn(model_name, train, valid, test, params, RESULT_NAME) -> dict
and the returned dict must contain at least:
    'pearson_valid', 'mse_valid'   (floats, or None/NaN if valid is empty)
plus whatever else the caller wants to read back out of `final_result`
after tuning (pearson_test, mse_test, effect, predicted_test, ... - see
INTEGRATION_GUIDE.md for the exact shape genomic_prediction.py expects).
"""

import itertools
import math
import time

import numpy as np
from scipy.optimize import minimize

try:
    from bayes_opt import BayesianOptimization
except ImportError:  # pragma: no cover - already a project dependency
    BayesianOptimization = None


# --------------------------------------------------------------------------- #
# Naming
# --------------------------------------------------------------------------- #

TUNE_SUFFIX_SEP = '__'

ALGORITHMS = ['Grid', 'Random', 'Bayesian', 'Nelder-Mead', 'Powell']

ALGO_TAG = {
    'Grid': 'Grid',
    'Random': 'Random',
    'Bayesian': 'Bayesian',
    'Nelder-Mead': 'NelderMead',
    'Powell': 'Powell',
}


def tuned_model_name(base_model, algorithm, n_algorithms_for_model):
    """Only suffix when more than one tuning algorithm is used for this
    particular model, so single-algorithm runs keep today's plain model
    names in every output file (requirement: suffix only when it's needed
    to disambiguate)."""
    if n_algorithms_for_model <= 1:
        return base_model
    return f"{base_model}{TUNE_SUFFIX_SEP}{ALGO_TAG[algorithm]}"


def base_of(tuned_name):
    """Inverse of tuned_model_name: 'RF__Bayesian' -> 'RF'. A plain,
    unsuffixed name (or a model whose real name happens to contain '__',
    which none currently do) is returned unchanged."""
    return tuned_name.split(TUNE_SUFFIX_SEP, 1)[0]


def schema_key_of(model_name):
    """Map a MODEL-list entry to the key it should be looked up under in
    HPARAM_SPECS / DIAGNOSTIC_FLAG_FIELDS / MCMC_LENGTH_FIELDS.

    Almost always the identity function - but genomic_prediction.py allows
    more than one independent GAT_biological_prior_knowledge instance in
    the same run, named 'GAT_biological_prior_knowledge_2',
    'GAT_biological_prior_knowledge_3', etc. (see _is_bio_prior_model() in
    genomic_prediction.py), each with its own HPARAMETERS entry but all
    sharing the exact same field layout. Every registry in this module is
    keyed by the base type name only, so an instance suffix must be
    stripped before any lookup - this is that single, shared place it
    happens, rather than every call site re-deriving it.

    Call this (not base_of()) whenever indexing into HPARAM_SPECS,
    DIAGNOSTIC_FLAG_FIELDS, or MCMC_LENGTH_FIELDS. Use base_of() (not this)
    when inverting a tuning-algorithm suffix - the two are orthogonal:
    'GAT_biological_prior_knowledge_2__Grid' has an algorithm suffix
    (base_of strips) AND an instance suffix (schema_key_of strips), and
    each needs stripping for a different purpose at a different point."""
    if model_name == 'GAT_biological_prior_knowledge':
        return model_name
    if model_name.startswith('GAT_biological_prior_knowledge_'):
        suffix = model_name[len('GAT_biological_prior_knowledge_'):]
        if suffix.isdigit():
            return 'GAT_biological_prior_knowledge'
    return model_name


def algo_of(tuned_name):
    """Inverse of tuned_model_name: 'RF__Bayesian' -> 'Bayesian'. Returns
    None for an unsuffixed name."""
    if TUNE_SUFFIX_SEP not in tuned_name:
        return None
    tag = tuned_name.split(TUNE_SUFFIX_SEP, 1)[1]
    for algo, t in ALGO_TAG.items():
        if t == tag:
            return algo
    return None


# --------------------------------------------------------------------------- #
# Parameter space
# --------------------------------------------------------------------------- #

class ParamSpec:
    __slots__ = ('index', 'name', 'kind', 'low', 'high', 'choices', 'step')

    def __init__(self, index, name, kind, low=None, high=None, choices=None, step=None):
        self.index = index
        self.name = name
        self.kind = kind          # 'int' | 'float' | 'categorical'
        self.low = low
        self.high = high
        self.choices = choices    # list, only for 'categorical'
        self.step = step          # optional, grid search only

    def __repr__(self):
        return f"ParamSpec({self.name!r}, kind={self.kind!r}, idx={self.index})"


def build_param_specs(model_name, hparam_specs):
    """Read hparam_specs.HPARAM_SPECS[model_name] (main_app.py's own
    schema) and return the ordered list of ParamSpec for every field
    carrying a 'tunable' entry. Returns [] for a model with nothing marked
    tunable - always safe to call, never raises for an unknown/unannotated
    model. `model_name` is passed through schema_key_of() first, so a
    numbered GAT_biological_prior_knowledge instance resolves to the shared
    base schema automatically."""
    specs = []
    fields = hparam_specs.get(schema_key_of(model_name), [])
    for idx, field in enumerate(fields):
        tunable = field.get('tunable')
        if not tunable:
            continue
        ftype = field['type']
        name = field['label']

        if ftype in ('int', 'float'):
            specs.append(ParamSpec(
                idx, name, ftype,
                low=tunable['low'], high=tunable['high'], step=tunable.get('step'),
            ))
        elif ftype == 'bool':
            specs.append(ParamSpec(idx, name, 'categorical', choices=[True, False]))
        elif ftype in ('str', 'rf_max_features', 'svr_gamma', 'int_float_or_none'):
            choices = tunable.get('choices', field.get('choices'))
            if not choices:
                raise ValueError(
                    f"{model_name!r} field {name!r} is marked tunable but has no "
                    f"'choices' (required for type {ftype!r})."
                )
            specs.append(ParamSpec(idx, name, 'categorical', choices=list(choices)))
        else:
            # int_or_all / top_pct and any other diagnostic-only composite
            # type are never tunable - they gate expensive explainability
            # output, not the fit - ignored even if 'tunable' were present.
            continue
    return specs


# --------------------------------------------------------------------------- #
# Continuous-relaxation encode/decode, shared by every non-grid algorithm
# below (the same trick already used for ensemble-weight search in
# models/Bayesian_optimisation.py, generalised to mixed types).
# --------------------------------------------------------------------------- #

def _encode_bounds(specs):
    bounds = []
    for s in specs:
        if s.kind in ('int', 'float'):
            bounds.append((s.low, s.high))
        else:  # categorical - encoded as a float index into s.choices
            bounds.append((0.0, len(s.choices) - 1e-6))
    return bounds


def _decode(specs, x):
    values = {}
    for s, v in zip(specs, x):
        if s.kind == 'int':
            values[s.index] = int(round(min(max(v, s.low), s.high)))
        elif s.kind == 'float':
            values[s.index] = float(min(max(v, s.low), s.high))
        else:
            i = int(math.floor(min(max(v, 0), len(s.choices) - 1)))
            values[s.index] = s.choices[i]
    return values


def _random_point(specs, rng):
    x = []
    for s in specs:
        if s.kind == 'int':
            x.append(rng.integers(s.low, s.high + 1))
        elif s.kind == 'float':
            x.append(rng.uniform(s.low, s.high))
        else:
            x.append(rng.uniform(0, len(s.choices) - 1e-6))
    return np.array(x, dtype=float)


# --------------------------------------------------------------------------- #
# Cost control for candidate evaluations (search phase only)
# --------------------------------------------------------------------------- #

# nIter / burnIn positional indices for the BGLR-backed R models. Used only
# to cheapen candidate evaluation during search (never the final
# confirmatory fit). Add an entry here if another MCMC-based model is
# introduced later.
MCMC_LENGTH_FIELDS = {
    'rrBLUP': (0, 1),
    'GBLUP': (0, 1),
    'BayesB': (0, 1),
    'RKHS': (0, 1),
}
MCMC_SEARCH_SCALE = 0.15
MCMC_SEARCH_MIN = {'nIter': 300, 'burnIn': 50}

# Positional index of the boolean flag that gates SHAP / interaction /
# marker-effect computation for each model - forced False during search
# (restored for the final confirmatory fit). Extend this alongside
# hparam_specs.HPARAM_SPECS as new models are annotated for tuning.
DIAGNOSTIC_FLAG_FIELDS = {
    'RF': 5, 'SVR': 6, 'KNN': 3, 'RKHS': 3,
    'GBLUP': 2,
    # BayesB has NO explainability gate at all - confirmed directly from
    # models/BayesB.R, which returns its marker-effect table unconditionally
    # on every call (4 params total: nIter, burnIn, probIn, counts). No
    # entry here is correct/required for it, not an oversight.
    'GAT_prior_knowledge': 8,
    'GAT_fully_connected': 7,
    'GAT_infinitesimal': 7,             # marker_effect
    'GAT_infinitesimal_node_level': 8,  # marker_effect (position 8 here,
                                         # not 7 - this model's samples/
                                         # marker_effect order is swapped
                                         # relative to GAT_infinitesimal)
    'GAT_biological_prior_knowledge': 14,  # marker_effect
}


def _disable_explainability(model_name, params):
    idx = DIAGNOSTIC_FLAG_FIELDS.get(schema_key_of(model_name))
    if idx is not None:
        params = list(params)
        params[idx] = False
    return params


def _cheapen_mcmc(model_name, params, base_params, tuned_indices):
    fields = MCMC_LENGTH_FIELDS.get(schema_key_of(model_name))
    if fields is None:
        return params
    params = list(params)
    n_idx, b_idx = fields
    if n_idx not in tuned_indices:
        params[n_idx] = max(MCMC_SEARCH_MIN['nIter'], int(base_params[n_idx] * MCMC_SEARCH_SCALE))
    if b_idx not in tuned_indices:
        params[b_idx] = max(MCMC_SEARCH_MIN['burnIn'], int(base_params[b_idx] * MCMC_SEARCH_SCALE))
    if params[b_idx] >= params[n_idx]:
        params[b_idx] = max(1, params[n_idx] // 4)
    return params


# --------------------------------------------------------------------------- #
# Objective
# --------------------------------------------------------------------------- #

SCORE_CAP = 1e6  # guards a near-zero validation MSE from producing inf/nan


def make_objective(model_name, base_params, specs, train, valid, test,
                    run_model_fn, RESULT_NAME, reduced_cost_search=True):
    """Returns f(x) -> score-to-maximise (float) for a continuous-encoded
    candidate x, evaluated on the VALIDATION set only."""
    tuned_indices = {s.index for s in specs}

    def objective(x):
        params = list(base_params)
        for idx, val in _decode(specs, x).items():
            params[idx] = val

        params = _disable_explainability(model_name, params)
        if reduced_cost_search:
            params = _cheapen_mcmc(model_name, params, base_params, tuned_indices)

        try:
            result = run_model_fn(model_name, train, valid, test, params, RESULT_NAME)
        except Exception as exc:
            print(f"[hyperparameter_tuning] Candidate failed for {model_name} "
                  f"({type(exc).__name__}: {exc}); scored as worst and skipped.")
            return -SCORE_CAP

        r = result.get('pearson_valid')
        mse = result.get('mse_valid')
        if r is None or mse is None or not np.isfinite(mse):
            return -SCORE_CAP
        if not np.isfinite(r):
            # Undefined correlation means this candidate's validation
            # predictions have ZERO variance (a constant/degenerate
            # predictor - e.g. RF collapsing to a single leaf per tree with
            # too few training samples relative to min_samples_leaf). That
            # provides no genomic-prediction value at all and must never be
            # preferred over a model that at least attempts to
            # differentiate individuals, even one that currently
            # generalises poorly (negative r).
            #
            # Scoring this as r=0 (i.e. score=0/mse=0, a "neutral" outcome)
            # instead of worst-case was a real bug, not just a theoretical
            # edge case: confirmed by direct reproduction against the real
            # RF model on a small, weak-signal dataset (realistic for
            # genomic prediction) - most candidates that generalise poorly
            # score NEGATIVE (r<0), so the "neutral" 0.0 a constant
            # predictor got under the old logic would systematically beat
            # them and get selected as the tuning "winner", despite being
            # strictly useless. Constant predictors are now scored as the
            # worst possible outcome instead.
            return -SCORE_CAP
        if mse <= 0:
            return SCORE_CAP
        score = r / mse
        return float(np.clip(score, -SCORE_CAP, SCORE_CAP)) if np.isfinite(score) else -SCORE_CAP

    return objective


# --------------------------------------------------------------------------- #
# Search algorithms - each returns (best_x, best_score, trials)
# --------------------------------------------------------------------------- #

def search_grid(objective, specs, n_points=5):
    axes = []
    for s in specs:
        if s.kind == 'categorical':
            axes.append(list(range(len(s.choices))))
        else:
            step = s.step
            if step:
                pts = np.arange(s.low, s.high + step / 2, step)
            else:
                pts = np.linspace(s.low, s.high, n_points)
            axes.append(pts.tolist())

    best_x, best_score, trials = None, -math.inf, []
    for combo in itertools.product(*axes):
        x = np.array(combo, dtype=float)
        score = objective(x)
        trials.append((x, score))
        if score > best_score:
            best_score, best_x = score, x
    if best_x is None:  # zero tunable dims edge case
        best_x = np.array([])
    return best_x, best_score, trials


def search_random(objective, specs, n_iter=40, seed=0):
    rng = np.random.default_rng(seed)
    best_x, best_score, trials = None, -math.inf, []
    for _ in range(max(1, n_iter)):
        x = _random_point(specs, rng)
        score = objective(x)
        trials.append((x, score))
        if score > best_score:
            best_score, best_x = score, x
    return best_x, best_score, trials


def search_bayesian(objective, specs, n_iter=30, init_points=5, seed=0):
    if BayesianOptimization is None:
        raise ImportError(
            "bayes_opt is required for Bayesian hyperparameter tuning "
            "(already a project dependency via models/Bayesian_optimisation.py)."
        )
    bounds = _encode_bounds(specs)
    names = [f'x{i}' for i in range(len(specs))]
    pbounds = {n: b for n, b in zip(names, bounds)}

    def wrapped(**kwargs):
        x = np.array([kwargs[n] for n in names])
        return objective(x)

    optimizer = BayesianOptimization(f=wrapped, pbounds=pbounds, random_state=seed, verbose=0)
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    best = optimizer.max
    best_x = np.array([best['params'][n] for n in names])
    return best_x, best['target'], optimizer.res


def _multistart_local_search(method, objective, specs, n_iter, seed, n_restarts, **method_options):
    """Nelder-Mead and Powell are local searches: on a mixed-type
    continuous relaxation with categorical dimensions, a single run can
    converge to a simplex/direction-set that never revisits a categorical
    choice it started away from (verified empirically in this project's own
    test suite - see test_hyperparameter_tuning.py). Multi-restart from
    several random points and keeping the best is the standard mitigation,
    and costs nothing extra beyond distributing the same total evaluation
    budget across restarts instead of spending it all on one."""
    n_restarts = max(1, n_restarts)
    per_restart_iter = max(5, n_iter // n_restarts)
    bounds = _encode_bounds(specs)
    trials = []
    best_x, best_score = None, -math.inf

    rng = np.random.default_rng(seed)
    for i in range(n_restarts):
        x0 = _random_point(specs, rng)

        def neg_obj(x):
            score = objective(x)
            trials.append((x.copy(), score))
            return -score

        res = minimize(neg_obj, x0, method=method, bounds=bounds,
                        options={'maxiter': per_restart_iter, **method_options})
        if -res.fun > best_score:
            best_score, best_x = -res.fun, res.x

    return best_x, best_score, trials


def search_nelder_mead(objective, specs, n_iter=40, seed=0, n_restarts=4):
    return _multistart_local_search(
        'Nelder-Mead', objective, specs, n_iter, seed, n_restarts,
        xatol=1e-3, fatol=1e-3, adaptive=True,
    )


def search_powell(objective, specs, n_iter=40, seed=0, n_restarts=4):
    return _multistart_local_search(
        'Powell', objective, specs, n_iter, seed, n_restarts,
        xtol=1e-3, ftol=1e-3,
    )


SEARCH_FUNCS = {
    'Grid': search_grid,
    'Random': search_random,
    'Bayesian': search_bayesian,
    'Nelder-Mead': search_nelder_mead,
    'Powell': search_powell,
}


def run_search(algorithm, objective, specs, budget_kwargs):
    if algorithm not in SEARCH_FUNCS:
        raise ValueError(f"Unknown hyperparameter search algorithm {algorithm!r}. "
                          f"Choose from {list(SEARCH_FUNCS)}.")
    return SEARCH_FUNCS[algorithm](objective, specs, **(budget_kwargs or {}))


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #

def tune_model_hyperparameters(model_name, base_params, train, valid, test,
                                run_model_fn, RESULT_NAME, algorithm,
                                budget_kwargs, hparam_specs,
                                reduced_cost_search=True):
    """Search `algorithm`'s best hyperparameters for `model_name` on the
    validation set, then perform ONE final confirmatory fit at the winning
    hyperparameters with the user's real explainability flags and MCMC
    length restored.

    Returns
    -------
    final_params : list          - full HPARAMETERS[model_name]-shaped list
    final_result : dict          - exactly what run_model_fn returns
    best_valid_score : float     - Pearson r / MSE achieved on validation
    tuned_values : dict          - {field_label: final_value}, for reporting
    elapsed : float              - wall-clock seconds
    """
    specs = build_param_specs(model_name, hparam_specs)
    if not specs:
        raise ValueError(
            f"No tunable fields are marked for {model_name!r} in HPARAM_SPECS "
            f"(add a 'tunable': {{...}} entry to at least one of its fields "
            f"before enabling tuning for this model)."
        )

    t0 = time.time()
    objective = make_objective(model_name, base_params, specs, train, valid, test,
                                run_model_fn, RESULT_NAME, reduced_cost_search)
    best_x, best_score, _trials = run_search(algorithm, objective, specs, budget_kwargs)

    final_params = list(base_params)
    tuned_values = {}
    for s, val in zip(specs, _decode(specs, best_x).values()):
        final_params[s.index] = val
        tuned_values[s.name] = val

    # Final confirmatory fit: full explainability + full MCMC length exactly
    # as the user configured them - identical in every respect to a normal,
    # untuned model call.
    final_result = run_model_fn(model_name, train, valid, test, final_params, RESULT_NAME)
    elapsed = time.time() - t0
    return final_params, final_result, best_score, tuned_values, elapsed


# --------------------------------------------------------------------------- #
# Ensemble grouping (per-method vs across-methods)
# --------------------------------------------------------------------------- #
#
# genomic_prediction.py's ensemble()/Linear_transformation()/Nelder_Mead()/
# Bayesian() functions already select their input columns purely by name
# from a user-supplied MODEL list - see models/ensemble.py's
# `model_selected = MODEL.copy()`. That means "ensemble across every tuning
# method" needs no new code at all: it's simply what already happens when
# those functions are called once with the full suffixed MODEL list.
# "Ensemble per tuning method" instead calls them once per algorithm group,
# each time with a MODEL list restricted to that group's variant columns
# (plus every model that wasn't multi-algorithm-tuned, which has no group to
# belong to and is included in all of them) - implemented here as pure list
# bookkeeping, with zero changes required to the ensemble-weighting model
# files themselves.

def multi_algorithm_models(model_config):
    """model_config: {base_model_name: {'enabled': bool, 'algorithms': [...], ...}}
    (this is exactly EasiGP's HP_TUNE dict - see INTEGRATION_GUIDE.md).
    Returns {base_model_name: [algorithm, ...]} for every model tuned with
    more than one algorithm - i.e. every model whose outputs actually need
    a suffix per tuned_model_name()'s own rule."""
    groups = {}
    if not model_config:
        return groups
    for m, cfg in model_config.items():
        if cfg.get('enabled') and len(cfg.get('algorithms', [])) > 1:
            groups[m] = list(cfg['algorithms'])
    return groups


def expand_model_list(model_base, model_config):
    """Turn the user's plain MODEL selection (e.g. ['RF','SVR','ensemble'])
    into the suffixed list actually iterated/produced this run (e.g.
    ['RF__Grid','RF__Bayesian','SVR','ensemble']), using tuned_model_name()'s
    own single-vs-multi-algorithm rule. 'ensemble' (if present) is always
    moved to the end, matching genomic_prediction.py's existing convention
    of running it after every base model."""
    groups = multi_algorithm_models(model_config)
    expanded = []
    for m in model_base:
        if m == 'ensemble':
            continue
        if m in groups:
            for algo in groups[m]:
                expanded.append(tuned_model_name(m, algo, len(groups[m])))
        else:
            expanded.append(m)
    if 'ensemble' in model_base:
        expanded.append('ensemble')
    return expanded


def ensemble_groups(model_base, model_config, mode):
    """Returns a list of (group_label, model_list) pairs describing how many
    times, and with which input columns, the ensemble-weighting step should
    run.

    mode='across_methods' -> a single group using every variant together
        (group_label=None signals "don't rename the output column/model
        label - behave exactly as today").
    mode='per_method'     -> one group per algorithm name that was actually
        used for at least one multi-algorithm-tuned model. Each group
        contains that algorithm's variant of every multi-algorithm-tuned
        model, plus every model that was NOT multi-algorithm-tuned (single-
        algorithm-tuned or not tuned at all - it has no group of its own,
        so it belongs to all of them), plus 'ensemble' itself as the
        trigger token the existing ensemble()/W_OPT functions expect.
    """
    groups = multi_algorithm_models(model_config)
    fixed_models = [m for m in model_base if m != 'ensemble' and m not in groups]

    if mode == 'across_methods' or not groups:
        return [(None, expand_model_list(model_base, model_config))]

    if mode != 'per_method':
        raise ValueError(f"Unknown HP_TUNE_ENSEMBLE_MODE {mode!r}; expected "
                          f"'per_method' or 'across_methods'.")

    algo_names = sorted({algo for algos in groups.values() for algo in algos})
    result = []
    for algo in algo_names:
        variant_models = [
            tuned_model_name(m, algo, len(algos))
            for m, algos in groups.items() if algo in algos
        ]
        group_models = fixed_models + variant_models
        if 'ensemble' in model_base:
            group_models = group_models + ['ensemble']
        result.append((algo, group_models))
    return result
