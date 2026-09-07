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
import traceback

import numpy as np
from scipy.optimize import minimize

# --------------------------------------------------------------------------- #
# ver4-7 R-fix (see PATCH_NOTES: "TypeError: Parallel.__init__() got an
# unexpected keyword argument 'initializer'"): joblib only grew the ability
# to forward arbitrary `**backend_kwargs` (including `initializer`/
# `initargs`) straight through `Parallel(...)` in joblib >= 1.5. On the
# joblib==1.4.2 pin shipped in environment_linux.yml (and on any older
# joblib a user's own environment happens to have), `Parallel.__init__`
# has a FIXED keyword list with no `**backend_kwargs` catch-all at all, so
# `Parallel(n_jobs=n_jobs, backend='loky', initializer=..., initargs=...)`
# raises exactly that TypeError before a single candidate is evaluated.
# `_evaluate_batch()`'s own `except Exception` already caught this and fell
# back to serial evaluation - so no run was ever WRONG, only slower than it
# should have been (see that function's own NOTE text) - but the fan-out
# this module exists to provide was silently lost on every joblib<1.5
# install.
#
# Fix: never rely on `Parallel(..., initializer=...)` at all. Build a small
# `loky` backend SUBCLASS that carries `initializer`/`initargs` as its own
# instance state and injects them into its `configure()` call itself -
# `configure()` has accepted `initializer`/`initargs` via
# `**memmappingexecutor_args` since loky was first vendored into joblib, so
# this path is stable across every joblib version this project supports
# (verified against both 1.4.2 and 1.5.3). Passing an already-built BACKEND
# INSTANCE to `Parallel(backend=...)` is itself a long-stable public joblib
# pattern - unlike passing new keyword arguments to `Parallel.__init__`
# itself, which is exactly what changed between versions.
def _loky_parallel(n_jobs, worker_init=None, worker_init_args=()):
    """Return a ``joblib.Parallel`` configured for the ``loky`` (process)
    backend, optionally running ``worker_init(*worker_init_args)`` once in
    every freshly-spawned worker - version-agnostic across joblib releases
    that do, and do not, accept ``initializer=``/``initargs=`` directly as
    ``Parallel(...)`` keyword arguments."""
    from joblib import Parallel

    if worker_init is None:
        return Parallel(n_jobs=n_jobs, backend='loky')

    from joblib._parallel_backends import LokyBackend

    class _InitializingLokyBackend(LokyBackend):
        def configure(self, *args, **kwargs):
            kwargs.setdefault('initializer', worker_init)
            kwargs.setdefault('initargs', worker_init_args)
            return super().configure(*args, **kwargs)

    return Parallel(n_jobs=n_jobs, backend=_InitializingLokyBackend())

try:
    from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
    from bayes_opt.acquisition import ExpectedImprovement
except ImportError:  # pragma: no cover - already a project dependency
    BayesianOptimization = None
    SequentialDomainReductionTransformer = None
    ExpectedImprovement = None


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
    # Update ID ver4-6, R1.2(b): 'scale' records whether this dimension's
    # unit-cube <-> native mapping (_decode()/_encode_from_params() below)
    # is linear (default) or logarithmic. Declared per-field inside
    # hparam_specs.HPARAM_SPECS[<model>][<index>]['tunable']['scale'] - a
    # NEW KEY inside the existing 'tunable' dict, so the positional field
    # list itself is untouched (invariant I5).
    __slots__ = ('index', 'name', 'kind', 'low', 'high', 'choices', 'step', 'scale')

    def __init__(self, index, name, kind, low=None, high=None, choices=None, step=None,
                 scale='linear'):
        self.index = index
        self.name = name
        self.kind = kind          # 'int' | 'float' | 'categorical'
        self.low = low
        self.high = high
        self.choices = choices    # list, only for 'categorical'
        self.step = step          # optional, grid search only
        self.scale = scale        # 'linear' (default) | 'log' - ver4-6 R1.2(b)

    def __repr__(self):
        return f"ParamSpec({self.name!r}, kind={self.kind!r}, idx={self.index}, scale={self.scale!r})"


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
            # Update ID ver4-6, R1.2(b): an optional 'scale' key inside
            # this SAME 'tunable' dict (no field-list index moves -
            # invariant I5 preserved). 'log' is only valid with a
            # strictly-positive lower bound (log(0) is undefined) - caught
            # here, at search-configuration time, as a schema error rather
            # than surfacing later as a silent NaN inside _decode().
            scale = tunable.get('scale', 'linear')
            if scale not in ('linear', 'log'):
                raise ValueError(
                    f"{model_name!r} field {name!r} declares unknown tunable "
                    f"scale {scale!r}; expected 'linear' or 'log'."
                )
            if scale == 'log' and not (tunable['low'] > 0):
                raise ValueError(
                    f"{model_name!r} field {name!r} declares tunable scale='log' "
                    f"but tunable['low']={tunable['low']!r} is not > 0 - a "
                    f"logarithmic mapping requires a strictly positive lower "
                    f"bound. Fix hparam_specs.py's HPARAM_SPECS entry for this "
                    f"model/field (raise 'low' above 0, or remove 'scale': 'log')."
                )
            specs.append(ParamSpec(
                idx, name, ftype,
                low=tunable['low'], high=tunable['high'], step=tunable.get('step'),
                scale=scale,
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
    """Update ID ver4-6, R1.2(a): every dimension - int, float, categorical
    - is now encoded on the SAME unit interval [0, 1], regardless of its
    own native range or scale. This is what lets a single isotropic length
    scale (bayes_opt's GaussianProcessRegressor - see blueprint E1) resolve
    every tunable dimension equally well: before this change, a shared
    length scale fitted on raw NATIVE units (e.g. 'Epoch' range 190 vs
    'Decay' range 0.01) was blind to whichever dimension had the smaller
    native range, driving held-out rank correlation NEGATIVE (measured
    -0.059 on the real GAT_* tunable box). Native-unit decoding still
    happens in _decode() below; nothing downstream of _decode() ever sees
    a unit-cube value."""
    return [(0.0, 1.0)] * len(specs)


def _snap_to_step(value, low, high, step):
    """Update ID ver4-6, R1.2(c): round `value` onto the nearest reachable
    point of the `step`-spaced axis starting at `low`, then clamp back into
    [low, high]. Previously only Grid search's own axis-building honoured
    `step` - every other algorithm decoded a raw `int(round(v))`, so e.g.
    rrBLUP's 'Iteration number' (declared step=2000) was searched over all
    18,001 integers in [2000, 20000] instead of the 10 the schema actually
    declares reachable - near-duplicate evaluations that inform the
    surrogate no better than the step points already do, for no modelling
    benefit."""
    if not step:
        return value
    n_steps = round((value - low) / step)
    snapped = low + n_steps * step
    return min(max(snapped, low), high)


def _decode(specs, x):
    """Update ID ver4-6, R1.2(a)/(b)/(c): `x` is now always a unit-cube
    point (every coordinate in [0, 1] - see _encode_bounds() above), mapped
    back here to each dimension's own native units, honouring its declared
    'scale' ('linear' or 'log') and 'step'. This is the ONLY place that
    mapping happens, so every search algorithm that routes through it
    (Random, Bayesian, Nelder-Mead, Powell - Grid builds its own on-step
    native axes directly and is unaffected) sees an identical,
    isotropically-scaled search space."""
    values = {}
    for s, v in zip(specs, x):
        u = min(max(float(v), 0.0), 1.0)
        if s.kind in ('int', 'float'):
            if s.scale == 'log':
                log_low, log_high = math.log(s.low), math.log(s.high)
                native = math.exp(log_low + u * (log_high - log_low))
            else:
                native = s.low + u * (s.high - s.low)
            native = _snap_to_step(native, s.low, s.high, s.step)
            native = min(max(native, s.low), s.high)
            values[s.index] = int(round(native)) if s.kind == 'int' else float(native)
        else:
            i = int(math.floor(u * len(s.choices)))
            i = min(max(i, 0), len(s.choices) - 1)
            values[s.index] = s.choices[i]
    return values


def _encode_from_params(specs, params):
    """Inverse of _decode(): given a full HPARAMETERS[model]-shaped params
    list (e.g. the user's own current/default values), return the UNIT-
    CUBE-encoded x (ver4-6 R1.2(a)) a search algorithm would need to
    reproduce it - so those values can be scored through the exact same
    objective() every search candidate is scored through (see the
    default-vs-winner safeguard in tune_model_hyperparameters()). A
    categorical value not found in s.choices (shouldn't happen for a
    well-formed HPARAM_SPECS, but not this function's place to enforce
    that) falls back to index 0 rather than raising - that only affects
    the safeguard comparison below, never the params actually used for a
    fit."""
    x = []
    for s in specs:
        val = params[s.index]
        if s.kind == 'categorical':
            try:
                i = s.choices.index(val)
            except ValueError:
                i = 0
            # Centre of that index's own unit-cube slice - the stable
            # inverse of _decode()'s floor(u * len(choices)) mapping.
            x.append((i + 0.5) / len(s.choices))
        else:
            native = min(max(float(val), s.low), s.high)
            if s.scale == 'log':
                low = max(s.low, 1e-300)
                native = max(native, low)
                span = math.log(s.high) - math.log(low)
                u = (math.log(native) - math.log(low)) / span if span else 0.0
            else:
                span = s.high - s.low
                u = (native - s.low) / span if span else 0.0
            x.append(min(max(u, 0.0), 1.0))
    return np.array(x, dtype=float)


def _random_point(specs, rng):
    """Update ID ver4-6, R1.2(a): a uniform draw on the unit cube -
    _decode() above then maps it into native space, so this reaches the
    same native-space distribution as before for a 'linear'-scaled field,
    and a log-uniform native distribution for a 'scale': 'log' field (the
    correct prior for a quantity - e.g. a learning rate - that spans
    orders of magnitude)."""
    return rng.uniform(0.0, 1.0, size=len(specs))


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
#
# Update ID ver4-5, R2 (blueprint §4.3): a value may be a single int OR a
# tuple of ints, for a model with more than one such flag - GAT_prior_
# knowledge is the first (marker_effect at 8, and its new emit_interaction
# at 10). _disable_explainability() below normalises either shape.
DIAGNOSTIC_FLAG_FIELDS = {
    'RF': 5, 'ExtraTrees': 5, 'GBDT': 6, 'XGBoost': 6,
    # Update ID ver4-5, R2 Stage 11: EBM's own 'get_interaction' flag is
    # unusual for this dict - it doesn't just gate a post-hoc explanation
    # step, it also controls whether EBM's OWN FIT searches for pairwise
    # terms at all (blueprint §4.2 Layer 3: "native pairwise terms are
    # the model"). Forcing it False during a hyperparameter search trial
    # therefore means the CANDIDATE is scored on a slightly simpler
    # model class (no pairwise terms) than the eventual confirmatory fit
    # (which restores the user's real setting) - a disclosed, accepted
    # trade-off (ver4-5 Change Summary §7/§8): running the pairwise
    # search on every trial would be far more expensive for a benefit
    # (interaction output) the search itself never uses to score a
    # candidate (only pearson_valid/mse_valid do).
    'EBM': 4,
    'SVR': (6, 11), 'KNN': (3, 8), 'MLP': 8,
    'RKHS': (3, 10),
    'GBLUP': (2, 9),
    # Update ID ver4-5, R2 Stage 10: rrBLUP/BayesB now also gain a
    # SURROGATE interaction toggle (index 4 for both - see
    # genomic_prediction.py::_r_model_interaction_fields()). Forcing it
    # off during search matters more here than almost anywhere else in
    # this dict: the surrogate step fits an entire extra
    # RandomForestRegressor (default 200 trees) PER TRIAL if left on -
    # by far the most expensive diagnostic step a tuning candidate could
    # accidentally pay for.
    'rrBLUP': 4, 'BayesB': 4,
    'GAT_prior_knowledge': (8, 10),  # marker_effect, emit_interaction (R2)
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
        # Update ID ver4-5, R2: idx may now be a single int (every
        # pre-existing entry - unchanged behaviour) or a tuple of ints
        # (a model with more than one explainability-gating flag).
        indices = idx if isinstance(idx, tuple) else (idx,)
        for _i in indices:
            if _i < len(params):
                params[_i] = False
    return params


def _enforce_mcmc_burnin_invariant(model_name, params):
    """Ensures burnIn < nIter for the BGLR-backed MCMC models
    (MCMC_LENGTH_FIELDS), no matter where nIter/burnIn came from - a
    cheapened search candidate, an uncheapened one (reduced_cost_search=
    False), or the winning hyperparameters selected for the final
    confirmatory fit.

    Both 'Iteration number' (nIter) and 'Burn-in' (burnIn) are tunable as
    INDEPENDENT dimensions in HPARAM_SPECS (see hparam_specs.py), so
    nothing about the search space itself stops the optimiser from
    sampling/selecting a combination where burnIn >= nIter. When that
    happens, BGLR has zero (or a degenerate single) post-burn-in MCMC
    sample to average over its posterior, so every model built on it
    (rrBLUP/GBLUP/BayesB/RKHS) collapses to a near-constant prediction -
    which has zero variance, making the validation Pearson r undefined
    (NaN). This is exactly the failure mode make_objective() below scores
    as a failure (ver4-6 R2.2: the worst-finite-score-seen-so-far, minus a
    margin - see FAILURE_FLOOR/_failure_score()) for candidates evaluated
    during search - this function is what keeps that same invalid
    combination from ever being handed to
    run_model_fn in the first place, at every call site, including the
    one call site (the final confirmatory fit) that isn't scored at all
    and so can't be caught that way.
    """
    fields = MCMC_LENGTH_FIELDS.get(schema_key_of(model_name))
    if fields is None:
        return params
    params = list(params)
    n_idx, b_idx = fields
    if params[b_idx] >= params[n_idx]:
        params[b_idx] = max(1, params[n_idx] // 4)
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
    return _enforce_mcmc_burnin_invariant(model_name, params)


# Update ID ver4-6, R4.2a: generalises _cheapen_mcmc()'s idea (temporarily
# shrink an expensive fit-size hyperparameter during the SEARCH phase only,
# never the final confirmatory fit) from the four BGLR/MCMC models to every
# other model whose fit cost scales with one hyperparameter this module can
# point at directly - tree/boosting-round counts and neural epoch counts,
# which is where a Bayesian search's budget is actually spent (blueprint
# R4.1's cost model: 38 model fits per (task, model) at the default
# budget). Models already handled by MCMC_LENGTH_FIELDS are skipped here to
# avoid cheapening the same underlying fit cost twice.
FIDELITY_FIELDS = {
    # model: {positional index: (scale, floor)} - `scale` is applied to the
    # BASE (untuned/current) value of that field; `floor` is the minimum
    # cheapened value regardless of how small `scale * base` would be.
    'RF': {0: (0.25, 100)},            # Tree number
    'ExtraTrees': {0: (0.25, 100)},    # Tree number
    'GBDT': {0: (0.30, 50)},           # Boosting iterations
    'XGBoost': {0: (0.30, 50)},        # Boosting iterations
    'MLP': {4: (0.30, 10)},            # Epoch
    'GAT_infinitesimal': {4: (0.30, 10)},              # Epoch
    'GAT_fully_connected': {4: (0.30, 10)},             # Epoch
    'GAT_prior_knowledge': {4: (0.30, 10)},             # Epoch
    'GAT_biological_prior_knowledge': {4: (0.30, 10)},  # Epoch
    'GAT_infinitesimal_node_level': {4: (0.30, 10)},    # Epoch
    'EBM': {3: (0.50, 2)},             # Outer bags
}


def _cheapen_fidelity(model_name, params, base_params, tuned_indices):
    """Search-phase-only low-fidelity rung (ver4-6 R4.2a), mirroring
    _cheapen_mcmc()'s own contract exactly: a field that is ITSELF being
    tuned is left alone (the search owns that dimension); a model already
    present in MCMC_LENGTH_FIELDS is skipped here entirely (its own
    fidelity rung is _cheapen_mcmc(), applied separately - see
    make_objective() below for the call order). Never applied to the final
    confirmatory fit - make_objective() only calls this inside the
    candidate-scoring `objective()` closure, never after the search
    concludes."""
    key = schema_key_of(model_name)
    if key in MCMC_LENGTH_FIELDS:
        return params
    fields = FIDELITY_FIELDS.get(key)
    if fields is None:
        return params
    params = list(params)
    for idx, (scale, floor) in fields.items():
        if idx in tuned_indices:
            continue
        if idx >= len(base_params):
            continue
        params[idx] = max(floor, int(round(base_params[idx] * scale)))
    return params


# --------------------------------------------------------------------------- #
# Objective
# --------------------------------------------------------------------------- #

# Update ID ver4-6, R2.2: SCORE_CAP is now a FINITE-VALUE CLIP only (a
# pathological but genuinely finite score - e.g. from a near-zero
# validation MSE - still cannot become inf); it is NO LONGER used as the
# score for a FAILED candidate (see FAILURE_FLOOR/_failure_score() below -
# that used to be this constant's job, at magnitude 1e6). Dropped from 1e6
# to 100.0: the live objective range for `r - mse/var` is roughly [-3, +1]
# (see the scoring comment inside `objective()` below), so 100.0 stays far
# outside anything a real score can reach while no longer being six orders
# of magnitude off. The OLD 1e6 value, previously ALSO used as the failure
# sentinel, is what collapsed the Gaussian-process surrogate's fitted
# length scale to its lower bound and flattened its posterior mean to a
# constant the moment a single candidate failed (blueprint E2: fitted
# length scale 0.513 -> 1e-5, posterior-mean spread 0.377 -> 0.000).
SCORE_CAP = 100.0

# Update ID ver4-6, R2.2: a failed / non-finite / constant-predictor
# candidate is now scored as strictly worse than every FINITE score
# actually observed so far THIS SEARCH (see _failure_score() inside
# make_objective() below), never as a fixed extreme constant - so its
# magnitude always stays on the same order as real observations and
# bayes_opt's `normalize_y=True` target standardisation is never distorted
# by an outlier six orders of magnitude away. FAILURE_FLOOR is only ever
# used before ANY candidate has succeeded yet this search (the very first
# evaluation happens to fail) - roughly 3x worse than the worst plausible
# real score for `r - mse/var` (whose own floor is roughly -3, for an
# anti-correlated, high-error model).
FAILURE_FLOOR = -10.0
# How much worse than the worst-finite-score-seen-so-far a later failure
# is scored - keeps every failure strictly ordered below every real
# observation (so argmax/EI can never select one) without introducing a
# second cliff of its own fixed magnitude the way the old sentinel did.
FAILURE_MARGIN = 0.05


def _actual_variance(valid):
    """Variance of the validation set's own phenotype column (its last
    column, matching every model's own train/valid/test convention - see
    e.g. models/BayesB.R). Returns None when it can't be computed (empty/
    degenerate valid, non-numeric column, etc.) - callers fall back to
    raw (unnormalised) MSE in that case, which is the pre-existing
    behaviour this only refines, not replaces."""
    try:
        actual = np.asarray(valid.iloc[:, -1], dtype=float)
        var = float(np.nanvar(actual))
    except Exception:
        return None
    return var if np.isfinite(var) and var > 0 else None


def _split_train_rows(train, seed):
    """Row-only train/validation split of `train`, used by the repeated-
    resample objective (ver4-6 R3.2c). NEVER touches the task's own
    `valid` or `test` frames (invariant I7 - the objective may only ever
    consume the current task's TRAINING split when estimating an inner
    resample). A fixed 75/25 inner split mirrors the shape of a typical
    outer RATIO closely enough for a tuning-only estimate - it is not the
    user's own configured RATIO, and never touches it. Returns
    `(None, None)` when `train` is too small to split meaningfully; the
    caller falls back to `valid_repeats=1` in that case."""
    if len(train) < 4:
        return None, None
    rng = np.random.default_rng(seed)
    idx = train.index.to_numpy()
    shuffled = idx.copy()
    rng.shuffle(shuffled)
    n_valid = max(1, int(round(len(shuffled) * 0.25)))
    n_valid = min(n_valid, len(shuffled) - 1)
    va_idx = shuffled[:n_valid]
    tr_idx = shuffled[n_valid:]
    return train.loc[tr_idx], train.loc[va_idx]


def make_objective(model_name, base_params, specs, train, valid, test,
                    run_model_fn, RESULT_NAME, reduced_cost_search=True,
                    valid_repeats=1, inner_split_seed_base=0):
    """Returns f(x) -> score-to-maximise (float) for a continuous-encoded
    candidate x.

    By default (`valid_repeats=1`, today's behaviour) a candidate is
    scored ONCE, on the VALIDATION set only. When `valid_repeats > 1`
    (ver4-6 R3.2c), a candidate is instead scored as the MEAN over
    `valid_repeats` independent inner train/validation resamples drawn
    from the current task's own TRAINING split only - the task's real
    `valid`/`test` frames are never consumed by the inner loop, and remain
    available afterwards for the default-vs-winner comparison, for
    `Prediction_result_valid.csv`, and for weighted-ensemble fitting,
    exactly as today (invariant I7).

    Parameters
    ----------
    valid_repeats : int, default 1. Reduces the score's own sampling
        variance (standard error shrinks roughly by 1/sqrt(valid_repeats))
        at a cost of `valid_repeats`x model fits per candidate - see the
        HP_TUNE_VALID_REPEATS design note in genomic_prediction.py's
        HP_TUNE dispatch for why this ships default-1.
    inner_split_seed_base : int, default 0. Base seed for the inner
        resamples; genomic_prediction.py derives this from the current
        task's own `sample` (replicate) value, so the sequence of inner
        splits is deterministic and identical regardless of sharding
        (invariant I9), like every other seeded operation in GP().
    """
    tuned_indices = {s.index for s in specs}
    # Computed once (the validation split doesn't change across
    # candidates) - see the scoring fix below for why this is needed.
    valid_var = _actual_variance(valid)

    effective_repeats = max(1, int(valid_repeats))
    # Update ID ver4-6, R3.2c failure mode: never silently reduce
    # valid_repeats without saying so - the codebase's "log negative
    # states unconditionally" discipline (architecture doc §17). Requires
    # at least 4 rows per inner fold as a floor for a meaningful 75/25
    # split; falls back to scoring once on the task's own `valid` split.
    if effective_repeats > 1 and len(train) < 4 * effective_repeats:
        print(f"[hyperparameter_tuning] NOTE: {model_name}'s training split "
              f"({len(train)} rows) is too small to draw {effective_repeats} "
              f"independent inner resamples from (need at least "
              f"{4 * effective_repeats}) - falling back to scoring once on "
              f"this task's own validation split for this search "
              f"(HP_TUNE_VALID_REPEATS effectively 1 for this task only).")
        effective_repeats = 1

    # Update ID ver4-6, R2.2: closure-local tracking of the worst FINITE
    # score seen so far THIS SEARCH - a failure is scored relative to this
    # (see _failure_score() below), not as a fixed extreme constant, so it
    # never distorts the Gaussian-process surrogate's target
    # standardisation the way the old `-SCORE_CAP` sentinel did (E2).
    # Deliberately closure-local, never shared across processes - a batch-
    # mode worker (models/parallel_search.py) falls back to FAILURE_FLOOR
    # for its own first failure, which is already well-scaled on its own
    # (RK-10, disclosed in this module's own R2 change).
    state = {'worst_finite': None}

    def _failure_score():
        w = state['worst_finite']
        if w is None:
            return FAILURE_FLOOR
        return w - max(FAILURE_MARGIN, 0.10 * abs(w))

    def _score_once(fit_train, fit_valid, params):
        """Fit once on `fit_train`, score once on `fit_valid`. Returns
        (score, succeeded). `succeeded=False` scores are ALREADY the
        correctly-computed failure score - the caller does not need to
        special-case them further, only decide whether to update
        `state['worst_finite']` from them (it must not: only a genuine,
        finite score is ever "the worst finite score seen so far")."""
        fit_var = valid_var if fit_valid is valid else _actual_variance(fit_valid)
        try:
            result = run_model_fn(model_name, fit_train, fit_valid, test, params, RESULT_NAME)
        except Exception as exc:
            # Req 2 fix (2026-09): the full traceback is now included,
            # not just repr(exc) - every prior investigation of a search-
            # candidate failure in this codebase hit the same wall (see
            # REQ2_Handoff.md §6 item 1): repr(exc) alone cannot
            # distinguish "raised inside run_model_fn's own R call" from
            # "raised inside pipeline_utils.r_list_get() while unpacking
            # that call's result" from any other frame in between, and
            # every occurrence of this print is, by construction, a
            # SWALLOWED exception with no other record of where it came
            # from. Printed on every candidate failure (not just the
            # first) deliberately - a search that fails identically many
            # times in a row is itself diagnostic (see the same handoff
            # note), and this is the only place that fact is visible.
            print(f"[hyperparameter_tuning] Candidate failed for {model_name} "
                  f"({type(exc).__name__}: {exc}); scored as worst and skipped.\n"
                  f"{traceback.format_exc()}")
            return _failure_score(), False

        r = result.get('pearson_valid')
        mse = result.get('mse_valid')
        if r is None or mse is None or not np.isfinite(mse):
            return _failure_score(), False
        if not np.isfinite(r):
            # Undefined correlation means this candidate's validation
            # predictions have ZERO variance (a constant/degenerate
            # predictor - e.g. RF collapsing to a single leaf per tree with
            # too few training samples relative to min_samples_leaf). That
            # provides no genomic-prediction value at all and must never be
            # preferred over a model that at least attempts to
            # differentiate individuals, even one that currently
            # generalises poorly (negative r). Constant predictors are
            # scored as the worst possible outcome, never as a "neutral"
            # 0.0 (which most real, poorly-generalising candidates - r<0 -
            # would then lose to, despite being strictly more useful).
            return _failure_score(), False
        if mse < 0:
            return _failure_score(), False

        # Bug fix (tuned models underperforming their own untuned
        # defaults): this used to be `score = r / mse`. On the small
        # validation splits typical of genomic-prediction sample sizes,
        # MSE is a noisy estimate, and a RATIO is extremely sensitive to
        # that noise right where it matters most - a validation MSE that
        # happens to be tiny purely from sampling luck (not genuine model
        # quality) inflates 1/mse arbitrarily and swamps r's own
        # contribution entirely, so the search ends up chasing whichever
        # candidate got lucky on THIS validation split's noise, rather
        # than the candidate that actually generalises best.
        #
        # Normalising mse by the validation set's own phenotype variance
        # (giving an R2-like, unitless, roughly-[0,~2] quantity instead of
        # a raw, trait-scale-dependent error) and combining it with r
        # ADDITIVELY rather than as a ratio keeps both terms on comparable
        # scales and removes the "explodes near zero" failure mode a
        # denominator has. Falls back to raw mse (added as a penalty
        # rather than divided) when the variance isn't computable, which
        # only softens - it doesn't reintroduce - the old ratio's blow-up.
        mse_penalty = (mse / fit_var) if fit_var else mse
        score = r - mse_penalty
        if not np.isfinite(score):
            return _failure_score(), False
        return float(np.clip(score, -SCORE_CAP, SCORE_CAP)), True

    def _prepare_params(x):
        params = list(base_params)
        for idx, val in _decode(specs, x).items():
            params[idx] = val

        params = _disable_explainability(model_name, params)
        # Applied unconditionally (not just under reduced_cost_search) so an
        # uncheapened candidate (reduced_cost_search=False) can never be
        # scored on an invalid burnIn>=nIter combination either -
        # _cheapen_mcmc already re-applies this same invariant itself after
        # cheapening, so this is a no-op in that branch, not a duplicate
        # correction.
        params = _enforce_mcmc_burnin_invariant(model_name, params)
        if reduced_cost_search:
            params = _cheapen_mcmc(model_name, params, base_params, tuned_indices)
            # Update ID ver4-6, R4.2a: generalised low-fidelity rung for
            # every non-MCMC model with a registered FIDELITY_FIELDS entry
            # (trees/boosting rounds/epochs) - applied AFTER _cheapen_mcmc
            # so the two never double-cheapen the same model
            # (_cheapen_fidelity is itself a no-op for any model already
            # in MCMC_LENGTH_FIELDS).
            params = _cheapen_fidelity(model_name, params, base_params, tuned_indices)
        return params

    def objective(x):
        params = _prepare_params(x)

        if effective_repeats <= 1:
            score, ok = _score_once(train, valid, params)
            if ok:
                state['worst_finite'] = (
                    score if state['worst_finite'] is None else min(state['worst_finite'], score)
                )
            return score

        # ver4-6 R3.2c: mean over `effective_repeats` independent inner
        # resamples of the TRAINING split only.
        fold_scores = []
        for b in range(effective_repeats):
            tr_b, va_b = _split_train_rows(train, inner_split_seed_base + b)
            if tr_b is None:
                fold_scores.append(_failure_score())
                continue
            # Leakage guard (AC-R3.5, invariant I7): every inner
            # validation fold's index must be a subset of `train`'s own
            # index - never `valid`'s or `test`'s. This must never fire;
            # if it does, `_split_train_rows()` itself has a bug.
            assert set(va_b.index).issubset(set(train.index)), (
                f"[hyperparameter_tuning] internal error: an inner "
                f"resample validation fold for {model_name} was not a "
                f"subset of the task's own training split - this would be "
                f"a data-leakage bug (invariant I7) and must never happen."
            )
            fold_score, ok = _score_once(tr_b, va_b, params)
            fold_scores.append(fold_score)
            if ok:
                state['worst_finite'] = (
                    fold_score if state['worst_finite'] is None
                    else min(state['worst_finite'], fold_score)
                )
        return float(np.mean(fold_scores))

    return objective


# --------------------------------------------------------------------------- #
# Search algorithms - each returns (best_x, best_score, trials,
# default_score_or_None) as of ver4-6 R4.2c/R4.4. `default_score` is the
# search's own evaluation of the caller-supplied `default_x` where one was
# actually scored (every algorithm except Grid, which never evaluates
# `default_x` at all - see search_grid()'s own comment), letting
# tune_model_hyperparameters() skip a purely redundant re-evaluation of
# the exact same point. `None` means "this algorithm did not score
# default_x itself" - the caller falls back to evaluating it fresh, which
# is byte-for-byte today's (pre-ver4-6) behaviour.
# --------------------------------------------------------------------------- #

# ver4-4 R3.f (blueprint §2.3.2/§2.3.3): shared candidate-evaluation helper
# for search_grid()/search_random() below - the only two algorithms the
# blueprint marks parallelisable (Bayesian/Nelder-Mead/Powell are
# inherently sequential local/surrogate searches and are NOT touched here -
# see search_bayesian()/_multistart_local_search()'s own call sites, which
# never pass n_jobs).
#
# n_jobs<=1 (the default, every existing caller before this stage) takes
# the exact serial branch below - no joblib import is even attempted -
# which is byte-for-byte today's code path with zero new failure surface.
#
# n_jobs>1 attempts a joblib.Parallel(..., backend='loky') fan-out - a
# PROCESS backend only, never threads (invariant I10: the BGLR-backed
# models rrBLUP/GBLUP/BayesB/RKHS are all tunable - see MCMC_LENGTH_FIELDS
# above - and BGLR's saveAt trace-file naming depends on distinct PIDs;
# threads sharing one PID would collide exactly as R3.d's own design
# record already explains for that requirement). Every candidate `x` is
# submitted in ORDER (`xs` is a plain, pre-built list - never a generator
# consumed lazily per worker), and joblib.Parallel is documented to always
# return results in call-SUBMISSION order regardless of completion order
# or backend (the same guarantee genomic_prediction.py's own
# _run_gblup_or_rkhs() Shapley fan-out already relies on) - so the
# (x, score) pairing below, and therefore which candidate ends up
# `best_x`/`best_score` on a tie (`score > best_score`, strictly-greater,
# first-write-wins), is IDENTICAL to n_jobs=1 evaluated serially in the
# same order.
#
# `objective` is supplied by THIS MODULE's caller (see the module
# docstring - this file deliberately imports nothing project-specific) and
# this function has no way to know what it closes over. Whenever that
# closure is not itself process-picklable - e.g. genomic_prediction.py's
# own run_model_fn, which wraps GP()'s _call_model() and (see that
# function's own module-level R3.d design note) unconditionally closes
# over the live, per-process rpy2-bound rrBLUP/GBLUP/BayesB/RKHS R
# function objects GP() sources near its own top, for EVERY model tuned
# through it, not only the four R ones - joblib's submission-time pickling
# fails with an ordinary Python exception (TypeError/PicklingError, not a
# hang: rpy2's Sexp-backed objects have no __reduce__/__getstate__ support
# for a C-extension SEXP proxy tied to one specific embedded R session).
# That exception is caught here, ONCE, around the whole batch (not
# per-candidate: a closure that fails to pickle for candidate 1 fails
# identically for every other candidate, so there is nothing to gain by
# retrying each one individually), logged, and every candidate in `xs` is
# then evaluated serially instead - exactly n_jobs=1's own code path,
# just reached via a different route. Never raises; correctness never
# depends on which path actually ran, only wall-clock time does - the
# same "fall back to one serial call, print, continue" contract
# pipeline_utils.parallel_shap_values() and genomic_prediction.py's
# _run_gblup_or_rkhs() already use for an identical class of problem.
def _evaluate_batch(objective, xs, n_jobs, worker_init=None, worker_init_args=()):
    """
    Parameters
    ----------
    worker_init, worker_init_args : ver4-5 R1.i - an optional worker
        PROCESS initialiser (and its plain-data arguments) run once, in
        every freshly-spawned worker, BEFORE that worker evaluates its
        first candidate. `None` (every pre-R1.i caller, and every call
        with n_jobs<=1) reproduces this function's exact pre-R1.i
        behaviour - no `initializer=` is even passed to
        `joblib.Parallel`. Supplying one restores per-process state
        (this run's own resolved compute-resource dict, R environment,
        GPU semaphore) that a freshly spawned worker does NOT otherwise
        inherit - see genomic_prediction.py::_trial_worker_init()'s own
        docstring for exactly why this matters (silent oversubscription
        / device mismatch, not a crash, if omitted).
    """
    if n_jobs is None or n_jobs in (0, 1) or len(xs) <= 1:
        return [objective(x) for x in xs]
    try:
        from joblib import delayed
        return _loky_parallel(n_jobs, worker_init, worker_init_args)(
            delayed(objective)(x) for x in xs
        )
    except Exception as exc:
        print(f"[hyperparameter_tuning] NOTE: parallel trial evaluation failed "
              f"({type(exc).__name__}: {exc}) - falling back to serial evaluation "
              f"for this search (n_jobs={n_jobs} requested). This is commonly "
              f"caused by a run_model_fn whose closure cannot cross a process "
              f"boundary (e.g. one that reaches a live, per-process rpy2 R "
              f"session - see genomic_prediction.py's own R3.d design note for "
              f"why that can happen for every model, not only the R-backed "
              f"ones). This search's own RESULT is unaffected, only wall-clock "
              f"time - the trials below were evaluated serially instead.")
        return [objective(x) for x in xs]


def search_grid(objective, specs, n_points=5, default_x=None, n_jobs=1,
                 worker_init=None, worker_init_args=()):
    # default_x accepted for a uniform SEARCH_FUNCS interface (see
    # run_search()) but not specially used here: Grid already evaluates
    # every combination on its own axes, and the default's own value for a
    # given field won't generally fall exactly on one of those axis points
    # anyway, so there's no clean way to "seed" a grid with it.
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

    xs = [np.array(combo, dtype=float) for combo in itertools.product(*axes)]
    scores = _evaluate_batch(objective, xs, n_jobs, worker_init=worker_init,
                              worker_init_args=worker_init_args)

    best_x, best_score, trials = None, -math.inf, []
    for x, score in zip(xs, scores):
        trials.append((x, score))
        if score > best_score:
            best_score, best_x = score, x
    if best_x is None:  # zero tunable dims edge case
        best_x = np.array([])
    # Grid never scores `default_x` itself (see the module-level comment
    # for this file's dispatch-time note above and search_grid()'s own
    # docstring) - no default_score to hand back.
    return best_x, best_score, trials, None


def search_random(objective, specs, n_iter=40, seed=0, default_x=None, n_jobs=1,
                   worker_init=None, worker_init_args=()):
    rng = np.random.default_rng(seed)
    best_x, best_score, trials = None, -math.inf, []
    default_score = None
    # Evaluate the user's own default/current hyperparameters as an
    # explicit extra trial, on the same footing as every randomly-sampled
    # one - cheap (one extra evaluation), and means the "best of n_iter
    # random draws" this search returns is never worse than the point the
    # user started from, without waiting for tune_model_hyperparameters()'s
    # own post-search safeguard to catch it. Kept as its own single, always-
    # serial evaluation (n_jobs never applies to this one point) - it's
    # cheap enough on its own, and keeping it out of the batch below means
    # the batch's own random draws are generated exactly the same way
    # (same rng calls, same order) regardless of n_jobs.
    if default_x is not None:
        default_score = objective(default_x)
        trials.append((default_x, default_score))
        best_x, best_score = default_x, default_score

    # ver4-4 R3.f: every random draw is generated UP FRONT, from the same
    # rng in the same order, before any evaluation happens - so the
    # SEQUENCE of x's drawn (and therefore this search's own result) is
    # identical regardless of n_jobs; only whether they are then scored
    # serially or via _evaluate_batch()'s own process fan-out changes.
    xs = [_random_point(specs, rng) for _ in range(max(1, n_iter))]
    scores = _evaluate_batch(objective, xs, n_jobs, worker_init=worker_init,
                              worker_init_args=worker_init_args)
    for x, score in zip(xs, scores):
        trials.append((x, score))
        if score > best_score:
            best_score, best_x = score, x
    # ver4-6 R4.2c: this search already scored `default_x` above (when one
    # was supplied) - handed back so tune_model_hyperparameters() does not
    # pay for a second, identical evaluation of it.
    return best_x, best_score, trials, default_score


def _resolve_q(n_jobs, batch, batch_max):
    """This search's batch WIDTH `q` (ver4-5 R1.b) - collapses to 1
    (today's exact serial `optimizer.maximize()` path) whenever batch
    mode is off, `n_jobs` isn't a usable positive worker count, or (via
    the caller's own `nested_safe_n_jobs()` resolution, upstream of this
    module - see pipeline_utils.py) this call is itself already running
    inside a daemonic worker process. Never raises."""
    if not batch:
        return 1
    try:
        _n = int(n_jobs)
    except (TypeError, ValueError):
        return 1
    if _n <= 1:
        return 1
    return max(1, min(_n, max(1, int(batch_max))))


def search_bayesian(objective, specs, n_iter=30, init_points=5, seed=0, default_x=None,
                     n_jobs=1, batch=True, batch_max=8, liar='max',
                     worker_init=None, worker_init_args=(), domain_reduction='auto'):
    """
    Parameters
    ----------
    domain_reduction : {'auto', 'always', 'never'}, default 'auto' (ver4-6
        R1.2(d)). 'auto' enables sequential domain reduction (SDR) only
        when this model's tunable box has NO categorical dimension -
        SequentialDomainReductionTransformer only supports all-float
        parameter spaces; under unit-cube encoding a categorical dimension
        is still float-typed to bayes_opt, so SDR would run WITHOUT
        raising, but would progressively narrow a categorical axis toward
        a handful of its choices on the strength of a few early, noisy
        observations - permanently excluding the rest (see the ver4-6
        blueprint R1.1 "amplifiers" for a concrete SVR::Kernel-type
        example). 'auto' prints one [GP] NOTE: naming the categorical
        field(s) and skips constructing the transformer at all when it
        would apply. 'always' restores ver4-5 behaviour (SDR unconditional,
        including alongside categoricals). 'never' never constructs one.
    n_jobs, batch, batch_max, liar : ver4-5 R1.b (blueprint §3.4) - when
        `batch` is True AND `n_jobs` names more than one usable core
        (`q = _resolve_q(...) > 1`), this search runs in constant-liar
        BATCH mode (see models/parallel_search.py::batch_bayesian_
        maximise) instead of bayes_opt's own strictly-sequential
        `optimizer.maximize()`. `batch=False`, or `n_jobs<=1`, keeps
        today's EXACT serial code path below, byte-for-byte - the
        overwhelming majority of existing calls (every one predating
        this parameter). A batch-mode failure for any reason (library
        internals moved, submission-time picklability failure inside
        `_evaluate_batch()`, ...) falls back to the serial path below
        with one NOTE line - this function's own RESULT is never worse
        for having tried batch mode, only its wall-clock time and the
        exact sequence of candidates evaluated (see the `[GP] NUMERICS:`
        line below).
    worker_init, worker_init_args : ver4-5 R1.i - forwarded to
        `_evaluate_batch()` (batch mode's own trial-evaluation fan-out
        only; irrelevant to the serial path).
    """
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

    # Update ID ver4-6, R1.2(d): resolve whether SDR runs at all for this
    # search. Under unit-cube encoding (R1.2(a)) every dimension's own
    # (low, high) is now (0.0, 1.0), so the proportional per-dimension
    # `minimum_window` this codebase already preferred over one fixed
    # absolute value collapses to the SAME single fraction for every
    # dimension - `[0.02] * d` - which is simpler and exactly equivalent
    # to the prior intent (see the comment block further down this
    # function for the full "why a proportional floor" reasoning, still
    # accurate under the new encoding).
    _categorical_names = [s.name for s in specs if s.kind == 'categorical']
    if domain_reduction == 'always':
        _use_sdr = True
    elif domain_reduction == 'never':
        _use_sdr = False
    elif domain_reduction == 'auto':
        _use_sdr = not _categorical_names
        if _categorical_names:
            print(f"[GP] NOTE: sequential domain reduction skipped for this "
                  f"Bayesian hyperparameter search (HP_TUNE_BAYES_DOMAIN_"
                  f"REDUCTION='auto') because SequentialDomainReductionTransformer "
                  f"is not supported alongside a categorical tunable field: "
                  f"{', '.join(_categorical_names)}. Set HP_TUNE_BAYES_DOMAIN_"
                  f"REDUCTION='always' to force it on anyway (not recommended - "
                  f"it can permanently exclude some categorical choices after a "
                  f"handful of noisy early observations), or 'never' to silence "
                  f"this note.")
    else:
        raise ValueError(
            f"Unknown domain_reduction {domain_reduction!r}; expected "
            f"'auto', 'always', or 'never'."
        )

    def _make_bounds_transformer():
        if not _use_sdr or SequentialDomainReductionTransformer is None:
            return None
        return SequentialDomainReductionTransformer(minimum_window=[0.02] * len(bounds))

    def _make_acquisition():
        return ExpectedImprovement(xi=0.01) if ExpectedImprovement is not None else None

    q = _resolve_q(n_jobs, batch, batch_max)
    if q > 1 and len(specs) > 0:
        print(f"[GP] NUMERICS: Bayesian hyperparameter search running in BATCH mode "
              f"(q={q}, liar={liar!r}); candidate sequence differs from serial (q=1). "
              f"Set HP_TUNE_BAYES_BATCH=false to restore.")
        from .parallel_search import batch_bayesian_maximise

        def objective_batch(xs):
            return _evaluate_batch(objective, xs, n_jobs, worker_init=worker_init,
                                    worker_init_args=worker_init_args)

        try:
            _batch_best_x, _batch_best_score, _batch_trials = batch_bayesian_maximise(
                objective_batch, bounds, n_iter=n_iter, init_points=init_points, seed=seed, q=q,
                probe_points=([default_x] if default_x is not None else None),
                acquisition_factory=_make_acquisition,
                bounds_transformer_factory=_make_bounds_transformer,
                liar=liar,
            )
            # ver4-6 R4.2c: `probe_points=[default_x]` is always evaluated
            # FIRST, as round 0's first candidate (see
            # batch_bayesian_maximise()'s own docstring) - so `trials[0]`
            # IS the default's own score whenever one was supplied, with
            # no need for a second, separate evaluation of it.
            _batch_default_score = None
            if default_x is not None and _batch_trials:
                _first_x, _first_score = _batch_trials[0]
                if np.allclose(_first_x, default_x):
                    _batch_default_score = _first_score
            return _batch_best_x, _batch_best_score, _batch_trials, _batch_default_score
        except Exception as exc:
            print(f"[hyperparameter_tuning] NOTE: batch Bayesian search failed "
                  f"({type(exc).__name__}: {exc}) - falling back to the serial Bayesian "
                  f"search (q=1) for this model. This search's own RESULT is unaffected, "
                  f"only wall-clock time.")
            # Falls through to the serial body below, unchanged.

    # Further Bayesian-optimisation-quality improvements, on top of the
    # default-seeding above and the hard non-regression floor in
    # tune_model_hyperparameters() below:
    #
    # 1. Expected Improvement (xi=0.01) instead of bayes_opt's own default
    #    Upper Confidence Bound. EI's acquisition value is explicitly
    #    "how much better than the best point seen so far is this
    #    candidate likely to be", which matches this module's actual goal
    #    (beat the seeded default / current incumbent) more directly than
    #    UCB's generic explore/exploit trade-off. xi=0.01 is bayes_opt's
    #    own chosen default for EI elsewhere in its codebase.
    # 2. A larger GP observation-noise assumption (alpha). Each candidate
    #    here is scored from a SINGLE model fit on one (often small, for
    #    genomic-prediction sample sizes) validation split - repeating the
    #    exact same hyperparameters would not reproduce an identical score
    #    (validation-split sampling noise, plus genuine MCMC stochasticity
    #    for the BGLR-backed models). bayes_opt's default alpha (1e-6) is
    #    tuned for noiseless mathematical test functions and makes the
    #    surrogate overconfident about any single evaluation - a bigger
    #    alpha tells it to trust each individual score less and smooth
    #    across nearby points more, which is the standard mitigation for a
    #    noisy black-box objective. Not exposed as a BayesianOptimization
    #    constructor argument, so it's set directly on the underlying
    #    scikit-learn GaussianProcessRegressor afterwards; guarded because
    #    that's a private attribute of the bayes_opt library, not a stable
    #    part of its public API - if a future bayes_opt version changes
    #    its internals, this becomes a harmless no-op rather than a crash.
    # 3. Sequential domain reduction: progressively narrows pbounds toward
    #    the region the search's own evidence favours, concentrating the
    #    (typically modest, given how expensive each candidate is here)
    #    iteration budget where it matters most instead of continuing to
    #    sample the full original box throughout. Confirmed compatible
    #    with this function's plain per-dimension float bounds (verified
    #    separately against models/Bayesian_optimisation.py's CUSTOM
    #    multi-dimensional weight parameter, which is NOT compatible with
    #    it - bayes_opt raises "Domain reduction is only supported for
    #    all-FloatParameter optimization" there, which is exactly why this
    #    is only enabled here, not in that module).
    #
    #    minimum_window is set PER-DIMENSION, to a fraction of that
    #    dimension's own [low, high] range, rather than one fixed absolute
    #    value shared across every dimension - confirmed empirically (see
    #    models/Bayesian_optimisation.py's matching fix for the full
    #    experiment) that a fixed absolute floor biases the search: it
    #    represents a tiny fraction of a wide-ranging hyperparameter (e.g.
    #    'Iteration number', tunable 2000-20000) but a huge fraction of a
    #    narrow one (e.g. 'Prior probability of a nonzero effect (probIn)',
    #    tunable 0.01-0.9) - a single shared value can only ever be
    #    correctly scaled for one of them. A proportional per-dimension
    #    floor keeps the narrowing behaviour consistent across every
    #    tunable field regardless of that field's own natural scale.
    # 4. allow_duplicate_points=True: the explicit default_x probe below
    #    can otherwise collide with a later-suggested point (bayes_opt
    #    raises by default rather than silently skipping a duplicate),
    #    which would abort the whole search over a single re-tried point.
    bounds_transformer = _make_bounds_transformer()
    acquisition_function = _make_acquisition()
    optimizer = BayesianOptimization(f=wrapped, pbounds=pbounds, random_state=seed, verbose=0,
                                      acquisition_function=acquisition_function,
                                      bounds_transformer=bounds_transformer,
                                      allow_duplicate_points=True)
    try:
        optimizer._gp.set_params(alpha=1e-3)
    except Exception:
        pass

    # Bug fix (tuned models failing to outperform their own untuned
    # defaults - reported specifically against this Bayesian algorithm):
    # explicitly evaluate the user's own default/current hyperparameters
    # as this search's very FIRST point, rather than leaving the
    # optimiser's Gaussian-process surrogate to rely entirely on
    # init_points random draws to ever land anywhere near a known-
    # reasonable configuration. With a small init_points budget (a
    # handful of purely random samples across a wide search space), the
    # surrogate can spend its whole budget exploring regions with no
    # relation to a genuinely competitive configuration, and the
    # acquisition function has no anchor telling it "here is a baseline
    # actually worth beating" - only unrelated random samples. Seeding the
    # default first gives it that anchor from iteration one, so the
    # acquisition function can deliberately explore the NEIGHBOURHOOD of a
    # known-good point instead of searching blind - directly increasing
    # the chance of finding something that genuinely beats the default,
    # not just of not losing to it.
    #
    # This is complementary to, not a replacement for, the hard
    # default-vs-winner floor in tune_model_hyperparameters(): that floor
    # is what GUARANTEES the final result can never score worse than the
    # defaults (covers every algorithm, not just this one); seeding here
    # is what makes actually beating them more likely in the first place.
    #
    # Update ID ver4-6, R4.2c: evaluated ONCE, explicitly, right here -
    # via `optimizer.register()` (adds the observation to the GP directly,
    # no re-evaluation), rather than the pre-ver4-6 `optimizer.probe(...,
    # lazy=True)` (which queues it to be evaluated INSIDE maximize() below
    # instead - functionally equivalent for the optimiser, but hides the
    # resulting score from this function's own caller). Handing the score
    # back as this function's own 4th return value lets
    # tune_model_hyperparameters() skip its own former second, purely
    # redundant evaluation of the exact same point - one full model fit
    # saved per (task, model) under this algorithm, at unchanged total
    # search budget (`optimizer.maximize()` below still draws its own full
    # `init_points` random points; nothing here reduces that count).
    default_score = None
    if default_x is not None:
        default_score = objective(default_x)
        try:
            optimizer.register(params=dict(zip(names, default_x)), target=default_score)
        except Exception as exc:
            print(f"[hyperparameter_tuning] NOTE: could not register the default-"
                  f"parameter probe point with the Bayesian optimizer "
                  f"({type(exc).__name__}: {exc}) - continuing without seeding the "
                  f"search with it. The default-vs-winner safeguard in "
                  f"tune_model_hyperparameters() still applies regardless.")

    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    best = optimizer.max
    best_x = np.array([best['params'][n] for n in names])
    return best_x, best['target'], optimizer.res, default_score


def _run_one_restart(objective, x0, method, bounds, options):
    """MODULE-LEVEL (picklable - invariant I13) worker body for ONE
    Nelder-Mead/Powell local-search restart (ver4-5 R1.c). Deliberately
    NOT nested inside `_multistart_local_search()` - a nested closure
    that appends to a single, shared `trials` list is exactly the shape
    that breaks across a process boundary (each worker would only ever
    see, and mutate, its OWN empty copy - none of it would find its way
    back to the parent) - so this returns its own LOCAL trials list
    explicitly instead, for the caller to concatenate once every restart
    has finished.

    Returns
    -------
    (x, score, trials) - `trials` is this restart's own list of
    `(x, score)` tuples, in evaluation order.
    """
    trials_local = []

    def neg_obj(x):
        score = objective(x)
        trials_local.append((x.copy(), score))
        return -score

    res = minimize(neg_obj, x0, method=method, bounds=bounds, options=options)
    return res.x, -res.fun, trials_local


def _multistart_local_search(method, objective, specs, n_iter, seed, n_restarts,
                              default_x=None, n_jobs=1, worker_init=None,
                              worker_init_args=(), **method_options):
    """Nelder-Mead and Powell are local searches: on a mixed-type
    continuous relaxation with categorical dimensions, a single run can
    converge to a simplex/direction-set that never revisits a categorical
    choice it started away from (verified empirically in this project's own
    test suite - see test_hyperparameter_tuning.py). Multi-restart from
    several random points and keeping the best is the standard mitigation,
    and costs nothing extra beyond distributing the same total evaluation
    budget across restarts instead of spending it all on one.

    ver4-5 R1.c (blueprint §3.4): once R1.a makes `objective` itself
    process-picklable (see genomic_prediction.py::ModelTrialRunner),
    these restarts - already fully independent of each other, exactly
    like Grid/Random's own candidates - are dispatched across `n_jobs`
    worker PROCESSES via the same joblib fan-out-with-fallback pattern
    `_evaluate_batch()` already uses for Grid/Random, rather than a
    bespoke third implementation of it. `n_jobs<=1` (every pre-R1.c
    caller) keeps today's EXACT serial `for` loop below, byte-for-byte -
    no joblib import is even attempted in that case.
    """
    n_restarts = max(1, n_restarts)
    per_restart_iter = max(5, n_iter // n_restarts)
    bounds = _encode_bounds(specs)
    options = {'maxiter': per_restart_iter, **method_options}
    trials = []
    best_x, best_score = None, -math.inf

    rng = np.random.default_rng(seed)
    # One restart starts from the user's own default/current
    # hyperparameters (when available) instead of a random point, so
    # this local search always has at least one descent path anchored
    # at a known-reasonable configuration rather than relying on every
    # restart being purely random. Every x0 is drawn UP FRONT, from the
    # same rng in the same order, before any evaluation happens - so the
    # SEQUENCE of starting points (and therefore this search's own
    # result) is identical regardless of n_jobs; only whether restarts
    # are then run serially or via a process fan-out changes.
    x0_list = [
        default_x if (i == 0 and default_x is not None) else _random_point(specs, rng)
        for i in range(n_restarts)
    ]

    results = None
    if n_jobs is not None and n_jobs not in (0, 1) and n_restarts > 1:
        try:
            from joblib import delayed
            results = _loky_parallel(n_jobs, worker_init, worker_init_args)(
                delayed(_run_one_restart)(objective, x0, method, bounds, options)
                for x0 in x0_list
            )
        except Exception as exc:
            print(f"[hyperparameter_tuning] NOTE: parallel {method} restart fan-out failed "
                  f"({type(exc).__name__}: {exc}) - falling back to serial restarts "
                  f"(n_jobs={n_jobs} requested). This search's own result is unaffected, "
                  f"only wall-clock time.")
            results = None

    if results is None:
        results = [_run_one_restart(objective, x0, method, bounds, options) for x0 in x0_list]

    # First-write-wins on ties, in restart order - identical reduction to
    # the pre-R1.c serial loop's own `if -res.fun > best_score`, and
    # joblib.Parallel is documented to always return results in call-
    # SUBMISSION order (never completion order), so this is deterministic
    # regardless of n_jobs.
    for x, score, trials_local in results:
        trials.extend(trials_local)
        if score > best_score:
            best_score, best_x = score, x

    # Update ID ver4-6, R4.2c: restart 0 starts from `default_x` whenever
    # one was supplied (see x0_list's own construction above), and
    # scipy.optimize.minimize's Nelder-Mead/Powell implementations both
    # evaluate the objective AT their own starting point as the very
    # first call (needed to build the initial simplex / establish Powell's
    # first line-search baseline) - so restart 0's own FIRST recorded
    # trial IS the default's score. Reused here so
    # tune_model_hyperparameters() does not pay for evaluating it a second
    # time. `None` (not restart 0, or its first trial's x doesn't match
    # default_x for any reason) falls back to that fresh re-evaluation,
    # exactly today's (pre-ver4-6) behaviour.
    default_score = None
    if default_x is not None and results and results[0][2]:
        first_x, first_score = results[0][2][0]
        if np.allclose(first_x, default_x):
            default_score = first_score

    return best_x, best_score, trials, default_score


def search_nelder_mead(objective, specs, n_iter=40, seed=0, n_restarts=4, default_x=None,
                        n_jobs=1, worker_init=None, worker_init_args=()):
    return _multistart_local_search(
        'Nelder-Mead', objective, specs, n_iter, seed, n_restarts, default_x=default_x,
        n_jobs=n_jobs, worker_init=worker_init, worker_init_args=worker_init_args,
        xatol=1e-3, fatol=1e-3, adaptive=True,
    )


def search_powell(objective, specs, n_iter=40, seed=0, n_restarts=4, default_x=None,
                   n_jobs=1, worker_init=None, worker_init_args=()):
    return _multistart_local_search(
        'Powell', objective, specs, n_iter, seed, n_restarts, default_x=default_x,
        n_jobs=n_jobs, worker_init=worker_init, worker_init_args=worker_init_args,
        xtol=1e-3, ftol=1e-3,
    )


SEARCH_FUNCS = {
    'Grid': search_grid,
    'Random': search_random,
    'Bayesian': search_bayesian,
    'Nelder-Mead': search_nelder_mead,
    'Powell': search_powell,
}


def run_search(algorithm, objective, specs, budget_kwargs, default_x=None, n_jobs=1,
                worker_init=None, worker_init_args=(), bayes_batch=True, bayes_batch_max=8,
                bayes_liar='max', parallel_restarts=True, bayes_domain_reduction='auto'):
    """Dispatch to `algorithm`'s own search function. Returns whatever that
    function returns - as of ver4-6 R4.4, always a 4-tuple
    `(best_x, best_score, trials, default_score_or_None)` for every
    algorithm (see the module-level comment above search_grid() for what
    the 4th element means).

    ver4-4 R3.f / ver4-5 R1.b/R1.c: `n_jobs` (plus, for their own
    algorithms, the batch/restart controls below) is forwarded to EVERY
    algorithm now - 'Grid'/'Random' fan out across candidate
    evaluations, 'Bayesian' can additionally run in constant-liar BATCH
    mode (see search_bayesian()'s own docstring), and 'Nelder-Mead'/
    'Powell' can dispatch their own independent multi-restarts across
    processes (see _multistart_local_search()'s own docstring). Every
    one of these collapses to today's EXACT pre-ver4-5 serial code path
    whenever `n_jobs<=1` or its own enabling flag is off - this was
    previously true only for Grid/Random; as of ver4-5 it is true for
    every algorithm, so a caller that only ever sets `n_jobs=1` (or
    leaves the three new flags at their own conservative-off setting)
    observes zero behavioural change from before this update.

    `worker_init`/`worker_init_args` (ver4-5 R1.i) are forwarded
    unconditionally too - each algorithm's own fan-out path only
    actually USES them when it decides to fan out at all; a serial call
    never touches them.
    """
    if algorithm not in SEARCH_FUNCS:
        raise ValueError(f"Unknown hyperparameter search algorithm {algorithm!r}. "
                          f"Choose from {list(SEARCH_FUNCS)}.")
    kwargs = dict(budget_kwargs or {})
    if algorithm in ('Grid', 'Random'):
        kwargs['n_jobs'] = n_jobs
        kwargs['worker_init'] = worker_init
        kwargs['worker_init_args'] = worker_init_args
    elif algorithm == 'Bayesian':
        kwargs['n_jobs'] = n_jobs
        kwargs['batch'] = bayes_batch
        kwargs['batch_max'] = bayes_batch_max
        kwargs['liar'] = bayes_liar
        kwargs['worker_init'] = worker_init
        kwargs['worker_init_args'] = worker_init_args
        # Update ID ver4-6, R1.2(d): threaded through to
        # search_bayesian()'s own `domain_reduction=` keyword.
        kwargs['domain_reduction'] = bayes_domain_reduction
    elif algorithm in ('Nelder-Mead', 'Powell'):
        # parallel_restarts=False reproduces the pre-R1.c serial `for`
        # loop unconditionally, regardless of n_jobs - a separate switch
        # from Grid/Random/Bayesian's own n_jobs<=1 collapse, since a
        # user may want Grid/Random parallelised without also wanting
        # local-search restarts to fan out (e.g. while diagnosing R1.c
        # in isolation).
        kwargs['n_jobs'] = n_jobs if parallel_restarts else 1
        kwargs['worker_init'] = worker_init
        kwargs['worker_init_args'] = worker_init_args
    return SEARCH_FUNCS[algorithm](objective, specs, default_x=default_x, **kwargs)


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #

def tune_model_hyperparameters(model_name, base_params, train, valid, test,
                                run_model_fn, RESULT_NAME, algorithm,
                                budget_kwargs, hparam_specs,
                                reduced_cost_search=True, n_jobs=1,
                                bayes_batch=True, bayes_batch_max=8, bayes_liar='max',
                                parallel_restarts=True, worker_init=None, worker_init_args=(),
                                bayes_domain_reduction='auto', selection_margin=0.0,
                                valid_repeats=1, inner_split_seed_base=0):
    """Search `algorithm`'s best hyperparameters for `model_name` on the
    validation set, then perform ONE final confirmatory fit at the winning
    hyperparameters with the user's real explainability flags and MCMC
    length restored.

    Guarantees the returned hyperparameters score at least as well
    (within `selection_margin`) on the validation set as the user's own
    base_params (defaults/current values) would - the search is never
    allowed to hand back something worse than simply not tuning at all
    (see the margin-based default-vs-winner comparison below).

    Parameters
    ----------
    bayes_domain_reduction : {'auto', 'always', 'never'}, default 'auto'
        (ver4-6 R1.2(d)). Forwarded to run_search() -> search_bayesian()'s
        own `domain_reduction=` keyword; irrelevant to every other
        algorithm. See search_bayesian()'s own docstring.
    selection_margin : float, default 0.0 (ver4-6 R3.2b). The winning
        candidate must beat `base_params`'s own score by at least this
        much (in objective-score units, i.e. `r - mse/var`) or the
        defaults are kept instead. `0.0` reproduces the exact pre-ver4-6
        comparison (`default_score >= best_score`) byte-for-byte.
        Comparing one single noisy default draw against the MAXIMUM of
        many search draws is a textbook winner's-curse comparison (E[max
        of n draws] exceeds the mean by roughly sigma*sqrt(2*ln n) - about
        2.7 sigma at n=37 for a typical Bayesian search budget) - a
        positive margin makes "the search must beat the default" harder
        to satisfy on sampling noise alone. genomic_prediction.py's own
        resolved HP_TUNE_SELECTION_MARGIN config key defaults to 0.02 and
        is passed in here explicitly by the caller; this function's own
        default stays conservative (0.0) so a caller that does not pass
        it sees no behaviour change.
    valid_repeats, inner_split_seed_base : ver4-6 R3.2c - forwarded to
        make_objective() unchanged. See that function's own docstring.
    n_jobs : int, default 1 (ver4-4 R3.f; ver4-5 R1.b/R1.c extend this to
        every algorithm, not only Grid/Random - see run_search()'s own
        docstring)
        Forwarded to run_search(), which forwards it on to whichever
        algorithm-specific search function is actually running.
        `n_jobs<=1` reproduces this function's exact pre-R3.f behaviour
        (the serial loop, byte-for-byte, for every algorithm). `n_jobs>1`
        requests a joblib.Parallel PROCESS-backend fan-out (invariant
        I10) - see search_grid()/search_random()'s own `_evaluate_batch()`
        helper, and search_bayesian()/`_multistart_local_search()`'s own
        docstrings for their own fan-out shapes - for the guaranteed,
        silent, exception-triggered fallback to serial evaluation
        whenever `run_model_fn` (supplied by THIS function's own caller,
        not this module) closes over anything that cannot cross a
        process boundary. This module never assumes `run_model_fn` IS
        process-safe - only that requesting parallelism is always SAFE
        to try, never that it will necessarily succeed.
    bayes_batch, bayes_batch_max, bayes_liar : ver4-5 R1.b - forwarded to
        `run_search()` -> `search_bayesian()` unchanged; irrelevant to
        every other algorithm. See `search_bayesian()`'s own docstring.
    parallel_restarts : bool, default True (ver4-5 R1.c) - forwarded to
        `run_search()`; controls whether Nelder-Mead/Powell's own multi-
        restart local search fans out across processes. See
        `_multistart_local_search()`'s own docstring.
    worker_init, worker_init_args : ver4-5 R1.i - forwarded to every
        algorithm's own fan-out path (only actually used by whichever
        one decides to fan out at all). Restores this run's own resolved
        compute-resource state (and, where applicable, R/rpy2 and the
        GPU semaphore) inside a freshly spawned trial worker - see
        genomic_prediction.py::_trial_worker_init()'s own docstring for
        why this is mandatory, not optional, whenever `n_jobs>1` reaches
        a model that reads that state (i.e. every model - RK-11).

    Returns
    -------
    final_params : list          - full HPARAMETERS[model_name]-shaped list
    final_result : dict          - exactly what run_model_fn returns
    best_valid_score : float     - the winning candidate's objective score
                                    (see make_objective() for its exact
                                    definition - no longer a plain r/MSE
                                    ratio)
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
                                run_model_fn, RESULT_NAME, reduced_cost_search,
                                valid_repeats=valid_repeats,
                                inner_split_seed_base=inner_split_seed_base)

    # Computed BEFORE the search so it can be handed to run_search() below:
    # search_bayesian() (and, for the same reason, search_random()/the
    # Nelder-Mead & Powell local searches) explicitly evaluates this point
    # itself - see each function's own comment for why seeding the search
    # with a known-reasonable anchor makes actually beating it more likely,
    # not just possible.
    default_x = _encode_from_params(specs, base_params)
    _search_result = run_search(
        algorithm, objective, specs, budget_kwargs, default_x=default_x, n_jobs=n_jobs,
        worker_init=worker_init, worker_init_args=worker_init_args,
        bayes_batch=bayes_batch, bayes_batch_max=bayes_batch_max, bayes_liar=bayes_liar,
        parallel_restarts=parallel_restarts, bayes_domain_reduction=bayes_domain_reduction,
    )
    # ver4-6 R4.4: run_search() returns a 4-tuple as of this update -
    # unpacked DEFENSIVELY (`res[:3]` + `res[3] if len(res) > 3 else None`)
    # so a 3-tuple-returning caller/algorithm somewhere else in the tree
    # can never break this function; it just means "no default score was
    # supplied", handled identically to every pre-ver4-6 call.
    best_x, best_score, _trials = _search_result[:3]
    default_score = _search_result[3] if len(_search_result) > 3 else None

    # Bug fix (tuned models underperforming their own untuned defaults):
    # score the user's own default/current hyperparameters through the
    # EXACT SAME objective every search candidate was scored through (same
    # validation split, same scoring function, same explainability-
    # disabled/cheapened-MCMC evaluation conditions), and never hand back
    # a "winner" that didn't actually beat that baseline. A search over a
    # wide hyperparameter space, scored on one (often small, for genomic-
    # prediction sample sizes) validation split, can easily land on a
    # candidate that happens to fit that split's own sampling noise
    # better than the defaults do, without genuinely generalising better -
    # exactly the reported symptom. This makes "no worse than the
    # defaults" an explicit guarantee rather than something the search is
    # merely likely to achieve - on top of, not instead of, the seeding
    # above (an algorithm that doesn't accept default_x at all, or a
    # search whose surrogate model still wanders away from a good anchor
    # despite being seeded with it, is covered here regardless).
    #
    # Update ID ver4-6, R4.2c: reused from the search itself whenever one
    # was supplied (every algorithm except Grid - see the module-level
    # comment above search_grid()) instead of paying for a second, purely
    # redundant model fit at the exact same point; only re-evaluated fresh
    # here when the search did not supply one (Grid, or a defensively-
    # unpacked missing 4th element), which is byte-for-byte the pre-ver4-6
    # behaviour for that case.
    if default_score is None:
        default_score = objective(default_x)

    # Update ID ver4-6, R3.2b: MARGIN comparison, not a bare `>=`.
    # Comparing one noisy default draw against the MAXIMUM of many search
    # draws is a textbook winner's-curse comparison (see this function's
    # own `selection_margin` docstring above) - requiring the winner to
    # beat the default by at least `selection_margin` makes the guarantee
    # meaningfully harder to satisfy by sampling luck alone.
    # `selection_margin=0.0` (this function's own default) reproduces the
    # exact pre-ver4-6 `>=` comparison byte-for-byte.
    margin = max(0.0, selection_margin)
    if default_score >= best_score - margin:
        best_x, best_score = default_x, default_score

    final_params = list(base_params)
    tuned_values = {}
    for s, val in zip(specs, _decode(specs, best_x).values()):
        final_params[s.index] = val
        tuned_values[s.name] = val

    # Bug fix: the winning `best_x` recorded by the search algorithm is the
    # RAW sampled/selected point - for MCMC models (rrBLUP/GBLUP/BayesB/
    # RKHS) this can carry a burnIn >= nIter combination even though that
    # exact combination was NEVER actually scored in that state (search-time
    # evaluation always ran it through _enforce_mcmc_burnin_invariant/
    # _cheapen_mcmc first - see make_objective() above - so the score
    # attached to this x came from a corrected version of it, not this one).
    # Left uncorrected here, the confirmatory fit below could silently run
    # BGLR with burnIn >= nIter: zero post-burn-in posterior samples to
    # average, collapsing to a near-constant prediction whose Pearson r is
    # undefined (NaN) - reported as the tuning "winner" despite never having
    # been validated in this state. Re-applying the same invariant here
    # keeps the final fit consistent with what was actually scored.
    final_params = _enforce_mcmc_burnin_invariant(model_name, final_params)
    # Keep the reported "winning hyperparameters" (tuned_values - what
    # callers log/display, e.g. run_step1_batch.py's "tuned via Bayesian:
    # ... -> {...}" line) in sync with any correction just made, so what's
    # reported always matches what the confirmatory fit below actually used.
    for s in specs:
        tuned_values[s.name] = final_params[s.index]

    # Final confirmatory fit: full explainability + full MCMC length exactly
    # as the user configured them - identical in every respect to a normal,
    # untuned model call.
    #
    # ver4-4 R5 Fix 3 - guard the confirmatory fit. The search above always
    # runs candidates through _cheapen_mcmc() (reduced MCMC length) with
    # explainability disabled (_disable_explainability()), so a
    # hyperparameter combination that scored cleanly during the search can
    # still diverge once refit here at the user's real, full MCMC length /
    # full explainability - a BGLR chain that doesn't converge at the tuned
    # nIter/burnIn/h combination is the concrete case (R5 root cause).
    # Left unguarded, that exception propagates out of _process_one_task
    # to the task-loop's own except block, which rolls back the task and
    # RE-RAISES (architecture doc §14) - killing the entire batch over what
    # is, functionally, one bad tuning outcome rather than a configuration
    # error. Retry ONCE at the user's own base_params (the untuned
    # defaults): this function's own contract (see docstring above,
    # "Guarantees the returned hyperparameters score at least as well ...")
    # already establishes that base_params can never score worse than
    # anything the search could have picked, so falling back to it here can
    # never be a worse OUTCOME than simply not tuning this task at all - it
    # only forgoes the (already-failed) tuning attempt. If base_params ALSO
    # fails to fit, the model cannot fit at its own user-configured defaults
    # at all - a genuine configuration error, not a tuning artefact - so it
    # is re-raised exactly as an untuned call to this model would be today.
    try:
        final_result = run_model_fn(model_name, train, valid, test, final_params, RESULT_NAME)
    except Exception as exc:
        # Req 2 fix (2026-09): full traceback added - this is the exact
        # print whose ABSENCE from the production log was the central,
        # unresolved puzzle in the original investigation
        # (REQ2_Handoff.md §4): a confirmatory-fit failure here MUST
        # print this line before anything else can happen, so if it
        # never appears, the failure genuinely did not originate here -
        # but that was never fully verifiable from repr(exc) alone.
        # Keeping the traceback attached going forward removes that
        # ambiguity for any future occurrence, here and at the retry
        # below.
        print(f"[hyperparameter_tuning] NOTE: confirmatory fit for {model_name} "
              f"failed at tuned hyperparameters {tuned_values} "
              f"({type(exc).__name__}: {exc}) - retrying once at the untuned "
              f"defaults (base_params) before giving up on this task.\n"
              f"{traceback.format_exc()}")
        try:
            final_params = list(base_params)
            final_result = run_model_fn(model_name, train, valid, test, final_params, RESULT_NAME)
        except Exception:
            print(f"[hyperparameter_tuning] {model_name} also failed to fit at its "
                  f"own configured (untuned) defaults for this task - this is a "
                  f"genuine configuration error, not a tuning artefact; re-raising.\n"
                  f"{traceback.format_exc()}")
            raise
        tuned_values = {s.name: final_params[s.index] for s in specs}
        print(f"[hyperparameter_tuning] NOTE: {model_name} tuning fell back to its "
              f"untuned defaults for this task after the confirmatory-fit failure "
              f"above - 'valid_pearson_over_mse' below still reports the search's "
              f"own best score, but the fit actually used was untuned.")
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
