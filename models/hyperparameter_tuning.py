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
- ver4-85: with SUPER_LEARNER['oof_tuning'] the score is instead an OUT-OF-FOLD score over
  the TRAINING split (`oof_folds`, see make_objective()); the task then needs no validation
  split. Measured in ice_sim26 (tuple ratio: +0.003 r for 1.3-4.3x the fits, not adopted as
  the default) and ice_sim27 (float ratio, reshuffled Super-learner folds).
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
import os
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

# ver4-36, R1.e: a REUSABLE loky executor, for callers that submit work item by item
# (the joint search's asynchronous library search) instead of handing a whole batch
# to `joblib.Parallel`. `get_reusable_executor` keeps the same worker processes
# alive between calls while the arguments are unchanged, so the per-worker start-up
# (importing the modules, starting R through rpy2) is paid once per run rather than
# once per batch - which matters most on Windows, where loky spawns its workers.
_LOKY_EXECUTOR_IDLE_SECONDS = 300  # only used by the fallback path (joblib's own idle limit is also 300 s)
_HELD_PARALLEL = [None]  # the open `joblib.Parallel` whose executor `_loky_executor()` last returned


def _loky_executor(n_jobs, worker_init=None, worker_init_args=()):
    """Return a reusable ``loky`` executor with ``n_jobs`` workers, running
    ``worker_init(*worker_init_args)`` once in every freshly started worker (the same
    initializer contract as ``_loky_parallel``), or ``None`` when ``n_jobs <= 1``
    (the caller then runs the work inline - a serial run with identical results).

    The executor is ``concurrent.futures``-compatible: ``executor.submit(fn, *args)``
    returns a ``Future``. Because it is reusable, two calls with the same arguments
    return the SAME executor (workers are not restarted); workers that stay idle for
    ``_LOKY_EXECUTOR_IDLE_SECONDS`` are shut down by loky and started again on demand
    (the idle limit is long on purpose: a loky default of 10 s would restart them
    during every gate or confirmatory fit).

    Do not call ``shutdown`` on it - other callers may be sharing the pool.

    Args: ``n_jobs`` worker count (None / <= 1 returns None); ``worker_init`` optional
    picklable callable; ``worker_init_args`` its plain-data arguments.
    Raises ``ImportError`` if joblib/loky is missing (a project dependency)."""
    if n_jobs is None or int(n_jobs) <= 1:
        return None
    # loky keeps ONE process-wide reusable executor, shared with every
    # `joblib.Parallel(backend='loky')` call (`_evaluate_batch`, the gate / confirm stages,
    # every pre-ver4-36 search). Two things follow, both found by measurement:
    #  1. a plain `loky.get_reusable_executor` replaces it with an executor that lacks
    #     joblib's `_temp_folder_manager`, and every later `joblib.Parallel` call in the same
    #     process then fails ("'_ReusablePoolExecutor' object has no attribute
    #     '_temp_folder_manager'", joblib 1.5.3) and falls back to serial;
    #  2. joblib restarts the pool whenever the arguments differ from the previous call
    #     (it also passes per-worker BLAS/OpenMP thread limits, memmapping settings and the
    #     start method) - so an executor built with different arguments makes the run
    #     alternate between "pool for the async search" and "pool for the gate fits",
    #     paying the worker start-up (importing the models, starting R) at every switch,
    #     and its workers would run without the thread limits (BLAS oversubscription).
    # Hence the executor is obtained from a `joblib.Parallel` of the SAME configuration as
    # `_loky_parallel` builds: same arguments -> same pool, same thread limits, no restarts.
    # The Parallel context is deliberately kept OPEN (module-level holder) for as long as
    # the executor may be in use: closing it unregisters its temporary-folder context and
    # a later submit of a large array (> 1 MB, i.e. any real genotype frame) would then
    # fail to pickle. The next call replaces (and closes) the previous holder.
    par = None
    try:
        par = _loky_parallel(int(n_jobs), worker_init, worker_init_args)
        par.__enter__()
        executor = par._backend._workers
        if executor is None or not hasattr(executor, 'submit'):
            raise AttributeError('joblib loky backend exposes no executor')
    except Exception:  # pragma: no cover - joblib internals differ: documented fallback
        if par is not None:
            try:
                par.__exit__(None, None, None)
            except Exception:
                pass
        kwargs = {'timeout': _LOKY_EXECUTOR_IDLE_SECONDS}
        if worker_init is not None:
            kwargs['initializer'] = worker_init
            kwargs['initargs'] = tuple(worker_init_args)
        from joblib.executor import get_memmapping_executor
        return get_memmapping_executor(int(n_jobs), **kwargs)
    previous, _HELD_PARALLEL[0] = _HELD_PARALLEL[0], par
    if previous is not None:
        try:
            previous.__exit__(None, None, None)
        except Exception:
            pass
    return executor


# Startup-time optimisation: bayes_opt (-> scipy.optimize + scikit-learn's Gaussian-process stack) is
# imported lazily, on the first Bayesian search, instead of when this module is imported (the GUI
# imports this module at launch only for ALGORITHMS / spec helpers). The three names keep their
# old module-level meaning - the class, or None when bayes_opt is missing - once
# `_ensure_bayes_opt()` has run; it is called at the top of the only function that uses them.
BayesianOptimization = None
SequentialDomainReductionTransformer = None
ExpectedImprovement = None
_BAYES_OPT_LOADED = False


def _ensure_bayes_opt():
    """Import bayes_opt once (idempotent); leaves the three names None if it is not installed."""
    global BayesianOptimization, SequentialDomainReductionTransformer, ExpectedImprovement
    global _BAYES_OPT_LOADED
    if _BAYES_OPT_LOADED:
        return
    try:
        from bayes_opt import BayesianOptimization as _BO, SequentialDomainReductionTransformer as _SDR
        from bayes_opt.acquisition import ExpectedImprovement as _EI
    except ImportError:  # pragma: no cover - already a project dependency
        _BAYES_OPT_LOADED = True
        return
    BayesianOptimization, SequentialDomainReductionTransformer, ExpectedImprovement = _BO, _SDR, _EI
    _BAYES_OPT_LOADED = True


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


def _apply_baseline_floor(model_name, field_name, tunable, idx, base_params):
    """Bug fix (Bayesian-tuned rrBLUP/GBLUP/BayesB/RKHS/RF/ExtraTrees
    underperforming their own untuned defaults - see Requirement.md /
    the step1_config.json this was diagnosed against): returns the
    EFFECTIVE lower bound to search for one numeric field, raising the
    schema's own `tunable['low']` up to this run's own configured
    baseline value whenever the field's 'tunable' dict sets
    `'floor_at_baseline': True`. A field with no such key (the default,
    unchanged for every existing tunable field) is returned exactly as
    hparam_specs.py declares it - byte-for-byte the pre-fix behaviour.

    ROOT CAUSE. `MCMC_LENGTH_FIELDS` ('Iteration number' / 'Burn-in' for
    rrBLUP/GBLUP/BayesB/RKHS) and RF/ExtraTrees's 'Tree number' are not
    ordinary model-complexity hyperparameters with a genuine bias/
    variance trade-off (like SVR's C, or a tree's max depth) - they are
    FIDELITY/CONVERGENCE knobs. Running BGLR's MCMC chain longer (given
    adequate burn-in), or averaging more trees in a Random Forest, is
    monotonically at least as good for genuine generalisation - it never
    makes the underlying estimate WORSE, only slower to compute. There is
    therefore no real "sweet spot" to discover by searching BELOW the
    user's own configured value the way there is for a true regularisation
    hyperparameter. Before this fix, hparam_specs.py declared
    'Iteration number' tunable down to 2000 (default 12000) and 'Burn-in'
    down to 200 (default 2000) - a Bayesian search could, and empirically
    did, land on a severely under-converged short MCMC chain that scored
    well on one (often small, for genomic-prediction sample sizes)
    validation split purely from MCMC/sampling NOISE rather than genuine
    predictive quality. Because 'Iteration number'/'Burn-in' are
    themselves tunable dimensions, `_cheapen_mcmc()` never re-cheapens or
    corrects them (that function only touches a field NOT already owned
    by the search - see its own docstring), so this under-converged value
    is carried unchanged into the final confirmatory fit and reported as
    the tuned result: a chain that is worse-converged, and so worse-
    generalising, than the well-converged default it was meant to beat.
    This is a genuine instance of "overfitting to the validation set" -
    not via extra model capacity, but via exploiting MCMC sampling noise
    the validation objective cannot distinguish from a real signal.

    THE FIX. For a field flagged `'floor_at_baseline': True`, the search
    is only ever allowed to make that field LARGER than (or equal to)
    whatever this run's own HPARAMETERS baseline already configures for
    it - it may still search upward (e.g. trying an even longer chain, in
    case this dataset needs it), but can never sample a value that would
    make the fit's own fidelity WORSE than what the user already asked
    for. This mirrors RF/ExtraTrees's own 'Tree number' field, which
    already (independently) happened to set 'low' equal to its schema
    default - now made an explicit, self-documenting, and baseline-aware
    invariant instead of a value that only worked so long as a user's own
    configured HPARAMETERS happened to match hparam_specs.py's schema
    default exactly.

    Falls back to the schema's own static 'low' whenever `base_params` is
    not supplied (`None` - e.g. main_app.py's "does this model have any
    tunable fields at all" check, which never fits anything) or the
    baseline value can't be read/parsed - never raises, since a schema
    validity issue is exactly what the log-and-continue diagnostics
    discipline elsewhere in this codebase (architecture doc S18) prefers
    over aborting a run.
    """
    low = tunable['low']
    if not tunable.get('floor_at_baseline') or base_params is None:
        return low
    try:
        baseline_val = float(base_params[idx])
    except (TypeError, ValueError, IndexError):
        #print(f"[hyperparameter_tuning] NOTE: {model_name!r} field {field_name!r} "
        #      f"declares tunable['floor_at_baseline']=True but its baseline value "
         #     f"(HPARAMETERS[{model_name!r}][{idx}]) could not be read as a number - "
          #    f"falling back to the schema's own static low bound ({low}) for this "
          #    f"search.")
        return low
    high = tunable['high']
    # Never below the schema's own declared low (a baseline somehow BELOW
    # the schema's own sane floor is still bounded by it), never above the
    # schema's own declared high (an out-of-schema-range baseline is
    # clamped to 'high' rather than widening the box beyond what
    # hparam_specs.py declares safe/supported for this field).
    floored = min(max(low, baseline_val), high)
    #if floored > low:
        #print(f"[hyperparameter_tuning] NOTE: {model_name!r} field {field_name!r} "
        #      f"search lower bound raised from {low} to {floored} to match this "
        #      f"run's own configured baseline (tunable['floor_at_baseline']=True) - "
        #      f"the search may still try LARGER values, but will never try anything "
        #     f"more under-converged/lower-fidelity than what was already "
         #     f"configured.")
    return floored


def build_param_specs(model_name, hparam_specs, base_params=None):
    """Read hparam_specs.HPARAM_SPECS[model_name] (main_app.py's own
    schema) and return the ordered list of ParamSpec for every field
    carrying a 'tunable' entry. Returns [] for a model with nothing marked
    tunable - always safe to call, never raises for an unknown/unannotated
    model. `model_name` is passed through schema_key_of() first, so a
    numbered GAT_biological_prior_knowledge instance resolves to the shared
    base schema automatically.

    base_params : optional, this run's own configured HPARAMETERS[model]
        list (i.e. `tune_model_hyperparameters()`'s own `base_params`).
        Only consulted for a field whose 'tunable' dict sets
        `'floor_at_baseline': True` (see MCMC_LENGTH_FIELDS's own fields
        in hparam_specs.py) - see `_apply_baseline_floor()`'s docstring
        for why such a field's search-time lower bound must never sit
        below the value the user actually configured. Every other field
        is completely unaffected by this argument - passing `None` (every
        pre-existing caller, e.g. main_app.py's own "does this model have
        anything tunable at all" check) reproduces this function's exact
        prior behaviour."""
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
            low = _apply_baseline_floor(model_name, name, tunable, idx, base_params)
            specs.append(ParamSpec(
                idx, name, ftype,
                low=low, high=tunable['high'], step=tunable.get('step'),
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


def _is_fidelity_field(model: str, idx: int, hparam_specs: dict) -> bool:
    """ver4-32, R1: True when field `idx` of `model` is a run-length knob -
    i.e. `HPARAM_SPECS[schema_key_of(model)][idx]['tunable']` carries
    `floor_at_baseline: True` (MCMC iterations/burn-in, tree counts).

    Why this exists: those fields have no real sweet spot (more is simply
    slower and rarely worse), yet while they were part of the joint search
    `_cheapen_mcmc()`/`_cheapen_fidelity()` skipped them (`tuned_indices`),
    so every search fit ran at full length. `floor_at_baseline` stays the
    single source for this decision (invariant I3) - no second list of
    "run-length fields" is kept here.

    Args: `model` (a MODEL-list name, numbered GAT instances allowed),
    `idx` (positional index into the model's HPARAM_SPECS field list),
    `hparam_specs` (the HPARAM_SPECS dict). Returns False for an unknown
    model or an out-of-range index; never raises."""
    fields = hparam_specs.get(schema_key_of(model), [])
    if idx < 0 or idx >= len(fields):
        return False
    tunable = fields[idx].get('tunable') or {}
    # ver4-46, R2: `run_length: True` is a second, metadata-only way to mark a run-length
    # knob (GAT 'Epoch'): unlike `floor_at_baseline` it does NOT raise the field's lower
    # bound in the legacy search (so legacy tuning is unchanged), it only lets
    # `filter_param_specs(freeze_fidelity=True)` take the field out of the search.
    return bool(tunable.get('floor_at_baseline') or tunable.get('run_length'))


def _is_priority_field(model: str, idx: int, hparam_specs: dict) -> bool:
    """ver4-34, R2.c: True when field `idx` of `model` is marked as influential -
    `HPARAM_SPECS[schema_key_of(model)][idx]['tunable']` carries `priority: True`.

    Why this exists: searching 25 dimensions on ~50-150 rows makes the arg-max
    fit noise (winner's curse). `search_space='priority'` keeps only the fields a
    domain expert marked as influential. The mark is plain metadata INSIDE the
    existing `tunable` dict (invariant I5: no positional change, and
    `hyperparameter_tuning.build_param_specs()` ignores unknown keys - P4), and
    this is the single place that reads it (I3).

    Args: `model` (a MODEL-list name, numbered GAT instances allowed), `idx`
    (positional index into the model's HPARAM_SPECS field list), `hparam_specs`
    (the HPARAM_SPECS dict). Returns False for an unknown model or an
    out-of-range index; never raises."""
    fields = hparam_specs.get(schema_key_of(model), [])
    if idx < 0 or idx >= len(fields):
        return False
    tunable = fields[idx].get('tunable') or {}
    return bool(tunable.get('priority'))


def filter_param_specs(model, specs, hparam_specs, *, freeze_fidelity=False,
                       priority_only=False):
    """ver4-40, R1 (C1/C6): drop run-length and/or non-influential dimensions from a
    model's own `ParamSpec` list. This is the SINGLE authority for both filters (I3):
    the joint search (`joint_tuning.build_joint_param_specs`) and single-model tuning
    (`tune_model_hyperparameters(fast=...)`) both call it, so the two paths can never
    disagree on what "frozen" or "priority" means.

    `freeze_fidelity=True` removes every spec whose field is a run-length knob
    (`_is_fidelity_field`). Why: while such a field is in the search space,
    `_cheapen_mcmc()` / `_cheapen_fidelity()` skip it (they skip `tuned_indices`), so every
    search fit runs at >= the configured chain length instead of the cheap search length.
    `priority_only=True` keeps only fields marked `priority`; a model with NO marked field
    keeps everything (an unannotated model is never silently emptied). The marks are looked
    up on the FULL tunable list, before `freeze_fidelity` removes anything, so the two
    filters compose in either order. Returns a new list (input untouched); both flags False
    returns an equal list. Never raises for an unknown model."""
    specs = list(specs)
    has_marks = any(_is_priority_field(model, sp.index, hparam_specs) for sp in specs)
    if freeze_fidelity:
        specs = [sp for sp in specs if not _is_fidelity_field(model, sp.index, hparam_specs)]
    if priority_only:
        specs = [sp for sp in specs
                 if (not has_marks) or _is_priority_field(model, sp.index, hparam_specs)]
    return specs


def cap_search_cost(model, specs, hparam_specs, base_params):
    """ver4-46, R2: lower the UPPER bound of cost-driving dimensions for the SEARCH only.

    A field marks itself with ``'search_high': <value>`` inside its own ``tunable`` dict
    (plain metadata, ignored by ``build_param_specs`` - invariant I5). Why: the cost of one
    GAT fit grows with neurons x heads x epochs, so the full box spans a ~250x cost range and
    a few random corner candidates dominate the whole search. The bound never goes below the
    value the user configured (`base_params`), so the default point stays reachable and the
    default-vs-winner safeguard still compares the user's real configuration. Returns a new
    list; fields without the mark (and every field of a model without marks) are untouched.
    Args: `model` (MODEL-list name), `specs` (list of ParamSpec), `hparam_specs`
    (HPARAM_SPECS), `base_params` (the configured HPARAMETERS list). Never raises."""
    fields = hparam_specs.get(schema_key_of(model), [])
    out = []
    for sp in specs:
        high = sp.high
        try:
            cap = (fields[sp.index].get('tunable') or {}).get('search_high') \
                if 0 <= sp.index < len(fields) else None
            if cap is not None and sp.kind in ('int', 'float'):
                keep = base_params[sp.index] if (base_params is not None
                                                 and sp.index < len(base_params)) else sp.low
                new_high = max(float(cap), float(keep), float(sp.low))
                if new_high < float(sp.high):
                    high = int(new_high) if sp.kind == 'int' else float(new_high)
        except Exception:
            high = sp.high
        if high != sp.high:
            sp = ParamSpec(sp.index, sp.name, sp.kind, low=sp.low, high=high, choices=sp.choices,
                           step=sp.step, scale=sp.scale)
        out.append(sp)
    return out


def search_box_highs(model, base_params, hparam_specs, fast=None):
    """ver4-61, R2: the upper bound of every numeric searched dimension, as
    ``{field_index: high}``, after the SAME narrowing the search applies.

    Why: the trial-memory guard needs the largest batch size and heads x neurons a search
    can draw (the corner of the box), and must see the box the search will really use -
    including ``HP_TUNE_FAST`` ``freeze_fidelity`` / ``search_space`` / ``cap_search_cost``.
    This reuses `build_param_specs`, `filter_param_specs` and `cap_search_cost`, so the two
    can never disagree (I3). Never lowers a bound below `base_params` (the default point
    must stay inside the box, and the confirmatory fit runs at the winner).

    Args: `model` MODEL-list name; `base_params` the configured positional list;
    `hparam_specs` HPARAM_SPECS; `fast` the resolved HP_TUNE_FAST dict or None.
    Returns a dict (empty when nothing is searchable). Raises ValueError only when
    `build_param_specs` does (no tunable field declared)."""
    specs = build_param_specs(model, hparam_specs, base_params=base_params)
    if fast:
        specs = filter_param_specs(
            model, specs, hparam_specs,
            freeze_fidelity=bool(fast.get('freeze_fidelity', False)),
            priority_only=(fast.get('search_space', 'all') == 'priority'))
        if fast.get('cap_search_cost'):
            specs = cap_search_cost(model, specs, hparam_specs, base_params)
    highs = {}
    for sp in specs:
        if sp.kind not in ('int', 'float'):
            continue
        high = float(sp.high)
        if base_params is not None and sp.index < len(base_params):
            try:
                high = max(high, float(base_params[sp.index]))
            except (TypeError, ValueError):
                pass
        highs[sp.index] = high
    return highs


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
    # Requirements.md item 3 (this update): RF gained its own, separate
    # marker-EFFECT toggle (field 18, 'Return marker effect?') alongside
    # its pre-existing marker-PAIR-interaction toggle (field 5) - both are
    # now forced off during every search trial (restored for the final
    # confirmatory fit), matching SVR/KNN's own two-flag entries below
    # rather than leaving the new 'shapiq' effect route (potentially
    # expensive, like every other shapiq-based route in this dict) free
    # to run on each candidate.
    # Requirements.md item 3 (this update): ExtraTrees gained its own,
    # separate marker-EFFECT toggle (field 11, 'Return marker effect?')
    # alongside its pre-existing marker-PAIR-interaction toggle (field 5)
    # - see RF's identical (5, 18) entry's own comment above.
    # Requirements.md item 3 (this update): XGBoost gained its own,
    # separate marker-EFFECT toggle (field 12, 'Return marker effect?')
    # alongside its pre-existing marker-PAIR-interaction toggle (field 6)
    # - see RF's identical (5, 18) entry's own comment above. GBDT is
    # untouched by this update (not in Requirements.md item 3's scope -
    # see this codebase's own note on GBDT having no comparable toggle
    # structure to reorder).
    'RF': (5, 18), 'ExtraTrees': (5, 11), 'GBDT': 6, 'XGBoost': (6, 12),
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


# FEATURE (multi-fidelity screening for standard, per-model Bayesian
# hyperparameter tuning - the non-joint counterpart of models/joint_
# tuning.py's own Round 7 multi-fidelity screening, same underlying
# rationale: each candidate here costs one full model fit, scored on an
# often-small, noisy validation split, against a search budget that only
# ever runs a few dozen candidates - exactly the regime where a cheap,
# low-fidelity pre-screen of a batch round's candidates (evaluate all of
# them fit on a SMALL random subsample of this task's own training rows,
# then only re-fit the more promising half at the FULL training row count)
# spends the same iteration budget more usefully.
#
# Deliberately mirrors this module's OWN established fidelity-reduction
# idiom, not joint_tuning.py's: _cheapen_mcmc()/_cheapen_fidelity() above
# both apply a SINGLE fixed cheapening for an entire search, never
# re-randomized round to round, with one full-fidelity confirmatory fit at
# the very end - so this feature's own reduced-row-count training
# subsample is likewise drawn ONCE per (task, model) search (see
# make_objective()'s own use of this below), not re-drawn every batch
# round the way joint_tuning.py's Round 7 does it. Never touches `valid`/
# `test` (invariant I7) - only `train` is ever subsampled, and every
# screening evaluation is still scored against the task's REAL validation
# split, so a screening score stays directly comparable in scale to a
# full-fidelity one (just noisier), never on some other footing entirely.
_MULTI_FIDELITY_TRAIN_FRACTION = 0.35
_MULTI_FIDELITY_MIN_ROWS = 30
_MULTI_FIDELITY_KEEP_FRACTION = 0.5


def _screen_subsample_train(train, fraction, min_rows, seed):
    """A random ROW subsample of `train` ONLY - never touches `valid`/
    `test` (invariant I7), same guarantee `_split_train_rows()` above
    already gives the repeated-resample path. Returns `None` when `train`
    is already too small to subsample meaningfully (the requested
    subsample would not actually be smaller than `train` itself) - the
    caller then skips multi-fidelity screening entirely for this task,
    exactly like `_split_train_rows()` falls back to `valid_repeats=1` for
    the same reason.
    """
    n = len(train)
    n_screen = max(int(min_rows), int(round(n * fraction)))
    if n_screen >= n:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.choice(train.index.to_numpy(), size=n_screen, replace=False)
    return train.loc[idx]


_ENGINE_NOTED = set()   # (model, exception type) pairs already reported, once per process


def _engine_metrics(model_name, fit_train, fit_valid, test, params):
    """ver4-40 R1 (C3): the exact-kernel stand-in for one rrBLUP / RKHS search fit.

    Returns ``{'pearson_valid': r, 'mse_valid': mse}`` computed from
    `kernel_effects.exact_kernel_column()` (closed form, same rows and priors as the
    BGLR fit, no MCMC) - or ``None`` when the engine raises, so the caller falls back to
    the real runner (one NOTE per process and exception type; an engine failure costs
    time, never a result). Unweighted metrics are correct because the engine is switched
    off whenever BLUE weighting is on (`shortcut_eligible=False`). I7: only the training
    phenotype is read by the engine; `fit_valid`'s phenotype is used here only to SCORE."""
    try:
        from .kernel_effects import exact_kernel_column
        from pipeline_utils import safe_regression_metrics
        col = exact_kernel_column(model_name, fit_train, fit_valid, test, params)
        pred = np.asarray(col['predicted_valid'], dtype=float)
        actual = np.asarray(fit_valid.iloc[:, -1], dtype=float)
        r, mse = safe_regression_metrics(actual, pred)
        return {'pearson_valid': r, 'mse_valid': mse}
    except Exception as exc:
        tag = (model_name, type(exc).__name__)
        if tag not in _ENGINE_NOTED:
            _ENGINE_NOTED.add(tag)
            print(f"[hyperparameter_tuning] NOTE: the exact kernel engine failed for "
                  f"{model_name!r} ({type(exc).__name__}: {exc}) - its search fits fall "
                  f"back to the real model fit (slower, same result; shown once per "
                  f"process).")
        return None


def make_objective(model_name, base_params, specs, train, valid, test,
                    run_model_fn, RESULT_NAME, reduced_cost_search=True,
                    valid_repeats=1, inner_split_seed_base=0,
                    multi_fidelity_screen=False, shortcuts=None, search_fit=None,
                    oof_folds=None, oof_options=None):
    """Returns f(x) -> score-to-maximise (float) for a continuous-encoded
    candidate x.

    ver4-85: `oof_folds` (default None = every earlier call, unchanged) switches the score to
    OUT-OF-FOLD: a list of >= 2 disjoint arrays of row POSITIONS of `train` that together cover
    it. A candidate is then fitted once per fold on the other folds' rows (`train.iloc[...]`,
    labels kept, so BLUE fitting weights align) and predicts the held-out rows; the pooled
    held-out predictions of all training rows are scored with the same `r - mse/var` formula,
    against the training phenotype. `valid` is not read (it may be empty), so `valid_repeats`,
    the multi-fidelity screen and the search shortcuts are not used in this mode (the caller,
    `tune_model_hyperparameters()`, switches them off and says so). The score is unweighted
    under BLUE weighting (the fits are weighted), like the Super learner's own weights. A fold
    whose fit fails or returns a missing, wrong-length or non-finite column fails the candidate.

    ver4-86: `oof_options` (out-of-fold mode only; None = the ver4-85 scorer) holds the cost
    savings decided by `tune_model_hyperparameters()`: ``test_stub`` (fold fits get a 1-row test
    frame), ``engine`` (rrBLUP / RKHS fold columns from `kernel_effects.exact_kernel_column`, a
    failing fold falling back to the real fit) and ``race`` (``(margin, min_folds)``: once a
    racing bar is set by `objective.prime_oof_race(x)`, a candidate whose pooled score over the
    folds done so far - at least `min_folds`, fewer than all - is below bar - margin stops there
    and that partial score is returned). Scores are memoised per decoded candidate (search points
    that snap to the same settings are fitted once), so the primed default point is never fitted
    twice (the memo travels with the objective to the workers).

    ver4-40 R1: `shortcuts` (default None = every pre-ver4-40 call, unchanged) is the
    per-(task, model) result of `_resolve_search_shortcuts()`:
    ``{'test_stub': bool, 'engine': 'exact_kernel' | None}``. With `test_stub` the
    runner receives ``test.iloc[:1]`` instead of the full test frame (the search never
    reads test predictions - BGLR masks and predicts every valid+test row, so unread test
    rows are pure cost; C2). With `engine='exact_kernel'` the candidate's validation
    column comes from the closed-form `kernel_effects.exact_kernel_column()` (rrBLUP /
    RKHS only; C3) and its r / MSE from `pipeline_utils.safe_regression_metrics()`; any
    exception there falls back to the runner with ONE NOTE per process. Both shortcuts
    were checked for parity at the default point by `_resolve_search_shortcuts()` before
    being switched on, and both only ever apply to SEARCH fits - the confirmatory fit in
    `tune_model_hyperparameters()` always gets the full `test` (I7). The returned
    objective also carries `.prepare_params` (the search-time parameter preparation) so
    the parity check scores exactly the parameters the search would.

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
    multi_fidelity_screen : bool, default False. When True, additionally
        builds a cheap, reduced-training-row-count screening evaluator and
        attaches it as `objective.screen` (never present at all when this
        is False, or when this task's own `train` is too small to
        subsample meaningfully - see `_screen_subsample_train()`'s own
        docstring) - `search_bayesian()`'s own BATCH search path uses it,
        when present, to pre-screen a round's candidates before spending
        the full evaluation budget only on the more promising half. Every
        other search algorithm, and the serial (non-batch) Bayesian path,
        never look at this attribute at all - see that function's own
        docstring for the full mechanism.
    """
    tuned_indices = {s.index for s in specs}
    # ver4-46, R1: `search_fit` (None = every pre-ver4-46 call) is the dict handed to
    # `pipeline_utils.search_fit_context()` around every candidate fit, e.g. {'lean': True}:
    # a model that knows how (the GAT family) then skips the outputs the search never reads.
    _search_fit = dict(search_fit) if search_fit else None
    _use_stub = bool(shortcuts and shortcuts.get('test_stub'))
    _engine = (shortcuts or {}).get('engine')
    # Computed once (the validation split doesn't change across
    # candidates) - see the scoring fix below for why this is needed.
    valid_var = _actual_variance(valid) if len(valid) else None   # ver4-86: empty in OOF runs

    # ver4-85: out-of-fold scoring - validated once here (a contract violation raises).
    _oof_folds = None
    _oof_y = None
    _oof_var = None
    if oof_folds is not None:
        _oof_folds = [np.asarray(f, dtype=int).ravel() for f in oof_folds]
        _all = np.concatenate(_oof_folds) if _oof_folds else np.empty(0, dtype=int)
        if (len(_oof_folds) < 2 or any(f.size == 0 for f in _oof_folds)
                or _all.size != len(train) or np.unique(_all).size != _all.size
                or _all.min() < 0 or _all.max() >= len(train)):
            raise ValueError(
                f"oof_folds for {model_name!r} must be >= 2 non-empty, disjoint arrays of row "
                f"positions covering all {len(train)} training rows")
        _oof_y = np.asarray(train.iloc[:, -1], dtype=float)
        _oof_var = _actual_variance(train)
        valid_repeats = 1
        multi_fidelity_screen = False
    # ver4-86: out-of-fold cost savings (None / {} = the ver4-85 scorer).
    _oof_opts = dict(oof_options or {}) if _oof_folds is not None else {}
    _oof_stub = bool(_oof_opts.get('test_stub', False))
    _oof_engine = bool(_oof_opts.get('engine', False))
    _oof_race = _oof_opts.get('race')

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

    # FEATURE (multi-fidelity screening): drawn ONCE for this task's whole
    # search, not re-drawn per batch round - see the module-level comment
    # above _screen_subsample_train() for why. `None` whenever the caller
    # didn't ask for this (multi_fidelity_screen=False, every existing
    # caller before this feature) or this task's own training split is too
    # small to subsample meaningfully - either way `objective.screen` is
    # simply never attached below (see `objective` at the end of this
    # function), and search_bayesian() falls back to full-fidelity
    # evaluation for every candidate, exactly as if this feature did not
    # exist. Offset from `inner_split_seed_base` (rather than reusing it
    # directly) so this draw never coincides with any of the
    # `inner_split_seed_base + b` inner-resample draws `_split_train_rows`
    # is called with above, while still being deterministic and
    # sharding-invariant (invariant I9) like every other seeded operation
    # in this module.
    _screen_train = (
        _screen_subsample_train(train, _MULTI_FIDELITY_TRAIN_FRACTION, _MULTI_FIDELITY_MIN_ROWS,
                                 seed=inner_split_seed_base + 10_000)
        if multi_fidelity_screen else None
    )

    # Update ID ver4-6, R2.2: closure-local tracking of the worst FINITE
    # score seen so far THIS SEARCH - a failure is scored relative to this
    # (see _failure_score() below), not as a fixed extreme constant, so it
    # never distorts the Gaussian-process surrogate's target
    # standardisation the way the old `-SCORE_CAP` sentinel did (E2).
    # Deliberately closure-local, never shared across processes - a batch-
    # mode worker (models/parallel_search.py) falls back to FAILURE_FLOOR
    # for its own first failure, which is already well-scaled on its own
    # (RK-10, disclosed in this module's own R2 change).
    state = {'worst_finite': None, 'memo': {}, 'race_bar': None, 'fold_scores': {}}

    def _failure_score():
        w = state['worst_finite']
        if w is None:
            return FAILURE_FLOOR
        return w - max(FAILURE_MARGIN, 0.10 * abs(w))

    def _run_fit(fit_train, fit_valid, params, stub=None):
        """One real model fit through `run_model_fn` (the test stub and the search-fit context
        applied as configured). Raises whatever the fit raises. ver4-85: shared by
        `_score_once()` and the out-of-fold scorer `_score_oof()`; ver4-86: `stub` overrides the
        shortcut's test-stub choice (out-of-fold mode)."""
        _test_arg = test.iloc[:1] if (_use_stub if stub is None else stub) else test
        if _search_fit:
            from pipeline_utils import search_fit_context
            _ctx = dict(_search_fit)
            # ver4-61 R3: the per-fit cap arrives as a DURATION and becomes an absolute
            # monotonic deadline here, inside the process that runs the fit (a loky
            # worker has its own clock origin, so a deadline made in the parent would
            # be meaningless there). Each fit gets its own window.
            _limit = _ctx.pop('trial_time_limit_s', None)
            if _limit:
                _ctx['trial_deadline_s'] = time.monotonic() + float(_limit)
            with search_fit_context(**_ctx):
                return run_model_fn(model_name, fit_train, fit_valid, _test_arg, params,
                                    RESULT_NAME)
        return run_model_fn(model_name, fit_train, fit_valid, _test_arg, params, RESULT_NAME)

    def _oof_fold_column(fit_train, fit_valid, params):
        """ver4-86: one fold's held-out column - the exact kernel engine when on (falling back
        to the real fit on any engine error, one NOTE per process), else the real fit."""
        if _oof_engine:
            try:
                from .kernel_effects import exact_kernel_column
                _t = test.iloc[:1] if _oof_stub else test
                return {'predicted_valid': exact_kernel_column(
                    schema_key_of(model_name), fit_train, fit_valid, _t, params)['predicted_valid']}
            except Exception as exc:
                tag = (model_name, type(exc).__name__, 'oof')
                if tag not in _ENGINE_NOTED:
                    _ENGINE_NOTED.add(tag)
                    print(f"[hyperparameter_tuning] NOTE: the exact kernel engine failed for an "
                          f"out-of-fold fit of {model_name!r} ({type(exc).__name__}: {exc}) - "
                          f"that fold uses the real model fit (shown once per process).")
        return _run_fit(fit_train, fit_valid, params, stub=_oof_stub)

    def _score_oof(params):
        """ver4-85: the out-of-fold score of one candidate (see `oof_folds` in the docstring).
        Returns (score, succeeded) with the same contract as `_score_once()`. ver4-86: fold
        racing (see `oof_options`)."""
        n = len(train)
        pooled = np.full(n, np.nan)
        _bar = state.get('race_bar')
        _done = []
        try:
            for fi, held in enumerate(_oof_folds):
                keep = np.setdiff1d(np.arange(n), held)
                result = _oof_fold_column(train.iloc[keep], train.iloc[held], params)
                arr = result.get('predicted_valid') if isinstance(result, dict) else None
                vec = None if arr is None else np.asarray(arr, dtype=float).ravel()
                if vec is None or vec.shape[0] != held.shape[0] or not np.all(np.isfinite(vec)):
                    print(f"[hyperparameter_tuning] Candidate failed for {model_name}: an "
                          f"out-of-fold fit returned a missing, wrong-length or non-finite "
                          f"held-out column; scored as worst and skipped.")
                    return _failure_score(), False
                pooled[held] = vec
                _done.append(held)
                if (_oof_race is not None and _bar is not None and fi + 1 >= _oof_race[1]
                        and fi + 1 < len(_oof_folds)):
                    rows = np.concatenate(_done)
                    yp, pp = _oof_y[rows], pooled[rows]
                    if float(np.std(pp)) > 0.0 and float(np.var(yp)) > 0.0:
                        part = (float(np.corrcoef(yp, pp)[0, 1])
                                - float(np.mean((yp - pp) ** 2)) / float(np.var(yp)))
                        if np.isfinite(part) and part < _bar - _oof_race[0]:
                            _RACE_STATS['stopped'] += 1
                            _RACE_STATS['folds_saved'] += len(_oof_folds) - (fi + 1)
                            return float(np.clip(part, -SCORE_CAP, SCORE_CAP)), True
        except Exception as exc:
            print(f"[hyperparameter_tuning] Candidate failed for {model_name} "
                  f"({type(exc).__name__}: {exc}) in an out-of-fold fit; scored as worst and "
                  f"skipped.\n{traceback.format_exc()}")
            return _failure_score(), False
        if not np.all(np.isfinite(pooled)) or float(np.std(pooled)) == 0.0:
            return _failure_score(), False
        # ver4-87: each fold's own r - mse/var, kept for the paired one-standard-error rule
        # (tune_model_hyperparameters); a degenerate fold is NaN.
        _fs = []
        for held in _oof_folds:
            yh, ph = _oof_y[held], pooled[held]
            if held.size > 2 and float(np.std(ph)) > 0.0 and float(np.var(yh)) > 0.0:
                _fs.append(float(np.corrcoef(yh, ph)[0, 1])
                           - float(np.mean((yh - ph) ** 2)) / float(np.var(yh)))
            else:
                _fs.append(float('nan'))
        state['fold_scores'][repr(list(params))] = np.asarray(_fs, dtype=float)
        r = float(np.corrcoef(_oof_y, pooled)[0, 1])
        mse = float(np.mean((_oof_y - pooled) ** 2))
        if not np.isfinite(r) or not np.isfinite(mse):
            return _failure_score(), False
        score = r - ((mse / _oof_var) if _oof_var else mse)
        if not np.isfinite(score):
            return _failure_score(), False
        return float(np.clip(score, -SCORE_CAP, SCORE_CAP)), True

    def _score_once(fit_train, fit_valid, params):
        """Fit once on `fit_train`, score once on `fit_valid`. Returns
        (score, succeeded). `succeeded=False` scores are ALREADY the
        correctly-computed failure score - the caller does not need to
        special-case them further, only decide whether to update
        `state['worst_finite']` from them (it must not: only a genuine,
        finite score is ever "the worst finite score seen so far")."""
        fit_var = valid_var if fit_valid is valid else _actual_variance(fit_valid)
        try:
            result = None
            if _engine == 'exact_kernel':
                result = _engine_metrics(model_name, fit_train, fit_valid, test, params)
            if result is None:
                result = _run_fit(fit_train, fit_valid, params)
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

    def _evaluate_detailed(x):
        """Bug fix (Bayesian-tuned RF/SVR/KNN also underperforming their
        own untuned defaults - Requirement.md follow-up): the SAME
        computation `objective()` below performs, but also returning the
        candidate's own score UNCERTAINTY (standard error across the
        `effective_repeats` inner folds), for `tune_model_hyperparameters()`
        own '1-SE rule' comparison (see that function's own docstring).

        ROOT CAUSE (RF/SVR/KNN specifically). Unlike the MCMC-length fields
        fixed via `floor_at_baseline` above, RF's 'Minimum samples per
        leaf'/'Maximum tree depth', SVR's 'Constraint (C)'/'Epsilon', and
        KNN's 'Number of neighbours' are genuine, two-directional
        regularisation hyperparameters - there IS a real bias/variance
        trade-off in both directions, so a hard floor at the default would
        defeat the entire point of tuning them. But the LOW-regularisation
        end of each of these ranges (small K, high C, small epsilon, no
        depth limit) is also the HIGH-VARIANCE end: a k=1 KNN, for example,
        is far more sensitive to exactly which individuals happen to be
        nearest than a k=15 KNN is. A high-variance candidate's validation
        score is therefore itself a noisier ESTIMATE of its true
        generalisation than a low-variance candidate's score is of ITS
        true generalisation - so a search comparing raw scores across
        candidates of very different variance is disproportionately
        likely to crown a high-variance candidate the "winner" on a lucky
        draw, not because it actually generalises better (the textbook
        argument for scikit-learn/glmnet's own "one-standard-error rule").
        A FIXED `selection_margin` (0.02, same units for every candidate)
        does not account for this - it under-protects against exactly the
        candidates most likely to be spurious winners.

        Returns (mean_score, standard_error, n_folds). `standard_error` is
        `0.0` whenever fewer than 2 folds were evaluated (valid_repeats<=1,
        or a too-small training split fell back to a single evaluation -
        see make_objective()'s own fallback note) - the 1-SE rule has
        nothing to estimate from a single point and
        `tune_model_hyperparameters()` falls back to its plain
        `selection_margin` in that case, exactly as before this fix.
        """
        if _oof_folds is not None:
            # ver4-86: memoised on the DECODED settings, so search points that snap to the same
            # hyperparameters (integer / stepped fields) are fitted once.
            _params = _prepare_params(x)
            _key = repr(list(_params)) if _oof_opts else None   # no memo in the ver4-85 search
            if _key is not None and _key in state['memo']:
                return state['memo'][_key], 0.0, 1
            score, ok = _score_oof(_params)
            if _key is not None:
                state['memo'][_key] = score
            if ok:
                state['worst_finite'] = (
                    score if state['worst_finite'] is None else min(state['worst_finite'], score)
                )
            return score, 0.0, 1
        params = _prepare_params(x)
        if effective_repeats <= 1:
            score, ok = _score_once(train, valid, params)
            if ok:
                state['worst_finite'] = (
                    score if state['worst_finite'] is None else min(state['worst_finite'], score)
                )
            return score, 0.0, 1

        fold_scores = []
        for b in range(effective_repeats):
            tr_b, va_b = _split_train_rows(train, inner_split_seed_base + b)
            if tr_b is None:
                fold_scores.append(_failure_score())
                continue
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
        arr = np.asarray(fold_scores, dtype=float)
        mean = float(arr.mean())
        se = float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
        return mean, se, len(arr)

    def objective(x):
        return _evaluate_detailed(x)[0]

    def _evaluate_screening(x):
        """FEATURE (multi-fidelity screening): the SAME `_prepare_params()`
        -> `_score_once()` computation `objective()` above performs, but
        fit on `_screen_train` (a fixed, reduced-row-count subsample of
        this task's own training split - see the module-level comment
        above `_screen_subsample_train()`) instead of the task's real,
        full-size `train`. Still scored against the task's REAL validation
        split (`valid`, unchanged), so a screening score stays directly
        comparable in scale to a full-fidelity `objective(x)` score - it's
        only a noisier ESTIMATE of it, from a smaller fit.

        Deliberately bypasses the `effective_repeats` inner-resample loop
        `_evaluate_detailed()` uses (a single `_score_once()` call, always)
        even when `valid_repeats > 1` for this search: this is already the
        CHEAP, low-fidelity pass whose only purpose is ranking a batch's
        candidates well enough to decide which half gets promoted to a
        real, full-fidelity (and, if `valid_repeats > 1`, full-repeat)
        evaluation - paying for repeated inner resamples on a screening
        pass would undercut the entire point of screening being cheap.

        Only ever called by `search_bayesian()`'s own batch path, and only
        when `objective.screen` (this function) is actually attached below
        (i.e. `multi_fidelity_screen=True` was requested AND this task's
        `train` was large enough to subsample - see `_screen_train` above).
        """
        params = _prepare_params(x)
        score, ok = _score_once(_screen_train, valid, params)
        if ok:
            state['worst_finite'] = (
                score if state['worst_finite'] is None else min(state['worst_finite'], score)
            )
        return score

    # Attached rather than returned as a second value, so every EXISTING
    # caller of make_objective() (there is only one, but this keeps the
    # plain f(x)->float contract every search algorithm below actually
    # needs) is completely unaffected - only tune_model_hyperparameters()'s
    # own 1-SE comparison reaches for this attribute.
    objective.evaluate_with_se = _evaluate_detailed
    objective.prepare_params = _prepare_params

    def _rescore(x, salts):
        """GAT seed-robust selection (models/gat_stability.py, F3): the score of candidate `x`
        re-evaluated once per seed salt (a different, but candidate-independent, random draw
        for each salt), through the SAME scoring path as the search. NaN for a failed draw.
        Bypasses the memo and restores the stored per-fold scores afterwards, so the paired
        out-of-fold 1-SE rule keeps reading the salt-0 fold scores."""
        from models.gat_stability import seed_salt
        _p = _prepare_params(x)
        _fs_key = repr(list(_p))
        _saved = state['fold_scores'].get(_fs_key)
        out = []
        for salt in salts:
            with seed_salt(salt):
                if _oof_folds is not None:
                    sc, ok = _score_oof(_p)
                elif effective_repeats > 1:
                    sc, ok = _evaluate_detailed(x)[0], True
                else:
                    sc, ok = _score_once(train, valid, _p)
            out.append(float(sc) if ok else float('nan'))
        if _saved is not None:
            state['fold_scores'][_fs_key] = _saved
        return out
    objective.rescore = _rescore

    def _prime_oof_race(x):
        """ver4-86: score `x` (the default point) on all folds, in this process, BEFORE the
        search starts, and make that score the fold-racing bar. Its memo entry makes the
        search's own evaluation of `x` free, in this process or in a worker (the objective is
        pickled with its state after this call). Returns the score."""
        _score = objective(x)
        state['race_bar'] = _score
        return _score
    if _oof_folds is not None and _oof_race is not None:
        objective.prime_oof_race = _prime_oof_race

    def _oof_fold_scores(x):
        """ver4-87: the per-fold scores of candidate `x` (an array, one value per fold), scoring
        it on all folds in THIS process first when they are not known here (a candidate scored
        in a worker, or stopped by fold racing - racing never stops the winner or the primed
        default). None when the candidate fails."""
        _p = _prepare_params(x)
        _k = repr(list(_p))
        if _k not in state['fold_scores']:
            _s, _ok = _score_oof(_p)
            if not _ok or _k not in state['fold_scores']:
                return None
        return state['fold_scores'][_k]
    if _oof_folds is not None:
        objective.oof_fold_scores = _oof_fold_scores
    # FEATURE (multi-fidelity screening): likewise only ever present when
    # `_screen_train` above is not None - absent (no attribute at all, not
    # merely None) for every existing caller, since `getattr(objective,
    # 'screen', None)` is exactly how search_bayesian() checks for it.
    if _screen_train is not None:
        objective.screen = _evaluate_screening
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
# ver4-35, R1.a: per-candidate cost of a search, for the per-task `[timing]` line.
# A candidate's seconds are measured INSIDE whichever process evaluates it (the
# parent for a serial search, a loky worker for a parallel one), because only
# that duration shows what the fit itself cost - the parent only sees the wall
# time of the whole batch. `_evaluate_batch()` is the one place every
# Grid/Random/Bayesian batch goes through, so the totals are kept here and read
# by `tune_model_hyperparameters()`. One-point-at-a-time evaluations that do not
# use `_evaluate_batch()` (the default point, Nelder-Mead / Powell steps) are
# not counted; they are inside the search's own wall time.
_TRIAL_STATS = {'trials': 0, 'secs': 0.0}
# ver4-86: fold racing counters of out-of-fold searches evaluated in THIS process (a worker's own
# counts stay in the worker; the search log reports what the parent saw).
_RACE_STATS = {'stopped': 0, 'folds_saved': 0}


class _TimedCandidate:
    """Picklable wrapper: ``wrapper(x)`` returns ``(objective(x), seconds)``,
    with the seconds measured in the process that ran the call."""

    def __init__(self, objective):
        self._objective = objective

    def __call__(self, x):
        started = time.perf_counter()
        score = self._objective(x)
        return score, time.perf_counter() - started


def _unwrap_timed(results):
    """Split ``[(score, secs), ...]`` into the scores (what every caller of
    `_evaluate_batch()` expects) and add the seconds to `_TRIAL_STATS`."""
    scores = [score for score, _ in results]
    _TRIAL_STATS['trials'] += len(results)
    _TRIAL_STATS['secs'] += float(sum(secs for _, secs in results))
    return scores


# ver4-46, R4 - "ship the objective once".
#
# A search objective is a closure over the task's train / valid / test frames (and the model
# runner). Submitting `_TimedCandidate(objective)` to a worker pickles ALL of that for every
# candidate - 30-40 times per search, at q candidates per round. With `ship_once` the closure
# is written ONCE to a temporary file (cloudpickle, the serializer loky itself uses) and every
# candidate carries only the file name; each worker process loads it on first use and keeps it
# in `_WORKER_OBJECTIVES`. The loaded objective is the same function with the same data, so the
# scores are identical; any failure to ship leaves the normal path untouched.
_WORKER_OBJECTIVES = {}
_WORKER_OBJECTIVES_MAX = 2


class _SharedObjective:
    """Picklable stand-in for an objective that lives in a file (see the comment above)."""

    def __init__(self, path, which):
        self.path = path
        self.which = which          # 0 = full-fidelity objective, 1 = screening evaluator

    def __call__(self, x):
        pair = _WORKER_OBJECTIVES.get(self.path)
        if pair is None:
            import pickle
            with open(self.path, 'rb') as handle:
                pair = pickle.load(handle)
            while len(_WORKER_OBJECTIVES) >= _WORKER_OBJECTIVES_MAX:
                _WORKER_OBJECTIVES.pop(next(iter(_WORKER_OBJECTIVES)))
            _WORKER_OBJECTIVES[self.path] = pair
        return pair[self.which](x)


def _ship_objective(objective, screen=None):
    """Write `(objective, screen)` to a temporary file once. Returns
    ``(shared_objective, shared_screen_or_None, cleanup)`` or ``None`` when shipping is not
    possible (the caller then keeps the normal per-candidate pickling; one NOTE is printed)."""
    import shutil
    import tempfile
    directory = None
    try:
        try:
            from joblib.externals import cloudpickle
        except ImportError:          # some distro builds of joblib strip the vendored copy
            import cloudpickle
        directory = tempfile.mkdtemp(prefix='easigp_objective_')
        path = os.path.join(directory, 'objective.pkl')
        with open(path, 'wb') as handle:
            cloudpickle.dump((objective, screen), handle)
        _cleanup = lambda: shutil.rmtree(directory, ignore_errors=True)  # noqa: E731
        return (_SharedObjective(path, 0),
                _SharedObjective(path, 1) if screen is not None else None, _cleanup)
    except Exception as exc:
        if directory:
            shutil.rmtree(directory, ignore_errors=True)
        print(f"[hyperparameter_tuning] NOTE: ship_once could not store the search objective "
              f"({type(exc).__name__}: {exc}) - candidates carry it as before.")
        return None


def _clean_limit(value, key, model_name):
    """ver4-61 R3: validate a time limit given in SECONDS. Returns a positive float, or None
    for absent/0 ("off"). A negative, non-numeric or non-finite value is reported as a
    WARNING and ignored (never fatal: a bad limit must not abort a long run)."""
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        seconds = float('nan')
    if not math.isfinite(seconds) or seconds < 0:
        print(f"[hyperparameter_tuning] WARNING: {model_name} {key}={value!r} is not a number "
              f">= 0 - the limit is ignored (treated as off).")
        return None
    return seconds if seconds > 0 else None


class _KillSwitchCandidate:
    """ver4-61, R2: TEST HOOK, only ever built when ``EASIGP_TEST_KILL_TRIAL`` is set.
    Picklable stand-in for a candidate whose worker is SIGKILLed before it evaluates
    anything - the same failure the kernel's out-of-memory killer causes - so the
    worker-death isolation in `_evaluate_batch_isolated()` can be exercised on demand."""

    def __call__(self, x):
        import signal
        os.kill(os.getpid(), getattr(signal, 'SIGKILL', signal.SIGTERM))


def _is_worker_death(exc):
    """True when `exc` is a process-pool breakage (a worker was killed, e.g. by the
    out-of-memory killer): loky's ``TerminatedWorkerError`` and ``BrokenProcessPool`` both
    derive from ``concurrent.futures.process.BrokenProcessPool``."""
    from concurrent.futures.process import BrokenProcessPool
    return isinstance(exc, BrokenProcessPool)


def _test_kill_index():
    """The 1-based candidate number to SIGKILL (test hook), or None when the hook is off
    or malformed (a malformed value is reported once per call and ignored)."""
    raw = os.environ.get('EASIGP_TEST_KILL_TRIAL')
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        print(f"[hyperparameter_tuning] NOTE: EASIGP_TEST_KILL_TRIAL={raw!r} is not an "
              f"integer - the test hook is ignored.")
        return None


def _evaluate_batch_isolated(timed, xs, n_jobs, worker_init, worker_init_args, executor,
                             state):
    """ver4-61, R2: evaluate one round with one future per candidate so that a worker death
    loses only the candidates it took down, not the whole round.

    Why: the joblib ``Parallel`` path discards the entire round when one worker is
    SIGKILLed (out of memory on a bio-prior search; the attached run lost 8 candidates to
    one kill and then retried at the same width every round).

    Behaviour: candidates run on a reusable loky executor of ``state['width']`` workers.
    Any candidate whose future fails with a worker-death error (the killed worker's own
    candidate and, because loky marks the whole pool broken, the candidates in flight
    beside it) is re-run with the same objective, so its score is the one an unharmed run
    would have produced; candidates that had already finished are kept. Each re-run uses a
    fresh executor at HALF the previous width (a repeat death halves again), ending with
    an in-process serial run at width 1. ``state['width']`` records the width finally used
    and is sticky: later rounds, and later calls sharing `state`, start from it.
    (The blueprint's wording is "re-run serially in-process"; re-running up to q lost
    trials one by one would cost q trial-lengths, hours for a bio-prior fit, so the halved
    pool is tried first - same scores, see the Change Summary, deviation D-R2a.)

    Args: `timed` the picklable `_TimedCandidate`; `xs` candidates; `n_jobs` requested
    width; `worker_init`/`worker_init_args` as for `_loky_executor`; `executor` the
    caller's executor (used only while the width is unchanged); `state` the mutable dict
    that carries ``width``, ``executor`` and the test-hook counter between rounds.
    Returns ``[(score, seconds), ...]`` in submission order, or ``None`` when a non-death
    failure occurs (the caller then uses its legacy path). Never raises for worker death."""
    width = max(1, min(int(state.setdefault('width', int(n_jobs))), int(n_jobs)))
    if width <= 1:
        return [timed(x) for x in xs]
    pool = state.get('executor')
    if pool is None:
        pool = executor if (executor is not None and width == int(n_jobs)) else None
    if pool is None:
        pool = _loky_executor(width, worker_init, worker_init_args)
    state['executor'] = pool
    kill_at = _test_kill_index()
    submitted = []
    for x in xs:
        state['n_submitted'] = int(state.get('n_submitted', 0)) + 1
        fn = timed
        if kill_at is not None and state['n_submitted'] == kill_at \
                and not state.get('kill_fired'):
            state['kill_fired'] = True
            fn = _KillSwitchCandidate()
            print(f"[hyperparameter_tuning] NOTE: EASIGP_TEST_KILL_TRIAL={kill_at}: the "
                  f"worker of candidate {kill_at} is SIGKILLed on purpose (test hook).")
        try:
            submitted.append((pool.submit(fn, x), None))
        except Exception as exc:
            submitted.append((None, exc))
    results = [None] * len(xs)
    dead = []
    for i, (future, submit_error) in enumerate(submitted):
        try:
            if submit_error is not None:
                raise submit_error
            results[i] = future.result()
        except Exception as exc:
            if _is_worker_death(exc):
                dead.append(i)
                continue
            print(f"[hyperparameter_tuning] NOTE: candidate {i + 1} of this round failed in "
                  f"its worker with a non-death error ({type(exc).__name__}: {exc}) - the "
                  f"round is evaluated through the legacy path instead.")
            return None
    if dead:
        state['worker_deaths'] = int(state.get('worker_deaths', 0)) + 1
        print(f"[hyperparameter_tuning] WARNING: {len(dead)} of {len(xs)} candidate(s) in "
              f"this round lost their worker process (position(s) "
              f"{[i + 1 for i in dead]}; most likely the operating system killed a worker "
              f"for lack of memory, and the pool discards the candidates that were in "
              f"flight beside it) - finished candidates are kept and only these are "
              f"re-run, at half the width each time until they complete.")
        pending = dead
        while pending:
            width = max(1, width // 2)
            state['width'] = width
            state['executor'] = None
            if width <= 1:
                print(f"[hyperparameter_tuning] WARNING: re-running {len(pending)} "
                      f"candidate(s) one at a time in this process; later rounds also run "
                      f"serially (sticky).")
                for i in pending:
                    results[i] = timed(xs[i])
                pending = []
                break
            print(f"[hyperparameter_tuning] WARNING: re-running {len(pending)} candidate(s) "
                  f"on {width} worker(s); later rounds use {width} worker(s) as well "
                  f"(sticky).")
            retry_pool = _loky_executor(width, worker_init, worker_init_args)
            state['executor'] = retry_pool
            retry = [(i, retry_pool.submit(timed, xs[i])) for i in pending]
            pending = []
            for i, future in retry:
                try:
                    results[i] = future.result()
                except Exception as exc:
                    if not _is_worker_death(exc):
                        print(f"[hyperparameter_tuning] NOTE: candidate {i + 1} failed on "
                              f"re-run with a non-death error ({type(exc).__name__}: {exc})"
                              f" - the round is evaluated through the legacy path instead.")
                        return None
                    pending.append(i)
    return results


def _evaluate_batch(objective, xs, n_jobs, worker_init=None, worker_init_args=(),
                    executor=None, *, isolate_failures=False, state=None):
    """
    Parameters
    ----------
    isolate_failures, state : ver4-61 R2 - when True (and the batch fans out), each candidate
        is its own future on a reusable loky executor, so a killed worker costs only the
        candidates it took down (re-run serially) rather than the whole round, and the
        usable width is halved for every later round (``state['width']``, sticky across
        calls that share the `state` dict; a fresh dict is made when None). Any unexpected
        failure falls through to the legacy paths below, so correctness never depends on it.
        False (every pre-ver4-61 caller) leaves the body below unchanged.
    executor : ver4-40 R1 (HP_TUNE_FAST['persistent_executor']) - an optional,
        already-built reusable ``concurrent.futures``-style executor (see
        `_loky_executor()`). When given and the batch has more than one candidate, the
        candidates are submitted to it instead of building a fresh
        ``joblib.Parallel`` every round, so worker start-up (module imports, R/rpy2) is
        paid once per run rather than once per batch (root cause C5). Results are read
        back in SUBMISSION order, so the (x, score) pairing is identical to the
        ``Parallel`` path. Any executor failure logs a NOTE and falls through to the
        ``_loky_parallel`` path below, so the result never depends on it. `None` (every
        pre-ver4-40 caller) leaves the body below byte-for-byte unchanged.
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
    timed = _TimedCandidate(objective)
    if n_jobs is None or n_jobs in (0, 1) or len(xs) <= 1:
        return _unwrap_timed([timed(x) for x in xs])
    if isolate_failures:
        _iso_state = state if state is not None else {}
        try:
            _isolated = _evaluate_batch_isolated(timed, xs, n_jobs, worker_init,
                                                 worker_init_args, executor, _iso_state)
        except Exception as exc:
            print(f"[hyperparameter_tuning] NOTE: failure-isolated trial evaluation could "
                  f"not run ({type(exc).__name__}: {exc}) - this round uses the legacy "
                  f"evaluation path (same result, no isolation).")
            _isolated = None
        if _isolated is not None:
            return _unwrap_timed(_isolated)
    if executor is not None:
        try:
            futures = [executor.submit(timed, x) for x in xs]
            return _unwrap_timed([f.result() for f in futures])
        except Exception as exc:
            print(f"[hyperparameter_tuning] NOTE: the persistent trial executor failed "
                  f"({type(exc).__name__}: {exc}) - this round is evaluated through a "
                  f"fresh joblib pool instead (same result, slower start-up).")
    try:
        from joblib import delayed
        return _unwrap_timed(_loky_parallel(n_jobs, worker_init, worker_init_args)(
            delayed(timed)(x) for x in xs
        ))
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
        return _unwrap_timed([timed(x) for x in xs])


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
    mode is off, `n_jobs` isn't a usable worker count, or (via the
    caller's own `nested_safe_n_jobs()` resolution, upstream of this
    module - see pipeline_utils.py) this call is itself already running
    inside a daemonic worker process. Never raises.

    PERFORMANCE FIX (Requirement: "Bayesian hyperparameter tuning is
    prohibitively slow - at least 8x a non-tuning run"). This function
    used to treat ANY `n_jobs <= 1` - including 0 or negative - as "not
    usable", collapsing q to 1. That is correct for `n_jobs in (0, 1)`,
    but WRONG for a negative value: -1 (and, by the same joblib/sklearn
    convention this whole codebase otherwise honours - see
    pipeline_utils.resolve_compute_resources()'s own N_JOBS default and
    nested_safe_n_jobs()'s docstring - -2, -3, ... "all but N-1 cores")
    is the standard sentinel for "use every available core", not "run
    single-threaded". N_JOBS defaults to -1, and genomic_prediction.py's
    HP_TUNE dispatch site hands that value to
    tune_model_hyperparameters()/run_search()/search_bayesian() UNRESOLVED
    (unlike models/Bayesian_optimisation.py's own restart fan-out, which
    already routes through pipeline_utils.nested_safe_n_jobs() first) -
    so under the out-of-the-box default this function ALWAYS returned
    q=1, silently disabling HP_TUNE_BAYES_BATCH's parallel trial
    evaluation regardless of how many CPUs the run actually had
    available. Every candidate (init_points + n_iter - 35 of them at the
    GUI's own default budget) was then evaluated strictly one at a time,
    while Grid/Random (via `_evaluate_batch()`) and Nelder-Mead/Powell
    (via `_multistart_local_search()`) - both of which already treat any
    `n_jobs not in (0, 1)` as usable and hand it straight to
    `joblib.Parallel` - parallelised correctly. Combined with the search-
    phase fidelity reduction (`_cheapen_fidelity()`/`_cheapen_mcmc()`,
    which only lowers per-trial COMPUTE, not wall-clock trial COUNT),
    this made Bayesian tuning's wall-clock cost scale with the full
    serial trial count instead of `trial_count / q` - the dominant
    contributor to the reported >=8x slowdown, specific to this one
    algorithm. Resolving the sentinel here (self-contained, so this
    function's contract is correct regardless of whether a future caller
    happens to pre-resolve n_jobs or not) restores the parallel fan-out
    HP_TUNE_BAYES_BATCH was always meant to provide.
    """
    if not batch:
        return 1
    # Mirrors pipeline_utils.nested_safe_n_jobs()'s own daemon check: a
    # daemonic process (this HP_TUNE call already running inside an
    # intra_batch_parallel/intra_task_parallel worker) cannot spawn its
    # own children, so resolving the -1 sentinel to os.cpu_count() below
    # and then trying to fan out anyway would just make _evaluate_batch()
    # pay for a doomed spawn attempt (caught and serially retried) on
    # every batch round instead of running q=1 cleanly from the start.
    try:
        import multiprocessing
        if multiprocessing.current_process().daemon:
            return 1
    except Exception:
        pass
    try:
        _n = int(n_jobs)
    except (TypeError, ValueError):
        return 1
    if _n < 0:
        _n = os.cpu_count() or 1
    if _n <= 1:
        return 1
    return max(1, min(_n, max(1, int(batch_max))))


# --------------------------------------------------------------------------- #
# PERFORMANCE FIX, part 2 (Requirement: "Bayesian hyperparameter tuning is
# prohibitive - at least 8x a non-tuning run"). Resolving `q` in
# `_resolve_q()` (above) fixed the FIRST bug - candidates now actually run
# concurrently across worker PROCESSES instead of one at a time. But making
# them run concurrently exposed a SECOND, previously-latent bug: every one of
# those worker processes was being initialised (via `_trial_worker_init()` in
# genomic_prediction.py) with this run's own already-resolved, UN-DIVIDED
# per-process compute budget (`n_jobs` for scikit-learn's RF/ExtraTrees/GBDT/
# SVR/KNN and their SHAP calls, `r_blas_threads` for the four BGLR-backed R
# models' underlying BLAS, `torch_num_threads` for MLP/GAT) - the exact same
# budget a SINGLE serial model fit is sized for. With `q` (or, for Grid/
# Random/Nelder-Mead/Powell, `n_jobs`/the restart count) copies of that same
# full budget now running AT THE SAME TIME, a run configured with
# N_JOBS=12 and HP_TUNE_BAYES_BATCH_MAX=8 was asking each of 8 concurrent
# trial workers to use 12 BLAS/scikit-learn threads simultaneously - up to
# 96 threads contending for however many cores this node actually has. CPU
# oversubscription of this kind does not merely fail to speed anything up;
# thread contention/context-switching overhead can make the parallel path
# markedly SLOWER than the serial one it was meant to replace - which is
# exactly the "batch_max=8, yet still >=8x slower" shape of the reported
# symptom, and scales with the batch width the same way the report's own
# number does.
#
# `resolve_trial_fanout_width()` is the single-authority answer to "how many
# trial-evaluation WORKER PROCESSES could tune_model_hyperparameters(
# algorithm=...) have running at the same instant, for these settings?" -
# the number every caller needs in order to divide (never re-multiply) a
# per-process resource budget before handing it to a freshly spawned trial
# worker (the same "divide, never multiply" discipline
# pipeline_utils.nested_safe_n_jobs() already documents for the other nested-
# parallelism cases). Deliberately a query-only function - it does not run
# anything, and it does not know whether a given search will actually reach
# this width (a picklability failure, a disabled batch flag, or an already-
# daemonic caller can all silently reduce the REAL concurrency at run time -
# see _evaluate_batch()'s/_multistart_local_search()'s own fallback notes).
# Callers should treat this as a worst case and divide accordingly: sizing a
# worker's budget for a fallback-to-serial run that never materialises only
# leaves a little headroom unused (mild under-utilisation) - the failure mode
# this function exists to prevent (oversubscription) is the far more costly
# one, so the conservative (larger) estimate is always the safe one.
def resolve_trial_fanout_width(algorithm, n_jobs, bayes_batch=True, bayes_batch_max=8,
                                parallel_restarts=True, budget_kwargs=None):
    """Upper bound on concurrent trial-evaluation worker processes for
    `algorithm` under these settings. See the module-level comment above
    this function for the full rationale.

    Mirrors each algorithm's OWN concurrency resolution exactly, rather than
    a separate, potentially-drifting estimate of it:
      - 'Bayesian': identical to `_resolve_q()` (this module's own single
        authority for that algorithm's batch width).
      - 'Nelder-Mead' / 'Powell': `min(n_jobs, n_restarts)` - joblib never
        spawns more workers than there are submitted restarts - or `1` when
        `parallel_restarts` is off (see `run_search()`'s own handling of
        this flag).
      - 'Grid' / 'Random': `n_jobs` itself (`_evaluate_batch()`'s own fan-
        out width; joblib further self-limits to the candidate count, which
        only makes the real number smaller, never larger).
      - anything else (unrecognised algorithm): `1` - never invents
        concurrency for an algorithm this module doesn't itself parallelise.

    `n_jobs` of `None` or a negative sentinel (e.g. `-1`, "every core") is
    resolved via `os.cpu_count()` first, the same convention `_resolve_q()`
    and `pipeline_utils.nested_safe_n_jobs()` already use.
    """
    try:
        _n = int(n_jobs) if n_jobs is not None else -1
    except (TypeError, ValueError):
        _n = -1
    if _n < 0:
        _n = os.cpu_count() or 1
    _n = max(1, _n)

    if algorithm == 'Bayesian':
        return _resolve_q(_n, bayes_batch, bayes_batch_max)
    if algorithm in ('Nelder-Mead', 'Powell'):
        if not parallel_restarts:
            return 1
        _n_restarts = max(1, int((budget_kwargs or {}).get('n_restarts', 4)))
        return max(1, min(_n, _n_restarts))
    if algorithm in ('Grid', 'Random'):
        return _n
    return 1


def search_bayesian(objective, specs, n_iter=30, init_points=5, seed=0, default_x=None,
                     n_jobs=1, batch=True, batch_max=8, liar='max',
                     worker_init=None, worker_init_args=(), domain_reduction='auto',
                     multi_fidelity=False, should_stop=None, executor=None,
                     async_batch=False, ship_once=False, time_stop=None,
                     isolate_failures=False):
    """
    Parameters
    ----------
    time_stop : callable or None, default None (ver4-61 R3) - the wall-clock rule from
        `parallel_search.time_budget_stop()`. Kept apart from `should_stop` because the two
        differ inside the serial path: the patience rule only starts after the initial random
        block (comparing random draws would mean nothing), whereas a time budget must be able
        to end the search anywhere, so the serial path then steps the initial points one at
        a time as well (`[GP] NUMERICS` line). In batch mode both are OR-composed into the
        single `should_stop` the batch engines take. None leaves every earlier behaviour.
    isolate_failures : bool, default False (ver4-61 R2) - batch mode only; passed to
        `_evaluate_batch(isolate_failures=...)` with one `state` dict shared by every round
        of this search, so a worker death loses only the candidates it took down and the
        usable width stays halved for the rest of the search.
    async_batch : bool, default False (ver4-46 R4, HP_TUNE_FAST['async_batch']) - batch mode
        only, and only with a persistent `executor`. Instead of waiting for the slowest of
        `q` candidates every round, keeps `q` candidates in flight and starts a new one (chosen
        by the surrogate with a constant-liar value for every candidate still running) the
        moment a worker frees up. Candidate costs differ by up to ~250x for the GAT models,
        so a synchronous round leaves most workers idle. The candidate SEQUENCE then depends
        on completion timing (a `[GP] NUMERICS:` line says so); the evaluation budget is
        unchanged. Falls back to the synchronous batch path on any failure.
    ship_once : bool, default False (ver4-46 R4) - batch mode only. The search objective (the
        train/valid/test frames and the model runner) is written once to a temporary file and
        each worker loads it once, instead of every submitted candidate carrying a pickled
        copy. Scores are identical.
    should_stop : callable(trials) -> bool or None, default None (ver4-40 R1, root
        cause C4). Early-stopping hook, built by `parallel_search.patience_stop()`
        (HP_TUNE_FAST['patience']). `trials` is the ``[(x, score), ...]`` history so far.
        The batch path hands it to ``batch_bayesian_maximise(should_stop=)`` (checked
        between rounds). The serial path replaces ``optimizer.maximize(n_iter=N)`` by
        ``maximize(init_points=init_points, n_iter=0)`` followed by up to N single
        ``maximize(init_points=0, n_iter=1)`` steps and checks the hook after each one -
        the candidate sequence can differ slightly from a single ``maximize`` call, so a
        ``[GP] NUMERICS:`` line is logged. `None` keeps the legacy single ``maximize``
        call byte-for-byte.
    executor : ver4-40 R1 - forwarded to `_evaluate_batch(executor=)` by the batch path
        (HP_TUNE_FAST['persistent_executor']); irrelevant to the serial path.
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
    multi_fidelity : bool, default False (HP_TUNE_BAYES_MULTI_FIDELITY) -
        only has any effect when this call is ALSO running in batch mode
        (`q > 1 and len(specs) > 0` above) AND `objective` itself was built
        with a `.screen` attribute (see `make_objective()`'s own
        `multi_fidelity_screen=` parameter) - every other combination is a
        documented, logged no-op (see the NOTE lines below), never a
        silent one. When both hold, each batch round first scores every
        candidate cheaply via `objective.screen(x)` (a fit on a small,
        fixed row-count subsample of this task's own training split - see
        `make_objective()`'s own docstring), then re-scores only the
        better half of that round via the real, full-fidelity
        `objective(x)` - the rest keep their cheaper screening score. The
        search's own RESULT can differ slightly from `multi_fidelity=False`
        (a screening-fidelity score is a noisier estimate than a
        full-fidelity one), trading a small amount of search quality for
        fewer full model fits per round - see genomic_prediction.py's own
        HP_TUNE_BAYES_MULTI_FIDELITY design note for the full rationale.
    """
    _ensure_bayes_opt()
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
    if multi_fidelity and not (q > 1 and len(specs) > 0):
        # FEATURE (multi-fidelity screening): logged unconditionally (I11 -
        # "resolved state logged unconditionally"), exactly like every
        # other requested-but-inapplicable setting in this module (see the
        # domain_reduction 'auto'-skips-categorical NOTE just above for the
        # same discipline). This is the ONLY place that can print this
        # NOTE - `objective.screen` may perfectly well be present (a
        # perfectly normal, request-independent fact about `objective`
        # itself) even when the batch path isn't in use at all, so the
        # decision of whether the request had any effect belongs here,
        # not to a bare attribute check.
        print(f"[GP] NOTE: Bayesian hyperparameter search's 'multi_fidelity' "
              f"(HP_TUNE_BAYES_MULTI_FIDELITY) was requested but has no effect for this "
              f"search - it only applies to the BATCH search path (HP_TUNE_BAYES_BATCH=True "
              f"with more than one usable worker AND at least one tunable field), which isn't "
              f"in use here (batch={bool(batch)}, q={q}, n_tunable_fields={len(specs)}). Every "
              f"candidate is evaluated at the normal full-fidelity, exactly as if "
              f"'multi_fidelity' were off.")
    if q > 1 and len(specs) > 0:
        print(f"[GP] NUMERICS: Bayesian hyperparameter search running in BATCH mode "
              f"(q={q}, liar={liar!r}); candidate sequence differs from serial (q=1). "
              f"Set HP_TUNE_BAYES_BATCH=false to restore.")
        from .parallel_search import any_stop, batch_bayesian_maximise

        _batch_stop = any_stop(should_stop, time_stop)
        _iso_state = {'width': int(n_jobs)} if isolate_failures else None
        _screen = getattr(objective, 'screen', None) if multi_fidelity else None
        if multi_fidelity and _screen is None:
            print(f"[GP] NOTE: Bayesian hyperparameter search's 'multi_fidelity' "
                  f"(HP_TUNE_BAYES_MULTI_FIDELITY) was requested but this task's own training "
                  f"split has no usable reduced-row-count screening evaluator (see "
                  f"make_objective()'s own multi_fidelity_screen= parameter and "
                  f"_screen_subsample_train()'s own docstring for when that happens - most "
                  f"commonly, too few training rows to subsample meaningfully). Every candidate "
                  f"in this batch search is evaluated at full fidelity, exactly as if "
                  f"'multi_fidelity' were off.")

        # ver4-46, R4: with ship_once the objective is stored once and workers load it once.
        _pool_objective, _pool_screen, _ship_cleanup = objective, _screen, None
        if ship_once and n_jobs and n_jobs > 1:
            _shipped = _ship_objective(objective, _screen)
            if _shipped is not None:
                _pool_objective, _shipped_screen, _ship_cleanup = _shipped
                if _screen is not None:
                    _pool_screen = _shipped_screen

        if _screen is not None:
            # FEATURE (multi-fidelity screening): every candidate in this
            # round is first scored cheaply via `_screen` (a fit on a
            # small, fixed subsample of this task's own training rows -
            # see make_objective()'s own docstring), then only the
            # better HALF of the round is re-scored via the real,
            # full-fidelity `objective` - the rest keep their cheaper
            # screening score. Both passes reuse `_evaluate_batch()`'s own
            # n_jobs-parallel fan-out, exactly like the plain
            # (non-multi-fidelity) branch below, so this preserves
            # whatever wall-clock benefit HP_TUNE_PARALLEL_TRIALS/n_jobs
            # already gives - the saving here is in fitting FEWER
            # candidates at the full training row count per round, not in
            # how each individual fit itself is parallelised.
            def objective_batch(xs):
                if len(xs) <= 1:
                    return _evaluate_batch(_pool_objective, xs, n_jobs, worker_init=worker_init,
                                            worker_init_args=worker_init_args, executor=executor,
                                            isolate_failures=isolate_failures, state=_iso_state)
                screen_scores = _evaluate_batch(_pool_screen, xs, n_jobs, worker_init=worker_init,
                                                 worker_init_args=worker_init_args,
                                                 executor=executor,
                                                 isolate_failures=isolate_failures,
                                                 state=_iso_state)
                n_keep = max(1, math.ceil(len(xs) * _MULTI_FIDELITY_KEEP_FRACTION))
                order = sorted(range(len(xs)), key=lambda i: screen_scores[i], reverse=True)
                promoted = sorted(order[:n_keep])
                promoted_scores = _evaluate_batch(
                    _pool_objective, [xs[i] for i in promoted], n_jobs,
                    worker_init=worker_init, worker_init_args=worker_init_args,
                    executor=executor, isolate_failures=isolate_failures, state=_iso_state,
                )
                final_scores = list(screen_scores)
                for pos, i in enumerate(promoted):
                    final_scores[i] = promoted_scores[pos]
                return final_scores
        else:
            def objective_batch(xs):
                return _evaluate_batch(_pool_objective, xs, n_jobs, worker_init=worker_init,
                                        worker_init_args=worker_init_args, executor=executor,
                                        isolate_failures=isolate_failures, state=_iso_state)

        try:
            if async_batch and executor is not None and _screen is None:
                try:
                    from .parallel_search import async_bayesian_maximise
                    _timed = _TimedCandidate(_pool_objective)

                    def _async_result(future):
                        _score, _secs = future.result()
                        _TRIAL_STATS['trials'] += 1
                        _TRIAL_STATS['secs'] += float(_secs)
                        return _score

                    print(f"[GP] NUMERICS: asynchronous batch Bayesian search (q={q}, "
                          f"liar={liar!r}): a new candidate starts as soon as a worker frees "
                          f"up, so the candidate sequence depends on completion timing. Set "
                          f"HP_TUNE_FAST['async_batch']=false to restore synchronous rounds.")
                    _a_x, _a_score, _a_trials = async_bayesian_maximise(
                        lambda x: executor.submit(_timed, x), _async_result, bounds,
                        n_iter=n_iter, init_points=init_points, seed=seed, q=q,
                        probe_points=([default_x] if default_x is not None else None),
                        acquisition_factory=_make_acquisition,
                        bounds_transformer_factory=_make_bounds_transformer,
                        liar=liar, should_stop=_batch_stop,
                    )
                    _a_default = None
                    if default_x is not None and _a_trials:
                        for _tx, _ts in _a_trials:
                            if np.allclose(_tx, default_x):
                                _a_default = _ts
                                break
                    return _a_x, _a_score, _a_trials, _a_default
                except Exception as exc:
                    print(f"[hyperparameter_tuning] NOTE: asynchronous batch search failed "
                          f"({type(exc).__name__}: {exc}) - using synchronous batch rounds "
                          f"for this model. This search's own RESULT is unaffected, only "
                          f"wall-clock time.")
            elif async_batch:
                print(f"[hyperparameter_tuning] NOTE: async_batch needs the persistent trial "
                      f"executor (HP_TUNE_FAST['persistent_executor']) and no multi-fidelity "
                      f"screening - synchronous batch rounds are used.")

            try:
                _batch_best_x, _batch_best_score, _batch_trials = batch_bayesian_maximise(
                    objective_batch, bounds, n_iter=n_iter, init_points=init_points, seed=seed, q=q,
                    probe_points=([default_x] if default_x is not None else None),
                    acquisition_factory=_make_acquisition,
                    bounds_transformer_factory=_make_bounds_transformer,
                    liar=liar, should_stop=_batch_stop,
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
        finally:
            if _ship_cleanup is not None:
                _ship_cleanup()

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

    if should_stop is None and time_stop is None:
        optimizer.maximize(init_points=init_points, n_iter=n_iter)
    else:
        # ver4-40 R1 (C4): same total budget, but one guided candidate at a time so the
        # patience rule can end a stalled search. The initial random block always runs in
        # full for the patience rule (stopping inside it would compare random draws, not a
        # converging search).
        # ver4-61 R3: a wall-clock rule (`time_stop`) must be able to end the search at any
        # point, so with it the initial block is stepped one point at a time too. Measured
        # (P0.6, bayes_opt 3.4.0): k calls of maximize(init_points=1, n_iter=0) draw exactly
        # the points one maximize(init_points=k, n_iter=0) call draws, so stepping changes
        # no candidate - only the stopping granularity.
        _rules = ' and '.join(r for r, on in (
            ("a patience stop (HP_TUNE_FAST['patience'])", should_stop is not None),
            ("a wall-clock stop (HP_TUNE[model]['time_limit_min'])", time_stop is not None))
            if on)
        print(f"[GP] NUMERICS: serial Bayesian hyperparameter search stepping one candidate "
              f"at a time with {_rules}; the candidate sequence can differ slightly from a "
              f"single maximize() call. Set patience=0 / remove time_limit_min to restore.")

        def _history_now():
            return [(np.array([r['params'][n] for n in names]), float(r['target']))
                    for r in optimizer.res]

        _time_hit = False
        if time_stop is None:
            optimizer.maximize(init_points=init_points, n_iter=0)
        else:
            for _ in range(int(init_points)):
                if time_stop(_history_now()):
                    _time_hit = True
                    break
                optimizer.maximize(init_points=1, n_iter=0)
        _done = 0
        while _done < int(n_iter) and not _time_hit:
            _history = _history_now()
            try:
                _patience_hit = bool(should_stop(_history)) if should_stop is not None \
                    else False
            except Exception as exc:
                print(f"[hyperparameter_tuning] NOTE: should_stop raised {exc!r} - "
                      f"the patience stop is disabled for the rest of this search.")
                should_stop = None
                if time_stop is None:
                    optimizer.maximize(init_points=0, n_iter=int(n_iter) - _done)
                    break
                _patience_hit = False
            if time_stop is not None and time_stop(_history):
                _time_hit = True
                print(f"[hyperparameter_tuning] NOTE: Bayesian search ended by the time "
                      f"budget after {len(_history)} candidates; {int(n_iter) - _done} "
                      f"guided candidates skipped.")
                break
            if _patience_hit:
                print(f"[hyperparameter_tuning] NOTE: Bayesian search stopped early after "
                      f"{len(_history)} candidates (patience rule: no improvement in the "
                      f"recent window); {int(n_iter) - _done} guided candidates skipped.")
                break
            optimizer.maximize(init_points=0, n_iter=1)
            _done += 1
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
                bayes_liar='max', parallel_restarts=True, bayes_domain_reduction='auto',
                bayes_multi_fidelity=False, should_stop=None, executor=None,
                async_batch=False, ship_once=False, time_stop=None, isolate_failures=False):
    """Dispatch to `algorithm`'s own search function.

    ver4-61: `time_stop` (R3, wall-clock rule) and `isolate_failures` (R2, worker-death
    isolation) are forwarded to 'Bayesian' ONLY, and only when set - every other call is
    exactly the earlier call.

    ver4-46 R4: `async_batch` / `ship_once` (HP_TUNE_FAST) are forwarded to 'Bayesian' ONLY,
    and only when True - every other call is exactly the pre-ver4-46 call.

    ver4-40 R1: `should_stop` / `executor` (HP_TUNE_FAST patience / persistent executor)
    are forwarded to 'Bayesian' ONLY, and only when not None - so every other algorithm,
    and every call that leaves them at None, makes exactly the pre-ver4-40 call.
 Returns whatever that
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
        # FEATURE (multi-fidelity screening): threaded through to
        # search_bayesian()'s own `multi_fidelity=` keyword - see that
        # function's own docstring for the full mechanism.
        kwargs['multi_fidelity'] = bayes_multi_fidelity
        if should_stop is not None:
            kwargs['should_stop'] = should_stop
        if executor is not None:
            kwargs['executor'] = executor
        if async_batch:
            kwargs['async_batch'] = True
        if ship_once:
            kwargs['ship_once'] = True
        if time_stop is not None:
            kwargs['time_stop'] = time_stop
        if isolate_failures:
            kwargs['isolate_failures'] = True
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


def _column_corr(a, b):
    """Pearson r of two prediction columns; NaN when either is too short, non-finite or
    constant (a constant column carries no information, so it can never pass a parity
    gate)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size < 3 or not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return float('nan')
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def _resolve_search_shortcuts(model_name, objective, default_x, train, valid, test,
                              run_model_fn, RESULT_NAME, *, want_stub, want_engine,
                              stub_min, engine_min):
    """ver4-40 R1 (C2/C3): decide, once per (task, model), whether the 1-row test stub and
    the closed-form exact-kernel engine may be used for this model's SEARCH fits.

    Why a check: both shortcuts claim to leave the validation column the search reads
    unchanged (the stub only drops test rows the search never reads; the engine replaces
    MCMC by the closed-form posterior of the same model). The claim is cheap to test at the
    one point every search evaluates anyway, `default_x`: the reference is a normal fit
    (full `test`, search-fidelity parameters from `objective.prepare_params`); the stub
    column is the same fit with ``test.iloc[:1]``; the engine column is
    `kernel_effects.exact_kernel_column()` at the same parameters. A shortcut is kept iff
    corr(reference, shortcut) >= its minimum (`stub_min`, default 0.995; `engine_min`,
    default 0.98 - the same defaults and the same gate pattern as
    `joint_tuning._parity_guards`). A failed fit, a non-finite or constant column, or an
    engine exception turns that shortcut OFF for this model with a NOTE; nothing here is
    ever fatal. Models outside `model_registry.search_test_stub_safe` /
    `exact_kernel_search_capable` are never eligible. No validation rows -> both off.
    I7: only train phenotypes feed the fits; `valid` is read only to compare columns.

    Returns ``{'test_stub': bool, 'engine': 'exact_kernel' | None, 'corr': {...}}``."""
    from model_registry import search_test_stub_safe, exact_kernel_search_capable

    out = {'test_stub': False, 'engine': None, 'corr': {}}
    do_stub = bool(want_stub)
    do_engine = bool(want_engine)
    if do_stub and not search_test_stub_safe(model_name):
        print(f"[hyperparameter_tuning] NOTE: test_stub does not apply to {model_name!r} "
              f"(its search fit does not mask test rows) - it keeps the full test frame.")
        do_stub = False
    if do_engine and not exact_kernel_search_capable(model_name):
        print(f"[hyperparameter_tuning] NOTE: search_engine='exact_kernel' does not apply to "
              f"{model_name!r} (no closed-form engine for it) - its search fits stay real fits.")
        do_engine = False
    if not (do_stub or do_engine):
        return out
    if valid is None or len(valid) == 0:
        print(f"[hyperparameter_tuning] NOTE: {model_name} has no validation rows, so the "
              f"stub / engine parity check cannot run - both are off for this task.")
        return out

    params = objective.prepare_params(default_x)
    try:
        ref = run_model_fn(model_name, train, valid, test, params, RESULT_NAME)
        ref_col = np.asarray(ref.get('predicted_valid'), dtype=float)
    except Exception as exc:
        print(f"[hyperparameter_tuning] NOTE: the reference fit of {model_name} for the "
              f"stub / engine parity check failed ({type(exc).__name__}: {exc}) - both "
              f"shortcuts are OFF for this task.")
        return out
    if do_stub:
        c = float('nan')
        try:
            stub = run_model_fn(model_name, train, valid, test.iloc[:1], params, RESULT_NAME)
            c = _column_corr(ref_col, stub.get('predicted_valid'))
        except Exception as exc:
            print(f"[hyperparameter_tuning] NOTE: the 1-row-test fit of {model_name} failed "
                  f"({type(exc).__name__}: {exc}).")
        out['corr']['stub'] = c
        if np.isfinite(c) and c >= stub_min:
            out['test_stub'] = True
            print(f"[hyperparameter_tuning] stub parity {model_name}: corr(full test, 1-row "
                  f"test)={c:.5f} >= {stub_min:g} - the 1-row test stub is ON.")
        else:
            print(f"[hyperparameter_tuning] NOTE: stub parity {model_name}: corr(full test, "
                  f"1-row test)={c:.5f} < {stub_min:g} (or not computable) - the 1-row test "
                  f"stub is OFF; search fits keep the full test frame.")
    if do_engine:
        c = float('nan')
        t_e = time.perf_counter()
        try:
            from .kernel_effects import exact_kernel_column
            col = exact_kernel_column(model_name, train, valid, test, params)
            c = _column_corr(ref_col, col['predicted_valid'])
        except Exception as exc:
            print(f"[hyperparameter_tuning] NOTE: the exact kernel engine failed on the "
                  f"reference point of {model_name} ({type(exc).__name__}: {exc}) - "
                  f"search_engine='exact_kernel' is OFF.")
        out['corr']['engine'] = c
        if np.isfinite(c) and c >= engine_min:
            out['engine'] = 'exact_kernel'
            print(f"[hyperparameter_tuning] engine parity {model_name}: corr(BGLR column, "
                  f"exact column)={c:.5f} >= {engine_min:g} "
                  f"({(time.perf_counter() - t_e) * 1000:.0f} ms) - the exact kernel engine "
                  f"is ON.")
        else:
            print(f"[hyperparameter_tuning] NOTE: engine parity {model_name}: corr(BGLR "
                  f"column, exact column)={c:.5f} < {engine_min:g} (or not computable) - "
                  f"the exact kernel engine is OFF; search fits stay real fits.")
    return out


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #

def _oof_fast_objective(plan, model_name, base_params, specs, train, valid, test, run_model_fn,
                        RESULT_NAME, reduced_cost_search, oof_folds, search_ctx, default_x,
                        shortcut_eligible, fast=None):
    """ver4-86: build the out-of-fold objective with the per-model cost savings of `plan`
    (``pipeline_utils._SL_OOF_FAST``) switched on where they are safe, and prime fold racing.

    test stub: only for models registered in ``model_registry.search_test_stub_safe``.
    exact engine: only for ``exact_kernel_search_capable`` models, never under BLUE weighting
    (`shortcut_eligible` False), and only when its held-out column on fold 1 at the default point
    correlates >= ``engine_parity_min`` with a real fit there (one extra real fit; any error
    switches the engine off). Racing: the default point is scored on all folds here, first, and
    becomes the bar. One summary line is printed. Returns the objective.

    ver4-87: when `fast` (the resolved HP_TUNE_FAST dict) is set, its ``engine_parity_min``
    replaces the plan's, and the test stub must also pass ITS check: on fold 1 at the default
    point a stub fit's held-out column must correlate >= ``stub_parity_min`` with a full-test
    fit (one extra real fit). Without HP_TUNE_FAST the stub follows the model registry alone, as
    in ver4-86. The time spent here priming racing is stored as ``objective.oof_prime_secs``
    (the caller counts it as one search candidate)."""
    from model_registry import search_test_stub_safe, exact_kernel_search_capable
    base = schema_key_of(model_name)
    stub = bool(plan.get('test_stub')) and bool(search_test_stub_safe(base))
    engine_want = bool(plan.get('exact_engine')) and bool(exact_kernel_search_capable(base))
    engine_min = float((fast or {}).get('engine_parity_min', plan.get('engine_parity_min', 0.98)))
    stub_min = float(fast['stub_parity_min']) if (fast and 'stub_parity_min' in fast) else None
    engine = False
    parity = None
    stub_txt = 'on' if stub else 'off'

    def _corr(a, b):
        a = np.asarray(a, dtype=float).ravel(); b = np.asarray(b, dtype=float).ravel()
        if a.shape != b.shape or a.size <= 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
            return float('nan')
        return float(np.corrcoef(a, b)[0, 1])
    if (stub and stub_min is not None) or (engine_want and shortcut_eligible):
        try:
            plain = make_objective(model_name, base_params, specs, train, valid, test,
                                   run_model_fn, RESULT_NAME, reduced_cost_search,
                                   search_fit=search_ctx, oof_folds=oof_folds)
            params = plain.prepare_params(default_x)
            held = np.asarray(oof_folds[0], dtype=int)
            keep = np.setdiff1d(np.arange(len(train)), held)
            ref = None
            if stub and stub_min is not None:
                full = run_model_fn(model_name, train.iloc[keep], train.iloc[held], test, params,
                                    RESULT_NAME)['predicted_valid']
                stubbed = run_model_fn(model_name, train.iloc[keep], train.iloc[held],
                                       test.iloc[:1], params, RESULT_NAME)['predicted_valid']
                r_s = _corr(full, stubbed)
                stub = bool(np.isfinite(r_s) and r_s >= stub_min)
                stub_txt = (f"parity r={r_s:.4f} (min {stub_min}) -> {'on' if stub else 'off'}")
                ref = stubbed if stub else full
            if engine_want and shortcut_eligible:
                from .kernel_effects import exact_kernel_column
                t_arg = test.iloc[:1] if stub else test
                if ref is None:
                    ref = run_model_fn(model_name, train.iloc[keep], train.iloc[held], t_arg,
                                       params, RESULT_NAME)['predicted_valid']
                ex = exact_kernel_column(base, train.iloc[keep], train.iloc[held], t_arg,
                                         params)['predicted_valid']
                r = _corr(ref, ex)
                engine = bool(np.isfinite(r) and r >= engine_min)
                parity = f"parity r={r:.4f} (min {engine_min}) -> {'on' if engine else 'off'}"
        except Exception as exc:
            if stub and stub_min is not None and stub_txt == 'on':
                stub, stub_txt = False, f"off (check failed: {type(exc).__name__})"
            parity = f"off ({type(exc).__name__}: {exc})" if engine_want else parity
    if engine_want and not shortcut_eligible:
        parity = 'off (BLUE weighting)'
    race = None
    if plan.get('race_margin') is not None:
        race = (float(plan['race_margin']), int(plan.get('race_min_folds', 2)))
    objective = make_objective(model_name, base_params, specs, train, valid, test, run_model_fn,
                               RESULT_NAME, reduced_cost_search, search_fit=search_ctx,
                               oof_folds=oof_folds,
                               oof_options={'test_stub': stub, 'engine': engine, 'race': race})
    bar = None
    objective.oof_prime_secs = None
    if race is not None:
        _tp = time.time()
        bar = objective.prime_oof_race(default_x)
        objective.oof_prime_secs = time.time() - _tp
    print(f"[hyperparameter_tuning] {model_name}: out-of-fold fast mode - test stub "
          f"{stub_txt}, exact engine {parity or 'n/a'}, fold racing "
          + (f"on (bar = default score {bar:.4f} - {race[0]}, after >= {race[1]} folds)"
             if race is not None else "off") + ".")
    return objective


def _final_fit_role():
    """GAT seed-ensemble role (models/gat_stability.py, F2) for the confirmatory fit; a no-op
    context for every other model and whenever the ensemble is off."""
    from models.gat_stability import final_fit_role
    return final_fit_role()


def _tune_model_hyperparameters_impl(model_name, base_params, train, valid, test,
                                run_model_fn, RESULT_NAME, algorithm,
                                budget_kwargs, hparam_specs,
                                reduced_cost_search=True, n_jobs=1,
                                bayes_batch=True, bayes_batch_max=8, bayes_liar='max',
                                parallel_restarts=True, worker_init=None, worker_init_args=(),
                                bayes_domain_reduction='auto', selection_margin=0.0,
                                valid_repeats=1, inner_split_seed_base=0,
                                bayes_multi_fidelity=False, fast=None,
                                shortcut_eligible=True, time_limit_s=None,
                                trial_time_limit_s=None, isolate_failures=False,
                                oof_folds=None, oof_fast=None):
    """Search `algorithm`'s best hyperparameters for `model_name` on the
    validation set, then perform ONE final confirmatory fit at the winning
    hyperparameters with the user's real explainability flags and MCMC
    length restored.

    ver4-85: `oof_folds` (default None = holdout scoring on `valid`, unchanged) scores every
    candidate - the default point included - on OUT-OF-FOLD predictions over `train` instead (see
    `make_objective()`); `valid` may then be empty. It is set by `genomic_prediction` from
    `models.super_learner.sl_tuning_folds()` when ``SUPER_LEARNER['oof_tuning']`` is on. In this
    mode `valid_repeats`, multi-fidelity screening and the HP_TUNE_FAST test-stub / exact-engine
    shortcuts (whose parity gates read `valid`) are off, each with a NOTE; every other setting,
    the default-vs-winner margin and the confirmatory fit are unchanged. Each candidate costs
    ``len(oof_folds)`` fits.

    ver4-86: `oof_fast` (out-of-fold mode only; None = the ver4-85 search) is
    ``pipeline_utils.sl_oof_tuning_plan()``'s dict of cost savings:
    ``freeze_fidelity`` (run-length fields leave the search, so `_cheapen_mcmc` /
    `_cheapen_fidelity` apply the cheap search length to every fold fit; a model left with no
    searchable field gets ONE fit at `base_params`, tuning_source 'no_search_space'),
    ``evals_per_dim`` (Bayesian budget ``min(configured, max(4, 1 + evals_per_dim x d))``),
    ``test_stub`` (1-row test frame for `model_registry.search_test_stub_safe` models),
    ``exact_engine`` / ``engine_parity_min`` (rrBLUP / RKHS fold columns by the closed-form
    engine, kept only when its fold-1 column at the default point correlates >= the minimum with
    a real fit; off under BLUE weighting) and ``race_margin`` / ``race_min_folds`` (fold racing
    against the default point's score, which is then computed first, in this process).

    Guarantees the returned hyperparameters score at least as well
    (within `selection_margin`) on the validation set as the user's own
    base_params (defaults/current values) would - the search is never
    allowed to hand back something worse than simply not tuning at all
    (see the margin-based default-vs-winner comparison below).

    Parameters
    ----------
    time_limit_s : float or None, default None (ver4-61 R3, HP_TUNE[model]['time_limit_min']
        in seconds). Wall-clock budget of the SEARCH phase (the confirmatory fit is not
        counted). Enforced through `parallel_search.time_budget_stop()` composed with the
        patience rule: no round is started that is expected to overrun the budget, so the
        search can exceed it by at most one trial's overrun. 'Bayesian' only (the other
        algorithms have no stopping hook; a NOTE says so). None/0 = off.
    trial_time_limit_s : float or None, default None (ver4-61 R3,
        HP_TUNE[model]['trial_time_limit_min'] in seconds). Per-fit cap for SEARCH fits,
        carried in the search-fit context and read by `GAT_biological_prior_knowledge`
        (training stops at the next epoch boundary once exceeded). The confirmatory fit is
        never capped. None/0 = off.
    isolate_failures : bool, default False (ver4-61 R2). Forwarded to the Bayesian batch path:
        a killed worker loses only the candidates it took down (see
        `_evaluate_batch_isolated()`). Set by the caller from ACCELERATION
        ['trial_memory_guard'].
    fast : dict or None, default None (ver4-40 R1). The resolved `HP_TUNE_FAST` dict
        (`pipeline_utils.resolve_compute_resources()` fills every key). `None` = legacy:
        the code path below is then identical to ver4-39 (I11). Keys: `freeze_fidelity`
        and `search_space` ('all' | 'priority') are applied through
        `filter_param_specs()`; when NOTHING is left to search, no search runs - ONE fit
        at `base_params` is made and returned with `result['_tuning_source'] =
        'no_search_space'` (GBLUP: its only tunable fields are run-length knobs).
        `auto_budget` / `evals_per_dim` shrink a Bayesian budget to
        ``min(user_total, max(4, 1 + evals_per_dim * d))`` candidates for `d` searched
        dimensions (C4). `patience` > 0 stops a stalled Bayesian search (C4).
        `test_stub` / `search_engine` are decided per (task, model) by
        `_resolve_search_shortcuts()` behind parity gates (C2, C3). `persistent_executor`
        reuses one loky pool across batch rounds (C5). When `fast` is not None the
        returned result also carries the private key `'_fast_info'` (search_dims,
        n_candidates, stub, engine, stopped_early) for the hyperparameter.csv row;
        callers pop it.
    shortcut_eligible : bool, default True (ver4-40 R1). False switches `test_stub` and
        `search_engine` off for this call: the caller passes False when BLUE weighting is
        active (`ModelTrialRunner.sample_weight_ctx is not None`), because the runner then
        returns BLUE-weighted validation metrics that the unweighted engine metrics could
        not reproduce (same reason as joint D4).
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
    bayes_multi_fidelity : bool, default False (HP_TUNE_BAYES_MULTI_
        FIDELITY) - forwarded BOTH to `make_objective()`'s own
        `multi_fidelity_screen=` keyword (so the objective it builds
        carries a `.screen` attribute at all) AND to `run_search()` ->
        `search_bayesian()`'s own `multi_fidelity=` keyword (so the batch
        search path actually uses it); irrelevant to every other
        algorithm. See `search_bayesian()`'s own docstring for the full
        mechanism, and `make_objective()`'s for what `.screen` is.
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
    specs = build_param_specs(model_name, hparam_specs, base_params=base_params)
    if not specs:
        raise ValueError(
            f"No tunable fields are marked for {model_name!r} in HPARAM_SPECS "
            f"(add a 'tunable': {{...}} entry to at least one of its fields "
            f"before enabling tuning for this model)."
        )
    # ver4-86: cost savings of out-of-fold tuning (SUPER_LEARNER['oof_tuning_fast']).
    _oof_plan = dict(oof_fast) if (oof_folds is not None and oof_fast) else None
    if _oof_plan and _oof_plan.get('freeze_fidelity'):
        _frozen = filter_param_specs(model_name, specs, hparam_specs, freeze_fidelity=True)
        if len(_frozen) < len(specs):
            print(f"[hyperparameter_tuning] {model_name}: out-of-fold tuning searches without "
                  f"its run-length field(s) "
                  f"{[sp.name for sp in specs if sp not in _frozen]} (kept at the configured "
                  f"value; search fits use the cheaper search length).")
        if not _frozen:
            _t0_nss = time.time()
            print(f"[hyperparameter_tuning] NOTE: {model_name} has only run-length "
                  f"hyperparameters - out-of-fold tuning has nothing to search: ONE fit at the "
                  f"configured hyperparameters replaces it (tuning_source='no_search_space').")
            _final_params = list(base_params)
            _final_result = run_model_fn(model_name, train, valid, test, _final_params,
                                         RESULT_NAME)
            if isinstance(_final_result, dict):
                _final_result['_trial_secs'] = (0, 0.0)
                _final_result['_tuning_source'] = 'no_search_space'
            return _final_params, _final_result, float('nan'), {}, time.time() - _t0_nss
        specs = _frozen

    t0 = time.time()
    if oof_folds is not None:
        _off = [name for name, on in (
            (f'HP_TUNE_VALID_REPEATS={valid_repeats}', valid_repeats and int(valid_repeats) > 1),
            ('HP_TUNE_BAYES_MULTI_FIDELITY', bool(bayes_multi_fidelity)),
            ("HP_TUNE_FAST test_stub / search_engine",
             bool(fast and (fast.get('test_stub') or
                            fast.get('search_engine', 'bglr') == 'exact_kernel')))) if on]
        print(f"[hyperparameter_tuning] {model_name} ({algorithm}): out-of-fold scoring - "
              f"{len(oof_folds)} folds over {len(train)} training rows, each candidate costs "
              f"{len(oof_folds)} fits; score = pooled out-of-fold r - mse/var"
              + (f". Not used in this mode: {', '.join(_off)}." if _off else "."))
        valid_repeats = 1
        bayes_multi_fidelity = False
    # ver4-40 R1 (C1/C6): optional narrowing of the search space. Raising above stays
    # keyed on the UNFILTERED list: "nothing tunable in HPARAM_SPECS" is a configuration
    # error, whereas "everything was filtered away" is a legitimate fast-mode outcome.
    # `fast is None` leaves `specs` untouched (legacy).
    _n_specs_unfiltered = len(specs)
    _fast_stub = _fast_engine = False
    if fast:
        specs = filter_param_specs(
            model_name, specs, hparam_specs,
            freeze_fidelity=bool(fast.get('freeze_fidelity', False)),
            priority_only=(fast.get('search_space', 'all') == 'priority'))
        if fast.get('cap_search_cost'):
            # ver4-46, R2: search-only upper bounds on cost-driving fields (never below the
            # configured value, see cap_search_cost()).
            _before = {sp.name: sp.high for sp in specs}
            specs = cap_search_cost(model_name, specs, hparam_specs, base_params)
            _capped = {sp.name: (_before[sp.name], sp.high) for sp in specs
                       if sp.high != _before[sp.name]}
            if _capped:
                print(f"[hyperparameter_tuning] fast: {model_name} search box capped for cost: "
                      + ', '.join(f"{n} upper bound {a} -> {b}" for n, (a, b) in _capped.items()))
        if not specs:
            print(f"[hyperparameter_tuning] NOTE: HP_TUNE_FAST left {model_name} with no "
                  f"searchable hyperparameter ({_n_specs_unfiltered} tunable field(s), all "
                  f"removed by freeze_fidelity={bool(fast.get('freeze_fidelity'))} / "
                  f"search_space={fast.get('search_space', 'all')!r}) - no search is run: "
                  f"ONE fit at the configured hyperparameters replaces it "
                  f"(tuning_source='no_search_space').")
            _final_params = list(base_params)
            _final_result = run_model_fn(model_name, train, valid, test, _final_params,
                                         RESULT_NAME)
            if isinstance(_final_result, dict):
                _final_result['_trial_secs'] = (0, 0.0)
                _final_result['_tuning_source'] = 'no_search_space'
                _final_result['_fast_info'] = {
                    'search_dims': 0, 'n_candidates': 0, 'stub': False, 'engine': '',
                    'stopped_early': False}
            return _final_params, _final_result, float('nan'), {}, time.time() - t0
        _fast_stub = bool(fast.get('test_stub', False)) and oof_folds is None
        _fast_engine = (fast.get('search_engine', 'bglr') == 'exact_kernel') and oof_folds is None
        if _fast_engine and not fast.get('freeze_fidelity', False):
            print(f"[hyperparameter_tuning] NOTE: search_engine='exact_kernel' needs "
                  f"freeze_fidelity=True (the closed-form engine ignores nIter/burnIn, so "
                  f"they would be flat search dimensions) - the engine stays OFF for "
                  f"{model_name}.")
            _fast_engine = False
        if (_fast_stub or _fast_engine) and not shortcut_eligible:
            print(f"[hyperparameter_tuning] NOTE: test_stub / search_engine are OFF for "
                  f"{model_name} because BLUE weighting is active (the runner scores "
                  f"BLUE-weighted metrics that the shortcut paths cannot reproduce).")
            _fast_stub = _fast_engine = False

    # ver4-46, R1/R3: lean GAT search fits. Only the GAT family reads the search-fit
    # context; for every other model it is a no-op, so the flag is not gated by model here.
    _search_fit = {'lean': True} if (fast and fast.get('gat_lean_fit')) else None
    time_limit_s = _clean_limit(time_limit_s, 'time_limit_min', model_name)
    trial_time_limit_s = _clean_limit(trial_time_limit_s, 'trial_time_limit_min', model_name)
    _es_patience = int((fast or {}).get('gat_search_early_stop', 0) or 0)
    # ver4-61 R3/R4: the per-fit cap and the search early-stop patience travel to the model in
    # the search-fit context, in a SEPARATE dict from the lean one above - `_search_fit`
    # (truthy => `warm_search_caches`) keeps its ver4-46 meaning exactly. Absent limits leave
    # the context byte-identical to ver4-60 (I11).
    _search_ctx = _search_fit
    if trial_time_limit_s or _es_patience:
        _search_ctx = dict(_search_fit or {})
        if trial_time_limit_s:
            _search_ctx['trial_time_limit_s'] = float(trial_time_limit_s)
        if _es_patience:
            _search_ctx['early_stop_patience'] = int(_es_patience)
    if time_limit_s or trial_time_limit_s or _es_patience:
        print(f"[hyperparameter_tuning] limits {model_name}: search time budget="
              f"{'%.1f min' % (time_limit_s / 60.0) if time_limit_s else 'off'}, per-fit "
              f"cap={'%.1f min' % (trial_time_limit_s / 60.0) if trial_time_limit_s else 'off'}"
              f", search early-stop patience={_es_patience or 'off'}.")
    if (trial_time_limit_s or _es_patience) and schema_key_of(model_name) \
            != 'GAT_biological_prior_knowledge':
        print(f"[hyperparameter_tuning] NOTE: the per-fit time cap and the search early-stop "
              f"patience are read only by GAT_biological_prior_knowledge; {model_name} runs "
              f"its search fits unchanged.")
    if time_limit_s and algorithm != 'Bayesian':
        print(f"[hyperparameter_tuning] NOTE: time_limit_min applies to the 'Bayesian' "
              f"algorithm only; {model_name} runs {algorithm!r} with its configured budget.")
    objective = make_objective(model_name, base_params, specs, train, valid, test,
                                run_model_fn, RESULT_NAME, reduced_cost_search,
                                valid_repeats=valid_repeats,
                                inner_split_seed_base=inner_split_seed_base,
                                multi_fidelity_screen=bayes_multi_fidelity,
                                search_fit=_search_ctx, oof_folds=oof_folds)

    # Computed BEFORE the search so it can be handed to run_search() below:
    # search_bayesian() (and, for the same reason, search_random()/the
    # Nelder-Mead & Powell local searches) explicitly evaluates this point
    # itself - see each function's own comment for why seeding the search
    # with a known-reasonable anchor makes actually beating it more likely,
    # not just possible.
    default_x = _encode_from_params(specs, base_params)
    # ver4-87: the search time budget starts HERE, so the default point scored up front for fold
    # racing (and the parity fits) count against it.
    _t_search0 = time.monotonic()
    _oof_prime_secs = None
    if _oof_plan:
        objective = _oof_fast_objective(
            _oof_plan, model_name, base_params, specs, train, valid, test, run_model_fn,
            RESULT_NAME, reduced_cost_search, oof_folds, _search_ctx, default_x,
            shortcut_eligible, fast=fast)
        _oof_prime_secs = getattr(objective, 'oof_prime_secs', None)
    _should_stop = None
    _executor = None
    _fast_stub_on, _fast_engine_on = False, None
    if fast:
        # ver4-40 R1: shortcut parity gates (one reference fit + one stub fit + one engine
        # call per (task, model)); a shortcut that passes is built into a NEW objective,
        # the plain one above only supplied `.prepare_params`.
        if _fast_stub or _fast_engine:
            _sc = _resolve_search_shortcuts(
                model_name, objective, default_x, train, valid, test, run_model_fn,
                RESULT_NAME, want_stub=_fast_stub, want_engine=_fast_engine,
                stub_min=float(fast.get('stub_parity_min', 0.995)),
                engine_min=float(fast.get('engine_parity_min', 0.98)))
            _fast_stub_on = bool(_sc['test_stub'])
            _fast_engine_on = _sc['engine']
            if _fast_stub_on or _fast_engine_on:
                objective = make_objective(
                    model_name, base_params, specs, train, valid, test, run_model_fn,
                    RESULT_NAME, reduced_cost_search, valid_repeats=valid_repeats,
                    inner_split_seed_base=inner_split_seed_base,
                    multi_fidelity_screen=bayes_multi_fidelity, shortcuts=_sc,
                    search_fit=_search_ctx, oof_folds=oof_folds)
        budget_kwargs = dict(budget_kwargs or {})
        if algorithm == 'Bayesian':
            if fast.get('auto_budget'):
                _n0 = int(budget_kwargs.get('n_iter', 30))
                _i0 = int(budget_kwargs.get('init_points', 5))
                _user_total = 1 + _i0 + _n0
                _total = min(_user_total,
                             max(4, 1 + int(fast.get('evals_per_dim', 6)) * len(specs)))
                budget_kwargs['n_iter'] = max(0, _total - 1 - _i0)
                print(f"[hyperparameter_tuning] fast: {model_name} auto_budget d={len(specs)} "
                      f"-> {1 + _i0 + budget_kwargs['n_iter']} candidates "
                      f"(configured {_user_total}).")
            from .parallel_search import patience_stop
            _should_stop = patience_stop(int(fast.get('patience', 0) or 0))
            if fast.get('persistent_executor') and n_jobs and n_jobs > 1:
                try:
                    _executor = _loky_executor(n_jobs, worker_init, worker_init_args)
                except Exception as exc:
                    print(f"[hyperparameter_tuning] NOTE: could not create the persistent "
                          f"trial executor ({type(exc).__name__}: {exc}) - using a fresh "
                          f"pool per round.")
                    _executor = None
        elif fast.get('auto_budget') or fast.get('patience') or fast.get('persistent_executor'):
            print(f"[hyperparameter_tuning] NOTE: auto_budget / patience / persistent_executor "
                  f"apply to the 'Bayesian' algorithm only; {model_name} runs "
                  f"{algorithm!r} with its configured budget.")
        print(f"[hyperparameter_tuning] fast: {model_name} dims={len(specs)}/"
              f"{_n_specs_unfiltered} stub={'on' if _fast_stub_on else 'off'} "
              f"engine={_fast_engine_on or 'off'} patience="
              f"{int(fast.get('patience', 0) or 0)} executor={'on' if _executor else 'off'} "
              f"lean={'on' if fast.get('gat_lean_fit') else 'off'} "
              f"async={'on' if fast.get('async_batch') else 'off'} "
              f"ship_once={'on' if fast.get('ship_once') else 'off'}")
    if _search_fit:
        # ver4-46, R3: compute, once and here, whatever every candidate would otherwise
        # recompute (GAT_prior_knowledge's topology forest). The confirmatory fit below
        # reuses it (the wrapper keeps the cache enabled for the whole call).
        try:
            from models.gat_search import warm_search_caches
            _warmed = warm_search_caches(
                schema_key_of(model_name), base_params, train,
                getattr(run_model_fn, 'sample_weight_ctx', None))
            if _warmed:
                print(f"[hyperparameter_tuning] fast: {model_name} search caches warmed: "
                      f"{sorted(_warmed)}")
        except Exception as exc:  # noqa: BLE001 - torch/PyG missing or any failure: no warm-up
            print(f"[hyperparameter_tuning] NOTE: search-cache warm-up skipped "
                  f"({type(exc).__name__}: {exc}).")
    if _oof_plan and algorithm == 'Bayesian' and _oof_plan.get('evals_per_dim'):
        # ver4-86: out-of-fold scores are far less noisy than one validation split, so a
        # dimension-sized budget finds the same optimum (ice_sim26 CV36 vs CV12, ice_sim28).
        budget_kwargs = dict(budget_kwargs or {})
        _n0 = int(budget_kwargs.get('n_iter', 30))
        _i0 = int(budget_kwargs.get('init_points', 5))
        _user_total = 1 + _i0 + _n0
        _total = min(_user_total, max(4, 1 + int(_oof_plan['evals_per_dim']) * len(specs)))
        if _total < _user_total:
            budget_kwargs['n_iter'] = max(0, _total - 1 - _i0)
            print(f"[hyperparameter_tuning] {model_name}: out-of-fold Bayesian budget "
                  f"{1 + _i0 + budget_kwargs['n_iter']} candidates for {len(specs)} searched "
                  f"dimension(s) (configured {_user_total}).")
    _RACE_STATS['stopped'] = 0
    _RACE_STATS['folds_saved'] = 0
    _TRIAL_STATS['trials'] = 0
    _TRIAL_STATS['secs'] = 0.0
    if _oof_prime_secs is not None:
        # ver4-87: the primed default is one scored candidate - it also gives the time rule a
        # per-candidate cost from the start.
        _TRIAL_STATS['trials'] = 1
        _TRIAL_STATS['secs'] = float(_oof_prime_secs)
    _time_stop = None
    if time_limit_s and algorithm == 'Bayesian':
        from .parallel_search import time_budget_stop
        _time_stop = time_budget_stop(time_limit_s,
                                      _t_search0 if oof_folds is not None else time.monotonic(),
                                      _TRIAL_STATS)
    _search_result = run_search(
        algorithm, objective, specs, budget_kwargs, default_x=default_x, n_jobs=n_jobs,
        worker_init=worker_init, worker_init_args=worker_init_args,
        bayes_batch=bayes_batch, bayes_batch_max=bayes_batch_max, bayes_liar=bayes_liar,
        parallel_restarts=parallel_restarts, bayes_domain_reduction=bayes_domain_reduction,
        bayes_multi_fidelity=bayes_multi_fidelity,
        should_stop=_should_stop, executor=_executor,
        async_batch=bool(fast and fast.get('async_batch')),
        ship_once=bool(fast and fast.get('ship_once')),
        time_stop=_time_stop, isolate_failures=bool(isolate_failures),
    )
    # ver4-6 R4.4: run_search() returns a 4-tuple as of this update -
    # unpacked DEFENSIVELY (`res[:3]` + `res[3] if len(res) > 3 else None`)
    # so a 3-tuple-returning caller/algorithm somewhere else in the tree
    # can never break this function; it just means "no default score was
    # supplied", handled identically to every pre-ver4-6 call.
    best_x, best_score, _trials = _search_result[:3]
    default_score = _search_result[3] if len(_search_result) > 3 else None
    if _oof_plan and _oof_plan.get('race_margin') is not None and _RACE_STATS['stopped']:
        print(f"[hyperparameter_tuning] {model_name}: fold racing stopped "
              f"{_RACE_STATS['stopped']} candidate(s) early, {_RACE_STATS['folds_saved']} fold "
              f"fit(s) saved (counted in this process only).")
    # ver4-35, R1.a: candidates scored through _evaluate_batch() and the
    # seconds their fits took (summed over workers). Read now, before the
    # 1-SE / confirmatory fits below can add anything.
    _search_trials = (int(_TRIAL_STATS['trials']), float(_TRIAL_STATS['secs']))

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

    # GAT seed-robust winner selection (models/gat_stability.py, F3). Every GAT fit is a random
    # draw; the search's best score is therefore biased upwards and its ranking partly noise. The
    # top candidates and the default are re-scored under extra seed draws and the winner is the best
    # MEAN; the paired SE of (winner - default) over the draws raises the margin below.
    _seed_se_margin = 0.0
    try:
        from models.gat_stability import (gat_stability_settings, is_gat_model,
                                          top_distinct, select_by_seed_mean)
        _gcfg = gat_stability_settings() if is_gat_model(model_name) else None
    except Exception:
        _gcfg = None
    if (_gcfg and _gcfg['seed'] is not None and _gcfg['rescore_top'] > 0
            and _gcfg['rescore_seeds'] > 0):
        if not hasattr(objective, 'rescore'):
            print(f"[hyperparameter_tuning] NOTE: {model_name}: seed-robust re-scoring is not "
                  f"available for this search mode; the search's own winner is used.")
        else:
            try:
                _cands = top_distinct(_trials, len(specs), _gcfg['rescore_top'])
                _t_rs = time.monotonic()
                _sel = select_by_seed_mean(
                    _cands, (np.asarray(default_x, dtype=float), float(default_score)),
                    objective.rescore, _gcfg['rescore_seeds'])
            except Exception as exc:
                _sel = None
                print(f"[hyperparameter_tuning] NOTE: {model_name}: seed-robust re-scoring "
                      f"failed ({type(exc).__name__}: {exc}); the search's own winner is used.")
            if _sel is not None:
                _n_draws = int(_gcfg['rescore_seeds']) + 1
                print(f"[hyperparameter_tuning] {model_name}: re-scored the top {len(_cands)} "
                      f"candidate(s) + default over {_n_draws} seed draws in "
                      f"{time.monotonic() - _t_rs:.1f}s; search best {float(best_score):.4f} -> "
                      f"seed-mean {_sel['mean']:.4f} (default seed-mean "
                      f"{_sel['default_mean']:.4f}, paired SE {_sel['paired_se']:.4f}).")
                best_x, best_score = _sel['x'], _sel['mean']
                default_score = _sel['default_mean']
                _seed_se_margin = _sel['paired_se']

    # Bug fix (Bayesian-tuned RF/SVR/KNN also underperforming their own
    # untuned defaults - Requirement.md follow-up to the MCMC-fidelity fix
    # above): a FIXED selection_margin protects against sampling noise
    # equally for every candidate, but the candidates most likely to WIN
    # on noise alone are exactly the high-variance ones (small k for KNN,
    # high C / small epsilon for SVR, an unconstrained tree for RF, ...) -
    # see make_objective()'s own `_evaluate_detailed()` docstring for the
    # full argument. Applying the textbook "one-standard-error rule"
    # (scikit-learn's GridSearchCV/glmnet's own `lambda.1se` use the same
    # idea) on top of the fixed margin: the winner must beat the default
    # by more than the estimated standard error of that DIFFERENCE, not
    # just by `selection_margin` - so a high-variance winner is held to a
    # correspondingly higher bar, without changing anything for a
    # low-variance one.
    #
    # Only computable when `valid_repeats > 1` gave each candidate more
    # than one inner-fold score to estimate a standard error FROM - with
    # `valid_repeats=1` (a single point score, no resampling) there is
    # nothing to estimate, and this reproduces the plain selection_margin
    # behaviour exactly (byte-for-byte pre-existing behaviour for that
    # case). Two extra, bounded evaluations (one for `default_x`, one for
    # `best_x`, each `valid_repeats` model fits) - paid once, after the
    # search has already finished, never inside the search loop itself.
    se_margin = 0.0
    if valid_repeats and valid_repeats > 1 and hasattr(objective, 'evaluate_with_se'):
        _default_mean, se_default, _n_default = objective.evaluate_with_se(default_x)
        _best_mean, se_best, _n_best = objective.evaluate_with_se(best_x)
        # Standard error of an independent difference of two means.
        se_margin = math.sqrt(se_default ** 2 + se_best ** 2)
        if se_margin > selection_margin:
            print(f"[hyperparameter_tuning] NOTE: {model_name} 1-SE rule raised the "
                  f"required winning margin from {max(0.0, selection_margin):.4f} to "
                  f"{se_margin:.4f} (se_default={se_default:.4f}, se_best={se_best:.4f}) - "
                  f"the search's own winning candidate carries enough score uncertainty "
                  f"across the {valid_repeats} inner resamples that it must beat the "
                  f"defaults by more than that to be trusted as a genuine improvement "
                  f"rather than sampling noise.")
    elif (_oof_plan and _oof_plan.get('one_se') and hasattr(objective, 'oof_fold_scores')
          and not np.array_equal(np.asarray(best_x), np.asarray(default_x))):
        # ver4-87: out-of-fold counterpart of the 1-SE rule above, PAIRED over the folds (the
        # winner and the default are scored on the same folds, so the SE of their fold-wise
        # difference is the right one; ice_sim29). Folds with a degenerate score are skipped.
        _fd = objective.oof_fold_scores(default_x)
        _fb = objective.oof_fold_scores(best_x)
        if _fd is not None and _fb is not None:
            _d = np.asarray(_fb, dtype=float) - np.asarray(_fd, dtype=float)
            _d = _d[np.isfinite(_d)]
            if _d.size >= 2:
                se_margin = float(np.std(_d, ddof=1) / math.sqrt(_d.size))
                if se_margin > selection_margin:
                    print(f"[hyperparameter_tuning] NOTE: {model_name} paired 1-SE rule raised "
                          f"the required winning margin from {max(0.0, selection_margin):.4f} "
                          f"to {se_margin:.4f} (SE of the winner-minus-default difference over "
                          f"{_d.size} folds).")

    se_margin = max(se_margin, _seed_se_margin)

    # Update ID ver4-6, R3.2b: MARGIN comparison, not a bare `>=`.
    # Comparing one noisy default draw against the MAXIMUM of many search
    # draws is a textbook winner's-curse comparison (see this function's
    # own `selection_margin` docstring above) - requiring the winner to
    # beat the default by at least `selection_margin` makes the guarantee
    # meaningfully harder to satisfy by sampling luck alone.
    # `selection_margin=0.0` (this function's own default) reproduces the
    # exact pre-ver4-6 `>=` comparison byte-for-byte. `se_margin` (above)
    # can only ever raise this bar further, never lower it.
    margin = max(0.0, selection_margin, se_margin)
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
        with _final_fit_role():
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
            with _final_fit_role():
                final_result = run_model_fn(model_name, train, valid, test, final_params,
                                            RESULT_NAME)
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
    if isinstance(final_result, dict):
        # ver4-35, R1.a: ``(candidates, summed worker seconds)`` - the caller
        # logs it as ``trials:<model>=n/secs`` in the task's [timing] line.
        # An extra key is harmless: every consumer reads the result by name.
        final_result['_trial_secs'] = _search_trials
        if fast:
            # ver4-40 R1: columns of the hyperparameter.csv row (popped by the caller).
            _planned = (1 + int((budget_kwargs or {}).get('init_points', 5))
                        + int((budget_kwargs or {}).get('n_iter', 30)))
            final_result['_fast_info'] = {
                'search_dims': len(specs), 'n_candidates': len(_trials),
                'stub': _fast_stub_on, 'engine': _fast_engine_on or '',
                'stopped_early': bool(_should_stop is not None and len(_trials) < _planned)}
    return final_params, final_result, best_score, tuned_values, elapsed


def tune_model_hyperparameters(*args, **kwargs):
    """Public entry point - see `_tune_model_hyperparameters_impl` for the full contract
    (every argument and the 5-tuple return are unchanged).

    ver4-46, R3: when `fast` (HP_TUNE_FAST) asks for the lean GAT path, the whole call - the
    warm-up, the search (in this process) and the confirmatory fit - runs with the
    GAT_prior_knowledge topology cache enabled, so the forest that decides the graph topology is
    computed once per task instead of once per candidate. Without it this is a plain call."""
    fast = kwargs.get('fast')
    if fast and fast.get('gat_lean_fit'):
        try:
            from models.gat_search import topology_cache_context
        except Exception:  # torch / PyG not importable here: nothing to cache
            return _tune_model_hyperparameters_impl(*args, **kwargs)
        with topology_cache_context(True):
            return _tune_model_hyperparameters_impl(*args, **kwargs)
    return _tune_model_hyperparameters_impl(*args, **kwargs)


tune_model_hyperparameters.__doc__ = (tune_model_hyperparameters.__doc__ or '') + (
    _tune_model_hyperparameters_impl.__doc__ or '')


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
