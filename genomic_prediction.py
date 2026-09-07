import pandas as pd
import numpy as np
import os
import time
import traceback
# Update ID ver4-6, R3.2a: used ONCE, to build HPARAMETERS_BASELINE - a
# frozen, never-mutated snapshot of the user's own configured HPARAMETERS,
# taken at the top of GP() (see the MODEL_RUN construction below). stdlib,
# no new dependency (I12).
import copy
from datetime import datetime
from itertools import product
from typing import Optional, Sequence
from sklearn.model_selection import train_test_split

# Absolute path to the directory this module itself lives in, resolved once
# at import time. Used to locate co-located resources (the bundled R model
# sources under models/) by an absolute path, independent of the caller's
# current working directory - see the r_source() calls inside GP() for why
# this matters (headless HPC array jobs cannot be relied on to start with
# cwd == the EasiGP install directory).
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_R_MODELS_DIR = os.path.join(_MODULE_DIR, 'models')

from models.RF import *
from models.ExtraTrees import *
from models.GBDT import *
from models.XGBoost import *
from models.EBM import *
from models.SVR import *
from models.KNN import *
from models.MLP import *
from models.GAT_infinitesimal_node_level import *
from models.GAT_infinitesimal import *
from models.GAT_fully_connected import *
from models.GAT_prior_knowledge import *
from models.GAT_biological_prior_knowledge import *
from models.ensemble import *

from Preprocess.LD_pruning import *
from Preprocess.LD_decay_plot import (
    compute_ld_decay_data, save_ld_decay_data, plot_ld_decay,
    resolve_snp_info_for_decay, sanitize_for_filename, unique_path, write_keep_log_marker,
    WINDOW_UNITS as LD_DECAY_WINDOW_UNITS, DEFAULT_MAX_DISTANCE as LD_DECAY_DEFAULT_MAX_DISTANCE,
    DEFAULT_BIN_WIDTH as LD_DECAY_DEFAULT_BIN_WIDTH,
)
from Preprocess.RF_marker_filtering import RF_marker_filtering
from Preprocess.data_driven_prior_network import (
    select_markers_for_data_driven_network, ensure_bio_prior_merge_cache,
)
from Preprocess.gene_network_prior import (
    phenotype_matches_network_metadata, load_network_json, extract_candidate_genes, build_gene_list,
)
from Preprocess.plink_io import (
    validate_plink_fileset, read_bim_marker_info, read_fam_iids, plink_to_genotype_df, ld_prune_plink_native,
    ld_prune_marker_list_native,
)
# _map_genes_to_markers is a "private" (leading-underscore) helper inside the
# model file, reused here deliberately - it's the exact same SNP<->gene
# interval-overlap logic GAT_biological_prior_knowledge.py itself uses, and
# reusing it (rather than re-implementing the same logic a second time) is
# what lets this module determine "which markers fall inside a gene window"
# up front, to --extract only those from a PLINK fileset, while guaranteeing
# it's byte-for-byte the same computation the model will redo internally.
from models.GAT_biological_prior_knowledge import _map_genes_to_markers

from models.Linear_transformation import *
from models.Nelder_Mead import *
from models.Bayesian_optimisation import *
from models.Analytic_least_squares import *
from models.interaction_extraction import weighted_ensemble_interactions

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

from hparam_specs import HPARAM_SPECS
from models.hyperparameter_tuning import (
    tune_model_hyperparameters, expand_model_list, ensemble_groups, base_of, algo_of,
    schema_key_of,
)
from model_registry import (
    emits_interactions, has_no_interaction_capability, no_interaction_capability_reason,
    is_available, optional_dependency_for, interaction_method_default,
)
import checkpoint_utils as _ckpt
from pipeline_utils import (
    unify_columns_by_position, resolve_compute_resources, set_active_compute_resources,
    apply_r_blas_threads, result_dir_path, cleanup_bglr_output,
    r_list_get, safe_regression_metrics, nested_safe_n_jobs, configure_r_environment,
    init_rpy2_conversion, install_gpu_semaphore, get_gpu_semaphore,
)
from models.interaction_extraction import (
    surrogate_tree_interactions, surrogate_h_statistic_interactions, shortlist_markers, top_select,
)


def _is_bio_prior_model(name):
    """True for 'GAT_biological_prior_knowledge' itself, or any numbered
    instance of it ('GAT_biological_prior_knowledge_1',
    '..._2', ...) - see MODEL's docstring-equivalent note in GP() for why
    there can be more than one: each instance is an independent model with
    its own HPARAMETERS[name] entry (its own network JSON / gene-location
    CSV / etc.), letting a user compare several FlashP samples (or several
    hand-uploaded networks) for the same phenotype(s) as separate models in
    one run, rather than only ever having exactly one
    GAT_biological_prior_knowledge in a given MODEL list.
    """
    return name == 'GAT_biological_prior_knowledge' or name.startswith('GAT_biological_prior_knowledge_')


def _bio_prior_model_names(model_list):
    """All bio-prior instance names actually present in `model_list`, in
    the order they appear."""
    return [m for m in model_list if _is_bio_prior_model(m)]


# ---------------------------------------------------------------------------
# Disk-quota remediation (2026-08-22).
#
# rrBLUP/GBLUP/BayesB/RKHS's BGLR() MCMC engine writes several small
# "saveAt"-prefixed trace files (mu.dat, varE.dat, ETA_<component>_*.dat)
# to Result/<RESULT_NAME>/BGLR_output/ on EVERY call - not once per task,
# but once per Bayesian hyperparameter-tuning TRIAL (~20-30 trials per
# model per task inside tune_model_hyperparameters(), PLUS once more for
# the final confirmatory fit below) - and never cleans them up. Only the
# final, best-scoring trial's result is ever used; every trial's trace
# files are written and then never read again by anything in this
# codebase. Across 4 models x ~20-30 tuning trials x many tasks x many
# batches, this accumulates into tens of thousands of small files, which
# exhausts an INODE quota on quota-managed HPC filesystems (confirmed:
# EasiGP_Cotton_step1_0.error, batch 90 - OSError(122, 'Disk quota
# exceeded') from exactly this cause).
#
# `RESULT_NAME` here is whatever was passed into rrBLUP()/GBLUP()/
# BayesB()/RKHS() for THIS call - for a Sequential run that's the plain
# RESULT_NAME; for an intra-batch-parallel task it's already the
# per-batch/per-task-isolated value the caller (intra_batch_parallel.py)
# constructed, so `result_dir_path(RESULT_NAME)` below always resolves to
# the SAME Result/... directory this call's own CSV outputs
# (checkpoint_utils.py) are landing in - exactly the directory BGLR()
# itself was just told (via its own saveAt logic) to write into, so this
# can never touch another task's/batch's files.
#
# See pipeline_utils.py's cleanup_bglr_output() docstring for the full
# writeup and pipeline_utils.py's own unit tests (test_bglr_cleanup.py)
# for independent verification. Best-effort and never raises - a courtesy
# cleanup pass that this pipeline's correctness never depends on.
# ---------------------------------------------------------------------------
def _cleanup_bglr_output_for(result_name):
    """Purge this task's BGLR MCMC trace files after an R/BGLR model call
    (rrBLUP/GBLUP/BayesB/RKHS) returns - called from both _call_model()
    (hyperparameter-tuning trials) and the main per-task dispatch loop's
    final confirmatory fit below. See module-level note above."""
    cleanup_bglr_output(os.path.join(result_dir_path(result_name), 'BGLR_output'))


# ---------------------------------------------------------------------------
# ver4-4 R3.d (blueprint §2.3.2): the Shapley row-range fan-out worker.
#
# Deliberately MODULE-LEVEL, not nested inside GP() (unlike almost every
# other per-task helper in this file) - and deliberately takes no rpy2/R
# object as an argument, ever. This is the one piece the ver4-4 Stage 5
# hand-off note (§6.3/§8) flagged as needing a live rpy2 session to get
# right and left unfinished; the actual blocking design question, worked
# out here without needing one, is this: an embedded rpy2 R session
# (robjects.globalenv[...], the bound GBLUP/RKHS function objects GP()
# creates near its own top) is NOT something that can be hooked to a
# separate OS process - it wraps a live SEXP proxy tied to the specific R
# engine instance that created it. R3.d's own design record already rules
# out forking an embedded rpy2 session for exactly this reason (BGLR's
# saveAt counter is a fork-unsafe `<<-` closure); the identical
# unsafety applies to handing an ALREADY-BOUND R function object to a
# *different*, freshly-spawned process (joblib's default 'loky' backend,
# required here anyway per invariant I10) rather than forking the
# process that bound it. The correct fix is not to try - this function
# receives only ordinarily-picklable arguments (dataframes, a params
# list, plain strings/ints) and bootstraps its OWN, completely
# independent embedded R session from scratch, inside whichever fresh
# subprocess joblib hands it to - the exact same R_HOME / rpy2 / source()
# sequence GP() itself runs once per call (see GP()'s own R import block),
# scoped here to just the one .R file this call needs.
# ---------------------------------------------------------------------------
def _gblup_rkhs_shapley_row_worker(model_name, r_path, train, valid, test, params,
                                    result_name, k_precomputed, row_offset, row_count):
    """PROCESS-isolated worker for one row-range slice of a GBLUP/RKHS
    Shapley computation - dispatched via joblib.Parallel (process
    backend) from ``_run_gblup_or_rkhs()`` inside ``GP()``; never called
    directly by application code.

    Parameters
    ----------
    model_name : 'GBLUP' or 'RKHS' - selects which .R file to source and
        which params index pair (shap_row_offset/shap_row_count) to set.
    r_path : the run's own R_PATH (GP()'s own parameter, forwarded
        explicitly since this worker has no access to GP()'s closure) -
        exported as R_HOME before rpy2 is imported, mirroring GP()'s own
        R-environment bootstrap exactly.
    train, valid, test : the SAME three DataFrames every worker receives
        (task-level pools are already frozen and identical for every
        worker of one task - only the Shapley row RANGE differs between
        them).
    params : this model's full positional HPARAMETERS list, with the
        'all' -> test.shape[0] substitution (Shapley_num) already applied
        by the caller - copied and overwritten with THIS worker's own
        row_offset/row_count before the R call (never mutates the
        caller's own list).
    result_name : forwarded to the R function unchanged (BGLR_output
        directory naming) - already includes the run's own RESULT_NAME;
        each worker's own PID-based run_id (GBLUP.R's/RKHS.R's own
        next_save_prefix()) keeps concurrent workers' BGLR trace files
        from colliding regardless.
    k_precomputed : the SAME precomputed kernel/distance matrix (or
        None) every worker receives - see ``_build_grm()``'s own
        docstring; recomputing it per worker would be wasteful (it does
        not depend on the Shapley row range at all) and risks a subtle
        numerical divergence between workers, so the caller builds it
        exactly ONCE and broadcasts the same array to every worker.
    row_offset, row_count : THIS worker's own slice of the
        [0, len) Shapley row range - see GBLUP.R's/RKHS.R's own
        shap_row_offset/shap_row_count docstrings for the exact
        semantics (0-based offset, contiguous count).

    Returns
    -------
    A PLAIN Python dict with keys 'r_pearson', 'r_MSE', 'r_effect',
    'r_y_predicted', 'r_y_predicted_valid', 'r_y_predicted_train' -
    extracted via ``r_list_get()`` from this worker's OWN raw R result
    *before* returning, specifically so nothing rpy2-specific ever has to
    cross back over the process boundary (only plain floats/numpy
    arrays/pandas DataFrames, which pickle the ordinary way). This is the
    SAME key set a raw ``GBLUP(...)``/``RKHS(...)`` call returns, so
    ``_run_gblup_or_rkhs()`` (and, transitively, every existing
    ``r_list_get(result, 'r_pearson')[0]``-style call site downstream) can
    treat one worker's own result exactly like an unsplit call's result.

    This worker's own R call runs GBLUP()'s/RKHS()'s ENTIRE body -
    including the main BGLR MCMC fit, not merely the Shapley loop - since
    there is no safe way to share an already-fitted `fm` object across
    the process boundary either. BGLR does not fix a global RNG seed, so
    every worker's own main fit is an independent MCMC draw exactly like
    any two ordinary (non-fanned-out) calls to this same model already
    are relative to each other - the fan-out introduces no NEW source of
    run-to-run stochasticity, only N redundant (but now wall-clock-
    parallel) copies of a cost that is small relative to the Shapley
    loop's own O(2*M*Shapley_num) dominant cost (see GBLUP.R's/RKHS.R's
    own module docstrings) whenever fan-out is actually worth triggering.
    """
    if r_path:
        os.environ['R_HOME'] = r_path
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()

    r_source = robjects.r['source']
    r_filename = 'GBLUP.R' if model_name == 'GBLUP' else 'RKHS.R'
    r_source(os.path.join(_R_MODELS_DIR, r_filename))
    r_fn = robjects.globalenv[model_name]

    offset_idx, count_idx = (7, 8) if model_name == 'GBLUP' else (8, 9)
    worker_params = list(params)
    worker_params[offset_idx] = int(row_offset)
    worker_params[count_idx] = int(row_count)

    r_kwargs = {}
    if k_precomputed is not None:
        r_kwargs['K_precomputed'] = k_precomputed

    raw_result = r_fn(train, valid, test, worker_params, result_name, **r_kwargs)
    return dict(
        r_pearson=r_list_get(raw_result, 'r_pearson'),
        r_MSE=r_list_get(raw_result, 'r_MSE'),
        r_effect=r_list_get(raw_result, 'r_effect'),
        r_y_predicted=r_list_get(raw_result, 'r_y_predicted'),
        r_y_predicted_valid=r_list_get(raw_result, 'r_y_predicted_valid'),
        r_y_predicted_train=r_list_get(raw_result, 'r_y_predicted_train'),
    )


# ---------------------------------------------------------------------------
# Requirements.md item 1 fix (production crash on real HPC hardware):
# ``TypeError("unsupported operand type(s) for +: 'OrdDict' and
# 'DataFrame'")`` inside ``_run_gblup_or_rkhs()``'s worker-reduction step
# below, first seen running RKHS's Shapley fan-out across multiple CPUs
# (``N_JOBS>1``).
#
# Root cause: each Shapley row-range fan-out worker bootstraps its OWN,
# completely independent embedded R/rpy2 session (see
# ``_gblup_rkhs_shapley_row_worker()``'s own docstring above for why - a
# live rpy2 session cannot cross a process boundary). rpy2's ACTIVE
# conversion context is what decides whether a given R value comes back
# as a pandas ``DataFrame`` or as a mapping (``rpy2.rlike.container.
# OrdDict`` / plain ``dict``) - see ``pipeline_utils.r_list_get()``'s own
# docstring, "Problem 2 - the accessor". This was previously ASSUMED to
# be a property of the R *return shape* (rrBLUP/GBLUP/BayesB/RKHS vs. an
# unconverted ``ListVector``), i.e. constant across every call in a
# given process - true for every call site that was actually exercised
# against a real R session before this fix. Running for real on HPC
# hardware (ver4-4 Stage 5/6's own §8 disclosed this had NEVER been
# tested against a real rpy2/R/BGLR session) showed this assumption
# doesn't hold across SEPARATE worker processes: two independent rpy2
# sessions - same R code, same ``effect <- data.frame(...)`` shape on
# the R side - resolved ``r_effect`` to two DIFFERENT Python types.
# Rather than chase the exact rpy2-internal reason one particular
# process's conversion context diverged (not something this codebase
# controls), ``_coerce_effect_values()`` below coerces AFTER the fact:
# it accepts any of the shapes ``r_list_get()`` can plausibly hand back
# for a "one effect value per marker" result and always returns a flat
# numeric numpy array, so the actual summation in
# ``_run_gblup_or_rkhs()`` is always plain numpy addition - never
# dependent on two arbitrary Python objects' own ``__add__`` agreeing.
# ---------------------------------------------------------------------------
def _coerce_effect_values(effect_obj, n_expected):
    """Normalise ONE Shapley fan-out worker's already-``r_list_get()``-
    extracted ``r_effect`` value into a flat, numeric, length-
    ``n_expected`` numpy array, regardless of which concrete Python type
    rpy2's active conversion context happened to hand back for it this
    call (pandas ``DataFrame``/``Series``, ``OrdDict``/plain ``dict``,
    or anything ``numpy.asarray()`` can flatten - a plain list, a numpy
    array, or an unconverted rpy2 vector). See the module-level note
    immediately above for why this is needed.

    A 0-row DataFrame (the "no explainable rows in this worker's own
    range" edge case - never actually reachable today since
    ``_run_gblup_or_rkhs()`` only creates non-empty row-bounds, but
    handled defensively here rather than assumed) coerces to an
    all-zero vector, matching what RKHS.R's/GBLUP.R's own "0 explained
    rows" branch already reduces to internally (see RKHS.R's
    ``setNames(rep(0, length(top_positions)), ...)`` branch).

    Raises ``ValueError`` (never silently truncates/pads) if the
    resolved vector's length doesn't match ``n_expected`` - a genuine
    width mismatch here means something more fundamental is wrong with
    this worker's own result, which should surface loudly rather than
    silently corrupt every marker's summed effect.
    """
    if isinstance(effect_obj, pd.DataFrame):
        values = (np.zeros(n_expected, dtype=float) if effect_obj.shape[0] == 0
                  else effect_obj.iloc[0].to_numpy(dtype=float))
    elif isinstance(effect_obj, pd.Series):
        values = effect_obj.to_numpy(dtype=float)
    elif isinstance(effect_obj, dict):
        # Covers both rpy2.rlike.container.OrdDict and a plain dict -
        # both preserve R's own column order as Python insertion order,
        # so this is positionally equivalent to the DataFrame branch
        # above.
        values = np.asarray(list(effect_obj.values()), dtype=float).ravel()
    else:
        # Last resort: a plain list/tuple, a numpy array, or an
        # unconverted rpy2 vector - anything numpy can flatten.
        values = np.asarray(effect_obj, dtype=float).ravel()

    if values.shape[0] != n_expected:
        raise ValueError(
            f"a Shapley fan-out worker's r_effect resolved to "
            f"{values.shape[0]} value(s) (from a {type(effect_obj).__name__!r} "
            f"object) but this task expects {n_expected} (one per marker)."
        )
    return values


# ---------------------------------------------------------------------------
# ver4-5 blueprint R1.a (Stage 4, §3.4) - the closure refactor that
# unblocks multi-CPU Bayesian optimisation.
#
# ROOT CAUSE (blueprint §3.2 cause 2): GP()'s own nested _call_model()
# (below) closes over the live, per-process rpy2-bound rrBLUP/GBLUP/
# BayesB/RKHS R function objects GP() sources near its own top - for
# EVERY model it can dispatch, including pure-Python ones, since Python
# closures capture every name a function BODY references regardless of
# which branch actually runs. rpy2's Sexp-backed proxies wrap a live
# embedded R engine and have no pickle support at all, so handing
# _call_model() (or a lambda wrapping it) to a joblib process pool as
# run_model_fn fails at SUBMISSION time, before any candidate is
# evaluated remotely - this is why ver4-4's own Grid/Random trial
# fan-out (already built, already wired) could never actually fire for
# ANY model tuned through GP(), on any configuration (see Change
# Summary 6 §7/§8's own disclosure).
#
# THE FIX: never let a live rpy2 R function object be CAPTURED by a
# closure that might cross a process boundary. _get_r_model_fn() below
# resolves one fresh, per-PROCESS, per-model R function object on
# demand (sourcing the relevant .R file into THIS process's own R
# global environment on first use, exactly like GP()'s own R bootstrap
# block above, and _gblup_rkhs_shapley_row_worker()'s identical
# bootstrap sequence) and caches it - so a worker process resolves its
# OWN R function objects the first time it needs them, rather than ever
# receiving someone else's already-bound ones. ModelTrialRunner (below)
# is the resulting picklable stand-in for _call_model(): constructed
# from PLAIN DATA ONLY (invariant I13 - a result name, an R path
# string, a resolved compute-resources dict, and an optional
# bio_prior_ctx of DataFrames/paths/dicts), it holds no rpy2 object, no
# torch module, and nothing reachable from Streamlit.
# ---------------------------------------------------------------------------
_R_FN_CACHE: "dict[str, object]" = {}


def _get_r_model_fn(model_name, r_path):
    """Return THIS PROCESS's own rpy2-bound R function object for one of
    'rrBLUP'/'GBLUP'/'BayesB'/'RKHS', sourcing ``models/<model_name>.R``
    into this process's own R global environment on first use and
    caching it thereafter (module-level ``_R_FN_CACHE``, one entry per
    model name).

    Exists so no caller ever has to CAPTURE an already-bound R function
    object in a closure (invariant I13) - every call site that needs one
    (``ModelTrialRunner.__call__`` below) asks for it fresh, by name,
    every time. The bootstrap sequence (export R_HOME -> import rpy2 ->
    ``pandas2ri.activate()`` -> ``source()`` -> ``globalenv[model_name]``)
    is IDENTICAL to ``_gblup_rkhs_shapley_row_worker()``'s own (see that
    function's own docstring), which is itself identical to ``GP()``'s
    own R bootstrap block above - three independent call sites,
    deliberately kept in lockstep rather than sharing a single helper
    across a process boundary, since none of them can call each other
    (each must run in whichever process it happens to execute in).

    The cache is per PROCESS and per PYTHON INTERPRETER (a plain module-
    level dict) - a freshly spawned worker process starts with an empty
    cache and builds its own; the parent process keeps its own,
    completely independently. Never shared, never pickled.

    Parameters
    ----------
    model_name : one of 'rrBLUP'/'GBLUP'/'BayesB'/'RKHS' - selects which
        ``models/<model_name>.R`` file to source.
    r_path : this run's own ``R_PATH`` (``None`` = auto-detect, exactly
        like ``GP()``'s own top-level R bootstrap) - passed explicitly
        since this function has no closure to read it from.

    Returns
    -------
    The bound R function object (an ``rpy2.robjects.functions.Function``),
    callable exactly like ``rrBLUP(train, valid, test, params,
    result_name)`` etc. already are inside ``GP()``'s own main dispatch
    loop.
    """
    if model_name in _R_FN_CACHE:
        return _R_FN_CACHE[model_name]
    if r_path:
        os.environ['R_HOME'] = r_path
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    r_source = robjects.r['source']
    r_source(os.path.join(_R_MODELS_DIR, f'{model_name}.R'))
    fn = robjects.globalenv[model_name]
    _R_FN_CACHE[model_name] = fn
    return fn


def _run_gblup_or_rkhs_for_trial(model_name, r_path, compute_resources, train, valid, test,
                                  params, result_name, k_precomputed=None):
    """Picklable-context twin of ``GP()``'s own NESTED
    ``_run_gblup_or_rkhs()`` (see that function's own extensive
    docstring for the full row-range Shapley fan-out design - every line
    of the actual logic below mirrors it exactly). Used by
    ``ModelTrialRunner.__call__()`` (R1.a), which executes OUTSIDE
    ``GP()``'s own closure - either inside a hyperparameter-tuning trial
    worker process, or in the parent process for the tuner's final
    confirmatory fit - and so cannot reach ``GP()``'s own ``R_PATH``/
    ``_compute_resources`` locals the way the nested version does. Takes
    both explicitly instead.

    This is a NECESSARY duplication, not a drift risk (RK-2 applies to
    ``ModelTrialRunner`` vs. the main per-task dispatch loop, not to this
    helper): the two ``_run_gblup_or_rkhs*`` functions serve callers that
    cannot share a single Python closure across a process boundary, and
    neither branches on anything the other doesn't also branch on -
    keeping them in lockstep is a matter of copying edits across, the
    same discipline invariant I3/RK-2 already requires elsewhere in this
    module.

    ``nested_safe_n_jobs()`` (``pipeline_utils``) replaces a bare
    ``compute_resources['n_jobs']`` read here (unlike the nested
    version, which reads GP()'s own already-top-level-resolved value
    directly) - this guards against a FOURTH level of process fan-out
    being requested from INSIDE an already-parallel hyperparameter-
    tuning trial worker (R1.h/RK-6): if this call happens to execute
    inside a daemonic worker process, Python forbids it from spawning
    children at all, so this collapses to a safe, correct SERIAL call
    instead of raising - exactly ``_run_gblup_or_rkhs()``'s own
    ``n_jobs<=1`` path, just reached via a different, daemon-aware
    route. In practice this branch is rarely exercised during a search
    at all: ``_disable_explainability()`` forces GBLUP's/RKHS's own
    ``get_effect`` flag False for every search CANDIDATE (explainability
    is deliberately switched off during search - see
    ``models/hyperparameter_tuning.py::_disable_explainability``), so
    the fan-out below only has a chance to fire for the tuner's own
    FINAL confirmatory fit, which always executes in the parent process
    (never inside a trial worker) - so nesting is the exceptional case
    guarded against, not the common one.
    """
    params = list(params)
    if model_name == 'GBLUP':
        get_effect_idx, shapley_num_idx = 2, 3
    elif model_name == 'RKHS':
        get_effect_idx, shapley_num_idx = 3, 4
    else:
        raise ValueError(
            f"_run_gblup_or_rkhs_for_trial only supports 'GBLUP'/'RKHS', got {model_name!r}")

    r_kwargs = {}
    if k_precomputed is not None:
        r_kwargs['K_precomputed'] = k_precomputed

    r_fn = _get_r_model_fn(model_name, r_path)
    get_effect = bool(params[get_effect_idx])
    _raw_n_jobs = nested_safe_n_jobs(compute_resources, (compute_resources or {}).get('n_jobs'))
    if not get_effect or not _raw_n_jobs or _raw_n_jobs == 1:
        return r_fn(train, valid, test, params, result_name, **r_kwargs)

    shapley_num = params[shapley_num_idx]
    len_rows = min(test.shape[0], shapley_num) if shapley_num != 'all' else test.shape[0]
    if len_rows <= 1:
        return r_fn(train, valid, test, params, result_name, **r_kwargs)

    _chunk_count = int(_raw_n_jobs) if int(_raw_n_jobs) > 0 else (os.cpu_count() or 1)
    n_workers = min(_chunk_count, len_rows)
    if n_workers <= 1:
        return r_fn(train, valid, test, params, result_name, **r_kwargs)

    row_bounds = [b for b in np.array_split(np.arange(len_rows), n_workers) if len(b) > 0]

    try:
        from joblib import Parallel, delayed
        worker_results = Parallel(n_jobs=_raw_n_jobs)(
            delayed(_gblup_rkhs_shapley_row_worker)(
                model_name, r_path, train, valid, test, params, result_name,
                k_precomputed, int(bound[0]), int(len(bound)),
            )
            for bound in row_bounds
        )
    except Exception as exc:
        print(f"[ModelTrialRunner] NOTE: parallel Shapley row-range fan-out for {model_name} "
              f"failed ({exc!r}) - falling back to a single, unsplit call. This result is "
              f"unaffected, only wall-clock time.")
        return r_fn(train, valid, test, params, result_name, **r_kwargs)

    base_result = dict(worker_results[0])
    try:
        marker_columns = list(train.columns[:-1])
        summed_values = np.zeros(len(marker_columns), dtype=float)
        for worker_result in worker_results:
            this_effect = r_list_get(worker_result, 'r_effect')
            summed_values += _coerce_effect_values(this_effect, len(marker_columns))
        base_result['r_effect'] = pd.DataFrame([summed_values], columns=marker_columns)
    except Exception as exc:
        print(f"[ModelTrialRunner] NOTE: could not aggregate {model_name} Shapley row-range "
              f"fan-out results across {len(row_bounds)} worker process(es) ({exc!r}) - "
              f"falling back to a single, unsplit call.")
        return r_fn(train, valid, test, params, result_name, **r_kwargs)
    print(f"[ModelTrialRunner] {model_name} Shapley row-range fan-out: {len_rows} row(s) split "
          f"across {len(row_bounds)} worker process(es) (n_jobs={_raw_n_jobs}).")
    return base_result


class ModelTrialRunner:
    """Picklable stand-in for ``GP()``'s nested ``_call_model()``, used
    as ``tune_model_hyperparameters()``'s ``run_model_fn`` (ver4-5 R1.a -
    blueprint §3.4).

    Constructed from PLAIN DATA ONLY (invariant I13): a result name, an
    R path string, the resolved compute-resources dict, the run's
    ``GENOTYPE_FORMAT``, and an optional ``bio_prior_ctx`` of DataFrames/
    paths/dicts. Holds no rpy2 object, no torch module, no open file
    handle, no nested function, and nothing reachable from Streamlit -
    every R model function is obtained FRESH, per call, via
    ``_get_r_model_fn()`` (a module-level cache, keyed per PROCESS),
    never captured in ``__init__`` or held as an instance attribute.

    ``__call__(model_name, train, valid, test, params, result_name) ->
    dict``, in the SAME uniform shape ``_call_model()`` itself returns
    (invariant I3): a plain dict with ``'pearson_test'``/``'mse_test'``/
    ``'effect'``/``'predicted_test'``/``'predicted_valid'``/
    ``'predicted_train'``/``'pearson_valid'``/``'mse_valid'`` and, for
    models that emit them, ``'interaction'``/``'attention'``.

    This is a STRICT PORT of ``_call_model()``'s twelve model branches -
    moved VERBATIM, not "tidied" while moving (RK-2). ``GP()``'s own
    ``_call_model()`` (below) now delegates to an instance of this class
    rather than duplicating the branch logic itself - see that
    function's own (now much shorter) docstring. The LOCKSTEP WARNING
    that used to live on ``_call_model()`` applies here instead: every
    branch below must be kept in sync with the main per-task dispatch
    loop's own per-model branches further down ``GP()`` - the two remain
    a separate implementation of "how to call a model", by design (see
    the main dispatch loop's own comment), not a refactor of it.
    """

    def __init__(self, result_name, r_path, compute_resources, genotype_format,
                 bio_prior_ctx=None):
        self.result_name = result_name
        self.r_path = r_path
        self.compute_resources = dict(compute_resources) if compute_resources else {}
        self.genotype_format = genotype_format
        self.bio_prior_ctx = bio_prior_ctx  # plain dict of DataFrames/paths/dict, or None

    def __call__(self, model_name, train, valid, test, params, result_name):
        params = list(params)
        _scoring_valid = valid  # overridden below only by the bio_prior branch

        if model_name == 'rrBLUP':
            r_fn = _get_r_model_fn('rrBLUP', self.r_path)
            result = r_fn(train, valid, test, params, result_name)
            _cleanup_bglr_output_for(result_name)
            out = dict(pearson_test=r_list_get(result, 'r_pearson')[0], mse_test=r_list_get(result, 'r_MSE')[0],
                       effect=r_list_get(result, 'r_effect'), predicted_test=r_list_get(result, 'r_y_predicted'),
                       predicted_valid=r_list_get(result, 'r_y_predicted_valid'), predicted_train=r_list_get(result, 'r_y_predicted_train'))
            # Update ID ver4-5, R2 Stage 10 (I3 lockstep with the main
            # dispatch's equivalent branch, and with _r_model_surrogate_
            # interaction()'s own "one implementation, two call sites"
            # note): a Python-only, post-R-call surrogate interaction step.
            out['interaction'] = _r_model_surrogate_interaction(
                'rrBLUP', train, out['predicted_train'], params, self.compute_resources.get('n_jobs'),
                test=test,
            )

        elif model_name == 'GBLUP':
            if params[3] == 'all':
                params[3] = test.shape[0]
            _gblup_k_precomputed = (
                _build_grm(train, valid, test, self.compute_resources.get('device', 'cpu'))
                if self.compute_resources.get('gpu_kernel_precompute') else None
            )
            result = _run_gblup_or_rkhs_for_trial('GBLUP', self.r_path, self.compute_resources,
                                                   train, valid, test, params, result_name,
                                                   k_precomputed=_gblup_k_precomputed)
            _cleanup_bglr_output_for(result_name)
            out = dict(pearson_test=r_list_get(result, 'r_pearson')[0], mse_test=r_list_get(result, 'r_MSE')[0],
                       effect=r_list_get(result, 'r_effect'), predicted_test=r_list_get(result, 'r_y_predicted'),
                       predicted_valid=r_list_get(result, 'r_y_predicted_valid'), predicted_train=r_list_get(result, 'r_y_predicted_train'))
            out['interaction'] = _r_model_surrogate_interaction(
                'GBLUP', train, out['predicted_train'], params, self.compute_resources.get('n_jobs'),
                test=test,
            )

        elif model_name == 'BayesB':
            r_fn = _get_r_model_fn('BayesB', self.r_path)
            result = r_fn(train, valid, test, params, result_name)
            _cleanup_bglr_output_for(result_name)
            out = dict(pearson_test=r_list_get(result, 'r_pearson')[0], mse_test=r_list_get(result, 'r_MSE')[0],
                       effect=r_list_get(result, 'r_effect'), predicted_test=r_list_get(result, 'r_y_predicted'),
                       predicted_valid=r_list_get(result, 'r_y_predicted_valid'), predicted_train=r_list_get(result, 'r_y_predicted_train'))
            out['interaction'] = _r_model_surrogate_interaction(
                'BayesB', train, out['predicted_train'], params, self.compute_resources.get('n_jobs'),
                test=test,
            )

        elif model_name == 'RKHS':
            if params[4] == 'all':
                params[4] = test.shape[0]
            result = _run_gblup_or_rkhs_for_trial('RKHS', self.r_path, self.compute_resources,
                                                   train, valid, test, params, result_name)
            _cleanup_bglr_output_for(result_name)
            out = dict(pearson_test=r_list_get(result, 'r_pearson')[0], mse_test=r_list_get(result, 'r_MSE')[0],
                       effect=r_list_get(result, 'r_effect'), predicted_test=r_list_get(result, 'r_y_predicted'),
                       predicted_valid=r_list_get(result, 'r_y_predicted_valid'), predicted_train=r_list_get(result, 'r_y_predicted_train'))
            out['interaction'] = _r_model_surrogate_interaction(
                'RKHS', train, out['predicted_train'], params, self.compute_resources.get('n_jobs'),
                use_gpu=self.compute_resources.get('use_gpu_sklearn', False), test=test,
            )

        elif model_name == 'RF':
            if params[6] == 'all':
                params[6] = test.shape[0]
            r, mse, effect, interaction, pt, pv, ptr = RF(train, valid, test, params)
            out = dict(pearson_test=r, mse_test=mse, effect=effect, interaction=interaction,
                       predicted_test=pt, predicted_valid=pv, predicted_train=ptr)

        elif model_name == 'ExtraTrees':
            if params[6] == 'all':
                params[6] = test.shape[0]
            r, mse, effect, interaction, pt, pv, ptr = ExtraTrees(train, valid, test, params)
            out = dict(pearson_test=r, mse_test=mse, effect=effect, interaction=interaction,
                       predicted_test=pt, predicted_valid=pv, predicted_train=ptr)

        elif model_name == 'GBDT':
            if params[7] == 'all':
                params[7] = test.shape[0]
            r, mse, effect, interaction, pt, pv, ptr = GBDT(train, valid, test, params)
            out = dict(pearson_test=r, mse_test=mse, effect=effect, interaction=interaction,
                       predicted_test=pt, predicted_valid=pv, predicted_train=ptr)

        elif model_name == 'XGBoost':
            if params[7] == 'all':
                params[7] = test.shape[0]
            r, mse, effect, interaction, pt, pv, ptr = XGBoost(train, valid, test, params)
            out = dict(pearson_test=r, mse_test=mse, effect=effect, interaction=interaction,
                       predicted_test=pt, predicted_valid=pv, predicted_train=ptr)

        elif model_name == 'EBM':
            r, mse, effect, interaction, pt, pv, ptr = EBM(train, valid, test, params)
            out = dict(pearson_test=r, mse_test=mse, effect=effect, interaction=interaction,
                       predicted_test=pt, predicted_valid=pv, predicted_train=ptr)

        elif model_name == 'SVR':
            if params[7] == 'all':
                params[7] = test.shape[0]
            r, mse, effect, interaction, pt, pv, ptr = SV_Regression(train, valid, test, params)
            out = dict(pearson_test=r, mse_test=mse, effect=effect, interaction=interaction,
                       predicted_test=pt, predicted_valid=pv, predicted_train=ptr)

        elif model_name == 'KNN':
            if params[4] == 'all':
                params[4] = test.shape[0]
            r, mse, effect, interaction, pt, pv, ptr = KNN(train, valid, test, params)
            out = dict(pearson_test=r, mse_test=mse, effect=effect, interaction=interaction,
                       predicted_test=pt, predicted_valid=pv, predicted_train=ptr)

        elif model_name == 'MLP':
            if params[7] == 'all':
                params[7] = test.shape[0]
            r, mse, effect, interaction, pt, pv, ptr = ML_Perceptron(train, valid, test, params)
            out = dict(pearson_test=r, mse_test=mse, effect=effect, interaction=interaction,
                       predicted_test=pt, predicted_valid=pv, predicted_train=ptr)

        elif model_name == 'GAT_infinitesimal_node_level':
            if params[-2] == 'all':
                params[-2] = test.shape[0]
            r, mse, effect, pt, pv, ptr = GAT_infinitesimal_node_level(train, valid, test, params)
            out = dict(pearson_test=r, mse_test=mse, effect=effect,
                       predicted_test=pt, predicted_valid=pv, predicted_train=ptr)

        elif model_name == 'GAT_infinitesimal':
            if params[-1] == 'all':
                params[-1] = test.shape[0]
            r, mse, effect, pt, pv, ptr = GAT_infinitesimal(train, valid, test, params)
            out = dict(pearson_test=r, mse_test=mse, effect=effect,
                       predicted_test=pt, predicted_valid=pv, predicted_train=ptr)

        elif model_name == 'GAT_fully_connected':
            if params[-1] == 'all':
                params[-1] = test.shape[0]
            r, mse, effect, pt, pv, ptr, attn = GAT_fully_connected(train, valid, test, params)
            out = dict(pearson_test=r, mse_test=mse, effect=effect, attention=attn,
                       predicted_test=pt, predicted_valid=pv, predicted_train=ptr)

        elif model_name == 'GAT_prior_knowledge':
            if params[-2] == 'all':
                params[-2] = test.shape[0]
            # Update ID ver4-5, R2 (I3 lockstep with the main dispatch
            # branch, and with _call_model() above it): GAT_prior_knowledge
            # returns an 8th value (its own already-computed interaction
            # pairs, empty unless emit_interaction is set).
            r, mse, effect, pt, pv, ptr, attn, interaction = GAT_prior_knowledge(train, valid, test, params)
            out = dict(pearson_test=r, mse_test=mse, effect=effect, attention=attn, interaction=interaction,
                       predicted_test=pt, predicted_valid=pv, predicted_train=ptr)

        elif _is_bio_prior_model(model_name):
            if self.bio_prior_ctx is None:
                raise RuntimeError(
                    f"ModelTrialRunner: bio_prior_ctx is required for bio-prior model "
                    f"{model_name!r} but was not provided at construction."
                )
            bio_prior_pools_ctx = self.bio_prior_ctx['bio_prior_pools']
            train_unpruned_ctx = self.bio_prior_ctx['train_unpruned']
            valid_unpruned_ctx = self.bio_prior_ctx['valid_unpruned']
            test_unpruned_ctx = self.bio_prior_ctx['test_unpruned']
            bio_prior_merge_pools_ctx = self.bio_prior_ctx['bio_prior_merge_pools']
            full_marker_pool_ctx = self.bio_prior_ctx['full_marker_pool']
            bim_marker_info_path_ctx = self.bio_prior_ctx['bim_marker_info_path']
            network_cache_ctx = self.bio_prior_ctx.get('network_cache')
            phenotype_name_ctx = self.bio_prior_ctx.get('phenotype_name')

            bio_prior_params = list(params)
            bio_train, bio_valid, bio_test = bio_prior_pools_ctx.get(model_name, (train_unpruned_ctx, valid_unpruned_ctx, test_unpruned_ctx)) \
                if self.genotype_format == 'plink' else (train_unpruned_ctx, valid_unpruned_ctx, test_unpruned_ctx)
            _merge_source_data = bio_prior_merge_pools_ctx.get(model_name) if self.genotype_format == 'plink' else full_marker_pool_ctx

            if bio_prior_params[-1] == 'all':
                bio_prior_params[-1] = bio_test.shape[0]
            if self.genotype_format == 'plink':
                bio_prior_params[9] = bim_marker_info_path_ctx

            try:
                matched, declared = phenotype_matches_network_metadata(
                    bio_prior_params[7], phenotype_name_ctx
                )
                if matched is False:
                    print(
                        f"[{model_name}] WARNING: '{bio_prior_params[7]}' "
                        f"declares itself to be for phenotype {declared!r}, which doesn't "
                        f"obviously match the phenotype {phenotype_name_ctx!r} it's about "
                        f"to be used for. This is a heuristic name check, not proof of a "
                        f"mistake - double-check this is the right file for this trait."
                    )
            except Exception:
                pass  # never let this best-effort check block a real run

            r, mse, effect, pt, pv, ptr, attn = GAT_biological_prior_knowledge(
                bio_train, bio_valid, bio_test, bio_prior_params, result_name, phenotype_name_ctx, model_name,
                merge_source_data=_merge_source_data, network_cache=network_cache_ctx,
            )
            out = dict(pearson_test=r, mse_test=mse, effect=effect, attention=attn,
                       predicted_test=pt, predicted_valid=pv, predicted_train=ptr)
            _scoring_valid = bio_valid

        else:
            raise ValueError(f"ModelTrialRunner: unrecognised model {model_name!r}")

        if _scoring_valid.shape[0] != 0 and len(out.get('predicted_valid', [])) > 0:
            actual_valid = _scoring_valid.iloc[:, -1].values.tolist()
            out['pearson_valid'], out['mse_valid'] = safe_regression_metrics(
                actual_valid, out['predicted_valid'])
        else:
            out['pearson_valid'] = None
            out['mse_valid'] = None

        return out


def _trial_worker_init(result_name, r_path, resources_dict, gpu_semaphore=None):
    """Restore the per-process state every model module reads, inside a
    FRESHLY SPAWNED hyperparameter-tuning trial worker (ver4-5 R1.i -
    blueprint §3.4/§11.7). Mirrors ``intra_batch_parallel._worker_init()``
    - same steps, same order, and deliberately built from that module's
    OWN reusable primitives (``configure_r_environment`` /
    ``init_rpy2_conversion`` / ``install_gpu_semaphore``, all imported
    from ``pipeline_utils``) rather than a second implementation of them.
    The one addition beyond ``_worker_init()`` is ``set_active_compute_
    resources()`` - not needed by the task-level worker pool that
    function serves (which resolves its OWN compute resources fresh, at
    the top of its own ``GP()`` call), but essential here, since a trial
    worker never calls ``GP()`` at all.

    THIS IS MANDATORY, NOT OPTIONAL (RK-11) - and the reason is
    unusually easy to miss because it FAILS SILENTLY. A trial worker
    that never runs this: (1) has no R environment configured at all,
    so any R-backed model raises immediately (caught by
    ``_evaluate_batch()``'s own fallback - loud, at least); but worse,
    (2) ``get_active_compute_resources()`` returns the fully-generic
    CPU-only DEFAULT rather than this run's own resolved settings, so
    ``n_jobs`` resolves to ``-1`` ("every core") in EVERY worker
    simultaneously - the exact q-fold oversubscription
    ``nested_safe_n_jobs()``/R1.h exists to prevent, arriving by a
    different route - and ``device`` resolves to ``'cpu'``, so a GPU-
    tuned GAT/MLP trial silently trains on CPU while the confirmatory
    fit (run in the parent process, WITH this run's real resolved
    device) uses the GPU: candidates and the eventual winner would then
    be evaluated under DIFFERENT conditions, which is a correctness
    problem for the search's own outcome, not merely a speed one. No
    exception is ever raised for this - it must be prevented, not
    caught.

    Parameters
    ----------
    result_name, r_path, resources_dict : plain data, forwarded
        unchanged from the parent process (``resources_dict`` is exactly
        the dict ``resolve_compute_resources()`` already returned there
        - this worker must never RE-RESOLVE it itself, or a worker on a
        GPU node could resolve a different device than the parent
        already chose for the confirmatory fit).
    gpu_semaphore : the parent's own installed GPU semaphore object (see
        ``pipeline_utils.get_gpu_semaphore()``), or ``None`` when none
        has been installed (every CPU-only run, and every GPU run that
        isn't ALSO using intra-batch task-level parallelism) - a no-op
        in that case, matching ``install_gpu_semaphore(None)``'s own
        documented behaviour.
    """
    try:
        configure_r_environment(r_path, r_blas_threads=(resources_dict or {}).get('r_blas_threads'))
    except Exception as exc:
        print(f"[ModelTrialRunner] NOTE: _trial_worker_init could not configure the R "
              f"environment in worker PID {os.getpid()} ({exc!r}) - any R-backed model "
              f"tuned in this worker will fail its own R bootstrap and fall back to serial "
              f"evaluation (see _evaluate_batch()'s own NOTE line).")
    try:
        init_rpy2_conversion()
    except Exception as exc:
        print(f"[ModelTrialRunner] NOTE: _trial_worker_init could not initialise rpy2 "
              f"conversion in worker PID {os.getpid()} ({exc!r}).")
    set_active_compute_resources(resources_dict)
    if gpu_semaphore is not None:
        install_gpu_semaphore(gpu_semaphore)


# ---------------------------------------------------------------------------
# ver4-4 R4.h (blueprint §2.4.2): Python-side GBLUP genomic-relationship-
# matrix precompute. Module-level (no GP()-local state needed) - builds
# G = scale(X) @ scale(X).T / p over rbind(train, valid, test) in that
# EXACT row order (matching GBLUP.R's own `data <- rbind(train, valid,
# test)`), optionally on a CUDA device, then hands the finished matrix to
# GBLUP.R via its own optional K_precomputed argument - skipping that
# function's own O(N^2*M) scale()+tcrossprod build entirely (R4.h's own
# design record).
#
# Deliberately GBLUP-only for this stage - see the ver4-4 Stage 5 hand-off
# note §5.5/§6.4 for the disclosed scope decision: RKHS's own kernel
# K=exp(-h*D) depends on the per-call, potentially-TUNED bandwidth h, so
# only D (the h-independent squared-distance matrix) could ever be
# precomputed once outside a hyperparameter search - and RKHS.R already
# received its own, unconditional (not GPU_KERNEL_PRECOMPUTE-gated)
# dist()->GEMM speed-up on the R side (see that file's own
# .squared_euclidean_dist_gemm()), which is the improvement R4.h actually
# promises for RKHS this stage. A Python-side D-precompute helper for RKHS
# remains a disclosed, not-yet-implemented follow-up (see the Change
# Summary's own §7/§8).
# ---------------------------------------------------------------------------
def _build_grm(train, valid, test, device):
    """Genomic relationship matrix G = scale(X) @ scale(X).T / p, over
    rbind(train, valid, test) in that exact row order - torch on a CUDA
    device, NumPy otherwise. `scale()` uses R's own convention (centre by
    the column mean, scale by the column SAMPLE standard deviation,
    ddof=1) so the result matches what GBLUP.R's own
    `scale(data_qtl, center=T, scale=T)` would have produced, to
    floating-point-rounding tolerance.

    A zero-variance (monomorphic) marker column would make R's own
    `scale()` produce NaN/Inf for that column (division by a zero SD) -
    reproduced here explicitly (`ddof=1` std of a constant column is
    exactly 0.0) and then replaced with 0.0 post-hoc (`np.nan_to_num`),
    matching what that column's own contribution to a GEMM product built
    from R's NaN/Inf output would numerically collapse to in practice
    (R's own `%*%` on a column containing NaN/Inf would itself poison the
    ENTIRE matrix with NaN/Inf, which is arguably worse - this Python
    path is therefore not merely equivalent but strictly more robust to a
    monomorphic marker than the R path it replaces, a deliberate,
    disclosed improvement rather than an exact-bug-for-bug reproduction).

    Returns None (never raises) on ANY failure - including a resulting
    matrix that is non-finite or insufficiently symmetric - in which case
    the caller (see GP()'s own GBLUP dispatch, both call sites) falls
    back to letting GBLUP.R build its own G exactly as it always has.
    """
    try:
        marker_frames = [df.iloc[:, :-1] for df in (train, valid, test) if df.shape[0] != 0]
        x = pd.concat(marker_frames, axis=0).to_numpy(dtype=np.float64)
        col_mean = x.mean(axis=0)
        col_std = x.std(axis=0, ddof=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            x_scaled = (x - col_mean) / col_std
        x_scaled = np.nan_to_num(x_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        if str(device).startswith('cuda'):
            import torch
            t = torch.as_tensor(x_scaled, dtype=torch.float64, device=torch.device(device))
            grm = (t @ t.T / x_scaled.shape[1]).detach().cpu().numpy()
        else:
            grm = (x_scaled @ x_scaled.T) / x_scaled.shape[1]

        if not np.all(np.isfinite(grm)) or not np.allclose(grm, grm.T, atol=1e-8, rtol=1e-5):
            print("[GP] NOTE: Python-side GBLUP GRM precompute produced a non-finite or "
                  "non-symmetric matrix - falling back to GBLUP.R's own kernel build for "
                  "this call (GPU_KERNEL_PRECOMPUTE remains enabled for future calls).")
            return None
        return grm
    except Exception as exc:
        print(f"[GP] NOTE: Python-side GBLUP GRM precompute failed ({exc!r}) - falling back "
              f"to GBLUP.R's own kernel build for this call (GPU_KERNEL_PRECOMPUTE remains "
              f"enabled for future calls).")
        return None


# ---------------------------------------------------------------------------
# Update ID ver4-5, R2 Stage 10 (blueprint §4.2 Layer 3, §4.4): the ONLY
# route this codebase offers to reach rrBLUP/BayesB/GBLUP/RKHS's marker-pair
# interactions - see models.interaction_extraction.surrogate_tree_
# interactions()'s own docstring for the full "why a surrogate, and why it
# is APPROXIMATE, not exact" explanation (RK-8).
#
# Deliberately scoped to the SURROGATE path only for all four R models -
# the blueprint's own §4.2 Layer 3 table also offers rrBLUP/BayesB an
# EPISTATIC PRODUCT-COLUMN path inside their own .R files (fitting extra
# marker(i)*marker(j) design columns directly in BGLR). That path is NOT
# implemented in this delivery (disclosed scope decision, ver4-5 Change
# Summary §7/§8) - the surrogate path alone already satisfies "GBLUP, RKHS:
# surrogate only" for two of the four models, and is offered here for
# rrBLUP/BayesB too (the blueprint's own "... OR surrogate" alternative for
# those two), so every one of the four R models gains a working, if
# approximate, interaction route from ONE mechanism instead of two.
#
# Because this is a PURE PYTHON post-processing step (it never touches R or
# rpy2 - it fits a plain scikit-learn surrogate on values BOTH ModelTrialRunner
# and the main per-task dispatch loop already have in hand: this task's own
# train split and the R model's own predicted_train output), there is only
# ONE implementation of it, called identically from both dispatch sites -
# this satisfies invariant I3's "lockstep" requirement for these four models
# by construction, rather than by discipline (there is no second copy to
# drift out of sync).
#
# Update (RKHS H-statistic interactions): RKHS's own branch below now
# explains its surrogate via Friedman's H-statistic
# (models.interaction_extraction.surrogate_h_statistic_interactions())
# instead of pairwise TreeSHAP - rrBLUP/BayesB/GBLUP are UNCHANGED and keep
# the TreeSHAP surrogate path. RKHS was singled out because its kernel
# already captures non-additive structure (unlike rrBLUP/GBLUP's purely
# additive fit or BayesB's additive variable selection), so its own
# "Return marker-pair interactions?" toggle is unhidden again in
# main_app.py's HIDDEN_HPARAM_FIELDS (rrBLUP/BayesB/GBLUP's stay hidden) -
# see that dict's own comment for the full history. Both extraction
# algorithms are dispatched from this SAME single function (by model_name),
# so the lockstep guarantee above is unaffected - there is still only one
# call site's worth of branching logic to keep in sync, not two.
# ---------------------------------------------------------------------------
_R_MODEL_BASE_PARAM_COUNT = {'rrBLUP': 4, 'BayesB': 4, 'GBLUP': 9, 'RKHS': 10}


def _r_model_interaction_fields(model_name, params):
    """Read the appended, PYTHON-ONLY interaction fields every R model
    (rrBLUP/BayesB/GBLUP/RKHS) now carries AFTER its own existing,
    R-consumed params (invariant I5 - append-only, read defensively so a
    config predating this field simply has interactions off, exactly
    like every other model's own newly-added toggle).

    These fields are NEVER passed into the R function itself -
    every R model file here (`models/rrBLUP.R` etc.) only ever reads
    `params[1]` through its own fixed, documented count and ignores
    anything appended after that (each file's own `unlist(params)` call
    simply produces a longer R vector that the rest of that file never
    indexes into - verified directly by reading every one of the four
    files, not assumed), so leaving them in the list actually passed to
    `r_fn(...)` would already have been harmless; they are threaded
    through the CALLER's own params list purely so HPARAMETERS keeps
    ONE flat list per model (matching every other model in this
    codebase), not because the R side requires them to be separated
    out.

    Update ID ver4-6, R1/R1b (blueprint §2.4/§2.10.4): three FURTHER
    fields - `interaction_screen`/`interaction_screen_top`/
    `interaction_grid_resolution` - are read here for EVERY R model
    (rrBLUP/BayesB/GBLUP too, not just RKHS), at their own no-op
    defaults, purely so this stays ONE shared reader rather than a
    RKHS-specific fork of it. Only `_r_model_surrogate_interaction()`'s
    OWN `model_name == 'RKHS'` branch actually forwards them onward
    (into `surrogate_h_statistic_interactions()`'s own `surrogate_cfg`)
    - rrBLUP/BayesB/GBLUP keep calling `surrogate_tree_interactions()`
    (exact pairwise TreeSHAP on a surrogate, not Friedman's H-statistic),
    which has no `screen`/`grid_resolution` concept at all, so these
    three values are read but silently unused for those three models.

    Returns
    -------
    (get_interaction: bool, max_interaction_features: int | 'all',
     interaction_background: int, interaction_top: float | int | 'all',
     interaction_screen: str, interaction_screen_top: float | 'all',
     interaction_grid_resolution: int)
    """
    base_count = _R_MODEL_BASE_PARAM_COUNT[model_name]
    get_interaction = params[base_count] if len(params) > base_count else False
    max_interaction_features = params[base_count + 1] if len(params) > base_count + 1 else 500
    interaction_background = params[base_count + 2] if len(params) > base_count + 2 else 100
    interaction_top = params[base_count + 3] if len(params) > base_count + 3 else 'all'
    interaction_screen = params[base_count + 4] if len(params) > base_count + 4 else 'off'
    interaction_screen_top = params[base_count + 5] if len(params) > base_count + 5 else 2.0
    interaction_grid_resolution = params[base_count + 6] if len(params) > base_count + 6 else 3
    return (get_interaction, max_interaction_features, interaction_background, interaction_top,
            interaction_screen, interaction_screen_top, interaction_grid_resolution)


def _r_model_surrogate_interaction(model_name, train, predicted_train, params, n_jobs, use_gpu=False,
                                    test=None):
    """Fit a lightweight surrogate on `model_name`'s OWN predicted_train
    output (never the true phenotype - see models.interaction_
    extraction.surrogate_tree_interactions()'s/surrogate_h_statistic_
    interactions()'s own Raises note) and return its pairwise
    interactions, shortlisted and top-filtered per this model's own
    appended HPARAMETERS fields (see _r_model_interaction_fields()
    above).

    Dispatches to ONE of two interaction-extraction algorithms run on
    that SAME surrogate, by `model_name`:

      - `model_name == 'RKHS'`: Friedman's H-statistic
        (models.interaction_extraction.surrogate_h_statistic_
        interactions()) - RKHS's own route, singled out because its
        kernel already captures non-additive structure (see the
        module-level comment above this function for the full
        rationale). `use_gpu` (forwarded from this run's
        `USE_GPU_SKLEARN` compute-resource setting, exactly like
        RF/SVR/KNN's own model fits already do) lets the surrogate fit
        attempt a cuML GPU backend here - safe for THIS algorithm (see
        `surrogate_h_statistic_interactions()`'s own docstring for why),
        unlike the TreeSHAP branch below.
      - Every other model (rrBLUP/BayesB/GBLUP): exact pairwise
        TreeSHAP (models.interaction_extraction.surrogate_tree_
        interactions()) - UNCHANGED from before this branch existed.
        `use_gpu` is accepted but ignored for these three (TreeSHAP
        cannot introspect a GPU-fitted forest - see
        `models/RF.py`'s own equivalent comment).

    NEVER RAISES - returns an empty DataFrame if get_interaction is
    False, or if anything about the surrogate fit itself fails. A
    surrogate interaction step failing must never fail the model's own
    real prediction, which by the time this is called has already
    succeeded and been scored.

    Update (Requirements.md item 2 - cross-model H-index result
    diversity): `test`, when given (this task's own `test` split - now
    passed from every one of this function's own call sites, both in
    ModelTrialRunner.__call__ and the main per-task dispatch loop), is
    what the surrogate is EXPLAINED on - the surrogate itself is still
    FITTED on `train`/`predicted_train` exactly as before (a larger,
    more informative fit population). Previously this function always
    fit AND explained on `train`, unlike RF/SVR/KNN's own direct
    (non-surrogate) interaction routes, which always explain on `test`
    (see models/RF.py, models/SVR.py, models/KNN.py) - comparing a
    train-explained surrogate ring against a test-explained direct ring
    is an apples-to-oranges population mismatch (different sample size,
    different realised allele frequencies) that was a genuine,
    structural contributor to "the number of extracted interactions
    among the prediction models (RKHS, RF, SVR and KNN) are quite
    diverse even under the same configuration". `test=None` (or an
    empty test split) falls back to explaining on `train`, this
    function's previous behaviour, so any caller that cannot supply a
    test split yet is unaffected.
    """
    (get_interaction, max_interaction_features, interaction_background, interaction_top,
     interaction_screen, interaction_screen_top, interaction_grid_resolution) = \
        _r_model_interaction_fields(model_name, params)
    if not get_interaction:
        return pd.DataFrame()

    try:
        train_x = train.iloc[:, :-1]
        train_y = train.iloc[:, -1]
        shortlist_idx = shortlist_markers(
            train_x.corrwith(train_y).abs().fillna(0).to_numpy(), max_interaction_features,
        )
        marker_names = list(train_x.columns[shortlist_idx])
        x_shortlist = train_x.iloc[:, shortlist_idx]

        y_pred = np.asarray(predicted_train, dtype=float).ravel()
        if y_pred.shape[0] != x_shortlist.shape[0]:
            raise ValueError(
                f"{model_name}: predicted_train has {y_pred.shape[0]} values but this task's "
                f"train split has {x_shortlist.shape[0]} rows - cannot fit a surrogate on "
                f"mismatched rows."
            )

        # Requirements.md item 2: explain the surrogate on THIS task's
        # own TEST split (the SAME shortlisted marker columns picked
        # above from train) rather than on `train` itself - see this
        # function's own module-level update note above for the full
        # rationale. Falls back to `None` (explain on train, the
        # previous behaviour) whenever a usable test split isn't
        # available, so this remains backward compatible.
        explain_pool = None
        if test is not None and test.shape[0] > 0:
            explain_pool = test.iloc[:, :-1][marker_names]
        _explain_n = explain_pool.shape[0] if explain_pool is not None else x_shortlist.shape[0]

        if model_name == 'RKHS':
            # All pairs among the already-shortlisted markers - the SAME
            # "shortlist first, then every pair within it" convention
            # models/SVR.py and models/KNN.py already use for their own
            # h_statistic_interactions() calls (h_statistic_interactions()
            # itself requires an explicit pairs list - see its own
            # docstring).
            pairs = [(a, b) for a in range(len(marker_names)) for b in range(a + 1, len(marker_names))]
            # Update ID ver4-6, R1/R1b (blueprint §2.4/§2.10.4): RKHS's
            # own three appended fields forwarded into
            # surrogate_h_statistic_interactions()'s own surrogate_cfg -
            # 'off'/3 (the no-op defaults) reproduce ver4-5 numbers
            # exactly for any config predating this update (AC1.7).
            # `interaction_screen_top` is a percentage (0-100, or 'all'),
            # exactly like every other 'top_pct' field in this schema -
            # converted to h_statistic_interactions()'s own (0, 1]
            # fraction convention here, mirroring models/SVR.py's own
            # identical conversion for its three direct call sites.
            _screen = interaction_screen if interaction_screen != 'off' else None
            _screen_keep = (
                float(interaction_screen_top) / 100.0 if interaction_screen_top != 'all' else 1.0
            )
            interaction_sample = surrogate_h_statistic_interactions(
                x_shortlist, y_pred, marker_names, pairs=pairs,
                surrogate_cfg={
                    'n_background': min(int(interaction_background), _explain_n),
                    'grid': 'auto',
                    'grid_resolution': int(interaction_grid_resolution),
                    'screen': _screen,
                    'screen_keep': _screen_keep,
                },
                n_jobs=(n_jobs if n_jobs and n_jobs > 0 else 1),
                use_gpu=bool(use_gpu),
                explain_pool=explain_pool,
            )
            method_label = 'surrogate-H-statistic'
        else:
            interaction_sample = surrogate_tree_interactions(
                x_shortlist, y_pred, marker_names,
                surrogate_cfg={'explain_sample_size': min(int(interaction_background), _explain_n)},
                n_jobs=(n_jobs if n_jobs and n_jobs > 0 else 1),
                explain_pool=explain_pool,
            )
            method_label = 'surrogate-TreeSHAP'

        interaction_sample = top_select(interaction_sample, 'percentage', interaction_top)
        print(f"[GP] {model_name}: {method_label} interaction extraction produced "
              f"{interaction_sample.shape[0]} pair(s) from a {len(marker_names)}-marker "
              f"shortlist - APPROXIMATE (surrogate of this model's own predictions, not "
              f"this model's own internal computation - see the GUI help text).")
        return interaction_sample
    except Exception as exc:
        print(f"[GP] NOTE: {model_name} surrogate interaction extraction failed ({exc!r}) - "
              f"this model's own predictions and marker effects are UNAFFECTED, only its "
              f"Interaction.csv rows for this task are skipped.")
        return pd.DataFrame()


def GP(GENOTYPE_FILE_NAME, PHENOTYPE_FILE_NAME, MODEL, PHENOTYPE, RATIO, SAMPLE_NUM, HPARAMETERS, R_PATH, W_OPT, RESULT_NAME, HYPERPARAMETERS_OPT, SCENARIO, PARALLEL=None, LD_prune=None, RF_filter=None, GENOTYPE_FORMAT='csv', GENOTYPE_PLINK_PATH='plink2', OTHER_MODELS_MARKER_SOURCE='full_or_filtered', progress_callback=None, HP_TUNE=None, HP_TUNE_ENSEMBLE_MODE='per_method', MIN_DATA_POINTS=100, USE_GPU_SKLEARN=False, N_JOBS=-1, PLINK_THREADS=1, R_BLAS_THREADS=None, TORCH_DEVICE=None, CUDNN_BENCHMARK=True, USE_AMP=False, N_CPU_WORKERS=1, N_GPU_SLOTS=None, MODEL_DISPATCH_FILTER=None, GPU_SLOTS_PER_DEVICE=1, R_BLAS_FOLLOWS_N_JOBS=True, TORCH_NUM_THREADS=None, TORCH_DATALOADER_WORKERS=0, GPU_EVAL_BATCH=32, GPU_LD_R2=True, GPU_KERNEL_PRECOMPUTE=True, HP_TUNE_PARALLEL_TRIALS=True, HP_TUNE_BAYES_BATCH=True, HP_TUNE_BAYES_BATCH_MAX=8, HP_TUNE_BAYES_LIAR='max', HP_TUNE_PARALLEL_RESTARTS=True, MODEL_AVAILABILITY_STRICT=True,
       HP_TUNE_BAYES_DOMAIN_REDUCTION='auto', HP_TUNE_WARM_START=False,
       HP_TUNE_SELECTION_MARGIN=0.02, HP_TUNE_VALID_REPEATS=1, HP_TUNE_SCOPE='per_task',
       W_OPT_ANALYTIC_SEED=False, W_OPT_VALIDATION_FLOOR=False, W_OPT_SIMPLEX_SEARCH=True,
       RESULT_COMPRESSION='gzip'):

    # Update ID 2, R1 (blueprint §10.3): MODEL_DISPATCH_FILTER is a
    # runtime-only allow-list of post-expansion MODEL_RUN entries (e.g.
    # 'RF__Grid', 'GAT_biological_prior_knowledge_2') - set SOLELY by
    # intra_task_parallel.py's worker entry point when fanning one task's
    # model dispatch out across worker processes. None (the default,
    # every existing caller) reproduces this function's pre-Update-2
    # behaviour bit for bit - see the three gates below (G1/G2/G3) for the
    # only three places this is ever consulted. Deliberately NOT a config
    # key: a user-supplied filter in a hand-edited *_config.json would
    # silently drop models from a run with no other symptom, so
    # run_step1_batch.py/run_sequential.py/main_app.py never read or set
    # this from cfg (see those files' own handling if a config somehow
    # contains it anyway). Normalised to a frozenset once, here, so every
    # gate below does an O(1) membership test against the same immutable
    # set rather than re-deriving it.
    _dispatch_filter = frozenset(MODEL_DISPATCH_FILTER) if MODEL_DISPATCH_FILTER is not None else None
    print(f"[GP] MODEL_DISPATCH_FILTER resolved: "
          f"{'inactive (dispatching every model in MODEL_RUN, as always)' if _dispatch_filter is None else sorted(_dispatch_filter)}.")

    # Update ID ver4-9, R7 (design blueprint §2.4.7 - "Diagnostics as a
    # design principle", architecture doc §17): the ONE parameter R7 adds
    # to GP() itself - forwarded, unchanged, to every checkpoint_utils.py
    # call below (result_file_paths()/save_partial_results()/
    # append_partial_results() - load_partial_results()/clear_result_files()
    # take NO compression argument at all; see their own docstrings for
    # why: they always resolve/clear BOTH possible extensions
    # unconditionally, regardless of what this run's own value is, so a
    # resumed run is never tripped up by a since-changed setting, RK-6/
    # RK-7). Logged unconditionally, every run, matching this codebase's
    # own established discipline of never leaving a resolved feature state
    # to silent inference (e.g. the MODEL_DISPATCH_FILTER line just
    # above, or LD-decay's own resolved-state line elsewhere in this
    # function).
    if RESULT_COMPRESSION not in ('gzip', 'none'):
        raise ValueError(
            f"[GP] RESULT_COMPRESSION must be 'gzip' or 'none', got {RESULT_COMPRESSION!r}."
        )
    print(f"[GP] RESULT_COMPRESSION resolved: {RESULT_COMPRESSION!r} - "
          f"{'the six large result files (Prediction_result_{train,valid,test}.csv, Marker_effect.csv, Interaction.csv, Attention.csv) will be gzip-compressed (.csv.gz); the four small ones (Metric.csv, Weight.csv, hyperparameter.csv, Basic_stats.csv) never are' if RESULT_COMPRESSION == 'gzip' else 'every result file will be plain, uncompressed CSV, matching every pre-ver4-9 run'}.")

    # Update ID ver4-5, R2 (blueprint §4.6 failure-mode table): Tier 2
    # models (XGBoost, EBM - decision D2) are OPTIONAL dependencies. A
    # headless run naming one of them in MODEL should fail HERE, before
    # any task starts, rather than mid-run after hours of fitting other
    # models. MODEL_AVAILABILITY_STRICT=True (the default) raises
    # immediately with an actionable install instruction; False logs the
    # same information as a warning and lets the run proceed (the model's
    # own dispatch branch will then raise when it is actually reached -
    # a strictly worse failure mode, offered only for a caller that wants
    # to probe availability itself before calling GP()).
    _unavailable = []
    for _m in MODEL:
        if _m == 'ensemble':
            continue
        _base = schema_key_of(base_of(_m))
        _pkg = optional_dependency_for(_base)
        if _pkg is not None and not is_available(_base):
            _unavailable.append((_m, _base, _pkg))
    if _unavailable:
        _lines = "\n".join(
            f"  - {name!r} (needs the optional '{pkg}' package)" for name, _base, pkg in _unavailable
        )
        _message = (
            f"[GP] The following selected model(s) need an optional dependency that is not "
            f"installed in this environment:\n{_lines}\n"
            f"Install the missing package(s) (e.g. 'pip install xgboost' / 'pip install "
            f"interpret'), or remove these models from MODEL."
        )
        if MODEL_AVAILABILITY_STRICT:
            raise ImportError(_message)
        else:
            print(f"[GP] WARNING (MODEL_AVAILABILITY_STRICT=False): {_message}\nProceeding "
                  f"anyway - the affected model(s) will fail with the same message the first "
                  f"time they are actually dispatched.")
    else:
        print("[GP] MODEL_AVAILABILITY_STRICT: every selected model's optional dependencies "
              "(if any) are available.")

    # Phase 2, Requirement 6: resolve every compute-hardware setting
    # (device, n_jobs, PLINK/BLAS thread counts, ...) ONCE here, from this
    # call's own arguments, and publish it via
    # pipeline_utils.set_active_compute_resources() so every model module
    # (RF/SVR/KNN/MLP/GAT_*.py, plink_io.py) reads the SAME resolved
    # settings for the rest of this run - see
    # pipeline_utils.resolve_compute_resources()'s own docstring for why
    # this is done via a process-global rather than threading a new
    # positional HPARAMETERS entry through every model's contract. Applied
    # BEFORE the R/rpy2 import block below so R_BLAS_THREADS (if any) is
    # already exported to the environment before R itself is ever touched.
    # GPU_SLOTS_PER_DEVICE (Update ID 2, R2 - Test Report D1 fix): read
    # here and forwarded into resolve_compute_resources() alongside
    # N_GPU_SLOTS itself, so a hand-edited (or GUI-exported)
    # GPU_SLOTS_PER_DEVICE value in a config JSON actually reaches the
    # runtime n_gpu_slots resolution instead of only ever being consumed
    # by resource_profiles.estimate_resources() (the GUI advisor). Default
    # 1 reproduces the exact pre-fix/pre-Update-2 "one slot per visible
    # device" behaviour (A2.2 flag-off byte-identity) for every existing
    # caller that never passes this argument.
    # ver4-4 Stage 5 (blueprint §10 Stage 5 checklist item 1 / §2.3.2 R3.b-c
    # / §2.4.2 R4.b-h): six further compute-resource kwargs, every one
    # defaulted to reproduce pre-ver4-4 behaviour exactly (I11) -
    # R_BLAS_FOLLOWS_N_JOBS/TORCH_NUM_THREADS existed as resolve_compute_
    # resources() config keys since Stage 3 but were never threaded onto
    # GP()'s OWN signature until now, so no caller could actually reach
    # them (the R_BLAS_FOLLOWS_N_JOBS=false rollback-plan row - blueprint
    # §8 - could never take effect via GP() until this fix); the
    # remaining four (TORCH_DATALOADER_WORKERS/GPU_EVAL_BATCH/GPU_LD_R2/
    # GPU_KERNEL_PRECOMPUTE) are genuinely new this stage. See
    # pipeline_utils.resolve_compute_resources()'s own docstring for each
    # key's full meaning/default rationale - this dict only forwards
    # GP()'s own already-defaulted arguments into it, never re-deciding a
    # default independently (a second, drifting copy of the same default
    # is exactly the kind of divergence invariant I6/I11 style guards
    # against elsewhere in this codebase).
    _compute_cfg = {
        'USE_GPU_SKLEARN': USE_GPU_SKLEARN, 'N_JOBS': N_JOBS, 'PLINK_THREADS': PLINK_THREADS,
        'R_BLAS_THREADS': R_BLAS_THREADS, 'TORCH_DEVICE': TORCH_DEVICE,
        'CUDNN_BENCHMARK': CUDNN_BENCHMARK, 'USE_AMP': USE_AMP,
        # PARALLEL (this run's own batch_id/batch_size dict, already a GP()
        # parameter - not new) is threaded through here purely so
        # resolve_compute_resources() can WARN when N_CPU_WORKERS_TASK
        # requests more concurrent task-level workers than this batch's own
        # batch_size can ever run concurrently while N_JOBS/PLINK_THREADS
        # are left single-threaded - see that function's own comment for
        # the full "LD filtering and RF filtering take too much time"
        # story this was added for. Read-only: nothing here changes what
        # PARALLEL itself does (the task-slicing logic further below is
        # unaffected), and resolve_compute_resources() never mutates it.
        'PARALLEL': PARALLEL,
        'N_CPU_WORKERS': N_CPU_WORKERS, 'N_GPU_SLOTS': N_GPU_SLOTS,
        'GPU_SLOTS_PER_DEVICE': GPU_SLOTS_PER_DEVICE,
        'R_BLAS_FOLLOWS_N_JOBS': R_BLAS_FOLLOWS_N_JOBS, 'TORCH_NUM_THREADS': TORCH_NUM_THREADS,
        'TORCH_DATALOADER_WORKERS': TORCH_DATALOADER_WORKERS, 'GPU_EVAL_BATCH': GPU_EVAL_BATCH,
        'GPU_LD_R2': GPU_LD_R2, 'GPU_KERNEL_PRECOMPUTE': GPU_KERNEL_PRECOMPUTE,
        # ver4-4 Stage 6 (blueprint §10 Stage 6 checklist / §2.3.2 R3.f):
        # one further compute-resource kwarg, defaulted to reproduce
        # pre-ver4-4 behaviour (n_jobs<=1 is always byte-identical
        # regardless of this flag - see resolve_compute_resources()'s own
        # docstring). See the HP_TUNE dispatch site below for the
        # disclosed, real-world caveat on what this flag can actually
        # achieve through THIS call site today.
        'HP_TUNE_PARALLEL_TRIALS': HP_TUNE_PARALLEL_TRIALS,
        # ver4-5 blueprint R1 (§3.6): four further compute-resource
        # kwargs, every one defaulted to reproduce pre-ver4-5 behaviour
        # exactly (I11) - see resolve_compute_resources()'s own
        # docstring and the HP_TUNE dispatch site below for how each is
        # actually consumed.
        'HP_TUNE_BAYES_BATCH': HP_TUNE_BAYES_BATCH, 'HP_TUNE_BAYES_BATCH_MAX': HP_TUNE_BAYES_BATCH_MAX,
        'HP_TUNE_BAYES_LIAR': HP_TUNE_BAYES_LIAR, 'HP_TUNE_PARALLEL_RESTARTS': HP_TUNE_PARALLEL_RESTARTS,
        # Update ID ver4-6 (blueprint §4 config schema delta): eight
        # further compute-resource kwargs - five for the hyperparameter-
        # tuning track (R1/R3/R4), three for the weight-optimisation
        # track (R5/R6/R7). Every one defaulted to its own §4-documented
        # value (NOT necessarily byte-identical to ver4-5 - several are
        # deliberate, disclosed default flips; see resolve_compute_
        # resources()'s own docstring and each key's own design record
        # for why). Forwarded here exactly like every key above -
        # GP()'s own already-defaulted arguments only, never a second,
        # independently-decided default.
        'HP_TUNE_BAYES_DOMAIN_REDUCTION': HP_TUNE_BAYES_DOMAIN_REDUCTION,
        'HP_TUNE_WARM_START': HP_TUNE_WARM_START,
        'HP_TUNE_SELECTION_MARGIN': HP_TUNE_SELECTION_MARGIN,
        'HP_TUNE_VALID_REPEATS': HP_TUNE_VALID_REPEATS,
        'HP_TUNE_SCOPE': HP_TUNE_SCOPE,
        'W_OPT_ANALYTIC_SEED': W_OPT_ANALYTIC_SEED,
        'W_OPT_VALIDATION_FLOOR': W_OPT_VALIDATION_FLOOR,
        'W_OPT_SIMPLEX_SEARCH': W_OPT_SIMPLEX_SEARCH,
    }
    _compute_resources = resolve_compute_resources(_compute_cfg)
    set_active_compute_resources(_compute_resources)
    apply_r_blas_threads(_compute_resources['r_blas_threads'])
    print(f"[GP] Compute resources resolved: device={_compute_resources['device']}, "
          f"n_jobs={_compute_resources['n_jobs']}, use_gpu_sklearn={_compute_resources['use_gpu_sklearn']}, "
          f"plink_threads={_compute_resources['plink_threads']}, "
          f"r_blas_threads={_compute_resources['r_blas_threads']}, "
          f"n_cpu_workers={_compute_resources['n_cpu_workers']}, n_gpu_slots={_compute_resources['n_gpu_slots']} "
          f"(gpu_slots_per_device={_compute_resources['gpu_slots_per_device']}).")
    # ver4-4 Stage 5: the six additional resolved keys above (§10 Stage 5
    # checklist item 1) - a separate print so the original line above
    # stays byte-identical for any log-scraping tooling that already
    # parses it.
    print(f"[GP] Compute resources resolved: "
          f"torch_num_threads={_compute_resources['torch_num_threads']}, "
          f"torch_dataloader_workers={_compute_resources['torch_dataloader_workers']}, "
          f"gpu_eval_batch={_compute_resources['gpu_eval_batch']}, "
          f"gpu_ld_r2={_compute_resources['gpu_ld_r2']}, "
          f"gpu_kernel_precompute={_compute_resources['gpu_kernel_precompute']}.")
    # ver4-4 Stage 6 (blueprint §10 Stage 6 checklist item 1 / §2.3.2 R3.f):
    # logged unconditionally (I11 - "resolved state logged unconditionally"),
    # its own separate line for the same log-scraping-stability reason as
    # the Stage 5 line above. See the HP_TUNE dispatch site further down
    # for a disclosed NOTE about what this flag can currently achieve
    # through this call site (genuine parallelism vs. a safe, logged
    # fallback to serial evaluation).
    print(f"[GP] Compute resources resolved: "
          f"hp_tune_parallel_trials={_compute_resources['hp_tune_parallel_trials']}.")
    # ver4-5 R1 (blueprint §3.6): logged unconditionally (I11), its own
    # separate line for the same log-scraping-stability reason as the
    # Stage 5/6 lines above. Batch/restart mode itself is announced
    # again, per-search, via the "[GP] NUMERICS: Bayesian hyperparameter
    # search running in BATCH mode..." line inside
    # models/hyperparameter_tuning.py::search_bayesian (§3.7) - this
    # line reports the RESOLVED SETTINGS that decide whether that will
    # ever fire, not a per-search event.
    print(f"[GP] Compute resources resolved: "
          f"hp_tune_bayes_batch={_compute_resources['hp_tune_bayes_batch']}, "
          f"hp_tune_bayes_batch_max={_compute_resources['hp_tune_bayes_batch_max']}, "
          f"hp_tune_bayes_liar={_compute_resources['hp_tune_bayes_liar']!r}, "
          f"hp_tune_parallel_restarts={_compute_resources['hp_tune_parallel_restarts']}.")
    # Update ID ver4-6 (blueprint §4/§5, I11 - "resolved state logged
    # unconditionally"): the eight new keys, on their own lines for the
    # same log-scraping-stability reason as every block above. RK-3/RK-4
    # (blueprint §7): several of these default to NEW, non-ver4-5-
    # equivalent behaviour (domain reduction now 'auto' not 'always',
    # weighted ensembles now floor- and analytic-seeded and objective_mode
    # defaults to 'ensemble_mse' - see main_app.py's own HYPERPARAM_OPT_
    # SPECS default and models/ensemble_regularization.py::
    # read_regularization_settings' own fallback) - printing the resolved
    # value and, where relevant, what restores the prior behaviour is
    # exactly what keeps that change from being a silent surprise.
    print(f"[GP] Compute resources resolved: "
          f"hp_tune_bayes_domain_reduction={_compute_resources['hp_tune_bayes_domain_reduction']!r}, "
          f"hp_tune_warm_start={_compute_resources['hp_tune_warm_start']} "
          f"(set 'always'/True respectively to restore ver4-5 behaviour), "
          f"hp_tune_selection_margin={_compute_resources['hp_tune_selection_margin']} "
          f"(0.0 restores the ver4-5 bare '>=' floor comparison), "
          f"hp_tune_valid_repeats={_compute_resources['hp_tune_valid_repeats']}, "
          f"hp_tune_scope={_compute_resources['hp_tune_scope']!r}"
          + (" (tuned parameters are reused across replicates within this shard; "
             "results are shard-dependent - set HP_TUNE_SCOPE=per_task for "
             "shard-independent results)" if _compute_resources['hp_tune_scope'] == 'per_scenario'
             else "") + ".")
    print(f"[GP] Compute resources resolved: "
          f"w_opt_analytic_seed={_compute_resources['w_opt_analytic_seed']}, "
          f"w_opt_validation_floor={_compute_resources['w_opt_validation_floor']}, "
          f"w_opt_simplex_search={_compute_resources['w_opt_simplex_search']} "
          f"(set every one of these False to restore ver4-5 weight-optimisation "
          f"behaviour byte-for-byte, alongside HYPERPARAMETERS_OPT's own "
          f"'dpt_ratio' objective - see §8 of the ver4-6 Change Summary).")

    # ver4-4 R3.d / risk RK-8 (blueprint §2.3.4, "This padding is mandatory
    # and is the single most likely I11 regression in R3"): GBLUP/RKHS each
    # gained two APPENDED HPARAM_SPECS entries this update
    # (shap_row_offset=0, shap_row_count=-1 - see hparam_specs.py and
    # models/GBLUP.R / models/RKHS.R). A HPARAMETERS['GBLUP']/['RKHS'] list
    # loaded from a config written before ver4-4 (7 and 8 elements
    # respectively) is therefore SHORTER than HPARAM_SPECS now expects.
    # Padded HERE, once, before any task reads HPARAMETERS - covers BOTH
    # _call_model() and the main per-task dispatch loop below for free,
    # since both read HPARAMETERS[base_model_name] from this SAME
    # (now-already-padded) dict; no separate lockstep edit is needed for
    # the padding itself (only for the Shapley fan-out logic that consumes
    # the two new fields - see _run_gblup_or_rkhs() below, applied in both
    # places per invariant I5/RK-2). Every appended value is each field's
    # own HPARAM_SPECS-documented default (never a hardcoded literal here),
    # so this can never drift from hparam_specs.py's own definition.
    # Logged unconditionally (I11 - "resolved state logged unconditionally")
    # so a silent list-length change is never invisible in the run log.
    for _padded_model_name in ('GBLUP', 'RKHS'):
        if _padded_model_name not in HPARAMETERS:
            continue
        _spec = HPARAM_SPECS[_padded_model_name]
        _entry = HPARAMETERS[_padded_model_name]
        # GAT_biological_prior_knowledge's own per-phenotype dict shape
        # (see §4.4 of the architecture doc / hparam_specs.py's own
        # module docstring) does not apply to GBLUP/RKHS - both are
        # always a single flat positional list - but guard defensively
        # anyway rather than assuming, since a malformed hand-edited
        # config could in principle hand this a non-list.
        if not isinstance(_entry, list):
            continue
        _n_before = len(_entry)
        if _n_before < len(_spec):
            _added_defaults = [field['default'] for field in _spec[_n_before:]]
            HPARAMETERS[_padded_model_name] = list(_entry) + _added_defaults
            print(f"[GP] NOTE: HPARAMETERS[{_padded_model_name!r}] was {_n_before} element(s) long "
                  f"(a config predating ver4-4 R3.d) - padded to {len(_spec)} with each new "
                  f"field's own documented default: {_added_defaults!r}.")

    # ver4-4 §4a (reproducibility policy - blueprint §4a/§10 Final
    # verification: "The run log contains exactly one [GP] NUMERICS: line
    # listing enabled non-bit-identical paths"): several accelerations
    # this update ships DEFAULT ON change a run's RESULTS at
    # floating-point-rounding level, not merely wall-clock time - unlike
    # every n_jobs/N_CPU_WORKERS-gated fan-out above (associative
    # summation/order-preserving concatenation, verified bit-identical),
    # these genuinely reorder floating-point operations. Per the standing
    # policy, they still ship enabled by default, but are announced here,
    # unconditionally, every run - including when every one of them is
    # off, in which case this says so explicitly - exactly mirroring the
    # existing LD-decay "log negative states unconditionally" convention
    # elsewhere in this function (see the LD_prune_effective/RF_filter_
    # effective NOTE above).
    #
    # PLOT_EFFECT_FLOAT32 (the blueprint's own §4a example NUMERICS line
    # includes it) is deliberately NOT listed here: it is consumed by
    # batch_reader.ResultSet.effect(), at PLOTTING time - run_sequential.py,
    # run_step2_assemble.py and main_app.py's in-process blocks each
    # resolve/consume it well after GP() has already returned (Parallel
    # Step 1's own GP() call, via run_step1_batch.py, never touches
    # plotting at all - PLOT_EFFECT_FLOAT32 isn't even meaningful yet at
    # that point, since Step 2 hasn't run). Threading a display-only,
    # plotting-time setting into GP() itself - which per the architecture
    # document's own Layer 4/Layer 3c boundary never touches
    # visualisation - would be a layering violation for a cosmetic-only
    # benefit. Every one of those three call sites ALREADY logs
    # PLOT_EFFECT_FLOAT32's own resolved value unconditionally, every run
    # (see e.g. run_sequential.py's own
    # "[run_sequential] PLOT_EFFECT_FLOAT32=..." print, added alongside
    # this same update) - so I11 ("resolved state logged unconditionally")
    # is satisfied for that key too, just under a different, more
    # accurate log prefix rather than a misleading claim that GP() itself
    # observed it.
    _RKHS_GEMM_NOTE = "RKHS dist() -> GEMM identity  (always on; float rounding differs from dist())"
    _numerics_optional_lines = []
    if _compute_resources['use_amp']:
        _numerics_optional_lines.append("USE_AMP=True                  (mixed precision in MLP/GAT training)")
    if _compute_resources['use_gpu_sklearn']:
        _numerics_optional_lines.append("USE_GPU_SKLEARN=True          (cuML forests; only active where cuML loaded)")
    if _compute_resources['gpu_kernel_precompute']:
        _numerics_optional_lines.append("GPU_KERNEL_PRECOMPUTE=True    (GBLUP/RKHS kernel built in float64, Python-side)")

    if _numerics_optional_lines:
        print("[GP] NUMERICS: the following accelerations are ENABLED and make this run's")
        print("    results NOT bit-identical to a run with them disabled:")
        for _numerics_line in _numerics_optional_lines:
            print(f"      {_numerics_line}")
        # Always on, unconditionally - not gated by any flag (blueprint
        # R4.h: "even with the flag off, RKHS.R:84's dist()... should be
        # replaced"). Listed alongside the optional flags above whenever
        # at least one of them is also on, exactly as the blueprint's own
        # §4a example NUMERICS line shows it.
        print(f"      {_RKHS_GEMM_NOTE}")
        print("    Differences are at floating-point rounding level, not method level. To")
        print("    reproduce a pre-ver4-4 run exactly, set each of these to false in the config.")
    else:
        # Every OPTIONAL flag is off - still printed unconditionally (I11):
        # silence here would be indistinguishable from "this run never
        # checked", exactly the failure mode the LD-decay reporting
        # convention this mirrors exists to avoid. RKHS dist()->GEMM is
        # always on regardless (it is not gated by any flag at all), so it
        # is still the one genuine exception even in the "nothing enabled"
        # case.
        print(f"[GP] NUMERICS: no non-bit-identical accelerations are enabled for this run "
              f"({_RKHS_GEMM_NOTE.strip()} is the sole exception).")
    print("    PLOT_EFFECT_FLOAT32 is a display-only accelerant resolved separately at")
    print("    plotting time (see this run's own 'PLOT_EFFECT_FLOAT32=...' log line from")
    print("    whichever script drives plotting) - GP() itself never touches plotting.")

    # GAT_biological_prior_knowledge always determines its own marker subset
    # from the gene-interaction network (see
    # models/GAT_biological_prior_knowledge.py) - there is no toggle for
    # this model itself; it never sees LD-pruned/RF-importance-filtered
    # data, unconditionally, regardless of what's enabled for other models.
    # The one exemption to this (requirement: "introduce an exemption...
    # when the new function is called") is internal to that model's own
    # data-driven prior-network merge feature (its params[13]) - see that
    # file's own module docstring; it does not change what GP() itself
    # routes to it here.
    #
    # OTHER_MODELS_MARKER_SOURCE is the model-independent choice for every
    # OTHER selected model instead:
    #   - 'full_or_filtered' (default): the original behaviour - other
    #     models use the full marker set, or the LD-pruned/RF-importance-
    #     filtered pool if either is enabled below.
    #   - 'gene_network': other models are instead restricted to the SAME
    #     markers GAT_biological_prior_knowledge selected via the gene
    #     network - LD pruning/RF filtering are not applied on top of this
    #     (this is an "instead of", not an additional narrowing step); see
    #     the per-task gene-window-marker computation further down. If more
    #     than one bio-prior instance is selected (see _is_bio_prior_model
    #     above), the *first* one (in MODEL's own order) is the reference
    #     network for this - a documented simplification, since "restrict
    #     other models to several different gene networks at once" has no
    #     single well-defined meaning.
    #   - 'gene_network_plus_rf' (requirement 6): like 'gene_network', but
    #     the marker pool other models are restricted to is the UNION of
    #     (a) markers included inside a gene node, and (b) markers selected
    #     by RF filtering with the top-M/top-Y% approach - i.e. exactly the
    #     marker set the reference bio-prior instance's own data-driven
    #     merge feature uses (requirement 2), whether or not any of those
    #     RF-selected markers actually ended up with a surviving data-driven
    #     edge after top_rate filtering. Requires that reference instance's
    #     own data-driven merge feature (params[13]['enabled']) to be
    #     turned on - validated per-task below, once that instance's own
    #     params for the current phenotype are known.
    # Requires GAT_biological_prior_knowledge to also be selected (its gene
    # network is what defines the marker set) - validated below.
    if OTHER_MODELS_MARKER_SOURCE not in ('full_or_filtered', 'gene_network', 'gene_network_plus_rf'):
        raise ValueError(
            f"OTHER_MODELS_MARKER_SOURCE must be 'full_or_filtered', 'gene_network', or "
            f"'gene_network_plus_rf', got {OTHER_MODELS_MARKER_SOURCE!r}"
        )
    if OTHER_MODELS_MARKER_SOURCE in ('gene_network', 'gene_network_plus_rf') and not _bio_prior_model_names(MODEL):
        raise ValueError(
            f"OTHER_MODELS_MARKER_SOURCE={OTHER_MODELS_MARKER_SOURCE!r} requires "
            f"GAT_biological_prior_knowledge to also be selected in MODEL - its gene-interaction "
            f"network is what defines the marker set every other model would be restricted to."
        )

    _other_models_selected = any(m != 'ensemble' and not _is_bio_prior_model(m) for m in MODEL)
    # LD pruning / RF importance filtering only ever apply to the
    # full/filtered pool - so they're skipped entirely (not just ignored)
    # whenever nothing would actually use that pool: either no other model
    # is selected at all, or every other model has been redirected to the
    # gene-network marker set instead.
    _other_models_use_full_pool = _other_models_selected and OTHER_MODELS_MARKER_SOURCE == 'full_or_filtered'
    LD_prune_effective = LD_prune if _other_models_use_full_pool else None
    RF_filter_effective = RF_filter if _other_models_use_full_pool else None

    # Diagnostic (prevent this from happening silently): LD_prune/RF_filter
    # were configured by the caller but end up doing NOTHING for this run,
    # because of the model-routing logic just above rather than because LD
    # pruning/RF filtering/the LD decay plot were themselves turned off.
    # Without this line, a user who enabled LD pruning (and, per this
    # feature, the LD decay plot) would see zero LD-pruning-related output
    # anywhere - no '[GP] Task .. | LD pruning ...' lines, no LD decay
    # plots, no explanation - and average_and_plot_ld_decay() would later
    # report '0 average LD decay plot(s) generated' with nothing upstream
    # in the log to explain why.
    if (LD_prune is not None or RF_filter is not None) and not _other_models_use_full_pool:
        _why = (
            "no model other than GAT_biological_prior_knowledge/ensemble is selected in MODEL"
            if not _other_models_selected else
            f"OTHER_MODELS_MARKER_SOURCE={OTHER_MODELS_MARKER_SOURCE!r} routes every other "
            f"selected model to the gene-network marker pool instead of the full/filtered one"
        )
        print(f"[GP] NOTE: LD pruning/RF filtering/LD decay plot were configured for this run, "
              f"but will have NO effect - {_why}, so there is no 'full/filtered marker pool' "
              f"for them to apply to. This is expected if that's what you intended; if not, "
              f"either select another model, or set OTHER_MODELS_MARKER_SOURCE back to "
              f"'full_or_filtered'.")

    # Create the output directory up front - the R model functions (rrBLUP/GBLUP/BayesB/RKHS)
    # and the CSV writers below all assume './Result/<RESULT_NAME>/' already exists
    os.makedirs(result_dir_path(RESULT_NAME), exist_ok=True)

    # ---------------------------------------------------------------------- #
    # Optional LD decay plot (Preprocess/LD_decay_plot.py) - an add-on
    # diagnostic for the LD pruning step above, entirely opt-in via
    # LD_prune['decay_plot']. Only ever considered when LD pruning itself is
    # effectively enabled (LD_prune_effective is not None) - a decay plot
    # needs the same pre-pruning training-set genotypes LD pruning itself
    # works from for a given task, so it only makes sense alongside LD
    # pruning being enabled for that same task.
    #
    # A decay curve can be requested for any subset of the three window
    # units Preprocess.LD_pruning itself supports ('kb', 'cm', 'variants') -
    # LD_prune['decay_plot']['window_units'] is a LIST, independent of
    # whatever window_unit LD pruning itself is actually using at the
    # config-schema level, so e.g. a run pruning with window_unit='kb'
    # could in principle still ask for cm- and variant-based decay curves
    # too. The GUI (main_app.py's build_ld_decay_plot_config()) always
    # sets this to a single-element list matching LD pruning's own
    # 'Window unit' setting exactly - a decay plot's whole point is to
    # help judge the window/r^2 threshold LD pruning is actually using,
    # which only makes sense expressed in that same unit - but a
    # hand-built config (headless use) can still request a different, or
    # multiple, unit(s) if that's genuinely useful for a particular
    # analysis.
    # ---------------------------------------------------------------------- #
    _decay_cfg = LD_prune_effective.get('decay_plot') if LD_prune_effective is not None else None
    _decay_plot_enabled = bool(_decay_cfg and _decay_cfg.get('enabled'))

    # Diagnostic (prevent this from happening silently): report the LD
    # decay plot's resolved on/off state UNCONDITIONALLY, every run - not
    # just when it ends up enabled - so it's always immediately obvious
    # from the log whether/why it will run, rather than requiring anyone to
    # infer it from the absence of other messages. This specifically covers
    # the case of a sequential_config.json/step1_config.json/
    # step2_config.json saved (e.g. via the GUI's 'Generate and save job
    # files') BEFORE this feature existed: such a file's LD_PRUNE dict has
    # no 'decay_plot' key at all, which is otherwise indistinguishable from
    # simply never having asked for the feature - regenerating/resaving the
    # config from the current GUI (or just using 'Run pipeline' directly,
    # which always gathers a fresh config) is required to pick it up.
    if LD_prune is None:
        print("[GP] LD decay plot: not requested (LD pruning itself is not configured for "
              "this run - LD_prune=None).")
    elif LD_prune_effective is None:
        pass  # already explained above by the 'NOTE: LD pruning/RF filtering/LD decay plot ...' message
    elif _decay_cfg is None:
        print("[GP] LD decay plot: LD pruning is configured, but its config has no "
              "'decay_plot' entry at all - the LD decay plot will NOT be generated. If you "
              "expected it to run: a config saved via 'Generate and save job files' BEFORE "
              "this feature was added to the GUI won't have this key - regenerate/resave that "
              "config file (or use 'Run pipeline' directly, which always builds a fresh "
              "config) after checking 'Generate LD decay plots' on the LD pruning tab.")
    elif not _decay_cfg.get('enabled'):
        print("[GP] LD decay plot: configured but not enabled (LD_prune['decay_plot']"
              "['enabled'] is False) - the LD decay plot will NOT be generated for this run.")

    _decay_snp_info = None
    _decay_window_units = []
    if _decay_plot_enabled:
        # Backward compatible with a config saved by an earlier, kb-only
        # version of this feature (which had no 'window_units' key at
        # all) - default to ['kb'] in that case, exactly matching that
        # version's only-ever-kb behaviour.
        _requested_units = _decay_cfg.get('window_units') or ['kb']
        _invalid_units = [u for u in _requested_units if u not in LD_DECAY_WINDOW_UNITS]
        _requested_units = [u for u in _requested_units if u in LD_DECAY_WINDOW_UNITS]
        if _invalid_units:
            print(f"[GP] LD decay plot: ignoring invalid window unit(s) {_invalid_units} in "
                  f"'decay_plot.window_units' - must be a subset of {LD_DECAY_WINDOW_UNITS}.")
        if not _requested_units:
            print("[GP] LD decay plot: 'window_units' has no valid entries - the LD decay "
                  "plot will NOT be generated for this run.")
            _decay_plot_enabled = False
        else:
            _decay_snp_info = resolve_snp_info_for_decay(LD_prune_effective)
            _needs_snp_info = [u for u in _requested_units if u in ('kb', 'cm')]
            if _needs_snp_info and _decay_snp_info is None:
                print(f"[GP] LD decay plot: no SNP info (CHR/POS/CM map) is configured for LD "
                      f"pruning, so window unit(s) {_needs_snp_info} can't be computed (they "
                      f"need real marker positions/genetic distances) - 'variants' doesn't "
                      f"need one, but {_needs_snp_info} do.")
                _requested_units = [u for u in _requested_units if u not in ('kb', 'cm')]
            _decay_window_units = _requested_units
            _decay_plot_enabled = bool(_decay_window_units)
            if not _decay_plot_enabled:
                print("[GP] LD decay plot: no window unit(s) remain usable for this run "
                      "(see message(s) above) - the LD decay plot will NOT be generated.")

    if _decay_plot_enabled:
        _decay_frequency = max(1, int(_decay_cfg.get('frequency', 1)))
        _decay_subfolder = _decay_cfg.get('subfolder') or 'LD_decay_plots'
        _decay_max_pairs_per_chr = int(_decay_cfg.get('max_pairs_per_chr', 20000))
        # Per-window-unit max_distance/bin_width - e.g.
        # decay_plot['kb'] = {'max_distance':.., 'bin_width':..}. Falls
        # back to Preprocess.LD_decay_plot's own documented defaults for
        # any unit whose settings weren't explicitly configured (also
        # covers a kb-only legacy config, which has no 'kb'/'cm'/
        # 'variants' sub-dicts at all - see 'window_units' fallback
        # above - by falling back to the OLD flat
        # max_distance_kb/bin_width_kb keys first, then the module's
        # defaults, so an old saved config's kb settings still apply
        # unchanged).
        _decay_settings = {}
        for _unit in _decay_window_units:
            _unit_cfg = _decay_cfg.get(_unit) or {}
            _legacy_max = _decay_cfg.get(f'max_distance_{_unit}')
            _legacy_bin = _decay_cfg.get(f'bin_width_{_unit}')
            _decay_settings[_unit] = {
                'max_distance': float(_unit_cfg.get('max_distance', _legacy_max) or LD_DECAY_DEFAULT_MAX_DISTANCE[_unit]),
                'bin_width': float(_unit_cfg.get('bin_width', _legacy_bin) or LD_DECAY_DEFAULT_BIN_WIDTH[_unit]),
            }
        _decay_plot_dir = os.path.join(result_dir_path(RESULT_NAME), _decay_subfolder)
        _decay_data_dir = os.path.join(_decay_plot_dir, 'data')
        os.makedirs(_decay_data_dir, exist_ok=True)
        # 'Keep the per-scenario LD decay data CSVs after averaging, or
        # discard them once the average(s) have been computed to save
        # disk space?' - see Preprocess.LD_decay_plot's module docstring
        # ('KEEPING OR DISCARDING THE PER-SCENARIO "LOG"'). Recorded to a
        # tiny marker file (not just kept in this function's local
        # variables) because the actual cleanup happens in
        # average_and_plot_ld_decay(), which for the Parallel workflow
        # runs from Step 2 - a separate process that never sees this
        # LD_PRUNE config at all. Default True (keep everything) so an
        # older saved config with no 'keep_log' key behaves exactly as
        # before this option existed.
        _decay_keep_log = bool(_decay_cfg.get('keep_log', True))
        write_keep_log_marker(_decay_plot_dir, _decay_keep_log)
        print(f"[GP] LD decay plot enabled for window unit(s) {_decay_window_units}: a plot "
              f"will be generated every {_decay_frequency} scenario(s) that LD pruning runs "
              f"for, saved under '{_decay_plot_dir}' "
              f"({'keeping' if _decay_keep_log else 'discarding after averaging'} the "
              f"per-scenario data CSVs).")

    # Requirement: LD decay is a property of the genotypes (which
    # individuals end up in a scenario's training set), not of the
    # phenotype being predicted - GP()'s own train_test_split calls below
    # are keyed only on population/ratio/replicate ('sample'), never on
    # phenotype, so every phenotype sharing a given (population, ratio,
    # replicate) combination has an IDENTICAL training-set genotype and
    # would produce an identical decay curve. A decay sample is therefore
    # taken (or, per the configured frequency, deliberately skipped) at
    # most ONCE per (population, ratio, replicate) combination - never
    # re-sampled for a second, third, ... phenotype that happens to share
    # it - both cutting the amount of data stored and making
    # Preprocess.LD_decay_plot.average_and_plot_ld_decay()'s later
    # per-population averaging correctly reflect only genuinely distinct
    # genotype samples (see that module's own docstring).
    _ld_decay_seen_keys = set()
    _ld_decay_call_count = 0

    def _ld_decay_due(key):
        """Whether THIS (population, ratio, replicate) `key` should have
        an LD decay sample generated. Decided ONCE per key - every
        subsequent task sharing that same key (necessarily a different
        phenotype - population/ratio/replicate together already uniquely
        identify a task within a single phenotype) always returns False,
        since the training-set genotypes (and therefore the decay curve)
        would be identical to whatever was already sampled - or
        deliberately skipped, per the configured frequency - the first
        time this key was seen. Kept separate from the actual plot-
        generation work below so the (potentially expensive, for the
        native-PLINK path - see that branch's own call site) pre-pruning
        genotype materialisation only ever happens on calls that are
        actually due."""
        nonlocal _ld_decay_call_count
        if not _decay_plot_enabled:
            return False
        if key in _ld_decay_seen_keys:
            return False
        _ld_decay_seen_keys.add(key)
        _ld_decay_call_count += 1
        return _ld_decay_call_count % _decay_frequency == 0

    def _save_ld_decay_plot(genotype_supplier, population_label, sample_label, ratio_label):
        """Requirements 1-5: compute this (population, ratio, replicate)
        combination's LD decay curve(s) - one per requested window unit
        (kb/cm/variants) - from its own (pre-pruning) training-set
        genotypes, and save both the plot (PNG) and the data used to draw
        it (CSV) under Result/<RESULT_NAME>/<subfolder>/ for each. Only
        ever called once _ld_decay_due() has already returned True for
        this task.

        genotype_supplier : zero-arg callable returning the PRE-pruning
            training-set genotype DataFrame (marker columns only) for this
            scenario - called lazily, right here, ONCE (reused for every
            requested window unit), so the native-PLINK path's extra
            genotype materialisation (see that branch's own comment) only
            ever happens once per due scenario, however many window units
            were requested.

        Requirement 7 ("work without errors, sequential and parallel"):
        never raises - a diagnostic plot failing to render must never take
        down an entire prediction run. Any error (including materialising
        the genotype itself) is caught, logged, and this task's actual
        prediction work continues exactly as if the plot had simply been
        skipped.
        """
        try:
            pre_prune_genotype_df = genotype_supplier()
        except Exception as exc:
            print(f"[GP] WARNING: could not materialise genotypes for the LD decay plot "
                  f"(population={population_label}, sample={sample_label}): {exc!r}. "
                  f"Continuing without it.")
            return

        for window_unit in _decay_window_units:
            try:
                settings = _decay_settings[window_unit]
                decay_df = compute_ld_decay_data(
                    pre_prune_genotype_df, _decay_snp_info, window_unit=window_unit,
                    max_distance=settings['max_distance'], bin_width=settings['bin_width'],
                    max_pairs_per_chr=_decay_max_pairs_per_chr,
                )
                # Requirement 4: filename combines population + sampling
                # number (plus the window unit, so kb/cm/variants curves
                # for the same combination never collide on one name) -
                # NOT phenotype, since this sample already represents
                # every phenotype sharing this (population, ratio,
                # replicate) combination (see the dedup note above), not
                # any one specific phenotype. unique_path() guards the
                # rare case where population + sample still doesn't
                # uniquely identify a single combination on its own (e.g.
                # more than one split RATIO configured for the 'within'
                # scenario reuses the same replicate numbers for every
                # ratio) rather than silently overwriting an earlier
                # combination's files.
                name_stub = (
                    f"LDdecay_{window_unit}_{sanitize_for_filename(population_label)}_"
                    f"{sanitize_for_filename(sample_label)}"
                )
                csv_path = unique_path(os.path.join(_decay_data_dir, f'{name_stub}.csv'))
                png_path = unique_path(os.path.join(_decay_plot_dir, f'{name_stub}.png'))
                # Requirement 6: window_unit/population metadata is what
                # lets average_and_plot_ld_decay() (called once, after
                # every prediction scenario has finished - see
                # run_sequential.py / run_step2_assemble.py) group this
                # combination's data with every other replicate/ratio
                # sharing the same window_unit+population (and never mix
                # it with a different window unit's or population's data).
                save_ld_decay_data(decay_df, csv_path, metadata={
                    'window_unit': window_unit, 'population': population_label,
                    'ratio': ratio_label, 'sample': sample_label,
                })
                plotted = plot_ld_decay(
                    decay_df, png_path,
                    title=(f'LD decay ({window_unit}) - population={population_label}, '
                           f'sample={sample_label}'),
                    window_unit=window_unit,
                )
                if plotted:
                    print(f"[GP] LD decay plot ({window_unit}) saved: {png_path} (data: {csv_path})")
                else:
                    print(f"[GP] LD decay plot ({window_unit}) skipped for this scenario (no "
                          f"marker pair within {settings['max_distance']} {window_unit} had both "
                          f"a usable position) - the (empty) data file was still written to "
                          f"{csv_path} for the record.")
            except Exception as exc:
                print(f"[GP] WARNING: LD decay plot ({window_unit}) generation failed for this "
                      f"scenario (population={population_label}, sample={sample_label}): "
                      f"{exc!r}. Continuing without it.")

    # PLINK2 subprocess working files (Preprocess/plink_io.py's genotype
    # conversion/native LD pruning) are NOT persisted anywhere in the
    # Result folder - every call below omits work_dir (or passes None),
    # which is each function's documented signal to create its own
    # temporary directory and clean it up automatically once that call
    # finishes. Nothing under Result/<RESULT_NAME>/ is used for this.
    def _new_plink_work_dir():
        return None

    # Import R modules
    if R_PATH != None:
        os.environ['R_HOME'] = R_PATH
    
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    
    r_source = robjects.r['source']
    # NOTE: these MUST be resolved relative to this module's own location
    # (_R_MODELS_DIR, computed once at import time from __file__), never to
    # the process's current working directory. A bare './models/rrBLUP.R'
    # only sources successfully when GP() happens to be invoked from the
    # EasiGP install directory - true for an interactive Streamlit session,
    # but NOT guaranteed for a headless HPC run: PBS/Slurm array jobs start
    # in whatever directory the scheduler resolves (submission directory,
    # $PBS_O_WORKDIR, $SLURM_SUBMIT_DIR, ...), which need not be the install
    # directory at all. Sourcing via an absolute, __file__-derived path
    # makes GP() work identically regardless of caller cwd - this is what
    # fixed the "cannot open file './models/rrBLUP.R'" crash seen on
    # parallel-mode HPC batches.
    r_source(os.path.join(_R_MODELS_DIR, 'rrBLUP.R'))
    rrBLUP = robjects.globalenv['rrBLUP']
    r_source(os.path.join(_R_MODELS_DIR, 'GBLUP.R'))
    GBLUP = robjects.globalenv['GBLUP']
    r_source(os.path.join(_R_MODELS_DIR, 'BayesB.R'))
    BayesB = robjects.globalenv['BayesB']
    r_source(os.path.join(_R_MODELS_DIR, 'RKHS.R'))
    RKHS = robjects.globalenv['RKHS']

    def _run_gblup_or_rkhs(r_fn, model_name, train, valid, test, params, result_name,
                            k_precomputed=None):
        """Drop-in replacement for a direct ``r_fn(train, valid, test,
        params, result_name)`` call (``r_fn`` is the ``GBLUP``/``RKHS``
        bound R function object above) - called from BOTH
        ``_call_model()`` and the main per-task dispatch loop below, in
        lockstep (invariant I3/RK-2 - see this module's own docstring for
        why those two call sites must never drift apart).

        Returns EXACTLY what ``r_fn(train, valid, test, params,
        result_name, K_precomputed=k_precomputed)`` would return whenever
        no fan-out actually happens - i.e. the raw rpy2 result object,
        completely unchanged - which is every case except
        ``get_effect=True`` AND this run's own resolved ``n_jobs`` names
        more than one usable worker AND there are at least 2 explainable
        rows: the overwhelmingly common ``N_JOBS<=1`` (today's default)
        case is therefore byte-for-byte today's exact code path, just
        with one extra (never-true) condition check.

        When fan-out DOES happen, splits the ``[0, len)`` Shapley row
        range across ``n_jobs`` PROCESSES (never threads - I10) via
        ``_gblup_rkhs_shapley_row_worker()`` (module-level, above - see
        its own extensive docstring for why this can't simply share the
        already-bound ``r_fn``/live rpy2 session with the workers), sums
        every worker's own partial ``colSums(abs(...))`` effect vector
        (linear/associative over row-disjoint partitions - ver4-4 Stage 5
        hand-off note §5.4), and returns a PLAIN Python dict carrying the
        SAME ``'r_pearson'``/``'r_MSE'``/``'r_effect'``/
        ``'r_y_predicted'``/``'r_y_predicted_valid'``/
        ``'r_y_predicted_train'`` keys a raw R result has -
        ``pipeline_utils.r_list_get()``'s own Strategy 1 (plain
        ``__getitem__`` access) handles a plain dict exactly like it
        already handles an ``OrdDict``, so every EXISTING downstream
        ``r_list_get(result, ...)`` call at both call sites works
        completely UNCHANGED regardless of which shape actually comes
        back - this is the deliberate, minimal resolution of the rpy2-
        shape ambiguity the ver4-4 Stage 5 hand-off note (§6.3/§8) left
        unresolved, achieved by never letting a live rpy2 object cross a
        process boundary at all rather than by trying to mutate one.

        Falls back to a single, unsplit call - identical to the
        ``n_jobs<=1`` path above - on ANY exception raised while
        attempting fan-out (unpicklable input, a worker process dying,
        ...), so a parallelisation failure never aborts a task that a
        plain serial call would have completed successfully.
        """
        params = list(params)
        if model_name == 'GBLUP':
            get_effect_idx, shapley_num_idx = 2, 3
        elif model_name == 'RKHS':
            get_effect_idx, shapley_num_idx = 3, 4
        else:
            raise ValueError(f"_run_gblup_or_rkhs only supports 'GBLUP'/'RKHS', got {model_name!r}")

        r_kwargs = {}
        if k_precomputed is not None:
            r_kwargs['K_precomputed'] = k_precomputed

        get_effect = bool(params[get_effect_idx])
        _raw_n_jobs = _compute_resources['n_jobs']
        if not get_effect or not _raw_n_jobs or _raw_n_jobs == 1:
            return r_fn(train, valid, test, params, result_name, **r_kwargs)

        shapley_num = params[shapley_num_idx]
        len_rows = min(test.shape[0], shapley_num) if shapley_num != 'all' else test.shape[0]
        if len_rows <= 1:
            # Nothing meaningful to split (0 or 1 explainable row) -
            # falls through to the single, unsplit call, today's exact
            # behaviour for this edge case.
            return r_fn(train, valid, test, params, result_name, **r_kwargs)

        # -1 ("every core", sklearn's own convention, and what
        # resolve_compute_resources()'s own n_jobs can legitimately
        # resolve to) is meaningful to joblib.Parallel(n_jobs=...)
        # directly but is NOT a usable literal CHUNK COUNT for splitting
        # the row range below - resolve a separate, always-positive
        # chunk count for that purpose only, mirroring
        # Preprocess.LD_decay_plot._decay_pairs_for_group's identical
        # -1-sentinel handling exactly.
        _chunk_count = int(_raw_n_jobs) if int(_raw_n_jobs) > 0 else (os.cpu_count() or 1)
        n_workers = min(_chunk_count, len_rows)
        if n_workers <= 1:
            return r_fn(train, valid, test, params, result_name, **r_kwargs)

        row_bounds = [b for b in np.array_split(np.arange(len_rows), n_workers) if len(b) > 0]

        try:
            from joblib import Parallel, delayed
            worker_results = Parallel(n_jobs=_raw_n_jobs)(
                delayed(_gblup_rkhs_shapley_row_worker)(
                    model_name, R_PATH, train, valid, test, params, result_name,
                    k_precomputed, int(bound[0]), int(len(bound)),
                )
                for bound in row_bounds
            )
        except Exception as exc:
            print(f"[GP] NOTE: parallel Shapley row-range fan-out for {model_name} failed "
                  f"({exc!r}) - falling back to a single, unsplit call (n_jobs<=1 behaviour). "
                  f"This task's own result is unaffected, only wall-clock time.")
            return r_fn(train, valid, test, params, result_name, **r_kwargs)

        # joblib.Parallel always returns results in call-SUBMISSION order
        # (never completion order), and row_bounds[0] is always the
        # offset=0 chunk (np.array_split on an ascending arange never
        # reorders) - so worker_results[0] is unconditionally the
        # offset=0 worker's own result, matching this function's own
        # docstring ("the offset=0 worker's OWN complete, independent
        # BGLR fit").
        base_result = dict(worker_results[0])
        try:
            # See _coerce_effect_values()'s own module-level note above
            # (Requirements.md item 1) for why every worker's r_effect is
            # coerced to a plain numeric vector before summing, rather
            # than summed via each worker's own object's '+' directly -
            # two independent rpy2 worker sessions can hand back
            # DIFFERENT Python shapes (DataFrame vs. OrdDict/dict) for
            # the identical R-side computation, and this makes the
            # summation immune to that regardless of which shapes any
            # given run happens to hit.
            marker_columns = list(train.columns[:-1])
            summed_values = np.zeros(len(marker_columns), dtype=float)
            for worker_result in worker_results:
                this_effect = r_list_get(worker_result, 'r_effect')
                summed_values += _coerce_effect_values(this_effect, len(marker_columns))
            base_result['r_effect'] = pd.DataFrame([summed_values], columns=marker_columns)
        except Exception as exc:
            print(f"[GP] NOTE: could not aggregate {model_name} Shapley row-range "
                  f"fan-out results across {len(row_bounds)} worker process(es) "
                  f"({exc!r}) - falling back to a single, unsplit call (n_jobs<=1 "
                  f"behaviour). This task's own result is unaffected, only "
                  f"wall-clock time (the fan-out's own work is discarded and "
                  f"redone serially).")
            return r_fn(train, valid, test, params, result_name, **r_kwargs)
        print(f"[GP] {model_name} Shapley row-range fan-out: {len_rows} row(s) split across "
              f"{len(row_bounds)} worker process(es) (n_jobs={_raw_n_jobs}).")
        return base_result

    def _call_model(model_name, train, valid, test, params, bio_prior_ctx=None):
        """Thin adapter (ver4-5 R1.a - blueprint \u00a73.4(iii)): delegates to
        ModelTrialRunner, a module-level, PICKLABLE twin of this
        function's own former body (moved verbatim - see
        ModelTrialRunner's own docstring above). Kept as a nested
        function purely as a convenience entry point for any FUTURE
        in-process caller within this GP() call that wants "call one
        model by base name, get the uniform result dict back" without
        constructing its own ModelTrialRunner - it is not currently
        called anywhere except historically (the hyperparameter-tuning
        call site below now constructs and passes a ModelTrialRunner
        directly, per the blueprint, rather than wrapping this function
        in a lambda - see that call site's own comment for why: a
        lambda wrapping this closure would itself be an unnecessary,
        if-now-harmless, extra layer of indirection).

        LOCKSTEP WARNING (unchanged from before, re-pointed): every one
        of ModelTrialRunner's twelve model branches must be kept in sync
        with the main per-task dispatch loop's own per-model branches
        further down GP() - the two are still a separate implementation
        of "how to call a model", by design (see the main dispatch
        loop's own comment), not a refactor of it. This function itself
        never needs editing when a model signature changes - only
        ModelTrialRunner does - since it does no work beyond
        constructing a runner and forwarding.

        bio_prior_ctx : dict or None - see ModelTrialRunner's own
            docstring; forwarded unchanged into the runner's own
            constructor.
        """
        runner = ModelTrialRunner(RESULT_NAME, R_PATH, _compute_resources, GENOTYPE_FORMAT,
                                   bio_prior_ctx=bio_prior_ctx)
        return runner(model_name, train, valid, test, params, RESULT_NAME)



    # Read genotype and phenotype data.
    #
    # GENOTYPE_FORMAT == 'csv' (default): completely unchanged from before -
    # the whole genotype CSV is loaded up front, exactly as it always has
    # been.
    #
    # GENOTYPE_FORMAT == 'plink': GENOTYPE_FILE_NAME is a PLINK bed/bim/fam
    # file STEM (e.g. 'mydata' for 'mydata.bed'/'mydata.bim'/'mydata.fam'),
    # not a CSV path. The genotype matrix itself is *never* loaded here -
    # only the .bim file's marker positions (cheap: no genotype values
    # touched at all) - so a run with only GAT_biological_prior_knowledge
    # selected never has to materialise more than the specific markers that
    # actually fall inside a gene's window, and a run with LD pruning
    # enabled never has to materialise the pre-pruning genotype matrix at
    # all. Both 'ID' and 'population' come from the phenotype file (a .fam
    # file has neither) - see Preprocess/plink_io.py's module docstring.
    # The actual per-sample, per-marker genotype DataFrames are built
    # per-task, inside the main loop below.
    data_phenotype_original = pd.read_csv(PHENOTYPE_FILE_NAME)
    # Requirement 8: the phenotype file's first two columns are always
    # 'ID' and 'population' BY POSITION, regardless of what header text
    # the file actually uses for them - unify internally so the rest of
    # GP() can always rely on those exact names. Every column AFTER
    # these two is left completely untouched - those are phenotype TRAIT
    # names, which are meaningful, user-chosen values (e.g. 'days2anthesis')
    # that must never be renamed.
    data_phenotype_original = unify_columns_by_position(
        data_phenotype_original, ['ID', 'population'], 'phenotype file', first_n=2
    )
    # Normalise 'ID' to string dtype right away. Preprocess/plink_io.py
    # always returns string IDs (a .fam file's IID column is read as text),
    # so for GENOTYPE_FORMAT == 'plink', a phenotype file whose 'ID' column
    # happens to look purely numeric (e.g. 1001, 1002) would otherwise be
    # read by pandas as int64 - merging that against plink_io's string IDs
    # then fails with "You are trying to merge on object and int64 columns"
    # (pandas refuses to silently coerce). Casting both sides to string here
    # - unconditionally, for both genotype formats, so the CSV path stays
    # internally consistent with itself too - makes ID matching robust to
    # this regardless of what the IDs happen to look like, with no change
    # in which samples match (a numeric ID and its string form identify the
    # exact same sample either way).
    data_phenotype_original['ID'] = data_phenotype_original['ID'].astype(str)

    if GENOTYPE_FORMAT == 'csv':
        data_genotype_original = pd.read_csv(GENOTYPE_FILE_NAME)
        # Requirement 8: same positional unification as the phenotype file
        # above - 'ID'/'population' by position, every marker column after
        # them left exactly as-is (marker names are meaningful, user-chosen
        # identifiers, never renamed).
        data_genotype_original = unify_columns_by_position(
            data_genotype_original, ['ID', 'population'], 'genotype file', first_n=2
        )
        data_genotype_original['ID'] = data_genotype_original['ID'].astype(str)
        POPULATION = pd.unique(data_genotype_original['population'])
        bim_marker_info = None
        fam_iid_set = None
    elif GENOTYPE_FORMAT == 'plink':
        validate_plink_fileset(GENOTYPE_FILE_NAME)
        data_genotype_original = None
        POPULATION = pd.unique(data_phenotype_original['population'])
        bim_marker_info = read_bim_marker_info(f"{GENOTYPE_FILE_NAME}.bim")
        # Requirement: merge genotype and phenotype BEFORE splitting into
        # train/valid/test, not per-split afterwards - see
        # _process_one_task()'s own use of this, right after its
        # phenotype-NaN dropna. Reading just the .fam file's ID column
        # (not the - potentially far larger - genotype matrix itself) is
        # cheap enough to do once, up front, and reuse for every task.
        fam_iid_set = set(read_fam_iids(f"{GENOTYPE_FILE_NAME}.fam"))
    else:
        raise ValueError(f"GENOTYPE_FORMAT must be 'csv' or 'plink', got {GENOTYPE_FORMAT!r}")

    # GAT_biological_prior_knowledge's own marker_info_path (whatever the
    # user configured in its hyperparameters) is only meaningful for
    # GENOTYPE_FORMAT == 'csv'. For 'plink', marker positions can only
    # correctly come from the SAME .bim file that any extracted genotype
    # columns were pulled from - using a different, hand-supplied
    # marker_info.csv here would silently desynchronise marker names from
    # their true positions. So for 'plink', the .bim-derived marker_info is
    # written once - shared by every bio-prior instance, since they all
    # read from the same underlying .bim regardless of which network each
    # is paired with - and every GAT_biological_prior_knowledge* call this
    # run has its own params[9] (marker_info_path) OVERRIDDEN to point at
    # it, regardless of whatever was configured in the GUI - see the
    # dispatch branch below.
    bim_marker_info_path = None
    if GENOTYPE_FORMAT == 'plink' and _bio_prior_model_names(MODEL):
        bim_marker_info_path = os.path.join(result_dir_path(RESULT_NAME), '_plink_bim_marker_info.csv')
        os.makedirs(os.path.dirname(bim_marker_info_path), exist_ok=True)
        bim_marker_info.to_csv(bim_marker_info_path, index=False, encoding='utf-8')

    if (type(PHENOTYPE) is not list) and (PHENOTYPE == 'all'):
        PHENOTYPE = list(data_phenotype_original.columns[2:])
    
    # Create the total number of combinations of prediction scenarios
    if SCENARIO == 'within':
        sample = pd.DataFrame({'population':[item for item in POPULATION for i in range(SAMPLE_NUM*len(PHENOTYPE)*len(RATIO))],
                                'phenotype':[item for item in PHENOTYPE for i in range(SAMPLE_NUM*len(RATIO))]*len(POPULATION),
                                'ratio':[item for item in RATIO for i in range(SAMPLE_NUM)]*len(POPULATION)*len(PHENOTYPE),
                                'sample':list(range(1,SAMPLE_NUM+1)) *len(PHENOTYPE)*len(POPULATION)*len(RATIO)})
    elif SCENARIO == 'between':
        if W_OPT is None:
            comb = [str(x)+'->'+str(y) for x in POPULATION for y in POPULATION]
            sample = pd.DataFrame({'population':[item for item in comb] *len(PHENOTYPE),
                                    'phenotype':[item for item in PHENOTYPE for i in range(len(POPULATION))]*len(POPULATION),
                                    'ratio':[-1]*len(POPULATION)*len(PHENOTYPE)*len(POPULATION),
                                    'sample':[-1] *len(PHENOTYPE)*len(POPULATION)*len(POPULATION)})
            tmp = sample['population'].str.split('->',expand=True)
            sample = sample[tmp[0]!=tmp[1]].reset_index(drop=True)
        else:
            comb = [str(x)+'->'+str(y)+'->'+str(z) for x in POPULATION for y in POPULATION for z in POPULATION]
            sample = pd.DataFrame({'population':[item for item in comb] *len(PHENOTYPE),
                                    'phenotype':[item for item in PHENOTYPE for i in range(len(POPULATION))]*len(POPULATION)*len(POPULATION),
                                    'ratio':[-1]*len(POPULATION)*len(PHENOTYPE)*len(POPULATION)*len(POPULATION),
                                    'sample':[-1] *len(PHENOTYPE)*len(POPULATION)*len(POPULATION)*len(POPULATION)})
            tmp = sample['population'].str.split('->',expand=True)
            sample = sample[(tmp[0]!=tmp[1]) & (tmp[0]!=tmp[2]) & (tmp[1]!=tmp[2])].reset_index(drop=True)

    record = pd.DataFrame()          #store performance metrics
    result_train = pd.DataFrame()    #store predicted phenotypes for train set
    result_valid = pd.DataFrame()    #store predicted phenotypes for validation set
    result_test = pd.DataFrame()     #store predicted phenotypes for test set
    effect = pd.DataFrame()          #store genomic marker effects
    interactions = pd.DataFrame()    #store marker interaction effects
    weight = pd.DataFrame()          #store weight values for weight optimisation
    attention_total = pd.DataFrame() #store attention values  GAT models
    hp_record = pd.DataFrame()       #store optimised hyperparameters found via HP_TUNE
    stats = pd.DataFrame()           #store basic per-scenario statistics (split sizes, marker count)

    # MODEL_BASE is the user's original, unsuffixed model selection - used
    # for everything that must key off a model's TYPE regardless of tuning
    # (HP_TUNE lookups, the ensemble-grouping helpers). MODEL_RUN is what the
    # per-task dispatch loop below actually iterates: identical to
    # MODEL_BASE unless a model is tuned with more than one algorithm, in
    # which case it's expanded into one suffixed entry per algorithm (e.g.
    # 'RF' -> 'RF__Grid', 'RF__Bayesian') - see
    # models.hyperparameter_tuning.expand_model_list. When HP_TUNE is None
    # (the default), MODEL_RUN == MODEL_BASE exactly, so every existing
    # caller that doesn't pass HP_TUNE sees no change here at all.
    MODEL_BASE = MODEL
    MODEL_RUN = expand_model_list(MODEL_BASE, HP_TUNE)

    # Update ID ver4-6, R3.2a (blueprint §2, "Freeze the tuning anchor" -
    # the highest value-to-risk change in this update). A DEEP COPY of
    # the user's own configured HPARAMETERS, taken ONCE here and never
    # mutated again anywhere in this function (unless HP_TUNE_WARM_START
    # is explicitly set True - see the HP_TUNE dispatch site further
    # down). Every tuned model's "untuned baseline" - what the R3.2b
    # margin floor compares the search's winner against, and what the
    # tuner's own search starts FROM - is resolved from THIS frozen dict.
    #
    # Before this fix, `HPARAMETERS[base_model_name] = final_params` (the
    # HP_TUNE dispatch site further down) mutated the CALLER's dict in
    # place, so task i+1's own "untuned defaults" were silently task i's
    # tuned winner - drifting further from the user's real config with
    # every task - and a Parallel shard's first task started from the
    # user's real config while the Sequential run's task k*batch_size did
    # not, so the SAME config produced DIFFERENT numbers purely depending
    # on how it was sharded (a latent invariant I9 violation). The same
    # class of bug also affected every `HPARAMETERS[base_model_name][idx]
    # == 'all'` substitution throughout this function's untuned dispatch
    # branches (fixed alongside this, each now task-local - see e.g. the
    # GBLUP branch's own comment further down for the full detail).
    HPARAMETERS_BASELINE = copy.deepcopy(HPARAMETERS)
    print(f"[GP] HP_TUNE_WARM_START={_compute_resources['hp_tune_warm_start']} - " + (
        "tuning for task i+1 starts from task i's own tuned winner for the same "
        "model (ver4-5 carry-over behaviour restored)."
        if _compute_resources['hp_tune_warm_start'] else
        "every tuned (task, model) call starts its search from the user's own "
        "configured HPARAMETERS baseline, never a previous task's tuned result "
        "(ver4-6 default - fixes the sequential-vs-sharded divergence above)."
    ))

    # Update ID ver4-6, R4.2b (blueprint §2, "Group-scoped tuning"): a
    # GP()-scope cache, reused across tasks WITHIN THIS CALL (i.e. within
    # one shard) only - never persisted, never shared across processes
    # (invariant I9/I13) - keyed by (population, phenotype, ratio,
    # base_model_name, algorithm). Under HP_TUNE_SCOPE='per_scenario', a
    # cache hit skips the search entirely and reuses the winning
    # parameters for every remaining replicate in that group; under the
    # default 'per_task' this dict is populated but never consulted for a
    # hit (see the HP_TUNE dispatch site below), so the default is a
    # strict no-op. Rebuilt from `hp_record` on resume (see the checkpoint
    # rehydration block below) so a resumed run reproduces an
    # uninterrupted one rather than re-tuning mid-group.
    _tuning_cache = {}

    def _rebuild_tuning_cache_from_hp_record(hp_record_df):
        """ver4-6 R4.2b/I8: rebuild `_tuning_cache` from a REHYDRATED
        `hp_record` (i.e. on resume), so a resumed run under
        HP_TUNE_SCOPE='per_scenario' reuses the SAME tuned parameters for
        the rest of a group as an uninterrupted run would, instead of
        re-tuning mid-group and silently diverging from the pre-crash
        run. Only rows whose own `tuning_source` column is 'searched' are
        used (a 'reused:<n>' row was itself derived from a 'searched' row
        already being rebuilt here, so re-deriving from it adds nothing
        but risk of drift) - an `hp_record` predating this column (no
        `tuning_source` at all) contributes nothing, which is safe:
        HP_TUNE_SCOPE='per_scenario' simply re-searches once more for
        every such pre-existing group after a resume, at worst - never
        incorrect, only a missed reuse opportunity for that one group.

        Reconstructs each cached `final_params` list from
        HPARAMETERS_BASELINE (the SAME frozen anchor a fresh search would
        have started from) with only the TUNABLE fields overwritten from
        `hp_record`'s own tuned-value columns (matched by HPARAM_SPECS
        field label) - `hp_record` never stores a model's full positional
        list, only the fields that were actually searched over.
        """
        cache = {}
        if hp_record_df is None or hp_record_df.shape[0] == 0:
            return cache
        if 'tuning_source' not in hp_record_df.columns:
            return cache
        for _, row in hp_record_df.iterrows():
            if row.get('tuning_source') != 'searched':
                continue
            base_model_name = base_of(str(row['model']))
            algorithm = row.get('algorithm')
            schema_key = schema_key_of(base_model_name)
            fields = HPARAM_SPECS.get(schema_key)
            if not fields:
                continue
            baseline_entry = HPARAMETERS_BASELINE.get(base_model_name)
            if isinstance(baseline_entry, dict):
                base_params = baseline_entry.get(row['phenotype'])
            else:
                base_params = baseline_entry
            if base_params is None:
                continue
            final_params = list(base_params)
            for _idx, _field in enumerate(fields):
                if not _field.get('tunable'):
                    continue
                _label = _field['label']
                if _label not in row.index or pd.isna(row[_label]):
                    continue
                _value = row[_label]
                _ftype = _field['type']
                try:
                    if _ftype == 'int':
                        _value = int(round(float(_value)))
                    elif _ftype == 'float':
                        _value = float(_value)
                    elif _ftype == 'bool' and isinstance(_value, str):
                        _value = _value.strip().lower() in ('true', '1', 'yes')
                except (TypeError, ValueError):
                    pass  # keep the raw rehydrated value rather than lose the row entirely
                final_params[_idx] = _value
            key = (row['population'], row['phenotype'], row['ratio'], base_model_name, algorithm)
            cache[key] = final_params
        return cache

    def _resolve_tuned_params(base_model_name, algorithm, i, _base_params_for_tuning,
                               train, valid, test, _tuning_run_model_fn, hp_tuning_cfg,
                               _tuning_n_jobs, _trial_worker_init_args):
        """ver4-6 R4.2b: HP_TUNE_SCOPE cache lookup wrapping
        tune_model_hyperparameters(). Returns
        `(final_params, tuned_result, best_valid_score, tuned_values,
        tune_elapsed, tuning_source)`.

        `tuning_source` is `'searched'` on a fresh search - which, under
        `HP_TUNE_SCOPE='per_scenario'`, ALSO populates `_tuning_cache` for
        the rest of this group - or `'reused:<sample>'` on a cache hit
        (this task's own replicate id, so every reuse is individually
        traceable in `hyperparameter.csv`). A cache hit still pays for
        exactly ONE confirmatory fit at the cached hyperparameters on
        THIS task's own train/valid/test split - only the SEARCH itself
        (the expensive part) is skipped, per the blueprint's own R4.2b
        cost table.

        Under the default `HP_TUNE_SCOPE='per_task'`, `_tuning_cache` is
        never consulted for a hit here (only ever populated, never read),
        so every task always runs its own independent search - this
        function is then a transparent, behaviour-preserving wrapper
        around `tune_model_hyperparameters()`.
        """
        _scope = _compute_resources['hp_tune_scope']
        _cache_key = (
            sample.loc[i, 'population'], sample.loc[i, 'phenotype'],
            sample.loc[i, 'ratio'], base_model_name, algorithm,
        )
        _cache_hit = _tuning_cache.get(_cache_key) if _scope == 'per_scenario' else None

        if _cache_hit is not None:
            _t0 = time.time()
            _final_params = list(_cache_hit['final_params'])
            _result = _tuning_run_model_fn(base_model_name, train, valid, test, _final_params, RESULT_NAME)
            _elapsed = time.time() - _t0
            return (_final_params, _result, _cache_hit['best_valid_score'],
                    dict(_cache_hit['tuned_values']), _elapsed, f"reused:{sample.loc[i, 'sample']}")

        # ver4-6 R3.2c (invariant I9): the inner-resample seed base is
        # derived from THIS task's own replicate ('sample') value, so the
        # sequence of inner train/validation resamples is deterministic
        # and IDENTICAL regardless of sharding - the same seeding
        # discipline every other per-task random operation in GP()
        # already follows (e.g. train_test_split's own random_state).
        try:
            _inner_seed_base = int(sample.loc[i, 'sample'])
        except (TypeError, ValueError):
            _inner_seed_base = 0

        _final_params, _result, _best_score, _tuned_values, _elapsed = tune_model_hyperparameters(
            model_name=base_model_name,
            base_params=_base_params_for_tuning,
            train=train, valid=valid, test=test,
            run_model_fn=_tuning_run_model_fn,
            RESULT_NAME=RESULT_NAME,
            algorithm=algorithm,
            budget_kwargs=hp_tuning_cfg.get('budget', {}).get(algorithm, {}),
            hparam_specs=HPARAM_SPECS,
            reduced_cost_search=hp_tuning_cfg.get('reduced_cost_search', True),
            n_jobs=_tuning_n_jobs,
            # ver4-5 R1.b/R1.c (blueprint §3.4): every one of these
            # collapses to today's exact serial behaviour when
            # _tuning_n_jobs<=1, regardless of its own value.
            bayes_batch=_compute_resources['hp_tune_bayes_batch'],
            bayes_batch_max=_compute_resources['hp_tune_bayes_batch_max'],
            bayes_liar=_compute_resources['hp_tune_bayes_liar'],
            parallel_restarts=_compute_resources['hp_tune_parallel_restarts'],
            worker_init=_trial_worker_init,
            worker_init_args=_trial_worker_init_args,
            # ver4-6 R1/R3/R4: five new resolved keys - see
            # resolve_compute_resources()'s own docstring for each one's
            # default/rationale, and the "[GP] Compute resources
            # resolved:" log block near the top of this function for
            # their unconditionally-printed resolved values.
            bayes_domain_reduction=_compute_resources['hp_tune_bayes_domain_reduction'],
            selection_margin=_compute_resources['hp_tune_selection_margin'],
            valid_repeats=_compute_resources['hp_tune_valid_repeats'],
            inner_split_seed_base=_inner_seed_base,
        )
        if _scope == 'per_scenario':
            _tuning_cache[_cache_key] = {
                'final_params': list(_final_params),
                'tuned_values': dict(_tuned_values),
                'best_valid_score': _best_score,
            }
        return _final_params, _result, _best_score, _tuned_values, _elapsed, 'searched'

    # Update ID ver4-5, R2 (blueprint §2, invariant I14): log, once per
    # GP() call, which selected models have NO marker-pair interaction/
    # attention capability at all (architecture doc §17's "log negative
    # states unconditionally" principle) - so "why does this model
    # produce no Interaction.csv/Attention.csv rows?" always has a
    # documented answer in the run log, rather than only working it out
    # from silence. dict.fromkeys(...) both de-duplicates and preserves
    # first-seen order across MODEL_RUN (which may repeat a base model
    # more than once under different tuning-algorithm suffixes).
    for _base_model in dict.fromkeys(schema_key_of(base_of(m)) for m in MODEL_RUN if m != 'ensemble'):
        if has_no_interaction_capability(_base_model):
            print(f"[GP] NOTE: {_base_model} writes no marker-pair interactions or GAT "
                  f"attention weights this run - {no_interaction_capability_reason(_base_model)}")

    if PARALLEL is not None:
        idx = PARALLEL['batch_id']
        interval = PARALLEL['batch_size']
    else:
        idx = 0
        interval = sample.shape[0]

    # Total number of population/phenotype/ratio/replicate combinations ('sample'
    # rows) this call will actually process. Guard against a batch_id that
    # starts beyond the end of `sample` (nothing left to do for this batch).
    total_tasks = max(0, min(interval, sample.shape[0] - idx*interval))

    # Progress is tracked per (task, model) pair rather than per task: reporting
    # only once an entire task finishes means nothing is shown until every
    # selected model (which can include slow GAT fits) has run for the very
    # first task. Ticking after each individual model gives visible movement
    # much earlier, including within the first task. LD pruning (when enabled)
    # runs once per task too, so it gets its own unit alongside each task's
    # models rather than being invisible.
    # Update ID 2, R1, T5 (blueprint §10.3, cosmetic only - no output file
    # depends on this count): when a dispatch filter is active, progress
    # reporting should reflect only the models THIS sub-run will actually
    # fit, not the task's full MODEL_RUN - otherwise a filtered sub-run's
    # own progress bar would appear stuck well below 100% even after
    # every model it was assigned has finished.
    _progress_model_run = (
        [m for m in MODEL_RUN if m in _dispatch_filter] if _dispatch_filter is not None else MODEL_RUN
    )
    n_models = len([m for m in _progress_model_run if m != 'ensemble']) or 1
    units_per_task = n_models + (1 if LD_prune_effective is not None else 0) + (1 if RF_filter_effective is not None else 0)
    total_units = total_tasks * units_per_task
    completed_units = 0

    # ---------------------------------------------------------------------- #
    # Checkpoint/resume (checkpoint_utils.py): if a PREVIOUS run of this
    # exact same job (same RESULT_NAME, same PARALLEL batch if any, same
    # population/phenotype/ratio/replicate scenario list) failed partway
    # through, every scenario it finished before failing was already saved
    # to disk (see the try/except around the per-task loop below) - load
    # that back in now, so those scenarios are neither lost nor re-run.
    # `PARALLEL is None` (Sequential) and each Parallel batch each get
    # their OWN independent checkpoint - see checkpoint_utils.py's module
    # docstring.
    # ---------------------------------------------------------------------- #
    _is_parallel = PARALLEL is not None
    # Update ID ver4-6, R4.6 (blueprint §5/§7 RK-5): the fingerprint now
    # ALSO hashes HP_TUNE and the run's own frozen HPARAMETERS_BASELINE,
    # not just the scenario list - so editing which models are tuned,
    # their algorithm/budget, or their own untuned baseline values (none
    # of which changes `sample` itself) now correctly REFUSES a resume
    # against a stale checkpoint too, rather than silently reusing tuned
    # values computed under the OLD config. This intentionally
    # invalidates every in-flight checkpoint the first time a tree
    # upgrades to include this fix (disclosed in the ver4-6 Change
    # Summary as a one-time, expected, correct durability fix - not a
    # regression: a resubmitted job simply restarts that batch's task
    # loop from scratch).
    _sample_fp = _ckpt.sample_fingerprint(
        sample.iloc[idx*interval: idx*interval+interval],
        extra={'HP_TUNE': HP_TUNE, 'HPARAMETERS_BASELINE': HPARAMETERS_BASELINE},
    )
    _resume_last_completed_i = _ckpt.load_checkpoint(RESULT_NAME, idx, _is_parallel, _sample_fp, total_tasks)
    _static_write_flags = _ckpt.static_write_flags_for(MODEL_RUN, W_OPT)

    if _resume_last_completed_i is not None:
        _start_i = idx*interval + _resume_last_completed_i + 1
        _n_already_done = _resume_last_completed_i + 1
        if _n_already_done >= total_tasks:
            # Bug fix: every task was already completed and saved when this
            # checkpoint was written, but the checkpoint file still exists -
            # meaning the run's finalisation step (naive/weighted-ensemble
            # combination) hadn't succeeded yet (see
            # checkpoint_utils.load_checkpoint()'s own comment). The main
            # per-task loop below naturally runs zero iterations from here
            # (_start_i is already past its own end), falling straight
            # through to a fresh attempt at just that finalisation step.
            print(f"[GP] Resuming from a previous checkpoint: all {total_tasks} task(s) in "
                  f"this batch already completed and saved - only the final ensemble/"
                  f"aggregation step needs to be (re-)run.")
        else:
            print(f"[GP] Resuming from a previous checkpoint: {_n_already_done}/{total_tasks} task(s) "
                  f"in this batch already completed and saved - continuing from task "
                  f"{_n_already_done + 1}/{total_tasks} instead of starting over.")
        _loaded = _ckpt.load_partial_results(RESULT_NAME, idx, _is_parallel)
        record, result_train, result_valid, result_test = (
            _loaded['record'], _loaded['result_train'], _loaded['result_valid'], _loaded['result_test'],
        )
        effect, interactions, attention_total, weight, hp_record, stats = (
            _loaded['effect'], _loaded['interactions'], _loaded['attention_total'],
            _loaded['weight'], _loaded['hp_record'], _loaded['stats'],
        )
        completed_units = _n_already_done * units_per_task
        # Update ID ver4-6, R4.2b/I8: rebuild the per-scenario tuning
        # cache from the just-rehydrated hp_record (see
        # _rebuild_tuning_cache_from_hp_record()'s own docstring above) -
        # a strict no-op under the default HP_TUNE_SCOPE='per_task'
        # (nothing ever consults this cache in that mode), and required
        # for AC-R4.3 (a SIGKILL mid-group followed by resume reproduces
        # an uninterrupted run's own tuned values for the remaining
        # replicates) under 'per_scenario'.
        _tuning_cache = _rebuild_tuning_cache_from_hp_record(hp_record)
        if _tuning_cache:
            print(f"[GP] HP_TUNE_SCOPE={_compute_resources['hp_tune_scope']!r}: rebuilt "
                  f"{len(_tuning_cache)} tuned-parameter cache entrie(s) from the resumed "
                  f"hyperparameter.csv - remaining replicates in an already-searched group "
                  f"will reuse them rather than re-searching.")
    else:
        _start_i = idx*interval
        # Bug fix: starting this batch fresh (no valid checkpoint to
        # resume from) doesn't guarantee its result files are actually
        # empty/absent - a previous run of this exact batch that finished
        # successfully (and so had its checkpoint cleared), or an
        # unrelated earlier run that happened to reuse this same
        # RESULT_NAME, can leave old result files sitting at these exact
        # paths. append_partial_results() below (called after every task
        # succeeds) decides whether to write a CSV header purely from
        # whether a file already exists at that path - exactly right when
        # actually resuming, but if any such stale file exists here, this
        # run's very first task would otherwise get silently appended
        # after that old content instead of starting clean. See
        # checkpoint_utils.clear_result_files()'s own docstring for the
        # full detail, including why this also matters for the
        # conditional files (Interaction.csv, Weight.csv, etc.).
        _ckpt.clear_result_files(RESULT_NAME, idx, _is_parallel)

    # Tracks when the previous progress report happened, so each new report
    # can show how long the just-finished step took - this is what lets you
    # read task/model durations directly off the log instead of having to
    # subtract timestamps yourself.
    last_report_time = [time.time()]

    def _report_progress(completed, total, label=None):
        if total <= 0:
            return
        now = time.time()
        elapsed = now - last_report_time[0]
        last_report_time[0] = now
        if progress_callback is not None:
            # The GUI's progress bar caption has no other timestamp source,
            # so include a clock reading here for its display.
            timestamp = datetime.now().strftime('%H:%M:%S')
            timing = f'[{timestamp}, previous step took {elapsed:.1f}s]'
            timed_label = f'{label} {timing}' if label else timing
            progress_callback(completed, total, timed_label)
        else:
            # Default: a lightweight console progress line, useful for headless
            # (HPC) runs where no GUI is available to show a progress bar. No
            # need to add our own clock reading here - the calling script
            # (e.g. run_step1_batch.py) wraps sys.stdout in a TimestampedWriter
            # that already prefixes every line with a timestamp; this just adds
            # the computed step duration, which that wrapper can't know.
            timing = f'[previous step took {elapsed:.1f}s]'
            timed_label = f'{label} {timing}' if label else timing
            print(f'[GP] Progress: {completed}/{total} steps complete ({completed/total*100:.1f}%) - {timed_label}')

    # Report the starting state immediately, before any work has been done,
    # so a 0% progress display appears right away instead of staying blank.
    _report_progress(completed_units, total_units, label='Starting...')

    def _below_min_data_points(n_points, task_i, stage_label):
        """Requirement 5: if fewer than MIN_DATA_POINTS individuals remain
        for this scenario (summed across train/valid/test) after merging
        genotype and phenotype, report it and tell the caller
        (_process_one_task) to skip this task entirely - predictions from
        very few data points aren't reliable enough to be worth computing
        (and can make some models error out outright on tiny splits).
        Returns True/False; does not itself `return` out of the task -
        this is a plain helper function, not the task loop body - the
        caller acts on the result."""
        nonlocal completed_units
        if n_points >= MIN_DATA_POINTS:
            return False
        print(f"[GP] Task {task_i - idx*interval + 1}/{total_tasks} | population {sample.loc[task_i,'population']} | "
              f"phenotype {sample.loc[task_i,'phenotype']} | ratio {sample.loc[task_i,'ratio']} | "
              f"replicate {sample.loc[task_i,'sample']} | Skipping this task - only {n_points} data "
              f"point(s) remain {stage_label} (below the configured minimum of {MIN_DATA_POINTS}).")
        completed_units += units_per_task
        _report_progress(completed_units, total_units, label='Skipped (below minimum data points)')
        return True

    # ---------------------------------------------------------------------- #
    # Checkpoint/resume (continued): everything from here through the end of
    # this task's processing is wrapped in a function (rather than being the
    # for-loop's own body directly) purely so the driving loop just below can
    # catch an exception raised anywhere inside a single task's processing,
    # roll back any partial contribution that task may already have made to
    # the accumulators (nonlocal below), save every FULLY completed task's
    # results to disk, and re-raise - see that loop's own comments for the
    # full mechanism. Nothing about the task-processing logic itself changes
    # here; this is a structural wrapper only.
    # ---------------------------------------------------------------------- #
    def _process_one_task(i):
        nonlocal record, result_train, result_valid, result_test, effect, interactions
        nonlocal weight, attention_total, hp_record, stats, completed_units

        # Requirement 11 (efficiency): ONE cache dict for this task's own
        # GAT_biological_prior_knowledge data-driven-merge results
        # (LD-pruned/RF-selected markers, pairwise-Shapley interactions) -
        # created fresh for every task (never shared across tasks - a
        # different population/phenotype/ratio/replicate genuinely needs
        # its own network), but shared by EVERY caller within this one
        # task that needs this instance's merge result: the
        # OTHER_MODELS_MARKER_SOURCE='gene_network_plus_rf' marker-pool
        # restriction below, and every call this task makes to
        # GAT_biological_prior_knowledge() itself (including once per
        # hyperparameter-tuning trial, plus the final confirmatory fit -
        # all of which need the exact same network, computed once). See
        # that function's own network_cache docstring entry for exactly
        # how each entry gets filled in and reused.
        _bio_prior_merge_cache = {}

        # Convert the data structure
        if SCENARIO == 'within':
            data_phenotype = data_phenotype_original.loc[data_phenotype_original['population']==sample.loc[i,'population'], ['ID','population',sample.loc[i,'phenotype']]].reset_index(drop=True)
        elif SCENARIO == 'between':
            if W_OPT is None:
                data_phenotype = data_phenotype_original.loc[(data_phenotype_original['population'].astype(str) ==sample.loc[i,'population'].split('->')[0]) | 
                                                             (data_phenotype_original['population'].astype(str) ==sample.loc[i,'population'].split('->')[1]), 
                                                             ['ID','population',sample.loc[i,'phenotype']]].reset_index(drop=True)
            else:
                data_phenotype = data_phenotype_original.loc[(data_phenotype_original['population'].astype(str) ==sample.loc[i,'population'].split('->')[0]) | 
                                                             (data_phenotype_original['population'].astype(str) ==sample.loc[i,'population'].split('->')[1]) |
                                                             (data_phenotype_original['population'].astype(str) ==sample.loc[i,'population'].split('->')[2]), 
                                                             ['ID','population',sample.loc[i,'phenotype']]].reset_index(drop=True)

        # Requirement (bugfix): a genotype ID can only ever correspond to
        # ONE row from here on - every downstream step (train/valid/test
        # splitting, genotype<->phenotype merging, and especially the
        # native-PLINK path's PLINK --keep + pandas merge combination)
        # assumes a strict one-to-one ID<->individual correspondence.
        # Duplicate ID rows in the phenotype file (e.g. the same
        # accession measured in more than one environment/replicate -
        # common in real multi-site trial data) silently break this: the
        # same individual can end up split across train/valid/test (or
        # duplicated within one of them by train_test_split, which
        # partitions ROWS, not unique IDs), and - for PLINK input
        # specifically - a native extraction (which only ever returns ONE
        # row per unique ID, since PLINK's own --keep de-duplicates)
        # merged back against duplicated phenotype rows produces a
        # "fan-out" join that inflates that split's row count beyond what
        # id_train/id_valid/id_test (and every model's output, keyed
        # one-to-one against those same IDs) still expects - eventually
        # surfacing as a totally opaque "All arrays must be of the same
        # length" error deep inside some downstream model. Caught and
        # fixed HERE, once, for every phenotype/genotype format, rather
        # than symptom-by-symptom further down the pipeline.
        _dup_id_mask = data_phenotype['ID'].duplicated(keep='first')
        if _dup_id_mask.any():
            _dup_ids = sorted(set(data_phenotype.loc[_dup_id_mask, 'ID'].astype(str)))
            data_phenotype = data_phenotype[~_dup_id_mask].reset_index(drop=True)
            print(f"[GP] Task {i - idx*interval + 1}/{total_tasks} | population {sample.loc[i,'population']} | "
                  f"phenotype {sample.loc[i,'phenotype']} | WARNING: {len(_dup_ids)} individual ID(s) appeared "
                  f"more than once in the phenotype file for this population (e.g. measured in more than one "
                  f"environment/replicate) - kept only the FIRST record for each and dropped the rest, since "
                  f"genomic prediction needs exactly one phenotype value per individual. If you want a "
                  f"different rule (e.g. averaging across replicates), pre-process the phenotype file "
                  f"yourself before giving it to EasiGP. Affected ID(s)"
                  f"{' (showing first 20)' if len(_dup_ids) > 20 else ''}: {_dup_ids[:20]}")

        # ld_pruning_done_natively: set True only by the GENOTYPE_FORMAT ==
        # 'plink' branch below, when it already ran PLINK2's own
        # --indep-pairwise directly on the bed/bim/fam fileset - the
        # existing LD_prune_effective block further down must then skip its
        # own (DataFrame-based) pruning call, since pruning has already
        # happened.
        ld_pruning_done_natively = False
        # True, unrestricted (every marker, every individual in this task's
        # train/valid/test split) genotype pool - set below, in whichever
        # branch actually has it. Only ever populated when something this
        # task actually needs it (an 'other' model using the full/filtered
        # pool, OTHER_MODELS_MARKER_SOURCE == 'gene_network_plus_rf', or a
        # bio-prior instance's own data-driven merge feature - see
        # need_full_pool below): stays None otherwise, since for PLINK
        # format materialising it is a real, avoidable cost. GAT_biological_
        # prior_knowledge's own bio_train/bio_valid/bio_test (always the
        # gene-network-restricted pool, per its own model contract) are a
        # SEPARATE thing from this - full_marker_pool is only ever used as
        # merge_source_data, i.e. the marker universe the merge feature's
        # own internal RF-selection step chooses from, never as what the
        # model itself trains on.
        full_marker_pool = None

        def _resolve_bio_prior_params(bp_name):
            """This bio-prior instance's flat params list for the CURRENT
            task's phenotype - handles both the flat-list and
            per-phenotype-dict HPARAMETERS[bp_name] shapes (see
            _is_bio_prior_model's own docstring note on this), so every
            caller below (both the CSV and PLINK genotype-format paths)
            shares one, single place for this lookup instead of repeating
            the isinstance(dict, ...) branch. Defined here, before the
            GENOTYPE_FORMAT split below, specifically so it's available to
            BOTH branches - it only depends on HPARAMETERS/sample/i, none
            of which are genotype-format-specific."""
            bio_prior_hparams = HPARAMETERS[bp_name]
            if isinstance(bio_prior_hparams, dict):
                resolved = bio_prior_hparams.get(sample.loc[i, 'phenotype'])
                if resolved is None:
                    raise KeyError(
                        f"HPARAMETERS[{bp_name!r}] has no entry for phenotype "
                        f"{sample.loc[i,'phenotype']!r}."
                    )
                return resolved
            return bio_prior_hparams

        def _bio_prior_merge_enabled(bp_name):
            """Requirement 2's data-driven merge config (params[13])
            'enabled' flag for this bio-prior instance, for the current
            task's phenotype. False for anything that isn't the new dict
            shape (e.g. a stale/hand-built params list that still has a
            bare string in that slot from before requirement 5's upgrade),
            rather than raising - this is only ever used to decide whether
            EXTRA work is worth doing, never required for correctness of
            the plain gene-network path. Also defined here (see
            _resolve_bio_prior_params above) so both genotype-format paths
            share it."""
            merge_cfg = _resolve_bio_prior_params(bp_name)[13]
            return isinstance(merge_cfg, dict) and bool(merge_cfg.get('enabled', False))

        if GENOTYPE_FORMAT == 'csv':
            # ---------------------------------------------------------- #
            # Completely unchanged from before.
            # ---------------------------------------------------------- #
            if SCENARIO == 'within':
                data_genotype = data_genotype_original[data_genotype_original['population']==sample.loc[i,'population']].reset_index(drop=True)
            elif SCENARIO == 'between':
                if W_OPT is None:
                    data_genotype = data_genotype_original[(data_genotype_original['population'].astype(str) == sample.loc[i, 'population'].split('->')[0]) | 
                                                           (data_genotype_original['population'].astype(str) == sample.loc[i, 'population'].split('->')[1])].reset_index(drop=True)
                else:
                    data_genotype = data_genotype_original[(data_genotype_original['population'].astype(str) == sample.loc[i, 'population'].split('->')[0]) | 
                                                           (data_genotype_original['population'].astype(str) == sample.loc[i, 'population'].split('->')[1]) |
                                                           (data_genotype_original['population'].astype(str) == sample.loc[i, 'population'].split('->')[2])].reset_index(drop=True)

            # Requirement: rather than letting rows with no matching ID on
            # one side, or missing genotype/phenotype values on either
            # side, silently (or fatally) break the downstream split/model
            # fitting, drop them here explicitly and log what happened.
            _merged_raw = data_genotype.merge(data_phenotype, on=['ID', 'population'], how='inner')
            data = _merged_raw.dropna().reset_index(drop=True)
            _n_unmatched = len(
                (set(zip(data_genotype['ID'], data_genotype['population'])) |
                 set(zip(data_phenotype['ID'], data_phenotype['population'])))
                - set(zip(_merged_raw['ID'], _merged_raw['population']))
            )
            _n_missing_values = _merged_raw.shape[0] - data.shape[0]
            if _n_unmatched > 0 or _n_missing_values > 0:
                print(f"[GP] Task {i - idx*interval + 1}/{total_tasks} | population {sample.loc[i,'population']} | "
                      f"phenotype {sample.loc[i,'phenotype']} | Dropped {_n_unmatched} individual(s) with no "
                      f"matching genotype/phenotype record and {_n_missing_values} individual(s) with missing "
                      f"genotype or phenotype value(s); {data.shape[0]} individual(s) remain.")
            if _below_min_data_points(data.shape[0], i, 'after dropping individuals with missing or unmatched genotype/phenotype information'):
                return

            if SCENARIO =='within':
                if type(sample.loc[i,'ratio']) is not tuple:
                    train, test = train_test_split(data,train_size=sample.loc[i,'ratio'], random_state=sample.loc[i,'sample'])
                    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
                    id_train, id_valid, id_test = train.iloc[:,0], pd.DataFrame(), test.iloc[:,0]
                    train, valid, test = train.iloc[:,2:], pd.DataFrame(), test.iloc[:,2:]
                elif type(sample.loc[i,'ratio']) is tuple:
                    train, valid = train_test_split(data,train_size=sample.loc[i,'ratio'][0], random_state=sample.loc[i,'sample'])
                    valid, test = train_test_split(valid,train_size=sample.loc[i,'ratio'][1]/(sample.loc[i,'ratio'][2]+sample.loc[i,'ratio'][1]), random_state=sample.loc[i,'sample'])
                    train, valid, test = train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)
                    id_train, id_valid, id_test = train.iloc[:,0], valid.iloc[:,0], test.iloc[:,0]
                    train, valid, test = train.iloc[:,2:], valid.iloc[:,2:], test.iloc[:,2:]
            elif SCENARIO =='between':
                tmp = sample.loc[i, 'population'].split('->')
                if W_OPT is None:
                    train, test = data[data['population'].astype(str)==tmp[0]], data[data['population'].astype(str)==tmp[1]]
                    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
                    id_train, id_valid, id_test = train.iloc[:,0], pd.DataFrame(), test.iloc[:,0]
                    train, valid, test = train.iloc[:,2:], pd.DataFrame(), test.iloc[:,2:] 
                elif W_OPT is not None:
                    train, valid, test = data[data['population'].astype(str)==tmp[0]], data[data['population'].astype(str)==tmp[1]], data[data['population'].astype(str)==tmp[2]]
                    train, valid, test = train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)
                    id_train, id_valid, id_test = train.iloc[:,0], valid.iloc[:,0], test.iloc[:,0]
                    train, valid, test = train.iloc[:,2:], valid.iloc[:,2:], test.iloc[:,2:]

        else:  # GENOTYPE_FORMAT == 'plink'
            # ---------------------------------------------------------- #
            # Deferred-conversion path: the bed/bim/fam genotype matrix is
            # only ever materialised (via Preprocess.plink_io) for exactly
            # the sample IDs and markers this task actually needs.
            #
            # Requirement: genotype and phenotype are matched up-front,
            # BEFORE the train/valid/test split - not per-split afterwards
            # (which risked one split ending up completely empty if none
            # of ITS assigned IDs happened to have a genotype record,
            # discovered only after the split had already been committed
            # to). Individuals with a missing value for THIS phenotype are
            # dropped first (mirroring what the CSV path's own
            # data.merge(...).dropna() already does, just earlier here,
            # since there's no single merged `data` to dropna() on this
            # path); the survivors are then intersected with the .fam
            # file's own ID list (fam_iid_set, read once outside this
            # loop - see just above where bim_marker_info is read) - an
            # individual with a phenotype record but no matching genotype
            # record (or vice versa) is dropped here, reported, and never
            # reaches the split at all. This keeps id_train/id_valid/
            # id_test (computed right below, from this same filtered
            # data_phenotype) stable for the rest of this task: nothing
            # later is allowed to drop a row for a *phenotype* or *ID-
            # matching* reason again. Missing individual GENOTYPE VALUES
            # (not missing genotype RECORDS - PLINK's own missing-call
            # encoding for markers that were called for some but not all
            # kept individuals) are a separate matter, handled without
            # dropping rows at all - see _attach_phenotype() below.
            # ---------------------------------------------------------- #
            data_phenotype = data_phenotype.dropna(subset=[sample.loc[i, 'phenotype']]).reset_index(drop=True)

            _n_before_fam_match = data_phenotype.shape[0]
            _unmatched_ids = sorted(set(data_phenotype['ID'].astype(str)) - fam_iid_set)
            if _unmatched_ids:
                data_phenotype = data_phenotype[~data_phenotype['ID'].astype(str).isin(_unmatched_ids)].reset_index(drop=True)
                print(f"[GP] Task {i - idx*interval + 1}/{total_tasks} | population {sample.loc[i,'population']} | "
                      f"phenotype {sample.loc[i,'phenotype']} | Dropped {len(_unmatched_ids)} individual(s) with "
                      f"a phenotype record but no matching ID in the PLINK fileset ({GENOTYPE_FILE_NAME}.fam) "
                      f"out of {_n_before_fam_match} candidate(s); {data_phenotype.shape[0]} individual(s) remain: "
                      f"{_unmatched_ids}")

            # Requirement (empty-phenotype / no-matched-ID safeguard):
            # nothing left to split (either every individual was missing
            # this phenotype, or none of the ones that had it also have a
            # genotype record) - skip this task entirely rather than
            # letting train_test_split() raise on an empty DataFrame, or
            # silently producing an all-empty split further down.
            if _below_min_data_points(data_phenotype.shape[0], i, 'with both a non-missing value for this phenotype and a matching genotype record'):
                return

            if SCENARIO == 'within':
                if type(sample.loc[i,'ratio']) is not tuple:
                    pheno_train, pheno_test = train_test_split(data_phenotype, train_size=sample.loc[i,'ratio'], random_state=sample.loc[i,'sample'])
                    pheno_train, pheno_test = pheno_train.reset_index(drop=True), pheno_test.reset_index(drop=True)
                    pheno_valid = pd.DataFrame()
                else:
                    pheno_train, pheno_valid = train_test_split(data_phenotype, train_size=sample.loc[i,'ratio'][0], random_state=sample.loc[i,'sample'])
                    pheno_valid, pheno_test = train_test_split(pheno_valid, train_size=sample.loc[i,'ratio'][1]/(sample.loc[i,'ratio'][2]+sample.loc[i,'ratio'][1]), random_state=sample.loc[i,'sample'])
                    pheno_train, pheno_valid, pheno_test = pheno_train.reset_index(drop=True), pheno_valid.reset_index(drop=True), pheno_test.reset_index(drop=True)
            elif SCENARIO == 'between':
                tmp = sample.loc[i, 'population'].split('->')
                if W_OPT is None:
                    pheno_train = data_phenotype[data_phenotype['population'].astype(str)==tmp[0]].reset_index(drop=True)
                    pheno_test = data_phenotype[data_phenotype['population'].astype(str)==tmp[1]].reset_index(drop=True)
                    pheno_valid = pd.DataFrame()
                else:
                    pheno_train = data_phenotype[data_phenotype['population'].astype(str)==tmp[0]].reset_index(drop=True)
                    pheno_valid = data_phenotype[data_phenotype['population'].astype(str)==tmp[1]].reset_index(drop=True)
                    pheno_test = data_phenotype[data_phenotype['population'].astype(str)==tmp[2]].reset_index(drop=True)

            phenotype_col_name = sample.loc[i, 'phenotype']
            id_train = pheno_train['ID'].astype(str).tolist()
            id_valid = pheno_valid['ID'].astype(str).tolist() if pheno_valid.shape[0] != 0 else []
            id_test = pheno_test['ID'].astype(str).tolist()

            def _attach_phenotype(geno_df, pheno_df, split_label='split'):
                if geno_df.shape[0] == 0:
                    return pd.DataFrame()
                merged = geno_df.merge(pheno_df[['ID', phenotype_col_name]], on='ID').reset_index(drop=True)
                if merged.shape[0] > pheno_df.shape[0]:
                    # A "fan-out" join - MORE rows came out than individuals
                    # went in - meaning one side had a duplicate ID for at
                    # least one individual. data_phenotype is already
                    # deduplicated by ID once, up front, for the whole task
                    # (see its own dedup step above) - this branch is a
                    # SECOND, defensive layer in case geno_df itself somehow
                    # contains a duplicate ID (e.g. an unusual PLINK
                    # fileset). Silently keeping a fanned-out split would
                    # desync it from id_train/id_valid/id_test (and every
                    # downstream array keyed one-to-one against those same
                    # IDs) - exactly the failure mode that used to surface as
                    # an opaque "All arrays must be of the same length" error
                    # deep inside some model instead - so this keeps only the
                    # first match per ID, same rule as the up-front dedup.
                    _dup_mask = merged['ID'].duplicated(keep='first')
                    print(f"[GP] Task {i - idx*interval + 1}/{total_tasks} | population {sample.loc[i,'population']} | "
                          f"phenotype {sample.loc[i,'phenotype']} | {split_label} split: WARNING - merging "
                          f"genotype with phenotype data produced {merged.shape[0]} row(s) for only "
                          f"{pheno_df.shape[0]} expected individual(s) (a duplicate ID on the genotype side) "
                          f"- kept only the first match per ID.")
                    merged = merged[~_dup_mask].reset_index(drop=True)
                elif merged.shape[0] < pheno_df.shape[0]:
                    # Requirement: an individual expected in this split but not
                    # found in the genotype data extracted from the PLINK
                    # fileset has missing genotype information - rather than
                    # treating this as a fatal internal error, drop it (the
                    # same way a missing phenotype value is already dropped,
                    # above, before the split) and log what was dropped.
                    _dropped_ids = sorted(set(pheno_df['ID'].astype(str)) - set(merged['ID'].astype(str)))
                    print(f"[GP] Task {i - idx*interval + 1}/{total_tasks} | population {sample.loc[i,'population']} | "
                          f"phenotype {sample.loc[i,'phenotype']} | {split_label} split: dropped "
                          f"{len(_dropped_ids)} individual(s) with missing genotype information (no matching "
                          f"record in the PLINK fileset extraction) out of {pheno_df.shape[0]} expected: "
                          f"{_dropped_ids}")
                if merged.shape[0] == 0:
                    return pd.DataFrame()
                # Missing GENOTYPE values (PLINK's own missing-call encoding,
                # exported as blank/NA by --export A) are imputed with each
                # marker's own mean within this split, rather than dropping
                # the individual - dropping here would silently shrink this
                # split's row count below what id_train/id_valid/id_test
                # (computed above, from the phenotype-only split) already
                # committed to, which would corrupt the later results-
                # assembly step that pairs each model's predictions with
                # those IDs one-to-one. Any marker column that turns out to
                # be entirely missing within this split (mean itself is NaN)
                # falls back to 0, since there is nothing to average.
                marker_cols = [c for c in merged.columns if c not in ('ID', 'population', phenotype_col_name)]
                if merged[marker_cols].isna().any().any():
                    merged[marker_cols] = merged[marker_cols].fillna(merged[marker_cols].mean()).fillna(0)
                return merged.drop(columns=['ID', 'population'])

            bio_prior_names_now = _bio_prior_model_names(MODEL)
            other_models_selected_now = any(m != 'ensemble' and not _is_bio_prior_model(m) for m in MODEL)

            need_gene_pool = bool(bio_prior_names_now)
            _any_bio_prior_merge_now = any(_bio_prior_merge_enabled(n) for n in bio_prior_names_now)
            # 'gene_network_plus_rf' pool restriction (requirement 6) and
            # any bio-prior instance's own data-driven merge feature
            # (requirement 2) both need the TRUE full, unrestricted marker
            # set for this task - for GENOTYPE_FORMAT == 'csv' this is
            # already free (data_genotype_original always has every
            # marker); for 'plink' it means the full-fileset extraction
            # below must run even when no OTHER model would otherwise need
            # it (e.g. only GAT_biological_prior_knowledge itself is
            # selected, with its own merge feature turned on).
            need_full_pool = (
                (other_models_selected_now and OTHER_MODELS_MARKER_SOURCE == 'full_or_filtered')
                or _any_bio_prior_merge_now
            )

            # --- Requirement: gene-window-restricted pool for every bio-prior instance ---
            # Each instance (GAT_biological_prior_knowledge, ..._2, ... - see
            # _is_bio_prior_model) has its own HPARAMETERS entry, and so its
            # own network -> its own gene-window marker set -> its own
            # extraction from the PLINK fileset. bio_prior_pools holds all of
            # them; the dispatch loop further below looks up each instance's
            # own pool by name when it actually runs that model.
            bio_prior_pools = {}
            bio_prior_gene_window_markers = {}
            bio_prior_merge_pools = {}
            # Raw, fully unrestricted PLINK extraction (every marker, no
            # --extract list) - lazily computed at most once per task, the
            # first time any bio-prior instance's own data-driven merge
            # feature actually needs it (its own RF-selection step must
            # rank among the TRUE full candidate pool, exactly like the CSV
            # path already does via train_unpruned - never among an
            # already gene-window-restricted subset, which would bias
            # which markers "win" RF selection). Shared across every
            # merge-enabled instance in this task, since it doesn't depend
            # on any particular network.
            _raw_full_pool = None

            if need_gene_pool:
                for _bp_name in bio_prior_names_now:
                    _bp_preview = _resolve_bio_prior_params(_bp_name)

                    network_preview = load_network_json(_bp_preview[7])
                    candidate_genes_preview = extract_candidate_genes(network_preview)
                    gene_list_preview, _ = build_gene_list(candidate_genes_preview, _bp_preview[8], unit=_bp_preview[10])
                    gene_to_markers_preview = _map_genes_to_markers(gene_list_preview, bim_marker_info, bim_marker_info['name'])
                    gene_window_markers = sorted({m for markers in gene_to_markers_preview for m in markers})
                    if not gene_window_markers:
                        raise RuntimeError(
                            f"No markers in the PLINK fileset fall inside any gene window for "
                            f"{_bp_name} - cannot build its gene-network marker pool."
                        )
                    bio_prior_gene_window_markers[_bp_name] = gene_window_markers

                    _extract_markers = gene_window_markers
                    if _bio_prior_merge_enabled(_bp_name):
                        if _raw_full_pool is None:
                            _raw_train_geno = plink_to_genotype_df(GENOTYPE_FILE_NAME, pheno_train, keep_iid_list=id_train, plink_path=GENOTYPE_PLINK_PATH, work_dir=_new_plink_work_dir())
                            _raw_valid_geno = plink_to_genotype_df(GENOTYPE_FILE_NAME, pheno_valid, keep_iid_list=id_valid, plink_path=GENOTYPE_PLINK_PATH, work_dir=_new_plink_work_dir()) if pheno_valid.shape[0] != 0 else pd.DataFrame()
                            _raw_test_geno = plink_to_genotype_df(GENOTYPE_FILE_NAME, pheno_test, keep_iid_list=id_test, plink_path=GENOTYPE_PLINK_PATH, work_dir=_new_plink_work_dir())
                            _raw_full_pool = (
                                _attach_phenotype(_raw_train_geno, pheno_train, split_label='train'),
                                _attach_phenotype(_raw_valid_geno, pheno_valid, split_label='valid') if pheno_valid.shape[0] != 0 else pd.DataFrame(),
                                _attach_phenotype(_raw_test_geno, pheno_test, split_label='test'),
                            )
                        bio_prior_merge_pools[_bp_name] = _raw_full_pool

                        # data_train/valid/test (what the model actually
                        # trains on, via bio_prior_pools below) must already
                        # CONTAIN every column the merge step might add as a
                        # "bare marker" node, or node-feature construction
                        # inside the model would KeyError on a missing
                        # column - so extend this instance's own extraction
                        # list with its RF-selected candidates too. This is
                        # purely about which columns get extracted here;
                        # RF SELECTION ITSELF still ranks against
                        # _raw_full_pool (passed as merge_source_data below),
                        # not this already-narrowed list, so which markers
                        # "win" is unaffected by this extension.
                        _merge_cfg_preview = _bp_preview[13]
                        _rf_x_preview, _rf_y_preview = _raw_full_pool[0].iloc[:, :-1], _raw_full_pool[0].iloc[:, -1]
                        _ld_cfg_preview = _merge_cfg_preview.get('ld_prune')
                        if _ld_cfg_preview is not None:
                            # PERFORMANCE FIX: _raw_full_pool[0] just above
                            # was already materialised straight from the
                            # PLINK fileset (--export A) purely so the model
                            # can later see the TRUE full marker pool (to
                            # identify "bare" markers) - it isn't needed
                            # here just to decide which markers survive LD
                            # pruning. Preprocess.LD_pruning.LD_pruning()
                            # would otherwise re-serialise this same
                            # in-memory DataFrame back out to PED/MAP text
                            # and rebuild a BED fileset from it, purely to
                            # hand plink2 CHR/POS information it already
                            # has natively in the .bim file on disk -
                            # exactly the "write PED, then indep-pairwise"
                            # round trip that made this step slow (see
                            # Preprocess/LD_pruning.py's own PERFORMANCE FIX
                            # comments). ld_prune_marker_list_native() skips
                            # all of that: --keep + --indep-pairwise run
                            # directly against the ORIGINAL bed/bim/fam, and
                            # the survivors are then selected out of the
                            # DataFrame already in memory - a free, in-
                            # memory column selection. Falls back to the
                            # original DataFrame-based path for
                            # window_unit == 'cm' (no PLINK2-native cM
                            # pruning) - see ld_prune_marker_list_native()'s
                            # own docstring for the one documented
                            # behavioural difference this trades for the
                            # speed-up: it has no "unmapped SNP" concept,
                            # since a .bim file always carries real
                            # coordinates (unlike a hand-supplied snp_info
                            # CSV, which this bypasses entirely for kb/
                            # variants windows).
                            if _ld_cfg_preview.get('window_unit', 'kb') != 'cm':
                                _pruned_markers_preview = ld_prune_marker_list_native(
                                    GENOTYPE_FILE_NAME, id_train, _ld_cfg_preview,
                                    work_dir=_new_plink_work_dir(), bim_marker_info=bim_marker_info,
                                )
                                _rf_x_preview = _rf_x_preview[_pruned_markers_preview]
                            else:
                                _rf_x_preview, _, _ = LD_pruning(
                                    _rf_x_preview, pd.DataFrame(), _raw_full_pool[2].iloc[:, :-1],
                                    _ld_cfg_preview
                                )
                        _rf_selected_preview, _ = select_markers_for_data_driven_network(
                            _rf_x_preview, _rf_y_preview, _merge_cfg_preview['rf_filter']
                        )
                        # Requirement 11 (efficiency): this is the ONLY
                        # place LD pruning + RF filtering for this
                        # instance's data-driven merge should ever run for
                        # this task - cache the result so neither the
                        # OTHER_MODELS_MARKER_SOURCE='gene_network_plus_rf'
                        # restriction below, nor GAT_biological_prior_
                        # knowledge() itself (however many times it's
                        # called this task - once per hyperparameter-
                        # tuning trial, if enabled), ever redo this same,
                        # often very slow (large genotype files), work.
                        _bio_prior_merge_cache.setdefault(_bp_name, {})['rf_selected_markers'] = _rf_selected_preview
                        _extract_markers = sorted(set(gene_window_markers) | set(_rf_selected_preview))
                        print(f"[GP] Task {i - idx*interval + 1}/{total_tasks} | {_bp_name} data-driven "
                              f"merge: {len(gene_window_markers)} gene-window + {len(_rf_selected_preview)} "
                              f"RF-selected = {len(_extract_markers)} unique marker(s) to extract.")

                    gene_train_geno = plink_to_genotype_df(GENOTYPE_FILE_NAME, pheno_train, extract_snp_list=_extract_markers, keep_iid_list=id_train, plink_path=GENOTYPE_PLINK_PATH, work_dir=_new_plink_work_dir())
                    gene_valid_geno = plink_to_genotype_df(GENOTYPE_FILE_NAME, pheno_valid, extract_snp_list=_extract_markers, keep_iid_list=id_valid, plink_path=GENOTYPE_PLINK_PATH, work_dir=_new_plink_work_dir()) if pheno_valid.shape[0] != 0 else pd.DataFrame()
                    gene_test_geno = plink_to_genotype_df(GENOTYPE_FILE_NAME, pheno_test, extract_snp_list=_extract_markers, keep_iid_list=id_test, plink_path=GENOTYPE_PLINK_PATH, work_dir=_new_plink_work_dir())

                    _train_unpruned = _attach_phenotype(gene_train_geno, pheno_train, split_label='train')
                    _valid_unpruned = _attach_phenotype(gene_valid_geno, pheno_valid, split_label='valid') if pheno_valid.shape[0] != 0 else pd.DataFrame()
                    _test_unpruned = _attach_phenotype(gene_test_geno, pheno_test, split_label='test')
                    bio_prior_pools[_bp_name] = (_train_unpruned, _valid_unpruned, _test_unpruned)
                    print(f"[GP] Task {i - idx*interval + 1}/{total_tasks} | {_bp_name} "
                          f"gene-window marker pool: {len(_extract_markers)} marker(s) extracted from the "
                          f"PLINK fileset (out of {bim_marker_info.shape[0]} total).")

                # For downstream code that still wants a single reference
                # triple (the "other models restricted to gene_network" case
                # just below, and the per-task LD/RF-skip fallback) - the
                # FIRST bio-prior instance, in MODEL's own order, is that
                # reference (see OTHER_MODELS_MARKER_SOURCE's note near the
                # top of GP()). Each instance's OWN dispatch in the per-model
                # loop further below still uses its own pool from
                # bio_prior_pools, regardless of this.
                train_unpruned, valid_unpruned, test_unpruned = bio_prior_pools[bio_prior_names_now[0]]

            # --- Requirement: full/LD-pruned/RF-filtered pool for every other selected model ---
            if need_full_pool:
                # Requirement 11 (efficiency, follow-up): the TRUE,
                # unrestricted full pool is only ever actually consumed as
                # "other models"' own training data when
                # OTHER_MODELS_MARKER_SOURCE == 'full_or_filtered' - for
                # 'gene_network'/'gene_network_plus_rf', other models are
                # always restricted to a narrower pool further below and
                # never see this one, and merge_source_data for
                # GENOTYPE_FORMAT == 'plink' always comes from
                # bio_prior_merge_pools (== _raw_full_pool), never from
                # full_marker_pool - see ModelTrialRunner/_call_model's own
                # dispatch. So whenever need_full_pool is True purely
                # because of the bio-prior data-driven merge feature (the
                # common case for these two modes, and exactly the
                # Arabidopsis/OTHER_MODELS_MARKER_SOURCE='gene_network_
                # plus_rf' configuration this was tracked down from), this
                # must NOT pay for a second, identical PLINK2 --export A
                # extraction of the (often very large) full marker set
                # here - the preview loop above already did this exact
                # extraction once, as `_raw_full_pool`, purely to rank RF
                # candidates for the merge, and immediately discarded most
                # of its columns. Without this guard, every task extracted
                # the entire genotype matrix from PLINK twice merely to
                # throw the second copy away, which is what made LD-
                # pruning/RF-filtering runs under this mode so slow.
                _other_models_want_full_pool = other_models_selected_now and OTHER_MODELS_MARKER_SOURCE == 'full_or_filtered'

                if _other_models_want_full_pool:
                    if LD_prune_effective is not None:
                        task_num = i - idx*interval + 1
                        _report_progress(
                            completed_units, total_units,
                            label=(f"Task {task_num}/{total_tasks} | population {sample.loc[i,'population']} | "
                                   f"phenotype {sample.loc[i,'phenotype']} | ratio {sample.loc[i,'ratio']} | "
                                   f"replicate {sample.loc[i,'sample']} | LD pruning (native PLINK)")
                        )
                        ld_start_time = time.time()
                        train_full_geno = ld_prune_plink_native(GENOTYPE_FILE_NAME, pheno_train, id_train, LD_prune_effective, work_dir=_new_plink_work_dir(), bim_marker_info=bim_marker_info)
                        pruned_markers = [c for c in train_full_geno.columns if c not in ('ID', 'population')]
                        valid_full_geno = plink_to_genotype_df(GENOTYPE_FILE_NAME, pheno_valid, extract_snp_list=pruned_markers, keep_iid_list=id_valid, plink_path=GENOTYPE_PLINK_PATH, work_dir=_new_plink_work_dir()) if pheno_valid.shape[0] != 0 else pd.DataFrame()
                        test_full_geno = plink_to_genotype_df(GENOTYPE_FILE_NAME, pheno_test, extract_snp_list=pruned_markers, keep_iid_list=id_test, plink_path=GENOTYPE_PLINK_PATH, work_dir=_new_plink_work_dir())
                        print(f"[GP] Task {task_num}/{total_tasks} | Native PLINK LD pruning finished: "
                              f"{bim_marker_info.shape[0]} -> {len(pruned_markers)} markers "
                              f"(took {time.time() - ld_start_time:.1f}s)")
                        completed_units += 1
                        ld_pruning_done_natively = True

                        if _ld_decay_due((sample.loc[i, 'population'], sample.loc[i, 'ratio'], sample.loc[i, 'sample'])):
                            # Requirement 7: the native-PLINK path never
                            # materialises the pre-pruning genotype matrix in
                            # Python (that's the whole point of "native" - see
                            # ld_prune_plink_native's own docstring), so an
                            # extra, otherwise-unneeded PLINK export is done
                            # here purely for the decay plot - but ONLY on
                            # calls that are actually due (frequency-gated,
                            # and deduplicated across phenotypes, above), and
                            # entirely opt-in in the first place, so this cost
                            # is bounded and user-controlled.
                            _save_ld_decay_plot(
                                lambda: plink_to_genotype_df(
                                    GENOTYPE_FILE_NAME, pheno_train, keep_iid_list=id_train,
                                    plink_path=GENOTYPE_PLINK_PATH, work_dir=_new_plink_work_dir(),
                                ).iloc[:, 2:],
                                sample.loc[i, 'population'], sample.loc[i, 'sample'], sample.loc[i, 'ratio'],
                            )
                    else:
                        train_full_geno = plink_to_genotype_df(GENOTYPE_FILE_NAME, pheno_train, keep_iid_list=id_train, plink_path=GENOTYPE_PLINK_PATH, work_dir=_new_plink_work_dir())
                        valid_full_geno = plink_to_genotype_df(GENOTYPE_FILE_NAME, pheno_valid, keep_iid_list=id_valid, plink_path=GENOTYPE_PLINK_PATH, work_dir=_new_plink_work_dir()) if pheno_valid.shape[0] != 0 else pd.DataFrame()
                        test_full_geno = plink_to_genotype_df(GENOTYPE_FILE_NAME, pheno_test, keep_iid_list=id_test, plink_path=GENOTYPE_PLINK_PATH, work_dir=_new_plink_work_dir())

                    train = _attach_phenotype(train_full_geno, pheno_train, split_label='train')
                    valid = _attach_phenotype(valid_full_geno, pheno_valid, split_label='valid') if pheno_valid.shape[0] != 0 else pd.DataFrame()
                    test = _attach_phenotype(test_full_geno, pheno_test, split_label='test')
                    full_marker_pool = (train, valid, test)
                else:
                    # need_full_pool is True purely because of the bio-prior
                    # data-driven merge feature - reuse the preview loop's
                    # own raw extraction (`_raw_full_pool`) instead of
                    # redoing it; `train`/`valid`/`test` are assigned just
                    # below, from whichever narrower pool "other models"
                    # (if any) actually need for this OTHER_MODELS_MARKER_
                    # SOURCE mode.
                    full_marker_pool = _raw_full_pool

                if other_models_selected_now and OTHER_MODELS_MARKER_SOURCE == 'gene_network_plus_rf':
                    # Requirement 6, PLINK path: "other models"' marker pool
                    # is the reference bio-prior instance's gene-window
                    # markers UNION its own RF-selected markers - mirrors
                    # the CSV branch above; see its comments for the full
                    # reasoning.
                    _bio_prior_name_plink = bio_prior_names_now[0]
                    _bp_preview_plink = _resolve_bio_prior_params(_bio_prior_name_plink)
                    _merge_cfg_plink = _bp_preview_plink[13]
                    if not (isinstance(_merge_cfg_plink, dict) and _merge_cfg_plink.get('enabled', False)):
                        raise ValueError(
                            f"OTHER_MODELS_MARKER_SOURCE='gene_network_plus_rf' requires "
                            f"{_bio_prior_name_plink}'s own data-driven prior network (merge) "
                            f"feature to be enabled - it wasn't for phenotype "
                            f"{sample.loc[i,'phenotype']!r}."
                        )
                    # Requirement 11 (efficiency, follow-up): bio_prior_pools
                    # [_bio_prior_name_plink] (built by the gene-pool loop
                    # above) was already extracted directly from PLINK using
                    # EXACTLY this marker set - gene-window markers UNION
                    # this instance's own RF-selected markers (that loop's
                    # own `_extract_markers`) - since that extraction always
                    # folds in the RF-selected markers whenever this
                    # instance's merge feature is enabled (which the
                    # ValueError check just above already confirmed).
                    # Reusing it here avoids BOTH a second full-fileset
                    # PLINK extraction AND redoing the LD pruning/RF
                    # filtering that would otherwise be needed to know which
                    # markers to narrow down to - that work has already
                    # been done once, in the preview loop above, and its
                    # result is exactly this pool's own column set. (Falls
                    # back defensively to a fresh narrowing from the full
                    # pool if, for some reason, this instance's pool isn't
                    # in bio_prior_pools - shouldn't normally happen given
                    # need_gene_pool always populates it whenever any
                    # bio-prior model is selected.)
                    if _bio_prior_name_plink in bio_prior_pools:
                        train, valid, test = bio_prior_pools[_bio_prior_name_plink]
                        # bio_prior_pools' own extraction preserves whatever
                        # column order plink2 --extract happened to return
                        # (fileset/.bim order), not necessarily the sorted-
                        # by-name order the old, separately-extracted-and-
                        # narrowed pool used - reorder columns (a cheap,
                        # in-memory reindex, not a PLINK re-extraction) so
                        # "other models" see byte-identical column order to
                        # before, regardless of which code path filled
                        # train/valid/test.
                        _keep_cols_plink = sorted(c for c in train.columns if c != train.columns[-1]) + [train.columns[-1]]
                        train = train[_keep_cols_plink]
                        if valid.shape[0] != 0:
                            valid = valid[_keep_cols_plink]
                        test = test[_keep_cols_plink]
                        _rf_selected_plink = _bio_prior_merge_cache.get(_bio_prior_name_plink, {}).get('rf_selected_markers', [])
                        print(f"[GP] Task {i - idx*interval + 1}/{total_tasks} | Other model(s) restricted to "
                              f"{_bio_prior_name_plink}'s gene_network_plus_rf marker pool: "
                              f"{len(bio_prior_gene_window_markers[_bio_prior_name_plink])} gene-window + "
                              f"{len(_rf_selected_plink)} RF-selected = {train.shape[1] - 1} unique "
                              f"marker(s) (out of {bim_marker_info.shape[0]} total) - reused directly from "
                              f"the extraction already done above (no second PLINK extraction needed).")
                    else:
                        if full_marker_pool is None:
                            train_full_geno = plink_to_genotype_df(GENOTYPE_FILE_NAME, pheno_train, keep_iid_list=id_train, plink_path=GENOTYPE_PLINK_PATH, work_dir=_new_plink_work_dir())
                            valid_full_geno = plink_to_genotype_df(GENOTYPE_FILE_NAME, pheno_valid, keep_iid_list=id_valid, plink_path=GENOTYPE_PLINK_PATH, work_dir=_new_plink_work_dir()) if pheno_valid.shape[0] != 0 else pd.DataFrame()
                            test_full_geno = plink_to_genotype_df(GENOTYPE_FILE_NAME, pheno_test, keep_iid_list=id_test, plink_path=GENOTYPE_PLINK_PATH, work_dir=_new_plink_work_dir())
                            train = _attach_phenotype(train_full_geno, pheno_train, split_label='train')
                            valid = _attach_phenotype(valid_full_geno, pheno_valid, split_label='valid') if pheno_valid.shape[0] != 0 else pd.DataFrame()
                            test = _attach_phenotype(test_full_geno, pheno_test, split_label='test')
                            full_marker_pool = (train, valid, test)
                        else:
                            train, valid, test = full_marker_pool
                        _cached_merge_plink = _bio_prior_merge_cache.get(_bio_prior_name_plink, {})
                        if 'rf_selected_markers' in _cached_merge_plink:
                            _rf_selected_plink = _cached_merge_plink['rf_selected_markers']
                        else:
                            _rf_train_x_plink, _rf_train_y_plink = train.iloc[:, :-1], train.iloc[:, -1]
                            _ld_cfg_plink_fallback = _merge_cfg_plink.get('ld_prune')
                            if _ld_cfg_plink_fallback is not None:
                                # PERFORMANCE FIX: same reasoning as the
                                # preview loop's own native-pruning call
                                # above - train/test here already come from
                                # a PLINK --export A extraction (full_
                                # marker_pool, just above), so deciding
                                # which markers survive pruning natively
                                # against the original bed/bim/fam (no PED/
                                # BED round trip) and then column-selecting
                                # this already-in-memory DataFrame is a pure
                                # win, for the same 'cm' caveat documented on
                                # ld_prune_marker_list_native().
                                if _ld_cfg_plink_fallback.get('window_unit', 'kb') != 'cm':
                                    _pruned_markers_plink_fallback = ld_prune_marker_list_native(
                                        GENOTYPE_FILE_NAME, id_train, _ld_cfg_plink_fallback,
                                        work_dir=_new_plink_work_dir(), bim_marker_info=bim_marker_info,
                                    )
                                    _rf_train_x_plink = _rf_train_x_plink[_pruned_markers_plink_fallback]
                                else:
                                    _rf_train_x_plink, _, _ = LD_pruning(
                                        _rf_train_x_plink, pd.DataFrame(), test.iloc[:, :-1], _ld_cfg_plink_fallback
                                    )
                            _rf_selected_plink, _ = select_markers_for_data_driven_network(
                                _rf_train_x_plink, _rf_train_y_plink, _merge_cfg_plink['rf_filter']
                            )
                            _bio_prior_merge_cache.setdefault(_bio_prior_name_plink, {})['rf_selected_markers'] = _rf_selected_plink
                        extended_markers_plink = set(bio_prior_gene_window_markers[_bio_prior_name_plink]) | set(_rf_selected_plink)
                        phenotype_col_plink = train.columns[-1]
                        keep_cols_plink = sorted(extended_markers_plink) + [phenotype_col_plink]
                        train = train[keep_cols_plink]
                        if valid.shape[0] != 0:
                            valid = valid[keep_cols_plink]
                        test = test[keep_cols_plink]
                        print(f"[GP] Task {i - idx*interval + 1}/{total_tasks} | Other model(s) restricted to "
                              f"{_bio_prior_name_plink}'s gene_network_plus_rf marker pool: "
                              f"{len(bio_prior_gene_window_markers[_bio_prior_name_plink])} gene-window + "
                              f"{len(_rf_selected_plink)} RF-selected = {len(extended_markers_plink)} unique "
                              f"marker(s) (out of {bim_marker_info.shape[0]} total).")
                elif not _other_models_want_full_pool:
                    # OTHER_MODELS_MARKER_SOURCE == 'gene_network': other
                    # models get only the reference instance's gene-window
                    # pool - nothing from the (never-extracted-here) full
                    # pool.
                    train, valid, test = train_unpruned, valid_unpruned, test_unpruned
                # else (_other_models_want_full_pool is True): train/valid/
                # test were already set above, from the full (optionally
                # LD-pruned) pool - nothing further to do here.
            else:
                # Either only GAT_biological_prior_knowledge is selected, or
                # other selected models were explicitly redirected to its
                # gene-network marker pool (OTHER_MODELS_MARKER_SOURCE ==
                # 'gene_network', validated at the top of GP() to require
                # GAT_biological_prior_knowledge to be selected too) - either
                # way, nothing needs anything beyond the gene-window pool
                # already computed above (LD_prune_effective/
                # RF_filter_effective are already guaranteed None in both
                # cases - see their computation near the top of GP()).
                train, valid, test = train_unpruned, valid_unpruned, test_unpruned

        # Requirement: if either split ends up with no data points after
        # dropping individuals with missing/unmatched genotype or phenotype
        # information (CSV path: handled above at data-merge time; PLINK
        # path: handled just above in _attach_phenotype), there is nothing
        # to train or evaluate this task's model(s) on - skip the whole
        # task rather than letting an empty DataFrame crash the model
        # fitting/scoring code further down.
        if train.shape[0] == 0 or test.shape[0] == 0:
            _empty_split = 'training' if train.shape[0] == 0 else 'test'
            print(f"[GP] Task {i - idx*interval + 1}/{total_tasks} | population {sample.loc[i,'population']} | "
                  f"phenotype {sample.loc[i,'phenotype']} | ratio {sample.loc[i,'ratio']} | "
                  f"replicate {sample.loc[i,'sample']} | Skipping this task - the {_empty_split} split has "
                  f"no data points left after dropping missing/unmatched genotype or phenotype information.")
            completed_units += units_per_task
            _report_progress(completed_units, total_units, label='Skipped (empty train/test split)')
            return

        # Kept aside so GAT_biological_prior_knowledge can always be routed
        # around LD pruning below, even when it's applied for other models
        # in this same task.
        if GENOTYPE_FORMAT == 'csv':
            train_unpruned, valid_unpruned, test_unpruned = train, valid, test
            full_marker_pool = (train_unpruned, valid_unpruned, test_unpruned)

            if OTHER_MODELS_MARKER_SOURCE in ('gene_network', 'gene_network_plus_rf'):
                # Restrict what every OTHER selected model receives (train,
                # valid, test - about to become train_pruned_full below) to
                # the gene-window markers GAT_biological_prior_knowledge
                # itself uses (and, for 'gene_network_plus_rf', additionally
                # the same RF-selected markers its own data-driven merge
                # feature would pick - requirement 6) - computed here the
                # same way that model builds its own gene<->marker mapping
                # internally, since for CSV input the full genotype table is
                # already in memory anyway (unlike the PLINK path, where
                # this same computation is what decides which markers are
                # worth converting in the first place - see the
                # GENOTYPE_FORMAT == 'plink' branch above).
                # GAT_biological_prior_knowledge itself (train_unpruned etc.)
                # is untouched - it always determines its own subset
                # regardless of what's passed to it. If more than one
                # bio-prior instance is selected, the FIRST one (in MODEL's
                # own order) is the reference network for this restriction -
                # same documented simplification as the PLINK path above.
                _bio_prior_name_csv = _bio_prior_model_names(MODEL)[0]
                _bp_preview_csv = _resolve_bio_prior_params(_bio_prior_name_csv)

                _marker_info_csv = pd.read_csv(_bp_preview_csv[9])
                _network_csv = load_network_json(_bp_preview_csv[7])
                _candidate_genes_csv = extract_candidate_genes(_network_csv)
                _gene_list_csv, _ = build_gene_list(_candidate_genes_csv, _bp_preview_csv[8], unit=_bp_preview_csv[10])
                _gene_to_markers_csv = _map_genes_to_markers(_gene_list_csv, _marker_info_csv, train.columns[:-1])
                gene_window_markers_csv = sorted({m for markers in _gene_to_markers_csv for m in markers})
                if not gene_window_markers_csv:
                    raise RuntimeError(
                        "No markers in the genotype table fall inside any gene window for "
                        "GAT_biological_prior_knowledge - cannot restrict other models to its "
                        f"gene-network marker pool (OTHER_MODELS_MARKER_SOURCE={OTHER_MODELS_MARKER_SOURCE!r})."
                    )

                extended_markers_csv = set(gene_window_markers_csv)
                if OTHER_MODELS_MARKER_SOURCE == 'gene_network_plus_rf':
                    _merge_cfg_csv = _bp_preview_csv[13]
                    if not (isinstance(_merge_cfg_csv, dict) and _merge_cfg_csv.get('enabled', False)):
                        raise ValueError(
                            f"OTHER_MODELS_MARKER_SOURCE='gene_network_plus_rf' requires "
                            f"{_bio_prior_name_csv}'s own data-driven prior network (merge) "
                            f"feature to be enabled - it wasn't for phenotype "
                            f"{sample.loc[i,'phenotype']!r}."
                        )
                    # Requirement 11 (efficiency): check the cache first
                    # (defensive - nothing populates it before this point
                    # for CSV-format input, unlike the PLINK path's own
                    # preview loop, but checking costs nothing and keeps
                    # this block correct if that ever changes), then WRITE
                    # the result into it either way, so GAT_biological_
                    # prior_knowledge() itself - whenever it actually runs
                    # later this task, including once per hyperparameter-
                    # tuning trial if enabled - can reuse this exact
                    # result instead of redoing LD pruning + RF filtering
                    # all over again.
                    _cached_merge_csv = _bio_prior_merge_cache.get(_bio_prior_name_csv, {})
                    if 'rf_selected_markers' in _cached_merge_csv:
                        _rf_selected_csv = _cached_merge_csv['rf_selected_markers']
                        print(f"[GP] Task {i - idx*interval + 1}/{total_tasks} | reusing "
                              f"{len(_rf_selected_csv)} RF-selected marker(s) already computed "
                              f"for {_bio_prior_name_csv} this task (LD pruning/RF filtering skipped).")
                    else:
                        _rf_train_x_csv, _rf_train_y_csv = train_unpruned.iloc[:, :-1], train_unpruned.iloc[:, -1]
                        if _merge_cfg_csv.get('ld_prune') is not None:
                            _rf_train_x_csv, _, _ = LD_pruning(
                                _rf_train_x_csv, pd.DataFrame(), test_unpruned.iloc[:, :-1], _merge_cfg_csv['ld_prune']
                            )
                        _rf_selected_csv, _ = select_markers_for_data_driven_network(
                            _rf_train_x_csv, _rf_train_y_csv, _merge_cfg_csv['rf_filter']
                        )
                        _bio_prior_merge_cache.setdefault(_bio_prior_name_csv, {})['rf_selected_markers'] = _rf_selected_csv
                    extended_markers_csv.update(_rf_selected_csv)
                    print(f"[GP] Task {i - idx*interval + 1}/{total_tasks} | "
                          f"{_bio_prior_name_csv}'s data-driven merge marker set: "
                          f"{len(gene_window_markers_csv)} gene-window + {len(_rf_selected_csv)} "
                          f"RF-selected = {len(extended_markers_csv)} unique marker(s).")

                gene_window_markers_csv = sorted(extended_markers_csv)
                phenotype_col_csv = train.columns[-1]
                keep_cols_csv = gene_window_markers_csv + [phenotype_col_csv]
                train = train[keep_cols_csv]
                if valid.shape[0] != 0:
                    valid = valid[keep_cols_csv]
                test = test[keep_cols_csv]
                print(f"[GP] Task {i - idx*interval + 1}/{total_tasks} | Other model(s) restricted to "
                      f"GAT_biological_prior_knowledge's {OTHER_MODELS_MARKER_SOURCE} marker pool: "
                      f"{len(gene_window_markers_csv)} marker(s) (out of {len(train_unpruned.columns) - 1} total).")

        if LD_prune_effective is not None and not ld_pruning_done_natively:
            task_num = i - idx*interval + 1
            _report_progress(
                completed_units, total_units,
                label=(f"Task {task_num}/{total_tasks} | population {sample.loc[i,'population']} | "
                       f"phenotype {sample.loc[i,'phenotype']} | ratio {sample.loc[i,'ratio']} | "
                       f"replicate {sample.loc[i,'sample']} | LD pruning")
            )
            n_markers_before = train.shape[1] - 1
            _pre_prune_train_x = train.iloc[:, :-1]  # captured before `train` is reassigned below (needed for the decay plot)
            ld_start_time = time.time()
            train_pruned, valid_pruned, test_pruned = LD_pruning(_pre_prune_train_x, valid.iloc[:,:-1], test.iloc[:,:-1], LD_prune_effective)
            train, test = pd.concat([train_pruned,train.iloc[:,-1]],axis=1), pd.concat([test_pruned, test.iloc[:,-1]],axis=1)
            if valid.shape[0] != 0:
                valid = pd.concat([valid_pruned, valid.iloc[:,-1]],axis=1)
            n_markers_after = train.shape[1] - 1
            print(f"[GP] Task {task_num}/{total_tasks} | LD pruning finished: "
                  f"{n_markers_before} -> {n_markers_after} markers "
                  f"(took {time.time() - ld_start_time:.1f}s)")
            completed_units += 1

            if _ld_decay_due((sample.loc[i, 'population'], sample.loc[i, 'ratio'], sample.loc[i, 'sample'])):
                _save_ld_decay_plot(
                    lambda: _pre_prune_train_x,
                    sample.loc[i, 'population'], sample.loc[i, 'sample'], sample.loc[i, 'ratio'],
                )

        # RF marker importance filtering (requirement: runs AFTER LD pruning
        # when both are enabled, so it narrows whatever LD pruning already
        # produced above - or the original marker set, if LD pruning is
        # off). Fit on this task's own training set only, exactly like LD
        # pruning above - never on validation/test data.
        if RF_filter_effective is not None:
            task_num = i - idx*interval + 1
            _report_progress(
                completed_units, total_units,
                label=(f"Task {task_num}/{total_tasks} | population {sample.loc[i,'population']} | "
                       f"phenotype {sample.loc[i,'phenotype']} | ratio {sample.loc[i,'ratio']} | "
                       f"replicate {sample.loc[i,'sample']} | RF marker importance filtering")
            )
            n_markers_before = train.shape[1] - 1
            rf_start_time = time.time()
            train, valid, test = RF_marker_filtering(train, valid, test, RF_filter_effective)
            n_markers_after = train.shape[1] - 1
            print(f"[GP] Task {task_num}/{total_tasks} | RF marker importance filtering finished: "
                  f"{n_markers_before} -> {n_markers_after} markers "
                  f"(took {time.time() - rf_start_time:.1f}s)")
            completed_units += 1

        # What every model except GAT_biological_prior_knowledge actually
        # trains on: LD-pruned and/or RF-importance-filtered if either ran
        # just above, otherwise identical to
        # train_unpruned/valid_unpruned/test_unpruned.
        train_pruned_full, valid_pruned_full, test_pruned_full = train, valid, test

        # Requirement 6: basic per-scenario statistics - one row per task,
        # recorded here (train/valid/test finalised, right before the
        # per-model dispatch loop) rather than per-model, since split
        # sizes and marker count are task-level, not model-level,
        # quantities (every model in this task trains on the same row
        # counts, even though some - e.g. GAT_biological_prior_knowledge -
        # use a different, smaller MARKER pool than 'train' reflects
        # here; this row always describes the main/full-or-filtered pool
        # 'other' models actually use). Saved/checkpointed/assembled
        # exactly like every other per-scenario output (see
        # checkpoint_utils.py's RESULT_FILE_NAMES / assemble.py).
        stats = pd.concat([stats, pd.DataFrame([{
            'population': sample.loc[i, 'population'],
            'phenotype': sample.loc[i, 'phenotype'],
            'ratio': sample.loc[i, 'ratio'],
            'sample': sample.loc[i, 'sample'],
            'n_train': train.shape[0],
            'n_valid': valid.shape[0],
            'n_test': test.shape[0],
            'n_total': train.shape[0] + valid.shape[0] + test.shape[0],
            'n_markers': train.shape[1] - 1,
        }])], ignore_index=True)
        
        result_train_sample = pd.DataFrame()
        result_valid_sample = pd.DataFrame()
        result_test_sample = pd.DataFrame()
        
        # Prediction model implementation
        for jj in range(len(MODEL_RUN)):
            if MODEL_RUN[jj] == 'ensemble':
                continue

            # Update ID 2, R1, gate G1 (blueprint §10.3): when a dispatch
            # filter is active, skip every model NOT in it. Nothing above
            # this line in _process_one_task changes for a filtered
            # sub-run - Steps 1-8 pool construction (train/valid/test,
            # LD_prune_effective, RF_filter_effective, the
            # OTHER_MODELS_MARKER_SOURCE reference instance, ...) already
            # ran identically, once, for the FULL MODEL/MODEL_RUN list
            # before this loop started, regardless of this filter (I6).
            if _dispatch_filter is not None and MODEL_RUN[jj] not in _dispatch_filter:
                continue

            # MODEL_RUN[jj] is the (possibly hyperparameter-tuning-algorithm-
            # suffixed, e.g. 'RF__Grid') name used for every OUTPUT label
            # below (record/effect/predictions columns) - base_model_name is
            # what actually decides which model function to call and which
            # HPARAMETERS entry to read, since HPARAMETERS is keyed by base
            # model name only. When HP_TUNE isn't used, or this model was
            # only tuned with a single algorithm, the two are identical (see
            # models.hyperparameter_tuning.tuned_model_name / base_of) - so
            # this line changes nothing for any run that doesn't use
            # multi-algorithm tuning. base_model_name is also what's used to
            # look up HP_TUNE below - deliberately re-derived from
            # MODEL_RUN[jj] every iteration (not cached), since a task
            # further down the sample loop may have no validation split even
            # when an earlier one did (mixed single-float/tuple RATIO
            # entries), which the tuning check below re-evaluates per task.
            base_model_name = base_of(MODEL_RUN[jj])

            # Every model sees train_pruned_full/valid_pruned_full/test_pruned_full
            # by default (identical to the unpruned/unfiltered data if neither LD
            # pruning nor RF filtering ran this task). GAT_biological_prior_knowledge
            # is the one exception: its own dispatch branch below overrides this
            # again, based on its 'marker_source_mode' hyperparameter, once that
            # model's own params list has been resolved (it may be a per-phenotype
            # dict - see that branch).
            train, valid, test = train_pruned_full, valid_pruned_full, test_pruned_full

            # Update ID ver4-5, R2: reset per iteration, BEFORE any branch
            # below runs. Previously only the RF branch (and the tuning
            # path, via tuned_result.get('interaction', ...)) ever
            # assigned sample_interaction at all - every other branch left
            # it holding whatever the PREVIOUS model in this same task's
            # MODEL_RUN loop last set it to. That was invisible while the
            # accumulation check below compared against the literal
            # string 'RF' (only RF's own branch could ever match, so only
            # RF's own freshly-set value was ever read) - generalising
            # that check to model_registry.emits_interactions() (below)
            # would otherwise let a later model in the SAME task silently
            # inherit and misattribute an earlier model's stale
            # interaction rows. Overwritten below by whichever branch
            # (RF, GAT_prior_knowledge, or the tuning path) actually
            # computes a real value for this model.
            sample_interaction = pd.DataFrame()

            task_num = i - idx*interval + 1
            _report_progress(
                completed_units, total_units,
                label=(f"Task {task_num}/{total_tasks} | population {sample.loc[i,'population']} | "
                       f"phenotype {sample.loc[i,'phenotype']} | ratio {sample.loc[i,'ratio']} | "
                       f"replicate {sample.loc[i,'sample']} | model: {MODEL_RUN[jj]}")
            )
            model_start_time = time.time()

            # Hyperparameter tuning (opt-in, requires HP_TUNE[base_model_name]
            # enabled AND a real validation split for THIS task - Requirement:
            # only ever active when the dataset is split into train/valid/test).
            # Runs the winning algorithm's search on the validation set, then
            # a final confirmatory fit whose outputs (sample_pearson,
            # sample_effect, predicted_test, ...) feed into exactly the same
            # "Store prediction results" code below as every other model -
            # everything downstream is keyed off MODEL_RUN[jj], so it needs no
            # further changes for tuned models to be reported correctly.
            hp_tuning_cfg = (HP_TUNE or {}).get(base_model_name)
            _do_tuning = bool(hp_tuning_cfg) and hp_tuning_cfg.get('enabled') and valid.shape[0] != 0

            if _do_tuning:
                algorithm = algo_of(MODEL_RUN[jj]) or hp_tuning_cfg['algorithms'][0]

                # Update ID ver4-6, R3.2a: source this task's own "untuned
                # baseline" from HPARAMETERS_BASELINE (the frozen anchor
                # taken once at the top of GP() - see its own comment near
                # MODEL_RUN) unless HP_TUNE_WARM_START explicitly asks for
                # ver4-5's carry-over behaviour, in which case the search
                # instead starts from whatever HPARAMETERS[base_model_name]
                # currently holds - which, under warm-start, this same
                # dispatch block's post-search mutation below intentionally
                # keeps updating, exactly as ver4-5 did.
                _warm_start = bool(_compute_resources['hp_tune_warm_start'])
                _hparams_source = HPARAMETERS if _warm_start else HPARAMETERS_BASELINE
                _hparams_entry = _hparams_source[base_model_name]
                if _is_bio_prior_model(base_model_name) and isinstance(_hparams_entry, dict):
                    _current_phenotype = sample.loc[i, 'phenotype']
                    if _current_phenotype not in _hparams_entry:
                        raise KeyError(
                            f"HPARAMETERS[{base_model_name!r}] is a per-phenotype dict but has "
                            f"no entry for phenotype {_current_phenotype!r} - cannot tune it. Add "
                            f"one, or pass a single flat params list instead to share one network "
                            f"across every phenotype in PHENOTYPE."
                        )
                    _base_params_for_tuning = _hparams_entry[_current_phenotype]
                else:
                    _base_params_for_tuning = _hparams_entry

                # PERFORMANCE FIX (Requirement 11 follow-up): pre-warm
                # this instance's data-driven-merge cache entry - BOTH
                # 'rf_selected_markers' AND 'pair_df' - in THIS (the
                # main/driving) process, before _tuning_run_model_fn is
                # even constructed below, let alone handed to
                # tune_model_hyperparameters(). params[13] (the merge
                # config) is never itself a tunable field (hparam_specs.py's
                # own module docstring), so its result is byte-identical
                # for every trial this search will ever run - the ONLY
                # reason to compute it more than once per task is that,
                # without this call, the first computation would instead
                # happen lazily inside GAT_biological_prior_knowledge()
                # itself, the first time IT runs - which, under
                # HP_TUNE_PARALLEL_TRIALS=True (default) with n_jobs>1,
                # means inside a joblib 'loky' WORKER PROCESS. A worker's
                # own cache writes never make it back to this process
                # (only the trial's score does), so a cold cache at
                # trial-dispatch time gets silently recomputed once per
                # trial, independently, in every worker that ever draws
                # this model - see
                # Preprocess.data_driven_prior_network.ensure_bio_prior_
                # merge_cache()'s own docstring for the full analysis.
                # Filling it in here instead means _bio_prior_merge_cache
                # is ALREADY COMPLETE (whether populated just now, or
                # already primed earlier this task by the
                # OTHER_MODELS_MARKER_SOURCE='gene_network_plus_rf' pool-
                # routing step above - that step's own 'rf_selected_
                # markers' entry is reused here as-is, only 'pair_df' is
                # ever still missing at this point) the moment
                # ModelTrialRunner is pickled for submission to a worker
                # a few lines below - so every worker, for every trial,
                # unpickles an already-primed cache and takes the
                # existing cache-HIT branch inside
                # GAT_biological_prior_knowledge() itself, doing zero
                # recomputation, regardless of how many trials run or how
                # many separate processes they run in. A no-op (cheap
                # dict/None checks only) whenever this model's own merge
                # feature is disabled, or this isn't a bio-prior model at
                # all.
                if _is_bio_prior_model(base_model_name):
                    _merge_source_for_tuning = (
                        bio_prior_merge_pools.get(base_model_name) if GENOTYPE_FORMAT == 'plink'
                        else full_marker_pool
                    )
                    if _merge_source_for_tuning is not None:
                        _prewarm_train, _prewarm_valid, _prewarm_test = _merge_source_for_tuning
                        ensure_bio_prior_merge_cache(
                            base_model_name, _prewarm_train, _prewarm_valid, _prewarm_test,
                            _base_params_for_tuning[13], _bio_prior_merge_cache,
                        )

                # ver4-5 R1.a (blueprint §3.4(iii)): pass a directly-
                # constructed ModelTrialRunner as run_model_fn, rather
                # than a lambda wrapping _call_model() - both would work
                # equally correctly now that _call_model()'s own closure
                # holds nothing but plain data (R1.a's whole point), but
                # constructing the runner once here, bundling only the
                # bio_prior_ctx this task's own tuning call actually
                # needs, avoids re-building it on every trial evaluation
                # and avoids pickling the loose bio_prior_pools/
                # train_unpruned/... names individually (a lambda would
                # close over each of them separately, even for a
                # non-bio-prior model_name, since Python closures capture
                # every name a function body references regardless of
                # branch - see ModelTrialRunner's own docstring).
                #
                # phenotype_name is threaded in explicitly (unlike
                # _call_model()'s own former closure, which reached
                # `sample.loc[i, 'phenotype']` directly) - ModelTrialRunner
                # is module-level, not nested inside this loop, and so
                # cannot see this loop's own `sample`/`i` locals through
                # ordinary Python closure rules; this is the SAME
                # "bundle task-local values into bio_prior_ctx" pattern
                # every other bio-prior field here already uses.
                _tuning_run_model_fn = ModelTrialRunner(
                    RESULT_NAME, R_PATH, _compute_resources, GENOTYPE_FORMAT,
                    bio_prior_ctx={
                        'bio_prior_pools': bio_prior_pools,
                        'train_unpruned': train_unpruned, 'valid_unpruned': valid_unpruned,
                        'test_unpruned': test_unpruned,
                        'bio_prior_merge_pools': bio_prior_merge_pools,
                        'full_marker_pool': full_marker_pool,
                        'bim_marker_info_path': bim_marker_info_path,
                        'network_cache': _bio_prior_merge_cache,
                        'phenotype_name': sample.loc[i, 'phenotype'],
                    } if _is_bio_prior_model(base_model_name) else None,
                )

                # ver4-4 R3.f / ver4-5 R1 (blueprint §3.4): only request
                # more than one trial-evaluation worker when this run's
                # own N_JOBS names more than one usable core AND
                # HP_TUNE_PARALLEL_TRIALS is enabled - n_jobs<=1 (either
                # because N_JOBS=1 or the flag is off) reproduces
                # tune_model_hyperparameters()'s exact serial behaviour,
                # byte-for-byte, for every algorithm. Once R1.a made
                # ModelTrialRunner genuinely process-picklable, this
                # n_jobs value now ACTUALLY reaches worker processes
                # instead of reliably hitting the parallel-evaluation
                # fallback for every model, as it did before R1.a landed
                # (see the ver4-4 Change Summary §7/§8's own disclosure
                # of that limitation, and this update's own Change
                # Summary §9 acceptance-criteria evidence that the
                # fallback path is no longer taken).
                _tuning_n_jobs = (
                    _compute_resources['n_jobs']
                    if _compute_resources['hp_tune_parallel_trials'] else 1
                )
                # ver4-5 R1.i: the trial-worker initialiser + its plain-
                # data arguments, forwarded all the way down through
                # tune_model_hyperparameters() -> run_search() -> each
                # search function's own fan-out - see
                # _trial_worker_init()'s own docstring above for why
                # this is mandatory whenever _tuning_n_jobs > 1.
                _trial_worker_init_args = (
                    RESULT_NAME, R_PATH, _compute_resources, get_gpu_semaphore(),
                )

                final_params, tuned_result, best_valid_score, tuned_values, tune_elapsed, _tuning_source = \
                    _resolve_tuned_params(
                        base_model_name=base_model_name, algorithm=algorithm, i=i,
                        _base_params_for_tuning=_base_params_for_tuning,
                        train=train, valid=valid, test=test,
                        _tuning_run_model_fn=_tuning_run_model_fn,
                        hp_tuning_cfg=hp_tuning_cfg, _tuning_n_jobs=_tuning_n_jobs,
                        _trial_worker_init_args=_trial_worker_init_args,
                    )
                if _warm_start:
                    if _is_bio_prior_model(base_model_name) and isinstance(_hparams_entry, dict):
                        HPARAMETERS[base_model_name][_current_phenotype] = final_params
                    else:
                        HPARAMETERS[base_model_name] = final_params
                # else (ver4-6 R3.2a default): HPARAMETERS is never
                # mutated here - final_params stays task-local, used only
                # for this (task, model)'s own confirmatory-fit result and
                # hp_record row below.

                sample_pearson = tuned_result['pearson_test']
                sample_mse = tuned_result['mse_test']
                sample_effect = tuned_result.get('effect', pd.DataFrame())
                sample_interaction = tuned_result.get('interaction', pd.DataFrame())
                sample_attention = tuned_result.get('attention', pd.DataFrame())
                predicted_test = tuned_result['predicted_test']
                predicted_valid = tuned_result['predicted_valid']
                predicted_train = tuned_result['predicted_train']

                hp_record = pd.concat([hp_record, pd.DataFrame([{
                    'population': sample.loc[i, 'population'], 'phenotype': sample.loc[i, 'phenotype'],
                    'model': MODEL_RUN[jj], 'ratio': sample.loc[i, 'ratio'], 'sample': sample.loc[i, 'sample'],
                    'algorithm': algorithm, 'valid_pearson_over_mse': best_valid_score,
                    # Update ID ver4-6, R3.2d: 'valid_pearson_over_mse' has
                    # not actually been r/mse since ver4-5 (architecture
                    # doc §0.3 D7) - kept populated, identically, for one
                    # release (deprecation window, removal planned for
                    # ver4-7); 'valid_objective_score' is the new,
                    # correctly-named column carrying the SAME value.
                    # R4.2b: 'tuning_source' records whether this row was
                    # actually searched for THIS task, or reused from an
                    # earlier replicate in the same scenario
                    # (HP_TUNE_SCOPE='per_scenario' only).
                    'valid_objective_score': best_valid_score, 'tuning_source': _tuning_source,
                    **tuned_values,
                }])])
                _tune_verb = f"tuned via {algorithm}" if _tuning_source == 'searched' else \
                    f"reused tuned hyperparameters from {_tuning_source.split(':', 1)[-1]!r} " \
                    f"(HP_TUNE_SCOPE='per_scenario' - confirmatory fit only)"
                print(f"[GP] Task {task_num}/{total_tasks} | {MODEL_RUN[jj]} {_tune_verb}: "
                      f"valid Pearson/MSE={best_valid_score:.4f} (took {tune_elapsed:.1f}s) -> {tuned_values}")

            elif base_model_name == 'rrBLUP':
                result_rrBLUP = rrBLUP(train, valid, test, HPARAMETERS[base_model_name], RESULT_NAME)
                _cleanup_bglr_output_for(RESULT_NAME)  # disk-quota fix - see module-level note above
                # ver4-4 R5 Fix 1: r_list_get() replaces direct
                # result_rrBLUP['r_*'] indexing - see pipeline_utils.
                # r_list_get()'s docstring and the R5 design record.
                # Applied here AND in _call_model()'s equivalent branch
                # above, in lockstep (per _call_model's own docstring).
                sample_pearson, sample_mse = r_list_get(result_rrBLUP, 'r_pearson')[0], r_list_get(result_rrBLUP, 'r_MSE')[0]
                sample_effect = r_list_get(result_rrBLUP, 'r_effect')
                predicted_test = r_list_get(result_rrBLUP, 'r_y_predicted')
                predicted_valid = r_list_get(result_rrBLUP, 'r_y_predicted_valid')
                predicted_train = r_list_get(result_rrBLUP, 'r_y_predicted_train')
                # Update ID ver4-5, R2 Stage 10 (I3 lockstep with
                # ModelTrialRunner's equivalent branch - both call the
                # SAME shared, module-level helper, so there is only one
                # implementation to keep in sync, not two - see that
                # helper's own docstring).
                sample_interaction = _r_model_surrogate_interaction(
                    'rrBLUP', train, predicted_train, HPARAMETERS[base_model_name],
                    _compute_resources.get('n_jobs'), test=test,
                )
            elif base_model_name == 'GBLUP':
                # GBLUP params: [nIter, burnIn, get_effect, Shapley_num,
                # max_shap_features, Shapley_nIter, Shapley_burnIn, shap_row_offset,
                # shap_row_count, get_interaction, max_interaction_features,
                # interaction_background, interaction_top] -> Shapley_num is at
                # index 3; the last four fields (ver4-5 R2 Stage 10) are
                # Python-only, never passed to GBLUP.R (see
                # _r_model_interaction_fields()'s own docstring).
                # Update ID ver4-6, R3.2a: a TASK-LOCAL copy - 'all' is
                # re-resolved against THIS task's own test.shape[0] every
                # time, never written back onto the shared HPARAMETERS
                # dict (which the ver4-6 baseline freeze at the top of
                # GP() no longer mutates at all by default - see
                # HPARAMETERS_BASELINE). Before this fix, the substitution
                # below persisted onto HPARAMETERS[base_model_name]
                # itself, so only the FIRST task to reach this branch
                # ever actually re-resolved 'all' - every later task with
                # a differently-sized test split silently reused task
                # 1's own test.shape[0] instead of its own.
                _gblup_params = list(HPARAMETERS[base_model_name])
                if _gblup_params[3] == 'all':
                    _gblup_params[3] = test.shape[0]
                # ver4-4 R4.h - see _call_model()'s equivalent GBLUP
                # branch above (lockstep, per _call_model's own
                # docstring) for the full explanation of both this and
                # the R3.d fan-out call immediately below.
                _gblup_k_precomputed = (
                    _build_grm(train, valid, test, _compute_resources['device'])
                    if _compute_resources['gpu_kernel_precompute'] else None
                )
                result_GBLUP = _run_gblup_or_rkhs(GBLUP, 'GBLUP', train, valid, test,
                                                   _gblup_params, RESULT_NAME,
                                                   k_precomputed=_gblup_k_precomputed)
                _cleanup_bglr_output_for(RESULT_NAME)  # disk-quota fix - see module-level note above
                # ver4-4 R5 Fix 1 - see rrBLUP branch above.
                sample_pearson, sample_mse = r_list_get(result_GBLUP, 'r_pearson')[0], r_list_get(result_GBLUP, 'r_MSE')[0]
                sample_effect = r_list_get(result_GBLUP, 'r_effect')
                predicted_test = r_list_get(result_GBLUP, 'r_y_predicted')
                predicted_valid = r_list_get(result_GBLUP, 'r_y_predicted_valid')
                predicted_train = r_list_get(result_GBLUP, 'r_y_predicted_train')
                sample_interaction = _r_model_surrogate_interaction(
                    'GBLUP', train, predicted_train, _gblup_params,
                    _compute_resources.get('n_jobs'), test=test,
                )
            elif base_model_name  == 'BayesB':
                result_BayesB = BayesB(train, valid, test, HPARAMETERS[base_model_name], RESULT_NAME)
                _cleanup_bglr_output_for(RESULT_NAME)  # disk-quota fix - see module-level note above
                # ver4-4 R5 Fix 1 - see rrBLUP branch above.
                sample_pearson, sample_mse = r_list_get(result_BayesB, 'r_pearson')[0], r_list_get(result_BayesB, 'r_MSE')[0]
                sample_effect = r_list_get(result_BayesB, 'r_effect')
                predicted_test = r_list_get(result_BayesB, 'r_y_predicted')
                predicted_valid = r_list_get(result_BayesB, 'r_y_predicted_valid')
                predicted_train = r_list_get(result_BayesB, 'r_y_predicted_train')
                sample_interaction = _r_model_surrogate_interaction(
                    'BayesB', train, predicted_train, HPARAMETERS[base_model_name],
                    _compute_resources.get('n_jobs'), test=test,
                )
            elif base_model_name  == 'RKHS':
                # RKHS params: [nIter, burnIn, h, get_effect, Shapley_num,
                # max_shap_features, Shapley_nIter, Shapley_burnIn, shap_row_offset,
                # shap_row_count, get_interaction, max_interaction_features,
                # interaction_background, interaction_top, interaction_screen,
                # interaction_screen_top, interaction_grid_resolution] -> Shapley_num
                # is at index 4; the last SEVEN fields (ver4-5 R2 Stage 10 +
                # ver4-6 R1/R1b) are Python-only, never passed to RKHS.R.
                # Update ID ver4-6, R3.2a - see the GBLUP branch above for
                # the full rationale (task-local, re-resolved every task).
                _rkhs_params = list(HPARAMETERS[base_model_name])
                if _rkhs_params[4] == 'all':
                    _rkhs_params[4] = test.shape[0]
                # ver4-4 R3.d - see _call_model()'s equivalent RKHS branch
                # above (lockstep). No Python-side kernel precompute for
                # RKHS this stage - see _build_grm()'s own module
                # docstring for the disclosed scope decision.
                result_RKHS = _run_gblup_or_rkhs(RKHS, 'RKHS', train, valid, test,
                                                  _rkhs_params, RESULT_NAME)
                _cleanup_bglr_output_for(RESULT_NAME)  # disk-quota fix - see module-level note above
                # ver4-4 R5 Fix 1 - see rrBLUP branch above.
                sample_pearson, sample_mse = r_list_get(result_RKHS, 'r_pearson')[0], r_list_get(result_RKHS, 'r_MSE')[0]
                sample_effect = r_list_get(result_RKHS, 'r_effect')
                predicted_test = r_list_get(result_RKHS, 'r_y_predicted')
                predicted_valid = r_list_get(result_RKHS, 'r_y_predicted_valid')
                predicted_train = r_list_get(result_RKHS, 'r_y_predicted_train')
                sample_interaction = _r_model_surrogate_interaction(
                    'RKHS', train, predicted_train, _rkhs_params,
                    _compute_resources.get('n_jobs'),
                    use_gpu=_compute_resources.get('use_gpu_sklearn', False), test=test,
                )
            elif base_model_name  == 'RF':
                # RF params: [estimators, features_max, sample_max, max_depth,
                # min_samples_leaf, get_interaction, shapley_num, threshold,
                # max_interaction_features, interaction_method] -> shapley_num
                # is at index 6. interaction_method (Requirements.md item 3,
                # appended-only index 9) is read defensively inside RF()
                # itself and needs no substitution here.
                # Update ID ver4-6, R3.2a - task-local, re-resolved every
                # task (see the GBLUP branch above for the full rationale).
                _rf_params = list(HPARAMETERS[base_model_name])
                if _rf_params[6] == 'all':
                    _rf_params[6] = test.shape[0]
                sample_pearson, sample_mse, sample_effect, sample_interaction, predicted_test, predicted_valid, predicted_train = RF(train, valid, test, _rf_params)
            elif base_model_name  == 'ExtraTrees':
                # Update ID ver4-5, R2 Stage 8: ExtraTrees params mirror
                # RF's own layout exactly - shapley_num is at index 6.
                # Update ID ver4-6, R3.2a - task-local, re-resolved every task.
                _extratrees_params = list(HPARAMETERS[base_model_name])
                if _extratrees_params[6] == 'all':
                    _extratrees_params[6] = test.shape[0]
                sample_pearson, sample_mse, sample_effect, sample_interaction, predicted_test, predicted_valid, predicted_train = ExtraTrees(train, valid, test, _extratrees_params)
            elif base_model_name  == 'GBDT':
                # Update ID ver4-5, R2 Stage 8: GBDT params: [max_iter,
                # learning_rate, max_depth, max_leaf_nodes, min_samples_leaf,
                # l2_regularization, get_interaction, shapley_num, threshold,
                # max_interaction_features] -> shapley_num is at index 7.
                # Update ID ver4-6, R3.2a - task-local, re-resolved every task.
                _gbdt_params = list(HPARAMETERS[base_model_name])
                if _gbdt_params[7] == 'all':
                    _gbdt_params[7] = test.shape[0]
                sample_pearson, sample_mse, sample_effect, sample_interaction, predicted_test, predicted_valid, predicted_train = GBDT(train, valid, test, _gbdt_params)
            elif base_model_name  == 'XGBoost':
                # Update ID ver4-5, R2 Stage 11 (Tier 2 - optional
                # dependency, gated by MODEL_AVAILABILITY_STRICT at
                # config-validation time, see near the top of GP()).
                # Params: [n_estimators, max_depth, learning_rate, subsample,
                # colsample_bytree, reg_lambda, get_interaction, shapley_num,
                # threshold, max_interaction_features] -> shapley_num is at
                # index 7.
                # Update ID ver4-6, R3.2a - task-local, re-resolved every task.
                _xgboost_params = list(HPARAMETERS[base_model_name])
                if _xgboost_params[7] == 'all':
                    _xgboost_params[7] = test.shape[0]
                sample_pearson, sample_mse, sample_effect, sample_interaction, predicted_test, predicted_valid, predicted_train = XGBoost(train, valid, test, _xgboost_params)
            elif base_model_name  == 'EBM':
                # Update ID ver4-5, R2 Stage 11 (Tier 2 - optional
                # dependency). No shapley_num/'all' substitution - EBM's
                # interactions are its own already-fitted pairwise terms,
                # not an explained-sample-count computation.
                sample_pearson, sample_mse, sample_effect, sample_interaction, predicted_test, predicted_valid, predicted_train = EBM(train, valid, test, HPARAMETERS[base_model_name])
            elif base_model_name  == 'SVR':
                # SVR params: [ker, eps, con, deg, gam, coef0, get_effect,
                # shapley_num, max_shap_features, shap_background_size,
                # shap_nsamples, get_interaction, max_interaction_features,
                # interaction_background, interaction_top] -> shapley_num is
                # at index 7; the last four fields are ver4-5 R2 Stage 9.
                # Update ID ver4-6, R3.2a - task-local, re-resolved every task.
                _svr_params = list(HPARAMETERS[base_model_name])
                if _svr_params[7] == 'all':
                    _svr_params[7] = test.shape[0]
                sample_pearson, sample_mse, sample_effect, sample_interaction, predicted_test, predicted_valid, predicted_train = SV_Regression(train, valid, test, _svr_params)
            elif base_model_name  == 'KNN':
                # KNN params: [n_neighbours, weights, p, get_effect, shapley_num,
                # max_shap_features, shap_background_size, shap_nsamples,
                # get_interaction, max_interaction_features, interaction_background,
                # interaction_top] -> shapley_num is at index 4; the last four
                # fields are ver4-5 R2 Stage 9.
                # Update ID ver4-6, R3.2a - task-local, re-resolved every task.
                _knn_params = list(HPARAMETERS[base_model_name])
                if _knn_params[4] == 'all':
                    _knn_params[4] = test.shape[0]
                sample_pearson, sample_mse, sample_effect, sample_interaction, predicted_test, predicted_valid, predicted_train = KNN(train, valid, test, _knn_params)
            elif base_model_name  == 'MLP':
                # MLP params: [neurons, dout, lrate, decay, ep, bsize, neurons2,
                # shapley_num, get_interaction, max_interaction_features,
                # interaction_top] -> shapley_num is at index 7; the last
                # three fields are ver4-5 R2 Stage 9.
                # Update ID ver4-6, R3.2a - task-local, re-resolved every task.
                _mlp_params = list(HPARAMETERS[base_model_name])
                if _mlp_params[7] == 'all':
                    _mlp_params[7] = test.shape[0]
                sample_pearson, sample_mse, sample_effect, sample_interaction, predicted_test, predicted_valid, predicted_train = ML_Perceptron(train, valid, test, _mlp_params)
            elif base_model_name  == 'GAT_infinitesimal_node_level':
                # GAT_infinitesimal_node_level params: [..., samples, marker_effect] -> samples is second-to-last
                # Update ID ver4-6, R3.2a - task-local, re-resolved every task.
                _gat_inl_params = list(HPARAMETERS[base_model_name])
                if _gat_inl_params[-2] == 'all':
                    _gat_inl_params[-2] = test.shape[0]
                sample_pearson, sample_mse, sample_effect, predicted_test, predicted_valid, predicted_train = GAT_infinitesimal_node_level(train, valid, test, _gat_inl_params)
            elif base_model_name  == 'GAT_infinitesimal':
                # GAT_infinitesimal params: [..., marker_effect, samples] -> samples is the last element
                # Update ID ver4-6, R3.2a - task-local, re-resolved every task.
                _gat_inf_params = list(HPARAMETERS[base_model_name])
                if _gat_inf_params[-1] == 'all':
                    _gat_inf_params[-1] = test.shape[0]
                sample_pearson, sample_mse, sample_effect, predicted_test, predicted_valid, predicted_train = GAT_infinitesimal(train, valid, test, _gat_inf_params)
            elif base_model_name  == 'GAT_fully_connected':
                # GAT_fully_connected params: [..., marker_effect, samples] -> samples is the last element
                # Update ID ver4-6, R3.2a - task-local, re-resolved every task.
                _gat_fc_params = list(HPARAMETERS[base_model_name])
                if _gat_fc_params[-1] == 'all':
                    _gat_fc_params[-1] = test.shape[0]
                sample_pearson, sample_mse, sample_effect, predicted_test, predicted_valid, predicted_train, sample_attention = GAT_fully_connected(train, valid, test, _gat_fc_params)
            elif base_model_name  == 'GAT_prior_knowledge':
                # GAT_prior_knowledge params: [..., marker_effect, samples, top_rate] -> samples is second-to-last
                # Update ID ver4-6, R3.2a - task-local, re-resolved every task.
                _gat_pk_params = list(HPARAMETERS[base_model_name])
                if _gat_pk_params[-2] == 'all':
                    _gat_pk_params[-2] = test.shape[0]
                # Update ID ver4-5, R2 (I3 lockstep with _call_model()
                # above): 8th return value is this model's own already-
                # computed interaction pairs (empty unless its appended
                # emit_interaction hyperparameter is set).
                sample_pearson, sample_mse, sample_effect, predicted_test, predicted_valid, predicted_train, sample_attention, sample_interaction = GAT_prior_knowledge(train, valid, test, _gat_pk_params)
            elif _is_bio_prior_model(base_model_name):
                # HPARAMETERS[base_model_name] can be either:
                #   - a flat params list [neuron, dropout, lrate, decay, epoch, bsize, heads,
                #     network_json_path, gene_location_csv_path, marker_info_path, unit,
                #     include_mediated_edges, max_hops, data_driven_merge (dict, requirement 2),
                #     marker_effect, samples] -> the SAME network is used for every phenotype in
                #     PHENOTYPE, or
                #   - a dict {phenotype_name: params_list, ...} so each phenotype gets its own
                #     network_json_path/gene_location_csv_path (built from a different FLASH-P
                #     run or a different uploaded JSON per trait) - see
                #     Preprocess/gene_network_prior.py and the GUI's 'Biological Prior Network'
                #     tab. In both cases, samples is the last element of whichever list is used.
                #
                # MODEL_RUN[jj] is 'GAT_biological_prior_knowledge' itself, or a
                # numbered instance of it ('GAT_biological_prior_knowledge_2',
                # etc. - see _is_bio_prior_model) - each such name is its own,
                # completely independent key into HPARAMETERS, with its own
                # network/gene-location/etc. This is what lets a user compare
                # several different networks for the same phenotype(s) - e.g.
                # several separate FlashP samples, or several hand-uploaded
                # JSON files - as separate models in one run, each showing up
                # under its own name in every output file's 'model' column.
                bio_prior_hparams = HPARAMETERS[base_model_name]
                if isinstance(bio_prior_hparams, dict):
                    current_phenotype = sample.loc[i, 'phenotype']
                    if current_phenotype not in bio_prior_hparams:
                        raise KeyError(
                            f"HPARAMETERS[{MODEL_RUN[jj]!r}] is a per-phenotype "
                            f"dict but has no entry for phenotype '{current_phenotype}'. Add one, "
                            f"or pass a single flat params list instead to share one network "
                            f"across every phenotype in PHENOTYPE."
                        )
                    # Update ID ver4-6, R3.2a: list(...) - without this,
                    # bio_prior_params was a REFERENCE to the very same
                    # list object stored inside HPARAMETERS[base_model_
                    # name][current_phenotype], so the 'all' substitution
                    # below silently mutated the shared HPARAMETERS dict
                    # in place (the same class of bug as every other
                    # model branch's own 'all' substitution above - see
                    # the GBLUP branch's own comment for the full
                    # rationale), even though this branch already used a
                    # differently-named local variable.
                    bio_prior_params = list(bio_prior_hparams[current_phenotype])
                else:
                    bio_prior_params = list(bio_prior_hparams)

                if bio_prior_params[-1] == 'all':
                    bio_prior_params[-1] = test.shape[0]

                # Always the full, unpruned/unfiltered marker set - this
                # model picks its own candidate markers via the gene
                # network regardless of what LD pruning/RF importance
                # filtering did for other models this task (see
                # OTHER_MODELS_MARKER_SOURCE for the model-independent
                # choice that actually affects those other models instead).
                # For GENOTYPE_FORMAT == 'plink', bio_prior_pools (computed
                # further up) has this exact instance's OWN gene-window
                # extraction; for 'csv' (or if this instance's pool is
                # somehow missing) fall back to the shared
                # train_unpruned/valid_unpruned/test_unpruned triple, which
                # for 'csv' is simply the full, unrestricted genotype table -
                # the model filters it down to its own gene-network markers
                # internally regardless of which instance this is.
                train, valid, test = bio_prior_pools.get(base_model_name, (train_unpruned, valid_unpruned, test_unpruned)) \
                    if GENOTYPE_FORMAT == 'plink' else (train_unpruned, valid_unpruned, test_unpruned)

                _merge_source_data = bio_prior_merge_pools.get(base_model_name) \
                    if GENOTYPE_FORMAT == 'plink' else full_marker_pool

                # For GENOTYPE_FORMAT == 'plink', marker positions can only
                # correctly come from the SAME .bim file the extracted
                # genotype columns were pulled from above - override
                # whatever marker_info_path was configured in the GUI
                # (params[9]) with the .bim-derived one written near the top
                # of GP(), rather than risk desynchronising marker names
                # from their true positions.
                if GENOTYPE_FORMAT == 'plink':
                    bio_prior_params = list(bio_prior_params)
                    bio_prior_params[9] = bim_marker_info_path

                # Best-effort check that the network JSON this call is about
                # to use is actually the one meant for this phenotype -
                # catches e.g. an 'Upload' or 'Use a previous local FLASH-P
                # run' path pointed at the wrong file (a FLASH-P-generated
                # file's own metadata locks this in automatically; this is
                # the safety net for everything else, and for runs driven
                # from a saved config / HPC script rather than the GUI).
                # Heuristic and non-fatal on purpose (see
                # Preprocess/gene_network_prior.py's phenotype_matches_network_metadata
                # docstring) - printed as a warning, never raised.
                try:
                    matched, declared = phenotype_matches_network_metadata(
                        bio_prior_params[7], sample.loc[i, 'phenotype']
                    )
                    if matched is False:
                        print(
                            f"[{MODEL_RUN[jj]}] WARNING: '{bio_prior_params[7]}' "
                            f"declares itself to be for phenotype {declared!r}, which doesn't "
                            f"obviously match the phenotype {sample.loc[i,'phenotype']!r} it's "
                            f"about to be used for. This is a heuristic name check, not proof of "
                            f"a mistake - double-check this is the right file for this trait."
                        )
                except Exception:
                    pass  # never let this best-effort check block a real run

                sample_pearson, sample_mse, sample_effect, predicted_test, predicted_valid, predicted_train, sample_attention = GAT_biological_prior_knowledge(
                    train, valid, test, bio_prior_params, RESULT_NAME, sample.loc[i, 'phenotype'], MODEL_RUN[jj],
                    merge_source_data=_merge_source_data, network_cache=_bio_prior_merge_cache,
                )
            
            # Store prediction results
            record_sample = pd.DataFrame([{'population': sample.loc[i,'population'],
                                           'phenotype': sample.loc[i,'phenotype'],
                                           'model': MODEL_RUN[jj],
                                           'ratio': sample.loc[i,'ratio'],
                                           'sample': sample.loc[i,'sample'],
                                           'Pearson correlation': sample_pearson,
                                           'MSE': sample_mse}
                                          ])
            record = pd.concat([record, record_sample])
            
            if result_test_sample.shape[0]==0:
                result_test_sample = pd.DataFrame({'id': id_test,
                                                   'population': [sample.loc[i,'population']] * len(id_test),
                                                   'ratio': [sample.loc[i,'ratio']] * len(id_test),
                                                   'phenotype': [sample.loc[i,'phenotype']] * len(id_test),
                                                   'sample': [sample.loc[i,'sample']] * len(id_test),
                                                   'actual':test.iloc[:,-1],
                                                   MODEL_RUN[jj]:predicted_test
                                                  })
            else:
                result_test_sample = pd.concat([result_test_sample,
                                                pd.DataFrame({MODEL_RUN[jj]:predicted_test})
                                              ], axis=1)
           
            if (result_valid_sample.shape[0]==0 and type(sample.loc[i,'ratio']) is tuple) or (result_valid_sample.shape[0]==0 and W_OPT is not None and SCENARIO == 'between'):
                result_valid_sample = pd.DataFrame({'id': id_valid,
                                                   'population': [sample.loc[i,'population']] * len(id_valid),
                                                   'ratio': [sample.loc[i,'ratio']] * len(id_valid),
                                                   'phenotype': [sample.loc[i,'phenotype']] * len(id_valid),
                                                   'sample': [sample.loc[i,'sample']] * len(id_valid),
                                                   'actual':valid.iloc[:,-1],
                                                   MODEL_RUN[jj]:predicted_valid
                                                  })
            elif (result_valid_sample.shape[0]!=0 and type(sample.loc[i,'ratio']) is tuple) or (W_OPT is not None and SCENARIO == 'between'):
                result_valid_sample = pd.concat([result_valid_sample,
                                                pd.DataFrame({MODEL_RUN[jj]:predicted_valid})
                                              ], axis=1) 
            else:
                result_valid_sample = pd.DataFrame()
            
            if result_train_sample.shape[0]==0:
                result_train_sample = pd.DataFrame({'id': id_train,
                                                   'population': [sample.loc[i,'population']] * len(id_train),
                                                   'ratio': [sample.loc[i,'ratio']] * len(id_train),
                                                   'phenotype': [sample.loc[i,'phenotype']] * len(id_train),
                                                   'sample': [sample.loc[i,'sample']] * len(id_train),
                                                   'actual':train.iloc[:,-1],
                                                   MODEL_RUN[jj]:predicted_train
                                                  })
            else:
                result_train_sample = pd.concat([result_train_sample,
                                                pd.DataFrame({MODEL_RUN[jj]:predicted_train})
                                              ], axis=1)  
            
            if sample_effect.shape[0] != 0:
                expected_markers = train.columns.tolist()[:-1]
                if sample_effect.shape[1] != len(expected_markers):
                    raise ValueError(
                        f"Model '{MODEL_RUN[jj]}' returned a marker-effect table with "
                        f"{sample_effect.shape[1]} columns, but this task's training data "
                        f"(train.shape={train.shape}) has {len(expected_markers)} markers. "
                        f"Every model is expected to return exactly one effect value per "
                        f"marker.\n"
                        f"Diagnostic context for this failure:\n"
                        f"  Task: population={sample.loc[i,'population']!r} "
                        f"phenotype={sample.loc[i,'phenotype']!r} "
                        f"ratio={sample.loc[i,'ratio']!r} sample={sample.loc[i,'sample']!r}\n"
                        f"  HPARAMETERS['{base_model_name}'] = {HPARAMETERS[base_model_name]!r}\n"
                        f"  LD_prune applied to this model: "
                        f"{not _is_bio_prior_model(base_model_name) and LD_prune_effective is not None}"
                        + (f" | LD_prune config = {LD_prune_effective!r}" if LD_prune_effective is not None else "")
                        + f"\n  RF_filter applied to this model: "
                        f"{not _is_bio_prior_model(base_model_name) and RF_filter_effective is not None}"
                        + (f" | RF_filter config = {RF_filter_effective!r}" if RF_filter_effective is not None else "")
                    )
                sample_effect.columns = expected_markers
                effect_sample = pd.DataFrame([{'population': sample.loc[i,'population'],
                                               'phenotype': sample.loc[i,'phenotype'],
                                               'model': MODEL_RUN[jj],
                                               'ratio': sample.loc[i,'ratio'],
                                               'sample': sample.loc[i,'sample'],
                                               }])
                effect_sample = pd.concat([effect_sample,
                                           sample_effect.reset_index(drop=True),
                                          ],axis=1)
                effect = pd.concat([effect, effect_sample])
           
            # Update ID ver4-5, R2 (blueprint §4.4, invariant I14): was
            # `if base_model_name == 'RF' and ...` - any model the
            # registry says CAN emit interactions is now handled
            # identically here, not just RF. Whether it actually DID for
            # THIS task is still decided by sample_interaction itself
            # being non-empty (each model's own opt-in flag, e.g. RF's
            # get_interaction or GAT_prior_knowledge's emit_interaction,
            # already decided that inside the model call above).
            if emits_interactions(schema_key_of(base_model_name)) and sample_interaction.shape[0] != 0:
                sample_interaction['population'] = sample.loc[i,'population']
                sample_interaction['phenotype'] = sample.loc[i,'phenotype']
                sample_interaction['model'] = MODEL_RUN[jj]
                sample_interaction['ratio'] = str(sample.loc[i,'ratio'])
                sample_interaction['sample'] = sample.loc[i,'sample'] 
                
                interactions = pd.concat([interactions, sample_interaction])   
            
            if base_model_name == 'GAT_fully_connected' or base_model_name == 'GAT_prior_knowledge' or _is_bio_prior_model(base_model_name):
                sample_attention['population'] = sample.loc[i,'population']
                sample_attention['model'] = MODEL_RUN[jj]
                sample_attention['ratio'] = str(sample.loc[i,'ratio'])
                sample_attention['phenotype'] = sample.loc[i,'phenotype']
                sample_attention['sample'] = sample.loc[i,'sample'] 
                sample_attention.columns = ['marker1','marker2','value','population','model','ratio','phenotype','sample']
                sample_attention = sample_attention.loc[:,['population','phenotype','model','ratio','sample', 'marker1', 'marker2', 'value']]
                sample_attention = pd.concat([attention_total, sample_attention],axis=0)
                
                attention_total = pd.concat([attention_total, sample_attention], axis=0)

            print(f"[GP] Task {task_num}/{total_tasks} | model {MODEL_RUN[jj]} finished: "
                  f"Pearson r={sample_pearson:.4f}, MSE={sample_mse:.4f} "
                  f"(took {time.time() - model_start_time:.1f}s)")
            completed_units += 1

        # Weight optimisation
        #
        # Requirement: when more than one hyperparameter-tuning algorithm was
        # used for at least one model AND a weighted-ensemble method is
        # selected, the user chooses whether to combine models per tuning
        # method (one ensemble per algorithm - HP_TUNE_ENSEMBLE_MODE ==
        # 'per_method') or across every tuning method at once ('across_methods').
        # ensemble_groups() collapses to a single, unchanged group whenever
        # nothing was actually multi-algorithm-tuned (the overwhelmingly
        # common case), so this loop runs exactly once with group_label=None
        # and group_models == MODEL_RUN for every run that doesn't use this
        # feature - byte-for-byte the same calls as before this change.
        # Update ID 2, R1, gate G2 (blueprint §10.3/§10.6, decision D1):
        # weighted-ensemble finalisation needs BOTH every base model's
        # record/effect/weight rows for this task AND the frozen Step 8
        # pool (data_train/valid/test) - the pool only exists inside this
        # closure, so a filtered (model-level-fan-out) sub-run can never
        # correctly run it here. First delivery scope is base-model
        # dispatch only: a task with W_OPT active is made INELIGIBLE for
        # fan-out before any unit is submitted (see
        # intra_task_parallel.task_is_model_fanout_eligible()), so this
        # branch is defence-in-depth, not the primary safeguard - it
        # should never actually fire in a task the eligibility gate
        # already screened, but is logged loudly if it somehow does.
        if _dispatch_filter is not None and W_OPT is not None and \
                (type(sample.loc[i,'ratio']) is tuple or SCENARIO=='between'):
            print(f"[GP] Task {i - idx*interval + 1}/{total_tasks} | NOTE: MODEL_DISPATCH_FILTER "
                  f"active - W_OPT weighted-ensemble finalisation skipped in this sub-run (see "
                  f"intra_task_parallel.py's eligibility gate; a task with W_OPT active and a "
                  f"validation split should have been marked ineligible for fan-out before "
                  f"reaching here - this task's weighted ensembles will be missing unless that "
                  f"gate is investigated).")
        elif W_OPT is not None and (type(sample.loc[i,'ratio']) is tuple or SCENARIO=='between'):
            # Each weighted-ensemble method has its own hardcoded output
            # column name and record/effect/weight 'model' label (confirmed
            # from models/Linear_transformation.py, Nelder_Mead.py,
            # Bayesian_optimisation.py, Analytic_least_squares.py - note
            # 'Nelder Mead' uses a DIFFERENT string for its column
            # ('Nelder-Mead', hyphenated) than for its record/effect/weight
            # label ('Nelder Mead', spaced) - these are that same
            # distinction, not a typo).
            _WOPT_LABELS = {
                'Linear transformation': {'column': 'Linear transformation', 'label': 'Linear transformation'},
                'Nelder Mead': {'column': 'Nelder-Mead', 'label': 'Nelder Mead'},
                'Bayesian optimisation': {'column': 'Bayesian', 'label': 'Bayesian optimisation'},
                # Requirements.md item 5.
                'Analytic least-squares': {'column': 'Analytic least-squares', 'label': 'Analytic least-squares'},
            }
            for _group_label, _group_models in ensemble_groups(MODEL_BASE, HP_TUNE, HP_TUNE_ENSEMBLE_MODE):
                for kk in range(len(W_OPT)):
                    # Each group gets a CLEAN snapshot of record/effect/weight
                    # (not the growing, already-mutated one from a previous
                    # group in this same loop) as the input these functions
                    # mutate internally - Linear_transformation.py/
                    # Nelder_Mead.py/Bayesian_optimisation.py were only ever
                    # exercised once per task in the original design, and
                    # each duplicates-and-overwrites its OWN input's last row
                    # internally; chaining that through a second call in the
                    # same task (needed for per-method grouping) corrupts
                    # wide numeric columns on some pandas versions (verified:
                    # cells silently become 1-element lists/arrays instead of
                    # floats, later crashing the final ensemble()'s .abs()).
                    # Only predicted_test/valid/train_sample chain forward
                    # normally between groups - that's the *intended*,
                    # already-working behaviour (each method adds its own
                    # extra column to the same table).
                    _record_snapshot = record.copy()
                    _effect_snapshot = effect.copy()
                    _weight_snapshot = weight.copy()

                    if W_OPT[kk]  == 'Linear transformation':
                        _record_out, _effect_out, predicted_test_sample, predicted_valid_sample, predicted_train_sample, _weight_out = Linear_transformation(result_train_sample, result_valid_sample, result_test_sample, _record_snapshot, _effect_snapshot, _weight_snapshot, _group_models, HYPERPARAMETERS_OPT['Linear transformation'])
                    elif W_OPT[kk] == 'Nelder Mead':
                        _record_out, _effect_out, predicted_test_sample, predicted_valid_sample, predicted_train_sample, _weight_out = Nelder_Mead(result_train_sample, result_valid_sample, result_test_sample, _record_snapshot, _effect_snapshot, _weight_snapshot, _group_models, HYPERPARAMETERS_OPT['Nelder Mead'])
                    elif W_OPT[kk] == 'Bayesian optimisation':
                        _record_out, _effect_out, predicted_test_sample, predicted_valid_sample, predicted_train_sample, _weight_out = Bayesian(result_train_sample, result_valid_sample, result_test_sample, _record_snapshot, _effect_snapshot, _weight_snapshot, _group_models, HYPERPARAMETERS_OPT['Bayesian optimisation'])
                    elif W_OPT[kk] == 'Analytic least-squares':
                        _record_out, _effect_out, predicted_test_sample, predicted_valid_sample, predicted_train_sample, _weight_out = Analytic_least_squares(result_train_sample, result_valid_sample, result_test_sample, _record_snapshot, _effect_snapshot, _weight_snapshot, _group_models, HYPERPARAMETERS_OPT['Analytic least-squares'])
                    else:
                        continue

                    # Only the row(s) this call actually added are new -
                    # append those onto the real accumulators (clean concat,
                    # not chained mutation).
                    _new_record = _record_out.iloc[_record_snapshot.shape[0]:]
                    _new_effect = _effect_out.iloc[_effect_snapshot.shape[0]:]
                    _new_weight = _weight_out.iloc[_weight_snapshot.shape[0]:]

                    if _group_label is not None:
                        # Differentiate this group's weighted-ensemble output
                        # from every other algorithm-group's (Requirement:
                        # suffix models per tuning method when more than one
                        # is in play) - the three model files above are
                        # otherwise untouched; this is purely a rename of
                        # their already-produced output.
                        _col = _WOPT_LABELS[W_OPT[kk]]['column']
                        _lbl = _WOPT_LABELS[W_OPT[kk]]['label']
                        _new_lbl = f'{_lbl}__{_group_label}'
                        for _df in (predicted_test_sample, predicted_valid_sample, predicted_train_sample):
                            if isinstance(_df, pd.DataFrame) and _col in _df.columns:
                                _df.rename(columns={_col: _new_lbl}, inplace=True)
                        if _new_record.shape[0] != 0:
                            _new_record = _new_record.copy()
                            _new_record.loc[_new_record['model'] == _lbl, 'model'] = _new_lbl
                        if _new_effect.shape[0] != 0:
                            _new_effect = _new_effect.copy()
                            _new_effect.loc[_new_effect['model'] == _lbl, 'model'] = _new_lbl
                        if _new_weight.shape[0] != 0 and 'model' in _new_weight.columns:
                            _new_weight = _new_weight.copy()
                            _new_weight.loc[_new_weight['model'] == _lbl, 'model'] = _new_lbl

                    record = pd.concat([record, _new_record])
                    effect = pd.concat([effect, _new_effect])
                    weight = pd.concat([weight, _new_weight])

                    # Requirement_patch3.md item 1: weighted-ensemble
                    # marker-PAIR interaction ring for THIS task, reusing
                    # this method's own already-computed per-model weight
                    # row (_new_weight - the SAME weights this method just
                    # used, above, to build _new_effect's weighted marker-
                    # effect ring) - see models/interaction_extraction.py
                    # ::weighted_ensemble_interactions()'s own docstring
                    # for the full contract. `_new_weight` already carries
                    # this call's (possibly tuning-group-relabelled, see
                    # the `if _group_label is not None:` block just above)
                    # own 'model' label, which becomes this interaction
                    # ring's own 'model' value too - keeping the
                    # interaction ring's name identical to its
                    # already-written effect/record ring's name for the
                    # exact same task. A no-op (nothing appended) whenever
                    # this task has no interaction data at all yet, or
                    # none of `_group_models` reported any for it.
                    if _new_weight.shape[0] != 0 and interactions.shape[0] != 0:
                        _method_label = _new_weight['model'].iloc[-1]
                        _model_weights = {
                            _m: float(_new_weight.iloc[-1][_m])
                            for _m in _group_models if _m in _new_weight.columns
                        }
                        _new_interaction_ensemble = weighted_ensemble_interactions(
                            interactions, _group_models, _model_weights,
                            sample.loc[i, 'population'], sample.loc[i, 'phenotype'],
                            sample.loc[i, 'ratio'], sample.loc[i, 'sample'], _method_label,
                        )
                        interactions = pd.concat([interactions, _new_interaction_ensemble])
            
        result_test = pd.concat([result_test, result_test_sample],axis=0)
        result_valid = pd.concat([result_valid, result_valid_sample],axis=0)
        result_train = pd.concat([result_train, result_train_sample],axis=0)
        
        result_train = result_train.sort_values(['id']).reset_index(drop=True)
        
        if type(sample.loc[i,'ratio']) is tuple:
            result_valid = result_valid.sort_values(['id']).reset_index(drop=True)
        result_test = result_test.sort_values(['id']).reset_index(drop=True)
        
    # ---------------------------------------------------------------------- #
    # Checkpoint/resume (continued): the actual driving loop. Each task gets
    # a full snapshot of every accumulator taken before it starts - if it
    # raises, those snapshots are restored (discarding any partial
    # contribution the failed task may already have made - e.g. its first
    # model succeeded and got appended to `record` before its second model
    # raised), so a checkpoint is always all-or-nothing per task: a resumed
    # run always re-does the ENTIRE failed task cleanly, never a
    # half-finished one, which would otherwise risk duplicate rows for
    # whichever model(s) did complete before the failure. Every task that
    # finishes cleanly falls through to the next iteration normally, exactly
    # as the original single for-loop did - nothing about a successful run's
    # behaviour changes here.
    # ---------------------------------------------------------------------- #
    for i in range(_start_i, idx*interval+interval):

        if i >= sample.shape[0]:
            break

        _pre_task_snapshot = {
            'record': record, 'result_train': result_train, 'result_valid': result_valid,
            'result_test': result_test, 'effect': effect, 'interactions': interactions,
            'weight': weight, 'attention_total': attention_total, 'hp_record': hp_record,
            'stats': stats,
        }
        try:
            _process_one_task(i)
        except Exception as _task_exc:
            # Requirement: "save all the results so far when an error
            # occurs" - roll back this task's own (possibly partial)
            # contribution first, then persist everything that finished
            # BEFORE it, to the exact same files a successful run would
            # produce (see save_partial_results()'s own docstring for why
            # that's the same helper the final, successful-run save below
            # also uses).
            record = _pre_task_snapshot['record']
            result_train = _pre_task_snapshot['result_train']
            result_valid = _pre_task_snapshot['result_valid']
            result_test = _pre_task_snapshot['result_test']
            effect = _pre_task_snapshot['effect']
            interactions = _pre_task_snapshot['interactions']
            weight = _pre_task_snapshot['weight']
            attention_total = _pre_task_snapshot['attention_total']
            hp_record = _pre_task_snapshot['hp_record']
            stats = _pre_task_snapshot['stats']

            _task_num = i - idx*interval  # 0-based position of the FAILED task within this batch
            # Req 2 fix (2026-09): full traceback added, not just
            # repr(_task_exc) - this line is the LAST resort for
            # diagnosing a task failure (everything upstream that could
            # have printed a more specific NOTE either already did, or
            # the exception travelled here from somewhere that never had
            # its own except block at all). Without a traceback, this
            # line alone cannot distinguish which of dozens of possible
            # call sites actually raised - exactly the ambiguity that
            # made this requirement's root cause take an entire prior
            # investigation session to track down (see
            # REQ2_Handoff.md §6 item 1, the top recommended next step).
            print(f"[GP] ERROR: task {_task_num + 1}/{total_tasks} (population="
                  f"{sample.loc[i,'population']!r}, phenotype={sample.loc[i,'phenotype']!r}, "
                  f"ratio={sample.loc[i,'ratio']!r}, replicate={sample.loc[i,'sample']!r}) "
                  f"failed: {_task_exc!r}\n"
                  f"{traceback.format_exc()}")
            print(f"[GP] Saving the {_task_num} task(s) completed successfully before this "
                  f"failure, so they aren't lost...")
            _ckpt.save_partial_results(
                RESULT_NAME, idx, _is_parallel,
                {'record': record, 'result_train': result_train, 'result_valid': result_valid,
                 'result_test': result_test, 'effect': effect, 'interactions': interactions,
                 'attention_total': attention_total, 'weight': weight, 'hp_record': hp_record,
                 'stats': stats},
                _static_write_flags, compression=RESULT_COMPRESSION,
            )
            _ckpt.save_checkpoint(RESULT_NAME, idx, _is_parallel, _sample_fp,
                                   last_completed_i=_task_num - 1, total_tasks=total_tasks)
            print(f"[GP] Partial results saved under './Result/{RESULT_NAME}/'. Fix the error "
                  f"above and re-submit the same job (same config{', same PARALLEL batch' if _is_parallel else ''}) "
                  f"- EasiGP will automatically resume from task {_task_num + 1}/{total_tasks} "
                  f"instead of starting over.")
            raise
        else:
            # Requirement (bugfix - output files existing but empty after
            # an OOM/walltime kill, so a resumed run silently re-does
            # everything): the block above only ever saves anything on a
            # CAUGHT Python exception - which a SIGKILL (OOM) or a
            # scheduler's walltime kill is not, and by design can never
            # be. Without a save point here too, a kill landing between
            # two successfully-finished tasks (or mid-way through the
            # next one) loses EVERY task this batch has completed, not
            # just the one actually running at the time - exactly the
            # reported symptom (checkpoint files existing, but the result
            # files they pointed at empty, because nothing had ever
            # written them). This runs after EVERY task that finishes
            # without raising (a plain try/except/else - 'else' means
            # "the try block did NOT raise", not "always", so this never
            # runs for a task the block above already handled) -
            # incrementally APPENDING just this one task's own new rows
            # (see append_partial_results()'s own docstring for why an
            # append, not the same full-rewrite save_partial_results()
            # uses, is what keeps this affordable to do on every single
            # task rather than only occasionally) - and advancing the
            # checkpoint to match, so a kill immediately afterward has
            # nothing left to lose except whatever task hadn't finished
            # yet, which is unavoidable no matter what strategy is used.
            _task_num = i - idx*interval  # 0-based position of the task that just finished
            _delta_frames = {
                'record': record.iloc[len(_pre_task_snapshot['record']):],
                'result_train': result_train.iloc[len(_pre_task_snapshot['result_train']):],
                'result_valid': result_valid.iloc[len(_pre_task_snapshot['result_valid']):],
                'result_test': result_test.iloc[len(_pre_task_snapshot['result_test']):],
                'effect': effect.iloc[len(_pre_task_snapshot['effect']):],
                'interactions': interactions.iloc[len(_pre_task_snapshot['interactions']):],
                'attention_total': attention_total.iloc[len(_pre_task_snapshot['attention_total']):],
                'weight': weight.iloc[len(_pre_task_snapshot['weight']):],
                'hp_record': hp_record.iloc[len(_pre_task_snapshot['hp_record']):],
                'stats': stats.iloc[len(_pre_task_snapshot['stats']):],
            }
            _ckpt.append_partial_results(RESULT_NAME, idx, _is_parallel, _delta_frames, _static_write_flags,
                                          compression=RESULT_COMPRESSION)
            _ckpt.save_checkpoint(RESULT_NAME, idx, _is_parallel, _sample_fp,
                                   last_completed_i=_task_num, total_tasks=total_tasks)

    _report_progress(completed_units, total_units, label='Finalising results...')

    # Run the (naive, arithmetic-mean) ensemble model in the end - same
    # per-method/across-methods grouping as the weighted W_OPT loop above,
    # and the same guarantee: collapses to the original single call whenever
    # nothing was multi-algorithm-tuned.
    #
    # Bug fix ("tasks finished but no output files saved"): this step runs
    # AFTER every task in the main loop above has already succeeded and
    # been saved - but until now, nothing protected this step ITSELF. A
    # failure here (confirmed in production - see models/ensemble.py's own
    # bugfix history for the exact crash) used to propagate as a bare,
    # uncaught exception, discarding this entire run's output even though
    # every task's own (expensive, hours-long) results had already been
    # safely written to disk well before this point - the crash just
    # happened after the LAST point anything got saved. Wrapped in the
    # same snapshot/rollback/save-on-error pattern the main per-task loop
    # above already uses for the same reason, so a crash here now only
    # ever costs re-running this (comparatively fast) aggregation step,
    # never any of the per-task model tuning that already finished.
    _pre_finalise_snapshot = {
        'record': record, 'result_train': result_train,
        'result_valid': result_valid, 'result_test': result_test, 'effect': effect,
        # Requirement_patch3.md item 1: finalisation now also appends the
        # naive-ensemble interaction rows to `interactions` (see the
        # 'ensemble' in MODEL_BASE branch below) - snapshotted here for
        # exactly the same reason record/result_*/effect already are (see
        # the except: block's own comment on why a partial finalisation
        # must roll back ENTIRELY, not just save whatever succeeded).
        'interactions': interactions,
    }
    try:
        # Update ID 2, R1, gate G3 (blueprint §10.3): a filtered
        # (model-level-fan-out) sub-run only ever dispatches a SUBSET of
        # MODEL_RUN, so its own naive/weighted-ensemble finalisation
        # would average the wrong (partial) set of models - the real
        # ensemble is recomputed once, after every model-group's results
        # for this task have been tier-1-merged back together, by
        # intra_task_parallel.finalise_task_from_merged() (which calls
        # this exact same models.ensemble.ensemble() on the merged
        # frames - see that function's own docstring). Skipped here and
        # logged once per (filtered) run rather than per task, since this
        # finalisation block itself runs exactly once per GP() call, not
        # once per task.
        if _dispatch_filter is not None:
            print(f"[GP] NOTE: MODEL_DISPATCH_FILTER active - naive/weighted-ensemble "
                  f"finalisation skipped for this sub-run (recomputed after the tier-1 "
                  f"model-group merge instead - see intra_task_parallel.finalise_task_from_merged()).")
        elif 'ensemble' in MODEL_BASE:
            for _group_label, _group_models in ensemble_groups(MODEL_BASE, HP_TUNE, HP_TUNE_ENSEMBLE_MODE):
                # Requirement_patch3.md item 1: `interactions` is now
                # passed through too, so this same finalisation call also
                # produces a naive-ensemble marker-pair interaction ring
                # (model_registry.emits_interactions() decides which of
                # `_group_models` actually contributed anything - see
                # models/interaction_extraction.py::naive_ensemble_
                # interactions()'s own contract) whenever at least one
                # selected model returned interaction data for at least
                # one task - a no-op (empty sample_interaction_ensemble)
                # otherwise, exactly like sample_effect already is when no
                # selected model returned marker effects.
                result_train, result_valid, result_test, sample_record, sample_effect, sample_interaction_ensemble = ensemble(result_train, result_valid, result_test, effect, _group_models, interactions)
                if _group_label is not None:
                    _new_lbl = f'ensemble__{_group_label}'
                    for _df in (result_train, result_valid, result_test):
                        if isinstance(_df, pd.DataFrame) and 'ensemble' in _df.columns:
                            _df.rename(columns={'ensemble': _new_lbl}, inplace=True)
                    sample_record['model'] = _new_lbl
                    if sample_effect.shape[0] != 0:
                        sample_effect['model'] = _new_lbl
                    if sample_interaction_ensemble.shape[0] != 0:
                        sample_interaction_ensemble = sample_interaction_ensemble.copy()
                        sample_interaction_ensemble['model'] = _new_lbl
                record = pd.concat([record, sample_record])
                effect = pd.concat([effect, sample_effect])
                interactions = pd.concat([interactions, sample_interaction_ensemble])
    except Exception as _finalise_exc:
        # Roll back this (possibly partial) finalisation attempt entirely,
        # rather than trying to save whatever partly succeeded: unlike a
        # single task in the main loop above (which either fully completes
        # or contributes nothing), a partially-completed finalisation
        # could already have renamed an 'ensemble' column and/or appended
        # one ensemble group's rows before a LATER group failed -
        # re-running it from that partial state on resume risks silently
        # double-adding those rows or re-deriving from already-renamed
        # columns. Rolling all the way back to right before finalisation
        # started keeps "resume = redo the whole finalisation step from
        # scratch" exactly correct and duplicate-free - cheap to redo in
        # full, since it's simple arithmetic aggregation over
        # already-computed predictions, not model retraining.
        record = _pre_finalise_snapshot['record']
        result_train = _pre_finalise_snapshot['result_train']
        result_valid = _pre_finalise_snapshot['result_valid']
        result_test = _pre_finalise_snapshot['result_test']
        effect = _pre_finalise_snapshot['effect']
        interactions = _pre_finalise_snapshot['interactions']

        print(f"[GP] ERROR: the final ensemble/aggregation step failed: {_finalise_exc!r}")
        print(f"[GP] Saving the results from all {total_tasks} already-completed task(s) so "
              f"they aren't lost...")
        _ckpt.save_partial_results(
            RESULT_NAME, idx, _is_parallel,
            {'record': record, 'result_train': result_train, 'result_valid': result_valid,
             'result_test': result_test, 'effect': effect, 'interactions': interactions,
             'attention_total': attention_total, 'weight': weight, 'hp_record': hp_record,
             'stats': stats},
            _static_write_flags, compression=RESULT_COMPRESSION,
        )
        # Deliberately NOT clearing the checkpoint here - it still
        # correctly records every task as completed (see
        # checkpoint_utils.load_checkpoint()'s own comment for why that no
        # longer means "nothing to resume"), which is exactly what lets a
        # re-submitted run skip straight past the per-task loop above and
        # retry just this finalisation step, instead of redoing every
        # task's model tuning.
        print(f"[GP] Partial results saved under './Result/{RESULT_NAME}/'. Fix the error "
              f"above and re-submit the same job (same config{', same PARALLEL batch' if _is_parallel else ''}) "
              f"- EasiGP will automatically resume by re-running just the final ensemble/"
              f"aggregation step, without redoing any task's model tuning.")
        raise

    # Store the results. Reuses checkpoint_utils.save_partial_results() -
    # the SAME helper an in-progress task failure above uses to persist
    # whatever completed before it - so a full, successful run and a
    # not-yet-finished, checkpointed run always produce byte-for-byte the
    # same kind of output for the data they each actually have (see that
    # helper's own docstring).
    _ckpt.save_partial_results(
        RESULT_NAME, idx, _is_parallel,
        {'record': record, 'result_train': result_train, 'result_valid': result_valid,
         'result_test': result_test, 'effect': effect, 'interactions': interactions,
         'attention_total': attention_total, 'weight': weight, 'hp_record': hp_record,
         'stats': stats},
        _static_write_flags, compression=RESULT_COMPRESSION,
    )

    # Every task in this batch finished successfully - nothing left to
    # resume, so the checkpoint (if one existed, from an earlier failed
    # attempt at this same batch) is no longer needed. Left in place, it
    # could otherwise make an unrelated FUTURE run that happens to reuse
    # this RESULT_NAME think it should skip tasks it hasn't actually run.
    _ckpt.clear_checkpoint(RESULT_NAME, idx, _is_parallel)

    return record, result_train, result_test, effect, interactions, POPULATION, PHENOTYPE, attention_total
