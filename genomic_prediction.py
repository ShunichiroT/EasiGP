import pandas as pd
import numpy as np
import os
import time
import traceback
import hashlib
import io
import math
# Update ID ver4-6, R3.2a: used ONCE, to build HPARAMETERS_BASELINE - a
# frozen, never-mutated snapshot of the user's own configured HPARAMETERS,
# taken at the top of GP() (see the MODEL_RUN construction below). stdlib,
# no new dependency (I12).
import copy
import dataclasses
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
# ver4-102 (P4): opt-in seed bagging (explicit imports; the module has no star-exported names).
from models.seed_bag import (resolve_seed_bag, fingerprint_view as seed_bag_fingerprint_view,
                             bag_applies as seed_bag_applies, bag_call as seed_bag_call)
from models.GAT_fully_connected import *
from models.GAT_prior_knowledge import *
from models.GAT_biological_prior_knowledge import *
from models.ensemble import *
# ver4-25, Stage 7 (design blueprint R13): exact-predictor marker-effect
# routes for GBLUP ('closed_form')/RKHS ('shapiq_predictor') and the
# shared closed-form linear-Shapley helper Requirements3.md item 1 reuses
# for rrBLUP/BayesB - see models/kernel_effects.py's own module docstring.
from models import kernel_effects

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
# ver4-33, R1.c: memoised drop-in for select_markers_for_data_driven_network
# (same arguments, same (markers, fitted_rf) return; fitted_rf is None on a
# memo hit - no caller in this file uses it).
from Preprocess.preprocess_memo import memo_select_markers
from Preprocess.gene_network_prior import (
    phenotype_matches_network_metadata, load_network_json, extract_candidate_genes, build_gene_list,
)
from Preprocess.plink_io import (
    validate_plink_fileset, read_bim_marker_info, write_shared_bim_marker_info, read_fam_iids,
    plink_to_genotype_df, ld_prune_plink_native, ld_prune_marker_list_native,
    PlinkFilesetError,   # ver4-94 (R3)
)
# _map_genes_to_markers is a "private" (leading-underscore) helper inside the
# model file, reused here deliberately - it's the exact same SNP<->gene
# interval-overlap logic GAT_biological_prior_knowledge.py itself uses, and
# reusing it (rather than re-implementing the same logic a second time) is
# what lets this module determine "which markers fall inside a gene window"
# up front, to --extract only those from a PLINK fileset, while guaranteeing
# it's byte-for-byte the same computation the model will redo internally.
from models.GAT_biological_prior_knowledge import _map_genes_to_markers, preview_gene_window_markers
# ver4-61, R1: build the hyperparameter-independent bio-prior graph once per task.
from models.GAT_biological_prior_knowledge import prewarm_bio_graph, bio_prior_trial_peak_bytes

from models.Linear_transformation import *
from models.Nelder_Mead import *
from models.Bayesian_optimisation import *
from models.Analytic_least_squares import *
# ver4-57 (R2): the opt-in Iterative ensemble (Step 11c). Pure module - imports models.* and
# pipeline_utils only, never this file (I2).
from models.iterative_ensemble import IterMember, run_iterative_ensemble, two_phase_wave_sizes
from models.super_learner import (run_super_learner, sl_unit_counts, sl_tuning_folds,   # ver4-77 (R2); ver4-85
                                  _SL_POP_OFFSET)   # ver4-94 (R3)
from models.relatedness import build_population_source, MIN_SOURCE_LINES   # ver4-94 (R3)
from models.interaction_extraction import weighted_ensemble_interactions

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

from hparam_specs import HPARAM_SPECS, field_index
from models.hyperparameter_tuning import (
    tune_model_hyperparameters, expand_model_list, ensemble_groups, base_of, algo_of,
    schema_key_of, resolve_trial_fanout_width, search_box_highs,
)

# ver4-61, R4: position of the bio-prior 'Number of samples for marker effects' field.
# The 'all' -> test-row-count substitution used to read params[-1], which pointed at the wrong
# field as soon as a field was appended after it (I5). Looked up once, by label.
_BIO_PRIOR_SAMPLES_IDX = field_index(
    'GAT_biological_prior_knowledge', 'Number of samples for marker effects')
# ver4-26, R2: the joint hyperparameter+ensemble-weight search - a single
# self-contained module (see its own docstring) that composes the
# single-authority helpers above rather than re-deriving them.
from models.joint_tuning import (
    tune_joint_hyperparameters_and_weights, _v428_nondefault,
    # ver4-44 (R0): the pre-flight check of search_strategy (see _GP_single).
    validate_search_strategy,
)
from model_registry import (
    emits_interactions, has_no_interaction_capability, no_interaction_capability_reason,
    is_available, optional_dependency_for, interaction_method_default,
    lane_capable, global_rng_streams,
)
import async_lane
import checkpoint_utils as _ckpt
# ver4-56 (R1): log-only per-model/task/job usage records (no EasiGP imports, I2).
import resource_telemetry as _rtele
from resource_calibration import pool_signature as _calib_pool_signature
from pipeline_utils import (
    unify_columns_by_position, resolve_compute_resources, set_active_compute_resources,
    apply_r_blas_threads, result_dir_path, cleanup_bglr_output, wide_safe_r_function, ensure_rpy2_thread_safe_release,
    r_list_get, safe_regression_metrics, nested_safe_n_jobs, configure_r_environment,
    # ver4-79 (R1): paired gap in r (with bootstrap SE) for the Super learner's log line.
    paired_r_gap,
    # ver4-38, R3: one deduplicated WARN when a model's prediction is constant.
    warn_if_constant_prediction, is_lean_search_fit, tuning_sanity_report,
    init_rpy2_conversion, install_gpu_semaphore, get_gpu_semaphore,
    # ver4-33 (R1.a / R1.c): per-task stage timing and the shared memo dir.
    StageTimer, set_current_stage_timer, timed_stage, begin_stage, end_stage,
    ensure_memo_dir_env, release_memo_dir_env,
    # ver4-56 (R1): where the telemetry JSONL files go (set once by the outermost entry point).
    ensure_telemetry_dir_env, release_telemetry_dir_env,
    # ver4-35 (R1.e / R1.f): worker thread caps, the GPU arbiter's lifetime.
    apply_worker_thread_limits, ensure_gpu_arbiter, release_gpu_arbiter, current_stage_timer,
    # ver4-43 (R1): the run's resolved ACCELERATION values, read by the RKHS
    # `baseline_shap` interaction route (which has no compute-resources parameter).
    get_active_compute_resources,
    # ver4-61 (R2): free host memory, for the bio-prior trial-memory guard.
    available_memory_bytes,
    # ver4-61 (R6): one normaliser for RATIO values in tuning-cache keys.
    restore_ratio,
    # ver4-44 (R0): the engine version and the stale-engine check run at the top of GP().
    ENGINE_VERSION, stale_check_enabled, stale_engine_modules, stale_engine_message,
    # ver4-57 (R2/R5): the ITERATIVE_ENSEMBLE resolver (single authority for defaults/ranges).
    resolve_iterative_ensemble,
    # ver4-77 (R2): the SUPER_LEARNER resolver (single authority for defaults/ranges).
    resolve_super_learner,
    # ver4-80 (R1): which weight methods the Super learner writes rows for, and their labels.
    sl_weight_methods, sl_row_labels,
    # ver4-94 (R3): the labels of the optional pooled row.
    sl_tagged_row_labels,
    sl_member_row_labels,
    # ver4-85: out-of-fold hyperparameter tuning switch and the SL fingerprint view.
    sl_oof_tuning_on, sl_fingerprint_view,
    # ver4-86: the cost-saving plan of out-of-fold tuning (SUPER_LEARNER['oof_tuning_fast']).
    sl_oof_tuning_plan,
    # ver4-68 (R1): the two-phase mode's checkpoint-fingerprint view, tier name and ignored keys.
    iter_ens_fingerprint_view, iter_ens_tier_of, _ITER_ENS_TWO_PHASE_IGNORED,
    derive_population_series, collapse_raw_to_analysed_level, resolve_trait_columns,
    # Requirements.md (between-prediction validation-set patch): the SAME
    # population-combination authority main_app.py uses to populate the
    # exclusion multiselect (see that module's own comment) - used below
    # to build the 'between' `sample` table's population pairs/triples.
    between_population_combinations, build_sample_table,
    # ver4-25, Stage 7 (R13): the SAME shap.KernelExplainer helpers
    # models/KNN.py's/models/SVR.py's own Shapley marker-effect routes
    # already use - reused here for RKHS's 'shap_predictor' route (the
    # renamed pre-existing implementation - see
    # _apply_rkhs_shapiq_predictor_effect()'s own docstring) so its
    # explanation follows this codebase's one established KernelExplainer
    # convention rather than a second, bespoke one.
    safe_shap_kmeans_background, parallel_shap_values,
    # This update (RKHS marker-effect comparability fix): RKHS's own
    # 'shapiq_predictor' choice previously never actually called the
    # `shapiq` package despite its name (it called `shap.KernelExplainer`
    # above, exactly like 'shap_predictor' now does) - the SAME real-
    # shapiq helper SVR/KNN/RF/GBDT/XGBoost/ExtraTrees already use for
    # their own 'shapiq' marker-effect choice, imported here so RKHS's
    # 'shapiq_predictor' finally means the same thing everywhere in this
    # codebase and marker effects are genuinely comparable across models.
    shapiq_marker_effects,
    # Requirements.md item 1 (this update): "the current KernelSHAP is
    # very slow ... when the total number of markers is > 10,000 for
    # GBLUP, RKHS, SVR and KNN" - the SAME shared-coalition, batched
    # KernelSHAP estimator models/SVR.py's/models/KNN.py's own
    # 'kernel_shap' route now switches to above that marker count, reused
    # here for RKHS's 'shap_predictor' route (its own only Python-side
    # shap.KernelExplainer call site).
    fast_kernel_shap_effects, FAST_KERNEL_SHAP_MARKER_THRESHOLD,
    # This update (RKHS non-surrogate 'shapiq_predictor' interaction
    # route - see _apply_rkhs_predictor_interaction()'s own docstring):
    # the SAME real-shapiq pairwise-interaction helper models/SVR.py's/
    # models/KNN.py's own 'shapiq' interaction choice already calls,
    # reused here so RKHS can explain its OWN exact kernel reconstruction
    # (RKHSPredictor - already used, and already V13.2-gated, for marker
    # EFFECTS via _apply_rkhs_shapiq_predictor_effect()) directly, instead
    # of only ever being able to explain a fitted-on-predictions RF/H-
    # statistic surrogate the way _r_model_surrogate_interaction() does.
    shapiq_pairwise_interactions, resolve_shapiq_explain_row_count,
)
import shap
# Update ID ver4-12 (design blueprint SS2.5 "Module layout"): asreml_syntax
# is a Layer 3d shared authority (zero project imports - see its own module
# docstring), safe to import unconditionally here exactly like
# model_registry/hparam_specs/circos_geometry already are. Preprocess.
# asreml_license/asreml_blup are ordinary Layer 3a preprocessing modules -
# neither imports rpy2 (or anything else R-related) at MODULE scope, only
# inside the specific functions that actually need it (mirroring every
# other Preprocess module in this import block), so importing them here
# unconditionally never requires rpy2/asreml to be installed just to import
# genomic_prediction.py itself (e.g. for static analysis, or a script that
# imports every models/*.py without ever calling GP()).
import asreml_syntax
from Preprocess import asreml_license
from Preprocess.asreml_blup import (
    asreml_adjust_phenotypes, apply_to_splits, build_or_load_grm,
    # ver4-19, Requirements.md R1/R2/R3 - see this module's own Step 6b
    # block (near ASREML dispatch) for how each of these is used.
    resolve_grm_genotype_universe, log_grm_universe_diagnostics,
    resolve_response_unit, fit_unmasked_adjustment, fit_concurrent_variance_model,
    describe_genotype_effect_strategy,
    # ver4-25, R7 (design blueprint stage 4) - the two-step refined-BLUE
    # procedure's own step (i).
    fit_blue_refinement_step1,
    # ver4-7, R1 (design blueprint "Per-Set GRM Generation for GBLUP") -
    # the 3-independent-fit per-split GRM dispatch.
    fit_per_set_grm_adjustment,
)
from Preprocess import asreml_blup
from models.interaction_extraction import (
    surrogate_tree_interactions, surrogate_h_statistic_interactions, surrogate_shapiq_interactions,
    shortlist_markers, top_select_by_order, pad_shapiq_pairs_to_shap_parity,
    # This update - see _apply_rkhs_predictor_interaction()'s own
    # docstring: the SAME canonicalisation SVR.py's/KNN.py's own
    # 'shapiq' interaction route already applies to
    # pipeline_utils.shapiq_pairwise_interactions()'s raw output.
    canonicalize_shapiq_table,
    # This update (RKHS 'friedman_h_predictor' interaction route - see
    # _apply_rkhs_predictor_interaction()'s own docstring): the SAME
    # native H-statistic helper models/SVR.py's/models/KNN.py's own
    # DEFAULT 'friedman_h' choice already calls directly on their real
    # fitted model's own `.predict` (no RandomForest surrogate) - reused
    # here on the masked RKHSPredictor.predict() instead, giving RKHS a
    # SECOND non-surrogate, non-'shapiq' interaction route (alongside
    # 'kernel_hessian', below) for anyone who does not want to rely on
    # the optional 'shapiq' package.
    h_statistic_interactions,
    # This update (user feedback, backed by this project's own
    # experimental results: "friedman-h was not reliable ... did not
    # capture correct interactions" and "shapiq ... too computationally
    # expensive"): a FOURTH non-surrogate RKHS interaction route,
    # directly comparable in units/aggregation to models/RF.py's own
    # TreeSHAP interaction table (same `abs_then_sum` reduction), and
    # cheaper than shapiq at equal background-size settings - see
    # baseline_shap_interactions()'s own module-level docstring (models/
    # interaction_extraction.py) for the full rationale.
    baseline_shap_interactions,
    # ver4-43 (R2): RKHS's opt-in per-marker 'baseline_shap' effect.
    baseline_shap_effects,
    # This update ("use marker effects to select markers for interaction
    # search?" checkbox) - the SAME single-authority eligibility function
    # models/RF.py's, models/ExtraTrees.py's, models/XGBoost.py's,
    # models/SVR.py's and models/KNN.py's own identical checkbox already
    # calls. RKHS's own interaction routes (_r_model_surrogate_interaction()
    # and _apply_rkhs_predictor_interaction(), below) call this too, gated
    # on `model_name == 'RKHS'` so rrBLUP/BayesB/GBLUP (which share
    # _r_model_surrogate_interaction() but have no marker-effect shortlist
    # of their own) are completely unaffected (I11).
    should_use_effect_for_interaction_shortlist,
)
# Requirements.md (BLUEs/BLUPs accuracy-metrics phase) - the three extra
# per-model accuracy metrics (adjusted prediction accuracy, Spearman
# correlation, Top-k Coincidence). Layer 3d shared authority, zero project
# imports (see its own module docstring) - safe to import unconditionally
# exactly like asreml_syntax/model_registry already are.
import asreml_accuracy_metrics
import blue_weights


def _joint_safe_row_normalize(df):
    """ver4-26: a local, module-level copy of
    `models.Bayesian_optimisation._safe_row_normalize()` (a private,
    underscore-prefixed name - not reachable via this file's own
    `from models.Bayesian_optimisation import *` at the top of this
    module, since a wildcard import never pulls in underscore-prefixed
    names). Used only by the new Step 11 `elif 'Joint Bayesian
    optimisation' in (JOINT_W_OPT or []):` branch below, to row-normalise
    marker-effect contributions before weighting them - identical
    behaviour and identical bug-fix rationale (never divide by an exact
    0, force float64 first) as the original; kept as a small, separate
    copy rather than importing the private name directly, matching this
    codebase's own precedent of small intentional duplication over a
    cross-module private import (see e.g. `_WOPT_LABEL_PREFIXES`'s own
    independent copies in pipeline_utils.py/assemble.py/batch_reader.py)."""
    df = df.astype(float)
    row_sums = df.sum(axis=1).replace(0, np.nan)
    return df.div(row_sums, axis=0).fillna(0)


def _ns_hp_columns(ns, include_v450=False, include_v451=False):
    """ver4-48: the three hp_record provenance columns of a neural-stack task (ver4-50: plus two
    more when ``include_v450``).

    Why: the evaluation must be able to tell which lambda the network used (`'inf'` = the closed-form
    untuned naive ensemble, a string because the CSV round trip of a float infinity is not stable
    across readers), how many library columns were deployed, and how concentrated the gates were.
    `ns` is `details['ns']` of `tune_joint_hyperparameters_and_weights()` (None when absent, e.g. a
    task that fell back to the monolithic search - then the cells are empty, never missing, so the
    schema of a neural_stack run is the same on every task). Returns
    `{'joint_ns_lambda', 'joint_ns_members', 'joint_ns_entropy'}`.

    ver4-50 (R2, R3): with ``include_v450`` the dict also holds `joint_ns_objective` (the training
    objective, `'ensemble_score'` | `'pearson'`) and `joint_ns_swaps` (the paired swap test's
    verdicts, ``"<model>:<entry>:<admitted|rejected>"`` joined by ``;`` like `joint_fallbacks`;
    empty when no model's gate left entry 0 or the rule is `'gate'`). The caller sets the flag only
    when ns_objective / ns_swap_rule is away from its legacy value, so a default run keeps the exact
    ver4-49 hyperparameter.csv schema (P3, I11).

    ver4-51 (R2): with ``include_v451`` the dict also holds `joint_ns_outer_gain`,
    `joint_ns_outer_se` (the nested outer gate's out-of-sample gain over equal weights and its
    Nadeau-Bengio standard error; empty when the gate did not run) and `joint_ns_outer_decision`
    (`admitted` | `rejected` | `too_small` | `error` | `not_needed`, the last when lambda = inf was
    already chosen). The caller sets the flag only when ``ns_outer_gate != 'off'`` (P3, I11)."""
    ns = ns or {}
    lam = ns.get('lambda')
    if lam is None:
        lam_out = None
    else:
        try:
            lam_f = float(lam)
            lam_out = 'inf' if np.isinf(lam_f) else format(lam_f, 'g')
        except (TypeError, ValueError):
            lam_out = str(lam)
    out = {'joint_ns_lambda': lam_out,
           'joint_ns_members': ns.get('members'),
           'joint_ns_entropy': ns.get('entropy')}
    if include_v450:
        swaps = ns.get('swaps') or []
        out['joint_ns_objective'] = ns.get('ns_objective')
        out['joint_ns_swaps'] = ';'.join(
            f"{r.get('model')}:{r.get('entry')}:{'admitted' if r.get('admitted') else 'rejected'}"
            for r in swaps)
    if include_v451:
        outer = ns.get('outer') or {}

        def _num(v):
            try:
                v = float(v)
            except (TypeError, ValueError):
                return None
            return v if np.isfinite(v) else None

        out['joint_ns_outer_gain'] = _num(outer.get('gain'))
        out['joint_ns_outer_se'] = _num(outer.get('se'))
        out['joint_ns_outer_decision'] = outer.get('decision')
    return out


def _asreml_broadcast_scenario_cols(_df, _cols):
    """ver4-19 BUGFIX ("Length of values (3) does not match length of
    index (1500)" - production failure, task 1/25 of
    EasiGP_BLUEs_BLUPs_ASReml_0.5): broadcast a dict of scenario-
    identifier values (population/phenotype/ratio/sample) onto every row
    of `_df`. `ratio` (architecture doc §6 - "either a float ... or a
    3-tuple (train, valid, test)") can be a TUPLE, and pandas' own
    `df[col] = scalar_value` broadcast only works for a genuine scalar -
    handed a 3-element tuple, pandas instead tries to assign it as THREE
    per-row values, which fails the instant `_df`'s own row count isn't
    exactly 3 (i.e. on every real task). This is exactly the same hazard
    `sample_interaction['ratio']`/`sample_attention['ratio']` already
    guard against elsewhere in `_process_one_task()` (via `str(sample.
    loc[i, 'ratio'])`) - that precedent simply never reached the newer
    ASReml accumulation code (Step 6b) this helper now serves. Fixed
    generically, for every column (not only 'ratio'), by constructing an
    explicit length-N list of the SAME value first: a list of N copies of
    one tuple is still assigned as N single-cell (object-dtype) values,
    never unpacked - unlike the interaction/attention precedent, this
    preserves the value's own original type (float or tuple) rather than
    stringifying it, matching how every OTHER ASReml-free output
    (`stats`, `hp_record`, `effect`) already records it. Returns `_df`
    (mutated in place, exactly like the plain assignment loop it
    replaces) for the caller's own convenience.
    """
    for _col, _val in _cols.items():
        _df[_col] = [_val] * len(_df)
    return _df


# =============================================================================
# ver4-61 helpers: tuning-cache keys / entries (R6, fixes D4), the trial-memory
# guard arithmetic (R2) and the minutes -> seconds conversion of the per-model
# time limits (R3). Module-level and pure so they can be tested without a run.
# =============================================================================

def _tuning_key(population, phenotype, ratio, model, algorithm):
    """The one key under which a tuned-parameter entry is stored and looked up (ver4-61, R6).

    Why: the in-run cache (`HP_TUNE_SCOPE='per_scenario'`), its rebuild from a resumed
    ``hyperparameter.csv`` and the new ``HP_TUNE_PRESET_FILE`` must agree on the key, but
    the pieces arrive with different types - ``ratio`` is a float or a tuple in the sample
    table and a text such as ``'0.8'`` / ``'(0.7, 0.15, 0.15)'`` after a CSV round-trip
    (or a list after a JSON round-trip), ``population`` an int in one place and a string
    in the other. Without one normalisation a lookup silently never matches (blueprint D4).

    Normalisation: ``str()`` for population, phenotype and algorithm; the ratio is parsed
    back from text, lists become tuples (`restore_ratio`), numbers become ``float`` and the
    result is ``repr``-ed; the model goes through `base_of`. Returns a 5-tuple of ``str``."""
    ratio_value = ratio
    if isinstance(ratio_value, str):
        try:
            import ast
            ratio_value = ast.literal_eval(ratio_value.strip())
        except (ValueError, SyntaxError):
            pass

    def _plain(value):
        if isinstance(value, (list, tuple)):
            return tuple(_plain(v) for v in value)
        if isinstance(value, (bool, str)):
            return value
        try:
            return float(value)
        except (TypeError, ValueError):
            return value

    ratio_value = restore_ratio([_plain(ratio_value)], None)[0]
    return (str(population), str(phenotype), repr(ratio_value), base_of(str(model)),
            str(algorithm))


def _tuned_entry_from_row(base_model_name, phenotype, row, hparams_baseline):
    """Rebuild one tuned-parameter entry from a ``hyperparameter.csv`` row (ver4-61, R6).

    The single builder shared by the resume rebuild (`_rebuild_tuning_cache_from_hp_record`)
    and the preset reader, so both produce the exact dict `_resolve_tuned_params` reads:
    ``{'final_params': list, 'tuned_values': dict, 'best_valid_score': float, 'sample': ...}``.
    ``final_params`` starts from `hparams_baseline` (the frozen configured values, the same
    anchor a fresh search starts from) with only the TUNABLE fields overwritten from the
    row's value columns (matched by HPARAM_SPECS label); ``tuned_values`` holds those same
    ``{label: value}`` pairs. ``hyperparameter.csv`` never stores a model's full positional
    list, so fields that were not searched keep their configured value.

    Returns ``None`` (the caller skips the row, with a count) when the model has no schema,
    no configured baseline, or the row carries no tunable value at all."""
    fields = HPARAM_SPECS.get(schema_key_of(base_model_name))
    if not fields:
        return None
    baseline_entry = hparams_baseline.get(base_model_name)
    base_params = (baseline_entry.get(phenotype) if isinstance(baseline_entry, dict)
                   else baseline_entry)
    if base_params is None:
        return None
    final_params = list(base_params)
    tuned_values = {}
    for idx, field in enumerate(fields):
        if not field.get('tunable'):
            continue
        label = field['label']
        if label not in row.index or pd.isna(row[label]):
            continue
        value = row[label]
        ftype = field['type']
        try:
            if ftype == 'int':
                value = int(round(float(value)))
            elif ftype == 'float':
                value = float(value)
            elif ftype == 'bool' and isinstance(value, str):
                value = value.strip().lower() in ('true', '1', 'yes')
        except (TypeError, ValueError):
            pass  # keep the raw value rather than lose the row entirely
        final_params[idx] = value
        tuned_values[label] = value
    if not tuned_values:
        return None
    score = row.get('valid_objective_score', row.get('valid_pearson_over_mse'))
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = float('nan')
    sample_id = row.get('sample')
    try:
        sample_id = int(sample_id) if float(sample_id) == int(float(sample_id)) else sample_id
    except (TypeError, ValueError):
        pass
    return {'final_params': final_params, 'tuned_values': tuned_values,
            'best_valid_score': score, 'sample': sample_id}


def _read_hp_tune_preset(path, hparams_baseline):
    """Read a pilot run's ``hyperparameter.csv[.gz]`` into tuned-parameter entries (ver4-61, R6).

    Only rows whose ``tuning_source`` is ``'searched'`` are used (a reused / preset row was
    itself derived from one); for every `_tuning_key` the row with the smallest ``sample``
    wins, so the choice never depends on row order. Rows of a model the schema does not
    know, or whose baseline is missing in THIS run's config, are skipped and counted.

    Args: `path` the file (``~`` is expanded); `hparams_baseline` this run's frozen
    HPARAMETERS_BASELINE. Returns ``(entries, sha256_hex, skipped)`` where `entries` maps
    `_tuning_key` -> entry and `skipped` is the number of unusable searched rows.
    Raises RuntimeError - before any task runs, because the user asked for reuse - when the
    file cannot be read or parsed, or lacks the columns the key needs."""
    expanded = os.path.expanduser(str(path))
    try:
        with open(expanded, 'rb') as handle:
            raw = handle.read()
    except OSError as exc:
        raise RuntimeError(
            f"[GP] HP_TUNE_PRESET_FILE={path!r} cannot be read ({type(exc).__name__}: {exc}) - "
            f"the run is stopped before task 1. Point it at a pilot run's "
            f"hyperparameter.csv or hyperparameter.csv.gz, or clear the setting.") from exc
    digest = hashlib.sha256(raw).hexdigest()
    try:
        frame = pd.read_csv(io.BytesIO(raw),
                            compression='gzip' if raw[:2] == b'\x1f\x8b' else None)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, OSError,
            ValueError) as exc:
        raise RuntimeError(
            f"[GP] HP_TUNE_PRESET_FILE={path!r} is not a readable CSV "
            f"({type(exc).__name__}: {exc}) - the run is stopped before task 1.") from exc
    needed = ['population', 'phenotype', 'ratio', 'sample', 'model', 'algorithm',
              'tuning_source']
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise RuntimeError(
            f"[GP] HP_TUNE_PRESET_FILE={path!r} lacks column(s) {missing} (found "
            f"{list(frame.columns)[:12]}) - it must be a hyperparameter.csv written by an "
            f"EasiGP run with HP_TUNE. The run is stopped before task 1.")
    frame = frame[frame['tuning_source'].astype(str) == 'searched'].copy()
    frame['_sample_num'] = pd.to_numeric(frame['sample'], errors='coerce')
    frame = frame.sort_values('_sample_num', kind='mergesort')
    entries, skipped = {}, 0
    for _, row in frame.iterrows():
        base_name = base_of(str(row['model']))
        key = _tuning_key(row['population'], row['phenotype'], row['ratio'], base_name,
                          row['algorithm'])
        if key in entries:
            continue
        entry = _tuned_entry_from_row(base_name, str(row['phenotype']), row, hparams_baseline)
        if entry is None:
            skipped += 1
            continue
        entries[key] = entry
    return entries, digest, skipped


def trial_memory_concurrency(peak_bytes, available_bytes, n_model_workers=1, fraction=0.7):
    """How many bio-prior trials fit in memory at once (ver4-61, R2): ``floor(fraction *
    available / peak)``, divided by the number of model workers that share the node.

    Why 0.7: the estimate is a model, not a measurement, and the rest of the run (parent
    process, other models, page cache) needs room; the blueprint fixes it at 70 %.
    Returns an int >= 0 (0 means not even one trial is predicted to fit).
    Raises ValueError for a non-positive `peak_bytes`."""
    if not peak_bytes or float(peak_bytes) <= 0:
        raise ValueError(f"peak_bytes must be positive, got {peak_bytes!r}")
    share = max(1, int(n_model_workers or 1))
    return int(math.floor(float(fraction) * float(available_bytes) / float(peak_bytes) / share))


def _limit_minutes_to_seconds(value, key, model_name):
    """``HP_TUNE[model][key]`` (minutes, float >= 0) as seconds, or None for absent / 0 / invalid
    (ver4-61, R3). An invalid value (negative, non-numeric, non-finite) prints a WARNING and
    is ignored - a typo must not abort a long run, and 'off' is the safe reading."""
    if value is None or value == '':
        return None
    try:
        minutes = float(value)
    except (TypeError, ValueError):
        minutes = float('nan')
    if not math.isfinite(minutes) or minutes < 0:
        print(f"[GP] WARNING: HP_TUNE[{model_name!r}][{key!r}]={value!r} is not a number >= 0 "
              f"(minutes) - the limit is ignored (treated as off).")
        return None
    return minutes * 60.0 if minutes > 0 else None


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
                                    result_name, k_precomputed, row_offset, row_count,
                                    weights=None):
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
    ensure_rpy2_thread_safe_release()  # ver4-59: rpy2 releases R objects thread-safely

    r_source = robjects.r['source']
    r_filename = 'GBLUP.R' if model_name == 'GBLUP' else 'RKHS.R'
    r_source(os.path.join(_R_MODELS_DIR, r_filename))
    r_fn = wide_safe_r_function(robjects.globalenv[model_name])  # ver4-59: chunked transfer of very wide marker frames

    offset_idx, count_idx = (7, 8) if model_name == 'GBLUP' else (8, 9)
    worker_params = list(params)
    worker_params[offset_idx] = int(row_offset)
    worker_params[count_idx] = int(row_count)

    r_kwargs = {}
    if k_precomputed is not None:
        r_kwargs['K_precomputed'] = k_precomputed
    # ver4-25, R8 (design blueprint SS2, Stage 6): the SAME shared
    # train-split fitting weight every non-fanned-out call would have
    # received - see GBLUP.R's/RKHS.R's own updated `weights=` parameter.
    if weights is not None:
        r_kwargs['weights'] = weights

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
    ensure_rpy2_thread_safe_release()  # ver4-59: rpy2 releases R objects thread-safely
    r_source = robjects.r['source']
    r_source(os.path.join(_R_MODELS_DIR, f'{model_name}.R'))
    fn = wide_safe_r_function(robjects.globalenv[model_name])  # ver4-59: chunked transfer of very wide marker frames
    _R_FN_CACHE[model_name] = fn
    return fn


def _run_gblup_or_rkhs_for_trial(model_name, r_path, compute_resources, train, valid, test,
                                  params, result_name, k_precomputed=None, weights=None):
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
    # ver4-25, R8 - see _gblup_rkhs_shapley_row_worker()'s identical comment.
    if weights is not None:
        r_kwargs['weights'] = weights

    r_fn = _get_r_model_fn(model_name, r_path)
    get_effect = bool(params[get_effect_idx])
    # ver4-25, Stage 7 (R13): the legacy row-range Shapley fan-out below
    # only makes sense for the 'iml_shapley' effect method (the ONLY one
    # that still runs a per-row iml::Shapley perturbation-refit loop
    # inside GBLUP.R/RKHS.R) - 'closed_form'/'shapiq_predictor' need
    # exactly ONE call (this fit's own r_u/r_mu/r_d_scale, computed once
    # regardless of Shapley_num), so they always take the single, unsplit
    # call below, same as get_effect=False. Their own (cheaper, Python-
    # side) explanation step parallelises separately, via
    # parallel_shap_values()'s own n_jobs - see
    # _apply_rkhs_shapiq_predictor_effect()'s docstring.
    _effect_method_idx = 13 if model_name == 'GBLUP' else 17
    _effect_method = params[_effect_method_idx] if len(params) > _effect_method_idx else 'iml_shapley'
    _raw_n_jobs = nested_safe_n_jobs(compute_resources, (compute_resources or {}).get('n_jobs'))
    if not get_effect or not _raw_n_jobs or _raw_n_jobs == 1 or _effect_method != 'iml_shapley':
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
                k_precomputed, int(bound[0]), int(len(bound)), weights=weights,
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


def _final_gat_fit(model_fn, *args, **kwargs):
    """Call a GAT model function as a FINAL fit (the untuned main-loop fit): the seed ensemble
    (ACCELERATION 'GAT_SEED_ENSEMBLE', models/gat_stability.py F2) applies; with the default
    of 1 member this is the plain call."""
    from models.gat_stability import final_fit_role
    with final_fit_role():
        return model_fn(*args, **kwargs)


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
                 bio_prior_ctx=None, sample_weight_ctx=None):
        self.result_name = result_name
        self.r_path = r_path
        self.compute_resources = dict(compute_resources) if compute_resources else {}
        self.genotype_format = genotype_format
        self.bio_prior_ctx = bio_prior_ctx  # plain dict of DataFrames/paths/dict, or None
        # ver4-25, R8 (design blueprint SS2, Stage 6): the task's shared
        # BlueWeightContext (blue_weights.BlueWeightContext), or None
        # when BLUE-reliability weighting is off - a plain, picklable
        # dataclass of ndarrays (invariant I13), threaded into every
        # model branch below exactly like bio_prior_ctx.
        self.sample_weight_ctx = sample_weight_ctx

    def with_cv_weights(self):
        """ver4-52: a shallow copy of this runner whose BLUE context is the CROSS-VALIDATION view
        (`blue_weights.cv_weight_view()`: `w_fit` = the train+valid fitting weights). The joint
        search's CV objective fits fold models on frames cut from `joint_tuning._cv_frame()`
        (train rows first, then validation rows, labels `0..n_tv-1`); `__call__` then aligns this
        vector to each fold's training rows by label. Used per fit unit (`_FitUnit.cv_weights`)
        so the confirmatory and every other train-only fit keep the task's own `w_fit`.
        Returns `self` when weighting is off. Raises ValueError (via `cv_weight_view`) when the
        context was built without `with_cv_weights=True`."""
        if self.sample_weight_ctx is None:
            return self
        clone = copy.copy(self)
        clone.sample_weight_ctx = blue_weights.cv_weight_view(self.sample_weight_ctx)
        return clone

    def __call__(self, model_name, train, valid, test, params, result_name):
        params = list(params)
        _scoring_valid = valid  # overridden below only by the bio_prior branch
        # ver4-25, R8: the shared TRAIN-split fitting weight, or None
        # (byte-identical to before this feature existed - I11).
        _w_fit = self.sample_weight_ctx.w_fit if self.sample_weight_ctx is not None else None
        # ver4-39: `train` may be a ROW SUBSET of the task's train split (HP-tuning inner
        # resamples, multi-fidelity screen, joint-tuning folds) - align w_fit to it.
        # Unchanged object when len(train) == len(w_fit) (main dispatch, I11).
        _w_fit = blue_weights.align_fit_weights(_w_fit, train)
        # ver4-52: the sklearn / torch model functions (RF, ExtraTrees, GBDT, XGBoost, EBM, SVR,
        # KNN, MLP, every GAT) read `sample_weight_ctx.w_fit` themselves and do NOT align it, so a
        # row-subset fit (a CV fold, an inner resample) handed them the task-length vector and
        # failed on the length mismatch. They now receive a context whose `w_fit` is the aligned
        # vector `_w_fit` computed above (the same object, hence byte-identical, whenever
        # `train` is the task's own train split - I11).
        _swc = self.sample_weight_ctx
        if _swc is not None and _w_fit is not _swc.w_fit:
            _swc = dataclasses.replace(_swc, w_fit=_w_fit)

        if model_name == 'rrBLUP':
            r_fn = _get_r_model_fn('rrBLUP', self.r_path)
            result = r_fn(train, valid, test, params, result_name,
                          **({'weights': _w_fit} if _w_fit is not None else {}))
            _cleanup_bglr_output_for(result_name)
            out = dict(pearson_test=r_list_get(result, 'r_pearson')[0], mse_test=r_list_get(result, 'r_MSE')[0],
                       effect=r_list_get(result, 'r_effect'), predicted_test=r_list_get(result, 'r_y_predicted'),
                       predicted_valid=r_list_get(result, 'r_y_predicted_valid'), predicted_train=r_list_get(result, 'r_y_predicted_train'))
            # Requirements3.md item 1 (ver4-25, Stage 7 sibling patch):
            # 'closed_form' converts the SAME beta above into an exact
            # linear-Shapley effect; 'beta_direct' (default) is a no-op
            # passthrough - see _apply_linear_closed_form_effect()'s own
            # docstring (I3 lockstep with the main dispatch's equivalent
            # branch).
            out['effect'] = _apply_linear_closed_form_effect('rrBLUP', params, out['effect'], train, test)
            # Update ID ver4-5, R2 Stage 10 (I3 lockstep with the main
            # dispatch's equivalent branch, and with _r_model_surrogate_
            # interaction()'s own "one implementation, two call sites"
            # note): a Python-only, post-R-call surrogate interaction step.
            out['interaction'] = _r_model_surrogate_interaction(
                'rrBLUP', train, out['predicted_train'], params, self.compute_resources.get('n_jobs'),
                test=test, row_parallel=self.compute_resources.get('shapiq_row_parallel', True),
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
                                                   k_precomputed=_gblup_k_precomputed, weights=_w_fit)
            _cleanup_bglr_output_for(result_name)
            out = dict(pearson_test=r_list_get(result, 'r_pearson')[0], mse_test=r_list_get(result, 'r_MSE')[0],
                       effect=r_list_get(result, 'r_effect'), predicted_test=r_list_get(result, 'r_y_predicted'),
                       predicted_valid=r_list_get(result, 'r_y_predicted_valid'), predicted_train=r_list_get(result, 'r_y_predicted_train'))
            # ver4-25, Stage 7 (R13): 'closed_form' builds the exact
            # VanRaden-identity effect from this fit's own r_u/r_mu;
            # 'iml_shapley' (default) is a no-op passthrough - see
            # _apply_gblup_closed_form_effect()'s own docstring.
            out['effect'] = _apply_gblup_closed_form_effect(result, params, train, test)
            out['interaction'] = _r_model_surrogate_interaction(
                'GBLUP', train, out['predicted_train'], params, self.compute_resources.get('n_jobs'),
                test=test, row_parallel=self.compute_resources.get('shapiq_row_parallel', True),
            )

        elif model_name == 'BayesB':
            r_fn = _get_r_model_fn('BayesB', self.r_path)
            result = r_fn(train, valid, test, params, result_name,
                          **({'weights': _w_fit} if _w_fit is not None else {}))
            _cleanup_bglr_output_for(result_name)
            out = dict(pearson_test=r_list_get(result, 'r_pearson')[0], mse_test=r_list_get(result, 'r_MSE')[0],
                       effect=r_list_get(result, 'r_effect'), predicted_test=r_list_get(result, 'r_y_predicted'),
                       predicted_valid=r_list_get(result, 'r_y_predicted_valid'), predicted_train=r_list_get(result, 'r_y_predicted_train'))
            # Requirements3.md item 1 - see the rrBLUP branch above.
            out['effect'] = _apply_linear_closed_form_effect('BayesB', params, out['effect'], train, test)
            out['interaction'] = _r_model_surrogate_interaction(
                'BayesB', train, out['predicted_train'], params, self.compute_resources.get('n_jobs'),
                test=test, row_parallel=self.compute_resources.get('shapiq_row_parallel', True),
            )

        elif model_name == 'RKHS':
            if params[4] == 'all':
                params[4] = test.shape[0]
            result = _run_gblup_or_rkhs_for_trial('RKHS', self.r_path, self.compute_resources,
                                                   train, valid, test, params, result_name,
                                                   weights=_w_fit)
            _cleanup_bglr_output_for(result_name)
            out = dict(pearson_test=r_list_get(result, 'r_pearson')[0], mse_test=r_list_get(result, 'r_MSE')[0],
                       effect=r_list_get(result, 'r_effect'), predicted_test=r_list_get(result, 'r_y_predicted'),
                       predicted_valid=r_list_get(result, 'r_y_predicted_valid'), predicted_train=r_list_get(result, 'r_y_predicted_train'))
            # ver4-25, Stage 7 (R13): 'shapiq_predictor' explains a
            # pure-NumPy reconstruction of this fit's own kernel
            # regression with shap.KernelExplainer (V13.2-gated);
            # 'iml_shapley' (default) is a no-op passthrough - see
            # _apply_rkhs_shapiq_predictor_effect()'s own docstring.
            # ver4-43 (E2/E3): one memo per RKHS model call, so the effect and
            # interaction routes share ONE gated reconstruction; the two
            # explanation stages are timed on their own. Lockstep (RK-2) with
            # the main dispatch loop's equivalent branch below.
            _rkhs_memo = {}
            with timed_stage('xai_effect:RKHS'):
                out['effect'] = _apply_rkhs_shapiq_predictor_effect(
                    _get_r_model_fn('RKHS', self.r_path), result, params, train, valid, test, result_name,
                    out['predicted_test'], ({'weights': _w_fit} if _w_fit is not None else {}),
                    n_jobs=self.compute_resources.get('n_jobs'),
                    predictor_memo=_rkhs_memo,
                )
            # This update: 'shapiq_predictor' (params[18]) reuses the SAME
            # V13.2-gated RKHSPredictor reconstruction the effect route
            # above just used, explaining it directly with shapiq instead
            # of falling back to an RF/H-statistic surrogate of this
            # model's own predictions - see
            # _apply_rkhs_predictor_interaction()'s own docstring. Every
            # OTHER 'Marker interaction method' choice is unaffected
            # (byte-identical passthrough to _r_model_surrogate_
            # interaction(), I11).
            with timed_stage('xai_inter:RKHS'):
                out['interaction'] = _apply_rkhs_predictor_interaction(
                    result, params, train, test, out['predicted_train'], out['predicted_test'],
                    valid=valid,
                    n_jobs=self.compute_resources.get('n_jobs'),
                    use_gpu=self.compute_resources.get('use_gpu_sklearn', False),
                    row_parallel=self.compute_resources.get('shapiq_row_parallel', True),
                    # This update - see out['effect'] above (already computed,
                    # I3 lockstep with the main dispatch loop's equivalent
                    # branch below) and _apply_rkhs_predictor_interaction()'s
                    # own 'effect_values' parameter docstring.
                    effect_values=out['effect'],
                    # Bug fix (V13.2 gate false-positive under BLUE-reliability
                    # weighting) - see _apply_rkhs_predictor_interaction()'s/
                    # _build_gated_rkhs_predictor()'s own 'weights' parameter
                    # docstring. Lockstep (RK-2) with the main dispatch loop's
                    # equivalent branch below.
                    weights=_w_fit,
                    predictor_memo=_rkhs_memo,
                )

        elif model_name == 'RF':
            if params[6] == 'all':
                params[6] = test.shape[0]
            r, mse, effect, interaction, pt, pv, ptr = RF(train, valid, test, params, sample_weight_ctx=_swc)
            out = dict(pearson_test=r, mse_test=mse, effect=effect, interaction=interaction,
                       predicted_test=pt, predicted_valid=pv, predicted_train=ptr)

        elif model_name == 'ExtraTrees':
            if params[6] == 'all':
                params[6] = test.shape[0]
            r, mse, effect, interaction, pt, pv, ptr = ExtraTrees(train, valid, test, params, sample_weight_ctx=_swc)
            out = dict(pearson_test=r, mse_test=mse, effect=effect, interaction=interaction,
                       predicted_test=pt, predicted_valid=pv, predicted_train=ptr)

        elif model_name == 'GBDT':
            if params[7] == 'all':
                params[7] = test.shape[0]
            r, mse, effect, interaction, pt, pv, ptr = GBDT(train, valid, test, params, sample_weight_ctx=_swc)
            out = dict(pearson_test=r, mse_test=mse, effect=effect, interaction=interaction,
                       predicted_test=pt, predicted_valid=pv, predicted_train=ptr)

        elif model_name == 'XGBoost':
            if params[7] == 'all':
                params[7] = test.shape[0]
            r, mse, effect, interaction, pt, pv, ptr = XGBoost(train, valid, test, params, sample_weight_ctx=_swc)
            out = dict(pearson_test=r, mse_test=mse, effect=effect, interaction=interaction,
                       predicted_test=pt, predicted_valid=pv, predicted_train=ptr)

        elif model_name == 'EBM':
            r, mse, effect, interaction, pt, pv, ptr = EBM(train, valid, test, params, sample_weight_ctx=_swc)
            out = dict(pearson_test=r, mse_test=mse, effect=effect, interaction=interaction,
                       predicted_test=pt, predicted_valid=pv, predicted_train=ptr)

        elif model_name == 'SVR':
            if params[7] == 'all':
                params[7] = test.shape[0]
            r, mse, effect, interaction, pt, pv, ptr = SV_Regression(train, valid, test, params, sample_weight_ctx=_swc)
            out = dict(pearson_test=r, mse_test=mse, effect=effect, interaction=interaction,
                       predicted_test=pt, predicted_valid=pv, predicted_train=ptr)

        elif model_name == 'KNN':
            if params[4] == 'all':
                params[4] = test.shape[0]
            r, mse, effect, interaction, pt, pv, ptr = KNN(train, valid, test, params, sample_weight_ctx=_swc)
            out = dict(pearson_test=r, mse_test=mse, effect=effect, interaction=interaction,
                       predicted_test=pt, predicted_valid=pv, predicted_train=ptr)

        elif model_name == 'MLP':
            if params[7] == 'all':
                params[7] = test.shape[0]
            r, mse, effect, interaction, pt, pv, ptr = ML_Perceptron(train, valid, test, params, sample_weight_ctx=_swc)
            out = dict(pearson_test=r, mse_test=mse, effect=effect, interaction=interaction,
                       predicted_test=pt, predicted_valid=pv, predicted_train=ptr)

        elif model_name == 'GAT_infinitesimal_node_level':
            if params[-2] == 'all':
                params[-2] = test.shape[0]
            r, mse, effect, pt, pv, ptr = GAT_infinitesimal_node_level(train, valid, test, params, sample_weight_ctx=_swc)
            out = dict(pearson_test=r, mse_test=mse, effect=effect,
                       predicted_test=pt, predicted_valid=pv, predicted_train=ptr)

        elif model_name == 'GAT_infinitesimal':
            if params[-1] == 'all':
                params[-1] = test.shape[0]
            r, mse, effect, pt, pv, ptr = GAT_infinitesimal(train, valid, test, params, sample_weight_ctx=_swc)
            out = dict(pearson_test=r, mse_test=mse, effect=effect,
                       predicted_test=pt, predicted_valid=pv, predicted_train=ptr)

        elif model_name == 'GAT_fully_connected':
            if params[-1] == 'all':
                params[-1] = test.shape[0]
            r, mse, effect, pt, pv, ptr, attn = GAT_fully_connected(train, valid, test, params, sample_weight_ctx=_swc)
            out = dict(pearson_test=r, mse_test=mse, effect=effect, attention=attn,
                       predicted_test=pt, predicted_valid=pv, predicted_train=ptr)

        elif model_name == 'GAT_prior_knowledge':
            if params[-2] == 'all':
                params[-2] = test.shape[0]
            # Update ID ver4-5, R2 (I3 lockstep with the main dispatch
            # branch, and with _call_model() above it): GAT_prior_knowledge
            # returns an 8th value (its own already-computed interaction
            # pairs, empty unless emit_interaction is set).
            r, mse, effect, pt, pv, ptr, attn, interaction = GAT_prior_knowledge(train, valid, test, params, sample_weight_ctx=_swc)
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

            if len(bio_prior_params) > _BIO_PRIOR_SAMPLES_IDX and \
                    bio_prior_params[_BIO_PRIOR_SAMPLES_IDX] == 'all':
                bio_prior_params[_BIO_PRIOR_SAMPLES_IDX] = bio_test.shape[0]
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
                sample_weight_ctx=_swc,
            )
            out = dict(pearson_test=r, mse_test=mse, effect=effect, attention=attn,
                       predicted_test=pt, predicted_valid=pv, predicted_train=ptr)
            _scoring_valid = bio_valid

        else:
            raise ValueError(f"ModelTrialRunner: unrecognised model {model_name!r}")

        if _scoring_valid.shape[0] != 0 and len(out.get('predicted_valid', [])) > 0:
            actual_valid = _scoring_valid.iloc[:, -1].values.tolist()
            # ver4-25, R8: weighted valid scoring when BLUE weighting is
            # on - this is what makes hyperparameter tuning's own
            # pearson_valid/mse_valid objective (models/hyperparameter_
            # tuning.py reads exactly these two keys from whatever
            # run_model_fn returns) weighted for free, with NO change
            # needed to that module itself. `{}` (mode='none' or an
            # empty split) reproduces the exact unweighted call below.
            # (blue_weights is imported at module level; a local import here would make the name
            # function-local for the WHOLE method and break the align_fit_weights() call above.)
            out['pearson_valid'], out['mse_valid'] = safe_regression_metrics(
                actual_valid, out['predicted_valid'],
                **blue_weights.scoring_for(self.sample_weight_ctx, 'valid'))
        else:
            out['pearson_valid'] = None
            out['mse_valid'] = None

        # ver4-38, R3: one deduplicated WARN per (run, model) per process
        # when this fit's test prediction is constant (all tuning, joint
        # search and refit fits pass through here; the production rows are
        # checked at the matching point in GP()'s main loop - same helper).
        # ver4-46, R1: a lean GAT search fit returns no test predictions on purpose (the
        # search never reads them) - an empty column must not raise the "constant
        # prediction" warning.
        if not is_lean_search_fit():
            warn_if_constant_prediction(model_name, out.get('predicted_test', []),
                                        task_tag=str(self.result_name))

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
    # ver4-35, R1.e1: the libraries above were imported at full node size
    # before the divided OMP/BLAS variables were exported, so cap the native
    # pools at this worker's own budget. A no-op unless
    # ACCELERATION['worker_thread_limits'] is on (legacy: nothing changes).
    apply_worker_thread_limits(resources_dict or {})


# PERFORMANCE FIX ("Bayesian hyperparameter tuning is prohibitive") - see the
# single-model HP_TUNE dispatch's own call site below for the full diagnosis
# this helper's dict-division logic exists to fix (thread-contention
# oversubscription when several trial workers each get this run's own
# UNDIVIDED per-process compute budget at once). Extracted here (ver4-28, R1)
# purely so "Step 8.5" (the joint search, `models/joint_tuning.py`) can divide
# ITS OWN trial-worker budget across up to `parallel_trials` fit-unit workers
# the exact same way, via `nested_safe_n_jobs()`, without a second copy of
# this division logic. Behaviour-identical to the pre-ver4-28 inline block at
# every existing call site (I11) - only the log line stays at each CALLER,
# unchanged, since only the caller knows whether to phrase it in terms of a
# model+algorithm (single-model tuning) or a joint task (Step 8.5).
def _trial_worker_resources_for(compute_resources, width):
    """Divide `compute_resources` (n_jobs/r_blas_threads/torch_num_threads)
    so that `width` trial workers running at once sum back to roughly this
    run's own resolved per-process budget, via `nested_safe_n_jobs()` (the
    same "divide, never multiply" helper every other nested-parallelism case
    in this codebase already uses). `width<=1` returns `compute_resources`
    UNCHANGED (not a copy) - no worker is ever spawned to receive a divided
    budget in that case, so the caller's own single, serial fit stays sized
    identically to every other, non-tuned model call in this run."""
    if width > 1:
        divided = dict(compute_resources)
        divided['n_jobs'] = nested_safe_n_jobs(compute_resources, compute_resources.get('n_jobs'), width)
        if compute_resources.get('r_blas_threads'):
            divided['r_blas_threads'] = nested_safe_n_jobs(
                compute_resources, compute_resources.get('r_blas_threads'), width)
        if compute_resources.get('torch_num_threads'):
            divided['torch_num_threads'] = nested_safe_n_jobs(
                compute_resources, compute_resources.get('torch_num_threads'), width)
        return divided
    return compute_resources


# ver4-35, R1.f: reasons the asynchronous model lane was declined, printed once
# per process (a long batch would otherwise repeat them every task).
_LANE_OFF_NOTED = set()


def _note_lane_off_once(reason):
    """Print "the lane is off because ..." the first time `reason` occurs."""
    if reason not in _LANE_OFF_NOTED:
        _LANE_OFF_NOTED.add(reason)
        print(f"[lane] NOTE: the asynchronous model lane is not used: {reason}. Models run one "
              f"after another exactly as before (ACCELERATION['async_model_lane'] stays on for "
              f"tasks where it applies).")


def _lane_worker_resources(compute_resources):
    """The resources dict the lane worker restores (``_trial_worker_init``):
    this run's own, with ``torch_num_threads`` set to
    ``ASYNC_LANE_THREADS`` when that is an integer ('match' keeps the
    parent's value, so a lane fit trains with the thread count an inline fit
    would have used)."""
    resources = dict(compute_resources)
    threads = resources.get('accel_async_lane_threads', 'match')
    if threads != 'match':
        resources['torch_num_threads'] = int(threads)
    return resources


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
                                    test=None, row_parallel: bool = True, effect_values=None):
    """Fit a lightweight surrogate on `model_name`'s OWN predicted_train
    output (never the true phenotype - see models.interaction_
    extraction.surrogate_tree_interactions()'s/surrogate_h_statistic_
    interactions()'s own Raises note) and return its pairwise
    interactions, shortlisted and top-filtered per this model's own
    appended HPARAMETERS fields (see _r_model_interaction_fields()
    above).

    Dispatches to ONE of three interaction-extraction algorithms run on
    that SAME surrogate, by `model_name` (and, for RKHS, by its own
    "Marker interaction method" field):

      - `model_name == 'RKHS'`, field (params[18]) == 'friedman_h'
        (default, and what any config predating this field keeps using
        - the OLD spelling 'surrogate_friedman_h' is also still accepted,
        see this function's own inline comment at the dispatch site):
        Friedman's H-statistic (models.interaction_extraction.
        surrogate_h_statistic_interactions()) - RKHS's own route,
        singled out because its kernel already captures non-additive
        structure (see the module-level comment above this function for
        the full rationale). `use_gpu` (forwarded from this run's
        `USE_GPU_SKLEARN` compute-resource setting, exactly like
        RF/SVR/KNN's own model fits already do) lets the surrogate fit
        attempt a cuML GPU backend here - safe for THIS algorithm (see
        `surrogate_h_statistic_interactions()`'s own docstring for why),
        unlike the TreeSHAP branch below.
      - `model_name == 'RKHS'`, field == 'shapiq' (ver4-25, Stage 8,
        design blueprint R11; the OLD spelling 'surrogate_shapiq' is
        also still accepted - Requirements.md item 1 dropped the word
        "surrogate" from this field's own choices, since every route on
        it was already surrogate-based): the SAME surrogate explained
        with the optional `shapiq` package's own Shapley interaction
        indices instead (models.interaction_extraction.surrogate_
        shapiq_interactions()) - still surrogate-based, APPROXIMATE,
        not RKHS's own exact kernel (see that function's own docstring).
        Wrapped in its own try/except here, separate from this whole
        function's outer one, so a `shapiq`-specific failure (package
        missing, API mismatch) falls back to 'friedman_h' for that task
        with a printed NOTE, rather than losing the interaction step
        entirely.
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

    row_parallel : bool, keyword-only, default True
        Design blueprint R1: gates whether the RKHS 'shapiq' branch's
        underlying `models.interaction_extraction.
        surrogate_shapiq_interactions()` call fans its per-row `shapiq`
        explain loop out across `n_jobs` worker processes (forwarded as
        `n_jobs if row_parallel else 1`; `n_jobs`'s OWN use for that
        function's surrogate model FIT is unaffected either way - see
        `surrogate_shapiq_interactions()`'s own R1 docstring note for why
        the two share one parameter). `True` (the default) is byte-
        identical to this function's pre-R1 behaviour whenever `n_jobs
        <= 1` (the common single-core case, I11). Every OTHER interaction
        route this function dispatches to (`surrogate_h_statistic_
        interactions()`, `surrogate_tree_interactions()`) already had its
        own `n_jobs` handling before this update and is UNCHANGED by this
        parameter - only the shapiq route's NEW row-level parallelism is
        gated here, per design blueprint R1's own stated scope.

    effect_values : pandas.DataFrame or None, keyword-only, default None
        This update ("use marker effects to select markers for
        interaction search?" checkbox, RKHS only - see models/RF.py's
        identical checkbox for the full rationale). RKHS's own
        already-computed marker-EFFECT values (a 1-row DataFrame, one
        column per marker - the SAME shape `_apply_rkhs_shapiq_predictor_
        effect()` returns), forwarded by both RKHS call sites (this
        function is called AFTER the effect step at both, I3 lockstep).
        Only ever consulted when `model_name == 'RKHS'` - rrBLUP/BayesB/
        GBLUP share this function but have no marker-effect shortlist of
        their own, so this parameter is a silent no-op for them (I11).
        `None` (the default, and what every non-RKHS caller passes)
        reproduces this function's pre-existing correlation-based
        shortlist exactly.
    """
    (get_interaction, max_interaction_features, interaction_background, interaction_top,
     interaction_screen, interaction_screen_top, interaction_grid_resolution) = \
        _r_model_interaction_fields(model_name, params)
    if not get_interaction:
        return pd.DataFrame()

    try:
        train_x = train.iloc[:, :-1]
        train_y = train.iloc[:, -1]

        # This update - see this function's own 'effect_values' parameter
        # docstring above. `_rkhs_use_effect_shortlist` decides whether
        # RKHS's shortlist below is ranked by its own already-computed
        # marker-effect values instead of trait correlation; every other
        # model dispatched here (rrBLUP/BayesB/GBLUP) is untouched
        # (`model_name == 'RKHS'` gate).
        _rkhs_use_effect_shortlist = False
        if model_name == 'RKHS' and effect_values is not None:
            _rkhs_get_effect = bool(params[3]) if len(params) > 3 else False
            _rkhs_effect_cap = (params[5] if len(params) > 5 else 500) if _rkhs_get_effect else None
            _rkhs_use_effect_for_shortlist = params[24] if len(params) > 24 else True
            _rkhs_use_effect_shortlist = should_use_effect_for_interaction_shortlist(
                _rkhs_use_effect_for_shortlist, max_interaction_features, _rkhs_effect_cap,
            )
        if _rkhs_use_effect_shortlist:
            _interaction_ranking = (
                np.asarray(effect_values.iloc[0].reindex(train_x.columns).fillna(0)).ravel()
            )
        else:
            _interaction_ranking = train_x.corrwith(train_y).abs().fillna(0).to_numpy()

        shortlist_idx = shortlist_markers(
            np.abs(_interaction_ranking), max_interaction_features,
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

            # ver4-25, Stage 8 (design blueprint R11): RKHS's own "Marker
            # interaction method" field (hparam_specs.py's appended index
            # 18) - 'friedman_h' (default, unchanged behaviour for any
            # config predating this field) keeps using the H-statistic
            # route above; 'shapiq' routes through the SAME surrogate but
            # explains it with the optional shapiq package's own Shapley
            # interaction indices instead (see models.interaction_
            # extraction.surrogate_shapiq_interactions()'s own docstring
            # for why this is still surrogate-based, not RKHS's own exact
            # kernel). Read defensively (index 18/19/20/21 are all
            # appended-after fields - I5) so a config predating this
            # field simply keeps the H-statistic route, exactly like
            # every other newly-added toggle in this file.
            #
            # Requirements.md item 1: this field's own two choices used
            # to be spelled 'surrogate_friedman_h'/'surrogate_shapiq' -
            # renamed to plain 'friedman_h'/'shapiq' (the word "surrogate"
            # added nothing a user could act on; every route on this
            # field was ALREADY surrogate-based, see this function's own
            # docstring). Both the NEW spelling (from hparam_specs.py's
            # own choices, or main_app.py's own selectbox self-heal for
            # an older saved GUI state) and the OLD spelling (a raw
            # headless/JSON config or checkpoint written before this
            # rename, which bypasses the GUI widget - and so never goes
            # through that self-heal - entirely) are accepted here, so
            # neither a fresh run nor an old saved one silently changes
            # which route actually runs (I11).
            _interaction_method = params[18] if len(params) > 18 else 'friedman_h'
            if _interaction_method in ('shapiq', 'surrogate_shapiq'):
                try:
                    _orders_raw = params[19] if len(params) > 19 else '2'
                    _orders = tuple(
                        int(_o.strip()) for _o in str(_orders_raw).split(',') if _o.strip()
                    ) or (2,)
                    _shapiq_index = params[20] if len(params) > 20 else 'k-SII'
                    _shapiq_budget = int(params[21]) if len(params) > 21 else 4096
                    # Design blueprint R1: `row_parallel` gates ONLY the
                    # NEW row-level shapiq fan-out - the surrogate fit's
                    # OWN use of `n_jobs` (inside surrogate_shapiq_
                    # interactions() itself) is unaffected either way,
                    # since both share that function's single `n_jobs`
                    # parameter (see this function's own docstring above).
                    _shapiq_n_jobs = (
                        (n_jobs if n_jobs and n_jobs > 0 else 1) if row_parallel else 1
                    )
                    interaction_sample = surrogate_shapiq_interactions(
                        x_shortlist, y_pred, marker_names,
                        surrogate_cfg={
                            'n_background': min(int(interaction_background), _explain_n),
                        },
                        n_jobs=_shapiq_n_jobs,
                        use_gpu=bool(use_gpu),
                        explain_pool=explain_pool,
                        orders=_orders,
                        budget=_shapiq_budget,
                        index=_shapiq_index,
                    )
                    method_label = 'surrogate-shapiq'
                except Exception as _shapiq_exc:
                    print(f"[GP] NOTE: RKHS surrogate_shapiq_interactions() failed "
                          f"({_shapiq_exc!r}) - falling back to 'friedman_h' "
                          f"(Friedman's H-statistic) for this task instead.")
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
                    method_label = 'surrogate-H-statistic (shapiq fallback)'
            else:
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

        interaction_sample = top_select_by_order(interaction_sample, 'percentage', interaction_top)

        # Requirements3.md item 4 - see models/SVR.py's own identical
        # note: RKHS's own 'shapiq' route (surrogate_shapiq_interactions())
        # genuinely finds fewer nonzero pairs than a dense SHAP-based
        # route (surrogate_tree_interactions(), the rrBLUP/BayesB/GBLUP
        # route below, or any of RF/SVR/KNN's own TreeSHAP/'pairwise_shap'
        # output) at the same task scale, so its own Interaction.csv row
        # count needs the SAME parity padding RF.py's 'shapiq_tree' route
        # already applies - against the SAME candidate universe this
        # branch itself searched (marker_names) at the SAME threshold
        # (interaction_top). Gated on `_interaction_method` (the
        # CONFIGURED choice), exactly like models/RF.py's own identical
        # `if interaction_method == 'shapiq_tree':` gate - not on whether
        # the shapiq call itself happened to succeed this task, so a
        # shapiq failure that fell back to 'friedman_h' above is padded
        # to the same target too (harmless/no-op whenever that fallback
        # already reached it - see pad_shapiq_pairs_to_shap_parity()'s
        # own "no-op if already at target" return, identical to the
        # dense 'pairwise_shap' fallback case RF.py's own docstring
        # already covers).
        if model_name == 'RKHS' and _interaction_method in ('shapiq', 'surrogate_shapiq'):
            interaction_sample = pad_shapiq_pairs_to_shap_parity(
                interaction_sample, marker_names, interaction_top,
            )
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


def _split_interaction_by_order(interaction_df):
    """ver4-25, Stage 9 (design blueprint R12): split a model's own
    'interaction' return-slot DataFrame into its k=2 half (today's
    'marker1'/'marker2'/'value' shape, routed into Interaction.csv,
    UNCHANGED - I11) and its k>=3 half ('markers'/'order'/'value',
    routed into the new Interaction_higher_order.csv).

    Called ONCE per task, right where every per-model dispatch branch's
    own result converges into a single `sample_interaction` variable -
    in BOTH the code path that consumes `ModelTrialRunner.__call__`'s
    own output (`tuned_result.get('interaction', ...)`) and the main
    per-task dispatch loop's own direct-call branches (lockstep, RK-2)
    - rather than duplicated inside every one of those branches, since
    every model's own 'interaction' output already carries the SAME two
    possible shapes by the time it reaches this function:

      - No 'order' column at all: every non-shapiq route (Friedman's
        H-statistic, exact/surrogate TreeSHAP, NID, EBM's own native
        pairwise terms, GAT_prior_knowledge's own pairwise SHAP) - the
        WHOLE frame is order=2, exactly as before this function existed
        ("order absent => all k=2 => today's path", the design
        blueprint's own stated contract). Returned unchanged as the k=2
        half; the k>=3 half is empty.
      - An 'order' column present: a shapiq-based route
        (`models.interaction_extraction.canonicalize_shapiq_table()`'s
        own output shape - 'marker1'/'marker2' populated only for
        order==2 rows, 'markers' populated for every row, any order) -
        split by `order` into the two halves below.

    Parameters
    ----------
    interaction_df : pandas.DataFrame
        A model's own 'interaction' return-slot value for this task
        (`sample_interaction`/`out['interaction']`) - may be empty.

    Returns
    -------
    (pairs_df, hok_df) : both pandas.DataFrame.
        `pairs_df` has columns ``['marker1', 'marker2', 'value']`` (plus
        whatever other columns `interaction_df` already had, unchanged -
        this never drops a column some other extractor might have added).
        `hok_df` has columns ``['markers', 'order', 'value']`` - `markers`
        is a ``';'``-joined STRING of marker names (not a Python tuple):
        this frame is later written straight to a CSV/gzip file by
        `checkpoint_utils.py`'s existing, unmodified, generic per-key
        writers, exactly like `pairs_df`'s own `marker1`/`marker2`
        strings already are; a raw tuple would round-trip through
        `to_csv`/`read_csv` as its own `repr()` text, which is fragile to
        parse back reliably - `models/circos_plot.py`'s own
        Interaction_higher_order.csv reader splits this same column on
        ``';'``.
    """
    if interaction_df is None or interaction_df.shape[0] == 0:
        return (
            pd.DataFrame(columns=['marker1', 'marker2', 'value']),
            pd.DataFrame(columns=['markers', 'order', 'value']),
        )
    if 'order' not in interaction_df.columns:
        # Today's path - every row is implicitly order=2 - unchanged (I11).
        return interaction_df, pd.DataFrame(columns=['markers', 'order', 'value'])

    pairs_df = interaction_df[interaction_df['order'] == 2].copy()
    if not pairs_df.empty:
        _keep_cols = [c for c in pairs_df.columns if c not in ('markers', 'order')]
        pairs_df = pairs_df[_keep_cols].reset_index(drop=True)
    else:
        pairs_df = pd.DataFrame(columns=['marker1', 'marker2', 'value'])

    hok_df = interaction_df[interaction_df['order'] > 2].copy()
    if not hok_df.empty:
        hok_df['markers'] = hok_df['markers'].apply(lambda m: ';'.join(m))
        hok_df = hok_df[['markers', 'order', 'value']].reset_index(drop=True)
    else:
        hok_df = pd.DataFrame(columns=['markers', 'order', 'value'])

    return pairs_df, hok_df


# ---------------------------------------------------------------------------
# ver4-25, Stage 7 (design blueprint R13) / Requirements3.md item 1: the
# Python-side "other half" of GBLUP's/RKHS's/rrBLUP's/BayesB's new exact-
# predictor marker-effect routes - models/kernel_effects.py supplies the
# actual numerics (VanRaden back-solving, RKHSPredictor), the three
# functions below turn a completed R call's own output into the SAME
# 'effect' DataFrame shape every other model/method already produces
# (invariant I3), called identically from BOTH ModelTrialRunner.__call__
# and the main per-task dispatch loop (lockstep, RK-2) - there is exactly
# ONE implementation of each, not two copies to drift apart.
#
# Every function below is 'iml_shapley'/'beta_direct' (the legacy,
# default-for-old-configs value) a strict passthrough of the R call's own
# r_effect - so a config predating this update, or a user who never
# touches the new "Marker effect method" field, gets today's EXACT
# behaviour, byte-for-byte (I11).
# ---------------------------------------------------------------------------


def _apply_linear_closed_form_effect(model_name, params, effect_df, train, test):
    """rrBLUP/BayesB (Requirements3.md item 1): convert the R call's own
    directly-extracted marker-effect beta (`fm$ETA$mrk$b` - ALWAYS
    computed unconditionally by `rrBLUP.R`/`BayesB.R`, regardless of this
    field's value) into an exact linear-model Shapley effect, when the
    user has selected the 'closed_form' Marker effect method
    (hparam_specs.py's appended, append-only index 8 for both models).

    Unlike GBLUP/RKHS (which needed R-side changes to SKIP an expensive
    iml::Shapley refit block), this is a PURE Python post-processing step
    - the R call itself, and every other branch of it (pearson/MSE/
    predicted_*), is completely unaffected by this field; only how
    `effect_df` is interpreted afterwards changes.

    'beta_direct' (default, legacy behaviour for any config predating
    this field) returns `effect_df` unchanged. 'closed_form' applies
    `models.kernel_effects.linear_closed_form_effects()` to the SAME beta
    vector R already returned, converting it into an aggregated |phi|
    Shapley-style effect - directly comparable in magnitude/
    interpretation to GBLUP's own 'closed_form' route and to the
    Shapley-based effects every other model in this software already
    reports, computed at ZERO extra model-fitting cost (no refits, no
    coalitions - `beta` is already known).

    NEVER RAISES - falls back to the raw beta (with a printed NOTE) if
    anything about the conversion fails, matching
    `_r_model_surrogate_interaction()`'s own "never break a real,
    already-succeeded prediction" convention.

    Parameters
    ----------
    model_name : 'rrBLUP' or 'BayesB' (used only for the NOTE message).
    params : this task's own (possibly padded) HPARAMETERS[model_name]
        list - params[8] (append-only, I5) is the 'Marker effect method'
        field; absent/legacy configs default to 'beta_direct'.
    effect_df : the R call's own `r_list_get(result, 'r_effect')` - a
        1-row DataFrame of raw beta values, one column per marker, in
        `train`'s own marker-column order.
    train, test : this task's own train/test splits (marker columns plus
        a trailing phenotype column, exactly like every other caller in
        this module).

    Returns
    -------
    A DataFrame of the SAME shape as `effect_df` - either `effect_df`
    itself (unchanged) or the closed-form replacement.
    """
    effect_method = params[8] if len(params) > 8 else 'beta_direct'
    if effect_method != 'closed_form':
        return effect_df
    if effect_df is None or effect_df.shape[0] == 0:
        # No beta to convert (shouldn't normally happen - rrBLUP.R/
        # BayesB.R compute it unconditionally - but handled defensively
        # rather than assumed, exactly like every other effect-shape
        # check in this module).
        return effect_df
    try:
        beta = effect_df.to_numpy(dtype=float).reshape(-1)
        X_train = train.iloc[:, :-1].to_numpy(dtype=float)
        if test is not None and test.shape[0] > 0:
            X_explain = test.iloc[:, :-1].to_numpy(dtype=float)
        else:
            # No test rows to explain (an edge-case split) - explain the
            # training rows themselves rather than producing an empty
            # effect table, mirroring gblup_closed_form_effects()'s own
            # "always returns a full-width vector" contract.
            X_explain = X_train
        phi = kernel_effects.linear_closed_form_effects(beta, X_explain, X_train, aggregate=True)
        return pd.DataFrame([phi], columns=effect_df.columns)
    except Exception as exc:
        print(f"[GP] NOTE: {model_name} closed_form marker-effect conversion failed ({exc!r}) - "
              f"falling back to this model's own raw beta (the 'beta_direct' method's output) "
              f"for this task instead.")
        return effect_df


def _apply_gblup_closed_form_effect(result_GBLUP, gblup_params, train, test):
    """GBLUP (design blueprint R13): when the 'Marker effect method'
    field (hparam_specs.py's appended, append-only index 13) is
    'closed_form', turn this fit's own `r_u`/`r_mu` (GBLUP.R's own
    already-completed main BGLR fit output - no extra BGLR call) into an
    exact VanRaden-identity closed-form Shapley effect via
    `models.kernel_effects.gblup_closed_form_effects()`. 'iml_shapley'
    (default for any config predating this field) returns
    `r_list_get(result_GBLUP, 'r_effect')` completely unchanged - the
    legacy iml::Shapley perturbation-refit route GBLUP.R still runs for
    that value (I11).

    NEVER RAISES - falls back to an all-zero effect vector (with a
    printed NOTE) on failure: unlike rrBLUP/BayesB's own closed_form
    route above, there is no "raw beta" to fall back to here - GBLUP.R's
    own 'closed_form' branch deliberately skips computing ANY r_effect
    (returns `data.frame()`) when this method is selected, since the
    entire point is to never pay for the legacy route at all.

    Parameters
    ----------
    result_GBLUP : this call's raw R result (or the process-isolated
        fan-out's own plain-dict equivalent - both work identically with
        `r_list_get()`).
    gblup_params : this task's own (possibly padded) HPARAMETERS['GBLUP']
        list, with 'all' -> test.shape[0] (Shapley_num, index 3) already
        resolved by the caller, exactly as every other GBLUP branch
        expects.
    train, test : this task's own train/test splits.

    Returns
    -------
    A 1-row DataFrame, one column per marker (train's own marker-column
    order) - the SAME shape `r_list_get(result_GBLUP, 'r_effect')` always
    has.
    """
    effect_method = gblup_params[13] if len(gblup_params) > 13 else 'iml_shapley'
    if effect_method != 'closed_form':
        return r_list_get(result_GBLUP, 'r_effect')
    get_effect = bool(gblup_params[2]) if len(gblup_params) > 2 else False
    marker_cols = list(train.columns[:-1])
    if not get_effect:
        # GBLUP.R's own 'else' branch (get_effect==FALSE) already
        # returned an empty r_effect - nothing to compute either way.
        return r_list_get(result_GBLUP, 'r_effect')
    try:
        r_u = np.asarray(r_list_get(result_GBLUP, 'r_u'), dtype=float).reshape(-1)
        n_train = train.shape[0]
        u_train = r_u[:n_train]
        X_train = train.iloc[:, :-1].to_numpy(dtype=float)
        Shapley_num = gblup_params[3]
        if test is not None and test.shape[0] > 0:
            len_rows = test.shape[0] if Shapley_num == 'all' else max(0, min(int(Shapley_num), test.shape[0]))
            len_rows = len_rows if len_rows > 0 else test.shape[0]
            X_explain = test.iloc[:len_rows, :-1].to_numpy(dtype=float)
        else:
            X_explain = X_train
        scale = X_train.shape[1]
        phi = kernel_effects.gblup_closed_form_effects(X_train, u_train, X_explain, scale=scale, aggregate=True)
        return pd.DataFrame([phi], columns=marker_cols)
    except Exception as exc:
        print(f"[GP] NOTE: GBLUP closed_form marker-effect computation failed ({exc!r}) - "
              f"reporting an all-zero effect vector for this task instead of a possibly-wrong "
              f"one (this model's own predictions are UNAFFECTED - only its marker effects for "
              f"this task are affected). Switching 'Marker effect method' back to 'iml_shapley' "
              f"recovers the legacy route.")
        return pd.DataFrame([np.zeros(len(marker_cols))], columns=marker_cols)


# Bug fix (V13.2 gate miscalibration - the root cause of "shapiq_predictor
# gets forced to iml_shapley for every RKHS task", distinct from the
# already-fixed `RKHSPredictor` blend bug documented in
# `models/kernel_effects.py`): unlike GBLUP's `closed_form` route (an
# exact, deterministic VanRaden back-solve - no randomness anywhere, so
# comparing to `1e-6 * sd(y)` floating-point-level precision is correct
# there), RKHS's own main fit is a Gibbs/MCMC (BGLR) estimate, so its
# reported test-row predictions are themselves only a finite-sample MCMC
# average that carries its own irreducible sampling noise (the standard
# GP "conditional variance not explained by the training block" term,
# realised through finitely many post-burn-in draws) - no amount of
# numerical fix to the reconstruction itself (e.g. the blend fix above)
# can make it match that noisy target to floating-point precision.
#
# History: this was first "fixed" with one FIXED relative tolerance
# (`RKHS_RECON_MC_TOL_REL` below, checked against the single worst-case
# `max|reconstruction-real|` across all explained test rows). Simulating
# BGLR's own RKHS Gibbs sampler showed why that doesn't actually work: the
# MC noise on a single, worst-covered test row (the one farthest, in
# kernel distance, from every training individual) and the signature of a
# genuinely SMALL reconstruction bug overlap heavily in this metric - e.g.
# a `h` off by just 10% produced a worst-row mismatch of ~2.4% of sd(y),
# barely above the ~1.2% a CORRECT reconstruction's own MC noise can
# reach, and a real production run measured 6.6% with no bug at all
# (max|diff|=0.066 against a 5%-of-sd(y)=0.0353 tolerance) - there is no
# single dataset-agnostic percentage that cleanly separates "noisy but
# correct" from "wrong", because how noisy a correct reconstruction looks
# depends on THIS task's own fitted variance components and how well each
# test row happens to be covered by the training kernel, not on a
# universal constant.
#
# Fix: BGLR already estimates exactly the two quantities needed to build
# a task-specific, statistically-grounded tolerance instead of guessing
# one - `fm$ETA[[1]]$varU`/`fm$varE` (`RKHS.R`'s `need_reconstruction`
# block, returned as `r_var_u`/`r_var_e`). Per test row, the expected
# standard error of the MC-averaged reconstruction is
# `sqrt(varU * conditional_variance_unit(row) / n_eff)` -
# `conditional_variance_unit()` (`models.kernel_effects.RKHSPredictor.
# conditional_variance()`) is the row's own GP "not explained by
# training" variance (small for a row with close training neighbours,
# large for a poorly-covered one - exactly the source of the single-
# worst-row failures above), and `n_eff = nIter - burnIn` (the RAW
# post-burn-in iteration count, deliberately not corrected for Gibbs
# autocorrelation - i.e. an OPTIMISTIC/small standard error, which
# `RKHS_RECON_MC_TOL_SIGMA` below compensates for with a generous
# multiplier rather than pretending to estimate the chain's true
# effective sample size). Calibration evidence (same simulation): a
# CORRECT reconstruction never needed more than ~2.1 of these standard
# errors on its worst row (median ~0.8); a `h` off by 50% needed ~17.3
# (median ~4.1); off by 2x needed ~31 (median ~7.1) - an 8x-plus gap
# between "correct" and even a middling bug, unlike the ~2x gap the flat
# percentage gave. `RKHS_RECON_MC_TOL_SIGMA = 8` sits in that gap with
# comfortable margin on both sides.
#
# `RKHS_RECON_MC_TOL_REL` is kept ONLY as a defensive fallback for the
# (expected-never) case where `r_var_u`/`r_var_e` come back missing/NaN -
# e.g. an `RKHS.R` that predates this fix, or an unexpected BGLR ETA
# shape - so a result missing those two fields still gets a real check
# rather than either crashing or skipping the gate outright.
#
# Follow-up bug fix (systematic, batch-wide fallback that survived the
# above): the calibration above assumes the reconstruction's only source
# of error against `predicted_test` is genuine Monte Carlo noise around
# an otherwise-correct point estimate. That assumption itself was wrong
# often enough to matter - BGLR's own RKHS fit eigen-truncates the
# WEIGHTED, FULL train+valid+test grid kernel (its own `tolD`, default
# 1e-10) before fitting, which the training-block-only reconstruction
# `_build_gated_rkhs_predictor()` used to always build never replicated,
# producing a genuine, systematic BIAS (not noise a sigma-based tolerance
# can average away) whenever BGLR's own truncation actually dropped
# eigenvectors - a property of the population's own kernel (related/
# structured individuals), reproduced under both heterogeneous AND
# near-uniform BLUE-reliability weights, and "across the whole batch" -
# see `kernel_effects.solve_rkhs_grid_alpha()`'s own module-level
# docstring for the full derivation and numerical validation.
# `_build_gated_rkhs_predictor()` below now attempts that exact,
# truncation-consistent full-grid reconstruction FIRST (whenever this
# fit's own `varU`/`varE` are available and `valid` was supplied),
# falling back to the training-block-only route unchanged otherwise -
# this tolerance derivation itself did not need to change; it was
# already correct for what it actually measures (MC noise around a
# correct point estimate), it just used to be checked against a
# point estimate that was not always correct.
RKHS_RECON_MC_TOL_SIGMA = 8.0
RKHS_RECON_MC_TOL_REL = 0.05

# Bug fix (2026-09, "which reconstruction to trust when BOTH the full-grid
# AND the training-block-only routes fail the strict gate" - a gap the
# retry logic below used not to handle at all: it always discarded the
# full-grid reconstruction on ANY gate failure and bet everything on the
# training-block-only retry, on the documented assumption that the retry
# is "the better-calibrated estimator" - see `_build_gated_rkhs_predictor`'s
# own docstring's "V13.2 gate false-positive specific to the full-grid
# reconstruction" section. Production evidence (a task with highly
# heterogeneous BLUE-reliability weights, ~134x train-weight range) showed
# that assumption can fail: the full-grid reconstruction missed its own
# tolerance by only ~1.3x (a plausible elevated-noise case) on one row,
# while the training-block-only retry the code unconditionally switched to
# then missed by ~375x on a DIFFERENT row - two orders of magnitude past
# even this fit's own worst documented BUG signature (see below), yet the
# old code never compared the two: it discarded the mildly-off
# reconstruction in favour of the catastrophically-off one and then gave up
# entirely (all-zero effect vector / surrogate-method fallback), even
# though the mildly-off reconstruction was sitting right there.
#
# `_build_gated_rkhs_predictor()` now compares the two reconstructions'
# own worst-row ratios (`row_diff / row_tol`, computed by
# `_rkhs_v132_worst_ratio()` below) whenever neither cleanly passes
# (ratio <= 1.0), and accepts whichever is smaller PROVIDED it stays under
# this constant - deliberately not an arbitrary "close enough" number:
# `RKHS_RECON_MC_TOL_SIGMA`'s own calibration comment (above) reports a
# correct reconstruction's worst row needing at most ~2.1 of these
# standard errors (comfortably under the already-generous 8-sigma primary
# tolerance, i.e. ratio well under 1.0 in the ordinary case), while even
# the SMALLEST simulated bug ("h off by 50%") needed ~17.3 standard errors
# at its own worst case - i.e. a ratio of ~17.3/8 = 2.16x this tolerance.
# `RKHS_RECON_ACCEPT_RATIO = 2.0` sits just under that smallest known
# real-bug signature, so this secondary band only ever accepts a
# reconstruction whose own elevated noise the codebase's own calibration
# evidence has never associated with an actual reconstruction bug - it
# does NOT widen the gate into the ambiguous region a flat-percentage
# tolerance used to fail in (see `RKHS_RECON_MC_TOL_SIGMA`'s own comment
# for why that approach was replaced). A ratio at or beyond this band
# (this task's own 375x training-block failure among them) is never
# accepted regardless of which reconstruction produced it or whether it
# is merely "the smaller of two bad options" - `_build_gated_rkhs_
# predictor()` still raises `_RKHSGateFailure` for real in that case, so a
# task whose explanation truly cannot be trusted still falls through to
# the existing all-zero-effect / surrogate-interaction safety net exactly
# as before this fix (I11 for that path).
RKHS_RECON_ACCEPT_RATIO = 2.0


def _rkhs_v132_worst_ratio(predictor, test, predicted_test, rkhs_params, r_var_u, r_var_e,
                           have_var_components, weights):
    """The same reconstruction-fidelity computation `_rkhs_v132_gate_check()`
    uses, factored out to return the worst-row ``(diff / tolerance)`` ratio
    and its own diagnostic message instead of raising - lets
    `_build_gated_rkhs_predictor()`'s own accept/retry policy (see that
    function's own docstring and `RKHS_RECON_ACCEPT_RATIO`'s own comment
    above) compare the full-grid and training-block-only reconstructions'
    fidelity NUMERICALLY, rather than only learning pass/fail from a raised
    exception. `_rkhs_v132_gate_check()` itself is unchanged in behaviour
    (still raises `_RKHSGateFailure` at ratio > 1.0, byte-identical to
    before this update - I11) and now calls this helper internally so the
    tolerance derivation is stated in exactly one place.

    Returns
    -------
    (worst_ratio, message) : `worst_ratio` is the worst offending row's own
        `row_diff / row_tol` (or `max_diff / tol` in the defensive
        `have_var_components=False` path below) - `<= 1.0` means the
        primary V13.2 gate would pass outright. `message` is the same
        diagnostic string `_RKHSGateFailure` would have carried, or `None`
        when `worst_ratio <= 1.0` (nothing to report).
    """
    X_test_full = test.iloc[:, :-1].to_numpy(dtype=float)
    recon_test = predictor.predict(X_test_full)
    real_test = np.asarray(predicted_test, dtype=float).reshape(-1)
    if real_test.shape[0] != recon_test.shape[0]:
        return float('inf'), (
            f"reconstruction produced {recon_test.shape[0]} predictions but this fit's own "
            f"BGLR test predictions has {real_test.shape[0]}")
    y_sd = float(np.std(real_test)) if real_test.size > 1 else float(np.abs(real_test).max() if real_test.size else 0.0)
    row_diff = np.abs(recon_test - real_test)

    # Bug fix (see RKHS_RECON_MC_TOL_SIGMA's own module-level comment
    # for the full derivation/calibration): was a single FIXED
    # relative tolerance (`RKHS_RECON_MC_TOL_REL * y_sd`, itself
    # replacing an even stricter `1e-6 * y_sd` copied from GBLUP's
    # exact closed-form gate) checked against the single worst-case
    # `max(row_diff)` - RKHS's BGLR fit is Monte-Carlo estimated, so
    # how noisy a CORRECT reconstruction looks varies per task and
    # per row (worse for a test individual poorly covered by the
    # training kernel), and a flat percentage cannot tell that noise
    # apart from a real, but small, reconstruction bug - both a
    # production run (max|diff|=6.6% of sd(y), no bug) and a small
    # simulated `h` bug (~2.4-10.6%) can land on the SAME side of any
    # single fixed cutoff. Preferred check: build each row's own
    # tolerance from this fit's fitted variance components (BGLR's
    # `varU`/`varE`, `r_var_u`/`r_var_e` below) and that row's GP
    # "not explained by training" conditional variance - a
    # statistically-grounded, task- and row-adaptive standard error,
    # not a guessed constant.
    if have_var_components:
        n_eff = max(1.0, float(rkhs_params[0]) - float(rkhs_params[1]))  # nIter - burnIn
        # Bug fix (V13.2 gate false-positive under BLUE-reliability
        # weighting - see kernel_effects.RKHSPredictor.conditional_
        # variance()'s own docstring for the full derivation): forward
        # THIS fit's own per-training-row fitting weight (`weights` -
        # the SAME `w_fit` `RKHS.R`'s own weighted BGLR call used) and
        # residual variance (`r_var_e`) so a test row whose nearest
        # training neighbours were fit with LOW reliability weight (and
        # are therefore genuinely noisier evidence, even though nothing
        # about the reconstruction ITSELF is wrong) gets a correspondingly
        # WIDER, not falsely tight, tolerance. `weights=None` (no BLUE
        # weighting active, or an unweighted run) reduces byte-identically
        # to the previous, unregularised formula (I11).
        cond_var = predictor.conditional_variance(
            X_test_full, var_u=float(r_var_u[0]), var_e=float(r_var_e[0]), weights=weights,
        )
        row_tol = RKHS_RECON_MC_TOL_SIGMA * np.sqrt(cond_var / n_eff) + 1e-9
        ratio = row_diff / row_tol
        worst = int(np.argmax(ratio))
        worst_ratio = float(ratio[worst])
        message = None
        if worst_ratio > 1.0:
            message = (
                f"row {worst}: |reconstruction - real BGLR prediction|={row_diff[worst]:.3g} "
                f"exceeds its own MC-noise-scaled tolerance {row_tol[worst]:.3g} "
                f"({RKHS_RECON_MC_TOL_SIGMA:g} sigma using this fit's own varU={r_var_u[0]:.3g}, "
                f"n_eff={n_eff:.0f}, V13.2 gate)")
        return worst_ratio, message
    else:
        # Defensive fallback (expected-never - see RKHS_RECON_MC_TOL_REL's
        # own comment): r_var_u/r_var_e missing or non-finite, e.g. an
        # RKHS.R predating this fix. A real check, not a skip.
        tol = max(RKHS_RECON_MC_TOL_REL * y_sd, 1e-9)
        max_diff = float(row_diff.max()) if row_diff.size else 0.0
        worst_ratio = max_diff / tol
        message = None
        if worst_ratio > 1.0:
            message = (
                f"max|reconstruction - real BGLR prediction|={max_diff:.3g} exceeds fallback "
                f"tolerance {tol:.3g} ({RKHS_RECON_MC_TOL_REL:.0%} of sd(y) - r_var_u/r_var_e "
                f"unavailable, V13.2 gate fallback)")
        return worst_ratio, message


def _rkhs_v132_gate_check(predictor, test, predicted_test, rkhs_params, r_var_u, r_var_e,
                           have_var_components, weights):
    """Blueprint V13.2 reconstruction-fidelity check - factored out of
    `_build_gated_rkhs_predictor()` (byte-identical check, just no longer
    inlined) so this update's own grid-vs-training-block retry (see that
    function's own docstring) can run it TWICE, against two different
    reconstructions of the SAME task, without duplicating the tolerance
    derivation itself. Raises `_RKHSGateFailure` on failure; returns
    `None` on success.

    Delegates to `_rkhs_v132_worst_ratio()` (byte-identical computation,
    I11) so `_build_gated_rkhs_predictor()`'s own accept/retry policy can
    reuse the exact same tolerance derivation numerically instead of only
    via this function's raise/no-raise interface.
    """
    worst_ratio, message = _rkhs_v132_worst_ratio(
        predictor, test, predicted_test, rkhs_params, r_var_u, r_var_e, have_var_components, weights)
    if worst_ratio > 1.0:
        raise _RKHSGateFailure(message)


def _build_gated_rkhs_predictor(result_RKHS, rkhs_params, train, test, predicted_test, weights=None,
                                 valid=None):
    """Reconstruct this RKHS fit's own kernel regression as a pure-NumPy
    `models.kernel_effects.RKHSPredictor` and apply the V13.2
    reconstruction-fidelity gate documented in the module-level comment
    directly above this function - the single "reconstruct, then gate"
    step shared by BOTH of RKHS's own non-surrogate explanation routes:

      - the marker-EFFECT 'shap_predictor'/'shapiq_predictor' routes
        (`_apply_rkhs_shapiq_predictor_effect()`, below - this is a pure
        extraction of what used to be inlined at the top of that
        function's own `try` block, byte-identical otherwise);
      - the marker-PAIR-INTERACTION 'shapiq_predictor' route
        (`_apply_rkhs_predictor_interaction()`, further below - a NEW
        route added by this update, so that RKHS's own interaction
        search can, like SVR's/KNN's own 'shapiq' choice, explain this
        model's OWN prediction function directly with real Shapley
        interaction indices, instead of only ever being able to explain
        a lightweight surrogate fitted to this model's own predictions
        the way `_r_model_surrogate_interaction()` does).

    Keeping the gate in exactly one place means both routes are held to
    the identical fidelity standard before either is trusted, and a
    future change to the tolerance derivation only has to be made once -
    the same single-authority discipline `_map_genes_to_markers` already
    established elsewhere in this codebase (see the architecture
    document's own §8/§19.2 for that precedent).

    Parameters
    ----------
    result_RKHS, rkhs_params, train : see
        `_apply_rkhs_shapiq_predictor_effect()`'s own identical
        parameters.
    test : this task's own test split - MUST be non-empty. Both callers
        check `test is None or test.shape[0] == 0` themselves first
        (they return different empty-result shapes for that case - a
        1-row all-zero effect DataFrame vs. an empty interaction
        DataFrame - so that check is deliberately not duplicated here).
    predicted_test : this call's own real BGLR test predictions - the
        gate's own comparison target.
    weights : (n_train,) array-like or None, keyword-only, default None
        Bug fix (V13.2 gate false-positive under BLUE-reliability
        weighting): this fit's own per-TRAINING-row fitting weight - the
        SAME `w_fit` value (in `train`'s own row order, BEFORE the
        `sqrt()` `RKHS.R`'s own `BGLR(..., weights=sqrt(w_fit), ...)`
        call applies) both callers already forward to `RKHS.R` itself.
        Forwarded to
        `models.kernel_effects.RKHSPredictor.conditional_variance()` so
        the gate's own per-row tolerance correctly widens for a test row
        whose nearest training neighbours were fit with LOW reliability
        weight (genuinely noisier evidence, even when nothing about the
        reconstruction itself is wrong) - see that method's own
        docstring for the full derivation. `None` (default - no BLUE
        weighting active) keeps the gate byte-identical to before this
        fix (I11).
    valid : this task's own validation split, or `None`/keyword-only,
        default `None`. Bug fix (systematic, batch-wide V13.2 gate
        fallback - see `kernel_effects.solve_rkhs_grid_alpha()`'s own
        module-level docstring for the full root-cause derivation):
        BGLR's own RKHS fit truncates near-zero eigenvalues of the
        WEIGHTED, FULL `train+valid+test` grid kernel before fitting -
        something the training-block-only reconstruction below can never
        replicate, regardless of BLUE-reliability weighting. Supplying
        `valid` (both callers now always do, since `RKHS.R` itself always
        receives it) lets this function attempt the exact, truncation-
        consistent full-grid reconstruction instead; `None` (or this
        fit's own `r_var_u`/`r_var_e` being unavailable) skips that
        attempt and falls back to the pre-existing training-block-only
        route byte-for-byte (I11) - an empty (zero-row) `valid` is fine
        (a task with no validation split still has a well-defined
        `train+test` grid).

    Returns
    -------
    (predictor, X_train_full) : the constructed
        `models.kernel_effects.RKHSPredictor` and the raw `(n_train, M)`
        training marker matrix it was built from (handed back so a
        caller can build its own masking baseline - see
        `_apply_rkhs_shapiq_predictor_effect()`'s/`_apply_rkhs_predictor_
        interaction()`'s own `baseline = X_train_full.mean(axis=0)` -
        without re-slicing `train` a second time). NOTE: `X_train_full`
        is always the TRAINING block only, even when `predictor` itself
        is using the new full-grid reconstruction internally - a caller
        needing the SAME reference-row set `predictor.predict()` is
        actually using (e.g. for a closed-form interaction Hessian) must
        use `predictor.X_ref`/`predictor.alpha_ref` instead, never
        `X_train_full`/`predictor.alpha` directly (see `RKHSPredictor`'s
        own docstring).

    Bug fix (this update - V13.2 gate false-positive specific to the
    full-grid reconstruction): `RKHS_RECON_MC_TOL_SIGMA`'s own calibration
    (module-level comment above) was derived and validated against the
    OLDER training-block-only reconstruction (`alpha = K_train^-1 @
    u_train`), which shares `u_train`'s own MCMC sampling noise with
    `predicted_test` (BGLR's own `yHat`, drawn from the SAME chain) - a
    common noise source the two largely cancel against each other, which
    is exactly what that calibration's "a correct reconstruction's worst
    row never needed more than ~2.1 of these standard errors" evidence
    measured. The full-grid route's own `alpha_grid`
    (`kernel_effects.solve_rkhs_grid_alpha()`) is instead the ANALYTIC
    ridge-regression solution derived directly from `y_train`/`mu`/
    `varU`/`varE`, deliberately bypassing `u_train` (see that function's
    own docstring) - it has no such shared noise to cancel, so it can
    (rarely, and more easily when this task's own BLUE-reliability
    weights are highly heterogeneous) need more sigma than that
    calibration allows for, EVEN ON A TASK WHERE BGLR'S OWN EIGENVALUE
    TRUNCATION NEVER ACTUALLY FIRED (`n_grid_dropped_eigs == 0`) - i.e.
    exactly the case where the full-grid route's only real advantage over
    the training-block route (truncation-consistency) does not apply, so
    there is nothing to lose by preferring the better-calibrated
    estimator instead. Below, a V13.2 gate failure on the full-grid
    reconstruction is therefore no longer immediately fatal to this call:
    it is compared against the training-block-only reconstruction (the
    estimator the gate's own tolerance was actually validated against)
    before either using whichever one actually passes, or raising
    `_RKHSGateFailure` for real. This is what lets both callers below stop
    needing a fresh, expensive `iml_shapley` BGLR refit as their own last
    resort (removed by this update - see
    `_apply_rkhs_shapiq_predictor_effect()`'s own docstring for why that
    fallback is no longer taken).

    Bug fix (2026-09, "which reconstruction to trust when BOTH fail the
    strict gate" - see `RKHS_RECON_ACCEPT_RATIO`'s own module-level
    comment for the full production evidence and derivation): the
    original version of the fix above always discarded the full-grid
    reconstruction on ANY gate failure and bet everything on the
    training-block-only retry, uncritically, on the assumption above that
    it is "the better-calibrated estimator" - true on average, but not
    guaranteed for every task. When the training-block-only retry ALSO
    fails the gate, this function now compares the two reconstructions'
    own worst-row tolerance-violation ratios (`_rkhs_v132_worst_ratio()`)
    and accepts whichever is smaller, provided it stays under
    `RKHS_RECON_ACCEPT_RATIO` - a band bounded by this fit's own
    documented bug-calibration evidence, not an arbitrary "close enough"
    cutoff. A reconstruction whose own ratio exceeds that band is never
    accepted, regardless of whether it is "the smaller of two bad
    options" - seeing SUCH a reconstruction with `baseline_shap`/
    `shap_predictor` would be scientifically indefensible even though the
    caller would prefer to avoid falling back.

    Raises
    ------
    _RKHSGateFailure
        If NEITHER the full-grid reconstruction NOR the training-block-only
        one reproduces `predicted_test` within `RKHS_RECON_ACCEPT_RATIO`
        of the gate's own tolerance - see that constant's own module-level
        comment for the full acceptance policy. Any OTHER exception (a
        malformed `result_RKHS`, a singular kernel) propagates as-is;
        both callers already wrap this call in their own `try`/`except`.
    """
    r_u = np.asarray(r_list_get(result_RKHS, 'r_u'), dtype=float).reshape(-1)
    r_mu = float(np.asarray(r_list_get(result_RKHS, 'r_mu'), dtype=float).reshape(-1)[0])
    r_d_scale = float(np.asarray(r_list_get(result_RKHS, 'r_d_scale'), dtype=float).reshape(-1)[0])
    h = float(rkhs_params[2])
    n_train = train.shape[0]
    u_train = r_u[:n_train]
    X_train_full = train.iloc[:, :-1].to_numpy(dtype=float)

    # Bug fix (systematic, batch-wide V13.2 gate fallback - see
    # kernel_effects.solve_rkhs_grid_alpha()'s own module-level docstring
    # for the full root-cause derivation and numerical validation): BGLR's
    # own RKHS fit eigen-truncates the WEIGHTED, FULL train+valid+test
    # grid kernel before fitting - a structural property of the fit
    # itself, unrelated to BLUE-reliability weighting (confirmed: this
    # reproduces both under highly heterogeneous fitting weights and
    # under near-uniform ones, and across an entire batch, not as an
    # isolated tail event) - which the training-block-only back-solve
    # below can never replicate no matter how well-conditioned ITS OWN
    # solve is made. `r_var_u`/`r_var_e` (BGLR's own fitted variance
    # components) are needed for the ridge term in that reconstruction;
    # `have_var_components` mirrors the identical check the V13.2 gate's
    # own tolerance derivation further below already makes for the same
    # reason (an RKHS.R predating that fix, or an unexpected BGLR ETA
    # shape, genuinely lacks these) - computed here, ahead of that later
    # use, so both can share one `r_var_u`/`r_var_e` extraction.
    r_var_u_raw = r_list_get(result_RKHS, 'r_var_u', default=np.nan)
    r_var_e_raw = r_list_get(result_RKHS, 'r_var_e', default=np.nan)
    r_var_u = np.asarray(r_var_u_raw, dtype=float).reshape(-1)
    r_var_e = np.asarray(r_var_e_raw, dtype=float).reshape(-1)
    have_var_components = (
        r_var_u.size > 0 and r_var_e.size > 0
        and np.isfinite(r_var_u[0]) and np.isfinite(r_var_e[0]) and r_var_u[0] > 0
    )

    grid_kwargs = {}
    if have_var_components and valid is not None:
        try:
            n_markers = X_train_full.shape[1]
            X_valid_full = (
                valid.iloc[:, :-1].to_numpy(dtype=float) if valid.shape[0]
                else np.empty((0, n_markers))
            )
            X_test_full_grid = test.iloc[:, :-1].to_numpy(dtype=float)
            X_grid = np.vstack([X_train_full, X_valid_full, X_test_full_grid])
            n_valid_test = X_valid_full.shape[0] + X_test_full_grid.shape[0]
            if weights is not None:
                weights_full = np.concatenate(
                    [np.sqrt(np.asarray(weights, dtype=float).reshape(-1)), np.ones(n_valid_test)])
            else:
                weights_full = np.ones(n_train + n_valid_test)
            y_train_actual = train.iloc[:, -1].to_numpy(dtype=float)
            alpha_grid, _n_kept, _n_dropped = kernel_effects.solve_rkhs_grid_alpha(
                X_grid, weights_full, y_train_actual, r_mu, h, r_d_scale,
                float(r_var_u[0]), float(r_var_e[0]), n_train,
            )
            grid_kwargs = dict(X_grid=X_grid, alpha_grid=alpha_grid, n_grid_dropped_eigs=_n_dropped)
            if _n_dropped:
                print(f"[RKHS] NOTE: this task's own weighted train+valid+test grid kernel "
                      f"eigendecomposition dropped {_n_dropped} near-zero eigenvalue(s) below "
                      f"BGLR's own tolD=1e-10 threshold (kept {_n_kept}/{_n_kept + _n_dropped}) - "
                      f"related/structured individuals made the grid kernel numerically "
                      f"rank-deficient, exactly as BGLR's own fit truncates it. The reconstruction "
                      f"below now replicates that SAME truncation exactly (see "
                      f"kernel_effects.solve_rkhs_grid_alpha's own docstring), rather than the "
                      f"untruncated training-block-only back-solve that used to cause spurious "
                      f"V13.2 gate fallbacks on exactly this kind of population.")
        except Exception as exc:
            print(f"[RKHS] NOTE: full-grid weighted-kernel reconstruction failed for this task "
                  f"({exc!r}) - falling back to the training-block-only reconstruction (the "
                  f"pre-existing route, still gated by the V13.2 check below).")
            grid_kwargs = {}

    predictor = kernel_effects.RKHSPredictor(X_train_full, u_train, r_mu, h, r_d_scale, **grid_kwargs)
    # Bug fix (kernel_effects._solve_kernel_alpha - silent ill-conditioning):
    # surfaced here, not just left as a silent internal attribute, so a
    # near-singular training-block kernel (near-duplicate individuals in
    # marker space - routine once marker count far exceeds training-set
    # size with any family/population structure) is visible in this run's
    # own log, even on a task where the V13.2 gate below still passes.
    # (Diagnostic only even when the full-grid route above is in use - the
    # training-block-only alpha/L_train it floors are still built every
    # time, for conditional_variance()'s own use.)
    _n_floored = getattr(predictor, '_n_floored_eigs', 0)
    if _n_floored:
        print(f"[RKHS] NOTE: the training-block kernel matrix for this task was near-singular "
              f"(near-duplicate training individuals in marker space) - "
              f"kernel_effects.RKHSPredictor floored {_n_floored} near-zero eigenvalue(s) to "
              f"reconstruct a numerically stable back-solve (see "
              f"kernel_effects._solve_kernel_alpha's own docstring).")

    # V13.2 gate. `_rkhs_v132_worst_ratio()` (used here instead of the
    # raise/no-raise `_rkhs_v132_gate_check()`) reports HOW BADLY this
    # reconstruction misses its own tolerance, not just whether it misses,
    # so the accept/retry policy below can compare the full-grid and
    # training-block-only reconstructions numerically instead of always
    # discarding the former on any failure - see RKHS_RECON_ACCEPT_RATIO's
    # own module-level comment for the full rationale and the production
    # case that motivated it.
    grid_ratio, grid_msg = _rkhs_v132_worst_ratio(
        predictor, test, predicted_test, rkhs_params, r_var_u, r_var_e, have_var_components, weights)
    if grid_ratio <= 1.0:
        # Cleanly passes - byte-identical to before this update (I11).
        return predictor, X_train_full

    if not predictor.uses_grid_reconstruction:
        # Already the training-block-only reconstruction (no grid route
        # was attempted, or it wasn't available for this fit) - nothing
        # left to compare against (I11: byte-identical to before this
        # update for this case).
        raise _RKHSGateFailure(grid_msg)

    # Bug fix (this update) - see this function's own docstring's
    # "V13.2 gate false-positive specific to the full-grid reconstruction"
    # section for why the full-grid route's own gate failures are
    # disproportionately likely to be false positives. Compute the
    # training-block-only reconstruction's own ratio too, so the decision
    # below is based on both reconstructions' actual fidelity rather than
    # unconditionally preferring one.
    print(f"[RKHS] NOTE: this task's own full-grid reconstruction failed the V13.2 gate "
          f"({grid_msg}) - computing the training-block-only reconstruction's own fidelity "
          f"before deciding which one this task's marker effect/interaction explanation should "
          f"use (see RKHS_RECON_ACCEPT_RATIO's own module-level comment for the policy).")
    block_predictor = kernel_effects.RKHSPredictor(X_train_full, u_train, r_mu, h, r_d_scale)
    block_ratio, block_msg = _rkhs_v132_worst_ratio(
        block_predictor, test, predicted_test, rkhs_params, r_var_u, r_var_e, have_var_components,
        weights)

    if block_ratio <= 1.0:
        # Bug fix (this update) - the ordinary "retry succeeds" case,
        # unchanged in outcome from before this update (I11): the
        # training-block-only reconstruction (the estimator
        # RKHS_RECON_MC_TOL_SIGMA's own calibration was actually validated
        # against) cleanly passes on its own, so it is preferred outright
        # regardless of how the full-grid route's own (failing) ratio
        # compares.
        print(f"[RKHS] NOTE: the training-block-only reconstruction passed the V13.2 gate - this "
              f"task's marker effect/interaction explanation uses that reconstruction instead of "
              f"the full-grid one for this task only.")
        return block_predictor, X_train_full

    # Bug fix (this update) - neither reconstruction cleanly passes.
    # Previously this was an immediate, unconditional `_RKHSGateFailure`
    # (the caller then reports an all-zero effect vector, or falls back to
    # a surrogate interaction method) - even when, as in the production
    # case that motivated this fix, one of the two reconstructions was
    # only mildly over its own tolerance while the other was catastrophically
    # over it. Accept whichever of the two has the SMALLER ratio, but only
    # while it stays under RKHS_RECON_ACCEPT_RATIO - see that constant's own
    # module-level comment for why this band is bounded by this codebase's
    # own bug-calibration evidence rather than picked arbitrarily, and why a
    # ratio beyond it (like this task's own training-block-only failure,
    # which needed ~375x its tolerance) is never accepted regardless of
    # which reconstruction produced it.
    if grid_ratio <= block_ratio:
        best_ratio, best_msg, best_predictor, best_label = grid_ratio, grid_msg, predictor, 'full-grid'
    else:
        best_ratio, best_msg, best_predictor, best_label = (
            block_ratio, block_msg, block_predictor, 'training-block-only')

    if best_ratio <= RKHS_RECON_ACCEPT_RATIO:
        print(f"[RKHS] NOTE: neither reconstruction cleanly passed the V13.2 gate for this task, "
              f"but the {best_label} reconstruction's own worst-row ratio ({best_ratio:.2g}x its "
              f"tolerance) stays under the {RKHS_RECON_ACCEPT_RATIO:g}x band this fit's own "
              f"calibration associates with elevated-but-plausible Monte Carlo noise rather than a "
              f"genuine reconstruction bug - accepting it (this task's marker effect/interaction "
              f"explanation keeps using 'baseline_shap'/'shap_predictor' rather than falling back) "
              f"instead of treating this task's own explanation as unreliable ({best_msg}).")
        return best_predictor, X_train_full

    raise _RKHSGateFailure(
        f"both the full-grid ({grid_ratio:.3g}x tolerance) and training-block-only "
        f"({block_ratio:.3g}x tolerance) reconstructions exceed the {RKHS_RECON_ACCEPT_RATIO:g}x "
        f"acceptance band this fit's own calibration allows for elevated-but-plausible MC noise - "
        f"best available: {best_label} ({best_msg})")


# Bug fix (RKHS baseline_shap OOM / worker SIGKILL during marker-effect
# and marker-interaction Shapley extraction - user-reported crash,
# 2026-09): both `_apply_rkhs_shapiq_predictor_effect()`'s and
# `_apply_rkhs_predictor_interaction()`'s own `_masked_predict()` rebuild
# the FULL original marker matrix (`n_features` columns - every marker the
# BGLR/RKHS fit ever saw, NOT just the shortlisted ones being explained)
# for every batched call, because `RKHSPredictor` cannot be refit on a
# marker subset (its kernel/`d_scale` are tied to the exact original fit -
# see the comment above each `_masked_predict` definition). Every OTHER
# caller-side memory knob (`baseline_shap_interactions`'s own
# `pair_chunk_size`/`interaction_background`, its row-parallel `n_jobs`
# fan-out, `fast_kernel_shap_effects`'s "batch as memory allows") sizes its
# batches against the SHORTLIST width (`len(interaction_features)`,
# capped at 500 by default) - none of them know about this hidden
# `n_features / len(shortlist)` multiplier, which is exactly what
# `models/SVR.py`'s/`models/KNN.py`'s own 'baseline_shap' routes never pay:
# those two models refit directly on the shortlist (`train_x[interaction_
# features]`), so their `predict_fn` never rebuilds anything wider than the
# shortlist itself. On a whole-genome marker panel that multiplier can be
# 40-100x, silently turning a caller-side batch that was sized to be safe
# (e.g. `pair_chunk_size=2000` scored pairs x 50 background rows =
# 100,000 shortlist-width rows) into tens of gigabytes at the ACTUAL
# full-marker width - independently, per concurrently-running joblib
# worker - which is exactly the "excessive memory usage causing the
# Operating System to kill the worker" `baseline_shap_interactions()`'s own
# fan-out-failure NOTE reports (`SIGKILL(-9)`).
#
# Fixed at the source, transparent to every caller (`shap.KernelExplainer`,
# `fast_kernel_shap_effects`, `shapiq_marker_effects`,
# `baseline_shap_interactions` alike): `_masked_predict()` now bounds the
# row count it actually hands `predictor.predict()` in ONE call by THIS
# fit's own `n_features` (the full width it must reconstruct), re-deriving
# a safe sub-batch size from a fixed element budget rather than trusting
# whatever the caller chunked for the much narrower shortlist. Numerically
# identical to before (same `predictor.predict()` values, same order),
# just computed in bounded-size sub-batches instead of one giant call -
# never changes which rows are explained or how they are reduced.
_RKHS_MASKED_PREDICT_MAX_ELEMENTS = 20_000_000  # ~160MB per float64 sub-batch


def _rkhs_masked_predict_batched(predictor, baseline, shortlist_idx, X_short):
    """Memory-bounded stand-in for ``predictor.predict(X_full)`` where
    ``X_full`` is `X_short` (shortlist-width) expanded back to `baseline`'s
    full marker width. Splits `X_short` into sub-batches so that no single
    call to `predictor.predict()` ever materialises more than
    `_RKHS_MASKED_PREDICT_MAX_ELEMENTS` array elements, regardless of how
    many rows the caller requested at once - see the module-level comment
    above for why this is necessary specifically for RKHS's masking-trick
    reconstruction. Returns a 1-D `np.ndarray`, length `X_short.shape[0]`,
    identical to concatenating `predictor.predict()` over those same rows
    in one call."""
    X_short = np.atleast_2d(np.asarray(X_short, dtype=float))
    n_rows = X_short.shape[0]
    n_features_full = baseline.shape[0]
    max_rows = max(1, _RKHS_MASKED_PREDICT_MAX_ELEMENTS // max(1, n_features_full))
    if n_rows <= max_rows:
        X_full = np.tile(baseline, (n_rows, 1))
        X_full[:, shortlist_idx] = X_short
        return np.asarray(predictor.predict(X_full), dtype=float).ravel()
    out = np.empty(n_rows, dtype=float)
    for start in range(0, n_rows, max_rows):
        end = min(start + max_rows, n_rows)
        X_full = np.tile(baseline, (end - start, 1))
        X_full[:, shortlist_idx] = X_short[start:end]
        out[start:end] = np.asarray(predictor.predict(X_full), dtype=float).ravel()
    return out


# Perf fix (RKHS masked-predict routes - both the marker-EFFECT routes,
# `_apply_rkhs_shapiq_predictor_effect()`'s 'shap_predictor'/'shapiq_
# predictor', and the marker-INTERACTION routes below, 'baseline_shap'/
# 'friedman_h_predictor'/'kernel_hessian' - every one of them ultimately
# calls a masked `predict_fn` many times per task): `_rkhs_masked_predict_
# batched()` above only bounds the MEMORY the "rebuild the full
# n_features_full-width row for every query" masking trick costs - it
# does not remove the wasted FLOPs that same reconstruction spends on
# every call, which `kernel_effects.RKHSMaskedPredictor` removes
# algebraically instead (see that class's own module-level docstring in
# `models/kernel_effects.py` for the full derivation - verified
# numerically identical to this function's own tile-and-overwrite
# computation, not merely derived). `_build_rkhs_masked_predict()` below
# is the single place both masked-predictor call sites build their own
# `predict_fn` from, so the speedup applies to every one of them
# uniformly, with a defensive fallback to the (slower, but
# previously-proven) `_rkhs_masked_predict_batched()` route above if the
# fast path cannot be built for any reason - never a new failure mode for
# a task that used to work.
def _build_rkhs_masked_predict(predictor, baseline, shortlist_idx):
    """Return a fast `predict_fn(X_short)` callable implementing the
    "every non-shortlist column fixed at `baseline`" masking trick every
    RKHS masked-predictor effect/interaction route in this codebase uses -
    see the module-level comment directly above for what this replaces
    and why. NEVER RAISES: falls back to the slower, previously-used
    `_rkhs_masked_predict_batched()` route (via a lightweight
    `kernel_effects.RKHSPredictCore`, exactly as every call site built
    before this fix existed) with a printed NOTE if
    `kernel_effects.RKHSMaskedPredictor` cannot be constructed for any
    reason, so a bug in this new fast path can never turn a task that used
    to work into one that fails."""
    try:
        fast = kernel_effects.RKHSMaskedPredictor(predictor, baseline, shortlist_idx)
        return fast.predict
    except Exception as exc:
        print(f"[GP] NOTE: RKHSMaskedPredictor (fast masked-predict path) failed to build "
              f"({exc!r}) - falling back to the slower _rkhs_masked_predict_batched() route for "
              f"this task (numerically identical, just significantly slower on a large marker "
              f"panel).")
        _predict_core = kernel_effects.RKHSPredictCore(predictor)
        return lambda X_short: _rkhs_masked_predict_batched(_predict_core, baseline, shortlist_idx, X_short)


class _RKHSGateFailure(Exception):
    """Internal-only signal for `_apply_rkhs_shapiq_predictor_effect()`'s
    own V13.2 reconstruction-mismatch fallback - never escapes that
    function."""


def _gated_rkhs_predictor_cached(memo, result_RKHS, rkhs_params, train, test, predicted_test,
                                 weights=None, valid=None):
    """ver4-43 (E2): `_build_gated_rkhs_predictor()` once per RKHS model call.

    The RKHS marker-effect route and the RKHS marker-interaction route of ONE
    model call both rebuild the same V13.2-gated reconstruction (the log showed
    two gate-NOTE pairs per task). With a `memo` dict (one fresh `{}` per
    model call, created at the two call sites) the second caller reuses the
    first caller's result. A gate failure is memoised too and re-raised, so the
    fallback behaviour of each route is exactly what it was.

    `memo=None` calls straight through (the pre-ver4-43 behaviour). The memo is
    only honoured when the inputs are the very same objects (and equal
    hyperparameters) - otherwise the reconstruction is rebuilt and the memo
    refreshed, so a stale hit is impossible.

    Returns what `_build_gated_rkhs_predictor()` returns; raises what it raises.
    """
    if memo is None:
        return _build_gated_rkhs_predictor(
            result_RKHS, rkhs_params, train, test, predicted_test, weights=weights, valid=valid)
    key = (id(result_RKHS), id(train), id(test), id(valid), id(predicted_test), id(weights),
           repr(list(rkhs_params)))
    entry = memo.get('gated')
    if entry is None or entry[0] != key:
        try:
            outcome = ('ok', _build_gated_rkhs_predictor(
                result_RKHS, rkhs_params, train, test, predicted_test, weights=weights, valid=valid))
        except Exception as exc:      # includes _RKHSGateFailure - memoised, re-raised below
            outcome = ('fail', exc)
        entry = (key, outcome)
        memo['gated'] = entry
    kind, payload = entry[1]
    if kind == 'fail':
        raise payload
    return payload


def _apply_rkhs_shapiq_predictor_effect(r_fn, result_RKHS, rkhs_params, train, valid, test,
                                         result_name, predicted_test, r_kwargs, n_jobs=1,
                                         predictor_memo=None):
    """RKHS (design blueprint R13): when the 'Marker effect method' field
    (hparam_specs.py's appended, append-only index 17) is 'shap_predictor'
    or 'shapiq_predictor', reconstruct this fit's own kernel regression as
    a pure-NumPy `models.kernel_effects.RKHSPredictor` (from `r_u`/`r_mu`/
    `r_d_scale` - RKHS.R's own already-completed main BGLR fit output, no
    extra BGLR call) and explain it - no extra model refits either way,
    only WHICH explainer engine differs between the two choices.
    'iml_shapley' (default for any config predating this field) returns
    `r_list_get(result_RKHS, 'r_effect')` unchanged - RKHS.R's own legacy
    iml::Shapley route (I11).

    Bug fix (this update - RKHS marker-effect/shapiq comparability):
    'shapiq_predictor' USED TO explain the reconstruction with
    `shap.KernelExplainer` - the classic `shap` package, never the
    `shapiq` package - despite its own name/help text claiming to use
    "shapiq/SHAP". That was a real gap: SVR/KNN/RF/GBDT/XGBoost/
    ExtraTrees's own 'shapiq' marker-effect choice genuinely calls
    `pipeline_utils.shapiq_marker_effects()` (real `shapiq.
    TabularExplainer`), a numerically different estimator of the same
    order-1 Shapley quantity, so an ensemble mixing RKHS with those models
    was never actually comparing "shapiq vs shapiq" the way selecting the
    same-named option everywhere implied. Fixed by SPLITTING the one
    option in two:
      - 'shap_predictor' (renamed, otherwise BYTE-IDENTICAL to the old
        'shapiq_predictor' - I11 under its new name): reconstruction +
        V13.2 gate + `shap.KernelExplainer`, exactly as before.
      - 'shapiq_predictor' (now the new default, per the new 'default'
        value below): reconstruction + V13.2 gate + the SAME real
        `pipeline_utils.shapiq_marker_effects()` every other model's own
        'shapiq' choice uses (including this update's own fix making its
        `approximator='auto'` default automatically escalate to shapiq's
        SPEX approximator once the marker shortlist grows past 64 markers
        - the same numerical-stability fix, and the same computational-
        time-minimizing default, every other model's 'shapiq' marker-
        effect route already benefits from). Never lets a `shapiq`
        failure fail this task either - falls back to the 'shap_predictor'
        route (not all the way to 'iml_shapley') with a printed NOTE,
        mirroring the exact "try shapiq, catch, fall back to the proven
        non-shapiq route" pattern every other model's own 'shapiq' choice
        already uses.
    An existing saved config/checkpoint whose HPARAMETERS['RKHS'][17] is
    still the string 'shapiq_predictor' now gets the NEW, genuinely-shapiq
    behaviour for that field the next time it runs - a deliberate,
    requested exception to this codebase's usual "a value's own meaning
    never changes under an existing config" discipline, because the old
    behaviour never matched what that value's own name/help text already
    promised. Anyone who needs the OLD exact numbers back can select
    'shap_predictor' explicitly.

    Blueprint V13.2 gate: before trusting the reconstruction for an
    explanation, its own test-set predictions must reproduce RKHS.R's
    REAL BGLR test predictions (`predicted_test`), row by row, within a
    tolerance built from THIS fit's own fitted variance components
    (`r_var_u`/`r_var_e`, BGLR's `varU`/`varE`) and each row's own GP
    "not explained by training" conditional variance
    (`models.kernel_effects.RKHSPredictor.conditional_variance()`) -
    `RKHS_RECON_MC_TOL_SIGMA` standard errors (module-level constant
    above this function, default 8; see its own comment for the full
    derivation and calibration evidence against both a correct
    reconstruction and a genuinely broken one). This replaces a flat
    `RKHS_RECON_MC_TOL_REL * sd(predicted_test)` relative tolerance (kept
    only as a fallback for the rare case `r_var_u`/`r_var_e` are missing)
    which itself had replaced an even older `1e-6 * sd()` floating-point-
    level bound copied from GBLUP's exact closed form - that bound is
    correct for GBLUP (fully deterministic) but was never achievable for
    RKHS's MCMC-fitted test predictions, and using it here was what
    forced EVERY 'shapiq_predictor' RKHS task back onto 'iml_shapley' in
    the first place; the flat-percentage fallback fixed the worst of that
    but still occasionally over- or under-rejects (see its own comment)
    since "normal" MC noise is task- and row-specific, not universal.

    Bug fix (this update - stop falling back to `iml_shapley`): a gate
    failure on the full-grid reconstruction USED TO immediately trigger a
    FRESH `r_fn(...)` call with `effect_method` forced back to
    'iml_shapley' - an expensive per-coalition BGLR refit this whole R13
    exact-predictor route exists specifically to avoid. `_build_gated_
    rkhs_predictor()` now retries a gate failure once against the
    training-block-only reconstruction FIRST (the estimator
    `RKHS_RECON_MC_TOL_SIGMA`'s own calibration was actually validated
    against - see that function's own docstring for why the full-grid
    route's gate failures are disproportionately false positives) before
    this call ever sees a `_RKHSGateFailure` at all; on the rare
    occasion neither reconstruction passes, this function now reports an
    all-zero effect vector (with a printed NOTE) instead of calling
    `iml_shapley` - never silently returning a possibly-wrong effect
    vector, and never a fresh BGLR refit either.

    NEVER RAISES - falls back to an all-zero effect vector (with a
    printed NOTE) if `_build_gated_rkhs_predictor()` itself ultimately
    raises `_RKHSGateFailure` (both reconstructions failed the gate) or
    any other exception, exactly like `_apply_gblup_closed_form_effect()`'s
    own last-resort behaviour.

    Because the shortlist-by-correlation prefilter (`max_shap_features`)
    narrows which markers `shap.KernelExplainer` actually perturbs, but
    `RKHSPredictor` cannot be REFIT on a marker subset the way KNN/SVR
    can (its kernel/`d_scale` are tied to the FULL marker set this exact
    BGLR fit used - refitting on fewer markers would silently be a
    DIFFERENT kernel, breaking the V13.2 exact-reproduction guarantee),
    the shortlist is explained via feature MASKING instead: every column
    OUTSIDE the shortlist is held fixed at its training-mean value for
    every perturbed coalition (an ordinary SHAP background convention
    for a feature that is never perturbed), while the shortlisted columns
    vary normally.

    Parameters
    ----------
    r_fn : this process's own bound RKHS R function object (from
        `_get_r_model_fn('RKHS', r_path)` or the main dispatch loop's own
        module-level `RKHS`). Bug fix (this update): no longer called by
        this function (there is no more `iml_shapley` fallback re-call to
        make with it) - kept as a parameter only so this function's own
        call site doesn't need to change; harmless to keep passing.
    result_RKHS : this call's raw R result (or the fan-out's own plain-
        dict equivalent).
    rkhs_params : this task's own (possibly padded) HPARAMETERS['RKHS']
        list, with 'all' -> test.shape[0] (Shapley_num, index 4) already
        resolved by the caller.
    train, valid, test : this task's own splits (`valid` is forwarded to
        `_build_gated_rkhs_predictor()`'s own full-grid reconstruction
        attempt).
    result_name : unused by this function directly (kept for call-site
        stability - see `r_fn`'s own note above).
    predicted_test : this call's own `r_list_get(result_RKHS,
        'r_y_predicted')` - the REAL BGLR test predictions the V13.2 gate
        checks the reconstruction against.
    r_kwargs : the SAME `K_precomputed`/`weights` kwargs dict the
        original call used - only its 'weights' key is still read
        (forwarded to `_build_gated_rkhs_predictor()`'s own gate
        tolerance/full-grid reconstruction).
    n_jobs : this run's own resolved compute-resources n_jobs, for
        `parallel_shap_values()`'s own multiprocess fan-out (only used by
        the 'shap_predictor' route; `shapiq_marker_effects()` has no
        n_jobs of its own, exactly like every other model's 'shapiq'
        marker-effect call) and by the 'baseline_shap' route's row fan-out.
    predictor_memo : dict or None, keyword-only, default None
        ver4-43 (E2): one fresh dict per RKHS model call, shared with
        `_apply_rkhs_predictor_interaction()` so the gated reconstruction is
        built once (see `_gated_rkhs_predictor_cached()`). `None` = rebuild.

    ver4-43 (R2): `rkhs_params[17] == 'baseline_shap'` is a NEW, opt-in effect
    method - `phi_q(x) = mean over REAL training background rows z of [f(x) -
    f(x^{q->z})]` for every shortlisted marker, computed through the exact
    deduplicated engine and the closed-form RBF evaluator
    (`baseline_shap_effects()`); background size is `rkhs_params[22]` (default
    50). Not numerically comparable with 'shap_predictor'/'shapiq_predictor'.
    A failure reports the usual all-zero vector with a NOTE.

    Returns
    -------
    A 1-row DataFrame, one column per marker.
    """
    effect_method = rkhs_params[17] if len(rkhs_params) > 17 else 'iml_shapley'
    if effect_method not in ('shap_predictor', 'shapiq_predictor', 'baseline_shap'):
        return r_list_get(result_RKHS, 'r_effect')
    get_effect = bool(rkhs_params[3]) if len(rkhs_params) > 3 else False
    marker_cols = list(train.columns[:-1])
    n_features = len(marker_cols)
    if not get_effect:
        return r_list_get(result_RKHS, 'r_effect')

    def _fallback_all_zero_effect(reason):
        # Bug fix (this update - stop falling back to `iml_shapley`): this
        # used to re-call `r_fn(...)` with `effect_method` forced to
        # 'iml_shapley' - a fresh, expensive per-coalition BGLR refit,
        # exactly what this R13 exact-predictor route exists to avoid.
        # `_build_gated_rkhs_predictor()` already retries a V13.2 gate
        # failure once against the training-block-only reconstruction
        # before raising `_RKHSGateFailure` at all (see that function's
        # own docstring), so reaching this point means NEITHER
        # reconstruction reproduced this fit's own real BGLR test
        # predictions closely enough - a genuinely rare case. Reporting
        # an all-zero effect vector here (with a printed NOTE) - rather
        # than a fresh BGLR refit - keeps this route's own "no extra BGLR
        # calls, ever" guarantee absolute; this model's own predictions
        # are UNAFFECTED either way, exactly like `_apply_gblup_closed_
        # form_effect()`'s own last-resort behaviour.
        print(f"[GP] NOTE: RKHS {effect_method} marker-effect route could not trust its own "
              f"reconstruction for this task ({reason}) - reporting an all-zero effect vector "
              f"instead of a possibly-wrong one or a fresh iml_shapley BGLR refit (this model's "
              f"own predictions are UNAFFECTED).")
        return pd.DataFrame([np.zeros(n_features)], columns=marker_cols)

    try:
        if test is None or test.shape[0] == 0:
            return pd.DataFrame([np.zeros(n_features)], columns=marker_cols)

        # This update: the reconstruction + V13.2 gate steps that used to
        # be inlined here are now shared with the new, non-surrogate
        # 'shapiq_predictor' marker-INTERACTION route
        # (_apply_rkhs_predictor_interaction(), below) via
        # _build_gated_rkhs_predictor() - see that function's own
        # docstring. Pure extraction: every check, tolerance formula and
        # error message below is byte-identical to before this update: it
        # simply now lives in one place both callers reach.
        #
        # Bug fix (V13.2 gate false-positive under BLUE-reliability
        # weighting): `r_kwargs` carries this fit's own 'weights' key (the
        # shared `w_fit`) whenever BLUE weighting is on. Forwarding it
        # here too, into the gate's own tolerance, is what actually fixes
        # the false-positive fallback this update addresses - see
        # kernel_effects.RKHSPredictor.conditional_variance()'s own
        # docstring for the full derivation.
        predictor, X_train_full = _gated_rkhs_predictor_cached(
            predictor_memo, result_RKHS, rkhs_params, train, test, predicted_test,
            weights=(r_kwargs or {}).get('weights'), valid=valid,
        )

        Shapley_num = rkhs_params[4]
        shapley_num_rows = test.shape[0] if Shapley_num == 'all' else max(0, min(int(Shapley_num), test.shape[0]))
        if shapley_num_rows <= 0:
            return pd.DataFrame([np.zeros(n_features)], columns=marker_cols)

        max_shap_features = rkhs_params[5]
        train_x = train.iloc[:, :-1]
        train_y = train.iloc[:, -1]
        if max_shap_features != 'all' and n_features > int(max_shap_features):
            correlations = train_x.corrwith(train_y).abs().fillna(0)
            top_features = list(correlations.sort_values(ascending=False).index[:int(max_shap_features)])
        else:
            top_features = marker_cols

        baseline = X_train_full.mean(axis=0)
        col_index = {name: i for i, name in enumerate(marker_cols)}
        shortlist_idx = np.array([col_index[name] for name in top_features], dtype=int)

        # Perf fix (see `_build_rkhs_masked_predict()`'s own module-level
        # comment, just above `_RKHSGateFailure`): builds a fast, exact
        # `predict_fn` for the "every non-shortlist column fixed at
        # `baseline`" masking trick this route needs (`RKHSPredictor`
        # cannot be refit on a marker subset), without the O(n_features_
        # full) per-call cost the previous `_rkhs_masked_predict_batched()`
        # route paid on every single call. Also still avoids the earlier
        # "un-serialize" `joblib` fan-out failure `kernel_effects.
        # RKHSPredictCore`'s own docstring documents (an `RKHSMaskedPredictor`
        # instance is smaller still - see that class's own docstring).
        _masked_predict = _build_rkhs_masked_predict(predictor, baseline, shortlist_idx)

        effect_train_x = train_x[top_features]
        effect_test_x = test.iloc[:, :-1][top_features]

        def _shap_predictor_effect():
            """The pre-existing 'shap_predictor' route (BYTE-IDENTICAL to
            what 'shapiq_predictor' itself did before this update, under
            its new name) - explains `_masked_predict` with `shap.
            KernelExplainer`. Also the fallback target if the real-shapiq
            route below raises."""
            background_size = min(50, effect_train_x.shape[0])
            background = safe_shap_kmeans_background(effect_train_x, background_size, log_prefix="[RKHS]")

            # Requirements.md item 1 (this update): "the current KernelSHAP
            # is very slow ... when the total number of markers is >
            # 10,000" - see models/SVR.py's own identical block and
            # pipeline_utils.fast_kernel_shap_effects()'s module-level
            # comment for the full algorithmic rationale (one shared
            # coalition sample + one WLS factorisation for the whole task,
            # reused across every explained row, instead of the per-row
            # shap.KernelExplainer fan-out below). NEVER lets a failure
            # here fail the task - falls back to the proven per-row route
            # with a printed NOTE, exactly like this file's own
            # 'shapiq_predictor' fallback above.
            if len(top_features) > FAST_KERNEL_SHAP_MARKER_THRESHOLD:
                _explain_rows = shap.sample(effect_test_x, shapley_num_rows)
                _occlusion_baseline = np.average(
                    background.data, axis=0, weights=background.weights,
                )
                # Perf fix ("extracting marker effects using shap_predictor
                # is computationally expensive [for RKHS] ... minimise the
                # computational cost when the total number of markers is >
                # 20,000"): RKHSPredictor's own kernel is ALWAYS Gaussian/
                # RBF, so `fast_kernel_shap_effects()`'s own generic,
                # black-box occlusion loop below can be replaced with the
                # closed-form, bit-identical (see `kernel_effects.
                # rbf_kernel_occlusion_effects()`'s own module-level
                # comment) `O(n_ref * n_features)` computation instead of
                # that loop's own `O(n_ref * n_features^2)` - a ~`n_features`
                # -fold reduction exactly where this matters most (whole-
                # genome, 20,000+ marker 'all'-marker effect runs). Tried
                # FIRST (fastest, exact); falls through to the proven
                # `fast_kernel_shap_effects()` route on ANY failure (e.g. an
                # unexpected `predictor` shape), which itself still falls
                # through to `shap.KernelExplainer` below - never a new
                # failure mode for a task that used to work.
                try:
                    _occ_predictor = kernel_effects.RKHSMaskedPredictor(predictor, baseline, shortlist_idx)
                    _coeffs_eff, _X_ref_short, _gamma = _occ_predictor.as_rbf_occlusion_kernel()
                    return kernel_effects.rbf_kernel_occlusion_effects(
                        _coeffs_eff, _X_ref_short, _explain_rows, _occlusion_baseline, _gamma,
                    )
                except Exception as exc:
                    print(f"[GP] NOTE: RKHS closed-form rbf_kernel_occlusion_effects failed "
                          f"({exc!r}) - falling back to fast_kernel_shap_effects for this task "
                          f"instead.")
                try:
                    return fast_kernel_shap_effects(
                        _masked_predict, background.data, _explain_rows,
                        nsamples=200, background_weights=background.weights,
                        log_prefix="[RKHS]",
                    )
                except Exception as exc:
                    print(f"[GP] NOTE: RKHS fast_kernel_shap_effects failed ({exc!r}) - falling "
                          f"back to the per-row shap.KernelExplainer route for this task instead.")

            explainer = shap.KernelExplainer(_masked_predict, background)
            # Bug fix - see models/SVR.py's own identical note (and
            # pipeline_utils.parallel_shap_values()'s "Bug fix" docstring):
            # shap.KernelExplainer.shap_values()'s own `l1_reg` default
            # (shap>=0.47.0, 'num_features(10)') silently zeroes every marker
            # beyond its own top ~10 per explained row, independent of
            # 'max_shap_features' above.
            #
            # Bug fix (this update - see models/SVR.py's own identical, more
            # detailed note): explicitly requesting `l1_reg=f'num_features(M)'`
            # is still incomplete on its own - shap's own LARS-based feature
            # selection under that setting can never resolve more markers
            # than the coalition-sample budget (hardcoded 200 here, not
            # 'max_shap_features' above) actually supports, so a large marker
            # shortlist still came back with most markers silently zeroed.
            # `l1_reg=False` skips that LARS step entirely and solves
            # directly across every one of THIS task's own shortlisted
            # markers (`top_features`), restoring the setting's actual
            # intent (and running faster, since the LARS step itself is
            # skipped).
            return abs(parallel_shap_values(
                explainer, shap.sample(effect_test_x, shapley_num_rows), 200,
                (n_jobs if n_jobs and n_jobs > 0 else 1),
                l1_reg=False,
            )).sum(axis=0)

        def _baseline_shap_effect():
            """ver4-43 (R2): see this function's docstring. Background rows are
            real TRAINING individuals (I7), drawn with a fixed seed (I9),
            restricted to the shortlist; non-shortlist columns stay at the
            training mean inside `_masked_predict`, as in every RKHS route."""
            _n_train_rows = X_train_full.shape[0]
            _bg_n = min(int(rkhs_params[22]) if len(rkhs_params) > 22 else 50, _n_train_rows)
            _bg_idx = np.random.default_rng(0).choice(_n_train_rows, size=_bg_n, replace=False)
            _X_bg = X_train_full[_bg_idx][:, shortlist_idx]
            _X_ex = np.asarray(shap.sample(effect_test_x, shapley_num_rows), dtype=float)
            _fe = None
            try:
                _fe = kernel_effects.rkhs_baseline_shap_fast_eval(
                    predictor, baseline, shortlist_idx, _X_bg)
            except Exception as exc:
                print(f"[GP] NOTE: RKHS closed-form baseline_shap evaluator could not be built "
                      f"({exc!r}) - baseline_shap evaluates the masked predictor directly for "
                      f"this task instead (slower, same values).")
            _scores = baseline_shap_effects(
                _masked_predict, _X_ex, _X_bg, kernel_fast_eval=_fe,
                n_jobs=(n_jobs if n_jobs and n_jobs > 0 else 1), log_prefix="[RKHS]",
            )
            print(f"[GP] NOTE: RKHS 'baseline_shap' marker effects computed for {len(_scores)} "
                  f"marker(s) from {_X_ex.shape[0]} explained row(s) and {_X_bg.shape[0]} real "
                  f"background row(s) (not numerically comparable with 'shap_predictor'/"
                  f"'shapiq_predictor' effects - different baseline).")
            return _scores

        if effect_method == 'baseline_shap':
            effect_scores = _baseline_shap_effect()
        elif effect_method == 'shapiq_predictor':
            # This update: the SAME real `shapiq` route (and the SAME
            # `approximator='auto'` -> automatic SPEX escalation past 64
            # markers, this update's own fix for the numerical-stability/
            # speed problem shared by every model's order-1 'shapiq'
            # marker-effect call) that SVR/KNN/RF/GBDT/XGBoost/ExtraTrees
            # already use - see pipeline_utils.shapiq_marker_effects()'s
            # own docstring. `effect_shapiq_background`/`effect_shapiq_
            # budget` are hparam_specs.py's appended RKHS fields 22/23,
            # read defensively (I5) so a config predating them keeps
            # reproducing the same 50/200 defaults 'shap_predictor' itself
            # always hardcoded. NEVER lets a `shapiq` failure fail this
            # task - falls back to 'shap_predictor' (not all the way to
            # 'iml_shapley') with a printed NOTE, exactly like every other
            # model's own 'shapiq' marker-effect route falls back to its
            # proven non-shapiq route.
            effect_shapiq_background = rkhs_params[22] if len(rkhs_params) > 22 else 50
            effect_shapiq_budget = rkhs_params[23] if len(rkhs_params) > 23 else 200
            try:
                effect_scores = shapiq_marker_effects(
                    _masked_predict, effect_train_x, shap.sample(effect_test_x, shapley_num_rows),
                    background_size=min(int(effect_shapiq_background), effect_train_x.shape[0]),
                    budget=int(effect_shapiq_budget),
                )
            except Exception as exc:
                print(f"[GP] NOTE: RKHS shapiq_predictor marker-effect computation failed ({exc!r}) "
                      f"- falling back to the 'shap_predictor' route (shap.KernelExplainer) for "
                      f"this task instead. ('shapiq' is an optional dependency - 'pip install "
                      f"shapiq' if not yet installed; this may also mean this shapiq version's API "
                      f"differs from what this codebase expects.)")
                effect_scores = _shap_predictor_effect()
        else:
            effect_scores = _shap_predictor_effect()

        effect_full = pd.Series(0.0, index=marker_cols)
        effect_full.loc[top_features] = effect_scores
        return pd.DataFrame(effect_full).T
    except _RKHSGateFailure as exc:
        return _fallback_all_zero_effect(str(exc))
    except Exception as exc:
        print(f"[GP] NOTE: RKHS {effect_method} marker-effect computation failed ({exc!r}) - "
              f"reporting an all-zero effect vector for this task instead of a possibly-wrong "
              f"one (this model's own predictions are UNAFFECTED).")
        return pd.DataFrame([np.zeros(n_features)], columns=marker_cols)


def _apply_rkhs_predictor_interaction(result_RKHS, rkhs_params, train, test, predicted_train,
                                       predicted_test, n_jobs=1, use_gpu=False, row_parallel=True,
                                       effect_values=None, weights=None, valid=None,
                                       predictor_memo=None):
    """RKHS's own non-surrogate marker-PAIR-interaction routes: when the
    'Marker interaction method' field (hparam_specs.py's index 18) is
    'shapiq_predictor', 'kernel_hessian', or 'friedman_h_predictor',
    reuse the SAME exact, V13.2-gated kernel reconstruction
    (`_build_gated_rkhs_predictor()` - `models.kernel_effects.
    RKHSPredictor`, already used and already trusted for RKHS's own
    marker-EFFECT 'shapiq_predictor' choice) and explain IT directly -
    never a surrogate.

    Why this exists (the gap this closes)
    --------------------------------------
    Every OTHER RKHS interaction route ('friedman_h'/'shapiq'/legacy
    'surrogate_shapiq', all still available and UNCHANGED - see
    `_r_model_surrogate_interaction()`'s own docstring) fits a lightweight
    `RandomForestRegressor` to THIS model's own predictions and explains
    that surrogate - never RKHS's own kernel directly, because a naive
    per-coalition explanation would need a full BGLR MCMC refit for every
    perturbed marker coalition (`_r_model_surrogate_interaction()`'s own
    module-level comment). The marker-EFFECT side of this file already
    solved the identical problem for RKHS's own 'shap_predictor'/
    'shapiq_predictor' effect routes (R13, ver4-25 Stage 7): reconstruct
    the ALREADY-COMPLETED main BGLR fit as a pure-NumPy predictor (one
    linear-algebra solve, no MCMC, no refits) and explain THAT, gated by
    a reconstruction-fidelity check so a bad reconstruction never
    silently reports wrong effects. This function applies that same idea
    to interactions instead of effects, and now offers THREE such
    non-surrogate routes rather than one - see the follow-up note below
    ("shapiq is unreliable") for why a second and third were added:

      - 'shapiq_predictor' (the original of the three): explains the
        reconstruction with `pipeline_utils.shapiq_pairwise_
        interactions()` - the SAME real Shapley-interaction-index
        computation `models/SVR.py`'s/`models/KNN.py`'s own 'shapiq'
        interaction choice already uses on THEIR own `.predict`.
      - 'kernel_hessian' (NEW - user feedback: "shapiq is unreliable and
        should not rely on this"): explains the reconstruction with
        `models.kernel_effects.rbf_kernel_hessian_interactions()` - a
        fully closed-form, DETERMINISTIC computation exploiting the fact
        that RKHSPredictor's own kernel is an exact, known,
        twice-differentiable Gaussian/RBF function of the marker vector.
        No sampling, no coalitions, no background/budget knobs, and no
        dependency on the optional `shapiq` package at all - see that
        function's own docstring for the full derivation. This is the
        route to use when shapiq's own coalition-sampling variance is a
        concern, or when `shapiq` is not installed / not trusted.
      - 'friedman_h_predictor' (NEW, same feedback): explains the
        reconstruction with `models.interaction_extraction.
        h_statistic_interactions()` - the SAME native Friedman's
        H-statistic computation `models/SVR.py`'s/`models/KNN.py`'s own
        DEFAULT 'friedman_h' choice already runs directly on their real
        fitted model's `.predict` (no RandomForest surrogate anywhere).
        Also does not use `shapiq`, but (unlike 'kernel_hessian') is a
        finite-difference estimate over a background sample rather than
        an exact analytic derivative, so it is offered as a second,
        independent cross-check rather than a replacement for
        'kernel_hessian'. NOTE (later user feedback, backed by this
        project's own experimental results): Friedman's H itself "was
        not reliable ... did not capture correct interactions" - see
        'baseline_shap' below for the reason (its own PDP-style grid
        assumes marker independence, which real linkage disequilibrium
        violates) and the replacement this motivated.
      - 'baseline_shap' (NEWEST - see the note above): explains the
        reconstruction with `models.interaction_extraction.
        baseline_shap_interactions()` - a background-referenced discrete
        second difference computed only at REAL hybrid rows (columns
        copied wholesale from either the row being explained or a real
        background individual, never a synthetic grid cell), so linkage
        disequilibrium is never violated the way a PDP-style grid
        (Friedman's H) can be, and cost is fixed (no coalition-sampling
        budget the way `shapiq` needs). Reduced with the SAME
        `abs_then_sum` convention `models/RF.py`'s own TreeSHAP
        interaction route uses, so its numbers sit on the SAME
        scale/units for direct, side-by-side comparison - see that
        function's own module-level docstring for the full derivation,
        cost analysis, and disclosed scope (a single-group-vs-rest
        reduction, not a full per-feature Shapley marginalisation).

    Falls back to the proven surrogate route (`_r_model_surrogate_
    interaction()`, forced to its own 'shapiq' choice for the best
    available comparability - never all the way to 'friedman_h' unless
    THAT call itself also needs its own internal fallback) whenever:
      - the V13.2 gate rejects the reconstruction for this task
        (`_RKHSGateFailure`);
      - the selected route itself raises for any other reason (e.g.
        `shapiq` missing, for the 'shapiq_predictor' choice only -
        'kernel_hessian'/'friedman_h_predictor' never touch `shapiq`).
    Every other 'Marker interaction method' choice ('friedman_h',
    'shapiq', the legacy 'surrogate_shapiq' spelling) is a complete,
    byte-identical passthrough to `_r_model_surrogate_interaction()` -
    this function changes NOTHING about their behaviour (I11).

    NEVER RAISES - any failure of an exact-predictor route falls back to
    the surrogate route above; a failure of THAT call is already handled
    (returns an empty DataFrame) by `_r_model_surrogate_interaction()`'s
    own outer `try`/`except`.

    Parameters
    ----------
    result_RKHS, rkhs_params, train, test, predicted_test : see
        `_build_gated_rkhs_predictor()`'s own identical parameters.
    predicted_train : this call's own `r_list_get(result_RKHS,
        'r_y_predicted_train')` - forwarded, unused by any of the three
        exact-predictor branches themselves, only to the surrogate
        fallback (which needs it to fit its own surrogate, exactly like
        every other call to `_r_model_surrogate_interaction()`).
    n_jobs, use_gpu, row_parallel : forwarded unchanged to the surrogate
        fallback - see `_r_model_surrogate_interaction()`'s own
        identical parameters. `use_gpu` is never consulted by any
        exact-predictor route (`RKHSPredictor` is plain NumPy - there is
        no surrogate fit or GPU dispatch to gate); `row_parallel` gates
        the SAME `n_jobs`-derived worker count 'shapiq_predictor'/
        'friedman_h_predictor' use for their own per-row fan-out
        ('kernel_hessian' is a single vectorised NumPy computation with
        nothing to fan out).
    effect_values : pandas.DataFrame or None, keyword-only, default None
        This update ("use marker effects to select markers for
        interaction search?" checkbox) - RKHS's own already-computed
        marker-EFFECT values (a 1-row DataFrame, one column per marker),
        forwarded unchanged to the surrogate fallback (which applies the
        SAME checkbox itself - see `_r_model_surrogate_interaction()`'s
        own identical parameter) and consulted directly by this
        function's own exact-predictor shortlist below.
    weights : (n_train,) array-like or None, keyword-only, default None
        Bug fix (V13.2 gate false-positive under BLUE-reliability
        weighting) - see `_build_gated_rkhs_predictor()`'s own identical
        parameter and `models.kernel_effects.RKHSPredictor.
        conditional_variance()`'s own docstring for the full derivation.
        Forwarded unchanged to `_build_gated_rkhs_predictor()` below;
        `None` (default) keeps this route byte-identical to before this
        fix (I11).
    valid : this task's own validation split, or `None`, keyword-only,
        default `None`. Bug fix (systematic, batch-wide V13.2 gate
        fallback) - see `_build_gated_rkhs_predictor()`'s own identical
        parameter for the full derivation. Forwarded unchanged; both
        callers of THIS function now always pass their own `valid`
        (`ModelTrialRunner.__call__`'s and the main dispatch loop's own
        RKHS branches both already have it in scope).

    predictor_memo : dict or None, keyword-only, default None
        ver4-43 (E2): the dict shared with `_apply_rkhs_shapiq_predictor_effect()`
        for this model call (see `_gated_rkhs_predictor_cached()`); `None`
        rebuilds the gated reconstruction as before.

    ver4-43 (R1): for the 'baseline_shap' method, when the run's
    `ACCELERATION['baseline_shap_engine']` is 'fast' the hybrid rows are scored
    by the closed-form RBF evaluator of the masked predictor
    (`kernel_effects.rkhs_baseline_shap_fast_eval`) and de-duplicated by
    background pattern; 'legacy' (the default) is the ver4-42 loop.

    Returns
    -------
    pandas.DataFrame - the SAME canonical shape every other interaction
    route in this codebase returns (`models.interaction_extraction.
    canonicalize_shapiq_table()`'s own two-shape table when orders above
    2 were requested; the plain `['marker1', 'marker2', 'value']` shape
    otherwise - 'kernel_hessian'/'friedman_h_predictor' only ever produce
    the plain order-2 shape, matching Friedman's H-statistic's own
    order-2-only scope everywhere else in this codebase).
    """
    _interaction_method = rkhs_params[18] if len(rkhs_params) > 18 else 'friedman_h'
    _EXACT_PREDICTOR_METHODS = (
        'shapiq_predictor', 'kernel_hessian', 'friedman_h_predictor', 'baseline_shap',
    )
    if _interaction_method not in _EXACT_PREDICTOR_METHODS:
        # Every pre-existing choice - byte-identical passthrough (I11).
        return _r_model_surrogate_interaction(
            'RKHS', train, predicted_train, rkhs_params, n_jobs, use_gpu=use_gpu, test=test,
            row_parallel=row_parallel, effect_values=effect_values,
        )

    def _fallback_surrogate(reason):
        print(f"[GP] NOTE: RKHS '{_interaction_method}' interaction route fell back to the "
              f"surrogate 'shapiq' route for this task ({reason}).")
        fallback_params = list(rkhs_params)
        fallback_params[18] = 'shapiq'
        return _r_model_surrogate_interaction(
            'RKHS', train, predicted_train, fallback_params, n_jobs, use_gpu=use_gpu, test=test,
            row_parallel=row_parallel, effect_values=effect_values,
        )

    # Field indices below mirror _r_model_interaction_fields()'s own
    # RKHS layout (base_count=10) - get_interaction=10, max_interaction_
    # features=11, interaction_background=12, interaction_top=13 - plus
    # the SAME shapiq-only fields (19/20/21) 'shapiq'/'surrogate_shapiq'
    # already read in _r_model_surrogate_interaction() above, reused
    # here unchanged (only by the 'shapiq_predictor' branch) rather than
    # adding a parallel set of fields for a route that answers the exact
    # same three questions (which orders, which index, what budget).
    # 'kernel_hessian'/'friedman_h_predictor' do not use any of the
    # three (see each branch below for what they use instead).
    get_interaction = rkhs_params[10] if len(rkhs_params) > 10 else False
    if not get_interaction:
        return pd.DataFrame()
    if test is None or test.shape[0] == 0:
        return pd.DataFrame()

    try:
        predictor, X_train_full = _gated_rkhs_predictor_cached(
            predictor_memo, result_RKHS, rkhs_params, train, test, predicted_test,
            weights=weights, valid=valid,
        )
    except _RKHSGateFailure as exc:
        return _fallback_surrogate(str(exc))
    except Exception as exc:
        return _fallback_surrogate(f"reconstruction failed ({exc!r})")

    try:
        marker_cols = list(train.columns[:-1])
        train_x = train.iloc[:, :-1]
        train_y = train.iloc[:, -1]
        test_x = test.iloc[:, :-1]

        max_interaction_features = rkhs_params[11] if len(rkhs_params) > 11 else 500
        interaction_background = rkhs_params[12] if len(rkhs_params) > 12 else 100
        interaction_top = rkhs_params[13] if len(rkhs_params) > 13 else 'all'
        # Update (perf): the SAME 'Marker interaction screen' fields
        # _r_model_interaction_fields() already exposes at this identical
        # base_count=10 offset (14/15 = base_count+4/+5) for RKHS's
        # surrogate routes - read defensively (I5), never a new
        # hyperparameter, so an older/shorter rkhs_params still resolves
        # to 'off' (screen=None below) exactly as before (I11).
        interaction_screen = rkhs_params[14] if len(rkhs_params) > 14 else 'off'
        interaction_screen_top = rkhs_params[15] if len(rkhs_params) > 15 else 2.0

        # This update ("use marker effects to select markers for
        # interaction search?" checkbox) - see this function's own
        # 'effect_values' parameter docstring above and models/RF.py's
        # identical checkbox for the full rationale. RKHS's own effect
        # cap (field 5, "Max markers considered for Shapley scores")
        # applies to every effect method uniformly (see hparam_specs.py's
        # own note on that field) whenever effect extraction (field 3) is
        # on, exactly like this file's other RKHS branch
        # (_r_model_surrogate_interaction() above).
        _rkhs_get_effect = bool(rkhs_params[3]) if len(rkhs_params) > 3 else False
        _rkhs_effect_cap = (
            (rkhs_params[5] if len(rkhs_params) > 5 else 500) if _rkhs_get_effect else None
        )
        _rkhs_use_effect_for_shortlist = rkhs_params[24] if len(rkhs_params) > 24 else True
        _rkhs_use_effect_shortlist = (
            effect_values is not None and should_use_effect_for_interaction_shortlist(
                _rkhs_use_effect_for_shortlist, max_interaction_features, _rkhs_effect_cap,
            )
        )
        if _rkhs_use_effect_shortlist:
            _interaction_ranking = effect_values.iloc[0].reindex(marker_cols).fillna(0).abs()
        else:
            _interaction_ranking = train_x.corrwith(train_y).abs().fillna(0)

        n_features = train_x.shape[1]
        if max_interaction_features != 'all' and n_features > int(max_interaction_features):
            interaction_features = list(
                _interaction_ranking.sort_values(ascending=False).index[:int(max_interaction_features)]
            )
        else:
            interaction_features = marker_cols
        if not interaction_features:
            return pd.DataFrame()

        # RKHSPredictor cannot be refit on a marker subset (its kernel/
        # d_scale are tied to the FULL marker set this exact BGLR fit
        # used - see _apply_rkhs_shapiq_predictor_effect()'s own
        # docstring for the identical constraint on the effect side), so
        # the interaction shortlist is explained via the SAME feature-
        # masking trick that route already uses: every column OUTSIDE
        # the shortlist is held fixed at its own training-mean value for
        # every perturbed coalition, while the shortlisted columns vary.
        baseline = X_train_full.mean(axis=0)
        col_index = {name: i for i, name in enumerate(marker_cols)}
        shortlist_idx = np.array([col_index[name] for name in interaction_features], dtype=int)

        # Perf fix (see `_build_rkhs_masked_predict()`'s own module-level
        # comment, just above `_RKHSGateFailure`): this is the exact route
        # that used to pay the full `n_features_full`-width masking cost
        # on EVERY one of `baseline_shap_interactions()`'s own `1 + M' + P`
        # `predict_fn` calls per explained row (and, before the OOM fix,
        # the exact route a crash log's own "[interaction_extraction]
        # NOTE: baseline_shap_interactions fan-out failed (... SIGKILL
        # ...)" message was reported from) - `RKHSMaskedPredictor` removes
        # that per-call cost algebraically rather than only bounding its
        # memory, and remains a small, `joblib`-fan-out-safe payload
        # (avoiding the identical "un-serialize" failure `kernel_effects.
        # RKHSPredictCore`'s own docstring already documents), so
        # `baseline_shap_interactions()`'s/`h_statistic_interactions()`'s
        # own `n_jobs`-way fan-out is unaffected by this change.
        _masked_predict = _build_rkhs_masked_predict(predictor, baseline, shortlist_idx)

        Shapley_num = rkhs_params[4]
        _explain_n = (
            test_x.shape[0] if Shapley_num == 'all' else max(0, min(int(Shapley_num), test_x.shape[0]))
        )
        if _explain_n <= 0:
            return pd.DataFrame()

        if _interaction_method == 'kernel_hessian':
            # NEW route (user feedback: "shapiq is unreliable and should
            # not rely on this") - a fully closed-form, deterministic
            # computation on RKHSPredictor's own exact Gaussian/RBF
            # kernel (`K = exp(-h*D/d_scale)`, so `gamma = h/d_scale`)
            # via kernel_effects.rbf_kernel_hessian_interactions() - see
            # that function's own docstring for the derivation. No
            # sampling of any kind, so this call is bit-identical on
            # every run against the SAME reconstruction/shortlist/
            # explained rows - nothing here can be "unreliable" in the
            # run-to-run-variance sense shapiq's own coalition sampling
            # can be.
            explain_rows = np.asarray(shap.sample(test_x[interaction_features], _explain_n), dtype=float)
            gamma = predictor.h / predictor.d_scale
            # Bug fix (systematic, batch-wide V13.2 gate fallback): uses
            # `predictor.alpha_ref`/`predictor.X_ref` - the SAME
            # (reference rows, coefficients) pair `predictor.predict()`
            # itself actually uses - rather than `predictor.alpha`/
            # `X_train_full` directly, which are the training-block-only
            # values ONLY (see `RKHSPredictor`'s own docstring). When this
            # task's own reconstruction used the full train+valid+test
            # grid (the normal case now - `_build_gated_rkhs_predictor()`
            # above), `X_ref`/`alpha_ref` are the grid-sized pair; when it
            # fell back to the training-block-only route (BGLR variance
            # components unavailable), they ARE `X_train_full`/`alpha`,
            # so this reduces to the previous call byte-for-byte (I11).
            interaction_sample = kernel_effects.rbf_kernel_hessian_interactions(
                predictor.alpha_ref, predictor.X_ref[:, shortlist_idx], explain_rows, gamma,
                interaction_features,
            )
            interaction_sample = top_select_by_order(interaction_sample, 'percentage', interaction_top)
            print(f"[GP] RKHS: analytic kernel-Hessian interaction extraction produced "
                  f"{interaction_sample.shape[0]} pair(s) from a {len(interaction_features)}-marker "
                  f"shortlist - a closed-form, deterministic computation on this fit's OWN kernel "
                  f"(V13.2-gated reconstruction); no sampling, no 'shapiq' dependency.")
            return interaction_sample

        if _interaction_method == 'friedman_h_predictor':
            # NEW route (same feedback) - Friedman's H-statistic computed
            # the SAME way models/SVR.py's/models/KNN.py's own DEFAULT
            # 'friedman_h' choice already computes it: directly on the
            # real fitted model's own `.predict` (here, the masked
            # RKHSPredictor), no RandomForest surrogate anywhere, and no
            # dependency on the optional 'shapiq' package either.
            interaction_pairs = [
                (a, b)
                for a in range(len(interaction_features)) for b in range(a + 1, len(interaction_features))
            ]
            _h_n_jobs = (n_jobs if n_jobs and n_jobs > 0 else 1) if row_parallel else 1
            interaction_sample = h_statistic_interactions(
                _masked_predict, test_x[interaction_features], list(interaction_features),
                pairs=interaction_pairs, n_background=min(int(interaction_background), test_x.shape[0]),
                n_jobs=_h_n_jobs,
            )
            interaction_sample = top_select_by_order(interaction_sample, 'percentage', interaction_top)
            print(f"[GP] RKHS: native Friedman's H-statistic interaction extraction produced "
                  f"{interaction_sample.shape[0]} pair(s) from a {len(interaction_features)}-marker "
                  f"shortlist - explains this fit's OWN kernel regression directly (V13.2-gated "
                  f"reconstruction), NOT a surrogate model, and does not use 'shapiq'.")
            return interaction_sample

        if _interaction_method == 'baseline_shap':
            # NEW route (user feedback, backed by this project's own
            # experimental results: "friedman-h was not reliable ... did
            # not capture correct interactions" and "shapiq ... too
            # computationally expensive") - see baseline_shap_
            # interactions()'s own module-level docstring (models/
            # interaction_extraction.py) for the full derivation. Applied
            # to the SAME masked RKHSPredictor.predict() every other
            # exact-predictor route here uses; background rows are REAL
            # training individuals (restricted to the shortlisted
            # columns, exactly like every other background here), so
            # linkage disequilibrium AMONG SHORTLISTED markers is
            # preserved exactly - markers OUTSIDE the shortlist remain
            # fixed at their training-mean baseline throughout (the SAME
            # pre-existing masking limitation every other exact-predictor
            # RKHS route already carries, since RKHSPredictor cannot be
            # refit on a marker subset - see _masked_predict's own
            # comment above).
            interaction_pairs = [
                (a, b)
                for a in range(len(interaction_features)) for b in range(a + 1, len(interaction_features))
            ]
            _h_n_jobs = (n_jobs if n_jobs and n_jobs > 0 else 1) if row_parallel else 1
            _bg_size = min(int(interaction_background), X_train_full.shape[0])
            _bg_rng = np.random.default_rng(0)
            _bg_row_idx = _bg_rng.choice(X_train_full.shape[0], size=_bg_size, replace=False)
            X_background = X_train_full[_bg_row_idx][:, shortlist_idx]
            explain_rows = np.asarray(shap.sample(test_x[interaction_features], _explain_n), dtype=float)
            # ver4-43 (R1, D1b): under ACCELERATION['baseline_shap_engine'] ==
            # 'fast' the hybrid rows go through the masked predictor's closed-form
            # RBF evaluator (O(n_grid) per hybrid instead of an O(n_grid * M')
            # GEMM) and are de-duplicated by background pattern inside
            # baseline_shap_interactions(). 'legacy' (an absent key) passes
            # nothing, i.e. the ver4-42 call (I11).
            _fast_eval = None
            if get_active_compute_resources().get('accel_baseline_shap_engine', 'legacy') == 'fast':
                try:
                    _fast_eval = kernel_effects.rkhs_baseline_shap_fast_eval(
                        predictor, baseline, shortlist_idx, X_background)
                except Exception as exc:
                    print(f"[GP] NOTE: RKHS closed-form baseline_shap evaluator could not be "
                          f"built ({exc!r}) - hybrid rows are scored by the masked predictor "
                          f"for this task instead (slower, same values).")
            interaction_sample = baseline_shap_interactions(
                _masked_predict, explain_rows, X_background, interaction_features,
                pairs=interaction_pairs, n_jobs=_h_n_jobs, kernel_fast_eval=_fast_eval,
                # Update (perf): reuses the SAME pre-existing 'Marker
                # interaction screen' fields read above (rkhs_params[14]/
                # [15]) - no new hyperparameter. 'off' (the field's own
                # default) resolves to screen=None, byte-identical to
                # this parameter not existing (I11) - a no-op for every
                # existing config, and a real lever (cuts the O(pairs)
                # cost directly) for anyone who turns screening on
                # because the unscreened cost was prohibitive on a large
                # marker shortlist.
                screen=(interaction_screen if interaction_screen != 'off' else None),
                screen_keep=(float(interaction_screen_top) / 100.0 if interaction_screen_top != 'all' else 1.0),
            )
            interaction_sample = top_select_by_order(interaction_sample, 'percentage', interaction_top)
            print(f"[GP] RKHS: baseline-referenced pairwise interaction extraction produced "
                  f"{interaction_sample.shape[0]} pair(s) from a {len(interaction_features)}-marker "
                  f"shortlist - directly comparable (same units/reduction) to RF's own TreeSHAP "
                  f"interaction table; explains this fit's OWN kernel regression directly (V13.2-"
                  f"gated reconstruction), NOT a surrogate model, and does not use 'shapiq'.")
            return interaction_sample

        # 'shapiq_predictor' - unchanged from before this update.
        _orders_raw = rkhs_params[19] if len(rkhs_params) > 19 else '2'
        _orders = tuple(int(_o.strip()) for _o in str(_orders_raw).split(',') if _o.strip()) or (2,)
        _shapiq_index = rkhs_params[20] if len(rkhs_params) > 20 else 'k-SII'
        _shapiq_budget = int(rkhs_params[21]) if len(rkhs_params) > 21 else 4096
        # ver4-28-style order-aware row shrink - the SAME helper SVR's/
        # KNN's/the surrogate route's own 'shapiq' branches already apply
        # (max(_orders) <= 2 leaves _explain_n completely unchanged - I11).
        _explain_n = resolve_shapiq_explain_row_count(max(_orders), _explain_n)
        _shapiq_n_jobs = (n_jobs if n_jobs and n_jobs > 0 else 1) if row_parallel else 1

        raw = shapiq_pairwise_interactions(
            _masked_predict, train_x[interaction_features],
            shap.sample(test_x[interaction_features], _explain_n), interaction_features,
            orders=_orders, background_size=min(int(interaction_background), train_x.shape[0]),
            budget=_shapiq_budget, index=_shapiq_index, n_jobs=_shapiq_n_jobs,
        )
        interaction_sample = canonicalize_shapiq_table(raw)
        interaction_sample = top_select_by_order(interaction_sample, 'percentage', interaction_top)
        # Requirements3.md item 4-style parity padding - see
        # models/SVR.py's/_r_model_surrogate_interaction()'s own
        # identical note: makes this route's row count directly
        # comparable to a dense SHAP-based route at the same threshold.
        interaction_sample = pad_shapiq_pairs_to_shap_parity(
            interaction_sample, interaction_features, interaction_top,
        )
        print(f"[GP] RKHS: exact-predictor shapiq interaction extraction produced "
              f"{interaction_sample.shape[0]} pair(s) from a {len(interaction_features)}-marker "
              f"shortlist - explains this fit's OWN kernel regression directly (V13.2-gated "
              f"reconstruction), NOT a surrogate model.")
        return interaction_sample
    except Exception as exc:
        return _fallback_surrogate(f"{_interaction_method} computation failed ({exc!r})")


# ver4-97: rough peak host memory of ONE Super-learner fit, in copies of its training matrix. The R
# models (rrBLUP / BayesB / RKHS) copy the matrix several times (rbind, data.frame, scale, the
# BGLR design); 6 is a deliberately round, slightly generous figure, not a measurement.
_SL_FIT_MEM_COPIES = 6


def _sl_parallel_width(compute_resources, n_fits, n_rows, n_markers, label):
    """ver4-97: how many Super-learner fit units may run at once.

    Bug fixed: the width used to be ``max(1, min(int(n_jobs or 1), n_fits))``. The default
    ``N_JOBS=-1`` ("every core") is NOT a worker count, so that expression gave ``max(1, -1)`` =
    1 and the whole Super-learner stage - the task's own out-of-fold fits and the pooled ones -
    ran on a single core. ``-1`` / ``None`` now resolve to the core count through
    ``nested_safe_n_jobs`` (1 inside a daemonic worker; an explicit ``0`` / ``1`` stays serial).

    A memory cap follows, because a parallel fit of the pooled stage trains on thousands of rows:
    the width is lowered until ``width * peak <= 70 %`` of the available memory (the 70 % rule of
    the bio-prior trial guard), with ``peak = _SL_FIT_MEM_COPIES * n_rows * n_markers * 8`` bytes.
    Unknown free memory leaves the width as it is. Returns ``(width, note)``; `note` is a log
    line or ''. Pure apart from reading the free memory; never raises."""
    if not compute_resources.get('hp_tune_parallel_trials'):
        return 1, ''
    n_jobs = compute_resources.get('n_jobs')
    if n_jobs is not None and int(n_jobs) in (0, 1):
        return 1, ''
    cores = nested_safe_n_jobs(compute_resources, n_jobs)
    width = max(1, min(int(cores), int(n_fits)))
    if width <= 1:
        return 1, ''
    note = ''
    try:
        peak = float(_SL_FIT_MEM_COPIES) * float(n_rows) * float(n_markers) * 8.0
        avail = available_memory_bytes()
        if avail is not None and peak > 0:
            cap = int(0.7 * avail // peak)
            if cap < width:
                note = (f"[GP] {label}: memory guard lowered the parallel width from {width} to "
                        f"{max(cap, 1)} (one fit on {int(n_rows)} rows x {int(n_markers)} markers "
                        f"needs about {peak / 1e9:.1f} GB; {avail / 1e9:.1f} GB is available, 70 % "
                        f"rule). Results do not depend on the width.")
                width = max(cap, 1)
    except (TypeError, ValueError, OverflowError):
        pass
    return width, note


def _sl_pooled_source(task_tag, cfg, scenario, asreml_on, blue_on, pheno_all, phenotype,
                      own_pop, own_ids, members, replicate, loader_factory):
    """ver4-94 (R3): the other populations' lines for the Super learner's pooled columns, or
    ``None`` (with a NOTE that says why) when the pooled row cannot be built for this task.

    `cfg` is the resolved SUPER_LEARNER dict; `pheno_all` the whole phenotype table (ID,
    population, phenotypes); `own_ids` the task's own train / valid / test IDs as sets of str;
    `members` the task's Step 9 members (``IterMember``); `replicate` the task's ``sample``;
    `loader_factory(marker_cols)` returns the ``genotype_loader`` of
    ``models.relatedness.build_population_source`` for the CSV or PLINK path. Why a module-level
    function: the rules below are the same for both genotype paths and are easier to read and
    test outside the 600-line task function. Never raises for a data reason: a failure becomes a
    NOTE and no pooled row, and the unchanged Super learner rows are written as usual (the pooled
    row is an add-on, never a precondition)."""
    prefix = f"[GP] {task_tag} | NOTE: SUPER_LEARNER pooled"
    if scenario != 'within':
        print(f"{prefix} - SCENARIO={scenario!r}: borrowing other populations applies to "
              f"'within' runs only - no (pooled) row.")
        return None
    if asreml_on:
        print(f"{prefix} - ASReml is on: the other populations' phenotypes are raw while this "
              f"task's are BLUE / BLUP (a different scale) - no (pooled) row.")
        return None
    if blue_on:
        print(f"{prefix} - BLUE weighting is on: the source lines have no fitting weights - no "
              f"(pooled) row.")
        return None
    n_pop = int(pheno_all['population'].astype(str).nunique())
    if n_pop < 2:
        print(f"{prefix} - the phenotype file has {n_pop} population(s): nothing to borrow - no "
              f"(pooled) row.")
        return None
    marker_members = [m for m in members if m.marker_channel]
    if len(marker_members) < 2:
        print(f"{prefix} - fewer than 2 marker-channel members - no (pooled) row.")
        return None
    marker_cols = []
    seen = set()
    for m in marker_members:
        for c in m.train.columns[:-1]:
            if c not in seen:
                seen.add(c)
                marker_cols.append(c)
    first = marker_members[0]
    tv = first.train if first.valid.shape[0] == 0 else pd.concat(
        [first.train, first.valid], ignore_index=True)
    y_tv_mean = float(tv.iloc[:, -1].mean())
    y_tv_sd = float(tv.iloc[:, -1].std(ddof=1)) if tv.shape[0] > 1 else float('nan')   # pool-fix
    common = [c for c in first.train.columns[:-1]
              if all(c in m.train.columns for m in marker_members)]
    X_tv = tv[common].to_numpy(dtype=float) if cfg['pop_select'] == 'related' else None
    seed = int(cfg['seed']) + _SL_POP_OFFSET + 7919 * max(int(replicate), 0)
    try:
        src, diag = build_population_source(
            pheno_all, phenotype, own_pop, own_ids, marker_cols, y_tv_mean, X_tv, cfg, seed,
            loader_factory(marker_cols), score_cols=common, y_tv_sd=y_tv_sd)
    except (ValueError, KeyError, OSError, PlinkFilesetError) as exc:
        print(f"{prefix} - building the source lines failed ({type(exc).__name__}: {exc}) - "
              f"no (pooled) row.")
        return None
    print(f"[GP] {task_tag} | SUPER_LEARNER pooled source: {diag['n_selected']} line(s) from "
          f"{diag['n_populations']} other population(s) (mode {diag['mode']!r}, centred="
          f"{bool(cfg['pop_center'])}, rescaled={bool(cfg.get('pop_scale', False))}); candidates {diag['n_candidates']}, missing phenotype "
          f"{diag['n_missing_pheno']}, own train/valid/test IDs dropped {diag['n_dropped_own']}, "
          f"duplicates {diag['n_duplicate']}, no genotype {diag['n_no_genotype']}.")
    if diag.get('note'):
        print(f"{prefix} - {diag['note']}")
    if diag.get('scale_note'):
        print(f"{prefix} - {diag['scale_note']}")
    elif diag.get('scale_factor'):
        print(f"{prefix} - source phenotypes rescaled to the own spread (sd {y_tv_sd:.4g}); "
              f"factor per population {{" + ", ".join(
                  f"{k}: {v:.3g}" for k, v in diag['scale_factor'].items()) + "}")
    # ver4-97: cost visibility. The pooled fits train on (own rows + every borrowed line); the
    # gate (ice_sim40) used about as many source lines as own lines, not 20x as many.
    _n_own_tv = tv.shape[0]
    if src.shape[0] > 3 * max(_n_own_tv, 1):
        print(f"{prefix} - NOTE: {src.shape[0]} borrowed lines is {src.shape[0] / max(_n_own_tv, 1):.0f}x "
              f"this task's {_n_own_tv} train+valid lines, so each pooled fit trains on about "
              f"{src.shape[0] / max(_n_own_tv, 1):.0f}x more rows than the member's own fits "
              f"({len(marker_members)} member(s) x {int(cfg['folds']) + 1} extra fits). The "
              f"simulation gate used a source about the size of the own lines. To bound the cost "
              f"set SUPER_LEARNER['pop_max_lines'] (e.g. {3 * _n_own_tv}) and, optionally, "
              f"SUPER_LEARNER['pop_select']='related' to keep the most related lines.")
    if diag.get('reason') or src.shape[0] < MIN_SOURCE_LINES:
        print(f"{prefix} - {diag.get('reason', 'too few source lines')} - no (pooled) row.")
        return None
    return src


def _GP_single(GENOTYPE_FILE_NAME, PHENOTYPE_FILE_NAME, MODEL, PHENOTYPE, RATIO, SAMPLE_NUM, HPARAMETERS, R_PATH, W_OPT, RESULT_NAME, HYPERPARAMETERS_OPT, SCENARIO, PARALLEL=None, LD_prune=None, RF_filter=None, GENOTYPE_FORMAT='csv', GENOTYPE_PLINK_PATH='plink2', OTHER_MODELS_MARKER_SOURCE='full_or_filtered', progress_callback=None, HP_TUNE=None, HP_TUNE_ENSEMBLE_MODE='per_method', MIN_DATA_POINTS=100, USE_GPU_SKLEARN=False, N_JOBS=-1, PLINK_THREADS=1, R_BLAS_THREADS=None, TORCH_DEVICE=None, CUDNN_BENCHMARK=True, USE_AMP=False, N_CPU_WORKERS=1, N_GPU_SLOTS=None, MODEL_DISPATCH_FILTER=None, GPU_SLOTS_PER_DEVICE=1, R_BLAS_FOLLOWS_N_JOBS=True, TORCH_NUM_THREADS=None, TORCH_DATALOADER_WORKERS=0, GPU_EVAL_BATCH=32, GPU_LD_R2=True, GPU_KERNEL_PRECOMPUTE=True, HP_TUNE_PARALLEL_TRIALS=True, HP_TUNE_BAYES_BATCH=True, HP_TUNE_BAYES_BATCH_MAX=8, HP_TUNE_BAYES_LIAR='max', HP_TUNE_PARALLEL_RESTARTS=True, MODEL_AVAILABILITY_STRICT=True,
       HP_TUNE_BAYES_DOMAIN_REDUCTION='auto', HP_TUNE_WARM_START=False,
       HP_TUNE_BAYES_MULTI_FIDELITY=False,
       HP_TUNE_FAST=None,   # ver4-40 R1: opt-in fast single-model tuning (None = legacy)
       HP_TUNE_SELECTION_MARGIN=0.02, HP_TUNE_VALID_REPEATS=1, HP_TUNE_SCOPE='per_task',
       W_OPT_ANALYTIC_SEED=False, W_OPT_VALIDATION_FLOOR=False, W_OPT_SIMPLEX_SEARCH=True,
       RESULT_COMPRESSION='gzip',
       ASREML=None, ASREML_STRICT=True, PHENOTYPE_FILE_MODE='analysed',
       PHENOTYPE_TRAIT_COLUMNS=None, POPULATION_SOURCE=None,
       # Requirements.md (between-prediction validation-set patch), items
       # 1-4. Every one of these is ONLY ever consulted when
       # SCENARIO=='between'; a 'within' run's behaviour is completely
       # unaffected regardless of what they're set to.
       #
       # BETWEEN_VALIDATION: None (the default) means "not specified" -
       # NOT the same as False - and reproduces this function's exact
       # pre-patch behaviour (I11): whether a 'between' run built
       # validation triples used to be inferred SOLELY from whether W_OPT
       # was set (see the pre-patch `elif SCENARIO == 'between': if W_OPT
       # is None: ... else: ...` branch this patch replaces). Every
       # existing caller/config - which never sets this new key at all -
       # keeps going through that exact legacy inference (see
       # `_between_validation_source`/`_between_has_validation` just below
       # GP()'s own docstring-adjacent setup code). Only the NEW GUI
       # (main_app.py's 'between'-scenario checkbox) ever passes an
       # explicit True/False here, at which point the decoupled behaviour
       # this patch actually adds takes over.
       BETWEEN_VALIDATION=None,
       # BETWEEN_VALIDATION_SOURCE: one of 'third_population' (default -
       # train, validation and test each an entire, distinct population -
       # this is what the legacy W_OPT-driven triples behaviour already
       # did), 'split_train' (validation individuals carved out of the
       # TRAINING population's own individuals) or 'split_test' (carved
       # out of the TEST population's own individuals instead). Only
       # consulted when BETWEEN_VALIDATION is truthy.
       BETWEEN_VALIDATION_SOURCE=None,
       # BETWEEN_RATIO: a list of plain floats (never tuples - unlike
       # RATIO, this only ever describes ONE split - what fraction of the
       # chosen population's individuals STAY in their original role
       # (train stays train / test stays test), the remainder becoming
       # the validation set). Required (non-empty) when
       # BETWEEN_VALIDATION_SOURCE is 'split_train' or 'split_test';
       # ignored otherwise.
       BETWEEN_RATIO=None,
       # EXCLUDED_POPULATION_COMBINATIONS: an iterable of 'X->Y' (or
       # 'X->Y->Z', for the 'third_population' source) strings - in the
       # exact same format `between_population_combinations()` produces -
       # removed from this run's own `comb` list before the `sample`
       # table is built, i.e. before ANY task for that combination is
       # ever enumerated (Requirements.md item 3: "The selected pairs
       # will be removed from the subsequent genomic prediction tasks.").
       # None/empty (the default) excludes nothing - every possible
       # combination is used, matching this function's pre-patch
       # behaviour exactly (I11).
       EXCLUDED_POPULATION_COMBINATIONS=None,
       # ver4-26, R1/R2 (design blueprint `EasiGP_ver4-26_Design_Blueprint.md`):
       # opt-in JOINT hyperparameter + ensemble-weight search. None (the
       # default, every existing caller/config) reproduces this function's
       # exact pre-ver4-26 behaviour bit for bit - HP_TUNE and W_OPT keep
       # working exactly as before, and none of the code below this point
       # is reachable unless main_app.py's own new Tab-4 checkbox sets
       # JOINT_W_OPT explicitly (I11). Mutually exclusive with HP_TUNE and
       # W_OPT at the GUI layer (main_app.py forces both to None whenever
       # the joint checkbox is on - see that module's `gather_config`);
       # this function itself also treats JOINT_W_OPT as authoritative
       # over HP_TUNE/W_OPT if a headless/hand-edited config somehow sets
       # more than one, logging a NOTE so the conflict is never silent
       # (see the resolution just below Step 8).
       #
       # JOINT_W_OPT: None (default) or the single-element list
       # ['Joint Bayesian optimisation'] - a list, not a bare string, to
       # keep the same shape as W_OPT (R1) so downstream label/prefix
       # matching (`'Joint Bayesian optimisation' in (JOINT_W_OPT or [])`)
       # reads identically to the existing W_OPT idiom used throughout
       # this file.
       JOINT_W_OPT=None,
       # HYPERPARAMETERS_JOINT_OPT: same shape/role as HYPERPARAMETERS_OPT
       # (a dict of per-field search-space settings keyed by the joint
       # panel's own field names - see main_app.py's
       # JOINT_HYPERPARAM_OPT_SPECS) - consumed only by
       # `_resolve_joint_tuned_params` below, and only when JOINT_W_OPT is
       # truthy.
       HYPERPARAMETERS_JOINT_OPT=None,
       # ver4-57 (blueprint R2): opt-in ITERATIVE ENSEMBLE - None (the default, every
       # existing caller/config) reproduces ver4-56 byte for byte (plus one log line); a dict
       # (see pipeline_utils._ITER_ENS_DEFAULTS for every key and default; `{}` = all
       # defaults) adds Step 11c: after every model's Step 9 fit, markers are ranked once from
       # the members' effects, the non-bio-prior members are refitted on smaller marker
       # subsets (explainability off), and one equal-weight row labelled 'Iterative ensemble'
       # is written when a round clearly beats the naive ensemble on the validation rows.
       # Compatible with W_OPT and JOINT_W_OPT; needs a validation split.
       ITERATIVE_ENSEMBLE=None,
       # Requirements.md (multi-option update), item 4: Spearman
       # correlation / Top-k Coincidence no longer depend on ASReml, so
       # their per-phenotype "higher is better"/"lower is better"
       # direction is now a TOP-LEVEL setting: `{phenotype: 'higher' |
       # 'lower'}` (missing phenotypes default to 'higher'). None (the
       # default) falls back to `ASREML['accuracy_metrics']['direction']`
       # when that older location is populated, so an existing config
       # keeps producing the same Top-k values (I11).
       METRIC_DIRECTIONS=None,
       # ver4-33 (design blueprint SS4): the PLINK / bio-prior-GAT
       # acceleration switches - a dict with any of 'plink_direct_read'
       # ('off'|'auto'|'on'), 'preprocess_memo', 'gat_dense_batching',
       # 'gat_batched_ig', 'gpu_device_round_robin' (bools), plus the
       # engine-only 'BIO_PRIOR_PREP_FAST' / 'BIO_PRIOR_LEAN_POOLS' /
       # 'GAT_IG_INTERNAL_BATCH'. None (the default, every pre-ver4-33
       # caller/config) resolves to the ver4-32 behaviour (I11); resolved
       # values are logged with the compute-resource block below.
       ACCELERATION=None,
       # ver4-56 (blueprint R1): record per-model/task/job resource usage as JSON lines
       # (log-only; results are byte-identical on or off, AC1.6). True = the legacy-
       # preserving default for a log-only feature (I11); the interval is the memory
       # sampling period in seconds.
       RESOURCE_TELEMETRY=True, RESOURCE_TELEMETRY_INTERVAL_S=0.5,
       # ver4-61 (blueprint R6): path of a pilot run's hyperparameter.csv[.gz]. Every task
       # whose (population, phenotype, ratio, model, algorithm) matches a 'searched' row
       # skips the search and runs ONE confirmatory fit at that row's tuned values
       # (tuning_source='preset:<sample>'). None (the default, every older config) = no
       # preset, behaviour unchanged (I11). An unreadable file stops the run before task 1.
       HP_TUNE_PRESET_FILE=None,
       # ver4-77 (blueprint R2): opt-in SUPER LEARNER - None (the default, every existing
       # caller/config) reproduces the previous version byte for byte (plus one log line); a
       # dict (see pipeline_utils._SL_DEFAULTS for every key and default; `{}` = all defaults)
       # adds Step 11d: every model's Step 9 fit is repeated out of fold over train+valid, one
       # weight vector is chosen by cross-validation over those out-of-fold columns, and (with
       # a validation split) every model is refitted once on train+valid; one row labelled
       # 'Super learner' is written. Its train/valid prediction columns are OUT-OF-FOLD. Works
       # with or without a validation split, and next to W_OPT, JOINT_W_OPT and
       # ITERATIVE_ENSEMBLE. Amends I7: test rows still never enter a fit, weight or selection.
       SUPER_LEARNER=None,
       # ver4-102 (P4): opt-in seed bagging of MLP / GAT fits - None (default) = off, an integer n
       # (2..20) or {'n': int, 'seed': int}; see models/seed_bag.py for what it covers and omits.
       SEED_BAG=None):

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
          f"{'the six large result files (Prediction_result_{train,valid,test}.csv, Marker_effect.csv, Interaction.csv, Attention.csv) will be gzip-compressed (.csv.gz); the four small ones (Metric.csv, Weight.csv, hyperparameter.csv, Basic_stats.csv) never are' if RESULT_COMPRESSION == 'gzip' else 'every result file will be plain, uncompressed CSV, matching earlier runs'}.")

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

    # ver4-102 (D01 / R2): model-class coverage guidance. Log-only, no key, no effect on any
    # result: a list with no sparse, smooth (additive / kernel) or nonlinear (tree / local)
    # member gets one NOTE (`model_registry.class_coverage_note`). Never fatal.
    try:
        from model_registry import class_coverage_note as _class_coverage_note
        _coverage_note = _class_coverage_note(MODEL)
        if _coverage_note:
            print(f"[GP] NOTE: {_coverage_note}")
    except Exception as _coverage_exc:   # guidance must never stop a run
        print(f"[GP] NOTE: model-class coverage check skipped ({_coverage_exc!r}).")

    # Update ID ver4-12, R13 (design blueprint SS2.13): 'asreml' is a
    # licensed, commercial R package (never pinned in environment_linux.yml
    # - I12) that requires per-machine activation, not merely an importable
    # Python package - so it is deliberately NOT folded into
    # model_registry.OPTIONAL_DEPENDENCY (which only maps MODEL names to
    # Python packages - see that module's own cross-reference comment).
    # This is Layer 2 of R13's three gating layers (Layer 1 is main_app.py's
    # own live status badge; Layer 3 is Preprocess/asreml_blup.py raising if
    # called anyway despite this check somehow being bypassed) - it runs
    # BEFORE task 0, exactly like the MODEL_AVAILABILITY_STRICT block just
    # above, so a missing/unlicensed installation on an HPC compute node
    # fails fast with an actionable message instead of mid-run inside rpy2.
    # ASREML=None (the default - stage off) skips this entirely: nothing
    # about a run that never enables ASReml should ever depend on whether
    # the package happens to be installed at all.
    if ASREML is not None:
        try:
            asreml_license.preflight_check({'ASREML': ASREML, 'R_PATH': R_PATH})
        except Exception as exc:
            if ASREML_STRICT:
                raise
            print(f"[GP] WARNING (ASREML_STRICT=False): {exc!r}\nProceeding anyway - the "
                  f"ASReml-R stage will fail with the same message the first time it is "
                  f"actually reached (Step 6b, per task).")
        else:
            print("[GP] ASReml-R preflight check: package is installed and licensed.")

        # Config-only sibling of the license preflight just above, same
        # Layer 2 / "before task 0" placement: `asreml_syntax.
        # validate_structure()` runs every check that needs nothing but
        # `ASREML` itself (no phenotype-file read required, unlike
        # `asreml_syntax.validate()`'s full column-existence checks,
        # which the GUI already ran at commit time - R7). This is what
        # catches a hand-edited/headless config's `output_level`
        # referencing a column that was never actually given to the
        # model (e.g. a population-defining column left in
        # `output_level` after being removed from `fixed`/`random`) -
        # previously only surfaced as an opaque ASReml-R "<column> is not
        # in the model, unable to predict using it" error, deep inside
        # rpy2, after task 1 had already spent real fitting time. Not a
        # duplicate of the GUI's own commit-time call: this one has to
        # exist for a config that never went through the GUI at all.
        _asreml_structural_errors = asreml_syntax.validate_structure(ASREML)
        if _asreml_structural_errors:
            _message = (
                "[GP] ASREML config is invalid:\n"
                + "\n".join(f"  - {e}" for e in _asreml_structural_errors)
            )
            if ASREML_STRICT:
                raise ValueError(_message)
            print(f"[GP] WARNING (ASREML_STRICT=False): {_message}\nProceeding anyway - the "
                  f"ASReml-R stage will fail with the same problem the first time it is "
                  f"actually reached (Step 6b, per task).")

        # ver4-19 - `legacy_masked_holdout` (default False - see
        # Requirements.md R2/the EasiGP_ASReml_BLUE_BLUP_Redesign
        # blueprint SS2.7) decides which of the TWO Step-6b engines this
        # run uses. `True` reproduces every pre-ver4-19 run bit-for-bit,
        # INCLUDING the masked/two-fit legality question
        # `validate_scenario_compatibility()` exists to police (Requirements.
        # md item 2 (ver4-14)'s own five-pattern system - see that
        # function's own docstring). `False` (the new default) uses the
        # unmasked engine instead (`Preprocess/asreml_blup.py::
        # fit_unmasked_adjustment()`), for which that whole question is
        # moot: nothing is masked, so there is no "held-out individual
        # collapses to a constant" failure mode to legislate against in
        # the first place (see this module's own Step 6b dispatch,
        # further below, and asreml_syntax.py's own "ver4-19" section
        # header for the full reasoning).
        _asreml_legacy_mode = bool(ASREML.get('legacy_masked_holdout', False))
        # Requirements.md item 2 (ver4-14) / ver4-20 follow-up - the SAME
        # "Layer 2, before task 0, config-only" placement as
        # validate_structure() just above, checking a DIFFERENT thing:
        # not whether ASREML is internally well-formed, but whether its
        # genotype_effect/use_grm/masking combination is even LEGAL for
        # this run's own SCENARIO - see asreml_syntax.
        # validate_scenario_compatibility()'s own docstring for the full
        # reasoning (in short: several combinations produce a CONSTANT or
        # entirely inestimable validation/test result rather than a
        # clean error, which is why this needs its own explicit check
        # rather than being left to surface as a confusing downstream
        # symptom). Now called UNCONDITIONALLY for either engine - the
        # function itself branches on `legacy_masked_holdout` internally
        # (ver4-20). Catches a hand-written/headless config exactly the
        # same way validate_structure() does; the GUI's own commit-time
        # call to the same function is what stops this from ever being
        # reachable from main_app.py in the first place.
        _asreml_scenario_errors = asreml_syntax.validate_scenario_compatibility(ASREML, SCENARIO)
        if _asreml_scenario_errors:
            _message = (
                "[GP] ASREML config is not compatible with SCENARIO="
                f"{SCENARIO!r}:\n" + "\n".join(f"  - {e}" for e in _asreml_scenario_errors)
            )
            if ASREML_STRICT:
                raise ValueError(_message)
            print(f"[GP] WARNING (ASREML_STRICT=False): {_message}\nProceeding anyway - "
                  f"validation/test individuals may all receive the SAME (population-mean) "
                  f"adjusted phenotype, or an entirely inestimable one, per Requirements.md "
                  f"item 1's own root cause.")
        if _asreml_legacy_mode:
            pass
        else:
            _asreml_strategy = asreml_syntax.resolve_genotype_effect_strategy(ASREML)
            _asreml_mask_preflight = asreml_syntax.resolve_mask_valid_test(ASREML)
            print(f"[GP] ASREML['legacy_masked_holdout']=False (default) - "
                  f"using the NEW Step 6b engine: genotype_effect_strategy={_asreml_strategy!r} "
                  f"({describe_genotype_effect_strategy(_asreml_strategy)['label']}), fit ONCE "
                  f"per phenotype trait over every individual's own train+valid+test record - "
                  + ("validation/test phenotypes MASKED with NA before fitting "
                     "(ASREML['mask_valid_test']=True)."
                     if _asreml_mask_preflight else
                     "nothing masked, nothing held out (recommended default)."))
    # ver4-19: the legacy engine's `estimand_description()` (mixed
    # 'BLUE (training) + BLUP/GBLUP (validation/test)' wording) only
    # describes the MASKED, two-fit design - printing it unconditionally
    # for the new unmasked engine would be actively misleading (nothing
    # is masked or two-fit there at all), so this summary line is
    # resolved per-engine instead. The new engine's own resolved-strategy
    # line was already printed above, immediately after
    # `legacy_masked_holdout` was resolved.
    if ASREML is not None and not ASREML.get('legacy_masked_holdout', False):
        _asreml_estimand_for_log = (
            f"{describe_genotype_effect_strategy(asreml_syntax.resolve_genotype_effect_strategy(ASREML))['label']} "
            + ("(masked validation/test)"
               if asreml_syntax.resolve_mask_valid_test(ASREML)
               else "(unmasked - Requirements.md R2)")
        )
    elif ASREML is not None:
        _asreml_estimand_for_log = asreml_syntax.estimand_description(ASREML)
    else:
        _asreml_estimand_for_log = None
    print(f"[GP] ASREML resolved: {'ENABLED' if ASREML is not None else 'disabled (default)'}"
          + (f" - estimand {_asreml_estimand_for_log!r}, "
             f"genotype column {ASREML.get('genotype_col')!r}, "
             f"GRM {'on' if ASREML.get('use_grm', False) else 'off'}, "
             f"on_nonconvergence={ASREML.get('on_nonconvergence', 'fail')!r}"
             if ASREML is not None else "") + ".")

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
        # FEATURE (multi-fidelity screening for standard Bayesian
        # hyperparameter tuning) - forwarded exactly like every key above:
        # GP()'s own already-defaulted argument only, never a second,
        # independently-decided default. See models/hyperparameter_
        # tuning.py::search_bayesian()'s own docstring for the full
        # mechanism, and pipeline_utils.resolve_compute_resources()'s own
        # docstring for this key's default/rationale.
        'HP_TUNE_BAYES_MULTI_FIDELITY': HP_TUNE_BAYES_MULTI_FIDELITY,
        # ver4-40 R1: validated and defaulted by pipeline_utils.resolve_hp_tune_fast()
        # (None = legacy single-model tuning, byte-identical to ver4-39).
        'HP_TUNE_FAST': HP_TUNE_FAST,
        'HP_TUNE_SELECTION_MARGIN': HP_TUNE_SELECTION_MARGIN,
        'HP_TUNE_VALID_REPEATS': HP_TUNE_VALID_REPEATS,
        'HP_TUNE_SCOPE': HP_TUNE_SCOPE,
        'W_OPT_ANALYTIC_SEED': W_OPT_ANALYTIC_SEED,
        'W_OPT_VALIDATION_FLOOR': W_OPT_VALIDATION_FLOOR,
        'W_OPT_SIMPLEX_SEARCH': W_OPT_SIMPLEX_SEARCH,
        # ver4-33: resolved by resolve_compute_resources() into the accel_*
        # keys (None -> every switch at its ver4-32 value).
        'ACCELERATION': ACCELERATION,
        # ver4-56 (R1): resolved + logged by resolve_compute_resources().
        'RESOURCE_TELEMETRY': RESOURCE_TELEMETRY,
        'RESOURCE_TELEMETRY_INTERVAL_S': RESOURCE_TELEMETRY_INTERVAL_S,
    }
    _compute_resources = resolve_compute_resources(_compute_cfg)
    set_active_compute_resources(_compute_resources)
    # ver4-56 (R1): tell the telemetry recorder the concurrency layout this run resolved
    # (written into every model record; a no-op when telemetry is off).
    _rtele.set_context({
        'workers': _compute_resources.get('n_cpu_workers_task'),
        'n_jobs': _compute_resources.get('n_jobs'),
        'blas_threads': _compute_resources.get('r_blas_threads'),
        'torch_threads': _compute_resources.get('torch_num_threads'),
        'device': str(_compute_resources.get('device')),
        # Config switches that change the per-task pool cost (Steps 1-8); lets the
        # calibration keep runs with different pruning/filtering apart.
        'pool_sig': _calib_pool_signature(LD_prune, RF_filter, OTHER_MODELS_MARKER_SOURCE,
                                          GENOTYPE_FORMAT, ASREML, JOINT_W_OPT),
    })
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
    # ver4-33 (blueprint SS4): the acceleration switches, logged
    # unconditionally (I11) on their own line, like the lines above.
    print(f"[GP] Acceleration resolved: "
          f"plink_direct_read={_compute_resources['accel_plink_direct_read']!r}, "
          f"preprocess_memo={_compute_resources['accel_preprocess_memo']}, "
          f"gat_dense_batching={_compute_resources['accel_gat_dense_batching']}, "
          f"gat_batched_ig={_compute_resources['accel_gat_batched_ig']}, "
          f"gpu_device_round_robin={_compute_resources['accel_gpu_device_round_robin']}, "
          f"bio_prior_prep_fast={_compute_resources['accel_bio_prior_prep_fast']}, "
          f"bio_prior_lean_pools={_compute_resources['accel_bio_prior_lean_pools']}, "
          f"gat_ig_internal_batch={_compute_resources['accel_gat_ig_internal_batch']}"
          f"{' (ACCELERATION absent: default behaviour)' if ACCELERATION is None else ''}.")
    # ver4-35 (blueprint SS4): the new switches, on their own line (I11), so
    # the ver4-33 line above keeps its exact text for log scrapers.
    print(f"[GP] Acceleration resolved: "
          f"gat_factorised_input={_compute_resources['accel_gat_factorised_input']}, "
          f"gat_sync_free={_compute_resources['accel_gat_sync_free']!r}, "
          f"worker_thread_limits={_compute_resources['accel_worker_thread_limits']}, "
          f"gpu_slots={_compute_resources['accel_gpu_slots']!r}, "
          f"torch_device_policy={_compute_resources['accel_torch_device_policy']!r}, "
          f"async_model_lane={_compute_resources['accel_async_model_lane']}"
          f"(rng={_compute_resources['accel_async_lane_rng']!r}, "
          f"threads={_compute_resources['accel_async_lane_threads']!r}), "
          f"ld_cm_banded={_compute_resources['accel_ld_cm_banded']}, "
          f"bio_prior_cost_guard={_compute_resources['accel_bio_prior_cost_guard']!r}, "
          f"merge_shap_device={_compute_resources['accel_merge_shap_device']!r}, "
          f"shap_stream_reduce={_compute_resources['accel_shap_stream_reduce']}, "
          f"shared_bim_marker_info={_compute_resources['accel_shared_bim_marker_info']}"
          f"{' (ACCELERATION absent: default behaviour)' if ACCELERATION is None else ''}.")
    # ver4-43 (R1): the baseline_shap engine, on its own line (I11) so the two
    # lines above keep their exact text for log scrapers.
    print(f"[GP] Acceleration resolved: "
          f"baseline_shap_engine={_compute_resources['accel_baseline_shap_engine']!r}"
          f"{' (ACCELERATION absent: default behaviour)' if ACCELERATION is None else ''}.")
    # ver4-61 (R2): the trial-memory guard on its own line (I11) so the lines above keep their text.
    print(f"[GP] Acceleration resolved: "
          f"trial_memory_guard={_compute_resources['accel_trial_memory_guard']!r}"
          f"{'' if ACCELERATION and 'trial_memory_guard' in ACCELERATION else ' (default: auto)'}.")
    # ver4-46, R5: settings that make GAT hyperparameter tuning far slower than necessary,
    # logged unconditionally when a GAT model is tuned (nothing is changed by this check).
    for _tuning_warning in tuning_sanity_report(
            MODEL, HP_TUNE, _compute_resources, ACCELERATION is None,
            _compute_resources.get('hp_tune_fast')):
        print(f"[GP] TUNING CHECK: {_tuning_warning}")
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
          #f"(set 'always'/True respectively to restore ver4-5 behaviour), "
          f"hp_tune_selection_margin={_compute_resources['hp_tune_selection_margin']} "
          #f"(0.0 restores the ver4-5 bare '>=' floor comparison), "
          f"hp_tune_valid_repeats={_compute_resources['hp_tune_valid_repeats']}, "
          f"hp_tune_bayes_multi_fidelity={_compute_resources['hp_tune_bayes_multi_fidelity']}, "
          f"hp_tune_scope={_compute_resources['hp_tune_scope']!r}"
          + (" (tuned parameters are reused across replicates within this shard; "
             "results are shard-dependent - set HP_TUNE_SCOPE=per_task for "
             "shard-independent results)" if _compute_resources['hp_tune_scope'] == 'per_scenario'
             else "") + ".")
    print(f"[GP] Compute resources resolved: "
          f"w_opt_analytic_seed={_compute_resources['w_opt_analytic_seed']}, "
          f"w_opt_validation_floor={_compute_resources['w_opt_validation_floor']}, "
          f"w_opt_simplex_search={_compute_resources['w_opt_simplex_search']} "
          #f"(set every one of these False to restore ver4-5 weight-optimisation "
          #f"behaviour byte-for-byte, alongside HYPERPARAMETERS_OPT's own "
          #f"'dpt_ratio' objective).")# - see §8 of the ver4-6 Change Summary).")
          )
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
                  f"(an older config) - padded to {len(_spec)} with each new "
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
    if _compute_resources.get('accel_baseline_shap_engine') == 'fast':
        # ver4-43 (R1): rounding-level change for SVR/RKHS; KNN pair values change
        # wherever the legacy sklearn predictor carried distance-tie noise.
        _numerics_optional_lines.append(
            "baseline_shap_engine='fast'      (background-pattern dedup; KNN ties -> lower training index)")

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
        print("    Differences are at floating-point rounding level, not method level.")# To")
        #print("    reproduce a pre-ver4-4 run exactly, set each of these to false in the config.")
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

    def _save_ld_decay_plot_untimed(genotype_supplier, population_label, sample_label, ratio_label):
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

    def _save_ld_decay_plot(genotype_supplier, population_label, sample_label, ratio_label):
        """ver4-35, R1.a: ``_save_ld_decay_plot_untimed`` (above, unchanged)
        run inside the 'ld_decay' stage, so the diagnostic plot's wall time
        shows up in the per-task stage line instead of hiding in "other"."""
        with timed_stage('ld_decay'):
            return _save_ld_decay_plot_untimed(genotype_supplier, population_label,
                                               sample_label, ratio_label)

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
    ensure_rpy2_thread_safe_release()  # ver4-59: rpy2 releases R objects thread-safely
    
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
    rrBLUP = wide_safe_r_function(robjects.globalenv['rrBLUP'])  # ver4-59: chunked transfer of very wide marker frames
    r_source(os.path.join(_R_MODELS_DIR, 'GBLUP.R'))
    GBLUP = wide_safe_r_function(robjects.globalenv['GBLUP'])  # ver4-59: chunked transfer of very wide marker frames
    r_source(os.path.join(_R_MODELS_DIR, 'BayesB.R'))
    BayesB = wide_safe_r_function(robjects.globalenv['BayesB'])  # ver4-59: chunked transfer of very wide marker frames
    r_source(os.path.join(_R_MODELS_DIR, 'RKHS.R'))
    RKHS = wide_safe_r_function(robjects.globalenv['RKHS'])  # ver4-59: chunked transfer of very wide marker frames

    def _run_gblup_or_rkhs(r_fn, model_name, train, valid, test, params, result_name,
                            k_precomputed=None, weights=None):
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
        # ver4-25, R8 (design blueprint SS2, Stage 6).
        if weights is not None:
            r_kwargs['weights'] = weights

        get_effect = bool(params[get_effect_idx])
        # ver4-25, Stage 7 (R13) - see _run_gblup_or_rkhs_for_trial()'s
        # identical bypass (lockstep, RK-2) for the full rationale.
        _effect_method_idx = 13 if model_name == 'GBLUP' else 17
        _effect_method = params[_effect_method_idx] if len(params) > _effect_method_idx else 'iml_shapley'
        _raw_n_jobs = _compute_resources['n_jobs']
        if not get_effect or not _raw_n_jobs or _raw_n_jobs == 1 or _effect_method != 'iml_shapley':
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
                    k_precomputed, int(bound[0]), int(len(bound)), weights=weights,
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
    data_phenotype_original_raw = pd.read_csv(PHENOTYPE_FILE_NAME)

    # Update ID ver4-12, R2/R3 (design blueprint SS2.2/SS2.3): two new,
    # independently-defaulted config keys change how the phenotype file
    # is read and how 'population' is determined. Both default to EXACTLY
    # ver4-11's behaviour (invariant I11), and this whole block is
    # structured so that leaving both at their defaults takes the ORIGINAL
    # code path below verbatim - no derive_population_series() call, no
    # raw/analysed collapse - so a pre-ver4-12 config is byte-for-byte
    # unaffected (R1's own acceptance criterion 2 pins this).
    if PHENOTYPE_FILE_MODE not in ('analysed', 'raw'):
        raise ValueError(
            f"PHENOTYPE_FILE_MODE must be 'analysed' or 'raw', got {PHENOTYPE_FILE_MODE!r}."
        )
    _population_source_resolved = POPULATION_SOURCE or {'mode': 'column', 'column': 'population'}
    _is_legacy_phenotype_handling = (
        PHENOTYPE_FILE_MODE == 'analysed'
        and _population_source_resolved.get('mode', 'column') == 'column'
        and _population_source_resolved.get('column', 'population') == 'population'
    )

    # BUGFIX ("object 'Genotype' not found" - production failure, task
    # 1/25 of EasiGP_BLUEs_BLUPs_ASReml_0.5, population='Cluster_5',
    # phenotype='Maturity'): under PHENOTYPE_FILE_MODE='raw', the ASReml
    # genotype-column picker (main_app.py::_asreml_available_columns())
    # necessarily offers the phenotype file's OWN raw header text, read
    # BEFORE the positional ID-unification just below (Requirement 3) -
    # so it is entirely legitimate, and for a plot-level trial file whose
    # individual identifier is itself literally headed 'Genotype' the
    # NATURAL choice, for ASREML['genotype_col'] to name the exact same
    # physical column Requirement 3 is about to rename to 'ID'. Captured
    # here, before either branch below renames anything, so the alias
    # restored further down knows which column each unified name used to
    # be called.
    _phenotype_original_header_by_new_name = {}

    if _is_legacy_phenotype_handling:
        # Requirement 8 (pre-ver4-12): the phenotype file's first two
        # columns are always 'ID' and 'population' BY POSITION, regardless
        # of what header text the file actually uses for them - unify
        # internally so the rest of GP() can always rely on those exact
        # names. Every column AFTER these two is left completely
        # untouched - those are phenotype TRAIT names, which are
        # meaningful, user-chosen values (e.g. 'days2anthesis') that must
        # never be renamed.
        _phenotype_original_header_by_new_name = dict(
            zip(['ID', 'population'], data_phenotype_original_raw.columns[:2])
        )
        data_phenotype_original = unify_columns_by_position(
            data_phenotype_original_raw, ['ID', 'population'], 'phenotype file', first_n=2
        )
        data_phenotype_original['ID'] = data_phenotype_original['ID'].astype(str)
        # The true, un-collapsed phenotype table Step 6b (ASReml) would
        # read - identical to the analysed table here, since 'analysed'
        # mode is ALREADY one row per (ID x population) by definition.
        data_phenotype_raw_for_asreml = data_phenotype_original
    else:
        print(f"[GP] PHENOTYPE_FILE_MODE resolved: {PHENOTYPE_FILE_MODE!r}; POPULATION_SOURCE "
              f"resolved: {_population_source_resolved!r}.")
        if PHENOTYPE_FILE_MODE == 'raw':
            # Requirement 3: only the 'ID' column is positionally unified -
            # a raw, plot-level file's remaining columns are a mix of
            # design columns (Row, Range, Location, Year, ...) and trait
            # columns, distinguished only by PHENOTYPE_TRAIT_COLUMNS
            # (resolve_trait_columns() below), never by position.
            _phenotype_original_header_by_new_name = dict(
                zip(['ID'], data_phenotype_original_raw.columns[:1])
            )
            data_phenotype_original_raw = unify_columns_by_position(
                data_phenotype_original_raw, ['ID'], 'phenotype file', first_n=1
            )
        else:
            _phenotype_original_header_by_new_name = dict(
                zip(['ID', 'population'], data_phenotype_original_raw.columns[:2])
            )
            data_phenotype_original_raw = unify_columns_by_position(
                data_phenotype_original_raw, ['ID', 'population'], 'phenotype file', first_n=2
            )
        data_phenotype_original_raw['ID'] = data_phenotype_original_raw['ID'].astype(str)

        # Requirement 2: population is now a PURE FUNCTION of config plus
        # this one header/column read - never of a fit, a split, or task
        # order (invariant I9 - deterministic sharding). See
        # pipeline_utils.derive_population_series()'s own docstring for
        # the three supported modes.
        _population_series = derive_population_series(
            data_phenotype_original_raw, _population_source_resolved
        )
        data_phenotype_raw_for_asreml = data_phenotype_original_raw.copy()
        data_phenotype_raw_for_asreml['population'] = _population_series.values

        _trait_columns = resolve_trait_columns(
            {'PHENOTYPE_FILE_MODE': PHENOTYPE_FILE_MODE, 'PHENOTYPE_TRAIT_COLUMNS': PHENOTYPE_TRAIT_COLUMNS},
            PHENOTYPE_FILE_NAME,
        )
        # Requirement 8: collapse the (possibly plot-level, many-rows-per-
        # individual) raw table down to exactly the shape every existing
        # row-level guard downstream already assumes - one row per
        # (ID x population). Trait values here are a PLACEHOLDER (each
        # individual's own raw-record mean - see the function's own
        # docstring for why this is not a leakage channel) that Step 6b
        # OVERWRITES UNCONDITIONALLY when ASREML is enabled.
        data_phenotype_original = collapse_raw_to_analysed_level(
            data_phenotype_original_raw, 'ID', _population_series, _trait_columns
        )
        print(f"[GP] NOTE: PHENOTYPE_FILE_MODE={PHENOTYPE_FILE_MODE!r} - collapsed "
              f"{data_phenotype_original_raw.shape[0]} raw record(s) into "
              f"{data_phenotype_original.shape[0]} individual-level row(s) (one row per "
              f"ID x population). Trait value(s) are a PLACEHOLDER (each individual's own "
              f"raw-record mean) until Step 6b's ASReml-R stage overwrites them "
              f"unconditionally, if ASREML is enabled" +
              (" (it is NOT enabled for this run - the placeholder mean will reach every "
               "model as-is)." if ASREML is None else "."))

    # Restore the collision alias, if one is actually needed. A no-op
    # (nothing added) whenever ASREML is off, whenever genotype_col is
    # already a real column post-unification (the overwhelmingly common
    # case - e.g. a file whose individual-identifier column is already
    # called something other than 'ID'/'population'), or whenever
    # genotype_col names some unrelated column entirely. Left unhandled,
    # every task of a run in this exact configuration would fail
    # identically deep inside rpy2 with "object '<genotype_col>' not
    # found" - not merely the one task that happened to hit it first,
    # since the collision is structural (config + file layout), not
    # task-specific.
    if ASREML is not None:
        _asreml_genotype_col_for_alias = ASREML.get('genotype_col')
        if (_asreml_genotype_col_for_alias
                and _asreml_genotype_col_for_alias not in data_phenotype_raw_for_asreml.columns):
            _alias_source_new_name = next(
                (new for new, orig in _phenotype_original_header_by_new_name.items()
                 if str(orig) == _asreml_genotype_col_for_alias),
                None,
            )
            if _alias_source_new_name is not None:
                data_phenotype_raw_for_asreml = data_phenotype_raw_for_asreml.copy()
                data_phenotype_raw_for_asreml[_asreml_genotype_col_for_alias] = (
                    data_phenotype_raw_for_asreml[_alias_source_new_name]
                )
                print(f"[GP] NOTE: ASREML['genotype_col']={_asreml_genotype_col_for_alias!r} is "
                      f"the phenotype file's own original header for what Requirement 2/3's "
                      f"positional unification just renamed to {_alias_source_new_name!r} above "
                      f"- restoring {_asreml_genotype_col_for_alias!r} here as an identical "
                      f"alias column (never a rename - {_alias_source_new_name!r} is still what "
                      f"every other Step in GP() joins/splits on) so the committed ASReml-R "
                      f"syntax's own reference to {_asreml_genotype_col_for_alias!r} resolves "
                      f"correctly.")

    # ver4-18 BUGFIX ("Data has N observations but the residual model
    # implies M" - STILL reproduced after ver4-17, task 1/25 of
    # EasiGP_BLUEs_BLUPs_ASReml_0.5, population='Cluster_1', phenotype=
    # 'Glaucousness'): ver4-17 (just above) stopped RESTRICTING these raw
    # records to just this task's own population whenever a grid-
    # structured residual is configured - necessary, but not sufficient,
    # whenever the trial's own raw file itself has gaps in its physical
    # Range/Row grid (a plot never planted/harvested/recorded for ANY
    # trait at all) that no amount of population-restriction-lifting can
    # fix, since ver4-17's own fix already keeps every row the file has.
    # `asreml_syntax.pad_residual_grid()` is the single authority
    # (mirroring `residual_requires_full_grid()` just used above) for
    # explicitly completing that Cartesian product with NaN-filled
    # "filler" rows - see that function's own docstring for the full
    # mechanics and why NaN-filling every non-dimension column is safe
    # under this integration's own unconditional `na.action = na.method(
    # x = "include")`. Done ONCE here, for the whole run, rather than
    # once per task, since the padding depends only on the phenotype
    # file's own dimension columns, never on any one task's own
    # individuals - `Preprocess/asreml_blup.py::_fit_single_asreml_
    # model()` calls this same function again as an idempotent defensive
    # backstop, so this is never the ONLY place it is applied.
    if ASREML is not None and asreml_syntax.residual_requires_full_grid(ASREML):
        data_phenotype_raw_for_asreml = asreml_syntax.pad_residual_grid(
            data_phenotype_raw_for_asreml, ASREML
        )

    if GENOTYPE_FORMAT == 'csv':
        data_genotype_original = pd.read_csv(GENOTYPE_FILE_NAME)
        # Requirement 8: same positional unification as the phenotype file
        # above - 'ID'/'population' by position, every marker column after
        # them left exactly as-is (marker names are meaningful, user-chosen
        # identifiers, never renamed). Update ID ver4-12, R1: this
        # genotype-side 'population' column is retained ONLY long enough to
        # be dropped, unconditionally, immediately before the ID-only merge
        # in _process_one_task() below - it is no longer read for anything
        # else (see R2: POPULATION itself now always comes from the
        # phenotype side, for both genotype formats).
        data_genotype_original = unify_columns_by_position(
            data_genotype_original, ['ID', 'population'], 'genotype file', first_n=2
        )
        data_genotype_original['ID'] = data_genotype_original['ID'].astype(str)
        bim_marker_info = None
        fam_iid_set = None
    elif GENOTYPE_FORMAT == 'plink':
        validate_plink_fileset(GENOTYPE_FILE_NAME)
        data_genotype_original = None
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

    # ver4-19, Requirements.md R1 - every id that actually HAS marker data
    # available, regardless of task/population scope - the "has marker
    # data" half of `Preprocess/asreml_blup.py::
    # resolve_grm_genotype_universe()`'s own genotype-universe split (see
    # that function's own docstring). Computed once, here, for the whole
    # run - cheap (a column of already-loaded IDs for CSV; the already-
    # read .fam ID set for PLINK) and reused by every Step 6b call below.
    _asreml_marker_available_ids = (
        set(data_genotype_original['ID']) if GENOTYPE_FORMAT == 'csv' else set(fam_iid_set)
    )

    def _asreml_universe_marker_frames(universe_ids):
        """ver4-19, Requirements.md R1 - a `build_or_load_grm()`-shaped
        `(train_df, valid_df, test_df)` marker source covering EXACTLY
        `universe_ids`, drawn from the FULL genotype file/fileset - never
        from a task-scoped pool - so Step 6b's genomic relationship
        matrix can cover every genotype the fitted grid actually
        contains, not just the current task's own population subset.
        `universe_ids` is assumed already filtered to ids WITH marker
        data (`_asreml_marker_available_ids` above /
        `resolve_grm_genotype_universe()`'s own return value) - this
        function does not re-check that itself.
        """
        universe_ids = [str(u) for u in universe_ids]
        if not universe_ids:
            return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        if GENOTYPE_FORMAT == 'csv':
            _subset = data_genotype_original[
                data_genotype_original['ID'].isin(set(universe_ids))
            ].drop_duplicates(subset='ID')
            _subset = _subset.set_index('ID').loc[universe_ids].reset_index()
            _marker_only = _subset.drop(columns=['ID', 'population'], errors='ignore').copy()
        else:
            _universe_pheno = pd.DataFrame({'ID': universe_ids, 'population': 'all'})
            _geno_df = plink_to_genotype_df(
                GENOTYPE_FILE_NAME, _universe_pheno, keep_iid_list=universe_ids,
                plink_path=GENOTYPE_PLINK_PATH, work_dir=_new_plink_work_dir(),
            ).drop_duplicates(subset='ID')
            _geno_df = _geno_df.set_index('ID').loc[universe_ids].reset_index()
            _marker_only = _geno_df.drop(columns=['ID', 'population'], errors='ignore').copy()
        # `_build_grm()` (via `build_or_load_grm()`) only ever reads
        # `df.iloc[:, :-1]` - a dummy trailing column keeps this frame
        # shaped exactly like the (markers + phenotype) frames that
        # function normally receives, without needing a real response
        # value (which this genotype-only universe frame has no use
        # for at all).
        _marker_only['__asreml_grm_dummy_response__'] = 0.0
        return (_marker_only, pd.DataFrame(), pd.DataFrame())

    # Update ID ver4-12, R2 (design blueprint SS2.2, Appendix A finding A3):
    # POPULATION now comes from the phenotype file's own (possibly DERIVED)
    # population column on BOTH genotype formats - previously CSV read it
    # from the GENOTYPE file and PLINK read it from the phenotype file, two
    # independent authorities for the same fact. A genotype CSV's own
    # 'population' column (if it has one at all) is no longer authoritative
    # for anything beyond the pre-merge drop noted above.
    POPULATION = pd.unique(data_phenotype_original['population'])

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
        # ver4-35, R1.g2: with the run-wide memo directory set, every process
        # of a fan-out run points at ONE content-addressed copy of this table
        # (byte-identical to the per-run file) instead of each writing and
        # re-parsing its own. Any failure keeps the per-run file below.
        _shared_bim_path = None
        if _compute_resources.get('accel_shared_bim_marker_info', True):
            _shared_bim_path = write_shared_bim_marker_info(
                bim_marker_info, f"{GENOTYPE_FILE_NAME}.bim", os.environ.get('EASIGP_MEMO_DIR'))
        if _shared_bim_path is not None:
            bim_marker_info_path = _shared_bim_path
            print(f"[GP] NOTE: marker positions for the bio-prior model(s) are shared through "
                  f"'{_shared_bim_path}' (ACCELERATION['SHARED_BIM_MARKER_INFO']).")
        else:
            os.makedirs(os.path.dirname(bim_marker_info_path), exist_ok=True)
            bim_marker_info.to_csv(bim_marker_info_path, index=False, encoding='utf-8')

    if (type(PHENOTYPE) is not list) and (PHENOTYPE == 'all'):
        PHENOTYPE = list(data_phenotype_original.columns[2:])
    
    # Requirements.md (between-prediction validation-set patch), items 1-4.
    # Resolved ONCE, here - before `sample` is built AND before any task
    # runs - so every later reference to "does this 'between' run have a
    # validation split, and from where" (the per-task train/valid/test
    # split below, the Step 11 weighted-ensemble gate, the ASReml
    # genotype_by_population masking branch) reads the exact same
    # decision instead of re-deriving it.
    #
    # `BETWEEN_VALIDATION is None` ("not specified") reproduces this
    # function's exact pre-patch behaviour (I11): a 'between' run only
    # ever built validation triples because W_OPT was set, never via an
    # independent choice. Every existing caller/config - which never sets
    # BETWEEN_VALIDATION at all - keeps taking that legacy path
    # unchanged. Only the new GUI's explicit checkbox (main_app.py) sets
    # BETWEEN_VALIDATION to a real True/False, at which point it - not
    # W_OPT - decides whether a validation split exists at all.
    if SCENARIO == 'between':
        if BETWEEN_VALIDATION is None:
            _between_has_validation = W_OPT is not None
            _between_validation_source = 'third_population' if _between_has_validation else None
        else:
            _between_has_validation = bool(BETWEEN_VALIDATION)
            _between_validation_source = (BETWEEN_VALIDATION_SOURCE or 'third_population') if _between_has_validation else None
        print(f"[GP] SCENARIO='between' validation resolved: "
              f"{'none (train->test pairs only)' if not _between_has_validation else _between_validation_source!r}.")
    else:
        _between_has_validation = False
        _between_validation_source = None

    # Create the total number of combinations of prediction scenarios.
    # ver4-56 (R3): the table is built by pipeline_utils.build_sample_table(),
    # the verbatim move of the block that used to be inline here, so the
    # resource advisor and the pilot job enumerate exactly these tasks (I9).
    sample = build_sample_table(
        SCENARIO, POPULATION, PHENOTYPE, RATIO, SAMPLE_NUM,
        between_validation_source=_between_validation_source,
        between_ratio=BETWEEN_RATIO,
        excluded=EXCLUDED_POPULATION_COMBINATIONS,
    )

    record = pd.DataFrame()          #store performance metrics
    result_train = pd.DataFrame()    #store predicted phenotypes for train set
    result_valid = pd.DataFrame()    #store predicted phenotypes for validation set
    result_test = pd.DataFrame()     #store predicted phenotypes for test set
    effect = pd.DataFrame()          #store genomic marker effects
    interactions = pd.DataFrame()    #store marker interaction effects
    # ver4-25, Stage 9 (design blueprint R12): the k>=3 sibling of
    # 'interactions' above (k=2 only, schema UNCHANGED - I11) - a
    # SEPARATE accumulator, since a k>=3 row's own schema
    # (population/phenotype/model/ratio/sample/order/markers/value) does
    # not fit 'interactions'' own marker1/marker2 columns. Populated only
    # by a model whose 'Marker interaction method' is a shapiq-based
    # route AND whose 'Interaction orders to search for' includes an
    # order >= 3 - see _split_interaction_by_order() above.
    interactions_hok = pd.DataFrame()  #store k>=3 marker-SET interactions (R12)
    # ver4-57 (R4): per-round diagnostics of the opt-in Iterative ensemble (Step 11c). Stays
    # empty (and is never written - static_write_flags_for()'s 'iter_rounds' flag) unless
    # ITERATIVE_ENSEMBLE is set with write_rounds on. Wired exactly like `interactions_hok`:
    # init, resume load, nonlocal, snapshot, rollback, delta append, save.
    iter_rounds = pd.DataFrame()
    weight = pd.DataFrame()          #store weight values for weight optimisation
    attention_total = pd.DataFrame() #store attention values  GAT models
    hp_record = pd.DataFrame()       #store optimised hyperparameters found via HP_TUNE
    stats = pd.DataFrame()           #store basic per-scenario statistics (split sizes, marker count)
    # Update ID ver4-12, R10 (design blueprint SS2.10): two new accumulators
    # for Step 6b's ASReml-R adjusted-phenotype/variance-component output -
    # registered in checkpoint_utils.RESULT_FILE_NAMES under a subfolder
    # ('ASReml/ASReml_phenotype.csv[.gz]', 'ASReml/ASReml_variance_
    # components.csv'), so they flow through save/append/load/clear exactly
    # like the ten pre-existing accumulators above (see that module's own
    # updated docstrings). Both stay empty (and are never written at all -
    # static_write_flags_for()'s own asreml_enabled flag, below) whenever
    # ASREML is None - a run with the stage off produces no ASReml/ folder
    # at all, matching the rollback plan's "config-only switch" (SS8).
    asreml_pheno = pd.DataFrame()    #store per-individual converged BLUE/BLUP values
    asreml_varcomp = pd.DataFrame()  #store per-scenario/per-split ASReml-R variance components
    # ver4-25, R9-S0/R3 (design blueprint stage 3): two further
    # accumulators, registered in checkpoint_utils.RESULT_FILE_NAMES the
    # same way - 'asreml_fit_timing' (one row per real R fit call, purely
    # diagnostic) and 'asreml_accuracy_ctx' (one task's own accuracy-
    # metrics context, persisted the instant it is first computed - see
    # `_asreml_accuracy_ctx_resolver()`'s own updated comment below for
    # why this replaces the old process-local `_asreml_task_scope_
    # registry`).
    asreml_fit_timing = pd.DataFrame()
    asreml_accuracy_ctx = pd.DataFrame()
    # ver4-25, R8 Stage 6 completion: the 'full'-mode dense per-scenario
    # test-weight-matrix accumulator - see checkpoint_utils.
    # RESULT_FILE_NAMES['asreml_test_weights']'s own comment for the full
    # rationale. Stays empty (and is never written - static_write_flags_
    # for()'s own asreml_blue_weighting=='full' gate, below) for every
    # run that doesn't select 'full'-mode weighting, matching every other
    # ASReml accumulator's "config-only switch" behaviour (SS8).
    asreml_test_weights = pd.DataFrame()
    # ver4-7, R1 ("Per-Set GRM Generation for GBLUP"): the 3 GRM-matrix
    # diagnostic accumulators - one per split - registered in checkpoint_
    # utils.RESULT_FILE_NAMES the same way ('ASReml/GRM_{split}.csv[.gz]',
    # long upper-triangle form mirroring 'asreml_test_weights'). Stay
    # empty (and are never written - static_write_flags_for()'s own
    # asreml_per_set_grm gate, below) for every run that doesn't select
    # ASREML['per_set_grm']=True, matching every other ASReml
    # accumulator's "config-only switch" behaviour (SS8).
    # F2 (optimise_scan): nothing in this engine ever appends to these three
    # frames - the per-split matrices that would feed them were never wired
    # (the dead `_asreml_last_per_set_grm_matrices` slot and the unused
    # `nonlocal` were removed). They are kept ONLY because the file names,
    # checkpoint headers and snapshot/rollback/resume keys ('grm_train',
    # 'grm_valid', 'grm_test') are part of the result/checkpoint contract;
    # with ASREML['per_set_grm']=True the three GRM_{split} files are
    # therefore header-only. Removing the keys needs a checkpoint-format
    # migration, not a cleanup.
    asreml_grm_train = pd.DataFrame()
    asreml_grm_valid = pd.DataFrame()
    asreml_grm_test = pd.DataFrame()
    # ver4-25, R10 ("Stop phenotype file"): ASReml_phenotype.csv
    # duplicates data already present as `actual` in
    # Prediction_result_*.csv and is read by no plot/summary consumer -
    # resolved once here so both the write-flag computation (near
    # static_write_flags_for(), below) and the per-task accumulation
    # guard (Step 6b) agree on the SAME resolved value. Escape hatch:
    # ASREML['write_phenotype_file'] (absent -> False, I11).
    _asreml_write_phenotype_file = (
        bool(ASREML.get('write_phenotype_file', False)) if ASREML is not None else False
    )

    def _asreml_append_fit_timing(scenario_cols: dict, timing) -> None:
        """ver4-25, R9-S0: turns an `AsremlResult.timing` value - `None`
        (a cache hit - nothing new happened), a single dict (one fit),
        or a list of dicts (the legacy two-fit path) - into 0+ rows of
        the `asreml_fit_timing` accumulator, broadcasting `scenario_cols`
        (`population`/`phenotype`/`ratio`/`sample`) onto every one. A
        pure bookkeeping helper - never raises, never affects anything
        this run actually predicts with."""
        nonlocal asreml_fit_timing
        if not timing:
            return
        timing_list = timing if isinstance(timing, list) else [timing]
        if not timing_list:
            return
        rows = pd.DataFrame(timing_list)
        for _col, _value in scenario_cols.items():
            if isinstance(_value, (tuple, list)):
                # Convert it directly to a hashable string representation like "(1.2, 3.4)"
                rows[_col] = str(tuple(_value))
            else:
                # Keeps normal behavior for standard numbers or floats
                rows[_col] = _value


        rows = rows[list(scenario_cols.keys()) + [
            c for c in rows.columns if c not in scenario_cols
        ]]
        asreml_fit_timing = pd.concat([asreml_fit_timing, rows], axis=0, ignore_index=True)

    # Update ID ver4-12, R12 (design blueprint SS2.12 "Caching"): the
    # genomic relationship matrix depends only on the marker file and the
    # individual set - never on the split, the trait, or the replicate -
    # so it is built once per DISTINCT individual set and cached here, for
    # the lifetime of this one GP() call (i.e. once per worker PROCESS
    # under Parallel/intra-batch/intra-task fan-out - a process-local
    # cache, not a cross-process one, exactly as documented). Passed by
    # reference into every Step 6b call below; never read/written by
    # anything outside Preprocess/asreml_blup.build_or_load_grm().
    _asreml_grm_cache = {}

    # ver4-19, Requirements.md R1/R2/R3 - resolved ONCE per run (never per
    # task): which engine (`legacy_masked_holdout`), which of the three
    # new strategies, and which response unit this run uses. Response
    # unit in particular is a property of the WHOLE phenotype file's own
    # genotype/population layout (`resolve_response_unit()` - R3), not of
    # any one task/trait, so resolving it once here (rather than
    # redundantly inside the per-trait cache below) means every trait's
    # own log line and fit agree on it by construction.
    _asreml_legacy_mode = bool(ASREML.get('legacy_masked_holdout', False)) if ASREML is not None else False
    # ver4-20 follow-up (Requirements.md - simplified GUI's masking
    # toggle): resolved once per run, alongside every other new-engine,
    # run-level setting - see `_asreml_get_unmasked_result()`'s own
    # comment (near `_asreml_unmasked_cache`'s definition) for how this
    # changes that function's caching behaviour.
    _asreml_mask_valid_test = (
        asreml_syntax.resolve_mask_valid_test(ASREML)
        if ASREML is not None and not _asreml_legacy_mode else False
    )
    # Requirements.md (latest pass), item 1 - resolved once per run,
    # alongside `_asreml_mask_valid_test` above (single-authority
    # resolution - `asreml_syntax.resolve_use_entire_dataset()` also
    # applies the grid-residual override, so this one call is
    # sufficient; see that function's own docstring). `False` (the
    # default) restricts `_asreml_get_unmasked_result()` (see its own
    # updated docstring/comment below) to just the target population(s)
    # of whichever task asks for a given trait - matching what every
    # OTHER Step in `_process_one_task()` already does (Step 1's own
    # phenotype subsetting keeps only the target population's rows
    # before this Step ever runs). `True` is the explicit, opt-in
    # WIDENING: every population in the phenotype file is pooled into
    # ONE fit per trait, reused across every task that predicts it
    # (any population/ratio/replicate) whenever masking is also off.
    _asreml_use_entire_dataset = (
        asreml_syntax.resolve_use_entire_dataset(ASREML)
        if ASREML is not None and not _asreml_legacy_mode else False
    )
    # ver4-7, R1 ("Per-Set GRM Generation for GBLUP") - resolved once per
    # run, alongside every other new-engine, run-level setting. Mutually
    # exclusive with `_asreml_mask_valid_test` - `asreml_syntax.
    # validate_structure()`'s own preflight check (below) refuses both
    # being True together before any task runs, so this flag and masking
    # are never simultaneously honoured here.
    _asreml_per_set_grm = (
        asreml_syntax.resolve_per_set_grm(ASREML)
        if ASREML is not None and not _asreml_legacy_mode else False
    )
    _asreml_response_unit = None
    # Requirements.md item 4 fix - production failure ("Error : population
    # is not in the model, unable to predict using it", deep inside rpy2,
    # from fit_unmasked_adjustment()): whenever ASREML['response_unit'] is
    # 'auto', the GUI's own live syntax preview cannot re-check the real
    # phenotype file at commit time (main_app.py's own caption says so
    # explicitly), so a committed syntax built under 'auto' is always
    # rendered for the 'genotype' (unshared) shape. If THIS run's real
    # file resolves 'auto' to 'genotype_by_population' instead (just
    # above), the committed model has no `population` term at all, while
    # `fit_unmasked_adjustment()` separately adds `population` to
    # predict(classify=...) based on the (correctly) resolved response
    # unit - exactly the mismatch that error message reports.
    # `reconcile_committed_syntax_with_response_unit()` catches this here,
    # once per run, BEFORE any task/fit is attempted (the same "Layer 2,
    # before task 0" placement `asreml_syntax.validate_structure()` just
    # below already uses for the config-only version of this same class
    # of error) - never inside `_asreml_get_unmasked_result()` itself,
    # since a per-trait cache miss should never be where a run-level
    # config mismatch is first discovered.
    _asreml_resolved_committed_syntax = None
    if ASREML is not None and not _asreml_legacy_mode:
        # Requirements.md (latest pass), item 1 - `response_unit` is
        # deliberately resolved from the FULL, unrestricted
        # `data_phenotype_raw_for_asreml` here, regardless of
        # `_asreml_use_entire_dataset` - it is a run-level, structural
        # question ("does ANY genotype id in this trial appear in more
        # than one population, anywhere?"), not a per-task one, so it
        # must not itself depend on which population(s) one task
        # happens to restrict its OWN fit to (a restricted fit simply
        # ends up with a single-level `population` term when
        # response_unit resolves to 'genotype_by_population' but this
        # task's own target population set has only one member - a
        # degenerate, but not incorrect, special case of the same
        # formula every task shares).
        _asreml_response_unit = resolve_response_unit(
            ASREML, data_phenotype_raw_for_asreml, ASREML.get('genotype_col'), 'population',
        )
        _asreml_resolved_committed_syntax = asreml_syntax.reconcile_committed_syntax_with_response_unit(
            ASREML, ASREML.get('committed_syntax'), _asreml_response_unit, population_col='population',
        )

    # ver4-19, Requirements.md R1 SS1.4/SS4.3 - the genotype-universe
    # accounting from the MOST RECENT `_asreml_build_grm_for_universe()`
    # call (whichever engine/task last built or reused a GRM) - read by
    # the varcomp-accumulation code below so
    # `ASReml_variance_components.csv` carries these counts alongside the
    # run log's own diagnostic line, without threading them through every
    # call site as extra return values. `None` values mean "no GRM was
    # built for this call" (ASREML['use_grm'] is False, or this
    # particular fit never needed one - e.g. a 'blue_fixed'/'blup_iid'
    # main adjustment, or an 'iid' concurrent variance model).
    _asreml_last_grm_diagnostics = {'n_total': None, 'n_included': None, 'n_excluded': None}

    def _asreml_build_grm_for_universe(raw_records_for_universe, *,
                                        task_marker_source=None, task_ids=None,
                                        require_grm: bool = False,
                                        grm_blend_override: "Optional[float]" = None):
        """ver4-19, Requirements.md R1 - single authority for building
        Step 6b's genomic relationship matrix, shared by BOTH engines
        (the legacy one calls this per task; the new one calls this once
        per phenotype trait - see each call site below). Returns
        `(grm_inv, grm_ids)` - `(None, None)` when `ASREML['use_grm']` is
        False AND `require_grm` is also False.

        `ASREML['grm_source'] == 'task_pool'` is a deliberate, explicit
        opt-OUT of the R1 fix (kept for anyone who needs the narrower,
        cheaper pre-ver4-19 scope for a specific reason) - only honoured
        when the caller also supplies `task_marker_source`/`task_ids`
        (the legacy engine's own task-scoped pool; the new engine never
        passes these, since it has no per-task pool at all - there is no
        "task" in a once-per-trait fit). Every other case - including
        the default `grm_source == 'genotype_file'` - uses
        `resolve_grm_genotype_universe()` to cover every genotype id
        actually present in `raw_records_for_universe` that has marker
        data, not merely one task's own population subset.

        require_grm : ver4-25, R7 (design blueprint stage 4) - `True`
            forces a GRM to be built REGARDLESS of `ASREML['use_grm']` -
            the two-step refined-BLUE procedure's own step (i) always
            fits genotype random WITH a GRM (Q3's confirmed decision,
            Holland & Piepho 2024 Fig. 2), independent of whatever the
            run's own MAIN strategy configures (a `blue_fixed` strategy
            commonly has `use_grm=False`, since a plain BLUE needs no
            relationship matrix at all - step (i) still needs one).
        grm_blend_override : ver4-25, R7 - when given, used INSTEAD of
            `ASREML['grm_blend']` (R7's own, independently-configurable
            `ASREML['blue_refinement']['grm_blend']`, default 0.01,
            rather than forcing the main GRM's own blend value on a
            structurally different fit).
        """
        if not require_grm and not (ASREML is not None and ASREML.get('use_grm', False)):
            _asreml_last_grm_diagnostics['n_total'] = None
            _asreml_last_grm_diagnostics['n_included'] = None
            _asreml_last_grm_diagnostics['n_excluded'] = None
            return None, None
        _grm_blend = (
            ASREML.get('grm_blend', 0.01) if grm_blend_override is None else grm_blend_override
        )
        if (not require_grm and ASREML.get('grm_source', 'genotype_file') == 'task_pool'
                and task_marker_source is not None):
            print(f"[ASReml] grm_source='task_pool' - using the already-materialised "
                  f"(possibly LD-pruned) task-scoped marker pool instead of the Requirements.md "
                  f"R1 genotype-universe fix. This is a deliberate, narrower, cheaper opt-out; "
                  f"if validation/test individuals unexpectedly collapse to a constant adjusted "
                  f"value, switch back to the default grm_source='genotype_file'.")
            grm_inv = build_or_load_grm(
                task_marker_source, task_ids, _asreml_grm_cache, grm_blend=_grm_blend,
            )
            _asreml_last_grm_diagnostics['n_total'] = len(task_ids) if task_ids is not None else None
            _asreml_last_grm_diagnostics['n_included'] = len(task_ids) if task_ids is not None else None
            _asreml_last_grm_diagnostics['n_excluded'] = 0
            return grm_inv, list(task_ids)
        genotype_col = ASREML.get('genotype_col')
        universe_ids, n_total, excluded_ids = resolve_grm_genotype_universe(
            raw_records_for_universe, genotype_col, _asreml_marker_available_ids,
        )
        log_grm_universe_diagnostics(universe_ids, n_total, excluded_ids)
        _asreml_last_grm_diagnostics['n_total'] = n_total
        _asreml_last_grm_diagnostics['n_included'] = len(universe_ids)
        _asreml_last_grm_diagnostics['n_excluded'] = len(excluded_ids)
        marker_source = _asreml_universe_marker_frames(universe_ids)
        grm_inv = build_or_load_grm(
            marker_source, universe_ids, _asreml_grm_cache, grm_blend=_grm_blend,
        )
        return grm_inv, universe_ids

    # ver4-19, Requirements.md R2 SS2.4 - the new engine's fit is a
    # property of (genotype universe, trait, strategy, response_unit)
    # only - never of task/replicate/split (mirroring the
    # `_bio_prior_merge_cache` precedent - architecture doc SS13.4, and
    # the LD-decay "identical genotype split -> deduplicate the expensive
    # step" discipline - architecture doc SS6) - AS LONG AS nothing is
    # masked. Originally (ver4-19/ver4-20) keyed by trait ALONE, since
    # the fit unconditionally pooled every population; Requirements.md
    # (latest pass), item 1 adds the population SCOPE to that key by
    # default (see `_asreml_unmasked_cache`'s own updated comment just
    # below) - the FIRST task to touch a given (trait, population-scope)
    # pays for the real ASReml-R fit, every subsequent task sharing that
    # same key reuses it.
    #
    # ver4-20 follow-up (Requirements.md's masking toggle,
    # `_asreml_mask_valid_test` above): once masking is on, the result
    # ALSO depends on this task's own validation/test membership
    # (`mask_keys`), which differs per task - hoisting/caching across
    # tasks would silently reuse the WRONG mask. This function therefore
    # bypasses `_asreml_unmasked_cache` entirely whenever
    # `_asreml_mask_valid_test` is True, fitting fresh every task (the
    # same per-task cost the LEGACY engine has always paid) - only the
    # GRM build itself (via `_asreml_build_grm_for_universe()`, which has
    # its own independent cache keyed on the genotype universe alone)
    # stays cheap across tasks.
    #
    # Requirements.md (latest pass), item 1 - the cache key now also
    # carries the population SCOPE the fit actually used, alongside the
    # trait: a `frozenset` of this task's own target population(s) when
    # `_asreml_use_entire_dataset` is False (the DEFAULT - a task
    # predicting the same trait for a DIFFERENT population, or a
    # different 'X->Y' pair, is a genuinely different, independent fit -
    # see `_asreml_target_populations_for_task()` below for how that set
    # is derived), or `None` when it is True (the explicit, opt-in
    # WIDENING - every population pools into ONE fit per trait, so the
    # scope no longer matters and every task sharing that trait hits the
    # SAME cache entry - this is the "same results ... can be used for
    # the same target phenotype[s] to save computational time" the
    # requirement itself calls out, and it is exactly the pre-existing
    # ver4-19/ver4-20 cache key, unchanged). Either way, a restricted fit
    # is still hoisted/shared across every task that repeats the SAME
    # (trait, population-set) combination (e.g. every ratio/replicate of
    # one population's own within-scenario runs) - only the SCOPE of
    # what counts as "the same fit" has narrowed by default.
    _asreml_unmasked_cache = {}

    # ver4-25, R9-S4 (design blueprint stage 5): the most recently
    # converged variance-parameter estimates for a masked, per-task main
    # fit, keyed identically to `_asreml_unmasked_cache` above - used to
    # SEED (never fix) the NEXT task's own fit at the same (trait,
    # population-scope) when `ASREML['warm_start_per_task']=True` (default
    # `False`, I11). Only ever populated/consulted when masking is on
    # (`_asreml_mask_valid_test`) - an unmasked, hoisted fit is only ever
    # fit once per trait, so there is nothing to warm-start across.
    _asreml_warm_start_cache: dict = {}

    def _asreml_target_populations_for_task(population_value):
        """The population label(s) actually relevant to ONE task -
        used ONLY to restrict Step 6b's raw records (and, by extension,
        its GRM genotype universe - see `_asreml_build_grm_for_universe()`'s
        own docstring, "coordinated with GRM" per Requirements.md item 1)
        when `_asreml_use_entire_dataset` is False (the default). A
        single-element set for SCENARIO='within' (this task's own
        population); every population NAMED in the 'X->Y'/'X->Y->Z'
        string for 'between' - never the run's full POPULATION universe,
        only the one(s) this ONE task actually predicts between. Mirrors
        the identical restriction `_process_one_task()`'s own Step 1
        (phenotype subsetting) already applies to every OTHER Step, for
        the same reason."""
        _pop_raw = str(population_value)
        if SCENARIO == 'between':
            return frozenset(_pop_raw.split('->'))
        return frozenset({_pop_raw})

    # ver4-25, R7 (design blueprint stage 4): step (i)'s own converged
    # variance-parameter estimates, cached per (trait, population-scope)
    # exactly like `_asreml_unmasked_cache` above (step (i) has no
    # per-task dependency either - it never sees `mask_keys`, since Fig.
    # 2's method fits on every unmasked observation available).
    _asreml_blue_refinement_cache: dict = {}
    _asreml_blue_refinement_cfg = (
        (ASREML.get('blue_refinement') or {}) if ASREML is not None and not _asreml_legacy_mode else {}
    )
    _asreml_blue_refinement_enabled = bool(_asreml_blue_refinement_cfg.get('enabled', False))

    def _asreml_get_blue_refinement_step1(trait, raw_records_for_fit, population_scope):
        """ver4-25, R7 - step (i): fit genotype random WITH a genomic
        relationship matrix on `raw_records_for_fit` (the SAME unmasked
        table the main fit for this trait/population-scope will use),
        cached exactly like `_asreml_get_unmasked_result()` above.
        Returns `None` (never raises) on any failure - a NOTE is printed
        and the caller falls back to the UNREFINED single fit, since this
        is a best-effort quality improvement, never a hard requirement.
        """
        _cache_key = (trait, population_scope)
        if _cache_key in _asreml_blue_refinement_cache:
            return _asreml_blue_refinement_cache[_cache_key]
        try:
            _step1_grm_inv, _step1_grm_ids = _asreml_build_grm_for_universe(
                raw_records_for_fit, require_grm=True,
                grm_blend_override=_asreml_blue_refinement_cfg.get('grm_blend', 0.01),
            )
            _excluded = _asreml_last_grm_diagnostics.get('n_excluded') or 0
            _total = _asreml_last_grm_diagnostics.get('n_total') or 0
            _max_excluded_fraction = _asreml_blue_refinement_cfg.get('max_excluded_fraction', 0.05)
            if _total and (_excluded / _total) > _max_excluded_fraction:
                print(f"[GP] NOTE: ASREML['blue_refinement'] step (i) would exclude "
                      f"{_excluded}/{_total} genotype(s) ({_excluded / _total:.1%}) lacking "
                      f"marker data - above max_excluded_fraction={_max_excluded_fraction:.1%}. "
                      f"Falling back to the UNREFINED single fit for trait {trait!r} rather than "
                      f"refine BLUEs from a GRM covering only a small, possibly unrepresentative "
                      f"fraction of genotypes.")
                _asreml_blue_refinement_cache[_cache_key] = None
                return None
            _t0 = time.time()
            step1_vparameters = fit_blue_refinement_step1(
                raw_records_for_fit, trait, ASREML.get('genotype_col'), ASREML,
                _step1_grm_ids, grm_inv=_step1_grm_inv, grm_ids=_step1_grm_ids,
                workspace=ASREML.get('workspace', '4gb'),
                max_update_rounds=ASREML.get('max_update_rounds', 20),
                threads=ASREML.get('threads'),
            )
            print(f"[GP] [ASReml] R7 step (i) (genotype random + GRM) finished for trait "
                  f"{trait!r}: {len(step1_vparameters)} variance component(s) estimated (took "
                  f"{time.time() - _t0:.1f}s) - held fixed for step (ii)'s own genotype-FIXED "
                  f"refit (Holland & Piepho 2024, Fig. 2).")
        except Exception as exc:
            print(f"[GP] NOTE: ASREML['blue_refinement'] step (i) failed for trait {trait!r}: "
                  f"{exc!r} - falling back to the UNREFINED single fit for this trait; the "
                  f"main BLUE adjustment is otherwise unaffected.")
            step1_vparameters = None
        _asreml_blue_refinement_cache[_cache_key] = step1_vparameters
        return step1_vparameters

    def _asreml_get_unmasked_result(trait, mask_keys=None, target_populations=None):
        _population_scope = (
            None if _asreml_use_entire_dataset or target_populations is None
            else frozenset(target_populations)
        )
        _cache_key = (trait, _population_scope)
        if not _asreml_mask_valid_test and _cache_key in _asreml_unmasked_cache:
            return _asreml_unmasked_cache[_cache_key]
        # Requirements.md (latest pass), item 1 - the DEFAULT branch:
        # restrict the raw records THIS fit sees (and, immediately
        # below, the GRM's own genotype universe) to just
        # `target_populations` - mirroring the restriction every OTHER
        # Step in `_process_one_task()` already applies (Step 1's own
        # phenotype subsetting keeps only the target population's rows
        # before ANY other Step, including this one, ever runs).
        # `_population_scope` is `None` (no restriction - the ver4-19/
        # ver4-20 pooled-whole-file behaviour) only when
        # `ASREML['use_entire_dataset']` has been explicitly turned ON
        # (or a grid-structured residual forces it on regardless - see
        # `asreml_syntax.resolve_use_entire_dataset()`).
        if _population_scope is None:
            _raw_records_for_fit = data_phenotype_raw_for_asreml
        else:
            _raw_records_for_fit = data_phenotype_raw_for_asreml[
                data_phenotype_raw_for_asreml['population'].astype(str).isin(
                    {str(_p) for _p in _population_scope}
                )
            ].reset_index(drop=True)
        # Requirements.md item 1 - "coordinated with GRM": the genomic
        # relationship matrix's own genotype universe follows the SAME
        # restriction as the phenotype records it adjusts, via
        # `_raw_records_for_fit` here (rather than always the full,
        # unrestricted `data_phenotype_raw_for_asreml`) - a GRM covering
        # individuals from populations the fit itself never sees would
        # be needlessly expensive to build and inconsistent with what
        # the fit actually used it for.
        _grm_inv, _grm_ids = _asreml_build_grm_for_universe(_raw_records_for_fit)
        # Requirements.md item 4 fix - already validated AND reconciled
        # against the resolved response_unit once, before task 0 (see
        # `_asreml_resolved_committed_syntax`'s own definition, near
        # `_asreml_response_unit`'s) - never re-read from `ASREML` raw
        # here, so every trait's fit uses the SAME (possibly transparently
        # upgraded) syntax text, never the stale, un-reconciled one.
        _committed_syntax = _asreml_resolved_committed_syntax
        if _asreml_mask_valid_test:
            print(f"[GP] [ASReml] Fitting phenotype {trait!r} for this task only - "
                  f"ASREML['mask_valid_test']=True, so the hoisted "
                  f"per-trait cache is bypassed (masked validation/test membership differs "
                  f"per task).")
        elif _population_scope is None:
            print(f"[GP] [ASReml] Hoisted unmasked fit for phenotype {trait!r} - this fit is "
                  f"shared by every task that predicts this trait (Requirements.md R2 SS2.4).")
        else:
            print(f"[GP] [ASReml] Hoisted unmasked fit for phenotype {trait!r}, restricted to "
                  f"target population(s) {sorted(_population_scope)!r} "
                  f"({_raw_records_for_fit.shape[0]} of {data_phenotype_raw_for_asreml.shape[0]} "
                  f"raw row(s)) - ASREML['use_entire_dataset']=False (Requirements.md latest "
                  f"pass, item 1), so every OTHER population in the phenotype file is excluded "
                  f"from this fit. Shared by every task that predicts this SAME trait for this "
                  f"SAME population (set).")
        _t0 = time.time()
        # ver4-25, R7 (design blueprint stage 4): step (i) only applies to
        # a genotype-FIXED main strategy (Holland & Piepho 2024 Fig. 2 is
        # specifically a BLUE-quality improvement - see `fit_unmasked_
        # adjustment()`'s own docstring on `step1_vparameters` for why a
        # random genotype term has nothing to refine). Computed here
        # (rather than inside `fit_unmasked_adjustment()` itself) so its
        # own per-trait caching/preflight lives alongside every other
        # ASReml caching decision in this module.
        _step1_vparameters = None
        if _asreml_blue_refinement_enabled and ASREML.get('genotype_effect', 'random') == 'fixed':
            _step1_vparameters = _asreml_get_blue_refinement_step1(
                trait, _raw_records_for_fit, _population_scope
            )
        # ver4-25, R9-S4 (design blueprint stage 5, "Warm start per-task
        # fits") - only meaningful when THIS call is a genuine per-task
        # refit (masking on - see this function's own cache-bypass logic
        # above); a hoisted, cached fit is only ever fit once per trait
        # anyway, so there is nothing to warm-start across. Mutually
        # exclusive with R7's own step1_vparameters (fit_unmasked_
        # adjustment() itself only ever consults warm_start_vparameters
        # when step1_vparameters is falsy).
        _warm_start_vparameters = (
            _asreml_warm_start_cache.get(_cache_key)
            if _asreml_mask_valid_test and ASREML.get('warm_start_per_task', False) else None
        )
        result = fit_unmasked_adjustment(
            _raw_records_for_fit, trait, ASREML.get('genotype_col'),
            ASREML, _committed_syntax,
            response_unit=_asreml_response_unit, population_col='population',
            grm_inv=_grm_inv, grm_ids=_grm_ids,
            workspace=ASREML.get('workspace', '4gb'),
            max_update_rounds=ASREML.get('max_update_rounds', 20),
            mask_keys=mask_keys, role='main_unmasked', step1_vparameters=_step1_vparameters,
            warm_start_vparameters=_warm_start_vparameters, threads=ASREML.get('threads'),
            # ver4-25, R8 (design blueprint SS2, Stage 6): request the
            # dense first-stage covariance matrix from the SAME predict()
            # call whenever 'full'-mode BLUE-reliability weighting is on
            # - no second ASReml-R call is ever made for this. `False`
            # (every pre-ver4-25 caller, and every 'none'/'diagonal' run)
            # is byte-identical (I11) - `_asreml_blue_weighting` is
            # resolved once, below, before any task runs; this closure
            # picks it up via ordinary late-binding at CALL time (every
            # task's own call happens well after that resolution).
            return_vcov=(_asreml_blue_weighting == 'full'),
        )
        if _asreml_mask_valid_test and ASREML.get('warm_start_per_task', False):
            _asreml_warm_start_cache[_cache_key] = dict(
                zip(result.varcomp['component'].astype(str), result.varcomp['estimate'].astype(float))
            )
        print(f"[GP] [ASReml] Fit for phenotype {trait!r} finished: "
              f"{'converged' if result.converged else 'DID NOT CONVERGE'} after "
              f"{result.iterations} update round(s) (took {time.time() - _t0:.1f}s) - "
              f"{result.values.shape[0]} adjusted value(s)"
              + (" cached for every future task on this (trait, population-scope)."
                 if not _asreml_mask_valid_test else " (not cached - masking is on)."))
        if not _asreml_mask_valid_test:
            _asreml_unmasked_cache[_cache_key] = result
        return result

    # ver4-7, R1 ("Per-Set GRM Generation for GBLUP") - AMB-1 resolved as
    # a 3rd held-out estimation mode. RK-2's own mitigation ("cache per
    # (trait, split, population-scope) like _asreml_unmasked_cache"):
    # keyed by this task's own EXACT split membership - two tasks that
    # happen to share the identical (train, valid, test) individual sets
    # for the same trait (e.g. two model-selection replicates over the
    # same scenario) reuse the same 3-fit result rather than re-fitting;
    # any task with even one different individual pays for its own fresh
    # 3 fits, since "no cross-contamination" means each split's own GRM
    # universe genuinely depends on that split's own membership.
    _asreml_per_set_grm_cache: dict = {}
    def _asreml_get_per_set_grm_result(trait, id_train, id_valid, id_test, target_populations,
                                        population_train=None, population_valid=None,
                                        population_test=None):
        """Builds each split's own GRM and dispatches `fit_per_set_grm_
        adjustment()`. `id_train`/`id_valid`/`id_test` are THIS task's
        own split membership (already population-restricted by Step 1,
        upstream, exactly like every other Step in `_process_one_task()`).

        Scope of each split's fit and GRM:
          - `_asreml_use_entire_dataset` False (default): every split is
            restricted to just its own individuals ("no cross-
            contamination").
          - `_asreml_use_entire_dataset` True: every split is
            `own individuals + every individual that belongs to none of
            this task's own train/valid/test sets` (i.e. every OTHER
            population). ver4-57 BUGFIX - this used to widen the TRAIN
            split only, so the other populations' information never
            reached the validation/test fits or their GRMs. No individual
            of one of this task's own splits is ever pooled into a
            different split's fit, so train/valid/test stay independent.
        """
        id_train = [str(v) for v in id_train]
        id_valid = [str(v) for v in id_valid]
        id_test = [str(v) for v in id_test]

        _cache_key = (trait, tuple(sorted(id_train)), tuple(sorted(id_valid)), tuple(sorted(id_test)))
        if _cache_key in _asreml_per_set_grm_cache:
            return _asreml_per_set_grm_cache[_cache_key]

        if _asreml_use_entire_dataset:
            # Bugfix (production failure, task 1/64): widen train to every
            # OTHER population's individuals, but never to this task's own
            # held-out valid/test individuals - mirroring the identical
            # "widen to everyone except this task's own held-out ids"
            # exclusion already applied to the concurrent-variance-model
            # fit, below. Without this exclusion, any individual in this
            # task's own id_valid/id_test also ends up in the widened
            # train scope, so it gets fit twice (once in the train split's
            # own physically-independent GRM fit, once in its own
            # valid/test fit) - fit_per_set_grm_adjustment()'s own
            # combined-result sanity check then correctly raises
            # ValueError("N individual ID(s) appear in more than one of
            # the per-set-GRM fits after combining").
            _all_raw_ids = set(data_phenotype_raw_for_asreml['ID'].dropna().astype(str))
            _train_scope_ids = sorted(
                (set(id_train) | _all_raw_ids) - set(id_valid) - set(id_test)
            )
            # ver4-57 BUGFIX: the individuals that belong to NONE of this
            # task's own train/valid/test sets (every other population,
            # plus any of the target population's own individuals this
            # task does not use) - pooled into the valid and test fits
            # and GRMs exactly as `_train_scope_ids` already pools them
            # into the train fit. They are excluded from every own-split
            # list, so no train/valid/test individual leaks across
            # splits, and the train fit alone reports them.
            _context_ids = sorted(
                _all_raw_ids - set(id_train) - set(id_valid) - set(id_test)
            )
        else:
            _train_scope_ids = id_train
            _context_ids = []

        # The ids each split's GRM / fit actually covers (own + context).
        # With the default (entire-dataset off) the context is empty and
        # these reduce to exactly the pre-ver4-57 scopes.
        _valid_scope_ids = sorted(set(id_valid) | set(_context_ids)) if id_valid else []
        _test_scope_ids = sorted(set(id_test) | set(_context_ids)) if id_test else []

        def _records_for(ids):
            return data_phenotype_raw_for_asreml[
                data_phenotype_raw_for_asreml['ID'].astype(str).isin(set(ids))
            ].reset_index(drop=True)

        _train_universe_records = _records_for(_train_scope_ids)
        _valid_universe_records = _records_for(_valid_scope_ids) if id_valid else pd.DataFrame()
        _test_universe_records = _records_for(_test_scope_ids) if id_test else pd.DataFrame()

        _grm_inv_train = _grm_ids_train = None
        if _train_scope_ids:
            _grm_inv_train, _grm_ids_train = _asreml_build_grm_for_universe(
                _train_universe_records, require_grm=True,
            )
        _grm_inv_valid = _grm_ids_valid = None
        if id_valid:
            # ver4-57: covers `_valid_scope_ids` (own + other populations
            # when use_entire_dataset is on; own only otherwise).
            _grm_inv_valid, _grm_ids_valid = _asreml_build_grm_for_universe(
                _valid_universe_records, require_grm=True,
            )
        _grm_inv_test = _grm_ids_test = None
        if id_test:
            _grm_inv_test, _grm_ids_test = _asreml_build_grm_for_universe(
                _test_universe_records, require_grm=True,
            )

        print(f"[GP] [ASReml] Per-set GRM: fitting phenotype {trait!r} with 3 "
              f"INDEPENDENT genotype-random+GRM models - train ({len(_train_scope_ids)} "
              f"individual(s)), valid ({len(_valid_scope_ids)} individual(s): "
              f"{len(id_valid)} own), test ({len(_test_scope_ids)} individual(s): "
              f"{len(id_test)} own)"
              + (f" - ASREML['use_entire_dataset']=True, so every split's GRM and fit also "
                 f"pools the {len(_context_ids)} individual(s) of every other population"
                 f"; no individual of another train/valid/test split of this task is "
                 f"shared ('no cross-contamination')."
                 if _asreml_use_entire_dataset else
                 " - each split is scoped to just its own individuals ('no cross-"
                 "contamination')."))

        # ver4-57: `id_train`/`id_valid`/`id_test` are handed over as this
        # task's OWN split membership, and the other populations'
        # individuals travel separately as `context_ids` (empty unless
        # `use_entire_dataset`), which `fit_per_set_grm_adjustment()` pools
        # into ALL THREE fits. The train fit therefore still widens to
        # every population's individuals exactly as before (and still
        # reports them, once), while the valid/test fits now widen too and
        # report only their own individuals. (Before ver4-57 the widened
        # `_train_scope_ids` was passed as the train list and valid/test
        # were never widened.) The caller's own subsequent
        # `apply_to_splits()` call still only ever LOOKS UP this task's own
        # (narrower) ids, so nothing extra ever reaches this task's own
        # `train`/`valid`/`test` DataFrames.
        _result = fit_per_set_grm_adjustment(
            data_phenotype_raw_for_asreml, trait, id_train, id_valid, id_test, ASREML,
            _asreml_resolved_committed_syntax,
            response_unit=_asreml_response_unit, population_col='population',
            grm_inv_train=_grm_inv_train, grm_ids_train=_grm_ids_train,
            grm_inv_valid=_grm_inv_valid, grm_ids_valid=_grm_ids_valid,
            grm_inv_test=_grm_inv_test, grm_ids_test=_grm_ids_test,
            workspace=ASREML.get('workspace', '4gb'), max_update_rounds=ASREML.get('max_update_rounds', 20),
            threads=ASREML.get('threads'),
            context_ids=_context_ids,
        )
        _asreml_per_set_grm_cache[_cache_key] = _result
        return _result

    # ------------------------------------------------------------------- #
    # Requirements.md (BLUEs/BLUPs accuracy-metrics phase), items 1-7 -
    # the extra per-model metrics (adjusted prediction accuracy and the
    # Heritability it divides by), computed only when ASREML is enabled
    # (non-legacy engine) AND the user has opted in via
    # ASREML['accuracy_metrics']['enabled']. See asreml_accuracy_metrics.py
    # for the maths; everything here is only about WHICH extra ASReml
    # "companion" fit(s) feed it, at what scope.
    #
    # REQUIRED-COMPANIONS-ONLY (multi-option update): each option builds
    # only the fit(s) it truly needs, decided in ONE place -
    # `asreml_syntax.accuracy_companion_plan()`:
    #
    #   option           | companion supplying H^2 (Heritability)
    #   -----------------|----------------------------------------------
    #   BLUEs            | one BLUP companion (a BLUE has no variance
    #                    | information of its own): GRM-less by default,
    #                    | GRM-based if accuracy_metrics.use_grm
    #   BLUPs w/o GRM    | NONE - the option's own fit is the reference
    #   BLUPs with GRM   | NONE - the option's own fit is the reference
    #
    # No option fits a BLUE companion: every option's adjusted accuracy is
    # its OWN Pearson correlation against its OWN test target divided by
    # sqrt(H^2) (numerator_mode='reuse_pearson'). The two-step-BLUE
    # refinement fit (BLUEs only) is independent of this block and is
    # gated on ASREML['blue_refinement']['enabled'] alone.
    #
    # A companion, when needed, is fit via the SAME
    # `fit_unmasked_adjustment()` the main fit already uses, against a
    # COPY of `ASREML` with only `genotype_effect`/`use_grm` overridden -
    # so every other configured term (fixed/random effects, residual
    # structure, response_unit, workspace) and every scope rule (entire-
    # dataset, masking, population-restriction) is IDENTICAL to the main
    # fit. Each is cached under the SAME rule `_asreml_get_unmasked_
    # result()` above already uses (hoisted per (role, trait, population-
    # scope) unless masking is on).
    # ------------------------------------------------------------------- #
    _asreml_accuracy_metrics_cfg = (
        (ASREML.get('accuracy_metrics') or {}) if (ASREML is not None and not _asreml_legacy_mode) else {}
    )
    _asreml_accuracy_metrics_enabled = bool(_asreml_accuracy_metrics_cfg.get('enabled', False))
    # Requirements.md (multi-option update), item 4: ONE direction dict
    # serves BOTH the (ASReml-only) adjusted-accuracy context and the
    # always-on Spearman/Top-k rows. The top-level `METRIC_DIRECTIONS`
    # wins; the legacy `ASREML['accuracy_metrics']['direction']` fills in
    # whatever it leaves unspecified.
    _metric_direction_by_phenotype = dict(_asreml_accuracy_metrics_cfg.get('direction', {}) or {})
    _metric_direction_by_phenotype.update(dict(METRIC_DIRECTIONS or {}))
    _asreml_accuracy_direction_by_phenotype = _metric_direction_by_phenotype
    # Requirements.md item 2 - the companion fit that supplies BOTH
    # Heritability (Cullis et al.) AND the Spearman/Top-k reference values
    # used to always be a GRM-less ('BLUP-without-GRM') fit, regardless of
    # the main strategy. `ASREML['accuracy_metrics']['use_grm']` (default
    # False, I11 - an absent/older config reproduces today's exact
    # behaviour) lets a user opt this reference INTO a GRM-based
    # ('BLUP-with-GRM') fit instead - see `_asreml_get_accuracy_metrics_
    # context()`'s own updated dispatch, below, for exactly how this is
    # honoured for each main strategy, including "the choices of
    # use_entire_dataset/mask_valid_test/per_set_grm apply here too" (the
    # requirement's own item 2 text) - each is threaded through either by
    # reusing `main_result` directly (when the main strategy is already
    # 'blup_grm' - see that branch's own comment) or via the SAME
    # `_asreml_get_companion_result()` scope machinery every OTHER
    # companion already goes through.
    _asreml_accuracy_metrics_use_grm = bool(_asreml_accuracy_metrics_cfg.get('use_grm', False))

    # ver4-25, R3 (design blueprint stage 3 - naive-ensemble accuracy-
    # metrics bugfix): REPLACES the old `_asreml_task_scope_registry`
    # (which stored just enough state - mask_keys/target_populations - to
    # RE-DERIVE a task's own accuracy context later, at finalisation time
    # - see this module's git history / the ver4-25 design blueprint's
    # own E7 finding for the full root-cause account). This dict instead
    # caches the ALREADY-COMPUTED ctx dict itself, the instant it is
    # first resolved (see the per-task dispatch loop's own updated
    # comment, below, at the point this is populated) - so the naive
    # ensemble's own finalisation step never needs to re-fit or re-derive
    # anything for a task THIS process itself already processed. For a
    # task this process did NOT process (a resumed run's own pre-resume
    # tasks; a task finalised by intra_task_parallel.finalise_task_from_
    # merged() in a completely separate process), `_asreml_accuracy_ctx_
    # resolver()` below falls back to the PERSISTED `asreml_accuracy_ctx`
    # accumulator (asreml_accuracy_metrics.ctx_resolver_from_frame()) -
    # which, being an ordinary accumulator, is populated identically
    # whether its rows came from this run or a prior one.
    _asreml_ctx_scenario_cache: dict = {}
    # Built lazily, once, right before the naive-ensemble finalisation
    # block runs (see that block's own comment) - `None` until then, so
    # `_asreml_accuracy_ctx_resolver()`'s own "live cache miss" branch has
    # nothing to fall back to for any earlier, purely-per-task use of the
    # resolver (there is none today - the resolver is only ever invoked
    # at finalisation - but this keeps the fallback well-defined
    # regardless).
    _asreml_accuracy_ctx_sidecar_resolver = None

    _asreml_companion_cache = {}          # {(role, trait, population_scope): AsremlResult}

    def _asreml_build_companion_cfg(role):
        """`role` is `'blup_no_grm'` or (Requirements.md item 2)
        `'blup_grm_ref'` - see the block comment above this section for
        when each is actually needed (`asreml_syntax.accuracy_companion_
        plan()` decides). There is deliberately NO `'blue'` role: no
        option needs a BLUE companion any more. Everything else (fixed/
        random effects, residual structure, output_level, workspace, GRM
        source/blend settings) is copied verbatim from the run's own
        `ASREML`."""
        _cfg = dict(ASREML)
        if role == 'blup_no_grm':
            _cfg['genotype_effect'] = 'random'
            _cfg['use_grm'] = False
        elif role == 'blup_grm_ref':
            # Requirements.md item 2 - the GRM-based reference companion,
            # only ever built when the main strategy does NOT already use
            # a GRM itself (see `_asreml_get_accuracy_metrics_context()`'s
            # own dispatch - a 'blup_grm' main strategy reuses
            # `main_result` directly instead of fitting this a second
            # time).
            _cfg['genotype_effect'] = 'random'
            _cfg['use_grm'] = True
        else:
            raise ValueError(f"_asreml_build_companion_cfg: unknown role {role!r}")
        return _cfg

    def _asreml_build_companion_syntax(companion_cfg, trait):
        _spec = asreml_syntax.build_model_spec(
            companion_cfg, trait, response_unit=_asreml_response_unit, population_col='population',
        )
        return asreml_syntax.render_syntax(_spec, data_symbol='{DATA}', grm_symbol=None)

    def _asreml_get_companion_result(role, trait, mask_keys=None, target_populations=None):
        """Fetch (fitting + caching, mirroring `_asreml_get_unmasked_
        result()` above exactly) the `role` companion fit for `trait`, at
        the SAME data scope (entire-dataset / population-restriction /
        masking) the main fit uses for this trait."""
        _population_scope = (
            None if _asreml_use_entire_dataset or target_populations is None
            else frozenset(target_populations)
        )
        _cache_key = (role, trait, _population_scope)
        if not _asreml_mask_valid_test and _cache_key in _asreml_companion_cache:
            return _asreml_companion_cache[_cache_key]

        if _population_scope is None:
            _raw_records_for_fit = data_phenotype_raw_for_asreml
        else:
            _raw_records_for_fit = data_phenotype_raw_for_asreml[
                data_phenotype_raw_for_asreml['population'].astype(str).isin(
                    {str(_p) for _p in _population_scope}
                )
            ].reset_index(drop=True)

        _companion_cfg = _asreml_build_companion_cfg(role)
        _companion_syntax = _asreml_build_companion_syntax(_companion_cfg, trait)

        # Requirements.md item 2 - the 'blup_grm_ref' companion role is
        # the GRM-based reference fit used for the Spearman/Top-k
        # accuracy-metrics calculation when the user has opted into
        # scoring those against a GRM-based BLUP instead of the
        # no-GRM BLUP every other role/companion still uses. Build the
        # GRM the same way the main fit would (`require_grm=True` so
        # this happens regardless of the run's own top-level
        # `ASREML['use_grm']`, mirroring the `blue_refinement` step (i)
        # precedent above) over this companion's own `_raw_records_for_fit`
        # scope (already restricted to the same entire-dataset/population
        # scope the main fit uses for this trait).
        if role == 'blup_grm_ref':
            _companion_grm_inv, _companion_grm_ids = _asreml_build_grm_for_universe(
                _raw_records_for_fit, require_grm=True,
            )
        else:
            _companion_grm_inv, _companion_grm_ids = None, None

        print(f"[GP] [ASReml accuracy metrics] Fitting {role!r} companion model for phenotype "
              f"{trait!r} ({'mask=on' if mask_keys else 'unmasked'}, "
              f"{'entire dataset' if _population_scope is None else sorted(_population_scope)}).")
        _t0 = time.time()
        result = fit_unmasked_adjustment(
            _raw_records_for_fit, trait, ASREML.get('genotype_col'),
            _companion_cfg, _companion_syntax,
            response_unit=_asreml_response_unit, population_col='population',
            grm_inv=_companion_grm_inv, grm_ids=_companion_grm_ids,
            workspace=ASREML.get('workspace', '4gb'),
            max_update_rounds=ASREML.get('max_update_rounds', 20),
            mask_keys=mask_keys,
            log_prefix=f"[ASReml accuracy/{role}]", role=f'companion_{role}',
        )
        print(f"[GP] [ASReml accuracy metrics] {role!r} companion for {trait!r} finished: "
              f"{'converged' if result.converged else 'DID NOT CONVERGE'} after "
              f"{result.iterations} update round(s) (took {time.time() - _t0:.1f}s).")
        if not _asreml_mask_valid_test:
            _asreml_companion_cache[_cache_key] = result
        return result

    def _asreml_values_lookup(result, population_test):
        """`{id: value}` from an `AsremlResult.values` frame, restricted
        to `population_test` when the fit is `(id, population)`-keyed
        (`response_unit == 'genotype_by_population'`) - mirrors the exact
        join rule the main Step 6b block already applies (see
        `_asreml_values_this_task`'s own construction, above in
        `_process_one_task`)."""
        _df = result.values
        if 'population' in _df.columns:
            if population_test is not None:
                _df = _df[_df['population'].astype(str) == str(population_test)]
            else:
                _df = _df.drop_duplicates(subset='id', keep='first')
        return dict(zip(_df['id'].astype(str), _df['value'].astype(float)))

    def _asreml_se_lookup(result, population_test):
        """ver4-25, R8 (design blueprint SS2, Stage 6): `{id: std_error}`
        from an `AsremlResult.values` frame - the SAME restriction rule
        as `_asreml_values_lookup()` above, reading the `std_error`
        column instead of `value`. This is the ONLY new ASReml-side data
        `diagonal`-mode BLUE-reliability weighting needs - `std_error` is
        already returned by every existing Step 6b fit (`predict()$pvals`
        - see `Preprocess/asreml_blup.py::AsremlResult`'s own docstring),
        so `diagonal` mode requires no new R-side call at all."""
        _df = result.values
        if 'population' in _df.columns:
            if population_test is not None:
                _df = _df[_df['population'].astype(str) == str(population_test)]
            else:
                _df = _df.drop_duplicates(subset='id', keep='first')
        return dict(zip(_df['id'].astype(str), _df['std_error'].astype(float)))

    # ver4-25, R8 (design blueprint SS2, Stage 6): resolved once, before
    # any task runs (config is immutable across the run) - `'none'`
    # (absent key, default) reproduces every pre-ver4-25 run byte-
    # identically (I11), since `_asreml_get_blue_weight_context()` below
    # is then never even called and every model/scoring call site's own
    # `sample_weight_ctx`/`**scoring_for(...)` argument stays `{}`/`None`.
    #
    # `'full'` mode's own scope limits (design blueprint SS2 R8): requires
    # `genotype_effect_strategy == 'blue_fixed'` (the two-fit BLUP paths
    # never produce the fixed-effects covariance structure this weighting
    # is defined over) and the NEW (non-legacy) Step 6b engine (the legacy
    # masked-holdout engine's own `_fit_single_asreml_model()` supports
    # `return_vcov` too, but was never exercised for this - the modern
    # engine is the only one actually wired into `_asreml_get_unmasked_
    # result()`'s own `return_vcov=` call site above). Checked ONCE, here,
    # rather than per-task, mirroring `MODEL_AVAILABILITY_STRICT`'s own
    # "loud and early, not a silent wrong answer" precedent.
    _asreml_blue_weighting = (
        (ASREML.get('blue_weighting', 'none') or 'none') if (ASREML is not None and not _asreml_legacy_mode) else 'none'
    )
    if _asreml_blue_weighting not in ('none', 'diagonal', 'full'):
        raise ValueError(f"ASREML['blue_weighting']={_asreml_blue_weighting!r} is not one of "
                          f"'none'/'diagonal'/'full'.")
    if _asreml_blue_weighting == 'full':
        if _asreml_legacy_mode:
            raise ValueError(
                "ASREML['blue_weighting']='full' requires the modern (non-legacy) Step 6b "
                "engine - ASREML['legacy_masked_holdout'] must be False (or unset). Use "
                "ASREML['blue_weighting']='diagonal' with the legacy engine instead, or "
                "disable legacy_masked_holdout."
            )
        _asreml_strategy_for_weighting = asreml_syntax.resolve_genotype_effect_strategy(ASREML)
        if _asreml_strategy_for_weighting != 'blue_fixed':
            raise ValueError(
                f"ASREML['blue_weighting']='full' requires genotype_effect_strategy="
                f"'blue_fixed' (got {_asreml_strategy_for_weighting!r}) - the full "
                f"variance-covariance matrix is only meaningful over a fixed-effects BLUE; "
                f"a BLUP/GBLUP path's own prediction covariance has a different structure "
                f"this weighting scheme does not cover. Use ASREML['blue_weighting']="
                f"'diagonal' instead, or 'none'."
            )
    # ver4-25, R8: dense N x N inversion is only attempted below this
    # many individuals per split (blue_weights.build_blue_weight_context's
    # own `max_dense_n` guard) - configurable since "how large is too
    # large" depends entirely on available memory; read once, here,
    # rather than re-reading `ASREML` inside every task.
    _asreml_blue_weighting_max_dense_n = int(ASREML.get('blue_weighting_max_dense_n', 20000)) \
        if ASREML is not None else 20000
    # ver4-25, R8 bugfix follow-up #2 (this session): two opt-in remedies
    # for the weight-mass-concentration phenomenon `blue_weights.
    # _kish_effective_n()`/`_kish_effective_n_matrix()` were added to
    # DIAGNOSE (ver4-25 Change Summary Part 4) - a second, independent
    # review of that same diagnostic confirmed it as the concrete
    # mechanism behind "weighted accuracy looks lower than unweighted",
    # and specifically for 'full' mode's small-sample GLS scoring matrix.
    # Both default to their strict I11 no-op value, so this is inert for
    # every config that doesn't explicitly opt in - see blue_weights.
    # build_blue_weight_context()'s own docstring for the exact math and
    # `_soften_raw_weights()`/`_block_weights()`'s docstrings for the
    # full rationale of each.
    _asreml_blue_weighting_smoothing_power = (
        float(ASREML.get('blue_weighting_smoothing_power', 1.0)) if ASREML is not None else 1.0
    )
    _asreml_blue_weighting_full_shrinkage = (
        float(ASREML.get('blue_weighting_full_shrinkage', 0.0)) if ASREML is not None else 0.0
    )
    if _asreml_blue_weighting != 'none':
        print(f"[GP] BLUE-reliability weighting: mode={_asreml_blue_weighting!r} - every model "
              f"and ensemble this run selects will be fitted with the same shared reliability "
              f"weights, and Metric.csv's Pearson correlation/MSE will be weighted accordingly "
              f"(unweighted copies kept in extra columns)."
              + (f" 'full' mode's own dense-matrix guard: max_dense_n="
                 f"{_asreml_blue_weighting_max_dense_n} individuals per split."
                 if _asreml_blue_weighting == 'full' else "")
              + (f" Weight softening ON: ASREML['blue_weighting_smoothing_power']="
                 f"{_asreml_blue_weighting_smoothing_power:.3g} (1.0 = off)."
                 if _asreml_blue_weighting_smoothing_power != 1.0 else "")
              + (f" Full-mode covariance shrinkage ON: ASREML['blue_weighting_full_shrinkage']="
                 f"{_asreml_blue_weighting_full_shrinkage:.3g} (0.0 = off)."
                 if _asreml_blue_weighting_full_shrinkage != 0.0 else ""))

    # ver4-25, R8: print-once flag for the 'full'-mode naive-ensemble
    # scope-decision NOTE (see the sidecar-write block below) - a
    # single-element list (not a plain bool) so `_process_one_task`
    # (a sibling closure) can mutate it without needing its own
    # `nonlocal` declaration for one more name.
    _asreml_full_weighting_naive_ensemble_note_printed = [False]

    # ver4-77 (R2): resolved HERE (hoisted above its first reader, the BLUE context builder just
    # below, which needs to know whether the Super learner will fit on train+valid rows); it is
    # logged once, next to the ITERATIVE_ENSEMBLE line further down. Never raises (I11).
    _sl_cfg = resolve_super_learner(SUPER_LEARNER)

    # ver4-102 (P4): opt-in seed bagging of the neural models' untuned main fit. None (the default)
    # leaves every fit call exactly as it was (I11): `_seed_bag_fit` then just calls the fit.
    _seed_bag_cfg = resolve_seed_bag(SEED_BAG)

    def _seed_bag_fit(model_name, fit, task_sample, test_df):
        """Run `fit()` once (SEED_BAG off) or as a seed bag (see models/seed_bag.py)."""
        if _seed_bag_cfg is None or not seed_bag_applies(model_name):
            return fit()
        return seed_bag_call(fit, model_name, _seed_bag_cfg['n'], _seed_bag_cfg['seed'],
                             int(task_sample), test_df.iloc[:, -1])

    def _asreml_get_blue_weight_context(main_result, population_test, id_train, id_valid, id_test):
        """ver4-25, R8: build this task's own `blue_weights.
        BlueWeightContext` from the MAIN BLUE/BLUP fit's own per-
        individual standard errors (`diagonal` mode) or its own dense
        prediction-covariance matrix (`full` mode - via
        `Preprocess.asreml_blup.restrict_vcov_to_population()`, which
        resolves `main_result.vcov`/`vcov_ids` down to a plain, this-
        population-restricted `(matrix, id_list)` pair - see that
        function's own docstring). `id_train`/`id_valid`/`id_test` are
        passed in EXPLICITLY (this function is defined at GP()'s own
        scope, a SIBLING of `_process_one_task` - not nested inside it -
        so it cannot close over that function's own task-local id lists;
        the caller inside `_process_one_task` passes its own `id_train`/
        `id_valid`/`id_test` locals here every task).
        Returns `None` when weighting is off, OR (non-fatal - mirrors
        `_asreml_get_accuracy_metrics_context()`'s own failure handling)
        when the context cannot be built for this one task, so a
        transient data problem degrades this task to UNWEIGHTED scoring/
        fitting rather than aborting the whole run. **'full' mode is
        unverified against a licensed ASReml-R install** - see the
        ver4-25 Change Summary's own disclosure."""
        if _asreml_blue_weighting == 'none':
            return None
        # ver4-52: the train+valid fitting weights are built only when the joint search can use
        # the cross-validated objective (it fits fold models on train+valid rows, and a validation
        # row has no entry in the train-split `w_fit`). Every other run builds exactly what it
        # built before (I11).
        _joint_cfg_for_cv = HYPERPARAMETERS_JOINT_OPT or {}
        _want_cv_weights = bool(JOINT_W_OPT) and (
            _joint_cfg_for_cv.get('objective_data') == 'cv_train_valid'
            or _joint_cfg_for_cv.get('search_strategy') in ('weight_first_cv', 'neural_stack')
        )
        # ver4-77 (R2): the Super learner also fits fold models on train+valid frames
        _want_cv_weights = _want_cv_weights or (_sl_cfg is not None)
        try:
            if _asreml_blue_weighting == 'diagonal':
                se_by_id = _asreml_se_lookup(main_result, population_test)
                return blue_weights.build_blue_weight_context(
                    'diagonal', id_train, id_valid, id_test, se_by_id=se_by_id,
                    log_prefix='[GP] [blue_weights]',
                    smoothing_power=_asreml_blue_weighting_smoothing_power,
                    with_cv_weights=_want_cv_weights,
                )
            # 'full'
            vcov_sub, vcov_ids_sub = asreml_blup.restrict_vcov_to_population(main_result, population_test)
            if vcov_sub is None:
                raise ValueError(
                    "the main fit's own AsremlResult carries no vcov (return_vcov was not "
                    "honoured for this fit) - see _asreml_get_unmasked_result()'s own "
                    "return_vcov= call site."
                )
            return blue_weights.build_blue_weight_context(
                'full', id_train, id_valid, id_test, vcov=vcov_sub, vcov_ids=vcov_ids_sub,
                max_dense_n=_asreml_blue_weighting_max_dense_n, log_prefix='[GP] [blue_weights]',
                full_shrinkage=_asreml_blue_weighting_full_shrinkage,
                with_cv_weights=_want_cv_weights,
            )
        except Exception as exc:
            print(f"[GP] NOTE: could not build this task's BLUE-reliability weighting context "
                  f"({exc!r}) - this task's models/ensembles will be fit and scored UNWEIGHTED "
                  f"instead. Never fatal.")
            return None

    def _asreml_degraded_accuracy_ctx(direction):
        """A well-formed, `enabled=True` context whose reference/BLUE
        values are simply empty and whose heritability is `None` - every
        metric resolves to `NaN` for this task (via
        `asreml_accuracy_metrics.compute_accuracy_metrics_row()`'s own
        NaN-on-missing-data handling) rather than the run losing the new
        Metric.csv columns entirely for this one row (see the calling
        `except` block's own comment)."""
        return {
            'enabled': True, 'direction': direction, 'heritability': None,
            'heritability_classic': None,
            'accuracy_numerator_mode': 'reuse_pearson', 'blue_values': {}, 'reference_values': {},
        }

    def _asreml_get_accuracy_metrics_context(trait, main_strategy, main_result,
                                              mask_keys=None, target_populations=None,
                                              population_test=None):
        """Assemble the plain dict `asreml_accuracy_metrics.
        compute_accuracy_metrics_row()` consumes for this task's own
        phenotype - `None` when the feature is off entirely; a
        `_asreml_degraded_accuracy_ctx()` (never raises further) if a
        companion fit fails, mirroring `fit_concurrent_variance_model()`'s
        own non-fatal precedent (these are diagnostic metrics, never part
        of any model's own training target, so a failure here must never
        fail the task itself)."""
        if not _asreml_accuracy_metrics_enabled:
            return None
        _direction = _asreml_accuracy_direction_by_phenotype.get(
            trait, asreml_accuracy_metrics.DEFAULT_DIRECTION
        )
        if _direction not in asreml_accuracy_metrics.VALID_DIRECTIONS:
            _direction = asreml_accuracy_metrics.DEFAULT_DIRECTION
        # Requirements.md item 2 - when the user has opted into scoring
        # Spearman correlation/Top-k Coincidence against a GRM-based BLUP
        # reference instead of the no-GRM BLUP every branch below used
        # unconditionally before, the reference-fit role for each main
        # strategy becomes:
        #   - blue_fixed: 'blup_grm_ref' instead of 'blup_no_grm' (a
        #     fresh GRM-based companion is always needed here, since the
        #     main fit itself has no genotype-random term at all).
        #   - blup_iid: 'blup_grm_ref' instead of reusing `main_result`
        #     directly (the main fit IS a BLUP, but explicitly without a
        #     GRM - the whole point of this option is to score against a
        #     GRM-based reference instead).
        #   - blup_grm: `main_result` reused directly instead of a fresh
        #     'blup_no_grm' companion - the main fit is ALREADY the
        #     GRM-based reference the user wants, so no second fit is
        #     needed; this also means `main_result`'s own entire_dataset/
        #     mask_valid_test/per_set_grm scoping (Requirements.md items
        #     1/3) is honoured automatically, with no extra engineering.
        # (These per-strategy rules now live in `asreml_syntax.accuracy_
        # companion_plan()`; no strategy fits a BLUE companion.)
        #
        # Required-companions-only rule (multi-option update): which extra
        # fit each option needs is decided in ONE place,
        # `asreml_syntax.accuracy_companion_plan()`. No option fits a BLUE
        # companion any more - every option's adjusted accuracy is its own
        # Pearson correlation against its own test target / sqrt(H^2)
        # (`numerator_mode='reuse_pearson'`), so a BLUP option no longer
        # pays for a BLUE fit it never needed. The only companion that can
        # remain is the one supplying H^2 when the option's own fit is not
        # the requested kind of genotype-random fit (always the case for
        # BLUEs, which carry no variance information of their own).
        try:
            _plan = asreml_syntax.accuracy_companion_plan(
                main_strategy, _asreml_accuracy_metrics_use_grm
            )
            if _plan is None:
                print(f"[GP] NOTE: ASReml accuracy metrics are not supported for "
                      f"genotype_effect_strategy={main_strategy!r} (the legacy masked-holdout "
                      f"engine) - the three new metrics will be NaN for every task this run.")
                return _asreml_degraded_accuracy_ctx(_direction)

            _accuracy_numerator_mode = _plan['numerator_mode']
            _blue_values = None
            if _plan['reference'] == 'main':
                _href_result = main_result
            else:
                _href_result = _asreml_get_companion_result(
                    _plan['reference'], trait, mask_keys=mask_keys,
                    target_populations=target_populations
                )

            _heritability = asreml_accuracy_metrics.cullis_heritability(
                _href_result.varcomp, _href_result.values, ASREML.get('genotype_col')
            )
            # Requirements.md (multi-option update), item 4: Spearman/
            # Top-k are scored against the row's OWN test target now, so
            # the companion fit's per-individual values are no longer a
            # "reference" for anything - only its Heritability (and, for
            # BLUP options, the BLUE `blue_values` numerator) is used.
            # `reference_values` stays in the ctx schema, empty, so
            # persisted sidecars/readers keep their shape.
            # Classic (entry-mean) heritability from the SAME reference fit
            # the Cullis value above just came from - so the two columns
            # are a like-for-like comparison. Carried on the fit's own
            # `AsremlResult.heritability_classic` (`getattr` default:
            # a result from any producer that never sets it -> None ->
            # NaN in Metric.csv, never an error).
            _heritability_classic = getattr(_href_result, 'heritability_classic', None)
            return {
                'enabled': True, 'direction': _direction, 'heritability': _heritability,
                'heritability_classic': _heritability_classic,
                'accuracy_numerator_mode': _accuracy_numerator_mode,
                'blue_values': _blue_values, 'reference_values': {},
            }
        except Exception as exc:
            print(f"[GP] NOTE: ASReml accuracy-metrics companion model(s) failed for phenotype "
                  f"{trait!r}: {exc!r} - the main BLUE/BLUP adjustment is unaffected; the three "
                  f"new metrics will be NaN for this task.")
            return _asreml_degraded_accuracy_ctx(_direction)

    def _asreml_accuracy_ctx_resolver(population, phenotype, ratio, sample, *, ids=None):
        """ver4-25, R3 (design blueprint stage 3 - naive-ensemble
        accuracy-metrics bugfix): resolves the SAME accuracy-metrics
        context a per-model row for this exact (population, phenotype,
        ratio, sample) task used, for the NAIVE ensemble's own
        finalisation (`models.ensemble.ensemble()`), which scores the
        whole assembled `result_test` AFTER every task has already run
        (architecture doc SS12.1) rather than inside `_process_one_
        task()` itself.

        Resolution order (never re-fits ASReml - the old version's
        biggest cost, and its root cause: see the module-level comment
        next to `_asreml_ctx_scenario_cache`'s own definition):
          1. LIVE - `_asreml_ctx_scenario_cache`, an in-memory dict of
             every ctx already computed by THIS process's own per-task
             loop (populated the instant `_asreml_get_accuracy_metrics_
             context()` returns one - see that call site's own updated
             comment). O(1), always correct, whether or not masking is
             on.
          2. SIDECAR - `_asreml_accuracy_ctx_sidecar_resolver` (built
             once, lazily, from the persisted/accumulated `asreml_
             accuracy_ctx`/`asreml_test_weights` accumulators right
             before this finalisation block runs - see that
             assignment's own comment) - covers every task NOT
             processed by this exact process (a resumed run's own
             pre-resume tasks; a task whose naive ensemble is instead
             recomputed by `intra_task_parallel.finalise_task_from_
             merged()` in a separate process, which builds its OWN
             sidecar resolver from that task's own merged file).
          3. NOTE - neither has an entry (accuracy metrics were toggled
             off when this scenario last ran, or a genuinely orphaned
             task): this row's own new metric columns are NaN, exactly
             as before, but now with an explicit, named reason.

        ver4-25, R8 bugfix (Requirements2.md Error #1, root cause D -
        found while investigating why the naive ensemble's own
        'full'/'diagonal'-mode weighting columns were persisted empty
        even though its Heritability/Spearman/Top-k columns were
        correctly populated): the LIVE cache (path 1 above) holds
        exactly what `_asreml_get_accuracy_metrics_context()` returns -
        an accuracy-metrics-ONLY dict (`heritability`/`blue_values`/
        `reference_values`) that predates this function's own R8
        `test_weights`/`test_weight_matrix` extension and NEVER gains
        those keys. Since this run's own live cache always has an entry
        for every scenario THIS process itself dispatched (the
        overwhelmingly common case - a single, non-resumed run), path 1
        used to return BEFORE path 2 (the only place `test_weights`/
        `test_weight_matrix` are ever populated - see `ctx_resolver_
        from_frame()`) was even consulted, silently discarding this
        scenario's own BLUE-reliability weighting for the naive
        ensemble's own row regardless of `ASREML['blue_weighting']`.
        Fixed by ALWAYS consulting the sidecar for `test_weights`/
        `test_weight_matrix` (passing `ids` through, so 'full' mode's
        own dense-matrix lookup - keyed by this scenario's own test
        individual order - can resolve too) and merging those two keys
        into whichever ctx (live or sidecar) otherwise wins - a strict
        no-op for `ASREML['blue_weighting']=='none'` (I11), since the
        sidecar resolver itself is `None` in that case (see its own
        assignment's `if (_asreml_accuracy_metrics_enabled or
        _asreml_blue_weighting != 'none') else None` gate) and both
        merged-in keys simply stay absent/`{}`/`None`.
        """
        if not _asreml_accuracy_metrics_enabled and _asreml_blue_weighting == 'none':
            return None
        _key = asreml_accuracy_metrics.scenario_key(population, phenotype, ratio, sample)
        _ctx = _asreml_ctx_scenario_cache.get(_key)
        _sidecar_ctx = None
        if _asreml_accuracy_ctx_sidecar_resolver is not None:
            _sidecar_ctx = _asreml_accuracy_ctx_sidecar_resolver(population, phenotype, ratio, sample, ids=ids)
        if _ctx is None and _sidecar_ctx is None:
            print(f"[GP] NOTE: no ASReml accuracy-metrics context found (neither live nor in the "
                  f"persisted ASReml_accuracy_context.csv sidecar) for (population={population!r}, "
                  f"phenotype={phenotype!r}, ratio={ratio!r}, sample={sample!r}) - its new metric "
                  f"columns will be NaN. This can happen for a scenario that was never processed "
                  f"with ASReml accuracy metrics enabled under this RESULT_NAME.")
            return None
        _resolved = dict(_ctx) if _ctx is not None else dict(_sidecar_ctx)
        # The live cache (when it's the source of `_ctx`) never carries
        # these two keys at all (see this function's own docstring) - the
        # sidecar is the only place they are ever computed, so they are
        # always taken from there, independently of which source won for
        # every OTHER key above.
        _resolved['test_weights'] = (_sidecar_ctx or {}).get('test_weights', {})
        _resolved['test_weight_matrix'] = (_sidecar_ctx or {}).get('test_weight_matrix')
        return _resolved

    # ver4-26, R1/R4 (design blueprint `EasiGP_ver4-26_Design_Blueprint.md`):
    # JOINT_W_OPT is authoritative over HP_TUNE and W_OPT if a headless/
    # hand-edited config somehow sets more than one - main_app.py's own
    # Tab 4 checkbox forces both off automatically (see that module's
    # `gather_config`), so this should never actually trigger from the
    # GUI. Logged unconditionally either way, matching this function's
    # "log negative/overridden states unconditionally" discipline (never
    # a silent override).
    _joint_active = bool(JOINT_W_OPT)
    if _joint_active and (HP_TUNE or W_OPT is not None):
        print(f"[GP] NOTE: JOINT_W_OPT={JOINT_W_OPT!r} is set - this takes priority over "
              f"HP_TUNE={HP_TUNE!r} and W_OPT={W_OPT!r}, both of which are IGNORED (forced "
              f"to None/inactive) for this run. This should never happen via the GUI (Tab "
              f"4's joint checkbox forces HP_TUNE/W_OPT off automatically) - if this NOTE "
              f"came from a hand-edited or headless config, remove HP_TUNE/W_OPT from it "
              f"to silence it.")
        HP_TUNE = None
        W_OPT = None
    print(f"[GP] JOINT_W_OPT resolved: "
          f"{'inactive (HP_TUNE/W_OPT, if set, behave exactly as before)' if not _joint_active else JOINT_W_OPT}.")
    if _joint_active:
        # ver4-44, R0: an unknown search_strategy stops the run HERE, before task 1. (Raised from
        # inside the search it would only be caught by the per-task call site, which carries on
        # with untuned models - the silent substitution this check exists to prevent.)
        _joint_strategy_checked = validate_search_strategy(HYPERPARAMETERS_JOINT_OPT)
        print(f"[GP] Joint Bayesian optimisation search_strategy={_joint_strategy_checked!r} is "
              f"recognised by EasiGP {ENGINE_VERSION}; safety_nets="
              f"{bool((HYPERPARAMETERS_JOINT_OPT or {}).get('safety_nets', True))}; "
              f"wf_cv_reweight={(HYPERPARAMETERS_JOINT_OPT or {}).get('wf_cv_reweight', 'off')!r}, "
              f"wf_cv_members={bool((HYPERPARAMETERS_JOINT_OPT or {}).get('wf_cv_members', False))}.")

    # ver4-57 (R2): resolve the opt-in iterative ensemble ONCE and log the result on every run
    # (I11 - the resolved state is always visible, including "inactive").
    _iter_cfg = resolve_iterative_ensemble(ITERATIVE_ENSEMBLE)
    if _iter_cfg is None:
        print("[GP] ITERATIVE_ENSEMBLE resolved: inactive (behaviour unchanged).")
    else:
        print(f"[GP] ITERATIVE_ENSEMBLE resolved: {_iter_cfg}.")
        if _iter_cfg['mode'] == 'two_phase':
            # ver4-68 (R1): say once per run what the mode is and which keys it does not read.
            print(f"[GP] ITERATIVE_ENSEMBLE mode='two_phase' (experimental), tier "
                  f"'{iter_ens_tier_of(_iter_cfg)}': weight_rule={_iter_cfg['weight_rule']!r}, "
                  f"p1_rounds={_iter_cfg['p1_rounds']}, p2_rounds={_iter_cfg['p2_rounds']}, "
                  f"oof_folds={_iter_cfg['oof_folds']}, oof_engine={_iter_cfg['oof_engine']!r}.")
            print(f"[GP] NOTE: ITERATIVE_ENSEMBLE mode='two_phase' does not read these keys: "
                  f"{list(_ITER_ENS_TWO_PHASE_IGNORED)} (they belong to mode='subset').")
        if MODEL_DISPATCH_FILTER is not None:
            print("[GP] NOTE: ITERATIVE_ENSEMBLE is set but MODEL_DISPATCH_FILTER is active in "
                  "this sub-run - Step 11c is skipped here (it needs every member's Step 9 "
                  "result; a task with a validation split is ineligible for model fan-out).")

    # ver4-77 (R2): the Super learner was resolved above (hoisted); say what it resolved to on
    # EVERY run, including "inactive" (I11 - the resolved state is always visible).
    if _sl_cfg is None:
        print("[GP] SUPER_LEARNER resolved: inactive (behaviour unchanged).")
    else:
        print(f"[GP] SUPER_LEARNER resolved: {_sl_cfg}.")
        if MODEL_DISPATCH_FILTER is not None:
            print("[GP] NOTE: SUPER_LEARNER is set but MODEL_DISPATCH_FILTER is active in this "
                  "sub-run - Step 11d is skipped here (it needs every member's Step 9 result; "
                  "a task with SUPER_LEARNER is ineligible for model fan-out).")
    # ver4-85: out-of-fold hyperparameter tuning (SUPER_LEARNER['oof_tuning']) - resolved state
    # printed on every run that tunes anything (I11).
    _sl_oof_tune = sl_oof_tuning_on(_sl_cfg)
    _sl_oof_plan = sl_oof_tuning_plan(_sl_cfg)    # ver4-86: None = the ver4-85 search
    if HP_TUNE:
        if _sl_oof_tune:
            print(f"[GP] Hyperparameter tuning objective: OUT-OF-FOLD over each task's training "
                  f"lines ({_sl_cfg['folds']} folds, SUPER_LEARNER['oof_tuning']=True; partition "
                  f"reshuffled relative to the Super learner's). No validation split is needed; "
                  f"each candidate costs up to {_sl_cfg['folds']} fits. Bio-prior models are not "
                  f"tuned this way (never fitted out of fold, I6). Cost savings "
                  f"(SUPER_LEARNER['oof_tuning_fast']): "
                  f"{_sl_oof_plan if _sl_oof_plan is not None else 'off (full-cost search)'}.")
        else:
            print("[GP] Hyperparameter tuning objective: the validation split (holdout).")

    # MODEL_BASE is the user's original, unsuffixed model selection - used
    # for everything that must key off a model's TYPE regardless of tuning
    # (HP_TUNE lookups, the ensemble-grouping helpers). MODEL_RUN is what the
    # per-task dispatch loop below actually iterates: identical to
    # MODEL_BASE unless a model is tuned with more than one algorithm, in
    # which case it's expanded into one suffixed entry per algorithm (e.g.
    # 'RF' -> 'RF__Grid', 'RF__Bayesian') - see
    # models.hyperparameter_tuning.expand_model_list. When HP_TUNE is None
    # (the default), MODEL_RUN == MODEL_BASE exactly, so every existing
    # caller that doesn't pass HP_TUNE sees no change here at all. JOINT_W_OPT
    # never suffixes MODEL_RUN (unlike HP_TUNE) - the joint search picks ONE
    # set of hyperparameters per model, not one per algorithm, so every
    # joint-group model keeps its plain, unsuffixed base name in MODEL_RUN.
    MODEL_BASE = MODEL
    MODEL_RUN = expand_model_list(MODEL_BASE, HP_TUNE)

    # ver4-102 (P4): when SEED_BAG is on, say which of this run's models it bags and which fit paths
    # it does not reach (never silent). Prints nothing when SEED_BAG is off (I11).
    if _seed_bag_cfg is not None:
        _bagged_here = [m for m in MODEL_RUN if seed_bag_applies(m)]
        print(f"[GP] SEED_BAG resolved: n={_seed_bag_cfg['n']}, seed={_seed_bag_cfg['seed']}; "
              f"bagged untuned fits: {_bagged_here if _bagged_here else 'none in this model list'}.")
        _not_reached = [m for m in MODEL_RUN if m.split('__')[0] in
                        ('MLP', 'GAT_infinitesimal', 'GAT_infinitesimal_node_level',
                         'GAT_fully_connected', 'GAT_prior_knowledge') and '__' in m]
        if _not_reached:
            print(f"[GP] NOTE: SEED_BAG does not bag the tuned model(s) {_not_reached} (the tuning "
                  f"trials and the confirmatory fit are a different call path); they run as before.")
        if any(m.startswith('GAT_biological_prior_knowledge') for m in MODEL_RUN):
            print("[GP] NOTE: SEED_BAG does not bag GAT_biological_prior_knowledge (its fit "
                  "function has a different, cached call signature); it runs as before.")
        if _sl_cfg is not None:
            print("[GP] NOTE: SEED_BAG does not reach the Super learner's out-of-fold and refit "
                  "fits; they run unbagged.")

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
        "model." #(ver4-5 carry-over behaviour restored)."
        if _compute_resources['hp_tune_warm_start'] else
        "every tuned (task, model) call starts its search from the user's own "
        "configured HPARAMETERS baseline, never a previous task's tuned result. "
        #"(ver4-6 default - fixes the sequential-vs-sharded divergence above)."
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

    # ver4-61, R6: the optional tuning preset (a pilot run's hyperparameter.csv). Read once,
    # here, in the parent process, before any task runs; every shard reads the same file, so
    # the entries are identical everywhere (I9) and its sha256 joins the checkpoint
    # fingerprint below (I8). `{}` = no preset.
    _hp_preset = {}
    _hp_preset_digest = None
    _hp_preset_missed = set()
    if HP_TUNE_PRESET_FILE:
        _hp_preset, _hp_preset_digest, _hp_preset_skipped = _read_hp_tune_preset(
            HP_TUNE_PRESET_FILE, HPARAMETERS_BASELINE)
        print(f"[GP] HP_TUNE_PRESET_FILE resolved: {HP_TUNE_PRESET_FILE!r} (sha256 "
              f"{_hp_preset_digest[:12]}...): {len(_hp_preset)} tuned entr"
              f"{'y' if len(_hp_preset) == 1 else 'ies'} usable, {_hp_preset_skipped} searched "
              f"row(s) skipped (unknown model or no configured baseline). Tasks that match an "
              f"entry skip the search and run one confirmatory fit at its tuned values.")
        print("[GP] NOTE: HP_TUNE_PRESET_FILE reuses hyperparameters that were selected on the "
              "pilot replicate's own validation split for replicates whose test sets may "
              "overlap it - the same optimistic bias as HP_TUNE_SCOPE='per_scenario'. Report "
              "it with the results.")
        if not _hp_preset:
            print("[GP] WARNING: HP_TUNE_PRESET_FILE holds no usable 'searched' row for this "
                  "run's models - every task runs its normal search.")
    else:
        print("[GP] HP_TUNE_PRESET_FILE resolved: not set (every tuned task runs its own "
              "search).")

    def _preset_entry_for(base_model_name, algorithm, i):
        """The preset entry for task `i` / `base_model_name` / `algorithm`, or None. A model
        that is tuned in this run but absent from the preset is reported once (the task then
        runs its normal search)."""
        if not _hp_preset:
            return None
        key = _tuning_key(sample.loc[i, 'population'], sample.loc[i, 'phenotype'],
                          sample.loc[i, 'ratio'], base_model_name, algorithm)
        entry = _hp_preset.get(key)
        if entry is None and key not in _hp_preset_missed:
            _hp_preset_missed.add(key)
            print(f"[GP] NOTE: HP_TUNE_PRESET_FILE has no 'searched' row for {key} - this task "
                  f"runs a normal search for {base_model_name} ({algorithm}).")
        return entry

    # ver4-26, R4.2b analogue: the SAME per-GP()-call, never-persisted
    # cache pattern as `_tuning_cache` just above, but keyed by
    # (population, phenotype, ratio, None, 'JOINT') - a single joint
    # search covers every joint-group model+the ensemble weights
    # together, so there is exactly one cache entry per scenario, not one
    # per model. Consulted only under HP_TUNE_SCOPE='per_scenario' (the
    # SAME resolved setting JOINT_W_OPT reuses rather than adding a
    # second scope knob - R3); a strict no-op under the default
    # 'per_task'. DISCLOSED LIMITATION (see this update's Change
    # Summary Sec.8): unlike `_tuning_cache`, this cache is NOT rebuilt
    # from a resumed `hp_record` on checkpoint resume - unresolved
    # because `hp_record` has no group-level row to rebuild it from
    # (only one row per model, not one row for the whole group's own
    # weights); a resumed run under JOINT_W_OPT + HP_TUNE_SCOPE=
    # 'per_scenario' therefore re-searches once for the first
    # post-resume task in each in-flight group instead of reusing the
    # pre-crash winner - correct, only slower than an uninterrupted run,
    # never incorrect.
    _joint_tuning_cache = {}

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
        # ver4-61, R6 (fixes D4): entries now have the dict shape `_resolve_tuned_params`
        # reads (the old code stored a bare list, which raised TypeError on the first
        # per_scenario cache hit after a resume) and are keyed through `_tuning_key`, the
        # same normaliser the lookup uses (CSV text vs. live ratio / population types never
        # matched before). Smallest `sample` wins a duplicate key, independent of row order.
        ordered = hp_record_df.copy()
        ordered['_sample_num'] = pd.to_numeric(ordered['sample'], errors='coerce')
        ordered = ordered.sort_values('_sample_num', kind='mergesort')
        for _, row in ordered.iterrows():
            if row.get('tuning_source') != 'searched':
                continue
            base_model_name = base_of(str(row['model']))
            entry = _tuned_entry_from_row(base_model_name, row['phenotype'], row,
                                          HPARAMETERS_BASELINE)
            if entry is None:
                continue
            key = _tuning_key(row['population'], row['phenotype'], row['ratio'],
                              base_model_name, row.get('algorithm'))
            cache.setdefault(key, entry)
        return cache

    def _fast_record_columns(_tuned_result):
        """ver4-40 R1: the hyperparameter.csv columns describing how a HP_TUNE_FAST search
        ran (search_dims, n_candidates, stub, engine, stopped_early). A tuned result that
        did not come from a fresh search (a per_scenario cache hit) has no '_fast_info'; its
        columns are left empty rather than invented."""
        _info = (_tuned_result.get('_fast_info') if isinstance(_tuned_result, dict) else None) or {}
        return {k: _info.get(k) for k in
                ('search_dims', 'n_candidates', 'stub', 'engine', 'stopped_early')}

    def _apply_trial_memory_guard(base_model_name, algorithm, base_params, hp_tuning_cfg,
                                  graph, train, valid, test, n_jobs, batch_max):
        """ver4-61, R2: lower a bio-prior search's trial concurrency so that the predicted
        peak memory of the concurrent trials stays within 70 % of the node's available
        memory. Returns ``(n_jobs, batch_max)`` - unchanged when no cap is needed or the
        estimate is unavailable (a NOTE says why: the guard never blocks on missing data).

        Why: see the dispatch comment above the call. The estimate is
        ``bio_prior_trial_peak_bytes`` at the corner of the search box (largest batch size
        and heads x neurons, after HP_TUNE_FAST narrowing), divided across the model workers
        that share the node. q_mem = 0 means not even one trial is predicted to fit: the
        search then runs serially with a WARNING, or stops with an error under
        ``bio_prior_cost_guard='raise'``. A lowered q changes the batch-Bayesian candidate
        sequence, which a ``[GP] NUMERICS`` line says (the one deliberate I11 exception of
        ver4-61; ``trial_memory_guard='off'`` restores ver4-60)."""
        label = f"{base_model_name} ({algorithm})"
        q_before = resolve_trial_fanout_width(
            algorithm, n_jobs, bayes_batch=_compute_resources['hp_tune_bayes_batch'],
            bayes_batch_max=batch_max,
            parallel_restarts=_compute_resources['hp_tune_parallel_restarts'],
            budget_kwargs=hp_tuning_cfg.get('budget', {}).get(algorithm, {}),
        ) if _compute_resources['hp_tune_parallel_trials'] else 1
        if graph is None:
            print(f"[GP] NOTE: trial memory guard inactive for {label}: the bio-prior graph "
                  f"was not pre-built (BIO_PRIOR_PREP_FAST off or the pre-warm failed), so its "
                  f"size is unknown here - concurrency stays {q_before}; worker deaths are "
                  f"still isolated.")
            return n_jobs, batch_max
        try:
            schema = schema_key_of(base_model_name)
            idx_neuron = field_index(schema, 'Neuron numbers')
            idx_batch = field_index(schema, 'Batch size')
            idx_heads = field_index(schema, 'Number of heads')
            highs = search_box_highs(base_model_name, base_params, HPARAM_SPECS,
                                     _compute_resources.get('hp_tune_fast'))
            batch_hi = int(highs.get(idx_batch, base_params[idx_batch]))
            hc_hi = int(highs.get(idx_neuron, base_params[idx_neuron])
                        * highs.get(idx_heads, base_params[idx_heads]))
            peak = bio_prior_trial_peak_bytes(
                graph, batch_hi=batch_hi, hc_hi=hc_hi, n_train=train.shape[0],
                n_eval=(valid.shape[0] if valid is not None else 0) + test.shape[0])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            print(f"[GP] NOTE: trial memory guard inactive for {label}: the peak-memory "
                  f"estimate could not be computed ({type(exc).__name__}: {exc}) - concurrency "
                  f"stays {q_before}.")
            return n_jobs, batch_max
        available = available_memory_bytes()
        if available is None:
            print(f"[GP] NOTE: trial memory guard inactive for {label}: the node's available "
                  f"memory is unknown - peak ~{peak / 1e9:.1f} GB per trial, concurrency stays "
                  f"{q_before}.")
            return n_jobs, batch_max
        workers = int(_compute_resources.get('n_model_workers') or 1)
        q_mem = trial_memory_concurrency(peak, available, workers)
        n_nodes = int(graph['n_genes'])
        e0 = 2 * (len(graph['edges_from']) - n_nodes) + n_nodes
        print(f"[GP] trial memory guard: {label} peak≈{peak / 1e9:.1f} GB/trial "
              f"(batch<={batch_hi}, heads x neurons<={hc_hi}, G={n_nodes}, E0~{e0}), "
              f"available {available / 1e9:.1f} GB, model workers {workers} -> concurrency "
              f"{q_before} -> {min(q_before, max(q_mem, 1))}"
              f"{' (not even one trial is predicted to fit)' if q_mem < 1 else ''}.")
        if q_mem >= q_before:
            return n_jobs, batch_max
        if q_mem < 1:
            if str(_compute_resources.get('accel_bio_prior_cost_guard')) == 'raise':
                raise RuntimeError(
                    f"[GP] trial memory guard: one {label} trial is predicted to need "
                    f"~{peak / 1e9:.1f} GB but only {available / 1e9:.1f} GB is available "
                    f"(70 % rule, {workers} model worker(s)) and bio_prior_cost_guard='raise'. "
                    f"Narrow the search box (HP_TUNE_FAST cap_search_cost), lower the data-driven "
                    f"merge's top_rate / max_edges, or run on a larger node.")
            print(f"[GP] WARNING: trial memory guard: one {label} trial is predicted to exceed "
                  f"the memory budget (~{peak / 1e9:.1f} GB vs {0.7 * available / 1e9:.1f} GB) "
                  f"- the search runs serially (one trial at a time); a trial may still be "
                  f"killed. Consider max_edges, a narrower box or a larger node.")
            q_mem = 1
        print(f"[GP] NUMERICS: trial memory guard lowered the concurrent {label} trials from "
              f"{q_before} to {q_mem}; the batch-Bayesian candidate sequence therefore differs "
              f"from the run without the guard. Set ACCELERATION['trial_memory_guard']='off' to restore it.")
        return (q_mem if q_mem > 1 else 1), min(int(batch_max), q_mem)

    def _resolve_tuned_params(base_model_name, algorithm, i, _base_params_for_tuning,
                               train, valid, test, _tuning_run_model_fn, hp_tuning_cfg,
                               _tuning_n_jobs, _trial_worker_init_args,
                               _bayes_batch_max=None, _isolate_failures=False, _oof_folds=None,
                               _oof_fast=None):
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
        _cache_key = _tuning_key(
            sample.loc[i, 'population'], sample.loc[i, 'phenotype'],
            sample.loc[i, 'ratio'], base_model_name, algorithm,
        )
        # ver4-61, R6: a matching preset entry is consulted before any search, in every
        # scope - one confirmatory fit at the pilot's tuned values, tuning_source
        # 'preset:<pilot sample>'. `_bayes_batch_max` / `_isolate_failures` (R2) and the
        # per-model time limits (R3) only matter for a real search.
        _preset_hit = _preset_entry_for(base_model_name, algorithm, i)
        if _preset_hit is not None:
            _t0 = time.time()
            _final_params = list(_preset_hit['final_params'])
            _result = _tuning_run_model_fn(
                base_model_name, train, valid, test, _final_params, RESULT_NAME)
            _elapsed = time.time() - _t0
            return (_final_params, _result, _preset_hit['best_valid_score'],
                    dict(_preset_hit['tuned_values']), _elapsed, f"preset:{_preset_hit['sample']}")
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
            bayes_batch_max=(_bayes_batch_max if _bayes_batch_max is not None
                             else _compute_resources['hp_tune_bayes_batch_max']),
            bayes_liar=_compute_resources['hp_tune_bayes_liar'],
            parallel_restarts=_compute_resources['hp_tune_parallel_restarts'],
            worker_init=_trial_worker_init,
            worker_init_args=_trial_worker_init_args,
            # ver4-61: R3 wall-clock limits of this model (HP_TUNE[model]['time_limit_min'] /
            # ['trial_time_limit_min'], minutes -> seconds; None = off) and R2 worker-death
            # isolation (set by the trial-memory guard for bio-prior models).
            time_limit_s=_limit_minutes_to_seconds(
                hp_tuning_cfg.get('time_limit_min'), 'time_limit_min', base_model_name),
            trial_time_limit_s=_limit_minutes_to_seconds(
                hp_tuning_cfg.get('trial_time_limit_min'), 'trial_time_limit_min',
                base_model_name),
            isolate_failures=bool(_isolate_failures),
            # ver4-6 R1/R3/R4: five new resolved keys - see
            # resolve_compute_resources()'s own docstring for each one's
            # default/rationale, and the "[GP] Compute resources
            # resolved:" log block near the top of this function for
            # their unconditionally-printed resolved values.
            bayes_domain_reduction=_compute_resources['hp_tune_bayes_domain_reduction'],
            selection_margin=_compute_resources['hp_tune_selection_margin'],
            valid_repeats=_compute_resources['hp_tune_valid_repeats'],
            inner_split_seed_base=_inner_seed_base,
            # FEATURE (multi-fidelity screening) - see this run's own
            # "[GP] Compute resources resolved:" log block for its
            # resolved value, and models/hyperparameter_tuning.py::
            # search_bayesian()'s own docstring for the full mechanism.
            bayes_multi_fidelity=_compute_resources['hp_tune_bayes_multi_fidelity'],
            # ver4-40 R1: None (legacy) or the resolved HP_TUNE_FAST dict. The stub and the
            # exact-kernel engine are never allowed under BLUE weighting - the runner then
            # returns BLUE-weighted valid metrics the unweighted shortcuts cannot reproduce.
            fast=_compute_resources['hp_tune_fast'],
            shortcut_eligible=(getattr(_tuning_run_model_fn, 'sample_weight_ctx', None) is None),
            # ver4-85: None = holdout scoring on `valid` (unchanged); a fold list = out-of-fold
            # scoring over `train` (SUPER_LEARNER['oof_tuning']).
            oof_folds=_oof_folds,
            # ver4-86: the out-of-fold cost-saving plan (None = ver4-85 search).
            oof_fast=_oof_fast if _oof_folds is not None else None,
        )
        # ver4-40 R1: tune_model_hyperparameters() reports "no search was needed" (every
        # tunable field removed by HP_TUNE_FAST) through a private result key, which becomes
        # the tuning_source value of this task's hyperparameter.csv row. '_fast_info' stays
        # in the result (like '_trial_secs') and is read by name where the row is built.
        _tuning_source_out = 'searched'
        if isinstance(_result, dict):
            _tuning_source_out = _result.pop('_tuning_source', None) or 'searched'
        if _scope == 'per_scenario':
            _tuning_cache[_cache_key] = {
                'final_params': list(_final_params),
                'tuned_values': dict(_tuned_values),
                'best_valid_score': _best_score,
            }
        return _final_params, _result, _best_score, _tuned_values, _elapsed, _tuning_source_out

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
    _fp_extra = {
        'HP_TUNE': HP_TUNE, 'HPARAMETERS_BASELINE': HPARAMETERS_BASELINE,
        # Update ID ver4-12, R11c (design blueprint SS2.11): the ASReml
        # stage's own identity is folded into the checkpoint fingerprint
        # the exact same way HP_TUNE/HPARAMETERS_BASELINE already are -
        # editing the mixed model (or the raw/analysed phenotype
        # handling, or how population is derived) and resuming an
        # existing checkpoint is then refused (fingerprint mismatch,
        # fresh start) rather than silently mixing tuned/adjusted values
        # computed under two different configurations. `config_fingerprint()`
        # already returns a stable constant for ASREML=None, so turning
        # the stage on/off is itself a fingerprint-changing event too.
        'ASREML': asreml_syntax.config_fingerprint(ASREML),
        'ASREML_SYNTAX': hashlib.sha256(
            (ASREML.get('committed_syntax') or '').encode('utf-8')
        ).hexdigest() if ASREML is not None else None,
        # Update ID ver4-13 ("Design decision: BLUE cannot be held
        # out"): the SECOND committed syntax a Fixed(BLUE) config
        # needs (Preprocess/asreml_blup.py::
        # _fit_blue_train_plus_holdout()) is folded in exactly the
        # same way the primary one is, just above - editing it and
        # resuming an existing checkpoint is refused (fresh start)
        # rather than silently mixing adjusted phenotypes computed
        # under two different holdout models. `None` for a config
        # that never set it (ASREML=None, or genotype_effect='random'
        # where it is simply unused) - stable, so it never spuriously
        # invalidates a checkpoint that never needed it.
        'ASREML_HOLDOUT_SYNTAX': hashlib.sha256(
            (ASREML.get('holdout_committed_syntax') or '').encode('utf-8')
        ).hexdigest() if ASREML is not None else None,
        'PHENOTYPE_FILE_MODE': PHENOTYPE_FILE_MODE,
        'POPULATION_SOURCE': POPULATION_SOURCE,
        # ver4-26, R4 (blueprint hand-off checklist item 2a): folded in
        # exactly like HP_TUNE/HPARAMETERS_BASELINE just above -
        # toggling JOINT_W_OPT, or editing its own
        # HYPERPARAMETERS_JOINT_OPT budget, now correctly REFUSES a
        # resume against a stale checkpoint too (fresh start), instead
        # of silently mixing joint-tuned values computed under the OLD
        # config. `None` for every pre-ver4-26 run/config (I11) -
        # never spuriously invalidates a checkpoint that never used
        # this feature.
        'JOINT_W_OPT': JOINT_W_OPT, 'HYPERPARAMETERS_JOINT_OPT': HYPERPARAMETERS_JOINT_OPT,
    }
    # ver4-40 R1 (P5): HP_TUNE_FAST joins the fingerprint ONLY when set, so a ver4-39
    # checkpoint (which never hashed the key) still resumes under HP_TUNE_FAST=None;
    # switching the fast mode on or off (or changing a lever) refuses a stale resume.
    if _compute_resources.get('hp_tune_fast') is not None:
        _fp_extra['HP_TUNE_FAST'] = _compute_resources['hp_tune_fast']
    # ver4-57 (R2): ITERATIVE_ENSEMBLE joins the fingerprint ONLY when set (same rule), so a
    # ver4-56 checkpoint still resumes with the key absent, and changing any iterative setting
    # refuses a stale resume (rows written under other settings are never mixed).
    # ver4-68 (R1): the view drops each ver4-68 key that is still at its default, so a run
    # resolved to mode='subset' hashes exactly what ver4-57..67 hashed (AC1.2).
    # ver4-102 (P4): SEED_BAG joins the fingerprint ONLY when active (same rule), so every older
    # checkpoint still resumes with the key absent and a changed bag size or seed refuses a stale
    # resume.
    if _seed_bag_cfg is not None:
        _fp_extra['SEED_BAG'] = seed_bag_fingerprint_view(_seed_bag_cfg)
    # ver4-103: the GAT stability settings (models/gat_stability.py) join the fingerprint ONLY when they
    # differ from the engine default (seeded, 1 network, no re-scoring), so older checkpoints still
    # resume and a changed seed / ensemble / re-scoring setting refuses a stale resume.
    _gat_stab_view = {
        'seed': _compute_resources.get('accel_gat_seed', 20240615),
        'ensemble': _compute_resources.get('accel_gat_seed_ensemble', 1),
        'rescore_top': _compute_resources.get('accel_gat_rescore_top', 0),
        'rescore_seeds': _compute_resources.get('accel_gat_rescore_seeds', 0),
    }
    if _gat_stab_view != {'seed': 20240615, 'ensemble': 1, 'rescore_top': 0, 'rescore_seeds': 0}:
        _fp_extra['GAT_STABILITY'] = _gat_stab_view
    if _iter_cfg is not None:
        _fp_extra['ITERATIVE_ENSEMBLE'] = iter_ens_fingerprint_view(_iter_cfg)
    # ver4-77 (R2): SUPER_LEARNER joins the fingerprint ONLY when set (same rule), so an older
    # checkpoint still resumes with the key absent, and toggling it or changing any of its
    # settings refuses a stale resume (a row written under other settings is never mixed in).
    if _sl_cfg is not None:
        # ver4-85: 'oof_tuning' joins only while True (sl_fingerprint_view), so a ver4-77..84
        # checkpoint with the Super learner on still resumes.
        _fp_extra['SUPER_LEARNER'] = sl_fingerprint_view(_sl_cfg)
    # ver4-61 (R6): the preset file's content hash joins the fingerprint ONLY when the key is set,
    # so an older checkpoint still resumes with the key absent, and editing (or swapping) the
    # file refuses a stale resume - rows tuned from another preset are never mixed in (I8).
    if _hp_preset_digest is not None:
        _fp_extra['HP_TUNE_PRESET'] = _hp_preset_digest
    _sample_fp = _ckpt.sample_fingerprint(
        sample.iloc[idx*interval: idx*interval+interval], extra=_fp_extra)
    _resume_last_completed_i = _ckpt.load_checkpoint(RESULT_NAME, idx, _is_parallel, _sample_fp, total_tasks)
    _static_write_flags = _ckpt.static_write_flags_for(
        MODEL_RUN, W_OPT, asreml_enabled=ASREML is not None,
        asreml_write_phenotype=_asreml_write_phenotype_file,
        # ver4-25, R8 Stage 6 completion: gates whether
        # ASReml_test_weight_matrix.csv is ever written at all this run.
        asreml_blue_weighting=_asreml_blue_weighting,
        # ver4-25, Stage 9 (R12): gates whether Interaction_higher_
        # order.csv is ever written at all this run - see
        # static_write_flags_for()'s own docstring.
        hparameters=HPARAMETERS,
        # ver4-7, R1 ("Per-Set GRM Generation for GBLUP"): gates whether
        # ASReml/GRM_{train,valid,test}.csv are ever written at all this
        # run.
        asreml_per_set_grm=_asreml_per_set_grm,
        # ver4-26, R4 (Change Summary Sec.7 - genuine blueprint gap, this
        # template's own "existing code behaviour" > blueprint authority
        # order applied): W_OPT is always forced to None whenever
        # JOINT_W_OPT is active (see the mutual-exclusion resolution near
        # MODEL_RUN above), so `'weight': w_opt is not None` alone would
        # silently stop Weight.csv being written under joint-only mode.
        # See checkpoint_utils.static_write_flags_for()'s own updated
        # docstring/signature.
        joint_w_opt=JOINT_W_OPT,
        # ver4-57 (R2/R4): the iterative ensemble writes an equal-weight Weight.csv row and
        # (unless write_rounds is off) Iterative_ensemble_rounds.csv.
        iterative_ensemble=_iter_cfg,
        # ver4-77 (R2): the Super learner writes a Weight.csv row (its cross-validated member
        # weights).
        super_learner=_sl_cfg,
    )

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
        # ver4-25, Stage 9 (R12): rehydrated generically like every other
        # accumulator (checkpoint_utils.load_partial_results() loops over
        # RESULT_FILE_NAMES, which now includes 'interactions_hok') - see
        # that function's own docstring.
        interactions_hok = _loaded['interactions_hok']
        iter_rounds = _loaded['iter_rounds']   # ver4-57 (R4): rehydrated generically too
        # Update ID ver4-12, R10/R11 - rehydrated generically like every
        # other accumulator above, via the same RESULT_FILE_NAMES-keyed
        # dict `load_partial_results()` already returns (both new keys
        # resolve to an empty DataFrame when ASREML is off, or when this
        # is a fresh RESULT_NAME with nothing on disk yet).
        asreml_pheno, asreml_varcomp = _loaded['asreml_pheno'], _loaded['asreml_varcomp']
        # ver4-25, R9-S0/R3: rehydrated the same generic way - critically,
        # this is what lets a RESUMED run's finalisation step resolve
        # accuracy-metrics context for every pre-resume task via the
        # sidecar resolver (see `_asreml_accuracy_ctx_resolver()`'s own
        # docstring) - without this line, `asreml_accuracy_ctx` would
        # silently restart empty on every resume, reproducing the exact
        # R3 bug this update fixes.
        asreml_fit_timing, asreml_accuracy_ctx = _loaded['asreml_fit_timing'], _loaded['asreml_accuracy_ctx']
        # ver4-25, R8 Stage 6 completion: rehydrated the same generic way
        # - see asreml_accuracy_ctx's own identical comment just above;
        # without this, a RESUMED 'full'-mode run's naive-ensemble/DPT
        # weighting would silently lose every pre-resume task's own
        # persisted test-weight matrix.
        asreml_test_weights = _loaded['asreml_test_weights']
        # ver4-7, R1 ("Per-Set GRM Generation for GBLUP") - same
        # rehydration reasoning as asreml_test_weights just above.
        asreml_grm_train = _loaded['grm_train']
        asreml_grm_valid = _loaded['grm_valid']
        asreml_grm_test = _loaded['grm_test']
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
    # Bugfix (checkpoint appended the wrong prediction rows): `_process_one_task()` re-sorts
    # `result_train`/`result_valid`/`result_test` by id after adding a task's rows, so the task's
    # own new rows are NOT the last `n_new` rows of those frames any more. The per-task
    # checkpoint append below used to slice them positionally
    # (`frame.iloc[len(pre_task_frame):]`), which after the sort returned the tail of the
    # id-sorted accumulator - rows of OTHER tasks - so the appended file silently held duplicated
    # rows and lost this task's own. `_process_one_task()` now records the exact rows it added
    # (taken BEFORE the sort) here, and the append reads them from this dict.
    _task_prediction_deltas = {}

    def _process_one_task(i):
        nonlocal record, result_train, result_valid, result_test, effect, interactions
        nonlocal interactions_hok, iter_rounds
        nonlocal weight, attention_total, hp_record, stats, completed_units
        nonlocal asreml_pheno, asreml_varcomp, asreml_accuracy_ctx, asreml_test_weights

        # Bugfix (Requirements.md #1): `task_num` used to be first assigned
        # only inside conditionally-entered blocks further down (e.g. "if
        # RF_filter_effective is not None:"), so any code path that reads
        # `task_num` before one of those blocks has run - such as the Step
        # 8.5 joint hyperparameter+weight optimisation block below, which
        # unconditionally logs `task_num` as soon as it starts - raised
        # `UnboundLocalError: cannot access local variable 'task_num'` the
        # moment RF filtering (and every earlier optional block defining
        # it) was disabled for the run. `task_num` depends only on `i`,
        # `idx` and `interval`, all fixed for the lifetime of this task, so
        # it is computed exactly once here, unconditionally, before any
        # branch can read it. The later per-block reassignments further
        # down recompute the identical value and are left in place
        # (harmless no-ops) rather than removed, to keep this fix minimal
        # and local.
        task_num = i - idx*interval + 1
        # ver4-56 (R1): marker count before filtering; set where the task's pools are built.
        _tele_p_raw = None

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
            # Requirements.md (between-prediction validation-set patch):
            # this subset only ever needs to know how many populations
            # THIS task's own 'population' string names (2 for a plain
            # 'X->Y' pair - true whether or not a validation split is
            # later carved out of one of those same two populations by
            # the 'split_train'/'split_test' sources below - or 3 for a
            # 'third_population' 'X->Y->Z' triple) - never W_OPT, which
            # no longer decides this (see `_between_validation_source`
            # above GP()'s own `sample`-construction code).
            if _between_validation_source != 'third_population':
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

        def _bio_prior_trial_ctx(model_names):
            """The ``bio_prior_ctx`` dict handed to a ``ModelTrialRunner`` for
            the bio-prior instance(s) `model_names` (tuning trials and the
            joint search; ver4-33, R1.e).

            Why: loky pickles the runner (and so this dict) for EVERY trial,
            and the legacy dict carried every instance's pool, the shared
            train/valid/test triple, the merge pools and ``full_marker_pool``
            - the whole raw marker matrix for PLINK runs - although a trial
            only ever reads its own instance's entries.

            PLINK + ``BIO_PRIOR_LEAN_POOLS`` (default): only the requested
            instances' ``bio_prior_pools`` / ``bio_prior_merge_pools`` /
            ``network_cache`` entries, the first requested instance's pool as
            the (otherwise unused) shared triple, and ``full_marker_pool`` =
            None - ``ModelTrialRunner`` reads it for PLINK nowhere. Anything
            else (CSV, flag off, an instance without a pool - logged) gets the
            legacy dict unchanged."""
            if GENOTYPE_FORMAT == 'plink' and _compute_resources.get('accel_bio_prior_lean_pools', True):
                _missing = [n for n in model_names if n not in bio_prior_pools]
                if not _missing and model_names:
                    _first_pool = bio_prior_pools[model_names[0]]
                    return {
                        'bio_prior_pools': {n: bio_prior_pools[n] for n in model_names},
                        'train_unpruned': _first_pool[0], 'valid_unpruned': _first_pool[1],
                        'test_unpruned': _first_pool[2],
                        'bio_prior_merge_pools': {n: bio_prior_merge_pools[n] for n in model_names
                                                  if n in bio_prior_merge_pools},
                        'full_marker_pool': None,
                        'bim_marker_info_path': bim_marker_info_path,
                        'network_cache': {n: _bio_prior_merge_cache[n] for n in model_names
                                          if n in _bio_prior_merge_cache},
                        'phenotype_name': sample.loc[i, 'phenotype'],
                    }
                print(f"[GP] Task {i - idx*interval + 1}/{total_tasks} | NOTE: lean trial context not "
                      f"possible (no gene-window pool for {_missing or list(model_names)}) - using the "
                      f"full bio-prior context for this runner (results unaffected).")
            return {
                'bio_prior_pools': bio_prior_pools,
                'train_unpruned': train_unpruned, 'valid_unpruned': valid_unpruned,
                'test_unpruned': test_unpruned,
                'bio_prior_merge_pools': bio_prior_merge_pools,
                'full_marker_pool': full_marker_pool,
                'bim_marker_info_path': bim_marker_info_path,
                'network_cache': _bio_prior_merge_cache,
                'phenotype_name': sample.loc[i, 'phenotype'],
            }

        def _prewarm_bio_graph_for(bp_name):
            """ver4-61, R1: build and memoise bio-prior instance `bp_name`'s graph in
            THIS process, so every pickled ``ModelTrialRunner`` carries it.

            Why: the graph (structure, data-driven merge, edge re-index, self-loops)
            does not depend on any tunable value, yet every trial used to rebuild it
            (~20 min per call with the merge on PLINK input). Uses exactly the frames,
            merge universe and (PLINK) ``.bim`` marker path ``ModelTrialRunner`` will
            use, so the memo key matches. Never raises: a failure is logged and the
            trials build the graph themselves. No-op for a non-bio-prior name.
            Returns the graph dict (the trial-memory guard reads its size), or None when
            nothing was built."""
            if not _is_bio_prior_model(bp_name):
                return None
            _pw_params = list(_resolve_bio_prior_params(bp_name))
            if GENOTYPE_FORMAT == 'plink':
                _pw_params[9] = bim_marker_info_path
                _pw_train, _pw_valid, _pw_test = bio_prior_pools.get(
                    bp_name, (train_unpruned, valid_unpruned, test_unpruned))
                _pw_merge_source = bio_prior_merge_pools.get(bp_name)
            else:
                _pw_train, _pw_valid, _pw_test = train_unpruned, valid_unpruned, test_unpruned
                _pw_merge_source = full_marker_pool
            return prewarm_bio_graph(
                bp_name, _pw_params, _pw_train, _pw_merge_source, _bio_prior_merge_cache,
                w_fit=_blue_weight_ctx.w_fit if _blue_weight_ctx is not None else None,
                valid=_pw_valid, test=_pw_test,
            )

        if GENOTYPE_FORMAT == 'csv':
            # ---------------------------------------------------------- #
            # Update ID ver4-12, R1 (design blueprint SS2.1): the genotype
            # table is NO LONGER pre-filtered by its OWN 'population'
            # column here. Since R2 makes 'population' a derived quantity
            # of the PHENOTYPE file alone (POPULATION_SOURCE), a genotype
            # CSV's own 'population' column - if it even has one that
            # matches - is no longer authoritative for anything: filtering
            # genotype rows against it here would silently drop
            # individuals whenever the genotype file's own population
            # labels don't happen to line up with the derived ones (e.g.
            # POPULATION_SOURCE='derived' from Location x Year, which a
            # genotype CSV's 'population' column has no way of knowing
            # about). `data_phenotype` (below) is ALREADY restricted to
            # this task's population(s) - see the phenotype-subsetting
            # block above this function's own dedup step - so the
            # ID-only inner merge two lines down achieves the exact same
            # restriction, correctly, regardless of what the genotype
            # file's own 'population' column (if any) contains.
            # ---------------------------------------------------------- #
            data_genotype = data_genotype_original

            # Requirement: rather than letting rows with no matching ID on
            # one side, or missing genotype/phenotype values on either
            # side, silently (or fatally) break the downstream split/model
            # fitting, drop them here explicitly and log what happened.
            #
            # Update ID ver4-12, R1: merge key drops from the composite
            # ['ID', 'population'] to ['ID'] alone - matching what the
            # PLINK path's own _attach_phenotype() has always done (A1).
            # The genotype side's own 'population' column (if present) is
            # dropped BEFORE the merge, guarded by a presence check, so no
            # '_x'/'_y' suffixing can ever occur and 'population' in the
            # merged frame always means the PHENOTYPE file's (derived)
            # value - never the genotype file's own, now-unused one.
            _geno_for_merge = (
                data_genotype.drop(columns=['population']) if 'population' in data_genotype.columns
                else data_genotype
            )
            _merged_raw = _geno_for_merge.merge(data_phenotype, on='ID', how='inner')
            data = _merged_raw.dropna().reset_index(drop=True)
            # Update ID ver4-12, R1: the unmatched-count diagnostic now
            # compares plain ID sets, not (ID, population) tuples - the
            # genotype side no longer carries a 'population' column to
            # zip against (see the drop just above), and comparing IDs
            # alone is the correct generalisation once 'population' is a
            # phenotype-side-only, potentially-derived quantity.
            _n_unmatched = len(
                (set(data_genotype['ID']) | set(data_phenotype['ID'])) - set(_merged_raw['ID'])
            )
            _n_missing_values = _merged_raw.shape[0] - data.shape[0]
            if _n_unmatched > 0 or _n_missing_values > 0:
                print(f"[GP] Task {i - idx*interval + 1}/{total_tasks} | population {sample.loc[i,'population']} | "
                      f"phenotype {sample.loc[i,'phenotype']} | Dropped {_n_missing_values} individual(s) with missing "
                      f"genotype or phenotype value(s); {data.shape[0]} individual(s) remain.")
            # Update ID ver4-12, R1 (RK-1 mitigation): a fan-out join
            # (more merged rows than the phenotype side had individuals)
            # means the GENOTYPE side had at least one duplicate 'ID' -
            # the PLINK path's own _attach_phenotype() has always guarded
            # against exactly this (its own docstring calls it a "fan-out
            # join"); the CSV path never could, because the composite
            # ('ID','population') key it used to merge on made a
            # genotype-side ID duplicate far less likely to collide
            # silently. Now that the key is ID-only, the same guard is
            # needed here too. `data` (post-merge, post-dropna) has
            # already been deduplicated implicitly by this point only if
            # dropna() happened to remove the extra row(s) - so this
            # check runs on `data` directly and de-duplicates by ID,
            # keeping the first match, exactly mirroring the phenotype-
            # side dedup earlier in this function.
            _geno_dup_mask = data['ID'].duplicated(keep='first')
            if _geno_dup_mask.any():
                _geno_dup_ids = sorted(set(data.loc[_geno_dup_mask, 'ID'].astype(str)))
                data = data[~_geno_dup_mask].reset_index(drop=True)
                print(f"[GP] Task {i - idx*interval + 1}/{total_tasks} | population {sample.loc[i,'population']} | "
                      f"phenotype {sample.loc[i,'phenotype']} | WARNING: {len(_geno_dup_ids)} individual ID(s) "
                      f"appeared more than once after merging genotype with phenotype data (a duplicate ID on "
                      f"the GENOTYPE side) - kept only the first match per ID. Affected ID(s)"
                      f"{' (showing first 20)' if len(_geno_dup_ids) > 20 else ''}: {_geno_dup_ids[:20]}")
            if _below_min_data_points(data.shape[0], i, 'after dropping individuals with missing or unmatched genotype/phenotype information'):
                return

            if SCENARIO =='within':
                if type(sample.loc[i,'ratio']) is not tuple:
                    train, test = train_test_split(data,train_size=sample.loc[i,'ratio'], random_state=sample.loc[i,'sample'])
                    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
                    # Update ID ver4-12, R1: named drop of ['ID','population']
                    # replaces the old positional `iloc[:, 2:]` strip - the
                    # merged frame's column order is no longer guaranteed
                    # to put ID/population first (the genotype side's own
                    # 'population' column is dropped pre-merge - see
                    # above - so the phenotype side's 'population' now
                    # lands AFTER the marker columns rather than before
                    # them). id_train/id_valid/id_test are still captured
                    # from the 'ID' column BEFORE the drop, positionally
                    # aligned to the split rows exactly as before (I4).
                    id_train, id_valid, id_test = train['ID'].astype(str), pd.DataFrame(), test['ID'].astype(str)
                    train, valid, test = train.drop(columns=['ID', 'population']), pd.DataFrame(), test.drop(columns=['ID', 'population'])
                elif type(sample.loc[i,'ratio']) is tuple:
                    train, valid = train_test_split(data,train_size=sample.loc[i,'ratio'][0], random_state=sample.loc[i,'sample'])
                    valid, test = train_test_split(valid,train_size=sample.loc[i,'ratio'][1]/(sample.loc[i,'ratio'][2]+sample.loc[i,'ratio'][1]), random_state=sample.loc[i,'sample'])
                    train, valid, test = train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)
                    id_train, id_valid, id_test = train['ID'].astype(str), valid['ID'].astype(str), test['ID'].astype(str)
                    train, valid, test = train.drop(columns=['ID', 'population']), valid.drop(columns=['ID', 'population']), test.drop(columns=['ID', 'population'])
            elif SCENARIO =='between':
                tmp = sample.loc[i, 'population'].split('->')
                # Requirements.md (between-prediction validation-set
                # patch), items 1-2: which of the four ways this task's
                # train/valid/test frames get built is decided by
                # `_between_validation_source` (resolved once, above
                # GP()'s own `sample`-construction code) - never by W_OPT.
                if _between_validation_source is None:
                    # No validation split at all - identical to the
                    # pre-patch W_OPT-is-None branch.
                    train, test = data[data['population'].astype(str)==tmp[0]], data[data['population'].astype(str)==tmp[1]]
                    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
                    id_train, id_valid, id_test = train['ID'].astype(str), pd.DataFrame(), test['ID'].astype(str)
                    train, valid, test = train.drop(columns=['ID', 'population']), pd.DataFrame(), test.drop(columns=['ID', 'population'])
                elif _between_validation_source == 'third_population':
                    # An entire third, independent population supplies
                    # validation - identical to the pre-patch
                    # W_OPT-is-not-None branch.
                    train, valid, test = data[data['population'].astype(str)==tmp[0]], data[data['population'].astype(str)==tmp[1]], data[data['population'].astype(str)==tmp[2]]
                    train, valid, test = train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)
                    id_train, id_valid, id_test = train['ID'].astype(str), valid['ID'].astype(str), test['ID'].astype(str)
                    train, valid, test = train.drop(columns=['ID', 'population']), valid.drop(columns=['ID', 'population']), test.drop(columns=['ID', 'population'])
                else:
                    # 'split_train' / 'split_test' (Requirements.md item
                    # 2): validation individuals are randomly carved out
                    # of the TRAINING population's own rows, or the TEST
                    # population's own rows, using this task's own
                    # BETWEEN_RATIO value (a plain float - the fraction
                    # that STAYS in its original role) and 'sample'
                    # replicate number as the random_state, exactly
                    # mirroring how the 'within' scenario's own plain-
                    # float ratio splits train from test above.
                    _pop_train_rows = data[data['population'].astype(str)==tmp[0]].reset_index(drop=True)
                    _pop_test_rows = data[data['population'].astype(str)==tmp[1]].reset_index(drop=True)
                    if _between_validation_source == 'split_train':
                        train, valid = train_test_split(_pop_train_rows, train_size=sample.loc[i,'ratio'], random_state=sample.loc[i,'sample'])
                        test = _pop_test_rows
                    else:  # 'split_test'
                        test, valid = train_test_split(_pop_test_rows, train_size=sample.loc[i,'ratio'], random_state=sample.loc[i,'sample'])
                        train = _pop_train_rows
                    train, valid, test = train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)
                    id_train, id_valid, id_test = train['ID'].astype(str), valid['ID'].astype(str), test['ID'].astype(str)
                    train, valid, test = train.drop(columns=['ID', 'population']), valid.drop(columns=['ID', 'population']), test.drop(columns=['ID', 'population'])

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
                # Requirements.md (between-prediction validation-set
                # patch), items 1-2 - mirrors the CSV-path branch above
                # exactly, just against `data_phenotype` (genotype is
                # attached per-split afterwards, via `_attach_phenotype()`
                # below, on this deferred-conversion path).
                if _between_validation_source is None:
                    pheno_train = data_phenotype[data_phenotype['population'].astype(str)==tmp[0]].reset_index(drop=True)
                    pheno_test = data_phenotype[data_phenotype['population'].astype(str)==tmp[1]].reset_index(drop=True)
                    pheno_valid = pd.DataFrame()
                elif _between_validation_source == 'third_population':
                    pheno_train = data_phenotype[data_phenotype['population'].astype(str)==tmp[0]].reset_index(drop=True)
                    pheno_valid = data_phenotype[data_phenotype['population'].astype(str)==tmp[1]].reset_index(drop=True)
                    pheno_test = data_phenotype[data_phenotype['population'].astype(str)==tmp[2]].reset_index(drop=True)
                else:  # 'split_train' / 'split_test'
                    _pop_train_pheno = data_phenotype[data_phenotype['population'].astype(str)==tmp[0]].reset_index(drop=True)
                    _pop_test_pheno = data_phenotype[data_phenotype['population'].astype(str)==tmp[1]].reset_index(drop=True)
                    if _between_validation_source == 'split_train':
                        pheno_train, pheno_valid = train_test_split(_pop_train_pheno, train_size=sample.loc[i,'ratio'], random_state=sample.loc[i,'sample'])
                        pheno_test = _pop_test_pheno
                    else:  # 'split_test'
                        pheno_test, pheno_valid = train_test_split(_pop_test_pheno, train_size=sample.loc[i,'ratio'], random_state=sample.loc[i,'sample'])
                        pheno_train = _pop_train_pheno
                    pheno_train, pheno_valid, pheno_test = pheno_train.reset_index(drop=True), pheno_valid.reset_index(drop=True), pheno_test.reset_index(drop=True)

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
                # ver4-19, Requirements.md R1: the ASReml GRM route (Step
                # 6b, below) used to also set this - it needed
                # `full_marker_pool` (this TASK's own unrestricted marker
                # matrix) whenever `grm_source == 'genotype_file'`. As of
                # ver4-19 the GRM is instead built from the genotype-
                # UNIVERSE helper (`_asreml_universe_marker_frames()`
                # above), which reads directly from
                # `data_genotype_original`/the PLINK fileset - never from
                # `full_marker_pool` - for BOTH `grm_source` values now,
                # so materialising `full_marker_pool` purely for ASReml's
                # sake is no longer needed at all (a bonus cost reduction
                # from the same fix that corrects Requirements.md R1's
                # own genotype-scoping bug - see
                # `Preprocess/asreml_blup.py::resolve_grm_genotype_
                # universe()` and this module's own Step 6b call site).
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

            # ver4-33, R1.e (BIO_PRIOR_LEAN_POOLS, default True - byte-identical
            # results): materialise only what the merge RF actually ranks, and
            # keep only the RF-selected columns of the merge pool, instead of
            # the full train/valid/test marker matrices (see the helpers below
            # and the narrowing right after each instance's own pool is built).
            _lean_pools = bool(_compute_resources.get('accel_bio_prior_lean_pools', True))
            _lean_raw_cache = {}

            def _lean_train_pool(markers):
                """Train-split pool (phenotype last) holding ONLY `markers`
                (None = every marker), cached per marker tuple for this task.
                Same extraction + `_attach_phenotype` the legacy raw pool used,
                so each column's values (incl. mean imputation) are identical."""
                _key = None if markers is None else tuple(markers)
                if _key not in _lean_raw_cache:
                    _geno = plink_to_genotype_df(
                        GENOTYPE_FILE_NAME, pheno_train, extract_snp_list=None if markers is None else list(markers),
                        keep_iid_list=id_train, plink_path=GENOTYPE_PLINK_PATH, work_dir=_new_plink_work_dir())
                    _lean_raw_cache[_key] = _attach_phenotype(_geno, pheno_train, split_label='train')
                return _lean_raw_cache[_key]

            def _lean_merge_rf_inputs(ld_cfg):
                """(train_x, train_y) for the data-driven-merge RF - the same
                matrix the legacy path built from its full raw pool (LD
                survivors, in the pruning list's own column order) - without
                materialising valid/test, or (kb/variants LD) the markers LD
                pruning discards. 'cm' windows have no native pruning and use
                the test split as LD_pruning's placeholder argument, so only
                that case also extracts the full test split."""
                if ld_cfg is not None and ld_cfg.get('window_unit', 'kb') != 'cm':
                    _pruned = ld_prune_marker_list_native(
                        GENOTYPE_FILE_NAME, id_train, ld_cfg,
                        work_dir=_new_plink_work_dir(), bim_marker_info=bim_marker_info,
                    )
                    _pool = _lean_train_pool(_pruned)
                    return _pool.iloc[:, :-1][_pruned], _pool.iloc[:, -1]
                _pool = _lean_train_pool(None)
                if ld_cfg is None:
                    return _pool.iloc[:, :-1], _pool.iloc[:, -1]
                if 'test' not in _lean_raw_cache:
                    _test_geno = plink_to_genotype_df(
                        GENOTYPE_FILE_NAME, pheno_test, keep_iid_list=id_test, plink_path=GENOTYPE_PLINK_PATH,
                        work_dir=_new_plink_work_dir())
                    _lean_raw_cache['test'] = _attach_phenotype(_test_geno, pheno_test, split_label='test')
                _x_cm, _, _ = LD_pruning(
                    _pool.iloc[:, :-1], pd.DataFrame(), _lean_raw_cache['test'].iloc[:, :-1], ld_cfg)
                return _x_cm, _pool.iloc[:, -1]

            # ver4-33, R1.d: key identifying this task's .bim content for the
            # per-process gene-window memo. A unique key (never matches a
            # later task) if the .bim cannot be stat-ed.
            try:
                _bim_stat = os.stat(f"{GENOTYPE_FILE_NAME}.bim")
                _bim_memo_key = (os.path.abspath(f"{GENOTYPE_FILE_NAME}.bim"), _bim_stat.st_size, _bim_stat.st_mtime_ns)
            except OSError:
                _bim_memo_key = ('unkeyed', time.time_ns())

            # ver4-33, R1.a: stage 'bio_pool' = the gene-window pool
            # preparation below (closed after the loop; the native LD, RF and
            # PLINK extraction inside it nest their own stages and are
            # excluded from this one's exclusive time).
            _bio_pool_stage = begin_stage('bio_pool')
            if need_gene_pool:
                print(f"[GP] Task {i - idx*interval + 1}/{total_tasks} | BIO_PRIOR_LEAN_POOLS={_lean_pools}, "
                      f"BIO_PRIOR_PREP_FAST={_compute_resources.get('accel_bio_prior_prep_fast', True)}.")
                for _bp_name in bio_prior_names_now:
                    _bp_preview = _resolve_bio_prior_params(_bp_name)

                    # ver4-33, R1.d: the same load_network_json ->
                    # extract_candidate_genes -> build_gene_list ->
                    # _map_genes_to_markers chain this loop always ran (the one
                    # shared mapper, I6), memoised per process.
                    gene_window_markers = preview_gene_window_markers(
                        _bp_preview[7], _bp_preview[8], _bp_preview[10], bim_marker_info, _bim_memo_key)
                    if not gene_window_markers:
                        raise RuntimeError(
                            f"No markers in the PLINK fileset fall inside any gene window for "
                            f"{_bp_name} - cannot build its gene-network marker pool."
                        )
                    bio_prior_gene_window_markers[_bp_name] = gene_window_markers

                    _extract_markers = gene_window_markers
                    if _bio_prior_merge_enabled(_bp_name):
                        if _raw_full_pool is None and not _lean_pools:
                            _raw_train_geno = plink_to_genotype_df(GENOTYPE_FILE_NAME, pheno_train, keep_iid_list=id_train, plink_path=GENOTYPE_PLINK_PATH, work_dir=_new_plink_work_dir())
                            _raw_valid_geno = plink_to_genotype_df(GENOTYPE_FILE_NAME, pheno_valid, keep_iid_list=id_valid, plink_path=GENOTYPE_PLINK_PATH, work_dir=_new_plink_work_dir()) if pheno_valid.shape[0] != 0 else pd.DataFrame()
                            _raw_test_geno = plink_to_genotype_df(GENOTYPE_FILE_NAME, pheno_test, keep_iid_list=id_test, plink_path=GENOTYPE_PLINK_PATH, work_dir=_new_plink_work_dir())
                            _raw_full_pool = (
                                _attach_phenotype(_raw_train_geno, pheno_train, split_label='train'),
                                _attach_phenotype(_raw_valid_geno, pheno_valid, split_label='valid') if pheno_valid.shape[0] != 0 else pd.DataFrame(),
                                _attach_phenotype(_raw_test_geno, pheno_test, split_label='test'),
                            )
                        if not _lean_pools:
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
                        _ld_cfg_preview = _merge_cfg_preview.get('ld_prune')
                        if _lean_pools:
                            # ver4-33, R1.e: LD survivors first, then extract
                            # only those train columns (see _lean_merge_rf_inputs)
                            _rf_x_preview, _rf_y_preview = _lean_merge_rf_inputs(_ld_cfg_preview)
                        else:
                            _rf_x_preview, _rf_y_preview = _raw_full_pool[0].iloc[:, :-1], _raw_full_pool[0].iloc[:, -1]
                        if _ld_cfg_preview is not None and not _lean_pools:
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
                        _rf_selected_preview, _ = memo_select_markers(
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
                    if _lean_pools and _bio_prior_merge_enabled(_bp_name):
                        # ver4-33, R1.e: the merge only ever reads the
                        # RF-selected columns of its pool (it slices
                        # `merge_train.loc[:, rf_selected]`; the RF result is
                        # already cached above), and this instance's pool
                        # already holds them (extraction = gene-window U
                        # RF-selected) with identical values - so that
                        # narrowed view IS the merge source.
                        if _train_unpruned.shape[0] != 0 and _test_unpruned.shape[0] != 0:
                            _narrow_cols = list(_rf_selected_preview) + [_train_unpruned.columns[-1]]
                            bio_prior_merge_pools[_bp_name] = (
                                _train_unpruned[_narrow_cols],
                                _valid_unpruned[_narrow_cols] if _valid_unpruned.shape[0] != 0 else pd.DataFrame(),
                                _test_unpruned[_narrow_cols],
                            )
                        else:
                            bio_prior_merge_pools[_bp_name] = (_train_unpruned, _valid_unpruned, _test_unpruned)
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
            _lean_raw_cache.clear()      # ver4-33, R1.e: release the lean RF-input frames
            end_stage(_bio_pool_stage)

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
                            _rf_selected_plink, _ = memo_select_markers(
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
                        _rf_selected_csv, _ = memo_select_markers(
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

        # ver4-56 (R1): the task's individual count and marker count BEFORE LD pruning / RF
        # filtering (for a PLINK run whose pruning already ran natively this is the pruned
        # count). Log-only.
        try:
            _tele_p_raw = train.shape[1] - 1
            _rtele.task_dims(train.shape[0] + valid.shape[0] + test.shape[0], _tele_p_raw)
        except (NameError, AttributeError, IndexError) as _tele_exc:
            print(f"[telemetry] WARNING: could not read the task's marker count "
                  f"({type(_tele_exc).__name__}: {_tele_exc}) - the task record has no p_raw.")
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

        # ------------------------------------------------------------- #
        # STEP 6b - ASReml-R BLUE/BLUP substitution (Update ID ver4-12,
        # R5/R8/R9 - design blueprint SS2.5 decision D1). Deliberately
        # placed HERE - after LD pruning (response-INDEPENDENT - operates
        # on the marker matrix alone, so it is unaffected by which
        # response values currently sit in `train`) and BEFORE RF marker
        # filtering (response-DEPENDENT - Preprocess/RF_marker_filtering.
        # select_rf_markers() ranks markers against whatever is in
        # `train`'s last column, so it must see the ADJUSTED phenotype,
        # not the raw/placeholder one, or it would rank markers against
        # the wrong response). See architecture doc SS7 and the ver4-12
        # blueprint SS2.5/SS2.8/SS2.9 for the full ordering/masking/
        # convergence contract this implements.
        #
        # A no-op (this whole block skipped) whenever ASREML is None -
        # the config-only rollback switch (blueprint SS8): every other
        # Step in this function is completely unaffected either way.
        # ------------------------------------------------------------- #
        # Requirements.md (BLUEs/BLUPs accuracy-metrics phase) - default,
        # always defined regardless of whether ASREML is enabled at all,
        # so Step 9/10/11's own record-row construction (further below)
        # can read it unconditionally; only overwritten inside the "NEW
        # DEFAULT ENGINE" branch below, when the feature is actually on.
        _asreml_accuracy_ctx = None
        # ver4-25, R8: same "always defined regardless" discipline -
        # None whenever ASREML is off/legacy-mode/weighting='none'.
        _blue_weight_ctx = None
        if ASREML is not None:
            task_num = i - idx*interval + 1

            def _ids_to_str_list(id_obj):
                """id_train/id_valid/id_test are a pd.Series (CSV path)
                or a plain list (PLINK path's own pheno_train/valid/test
                ['ID'].astype(str).tolist() - see _attach_phenotype's own
                callers above) when non-empty, and either pd.DataFrame()
                (CSV path, no validation split) or [] (PLINK path) when
                empty - normalised here to a plain list of strings either
                way, since Step 6b needs one uniform shape regardless of
                which genotype format produced it."""
                if isinstance(id_obj, pd.DataFrame):
                    return []
                return [str(v) for v in id_obj]

            _asreml_id_train = _ids_to_str_list(id_train)
            _asreml_id_valid = _ids_to_str_list(id_valid)
            _asreml_id_test = _ids_to_str_list(id_test)
            _asreml_all_ids = _asreml_id_train + _asreml_id_valid + _asreml_id_test
            _asreml_trait = sample.loc[i, 'phenotype']
            # ver4-19, Requirements.md R3 - only ever set (to a real
            # population label) by the new engine's own branch below,
            # when response_unit=='genotype_by_population'; initialised
            # here so the shared accumulation code further down can
            # reference them unconditionally regardless of which engine
            # ran.
            _asreml_pop_train = _asreml_pop_valid = _asreml_pop_test = None

            # ver4-19 - which of the TWO Step 6b engines this run uses was
            # already resolved once, before scenario enumeration (see
            # `_asreml_legacy_mode`'s own definition, near
            # `_asreml_grm_cache`'s). Both branches below populate the
            # SAME `_asreml_result`/varcomp-accumulation shape further
            # down, so everything from "Requirement 8: overwrite..."
            # onward is shared, engine-agnostic code.
            if _asreml_legacy_mode:
                # ---------------------------------------------------- #
                # LEGACY ENGINE (ver4-12...ver4-18) - masked validation/
                # test responses, the "BLUE cannot be held out" two-fit
                # design. Unchanged except for ONE thing: the genomic
                # relationship matrix's own genotype scope now goes
                # through `_asreml_build_grm_for_universe()`
                # (Requirements.md R1's fix), instead of being built
                # straight from `full_marker_pool`/`(train, valid,
                # test)` for exactly `_asreml_all_ids` - see that
                # function's own docstring for why R1 applies to BOTH
                # engines unconditionally, not only the new default one.
                # ---------------------------------------------------- #
                if asreml_syntax.residual_requires_full_grid(ASREML):
                    _asreml_raw_records = data_phenotype_raw_for_asreml.reset_index(drop=True)
                else:
                    _asreml_raw_records = data_phenotype_raw_for_asreml[
                        data_phenotype_raw_for_asreml['ID'].astype(str).isin(_asreml_all_ids)
                    ].reset_index(drop=True)

                _asreml_committed_syntax = ASREML.get('committed_syntax')
                if not _asreml_committed_syntax:
                    raise ValueError(
                        "[ASReml] ASREML['committed_syntax'] is empty - commit a syntax from the "
                        "GUI's ASReml section (Tab 2), or set it directly in a hand-written "
                        "config, before running with ASREML enabled."
                    )
                if '{TRAIT}' not in _asreml_committed_syntax:
                    raise ValueError(
                        "[ASReml] The committed ASReml-R syntax no longer contains the {TRAIT} "
                        "placeholder - it must be present so the SAME committed text is "
                        "reusable, unmodified, across every phenotype this run predicts."
                    )

                _asreml_grm_inv, _asreml_grm_ids = _asreml_build_grm_for_universe(
                    _asreml_raw_records, task_marker_source=(train, valid, test),
                    task_ids=_asreml_all_ids,
                )

                print(f"[GP] Task {task_num}/{total_tasks} | population {sample.loc[i,'population']} | "
                      f"phenotype {_asreml_trait} | ASReml-R (legacy/masked): fitting on "
                      f"{len(_asreml_id_train)} training record(s) "
                      f"({_asreml_raw_records.shape[0]} raw row(s) for "
                      f"{len(_asreml_all_ids)} individual(s) total"
                      + (" - a grid-structured residual is configured, so this includes every "
                         "OTHER population's row(s) too, plus any NaN filler row(s) needed to "
                         "complete the trial's own spatial grid"
                         if asreml_syntax.residual_requires_full_grid(ASREML)
                         else "") + "), predicting "
                      f"{len(_asreml_id_valid) + len(_asreml_id_test)} masked individual(s) "
                      f"(GRM {'on' if _asreml_grm_inv is not None else 'off'}).")
                asreml_start_time = time.time()
                _asreml_stage = begin_stage('asreml')   # ver4-56 (E8): closed in `finally` below
                try:
                    _asreml_result = asreml_adjust_phenotypes(
                        _asreml_raw_records, _asreml_trait,
                        _asreml_id_train, _asreml_id_valid, _asreml_id_test,
                        ASREML, _asreml_committed_syntax,
                        grm_inv=_asreml_grm_inv, grm_ids=_asreml_grm_ids,
                        workspace=ASREML.get('workspace', '4gb'),
                        max_update_rounds=ASREML.get('max_update_rounds', 20),
                        threads=ASREML.get('threads'),
                    )
                except Exception as asreml_exc:
                    _on_nonconvergence = ASREML.get('on_nonconvergence', 'fail')
                    if _on_nonconvergence == 'skip':
                        print(f"[ASReml] NON-CONVERGED / FAILED for task {task_num}/{total_tasks} "
                              f"(population={sample.loc[i,'population']!r}, phenotype={_asreml_trait!r}): "
                              f"{asreml_exc!r}. ASREML['on_nonconvergence']='skip' - skipping this task "
                              f"entirely (no rows written for it), the same precedent "
                              f"_below_min_data_points() already sets for an uncomputable scenario.")
                        completed_units += units_per_task
                        _report_progress(completed_units, total_units, label='Skipped (ASReml non-convergence)')
                        return
                    raise RuntimeError(
                        f"[ASReml] Fit failed for task {task_num}/{total_tasks} (population="
                        f"{sample.loc[i,'population']!r}, phenotype={_asreml_trait!r}): {asreml_exc!r}. "
                        f"Set ASREML['on_nonconvergence']='skip' to skip an unfittable scenario instead "
                        f"of stopping the whole run (default 'fail' stops it)."
                    ) from asreml_exc
                finally:
                    end_stage(_asreml_stage)

                print(f"[GP] Task {task_num}/{total_tasks} | ASReml-R finished: "
                      f"{'converged' if _asreml_result.converged else 'DID NOT CONVERGE'} after "
                      f"{_asreml_result.iterations} update round(s) (took "
                      f"{time.time() - asreml_start_time:.1f}s).")
                _asreml_append_fit_timing(
                    {'population': sample.loc[i, 'population'], 'phenotype': _asreml_trait,
                     'ratio': sample.loc[i, 'ratio'], 'sample': sample.loc[i, 'sample']},
                    _asreml_result.timing,
                )

                train, valid, test = apply_to_splits(
                    train, valid, test, _asreml_id_train, _asreml_id_valid, _asreml_id_test, _asreml_result,
                )
                _asreml_strategy_for_log = asreml_syntax.resolve_genotype_effect_strategy(ASREML)
                # F5 (optimise_scan): four write-only `_asreml_*_for_log` locals removed here.
            else:
                # ---------------------------------------------------- #
                # NEW DEFAULT ENGINE (ver4-19, Requirements.md R2/R3;
                # ver4-20 follow-up adds the optional masking toggle) -
                # ONE fit per phenotype trait, hoisted/shared across
                # every task predicting it UNLESS masking is enabled
                # (`_asreml_get_unmasked_result()`, defined near
                # `_asreml_grm_cache`'s own initialisation, for the exact
                # caching rule).
                # ---------------------------------------------------- #
                # Requirements.md R3 - a `response_unit ==
                # 'genotype_by_population'` result is keyed by (id,
                # population); every split of a `SCENARIO='within'` task
                # shares this task's own single population, while a
                # `SCENARIO='between'` task's population string encodes
                # each split's own population as `'X->Y'`/`'X->Y->Z'`
                # (the SAME parsing this module's own scenario-enumeration
                # code already applies when building train/valid/test).
                # Computed BEFORE the fit call (moved up from its
                # original position, ver4-20) because `mask_keys` below
                # needs each masked individual's own population when
                # `response_unit == 'genotype_by_population'`.
                _asreml_pop_train = _asreml_pop_valid = _asreml_pop_test = None
                if _asreml_response_unit == 'genotype_by_population':
                    _asreml_pop_raw = str(sample.loc[i, 'population'])
                    if SCENARIO == 'between':
                        _asreml_pop_parts = _asreml_pop_raw.split('->')
                        _asreml_pop_train = _asreml_pop_parts[0]
                        if len(_asreml_pop_parts) >= 3:
                            _asreml_pop_valid, _asreml_pop_test = _asreml_pop_parts[1], _asreml_pop_parts[2]
                        elif len(_asreml_pop_parts) == 2:
                            _asreml_pop_test = _asreml_pop_parts[1]
                            # Requirements.md (between-prediction
                            # validation-set patch): a 2-part 'X->Y'
                            # population string now ALSO covers the new
                            # 'split_train'/'split_test' sources, which
                            # DO have a real validation split - just one
                            # drawn from the same population as train (or
                            # test), never a distinct third one. Key the
                            # masking lookup by that same population label
                            # rather than leaving it None (which would
                            # otherwise silently mask no validation
                            # individual at all under this response unit).
                            if _between_validation_source == 'split_train':
                                _asreml_pop_valid = _asreml_pop_parts[0]
                            elif _between_validation_source == 'split_test':
                                _asreml_pop_valid = _asreml_pop_parts[1]
                    else:
                        _asreml_pop_train = _asreml_pop_valid = _asreml_pop_test = _asreml_pop_raw

                # ver4-20 follow-up - this task's own validation/test
                # (genotype[, population]) keys to mask, or `None` when
                # `ASREML['mask_valid_test']` is off (the recommended
                # default - see `fit_unmasked_adjustment()`'s own
                # docstring for the exact matching rule this feeds).
                _asreml_mask_keys = None
                if _asreml_mask_valid_test:
                    _asreml_mask_keys = set()
                    for _mid in _asreml_id_valid:
                        _asreml_mask_keys.add((_mid, _asreml_pop_valid))
                    for _mid in _asreml_id_test:
                        _asreml_mask_keys.add((_mid, _asreml_pop_test))

                # Requirements.md (latest pass), item 1 - this task's own
                # target population(s), consulted by
                # `_asreml_get_unmasked_result()` ONLY when
                # `_asreml_use_entire_dataset` is False (a no-op argument
                # otherwise - see that function's own docstring/comment).
                _asreml_target_populations = _asreml_target_populations_for_task(
                    sample.loc[i, 'population']
                )

                asreml_start_time = time.time()
                _asreml_stage = begin_stage('asreml')   # ver4-56 (E8): closed in `finally` below
                try:
                    # ver4-7, R1 ("Per-Set GRM Generation for GBLUP") -
                    # dispatch branch: when `ASREML['per_set_grm']` is
                    # True, route through the 3-independent-fit
                    # per-split GRM engine instead of the ordinary
                    # (hoisted, cached) unmasked fit. Mutually exclusive
                    # with masking - `_asreml_per_set_grm` and
                    # `_asreml_mask_valid_test` can never both be True at
                    # once (asreml_syntax.validate_structure()'s own
                    # preflight check refuses that combination before
                    # task 0), so there is no ambiguity about which
                    # branch a given run takes.
                    if _asreml_per_set_grm:
                        _asreml_result = _asreml_get_per_set_grm_result(
                            _asreml_trait, _asreml_id_train, _asreml_id_valid, _asreml_id_test,
                            _asreml_target_populations,
                            population_train=_asreml_pop_train, population_valid=_asreml_pop_valid,
                            population_test=_asreml_pop_test,
                        )
                    else:
                        _asreml_result = _asreml_get_unmasked_result(
                            _asreml_trait, mask_keys=_asreml_mask_keys,
                            target_populations=_asreml_target_populations,
                        )
                except Exception as asreml_exc:
                    _on_nonconvergence = ASREML.get('on_nonconvergence', 'fail')
                    if _on_nonconvergence == 'skip':
                        print(f"[ASReml] NON-CONVERGED / FAILED for phenotype {_asreml_trait!r} "
                              f"(task {task_num}/{total_tasks}): {asreml_exc!r}. "
                              f"ASREML['on_nonconvergence']='skip' - skipping this task entirely "
                              f"(no rows written for it).")
                        completed_units += units_per_task
                        _report_progress(completed_units, total_units, label='Skipped (ASReml non-convergence)')
                        return
                    raise RuntimeError(
                        f"[ASReml] Fit failed for phenotype {_asreml_trait!r} "
                        f"(task {task_num}/{total_tasks}): {asreml_exc!r}. Set "
                        f"ASREML['on_nonconvergence']='skip' to skip an unfittable trait instead "
                        f"of stopping the whole run (default 'fail' stops it)."
                    ) from asreml_exc
                finally:
                    end_stage(_asreml_stage)
                _asreml_append_fit_timing(
                    {'population': sample.loc[i, 'population'], 'phenotype': _asreml_trait,
                     'ratio': sample.loc[i, 'ratio'], 'sample': sample.loc[i, 'sample']},
                    _asreml_result.timing,
                )

                train, valid, test = apply_to_splits(
                    train, valid, test, _asreml_id_train, _asreml_id_valid, _asreml_id_test, _asreml_result,
                    population_train=_asreml_pop_train, population_valid=_asreml_pop_valid,
                    population_test=_asreml_pop_test,
                )
                _asreml_strategy_for_log = asreml_syntax.resolve_genotype_effect_strategy(ASREML)

                # ver4-25, R3 (design blueprint stage 3): resolve this
                # task's own adjusted-accuracy/Spearman/Top-k context
                # (companion ASReml fit(s) as needed - see `_asreml_get_
                # accuracy_metrics_context()`'s own block comment for
                # exactly which). `None` whenever the feature is off (the
                # default), in which case Step 10/11's own metric row
                # construction adds no new columns at all. The computed
                # ctx is cached LIVE (`_asreml_ctx_scenario_cache`, for
                # this process's own later finalisation) AND persisted as
                # `asreml_accuracy_ctx` accumulator rows (for a resumed
                # run, or a separate finalisation process, to resolve
                # from later) - see `_asreml_accuracy_ctx_resolver()`'s
                # own docstring for why this replaces the old scope
                # registry.
                if _asreml_accuracy_metrics_enabled:
                    _asreml_accuracy_ctx = _asreml_get_accuracy_metrics_context(
                        _asreml_trait, _asreml_strategy_for_log, _asreml_result,
                        mask_keys=_asreml_mask_keys, target_populations=_asreml_target_populations,
                        population_test=_asreml_pop_test,
                    )
                    _asreml_ctx_key = asreml_accuracy_metrics.scenario_key(
                        sample.loc[i, 'population'], _asreml_trait, sample.loc[i, 'ratio'], sample.loc[i, 'sample']
                    )
                    _asreml_ctx_scenario_cache[_asreml_ctx_key] = _asreml_accuracy_ctx
                    _asreml_accuracy_ctx_rows = asreml_accuracy_metrics.ctx_to_rows(
                        sample.loc[i, 'population'], _asreml_trait, sample.loc[i, 'ratio'], sample.loc[i, 'sample'],
                        _asreml_accuracy_ctx,
                    )
                    asreml_accuracy_ctx = pd.concat(
                        [asreml_accuracy_ctx, _asreml_accuracy_ctx_rows], axis=0, ignore_index=True
                    )

                # ver4-25, R8 (design blueprint SS2, Stage 6): this
                # task's own shared BLUE-reliability weighting context -
                # `None` whenever ASREML['blue_weighting'] is 'none' (the
                # default), in which case every model call below passes
                # `sample_weight_ctx=None` (a strict no-op, I11) and
                # Step 10's own scoring recompute is skipped entirely.
                _blue_weight_ctx = _asreml_get_blue_weight_context(
                    _asreml_result, _asreml_pop_test, id_train, id_valid, id_test,
                )
                if _blue_weight_ctx is not None:
                    # ver4-25, R8 diagnostics enhancement (prompted by a
                    # direct question: "is a lower reported accuracy under
                    # diagonal weighting expected or an error?"). The
                    # per-split min/max (mean-normalised) weight and the
                    # count of individuals whose own SE had to be imputed
                    # were already COMPUTED by `build_blue_weight_context()`
                    # (`ctx.diagnostics`) but never actually printed
                    # anywhere - only the fingerprint/n_train/n_valid/
                    # n_test were. Without this, there was no way to see,
                    # from the run log alone, whether this task's weights
                    # are a gentle correction (e.g. min/max close to 1) or
                    # a severe one (e.g. a 10x spread) - exactly the
                    # number needed to tell "weighting is legitimately
                    # emphasising/de-emphasising very differently-reliable
                    # individuals" apart from "something looks wrong with
                    # these weights". Printed unconditionally whenever a
                    # context was built at all (architecture doc SS18
                    # "log the fact, not just the number" discipline).
                    _d = _blue_weight_ctx.diagnostics or {}
                    _d_fit = _d.get('train', {}) or {}
                    _d_test = _d.get('test', {}) or {}
                    _d_cvrows = _d.get('cv_rows')
                    if _d_cvrows is not None:
                        if _blue_weight_ctx.w_fit_cv is not None:
                            print(f"[GP] [blue_weights] train+valid fitting weights built for the "
                                  f"cross-validated joint objective: n={_d_cvrows.get('n')} "
                                  f"range=[{_d_cvrows.get('min_weight')!r}, "
                                  f"{_d_cvrows.get('max_weight')!r}] (fold fits re-normalise them "
                                  f"to mean 1 over each fold's training rows).")
                        else:
                            print(f"[GP] NOTE: [blue_weights] train+valid fitting weights for the "
                                  f"cross-validated joint objective could not be built "
                                  f"({_d_cvrows.get('reason')}) - that objective will be "
                                  f"unavailable for this task.")
                    print(f"[GP] [blue_weights] task fingerprint={_blue_weight_ctx.fingerprint} "
                          f"mode={_blue_weight_ctx.mode} n_train={len(id_train)} "
                          f"n_valid={len(id_valid)} n_test={len(id_test)} "
                          f"train weight range=[{_d_fit.get('min_weight')!r}, "
                          f"{_d_fit.get('max_weight')!r}] "
                          f"(mean-normalised to 1; {_d_fit.get('n_imputed', 0)}/"
                          f"{_d_fit.get('n', len(id_train))} SE(s) imputed; "
                          f"effective_n={_d_fit.get('effective_n')!r}).")
                    # ver4-25, R8 bugfix follow-up (prompted by a direct
                    # report that weighted accuracy is LOWER than
                    # unweighted, especially for 'full' mode): the line
                    # above only ever logged the TRAINING split's own
                    # diagnostics - but the split that actually determines
                    # this task's own weighted Metric.csv row is the TEST
                    # split (`scoring_for(ctx, 'test')`, consumed by the
                    # central recompute a few hundred lines below). A
                    # small `effective_n` here (Kish's effective sample
                    # size - see blue_weights._kish_effective_n()/
                    # _kish_effective_n_matrix()'s own docstrings) relative
                    # to `n_test`, or (in 'full' mode specifically) a large
                    # `condition_number`, is the concrete, actionable
                    # signal that THIS task's own weighted test score is
                    # likely dominated by a couple of individuals rather
                    # than genuinely reflecting every test individual - a
                    # real statistical fragility of scoring against a
                    # DENSE, test-only covariance sub-block built from a
                    # small number of test individuals, not evidence by
                    # itself that the weights are wrong. Printed
                    # unconditionally, same discipline as the line above.
                    if _blue_weight_ctx.mode == 'diagonal':
                        print(f"[GP] [blue_weights] task fingerprint={_blue_weight_ctx.fingerprint} "
                              f"test weight range=[{_d_test.get('min_weight')!r}, "
                              f"{_d_test.get('max_weight')!r}] (mean-normalised to 1; "
                              f"{_d_test.get('n_imputed', 0)}/{_d_test.get('n', len(id_test))} "
                              f"SE(s) imputed; effective_n={_d_test.get('effective_n')!r} out of "
                              f"n_test={len(id_test)}).")
                    elif _blue_weight_ctx.mode == 'full':
                        print(f"[GP] [blue_weights] task fingerprint={_blue_weight_ctx.fingerprint} "
                              f"test GLS scoring matrix: condition_number="
                              f"{_d_test.get('condition_number')!r}, jitter="
                              f"{_d_test.get('jitter')!r}, effective_n="
                              f"{_d_test.get('effective_n')!r} out of n_test={len(id_test)}"
                              + (" - NOTE: effective_n is small relative to n_test, so this "
                                 "task's own weighted test score is dominated by very few "
                                 "individuals and may swing well away from the unweighted score "
                                 "for reasons unrelated to whether the weighting is correct."
                                 if (_d_test.get('effective_n') is not None and len(id_test) > 0
                                     and _d_test.get('effective_n') < 0.5 * len(id_test)) else "")
                              + ".")
                    # ver4-25, R8: the naive ensemble's OWN finalisation
                    # metric (models/ensemble.py::_metric(), called once
                    # per task over the WHOLE, possibly-resumed
                    # `result_test` accumulator - see that call site's own
                    # comment) needs a way to recover EVERY task's own
                    # weighting info regardless of which task actually
                    # produced a given group's rows, since a resumed run's
                    # in-memory `_blue_weight_ctx` only ever covers the
                    # CURRENT task. 'diagonal' mode solves this via the
                    # R3 sidecar (one scalar weight per test individual -
                    # cheap to persist as ordinary rows). 'full' mode's
                    # own dense per-scenario weight MATRIX has no such
                    # persistence wired up yet (disclosed scope decision,
                    # ver4-25 Change Summary Part 3/4 - see that
                    # document's own follow-on-session notes) - so it is
                    # deliberately NOT written to the sidecar here; every
                    # OTHER weighted quantity this run reports (every
                    # per-model Metric.csv row, every weighted-ensemble
                    # method's own row, hyperparameter tuning's own
                    # objective) is UNAFFECTED by this, since those all
                    # use the LIVE `_blue_weight_ctx` within their own
                    # task's dispatch, never a cross-task resolver - only
                    # the naive ensemble's OWN Pearson/MSE falls back to
                    # unweighted for a 'full'-mode run (a disclosed,
                    # narrower gap; `models/ensemble.py::_metric()`
                    # degrades to this automatically whenever the
                    # resolved ctx's own `test_weights` dict comes back
                    # empty - no special-casing needed there).
                    if _blue_weight_ctx.mode == 'diagonal':
                        # Persist this task's own per-test-individual
                        # diagonal weight onto the SAME R3 sidecar
                        # accumulator (`asreml_accuracy_ctx`,
                        # role='test_weight') that already makes
                        # Heritability/adjusted-accuracy/Spearman/Top-k
                        # resume-safe for the naive ensemble (see
                        # asreml_accuracy_metrics.ctx_resolver_from_
                        # frame()) - reusing that infrastructure rather
                        # than inventing a parallel one for this single
                        # extra per-id vector (ver4-25 Change Summary
                        # Part 3 SS7 deviation note). A '__meta__' row is
                        # required for ctx_resolver_from_frame() to ever
                        # look at this scenario's group at all - write one
                        # explicitly whenever the accuracy-metrics block
                        # above did NOT already write one for this task
                        # (accuracy metrics can be off while blue_
                        # weighting is on).
                        if _asreml_accuracy_ctx is None:
                            _meta_row = pd.DataFrame([{
                                'population': sample.loc[i, 'population'], 'phenotype': sample.loc[i, 'phenotype'],
                                'ratio': sample.loc[i, 'ratio'], 'sample': sample.loc[i, 'sample'],
                                'direction': asreml_accuracy_metrics.DEFAULT_DIRECTION,
                                'heritability': float('nan'), 'accuracy_numerator_mode': None,
                                'role': '__meta__', 'id': '', 'value': float('nan'),
                            }])
                            asreml_accuracy_ctx = pd.concat(
                                [asreml_accuracy_ctx, _meta_row], axis=0, ignore_index=True
                            )
                        # ver4-25, R8 bugfix (Requirements2.md, Error #1
                        # root cause): this used to be built as a SINGLE
                        # `pd.DataFrame([{...}])` row whose 'id'/'value'
                        # cells held the WHOLE `list(id_test)`/
                        # `list(_blue_weight_ctx.w_test)` objects, rather
                        # than one row per test individual - the same
                        # long-format convention every other role in this
                        # sidecar (`ctx_to_rows()`) already follows.
                        # Confirmed by direct reproduction: pandas stores
                        # a dict's list-valued entries as single-cell
                        # objects, not as one row per element, so the
                        # resulting frame was ONE row with 'id'/'value'
                        # holding embedded Python lists. That malformed
                        # row then reliably crashed
                        # `ctx_resolver_from_frame()`'s own `_values_for()`
                        # helper the moment it was ever resolved (`str()`
                        # of a whole list is not a real id; `float()` of a
                        # whole list raises `TypeError` immediately -
                        # reproduced for both a scalar and a 3-tuple
                        # `ratio`, the latter additionally failing even
                        # earlier, inside `frame.groupby(...)` itself,
                        # with `TypeError: unhashable type: 'list'`,
                        # since a `list` cell can never be a groupby key).
                        # Whenever this resolver is actually consulted -
                        # ASReml accuracy metrics enabled AND the naive
                        # ensemble selected, the two conditions under
                        # which genomic_prediction.py's own finalisation
                        # block ever builds one (see the `elif 'ensemble'
                        # in MODEL_BASE:` branch below) - this crashed the
                        # ENTIRE finalisation step for every
                        # 'diagonal'-mode weighted run, discarding the
                        # naive ensemble's own row (and, via the
                        # 'ensemble'-in-MODEL_BASE guard, sometimes taking
                        # the run's own Metric_summary.xlsx/DPT-summary
                        # generation down with it, since GP() re-raises
                        # after rolling back and saving whatever survived
                        # - see the caught exception's own `except
                        # Exception as _finalise_exc:` block far below).
                        # Fixed the same way `ctx_to_rows()` already
                        # builds its own 'blue'/'reference' role rows: a
                        # plain Python list of one dict PER INDIVIDUAL,
                        # handed to `pd.DataFrame(rows, columns=...)` in
                        # one shot.
                        _blue_weight_row_base = {
                            'population': sample.loc[i, 'population'], 'phenotype': sample.loc[i, 'phenotype'],
                            'ratio': sample.loc[i, 'ratio'], 'sample': sample.loc[i, 'sample'],
                            'direction': asreml_accuracy_metrics.DEFAULT_DIRECTION,
                            'heritability': float('nan'), 'accuracy_numerator_mode': None,
                        }
                        _blue_weight_rows = pd.DataFrame(
                            [dict(_blue_weight_row_base, role='test_weight', id=str(_tw_id), value=float(_tw_val))
                             for _tw_id, _tw_val in zip(id_test, _blue_weight_ctx.w_test)],
                            columns=list(asreml_accuracy_metrics.ASREML_ACCURACY_CTX_COLUMNS),
                        )
                        asreml_accuracy_ctx = pd.concat(
                            [asreml_accuracy_ctx, _blue_weight_rows], axis=0, ignore_index=True
                        )
                    elif _blue_weight_ctx.mode == 'full':
                        # ver4-25, R8 Stage 6 completion: persist this
                        # task's own dense TEST-split GLS scoring matrix
                        # (`W_test`) as long upper-triangle rows - the
                        # ONLY way the naive ensemble's own finalisation
                        # (run once, across the WHOLE assembled
                        # result_test, disconnected from any one task's
                        # own live BlueWeightContext - see
                        # models/ensemble.py::_metric()) and
                        # diversity_summary.py's own DPT terms can
                        # recover a 'full'-mode run's per-scenario
                        # weighting once this task's own in-memory
                        # context has gone out of scope (a resumed run,
                        # a separate Step 2 assemble process, or simply
                        # a later task). Mirrors the 'diagonal' branch
                        # above exactly, just writing to a SEPARATE
                        # accumulator (checkpoint_utils.RESULT_FILE_NAMES
                        # ['asreml_test_weights']) rather than reusing
                        # the R3 sidecar - a dense matrix has no natural
                        # one-scalar-per-individual row shape to share
                        # with that file's own per-individual roles. See
                        # blue_weights.weight_matrix_from_frame()'s own
                        # docstring for the exact read-side shape this
                        # produces (population, phenotype, ratio, sample,
                        # id1, id2, value - only the upper triangle,
                        # including the diagonal, is written; the matrix
                        # is symmetric by construction).
                        if _blue_weight_ctx.W_test is not None and len(id_test) > 0:
                            _tw_ids = [str(_id) for _id in id_test]
                            _tw_iu = np.triu_indices(len(_tw_ids))
                            _tw_ratio = sample.loc[i, 'ratio']
                            _tw_rows = [
                                {
                                    'population': sample.loc[i, 'population'],
                                    'phenotype': sample.loc[i, 'phenotype'],
                                    'ratio': _tw_ratio, 'sample': sample.loc[i, 'sample'],
                                    'id1': _tw_ids[_r], 'id2': _tw_ids[_c],
                                    'value': float(_blue_weight_ctx.W_test[_r, _c]),
                                }
                                for _r, _c in zip(*_tw_iu)
                            ]
                            asreml_test_weights = pd.concat(
                                [asreml_test_weights,
                                 pd.DataFrame(_tw_rows,
                                              columns=list(_ckpt.RESULT_FILE_BASE_COLUMNS['asreml_test_weights']))],
                                axis=0, ignore_index=True,
                            )
                        if not _asreml_full_weighting_naive_ensemble_note_printed[0]:
                            _asreml_full_weighting_naive_ensemble_note_printed[0] = True
                            print("[GP] NOTE: ASREML['blue_weighting']='full' is on - every "
                                  "model's own Metric.csv row, every weighted-ensemble method's "
                                  "own row, AND the naive ensemble's own row / "
                                  "diversity_summary.py's DPT terms are all GLS-weighted (the "
                                  "per-scenario test-weight matrix is persisted to ASReml/"
                                  "ASReml_test_weight_matrix.csv[.gz] for exactly this purpose). "
                                  "Printed once per run.")

                # ver4-25, R10 (design blueprint stage 3, "Variance-
                # components file redefined"): ONE training-only,
                # genotype-random-iid fit per task, for EVERY strategy
                # (no longer gated on 'blue_fixed' + an explicit opt-in -
                # `ASREML['concurrent_variance_model']` is now ignored,
                # superseded by this always-on computation). This is now
                # the ENTIRE content of ASReml_variance_components.csv -
                # the previous main-fit/companion varcomp rows are no
                # longer written at all (still available in memory on
                # `_asreml_result.varcomp`/`_href_result.varcomp` etc. for
                # Cullis H^2 and similar - just never persisted to this
                # file any more). Deliberately non-fatal: a failure here
                # logs a `[GP] NOTE:` and simply omits this task's row
                # from the file, never blocking the run (mirrors the
                # surrogate-interaction "informational side computation"
                # precedent).
                if (ASREML.get('concurrent_variance_model') or {}).get('enabled', False):
                    print(f"[GP] NOTE: ASREML['concurrent_variance_model']['enabled']=True is set "
                          f"but no longer read - the variance-components file is "
                          f"now always a single training-only, genotype-random-iid fit for every "
                          f"task/strategy; remove this key from your config.")
                # Requirements.md patch (this revision) - "For a fair
                # comparison [with the other ASReml fits within the same
                # iteration]": this concurrent/iid fit was ALWAYS
                # restricted to this task's own target-population
                # training individuals (`_asreml_id_train`) regardless of
                # `ASREML['use_entire_dataset']` - unlike every OTHER
                # ASReml fit this task runs (the main unmasked/BLUE fit -
                # `_asreml_get_unmasked_result()` - and both accuracy-
                # metrics companion fits - `_asreml_get_companion_
                # result()`), which all already pool every population's
                # raw rows via the IDENTICAL `_population_scope`
                # resolution when this setting is on. Left unfixed, this
                # produces a materially SMALLER, non-comparable fit
                # (observed: 630 raw rows / 172 individuals here vs. the
                # main fit's own 1907 rows / 1384 individuals in the same
                # run) - particularly consequential whenever spatial
                # factors (e.g. row/column, Experiment/Year) are
                # autocorrelated, since a much smaller fit sees a
                # different, non-comparable slice of that structure, and
                # this fit's own H2_cullis_train/variance-component
                # estimates feed straight into this task's own
                # adjusted-accuracy denominator.
                #
                # Fix: whenever `_asreml_use_entire_dataset` is True,
                # fit over the SAME entire-dataset raw rows every other
                # ASReml fit this task runs already does -
                # `fit_concurrent_variance_model()` already supports
                # exactly this ("keep the full grid, mask only the non-
                # training responses with NA" - its own `full_raw_
                # records`/`mask_non_training` parameters, previously
                # only ever auto-triggered for a grid-structured
                # residual); this simply asks for it explicitly here too,
                # instead of pre-filtering the raw rows down to this
                # task's own population before the function ever sees
                # them:
                #   - Unmasked strategies (`_asreml_mask_valid_test`
                #     False - the default, e.g. blue_fixed, and the
                #     scenario actually observed in the field): the main
                #     fit itself holds NOTHING out at all, so every
                #     individual across the whole trial counts as
                #     "training" here too - `ids_train` broadens to every
                #     distinct id in the entire dataset, so the row AND
                #     individual counts this fit reports genuinely match
                #     the main fit's own (`mask_non_training=True` is
                #     then a no-op - there is nothing left to mask).
                #   - Masked strategies: only THIS task's own held-out
                #     valid/test individuals are masked to NA (mirroring
                #     exactly what the main fit itself would mask under
                #     entire-dataset + masking) - every OTHER
                #     population's individual is still "training" here,
                #     exactly like it still contributes its own real
                #     response to the main fit.
                # `fit_concurrent_variance_model()`'s own internal
                # training-only counts/H2_cullis_train computation
                # already restricts back down to exactly the `ids_train`
                # passed here (see that function's own `train_records_
                # for_counts`/`train_values` filtering) - the SAME
                # "fit broad, then extract/restrict for the actual
                # statistic" pattern `_asreml_values_lookup(result,
                # population_test)` already applies for this task's own
                # Spearman-correlation/Top-k-coincidence-rate reference
                # values (via the companion-fit path above), so no
                # separate population-restriction is needed here beyond
                # passing the right `ids_train`.
                if _asreml_use_entire_dataset:
                    _asreml_train_raw = data_phenotype_raw_for_asreml
                    # ver4-26 bugfix (production failure, task 1/25 of
                    # BLUEs_adjusted_0.5): `.dropna()` BEFORE
                    # `.astype(str)` is required, not cosmetic. When a
                    # grid-structured residual is configured,
                    # `asreml_syntax.pad_residual_grid()` has already
                    # added NaN-filler rows to `data_phenotype_raw_for_
                    # asreml` for every missing (Range, Row) combination
                    # (see that function's own "residual grid completion
                    # added N filler row(s)" log line) - including a real
                    # `float('nan')` in the 'ID' column itself, since 'ID'
                    # is never one of the padded dimension columns.
                    # Pandas' `.astype(str)` special-cases NaN and leaves
                    # it as the float `nan` instead of converting it to
                    # the string `'nan'`, so without `.dropna()` first,
                    # `_asreml_all_ids` ends up a set of `str` AND one
                    # stray `float` - `sorted()` below then raises
                    # `TypeError: '<' not supported between instances of
                    # 'float' and 'str'`. Filler rows are structural
                    # padding only (NaN response, NaN every other column)
                    # and were never real individuals, so dropping them
                    # here is also the semantically correct fix, not just
                    # the safe one.
                    _asreml_all_ids = set(data_phenotype_raw_for_asreml['ID'].dropna().astype(str))
                    if _asreml_mask_valid_test:
                        _asreml_concurrent_ids_train = sorted(
                            _asreml_all_ids - set(_asreml_id_valid) - set(_asreml_id_test)
                        )
                    else:
                        _asreml_concurrent_ids_train = sorted(_asreml_all_ids)
                    _asreml_concurrent_mask_kwargs = {'mask_non_training': True}
                else:
                    _asreml_train_raw = data_phenotype_raw_for_asreml[
                        data_phenotype_raw_for_asreml['ID'].astype(str).isin(set(_asreml_id_train))
                    ].reset_index(drop=True)
                    _asreml_concurrent_ids_train = _asreml_id_train
                    _asreml_concurrent_mask_kwargs = {}  # unchanged - auto-detect from the grid flag
                try:
                    _asreml_concurrent_result = fit_concurrent_variance_model(
                        _asreml_train_raw, _asreml_trait, ASREML.get('genotype_col'), ASREML, 'iid',
                        workspace=ASREML.get('workspace', '4gb'),
                        max_update_rounds=ASREML.get('max_update_rounds', 20),
                        id_train=_asreml_concurrent_ids_train, full_raw_records=data_phenotype_raw_for_asreml,
                        threads=ASREML.get('threads'), **_asreml_concurrent_mask_kwargs,
                    )
                except Exception as _asreml_conc_exc:
                    print(f"[GP] NOTE: the training-only variance-components fit failed or was "
                          f"skipped for task {task_num}/{total_tasks} (population="
                          f"{sample.loc[i,'population']!r}, phenotype={_asreml_trait!r}): "
                          f"{_asreml_conc_exc!r} - the main BLUE/BLUP adjustment is unaffected; "
                          f"no row will be written to ASReml_variance_components.csv for this "
                          f"task.")
                    _asreml_concurrent_result = None
                if _asreml_concurrent_result is not None:
                    _asreml_append_fit_timing(
                        {'population': sample.loc[i, 'population'], 'phenotype': _asreml_trait,
                         'ratio': sample.loc[i, 'ratio'], 'sample': sample.loc[i, 'sample']},
                        _asreml_concurrent_result.timing,
                    )
                    _asreml_scenario_cols_varcomp = {
                        'population': sample.loc[i, 'population'], 'phenotype': _asreml_trait,
                        'ratio': sample.loc[i, 'ratio'], 'sample': sample.loc[i, 'sample'],
                    }
                    _asreml_conc_varcomp = _asreml_broadcast_scenario_cols(
                        _asreml_concurrent_result.varcomp.copy(), _asreml_scenario_cols_varcomp
                    )
                    # Design blueprint SS2.10 file schema convention
                    # (matches every other ASReml accumulator in this
                    # module): scenario-identifier block FIRST -
                    # `_asreml_broadcast_scenario_cols()` itself appends a
                    # genuinely NEW column at the END of the frame (plain
                    # pandas assignment semantics), so this reorder is
                    # required, not cosmetic-only, to match `checkpoint_
                    # utils.RESULT_FILE_BASE_COLUMNS['asreml_varcomp']`'s
                    # own column order (which an EMPTY run's own CSV
                    # header - `_with_header_fallback()` - always uses).
                    _asreml_conc_varcomp = _asreml_conc_varcomp[
                        list(_asreml_scenario_cols_varcomp.keys()) + [
                            c for c in _asreml_conc_varcomp.columns
                            if c not in _asreml_scenario_cols_varcomp
                        ]
                    ]
                    asreml_varcomp = pd.concat([asreml_varcomp, _asreml_conc_varcomp], axis=0, ignore_index=True)

            # Requirement 8: overwrite train/valid/test's LAST column
            # (the phenotype) in place, matched by ID - never by
            # position - with the converged BLUE/BLUP/GBLUP. Already
            # done above (both branches) via `apply_to_splits`.
            #
            # Requirement 10 (design blueprint SS2.10): accumulate this
            # task's own adjusted-phenotype and variance-component rows,
            # tagged with the same scenario-identifier block every other
            # result file uses, so they join 1:1 onto Basic_stats.csv /
            # Metric.csv on (population, phenotype, ratio, sample).
            _asreml_scenario_cols = {
                'population': sample.loc[i, 'population'], 'phenotype': _asreml_trait,
                'ratio': sample.loc[i, 'ratio'], 'sample': sample.loc[i, 'sample'],
            }
            # ver4-19 BUGFIX ("Length of values (3) does not match length
            # of index (1500)" - production failure, task 1/25 of
            # EasiGP_BLUEs_BLUPs_ASReml_0.5): see
            # `_asreml_broadcast_scenario_cols()`'s own docstring (module
            # scope, near `_is_bio_prior_model()`) for the full root-cause
            # explanation - `ratio` can be a 3-tuple, which a plain
            # `df[col] = value` broadcast mishandles.
            #
            # This task's own individuals only - `_asreml_result.values`
            # may hold every individual EasiGP has ever seen for this
            # trait (the new engine's cached, whole-file fit), so it is
            # filtered down to `_asreml_all_ids` here before being
            # accumulated, exactly reproducing the legacy engine's own
            # (already task-scoped) shape. Requirements.md R3: when the
            # result is (id, population)-keyed, filtering by id ALONE
            # would also pull in this same genotype's row(s) for every
            # OTHER population it happens to share (irrelevant to this
            # task) - so the filter joins on (id, population) instead,
            # using this task's own per-split population assignment
            # (`_asreml_pop_train`/`_asreml_pop_valid`/`_asreml_pop_test`,
            # resolved above).
            if 'population' in _asreml_result.values.columns:
                _asreml_id_pop_pairs = set()
                for _asreml_split_ids, _asreml_split_pop in (
                    (_asreml_id_train, _asreml_pop_train),
                    (_asreml_id_valid, _asreml_pop_valid),
                    (_asreml_id_test, _asreml_pop_test),
                ):
                    if _asreml_split_ids and _asreml_split_pop is not None:
                        _asreml_id_pop_pairs.update((_i, str(_asreml_split_pop)) for _i in _asreml_split_ids)
                _asreml_values_keys = list(zip(
                    _asreml_result.values['id'].astype(str), _asreml_result.values['population'].astype(str)
                ))
                _asreml_values_this_task = _asreml_result.values[
                    [k in _asreml_id_pop_pairs for k in _asreml_values_keys]
                ].copy()
            else:
                _asreml_values_this_task = _asreml_result.values[
                    _asreml_result.values['id'].astype(str).isin(set(_asreml_all_ids))
                ].copy()
            # ver4-19, Requirements.md R3: rename the response-unit key
            # column ('population' - one adjusted value per (genotype,
            # population) cell) to 'genotype_population' BEFORE
            # broadcasting the scenario-identifier block below, which
            # ALSO writes a 'population' column (this TASK's own
            # population/'X->Y' string) - `_asreml_broadcast_scenario_
            # cols()` overwrites any existing column of the same name, so
            # renaming afterwards would silently destroy the genotype-
            # population values it was trying to preserve.
            _asreml_has_genotype_population_col = 'population' in _asreml_values_this_task.columns
            if _asreml_has_genotype_population_col:
                _asreml_values_this_task = _asreml_values_this_task.rename(
                    columns={'population': 'genotype_population'}
                )
            _asreml_pheno_rows = _asreml_broadcast_scenario_cols(
                _asreml_values_this_task, _asreml_scenario_cols
            )
            # Design blueprint SS2.10 file schema - scenario-identifier
            # block first, matching Basic_stats.csv's own column order.
            if _asreml_has_genotype_population_col:
                _asreml_pheno_value_cols = ['id', 'genotype_population', 'split', 'value', 'estimand', 'std_error']
            else:
                _asreml_pheno_value_cols = ['id', 'split', 'value', 'estimand', 'std_error']
            _asreml_pheno_rows = _asreml_pheno_rows[
                list(_asreml_scenario_cols.keys()) + _asreml_pheno_value_cols
            ]
            # ver4-25, R10 ("Stop phenotype file"): only ever accumulated
            # when explicitly requested (ASREML['write_phenotype_file'] -
            # absent -> False, I11) - ASReml_phenotype.csv duplicates
            # `actual` in Prediction_result_*.csv and is read by no plot/
            # summary consumer (design blueprint R10 finding); skipping
            # the accumulation entirely (not just the eventual write) is
            # the memory saving the blueprint calls for.
            if _asreml_write_phenotype_file:
                asreml_pheno = pd.concat([asreml_pheno, _asreml_pheno_rows], axis=0, ignore_index=True)

            # ver4-25, R10: the main-fit/companion variance-component
            # rows this task's own `_asreml_result.varcomp` carried are NO
            # LONGER accumulated here at all - ASReml_variance_
            # components.csv's entire content is now the single,
            # always-on, training-only, genotype-random-iid fit appended
            # earlier in this same per-task block (the "R10 (design
            # blueprint stage 3...)" comment above) - see that block's own
            # comment for the full rationale ("Main-fit / companion /
            # legacy varcomp rows are no longer written", design blueprint
            # R10 "Chosen approach" item 2). `_asreml_result.varcomp` and
            # `_href_result.varcomp` etc. remain available in memory for
            # Cullis H^2 (the accuracy-metrics context, above) and similar
            # - only the PERSISTED file no longer carries them.

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
            with timed_stage('rf_filter'):
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
            # ver4-25, R8 (design blueprint SS2, Stage 6): 'none' for
            # every pre-ver4-25 run and every ver4-25 run with weighting
            # off - added at the END of the dict (append-only column
            # order, matching this file's own positional-hyperparameter
            # discipline elsewhere).
            'blue_weighting': _blue_weight_ctx.mode if _blue_weight_ctx is not None else 'none',
        }])], ignore_index=True)
        
        result_train_sample = pd.DataFrame()
        result_valid_sample = pd.DataFrame()
        result_test_sample = pd.DataFrame()
        
        # ver4-26, "Step 8.5" (design blueprint R2/R3/R5): JOINT
        # hyperparameter + ensemble-weight search - runs ONCE per task
        # (not once per model, unlike HP_TUNE), across every model in
        # MODEL_BASE (minus the 'ensemble' trigger token), producing this
        # task's own winning per-model hyperparameters
        # (_joint_final_params_by_model) and ensemble weights
        # (_joint_weights) BEFORE the per-model dispatch loop below
        # starts. R5: each joint-group model's own saved row is still
        # produced by a SEPARATE, full-fidelity confirmatory fit inside
        # that loop (see the new `elif _use_joint_params:` branch below),
        # through the SAME ModelTrialRunner canonical call path every
        # HP_TUNE'd model's own row already goes through - not by
        # re-using this search's own internal scoring-confirmatory fit
        # output - so there remains exactly ONE code path that ever turns
        # "(model, params) -> a saved row" into a saved row (this
        # codebase's own "LOCKSTEP WARNING" discipline: one call path,
        # kept in sync in exactly one place, rather than a second one).
        # This costs one extra full-fidelity fit per joint-group model
        # per task versus a design that threaded the search's own
        # scoring-fit outputs through directly - an accepted, bounded,
        # disclosed cost (Change Summary Sec.7/Sec.8), not a correctness
        # concern (same data, same winning params).
        #
        # A no-op (both dicts stay empty) whenever JOINT_W_OPT is
        # inactive or this task has no validation split - every existing
        # run, and every joint-enabled 'within'-scenario float-RATIO
        # task, sees no change here at all (I11).
        _joint_final_params_by_model = {}
        _joint_weights = {}
        _joint_best_valid_score = None
        _joint_tuning_source = None
        # ver4-28, R2/R6: reset alongside the dicts above - `_joint_confirm_
        # results` (R2) is a {model_name: full runner dict} populated only
        # when `reuse_confirmatory_fit` is on and this task's own search did
        # not hit the per_scenario cache; `_joint_details` (R6) is this
        # task's own `tune_joint_hyperparameters_and_weights(return_details=
        # True)` provenance dict. Both stay at these empty/None defaults for
        # every task where JOINT_W_OPT is inactive or has no validation
        # split (I11).
        _joint_confirm_results = {}
        _joint_details = None
        # ver4-38, R2.b2: the library ensemble's members (primaries and extras) of THIS task's
        # search, or [] (library_ensemble off, a cache-hit task, a failed search, or frames that
        # differ from the dispatch loop's). Step 11 reads it; [] reproduces ver4-37 exactly.
        _joint_library_members = []
        # BUGFIX (hyperparameter.csv recording): the joint search's own
        # per-model, per-field TUNED VALUES (label -> final value, the
        # exact same shape tune_model_hyperparameters()'s own
        # `tuned_values` already has for single-model HP_TUNE - see that
        # function's docstring) - previously never captured at all for
        # the joint path, so the `elif _use_joint_params:` hp_record row
        # below carried only the algorithm name + valid score and NONE
        # of the actual tuned hyperparameter columns every HP_TUNE'd
        # model's own row already has. Populated below whenever the
        # joint search actually ran or was reused from cache; stays {}
        # (no columns added) whenever JOINT_W_OPT is inactive - I11.
        _joint_tuned_values_by_model = {}
        # ver4-56 (E8): Step 8.5 is timed as stage 'joint_opt' (it used to fall into `other=`
        # of the [timing] line); closed right after the block, below.
        _joint_stage = (begin_stage('joint_opt')
                        if _joint_active and valid.shape[0] != 0 else None)
        if _joint_active and valid.shape[0] != 0:
            _joint_all_selected_models = [m for m in MODEL_BASE if m != 'ensemble']
            # ver4-27, R1: an optional 'group_models' key inside
            # HYPERPARAMETERS_JOINT_OPT (a list of base model names) restricts
            # the joint search to a user-chosen SUBSET of this task's selected
            # models, instead of always every one of them. A model left out of
            # that subset simply isn't joint-tuned - it keeps running with its
            # own configured HPARAMETERS unchanged, exactly as it would with
            # JOINT_W_OPT off (I11: omitting 'group_models', or leaving it
            # None/empty, reproduces the previous "every selected model"
            # behaviour bit for bit). This exists because the joint search's
            # own combined hyperparameter+weight space grows with EVERY
            # tunable field of EVERY included model (see
            # models/joint_tuning.py's own JOINT_SEARCH_DIM_WARN_THRESHOLD) -
            # letting a user joint-tune a small, deliberately-chosen subset
            # (e.g. 2-3 complementary models) while the rest of MODEL_BASE
            # runs normally is what keeps that space small enough for a
            # realistic search budget to actually converge, without forcing a
            # smaller MODEL_BASE for the whole run.
            _joint_requested_group = (HYPERPARAMETERS_JOINT_OPT or {}).get('group_models')
            if _joint_requested_group:
                _joint_requested_set = {base_of(m) for m in _joint_requested_group}
                _joint_group_models = [
                    m for m in _joint_all_selected_models if base_of(m) in _joint_requested_set
                ]
                _joint_missing = _joint_requested_set - {base_of(m) for m in _joint_group_models}
                if _joint_missing:
                    print(f"[GP] Task {task_num}/{total_tasks} | NOTE: "
                          f"HYPERPARAMETERS_JOINT_OPT['group_models'] names "
                          f"{sorted(_joint_missing)!r}, not among this task's selected "
                          f"MODEL_BASE {_joint_all_selected_models!r} - ignored; the joint "
                          f"search only ever covers models actually selected for this run.")
            else:
                _joint_group_models = _joint_all_selected_models
            if len(_joint_group_models) < 2:
                print(f"[GP] Task {task_num}/{total_tasks} | NOTE: JOINT_W_OPT is set but "
                      f"fewer than 2 models are available for the joint search "
                      f"({_joint_group_models!r}, out of this task's selected "
                      f"{_joint_all_selected_models!r}) - a joint hyperparameter+weight search "
                      f"needs at least 2 models to combine; skipped for this task. Every "
                      f"model here falls back to its own configured HPARAMETERS unchanged.")
            else:
                _joint_cache_key = (
                    sample.loc[i, 'population'], sample.loc[i, 'phenotype'],
                    sample.loc[i, 'ratio'], None, 'JOINT',
                )
                _joint_scope = _compute_resources['hp_tune_scope']
                _joint_cache_hit = (
                    _joint_tuning_cache.get(_joint_cache_key) if _joint_scope == 'per_scenario' else None
                )

                # BUGFIX (found while self-checking this delivery, before
                # any external review): `_joint_runner` is needed by the
                # per-model dispatch loop's own `elif _use_joint_params:`
                # branch below for EVERY joint-group model's confirmatory
                # fit, regardless of whether "Step 8.5" itself searched
                # fresh or reused a per-scenario cache hit - so it must be
                # built here, before the cache-hit/miss branch below,
                # never only inside the non-cache-hit ('else') branch (an
                # earlier draft of this block built it there only, which
                # would raise NameError on this task's own dispatch loop
                # the first time HP_TUNE_SCOPE='per_scenario' actually
                # produced a cache hit).
                _joint_group_has_bio_prior = any(
                    _is_bio_prior_model(base_of(m)) for m in _joint_group_models
                )
                for _joint_bp_name in dict.fromkeys(
                        base_of(m) for m in _joint_group_models if _is_bio_prior_model(base_of(m))):
                    _prewarm_bio_graph_for(_joint_bp_name)   # ver4-61, R1
                _joint_runner = ModelTrialRunner(
                    RESULT_NAME, R_PATH, _compute_resources, GENOTYPE_FORMAT,
                    bio_prior_ctx=_bio_prior_trial_ctx(list(dict.fromkeys(
                        base_of(m) for m in _joint_group_models if _is_bio_prior_model(base_of(m))
                    ))) if _joint_group_has_bio_prior else None,
                    sample_weight_ctx=_blue_weight_ctx,
                )

                if _joint_cache_hit is not None:
                    _joint_final_params_by_model = dict(_joint_cache_hit['final_params_by_model'])
                    _joint_weights = dict(_joint_cache_hit['weights'])
                    _joint_best_valid_score = _joint_cache_hit['best_valid_score']
                    _joint_tuning_source = f"reused:{sample.loc[i, 'sample']}"
                    # BUGFIX (hyperparameter.csv recording): cached alongside
                    # the params/weights above so a per_scenario cache-hit
                    # replicate's own hp_record row also carries the tuned
                    # values, not just the algorithm name/score (see this
                    # dict's own declaration comment above _joint_active).
                    _joint_tuned_values_by_model = {
                        m: dict(v) for m, v in _joint_cache_hit.get('tuned_values_by_model', {}).items()
                    }
                    print(f"[GP] Task {task_num}/{total_tasks} | Joint Bayesian optimisation: "
                          f"reused tuned hyperparameters+weights from replicate "
                          f"{_joint_cache_hit['source_sample']!r} (HP_TUNE_SCOPE='per_scenario').")
                else:
                    # Resolve each joint-group model's own task-local
                    # "untuned baseline" from HPARAMETERS_BASELINE (the
                    # SAME frozen anchor HP_TUNE's own non-warm-start
                    # default search starts from - see that comment near
                    # MODEL_RUN above; joint mode has no warm-start
                    # concept of its own) - handling a bio-prior model's
                    # per-phenotype dict exactly as the HP_TUNE dispatch
                    # branch below does.
                    _joint_base_params_by_model = {}
                    for _jm in _joint_group_models:
                        _jm_base = base_of(_jm)
                        _entry = HPARAMETERS_BASELINE[_jm_base]
                        if _is_bio_prior_model(_jm_base) and isinstance(_entry, dict):
                            _current_phenotype = sample.loc[i, 'phenotype']
                            if _current_phenotype not in _entry:
                                raise KeyError(
                                    f"HPARAMETERS[{_jm_base!r}] is a per-phenotype dict but has "
                                    f"no entry for phenotype {_current_phenotype!r} - cannot "
                                    f"joint-tune it. Add one, or pass a single flat params list "
                                    f"instead to share one network across every phenotype in "
                                    f"PHENOTYPE."
                                )
                            _joint_base_params_by_model[_jm_base] = _entry[_current_phenotype]
                        else:
                            _joint_base_params_by_model[_jm_base] = _entry

                    # `_joint_runner` (ModelTrialRunner.__call__ takes
                    # model_name per call - see its own docstring, so ONE
                    # instance correctly serves every joint-group model by
                    # name) was already built above, before this
                    # cache-hit/miss branch - reused here as this search's
                    # own `run_model_fn`.

                    try:
                        _joint_inner_seed_base = int(sample.loc[i, 'sample'])
                    except (TypeError, ValueError):
                        _joint_inner_seed_base = 0

                    # HYPERPARAMETERS_JOINT_OPT (ver4-26, R2 - design
                    # decision, Change Summary Sec.7): a single flat dict
                    # - not a positional list like HYPERPARAMETERS_OPT's
                    # per-W_OPT-method entries - holding this run's own
                    # joint-search budget ('n_iter'/'init_points'/'seed',
                    # the SAME shape a single HP_TUNE algorithm's own
                    # budget entry already uses) plus an optional
                    # 'objective_mode' key ('ensemble_score', the default,
                    # or 'dpt_ratio' - popped out here so the dict handed
                    # to `tune_joint_hyperparameters_and_weights` as
                    # `budget_kwargs` holds only budget keys). None/empty
                    # (I11) reproduces that function's own defaults.
                    _joint_opt_cfg = dict(HYPERPARAMETERS_JOINT_OPT) if HYPERPARAMETERS_JOINT_OPT else {}
                    _joint_objective_mode = _joint_opt_cfg.pop('objective_mode', 'ensemble_score')
                    # ver4-27, R1: 'group_models' was only ever a selection
                    # instruction for THIS block (resolved into
                    # _joint_group_models above) - popped here, exactly like
                    # 'objective_mode' just above, so it never leaks into
                    # budget_kwargs and gets misread as a search-budget
                    # setting by tune_joint_hyperparameters_and_weights().
                    _joint_opt_cfg.pop('group_models', None)

                    # ver4-28, P2 (design blueprint SS1a): `reuse_confirmatory_
                    # fit` (R2) hands this search's own confirmatory fit
                    # straight to the per-model dispatch loop below, which
                    # assumes it was fit on train_pruned_full/valid_pruned_
                    # full/test_pruned_full - true today (train/valid/test are
                    # assigned from those three splits immediately before this
                    # block runs, and nothing between there and here touches
                    # them), asserted here, log-only, so a future change that
                    # breaks it is caught rather than silently mis-attributing
                    # a row to the wrong split.
                    _joint_p2_ok = (
                        train.index.equals(train_pruned_full.index) and train.shape == train_pruned_full.shape
                        and valid.index.equals(valid_pruned_full.index) and valid.shape == valid_pruned_full.shape
                        and test.index.equals(test_pruned_full.index) and test.shape == test_pruned_full.shape
                    )
                    if not _joint_p2_ok:
                        print(f"[GP] Task {task_num}/{total_tasks} | NOTE (P2): this task's own "
                              f"train/valid/test at Step 8.5 differ from train_pruned_full/"
                              f"valid_pruned_full/test_pruned_full used by the per-model dispatch "
                              f"loop below - reuse_confirmatory_fit is treated as unsafe for this "
                              f"task even if requested; every joint-group model falls back to its "
                              f"own legacy confirmatory refit.")

                    # ver4-28, R1: fit units fan out across up to
                    # min(N_JOBS, q*K) worker processes when 'parallel_trials'
                    # is set AND this run's own global HP_TUNE_PARALLEL_TRIALS
                    # switch is on - the SAME double gate the single-model
                    # HP_TUNE dispatch above already applies
                    # (_compute_resources['hp_tune_parallel_trials']).
                    _joint_parallel_trials = (
                        bool(_joint_opt_cfg.get('parallel_trials'))
                        and _compute_resources['hp_tune_parallel_trials']
                    )
                    _joint_q = min(int(_compute_resources.get('hp_tune_bayes_batch_max') or 4), 4)
                    _joint_width = (
                        max(1, min(int(_compute_resources.get('n_jobs') or 1),
                                   _joint_q * len(_joint_group_models)))
                        if _joint_parallel_trials else 1
                    )
                    _joint_trial_worker_resources = _trial_worker_resources_for(_compute_resources, _joint_width)
                    # ver4-35, R1.e2: install the GPU arbiter (only with
                    # ACCELERATION['gpu_slots'] != 'legacy') BEFORE the workers
                    # that must inherit it are described.
                    if _joint_width > 1:
                        ensure_gpu_arbiter(_compute_resources)
                    _joint_worker_init_args = (
                        RESULT_NAME, R_PATH, _joint_trial_worker_resources, get_gpu_semaphore(),
                    )
                    if _joint_width > 1:
                        print(f"[GP] Task {task_num}/{total_tasks} | Joint Bayesian optimisation: "
                              f"fit units will fan out across up to {_joint_width} worker "
                              f"process(es) (parallel_trials=True) - per-worker budget reduced to "
                              f"n_jobs={_joint_trial_worker_resources.get('n_jobs')}.")

                    try:
                        (_joint_final_params_by_model, _joint_weights,
                         _joint_best_valid_score, _joint_elapsed,
                         _joint_tuned_values_by_model, _joint_details) = tune_joint_hyperparameters_and_weights(
                            group_models=[base_of(m) for m in _joint_group_models],
                            train=train, valid=valid, test=test,
                            run_model_fn=_joint_runner, RESULT_NAME=RESULT_NAME,
                            budget_kwargs=_joint_opt_cfg,
                            hparam_specs=HPARAM_SPECS,
                            base_params_by_model=_joint_base_params_by_model,
                            objective_mode=_joint_objective_mode,
                            # ver4-26, R3: reuses the SAME five HP_TUNE
                            # compute-resource cost-control levers rather
                            # than adding new ones - see this run's own
                            # "[GP] Compute resources resolved:" log block
                            # for their resolved values.
                            bayes_batch=_compute_resources['hp_tune_bayes_batch'],
                            bayes_batch_max=_compute_resources['hp_tune_bayes_batch_max'],
                            bayes_liar=_compute_resources['hp_tune_bayes_liar'],
                            # ver4-28, R1: n_jobs>1 only when 'parallel_trials'
                            # is set (see _joint_width above) - n_jobs=1
                            # (serial, in-process) reproduces every pre-
                            # ver4-28 run's exact behaviour byte for byte.
                            n_jobs=_joint_width, worker_init=_trial_worker_init,
                            worker_init_args=_joint_worker_init_args,
                            selection_margin=_compute_resources['hp_tune_selection_margin'],
                            inner_split_seed_base=_joint_inner_seed_base,
                            return_details=True,
                            # ver4-34, R2.b: the facts that decide whether
                            # HYPERPARAMETERS_JOINT_OPT['objective_data']=
                            # 'cv_train_valid' may run for THIS task (it is
                            # ineligible under BLUE weighting unless the context
                            # carries the train+valid fitting weights (ver4-52,
                            # 'blue_cv_weights'), with a bio-prior
                            # model in the group, or when the validation rows
                            # do not come from the training population - see
                            # tune_joint_hyperparameters_and_weights()). With
                            # the default objective_data='valid' the engine
                            # does not read it.
                            eligibility_ctx={
                                'scenario': SCENARIO,
                                'between_validation_source': _between_validation_source,
                                'blue_weighting': _blue_weight_ctx is not None,
                                # ver4-52: True when the context carries the train+valid
                                # fitting weights, so the CV objective (and with it
                                # search_strategy='neural_stack') stays available under BLUE
                                # weighting; the fold fits then use `_FitUnit.cv_weights`.
                                'blue_cv_weights': (_blue_weight_ctx is not None
                                                    and getattr(_blue_weight_ctx, 'w_fit_cv', None) is not None),
                                'has_bio_prior': _joint_group_has_bio_prior,
                            },
                            # ver4-36, R2.d: where the (opt-in, library_dump) library
                            # dump is written - a diagnostic sidecar, never read back.
                            diagnostics_ctx={
                                'task_tag': (f"{sample.loc[i, 'population']}_"
                                             f"{sample.loc[i, 'phenotype']}_"
                                             f"{sample.loc[i, 'ratio']}_{sample.loc[i, 'sample']}"),
                                'result_dir': f"./Result/{RESULT_NAME}",
                            },
                        )
                    except Exception as _joint_exc:
                        print(f"[GP] Task {task_num}/{total_tasks} | Joint Bayesian optimisation "
                              f"FAILED ({_joint_exc!r}) - every selected model falls back to its "
                              f"own configured HPARAMETERS unchanged for this task (no joint "
                              f"hyperparameters/weights applied).")
                        if not bool((HYPERPARAMETERS_JOINT_OPT or {}).get('safety_nets', True)):
                            # ver4-44, R1: a pure-joint evaluation must be able to tell this task
                            # apart: it has NO joint row (every model ran at its configured
                            # hyperparameters), so tools/joint_replay.py drops it.
                            print(f"[GP] Task {task_num}/{total_tasks} | WARN: "
                                  f"safety_nets=False and the joint search failed - this task "
                                  f"has no 'Joint Bayesian optimisation' row; the evaluation "
                                  f"excludes it.")
                        _joint_final_params_by_model = {}
                        _joint_weights = {}
                        _joint_best_valid_score = None
                        _joint_tuned_values_by_model = {}
                        _joint_details = None
                        _joint_library_members = []
                    else:
                        _joint_tuning_source = 'searched'
                        # ver4-38, R2.b2: keep the library ensemble's members only when the
                        # search frames are the dispatch loop's own (the same P2 condition as
                        # `reuse_confirmatory_fit`): the extras' predictions are positional.
                        _joint_library_members = list(
                            (_joint_details or {}).get('library_members') or [])
                        if _joint_library_members and not _joint_p2_ok:
                            print(f"[GP] Task {task_num}/{total_tasks} | NOTE: Joint Bayesian "
                                  f"optimisation library_ensemble produced "
                                  f"{len(_joint_library_members)} member(s) but this task's "
                                  f"frames differ from the per-model dispatch loop's (P2 above) - "
                                  f"the extras are ignored and the per-model weight sums weight "
                                  f"the primaries only.")
                            _joint_library_members = []
                        # ver4-28, R2: only store the confirmatory fit's own
                        # FULL runner dicts for reuse when requested AND P2's
                        # own log-only safety check passed for this task -
                        # absent either, the per-model dispatch loop's own
                        # `elif _use_joint_params:` branch takes its legacy
                        # refit path for every joint-group model (R2's own
                        # disclosed "absent -> legacy refit" contract).
                        if bool(_joint_opt_cfg.get('reuse_confirmatory_fit')) and _joint_p2_ok:
                            _joint_confirm_results = dict((_joint_details or {}).get('confirm_results') or {})
                        if _joint_scope == 'per_scenario':
                            _joint_tuning_cache[_joint_cache_key] = {
                                'final_params_by_model': dict(_joint_final_params_by_model),
                                'weights': dict(_joint_weights),
                                'best_valid_score': _joint_best_valid_score,
                                'source_sample': sample.loc[i, 'sample'],
                                # BUGFIX (hyperparameter.csv recording): see
                                # this dict's own declaration comment above.
                                'tuned_values_by_model': {
                                    m: dict(v) for m, v in _joint_tuned_values_by_model.items()
                                },
                            }
                        print(f"[GP] Task {task_num}/{total_tasks} | Joint Bayesian optimisation "
                              f"finished: {len(_joint_group_models)} model(s), valid "
                              f"objective={_joint_best_valid_score:.4f} (took "
                              f"{_joint_elapsed:.1f}s) -> weights={_joint_weights}")
        if _joint_stage is not None:
            end_stage(_joint_stage)

        # Requirements.md (multi-option update), item 4 - Spearman
        # correlation and Top-k Coincidence are now ALWAYS reported (like
        # Pearson correlation/MSE), against the SAME test target every
        # model's Pearson/MSE below is scored on (`test.iloc[:, -1]` - the
        # raw phenotype without ASReml, or this option's own BLUE/BLUP
        # target once Step 6b has substituted it). This task's row-level
        # ctx is therefore ALWAYS a dict from here on: a COPY of the
        # ASReml adjusted-accuracy ctx (`enabled=True`) when that feature
        # is on, or a bare `{'enabled': False}` one otherwise, plus
        # `direction` and the `{id: test target}` lookup that
        # `asreml_accuracy_metrics.compute_accuracy_metrics_row()` reads.
        # A COPY - the cached/persisted original (`_asreml_ctx_scenario_
        # cache`/`asreml_accuracy_ctx` accumulator) never carries the
        # per-task `actual_values`, keeping that sidecar exactly as small
        # as before.
        _row_ctx = dict(_asreml_accuracy_ctx) if _asreml_accuracy_ctx else {'enabled': False}
        _row_direction = _metric_direction_by_phenotype.get(
            sample.loc[i, 'phenotype'], asreml_accuracy_metrics.DEFAULT_DIRECTION)
        if _row_direction not in asreml_accuracy_metrics.VALID_DIRECTIONS:
            _row_direction = asreml_accuracy_metrics.DEFAULT_DIRECTION
        _row_ctx['direction'] = _row_direction
        _row_ctx['actual_values'] = dict(zip(
            [str(_v) for _v in list(id_test)],
            np.asarray(test.iloc[:, -1], dtype=float).tolist(),
        ))
        _asreml_accuracy_ctx = _row_ctx

        # ver4-35, R1.f - asynchronous model lane (opt-in:
        # ACCELERATION['async_model_lane']; off = this block does nothing and
        # every model below runs inline, exactly as in ver4-34).
        #
        # Why: the models of a task run strictly one after the other, so the
        # CPU models and the bio-prior GAT never work at the same time. A lane
        # chain fits lane-capable models (model_registry.LANE_CAPABLE) in one
        # persistent worker process WHILE this loop keeps fitting the other
        # models; each lane result is collected at that model's own turn
        # (below), so Step 10 and Step 11 see the same rows in the same order.
        # Stream hand-off, fallback and the planner: async_lane.py.
        def _plan_task_lane():
            """This task's :class:`async_lane.TaskLane`, or None when the lane
            is off, impossible or could not overlap anything."""
            if not _compute_resources.get('accel_async_model_lane'):
                return None
            _names = [m for m in MODEL_RUN if m != 'ensemble']
            if not any(lane_capable(m) for m in _names):
                return None
            import multiprocessing as _mp
            _reason = None
            if _dispatch_filter is not None:
                _reason = ("a model-dispatch filter is active (model-level fan-out already "
                           "overlaps models)")
            elif int(_compute_resources.get('n_cpu_workers') or 1) > 1:
                _reason = "task-level worker fan-out is active (N_CPU_WORKERS > 1)"
            elif _mp.current_process().daemon:
                _reason = "this process is a daemonic worker and may not spawn children"
            if _reason is not None:
                _note_lane_off_once(_reason)
                return None

            def _tuned_here(name):
                _cfg = (HP_TUNE or {}).get(base_of(name))
                return bool(_cfg) and bool(_cfg.get('enabled')) and valid_pruned_full.shape[0] != 0

            _eligible = {m for m in _names
                         if lane_capable(m) and base_of(m) == m and not _tuned_here(m)
                         and m not in _joint_final_params_by_model}
            # A model that tunes in this loop draws from the global streams in
            # amounts nobody declared (its search), so it counts as "any".
            _streams = {m: (None if _tuned_here(m) else global_rng_streams(base_of(m)))
                        for m in _names}
            _mode = _compute_resources.get('accel_async_lane_rng', 'exact')
            _slots = 1
            if _mode == 'per_job_seed':
                _slots = max(1, int(_compute_resources.get('n_gpu_slots') or 1))
            _chains = async_lane.schedule_lane(
                _names, _eligible, _streams, per_job_seed=(_mode == 'per_job_seed'),
                parallel_slots=_slots)
            _label = f"Task {i - idx*interval + 1}/{total_tasks}"
            if not _chains:
                print(f"[lane] {_label}: no overlap is possible for this model list "
                      f"(lane-capable models run inline).")
                return None

            _lane_resources = _lane_worker_resources(_compute_resources)
            ensure_gpu_arbiter(_compute_resources)
            _lane = async_lane.acquire_lane(
                (RESULT_NAME, R_PATH, _lane_resources, get_gpu_semaphore()),
                n_slots=min(_slots, len(_chains)))
            if not _lane.usable:
                return None
            _cap_bytes = int(
                _compute_resources.get('accel_async_lane_max_payload_mb', 2048)) * 1024 * 1024

            def _lane_bio_ctx(model_run_name):
                # PLINK: the same lean, per-instance context the tuning runner
                # gets. CSV: built here from the task's shared frames, because
                # _bio_prior_trial_ctx() reads bio_prior_pools /
                # bio_prior_merge_pools, which only the PLINK path assigns
                # (it raises NameError for a CSV run - pre-existing, ver4-34;
                # reported in the change summary, not changed here).
                if GENOTYPE_FORMAT == 'plink':
                    return _bio_prior_trial_ctx([model_run_name])
                return {
                    'bio_prior_pools': {},
                    'train_unpruned': train_unpruned, 'valid_unpruned': valid_unpruned,
                    'test_unpruned': test_unpruned,
                    'bio_prior_merge_pools': {},
                    'full_marker_pool': full_marker_pool,
                    'bim_marker_info_path': bim_marker_info_path,
                    'network_cache': _bio_prior_merge_cache,
                    'phenotype_name': sample.loc[i, 'phenotype'],
                }

            def _build_lane_job(model_run_name):
                _params = list(_resolve_bio_prior_params(model_run_name))
                _ctx = _lane_bio_ctx(model_run_name)
                _bytes = async_lane.payload_bytes(_ctx)
                if _bytes > _cap_bytes:
                    raise RuntimeError(
                        f"the job's data is about {_bytes / 2**20:.0f} MB, above "
                        f"ASYNC_LANE_MAX_PAYLOAD_MB={_cap_bytes // 2**20}")
                _job = async_lane.LaneJob(
                    model_run=model_run_name, base_model=model_run_name, params=_params,
                    runner_init=dict(result_name=RESULT_NAME, r_path=R_PATH,
                                     compute_resources=_lane_resources,
                                     genotype_format=GENOTYPE_FORMAT,
                                     sample_weight_ctx=_blue_weight_ctx))
                return _job, f"bio:{model_run_name}", {'frames': None, 'bio_prior_ctx': _ctx}

            def _streams_of(model_run_name):
                _declared = global_rng_streams(base_of(model_run_name))
                return async_lane.ALL_STREAMS if _declared is None else _declared

            print(f"[lane] {_label}: {len(_chains)} chain(s) "
                  + "; ".join(f"{c.models} released before model #{c.release_index + 1}, "
                              f"overlapping {c.overlap}" for c in _chains)
                  + f" (rng={_mode!r}, lane worker threads="
                  f"{_compute_resources.get('accel_async_lane_threads')!r}).")
            return async_lane.TaskLane(
                _lane, _label, _chains, _build_lane_job,
                f"{RESULT_NAME}|{i}|{os.getpid()}|{time.time_ns()}", _mode, _streams_of, i)

        _task_lane = _plan_task_lane()
        _lane_pos = -1

        # ver4-57 (R5, Step 10 capture): the opt-in Iterative ensemble (Step 11c below) needs
        # each model's Step 9 fit as it stands - frames, hyperparameters, predictions,
        # marker effects and k=2 interactions. They are collected here, after each model's own
        # Step 10 accumulation, only when the feature is on, a validation split exists (the
        # acceptance rule reads validation rows only, I7) and this is a full-model run
        # (a MODEL_DISPATCH_FILTER sub-run never sees every member, I9).
        _iter_members = {}
        _iter_has_valid = (type(sample.loc[i, 'ratio']) is tuple
                           or (SCENARIO == 'between' and _between_has_validation))
        _iter_capture = (_iter_cfg is not None and _dispatch_filter is None and _iter_has_valid)
        # ver4-77 (R2): the Super learner needs the same Step 9 captures but NOT a validation
        # split (it then uses the training rows only). `_member_capture` is the union of the two
        # reasons to capture; `_iter_skip_of` remembers, per member, why Step 11c must not use it.
        _sl_capture = (_sl_cfg is not None and _dispatch_filter is None)
        _member_capture = _iter_capture or _sl_capture
        _iter_skip_of = {}
        if _iter_cfg is not None and not _iter_capture:
            print(f"[GP] Task {i - idx*interval + 1}/{total_tasks} | NOTE: ITERATIVE_ENSEMBLE is "
                  f"set but this task "
                  + ("is a MODEL_DISPATCH_FILTER sub-run (the members are fitted in different "
                     "processes)" if _dispatch_filter is not None else
                     "has no validation split (the iterative ensemble accepts a round on "
                     "validation rows only, and a test-only task has none)")
                  + " - no 'Iterative ensemble' row is produced for it.")

        # Prediction model implementation
        for jj in range(len(MODEL_RUN)):
            if MODEL_RUN[jj] == 'ensemble':
                continue

            # ver4-35, R1.f: submit every lane chain whose release point is
            # this model (None when the lane is off - nothing happens).
            _lane_pos += 1
            if _task_lane is not None:
                _task_lane.release_due(_lane_pos)

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
            # ver4-33, R1.a: this model's wall time, tuning included, as stage
            # 'model:<name>' (closed just before the "finished" line below;
            # `gpu_wait` inside it is split out by the GPU slot itself).
            _model_stage = begin_stage(f"model:{MODEL_RUN[jj]}")
            _rtele.model_begin(MODEL_RUN[jj], base=base_model_name)
            _tele_gpu_wait0 = (current_stage_timer().seconds('gpu_wait')
                               if current_stage_timer() is not None else 0.0)

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
            # ver4-85: with SUPER_LEARNER['oof_tuning'] the candidates are scored out of fold over
            # the training lines, so no validation split is needed. A bio-prior model (never
            # fitted out of fold, I6) or a training split below 10 x folds rows cannot be tuned
            # that way: it falls back to holdout tuning when the task has a validation split,
            # and is left untuned otherwise (NOTE either way).
            _tuning_oof_folds = None
            if _sl_oof_tune and bool(hp_tuning_cfg) and hp_tuning_cfg.get('enabled'):
                _oof_tune_skip = None
                if _is_bio_prior_model(base_model_name):
                    _oof_tune_skip = "it is a bio-prior model (never fitted out of fold, I6)"
                elif train.shape[0] < 10 * int(_sl_cfg['folds']):
                    _oof_tune_skip = (f"the training split has {train.shape[0]} row(s), fewer "
                                      f"than 10 x folds ({10 * int(_sl_cfg['folds'])})")
                if _oof_tune_skip is None:
                    try:
                        _oof_rep = int(sample.loc[i, 'sample'])
                    except (TypeError, ValueError):
                        _oof_rep = 0
                    _tuning_oof_folds = sl_tuning_folds(train.shape[0], _sl_cfg, _oof_rep)
                    _do_tuning = True
                else:
                    print(f"[GP] Task {task_num}/{total_tasks} | NOTE: {MODEL_RUN[jj]} cannot be "
                          f"tuned out of fold: {_oof_tune_skip} - "
                          + ("it is tuned on the validation split instead." if _do_tuning else
                             "it runs at its configured hyperparameters (no validation split "
                             "to tune on)."))
            # ver4-56 (R1): the CONFIGURED parameters (frozen baseline, not a tuned result)
            # and the size the model sees - kept so a model that fails mid-fit is still recorded.
            _tele_params = HPARAMETERS_BASELINE.get(base_model_name)
            if isinstance(_tele_params, dict):
                _tele_params = _tele_params.get(sample.loc[i, 'phenotype'])
            _rtele.model_context(
                _tele_params, hp_tuning_cfg if _do_tuning else None,
                (train.shape[0], train.shape[1] - 1), valid.shape[0], test.shape[0],
                _tele_p_raw, _do_tuning)

            # ver4-26, R5: this model was jointly hyperparameter+weight
            # tuned by "Step 8.5" above (JOINT_W_OPT active, >=2 models
            # selected, a validation split exists, and the joint search
            # didn't fail for this task) - mutually exclusive with
            # _do_tuning by construction (JOINT_W_OPT forces HP_TUNE to
            # None at the top of GP(), so hp_tuning_cfg is always falsy
            # whenever this is true).
            _use_joint_params = base_model_name in _joint_final_params_by_model

            # ver4-35, R1.f: this model's result from the lane, or None -
            # not a lane model, its chain failed (a NOTE was printed and the
            # random streams were restored to the last good job), or the
            # lane is off - in which case the unchanged branch below runs it.
            _lane_hit = None
            if _task_lane is not None and not _do_tuning and not _use_joint_params \
                    and _task_lane.covers(MODEL_RUN[jj]):
                _lane_hit = _task_lane.consume(MODEL_RUN[jj])

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
                    # ver4-61, R1: the whole graph (not only the merge's RF/pair tables)
                    # is hyperparameter-independent - build it once here as well.
                    _bio_graph = _prewarm_bio_graph_for(base_model_name)

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
                    bio_prior_ctx=_bio_prior_trial_ctx([base_model_name])
                    if _is_bio_prior_model(base_model_name) else None,
                    # ver4-25, R8 (design blueprint SS2, Stage 6): this
                    # task's own shared BlueWeightContext (or None) -
                    # every search trial AND the confirmatory fit are
                    # fit with the SAME weights production dispatch
                    # would use, and the confirmatory fit's own
                    # pearson_valid/mse_valid (the tuning objective) is
                    # correctly weighted for free via ModelTrialRunner's
                    # own tail (see that class's __call__ docstring).
                    sample_weight_ctx=_blue_weight_ctx,
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

                # ver4-61, R2: trial-memory guard. A bio-prior GAT trial keeps B x E0 x HC
                # edge activations for backward; on a data-driven-merge graph that is tens
                # of GB per trial, so the legacy width (N_JOBS / HP_TUNE_BAYES_BATCH_MAX
                # alone) let the operating system SIGKILL workers. For a bio-prior model
                # whose search will really run, the width is capped from the predicted
                # memory of one trial at the search-box corner, and worker deaths are
                # isolated (only the dead candidates are re-run, at half the width).
                # ACCELERATION['trial_memory_guard']='off' keeps the ver4-60 behaviour.
                _bayes_batch_max_eff = _compute_resources['hp_tune_bayes_batch_max']
                _trial_isolate = False
                if _is_bio_prior_model(base_model_name) \
                        and _preset_entry_for(base_model_name, algorithm, i) is None:
                    if _compute_resources['accel_trial_memory_guard'] != 'auto':
                        print(f"[GP] trial memory guard: off for {base_model_name} "
                              f"(ACCELERATION trial_memory_guard='off') - concurrency comes from "
                              f"N_JOBS / HP_TUNE_BAYES_BATCH_MAX alone.")
                    else:
                        _trial_isolate = True
                        _tuning_n_jobs, _bayes_batch_max_eff = _apply_trial_memory_guard(
                            base_model_name, algorithm, _base_params_for_tuning, hp_tuning_cfg,
                            _bio_graph, train, valid, test, _tuning_n_jobs,
                            _bayes_batch_max_eff)

                # PERFORMANCE FIX (Requirement: "Bayesian hyperparameter
                # tuning is prohibitive - at least 8x a non-tuning run"),
                # part 2 - see resolve_trial_fanout_width()'s own module-
                # level comment in models/hyperparameter_tuning.py for the
                # full diagnosis. Fixing _resolve_q()'s negative-sentinel
                # bug (part 1) made trial candidates genuinely run
                # concurrently across worker PROCESSES - but every one of
                # those workers was still being handed THIS task's own
                # full, un-divided per-process compute budget
                # (_compute_resources: n_jobs for scikit-learn/SHAP,
                # r_blas_threads for the four BGLR-backed R models,
                # torch_num_threads for MLP/GAT), the exact budget a
                # SINGLE serial fit is sized for. Running `_trial_fanout_
                # width` copies of that budget AT THE SAME TIME
                # oversubscribes the node's CPUs by that same factor (e.g.
                # N_JOBS=12 x HP_TUNE_BAYES_BATCH_MAX=8 concurrent workers
                # each trying to use 12 BLAS/scikit-learn threads = up to
                # 96 threads contending for however many cores actually
                # exist) - thread-contention overhead of this kind does
                # not merely fail to speed things up, it can make the
                # "parallel" path slower than serial, which is the >=8x
                # slowdown this requirement reports, scaling with the same
                # batch width the report's own number does.
                #
                # Fix: give trial WORKERS a divided budget
                # (_trial_worker_resources), sized so `_trial_fanout_width`
                # of them running at once sums back to roughly this run's
                # own resolved budget - via nested_safe_n_jobs(), the same
                # "divide, never multiply" helper every other nested-
                # parallelism case in this codebase already uses - while
                # the PARENT process (which runs the tuner's own single,
                # serial confirmatory fit, and any HP_TUNE_SCOPE=
                # 'per_scenario' cache-hit reuse fit) keeps the untouched,
                # full-budget _compute_resources exactly as before this
                # fix: ModelTrialRunner is constructed with
                # _compute_resources (unchanged, below) precisely so the
                # confirmatory fit - which never runs inside a trial
                # worker - is sized identically to every other, non-tuned
                # model call in this run. Only the RESOURCES DICT bundled
                # into worker_init_args (installed process-globally, once,
                # inside each freshly spawned trial worker via
                # _trial_worker_init()) is divided.
                _trial_fanout_width = (
                    resolve_trial_fanout_width(
                        algorithm, _tuning_n_jobs,
                        bayes_batch=_compute_resources['hp_tune_bayes_batch'],
                        bayes_batch_max=_bayes_batch_max_eff,
                        parallel_restarts=_compute_resources['hp_tune_parallel_restarts'],
                        budget_kwargs=hp_tuning_cfg.get('budget', {}).get(algorithm, {}),
                    )
                    if _compute_resources['hp_tune_parallel_trials'] else 1
                )
                # ver4-28, R1: dict-division logic extracted into the
                # module-level `_trial_worker_resources_for()` helper (see
                # its own comment above `_trial_worker_init()`) so Step 8.5's
                # own joint-search parallelism can reuse it - this call site
                # is otherwise unchanged (I11).
                _trial_worker_resources = _trial_worker_resources_for(_compute_resources, _trial_fanout_width)
                if _trial_fanout_width > 1:
                    # Diagnostics-as-a-design-principle (architecture doc
                    # §18): log the resolved division unconditionally
                    # whenever it actually applies, so "why is each trial
                    # only using 1 thread?" always has a documented answer
                    # in the run log rather than only working it out from
                    # this comment.
                    print(f"[GP] NOTE: {base_model_name} ({algorithm}) hyperparameter search "
                          f"will run up to {_trial_fanout_width} trial(s) concurrently - "
                          f"per-worker budget reduced to n_jobs="
                          f"{_trial_worker_resources['n_jobs']}"
                          + (f", r_blas_threads={_trial_worker_resources['r_blas_threads']}"
                             if _compute_resources.get('r_blas_threads') else "")
                          + (f", torch_num_threads={_trial_worker_resources['torch_num_threads']}"
                             if _compute_resources.get('torch_num_threads') else "")
                          + f" (this run's own full per-process budget is n_jobs="
                          f"{_compute_resources.get('n_jobs')}"
                          + (f", r_blas_threads={_compute_resources.get('r_blas_threads')}"
                             if _compute_resources.get('r_blas_threads') else "")
                          + ") - dividing prevents the concurrent trial workers from "
                          f"oversubscribing this node's CPUs relative to a single serial fit.")

                # ver4-5 R1.i: the trial-worker initialiser + its plain-
                # data arguments, forwarded all the way down through
                # tune_model_hyperparameters() -> run_search() -> each
                # search function's own fan-out - see
                # _trial_worker_init()'s own docstring above for why
                # this is mandatory whenever _tuning_n_jobs > 1.
                if _tuning_n_jobs > 1:
                    ensure_gpu_arbiter(_compute_resources)  # ver4-35, R1.e2 (see Step 8.5 above)
                _trial_worker_init_args = (
                    RESULT_NAME, R_PATH, _trial_worker_resources, get_gpu_semaphore(),
                )

                final_params, tuned_result, best_valid_score, tuned_values, tune_elapsed, _tuning_source = \
                    _resolve_tuned_params(
                        base_model_name=base_model_name, algorithm=algorithm, i=i,
                        _base_params_for_tuning=_base_params_for_tuning,
                        train=train, valid=valid, test=test,
                        _tuning_run_model_fn=_tuning_run_model_fn,
                        hp_tuning_cfg=hp_tuning_cfg, _tuning_n_jobs=_tuning_n_jobs,
                        _trial_worker_init_args=_trial_worker_init_args,
                        _bayes_batch_max=_bayes_batch_max_eff,
                        _isolate_failures=_trial_isolate,
                        _oof_folds=_tuning_oof_folds,
                        _oof_fast=_sl_oof_plan,
                    )
                # ver4-35, R1.a: the search's candidate count and summed worker
                # seconds, as 'trials:<model>=n/secs' in this task's [timing]
                # line (absent when the tuned fit came from a cache).
                _trial_info = (tuned_result.get('_trial_secs')
                               if isinstance(tuned_result, dict) else None)
                if _trial_info is not None and current_stage_timer() is not None:
                    current_stage_timer().info(
                        f"trials:{MODEL_RUN[jj]}", f"{_trial_info[0]}/{_trial_info[1]:.1f}s")
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
                    # ver4-40 R1: five extra columns ONLY when HP_TUNE_FAST is active, so every
                    # legacy run's hyperparameter.csv stays column-for-column identical (I8).
                    **(_fast_record_columns(tuned_result) if _compute_resources['hp_tune_fast']
                       is not None else {}),
                    # ver4-85: present only for out-of-fold tuning, so a holdout-tuned run's
                    # hyperparameter.csv keeps exactly its earlier columns (I11).
                    # ver4-87: names the objective that actually CHOSE the values - an
                    # out-of-fold search of this run ('searched', or 'reused:' from this run's
                    # per-scenario cache), a pilot's preset ('preset': whatever the pilot used),
                    # or no search at all ('none').
                    **({'tuning_objective': (
                        f'oof_{len(_tuning_oof_folds)}fold'
                        if (_tuning_source == 'searched' or _tuning_source.startswith('reused:'))
                        else 'preset' if _tuning_source.startswith('preset:') else 'none')}
                       if _tuning_oof_folds is not None else {}),
                }])])
                _tune_verb = (f"tuned via {algorithm}"
                              + (f" on out-of-fold predictions ({len(_tuning_oof_folds)} folds over "
                                 f"the training lines)" if _tuning_oof_folds is not None else "")
                              if _tuning_source == 'searched' else
                              "ran one fit at the configured hyperparameters (HP_TUNE_FAST left "
                              "nothing to search)" if _tuning_source == 'no_search_space' else
                              f"reused tuned hyperparameters from preset row "
                              f"{_tuning_source.split(':', 1)[-1]!r} (HP_TUNE_PRESET_FILE - "
                              f"confirmatory fit only)" if _tuning_source.startswith('preset:')
                              else
                              f"reused tuned hyperparameters from "
                              f"{_tuning_source.split(':', 1)[-1]!r} "
                              f"(HP_TUNE_SCOPE='per_scenario' - confirmatory fit only)")
                print(f"[GP] Task {task_num}/{total_tasks} | {MODEL_RUN[jj]} {_tune_verb}: "
                      f"{'out-of-fold' if _tuning_oof_folds is not None else 'valid'} "
                      f"Pearson/MSE={best_valid_score:.4f} (took {tune_elapsed:.1f}s) -> {tuned_values}")

            elif _use_joint_params:
                # ver4-26, R5: a fresh, full-fidelity ModelTrialRunner
                # confirmatory fit at "Step 8.5"'s own winning joint
                # hyperparameters for THIS model - the SAME canonical
                # call path every HP_TUNE'd model's own row already goes
                # through (see Step 8.5's own comment above for why this
                # is a deliberately separate fit rather than re-using the
                # joint search's internal scoring-confirmatory fit
                # output). MODEL_DISPATCH_FILTER note: like W_OPT, a task
                # with JOINT_W_OPT active and a validation split is made
                # INELIGIBLE for model-level fan-out before any unit is
                # submitted (see intra_task_parallel.
                # task_is_model_fanout_eligible()'s ver4-26 extension) -
                # `_joint_runner` (built once in Step 8.5, in THIS same
                # process) is always in scope here for that reason.
                _joint_params_for_model = list(_joint_final_params_by_model[base_model_name])
                # ver4-44, R1 (N11): class C events of THIS model's own confirmatory refit, written
                # to `joint_fallbacks` in this model's hyperparameter.csv row (see below).
                _joint_n11_events = []
                _joint_confirm_t0 = time.time()
                # ver4-28, R2: reuse Step 8.5's own confirmatory fit for this
                # model instead of refitting it here, when it is available -
                # `_joint_confirm_results` only ever holds an entry when
                # `reuse_confirmatory_fit` was requested AND Step 8.5's own
                # P2 safety check passed for this task (see Step 8.5's own
                # comment); a `per_scenario` cache hit, a failed search, or
                # the key left at its default all leave it empty for this
                # model, and this branch falls straight through to the
                # LEGACY refit-plus-retry path below unchanged (I11).
                if base_model_name in _joint_confirm_results:
                    tuned_result = _joint_confirm_results.pop(base_model_name)
                    _joint_confirm_elapsed = 0.0
                    print(f"[GP] Task {task_num}/{total_tasks} | {MODEL_RUN[jj]} joint-tuned: "
                          f"reused Step 8.5's own confirmatory fit (reuse_confirmatory_fit=True) "
                          f"instead of refitting.")
                else:
                    # BUGFIX (joint-optimisation-underperformance-fix, Round 5):
                    # this per-model confirmatory re-fit previously had NO
                    # exception handling at all - explicitly flagged but left
                    # unfixed by the prior round of this same investigation
                    # (claude/joint-bayesian-optimisation-underperformance-fix.
                    # md: "genomic_prediction.py's elif _use_joint_params:
                    # dispatch branch ... has no exception-handling/retry-to-
                    # untuned-baseline fallback"). Any single joint-group
                    # model's own refit failure here (a numerical divergence, an
                    # R MCMC failure, ...) propagated straight out of
                    # _process_one_task to the task-loop's own except block,
                    # which rolls back and RE-RAISES the entire task - killing
                    # every OTHER model's already-computed row too - even though
                    # models/joint_tuning.py's own confirmatory fit inside the
                    # search already fit this exact model successfully at these
                    # exact hyperparameters (Step 8.5 only reaches here after
                    # that fit succeeded). A transient failure specific to THIS
                    # re-fit (not the hyperparameters themselves) is exactly the
                    # class of problem tune_model_hyperparameters()'s own
                    # confirmatory-fit guard already exists for (models/
                    # hyperparameter_tuning.py, retry-once-at-untuned-baseline -
                    # see that function's own comment for the full rationale,
                    # mirrored here verbatim). Retry ONCE at this model's own
                    # untuned baseline (HPARAMETERS_BASELINE[base_model_name],
                    # handling a bio-prior model's per-phenotype dict the same
                    # way Step 8.5 already resolves it above), re-raising only
                    # if that ALSO fails (a genuine configuration error, not a
                    # tuning artefact).
                    try:
                        tuned_result = _joint_runner(
                            base_model_name, train, valid, test, _joint_params_for_model, RESULT_NAME,
                        )
                    except Exception as _joint_confirm_exc:
                        print(f"[GP] Task {task_num}/{total_tasks} | NOTE: {base_model_name} joint-"
                              f"tuned confirmatory fit failed at the joint search's own winning "
                              f"hyperparameters {_joint_params_for_model!r} "
                              f"({type(_joint_confirm_exc).__name__}: {_joint_confirm_exc}) - "
                              f"retrying once at this model's own untuned baseline before giving up "
                              f"on this task.\n{traceback.format_exc()}")
                        _joint_n11_events.append(f"N11:{base_model_name}")
                        _joint_baseline_entry = HPARAMETERS_BASELINE[base_model_name]
                        if _is_bio_prior_model(base_model_name) and isinstance(_joint_baseline_entry, dict):
                            _joint_confirm_phenotype = sample.loc[i, 'phenotype']
                            if _joint_confirm_phenotype not in _joint_baseline_entry:
                                print(f"[GP] Task {task_num}/{total_tasks} | {base_model_name} has no "
                                      f"untuned baseline entry for phenotype "
                                      f"{_joint_confirm_phenotype!r} to retry at - re-raising the "
                                      f"original confirmatory-fit failure above.")
                                raise
                            _joint_params_for_model = list(_joint_baseline_entry[_joint_confirm_phenotype])
                        else:
                            _joint_params_for_model = list(_joint_baseline_entry)
                        try:
                            tuned_result = _joint_runner(
                                base_model_name, train, valid, test, _joint_params_for_model, RESULT_NAME,
                            )
                        except Exception:
                            print(f"[GP] Task {task_num}/{total_tasks} | {base_model_name} also "
                                  f"failed to fit at its own configured (untuned) baseline for this "
                                  f"task - this is a genuine configuration error, not a joint-tuning "
                                  f"artefact; re-raising.\n{traceback.format_exc()}")
                            raise
                        print(f"[GP] Task {task_num}/{total_tasks} | NOTE: {base_model_name} joint "
                              f"tuning fell back to its untuned baseline for this task after the "
                              f"confirmatory-fit failure above - the hp_record row below still "
                              f"reports the joint search's own best group objective score and tuned "
                              f"values, but the fit actually used for this model was untuned.")
                    _joint_confirm_elapsed = time.time() - _joint_confirm_t0

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
                    'algorithm': 'Joint Bayesian optimisation',
                    'valid_pearson_over_mse': _joint_best_valid_score,
                    'valid_objective_score': _joint_best_valid_score,
                    'tuning_source': _joint_tuning_source,
                    # BUGFIX (hyperparameter.csv recording): this row
                    # previously carried NO tuned-value columns at all -
                    # unlike every single-model HP_TUNE row just above
                    # (which spreads its own `tuned_values` the exact same
                    # way), the joint search's own winning per-field
                    # values for THIS model were computed but never
                    # written out. `_joint_tuned_values_by_model` is
                    # `{field_label: final_value}` (built in
                    # models/joint_tuning.py, mirroring
                    # tune_model_hyperparameters()'s own `tuned_values`
                    # exactly) - `.get(base_model_name, {})` is a no-op
                    # `{}` splat only for a model with nothing tunable in
                    # HPARAM_SPECS (a valid, zero-column contribution, not
                    # a bug - see build_joint_param_specs()'s docstring).
                    **_joint_tuned_values_by_model.get(base_model_name, {}),
                    # ver4-28, R6/P3: three additional provenance columns,
                    # added ONLY when at least one ver4-28 key inside
                    # HYPERPARAMETERS_JOINT_OPT is non-default
                    # (`_v428_nondefault()`) - every run that never touches
                    # these keys gets a byte-for-byte identical hp_record
                    # schema to a pre-ver4-28 run (P3, avoids a spurious
                    # `run_step2_assemble.py` column-unification surprise).
                    **({
                        'joint_strategy': (_joint_details or {}).get('strategy'),
                        'joint_gate_decision': (_joint_details or {}).get('gate_decision'),
                        'joint_n_fits': (_joint_details or {}).get('n_fits'),
                        # ver4-44, R1: the resolved `safety_nets` of this task and the class C
                        # fallback events that happened (';'-joined, empty = none), so the
                        # evaluation can tell a pure joint result from one a fallback produced.
                        'joint_safety_nets': bool((_joint_details or {}).get(
                            'safety_nets', (HYPERPARAMETERS_JOINT_OPT or {}).get('safety_nets', True))),
                        'joint_fallbacks': ';'.join(
                            list((_joint_details or {}).get('fallback_events') or [])
                            + _joint_n11_events),
                        # ver4-47: R5 / R6 provenance, only when one of them is switched on so
                        # every other run keeps its exact hp_record schema (P3).
                        **({'joint_wf_reweight': str((HYPERPARAMETERS_JOINT_OPT or {}).get(
                                'wf_cv_reweight', 'off')),
                            'joint_wf_members': bool((HYPERPARAMETERS_JOINT_OPT or {}).get(
                                'wf_cv_members', False))}
                           if ((HYPERPARAMETERS_JOINT_OPT or {}).get('wf_cv_reweight', 'off') != 'off'
                               or bool((HYPERPARAMETERS_JOINT_OPT or {}).get('wf_cv_members', False)))
                           else {}),
                        # ver4-40: the weight-first candidate that won, only on weight_first runs
                        # so every other run keeps its exact hp_record schema (P3).
                        **({'joint_weight_candidate':
                            (_joint_details or {}).get('joint_weight_candidate')}
                           if (HYPERPARAMETERS_JOINT_OPT or {}).get('search_strategy')
                           in ('weight_first', 'weight_first_cv') else {}),
                        # ver4-48: the neural stack's chosen lambda ('inf' = the untuned naive
                        # ensemble), the number of library members it deployed and the mean gate
                        # entropy - only on neural_stack runs (I5: appended, P3: every other run
                        # keeps its exact hp_record schema).
                        # ver4-50: plus the objective and the swap verdicts, only when one of the
                        # two ver4-50 keys is away from its legacy value (P3, I11).
                        **(_ns_hp_columns(
                            (_joint_details or {}).get('ns'),
                            include_v450=(
                                (HYPERPARAMETERS_JOINT_OPT or {}).get(
                                    'ns_objective', 'ensemble_score') != 'ensemble_score'
                                or (HYPERPARAMETERS_JOINT_OPT or {}).get(
                                    'ns_swap_rule', 'gate') != 'gate'),
                            include_v451=(
                                (HYPERPARAMETERS_JOINT_OPT or {}).get(
                                    'ns_outer_gate', 'off') != 'off'))
                           if (HYPERPARAMETERS_JOINT_OPT or {}).get('search_strategy')
                           == 'neural_stack' else {}),
                    } if _v428_nondefault(HYPERPARAMETERS_JOINT_OPT or {}) else {}),
                }])])
                print(f"[GP] Task {task_num}/{total_tasks} | {MODEL_RUN[jj]} joint-tuned "
                      f"(group valid objective={_joint_best_valid_score:.4f}, this model's own "
                      f"confirmatory fit took {_joint_confirm_elapsed:.1f}s) -> "
                      f"params={_joint_params_for_model}")

            elif base_model_name == 'rrBLUP':
                result_rrBLUP = rrBLUP(train, valid, test, HPARAMETERS[base_model_name], RESULT_NAME,
                                        **({'weights': _blue_weight_ctx.w_fit} if _blue_weight_ctx is not None else {}))
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
                # Requirements3.md item 1 (ver4-25, Stage 7 sibling patch)
                # - see ModelTrialRunner's equivalent rrBLUP branch above
                # (lockstep, RK-2): both call the SAME shared, module-
                # level helper, so there is only one implementation to
                # keep in sync.
                sample_effect = _apply_linear_closed_form_effect(
                    'rrBLUP', HPARAMETERS[base_model_name], sample_effect, train, test,
                )
                # Update ID ver4-5, R2 Stage 10 (I3 lockstep with
                # ModelTrialRunner's equivalent branch - both call the
                # SAME shared, module-level helper, so there is only one
                # implementation to keep in sync, not two - see that
                # helper's own docstring).
                sample_interaction = _r_model_surrogate_interaction(
                    'rrBLUP', train, predicted_train, HPARAMETERS[base_model_name],
                    _compute_resources.get('n_jobs'), test=test,
                    row_parallel=_compute_resources.get('shapiq_row_parallel', True),
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
                                                   k_precomputed=_gblup_k_precomputed,
                                                   weights=(_blue_weight_ctx.w_fit if _blue_weight_ctx is not None else None))
                _cleanup_bglr_output_for(RESULT_NAME)  # disk-quota fix - see module-level note above
                # ver4-4 R5 Fix 1 - see rrBLUP branch above.
                sample_pearson, sample_mse = r_list_get(result_GBLUP, 'r_pearson')[0], r_list_get(result_GBLUP, 'r_MSE')[0]
                sample_effect = r_list_get(result_GBLUP, 'r_effect')
                predicted_test = r_list_get(result_GBLUP, 'r_y_predicted')
                predicted_valid = r_list_get(result_GBLUP, 'r_y_predicted_valid')
                predicted_train = r_list_get(result_GBLUP, 'r_y_predicted_train')
                # ver4-25, Stage 7 (R13) - see ModelTrialRunner's
                # equivalent GBLUP branch above (lockstep, RK-2).
                sample_effect = _apply_gblup_closed_form_effect(result_GBLUP, _gblup_params, train, test)
                sample_interaction = _r_model_surrogate_interaction(
                    'GBLUP', train, predicted_train, _gblup_params,
                    _compute_resources.get('n_jobs'), test=test,
                    row_parallel=_compute_resources.get('shapiq_row_parallel', True),
                )
            elif base_model_name  == 'BayesB':
                result_BayesB = BayesB(train, valid, test, HPARAMETERS[base_model_name], RESULT_NAME,
                                        **({'weights': _blue_weight_ctx.w_fit} if _blue_weight_ctx is not None else {}))
                _cleanup_bglr_output_for(RESULT_NAME)  # disk-quota fix - see module-level note above
                # ver4-4 R5 Fix 1 - see rrBLUP branch above.
                sample_pearson, sample_mse = r_list_get(result_BayesB, 'r_pearson')[0], r_list_get(result_BayesB, 'r_MSE')[0]
                sample_effect = r_list_get(result_BayesB, 'r_effect')
                predicted_test = r_list_get(result_BayesB, 'r_y_predicted')
                predicted_valid = r_list_get(result_BayesB, 'r_y_predicted_valid')
                predicted_train = r_list_get(result_BayesB, 'r_y_predicted_train')
                # Requirements3.md item 1 - see the rrBLUP branch above.
                sample_effect = _apply_linear_closed_form_effect(
                    'BayesB', HPARAMETERS[base_model_name], sample_effect, train, test,
                )
                sample_interaction = _r_model_surrogate_interaction(
                    'BayesB', train, predicted_train, HPARAMETERS[base_model_name],
                    _compute_resources.get('n_jobs'), test=test,
                    row_parallel=_compute_resources.get('shapiq_row_parallel', True),
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
                                                  _rkhs_params, RESULT_NAME,
                                                  weights=(_blue_weight_ctx.w_fit if _blue_weight_ctx is not None else None))
                _cleanup_bglr_output_for(RESULT_NAME)  # disk-quota fix - see module-level note above
                # ver4-4 R5 Fix 1 - see rrBLUP branch above.
                sample_pearson, sample_mse = r_list_get(result_RKHS, 'r_pearson')[0], r_list_get(result_RKHS, 'r_MSE')[0]
                sample_effect = r_list_get(result_RKHS, 'r_effect')
                predicted_test = r_list_get(result_RKHS, 'r_y_predicted')
                predicted_valid = r_list_get(result_RKHS, 'r_y_predicted_valid')
                predicted_train = r_list_get(result_RKHS, 'r_y_predicted_train')
                # ver4-25, Stage 7 (R13) - see ModelTrialRunner's
                # equivalent RKHS branch above (lockstep, RK-2).
                # ver4-43 (E2/E3) - see ModelTrialRunner's equivalent branch
                # above (lockstep, RK-2): one memo per RKHS model call; the two
                # explanation stages are timed on their own.
                _rkhs_memo = {}
                with timed_stage('xai_effect:RKHS'):
                    sample_effect = _apply_rkhs_shapiq_predictor_effect(
                        RKHS, result_RKHS, _rkhs_params, train, valid, test, RESULT_NAME, predicted_test,
                        ({'weights': _blue_weight_ctx.w_fit} if _blue_weight_ctx is not None else {}),
                        n_jobs=_compute_resources.get('n_jobs'),
                        predictor_memo=_rkhs_memo,
                    )
                # This update - see ModelTrialRunner's equivalent RKHS
                # branch above (lockstep, RK-2) and
                # _apply_rkhs_predictor_interaction()'s own docstring.
                with timed_stage('xai_inter:RKHS'):
                    sample_interaction = _apply_rkhs_predictor_interaction(
                        result_RKHS, _rkhs_params, train, test, predicted_train, predicted_test,
                        valid=valid,
                        n_jobs=_compute_resources.get('n_jobs'),
                        use_gpu=_compute_resources.get('use_gpu_sklearn', False),
                        row_parallel=_compute_resources.get('shapiq_row_parallel', True),
                        # This update - see sample_effect above (already
                        # computed, I3 lockstep with ModelTrialRunner's
                        # equivalent RKHS branch) and
                        # _apply_rkhs_predictor_interaction()'s own
                        # 'effect_values' parameter docstring.
                        effect_values=sample_effect,
                        # Bug fix (V13.2 gate false-positive under BLUE-
                        # reliability weighting) - see ModelTrialRunner's
                        # equivalent RKHS branch above (lockstep, RK-2) and
                        # _apply_rkhs_predictor_interaction()'s own 'weights'
                        # parameter docstring.
                        weights=(_blue_weight_ctx.w_fit if _blue_weight_ctx is not None else None),
                        predictor_memo=_rkhs_memo,
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
                sample_pearson, sample_mse, sample_effect, sample_interaction, predicted_test, predicted_valid, predicted_train = RF(train, valid, test, _rf_params, sample_weight_ctx=_blue_weight_ctx)
            elif base_model_name  == 'ExtraTrees':
                # Update ID ver4-5, R2 Stage 8: ExtraTrees params mirror
                # RF's own layout exactly - shapley_num is at index 6.
                # Update ID ver4-6, R3.2a - task-local, re-resolved every task.
                _extratrees_params = list(HPARAMETERS[base_model_name])
                if _extratrees_params[6] == 'all':
                    _extratrees_params[6] = test.shape[0]
                sample_pearson, sample_mse, sample_effect, sample_interaction, predicted_test, predicted_valid, predicted_train = ExtraTrees(train, valid, test, _extratrees_params, sample_weight_ctx=_blue_weight_ctx)
            elif base_model_name  == 'GBDT':
                # Update ID ver4-5, R2 Stage 8: GBDT params: [max_iter,
                # learning_rate, max_depth, max_leaf_nodes, min_samples_leaf,
                # l2_regularization, get_interaction, shapley_num, threshold,
                # max_interaction_features] -> shapley_num is at index 7.
                # Update ID ver4-6, R3.2a - task-local, re-resolved every task.
                _gbdt_params = list(HPARAMETERS[base_model_name])
                if _gbdt_params[7] == 'all':
                    _gbdt_params[7] = test.shape[0]
                sample_pearson, sample_mse, sample_effect, sample_interaction, predicted_test, predicted_valid, predicted_train = GBDT(train, valid, test, _gbdt_params, sample_weight_ctx=_blue_weight_ctx)
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
                sample_pearson, sample_mse, sample_effect, sample_interaction, predicted_test, predicted_valid, predicted_train = XGBoost(train, valid, test, _xgboost_params, sample_weight_ctx=_blue_weight_ctx)
            elif base_model_name  == 'EBM':
                # Update ID ver4-5, R2 Stage 11 (Tier 2 - optional
                # dependency). No shapley_num/'all' substitution - EBM's
                # interactions are its own already-fitted pairwise terms,
                # not an explained-sample-count computation.
                sample_pearson, sample_mse, sample_effect, sample_interaction, predicted_test, predicted_valid, predicted_train = EBM(train, valid, test, HPARAMETERS[base_model_name], sample_weight_ctx=_blue_weight_ctx)
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
                sample_pearson, sample_mse, sample_effect, sample_interaction, predicted_test, predicted_valid, predicted_train = SV_Regression(train, valid, test, _svr_params, sample_weight_ctx=_blue_weight_ctx)
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
                sample_pearson, sample_mse, sample_effect, sample_interaction, predicted_test, predicted_valid, predicted_train = KNN(train, valid, test, _knn_params, sample_weight_ctx=_blue_weight_ctx)
            elif base_model_name  == 'MLP':
                # MLP params: [neurons, dout, lrate, decay, ep, bsize, neurons2,
                # shapley_num, get_interaction, max_interaction_features,
                # interaction_top] -> shapley_num is at index 7; the last
                # three fields are ver4-5 R2 Stage 9.
                # Update ID ver4-6, R3.2a - task-local, re-resolved every task.
                _mlp_params = list(HPARAMETERS[base_model_name])
                if _mlp_params[7] == 'all':
                    _mlp_params[7] = test.shape[0]
                sample_pearson, sample_mse, sample_effect, sample_interaction, predicted_test, predicted_valid, predicted_train = _seed_bag_fit(
                    'MLP', lambda: ML_Perceptron(train, valid, test, _mlp_params, sample_weight_ctx=_blue_weight_ctx), sample.loc[i, 'sample'], test)
            elif base_model_name  == 'GAT_infinitesimal_node_level':
                # GAT_infinitesimal_node_level params: [..., samples, marker_effect] -> samples is second-to-last
                # Update ID ver4-6, R3.2a - task-local, re-resolved every task.
                _gat_inl_params = list(HPARAMETERS[base_model_name])
                if _gat_inl_params[-2] == 'all':
                    _gat_inl_params[-2] = test.shape[0]
                sample_pearson, sample_mse, sample_effect, predicted_test, predicted_valid, predicted_train = _seed_bag_fit(
                    'GAT_infinitesimal_node_level', lambda: _final_gat_fit(GAT_infinitesimal_node_level, train, valid, test, _gat_inl_params, sample_weight_ctx=_blue_weight_ctx), sample.loc[i, 'sample'], test)
            elif base_model_name  == 'GAT_infinitesimal':
                # GAT_infinitesimal params: [..., marker_effect, samples] -> samples is the last element
                # Update ID ver4-6, R3.2a - task-local, re-resolved every task.
                _gat_inf_params = list(HPARAMETERS[base_model_name])
                if _gat_inf_params[-1] == 'all':
                    _gat_inf_params[-1] = test.shape[0]
                sample_pearson, sample_mse, sample_effect, predicted_test, predicted_valid, predicted_train = _seed_bag_fit(
                    'GAT_infinitesimal', lambda: _final_gat_fit(GAT_infinitesimal, train, valid, test, _gat_inf_params, sample_weight_ctx=_blue_weight_ctx), sample.loc[i, 'sample'], test)
            elif base_model_name  == 'GAT_fully_connected':
                # GAT_fully_connected params: [..., marker_effect, samples] -> samples is the last element
                # Update ID ver4-6, R3.2a - task-local, re-resolved every task.
                _gat_fc_params = list(HPARAMETERS[base_model_name])
                if _gat_fc_params[-1] == 'all':
                    _gat_fc_params[-1] = test.shape[0]
                sample_pearson, sample_mse, sample_effect, predicted_test, predicted_valid, predicted_train, sample_attention = _seed_bag_fit(
                    'GAT_fully_connected', lambda: _final_gat_fit(GAT_fully_connected, train, valid, test, _gat_fc_params, sample_weight_ctx=_blue_weight_ctx), sample.loc[i, 'sample'], test)
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
                sample_pearson, sample_mse, sample_effect, predicted_test, predicted_valid, predicted_train, sample_attention, sample_interaction = _seed_bag_fit(
                    'GAT_prior_knowledge', lambda: _final_gat_fit(GAT_prior_knowledge, train, valid, test, _gat_pk_params, sample_weight_ctx=_blue_weight_ctx), sample.loc[i, 'sample'], test)
            elif _lane_hit is not None:
                # ver4-35, R1.f: the lane fitted this model through
                # ModelTrialRunner's bio-prior branch (the 'all' substitution,
                # the PLINK marker-info override and the phenotype/network
                # check are the same statements as in the branch below).
                # Only the bookkeeping the branch below does is repeated here:
                # `train`/`valid`/`test` must be this instance's own pools
                # because "Store prediction results" reads them.
                train, valid, test = bio_prior_pools.get(base_model_name, (train_unpruned, valid_unpruned, test_unpruned)) \
                    if GENOTYPE_FORMAT == 'plink' else (train_unpruned, valid_unpruned, test_unpruned)
                sample_pearson = _lane_hit['pearson_test']
                sample_mse = _lane_hit['mse_test']
                sample_effect = _lane_hit['effect']
                predicted_test = _lane_hit['predicted_test']
                predicted_valid = _lane_hit['predicted_valid']
                predicted_train = _lane_hit['predicted_train']
                sample_attention = _lane_hit['attention']
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

                if len(bio_prior_params) > _BIO_PRIOR_SAMPLES_IDX and \
                        bio_prior_params[_BIO_PRIOR_SAMPLES_IDX] == 'all':
                    bio_prior_params[_BIO_PRIOR_SAMPLES_IDX] = test.shape[0]

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

                sample_pearson, sample_mse, sample_effect, predicted_test, predicted_valid, predicted_train, sample_attention = _final_gat_fit(
                    GAT_biological_prior_knowledge,
                    train, valid, test, bio_prior_params, RESULT_NAME, sample.loc[i, 'phenotype'], MODEL_RUN[jj],
                    merge_source_data=_merge_source_data, network_cache=_bio_prior_merge_cache,
                    sample_weight_ctx=_blue_weight_ctx,
                )
            
            # Store prediction results
            # ver4-25, R8 (design blueprint SS2, Stage 6): central,
            # single-site weighted-scoring recompute - covers BOTH the
            # tuned branch (sample_pearson/mse came from tuned_result,
            # itself the confirmatory fit's own UNWEIGHTED test score -
            # see ModelTrialRunner.__call__, which deliberately leaves
            # pearson_test/mse_test unweighted) and every direct-call
            # branch above (each model's own raw, unweighted r/mse). A
            # single recompute here means no per-model file needed its
            # own test-scoring math touched for R8 - only its FITTING
            # weight (sample_weight_ctx, already threaded above).
            # `_asreml_blue_weighting == 'none'` (every pre-ver4-25 run,
            # and every ver4-25 run with weighting off) skips this
            # entirely, so `sample_pearson`/`sample_mse` reach
            # `record_sample` completely unchanged - I11 byte-identical.
            #
            # ver4-25, R8 bugfix (Requirements2.md, Error #1): this used
            # to be gated on `_blue_weight_ctx is not None` - this TASK's
            # own context, which `_asreml_get_blue_weight_context()`
            # deliberately degrades to `None` (never fatal - a printed
            # NOTE only) whenever it cannot be built for one particular
            # task (a transient data problem - see that function's own
            # docstring). That meant a run's very own `Metric.csv` could
            # gain or lose these three columns TASK BY TASK depending on
            # whether any single scenario's own weighting context
            # happened to build successfully - and since
            # `checkpoint_utils.append_partial_results()` appends each
            # task's own new rows to the SAME file with `header=` set
            # only once (whenever the file doesn't exist yet), a later
            # task's row gaining columns an earlier task's row (the one
            # that wrote the header) never had - or the reverse - either
            # silently drops the new values off the end of a data row
            # with no header for them, or leaves an earlier row's own
            # columns simply absent, depending on which task ran first.
            # The fix: whenever weighting is configured for this RUN AT
            # ALL (`_asreml_blue_weighting`, resolved ONCE before any
            # task ever runs - see that variable's own top-of-GP()
            # comment), every task's own row carries all three columns,
            # from the very first row `Metric.csv` ever gets - a
            # STRUCTURALLY stable schema for the whole file, independent
            # of any single task's own success/failure building its
            # context. A task whose own context could not be built still
            # reports honestly: 'metric_weighting' is 'none' for THAT
            # row (this task was, in fact, scored unweighted, whatever
            # the run-level setting says), and its "(unweighted)" columns
            # simply repeat the same (unweighted) value the main columns
            # already show - never fabricated, never silently dropped.
            _blue_weighting_extra_cols = {}
            if _asreml_blue_weighting != 'none':
                _actual_test_arr = np.asarray(test.iloc[:, -1], dtype=float)
                sample_pearson_unweighted, sample_mse_unweighted = safe_regression_metrics(
                    _actual_test_arr, predicted_test)
                if _blue_weight_ctx is not None:
                    sample_pearson, sample_mse = safe_regression_metrics(
                        _actual_test_arr, predicted_test,
                        **blue_weights.scoring_for(_blue_weight_ctx, 'test'))
                    _row_metric_weighting = _blue_weight_ctx.mode
                else:
                    # This task's own context degraded to None (see the
                    # NOTE already printed by
                    # _asreml_get_blue_weight_context()) - score exactly
                    # as an unweighted run would (unchanged from before
                    # this fix), but still disclose that honestly via
                    # 'metric_weighting' rather than silently omitting
                    # the column this run's schema otherwise guarantees.
                    sample_pearson, sample_mse = sample_pearson_unweighted, sample_mse_unweighted
                    _row_metric_weighting = 'none'
                _blue_weighting_extra_cols = {
                    'Pearson correlation (unweighted)': sample_pearson_unweighted,
                    'MSE (unweighted)': sample_mse_unweighted,
                    'metric_weighting': _row_metric_weighting,
                }
            # Requirements.md (BLUEs/BLUPs accuracy-metrics phase), items
            # 1, 7 - the three extra metrics (up to 8 columns), added
            # AFTER 'MSE' as required. `compute_accuracy_metrics_row()`
            # returns `{}` whenever `_asreml_accuracy_ctx` is `None`
            # (ASReml accuracy metrics off, or ASREML off entirely), so
            # this is a strict no-op for every existing run/config.
            # ver4-38, R3 (P0.5 site, lockstep with ModelTrialRunner.__call__):
            # every model's test prediction has just been produced by one of
            # the dispatch branches above; a constant column is announced
            # once per (task, model) instead of surfacing only as a NaN
            # correlation cell in Metric.csv.
            warn_if_constant_prediction(MODEL_RUN[jj], predicted_test,
                                        task_tag=str(sample.loc[i, 'sample']))
            record_sample = pd.DataFrame([{'population': sample.loc[i,'population'],
                                           'phenotype': sample.loc[i,'phenotype'],
                                           'model': MODEL_RUN[jj],
                                           'ratio': sample.loc[i,'ratio'],
                                           'sample': sample.loc[i,'sample'],
                                           'Pearson correlation': sample_pearson,
                                           'MSE': sample_mse,
                                           **_blue_weighting_extra_cols,
                                           **asreml_accuracy_metrics.compute_accuracy_metrics_row(
                                               predicted_test, id_test, sample_pearson, _asreml_accuracy_ctx,
                                           )}
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
           
            # Requirements.md (between-prediction validation-set patch):
            # a 'between' task's own validation-presence is now decided by
            # `_between_has_validation` (any of the three sources, not
            # just W_OPT-driven third_population) - so predicted_valid is
            # recorded here whenever this task genuinely has a validation
            # split, whether or not a weighted-ensemble method is active.
            if (result_valid_sample.shape[0]==0 and type(sample.loc[i,'ratio']) is tuple) or (result_valid_sample.shape[0]==0 and SCENARIO == 'between' and _between_has_validation):
                result_valid_sample = pd.DataFrame({'id': id_valid,
                                                   'population': [sample.loc[i,'population']] * len(id_valid),
                                                   'ratio': [sample.loc[i,'ratio']] * len(id_valid),
                                                   'phenotype': [sample.loc[i,'phenotype']] * len(id_valid),
                                                   'sample': [sample.loc[i,'sample']] * len(id_valid),
                                                   'actual':valid.iloc[:,-1],
                                                   MODEL_RUN[jj]:predicted_valid
                                                  })
            elif (result_valid_sample.shape[0]!=0 and type(sample.loc[i,'ratio']) is tuple) or (SCENARIO == 'between' and _between_has_validation):
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
            #
            # ver4-25, Stage 9 (design blueprint R12): split BEFORE the
            # emits_interactions() gate below, since a shapiq-based
            # route's own 'interaction' output may now mix k=2 rows
            # (routed into 'interactions'/Interaction.csv, exactly as
            # before this split existed) with k>=3 rows (routed into the
            # NEW 'interactions_hok'/Interaction_higher_order.csv). A
            # model with no shapiq higher-order search configured at all
            # returns `sample_interaction` unchanged and an empty
            # `sample_interaction_hok` - see _split_interaction_by_order()
            # 's own docstring for why this is a byte-identical no-op
            # for every pre-R12 config (I11).
            sample_interaction, sample_interaction_hok = _split_interaction_by_order(sample_interaction)
            if emits_interactions(schema_key_of(base_model_name)) and sample_interaction.shape[0] != 0:
                sample_interaction['population'] = sample.loc[i,'population']
                sample_interaction['phenotype'] = sample.loc[i,'phenotype']
                sample_interaction['model'] = MODEL_RUN[jj]
                sample_interaction['ratio'] = str(sample.loc[i,'ratio'])
                sample_interaction['sample'] = sample.loc[i,'sample']

                interactions = pd.concat([interactions, sample_interaction])

            if emits_interactions(schema_key_of(base_model_name)) and sample_interaction_hok.shape[0] != 0:
                sample_interaction_hok['population'] = sample.loc[i,'population']
                sample_interaction_hok['phenotype'] = sample.loc[i,'phenotype']
                sample_interaction_hok['model'] = MODEL_RUN[jj]
                sample_interaction_hok['ratio'] = str(sample.loc[i,'ratio'])
                sample_interaction_hok['sample'] = sample.loc[i,'sample']

                interactions_hok = pd.concat([interactions_hok, sample_interaction_hok])

            # ver4-57 (R5): keep this model's Step 9 fit for Step 11c. Nothing above is read
            # differently or written differently because of it (I11: ITERATIVE_ENSEMBLE=None
            # never enters this branch).
            # ver4-77 (R2): the same capture also feeds Step 11d (Super learner), so it runs when
            # EITHER feature wants it. Step 11c keeps its exact old filter through `_iter_skip_of`
            # (below), and the iterative NOTE is printed only when the iterative ensemble is on.
            if _member_capture:
                if _do_tuning:
                    _iter_params = list(final_params)
                elif _use_joint_params:
                    _iter_params = list(_joint_params_for_model)
                else:
                    _iter_entry = HPARAMETERS.get(base_model_name)
                    if isinstance(_iter_entry, dict):
                        _iter_entry = _iter_entry.get(sample.loc[i, 'phenotype'])
                    _iter_params = list(_iter_entry) if _iter_entry is not None else []
                _iter_marker_channel = not _is_bio_prior_model(base_model_name)
                _iter_effect = None
                if sample_effect.shape[0] != 0:
                    _iter_effect = np.asarray(sample_effect.iloc[0], dtype=float)
                _iter_skip = None
                if _iter_marker_channel and not np.array_equal(
                        train.iloc[:, -1].to_numpy(dtype=float),
                        train_pruned_full.iloc[:, -1].to_numpy(dtype=float), equal_nan=True):
                    _iter_skip = "its training target differs from the shared training split"
                elif _iter_marker_channel and not _iter_params:
                    _iter_skip = "its Step 9 hyperparameters could not be resolved"
                _iter_skip_of[MODEL_RUN[jj]] = _iter_skip
                if _iter_skip is not None and _iter_capture:
                    print(f"[GP] Task {task_num}/{total_tasks} | NOTE: ITERATIVE_ENSEMBLE - "
                          f"model {MODEL_RUN[jj]} is left out of the iterative ensemble: "
                          f"{_iter_skip}.")
                # Stored unless its hyperparameters are unresolved (then no feature can refit
                # it). A member that only the iterative ensemble would skip for its target is
                # still stored: Step 11c filters it through `_iter_skip_of`, and the Super
                # learner checks the shared train+valid target itself.
                if (not _iter_marker_channel) or _iter_params:
                    _iter_members[MODEL_RUN[jj]] = IterMember(
                        name=MODEL_RUN[jj], base=base_model_name, params=_iter_params,
                        train=train, valid=valid, test=test,
                        pred={'train': np.asarray(predicted_train, dtype=float).ravel(),
                              # ver4-77: a task without a validation split has no valid
                              # predictions whatever a model returned for the empty frame
                              'valid': (np.asarray(predicted_valid, dtype=float).ravel()
                                        if valid.shape[0] != 0 else np.empty(0)),
                              'test': np.asarray(predicted_test, dtype=float).ravel()},
                        effect=_iter_effect,
                        interaction=(sample_interaction.copy()
                                     if sample_interaction.shape[0] != 0 else None),
                        marker_channel=_iter_marker_channel)

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

            end_stage(_model_stage)
            _rtele.model_end(
                MODEL_RUN[jj], _tele_params, hp_tuning_cfg if _do_tuning else None,
                (train.shape[0], train.shape[1] - 1), valid.shape[0], test.shape[0],
                _tele_p_raw, _do_tuning,
                gpu_wait_s=((current_stage_timer().seconds('gpu_wait') - _tele_gpu_wait0)
                            if current_stage_timer() is not None else None))
            print(f"[GP] Task {task_num}/{total_tasks} | model {MODEL_RUN[jj]} finished: "
                  f"Pearson r={sample_pearson:.4f}, MSE={sample_mse:.4f} "
                  f"(took {time.time() - model_start_time:.1f}s)")
            completed_units += 1

        # ver4-35, R1.f: forget the worker-side data of this task (every
        # chain was collected or cancelled above; an exception skips this and
        # is handled by cancel_active_task_lanes() in the task wrapper).
        if _task_lane is not None:
            _task_lane.finish()

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
                (type(sample.loc[i,'ratio']) is tuple or (SCENARIO=='between' and _between_has_validation)):
            print(f"[GP] Task {i - idx*interval + 1}/{total_tasks} | NOTE: MODEL_DISPATCH_FILTER "
                  f"active - W_OPT weighted-ensemble finalisation skipped in this sub-run (see "
                  f"intra_task_parallel.py's eligibility gate; a task with W_OPT active and a "
                  f"validation split should have been marked ineligible for fan-out before "
                  f"reaching here - this task's weighted ensembles will be missing unless that "
                  f"gate is investigated).")
        elif W_OPT is not None and (type(sample.loc[i,'ratio']) is tuple or (SCENARIO=='between' and _between_has_validation)):
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

                    # Requirements.md (BLUEs/BLUPs accuracy-metrics phase)
                    # - `_asreml_accuracy_ctx` (this TASK's own context,
                    # resolved once in Step 6b above) is threaded through
                    # to each weighted-ensemble method so its own record
                    # row gains the SAME up-to-8 new columns a per-model
                    # row does; `None` (the default) whenever the feature
                    # is off, in which case each function's own new-
                    # metrics call is a strict no-op (see each module's
                    # own `compute_accuracy_metrics_row()` call site).
                    if W_OPT[kk]  == 'Linear transformation':
                        _record_out, _effect_out, predicted_test_sample, predicted_valid_sample, predicted_train_sample, _weight_out = Linear_transformation(result_train_sample, result_valid_sample, result_test_sample, _record_snapshot, _effect_snapshot, _weight_snapshot, _group_models, HYPERPARAMETERS_OPT['Linear transformation'], asreml_accuracy_ctx=_asreml_accuracy_ctx, sample_weight_ctx=_blue_weight_ctx)
                    elif W_OPT[kk] == 'Nelder Mead':
                        _record_out, _effect_out, predicted_test_sample, predicted_valid_sample, predicted_train_sample, _weight_out = Nelder_Mead(result_train_sample, result_valid_sample, result_test_sample, _record_snapshot, _effect_snapshot, _weight_snapshot, _group_models, HYPERPARAMETERS_OPT['Nelder Mead'], asreml_accuracy_ctx=_asreml_accuracy_ctx, sample_weight_ctx=_blue_weight_ctx)
                    elif W_OPT[kk] == 'Bayesian optimisation':
                        _record_out, _effect_out, predicted_test_sample, predicted_valid_sample, predicted_train_sample, _weight_out = Bayesian(result_train_sample, result_valid_sample, result_test_sample, _record_snapshot, _effect_snapshot, _weight_snapshot, _group_models, HYPERPARAMETERS_OPT['Bayesian optimisation'], asreml_accuracy_ctx=_asreml_accuracy_ctx, sample_weight_ctx=_blue_weight_ctx)
                    elif W_OPT[kk] == 'Analytic least-squares':
                        _record_out, _effect_out, predicted_test_sample, predicted_valid_sample, predicted_train_sample, _weight_out = Analytic_least_squares(result_train_sample, result_valid_sample, result_test_sample, _record_snapshot, _effect_snapshot, _weight_snapshot, _group_models, HYPERPARAMETERS_OPT['Analytic least-squares'], asreml_accuracy_ctx=_asreml_accuracy_ctx, sample_weight_ctx=_blue_weight_ctx)
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

        # ver4-26, R2/R5 (design blueprint `EasiGP_ver4-26_Design_Blueprint.md`):
        # combine this task's joint-group models using the weights "Step
        # 8.5" already searched for - a SIBLING branch to the W_OPT
        # `elif` above, not a fifth dispatch function: none of the four
        # existing weighted-ensemble modules (Linear_transformation.py/
        # Nelder_Mead.py/Bayesian_optimisation.py/Analytic_least_squares.py)
        # accept an already-decided weight vector as input, so this
        # replicates their own combination-arithmetic pattern directly
        # (weighted sum of predictions per split, one new record/effect/
        # weight row) rather than routing through one of them - see
        # models/Bayesian_optimisation.py's own tail (read in full while
        # designing this) for the pattern being mirrored. Threads
        # asreml_accuracy_ctx/sample_weight_ctx through exactly like every
        # other weighted-ensemble method already does (a blueprint gap -
        # its own R2 signature sketch omitted both - resolved via this
        # template's own "existing code behaviour" > blueprint authority
        # order; recorded in the Change Summary).
        elif _joint_active and _joint_weights and \
                (type(sample.loc[i,'ratio']) is tuple or (SCENARIO=='between' and _between_has_validation)):
            _joint_group_models_out = [m for m in _joint_group_models if m in result_test_sample.columns]
            if len(_joint_group_models_out) < 2:
                print(f"[GP] Task {task_num}/{total_tasks} | NOTE: Joint Bayesian optimisation - "
                      f"fewer than 2 joint-group models have a prediction column in this task's "
                      f"own result table ({_joint_group_models_out!r}) - combination skipped for "
                      f"this task.")
            else:
                _jw = np.array([float(_joint_weights.get(m, 0.0)) for m in _joint_group_models_out])
                _jw_abs_sum = float(np.abs(_jw).sum())
                if _jw_abs_sum > 0 and np.isfinite(_jw_abs_sum):
                    _jw = _jw / _jw_abs_sum
                else:
                    print(f"[GP] Task {task_num}/{total_tasks} | NOTE: Joint Bayesian "
                          f"optimisation - searched weights summed to ~0 ({_joint_weights!r}) - "
                          f"falling back to equal weighting across {_joint_group_models_out!r} "
                          f"for this task's combination (the same degenerate-weights safeguard "
                          f"models/Bayesian_optimisation.py's own weight search already "
                          f"applies).")
                    _jw = np.full(len(_joint_group_models_out), 1.0 / len(_joint_group_models_out))

                def _joint_weighted_sum(_df, _cols=_joint_group_models_out, _w=_jw):
                    _sel = _df.loc[:, _cols].astype(float)
                    return (_sel * _w).sum(axis=1).reset_index(drop=True)

                # ver4-38, R2.b2: with a library ensemble the joint prediction is the member sum
                # sum_k w_k * column_k, where a PRIMARY member's column is the model's Step 9
                # column in the result table and an EXTRA member's column is the train-only
                # confirm fit made by the search. Every length is asserted; any mismatch ignores
                # ALL extras for this task (the per-model sums `_jw` then weight the primaries).
                _joint_members_use = []
                if _joint_library_members:
                    _lm_problem = None
                    for _mem in _joint_library_members:
                        if _mem['model'] not in _joint_group_models_out:
                            _lm_problem = f"member model {_mem['model']!r} has no result column"
                            break
                        if not _mem['primary']:
                            _lm_pred = _mem.get('pred') or {}
                            for _split, _frame in (('test', result_test_sample),
                                                   ('valid', result_valid_sample),
                                                   ('train', result_train_sample)):
                                _arr = _lm_pred.get(_split)
                                if _arr is None or len(_arr) != len(_frame):
                                    _lm_problem = (
                                        f"extra {_mem['model']!r} {_split} prediction has "
                                        f"{None if _arr is None else len(_arr)} row(s), the "
                                        f"result table has {len(_frame)}")
                                    break
                        if _lm_problem:
                            break
                    if _lm_problem:
                        print(f"[GP] Task {task_num}/{total_tasks} | NOTE: Joint Bayesian "
                              f"optimisation library_ensemble extras ignored for this task: "
                              f"{_lm_problem}.")
                    else:
                        _joint_members_use = list(_joint_library_members)

                def _joint_member_sum(_df, _split):
                    _tot = np.zeros(len(_df), dtype=float)
                    for _mem in _joint_members_use:
                        if _mem['primary']:
                            _col = _df[_mem['model']].astype(float).values
                        else:
                            _col = np.asarray(_mem['pred'][_split], dtype=float)
                        _tot = _tot + float(_mem['weight']) * _col
                    return pd.Series(_tot)

                if _joint_members_use:
                    predicted_test_j = _joint_member_sum(result_test_sample, 'test')
                    predicted_valid_j = _joint_member_sum(result_valid_sample, 'valid')
                    predicted_train_j = _joint_member_sum(result_train_sample, 'train')
                else:
                    predicted_test_j = _joint_weighted_sum(result_test_sample)
                    predicted_valid_j = _joint_weighted_sum(result_valid_sample)
                    predicted_train_j = _joint_weighted_sum(result_train_sample)

                actual_test_arr = result_test_sample['actual'].astype(float).values
                r_unweighted, mse_unweighted = safe_regression_metrics(actual_test_arr, predicted_test_j.values)
                _test_weight_kwargs = blue_weights.scoring_for(_blue_weight_ctx, 'test')
                if _test_weight_kwargs:
                    _r_j, _mse_j = safe_regression_metrics(
                        actual_test_arr, predicted_test_j.values, **_test_weight_kwargs)
                else:
                    _r_j, _mse_j = r_unweighted, mse_unweighted

                _record_snapshot = record.copy()

                _new_record = pd.concat(
                    [_record_snapshot, pd.DataFrame(_record_snapshot.iloc[-1, :]).T]
                ).reset_index(drop=True)
                _new_record.loc[_new_record.shape[0]-1, 'model'] = 'Joint Bayesian optimisation'
                _new_record.loc[_new_record.shape[0]-1, 'Pearson correlation'] = _r_j
                _new_record.loc[_new_record.shape[0]-1, 'MSE'] = _mse_j
                if _blue_weight_ctx is not None:
                    _new_record.loc[_new_record.shape[0]-1, 'Pearson correlation (unweighted)'] = r_unweighted
                    _new_record.loc[_new_record.shape[0]-1, 'MSE (unweighted)'] = mse_unweighted
                    _new_record.loc[_new_record.shape[0]-1, 'metric_weighting'] = _blue_weight_ctx.mode
                if _asreml_accuracy_ctx:
                    _new_metrics = asreml_accuracy_metrics.compute_accuracy_metrics_row(
                        list(predicted_test_j),
                        result_test_sample['id'].tolist() if 'id' in result_test_sample.columns else [],
                        _r_j, _asreml_accuracy_ctx,
                    )
                    for _col, _val in _new_metrics.items():
                        _new_record.loc[_new_record.shape[0]-1, _col] = _val
                _new_record = _new_record.iloc[_record_snapshot.shape[0]:]

                _weight_extracted = pd.DataFrame([_jw], columns=_joint_group_models_out)
                _weight_sample = pd.DataFrame(_new_record.iloc[-1, :]).T.drop(
                    ['Pearson correlation', 'MSE'] + list(asreml_accuracy_metrics.ACCURACY_METRIC_COLUMNS)
                    + list(blue_weights.METRIC_WEIGHTING_EXTRA_COLUMNS),
                    axis=1, errors='ignore',
                )
                _weight_sample['model'] = 'Joint Bayesian optimisation'
                _weight_sample = pd.concat(
                    [_weight_sample.reset_index(drop=True), _weight_extracted.reset_index(drop=True)], axis=1)
                _new_weight = _weight_sample

                _weight_extracted_normalised = _joint_safe_row_normalize(_weight_extracted)
                _effect_weighted = None
                for _m in _joint_group_models_out:
                    _m_effect_rows = effect[effect['model'] == _m]
                    if _m_effect_rows.shape[0] == 0:
                        continue
                    _m_contribution = _joint_safe_row_normalize(
                        _m_effect_rows.tail(1).iloc[:, 5:].abs().reset_index(drop=True)
                    ).mul(_weight_extracted_normalised[_m], axis=0).reset_index(drop=True)
                    _effect_weighted = _m_contribution if _effect_weighted is None else _effect_weighted + _m_contribution

                _new_effect = pd.DataFrame()
                if _effect_weighted is not None and effect.shape[0] != 0:
                    _new_effect = pd.DataFrame(effect.iloc[-1, :]).T.reset_index(drop=True)
                    _new_effect['model'] = 'Joint Bayesian optimisation'
                    _new_effect.loc[:, list(_effect_weighted.columns)] = _effect_weighted.values

                record = pd.concat([record, _new_record])
                effect = pd.concat([effect, _new_effect])
                weight = pd.concat([weight, _new_weight])

                for _df, _pred in (
                        (result_test_sample, predicted_test_j),
                        (result_valid_sample, predicted_valid_j),
                        (result_train_sample, predicted_train_j)):
                    _df['Joint Bayesian'] = _pred

                if _new_weight.shape[0] != 0 and interactions.shape[0] != 0:
                    _model_weights = {_m: float(_jw[_idx]) for _idx, _m in enumerate(_joint_group_models_out)}
                    _new_interaction_ensemble = weighted_ensemble_interactions(
                        interactions, _joint_group_models_out, _model_weights,
                        sample.loc[i, 'population'], sample.loc[i, 'phenotype'],
                        sample.loc[i, 'ratio'], sample.loc[i, 'sample'], 'Joint Bayesian optimisation',
                    )
                    interactions = pd.concat([interactions, _new_interaction_ensemble])

                print(f"[GP] Task {task_num}/{total_tasks} | Joint Bayesian optimisation "
                      f"combination: Pearson r={_r_j:.4f}, MSE={_mse_j:.4f} -> "
                      f"weights={dict(zip(_joint_group_models_out, _jw.tolist()))}"
                      + (f" | library_ensemble: {len(_joint_members_use)} member(s) "
                         f"({sum(1 for _x in _joint_members_use if _x['primary'])} primary + "
                         f"{sum(1 for _x in _joint_members_use if not _x['primary'])} extra); "
                         f"weights = per-model member sums; effect/interaction rows use these "
                         f"sums x the primaries' effects (approximation)"
                         if _joint_members_use else ""))

        # ------------------------------------------------------------------------------------ #
        # ver4-57 (R2/R5), Step 11c: the opt-in ITERATIVE ENSEMBLE. A sibling of the W_OPT and
        # joint branches above, never part of them, and entirely skipped (I11) unless
        # ITERATIVE_ENSEMBLE is set AND this task captured its members (validation split, full
        # model list). For each ensemble group: one consensus marker ranking from the Step 9
        # effects, refits of the non-bio-prior members on nested marker subsets (validation
        # decides acceptance, I7), and ONE new 'Iterative ensemble' row whose predictions are the
        # equal-weight mean of the selected round. Everything is built first and appended last,
        # so a failure leaves this task's tables exactly as Step 11 left them.
        # ------------------------------------------------------------------------------------ #
        if _iter_capture:
            _iter_stage = begin_stage('iter_ens')
            try:
                _iter_chrom = None
                if bim_marker_info is not None:
                    _iter_chrom = dict(zip(bim_marker_info['name'].astype(str),
                                           bim_marker_info['chromosome'].astype(str)))
                _iter_task_t0 = time.time()
                for _ie_group_label, _ie_group_models in ensemble_groups(
                        MODEL_BASE, HP_TUNE, HP_TUNE_ENSEMBLE_MODE):
                    _ie_label = ('Iterative ensemble' if _ie_group_label is None
                                 else f'Iterative ensemble__{_ie_group_label}')
                    _ie_members = [_iter_members[m] for m in _ie_group_models
                                   if m in _iter_members and m in result_test_sample.columns
                                   and _iter_skip_of.get(m) is None]   # ver4-77: see _sl_capture
                    if len(_ie_members) < 2:
                        print(f"[GP] Task {task_num}/{total_tasks} | NOTE: ITERATIVE_ENSEMBLE - "
                              f"fewer than 2 usable members for group "
                              f"{_ie_group_label if _ie_group_label is not None else '(all models)'}"
                              f" ({[m.name for m in _ie_members]!r}) - no '{_ie_label}' row.")
                        continue

                    # Parallel width: the same double gate HP_TUNE's trial fan-out uses
                    # (hp_tune_parallel_trials) - one process per refit member, at most N_JOBS.
                    _ie_n_refit = sum(1 for m in _ie_members if m.marker_channel)
                    if _iter_cfg['mode'] == 'two_phase':
                        # ver4-68 (R2): a wave is one parallel call, so the width is bounded by
                        # the largest wave (out-of-fold units included), not by the member count.
                        _ie_n_refit = max(two_phase_wave_sizes(_ie_members, _iter_cfg) or [1])
                    _ie_width = (max(1, min(int(_compute_resources.get('n_jobs') or 1), _ie_n_refit))
                                 if _compute_resources['hp_tune_parallel_trials'] else 1)
                    _ie_resources = _trial_worker_resources_for(_compute_resources, _ie_width)
                    if _ie_width > 1:
                        ensure_gpu_arbiter(_compute_resources)
                    _ie_ctx = dict(
                        n_jobs=_ie_width, worker_init=_trial_worker_init,
                        worker_init_args=(RESULT_NAME, R_PATH, _ie_resources, get_gpu_semaphore()))
                    _ie_runner = ModelTrialRunner(
                        RESULT_NAME, R_PATH, _compute_resources, GENOTYPE_FORMAT,
                        bio_prior_ctx=None, sample_weight_ctx=_blue_weight_ctx)

                    _ie_res = run_iterative_ensemble(
                        _ie_members, _ie_members[0].valid.iloc[:, -1].to_numpy(dtype=float),
                        settings=_iter_cfg, runner=_ie_runner, RESULT_NAME=RESULT_NAME,
                        runner_ctx=_ie_ctx,
                        valid_score_kwargs=blue_weights.scoring_for(_blue_weight_ctx, 'valid'),
                        chrom_of_marker=_iter_chrom)
                    for _note in _ie_res.notes:
                        print(f"[GP] Task {task_num}/{total_tasks} | NOTE: ITERATIVE_ENSEMBLE "
                              f"({_ie_label}) - {_note}")
                    if not _ie_res.pred:
                        print(f"[GP] Task {task_num}/{total_tasks} | ITERATIVE_ENSEMBLE: "
                              f"no '{_ie_label}' row for this group (see the note above).")
                        continue

                    _ie_out = [m for m in _ie_res.members if m in result_test_sample.columns]
                    _ie_pred_test = pd.Series(_ie_res.pred['test'])
                    _ie_pred_valid = pd.Series(_ie_res.pred['valid'])
                    _ie_pred_train = pd.Series(_ie_res.pred['train'])
                    _ie_actual = result_test_sample['actual'].astype(float).values
                    _ie_r_unw, _ie_mse_unw = safe_regression_metrics(_ie_actual, _ie_pred_test.values)
                    _ie_test_kw = blue_weights.scoring_for(_blue_weight_ctx, 'test')
                    if _ie_test_kw:
                        _ie_r, _ie_mse = safe_regression_metrics(
                            _ie_actual, _ie_pred_test.values, **_ie_test_kw)
                    else:
                        _ie_r, _ie_mse = _ie_r_unw, _ie_mse_unw

                    # record row (the SAME construction the joint branch uses)
                    _ie_rec_snap = record.copy()
                    _ie_rec = pd.concat(
                        [_ie_rec_snap, pd.DataFrame(_ie_rec_snap.iloc[-1, :]).T]
                    ).reset_index(drop=True)
                    _ie_rec.loc[_ie_rec.shape[0]-1, 'model'] = _ie_label
                    _ie_rec.loc[_ie_rec.shape[0]-1, 'Pearson correlation'] = _ie_r
                    _ie_rec.loc[_ie_rec.shape[0]-1, 'MSE'] = _ie_mse
                    if _blue_weight_ctx is not None:
                        _ie_rec.loc[_ie_rec.shape[0]-1, 'Pearson correlation (unweighted)'] = _ie_r_unw
                        _ie_rec.loc[_ie_rec.shape[0]-1, 'MSE (unweighted)'] = _ie_mse_unw
                        _ie_rec.loc[_ie_rec.shape[0]-1, 'metric_weighting'] = _blue_weight_ctx.mode
                    if _asreml_accuracy_ctx:
                        _ie_metrics = asreml_accuracy_metrics.compute_accuracy_metrics_row(
                            list(_ie_pred_test),
                            result_test_sample['id'].tolist() if 'id' in result_test_sample.columns else [],
                            _ie_r, _asreml_accuracy_ctx)
                        for _col, _val in _ie_metrics.items():
                            _ie_rec.loc[_ie_rec.shape[0]-1, _col] = _val
                    _ie_new_record = _ie_rec.iloc[_ie_rec_snap.shape[0]:]

                    # weight row: equal weights over the averaged members; ver4-68 two_phase: the
                    # share each member carries in the pooled prediction (w0, or the library
                    # column share with weight_rule='equal'). `member_weights` is None in 'subset'.
                    if _ie_res.member_weights is None:
                        _ie_w = np.full(len(_ie_out), 1.0 / len(_ie_out))
                    else:
                        _ie_w = np.array([float(_ie_res.member_weights.get(_m, 0.0))
                                          for _m in _ie_out])
                        _ie_w = _ie_w / _ie_w.sum()
                    _ie_w_frame = pd.DataFrame([_ie_w], columns=_ie_out)
                    _ie_new_weight = pd.DataFrame(_ie_new_record.iloc[-1, :]).T.drop(
                        ['Pearson correlation', 'MSE'] + list(asreml_accuracy_metrics.ACCURACY_METRIC_COLUMNS)
                        + list(blue_weights.METRIC_WEIGHTING_EXTRA_COLUMNS),
                        axis=1, errors='ignore')
                    _ie_new_weight['model'] = _ie_label
                    _ie_new_weight = pd.concat(
                        [_ie_new_weight.reset_index(drop=True), _ie_w_frame.reset_index(drop=True)],
                        axis=1)

                    # effect row: the Step 9 equal-weight consensus (same arithmetic as the naive
                    # ensemble's ring). A marker the selected round dropped keeps its Step 9
                    # value - it is never shown as zero; with explain_selected the refit values
                    # replace the kept markers' entries at the same total mass.
                    _ie_new_effect = pd.DataFrame()
                    _ie_cons = None
                    for _m in (_ie_out if effect.shape[0] != 0 else []):
                        _m_rows = effect[effect['model'] == _m]
                        if _m_rows.shape[0] == 0:
                            continue
                        _m_norm = _joint_safe_row_normalize(
                            _m_rows.tail(1).iloc[:, 5:].abs().reset_index(drop=True))
                        if _ie_res.member_weights is None:
                            _m_part = _m_norm / len(_ie_out)          # subset: equal (unchanged)
                        else:
                            # ver4-68 (D6): the w-weighted Step 9 consensus - no explained refit
                            _m_part = _m_norm * float(_ie_w[_ie_out.index(_m)])
                        _ie_cons = _m_part if _ie_cons is None else _ie_cons + _m_part
                    if _ie_cons is not None and effect.shape[0] != 0:
                        if _ie_res.effect_refit and _ie_res.kept is not None:
                            _ie_names = [str(n) for n, k in zip(_ie_res.effect.index, _ie_res.kept) if k]
                            _ie_names = [n for n in _ie_names if n in _ie_cons.columns]
                            _ie_refit = _ie_res.effect.copy()
                            _ie_refit.index = [str(n) for n in _ie_refit.index]
                            _ie_refit_vals = _ie_refit.loc[_ie_names].to_numpy(dtype=float)
                            _ie_mass = float(_ie_cons.loc[0, _ie_names].sum())
                            if _ie_refit_vals.sum() > 0 and _ie_mass > 0:
                                _ie_cons.loc[0, _ie_names] = _ie_refit_vals * (_ie_mass / _ie_refit_vals.sum())
                        _ie_new_effect = pd.DataFrame(effect.iloc[-1, :]).T.reset_index(drop=True)
                        _ie_new_effect['model'] = _ie_label
                        _ie_new_effect.loc[:, list(_ie_cons.columns)] = _ie_cons.values

                    # k=2 interaction ring from the Step 9 tables (single authority, I3)
                    _ie_new_inter = pd.DataFrame()
                    if interactions.shape[0] != 0:
                        _ie_new_inter = weighted_ensemble_interactions(
                            interactions, _ie_out, {_m: float(_w) for _m, _w in zip(_ie_out, _ie_w)},
                            sample.loc[i, 'population'], sample.loc[i, 'phenotype'],
                            sample.loc[i, 'ratio'], sample.loc[i, 'sample'], _ie_label)

                    # per-round diagnostics (test columns are reporting only, never a decision)
                    _ie_round_rows = []
                    if _iter_cfg['write_rounds']:
                        _ie_pool = (list(_ie_res.effect.index) if _ie_res.effect is not None else [])
                        for _rd, _rtest in zip(_ie_res.rounds, _ie_res.round_test_pred):
                            _rd_r, _rd_mse = safe_regression_metrics(_ie_actual, _rtest)
                            _kept_txt = ''
                            if _iter_cfg['write_kept_markers'] and _rd['kept_mask'] is not None:
                                _kept_txt = ';'.join(
                                    str(n) for n, k in zip(_ie_pool, _rd['kept_mask']) if k)
                            _ie_round_rows.append({
                                'population': sample.loc[i, 'population'],
                                'phenotype': sample.loc[i, 'phenotype'],
                                'model': _ie_label, 'ratio': sample.loc[i, 'ratio'],
                                'sample': sample.loc[i, 'sample'], 'round': _rd['round'],
                                'n_markers_kept': _rd['n_markers_kept'],
                                'n_reentered': _rd['n_reentered'], 'kept_markers': _kept_txt,
                                'accepted': _rd['accepted'],
                                'selected': _rd.get('selected', _rd['round'] == _ie_res.best_round),
                                'valid_pearson': _rd['valid_pearson'],
                                'valid_gain': _rd['valid_gain'],
                                'valid_gain_sd': _rd['valid_gain_sd'],
                                'test_pearson_diagnostic': _rd_r,
                                'test_mse_diagnostic': _rd_mse,
                                'n_fits': _rd['n_fits'], 'seconds': _rd['seconds']})

                    # ---- everything built: append it all ----
                    record = pd.concat([record, _ie_new_record])
                    weight = pd.concat([weight, _ie_new_weight])
                    if _ie_new_effect.shape[0] != 0:
                        effect = pd.concat([effect, _ie_new_effect])
                    if _ie_new_inter.shape[0] != 0:
                        interactions = pd.concat([interactions, _ie_new_inter])
                    if _ie_round_rows:
                        iter_rounds = pd.concat([iter_rounds, pd.DataFrame(_ie_round_rows)])
                    for _df, _pred in ((result_test_sample, _ie_pred_test),
                                       (result_valid_sample, _ie_pred_valid),
                                       (result_train_sample, _ie_pred_train)):
                        _df[_ie_label] = _pred.values

                    if _ie_res.diag is not None:
                        # ver4-68 (R1): the two-phase library - one line per snapshot (the
                        # `seconds` of a snapshot is the wall time of the WAVE that fitted it, so
                        # rows of one wave share it) and one summary line with the weights.
                        for _rd in _ie_res.rounds:
                            print(f"[GP] Task {task_num}/{total_tasks} | {_ie_label} round "
                                  f"{_rd['round']} ({_rd['phase']}): "
                                  f"markers={_rd['n_markers_kept']}, "
                                  f"valid r={_rd['valid_pearson']:.4f} (snapshot at the pool "
                                  f"weights), fits so far={_rd['n_fits']}, "
                                  f"wave {_rd['seconds']:.1f}s")
                        _ie_dg = _ie_res.diag
                        _ie_w_txt = {_m: round(float(_w), 3) for _m, _w in zip(_ie_out, _ie_w)}
                        print(f"[GP] Task {task_num}/{total_tasks} | {_ie_label}: two_phase "
                              f"tier '{_ie_dg['tier']}' ({_ie_dg['weight_rule']}), "
                              f"w0 source={_ie_dg['w0_source']}, "
                              f"alpha*={_ie_dg['alpha_star']:.2f}, "
                              f"w0={_ie_w_txt}, "
                              f"phase-1 rounds={_ie_dg['p1_rounds_done']}, "
                              f"phase-2 rounds={_ie_dg['p2_rounds_done']}, "
                              f"waves={_ie_dg['n_waves_run']}/{_ie_dg['n_waves_planned']}, "
                              f"extra fits={_ie_res.n_fits}, Pearson r={_ie_r:.4f}, "
                              f"MSE={_ie_mse:.4f}, members={_ie_out!r} "
                              f"(took {time.time() - _iter_task_t0:.1f}s)")
                        # ver4-69 (R1): where does the row gain or lose against the naive
                        # ensemble? Test columns are reporting only (never a decision).
                        if _ie_res.attribution:
                            _ie_att_names = (
                                ('step9_equal', 'naive ensemble (Step 9, equal)'),
                                ('step9_w0', 'Step 9 at w0'),
                                ('pool_equal', 'snapshot pool (equal members)'),
                                ('final', 'final row'))
                            _ie_att_parts = []
                            for _lbl, _txt in _ie_att_names:
                                if _lbl in _ie_res.attribution:
                                    _a_r, _a_mse = safe_regression_metrics(
                                        _ie_actual, _ie_res.attribution[_lbl])
                                    _ie_att_parts.append(f"{_txt}: r={_a_r:.4f} MSE={_a_mse:.4f}")
                            print(f"[GP] Task {task_num}/{total_tasks} | {_ie_label} attribution "
                                  f"(test, reporting only): " + " | ".join(_ie_att_parts))
                        _ie_ev = []
                        if _ie_dg.get('w0_path'):
                            _ie_cv = _ie_dg['w0_path']['cv_mse']
                            _ie_ev.append(
                                f"weights fitted on {_ie_dg['n_weight_rows']} rows, alpha cap "
                                f"{_ie_dg['alpha_max']:g}, CV MSE by alpha "
                                f"{ {round(a, 2): round(v, 4) for a, v in _ie_cv.items()} }")
                        if _ie_dg.get('oof_fit'):
                            _ie_ev.append("out-of-fold r per model "
                                          f"{ {n: round(v[0], 3) for n, v in _ie_dg['oof_fit'].items()} }")
                        if _ie_dg.get('valid_fit'):
                            _ie_ev.append("validation r per model "
                                          f"{ {n: round(v[0], 3) for n, v in _ie_dg['valid_fit'].items()} }")
                        if _ie_ev:
                            print(f"[GP] Task {task_num}/{total_tasks} | {_ie_label} weight "
                                  f"evidence: " + "; ".join(_ie_ev))
                    else:
                        for _rd in _ie_res.rounds:
                            print(f"[GP] Task {task_num}/{total_tasks} | {_ie_label} round "
                                  f"{_rd['round']}: markers={_rd['n_markers_kept']} "
                                  f"(re-entered {_rd['n_reentered']}), "
                                  f"valid r={_rd['valid_pearson']:.4f}, "
                                  f"gain={_rd['valid_gain']:.4f} (sd {_rd['valid_gain_sd']:.4f}), "
                                  f"accepted={_rd['accepted']}, fits so far={_rd['n_fits']}, "
                                  f"{_rd['seconds']:.1f}s")
                        _ie_sel_txt = ('naive ensemble kept' if _ie_res.best_round == 0
                                       else 'refitted members')
                        print(f"[GP] Task {task_num}/{total_tasks} | {_ie_label}: selected round "
                              f"{_ie_res.best_round} of {len(_ie_res.rounds) - 1} tried "
                              f"({_ie_sel_txt}), "
                              f"Pearson r={_ie_r:.4f}, MSE={_ie_mse:.4f}, "
                              f"extra fits={_ie_res.n_fits}, "
                              f"members={_ie_out!r} (took {time.time() - _iter_task_t0:.1f}s)")
            except Exception:
                print(f"[GP] Task {task_num}/{total_tasks} | NOTE: ITERATIVE_ENSEMBLE failed - "
                      f"this task keeps the result tables exactly as Step 11 left them (no "
                      f"'Iterative ensemble' row); the other rows are unaffected.\n"
                      f"{traceback.format_exc()}")
            finally:
                end_stage(_iter_stage)

        # ------------------------------------------------------------------------------------ #
        # ver4-77 (R2), Step 11d: the opt-in SUPER LEARNER. A sibling of the W_OPT, joint and
        # iterative branches above, never part of them, and entirely skipped (I11) unless
        # SUPER_LEARNER is set AND this task captured its members (a full-model run). For each
        # ensemble group: every member is fitted out of fold over train+valid, one weight vector
        # is chosen by cross-validation over those columns, every member is refitted once on
        # train+valid (with a validation split) and ONE new 'Super learner' row is written. Its
        # train/valid prediction columns are OUT-OF-FOLD (P[rows] @ w); the other models' are
        # in-sample. Everything is built first and appended last (per group), so a failure leaves
        # this task's tables exactly as the steps above left them. I7 (amended): test rows never
        # enter a fit, a weight or a selection.
        # ------------------------------------------------------------------------------------ #
        _sl_has_valid = _iter_has_valid     # the same "does this task have a validation split"
        _sl_skip_reason = None
        if _sl_capture:
            if (SCENARIO == 'between' and _between_validation_source is not None
                    and _between_validation_source not in _sl_cfg['between_sources']):
                _sl_skip_reason = (f"SCENARIO='between' with validation source "
                                   f"{_between_validation_source!r}, which is not in "
                                   f"between_sources={_sl_cfg['between_sources']!r} (refitting on "
                                   f"those rows would put target-population lines into a fit)")
            elif (_blue_weight_ctx is not None and _sl_has_valid
                  and getattr(_blue_weight_ctx, 'w_fit_cv', None) is None):
                _sl_skip_reason = ("BLUE weighting is on but its train+valid fitting weights "
                                   "could not be built for this task (see the [blue_weights] "
                                   "lines above)")
        if _sl_capture and _sl_skip_reason is not None:
            print(f"[GP] Task {task_num}/{total_tasks} | NOTE: SUPER_LEARNER - {_sl_skip_reason} "
                  f"- no 'Super learner' row for this task.")
        if _sl_capture and _sl_skip_reason is None:
            _sl_stage = begin_stage('super_learner')
            try:
                _sl_task_t0 = time.time()
                # ver4-94 (R3): the other populations' lines for the pooled columns, built ONCE per
                # task (its inputs are task-level). None = no pooled row (a NOTE says why).
                _sl_pop_src = None
                if 'pooled' in _sl_cfg['pop_sources']:
                    def _sl_pop_loader_factory(_marker_cols):
                        def _load(_ids):
                            _ids = [str(v) for v in _ids]
                            if GENOTYPE_FORMAT == 'csv':
                                _g = data_genotype_original
                                _sel = _g['ID'].astype(str).isin(set(_ids))
                                _sub = _g.loc[_sel, ['ID'] + list(_marker_cols)].copy()
                                # the task's own CSV merge drops rows with a missing genotype
                                # value; the source follows the same rule
                                return _sub.dropna().reset_index(drop=True)
                            _ids = [v for v in _ids if v in fam_iid_set]
                            _pheno = (data_phenotype_original.loc[
                                data_phenotype_original['ID'].astype(str).isin(set(_ids)),
                                ['ID', 'population']].drop_duplicates(subset='ID'))
                            _geno = plink_to_genotype_df(
                                GENOTYPE_FILE_NAME, _pheno, extract_snp_list=list(_marker_cols),
                                keep_iid_list=_ids, plink_path=GENOTYPE_PLINK_PATH,
                                work_dir=_new_plink_work_dir())
                            _mc = list(_marker_cols)
                            # PLINK missing calls: mean-imputed on this frame, as the task's own
                            # splits are (see _attach_phenotype)
                            if _geno[_mc].isna().any().any():
                                _geno[_mc] = _geno[_mc].fillna(_geno[_mc].mean()).fillna(0)
                            return _geno[['ID'] + _mc]
                        return _load
                    _sl_pop_src = _sl_pooled_source(
                        f"Task {task_num}/{total_tasks}", _sl_cfg, SCENARIO, ASREML is not None,
                        _blue_weight_ctx is not None, data_phenotype_original,
                        sample.loc[i, 'phenotype'], sample.loc[i, 'population'],
                        {'train': set(map(str, id_train)), 'valid': set(map(str, id_valid)),
                         'test': set(map(str, id_test))},
                        [_iter_members[m] for m in _iter_members], sample.loc[i, 'sample'],
                        _sl_pop_loader_factory)
                for _sl_group_label, _sl_group_models in ensemble_groups(
                        MODEL_BASE, HP_TUNE, HP_TUNE_ENSEMBLE_MODE):
                    _sl_label = ('Super learner' if _sl_group_label is None
                                 else f'Super learner__{_sl_group_label}')
                    _sl_group_txt = (_sl_group_label if _sl_group_label is not None
                                     else '(all models)')
                    _sl_members = [_iter_members[m] for m in _sl_group_models
                                   if m in _iter_members and m in result_test_sample.columns]
                    for _m in _sl_group_models:
                        if _m not in _iter_members and _m in result_test_sample.columns:
                            print(f"[GP] Task {task_num}/{total_tasks} | NOTE: SUPER_LEARNER - "
                                  f"model {_m} is left out of the Super learner: its Step 9 "
                                  f"hyperparameters could not be resolved.")
                    _sl_bio = [m.name for m in _sl_members if not m.marker_channel]
                    _sl_usable = [m for m in _sl_members if m.marker_channel]
                    if _sl_bio:
                        print(f"[GP] Task {task_num}/{total_tasks} | NOTE: SUPER_LEARNER - "
                              f"{_sl_bio!r} is/are bio-prior and never fitted out of fold (I6): "
                              f"left out of the Super learner for group {_sl_group_txt}.")
                    if len(_sl_usable) < 2:
                        print(f"[GP] Task {task_num}/{total_tasks} | NOTE: SUPER_LEARNER - fewer "
                              f"than 2 usable members for group {_sl_group_txt} "
                              f"({[m.name for m in _sl_usable]!r}) - no '{_sl_label}' row.")
                        continue

                    # Parallel width: the same double gate HP_TUNE's trial fan-out uses - one
                    # process per fit unit of the single wave, at most N_JOBS.
                    _sl_n_oof, _sl_n_refit = sl_unit_counts(
                        len(_sl_usable), _sl_cfg, _sl_has_valid,
                        n_pooled=(len(_sl_usable) if _sl_pop_src is not None else 0))
                    # ver4-97: N_JOBS=-1 ("every core") used to give width 1 here (see
                    # _sl_parallel_width), i.e. the whole stage ran on one core.
                    _sl_n_markers = max((m.train.shape[1] - 1 for m in _sl_usable), default=0)
                    _sl_rows_peak = (_sl_usable[0].train.shape[0] + _sl_usable[0].valid.shape[0]
                                     + (_sl_pop_src.shape[0] if _sl_pop_src is not None else 0))
                    _sl_width, _sl_width_note = _sl_parallel_width(
                        _compute_resources, _sl_n_oof + _sl_n_refit, _sl_rows_peak,
                        _sl_n_markers, f"Task {task_num}/{total_tasks} | SUPER_LEARNER")
                    if _sl_width_note:
                        print(_sl_width_note)
                    print(f"[GP] Task {task_num}/{total_tasks} | SUPER_LEARNER fit plan: "
                          f"{_sl_n_oof + _sl_n_refit} fit(s) (out-of-fold {_sl_n_oof}, refit "
                          f"{_sl_n_refit}), parallel width {_sl_width}, up to {_sl_rows_peak} "
                          f"training rows per fit.", flush=True)
                    _sl_resources = _trial_worker_resources_for(_compute_resources, _sl_width)
                    if _sl_width > 1:
                        ensure_gpu_arbiter(_compute_resources)
                    _sl_ctx = dict(
                        n_jobs=_sl_width, worker_init=_trial_worker_init,
                        worker_init_args=(RESULT_NAME, R_PATH, _sl_resources, get_gpu_semaphore()))
                    _sl_runner = ModelTrialRunner(
                        RESULT_NAME, R_PATH, _compute_resources, GENOTYPE_FORMAT,
                        bio_prior_ctx=None, sample_weight_ctx=_blue_weight_ctx)

                    # ver4-80 (R1): which weight methods get a row - the ticked W_OPT methods
                    # (ver4-83: none ticked = headless fallback, the legacy analytic row only)
                    _sl_methods = sl_weight_methods(_sl_cfg, W_OPT)
                    _sl_row_labels = sl_row_labels(_sl_cfg, W_OPT, _sl_group_label)
                    _sl_wopt_hp = {_m: (HYPERPARAMETERS_OPT or {}).get(_m)
                                   for _m in _sl_methods}
                    print(f"[GP] Task {task_num}/{total_tasks} | SUPER_LEARNER weight methods for "
                          f"group {_sl_group_txt}: {_sl_methods!r} -> rows "
                          f"{list(_sl_row_labels.values())!r} (W_OPT={W_OPT!r}, SUPER_LEARNER "
                          f"weight_optimisers={_sl_cfg.get('weight_optimisers')!r}).")
                    _sl_res = run_super_learner(
                        _sl_usable, settings=_sl_cfg, runner=_sl_runner, RESULT_NAME=RESULT_NAME,
                        runner_ctx=_sl_ctx, use_cv_weights=_blue_weight_ctx is not None,
                        weight_methods=_sl_methods, weight_hparams=_sl_wopt_hp,
                        extra_sources=({'pooled': _sl_pop_src} if _sl_pop_src is not None
                                       else None))
                    for _note in _sl_res.notes:
                        print(f"[GP] Task {task_num}/{total_tasks} | NOTE: SUPER_LEARNER "
                              f"({_sl_label}) - {_note}")
                    if not _sl_res.pred:
                        print(f"[GP] Task {task_num}/{total_tasks} | NOTE: SUPER_LEARNER - no "
                              f"'{_sl_label}' row for group {_sl_group_txt} (see the note "
                              f"above).")
                        continue

                    # ver4-80 (R1): one row per weight method (the legacy analytic row, then
                    # one per ticked W_OPT method). The member fits ran once above; each
                    # variant only re-weights the same out-of-fold matrix. A failure in one
                    # variant leaves the others (and every other table) as they are.
                    # ver4-94 (R3): the unchanged rows first, then the optional pooled row (a
                    # tagged row is written exactly like the others; its member weights already
                    # sum each member's own and pooled weight)
                    _sl_tag_labels = sl_tagged_row_labels(
                        _sl_cfg, W_OPT, 'pooled', _sl_group_label)
                    _sl_row_items = [(_sl_row_labels[_m], _m, _v, False)
                                     for _m, _v in _sl_res.variants.items()]
                    _sl_row_items += [(_sl_tag_labels[_m], _m, _v, True)
                                      for _m, _v in _sl_res.tagged_variants.get('pooled', {}).items()
                                      if _m in _sl_tag_labels]
                    # ver4-96: the optional per-member pooled rows (each model's own prediction
                    # when fitted with the other populations' lines; metrics and predictions only)
                    _sl_mem_labels = sl_member_row_labels(
                        _sl_cfg, W_OPT, _sl_res.members, 'pooled', _sl_group_label)
                    _sl_row_items += [(_sl_mem_labels[_m], _m, _v, 'member')
                                      for _m, _v in _sl_res.tagged_variants.get(
                                          'pooled_member', {}).items() if _m in _sl_mem_labels]
                    # pool-fix: the gated per-member rows (own + pooled column, shrunk to own)
                    _sl_gat_labels = sl_member_row_labels(
                        _sl_cfg, W_OPT, _sl_res.members, 'pooled_gated', _sl_group_label)
                    _sl_row_items += [(_sl_gat_labels[_m], _m, _v, 'member_gated')
                                      for _m, _v in _sl_res.tagged_variants.get(
                                          'pooled_member_gated', {}).items()
                                      if _m in _sl_gat_labels]
                    for _sl_label, _sl_method, _sl_var, _sl_is_pooled in _sl_row_items:
                        try:
                            _sl_out = [m for m in _sl_res.members if m in result_test_sample.columns]
                            _sl_w = np.array([float(_sl_var.weights[_m]) for _m in _sl_out])
                            _sl_w = _sl_w / _sl_w.sum()
                            _sl_pred_test = pd.Series(_sl_var.pred['test'])
                            _sl_pred_valid = pd.Series(_sl_var.pred['valid'])
                            _sl_pred_train = pd.Series(_sl_var.pred['train'])
                            # the prediction columns must line up with the result frames row for row
                            # (I4): check BEFORE anything is appended, so a desync leaves no half row.
                            for _df, _pred, _split in ((result_test_sample, _sl_pred_test, 'test'),
                                                       (result_train_sample, _sl_pred_train, 'train')):
                                if _df.shape[0] != _pred.shape[0]:
                                    raise ValueError(
                                        f"Super learner {_split} predictions have {_pred.shape[0]} "
                                        f"row(s) but result_{_split}_sample has {_df.shape[0]}")
                            _sl_write_valid = _sl_pred_valid.shape[0] != 0
                            if _sl_write_valid and result_valid_sample.shape[0] != _sl_pred_valid.shape[0]:
                                raise ValueError(
                                    f"Super learner valid predictions have {_sl_pred_valid.shape[0]} "
                                    f"row(s) but result_valid_sample has {result_valid_sample.shape[0]}")
                            _sl_actual = result_test_sample['actual'].astype(float).values
                            _sl_r_unw, _sl_mse_unw = safe_regression_metrics(_sl_actual, _sl_pred_test.values)
                            _sl_test_kw = blue_weights.scoring_for(_blue_weight_ctx, 'test')
                            if _sl_test_kw:
                                _sl_r, _sl_mse = safe_regression_metrics(
                                    _sl_actual, _sl_pred_test.values, **_sl_test_kw)
                            else:
                                _sl_r, _sl_mse = _sl_r_unw, _sl_mse_unw

                            # record row (the SAME construction the iterative branch uses)
                            _sl_rec_snap = record.copy()
                            _sl_rec = pd.concat(
                                [_sl_rec_snap, pd.DataFrame(_sl_rec_snap.iloc[-1, :]).T]
                            ).reset_index(drop=True)
                            _sl_rec.loc[_sl_rec.shape[0]-1, 'model'] = _sl_label
                            _sl_rec.loc[_sl_rec.shape[0]-1, 'Pearson correlation'] = _sl_r
                            _sl_rec.loc[_sl_rec.shape[0]-1, 'MSE'] = _sl_mse
                            if _blue_weight_ctx is not None:
                                _sl_rec.loc[_sl_rec.shape[0]-1, 'Pearson correlation (unweighted)'] = _sl_r_unw
                                _sl_rec.loc[_sl_rec.shape[0]-1, 'MSE (unweighted)'] = _sl_mse_unw
                                _sl_rec.loc[_sl_rec.shape[0]-1, 'metric_weighting'] = _blue_weight_ctx.mode
                            if _asreml_accuracy_ctx:
                                _sl_metrics = asreml_accuracy_metrics.compute_accuracy_metrics_row(
                                    list(_sl_pred_test),
                                    result_test_sample['id'].tolist() if 'id' in result_test_sample.columns else [],
                                    _sl_r, _asreml_accuracy_ctx)
                                for _col, _val in _sl_metrics.items():
                                    _sl_rec.loc[_sl_rec.shape[0]-1, _col] = _val
                            _sl_new_record = _sl_rec.iloc[_sl_rec_snap.shape[0]:]

                            # ver4-96: a per-member pooled row is a single model's prediction, not
                            # a combination: it gets a metric row and prediction columns, and no
                            # weight, effect or interaction row (it has no weights of its own)
                            if _sl_is_pooled in ('member', 'member_gated'):
                                record = pd.concat([record, _sl_new_record])
                                result_test_sample[_sl_label] = _sl_pred_test.values
                                result_train_sample[_sl_label] = _sl_pred_train.values
                                if _sl_write_valid:
                                    result_valid_sample[_sl_label] = _sl_pred_valid.values
                                _sl_pd = _sl_res.diag.get('pooled', {})
                                _sl_mem_txt = (f"{_sl_method}: Pearson r={_sl_r:.4f}, "
                                               f"MSE={_sl_mse:.4f}")
                                if _sl_is_pooled == 'member_gated':
                                    _sl_mem_txt += (
                                        f" (gated: weight on the pooled prediction "
                                        f"{_sl_pd.get('member_gated_share', {}).get(_sl_method, float('nan')):.3f}"
                                        f", alpha={float(_sl_var.alpha):.2f})")
                                try:
                                    _sl_mem_base = result_test_sample[_sl_method].astype(float).values
                                    _sl_mem_r0 = safe_regression_metrics(_sl_actual, _sl_mem_base)[0]
                                    _sl_mem_gap, _sl_mem_se = paired_r_gap(
                                        _sl_actual, _sl_pred_test.values, _sl_mem_base,
                                        seed=(int(_sl_cfg['seed'])
                                              + 7919 * max(int(sample.loc[i, 'sample']), 0)))
                                    _sl_mem_txt += (f"; plain {_sl_method!r} r={_sl_mem_r0:.4f}, "
                                                    f"\u0394r = {_sl_mem_gap:+.4f} \u00b1 "
                                                    f"{_sl_mem_se:.4f} (paired bootstrap SE, n_test "
                                                    f"= {len(_sl_actual)})")
                                except Exception as _sl_mem_err:
                                    _sl_mem_txt += (f"; the gap to the plain model could not be "
                                                    f"computed ({type(_sl_mem_err).__name__}: "
                                                    f"{_sl_mem_err})")
                                print(f"[GP] Task {task_num}/{total_tasks} | {_sl_label}: the "
                                      f"model fitted on its own lines plus "
                                      f"{_sl_pd.get('n_source')} source line(s) - {_sl_mem_txt}. "
                                      f"The train/valid columns of this row are out-of-fold "
                                      f"predictions; no extra fit was made.")
                                continue

                            # weight row: the K cross-validated member weights (sum 1)
                            _sl_w_frame = pd.DataFrame([_sl_w], columns=_sl_out)
                            _sl_new_weight = pd.DataFrame(_sl_new_record.iloc[-1, :]).T.drop(
                                ['Pearson correlation', 'MSE'] + list(asreml_accuracy_metrics.ACCURACY_METRIC_COLUMNS)
                                + list(blue_weights.METRIC_WEIGHTING_EXTRA_COLUMNS),
                                axis=1, errors='ignore')
                            _sl_new_weight['model'] = _sl_label
                            _sl_new_weight = pd.concat(
                                [_sl_new_weight.reset_index(drop=True), _sl_w_frame.reset_index(drop=True)],
                                axis=1)

                            # effect row: the w-weighted Step 9 consensus (the same arithmetic as the
                            # weighted-ensemble rings); with explain_refit (deploy='refit') a member's
                            # refit |effect| replaces its Step 9 row, over that member's own markers.
                            _sl_new_effect = pd.DataFrame()
                            _sl_cons = None
                            _sl_member_of = {_m.name: _m for _m in _sl_usable}
                            for _m in (_sl_out if effect.shape[0] != 0 else []):
                                _m_rows = effect[effect['model'] == _m]
                                if _m_rows.shape[0] == 0:
                                    continue
                                _m_row = _m_rows.tail(1).iloc[:, 5:].abs().reset_index(drop=True)
                                if _sl_res.effect and _m in _sl_res.effect:
                                    # the refit |effect| over the member's own markers; a marker outside
                                    # its view stays NaN, exactly like the member's Step 9 row
                                    _m_refit = pd.Series(_sl_res.effect[_m],
                                                         index=_sl_member_of[_m].train.columns[:-1])
                                    _m_row = _m_refit.reindex(_m_row.columns).to_frame().T.reset_index(
                                        drop=True)
                                _m_part = _joint_safe_row_normalize(_m_row) * float(_sl_w[_sl_out.index(_m)])
                                _sl_cons = _m_part if _sl_cons is None else _sl_cons + _m_part
                            if _sl_cons is not None and effect.shape[0] != 0:
                                _sl_new_effect = pd.DataFrame(effect.iloc[-1, :]).T.reset_index(drop=True)
                                _sl_new_effect['model'] = _sl_label
                                _sl_new_effect.loc[:, list(_sl_cons.columns)] = _sl_cons.values

                            # k=2 interaction ring from the Step 9 tables (single authority, I3)
                            _sl_new_inter = pd.DataFrame()
                            if interactions.shape[0] != 0:
                                _sl_new_inter = weighted_ensemble_interactions(
                                    interactions, _sl_out, {_m: float(_w) for _m, _w in zip(_sl_out, _sl_w)},
                                    sample.loc[i, 'population'], sample.loc[i, 'phenotype'],
                                    sample.loc[i, 'ratio'], sample.loc[i, 'sample'], _sl_label)

                            # ---- everything built: append it all ----
                            record = pd.concat([record, _sl_new_record])
                            weight = pd.concat([weight, _sl_new_weight])
                            if _sl_new_effect.shape[0] != 0:
                                effect = pd.concat([effect, _sl_new_effect])
                            if _sl_new_inter.shape[0] != 0:
                                interactions = pd.concat([interactions, _sl_new_inter])
                            result_test_sample[_sl_label] = _sl_pred_test.values
                            result_train_sample[_sl_label] = _sl_pred_train.values
                            if _sl_write_valid:
                                result_valid_sample[_sl_label] = _sl_pred_valid.values

                            _sl_dg = _sl_res.diag
                            _sl_w_txt = {_m: round(float(_w), 3) for _m, _w in zip(_sl_out, _sl_w)}
                            _sl_alpha_txt = (f"alpha*={_sl_var.alpha:.2f}, "
                                             if np.isfinite(_sl_var.alpha)
                                             else f"weights by {_sl_method}, ")
                            _sl_oof_txt = {_n: round(float(_v), 3) for _n, _v in _sl_dg['oof_r'].items()}
                            _sl_lt_txt = ("; the network combiner was itself fitted on those rows"
                                          if _sl_method == 'Linear transformation' else '')
                            print(f"[GP] Task {task_num}/{total_tasks} | {_sl_label}: "
                                  f"deploy={_sl_res.deploy}, folds={_sl_dg['folds']}, "
                                  f"members={_sl_out!r}, OOF r per model {_sl_oof_txt}, "
                                  f"{_sl_alpha_txt}weights {_sl_w_txt}, "
                                  f"extra fits={_sl_res.n_fits}, Pearson r={_sl_r:.4f}, "
                                  f"MSE={_sl_mse:.4f} (took {time.time() - _sl_task_t0:.1f}s)")
                            if _sl_is_pooled:
                                _sl_pd0 = _sl_res.diag.get('pooled', {})
                                # pool-methods: a non-analytic method's pooled row has its own gate
                                _sl_pd = dict(_sl_pd0)
                                _sl_pd.update(_sl_pd0.get('by_method', {}).get(_sl_method, {}))
                                _sl_pw = {_n: round(float(_v), 3)
                                          for _n, _v in _sl_pd.get('source_weight', {}).items()}
                                _sl_gate_txt = ''
                                if _sl_method != 'Analytic least-squares':
                                    _sl_gate_txt = (" (gate between the own row and the pooled "
                                                    "row, alpha=%.2f)"
                                                    % float(_sl_pd.get('alpha', float('nan'))))
                                print(f"[GP] Task {task_num}/{total_tasks} | {_sl_label}: "
                                      f"{_sl_pd0.get('n_source')} source line(s), weight on the "
                                      f"pooled columns {float(_sl_pd.get('source_share', 0.0)):.3f}"
                                      f"{_sl_gate_txt}"
                                      f", per member {_sl_pw}; the Weight.csv row sums each "
                                      f"member's own and pooled weight.")
                            print(f"[GP] Task {task_num}/{total_tasks} | {_sl_label}: the train"
                                  f"{'/valid' if _sl_write_valid else ''} prediction column(s) of this row "
                                  f"are OUT-OF-FOLD predictions over {_sl_dg['n_rows']} rows (the other "
                                  f"models' are in-sample){_sl_lt_txt}.")

                            # ver4-79 (R1): LOG ONLY. The paired gap between this row and the
                            # member with the highest unweighted test r, with a bootstrap SE, so
                            # a tie or a one-task loss is read for what it is (ice_sim24: one
                            # task cannot separate a gap below about 2 SE). No fit, weight or
                            # selection reads this (I7); a failure here leaves the row, which is
                            # already appended, untouched.
                            try:
                                _sl_cand = [(_m, safe_regression_metrics(
                                    _sl_actual, result_test_sample[_m].astype(float).values)[0])
                                    for _m in _sl_out]
                                _sl_bm = max(_sl_cand, key=lambda _t: _t[1] if np.isfinite(_t[1])
                                             else -np.inf)[0]
                                _sl_gap_seed = (int(_sl_cfg['seed'])
                                                + 7919 * max(int(sample.loc[i, 'sample']), 0))
                                _sl_gap, _sl_gap_se = paired_r_gap(
                                    _sl_actual, _sl_pred_test.values,
                                    result_test_sample[_sl_bm].astype(float).values,
                                    seed=_sl_gap_seed)
                                _sl_unw_txt = (', unweighted' if _blue_weight_ctx is not None
                                               else '')
                                if np.isfinite(_sl_gap) and np.isfinite(_sl_gap_se):
                                    _sl_gap_txt = f"{_sl_gap:+.4f} \u00b1 {_sl_gap_se:.4f}"
                                else:
                                    _sl_gap_txt = "n/a"
                                print(f"[GP] Task {task_num}/{total_tasks} | {_sl_label}: vs best "
                                      f"member on this test split ({_sl_bm}): \u0394r = "
                                      f"{_sl_gap_txt} (paired bootstrap SE{_sl_unw_txt}, n_test = "
                                      f"{len(_sl_actual)}); one task cannot separate a gap below "
                                      f"about 2 SE - compare over iterations.")
                            except Exception as _sl_gap_err:
                                print(f"[GP] Task {task_num}/{total_tasks} | NOTE: SUPER_LEARNER "
                                      f"({_sl_label}) - the gap to the best member could not be "
                                      f"computed ({type(_sl_gap_err).__name__}: {_sl_gap_err}); "
                                      f"the '{_sl_label}' row is unaffected.")
                        except Exception:
                            print(f"[GP] Task {task_num}/{total_tasks} | NOTE: SUPER_LEARNER "
                                  f"failed for '{_sl_label}' - this task keeps the result tables "
                                  f"as they were before that row (the other rows are "
                                  f"unaffected).\n{traceback.format_exc()}")
            except Exception:
                print(f"[GP] Task {task_num}/{total_tasks} | NOTE: SUPER_LEARNER failed - this "
                      f"task keeps the result tables as the steps above left them (no further "
                      f"'Super learner' row); the other rows are unaffected.\n"
                      f"{traceback.format_exc()}")
            finally:
                end_stage(_sl_stage)

        _n_prior_test, _n_prior_valid, _n_prior_train = (
            result_test.shape[0], result_valid.shape[0], result_train.shape[0])
        result_test = pd.concat([result_test, result_test_sample],axis=0)
        result_valid = pd.concat([result_valid, result_valid_sample],axis=0)
        result_train = pd.concat([result_train, result_train_sample],axis=0)
        # This task's own rows, captured BEFORE the id-sort below reorders the accumulators
        # (see `_task_prediction_deltas` above).
        _task_prediction_deltas['result_test'] = result_test.iloc[_n_prior_test:].reset_index(drop=True)
        _task_prediction_deltas['result_valid'] = result_valid.iloc[_n_prior_valid:].reset_index(drop=True)
        _task_prediction_deltas['result_train'] = result_train.iloc[_n_prior_train:].reset_index(drop=True)
        
        # optimise_scan F3: re-sorting the WHOLE accumulator after every task is O(T^2) over a
        # run (about 52 s per table at 1,000 tasks) and unstable for rows that share an 'id'.
        # ACCELERATION['DEFER_RESULT_SORT'] (opt-in, default off) skips it here and sorts once,
        # stably, before finalisation instead; nothing reads these accumulators in between.
        if not _compute_resources.get('accel_defer_result_sort', False):
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
            'interactions_hok': interactions_hok, 'iter_rounds': iter_rounds,
            'weight': weight, 'attention_total': attention_total, 'hp_record': hp_record,
            'stats': stats, 'asreml_pheno': asreml_pheno, 'asreml_varcomp': asreml_varcomp,
            'asreml_fit_timing': asreml_fit_timing, 'asreml_accuracy_ctx': asreml_accuracy_ctx,
            'asreml_test_weights': asreml_test_weights,
            'grm_train': asreml_grm_train, 'grm_valid': asreml_grm_valid, 'grm_test': asreml_grm_test,
        }
        # ver4-33, R1.a: one StageTimer per task; its single `[timing]` line
        # is printed when the task ends (also when it fails - the stages up
        # to the failure are exactly what a person debugging it wants).
        _stage_timer = StageTimer()
        set_current_stage_timer(_stage_timer)
        # ver4-56 (R1): open this task's usage record. `option` is the ASReml option's
        # scratch-folder tag for a nested multi-option run, else None.
        _tele_status = 'failed'
        _rtele.task_begin(
            i, sample.loc[i, 'population'], sample.loc[i, 'phenotype'],
            sample.loc[i, 'ratio'], sample.loc[i, 'sample'],
            option=(os.path.basename(str(RESULT_NAME))
                    if _ASREML_OPTION_SCRATCH_ROOT in str(RESULT_NAME) else None))
        try:
            try:
                _task_prediction_deltas.clear()
                _process_one_task(i)
                _tele_status = 'ok'
            finally:
                # ver4-35, R1.f: a task that raised must not leave lane
                # jobs queued behind it (a no-op when nothing is active).
                async_lane.cancel_active_task_lanes()
                set_current_stage_timer(None)
                _rtele.task_end(_tele_status, _stage_timer.snapshot())
                _timing_line = _stage_timer.line(f"{i - idx*interval + 1}/{total_tasks}")
                if _timing_line:
                    print(_timing_line)
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
            interactions_hok = _pre_task_snapshot['interactions_hok']
            iter_rounds = _pre_task_snapshot['iter_rounds']
            weight = _pre_task_snapshot['weight']
            attention_total = _pre_task_snapshot['attention_total']
            hp_record = _pre_task_snapshot['hp_record']
            stats = _pre_task_snapshot['stats']
            asreml_pheno = _pre_task_snapshot['asreml_pheno']
            asreml_varcomp = _pre_task_snapshot['asreml_varcomp']
            asreml_fit_timing = _pre_task_snapshot['asreml_fit_timing']
            asreml_accuracy_ctx = _pre_task_snapshot['asreml_accuracy_ctx']
            asreml_test_weights = _pre_task_snapshot['asreml_test_weights']
            asreml_grm_train = _pre_task_snapshot['grm_train']
            asreml_grm_valid = _pre_task_snapshot['grm_valid']
            asreml_grm_test = _pre_task_snapshot['grm_test']

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
                 'interactions_hok': interactions_hok, 'iter_rounds': iter_rounds,
                 'attention_total': attention_total, 'weight': weight, 'hp_record': hp_record,
                 'stats': stats, 'asreml_pheno': asreml_pheno, 'asreml_varcomp': asreml_varcomp,
                 'asreml_fit_timing': asreml_fit_timing, 'asreml_accuracy_ctx': asreml_accuracy_ctx,
                 'asreml_test_weights': asreml_test_weights,
                 'grm_train': asreml_grm_train, 'grm_valid': asreml_grm_valid,
                 'grm_test': asreml_grm_test},
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
                'result_train': _task_prediction_deltas.get(
                    'result_train', result_train.iloc[len(_pre_task_snapshot['result_train']):]),
                'result_valid': _task_prediction_deltas.get(
                    'result_valid', result_valid.iloc[len(_pre_task_snapshot['result_valid']):]),
                'result_test': _task_prediction_deltas.get(
                    'result_test', result_test.iloc[len(_pre_task_snapshot['result_test']):]),
                'effect': effect.iloc[len(_pre_task_snapshot['effect']):],
                'interactions': interactions.iloc[len(_pre_task_snapshot['interactions']):],
                'interactions_hok': interactions_hok.iloc[len(_pre_task_snapshot['interactions_hok']):],
                'iter_rounds': iter_rounds.iloc[len(_pre_task_snapshot['iter_rounds']):],
                'attention_total': attention_total.iloc[len(_pre_task_snapshot['attention_total']):],
                'weight': weight.iloc[len(_pre_task_snapshot['weight']):],
                'hp_record': hp_record.iloc[len(_pre_task_snapshot['hp_record']):],
                'stats': stats.iloc[len(_pre_task_snapshot['stats']):],
                'asreml_pheno': asreml_pheno.iloc[len(_pre_task_snapshot['asreml_pheno']):],
                'asreml_varcomp': asreml_varcomp.iloc[len(_pre_task_snapshot['asreml_varcomp']):],
                'asreml_fit_timing': asreml_fit_timing.iloc[len(_pre_task_snapshot['asreml_fit_timing']):],
                'asreml_accuracy_ctx': asreml_accuracy_ctx.iloc[len(_pre_task_snapshot['asreml_accuracy_ctx']):],
                'asreml_test_weights': asreml_test_weights.iloc[len(_pre_task_snapshot['asreml_test_weights']):],
                'grm_train': asreml_grm_train.iloc[len(_pre_task_snapshot['grm_train']):],
                'grm_valid': asreml_grm_valid.iloc[len(_pre_task_snapshot['grm_valid']):],
                'grm_test': asreml_grm_test.iloc[len(_pre_task_snapshot['grm_test']):],
            }
            _ckpt.append_partial_results(RESULT_NAME, idx, _is_parallel, _delta_frames, _static_write_flags,
                                          compression=RESULT_COMPRESSION)
            _ckpt.save_checkpoint(RESULT_NAME, idx, _is_parallel, _sample_fp,
                                   last_completed_i=_task_num, total_tasks=total_tasks)

    _report_progress(completed_units, total_units, label='Finalising results...')

    if _compute_resources.get('accel_defer_result_sort', False):
        # optimise_scan F3: the single, stable id-sort that replaces the per-task sorts above.
        for _acc_name in ('result_train', 'result_valid', 'result_test'):
            _acc = {'result_train': result_train, 'result_valid': result_valid,
                    'result_test': result_test}[_acc_name]
            if isinstance(_acc, pd.DataFrame) and _acc.shape[0] and 'id' in _acc.columns:
                _acc = _acc.sort_values(['id'], kind='stable').reset_index(drop=True)
                if _acc_name == 'result_train':
                    result_train = _acc
                elif _acc_name == 'result_valid':
                    result_valid = _acc
                else:
                    result_test = _acc
        print("[GP] ACCELERATION['DEFER_RESULT_SORT']: prediction tables sorted by 'id' once "
              "(stable), instead of after every task.")

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
        # ver4-25, Stage 9 (R12): snapshotted for the same reason
        # 'interactions' is - see this dict's own comment above. NOT
        # passed into ensemble() below (Q12 - naive/weighted-ensemble
        # interactions remain k=2 only, logged once when k>=3 rows exist
        # at all - see models/circos_plot.py's own R12 wiring), so it
        # never actually changes during this finalisation block; snapshot
        # only for consistency/future-proofing with the rollback below.
        'interactions_hok': interactions_hok,
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
            # ver4-25, R3: built ONCE, here, from the FULLY assembled
            # `asreml_accuracy_ctx` accumulator (every task this process
            # itself just ran, PLUS anything rehydrated from a resumed
            # checkpoint) - the fallback `_asreml_accuracy_ctx_resolver()`
            # reaches for whenever its own live, in-memory `_asreml_ctx_
            # scenario_cache` has no entry for a given scenario (see that
            # closure's own docstring for the full resolution order).
            # `None` (a no-op fallback that itself always returns `None`)
            # whenever NEITHER accuracy metrics NOR BLUE-reliability
            # weighting were ever enabled, matching `asreml_accuracy_ctx`/
            # `asreml_test_weights`'s own "stays empty" behaviour in that
            # case.
            #
            # ver4-25, R8 bugfix (found while completing Stage 6): this
            # used to be gated on `_asreml_accuracy_metrics_enabled`
            # ALONE - so a run with `ASREML['blue_weighting']` on
            # ('diagonal' OR 'full') but accuracy metrics OFF never built
            # a resolver here AT ALL, meaning the naive ensemble's own
            # Metric.csv row was silently scored UNWEIGHTED even though
            # every per-model row and every weighted-ensemble method's
            # row were correctly weighted - the two features are
            # independent opt-ins (Requirements.md's own accuracy-metrics
            # phase vs. this R8 weighting phase) and must not be coupled
            # like this. Now built whenever EITHER feature is on, and
            # also passes the 'full'-mode `asreml_test_weights`
            # accumulator (Stage 6 completion) - a strict no-op (empty
            # frame) whenever 'full' mode was never used this run.
            _asreml_accuracy_ctx_sidecar_resolver = (
                asreml_accuracy_metrics.ctx_resolver_from_frame(asreml_accuracy_ctx, asreml_test_weights)
                if (_asreml_accuracy_metrics_enabled or _asreml_blue_weighting != 'none') else None
            )
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
                # Requirements.md (BLUEs/BLUPs accuracy-metrics phase) -
                # `_asreml_accuracy_ctx_resolver` lets the naive ensemble
                # (scored across the WHOLE assembled result_test, after
                # every task has already run) rebuild each
                # (population, phenotype, ratio, sample) row's own
                # accuracy-metrics context on demand - see that closure's
                # own docstring for the caching/refit rule and its one
                # disclosed limitation (intra_task_parallel model-level
                # fan-out).
                result_train, result_valid, result_test, sample_record, sample_effect, sample_interaction_ensemble = ensemble(result_train, result_valid, result_test, effect, _group_models, interactions, asreml_accuracy_ctx_resolver=_asreml_accuracy_ctx_resolver, metric_directions=_metric_direction_by_phenotype)
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
        interactions_hok = _pre_finalise_snapshot['interactions_hok']

        print(f"[GP] ERROR: the final ensemble/aggregation step failed: {_finalise_exc!r}")
        print(f"[GP] Saving the results from all {total_tasks} already-completed task(s) so "
              f"they aren't lost...")
        _ckpt.save_partial_results(
            RESULT_NAME, idx, _is_parallel,
            {'record': record, 'result_train': result_train, 'result_valid': result_valid,
             'result_test': result_test, 'effect': effect, 'interactions': interactions,
             'interactions_hok': interactions_hok, 'iter_rounds': iter_rounds,
             'attention_total': attention_total, 'weight': weight, 'hp_record': hp_record,
             'stats': stats, 'asreml_pheno': asreml_pheno, 'asreml_varcomp': asreml_varcomp,
             'asreml_fit_timing': asreml_fit_timing, 'asreml_accuracy_ctx': asreml_accuracy_ctx,
             'asreml_test_weights': asreml_test_weights,
             'grm_train': asreml_grm_train, 'grm_valid': asreml_grm_valid,
             'grm_test': asreml_grm_test},
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
         'interactions_hok': interactions_hok, 'iter_rounds': iter_rounds,
         'attention_total': attention_total, 'weight': weight, 'hp_record': hp_record,
         'stats': stats, 'asreml_pheno': asreml_pheno, 'asreml_varcomp': asreml_varcomp,
         'asreml_fit_timing': asreml_fit_timing, 'asreml_accuracy_ctx': asreml_accuracy_ctx,
         'asreml_test_weights': asreml_test_weights,
         'grm_train': asreml_grm_train, 'grm_valid': asreml_grm_valid, 'grm_test': asreml_grm_test},
        _static_write_flags, compression=RESULT_COMPRESSION,
    )

    # Every task in this batch finished successfully - nothing left to
    # resume, so the checkpoint (if one existed, from an earlier failed
    # attempt at this same batch) is no longer needed. Left in place, it
    # could otherwise make an unrelated FUTURE run that happens to reuse
    # this RESULT_NAME think it should skip tasks it hasn't actually run.
    _ckpt.clear_checkpoint(RESULT_NAME, idx, _is_parallel)

    # ver4-25, Stage 9 (R12): `interactions_hok` is deliberately NOT added
    # to this positional return tuple - every one of its 3 external call
    # sites (run_sequential.py, main_app.py x2) unpacks this tuple
    # positionally, so appending to it is a real (if disclosed, I11-
    # respecting) breaking-surface risk for comparatively little benefit:
    # `interactions_hok` is already fully durable on disk (saved/appended/
    # loaded generically, exactly like `interactions` - see this
    # function's own save/append/load call sites above), and the design
    # blueprint's own R12 data-flow routes every consumer through
    # `batch_reader.ResultSet.interactions_hok_grouped()` (a streaming,
    # chunked read from Interaction_higher_order.csv, not a raw in-memory
    # frame) rather than through GP()'s own return value - exactly the
    # same "bounded memory on the interaction path" discipline the
    # existing `interactions_grouped()` already established for
    # Interaction.csv (see the architecture doc §16.1). A caller that
    # wants this run's own k>=3 rows reads them back via that same
    # ResultSet path, immediately after this call, rather than from here.
    return record, result_train, result_test, effect, interactions, POPULATION, PHENOTYPE, attention_total


# ============================================================================
# MULTI-OPTION WRAPPER (Requirements.md items 1-3, multi-option update).
#
# `_GP_single()` above is the (unchanged) single-strategy engine. `GP()` is
# what every caller imports: with ASReml off, or with exactly ONE genotype-
# effect option selected, it simply forwards to `_GP_single()` (byte-
# identical behaviour, I11). With MORE THAN ONE option selected (BLUEs /
# BLUPs without GRM / BLUPs with GRM - any subset), it
#
#   1. expands `ASREML` into one single-strategy config per option
#      (`asreml_syntax.expand_strategy_configs()`),
#   2. runs `_GP_single()` once per option, each into its OWN scratch
#      RESULT_NAME (`Result/<RESULT_NAME>/.asreml_options/<batch>/<tag>/`),
#      so every per-task cache, checkpoint, ASReml fit and accumulator is
#      exactly what a stand-alone single-option run would have had - which
#      is what guarantees each option's numbers equal a single-option run's,
#   3. merges the per-option result files into the standard files under the
#      caller's own RESULT_NAME, with the option column
#      (`strategy_results.OPTION_COLUMN`) as the FIRST column of every file,
#   4. returns the same 8-tuple `_GP_single()` does, with the option column
#      on the frames that have one, so in-process callers see the same data
#      the files contain.
#
# Resume: each option's `_GP_single()` call has its own checkpoint inside its
# scratch folder, so a re-submitted batch resumes the option that failed. An
# option that already FINISHED (marker file below, keyed on a fingerprint of
# every argument) is not re-run.
# ============================================================================
import functools as _functools
import inspect as _inspect
import json as _json

import strategy_results as _strategy_results

_ASREML_OPTION_SCRATCH_ROOT = '.asreml_options'
_ASREML_OPTION_MARKER = '.option_complete.json'


def _asreml_option_scratch_name(result_name, idx, is_parallel, label):
    batch = f'batch_{idx}' if is_parallel else 'sequential'
    return os.path.join(result_name, _ASREML_OPTION_SCRATCH_ROOT, batch, _strategy_results.option_tag(label))


def _asreml_option_fingerprint(arguments, option_cfg, label):
    """SHA-256 of every GP() argument (minus the callback and the
    RESULT_NAME, which differ per nested run by construction) with the
    option's own resolved config substituted for `ASREML` - a marker written
    under one fingerprint is only ever honoured by a run with the SAME one."""
    payload = {k: v for k, v in arguments.items() if k not in ('progress_callback', 'RESULT_NAME')}
    payload['ASREML'] = option_cfg
    payload['__option__'] = label
    try:
        blob = _json.dumps(payload, sort_keys=True, default=str)
    except Exception:
        blob = repr(sorted((k, repr(v)) for k, v in payload.items()))
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


@_functools.wraps(_GP_single)
def GP(*args, **kwargs):
    """Public entry point: runs :func:`_GP_multi_option` with the shared
    preprocessing-memo directory (``Result/<RESULT_NAME>/_memo``, ver4-33
    R1.c) installed in the environment for the duration of the call. An
    outer entry point's own ``EASIGP_MEMO_DIR`` always wins, and the variable
    is removed again afterwards if (and only if) this call set it, so a
    long-lived process (the Streamlit GUI) never leaks one run's memo
    location into the next."""
    # ver4-44, R0: say which engine this run executes, and refuse to run when a loaded
    # EasiGP module is older than its file on disk (a GUI that was not restarted after a
    # code update would otherwise run the old engine with the new GUI's settings).
    print(f"[GP] EasiGP engine {ENGINE_VERSION}")
    if stale_check_enabled():
        _stale = stale_engine_modules()
        if _stale:
            _msg = stale_engine_message(_stale)
            print(f"[GP] ERROR: {_msg} - the run is stopped before task 1.")
            raise RuntimeError(_msg)
    else:
        print("[GP] NOTE: EASIGP_STALE_CHECK=off - the stale-engine check is skipped for this "
              "run.")
    _result_name = _inspect.signature(_GP_single).bind(*args, **kwargs).arguments.get('RESULT_NAME')
    _memo_set_here = ensure_memo_dir_env(_result_name)
    # ver4-56 (R1): telemetry folder (outer value wins) + arm the recorder for this call.
    # One job record is written per GP() call even when ASReml options fan out into
    # several nested _GP_single() runs, because the nested runs share this process.
    _bound_arguments = _inspect.signature(_GP_single).bind(*args, **kwargs).arguments
    _telemetry_set_here = ensure_telemetry_dir_env(_result_name)
    _rtele.configure(
        enabled=bool(_bound_arguments.get('RESOURCE_TELEMETRY', True)),
        interval_s=_bound_arguments.get('RESOURCE_TELEMETRY_INTERVAL_S', 0.5),
        engine=ENGINE_VERSION, result_name=str(_result_name or ''),
        scenario=str(_bound_arguments.get('SCENARIO') or ''))
    try:
        return _GP_multi_option(*args, **kwargs)
    finally:
        _rtele.job_end()
        release_telemetry_dir_env(_telemetry_set_here)
        # ver4-35: the model lane's worker and the GPU arbiter's Manager live
        # for one GP() call (each is a no-op when it was never created here).
        async_lane.release_lanes()
        release_gpu_arbiter()
        release_memo_dir_env(_memo_set_here)


def _GP_multi_option(*args, **kwargs):
    arguments = dict(_inspect.signature(_GP_single).bind(*args, **kwargs).arguments)
    asreml_cfg = arguments.get('ASREML')
    if not asreml_cfg:
        return _GP_single(**arguments)

    # Raises ValueError for an EMPTY option list ("If no field is chosen, it
    # causes an error before running EasiGP") - before a single task runs.
    expanded = asreml_syntax.expand_strategy_configs(asreml_cfg)
    if len(expanded) <= 1:
        arguments['ASREML'] = expanded[0][1] if expanded else asreml_cfg
        return _GP_single(**arguments)

    result_name = arguments['RESULT_NAME']
    parallel_arg = arguments.get('PARALLEL')
    is_parallel = parallel_arg is not None
    idx = parallel_arg['batch_id'] if is_parallel else 0
    compression = arguments.get('RESULT_COMPRESSION', 'gzip')
    progress_callback = arguments.get('progress_callback')
    n_options = len(expanded)

    # Fail fast, before ANY option starts: every option's own structural /
    # scenario validity (a hand-written config may still carry a combination
    # the GUI would never offer, e.g. masking + BLUEs under 'within').
    scenario = arguments.get('SCENARIO')
    _errors = asreml_syntax.validate_all_strategies(asreml_cfg, scenario)
    if _errors:
        raise ValueError(
            "ASReml-R configuration is not valid for this run:\n"
            + "\n".join(f"  - {e}" for e in _errors)
        )

    print(f"[GP] Multi-option ASReml run: {n_options} option(s) will be run and reported "
          f"separately for the SAME iterations: "
          f"{[asreml_syntax.strategy_label(s) for s, _ in expanded]}. Result files under "
          f"'{result_name}' gain a leading '{_strategy_results.OPTION_COLUMN}' column.")

    os.makedirs(result_dir_path(result_name), exist_ok=True)
    per_option = []          # [(label, nested_name, 8-tuple-or-None)]
    for opt_i, (strategy, option_cfg) in enumerate(expanded):
        label = asreml_syntax.strategy_label(strategy)
        nested_name = _asreml_option_scratch_name(result_name, idx, is_parallel, label)
        nested_dir = result_dir_path(nested_name)
        marker_path = os.path.join(nested_dir, _ASREML_OPTION_MARKER)
        fingerprint = _asreml_option_fingerprint(arguments, option_cfg, label)

        reuse = False
        if os.path.isfile(marker_path):
            try:
                with open(marker_path) as _f:
                    reuse = (_json.load(_f).get('fingerprint') == fingerprint
                             and _ckpt.resolve_result_path(nested_name, idx, is_parallel, 'record') is not None)
            except (OSError, ValueError):
                reuse = False
        if reuse:
            print(f"[GP] ===== ASReml option {opt_i + 1}/{n_options}: {label} - already completed "
                  f"in an earlier attempt of this job (same configuration); reusing its results "
                  f"instead of re-running it. =====")
            per_option.append((label, nested_name, None))
            continue

        print(f"[GP] ===== ASReml option {opt_i + 1}/{n_options}: {label} "
              f"(strategy={strategy!r}) =====")
        kw = dict(arguments)
        kw['ASREML'] = option_cfg
        kw['RESULT_NAME'] = nested_name
        if progress_callback is not None:
            def _scaled_cb(completed, total, lbl=None, _i=opt_i, _label=label):
                progress_callback(_i * total + completed, n_options * total,
                                  f'[{_label}] {lbl}' if lbl else f'[{_label}]')
            kw['progress_callback'] = _scaled_cb
        out = _GP_single(**kw)
        os.makedirs(nested_dir, exist_ok=True)
        with open(marker_path, 'w') as _f:
            _json.dump({'fingerprint': fingerprint, 'option': label}, _f)
        per_option.append((label, nested_name, out))

    # -- merge every option's files into the caller's own RESULT_NAME -------
    _strategy_results.merge_option_results(
        result_name, idx, is_parallel, [(lab, nm) for lab, nm, _ in per_option],
        compression=compression,
    )

    # -- rebuild the in-memory return tuple (option column first) -----------
    def _frame(out, position, key, nested_name):
        if out is not None:
            return out[position]
        return _strategy_results.read_option_frame(nested_name, idx, is_parallel, key)

    positions = {'record': 0, 'result_train': 1, 'result_test': 2, 'effect': 3,
                 'interactions': 4, 'attention_total': 7}
    merged = {}
    for key, pos in positions.items():
        if key == 'result_train':
            # GP()'s own return tuple carries result_train at position 1 only
            # for backwards compatibility; same handling as every other frame.
            pass
        merged[key] = _strategy_results.concat_with_option(
            [(lab, _frame(out, pos, key, nm)) for lab, nm, out in per_option]
        )
    first_out = next((out for _, _, out in per_option if out is not None), None)
    if first_out is not None:
        population, phenotype = first_out[5], first_out[6]
    else:
        rec = merged['record']
        population = pd.unique(rec['population']).tolist() if 'population' in rec.columns else []
        phenotype = pd.unique(rec['phenotype']).tolist() if 'phenotype' in rec.columns else []

    # -- hoist side artefacts (LD-decay CSVs, gene-coordinate files, ...) of
    #    the FIRST option into the parent (identical across options: they
    #    depend only on the genotype data), then discard every scratch run.
    # The per-option completion markers are resume bookkeeping for the scratch
    # folders only - drop them first so they are never hoisted into the parent.
    for _, nm, _ in per_option:
        try:
            os.remove(os.path.join(result_dir_path(nm), _ASREML_OPTION_MARKER))
        except OSError:
            pass
    try:
        from intra_batch_parallel import relocate_side_artefacts as _relocate
        _relocate(per_option[0][1], result_name)
    except Exception as exc:  # never let a diagnostic-artefact problem fail a finished run
        print(f"[GP] NOTE: could not relocate side artefacts (LD-decay data etc.) out of the first "
              f"option's scratch folder: {exc!r}. Results are unaffected.")
    for _, nm, _ in per_option:
        _strategy_results.remove_nested_run(nm)
    _root = os.path.join(result_dir_path(result_name), _ASREML_OPTION_SCRATCH_ROOT,
                         f'batch_{idx}' if is_parallel else 'sequential')
    for _d in (_root, os.path.dirname(_root)):
        try:
            os.rmdir(_d)          # only if empty - never removes another batch's scratch
        except OSError:
            pass

    return (merged['record'], merged['result_train'], merged['result_test'], merged['effect'],
            merged['interactions'], population, phenotype, merged['attention_total'])
