"""
intra_task_parallel.py
========================

Update ID 2 (R1) - "intra-task model-level parallelism": within a SINGLE
prediction-scenario task (one row of ``genomic_prediction.py::GP()``'s own
``sample`` table - architecture doc §6), fan Step 9's per-model dispatch
loop (``for jj in MODEL_RUN`` - architecture doc §7) out across up to
``N_MODEL_WORKERS`` worker PROCESSES, composing with (not replacing)
``intra_batch_parallel.py``'s existing TASK-level fan-out (Phase 2 §7) so
one Parallel-mode array-job batch can use BOTH dimensions of parallelism
at once against one shared CPU/GPU budget.

Why this exists (see the ver4-3 design blueprint §R1 for the full design
record)
--------------------------------------------------------------------------
``intra_batch_parallel.py`` parallelises across TASK INDICES within one
outer batch, but each individual worker still executes Step 9 strictly
serially, one model at a time - and a batch whose own ``batch_size`` is
below ``min_tasks_for_parallel`` (the common one-task-per-array-element
HPC pattern) gets NO parallelism from it at all. This module addresses the
gap: a SECOND, orthogonal unit of work, ``(task_index, model_group)``,
addressed the exact same way Phase 2 already addresses a single task -
``PARALLEL={'batch_id': task_index, 'batch_size': 1}`` - PLUS a new,
narrow, additive ``genomic_prediction.py::GP()`` keyword,
``MODEL_DISPATCH_FILTER``, that restricts which of ``MODEL_RUN``'s
already-fully-resolved models a given call actually FITS, without ever
subsetting ``MODEL``/``MODEL_RUN`` themselves - so every Steps-1-8 pool-
construction decision (``LD_prune_effective``, ``RF_filter_effective``,
the ``OTHER_MODELS_MARKER_SOURCE`` reference bio-prior instance, ...)
resolves IDENTICALLY in every worker, regardless of which models that
worker will actually fit (architecture doc §8; I6).

No ``TaskContext`` extraction (a binding design decision - blueprint §10.0
D3): the only change inside ``genomic_prediction.py`` is that one keyword
argument plus three narrow gates (G1/G2/G3). This module never reaches
into ``GP()``'s own closure state; it only decides WHICH ``GP()`` calls to
make and how to MERGE their independent, disk-persisted output back
together afterward - the same "reuse the existing isolated-result-folder
mechanism, never re-implement it" strategy Phase 2 already established for
task-level fan-out (see the public aliases imported from
``intra_batch_parallel`` below).

Two-tier merge
--------------------------------------------------------------------------
Tier 1 (NEW, this module): merges one task's own model-GROUPS back into
that task's own Phase-2-shaped isolated result files
(``intra_batch_parallel.isolated_result_name(...)``) - see
``merge_model_groups_into_task()``/``finalise_task_from_merged()``. Not a
plain row-concat: ``checkpoint_utils.MODEL_GROUP_MERGE_SPEC`` declares
three different per-file semantics (rows / dedup_rows / columns - see that
module's own docstring), because different model-GROUPS of the SAME task
contribute different kinds of content (some files gain new ROWS, some
gain new prediction COLUMNS, ``Basic_stats.csv`` is IDENTICAL across every
group and must be deduplicated rather than multiplied).

Tier 2 (Phase 2, REUSED VERBATIM): once every eligible task's tier-1 merge
has produced a normal-looking isolated task folder,
``intra_batch_parallel.merge_isolated_results_into_batch()`` merges every
task index's own isolated folder into the outer batch's real result
files - EXACTLY as it already does for a batch that never used model-level
fan-out at all. This module never re-implements that step; it only makes
sure every task (whether it was eligible for model-level fan-out or not)
ends up with an isolated folder in the SAME shape Phase 2 already expects.

Eligibility (see ``task_is_model_fanout_eligible()``) - a task is routed
back to Phase 2's ordinary, single-unit, unfiltered path (``PARALLEL=
{'batch_id': task_index, 'batch_size': 1}``, ``MODEL_DISPATCH_FILTER=
None``) whenever: (1) ``W_OPT`` is active and this task provides a
validation split (Step 11 needs the frozen Step 8 pool, reachable only
from inside ``GP()``'s own closure - blueprint decision D1); (2) fewer
than ``MIN_MODELS_FOR_PARALLEL`` schedulable groups exist; or (3)
``resource_profiles.should_fan_out_models()`` says the redundant Steps-1-8
pool-rebuild cost outweighs the fit-time saved. A mixed batch (some tasks
eligible, some not) is fully supported - both paths land in the same
tier-1/tier-2 merge.
"""

from __future__ import annotations

import os
import random
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

import checkpoint_utils as _ckpt
from pipeline_utils import cheap_marker_and_sample_counts, result_dir_path, init_rpy2_conversion
# Import direction is strictly downward (I2): this module imports
# resource_profiles (for the shared cost table / packing / break-even
# predicate), never the reverse - resource_profiles.py stays a
# lightweight, dependency-free GUI/estimator module that never imports
# this one. `_profile_for` is a leading-underscore helper reused
# deliberately (not re-implemented) - the same "reuse, don't
# re-implement" precedent architecture doc §8 sets for
# `_map_genes_to_markers`: re-deriving the same base-name-to-cost-profile
# lookup a second time here would risk silent divergence from R2's own
# advisor.
from resource_profiles import (
    _profile_for as _cost_profile_for, pack_atoms_lpt, should_fan_out_models, estimate_pool_cost,
)
# Both of these are pure-Python modules with no R/rpy2 or GPU-framework
# import-time side effects (unlike genomic_prediction.py itself, which
# MUST be imported only after R_HOME/rpy2 are configured - see GP()'s own
# callers) - safe to import at module level here, mirroring
# checkpoint_utils.py's own existing module-level `from
# models.hyperparameter_tuning import base_of`.
from models.hyperparameter_tuning import base_of, expand_model_list, ensemble_groups
from models.ensemble import ensemble
from intra_batch_parallel import (
    isolated_result_name, task_already_complete, merge_isolated_results_into_batch,
    cleanup_isolated_folders, worker_init, relocate_side_artefacts,
    # Update ID 2, Defect D2/D3 fix (see PATCH_NOTES_D2.md) - reused
    # verbatim rather than re-implemented, the same "reuse, don't
    # re-derive" precedent this module already follows for every other
    # name imported from intra_batch_parallel above: `run_in_fresh_worker`
    # runs a retry attempt in its own fresh, disposable worker process
    # (never inline in this module's own retry loop, which would defeat
    # the same failure isolation intra_batch_parallel.py's own retry loop
    # was fixed to preserve); `MP_CONTEXT` is the explicit 'spawn'
    # multiprocessing context every ProcessPoolExecutor/Manager in BOTH
    # modules now uses, for the fork-safety reasons documented beside its
    # definition in intra_batch_parallel.py.
    run_in_fresh_worker, MP_CONTEXT,
    # Defect D4 fix (see intra_batch_parallel.py's own note beside
    # `_MP_CONTEXT`): reused verbatim, same "reuse, don't re-derive"
    # precedent as every other name imported from intra_batch_parallel
    # above.
    retry_transient_mount_error,
)


# ==========================================================================
# Model grouping - ATOMISE -> WEIGH -> BIN-PACK -> ORDER -> FINGERPRINT
# (blueprint §R1.6).
# ==========================================================================

def _is_bio_prior_base(base: str) -> bool:
    """Duplicated, one-line-for-one-line, from
    genomic_prediction.py's own ``_is_bio_prior_model()`` (and
    checkpoint_utils.py's own independent copy of the same check) - kept
    as an independent copy rather than importing genomic_prediction.py
    itself, which must never be imported at module level anywhere (it
    sources R model files at CALL time and requires R_HOME/rpy2 to
    already be configured - see this module's own worker entry points,
    which import it lazily, after that setup, exactly like
    intra_batch_parallel.py does)."""
    return base == 'GAT_biological_prior_knowledge' or base.startswith('GAT_biological_prior_knowledge_')


def _bio_prior_merge_enabled(instance_name: str, hparameters: Mapping[str, object]) -> bool:
    """Read-only check of ``HPARAMETERS[instance][13]`` (the 'enabled'
    flag of ``GAT_biological_prior_knowledge``'s data-driven prior-
    network merge feature - architecture doc §12.4) - READ ONLY, never
    written (I5), exactly matching blueprint §10.4's defensive read:
    ``len(params) > 13 and bool(params[13])``.

    ``HPARAMETERS[instance]`` may be a per-phenotype dict instead of a
    flat params list (architecture doc §4.4) - in that case this is
    merge-enabled if ANY phenotype's own params list has it enabled,
    since C2's underlying concern (one shared ``_bio_prior_merge_cache``
    per TASK, inside ``GP()``) applies regardless of which phenotype the
    current task happens to be for."""
    params = hparameters.get(instance_name) if hparameters else None
    if params is None:
        return False
    if isinstance(params, dict):
        return any(_flat_params_merge_enabled(p) for p in params.values())
    return _flat_params_merge_enabled(params)


def _flat_params_merge_enabled(params: Sequence) -> bool:
    try:
        return len(params) > 13 and bool(params[13])
    except TypeError:
        return False


def _atom_weight(members: Sequence[str]) -> float:
    """Sum of ``fit_cost_factor`` (``resource_profiles.MODEL_COST_
    PROFILE``) across an atom's/group's members - the SAME cost table
    R2's advisor uses, reused (not duplicated) as the one shared cost
    authority (blueprint §R1.6 "Weigh" step).

    ``members`` is expected to already be real, post-``expand_model_
    list()`` ``MODEL_RUN`` entries - a base model tuned with more than
    one algorithm therefore already appears as SEPARATE members (e.g.
    both ``'RF__Grid'`` and ``'RF__Bayesian'``), each contributing its
    own weight once; no further multiplication for multi-algorithm
    tuning is applied here (unlike ``resource_profiles.py``'s own,
    necessarily approximate, ``_atoms_for_costing()``, which only ever
    sees pre-expansion base names and so needs a separate estimate of
    how many algorithms a model will expand into)."""
    total = 0.0
    for member in members:
        profile = _cost_profile_for(member)
        total += float(profile.get('fit_cost_factor', profile['cpu_time_factor']))
    return total


def _contiguous_bins(n_atoms: int, n_bins: int) -> List[List[int]]:
    """Contiguous slices of ``range(n_atoms)`` into ``n_bins`` groups, as
    evenly sized as possible (earlier bins receive one extra atom when
    the count doesn't divide evenly) - the ``'contiguous'`` grouping
    mode, useful mainly for making group composition trivially auditable
    (see the ver4-3 design blueprint's rollback-plan note on
    ``MODEL_GROUPING='contiguous'``)."""
    n_bins = max(1, min(n_bins, n_atoms))
    base_size, remainder = divmod(n_atoms, n_bins)
    bins: List[List[int]] = []
    start = 0
    for b in range(n_bins):
        size = base_size + (1 if b < remainder else 0)
        bins.append(list(range(start, start + size)))
        start += size
    return bins


def build_model_groups(
    model_run: Sequence[str],
    hparameters: Optional[Mapping[str, object]],
    hp_tune: Optional[Mapping[str, Mapping]],
    n_model_workers: int,
    model_grouping: str = 'cost_balanced',
    bio_prior_grouping: str = 'affinity',
) -> List[List[str]]:
    """Build the authoritative model-group list for one task's dispatch
    fan-out (blueprint §R1.6): ATOMISE (apply the two hard, correctness-
    critical grouping constraints C1/C2), WEIGH each atom, BIN-PACK atoms
    into at most ``n_model_workers`` groups, then ORDER groups by their
    lowest ``MODEL_RUN`` index (so tier-1 merge output row/column order
    matches a serial run - see ``checkpoint_utils.merge_model_group_
    frames()``'s own ordering helpers).

    Parameters
    ----------
    model_run : the task's real, post-``expand_model_list()`` ``MODEL_
        RUN`` (e.g. includes ``'RF__Grid'``/``'RF__Bayesian'`` as
        SEPARATE entries if ``RF`` was tuned with both algorithms) -
        ``'ensemble'`` is excluded automatically (it is a finalisation
        step over already-fitted models, never itself dispatched).
    hparameters : the run's ``HPARAMETERS`` dict - consulted ONLY to
        check ``[instance][13]`` (the bio-prior merge-enabled flag) for
        C2; never read for anything else, never written (I5).
    hp_tune : accepted for interface symmetry with ``resource_profiles.
        estimate_resources()``'s analogous parameter, and reserved for a
        future extension - NOT currently consulted here, because
        ``model_run`` is expected to already be the real, expanded
        ``MODEL_RUN`` (multi-algorithm tuning already appears as
        separate members - see ``_atom_weight()``'s own docstring);
        applying a further per-algorithm multiplier here would double-
        count.
    n_model_workers : the maximum number of groups to bin-pack atoms
        into (``'per_model'`` grouping ignores this and returns one
        group per atom instead).
    model_grouping : ``'cost_balanced'`` (default, LPT greedy bin-
        packing - see ``resource_profiles.pack_atoms_lpt()``),
        ``'contiguous'`` (evenly-sized contiguous slices, for auditable
        group composition), or ``'per_model'`` (one group per atom,
        maximum parallelism at maximum Steps-1-8 rebuild cost).
    bio_prior_grouping : ``'affinity'`` (default - all merge-enabled
        ``GAT_biological_prior_knowledge*`` instances are forced into
        ONE atom, so ``_bio_prior_merge_cache`` is built once per task
        exactly as today) or ``'split'`` (each bio-prior instance is its
        own atom, freely splittable like any other model).

    Returns
    -------
    A list of groups, each a list of ``MODEL_RUN`` entry names (in
    ``MODEL_RUN`` dispatch order within the group). Empty if
    ``model_run`` contains nothing but (optionally) ``'ensemble'``.

    Raises
    ------
    ValueError if ``model_grouping`` isn't one of the three recognised
    modes.
    """
    hparameters = hparameters or {}
    dispatched = [name for name in model_run if name != 'ensemble']
    if not dispatched:
        return []

    # ATOMISE - C1 (base-affinity): every MODEL_RUN entry sharing the
    # SAME base_of() (which PRESERVES a bio-prior instance suffix, unlike
    # resource_profiles.py's own, deliberately coarser, cost-table-lookup
    # `_base_of()`) forms one atom, in MODEL_RUN order - required because
    # HP_TUNE mutates HPARAMETERS[base] IN PLACE inside GP()'s task loop
    # (architecture doc §18): splitting 'RF__Grid'/'RF__Bayesian' across
    # workers would silently stop 'RF__Bayesian' warm-starting from
    # 'RF__Grid''s tuned values.
    raw_keys = {name: base_of(name) for name in dispatched}

    # ATOMISE - C2 (bio-prior affinity, default 'affinity'): every
    # MERGE-ENABLED GAT_biological_prior_knowledge* instance is folded
    # into the FIRST such instance's own atom (by first appearance in
    # MODEL_RUN), so _bio_prior_merge_cache is still built exactly once
    # per task, shared by every instance and every tuning trial, as
    # today - splitting them ('split' mode) rebuilds that (LD-prune +
    # RF-filter + pairwise-SHAP-cost) cache once per extra instance.
    key_redirect: Dict[str, str] = {}
    if bio_prior_grouping == 'affinity':
        merge_enabled_keys_in_order: List[str] = []
        seen_keys = set()
        for name in dispatched:
            key = raw_keys[name]
            if key in seen_keys:
                continue
            seen_keys.add(key)
            if _is_bio_prior_base(key) and _bio_prior_merge_enabled(key, hparameters):
                merge_enabled_keys_in_order.append(key)
        if len(merge_enabled_keys_in_order) > 1:
            target_key = merge_enabled_keys_in_order[0]
            for other_key in merge_enabled_keys_in_order[1:]:
                key_redirect[other_key] = target_key
    elif bio_prior_grouping != 'split':
        raise ValueError(
            f"build_model_groups: bio_prior_grouping must be 'affinity' or 'split', got "
            f"{bio_prior_grouping!r}."
        )

    order: List[str] = []
    members_by_key: Dict[str, List[str]] = {}
    for name in dispatched:
        key = key_redirect.get(raw_keys[name], raw_keys[name])
        if key not in members_by_key:
            members_by_key[key] = []
            order.append(key)
        members_by_key[key].append(name)
    atoms = [members_by_key[key] for key in order]
    n_atoms = len(atoms)

    # WEIGH
    atom_costs = [_atom_weight(members) for members in atoms]

    # BIN-PACK
    if model_grouping == 'per_model':
        bins_by_atom = [[i] for i in range(n_atoms)]
    elif model_grouping == 'contiguous':
        bins_by_atom = _contiguous_bins(n_atoms, max(1, min(n_model_workers, n_atoms)))
    elif model_grouping == 'cost_balanced':
        bins_by_atom = pack_atoms_lpt(atom_costs, max(1, min(n_model_workers, n_atoms)))
    else:
        raise ValueError(
            f"build_model_groups: model_grouping must be 'cost_balanced', 'contiguous', or "
            f"'per_model', got {model_grouping!r}."
        )

    # ORDER: groups ordered by their lowest MODEL_RUN (dispatched) index,
    # so a resulting group's own internal member order, AND the order
    # groups are later merged in, matches a serial run closely enough for
    # checkpoint_utils's own ordering helpers to restore exact row/column
    # order afterward.
    dispatch_index = {name: i for i, name in enumerate(dispatched)}
    groups = [[m for i in b for m in atoms[i]] for b in bins_by_atom if b]
    groups.sort(key=lambda g: min(dispatch_index[m] for m in g))
    return groups


# ==========================================================================
# Eligibility gate (blueprint §10.6) - evaluated once per task, before any
# unit for that task is submitted.
# ==========================================================================

def _task_has_validation_split(gp_kwargs: dict, task_index: int) -> bool:
    """Cheap, deterministic replica of the ONE condition genomic_
    prediction.py::GP()'s own gate G2 tests
    (``type(sample.loc[i,'ratio']) is tuple or SCENARIO == 'between'``),
    computed WITHOUT reconstructing the full ``sample`` table (which
    would need the genotype/phenotype files' actual population list) -
    see the module's derivation:

    For ``SCENARIO == 'within'``, GP()'s own ``sample`` construction
    tiles a base pattern of length ``SAMPLE_NUM * len(RATIO)`` (RATIO
    entries, each repeated SAMPLE_NUM times) once per POPULATION x
    PHENOTYPE combination (architecture doc §6) - so ``sample.loc[i,
    'ratio']`` depends only on ``i``'s position WITHIN that base pattern,
    ``RATIO[(i % (SAMPLE_NUM * len(RATIO))) // SAMPLE_NUM]``, regardless
    of how many populations/phenotypes exist or what they're called
    (confirmed against V1: the ``sample`` table never depends on
    ``MODEL``, and its row-order construction is a pure function of
    ``SCENARIO``/``RATIO``/``SAMPLE_NUM`` alone for this purpose).

    For ``SCENARIO == 'between'``, gate G2's own condition already
    counts ANY task as having a validation split whenever ``W_OPT`` is
    active, regardless of the (always-sentinel -1) ratio value - so this
    returns ``True`` unconditionally in that case, matching G2 exactly.
    """
    scenario = gp_kwargs.get('SCENARIO')
    if scenario == 'between':
        return True
    if scenario != 'within':
        return False
    sample_num = int(gp_kwargs.get('SAMPLE_NUM') or 0)
    ratio_list = gp_kwargs.get('RATIO') or []
    if sample_num <= 0 or not ratio_list:
        return False
    cycle_len = sample_num * len(ratio_list)
    ratio_index = (task_index % cycle_len) // sample_num
    this_ratio = ratio_list[ratio_index]
    return isinstance(this_ratio, tuple)


def task_is_model_fanout_eligible(
    gp_kwargs: dict, groups: Sequence[Sequence[str]],
) -> Tuple[bool, str]:
    """Decide whether ONE task (identified by ``gp_kwargs['PARALLEL']
    ['batch_id']`` - the caller is expected to pass a per-task copy of
    ``gp_kwargs`` with ``PARALLEL`` already set to ``{'batch_id':
    task_index, 'batch_size': 1}``, mirroring every other per-unit
    ``GP()`` call this module makes) should be fanned out across
    ``groups``, or fall back to Phase 2's ordinary, single-unit, task-
    level-only path.

    Evaluated against the four conditions in blueprint §10.6, in order -
    the first one that fails makes the task ineligible:

    1. ``W_OPT`` is active AND this task provides a validation split -
       Step 11's weighted-ensemble finalisation needs the frozen Step 8
       pool, reachable only from inside ``GP()``'s own closure (decision
       D1 - never attempted under fan-out in this delivery).
    2. Fewer than ``gp_kwargs.get('MIN_MODELS_FOR_PARALLEL', 2)``
       schedulable groups exist after C1/C2 atomisation.
    3. ``resource_profiles.should_fan_out_models()`` (the SAME predicate
       R2's advisor uses - A2.6) says the redundant Steps-1-8 pool-
       rebuild cost outweighs the fit-time saved.
    4. Naive ensemble configured AND V4 resolved negative - V4 resolved
       POSITIVE for this delivery (``models.ensemble.ensemble()`` can be
       driven purely from tier-1-merged frames - see ``finalise_task_
       from_merged()``'s own docstring), so this condition never makes a
       task ineligible here; listed for completeness against the
       blueprint's own five-condition gate.

    Returns
    -------
    ``(eligible, reason)`` - ``reason`` is always a complete sentence,
    logged unconditionally by the caller regardless of the verdict.
    """
    task_index = gp_kwargs['PARALLEL']['batch_id']

    if gp_kwargs.get('W_OPT') is not None and _task_has_validation_split(gp_kwargs, task_index):
        return False, (
            "W_OPT is active and this task provides a validation split - Step 11's weighted-"
            "ensemble finalisation requires the frozen Step 8 pool, which exists only inside "
            "GP()'s own _process_one_task closure (blueprint decision D1)"
        )

    min_models_for_parallel = int(gp_kwargs.get('MIN_MODELS_FOR_PARALLEL', 2) or 2)
    n_groups = len(groups)
    if n_groups < min_models_for_parallel:
        return False, (
            f"only {n_groups} schedulable model group(s) after grouping, fewer than "
            f"MIN_MODELS_FOR_PARALLEL={min_models_for_parallel}"
        )

    n_markers, n_samples = cheap_marker_and_sample_counts(
        gp_kwargs.get('GENOTYPE_FORMAT', 'csv'), gp_kwargs['GENOTYPE_FILE_NAME'],
        gp_kwargs.get('PHENOTYPE_FILE_NAME'),
    )
    pool_cost = estimate_pool_cost(
        n_markers, n_samples, gp_kwargs.get('LD_prune') is not None, gp_kwargs.get('RF_filter') is not None,
        gp_kwargs.get('GENOTYPE_FORMAT', 'csv'),
    )
    group_fit_costs = [_atom_weight(members) for members in groups]
    recommended, reason = should_fan_out_models(pool_cost, group_fit_costs)
    if not recommended:
        return False, reason

    return True, f"{n_groups} model group(s), estimated pool_cost={pool_cost:.1f} RF-relative unit(s) - {reason}"


# ==========================================================================
# Scratch layout, tier-1 merge, finalisation.
# ==========================================================================

def _group_result_name(task_result_name: str, group_index: int, fingerprint: str) -> str:
    """RESULT_NAME (relative to ``Result/``, like every other RESULT_NAME
    in this codebase) for one model-group's own isolated ``GP()`` call -
    nested one level below the task's own isolated folder (blueprint
    §10.5's scratch-layout diagram):
    ``Result/<task_result_name>/.intra_task_parallel/group_<group_index>_<fingerprint>/``."""
    return os.path.join(task_result_name, '.intra_task_parallel', f'group_{group_index}_{fingerprint}')


def _read_csv_if_exists(path: str) -> Optional["pd.DataFrame"]:
    """Read a CSV, or ``None`` if missing/empty - matches intra_batch_
    parallel.py's own ``_read_csv_if_exists()`` convention exactly (kept
    as a small, independent copy here rather than importing that
    private helper, since it is not among row 2's five exposed public
    aliases)."""
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    if df.shape[0] == 0:
        return None
    return df


def _group_already_complete(group_result_name: str, task_index: int) -> bool:
    """True if this model-group's own isolated sub-run already finished
    successfully in a PREVIOUS attempt at this same task - mirrors
    intra_batch_parallel.py's own ``_task_already_complete()`` exactly
    (same 'Metric CSV present, no checkpoint file' definition of
    'finished'), re-implemented here (not imported - see the module-level
    note above on which five helpers are exposed) because a group's
    RESULT_NAME nests one level DEEPER than a task's own, via ``_group_
    result_name()`` above rather than ``intra_batch_parallel.
    isolated_result_name()``."""
    paths = _ckpt.result_file_paths(group_result_name, task_index, True)
    checkpoint_path = os.path.join(result_dir_path(group_result_name), f'.checkpoint_{task_index}.json')
    return os.path.isfile(paths['record']) and not os.path.isfile(checkpoint_path)


def _task_merge_already_complete(result_name: str, batch_id: int, task_index: int) -> bool:
    """True if this task's TIER-1 merge (and finalisation) already
    completed in a previous attempt - i.e. exactly Phase 2's own
    ``task_already_complete()`` check, REUSED VERBATIM (blueprint
    "reuse row 2's aliases; do not re-implement the layout"), since a
    merged task's own isolated result files land in EXACTLY the same
    place and shape a Phase-2 (non-fanned-out) isolated task run would
    have."""
    return task_already_complete(result_name, batch_id, task_index)


def _cleanup_task_groups(task_result_name: str) -> None:
    """Remove one task's own ``.intra_task_parallel/`` scratch tree
    (every model-group's isolated folder), once that task's tier-1 merge
    has succeeded - pure scratch space, never referenced again
    afterward. Mirrors intra_batch_parallel.py's own ``_cleanup_
    isolated_folders()`` exactly, one level deeper."""
    root = os.path.join(result_dir_path(task_result_name), '.intra_task_parallel')
    try:
        if os.path.isdir(root):
            shutil.rmtree(root)
    except OSError as exc:
        print(f"[intra_task_parallel] WARNING: could not remove scratch folder '{root}': {exc!r}. "
              f"This does not affect correctness (it will simply be skipped/overwritten on any "
              f"future re-run of this task) - safe to delete manually.")


def merge_model_groups_into_task(
    result_name: str, batch_id: int, task_index: int,
    groups: Sequence[Sequence[str]], model_run: Sequence[str],
) -> None:
    """Tier-1 merge (blueprint §R1.7): combine every model-group's own
    isolated result files for ONE task into that task's own Phase-2-
    layout isolated result files (``intra_batch_parallel.isolated_
    result_name(result_name, batch_id, task_index)``), so intra_batch_
    parallel.py's EXISTING tier-2 merge
    (``merge_isolated_results_into_batch()``) can read it completely
    unmodified downstream - it never needs to know model-level fan-out
    happened at all.

    Per-``RESULT_FILE_NAMES``-key merge semantics come from
    ``checkpoint_utils.MODEL_GROUP_MERGE_SPEC`` (rows / dedup_rows /
    columns - see that module's own docstring); this function is a thin
    I/O wrapper around the pure ``checkpoint_utils.merge_model_group_
    frames()``. Also relocates every group's side artefacts (F1 fix)
    into the REAL, top-level ``result_name`` before the group-level
    scratch folders are removed.

    Only ever called once every group's own unit has already been
    confirmed complete (see ``_group_already_complete``/the caller's own
    completion bookkeeping) - raises if an expected group result file is
    genuinely missing rather than silently producing an incomplete merge
    (blueprint R1.12: "Tier-1 merge finds a group's file missing -
    Fatal").
    """
    task_result_name = isolated_result_name(result_name, batch_id, task_index)
    out_paths = _ckpt.result_file_paths(task_result_name, task_index, True)
    group_result_names = []
    for g, members in enumerate(groups):
        fp8 = _ckpt.model_group_fingerprint(members)
        group_result_names.append(_group_result_name(task_result_name, g, fp8))

    for g, group_result_name in enumerate(group_result_names):
        metric_path = _ckpt.result_file_paths(group_result_name, task_index, True)['record']
        if not os.path.isfile(metric_path):
            raise RuntimeError(
                f"[intra_task_parallel] Tier-1 merge for task {task_index}: group {g}'s expected "
                f"result file '{metric_path}' is missing, even though this group was reported as "
                f"complete. This indicates a GP() write failure, not a scheduling one - re-submit "
                f"this batch to retry (the batch checkpoint will not have advanced past this task)."
            )

    for key in _ckpt.RESULT_FILE_NAMES:
        frames = []
        for group_result_name in group_result_names:
            in_path = _ckpt.result_file_paths(group_result_name, task_index, True)[key]
            df = _read_csv_if_exists(in_path)
            if df is not None:
                frames.append(df)
        merged = _ckpt.merge_model_group_frames(key, frames, model_run)
        if merged.shape[0] == 0:
            continue
        out_path = out_paths[key]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        merged.to_csv(out_path, index=False)

    # F1 fix - see intra_batch_parallel.relocate_side_artefacts()'s own
    # docstring: move every group's non-result-file artefacts (LD-decay
    # data, BGLR traces, PLINK/gene-coordinate CSVs, ...) into the real,
    # top-level RESULT_NAME before the group scratch folders below are
    # removed. R1 multiplies the affected paths by the group count on top
    # of Phase 2's task count, so this cannot be deferred to tier 2 alone.
    for group_result_name in group_result_names:
        relocate_side_artefacts(group_result_name, result_name)

    _cleanup_task_groups(task_result_name)


def finalise_task_from_merged(task_result_name: str, task_index: int, gp_kwargs: dict) -> None:
    """Recompute the naive (arithmetic-mean) ensemble for ONE task from
    its own TIER-1-MERGED result files - the model-level-fan-out
    equivalent of ``genomic_prediction.py::GP()``'s own finalisation
    block, run at exactly the SAME per-task granularity Phase 2's task-
    level fan-out already produces (each of ITS isolated sub-runs
    finalises over its own single task too), so this inherits Phase 2's
    already-accepted equivalence rather than introducing a new one.

    Verification finding V4 (recorded in the Update ID 2 Change Summary
    §1): ``models.ensemble.ensemble()`` takes no closure state - only
    ``(train, valid, test, effect, MODEL, interactions)`` - so it CAN be
    driven purely from these merged frames. The naive ensemble is a
    row-wise mean plus an L1-row-normalised effect mean, both
    order-invariant, so which order the tier-1 merge assembled the
    underlying model rows/columns in cannot perturb the result.
    Requirement_patch3.md item 1's own naive-ensemble interaction
    combination (``models.ensemble.ensemble()``'s now-optional
    ``interactions`` argument, see that function's own docstring) is
    grouped by each task's own identity columns internally, so it is
    equally order-invariant across however the tier-1 merge assembled
    `interactions` below.

    Weighted ensembles (``W_OPT``) are NEVER reached here - any task with
    ``W_OPT`` active and a validation split was already made ineligible
    for fan-out by ``task_is_model_fanout_eligible()`` (decision D1)
    before any unit for it was ever submitted; this function mirrors only
    ``GP()``'s naive-ensemble block.

    A no-op if ``'ensemble'`` was not selected in ``gp_kwargs['MODEL']``
    (``MODEL_BASE``) - matches ``GP()``'s own ``if 'ensemble' in
    MODEL_BASE:`` condition exactly - or if this task produced no rows at
    all (e.g. every group skipped it for the same ``MIN_DATA_POINTS``
    reason, since Steps 1-8 are byte-identical across groups).
    """
    model_base = gp_kwargs['MODEL']
    if 'ensemble' not in model_base:
        return

    paths = _ckpt.result_file_paths(task_result_name, task_index, True)

    def _load(key: str) -> "pd.DataFrame":
        df = _read_csv_if_exists(paths[key])
        return pd.DataFrame() if df is None else df

    result_train = _load('result_train')
    result_valid = _load('result_valid')
    result_test = _load('result_test')
    effect = _load('effect')
    record = _load('record')
    # Requirement_patch3.md item 1: whatever model-group-level interaction
    # rows the tier-1 merge already combined into this task's own isolated
    # Interaction.csv (merge_model_group_frames()'s own 'rows' semantics
    # for the 'interactions' key - see checkpoint_utils.MODEL_GROUP_MERGE_
    # SPEC) - possibly empty, exactly like `effect` above, if no group in
    # this task selected an interaction-capable model.
    interactions = _load('interactions')

    if result_test.shape[0] == 0:
        return

    hp_tune = gp_kwargs.get('HP_TUNE')
    hp_tune_ensemble_mode = gp_kwargs.get('HP_TUNE_ENSEMBLE_MODE', 'per_method')

    for group_label, group_models in ensemble_groups(model_base, hp_tune, hp_tune_ensemble_mode):
        result_train, result_valid, result_test, sample_record, sample_effect, sample_interaction_ensemble = ensemble(
            result_train, result_valid, result_test, effect, group_models, interactions,
        )
        if group_label is not None:
            new_label = f'ensemble__{group_label}'
            for df in (result_train, result_valid, result_test):
                if isinstance(df, pd.DataFrame) and 'ensemble' in df.columns:
                    df.rename(columns={'ensemble': new_label}, inplace=True)
            sample_record['model'] = new_label
            if sample_effect.shape[0] != 0:
                sample_effect['model'] = new_label
            if sample_interaction_ensemble.shape[0] != 0:
                sample_interaction_ensemble = sample_interaction_ensemble.copy()
                sample_interaction_ensemble['model'] = new_label
        record = pd.concat([record, sample_record], ignore_index=True)
        effect = pd.concat([effect, sample_effect], ignore_index=True)
        interactions = pd.concat([interactions, sample_interaction_ensemble], ignore_index=True)

    result_train.to_csv(paths['result_train'], index=False)
    if result_valid.shape[0] != 0:
        result_valid.to_csv(paths['result_valid'], index=False)
    result_test.to_csv(paths['result_test'], index=False)
    record.to_csv(paths['record'], index=False)
    effect.to_csv(paths['effect'], index=False)
    if interactions.shape[0] != 0:
        interactions.to_csv(paths['interactions'], index=False)


# ==========================================================================
# Worker entry point and the flat (task, group) unit space.
# ==========================================================================

def _run_unit(unit_gp_kwargs: dict, task_index: int, group_index: Optional[int]) -> Tuple[int, Optional[int], bool, Optional[str]]:
    """Worker-process entry point for ONE ``(task_index, group_index)``
    unit - or, when ``group_index`` is ``None``, one entire task run as a
    single, unfiltered unit (the eligibility gate's ineligible-task
    fallback - ``MODEL_DISPATCH_FILTER=None``). Mirrors intra_batch_
    parallel.py's own ``_run_isolated_task()``: never raises back to the
    pool - failures are reported in the return tuple so one worker's
    crash can't take down the whole ``ProcessPoolExecutor``.

    Returns ``(task_index, group_index, success, error_message_or_None)``.
    """
    try:
        # Update ID 2, Defect D3 fix (see PATCH_NOTES_D2.md): moved inside
        # the try/except, mirroring intra_batch_parallel.py's own
        # _run_isolated_task() fix - an import-time failure is now
        # handled identically to a failure inside GP() itself, rather
        # than propagating out of this function uncaught.
        # Defect D4 fix: same bounded retry, around the same class of
        # transient network-mount error, as intra_batch_parallel.py's own
        # `_run_isolated_task()` - see that module's note beside
        # `_MP_CONTEXT` for the production failure this addresses.
        def _import_gp():
            from genomic_prediction import GP
            return GP
        GP = retry_transient_mount_error(
            _import_gp, what=f"import genomic_prediction (task {task_index}, group {group_index})",
        )
        GP(**unit_gp_kwargs)
        return task_index, group_index, True, None
    except Exception as exc:  # noqa: BLE001 - deliberately broad: report, never crash the pool
        return task_index, group_index, False, f"{exc!r}\n{traceback.format_exc()}"


def run_batch_with_model_level_parallelism(
    gp_kwargs: dict,
    n_task_workers: int = 1,
    n_model_workers: int = 1,
    n_gpu_slots: int = 0,
    model_grouping: str = 'cost_balanced',
    bio_prior_grouping: str = 'affinity',
    min_models_for_parallel: int = 2,
    max_unit_retries: int = 2,
) -> None:
    """Process one Parallel-mode array-job batch's tasks using a flat
    ``(task_index, model_group)`` unit space, on ONE ``ProcessPoolExecutor``
    sized ``n_task_workers x n_model_workers`` (blueprint §R1.2 - "why the
    flat unit space, not a nested pool").

    Parameters
    ----------
    gp_kwargs : every keyword argument ``genomic_prediction.GP()`` accepts
        for this batch - exactly what ``run_step1_batch.py`` would
        otherwise pass to a single, direct ``GP(**gp_kwargs)`` call
        (including ``gp_kwargs['PARALLEL'] = {'batch_id': <outer batch
        id>, 'batch_size': <outer batch size>}``). NEVER mutated - every
        per-unit call gets its own shallow copy with ``RESULT_NAME``/
        ``PARALLEL``/``MODEL_DISPATCH_FILTER`` overridden.
    n_task_workers : the task-level width (Phase 2's own dimension -
        analogous to ``intra_batch_parallel.py``'s ``n_cpu_workers``).
    n_model_workers : the model-level width (this module's own new
        dimension). ``<= 1`` (or no ``PARALLEL`` context at all) is a
        defensive no-op fallback to a single, ordinary ``GP(**gp_kwargs)``
        call - ``run_step1_batch.py``'s own three-way route should never
        actually reach this function with ``n_model_workers <= 1``, but a
        direct caller might.
    n_gpu_slots : maximum number of GPU-dispatched model calls allowed to
        run CONCURRENTLY across every worker sharing one physical GPU -
        0 disables the limiter. See ``pipeline_utils.gpu_slot()``.
    model_grouping, bio_prior_grouping : see ``build_model_groups()``.
    min_models_for_parallel : see ``task_is_model_fanout_eligible()``.
    max_unit_retries : how many additional serial retry attempts a
        failed unit gets (in the PARENT process) before this function
        gives up and re-raises - mirrors intra_batch_parallel.py's own
        ``max_task_retries``.

    Raises
    ------
    RuntimeError if, after all retries, at least one unit still failed -
    exactly like intra_batch_parallel.py's own analogous function. A
    resubmission of the SAME batch skips every already-succeeded unit
    automatically (see the completion checks below).
    """
    parallel_cfg = gp_kwargs.get('PARALLEL')
    if n_model_workers <= 1 or parallel_cfg is None:
        print(f"[intra_task_parallel] N_MODEL_WORKERS={n_model_workers} (or no PARALLEL context) - "
              f"model-level fan-out not applicable here; falling back to a single, ordinary "
              f"GP(**gp_kwargs) call. (run_step1_batch.py's own three-way route should not "
              f"normally reach this function in that case - this is a defensive fallback for a "
              f"direct caller.)")
        # Req 2 fix (2026-09) - see intra_batch_parallel.run_batch_with_
        # intra_batch_parallelism()'s own equivalent fallback branch for
        # the full root-cause account. This branch also calls GP()
        # directly in the caller's own process with no forking below it,
        # so it needs the same init_rpy2_conversion() call for the same
        # reason, before any R-backed model (rrBLUP/GBLUP/BayesB/RKHS)
        # can run inside it.
        init_rpy2_conversion()
        from genomic_prediction import GP
        GP(**gp_kwargs)
        return

    result_name = gp_kwargs['RESULT_NAME']
    batch_id = int(parallel_cfg['batch_id'])
    interval = int(parallel_cfg['batch_size'])
    task_indices = list(range(batch_id * interval, batch_id * interval + interval))

    model_run = expand_model_list(gp_kwargs['MODEL'], gp_kwargs.get('HP_TUNE'))
    hparameters = gp_kwargs.get('HPARAMETERS') or {}

    print(f"[intra_task_parallel] Batch {batch_id}: evaluating model-level fan-out for "
          f"{len(task_indices)} task index(es); up to {n_task_workers} task-worker(s) x "
          f"{n_model_workers} model-worker(s) (n_gpu_slots={n_gpu_slots}, "
          f"model_grouping={model_grouping!r}, bio_prior_grouping={bio_prior_grouping!r}).")

    # Model grouping is a deterministic, config-only function - identical
    # for every task in this batch (it never depends on task_index) - so
    # it is computed exactly ONCE and reused for every task's own
    # eligibility check below, rather than re-derived per task.
    groups = build_model_groups(
        model_run, hparameters, gp_kwargs.get('HP_TUNE'), n_model_workers,
        model_grouping=model_grouping, bio_prior_grouping=bio_prior_grouping,
    )
    print(f"[intra_task_parallel] Batch {batch_id}: model grouping ({len(groups)} group(s)): "
          f"{groups}")

    # Per-task eligibility (blueprint §10.6) - the ONLY thing that can
    # differ from one task to the next is whether THIS task's own ratio
    # provides a validation split under an active W_OPT (condition 1).
    task_plan: Dict[int, Optional[List[List[str]]]] = {}
    for task_index in task_indices:
        # This probe dict is used SOLELY to evaluate eligibility for this
        # task - it is never passed to a real GP() call (see
        # _kwargs_for() below, which always starts from the ORIGINAL,
        # untouched gp_kwargs instead).
        eligibility_probe_kwargs = dict(gp_kwargs)
        eligibility_probe_kwargs['PARALLEL'] = {'batch_id': task_index, 'batch_size': 1}
        eligibility_probe_kwargs['MIN_MODELS_FOR_PARALLEL'] = min_models_for_parallel
        eligible, reason = task_is_model_fanout_eligible(eligibility_probe_kwargs, groups)
        print(f"[intra_task_parallel] Batch {batch_id}, task {task_index}: "
              f"{'ELIGIBLE for' if eligible else 'NOT eligible for'} model-level fan-out - {reason}.")
        task_plan[task_index] = groups if eligible else None

    # Flat (task_index, group_index) unit space - group_index is None for
    # an ineligible task's single, unfiltered unit.
    to_run: List[Tuple[int, Optional[int]]] = []
    already_done: List[Tuple[int, Optional[int]]] = []
    total_units = 0
    for task_index in task_indices:
        task_groups = task_plan[task_index]
        if task_groups is None:
            unit = (task_index, None)
            total_units += 1
            (already_done if task_already_complete(result_name, batch_id, task_index) else to_run).append(unit)
            continue
        if _task_merge_already_complete(result_name, batch_id, task_index):
            # The tier-1 merge already succeeded for this task in a
            # previous attempt - every group unit is therefore also done.
            for g in range(len(task_groups)):
                total_units += 1
                already_done.append((task_index, g))
            continue
        task_result_name = isolated_result_name(result_name, batch_id, task_index)
        for g, members in enumerate(task_groups):
            unit = (task_index, g)
            total_units += 1
            fp8 = _ckpt.model_group_fingerprint(members)
            group_result_name = _group_result_name(task_result_name, g, fp8)
            (already_done if _group_already_complete(group_result_name, task_index) else to_run).append(unit)

    if already_done:
        print(f"[intra_task_parallel] Batch {batch_id}: {len(already_done)}/{total_units} unit(s) "
              f"already completed in a previous attempt - skipping: {already_done}")

    # Update ID 2, Defect D2 fix (see PATCH_NOTES_D2.md): spawned via the
    # shared MP_CONTEXT (imported from intra_batch_parallel.py), not the
    # platform default (fork) - same fork-safety reasoning as that
    # module's own Manager() call.
    gpu_semaphore = MP_CONTEXT.Manager().Semaphore(n_gpu_slots) if n_gpu_slots > 0 else None
    r_path = gp_kwargs.get('R_PATH')
    r_blas_threads = gp_kwargs.get('R_BLAS_THREADS')

    def _kwargs_for(task_index: int, group_index: Optional[int]) -> dict:
        unit_kwargs = dict(gp_kwargs)
        unit_kwargs.pop('progress_callback', None)
        unit_kwargs['PARALLEL'] = {'batch_id': task_index, 'batch_size': 1}
        # Update ID ver4-9, R7 (correctness fix - see
        # intra_batch_parallel.py::_sub_kwargs_for()'s own identical
        # comment for the full rationale): this module's own merge-read
        # logic below (result_file_paths() calls with no `compression=`
        # argument) still assumes every isolated/group-level file is
        # plain CSV - forced here, unconditionally, regardless of what the
        # outer run's own RESULT_COMPRESSION is, so an isolated/group unit
        # never writes a gzip file this merge step would then silently
        # fail to find.
        unit_kwargs['RESULT_COMPRESSION'] = 'none'
        if group_index is None:
            unit_kwargs['RESULT_NAME'] = isolated_result_name(result_name, batch_id, task_index)
            unit_kwargs['MODEL_DISPATCH_FILTER'] = None
            return unit_kwargs
        task_groups = task_plan[task_index]
        members = task_groups[group_index]
        fp8 = _ckpt.model_group_fingerprint(members)
        task_result_name = isolated_result_name(result_name, batch_id, task_index)
        unit_kwargs['RESULT_NAME'] = _group_result_name(task_result_name, group_index, fp8)
        unit_kwargs['MODEL_DISPATCH_FILTER'] = list(members)
        return unit_kwargs

    failed: Dict[Tuple[int, Optional[int]], str] = {}
    if to_run:
        pool_size = max(1, min(n_task_workers * n_model_workers, len(to_run)))
        with ProcessPoolExecutor(
            max_workers=pool_size, mp_context=MP_CONTEXT,
            initializer=worker_init, initargs=(r_path, r_blas_threads, gpu_semaphore),
        ) as pool:
            futures = {pool.submit(_run_unit, _kwargs_for(t, g), t, g): (t, g) for t, g in to_run}
            for future in as_completed(futures):
                unit = futures[future]
                try:
                    _, _, success, error = future.result()
                except Exception as exc:  # noqa: BLE001 - the worker process itself died unexpectedly
                    success, error = False, f"worker process error: {exc!r}"
                if success:
                    print(f"[intra_task_parallel] Batch {batch_id}: unit (task={unit[0]}, "
                          f"group={unit[1]}) finished.")
                else:
                    print(f"[intra_task_parallel] Batch {batch_id}: unit (task={unit[0]}, "
                          f"group={unit[1]}) FAILED: {error}")
                    failed[unit] = error

    # Failure isolation (blueprint R1.12): retry only the units that
    # actually failed - each retry attempt gets its own FRESH, disposable
    # worker process (via run_in_fresh_worker(), Update ID 2 Defect D2/D3
    # fix - see PATCH_NOTES_D2.md), never inline in THIS (parent) process,
    # mirroring intra_batch_parallel.py's own retry loop fix exactly.
    for attempt in range(1, max_unit_retries + 1):
        if not failed:
            break
        retry_units = list(failed.keys())
        # Defect D4 fix: short backoff (with jitter) before retrying -
        # mirrors intra_batch_parallel.py's own retry-loop fix, see the
        # note there beside `_MP_CONTEXT` for the production failure this
        # addresses (a transient network-mount error needs real time to
        # clear, not an immediate same-second retry).
        _backoff = min(15.0, 2.0 * (2 ** (attempt - 1))) * (0.5 + random.random())
        print(f"[intra_task_parallel] Batch {batch_id}: retrying {len(retry_units)} failed unit(s) "
              f"in {_backoff:.1f}s, each in a fresh worker process (attempt "
              f"{attempt}/{max_unit_retries}): {retry_units}")
        time.sleep(_backoff)
        failed = {}
        for (task_index, group_index) in retry_units:
            _, _, success, error = run_in_fresh_worker(
                _run_unit, (_kwargs_for(task_index, group_index), task_index, group_index),
                r_path, r_blas_threads, gpu_semaphore,
            )
            if success:
                print(f"[intra_task_parallel] Batch {batch_id}: unit (task={task_index}, "
                      f"group={group_index}) succeeded on retry.")
            else:
                print(f"[intra_task_parallel] Batch {batch_id}: unit (task={task_index}, "
                      f"group={group_index}) FAILED again on retry: {error}")
                failed[(task_index, group_index)] = error

    if failed:
        raise RuntimeError(
            f"[intra_task_parallel] Batch {batch_id}: {len(failed)} unit(s) failed after "
            f"{max_unit_retries} retry attempt(s): {sorted(failed.keys())}. Re-submit this SAME "
            f"batch (same RESULT_NAME, same batch_id) to resume - every already-succeeded unit "
            f"will be skipped automatically. First failure detail: {next(iter(failed.values()))}"
        )

    # Tier 1: merge each ELIGIBLE task's model-groups into that task's own
    # Phase-2-layout isolated result files, then recompute its naive
    # ensemble - matching Phase 2's task-level isolated-folder shape
    # exactly, so tier 2 below needs no changes at all.
    for task_index in task_indices:
        task_groups = task_plan[task_index]
        if task_groups is None:
            continue
        if _task_merge_already_complete(result_name, batch_id, task_index):
            print(f"[intra_task_parallel] Batch {batch_id}, task {task_index}: tier-1 merge already "
                  f"completed in a previous attempt - skipping.")
            continue
        merge_model_groups_into_task(result_name, batch_id, task_index, task_groups, model_run)
        task_result_name = isolated_result_name(result_name, batch_id, task_index)
        finalise_task_from_merged(task_result_name, task_index, gp_kwargs)

    # Tier 2 (REUSED VERBATIM from intra_batch_parallel.py, blueprint T9):
    # merge every task's own isolated result files - whether fan-out-
    # produced or not, both land in the identical Phase-2 layout - into
    # the real batch-level result files, relocate side artefacts (F1),
    # then clean up the scratch tree.
    merge_isolated_results_into_batch(result_name, batch_id, task_indices)
    for task_index in task_indices:
        relocate_side_artefacts(isolated_result_name(result_name, batch_id, task_index), result_name)
    cleanup_isolated_folders(result_name, batch_id)
    print(f"[intra_task_parallel] Batch {batch_id}: all {len(task_indices)} task index(es) complete "
          f"and merged into the standard batch-level result files.")
