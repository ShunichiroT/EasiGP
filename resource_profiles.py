"""
resource_profiles.py
=====================

Phase 2, Requirement 8 - "Resource Advisor": a quick, non-data-reading
estimate of a reasonable ``ncpus``/``ngpus``/``mem_gb`` request for a given
EasiGP run, computed from cheap config-only inputs (total scenario count,
selected models) plus a small amount of cheap header/line-count I/O (marker
and sample counts - see ``pipeline_utils.cheap_marker_and_sample_counts()``)
- never a full genotype/phenotype materialisation.

This is explicitly a HEURISTIC, calibrated once empirically against a
reference dataset and refined over time - not a guarantee. The GUI
("Suggested Gadi resources" panel in ``main_app.py``) surfaces it with an
explicit disclaimer and a recommendation to pilot a small batch before
committing a large array submission; see ``estimate_resources()``'s own
docstring for the exact caveats.

Kept as its own module (rather than folded into ``pipeline_utils.py``)
because the calibration table below is a distinct, independently-tunable
concern from the generic cheap-read helpers ``pipeline_utils.py`` already
holds (which this module reuses).

--------------------------------------------------------------------------
Update ID 2 (R2) - a SECOND parallelism width.
--------------------------------------------------------------------------
Phase 2's ``estimate_resources()`` modelled exactly one concurrency
dimension (task-level fan-out - ``intra_batch_parallel.py``). Update ID 2
adds a second, orthogonal dimension - model-level fan-out within one task
(``intra_task_parallel.py``) - so this module now recommends a JOINT
``(n_cpu_workers_task, n_model_workers, n_gpu_slots)`` allocation instead
of a single CPU count. Phase 2's own expressions are preserved, byte for
byte, as ``_estimate_resources_phase2_core()`` - a VERBATIM extraction of
what used to be ``estimate_resources()``'s entire body - so that calling
``estimate_resources(..., model_parallel_enabled=False)`` (the default)
is structurally, not just empirically, identical to the pre-Update-2
behaviour (see ``estimate_resources()``'s own docstring, and blueprint
§R2.2's "extracting the Phase-2 body verbatim is what makes the
byte-identical-with-the-flag-off criterion structurally guaranteed rather
than tested-for").

Both new module-level constant groups below (``_POOL_COST_*``,
``_FANOUT_BREAKEVEN_MARGIN``, and the two new ``MODEL_COST_PROFILE``
fields) carry the exact same "provisional, one-time-calibrated, needs
recalibration against a real reference dataset" disclaimer Phase 2 already
attached to ``MODEL_COST_PROFILE``/``GADI_MAX_ARRAY_SUBJOBS`` - implementing
the ALLOCATION LOGIC first and deferring constant recalibration to a
separate follow-up is a binding decision recorded in the ver4-3 design
blueprint §R2.6 (decision D2), not an oversight.

--------------------------------------------------------------------------
Update ID 4 (Additional Requirements 9) - uneven batches, uncoordinated
batch-size/ncpus/N_JOBS suggestions, and marker-filter-blind memory sizing.
--------------------------------------------------------------------------
Four related gaps, all in how this module's suggestions relate to EACH
OTHER (none of them are about the underlying per-model cost table):

1. ``suggest_array_layout()`` used to pick a workload-sized ``batch_size``
   and simply take ``n_batches = ceil(total_tasks_all / batch_size)`` -
   the LAST batch then absorbs whatever remainder is left over (e.g. 400
   tasks at a batch size of 29 -> 13 batches of 29 plus one trailing
   batch of 23), silently under-using that final array subjob's own
   reserved CPUs/memory relative to every other subjob. It now instead
   searches for a batch size that DIVIDES ``total_tasks_all`` evenly
   (falling back to the closest achievable balance when no exact divisor
   exists near the workload-sized target) - see ``_best_balanced_batch_size()``.

2. Nothing previously connected the array-size suggestion to the
   ncpus/``N_CPU_WORKERS_TASK``/``N_JOBS`` suggestion - a person could
   apply a small suggested batch size and a separately-suggested ncpus
   figure that together left CPUs idle (more workers configured than
   that batch even has tasks for) or double-counted (``N_JOBS`` reserving
   threads no worker could ever use concurrently with the task-level
   width already claiming the whole allocation). ``suggest_compute_layout()``
   is a new, explicit joint recommendation for ONE batch of a GIVEN size
   (ideally the balanced size from (1)), reported as
   ``ncpus == n_cpu_workers_task * n_jobs`` always - never a triple that
   requests more than either level can actually use.

3. Investigation (see ``suggest_compute_layout()``'s own docstring for
   the reasoning in full): for this codebase's specific shape - an
   embarrassingly parallel task farm with per-task checkpointing
   (architecture doc §14) and a model roster mixing single-threaded R/BGLR
   MCMC, PLINK2 subprocesses, scikit-learn CPU fits, and GPU-dispatched
   PyTorch models - task-level (array-job) parallelism is the strictly
   preferred lever over ncpus-per-job: it is the only one of the three
   nested levels this module ever recommends (array width, task-level
   worker processes, per-worker ``N_JOBS`` threads) that scales across
   INDEPENDENT node allocations rather than contending for one node's
   shared memory bandwidth/cache, and it is the only one whose failure
   mode (one subjob dies) is cheap under this codebase's own
   fail-forward/checkpoint-per-task design rather than losing an entire
   large multi-CPU allocation's progress. ``N_JOBS`` (library-internal
   threading) is, conversely, the LEAST efficient lever - it is only
   honoured by two of twelve models at all (RF/KNN's own scikit-learn
   ``n_jobs``; SVR's libsvm backend is not internally multi-threaded,
   the four R/BGLR models use ``R_BLAS_THREADS`` instead, and every
   GPU-capable model is GPU-, not CPU-, thread-bound - see
   ``SVR.py``'s/``pipeline_utils.resolve_compute_resources()``'s own
   comments) and is well documented, generally, to show sub-linear/
   negative returns past a handful of threads for tree-ensemble fits.
   ``suggest_compute_layout()`` therefore grows task-level width FIRST
   (up to the batch's own task count), and only spends any remaining CPU
   budget on ``N_JOBS`` once every task in the batch already has its own
   concurrent worker - this ordering is the calibration this
   investigation produced, encoded directly in the allocation logic
   rather than left as a separate constant to tune.

4. LD pruning's and RF importance filtering's marker-count REDUCTION was
   never priced into ``mem_gb``/``ncpus`` at all - every per-model
   memory term was always computed against the RAW, PRE-filter marker
   count, which is a large, model-dependent over-estimate once either
   step is actually enabled (worst case: ``GAT_fully_connected``'s
   ``markers_squared`` scaling, where a 10x marker reduction is a 100x
   memory-term reduction this module previously could never see). RF
   filtering's OWN reduction is fully derivable from its config with zero
   extra data I/O (``mode='count'`` names the exact output size;
   ``mode='percent'`` names an exact ratio) - see
   ``estimate_post_filter_markers()`` - so RF filtering never needs to
   ask the person anything. LD pruning's reduction genuinely depends on
   the real linkage-disequilibrium structure of the data, which this
   advisor deliberately never reads in full (see the module's own
   opening paragraph) - so, per this requirement's own fallback
   instruction, the person is instead OFFERED an optional "estimated
   markers remaining after LD pruning" figure (GUI: inside 'Suggested
   compute resources') when LD pruning is enabled; supplying nothing
   falls back to the previous (over-estimating) behaviour with an
   explicit note explaining why, rather than silently guessing.

--------------------------------------------------------------------------
ver4-4 Stage 6 (RK-6) - suggest_compute_layout() is UNREFERENCED.
--------------------------------------------------------------------------
``suggest_compute_layout()`` (and the model-capability tables/
``_shapley_enabled()`` immediately above it) has no caller anywhere in
this codebase as of ``ver4-4`` - the GUI panel that used to call it was
deleted in Stage 2 (R2). See that function's own docstring for the full
disclosure; kept updated (not deleted) as reference material / a possible
future re-integration point only.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from pipeline_utils import cheap_marker_and_sample_counts
# ver4-4 Stage 6 (R3.f/R3.g blueprint §10 checklist - rewrite
# _N_JOBS_CAPABLE_MODELS/tier logic): reused, never re-derived (the same
# "one implementation, no drift" discipline architecture doc §8 already
# establishes for _map_genes_to_markers) - DIAGNOSTIC_FLAG_FIELDS is the
# existing, single source of truth for "which positional HPARAMETERS index
# is this model's own Shapley/interaction/marker-effect toggle", already
# covering every model this section's tier logic needs (RF/SVR/KNN, plus
# GBLUP/RKHS's own 'Return marker effect?' toggle - the SAME toggle R3.d's
# row-range fan-out is gated on in genomic_prediction.py::_run_gblup_or_
# rkhs). HPARAM_SPECS supplies each of those same fields' own documented
# default, so this module never hand-maintains a second, driftable copy of
# either mapping.
from models.hyperparameter_tuning import DIAGNOSTIC_FLAG_FIELDS
from hparam_specs import HPARAM_SPECS

# --------------------------------------------------------------------------
# Static per-model relative cost profile - see architecture doc §9 for what
# each model family actually does; this table is a coarse, ONE-TIME-
# CALIBRATED (against a reference dataset) approximation of relative
# CPU-time and memory scaling, not a precise cost model. Recalibrate these
# constants against real benchmark runs on your own reference dataset/
# hardware periodically - they will drift as models/data change.
#
# cpu_time_factor: relative CPU-seconds-per-task multiplier (RF == 1.0 is
#     the reference point every other factor is expressed relative to).
#     UNCHANGED by Update ID 2 - every existing value is preserved exactly,
#     which is what keeps _estimate_resources_phase2_core() byte-identical.
# mem_scaling: how memory scales with n_markers/n_samples - 'markers',
#     'markers_samples', 'markers_squared', or 'genes' (independent of raw
#     marker count, since bio-prior models operate on GENE nodes, not raw
#     markers - see architecture doc §12.3). UNCHANGED by Update ID 2.
# gpu_capable: whether this model can use a GPU at all (see
#     pipeline_utils.resolve_compute_resources()'s TORCH_DEVICE handling).
#     UNCHANGED by Update ID 2.
#
# Update ID 2 (R2, blueprint T15) adds two fields, ADDITIVE only:
# fit_cost_factor: cost of the model FIT alone (RF-relative), as distinct
#     from cpu_time_factor's original, coarser meaning of "everything this
#     model's dispatch does". R1's model-level fan-out pays the Steps-1-8
#     POOL cost once per (task, model-group) rather than once per task, so
#     pricing that trade-off (estimate_pool_cost() below) requires the fit
#     cost to be separable from the pool cost - cpu_time_factor alone
#     conflated the two. Set equal to the existing cpu_time_factor for
#     every entry in this delivery (D2: logic first, recalibration is a
#     separate follow-up) - NOT because the two are assumed identical in
#     reality, but because no better number exists yet without a benchmark
#     this advisor is deliberately designed to avoid needing.
# peak_mem_gb_per_worker: provisional, ABSOLUTE (not scaled by n_markers/
#     n_samples - unlike mem_scaling) resident-memory estimate, in GB, for
#     ONE worker process fitting ONE instance of this model - used by
#     estimate_worker_memory() for R2's memory clamp. Never previously
#     modelled at all (Phase 2's mem_gb was a single-process figure with
#     no per-model breakdown) - these are coarse, order-of-magnitude
#     placeholders (R/BGLR MCMC and GAT_fully_connected's O(markers^2)
#     attention matrix are given the highest figures; everything else a
#     modest, roughly uniform baseline), explicitly flagged provisional.
# --------------------------------------------------------------------------
MODEL_COST_PROFILE: Dict[str, Dict[str, object]] = {
    'rrBLUP':                          {'cpu_time_factor': 3.0, 'fit_cost_factor': 3.0, 'mem_scaling': 'markers_samples', 'peak_mem_gb_per_worker': 3.0, 'gpu_capable': False},
    'GBLUP':                           {'cpu_time_factor': 3.0, 'fit_cost_factor': 3.0, 'mem_scaling': 'markers_samples', 'peak_mem_gb_per_worker': 3.0, 'gpu_capable': False},
    'BayesB':                          {'cpu_time_factor': 6.0, 'fit_cost_factor': 6.0, 'mem_scaling': 'markers_samples', 'peak_mem_gb_per_worker': 4.0, 'gpu_capable': False},
    'RKHS':                            {'cpu_time_factor': 4.0, 'fit_cost_factor': 4.0, 'mem_scaling': 'markers_samples', 'peak_mem_gb_per_worker': 4.0, 'gpu_capable': False},
    'RF':                              {'cpu_time_factor': 1.0, 'fit_cost_factor': 1.0, 'mem_scaling': 'markers', 'peak_mem_gb_per_worker': 1.5, 'gpu_capable': False},
    # Update ID ver4-5, R2 Stages 8/11: Tier 1 (ExtraTrees, GBDT) and
    # Tier 2 (XGBoost, EBM - optional dependency, gpu_capable left False
    # for all four: no cuML/GPU backend is wired for any of them this
    # delivery, a disclosed scope decision - see each model's own file).
    # Figures are provisional placeholders in the SAME spirit as every
    # other entry in this table (D2 above) - relative to RF's own 1.0,
    # not independently benchmarked.
    'ExtraTrees':                      {'cpu_time_factor': 1.0, 'fit_cost_factor': 1.0, 'mem_scaling': 'markers', 'peak_mem_gb_per_worker': 1.5, 'gpu_capable': False},
    'GBDT':                            {'cpu_time_factor': 1.5, 'fit_cost_factor': 1.5, 'mem_scaling': 'markers', 'peak_mem_gb_per_worker': 1.5, 'gpu_capable': False},
    'XGBoost':                         {'cpu_time_factor': 0.8, 'fit_cost_factor': 0.8, 'mem_scaling': 'markers', 'peak_mem_gb_per_worker': 1.5, 'gpu_capable': False},
    'EBM':                             {'cpu_time_factor': 2.0, 'fit_cost_factor': 2.0, 'mem_scaling': 'markers', 'peak_mem_gb_per_worker': 1.5, 'gpu_capable': False},
    'SVR':                             {'cpu_time_factor': 1.2, 'fit_cost_factor': 1.2, 'mem_scaling': 'markers', 'peak_mem_gb_per_worker': 1.0, 'gpu_capable': False},
    'KNN':                             {'cpu_time_factor': 0.6, 'fit_cost_factor': 0.6, 'mem_scaling': 'markers', 'peak_mem_gb_per_worker': 1.0, 'gpu_capable': False},
    'MLP':                             {'cpu_time_factor': 0.8, 'fit_cost_factor': 0.8, 'mem_scaling': 'markers', 'peak_mem_gb_per_worker': 1.5, 'gpu_capable': True},
    'GAT_infinitesimal':               {'cpu_time_factor': 2.5, 'fit_cost_factor': 2.5, 'mem_scaling': 'markers', 'peak_mem_gb_per_worker': 2.0, 'gpu_capable': True},
    'GAT_infinitesimal_node_level':    {'cpu_time_factor': 2.7, 'fit_cost_factor': 2.7, 'mem_scaling': 'markers', 'peak_mem_gb_per_worker': 2.0, 'gpu_capable': True},
    'GAT_fully_connected':             {'cpu_time_factor': 8.0, 'fit_cost_factor': 8.0, 'mem_scaling': 'markers_squared', 'peak_mem_gb_per_worker': 6.0, 'gpu_capable': True},
    'GAT_prior_knowledge':             {'cpu_time_factor': 4.0, 'fit_cost_factor': 4.0, 'mem_scaling': 'markers', 'peak_mem_gb_per_worker': 2.5, 'gpu_capable': True},
    'GAT_biological_prior_knowledge':  {'cpu_time_factor': 3.0, 'fit_cost_factor': 3.0, 'mem_scaling': 'genes', 'peak_mem_gb_per_worker': 1.5, 'gpu_capable': True},
    'ensemble':                        {'cpu_time_factor': 0.1, 'fit_cost_factor': 0.1, 'mem_scaling': 'markers', 'peak_mem_gb_per_worker': 0.5, 'gpu_capable': False},
}
_DEFAULT_MODEL_PROFILE = {
    'cpu_time_factor': 2.0, 'fit_cost_factor': 2.0, 'mem_scaling': 'markers',
    'peak_mem_gb_per_worker': 2.0, 'gpu_capable': False,
}

# Additional Requirements 9 (production defect fix - see
# estimate_worker_memory()'s own docstring and the incident write-up
# beside intra_batch_parallel._systemic_failure_note()) - the four R/BGLR
# models, whose per-task MCMC fits all run inside the SAME persistent
# rpy2/R global environment for one GP() call (architecture doc §9.1), so
# their own peak_mem_gb_per_worker figures are SUMMED (not maxed) when
# more than one is selected for the same worker/task.
_R_ACCUMULATING_MODELS = frozenset({'rrBLUP', 'GBLUP', 'BayesB', 'RKHS'})

# Base memory (GB) even for a trivial run - Python/pandas/R/PyTorch process
# overhead, imports, and small fixed buffers.
_BASE_MEM_GB = 2.0

# Very small numbers of tasks make per-batch overhead (R sourcing, process
# startup) dominate - never suggest below this floor regardless of how
# small the run is.
_MIN_NCPUS = 1
_MIN_MEM_GB = 4.0

# Gadi standard node/queue increments (normal queue, Cascade Lake) - used
# only to round a raw ncpus estimate up to a sensible request; verify
# against current NCI documentation, this is not treated as authoritative.
_GADI_NCPUS_INCREMENT = 4

# --------------------------------------------------------------------------
# Patch 3, Requirement 4 - "the suggestion of optimum CPUs needs to be
# calibrated when using a GPU, since the use of a GPU needs more CPUs.
# This also relates to the updates on Req. 3": before this, `ncpus` was
# sized purely from `total_units` (model-fitting/tuning work) and never
# adjusted upward just because `ngpus > 0` - so a GPU-capable run got
# EXACTLY the same ncpus suggestion as an equivalent CPU-only one, despite
# two independent, concrete reasons a GPU run genuinely needs MORE:
#
#   1. Gadi's own GPU queue ('gpuvolta') REQUIRES ncpus to be a multiple
#      of 12 per GPU requested (main_app.py's own HPC-export help text
#      already documents this constraint, but nothing computed a
#      compliant value automatically before this fix).
#   2. LD pruning (plink2) and RF-based marker filtering/selection
#      (Preprocess/LD_pruning.py, Preprocess/RF_marker_filtering.py) run
#      on the CPU REGARDLESS of whether the model fit itself uses the
#      GPU, and now correctly respect the job's actual CPU allocation
#      (patch 3, Requirement 3 - see those modules' own
#      _resolve_plink_threads()/_resolve_n_jobs() docstrings) rather than
#      oversubscribing/thrashing on an under-provisioned allocation. That
#      fix only pays off if the allocation ITSELF is large enough - on a
#      lean, GPU-sized CPU request, correctly-threaded preprocessing is
#      merely correctly slow, not thrashing-slow, which is what motivated
#      this calibration: give GPU jobs enough real CPU headroom that
#      LD/RF preprocessing isn't starved while the GPU does the fit.
_GADI_GPU_NCPUS_PER_GPU = 12

# Extra CPU headroom multiplier applied ONLY when a GPU run ALSO has LD
# pruning and/or RF filtering enabled - i.e. exactly the combination
# Requirement 3 names ("especially true when combining filtering
# approaches and the GAT biological prior knowledge model"). Provisional,
# same footing as every other cost constant in this module (D2: logic
# first, recalibrate against a real benchmark later) - 1.5x means "50%
# more raw ncpus units than the base estimate already computed", not an
# absolute headcount.
_GPU_PREPROCESS_CPU_HEADROOM = 1.5
# --------------------------------------------------------------------------
# Update ID 2 (R2, blueprint §R2.2) - Steps-1-8 "pool" cost model, in the
# SAME RF-relative units as MODEL_COST_PROFILE's cpu_time_factor/
# fit_cost_factor. R1's model-level fan-out pays this cost once PER
# (task, model-group) rather than once per task - should_fan_out_models()
# below is the shared break-even predicate that prices the trade-off.
#
# PROVISIONAL, exactly like MODEL_COST_PROFILE itself (D2: logic first,
# recalibrate against real benchmark runs later) - _POOL_COST_BASE is
# calibrated to roughly one RF-fit's worth of phenotype-subsetting/
# dedup/split/marker-pool-routing overhead; the three additive terms
# reflect LD pruning, RF importance filtering, and PLINK --extract/--keep
# materialisation each adding a comparable, separately-toggleable amount
# of per-task preprocessing work (architecture doc §7 Steps 3/6/7).
# --------------------------------------------------------------------------
_POOL_COST_BASE = 1.0
_POOL_COST_LD_PRUNE = 1.5
_POOL_COST_RF_FILTER = 1.0
_POOL_COST_PLINK_EXTRACT = 0.5

# Update ID 3 (R4, blueprint §R4.5(2)) - PROVISIONAL, un-benchmarked, same
# footing as the four constants immediately above: how many "mem_units"
# (the SAME unit _mem_scale_factor()'s 'markers' case returns, i.e. what
# _estimate_resources_phase2_core()'s own mem_gb = _BASE_MEM_GB +
# mem_units/50.0 already divides by) one unit of `estimate_pool_cost()`'s
# RF-relative cost above `_POOL_COST_BASE` is worth, in
# `_apply_pool_cost_to_core()` below. Recalibration against a real
# benchmark stays explicitly deferred (EasiGP_2_Change_Summary.md §8);
# this requirement is about the advisor headline being filtering-AWARE at
# all (it previously wasn't, structurally - R4.2), not about calibrating
# it precisely.
_POOL_MEM_UNITS_PER_COST = 8.0

# Require the wall-clock time SAVED by fanning model fits out across G
# groups to exceed the redundant (G-1)x pool-rebuild cost by at least this
# fractional margin before recommending fan-out at all - prevents
# recommending (and the runtime eligibility gate agreeing to) a marginal
# benefit that real-world overhead (process startup, R sourcing) would
# likely erase. Shared, verbatim, between this module's advisor and
# intra_task_parallel.py's runtime eligibility gate (should_fan_out_models()
# below) - see A2.6.
_FANOUT_BREAKEVEN_MARGIN = 1.2

# Default GPU-slot capacity per physical GPU device - see
# pipeline_utils.gpu_slot()/resolve_compute_resources(); a device can
# usually host more than one concurrent GPU-dispatched model call if each
# model's own memory footprint is small relative to the device's, but 1
# (no oversubscription) is the safe, zero-configuration default.
_GPU_SLOTS_PER_DEVICE_DEFAULT = 1

# --------------------------------------------------------------------------
# Patch 3 v2, Requirement 1 - "the suggestion of optimum resources does not
# consider the total number of target phenotypes, selected prediction
# models and the numbers and different ratios ... considering such
# information enables the calculation of the optimum array size. The new
# version should also suggest the optimum size of the job array."
#
# The ncpus/mem_gb/ngpus headline above already folds in n_phenotype/
# n_ratio/n_population/sample_num (via compute_total_tasks(), called from
# _estimate_resources_phase2_core()) and every selected model (via
# MODEL_COST_PROFILE / units_per_task) - what was genuinely missing is a
# suggestion for the PARALLEL batch_size / resulting job-array size
# (n_batches) itself, which is what suggest_array_layout() below adds.
#
# _GADI_MAX_ARRAY_SUBJOBS mirrors main_app.py's own GADI_MAX_ARRAY_SUBJOBS
# constant - DUPLICATED rather than imported, following this module's own
# established "no cross-module coupling" design goal (see _base_of()'s own
# docstring, and the _resolve_plink_threads()/plink_io.py precedent
# documented in Preprocess/LD_pruning.py) - keep the two values in sync by
# hand if NCI's documented cap ever changes; VERIFY against NCI's current
# documentation at deployment time, same caveat main_app.py's own copy
# already carries.
#
# _TARGET_UNITS_PER_BATCH is PROVISIONAL, same footing as every other cost
# constant in this module (D2: logic first, recalibrate against a real
# benchmark later) - chosen so that, at the DEFAULT MODEL_COST_PROFILE
# scale (RF == 1.0), a batch of ~200 units corresponds to roughly the same
# "one workable chunk" scale as the ncpus estimate's own
# `raw_ncpus = ceil(total_units / 25.0)` step (i.e. ~8 raw ncpus' worth of
# work per batch before Gadi-increment rounding) - small enough that a
# single batch's own wall-clock stays modest, large enough that per-task
# fixed overhead (R sourcing, process startup, one BGLR_output/ directory
# per task) doesn't dominate.
# --------------------------------------------------------------------------
_GADI_MAX_ARRAY_SUBJOBS = 500
_TARGET_UNITS_PER_BATCH = 200.0
_MIN_SUGGESTED_BATCH_SIZE = 1

# --------------------------------------------------------------------------
# Patch 3 v3, Requirement 3 - "The suggested resource section is now for
# Gadi (at least it is written as it is). This part needs to be
# generalised to other HPCs such as Bunya as well."
#
# The two numbers this module's estimate genuinely varies by HPC target
# are (a) the job-array subjob cap (``suggest_array_layout()``'s own
# ``max_array_subjobs``, previously always ``_GADI_MAX_ARRAY_SUBJOBS``)
# and (b) the CPUs-required-per-GPU floor (``_apply_gpu_cpu_calibration_
# to_core()``'s own gpu_floor, previously always
# ``_GADI_GPU_NCPUS_PER_GPU``). Both are now looked up from a small named
# profile table instead of being hardcoded, so a person targeting a
# different cluster gets a calibration appropriate to THAT cluster rather
# than Gadi's - without changing any of the estimation LOGIC itself, only
# which two numbers it is parameterised by (mirrors this module's own
# established "keep the allocation logic generic, vary only the
# constants" pattern, e.g. MODEL_COST_PROFILE).
#
# Deliberately NOT generalised here: the "Gadi native job-array"
# submission-file generator in main_app.py (``run_batch_array.pbs`` /
# ``submit_arrays.sh``) is a genuinely NCI-Gadi-SPECIFIC mechanism (a
# native PBS job-array submitted in chunks via a driver script) - it is
# not part of "the suggested resource section" this requirement names,
# and main_app.py's scheduler choice already covers Slurm/PBS/NCI/Bunya
# (Bunya submits as plain Slurm) independently of this profile table.
#
# 'UQ Bunya' figures: Bunya's own user guide (https://github.com/
# UQ-RCC/hpc-docs/blob/main/guides/Bunya-User-Guide.md) documents 96
# cores/node and explicitly states its GPU nodes have VARYING CPU counts
# ("Users are reminded to check and request sensible CPU numbers with
# their GPU requests") - i.e., unlike Gadi's gpuvolta queue, there is no
# single fixed CPUs-per-GPU ratio to enforce, so ``ncpus_per_gpu`` is 0
# (no floor applied) for this profile. Its job-array cap is left at
# Slurm's own common default (``MaxArraySize``, typically 1001) since
# Bunya's guide does not document a site-specific override - VERIFY
# against Bunya's actual `scontrol show config | grep MaxArraySize` at
# deployment time, same "not authoritative, verify at deployment"
# footing every other constant in this module already carries.
#
# 'Generic / other HPC' is a deliberately conservative fallback (no GPU
# floor, a modest array cap) for any cluster that isn't Gadi or Bunya -
# the person can override either number directly via
# ``estimate_resources()``'s own ``max_array_subjobs_override``/
# ``ncpus_per_gpu_override`` keyword arguments (surfaced in the GUI as
# the "Custom" profile - see main_app.py's HPC cluster profile widget).
# --------------------------------------------------------------------------
HPC_RESOURCE_PROFILES: Dict[str, Dict[str, object]] = {
    'NCI Gadi': {
        'max_array_subjobs': _GADI_MAX_ARRAY_SUBJOBS,
        'ncpus_per_gpu': _GADI_GPU_NCPUS_PER_GPU,
        'note': (
            "NCI Gadi's gpuvolta queue requires ncpus to be a multiple of "
            f"{_GADI_GPU_NCPUS_PER_GPU} per GPU requested; PBS/NCI job arrays are capped at "
            f"{_GADI_MAX_ARRAY_SUBJOBS} subjobs."
        ),
    },
    'UQ Bunya': {
        'max_array_subjobs': 1001,
        'ncpus_per_gpu': 0,
        'note': (
            "UQ Bunya's GPU-node CPU count varies by node type (no single fixed "
            "CPUs-per-GPU ratio, unlike Gadi) - no automatic ncpus floor is applied; the "
            "1001 array-size cap is Slurm's common default MaxArraySize - verify against "
            "Bunya's own current Slurm configuration."
        ),
    },
    'Generic / other HPC': {
        'max_array_subjobs': 1000,
        'ncpus_per_gpu': 0,
        'note': (
            "No cluster-specific limits are known for this profile - verify your own "
            "scheduler's job-array cap and CPUs-per-GPU ratio, or pick 'Custom' to enter "
            "them directly."
        ),
    },
}
DEFAULT_HPC_PROFILE = 'NCI Gadi'


def resolve_hpc_profile(
    hpc_profile: Optional[str] = None,
    max_array_subjobs_override: Optional[int] = None,
    ncpus_per_gpu_override: Optional[int] = None,
) -> Tuple[int, int, str, str]:
    """Patch 3 v3, Requirement 3 - resolve the (max_array_subjobs,
    ncpus_per_gpu) pair this module's estimate should calibrate against,
    given a named profile (``HPC_RESOURCE_PROFILES`` above) and/or
    explicit overrides ('Custom' in the GUI - either override, supplied
    independently, wins over the named profile's own value for that ONE
    number only, so a person can e.g. keep Gadi's array cap but supply
    their own GPU/CPU ratio).

    An unrecognised/empty `hpc_profile` falls back to
    `DEFAULT_HPC_PROFILE` ('NCI Gadi') - i.e. any existing caller that
    never passes this argument at all gets EXACTLY today's Gadi-only
    behaviour, unchanged.

    Returns (max_array_subjobs, ncpus_per_gpu, profile_note,
    resolved_profile_name) - the last element is which profile key was
    ACTUALLY used (never the raw, possibly-unrecognised `hpc_profile`
    argument verbatim), so a caller that echoes it back (e.g.
    ``estimate_resources()``'s own ``core['hpc_profile']``) always
    reports a real, resolvable profile name.
    """
    resolved_name = hpc_profile if hpc_profile in HPC_RESOURCE_PROFILES else DEFAULT_HPC_PROFILE
    profile = HPC_RESOURCE_PROFILES[resolved_name]
    max_array_subjobs = (
        int(max_array_subjobs_override) if max_array_subjobs_override
        else int(profile['max_array_subjobs'])
    )
    ncpus_per_gpu = (
        int(ncpus_per_gpu_override) if ncpus_per_gpu_override is not None
        else int(profile['ncpus_per_gpu'])
    )
    return max_array_subjobs, ncpus_per_gpu, str(profile.get('note', '')), resolved_name


# Additional Requirements 9, Requirement 1 - how far around the workload-
# sized target n_batches (rounded) the balanced-fallback search in
# _best_balanced_batch_size() looks before giving up on finding a better
# balance than the target itself - a small, fixed window (cheap: at most
# 2*_BATCH_BALANCE_SEARCH_RADIUS+1 candidate n_batches values, each O(1))
# rather than an exhaustive scan of every possible n_batches up to
# max_array_subjobs.
_BATCH_BALANCE_SEARCH_RADIUS = 8

# Additional Requirements 9, Requirement 1 - how far from the workload-
# sized target an EXACT divisor of total_tasks_all is allowed to be before
# _best_balanced_batch_size() prefers it over the "closest achievable
# balance" fallback. Without this bound, a total task count with no
# divisor anywhere near the target (e.g. a PRIME total) would fall back to
# batch_size=1 (total_tasks_all's only small divisor) - technically
# "perfectly even" (every batch has exactly 1 task) but a strictly WORSE
# outcome than a slightly uneven batch near the intended size: batch_size=1
# means the maximum possible per-task FIXED overhead (R sourcing, process
# startup, one BGLR_output/ directory - see architecture doc §9.1/§17) is
# paid by every single task with no batching benefit at all. A candidate
# divisor is only accepted when it is within
# [target/_DIVISOR_SEARCH_TOLERANCE, target*_DIVISOR_SEARCH_TOLERANCE] of
# the workload-sized target.
_DIVISOR_SEARCH_TOLERANCE = 2.0


def _divisors(n: int) -> List[int]:
    """Every positive divisor of ``n`` (including 1 and ``n`` itself),
    ascending, found by trial division up to ``sqrt(n)`` - O(sqrt(n)) and
    trivially fast even for ``n`` in the millions (this module's own
    "well under a second" performance target - see the module docstring),
    since a GUI-facing advisor never needs this for anything larger than a
    run's own total task count. Returns ``[]`` for ``n <= 0``."""
    if n <= 0:
        return []
    divs: List[int] = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    return sorted(divs)


def _best_balanced_batch_size(
    total_tasks_all: int, target_batch_size: int, lo: int, hi: int,
) -> Tuple[int, int, bool]:
    """Additional Requirements 9, Requirement 1 - choose a ``batch_size``
    in ``[lo, hi]`` that keeps every batch's own task count as equal as
    possible, preferring one close to ``target_batch_size`` (the
    workload-sized figure ``suggest_array_layout()`` would otherwise have
    used outright).

    Three-stage search:

    1. **Exact balance, close to the target.** Any divisor of
       ``total_tasks_all`` within ``[lo, hi]`` AND within
       ``_DIVISOR_SEARCH_TOLERANCE`` of ``target_batch_size`` (see that
       constant's own docstring for why the tolerance exists) gives EVERY
       batch identically ``total_tasks_all / batch_size`` tasks - no
       trailing, smaller batch at all. Among all such divisors
       (``_divisors()`` above), pick the one closest to
       ``target_batch_size``; ties broken toward the SMALLER divisor (more,
       narrower batches) - the Requirement 3 calibration this module's own
       docstring documents (task-level/array-level parallelism is the
       preferred lever over a larger per-batch footprint).
    2. **Closest achievable balance.** When no divisor exists close enough
       to the target (``total_tasks_all`` is prime, or ``[lo, hi]``/the
       tolerance window is too narrow to contain one), search ``n_batches``
       in a small window around the workload-sized target and, for each,
       take ``batch_size = ceil(total_tasks_all / n_batches)`` - the SAME
       fixed-size-except-the-last scheme the runtime (``GP()``'s own
       sharding - architecture doc §6) actually uses - and keep whichever
       ``n_batches`` minimises the resulting imbalance (``batch_size -
       last_batch_size``; 0 would be exact balance), breaking ties toward
       whichever is closest to ``target_batch_size``. This step can still
       land on an exact divisor "by luck" (``evenly_divides`` is always
       derived from the actual ``total_tasks_all % batch_size`` outcome,
       never assumed).
    3. **Distant-divisor fallback.** Only if step 2 finds nothing valid at
       all (a pathologically narrow ``[lo, hi]``) does this widen the
       divisor search back out to the FULL ``[lo, hi]`` range regardless of
       distance from the target - a distant exact divisor is still a
       better floor than an arbitrary clamp.

    Returns ``(batch_size, n_batches, evenly_divides)``. Never returns a
    ``batch_size`` outside ``[lo, hi]`` (assuming ``lo <= hi``, which every
    caller in this module guarantees)."""
    lo = max(1, int(lo))
    hi = max(lo, int(hi))
    target_batch_size = max(lo, min(hi, int(target_batch_size)))

    divisor_lo = max(lo, math.floor(target_batch_size / _DIVISOR_SEARCH_TOLERANCE))
    divisor_hi = min(hi, math.ceil(target_batch_size * _DIVISOR_SEARCH_TOLERANCE))
    if divisor_lo <= divisor_hi:
        candidates = [d for d in _divisors(total_tasks_all) if divisor_lo <= d <= divisor_hi]
        if candidates:
            best_divisor = min(candidates, key=lambda d: (abs(d - target_batch_size), d))
            return best_divisor, total_tasks_all // best_divisor, True

    target_n_batches = max(1, round(total_tasks_all / target_batch_size))
    best: Optional[Tuple[Tuple[int, int], int, int]] = None
    lo_n = max(1, target_n_batches - _BATCH_BALANCE_SEARCH_RADIUS)
    hi_n = target_n_batches + _BATCH_BALANCE_SEARCH_RADIUS
    for n_batches in range(lo_n, hi_n + 1):
        batch_size = math.ceil(total_tasks_all / n_batches)
        if not (lo <= batch_size <= hi):
            continue
        last_batch_size = total_tasks_all - (n_batches - 1) * batch_size
        if last_batch_size <= 0:
            continue
        imbalance = batch_size - last_batch_size
        score = (imbalance, abs(batch_size - target_batch_size))
        if best is None or score < best[0]:
            best = (score, batch_size, n_batches)

    if best is not None:
        _, batch_size, n_batches = best
        return batch_size, n_batches, (total_tasks_all % batch_size == 0)

    # Stage 3 - the neighbourhood search above found nothing valid in
    # [lo, hi] (only possible for a pathologically narrow window): widen
    # the divisor search back out to the FULL [lo, hi] range regardless of
    # distance from the target - a distant exact divisor is still a better
    # floor than an arbitrary clamp.
    candidates = [d for d in _divisors(total_tasks_all) if lo <= d <= hi]
    if candidates:
        best_divisor = min(candidates, key=lambda d: (abs(d - target_batch_size), d))
        return best_divisor, total_tasks_all // best_divisor, True

    # Truly degenerate fallback (no divisor at all in [lo, hi] - can only
    # happen when total_tasks_all itself falls outside [lo, hi], which
    # every caller in this module is expected to avoid): clamp the target
    # directly, same floor every earlier version of this function used.
    batch_size = max(lo, min(hi, target_batch_size))
    n_batches = max(1, math.ceil(total_tasks_all / batch_size))
    return batch_size, n_batches, (total_tasks_all % batch_size == 0)


def enumerate_array_layouts(
    total_tasks_all: int,
    max_array_subjobs: int = _GADI_MAX_ARRAY_SUBJOBS,
    max_options: int = 12,
    preferred_batch_size: Optional[int] = None,
) -> List[Dict[str, object]]:
    """ver4-4, R1 (blueprint §2.1) - every ``(batch_size, n_batches)`` pair
    that divides ``total_tasks_all`` EXACTLY
    (``batch_size * n_batches == total_tasks_all``, so every batch in the
    resulting job array processes an IDENTICAL task count - no smaller
    trailing batch at all), filtered to ``n_batches <= max_array_subjobs``
    (a job array wider than the cluster's own hard cap is never offered),
    ordered by DESCENDING ``batch_size`` (fewest batches / widest
    per-batch footprint first), and capped to at most ``max_options``
    entries (evenly spread across the full candidate list, always
    keeping both extremes) so the panel never has to render an unusably
    long list for a highly-composite task count.

    Reuses ``_divisors()`` - the SAME trial-division machinery
    ``_best_balanced_batch_size()`` already uses for its own "exact
    balance" search stage - rather than a second, divergent
    implementation (I6's discipline, applied here to resourcing rather
    than markers; see this module's own docstring and the blueprint's
    §2.1.2 "Rejected alternative").

    Each returned dict is ``{'batch_size': int, 'n_batches': int,
    'is_recommended': bool}``. Exactly one entry - whichever
    ``batch_size`` is closest to ``preferred_batch_size`` (ties broken
    toward the SMALLER batch size, the same tie-break
    ``_best_balanced_batch_size()`` uses) - is marked
    ``is_recommended=True``, so the workload-sized heuristic still
    surfaces as a sensible default without forcing it. A caller should
    normally pass ``suggest_array_layout(...)['suggested_batch_size']``
    here. When ``preferred_batch_size`` is ``None`` (no workload
    estimate supplied), the single-batch pair (``batch_size ==
    total_tasks_all``, the fewest possible batches) is recommended
    instead - a neutral default that is always in range once
    ``max_array_subjobs >= 1``.

    Returns ``[]`` when ``total_tasks_all <= 0`` (nothing configured yet
    - the caller shows an info message in that case, exactly like
    today) or when every divisor of ``total_tasks_all`` produces more
    batches than ``max_array_subjobs`` allows (only possible for a
    pathologically large prime-like total against a very small cap).
    Callers should fall back to ``suggest_array_layout()``'s own single,
    closest-achievable-balance answer whenever this returns fewer than 2
    entries (a total that is prime, or otherwise has too few in-range
    divisors, offers no genuine choice) - the documented degradation
    path, not an error."""
    total_tasks_all = int(total_tasks_all)
    if total_tasks_all <= 0:
        return []
    max_array_subjobs = max(1, int(max_array_subjobs))
    max_options = max(1, int(max_options))

    pairs: List[Tuple[int, int]] = []
    for batch_size in _divisors(total_tasks_all):
        n_batches = total_tasks_all // batch_size
        if n_batches <= max_array_subjobs:
            pairs.append((batch_size, n_batches))
    if not pairs:
        return []

    # Descending batch_size - fewest batches (widest per-batch
    # footprint) first.
    pairs.sort(key=lambda p: p[0], reverse=True)
    if len(pairs) > max_options:
        # Keep an even spread across the full sorted list, always
        # including both extremes (single-batch and most-parallel), so
        # trimming never silently hides either end of the range.
        step = (len(pairs) - 1) / (max_options - 1) if max_options > 1 else 0
        keep_idx = sorted({round(i * step) for i in range(max_options)})
        pairs = [pairs[i] for i in keep_idx]

    target = total_tasks_all if preferred_batch_size is None else int(preferred_batch_size)
    recommended_batch_size = min((p[0] for p in pairs), key=lambda bs: (abs(bs - target), bs))

    return [
        {'batch_size': int(bs), 'n_batches': int(nb), 'is_recommended': bool(bs == recommended_batch_size)}
        for bs, nb in pairs
    ]


def suggest_array_layout(
    total_tasks_all: int,
    units_per_task: int,
    max_array_subjobs: int = _GADI_MAX_ARRAY_SUBJOBS,
    target_units_per_batch: float = _TARGET_UNITS_PER_BATCH,
) -> Dict[str, object]:
    """Patch 3 v2, Requirement 1 - suggest a PARALLEL ``batch_size`` (and the
    resulting job-array size, ``n_batches``) for this run, given its ACTUAL
    total scenario count (``total_tasks_all`` - already phenotype/model/
    ratio-aware; see ``compute_total_tasks()``) and per-task cost
    (``units_per_task``; see ``compute_units_per_task()``).

    A small, deliberately simple heuristic (D2: logic first):

    1. Aim for roughly ``target_units_per_batch`` "progress units" (the SAME
       units ``compute_units_per_task()``/this module's own
       ``raw_ncpus = ceil(total_units / 25.0)`` step use) per batch - too
       few tasks per batch pays disproportionate FIXED per-task overhead
       (R sourcing, process startup, checkpoint bookkeeping) once per
       scheduler subjob; too many makes one subjob's own wall-clock balloon
       and coarsens the checkpoint/resume grain unnecessarily.
    2. NEVER suggest a job array wider than ``max_array_subjobs`` (Gadi's
       own hard cap, architecture-independent of this run's workload) - if
       step 1's batch size alone would need more batches than that, the
       batch size is grown just enough to bring the array back under the
       cap (this constraint always wins over step 1's target).
    3. Never below ``_MIN_SUGGESTED_BATCH_SIZE`` (1) task/batch, and never a
       batch size larger than the total task count itself (a single batch
       covering the whole run is the natural floor on ``n_batches``, not
       something to exceed).

    Additional Requirements 9, Requirement 1 - within whatever range steps
    2-3 leave open, the batch size actually chosen now comes from
    ``_best_balanced_batch_size()`` above rather than a bare ``round()`` of
    step 1's target: every batch gets an IDENTICAL task count whenever
    ``total_tasks_all`` has a divisor near that target (the common case),
    and the closest achievable balance otherwise - never silently leaving
    a much-smaller trailing batch the way a plain
    ``ceil(total_tasks_all / batch_size)`` could.

    Returns
    -------
    Dict with ``suggested_batch_size`` (int), ``suggested_n_batches`` (int),
    ``last_batch_size`` (int - the actual task count of the FINAL batch;
    equals ``suggested_batch_size`` whenever ``batch_size_evenly_divides``
    is True), ``batch_size_evenly_divides`` (bool), and ``notes``
    (list[str] - e.g. explaining when the Gadi-cap constraint (2) was the
    binding one rather than the workload-sizing heuristic (1), and always
    stating plainly whether every batch ends up the same size). A run with
    zero total tasks (e.g. nothing configured yet) returns
    ``suggested_batch_size=1``, ``suggested_n_batches=0``,
    ``last_batch_size=0``, ``batch_size_evenly_divides=True``, no notes.
    """
    notes: List[str] = []
    total_tasks_all = max(0, int(total_tasks_all))
    units_per_task = max(1, int(units_per_task))
    max_array_subjobs = max(1, int(max_array_subjobs))

    if total_tasks_all <= 0:
        return {
            'suggested_batch_size': 1, 'suggested_n_batches': 0,
            'last_batch_size': 0, 'batch_size_evenly_divides': True, 'notes': notes,
        }

    raw_batch_size = max(_MIN_SUGGESTED_BATCH_SIZE, round(target_units_per_batch / units_per_task))
    min_batch_size_for_cap = math.ceil(total_tasks_all / max_array_subjobs)
    lo = max(_MIN_SUGGESTED_BATCH_SIZE, min_batch_size_for_cap)
    hi = total_tasks_all

    suggested_batch_size, suggested_n_batches, evenly_divides = _best_balanced_batch_size(
        total_tasks_all, raw_batch_size, lo, hi,
    )
    last_batch_size = total_tasks_all - (suggested_n_batches - 1) * suggested_batch_size

    if min_batch_size_for_cap > raw_batch_size:
        notes.append(
            f"Suggested batch size raised to at least {min_batch_size_for_cap} (from a "
            f"workload-only estimate of {raw_batch_size}) to keep the resulting job array at "
            f"or under Gadi's {max_array_subjobs}-subjob array cap for {total_tasks_all:,} "
            f"total task(s) - VERIFY this cap against NCI's current documentation."
        )

    if not evenly_divides:
        notes.append(
            f"{total_tasks_all:,} total task(s) does not divide evenly into any workable batch "
            f"size near the workload-sized target - the closest achievable balance is "
            f"{suggested_n_batches - 1} batch(es) of {suggested_batch_size} task(s) plus one "
            f"trailing batch of {last_batch_size} task(s) ({suggested_batch_size - last_batch_size} "
            f"fewer). If perfectly uniform batches matter for your queue accounting, a replicate/"
            f"ratio count that makes the total divide evenly would remove this entirely."
        )

    return {
        'suggested_batch_size': int(suggested_batch_size),
        'suggested_n_batches': int(suggested_n_batches),
        'last_batch_size': int(last_batch_size),
        'batch_size_evenly_divides': bool(evenly_divides),
        'notes': notes,
    }


def _base_of(model_name: str) -> str:
    """Strip a hyperparameter-tuning-algorithm suffix (``RF__Grid`` ->
    ``RF``) or a bio-prior instance suffix
    (``GAT_biological_prior_knowledge_2`` -> ``GAT_biological_prior_knowledge``)
    down to the base model name this table is keyed on. Kept as an
    independent, minimal re-implementation (rather than importing
    ``models.hyperparameter_tuning.base_of``) so this module stays a
    lightweight, dependency-free GUI/estimator helper importable without
    pulling in the full model stack.

    Note this collapses bio-prior INSTANCE numbers too (unlike
    ``models.hyperparameter_tuning.base_of``, which preserves them, since
    ``HPARAMETERS`` is keyed by the full instance name) - correct for a
    COST-TABLE lookup (one entry per model FAMILY), but a documented
    simplification when reused for atom grouping in
    ``_atoms_for_costing()`` below - see that function's own docstring."""
    base = model_name.split('__', 1)[0]
    if base.startswith('GAT_biological_prior_knowledge'):
        return 'GAT_biological_prior_knowledge'
    return base


def _profile_for(model_name: str) -> Dict[str, object]:
    return MODEL_COST_PROFILE.get(_base_of(model_name), _DEFAULT_MODEL_PROFILE)


def compute_total_tasks(
    scenario: str,
    n_population: int,
    n_phenotype: int,
    n_ratio: int,
    sample_num: int,
    w_opt_enabled: bool = False,
) -> int:
    """Total scenario count, computed purely from config (architecture doc
    §6) - zero data I/O. Mirrors ``genomic_prediction.py::GP()``'s own
    ``sample`` table construction closely enough for an estimate (the
    'between' population-pair/triple counting is an approximation when
    ``n_population`` is small, since GP() enforces all-distinct triples
    when a validation split/W_OPT is requested - see architecture doc §6);
    always a task-COUNT purely for resourcing purposes, never used to
    build the actual scenario list itself."""
    if scenario == 'between':
        if n_population < 2:
            return 0
        if w_opt_enabled and n_population >= 3:
            # X->Y->Z, all-distinct triples.
            return n_phenotype * n_population * (n_population - 1) * (n_population - 2)
        # X->Y pairs, self-pairs excluded.
        return n_phenotype * n_population * (n_population - 1)
    # 'within'
    return n_population * n_phenotype * n_ratio * sample_num


def compute_units_per_task(n_models: int, ld_prune_enabled: bool, rf_filter_enabled: bool) -> int:
    """Mirrors ``genomic_prediction.py::GP()``'s own progress-unit formula
    (architecture doc §6) - one unit per (task, model) pair, plus one each
    for LD pruning/RF filtering if enabled."""
    return n_models + (1 if ld_prune_enabled else 0) + (1 if rf_filter_enabled else 0)


def _mem_scale_factor(mem_scaling: str, n_markers: Optional[int], n_samples: Optional[int]) -> float:
    """Relative memory-scaling multiplier for one model, evaluated at the
    (cheaply-obtained) marker/sample counts - a dimensionless factor this
    module's own calibration constants are tuned against, not an absolute
    byte count."""
    markers = max(1, n_markers or 1)
    samples = max(1, n_samples or 1)
    if mem_scaling == 'markers_samples':
        return (markers * samples) / 1.0e6
    if mem_scaling == 'markers_squared':
        return (markers ** 2) / 1.0e7
    if mem_scaling == 'genes':
        # Independent of raw marker count by design (architecture §12.3) -
        # a small constant footprint per gene-window model instance.
        return 1.0
    # 'markers' (default)
    return markers / 1.0e4


# ==========================================================================
# Update ID 2 (R2, blueprint T16) - Phase 2 core, extracted VERBATIM.
#
# This function's BODY is character-for-character what Phase 2's own
# estimate_resources() used to be - same parameters, same logic, same
# eight-key return dict. Nothing below this point differs from the
# pre-Update-2 module. Extracting it (rather than adding an `if` branch
# inside the old function) is what makes "estimate_resources() with the
# flag off is byte-identical to Phase 2" a STRUCTURAL property (this
# function is simply called unmodified) rather than something that has to
# be separately tested for and could regress on a future edit - see
# blueprint §R2.2/rejected-alternative B1.
# ==========================================================================
def _estimate_resources_phase2_core(
    scenario: str,
    n_population: int,
    n_phenotype: int,
    n_ratio: int,
    sample_num: int,
    model_run: List[str],
    ld_prune_enabled: bool,
    rf_filter_enabled: bool,
    genotype_format: str,
    genotype_file_name: str,
    phenotype_file_name: Optional[str] = None,
    w_opt_enabled: bool = False,
    n_batches: int = 1,
) -> Dict[str, object]:
    """Cheap resource estimate for one EasiGP run (or, via ``n_batches``,
    for ONE batch of a Parallel run split across that many batches).

    Every input here is either taken straight from the config (zero I/O)
    or a cheap header/line-count read (``n_markers``/``n_samples`` via
    ``pipeline_utils.cheap_marker_and_sample_counts()``) - this function
    never loads a full genotype/phenotype matrix, and is expected to
    complete in well under a second for any realistic config.

    Returns
    -------
    Dict with: ``ncpus`` (int), ``mem_gb`` (float, rounded), ``ngpus``
    (int), ``est_relative_cost`` (float, unitless - for COMPARING configs
    against each other, not a wall-clock prediction), ``n_markers``,
    ``n_samples`` (the cheap counts actually used, or ``None`` if
    unavailable), and ``notes`` (list[str] of any caveats worth surfacing
    in the GUI, e.g. "marker count unavailable - using a conservative
    default").
    """
    notes: List[str] = []

    n_markers, n_samples = cheap_marker_and_sample_counts(
        genotype_format, genotype_file_name, phenotype_file_name,
    )
    if n_markers is None:
        notes.append(
            "Marker count unavailable (genotype file not found/unreadable yet) - "
            "using a conservative placeholder; re-check once the file path is set."
        )
        n_markers = 5000
    if n_samples is None:
        notes.append(
            "Sample count unavailable (genotype/phenotype file not found/unreadable "
            "yet) - using a conservative placeholder; re-check once the file path is set."
        )
        n_samples = 500

    total_tasks_all = compute_total_tasks(scenario, n_population, n_phenotype, n_ratio, sample_num, w_opt_enabled)
    n_batches = max(1, int(n_batches))
    total_tasks = max(0, math.ceil(total_tasks_all / n_batches))

    units_per_task = compute_units_per_task(len(model_run), ld_prune_enabled, rf_filter_enabled)
    total_units = total_tasks * units_per_task

    est_relative_cost = 0.0
    mem_units = 0.0
    any_gpu_capable = False
    for model_name in model_run:
        profile = _profile_for(model_name)
        est_relative_cost += total_tasks * float(profile['cpu_time_factor'])
        mem_units += _mem_scale_factor(str(profile['mem_scaling']), n_markers, n_samples) * float(profile['cpu_time_factor'])
        if profile['gpu_capable']:
            any_gpu_capable = True

    if not model_run:
        notes.append("No model selected yet - estimate reflects zero model-fitting cost.")

    # ncpus: scale with total_units (parallel model-fitting/tuning work per
    # task), rounded up to Gadi's standard node/queue increment, floored at
    # _MIN_NCPUS.
    raw_ncpus = max(_MIN_NCPUS, math.ceil(total_units / 25.0))
    ncpus = int(math.ceil(raw_ncpus / _GADI_NCPUS_INCREMENT) * _GADI_NCPUS_INCREMENT)
    ncpus = max(_MIN_NCPUS, ncpus)

    # mem_gb: base overhead + a mem_units-driven term, floored at _MIN_MEM_GB.
    mem_gb = _BASE_MEM_GB + mem_units / 50.0
    mem_gb = max(_MIN_MEM_GB, round(mem_gb, 1))

    ngpus = 1 if any_gpu_capable else 0

    return {
        'ncpus': ncpus,
        'mem_gb': mem_gb,
        'ngpus': ngpus,
        'est_relative_cost': round(est_relative_cost, 1),
        'n_markers': n_markers,
        'n_samples': n_samples,
        'total_tasks_per_batch': total_tasks,
        'total_tasks_all_batches': total_tasks_all,
        'notes': notes,
    }


# ==========================================================================
# Update ID 2 (R2) - pool-cost pricing, break-even predicate, worker-memory
# estimate, and the joint (task-level x model-level) allocator.
# ==========================================================================

def estimate_pool_cost(
    n_markers: Optional[int],
    n_samples: Optional[int],
    ld_prune_enabled: bool,
    rf_filter_enabled: bool,
    genotype_format: str,
) -> float:
    """Modelled Steps-1-8 ("pool construction" - architecture doc §7 Steps
    1-8: phenotype subsetting/dedup/split, marker-pool routing, LD
    pruning, RF importance filtering) cost for ONE task, in the same
    RF-relative units as ``MODEL_COST_PROFILE``'s ``fit_cost_factor`` -
    this is what R1's model-level fan-out pays ``G`` times per task
    (once per model-group) instead of once, and what
    ``should_fan_out_models()`` below weighs against the fit-time saved
    by fanning out.

    A small, deliberately simple model (D2: logic first) - a fixed base
    cost, a marker-count-proportional term (using the same order-of-
    magnitude scaling ``_mem_scale_factor``'s ``'markers'`` case uses, so
    a larger genotype is priced as a proportionally larger pool cost),
    plus a fixed additive term for each of LD pruning, RF filtering, and
    PLINK ``--extract``/``--keep`` materialisation (``genotype_format ==
    'plink'``) that is actually enabled for this run - each is a real,
    separately-toggleable amount of per-task preprocessing work.
    """
    markers = max(1, n_markers or 1)
    cost = _POOL_COST_BASE + markers / 1.0e4
    if ld_prune_enabled:
        cost += _POOL_COST_LD_PRUNE
    if rf_filter_enabled:
        cost += _POOL_COST_RF_FILTER
    if genotype_format == 'plink':
        cost += _POOL_COST_PLINK_EXTRACT
    return cost


def _apply_pool_cost_to_core(core: Dict[str, object], pool_cost: float, total_tasks: int) -> Dict[str, object]:
    """Update ID 3 (R4, blueprint §R4.5(2)) - fold the per-task Steps-1-8
    pool-construction cost (`estimate_pool_cost()`) into
    `_estimate_resources_phase2_core()`'s `ncpus`/`mem_gb`, WITHOUT calling
    or modifying that function itself - A2.2's flag-off byte-identity is a
    structural property of it being called unmodified (see the "Phase 2
    core, extracted VERBATIM" banner above `_estimate_resources_phase2_core`
    itself). Pure: returns a NEW dict; `core` is never mutated.

    Only the amount by which `pool_cost` exceeds `_POOL_COST_BASE` is
    added:
    - `ncpus`: the increment is treated as additional raw units on top of
      `core['ncpus']` (itself already `_GADI_NCPUS_INCREMENT`-quantised),
      run back through the SAME `ceil(.../25)` and
      `ceil(.../_GADI_NCPUS_INCREMENT)*_GADI_NCPUS_INCREMENT` rounding
      Phase 2's own core uses, so a zero increment is a true no-op
      (`ceil(core['ncpus']/_GADI_NCPUS_INCREMENT) == core['ncpus']` when
      `core['ncpus']` is already a multiple of it) and any positive
      increment can only push `ncpus` up, never down.
    - `mem_gb`: added directly in the same GB-space
      `_estimate_resources_phase2_core()`'s own `mem_gb = _BASE_MEM_GB +
      mem_units/50.0` already produces (equivalent to adding to
      `mem_units` then dividing by 50, since division distributes over
      addition), then the `_MIN_MEM_GB` floor is re-applied - a no-op
      whenever the floor was already cleared, since the increment is
      never negative.

    Note: `estimate_pool_cost()` always includes a marker-count-
    proportional term regardless of whether LD pruning/RF filtering are
    enabled (its own `_POOL_COST_BASE + markers/1.0e4 [+ ...]` shape), so
    "only the incremental cost of enabled steps" does not mean the
    increment is exactly zero with both filtering steps off - it means no
    step's own additive term (`_POOL_COST_LD_PRUNE`/`_POOL_COST_RF_FILTER`/
    `_POOL_COST_PLINK_EXTRACT`) is charged unless that step is actually
    enabled for this call. See the Update 3 change summary §7 for why this
    is implemented literally per the design blueprint's formula rather
    than reading it as "identically zero at base"."""
    total_tasks = max(0, int(total_tasks))
    increment = pool_cost - _POOL_COST_BASE

    extra_units = total_tasks * increment
    extra_raw_ncpus = math.ceil(extra_units / 25.0) if extra_units > 0 else 0
    ncpus = int(math.ceil((core['ncpus'] + extra_raw_ncpus) / _GADI_NCPUS_INCREMENT) * _GADI_NCPUS_INCREMENT)
    ncpus = max(_MIN_NCPUS, ncpus)

    mem_units_increment = _POOL_MEM_UNITS_PER_COST * increment * _mem_scale_factor(
        'markers', core.get('n_markers'), core.get('n_samples'))
    mem_gb = max(_MIN_MEM_GB, round(core['mem_gb'] + mem_units_increment / 50.0, 1))

    updated = dict(core)
    updated['ncpus'] = ncpus
    updated['mem_gb'] = mem_gb
    return updated


def estimate_post_filter_markers(
    n_markers: Optional[int],
    ld_prune_enabled: bool,
    rf_filter_enabled: bool,
    rf_filter_cfg: Optional[Mapping[str, object]] = None,
    ld_prune_estimated_n_markers: Optional[int] = None,
) -> Tuple[int, bool, List[str]]:
    """Additional Requirements 9, Requirement 4 - best-effort estimate of
    how many markers a NON-bio-prior model actually sees once LD pruning
    and/or RF importance filtering are applied, in the SAME order ``GP()``
    itself applies them (architecture doc §7, Steps 6-7: LD pruning first,
    RF filtering second, on whatever LD pruning left behind).

    RF filtering's own reduction is derivable EXACTLY from its config, with
    zero extra data I/O:
      - ``mode='count'``: the filtered set is exactly ``top_n`` markers
        (clamped to whatever came out of LD pruning first, since RF can
        never keep more markers than it was handed).
      - ``mode='percent'``: the filtered set is
        ``round(top_ratio * <markers RF actually received>)``.
    So RF filtering's contribution here is exact given its own config
    whenever it's enabled - this function never needs to ask the person
    anything for RF's own sake.

    LD pruning's reduction is NOT cheaply derivable - how many markers
    survive PLINK's ``--indep-pairwise`` depends on the real linkage-
    disequilibrium structure of the actual genotype data, which this
    advisor deliberately never reads in full (module docstring). When LD
    pruning is enabled, this function uses ``ld_prune_estimated_n_markers``
    if the person supplied one (GUI: 'Estimated markers remaining after
    LD pruning', inside 'Suggested compute resources'); otherwise it falls
    back to the RAW marker count (i.e. assumes LD pruning removes nothing)
    and reports that the estimate is uncertain, so the caller can caption
    it as a likely OVER-estimate of memory (and, for
    ``GAT_fully_connected`` specifically, a large one) rather than silently
    guessing a smaller number.

    Returns
    -------
    ``(estimated_markers, ld_estimate_uncertain, notes)`` - ``notes`` is a
    list of human-readable caveats (e.g. "no post-pruning estimate
    supplied") worth surfacing in the GUI. A completely disabled LD/RF
    pair (both `False`) returns ``(n_markers, False, [])`` unchanged - a
    true no-op, matching every other calibration helper in this module's
    own "only says something when it can actually change something"
    convention.
    """
    notes: List[str] = []
    current = max(1, int(n_markers or 1))
    ld_estimate_uncertain = False

    if ld_prune_enabled:
        if ld_prune_estimated_n_markers is not None and int(ld_prune_estimated_n_markers) > 0:
            current = min(current, int(ld_prune_estimated_n_markers))
        else:
            ld_estimate_uncertain = True
            notes.append(
                "LD pruning is enabled but no estimated post-pruning marker count was supplied - "
                "this estimate conservatively assumes LD pruning removes no markers, which likely "
                "OVER-estimates the memory (and, for GAT_fully_connected specifically, ncpus) "
                "actually needed once pruning runs. Supply an estimate for a tighter figure."
            )

    if rf_filter_enabled:
        if rf_filter_cfg:
            mode = rf_filter_cfg.get('mode', 'percent')
            if mode == 'count':
                top_n = int(rf_filter_cfg.get('top_n', current) or current)
                current = max(1, min(current, top_n))
            else:
                top_ratio = float(rf_filter_cfg.get('top_ratio', 1.0) or 1.0)
                top_ratio = min(1.0, max(0.0, top_ratio))
                current = max(1, round(current * top_ratio))
        else:
            notes.append(
                "RF marker filtering is enabled but its own percentage/count configuration was "
                "not supplied to this estimate, so its marker reduction could not be counted."
            )

    return int(current), ld_estimate_uncertain, notes


def _apply_marker_filter_calibration_to_core(
    core: Dict[str, object], model_run: Sequence[str],
    n_markers_raw: Optional[int], n_markers_effective: int,
) -> Dict[str, object]:
    """Additional Requirements 9, Requirement 4 - recompute each selected
    model's own ``mem_units`` contribution using ``n_markers_effective``
    (the post LD/RF-filter estimate - see ``estimate_post_filter_markers()``
    above) instead of the RAW marker count
    ``_estimate_resources_phase2_core()`` always used internally for every
    model regardless of filtering - and folds the DIFFERENCE into
    ``core['mem_gb']`` only. Pure: returns a NEW dict; `core` is never
    mutated - mirrors ``_apply_pool_cost_to_core()``'s own contract and its
    A2.2-preserving reasoning (``_estimate_resources_phase2_core()`` itself
    stays untouched).

    ``GAT_fully_connected``'s ``markers_squared`` mem-scaling is the single
    biggest source of the "memory suggestion becomes inaccurate once LD/RF
    filtering is selected" defect this requirement names: a 10x marker
    reduction from filtering is a 100x reduction in that ONE model's own
    memory term, invisible to any estimate that always prices every model
    against the pre-filter marker count.

    ``GAT_biological_prior_knowledge`` is architecturally EXEMPT from LD/RF
    filtering (architecture doc §8 - it always selects its own markers from
    its gene network and never sees LD-pruned or RF-filtered data) - its
    own contribution is therefore priced against `n_markers_raw` on BOTH
    sides of the difference (i.e. it never contributes to the delta this
    function applies), so filtering never incorrectly discounts it.

    A true no-op whenever ``n_markers_effective == n_markers_raw``
    (nothing to filter, or LD/RF both disabled/short-circuited) - not
    called at all by ``estimate_resources()`` in that case, matching every
    other calibration post-step's own "only run when it can change
    something" convention in this module.

    BUG FIX (GUI defect - "the suggested memory value does not change even
    after users type an estimated marker value"): the previous
    implementation added ``(mem_units_effective - mem_units_raw) / 50.0``
    onto ``core['mem_gb']`` and then re-applied ``max(_MIN_MEM_GB, ...)``.
    ``core['mem_gb']`` at that point is ``_estimate_resources_phase2_core()``'s
    own output, which is ALREADY floor-clamped
    (``max(_MIN_MEM_GB, round(_BASE_MEM_GB + mem_units/50.0, 1))``). Adding a
    (typically negative, since filtering only ever REDUCES markers) delta
    onto an already-clamped number and re-clamping the result silently
    discards the entire adjustment whenever the TRUE, unclamped baseline was
    at or below ``_MIN_MEM_GB`` - which is the common case for every model
    except ``GAT_fully_connected`` on a very large marker set, so the
    "Estimated markers remaining after LD pruning" field appeared to do
    nothing for most real configs even though ``n_markers_effective`` (and
    the "Memory sized for ..." caption) updated correctly.

    Fixed by reconstructing the UNCLAMPED effective ``mem_gb`` directly from
    ``mem_units_effective`` (the exact same ``_BASE_MEM_GB + mem_units/50.0``
    formula ``_estimate_resources_phase2_core()`` itself uses) and applying
    the ``_MIN_MEM_GB`` floor exactly ONCE, at the very end - never adding a
    delta on top of an already-floored value. ``mem_units_raw``/``model_run``
    are still computed the same way (including ``'ensemble'`` now, so this
    reconstruction matches ``_estimate_resources_phase2_core()``'s own
    accounting byte-for-byte whenever nothing is filtered) purely so the
    RAW side can be compared 1:1 against ``core['mem_gb']`` for callers that
    inspect the delta; only ``mem_units_effective`` actually drives the
    returned ``mem_gb``.
    """
    n_markers_raw = max(1, int(n_markers_raw or 1))
    n_markers_effective = max(1, int(n_markers_effective))
    if n_markers_effective == n_markers_raw:
        return core

    n_samples = core.get('n_samples')
    mem_units_raw = 0.0
    mem_units_effective = 0.0
    for model_name in model_run:
        profile = _profile_for(model_name)
        mem_scaling = str(profile['mem_scaling'])
        factor = float(profile['cpu_time_factor'])
        markers_for_effective = n_markers_raw if _base_of(model_name) == 'GAT_biological_prior_knowledge' \
            else n_markers_effective
        mem_units_raw += _mem_scale_factor(mem_scaling, n_markers_raw, n_samples) * factor
        mem_units_effective += _mem_scale_factor(mem_scaling, markers_for_effective, n_samples) * factor

    # Recompute mem_gb from scratch against the EFFECTIVE (post-filter)
    # marker count, using the same unfloored formula
    # _estimate_resources_phase2_core() itself uses, then floor exactly
    # once - see the BUG FIX note above for why this (rather than adding a
    # delta onto core['mem_gb']) is what actually makes the reduction
    # visible whenever it genuinely clears the floor.
    mem_gb_effective_unclamped = _BASE_MEM_GB + mem_units_effective / 50.0
    mem_gb = max(_MIN_MEM_GB, round(mem_gb_effective_unclamped, 1))

    updated = dict(core)
    updated['mem_gb'] = mem_gb
    return updated


def _apply_gpu_cpu_calibration_to_core(
    core: Dict[str, object], ld_prune_enabled: bool, rf_filter_enabled: bool,
    ncpus_per_gpu: int = _GADI_GPU_NCPUS_PER_GPU,
) -> Tuple[Dict[str, object], bool, int]:
    """Patch 3, Requirement 4 - the GPU-aware `ncpus` calibration described
    in this module's own constants block above. Pure: returns a NEW dict
    (`core` is never mutated), mirroring `_apply_pool_cost_to_core()`'s own
    contract and its A2.2-preserving reasoning (`_estimate_resources_
    phase2_core()` itself stays untouched - only what a wrapper does with
    its OUTPUT changes).

    A no-op (returns `core` unchanged, `applied=False`, `increment=0`)
    whenever `core['ngpus']` is 0 - a CPU-only estimate is completely
    unaffected by this function existing.

    When `core['ngpus'] > 0`:
    1. If LD pruning and/or RF filtering is ALSO enabled for this run,
       scale `ncpus` up by `_GPU_PREPROCESS_CPU_HEADROOM` (raw units, run
       back through the SAME ceil(.../25) rounding `_estimate_resources_
       phase2_core()`'s own `ncpus` computation uses, for consistency) -
       CPU-bound preprocessing needs real headroom alongside a
       GPU-resident model fit, not just any queue's bare minimum. This
       step is cluster-independent - it applies regardless of
       `ncpus_per_gpu`.
    2. Patch 3 v3, Requirement 3: IF `ncpus_per_gpu` is a known positive
       ratio (e.g. Gadi's gpuvolta queue: 12 CPUs per GPU - see
       `HPC_RESOURCE_PROFILES` above), raise `ncpus` to at least
       `ncpus_per_gpu * ngpus` and round UP to the nearest multiple of
       that floor (the strictest applicable constraint whenever a GPU is
       requested on a cluster with such a rule). Some HPCs (e.g. UQ
       Bunya - see `HPC_RESOURCE_PROFILES`) have no single fixed ratio at
       all; passing `ncpus_per_gpu=0` for those skips this floor/
       multiple-rounding step entirely and instead rounds to the generic
       `_GADI_NCPUS_INCREMENT`, same as a CPU-only estimate would.

    `ncpus_per_gpu` defaults to `_GADI_GPU_NCPUS_PER_GPU` (today's
    behaviour, unchanged) so every pre-Patch-3-v3 call site that never
    passes this argument gets an identical result to before.

    Returns (updated_core, applied, increment) - `applied` is True
    whenever this ran with ngpus > 0 (even if the increment happened to be
    zero, e.g. the base estimate already met the GPU floor); `increment`
    is `updated_core['ncpus'] - core['ncpus']` (>= 0 always - this
    function only ever pushes ncpus up, mirroring
    _apply_pool_cost_to_core()'s own guarantee)."""
    ngpus = int(core.get('ngpus') or 0)
    if ngpus <= 0:
        return core, False, 0

    base_ncpus = int(core['ncpus'])

    ncpus = base_ncpus
    if ld_prune_enabled or rf_filter_enabled:
        # Extra raw-unit headroom for CPU-bound LD/RF preprocessing
        # running alongside a GPU-resident model fit - run back through
        # the same ceil(.../25) unit-to-ncpus conversion the core
        # estimate's own `raw_ncpus = ceil(total_units / 25.0)` step used,
        # so this stays on the SAME cost scale rather than an
        # incommensurate ad hoc bump.
        extra_raw_ncpus = math.ceil(base_ncpus * (_GPU_PREPROCESS_CPU_HEADROOM - 1.0))
        ncpus = base_ncpus + extra_raw_ncpus

    if ncpus_per_gpu and ncpus_per_gpu > 0:
        gpu_floor = int(ncpus_per_gpu) * ngpus
        ncpus = max(ncpus, gpu_floor)
        # Round UP to a multiple of the per-GPU floor itself (the
        # strictest applicable constraint whenever a GPU is requested on
        # a cluster with a fixed ratio) - a plain _GADI_NCPUS_INCREMENT
        # (4) rounding would not, in general, land on a multiple of 12.
        ncpus = int(math.ceil(ncpus / gpu_floor) * gpu_floor)
    else:
        # Patch 3 v3, R3: no known fixed CPUs-per-GPU ratio for this
        # profile (e.g. UQ Bunya) - no floor to round up to, just the
        # same generic rounding a CPU-only estimate already uses.
        ncpus = int(math.ceil(ncpus / _GADI_NCPUS_INCREMENT) * _GADI_NCPUS_INCREMENT)
    ncpus = max(_MIN_NCPUS, ncpus)

    updated = dict(core)
    updated['ncpus'] = ncpus
    return updated, True, ncpus - base_ncpus


def should_fan_out_models(
    pool_cost: float,
    group_fit_costs: Sequence[float],
    margin: float = _FANOUT_BREAKEVEN_MARGIN,
) -> Tuple[bool, str]:
    """Shared break-even predicate: ``True`` iff the wall-clock time SAVED
    by fanning fits out across ``len(group_fit_costs)`` model-groups
    exceeds, by at least ``margin``, the REDUNDANT Steps-1-8 pool-rebuild
    cost that fan-out pays ``(G - 1)`` extra times (once per extra group,
    per task) - see the module docstring's cost-model summary and
    blueprint §R2.2.

    This EXACT function is imported and called, unmodified, by both this
    module's own joint allocator (``estimate_resources()`` below) and
    ``intra_task_parallel.task_is_model_fanout_eligible()`` (the runtime
    eligibility gate) - so the advisor's recommendation and the runtime's
    actual decision can never disagree for the same inputs (A2.6).

    Parameters
    ----------
    pool_cost : this task's modelled Steps-1-8 cost (``estimate_pool_cost()``).
    group_fit_costs : the summed ``fit_cost_factor`` (see
        ``MODEL_COST_PROFILE``) of each model-group's own members, for the
        grouping actually being evaluated - i.e. ``len(group_fit_costs)``
        IS ``G``, the number of groups.
    margin : required fractional safety margin (default
        ``_FANOUT_BREAKEVEN_MARGIN``) - see that constant's own docstring.

    Returns
    -------
    ``(recommended, reason)`` - ``reason`` is always a complete, human-
    readable sentence naming the actual numbers involved, suitable for
    both a console log line and a GUI caption.
    """
    group_fit_costs = list(group_fit_costs)
    n_groups = len(group_fit_costs)
    if n_groups < 2:
        return False, (
            f"only {n_groups} schedulable model group(s) after grouping - nothing to fan out across"
        )
    total_fit = sum(group_fit_costs)
    max_fit = max(group_fit_costs)
    saved = total_fit - max_fit
    redundant = margin * (n_groups - 1) * pool_cost
    if saved > redundant:
        return True, (
            f"fanning out across {n_groups} model group(s) saves an estimated {saved:.1f} "
            f"RF-relative unit(s) of serial fit time, which exceeds the {redundant:.1f}-unit "
            f"redundant Steps-1-8 pool-rebuild cost it costs to do so "
            f"({n_groups} groups x pool_cost={pool_cost:.1f}, margin={margin:.2f})"
        )
    return False, (
        f"fanning out across {n_groups} model group(s) would only save an estimated {saved:.1f} "
        f"RF-relative unit(s) of serial fit time, not enough to justify the {redundant:.1f}-unit "
        f"redundant Steps-1-8 pool-rebuild cost it costs to do so "
        f"({n_groups} groups x pool_cost={pool_cost:.1f}, margin={margin:.2f})"
    )


def estimate_worker_memory(model_names: Sequence[str]) -> float:
    """Estimated peak resident memory (GB) of ONE worker process
    dispatching `model_names` (either a single model-group's members - see
    `intra_task_parallel.build_model_groups()` - or, for a plain TASK-
    level-only worker with no model-level fan-out, the run's FULL
    `MODEL_RUN`, since that one worker fits every selected model
    sequentially for its own task - see `suggest_compute_layout()`'s own
    use of this function).

    Members of a group/task are still fit ONE AT A TIME inside that
    worker - R1 only parallelises ACROSS model-groups, never within one
    (architecture doc §7 Step 9's ``for jj in MODEL_RUN`` loop is
    otherwise untouched inside a single worker).

    Additional Requirements 9 (production defect - see the incident
    write-up beside `intra_batch_parallel._systemic_failure_note()`, a
    real N_CPU_WORKERS=20 batch OOM-killed by exactly this
    under-estimate): the four R/BGLR models (`_R_ACCUMULATING_MODELS`
    below) all `source()` into, and MCMC-fit inside, the SAME persistent
    rpy2/R global environment for the entire life of one `GP()` call
    (architecture doc §9.1) - R's own garbage collector does not reliably
    return memory to the OS between successive models run in that SAME
    session, so a worker fitting e.g. rrBLUP THEN BayesB THEN RKHS (all
    selected, all in the same task) tends toward the SUM of their own
    peaks, not just whichever peaks highest alone. Every other model
    family (RF/SVR/KNN/MLP/GAT variants) runs in ordinary, independently
    garbage-collected Python objects between calls, so a plain MAX across
    those remains the appropriate (not over-conservative) approximation -
    only the R/BGLR subset is summed, plus the MAX of everything else,
    plus the fixed per-process overhead (`_BASE_MEM_GB`) every EasiGP
    worker pays regardless of what it fits."""
    if not model_names:
        return _BASE_MEM_GB
    r_peaks: List[float] = []
    other_peaks: List[float] = []
    for name in model_names:
        peak = float(_profile_for(name).get('peak_mem_gb_per_worker', _DEFAULT_MODEL_PROFILE['peak_mem_gb_per_worker']))
        if _base_of(name) in _R_ACCUMULATING_MODELS:
            r_peaks.append(peak)
        else:
            other_peaks.append(peak)
    accumulated = sum(r_peaks) + (max(other_peaks) if other_peaks else 0.0)
    return _BASE_MEM_GB + accumulated


def pack_atoms_lpt(atom_costs: Sequence[float], n_bins: int) -> List[List[int]]:
    """Longest-Processing-Time-first greedy bin-packing: assigns atom
    INDICES (sorted by cost descending, ties broken by lowest index) one
    at a time to whichever bin currently has the LOWEST total cost so far
    (ties broken toward the lowest bin index) - the same ``'cost_balanced'``
    algorithm `intra_task_parallel.build_model_groups()` uses for its own,
    authoritative runtime grouping (blueprint §R1.6). Defined here (not
    duplicated in `intra_task_parallel.py`) and imported from there, so
    the advisor's cost model and the runtime's actual packing can never
    silently diverge (A2.6) - see that module's own import of this
    function.

    Returns a list of ``min(n_bins, len(atom_costs))`` bins, each a list
    of ORIGINAL indices into `atom_costs` assigned to that bin, in
    assignment order. An empty `atom_costs` returns ``[]``."""
    if not atom_costs:
        return []
    n_bins = max(1, min(int(n_bins), len(atom_costs)))
    order = sorted(range(len(atom_costs)), key=lambda idx: (-atom_costs[idx], idx))
    bins: List[List[int]] = [[] for _ in range(n_bins)]
    bin_totals = [0.0] * n_bins
    for atom_idx in order:
        target = min(range(n_bins), key=lambda b: (bin_totals[b], b))
        bins[target].append(atom_idx)
        bin_totals[target] += atom_costs[atom_idx]
    return bins


def _atoms_for_costing(
    model_run: Sequence[str], hp_tune: Optional[Mapping[str, Mapping]] = None,
) -> Tuple[List[List[str]], List[float]]:
    """A lightweight, COST-ESTIMATION-ONLY atomisation of `model_run`
    (base model names, e.g. `cfg['MODEL']` - see the module-level note on
    why this advisor works with base names, matching Phase 2's own
    existing `compute_units_per_task()` precedent) into indivisible
    groups, together with each atom's total `fit_cost_factor` weight -
    APPROXIMATING (not replicating exactly) `intra_task_parallel.
    build_model_groups()`'s real C1/C2 grouping constraints closely
    enough to price the joint allocator's trade-off.

    This is NOT the authoritative grouping - that lives in
    `intra_task_parallel.build_model_groups()`, which this module is
    never imported by and never imports (this module's own established
    "lightweight, dependency-free" design goal - see `_base_of()`'s own
    docstring). Grouping here uses THIS module's `_base_of()`, which
    collapses BOTH a hyperparameter-tuning-algorithm suffix (matching the
    real C1 rule exactly) AND a bio-prior instance suffix (an
    approximation of the real C2 rule - the real rule only merges
    MERGE-ENABLED bio-prior instances, which this advisor has no cheap
    way to check without reading `HPARAMETERS[instance][13]` for every
    instance up front). The only practical effect of this simplification
    is that separate bio-prior instances are always treated as co-
    resident here, which happens to match the `BIO_PRIOR_GROUPING=
    'affinity'` DEFAULT anyway - and otherwise this can only ever produce
    FEWER, COARSER atoms than the real grouping would, never more/finer
    ones, so this advisor can under-estimate achievable
    `n_model_workers` but never over-estimate it. That is the safe
    direction for a resource-REQUEST estimator to err in: an over-
    generous ncpus/mem_gb suggestion risks scheduler queuing or an OOM;
    an under-generous `n_model_workers` suggestion merely leaves some
    achievable parallelism unclaimed, which the person can still raise
    by hand.

    `hp_tune` (if given) scales an atom's weight by however many tuning
    algorithms are configured for its member (a multi-algorithm-tuned
    base model expands into one `MODEL_RUN` entry per algorithm -
    architecture doc §9.4 - each an independent fit the SAME worker must
    still run, one at a time, before that group is 'done' for a task) -
    without attempting to reconstruct the literal `'RF__Grid'`-style
    expanded names themselves, which would require importing
    `models.hyperparameter_tuning.expand_model_list`.

    Returns ``(atoms, atom_costs)`` - parallel lists; ``atoms[i]`` is the
    list of base model names in atom ``i``, ``atom_costs[i]`` is that
    atom's total weight. `'ensemble'` is excluded (it is a finalisation
    step over already-fitted models, not itself a dispatched fit)."""
    order: List[str] = []
    members_by_key: Dict[str, List[str]] = {}
    for name in model_run:
        if name == 'ensemble':
            continue
        key = _base_of(name)
        if key not in members_by_key:
            members_by_key[key] = []
            order.append(key)
        members_by_key[key].append(name)

    atoms: List[List[str]] = []
    atom_costs: List[float] = []
    for key in order:
        members = members_by_key[key]
        weight = 0.0
        for member in members:
            profile = _profile_for(member)
            factor = float(profile.get('fit_cost_factor', profile['cpu_time_factor']))
            n_algorithms = 1
            if hp_tune:
                tune_cfg = hp_tune.get(member) or hp_tune.get(key) or {}
                algorithms = tune_cfg.get('algorithms') if tune_cfg.get('enabled') else None
                if algorithms:
                    n_algorithms = max(1, len(algorithms))
            weight += factor * n_algorithms
        atoms.append(members)
        atom_costs.append(weight)
    return atoms, atom_costs


# Additional Requirements 9, Requirements 2/3 - which model families can
# ever put a CPU `n_jobs` knob to use SOMEWHERE in their own per-task
# work, and (crucially, per this requirement's own follow-up) WHERE that
# use actually comes from - because it changes which lever
# (N_CPU_WORKERS_TASK vs N_JOBS) is actually the efficient one:
#
# - RF (models/RF.py): its own fit (`RandomForestRegressor(...,
#   n_jobs=...)`) is scikit-learn's OWN joblib-backed, true multi-process
#   tree-building parallelism - n_jobs helps regardless of whether
#   interaction search ("Return marker effect for interactions?") is on.
#   When it IS on, the interaction step's own forest REFIT also honours
#   n_jobs; the SHAP computation itself (`shap.TreeExplainer.
#   shap_interaction_values()`) is a fast, exact, C-level algorithm that
#   is NOT further parallelised by n_jobs. ver4-4 R3.h: that interaction
#   SHAP step is now ALSO fanned out row-wise across n_jobs PROCESSES via
#   `pipeline_utils.parallel_shap_values(interaction=True)` (previously
#   single-threaded), so this model's "strong" tier below is stronger
#   than pre-ver4-4.
# - KNN (models/KNN.py): its own fit (neighbour search) honours n_jobs.
#   When "Return marker effect?" is on, its Shapley/effect computation
#   ALSO uses n_jobs - via `pipeline_utils.parallel_shap_values()` (R3.h:
#   promoted from a local, byte-for-byte-duplicated `_parallel_kernel_
#   shap_values()` in SVR.py/KNN.py), fanning the model-agnostic
#   `shap.KernelExplainer` out one explained row at a time across true
#   worker PROCESSES (not GIL-bound threads). Both files' own comments
#   describe this Shapley step as "the dominant cost" for these two
#   models once enabled - i.e. for KNN specifically, MOST of n_jobs'
#   actual benefit comes from Shapley, not the (comparatively cheap)
#   neighbour-search fit.
# - SVR (models/SVR.py): the OPPOSITE split from KNN - its own fit
#   (libsvm) is NOT internally multi-threaded at all (see that module's
#   own top-of-file comment) and gets ZERO benefit from n_jobs. Its
#   Shapley/effect computation, when "Return marker effect?" is on, uses
#   the exact SAME `pipeline_utils.parallel_shap_values()` fan-out as KNN
#   above - so SVR benefits from n_jobs ONLY when that toggle is on, and
#   not at all otherwise. (SVR.py's own comments describe this step,
#   unparallelised, turning an 8,000-marker run into 14+ days - i.e. when
#   it IS enabled, it typically dominates SVR's entire per-task cost.)
# - GBLUP/RKHS (models/GBLUP.R, models/RKHS.R - ver4-4 R3.d, NEW this
#   update): when "Return marker effect?" is on AND n_jobs names more
#   than one usable worker AND at least 2 test rows are being explained,
#   genomic_prediction.py::_run_gblup_or_rkhs() splits the per-row
#   Shapley loop across n_jobs worker PROCESSES (never threads - I10;
#   each worker bootstraps its own independent embedded R session - see
#   that function's own module-level design note), typically the
#   dominant per-task cost once enabled (the same O(2*M*Shapley_num)
#   loop architecture doc §9.1 already documents as this family's
#   dominant cost). Off (the pre-ver4-4-equivalent default for this
#   toggle), or n_jobs<=1, falls straight back to today's exact unsplit
#   call - so these two now belong in the SAME toggle-gated shape as
#   RF/SVR/KNN above, not in the "no benefit at all" bucket they were in
#   before ver4-4 R3.d existed.
# - rrBLUP/BayesB (R/BGLR MCMC, using R_BLAS_THREADS instead of N_JOBS -
#   NEITHER has a per-task Shapley/effect fan-out at all: both return
#   their own marker-effect table UNCONDITIONALLY, with no user-facing
#   "return effect" toggle to fan out in the first place, and R3.d's own
#   scope was GBLUP/RKHS specifically - see hparam_specs.py, which only
#   appended the two new shap_row_offset/shap_row_count fields to those
#   two models, not rrBLUP/BayesB) - still genuinely get NO benefit from
#   N_JOBS.
# - GAT_prior_knowledge (models/GAT_prior_knowledge.py - ver4-4 R3.a/R3.h,
#   NEW this update): its own internal `RandomForestRegressor` fit (which
#   defines this model's OWN graph topology, not a diagnostic side-
#   output) now honours n_jobs (R3.a - previously single-threaded, no
#   `n_jobs` argument at all), and the interaction SHAP step immediately
#   after it is now ALSO fanned out via `parallel_shap_values(interaction=
#   True)` (R3.h). BOTH of these run UNCONDITIONALLY, every call,
#   regardless of this model's OWN "Return marker effects?" toggle
#   (HPARAM_SPECS index 8) - that toggle instead gates a SEPARATE,
#   GPU-dispatched/gradient-based explainability step
#   (`Explainer`/`CaptumExplainer`) that n_jobs never touches at all. So
#   unlike RF/SVR/KNN/GBLUP/RKHS above, this model's N_JOBS benefit is
#   NOT gated by its own DIAGNOSTIC_FLAG_FIELDS toggle - it is always
#   present whenever N_JOBS>1, but (being one contribution to an overall
#   GPU-dominated per-task cost, not the model's own dominant cost the
#   way RF/KNN/SVR's Shapley step usually is) is bucketed as a
#   consistent, always-on "weak" benefit below, never "strong".
# - Every other model (MLP and the four remaining GAT variants -
#   GPU-dispatched, and even their own CPU-fallback explainers
#   (`shap.DeepExplainer`/`CaptumExplainer`) are gradient/backprop-based,
#   not joblib-parallelised) gets no benefit from N_JOBS at all.
#
# `_TOGGLE_GATED_N_JOBS_MODELS` is the RF/KNN/SVR/GBLUP/RKHS union above -
# every member also present in `DIAGNOSTIC_FLAG_FIELDS`, whose own boolean
# toggle genuinely gates whether N_JOBS does anything for that model this
# run. `_ALWAYS_N_JOBS_MODELS` is GAT_prior_knowledge alone - its own
# N_JOBS benefit is unconditional, not toggle-gated (see above).
# `_N_JOBS_CAPABLE_MODELS` (the union of both) is kept as a single
# membership set for any caller that only needs "does N_JOBS ever help
# this model at all", without needing the toggle-gated/always-on
# distinction itself. `_shapley_enabled()` further BELOW refines the
# toggle-gated set per-run, since SVR (and, to a lesser extent given its
# cheaper fit, KNN) only actually realise that benefit when their own
# Shapley toggle is on.
# Update ID ver4-5, R2 Stages 8-11: ExtraTrees/GBDT/XGBoost gain their own
# n_jobs-capable interaction step exactly like RF's (TreeSHAP row fan-out,
# toggle-gated by their own get_interaction field); rrBLUP/BayesB gain one
# too, via their new surrogate-interaction step (also toggle-gated, and
# genuinely the MOST valuable n_jobs benefit in this whole set - see
# DIAGNOSTIC_FLAG_FIELDS's own note on why forcing it off during tuning
# matters so much: an unfiltered surrogate RandomForestRegressor fit is
# real work). EBM's OWN n_jobs usage (its own `outer_bags` bagging) is
# UNCONDITIONAL - not gated by its interaction toggle at all, since outer
# bagging happens regardless of whether pairwise terms are searched for -
# so it joins GAT_prior_knowledge in `_ALWAYS_N_JOBS_MODELS` instead of
# the toggle-gated set below.
_TOGGLE_GATED_N_JOBS_MODELS = frozenset(
    m for m in ('RF', 'KNN', 'SVR', 'GBLUP', 'RKHS', 'ExtraTrees', 'GBDT', 'XGBoost', 'rrBLUP', 'BayesB')
    if m in DIAGNOSTIC_FLAG_FIELDS
)
_ALWAYS_N_JOBS_MODELS = frozenset({'GAT_prior_knowledge', 'EBM'})
_N_JOBS_CAPABLE_MODELS = _TOGGLE_GATED_N_JOBS_MODELS | _ALWAYS_N_JOBS_MODELS

# Positional index (within HPARAMETERS[<model>], the SAME flat positional
# list architecture doc §4.4 describes) of each toggle-gated model's own
# "compute Shapley/interaction/marker-effect scores?" toggle - read
# directly from `models.hyperparameter_tuning.DIAGNOSTIC_FLAG_FIELDS`
# (the single source of truth this same field index already has - see the
# import comment above) rather than a second, hand-maintained copy. Used
# ONLY to caption whether N_JOBS>1 will actually be exercised for THIS
# run's own hyperparameters - never to change the numeric ncpus/
# n_cpu_workers_task/n_jobs recommendation itself (a person may still
# want the headroom for a toggle they haven't enabled yet).
# Update (post-ver4-5 R2 fix): DIAGNOSTIC_FLAG_FIELDS entries may now be a
# single int OR a tuple of ints (models.hyperparameter_tuning's own
# _disable_explainability() already normalises this the same way - see its
# comment). Everywhere in this module that only wants ONE representative
# "own Shapley/marker-effect toggle" index, we take the FIRST element of
# the tuple - by construction (verified against hparam_specs.py) that
# first element is always this model's own "Return marker effect?" flag
# (e.g. SVR (6, 11) -> 6, KNN (3, 8) -> 3, RKHS (3, 10) -> 3, GBLUP (2, 9)
# -> 2); the second element, where present, gates a separate marker-PAIR
# interaction flag this module doesn't track.
def _primary_diagnostic_flag_index(idx: Union[int, tuple]) -> int:
    return idx[0] if isinstance(idx, tuple) else idx


_SHAPLEY_TOGGLE_INDEX: Dict[str, int] = {
    m: _primary_diagnostic_flag_index(DIAGNOSTIC_FLAG_FIELDS[m]) for m in _TOGGLE_GATED_N_JOBS_MODELS
}
# hparam_specs.py's own 'default' for that same field, read directly out
# of HPARAM_SPECS - RF's interaction search defaults ON; SVR/KNN's own
# Shapley effect computation and GBLUP/RKHS's own "Return marker effect?"
# all default OFF (hparam_specs.py, verified directly against each
# model's own field list).
_SHAPLEY_TOGGLE_DEFAULT: Dict[str, bool] = {
    m: bool(HPARAM_SPECS[m][idx]['default']) for m, idx in _SHAPLEY_TOGGLE_INDEX.items()
}


def _shapley_enabled(model_base: str, hparameters: Optional[Mapping[str, Sequence]]) -> Optional[bool]:
    """Whether `model_base`'s own Shapley/interaction toggle is enabled,
    read positionally out of `hparameters` (a run's `HPARAMETERS` dict)
    per `_SHAPLEY_TOGGLE_INDEX` above. Returns
    `_SHAPLEY_TOGGLE_DEFAULT[model_base]` whenever `hparameters` is
    missing, doesn't have an entry for this model, or that entry is
    shorter than expected (never raises) - and `None` for any model this
    concept doesn't apply to at all (only RF/SVR/KNN are in
    `_SHAPLEY_TOGGLE_INDEX`)."""
    if model_base not in _SHAPLEY_TOGGLE_INDEX:
        return None
    idx = _SHAPLEY_TOGGLE_INDEX[model_base]
    default = _SHAPLEY_TOGGLE_DEFAULT[model_base]
    if not hparameters:
        return default
    params = hparameters.get(model_base)
    if not isinstance(params, (list, tuple)) or len(params) <= idx:
        return default
    try:
        return bool(params[idx])
    except Exception:
        return default


def suggest_compute_layout(
    total_tasks_per_batch: int,
    model_run: Sequence[str],
    budget_ncpus: Optional[int] = None,
    hparameters: Optional[Mapping[str, Sequence]] = None,
) -> Dict[str, object]:
    """ver4-4 Stage 6 (RK-6, blueprint §7 risk register / §10 Stage 6
    checklist): **UNREFERENCED as of ver4-4** - confirmed by direct
    ``grep`` across the whole tree, this function has no caller anywhere
    outside this module (`main_app.py`'s own "joint-allocation" advisor
    panel, the only UI that ever called it, was deleted in Stage 2 - R2,
    blueprint §2.2.2 - which pinned every value that panel used to derive
    from this function's own output to fixed, non-coordinated defaults
    instead). Kept in the tree, updated rather than deleted, because its
    logic (and the model-capability tiers just above it) may still be
    useful reference material or a future re-integration point - but no
    code path in this codebase currently reaches it, and it should NOT be
    assumed live when reasoning about what a config/run actually does.
    The `_N_JOBS_CAPABLE_MODELS`/`_TOGGLE_GATED_N_JOBS_MODELS`/
    `_ALWAYS_N_JOBS_MODELS`/`_SHAPLEY_TOGGLE_INDEX`/`_SHAPLEY_TOGGLE_
    DEFAULT` module-level tables just above, and `_shapley_enabled()`,
    exist SOLELY to support this one function and share its same
    unreferenced status.

    Additional Requirements 9, Requirement 2 - a COORDINATED
    ``(n_cpu_workers_task, n_jobs, ncpus)`` triple for ONE batch of a
    KNOWN size (``total_tasks_per_batch`` - ideally
    ``suggest_array_layout()``'s own ``suggested_batch_size``, so the two
    suggestions describe the SAME batch shape rather than being computed
    independently and potentially disagreeing).

    Previously nothing connected the array-size suggestion to the
    ncpus/``N_CPU_WORKERS_TASK``/``N_JOBS`` suggestion at all - a person
    could apply a small suggested batch size together with a
    separately-derived ncpus figure and end up with either configured
    task-level workers that outnumber the batch's own tasks (wasted,
    permanently-idle worker processes - Requirement 2's "only five of them
    will be used for the second round" example) or a leftover CPU
    remainder that ``N_JOBS`` could not evenly divide either. This
    function instead derives all three together, FOR THE SAME
    ``total_tasks_per_batch``, and always reports ``ncpus`` as EXACTLY
    ``n_cpu_workers_task * n_jobs`` - never a number either level could not
    actually put to work.

    Calibration (Requirement 3's investigation - see the module docstring's
    "Update ID 4" banner for the reasoning in full): task-level width
    (``n_cpu_workers_task``, true independent OS processes - the
    architecture this codebase's own per-task checkpointing/fail-forward
    design already assumes, and the ONE lever that scales across
    INDEPENDENT node allocations rather than one node's shared memory
    bandwidth/cache) is grown FIRST, up to ``min(total_tasks_per_batch,
    budget_ncpus)``. Only once every task in the batch already has its own
    concurrent worker does any REMAINING CPU budget get spent on
    ``n_jobs`` (per-worker library-internal threading) instead.

    This ordering is deliberately kept as the general-purpose default even
    though it is NOT uniformly the most efficient choice - see
    ``_N_JOBS_CAPABLE_MODELS``'s own comment above for the full
    investigation. In short: for RF/KNN/SVR specifically, a large share of
    ``N_JOBS``'s benefit comes from their SHAPLEY/marker-effect
    computation (``shap.KernelExplainer`` for SVR/KNN, the interaction
    forest refit for RF) when that output is actually requested - and,
    unlike scikit-learn's other internal threading, that Shapley fan-out
    (``_parallel_kernel_shap_values()`` in SVR.py/KNN.py) is TRUE joblib
    multiprocessing, not GIL-bound threads, and both files' own comments
    describe it as often the DOMINANT per-task cost once enabled. So when
    every selected model is one of RF/KNN/SVR AND has its own Shapley/
    interaction toggle on, ``N_JOBS`` is a genuinely strong lever, not
    merely "whatever's left over" - this function's returned ``notes``
    say so explicitly (via ``hparameters``, when supplied) rather than
    always defaulting to "prefer task-level width" advice that would
    under-sell N_JOBS for exactly this common case. The task-level-first
    ORDER itself is unchanged regardless (task-level width is still at
    least as effective for every model family, including the R/BGLR/GPU
    ones N_JOBS never helps at all) - only the framing of "why any
    leftover budget went to N_JOBS" becomes model-aware.

    Parameters
    ----------
    total_tasks_per_batch : this batch's own task count (NOT the whole
        run's total - see ``suggest_array_layout()``'s
        ``suggested_batch_size``/``last_batch_size``).
    model_run : the run's selected ``MODEL_RUN`` list - used only to
        caption whether ``n_jobs > 1`` will actually be exercised by the
        selected roster, never to change the numeric recommendation
        itself (a person may still want ``N_JOBS`` headroom for a model
        added later).
    budget_ncpus : total CPUs available to this ONE batch. ``None`` falls
        back to ``_MIN_NCPUS`` (1) - i.e. "task-level width only, no
        internal threading" - callers normally pass the run's own
        Phase-2 ``ncpus`` headline (or a custom budget) here instead.
    hparameters : the run's own ``HPARAMETERS`` dict, if available -
        refines the RF/SVR/KNN captions above using each model's own
        actual Shapley/interaction toggle (``_shapley_enabled()``) rather
        than only its family membership. Entirely optional: omitting it
        falls back to each model's own ``hparam_specs.py`` default.

    Returns
    -------
    Dict with ``n_cpu_workers_task`` (int), ``n_jobs`` (int), ``ncpus``
    (int, ``== n_cpu_workers_task * n_jobs`` always), ``rounds_per_batch``
    (int - ``ceil(total_tasks_per_batch / n_cpu_workers_task)``; ``1``
    means every task in the batch runs in a single concurrent round, i.e.
    no worker ever sits idle through a smaller trailing round), and
    ``notes`` (list[str]).
    """
    total_tasks_per_batch = max(1, int(total_tasks_per_batch))
    budget = max(1, int(budget_ncpus)) if budget_ncpus else _MIN_NCPUS

    notes: List[str] = []

    n_cpu_workers_task = max(1, min(total_tasks_per_batch, budget))
    n_jobs = max(1, budget // n_cpu_workers_task)
    ncpus = n_cpu_workers_task * n_jobs

    rounds_per_batch = math.ceil(total_tasks_per_batch / n_cpu_workers_task)
    if rounds_per_batch > 1 and total_tasks_per_batch % n_cpu_workers_task != 0:
        last_round_workers = total_tasks_per_batch - (rounds_per_batch - 1) * n_cpu_workers_task
        notes.append(
            f"{total_tasks_per_batch} task(s)/batch across {n_cpu_workers_task} task-level "
            f"worker(s) means {rounds_per_batch} concurrent round(s), and the LAST round only "
            f"uses {last_round_workers} of those {n_cpu_workers_task} worker(s) - a batch size "
            f"that is an exact multiple of the worker count (see 'Suggested job-array size' "
            f"above) avoids this."
        )

    used_bases = {_base_of(m) for m in model_run if m != 'ensemble'}
    if n_jobs > 1:
        # Additional Requirements 9 - bucket every selected model into
        # exactly one of three tiers of N_JOBS benefit, using each model's
        # OWN Shapley/interaction toggle (not just its family) where that
        # matters - see the `_N_JOBS_CAPABLE_MODELS` comment above for the
        # full reasoning behind each tier:
        #   'strong' - a toggle-gated model (RF/KNN/SVR/GBLUP/RKHS) with
        #       its own Shapley/interaction/marker-effect output ON: that
        #       computation is TRUE joblib multiprocessing and typically
        #       dominates the per-task cost once enabled.
        #   'weak'   - either (a) a toggle-gated model (RF/KNN/GBLUP/RKHS)
        #       with that output OFF, which still benefits from N_JOBS via
        #       the model FIT itself alone (RF/KNN) or gets nothing extra
        #       beyond the toggle-gated fan-out (GBLUP/RKHS - their own
        #       MCMC fit itself uses R_BLAS_THREADS, not N_JOBS), or (b)
        #       GAT_prior_knowledge, whose own graph-topology-building
        #       N_JOBS benefit is unconditional but not this model's
        #       dominant per-task cost (see `_ALWAYS_N_JOBS_MODELS`'s own
        #       comment above).
        #   'none'   - rrBLUP/BayesB (R_BLAS_THREADS instead; no per-task
        #       Shapley/effect fan-out exists for either), every other
        #       GPU-dispatched model (GPU-bound), and SVR specifically
        #       with Shapley output OFF (its fit alone is single-threaded,
        #       so with the toggle off it gets nothing from N_JOBS at all).
        strong, weak, none_bucket = [], [], []
        for base in sorted(used_bases):
            if base in _ALWAYS_N_JOBS_MODELS:
                weak.append(base)
            elif base in _TOGGLE_GATED_N_JOBS_MODELS:
                if _shapley_enabled(base, hparameters):
                    strong.append(base)
                elif base == 'SVR':
                    none_bucket.append(base)
                elif base in ('GBLUP', 'RKHS'):
                    # ver4-4 R3.d: unlike RF/KNN (whose own model FIT
                    # itself already honours n_jobs regardless of this
                    # toggle), GBLUP/RKHS's own MCMC fit uses
                    # R_BLAS_THREADS, not N_JOBS - with the Shapley toggle
                    # off, N_JOBS genuinely does nothing for these two,
                    # exactly like SVR with its own toggle off.
                    none_bucket.append(base)
                else:
                    weak.append(base)
            else:
                none_bucket.append(base)

        if none_bucket and not strong and not weak:
            reasons = []
            if any(b not in ('SVR', 'GBLUP', 'RKHS') for b in none_bucket):
                reasons.append(
                    "rrBLUP/BayesB use R_BLAS_THREADS instead (neither has a per-task Shapley/"
                    "effect fan-out to parallelise); GPU-dispatched models are GPU-, not "
                    "CPU-thread-, bound"
                )
            if any(b in ('SVR', 'GBLUP', 'RKHS') for b in none_bucket):
                reasons.append(
                    "SVR/GBLUP/RKHS's own model fit isn't internally multi-threaded and each "
                    "one's own Shapley/marker-effect output is off"
                )
            notes.append(
                f"N_JOBS={n_jobs}, but none of your selected model(s) "
                f"({', '.join(none_bucket)}) benefit from it at all ({'; '.join(reasons)}) - "
                f"those {n_jobs - 1} extra CPU(s)/worker will sit unused. Consider "
                f"N_CPU_WORKERS_TASK={budget} (task-level width only, N_JOBS=1) instead, so "
                f"every reserved CPU is claimed by an independent worker process."
            )
        else:
            parts = []
            if strong:
                parts.append(
                    f"{', '.join(strong)} (Shapley/interaction/marker-effect output on - N_JOBS "
                    f"is a strong lever here via true joblib multiprocessing, often the dominant "
                    f"per-task cost once enabled)"
                )
            if weak:
                parts.append(
                    f"{', '.join(weak)} (N_JOBS still helps here - either the model fit itself, "
                    f"or an always-on internal step - just less than the 'strong' models above)"
                )
            if none_bucket:
                parts.append(f"{', '.join(none_bucket)} (no N_JOBS benefit at all)")
            notes.append(f"N_JOBS={n_jobs} benefit by model: " + "; ".join(parts) + ".")

    if budget_ncpus and ncpus < int(budget_ncpus):
        notes.append(
            f"{int(budget_ncpus) - ncpus} of the {int(budget_ncpus)} budgeted CPU(s) are not "
            f"used by this recommendation ({n_cpu_workers_task} task-worker(s) \u00d7 {n_jobs} "
            f"N_JOBS = {ncpus}) - request {ncpus} instead, or increase this batch's own task "
            f"count so there is more task-level work to spread the extra CPU(s) across."
        )

    return {
        'n_cpu_workers_task': int(n_cpu_workers_task),
        'n_jobs': int(n_jobs),
        'ncpus': int(ncpus),
        'rounds_per_batch': int(rounds_per_batch),
        'notes': notes,
    }


def estimate_resources(
    scenario: str,
    n_population: int,
    n_phenotype: int,
    n_ratio: int,
    sample_num: int,
    model_run: List[str],
    ld_prune_enabled: bool,
    rf_filter_enabled: bool,
    genotype_format: str,
    genotype_file_name: str,
    phenotype_file_name: Optional[str] = None,
    w_opt_enabled: bool = False,
    n_batches: int = 1,
    *,
    model_parallel_enabled: bool = False,
    budget_ncpus: Optional[int] = None,
    budget_ngpus: Optional[int] = None,
    budget_mem_gb: Optional[float] = None,
    gpu_slots_per_device: int = _GPU_SLOTS_PER_DEVICE_DEFAULT,
    hparameters: Optional[Mapping[str, Sequence]] = None,
    hp_tune: Optional[Mapping[str, Mapping]] = None,
    other_models_marker_source: str = 'full_or_filtered',
    pool_cost_in_headline: bool = True,
    gpu_cpu_calibration_in_headline: bool = True,
    hpc_profile: Optional[str] = None,
    max_array_subjobs_override: Optional[int] = None,
    ncpus_per_gpu_override: Optional[int] = None,
    rf_filter_cfg: Optional[Mapping[str, object]] = None,
    ld_prune_estimated_n_markers: Optional[int] = None,
    coordinated_budget_ncpus: Optional[int] = None,
) -> Dict[str, object]:
    """Cheap resource estimate for one EasiGP run (or, via ``n_batches``,
    for ONE batch of a Parallel run split across that many batches) -
    Update ID 2 (R2)'s joint-allocation-aware successor to Phase 2's
    single-width estimator.

    Every positional argument above the ``*`` is UNCHANGED from Phase 2 -
    see ``_estimate_resources_phase2_core()``'s own docstring for what
    they mean. Everything below the ``*`` is NEW, keyword-only, and
    defaulted so that a caller supplying none of them (every Phase-2-era
    call site, unless it opts in) gets EXACTLY Phase 2's original
    single-width behaviour: with ``model_parallel_enabled=False`` (the
    default), this function does nothing but call
    ``_estimate_resources_phase2_core()`` with the exact same arguments
    Phase 2 always passed and return its dict, augmented with a second
    set of keys describing the (inert, width-1) model-level dimension -
    the original eight keys (``ncpus``, ``mem_gb``, ``ngpus``,
    ``est_relative_cost``, ``n_markers``, ``n_samples``,
    ``total_tasks_per_batch``, ``total_tasks_all_batches``, ``notes``)
    are byte-identical to Phase 2's own output for the same positional
    arguments (A2.2).

    Parameters (keyword-only, new in Update ID 2)
    -----------------------------------------------
    model_parallel_enabled : whether to compute a joint
        ``(n_cpu_workers_task, n_model_workers, n_gpu_slots)`` allocation
        at all (mirrors the ``N_MODEL_WORKERS`` config feature flag - the
        GUI passes ``True`` only when the person has actually turned on
        model-level fan-out).
    budget_ncpus, budget_ngpus, budget_mem_gb : the per-node/per-batch
        hardware budget to allocate within. ``None`` (any of them) falls
        back to this same call's own Phase-2 ``ncpus``/``ngpus``
        recommendation as the budget (``budget_mem_gb=None`` instead
        disables the memory clamp entirely, with a note - there is no
        Phase-2 memory "budget" figure to fall back to, only a
        single-process ``mem_gb`` estimate).
    gpu_slots_per_device : see ``_GPU_SLOTS_PER_DEVICE_DEFAULT``.
    hparameters, hp_tune : optional, used only to weight atom costs more
        realistically when a model is multi-algorithm-tuned (see
        ``_atoms_for_costing()``) - never required for a usable estimate.

    Parameters (keyword-only, new in Update ID 3, R4)
    ---------------------------------------------------
    other_models_marker_source : mirrors the ``OTHER_MODELS_MARKER_SOURCE``
        config key ``GP()`` itself already reads (architecture doc §8) -
        when this is anything other than ``'full_or_filtered'``, LD
        pruning and RF filtering are skipped ENTIRELY at run time for
        every non-bio-prior model, so this estimator prices as if both
        ``ld_prune_enabled``/``rf_filter_enabled`` were ``False``
        regardless of what was actually passed, and appends a note saying
        so - modelling the same short-circuit ``GP()`` applies, never
        re-deriving its semantics independently (I6).
    pool_cost_in_headline : fold ``estimate_pool_cost()``'s per-task
        Steps-1-8 pool-construction cost into the headline ``ncpus``/
        ``mem_gb`` (see ``_apply_pool_cost_to_core()``) - default ``True``.
        ``False`` reproduces the pre-Update-3 headline exactly (the
        rollback path; ``POOL_COST_IN_HEADLINE`` config key).

    Parameters (keyword-only, new in Patch 3, R4)
    ------------------------------------------------
    gpu_cpu_calibration_in_headline : fold the GPU-aware ``ncpus`` floor/
        headroom (see ``_apply_gpu_cpu_calibration_to_core()``) into the
        headline whenever this estimate recommends a GPU (``ngpus > 0``) -
        default ``True``. A no-op whenever ``ngpus == 0`` regardless of
        this flag. ``False`` reproduces the pre-Patch-3 headline exactly
        for a GPU-recommending estimate too (the rollback path;
        ``GPU_CPU_CALIBRATION_IN_HEADLINE`` config key).

    Parameters (keyword-only, new in Patch 3 v3, R3)
    ------------------------------------------------
    hpc_profile : one of ``HPC_RESOURCE_PROFILES``' keys (e.g.
        ``'NCI Gadi'``, ``'UQ Bunya'``, ``'Generic / other HPC'``) -
        selects which cluster's job-array cap / CPUs-per-GPU ratio this
        estimate calibrates against (see ``resolve_hpc_profile()``).
        ``None`` (the default) resolves to ``DEFAULT_HPC_PROFILE``
        ('NCI Gadi') - i.e. any pre-Patch-3-v3 caller that never passes
        this gets EXACTLY today's Gadi-only numbers, unchanged.
    max_array_subjobs_override, ncpus_per_gpu_override : explicit numeric
        overrides (the GUI's 'Custom' profile) - either one, supplied
        independently of the other, wins over the named profile's own
        value for that ONE number only.

    Parameters (keyword-only, new in Additional Requirements 9)
    -------------------------------------------------------------
    rf_filter_cfg : the run's own ``RF_FILTER`` config dict (``mode``/
        ``top_ratio``/``top_n`` - see ``build_rf_filter_config()`` in
        main_app.py), if RF filtering is enabled - lets
        ``estimate_post_filter_markers()`` derive RF's own EXACT marker
        reduction with zero extra input from the person. ``None`` (the
        default) leaves RF filtering's contribution un-costed, with a
        note explaining why (R4).
    ld_prune_estimated_n_markers : the person's own optional estimate of
        how many markers will remain after LD pruning (GUI: 'Estimated
        markers remaining after LD pruning'). ``None`` (the default, and
        every pre-Additional-Requirements-9 caller) falls back to
        assuming LD pruning removes nothing, which OVER-estimates memory
        for a run that actually has LD pruning enabled (R4's own
        documented fallback instruction) - see ``estimate_post_filter_markers()``.
    coordinated_budget_ncpus : CPU budget for the NEW ``suggest_compute_layout()``
        recommendation (R2) - a separate, always-computed
        ``(n_cpu_workers_task, n_jobs, ncpus)`` triple SIZED FOR THIS RUN'S
        OWN ``suggested_batch_size`` (never for whatever ``n_batches``/
        per-call batch happens to be passed in above - see
        ``suggest_array_layout()``'s own note on this). ``None`` (the
        default) falls back to this SAME call's own (post pool-cost/GPU-
        calibration) ``ncpus`` headline as the budget.

    Returns
    -------
    Dict with the original Phase-2 keys (see above) PLUS:
    ``n_cpu_workers_task``, ``n_model_workers``, ``n_gpu_slots``,
    ``cpus_per_model_worker``, ``model_groups``, ``pool_cost_factor``,
    ``redundant_pool_cost``, ``fanout_recommended``, ``fanout_reason``,
    ``w_opt_limitation`` - see the ver4-3 design blueprint §R2.5 for the
    meaning of each - PLUS, new in Update ID 3: ``pool_cost_headline_applied``
    (bool - whether the post-step above actually ran) and
    ``pool_cost_increment`` (float - ``pool_cost - _POOL_COST_BASE`` at the
    ``ld``/``rf`` state actually charged, i.e. 0.0 whenever
    ``pool_cost_in_headline=False``) - PLUS, new in Patch 3, R4:
    ``gpu_cpu_calibration_applied`` (bool - whether the GPU ncpus
    calibration actually ran, i.e. ``ngpus > 0`` and
    ``gpu_cpu_calibration_in_headline`` was True) and
    ``gpu_ncpus_increment`` (int - how many extra ncpus the calibration
    added on top of the pool-cost-adjusted base, 0 whenever it didn't run
    or the base already met the GPU floor) - PLUS, new in Patch 3 v2,
    Requirement 1: ``suggested_batch_size`` (int) and
    ``suggested_n_batches`` (int) - see ``suggest_array_layout()`` for how
    these are derived from this SAME call's phenotype/model/ratio-aware
    ``total_tasks_all_batches`` and per-task cost. Unlike every other key
    in this dict, these two describe the run as a WHOLE (independent of
    the ``n_batches`` argument passed in for THIS call's own per-batch
    costing) - they are a recommendation for what ``n_batches``/
    ``PARALLEL['batch_size']`` should be, not a function of what was
    passed in. PLUS, new in Additional Requirements 9: ``last_batch_size``
    (int) and ``batch_size_evenly_divides`` (bool) alongside the two keys
    above (R1); ``n_markers_effective`` (int), ``ld_estimate_uncertain``
    (bool), and ``marker_filter_calibration_applied`` (bool) describing
    the post LD/RF-filter marker estimate actually used to size
    ``mem_gb`` (R4); ``marker_filter_changed_mem_gb`` (bool - GUI defect
    fix, see the call site's own comment - whether pricing against
    ``n_markers_effective`` instead of the raw marker count actually moved
    the headline ``mem_gb`` figure; always ``False`` when
    ``marker_filter_calibration_applied`` is ``False``, and can legitimately
    be ``False`` even when it's ``True`` whenever both marker counts price
    out at/below ``_MIN_MEM_GB`` - the GUI uses this to say so plainly
    rather than implying a change that didn't happen); and
    ``coordinated_ncpus``, ``coordinated_n_cpu_workers_task``,
    ``coordinated_n_jobs``, ``coordinated_rounds_per_batch``, ``coordinated_mem_gb``
    - a SEPARATE, always-computed recommendation sized for THIS run's own
    ``suggested_batch_size`` (R2/R3 - see ``suggest_compute_layout()``) -
    ``coordinated_mem_gb`` specifically is the AGGREGATE memory needed for
    ``coordinated_n_cpu_workers_task`` concurrent worker processes (unlike
    the plain ``mem_gb`` above, which is, and always was, a single-process
    estimate never multiplied by worker count - a real production defect;
    see ``estimate_worker_memory()``'s own docstring).
    """
    # Update ID 3 (R4) - resolve what LD pruning/RF filtering ACTUALLY cost
    # at run time for THIS call's OTHER_MODELS_MARKER_SOURCE BEFORE calling
    # _estimate_resources_phase2_core() - architecture doc §8: when models
    # other than the ones pointed at the full/filtered pool use a
    # gene-network-derived marker set, GP() sets LD_prune_effective/
    # RF_filter_effective to None and skips BOTH steps ENTIRELY at run
    # time, so neither should be charged anywhere in this estimate - not
    # just in the headline post-step below, but also in
    # _estimate_resources_phase2_core()'s own units_per_task accounting
    # (compute_units_per_task()), which otherwise still adds its own +1
    # per enabled flag regardless of marker source. `_estimate_resources_
    # phase2_core()` itself is NOT modified (A2.2) - only WHICH arguments
    # this wrapper passes it changes, which is this function's own call to
    # make, not that one's.
    _effective_ld = ld_prune_enabled
    _effective_rf = rf_filter_enabled
    _marker_source_note = None
    if other_models_marker_source != 'full_or_filtered':
        _effective_ld = False
        _effective_rf = False
        _marker_source_note = (
            f"OTHER_MODELS_MARKER_SOURCE={other_models_marker_source!r} - LD pruning and RF "
            f"filtering are skipped entirely at run time for this marker source, so no "
            f"filtering cost is charged."
        )

    core = _estimate_resources_phase2_core(
        scenario, n_population, n_phenotype, n_ratio, sample_num, model_run,
        _effective_ld, _effective_rf, genotype_format, genotype_file_name,
        phenotype_file_name, w_opt_enabled, n_batches,
    )
    if _marker_source_note is not None:
        core = dict(core)
        core['notes'] = list(core['notes']) + [_marker_source_note]

    # Additional Requirements 9, Requirement 4 - price mem_gb against how
    # many markers a NON-bio-prior model actually sees once LD pruning/RF
    # filtering are applied (`_effective_ld`/`_effective_rf` - the SAME
    # already-short-circuited flags just used above, so this never charges
    # a filtering discount that OTHER_MODELS_MARKER_SOURCE would not
    # actually apply at run time either - I6, same reasoning as the
    # marker-source short-circuit immediately above). A true no-op
    # (`n_markers_effective == core['n_markers']`) whenever neither
    # filtering step is effectively enabled, so every pre-Additional-
    # Requirements-9 caller (which never passes `rf_filter_cfg`/
    # `ld_prune_estimated_n_markers`) still sees this reduce to "LD/RF off
    # -> nothing to estimate" exactly as before this feature existed
    # UNLESS LD/RF is genuinely on, in which case the fallback path inside
    # estimate_post_filter_markers() itself (assume LD removes nothing) is
    # what ran even before this feature existed too - see that function's
    # own docstring for why that fallback is a safe, over-estimating
    # default rather than a behaviour change.
    n_markers_effective, ld_estimate_uncertain, _marker_filter_notes = estimate_post_filter_markers(
        core['n_markers'], _effective_ld, _effective_rf, rf_filter_cfg, ld_prune_estimated_n_markers,
    )
    marker_filter_calibration_applied = n_markers_effective != core['n_markers']
    marker_filter_changed_mem_gb = False
    if marker_filter_calibration_applied:
        _mem_gb_before_marker_filter = core['mem_gb']
        core = _apply_marker_filter_calibration_to_core(
            core, model_run, core['n_markers'], n_markers_effective,
        )
        # GUI defect fix - "the suggested memory value does not change even
        # after users type an estimated marker value": recomputing mem_gb
        # (see _apply_marker_filter_calibration_to_core()'s own docstring
        # for the arithmetic fix) still legitimately leaves mem_gb
        # UNCHANGED whenever BOTH the raw and the filtered marker counts
        # price out at or below `_MIN_MEM_GB` - a small/pilot-scale run is
        # already at the practical memory floor either way, so there is
        # nothing left for the estimate to reduce. That is a genuine,
        # correct answer, not a bug - but the caption that names this
        # feature ("Memory sized for ~X marker(s) after LD/RF filtering")
        # would otherwise silently claim the figure reflects the filtered
        # count even when it didn't move at all, reproducing exactly the
        # "doesn't change" complaint. Recording whether mem_gb actually
        # moved lets the caller (main_app.py) say so plainly instead - see
        # "diagnostics as a design principle" (architecture doc §17).
        marker_filter_changed_mem_gb = core['mem_gb'] != _mem_gb_before_marker_filter
    if _marker_filter_notes:
        core = dict(core)
        core['notes'] = list(core['notes']) + _marker_filter_notes
    core = dict(core)
    core['n_markers_effective'] = n_markers_effective
    core['ld_estimate_uncertain'] = ld_estimate_uncertain
    core['marker_filter_calibration_applied'] = marker_filter_calibration_applied
    core['marker_filter_changed_mem_gb'] = marker_filter_changed_mem_gb

    # Patch 3 v3, Requirement 3 - resolve which HPC's job-array cap /
    # CPUs-per-GPU ratio this call should calibrate against, ONCE, before
    # either the GPU-calibration fold or the job-array-size fold below (so
    # the two can never disagree about which cluster this estimate is
    # for). `hpc_profile=None` (every pre-Patch-3-v3 call site) resolves
    # to `DEFAULT_HPC_PROFILE` ('NCI Gadi') - unchanged pre-existing
    # behaviour.
    _max_array_subjobs, _ncpus_per_gpu, _hpc_profile_note, _resolved_hpc_profile = resolve_hpc_profile(
        hpc_profile, max_array_subjobs_override, ncpus_per_gpu_override,
    )

    # Fold the Steps-1-8 pool-construction cost into the headline ncpus/
    # mem_gb HERE, before branching on model_parallel_enabled, so every
    # return point below (all three `result = dict(core)` sites) picks it
    # up identically without duplicating the adjustment three times. See
    # `_apply_pool_cost_to_core()`'s own docstring for why this is
    # structurally safe alongside A2.2.
    pool_cost_headline_applied = False
    pool_cost_increment = 0.0
    if pool_cost_in_headline:
        _headline_pool_cost = estimate_pool_cost(
            core['n_markers'], core['n_samples'], _effective_ld, _effective_rf, genotype_format,
        )
        core = _apply_pool_cost_to_core(core, _headline_pool_cost, core['total_tasks_per_batch'])
        pool_cost_headline_applied = True
        pool_cost_increment = round(_headline_pool_cost - _POOL_COST_BASE, 2)

    # Patch 3, Requirement 4 - GPU-aware ncpus floor/headroom, applied
    # HERE (after the pool-cost fold, before branching on
    # model_parallel_enabled) for the exact same reason pool_cost_in_
    # headline is applied here: every return point below (all three
    # `result = dict(core)` sites) picks it up identically without
    # duplicating the adjustment three times. See
    # _apply_gpu_cpu_calibration_to_core()'s own docstring.
    gpu_cpu_calibration_applied = False
    gpu_ncpus_increment = 0
    if gpu_cpu_calibration_in_headline and core.get('ngpus'):
        core, gpu_cpu_calibration_applied, gpu_ncpus_increment = _apply_gpu_cpu_calibration_to_core(
            core, _effective_ld, _effective_rf, ncpus_per_gpu=_ncpus_per_gpu,
        )
        if gpu_cpu_calibration_applied and gpu_ncpus_increment > 0:
            core = dict(core)
            _has_filtering = bool(_effective_ld or _effective_rf)
            if _ncpus_per_gpu and _ncpus_per_gpu > 0:
                # Patch 3 v3, R3: name the SELECTED profile's own ratio,
                # not always "NCI Gadi" - the estimate may be calibrated
                # against a different HPC's floor now.
                _gpu_note = (
                    f"ncpus increased by {gpu_ncpus_increment} for GPU use: the selected HPC "
                    f"profile requires ncpus to be a multiple of {_ncpus_per_gpu} per GPU requested"
                    + (", plus headroom so CPU-bound LD pruning/RF filtering aren't starved "
                       "while the GPU runs the model fit" if _has_filtering else "") + "."
                )
            else:
                # No fixed per-GPU ncpus floor for this profile (e.g. UQ
                # Bunya) - the increment here is purely the LD/RF
                # preprocessing headroom, so say that plainly rather than
                # naming a floor that doesn't apply.
                _gpu_note = (
                    f"ncpus increased by {gpu_ncpus_increment} for GPU use, giving CPU-bound LD "
                    f"pruning/RF filtering headroom alongside the GPU-resident model fit "
                    f"(the selected HPC profile has no fixed CPUs-per-GPU ratio to floor to)."
                )
            core['notes'] = list(core['notes']) + [_gpu_note]

    # Patch 3 v2, Requirement 1 - fold the job-array-size ("batch size")
    # suggestion into `core` HERE, same position/reasoning as the pool-cost
    # and GPU-calibration folds just above: every return point below (both
    # `result = dict(core)` sites) inherits `suggested_batch_size`/
    # `suggested_n_batches` identically without duplicating the call twice.
    # Uses `_effective_ld`/`_effective_rf` (not the raw, possibly-
    # short-circuited `ld_prune_enabled`/`rf_filter_enabled` arguments) so
    # units_per_task here matches EXACTLY what `_estimate_resources_
    # phase2_core()` itself already priced above (I6 - see the
    # OTHER_MODELS_MARKER_SOURCE short-circuit note earlier in this
    # function).
    _array_units_per_task = compute_units_per_task(len(model_run), _effective_ld, _effective_rf)
    _array_layout = suggest_array_layout(
        core['total_tasks_all_batches'], _array_units_per_task,
        max_array_subjobs=_max_array_subjobs,
    )
    core = dict(core)
    core['suggested_batch_size'] = _array_layout['suggested_batch_size']
    core['suggested_n_batches'] = _array_layout['suggested_n_batches']
    core['last_batch_size'] = _array_layout['last_batch_size']
    core['batch_size_evenly_divides'] = _array_layout['batch_size_evenly_divides']
    if _array_layout['notes']:
        core['notes'] = list(core['notes']) + _array_layout['notes']
    # Patch 3 v3, R3: echo back exactly which numbers this call actually
    # calibrated against, so the GUI can display them without re-deriving
    # resolve_hpc_profile() itself (and so they can never drift out of
    # sync with what was actually used above).
    core['hpc_profile'] = _resolved_hpc_profile
    core['max_array_subjobs'] = _max_array_subjobs
    core['ncpus_per_gpu'] = _ncpus_per_gpu
    core['hpc_profile_note'] = _hpc_profile_note

    # Additional Requirements 9, Requirements 2/3 - a SEPARATE, always-on
    # (n_cpu_workers_task, n_jobs, ncpus) recommendation, coordinated with
    # the array-size suggestion just folded in above by sizing it for
    # THIS SAME run's own `suggested_batch_size` - not for whatever
    # `n_batches`/per-call batch this particular estimate_resources() call
    # happened to be sized against (see suggest_compute_layout()'s own
    # docstring). This is unconditional (unlike the `model_parallel_
    # enabled`-gated joint allocator further below, which recommends a
    # DIFFERENT thing - splitting CPUs across task-level AND model-level
    # width) - every caller gets a coordinated ncpus/N_CPU_WORKERS_TASK/
    # N_JOBS triple by default now, closing the "no coordination between
    # the suggested job-array size and the suggested compute resources"
    # gap Requirement 2 names, regardless of whether model-level
    # parallelism is in play at all.
    _coordinated_budget = int(coordinated_budget_ncpus) if coordinated_budget_ncpus else int(core['ncpus'])
    _coordinated = suggest_compute_layout(
        core['suggested_batch_size'], model_run, budget_ncpus=_coordinated_budget, hparameters=hparameters,
    )
    core['coordinated_ncpus'] = _coordinated['ncpus']
    core['coordinated_n_cpu_workers_task'] = _coordinated['n_cpu_workers_task']
    core['coordinated_n_jobs'] = _coordinated['n_jobs']
    core['coordinated_rounds_per_batch'] = _coordinated['rounds_per_batch']
    core['coordinated_notes'] = list(_coordinated['notes'])

    # Additional Requirements 9 (production defect fix - a real batch with
    # N_CPU_WORKERS_TASK=20 and a memory request sized for ONE process was
    # OOM-killed - see the incident write-up beside
    # intra_batch_parallel._systemic_failure_note()) - `core['mem_gb']`
    # above (and the plain 'Suggested mem' headline it feeds in the GUI)
    # is, and always was, a SINGLE-PROCESS estimate: it was never
    # multiplied by how many CONCURRENT task-level workers actually run.
    # `coordinated_mem_gb` is the memory this run's own
    # `coordinated_n_cpu_workers_task` concurrent, independent worker
    # processes actually need in aggregate - each running the FULL
    # `model_run` sequentially for its own task (see
    # `estimate_worker_memory()`'s own R/BGLR-accumulation-aware
    # docstring) - so a person sizing their job's requested memory off
    # THIS figure (rather than the plain per-process headline) whenever
    # they configure N_CPU_WORKERS_TASK > 1 should not hit that failure
    # again.
    _coordinated_worker_mem = estimate_worker_memory(model_run)
    # Exposed independently of `coordinated_mem_gb` (which is already
    # multiplied by `coordinated_n_cpu_workers_task`) so a CALLER applying
    # a DIFFERENT worker count than this function's own coordinated one
    # (e.g. main_app.py's original, uncapped `n_cpu_workers_task ==
    # ncpus` path when model-level parallelism isn't being considered -
    # see that button's own comment) can still scale memory correctly
    # against WHATEVER worker count it ends up applying, rather than
    # silently pairing a large worker count with this call's own
    # (possibly smaller) `coordinated_mem_gb`.
    core['per_worker_mem_gb'] = round(_coordinated_worker_mem, 2)
    core['coordinated_mem_gb'] = round(
        max(float(core['mem_gb']), _coordinated_worker_mem * core['coordinated_n_cpu_workers_task']), 1,
    )
    if core['coordinated_n_cpu_workers_task'] > 1:
        core['coordinated_notes'] = core['coordinated_notes'] + [
            f"Memory: {core['coordinated_n_cpu_workers_task']} concurrent task-level worker(s) x "
            f"~{_coordinated_worker_mem:.1f} GB/worker (this run's own full model roster, R/BGLR "
            f"sessions accumulating memory across sequential model calls within one task) \u2248 "
            f"{core['coordinated_mem_gb']:.1f} GB total - request AT LEAST this much memory for "
            f"this batch, not the single-process 'Suggested mem' headline above; requesting too "
            f"little is what causes worker processes to be OOM-killed "
            f"('BrokenProcessPool'/'terminated abruptly' errors) partway through a batch."
        ]

    w_opt_limitation = bool(w_opt_enabled)


    if not model_parallel_enabled:
        # Flag off: the SAME single-width recommendation Phase 2 always
        # gave, just re-expressed under the new joint-allocation key
        # names too, so a caller that has already switched to reading
        # the new keys sees a sensible (width-1) value regardless of
        # whether model-level fan-out is actually in use for this
        # particular estimate.
        result = dict(core)
        result.update({
            'n_cpu_workers_task': core['ncpus'],
            'n_model_workers': 1,
            'n_gpu_slots': core['ngpus'],
            'cpus_per_model_worker': max(1, core['ncpus']),
            'model_groups': [[m for m in model_run if m != 'ensemble']] if model_run else [],
            'pool_cost_factor': None,
            'redundant_pool_cost': 0.0,
            'fanout_recommended': False,
            'fanout_reason': "model-level parallelism not enabled for this estimate (model_parallel_enabled=False).",
            'w_opt_limitation': w_opt_limitation,
            'pool_cost_headline_applied': pool_cost_headline_applied,
            'pool_cost_increment': pool_cost_increment,
            'gpu_cpu_calibration_applied': gpu_cpu_calibration_applied,
            'gpu_ncpus_increment': gpu_ncpus_increment,
        })
        return result

    # ---------------------------------------------------------------- #
    # model_parallel_enabled=True: joint allocator.
    # ---------------------------------------------------------------- #
    notes: List[str] = list(core['notes'])

    if budget_ncpus is None:
        budget_ncpus = core['ncpus']
        notes.append(
            f"No CPU budget supplied - using this estimate's own Phase-2 'ncpus' recommendation "
            f"({budget_ncpus}) as the joint-allocation budget."
        )
    if budget_ngpus is None:
        budget_ngpus = core['ngpus']
        notes.append(
            f"No GPU budget supplied - using this estimate's own Phase-2 'ngpus' recommendation "
            f"({budget_ngpus}) as the joint-allocation budget."
        )
    if budget_mem_gb is None:
        notes.append(
            "No memory budget supplied - the joint allocation is NOT memory-clamped; "
            "set a memory budget to avoid recommending a width that would OOM the node."
        )
    budget_ncpus = max(1, int(budget_ncpus))
    budget_ngpus = max(0, int(budget_ngpus))
    gpu_slots_per_device = max(1, int(gpu_slots_per_device))

    n_markers = core['n_markers']
    pool_cost = estimate_pool_cost(n_markers, core['n_samples'], _effective_ld, _effective_rf, genotype_format)
    total_tasks_per_batch = max(1, int(core['total_tasks_per_batch']) or 1)

    atoms, atom_costs = _atoms_for_costing(model_run, hp_tune)
    n_atoms = len(atoms)

    if w_opt_limitation:
        notes.append(
            "W_OPT (weighted-ensemble methods) is configured - per the ver4-3 design blueprint "
            "(decision D1), any task with W_OPT active AND a validation split falls back to "
            "task-level-only execution at RUN TIME regardless of N_MODEL_WORKERS. This estimate's "
            "model-level width may therefore not be achievable for every task in this run."
        )

    if n_atoms < 2:
        _n_cpu_workers_task_small = min(total_tasks_per_batch, budget_ncpus)
        # Additional Requirements 9 (production defect fix - see
        # intra_batch_parallel._systemic_failure_note()'s own incident
        # write-up): this branch's own worker count implies THIS many
        # concurrent, independent processes, each running the full
        # `model_run` for its own task - `core['mem_gb']` alone is a
        # single-process figure that was never multiplied by that count.
        _mem_gb_small = round(
            max(float(core['mem_gb']), estimate_worker_memory(model_run) * _n_cpu_workers_task_small), 1,
        )
        result = dict(core)
        result['notes'] = notes
        result.update({
            'mem_gb': _mem_gb_small,
            'n_cpu_workers_task': _n_cpu_workers_task_small,
            'n_model_workers': 1,
            'n_gpu_slots': min(budget_ngpus * gpu_slots_per_device, budget_ngpus) if core['ngpus'] else 0,
            'cpus_per_model_worker': max(1, budget_ncpus // max(1, _n_cpu_workers_task_small)),
            'model_groups': atoms,
            'pool_cost_factor': round(pool_cost, 2),
            'redundant_pool_cost': 0.0,
            'fanout_recommended': False,
            'fanout_reason': f"only {n_atoms} model atom(s) available after grouping - nothing to fan out across.",
            'w_opt_limitation': w_opt_limitation,
            'pool_cost_headline_applied': pool_cost_headline_applied,
            'pool_cost_increment': pool_cost_increment,
            'gpu_cpu_calibration_applied': gpu_cpu_calibration_applied,
            'gpu_ncpus_increment': gpu_ncpus_increment,
        })
        return result

    # Stage 1: is fanning model fits out worth it AT ALL, cost-model-wise,
    # in the BEST case (the maximum model-level width this budget and
    # atom count allow)? Answered once, up front, via the SAME shared
    # predicate the runtime eligibility gate uses (A2.6) - this is what
    # lets the reason string always name the actual atom count (A2.3),
    # regardless of what a later task-throughput trade-off (Stage 2) ends
    # up choosing.
    max_n_model = max(1, min(n_atoms, budget_ncpus))
    bins_at_max = pack_atoms_lpt(atom_costs, max_n_model)
    group_fit_costs_at_max = [sum(atom_costs[i] for i in b) for b in bins_at_max]
    cost_model_recommends, cost_model_reason = should_fan_out_models(pool_cost, group_fit_costs_at_max)
    atom_summary = f"{n_atoms} model atom(s) available for model-level fan-out"

    # Reported UNCONDITIONALLY at the maximum feasible width, regardless
    # of which width is ultimately recommended below - an INFORMATIONAL
    # "what fanning out all the way would cost in redundant Steps-1-8
    # rebuilds" figure. Deliberately NOT recomputed against whatever
    # width is finally chosen: `len(bins_at_max)` depends only on
    # `n_atoms`/`budget_ncpus` (never on `pool_cost` itself), so this
    # value is a pure, monotonically-increasing function of `pool_cost`
    # alone (A2.4) - snapping it to 0 whenever the final recommendation
    # happens to be "don't fan out" would make it jump discontinuously
    # right at the break-even boundary, which is exactly the case a
    # config toggling LD pruning/RF filtering on is most likely to cross.
    redundant_pool_cost = round(max(0, len(bins_at_max) - 1) * pool_cost * total_tasks_per_batch, 2)

    def _single_group_candidate() -> dict:
        n_task = max(1, min(total_tasks_per_batch, budget_ncpus))
        group_members = [[m for atom in atoms for m in atom]]
        worker_mem = estimate_worker_memory(group_members[0])
        return {
            'n_model': 1, 'n_task': n_task, 'model_groups': group_members,
            'mem_estimate': worker_mem * n_task,
        }

    if not cost_model_recommends:
        # Even the BEST-CASE fan-out doesn't clear the break-even margin -
        # no point enumerating narrower widths, they can only be worse.
        best = _single_group_candidate()
        fanout_recommended = False
        fanout_reason = f"{atom_summary}, but {cost_model_reason}."
    else:
        # Stage 2: fanning out is worth it in principle - now find the
        # THROUGHPUT-optimal width, enumerating n_model = 1..max_n_model,
        # deriving n_task from the remaining budget at each, packing atoms
        # via pack_atoms_lpt() (the SAME algorithm the runtime uses), and
        # pricing the resulting batch wall-clock (blueprint §R2.2's cost
        # model). Ties are broken toward the SMALLER n_model by
        # construction: the loop runs n_model ascending and only replaces
        # `best` on a STRICTLY lower wall-clock, so the first (smallest)
        # n_model to reach the minimum keeps it.
        best = None
        for n_model in range(1, max_n_model + 1):
            n_task = max(1, min(total_tasks_per_batch, budget_ncpus // n_model))
            bins = pack_atoms_lpt(atom_costs, n_model)
            group_fit_costs = [sum(atom_costs[i] for i in b) for b in bins]
            group_members = [[m for i in b for m in atoms[i]] for b in bins]

            fanout_per_task = pool_cost + (max(group_fit_costs) if group_fit_costs else 0.0)
            batch_wall_clock = math.ceil(total_tasks_per_batch / n_task) * fanout_per_task
            redundant_pool_cost = max(0, len(bins) - 1) * pool_cost * total_tasks_per_batch

            worker_mem = max((estimate_worker_memory(m) for m in group_members), default=_BASE_MEM_GB)
            mem_estimate = worker_mem * n_task * n_model

            if budget_mem_gb is not None and mem_estimate > budget_mem_gb and n_model > 1:
                # Infeasible under the memory budget - skip, UNLESS this is
                # the n_model=1 floor, which is always kept as a
                # last-resort candidate regardless of the memory clamp
                # (there is no narrower width to fall back to).
                continue

            candidate = {
                'n_model': n_model, 'n_task': n_task, 'batch_wall_clock': batch_wall_clock,
                'redundant_pool_cost': redundant_pool_cost, 'group_fit_costs': group_fit_costs,
                'model_groups': group_members, 'mem_estimate': mem_estimate,
            }
            if best is None or candidate['batch_wall_clock'] < best['batch_wall_clock'] - 1e-9:
                best = candidate

        if best is None:
            # Every n_model from 1 upward exceeded the memory budget (a
            # pathological, very tight budget) - fall back to the
            # n_model=1 floor unconditionally, with a note.
            best = _single_group_candidate()
            best['redundant_pool_cost'] = 0.0
            notes.append(
                f"Even the narrowest model-level width (N_MODEL_WORKERS=1) is estimated at "
                f"~{best['mem_estimate']:.1f} GB total, above the supplied memory budget "
                f"({budget_mem_gb:.1f} GB) - consider raising the memory budget or reducing "
                f"'Tasks in parallel per batch'."
            )
        elif budget_mem_gb is not None and best['mem_estimate'] > budget_mem_gb:
            notes.append(
                f"The recommended allocation (~{best['mem_estimate']:.1f} GB) exceeds the "
                f"supplied memory budget ({budget_mem_gb:.1f} GB) - this can only happen at "
                f"N_MODEL_WORKERS=1, where there is no narrower width to fall back to; "
                f"consider reducing 'Tasks in parallel per batch' instead."
            )

        fanout_recommended = best['n_model'] > 1
        if fanout_recommended:
            fanout_reason = cost_model_reason
        else:
            fanout_reason = (
                f"{atom_summary} and {cost_model_reason}, but with {total_tasks_per_batch} task(s) "
                f"to process against a budget of {budget_ncpus} CPU(s), maximising TASK-level "
                f"parallelism (N_CPU_WORKERS_TASK) instead minimises this batch's estimated "
                f"wall-clock time - degrading to the single-width recommendation."
            )

    n_model_workers = best['n_model']
    n_cpu_workers_task = best['n_task']
    n_gpu_capable_groups = sum(
        1 for members in best['model_groups'] if any(bool(_profile_for(m)['gpu_capable']) for m in members)
    )
    n_gpu_slots = min(budget_ngpus * gpu_slots_per_device, n_cpu_workers_task * max(1, n_gpu_capable_groups)) \
        if n_gpu_capable_groups else 0
    cpus_per_model_worker = max(1, budget_ncpus // max(1, n_cpu_workers_task * n_model_workers))

    # Additional Requirements 9 (production defect fix - see
    # intra_batch_parallel._systemic_failure_note()'s own incident
    # write-up): `best['mem_estimate']` (computed above, per-candidate,
    # specifically to price the memory-budget clamp) was previously
    # discarded once the winning candidate was chosen - `core['mem_gb']`
    # (a single-process figure) was returned as `mem_gb` unchanged
    # regardless, so a person applying "Suggested mem" for a
    # model-parallel recommendation with `n_cpu_workers_task x
    # n_model_workers > 1` concurrent processes still got a per-process
    # number. Surfaced here so `mem_gb` actually reflects what THIS
    # recommendation needs in aggregate.
    _mem_gb_joint = round(max(float(core['mem_gb']), float(best.get('mem_estimate', 0.0))), 1)

    result = dict(core)
    result['notes'] = notes
    result.update({
        'mem_gb': _mem_gb_joint,
        'n_cpu_workers_task': n_cpu_workers_task,
        'n_model_workers': n_model_workers,
        'n_gpu_slots': n_gpu_slots,
        'cpus_per_model_worker': cpus_per_model_worker,
        'model_groups': best['model_groups'],
        'pool_cost_factor': round(pool_cost, 2),
        'redundant_pool_cost': redundant_pool_cost,
        'fanout_recommended': fanout_recommended,
        'fanout_reason': fanout_reason,
        'w_opt_limitation': w_opt_limitation,
        'pool_cost_headline_applied': pool_cost_headline_applied,
        'pool_cost_increment': pool_cost_increment,
        'gpu_cpu_calibration_applied': gpu_cpu_calibration_applied,
        'gpu_ncpus_increment': gpu_ncpus_increment,
    })
    return result
