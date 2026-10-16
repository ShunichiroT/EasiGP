#!/usr/bin/env python3
"""
EasiGP - headless, non-interactive runner for the full Sequential pipeline
(data setup -> model fitting -> ensemble weighting -> scatter plot ->
circos plot) in a single pass. This script is meant to be launched directly
by an HPC job scheduler (Slurm/PBS) as a normal (non-array) batch job - NOT
through Streamlit - so a single GUI configuration step is all that's ever
needed, with no browser session required to stay open on the cluster.

Workflow
--------
1. In the Streamlit GUI (streamlit_app.py), choose Sequential mode,
   configure everything, then under 'Run pipeline' generate a script draft,
   edit it if needed, and save it into this project's folder. This also
   writes Result/<RESULT_NAME>/sequential_config.json.

2. Submit the saved script, e.g.:

       Slurm:  sbatch EasiGP_<RESULT_NAME>_sequential.sh
       PBS:    qsub  EasiGP_<RESULT_NAME>_sequential.sh

   The script simply calls:

       python run_sequential.py --config Result/<RESULT_NAME>/sequential_config.json

3. You can also run it directly for a local test:

       python run_sequential.py --config Result/<RESULT_NAME>/sequential_config.json

4. To deliberately redo this run from scratch - discarding any previously
   saved checkpoint/result files, rather than the default checkpoint/resume
   behaviour of picking up where a previous attempt left off - add
   --overwrite:

       python run_sequential.py --config Result/<RESULT_NAME>/sequential_config.json --overwrite

   (Update ID 2, Defect D4 fix - see PATCH_NOTES_D2.md. Checkpoint/resume
   remains the default for a routine retry after a crash/timeout; this flag
   is only for the deliberate "no, really, start this over" case.)
"""

import argparse
import plot_style as _plot_style   # ver4-31 R15
import json
import sys
import time

import pandas as pd

import checkpoint_utils as _ckpt
from pipeline_utils import (
    configure_r_environment, init_rpy2_conversion, restore_ratio, iterative_labels,
    super_learner_labels,   # ver4-77
    TimestampedWriter, make_run_log_path, apply_scratch_tmp_dir,
    # ver4-4 Stage 6 (R3.g): cheap, header/column-only reads used SOLELY to
    # compute an authoritative total-task count BEFORE calling GP() at all
    # (never a substitute for GP()'s own, real POPULATION/PHENOTYPE
    # resolution) - see the SEQUENTIAL_INTRA_BATCH branch below for why
    # this must be exact, not a best-effort estimate.
    count_unique_populations, list_phenotype_columns, resolve_trait_columns,
    # ver4-25, R14: log-only check for fan-out/CPU-threading keys the GUI
    # can no longer produce (see pipeline_utils.PINNED_FANOUT_DEFAULTS).
    log_pinned_fanout_deviations,
    # ver4-33: ACCELERATION dict (+ engine-only keys) to forward into GP().
    acceleration_from_cfg,
)
# ver4-4 Stage 6 (R3.g): resource_profiles.py imports nothing Streamlit-
# specific (see its own module-level imports - just pipeline_utils and the
# standard library), so it's safe to import here, at module load time,
# unlike the R/rpy2-dependent engine imports below (which are deliberately
# deferred until AFTER configure_r_environment()/init_rpy2_conversion()).
from resource_profiles import compute_total_tasks


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--config', required=True,
        help='Path to the sequential_config.json produced by the GUI ("Generate and save job files").'
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help=(
            'Force this run to start completely FRESH: delete its checkpoint and result '
            'files before running - rather than the default behaviour of automatically '
            'resuming from wherever a previous attempt left off. Also honoured as a '
            'config key ("OVERWRITE_RESULTS": true in the JSON itself). Use this to '
            'deliberately redo a run from scratch; do NOT use it as a routine retry '
            'after a crash/timeout - checkpoint/resume already handles that '
            'automatically, without re-doing already-successful, possibly hours-long '
            'work.'
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, 'r', encoding='utf-8-sig') as f:
        cfg = json.load(f)

    # ver4-54 (R6): a Tool config belongs to run_tool.py - refuse it here, before any run.
    if cfg.get('RUN_KIND') == 'tool':
        sys.exit("This is a Tool config - run it with run_tool.py")

    # Update ID 2 (R1): intra-task model-level parallelism
    # (intra_task_parallel.py) is a Parallel-mode-only feature - it fans
    # a BATCH's own per-task GP() calls out across model-groups, which
    # only exists as a concept under run_step1_batch.py's own PARALLEL
    # batching. Sequential mode calls GP() exactly once, directly, with
    # no PARALLEL context at all (see below) - there is no batch of
    # per-task units here for N_MODEL_WORKERS to fan out across, so this
    # is a LOG-ONLY, no-behaviour-change check: if a Sequential config
    # somehow carries N_MODEL_WORKERS > 1 (e.g. copied over from a
    # Parallel config by hand), it is simply ignored, with a warning,
    # rather than silently doing nothing with no explanation.
    _n_model_workers = int(cfg.get('N_MODEL_WORKERS', 1) or 1)
    if _n_model_workers > 1:
        print(f"[run_sequential] NOTE: N_MODEL_WORKERS={_n_model_workers} is set in this config, "
              f"but intra-task model-level parallelism (Update ID 2, R1) only applies to Parallel "
              f"mode's own batched GP() calls (see run_step1_batch.py) - it has no effect here and "
              f"is being ignored. Sequential mode calls GP() once, directly, unchanged.")

    # Wrap stdout so every line this process prints (from this script, from
    # genomic_prediction.py's own progress output, etc.) gets a timestamp
    # prefix automatically, AND save it to Result/<RESULT_NAME>/logs/ - the
    # same place the GUI's own local-run option already saves a log (see
    # pipeline_utils.make_run_log_path) - in addition to whatever the job
    # scheduler itself captures to its own job-output file, so the log
    # travels with this run's other output either way.
    log_file_path = make_run_log_path(cfg['RESULT_NAME'], 'sequential')
    log_file = open(log_file_path, 'w', encoding='utf-8')
    sys.stdout = TimestampedWriter(sys.stdout, log_file)
    print(f'[run_sequential] Logging to {log_file_path}')

    # PATCH_NOTES (worker-crash fix): see pipeline_utils.apply_scratch_tmp_dir()'s
    # own docstring - a no-op unless EASIGP_SCRATCH_TMPDIR (or this config's
    # own 'SCRATCH_TMP_DIR' key) is set. Resolved before configure_r_environment()
    # and before any heavy (torch/torch_geometric) import happens further down.
    apply_scratch_tmp_dir(result_name=cfg['RESULT_NAME'], explicit=cfg.get('SCRATCH_TMP_DIR'))

    # Update ID 2, Defect D4 fix (see PATCH_NOTES_D2.md): an explicit
    # --overwrite (or config-level "OVERWRITE_RESULTS": true) means the
    # person deliberately wants this run redone from scratch, NOT
    # auto-resumed. Deleting the checkpoint file alone is enough - GP()'s
    # own `_ckpt.load_checkpoint()` then returns None, so GP() takes its
    # existing "starting fresh" branch and clears its own result files
    # automatically (see genomic_prediction.py) - this just makes that
    # reachable without hand-deleting files on the cluster.
    overwrite = bool(args.overwrite or cfg.get('OVERWRITE_RESULTS', False))
    if overwrite:
        print(f"[run_sequential] --overwrite requested: discarding any previously saved "
              f"checkpoint/result files for RESULT_NAME={cfg['RESULT_NAME']} before running - "
              f"this run will be redone from scratch rather than resumed.")
        _ckpt.clear_checkpoint(cfg['RESULT_NAME'], 0, False)
        _ckpt.clear_result_files(cfg['RESULT_NAME'], 0, False)
        from pipeline_utils import clear_memo_dir as _clear_memo_dir
        _clear_memo_dir(cfg['RESULT_NAME'])

    configure_r_environment(cfg.get('R_PATH'), r_max_ppsize=cfg.get('R_MAX_PPSIZE', 500000))
    init_rpy2_conversion()

    # Imported after R/rpy2 setup so R_HOME/PATH are already correct.
    from genomic_prediction import GP
    from metric_summary import write_metric_summary
    from metric_plot import metric_plot
    from accuracy_metrics_plot import accuracy_metrics_plot, any_plot_requested as any_accuracy_plot_requested
    from scatter_plot import scatter_plot
    from circos_plot import circos_plot, _broadcast_population_info, _clean_population_label
    from attention_histogram import attention_distribution
    from Preprocess.LD_decay_plot import average_and_plot_ld_decay
    # ver4-4 R7.g: the SAME "5 leading metadata columns, everything past
    # that is a marker" convention batch_reader.ResultSet.effect() already
    # uses for its own float32 downcast - imported rather than
    # re-declaring the same literal width a second time in this file.
    from batch_reader import _EFFECT_METADATA_WIDTH, load_combined
    # Update ID ver4-9, R5.
    from diversity_summary import build_dpt_terms, write_dpt_summary, resolve_ensemble_members, dpt_model_order
    # Update ID ver4-9, R6.
    from weight_plot import weight_plot

    ratio = restore_ratio(cfg['RATIO'], cfg['SCENARIO'])

    print(f"[run_sequential] RESULT_NAME={cfg['RESULT_NAME']} models={cfg['MODEL']}")
    # ver4-25, R14: purely informational - see log_pinned_fanout_deviations()'s
    # own docstring. Never changes cfg or how GP() is called below.
    log_pinned_fanout_deviations(cfg, log_prefix='[run_sequential]')

    # ver4-4 R7.g/R7.h: resolved once, here, and reused at every plotting
    # call below - PLOT_EFFECT_FLOAT32 (default True) and PLOT_DPI - see
    # run_step2_assemble.py's identical note for the full rationale.
    # Logged unconditionally so a person comparing output across
    # runs/versions can see what was actually used.
    #
    # R7.h follow-up: DPI is now THREE INDEPENDENT config keys - one per
    # plot section (METRIC_PLOT_DPI/SCATTER_PLOT_DPI/CIRCOS_PLOT_DPI) -
    # rather than one shared 'PLOT_DPI', so a config can ask for, e.g., a
    # print-quality circos plot without also slowing down the quick-look
    # violin/scatter diagnostics. Each key falls back to the legacy
    # shared 'PLOT_DPI' (then to 300) so a *_config.json written before
    # this split still reproduces its own exact behaviour unchanged.
    _plot_effect_float32 = bool(cfg.get('PLOT_EFFECT_FLOAT32', True))
    _legacy_plot_dpi = cfg.get('PLOT_DPI', 300)
    _metric_plot_dpi = int(cfg.get('METRIC_PLOT_DPI', _legacy_plot_dpi))
    _scatter_plot_dpi = int(cfg.get('SCATTER_PLOT_DPI', _legacy_plot_dpi))
    _circos_plot_dpi = int(cfg.get('CIRCOS_PLOT_DPI', _legacy_plot_dpi))
    print(f"[run_sequential] PLOT_EFFECT_FLOAT32={_plot_effect_float32}, "
          f"METRIC_PLOT_DPI={_metric_plot_dpi}, SCATTER_PLOT_DPI={_scatter_plot_dpi}, "
          f"CIRCOS_PLOT_DPI={_circos_plot_dpi}")

    # ver4-4 Stage 6 (blueprint §2.3.2 R3.f/§10 Stage 6 checklist item 1):
    # every keyword argument GP() accepts, built ONCE as a plain dict -
    # both the legacy direct `GP(**gp_kwargs)` call below AND the new
    # SEQUENTIAL_INTRA_BATCH route (`run_batch_with_intra_batch_
    # parallelism(gp_kwargs, ...)`, which needs exactly this same dict
    # shape - see intra_batch_parallel.py's own docstring) consume the
    # IDENTICAL dict, so the two routes can never silently drift apart on
    # which keys/defaults they forward - the same "one implementation
    # reached from two places" discipline R3.c's own resolve_compute_
    # resources() deviation already established for this codebase.
    gp_kwargs = dict(
        GENOTYPE_FILE_NAME=cfg['GENOTYPE_FILE_NAME'], PHENOTYPE_FILE_NAME=cfg['PHENOTYPE_FILE_NAME'],
        MODEL=cfg['MODEL'], PHENOTYPE=cfg['PHENOTYPE'], RATIO=ratio, SAMPLE_NUM=cfg['ITER_NUM'],
        HPARAMETERS=cfg['HPARAMETERS'], R_PATH=cfg['R_PATH'], W_OPT=cfg['W_OPT'],
        RESULT_NAME=cfg['RESULT_NAME'], HYPERPARAMETERS_OPT=cfg['HYPERPARAMETERS_OPT'],
        SCENARIO=cfg['SCENARIO'], LD_prune=cfg.get('LD_PRUNE'), RF_filter=cfg.get('RF_FILTER'),
        GENOTYPE_FORMAT=cfg.get('GENOTYPE_FORMAT', 'csv'), GENOTYPE_PLINK_PATH=cfg.get('GENOTYPE_PLINK_PATH', 'plink2'),
        OTHER_MODELS_MARKER_SOURCE=cfg.get('OTHER_MODELS_MARKER_SOURCE', 'full_or_filtered'),
        HP_TUNE=cfg.get('HP_TUNE'), HP_TUNE_ENSEMBLE_MODE=cfg.get('HP_TUNE_ENSEMBLE_MODE', 'per_method'),
        # ver4-26, R4 (design blueprint `EasiGP_ver4-26_Design_Blueprint.md`
        # hand-off checklist): forward the two new joint-search keys into
        # GP() - None/absent for every pre-ver4-26 config (I11), matching
        # this dict's own established "forward the new keys into GP()"
        # precedent (see ver4-4 Stage 5's identical note just below).
        JOINT_W_OPT=cfg.get('JOINT_W_OPT'), HYPERPARAMETERS_JOINT_OPT=cfg.get('HYPERPARAMETERS_JOINT_OPT'),
        ITERATIVE_ENSEMBLE=cfg.get('ITERATIVE_ENSEMBLE'),  # ver4-57: opt-in, None = off
        SUPER_LEARNER=cfg.get('SUPER_LEARNER'),  # ver4-77: opt-in, None = off
        SEED_BAG=cfg.get('SEED_BAG'),  # ver4-102 (P4): opt-in, None = off
        MIN_DATA_POINTS=cfg.get('MIN_DATA_POINTS', 100),
        # Update ID 2 (R2) - Test Report D1 fix: forward GPU_SLOTS_PER_DEVICE
        # from the config JSON through to GP() (-> resolve_compute_resources()),
        # matching run_step1_batch.py's own fix, so a hand-edited value
        # actually takes effect at run time here too. Default 1 matches
        # every config written before this key existed.
        GPU_SLOTS_PER_DEVICE=cfg.get('GPU_SLOTS_PER_DEVICE', 1),
        # ver4-4 Stage 5 completion (blueprint §10 Stage 5 checklist / §2.4.3
        # R4 touch points "run_step1_batch.py, run_sequential.py | Forward
        # the new keys into GP()."): closes a gap disclosed since Stage 3 -
        # unlike run_step1_batch.py's own gp_kwargs (which has always
        # forwarded these), THIS call site never forwarded ANY of
        # USE_GPU_SKLEARN/N_JOBS/PLINK_THREADS/R_BLAS_THREADS/TORCH_DEVICE/
        # CUDNN_BENCHMARK/USE_AMP/N_CPU_WORKERS/N_GPU_SLOTS at all, so a
        # Sequential-mode run had NO way to reach any of Stage 3's own
        # threading/GPU work, or R4.a's GUI-exposed device/AMP settings,
        # regardless of what main_app.py's gather_config() wrote into
        # sequential_config.json - every one of those keys was silently
        # dropped right here. Fixed by forwarding the SAME full set
        # run_step1_batch.py's gp_kwargs already forwards, with THE SAME
        # per-key defaults (never re-decided independently - see the
        # blueprint §4 config schema table), so an old sequential_config.
        # json missing every one of these keys reproduces pre-ver4-4
        # behaviour exactly (I11). N_CPU_WORKERS/N_GPU_SLOTS are forwarded
        # for completeness/consistency with run_step1_batch.py's own
        # gp_kwargs (legacy Sequential mode calls GP() exactly once,
        # directly, with no PARALLEL/intra-batch context - see the
        # N_MODEL_WORKERS log-only note above for why intra-task fan-out
        # itself does not apply here - but GP() still resolves
        # n_cpu_workers/n_gpu_slots into its own compute-resource dict
        # regardless of PARALLEL, e.g. for W_OPT weighted-ensemble
        # gpu_slot() usage).
        USE_GPU_SKLEARN=cfg.get('USE_GPU_SKLEARN', False), N_JOBS=cfg.get('N_JOBS', -1),
        PLINK_THREADS=cfg.get('PLINK_THREADS', 1), R_BLAS_THREADS=cfg.get('R_BLAS_THREADS'),
        TORCH_DEVICE=cfg.get('TORCH_DEVICE'), CUDNN_BENCHMARK=cfg.get('CUDNN_BENCHMARK', True),
        USE_AMP=cfg.get('USE_AMP', False), N_CPU_WORKERS=cfg.get('N_CPU_WORKERS', 1),
        N_GPU_SLOTS=cfg.get('N_GPU_SLOTS'),
        R_BLAS_FOLLOWS_N_JOBS=cfg.get('R_BLAS_FOLLOWS_N_JOBS', True),
        TORCH_NUM_THREADS=cfg.get('TORCH_NUM_THREADS'),
        TORCH_DATALOADER_WORKERS=cfg.get('TORCH_DATALOADER_WORKERS', 0),
        GPU_EVAL_BATCH=cfg.get('GPU_EVAL_BATCH', 32),
        GPU_LD_R2=cfg.get('GPU_LD_R2', True),
        GPU_KERNEL_PRECOMPUTE=cfg.get('GPU_KERNEL_PRECOMPUTE', True),
        # ver4-4 Stage 6 (blueprint §10 Stage 6 checklist item 5 - "forward
        # every new key into GP()"): default True per §4a - see
        # genomic_prediction.py's own HP_TUNE dispatch site for the
        # disclosed caveat on what this can currently achieve.
        HP_TUNE_PARALLEL_TRIALS=cfg.get('HP_TUNE_PARALLEL_TRIALS', True),
        # ver4-5 R1 (blueprint §3.6) - same forward-every-new-key
        # discipline, same legacy-preserving defaults.
        HP_TUNE_BAYES_BATCH=cfg.get('HP_TUNE_BAYES_BATCH', True),
        HP_TUNE_BAYES_BATCH_MAX=cfg.get('HP_TUNE_BAYES_BATCH_MAX', 8),
        HP_TUNE_BAYES_LIAR=cfg.get('HP_TUNE_BAYES_LIAR', 'max'),
        HP_TUNE_PARALLEL_RESTARTS=cfg.get('HP_TUNE_PARALLEL_RESTARTS', True),
        # Update ID ver4-6 (blueprint §4/§10) - eight further keys, same
        # forward-every-new-key discipline, same legacy-preserving (or,
        # where §4 documents a deliberate default flip, §4's own
        # documented new-default) values - see pipeline_utils.
        # resolve_compute_resources()'s own docstring for each key's full
        # meaning.
        HP_TUNE_BAYES_DOMAIN_REDUCTION=cfg.get('HP_TUNE_BAYES_DOMAIN_REDUCTION', 'auto'),
        HP_TUNE_BAYES_MULTI_FIDELITY=cfg.get('HP_TUNE_BAYES_MULTI_FIDELITY', False),
        HP_TUNE_FAST=cfg.get('HP_TUNE_FAST'),   # ver4-40 R1 (None = legacy tuning)
        HP_TUNE_PRESET_FILE=cfg.get('HP_TUNE_PRESET_FILE'),   # ver4-61 R6 (None = no preset)
        HP_TUNE_WARM_START=cfg.get('HP_TUNE_WARM_START', False),
        HP_TUNE_SELECTION_MARGIN=cfg.get('HP_TUNE_SELECTION_MARGIN', 0.02),
        HP_TUNE_VALID_REPEATS=cfg.get('HP_TUNE_VALID_REPEATS', 1),
        HP_TUNE_SCOPE=cfg.get('HP_TUNE_SCOPE', 'per_task'),
        W_OPT_ANALYTIC_SEED=cfg.get('W_OPT_ANALYTIC_SEED', False),
        W_OPT_VALIDATION_FLOOR=cfg.get('W_OPT_VALIDATION_FLOOR', False),
        W_OPT_SIMPLEX_SEARCH=cfg.get('W_OPT_SIMPLEX_SEARCH', True),
        # Update ID ver4-5, R2 Stage 11 - see main_app.py's own
        # MODEL_AVAILABILITY_STRICT note for the full rationale.
        MODEL_AVAILABILITY_STRICT=cfg.get('MODEL_AVAILABILITY_STRICT', True),
        # Update ID ver4-9, R7 - same forward-every-new-key discipline;
        # default 'gzip' matches GP()'s own default and every other new
        # RESULT_COMPRESSION-aware entry point in this codebase.
        RESULT_COMPRESSION=cfg.get('RESULT_COMPRESSION', 'gzip'),
        # Update ID ver4-12 (design blueprint SS4 config schema delta) -
        # five further keys, same forward-every-new-key discipline, same
        # legacy-preserving defaults (I11): a ver4-11 sequential_config.json
        # lacking every one of these reproduces ver4-11 behaviour exactly -
        # ASREML off, phenotype file handling and population derivation
        # both unchanged.
        ASREML=cfg.get('ASREML'),
        ASREML_STRICT=cfg.get('ASREML_STRICT', True),
        # Requirements.md item 4: per-phenotype 'higher'/'lower' direction for the
        # always-computed Top-k Coincidence / Czekanowski metrics (independent of ASReml).
        METRIC_DIRECTIONS=cfg.get('METRIC_DIRECTIONS'),
        # ver4-33 (design blueprint SS4): None for every pre-ver4-33 config
        # (-> ver4-32 behaviour, I11).
        ACCELERATION=acceleration_from_cfg(cfg),
        PHENOTYPE_FILE_MODE=cfg.get('PHENOTYPE_FILE_MODE', 'analysed'),
        PHENOTYPE_TRAIT_COLUMNS=cfg.get('PHENOTYPE_TRAIT_COLUMNS'),
        POPULATION_SOURCE=cfg.get('POPULATION_SOURCE'),
        # Requirements.md (between-prediction validation-set patch),
        # items 1-4 - same forward-every-new-key discipline. Every
        # default reproduces GP()'s own pre-patch behaviour exactly
        # (I11): BETWEEN_VALIDATION absent -> None -> validation-presence
        # for a 'between' run is still inferred from W_OPT alone, and
        # EXCLUDED_POPULATION_COMBINATIONS absent -> None -> excludes
        # nothing.
        BETWEEN_VALIDATION=cfg.get('BETWEEN_VALIDATION'),
        BETWEEN_VALIDATION_SOURCE=cfg.get('BETWEEN_VALIDATION_SOURCE'),
        BETWEEN_RATIO=cfg.get('BETWEEN_RATIO'),
        EXCLUDED_POPULATION_COMBINATIONS=cfg.get('EXCLUDED_POPULATION_COMBINATIONS'),
        # ver4-56 (blueprint R1): log-only usage records; an absent key means True (I11,
        # justified in the blueprint: log-only, overhead < 0.1 %). Resolved + logged by
        # pipeline_utils.resolve_compute_resources().
        RESOURCE_TELEMETRY=cfg.get('RESOURCE_TELEMETRY', True),
        RESOURCE_TELEMETRY_INTERVAL_S=cfg.get('RESOURCE_TELEMETRY_INTERVAL_S', 0.5),
    )

    # ver4-4 Stage 6 (blueprint §2.3.2 R3.g, "Sequential intra-batch
    # routing (highest risk)"): gated behind SEQUENTIAL_INTRA_BATCH,
    # default False - the blueprint's own one deliberate "not default on"
    # exception (§4a), since this is a control-flow change, not a
    # numerics one, and the naming/dispatch mechanics below were only
    # worked out (not merely assumed) during this implementation - see
    # intra_batch_parallel.py's own PARALLEL docstring note on
    # run_batch_with_intra_batch_parallelism() for the resolved PC-2
    # pre-check.
    sequential_intra_batch = bool(cfg.get('SEQUENTIAL_INTRA_BATCH', False))
    _n_cpu_workers_requested = int(cfg.get('N_CPU_WORKERS', 1) or 1)
    _total_tasks = None

    if sequential_intra_batch and _n_cpu_workers_requested <= 1:
        print(f"[run_sequential] SEQUENTIAL_INTRA_BATCH=True but N_CPU_WORKERS="
              f"{_n_cpu_workers_requested} - nothing to fan out across; running GP()'s own "
              f"ordinary, fully serial per-task loop unchanged (byte-identical to "
              f"SEQUENTIAL_INTRA_BATCH=False).")
        sequential_intra_batch = False

    if sequential_intra_batch:
        # ver4-4 Stage 6 (R3.g): an AUTHORITATIVE (never approximate) total
        # task count is required here - `run_batch_with_intra_batch_
        # parallelism()` dispatches EXACTLY `total_tasks` isolated
        # single-task GP() sub-calls (task indices [0, total_tasks)); an
        # UNDER-count would silently DROP real scenarios from this run
        # (the ones at indices >= total_tasks are simply never dispatched,
        # with no error - see that function's own PARALLEL docstring
        # note), which is a far worse failure mode than merely falling
        # back to the slower, always-correct serial path. So this branch
        # computes the count from the SAME cheap, header/column-only reads
        # GP() itself would use to resolve POPULATION/PHENOTYPE (never
        # duplicated logic - resource_profiles.compute_total_tasks()'s own
        # formula is verified, by direct comparison against genomic_
        # prediction.py::GP()'s own `sample` table construction, to be an
        # EXACT match for both 'within' and 'between' scenarios, not
        # merely an estimate - the "approximation" language in that
        # function's own docstring refers to a THEORETICAL edge case this
        # verification did not find in the actual code), and refuses to
        # attempt fan-out (falling back to the always-correct legacy path
        # instead, with a clear NOTE) whenever any input to that formula
        # can't be determined cheaply and reliably.
        # Update ID ver4-12, R2 (design blueprint SS2.2, Appendix A finding
        # A3): GP() now resolves POPULATION from the PHENOTYPE file's own
        # (possibly DERIVED) population column for BOTH genotype formats -
        # previously CSV read it from the GENOTYPE file. Reading the
        # genotype file's own column here (the pre-ver4-12 'else' branch)
        # would now silently MISCOUNT population whenever POPULATION_SOURCE
        # derives population from something other than a literal
        # 'population' column - exactly the under-count this whole block's
        # own comment warns is worse than falling back to the legacy path.
        # `count_unique_populations()`'s own `population_source` argument
        # (`None` reproduces ver4-11 exactly - I11) is the SAME single
        # authority GP() itself calls via `derive_population_series()`.
        _n_population = count_unique_populations(cfg['PHENOTYPE_FILE_NAME'], cfg.get('POPULATION_SOURCE'))

        if isinstance(cfg['PHENOTYPE'], list):
            _n_phenotype = len(cfg['PHENOTYPE'])
        else:
            # PHENOTYPE == 'all' - GP() itself resolves this via
            # resolve_trait_columns() (R3), which is list_phenotype_
            # columns() unchanged under the default PHENOTYPE_FILE_MODE=
            # 'analysed' (I11), and PHENOTYPE_TRAIT_COLUMNS under 'raw'.
            # Falls back to 0 (triggering the always-correct legacy path
            # just below) rather than raising, since a headless config
            # error here should not crash resource sizing - GP() itself
            # will raise with a much more specific message when it
            # actually loads the file.
            try:
                _n_phenotype = len(resolve_trait_columns(cfg, cfg['PHENOTYPE_FILE_NAME']))
            except Exception:
                _n_phenotype = 0

        if _n_population is None or _n_population == 0 or _n_phenotype == 0:
            print(f"[run_sequential] NOTE: SEQUENTIAL_INTRA_BATCH=True but this run's own total "
                  f"task count could not be determined reliably up front (n_population="
                  f"{_n_population!r}, n_phenotype={_n_phenotype!r}) - falling back to GP()'s "
                  f"own ordinary, fully serial per-task loop rather than risk silently dropping "
                  f"scenarios from an under-counted fan-out. Check that GENOTYPE_FILE_NAME/"
                  f"PHENOTYPE_FILE_NAME are readable at this path.")
            sequential_intra_batch = False
        else:
            _total_tasks = compute_total_tasks(
                cfg['SCENARIO'], _n_population, _n_phenotype, len(ratio), cfg['ITER_NUM'],
                w_opt_enabled=cfg.get('W_OPT') is not None,
            )
            if _total_tasks <= 0:
                print(f"[run_sequential] NOTE: SEQUENTIAL_INTRA_BATCH=True but this run's own "
                      f"total task count resolved to {_total_tasks} (n_population={_n_population}, "
                      f"n_phenotype={_n_phenotype}, scenario={cfg['SCENARIO']!r}) - nothing to fan "
                      f"out; falling back to GP()'s own ordinary per-task loop, which will report "
                      f"the same zero-scenario condition itself.")
                sequential_intra_batch = False

    _t0 = time.time()
    if sequential_intra_batch:
        # ver4-4 Stage 6 (R3.g): imported here, not at module load time,
        # for the SAME reason the R/rpy2-dependent engine imports below
        # are deferred - intra_batch_parallel.py imports pipeline_utils'
        # own configure_r_environment()/init_rpy2_conversion() at ITS
        # module level too, so importing it before this script's own R
        # environment setup above would risk the identical "R_HOME set
        # too late" failure shape Patch 9 Bug 3 already documented for a
        # different module.
        from intra_batch_parallel import run_batch_with_intra_batch_parallelism
        from batch_reader import ResultSet
        gp_kwargs['PARALLEL'] = {'batch_id': 0, 'batch_size': _total_tasks}
        print(f"[run_sequential] SEQUENTIAL_INTRA_BATCH=True: fanning this run's own "
              f"{_total_tasks} task(s) out across up to {_n_cpu_workers_requested} worker "
              f"process(es) via intra_batch_parallel, instead of GP()'s own serial per-task loop.")
        run_batch_with_intra_batch_parallelism(
            gp_kwargs, _n_cpu_workers_requested,
            n_gpu_slots=int(cfg.get('N_GPU_SLOTS') or 0),
            sequential_mode=True,
        )
        # ver4-4 Stage 6 (R3.g): run_batch_with_intra_batch_parallelism()
        # writes to disk and returns nothing (unlike GP()'s own in-memory
        # accumulators) - rebuild this script's own 8 result values from
        # the merged, UNSUFFIXED files it just wrote, following
        # run_step2_assemble.py:143-223's existing ResultSet(source=
        # 'combined') consumption pattern exactly, so every plotting call
        # below needs no further change regardless of which route
        # produced these values.
        #
        # DISCLOSED semantic note: `population`/`phenotype` below are the
        # DISTINCT values actually PRESENT in this run's own merged
        # Metric.csv (via ResultSet's own pd.unique() on the metric frame)
        # - i.e. scenarios that actually produced at least one result row -
        # rather than GP()'s own direct-call return, which echoes back the
        # full, nominal POPULATION/PHENOTYPE lists it resolved up front
        # regardless of whether every one of them ended up producing
        # output (e.g. a population/phenotype combination skipped entirely
        # by the MIN_DATA_POINTS gate). This exactly matches how Parallel
        # mode's own run_step2_assemble.py has always behaved - ordinary
        # Sequential mode (SEQUENTIAL_INTRA_BATCH=False, the default) is
        # completely unaffected by this note.
        result = ResultSet(cfg['RESULT_NAME'], source='combined')
        metrics = result.metric
        predicted_result_train = result.prediction('train')
        predicted_result_test = result.prediction('test')
        effect = result.effect()
        interactions = result.interactions()
        population = result.population
        phenotype = result.phenotype
        attention = result.attention()
        asreml_varcomp_for_summary = result.asreml_varcomp()
    else:
        (metrics, predicted_result_train, predicted_result_test, effect,
         interactions, population, phenotype, attention) = GP(**gp_kwargs)
        # ver4-25, R10 (design blueprint stage 3): `GP()`'s own return
        # contract (architecture doc SS7) is an 8-tuple and is NOT
        # extended here - `ASReml_variance_components.csv` is instead
        # read back from disk exactly like the SEQUENTIAL_INTRA_BATCH
        # branch above already does via `ResultSet`, since `GP()` itself
        # already wrote it (or an empty placeholder) before returning.
        # Cheap even here - `ResultSet.__init__` only eagerly reads the
        # small `Metric.csv`; `asreml_varcomp()` is read lazily, on this
        # one call, and never again.
        asreml_varcomp_for_summary = ResultSet(cfg['RESULT_NAME'], source='combined').asreml_varcomp()
    print(f'[run_sequential] Genomic prediction finished (took {time.time() - _t0:.1f}s). '
          f'See the [GP] lines above for LD-pruning and per-model detail.')

    # Requirements.md item 3 (multi-option update): two or more selected
    # genotype-effect options -> every result file carries the leading
    # option column and the reporting chain runs once per option (own
    # sub-folder) plus the cross-option plots. A single-option run takes
    # the unchanged chain below (I11).
    from reporting_pipeline import is_multi_option_run, run_multi_option_reports
    if is_multi_option_run(cfg):
        from batch_reader import ResultSet
        run_multi_option_reports(cfg, ResultSet(cfg['RESULT_NAME'], source='combined'),
                                 log=lambda m: print(f'[run_sequential] {m}'))
        return

    # Update ID 3, R3: the Metric_summary.xlsx workbook (or its CSV
    # degrade), written immediately once GP() returns and before the first
    # plot - same position in every one of R3's four call sites (blueprint
    # §R3.9). Sequential mode gets this too (§R3.6) - its Metric.csv has
    # the identical schema, from the identical GP() code path, as Parallel
    # mode's combined Metric.csv.
    #
    # model_labels is what metric_plot() groups/filters by, AND
    # (Requirements.md item 8) what write_metric_summary() now uses to
    # order the metric summary's own model columns identically. Ordinarily
    # this is just every selected model name plus every weighted-ensemble
    # method name (cfg['MODEL'] never gets rewritten with tuning-algorithm
    # suffixes - that expansion only ever happens inside GP() itself). But
    # when HP_TUNE is in use, the 'model' column metrics actually contains
    # can include suffixed variants (e.g. 'RF__Grid', 'RF__Bayesian')
    # and/or per-method ensemble labels (e.g. 'ensemble__Grid', 'Nelder
    # Mead__Bayesian') that this reconstruction from cfg alone can't
    # predict - so in that case, read the real label set directly out of
    # metrics itself (the actual source of truth for what GP() produced)
    # instead of guessing. Computed here (moved up from just before
    # metric_plot() below) so both call sites use the exact same list.
    if cfg.get('HP_TUNE'):
        model_labels = pd.unique(metrics['model']).tolist()
    else:
        model_labels = cfg['MODEL'] + cfg['W_OPT'] if cfg['W_OPT'] is not None else cfg['MODEL']
        model_labels = list(model_labels) + iterative_labels(cfg)   # ver4-57
        model_labels = list(model_labels) + super_learner_labels(cfg)   # ver4-77

    _t0 = time.time()
    _summary_paths = write_metric_summary(
        metrics, cfg['RESULT_NAME'], create=cfg.get('METRIC_SUMMARY_CREATE', True),
        model_order=model_labels, asreml_varcomp=asreml_varcomp_for_summary,
        label_style=cfg.get('MODEL_LABEL_STYLE', 'full'),
    )
    if _summary_paths:
        print(f"[run_sequential] Metric summary written: {', '.join(_summary_paths)} "
              f"(took {time.time() - _t0:.1f}s).")

    # Update ID ver4-9, R5: the Diversity Prediction Theorem summary,
    # written right after the metric summary (same position as every
    # other R3/R5 call site). GP() itself returns no 'weight' (its own
    # 8-tuple - see genomic_prediction.py::GP()'s own return statement),
    # so both Prediction_result_test.csv and Weight.csv are read fresh
    # off disk here via load_combined() - the SAME combined-file path
    # authority ResultSet uses internally, so this never disagrees with
    # what GP() actually wrote, regardless of which route (ordinary
    # serial loop or SEQUENTIAL_INTRA_BATCH) produced it.
    _t0 = time.time()
    # ver4-25, R8 (design blueprint SS2, Stage 6; Stage 6 completion for
    # 'full' mode): build a per-scenario BLUE-reliability weight resolver
    # from the SAME resume-safe sidecar `models/ensemble.py`'s own
    # naive-ensemble finalisation already reads
    # (`asreml_accuracy_metrics.ctx_resolver_from_frame()`), PLUS the
    # 'full'-mode test-weight-matrix accumulator (only ever non-empty for
    # a 'full'-mode run); `None` whenever NEITHER sidecar was ever
    # written at all (ASReml/blue_weighting off for this whole run),
    # which keeps `build_dpt_terms()` a byte-identical, unweighted
    # computation (I11).
    _asreml_accuracy_ctx_for_dpt = load_combined(cfg['RESULT_NAME'], 'asreml_accuracy_ctx')
    _asreml_test_weights_for_dpt = load_combined(cfg['RESULT_NAME'], 'asreml_test_weights')
    _dpt_weight_resolver = None
    if ((_asreml_accuracy_ctx_for_dpt is not None and _asreml_accuracy_ctx_for_dpt.shape[0] != 0)
            or (_asreml_test_weights_for_dpt is not None and _asreml_test_weights_for_dpt.shape[0] != 0)):
        import asreml_accuracy_metrics as _aam
        _dpt_weight_resolver = _aam.ctx_resolver_from_frame(
            _asreml_accuracy_ctx_for_dpt, _asreml_test_weights_for_dpt)

    _dpt_terms = build_dpt_terms(
        load_combined(cfg['RESULT_NAME'], 'result_test'),
        load_combined(cfg['RESULT_NAME'], 'weight'),
        metrics,
        sample_weight_resolver=_dpt_weight_resolver,
    )
    _dpt_paths = write_dpt_summary(
        _dpt_terms, cfg['RESULT_NAME'], create=cfg.get('DPT_SUMMARY_CREATE', True),
        # Requirements.md item 1: ensemble labels only - see
        # diversity_summary.dpt_model_order()'s own docstring for why the
        # full model_labels list (as used for the metric summary/violin
        # plots) would otherwise reintroduce a permanently-blank column
        # per single-prediction model.
        model_order=dpt_model_order(model_labels),
        label_style=cfg.get('MODEL_LABEL_STYLE', 'full'),
    )
    if _dpt_paths:
        print(f"[run_sequential] Diversity Prediction Theorem summary written: "
              f"{', '.join(_dpt_paths)} (took {time.time() - _t0:.1f}s).")

    # ver4-99 (R2) / ver4-103 (X03): marker / pair stability across iterations. Off unless
    # STABILITY_SUMMARY_CREATE is true; never fatal (the call logs and returns on any failure).
    from stability_summary import run_stability_report as _run_stability_report
    _run_stability_report(cfg, cfg['RESULT_NAME'], lambda: load_combined(cfg['RESULT_NAME'], 'effect'), print)

    # Requirement 6: once every prediction scenario has finished, average
    # every per-scenario LD decay CSV (written above, inside GP(), if the
    # LD decay plot feature was enabled) into one plot per population x
    # phenotype combination. Safe to call unconditionally - a no-op (with
    # an informational log line) whenever the feature wasn't used.
    _t0 = time.time()
    _n_ld_decay_plots = average_and_plot_ld_decay(cfg['RESULT_NAME'])
    if _n_ld_decay_plots:
        print(f'[run_sequential] {_n_ld_decay_plots} average LD decay plot(s) generated '
              f'(took {time.time() - _t0:.1f}s).')

    if 'GAT_fully_connected' in cfg['MODEL'] or 'GAT_prior_knowledge' in cfg['MODEL'] or any(m.startswith('GAT_biological_prior_knowledge') for m in cfg['MODEL']):
        _t0 = time.time()
        attention_distribution(attention, cfg['RESULT_NAME'], 10)
        print(f'[run_sequential] Attention distribution plots generated (took {time.time() - _t0:.1f}s).')

    if cfg.get('METRIC_PLOT_CREATE', True):
        _t0 = time.time()
        metric_plot(metrics.copy(), model_labels, cfg['RESULT_NAME'], cfg['SCENARIO'], _plot_style.with_extra_formats(cfg.get('METRIC_PLOT_CONFIG'), cfg), PLOT_DPI=_metric_plot_dpi)
        print(f'[run_sequential] Metric plots generated (took {time.time() - _t0:.1f}s).')

    # Requirements.md (BLUEs/BLUPs accuracy-metrics phase), item 8 - the
    # five accuracy-metric visualisations, gated on their own dedicated
    # Tab 5 checkbox (default False - a new plot type, opt-in rather than
    # appearing unannounced, mirroring WEIGHT_PLOT_CREATE's own default-
    # off precedent). A no-op regardless if Metric.csv never gained the
    # new columns in the first place (accuracy_metrics_plot()'s own
    # column-presence checks).
    if any_accuracy_plot_requested(cfg):
        _t0 = time.time()
        accuracy_metrics_plot(metrics.copy(), model_labels, cfg['RESULT_NAME'], cfg['SCENARIO'],
                               cfg.get('ACCURACY_METRICS_PLOT_CONFIG'),
                               ACCURACY_BUBBLE_PLOT_CREATE=cfg.get('ACCURACY_BUBBLE_PLOT_CREATE', False),
                               prediction_test=predicted_result_test, cfg=cfg,
                               effect=effect)
        print(f'[run_sequential] Accuracy-metric plots generated (took {time.time() - _t0:.1f}s).')

    # Update ID ver4-9, R6: WEIGHT_PLOT_CREATE default False when absent
    # (a standalone/legacy JSON config predating this key) - a
    # behaviour-preserving default (I11), unlike DPT_SUMMARY_CREATE/
    # METRIC_SUMMARY_CREATE's own default-True (both purely additive
    # artefacts that overwrite nothing). naive_models is resolved fresh
    # here (the SAME resolve_ensemble_members() call R5 already makes) -
    # None when this run's own Metric.csv has no naive-ensemble label at
    # all, in which case weight_plot() simply omits that bar.
    if cfg.get('WEIGHT_PLOT_CREATE', False):
        _t0 = time.time()
        _weight_for_plot = load_combined(cfg['RESULT_NAME'], 'weight')
        _members_for_plot = resolve_ensemble_members(
            metrics, load_combined(cfg['RESULT_NAME'], 'result_test'), _weight_for_plot,
        )
        _naive_for_plot = next((info['models'] for info in _members_for_plot.values() if info['is_naive']), None)
        _weight_plot_paths = weight_plot(
            _weight_for_plot, model_labels, cfg['RESULT_NAME'], cfg['SCENARIO'],
            _plot_style.with_extra_formats(cfg.get('WEIGHT_PLOT_CONFIG'), cfg), PLOT_DPI=cfg.get('WEIGHT_PLOT_DPI', 300),
            naive_models=_naive_for_plot,
        )
        if _weight_plot_paths:
            print(f"[run_sequential] Weight plot(s) generated: {', '.join(_weight_plot_paths)} "
                  f"(took {time.time() - _t0:.1f}s).")

    # ver4-4 R7.g: apply the SAME float32 downcast batch_reader.ResultSet.
    # effect() offers Parallel mode's plotting path - Sequential mode has
    # no ResultSet at all (`effect` is GP()'s own in-memory accumulator),
    # so this is done directly here, once, and the SAME downcast frame is
    # reused by both scatter_plot() and circos_plot() below. Marker_effect.
    # csv itself (already written to disk by GP(), long before this point)
    # is completely unaffected - this only ever touches an in-memory COPY
    # used for plotting. When PLOT_EFFECT_FLOAT32 is off, `effect_for_plots`
    # is the SAME object as `effect` (no copy) - byte-identical to this
    # script's pre-ver4-4 behaviour.
    if _plot_effect_float32 and effect.shape[1] > _EFFECT_METADATA_WIDTH:
        _marker_cols = effect.columns[_EFFECT_METADATA_WIDTH:]
        effect_for_plots = effect.copy()
        effect_for_plots[_marker_cols] = effect_for_plots[_marker_cols].astype('float32')
    else:
        effect_for_plots = effect

    if cfg.get('SCATTER_CREATE', True):
        _t0 = time.time()
        # scatter_plot() selects columns out of predicted_result_test/effect
        # by name (see models/ensemble.py-style 'model_selected' pattern),
        # so it needs the same real, possibly-suffixed names as
        # model_labels above - NOT cfg['MODEL'] (never suffixed) - whenever
        # HP_TUNE is in play. Unlike metric_plot, scatter_plot was never
        # given weighted-ensemble (W_OPT) method labels even before this
        # change, so those are filtered back out here to match that existing
        # scope exactly.
        if cfg.get('HP_TUNE'):
            # ver4-57: 'Iterative ensemble' joins the weighted labels filtered out here - with HP_TUNE
            # model_labels is read from Metric.csv, which now carries that label when it is on.
            # ver4-77: 'Super learner' joins them for the same reason.
            _wopt_prefixes = ('Linear transformation', 'Nelder Mead', 'Bayesian optimisation', 'Analytic least-squares',
                              'Iterative ensemble', 'Super learner')
            scatter_models = [m for m in model_labels
                               if not any(m == p or m.startswith(p + '__') for p in _wopt_prefixes)]
        else:
            scatter_models = cfg['MODEL']
        # ver4-4 R6: QTL_WINDOW/QTL_WINDOW_MODE default to 0.0/'all_in_window'
        # (today's exact-match-only behaviour) for a config predating this
        # feature - cfg.get(...), not cfg[...], so an old sequential_config.json
        # loaded from disk still runs unchanged. MARKER_INFO is already a
        # required, always-populated config key (see gather_config()'s own
        # CIRCOS_CREATE-gated validation) - passed through unconditionally;
        # it's a no-op for a legacy 2-column QTL file or QTL=None.
        scatter_plot(scatter_models, phenotype, predicted_result_test, effect_for_plots,
                     cfg['QTL'], _plot_style.with_extra_formats(cfg['SCATTER_CONFIG'], cfg), cfg['RESULT_NAME'],
                     MARKER_INFO=cfg.get('MARKER_INFO'),
                     QTL_WINDOW=cfg.get('QTL_WINDOW', 0.0),
                     QTL_WINDOW_MODE=cfg.get('QTL_WINDOW_MODE', 'all_in_window'),
                     PLOT_DPI=_scatter_plot_dpi,
                     # Requirements.md item 4: cfg.get(...) with a False
                     # default reproduces a pre-existing sequential_config.json's
                     # exact (level-based) colouring unchanged.
                     COLOR_BY_POPULATION=cfg.get('SCATTER_COLOR_BY_POPULATION', False),
                     # Requirements.md item 5 (this session): cfg.get(...)
                     # with a True default reproduces a pre-existing
                     # sequential_config.json's exact (always-normalised)
                     # marker-effect scaling unchanged.
                     NORMALIZE_MARKER_EFFECT=cfg.get('SCATTER_NORMALIZE_MARKER_EFFECT', True))
        print(f'[run_sequential] Scatter plot matrix generated (took {time.time() - _t0:.1f}s).')

    if cfg.get('CIRCOS_CREATE', True):
        _t0 = time.time()
        # Requirement (bugfix - see run_step2_assemble.py's identical
        # comment): this script never had the 'Chromosome/gene lengths
        # are the same for every population' broadcast logic
        # main_app.py's own interactive GUI code has - ticking that
        # checkbox in the GUI never actually took effect for a
        # Sequential job submitted this way.
        _chrom_info_path, _gene_info_path = cfg['CHROMOSOME_INFO'], cfg['GENE_INFO']
        print(f"[run_sequential] Broadcast checkbox: {cfg.get('CIRCOS_BROADCAST_POPULATION')} | "
              f"population from results: {list(population)} | gene info path set: {bool(_gene_info_path)}")
        if cfg.get('CIRCOS_BROADCAST_POPULATION'):
            _target_pop_source = population
            if cfg['SCENARIO'] == 'between':
                _target_pop_source = [
                    p.split('->')[-1] if isinstance(p, str) and '->' in p else p
                    for p in population
                ]
            _target_pops = [_clean_population_label(p) for p in _target_pop_source] + ['all']
            _chrom_info_path = _broadcast_population_info(_chrom_info_path, _target_pops, 'chrom')
            if _gene_info_path:
                _gene_info_path = _broadcast_population_info(_gene_info_path, _target_pops, 'gene')
            print(f"[run_sequential] Broadcast target populations: {_target_pops} | "
                  f"broadcast chrom file: {_chrom_info_path} | broadcast gene file: {_gene_info_path}")
        # Update ID ver4-25, R12: `interactions` above is GP()'s own
        # in-memory accumulator, which never carries k>=3 rows (see
        # genomic_prediction.py's own comment on why GP()'s positional
        # return tuple was NOT extended) - read the durable-only
        # Interaction_higher_order.csv off disk here via load_combined(),
        # the SAME pattern this script already uses for Weight.csv (see
        # `_weight_for_plot` above) for the identical reason.
        _interactions_hok = load_combined(cfg['RESULT_NAME'], 'interactions_hok')
        circos_plot(effect_for_plots, interactions, cfg['MARKER_INFO'], _chrom_info_path,
                    _gene_info_path, population, phenotype, _plot_style.with_extra_formats(cfg['CIRCOS_CONFIG'], cfg),
                    cfg['END_ADJUST'], cfg['WINDOW'], cfg['CYTOBAND_COLORMAP'],
                    cfg['RESULT_NAME'], attention, cfg['SCENARIO'], cfg['ASCENDING'],
                    gene_adjust=cfg.get('GENE_ADJUST', 0), plot_dpi=_circos_plot_dpi,
                    interactions_hok=_interactions_hok,
                    # Update ID ver4-25, R12: absent from every config that
                    # predates this feature, so `.get(..., (2,))`/
                    # `.get(..., 'per_order')` reproduce today's k=2-only,
                    # unconditional render exactly (I11/AC12.3).
                    interaction_orders=cfg.get('CIRCOS_INTERACTION_ORDERS', (2,)),
                    higher_order_layout=cfg.get('CIRCOS_HIGHER_ORDER_LAYOUT', 'per_order'),
                    # Performance (multi-CPU circos-plot rendering): absent
                    # from every config that predates this feature, so
                    # `.get(..., 1)` reproduces today's fully-serial
                    # rendering exactly - see circos_plot.py::
                    # _resolve_circos_plot_workers()'s own docstring for
                    # the daemon-safety/capping this value is still
                    # subject to even when set above 1.
                    n_workers=cfg.get('CIRCOS_PLOT_WORKERS'),
                    # N_JOBS fix: when CIRCOS_PLOT_WORKERS is absent
                    # (None), circos_plot() now takes its worker count
                    # from N_JOBS instead of silently rendering serially
                    # - see circos_plot._resolve_circos_plot_workers().
                    n_jobs=cfg.get('N_JOBS'))
        print(f'[run_sequential] Circos plot generated (took {time.time() - _t0:.1f}s).')
    print('[run_sequential] Pipeline completed successfully.')


if __name__ == '__main__':
    main()
