#!/usr/bin/env python3
"""
EasiGP - headless, non-interactive runner for a single Step 1 (parallel
model-fitting) batch. This script is meant to be launched directly by an
HPC job scheduler (Slurm/PBS) as the command for each task in an array job -
NOT through Streamlit - so that submitting an array of thousands of jobs
never requires configuring the GUI more than once.

Typical workflow
-----------------
1. In the Streamlit GUI (streamlit_app.py), go to Parallel > Step 1,
   configure everything as usual (data paths, models, hyperparameters,
   batch size, total number of batches, etc.), then click
   "Save configuration for HPC array job". This writes a JSON file
   (by default: Result/<RESULT_NAME>/step1_config.json) capturing every
   setting EXCEPT the batch ID, which is different for every task and is
   therefore resolved at run time instead (see below). The GUI also
   generates ready-to-submit slurm_step1_array.sh / pbs_step1_array.sh
   scripts for you.

2. Submit the array job, e.g.:

       Slurm:  sbatch slurm_step1_array.sh
       PBS:    qsub  pbs_step1_array.sh

   Each task in the array automatically calls:

       python run_step1_batch.py --config Result/<RESULT_NAME>/step1_config.json

   and this script resolves its own batch ID from whichever scheduler
   environment variable is set (SLURM_ARRAY_TASK_ID, or
   PBS_ARRAY_INDEX / PBS_ARRAYID) - no manual input, no GUI, per task.

3. You can also run (or re-run) a single batch manually, overriding the
   batch ID explicitly:

       python run_step1_batch.py --config Result/<RESULT_NAME>/step1_config.json --batch-id 0

4. To deliberately redo one specific batch from scratch - discarding any
   previously saved checkpoint/result files (and, if intra-batch/intra-task
   parallelism produced one, that batch's own scratch folder of per-task/
   per-model-group isolated results) for that batch ID, rather than the
   default checkpoint/resume behaviour of picking up where a previous
   attempt left off - add --overwrite:

       python run_step1_batch.py --config Result/<RESULT_NAME>/step1_config.json --batch-id 0 --overwrite

   (Update ID 2, Defect D4 fix - see PATCH_NOTES_D2.md. Checkpoint/resume
   remains the default for a routine retry after a crash/timeout; this flag
   is only for the deliberate "no, really, start this one over" case.)
"""

import argparse
import json
import sys
import time

import checkpoint_utils as _ckpt
from pipeline_utils import (
    configure_r_environment, init_rpy2_conversion,
    resolve_batch_id_from_env, restore_ratio, TimestampedWriter, make_run_log_path,
    apply_scratch_tmp_dir,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--config', required=True,
        help='Path to the step1_config.json produced by the GUI ("Generate and save job files").'
    )
    parser.add_argument(
        '--batch-id', type=int, default=None,
        help=(
            'Explicit batch ID for this run. If omitted (the normal case inside an '
            'array job), the batch ID is read automatically from SLURM_ARRAY_TASK_ID '
            'or PBS_ARRAY_INDEX/PBS_ARRAYID.'
        ),
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help=(
            'Force this batch to start completely FRESH: delete this batch\'s own '
            'checkpoint and result files, and (if present) its intra-batch/intra-task '
            'parallelism scratch folder, before running - rather than the default '
            'behaviour of automatically resuming from wherever a previous attempt left '
            'off. Also honoured as a config key ("OVERWRITE_RESULTS": true in the JSON '
            'itself), so a resubmission script can request it without a command-line '
            'change. Use this to deliberately redo a batch from scratch (e.g. its '
            'config changed in a way that does not itself change RESULT_NAME/batch_id); '
            'do NOT use it as a routine retry after a crash/timeout - checkpoint/resume '
            'already handles that automatically, without re-doing already-successful, '
            'possibly hours-long work.'
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, 'r', encoding='utf-8-sig') as f:
        cfg = json.load(f)

    batch_id = args.batch_id
    if batch_id is None:
        try:
            batch_id = resolve_batch_id_from_env()
        except RuntimeError as exc:
            raise SystemExit(str(exc))
    batch_size = cfg['PARALLEL']['batch_size']
    parallel = {'batch_id': batch_id, 'batch_size': batch_size}

    # Wrap stdout so every line this process prints gets a timestamp prefix
    # automatically, AND save it to Result/<RESULT_NAME>/logs/ - the same
    # place the GUI's own local-run option already saves a log (see
    # pipeline_utils.make_run_log_path) - in addition to whatever the job
    # scheduler itself captures to the array task's own job-output file.
    # The batch ID is included in the label (not just the log's own
    # timestamp) so two array tasks that happen to start within the same
    # second never collide on the same log filename.
    log_file_path = make_run_log_path(cfg['RESULT_NAME'], f'step1_batch{batch_id}')
    log_file = open(log_file_path, 'w', encoding='utf-8')
    sys.stdout = TimestampedWriter(sys.stdout, log_file)
    print(f'[run_step1_batch] Logging to {log_file_path}')

    # PATCH_NOTES (worker-crash fix): resolve an optional, persistent
    # scratch-temp override BEFORE any heavy import (configure_r_environment,
    # and - much further down - `from genomic_prediction import GP`, which
    # transitively imports torch/torch_geometric) gets a chance to run, and
    # before any loky/joblib worker process is ever spawned (those inherit
    # os.environ at spawn time) - see pipeline_utils.apply_scratch_tmp_dir()'s
    # own docstring for the failure mode this prevents. A no-op unless
    # EASIGP_SCRATCH_TMPDIR is set (or this config has a 'SCRATCH_TMP_DIR'
    # key), so existing configs/installations are unaffected by default.
    apply_scratch_tmp_dir(result_name=cfg['RESULT_NAME'], explicit=cfg.get('SCRATCH_TMP_DIR'))

    # Update ID 2, Defect D4 fix: an explicit --overwrite (or config-level
    # "OVERWRITE_RESULTS": true) means the person deliberately wants this
    # batch redone from scratch, NOT auto-resumed. Without this, there was
    # no way to ask for that short of manually hunting down and deleting
    # this batch's checkpoint/result files (and, worse, its intra-batch/
    # intra-task scratch folder - `_task_already_complete()`/
    # `_group_already_complete()` in intra_batch_parallel.py/
    # intra_task_parallel.py decide "already done" purely from that
    # scratch folder's own contents, WITHOUT genomic_prediction.py::GP()'s
    # own sample-fingerprint staleness check ever getting a chance to run
    # for a task/unit skipped this way - see PATCH_NOTES_D2.md) on the
    # cluster by hand. Deleting the checkpoint file alone would already be
    # enough to make GP() itself start fresh (its own
    # `_ckpt.load_checkpoint()` returns None -> it clears its own result
    # files automatically - see genomic_prediction.py), but that's true
    # only for a call GP() actually makes it to; the isolated scratch
    # folder must also be cleared so intra_batch_parallel.py/
    # intra_task_parallel.py's own pre-GP() "already complete" checks don't
    # keep skipping stale per-task/per-model-group results left over from
    # BEFORE whatever changed. Done for the OUTER batch_id's own state
    # regardless of which of the three dispatch routes below ends up
    # running - all three read/write the SAME outer checkpoint/result
    # files, and the two parallel routes also consult the same scratch
    # folder.
    overwrite = bool(args.overwrite or cfg.get('OVERWRITE_RESULTS', False))
    if overwrite:
        print(f"[run_step1_batch] --overwrite requested: discarding any previously saved "
              f"checkpoint/result files for batch {batch_id} (RESULT_NAME={cfg['RESULT_NAME']}), "
              f"and this batch's own intra-batch/intra-task scratch folder if one exists, "
              f"before running - this batch will be redone from scratch rather than resumed.")
        _ckpt.clear_checkpoint(cfg['RESULT_NAME'], batch_id, True)
        _ckpt.clear_result_files(cfg['RESULT_NAME'], batch_id, True)
        from intra_batch_parallel import cleanup_isolated_folders as _wipe_scratch_folder
        _wipe_scratch_folder(cfg['RESULT_NAME'], batch_id)

    configure_r_environment(
        cfg.get('R_PATH'), r_blas_threads=cfg.get('R_BLAS_THREADS'),
        r_max_ppsize=cfg.get('R_MAX_PPSIZE', 500000),
    )
    # NOTE (Update ID 2, Defect D2 fix - see PATCH_NOTES_D2.md): unlike
    # configure_r_environment() above (which only sets environment
    # variables - always safe, no matter what runs next), init_rpy2_conversion()
    # actually EMBEDS/STARTS the R interpreter in THIS process the moment
    # it's called. rpy2/R is not fork-safe once R (or a threaded BLAS/OpenMP
    # library it's linked against) has started even one thread - forking
    # worker processes (as the N_MODEL_WORKERS/N_CPU_WORKERS routes below
    # do, via ProcessPoolExecutor) out of a parent that already initialised
    # R risks silently corrupting each child's heap/import state, which can
    # then surface unpredictably, and sometimes hours later, as a SIGBUS,
    # a nonsensical ImportError, or worse - exactly what production hit
    # running a real batch with N_CPU_WORKERS=4. Each worker process
    # already safely initialises its OWN, fresh rpy2/R instance itself (see
    # `_worker_init()` in intra_batch_parallel.py/intra_task_parallel.py) -
    # so init_rpy2_conversion() is now called HERE, in the parent, ONLY
    # in the plain-serial branch below, where the parent itself is about to
    # call GP() directly (no forking involved at all).

    ratio = restore_ratio(cfg['RATIO'], cfg['SCENARIO'])

    print(f"[run_step1_batch] RESULT_NAME={cfg['RESULT_NAME']} "
          f"batch_id={batch_id} batch_size={batch_size} models={cfg['MODEL']}")

    gp_kwargs = dict(
        GENOTYPE_FILE_NAME=cfg['GENOTYPE_FILE_NAME'], PHENOTYPE_FILE_NAME=cfg['PHENOTYPE_FILE_NAME'],
        MODEL=cfg['MODEL'], PHENOTYPE=cfg['PHENOTYPE'], RATIO=ratio, SAMPLE_NUM=cfg['ITER_NUM'],
        HPARAMETERS=cfg['HPARAMETERS'], R_PATH=cfg['R_PATH'], W_OPT=cfg['W_OPT'],
        RESULT_NAME=cfg['RESULT_NAME'], HYPERPARAMETERS_OPT=cfg['HYPERPARAMETERS_OPT'],
        SCENARIO=cfg['SCENARIO'], PARALLEL=parallel,
        LD_prune=cfg.get('LD_PRUNE'), RF_filter=cfg.get('RF_FILTER'),
        GENOTYPE_FORMAT=cfg.get('GENOTYPE_FORMAT', 'csv'), GENOTYPE_PLINK_PATH=cfg.get('GENOTYPE_PLINK_PATH', 'plink2'),
        OTHER_MODELS_MARKER_SOURCE=cfg.get('OTHER_MODELS_MARKER_SOURCE', 'full_or_filtered'),
        HP_TUNE=cfg.get('HP_TUNE'), HP_TUNE_ENSEMBLE_MODE=cfg.get('HP_TUNE_ENSEMBLE_MODE', 'per_method'),
        MIN_DATA_POINTS=cfg.get('MIN_DATA_POINTS', 100),
        # Phase 2, Requirement 6/7/8 - all optional, all default to the
        # exact same behaviour as before these keys existed.
        USE_GPU_SKLEARN=cfg.get('USE_GPU_SKLEARN', False), N_JOBS=cfg.get('N_JOBS', -1),
        PLINK_THREADS=cfg.get('PLINK_THREADS', 1), R_BLAS_THREADS=cfg.get('R_BLAS_THREADS'),
        TORCH_DEVICE=cfg.get('TORCH_DEVICE'), CUDNN_BENCHMARK=cfg.get('CUDNN_BENCHMARK', True),
        USE_AMP=cfg.get('USE_AMP', False), N_CPU_WORKERS=cfg.get('N_CPU_WORKERS', 1),
        N_GPU_SLOTS=cfg.get('N_GPU_SLOTS'),
        # Update ID 2 (R2) - Test Report D1 fix: forward GPU_SLOTS_PER_DEVICE
        # from the config JSON through to GP() (-> resolve_compute_resources())
        # so a hand-edited value actually takes effect at run time, rather
        # than only ever being consumed by resource_profiles.estimate_resources()
        # (the GUI advisor). Default 1 matches every config written before
        # this key existed.
        GPU_SLOTS_PER_DEVICE=cfg.get('GPU_SLOTS_PER_DEVICE', 1),
        # ver4-4 Stage 5 (blueprint §10 Stage 5 checklist / §2.4.3 R4 touch
        # points "run_step1_batch.py, run_sequential.py | Forward the new
        # keys into GP()."), PLUS the two Stage-3-vintage keys
        # (R_BLAS_FOLLOWS_N_JOBS/TORCH_NUM_THREADS) that gained a GP()
        # parameter only this stage - see genomic_prediction.py::GP()'s own
        # signature docstring for why those two needed fixing here too, not
        # just the four genuinely-new ones. Every default below matches
        # GP()'s own parameter default exactly (never re-decided
        # independently - see the blueprint §4 config schema table), so an
        # older step1_config.json missing every one of these keys reproduces
        # pre-ver4-4 behaviour exactly (I11). Since gp_kwargs (this whole
        # dict) is what BOTH intra_task_parallel.run_batch_with_model_level_
        # parallelism() and intra_batch_parallel.run_batch_with_intra_batch_
        # parallelism() forward on to their own internal GP(**gp_kwargs)/
        # GP(**unit_gp_kwargs) calls (each worker copies THIS dict via
        # dict(gp_kwargs) before overriding only its own task-specific keys -
        # see those two modules' own _run_unit()/worker functions), adding
        # these keys HERE, once, is sufficient to reach every one of
        # run_step1_batch.py's three dispatch routes (plain, intra-batch,
        # intra-task) - no separate edit needed in either of those two
        # modules.
        R_BLAS_FOLLOWS_N_JOBS=cfg.get('R_BLAS_FOLLOWS_N_JOBS', True),
        TORCH_NUM_THREADS=cfg.get('TORCH_NUM_THREADS'),
        TORCH_DATALOADER_WORKERS=cfg.get('TORCH_DATALOADER_WORKERS', 0),
        GPU_EVAL_BATCH=cfg.get('GPU_EVAL_BATCH', 32),
        GPU_LD_R2=cfg.get('GPU_LD_R2', True),
        GPU_KERNEL_PRECOMPUTE=cfg.get('GPU_KERNEL_PRECOMPUTE', True),
        # ver4-4 Stage 6 (blueprint §10 Stage 6 checklist item 5 - "forward
        # every new key into GP()"). Reaches all three of this script's own
        # dispatch routes (plain/intra-batch/intra-task) for free, via the
        # SAME gp_kwargs-forwarding mechanism the comment above already
        # explains for the Stage 5 keys. Default True per §4a; see
        # genomic_prediction.py's own HP_TUNE dispatch site for the
        # disclosed caveat on what this can currently achieve in practice.
        HP_TUNE_PARALLEL_TRIALS=cfg.get('HP_TUNE_PARALLEL_TRIALS', True),
        # ver4-5 R1 (blueprint §3.6) - same forward-every-new-key
        # discipline, reaching all three of this script's own dispatch
        # routes via the same gp_kwargs mechanism; same legacy-preserving
        # defaults.
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
        # Update ID ver4-9, R7 - same forward-every-new-key discipline as
        # run_sequential.py's own gp_kwargs; default 'gzip' matches GP()'s
        # own default.
        RESULT_COMPRESSION=cfg.get('RESULT_COMPRESSION', 'gzip'),
    )

    _t0 = time.time()
    # Update ID 2 (R1) - three-way route, in priority order:
    #   1. N_MODEL_WORKERS > 1  -> intra_task_parallel (NEW: task-level AND
    #      model-level fan-out together, on one shared process pool).
    #   2. N_CPU_WORKERS[_TASK] > 1 -> intra_batch_parallel (Phase 2,
    #      UNCHANGED - task-level fan-out only).
    #   3. neither -> a single, ordinary GP() call (UNCHANGED).
    # N_MODEL_WORKERS defaults to 1 (inert) for every config written
    # before this feature existed, so route 1 is never taken unless a
    # person explicitly opted in via the GUI's "Model-level parallelism"
    # widgets - routes 2/3 are exactly Phase 2's own behaviour, untouched.
    n_model_workers = int(cfg.get('N_MODEL_WORKERS', 1) or 1)
    n_cpu_workers_legacy = int(cfg.get('N_CPU_WORKERS', 1) or 1)

    if n_model_workers > 1:
        n_cpu_workers_task = int(cfg.get('N_CPU_WORKERS_TASK', n_cpu_workers_legacy) or 1)

        # Pre-divide N_JOBS/PLINK_THREADS/R_BLAS_THREADS by N_MODEL_WORKERS
        # HERE, once, at orchestration time - exactly the way main_app.py's
        # own HPC export handler already pre-divides N_JOBS by
        # N_CPU_WORKERS_TASK at CONFIG-WRITE time (cfg['N_JOBS'] above is
        # therefore already that task-level share) - see
        # pipeline_utils.resolve_compute_resources()'s own documented
        # deviation for why the two division layers are split this way.
        # GP()'s OWN internal resolve_compute_resources() call never
        # divides by N_CPU_WORKERS_TASK/N_MODEL_WORKERS itself (its own
        # _compute_cfg never carries N_MODEL_WORKERS at all), so this is
        # the ONLY place this division happens - no risk of dividing twice.
        from pipeline_utils import resolve_compute_resources as _resolve_for_orchestration
        _pre_divided = _resolve_for_orchestration(cfg)
        gp_kwargs['N_JOBS'] = _pre_divided['n_jobs']
        gp_kwargs['PLINK_THREADS'] = _pre_divided['plink_threads']
        gp_kwargs['R_BLAS_THREADS'] = _pre_divided['r_blas_threads']

        from intra_task_parallel import run_batch_with_model_level_parallelism
        print(f"[run_step1_batch] N_MODEL_WORKERS={n_model_workers} "
              f"(N_CPU_WORKERS_TASK={n_cpu_workers_task}) - using intra-task model-level "
              f"parallelism for this batch's {batch_size} task(s); pre-divided "
              f"N_JOBS={gp_kwargs['N_JOBS']}, PLINK_THREADS={gp_kwargs['PLINK_THREADS']}, "
              f"R_BLAS_THREADS={gp_kwargs['R_BLAS_THREADS']}.")
        run_batch_with_model_level_parallelism(
            gp_kwargs,
            n_task_workers=n_cpu_workers_task,
            n_model_workers=n_model_workers,
            n_gpu_slots=cfg.get('N_GPU_SLOTS') or 0,
            model_grouping=cfg.get('MODEL_GROUPING', 'cost_balanced'),
            bio_prior_grouping=cfg.get('BIO_PRIOR_GROUPING', 'affinity'),
            min_models_for_parallel=cfg.get('MIN_MODELS_FOR_PARALLEL', 2),
        )
    elif n_cpu_workers_legacy > 1:
        # Phase 2, Requirement 7 - intra-batch multi-CPU/GPU parallelism:
        # fan this batch's own tasks out across up to N_CPU_WORKERS worker
        # processes instead of GP()'s fully serial driving loop. Falls
        # back to a single ordinary GP() call internally (see
        # intra_batch_parallel.py's own docstring) whenever this batch has
        # too few tasks to be worth it, so this is always safe to enable.
        # UNCHANGED by Update ID 2.
        from intra_batch_parallel import run_batch_with_intra_batch_parallelism
        print(f"[run_step1_batch] N_CPU_WORKERS={n_cpu_workers_legacy} - using intra-batch "
              f"parallelism for this batch's {batch_size} task(s).")
        run_batch_with_intra_batch_parallelism(
            gp_kwargs, n_cpu_workers=n_cpu_workers_legacy, n_gpu_slots=cfg.get('N_GPU_SLOTS') or 0,
        )
    else:
        # This is the one route where THIS (parent) process calls GP()
        # directly, in-process, with no forking involved at all - so,
        # unlike the two routes above, it's safe (and necessary) to
        # initialise rpy2/R here. See the note beside
        # configure_r_environment() near the top of main() for why this
        # is no longer done unconditionally before the three-way branch.
        init_rpy2_conversion()
        # Imported after R/rpy2 setup so R_HOME/PATH are already correct.
        from genomic_prediction import GP
        GP(**gp_kwargs)

    print(f'[run_step1_batch] Batch {batch_id} finished (took {time.time() - _t0:.1f}s). '
          f'See the [GP] lines above for LD-pruning and per-model detail.')


if __name__ == '__main__':
    main()
