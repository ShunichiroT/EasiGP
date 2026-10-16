"""resource_audit.py - ver4-56, R0 + R5: how far off are the resource requests?

Why: the old advisor never recorded an error figure (R0), and the calibrated advisor needs a
before/after yardstick (R5). This offline tool compares what a run was *predicted* and
*requested* to need with what it *used*, one row per subjob (batch).

Two modes (both read-only; neither imports ``main_app``, R or a model - I1):

``--baseline`` (R0) - for runs that pre-date telemetry. Reads ``[timing]`` lines from
    ``Result/<name>/logs/log_*.txt`` and, when ``--sched-logs`` is given, the scheduler
    footers (Gadi PBS "Resource Usage" block, Slurm ``sacct`` text). It recomputes B0 - the
    LEGACY advisor's ncpus/memory for the stored config - and writes
    ``resource_audit_baseline.csv`` with the median and worst-case ratios.

default (R5) - for runs made with ver4-56. Joins ``RESOURCE_ADVISOR.prediction`` (stored in the
    exported config by the GUI or ``tools/resource_advise.py --rewrite-config``) with the job and
    task telemetry records and the scheduler footers, writes ``resource_audit.csv`` and prints
    the AC3.7 pass/fail summary (plus B0 when the config allows it).

Missing inputs degrade, they do not fail: a footer that cannot be parsed leaves its fields blank
and prints a WARNING; with no footers the telemetry-only columns are still written (AC5.2).

Examples::

    python tools/resource_audit.py --result Result/run1 --baseline --sched-logs pbs_logs
    python tools/resource_audit.py --result Result/run1 --sched-logs pbs_logs \\
        --baseline-csv Result/run1/resource_audit_baseline.csv

Exit codes: 0 = report written, 1 = error, 3 = nothing to audit (no usable records).
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXIT_OK, EXIT_ERROR, EXIT_NOTHING = 0, 1, 3

#: AC3.7 thresholds (blueprint section 2, R3).
WALL_LOG_RATIO_MAX = math.log(1.25)
WALL_COVER_MIN = 0.90
WALL_REQ_OVER_USED_MAX = 1.6
MEM_REQ_OVER_PEAK_MAX = 1.35
CPU_EFF_MIN = 0.6

_TIMESTAMP = re.compile(r'^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]')
_TIMING = re.compile(r'\[timing\]\s+task=(\d+)/(\d+)\s*\|(.*)')
_TOTAL = re.compile(r'\btotal=([\d.]+)s')
_MODEL_STAGE = re.compile(r'\bmodel:([^\s=]+)=([\d.]+)s')
_BATCH_IN_TEXT = re.compile(r'log_step1_batch(\d+)_')
_UNITS = {'': 1.0, 'B': 1.0, 'K': 1.0 / 1024 ** 2, 'M': 1.0 / 1024, 'G': 1.0, 'T': 1024.0}


def _warn(what: str, why: str, nxt: str) -> None:
    """One ``[resource_audit]`` WARNING line: what happened, why, what happens next."""
    print(f"[resource_audit] WARNING: {what} - {why} - {nxt}")


# --------------------------------------------------------------------------- #
# Small parsers (pure; unit-tested without a cluster)
# --------------------------------------------------------------------------- #
def parse_hms(text: Any) -> Optional[float]:
    """Seconds from ``HH:MM:SS``, ``D-HH:MM:SS``, ``MM:SS(.fff)`` or a plain number; else None."""
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    try:
        days = 0.0
        if '-' in text:
            day_part, text = text.split('-', 1)
            days = float(day_part)
        parts = [float(piece) for piece in text.split(':')]
    except ValueError:
        return None
    if not 1 <= len(parts) <= 3:
        return None
    seconds = 0.0
    for piece in parts:
        seconds = seconds * 60.0 + piece
    return days * 86400.0 + seconds


def parse_mem_gb(text: Any) -> Optional[float]:
    """GB (1024-based) from ``45.2GB``, ``64G``, ``190000M``, ``4000Mc`` (per-CPU suffix is
    ignored here - the caller multiplies), ``512KB``; else None."""
    if text is None:
        return None
    match = re.match(r'^\s*([\d.]+)\s*([KMGTkmgt]?)[Bb]?[cn]?\s*$', str(text))
    if not match:
        return None
    try:
        return float(match.group(1)) * _UNITS[match.group(2).upper()]
    except (ValueError, KeyError):
        return None


def parse_pbs_footer(text: str) -> Optional[Dict[str, Any]]:
    """The PBS Pro "Resource Usage" footer (Gadi layout) as a dict, or None when absent.

    Keys (each None when its line is missing/unparseable): ``job_id``, ``ncpus_req``,
    ``ncpus_used``, ``cpu_time_s``, ``mem_req_gb``, ``mem_used_gb``, ``wall_req_s``,
    ``wall_used_s``, ``su``, ``array_idx``. Tolerant: label case and spacing are ignored.
    """
    if 'Resource Usage' not in text and 'Walltime Used' not in text:
        return None

    def grab(pattern: str) -> Optional[str]:
        found = re.search(pattern, text, flags=re.IGNORECASE)
        return found.group(1) if found else None

    job_id = grab(r'Job Id:\s*(\S+)')
    array_idx = None
    if job_id:
        indexed = re.search(r'\[(\d+)\]', job_id)
        array_idx = int(indexed.group(1)) if indexed else None
    ncpus_req = grab(r'NCPUs Requested:\s*(\d+)')
    ncpus_used = grab(r'NCPUs Used:\s*(\d+)')
    su = grab(r'Service Units:\s*([\d.]+)')
    return {
        'job_id': job_id, 'array_idx': array_idx,
        'ncpus_req': int(ncpus_req) if ncpus_req else None,
        'ncpus_used': int(ncpus_used) if ncpus_used else None,
        'cpu_time_s': parse_hms(grab(r'CPU Time Used:\s*([\d:\-.]+)')),
        'mem_req_gb': parse_mem_gb(grab(r'Memory Requested:\s*([\d.]+\s*[KMGT]?B)')),
        'mem_used_gb': parse_mem_gb(grab(r'Memory Used:\s*([\d.]+\s*[KMGT]?B)')),
        'wall_req_s': parse_hms(grab(r'Walltime requested:\s*([\d:\-]+)')),
        'wall_used_s': parse_hms(grab(r'Walltime Used:\s*([\d:\-]+)')),
        'su': float(su) if su else None,
    }


def parse_sacct_text(text: str) -> List[Dict[str, Any]]:
    """Rows of ``sacct -P -n -o JobID,Elapsed,TotalCPU,AllocCPUS,ReqMem,MaxRSS`` as dicts of the
    same shape as :func:`parse_pbs_footer`. Step rows (``123_4.batch``) only contribute their
    MaxRSS to the parent job; ``ReqMem`` with a trailing ``c``/``n`` is per CPU / per node."""
    jobs: Dict[str, Dict[str, Any]] = {}
    for line in text.splitlines():
        fields = line.strip().split('|')
        if len(fields) < 6 or not fields[0]:
            continue
        job_id, elapsed, total_cpu, alloc, req_mem, max_rss = [f.strip() for f in fields[:6]]
        base = job_id.split('.')[0]
        entry = jobs.setdefault(base, {
            'job_id': base, 'array_idx': None, 'ncpus_req': None, 'ncpus_used': None,
            'cpu_time_s': None, 'mem_req_gb': None, 'mem_used_gb': None, 'wall_req_s': None,
            'wall_used_s': None, 'su': None})
        if '_' in base and base.rsplit('_', 1)[1].isdigit():
            entry['array_idx'] = int(base.rsplit('_', 1)[1])
        if '.' in job_id:
            rss = parse_mem_gb(max_rss)
            if rss is not None:
                entry['mem_used_gb'] = max(entry['mem_used_gb'] or 0.0, rss)
            continue
        entry['wall_used_s'] = parse_hms(elapsed)
        entry['cpu_time_s'] = parse_hms(total_cpu)
        entry['ncpus_req'] = int(alloc) if alloc.isdigit() else None
        mem = parse_mem_gb(req_mem)
        if mem is not None and entry['ncpus_req'] and req_mem.endswith('c'):
            mem *= entry['ncpus_req']
        entry['mem_req_gb'] = mem
    return list(jobs.values())


def parse_timing_lines(text: str) -> Tuple[List[Dict[str, Any]], Optional[float]]:
    """``[timing]`` lines of one log as ``[{'task', 'of', 'total_s', 'model_s'}]``, plus the log's
    first-to-last timestamp span in seconds (a lower bound of the job's walltime)."""
    rows: List[Dict[str, Any]] = []
    first = last = None
    for line in text.splitlines():
        stamp = _TIMESTAMP.match(line)
        if stamp:
            try:
                moment = datetime.strptime(stamp.group(1), '%Y-%m-%d %H:%M:%S')
            except ValueError:
                moment = None
            if moment is not None:
                first = first or moment
                last = moment
        found = _TIMING.search(line)
        if not found:
            continue
        total = _TOTAL.search(found.group(3))
        if total is None:
            continue
        models = sum(float(sec) for _, sec in _MODEL_STAGE.findall(found.group(3)))
        rows.append({'task': int(found.group(1)), 'of': int(found.group(2)),
                     'total_s': float(total.group(1)), 'model_s': models})
    span = (last - first).total_seconds() if first and last else None
    return rows, span


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def _batch_from_name(path: str) -> Optional[int]:
    """Batch id from ``log_step1_batch<N>_<stamp>.txt``; None for any other name."""
    found = _BATCH_IN_TEXT.search(os.path.basename(path))
    return int(found.group(1)) if found else None


def scan_timing_logs(result_dir: str) -> Dict[int, Dict[str, Any]]:
    """``{batch_id: {'tasks', 'timing_total_s', 'model_s', 'log_wall_s', 'log'}}`` from
    ``logs/log_step1_batch<N>_*.txt`` (sequential logs are reported as batch 0). Several logs
    for one batch (a resubmission) keep the latest."""
    out: Dict[int, Dict[str, Any]] = {}
    paths = sorted(glob.glob(os.path.join(result_dir, 'logs', 'log_*.txt')), key=os.path.getmtime)
    for path in paths:
        base = os.path.basename(path)
        if base.startswith('log_pilot'):
            continue
        batch = _batch_from_name(path)
        if batch is None:
            if not base.startswith('log_sequential'):
                continue
            batch = 0
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as handle:
                rows, span = parse_timing_lines(handle.read())
        except OSError as exc:
            _warn(f"cannot read {path}", str(exc), "skipping this log")
            continue
        if not rows:
            continue
        out[batch] = {
            'tasks': len(rows), 'timing_total_s': sum(r['total_s'] for r in rows),
            'model_s': sum(r['model_s'] for r in rows), 'log_wall_s': span, 'log': base}
    return out


def scan_sched_logs(directory: Optional[str], use_sacct: bool = False,
                    job_ids: Sequence[str] = ()) -> List[Dict[str, Any]]:
    """Scheduler footers found under ``directory`` (PBS ``.o*`` / Slurm ``*.out``), each tagged
    with ``batch`` when the file's own text names its batch log; optionally ``sacct`` rows.
    Unparseable files are warned about and skipped."""
    footers: List[Dict[str, Any]] = []
    if directory:
        if not os.path.isdir(directory):
            _warn(f"--sched-logs '{directory}' is not a folder", "nothing to read there",
                  "continuing with telemetry only")
        else:
            for name in sorted(os.listdir(directory)):
                path = os.path.join(directory, name)
                if not os.path.isfile(path):
                    continue
                try:
                    with open(path, 'r', encoding='utf-8', errors='replace') as handle:
                        text = handle.read()
                except OSError as exc:
                    _warn(f"cannot read {path}", str(exc), "skipping it")
                    continue
                footer = parse_pbs_footer(text)
                if footer is None:
                    continue
                if all(footer[k] is None for k in ('wall_used_s', 'mem_used_gb', 'cpu_time_s')):
                    _warn(f"footer in {name} has no parseable usage fields",
                          "the layout differs from the Gadi PBS footer", "leaving them blank")
                    continue
                found = _BATCH_IN_TEXT.search(text)
                footer['batch'] = int(found.group(1)) if found else None
                footer['file'] = name
                footers.append(footer)
    if use_sacct and job_ids:
        if shutil.which('sacct') is None:
            _warn("--sacct requested but 'sacct' is not on PATH", "not a Slurm login node",
                  "continuing without it")
        else:
            try:
                result = subprocess.run(
                    ['sacct', '-P', '-n', '-j', ','.join(job_ids), '-o',
                     'JobID,Elapsed,TotalCPU,AllocCPUS,ReqMem,MaxRSS'],
                    capture_output=True, text=True, timeout=60, check=False)
                for row in parse_sacct_text(result.stdout):
                    row['batch'] = None
                    row['file'] = 'sacct'
                    footers.append(row)
            except (OSError, subprocess.SubprocessError) as exc:
                _warn("sacct failed", str(exc), "continuing without it")
    return footers


def load_config(result_dir: str, explicit: Optional[str]) -> Optional[Dict[str, Any]]:
    """The run's stored config: ``--config``, else ``step1_config.json``, else the newest
    ``*_config.json`` in the result folder; None when there is none."""
    candidates = [explicit] if explicit else [os.path.join(result_dir, 'step1_config.json')]
    if not explicit:
        candidates += sorted(glob.glob(os.path.join(result_dir, '*_config.json')),
                             key=os.path.getmtime, reverse=True)
    for path in candidates:
        if path and os.path.isfile(path):
            try:
                with open(path, 'r', encoding='utf-8') as handle:
                    cfg = json.load(handle)
                if isinstance(cfg, dict):
                    return cfg
            except (OSError, ValueError) as exc:
                _warn(f"cannot read config {path}", str(exc), "trying the next candidate")
    return None


def legacy_advice(cfg: Dict[str, Any], hpc_profile: Optional[str]) -> Optional[Dict[str, Any]]:
    """B0: the LEGACY advisor's answer for ``cfg``, called the way the GUI panel calls it (same
    kwargs, same two-call batch-size pattern). None - with a WARNING - when it cannot be
    computed, e.g. a missing phenotype file."""
    try:
        from resource_profiles import DEFAULT_HPC_PROFILE, estimate_resources, _n_asreml_options
        from pipeline_utils import (build_sample_table_from_cfg, count_unique_populations,
                                    resolve_trait_columns)
        n_population = count_unique_populations(
            cfg.get('PHENOTYPE_FILE_NAME'), cfg.get('POPULATION_SOURCE')) or 1
        phenotype = cfg.get('PHENOTYPE', 'all')
        n_phenotype = len(phenotype) if phenotype != 'all' else max(
            1, len(resolve_trait_columns(cfg, cfg.get('PHENOTYPE_FILE_NAME', ''))))
        kwargs = dict(
            scenario=cfg['SCENARIO'], n_population=n_population, n_phenotype=n_phenotype,
            n_ratio=len(cfg.get('RATIO', [1])), sample_num=cfg.get('ITER_NUM', 1),
            model_run=cfg['MODEL'], ld_prune_enabled=cfg.get('LD_PRUNE') is not None,
            rf_filter_enabled=cfg.get('RF_FILTER') is not None,
            genotype_format=cfg.get('GENOTYPE_FORMAT', 'csv'),
            genotype_file_name=cfg.get('GENOTYPE_FILE_NAME', ''),
            phenotype_file_name=cfg.get('PHENOTYPE_FILE_NAME'),
            w_opt_enabled=cfg.get('W_OPT') is not None,
            between_validation_source=(cfg.get('BETWEEN_VALIDATION_SOURCE')
                                       if cfg.get('BETWEEN_VALIDATION') else None),
            n_between_ratio=max(1, len(cfg.get('BETWEEN_RATIO') or [1])),
            n_excluded_combinations=len(cfg.get('EXCLUDED_POPULATION_COMBINATIONS') or []),
            n_options=_n_asreml_options(cfg),
            model_parallel_enabled=False, budget_ncpus=None, budget_ngpus=None,
            budget_mem_gb=None, hparameters=cfg.get('HPARAMETERS'), hp_tune=cfg.get('HP_TUNE'),
            other_models_marker_source=cfg.get('OTHER_MODELS_MARKER_SOURCE', 'full_or_filtered'),
            pool_cost_in_headline=cfg.get('POOL_COST_IN_HEADLINE', True),
            gpu_cpu_calibration_in_headline=cfg.get('GPU_CPU_CALIBRATION_IN_HEADLINE', True),
            hpc_profile=hpc_profile or DEFAULT_HPC_PROFILE,
            max_array_subjobs_override=None, ncpus_per_gpu_override=None,
            rf_filter_cfg=cfg.get('RF_FILTER'), ld_prune_estimated_n_markers=None)
        batch_size = (cfg.get('PARALLEL') or {}).get('batch_size')
        n_batches = 1
        if batch_size:
            try:
                total = len(build_sample_table_from_cfg(cfg))
            except (ValueError, KeyError, TypeError):
                total = (n_population * n_phenotype * kwargs['n_ratio'] * kwargs['sample_num'])
            n_batches = max(1, math.ceil(max(1, total) / int(batch_size)))
        estimate = estimate_resources(n_batches=n_batches, **kwargs)
        if not batch_size and estimate.get('suggested_n_batches'):
            estimate = estimate_resources(
                n_batches=max(1, int(estimate['suggested_n_batches'])), **kwargs)
        return estimate
    except Exception as exc:  # noqa: BLE001 - the baseline column is optional
        _warn("could not recompute B0 (legacy advisor)", f"{type(exc).__name__}: {exc}",
              "predicted columns left blank")
        return None


# --------------------------------------------------------------------------- #
# Telemetry -> per-batch usage
# --------------------------------------------------------------------------- #
def telemetry_by_batch(records: List[dict], batch_size: Optional[int], array_start: int
                       ) -> Dict[int, Dict[str, Any]]:
    """Per-batch usage from telemetry. Pilot runs (any run whose tasks are tagged ``pilot``) are
    ignored. A job record is attributed to the batch its run's tasks belong to; a run without
    tasks (the parent of an intra-batch pool) is attributed through its scheduler array index
    minus ``array_start``. Returns ``{batch: {'wall_s', 'mem_peak_gb', 'alloc_ncpus',
    'alloc_mem_gb', 'n_tasks', 'tasks_total_s'}}`` - absent fields are None."""
    pilot_runs = {r.get('run_id') for r in records
                  if r.get('kind') == 'task' and (r.get('task') or {}).get('pilot')}
    task_batches: Dict[Any, List[int]] = {}
    out: Dict[int, Dict[str, Any]] = {}

    def slot(batch: int) -> Dict[str, Any]:
        return out.setdefault(batch, {
            'wall_s': None, 'mem_peak_gb': None, 'alloc_ncpus': None, 'alloc_mem_gb': None,
            'n_tasks': 0, 'tasks_total_s': 0.0})

    for record in records:
        if record.get('run_id') in pilot_runs:
            continue
        kind = record.get('kind')
        if kind not in ('task', 'model'):
            continue
        row = (record.get('task') or {}).get('row')
        if row is None:
            continue
        batch = int(row) // int(batch_size) if batch_size else 0
        if kind == 'task':
            task_batches.setdefault(record.get('run_id'), []).append(batch)
            entry = slot(batch)
            entry['n_tasks'] += 1
            entry['tasks_total_s'] += float(record.get('total_s') or 0.0)
        peak = record.get('rss_tree_peak_gb')
        if peak is not None:
            entry = slot(batch)
            entry['mem_peak_gb'] = max(entry['mem_peak_gb'] or 0.0, float(peak))
    for record in records:
        if record.get('kind') != 'job' or record.get('run_id') in pilot_runs:
            continue
        batches = set(task_batches.get(record.get('run_id'), []))
        job = record.get('job') or {}
        if len(batches) == 1:
            batch = next(iter(batches))
        else:
            idx = job.get('array_idx')
            try:
                batch = int(idx) - array_start
            except (TypeError, ValueError):
                continue
        entry = slot(batch)
        wall = record.get('wall_s')
        if wall is not None:
            entry['wall_s'] = max(entry['wall_s'] or 0.0, float(wall))
        for key, src in (('alloc_ncpus', 'alloc_ncpus'), ('alloc_mem_gb', 'alloc_mem_gb')):
            if job.get(src) is not None:
                entry[key] = job[src]
        cgroup = record.get('cgroup_peak_gb')
        if cgroup is not None:
            entry['mem_peak_gb'] = max(entry['mem_peak_gb'] or 0.0, float(cgroup))
    return out


def _footer_for(batch: int, footers: List[Dict[str, Any]], array_start: int
                ) -> Optional[Dict[str, Any]]:
    """The footer for ``batch``: by the batch named inside the file, else by array index."""
    for footer in footers:
        if footer.get('batch') == batch:
            return footer
    for footer in footers:
        if footer.get('batch') is None and footer.get('array_idx') is not None \
                and int(footer['array_idx']) - array_start == batch:
            return footer
    return None


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def _ratio(num: Optional[float], den: Optional[float]) -> Optional[float]:
    """``num/den`` or None when either is missing or ``den`` is not positive."""
    if num is None or den is None or den <= 0:
        return None
    return num / den


def _median(values: Sequence[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return statistics.median(clean) if clean else None


def _fmt(value: Any, digits: int = 3) -> str:
    """CSV/summary cell: blank for None, rounded for floats."""
    if value is None:
        return ''
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_csv(path: str, header: Sequence[str], rows: List[Dict[str, Any]]) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow([_fmt(row.get(col)) for col in header])


def run_baseline(args: argparse.Namespace, result_dir: str) -> int:
    """R0: write ``resource_audit_baseline.csv`` (legacy prediction vs usage per batch)."""
    timing = scan_timing_logs(result_dir)
    if not timing:
        print(f"[resource_audit] nothing to audit: no [timing] lines under {result_dir}/logs "
              "(was the run made with a current engine, with logs kept?)")
        return EXIT_NOTHING
    footers = scan_sched_logs(args.sched_logs, args.sacct)
    cfg = load_config(result_dir, args.config)
    b0 = legacy_advice(cfg, args.hpc_profile) if cfg else None
    if cfg is None:
        _warn("no stored config found", "step1_config.json is missing",
              "predicted columns left blank")
    rows: List[Dict[str, Any]] = []
    for batch in sorted(timing):
        info = timing[batch]
        footer = _footer_for(batch, footers, args.array_start)
        wall = (footer or {}).get('wall_used_s') or info['log_wall_s']
        cpu_time = (footer or {}).get('cpu_time_s')
        busy = _ratio(cpu_time, wall)
        row = {
            'batch': batch, 'tasks': info['tasks'], 'timing_total_s': info['timing_total_s'],
            'model_s': info['model_s'], 'pred_ncpus': (b0 or {}).get('ncpus'),
            'pred_mem_gb': (b0 or {}).get('mem_gb'),
            'req_ncpus': (footer or {}).get('ncpus_req'),
            'req_mem_gb': (footer or {}).get('mem_req_gb'),
            'used_wall_s': wall, 'used_mem_gb': (footer or {}).get('mem_used_gb'),
            'used_cpu_time_s': cpu_time, 'used_busy_cpus': busy,
            'wall_source': 'scheduler' if (footer or {}).get('wall_used_s') else 'log_timestamps',
        }
        row['mem_pred_over_used'] = _ratio(row['pred_mem_gb'], row['used_mem_gb'])
        row['ncpus_pred_over_busy'] = _ratio(row['pred_ncpus'], busy)
        rows.append(row)
    header = ['batch', 'tasks', 'timing_total_s', 'model_s', 'pred_ncpus', 'pred_mem_gb',
              'req_ncpus', 'req_mem_gb', 'used_wall_s', 'used_mem_gb', 'used_cpu_time_s',
              'used_busy_cpus', 'wall_source', 'mem_pred_over_used', 'ncpus_pred_over_busy']
    path = os.path.join(result_dir, 'resource_audit_baseline.csv')
    _write_csv(path, header, rows)
    print(f"[resource_audit] wrote {path} ({len(rows)} batch row(s))")
    for label, key in (('memory   (pred/used)', 'mem_pred_over_used'),
                       ('ncpus    (pred/busy)', 'ncpus_pred_over_busy')):
        values = [r[key] for r in rows if r.get(key) is not None]
        if values:
            worst = max(values, key=lambda v: abs(math.log(v)))
            print(f"  B0 {label}: median {statistics.median(values):.2f}x, worst {worst:.2f}x "
                  f"over {len(values)} batch(es)")
        else:
            print(f"  B0 {label}: not computable (needs scheduler footers and a stored config)")
    if b0 is not None:
        print(f"  B0 legacy advice for this config: ncpus={b0.get('ncpus')}, "
              f"mem_gb={b0.get('mem_gb')}, ngpus={b0.get('ngpus')} (no walltime: defect E6)")
    return EXIT_OK


def audit_rows(cfg: Dict[str, Any], tele: Dict[int, Dict[str, Any]],
               footers: List[Dict[str, Any]], array_start: int) -> List[Dict[str, Any]]:
    """One audit row per batch: prediction (from ``RESOURCE_ADVISOR.prediction``), request
    (footer, else the telemetry allocation, else the prediction's own request) and use."""
    pred = (cfg.get('RESOURCE_ADVISOR') or {}).get('prediction') or {}
    rows: List[Dict[str, Any]] = []
    for batch in sorted(set(tele) | {f['batch'] for f in footers if f.get('batch') is not None}):
        info = tele.get(batch, {})
        footer = _footer_for(batch, footers, array_start)
        ncpus_req = (footer or {}).get('ncpus_req') or info.get('alloc_ncpus') or pred.get('ncpus')
        mem_req = (footer or {}).get('mem_req_gb') or info.get('alloc_mem_gb') \
            or pred.get('mem_gb_req')
        wall_req = (footer or {}).get('wall_req_s') or pred.get('walltime_s_req')
        wall_used = (footer or {}).get('wall_used_s') or info.get('wall_s')
        mem_used = (footer or {}).get('mem_used_gb')
        mem_source = 'scheduler'
        if mem_used is None:
            mem_used, mem_source = info.get('mem_peak_gb'), 'telemetry'
        cpu_time = (footer or {}).get('cpu_time_s')
        cpu_eff = _ratio(cpu_time, (wall_used * ncpus_req) if wall_used and ncpus_req else None)
        wall_pred = pred.get('walltime_s_pred_mean')
        row = {
            'batch': batch, 'n_tasks': info.get('n_tasks'),
            'ncpus_pred': pred.get('ncpus'), 'ncpus_req': ncpus_req,
            'mem_gb_pred': pred.get('mem_gb_pred'), 'mem_gb_req': mem_req,
            'mem_gb_used': mem_used, 'mem_source': mem_source if mem_used is not None else '',
            'wall_s_pred_mean': wall_pred, 'wall_s_req': wall_req, 'wall_s_used': wall_used,
            'cpu_time_s_used': cpu_time, 'cpu_eff_pred': pred.get('cpu_eff_pred'),
            'cpu_eff_used': cpu_eff,
            'wall_log_ratio': (math.log(wall_used / wall_pred)
                               if wall_used and wall_pred and wall_pred > 0 else None),
            'wall_req_over_used': _ratio(wall_req, wall_used),
            'mem_req_over_used': _ratio(mem_req, mem_used),
            'wall_covered': (None if wall_req is None or wall_used is None
                             else int(wall_req >= wall_used)),
            'mem_covered': (None if mem_req is None or mem_used is None
                            else int(mem_req >= mem_used)),
        }
        rows.append(row)
    return rows


def ac37_summary(rows: List[Dict[str, Any]], baseline: Optional[Dict[str, Optional[float]]]
                 ) -> List[Tuple[str, Optional[float], bool, str]]:
    """The AC3.7 metrics as ``(label, value, passed, note)``; ``passed`` is False when the value
    is missing (an unmeasured criterion is not a pass)."""
    def col(key: str) -> List[float]:
        return [r[key] for r in rows if r.get(key) is not None]

    abs_log = [abs(v) for v in col('wall_log_ratio')]
    cover_w, cover_m = col('wall_covered'), col('mem_covered')
    checks: List[Tuple[str, Optional[float], bool, str]] = []
    med_log = statistics.median(abs_log) if abs_log else None
    checks.append(('median |log(wall_used/wall_pred)| <= log 1.25', med_log,
                   med_log is not None and med_log <= WALL_LOG_RATIO_MAX,
                   f"limit {WALL_LOG_RATIO_MAX:.3f}"))
    frac_w = sum(cover_w) / len(cover_w) if cover_w else None
    checks.append(('walltime_req >= used in >= 90% of subjobs', frac_w,
                   frac_w is not None and frac_w >= WALL_COVER_MIN, 'limit 0.90'))
    med_w = _median(col('wall_req_over_used'))
    checks.append(('median walltime_req/used <= 1.6', med_w,
                   med_w is not None and med_w <= WALL_REQ_OVER_USED_MAX, 'limit 1.60'))
    frac_m = sum(cover_m) / len(cover_m) if cover_m else None
    checks.append(('mem_req >= peak in 100% of subjobs', frac_m,
                   frac_m is not None and frac_m >= 1.0, 'limit 1.00'))
    med_m = _median(col('mem_req_over_used'))
    checks.append(('median mem_req/peak <= 1.35', med_m,
                   med_m is not None and med_m <= MEM_REQ_OVER_PEAK_MAX, 'limit 1.35'))
    med_c = _median(col('cpu_eff_used'))
    checks.append(('median CPU efficiency >= 0.6', med_c,
                   med_c is not None and med_c >= CPU_EFF_MIN,
                   'needs a scheduler footer (CPU time)'))
    if baseline is not None:
        b_mem, b_cpu = baseline.get('mem_over'), baseline.get('cpu_eff')
        checks.append(('mem_req/peak strictly better than B0 (closer to 1)', med_m,
                       med_m is not None and b_mem is not None
                       and abs(math.log(med_m)) < abs(math.log(b_mem)),
                       f"B0 {_fmt(b_mem, 2)}x"))
        checks.append(('CPU efficiency strictly better than B0', med_c,
                       med_c is not None and b_cpu is not None and med_c > b_cpu,
                       f"B0 {_fmt(b_cpu, 2)}"))
        checks.append(('walltime strictly better than B0', None, False,
                       'B0 predicts no walltime (defect E6): compare req/used by hand'))
    return checks


def _baseline_from_csv(path: Optional[str]) -> Optional[Dict[str, Optional[float]]]:
    """Median B0 memory ratio and CPU efficiency from a previous ``--baseline`` CSV."""
    if not path:
        return None
    try:
        with open(path, 'r', encoding='utf-8', newline='') as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        _warn(f"cannot read --baseline-csv {path}", str(exc), "B0 comparison skipped")
        return None

    def values(key: str) -> List[float]:
        out = []
        for row in rows:
            try:
                out.append(float(row[key]))
            except (KeyError, TypeError, ValueError):
                continue
        return out

    mem = values('mem_pred_over_used')
    busy, req = values('used_busy_cpus'), values('req_ncpus')
    eff = [b / r for b, r in zip(busy, req) if r > 0]
    return {'mem_over': statistics.median(mem) if mem else None,
            'cpu_eff': statistics.median(eff) if eff else None}


def run_audit(args: argparse.Namespace, result_dir: str) -> int:
    """R5: write ``resource_audit.csv`` and print the AC3.7 summary."""
    from resource_calibration import load_telemetry
    cfg = load_config(result_dir, args.config)
    if cfg is None:
        print(f"[resource_audit] no stored config under {result_dir} - pass --config "
              "(it must carry RESOURCE_ADVISOR.prediction)")
        return EXIT_NOTHING
    if not (cfg.get('RESOURCE_ADVISOR') or {}).get('prediction'):
        _warn("the config has no RESOURCE_ADVISOR.prediction",
              "it was not exported with the calibrated advisor",
              "prediction columns are blank; request/used are still reported")
    notes: List[str] = []
    records = load_telemetry([result_dir], notes)
    for note in notes:
        print(f"[resource_audit] {note}")
    batch_size = (cfg.get('PARALLEL') or {}).get('batch_size') \
        or ((cfg.get('RESOURCE_ADVISOR') or {}).get('prediction') or {}).get('batch_size')
    tele = telemetry_by_batch(records, batch_size, args.array_start)
    job_ids = sorted({str((r.get('job') or {}).get('job_id')) for r in records
                      if r.get('kind') == 'job' and (r.get('job') or {}).get('job_id')})
    footers = scan_sched_logs(args.sched_logs, args.sacct, job_ids)
    if not tele and not footers:
        print("[resource_audit] nothing to audit: no telemetry records or scheduler footers "
              f"found for {result_dir}")
        return EXIT_NOTHING
    if not footers:
        print("[resource_audit] no scheduler footers: writing telemetry-only columns "
              "(CPU efficiency and scheduler memory stay blank)")
    rows = audit_rows(cfg, tele, footers, args.array_start)
    header = ['batch', 'n_tasks', 'ncpus_pred', 'ncpus_req', 'mem_gb_pred', 'mem_gb_req',
              'mem_gb_used', 'mem_source', 'wall_s_pred_mean', 'wall_s_req', 'wall_s_used',
              'cpu_time_s_used', 'cpu_eff_pred', 'cpu_eff_used', 'wall_log_ratio',
              'wall_req_over_used', 'mem_req_over_used', 'wall_covered', 'mem_covered']
    path = os.path.join(result_dir, 'resource_audit.csv')
    _write_csv(path, header, rows)
    print(f"[resource_audit] wrote {path} ({len(rows)} subjob row(s))")
    b0 = legacy_advice(cfg, args.hpc_profile)
    if b0 is not None:
        print(f"  B0 (legacy advisor) for this config: ncpus={b0.get('ncpus')}, "
              f"mem_gb={b0.get('mem_gb')}, ngpus={b0.get('ngpus')}, no walltime")
    print("  AC3.7 metrics:")
    for label, value, passed, note in ac37_summary(rows, _baseline_from_csv(args.baseline_csv)):
        state = 'PASS' if passed else ('FAIL' if value is not None else 'N/A ')
        print(f"    [{state}] {label}: {_fmt(value, 3) or 'not measured'} ({note})")
    return EXIT_OK


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--result', required=True, help="Result/<name> folder of the run")
    parser.add_argument('--baseline', action='store_true',
                        help="R0: audit a run without telemetry using [timing] lines + footers")
    parser.add_argument('--sched-logs', default=None,
                        help="folder with the scheduler's job-output files (PBS .o*, Slurm .out)")
    parser.add_argument('--sacct', action='store_true',
                        help="also query Slurm sacct for the telemetry's job ids")
    parser.add_argument('--config', default=None, help="config JSON (default: the run's own)")
    parser.add_argument('--hpc-profile', default=None, help="profile for the B0 recomputation")
    parser.add_argument('--array-start', type=int, default=0,
                        help="scheduler array index of batch 0 (default 0)")
    parser.add_argument('--baseline-csv', default=None,
                        help="resource_audit_baseline.csv to compare against (R5 mode)")
    args = parser.parse_args(argv)
    result_dir = args.result
    if not os.path.isdir(result_dir):
        print(f"[resource_audit] '{result_dir}' is not a folder - pass Result/<name>")
        return EXIT_ERROR
    try:
        return run_baseline(args, result_dir) if args.baseline else run_audit(args, result_dir)
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report, do not traceback at the user
        print(f"[resource_audit] failed: {type(exc).__name__}: {exc}")
        return EXIT_ERROR


if __name__ == '__main__':
    sys.exit(main())
