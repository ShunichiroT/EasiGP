"""resource_advise.py - ver4-56, R3: headless, measured HPC resource advice.

Why: the GUI advisor needs the telemetry of earlier runs, but those usually live on the cluster
while the GUI runs on a laptop. This script runs on the cluster login node (no Streamlit, no R,
no GPU needed), prints the same recommendation the GUI panel shows (both call
``resource_profiles.estimate_resources_calibrated`` with the same inputs, AC3.8) and can write it
into the job script and the config so the three stay consistent:

* job script (``--rewrite-script``): only the resource lines are rewritten -
  ``#PBS -l ncpus= / mem= / walltime= / ngpus=`` and ``#SBATCH --cpus-per-task / --mem / --time /
  --gres=gpu``; when the batch size changes, the END of the array range (``#PBS -J`` /
  ``#SBATCH -a``) follows it, because a stale range would silently skip or repeat batches.
* config (``--rewrite-config``): only ``N_CPU_WORKERS`` and ``N_CPU_WORKERS_TASK`` (the same
  worker count: ``run_step1_batch.py`` fans out on the first, the engine's resolver reads the
  second), ``N_JOBS``, ``PLINK_THREADS`` (kept equal to ``N_JOBS``, as the GUI export does) and
  ``PARALLEL.batch_size``; plus
  ``RESOURCE_ADVISOR.prediction`` (what ``tools/resource_audit.py`` compares against later).
  These keys are not in the checkpoint fingerprint (blueprint P9), so resume stays valid.

Examples::

    python tools/resource_advise.py --config Result/run1/step1_config.json
    python tools/resource_advise.py --config cfg.json --telemetry Result/pilot1 \\
        --rewrite-script run_batch.pbs --rewrite-config

Exit codes: 0 = advice produced, 1 = error, 3 = nothing to do (the config produces no tasks).
It never imports ``main_app`` and never fits a model (I1).
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXIT_OK, EXIT_ERROR, EXIT_NOTHING = 0, 1, 3


def format_walltime(seconds):
    """``HH:MM:SS`` (hours may exceed 24: schedulers expect total hours)."""
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    return f"{hours:02d}:{rest // 60:02d}:{rest % 60:02d}"


def _replace_tokens(line, replacements):
    """Replace ``key=value`` tokens of one ``#PBS -l`` line (comma-separated list)."""
    head, _, body = line.partition('-l')
    tokens = body.strip().split(',')
    changed = False
    for index, token in enumerate(tokens):
        key = token.split('=', 1)[0].strip()
        if key in replacements and '=' in token:
            new_token = f"{key}={replacements[key]}"
            changed = changed or (token.strip() != new_token)
            tokens[index] = new_token
    return head + '-l ' + ','.join(tokens), changed


def rewrite_script_text(text, result, old_batch_size=None):
    """Rewrite the resource lines of a PBS / Slurm script.

    Args:
        text: the script text.
        result: an ``estimate_resources_calibrated`` result.
        old_batch_size: the batch size the script was written for (only used to
            decide whether the array range must change).

    Returns:
        ``(new_text, changes, missing)``: ``changes`` is a list of
        ``'old -> new'`` strings, ``missing`` the resource directives the script
        has no line for (never inserted - the script's author decides where they go).
    """
    values = {'ncpus': str(result['ncpus']), 'mem': f"{int(result['mem_gb'])}GB",
              'walltime': format_walltime(result['walltime_s']), 'ngpus': str(result['ngpus'])}
    n_batches = int(result.get('suggested_n_batches') or 0)
    batch_changed = (bool(old_batch_size)
                     and int(old_batch_size) != int(result['suggested_batch_size']))
    found = set()
    changes = []
    out_lines = []
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        new_line = line
        if re.match(r'#PBS\s+-l\s', stripped):
            rewritten, changed = _replace_tokens(stripped, values)
            for key in values:
                if re.search(rf'(?:^|[\s,]){key}=', stripped):
                    found.add(key)
            if changed:
                new_line = rewritten + ('\n' if line.endswith('\n') else '')
        elif batch_changed and n_batches and re.match(
                r'#(PBS\s+-J|SBATCH\s+(-a|--array))\b', stripped):
            new_line = _rewrite_array_end(line, n_batches)
        elif stripped.startswith('#SBATCH'):
            new_line = _rewrite_sbatch(line, values, result, found)
        if new_line != line:
            changes.append(f"{line.strip()}  ->  {new_line.strip()}")
        out_lines.append(new_line)
    if batch_changed and n_batches == 1 and any(
            '-J' in c or '-a' in c or '--array' in c for c in changes):
        changes.append("NOTE: the array now has a single subjob; PBS Pro rejects some single-index "
                       "ranges - submit it as a plain job (no -J) if qsub complains.")
    wanted = {'ncpus', 'mem', 'walltime'} | ({'ngpus'} if result['ngpus'] else set())
    missing = sorted(wanted - found)
    return ''.join(out_lines), changes, missing


def _rewrite_sbatch(line, values, result, found):
    stripped = line.strip()
    patterns = (
        (r'^(#SBATCH\s+--cpus-per-task[=\s]+)\S+', values['ncpus'], 'ncpus'),
        (r'^(#SBATCH\s+-c\s+)\S+', values['ncpus'], 'ncpus'),
        (r'^(#SBATCH\s+--mem[=\s]+)\S+', f"{int(result['mem_gb'])}G", 'mem'),
        (r'^(#SBATCH\s+--time[=\s]+)\S+', values['walltime'], 'walltime'),
        (r'^(#SBATCH\s+-t\s+)\S+', values['walltime'], 'walltime'),
    )
    for pattern, value, key in patterns:
        if re.match(pattern, stripped):
            found.add(key)
            return re.sub(pattern, lambda m: m.group(1) + value, line.rstrip('\n'), count=1) + (
                '\n' if line.endswith('\n') else '')
    match = re.match(r'^(#SBATCH\s+--gres[=\s]+gpu)(:[A-Za-z0-9_.-]+)?(:\d+)?\s*$', stripped)
    if match:
        found.add('ngpus')
        model = match.group(2) if (match.group(2) and not match.group(2)[1:].isdigit()) else ''
        return f"{match.group(1)}{model}:{values['ngpus']}" + ('\n' if line.endswith('\n') else '')
    return line


def _rewrite_array_end(line, n_batches):
    """Set the END of ``start-end[%throttle]`` to ``start + n_batches - 1``."""
    def repl(match):
        start = int(match.group(1))
        return f"{start}-{start + n_batches - 1}{match.group(3) or ''}"
    return re.sub(r'(\d+)-(\d+)(%\d+)?', repl, line, count=1)


def rewrite_config_dict(cfg, result):
    """A copy of ``cfg`` with the advised layout and the stored prediction."""
    new = json.loads(json.dumps(cfg))
    # run_step1_batch.py fans tasks out on N_CPU_WORKERS (legacy name); the engine's
    # resolver reads N_CPU_WORKERS_TASK. Both carry the same worker count so the layout
    # that was priced is the layout that runs; N_JOBS is the per-worker thread count
    # (config-write-time division, see intra_batch_parallel's reclaim note).
    new['N_CPU_WORKERS'] = int(result['n_cpu_workers_task'])
    new['N_CPU_WORKERS_TASK'] = int(result['n_cpu_workers_task'])
    new['N_JOBS'] = int(result['n_jobs'])
    new['PLINK_THREADS'] = int(result['n_jobs'])
    parallel = dict(new.get('PARALLEL') or {})
    parallel['batch_size'] = int(result['suggested_batch_size'])
    new['PARALLEL'] = parallel
    advisor = dict(new.get('RESOURCE_ADVISOR') or {})
    advisor['prediction'] = result['prediction']
    new['RESOURCE_ADVISOR'] = advisor
    return new


def advise(cfg, args):
    """Run the estimator for ``cfg`` (shared by the CLI and the tests)."""
    import resource_profiles as rp
    advisor = cfg.get('RESOURCE_ADVISOR') or {}
    dirs = rp.calibration_dirs(cfg, args.telemetry)
    calibration = rp.load_calibration(dirs, machine_key=args.machine_key, verbose=True)
    print(f"[resource_advise] calibration: {calibration.n_records} record(s) from "
          f"{len(dirs)} folder(s); machine_key={calibration.machine_key}; "
          f"measured models: {', '.join(calibration.models_measured()) or 'none'}")
    if not calibration.has_data():
        # ver4-95: no telemetry -> the calibrated model has only priors (tier D) and is far
        # less accurate than the static heuristic, so answer with the static estimate, like
        # the GUI panel does; the calibrated advice takes over once a pilot/run has telemetry.
        print("[resource_advise] no telemetry found - using the STATIC estimate (no walltime, "
              "no worker/thread split). Run the pilot job (run_step1_batch.py --pilot) and "
              "re-run this tool for the calibrated advice.")
        static = rp.estimate_resources_static_from_cfg(
            cfg, args.hpc_profile or cfg.get('HPC_PROFILE') or rp.DEFAULT_HPC_PROFILE)
        static['total_tasks_all_batches'] = int(static.get('total_tasks_all_batches') or 0)
        return static
    result = rp.estimate_resources_calibrated(
        cfg, calibration,
        hpc_profile=args.hpc_profile or cfg.get('HPC_PROFILE') or rp.DEFAULT_HPC_PROFILE,
        objective=args.objective or advisor.get('objective', 'cost'),
        target_walltime_h=(args.target_walltime_h if args.target_walltime_h is not None
                           else advisor.get('target_walltime_h', 12.0)),
        walltime_quantile=(args.walltime_quantile if args.walltime_quantile is not None
                           else advisor.get('walltime_quantile', 0.90)),
        mem_quantile=(args.mem_quantile if args.mem_quantile is not None
                      else advisor.get('mem_quantile', 0.95)),
        fixed=({'batch_size': int(cfg['PARALLEL']['batch_size'])}
               if getattr(args, 'keep_batch_size', False)
               and (cfg.get('PARALLEL') or {}).get('batch_size') else None),
    )
    return result


def print_static_report(result):
    """Human-readable summary of the static estimate (what the GUI panel shows while no
    telemetry exists): ncpus / memory / GPU and the job-array layout, no walltime."""
    print("\n=== Static resource advice (no telemetry yet) ===")
    print(f"  tasks            : {result['total_tasks_all_batches']} "
          f"in {result.get('suggested_n_batches')} batch(es) of {result.get('suggested_batch_size')}")
    print(f"  per subjob       : ncpus={result['ncpus']}  mem={result['mem_gb']}GB  "
          f"ngpus={int(result.get('ngpus') or 0)}  walltime=not estimated")
    for note in result.get('notes') or []:
        print(f"  NOTE: {note}")


def print_report(result):
    """Human-readable summary (what the GUI panel shows, as text)."""
    print("\n=== Calibrated resource advice ===")
    print(f"  tasks            : {result['total_tasks_all_batches']} "
          f"in {result['suggested_n_batches']} batch(es) of {result['suggested_batch_size']}")
    print(f"  per subjob       : ncpus={result['ncpus']}  mem={result['mem_gb']}GB  "
          f"ngpus={result['ngpus']}  walltime={result['walltime_hms']}")
    print(f"  layout           : N_CPU_WORKERS_TASK={result['n_cpu_workers_task']} x "
          f"N_JOBS={result['n_jobs']}   predicted CPU efficiency {result['cpu_eff_pred']:.0%}")
    print(f"  cost             : {result['su_total']:.1f} core-hours (whole array)   "
          f"confidence tier (worst): {result['tier_min']}")
    for entry in result['per_model']:
        print(f"    {entry['model']:<34} tier {entry['tier']}  "
              f"{entry['sec_per_task']:9.1f} s/task  "
              f"({entry['share']:.0%} of time, {entry['n_obs']} measured)")
    for note in result['notes']:
        print(f"  NOTE: {note}")
    if result.get('infeasible'):
        print(f"  WARNING: no layout satisfies every limit (binding: {result.get('binding')}).")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Measured HPC resource advice (EasiGP).")
    parser.add_argument('--config', required=True, help="the run's JSON config")
    parser.add_argument('--telemetry', nargs='*', default=[],
                        help="extra telemetry sources: Result names or folders")
    parser.add_argument('--hpc-profile', default=None)
    parser.add_argument('--objective', choices=['cost', 'turnaround'], default=None)
    parser.add_argument('--target-walltime-h', type=float, default=None)
    parser.add_argument('--walltime-quantile', type=float, default=None)
    parser.add_argument('--mem-quantile', type=float, default=None)
    parser.add_argument('--keep-batch-size', action='store_true',
                        help="price the run for PARALLEL.batch_size of the config instead of "
                             "letting the advisor choose the batch size")
    parser.add_argument('--machine-key', default=None,
                        help="machine the advice is FOR (default: this machine)")
    parser.add_argument('--rewrite-script', default=None,
                        help="PBS/Slurm script to update in place")
    parser.add_argument('--rewrite-config', action='store_true',
                        help="write the advised keys into --config (a .bak copy is kept)")
    parser.add_argument('--json', action='store_true', help="also print the result as JSON")
    args = parser.parse_args(argv)

    try:
        with open(args.config, 'r', encoding='utf-8') as handle:
            cfg = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[resource_advise] ERROR: cannot read config {args.config!r}: {exc}")
        return EXIT_ERROR
    try:
        result = advise(cfg, args)
    except (ValueError, KeyError) as exc:
        print(f"[resource_advise] ERROR: {type(exc).__name__}: {exc}")
        return EXIT_ERROR
    if result['total_tasks_all_batches'] == 0:
        print("[resource_advise] nothing to do: the configuration produces no tasks.")
        return EXIT_NOTHING
    if result.get('engine') == 'static':
        print_static_report(result)
        if args.json:
            print(json.dumps({k: v for k, v in result.items() if k != 'per_model'},
                             indent=2, default=str))
        if args.rewrite_script or args.rewrite_config:
            print("[resource_advise] --rewrite-script / --rewrite-config need the calibrated "
                  "advice (walltime and the worker/thread layout); nothing was rewritten. Run "
                  "the pilot job first, then run this tool again.")
        return EXIT_OK
    print_report(result)
    if args.json:
        summary = {k: v for k, v in result.items() if k != 'per_model'}
        print(json.dumps(summary, indent=2, default=str))

    if args.rewrite_script:
        try:
            with open(args.rewrite_script, 'r', encoding='utf-8') as handle:
                text = handle.read()
        except OSError as exc:
            print(f"[resource_advise] ERROR: cannot read script {args.rewrite_script!r}: {exc}")
            return EXIT_ERROR
        old_batch = (cfg.get('PARALLEL') or {}).get('batch_size')
        new_text, changes, missing = rewrite_script_text(text, result, old_batch)
        n_lines = sum(1 for change in changes if not change.startswith('NOTE:'))
        if n_lines:
            with open(args.rewrite_script, 'w', encoding='utf-8') as handle:
                handle.write(new_text)
        print(f"[resource_advise] {args.rewrite_script}: {n_lines} line(s) rewritten.")
        for change in changes:
            print(f"    {change}")
        if missing:
            print(f"[resource_advise] WARNING: the script has no line for {missing}; "
                  f"add them by hand (ncpus={result['ncpus']}, mem={result['mem_gb']}GB, "
                  f"walltime={result['walltime_hms']}, ngpus={result['ngpus']}).")
    if args.rewrite_config:
        new_cfg = rewrite_config_dict(cfg, result)
        try:
            with open(args.config + '.bak', 'w', encoding='utf-8') as handle:
                json.dump(cfg, handle, indent=2)
            with open(args.config, 'w', encoding='utf-8') as handle:
                json.dump(new_cfg, handle, indent=2)
        except OSError as exc:
            print(f"[resource_advise] ERROR: cannot write config: {exc}")
            return EXIT_ERROR
        print(f"[resource_advise] {args.config} updated (previous copy: {args.config}.bak).")
    return EXIT_OK


if __name__ == '__main__':
    sys.exit(main())
