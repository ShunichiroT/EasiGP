"""bio_prior_cost_report.py - ver4-61, R5 / R7: calibrate and trim the bio-prior GAT's cost.

Why: on the attached run the data-driven merge made 98.8 % of the graph's edges, and the edge
scatter/gather (not the node GEMM) set both the training time and the host memory. The legacy
estimator prices the edges like a GEMM and was off by 49x-380x, so users could not tell a
10-minute fit from a 10-hour one. `resource_profiles.estimate_gat_step_seconds_calibrated`
(R5) is log-only and starts from two provisional anchors; this script fits its table from YOUR
logs and, for R7, measures what capping the merge's edges costs in accuracy. It fits nothing
itself, never imports torch and is never imported by the engine.

Usage (run from the EasiGP source directory):

  1. R5 / AC5.1 - calibration table from `[gat-shape]` lines of finished runs:

       python tools/bio_prior_cost_report.py report --logs 'Result/*/logs/*.txt' [--threads 2]

     One row per distinct CPU shape: measured ms/step, the legacy and calibrated estimates, their
     ratios to the measurement, and whether the calibrated one is within `--factor` (default 2).
     It prints `AC5.1: PASS|FAIL` (>= 12 of 14 shapes, scaled to the number of shapes found) and a
     `CPU_EDGE_NS_PER_CHANNEL = {...}` line fitted from the rows when `--threads` is given; paste
     that line into `resource_profiles.py` to recalibrate. The legacy `est_step_ms` is only read.

  2. R7 - the merge-density sweep (top_rate x max_edges, untuned, >= 5 replicates):

       python tools/bio_prior_cost_report.py make-sweep --config Result/X/sequential_config.json \
           --out-dir sweep_cfgs [--top-rates 50,10,2] [--max-edges none,5000,1000] [--replicates 5]
       # run each sweep_cfgs/<cell>_config.json with run_sequential.py (or as an array job), then:
       python tools/bio_prior_cost_report.py analyse-sweep --base-name X

     `analyse-sweep` reads the standard `Metric` files and the `[gat-shape]` log lines of the
     `Result/<base>_sweep_<cell>` folders and prints Pearson mean +/- SE, fit time, G and E0 per
     cell, then a `DECISION:` line (the cheapest cell within one SE of the best). It is advice for
     the Change Summary; no default is ever changed by this script.
"""
import argparse
import copy
import glob
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_BIO = 'GAT_biological_prior_knowledge'
_KV = re.compile(r"(\w+)=(\S+)")
_EST = re.compile(r"est_step_ms\(cpu,gpu\)=(?P<cpu>[0-9.]+),(?P<gpu>[0-9.]+)")
_SHAPE_KEYS = ('N_train', 'G', 'S', 'F', 'E0', 'B', 'epochs', 'HC')
_AC51_MIN_FRACTION = 12.0 / 14.0     # blueprint AC5.1: 12 of the 14 shapes of the attached run
_MERGE_IDX = 13                      # params[13] = data_driven_merge (append-only contract, I5)


# ------------------------------------------------------------------ report (R5)

def _num(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return float('nan')


def parse_shape_lines(paths):
    """`[gat-shape]` rows that carry a measured step time (the per-fit line, not the
    `epochs_run=` one). Returns a list of dicts; never raises on an unreadable line."""
    rows = []
    for path in paths:
        with open(path, encoding='utf-8', errors='replace') as fh:
            for line in fh:
                if '[gat-shape]' not in line or 'measured_step_ms=' not in line:
                    continue
                fields = dict(_KV.findall(line))
                est = _EST.search(line)
                row = {k: _num(fields.get(k)) for k in _SHAPE_KEYS}
                row.update(
                    device=fields.get('device', ''),
                    measured_ms=_num(fields.get('measured_step_ms')),
                    legacy_cpu_ms=_num(est.group('cpu')) if est else float('nan'),
                    cal_ms=_num(fields.get('est_cal_ms')), log=os.path.basename(path))
                if all(math.isfinite(row[k]) for k in ('G', 'S', 'E0', 'B', 'HC', 'N_train')):
                    rows.append(row)
    return rows


def _shape_of(row):
    return {'G': row['G'], 'S': row['S'], 'N_train': row['N_train'], 'E0': row['E0'],
            'B': row['B'], 'HC': row['HC'], 'epochs': row['epochs'],
            'factorised': bool(row['F'] == row['S'])}


def build_report(rows, threads, factor):
    """One row per distinct CPU shape (median measured time) with the estimate ratios."""
    cpu = pd.DataFrame([r for r in rows
                        if r['device'] == 'cpu' and math.isfinite(r['measured_ms'])])
    if cpu.empty:
        return cpu
    if threads:
        from resource_profiles import estimate_gat_step_seconds_calibrated as _cal
        missing = ~np.isfinite(cpu['cal_ms'])
        cpu.loc[missing, 'cal_ms'] = [_cal(_shape_of(r), threads) * 1000.0
                                      for r in cpu[missing].to_dict('records')]
    keys = [k for k in _SHAPE_KEYS]
    out = cpu.groupby(keys, as_index=False).agg(
        n=('measured_ms', 'size'), measured_ms=('measured_ms', 'median'),
        legacy_cpu_ms=('legacy_cpu_ms', 'first'), cal_ms=('cal_ms', 'first'))
    out['legacy_ratio'] = out['legacy_cpu_ms'] / out['measured_ms']
    out['cal_ratio'] = out['cal_ms'] / out['measured_ms']
    out['within'] = (out['cal_ratio'] >= 1.0 / factor) & (out['cal_ratio'] <= factor)
    return out


def fit_edge_ns(report, threads):
    """Median implied ns per (batch x edge x channel) after removing the node/Python terms."""
    from resource_profiles import estimate_gat_step_seconds_calibrated as _cal
    implied = []
    for r in report.to_dict('records'):
        shape = _shape_of(r)
        base_s = _cal(dict(shape, E0=1e-9), threads)        # node + python overhead only
        edge_s = r['measured_ms'] / 1000.0 - base_s
        if edge_s > 0:
            implied.append(edge_s * 1e9 / (r['B'] * r['E0'] * r['HC']))
    return float(np.median(implied)) if implied else float('nan')


def cmd_report(args):
    paths = sorted({p for pattern in args.logs for p in glob.glob(pattern)})
    if not paths:
        raise SystemExit(f"[COST_REPORT] no log matched {args.logs} - give --logs a glob such as "
                         f"'Result/*/logs/*.txt' - nothing was read.")
    rows = parse_shape_lines(paths)
    report = build_report(rows, args.threads, args.factor)
    if report.empty:
        raise SystemExit(f"[COST_REPORT] {len(paths)} log(s) read but no CPU '[gat-shape]' line "
                         f"with a measured step time - run with a current engine - nothing "
                         f"to do.")
    with pd.option_context('display.width', 200, 'display.max_columns', 30):
        print(report.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    n_ok, n = int(report['within'].sum()), len(report)
    need = math.ceil(_AC51_MIN_FRACTION * n - 1e-9)
    legacy_med = float(np.nanmedian(report['legacy_ratio']))
    print(f"[COST_REPORT] {n} CPU shape(s) from {len(paths)} log(s); legacy est/measured median "
          f"{legacy_med:.3g}x; calibrated within x{args.factor:g}: {n_ok} of {n}.")
    print(f"AC5.1: {'PASS' if n_ok >= need else 'FAIL'} ({n_ok} >= {need} of {n} required)")
    if args.threads:
        ns = fit_edge_ns(report, args.threads)
        print(f"[COST_REPORT] fitted at {args.threads} thread(s) (one fit alone assumed): "
              f"CPU_EDGE_NS_PER_CHANNEL = {{{args.threads}: {ns:.1f}}}  "
              f"# merge into resource_profiles.py")
    else:
        print("[COST_REPORT] NOTE: --threads not given - the ns/(edge x channel) fit is skipped "
              "and rows without a logged est_cal_ms were left out of the ratio.")


# ------------------------------------------------------------------ sweep (R7)

def _cell_name(top_rate, max_edges):
    return f"tr{top_rate:g}_me{'none' if max_edges is None else max_edges}"


def _parse_edges(text):
    return [None if t.strip().lower() == 'none' else int(t) for t in text.split(',')]


def _bio_param_lists(hparams):
    """Every bio-prior params list in an HPARAMETERS dict (plain, per-phenotype or numbered)."""
    for model, entry in hparams.items():
        if not str(model).startswith(_BIO):
            continue
        for params in (entry.values() if isinstance(entry, dict) else [entry]):
            yield model, params


def make_sweep_config(base, top_rate, max_edges, replicates):
    cfg = copy.deepcopy(base)
    cfg['MODEL'] = [m for m in base['MODEL'] if str(m).startswith(_BIO)]
    if not cfg['MODEL']:
        raise ValueError(f"[COST_REPORT] the config's MODEL list {base['MODEL']} has no "
                         f"{_BIO} entry - nothing to sweep.")
    cfg['HP_TUNE'], cfg['W_OPT'], cfg['HYPERPARAMETERS_OPT'] = None, None, {}
    cfg['ITER_NUM'] = int(replicates)
    n_patched = 0
    for model, params in _bio_param_lists(cfg['HPARAMETERS']):
        merge = params[_MERGE_IDX] if len(params) > _MERGE_IDX else None
        if not isinstance(merge, dict) or not merge.get('enabled', False):
            raise ValueError(f"[COST_REPORT] {model}: params[{_MERGE_IDX}] must be an enabled "
                             f"data_driven_merge dict, got {merge!r} - the sweep varies that "
                             f"merge.")
        merge['top_rate'] = float(top_rate)
        if max_edges is None:
            merge.pop('max_edges', None)
        else:
            merge['max_edges'] = int(max_edges)
        n_patched += 1
    if not n_patched:
        raise ValueError(f"[COST_REPORT] HPARAMETERS holds no {_BIO} entry - nothing to sweep.")
    cfg['RESULT_NAME'] = f"{base['RESULT_NAME']}_sweep_{_cell_name(top_rate, max_edges)}"
    return cfg


def cmd_make_sweep(args):
    with open(args.config, encoding='utf-8') as fh:
        base = json.load(fh)
    os.makedirs(args.out_dir, exist_ok=True)
    for top_rate in [float(t) for t in args.top_rates.split(',')]:
        for max_edges in _parse_edges(args.max_edges):
            cfg = make_sweep_config(base, top_rate, max_edges, args.replicates)
            path = os.path.join(args.out_dir, f"{_cell_name(top_rate, max_edges)}_config.json")
            with open(path, 'w', encoding='utf-8') as fh:
                json.dump(cfg, fh, indent=2)
            print(f"[COST_REPORT] wrote {path} (RESULT_NAME={cfg['RESULT_NAME']})")
    print(f"[COST_REPORT] run each with: python run_sequential.py --config <file> "
          f"(>= {args.replicates} replicates each, untuned).")


def _cell_frame(folder):
    """The cell's bio-prior Metric rows: the combined file when present, else every batch file."""
    for name in ('Metric.csv.gz', 'Metric.csv'):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            paths = [path]
            break
    else:
        paths = sorted(glob.glob(os.path.join(folder, 'Metric_*.csv')))
    if not paths:
        return None
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    return df[df['model'].astype(str).str.startswith(_BIO)]


def summarise_cell(folder):
    metric = _cell_frame(folder)
    if metric is None or metric.empty or 'Pearson correlation' not in metric.columns:
        return None
    r = metric['Pearson correlation'].astype(float).dropna().to_numpy()
    rows = parse_shape_lines(sorted(glob.glob(os.path.join(folder, 'logs', '*.txt'))))
    steps = [x['measured_ms'] * x['epochs'] * math.ceil(x['N_train'] / max(1.0, x['B'])) / 1000.0
             for x in rows if math.isfinite(x['measured_ms']) and math.isfinite(x['epochs'])]
    se = float(np.std(r, ddof=1) / math.sqrt(len(r))) if len(r) > 1 else float('nan')
    return {'n': len(r), 'r_mean': float(r.mean()), 'r_se': se,
            'fit_s': float(np.mean(steps)) if steps else float('nan'),
            'G': float(np.median([x['G'] for x in rows])) if rows else float('nan'),
            'E0': float(np.median([x['E0'] for x in rows])) if rows else float('nan')}


def decide(table):
    """Cheapest cell whose mean r is within one SE of the best cell's mean r."""
    ok = table.dropna(subset=['r_mean'])
    if ok.empty:
        return None
    best = ok.loc[ok['r_mean'].idxmax()]
    slack = float(best['r_se']) if math.isfinite(best['r_se']) else 0.0
    near = ok[ok['r_mean'] >= best['r_mean'] - slack]
    if near['fit_s'].notna().any():
        near = near.dropna(subset=['fit_s'])
    return near.loc[near['fit_s'].idxmin()] if not near['fit_s'].isna().all() else best


def cmd_analyse_sweep(args):
    base = os.path.join(args.result_dir, f"{args.base_name}_sweep_")
    cells = {}
    for folder in sorted(glob.glob(base + '*')):
        summary = summarise_cell(folder)
        if summary is not None:
            cells[folder[len(base):]] = summary
    if not cells:
        raise SystemExit(f"[COST_REPORT] no finished sweep cell under {base}* - run the configs "
                         f"from `make-sweep` first (each needs a Metric file) - nothing to report.")
    table = pd.DataFrame.from_dict(cells, orient='index')
    table.index.name = 'cell'
    print(table.to_string(float_format=lambda v: f"{v:.4g}"))
    short = [c for c, v in cells.items() if v['n'] < 5]
    if short:
        print(f"[COST_REPORT] NOTE: cell(s) {short} have fewer than the 5 replicates the "
              f"blueprint asks for - read the SE with care.")
    pick = decide(table)
    print(f"DECISION: {pick.name} is the cheapest cell within one SE of the best mean r "
          f"(r={pick['r_mean']:.4f}, fit={pick['fit_s']:.4g} s, G={pick['G']:.0f}, "
          f"E0={pick['E0']:.0f}); "
          f"record it in the Change Summary - this tool never changes a default.")


def _parser():
    parser = argparse.ArgumentParser(description="Bio-prior GAT cost report.")
    sub = parser.add_subparsers(dest='cmd', required=True)
    rep = sub.add_parser('report', help='calibration table from [gat-shape] log lines (R5)')
    rep.add_argument('--logs', nargs='+', required=True, help='log file globs')
    rep.add_argument('--threads', type=int, default=0, help='threads per fit in those runs')
    rep.add_argument('--factor', type=float, default=2.0, help='AC5.1 tolerance (default 2)')
    mk = sub.add_parser('make-sweep', help='write one config per top_rate x max_edges cell (R7)')
    mk.add_argument('--config', required=True)
    mk.add_argument('--out-dir', required=True)
    mk.add_argument('--top-rates', default='50,10,2')
    mk.add_argument('--max-edges', default='none,5000,1000')
    mk.add_argument('--replicates', type=int, default=5)
    an = sub.add_parser('analyse-sweep', help='summarise the finished sweep cells (R7)')
    an.add_argument('--base-name', required=True, help="the base config's RESULT_NAME")
    an.add_argument('--result-dir', default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Result'))
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    {'report': cmd_report, 'make-sweep': cmd_make_sweep,
     'analyse-sweep': cmd_analyse_sweep}[args.cmd](args)


if __name__ == '__main__':
    main()
