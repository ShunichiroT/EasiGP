"""bglr_noise_probe.py - ver4-34, R1.e (INVESTIGATE): offline measurement script.

Why: before the engine gains a 1-row test stub, fold racing, or a revised `priority` mark set,
three questions need numbers from the user's own data (blueprint R1.e):
  (i)   which BGLR models accept a 1-row `test` stub, and how much faster is the fit;
  (iii) at MCMC_SEARCH_SCALE, is the seed-to-seed SD of a validation column larger than the SD
        across the model's tunable settings (then tuning those priors fits noise);
  (ii)  does 2-fold-first successive halving over the CV folds keep the best setting.

Never imported by the engine. Run from the EasiGP source directory (needs R + rpy2):

  python tools/bglr_noise_probe.py --config step1_config.json --train tr.csv --valid va.csv \
      --test te.csv --out probe_out [--models rrBLUP,GBLUP,BayesB,RKHS] [--seeds 20 --settings 10]

`--config` is an engine config JSON (keys R_PATH, GENOTYPE_FORMAT, HPARAMETERS). Frames are
CSVs in the pipeline's layout: marker columns first, phenotype last (I4). Writes
`probe_fits.csv` (one row per fit) and `probe_summary.csv` (one row per model).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_WARNED_NO_RPY2 = False
hp = hp_specs = None  # imported in main() so `--help` works without R installed


def _pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def _set_r_seed(seed):
    """Seed R's RNG (BGLR draws from it); a missing rpy2 is reported, not hidden."""
    global _WARNED_NO_RPY2
    try:
        import rpy2.robjects as ro
        ro.r(f'set.seed({int(seed)})')
    except ImportError:
        if not _WARNED_NO_RPY2:
            _WARNED_NO_RPY2 = True
            print("[PROBE] rpy2 is not installed - R seed not set; seed-to-seed SD is NOT "
                  "meaningful. Install with: pip install rpy2")


def _timed(runner, model, tr, va, te, params, tag):
    t0 = time.perf_counter()
    res = runner(model, tr, va, te, params, tag)
    return res, time.perf_counter() - t0


def _settings(model, base, n, rng, hp):
    """`n` parameter lists: `base` with the NON-run-length tunable fields redrawn."""
    specs = [s for s in hp.build_param_specs(model, hp_specs.HPARAM_SPECS, base)
             if not _is_runlength(model, s.index)]
    tuned = {s.index for s in specs}
    out = []
    for _ in range(n):
        p = list(base)
        for idx, val in hp._decode(specs, hp._random_point(specs, rng)).items():
            p[idx] = val
        out.append(hp._cheapen_mcmc(model, p, base, tuned))
    return out, specs


def _is_runlength(model, idx):
    f = hp_specs.HPARAM_SPECS[hp.schema_key_of(model)][idx].get('tunable') or {}
    return bool(f.get('floor_at_baseline'))


def probe_model(model, base, tr, va, te, args, runner, rng, fit_rows):
    from models import joint_tuning as jt
    settings, specs = _settings(model, base, args.settings, rng, hp)
    if not specs:
        print(f"[PROBE] {model}: no non-run-length tunable fields - skipped")
        return None
    y_va = va.iloc[:, -1].to_numpy()
    r = np.full((args.settings, args.seeds), np.nan)
    secs = []
    for k, params in enumerate(settings):
        for s in range(args.seeds):
            _set_r_seed(1000 + s)
            res, dt = _timed(runner, model, tr, va, te, params, f'probe_{model}')
            r[k, s] = _pearson(res['predicted_valid'], y_va)
            secs.append(dt)
            fit_rows.append(dict(model=model, setting=k, seed=s, secs=dt, pearson_valid=r[k, s]))
    sd_seed = float(np.nanmean(np.nanstd(r, axis=1, ddof=1)))
    sd_set = float(np.nanstd(np.nanmean(r, axis=1), ddof=1))
    # (i) 1-row test stub: same fit, test frame cut to one row
    _set_r_seed(1000)
    full, t_full = _timed(runner, model, tr, va, te, settings[0], f'probe_{model}')
    stub_ok, t_stub = False, float('nan')
    try:
        _set_r_seed(1000)
        stub, t_stub = _timed(runner, model, tr, va, te.iloc[:1], settings[0], f'probe_{model}')
        stub_ok = _pearson(stub['predicted_valid'], full['predicted_valid']) > 0.999
    except (RuntimeError, ValueError, KeyError, IndexError) as exc:  # includes rpy2 errors
        print(f"[PROBE] {model}: 1-row test stub failed - {exc!r} - recorded as not accepted")
    # (ii) fold racing: rank settings by their first 2 folds, keep the top half
    tv = jt._cv_frame(tr, va)
    part = jt._cv_partition(len(tv), args.folds, 0)
    cols = jt._cv_fit_columns([(model, p) for p in settings], tv=tv, partition=part, seed=0,
                              test=te.iloc[:1], runner=runner, RESULT_NAME=f'probe_{model}')
    y_tv = tv.iloc[:, -1].to_numpy()
    per_fold = np.array([[_pearson(c[h], y_tv[h]) for h in part] if c is not None
                         else [np.nan] * len(part) for c in cols])
    full_score, race_score = np.nanmean(per_fold, axis=1), np.nanmean(per_fold[:, :2], axis=1)
    keep = set(np.argsort(-race_score)[: max(1, args.settings // 2)].tolist())
    n_f, n_s = len(part), args.settings
    return dict(model=model, sd_seed=sd_seed, sd_across_settings=sd_set,
                noise_ratio=sd_seed / sd_set if sd_set > 0 else float('inf'),
                priors_fit_noise=bool(sd_seed > sd_set), mean_fit_secs=float(np.mean(secs)),
                stub_accepted=stub_ok, full_test_secs=t_full, stub_test_secs=t_stub,
                race_keeps_best=bool(int(np.nanargmax(full_score)) in keep),
                race_fit_saving=1 - (2 * n_s + len(keep) * (n_f - 2)) / float(n_s * n_f))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--config', required=True)
    ap.add_argument('--train', required=True)
    ap.add_argument('--valid', required=True)
    ap.add_argument('--test', required=True)
    ap.add_argument('--out', default='probe_out')
    ap.add_argument('--models', default='rrBLUP,GBLUP,BayesB,RKHS')
    ap.add_argument('--seeds', type=int, default=20)
    ap.add_argument('--settings', type=int, default=10)
    ap.add_argument('--folds', type=int, default=5)
    args = ap.parse_args()
    global hp, hp_specs
    import hparam_specs as hp_specs
    from models import hyperparameter_tuning as hp
    from genomic_prediction import ModelTrialRunner
    with open(args.config, encoding='utf-8') as fh:
        cfg = json.load(fh)
    tr, va, te = (pd.read_csv(p) for p in (args.train, args.valid, args.test))
    runner = ModelTrialRunner('probe', cfg['R_PATH'], cfg.get('compute_resources') or {},
                              cfg.get('GENOTYPE_FORMAT', 'CSV'))
    os.makedirs(args.out, exist_ok=True)
    rng, fit_rows, summary = np.random.default_rng(0), [], []
    for model in [m.strip() for m in args.models.split(',') if m.strip()]:
        row = probe_model(model, cfg['HPARAMETERS'][model], tr, va, te, args, runner, rng, fit_rows)
        if row:
            summary.append(row)
            print(f"[PROBE] {row}")
    pd.DataFrame(fit_rows).to_csv(os.path.join(args.out, 'probe_fits.csv'), index=False)
    pd.DataFrame(summary).to_csv(os.path.join(args.out, 'probe_summary.csv'), index=False)
    print(f"[PROBE] wrote {len(fit_rows)} fit rows and {len(summary)} model rows to {args.out}")


if __name__ == '__main__':
    main()
