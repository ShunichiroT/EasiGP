"""Self-test for models/gat_stability.py (GAT run-to-run stability). Needs torch + torch_geometric.

    python tools/gat_stability_check.py        # prints PASS/FAIL lines and 'ALL OK'

Checks: seeded fits are reproducible; the global RNG streams are untouched; GAT_SEED='off' restores
the legacy unseeded behaviour; the seed ensemble equals the mean of its members and is never used
outside a final fit / inside a search fit; seed-robust selection prefers the better MEAN over the
luckiest draw; and an end-to-end Bayesian tuning run through the real tuner is reproducible.
"""
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, pandas as pd, torch
import pipeline_utils as pu
from models.GAT_infinitesimal import GAT_infinitesimal
from models import gat_stability as gs
from models import hyperparameter_tuning as ht

rng = np.random.RandomState(0)
n, m = 36, 8
X = rng.randint(0, 3, size=(n, m)).astype(float)
y = X[:, :3].sum(1) + rng.randn(n) * 0.5
cols = [f'm{i}' for i in range(m)] + ['y']
df = pd.DataFrame(np.c_[X, y], columns=cols, index=[f's{i}' for i in range(n)])
tr, va, te = df.iloc[:20], df.iloc[20:28], df.iloc[28:]
params = [8, 0.2, 0.01, 1e-4, 3, 8, 2, False, 5]   # neuron, dropout, lr, decay, epoch, bsize, heads, marker_effect, samples

def fit(**kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return GAT_infinitesimal(tr, va, te, list(params))


def setcfg(**acc):
    pu._ACTIVE_COMPUTE_RESOURCES = pu.resolve_compute_resources({'ACCELERATION': dict(acc)} if acc else None)

def pv(res): return np.asarray(res[4]); 
ok = True
def check(name, cond):
    global ok
    ok &= bool(cond); print(('PASS ' if cond else 'FAIL ') + name)

# --- F1: deterministic
setcfg()
r1, r2 = fit(), fit()
check('default engine: two fits identical (valid)', np.array_equal(pv(r1), pv(r2)))
check('default engine: two fits identical (test/train)', np.array_equal(r1[3], r2[3]) and np.array_equal(r1[5], r2[5]))
check('default engine: r/mse identical', r1[0] == r2[0] and r1[1] == r2[1])

# global RNG streams untouched by the fit
torch.manual_seed(123); np.random.seed(123)
a = (torch.rand(3).tolist(), np.random.rand(3).tolist())
torch.manual_seed(123); np.random.seed(123)
fit()
b = (torch.rand(3).tolist(), np.random.rand(3).tolist())
check('global torch/numpy RNG streams restored after the fit', a == b)

# legacy: seed off -> differ
setcfg(GAT_SEED='off')
l1, l2 = fit(), fit()
check('GAT_SEED=off: legacy unseeded fits differ', not np.array_equal(pv(l1), pv(l2)))

# different base seed / salt -> different
setcfg(GAT_SEED=7)
s7 = fit()
setcfg()
check('different base seed -> different draw', not np.array_equal(pv(s7), pv(r1)))
with gs.seed_salt(1):
    sl = fit()
check('seed_salt changes the draw', not np.array_equal(pv(sl), pv(r1)))
with gs.seed_salt(1):
    sl2 = fit()
check('seed_salt draw is itself reproducible', np.array_equal(pv(sl), pv(sl2)))
# data dependence
with contextlib.redirect_stdout(io.StringIO()):
    d2 = GAT_infinitesimal(tr.iloc[:19], va, te, list(params))
check('different training data -> different draw (smoke)', len(d2[4]) == len(r1[4]))

# --- F2: ensemble
setcfg(GAT_SEED_ENSEMBLE=3)
e_plain = fit()
check('ensemble outside final role: single fit (== member 0)', np.array_equal(pv(e_plain), pv(r1)))
with gs.final_fit_role():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        e = GAT_infinitesimal(tr, va, te, list(params))
check('ensemble in final role: prediction differs from single', not np.array_equal(pv(e), pv(r1)))
# manual members
digest = gs._data_digest(tr)
mem = []
for k in range(3):
    seed = gs.derive_seed('GAT_infinitesimal', digest, gs.DEFAULT_GAT_SEED, 0, k)
    with gs.rng_isolated(seed):
        with contextlib.redirect_stdout(io.StringIO()):
            mem.append(GAT_infinitesimal.__wrapped__(tr, va, te, list(params)))
check('member 0 equals the single fit', np.array_equal(pv(mem[0]), pv(r1)))
exp_valid = np.mean([pv(x) for x in mem], axis=0)
exp_test = np.mean([np.asarray(x[3]) for x in mem], axis=0)
check('ensemble valid == mean of 3 members', np.allclose(pv(e), exp_valid, atol=1e-6))
check('ensemble test == mean of 3 members', np.allclose(np.asarray(e[3]), exp_test, atol=1e-6))
from scipy.stats import pearsonr
check('ensemble r recomputed from averaged test preds', abs(e[0] - pearsonr(te['y'], exp_test)[0]) < 1e-6)
check('ensemble mse recomputed', abs(e[1] - float(np.mean((te['y'].values - exp_test) ** 2))) < 1e-6)
check('ensemble logged', 'averaged 3 seeded networks' in buf.getvalue())
# inside a search fit: no ensemble
with gs.final_fit_role():
    with pu.search_fit_context(lean=False):
        with contextlib.redirect_stdout(io.StringIO()):
            s_ = GAT_infinitesimal(tr, va, te, list(params))
check('search-fit context: never ensembles', np.array_equal(pv(s_), pv(r1)))
# reproducible ensemble
with gs.final_fit_role():
    with contextlib.redirect_stdout(io.StringIO()):
        e2 = GAT_infinitesimal(tr, va, te, list(params))
check('ensemble reproducible', np.array_equal(pv(e), pv(e2)))

# --- interaction with SEED_BAG (models/seed_bag.py): bag members must stay distinct
try:
    from models import seed_bag
    setcfg()
    def _fit_once():
        with contextlib.redirect_stdout(io.StringIO()):
            return GAT_infinitesimal(tr, va, te, list(params))
    bag = seed_bag.bag_call(_fit_once, 'GAT_infinitesimal', 2, 0, 5, te['y'].values)
    with seed_bag.isolated_rng(seed_bag.member_seed(0, 5, 0)), gs.external_seed_scope():
        m0 = _fit_once()
    with seed_bag.isolated_rng(seed_bag.member_seed(0, 5, 1)), gs.external_seed_scope():
        m1 = _fit_once()
    check('SEED_BAG members differ (decorator does not re-seed inside a bag)', not np.array_equal(pv(m0), pv(m1)))
    check('SEED_BAG result == mean of its two members', np.allclose(np.asarray(bag[4]), (pv(m0) + pv(m1)) / 2, atol=1e-6))
    bag2 = seed_bag.bag_call(_fit_once, 'GAT_infinitesimal', 2, 0, 5, te['y'].values)
    check('SEED_BAG run is reproducible', np.array_equal(np.asarray(bag[4]), np.asarray(bag2[4])))
    with gs.final_fit_role():
        setcfg(GAT_SEED_ENSEMBLE=3)
        bag3 = seed_bag.bag_call(_fit_once, 'GAT_infinitesimal', 2, 0, 5, te['y'].values)
    check('no GAT seed ensemble on top of a bag', np.array_equal(np.asarray(bag[4]), np.asarray(bag3[4])))
except ImportError:
    print('SKIP seed_bag checks (models/seed_bag.py not present in this tree)')

# --- config resolver
setcfg(GAT_SEED='off', GAT_SEED_ENSEMBLE=4, GAT_RESCORE_TOP=2, GAT_RESCORE_SEEDS=5)
st = gs.gat_stability_settings()
check('settings parse', st == {'seed': None, 'ensemble': 4, 'rescore_top': 2, 'rescore_seeds': 5})
setcfg(); st = gs.gat_stability_settings()
check('engine defaults', st == {'seed': 20240615, 'ensemble': 1, 'rescore_top': 0, 'rescore_seeds': 0})

def check(n, c):
    global ok; ok &= bool(c); print(('PASS ' if c else 'FAIL ') + n)

# --- pure selection logic
trials = [(np.array([1., 2.]), 0.5), (np.array([1., 2.]), 0.5), (np.array([3., 4.]), 0.9),
          {'params': {'x0': 5., 'x1': 6.}, 'target': 0.7}, (np.array([7., 8.]), float('nan'))]
top = gs.top_distinct(trials, 2, 3)
check('top_distinct: dedupes, drops NaN, sorted, handles dict trials',
      [round(s, 2) for _, s in top] == [0.9, 0.7, 0.5])
# candidate A looks best at salt 0 (lucky) but is worse on average; B is truly better
true = {(3., 4.): 0.50, (5., 6.): 0.60, (0., 0.): 0.40}
rs = np.random.RandomState(1)
def rescore(x, salts): return [true[tuple(x)] + rs.randn() * 0.01 for _ in salts]
sel = gs.select_by_seed_mean([(np.array([3., 4.]), 0.9), (np.array([5., 6.]), 0.62)],
                             (np.array([0., 0.]), 0.41), rescore, 4)
check('winner chosen by seed-mean, not by lucky first score', tuple(sel['x']) == (5., 6.))
check('seed-mean winner score is de-biased', abs(sel['mean'] - 0.6) < 0.03)
check('paired SE finite & small', 0 <= sel['paired_se'] < 0.05)
none = gs.select_by_seed_mean([(np.array([3., 4.]), 0.9)], (np.array([0., 0.]), 0.4), lambda x, s: [float('nan')] * len(s), 2)
check('all-NaN rescoring -> None (caller falls back)', none is None)

# --- end to end through the real tuner with a real GAT_infinitesimal
from models.GAT_infinitesimal import GAT_infinitesimal
rng = np.random.RandomState(0)
n, m = 60, 8
X = rng.randint(0, 3, size=(n, m)).astype(float)
y = X[:, :3].sum(1) + rng.randn(n) * 0.7
df = pd.DataFrame(np.c_[X, y], columns=[f'm{i}' for i in range(m)] + ['y'], index=[f's{i}' for i in range(n)])
train, valid, test = df.iloc[:36], df.iloc[36:48], df.iloc[48:]

from hparam_specs import HPARAM_SPECS
base = [8, 0.1, 0.01, 1e-4, 3, 16, 2, False, 5]
calls = {'n': 0, 'final_role': 0}
def run_model_fn(model_name, tr, va, te, params, RESULT_NAME):
    calls['n'] += 1
    if gs.in_final_fit_role(): calls['final_role'] += 1
    with contextlib.redirect_stdout(io.StringIO()) as b:
        r, mse, eff, pt, pv, ptr = GAT_infinitesimal(tr, va, te, list(params))
    out = io.StringIO.getvalue(b)
    if 'gat_stability' in out: print(out.strip().splitlines()[-1])
    rv, mv = pu.safe_regression_metrics(va.iloc[:, -1].values, np.asarray(pv))
    return dict(pearson_test=r, mse_test=mse, pearson_valid=rv, mse_valid=mv, predicted_test=pt,
                predicted_valid=pv, predicted_train=ptr, effect=eff)

def tune(**acc):
    pu._ACTIVE_COMPUTE_RESOURCES = pu.resolve_compute_resources({'ACCELERATION': dict(acc)})
    calls.update(n=0, final_role=0)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = ht.tune_model_hyperparameters('GAT_infinitesimal', list(base), train, valid, test, run_model_fn,
                                            'res', 'Bayesian', {'init_points': 3, 'n_iter': 3}, HPARAM_SPECS,
                                            n_jobs=1, bayes_batch=False)
    return res, buf.getvalue()


def sig(r):
    d = [x for x in r if isinstance(x, dict)]
    return [repr(x) for x in r[:2]], {k: round(float(v), 6) for k, v in (d[0].items() if d else []) if k in ('pearson_test', 'mse_test', 'pearson_valid', 'mse_valid')}
accs = dict(GAT_RESCORE_TOP=2, GAT_RESCORE_SEEDS=2, GAT_SEED_ENSEMBLE=2)
res, log = tune(**accs)
check('tuner: seed-robust rescoring ran', 're-scored the top' in log)
check('tuner: confirmatory fit ran in the final-fit role', calls['final_role'] >= 1)
res_b, _ = tune(**accs)
check('tuner: end-to-end run is reproducible (same winner, same score, same metrics)', sig(res) == sig(res_b))
_, log0 = tune()
check('tuner: engine defaults do not rescore', 're-scored' not in log0)
print('ALL OK' if ok else 'SOME FAILED')
sys.exit(0 if ok else 1)
