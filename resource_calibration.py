"""
EasiGP - resource calibration (ver4-56, blueprint R2).

Purpose
-------
Turn the usage records written by ``resource_telemetry.py`` into a per-model
prediction of seconds and peak memory, each with an uncertainty and a
confidence tier (A-D) that says how much measured data backs it. The HPC
advisor (``resource_profiles.estimate_resources_calibrated``) uses it to size
ncpus, memory and walltime.

Key property (blueprint section 0.1): within one run every task uses the SAME
model configuration; only the data sizes differ. So a few measured tasks of
that configuration predict the rest closely, and the problem reduces to
interpolating over the number of training individuals ``n`` and markers ``p``.

Method
------
The calibration unit is ``(machine_key, base model, structural signature)``.
The signature (:func:`structural_sig`) hashes every hyper-parameter that is NOT
a numeric cost driver and NOT declared cost-neutral, plus the tuning budget
and the device. Anything not declared is therefore in the signature, so an
unrecognised change lands in a lower tier and is never reused wrongly.

Seconds: ``log s = X b + e`` with covariates (centred on a reference size)
``[1, log n, log p, log driver_k ..., log threads, log workers]``. The prior is
``b ~ N(b0, s0^2)`` with slopes ``s0 = 0.3`` (analytic scaling exponents in
:data:`SCALING_PRIOR`) and ``sigma^2 ~ InvGamma(2, 0.1^2)``. This is the
Normal-Inverse-Gamma model, so the posterior and the Student-t predictive are
closed form (:func:`_nig_fit`).

Memory: ``log(peak - baseline)`` on ``[1, log(n p), log n]`` with
:data:`MEM_PRIOR`.

Uncertainty is split into ``sd_common`` (uncertainty of the shared parameters:
fully correlated across the tasks of one job) and ``sd_idio`` (task-to-task
scatter: averages out across the tasks of one worker).

Deviation from the textbook NIG shape (blueprint section 2 R2 asks for the plain
conjugate posterior): the shape is ``max(a0, 1 + (m - k_eff)/2)`` with
``k_eff = tr(X Lam_n^-1 X')`` instead of ``a0 + m/2``. The textbook value made the
90 % predictive intervals cover only 74-88 % of held-out points; this one covers
87-94 % for m >= 25 (see ``_nig_fit``).

Tiers (blueprint section 2 R2)
------------------------------
A  same machine + signature, >= 3 observations spanning ``n`` (max/min >= 1.3);
B  same machine + signature otherwise (slopes come from the prior), +0.15 common;
C  signature measured only on another machine, or this machine has other
   models: shifted by a machine speed factor, +0.5 common;
D  no telemetry: priors only, sd 1.0 (log).

No imports from other EasiGP modules except ``hparam_specs`` (I2).
"""

import hashlib
import json
import math
import os
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from hparam_specs import HPARAM_SPECS

# ---------------------------------------------------------------------------
# Tables (provisional where marked; every one is replaced by data as it comes)
# ---------------------------------------------------------------------------

# Reference problem the covariates are centred on: 500 training individuals,
# 2000 markers. Centring makes the intercept "seconds at the reference size".
N_REF = 500.0
P_REF = 2000.0

# Seconds that ONE unit of ``cpu_time_factor`` (resource_profiles.MODEL_COST_PROFILE,
# RF = 1.0) represents at the reference size. PROVISIONAL (tier D only): the
# blueprint named an ``_RF_REF_SECONDS`` constant that does not exist in the code,
# so this module defines its own and flags it.
PRIOR_REF_SECONDS = 60.0

# Fall-back copy of MODEL_COST_PROFILE's cpu_time_factor / peak_mem_gb_per_worker
# (ver4-56). resource_profiles passes its live table through ``fit(priors=...)``,
# which wins; this copy only keeps the module usable on its own.
_DEFAULT_PRIORS: Dict[str, Dict[str, float]] = {
    'rrBLUP': {'cpu_time_factor': 3.0, 'peak_mem_gb_per_worker': 3.0},
    'GBLUP': {'cpu_time_factor': 3.0, 'peak_mem_gb_per_worker': 3.0},
    'BayesB': {'cpu_time_factor': 6.0, 'peak_mem_gb_per_worker': 4.0},
    'RKHS': {'cpu_time_factor': 4.0, 'peak_mem_gb_per_worker': 4.0},
    'RF': {'cpu_time_factor': 1.0, 'peak_mem_gb_per_worker': 1.5},
    'ExtraTrees': {'cpu_time_factor': 1.0, 'peak_mem_gb_per_worker': 1.5},
    'GBDT': {'cpu_time_factor': 1.5, 'peak_mem_gb_per_worker': 1.5},
    'XGBoost': {'cpu_time_factor': 0.8, 'peak_mem_gb_per_worker': 1.5},
    'EBM': {'cpu_time_factor': 2.0, 'peak_mem_gb_per_worker': 1.5},
    'SVR': {'cpu_time_factor': 1.2, 'peak_mem_gb_per_worker': 1.0},
    'KNN': {'cpu_time_factor': 0.6, 'peak_mem_gb_per_worker': 1.0},
    'MLP': {'cpu_time_factor': 0.8, 'peak_mem_gb_per_worker': 1.5},
    'GAT_infinitesimal': {'cpu_time_factor': 2.5, 'peak_mem_gb_per_worker': 2.0},
    'GAT_infinitesimal_node_level': {'cpu_time_factor': 2.7, 'peak_mem_gb_per_worker': 2.0},
    'GAT_fully_connected': {'cpu_time_factor': 8.0, 'peak_mem_gb_per_worker': 6.0},
    'GAT_prior_knowledge': {'cpu_time_factor': 4.0, 'peak_mem_gb_per_worker': 2.5},
    'GAT_biological_prior_knowledge': {'cpu_time_factor': 3.0, 'peak_mem_gb_per_worker': 1.5},
}
_UNKNOWN_PRIOR = {'cpu_time_factor': 2.0, 'peak_mem_gb_per_worker': 2.0}

# Analytic scaling exponents of seconds in (n_train, p): the prior MEAN of the
# two slopes. The prior sd is SLOPE_PRIOR_SD for every slope.
SCALING_PRIOR: Dict[str, Tuple[float, float]] = {
    'rrBLUP': (1.0, 1.0), 'BayesB': (1.0, 1.0),
    'GBLUP': (2.0, 0.7), 'RKHS': (2.0, 0.7),
    'RF': (1.1, 0.9), 'ExtraTrees': (1.0, 0.9),
    'GBDT': (0.9, 0.8), 'XGBoost': (0.9, 0.8), 'EBM': (0.9, 1.0),
    'SVR': (2.0, 1.0), 'KNN': (1.5, 1.0),
    'MLP': (1.0, 1.0),
    'GAT_infinitesimal': (1.0, 1.0), 'GAT_infinitesimal_node_level': (1.0, 1.0),
    'GAT_fully_connected': (1.0, 1.5), 'GAT_prior_knowledge': (1.0, 1.0),
    'GAT_biological_prior_knowledge': (1.0, 0.5),
}
_UNKNOWN_SCALING = (1.0, 1.0)
SLOPE_PRIOR_SD = 0.3

# Memory increment over the model's starting RSS: prior (slope on log(n*p),
# slope on log n). Intercept comes from peak_mem_gb_per_worker.
MEM_PRIOR: Dict[str, Tuple[float, float]] = {
    'rrBLUP': (0.7, 0.1), 'BayesB': (0.7, 0.1), 'GBLUP': (0.5, 0.8), 'RKHS': (0.5, 0.8),
    'RF': (0.7, 0.1), 'ExtraTrees': (0.7, 0.1), 'GBDT': (0.7, 0.1), 'XGBoost': (0.7, 0.1),
    'EBM': (0.7, 0.1), 'SVR': (0.5, 0.8), 'KNN': (0.6, 0.4), 'MLP': (0.7, 0.1),
    'GAT_infinitesimal': (0.7, 0.1), 'GAT_infinitesimal_node_level': (0.7, 0.1),
    'GAT_fully_connected': (1.0, 0.1), 'GAT_prior_knowledge': (0.7, 0.1),
    'GAT_biological_prior_knowledge': (0.3, 0.1),
}
_UNKNOWN_MEM = (0.7, 0.2)
MEM_SLOPE_PRIOR_SD = 0.3
# Share of ``peak_mem_gb_per_worker`` that is the interpreter/libraries
# (resource_profiles._BASE_MEM_GB = 2.0): the rest is the model's own increment.
_PRIOR_BASE_GB = 1.0
_MIN_MEM_INCREMENT_GB = 0.05

# Largest thread count that still speeds a model up (prior knowledge, flagged
# provisional: a thread count above the cap adds nothing to the prediction).
THREAD_CAP: Dict[str, int] = {
    'RF': 16, 'ExtraTrees': 16, 'KNN': 16, 'GBDT': 8, 'XGBoost': 16, 'EBM': 8,
    'rrBLUP': 2, 'BayesB': 2, 'GBLUP': 4, 'RKHS': 4, 'SVR': 1,
    'MLP': 8, 'GAT_infinitesimal': 8, 'GAT_infinitesimal_node_level': 8,
    'GAT_fully_connected': 8, 'GAT_prior_knowledge': 8, 'GAT_biological_prior_knowledge': 8,
}
_THREAD_SLOPE_PRIOR = (-0.7, 0.3)      # sub-linear speed-up
_WORKER_SLOPE_PRIOR = (0.1, 0.2)       # contention between concurrent workers
_INTERCEPT_PRIOR_SD = 2.0

# Numeric cost drivers, declared by HPARAM_SPECS LABEL (I5: never by position).
# Value = prior slope of log(seconds) on log(driver) when the driver is ACTIVE.
# A driver whose spec has ``depends_on`` is active only while that toggle is on.
COST_DRIVERS: Dict[str, float] = {
    'Iteration number': 1.0,
    'Tree number': 1.0,
    'Tree number (n_estimators)': 1.0,
    'Boosting iterations (max_iter)': 1.0,
    'Epoch': 1.0,
    'Number of samples for Shapley scores': 0.5,
    'Number of samples for marker effects': 0.5,
    'Number of samples for marker effect interactions': 0.5,
    'Number of iterations for Shapley scores': 0.5,
    'Background sample size for marker effect (shapiq)': 0.3,
    'Background sample size for Shapley scores': 0.3,
    'Background sample size for interaction search': 0.3,
    'Background sample size for interaction (shapiq)': 0.3,
    'Background sample size for Friedman H-statistic interactions ("friedman_h" method only)': 0.3,
    'Background sample size for marker effect (shapiq_predictor method only)': 0.3,
    'Number of coalition samples for marker effect (shapiq)': 0.5,
    'Number of coalition samples for Shapley scores (nsamples)': 0.5,
    'Number of coalition samples for marker effect (shapiq_predictor method only)': 0.5,
    'Coalition budget per explained row (shapiq method only)': 0.5,
    'Coalition budget per explained row (shapiq methods only)': 0.5,
    'Coalition budget per explained row for interaction (shapiq)': 0.5,
    'Coalition sample budget for 3-way-or-higher shapiq_tree search (shapiq_tree method only)': 0.5,
    'Max markers considered for interaction search ("all" for every marker)': 1.0,
    'Max markers considered for Shapley scores ("all" for every marker)': 1.0,
    'Max markers considered for marker effect (shapiq) ("all" for every marker)': 1.0,
    'Number of individuals used to explain the model (effect method "shap" or "shapiq" only)': 0.5,
    'Number of individuals used to explain the surrogate model': 0.5,
    'Number of individuals used to explain the model': 0.5,
}

# Labels whose value does not change the cost: they do not split the signature.
COST_NEUTRAL_LABELS = frozenset({
    'Burn-in',
    'Prior degrees of freedom (df0)',
    'Expected proportion of variance explained (R2)',
    'Prior probability of a nonzero effect (probIn)',
    'Prior counts (counts)',
    'Kernel bandwidth (h)',
    'Dropout',
    'Decay',
    'L2 regularisation',
    'L2 regularisation (reg_lambda)',
})

# Confidence-tier inflation (log scale) and the tier-D / tier-C-from-prior sd.
_TIER_B_COMMON = 0.15
_TIER_C_COMMON = 0.5
_TIER_C_PRIOR_SD = 0.7
_TIER_D_SD = 1.0
# Memory-only log-sd used when no memory posterior exists (tier D, and tier C whose
# memory is still prior-only).  Lower than the seconds sd because the R-model memory
# prior is anchored on n*p matrix sizes; with sd=1.0 and a 0.98 quantile the request
# was inflated ~7.8x on the increments (user report: memory over-allocated).
_TIER_D_MEM_SD = 0.5
_SPAN_RATIO_FOR_TIER_A = 1.3
_MIN_OBS_FOR_TIER_A = 3

# NIG prior for sigma^2: InvGamma(2, 0.1^2).
_A0 = 2.0
_B0 = 0.1 ** 2
_SIGMA_REF2 = _B0 / (_A0 - 1.0)

_TIERS = ('A', 'B', 'C', 'D')
_FLOOR_SECONDS = 1e-3
# Process memory before any model runs: Python + R + torch runtimes, and the number
# of float64 copies of the n_train x p table held at once when nothing was measured
# (provisional; replaced by the measured per-machine coefficient when records exist).
_BASELINE_RUNTIME_GB = 0.8
_BASELINE_COPIES = 3.0

# ---------------------------------------------------------------------------
# Validation of the driver table against HPARAM_SPECS (AC2.3 / P8)
# ---------------------------------------------------------------------------


def unresolved_cost_driver_labels() -> List[str]:
    """Labels in :data:`COST_DRIVERS` / :data:`COST_NEUTRAL_LABELS` that no
    ``HPARAM_SPECS`` entry carries (a typo or a renamed field)."""
    known = {spec.get('label') for specs in HPARAM_SPECS.values() for spec in specs}
    missing = [label for label in COST_DRIVERS if label not in known]
    missing += [label for label in sorted(COST_NEUTRAL_LABELS) if label not in known]
    return missing


def assert_cost_drivers_resolve() -> None:
    """Test hook: raise ``AssertionError`` naming every unresolved label."""
    missing = unresolved_cost_driver_labels()
    assert not missing, f"COST_DRIVERS / COST_NEUTRAL_LABELS labels not in HPARAM_SPECS: {missing}"


def _drop_unresolved_labels() -> None:
    """Runtime behaviour of the validation: print one WARNING and drop."""
    global COST_NEUTRAL_LABELS
    missing = unresolved_cost_driver_labels()
    if not missing:
        return
    print(f"[calibration] WARNING: {len(missing)} cost-driver label(s) are not in HPARAM_SPECS "
          f"({missing}) - they are ignored, so a change of that field will not scale the "
          f"prediction. Fix resource_calibration.COST_DRIVERS.")
    for label in missing:
        COST_DRIVERS.pop(label, None)
    COST_NEUTRAL_LABELS = frozenset(label for label in COST_NEUTRAL_LABELS if label not in missing)


_drop_unresolved_labels()


def _base_name(model: str) -> str:
    """Base model name for ``model`` (strips a ``__<algorithm>`` suffix)."""
    text = str(model)
    if text in HPARAM_SPECS:
        return text
    head = text.split('__', 1)[0]
    return head if head in HPARAM_SPECS else text


def _param_value(params: Optional[Sequence], index: int, spec: Mapping) -> Any:
    """``params[index]``, else the spec default (short / padded lists)."""
    try:
        return params[index] if params is not None else spec.get('default')
    except (IndexError, KeyError, TypeError):
        return spec.get('default')


def _driver_layout(base: str) -> List[Tuple[int, str, Optional[Tuple[int, Any]]]]:
    """``[(param index, label, depends_on)]`` of the cost drivers of ``base``,
    resolved through HPARAM_SPECS labels."""
    layout = []
    for index, spec in enumerate(HPARAM_SPECS.get(base, [])):
        label = spec.get('label')
        if label in COST_DRIVERS:
            layout.append((index, label, spec.get('depends_on')))
    return layout


def _is_active(params: Optional[Sequence], base: str,
               depends_on: Optional[Tuple[int, Any]]) -> bool:
    if not depends_on:
        return True
    try:
        index, wanted = depends_on
        specs = HPARAM_SPECS[base]
        return _param_value(params, int(index), specs[int(index)]) == wanted
    except (IndexError, KeyError, TypeError, ValueError):
        return True


def _as_number(value: Any, p: float) -> Optional[float]:
    """A numeric driver value; ``'all'`` means every marker (``p``)."""
    if isinstance(value, str):
        if value.strip().lower() == 'all':
            return float(p)
        try:
            return float(value)
        except ValueError:
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def driver_log_ratios(base: str, params: Optional[Sequence], p: float) -> List[float]:
    """``log(value / reference)`` per cost driver of ``base`` (0.0 when the
    driver is inactive, unreadable or equal to the spec default)."""
    ratios = []
    for index, label, depends_on in _driver_layout(base):
        spec = HPARAM_SPECS[base][index]
        if not _is_active(params, base, depends_on):
            ratios.append(0.0)
            continue
        value = _as_number(_param_value(params, index, spec), p)
        reference = _as_number(spec.get('default'), p)
        if value is None or reference is None:
            ratios.append(0.0)
            continue
        ratios.append(math.log(max(value, 1.0) / max(reference, 1.0)))
    return ratios


def pool_signature(ld_prune: Any, rf_filter: Any, marker_source: Any, genotype_format: Any,
                   asreml: Any, joint: Any) -> str:
    """Signature of the config switches that change the per-task cost of
    Steps 1-8 (everything that is not a model fit).

    Why one function: ``GP()`` writes this into every telemetry record and the
    advisor must look it up with the SAME string, so both call this helper
    rather than each building the hash by hand.
    """
    blob = json.dumps({'ld': ld_prune, 'rf': rf_filter, 'src': marker_source,
                       'fmt': genotype_format, 'asreml': bool(asreml), 'joint': bool(joint)},
                      sort_keys=True, default=str)
    return hashlib.sha1(blob.encode('utf-8')).hexdigest()[:10]


def _device_class(base: str, device: Optional[str]) -> str:
    """``'cuda'`` / ``'cpu'`` for the models whose cost depends on the device
    (MLP and the GAT family), ``''`` for every other model.

    Why: ``'cuda'`` and ``'cuda:0'`` are the same cost class, and a CPU-only
    model such as rrBLUP costs the same on a GPU node, so putting the raw
    device string into every signature would needlessly drop a perfectly
    good calibration to a lower tier.
    """
    if not (base == 'MLP' or base.startswith('GAT_')):
        return ''
    return 'cuda' if str(device or '').lower().startswith('cuda') else 'cpu'


def structural_sig(base: str, params: Optional[Sequence], hp_tune: Optional[Mapping],
                   device: Optional[str]) -> str:
    """Signature of everything that changes the COST STRUCTURE of a model.

    Hashes, for ``base``: every hyper-parameter that is neither a numeric cost
    driver nor cost-neutral (by HPARAM_SPECS label), the tuning configuration
    and the device. Numeric drivers are excluded because their effect is
    modelled by a slope; anything NOT declared is included, so an unrecognised
    change produces a new signature (a lower tier) instead of a wrong reuse.
    """
    base = _base_name(base)
    parts: List[Any] = []
    for index, spec in enumerate(HPARAM_SPECS.get(base, [])):
        label = spec.get('label')
        if label in COST_DRIVERS or label in COST_NEUTRAL_LABELS:
            continue
        parts.append([label, _canon(_param_value(params, index, spec))])
    blob = json.dumps({'base': base, 'params': parts, 'hp_tune': _canon(hp_tune),
                       'device': _device_class(base, device)},
                      sort_keys=True, separators=(',', ':'))
    return hashlib.sha1(blob.encode('utf-8')).hexdigest()[:12]


_SIG_CACHE: Dict[str, str] = {}


def _cached_sig(base: str, params: Optional[Sequence], hp_tune: Optional[Mapping],
                device: Optional[str]) -> str:
    """:func:`structural_sig` memoised on the JSON of its inputs (a fit sees the
    same few configurations in thousands of records)."""
    try:
        key = base + '|' + json.dumps([params, hp_tune, device], sort_keys=True, default=str)
    except (TypeError, ValueError):
        return structural_sig(base, params, hp_tune, device)
    sig = _SIG_CACHE.get(key)
    if sig is None:
        if len(_SIG_CACHE) > 50000:
            _SIG_CACHE.clear()
        sig = structural_sig(base, params, hp_tune, device)
        _SIG_CACHE[key] = sig
    return sig


def _canon(value: Any) -> Any:
    """Canonical JSON-able form (a tuple and a list hash the same; floats that
    are whole numbers hash like ints)."""
    if isinstance(value, Mapping):
        return {str(k): _canon(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_canon(v) for v in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


# ---------------------------------------------------------------------------
# Normal-Inverse-Gamma regression
# ---------------------------------------------------------------------------

@dataclass
class _Posterior:
    """Posterior of one log-linear regression."""
    mean: np.ndarray          # posterior mean of the coefficients
    precision: np.ndarray     # Lambda_n
    a: float
    b: float
    n_obs: int

    def predict(self, x: np.ndarray) -> Tuple[float, float, float, float]:
        """``(mu, sd_common, sd_idio, df)`` on the log scale at covariates ``x``.

        ``sd_idio**2 + sd_common**2`` is the variance of the Student-t
        predictive, ``E[sigma^2] (1 + x' Lambda_n^-1 x)``.
        """
        mu = float(x @ self.mean)
        quad = float(x @ np.linalg.solve(self.precision, x))
        sigma2 = self.b / max(self.a - 1.0, 1e-9)
        return mu, math.sqrt(max(sigma2 * quad, 0.0)), math.sqrt(max(sigma2, 0.0)), 2.0 * self.a


def _nig_fit(X: np.ndarray, y: np.ndarray, mu0: np.ndarray, sd0: np.ndarray) -> _Posterior:
    """Normal-Inverse-Gamma posterior (closed form).

    Prior: ``b | s2 ~ N(mu0, s2 V0)`` with ``V0 = diag(sd0^2 / s2_ref)`` so that
    at ``s2 = s2_ref`` the prior sd of coefficient k is ``sd0[k]``;
    ``s2 ~ InvGamma(_A0, _B0)``. ``X`` may have zero rows (prior only).
    """
    prior_precision = np.diag(_SIGMA_REF2 / np.square(sd0))
    precision = prior_precision + X.T @ X
    rhs = prior_precision @ mu0 + X.T @ y
    mean = np.linalg.solve(precision, rhs)
    b = _B0 + 0.5 * float(y @ y + mu0 @ prior_precision @ mu0 - mean @ precision @ mean)
    # Shape: the textbook value is a0 + m/2. That ignores the degrees of freedom the
    # coefficients use up and counts the prior's own pseudo-observations, which makes
    # the predictive intervals too narrow for small m (measured: 90 % intervals covered
    # 74-88 %). Using the EFFECTIVE number of fitted parameters, k_eff = tr(X Lam_n^-1 X'),
    # gives a nearly unbiased sigma^2 (covers 87-91 % for m >= 25); the shape never falls
    # below the prior's a0, so a nearly empty fit keeps the prior (blueprint R2).
    if len(y) > 0:
        k_eff = float(np.trace(np.linalg.solve(precision, X.T @ X)))
        a = max(_A0, 1.0 + 0.5 * (len(y) - k_eff))
    else:
        a = _A0
    return _Posterior(mean=mean, precision=precision, a=a, b=max(b, 1e-12), n_obs=int(len(y)))


# ---------------------------------------------------------------------------
# Records -> observations
# ---------------------------------------------------------------------------

def load_telemetry(dirs: Sequence[str], notes: Optional[List[str]] = None) -> List[dict]:
    """Read every ``*.jsonl`` telemetry file under ``dirs``.

    A directory may be a telemetry folder itself or a ``Result/<name>`` folder
    (its ``logs/resource_telemetry`` is then used). Files are read oldest
    first and lines in order, so a later record of the same task wins when
    :func:`fit` de-duplicates. Malformed lines are skipped; a count is appended
    to ``notes``.
    """
    records: List[dict] = []
    bad_lines = 0
    unreadable = 0
    files: List[Tuple[float, str]] = []
    for directory in dirs or []:
        directory = str(directory)
        nested = os.path.join(directory, 'logs', 'resource_telemetry')
        folder = nested if os.path.isdir(nested) else directory
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if name.endswith('.jsonl'):
                path = os.path.join(folder, name)
                try:
                    files.append((os.path.getmtime(path), path))
                except OSError:
                    unreadable += 1
    for _, path in sorted(files):
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except ValueError:
                        bad_lines += 1
                        continue
                    if isinstance(record, dict) and 'kind' in record:
                        records.append(record)
                    else:
                        bad_lines += 1
        except OSError:
            unreadable += 1
    if notes is not None:
        if bad_lines:
            notes.append(f"{bad_lines} malformed telemetry line(s) skipped")
        if unreadable:
            notes.append(f"{unreadable} telemetry file(s) could not be read")
    return records


def _task_key(record: Mapping) -> Tuple:
    task = record.get('task') or {}
    return (task.get('population'), task.get('phenotype'), json.dumps(task.get('ratio')),
            json.dumps(task.get('sample')), json.dumps(task.get('option')))


def _threads_of(record: Mapping) -> int:
    for key in ('n_jobs', 'blas_threads', 'torch_threads'):
        value = record.get(key)
        if isinstance(value, (int, float)) and value >= 1:
            return int(value)
    return 1


@dataclass
class _Obs:
    machine: str
    base: str
    sig: str
    x_sec: np.ndarray
    y_sec: Optional[float]
    x_mem: np.ndarray
    y_mem: Optional[float]
    n_train: float
    status: str


def _seconds_covariates(base: str, params: Optional[Sequence], n_train: float, p: float,
                        threads: int, workers: int) -> np.ndarray:
    cap = THREAD_CAP.get(base, 1)
    effective = max(1, min(int(threads), cap))
    row = [1.0, math.log(max(n_train, 1.0) / N_REF), math.log(max(p, 1.0) / P_REF)]
    row += driver_log_ratios(base, params, p)
    row += [math.log(effective), math.log(max(int(workers), 1))]
    return np.array(row, dtype=float)


def _memory_covariates(n_train: float, p: float) -> np.ndarray:
    n_train = max(n_train, 1.0)
    p = max(p, 1.0)
    return np.array([1.0, math.log(n_train * p / (N_REF * P_REF)), math.log(n_train / N_REF)])


def _seconds_prior(base: str, params: Optional[Sequence], cpu_factor: float
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Prior mean and sd of the seconds-regression coefficients of ``base``."""
    slope_n, slope_p = SCALING_PRIOR.get(base, _UNKNOWN_SCALING)
    drivers = _driver_layout(base)
    mean = [math.log(max(cpu_factor, 1e-3) * PRIOR_REF_SECONDS), slope_n, slope_p]
    mean += [COST_DRIVERS[label] for _, label, _ in drivers]
    mean += [_THREAD_SLOPE_PRIOR[0], _WORKER_SLOPE_PRIOR[0]]
    sd = [_INTERCEPT_PRIOR_SD, SLOPE_PRIOR_SD, SLOPE_PRIOR_SD]
    sd += [SLOPE_PRIOR_SD] * len(drivers)
    sd += [_THREAD_SLOPE_PRIOR[1], _WORKER_SLOPE_PRIOR[1]]
    return np.array(mean, dtype=float), np.array(sd, dtype=float)


def _memory_prior(base: str, peak_mem_gb: float) -> Tuple[np.ndarray, np.ndarray]:
    slope_np, slope_n = MEM_PRIOR.get(base, _UNKNOWN_MEM)
    increment = max(peak_mem_gb - _PRIOR_BASE_GB, _MIN_MEM_INCREMENT_GB * 4)
    return (np.array([math.log(increment), slope_np, slope_n]),
            np.array([_INTERCEPT_PRIOR_SD, MEM_SLOPE_PRIOR_SD, MEM_SLOPE_PRIOR_SD]))


def _memory_target(record: Mapping) -> Optional[float]:
    peak = record.get('rss_tree_peak_gb')
    if peak is None:
        peak = record.get('rss_self_peak_gb')
    if peak is None:
        return None
    baseline = record.get('rss_baseline_gb') or 0.0
    return math.log(max(float(peak) - float(baseline), _MIN_MEM_INCREMENT_GB))


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Pred:
    """Prediction for ONE task of ONE model.

    ``sec_mu`` is the expected seconds (log-normal mean over the task-to-task
    scatter ``sd_idio``); ``sd_common`` (log) is the shared-parameter
    uncertainty, fully correlated across tasks; ``mem_mu`` is the expected
    peak-memory increment in GB and ``mem_sd`` its total log sd. ``tier`` is
    'A'..'D'; ``n_obs`` the number of measured tasks behind it. ``mu_log`` and
    ``df`` give the full Student-t predictive for :meth:`quantile`.
    """
    sec_mu: float
    mem_mu: float
    sd_common: float
    sd_idio: float
    mem_sd: float
    tier: str
    n_obs: int
    mu_log: float = 0.0
    df: float = 4.0

    def quantile(self, q: float) -> float:
        """Seconds at predictive quantile ``q`` of one new task (common and
        task-to-task uncertainty together)."""
        total = math.sqrt(self.sd_common ** 2 + self.sd_idio ** 2)
        scale = total * math.sqrt(max(self.df - 2.0, 1e-9) / self.df)
        return math.exp(self.mu_log + _t_ppf(q, self.df) * scale)


def _t_ppf(q: float, df: float) -> float:
    """Student-t quantile (Cornish-Fisher expansion about the normal one;
    accurate to a few 1e-3 for df >= 4, which the NIG prior guarantees)."""
    q = min(max(q, 1e-9), 1.0 - 1e-9)
    z = statistics.NormalDist().inv_cdf(q)
    g1 = (z ** 3 + z) / 4.0
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96.0
    g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384.0
    return z + g1 / df + g2 / df ** 2 + g3 / df ** 3


def normal_ppf(q: float) -> float:
    """Standard-normal quantile (shared with the estimator)."""
    return statistics.NormalDist().inv_cdf(min(max(q, 1e-9), 1.0 - 1e-9))


@dataclass
class _Group:
    base: str
    sig: str
    machine: str
    sec: Optional[_Posterior]
    mem: Optional[_Posterior]
    n_obs: int
    spans_n: bool


@dataclass
class Calibration:
    """A fitted calibration (see the module docstring)."""
    machine_key: str
    priors: Mapping[str, Mapping[str, float]]
    groups: Dict[Tuple[str, str, str], _Group] = field(default_factory=dict)
    machine_factor: Dict[str, float] = field(default_factory=dict)
    pseudo: Dict[Tuple[str, str, str], _Posterior] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    n_records: int = 0
    # Per-machine median of (baseline RSS - process base) per GB of n_train*p*8 bytes.
    baseline_c: Dict[str, float] = field(default_factory=dict)

    # -- prior helpers -----------------------------------------------------

    def _prior_for(self, base: str) -> Mapping[str, float]:
        return self.priors.get(base) or _DEFAULT_PRIORS.get(base) or _UNKNOWN_PRIOR

    def baseline_gb(self, n_train: float, p: float) -> Tuple[float, bool]:
        """Resident memory (GB) a task process already holds when its first model
        starts (the loaded genotype table, the train/test copies, the Python/R/torch
        runtimes), as ``(gb, measured)``.

        ``measured`` is True when this machine has telemetry records to scale from,
        otherwise the documented default ``_BASELINE_COPIES`` applies. Why: the
        per-model memory model predicts only the INCREMENT over this baseline
        (blueprint R2), but the HPC memory request must cover the whole process.
        """
        data_gb = max(float(n_train), 1.0) * max(float(p), 1.0) * 8.0e-9
        coefficient = self.baseline_c.get(self.machine_key)
        if coefficient is None:
            return _BASELINE_RUNTIME_GB + _BASELINE_COPIES * data_gb, False
        return _BASELINE_RUNTIME_GB + coefficient * data_gb, True

    def has_data(self) -> bool:
        """True when at least one measured model group exists."""
        return bool(self.groups)

    def models_measured(self) -> List[str]:
        """Sorted base models that have at least one group on this machine."""
        return sorted({g.base for g in self.groups.values() if g.machine == self.machine_key})

    # -- prediction --------------------------------------------------------

    def predict(self, model: str, params: Optional[Sequence], hp_tune: Optional[Mapping],
                device: Optional[str], n_train: int, p: int, threads: int,
                workers: int) -> Pred:
        """Predict one task of ``model`` at size ``(n_train, p)``.

        ``threads`` is the run's per-task thread count (``N_JOBS`` / BLAS /
        torch threads), ``workers`` the number of concurrent task workers.
        Never raises: an unknown model falls back to the tier-D default prior.
        """
        base = _base_name(model)
        prior = self._prior_for(base)
        sig = structural_sig(base, params, hp_tune, device)
        x_sec = _seconds_covariates(base, params, float(n_train), float(p), threads, workers)
        x_mem = _memory_covariates(float(n_train), float(p))
        own = self.groups.get((self.machine_key, base, sig))
        if own is not None and own.sec is not None:
            return self._from_group(
                own, x_sec, x_mem, tier=('A' if _tier_a(own) else 'B'), shift=0.0,
                extra_common=(0.0 if _tier_a(own) else _TIER_B_COMMON))
        others = [g for (machine, b, s), g in self.groups.items()
                  if b == base and s == sig and machine != self.machine_key]
        factor_here = self.machine_factor.get(self.machine_key)
        if others:
            donor = max(others, key=lambda g: g.n_obs)
            shift = 0.0
            if factor_here is not None and donor.machine in self.machine_factor:
                shift = factor_here - self.machine_factor[donor.machine]
            return self._from_group(donor, x_sec, x_mem, tier='C', shift=shift,
                                    extra_common=_TIER_C_COMMON)
        return self._from_prior(base, params, prior, x_sec, x_mem, factor_here,
                                mem_posterior=(own.mem if own is not None else None))

    def _from_group(self, group: _Group, x_sec: np.ndarray, x_mem: np.ndarray, tier: str,
                    shift: float, extra_common: float) -> Pred:
        mu, sd_c, sd_i, df = group.sec.predict(x_sec)
        mem_mu, mem_sd = self._mem_from(group.mem, group.base, x_mem)
        sd_c = math.hypot(sd_c, extra_common)
        mu += shift
        return Pred(sec_mu=max(math.exp(mu + 0.5 * sd_i ** 2), _FLOOR_SECONDS),
                    mem_mu=mem_mu, sd_common=sd_c, sd_idio=sd_i,
                    mem_sd=math.hypot(mem_sd, extra_common), tier=tier,
                    n_obs=group.n_obs, mu_log=mu, df=df)

    def _mem_from(self, posterior: Optional[_Posterior], base: str,
                  x_mem: np.ndarray) -> Tuple[float, float]:
        if posterior is None:
            prior = self._prior_for(base)
            mean, sd = _memory_prior(base, float(prior['peak_mem_gb_per_worker']))
            posterior = _nig_fit(np.zeros((0, 3)), np.zeros(0), mean, sd)
        mu, sd_c, sd_i, _ = posterior.predict(x_mem)
        sd_total = math.hypot(sd_c, sd_i)
        return math.exp(mu + 0.5 * sd_i ** 2), sd_total

    def _from_prior(self, base: str, params: Optional[Sequence], prior: Mapping[str, float],
                    x_sec: np.ndarray, x_mem: np.ndarray, factor_here: Optional[float],
                    mem_posterior: Optional[_Posterior] = None) -> Pred:
        mean, sd = _seconds_prior(base, params, float(prior['cpu_time_factor']))
        mu = float(x_sec @ mean)
        sd_idio = math.sqrt(_SIGMA_REF2 * 4.0)
        if factor_here is not None:
            tier, sd_c, mu = 'C', _TIER_C_PRIOR_SD, mu + factor_here
        else:
            tier, sd_c = 'D', _TIER_D_SD
        mem_mu, mem_sd = self._mem_from(mem_posterior, base, x_mem)
        return Pred(sec_mu=max(math.exp(mu + 0.5 * sd_idio ** 2), _FLOOR_SECONDS),
                    mem_mu=mem_mu, sd_common=sd_c, sd_idio=sd_idio,
                    mem_sd=(mem_sd if mem_posterior is not None else _TIER_D_MEM_SD), tier=tier,
                    n_obs=0, mu_log=mu, df=2.0 * _A0)

    def predict_pseudo(self, name: str, pool_sig: Optional[str], n: float, p: float,
                       workers: int, default_seconds: float) -> Pred:
        """Predict a non-model cost: ``'__pool__'`` (Steps 1-8 per task),
        ``'__asreml__'``, ``'__joint__'`` (per task), ``'__startup__'`` (per job)
        or ``'__tail__'`` (per task: ensemble finalisation, saving).

        ``default_seconds`` is the tier-D guess used when nothing was measured.
        """
        posterior = self.pseudo.get((self.machine_key, name, str(pool_sig)))
        x = np.array([1.0, math.log(max(n, 1.0) / N_REF), math.log(max(p, 1.0) / P_REF),
                      math.log(max(int(workers), 1))])
        if name in ('__startup__', '__tail__'):
            x = np.array([1.0])
        if posterior is None:
            sd_idio = math.sqrt(_SIGMA_REF2 * 4.0)
            mu = math.log(max(default_seconds, _FLOOR_SECONDS))
            return Pred(sec_mu=max(default_seconds, _FLOOR_SECONDS), mem_mu=0.0,
                        sd_common=_TIER_D_SD, sd_idio=sd_idio, mem_sd=_TIER_D_MEM_SD, tier='D',
                        n_obs=0, mu_log=mu, df=2.0 * _A0)
        mu, sd_c, sd_i, df = posterior.predict(x)
        tier = 'A' if posterior.n_obs >= _MIN_OBS_FOR_TIER_A else 'B'
        if tier == 'B':
            sd_c = math.hypot(sd_c, _TIER_B_COMMON)
        return Pred(sec_mu=max(math.exp(mu + 0.5 * sd_i ** 2), _FLOOR_SECONDS), mem_mu=0.0,
                    sd_common=sd_c, sd_idio=sd_i, mem_sd=sd_c, tier=tier,
                    n_obs=posterior.n_obs, mu_log=mu, df=df)


def _tier_a(group: _Group) -> bool:
    return group.n_obs >= _MIN_OBS_FOR_TIER_A and group.spans_n


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------

def _dedupe(records: Iterable[Mapping]) -> List[Mapping]:
    """Keep ONE record per (machine, task, kind[, model, signature]).

    A later ``ok`` record replaces an earlier one (a resumed task counts once,
    AC2.6) and a ``failed`` record is dropped when the same task also has an
    ``ok`` one (the resume re-ran it).
    """
    kept: Dict[Tuple, Mapping] = {}
    for record in records:
        kind = record.get('kind')
        machine = record.get('machine_key')
        ok = record.get('status') == 'ok'
        if kind == 'model':
            base = _base_name(record.get('base') or record.get('model') or '')
            sig = _cached_sig(base, record.get('params'), record.get('hp_tune'),
                              record.get('device'))
            key = ('model', machine, _task_key(record), record.get('model'), sig)
        elif kind == 'task':
            key = ('task', machine, _task_key(record))
        elif kind == 'job':
            key = ('job', machine, record.get('run_id'))
            ok = True
        else:
            continue
        previous = kept.get(key)
        if previous is not None and previous.get('status') == 'ok' and not ok:
            continue                      # never let a failure replace a success
        kept[key] = record
    return list(kept.values())


def _model_observations(records: Sequence[Mapping], notes: List[str]
                        ) -> List[_Obs]:
    observations: List[_Obs] = []
    skipped = 0
    for record in records:
        if record.get('kind') != 'model':
            continue
        n_train, p = record.get('n_train'), record.get('p')
        if not isinstance(n_train, (int, float)) or not isinstance(p, (int, float)) \
                or n_train <= 0 or p <= 0:
            skipped += 1
            continue
        base = _base_name(record.get('base') or record.get('model') or '')
        params = record.get('params')
        sig = _cached_sig(base, params, record.get('hp_tune'), record.get('device'))
        status = str(record.get('status'))
        seconds = record.get('seconds')
        y_sec = (math.log(max(float(seconds), _FLOOR_SECONDS))
                 if status == 'ok' and isinstance(seconds, (int, float)) else None)
        workers = record.get('workers') or 1
        x_sec = _seconds_covariates(base, params, float(n_train), float(p),
                                    _threads_of(record), int(workers))
        observations.append(_Obs(
            machine=str(record.get('machine_key')), base=base, sig=sig, x_sec=x_sec,
            y_sec=y_sec, x_mem=_memory_covariates(float(n_train), float(p)),
            y_mem=_memory_target(record), n_train=float(n_train), status=status))
    if skipped:
        notes.append(f"{skipped} model record(s) without n_train/p skipped")
    return observations


def _fit_group(base: str, group_obs: Sequence[_Obs], params_for_prior: Optional[Sequence],
               prior: Mapping[str, float], notes: List[str]) -> _Group:
    ok = [o for o in group_obs if o.y_sec is not None]
    sec = None
    if ok:
        mean, sd = _seconds_prior(base, params_for_prior, float(prior['cpu_time_factor']))
        sec = _nig_fit(np.vstack([o.x_sec for o in ok]), np.array([o.y_sec for o in ok]),
                       mean, sd)
    mem_mean, mem_sd = _memory_prior(base, float(prior['peak_mem_gb_per_worker']))
    mem_obs = [o for o in group_obs if o.status == 'ok' and o.y_mem is not None]
    mem = _nig_fit(np.vstack([o.x_mem for o in mem_obs]) if mem_obs else np.zeros((0, 3)),
                   np.array([o.y_mem for o in mem_obs]), mem_mean, mem_sd)
    # A failed model's memory is a LOWER bound (AC2.4): use it only when it
    # exceeds what the successful runs predict, so it can only raise the figure.
    failed = [o for o in group_obs if o.status != 'ok' and o.y_mem is not None]
    useful = [o for o in failed if o.y_mem > float(o.x_mem @ mem.mean)]
    if useful:
        mem_obs = mem_obs + useful
        mem = _nig_fit(np.vstack([o.x_mem for o in mem_obs]), np.array([o.y_mem for o in mem_obs]),
                       mem_mean, mem_sd)
        notes.append(f"{base}: {len(useful)} failed run(s) used as memory lower bound")
    sizes = [o.n_train for o in group_obs]
    spans = bool(sizes) and (max(sizes) / max(min(sizes), 1.0) >= _SPAN_RATIO_FOR_TIER_A)
    return _Group(base=base, sig=group_obs[0].sig, machine=group_obs[0].machine, sec=sec,
                  mem=mem, n_obs=len(ok), spans_n=spans)


def _pseudo_observations(records: Sequence[Mapping]) -> Dict[Tuple[str, str, str], List[Tuple]]:
    """Observations of the non-model costs, from task and job records.

    ``__pool__`` = task wall time minus the model records of the same task
    (and ASReml / joint-optimisation time). ``__asreml__`` and ``__joint__``
    come from those stages of the task record; ``__startup__`` from the job
    record's ``startup_s``; ``__tail__`` from ``tail_s`` per completed task.
    Each observation is ``(x, log seconds)``.
    """
    model_seconds: Dict[Tuple, float] = {}
    for record in records:
        if record.get('kind') == 'model' and record.get('status') == 'ok':
            key = (record.get('run_id'), _task_key(record))
            model_seconds[key] = model_seconds.get(key, 0.0) + float(record.get('seconds') or 0.0)
    out: Dict[Tuple[str, str, str], List[Tuple]] = {}

    def add(machine, name, pool_sig, x, seconds):
        if seconds is None or seconds <= 0:
            return
        out.setdefault((str(machine), name, str(pool_sig)), []).append(
            (np.array(x, dtype=float), math.log(max(seconds, _FLOOR_SECONDS))))

    for record in records:
        machine, pool_sig = record.get('machine_key'), record.get('pool_sig')
        kind = record.get('kind')
        if kind == 'task' and record.get('status') == 'ok':
            n_task, p_raw = record.get('n_task'), record.get('p_raw')
            if not n_task or not p_raw:
                continue
            workers = record.get('workers') or 1
            stages = record.get('stages') or {}
            total = record.get('total_s')
            x = [1.0, math.log(max(n_task, 1.0) / N_REF), math.log(max(p_raw, 1.0) / P_REF),
                 math.log(max(int(workers), 1))]
            key = (record.get('run_id'), _task_key(record))
            if total is not None and key in model_seconds:
                pool = (float(total) - model_seconds[key] - float(stages.get('asreml', 0.0))
                        - float(stages.get('joint_opt', 0.0))
                        - float(stages.get('iter_ens', 0.0))    # ver4-57: opt-in Step 11c
                        - float(stages.get('super_learner', 0.0)))   # ver4-77: opt-in Step 11d
                add(machine, '__pool__', pool_sig, x, pool)
            if 'asreml' in stages:
                add(machine, '__asreml__', pool_sig, x, float(stages['asreml']))
            if 'joint_opt' in stages:
                add(machine, '__joint__', pool_sig, x, float(stages['joint_opt']))
        elif kind == 'job':
            if record.get('startup_s') is not None:
                add(machine, '__startup__', pool_sig, [1.0], float(record['startup_s']))
            done = record.get('n_tasks_done') or 0
            if done and record.get('tail_s') is not None:
                add(machine, '__tail__', pool_sig, [1.0], float(record['tail_s']) / done)
    return out


def _fit_pseudo(observations: Mapping[Tuple[str, str, str], List[Tuple]]
                ) -> Dict[Tuple[str, str, str], _Posterior]:
    fitted: Dict[Tuple[str, str, str], _Posterior] = {}
    for key, rows in observations.items():
        X = np.vstack([r[0] for r in rows])
        y = np.array([r[1] for r in rows])
        k = X.shape[1]
        mean = np.zeros(k)
        sd = np.full(k, _INTERCEPT_PRIOR_SD)
        mean[0] = float(np.mean(y))                  # weak prior centred on the data's own level
        if k == 4:
            mean[1:3] = (0.8, 0.8)                   # pool cost grows roughly linearly in n and p
            sd[1:4] = (SLOPE_PRIOR_SD, SLOPE_PRIOR_SD, 0.2)
        fitted[key] = _nig_fit(X, y, mean, sd)
    return fitted


def _machine_factors(groups: Mapping[Tuple[str, str, str], _Group],
                     observations: Sequence[_Obs], priors: Mapping[str, Mapping[str, float]]
                     ) -> Dict[str, float]:
    """Machine speed factor = median over its measured groups of
    ``log(observed) - log(tier-D prediction)`` at the group's own points."""
    residuals: Dict[str, List[float]] = {}
    by_group: Dict[Tuple[str, str, str], List[_Obs]] = {}
    for obs in observations:
        if obs.y_sec is not None:
            by_group.setdefault((obs.machine, obs.base, obs.sig), []).append(obs)
    prior_mean: Dict[str, np.ndarray] = {}
    for (machine, base, _), rows in by_group.items():
        if base not in prior_mean:
            prior = priors.get(base) or _DEFAULT_PRIORS.get(base) or _UNKNOWN_PRIOR
            prior_mean[base] = _seconds_prior(base, None, float(prior['cpu_time_factor']))[0]
        for obs in rows:
            residuals.setdefault(machine, []).append(
                obs.y_sec - float(obs.x_sec @ prior_mean[base]))
    return {machine: float(np.median(values)) for machine, values in residuals.items()}


def _baseline_coefficients(records: Sequence[Mapping]) -> Dict[str, float]:
    """Per machine: median of ``(rss_baseline_gb - runtime) / (n_train * p * 8e-9)``
    over the successful model records (a robust, one-number size scaling of the
    process baseline; see :meth:`Calibration.baseline_gb`)."""
    ratios: Dict[str, List[float]] = {}
    for record in records:
        if record.get('kind') != 'model' or record.get('status') != 'ok':
            continue
        baseline, n_train, p = record.get('rss_baseline_gb'), record.get('n_train'), record.get('p')
        if not all(isinstance(v, (int, float)) for v in (baseline, n_train, p)):
            continue
        if baseline <= 0 or n_train <= 0 or p <= 0:
            continue
        data_gb = float(n_train) * float(p) * 8.0e-9
        ratios.setdefault(str(record.get('machine_key')), []).append(
            max(float(baseline) - _BASELINE_RUNTIME_GB, 0.0) / data_gb)
    return {machine: float(np.median(values)) for machine, values in ratios.items()}


def fit(records: List[dict], machine_key: str,
        priors: Optional[Mapping[str, Mapping[str, float]]] = None) -> Calibration:
    """Fit a :class:`Calibration` from telemetry ``records`` for ``machine_key``.

    Args:
        records: output of :func:`load_telemetry` (any mix of kinds).
        machine_key: ``resource_telemetry.machine_key()`` of the machine the
            predictions are for.
        priors: optional ``{base: {'cpu_time_factor', 'peak_mem_gb_per_worker'}}``
            (resource_profiles passes its live ``MODEL_COST_PROFILE``); the
            module's own copy is used for anything missing.

    Returns:
        A :class:`Calibration`; with no usable records it is simply empty and
        every prediction is tier D. Malformed records are skipped and counted
        in ``Calibration.notes``. Never raises on bad data.
    """
    notes: List[str] = []
    merged_priors: Dict[str, Mapping[str, float]] = dict(_DEFAULT_PRIORS)
    for base, values in (priors or {}).items():
        fallback = _DEFAULT_PRIORS.get(base, _UNKNOWN_PRIOR)
        merged_priors[base] = {
            'cpu_time_factor': float(values.get('cpu_time_factor', fallback['cpu_time_factor'])),
            'peak_mem_gb_per_worker': float(values.get(
                'peak_mem_gb_per_worker', fallback['peak_mem_gb_per_worker'])),
        }
    clean = [r for r in (records or []) if isinstance(r, Mapping) and r.get('v', 1) == 1]
    if len(clean) != len(records or []):
        notes.append(f"{len(records or []) - len(clean)} record(s) with an unknown schema skipped")
    unique = _dedupe(clean)
    observations = _model_observations(unique, notes)
    grouped: Dict[Tuple[str, str, str], List[_Obs]] = {}
    params_by_group: Dict[Tuple[str, str, str], Optional[Sequence]] = {}
    model_records = [r for r in unique if r.get('kind') == 'model']
    # _model_observations skips some records, so pair params by rebuilding keys.
    for record in model_records:
        base = _base_name(record.get('base') or record.get('model') or '')
        sig = _cached_sig(base, record.get('params'), record.get('hp_tune'), record.get('device'))
        params_by_group.setdefault(
            (str(record.get('machine_key')), base, sig), record.get('params'))
    for obs in observations:
        grouped.setdefault((obs.machine, obs.base, obs.sig), []).append(obs)
    groups: Dict[Tuple[str, str, str], _Group] = {}
    for key, rows in grouped.items():
        prior = merged_priors.get(key[1]) or _UNKNOWN_PRIOR
        groups[key] = _fit_group(key[1], rows, params_by_group.get(key), prior, notes)
    factors = _machine_factors(groups, observations, merged_priors)
    pseudo = _fit_pseudo(_pseudo_observations(unique))
    return Calibration(machine_key=machine_key, priors=merged_priors, groups=groups,
                       machine_factor=factors, pseudo=pseudo, notes=notes,
                       n_records=len(unique), baseline_c=_baseline_coefficients(unique))
