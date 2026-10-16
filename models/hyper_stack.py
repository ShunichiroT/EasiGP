"""
models/hyper_stack.py
=====================
ver4-48, R1/R2 (design blueprint `EasiGP_ver4-48_Design_Blueprint.md`): the "neural stack" -
hyperparameter choices AND ensemble weights as the trainable weights of one small network.

WHY THIS MODULE EXISTS
    The literal idea - back-propagate through every model fit to its own hyperparameters - is
    impossible here (MCMC/BGLR, trees, libsvm SVR and integer-k KNN have no gradient path, and
    the MLP/GAT training loops return a result tuple, not a graph). What CAN be trained is a
    continuous relaxation of the choice: every model m has a small LIBRARY of L_m already
    fitted configurations (their out-of-fold prediction columns are given here as ``P``), and

        W   = softmax(a)            a in R^K           (model weights)
        q_m = softmax(b_m)          b_m in R^{L_m}     (hyperparameter gate of model m)
        yhat = sum_m W_m sum_l q_{m,l} P[:, (m, l)]    (column weight w_c = W_m * q_{m,l})
        loss(lam) = -ensemble_score(yhat, y) + lam * Omega
        Omega     = KL(uniform_K || W) + mean_{m: L_m > 1} ( -log q_{m,0} )

    Entry 0 of every model is its CONFIGURED (untuned) parameter set. At ``lam = inf`` the
    network collapses to W = uniform and q_m = delta_0, i.e. yhat is exactly the mean of the
    untuned columns - the naive ensemble - which is a closed form (`prior_fit`) and a member of
    the lambda path. ``lambda_path`` chooses lam by repeated K-fold cross-validation over the
    rows of ``P``, so the naive ensemble is returned unless a fitted network generalises better
    (the "super learner" oracle property; this is theory only - the measurable go/no-go is gate
    A-NS.0 in `tools/joint_replay.py`).

WHAT THIS MODULE DOES NOT DO
    It never fits a genomic-prediction model, never reads a test row and never touches a result
    accumulator (invariants I3, I7, I8). It is pure numerical code: float64, CPU, L-BFGS (no
    randomness), so two identical calls return bitwise-identical weights (I9, AC1.3).
    `models.joint_tuning._ensemble_score_of` stays the single definition of the score; the
    torch twin `torch_ensemble_score` below only exists because the optimiser needs gradients,
    and a parity test keeps the two in step.

ver4-50 (blueprint `EasiGP_ver4-50_Design_Blueprint.md`, R2/R3) adds two opt-in pieces, both
without any extra model fit: ``objective='pearson'`` (train the network on ranking only; see
`torch_pearson_score`) and the paired swap test (`swap_test` / `force_gate`: a hyperparameter
switch is admitted only when its cross-validated gain beats its own noise). The defaults of every
changed function reproduce ver4-49 bit for bit.

ver4-51 (blueprint `EasiGP_ver4-51_Design_Blueprint.md`, R1/R2) makes the deployment honest, again
without any model fit. (R1) ``refit_swap_test``: a hyperparameter switch is judged against a FAIR
counterfactual - the network REFITTED with that model's gate clamped to entry 0 (``fixed_gates``) -
on a fresh partition, instead of the fitted network with one gate overwritten (whose model weight
was fitted for the switched column, which handicaps the baseline in proportion to that weight).
(R2) ``stack_procedure`` is the whole post-processing of the all-rows fit (lambda path -> swap rule
-> shrink -> deployed column weights) as ONE pure function, and ``nested_gate`` runs that same
function on the inner rows of an outer cross-validation and scores the deployed weights on the
outer rows against the equal-weight ensemble: the fitted network is deployed only when the whole
procedure beats equal weights out of sample. The defaults of every changed function reproduce
ver4-50 bit for bit.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .ensemble_regularization import _fold_indices

# Probability mass given to the configured (entry 0) parameters at the default initialisation.
_INIT_Q0 = 0.95
# Smallest probability used when a fitted gate is turned back into logits (warm start).
_LOG_FLOOR = 1.0e-12
_RULES = ('min', 'one_se')
_OBJECTIVES = ('ensemble_score', 'pearson')
_DEPLOYS = ('mixture', 'argmax')
_SWAP_RULES = ('gate', 'paired_se', 'paired_refit')


class HyperStackFitError(RuntimeError):
    """The optimiser produced a non-finite loss or non-finite weights for one (lam, rows) fit.

    Callers (`lambda_path`, and `joint_tuning._neural_stack_search`) treat it as "this lambda
    failed" - never as a reason to abort the whole task."""


@dataclass(frozen=True, eq=False)
class HyperStackFit:
    """One fitted network.

    Attributes:
      w: (C,) column weights ``W_m * q_{m,l}``, non-negative, sums to 1, library order.
      w_model: (K,) model weights ``W`` (sums to 1).
      q: K arrays; ``q[m]`` is model m's hyperparameter gate (sums to 1; length L_m).
      lam: the regularisation strength this fit used (``inf`` for the closed-form prior).
      objective: final loss value ``-score + lam * Omega`` (``nan`` for the prior, which is
        not optimised).
      n_iter: L-BFGS iterations used (0 for the prior).
      converged: True when a tolerance criterion, not the iteration cap, stopped L-BFGS
        (always True for the prior).
    """
    w: np.ndarray
    w_model: np.ndarray
    q: List[np.ndarray]
    lam: float
    objective: float
    n_iter: int
    converged: bool


# --------------------------------------------------------------------------- #
# Library geometry
# --------------------------------------------------------------------------- #

def group_slices(sizes: Sequence[int]) -> List[slice]:
    """Column slice of every model in the library: ``sizes=[3, 1, 2]`` -> ``[0:3], [3:4], [4:6]``.

    Raises ValueError for an empty list or a size below 1."""
    sizes_i = [int(s) for s in sizes]
    if not sizes_i:
        raise ValueError('[HYPER_STACK] sizes is empty - the library has no models')
    if any(s < 1 for s in sizes_i):
        raise ValueError(f'[HYPER_STACK] every library size must be >= 1, got {sizes_i}')
    out: List[slice] = []
    start = 0
    for s in sizes_i:
        out.append(slice(start, start + s))
        start += s
    return out


def prior_fit(sizes: Sequence[int]) -> HyperStackFit:
    """The lambda = infinity network, in closed form: ``W = 1/K`` and ``q_m = delta_0``.

    The resulting column weights put ``1/K`` on entry 0 of every model and 0 elsewhere, so the
    combined prediction is exactly the mean of the untuned columns (the naive ensemble)."""
    slices = group_slices(sizes)
    k = len(slices)
    c = slices[-1].stop
    w = np.zeros(c, dtype=float)
    q: List[np.ndarray] = []
    for sl in slices:
        gate = np.zeros(sl.stop - sl.start, dtype=float)
        gate[0] = 1.0
        q.append(gate)
        w[sl.start] = 1.0 / k
    return HyperStackFit(w=w, w_model=np.full(k, 1.0 / k, dtype=float), q=q, lam=float('inf'),
                         objective=float('nan'), n_iter=0, converged=True)


def stack_penalty(w_model: np.ndarray, q: Sequence[np.ndarray]) -> float:
    """Omega = KL(uniform_K || W) + mean over models with L_m > 1 of ``-log q_{m,0}``.

    Zero at the prior (W uniform, q = delta_0). Used for diagnostics and for the
    "Omega never increases with lambda" check (AC1.6); the optimiser uses the torch version."""
    w_model = np.asarray(w_model, dtype=float)
    k = int(w_model.size)
    kl = -math.log(k) - float(np.mean(np.log(np.clip(w_model, _LOG_FLOOR, None))))
    gate_terms = [-math.log(max(float(np.asarray(g)[0]), _LOG_FLOOR))
                  for g in q if np.asarray(g).size > 1]
    gate = float(np.mean(gate_terms)) if gate_terms else 0.0
    return float(kl + gate)


# --------------------------------------------------------------------------- #
# The score (torch twin of joint_tuning._ensemble_score_of)
# --------------------------------------------------------------------------- #

def torch_ensemble_score(pred: "torch.Tensor", y: "torch.Tensor") -> "torch.Tensor":
    """``pearson - mse/var`` of ``pred`` against ``y`` as a differentiable 0-d tensor.

    Why: the optimiser needs a gradient; `joint_tuning._ensemble_score_of` (numpy) remains the
    ONE definition of the objective (I3) and a parity test (AC1.2) keeps this twin identical to
    it. Population standard deviation / variance (ddof=0), like the numpy function.

    A degenerate input (fewer than 2 rows, no variance in ``y`` or in ``pred``, non-finite
    value) returns ``nan`` - the numpy function returns ``None`` for the same cases."""
    n = int(pred.numel())
    nan = torch.full((), float('nan'), dtype=pred.dtype)
    if n < 2 or n != int(y.numel()):
        return nan
    pred_c = pred - pred.mean()
    y_c = y - y.mean()
    var_y = (y_c * y_c).mean()
    var_p = (pred_c * pred_c).mean()
    var_y_val = float(var_y.detach())
    var_p_val = float(var_p.detach())
    if not math.isfinite(var_y_val) or var_y_val <= 0.0:
        return nan
    if not math.isfinite(var_p_val) or var_p_val <= 0.0:
        return nan
    sd = torch.sqrt(var_p) * torch.sqrt(var_y)
    pearson = (pred_c * y_c).mean() / sd
    mse = ((pred - y) ** 2).mean()
    return pearson - mse / var_y


def torch_pearson_score(pred: "torch.Tensor", y: "torch.Tensor") -> "torch.Tensor":
    """Pearson correlation of ``pred`` and ``y`` as a differentiable 0-d tensor (ver4-50, R2).

    Why: ``pearson - mse/var`` rewards matching the DISPERSION of the CV rows, and the CV rows
    are far more predictable than the test rows (blueprint R1, hypothesis H1), so a simplex can
    only "widen" by moving weight onto the most dispersed column. Pearson is invariant to
    ``c * pred + d``, so the weights are trained on ranking alone. Population moments (ddof=0),
    like `joint_tuning._pearson_score_of`, which is the numpy definition this twin must match.

    A degenerate input (fewer than 2 rows, a row-count mismatch, no variance in ``y`` or in
    ``pred``, a non-finite value) returns ``nan`` - the numpy function returns ``None``."""
    n = int(pred.numel())
    nan = torch.full((), float('nan'), dtype=pred.dtype)
    if n < 2 or n != int(y.numel()):
        return nan
    pred_c = pred - pred.mean()
    y_c = y - y.mean()
    var_y = (y_c * y_c).mean()
    var_p = (pred_c * pred_c).mean()
    var_y_val = float(var_y.detach())
    var_p_val = float(var_p.detach())
    if not math.isfinite(var_y_val) or var_y_val <= 0.0:
        return nan
    if not math.isfinite(var_p_val) or var_p_val <= 0.0:
        return nan
    sd = torch.sqrt(var_p) * torch.sqrt(var_y)
    return (pred_c * y_c).mean() / sd


# Single registry of the network's objectives (the numpy twins live in
# `joint_tuning._NS_SCORE_FNS`; `lambda_path` receives the numpy one as ``score_fn``).
_TORCH_SCORES = {'ensemble_score': torch_ensemble_score, 'pearson': torch_pearson_score}


def _check_objective(objective: str) -> str:
    """Return ``objective`` if it names a known torch score, else raise ValueError."""
    if objective not in _TORCH_SCORES:
        raise ValueError(f"[HYPER_STACK] unknown objective {objective!r} "
                         f"(expected one of {list(_OBJECTIVES)!r})")
    return objective


# --------------------------------------------------------------------------- #
# Fitting one network
# --------------------------------------------------------------------------- #

def _default_theta(sizes: Sequence[int], fixed: Optional[Mapping[int, int]] = None) -> np.ndarray:
    """Default start: ``a = 0`` (W uniform) and per gate ``b = [log(19 (L-1)), 0, ..., 0]``
    (q_0 = 0.95); models with L = 1 have no gate parameters, and neither has a model whose gate
    is fixed (ver4-51, ``fixed``: validated ``{model position: entry}``, ``None`` = none)."""
    parts: List[np.ndarray] = [np.zeros(len(sizes), dtype=float)]
    for m, size in enumerate(sizes):
        if fixed and m in fixed:
            continue
        if int(size) > 1:
            b = np.zeros(int(size), dtype=float)
            b[0] = math.log((_INIT_Q0 / (1.0 - _INIT_Q0)) * (int(size) - 1))
            parts.append(b)
    return np.concatenate(parts)


def _theta_from_fit(init: HyperStackFit, sizes: Sequence[int],
                    fixed: Optional[Mapping[int, int]] = None) -> Optional[np.ndarray]:
    """Logits of a previous fit (warm start); ``None`` when it cannot seed this problem.

    A prior (``lam = inf``) is a degenerate point (log of zero probabilities), so it is never
    used as a start - the caller falls back to the default initialisation. ver4-51: a model whose
    gate is fixed (``fixed``) has no logits, so its gate in ``init`` is skipped."""
    if not math.isfinite(init.lam):
        return None
    if len(init.q) != len(sizes) or init.w_model.size != len(sizes):
        return None
    parts: List[np.ndarray] = [np.log(np.clip(init.w_model, _LOG_FLOOR, None))]
    for m, (size, gate) in enumerate(zip(sizes, init.q)):
        if fixed and m in fixed:
            continue
        if int(size) > 1:
            if np.asarray(gate).size != int(size):
                return None
            parts.append(np.log(np.clip(np.asarray(gate, dtype=float), _LOG_FLOOR, None)))
    return np.concatenate(parts)


def _check_fixed_gates(fixed_gates: Optional[Mapping[int, int]],
                       sizes: Sequence[int]) -> Dict[int, int]:
    """Validate ``fixed_gates`` (``{model position: entry}``) against ``sizes``; ``{}`` for None.

    Raises ValueError for a position outside the library, an entry outside its gate, or a model
    with a single entry (it has no gate, so there is nothing to fix)."""
    if not fixed_gates:
        return {}
    slices = group_slices(sizes)
    out: Dict[int, int] = {}
    for m_raw, e_raw in dict(fixed_gates).items():
        m, e = int(m_raw), int(e_raw)
        if not (0 <= m < len(slices)):
            raise ValueError(f'[HYPER_STACK] fixed_gates: model position {m} outside '
                             f'0..{len(slices) - 1}')
        size_m = slices[m].stop - slices[m].start
        if size_m < 2:
            raise ValueError(f'[HYPER_STACK] fixed_gates: model position {m} has a single entry '
                             f'(no gate to fix)')
        if not (0 <= e < size_m):
            raise ValueError(f'[HYPER_STACK] fixed_gates: entry {e} outside 0..{size_m - 1} for '
                             f'model position {m}')
        out[m] = e
    return out


def fit_hyper_stack(P: np.ndarray, y: np.ndarray, sizes: Sequence[int], *, lam: float,
                    init: Optional[HyperStackFit] = None, max_iter: int = 200,
                    tol: float = 1e-10, objective: str = 'ensemble_score',
                    fixed_gates: Optional[Mapping[int, int]] = None) -> HyperStackFit:
    """Fit ``a`` and ``b`` by L-BFGS on ``-score + lam * Omega`` (see module docstring), where
    ``score`` is ``objective``: ``'ensemble_score'`` (ver4-48/49, ``pearson - mse/var``) or
    ``'pearson'`` (ver4-50, scale-invariant ranking objective).

    Args:
      P: (n, C) library columns, models in ``sizes`` order, entry 0 of each model first.
      y: (n,) targets (the same rows as ``P``).
      sizes: L_m per model; ``sum(sizes) == C``.
      lam: regularisation strength, ``>= 0``; ``inf`` returns `prior_fit` (not optimised).
      init: a previous (finite-lam) fit to warm-start from; default start otherwise.
      max_iter: L-BFGS iteration cap. tol: L-BFGS ``tolerance_change``.
      objective: which score the network maximises (see above); default = the ver4-49 score.
      fixed_gates: ver4-51, ``{model position: entry}``. The gate of such a model is the constant
        one-hot ``delta_entry`` (no parameters: the theta layout and the warm start skip it); its
        gate term ``-log delta_entry[0]`` (0 for entry 0) is a constant that stays in the mean of
        the gate terms, so Omega is on the same scale as the free fit. ``None`` (default) leaves
        the code path untouched, bit for bit. ``lam = inf`` returns the prior (entry 0 everywhere),
        which is only consistent with fixed entries of 0.

    Returns the fit. ``converged`` is False when the iteration cap (or the evaluation cap) ended
    the run - the iterate is kept, never discarded (blueprint R2 failure modes).

    Raises:
      ValueError: shape mismatch, NaN/inf in ``P`` or ``y``, ``sum(sizes) != C``, lam < 0,
        unknown ``objective``, an invalid ``fixed_gates``.
      HyperStackFitError: the loss or the fitted weights became non-finite.
    """
    P_np, y_np, slices = _check_inputs(P, y, sizes)
    score_t = _TORCH_SCORES[_check_objective(objective)]
    fixed = _check_fixed_gates(fixed_gates, sizes)
    lam = float(lam)
    if lam < 0.0 or math.isnan(lam):
        raise ValueError(f'[HYPER_STACK] lam must be >= 0 (or inf), got {lam}')
    if not math.isfinite(lam):
        return prior_fit(sizes)
    k = len(slices)
    if int(P_np.shape[1]) == 1:
        # One column: nothing to train (K=1, L=1). The weight is 1 by construction.
        return HyperStackFit(w=np.ones(1, dtype=float), w_model=np.ones(1, dtype=float),
                             q=[np.ones(1, dtype=float)], lam=lam, objective=float('nan'),
                             n_iter=0, converged=True)

    P_t = torch.from_numpy(P_np)
    y_t = torch.from_numpy(y_np)
    start = _theta_from_fit(init, sizes, fixed) if init is not None else None
    if start is None:
        start = _default_theta(sizes, fixed)
    theta = torch.tensor(start, dtype=torch.float64, requires_grad=True)
    # ver4-51: constant gate (and constant gate cost) of every fixed model
    fixed_gate_t: Dict[int, "torch.Tensor"] = {}
    fixed_cost_t: Dict[int, "torch.Tensor"] = {}
    for m_fixed, e_fixed in fixed.items():
        onehot = np.zeros(slices[m_fixed].stop - slices[m_fixed].start, dtype=np.float64)
        onehot[e_fixed] = 1.0
        fixed_gate_t[m_fixed] = torch.from_numpy(onehot)
        fixed_cost_t[m_fixed] = torch.tensor(-math.log(max(float(onehot[0]), _LOG_FLOOR)),
                                             dtype=torch.float64)
    gate_slices: List[Optional[slice]] = []
    offset = k
    for m_pos, sl in enumerate(slices):
        size = sl.stop - sl.start
        if size > 1 and m_pos not in fixed:
            gate_slices.append(slice(offset, offset + size))
            offset += size
        else:
            gate_slices.append(None)
    if offset != int(theta.numel()):
        raise ValueError(f'[HYPER_STACK] internal parameter layout mismatch: {offset} vs '
                         f'{int(theta.numel())}')

    def _column_weights(th: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor",
                                                    List["torch.Tensor"], "torch.Tensor"]:
        log_w = torch.log_softmax(th[:k], dim=0)
        w_model = torch.exp(log_w)
        gates: List["torch.Tensor"] = []
        pieces: List["torch.Tensor"] = []
        gate_cost: List["torch.Tensor"] = []
        for m, gs in enumerate(gate_slices):
            if m in fixed_gate_t:
                gate = fixed_gate_t[m].to(th.dtype)
                gate_cost.append(fixed_cost_t[m].to(th.dtype))
            elif gs is None:
                gate = torch.ones(1, dtype=th.dtype)
            else:
                log_q = torch.log_softmax(th[gs], dim=0)
                gate = torch.exp(log_q)
                gate_cost.append(-log_q[0])
            gates.append(gate)
            pieces.append(w_model[m] * gate)
        kl = -math.log(k) - log_w.mean()
        omega = kl + (torch.stack(gate_cost).mean() if gate_cost else torch.zeros((),
                                                                                  dtype=th.dtype))
        return torch.cat(pieces), w_model, gates, omega

    def _loss(th: "torch.Tensor") -> "torch.Tensor":
        w_col, _w_model, _gates, omega = _column_weights(th)
        score = score_t(P_t @ w_col, y_t)
        return -score + lam * omega

    optimiser = torch.optim.LBFGS([theta], lr=1.0, max_iter=int(max_iter),
                                  max_eval=int(max_iter) * 5, tolerance_change=float(tol),
                                  history_size=20, line_search_fn='strong_wolfe')

    def _closure() -> "torch.Tensor":
        optimiser.zero_grad()
        loss = _loss(theta)
        loss.backward()
        return loss

    start_loss = float(_loss(theta.detach()))
    if not math.isfinite(start_loss):
        raise HyperStackFitError(
            f'[HYPER_STACK] the starting loss is not finite (lam={lam:g}) - a library column or '
            f'the target is degenerate')
    try:
        optimiser.step(_closure)
    except (RuntimeError, FloatingPointError) as exc:
        raise HyperStackFitError(
            f'[HYPER_STACK] L-BFGS failed at lam={lam:g} ({type(exc).__name__}: {exc})') from exc

    with torch.no_grad():
        w_col, w_model, gates, _omega = _column_weights(theta)
        final_loss = float(_loss(theta))
    w_np = w_col.detach().numpy().astype(float)
    if not (math.isfinite(final_loss) and np.all(np.isfinite(w_np))):
        raise HyperStackFitError(
            f'[HYPER_STACK] the optimiser ended at a non-finite loss/weights (lam={lam:g})')
    state = optimiser.state[optimiser._params[0]]
    n_iter = int(state.get('n_iter', 0))
    func_evals = int(state.get('func_evals', 0))
    converged = bool(n_iter < int(max_iter) and func_evals < int(max_iter) * 5)
    return HyperStackFit(w=w_np, w_model=w_model.detach().numpy().astype(float),
                         q=[g.detach().numpy().astype(float) for g in gates], lam=lam,
                         objective=final_loss, n_iter=n_iter, converged=converged)


def _check_inputs(P, y, sizes) -> Tuple[np.ndarray, np.ndarray, List[slice]]:
    """Validate and normalise the arguments shared by `fit_hyper_stack` and `lambda_path`."""
    slices = group_slices(sizes)
    # copies: `torch.from_numpy` needs a writable array (a read-only view would warn)
    P_np = np.array(P, dtype=np.float64, order='C', copy=True)
    y_np = np.array(y, dtype=np.float64, order='C', copy=True).ravel()
    if P_np.ndim != 2:
        raise ValueError(f'[HYPER_STACK] P must be 2-D (rows x columns), got shape {P_np.shape}')
    n, c = P_np.shape
    if y_np.size != n:
        raise ValueError(f'[HYPER_STACK] row mismatch: P has {n} rows, y has {y_np.size}')
    if slices[-1].stop != c:
        raise ValueError(f'[HYPER_STACK] sum(sizes)={slices[-1].stop} but P has {c} columns')
    if not np.all(np.isfinite(P_np)):
        bad = int(np.count_nonzero(~np.isfinite(P_np)))
        raise ValueError(f'[HYPER_STACK] P holds {bad} non-finite value(s)')
    if not np.all(np.isfinite(y_np)):
        raise ValueError('[HYPER_STACK] y holds non-finite value(s)')
    return P_np, y_np, slices


def _two_start_fit(P: np.ndarray, y: np.ndarray, sizes: Sequence[int], lam: float,
                   prev: Optional[HyperStackFit], *, objective: str, max_iter: int, tol: float,
                   fixed_gates: Optional[Mapping[int, int]] = None) -> HyperStackFit:
    """Fit one network at ``lam`` from two starts and keep the better one (ver4-51: extracted
    unchanged from ``lambda_path._fit`` so `refit_swap_test` fits exactly the same way).

    Two starts per lambda (the loss is not convex): the warm start from ``prev``, the previous
    (more regularised, or here: the free) fit, and the default start. A softmax gate that
    saturated at a large lambda has a vanishing gradient (d q_j is proportional to q_j), so the
    warm chain alone can stay at q = delta_0 where a gate with real signal would have left it; the
    lower objective wins and a tie keeps the warm start. Deterministic. With ``prev=None`` only
    the default start is fitted. ``lam = inf`` returns the prior. ``fixed_gates``: see
    `fit_hyper_stack`.

    Raises ValueError / HyperStackFitError exactly as `fit_hyper_stack` does for the warm start;
    a failure of the second (default) start alone keeps the warm fit."""
    if not math.isfinite(lam):
        return prior_fit(sizes)
    warm = fit_hyper_stack(P, y, sizes, lam=lam, init=prev, max_iter=max_iter, tol=tol,
                           objective=objective, fixed_gates=fixed_gates)
    if prev is None:
        return warm
    try:
        cold = fit_hyper_stack(P, y, sizes, lam=lam, init=None, max_iter=max_iter, tol=tol,
                               objective=objective, fixed_gates=fixed_gates)
    except HyperStackFitError:
        return warm
    return cold if cold.objective < warm.objective - 1e-12 else warm


def stack_loss(P: np.ndarray, y: np.ndarray, sizes: Sequence[int], fit: HyperStackFit, *,
               lam: float, objective: str = 'ensemble_score') -> float:
    """The training loss ``-score(P w, y) + lam * Omega`` of an already fitted network.

    Why: a diagnostic and test helper (ver4-51, AC1.3: the clamped refit never ends above the
    fitted network with the same gate overwritten). ``Omega`` is `stack_penalty` of the fit's own
    ``W`` and gates, which for a one-hot gate adds the constant ``-log delta[0]`` exactly as
    `fit_hyper_stack` does for a fixed gate. Returns ``nan`` for a degenerate score."""
    P_np, y_np, _slices = _check_inputs(P, y, sizes)
    score_t = _TORCH_SCORES[_check_objective(objective)]
    pred = torch.from_numpy(P_np @ np.asarray(fit.w, dtype=np.float64))
    score = float(score_t(pred, torch.from_numpy(y_np)))
    return float(-score + float(lam) * stack_penalty(fit.w_model, fit.q))


# --------------------------------------------------------------------------- #
# The lambda path
# --------------------------------------------------------------------------- #

def lambda_path(P: np.ndarray, y: np.ndarray, sizes: Sequence[int], *,
                lambdas: Sequence[float], include_prior: bool, folds: int, repeats: int,
                seed: int, rule: str,
                score_fn: Callable[[np.ndarray, np.ndarray], Optional[float]],
                max_iter: int = 200, tol: float = 1e-10, nb_correction: bool = False,
                objective: str = 'ensemble_score', keep_fold_fits: bool = False
                ) -> Tuple[HyperStackFit, dict]:
    """Choose lambda by repeated K-fold cross-validation and return the network refitted at it.

    The candidates are the finite ``lambdas`` plus (``include_prior``) ``inf`` - the closed-form
    naive ensemble. They are visited in DESCENDING order and every fit is warm-started from the
    previous (more regularised) one, in the CV folds and in the final all-rows chain alike, so the
    chosen network is the one the CV scored. Every fit after the first is ALSO fitted from the
    default start and the lower-objective of the two is kept: a warm start from a strongly
    regularised solution can sit in a saturated softmax (gate stuck at entry 0, ~zero gradient),
    which left the whole path at the prior in the ver4-48 tests (the cost is about twice the path's
    fits, all on small matrices - no model fits).

    For repeat r the rows are partitioned by ``_fold_indices(n, folds, default_rng(seed + r))``;
    each candidate is fitted on the other folds and its held-out combination scored with
    ``score_fn(pred, y_held)`` (the caller passes `joint_tuning._ensemble_score_of`; ``None`` is a
    failure). A candidate with ANY failed (repeat, fold) fit is not eligible.

    ``rule='min'``: the highest mean held-out score (ties -> the larger lambda).
    ``rule='one_se'``: the largest lambda whose paired gap to the best, ``mean(best - score)``, is
    at most one standard error of that gap (``std(ddof=1) / sqrt(#fits)``) - the same paired
    definition as `ensemble_regularization.cv_shrunk_simplex_weights`, applied to fold scores.

    ``nb_correction`` (ver4-49): the ``repeats * folds`` scores are NOT independent (every repeat
    re-uses the same rows, and the training sets of different folds overlap), so
    ``std / sqrt(#fits)`` understates the standard error and let the one-SE rule leave the
    untuned ensemble for a tiny lambda on noise. With ``nb_correction=True`` the variance of the
    mean gap is ``var(gap) * (1/folds + 1/(folds - 1))`` - the Nadeau & Bengio (2003) correction,
    ``n_test / n_train = 1 / (folds - 1)`` - which is the default of the neural stack.
    ``False`` reproduces ver4-48 exactly.

    ``objective`` (ver4-50): the score the network maximises in every fit (``'ensemble_score'`` =
    ver4-49, ``'pearson'``). The caller must pass the numpy twin of the same objective as
    ``score_fn`` (parity is a test, not a runtime check). ``keep_fold_fits`` (ver4-50): keep the
    fitted network of every (repeat, fold) item for the CHOSEN lambda in ``info['fold_fits']`` as
    ``(held_row_indices, fit)`` pairs (``[]`` when the chosen lambda is infinity) - the input of
    `swap_test`. Only references are kept, no extra fit is made, and the default leaves every
    result bitwise identical.

    Args:
      P, y, sizes: as `fit_hyper_stack` (rows = the CV rows).
      lambdas: finite strengths ``>= 0``; may be empty when ``include_prior``.
      include_prior: add ``inf`` (the untuned naive ensemble) to the candidates.
      folds, repeats: the CV loop (``folds >= 2``, ``repeats >= 1``).
      seed: base seed of the fold partitions (I9).
      rule: ``'min'`` | ``'one_se'``.
      score_fn: held-out scoring function.
      max_iter, tol: L-BFGS settings of every fit.

    Returns ``(fit, info)``. ``info`` has ``lambdas`` (descending, ``'inf'`` for the prior),
    ``path`` (list of ``(lambda, cv_mean, cv_se, n_failed)`` rows; mean/se are None for an
    ineligible candidate), ``chosen`` (the lambda, ``float('inf')`` for the prior), ``rule``,
    ``n_finite_ok`` (finite lambdas that were eligible), ``full_path`` (per candidate fitted on
    all rows: ``(lambda, objective, omega, n_iter, converged)``), ``folds``, ``repeats``,
    ``nb_correction``, ``objective`` and - with ``keep_fold_fits`` - ``fold_fits``.

    Raises:
      ValueError: bad shapes, NaN in P, ``sum(sizes) != C``, unknown rule or objective,
        ``folds < 2``, fewer than ``2 * folds`` rows, a negative / non-finite lambda, or no
        candidate at all.
      HyperStackFitError: every candidate failed (no eligible lambda); the caller decides
        whether that is a pure-mode error or a fallback to the prior.
    """
    P_np, y_np, slices = _check_inputs(P, y, sizes)
    n = int(P_np.shape[0])
    _check_objective(objective)
    if rule not in _RULES:
        raise ValueError(f"[HYPER_STACK] unknown rule {rule!r} (expected 'min' or 'one_se')")
    folds = int(folds)
    repeats = int(repeats)
    if folds < 2 or repeats < 1:
        raise ValueError(f'[HYPER_STACK] need folds >= 2 and repeats >= 1, got {folds}/{repeats}')
    if n < 2 * folds:
        raise ValueError(f'[HYPER_STACK] too few rows to cross-validate: {n} < 2 x {folds} folds')
    finite = sorted({float(v) for v in lambdas}, reverse=True)
    for v in finite:
        if not math.isfinite(v) or v < 0.0:
            raise ValueError(f'[HYPER_STACK] lambdas must be finite and >= 0, got {v}')
    order: List[float] = ([float('inf')] if include_prior else []) + finite
    if not order:
        raise ValueError('[HYPER_STACK] no lambda candidates (empty list and include_prior=False)')
    n_cand = len(order)

    prior = prior_fit(sizes)

    def _fit(rows: np.ndarray, lam: float, prev: Optional[HyperStackFit]) -> HyperStackFit:
        # Two starts per lambda (warm chain + default start, the lower objective wins): the logic
        # lives in `_two_start_fit` since ver4-51 so the swap test fits identically.
        if not math.isfinite(lam):
            return prior
        return _two_start_fit(P_np[rows], y_np[rows], sizes, lam, prev, objective=objective,
                              max_iter=max_iter, tol=tol)

    scores = np.full((n_cand, repeats * folds), np.nan, dtype=float)
    failed = np.zeros(n_cand, dtype=int)
    # ver4-50: per candidate, the (held rows, fitted network) of every item (references only)
    fold_store: Dict[int, List[Tuple[np.ndarray, HyperStackFit]]] = {
        ci: [] for ci in range(n_cand)}
    item = 0
    for rep in range(repeats):
        rng = np.random.default_rng(int(seed) + rep)
        for held in _fold_indices(n, folds, rng):
            keep = np.ones(n, dtype=bool)
            keep[held] = False
            rows = np.flatnonzero(keep)
            prev: Optional[HyperStackFit] = None
            for ci, lam in enumerate(order):
                value: Optional[float] = None
                try:
                    fit = _fit(rows, lam, prev)
                    prev = fit if math.isfinite(lam) else None
                    value = score_fn(P_np[held] @ fit.w, y_np[held])
                    if keep_fold_fits:
                        fold_store[ci].append((np.asarray(held), fit))
                except HyperStackFitError:
                    prev = None
                if value is None or not math.isfinite(float(value)):
                    failed[ci] += 1
                else:
                    scores[ci, item] = float(value)
            item += 1
    n_items = item

    eligible = [ci for ci in range(n_cand) if failed[ci] == 0]
    cv_mean = np.full(n_cand, np.nan, dtype=float)
    for ci in eligible:
        cv_mean[ci] = float(np.mean(scores[ci, :n_items]))

    # Full-row chain (descending lambda, warm-started) - the network that is returned and the
    # diagnostic record of Omega along the path.
    all_rows = np.arange(n)
    chain: Dict[int, HyperStackFit] = {}
    full_path: List[tuple] = []
    prev_full: Optional[HyperStackFit] = None
    for ci, lam in enumerate(order):
        try:
            fit = _fit(all_rows, lam, prev_full)
        except HyperStackFitError:
            prev_full = None
            full_path.append((lam, None, None, 0, False))
            continue
        chain[ci] = fit
        prev_full = fit if math.isfinite(lam) else None
        omega = stack_penalty(fit.w_model, fit.q)
        full_path.append((lam, None if math.isnan(fit.objective) else float(fit.objective),
                          float(omega), int(fit.n_iter), bool(fit.converged)))

    usable = [ci for ci in eligible if ci in chain]
    if not usable:
        raise HyperStackFitError(
            f'[HYPER_STACK] every lambda candidate failed ({n_cand} tried, {repeats}x{folds} '
            f'CV fits each) - no network can be chosen')
    best_ci = max(usable, key=lambda c: (cv_mean[c], -c))
    chosen_ci = best_ci
    se_by_ci: Dict[int, float] = {}
    for ci in usable:
        gap = scores[best_ci, :n_items] - scores[ci, :n_items]
        if n_items <= 1:
            se_by_ci[ci] = 0.0
        elif nb_correction:
            se_by_ci[ci] = float(math.sqrt(gap.var(ddof=1) * (1.0 / folds + 1.0 / (folds - 1))))
        else:
            se_by_ci[ci] = float(gap.std(ddof=1) / math.sqrt(n_items))
    if rule == 'one_se':
        for ci in usable:                                  # descending lambda = ascending index
            gap = scores[best_ci, :n_items] - scores[ci, :n_items]
            if float(gap.mean()) <= se_by_ci[ci]:
                chosen_ci = ci
                break

    path = []
    for ci, lam in enumerate(order):
        if ci in eligible:
            path.append((lam, float(cv_mean[ci]), float(se_by_ci.get(ci, 0.0)), int(failed[ci])))
        else:
            path.append((lam, None, None, int(failed[ci])))
    n_finite_ok = sum(1 for ci in usable if math.isfinite(order[ci]))
    info = {'lambdas': [('inf' if not math.isfinite(v) else v) for v in order], 'path': path,
            'chosen': float(order[chosen_ci]), 'rule': rule, 'n_finite_ok': int(n_finite_ok),
            'full_path': full_path, 'folds': folds, 'repeats': repeats,
            'nb_correction': bool(nb_correction), 'objective': objective}
    if keep_fold_fits:
        # the chosen candidate is eligible, so every one of its items holds a fit; lambda = inf
        # has no fitted network to test (its gates are delta_0 by construction)
        info['fold_fits'] = ([] if not math.isfinite(order[chosen_ci])
                             else list(fold_store[chosen_ci]))
    return chain[chosen_ci], info


def shrink_fit(fit: HyperStackFit, sizes: Sequence[int], shrink: float) -> HyperStackFit:
    """ver4-49: pull a fitted network's weights towards the untuned naive ensemble.

    ``W' = (1 - s) * W + s / K`` and ``w' = (1 - s) * w + s * w_prior`` (``w_prior`` = `prior_fit`:
    ``1/K`` on every model's entry 0). The gates ``q`` are untouched, so the most probable entry
    of every model (the ``argmax`` deployment's primary) is unchanged. Why: the weights are fitted
    on out-of-fold columns of ~(k-1)/k of the rows, scored on a few dozen rows, and chosen from a
    library of correlated columns, so the fitted weights are an optimistic, over-concentrated
    estimate (a classic winner's curse / forecast-combination effect); averaging them with the
    equal-weight ensemble removes most of the concentration risk at a small price in the mean.
    ``shrink = 0`` returns ``fit`` itself; ``shrink = 1`` is the prior's weights. Non-negativity and
    the unit sum are preserved (a convex combination of two points of the simplex).

    Raises ValueError for ``shrink`` outside [0, 1]."""
    s = float(shrink)
    if not (0.0 <= s <= 1.0) or not math.isfinite(s):
        raise ValueError(f'[HYPER_STACK] shrink must be in [0, 1], got {shrink!r}')
    if s == 0.0 or not math.isfinite(fit.lam):
        return fit                                    # the prior is already the target
    prior = prior_fit(sizes)
    w = (1.0 - s) * np.asarray(fit.w, dtype=float) + s * prior.w
    w_model = (1.0 - s) * np.asarray(fit.w_model, dtype=float) + s * prior.w_model
    return HyperStackFit(w=w / w.sum(), w_model=w_model / w_model.sum(), q=fit.q, lam=fit.lam,
                         objective=fit.objective, n_iter=fit.n_iter, converged=fit.converged)


# --------------------------------------------------------------------------- #
# ver4-50, R3: the paired swap test
# --------------------------------------------------------------------------- #

def force_gate(fit: HyperStackFit, sizes: Sequence[int], m: int, entry: int) -> HyperStackFit:
    """Return a copy of ``fit`` whose gate of model ``m`` is the one-hot ``delta_entry``.

    ``W`` (``w_model``) and every other gate are unchanged, and the column weights are recomputed
    as ``W_j * q_j`` for every model, so the result is again a point of the simplex (non-negative,
    sums to 1). Pure: ``fit`` is not modified. Why: the swap test needs "the same network, but
    model ``m`` uses entry ``entry`` only" without a refit.

    Args:
      fit: a fitted network (its ``q`` has one gate per model in ``sizes`` order).
      sizes: L_m per model. m: model position in ``sizes``.
      entry: gate position, ``0 <= entry < L_m``.

    Raises ValueError for a position out of range or a ``fit`` that does not match ``sizes``."""
    slices = group_slices(sizes)
    if len(fit.q) != len(slices) or int(np.asarray(fit.w_model).size) != len(slices):
        raise ValueError(f'[HYPER_STACK] force_gate: the fit has {len(fit.q)} gate(s) but sizes '
                         f'describes {len(slices)} model(s)')
    if not (0 <= int(m) < len(slices)):
        raise ValueError(f'[HYPER_STACK] force_gate: model position {m} outside '
                         f'0..{len(slices) - 1}')
    size_m = slices[int(m)].stop - slices[int(m)].start
    if not (0 <= int(entry) < size_m):
        raise ValueError(f'[HYPER_STACK] force_gate: entry {entry} outside 0..{size_m - 1} for '
                         f'model position {m}')
    q_new: List[np.ndarray] = [np.asarray(g, dtype=float).copy() for g in fit.q]
    onehot = np.zeros(size_m, dtype=float)
    onehot[int(entry)] = 1.0
    q_new[int(m)] = onehot
    w_model = np.asarray(fit.w_model, dtype=float)
    w_new = np.concatenate([w_model[j] * q_new[j] for j in range(len(slices))])
    return HyperStackFit(w=w_new, w_model=w_model.copy(), q=q_new, lam=fit.lam,
                         objective=fit.objective, n_iter=fit.n_iter, converged=fit.converged)


def swap_test(P: np.ndarray, y: np.ndarray, sizes: Sequence[int],
              fold_fits: Sequence[Tuple[np.ndarray, HyperStackFit]], *,
              final_entries: Sequence[int],
              score_fn: Callable[[np.ndarray, np.ndarray], Optional[float]],
              folds: int) -> List[dict]:
    """ver4-50, R3: admit a hyperparameter switch only when its CV gain beats its own noise.

    For every model ``m`` with ``L_m > 1`` whose all-rows gate picks ``l* = final_entries[m] != 0``
    and for every (repeat, fold) item ``(held, fit_i)`` of ``fold_fits``, the held-out rows are
    scored twice with item ``i``'s own network: gate ``m`` forced to ``delta_{l*}`` and forced to
    ``delta_0`` (``force_gate``; ``W`` and the other gates unchanged). The paired gaps ``g_i`` are
    reduced to ``mean(g)`` and the Nadeau-Bengio standard error
    ``sqrt(var(g, ddof=1) * (1/folds + 1/(folds - 1)))`` (the formula `lambda_path` uses with
    ``nb_correction=True``); the swap is ``admitted`` only when ``mean(g) > se``. No model is
    fitted and no random number is drawn (I9). The items share the rows of the lambda choice, so
    the test is conservative relative to the legacy rule (which tests nothing) but not independent.

    Args:
      P, y, sizes: as `lambda_path` (rows = the CV rows; ``held`` indexes them).
      fold_fits: ``info['fold_fits']`` of `lambda_path(..., keep_fold_fits=True)`.
      final_entries: per model position, the gate position ``l*`` of the all-rows fit.
      score_fn: the numpy score of the network's objective. folds: the CV fold count.

    Returns one dict per tested model, in model order: ``{'m', 'entry', 'gap', 'se', 'n_items',
    'admitted', 'reason'}`` (``reason`` is ``''`` for a normal verdict, ``'degenerate'`` when any
    item's score was ``None``/non-finite, ``'no_fold_fits'`` when ``fold_fits`` is empty - both
    reject the swap, the conservative choice).

    Raises ValueError for a ``final_entries`` of the wrong length or an entry outside its gate."""
    P_np, y_np, slices = _check_inputs(P, y, sizes)
    if len(final_entries) != len(slices):
        raise ValueError(f'[HYPER_STACK] swap_test: final_entries has {len(final_entries)} '
                         f'item(s), the library has {len(slices)} model(s)')
    folds = int(folds)
    if folds < 2:
        raise ValueError(f'[HYPER_STACK] swap_test: folds must be >= 2, got {folds}')
    nb_factor = 1.0 / folds + 1.0 / (folds - 1)
    out: List[dict] = []
    for m, sl in enumerate(slices):
        size_m = sl.stop - sl.start
        l_star = int(final_entries[m])
        if not (0 <= l_star < size_m):
            raise ValueError(f'[HYPER_STACK] swap_test: entry {l_star} outside 0..{size_m - 1} '
                             f'for model position {m}')
        if size_m < 2 or l_star == 0:
            continue
        row = {'m': m, 'entry': l_star, 'gap': float('nan'), 'se': float('nan'),
               'n_items': len(fold_fits), 'admitted': False, 'reason': ''}
        if len(fold_fits) == 0:
            row['reason'] = 'no_fold_fits'
            out.append(row)
            continue
        gaps: List[float] = []
        for held, fit_i in fold_fits:
            held_idx = np.asarray(held)
            swapped = score_fn(P_np[held_idx] @ force_gate(fit_i, sizes, m, l_star).w,
                               y_np[held_idx])
            base = score_fn(P_np[held_idx] @ force_gate(fit_i, sizes, m, 0).w, y_np[held_idx])
            if (swapped is None or base is None or not math.isfinite(float(swapped))
                    or not math.isfinite(float(base))):
                gaps = []
                break
            gaps.append(float(swapped) - float(base))
        if not gaps:
            row['reason'] = 'degenerate'
            out.append(row)
            continue
        g = np.asarray(gaps, dtype=float)
        se = 0.0 if g.size <= 1 else float(math.sqrt(float(g.var(ddof=1)) * nb_factor))
        row['gap'] = float(g.mean())
        row['se'] = se
        row['admitted'] = bool(row['gap'] > se)
        out.append(row)
    return out


# --------------------------------------------------------------------------- #
# ver4-51, R1 / R2: one deployment procedure, a fair swap test and the nested gate
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class StackSpec:
    """Every setting of the neural stack's post-processing, so that the all-rows fit and every
    outer-CV item run the IDENTICAL procedure (blueprint R2: one procedure, one deploy rule).

    Attributes:
      lambdas: finite regularisation strengths; ``include_prior``: add ``inf``.
      folds, repeats, seed: the CV loop that chooses lambda (and the swap tests' fold count).
      rule: ``'min'`` | ``'one_se'``; ``objective``: ``'ensemble_score'`` | ``'pearson'``.
      nb_correction: Nadeau-Bengio standard error in the lambda rule.
      swap_rule: ``'gate'`` (no test) | ``'paired_se'`` (ver4-50) | ``'paired_refit'`` (ver4-51).
      shrink: weight shrink towards the equal-weight ensemble, in [0, 1].
      deploy: ``'mixture'`` | ``'argmax'``; ``member_min_weight`` / ``members_max``: the mixture's
        member rule (the engine passes ``_JOINT_LIBRARY_MEMBER_MIN_WEIGHT`` / library_members_max).
      max_iter, tol: L-BFGS settings.
      swap_seed_offset: added to ``seed`` for the partition of ``paired_refit`` (the engine passes
        ``_JOINT_NS_SWAP_SEED_OFFSET``; the default is the same number).
    """
    lambdas: Tuple[float, ...]
    include_prior: bool
    folds: int
    repeats: int
    seed: int
    rule: str
    objective: str
    nb_correction: bool
    swap_rule: str
    shrink: float
    deploy: str
    member_min_weight: float
    members_max: int
    max_iter: int
    tol: float
    swap_seed_offset: int = 7919


def _check_spec(spec: StackSpec) -> None:
    """Raise ValueError naming the offending field when ``spec`` is not usable."""
    if spec.rule not in _RULES:
        raise ValueError(f"[HYPER_STACK] StackSpec.rule {spec.rule!r} not in {list(_RULES)!r}")
    _check_objective(spec.objective)
    if spec.swap_rule not in _SWAP_RULES:
        raise ValueError(f"[HYPER_STACK] StackSpec.swap_rule {spec.swap_rule!r} not in "
                         f"{list(_SWAP_RULES)!r}")
    if spec.deploy not in _DEPLOYS:
        raise ValueError(f"[HYPER_STACK] StackSpec.deploy {spec.deploy!r} not in "
                         f"{list(_DEPLOYS)!r}")
    shrink = float(spec.shrink)
    if not (0.0 <= shrink <= 1.0) or not math.isfinite(shrink):
        raise ValueError(f"[HYPER_STACK] StackSpec.shrink must be in [0, 1], got {spec.shrink!r}")
    if int(spec.members_max) < 1:
        raise ValueError(f"[HYPER_STACK] StackSpec.members_max must be >= 1, got "
                         f"{spec.members_max!r}")


def deployed_column_weights(fit: HyperStackFit, sizes: Sequence[int], deploy: str, *,
                            member_min_weight: float, members_max: int
                            ) -> Tuple[np.ndarray, Optional[List[Tuple[int, int, float]]]]:
    """The column-weight vector that is actually DEPLOYED for ``fit`` (single authority, I3).

    Why: before ver4-51 the deployment rule (stage N3 of `joint_tuning._neural_stack_search`)
    lived inside the engine, so nothing else could score "the vector that is deployed" - the
    lambda path scores the gate MIXTURE ``P @ fit.w``, deployment uses the argmax primaries (or the
    heavy members). This function is that rule, extracted without change.

    ``deploy='argmax'``: ``W_m`` on the primary column of every model (primary = the most probable
    entry, ties -> the lowest position), 0 elsewhere. ``deploy='mixture'``: the columns whose
    ``fit.w`` is at least ``member_min_weight``, heaviest first (ties -> column order), at most
    ``members_max`` of them, renormalised to sum 1; when no column qualifies the argmax vector is
    returned (and ``members`` is ``None``).

    Args:
      fit: a fitted network (one gate per model of ``sizes``). sizes: L_m per model.
      deploy: ``'mixture'`` | ``'argmax'``.
      member_min_weight: smallest column weight kept as a member (``mixture`` only).
      members_max: cap on the number of members (``mixture`` only).

    Returns ``(w_dep, members)``: ``w_dep`` is ``(C,)`` and non-negative; ``members`` is ``None``
    for ``argmax`` and for an empty mixture, else a list of ``(model position, entry position,
    renormalised weight)`` in the order of decreasing ``fit.w`` (the raw weight is
    ``fit.w[column]``).

    Raises ValueError for an unknown ``deploy`` or a fit that does not match ``sizes``."""
    if deploy not in _DEPLOYS:
        raise ValueError(f"[HYPER_STACK] deployed_column_weights: unknown deploy {deploy!r} "
                         f"(expected one of {list(_DEPLOYS)!r})")
    slices = group_slices(sizes)
    n_cols = slices[-1].stop
    w_model = np.asarray(fit.w_model, dtype=float)
    w_cols = np.asarray(fit.w, dtype=float)
    if len(fit.q) != len(slices) or w_model.size != len(slices) or w_cols.size != n_cols:
        raise ValueError(f'[HYPER_STACK] deployed_column_weights: the fit does not match sizes '
                         f'({len(fit.q)} gate(s), {w_model.size} model weight(s), {w_cols.size} '
                         f'column weight(s) vs {len(slices)} model(s), {n_cols} column(s))')
    w_arg = np.zeros(n_cols, dtype=float)
    for m, sl in enumerate(slices):
        primary = int(np.argmax(np.asarray(fit.q[m], dtype=float)))
        w_arg[sl.start + primary] = float(w_model[m])
    if deploy == 'argmax':
        return w_arg, None
    owner: List[Tuple[int, int]] = []
    for m, sl in enumerate(slices):
        owner.extend((m, j) for j in range(sl.stop - sl.start))
    order = sorted(range(n_cols), key=lambda c: (-float(w_cols[c]), c))
    chosen = [c for c in order if float(w_cols[c]) >= float(member_min_weight)][:int(members_max)]
    if not chosen:
        return w_arg, None
    kept_w = np.array([float(w_cols[c]) for c in chosen], dtype=float)
    kept_w = kept_w / float(kept_w.sum())
    w_mix = np.zeros(n_cols, dtype=float)
    members: List[Tuple[int, int, float]] = []
    for c, wk in zip(chosen, kept_w):
        w_mix[c] = float(wk)
        members.append((owner[c][0], owner[c][1], float(wk)))
    return w_mix, members


def _finite_score(value: Optional[float]) -> bool:
    """True for a usable score (not ``None``, finite)."""
    return value is not None and math.isfinite(float(value))


def refit_swap_test(P: np.ndarray, y: np.ndarray, sizes: Sequence[int], *, lam: float,
                    final_entries: Sequence[int], folds: int, repeats: int, seed: int,
                    score_fn: Callable[[np.ndarray, np.ndarray], Optional[float]],
                    objective: str, deploy: str, member_min_weight: float, members_max: int,
                    max_iter: int, tol: float) -> List[dict]:
    """ver4-51, R1: admit a hyperparameter switch only when it beats a FAIR counterfactual.

    Why: `swap_test` compares the fitted network with model ``m`` forced to ``l*`` against the
    SAME network with the gate forced to entry 0 - but ``W`` stays what it was fitted to be while
    the gate chose ``l*``, so the baseline is handicapped in proportion to ``W_m`` (the ver4-50
    SVR switch "gained" more than the whole network is worth). Here the baseline is the network
    REFITTED with gate ``m`` clamped to entry 0 (``fit_hyper_stack(..., fixed_gates={m: 0})``: ``W``
    and the other gates re-optimised), and the items are a FRESH repeated partition, not the rows
    that chose lambda.

    For every (repeat, fold) item (partition seed ``seed + repeat``): one free fit ``free_i`` at
    ``lam`` on the inner rows (shared by every tested model); per tested model ``m`` with
    ``l* = final_entries[m] != 0`` a clamped refit ``clamp_i`` (warm start ``free_i``, plus the
    default start, lower objective wins - so its training loss never exceeds that of ``free_i``
    with gate ``m`` overwritten). The held rows are scored with the DEPLOYED vector
    (`deployed_column_weights`) of ``free_i`` with gate ``m`` forced to ``l*`` and of ``clamp_i``;
    the paired gaps are reduced exactly as in `swap_test`: ``admitted`` iff
    ``mean(gap) > sqrt(var(gap, ddof=1) * (1/folds + 1/(folds - 1)))``. The legacy forced gap
    (``force_gate`` l* minus ``force_gate`` 0, raw column weights, same items) is reported as
    ``gap_forced``. No model is fitted (only stacker fits on the matrix ``P``) and no random number
    is drawn (I9). The shrink is not applied here: the test decides the gate, the shrink follows.

    Args:
      P, y, sizes: the CV matrix, targets, library geometry (rows = the CV rows).
      lam: the all-rows lambda* (finite; ``inf`` returns ``[]`` - nothing to test).
      final_entries: per model, the gate position of the all-rows fit.
      folds, repeats, seed: the partition (``folds >= 2``, ``repeats >= 1``, ``2*folds`` rows).
      score_fn: numpy score of the network's objective; ``objective``: its name (for the fits).
      deploy, member_min_weight, members_max: the deployment rule (`deployed_column_weights`).
      max_iter, tol: L-BFGS settings.

    Returns one dict per tested model in model order: ``{'m', 'entry', 'gap', 'gap_forced', 'se',
    'n_items', 'admitted', 'reason'}``; ``reason`` is ``''`` or ``'degenerate'`` (a fit failed or a
    score was ``None``/non-finite in any item: the swap is rejected, the conservative choice).

    Raises ValueError for bad shapes, ``folds < 2``, ``repeats < 1``, too few rows, a
    ``final_entries`` of the wrong length or outside its gate."""
    P_np, y_np, slices = _check_inputs(P, y, sizes)
    _check_objective(objective)
    if len(final_entries) != len(slices):
        raise ValueError(f'[HYPER_STACK] refit_swap_test: final_entries has {len(final_entries)} '
                         f'item(s), the library has {len(slices)} model(s)')
    folds, repeats = int(folds), int(repeats)
    if folds < 2 or repeats < 1:
        raise ValueError(f'[HYPER_STACK] refit_swap_test: need folds >= 2 and repeats >= 1, got '
                         f'{folds}/{repeats}')
    n = int(P_np.shape[0])
    if n < 2 * folds:
        raise ValueError(f'[HYPER_STACK] refit_swap_test: too few rows: {n} < 2 x {folds} folds')
    tested: List[int] = []
    for m, sl in enumerate(slices):
        size_m = sl.stop - sl.start
        l_star = int(final_entries[m])
        if not (0 <= l_star < size_m):
            raise ValueError(f'[HYPER_STACK] refit_swap_test: entry {l_star} outside '
                             f'0..{size_m - 1} for model position {m}')
        if size_m >= 2 and l_star != 0:
            tested.append(m)
    lam = float(lam)
    if not tested or not math.isfinite(lam):
        return []
    nb_factor = 1.0 / folds + 1.0 / (folds - 1)
    gaps: Dict[int, List[float]] = {m: [] for m in tested}
    gaps_forced: Dict[int, List[float]] = {m: [] for m in tested}
    degenerate: Dict[int, bool] = {m: False for m in tested}
    dep_kw = dict(member_min_weight=member_min_weight, members_max=members_max)
    for rep in range(repeats):
        rng = np.random.default_rng(int(seed) + rep)
        for held in _fold_indices(n, folds, rng):
            held_idx = np.asarray(held)
            keep = np.ones(n, dtype=bool)
            keep[held_idx] = False
            rows = np.flatnonzero(keep)
            try:
                free_i: Optional[HyperStackFit] = _two_start_fit(
                    P_np[rows], y_np[rows], sizes, lam, None, objective=objective,
                    max_iter=max_iter, tol=tol)
            except HyperStackFitError:
                free_i = None
            for m in tested:
                if degenerate[m]:
                    continue
                if free_i is None:
                    degenerate[m] = True
                    continue
                l_star = int(final_entries[m])
                try:
                    clamp_i = _two_start_fit(P_np[rows], y_np[rows], sizes, lam, free_i,
                                             objective=objective, max_iter=max_iter, tol=tol,
                                             fixed_gates={m: 0})
                except HyperStackFitError:
                    degenerate[m] = True
                    continue
                free_star = force_gate(free_i, sizes, m, l_star)
                w_star, _mem = deployed_column_weights(free_star, sizes, deploy, **dep_kw)
                w_base, _mem = deployed_column_weights(clamp_i, sizes, deploy, **dep_kw)
                y_held = y_np[held_idx]
                s_star = score_fn(P_np[held_idx] @ w_star, y_held)
                s_base = score_fn(P_np[held_idx] @ w_base, y_held)
                s_forced_star = score_fn(P_np[held_idx] @ free_star.w, y_held)
                s_forced_base = score_fn(P_np[held_idx] @ force_gate(free_i, sizes, m, 0).w,
                                         y_held)
                if not all(_finite_score(v) for v in (s_star, s_base, s_forced_star,
                                                      s_forced_base)):
                    degenerate[m] = True
                    continue
                gaps[m].append(float(s_star) - float(s_base))
                gaps_forced[m].append(float(s_forced_star) - float(s_forced_base))
    out: List[dict] = []
    for m in tested:
        row = {'m': m, 'entry': int(final_entries[m]), 'gap': float('nan'),
               'gap_forced': float('nan'), 'se': float('nan'), 'n_items': folds * repeats,
               'admitted': False, 'reason': ''}
        if degenerate[m] or len(gaps[m]) != folds * repeats:
            row['reason'] = 'degenerate'
            out.append(row)
            continue
        g = np.asarray(gaps[m], dtype=float)
        se = 0.0 if g.size <= 1 else float(math.sqrt(float(g.var(ddof=1)) * nb_factor))
        row['gap'] = float(g.mean())
        row['gap_forced'] = float(np.mean(np.asarray(gaps_forced[m], dtype=float)))
        row['se'] = se
        row['admitted'] = bool(row['gap'] > se)
        out.append(row)
    return out


def stack_procedure(P: np.ndarray, y: np.ndarray, sizes: Sequence[int], spec: StackSpec,
                    score_fn: Callable[[np.ndarray, np.ndarray], Optional[float]]
                    ) -> Tuple[HyperStackFit, np.ndarray, dict]:
    """The WHOLE neural-stack procedure on one set of rows: lambda path -> swap rule -> shrink ->
    deployed column weights (ver4-51, R2; stages N2, N2b and the shrink of
    `joint_tuning._neural_stack_search`, moved here unchanged so the all-rows fit and every outer
    cross-validation item of `nested_gate` run identical code).

    Steps: `lambda_path` (``keep_fold_fits`` only for ``paired_se``); with a finite lambda and a
    gate away from entry 0, the swap rule (``'gate'``: none; ``'paired_se'``: `swap_test`;
    ``'paired_refit'``: `refit_swap_test`), each rejected model's gate reset to delta_0
    (`force_gate`); the shrink towards the equal-weight ensemble (`shrink_fit`, finite lambda
    only); `deployed_column_weights`. When EVERY lambda fails and ``spec.include_prior`` the prior
    is returned (the engine's behaviour); without the prior the HyperStackFitError propagates.

    Returns ``(fit, w_dep, info)``: the network after swap and shrink, the deployed ``(C,)`` column
    weights and a dict: ``lambda_info`` (the `lambda_path` info, or the engine's one-candidate
    fallback), ``lambda_note`` (``None`` or ``'every lambda failed'``), ``swaps`` (the rows of the
    swap test, keyed by model POSITION in ``sizes``; ``[]`` for ``'gate'``), ``swap_skipped``
    (``'lambda_inf'`` when a swap rule had nothing to test because lambda = inf, else ``None``),
    ``w_model_raw`` (``W`` before the shrink), ``members`` (`deployed_column_weights`) and
    ``fit_lambda`` (the network at lambda*, before the swap rule and the shrink - for diagnostics).

    Raises ValueError (bad inputs / spec) and HyperStackFitError (every lambda failed, no prior)."""
    _check_spec(spec)
    lambda_note: Optional[str] = None
    try:
        fit, linfo = lambda_path(
            P, y, sizes, lambdas=list(spec.lambdas), include_prior=bool(spec.include_prior),
            folds=int(spec.folds), repeats=int(spec.repeats), seed=int(spec.seed),
            rule=spec.rule, score_fn=score_fn, max_iter=int(spec.max_iter), tol=float(spec.tol),
            nb_correction=bool(spec.nb_correction), objective=spec.objective,
            keep_fold_fits=(spec.swap_rule == 'paired_se'))
    except HyperStackFitError:
        if not spec.include_prior:
            raise
        fit = prior_fit(sizes)
        linfo = {'lambdas': ['inf'], 'path': [(float('inf'), None, None, 0)],
                 'chosen': float('inf'), 'rule': spec.rule, 'n_finite_ok': 0, 'full_path': [],
                 'folds': int(spec.folds), 'repeats': int(spec.repeats)}
        lambda_note = 'every lambda failed'
    lam = float(fit.lam)
    fit_lambda = fit                                  # the network at lambda*, before swap / shrink
    swap_rows: List[dict] = []
    swap_skipped: Optional[str] = None
    if spec.swap_rule in ('paired_se', 'paired_refit'):
        if math.isfinite(lam):
            final_entries = [int(np.argmax(np.asarray(g, dtype=float))) for g in fit.q]
            if any(e != 0 for e in final_entries):
                if spec.swap_rule == 'paired_se':
                    swap_rows = swap_test(P, y, sizes, linfo.get('fold_fits') or [],
                                          final_entries=final_entries, score_fn=score_fn,
                                          folds=int(spec.folds))
                else:
                    swap_rows = refit_swap_test(
                        P, y, sizes, lam=lam, final_entries=final_entries,
                        folds=int(spec.folds), repeats=int(spec.repeats),
                        seed=int(spec.seed) + int(spec.swap_seed_offset), score_fn=score_fn,
                        objective=spec.objective, deploy=spec.deploy,
                        member_min_weight=float(spec.member_min_weight),
                        members_max=int(spec.members_max), max_iter=int(spec.max_iter),
                        tol=float(spec.tol))
                for row in swap_rows:
                    if not row['admitted']:
                        fit = force_gate(fit, sizes, row['m'], 0)
        else:
            swap_skipped = 'lambda_inf'
    w_model_raw = [float(v) for v in fit.w_model]
    if math.isfinite(lam) and float(spec.shrink) > 0.0:
        fit = shrink_fit(fit, sizes, float(spec.shrink))
    w_dep, members = deployed_column_weights(
        fit, sizes, spec.deploy, member_min_weight=float(spec.member_min_weight),
        members_max=int(spec.members_max))
    info = {'lambda_info': linfo, 'lambda_note': lambda_note, 'swaps': swap_rows,
            'swap_skipped': swap_skipped, 'w_model_raw': w_model_raw, 'members': members,
            'fit_lambda': fit_lambda}
    return fit, w_dep, info


def _nested_worker_init() -> None:
    """Pool initializer of `nested_gate`: one torch thread per worker (the items are tiny
    matrices; N workers x the run's torch threads would oversubscribe the node, blueprint E8)."""
    torch.set_num_threads(1)


def _nested_item(args: tuple) -> dict:
    """One outer-CV item of `nested_gate` (module-level so it pickles, I10).

    ``args = (index, P_inner, y_inner, P_outer, y_outer, sizes, spec, score_fn, w_prior)``. Runs the
    complete `stack_procedure` on the inner rows and scores its DEPLOYED weights and the
    equal-weight (entry-0 mean) vector on the outer rows. Returns plain floats (``None`` for a
    degenerate score) and whether the inner procedure chose lambda = infinity. Writes no file and
    draws no random number."""
    (index, p_in, y_in, p_out, y_out, sizes, spec, score_fn, w_prior) = args
    _fit, w_dep, info = stack_procedure(p_in, y_in, sizes, spec, score_fn)
    s_fit = score_fn(p_out @ w_dep, y_out)
    s_prior = score_fn(p_out @ w_prior, y_out)
    chosen = float(info['lambda_info'].get('chosen', float('inf')))
    return {'index': int(index),
            's_fit': (float(s_fit) if _finite_score(s_fit) else None),
            's_prior': (float(s_prior) if _finite_score(s_prior) else None),
            'inner_inf': bool(not math.isfinite(chosen))}


def _run_nested_items(args_list: List[tuple], n_jobs: int) -> Tuple[List[dict], str, int]:
    """Run `_nested_item` over ``args_list``; return ``(results in item order, mode, workers)``.

    ``mode`` is ``'pool'`` (a spawn-context process pool, one torch thread per worker) or
    ``'serial'`` (in this process with torch scoped to one thread and restored afterwards - the
    same arithmetic as a worker, so both modes give bitwise-identical results, AC2.5). The pool is
    skipped inside a daemonic process (Python forbids its children) and any pool failure is logged
    and answered by the serial path; an exception raised by an ITEM is re-raised by the serial
    re-run, never swallowed."""
    import multiprocessing
    import os
    try:
        requested = int(n_jobs)
    except (TypeError, ValueError):
        requested = 1
    if requested <= 0:
        requested = os.cpu_count() or 1
    workers = max(1, min(requested, len(args_list)))
    if workers > 1:
        try:
            daemon = bool(multiprocessing.current_process().daemon)
        except Exception:                                   # pragma: no cover - defensive
            daemon = False
        if daemon:
            print('[HYPER_STACK] nested gate: running inside a daemonic worker process, which '
                  'cannot start a pool - the outer items run serially.')
            workers = 1
    if workers > 1:
        try:
            from concurrent.futures import ProcessPoolExecutor
            ctx = multiprocessing.get_context('spawn')
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                     initializer=_nested_worker_init) as pool:
                results = list(pool.map(_nested_item, args_list))
            return results, 'pool', workers
        except Exception as exc:                            # pool start / IPC / worker failure
            print(f'[HYPER_STACK] nested gate: the process pool failed ({type(exc).__name__}: '
                  f'{exc}) - the outer items run serially.')
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        results = [_nested_item(a) for a in args_list]
    finally:
        torch.set_num_threads(previous)
    return results, 'serial', 1


def nested_gate(P: np.ndarray, y: np.ndarray, sizes: Sequence[int], spec: StackSpec,
                score_fn: Callable[[np.ndarray, np.ndarray], Optional[float]], *,
                outer_folds: int = 5, outer_repeats: int = 2, seed: int, n_jobs: int = 1,
                z: float = 1.0) -> dict:
    """ver4-51, R2: estimate the out-of-sample gain of the WHOLE procedure over equal weights.

    Why: the lambda path scores the network on the rows that selected lambda, after a swap chosen
    on the same rows, before the shrink and for the gate mixture rather than the deployed vector,
    so its best score is optimistic by about the size of the signal. Here an outer repeated
    K-fold over the rows re-runs `stack_procedure` (identical ``spec``) on the inner rows of every
    item and scores its deployed weights on the outer rows against the equal-weight ensemble (entry
    0 of every model, ``prior_fit``): ``gap_i = score(P[O] @ w_dep_I) - score(P[O] @ w_prior)``.
    The network is admitted iff ``mean(gap) > z * se`` with the Nadeau-Bengio standard error
    ``sqrt(var(gap, ddof=1) * (1/outer_folds + 1/(outer_folds - 1)))``.

    Determinism (I9, I10): the partitions are ``default_rng(seed + repeat)``; items run in a spawn
    process pool (``n_jobs > 1``) or serially with one torch thread; results are collected in item
    order, so ``n_jobs`` never changes the numbers. No model is fitted and no file is written.

    Args:
      P, y, sizes, spec, score_fn: as `stack_procedure` (rows = the CV rows).
      outer_folds (>= 2), outer_repeats (>= 1), seed, n_jobs (<= 0 -> all CPUs), z.

    Returns ``{'gain', 'se', 'n_items', 'admitted', 'reason', 'inner_prior_share', 'wall_s',
    'mode', 'workers'}``. ``reason`` is ``''``, ``'too_small'`` (fewer than ``2 * outer_folds``
    rows, or inner rows < ``2 * spec.folds``: the gate is not run and ``admitted`` is False) or
    ``'degenerate'`` (a score was ``None``/non-finite: not admitted). ``gain``/``se`` are ``nan``
    for the last two.

    Raises ValueError for ``outer_folds < 2``, ``outer_repeats < 1``, bad shapes or a bad spec
    (a failure inside an item propagates after one serial re-run)."""
    t_start = time.perf_counter()
    P_np, y_np, _slices = _check_inputs(P, y, sizes)
    _check_spec(spec)
    outer_folds, outer_repeats = int(outer_folds), int(outer_repeats)
    if outer_folds < 2 or outer_repeats < 1:
        raise ValueError(f'[HYPER_STACK] nested_gate: need outer_folds >= 2 and outer_repeats >= '
                         f'1, got {outer_folds}/{outer_repeats}')
    n = int(P_np.shape[0])
    out = {'gain': float('nan'), 'se': float('nan'), 'n_items': 0, 'admitted': False,
           'reason': '', 'inner_prior_share': float('nan'), 'wall_s': 0.0, 'mode': 'none',
           'workers': 0}
    n_inner_min = n - int(math.ceil(n / outer_folds))
    if n < 2 * outer_folds or n_inner_min < 2 * int(spec.folds):
        out['reason'] = 'too_small'
        out['wall_s'] = time.perf_counter() - t_start
        return out
    w_prior = prior_fit(sizes).w
    args_list: List[tuple] = []
    item = 0
    for rep in range(outer_repeats):
        rng = np.random.default_rng(int(seed) + rep)
        for outer in _fold_indices(n, outer_folds, rng):
            outer_idx = np.asarray(outer)
            keep = np.ones(n, dtype=bool)
            keep[outer_idx] = False
            inner_idx = np.flatnonzero(keep)
            args_list.append((item, P_np[inner_idx], y_np[inner_idx], P_np[outer_idx],
                              y_np[outer_idx], [int(s) for s in sizes], spec, score_fn, w_prior))
            item += 1
    results, mode, workers = _run_nested_items(args_list, n_jobs)
    results = sorted(results, key=lambda r: r['index'])
    out['n_items'] = len(results)
    out['mode'] = mode
    out['workers'] = int(workers)
    out['inner_prior_share'] = float(np.mean([1.0 if r['inner_inf'] else 0.0 for r in results]))
    if any(r['s_fit'] is None or r['s_prior'] is None for r in results):
        out['reason'] = 'degenerate'
        out['wall_s'] = time.perf_counter() - t_start
        return out
    gaps = np.asarray([r['s_fit'] - r['s_prior'] for r in results], dtype=float)
    nb_factor = 1.0 / outer_folds + 1.0 / (outer_folds - 1)
    se = 0.0 if gaps.size <= 1 else float(math.sqrt(float(gaps.var(ddof=1)) * nb_factor))
    out['gain'] = float(gaps.mean())
    out['se'] = se
    out['admitted'] = bool(out['gain'] > float(z) * se)
    out['wall_s'] = time.perf_counter() - t_start
    return out
