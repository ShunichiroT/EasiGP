"""
ld_kernels.py
=============

ver4-4 Requirement 4.g (blueprint §2.4.2): a shared, device-aware pairwise
genotype-correlation (r^2) kernel, reused by both ``Preprocess/LD_pruning.py``
(real LD pruning - the surviving marker set is downstream-visible, so this
path MUST reproduce ``_pairwise_r2``'s own pairwise-deletion semantics
EXACTLY, invariant I7) and ``Preprocess/LD_decay_plot.py`` (a diagnostic-only
plot - already documented as tolerating a mean-imputation approximation for
its own vectorised fast path, L442-450 of that file).

This is Layer 3a (Preprocessing) in the architecture document's layering -
a peer of ``LD_pruning.py``/``LD_decay_plot.py``, never imported BY a lower
layer, and importing only ``numpy``/``pandas`` at its own top level (an
optional ``torch`` import is deferred to inside the one function that can
actually use it, so merely importing this module - or importing
``LD_pruning.py``, which now imports it - never forces a torch import on a
machine that doesn't have (or need) one).
"""

from __future__ import annotations

import numpy as np


def pairwise_r2_matrix(geno_block: np.ndarray, *, device: str = 'cpu',
                        pairwise_complete: bool = True) -> np.ndarray:
    """r^2 between every column pair of `geno_block` (samples x markers,
    ``NaN`` = missing genotype call).

    Parameters
    ----------
    geno_block : (n_samples, n_markers) array of genotype dosages, ``NaN``
        for a missing call. Always converted to ``float64`` internally,
        regardless of the input dtype, so the returned r^2 values are at
        full double precision on both the NumPy and the torch/CUDA path
        (a CUDA device otherwise defaults to float32, which would NOT
        reproduce ``LD_pruning._pairwise_r2``'s own float64 NumPy
        arithmetic to the tolerance I7 requires).
    device : ``'cpu'`` (default) uses the existing NumPy path directly.
        Any string starting with ``'cuda'`` routes through torch instead -
        this function never decides GPU eligibility itself (that is
        ``resolve_compute_resources()``'s/the caller's job, via
        ``GPU_LD_R2``); it simply trusts whatever device string it is
        given. Falls back to the NumPy path on ANY torch/CUDA failure
        (missing torch install, OOM, driver error, ...), with one log
        line - this function never raises for a device-related reason.
    pairwise_complete : bool, default True. ``True`` reproduces
        ``LD_pruning._pairwise_r2``'s own pairwise-DELETION semantics
        EXACTLY (verified - see the ver4-4 Stage 5 design record and this
        module's own test evidence): for every pair (i, j), only the
        samples where BOTH markers have a non-missing call contribute to
        that pair's r^2 - implemented via the standard "presence-mask
        matrix product" identity (derived once, in this docstring, since
        it is the one non-obvious piece of arithmetic in this file):

            Let ``Y`` = ``geno_block`` with every ``NaN`` replaced by 0,
            and ``P`` = the 0/1 presence mask (1 where NOT missing).
            For every pair (i, j):
                ``N[i,j]``   = (P.T @ P)[i,j]     - joint sample count
                ``Sx[i,j]``  = (Y.T @ P)[i,j]     - sum of x_i over the
                                                     jointly-present rows
                ``Sy[i,j]``  = Sx[j,i]            - (Y.T@P).T == P.T@Y
                ``Sxy[i,j]`` = (Y.T @ Y)[i,j]      - sum of x_i*x_j over
                                                     the jointly-present
                                                     rows (Y is already
                                                     zero wherever either
                                                     column is missing, so
                                                     an absent sample
                                                     contributes exactly 0
                                                     to this product)
                ``Sxx[i,j]`` = ((Y*Y).T @ P)[i,j]  - sum of x_i^2 over the
                                                     jointly-present rows
                ``Syy[i,j]`` = Sxx[j,i]
            from which the usual mean/covariance/variance/r^2 formulas
            follow directly. This reproduces the elementwise result of
            calling ``LD_pruning._pairwise_r2`` on every column pair
            separately, to floating-point-rounding-order tolerance (the
            two differ only in SUMMATION ORDER - one pair-by-pair NumPy
            reduction each, vs. one batched matrix product across every
            pair at once - never in the underlying statistic), computed
            via 4 matrix products instead of one Python-level double loop
            over ``n_markers^2`` pairs (the actual, measured cost R3.e/
            R4.g exist to remove).
        ``False`` uses simple mean-imputation instead (every ``NaN``
        replaced by that marker's own column mean before an ordinary
        ``np.corrcoef``) - matching ``LD_decay_plot.py``'s own, already-
        documented diagnostic-only approximation (L442-450 of that file);
        never used for real LD pruning.

    Returns
    -------
    (n_markers, n_markers) array of r^2 values (``0.0``, not ``NaN``, for
    any pair with <3 jointly-present samples, or where either marker in
    the pair is monomorphic over the jointly-present samples - the exact
    same degenerate-case handling ``LD_pruning._pairwise_r2`` itself
    applies; see that function's own body). The diagonal is NOT special-
    cased: a marker correlated with itself naturally computes to exactly
    ``1.0`` through the same formula whenever it has >=3 non-missing
    calls and non-zero variance (``cov == var``, so ``r == var/|var| ==
    1``), and to ``0.0`` (matching ``_pairwise_r2(x, x)`` exactly, NOT a
    forced ``1.0``) whenever the marker itself is monomorphic or has <3
    non-missing calls - special-casing the diagonal to always read
    ``1.0`` was tried and found to be WRONG for a monomorphic marker
    (verified directly: ``_pairwise_r2(x, x)`` returns ``0.0`` for a
    constant `x`, since its own ``std()==0`` guard fires before any
    self-correlation is computed) and was removed for that reason.

    Raises
    ------
    Nothing - always returns a matrix. A construction failure on the
    requested device falls back to the NumPy path (with one log line),
    which itself never raises for numerically degenerate input (see
    above); an outright malformed `geno_block` (wrong ndim, zero markers)
    still raises the ordinary NumPy/ValueError a caller would expect from
    misusing any NumPy-based function this way - this function does not
    swallow programming errors, only device/hardware ones.
    """
    geno_block = np.asarray(geno_block, dtype=np.float64)
    if geno_block.ndim != 2:
        raise ValueError(f"pairwise_r2_matrix: geno_block must be 2-D (samples x markers), got shape {geno_block.shape}")

    if str(device).startswith('cuda'):
        try:
            return _pairwise_r2_matrix_torch(geno_block, device=str(device), pairwise_complete=pairwise_complete)
        except Exception as exc:
            print(f"[ld_kernels] NOTE: pairwise_r2_matrix could not run on device={device!r} "
                  f"({exc}) - falling back to the NumPy/CPU path. Results are unaffected "
                  f"(the CPU path computes the identical statistic), only wall-clock time.")

    return _pairwise_r2_matrix_numpy(geno_block, pairwise_complete=pairwise_complete)


def _pairwise_r2_matrix_numpy(geno_block: np.ndarray, *, pairwise_complete: bool) -> np.ndarray:
    """NumPy implementation shared by both the CPU path and the CUDA
    fallback path of `pairwise_r2_matrix()` above - see that function's
    own docstring for the derivation. Kept as a standalone, torch-free
    function so it can be unit-tested (and used as the CUDA path's own
    correctness reference) without a torch install."""
    n_markers = geno_block.shape[1]
    if pairwise_complete:
        presence = ~np.isnan(geno_block)
        filled = np.where(presence, geno_block, 0.0)
        p = presence.astype(np.float64)

        n_joint = p.T @ p
        sx = filled.T @ p
        sy = sx.T
        sxy = filled.T @ filled
        sxx = (filled * filled).T @ p
        syy = sxx.T

        with np.errstate(invalid='ignore', divide='ignore'):
            mean_x = sx / n_joint
            mean_y = sy / n_joint
            cov = sxy / n_joint - mean_x * mean_y
            var_x = sxx / n_joint - mean_x * mean_x
            var_y = syy / n_joint - mean_y * mean_y
            r = cov / np.sqrt(var_x * var_y)
            r2 = r * r

        degenerate = (n_joint < 3) | (var_x <= 0) | (var_y <= 0) | ~np.isfinite(r2)
        r2 = np.where(degenerate, 0.0, r2)
        return r2

    # Mean-imputation mode (LD_decay_plot.py's own, pre-existing,
    # documented diagnostic-only approximation) - reproduced verbatim
    # from that file's own inline fast path.
    col_mean = np.nanmean(geno_block, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    filled = np.where(np.isnan(geno_block), col_mean, geno_block)
    col_std = filled.std(axis=0)

    with np.errstate(invalid='ignore', divide='ignore'):
        corr = np.corrcoef(filled, rowvar=False)
    r2 = np.asarray(corr) ** 2
    zero_var = (col_std[:, None] == 0) | (col_std[None, :] == 0)
    r2 = np.where(np.isnan(r2) | zero_var, 0.0, r2)
    return r2


def _pairwise_r2_matrix_torch(geno_block: np.ndarray, *, device: str, pairwise_complete: bool) -> np.ndarray:
    """torch/CUDA implementation - the exact same 4-matrix-product
    (pairwise-complete) / mean-imputed-corrcoef (not pairwise-complete)
    arithmetic as `_pairwise_r2_matrix_numpy()` above, expressed in torch
    so it can execute on a CUDA device, always accumulated in float64
    (see `pairwise_r2_matrix()`'s own docstring for why). Imported
    lazily - this is the ONLY function in this module that ever touches
    torch, so a pure-CPU/no-torch environment importing `ld_kernels` (or
    `LD_pruning`/`LD_decay_plot`, which import it) never pays for or
    triggers a torch import unless GPU_LD_R2 is both enabled AND a CUDA
    device was actually resolved."""
    import torch

    dev = torch.device(device)
    x = torch.as_tensor(geno_block, dtype=torch.float64, device=dev)

    if pairwise_complete:
        presence = ~torch.isnan(x)
        filled = torch.where(presence, x, torch.zeros((), dtype=torch.float64, device=dev))
        p = presence.to(torch.float64)

        n_joint = p.T @ p
        sx = filled.T @ p
        sy = sx.T
        sxy = filled.T @ filled
        sxx = (filled * filled).T @ p
        syy = sxx.T

        mean_x = sx / n_joint
        mean_y = sy / n_joint
        cov = sxy / n_joint - mean_x * mean_y
        var_x = sxx / n_joint - mean_x * mean_x
        var_y = syy / n_joint - mean_y * mean_y
        r = cov / torch.sqrt(var_x * var_y)
        r2 = r * r

        degenerate = (n_joint < 3) | (var_x <= 0) | (var_y <= 0) | ~torch.isfinite(r2)
        r2 = torch.where(degenerate, torch.zeros((), dtype=torch.float64, device=dev), r2)
        return r2.detach().cpu().numpy()

    col_mean = torch.nanmean(x, dim=0)
    col_mean = torch.where(torch.isnan(col_mean), torch.zeros((), dtype=torch.float64, device=dev), col_mean)
    filled = torch.where(torch.isnan(x), col_mean.unsqueeze(0), x)
    col_std = filled.std(dim=0, unbiased=True)

    centred = filled - filled.mean(dim=0, keepdim=True)
    denom = centred.std(dim=0, unbiased=True)
    denom_safe = torch.where(denom == 0, torch.ones((), dtype=torch.float64, device=dev), denom)
    normed = centred / denom_safe
    corr = (normed.T @ normed) / (filled.shape[0] - 1)
    r2 = corr * corr

    zero_var = (col_std.unsqueeze(1) == 0) | (col_std.unsqueeze(0) == 0)
    r2 = torch.where(torch.isnan(r2) | zero_var, torch.zeros((), dtype=torch.float64, device=dev), r2)
    return r2.detach().cpu().numpy()


def pairwise_r2(x: np.ndarray, y: np.ndarray, *, degenerate: float = 0.0) -> float:
    """Single-pair convenience wrapper around the SAME pairwise-deletion
    r^2 statistic `pairwise_r2_matrix(..., pairwise_complete=True)`
    computes for a whole matrix at once - shared core for
    `Preprocess.LD_pruning._pairwise_r2` and
    `Preprocess.LD_decay_plot._pairwise_r2`, which differ ONLY in what
    sentinel value they each return for a degenerate pair (0.0 for
    LD_pruning.py; NaN for LD_decay_plot.py, since that module's callers
    filter degenerate pairs out via `np.isnan()` rather than treating
    them as a real, zero-LD data point) - preserved exactly via the
    `degenerate` keyword rather than silently unifying two call sites
    that intentionally disagree.

    Always runs on CPU (a single pair is far too small an operation to
    ever benefit from a CUDA dispatch) - `device` is deliberately not a
    parameter here.
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return degenerate
    xm, ym = x[mask], y[mask]
    if xm.std() == 0 or ym.std() == 0:
        return degenerate
    r = np.corrcoef(xm, ym)[0, 1]
    return degenerate if np.isnan(r) else float(r * r)
