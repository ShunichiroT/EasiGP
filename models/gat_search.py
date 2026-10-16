"""
EasiGP - models/gat_search.py
--------------------------------------------------------------------------
ver4-46 - cheaper GAT fits INSIDE a hyperparameter search.

WHY THIS MODULE EXISTS
--------------------------------------------------------------------------
A Bayesian search fits the same GAT 30-40 times on the SAME training split. Five
things in a normal fit never change between those fits, or are never read by the
search, and used to be paid for on every candidate:

  1. the per-individual PyG ``Data`` lists, the melt/one-hot frames and one
     ``DataLoader`` collation per step (the graph topology is identical for every
     individual of the four non-biological GAT variants);
  2. the TEST pass (with attention) and the TRAIN re-prediction - the search scores the
     VALIDATION predictions only;
  3. the M^2-long edge-name lists and the attention frame;
  4. for ``GAT_prior_knowledge``: a 500-tree random forest on ALL markers plus a 50-sample
     pairwise TreeSHAP, which only decide the graph topology. ``top_rate`` is not a tunable
     field, the forest is seeded and depends on (training genotypes, phenotype, weights)
     alone, so every candidate of a task recomputed the identical result;
  5. one-off bookkeeping (per-epoch ``print``).

``lean_search_fit()`` is a batched, tensor-resident re-implementation of the TRAINING LOOP
of those models that produces the same validation predictions: it draws the SAME random
numbers in the SAME order as ``torch_geometric``'s ``DataLoader(shuffle=True)`` does (see
``_loader_base_seed`` / ``_epoch_perm``, the emulation ``GAT_biological_prior_knowledge``
already ships and self-tests), builds the same node features and the same shared edge
index, and runs the same layers - so for a fixed seed on CPU its ``predicted_valid`` equals
the legacy fit's up to floating-point summation order (checked by
``tools/gat_search_check.py``).

It is used ONLY inside a search fit (``pipeline_utils.is_lean_search_fit()``, switched on by
``HP_TUNE_FAST['gat_lean_fit']``). The confirmatory fit and every untuned fit keep running
the unchanged legacy body (I11).

``topology_for()`` is the exact, content-addressed cache of cause 4. It is used inside a
lean search fit and by the confirmatory fit of the same tuning call; with it off the model
computes the forest exactly as before.

Invariants kept: no project module imports this one at import time except the GAT model
files (I2); everything returned is plain data; nothing here reads a validation or test
label (I7).
"""

import contextlib
import hashlib
import os
import tempfile
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_mean_pool

from pipeline_utils import (
    MEMO_DIR_ENV, apply_torch_compute_settings, get_active_compute_resources, gpu_slot,
    is_lean_search_fit,
)

# --------------------------------------------------------------------------- #
# RNG emulation of DataLoader(shuffle=True) - identical to the one the biological
# GAT's dense path uses (and self-tests): one draw for the iterator's base seed, one
# draw that seeds the private generator behind randperm.
# --------------------------------------------------------------------------- #


def _loader_base_seed():
    return int(torch.empty((), dtype=torch.int64).random_().item())


def _epoch_perm(n):
    generator = torch.Generator()
    generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
    return torch.randperm(n, generator=generator)


# --------------------------------------------------------------------------- #
# Models: verbatim copies of the nested ``GAT`` classes of the legacy bodies (same
# layers, same construction order -> same parameter initialisation under one seed).
# --------------------------------------------------------------------------- #


class GATWithAttentionArg(torch.nn.Module):
    """``GAT`` of GAT_fully_connected / GAT_prior_knowledge (forward takes
    ``return_attention``)."""

    def __init__(self, hidden_channels, out_channels, dpout, heads):
        super().__init__()
        self.conv1 = GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, heads=heads,
                               concat=True, dropout=dpout)
        self.conv2 = GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, heads=heads,
                               concat=False, dropout=dpout)
        self.lin1 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch, return_attention=None):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index, return_attention_weights=return_attention)
        x = F.elu(x)
        x = global_mean_pool(x, batch)
        return self.lin1(x)


class GATPlain(torch.nn.Module):
    """``GAT`` of GAT_infinitesimal (no attention argument)."""

    def __init__(self, hidden_channels, out_channels, dpout, heads):
        super().__init__()
        self.conv1 = GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, heads=heads,
                               concat=True, dropout=dpout)
        self.conv2 = GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, heads=heads,
                               concat=False, dropout=dpout)
        self.lin1 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = global_mean_pool(x, batch)
        return self.lin1(x)


# --------------------------------------------------------------------------- #
# shared edge indices
# --------------------------------------------------------------------------- #


def fully_connected_edge_index(n_markers):
    """(2, M^2) fully-connected edge index with self-loops, in the exact order of
    GAT_fully_connected's ``_fully_connected_edge_index``."""
    edges_from = np.repeat(np.arange(n_markers), n_markers)
    edges_to = np.tile(np.arange(n_markers), n_markers)
    return torch.stack([torch.from_numpy(edges_from).to(torch.long),
                        torch.from_numpy(edges_to).to(torch.long)], dim=0)


def undirected_edge_index(edges_from, edges_to, n_markers):
    """What ``T.ToUndirected()(Data(x=..., edge_index=...))`` returns for a graph of
    `n_markers` nodes - computed ONCE for all individuals (they share the topology)."""
    edge_index = torch.stack([torch.from_numpy(np.asarray(edges_from)).to(torch.long),
                              torch.from_numpy(np.asarray(edges_to)).to(torch.long)], dim=0)
    return T.ToUndirected()(Data(edge_index=edge_index, num_nodes=int(n_markers))).edge_index


# --------------------------------------------------------------------------- #
# batching of graphs that share one topology
# --------------------------------------------------------------------------- #

# a cached one-hot block above this size is rebuilt every step instead (memory guard)
_ONEHOT_CACHE_MAX_BYTES = 512 * 1024 * 1024


class _Batcher:
    """Node features / edge index / batch vector of a mini-batch of B graphs that all
    share `edge_index` - identical to what PyG's ``Batch.from_data_list`` builds for
    them. Per-size caches hold the batched edges and (memory permitting) the one-hot
    block, so a step only gathers the genotype column."""

    def __init__(self, edge_index, n_markers, device):
        self.edge_index = edge_index.to(device)
        self.m = int(n_markers)
        self.device = device
        self._edges = {}
        self._onehot = {}
        self._eye = None

    def _edges_for(self, b):
        if b not in self._edges:
            offsets = (torch.arange(b, device=self.device) * self.m).view(1, b, 1)
            batched = (self.edge_index.view(2, 1, -1) + offsets).reshape(2, -1)
            vec = torch.arange(b, device=self.device).repeat_interleave(self.m)
            self._edges[b] = (batched, vec)
        return self._edges[b]

    def _onehot_for(self, b):
        block = self._onehot.get(b)
        if block is not None:
            return block
        if self._eye is None:
            self._eye = torch.eye(self.m, device=self.device, dtype=torch.float)
        block = self._eye.repeat(b, 1)
        if block.numel() * block.element_size() <= _ONEHOT_CACHE_MAX_BYTES:
            self._onehot[b] = block
        return block

    def build(self, genotype_rows):
        """`genotype_rows` is a (B, M) float tensor on this batcher's device."""
        b = int(genotype_rows.shape[0])
        x = torch.cat([genotype_rows.reshape(-1, 1), self._onehot_for(b)], dim=1)
        edges, vec = self._edges_for(b)
        return x, edges, vec


def lean_search_fit(*, geno_train, y_train, geno_valid, w_fit, edge_index, n_markers,
                    neuron, dropout, lrate, decay, epochs, bsize, heads, attention_arg=True):
    """Train one GAT on `geno_train` and return its VALIDATION predictions (list of float).

    Training is statement-for-statement the loop of the legacy bodies (Adam, MSE or
    weighted MSE, AMP when the resolved device is CUDA) fed from resident tensors instead
    of a ``DataLoader``; no test, train or attention output is produced.

    Args: `geno_train` / `geno_valid` (DataFrame or 2-D array, rows = individuals,
    columns = markers/nodes, phenotype column already removed), `y_train` (1-D),
    `w_fit` (1-D fitting weights aligned to `geno_train`, or None), `edge_index` (the
    shared (2, E) long tensor), `n_markers` (nodes per graph), `neuron`, `dropout`,
    `lrate`, `decay`, `epochs`, `bsize`, `heads` as in the legacy params, `attention_arg`
    (True = GAT_fully_connected / GAT_prior_knowledge model, False = GAT_infinitesimal)."""
    resources = get_active_compute_resources()
    device = torch.device(resources['device'])
    apply_torch_compute_settings(resources)
    use_amp = bool(resources['use_amp']) and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Constructed BEFORE any random draw below, exactly where the legacy body builds it,
    # so the layers are initialised from the same point of the random stream.
    model = (GATWithAttentionArg if attention_arg else GATPlain)(neuron, 1, dropout, heads)

    g_train = torch.from_numpy(np.asarray(geno_train).astype(float)).to(torch.float)
    g_valid = torch.from_numpy(np.asarray(geno_valid).astype(float)).to(torch.float)
    y = torch.from_numpy(np.asarray(y_train)).to(torch.float)
    w = None if w_fit is None else torch.from_numpy(np.asarray(w_fit, dtype=float)).to(torch.float)
    n_train = int(g_train.shape[0])
    n_valid = int(g_valid.shape[0])

    started = time.perf_counter()
    with gpu_slot():
        model.to(device)
        g_train, g_valid, y = g_train.to(device), g_valid.to(device), y.to(device)
        if w is not None:
            w = w.to(device)
        batcher = _Batcher(edge_index, n_markers, device)

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lrate, weight_decay=decay)
        for _epoch in range(int(epochs)):
            _loader_base_seed()                  # DataLoader iterator base seed
            order = _epoch_perm(n_train)         # RandomSampler(shuffle=True) order
            for start in range(0, n_train, int(bsize)):
                idx = order[start:start + int(bsize)].to(device)
                x, edges, vec = batcher.build(g_train.index_select(0, idx))
                targets = y.index_select(0, idx)
                optimizer.zero_grad()
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    out = model(x, edges, vec, None) if attention_arg else model(x, edges, vec)
                    if w is None:
                        loss = F.mse_loss(torch.squeeze(out), targets)
                    else:
                        weights = w.index_select(0, idx)
                        sq_err = (torch.squeeze(out) - targets) ** 2
                        loss = (sq_err * weights).sum() / weights.sum().clamp_min(1e-12)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        model.eval()
        predicted = []
        with torch.no_grad():
            for start in range(0, n_valid, int(bsize)):
                rows = g_valid[start:start + int(bsize)]
                x, edges, vec = batcher.build(rows)
                out = model(x, edges, vec, None) if attention_arg else model(x, edges, vec)
                predicted += out.cpu().reshape(-1).tolist()
    _note_lean_once(n_train, n_valid, n_markers, bsize, epochs, time.perf_counter() - started)
    return predicted


_LEAN_NOTED = [False]


def _note_lean_once(n_train, n_valid, n_markers, bsize, epochs, seconds):
    """One log line per process, so the run log shows the lean path was taken."""
    if not _LEAN_NOTED[0]:
        _LEAN_NOTED[0] = True
        print(f"[gat_search] lean search fit active: {n_train} train / {n_valid} valid graph(s) of "
              f"{n_markers} node(s), batch {bsize}, {epochs} epoch(s) in {seconds:.1f}s "
              f"(validation predictions only; shown once per process).")


# --------------------------------------------------------------------------- #
# GAT_prior_knowledge topology: exact, content-addressed cache
# --------------------------------------------------------------------------- #


class Topology:
    """The retained marker pairs of GAT_prior_knowledge's topology forest: positional
    indices into the training columns plus the pair strength, in the original row order."""

    __slots__ = ('pos1', 'pos2', 'value')

    def __init__(self, pos1, pos2, value):
        self.pos1 = np.asarray(pos1, dtype=np.int64)
        self.pos2 = np.asarray(pos2, dtype=np.int64)
        self.value = np.asarray(value, dtype=np.float64)

    def as_named_frame(self, columns):
        """The ``marker1`` / ``marker2`` / ``value`` frame the legacy code produced (only
        needed when interactions are emitted)."""
        names = np.asarray(columns)
        return pd.DataFrame({'marker1': names[self.pos1], 'marker2': names[self.pos2],
                             'value': self.value})


def compute_topology(train_x, train_y, w_fit, top_rate, n_jobs, rf_trees, shap_samples):
    """The legacy topology computation, moved here unchanged: a seeded random forest on all
    markers, pairwise TreeSHAP interactions of `shap_samples` rows (``sum_then_abs``), and
    the top `top_rate` % strongest pairs. Returns a :class:`Topology`."""
    import shap
    from sklearn.ensemble import RandomForestRegressor
    from models.interaction_extraction import tree_shap_interactions

    rf = RandomForestRegressor(n_estimators=rf_trees, random_state=40, n_jobs=n_jobs)
    rf.fit(train_x, train_y, sample_weight=w_fit)
    pair_named = tree_shap_interactions(
        rf, shap.sample(train_x, shap_samples), train_x.columns,
        n_jobs=n_jobs, reduce='sum_then_abs',
    )
    pair_named = pair_named[
        pair_named['value'] >= np.quantile(pair_named['value'].to_numpy(), 1 - (top_rate / 100))
    ].reset_index(drop=True)
    col_pos = {name: pos for pos, name in enumerate(train_x.columns)}
    pos1 = pair_named['marker1'].map(col_pos).astype(np.int64).to_numpy()
    pos2 = pair_named['marker2'].map(col_pos).astype(np.int64).to_numpy()
    return Topology(pos1, pos2, pair_named['value'].to_numpy())


_TOPOLOGY_FLAG = [False]
_TOPOLOGY_CACHE = {}
_TOPOLOGY_CACHE_MAX = 4


@contextlib.contextmanager
def topology_cache_context(enabled=True):
    """Enable the topology cache for the enclosed code (the parent process of a tuning call:
    its warm-up and its confirmatory fit). A lean search fit enables it on its own."""
    previous = _TOPOLOGY_FLAG[0]
    _TOPOLOGY_FLAG[0] = bool(enabled)
    try:
        yield
    finally:
        _TOPOLOGY_FLAG[0] = previous


def topology_cache_on():
    return bool(_TOPOLOGY_FLAG[0] or is_lean_search_fit())


def _topology_key(train_x, train_y, w_fit, top_rate, rf_trees, shap_samples):
    import sklearn
    try:
        import shap
        shap_version = shap.__version__
    except Exception:
        shap_version = 'none'
    hasher = hashlib.blake2b(digest_size=16)
    parts = [
        b'gatpk-topology-v1',
        pd.util.hash_pandas_object(train_x, index=False).to_numpy(dtype=np.uint64).tobytes(),
        '\x1f'.join(map(str, train_x.columns)).encode('utf-8'),
        repr(tuple(train_x.shape)).encode('ascii'),
        np.ascontiguousarray(np.asarray(train_y, dtype=np.float64)).tobytes(),
        b'' if w_fit is None else np.ascontiguousarray(np.asarray(w_fit, dtype=np.float64)).tobytes(),
        repr((float(top_rate), int(rf_trees), int(shap_samples), sklearn.__version__,
              shap_version)).encode('utf-8'),
    ]
    for part in parts:
        hasher.update(len(part).to_bytes(8, 'little'))
        hasher.update(part)
    return hasher.hexdigest()


def _file_cache_dir():
    """The shared memo directory when the run enabled ``ACCELERATION['preprocess_memo']``."""
    if not bool(get_active_compute_resources().get('accel_preprocess_memo', False)):
        return None
    return os.environ.get(MEMO_DIR_ENV) or None


def _file_get(key):
    directory = _file_cache_dir()
    if not directory:
        return None
    path = os.path.join(directory, f"gatpk_{key}.npz")
    if not os.path.isfile(path):
        return None
    try:
        with np.load(path) as stored:
            return Topology(stored['pos1'], stored['pos2'], stored['value'])
    except Exception as exc:  # corrupt entry: recompute and overwrite
        print(f"[gat_search] NOTE: topology memo entry {os.path.basename(path)} is unreadable "
              f"({type(exc).__name__}: {exc}) - recomputing.")
        return None


def _file_put(key, topology):
    directory = _file_cache_dir()
    if not directory:
        return
    tmp_path = None
    try:
        os.makedirs(directory, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(dir=directory, prefix='.gatpk_', suffix='.tmp',
                                             delete=False)
        tmp_path = handle.name
        with handle:
            np.savez(handle, pos1=topology.pos1, pos2=topology.pos2, value=topology.value)
        os.replace(tmp_path, os.path.join(directory, f"gatpk_{key}.npz"))
    except OSError:
        if tmp_path and os.path.exists(tmp_path):
            with contextlib.suppress(OSError):
                os.remove(tmp_path)


def topology_for(train_x, train_y, w_fit, top_rate, *, n_jobs, rf_trees, shap_samples):
    """The topology of GAT_prior_knowledge for this training split - from the cache when it
    is on and holds the exact same inputs, otherwise computed (and then stored).

    Exactness: the key contains every input of the computation (genotype rows and column
    names, phenotype, fitting weights, ``top_rate``, forest size, SHAP sample count and the
    scikit-learn / shap versions); the forest is seeded and ``shap.sample`` is deterministic,
    so a hit equals a recomputation. With the cache off this is a plain call of
    :func:`compute_topology`."""
    if not topology_cache_on():
        return compute_topology(train_x, train_y, w_fit, top_rate, n_jobs, rf_trees, shap_samples)
    key = _topology_key(train_x, train_y, w_fit, top_rate, rf_trees, shap_samples)
    hit = _TOPOLOGY_CACHE.get(key)
    if hit is not None:
        return hit
    hit = _file_get(key)
    if hit is None:
        started = time.perf_counter()
        hit = compute_topology(train_x, train_y, w_fit, top_rate, n_jobs, rf_trees, shap_samples)
        print(f"[gat_search] GAT_prior_knowledge topology computed once for this training split "
              f"({len(hit.value)} retained pair(s), {time.perf_counter() - started:.1f}s, "
              f"key {key[:8]}); later candidates reuse it.")
        _file_put(key, hit)
    while len(_TOPOLOGY_CACHE) >= _TOPOLOGY_CACHE_MAX:
        _TOPOLOGY_CACHE.pop(next(iter(_TOPOLOGY_CACHE)))
    _TOPOLOGY_CACHE[key] = hit
    return hit


def warm_search_caches(base_model, base_params, train, sample_weight_ctx=None):
    """Compute, ONCE and in the parent process, whatever a search of `base_model` would
    otherwise recompute on every candidate. Returns a dict describing what was warmed
    (empty when the model has no such cache). Never raises: a failure only means the first
    candidate computes it instead."""
    if base_model != 'GAT_prior_knowledge':
        return {}
    try:
        from models.GAT_prior_knowledge import _RF_N_ESTIMATORS, _SHAP_INTERACTION_SAMPLE_SIZE
        w_fit = sample_weight_ctx.w_fit if sample_weight_ctx is not None else None
        train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1]
        with topology_cache_context(True):
            topology_for(train_x, train_y, w_fit, base_params[7],
                         n_jobs=get_active_compute_resources()['n_jobs'],
                         rf_trees=_RF_N_ESTIMATORS, shap_samples=_SHAP_INTERACTION_SAMPLE_SIZE)
        return {'topology': True}
    except Exception as exc:  # noqa: BLE001 - a warm-up must never abort a search
        print(f"[gat_search] NOTE: topology warm-up failed ({type(exc).__name__}: {exc}) - the "
              f"first candidate computes it instead.")
        return {}
