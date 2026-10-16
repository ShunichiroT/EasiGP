"""gat_bench.py - ver4-35, R1.a / R1.i: offline step-time benchmark of the dense GAT.

Why: the speed switches of ver4-35 (`gat_factorised_input`, `gat_sync_free`, the CPU/GPU
placement policy and `gpu_slots`) pay off by different amounts on different hardware and graph
sizes, and `resource_profiles.TORCH_DEVICE_CROSSOVER` is provisional until it is calibrated on the
machine that runs the jobs. This script times the REAL training code of
`GAT_biological_prior_knowledge` (`_try_dense_gat`, the same function a pipeline run uses) on a
synthetic graph of a size you give it, for every combination of {one-hot input, factorised input}
x {standard step, sync-free step} on the CPU and/or a CUDA device. It fits no real data, writes
nothing but its own CSV output and is never imported by the engine.

Usage (run from the EasiGP source directory):

  python tools/gat_bench.py --n-train 400 --genes 500 --max-snp 50 --edges 1500 \
      --bsize 8 --heads 4 --neuron 16 --steps 50 --device both [--out bench.csv]

One CSV row per variant on stdout (and in `--out`):
  device, factorised, sync_free, G, S, F, E0, B, HC, steps, ms_per_step, est_cpu_ms, est_gpu_ms,
  peak_mem_mb
`ms_per_step` is read from the `[gat-shape] measured_step_ms` line the fit itself prints (the
training loop only; the first fit of each device is a warm-up that is not reported).
`est_cpu_ms` / `est_gpu_ms` are the static estimates the placement policy uses, so the table shows
how far they are from the measurement. `peak_mem_mb` is the peak CUDA allocation on a GPU and the
process-wide peak RSS (a high-water mark that never decreases, so compare the first rows of a run)
on the CPU. `sync_free` only changes a CUDA step; on the CPU it is listed for completeness and is
inert.

Use the table to (1) decide whether the GPU beats the CPU at your graph size (R1.i question 2:
copy the crossover into `TORCH_DEVICE_CROSSOVER`), (2) check that the factorised input is faster
at your G (R1.b), and (3) compare steps/second with 1, 2, 4 ... parallel copies by running this
script several times at once (R1.i question 3).
"""
import argparse
import contextlib
import io
import os
import re
import resource
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_SHAPE_LINE = re.compile(r"\[gat-shape\].*?measured_step_ms=(?P<ms>[0-9.]+|n/a)")
_EST_FIELD = re.compile(r"est_step_ms\(cpu,gpu\)=(?P<cpu>[0-9.]+),(?P<gpu>[0-9.]+)")
_COLUMNS = ['device', 'factorised', 'sync_free', 'G', 'S', 'F', 'E0', 'B', 'HC', 'steps',
            'ms_per_step', 'est_cpu_ms', 'est_gpu_ms', 'peak_mem_mb']
_N_TEST = 4        # a correlation needs >= 2 test individuals; kept tiny, it is not timed


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Time the dense GAT training step (legacy/factorised x sync/sync-free) "
                    "on CPU and/or CUDA with a synthetic graph.")
    parser.add_argument('--n-train', type=int, default=400, help='training individuals')
    parser.add_argument('--genes', type=int, default=500, help='gene nodes per graph (G)')
    parser.add_argument('--max-snp', type=int, default=50, help='SNPs per gene node (S)')
    parser.add_argument('--edges', type=int, default=None,
                        help='directed edges before symmetrisation (default 3 x genes)')
    parser.add_argument('--bsize', type=int, default=8, help='batch size (graphs per step)')
    parser.add_argument('--heads', type=int, default=4, help='attention heads')
    parser.add_argument('--neuron', type=int, default=16, help='hidden channels per head')
    parser.add_argument('--steps', type=int, default=50,
                        help='training steps to time (rounded up to whole epochs)')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'both'], default='both')
    parser.add_argument('--factorised', choices=['on', 'off', 'both'], default='both',
                        help='factorised one-hot input (gat_factorised_input)')
    parser.add_argument('--sync-free', dest='sync_free', choices=['on', 'off', 'both'],
                        default='both', help='sync-free CUDA step (gat_sync_free)')
    parser.add_argument('--threads', type=int, default=0,
                        help='CPU threads for torch (0 = leave the default)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default=None, help='also write the CSV to this path')
    return parser.parse_args(argv)


def _choices(flag):
    return {'on': [True], 'off': [False], 'both': [True, False]}[flag]


def _synthetic_inputs(args, torch, np_rng):
    """Random padded node features, targets and a connected-ish random edge index of the
    requested size, shaped exactly like the arrays `GAT_biological_prior_knowledge` builds."""
    from torch_geometric.data import Data
    import torch_geometric.transforms as T
    n_genes, n_snp = args.genes, args.max_snp
    n_edges = args.edges if args.edges else 3 * n_genes
    src = np_rng.integers(0, n_genes, size=n_edges)
    dst = (src + 1 + np_rng.integers(0, max(1, n_genes - 1), size=n_edges)) % n_genes
    edge_index = torch.from_numpy(np.stack([src, dst]).astype(np.int64))
    edge_index = T.ToUndirected()(Data(edge_index=edge_index)).edge_index
    padded = {
        'train': np_rng.integers(0, 3, size=(args.n_train, n_genes, n_snp)).astype(np.float32),
        'valid': None,
        'test': np_rng.integers(0, 3, size=(_N_TEST, n_genes, n_snp)).astype(np.float32),
    }
    pheno = {'train': np_rng.normal(size=args.n_train), 'valid': None,
             'test': np_rng.normal(size=_N_TEST)}
    return padded, pheno, edge_index


def _peak_memory_mb(torch, device):
    if device == 'cuda':
        return torch.cuda.max_memory_allocated() / 1e6
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss      # kB on Linux
    return peak_kb / 1e3


def _one_fit(module, torch, inputs, args, device, factorised, sync_free, epochs):
    """One `_try_dense_gat` call; returns (stdout text, True) or (text, False) if the dense
    path declined (it prints why and the engine would fall back to the legacy loop)."""
    padded, pheno, edge_index = inputs
    n_genes = args.genes
    gene_names = np.array([f"G{i}" for i in range(n_genes)])
    gene_to_markers = [[f"snp{g}_{k}" for k in range(args.max_snp)] for g in range(n_genes)]
    shape = {'N_train': args.n_train, 'G': n_genes, 'S': args.max_snp, 'B': args.bsize,
             'epochs': epochs, 'HC': args.heads * args.neuron, 'E0': int(edge_index.shape[1]),
             'N_eval': _N_TEST, 'heads': args.heads, 'neuron': args.neuron}
    torch.manual_seed(args.seed)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = module._try_dense_gat(
            padded=padded, pheno=pheno, w_fit=None, dummy_matrix=None, edge_index=edge_index,
            n_genes=n_genes, gene_names_active=gene_names, gene_to_markers=gene_to_markers,
            effect_columns=[], neuron=args.neuron, dropout=0.1, lrate=0.01, decay=0.0,
            epochs=epochs, bsize=args.bsize, heads=args.heads, valid=False, marker_effect=False,
            samples=1, ig_batched=False, ig_internal_batch=1, prep_fast=True,
            factorised=factorised, sync_free=sync_free, device_name=device,
            shape_info=shape, model_name='gat_bench')
    return buffer.getvalue(), result is not None


def _row_from_log(text, device, factorised, sync_free, args, epochs, torch):
    shape_match = _SHAPE_LINE.search(text)
    est_match = _EST_FIELD.search(text)
    n_per_epoch = -(-args.n_train // args.bsize)
    ms = shape_match.group('ms') if shape_match else 'n/a'
    return {
        'device': device, 'factorised': int(factorised), 'sync_free': int(sync_free),
        'G': args.genes, 'S': args.max_snp,
        'F': args.max_snp if factorised else args.max_snp + args.genes,
        'E0': '', 'B': args.bsize, 'HC': args.heads * args.neuron,
        'steps': epochs * n_per_epoch, 'ms_per_step': ms,
        'est_cpu_ms': est_match.group('cpu') if est_match else 'n/a',
        'est_gpu_ms': est_match.group('gpu') if est_match else 'n/a',
        'peak_mem_mb': f"{_peak_memory_mb(torch, device):.0f}",
    }


def main(argv=None):
    args = _parse_args(argv)
    import torch
    import pipeline_utils as pu
    from models import GAT_biological_prior_knowledge as module

    devices = []
    if args.device in ('cpu', 'both'):
        devices.append('cpu')
    if args.device in ('cuda', 'both'):
        if torch.cuda.is_available():
            devices.append('cuda')
        else:
            print("[gat_bench] NOTE: no CUDA device is visible - the GPU rows are skipped.",
                  file=sys.stderr)
    if not devices:
        print("[gat_bench] nothing to run.", file=sys.stderr)
        return 1
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    np_rng = np.random.default_rng(args.seed)
    inputs = _synthetic_inputs(args, torch, np_rng)
    n_per_epoch = -(-args.n_train // args.bsize)
    epochs = max(1, -(-args.steps // n_per_epoch))
    rows = []
    for device in devices:
        resources = pu.resolve_compute_resources({
            'TORCH_DEVICE': device, 'USE_AMP': device == 'cuda',
            'N_JOBS': args.threads if args.threads > 0 else None,
            'ACCELERATION': {'gat_dense_batching': True}})
        pu.set_active_compute_resources(resources)
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        # Warm-up fit: pays imports, lazy initialisation and (CUDA) context creation once.
        _, ok = _one_fit(module, torch, inputs, args, device, False, False, 1)
        if not ok:
            print(f"[gat_bench] NOTE: the dense path declined on {device}; no rows for it.",
                  file=sys.stderr)
            continue
        for factorised in _choices(args.factorised):                 # factorised (lighter) first
            for sync_free in _choices(args.sync_free):
                if device == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                text, ok = _one_fit(module, torch, inputs, args, device, factorised, sync_free,
                                    epochs)
                if not ok:
                    print(f"[gat_bench] NOTE: {device} factorised={factorised} "
                          f"sync_free={sync_free} fell back to the legacy path; row skipped.",
                          file=sys.stderr)
                    continue
                row = _row_from_log(text, device, factorised, sync_free, args, epochs, torch)
                row['E0'] = int(inputs[2].shape[1])
                rows.append(row)

    lines = [','.join(_COLUMNS)] + [','.join(str(row[col]) for col in _COLUMNS) for row in rows]
    print('\n'.join(lines))
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as handle:
            handle.write('\n'.join(lines) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
