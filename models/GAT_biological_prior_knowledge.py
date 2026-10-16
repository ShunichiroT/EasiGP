import functools
import hashlib
import inspect
import math
import os
import threading
import time
import traceback

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.explain import Explainer, CaptumExplainer
from torch_geometric.nn.dense.linear import is_uninitialized_parameter

from Preprocess.gene_network_prior import (
    load_network_json, extract_candidate_genes, build_gene_list, build_gene_adjacency,
)
from pipeline_utils import (
    unify_columns_by_position, get_active_compute_resources, apply_torch_compute_settings, gpu_slot,
    result_dir_path, dataloader_num_workers, torch_eval_batch_size, split_batched_edge_attention,
    timed_stage, note_cuda_context, resolve_model_device, available_memory_bytes,
    is_lean_search_fit, search_fit_cfg,
)
from Preprocess.data_driven_prior_network import (
    merge_biological_and_data_driven_networks, ensure_bio_prior_merge_cache,
)
from resource_profiles import (
    estimate_gat_step, estimate_gat_step_seconds, estimate_gat_step_seconds_calibrated,
    estimate_gat_fit_seconds_calibrated,
)
from models.gat_stability import gat_stable


# =============================================================================
# Helpers implementing requirement 2 (SNP <-> gene mapping, and collapsing
# every gene's covering SNPs into a single, fixed-width node-feature vector).
#
# The gene list and gene-gene adjacency (requirement 1) are built fresh on
# every call, straight from the network JSON and a curated gene-location CSV
# (Preprocess/gene_network_prior.py's extract_candidate_genes/
# build_gene_list/build_gene_adjacency) - there is no separate, pre-built
# gene_list.csv/adjacency.csv to manage. The one thing that IS still a
# genuinely offline, curator-driven step is resolving each gene symbol to a
# genomic location in the first place (the gene-location CSV itself) - see
# Preprocess/gene_network_prior.py's module docstring, and
# Preprocess/gene_location_agent.py, for why *that* step is kept out of this
# file (it needs a human, or an LLM agent with real literature/database
# access, to avoid silently guessing at a gene's coordinates - something
# this function's own repeated calls should never risk doing).
# =============================================================================

def _normalise_chrom(series):
    """Cast a chromosome column to a comparable string form, stripping an
    optional 'chr'/'Chr'/'CHR' prefix. marker_info.csv in this codebase uses
    bare numbers (e.g. MaizeNAM's marker_info.csv: "1,2,...,10"), while
    region files such as gene_info.csv use "chr1,chr2,..."; a curated
    gene_list.csv could plausibly follow either convention, so both sides of
    every chromosome comparison in this file go through this same
    normalisation."""
    s = series.astype(str).str.strip()
    s = s.str.replace(r'(?i)^chr', '', regex=True)
    return s


# ver4-33, R1.d: process-level count of marker_info CSV parses per path, so
# the log can show the file is parsed once per process (blueprint A-d3).
_MARKER_INFO_PARSE_COUNT = {}


@functools.lru_cache(maxsize=8)
def _load_marker_info_cached(path, size, mtime_ns):
    """Parse a marker_info file ONCE per (path, size, mtime_ns).

    Why: for PLINK runs ``params[9]`` is ``_plink_bim_marker_info.csv`` - the
    whole .bim as CSV - and every model call and every tuning trial re-parsed
    it (blueprint D3). The key contains the file's size and mtime, so a
    rewritten file is re-read.
    """
    count = _MARKER_INFO_PARSE_COUNT.get(path, 0) + 1
    _MARKER_INFO_PARSE_COUNT[path] = count
    marker_info = pd.read_csv(path, encoding='utf-8-sig')
    # Requirement 8: unify by position - 'name' here is the marker's
    # identifying column HEADER (a fixed, structural field of this file's
    # schema), not the marker names themselves (the VALUES, left
    # untouched) - so positional renaming is safe regardless of what
    # header text the file actually uses. Still validated for column
    # COUNT (a genuine schema problem, not just a naming difference).
    missing_mi_cols = 4 - marker_info.shape[1]
    if missing_mi_cols > 0:
        raise ValueError(
            f"'{path}' has only {marker_info.shape[1]} column(s) - expected at "
            f"least 4 (chromosome, name, start, end, in that order)."
        )
    marker_info = unify_columns_by_position(
        marker_info, ['chromosome', 'name', 'start', 'end'], 'marker info file'
    )
    print(f"[GAT_biological_prior_knowledge] marker_info '{os.path.basename(path)}' parsed: "
          f"{marker_info.shape[0]} marker(s) (parse #{count} of this file in this process).")
    return marker_info


def _load_marker_info(path):
    """The unified ``['chromosome', 'name', 'start', 'end', ...]`` marker
    table for `path`, parsed at most once per process (see
    ``_load_marker_info_cached``). Returns a SHALLOW copy so a caller adding
    a column cannot disturb the cached frame; callers must not modify its
    values in place."""
    stat = os.stat(path)
    return _load_marker_info_cached(path, stat.st_size, stat.st_mtime_ns).copy(deep=False)


def _map_genes_to_markers_legacy(gene_list, marker_info, available_markers):
    """Requirement 2, step 1: for every row of `gene_list`, find every SNP in
    `marker_info` that (a) is one of `available_markers` (the genotype table
    actually being modelled for this task) and (b) genomically overlaps that
    gene's [start, end] window on the same chromosome.

    A marker itself spans an interval [start, end] in marker_info.csv (see
    Data/README.md: "start: the beginning location of the marker / end: the
    end location of the marker"), so genuine interval overlap is used
    (marker.start <= gene.end AND marker.end >= gene.start) rather than
    simple point-containment - this is the standard, more general "this SNP
    falls inside this gene's region" test and is robust to exactly how the
    marker intervals happen to have been binned upstream.

    Returns
    -------
    gene_to_markers : list of list of str, aligned with gene_list.index
        gene_to_markers[i] holds the SNP names assigned to
        gene_list.iloc[i], sorted by marker start position so that the
        resulting node-feature column order is deterministic and reproducible
        across train/valid/test.
    """
    marker_info = marker_info.copy()
    marker_info = marker_info[marker_info['name'].isin(available_markers)].reset_index(drop=True)
    marker_info['_chrom_norm'] = _normalise_chrom(marker_info['chromosome'])
    marker_info['start'] = marker_info['start'].astype(float)
    marker_info['end'] = marker_info['end'].astype(float)

    gene_chrom_norm = _normalise_chrom(gene_list['chromosome'])

    gene_to_markers = []
    for i in range(gene_list.shape[0]):
        chrom = gene_chrom_norm.iloc[i]
        g_start, g_end = float(gene_list['start'].iloc[i]), float(gene_list['end'].iloc[i])
        hit = marker_info[
            (marker_info['_chrom_norm'] == chrom) &
            (marker_info['start'] <= g_end) &
            (marker_info['end'] >= g_start)
        ].sort_values('start')
        gene_to_markers.append(hit['name'].tolist())

    return gene_to_markers



def _normalise_chrom_codes(series):
    """(codes, labels): `_normalise_chrom()` applied to the UNIQUE labels of
    `series` only (a few dozen chromosomes) instead of to every row - the
    same normalisation, orders of magnitude cheaper on a million-row .bim."""
    codes, uniques = pd.factorize(series.astype(str))
    labels = _normalise_chrom(pd.Series(uniques)).tolist() if len(uniques) else []
    return codes, labels


def _map_genes_to_markers_fast(gene_list, marker_info, available_markers):
    """Exact, vectorised equivalent of ``_map_genes_to_markers_legacy``.

    Why (blueprint D3): the legacy version tests every gene against the whole
    marker table with boolean masks - O(G x M), minutes at M >= 1e5, and it
    ran on every call. Here markers are indexed once per chromosome, sorted by
    ``start``; for each gene the candidate window is two ``searchsorted``
    calls, then the hits are taken in ORIGINAL row order and ordered with the
    same ``argsort(kind='quicksort')`` that ``DataFrame.sort_values('start')``
    uses internally, so the returned lists - including the order of markers
    with equal ``start`` - are identical element for element.
    """
    keep = marker_info['name'].isin(available_markers).to_numpy()
    start = marker_info['start'].to_numpy(dtype=float)[keep]
    end = marker_info['end'].to_numpy(dtype=float)[keep]
    names = marker_info['name'].to_numpy()[keep]
    chrom_codes, chrom_labels = _normalise_chrom_codes(marker_info['chromosome'][keep])

    # Per-chromosome index: marker rows (ascending row order) whose start and
    # end are real numbers (a NaN never satisfies the legacy comparisons).
    # Labels that normalise to the same text (e.g. '1' and 'chr1') are merged.
    usable = ~(np.isnan(start) | np.isnan(end))
    rows_by_label = {}
    for code, label in enumerate(chrom_labels):
        rows = np.flatnonzero((chrom_codes == code) & usable)
        if rows.size:
            rows_by_label.setdefault(label, []).append(rows)
    chrom_index = {}
    for label, parts in rows_by_label.items():
        rows = np.sort(np.concatenate(parts)) if len(parts) > 1 else parts[0]
        by_start = rows[np.argsort(start[rows], kind='stable')]
        max_len = max(0.0, float(np.max(end[rows] - start[rows])))
        chrom_index[label] = (by_start, start[by_start], max_len)

    gene_chrom = _normalise_chrom(gene_list['chromosome']).tolist()
    gene_start = gene_list['start'].astype(float).to_numpy()
    gene_end = gene_list['end'].astype(float).to_numpy()

    gene_to_markers = []
    for i in range(gene_list.shape[0]):
        entry = chrom_index.get(gene_chrom[i])
        g_start, g_end = gene_start[i], gene_end[i]
        if entry is None or np.isnan(g_start) or np.isnan(g_end):
            gene_to_markers.append([])
            continue
        by_start, start_sorted, max_len = entry
        low_edge = g_start - max_len
        low_edge -= abs(low_edge) * 1e-12 + 1e-9        # widen: the exact test follows below
        lo = int(np.searchsorted(start_sorted, low_edge, side='left'))
        hi = int(np.searchsorted(start_sorted, g_end, side='right'))
        window = by_start[lo:hi]
        cand = np.sort(window[end[window] >= g_start])   # original row order, as the legacy mask gives
        order = np.argsort(start[cand], kind='quicksort')  # == DataFrame.sort_values('start') on the hits
        gene_to_markers.append(names[cand[order]].tolist())
    return gene_to_markers


def _map_genes_to_markers(gene_list, marker_info, available_markers):
    """Requirement 2, step 1: for every row of `gene_list`, find every SNP in
    `marker_info` that (a) is one of `available_markers` (the genotype table
    actually being modelled for this task) and (b) genomically overlaps that
    gene's [start, end] window on the same chromosome.

    A marker itself spans an interval [start, end] in marker_info.csv (see
    Data/README.md: "start: the beginning location of the marker / end: the
    end location of the marker"), so genuine interval overlap is used
    (marker.start <= gene.end AND marker.end >= gene.start) rather than
    simple point-containment - this is the standard, more general "this SNP
    falls inside this gene's region" test and is robust to exactly how the
    marker intervals happen to have been binned upstream.

    Returns
    -------
    gene_to_markers : list of list of str, aligned with gene_list.index
        gene_to_markers[i] holds the SNP names assigned to
        gene_list.iloc[i], sorted by marker start position so that the
        resulting node-feature column order is deterministic and reproducible
        across train/valid/test.
    """

    if bool(get_active_compute_resources().get('accel_bio_prior_prep_fast', True)):
        try:
            return _map_genes_to_markers_fast(gene_list, marker_info, available_markers)
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            print(f"[GAT_biological_prior_knowledge] NOTE: vectorised gene<->marker mapping failed "
                  f"({type(exc).__name__}: {exc}) - using the legacy per-gene loop for this call "
                  f"(results are identical, only slower).")
    return _map_genes_to_markers_legacy(gene_list, marker_info, available_markers)


# ---------------------------------------------------------------------------
# ver4-33, R1.d - memoised graph structure.
#
# Everything in this block depends only on the network JSON, the gene
# location CSV, the marker table, a few scalar options and the SET of
# available markers - never on the training data values - yet it was rebuilt
# on EVERY model call and every tuning trial. It is memoised per process,
# keyed on those inputs (file size + mtime included), and always returned as
# copies so no caller can disturb the stored objects.
# ---------------------------------------------------------------------------

_GRAPH_MEMO = {}              # key -> structure dict (insertion-ordered; oldest dropped past the limit)
_GRAPH_MEMO_LIMIT = 8
_GRAPH_MEMO_LOCK = threading.Lock()
_PREVIEW_MEMO = {}            # key -> sorted list of gene-window marker names
_PREVIEW_MEMO_LIMIT = 16


def _file_stat_key(path):
    """(absolute path, size, mtime_ns) - changes when the file is rewritten."""
    stat = os.stat(path)
    return (os.path.abspath(path), stat.st_size, stat.st_mtime_ns)


def _names_digest(names):
    """Order-sensitive digest of a sequence of marker names."""
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update('\x1f'.join(map(str, names)).encode('utf-8'))
    return hasher.hexdigest()


def _build_graph_structure_uncached(network_json_path, gene_location_csv_path, marker_info_path, unit,
                                    include_mediated_edges, max_hops, available_markers, fast):
    """The legacy sequence "network JSON -> gene list -> adjacency -> marker
    table -> gene<->marker mapping", unchanged in content and error messages.
    `fast` only chooses between the cached marker loader and the original
    inline ``read_csv`` (BIO_PRIOR_PREP_FAST=False)."""
    network = load_network_json(network_json_path)
    candidate_genes = extract_candidate_genes(network)
    gene_list, dropped_at_location_lookup = build_gene_list(candidate_genes, gene_location_csv_path, unit=unit)
    if gene_list.shape[0] == 0:
        raise RuntimeError(
            f"None of the candidate genes in '{network_json_path}' could be resolved to a "
            f"genomic location using '{gene_location_csv_path}' - every candidate gene was "
            f"dropped: {dropped_at_location_lookup}. Check that the gene names in the network "
            f"JSON match the gene-location CSV's name column exactly."
        )

    edge_list = build_gene_adjacency(
        network, gene_list, include_mediated_edges=include_mediated_edges, max_hops=max_hops
    )[1]
    missing_adj_cols = {'gene1_index', 'gene2_index'} - set(edge_list.columns)
    if missing_adj_cols:
        raise ValueError(f"Internal error: built adjacency is missing column(s): {sorted(missing_adj_cols)}")

    if fast:
        marker_info = _load_marker_info(marker_info_path)
    else:
        marker_info = pd.read_csv(marker_info_path, encoding='utf-8-sig')
        missing_mi_cols = 4 - marker_info.shape[1]
        if missing_mi_cols > 0:
            raise ValueError(
                f"'{marker_info_path}' has only {marker_info.shape[1]} column(s) - expected at "
                f"least 4 (chromosome, name, start, end, in that order)."
            )
        marker_info = unify_columns_by_position(
            marker_info, ['chromosome', 'name', 'start', 'end'], 'marker info file'
        )

    gene_to_markers_full = _map_genes_to_markers(gene_list, marker_info, available_markers)
    return {
        'gene_list': gene_list,
        'dropped_at_location_lookup': dropped_at_location_lookup,
        'edge_list': edge_list,
        'marker_info': marker_info,
        'gene_to_markers_full': gene_to_markers_full,
    }


def _copy_structure(structure):
    """Defensive copies of a memoised structure (cheap next to a rebuild)."""
    return {
        'gene_list': structure['gene_list'].copy(),
        'dropped_at_location_lookup': structure['dropped_at_location_lookup'],
        'edge_list': structure['edge_list'].copy(),
        'marker_info': structure['marker_info'].copy(deep=False),
        'gene_to_markers_full': [list(m) for m in structure['gene_to_markers_full']],
    }


def _graph_structure(network_json_path, gene_location_csv_path, marker_info_path, unit,
                     include_mediated_edges, max_hops, available_markers, fast=True):
    """The gene list, adjacency edge list, marker table and gene<->marker
    mapping for one bio-prior call - memoised per process when `fast`.

    Why: blueprint D3 - rebuilt per call and per tuning trial, costing seconds
    to minutes at M >= 1e5 markers. The key holds the (size, mtime) of the
    three input files, the scalar options and a digest of the available
    marker names, so any change that could alter the result is a miss.

    Returns a dict with keys ``gene_list``, ``dropped_at_location_lookup``,
    ``edge_list``, ``marker_info``, ``gene_to_markers_full`` (fresh copies).
    On any failure to key/store the memo the structure is simply rebuilt.
    """
    if not fast:
        return _build_graph_structure_uncached(
            network_json_path, gene_location_csv_path, marker_info_path, unit,
            include_mediated_edges, max_hops, available_markers, False)
    try:
        key = (_file_stat_key(network_json_path), _file_stat_key(gene_location_csv_path),
               _file_stat_key(marker_info_path), str(unit), bool(include_mediated_edges),
               int(max_hops) if max_hops is not None else None, len(available_markers),
               _names_digest(available_markers))
    except (OSError, TypeError, ValueError) as exc:
        print(f"[GAT_biological_prior_knowledge] NOTE: graph-structure memo skipped "
              f"({type(exc).__name__}: {exc}) - rebuilding it for this call.")
        return _build_graph_structure_uncached(
            network_json_path, gene_location_csv_path, marker_info_path, unit,
            include_mediated_edges, max_hops, available_markers, True)
    with _GRAPH_MEMO_LOCK:
        hit = _GRAPH_MEMO.get(key)
    if hit is not None:
        print("[GAT_biological_prior_knowledge] Graph structure (gene list, adjacency, gene<->marker "
              "mapping) reused from this process's memo.")
        return _copy_structure(hit)
    structure = _build_graph_structure_uncached(
        network_json_path, gene_location_csv_path, marker_info_path, unit,
        include_mediated_edges, max_hops, available_markers, True)
    with _GRAPH_MEMO_LOCK:
        _GRAPH_MEMO[key] = _copy_structure(structure)
        while len(_GRAPH_MEMO) > _GRAPH_MEMO_LIMIT:
            _GRAPH_MEMO.pop(next(iter(_GRAPH_MEMO)))
    return structure


def preview_gene_window_markers(network_json_path, gene_location_csv_path, unit, marker_info, marker_key):
    """Sorted list of every marker of `marker_info` that falls inside any
    gene window of the given network - what ``genomic_prediction.py``'s PLINK
    preview loop needs to decide which markers to extract.

    ver4-33, R1.d: same ``load_network_json -> extract_candidate_genes ->
    build_gene_list -> _map_genes_to_markers`` chain the preview always ran
    (so the one shared mapper is still the single authority, I6), but
    memoised per process: that loop used to repeat it for every task and
    every bio-prior instance.

    Parameters
    ----------
    marker_info : DataFrame with ``chromosome/name/start/end`` (the .bim).
    marker_key : hashable identifying `marker_info`'s content (e.g. the .bim
        path with its size and mtime) - the memo is keyed on it.
    """
    fast = bool(get_active_compute_resources().get('accel_bio_prior_prep_fast', True))

    def _compute():
        network = load_network_json(network_json_path)
        candidate_genes = extract_candidate_genes(network)
        gene_list, _ = build_gene_list(candidate_genes, gene_location_csv_path, unit=unit)
        gene_to_markers = _map_genes_to_markers(gene_list, marker_info, marker_info['name'])
        return sorted({m for markers in gene_to_markers for m in markers})

    if not fast:
        return _compute()
    try:
        key = (_file_stat_key(network_json_path), _file_stat_key(gene_location_csv_path), str(unit), marker_key)
        hash(key)
    except (OSError, TypeError) as exc:
        print(f"[GAT_biological_prior_knowledge] NOTE: gene-window preview memo skipped "
              f"({type(exc).__name__}: {exc}) - recomputing for this call.")
        return _compute()
    with _GRAPH_MEMO_LOCK:
        hit = _PREVIEW_MEMO.get(key)
    if hit is not None:
        return list(hit)
    result = _compute()
    with _GRAPH_MEMO_LOCK:
        _PREVIEW_MEMO[key] = list(result)
        while len(_PREVIEW_MEMO) > _PREVIEW_MEMO_LIMIT:
            _PREVIEW_MEMO.pop(next(iter(_PREVIEW_MEMO)))
    return result


def _reindex_edges_fast(edge_list, active_rows, n_gene_rows):
    """Vectorised version of the legacy ``iterrows`` re-index loop: map the
    adjacency's ORIGINAL gene_list row indices to the compacted 0..n-1
    active-gene indices, dropping any edge touching an inactive row. Order of
    the surviving edges is preserved.

    Returns ``(edges_from_list, edges_to_list)`` as plain Python int lists
    (what the legacy loop produced).
    """
    lookup = np.full(max(int(n_gene_rows), 1), -1, dtype=np.int64)
    lookup[np.asarray(active_rows, dtype=np.int64)] = np.arange(len(active_rows), dtype=np.int64)
    g1 = edge_list['gene1_index'].to_numpy().astype(np.int64)
    g2 = edge_list['gene2_index'].to_numpy().astype(np.int64)
    in_range = (g1 >= 0) & (g1 < lookup.shape[0]) & (g2 >= 0) & (g2 < lookup.shape[0])
    new1 = np.where(in_range, lookup[np.clip(g1, 0, lookup.shape[0] - 1)], -1)
    new2 = np.where(in_range, lookup[np.clip(g2, 0, lookup.shape[0] - 1)], -1)
    ok = (new1 >= 0) & (new2 >= 0)
    return new1[ok].tolist(), new2[ok].tolist()


def _build_padded_node_features_legacy(data_QTL, gene_to_markers, max_snp, dtype=float):
    """The original per-gene loop of :func:`_build_padded_node_features`
    (one pandas column gather per gene). Kept verbatim: it is the
    ``BIO_PRIOR_PREP_FAST=False`` path, the fallback of the vectorised build
    and the reference the byte-equality test compares against."""
    n_samples = data_QTL.shape[0]
    n_genes = len(gene_to_markers)
    features = np.zeros((n_samples, n_genes, max_snp), dtype=dtype)

    for g, markers in enumerate(gene_to_markers):
        if len(markers) == 0:
            continue
        values = data_QTL[markers].to_numpy(dtype=float)  # (n_samples, k_g)
        features[:, g, :values.shape[1]] = values

    return features


def _build_padded_node_features(data_QTL, gene_to_markers, max_snp, dtype=float, fast=True):
    """Requirement 2, step 2: for every individual (row of data_QTL) and
    every gene, collect that gene's covering SNP dosage values into a single,
    fixed-width feature vector (zero-padded up to `max_snp`, the largest
    number of SNPs mapped to any one gene in this graph) instead of
    representing each SNP as its own graph node. GATv2Conv needs every node
    in a graph to share one feature width, and different genes will
    generally have different numbers of covering SNPs, so padding to the
    graph-wide maximum is what makes a single, shared node-feature matrix
    possible; the padding value 0 sits in the same trailing feature
    positions for every individual for a given gene, so it carries no
    per-individual signal, and the per-gene dummy one-hot block added
    alongside it (see main function body) lets the network tell which gene -
    and therefore how many of its leading feature dimensions are real SNPs
    versus trailing padding - each node corresponds to.

    ``dtype`` (ver4-33, R1.f): ``float`` (default, unchanged) or
    ``np.float32`` - the dense training path builds float32 directly, which
    holds exactly the values the legacy ``.to(torch.float)`` would produce
    (the same double -> float rounding, applied per element) at half the
    host memory.

    ``fast`` (ver4-35, R1.c, engine key ``BIO_PRIOR_PREP_FAST``): build every
    gene's block with ONE column gather per chunk of genes instead of one
    pandas ``__getitem__`` per gene (seconds per fit at G ~ 10^3, repeated for
    each split, each call and each tuning trial). The values are the same
    copies the per-gene loop writes - same float64 source, same cast to
    `dtype`, padding exactly 0.0 - so the result is byte-identical
    (``np.array_equal``; blueprint A-c1). ``fast=False`` runs the legacy loop.
    The vectorised build quietly falls back to the loop (so the loop raises
    its own, original error) when a marker is missing from `data_QTL` or a
    column label is duplicated.

    Returns a numpy array of shape (n_samples, n_genes, max_snp).
    """
    if not fast:
        return _build_padded_node_features_legacy(data_QTL, gene_to_markers, max_snp, dtype=dtype)

    n_samples = data_QTL.shape[0]
    n_genes = len(gene_to_markers)
    columns = data_QTL.columns
    if not columns.is_unique:
        print("[GAT_biological_prior_knowledge] NOTE: genotype table has duplicate column labels - "
              "using the per-gene feature build (results unaffected, only slower).")
        return _build_padded_node_features_legacy(data_QTL, gene_to_markers, max_snp, dtype=dtype)

    flat_names = [marker for markers in gene_to_markers for marker in markers]
    flat_positions = columns.get_indexer(flat_names)
    if (flat_positions < 0).any():
        # a marker the loop would raise a KeyError for: let the loop do it
        return _build_padded_node_features_legacy(data_QTL, gene_to_markers, max_snp, dtype=dtype)

    # padded (G, S) table of positions into `used`; -1 marks padding
    counts = np.fromiter((len(markers) for markers in gene_to_markers), dtype=np.int64,
                         count=n_genes)
    slot = np.full((n_genes, max_snp), -1, dtype=np.int64)
    if len(flat_positions) > 0:
        gene_of_flat = np.repeat(np.arange(n_genes, dtype=np.int64), counts)
        within_gene = np.arange(len(flat_positions), dtype=np.int64) - np.repeat(
            np.cumsum(counts) - counts, counts)
        slot[gene_of_flat, within_gene] = flat_positions
    valid = slot >= 0

    if not valid.any():
        return np.zeros((n_samples, n_genes, max_snp), dtype=dtype)
    used = np.unique(slot[valid])
    n_used = len(used)
    # `source` holds only the markers some gene uses, already in `dtype` (the
    # loop's per-element float64 -> dtype cast), plus one trailing all-zero
    # column that every padding slot points at - so padding is exactly 0.0
    # without a separate masking pass.
    source = np.zeros((n_samples, n_used + 1), dtype=dtype)
    source[:, :n_used] = data_QTL.iloc[:, used].to_numpy(dtype=float)
    compact = np.where(valid, np.searchsorted(used, np.where(valid, slot, used[0])), n_used)
    # one cache-friendly row gather per individual (a whole-array fancy index
    # is several times slower: it strides across the full table)
    features = np.empty((n_samples, n_genes, max_snp), dtype=dtype)
    for i in range(n_samples):
        features[i] = source[i][compact]
    return features


def _build_dummy_matrix(n_genes, fast=True):
    """The (n_genes, n_genes) one-hot identity as float64 - one row per gene
    node. ``fast`` (R1.c) is ``np.eye``; ``fast=False`` is the original
    ``pd.get_dummies`` recipe. The two are identical 0.0/1.0 matrices
    (blueprint A-c2)."""
    if fast:
        return np.eye(n_genes, dtype=float)
    dummy = pd.get_dummies(pd.DataFrame(list(range(n_genes))), columns=[0], dtype=float)
    return dummy.to_numpy(dtype=float)


# =============================================================================
# ver4-33, R1.f / R1.g - dense-batched training/evaluation and batched
# Integrated Gradients for this model.
#
# Why: the legacy path builds one PyG ``Data`` per individual (node features
# G x (max_snp + G) including a one-hot block, i.e. O(N*G^2) host memory),
# re-collates them with a ``DataLoader`` at every step of every epoch and
# holds a GPU slot for all of it, so the device idles while other workers
# queue (design blueprint D1). Every graph here has the SAME node count G
# and the SAME ``edge_index``, so a PyG mini-batch is exactly a graph-major
# concatenation: the dense path keeps ONE (N, G, max_snp) float32 tensor,
# appends the constant (G, G) one-hot on the device per batch and re-uses
# cached batched ``edge_index``/``batch`` vectors. Model class, optimiser,
# AMP, loss (incl. BLUE weights), epochs and batch size are unchanged, and
# the torch CPU random stream is consumed exactly as the legacy
# ``DataLoader`` iterations would consume it, so a CPU run under a fixed
# seed reproduces the legacy predictions (see ``_dense_rng_selftest``).
# =============================================================================

class _BioGAT(torch.nn.Module):
    """The GAT of ``GAT_biological_prior_knowledge`` (two GATv2Conv layers,
    mean pooling, one linear read-out) at module level, layer for layer
    identical to the class defined inside the legacy function body - same
    construction order, hence the same initial random draws. Used only by
    the dense path (the legacy body keeps its own inline class)."""

    def __init__(self, hidden_channels, out_channels, dpout, heads):
        super().__init__()
        self.conv1 = GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, heads=heads, concat=True, dropout=dpout)
        self.conv2 = GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, heads=heads, concat=False, dropout=dpout)
        self.lin1 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch, return_attention, n_graphs=None):
        """``n_graphs`` (ver4-35, R1.d): the mini-batch's graph count when the
        caller knows it. ``global_mean_pool`` then skips the
        ``int(batch.max()) + 1`` it otherwise performs, which is a blocking
        device-to-host copy on CUDA; the pooled values are identical."""
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        if return_attention:
            x, attention = self.conv2(x, edge_index, return_attention_weights=return_attention)
        else:
            x = self.conv2(x, edge_index, return_attention_weights=return_attention)
        x = F.elu(x)
        x = global_mean_pool(x, batch, size=n_graphs)
        x = self.lin1(x)
        if return_attention:
            return x, attention
        return x


# =============================================================================
# ver4-35, R1.b - exactly factorised one-hot node input.
#
# Why: every node of the legacy input carries a one-hot block that says which
# gene it is, so conv1's two ``Linear(S + G -> H*C)`` layers multiply a
# (B*G, S + G) matrix that is almost all zeros - O(G^2) work and memory in the
# forward pass, the backward pass and Integrated Gradients (blueprint E2). For
# node g the one-hot block contributes exactly column ``W[:, S + g]`` of the
# weight, i.e. one embedding row, so the product can be computed as
# ``x_snp @ W[:, :S].T + W[:, S:].T[gene_of_node]`` without ever materialising
# the one-hot input. The parameters are the SAME tensors (aliased, not
# re-created), so the initial draws, the ``state_dict`` and the Adam state are
# those of the unfactorised model (acceptance A-b5).
# =============================================================================

class _FactorCtx:
    """Per-forward lookup of the factorised input: which gene each node of
    the current mini-batch is (``gene_idx``) and the factor that multiplies
    its one-hot contribution (``scale``: ones in training / prediction, the
    Integrated-Gradients path value in attribution). Shared by the two
    wrapped layers of conv1; call :meth:`set` before every forward pass."""

    def __init__(self, n_genes, device):
        self.n_genes = int(n_genes)
        self.device = device
        self.gene_idx = None
        self.scale = None
        self._indices = {}
        self._ones = {}

    def set(self, n_graphs, scale=None):
        """Prime the context for a mini-batch of `n_graphs` graphs. `scale`
        is a flat ``(n_graphs * G,)`` float tensor, or ``None`` for ones."""
        indices = self._indices.get(n_graphs)
        if indices is None:
            indices = torch.arange(self.n_genes, device=self.device).repeat(n_graphs)
            self._indices[n_graphs] = indices
        self.gene_idx = indices
        if scale is None:
            scale = self._ones.get(n_graphs)
            if scale is None:
                scale = torch.ones(n_graphs * self.n_genes, device=self.device)
                self._ones[n_graphs] = scale
        self.scale = scale


class _FactorisedLin(torch.nn.Module):
    """Replacement for one lazily initialised PyG ``Linear(-1, out)`` of
    conv1 that takes the SNP block only. ``weight`` / ``bias`` ARE the wrapped
    layer's own Parameter objects (aliased), so the module tree, parameter
    order and ``state_dict`` keys are those of the original; the wrapped layer
    itself is held outside the module tree (a tuple) so it is not registered
    twice. The lazy initialisation happens at the same point, with the same
    draws, as the original layer's forward pre-hook would trigger it."""

    def __init__(self, lin, n_snp, n_genes, ctx):
        super().__init__()
        self._holder = (lin,)
        self.n_snp = int(n_snp)
        self.n_genes = int(n_genes)
        self.ctx = ctx
        self.weight = lin.weight
        if lin.bias is None:
            self.register_parameter('bias', None)
        else:
            self.bias = lin.bias

    def original(self):
        """The wrapped PyG ``Linear`` (same parameters)."""
        return self._holder[0]

    def forward(self, x_snp):
        lin = self._holder[0]
        if is_uninitialized_parameter(self.weight):
            lin.initialize_parameters(lin, (x_snp.new_empty(1, self.n_snp + self.n_genes),))
        weight = self.weight
        out = F.linear(x_snp, weight[:, :self.n_snp], self.bias)
        embedding = weight[:, self.n_snp:].t().index_select(0, self.ctx.gene_idx)
        return out + (self.ctx.scale.unsqueeze(1) * embedding).to(out.dtype)


def _wrap_factorised_input(conv, n_snp, n_genes, ctx):
    """Swap conv1's ``lin_l`` / ``lin_r`` for :class:`_FactorisedLin` (same
    parameters). Raises ``RuntimeError`` when the installed PyG does not
    expose the lazy-initialisation API this relies on (the caller then keeps
    the unfactorised input)."""
    for name in ('lin_l', 'lin_r'):
        lin = getattr(conv, name, None)
        if (lin is None or not hasattr(lin, 'initialize_parameters')
                or not is_uninitialized_parameter(getattr(lin, 'weight', None))):
            raise RuntimeError(f"conv1.{name} is not a lazily initialised PyG Linear")
    if conv.lin_l is conv.lin_r:
        raise RuntimeError("conv1 shares one weight between lin_l and lin_r")
    conv.lin_l = _FactorisedLin(conv.lin_l, n_snp, n_genes, ctx)
    conv.lin_r = _FactorisedLin(conv.lin_r, n_snp, n_genes, ctx)


def _unwrap_factorised_input(conv):
    """Undo :func:`_wrap_factorised_input` (a no-op on an unwrapped conv): the
    original layers, which hold the trained parameters, come back so the
    legacy full-width explainer can run on the model."""
    for name in ('lin_l', 'lin_r'):
        layer = getattr(conv, name)
        if isinstance(layer, _FactorisedLin):
            setattr(conv, name, layer.original())


_FACTORISE_SELFTEST = {}


def _factorisation_selftest():
    """True if the factorised conv1 input reproduces the original layer on
    this PyG / torch build: same lazy-initialisation draws (identical
    ``state_dict``) and the same output. Run once per process on a tiny
    graph, with the global random state saved and restored so the model's own
    random stream is untouched. On failure the factorised input is off for
    the process (logged); the dense path keeps running without it."""
    if 'ok' in _FACTORISE_SELFTEST:
        return _FACTORISE_SELFTEST['ok']
    ok, reason = True, ''
    try:
        n_genes, n_snp = 5, 4
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 0], [1, 2, 3, 4, 0, 0]], dtype=torch.long)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20240607)
            x_snp = torch.randn(n_genes, n_snp)
            full = torch.cat([x_snp, torch.eye(n_genes)], dim=1)
            torch.manual_seed(7)
            reference = GATv2Conv((-1, -1), 3, add_self_loops=False, heads=2, concat=True)
            expected = reference(full, edge_index)
            torch.manual_seed(7)
            candidate = GATv2Conv((-1, -1), 3, add_self_loops=False, heads=2, concat=True)
            ctx = _FactorCtx(n_genes, torch.device('cpu'))
            _wrap_factorised_input(candidate, n_snp, n_genes, ctx)
            ctx.set(1)
            observed = candidate(x_snp, edge_index)
            same_keys = list(reference.state_dict().keys()) == list(candidate.state_dict().keys())
            same_values = same_keys and all(
                torch.equal(a, b) for a, b in zip(reference.state_dict().values(),
                                                  candidate.state_dict().values()))
            if not same_values:
                ok, reason = False, "initial state_dict differs from the unfactorised layer"
            elif not torch.allclose(expected, observed, rtol=1e-5, atol=1e-6):
                ok, reason = False, "factorised output differs from the unfactorised layer"
    except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
        ok, reason = False, f"{type(exc).__name__}: {exc}"
    _FACTORISE_SELFTEST['ok'] = ok
    if not ok:
        print(f"[GAT_biological_prior_knowledge] NOTE: factorised-input self-test failed "
              f"({reason}) - gat_factorised_input is not used in this process "
              f"(results unaffected).")
    return ok


def _loader_base_seed():
    """One draw of ``DataLoader``'s per-iteration base seed (what
    ``_BaseDataLoaderIter.__init__`` takes from the global CPU generator)."""
    return int(torch.empty((), dtype=torch.int64).random_().item())


def _epoch_perm(n):
    """The shuffled index order ``RandomSampler(generator=None)`` would
    produce for `n` items: one draw seeds a private generator, which then
    drives ``randperm``."""
    generator = torch.Generator()
    generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
    return torch.randperm(n, generator=generator)


def _batched(edge_index, n_genes, n_graphs, device):
    """``edge_index`` / ``batch`` vectors of a mini-batch of `n_graphs`
    graphs that all share `edge_index` - identical to what PyG's
    ``Batch.from_data_list`` builds for such graphs."""
    offsets = (torch.arange(n_graphs, device=device) * n_genes).view(1, n_graphs, 1)
    batched_edges = (edge_index.view(2, 1, -1) + offsets).reshape(2, -1)
    batch_vector = torch.arange(n_graphs, device=device).repeat_interleave(n_genes)
    return batched_edges, batch_vector


class _BatchCache:
    """Per-size cache of the batched ``edge_index``/``batch`` vectors (a run
    only ever sees ``bsize``, the last partial size and the eval sizes)."""

    def __init__(self, edge_index, n_genes, device):
        self.edge_index = edge_index
        self.n_genes = n_genes
        self.device = device
        self._store = {}

    def get(self, n_graphs):
        if n_graphs not in self._store:
            self._store[n_graphs] = _batched(self.edge_index, self.n_genes, n_graphs, self.device)
        return self._store[n_graphs]


_DENSE_SELFTEST = {}


def _dense_rng_selftest():
    """True if this torch build still consumes the global CPU random stream
    the way ``_loader_base_seed`` / ``_epoch_perm`` assume. Compares, for a
    shuffled and a plain ``DataLoader``, the iteration order AND the random
    state afterwards against the emulation (the global state is restored).
    Cached per process; on a mismatch the dense path is skipped (logged)
    because it could no longer reproduce the legacy random stream."""
    if 'ok' in _DENSE_SELFTEST:
        return _DENSE_SELFTEST['ok']
    from torch.utils.data import DataLoader as _PlainLoader
    ok, reason = True, ''
    try:
        with torch.random.fork_rng(devices=[]):
            for shuffle in (True, False):
                torch.manual_seed(20240607)
                real_order = [int(v) for batch in _PlainLoader(list(range(11)), batch_size=1, shuffle=shuffle)
                              for v in batch]
                real_state = torch.get_rng_state().clone()
                torch.manual_seed(20240607)
                _loader_base_seed()
                emulated = _epoch_perm(11).tolist() if shuffle else list(range(11))
                if emulated != real_order or not torch.equal(torch.get_rng_state(), real_state):
                    ok = False
                    reason = f"shuffle={shuffle}: emulated order/random state differ from DataLoader's"
                    break
    except (RuntimeError, ValueError, TypeError) as exc:
        ok, reason = False, f"{type(exc).__name__}: {exc}"
    _DENSE_SELFTEST['ok'] = ok
    if not ok:
        print(f"[GAT_biological_prior_knowledge] NOTE: dense-batching random-stream self-test failed "
              f"({reason}) - the legacy per-graph path is used for this process (results unaffected).")
    return ok


def _is_oom(exc):
    """True for a CUDA / host out-of-memory error."""
    if isinstance(exc, (torch.cuda.OutOfMemoryError, MemoryError)):
        return True
    text = str(exc).lower()
    return isinstance(exc, RuntimeError) and ('out of memory' in text or "can't allocate memory" in text)


def _ig_sample_count(samples, n_available):
    """How many graphs the legacy IG loop explains: it stops when its counter
    EQUALS `samples`, and runs through every graph if that never happens."""
    for count in range(1, n_available + 1):
        if count == samples:
            return count
    return n_available


def _captum_ig_defaults():
    """(n_steps, method) PyG's CaptumExplainer ends up using for
    IntegratedGradients: it only overrides ``internal_batch_size``, so the
    rest are captum's own defaults - read from the installed signature."""
    from captum.attr import IntegratedGradients
    parameters = inspect.signature(IntegratedGradients.attribute).parameters
    return parameters['n_steps'].default, parameters['method'].default


class _BatchedGraphModel(torch.nn.Module):
    """Maps captum's ``(k, G, F)`` input to ``(k,)`` predictions by running
    the k graphs as one disconnected batch (graphs do not interact, so each
    graph's gradient equals the one-graph-at-a-time gradient).

    ver4-35, R1.b: with a factorised model (`ctx` given) captum attributes the
    TUPLE ``(x_snp (k, G, S), gene_scale (k, G))``; ``gene_scale`` is the
    one-hot diagonal (ones), scaled along the integration path like any
    other input, and is handed to the factorised layers through `ctx`."""

    def __init__(self, model, n_genes, cache, ctx=None):
        super().__init__()
        self.model = model
        self.n_genes = n_genes
        self.cache = cache
        self.ctx = ctx

    def forward(self, node_features, gene_scale=None):
        n_graphs = node_features.shape[0]
        batched_edges, batch_vector = self.cache.get(n_graphs)
        if self.ctx is not None:
            self.ctx.set(n_graphs, scale=gene_scale.reshape(-1))
        out = self.model(node_features.reshape(n_graphs * self.n_genes, -1), batched_edges, batch_vector, None)
        return out.reshape(n_graphs)


def _ig_gene_effect(model, gather, selected, n_genes, cache, internal_batch, factorised=False,
                    ctx=None):
    """Integrated-Gradients gene attribution averaged over the graphs in
    `selected` (ver4-33, R1.g): ``attr.sum(feature axis).sum(graph axis) / k``
    with captum's own defaults and a zero baseline - the same quantity the
    legacy per-graph PyG explainer accumulates, ``internal_batch`` expanded
    graphs at a time instead of one.

    Parameters
    ----------
    gather : callable mapping a LongTensor of graph indices to their
        ``(m, G, F)`` float32 node features on the model's device
        (``F = S`` when `factorised`: the SNP block only).
    selected : LongTensor of graph indices, in explanation order.
    factorised, ctx : ver4-35, R1.b. With a factorised model, captum
        attributes the tuple ``(x_snp, gene_scale)`` against zero baselines
        and the gene score is ``a_snp.sum(2) + a_scale`` summed over graphs.
        This equals the legacy ``attr.sum(2)`` over the full ``S + G`` width:
        the one-hot inputs off the diagonal and their baselines are exactly
        0, so their attribution is exactly 0, and the diagonal entry is the
        ``gene_scale`` input.
    Returns a float32 numpy array with one score per gene node, or ``None``
    if even ``internal_batch == 1`` ran out of memory (caller falls back).
    """
    from captum.attr import IntegratedGradients

    n_steps, method = _captum_ig_defaults()
    wrapped = _BatchedGraphModel(model, n_genes, cache, ctx if factorised else None)
    explainer = IntegratedGradients(wrapped)
    n_selected = int(selected.shape[0])
    ibs = max(1, int(internal_batch))
    while True:
        per_call = max(1, ibs // n_steps)
        try:
            total = None
            for start in range(0, n_selected, per_call):
                inputs = gather(selected[start:start + per_call])
                n_inputs = int(inputs.shape[0])
                if factorised:
                    gene_scale = torch.ones(n_inputs, n_genes, dtype=inputs.dtype,
                                            device=inputs.device)
                    attributions = explainer.attribute(
                        (inputs, gene_scale), baselines=None, target=None, n_steps=n_steps,
                        method=method, internal_batch_size=max(ibs, n_inputs),
                    )
                    part = (attributions[0].detach().sum(dim=2).sum(dim=0)
                            + attributions[1].detach().sum(dim=0))
                else:
                    attributions = explainer.attribute(
                        inputs, baselines=None, target=None, n_steps=n_steps, method=method,
                        internal_batch_size=max(ibs, n_inputs),
                    )
                    part = attributions.detach().sum(dim=2).sum(dim=0)
                total = part if total is None else total + part
            return (total / n_selected).cpu().numpy()
        except (RuntimeError, MemoryError) as exc:
            if not _is_oom(exc):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if ibs == 1:
                print("[GAT_biological_prior_knowledge] NOTE: batched Integrated Gradients ran out of "
                      "memory even with internal batch 1 - using the legacy per-graph explainer.")
                return None
            ibs = max(1, ibs // 2)
            print(f"[GAT_biological_prior_knowledge] NOTE: batched Integrated Gradients ran out of "
                  f"memory - retrying with internal batch {ibs}.")


def _legacy_explainer_gene_effect(model, data_test_list, device, num_workers, pin_memory, samples):
    """The legacy PyG ``Explainer(CaptumExplainer('IntegratedGradients'))``
    loop over a per-graph ``Data`` list, as in the legacy body - used by the
    dense path when batched IG is off (or unavailable). Returns one score per
    gene node."""
    explainer = Explainer(
        model=model,
        algorithm=CaptumExplainer('IntegratedGradients'),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type=None,  # do not change here
        model_config=dict(
            mode='regression',
            task_level='node',
            return_type='raw',
        ),
    )
    ig_loader = DataLoader(data_test_list, shuffle=True, batch_size=1,
                           num_workers=num_workers, pin_memory=pin_memory)
    explanation = pd.DataFrame()
    cnt = 0
    for batch in ig_loader:
        batch = batch.to(device)
        t = explainer(batch.x, batch.edge_index, batch=batch.batch, return_attention=None)
        t = pd.DataFrame(t['node_mask'].squeeze().detach().cpu()).sum(axis=1)
        if explanation.shape[0] == 0:
            explanation = t
        else:
            explanation += t
        cnt += 1
        if cnt == samples:
            break
    return (explanation / cnt).to_numpy().flatten()


def _broadcast_gene_effect(gene_effect, gene_to_markers, columns, fast=True):
    """One-row marker-effect frame: every gene's score broadcast onto the
    markers aggregated into that gene's node (later genes overwrite earlier
    ones for markers shared by overlapping genes), exactly 0 elsewhere.

    The vectorised form is used when `columns` are unique and every marker is
    one of them; otherwise (or ``fast=False``) the original per-marker
    ``.loc`` loop runs, which also reproduces its label-enlargement quirks."""
    flat = [marker for markers in gene_to_markers for marker in markers]
    if fast and columns.is_unique:
        positions = columns.get_indexer(flat)
        if not (positions < 0).any():
            counts = [len(markers) for markers in gene_to_markers]
            values = np.repeat(np.asarray(gene_effect, dtype=np.float64)[:len(counts)], counts)
            filled = np.zeros(len(columns), dtype=np.float64)
            # last occurrence wins, deterministically (reverse, take first hits)
            unique_pos, first_in_reversed = np.unique(positions[::-1], return_index=True)
            filled[unique_pos] = values[::-1][first_in_reversed]
            return pd.DataFrame(filled.reshape(1, -1), index=[0], columns=columns)
    effect = pd.DataFrame(0.0, index=[0], columns=columns)
    for g, markers in enumerate(gene_to_markers):
        for marker in markers:
            effect.loc[0, marker] = gene_effect[g]
    return effect


def _attention_frame(edge_name_from, edge_name_to, attention):
    """The attention table exactly as the legacy tail builds it."""
    return pd.concat([pd.DataFrame(edge_name_from),
                      pd.DataFrame(edge_name_to),
                      pd.DataFrame(attention).mean().T
                      ], axis=1)


def _batched_ig_from_arrays(model, padded_test, dummy_matrix, edge_index, n_genes, samples, device,
                            internal_batch):
    """Batched IG (R1.g) for the LEGACY path: reads the test node features
    from the padded numpy array the legacy body already built (no per-graph
    ``Data`` needed), draws the same explained-graph subset the legacy IG
    loader would (``_loader_base_seed`` + ``_epoch_perm`` consume the global
    random stream identically) and returns one score per gene node, or
    ``None`` if batched IG is not applicable / ran out of memory - then the
    legacy explainer runs (with the random state restored)."""
    if n_genes == 1:
        print("[GAT_biological_prior_knowledge] NOTE: batched Integrated Gradients is skipped for a "
              "single-gene graph (the legacy explainer's squeeze() changes its output layout there) "
              "- using the legacy per-graph explainer.")
        return None
    n_test = padded_test.shape[0]
    features = torch.from_numpy(np.ascontiguousarray(padded_test, dtype=np.float32))
    onehot = torch.from_numpy(np.asarray(dummy_matrix, dtype=np.float64)).to(torch.float).to(device)

    def gather(indices):
        rows = features[indices].to(device)
        return torch.cat([rows, onehot.expand(rows.shape[0], n_genes, n_genes)], dim=2)

    cache = _BatchCache(edge_index.to(device), n_genes, device)
    rng_state = torch.get_rng_state()
    _loader_base_seed()
    order = _epoch_perm(n_test)
    selected = order[:_ig_sample_count(samples, n_test)]
    gene_effect = _ig_gene_effect(model, gather, selected, n_genes, cache, internal_batch)
    if gene_effect is None:
        torch.set_rng_state(rng_state)
    return gene_effect


def _prepare_dense(padded, targets, w_fit, dummy_matrix, edge_index, pin):
    """Host-side tensors for the dense path (outside any GPU slot).

    padded : dict split -> (N, G, max_snp) float32 array (``None`` splits skipped)
    targets : dict split -> 1-D array of phenotypes
    dummy_matrix : the (G, G) one-hot, or ``None`` for the factorised input
        (ver4-35, R1.b), which never materialises it.
    Returns a dict with ``x[split]`` (N, G, max_snp) float32, ``y[split]``
    (N,) float32, ``w`` (N,) float32 or None, ``onehot`` (G, G) float32 or
    None and ``edge_index`` (2, E) long. Values equal what the legacy
    per-graph ``torch.from_numpy(...).to(torch.float)`` conversions produce.
    """
    prepared = {'x': {}, 'y': {}, 'w': None}
    for split, array in padded.items():
        if array is None:
            continue
        tensor = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
        if pin:
            try:
                tensor = tensor.pin_memory()
            except RuntimeError as exc:
                print(f"[GAT_biological_prior_knowledge] NOTE: could not pin host memory for the "
                      f"'{split}' features ({exc}) - using pageable memory.")
        prepared['x'][split] = tensor
        prepared['y'][split] = torch.from_numpy(np.asarray(targets[split], dtype=np.float64)).to(torch.float)
    if w_fit is not None:
        w_array = np.asarray(w_fit, dtype=np.float64)
        if w_array.shape[0] != prepared['x']['train'].shape[0]:
            raise ValueError(f"fitting weight has {w_array.shape[0]} value(s) for "
                             f"{prepared['x']['train'].shape[0]} training individual(s)")
        prepared['w'] = torch.from_numpy(w_array).to(torch.float)
    if dummy_matrix is None:
        prepared['onehot'] = None
    else:
        onehot = np.asarray(dummy_matrix, dtype=np.float64)
        prepared['onehot'] = torch.from_numpy(onehot).to(torch.float)
    prepared['edge_index'] = edge_index
    return prepared


def _stage_on_device(prepared, device):
    """Move the dense tensors to `device` inside the GPU slot. Features go
    to a CUDA device only if they take <= 25 % of its free memory (and the
    transfer succeeds); otherwise they stay on the (pinned) host and each
    batch is copied with ``non_blocking``. Returns ``resident`` (bool)."""
    prepared['y'] = {split: tensor.to(device) for split, tensor in prepared['y'].items()}
    if prepared['w'] is not None:
        prepared['w'] = prepared['w'].to(device)
    if prepared['onehot'] is not None:
        prepared['onehot'] = prepared['onehot'].to(device)
    prepared['edge_index'] = prepared['edge_index'].to(device)
    if device.type != 'cuda':
        return True
    needed = sum(t.numel() * t.element_size() for t in prepared['x'].values())
    free_bytes = torch.cuda.mem_get_info(device)[0]
    if needed > 0.25 * free_bytes:
        print(f"[GAT_biological_prior_knowledge] Dense features ({needed / 1e6:.0f} MB) exceed 25% of "
              f"free device memory ({free_bytes / 1e6:.0f} MB) - keeping them on the pinned host.")
        return False
    try:
        staged = {split: tensor.to(device, non_blocking=True) for split, tensor in prepared['x'].items()}
    except (RuntimeError, MemoryError) as exc:
        if not _is_oom(exc):
            raise
        torch.cuda.empty_cache()
        print("[GAT_biological_prior_knowledge] NOTE: device ran out of memory staging the dense "
              "features - keeping them on the pinned host.")
        return False
    prepared['x'] = staged
    return True


class _PinnedStager:
    """Double-buffered pinned staging area for HOST-resident features
    (ver4-35, R1.d). ``stage(indices)`` gathers the selected rows into one
    of two pinned buffers on the host and starts an asynchronous copy to the
    device, so the host does not wait on the transfer; a buffer is reused only
    after the copy that last read it has finished (a CUDA event, normally
    already complete). If pinned allocation fails the buffers are pageable
    (logged): still correct, the copy is then simply synchronous."""

    def __init__(self, features, max_rows, device):
        self.features = features
        self.device = device
        shape = (int(max_rows),) + tuple(features.shape[1:])
        self._buffers = []
        for _ in range(2):
            try:
                self._buffers.append(torch.empty(shape, dtype=features.dtype, pin_memory=True))
            except RuntimeError as exc:
                print(f"[GAT_biological_prior_knowledge] NOTE: could not allocate pinned staging "
                      f"buffers ({exc}) - using pageable memory for the per-batch copies.")
                self._buffers.append(torch.empty(shape, dtype=features.dtype))
        self._events = [None, None]
        self._turn = 0

    def stage(self, indices):
        slot = self._turn
        self._turn = 1 - slot
        pending = self._events[slot]
        if pending is not None:
            pending.synchronize()
        rows = self._buffers[slot][:int(indices.shape[0])]
        torch.index_select(self.features, 0, indices, out=rows)
        staged = rows.to(self.device, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self._events[slot] = event
        return staged


def _make_gather(prepared, split, factorised=False, stager=None, device=None):
    """gather(indices, indices_dev=None) -> node features of the selected
    individuals on the model's device: ``(m, G, S + G)`` with the constant
    one-hot block appended, or - `factorised` (ver4-35, R1.b) - just the
    ``(m, G, S)`` SNP block.

    ``indices_dev`` (R1.d, optional) is the same index tensor already on the
    device, so device-resident features are gathered without a host-to-device
    copy of the indices; `stager` (a :class:`_PinnedStager`) serves
    host-resident features. `device` is required when `factorised` (there is
    no one-hot tensor to take it from). Raises ``ValueError`` if `factorised`
    disagrees with whether `prepared` holds a one-hot block."""
    features = prepared['x'][split]
    onehot = prepared['onehot']
    if factorised != (onehot is None):
        raise ValueError("factorised input and the one-hot block disagree "
                         f"(factorised={factorised}, one-hot present={onehot is not None})")
    target_device = onehot.device if onehot is not None else device
    if target_device is None:
        raise ValueError("a factorised gather needs the model device")

    def gather(indices, indices_dev=None):
        if stager is not None:
            rows = stager.stage(indices)
        else:
            if indices_dev is not None and indices_dev.device == features.device:
                rows = torch.index_select(features, 0, indices_dev)
            else:
                rows = features[indices.to(features.device)]
            if rows.device != target_device:
                rows = rows.to(target_device, non_blocking=True)
        if factorised:
            return rows
        n_rows = rows.shape[0]
        return torch.cat([rows, onehot.expand(n_rows, onehot.shape[0], onehot.shape[1])], dim=2)

    return gather


def _predict_dense(model, gather, n_items, batch_size, cache, device, with_attention,
                   factor_ctx=None):
    """Eval-mode predictions (and, optionally, per-graph attention averaged
    over heads) for `n_items` individuals in order, `batch_size` at a time."""
    predictions, attentions = [], []
    n_genes = cache.n_genes
    with torch.no_grad():
        for start in range(0, n_items, batch_size):
            indices = torch.arange(start, min(start + batch_size, n_items))
            n_graphs = int(indices.shape[0])
            node_features = gather(indices).reshape(n_graphs * n_genes, -1)
            batched_edges, batch_vector = cache.get(n_graphs)
            if factor_ctx is not None:
                factor_ctx.set(n_graphs)
            if with_attention:
                result, att = model(node_features, batched_edges, batch_vector, True)
                alpha = att[1]
                attentions.append(alpha.reshape(n_graphs, -1, alpha.shape[-1]).mean(dim=2))
            else:
                result = model(node_features, batched_edges, batch_vector, None)
            predictions.append(result.reshape(-1))
    flat_predictions = torch.cat(predictions).cpu().tolist() if predictions else []
    if with_attention:
        rows = torch.cat(attentions).cpu().tolist() if attentions else []
        return flat_predictions, rows
    return flat_predictions, None


def _sync_free_effective(requested, device):
    """The sync-free step (R1.d) only exists to remove host/device
    synchronisations, so it is applied on a CUDA device only; on the CPU the
    key is inert and the standard step runs."""
    return bool(requested) and device.type == 'cuda'


def _make_optimizer(model, lrate, decay, want_fused, materialise):
    """The Adam of the legacy loop; with `want_fused` (ver4-35, R1.d: CUDA +
    AMP + ``gat_sync_free``) a fused Adam instead, which lets ``GradScaler``
    skip its per-step ``found_inf`` host copy. A fused optimiser inspects
    its parameters on construction and rejects the still-uninitialised lazy
    layers of this model, so `materialise()` (one eval-mode forward of a
    single graph, which draws the lazy initialisation exactly where the first
    training step would) runs first. Any refusal falls back to the plain
    optimiser with a NOTE - never an error."""
    if want_fused:
        try:
            materialise()
            return torch.optim.Adam(model.parameters(), lr=lrate, weight_decay=decay, fused=True)
        except (RuntimeError, ValueError, TypeError) as exc:
            print(f"[GAT_biological_prior_knowledge] NOTE: fused Adam is not available "
                  f"({type(exc).__name__}: {exc}) - using the standard Adam.")
    return torch.optim.Adam(model.parameters(), lr=lrate, weight_decay=decay)


_NOTED_ONCE = set()


def _note_once(key, message):
    """Print `[GAT_biological_prior_knowledge] NOTE: <message>` the first time `key` is
    seen in this process (ver4-61: the early-stopping / time-cap notes would otherwise
    repeat for every trial of a search)."""
    if key in _NOTED_ONCE:
        return
    _NOTED_ONCE.add(key)
    print(f"[GAT_biological_prior_knowledge] NOTE: {message}")


def _monitor_valid_mse(model, monitor, batch_size, cache, device, factor_ctx):
    """Validation MSE of `model` in eval mode for early stopping (ver4-61, R4).

    Why the RNG dance: training draws its shuffles and dropout masks from the global
    torch generators; an evaluation pass must not move them, or a monitored run would
    differ from an unmonitored one even when it never stops. The CPU (and, on CUDA, the
    per-device) generator state is saved before and restored after, and the model is put
    back in train mode. Unweighted by design (BLUE weighting is a documented limitation).

    Returns the MSE as a float. Raises whatever the prediction raises (the caller turns
    that into 'monitoring off')."""
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if device.type == 'cuda' else None
    model.eval()
    try:
        predicted, _ = _predict_dense(model, monitor['gather'], monitor['n'], batch_size, cache,
                                      device, False, factor_ctx)
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        model.train()
    diff = np.asarray(predicted, dtype=float) - monitor['y']
    return float(np.mean(diff * diff))


def _train_eval_dense(model, prepared, hyper, device, use_amp, scaler, eval_batch, valid, marker_effect,
                      samples, ig_batched, ig_internal_batch, n_genes, legacy_explain,
                      factor_ctx=None, sync_free=False, early_stop=None):
    """Everything that must run while a GPU slot is held (dense path):
    training, test/valid/train predictions, attention and the IG scores.
    Mirrors the legacy slot body statement for statement, with the loader
    random draws emulated in the legacy order (train epochs, test, valid,
    train eval, IG sampler). `legacy_explain(model)` runs the legacy PyG
    explainer (used when batched IG is off or unavailable). Returns a dict.

    ver4-35: `factor_ctx` (R1.b) is the :class:`_FactorCtx` of a factorised
    model (``None`` = the one-hot is concatenated as before). `sync_free`
    (R1.d, effective on CUDA only) removes the per-step host/device
    synchronisations of the training loop: the epoch's shuffled order is
    copied to the device once, targets / weights / resident features are
    gathered with device indices, host-resident features go through a
    double-buffered pinned stager, the pool size is passed to the model, and
    an AMP run uses a fused Adam. The random draws, the batches and the
    arithmetic are unchanged. The result also carries ``train_seconds`` and
    ``n_steps`` (for the ``[gat-shape]`` line).

    ver4-61, R4: `early_stop` (``None`` = train every epoch, exactly as before) is
    ``{'patience': p, 'min_delta': d, 'source': text}``. After every epoch the model is
    scored on the validation split (eval mode, unweighted MSE, through
    :func:`_predict_dense`) with the torch / CUDA random-number state saved and restored
    around the pass, so the shuffling and dropout draws of the next epoch are exactly
    those of a run without monitoring. A CPU copy of the best ``state_dict`` is kept;
    training stops after `p` epochs without an improvement larger than ``min_delta`` and
    the best weights are put back. When training ends because every epoch ran, the final
    weights are kept (so patience >= epochs reproduces patience 0 bit for bit). Needs a
    validation split (otherwise one NOTE per process and training is unchanged); a failure
    while monitoring switches monitoring off for this fit with a NOTE.

    ver4-61, R3: inside a search fit whose context carries ``trial_deadline_s`` (an
    absolute ``time.monotonic()`` value made by ``hyperparameter_tuning`` in the process
    that runs the fit) training also stops at the first epoch boundary after the deadline
    (a NOTE ``trial truncated at epoch e/E``), keeping the weights reached so far - or the
    best ones when early stopping is also active. Fits outside a search never see it.
    The result carries ``epochs_run`` and ``n_steps`` for the epochs actually run."""
    lrate, decay, n_epochs, bsize = hyper
    factorised = factor_ctx is not None
    sync_free = _sync_free_effective(sync_free, device)
    model.to(device)
    note_cuda_context(device)
    resident = _stage_on_device(prepared, device)
    cache = _BatchCache(prepared['edge_index'], n_genes, device)
    n_train = prepared['x']['train'].shape[0]
    n_test = prepared['x']['test'].shape[0]
    stager = None
    if sync_free and not resident:
        stager = _PinnedStager(prepared['x']['train'], bsize, device)
    gather_train = _make_gather(prepared, 'train', factorised, stager, device)
    gather_test = _make_gather(prepared, 'test', factorised, None, device)

    def materialise():
        model.eval()
        with torch.no_grad():
            first = torch.zeros(1, dtype=torch.long)
            first_dev = first.to(device) if sync_free else None
            features = gather_train(first, first_dev).reshape(n_genes, -1)
            edges_one, batch_one = cache.get(1)
            if factor_ctx is not None:
                factor_ctx.set(1)
            model(features, edges_one, batch_one, None, n_graphs=1)
        model.train()

    model.train()
    optimizer = _make_optimizer(model, lrate, decay, sync_free and use_amp, materialise)
    n_batches = math.ceil(n_train / bsize)
    y_train, w_train = prepared['y']['train'], prepared['w']

    _ctx = search_fit_cfg()
    deadline = _ctx.get('trial_deadline_s') if _ctx else None
    monitor = None
    if early_stop and int(early_stop.get('patience', 0)) > 0:
        if not valid:
            _note_once('es-novalid', "early stopping was requested but this fit has no "
                       "validation split - training runs every epoch.")
        else:
            monitor = {'patience': int(early_stop['patience']),
                       'min_delta': float(early_stop.get('min_delta', 1e-4)),
                       'best_mse': math.inf, 'best_epoch': -1, 'best_state': None, 'bad': 0,
                       'y': prepared['y']['valid'].detach().cpu().numpy().astype(float),
                       'gather': _make_gather(prepared, 'valid', factorised, None, device),
                       'n': int(prepared['x']['valid'].shape[0])}
    epochs_run = 0
    ended_early = False

    train_started = time.perf_counter()
    with timed_stage('bio:train'):
        for epoch in range(n_epochs):
            loss_train_sum = 0
            _loader_base_seed()                      # DataLoader iterator base seed
            order = _epoch_perm(n_train)             # RandomSampler(shuffle=True)
            order_dev = order.to(device, non_blocking=True) if sync_free else None
            for start in range(0, n_train, bsize):
                indices = order[start:start + bsize]
                n_graphs = int(indices.shape[0])
                indices_dev = order_dev[start:start + bsize] if sync_free else None
                node_features = gather_train(indices, indices_dev).reshape(n_graphs * n_genes, -1)
                batched_edges, batch_vector = cache.get(n_graphs)
                if factor_ctx is not None:
                    factor_ctx.set(n_graphs)
                if sync_free:
                    targets = y_train.index_select(0, indices_dev)
                else:
                    targets = y_train[indices.to(y_train.device)]
                optimizer.zero_grad()
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    if sync_free:
                        out = model(node_features, batched_edges, batch_vector, None,
                                    n_graphs=n_graphs)
                    else:
                        out = model(node_features, batched_edges, batch_vector, None)
                    if w_train is None:
                        loss = F.mse_loss(torch.squeeze(out), targets)
                    else:
                        sq_err = (torch.squeeze(out) - targets) ** 2
                        weights = (w_train.index_select(0, indices_dev) if sync_free
                                   else w_train[indices.to(w_train.device)])
                        loss = (sq_err * weights).sum() / weights.sum().clamp_min(1e-12)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_train_sum += loss.detach()
            print(f'Epoch {epoch:>3} | Train Loss: {loss_train_sum/n_batches:.5f}')
            epochs_run = epoch + 1
            if monitor is not None:
                try:
                    _mse = _monitor_valid_mse(model, monitor, bsize, cache, device, factor_ctx)
                except Exception as exc:  # noqa: BLE001 - monitoring must never lose the fit
                    print(f"[GAT_biological_prior_knowledge] NOTE: early-stopping monitor failed "
                          f"at epoch {epoch + 1} ({type(exc).__name__}: {exc}) - early stopping "
                          f"is off for the rest of this fit; training continues.")
                    monitor = None
                    _mse = None
                if _mse is not None:
                    if _mse < monitor['best_mse'] - monitor['min_delta']:
                        monitor.update(best_mse=_mse, best_epoch=epoch + 1, bad=0, best_state={
                            k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
                    else:
                        monitor['bad'] += 1
                    if monitor['bad'] >= monitor['patience'] and epoch + 1 < n_epochs:
                        ended_early = True
                        print(f"[GAT_biological_prior_knowledge] early stopping at epoch "
                              f"{epoch + 1}/{n_epochs}: no validation-MSE gain > "
                              f"{monitor['min_delta']:g} for {monitor['patience']} epoch(s); best "
                              f"epoch {monitor['best_epoch']} (valid MSE {monitor['best_mse']:.5f}"
                              f"; {early_stop.get('source', 'config')}).")
                        break
            if deadline is not None and epoch + 1 < n_epochs and time.monotonic() >= deadline:
                ended_early = True
                print(f"[GAT_biological_prior_knowledge] NOTE: trial truncated at epoch "
                      f"{epoch + 1}/{n_epochs} (per-fit time cap of this search reached; the "
                      f"weights reached so far are scored).")
                break
    train_seconds = time.perf_counter() - train_started
    if monitor is not None and ended_early and monitor['best_state'] is not None:
        model.load_state_dict(monitor['best_state'])
    if monitor is not None:
        print(f"[GAT_biological_prior_knowledge] early-stopping summary: epochs_run="
              f"{epochs_run}/{n_epochs} best_epoch={monitor['best_epoch']} best_valid_mse="
              f"{monitor['best_mse']:.5f} restored_best={bool(ended_early)}.")

    model.eval()
    results = {'train_seconds': train_seconds, 'n_steps': int(epochs_run) * int(n_batches),
               'epochs_run': int(epochs_run)}

    # ver4-46, R1: a hyperparameter-SEARCH fit that asked for the lean path reads the
    # VALIDATION predictions only - the test pass (with attention) and the train
    # re-prediction are skipped. Neither consumes random numbers that affect the
    # validation predictions (eval mode, no dropout), so those are unchanged.
    lean = bool(valid) and is_lean_search_fit()
    with timed_stage('bio:predict'):
        if lean:
            results['predicted_test'], results['attention'] = [], None
            results['actual_test'] = []
        else:
            _loader_base_seed()                      # test loader (not shuffled)
            results['predicted_test'], results['attention'] = _predict_dense(
                model, gather_test, n_test, eval_batch, cache, device, True, factor_ctx)
            results['actual_test'] = prepared['y']['test'].cpu().tolist()

        results['predicted_valid'] = []
        if valid:
            _loader_base_seed()                      # valid loader
            results['predicted_valid'], _ = _predict_dense(
                model, _make_gather(prepared, 'valid', factorised, None, device),
                prepared['x']['valid'].shape[0], bsize, cache, device, False, factor_ctx)

        if lean:
            results['predicted_train'] = []
        else:
            _loader_base_seed()                      # train-eval loader (shuffle=False)
            results['predicted_train'], _ = _predict_dense(
                model, _make_gather(prepared, 'train', factorised, None, device), n_train, bsize,
                cache, device, False, factor_ctx)

    results['gene_effect'] = None
    if marker_effect:
        with timed_stage('bio:ig'):
            if ig_batched and n_genes > 1:
                rng_state = torch.get_rng_state()
                _loader_base_seed()                  # IG loader: base seed ...
                order = _epoch_perm(n_test)          # ... and its shuffled order
                selected = order[:_ig_sample_count(samples, n_test)]
                results['gene_effect'] = _ig_gene_effect(
                    model, gather_test, selected, n_genes, cache, ig_internal_batch,
                    factorised=factorised, ctx=factor_ctx)
                if results['gene_effect'] is None:
                    torch.set_rng_state(rng_state)   # let the legacy loader draw its own order
            elif ig_batched:
                print("[GAT_biological_prior_knowledge] NOTE: batched Integrated Gradients is "
                      "skipped for a single-gene graph (the legacy explainer's squeeze() changes "
                      "its output layout there) - using the legacy per-graph explainer.")
            if results['gene_effect'] is None:
                results['gene_effect'] = legacy_explain(model)
    return results


def _log_gat_shape(model_name, shape_info, device, resources, train_seconds, n_steps):
    """One ``[gat-shape]`` line per fit (ver4-35, R1.a): the problem size, the
    static per-step estimate for CPU and GPU (the numbers behind the device
    policy) and the step time actually measured, so a breeder can see where
    the time went and whether the placement was sensible. Log only; an
    estimate that cannot be computed prints ``n/a``."""
    threads = int(resources.get('policy_threads') or resources.get('torch_num_threads') or 1)
    try:
        cpu_s, gpu_s = estimate_gat_step_seconds(shape_info, threads)
        estimate = f"{cpu_s * 1000.0:.2f},{gpu_s * 1000.0:.2f}"
    except (KeyError, TypeError, ValueError) as exc:
        estimate = f"n/a ({type(exc).__name__})"
    measured = f"{train_seconds / n_steps * 1000.0:.2f}" if n_steps else "n/a"
    # ver4-61, R5: calibrated CPU estimate, appended at the END of the line so scrapers that
    # read the earlier fields are unaffected; the legacy `est_step_ms(cpu,gpu)` is unchanged.
    try:
        calibrated = f"{estimate_gat_step_seconds_calibrated(shape_info, threads) * 1000.0:.2f}"
    except (KeyError, TypeError, ValueError) as exc:
        calibrated = f"n/a ({type(exc).__name__})"
    print(f"[gat-shape] model={model_name} N_train={shape_info['N_train']} G={shape_info['G']} "
          f"S={shape_info['S']} F={shape_info['F']} E0={shape_info['E0']} B={shape_info['B']} "
          f"epochs={shape_info['epochs']} HC={shape_info['HC']} device={device} "
          f"est_step_ms(cpu,gpu)={estimate} measured_step_ms={measured} "
          f"est_cal_ms={calibrated}")


def _try_dense_gat(*, padded, pheno, w_fit, dummy_matrix, edge_index, n_genes, gene_names_active,
                   gene_to_markers, effect_columns, neuron, dropout, lrate, decay, epochs, bsize, heads,
                   valid, marker_effect, samples, ig_batched, ig_internal_batch, prep_fast,
                   factorised=False, sync_free=False, device_name=None, shape_info=None,
                   model_name='', early_stop=None):
    """Dense-batched fit/evaluation (R1.f) returning the model's 7-tuple, or
    ``None`` when the dense path cannot be used - the caller then runs the
    unchanged legacy body (logged; results unaffected).

    The broad ``except Exception`` is deliberate and confined to the fit:
    whatever goes wrong inside the dense machinery must degrade to the
    legacy path rather than lose a task (blueprint R1.f failure modes).
    Metric / output assembly runs outside it, so a genuine data error
    (e.g. too few test individuals for a correlation) surfaces exactly as
    in the legacy path.

    Parameters mirror the legacy body's own locals: `padded` / `pheno` are
    dicts keyed 'train', 'valid' (None if absent), 'test'; `edge_index` is
    the shared undirected (2, E) edge index; `effect_columns` are
    ``data_test.iloc[:, :-1].columns`` (the effect width, I3).

    ver4-35 additions (all default to the ver4-33 behaviour): `dummy_matrix`
    may be ``None`` (built on demand, only for a fallback); `factorised`
    (R1.b) feeds conv1 the SNP block only; `sync_free` (R1.d) removes the
    host/device synchronisations of a CUDA training step; `device_name` (R1.e)
    is the device the placement policy chose (default: the resources'
    device); `shape_info` / `model_name` only feed the ``[gat-shape]`` line.

    ver4-61, R4: `early_stop` (``None`` = off) is forwarded to
    :func:`_train_eval_dense`; see there.
    """
    if not _dense_rng_selftest():
        return None
    resources = dict(get_active_compute_resources())
    if device_name is not None:
        resources['device'] = str(device_name)
    device = torch.device(resources['device'])
    factorised_on = bool(factorised) and _factorisation_selftest()
    try:
        model = _BioGAT(neuron, 1, dropout, heads)
        factor_ctx = None
        if factorised_on:
            factor_ctx = _FactorCtx(n_genes, device)
            try:
                _wrap_factorised_input(model.conv1, padded['train'].shape[2], n_genes, factor_ctx)
            except RuntimeError as exc:
                print(f"[GAT_biological_prior_knowledge] NOTE: factorised input not applied "
                      f"({exc}) - using the one-hot input for this call (results unaffected).")
                factor_ctx = None
        apply_torch_compute_settings(resources)
        use_amp = bool(resources['use_amp']) and device.type == 'cuda'
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        eval_batch = torch_eval_batch_size(resources, resources.get('gpu_eval_batch'))
        num_workers = dataloader_num_workers(resources)
        pin_memory = device.type == 'cuda'

        if factor_ctx is not None:
            onehot_source = None
        elif dummy_matrix is not None:
            onehot_source = dummy_matrix
        else:
            onehot_source = _build_dummy_matrix(n_genes, fast=prep_fast)
        prepared = _prepare_dense(padded, pheno, w_fit, onehot_source, edge_index, pin_memory)

        def legacy_explain(trained_model):
            # per-graph Data list for the test split, built only here (R1.g off
            # or unavailable) - same values as the legacy per-graph tensors.
            # A factorised model takes the SNP block only, so its original
            # (same-parameter) layers are put back first.
            _unwrap_factorised_input(trained_model.conv1)
            if prepared['onehot'] is None:
                onehot = torch.from_numpy(
                    _build_dummy_matrix(n_genes, fast=prep_fast)).to(torch.float)
            else:
                onehot = prepared['onehot'].cpu()
            test_x = padded['test']
            graphs = []
            for kk in range(test_x.shape[0]):
                graph = Data()
                graph.x = torch.cat([torch.from_numpy(np.ascontiguousarray(test_x[kk], dtype=np.float32)),
                                     onehot], dim=1)
                graph.y = prepared['y']['test'][kk:kk + 1].cpu()
                graph.edge_index = edge_index.clone()
                graphs.append(graph)
            return _legacy_explainer_gene_effect(trained_model, graphs, device, num_workers, pin_memory, samples)

        print(f"[GAT_biological_prior_knowledge] Dense-batched fit: {prepared['x']['train'].shape[0]} "
              f"train graph(s) of {n_genes} gene node(s), batch size {bsize}"
              f"{', factorised one-hot input' if factor_ctx is not None else ''}"
              f"{', sync-free step' if sync_free and device.type == 'cuda' else ''}.")
        with gpu_slot(device):
            results = _train_eval_dense(
                model, prepared, (lrate, decay, epochs, bsize), device, use_amp, scaler, eval_batch,
                valid, marker_effect, samples, ig_batched, ig_internal_batch, n_genes,
                legacy_explain, factor_ctx=factor_ctx, sync_free=sync_free,
                early_stop=early_stop)
    except Exception as exc:  # noqa: BLE001 - deliberate fallback boundary, see docstring
        print(f"[GAT_biological_prior_knowledge] NOTE: dense-batched path failed "
              f"({type(exc).__name__}: {exc}) - falling back to the legacy per-graph path for this call "
              f"(results unaffected, only slower).")
        traceback.print_exc()
        return None

    if shape_info is not None:
        shape_info = dict(shape_info, factorised=factor_ctx is not None)
        shape_info['F'] = (shape_info['S'] if factor_ctx is not None
                           else shape_info['S'] + shape_info['G'])
        if int(results.get('epochs_run', shape_info['epochs'])) != int(shape_info['epochs']):
            print(f"[gat-shape] model={model_name} epochs_run={results['epochs_run']}/"
                  f"{shape_info['epochs']} (training ended before the configured epochs).")
        _log_gat_shape(model_name, shape_info, device, resources, results['train_seconds'],
                       results['n_steps'])

    predicted_test = results['predicted_test']
    if bool(valid) and is_lean_search_fit():
        # ver4-46, R1: lean search fit - nothing below is read by the search.
        return (float('nan'), float('nan'), pd.DataFrame(), predicted_test,
                results['predicted_valid'], results['predicted_train'], pd.DataFrame())
    mse = mean_squared_error(results['actual_test'], predicted_test)
    r = pearsonr(results['actual_test'], predicted_test)[0]

    if marker_effect:
        effect = _broadcast_gene_effect(results['gene_effect'], gene_to_markers, effect_columns, fast=prep_fast)
    else:
        effect = pd.DataFrame()

    edge_name_from = list(gene_names_active[edge_index[0].tolist()])
    edge_name_to = list(gene_names_active[edge_index[1].tolist()])
    attention = _attention_frame(edge_name_from, edge_name_to, results['attention'])
    return (r, mse, effect, predicted_test, results['predicted_valid'], results['predicted_train'], attention)


_COST_GUARD_SEEN = set()


def _bio_prior_cost_guard(model_name, shape, host_bytes, merge_enabled, resources):
    """Print what one bio-prior GAT fit is about to cost, and stop it if the
    user asked for that (ver4-35, R1.h1).

    Why: the data-driven merge turns every RF-selected marker outside a gene
    into a graph node and adds up to ``top_rate`` % of M_sel^2 / 2 edges, so
    ``G`` (and with it the O(G * F) input work, the host feature memory and
    the per-step cost) can grow by orders of magnitude without any message
    (blueprint E7). Nothing here changes the model: ``'warn'`` (default)
    only logs; ``'raise'`` stops BEFORE the fit when a threshold is exceeded;
    ``'off'`` prints nothing.

    Thresholds: ``BIO_PRIOR_MAX_FIT_GFLOP`` (``resources['accel_bio_prior_max_fit_gflop']``,
    unset = no FLOP check) against the estimated training FLOPs of the fit,
    and 50 % of ``available_memory_bytes()`` against `host_bytes`, the host
    memory the node features need.

    Raises ``RuntimeError`` only in ``'raise'`` mode with a threshold exceeded.
    An estimate that cannot be computed is logged and skipped (never fatal)."""
    mode = str(resources.get('accel_bio_prior_cost_guard', 'warn'))
    if mode == 'off':
        return
    try:
        step = estimate_gat_step(shape)
        n_steps = int(shape['epochs']) * math.ceil(shape['N_train'] / max(1, shape['B']))
        fit_gflop = step['flop'] * n_steps / 1e9
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        print(f"[bio-prior cost] NOTE: {model_name}: the cost estimate could not be computed "
              f"({type(exc).__name__}: {exc}) - no cost guard for this fit.")
        return
    available = available_memory_bytes()
    mem_over = available is not None and host_bytes > 0.5 * available
    limit = resources.get('accel_bio_prior_max_fit_gflop')
    flop_over = limit is not None and fit_gflop > float(limit)
    key = (model_name, shape['G'], shape['S'], shape['N_train'], shape['B'], shape['epochs'],
           shape['HC'], bool(merge_enabled), mem_over, flop_over)
    available_text = f"{available / 1e9:.1f} GB" if available is not None else "unknown"
    if key not in _COST_GUARD_SEEN or (mode == 'raise' and (mem_over or flop_over)):
        _COST_GUARD_SEEN.add(key)
        # ver4-61, R5: calibrated CPU wall-clock estimate and the edge-activation memory
        # (R2), appended after the legacy text so earlier wording is unchanged.
        try:
            threads = int(resources.get('policy_threads') or resources.get('torch_num_threads')
                          or 1)
            fit_hours = estimate_gat_fit_seconds_calibrated(shape, threads) / 3600.0
            hours_text = f"≈ {fit_hours:.1f} h per fit on CPU ({threads} thread(s), calibrated)"
        except (KeyError, TypeError, ValueError) as exc:
            hours_text = f"calibrated time n/a ({type(exc).__name__})"
        edge_gb = step['bytes_edge_step'] / 1e9
        print(f"[bio-prior cost] {model_name}: G={shape['G']} S={shape['S']} "
              f"N_train={shape['N_train']} E0~{shape['E0']} -> ~{fit_gflop:,.0f} GFLOP per fit "
              f"(estimate, training steps only), node features need ~{host_bytes / 1e9:.2f} GB "
              f"host memory (available {available_text}); guard={mode}, data-driven merge "
              f"{'ON' if merge_enabled else 'off'}; {hours_text}; edge activations "
              f"≈ {edge_gb:.1f} GB per training step.")
        if mem_over or flop_over:
            reasons = []
            if flop_over:
                reasons.append(f"~{fit_gflop:,.0f} GFLOP exceeds "
                               f"BIO_PRIOR_MAX_FIT_GFLOP={float(limit):,.0f}")
            if mem_over:
                reasons.append(f"~{host_bytes / 1e9:.2f} GB of node features exceeds 50% of the "
                               f"{available_text} available")
            advice = ("lower the data-driven merge's top_rate, set its max_edges (keep "
                      "the N strongest pairs) or tighten its rf_filter (fewer selected markers "
                      "-> fewer bare nodes and edges), reduce epochs, or enable "
                      "gat_factorised_input / gat_dense_batching" if merge_enabled else
                      "reduce epochs or the network size, or enable gat_factorised_input / "
                      "gat_dense_batching")
            print(f"[bio-prior cost] WARNING: {model_name}: {'; '.join(reasons)}. To cut it: "
                  f"{advice}. (ACCELERATION bio_prior_cost_guard='off' silences this, 'raise' "
                  f"stops the fit.)")
    if mode == 'raise' and (mem_over or flop_over):
        raise RuntimeError(
            f"[bio-prior cost] {model_name}: the fit exceeds the configured cost limit "
            f"(G={shape['G']}, S={shape['S']}, N_train={shape['N_train']}, "
            f"~{fit_gflop:,.0f} GFLOP, ~{host_bytes / 1e9:.2f} GB host features) and "
            f"bio_prior_cost_guard='raise'. "
            f"Lower the data-driven merge's top_rate / set max_edges / tighten rf_filter, reduce "
            f"epochs, or set "
            f"bio_prior_cost_guard='warn'.")


def bio_prior_trial_peak_bytes(graph, *, batch_hi, hc_hi, n_train, n_eval):
    """Predicted peak host memory of ONE bio-prior trial, in bytes (ver4-61, R2).

    Why: the node-activation model behind ``estimate_gat_step`` ignored the edge tensors
    GATv2Conv keeps for backward (``B x E0 x HC`` floats, several of them), which on a
    data-driven-merge graph are tens of GB per trial; ``genomic_prediction.GP()`` sets the
    trial concurrency from this number so that the operating system does not SIGKILL the
    workers (blueprint R2). The memory model itself lives in ``resource_profiles`` (I3);
    this function only turns a built graph and the search-box corner into its input.

    Args: `graph` the dict `_prepare_bio_graph` returns (needs ``edges_from``, ``n_genes``,
    ``max_snp``); `batch_hi` the largest batch size the search can draw; `hc_hi` the
    largest ``heads * neuron`` it can draw (after any `cap_search_cost` narrowing);
    `n_train` / `n_eval` row counts of the task. Returns bytes (float, > 0).
    Raises KeyError / TypeError / ValueError if `graph` lacks the fields or a count is
    not positive (the caller turns that into 'guard inactive' with a NOTE)."""
    n_genes = int(graph['n_genes'])
    shape = {
        'G': n_genes, 'S': int(graph['max_snp']), 'N_train': int(n_train),
        'N_eval': int(n_eval), 'E0': 2 * (len(graph['edges_from']) - n_genes) + n_genes,
        'B': int(batch_hi), 'HC': int(hc_hi), 'epochs': 1, 'factorised': False,
        'ig_internal_batch': 1, 'F': int(graph['max_snp']) + n_genes,
    }
    if min(shape['G'], shape['S'], shape['N_train'], shape['B'], shape['HC']) <= 0:
        raise ValueError(f"non-positive shape for the trial memory estimate: {shape}")
    return float(estimate_gat_step(shape)['mem_trial_host'])


# =============================================================================
# ver4-61, R1: the hyperparameter-independent bio-prior graph, built once per
# task and instance.
#
# Why: every tuning trial (and the confirmatory fit) used to repeat the whole
# "structure -> data-driven merge -> edge re-index -> self-loops" sequence,
# although none of it depends on a tunable value (params[0:7]) - only on
# params[7:14] and the marker pool. With the data-driven merge on PLINK input
# the merge step alone cost ~20 minutes per call (blueprint ver4-61 R1).
# The graph is memoised inside the task-owned `network_cache[model_name]`
# entry, so it is created by the parent process (`prewarm_bio_graph`) and
# travels by value inside every pickled ModelTrialRunner (I9, I10).
# Gate: ACCELERATION.BIO_PRIOR_PREP_FAST (`accel_bio_prior_prep_fast`); False
# means the legacy rebuild on every call, with no `[bio-graph]` log lines.
# =============================================================================

def _bio_graph_key(params, available_markers):
    """Content key of the bio-prior graph for one task and instance.

    The graph depends on the network/location/marker-info files, the scalar
    network options and the data-driven merge config (all `params[7:14]`) and
    on which markers exist in the pool - never on `params[0:7]` (the tunable
    architecture/training values) nor on `params[14:16]` (explainer options).

    Args:
        params: the flat bio-prior parameter list (at least 14 entries).
        available_markers: the pool's marker names (`data_train.columns[:-1]`).

    Returns:
        A 32-character hex digest.

    Raises:
        OSError: a referenced input file cannot be stat-ed.
        TypeError, ValueError: `params[7:14]` cannot be serialised.
    """
    import json
    fixed = list(params[7:14])
    files = [_file_stat_key(str(fixed[0])), _file_stat_key(str(fixed[1])),
             _file_stat_key(str(fixed[2]))]
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(json.dumps(fixed, sort_keys=True, default=str).encode('utf-8'))
    hasher.update(json.dumps(files, default=str).encode('utf-8'))
    hasher.update(str(len(available_markers)).encode('utf-8'))
    hasher.update(_names_digest(available_markers).encode('utf-8'))
    return hasher.hexdigest()


def _copy_bio_graph(graph):
    """Defensive copy of a graph dict so a consumer can never alter the memo."""
    out = dict(graph)
    out['gene_list_active'] = graph['gene_list_active'].copy()
    out['gene_to_markers'] = [list(m) for m in graph['gene_to_markers']]
    out['edges_from'] = np.array(graph['edges_from'], copy=True)
    out['edges_to'] = np.array(graph['edges_to'], copy=True)
    out['active_rows'] = list(graph['active_rows'])
    return out


def _build_bio_graph_uncached(params, available_markers, merge_source, network_cache,
                              model_name, w_fit, prep_fast):
    """The legacy graph construction, moved verbatim out of
    `GAT_biological_prior_knowledge` (ver4-61, R1) - same steps, same messages,
    same errors. See `_prepare_bio_graph` for the contract."""
    network_json_path = params[7]
    gene_location_csv_path = params[8]
    marker_info_path = params[9]
    unit = params[10]
    include_mediated_edges = params[11]
    max_hops = params[12]
    data_driven_merge = params[13]

    with timed_stage('bio:graph'):
        _structure = _graph_structure(
            network_json_path, gene_location_csv_path, marker_info_path, unit,
            include_mediated_edges, max_hops, available_markers, fast=prep_fast,
        )
    gene_list = _structure['gene_list']
    dropped_at_location_lookup = _structure['dropped_at_location_lookup']
    edge_list = _structure['edge_list']
    marker_info = _structure['marker_info']
    gene_to_markers_full = _structure['gene_to_markers_full']

    # Requirement 2, step 1 (continued): a gene row with zero covering SNPs
    # cannot contribute a node feature and is dropped from the *graph* (its
    # position in gene_list is kept as an index so the adjacency edge list -
    # which references those original row indices - can be filtered
    # consistently below).
    active_rows = [i for i, m in enumerate(gene_to_markers_full) if len(m) > 0]
    dropped_rows = [i for i in range(gene_list.shape[0]) if i not in active_rows]
    if dropped_rows:
        print(f"[GAT_biological_prior_knowledge] {len(dropped_rows)} gene row(s) had no SNP "
              f"in the genotype table within their coordinates and were excluded from the "
              f"graph: {gene_list.loc[dropped_rows, 'name'].tolist()}")

    if len(active_rows) == 0:
        raise RuntimeError(
            "None of the genes in the gene list had any SNP falling inside their "
            "coordinates. Check that gene_location_csv_path and marker_info_path use the same "
            "chromosome naming convention (e.g. 'chr1' vs '1') and the same coordinate "
            "unit (bp vs cM)."
        )

    gene_list_active = gene_list.iloc[active_rows].reset_index(drop=True)

    # ver4-61, R1: the active-gene count is recorded so the caller can write the
    # gene-coordinate side file from the GENE rows only (exactly as before the
    # extraction - bare-marker rows appended below are never written to it).
    n_gene_nodes = len(active_rows)

    gene_to_markers = [gene_to_markers_full[i] for i in active_rows]
    n_genes = len(active_rows)

    # Requirement 3 (bare-marker extension) / requirement 5 (this replaces
    # the old orphan_marker_mode ('ignore'/'independent_node') setting
    # entirely): SNPs that fall outside every gene's window. When the
    # data-driven merge feature (data_driven_merge['enabled']) is off - the
    # unchanged default - these are simply left out of the graph, exactly
    # as the old 'ignore' mode always did (their reported effect is still
    # exactly 0 - see `effect` far below). When it's on, requirement 2's
    # side pipeline decides which of them (if any) become their own "bare
    # marker" graph nodes with real, data-driven edges - see
    # Preprocess/data_driven_prior_network.py for the actual selection/
    # interaction-search/merge algorithm; everything below just wires its
    # output into this function's own node list and edge list.
    covered_markers = set()
    for markers in gene_to_markers_full:
        covered_markers.update(markers)
    orphan_markers = [m for m in available_markers if m not in covered_markers]

    if not isinstance(data_driven_merge, dict):
        raise ValueError(
            f"params[13] (data_driven_merge) must be a dict, e.g. {{'enabled': False}}, "
            f"got {data_driven_merge!r} - see this function's own docstring."
        )

    # extra_edges (requirement 3's cross-network merge edges, in the FINAL
    # extended index space) is only ever non-empty when the merge feature is
    # on - spliced into edges_from_list/edges_to_list further below, right
    # after the structural gene-gene adjacency is built, so both kinds of
    # edge go through the exact same self-loop dedup step together.
    extra_edges = []

    if data_driven_merge.get('enabled', False):
        if orphan_markers:
            print(f"[GAT_biological_prior_knowledge] {len(orphan_markers)} marker(s) outside "
                  f"every gene are eligible to become data-driven 'bare marker' nodes "
                  f"(data_driven_merge enabled) - which of them actually do depends on RF "
                  f"selection + surviving pairwise-Shapley interactions below.")

        _merge_source = merge_source
        _merge_train, _merge_valid, _merge_test = _merge_source

        # Requirement 11 (efficiency) / PERFORMANCE FIX: reuse whatever
        # this task has already computed for this instance - see this
        # function's own network_cache docstring entry above - instead of
        # unconditionally redoing LD pruning / RF filtering / the Shapley
        # interaction search every single call. This used to be inlined
        # here; it is now
        # Preprocess.data_driven_prior_network.ensure_bio_prior_merge_cache()
        # (byte-identical logic, just pulled out into its own function) so
        # genomic_prediction.py's GP() can ALSO call it directly, up
        # front in the MAIN process, before dispatching this instance's
        # hyperparameter-tuning trials - see that function's own
        # docstring for exactly why a call from here alone is not enough
        # once trials run in separate worker processes.
        # ver4-25, R8: `_merge_train`'s row order is this task's own
        # `id_train` order regardless of whether `merge_source_data` (the
        # full/unrestricted marker pool) or the frozen `data_train` pool
        # is in use (architecture doc SS8/SS13.4 - both are sliced to the
        # SAME train/valid/test split membership) - so `w_fit` is always
        # expected to align row-for-row. Checked defensively rather than
        # assumed: a mismatch degrades THIS call to unweighted with a
        # NOTE, rather than crashing or silently misaligning weights to
        # individuals.
        _merge_sample_weight = w_fit
        if _merge_sample_weight is not None and len(_merge_sample_weight) != _merge_train.shape[0]:
            print(f"[{model_name}] NOTE: BLUE-reliability weighting is on, but the data-driven "
                  f"merge's own training pool has {_merge_train.shape[0]} row(s) while the shared "
                  f"fitting weight has {len(_merge_sample_weight)} - this should not happen "
                  f"(architecture doc SS8/SS13.4's own alignment guarantee) and needs "
                  f"investigating if it recurs. Falling back to UNWEIGHTED for this call only.")
            _merge_sample_weight = None
        rf_selected_markers, pair_df = ensure_bio_prior_merge_cache(
            model_name, _merge_train, _merge_valid, _merge_test, data_driven_merge, network_cache,
            sample_weight=_merge_sample_weight,
        )

        (bare_marker_names, bare_rows, bare_gene_to_markers,
         extra_edges) = merge_biological_and_data_driven_networks(
            gene_to_markers, pair_df, rf_selected_markers, marker_info)
        if bare_marker_names:
            gene_list_active = pd.concat([gene_list_active, bare_rows], ignore_index=True)
            gene_to_markers = gene_to_markers + bare_gene_to_markers
            n_genes = len(gene_to_markers)
            print(f"[GAT_biological_prior_knowledge] Data-driven merge: {len(bare_marker_names)} "
                  f"bare marker node(s) added, {len(extra_edges)} new edge(s) to/among gene and "
                  f"bare-marker nodes.")
    elif orphan_markers:
        preview = orphan_markers[:10]
        print(f"[GAT_biological_prior_knowledge] {len(orphan_markers)} marker(s) outside "
              f"every gene were ignored (data_driven_merge disabled): {preview}"
              f"{', ...' if len(orphan_markers) > 10 else ''}")

    with timed_stage('bio:graph'):
        max_snp = max(len(m) for m in gene_to_markers)

        # Re-index the adjacency edge list from original gene_list row indices to
        # the compacted 0..n_genes-1 indices used for the graph (requirement 3),
        # dropping any edge that touches a gene row filtered out just above.
        # ver4-33, R1.d: vectorised (same surviving edges, same order) unless
        # BIO_PRIOR_PREP_FAST=False, which keeps the original row loop.
        if prep_fast:
            edges_from_list, edges_to_list = _reindex_edges_fast(
                edge_list, active_rows, gene_list.shape[0])
        else:
            row_to_active_idx = {orig: new for new, orig in enumerate(active_rows)}
            edges_from_list, edges_to_list = [], []
            for _, e in edge_list.iterrows():
                i, j = int(e['gene1_index']), int(e['gene2_index'])
                if i in row_to_active_idx and j in row_to_active_idx:
                    edges_from_list.append(row_to_active_idx[i])
                    edges_to_list.append(row_to_active_idx[j])

        # Splice in the data-driven merge's own edges (requirement 3), computed
        # further up - empty when the merge feature is off. Both edge sources
        # go through the exact same self-loop dedup step immediately below, so
        # a data-driven edge that happens to coincide with a structural one (or
        # with another data-driven pair evaluated twice) is never double-counted.
        edges_from_list.extend(i for i, j in extra_edges)
        edges_to_list.extend(j for i, j in extra_edges)

        # Give EVERY gene/marker node a self-loop (i -> i), not just orphan
        # (zero-edge) nodes. GATv2Conv here is created with add_self_loops=False,
        # so without this a connected node's own feature vector only ever reaches
        # the next layer indirectly, filtered through attention over its
        # neighbours - it can never contribute its own raw signal directly. A
        # self-loop on every node means every node - whether or not it has
        # curated-network neighbours - always has at least its own feature vector
        # available to attend to, on top of whatever real gene-gene edges it has,
        # so no node's predictive information depends entirely on being wired to
        # something else. This generalises the old orphan-only self-loop (which
        # existed purely so isolated nodes weren't dead weight, producing a
        # constant input-independent output with no message to aggregate) into a
        # blanket rule that also benefits well-connected nodes.
        #
        # Dedup against any self-loop the curated network JSON may already encode
        # itself (e.g. a gene autoregulation edge gene_X -> gene_X), so no node
        # ends up attending to itself twice with double weight.
        existing_edges = set(zip(edges_from_list, edges_to_list))
        for idx in range(n_genes):
            if (idx, idx) not in existing_edges:
                edges_from_list.append(idx)
                edges_to_list.append(idx)
                existing_edges.add((idx, idx))

        if len(edges_from_list) == 0:
            raise RuntimeError(
                "After restricting the gene-gene adjacency to genes that have at least one "
                "covering SNP, zero edges remain - the resulting graph would have no attention "
                "structure to learn from. Re-check the adjacency edge list produced by "
                "Preprocess.gene_network_prior.build_gene_adjacency() (consider "
                "include_mediated_edges=True if the network's gene-gene relationships mostly "
                "run through hormone/metabolite mediator nodes, as in the shoot-branching "
                "example network)."
            )

        edges_from = np.array(edges_from_list)
        edges_to = np.array(edges_to_list)

        print(f"[GAT_biological_prior_knowledge] Graph: {n_genes} gene node(s) "
              f"(from {gene_list.shape[0]} candidate row(s) resolved from {network_json_path}), "
              f"{len(edges_from)} directed edge(s), up to {max_snp} SNP(s) aggregated per gene.")


    return {
        'gene_list_active': gene_list_active,
        'gene_to_markers': gene_to_markers,
        'n_genes': n_genes,
        'max_snp': max_snp,
        'edges_from': edges_from,
        'edges_to': edges_to,
        'active_rows': active_rows,
        'n_gene_nodes': n_gene_nodes,
    }


def _prepare_bio_graph(params, available_markers, merge_source, network_cache, model_name,
                       *, w_fit=None, prep_fast=True):
    """Return the bio-prior graph for this task and instance, built at most once.

    Args:
        params: flat bio-prior parameter list (`params[7:14]` define the graph).
        available_markers: marker names of the pool this call models.
        merge_source: `(train, valid, test)` frames for the data-driven merge's
            side pipeline (the full marker pool, or the call's own frames).
        network_cache: the task-owned cache dict, or None (no memo, legacy).
        model_name: the instance name (the `network_cache` entry key).
        w_fit: optional BLUE fitting weights (forwarded to the merge).
        prep_fast: False disables the memo and the `[bio-graph]` log lines.

    Returns:
        dict with keys ``gene_list_active`` (DataFrame: genes then bare-marker
        rows), ``gene_to_markers`` (list of list of str), ``n_genes``,
        ``max_snp``, ``edges_from``, ``edges_to`` (np.ndarray, directed, with
        self-loops), ``active_rows`` (list of int) and ``n_gene_nodes`` (how many
        leading rows of ``gene_list_active`` are real genes). The arrays and
        frames are copies; the caller may not rely on them aliasing the memo.

    Raises:
        Whatever the legacy construction raised (RuntimeError for an empty
        graph, ValueError for a malformed `data_driven_merge`, ...).
    """
    if not prep_fast or network_cache is None:
        return _build_bio_graph_uncached(
            params, available_markers, merge_source, network_cache, model_name, w_fit, prep_fast)

    try:
        key = _bio_graph_key(params, available_markers)
    except (OSError, TypeError, ValueError, IndexError) as exc:
        print(f"[bio-graph] NOTE: {model_name}: graph memo skipped ({type(exc).__name__}: {exc}) - "
              f"rebuilding the graph for this call.")
        return _build_bio_graph_uncached(
            params, available_markers, merge_source, network_cache, model_name, w_fit, prep_fast)

    entry = network_cache.setdefault(model_name, {})
    stored = entry.get('graph')
    if isinstance(stored, dict) and stored.get('key') == key and 'value' in stored:
        value = stored['value']
        print(f"[bio-graph] reused {model_name}: G={value['n_genes']} node(s), "
              f"{len(value['edges_from'])} directed edge(s) (built earlier this task).")
        return _copy_bio_graph(value)
    if stored is not None:
        print(f"[bio-graph] NOTE: {model_name}: the memoised graph does not match this call's "
              f"inputs (key mismatch) - recomputing it.")

    started = time.perf_counter()
    graph = _build_bio_graph_uncached(
        params, available_markers, merge_source, network_cache, model_name, w_fit, prep_fast)
    entry['graph'] = {'key': key, 'value': _copy_bio_graph(graph)}
    print(f"[bio-graph] built {model_name}: G={graph['n_genes']} node(s), "
          f"{len(graph['edges_from'])} directed edge(s) in {time.perf_counter() - started:.1f}s "
          f"(memoised for the rest of this task).")
    return graph


def prewarm_bio_graph(model_name, params, train, merge_source, network_cache, w_fit=None,
                      *, valid=None, test=None):
    """Build and memoise the bio-prior graph in the calling (parent) process.

    Why: a graph first built inside a loky worker never reaches the parent, so
    every trial would rebuild it. Calling this before the ModelTrialRunner is
    pickled puts the graph into `network_cache`, which the runner carries by
    value (the same mechanism `ensure_bio_prior_merge_cache` uses).

    Args:
        model_name: instance name (`network_cache` entry key).
        params: flat bio-prior parameter list for this task.
        train: the training frame whose columns define the marker pool.
        merge_source: `(train, valid, test)` of the merge's marker universe, or
            None to use `(train, valid, test)` exactly as the model call does.
        network_cache: the task-owned cache dict; None makes this a no-op.
        w_fit: optional BLUE fitting weights.
        valid, test: the call's own validation/test frames, used only when
            `merge_source` is None (the model falls back to them too).

    Returns:
        The graph dict, or None when pre-warming is off or failed (the trials
        then build it themselves at the legacy cost; the reason is logged).
    """
    prep_fast = bool(get_active_compute_resources().get('accel_bio_prior_prep_fast', True))
    if not prep_fast:
        print(f"[bio-graph] {model_name}: BIO_PRIOR_PREP_FAST is off - the graph is rebuilt on "
              f"every call (legacy behaviour).")
        return None
    if network_cache is None:
        return None
    if merge_source is None:
        merge_source = (
            train,
            valid if valid is not None else train.iloc[0:0],
            test if test is not None else train,
        )
    try:
        return _prepare_bio_graph(
            params, train.columns[:-1], merge_source, network_cache, model_name,
            w_fit=w_fit, prep_fast=True)
    except (RuntimeError, ValueError, OSError, KeyError, TypeError) as exc:
        print(f"[bio-graph] NOTE: {model_name}: pre-warm failed ({type(exc).__name__}: {exc}) - "
              f"each trial will build the graph itself (slower, results unchanged).")
        return None


@gat_stable('GAT_biological_prior_knowledge')
def GAT_biological_prior_knowledge(data_train, data_valid, data_test, params, RESULT_NAME=None, PHENOTYPE_NAME=None, MODEL_NAME='GAT_biological_prior_knowledge', merge_source_data=None, network_cache=None, *, sample_weight_ctx=None):
    """Biological prior-knowledge GAT.

    Same overall modelling / training / explanation flow as
    models/GAT_prior_knowledge.py (requirement 4), but the graph structure
    comes from a curated, external gene-interaction network - a FlashP-style
    JSON file (uploaded by a user, or generated by
    Preprocess.flash_p_integration.run_flash_p()) - instead of being learned
    from an RF+SHAP interaction search on the training fold. The gene list
    and gene-gene adjacency (requirement 1) are built automatically, every
    time this function runs, straight from that JSON file plus a curated
    gene-location lookup CSV (via Preprocess.gene_network_prior's
    extract_candidate_genes/build_gene_list/build_gene_adjacency) - there is
    no separate, pre-built gene_list.csv/adjacency.csv file to manage. Every
    graph node here is a *gene* (aggregating however many SNPs from the
    genotype table fall inside that gene's coordinates), not a single SNP
    (requirement 2), and the edges between gene nodes are exactly the
    (possibly re-indexed/filtered) edges of that freshly-built adjacency
    (requirement 3).

    Rebuilding the gene list/adjacency on every call is cheap (it's plain
    interval-overlap and graph-traversal bookkeeping over an already-curated
    lookup table, not a web lookup or a model fit), so doing it fresh each
    time - rather than caching a pre-built pair of CSVs - keeps this
    function's inputs to exactly what a user actually has in hand (the
    network JSON and a gene-location table), with one less
    build-then-remember-to-pass-the-right-file step.

    This function always receives the full, unpruned/unfiltered genotype
    data as data_train/data_valid/data_test - genomic_prediction.py's
    dispatch code never routes it the LD-pruned/RF-importance-filtered pool
    other models might be using, regardless of whether either of those is
    enabled. There is deliberately no toggle here to change that: this
    model always determines its own candidate markers from the gene
    network (via marker_info_path, unit, etc. below). If you instead want
    one or more *other* selected models to be restricted to this same
    gene-network marker set (rather than their usual full/LD-pruned/
    RF-filtered pool), that is a separate, model-independent choice - see
    GP()'s own OTHER_MODELS_MARKER_SOURCE parameter in
    genomic_prediction.py, not anything in this params list.

    The one exemption to "always the full, unpruned/unfiltered data" is the
    data-driven prior-network merge feature (params[13] below,
    requirement 2): when enabled, an internal LD-pruning-then-RF-filtering
    side pipeline runs (on `merge_source_data` if given - see below -
    otherwise on data_train/data_valid/data_test themselves) purely to
    decide which markers feed a second, data-driven interaction network
    that gets merged into the biological one (requirement 3). This side
    pipeline's own output never replaces what data_train/data_valid/
    data_test actually contribute as node features - every gene node still
    aggregates its FULL covering-marker set exactly as when the merge
    feature is off (see module-level design note below for why gene
    coverage must stay based on the full marker set, not the RF-narrowed
    one) - it only ever adds new, RF-selected "bare marker" nodes and their
    data-driven edges on top.

    Parameters
    ----------
    data_train, data_valid, data_test : pd.DataFrame
        Same convention as every other model in this codebase: all columns
        except the last are genomic markers (SNPs), the last column is the
        target phenotype. data_valid may be an empty DataFrame (no
        validation split requested for this task).
    params : list
        [neuron, dropout, lrate, decay, epoch, bsize, heads,
         network_json_path, gene_location_csv_path, marker_info_path,
         unit, include_mediated_edges, max_hops,
         data_driven_merge,
         marker_effect, samples]

        neuron                  : GATv2Conv hidden width
        dropout                 : GATv2Conv dropout
        lrate                   : Adam learning rate
        decay                   : Adam weight decay
        epoch                   : number of training epochs
        bsize                   : training/validation batch size
        heads                   : number of attention heads
        network_json_path       : path to the gene-interaction network JSON
                                   (uploaded by a user, or written by
                                   Preprocess.flash_p_integration.run_flash_p())
        gene_location_csv_path  : curated gene-location lookup CSV (columns:
                                   Gene_Name, Chromosome, Start_bp/Start_cM,
                                   End_bp/End_cM, and optionally
                                   AGI_Locus_ID, Source - see
                                   Preprocess.gene_network_prior.build_gene_list
                                   for the accepted column aliases), used
                                   together with network_json_path to build
                                   the gene list and gene-gene adjacency
        marker_info_path        : CSV with columns chromosome, name, start,
                                   end - same structure as
                                   Data/MaizeNAM/marker_info.csv
        unit                    : 'bp' or 'cM' - must match both
                                   gene_location_csv_path and
                                   marker_info_path's coordinate units
        include_mediated_edges  : bool - passed straight to
                                   Preprocess.gene_network_prior.build_gene_adjacency;
                                   see that function's docstring
        max_hops                : int - passed straight to
                                   build_gene_adjacency (ignored when
                                   include_mediated_edges is False)
        data_driven_merge         : dict, e.g. {'enabled': False} (default -
                                   no change from the original behaviour:
                                   SNPs outside every gene's window simply
                                   never become graph nodes, their effect
                                   still reported as exactly 0 - see
                                   `effect` below), or, when enabled
                                   (requirement 2):
                                   {'enabled': True,
                                    'top_rate': float in (0, 100],
                                    'rf_filter': dict (RF_marker_filtering-
                                        shaped config - which markers are
                                        eligible to become "bare" nodes),
                                    'ld_prune': dict or None (optional LD
                                        pruning applied before RF selection,
                                        in the side pipeline only),
                                    'shap_n_estimators_override': int or
                                        None}
                                   An RF-selected marker outside every gene
                                   becomes its own "bare" graph node with
                                   real, data-driven pairwise-Shapley edges
                                   to gene/other bare-marker nodes
                                   (requirement 3), instead of being
                                   excluded - see
                                   Preprocess/data_driven_prior_network.py.
                                   This is the requirement-5 upgrade of, and
                                   replacement for, the old
                                   orphan_marker_mode ('ignore'/
                                   'independent_node') setting, which no
                                   longer exists.
        marker_effect           : bool, whether to run the
                                   Integrated-Gradients marker-effect
                                   explainer (as in every other GAT model in
                                   this codebase)
        samples                 : number of test individuals to average the
                                   Integrated-Gradients explanation over
                                   ('all' is resolved to test.shape[0] by
                                   genomic_prediction.py before this function
                                   runs, the same convention used for
                                   GAT_infinitesimal / GAT_fully_connected)
    RESULT_NAME, PHENOTYPE_NAME : str, optional
        Not part of `params` - passed as separate trailing arguments, the
        same way rrBLUP/GBLUP/BayesB/RKHS receive RESULT_NAME, since (like
        those R-backed models) this function writes a side file to
        './Result/<RESULT_NAME>/'. Specifically: the coordinates of the
        genes actually used as graph nodes this call (chromosome, name,
        start, end - same schema as marker_info.csv) are written to
        './Result/<RESULT_NAME>/<MODEL_NAME>_gene_coordinates_<PHENOTYPE_NAME>.csv'.
        This is what lets circos_plot.py resolve this model's attention
        edges - whose marker1/marker2 values are GENE names, not SNP names -
        to genomic positions the same way it already resolves every other
        model's SNP-named markers via marker_info.csv (see
        circos_plot.py's _load_combined_marker_info()). RESULT_NAME/
        PHENOTYPE_NAME both default to None, in which case this side file is
        simply not written (e.g. for standalone use/tests outside the full
        GP() pipeline, where no Result folder is expected to exist) -
        everything else about the function's behaviour is unaffected either
        way.
    MODEL_NAME : str, default 'GAT_biological_prior_knowledge'
        This call's own model-instance name - genomic_prediction.py's GP()
        supports more than one independent GAT_biological_prior_knowledge in
        the same run (e.g. 'GAT_biological_prior_knowledge_2', one per
        FlashP sample or uploaded network a user wants to compare), each
        with its own HPARAMETERS entry; MODEL_NAME is simply whichever of
        those names this particular call is for, used only to keep each
        instance's own coordinate side file (above) from overwriting
        another instance's. Not used for anything else inside this
        function.
    merge_source_data : (pd.DataFrame, pd.DataFrame, pd.DataFrame) or None
        Only meaningful when params[13]['enabled'] is True (requirement 2).
        The TRUE, full/unrestricted-marker genotype pool
        (train, valid, test) for this task, used as the data-driven merge
        feature's OWN internal LD-pruning-then-RF-filtering side pipeline
        input - separately from data_train/data_valid/data_test themselves
        (see the module docstring above for why gene coverage must stay
        based on data_train/data_valid/data_test's full marker set
        regardless). Defaults to None, in which case data_train/
        data_valid/data_test are used for the side pipeline too - correct
        whenever data_train IS already the true full marker set (every CSV
        genotype format run, and every PLINK run where nothing has reason
        to narrow it) - genomic_prediction.py only ever passes something
        different here for PLINK-format runs where data_train/data_valid/
        data_test have already been narrowed to this instance's own
        gene-window markers (see GP()'s bio_prior_pools), which would
        otherwise silently hide every "bare" (outside-every-gene) marker
        from the side pipeline entirely.
    network_cache : dict or None
        Requirement 11 (efficiency): a plain dict, owned and created ONCE
        PER PREDICTION TASK by genomic_prediction.py's GP() (never shared
        across tasks/populations/phenotypes/replicates - see its own
        `_bio_prior_merge_cache`), used to avoid redoing the data-driven
        merge's own expensive LD-pruning / RF-filtering / pairwise-
        Shapley-interaction pipeline more than once per task - in
        particular across repeated calls for THIS SAME instance within
        one task, which happens whenever hyperparameter tuning is enabled
        (one call per trial, plus one final confirmatory fit - all on the
        exact same data and data_driven_merge config, so the resulting
        network structure is IDENTICAL every time; only this function's
        own GAT architecture/training hyperparameters differ between
        calls) and whenever GP()'s own OTHER_MODELS_MARKER_SOURCE=
        'gene_network_plus_rf' setting already computed (some of) the
        same result for a different purpose (restricting other models'
        marker pool).

        Keyed by MODEL_NAME (so multiple independent
        GAT_biological_prior_knowledge instances in the same task/run
        each get their own entry, never mixed up), each entry a dict that
        fills in progressively as more of the pipeline runs:
            'pruned_markers'      : marker names surviving the merge's own
                                     LD pruning (only present if
                                     data_driven_merge['ld_prune'] is set)
            'rf_selected_markers' : RF-selected marker names (the
                                     expensive-to-compute output of LD
                                     pruning + RF filtering)
            'pair_df'              : the pairwise-Shapley-interaction
                                     DataFrame (the expensive-to-compute
                                     output of the interaction search)
        Whatever keys are ALREADY present (written by a previous call to
        this function within the same task, or by GP() itself before
        this model ever ran) are reused as-is instead of being
        recomputed; whatever's missing is computed once here and written
        back for the next caller to reuse. None (the default) disables
        caching entirely - every call recomputes everything from
        scratch, exactly as before this parameter existed (fully
        backward compatible for any caller that doesn't pass it).

    Returns
    -------
    r, mse, effect, predicted_test, predicted_valid, predicted_train, attention
        Exactly the same 7-tuple contract as GAT_prior_knowledge() /
        GAT_fully_connected(). In particular, `effect` has exactly one column
        per column of data_train.iloc[:, :-1] - i.e. every *original* SNP,
        not just the ones that fell inside a gene - because
        genomic_prediction.py enforces that every model's marker-effect
        table has one column per marker in the input genotype table. A
        gene's Integrated-Gradients attribution is therefore broadcast back
        onto every SNP that was aggregated into that gene's node, and SNPs
        that were not covered by any gene in the network - and so never
        influenced the model at all - are reported with an effect of exactly
        0, rather than being silently omitted. `attention`, unlike `effect`,
        keeps GENE names (see RESULT_NAME/PHENOTYPE_NAME above for how those
        get resolved to coordinates downstream).
    """

    # ver4-25, R8 (design blueprint SS2, Stage 6): `_w_fit` weights this
    # model's own GAT training loop (per-graph `tmp.w`, further down -
    # see models/GAT_infinitesimal.py's identical mechanism) AND its
    # internal data-driven-merge RF (params[13], below - blueprint SS2's
    # own mechanism table: "GAT_biological_prior_knowledge data-driven
    # merge RF | sample_weight=w_fit"). `None` (every pre-ver4-25 caller)
    # is byte-identical (I11). TEST/VALID scoring for this model is
    # unaffected either way (computed centrally by the caller).
    _w_fit = sample_weight_ctx.w_fit if sample_weight_ctx is not None else None

    if data_valid.shape[0] != 0:
        VALID = True
    else:
        VALID = False

    neuron = params[0]
    dropout = params[1]
    lrate = params[2]
    decay = params[3]
    epoch = params[4]
    bsize = params[5]
    heads = params[6]
    # params[7:13] (network JSON, gene-location CSV, marker info, unit, mediated edges, max
    # hops) are read by `_prepare_bio_graph` / `_bio_graph_key` from the same list (ver4-61,
    # R1), so they are no longer unpacked here.
    data_driven_merge = params[13]
    marker_effect = params[14]
    samples = params[15]
    # ver4-61, R4: field 16 `Early stopping patience (epochs, 0 = off)` was appended after
    # the original 16 fields. Older configs and per-phenotype dict entries are not padded
    # by GP() for this model, so a short list means 0 (= off, the legacy behaviour).
    _es_field = int(params[16]) if len(params) > 16 and params[16] else 0

    # =========================================================================
    # 1. Build the gene list + gene-gene adjacency (requirement 1) directly
    #    from the network JSON and the curated gene-location CSV, load the
    #    SNP position table, and build the gene <-> SNP mapping (requirement
    #    2, step 1). Nothing here is read from a separately pre-built
    #    gene_list.csv/adjacency.csv - both are derived fresh, every call.
    # =========================================================================

    # ver4-33, R1.d: the whole "network JSON -> gene list -> adjacency -> marker
    # table -> gene<->marker mapping" chain (unchanged in content, errors and
    # messages) now lives in `_graph_structure`, which memoises it per process
    # - none of it depends on the training values. With
    # BIO_PRIOR_PREP_FAST=False it is rebuilt every call exactly as before.
    # Requirement 8 (unify marker_info columns by position) is applied inside.
    available_markers = data_train.columns[:-1]  # every SNP originally in the genotype table
    _prep_fast = bool(get_active_compute_resources().get('accel_bio_prior_prep_fast', True))
    # ver4-61, R1: graph construction (structure -> data-driven merge -> edge
    # re-index -> self-loops) lives in `_prepare_bio_graph`, memoised per task
    # and instance in `network_cache`; with BIO_PRIOR_PREP_FAST=False or
    # network_cache=None it is rebuilt every call exactly as before.
    _merge_source = (merge_source_data if merge_source_data is not None
                     else (data_train, data_valid, data_test))
    _graph = _prepare_bio_graph(
        params, available_markers, _merge_source, network_cache, MODEL_NAME,
        w_fit=_w_fit, prep_fast=_prep_fast,
    )
    gene_list_active = _graph['gene_list_active']
    gene_to_markers = _graph['gene_to_markers']
    n_genes = _graph['n_genes']
    max_snp = _graph['max_snp']
    edges_from = _graph['edges_from']
    edges_to = _graph['edges_to']

    # Persist the active genes' own coordinates, in the exact same schema as
    # marker_info.csv (chromosome, name, start, end) - see the
    # RESULT_NAME/PHENOTYPE_NAME docstring entry above for why. Named after
    # MODEL_NAME (this call's own model-instance name, e.g.
    # 'GAT_biological_prior_knowledge_2') rather than a fixed name, so
    # multiple bio-prior instances in the same run - each with a different
    # network - never overwrite each other's coordinate files; circos_plot.py
    # already discovers these by filename pattern
    # ('*_gene_coordinates_<phenotype>.csv'), not by a hardcoded model name.
    # Only the GENE rows are written (bare-marker rows follow them in
    # `gene_list_active` and were never part of this file).
    if RESULT_NAME is not None and PHENOTYPE_NAME is not None:
        coord_dir = result_dir_path(RESULT_NAME)
        os.makedirs(coord_dir, exist_ok=True)
        coord_path = os.path.join(
            coord_dir, f'{MODEL_NAME}_gene_coordinates_{PHENOTYPE_NAME}.csv'
        )
        gene_list_active.iloc[:_graph['n_gene_nodes']][
            ['chromosome', 'name', 'start', 'end']
        ].to_csv(coord_path, index=False, encoding='utf-8')

    # =========================================================================
    # 2. Build the per-individual node-feature tensors (requirement 2, steps
    #    2-3): each gene's covering SNP values (zero-padded to max_snp),
    #    plus one-hot dummy columns identifying which gene each node is - the
    #    same principle as lines 70-73 of GAT_prior_knowledge.py, generalised
    #    from "1 raw SNP value per node" to "up to max_snp raw SNP values per
    #    node".
    # =========================================================================

    if VALID:
        data_QTL_train, data_QTL_valid, data_QTL_test = (
            data_train.iloc[:, :-1].reset_index(drop=True),
            data_valid.iloc[:, :-1].reset_index(drop=True),
            data_test.iloc[:, :-1].reset_index(drop=True),
        )
        data_pheno_train, data_pheno_valid, data_pheno_test = (
            data_train.iloc[:, -1].reset_index(drop=True),
            data_valid.iloc[:, -1].reset_index(drop=True),
            data_test.iloc[:, -1].reset_index(drop=True),
        )
    else:
        data_QTL_train, data_QTL_test = (
            data_train.iloc[:, :-1].reset_index(drop=True),
            data_test.iloc[:, :-1].reset_index(drop=True),
        )
        data_pheno_train, data_pheno_test = (
            data_train.iloc[:, -1].reset_index(drop=True),
            data_test.iloc[:, -1].reset_index(drop=True),
        )

    # One dummy (one-hot) row per gene - identical in spirit to lines 70-73 of
    # GAT_prior_knowledge.py, just built directly at gene-node granularity
    # instead of via a melt of a per-marker table (see `_build_dummy_matrix`
    # for why float dummies, and for the np.eye fast form). ver4-35, R1.c/R1.b:
    # built lazily - the factorised dense path never needs it, so it is only
    # created when the legacy body, the legacy explainer or a fallback does.
    _dummy_holder = []

    def _get_dummy_matrix():
        if not _dummy_holder:
            _dummy_holder.append(_build_dummy_matrix(n_genes, fast=_prep_fast))
        return _dummy_holder[0]

    # ver4-33, R1.f / R1.g switches (ACCELERATION.gat_dense_batching /
    # .gat_batched_ig, engine key GAT_IG_INTERNAL_BATCH): both default OFF -
    # the legacy per-graph path below is then the one that runs. Logged on
    # every call, including the negative state.
    _accel = get_active_compute_resources()
    _dense_on = bool(_accel.get('accel_gat_dense_batching', False))
    _ig_batched_on = bool(_accel.get('accel_gat_batched_ig', False))
    _ig_internal_batch = int(_accel.get('accel_gat_ig_internal_batch', 256))
    print(f"[GAT_biological_prior_knowledge] Acceleration: gat_dense_batching={_dense_on}, "
          f"gat_batched_ig={_ig_batched_on}"
          f"{f' (IG internal batch {_ig_internal_batch})' if _ig_batched_on else ''}.")

    # ver4-35 switches (R1.b / R1.d / R1.e / R1.h1). All default to the
    # ver4-33 behaviour (I11); logged on every call, including the negative
    # state, with the reason a switch that is on cannot take effect.
    _factorised_on = bool(_accel.get('accel_gat_factorised_input', False))
    _sync_free_on = bool(_accel.get('accel_gat_sync_free', False))
    _device_policy = str(_accel.get('accel_torch_device_policy', 'legacy'))
    _cost_guard = str(_accel.get('accel_bio_prior_cost_guard', 'warn'))
    print(f"[GAT_biological_prior_knowledge] Acceleration: "
          f"gat_factorised_input={_factorised_on}, gat_sync_free={_sync_free_on}, "
          f"torch_device_policy={_device_policy}, bio_prior_cost_guard={_cost_guard}.")
    if _factorised_on and not _dense_on:
        print("[GAT_biological_prior_knowledge] NOTE: gat_factorised_input needs "
              "gat_dense_batching, which is off - the one-hot input is used (results unaffected).")
    if _sync_free_on and not _dense_on:
        print("[GAT_biological_prior_knowledge] NOTE: gat_sync_free needs gat_dense_batching, "
              "which is off - the standard training step is used (results unaffected).")

    # ver4-35, R1.a / R1.e / R1.h1: the fit's shape, the cost guard and the
    # device placement. Everything here is a pure function of the config and
    # the data shape (never of a timer or a queue), so every tuning trial and
    # the confirmatory fit of this model take the same decision (I9).
    # E0 is the upper bound of the undirected edge count (each non-self-loop
    # edge mirrored); the exact count is printed with the fit.
    _factorised_effective = _factorised_on and _dense_on
    _n_train_rows = data_QTL_train.shape[0]
    _n_eval_rows = data_QTL_test.shape[0] + (data_QTL_valid.shape[0] if VALID else 0)
    _shape = {
        'G': n_genes, 'S': max_snp, 'N_train': _n_train_rows, 'N_eval': _n_eval_rows,
        'E0': 2 * (len(edges_from) - n_genes) + n_genes, 'B': int(bsize),
        'HC': int(heads) * int(neuron), 'epochs': int(epoch),
        'factorised': _factorised_effective, 'ig_internal_batch': _ig_internal_batch,
        'F': max_snp if _factorised_effective else max_snp + n_genes,
    }
    if _dense_on:
        _host_feature_bytes = 4.0 * (_n_train_rows + _n_eval_rows) * n_genes * max_snp
    else:
        _host_feature_bytes = (_n_train_rows + _n_eval_rows) * (
            4.0 * n_genes * (max_snp + n_genes) + 8.0 * n_genes * max_snp)
    _bio_prior_cost_guard(MODEL_NAME, _shape, _host_feature_bytes,
                          bool(data_driven_merge.get('enabled', False)), _accel)
    _placed_device = resolve_model_device(_shape, _accel)
    print(f"[device] {MODEL_NAME}: torch_device_policy={_device_policy}, compute device "
          f"{_accel.get('device', 'cpu')}, this fit runs on {_placed_device}.")

    # float32 node features are exactly what the legacy `.to(torch.float)`
    # produces from the float64 array (same per-element rounding), at half
    # the host memory; the legacy path keeps its float64 build.
    _pad_dtype = np.float32 if _dense_on else float
    with timed_stage('bio:features'):
        padded_train = _build_padded_node_features(
            data_QTL_train, gene_to_markers, max_snp, dtype=_pad_dtype, fast=_prep_fast)
        if VALID:
            padded_valid = _build_padded_node_features(
                data_QTL_valid, gene_to_markers, max_snp, dtype=_pad_dtype, fast=_prep_fast)
        padded_test = _build_padded_node_features(
            data_QTL_test, gene_to_markers, max_snp, dtype=_pad_dtype, fast=_prep_fast)

    # =========================================================================
    # 3. Create one graph per individual (requirement 3), mirroring lines
    #    89-126 of GAT_prior_knowledge.py: same per-individual loop structure,
    #    same *shared* edge_index re-used for every individual (the
    #    biological prior network is fixed across individuals, exactly like
    #    the RF/SHAP-derived edges in GAT_prior_knowledge.py) - only the
    #    node-feature construction differs, per requirement 2.
    # =========================================================================

    data_train_edge_index = torch.stack([torch.from_numpy(edges_from).to(torch.long),
                                          torch.from_numpy(edges_to).to(torch.long)], dim=0)

    # ver4-4 R8 (blueprint §2.8): T.ToUndirected() only ever depends on
    # edge_index (and edge_attr, which this graph never has) - never on
    # x/y - and, per the comment above, every individual in every split
    # shares the exact SAME edge_index. The ORIGINAL code called
    # `T.ToUndirected()(tmp)` inside each per-individual loop below,
    # re-deriving and re-deduplicating the identical set of reverse edges
    # from scratch once per individual for no reason - N-1 wasted,
    # identical transform calls per split. Applying it ONCE here, to a
    # throwaway Data holding just this shared edge_index, and re-using
    # the resulting undirected edge_index for every individual removes
    # that redundant work without changing a single edge value (verified
    # against the R8 design record's acceptance criteria: identical
    # edge_name_from/edge_name_to and Attention.csv content). `.clone()`
    # per individual is kept - see the module-level R8 note in
    # GAT_fully_connected.py for why outright tensor SHARING across Data
    # instances is deferred pending the ver4-4 blueprint's PC-4 pre-check
    # (not executable in this build environment; see the Change Summary).
    _undirected_edge_index = T.ToUndirected()(Data(edge_index=data_train_edge_index)).edge_index

    # ver4-61, R3/R4: where the early-stopping patience comes from. A hyperparameter-SEARCH
    # fit takes it from the search-fit context (HP_TUNE_FAST['gat_search_early_stop']);
    # every other fit (untuned, confirmatory, final) takes field 16. 0 / absent = off, the
    # legacy behaviour. The resolved value is logged on every call (I11).
    _search_ctx = search_fit_cfg()
    if _search_ctx is not None:
        _es_patience = int(_search_ctx.get('early_stop_patience', 0) or 0)
        _es_source = "HP_TUNE_FAST['gat_search_early_stop']"
    else:
        _es_patience = _es_field
        _es_source = 'hyperparameter field 16'
    _early_stop = ({'patience': _es_patience, 'min_delta': 1e-4, 'source': _es_source}
                   if _es_patience > 0 else None)
    _deadline_on = bool(_search_ctx and _search_ctx.get('trial_deadline_s') is not None)
    print(f"[GAT_biological_prior_knowledge] Early stopping: "
          f"{'patience=%d (%s)' % (_es_patience, _es_source) if _early_stop else 'off'}; "
          f"per-fit time cap: {'on' if _deadline_on else 'off'}.")

    # ver4-33, R1.f: dense-batched fit/evaluation instead of the per-graph
    # `Data` lists + DataLoaders below. Returns the finished 7-tuple, or None
    # (self-test failed / any failure inside the dense machinery - already
    # logged) in which case the unchanged legacy body below runs.
    if _dense_on:
        _dense_result = _try_dense_gat(
            padded={'train': padded_train, 'valid': padded_valid if VALID else None, 'test': padded_test},
            pheno={'train': data_pheno_train.to_numpy(),
                   'valid': data_pheno_valid.to_numpy() if VALID else None,
                   'test': data_pheno_test.to_numpy()},
            w_fit=_w_fit, dummy_matrix=None if _factorised_on else _get_dummy_matrix(),
            edge_index=_undirected_edge_index, n_genes=n_genes,
            gene_names_active=gene_list_active['name'].to_numpy(), gene_to_markers=gene_to_markers,
            effect_columns=data_QTL_test.columns, neuron=neuron, dropout=dropout, lrate=lrate, decay=decay,
            epochs=epoch, bsize=bsize, heads=heads, valid=VALID, marker_effect=(marker_effect == True),
            samples=samples, ig_batched=_ig_batched_on, ig_internal_batch=_ig_internal_batch,
            prep_fast=_prep_fast, factorised=_factorised_on, sync_free=_sync_free_on,
            device_name=_placed_device, model_name=MODEL_NAME,
            shape_info=dict(_shape, E0=int(_undirected_edge_index.shape[1])),
            early_stop=_early_stop,
        )
        if _dense_result is not None:
            return _dense_result
    if _early_stop is not None or _deadline_on:
        _note_once('es-legacy', "early stopping and the per-fit time cap are implemented in the "
                   "dense-batched path only (ACCELERATION gat_dense_batching); this fit runs "
                   "the legacy per-graph path, which trains every epoch.")

    dummy_matrix = _get_dummy_matrix()   # ver4-35, R1.c: built lazily, see above
    with timed_stage('bio:features'):
        data_train = []
        for kk in range(data_pheno_train.shape[0]):
            tmp = Data()
            node_features_tmp = np.concatenate([padded_train[kk], dummy_matrix], axis=1)
            data_pheno_train_tmp = np.expand_dims(np.array(data_pheno_train[kk]), axis=0)
            tmp.x = torch.from_numpy(node_features_tmp).to(torch.float)
            tmp.y = torch.from_numpy(data_pheno_train_tmp).to(torch.float)
            if _w_fit is not None:
                # ver4-25, R8: built exactly like `tmp.y` above - see
                # models/GAT_infinitesimal.py's identical comment.
                tmp.w = torch.from_numpy(
                    np.expand_dims(np.array(_w_fit[kk], dtype=float), axis=0)).to(torch.float)
            tmp.edge_index = _undirected_edge_index.clone()
            data_train += [tmp]

        if VALID:
            data_valid = []
            for kk in range(data_pheno_valid.shape[0]):
                tmp = Data()
                node_features_tmp = np.concatenate([padded_valid[kk], dummy_matrix], axis=1)
                data_pheno_valid_tmp = np.expand_dims(np.array(data_pheno_valid[kk]), axis=0)
                tmp.x = torch.from_numpy(node_features_tmp).to(torch.float)
                tmp.y = torch.from_numpy(data_pheno_valid_tmp).to(torch.float)
                tmp.edge_index = _undirected_edge_index.clone()
                data_valid += [tmp]

        data_test = []
        for kk in range(data_pheno_test.shape[0]):
            tmp = Data()
            node_features_tmp = np.concatenate([padded_test[kk], dummy_matrix], axis=1)
            data_pheno_test_tmp = np.expand_dims(np.array(data_pheno_test[kk]), axis=0)
            tmp.x = torch.from_numpy(node_features_tmp).to(torch.float)
            tmp.y = torch.from_numpy(data_pheno_test_tmp).to(torch.float)
            tmp.edge_index = _undirected_edge_index.clone()
            data_test += [tmp]

    # edge_name_from/to (for the attention output below) use gene names, the
    # node identity at this graph's granularity - the direct analogue of how
    # GAT_prior_knowledge.py reads SNP names off data_test_columns using the
    # same edge_index. A defensive index pick (min(1, len-1)) replaces
    # GAT_prior_knowledge.py's hard-coded data_test[1], purely so a
    # single-individual test split doesn't crash here; every test graph
    # shares the identical edge_index, so the choice of which graph's
    # edge_index is read off makes no difference to the result.
    gene_names_active = gene_list_active['name'].to_numpy()
    _example_idx = min(1, len(data_test) - 1)
    edge_name_from = list(gene_names_active[data_test[_example_idx].edge_index[0].tolist()])
    edge_name_to = list(gene_names_active[data_test[_example_idx].edge_index[1].tolist()])

    # =========================================================================
    # 4. From here on, the flow is identical to GAT_prior_knowledge.py
    #    (requirement 4): same GAT architecture, training loop, metrics,
    #    Integrated-Gradients explainer and attention extraction.
    # =========================================================================

    ## Create GAT
    class GAT(torch.nn.Module):
        def __init__(self, hidden_channels, out_channels, dpout):
            super().__init__()

            self.conv1 = GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, heads=heads, concat=True, dropout=dpout)
            self.conv2 = GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, heads=heads, concat=False, dropout=dpout)
            self.lin1 = torch.nn.Linear(hidden_channels, out_channels)

        def forward(self, x, edge_index, batch, return_attention):
            x, edge_index, batch = x, edge_index, batch
            x = self.conv1(x, edge_index)
            x = F.elu(x)
            if return_attention:
                x, attention = self.conv2(x, edge_index, return_attention_weights=return_attention)
            else:
                x = self.conv2(x, edge_index, return_attention_weights=return_attention)
            x = F.elu(x)
            x = global_mean_pool(x, batch)
            x = self.lin1(x)

            if return_attention:
                return x, attention
            else:
                return x

    model = GAT(hidden_channels=neuron, out_channels=1, dpout=dropout)

    # ver4-4 R4.b/c/d/e: device/AMP/loader-hygiene settings resolved
    # BEFORE the loaders (moved up from their original position) - see
    # GAT_fully_connected.py's identical note for the full rationale.
    # ver4-35, R1.e4: `_placed_device` is the device the placement policy chose
    # for this fit (the resources' own device when torch_device_policy is
    # 'legacy', i.e. exactly the ver4-33 behaviour); a CPU placement also gets
    # the CPU thread settings.
    _resources = dict(get_active_compute_resources(), device=_placed_device)
    device = torch.device(_resources['device'])
    apply_torch_compute_settings(_resources)
    use_amp = bool(_resources['use_amp']) and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    _num_workers = dataloader_num_workers(_resources)
    _pin_memory = device.type == 'cuda'
    _eval_batch = torch_eval_batch_size(_resources, _resources.get('gpu_eval_batch'))

    train_loader = DataLoader(data_train,
                             shuffle=True,
                             batch_size=bsize,
                             num_workers=_num_workers, pin_memory=_pin_memory)
    if VALID:
        valid_loader = DataLoader(data_valid,
                                 batch_size=bsize,
                                 num_workers=_num_workers, pin_memory=_pin_memory)
    test_loader = DataLoader(data_test,
                             batch_size=_eval_batch,
                             num_workers=_num_workers, pin_memory=_pin_memory)

    with gpu_slot(device):
        model.to(device)
        note_cuda_context(device)

        ## Train GAT
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lrate, weight_decay=decay)

        _train_started = time.perf_counter()
        with timed_stage('bio:train'):
            for epoch in range(epoch):
                loss_train_sum = 0
                batch_size = len(train_loader)

                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    with torch.autocast(device_type=device.type, enabled=use_amp):
                        out = model(batch.x, batch.edge_index, batch.batch, None)
                        if _w_fit is None:
                            loss = F.mse_loss(torch.squeeze(out), batch.y)
                        else:
                            # ver4-25, R8: Sum(w_i*r_i^2)/Sum(w_i) - see
                            # models/GAT_infinitesimal.py's identical comment.
                            _sq_err = (torch.squeeze(out) - batch.y) ** 2
                            loss = (_sq_err * batch.w).sum() / batch.w.sum().clamp_min(1e-12)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    loss_train_sum += loss.detach()

                print(f'Epoch {epoch:>3} | Train Loss: {loss_train_sum/batch_size:.5f}')
        _train_seconds = time.perf_counter() - _train_started
        _shape_info = dict(_shape, E0=int(_undirected_edge_index.shape[1]), factorised=False,
                           F=max_snp + n_genes)
        _log_gat_shape(MODEL_NAME, _shape_info, device, _resources, _train_seconds,
                       _shape['epochs'] * len(train_loader))

        with timed_stage('bio:predict'):
            ## Predict phenotypes for the test data
            model.eval()

            predicted_test = []
            actual_test = []
            attention = []
            with torch.no_grad():
                for test in test_loader:
                    test = test.to(device)
                    result, att = model(test.x, test.edge_index, test.batch, True)
                    predicted_test += result.cpu().tolist()
                    actual_test += test.y.cpu().tolist()
                    # att[1] (alpha) has shape (num_edges_in_this_batch, heads);
                    # ver4-4 R4.b first splits the BATCHED alpha back into one
                    # (num_edges_per_graph, heads) slice per graph (see
                    # pipeline_utils.split_batched_edge_attention's own
                    # docstring) - unchanged from that point on: GAT_prior_
                    # knowledge.py's own ".flatten()" only matches edge_name_
                    # from/to's length (num_edges) when heads == 1 - for
                    # heads > 1 it interleaves per-head values into a
                    # num_edges*heads-long vector, silently misaligning every
                    # downstream name<->value pairing once averaged across
                    # individuals. Averaging across heads per edge first keeps
                    # this always exactly num_edges long, matching
                    # edge_name_from/to for any heads value.
                    for alpha_g in split_batched_edge_attention(att[1], test.num_graphs):
                        attention += [alpha_g.mean(dim=1).tolist()]

            predicted_test = [item for sublist in predicted_test for item in sublist]

            ## Calculate the metrics
            mse = mean_squared_error(actual_test, predicted_test)
            r = pearsonr(actual_test, predicted_test)[0]

            ## Predict phenotypes for the validation data
            predicted_valid = []
            actual_valid = []
            if VALID:
                with torch.no_grad():
                    for valid in valid_loader:
                        valid = valid.to(device)
                        result = model(valid.x, valid.edge_index, valid.batch, None)
                        predicted_valid += result.cpu().tolist()
                        actual_valid += valid.y.cpu().tolist()

                predicted_valid = [item for sublist in predicted_valid for item in sublist]

            ## Predict phenotypes for the train data
            train_loader = DataLoader(data_train,
                                     shuffle=False,
                                     batch_size=bsize,
                                     num_workers=_num_workers, pin_memory=_pin_memory)
            predicted_train = []
            with torch.no_grad():
                for train in train_loader:
                    train = train.to(device)
                    result = model(train.x, train.edge_index, train.batch, None)
                    predicted_train += result.cpu().tolist()

            predicted_train = [k for i in predicted_train for k in i]

        with timed_stage('bio:ig'):
            ## Extract genomic marker effects
            if marker_effect == True:
                # ver4-33, R1.g: batched Integrated Gradients (opt-in) replaces the
                # per-graph explainer loop below; None = not applicable / OOM, in
                # which case the unchanged legacy explainer runs.
                gene_effect = None
                if _ig_batched_on:
                    gene_effect = _batched_ig_from_arrays(
                        model, padded_test, dummy_matrix, _undirected_edge_index, n_genes, samples,
                        device, _ig_internal_batch)

                if gene_effect is None:
                    explainer = Explainer(
                        model=model,
                        algorithm=CaptumExplainer('IntegratedGradients'),
                        explanation_type='model',
                        node_mask_type='attributes',
                        edge_mask_type=None,  # do not change here
                        model_config=dict(
                            mode='regression',
                            task_level='node',
                            return_type='raw',
                            ),
                    )

                    test_loader = DataLoader(data_test,
                                            shuffle=True,
                                            batch_size=1,
                                            num_workers=_num_workers, pin_memory=_pin_memory)

                    explanation = pd.DataFrame()
                    cnt = 0
                    for batch in test_loader:
                        batch = batch.to(device)
                        t = explainer(
                            batch.x,
                            batch.edge_index,
                            batch=batch.batch,
                            return_attention=None
                        )
                        t = pd.DataFrame(t['node_mask'].squeeze().detach().cpu()).sum(axis=1)
                        if explanation.shape[0] == 0:
                            explanation = t
                        else:
                            explanation += t
                        cnt += 1

                        if cnt == samples:
                            break

                    # one IG score per active gene node
                    gene_effect = (explanation / cnt).to_numpy().flatten()

                # genomic_prediction.py requires `effect` to have exactly one column
                # per *original* SNP marker (data_QTL_test.columns, unfiltered), in
                # that order (see this function's docstring). Every gene's
                # Integrated-Gradients attribution is broadcast onto every SNP that
                # was aggregated into that gene's node; SNPs outside every gene in
                # the network get exactly 0, since the model never saw them.
                # ver4-33, R1.d: vectorised broadcast (same values, same
                # last-gene-wins rule) unless BIO_PRIOR_PREP_FAST=False.
                effect = _broadcast_gene_effect(gene_effect, gene_to_markers, data_QTL_test.columns,
                                                fast=_prep_fast)
            else:
                effect = pd.DataFrame()

        attention = pd.concat([pd.DataFrame(edge_name_from),
                               pd.DataFrame(edge_name_to),
                               pd.DataFrame(attention).mean().T
                               ], axis=1)

    return r, mse, effect, predicted_test, predicted_valid, predicted_train, attention
