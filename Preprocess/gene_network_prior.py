"""
Preprocess/gene_network_prior.py
=================================

Turns a FlashP-style (https://flash-p.com/) - or any similarly structured,
manually authored - gene-interaction network JSON file into the two curated
artefacts that the biological prior-knowledge GAT model
(models/GAT_biological_prior_knowledge.py) needs as input:

  1. a gene list table            -> columns: name, chromosome, start, end
                                      (one row per gene *locus*; a gene with
                                      more than one genomic location gets more
                                      than one row - see build_gene_list)
  2. a gene-by-gene adjacency     -> a sparse (n_gene_rows x n_gene_rows)
                                      matrix plus a human-readable edge list
                                      (gene1, gene2, sign, effect, mechanism,
                                      inferred)

This module implements requirement 1 of the "biological prior-knowledge GAT"
specification:
    - select candidate genes from the network's "nodes" section
    - keep only nodes that represent a gene ("type" == "GENE", or FLASH-P's
      newer, compact-schema "ty" == "G")
    - resolve each candidate gene to a genomic location (chromosome, start,
      end), in either bp or cM, as chosen by the curator
    - write the resulting gene list to a CSV
    - build a sparse gene-by-gene adjacency matrix, indexed for later re-use

Design note - why this is a *preprocessing* module and not part of the model
----------------------------------------------------------------------------
genomic_prediction.py's GP() loop calls every prediction model's function
once per (population, phenotype, train/test ratio, replicate) combination -
often hundreds of times per run. Resolving a gene symbol to a genomic
location is a *curation* problem, not a numerical one: gene symbols are
frequently ambiguous, reused across species, tied to a specific reference
genome build, or - as turns out to be the case for several nodes in the very
FlashP network shipped as the worked example for this feature - are not
single-locus genes at all (pathway names such as "MAX_Pathway", or a hormone
class such as "Strigolactones" that was mistakenly typed as "GENE" by the
LLM-based network extraction step; see also the metadata field
`"build_method": "raw qwen3 output, no polish"` in that example file, which
is an explicit warning from the tool itself that the typing has not been
manually reviewed).

For all of these reasons, gene-location resolution is treated here as an
*offline curation step*: this module never scrapes the web itself. It
consumes a `location_lookup` table that a curator has already prepared -
manually, via TAIR/MaizeGDB/literature, or with the help of an agentic coding
tool such as Claude Code or a CrewAI crew, exactly as described in the EasiGP
GUI - and is strict about anything that table cannot resolve: unresolved
genes are reported and *dropped*, never guessed at. Run this module once per
(network JSON, species, reference genome / genetic map) combination, save its
two output files, and point the GUI / HPARAMETERS at those saved files; do
not call it from inside the genomic-prediction loop.

Typical one-off usage
----------------------
    from Preprocess.gene_network_prior import prepare_biological_prior

    gene_list, adjacency, dropped = prepare_biological_prior(
        json_path='network.json',
        location_lookup='my_curated_gene_locations.csv',
        out_gene_list_csv='Data/Arabidopsis/gene_list_shoot_branching.csv',
        out_adjacency_csv='Data/Arabidopsis/gene_adjacency_shoot_branching.csv',
        unit='bp',
    )
"""

import json
import re
import warnings

import numpy as np
import pandas as pd
from scipy import sparse


# ---------------------------------------------------------------------------
# Step 1-2: load the network and pull out candidate genes
# ---------------------------------------------------------------------------

def load_network_json(json_path):
    """Load a FlashP-style network JSON file and do a minimal structural
    sanity check (it must have 'nodes' and 'edges' sections).

    Always reads as UTF-8 explicitly: FlashP's literature-derived content
    (author names, mechanism/evidence text) routinely contains non-ASCII
    characters (accents, arrows, etc.), and Python's default `open()`
    encoding is platform-dependent - on Windows it's typically cp1252
    ("charmap"), which cannot decode those bytes and raises
    UnicodeDecodeError. JSON is UTF-8 by specification, so this is always
    correct regardless of platform or locale. Uses 'utf-8-sig' rather than
    plain 'utf-8' specifically so a leading byte-order mark - which Windows
    tools (Notepad, some editors' "save as UTF-8" option) routinely add, and
    which plain 'utf-8' does NOT strip, causing json.load() to fail on an
    otherwise valid file - is transparently handled; behaves identically to
    'utf-8' when no BOM is present, so this is strictly safer with no
    downside.
    """
    with open(json_path, 'r', encoding='utf-8-sig') as f:
        network = json.load(f)

    for key in ('nodes', 'edges'):
        if key not in network:
            raise ValueError(
                f"'{json_path}' does not look like a FlashP-style network "
                f"file: missing a top-level '{key}' section."
            )
    return network


def _normalise_phenotype_name(name):
    """Lowercase, strip, and collapse whitespace/underscore/hyphen runs, so
    'Shoot Branching', 'shoot_branching', and 'shoot-branching' all compare
    equal."""
    return re.sub(r'[\s_\-]+', '', str(name).strip().lower())


def phenotype_matches_network_metadata(network, phenotype_name):
    """Best-effort check for whether a network's declared phenotype (its
    'metadata.phenotype' / 'metadata.phenotype_node' fields, if present)
    plausibly matches `phenotype_name` - the genotype/phenotype-file column
    name this network is about to be used for.

    This is deliberately NOT a strict equality check: `phenotype_name` here
    is a column name in a user's own genotype/phenotype CSV (e.g.
    'branching_score'), chosen freely by whoever prepared that file, while a
    network's declared phenotype is FLASH-P's own phenotype slug (e.g.
    'shoot_branching', from '/run-flashp Shoot Branching in ...') or a
    human-typed label for a hand-authored JSON - the two naming schemes have
    no guaranteed relationship even when the file is exactly the right one.
    After normalising away case/whitespace/underscore/hyphen differences,
    this checks for substring containment in either direction (catches e.g.
    'branching' inside 'shoot_branching'), which is intentionally permissive
    to avoid a flood of false-positive warnings for legitimately different
    but correct naming conventions - it will still miss some real
    correspondences (e.g. an abbreviation like 'DTA' for 'days_to_anthesis'),
    so a negative result is a prompt to double-check, not proof of a mistake.

    Parameters
    ----------
    network : dict | str
        Either an already-loaded network dict (as returned by
        load_network_json), or a path to load one from.
    phenotype_name : str

    Returns
    -------
    (matched, declared) : (bool | None, str | None)
        `matched` is None - not True/False - if the network has no
        phenotype metadata to compare against at all (e.g. a hand-authored
        JSON with no 'metadata' block); callers should treat None as
        "nothing to check", not as a mismatch. `declared` is whatever
        phenotype string was found (for display in a warning message), or
        None if there wasn't one.
    """
    if isinstance(network, dict):
        net = network
    else:
        try:
            net = load_network_json(network)
        except Exception:
            return None, None

    metadata = net.get('metadata', {}) if isinstance(net, dict) else {}
    declared = metadata.get('phenotype') or metadata.get('phenotype_node')
    if not declared:
        return None, None

    a_full = _normalise_phenotype_name(phenotype_name)
    b_full = _normalise_phenotype_name(declared)
    if not a_full or not b_full:
        return None, declared

    # Below this length, a "substring" match is essentially guaranteed by
    # chance (e.g. a 2-letter abbreviation like 'BR' is trivially found
    # inside almost any longer word) and isn't meaningful evidence either
    # way - treated as nothing to check rather than guessed at.
    min_comparable_len = 3

    # Whole-string containment after normalising separators (handles clean
    # renames like 'Shoot_Branching' vs 'shoot_branching', or one being a
    # prefix/suffix of the other).
    if min(len(a_full), len(b_full)) >= min_comparable_len:
        if a_full == b_full or a_full in b_full or b_full in a_full:
            return True, declared

    # Meaningful word overlap - handles compound names that share a word but
    # aren't full substrings of each other, e.g. 'branching_score' vs
    # 'shoot_branching' (both contain 'branching', but neither whole string
    # contains the other).
    tokens_a = {w for w in re.split(r'[\s_\-]+', str(phenotype_name).strip().lower()) if len(w) >= min_comparable_len}
    tokens_b = {w for w in re.split(r'[\s_\-]+', str(declared).strip().lower()) if len(w) >= min_comparable_len}
    if tokens_a and tokens_b:
        matched = any(ta == tb or ta in tb or tb in ta for ta in tokens_a for tb in tokens_b)
        return matched, declared

    # Nothing long enough to compare reliably (e.g. a bare abbreviation with
    # no separators, like 'BR') - don't guess either way.
    return None, declared


def _is_gene_node(node):
    """True if `node` represents a gene, under either the verbose schema
    this module was originally written against ('type': 'GENE') or FLASH-P's
    newer, compact schema ('ty': 'G') - confirmed against a real FLASH-P
    run's network.json using nodes shaped like
    {"id": "D27", "ty": "G", "fn": "DWARF27", "src": true}. Both are
    accepted transparently everywhere a node's type is checked, rather than
    picking one and breaking networks built under the other FLASH-P
    version (or a hand-authored file copying either convention)."""
    return node.get('type') == 'GENE' or node.get('ty') == 'G'


def _node_type_label(node):
    """Best-effort human-readable type label for diagnostics/print messages
    only - NOT used for any functional gene/non-gene filtering, see
    _is_gene_node for that. Whichever of 'type'/'ty' is present, verbatim
    (so a compact-schema file prints e.g. 'H'/'E'/'M' rather than an
    unhelpful 'None')."""
    return node.get('type', node.get('ty'))


def _edge_field(edge, *keys):
    """First present value among `keys` in `edge`, checked in the given
    order - pass the verbose-schema key(s) first, then the compact-schema
    equivalent, e.g. _edge_field(e, 'source', 's'). Returns None if none of
    `keys` are present at all (as opposed to being present but null), so a
    genuinely-absent optional field (e.g. 'effect', which has no
    compact-schema equivalent) still correctly comes back as None rather
    than raising."""
    for k in keys:
        if k in edge:
            return edge[k]
    return None


def extract_candidate_genes(network):
    """Requirement 1, steps 1-2: from the network's 'nodes' section, select
    only the nodes that represent a gene - 'type' == 'GENE' (this module's
    original, verbose schema) or 'ty' == 'G' (FLASH-P's newer, compact
    schema) - see _is_gene_node above.

    Returns the raw node dicts (id, type/ty, full_name/fn, description,
    is_source/src, ...) unchanged, so no information from the source JSON
    is lost at this stage - filtering by type is the only operation
    performed here.
    """
    nodes = network['nodes']
    genes = [n for n in nodes if _is_gene_node(n)]

    if len(genes) == 0:
        warnings.warn("No gene nodes ('type' == 'GENE' or 'ty' == 'G') were found in the network JSON.")

    non_gene_types = sorted({str(_node_type_label(n)) for n in nodes if not _is_gene_node(n)})
    if non_gene_types:
        print(f"[gene_network_prior] Ignoring {len(nodes) - len(genes)} non-gene node(s) "
              f"with type(s): {non_gene_types}")

    return genes


# ---------------------------------------------------------------------------
# Step 3-4: resolve genomic locations and build the gene list
# ---------------------------------------------------------------------------

def _load_location_lookup(location_lookup, unit='bp'):
    """Normalise `location_lookup` (a dict, DataFrame or CSV path) into a
    DataFrame with columns ['name', 'chromosome', 'start', 'end'] - one row
    per locus - plus, when present in the source, the provenance columns
    ['agi_locus_id', 'source']. A gene with several loci simply needs several
    rows sharing the same 'name'.

    Column names are resolved via aliasing so the *canonical* curated-table
    schema (as produced by a human curator or by
    Preprocess.gene_location_agent, and as shown in a real, hand-verified
    example: 'Gene_Name, Start_bp, End_bp, Chromosome, AGI_Locus_ID, Source')
    is accepted directly, alongside the older, more generic
    'name, chromosome, start, end' shape. `unit` selects between the
    *_bp and *_cM start/end columns when both could plausibly be present.
    """
    ALIASES = {
        'name': ['Gene_Name', 'gene_name', 'Name', 'name'],
        'chromosome': ['Chromosome', 'chromosome', 'Chrom', 'chrom'],
        'agi_locus_id': ['AGI_Locus_ID', 'agi_locus_id', 'AGI', 'Locus_ID', 'locus_id'],
        'source': ['Source', 'source'],
    }
    start_aliases = {
        'bp': ['Start_bp', 'start_bp', 'Start', 'start'],
        'cM': ['Start_cM', 'start_cM', 'start_cm', 'Start', 'start'],
    }
    end_aliases = {
        'bp': ['End_bp', 'end_bp', 'End', 'end'],
        'cM': ['End_cM', 'end_cM', 'end_cm', 'End', 'end'],
    }

    if isinstance(location_lookup, pd.DataFrame):
        table = location_lookup.copy()
    elif isinstance(location_lookup, dict):
        rows = []
        for gene_id, locs in location_lookup.items():
            # Accept either a single (chrom, start, end) tuple, or a list of
            # them for genes with multiple loci.
            if isinstance(locs, (list, tuple)) and len(locs) > 0 and isinstance(locs[0], (list, tuple)):
                loc_iter = locs
            else:
                loc_iter = [locs]
            for chrom, start, end in loc_iter:
                rows.append({'Gene_Name': gene_id, 'Chromosome': chrom,
                             f'Start_{unit}': start, f'End_{unit}': end})
        table = pd.DataFrame(rows)
    else:
        table = pd.read_csv(location_lookup, encoding='utf-8-sig')

    def _resolve(col_key, candidates):
        for c in candidates:
            if c in table.columns:
                return c
        return None

    name_col = _resolve('name', ALIASES['name'])
    chrom_col = _resolve('chromosome', ALIASES['chromosome'])
    start_col = _resolve('start', start_aliases[unit])
    end_col = _resolve('end', end_aliases[unit])
    agi_col = _resolve('agi_locus_id', ALIASES['agi_locus_id'])
    source_col = _resolve('source', ALIASES['source'])

    missing = [label for label, col in
               [('name (e.g. Gene_Name)', name_col), ('chromosome (e.g. Chromosome)', chrom_col),
                (f'start (e.g. Start_{unit})', start_col), (f'end (e.g. End_{unit})', end_col)]
               if col is None]
    if missing:
        raise ValueError(
            f"location_lookup is missing required column(s): {missing}. Found columns: "
            f"{list(table.columns)}. Expected either the canonical schema "
            f"(Gene_Name, Chromosome, Start_{unit}, End_{unit}[, AGI_Locus_ID, Source]) "
            f"or the generic schema (name, chromosome, start, end)."
        )

    out = pd.DataFrame({
        'name': table[name_col],
        'chromosome': table[chrom_col],
        'start': table[start_col],
        'end': table[end_col],
    })
    if agi_col is not None:
        out['agi_locus_id'] = table[agi_col]
    if source_col is not None:
        out['source'] = table[source_col]

    return out.reset_index(drop=True)


def build_gene_list(candidate_genes, location_lookup, unit='bp'):
    """Requirement 1, steps 3-4.

    Parameters
    ----------
    candidate_genes : list of node dicts, as returned by extract_candidate_genes()
    location_lookup : dict | pd.DataFrame | str (CSV path)
        A curator-prepared table/mapping of gene id -> genomic location(s).
        See `_load_location_lookup` for the accepted shapes. This is expected
        to have been produced offline (see module docstring) - e.g. by
        looking each gene up on https://www.arabidopsis.org/ (TAIR) or
        MaizeGDB, optionally with the help of an agentic coding tool.
    unit : {'bp', 'cM'}
        Recorded on the returned DataFrame (`gene_list.attrs['unit']`) purely
        as a label for downstream bookkeeping/plotting. No unit conversion is
        performed here: start/end must already be expressed in whichever
        unit the accompanying marker_info.csv also uses, because the two are
        later compared directly (interval overlap) in
        models/GAT_biological_prior_knowledge.py.

    Returns
    -------
    gene_list : pd.DataFrame, columns ['name', 'chromosome', 'start', 'end']
        One row per resolved gene locus, in candidate_genes order (genes with
        multiple loci contribute multiple, consecutive rows).
    dropped : list of str
        Gene ids from candidate_genes that had no entry in location_lookup
        and were therefore excluded (never guessed at).
    """
    if unit not in ('bp', 'cM'):
        raise ValueError(f"unit must be 'bp' or 'cM', got {unit!r}")

    lookup_table = _load_location_lookup(location_lookup, unit=unit)
    candidate_ids = [g['id'] for g in candidate_genes]
    has_provenance = 'agi_locus_id' in lookup_table.columns or 'source' in lookup_table.columns

    rows = []
    dropped = []
    for gene_id in candidate_ids:
        matches = lookup_table[lookup_table['name'] == gene_id]
        if matches.shape[0] == 0:
            dropped.append(gene_id)
            continue
        for _, r in matches.iterrows():
            row = {'name': gene_id, 'chromosome': str(r['chromosome']),
                   'start': float(r['start']), 'end': float(r['end'])}
            if has_provenance:
                row['agi_locus_id'] = r.get('agi_locus_id', '')
                row['source'] = r.get('source', '')
            rows.append(row)

    columns = ['name', 'chromosome', 'start', 'end'] + (['agi_locus_id', 'source'] if has_provenance else [])
    gene_list = pd.DataFrame(rows, columns=columns)

    # Basic sanity check on the loci themselves - a location with end < start
    # is either a curation typo or a strand-orientation mix-up, and would
    # silently break the interval-overlap test used later, so fail loudly.
    bad = gene_list[gene_list['end'] < gene_list['start']]
    if bad.shape[0] > 0:
        raise ValueError(
            f"{bad.shape[0]} row(s) in the resolved gene list have end < start "
            f"(check for swapped columns or a strand mix-up): \n{bad}"
        )

    if dropped:
        print(f"[gene_network_prior] {len(dropped)} candidate gene(s) had no resolvable "
              f"genomic location and were dropped from the gene list: {dropped}")

    gene_list.attrs['unit'] = unit
    return gene_list, dropped


def save_gene_list_csv(gene_list, path):
    """Requirement 1, step 4: write the gene list dataframe to CSV.

    Uses the same column *names* as Data/<dataset>/marker_info.csv
    ('chromosome', 'name', 'start', 'end'), so this file is a drop-in
    addition to (or replacement for) a marker_info.csv should a user want
    gene-level positions available to circos_plot.py as well.
    """
    gene_list.to_csv(path, index=False, encoding='utf-8')
    return path


# ---------------------------------------------------------------------------
# Step 5: build the sparse gene-by-gene adjacency matrix
# ---------------------------------------------------------------------------

def build_gene_adjacency(network, gene_list, include_mediated_edges=False, max_hops=3):
    """Requirement 1, step 5: a sparse (n_gene_rows x n_gene_rows) adjacency
    matrix between the genes in `gene_list`, built from the same network
    JSON used to build the gene list.

    Indexing
    --------
    Because a gene can occupy more than one row in `gene_list` (multiple
    loci), the adjacency matrix is indexed at the *row* level, not the gene
    *name* level: row i of gene_list <-> index i of the adjacency matrix.
    A JSON edge between gene symbols A and B is expanded to a link between
    every row-index of A and every row-index of B, since the interaction
    described in the network (e.g. "A activates B") is a property of the
    gene, not of one particular locus copy of it.

    Default behaviour (include_mediated_edges=False)
    --------------------------------------------------
    Keeps only *direct* edges from the source JSON where both endpoints are
    themselves genes present in gene_list. This is the least speculative,
    most literal reading of "use the information from the input json file"
    to connect the genes that made it into the located gene list: every kept
    edge corresponds to exactly one edge object in the source JSON.

    Optional behaviour (include_mediated_edges=True)
    ---------------------------------------------------
    Networks like the FlashP example are routinely interspersed with
    hormone / metabolite / protein-complex "mediator" nodes that are not
    themselves genes (e.g. GeneA -> Auxin -> GeneB) and so never appear in
    gene_list. With this flag on, gene A is additionally linked to gene B
    whenever a directed path from A to B exists that passes only through
    non-GENE nodes, up to `max_hops` hops - this recovers most of the
    biologically meaningful gene-gene connectivity that a "direct edges
    only" adjacency would otherwise miss (in the shipped example, e.g.
    D27 -> CCD7 -> CCD8 -> Strigolactone -> BRC1 has no *direct* GENE-GENE
    edge at all, yet is a well-established regulatory chain). Every such
    inferred edge is flagged `inferred=True` in the returned edge list, so it
    can always be told apart from an edge literally stated in the source
    JSON (`inferred=False`), and the mechanism/evidence columns record the
    chain of source-JSON edges (and their signs) it was collapsed from.

    Returns
    -------
    adjacency : scipy.sparse.csr_matrix, shape (n_rows, n_rows)
        Symmetric 0/1 adjacency matrix (an edge in either direction between
        two rows sets both (i, j) and (j, i) to 1 - the GAT itself is made
        undirected downstream via torch_geometric's ToUndirected transform,
        exactly as in models/GAT_prior_knowledge.py, so the stored matrix is
        pre-symmetrised for anyone inspecting it directly).
    edge_list : pd.DataFrame
        Human-readable, columns ['gene1_index','gene2_index','gene1_name',
        'gene2_name','sign','effect','mechanism','inferred']. One row per
        *directed* edge kept (before symmetrisation), for transparency / QC.
    """
    if gene_list.shape[0] == 0:
        raise ValueError("gene_list is empty - build_gene_list() must run (and resolve at "
                          "least one gene) before build_gene_adjacency().")

    n_rows = gene_list.shape[0]
    name_to_rows = {}
    for idx, name in enumerate(gene_list['name'].tolist()):
        name_to_rows.setdefault(name, []).append(idx)

    node_is_gene = {n['id']: _is_gene_node(n) for n in network['nodes']}

    direct_edges = []  # (src_name, tgt_name, sign, effect, mechanism)
    for e in network['edges']:
        # 'source'/'target' (verbose) or 's'/'t' (compact); 'sign' or 'x'
        # (compact schema folds sign into this one numeric field, e.g. +1/-1
        # - there's no separate compact-schema 'effect' magnitude, so effect
        # stays None for a compact-schema edge, which is an honest
        # reflection of that field simply not existing there, not a bug);
        # 'mechanism' (a text description) or, as the closest compact-schema
        # equivalent, 'd' (a DOI) - both are "why this edge exists"
        # documentation, so reusing the same column is the closest faithful
        # mapping rather than leaving it empty.
        direct_edges.append((
            _edge_field(e, 'source', 's'),
            _edge_field(e, 'target', 't'),
            _edge_field(e, 'sign', 'x'),
            _edge_field(e, 'effect'),
            _edge_field(e, 'mechanism', 'd'),
        ))

    edge_rows = []
    seen_directed_pairs = set()

    def _add_row_edges(src_name, tgt_name, sign, effect, mechanism, inferred):
        if src_name not in name_to_rows or tgt_name not in name_to_rows:
            return
        for i in name_to_rows[src_name]:
            for j in name_to_rows[tgt_name]:
                if i == j:
                    continue  # no self-loops (GATv2Conv is called with add_self_loops=False)
                key = (i, j, inferred)
                if key in seen_directed_pairs:
                    continue
                seen_directed_pairs.add(key)
                edge_rows.append({'gene1_index': i, 'gene2_index': j,
                                   'gene1_name': src_name, 'gene2_name': tgt_name,
                                   'sign': sign, 'effect': effect, 'mechanism': mechanism,
                                   'inferred': inferred})

    # --- direct GENE -> GENE edges, taken literally from the JSON ----------
    for src_name, tgt_name, sign, effect, mechanism in direct_edges:
        if node_is_gene.get(src_name) and node_is_gene.get(tgt_name):
            _add_row_edges(src_name, tgt_name, sign, effect, mechanism, inferred=False)

    # --- optional: gene -> ... (non-gene mediators only) ... -> gene -------
    if include_mediated_edges:
        adjacency_list = {}
        for src_name, tgt_name, sign, effect, mechanism in direct_edges:
            adjacency_list.setdefault(src_name, []).append((tgt_name, sign, effect, mechanism))

        gene_names = set(name_to_rows.keys())

        def _dfs(start_gene):
            # (current_node, path_signs, path_mechanisms, hops)
            stack = [(start_gene, [], [], 0)]
            visited_via = set()
            while stack:
                node, signs, mechs, hops = stack.pop()
                if hops >= max_hops:
                    continue
                for nxt, sign, effect, mechanism in adjacency_list.get(node, []):
                    if nxt == start_gene:
                        continue  # would be a self-loop
                    new_signs = signs + [sign]
                    new_mechs = mechs + [mechanism]
                    if nxt in gene_names:
                        combined_sign = 1
                        for s in new_signs:
                            if s is None:
                                combined_sign = None
                                break
                            combined_sign *= s
                        chain = ' -> '.join([start_gene] + [m for m in new_mechs if m] or [nxt])
                        _add_row_edges(start_gene, nxt, combined_sign,
                                        'inferred_' + ('activation' if combined_sign == 1
                                                        else 'inhibition' if combined_sign == -1
                                                        else 'mixed'),
                                        chain, inferred=True)
                        # Still keep exploring further in case the same
                        # mediator chain also reaches other genes downstream.
                        continue
                    state = (nxt, hops + 1)
                    if state in visited_via:
                        continue
                    visited_via.add(state)
                    stack.append((nxt, new_signs, new_mechs, hops + 1))

        for gname in gene_names:
            _dfs(gname)

    edge_list = pd.DataFrame(edge_rows, columns=['gene1_index', 'gene2_index', 'gene1_name',
                                                  'gene2_name', 'sign', 'effect', 'mechanism', 'inferred'])

    rows_idx = edge_list['gene1_index'].to_numpy(dtype=int) if edge_list.shape[0] else np.array([], dtype=int)
    cols_idx = edge_list['gene2_index'].to_numpy(dtype=int) if edge_list.shape[0] else np.array([], dtype=int)
    data = np.ones(len(rows_idx))

    adjacency = sparse.csr_matrix((data, (rows_idx, cols_idx)), shape=(n_rows, n_rows))
    adjacency = adjacency.maximum(adjacency.T)  # symmetrise for storage/inspection
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()

    if edge_list.shape[0] == 0:
        warnings.warn(
            "The gene-gene adjacency has zero edges. Every node in the resulting graph "
            "will be isolated and GATv2Conv's attention mechanism will have nothing to "
            "attend over. Consider include_mediated_edges=True, or check that the gene "
            "symbols in gene_list match the 'source'/'target' (or compact-schema 's'/'t') "
            "ids used in the network JSON."
        )

    return adjacency, edge_list


def save_adjacency(adjacency, edge_list, npz_path=None, edge_list_csv_path=None):
    """Persist the adjacency matrix (as a scipy .npz) and/or the
    human-readable edge list (as a CSV). Either path can be omitted."""
    if npz_path is not None:
        sparse.save_npz(npz_path, adjacency)
    if edge_list_csv_path is not None:
        edge_list.to_csv(edge_list_csv_path, index=False, encoding='utf-8')
    return npz_path, edge_list_csv_path


# ---------------------------------------------------------------------------
# End-to-end convenience wrapper
# ---------------------------------------------------------------------------

def prepare_biological_prior(json_path, location_lookup, out_gene_list_csv,
                               out_adjacency_csv, unit='bp',
                               include_mediated_edges=False, max_hops=3):
    """Run requirement-1 end to end: JSON network -> gene_list.csv +
    adjacency edge-list CSV. This is the function a curator runs once, after
    preparing (or generating with FlashP) the network JSON and the gene
    location lookup table, and before ever calling
    models/GAT_biological_prior_knowledge.py.

    Returns (gene_list, adjacency, edge_list, dropped_genes).
    """
    network = load_network_json(json_path)
    candidate_genes = extract_candidate_genes(network)
    gene_list, dropped = build_gene_list(candidate_genes, location_lookup, unit=unit)
    save_gene_list_csv(gene_list, out_gene_list_csv)

    adjacency, edge_list = build_gene_adjacency(
        network, gene_list, include_mediated_edges=include_mediated_edges, max_hops=max_hops
    )
    save_adjacency(adjacency, edge_list, edge_list_csv_path=out_adjacency_csv)

    print(f"[gene_network_prior] Wrote {gene_list.shape[0]} gene locus row(s) to "
          f"'{out_gene_list_csv}' and {edge_list.shape[0]} directed edge(s) to "
          f"'{out_adjacency_csv}' ({dropped and len(dropped) or 0} gene(s) dropped: {dropped}).")

    return gene_list, adjacency, edge_list, dropped
