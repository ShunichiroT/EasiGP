import os

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

from Preprocess.gene_network_prior import (
    load_network_json, extract_candidate_genes, build_gene_list, build_gene_adjacency,
)
from Preprocess.LD_pruning import LD_pruning
from pipeline_utils import unify_columns_by_position
from Preprocess.data_driven_prior_network import (
    select_markers_for_data_driven_network, compute_data_driven_interactions,
    merge_biological_and_data_driven_networks,
)


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


def _build_padded_node_features(data_QTL, gene_to_markers, max_snp):
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

    Returns a numpy array of shape (n_samples, n_genes, max_snp).
    """
    n_samples = data_QTL.shape[0]
    n_genes = len(gene_to_markers)
    features = np.zeros((n_samples, n_genes, max_snp), dtype=float)

    for g, markers in enumerate(gene_to_markers):
        if len(markers) == 0:
            continue
        values = data_QTL[markers].to_numpy(dtype=float)  # (n_samples, k_g)
        features[:, g, :values.shape[1]] = values

    return features


def GAT_biological_prior_knowledge(data_train, data_valid, data_test, params, RESULT_NAME=None, PHENOTYPE_NAME=None, MODEL_NAME='GAT_biological_prior_knowledge', merge_source_data=None, network_cache=None):
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
    network_json_path = params[7]
    gene_location_csv_path = params[8]
    marker_info_path = params[9]
    unit = params[10]
    include_mediated_edges = params[11]
    max_hops = params[12]
    data_driven_merge = params[13]
    marker_effect = params[14]
    samples = params[15]

    # =========================================================================
    # 1. Build the gene list + gene-gene adjacency (requirement 1) directly
    #    from the network JSON and the curated gene-location CSV, load the
    #    SNP position table, and build the gene <-> SNP mapping (requirement
    #    2, step 1). Nothing here is read from a separately pre-built
    #    gene_list.csv/adjacency.csv - both are derived fresh, every call.
    # =========================================================================

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

    marker_info = pd.read_csv(marker_info_path, encoding='utf-8-sig')
    # Requirement 8: unify by position - 'name' here is the marker's
    # identifying column HEADER (a fixed, structural field of this file's
    # schema), not the marker names themselves (the VALUES, left
    # untouched) - so positional renaming is safe regardless of what
    # header text the file actually uses. Still validated for column
    # COUNT (a genuine schema problem, not just a naming difference).
    missing_mi_cols = 4 - marker_info.shape[1]
    if missing_mi_cols > 0:
        raise ValueError(
            f"'{marker_info_path}' has only {marker_info.shape[1]} column(s) - expected at "
            f"least 4 (chromosome, name, start, end, in that order)."
        )
    marker_info = unify_columns_by_position(
        marker_info, ['chromosome', 'name', 'start', 'end'], 'marker info file'
    )

    available_markers = data_train.columns[:-1]  # every SNP originally in the genotype table
    gene_to_markers_full = _map_genes_to_markers(gene_list, marker_info, available_markers)

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

    # Persist the active genes' own coordinates, in the exact same schema as
    # marker_info.csv (chromosome, name, start, end) - see the
    # RESULT_NAME/PHENOTYPE_NAME docstring entry above for why. Named after
    # MODEL_NAME (this call's own model-instance name, e.g.
    # 'GAT_biological_prior_knowledge_2') rather than a fixed name, so
    # multiple bio-prior instances in the same run - each with a different
    # network - never overwrite each other's coordinate files; circos_plot.py
    # already discovers these by filename pattern
    # ('*_gene_coordinates_<phenotype>.csv'), not by a hardcoded model name.
    if RESULT_NAME is not None and PHENOTYPE_NAME is not None:
        coord_dir = os.path.join('.', 'Result', RESULT_NAME)
        os.makedirs(coord_dir, exist_ok=True)
        coord_path = os.path.join(
            coord_dir, f'{MODEL_NAME}_gene_coordinates_{PHENOTYPE_NAME}.csv'
        )
        gene_list_active[['chromosome', 'name', 'start', 'end']].to_csv(
            coord_path, index=False, encoding='utf-8'
        )

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

        rf_filter_cfg = data_driven_merge.get('rf_filter')
        if rf_filter_cfg is None:
            raise ValueError(
                "data_driven_merge['rf_filter'] is required when data_driven_merge['enabled'] "
                "is True - see this function's own docstring."
            )
        top_rate = data_driven_merge.get('top_rate')
        if top_rate is None:
            raise ValueError(
                "data_driven_merge['top_rate'] is required when data_driven_merge['enabled'] "
                "is True - see this function's own docstring."
            )

        _merge_source = merge_source_data if merge_source_data is not None else (data_train, data_valid, data_test)
        _merge_train, _merge_valid, _merge_test = _merge_source
        _merge_train_y = _merge_train.iloc[:, -1]

        # Requirement 11 (efficiency): reuse whatever this task has
        # already computed for this instance - see this function's own
        # network_cache docstring entry - instead of unconditionally
        # redoing LD pruning / RF filtering / the Shapley interaction
        # search every single call. Each of the two expensive steps below
        # is skipped independently if its own result is already cached,
        # so a partially-primed cache (e.g. GP() already computed
        # rf_selected_markers via OTHER_MODELS_MARKER_SOURCE=
        # 'gene_network_plus_rf', but this is still the first time THIS
        # function itself has run this task) still saves whatever it can.
        _cache_entry = network_cache.setdefault(MODEL_NAME, {}) if network_cache is not None else None

        if _cache_entry is not None and 'rf_selected_markers' in _cache_entry:
            rf_selected_markers = _cache_entry['rf_selected_markers']
            print(f"[GAT_biological_prior_knowledge] Data-driven merge: reusing "
                  f"{len(rf_selected_markers)} RF-selected marker(s) already computed earlier "
                  f"this task (LD pruning/RF filtering skipped).")
        else:
            _merge_train_x = _merge_train.iloc[:, :-1]
            if data_driven_merge.get('ld_prune') is not None:
                _merge_train_x, _, _ = LD_pruning(
                    _merge_train_x, pd.DataFrame(), _merge_test.iloc[:, :-1], data_driven_merge['ld_prune']
                )
            rf_selected_markers, _fitted_rf = select_markers_for_data_driven_network(
                _merge_train_x, _merge_train_y, rf_filter_cfg
            )
            print(f"[GAT_biological_prior_knowledge] Data-driven merge: {len(rf_selected_markers)} "
                  f"marker(s) selected by RF filtering (out of {_merge_train_x.shape[1]} candidate(s)).")
            if _cache_entry is not None:
                _cache_entry['rf_selected_markers'] = rf_selected_markers

        if _cache_entry is not None and 'pair_df' in _cache_entry:
            pair_df = _cache_entry['pair_df']
            print(f"[GAT_biological_prior_knowledge] Data-driven merge: reusing "
                  f"{pair_df.shape[0]} pairwise interaction(s) already computed earlier this task.")
        else:
            # _merge_train (not _merge_train_x, which may not even exist in
            # the cache-hit branch above) sliced directly by
            # rf_selected_markers - selecting marker COLUMNS by name is
            # always valid regardless of whether pruning happened in this
            # call or was reused from the cache, since pruning only ever
            # narrows which columns exist, never renames/transforms them.
            pair_df = compute_data_driven_interactions(
                _merge_train.loc[:, rf_selected_markers], _merge_train_y, rf_filter_cfg, top_rate,
                n_estimators_override=data_driven_merge.get('shap_n_estimators_override'),
            )
            print(f"[GAT_biological_prior_knowledge] Data-driven merge: {pair_df.shape[0]} unique "
                  f"pairwise interaction(s) kept (top {top_rate}% - see the "
                  f"data_driven_prior_network log line just above for the full candidate-pool count "
                  f"this percentage was taken of).")
            if _cache_entry is not None:
                _cache_entry['pair_df'] = pair_df

        bare_marker_names, bare_rows, bare_gene_to_markers, extra_edges = merge_biological_and_data_driven_networks(
            gene_to_markers, pair_df, rf_selected_markers, marker_info
        )
        if bare_marker_names:
            gene_list_active = pd.concat([gene_list_active, bare_rows], ignore_index=True)
            gene_to_markers = gene_to_markers + bare_gene_to_markers
            n_genes = len(gene_to_markers)
            print(f"[GAT_biological_prior_knowledge] Data-driven merge: {len(bare_marker_names)} "
                  f"bare marker node(s) added, {len(extra_edges)} new edge(s) to/among gene and "
                  f"bare-marker nodes.")
    elif orphan_markers:
        preview = orphan_markers[:10]
        print(f"[GAT_biological_prior_knowledge] {len(orphan_markers)} marker(s) outside every gene "
              f"were ignored (data_driven_merge disabled): {preview}"
              f"{', ...' if len(orphan_markers) > 10 else ''}")

    max_snp = max(len(m) for m in gene_to_markers)

    # Re-index the adjacency edge list from original gene_list row indices to
    # the compacted 0..n_genes-1 indices used for the graph (requirement 3),
    # dropping any edge that touches a gene row filtered out just above.
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

    # One dummy (one-hot) row per gene - identical in spirit to lines 70-73
    # of GAT_prior_knowledge.py, just built directly at gene-node granularity
    # instead of via a melt of a per-marker table. GAT_prior_knowledge.py's
    # own "dummy[dummy==False]=0 / dummy[dummy==True]=1" recipe raises a
    # pandas LossySetitemError on newer pandas releases (bool-dtype columns
    # reject an int 0/1 assignment in place); requesting float dummies
    # directly from get_dummies() produces the identical 0.0/1.0 one-hot
    # matrix without depending on that in-place cast, so it is used here
    # instead.
    dummy = pd.get_dummies(pd.DataFrame(list(range(n_genes))), columns=[0], dtype=float)
    dummy_matrix = dummy.to_numpy(dtype=float)  # (n_genes, n_genes)

    padded_train = _build_padded_node_features(data_QTL_train, gene_to_markers, max_snp)
    if VALID:
        padded_valid = _build_padded_node_features(data_QTL_valid, gene_to_markers, max_snp)
    padded_test = _build_padded_node_features(data_QTL_test, gene_to_markers, max_snp)

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

    data_train = []
    for kk in range(data_pheno_train.shape[0]):
        tmp = Data()
        node_features_tmp = np.concatenate([padded_train[kk], dummy_matrix], axis=1)
        data_pheno_train_tmp = np.expand_dims(np.array(data_pheno_train[kk]), axis=0)
        tmp.x = torch.from_numpy(node_features_tmp).to(torch.float)
        tmp.y = torch.from_numpy(data_pheno_train_tmp).to(torch.float)
        tmp.edge_index = data_train_edge_index.clone()
        tmp = T.ToUndirected()(tmp)
        data_train += [tmp]

    if VALID:
        data_valid = []
        for kk in range(data_pheno_valid.shape[0]):
            tmp = Data()
            node_features_tmp = np.concatenate([padded_valid[kk], dummy_matrix], axis=1)
            data_pheno_valid_tmp = np.expand_dims(np.array(data_pheno_valid[kk]), axis=0)
            tmp.x = torch.from_numpy(node_features_tmp).to(torch.float)
            tmp.y = torch.from_numpy(data_pheno_valid_tmp).to(torch.float)
            tmp.edge_index = data_train_edge_index.clone()
            tmp = T.ToUndirected()(tmp)
            data_valid += [tmp]

    data_test = []
    for kk in range(data_pheno_test.shape[0]):
        tmp = Data()
        node_features_tmp = np.concatenate([padded_test[kk], dummy_matrix], axis=1)
        data_pheno_test_tmp = np.expand_dims(np.array(data_pheno_test[kk]), axis=0)
        tmp.x = torch.from_numpy(node_features_tmp).to(torch.float)
        tmp.y = torch.from_numpy(data_pheno_test_tmp).to(torch.float)
        tmp.edge_index = data_train_edge_index.clone()
        tmp = T.ToUndirected()(tmp)
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

    train_loader = DataLoader(data_train,
                             shuffle=True,
                             batch_size=bsize)
    if VALID:
        valid_loader = DataLoader(data_valid,
                                 batch_size=bsize)
    test_loader = DataLoader(data_test,
                             batch_size=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ## Train GAT
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate, weight_decay=decay)

    for epoch in range(epoch):
        loss_train_sum = 0
        batch_size = len(train_loader)

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch, None)
            loss = F.mse_loss(torch.squeeze(out), batch.y)
            loss.backward()
            optimizer.step()
            loss_train_sum += loss

        print(f'Epoch {epoch:>3} | Train Loss: {loss_train_sum/batch_size:.5f}')

    ## Predict phenotypes for the test data
    model.eval()

    predicted_test = []
    actual_test = []
    attention = []
    for test in test_loader:
        result, att = model(test.x, test.edge_index, test.batch, True)
        predicted_test += result.tolist()
        actual_test += test.y.tolist()
        # att[1] (alpha) has shape (num_edges, heads); GAT_prior_knowledge.py's
        # own ".flatten()" only matches edge_name_from/to's length (num_edges)
        # when heads == 1 - for heads > 1 it interleaves per-head values into
        # a num_edges*heads-long vector, silently misaligning every
        # downstream name<->value pairing once averaged across individuals.
        # Averaging across heads per edge first keeps this always exactly
        # num_edges long, matching edge_name_from/to for any heads value.
        attention += [att[1].detach().mean(dim=1).tolist()]

    predicted_test = [item for sublist in predicted_test for item in sublist]

    ## Calculate the metrics
    mse = mean_squared_error(actual_test, predicted_test)
    r = pearsonr(actual_test, predicted_test)[0]

    ## Predict phenotypes for the validation data
    predicted_valid = []
    actual_valid = []
    if VALID:
        for valid in valid_loader:
            result = model(valid.x, valid.edge_index, valid.batch, None)
            predicted_valid += result.tolist()
            actual_valid += valid.y.tolist()

        predicted_valid = [item for sublist in predicted_valid for item in sublist]

    ## Predict phenotypes for the train data
    train_loader = DataLoader(data_train,
                             shuffle=False,
                             batch_size=bsize)
    predicted_train = []
    for train in train_loader:
        result = model(train.x, train.edge_index, train.batch, None)
        predicted_train += result.tolist()

    predicted_train = [k for i in predicted_train for k in i]

    ## Extract genomic marker effects
    if marker_effect == True:
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
                                batch_size=1)

        explanation = pd.DataFrame()
        cnt = 0
        for batch in test_loader:
            t = explainer(
                batch.x,
                batch.edge_index,
                batch=batch.batch,
                return_attention=None
            )
            t = pd.DataFrame(t['node_mask'].squeeze().detach()).sum(axis=1)
            if explanation.shape[0] == 0:
                explanation = t
            else:
                explanation += t
            cnt += 1

            if cnt == samples:
                break

        gene_effect = (explanation / cnt).to_numpy().flatten()  # one IG score per active gene node

        # genomic_prediction.py requires `effect` to have exactly one column
        # per *original* SNP marker (data_QTL_test.columns, unfiltered), in
        # that order (see this function's docstring). Every gene's
        # Integrated-Gradients attribution is broadcast onto every SNP that
        # was aggregated into that gene's node; SNPs outside every gene in
        # the network get exactly 0, since the model never saw them.
        effect = pd.DataFrame(0.0, index=[0], columns=data_QTL_test.columns)
        for g in range(n_genes):
            for marker in gene_to_markers[g]:
                effect.loc[0, marker] = gene_effect[g]
    else:
        effect = pd.DataFrame()

    attention = pd.concat([pd.DataFrame(edge_name_from),
                           pd.DataFrame(edge_name_to),
                           pd.DataFrame(attention).mean().T
                           ], axis=1)

    return r, mse, effect, predicted_test, predicted_valid, predicted_train, attention
