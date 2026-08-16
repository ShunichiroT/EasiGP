import pandas as pd
import os
import time
from datetime import datetime
from itertools import product
from sklearn.model_selection import train_test_split

from models.RF import *
from models.SVR import *
from models.KNN import *
from models.MLP import *
from models.GAT_infinitesimal_node_level import *
from models.GAT_infinitesimal import *
from models.GAT_fully_connected import *
from models.GAT_prior_knowledge import *
from models.GAT_biological_prior_knowledge import *
from models.ensemble import *

from Preprocess.LD_pruning import *
from Preprocess.LD_decay_plot import (
    compute_ld_decay_data, save_ld_decay_data, plot_ld_decay,
    resolve_snp_info_for_decay, sanitize_for_filename, unique_path, write_keep_log_marker,
    WINDOW_UNITS as LD_DECAY_WINDOW_UNITS, DEFAULT_MAX_DISTANCE as LD_DECAY_DEFAULT_MAX_DISTANCE,
    DEFAULT_BIN_WIDTH as LD_DECAY_DEFAULT_BIN_WIDTH,
)
from Preprocess.RF_marker_filtering import RF_marker_filtering
from Preprocess.data_driven_prior_network import select_markers_for_data_driven_network
from Preprocess.gene_network_prior import (
    phenotype_matches_network_metadata, load_network_json, extract_candidate_genes, build_gene_list,
)
from Preprocess.plink_io import (
    validate_plink_fileset, read_bim_marker_info, read_fam_iids, plink_to_genotype_df, ld_prune_plink_native,
)
# _map_genes_to_markers is a "private" (leading-underscore) helper inside the
# model file, reused here deliberately - it's the exact same SNP<->gene
# interval-overlap logic GAT_biological_prior_knowledge.py itself uses, and
# reusing it (rather than re-implementing the same logic a second time) is
# what lets this module determine "which markers fall inside a gene window"
# up front, to --extract only those from a PLINK fileset, while guaranteeing
# it's byte-for-byte the same computation the model will redo internally.
from models.GAT_biological_prior_knowledge import _map_genes_to_markers

from models.Linear_transformation import *
from models.Nelder_Mead import *
from models.Bayesian_optimisation import *

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

from hparam_specs import HPARAM_SPECS
from models.hyperparameter_tuning import (
    tune_model_hyperparameters, expand_model_list, ensemble_groups, base_of, algo_of,
)
import checkpoint_utils as _ckpt
from pipeline_utils import unify_columns_by_position


def _is_bio_prior_model(name):
    """True for 'GAT_biological_prior_knowledge' itself, or any numbered
    instance of it ('GAT_biological_prior_knowledge_1',
    '..._2', ...) - see MODEL's docstring-equivalent note in GP() for why
    there can be more than one: each instance is an independent model with
    its own HPARAMETERS[name] entry (its own network JSON / gene-location
    CSV / etc.), letting a user compare several FlashP samples (or several
    hand-uploaded networks) for the same phenotype(s) as separate models in
    one run, rather than only ever having exactly one
    GAT_biological_prior_knowledge in a given MODEL list.
    """
    return name == 'GAT_biological_prior_knowledge' or name.startswith('GAT_biological_prior_knowledge_')


def _bio_prior_model_names(model_list):
    """All bio-prior instance names actually present in `model_list`, in
    the order they appear."""
    return [m for m in model_list if _is_bio_prior_model(m)]

    
def GP(GENOTYPE_FILE_NAME, PHENOTYPE_FILE_NAME, MODEL, PHENOTYPE, RATIO, SAMPLE_NUM, HPARAMETERS, R_PATH, W_OPT, RESULT_NAME, HYPERPARAMETERS_OPT, SCENARIO, PARALLEL=None, LD_prune=None, RF_filter=None, GENOTYPE_FORMAT='csv', GENOTYPE_PLINK_PATH='plink2', OTHER_MODELS_MARKER_SOURCE='full_or_filtered', progress_callback=None, HP_TUNE=None, HP_TUNE_ENSEMBLE_MODE='per_method', MIN_DATA_POINTS=100):

    # GAT_biological_prior_knowledge always determines its own marker subset
    # from the gene-interaction network (see
    # models/GAT_biological_prior_knowledge.py) - there is no toggle for
    # this model itself; it never sees LD-pruned/RF-importance-filtered
    # data, unconditionally, regardless of what's enabled for other models.
    # The one exemption to this (requirement: "introduce an exemption...
    # when the new function is called") is internal to that model's own
    # data-driven prior-network merge feature (its params[13]) - see that
    # file's own module docstring; it does not change what GP() itself
    # routes to it here.
    #
    # OTHER_MODELS_MARKER_SOURCE is the model-independent choice for every
    # OTHER selected model instead:
    #   - 'full_or_filtered' (default): the original behaviour - other
    #     models use the full marker set, or the LD-pruned/RF-importance-
    #     filtered pool if either is enabled below.
    #   - 'gene_network': other models are instead restricted to the SAME
    #     markers GAT_biological_prior_knowledge selected via the gene
    #     network - LD pruning/RF filtering are not applied on top of this
    #     (this is an "instead of", not an additional narrowing step); see
    #     the per-task gene-window-marker computation further down. If more
    #     than one bio-prior instance is selected (see _is_bio_prior_model
    #     above), the *first* one (in MODEL's own order) is the reference
    #     network for this - a documented simplification, since "restrict
    #     other models to several different gene networks at once" has no
    #     single well-defined meaning.
    #   - 'gene_network_plus_rf' (requirement 6): like 'gene_network', but
    #     the marker pool other models are restricted to is the UNION of
    #     (a) markers included inside a gene node, and (b) markers selected
    #     by RF filtering with the top-M/top-Y% approach - i.e. exactly the
    #     marker set the reference bio-prior instance's own data-driven
    #     merge feature uses (requirement 2), whether or not any of those
    #     RF-selected markers actually ended up with a surviving data-driven
    #     edge after top_rate filtering. Requires that reference instance's
    #     own data-driven merge feature (params[13]['enabled']) to be
    #     turned on - validated per-task below, once that instance's own
    #     params for the current phenotype are known.
    # Requires GAT_biological_prior_knowledge to also be selected (its gene
    # network is what defines the marker set) - validated below.
    if OTHER_MODELS_MARKER_SOURCE not in ('full_or_filtered', 'gene_network', 'gene_network_plus_rf'):
        raise ValueError(
            f"OTHER_MODELS_MARKER_SOURCE must be 'full_or_filtered', 'gene_network', or "
            f"'gene_network_plus_rf', got {OTHER_MODELS_MARKER_SOURCE!r}"
        )
    if OTHER_MODELS_MARKER_SOURCE in ('gene_network', 'gene_network_plus_rf') and not _bio_prior_model_names(MODEL):
        raise ValueError(
            f"OTHER_MODELS_MARKER_SOURCE={OTHER_MODELS_MARKER_SOURCE!r} requires "
            f"GAT_biological_prior_knowledge to also be selected in MODEL - its gene-interaction "
            f"network is what defines the marker set every other model would be restricted to."
        )

    _other_models_selected = any(m != 'ensemble' and not _is_bio_prior_model(m) for m in MODEL)
    # LD pruning / RF importance filtering only ever apply to the
    # full/filtered pool - so they're skipped entirely (not just ignored)
    # whenever nothing would actually use that pool: either no other model
    # is selected at all, or every other model has been redirected to the
    # gene-network marker set instead.
    _other_models_use_full_pool = _other_models_selected and OTHER_MODELS_MARKER_SOURCE == 'full_or_filtered'
    LD_prune_effective = LD_prune if _other_models_use_full_pool else None
    RF_filter_effective = RF_filter if _other_models_use_full_pool else None

    # Diagnostic (prevent this from happening silently): LD_prune/RF_filter
    # were configured by the caller but end up doing NOTHING for this run,
    # because of the model-routing logic just above rather than because LD
    # pruning/RF filtering/the LD decay plot were themselves turned off.
    # Without this line, a user who enabled LD pruning (and, per this
    # feature, the LD decay plot) would see zero LD-pruning-related output
    # anywhere - no '[GP] Task .. | LD pruning ...' lines, no LD decay
    # plots, no explanation - and average_and_plot_ld_decay() would later
    # report '0 average LD decay plot(s) generated' with nothing upstream
    # in the log to explain why.
    if (LD_prune is not None or RF_filter is not None) and not _other_models_use_full_pool:
        _why = (
            "no model other than GAT_biological_prior_knowledge/ensemble is selected in MODEL"
            if not _other_models_selected else
            f"OTHER_MODELS_MARKER_SOURCE={OTHER_MODELS_MARKER_SOURCE!r} routes every other "
            f"selected model to the gene-network marker pool instead of the full/filtered one"
        )
        print(f"[GP] NOTE: LD pruning/RF filtering/LD decay plot were configured for this run, "
              f"but will have NO effect - {_why}, so there is no 'full/filtered marker pool' "
              f"for them to apply to. This is expected if that's what you intended; if not, "
              f"either select another model, or set OTHER_MODELS_MARKER_SOURCE back to "
              f"'full_or_filtered'.")

    # Create the output directory up front - the R model functions (rrBLUP/GBLUP/BayesB/RKHS)
    # and the CSV writers below all assume './Result/<RESULT_NAME>/' already exists
    os.makedirs(os.path.join('.', 'Result', RESULT_NAME), exist_ok=True)

    # ---------------------------------------------------------------------- #
    # Optional LD decay plot (Preprocess/LD_decay_plot.py) - an add-on
    # diagnostic for the LD pruning step above, entirely opt-in via
    # LD_prune['decay_plot']. Only ever considered when LD pruning itself is
    # effectively enabled (LD_prune_effective is not None) - a decay plot
    # needs the same pre-pruning training-set genotypes LD pruning itself
    # works from for a given task, so it only makes sense alongside LD
    # pruning being enabled for that same task.
    #
    # A decay curve can be requested for any subset of the three window
    # units Preprocess.LD_pruning itself supports ('kb', 'cm', 'variants') -
    # LD_prune['decay_plot']['window_units'] is a LIST, independent of
    # whatever window_unit LD pruning itself is actually using at the
    # config-schema level, so e.g. a run pruning with window_unit='kb'
    # could in principle still ask for cm- and variant-based decay curves
    # too. The GUI (main_app.py's build_ld_decay_plot_config()) always
    # sets this to a single-element list matching LD pruning's own
    # 'Window unit' setting exactly - a decay plot's whole point is to
    # help judge the window/r^2 threshold LD pruning is actually using,
    # which only makes sense expressed in that same unit - but a
    # hand-built config (headless use) can still request a different, or
    # multiple, unit(s) if that's genuinely useful for a particular
    # analysis.
    # ---------------------------------------------------------------------- #
    _decay_cfg = LD_prune_effective.get('decay_plot') if LD_prune_effective is not None else None
    _decay_plot_enabled = bool(_decay_cfg and _decay_cfg.get('enabled'))

    # Diagnostic (prevent this from happening silently): report the LD
    # decay plot's resolved on/off state UNCONDITIONALLY, every run - not
    # just when it ends up enabled - so it's always immediately obvious
    # from the log whether/why it will run, rather than requiring anyone to
    # infer it from the absence of other messages. This specifically covers
    # the case of a sequential_config.json/step1_config.json/
    # step2_config.json saved (e.g. via the GUI's 'Generate and save job
    # files') BEFORE this feature existed: such a file's LD_PRUNE dict has
    # no 'decay_plot' key at all, which is otherwise indistinguishable from
    # simply never having asked for the feature - regenerating/resaving the
    # config from the current GUI (or just using 'Run pipeline' directly,
    # which always gathers a fresh config) is required to pick it up.
    if LD_prune is None:
        print("[GP] LD decay plot: not requested (LD pruning itself is not configured for "
              "this run - LD_prune=None).")
    elif LD_prune_effective is None:
        pass  # already explained above by the 'NOTE: LD pruning/RF filtering/LD decay plot ...' message
    elif _decay_cfg is None:
        print("[GP] LD decay plot: LD pruning is configured, but its config has no "
              "'decay_plot' entry at all - the LD decay plot will NOT be generated. If you "
              "expected it to run: a config saved via 'Generate and save job files' BEFORE "
              "this feature was added to the GUI won't have this key - regenerate/resave that "
              "config file (or use 'Run pipeline' directly, which always builds a fresh "
              "config) after checking 'Generate LD decay plots' on the LD pruning tab.")
    elif not _decay_cfg.get('enabled'):
        print("[GP] LD decay plot: configured but not enabled (LD_prune['decay_plot']"
              "['enabled'] is False) - the LD decay plot will NOT be generated for this run.")

    _decay_snp_info = None
    _decay_window_units = []
    if _decay_plot_enabled:
        # Backward compatible with a config saved by an earlier, kb-only
        # version of this feature (which had no 'window_units' key at
        # all) - default to ['kb'] in that case, exactly matching that
        # version's only-ever-kb behaviour.
        _requested_units = _decay_cfg.get('window_units') or ['kb']
        _invalid_units = [u for u in _requested_units if u not in LD_DECAY_WINDOW_UNITS]
        _requested_units = [u for u in _requested_units if u in LD_DECAY_WINDOW_UNITS]
        if _invalid_units:
            print(f"[GP] LD decay plot: ignoring invalid window unit(s) {_invalid_units} in "
                  f"'decay_plot.window_units' - must be a subset of {LD_DECAY_WINDOW_UNITS}.")
        if not _requested_units:
            print("[GP] LD decay plot: 'window_units' has no valid entries - the LD decay "
                  "plot will NOT be generated for this run.")
            _decay_plot_enabled = False
        else:
            _decay_snp_info = resolve_snp_info_for_decay(LD_prune_effective)
            _needs_snp_info = [u for u in _requested_units if u in ('kb', 'cm')]
            if _needs_snp_info and _decay_snp_info is None:
                print(f"[GP] LD decay plot: no SNP info (CHR/POS/CM map) is configured for LD "
                      f"pruning, so window unit(s) {_needs_snp_info} can't be computed (they "
                      f"need real marker positions/genetic distances) - 'variants' doesn't "
                      f"need one, but {_needs_snp_info} do.")
                _requested_units = [u for u in _requested_units if u not in ('kb', 'cm')]
            _decay_window_units = _requested_units
            _decay_plot_enabled = bool(_decay_window_units)
            if not _decay_plot_enabled:
                print("[GP] LD decay plot: no window unit(s) remain usable for this run "
                      "(see message(s) above) - the LD decay plot will NOT be generated.")

    if _decay_plot_enabled:
        _decay_frequency = max(1, int(_decay_cfg.get('frequency', 1)))
        _decay_subfolder = _decay_cfg.get('subfolder') or 'LD_decay_plots'
        _decay_max_pairs_per_chr = int(_decay_cfg.get('max_pairs_per_chr', 20000))
        # Per-window-unit max_distance/bin_width - e.g.
        # decay_plot['kb'] = {'max_distance':.., 'bin_width':..}. Falls
        # back to Preprocess.LD_decay_plot's own documented defaults for
        # any unit whose settings weren't explicitly configured (also
        # covers a kb-only legacy config, which has no 'kb'/'cm'/
        # 'variants' sub-dicts at all - see 'window_units' fallback
        # above - by falling back to the OLD flat
        # max_distance_kb/bin_width_kb keys first, then the module's
        # defaults, so an old saved config's kb settings still apply
        # unchanged).
        _decay_settings = {}
        for _unit in _decay_window_units:
            _unit_cfg = _decay_cfg.get(_unit) or {}
            _legacy_max = _decay_cfg.get(f'max_distance_{_unit}')
            _legacy_bin = _decay_cfg.get(f'bin_width_{_unit}')
            _decay_settings[_unit] = {
                'max_distance': float(_unit_cfg.get('max_distance', _legacy_max) or LD_DECAY_DEFAULT_MAX_DISTANCE[_unit]),
                'bin_width': float(_unit_cfg.get('bin_width', _legacy_bin) or LD_DECAY_DEFAULT_BIN_WIDTH[_unit]),
            }
        _decay_plot_dir = os.path.join('.', 'Result', RESULT_NAME, _decay_subfolder)
        _decay_data_dir = os.path.join(_decay_plot_dir, 'data')
        os.makedirs(_decay_data_dir, exist_ok=True)
        # 'Keep the per-scenario LD decay data CSVs after averaging, or
        # discard them once the average(s) have been computed to save
        # disk space?' - see Preprocess.LD_decay_plot's module docstring
        # ('KEEPING OR DISCARDING THE PER-SCENARIO "LOG"'). Recorded to a
        # tiny marker file (not just kept in this function's local
        # variables) because the actual cleanup happens in
        # average_and_plot_ld_decay(), which for the Parallel workflow
        # runs from Step 2 - a separate process that never sees this
        # LD_PRUNE config at all. Default True (keep everything) so an
        # older saved config with no 'keep_log' key behaves exactly as
        # before this option existed.
        _decay_keep_log = bool(_decay_cfg.get('keep_log', True))
        write_keep_log_marker(_decay_plot_dir, _decay_keep_log)
        print(f"[GP] LD decay plot enabled for window unit(s) {_decay_window_units}: a plot "
              f"will be generated every {_decay_frequency} scenario(s) that LD pruning runs "
              f"for, saved under '{_decay_plot_dir}' "
              f"({'keeping' if _decay_keep_log else 'discarding after averaging'} the "
              f"per-scenario data CSVs).")

    # Requirement: LD decay is a property of the genotypes (which
    # individuals end up in a scenario's training set), not of the
    # phenotype being predicted - GP()'s own train_test_split calls below
    # are keyed only on population/ratio/replicate ('sample'), never on
    # phenotype, so every phenotype sharing a given (population, ratio,
    # replicate) combination has an IDENTICAL training-set genotype and
    # would produce an identical decay curve. A decay sample is therefore
    # taken (or, per the configured frequency, deliberately skipped) at
    # most ONCE per (population, ratio, replicate) combination - never
    # re-sampled for a second, third, ... phenotype that happens to share
    # it - both cutting the amount of data stored and making
    # Preprocess.LD_decay_plot.average_and_plot_ld_decay()'s later
    # per-population averaging correctly reflect only genuinely distinct
    # genotype samples (see that module's own docstring).
    _ld_decay_seen_keys = set()
    _ld_decay_call_count = 0

    def _ld_decay_due(key):
        """Whether THIS (population, ratio, replicate) `key` should have
        an LD decay sample generated. Decided ONCE per key - every
        subsequent task sharing that same key (necessarily a different
        phenotype - population/ratio/replicate together already uniquely
        identify a task within a single phenotype) always returns False,
        since the training-set genotypes (and therefore the decay curve)
        would be identical to whatever was already sampled - or
        deliberately skipped, per the configured frequency - the first
        time this key was seen. Kept separate from the actual plot-
        generation work below so the (potentially expensive, for the
        native-PLINK path - see that branch's own call site) pre-pruning
        genotype materialisation only ever happens on calls that are
        actually due."""
        nonlocal _ld_decay_call_count
        if not _decay_plot_enabled:
            return False
        if key in _ld_decay_seen_keys:
            return False
        _ld_decay_seen_keys.add(key)
        _ld_decay_call_count += 1
        return _ld_decay_call_count % _decay_frequency == 0

    def _save_ld_decay_plot(genotype_supplier, population_label, sample_label, ratio_label):
        """Requirements 1-5: compute this (population, ratio, replicate)
        combination's LD decay curve(s) - one per requested window unit
        (kb/cm/variants) - from its own (pre-pruning) training-set
        genotypes, and save both the plot (PNG) and the data used to draw
        it (CSV) under Result/<RESULT_NAME>/<subfolder>/ for each. Only
        ever called once _ld_decay_due() has already returned True for
        this task.

        genotype_supplier : zero-arg callable returning the PRE-pruning
            training-set genotype DataFrame (marker columns only) for this
            scenario - called lazily, right here, ONCE (reused for every
            requested window unit), so the native-PLINK path's extra
            genotype materialisation (see that branch's own comment) only
            ever happens once per due scenario, however many window units
            were requested.

        Requirement 7 ("work without errors, sequential and parallel"):
        never raises - a diagnostic plot failing to render must never take
        down an entire prediction run. Any error (including materialising
        the genotype itself) is caught, logged, and this task's actual
        prediction work continues exactly as if the plot had simply been
        skipped.
        """
        try:
            pre_prune_genotype_df = genotype_supplier()
        except Exception as exc:
            print(f"[GP] WARNING: could not materialise genotypes for the LD decay plot "
                  f"(population={population_label}, sample={sample_label}): {exc!r}. "
                  f"Continuing without it.")
            return

        for window_unit in _decay_window_units:
            try:
                settings = _decay_settings[window_unit]
                decay_df = compute_ld_decay_data(
                    pre_prune_genotype_df, _decay_snp_info, window_unit=window_unit,
                    max_distance=settings['max_distance'], bin_width=settings['bin_width'],
                    max_pairs_per_chr=_decay_max_pairs_per_chr,
                )
                # Requirement 4: filename combines population + sampling
                # number (plus the window unit, so kb/cm/variants curves
                # for the same combination never collide on one name) -
                # NOT phenotype, since this sample already represents
                # every phenotype sharing this (population, ratio,
                # replicate) combination (see the dedup note above), not
                # any one specific phenotype. unique_path() guards the
                # rare case where population + sample still doesn't
                # uniquely identify a single combination on its own (e.g.
                # more than one split RATIO configured for the 'within'
                # scenario reuses the same replicate numbers for every
                # ratio) rather than silently overwriting an earlier
                # combination's files.
                name_stub = (
                    f"LDdecay_{window_unit}_{sanitize_for_filename(population_label)}_"
                    f"{sanitize_for_filename(sample_label)}"
                )
                csv_path = unique_path(os.path.join(_decay_data_dir, f'{name_stub}.csv'))
                png_path = unique_path(os.path.join(_decay_plot_dir, f'{name_stub}.png'))
                # Requirement 6: window_unit/population metadata is what
                # lets average_and_plot_ld_decay() (called once, after
                # every prediction scenario has finished - see
                # run_sequential.py / run_step2_assemble.py) group this
                # combination's data with every other replicate/ratio
                # sharing the same window_unit+population (and never mix
                # it with a different window unit's or population's data).
                save_ld_decay_data(decay_df, csv_path, metadata={
                    'window_unit': window_unit, 'population': population_label,
                    'ratio': ratio_label, 'sample': sample_label,
                })
                plotted = plot_ld_decay(
                    decay_df, png_path,
                    title=(f'LD decay ({window_unit}) - population={population_label}, '
                           f'sample={sample_label}'),
                    window_unit=window_unit,
                )
                if plotted:
                    print(f"[GP] LD decay plot ({window_unit}) saved: {png_path} (data: {csv_path})")
                else:
                    print(f"[GP] LD decay plot ({window_unit}) skipped for this scenario (no "
                          f"marker pair within {settings['max_distance']} {window_unit} had both "
                          f"a usable position) - the (empty) data file was still written to "
                          f"{csv_path} for the record.")
            except Exception as exc:
                print(f"[GP] WARNING: LD decay plot ({window_unit}) generation failed for this "
                      f"scenario (population={population_label}, sample={sample_label}): "
                      f"{exc!r}. Continuing without it.")

    # PLINK2 subprocess working files (Preprocess/plink_io.py's genotype
    # conversion/native LD pruning) are NOT persisted anywhere in the
    # Result folder - every call below omits work_dir (or passes None),
    # which is each function's documented signal to create its own
    # temporary directory and clean it up automatically once that call
    # finishes. Nothing under Result/<RESULT_NAME>/ is used for this.
    def _new_plink_work_dir():
        return None

    # Import R modules
    if R_PATH != None:
        os.environ['R_HOME'] = R_PATH
    
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    
    r_source = robjects.r['source']
    r_source('./models/rrBLUP.R')
    rrBLUP = robjects.globalenv['rrBLUP']
    r_source('./models/GBLUP.R')
    GBLUP = robjects.globalenv['GBLUP']
    r_source('./models/BayesB.R')
    BayesB = robjects.globalenv['BayesB']
    r_source('./models/RKHS.R')
    RKHS = robjects.globalenv['RKHS']

    def _call_model(model_name, train, valid, test, params, bio_prior_ctx=None):
        """Single source of truth for 'given a BASE model name (never an
        algorithm- or instance-suffixed one) + its HPARAMETERS-shaped params
        list, run it and return a uniform result dict.' Used by the
        hyperparameter tuner (tune_model_hyperparameters, called from the
        main per-task dispatch loop below).

        This is a separate implementation - NOT a refactor of the main
        loop's own per-model dispatch - so both need updating together if a
        model's own function signature ever changes. Every model the main
        dispatch supports is supported here too, including
        GAT_biological_prior_knowledge: its branch below mirrors the main
        dispatch's own bio_prior branch (task-level marker pool resolution,
        PLINK marker-info override, phenotype/network metadata check).

        bio_prior_ctx : dict or None
            Task-level context GAT_biological_prior_knowledge's branch below
            needs (bio_prior_pools, train_unpruned/valid_unpruned/
            test_unpruned, bio_prior_merge_pools, full_marker_pool,
            bim_marker_info_path, network_cache) - passed in EXPLICITLY by
            the caller (_process_one_task, via the run_model_fn lambda
            given to tune_model_hyperparameters) rather than referenced via
            closure, since THIS function is defined once at GP()'s own
            top level (a sibling of _process_one_task, not nested inside
            it) and so cannot see _process_one_task's own local variables
            through ordinary Python closure rules - only the lambda that
            calls this function is itself nested inside _process_one_task,
            and can correctly close over them; bundling them into this one
            dict is what lets that lambda hand them across cleanly. Only
            ever None for a non-bio-prior model_name (every other branch
            below ignores this parameter entirely).

        params is used exactly as given - the caller decides whether 'all'
        substitutions/explainability flags are the user's real settings (the
        final confirmatory fit) or the tuner's cheapened search-time ones.
        """
        params = list(params)
        _scoring_valid = valid  # overridden below only by the bio_prior branch, which resolves its own pool

        if model_name == 'rrBLUP':
            result = rrBLUP(train, valid, test, params, RESULT_NAME)
            out = dict(pearson_test=result['r_pearson'][0], mse_test=result['r_MSE'][0],
                       effect=result['r_effect'], predicted_test=result['r_y_predicted'],
                       predicted_valid=result['r_y_predicted_valid'], predicted_train=result['r_y_predicted_train'])

        elif model_name == 'GBLUP':
            if params[3] == 'all':
                params[3] = test.shape[0]
            result = GBLUP(train, valid, test, params, RESULT_NAME)
            out = dict(pearson_test=result['r_pearson'][0], mse_test=result['r_MSE'][0],
                       effect=result['r_effect'], predicted_test=result['r_y_predicted'],
                       predicted_valid=result['r_y_predicted_valid'], predicted_train=result['r_y_predicted_train'])

        elif model_name == 'BayesB':
            result = BayesB(train, valid, test, params, RESULT_NAME)
            out = dict(pearson_test=result['r_pearson'][0], mse_test=result['r_MSE'][0],
                       effect=result['r_effect'], predicted_test=result['r_y_predicted'],
                       predicted_valid=result['r_y_predicted_valid'], predicted_train=result['r_y_predicted_train'])

        elif model_name == 'RKHS':
            if params[4] == 'all':
                params[4] = test.shape[0]
            result = RKHS(train, valid, test, params, RESULT_NAME)
            out = dict(pearson_test=result['r_pearson'][0], mse_test=result['r_MSE'][0],
                       effect=result['r_effect'], predicted_test=result['r_y_predicted'],
                       predicted_valid=result['r_y_predicted_valid'], predicted_train=result['r_y_predicted_train'])

        elif model_name == 'RF':
            if params[6] == 'all':
                params[6] = test.shape[0]
            r, mse, effect, interaction, pt, pv, ptr = RF(train, valid, test, params)
            out = dict(pearson_test=r, mse_test=mse, effect=effect, interaction=interaction,
                       predicted_test=pt, predicted_valid=pv, predicted_train=ptr)

        elif model_name == 'SVR':
            if params[7] == 'all':
                params[7] = test.shape[0]
            r, mse, effect, pt, pv, ptr = SV_Regression(train, valid, test, params)
            out = dict(pearson_test=r, mse_test=mse, effect=effect,
                       predicted_test=pt, predicted_valid=pv, predicted_train=ptr)

        elif model_name == 'KNN':
            if params[4] == 'all':
                params[4] = test.shape[0]
            r, mse, effect, pt, pv, ptr = KNN(train, valid, test, params)
            out = dict(pearson_test=r, mse_test=mse, effect=effect,
                       predicted_test=pt, predicted_valid=pv, predicted_train=ptr)

        elif model_name == 'MLP':
            if params[7] == 'all':
                params[7] = test.shape[0]
            r, mse, effect, pt, pv, ptr = ML_Perceptron(train, valid, test, params)
            out = dict(pearson_test=r, mse_test=mse, effect=effect,
                       predicted_test=pt, predicted_valid=pv, predicted_train=ptr)

        elif model_name == 'GAT_infinitesimal_node_level':
            if params[-2] == 'all':
                params[-2] = test.shape[0]
            r, mse, effect, pt, pv, ptr = GAT_infinitesimal_node_level(train, valid, test, params)
            out = dict(pearson_test=r, mse_test=mse, effect=effect,
                       predicted_test=pt, predicted_valid=pv, predicted_train=ptr)

        elif model_name == 'GAT_infinitesimal':
            if params[-1] == 'all':
                params[-1] = test.shape[0]
            r, mse, effect, pt, pv, ptr = GAT_infinitesimal(train, valid, test, params)
            out = dict(pearson_test=r, mse_test=mse, effect=effect,
                       predicted_test=pt, predicted_valid=pv, predicted_train=ptr)

        elif model_name == 'GAT_fully_connected':
            if params[-1] == 'all':
                params[-1] = test.shape[0]
            r, mse, effect, pt, pv, ptr, attn = GAT_fully_connected(train, valid, test, params)
            out = dict(pearson_test=r, mse_test=mse, effect=effect, attention=attn,
                       predicted_test=pt, predicted_valid=pv, predicted_train=ptr)

        elif model_name == 'GAT_prior_knowledge':
            if params[-2] == 'all':
                params[-2] = test.shape[0]
            r, mse, effect, pt, pv, ptr, attn = GAT_prior_knowledge(train, valid, test, params)
            out = dict(pearson_test=r, mse_test=mse, effect=effect, attention=attn,
                       predicted_test=pt, predicted_valid=pv, predicted_train=ptr)

        elif _is_bio_prior_model(model_name):
            # Requirement (bugfix): bio_prior_ctx bundles everything this
            # branch needs that ISN'T reachable via ordinary closure from
            # here (see this function's own docstring above for exactly
            # why) - GENOTYPE_FORMAT/RESULT_NAME (GP()'s own parameters)
            # and sample/i (the driving loop's own variables, also at
            # GP()'s top level) genuinely ARE reachable via closure as
            # normal, so only the true task-local pieces are bundled.
            if bio_prior_ctx is None:
                raise RuntimeError(
                    f"_call_model: bio_prior_ctx is required for bio-prior model {model_name!r} "
                    f"but was not provided by the caller."
                )
            bio_prior_pools_ctx = bio_prior_ctx['bio_prior_pools']
            train_unpruned_ctx = bio_prior_ctx['train_unpruned']
            valid_unpruned_ctx = bio_prior_ctx['valid_unpruned']
            test_unpruned_ctx = bio_prior_ctx['test_unpruned']
            bio_prior_merge_pools_ctx = bio_prior_ctx['bio_prior_merge_pools']
            full_marker_pool_ctx = bio_prior_ctx['full_marker_pool']
            bim_marker_info_path_ctx = bio_prior_ctx['bim_marker_info_path']
            network_cache_ctx = bio_prior_ctx.get('network_cache')

            # Deliberately ignores the train/valid/test PARAMETERS passed
            # into this function (unlike every other branch above) - this
            # model always resolves its own pool instead, since its
            # candidate markers come from its own gene network rather than
            # whatever LD pruning/RF filtering produced for other models
            # this task. base_params (and therefore `params` here) is
            # always a flat per-model list by this point even if
            # HPARAMETERS[model_name] is configured as a per-phenotype dict
            # in the GUI - the per-task dispatch loop below resolves which
            # phenotype's list to pass in before ever calling the tuner
            # (see its own comment on this), so this function never has to
            # know about that distinction itself.
            bio_prior_params = list(params)

            bio_train, bio_valid, bio_test = bio_prior_pools_ctx.get(model_name, (train_unpruned_ctx, valid_unpruned_ctx, test_unpruned_ctx)) \
                if GENOTYPE_FORMAT == 'plink' else (train_unpruned_ctx, valid_unpruned_ctx, test_unpruned_ctx)

            _merge_source_data = bio_prior_merge_pools_ctx.get(model_name) if GENOTYPE_FORMAT == 'plink' else full_marker_pool_ctx

            if bio_prior_params[-1] == 'all':
                bio_prior_params[-1] = bio_test.shape[0]

            if GENOTYPE_FORMAT == 'plink':
                bio_prior_params[9] = bim_marker_info_path_ctx

            try:
                matched, declared = phenotype_matches_network_metadata(
                    bio_prior_params[7], sample.loc[i, 'phenotype']
                )
                if matched is False:
                    print(
                        f"[{model_name}] WARNING: '{bio_prior_params[7]}' "
                        f"declares itself to be for phenotype {declared!r}, which doesn't "
                        f"obviously match the phenotype {sample.loc[i,'phenotype']!r} it's "
                        f"about to be used for. This is a heuristic name check, not proof of "
                        f"a mistake - double-check this is the right file for this trait."
                    )
            except Exception:
                pass  # never let this best-effort check block a real run

            r, mse, effect, pt, pv, ptr, attn = GAT_biological_prior_knowledge(
                bio_train, bio_valid, bio_test, bio_prior_params, RESULT_NAME, sample.loc[i, 'phenotype'], model_name,
                merge_source_data=_merge_source_data, network_cache=network_cache_ctx,
            )
            out = dict(pearson_test=r, mse_test=mse, effect=effect, attention=attn,
                       predicted_test=pt, predicted_valid=pv, predicted_train=ptr)
            # Scored against bio_valid (this model's own resolved validation
            # pool), not the generic `valid` parameter - see the scoring
            # block below, which is told to use this via _scoring_valid.
            _scoring_valid = bio_valid

        else:
            raise ValueError(f"_call_model: unrecognised model {model_name!r}")

        if _scoring_valid.shape[0] != 0 and len(out.get('predicted_valid', [])) > 0:
            actual_valid = _scoring_valid.iloc[:, -1].values.tolist()
            out['pearson_valid'] = pearsonr(actual_valid, out['predicted_valid'])[0]
            out['mse_valid'] = mean_squared_error(actual_valid, out['predicted_valid'])
        else:
            out['pearson_valid'] = None
            out['mse_valid'] = None

        return out


    # Read genotype and phenotype data.
    #
    # GENOTYPE_FORMAT == 'csv' (default): completely unchanged from before -
    # the whole genotype CSV is loaded up front, exactly as it always has
    # been.
    #
    # GENOTYPE_FORMAT == 'plink': GENOTYPE_FILE_NAME is a PLINK bed/bim/fam
    # file STEM (e.g. 'mydata' for 'mydata.bed'/'mydata.bim'/'mydata.fam'),
    # not a CSV path. The genotype matrix itself is *never* loaded here -
    # only the .bim file's marker positions (cheap: no genotype values
    # touched at all) - so a run with only GAT_biological_prior_knowledge
    # selected never has to materialise more than the specific markers that
    # actually fall inside a gene's window, and a run with LD pruning
    # enabled never has to materialise the pre-pruning genotype matrix at
    # all. Both 'ID' and 'population' come from the phenotype file (a .fam
    # file has neither) - see Preprocess/plink_io.py's module docstring.
    # The actual per-sample, per-marker genotype DataFrames are built
    # per-task, inside the main loop below.
    data_phenotype_original = pd.read_csv(PHENOTYPE_FILE_NAME)
    # Requirement 8: the phenotype file's first two columns are always
    # 'ID' and 'population' BY POSITION, regardless of what header text
    # the file actually uses for them - unify internally so the rest of
    # GP() can always rely on those exact names. Every column AFTER
    # these two is left completely untouched - those are phenotype TRAIT
    # names, which are meaningful, user-chosen values (e.g. 'days2anthesis')
    # that must never be renamed.
    data_phenotype_original = unify_columns_by_position(
        data_phenotype_original, ['ID', 'population'], 'phenotype file', first_n=2
    )
    # Normalise 'ID' to string dtype right away. Preprocess/plink_io.py
    # always returns string IDs (a .fam file's IID column is read as text),
    # so for GENOTYPE_FORMAT == 'plink', a phenotype file whose 'ID' column
    # happens to look purely numeric (e.g. 1001, 1002) would otherwise be
    # read by pandas as int64 - merging that against plink_io's string IDs
    # then fails with "You are trying to merge on object and int64 columns"
    # (pandas refuses to silently coerce). Casting both sides to string here
    # - unconditionally, for both genotype formats, so the CSV path stays
    # internally consistent with itself too - makes ID matching robust to
    # this regardless of what the IDs happen to look like, with no change
    # in which samples match (a numeric ID and its string form identify the
    # exact same sample either way).
    data_phenotype_original['ID'] = data_phenotype_original['ID'].astype(str)

    if GENOTYPE_FORMAT == 'csv':
        data_genotype_original = pd.read_csv(GENOTYPE_FILE_NAME)
        # Requirement 8: same positional unification as the phenotype file
        # above - 'ID'/'population' by position, every marker column after
        # them left exactly as-is (marker names are meaningful, user-chosen
        # identifiers, never renamed).
        data_genotype_original = unify_columns_by_position(
            data_genotype_original, ['ID', 'population'], 'genotype file', first_n=2
        )
        data_genotype_original['ID'] = data_genotype_original['ID'].astype(str)
        POPULATION = pd.unique(data_genotype_original['population'])
        bim_marker_info = None
        fam_iid_set = None
    elif GENOTYPE_FORMAT == 'plink':
        validate_plink_fileset(GENOTYPE_FILE_NAME)
        data_genotype_original = None
        POPULATION = pd.unique(data_phenotype_original['population'])
        bim_marker_info = read_bim_marker_info(f"{GENOTYPE_FILE_NAME}.bim")
        # Requirement: merge genotype and phenotype BEFORE splitting into
        # train/valid/test, not per-split afterwards - see
        # _process_one_task()'s own use of this, right after its
        # phenotype-NaN dropna. Reading just the .fam file's ID column
        # (not the - potentially far larger - genotype matrix itself) is
        # cheap enough to do once, up front, and reuse for every task.
        fam_iid_set = set(read_fam_iids(f"{GENOTYPE_FILE_NAME}.fam"))
    else:
        raise ValueError(f"GENOTYPE_FORMAT must be 'csv' or 'plink', got {GENOTYPE_FORMAT!r}")

    # GAT_biological_prior_knowledge's own marker_info_path (whatever the
    # user configured in its hyperparameters) is only meaningful for
    # GENOTYPE_FORMAT == 'csv'. For 'plink', marker positions can only
    # correctly come from the SAME .bim file that any extracted genotype
    # columns were pulled from - using a different, hand-supplied
    # marker_info.csv here would silently desynchronise marker names from
    # their true positions. So for 'plink', the .bim-derived marker_info is
    # written once - shared by every bio-prior instance, since they all
    # read from the same underlying .bim regardless of which network each
    # is paired with - and every GAT_biological_prior_knowledge* call this
    # run has its own params[9] (marker_info_path) OVERRIDDEN to point at
    # it, regardless of whatever was configured in the GUI - see the
    # dispatch branch below.
    bim_marker_info_path = None
    if GENOTYPE_FORMAT == 'plink' and _bio_prior_model_names(MODEL):
        bim_marker_info_path = os.path.join('.', 'Result', RESULT_NAME, '_plink_bim_marker_info.csv')
        os.makedirs(os.path.dirname(bim_marker_info_path), exist_ok=True)
        bim_marker_info.to_csv(bim_marker_info_path, index=False, encoding='utf-8')

    if (type(PHENOTYPE) is not list) and (PHENOTYPE == 'all'):
        PHENOTYPE = list(data_phenotype_original.columns[2:])
    
    # Create the total number of combinations of prediction scenarios
    if SCENARIO == 'within':
        sample = pd.DataFrame({'population':[item for item in POPULATION for i in range(SAMPLE_NUM*len(PHENOTYPE)*len(RATIO))],
                                'phenotype':[item for item in PHENOTYPE for i in range(SAMPLE_NUM*len(RATIO))]*len(POPULATION),
                                'ratio':[item for item in RATIO for i in range(SAMPLE_NUM)]*len(POPULATION)*len(PHENOTYPE),
                                'sample':list(range(1,SAMPLE_NUM+1)) *len(PHENOTYPE)*len(POPULATION)*len(RATIO)})
    elif SCENARIO == 'between':
        if W_OPT is None:
            comb = [str(x)+'->'+str(y) for x in POPULATION for y in POPULATION]
            sample = pd.DataFrame({'population':[item for item in comb] *len(PHENOTYPE),
                                    'phenotype':[item for item in PHENOTYPE for i in range(len(POPULATION))]*len(POPULATION),
                                    'ratio':[-1]*len(POPULATION)*len(PHENOTYPE)*len(POPULATION),
                                    'sample':[-1] *len(PHENOTYPE)*len(POPULATION)*len(POPULATION)})
            tmp = sample['population'].str.split('->',expand=True)
            sample = sample[tmp[0]!=tmp[1]].reset_index(drop=True)
        else:
            comb = [str(x)+'->'+str(y)+'->'+str(z) for x in POPULATION for y in POPULATION for z in POPULATION]
            sample = pd.DataFrame({'population':[item for item in comb] *len(PHENOTYPE),
                                    'phenotype':[item for item in PHENOTYPE for i in range(len(POPULATION))]*len(POPULATION)*len(POPULATION),
                                    'ratio':[-1]*len(POPULATION)*len(PHENOTYPE)*len(POPULATION)*len(POPULATION),
                                    'sample':[-1] *len(PHENOTYPE)*len(POPULATION)*len(POPULATION)*len(POPULATION)})
            tmp = sample['population'].str.split('->',expand=True)
            sample = sample[(tmp[0]!=tmp[1]) & (tmp[0]!=tmp[2]) & (tmp[1]!=tmp[2])].reset_index(drop=True)

    record = pd.DataFrame()          #store performance metrics
    result_train = pd.DataFrame()    #store predicted phenotypes for train set
    result_valid = pd.DataFrame()    #store predicted phenotypes for validation set
    result_test = pd.DataFrame()     #store predicted phenotypes for test set
    effect = pd.DataFrame()          #store genomic marker effects
    interactions = pd.DataFrame()    #store marker interaction effects
    weight = pd.DataFrame()          #store weight values for weight optimisation
    attention_total = pd.DataFrame() #store attention values  GAT models
    hp_record = pd.DataFrame()       #store optimised hyperparameters found via HP_TUNE
    stats = pd.DataFrame()           #store basic per-scenario statistics (split sizes, marker count)

    # MODEL_BASE is the user's original, unsuffixed model selection - used
    # for everything that must key off a model's TYPE regardless of tuning
    # (HP_TUNE lookups, the ensemble-grouping helpers). MODEL_RUN is what the
    # per-task dispatch loop below actually iterates: identical to
    # MODEL_BASE unless a model is tuned with more than one algorithm, in
    # which case it's expanded into one suffixed entry per algorithm (e.g.
    # 'RF' -> 'RF__Grid', 'RF__Bayesian') - see
    # models.hyperparameter_tuning.expand_model_list. When HP_TUNE is None
    # (the default), MODEL_RUN == MODEL_BASE exactly, so every existing
    # caller that doesn't pass HP_TUNE sees no change here at all.
    MODEL_BASE = MODEL
    MODEL_RUN = expand_model_list(MODEL_BASE, HP_TUNE)

    if PARALLEL is not None:
        idx = PARALLEL['batch_id']
        interval = PARALLEL['batch_size']
    else:
        idx = 0
        interval = sample.shape[0]

    # Total number of population/phenotype/ratio/replicate combinations ('sample'
    # rows) this call will actually process. Guard against a batch_id that
    # starts beyond the end of `sample` (nothing left to do for this batch).
    total_tasks = max(0, min(interval, sample.shape[0] - idx*interval))

    # Progress is tracked per (task, model) pair rather than per task: reporting
    # only once an entire task finishes means nothing is shown until every
    # selected model (which can include slow GAT fits) has run for the very
    # first task. Ticking after each individual model gives visible movement
    # much earlier, including within the first task. LD pruning (when enabled)
    # runs once per task too, so it gets its own unit alongside each task's
    # models rather than being invisible.
    n_models = len([m for m in MODEL_RUN if m != 'ensemble']) or 1
    units_per_task = n_models + (1 if LD_prune_effective is not None else 0) + (1 if RF_filter_effective is not None else 0)
    total_units = total_tasks * units_per_task
    completed_units = 0

    # ---------------------------------------------------------------------- #
    # Checkpoint/resume (checkpoint_utils.py): if a PREVIOUS run of this
    # exact same job (same RESULT_NAME, same PARALLEL batch if any, same
    # population/phenotype/ratio/replicate scenario list) failed partway
    # through, every scenario it finished before failing was already saved
    # to disk (see the try/except around the per-task loop below) - load
    # that back in now, so those scenarios are neither lost nor re-run.
    # `PARALLEL is None` (Sequential) and each Parallel batch each get
    # their OWN independent checkpoint - see checkpoint_utils.py's module
    # docstring.
    # ---------------------------------------------------------------------- #
    _is_parallel = PARALLEL is not None
    _sample_fp = _ckpt.sample_fingerprint(sample.iloc[idx*interval: idx*interval+interval])
    _resume_last_completed_i = _ckpt.load_checkpoint(RESULT_NAME, idx, _is_parallel, _sample_fp, total_tasks)
    _static_write_flags = _ckpt.static_write_flags_for(MODEL_RUN, W_OPT)

    if _resume_last_completed_i is not None:
        _start_i = idx*interval + _resume_last_completed_i + 1
        _n_already_done = _resume_last_completed_i + 1
        print(f"[GP] Resuming from a previous checkpoint: {_n_already_done}/{total_tasks} task(s) "
              f"in this batch already completed and saved - continuing from task "
              f"{_n_already_done + 1}/{total_tasks} instead of starting over.")
        _loaded = _ckpt.load_partial_results(RESULT_NAME, idx, _is_parallel)
        record, result_train, result_valid, result_test = (
            _loaded['record'], _loaded['result_train'], _loaded['result_valid'], _loaded['result_test'],
        )
        effect, interactions, attention_total, weight, hp_record, stats = (
            _loaded['effect'], _loaded['interactions'], _loaded['attention_total'],
            _loaded['weight'], _loaded['hp_record'], _loaded['stats'],
        )
        completed_units = _n_already_done * units_per_task
    else:
        _start_i = idx*interval

    # Tracks when the previous progress report happened, so each new report
    # can show how long the just-finished step took - this is what lets you
    # read task/model durations directly off the log instead of having to
    # subtract timestamps yourself.
    last_report_time = [time.time()]

    def _report_progress(completed, total, label=None):
        if total <= 0:
            return
        now = time.time()
        elapsed = now - last_report_time[0]
        last_report_time[0] = now
        if progress_callback is not None:
            # The GUI's progress bar caption has no other timestamp source,
            # so include a clock reading here for its display.
            timestamp = datetime.now().strftime('%H:%M:%S')
            timing = f'[{timestamp}, previous step took {elapsed:.1f}s]'
            timed_label = f'{label} {timing}' if label else timing
            progress_callback(completed, total, timed_label)
        else:
            # Default: a lightweight console progress line, useful for headless
            # (HPC) runs where no GUI is available to show a progress bar. No
            # need to add our own clock reading here - the calling script
            # (e.g. run_step1_batch.py) wraps sys.stdout in a TimestampedWriter
            # that already prefixes every line with a timestamp; this just adds
            # the computed step duration, which that wrapper can't know.
            timing = f'[previous step took {elapsed:.1f}s]'
            timed_label = f'{label} {timing}' if label else timing
            print(f'[GP] Progress: {completed}/{total} steps complete ({completed/total*100:.1f}%) - {timed_label}')

    # Report the starting state immediately, before any work has been done,
    # so a 0% progress display appears right away instead of staying blank.
    _report_progress(completed_units, total_units, label='Starting...')

    def _below_min_data_points(n_points, task_i, stage_label):
        """Requirement 5: if fewer than MIN_DATA_POINTS individuals remain
        for this scenario (summed across train/valid/test) after merging
        genotype and phenotype, report it and tell the caller
        (_process_one_task) to skip this task entirely - predictions from
        very few data points aren't reliable enough to be worth computing
        (and can make some models error out outright on tiny splits).
        Returns True/False; does not itself `return` out of the task -
        this is a plain helper function, not the task loop body - the
        caller acts on the result."""
        nonlocal completed_units
        if n_points >= MIN_DATA_POINTS:
            return False
        print(f"[GP] Task {task_i - idx*interval + 1}/{total_tasks} | population {sample.loc[task_i,'population']} | "
              f"phenotype {sample.loc[task_i,'phenotype']} | ratio {sample.loc[task_i,'ratio']} | "
              f"replicate {sample.loc[task_i,'sample']} | Skipping this task - only {n_points} data "
              f"point(s) remain {stage_label} (below the configured minimum of {MIN_DATA_POINTS}).")
        completed_units += units_per_task
        _report_progress(completed_units, total_units, label='Skipped (below minimum data points)')
        return True

    # ---------------------------------------------------------------------- #
    # Checkpoint/resume (continued): everything from here through the end of
    # this task's processing is wrapped in a function (rather than being the
    # for-loop's own body directly) purely so the driving loop just below can
    # catch an exception raised anywhere inside a single task's processing,
    # roll back any partial contribution that task may already have made to
    # the accumulators (nonlocal below), save every FULLY completed task's
    # results to disk, and re-raise - see that loop's own comments for the
    # full mechanism. Nothing about the task-processing logic itself changes
    # here; this is a structural wrapper only.
    # ---------------------------------------------------------------------- #
    def _process_one_task(i):
        nonlocal record, result_train, result_valid, result_test, effect, interactions
        nonlocal weight, attention_total, hp_record, stats, completed_units

        # Requirement 11 (efficiency): ONE cache dict for this task's own
        # GAT_biological_prior_knowledge data-driven-merge results
        # (LD-pruned/RF-selected markers, pairwise-Shapley interactions) -
        # created fresh for every task (never shared across tasks - a
        # different population/phenotype/ratio/replicate genuinely needs
        # its own network), but shared by EVERY caller within this one
        # task that needs this instance's merge result: the
        # OTHER_MODELS_MARKER_SOURCE='gene_network_plus_rf' marker-pool
        # restriction below, and every call this task makes to
        # GAT_biological_prior_knowledge() itself (including once per
        # hyperparameter-tuning trial, plus the final confirmatory fit -
        # all of which need the exact same network, computed once). See
        # that function's own network_cache docstring entry for exactly
        # how each entry gets filled in and reused.
        _bio_prior_merge_cache = {}

        # Convert the data structure
        if SCENARIO == 'within':
            data_phenotype = data_phenotype_original.loc[data_phenotype_original['population']==sample.loc[i,'population'], ['ID','population',sample.loc[i,'phenotype']]].reset_index(drop=True)
        elif SCENARIO == 'between':
            if W_OPT is None:
                data_phenotype = data_phenotype_original.loc[(data_phenotype_original['population'].astype(str) ==sample.loc[i,'population'].split('->')[0]) | 
                                                             (data_phenotype_original['population'].astype(str) ==sample.loc[i,'population'].split('->')[1]), 
                                                             ['ID','population',sample.loc[i,'phenotype']]].reset_index(drop=True)
            else:
                data_phenotype = data_phenotype_original.loc[(data_phenotype_original['population'].astype(str) ==sample.loc[i,'population'].split('->')[0]) | 
                                                             (data_phenotype_original['population'].astype(str) ==sample.loc[i,'population'].split('->')[1]) |
                                                             (data_phenotype_original['population'].astype(str) ==sample.loc[i,'population'].split('->')[2]), 
                                                             ['ID','population',sample.loc[i,'phenotype']]].reset_index(drop=True)

        # Requirement (bugfix): a genotype ID can only ever correspond to
        # ONE row from here on - every downstream step (train/valid/test
        # splitting, genotype<->phenotype merging, and especially the
        # native-PLINK path's PLINK --keep + pandas merge combination)
        # assumes a strict one-to-one ID<->individual correspondence.
        # Duplicate ID rows in the phenotype file (e.g. the same
        # accession measured in more than one environment/replicate -
        # common in real multi-site trial data) silently break this: the
        # same individual can end up split across train/valid/test (or
        # duplicated within one of them by train_test_split, which
        # partitions ROWS, not unique IDs), and - for PLINK input
        # specifically - a native extraction (which only ever returns ONE
        # row per unique ID, since PLINK's own --keep de-duplicates)
        # merged back against duplicated phenotype rows produces a
        # "fan-out" join that inflates that split's row count beyond what
        # id_train/id_valid/id_test (and every model's output, keyed
        # one-to-one against those same IDs) still expects - eventually
        # surfacing as a totally opaque "All arrays must be of the same
        # length" error deep inside some downstream model. Caught and
        # fixed HERE, once, for every phenotype/genotype format, rather
        # than symptom-by-symptom further down the pipeline.
        _dup_id_mask = data_phenotype['ID'].duplicated(keep='first')
        if _dup_id_mask.any():
            _dup_ids = sorted(set(data_phenotype.loc[_dup_id_mask, 'ID'].astype(str)))
            data_phenotype = data_phenotype[~_dup_id_mask].reset_index(drop=True)
            print(f"[GP] Task {i - idx*interval + 1}/{total_tasks} | population {sample.loc[i,'population']} | "
                  f"phenotype {sample.loc[i,'phenotype']} | WARNING: {len(_dup_ids)} individual ID(s) appeared "
                  f"more than once in the phenotype file for this population (e.g. measured in more than one "
                  f"environment/replicate) - kept only the FIRST record for each and dropped the rest, since "
                  f"genomic prediction needs exactly one phenotype value per individual. If you want a "
                  f"different rule (e.g. averaging across replicates), pre-process the phenotype file "
                  f"yourself before giving it to EasiGP. Affected ID(s)"
                  f"{' (showing first 20)' if len(_dup_ids) > 20 else ''}: {_dup_ids[:20]}")

        # ld_pruning_done_natively: set True only by the GENOTYPE_FORMAT ==
        # 'plink' branch below, when it already ran PLINK2's own
        # --indep-pairwise directly on the bed/bim/fam fileset - the
        # existing LD_prune_effective block further down must then skip its
        # own (DataFrame-based) pruning call, since pruning has already
        # happened.
        ld_pruning_done_natively = False
        # True, unrestricted (every marker, every individual in this task's
        # train/valid/test split) genotype pool - set below, in whichever
        # branch actually has it. Only ever populated when something this
        # task actually needs it (an 'other' model using the full/filtered
        # pool, OTHER_MODELS_MARKER_SOURCE == 'gene_network_plus_rf', or a
        # bio-prior instance's own data-driven merge feature - see
        # need_full_pool below): stays None otherwise, since for PLINK
        # format materialising it is a real, avoidable cost. GAT_biological_
        # prior_knowledge's own bio_train/bio_valid/bio_test (always the
        # gene-network-restricted pool, per its own model contract) are a
        # SEPARATE thing from this - full_marker_pool is only ever used as
        # merge_source_data, i.e. the marker universe the merge feature's
        # own internal RF-selection step chooses from, never as what the
        # model itself trains on.
        full_marker_pool = None

        def _resolve_bio_prior_params(bp_name):
            """This bio-prior instance's flat params list for the CURRENT
            task's phenotype - handles both the flat-list and
            per-phenotype-dict HPARAMETERS[bp_name] shapes (see
            _is_bio_prior_model's own docstring note on this), so every
            caller below (both the CSV and PLINK genotype-format paths)
            shares one, single place for this lookup instead of repeating
            the isinstance(dict, ...) branch. Defined here, before the
            GENOTYPE_FORMAT split below, specifically so it's available to
            BOTH branches - it only depends on HPARAMETERS/sample/i, none
            of which are genotype-format-specific."""
            bio_prior_hparams = HPARAMETERS[bp_name]
            if isinstance(bio_prior_hparams, dict):
                resolved = bio_prior_hparams.get(sample.loc[i, 'phenotype'])
                if resolved is None:
                    raise KeyError(
                        f"HPARAMETERS[{bp_name!r}] has no entry for phenotype "
                        f"{sample.loc[i,'phenotype']!r}."
                    )
                return resolved
            return bio_prior_hparams

        def _bio_prior_merge_enabled(bp_name):
            """Requirement 2's data-driven merge config (params[13])
            'enabled' flag for this bio-prior instance, for the current
            task's phenotype. False for anything that isn't the new dict
            shape (e.g. a stale/hand-built params list that still has a
            bare string in that slot from before requirement 5's upgrade),
            rather than raising - this is only ever used to decide whether
            EXTRA work is worth doing, never required for correctness of
            the plain gene-network path. Also defined here (see
            _resolve_bio_prior_params above) so both genotype-format paths
            share it."""
            merge_cfg = _resolve_bio_prior_params(bp_name)[13]
            return isinstance(merge_cfg, dict) and bool(merge_cfg.get('enabled', False))

        if GENOTYPE_FORMAT == 'csv':
            # ---------------------------------------------------------- #
            # Completely unchanged from before.
            # ---------------------------------------------------------- #
            if SCENARIO == 'within':
                data_genotype = data_genotype_original[data_genotype_original['population']==sample.loc[i,'population']].reset_index(drop=True)
            elif SCENARIO == 'between':
                if W_OPT is None:
                    data_genotype = data_genotype_original[(data_genotype_original['population'].astype(str) == sample.loc[i, 'population'].split('->')[0]) | 
                                                           (data_genotype_original['population'].astype(str) == sample.loc[i, 'population'].split('->')[1])].reset_index(drop=True)
                else:
                    data_genotype = data_genotype_original[(data_genotype_original['population'].astype(str) == sample.loc[i, 'population'].split('->')[0]) | 
                                                           (data_genotype_original['population'].astype(str) == sample.loc[i, 'population'].split('->')[1]) |
                                                           (data_genotype_original['population'].astype(str) == sample.loc[i, 'population'].split('->')[2])].reset_index(drop=True)

            # Requirement: rather than letting rows with no matching ID on
            # one side, or missing genotype/phenotype values on either
            # side, silently (or fatally) break the downstream split/model
            # fitting, drop them here explicitly and log what happened.
            _merged_raw = data_genotype.merge(data_phenotype, on=['ID', 'population'], how='inner')
            data = _merged_raw.dropna().reset_index(drop=True)
            _n_unmatched = len(
                (set(zip(data_genotype['ID'], data_genotype['population'])) |
                 set(zip(data_phenotype['ID'], data_phenotype['population'])))
                - set(zip(_merged_raw['ID'], _merged_raw['population']))
            )
            _n_missing_values = _merged_raw.shape[0] - data.shape[0]
            if _n_unmatched > 0 or _n_missing_values > 0:
                print(f"[GP] Task {i - idx*interval + 1}/{total_tasks} | population {sample.loc[i,'population']} | "
                      f"phenotype {sample.loc[i,'phenotype']} | Dropped {_n_unmatched} individual(s) with no "
                      f"matching genotype/phenotype record and {_n_missing_values} individual(s) with missing "
                      f"genotype or phenotype value(s); {data.shape[0]} individual(s) remain.")
            if _below_min_data_points(data.shape[0], i, 'after dropping individuals with missing or unmatched genotype/phenotype information'):
                return

            if SCENARIO =='within':
                if type(sample.loc[i,'ratio']) is not tuple:
                    train, test = train_test_split(data,train_size=sample.loc[i,'ratio'], random_state=sample.loc[i,'sample'])
                    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
                    id_train, id_valid, id_test = train.iloc[:,0], pd.DataFrame(), test.iloc[:,0]
                    train, valid, test = train.iloc[:,2:], pd.DataFrame(), test.iloc[:,2:]
                elif type(sample.loc[i,'ratio']) is tuple:
                    train, valid = train_test_split(data,train_size=sample.loc[i,'ratio'][0], random_state=sample.loc[i,'sample'])
                    valid, test = train_test_split(valid,train_size=sample.loc[i,'ratio'][1]/(sample.loc[i,'ratio'][2]+sample.loc[i,'ratio'][1]), random_state=sample.loc[i,'sample'])
                    train, valid, test = train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)
                    id_train, id_valid, id_test = train.iloc[:,0], valid.iloc[:,0], test.iloc[:,0]
                    train, valid, test = train.iloc[:,2:], valid.iloc[:,2:], test.iloc[:,2:]
            elif SCENARIO =='between':
                tmp = sample.loc[i, 'population'].split('->')
                if W_OPT is None:
                    train, test = data[data['population'].astype(str)==tmp[0]], data[data['population'].astype(str)==tmp[1]]
                    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
                    id_train, id_valid, id_test = train.iloc[:,0], pd.DataFrame(), test.iloc[:,0]
                    train, valid, test = train.iloc[:,2:], pd.DataFrame(), test.iloc[:,2:] 
                elif W_OPT is not None:
                    train, valid, test = data[data['population'].astype(str)==tmp[0]], data[data['population'].astype(str)==tmp[1]], data[data['population'].astype(str)==tmp[2]]
                    train, valid, test = train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)
                    id_train, id_valid, id_test = train.iloc[:,0], valid.iloc[:,0], test.iloc[:,0]
                    train, valid, test = train.iloc[:,2:], valid.iloc[:,2:], test.iloc[:,2:]

        else:  # GENOTYPE_FORMAT == 'plink'
            # ---------------------------------------------------------- #
            # Deferred-conversion path: the bed/bim/fam genotype matrix is
            # only ever materialised (via Preprocess.plink_io) for exactly
            # the sample IDs and markers this task actually needs.
            #
            # Requirement: genotype and phenotype are matched up-front,
            # BEFORE the train/valid/test split - not per-split afterwards
            # (which risked one split ending up completely empty if none
            # of ITS assigned IDs happened to have a genotype record,
            # discovered only after the split had already been committed
            # to). Individuals with a missing value for THIS phenotype are
            # dropped first (mirroring what the CSV path's own
            # data.merge(...).dropna() already does, just earlier here,
            # since there's no single merged `data` to dropna() on this
            # path); the survivors are then intersected with the .fam
            # file's own ID list (fam_iid_set, read once outside this
            # loop - see just above where bim_marker_info is read) - an
            # individual with a phenotype record but no matching genotype
            # record (or vice versa) is dropped here, reported, and never
            # reaches the split at all. This keeps id_train/id_valid/
            # id_test (computed right below, from this same filtered
            # data_phenotype) stable for the rest of this task: nothing
            # later is allowed to drop a row for a *phenotype* or *ID-
            # matching* reason again. Missing individual GENOTYPE VALUES
            # (not missing genotype RECORDS - PLINK's own missing-call
            # encoding for markers that were called for some but not all
            # kept individuals) are a separate matter, handled without
            # dropping rows at all - see _attach_phenotype() below.
            # ---------------------------------------------------------- #
            data_phenotype = data_phenotype.dropna(subset=[sample.loc[i, 'phenotype']]).reset_index(drop=True)

            _n_before_fam_match = data_phenotype.shape[0]
            _unmatched_ids = sorted(set(data_phenotype['ID'].astype(str)) - fam_iid_set)
            if _unmatched_ids:
                data_phenotype = data_phenotype[~data_phenotype['ID'].astype(str).isin(_unmatched_ids)].reset_index(drop=True)
                print(f"[GP] Task {i - idx*interval + 1}/{total_tasks} | population {sample.loc[i,'population']} | "
                      f"phenotype {sample.loc[i,'phenotype']} | Dropped {len(_unmatched_ids)} individual(s) with "
                      f"a phenotype record but no matching ID in the PLINK fileset ({GENOTYPE_FILE_NAME}.fam) "
                      f"out of {_n_before_fam_match} candidate(s); {data_phenotype.shape[0]} individual(s) remain: "
                      f"{_unmatched_ids}")

            # Requirement (empty-phenotype / no-matched-ID safeguard):
            # nothing left to split (either every individual was missing
            # this phenotype, or none of the ones that had it also have a
            # genotype record) - skip this task entirely rather than
            # letting train_test_split() raise on an empty DataFrame, or
            # silently producing an all-empty split further down.
            if _below_min_data_points(data_phenotype.shape[0], i, 'with both a non-missing value for this phenotype and a matching genotype record'):
                return

            if SCENARIO == 'within':
                if type(sample.loc[i,'ratio']) is not tuple:
                    pheno_train, pheno_test = train_test_split(data_phenotype, train_size=sample.loc[i,'ratio'], random_state=sample.loc[i,'sample'])
                    pheno_train, pheno_test = pheno_train.reset_index(drop=True), pheno_test.reset_index(drop=True)
                    pheno_valid = pd.DataFrame()
                else:
                    pheno_train, pheno_valid = train_test_split(data_phenotype, train_size=sample.loc[i,'ratio'][0], random_state=sample.loc[i,'sample'])
                    pheno_valid, pheno_test = train_test_split(pheno_valid, train_size=sample.loc[i,'ratio'][1]/(sample.loc[i,'ratio'][2]+sample.loc[i,'ratio'][1]), random_state=sample.loc[i,'sample'])
                    pheno_train, pheno_valid, pheno_test = pheno_train.reset_index(drop=True), pheno_valid.reset_index(drop=True), pheno_test.reset_index(drop=True)
            elif SCENARIO == 'between':
                tmp = sample.loc[i, 'population'].split('->')
                if W_OPT is None:
                    pheno_train = data_phenotype[data_phenotype['population'].astype(str)==tmp[0]].reset_index(drop=True)
                    pheno_test = data_phenotype[data_phenotype['population'].astype(str)==tmp[1]].reset_index(drop=True)
                    pheno_valid = pd.DataFrame()
                else:
                    pheno_train = data_phenotype[data_phenotype['population'].astype(str)==tmp[0]].reset_index(drop=True)
                    pheno_valid = data_phenotype[data_phenotype['population'].astype(str)==tmp[1]].reset_index(drop=True)
                    pheno_test = data_phenotype[data_phenotype['population'].astype(str)==tmp[2]].reset_index(drop=True)

            phenotype_col_name = sample.loc[i, 'phenotype']
            id_train = pheno_train['ID'].astype(str).tolist()
            id_valid = pheno_valid['ID'].astype(str).tolist() if pheno_valid.shape[0] != 0 else []
            id_test = pheno_test['ID'].astype(str).tolist()

            def _attach_phenotype(geno_df, pheno_df, split_label='split'):
                if geno_df.shape[0] == 0:
                    return pd.DataFrame()
                merged = geno_df.merge(pheno_df[['ID', phenotype_col_name]], on='ID').reset_index(drop=True)
                if merged.shape[0] > pheno_df.shape[0]:
                    # A "fan-out" join - MORE rows came out than individuals
                    # went in - meaning one side had a duplicate ID for at
                    # least one individual. data_phenotype is already
                    # deduplicated by ID once, up front, for the whole task
                    # (see its own dedup step above) - this branch is a
                    # SECOND, defensive layer in case geno_df itself somehow
                    # contains a duplicate ID (e.g. an unusual PLINK
                    # fileset). Silently keeping a fanned-out split would
                    # desync it from id_train/id_valid/id_test (and every
                    # downstream array keyed one-to-one against those same
                    # IDs) - exactly the failure mode that used to surface as
                    # an opaque "All arrays must be of the same length" error
                    # deep inside some model instead - so this keeps only the
                    # first match per ID, same rule as the up-front dedup.
                    _dup_mask = merged['ID'].duplicated(keep='first')
                    print(f"[GP] Task {i - idx*interval + 1}/{total_tasks} | population {sample.loc[i,'population']} | "
                          f"phenotype {sample.loc[i,'phenotype']} | {split_label} split: WARNING - merging "
                          f"genotype with phenotype data produced {merged.shape[0]} row(s) for only "
                          f"{pheno_df.shape[0]} expected individual(s) (a duplicate ID on the genotype side) "
                          f"- kept only the first match per ID.")
                    merged = merged[~_dup_mask].reset_index(drop=True)
                elif merged.shape[0] < pheno_df.shape[0]:
                    # Requirement: an individual expected in this split but not
                    # found in the genotype data extracted from the PLINK
                    # fileset has missing genotype information - rather than
                    # treating this as a fatal internal error, drop it (the
                    # same way a missing phenotype value is already dropped,
                    # above, before the split) and log what was dropped.
                    _dropped_ids = sorted(set(pheno_df['ID'].astype(str)) - set(merged['ID'].astype(str)))
                    print(f"[GP] Task {i - idx*interval + 1}/{total_tasks} | population {sample.loc[i,'population']} | "
                          f"phenotype {sample.loc[i,'phenotype']} | {split_label} split: dropped "
                          f"{len(_dropped_ids)} individual(s) with missing genotype information (no matching "
                          f"record in the PLINK fileset extraction) out of {pheno_df.shape[0]} expected: "
                          f"{_dropped_ids}")
                if merged.shape[0] == 0:
                    return pd.DataFrame()
                # Missing GENOTYPE values (PLINK's own missing-call encoding,
                # exported as blank/NA by --export A) are imputed with each
                # marker's own mean within this split, rather than dropping
                # the individual - dropping here would silently shrink this
                # split's row count below what id_train/id_valid/id_test
                # (computed above, from the phenotype-only split) already
                # committed to, which would corrupt the later results-
                # assembly step that pairs each model's predictions with
                # those IDs one-to-one. Any marker column that turns out to
                # be entirely missing within this split (mean itself is NaN)
                # falls back to 0, since there is nothing to average.
                marker_cols = [c for c in merged.columns if c not in ('ID', 'population', phenotype_col_name)]
                if merged[marker_cols].isna().any().any():
                    merged[marker_cols] = merged[marker_cols].fillna(merged[marker_cols].mean()).fillna(0)
                return merged.drop(columns=['ID', 'population'])

            bio_prior_names_now = _bio_prior_model_names(MODEL)
            other_models_selected_now = any(m != 'ensemble' and not _is_bio_prior_model(m) for m in MODEL)

            need_gene_pool = bool(bio_prior_names_now)
            _any_bio_prior_merge_now = any(_bio_prior_merge_enabled(n) for n in bio_prior_names_now)
            # 'gene_network_plus_rf' pool restriction (requirement 6) and
            # any bio-prior instance's own data-driven merge feature
            # (requirement 2) both need the TRUE full, unrestricted marker
            # set for this task - for GENOTYPE_FORMAT == 'csv' this is
            # already free (data_genotype_original always has every
            # marker); for 'plink' it means the full-fileset extraction
            # below must run even when no OTHER model would otherwise need
            # it (e.g. only GAT_biological_prior_knowledge itself is
            # selected, with its own merge feature turned on).
            need_full_pool = (
                (other_models_selected_now and OTHER_MODELS_MARKER_SOURCE == 'full_or_filtered')
                or _any_bio_prior_merge_now
            )

            # --- Requirement: gene-window-restricted pool for every bio-prior instance ---
            # Each instance (GAT_biological_prior_knowledge, ..._2, ... - see
            # _is_bio_prior_model) has its own HPARAMETERS entry, and so its
            # own network -> its own gene-window marker set -> its own
            # extraction from the PLINK fileset. bio_prior_pools holds all of
            # them; the dispatch loop further below looks up each instance's
            # own pool by name when it actually runs that model.
            bio_prior_pools = {}
            bio_prior_gene_window_markers = {}
            bio_prior_merge_pools = {}
            # Raw, fully unrestricted PLINK extraction (every marker, no
            # --extract list) - lazily computed at most once per task, the
            # first time any bio-prior instance's own data-driven merge
            # feature actually needs it (its own RF-selection step must
            # rank among the TRUE full candidate pool, exactly like the CSV
            # path already does via train_unpruned - never among an
            # already gene-window-restricted subset, which would bias
            # which markers "win" RF selection). Shared across every
            # merge-enabled instance in this task, since it doesn't depend
            # on any particular network.
            _raw_full_pool = None

            if need_gene_pool:
                for _bp_name in bio_prior_names_now:
                    _bp_preview = _resolve_bio_prior_params(_bp_name)

                    network_preview = load_network_json(_bp_preview[7])
                    candidate_genes_preview = extract_candidate_genes(network_preview)
                    gene_list_preview, _ = build_gene_list(candidate_genes_preview, _bp_preview[8], unit=_bp_preview[10])
                    gene_to_markers_preview = _map_genes_to_markers(gene_list_preview, bim_marker_info, bim_marker_info['name'])
                    gene_window_markers = sorted({m for markers in gene_to_markers_preview for m in markers})
                    if not gene_window_markers:
                        raise RuntimeError(
                            f"No markers in the PLINK fileset fall inside any gene window for "
                            f"{_bp_name} - cannot build its gene-network marker pool."
                        )
                    bio_prior_gene_window_markers[_bp_name] = gene_window_markers

                    _extract_markers = gene_window_markers
                    if _bio_prior_merge_enabled(_bp_name):
                        if _raw_full_pool is None:
                            _raw_train_geno = plink_to_genotype_df(GENOTYPE_FILE_NAME, pheno_train, keep_iid_list=id_train, plink_path=GENOTYPE_PLINK_PATH, work_dir=_new_plink_work_dir())
                            _raw_valid_geno = plink_to_genotype_df(GENOTYPE_FILE_NAME, pheno_valid, keep_iid_list=id_valid, plink_path=GENOTYPE_PLINK_PATH, work_dir=_new_plink_work_dir()) if pheno_valid.shape[0] != 0 else pd.DataFrame()
                            _raw_test_geno = plink_to_genotype_df(GENOTYPE_FILE_NAME, pheno_test, keep_iid_list=id_test, plink_path=GENOTYPE_PLINK_PATH, work_dir=_new_plink_work_dir())
                            _raw_full_pool = (
                                _attach_phenotype(_raw_train_geno, pheno_train, split_label='train'),
                                _attach_phenotype(_raw_valid_geno, pheno_valid, split_label='valid') if pheno_valid.shape[0] != 0 else pd.DataFrame(),
                                _attach_phenotype(_raw_test_geno, pheno_test, split_label='test'),
                            )
                        bio_prior_merge_pools[_bp_name] = _raw_full_pool

                        # data_train/valid/test (what the model actually
                        # trains on, via bio_prior_pools below) must already
                        # CONTAIN every column the merge step might add as a
                        # "bare marker" node, or node-feature construction
                        # inside the model would KeyError on a missing
                        # column - so extend this instance's own extraction
                        # list with its RF-selected candidates too. This is
                        # purely about which columns get extracted here;
                        # RF SELECTION ITSELF still ranks against
                        # _raw_full_pool (passed as merge_source_data below),
                        # not this already-narrowed list, so which markers
                        # "win" is unaffected by this extension.
                        _merge_cfg_preview = _bp_preview[13]
                        _rf_x_preview, _rf_y_preview = _raw_full_pool[0].iloc[:, :-1], _raw_full_pool[0].iloc[:, -1]
                        if _merge_cfg_preview.get('ld_prune') is not None:
                            _rf_x_preview, _, _ = LD_pruning(
                                _rf_x_preview, pd.DataFrame(), _raw_full_pool[2].iloc[:, :-1],
                                _merge_cfg_preview['ld_prune']
                            )
                        _rf_selected_preview, _ = select_markers_for_data_driven_network(
                            _rf_x_preview, _rf_y_preview, _merge_cfg_preview['rf_filter']
                        )
                        # Requirement 11 (efficiency): this is the ONLY
                        # place LD pruning + RF filtering for this
                        # instance's data-driven merge should ever run for
                        # this task - cache the result so neither the
                        # OTHER_MODELS_MARKER_SOURCE='gene_network_plus_rf'
                        # restriction below, nor GAT_biological_prior_
                        # knowledge() itself (however many times it's
                        # called this task - once per hyperparameter-
                        # tuning trial, if enabled), ever redo this same,
                        # often very slow (large genotype files), work.
                        _bio_prior_merge_cache.setdefault(_bp_name, {})['rf_selected_markers'] = _rf_selected_preview
                        _extract_markers = sorted(set(gene_window_markers) | set(_rf_selected_preview))
                        print(f"[GP] Task {i - idx*interval + 1}/{total_tasks} | {_bp_name} data-driven "
                              f"merge: {len(gene_window_markers)} gene-window + {len(_rf_selected_preview)} "
                              f"RF-selected = {len(_extract_markers)} unique marker(s) to extract.")

                    gene_train_geno = plink_to_genotype_df(GENOTYPE_FILE_NAME, pheno_train, extract_snp_list=_extract_markers, keep_iid_list=id_train, plink_path=GENOTYPE_PLINK_PATH, work_dir=_new_plink_work_dir())
                    gene_valid_geno = plink_to_genotype_df(GENOTYPE_FILE_NAME, pheno_valid, extract_snp_list=_extract_markers, keep_iid_list=id_valid, plink_path=GENOTYPE_PLINK_PATH, work_dir=_new_plink_work_dir()) if pheno_valid.shape[0] != 0 else pd.DataFrame()
                    gene_test_geno = plink_to_genotype_df(GENOTYPE_FILE_NAME, pheno_test, extract_snp_list=_extract_markers, keep_iid_list=id_test, plink_path=GENOTYPE_PLINK_PATH, work_dir=_new_plink_work_dir())

                    _train_unpruned = _attach_phenotype(gene_train_geno, pheno_train, split_label='train')
                    _valid_unpruned = _attach_phenotype(gene_valid_geno, pheno_valid, split_label='valid') if pheno_valid.shape[0] != 0 else pd.DataFrame()
                    _test_unpruned = _attach_phenotype(gene_test_geno, pheno_test, split_label='test')
                    bio_prior_pools[_bp_name] = (_train_unpruned, _valid_unpruned, _test_unpruned)
                    print(f"[GP] Task {i - idx*interval + 1}/{total_tasks} | {_bp_name} "
                          f"gene-window marker pool: {len(_extract_markers)} marker(s) extracted from the "
                          f"PLINK fileset (out of {bim_marker_info.shape[0]} total).")

                # For downstream code that still wants a single reference
                # triple (the "other models restricted to gene_network" case
                # just below, and the per-task LD/RF-skip fallback) - the
                # FIRST bio-prior instance, in MODEL's own order, is that
                # reference (see OTHER_MODELS_MARKER_SOURCE's note near the
                # top of GP()). Each instance's OWN dispatch in the per-model
                # loop further below still uses its own pool from
                # bio_prior_pools, regardless of this.
                train_unpruned, valid_unpruned, test_unpruned = bio_prior_pools[bio_prior_names_now[0]]

            # --- Requirement: full/LD-pruned/RF-filtered pool for every other selected model ---
            if need_full_pool:
                if LD_prune_effective is not None:
                    task_num = i - idx*interval + 1
                    _report_progress(
                        completed_units, total_units,
                        label=(f"Task {task_num}/{total_tasks} | population {sample.loc[i,'population']} | "
                               f"phenotype {sample.loc[i,'phenotype']} | ratio {sample.loc[i,'ratio']} | "
                               f"replicate {sample.loc[i,'sample']} | LD pruning (native PLINK)")
                    )
                    ld_start_time = time.time()
                    train_full_geno = ld_prune_plink_native(GENOTYPE_FILE_NAME, pheno_train, id_train, LD_prune_effective, work_dir=_new_plink_work_dir())
                    pruned_markers = [c for c in train_full_geno.columns if c not in ('ID', 'population')]
                    valid_full_geno = plink_to_genotype_df(GENOTYPE_FILE_NAME, pheno_valid, extract_snp_list=pruned_markers, keep_iid_list=id_valid, plink_path=GENOTYPE_PLINK_PATH, work_dir=_new_plink_work_dir()) if pheno_valid.shape[0] != 0 else pd.DataFrame()
                    test_full_geno = plink_to_genotype_df(GENOTYPE_FILE_NAME, pheno_test, extract_snp_list=pruned_markers, keep_iid_list=id_test, plink_path=GENOTYPE_PLINK_PATH, work_dir=_new_plink_work_dir())
                    print(f"[GP] Task {task_num}/{total_tasks} | Native PLINK LD pruning finished: "
                          f"{bim_marker_info.shape[0]} -> {len(pruned_markers)} markers "
                          f"(took {time.time() - ld_start_time:.1f}s)")
                    completed_units += 1
                    ld_pruning_done_natively = True

                    if _ld_decay_due((sample.loc[i, 'population'], sample.loc[i, 'ratio'], sample.loc[i, 'sample'])):
                        # Requirement 7: the native-PLINK path never
                        # materialises the pre-pruning genotype matrix in
                        # Python (that's the whole point of "native" - see
                        # ld_prune_plink_native's own docstring), so an
                        # extra, otherwise-unneeded PLINK export is done
                        # here purely for the decay plot - but ONLY on
                        # calls that are actually due (frequency-gated,
                        # and deduplicated across phenotypes, above), and
                        # entirely opt-in in the first place, so this cost
                        # is bounded and user-controlled.
                        _save_ld_decay_plot(
                            lambda: plink_to_genotype_df(
                                GENOTYPE_FILE_NAME, pheno_train, keep_iid_list=id_train,
                                plink_path=GENOTYPE_PLINK_PATH, work_dir=_new_plink_work_dir(),
                            ).iloc[:, 2:],
                            sample.loc[i, 'population'], sample.loc[i, 'sample'], sample.loc[i, 'ratio'],
                        )
                else:
                    train_full_geno = plink_to_genotype_df(GENOTYPE_FILE_NAME, pheno_train, keep_iid_list=id_train, plink_path=GENOTYPE_PLINK_PATH, work_dir=_new_plink_work_dir())
                    valid_full_geno = plink_to_genotype_df(GENOTYPE_FILE_NAME, pheno_valid, keep_iid_list=id_valid, plink_path=GENOTYPE_PLINK_PATH, work_dir=_new_plink_work_dir()) if pheno_valid.shape[0] != 0 else pd.DataFrame()
                    test_full_geno = plink_to_genotype_df(GENOTYPE_FILE_NAME, pheno_test, keep_iid_list=id_test, plink_path=GENOTYPE_PLINK_PATH, work_dir=_new_plink_work_dir())

                train = _attach_phenotype(train_full_geno, pheno_train, split_label='train')
                valid = _attach_phenotype(valid_full_geno, pheno_valid, split_label='valid') if pheno_valid.shape[0] != 0 else pd.DataFrame()
                test = _attach_phenotype(test_full_geno, pheno_test, split_label='test')
                full_marker_pool = (train, valid, test)

                if other_models_selected_now and OTHER_MODELS_MARKER_SOURCE == 'gene_network_plus_rf':
                    # Requirement 6, PLINK path: narrow the just-extracted
                    # TRUE full pool down to gene-window markers (already
                    # known from the bio-prior gene-pool loop above) UNION
                    # the reference instance's own RF-selected markers -
                    # mirrors the CSV branch above; see its comments for the
                    # full reasoning (this block only differs in extracting
                    # from PLINK rather than slicing an in-memory table).
                    _bio_prior_name_plink = bio_prior_names_now[0]
                    _bp_preview_plink = _resolve_bio_prior_params(_bio_prior_name_plink)
                    _merge_cfg_plink = _bp_preview_plink[13]
                    if not (isinstance(_merge_cfg_plink, dict) and _merge_cfg_plink.get('enabled', False)):
                        raise ValueError(
                            f"OTHER_MODELS_MARKER_SOURCE='gene_network_plus_rf' requires "
                            f"{_bio_prior_name_plink}'s own data-driven prior network (merge) "
                            f"feature to be enabled - it wasn't for phenotype "
                            f"{sample.loc[i,'phenotype']!r}."
                        )
                    # Requirement 11 (efficiency): this instance's
                    # data-driven merge already computed its RF-selected
                    # markers in the preview loop above (if enabled) -
                    # reuse that instead of redoing LD pruning + RF
                    # filtering a second time on the exact same data. Only
                    # ever falls through to a fresh computation if, for
                    # some reason, the preview didn't populate the cache
                    # for this instance (shouldn't normally happen when
                    # this branch's own ValueError check above already
                    # confirmed the merge feature is enabled, but handled
                    # defensively rather than assumed).
                    _cached_merge_plink = _bio_prior_merge_cache.get(_bio_prior_name_plink, {})
                    if 'rf_selected_markers' in _cached_merge_plink:
                        _rf_selected_plink = _cached_merge_plink['rf_selected_markers']
                        print(f"[GP] Task {i - idx*interval + 1}/{total_tasks} | reusing "
                              f"{len(_rf_selected_plink)} RF-selected marker(s) already computed "
                              f"for {_bio_prior_name_plink} this task (LD pruning/RF filtering skipped).")
                    else:
                        _rf_train_x_plink, _rf_train_y_plink = train.iloc[:, :-1], train.iloc[:, -1]
                        if _merge_cfg_plink.get('ld_prune') is not None:
                            _rf_train_x_plink, _, _ = LD_pruning(
                                _rf_train_x_plink, pd.DataFrame(), test.iloc[:, :-1], _merge_cfg_plink['ld_prune']
                            )
                        _rf_selected_plink, _ = select_markers_for_data_driven_network(
                            _rf_train_x_plink, _rf_train_y_plink, _merge_cfg_plink['rf_filter']
                        )
                        _bio_prior_merge_cache.setdefault(_bio_prior_name_plink, {})['rf_selected_markers'] = _rf_selected_plink
                    extended_markers_plink = set(bio_prior_gene_window_markers[_bio_prior_name_plink]) | set(_rf_selected_plink)
                    phenotype_col_plink = train.columns[-1]
                    keep_cols_plink = sorted(extended_markers_plink) + [phenotype_col_plink]
                    train = train[keep_cols_plink]
                    if valid.shape[0] != 0:
                        valid = valid[keep_cols_plink]
                    test = test[keep_cols_plink]
                    print(f"[GP] Task {i - idx*interval + 1}/{total_tasks} | Other model(s) restricted to "
                          f"{_bio_prior_name_plink}'s gene_network_plus_rf marker pool: "
                          f"{len(bio_prior_gene_window_markers[_bio_prior_name_plink])} gene-window + "
                          f"{len(_rf_selected_plink)} RF-selected = {len(extended_markers_plink)} unique "
                          f"marker(s) (out of {bim_marker_info.shape[0]} total).")
            else:
                # Either only GAT_biological_prior_knowledge is selected, or
                # other selected models were explicitly redirected to its
                # gene-network marker pool (OTHER_MODELS_MARKER_SOURCE ==
                # 'gene_network', validated at the top of GP() to require
                # GAT_biological_prior_knowledge to be selected too) - either
                # way, nothing needs anything beyond the gene-window pool
                # already computed above (LD_prune_effective/
                # RF_filter_effective are already guaranteed None in both
                # cases - see their computation near the top of GP()).
                train, valid, test = train_unpruned, valid_unpruned, test_unpruned

        # Requirement: if either split ends up with no data points after
        # dropping individuals with missing/unmatched genotype or phenotype
        # information (CSV path: handled above at data-merge time; PLINK
        # path: handled just above in _attach_phenotype), there is nothing
        # to train or evaluate this task's model(s) on - skip the whole
        # task rather than letting an empty DataFrame crash the model
        # fitting/scoring code further down.
        if train.shape[0] == 0 or test.shape[0] == 0:
            _empty_split = 'training' if train.shape[0] == 0 else 'test'
            print(f"[GP] Task {i - idx*interval + 1}/{total_tasks} | population {sample.loc[i,'population']} | "
                  f"phenotype {sample.loc[i,'phenotype']} | ratio {sample.loc[i,'ratio']} | "
                  f"replicate {sample.loc[i,'sample']} | Skipping this task - the {_empty_split} split has "
                  f"no data points left after dropping missing/unmatched genotype or phenotype information.")
            completed_units += units_per_task
            _report_progress(completed_units, total_units, label='Skipped (empty train/test split)')
            return

        # Kept aside so GAT_biological_prior_knowledge can always be routed
        # around LD pruning below, even when it's applied for other models
        # in this same task.
        if GENOTYPE_FORMAT == 'csv':
            train_unpruned, valid_unpruned, test_unpruned = train, valid, test
            full_marker_pool = (train_unpruned, valid_unpruned, test_unpruned)

            if OTHER_MODELS_MARKER_SOURCE in ('gene_network', 'gene_network_plus_rf'):
                # Restrict what every OTHER selected model receives (train,
                # valid, test - about to become train_pruned_full below) to
                # the gene-window markers GAT_biological_prior_knowledge
                # itself uses (and, for 'gene_network_plus_rf', additionally
                # the same RF-selected markers its own data-driven merge
                # feature would pick - requirement 6) - computed here the
                # same way that model builds its own gene<->marker mapping
                # internally, since for CSV input the full genotype table is
                # already in memory anyway (unlike the PLINK path, where
                # this same computation is what decides which markers are
                # worth converting in the first place - see the
                # GENOTYPE_FORMAT == 'plink' branch above).
                # GAT_biological_prior_knowledge itself (train_unpruned etc.)
                # is untouched - it always determines its own subset
                # regardless of what's passed to it. If more than one
                # bio-prior instance is selected, the FIRST one (in MODEL's
                # own order) is the reference network for this restriction -
                # same documented simplification as the PLINK path above.
                _bio_prior_name_csv = _bio_prior_model_names(MODEL)[0]
                _bp_preview_csv = _resolve_bio_prior_params(_bio_prior_name_csv)

                _marker_info_csv = pd.read_csv(_bp_preview_csv[9])
                _network_csv = load_network_json(_bp_preview_csv[7])
                _candidate_genes_csv = extract_candidate_genes(_network_csv)
                _gene_list_csv, _ = build_gene_list(_candidate_genes_csv, _bp_preview_csv[8], unit=_bp_preview_csv[10])
                _gene_to_markers_csv = _map_genes_to_markers(_gene_list_csv, _marker_info_csv, train.columns[:-1])
                gene_window_markers_csv = sorted({m for markers in _gene_to_markers_csv for m in markers})
                if not gene_window_markers_csv:
                    raise RuntimeError(
                        "No markers in the genotype table fall inside any gene window for "
                        "GAT_biological_prior_knowledge - cannot restrict other models to its "
                        f"gene-network marker pool (OTHER_MODELS_MARKER_SOURCE={OTHER_MODELS_MARKER_SOURCE!r})."
                    )

                extended_markers_csv = set(gene_window_markers_csv)
                if OTHER_MODELS_MARKER_SOURCE == 'gene_network_plus_rf':
                    _merge_cfg_csv = _bp_preview_csv[13]
                    if not (isinstance(_merge_cfg_csv, dict) and _merge_cfg_csv.get('enabled', False)):
                        raise ValueError(
                            f"OTHER_MODELS_MARKER_SOURCE='gene_network_plus_rf' requires "
                            f"{_bio_prior_name_csv}'s own data-driven prior network (merge) "
                            f"feature to be enabled - it wasn't for phenotype "
                            f"{sample.loc[i,'phenotype']!r}."
                        )
                    # Requirement 11 (efficiency): check the cache first
                    # (defensive - nothing populates it before this point
                    # for CSV-format input, unlike the PLINK path's own
                    # preview loop, but checking costs nothing and keeps
                    # this block correct if that ever changes), then WRITE
                    # the result into it either way, so GAT_biological_
                    # prior_knowledge() itself - whenever it actually runs
                    # later this task, including once per hyperparameter-
                    # tuning trial if enabled - can reuse this exact
                    # result instead of redoing LD pruning + RF filtering
                    # all over again.
                    _cached_merge_csv = _bio_prior_merge_cache.get(_bio_prior_name_csv, {})
                    if 'rf_selected_markers' in _cached_merge_csv:
                        _rf_selected_csv = _cached_merge_csv['rf_selected_markers']
                        print(f"[GP] Task {i - idx*interval + 1}/{total_tasks} | reusing "
                              f"{len(_rf_selected_csv)} RF-selected marker(s) already computed "
                              f"for {_bio_prior_name_csv} this task (LD pruning/RF filtering skipped).")
                    else:
                        _rf_train_x_csv, _rf_train_y_csv = train_unpruned.iloc[:, :-1], train_unpruned.iloc[:, -1]
                        if _merge_cfg_csv.get('ld_prune') is not None:
                            _rf_train_x_csv, _, _ = LD_pruning(
                                _rf_train_x_csv, pd.DataFrame(), test_unpruned.iloc[:, :-1], _merge_cfg_csv['ld_prune']
                            )
                        _rf_selected_csv, _ = select_markers_for_data_driven_network(
                            _rf_train_x_csv, _rf_train_y_csv, _merge_cfg_csv['rf_filter']
                        )
                        _bio_prior_merge_cache.setdefault(_bio_prior_name_csv, {})['rf_selected_markers'] = _rf_selected_csv
                    extended_markers_csv.update(_rf_selected_csv)
                    print(f"[GP] Task {i - idx*interval + 1}/{total_tasks} | "
                          f"{_bio_prior_name_csv}'s data-driven merge marker set: "
                          f"{len(gene_window_markers_csv)} gene-window + {len(_rf_selected_csv)} "
                          f"RF-selected = {len(extended_markers_csv)} unique marker(s).")

                gene_window_markers_csv = sorted(extended_markers_csv)
                phenotype_col_csv = train.columns[-1]
                keep_cols_csv = gene_window_markers_csv + [phenotype_col_csv]
                train = train[keep_cols_csv]
                if valid.shape[0] != 0:
                    valid = valid[keep_cols_csv]
                test = test[keep_cols_csv]
                print(f"[GP] Task {i - idx*interval + 1}/{total_tasks} | Other model(s) restricted to "
                      f"GAT_biological_prior_knowledge's {OTHER_MODELS_MARKER_SOURCE} marker pool: "
                      f"{len(gene_window_markers_csv)} marker(s) (out of {len(train_unpruned.columns) - 1} total).")

        if LD_prune_effective is not None and not ld_pruning_done_natively:
            task_num = i - idx*interval + 1
            _report_progress(
                completed_units, total_units,
                label=(f"Task {task_num}/{total_tasks} | population {sample.loc[i,'population']} | "
                       f"phenotype {sample.loc[i,'phenotype']} | ratio {sample.loc[i,'ratio']} | "
                       f"replicate {sample.loc[i,'sample']} | LD pruning")
            )
            n_markers_before = train.shape[1] - 1
            _pre_prune_train_x = train.iloc[:, :-1]  # captured before `train` is reassigned below (needed for the decay plot)
            ld_start_time = time.time()
            train_pruned, valid_pruned, test_pruned = LD_pruning(_pre_prune_train_x, valid.iloc[:,:-1], test.iloc[:,:-1], LD_prune_effective)
            train, test = pd.concat([train_pruned,train.iloc[:,-1]],axis=1), pd.concat([test_pruned, test.iloc[:,-1]],axis=1)
            if valid.shape[0] != 0:
                valid = pd.concat([valid_pruned, valid.iloc[:,-1]],axis=1)
            n_markers_after = train.shape[1] - 1
            print(f"[GP] Task {task_num}/{total_tasks} | LD pruning finished: "
                  f"{n_markers_before} -> {n_markers_after} markers "
                  f"(took {time.time() - ld_start_time:.1f}s)")
            completed_units += 1

            if _ld_decay_due((sample.loc[i, 'population'], sample.loc[i, 'ratio'], sample.loc[i, 'sample'])):
                _save_ld_decay_plot(
                    lambda: _pre_prune_train_x,
                    sample.loc[i, 'population'], sample.loc[i, 'sample'], sample.loc[i, 'ratio'],
                )

        # RF marker importance filtering (requirement: runs AFTER LD pruning
        # when both are enabled, so it narrows whatever LD pruning already
        # produced above - or the original marker set, if LD pruning is
        # off). Fit on this task's own training set only, exactly like LD
        # pruning above - never on validation/test data.
        if RF_filter_effective is not None:
            task_num = i - idx*interval + 1
            _report_progress(
                completed_units, total_units,
                label=(f"Task {task_num}/{total_tasks} | population {sample.loc[i,'population']} | "
                       f"phenotype {sample.loc[i,'phenotype']} | ratio {sample.loc[i,'ratio']} | "
                       f"replicate {sample.loc[i,'sample']} | RF marker importance filtering")
            )
            n_markers_before = train.shape[1] - 1
            rf_start_time = time.time()
            train, valid, test = RF_marker_filtering(train, valid, test, RF_filter_effective)
            n_markers_after = train.shape[1] - 1
            print(f"[GP] Task {task_num}/{total_tasks} | RF marker importance filtering finished: "
                  f"{n_markers_before} -> {n_markers_after} markers "
                  f"(took {time.time() - rf_start_time:.1f}s)")
            completed_units += 1

        # What every model except GAT_biological_prior_knowledge actually
        # trains on: LD-pruned and/or RF-importance-filtered if either ran
        # just above, otherwise identical to
        # train_unpruned/valid_unpruned/test_unpruned.
        train_pruned_full, valid_pruned_full, test_pruned_full = train, valid, test

        # Requirement 6: basic per-scenario statistics - one row per task,
        # recorded here (train/valid/test finalised, right before the
        # per-model dispatch loop) rather than per-model, since split
        # sizes and marker count are task-level, not model-level,
        # quantities (every model in this task trains on the same row
        # counts, even though some - e.g. GAT_biological_prior_knowledge -
        # use a different, smaller MARKER pool than 'train' reflects
        # here; this row always describes the main/full-or-filtered pool
        # 'other' models actually use). Saved/checkpointed/assembled
        # exactly like every other per-scenario output (see
        # checkpoint_utils.py's RESULT_FILE_NAMES / assemble.py).
        stats = pd.concat([stats, pd.DataFrame([{
            'population': sample.loc[i, 'population'],
            'phenotype': sample.loc[i, 'phenotype'],
            'ratio': sample.loc[i, 'ratio'],
            'sample': sample.loc[i, 'sample'],
            'n_train': train.shape[0],
            'n_valid': valid.shape[0],
            'n_test': test.shape[0],
            'n_total': train.shape[0] + valid.shape[0] + test.shape[0],
            'n_markers': train.shape[1] - 1,
        }])], ignore_index=True)
        
        result_train_sample = pd.DataFrame()
        result_valid_sample = pd.DataFrame()
        result_test_sample = pd.DataFrame()
        
        # Prediction model implementation
        for jj in range(len(MODEL_RUN)):
            if MODEL_RUN[jj] == 'ensemble':
                continue

            # MODEL_RUN[jj] is the (possibly hyperparameter-tuning-algorithm-
            # suffixed, e.g. 'RF__Grid') name used for every OUTPUT label
            # below (record/effect/predictions columns) - base_model_name is
            # what actually decides which model function to call and which
            # HPARAMETERS entry to read, since HPARAMETERS is keyed by base
            # model name only. When HP_TUNE isn't used, or this model was
            # only tuned with a single algorithm, the two are identical (see
            # models.hyperparameter_tuning.tuned_model_name / base_of) - so
            # this line changes nothing for any run that doesn't use
            # multi-algorithm tuning. base_model_name is also what's used to
            # look up HP_TUNE below - deliberately re-derived from
            # MODEL_RUN[jj] every iteration (not cached), since a task
            # further down the sample loop may have no validation split even
            # when an earlier one did (mixed single-float/tuple RATIO
            # entries), which the tuning check below re-evaluates per task.
            base_model_name = base_of(MODEL_RUN[jj])

            # Every model sees train_pruned_full/valid_pruned_full/test_pruned_full
            # by default (identical to the unpruned/unfiltered data if neither LD
            # pruning nor RF filtering ran this task). GAT_biological_prior_knowledge
            # is the one exception: its own dispatch branch below overrides this
            # again, based on its 'marker_source_mode' hyperparameter, once that
            # model's own params list has been resolved (it may be a per-phenotype
            # dict - see that branch).
            train, valid, test = train_pruned_full, valid_pruned_full, test_pruned_full

            task_num = i - idx*interval + 1
            _report_progress(
                completed_units, total_units,
                label=(f"Task {task_num}/{total_tasks} | population {sample.loc[i,'population']} | "
                       f"phenotype {sample.loc[i,'phenotype']} | ratio {sample.loc[i,'ratio']} | "
                       f"replicate {sample.loc[i,'sample']} | model: {MODEL_RUN[jj]}")
            )
            model_start_time = time.time()

            # Hyperparameter tuning (opt-in, requires HP_TUNE[base_model_name]
            # enabled AND a real validation split for THIS task - Requirement:
            # only ever active when the dataset is split into train/valid/test).
            # Runs the winning algorithm's search on the validation set, then
            # a final confirmatory fit whose outputs (sample_pearson,
            # sample_effect, predicted_test, ...) feed into exactly the same
            # "Store prediction results" code below as every other model -
            # everything downstream is keyed off MODEL_RUN[jj], so it needs no
            # further changes for tuned models to be reported correctly.
            hp_tuning_cfg = (HP_TUNE or {}).get(base_model_name)
            _do_tuning = bool(hp_tuning_cfg) and hp_tuning_cfg.get('enabled') and valid.shape[0] != 0

            if _do_tuning:
                algorithm = algo_of(MODEL_RUN[jj]) or hp_tuning_cfg['algorithms'][0]

                # HPARAMETERS[base_model_name] can be a per-phenotype dict
                # for GAT_biological_prior_knowledge (see _call_model's own
                # bio_prior branch and its main-dispatch counterpart further
                # down) - resolve which phenotype's flat params list applies
                # to THIS task before ever handing it to the tuner, since
                # tune_model_hyperparameters always works with a flat list
                # (same contract every other model already has). Every other
                # model's HPARAMETERS entry is already a flat list, so
                # _hparams_entry/_base_params_for_tuning are identical for
                # them - no behaviour change there.
                _hparams_entry = HPARAMETERS[base_model_name]
                if _is_bio_prior_model(base_model_name) and isinstance(_hparams_entry, dict):
                    _current_phenotype = sample.loc[i, 'phenotype']
                    if _current_phenotype not in _hparams_entry:
                        raise KeyError(
                            f"HPARAMETERS[{base_model_name!r}] is a per-phenotype dict but has "
                            f"no entry for phenotype {_current_phenotype!r} - cannot tune it. Add "
                            f"one, or pass a single flat params list instead to share one network "
                            f"across every phenotype in PHENOTYPE."
                        )
                    _base_params_for_tuning = _hparams_entry[_current_phenotype]
                else:
                    _base_params_for_tuning = _hparams_entry

                final_params, tuned_result, best_valid_score, tuned_values, tune_elapsed = tune_model_hyperparameters(
                    model_name=base_model_name,
                    base_params=_base_params_for_tuning,
                    train=train, valid=valid, test=test,
                    run_model_fn=lambda name, tr, va, te, p, rn: _call_model(
                        name, tr, va, te, p,
                        bio_prior_ctx={
                            'bio_prior_pools': bio_prior_pools,
                            'train_unpruned': train_unpruned, 'valid_unpruned': valid_unpruned,
                            'test_unpruned': test_unpruned,
                            'bio_prior_merge_pools': bio_prior_merge_pools,
                            'full_marker_pool': full_marker_pool,
                            'bim_marker_info_path': bim_marker_info_path,
                            'network_cache': _bio_prior_merge_cache,
                        } if _is_bio_prior_model(name) else None,
                    ),
                    RESULT_NAME=RESULT_NAME,
                    algorithm=algorithm,
                    budget_kwargs=hp_tuning_cfg.get('budget', {}).get(algorithm, {}),
                    hparam_specs=HPARAM_SPECS,
                    reduced_cost_search=hp_tuning_cfg.get('reduced_cost_search', True),
                )
                if _is_bio_prior_model(base_model_name) and isinstance(_hparams_entry, dict):
                    HPARAMETERS[base_model_name][_current_phenotype] = final_params
                else:
                    HPARAMETERS[base_model_name] = final_params

                sample_pearson = tuned_result['pearson_test']
                sample_mse = tuned_result['mse_test']
                sample_effect = tuned_result.get('effect', pd.DataFrame())
                sample_interaction = tuned_result.get('interaction', pd.DataFrame())
                sample_attention = tuned_result.get('attention', pd.DataFrame())
                predicted_test = tuned_result['predicted_test']
                predicted_valid = tuned_result['predicted_valid']
                predicted_train = tuned_result['predicted_train']

                hp_record = pd.concat([hp_record, pd.DataFrame([{
                    'population': sample.loc[i, 'population'], 'phenotype': sample.loc[i, 'phenotype'],
                    'model': MODEL_RUN[jj], 'ratio': sample.loc[i, 'ratio'], 'sample': sample.loc[i, 'sample'],
                    'algorithm': algorithm, 'valid_pearson_over_mse': best_valid_score,
                    **tuned_values,
                }])])
                print(f"[GP] Task {task_num}/{total_tasks} | {MODEL_RUN[jj]} tuned via {algorithm}: "
                      f"valid Pearson/MSE={best_valid_score:.4f} (took {tune_elapsed:.1f}s) -> {tuned_values}")

            elif base_model_name == 'rrBLUP':
                result_rrBLUP = rrBLUP(train, valid, test, HPARAMETERS[base_model_name], RESULT_NAME)
                sample_pearson, sample_mse = result_rrBLUP['r_pearson'][0], result_rrBLUP['r_MSE'][0]
                sample_effect = result_rrBLUP['r_effect']
                predicted_test = result_rrBLUP['r_y_predicted']
                predicted_valid = result_rrBLUP['r_y_predicted_valid']
                predicted_train = result_rrBLUP['r_y_predicted_train']
            elif base_model_name == 'GBLUP':
                # GBLUP params: [nIter, burnIn, get_effect, Shapley_num,
                # max_shap_features, Shapley_nIter, Shapley_burnIn] ->
                # Shapley_num is at index 3.
                if HPARAMETERS[base_model_name][3] == 'all':
                    HPARAMETERS[base_model_name][3] = test.shape[0]
                result_GBLUP = GBLUP(train, valid, test, HPARAMETERS[base_model_name], RESULT_NAME)
                sample_pearson, sample_mse = result_GBLUP['r_pearson'][0], result_GBLUP['r_MSE'][0]
                sample_effect = result_GBLUP['r_effect']
                predicted_test = result_GBLUP['r_y_predicted']
                predicted_valid = result_GBLUP['r_y_predicted_valid']
                predicted_train = result_GBLUP['r_y_predicted_train']
            elif base_model_name  == 'BayesB':
                result_BayesB = BayesB(train, valid, test, HPARAMETERS[base_model_name], RESULT_NAME)
                sample_pearson, sample_mse = result_BayesB['r_pearson'][0], result_BayesB['r_MSE'][0]
                sample_effect = result_BayesB['r_effect']
                predicted_test = result_BayesB['r_y_predicted']
                predicted_valid = result_BayesB['r_y_predicted_valid']
                predicted_train = result_BayesB['r_y_predicted_train']
            elif base_model_name  == 'RKHS':
                # RKHS params: [nIter, burnIn, h, get_effect, Shapley_num,
                # max_shap_features, Shapley_nIter, Shapley_burnIn] ->
                # Shapley_num is at index 4.
                if HPARAMETERS[base_model_name][4] == 'all':
                    HPARAMETERS[base_model_name][4] = test.shape[0]
                result_RKHS = RKHS(train, valid, test, HPARAMETERS[base_model_name], RESULT_NAME)
                sample_pearson, sample_mse = result_RKHS['r_pearson'][0], result_RKHS['r_MSE'][0]
                sample_effect = result_RKHS['r_effect']
                predicted_test = result_RKHS['r_y_predicted']
                predicted_valid = result_RKHS['r_y_predicted_valid']
                predicted_train = result_RKHS['r_y_predicted_train']
            elif base_model_name  == 'RF':
                # RF params: [estimators, features_max, sample_max, max_depth,
                # min_samples_leaf, get_interaction, shapley_num, threshold,
                # max_interaction_features] -> shapley_num is at index 6.
                if HPARAMETERS[base_model_name][6] == 'all':
                    HPARAMETERS[base_model_name][6] = test.shape[0] 
                sample_pearson, sample_mse, sample_effect, sample_interaction, predicted_test, predicted_valid, predicted_train = RF(train, valid, test, HPARAMETERS[base_model_name])
            elif base_model_name  == 'SVR':
                # SVR params: [ker, eps, con, deg, gam, coef0, get_effect,
                # shapley_num, max_shap_features, shap_background_size,
                # shap_nsamples] -> shapley_num is at index 7.
                if HPARAMETERS[base_model_name][7] == 'all':
                    HPARAMETERS[base_model_name][7] = test.shape[0] 
                sample_pearson, sample_mse, sample_effect, predicted_test, predicted_valid, predicted_train = SV_Regression(train, valid, test, HPARAMETERS[base_model_name])
            elif base_model_name  == 'KNN':
                # KNN params: [n_neighbours, weights, p, get_effect, shapley_num,
                # max_shap_features, shap_background_size, shap_nsamples] ->
                # shapley_num is at index 4.
                if HPARAMETERS[base_model_name][4] == 'all':
                    HPARAMETERS[base_model_name][4] = test.shape[0] 
                sample_pearson, sample_mse, sample_effect, predicted_test, predicted_valid, predicted_train = KNN(train, valid, test, HPARAMETERS[base_model_name])
            elif base_model_name  == 'MLP':
                # MLP params: [neurons, dout, lrate, decay, ep, bsize, neurons2,
                # shapley_num] -> shapley_num is at index 7.
                if HPARAMETERS[base_model_name][7] == 'all':
                    HPARAMETERS[base_model_name][7] = test.shape[0] 
                sample_pearson, sample_mse, sample_effect, predicted_test, predicted_valid, predicted_train = ML_Perceptron(train, valid, test, HPARAMETERS[base_model_name])
            elif base_model_name  == 'GAT_infinitesimal_node_level':
                # GAT_infinitesimal_node_level params: [..., samples, marker_effect] -> samples is second-to-last
                if HPARAMETERS[base_model_name][-2] == 'all':
                    HPARAMETERS[base_model_name][-2] = test.shape[0] 
                sample_pearson, sample_mse, sample_effect, predicted_test, predicted_valid, predicted_train = GAT_infinitesimal_node_level(train, valid, test, HPARAMETERS[base_model_name])
            elif base_model_name  == 'GAT_infinitesimal':
                # GAT_infinitesimal params: [..., marker_effect, samples] -> samples is the last element
                if HPARAMETERS[base_model_name][-1] == 'all':
                    HPARAMETERS[base_model_name][-1] = test.shape[0] 
                sample_pearson, sample_mse, sample_effect, predicted_test, predicted_valid, predicted_train = GAT_infinitesimal(train, valid, test, HPARAMETERS[base_model_name])
            elif base_model_name  == 'GAT_fully_connected':
                # GAT_fully_connected params: [..., marker_effect, samples] -> samples is the last element
                if HPARAMETERS[base_model_name][-1] == 'all':
                    HPARAMETERS[base_model_name][-1] = test.shape[0] 
                sample_pearson, sample_mse, sample_effect, predicted_test, predicted_valid, predicted_train, sample_attention = GAT_fully_connected(train, valid, test, HPARAMETERS[base_model_name])
            elif base_model_name  == 'GAT_prior_knowledge':
                # GAT_prior_knowledge params: [..., marker_effect, samples, top_rate] -> samples is second-to-last
                if HPARAMETERS[base_model_name][-2] == 'all':
                    HPARAMETERS[base_model_name][-2] = test.shape[0] 
                sample_pearson, sample_mse, sample_effect, predicted_test, predicted_valid, predicted_train, sample_attention = GAT_prior_knowledge(train, valid, test, HPARAMETERS[base_model_name])
            elif _is_bio_prior_model(base_model_name):
                # HPARAMETERS[base_model_name] can be either:
                #   - a flat params list [neuron, dropout, lrate, decay, epoch, bsize, heads,
                #     network_json_path, gene_location_csv_path, marker_info_path, unit,
                #     include_mediated_edges, max_hops, data_driven_merge (dict, requirement 2),
                #     marker_effect, samples] -> the SAME network is used for every phenotype in
                #     PHENOTYPE, or
                #   - a dict {phenotype_name: params_list, ...} so each phenotype gets its own
                #     network_json_path/gene_location_csv_path (built from a different FLASH-P
                #     run or a different uploaded JSON per trait) - see
                #     Preprocess/gene_network_prior.py and the GUI's 'Biological Prior Network'
                #     tab. In both cases, samples is the last element of whichever list is used.
                #
                # MODEL_RUN[jj] is 'GAT_biological_prior_knowledge' itself, or a
                # numbered instance of it ('GAT_biological_prior_knowledge_2',
                # etc. - see _is_bio_prior_model) - each such name is its own,
                # completely independent key into HPARAMETERS, with its own
                # network/gene-location/etc. This is what lets a user compare
                # several different networks for the same phenotype(s) - e.g.
                # several separate FlashP samples, or several hand-uploaded
                # JSON files - as separate models in one run, each showing up
                # under its own name in every output file's 'model' column.
                bio_prior_hparams = HPARAMETERS[base_model_name]
                if isinstance(bio_prior_hparams, dict):
                    current_phenotype = sample.loc[i, 'phenotype']
                    if current_phenotype not in bio_prior_hparams:
                        raise KeyError(
                            f"HPARAMETERS[{MODEL_RUN[jj]!r}] is a per-phenotype "
                            f"dict but has no entry for phenotype '{current_phenotype}'. Add one, "
                            f"or pass a single flat params list instead to share one network "
                            f"across every phenotype in PHENOTYPE."
                        )
                    bio_prior_params = bio_prior_hparams[current_phenotype]
                else:
                    bio_prior_params = bio_prior_hparams

                if bio_prior_params[-1] == 'all':
                    bio_prior_params[-1] = test.shape[0]

                # Always the full, unpruned/unfiltered marker set - this
                # model picks its own candidate markers via the gene
                # network regardless of what LD pruning/RF importance
                # filtering did for other models this task (see
                # OTHER_MODELS_MARKER_SOURCE for the model-independent
                # choice that actually affects those other models instead).
                # For GENOTYPE_FORMAT == 'plink', bio_prior_pools (computed
                # further up) has this exact instance's OWN gene-window
                # extraction; for 'csv' (or if this instance's pool is
                # somehow missing) fall back to the shared
                # train_unpruned/valid_unpruned/test_unpruned triple, which
                # for 'csv' is simply the full, unrestricted genotype table -
                # the model filters it down to its own gene-network markers
                # internally regardless of which instance this is.
                train, valid, test = bio_prior_pools.get(base_model_name, (train_unpruned, valid_unpruned, test_unpruned)) \
                    if GENOTYPE_FORMAT == 'plink' else (train_unpruned, valid_unpruned, test_unpruned)

                _merge_source_data = bio_prior_merge_pools.get(base_model_name) \
                    if GENOTYPE_FORMAT == 'plink' else full_marker_pool

                # For GENOTYPE_FORMAT == 'plink', marker positions can only
                # correctly come from the SAME .bim file the extracted
                # genotype columns were pulled from above - override
                # whatever marker_info_path was configured in the GUI
                # (params[9]) with the .bim-derived one written near the top
                # of GP(), rather than risk desynchronising marker names
                # from their true positions.
                if GENOTYPE_FORMAT == 'plink':
                    bio_prior_params = list(bio_prior_params)
                    bio_prior_params[9] = bim_marker_info_path

                # Best-effort check that the network JSON this call is about
                # to use is actually the one meant for this phenotype -
                # catches e.g. an 'Upload' or 'Use a previous local FLASH-P
                # run' path pointed at the wrong file (a FLASH-P-generated
                # file's own metadata locks this in automatically; this is
                # the safety net for everything else, and for runs driven
                # from a saved config / HPC script rather than the GUI).
                # Heuristic and non-fatal on purpose (see
                # Preprocess/gene_network_prior.py's phenotype_matches_network_metadata
                # docstring) - printed as a warning, never raised.
                try:
                    matched, declared = phenotype_matches_network_metadata(
                        bio_prior_params[7], sample.loc[i, 'phenotype']
                    )
                    if matched is False:
                        print(
                            f"[{MODEL_RUN[jj]}] WARNING: '{bio_prior_params[7]}' "
                            f"declares itself to be for phenotype {declared!r}, which doesn't "
                            f"obviously match the phenotype {sample.loc[i,'phenotype']!r} it's "
                            f"about to be used for. This is a heuristic name check, not proof of "
                            f"a mistake - double-check this is the right file for this trait."
                        )
                except Exception:
                    pass  # never let this best-effort check block a real run

                sample_pearson, sample_mse, sample_effect, predicted_test, predicted_valid, predicted_train, sample_attention = GAT_biological_prior_knowledge(
                    train, valid, test, bio_prior_params, RESULT_NAME, sample.loc[i, 'phenotype'], MODEL_RUN[jj],
                    merge_source_data=_merge_source_data, network_cache=_bio_prior_merge_cache,
                )
            
            # Store prediction results
            record_sample = pd.DataFrame([{'population': sample.loc[i,'population'],
                                           'phenotype': sample.loc[i,'phenotype'],
                                           'model': MODEL_RUN[jj],
                                           'ratio': sample.loc[i,'ratio'],
                                           'sample': sample.loc[i,'sample'],
                                           'Pearson correlation': sample_pearson,
                                           'MSE': sample_mse}
                                          ])
            record = pd.concat([record, record_sample])
            
            if result_test_sample.shape[0]==0:
                result_test_sample = pd.DataFrame({'id': id_test,
                                                   'population': [sample.loc[i,'population']] * len(id_test),
                                                   'ratio': [sample.loc[i,'ratio']] * len(id_test),
                                                   'phenotype': [sample.loc[i,'phenotype']] * len(id_test),
                                                   'sample': [sample.loc[i,'sample']] * len(id_test),
                                                   'actual':test.iloc[:,-1],
                                                   MODEL_RUN[jj]:predicted_test
                                                  })
            else:
                result_test_sample = pd.concat([result_test_sample,
                                                pd.DataFrame({MODEL_RUN[jj]:predicted_test})
                                              ], axis=1)
           
            if (result_valid_sample.shape[0]==0 and type(sample.loc[i,'ratio']) is tuple) or (result_valid_sample.shape[0]==0 and W_OPT is not None and SCENARIO == 'between'):
                result_valid_sample = pd.DataFrame({'id': id_valid,
                                                   'population': [sample.loc[i,'population']] * len(id_valid),
                                                   'ratio': [sample.loc[i,'ratio']] * len(id_valid),
                                                   'phenotype': [sample.loc[i,'phenotype']] * len(id_valid),
                                                   'sample': [sample.loc[i,'sample']] * len(id_valid),
                                                   'actual':valid.iloc[:,-1],
                                                   MODEL_RUN[jj]:predicted_valid
                                                  })
            elif (result_valid_sample.shape[0]!=0 and type(sample.loc[i,'ratio']) is tuple) or (W_OPT is not None and SCENARIO == 'between'):
                result_valid_sample = pd.concat([result_valid_sample,
                                                pd.DataFrame({MODEL_RUN[jj]:predicted_valid})
                                              ], axis=1) 
            else:
                result_valid_sample = pd.DataFrame()
            
            if result_train_sample.shape[0]==0:
                result_train_sample = pd.DataFrame({'id': id_train,
                                                   'population': [sample.loc[i,'population']] * len(id_train),
                                                   'ratio': [sample.loc[i,'ratio']] * len(id_train),
                                                   'phenotype': [sample.loc[i,'phenotype']] * len(id_train),
                                                   'sample': [sample.loc[i,'sample']] * len(id_train),
                                                   'actual':train.iloc[:,-1],
                                                   MODEL_RUN[jj]:predicted_train
                                                  })
            else:
                result_train_sample = pd.concat([result_train_sample,
                                                pd.DataFrame({MODEL_RUN[jj]:predicted_train})
                                              ], axis=1)  
            
            if sample_effect.shape[0] != 0:
                expected_markers = train.columns.tolist()[:-1]
                if sample_effect.shape[1] != len(expected_markers):
                    raise ValueError(
                        f"Model '{MODEL_RUN[jj]}' returned a marker-effect table with "
                        f"{sample_effect.shape[1]} columns, but this task's training data "
                        f"(train.shape={train.shape}) has {len(expected_markers)} markers. "
                        f"Every model is expected to return exactly one effect value per "
                        f"marker.\n"
                        f"Diagnostic context for this failure:\n"
                        f"  Task: population={sample.loc[i,'population']!r} "
                        f"phenotype={sample.loc[i,'phenotype']!r} "
                        f"ratio={sample.loc[i,'ratio']!r} sample={sample.loc[i,'sample']!r}\n"
                        f"  HPARAMETERS['{base_model_name}'] = {HPARAMETERS[base_model_name]!r}\n"
                        f"  LD_prune applied to this model: "
                        f"{not _is_bio_prior_model(base_model_name) and LD_prune_effective is not None}"
                        + (f" | LD_prune config = {LD_prune_effective!r}" if LD_prune_effective is not None else "")
                        + f"\n  RF_filter applied to this model: "
                        f"{not _is_bio_prior_model(base_model_name) and RF_filter_effective is not None}"
                        + (f" | RF_filter config = {RF_filter_effective!r}" if RF_filter_effective is not None else "")
                    )
                sample_effect.columns = expected_markers
                effect_sample = pd.DataFrame([{'population': sample.loc[i,'population'],
                                               'phenotype': sample.loc[i,'phenotype'],
                                               'model': MODEL_RUN[jj],
                                               'ratio': sample.loc[i,'ratio'],
                                               'sample': sample.loc[i,'sample'],
                                               }])
                effect_sample = pd.concat([effect_sample,
                                           sample_effect.reset_index(drop=True),
                                          ],axis=1)
                effect = pd.concat([effect, effect_sample])
           
            if base_model_name == 'RF' and sample_interaction.shape[0] != 0:
                sample_interaction['population'] = sample.loc[i,'population']
                sample_interaction['phenotype'] = sample.loc[i,'phenotype']
                sample_interaction['model'] = MODEL_RUN[jj]
                sample_interaction['ratio'] = str(sample.loc[i,'ratio'])
                sample_interaction['sample'] = sample.loc[i,'sample'] 
                
                interactions = pd.concat([interactions, sample_interaction])   
            
            if base_model_name == 'GAT_fully_connected' or base_model_name == 'GAT_prior_knowledge' or _is_bio_prior_model(base_model_name):
                sample_attention['population'] = sample.loc[i,'population']
                sample_attention['model'] = MODEL_RUN[jj]
                sample_attention['ratio'] = str(sample.loc[i,'ratio'])
                sample_attention['phenotype'] = sample.loc[i,'phenotype']
                sample_attention['sample'] = sample.loc[i,'sample'] 
                sample_attention.columns = ['marker1','marker2','value','population','model','ratio','phenotype','sample']
                sample_attention = sample_attention.loc[:,['population','phenotype','model','ratio','sample', 'marker1', 'marker2', 'value']]
                sample_attention = pd.concat([attention_total, sample_attention],axis=0)
                
                attention_total = pd.concat([attention_total, sample_attention], axis=0)

            print(f"[GP] Task {task_num}/{total_tasks} | model {MODEL_RUN[jj]} finished: "
                  f"Pearson r={sample_pearson:.4f}, MSE={sample_mse:.4f} "
                  f"(took {time.time() - model_start_time:.1f}s)")
            completed_units += 1

        # Weight optimisation
        #
        # Requirement: when more than one hyperparameter-tuning algorithm was
        # used for at least one model AND a weighted-ensemble method is
        # selected, the user chooses whether to combine models per tuning
        # method (one ensemble per algorithm - HP_TUNE_ENSEMBLE_MODE ==
        # 'per_method') or across every tuning method at once ('across_methods').
        # ensemble_groups() collapses to a single, unchanged group whenever
        # nothing was actually multi-algorithm-tuned (the overwhelmingly
        # common case), so this loop runs exactly once with group_label=None
        # and group_models == MODEL_RUN for every run that doesn't use this
        # feature - byte-for-byte the same calls as before this change.
        if W_OPT is not None and (type(sample.loc[i,'ratio']) is tuple or SCENARIO=='between'):
            # Each weighted-ensemble method has its own hardcoded output
            # column name and record/effect/weight 'model' label (confirmed
            # from models/Linear_transformation.py, Nelder_Mead.py,
            # Bayesian_optimisation.py - note 'Nelder Mead' uses a DIFFERENT
            # string for its column ('Nelder-Mead', hyphenated) than for its
            # record/effect/weight label ('Nelder Mead', spaced) - these are
            # that same distinction, not a typo).
            _WOPT_LABELS = {
                'Linear transformation': {'column': 'Linear transformation', 'label': 'Linear transformation'},
                'Nelder Mead': {'column': 'Nelder-Mead', 'label': 'Nelder Mead'},
                'Bayesian optimisation': {'column': 'Bayesian', 'label': 'Bayesian optimisation'},
            }
            for _group_label, _group_models in ensemble_groups(MODEL_BASE, HP_TUNE, HP_TUNE_ENSEMBLE_MODE):
                for kk in range(len(W_OPT)):
                    # Each group gets a CLEAN snapshot of record/effect/weight
                    # (not the growing, already-mutated one from a previous
                    # group in this same loop) as the input these functions
                    # mutate internally - Linear_transformation.py/
                    # Nelder_Mead.py/Bayesian_optimisation.py were only ever
                    # exercised once per task in the original design, and
                    # each duplicates-and-overwrites its OWN input's last row
                    # internally; chaining that through a second call in the
                    # same task (needed for per-method grouping) corrupts
                    # wide numeric columns on some pandas versions (verified:
                    # cells silently become 1-element lists/arrays instead of
                    # floats, later crashing the final ensemble()'s .abs()).
                    # Only predicted_test/valid/train_sample chain forward
                    # normally between groups - that's the *intended*,
                    # already-working behaviour (each method adds its own
                    # extra column to the same table).
                    _record_snapshot = record.copy()
                    _effect_snapshot = effect.copy()
                    _weight_snapshot = weight.copy()

                    if W_OPT[kk]  == 'Linear transformation':
                        _record_out, _effect_out, predicted_test_sample, predicted_valid_sample, predicted_train_sample, _weight_out = Linear_transformation(result_train_sample, result_valid_sample, result_test_sample, _record_snapshot, _effect_snapshot, _weight_snapshot, _group_models, HYPERPARAMETERS_OPT['Linear transformation'])
                    elif W_OPT[kk] == 'Nelder Mead':
                        _record_out, _effect_out, predicted_test_sample, predicted_valid_sample, predicted_train_sample, _weight_out = Nelder_Mead(result_train_sample, result_valid_sample, result_test_sample, _record_snapshot, _effect_snapshot, _weight_snapshot, _group_models, HYPERPARAMETERS_OPT['Nelder Mead'])
                    elif W_OPT[kk] == 'Bayesian optimisation':
                        _record_out, _effect_out, predicted_test_sample, predicted_valid_sample, predicted_train_sample, _weight_out = Bayesian(result_train_sample, result_valid_sample, result_test_sample, _record_snapshot, _effect_snapshot, _weight_snapshot, _group_models, HYPERPARAMETERS_OPT['Bayesian optimisation'])
                    else:
                        continue

                    # Only the row(s) this call actually added are new -
                    # append those onto the real accumulators (clean concat,
                    # not chained mutation).
                    _new_record = _record_out.iloc[_record_snapshot.shape[0]:]
                    _new_effect = _effect_out.iloc[_effect_snapshot.shape[0]:]
                    _new_weight = _weight_out.iloc[_weight_snapshot.shape[0]:]

                    if _group_label is not None:
                        # Differentiate this group's weighted-ensemble output
                        # from every other algorithm-group's (Requirement:
                        # suffix models per tuning method when more than one
                        # is in play) - the three model files above are
                        # otherwise untouched; this is purely a rename of
                        # their already-produced output.
                        _col = _WOPT_LABELS[W_OPT[kk]]['column']
                        _lbl = _WOPT_LABELS[W_OPT[kk]]['label']
                        _new_lbl = f'{_lbl}__{_group_label}'
                        for _df in (predicted_test_sample, predicted_valid_sample, predicted_train_sample):
                            if isinstance(_df, pd.DataFrame) and _col in _df.columns:
                                _df.rename(columns={_col: _new_lbl}, inplace=True)
                        if _new_record.shape[0] != 0:
                            _new_record = _new_record.copy()
                            _new_record.loc[_new_record['model'] == _lbl, 'model'] = _new_lbl
                        if _new_effect.shape[0] != 0:
                            _new_effect = _new_effect.copy()
                            _new_effect.loc[_new_effect['model'] == _lbl, 'model'] = _new_lbl
                        if _new_weight.shape[0] != 0 and 'model' in _new_weight.columns:
                            _new_weight = _new_weight.copy()
                            _new_weight.loc[_new_weight['model'] == _lbl, 'model'] = _new_lbl

                    record = pd.concat([record, _new_record])
                    effect = pd.concat([effect, _new_effect])
                    weight = pd.concat([weight, _new_weight])
            
        result_test = pd.concat([result_test, result_test_sample],axis=0)
        result_valid = pd.concat([result_valid, result_valid_sample],axis=0)
        result_train = pd.concat([result_train, result_train_sample],axis=0)
        
        result_train = result_train.sort_values(['id']).reset_index(drop=True)
        
        if type(sample.loc[i,'ratio']) is tuple:
            result_valid = result_valid.sort_values(['id']).reset_index(drop=True)
        result_test = result_test.sort_values(['id']).reset_index(drop=True)
        
    # ---------------------------------------------------------------------- #
    # Checkpoint/resume (continued): the actual driving loop. Each task gets
    # a full snapshot of every accumulator taken before it starts - if it
    # raises, those snapshots are restored (discarding any partial
    # contribution the failed task may already have made - e.g. its first
    # model succeeded and got appended to `record` before its second model
    # raised), so a checkpoint is always all-or-nothing per task: a resumed
    # run always re-does the ENTIRE failed task cleanly, never a
    # half-finished one, which would otherwise risk duplicate rows for
    # whichever model(s) did complete before the failure. Every task that
    # finishes cleanly falls through to the next iteration normally, exactly
    # as the original single for-loop did - nothing about a successful run's
    # behaviour changes here.
    # ---------------------------------------------------------------------- #
    for i in range(_start_i, idx*interval+interval):

        if i >= sample.shape[0]:
            break

        _pre_task_snapshot = {
            'record': record, 'result_train': result_train, 'result_valid': result_valid,
            'result_test': result_test, 'effect': effect, 'interactions': interactions,
            'weight': weight, 'attention_total': attention_total, 'hp_record': hp_record,
            'stats': stats,
        }
        try:
            _process_one_task(i)
        except Exception as _task_exc:
            # Requirement: "save all the results so far when an error
            # occurs" - roll back this task's own (possibly partial)
            # contribution first, then persist everything that finished
            # BEFORE it, to the exact same files a successful run would
            # produce (see save_partial_results()'s own docstring for why
            # that's the same helper the final, successful-run save below
            # also uses).
            record = _pre_task_snapshot['record']
            result_train = _pre_task_snapshot['result_train']
            result_valid = _pre_task_snapshot['result_valid']
            result_test = _pre_task_snapshot['result_test']
            effect = _pre_task_snapshot['effect']
            interactions = _pre_task_snapshot['interactions']
            weight = _pre_task_snapshot['weight']
            attention_total = _pre_task_snapshot['attention_total']
            hp_record = _pre_task_snapshot['hp_record']
            stats = _pre_task_snapshot['stats']

            _task_num = i - idx*interval  # 0-based position of the FAILED task within this batch
            print(f"[GP] ERROR: task {_task_num + 1}/{total_tasks} (population="
                  f"{sample.loc[i,'population']!r}, phenotype={sample.loc[i,'phenotype']!r}, "
                  f"ratio={sample.loc[i,'ratio']!r}, replicate={sample.loc[i,'sample']!r}) "
                  f"failed: {_task_exc!r}")
            print(f"[GP] Saving the {_task_num} task(s) completed successfully before this "
                  f"failure, so they aren't lost...")
            _ckpt.save_partial_results(
                RESULT_NAME, idx, _is_parallel,
                {'record': record, 'result_train': result_train, 'result_valid': result_valid,
                 'result_test': result_test, 'effect': effect, 'interactions': interactions,
                 'attention_total': attention_total, 'weight': weight, 'hp_record': hp_record,
                 'stats': stats},
                _static_write_flags,
            )
            _ckpt.save_checkpoint(RESULT_NAME, idx, _is_parallel, _sample_fp,
                                   last_completed_i=_task_num - 1, total_tasks=total_tasks)
            print(f"[GP] Partial results saved under './Result/{RESULT_NAME}/'. Fix the error "
                  f"above and re-submit the same job (same config{', same PARALLEL batch' if _is_parallel else ''}) "
                  f"- EasiGP will automatically resume from task {_task_num + 1}/{total_tasks} "
                  f"instead of starting over.")
            raise
        else:
            # Requirement (bugfix - output files existing but empty after
            # an OOM/walltime kill, so a resumed run silently re-does
            # everything): the block above only ever saves anything on a
            # CAUGHT Python exception - which a SIGKILL (OOM) or a
            # scheduler's walltime kill is not, and by design can never
            # be. Without a save point here too, a kill landing between
            # two successfully-finished tasks (or mid-way through the
            # next one) loses EVERY task this batch has completed, not
            # just the one actually running at the time - exactly the
            # reported symptom (checkpoint files existing, but the result
            # files they pointed at empty, because nothing had ever
            # written them). This runs after EVERY task that finishes
            # without raising (a plain try/except/else - 'else' means
            # "the try block did NOT raise", not "always", so this never
            # runs for a task the block above already handled) -
            # incrementally APPENDING just this one task's own new rows
            # (see append_partial_results()'s own docstring for why an
            # append, not the same full-rewrite save_partial_results()
            # uses, is what keeps this affordable to do on every single
            # task rather than only occasionally) - and advancing the
            # checkpoint to match, so a kill immediately afterward has
            # nothing left to lose except whatever task hadn't finished
            # yet, which is unavoidable no matter what strategy is used.
            _task_num = i - idx*interval  # 0-based position of the task that just finished
            _delta_frames = {
                'record': record.iloc[len(_pre_task_snapshot['record']):],
                'result_train': result_train.iloc[len(_pre_task_snapshot['result_train']):],
                'result_valid': result_valid.iloc[len(_pre_task_snapshot['result_valid']):],
                'result_test': result_test.iloc[len(_pre_task_snapshot['result_test']):],
                'effect': effect.iloc[len(_pre_task_snapshot['effect']):],
                'interactions': interactions.iloc[len(_pre_task_snapshot['interactions']):],
                'attention_total': attention_total.iloc[len(_pre_task_snapshot['attention_total']):],
                'weight': weight.iloc[len(_pre_task_snapshot['weight']):],
                'hp_record': hp_record.iloc[len(_pre_task_snapshot['hp_record']):],
                'stats': stats.iloc[len(_pre_task_snapshot['stats']):],
            }
            _ckpt.append_partial_results(RESULT_NAME, idx, _is_parallel, _delta_frames, _static_write_flags)
            _ckpt.save_checkpoint(RESULT_NAME, idx, _is_parallel, _sample_fp,
                                   last_completed_i=_task_num, total_tasks=total_tasks)

    _report_progress(completed_units, total_units, label='Finalising results...')

    # Run the (naive, arithmetic-mean) ensemble model in the end - same
    # per-method/across-methods grouping as the weighted W_OPT loop above,
    # and the same guarantee: collapses to the original single call whenever
    # nothing was multi-algorithm-tuned.
    if 'ensemble' in MODEL_BASE:
        for _group_label, _group_models in ensemble_groups(MODEL_BASE, HP_TUNE, HP_TUNE_ENSEMBLE_MODE):
            result_train, result_valid, result_test, sample_record, sample_effect = ensemble(result_train, result_valid, result_test, effect, _group_models)
            if _group_label is not None:
                _new_lbl = f'ensemble__{_group_label}'
                for _df in (result_train, result_valid, result_test):
                    if isinstance(_df, pd.DataFrame) and 'ensemble' in _df.columns:
                        _df.rename(columns={'ensemble': _new_lbl}, inplace=True)
                sample_record['model'] = _new_lbl
                if sample_effect.shape[0] != 0:
                    sample_effect['model'] = _new_lbl
            record = pd.concat([record, sample_record])
            effect = pd.concat([effect, sample_effect])
    
    # Store the results. Reuses checkpoint_utils.save_partial_results() -
    # the SAME helper an in-progress task failure above uses to persist
    # whatever completed before it - so a full, successful run and a
    # not-yet-finished, checkpointed run always produce byte-for-byte the
    # same kind of output for the data they each actually have (see that
    # helper's own docstring).
    _ckpt.save_partial_results(
        RESULT_NAME, idx, _is_parallel,
        {'record': record, 'result_train': result_train, 'result_valid': result_valid,
         'result_test': result_test, 'effect': effect, 'interactions': interactions,
         'attention_total': attention_total, 'weight': weight, 'hp_record': hp_record,
         'stats': stats},
        _static_write_flags,
    )

    # Every task in this batch finished successfully - nothing left to
    # resume, so the checkpoint (if one existed, from an earlier failed
    # attempt at this same batch) is no longer needed. Left in place, it
    # could otherwise make an unrelated FUTURE run that happens to reuse
    # this RESULT_NAME think it should skip tasks it hasn't actually run.
    _ckpt.clear_checkpoint(RESULT_NAME, idx, _is_parallel)

    return record, result_train, result_test, effect, interactions, POPULATION, PHENOTYPE, attention_total
