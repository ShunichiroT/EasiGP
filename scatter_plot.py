import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from pipeline_utils import unify_columns_by_position, markers_in_windows


def _save_scatter_legend(save_path, *, dpi=300):
    """Requirements.md item 3: a standalone legend PNG for the scatter-plot
    matrix, generated alongside the Scatter_plot_<phenotype>.png files -
    mirrors circos_plot.py's own '_legend.png' pattern (a separate,
    dedicated legend image, since the matrix's own per-subplot axes are
    already crowded and each subplot has its own legend removed - see the
    `axes[i,j].get_legend().remove()` call below).

    Three fixed entries, matching this module's own `markers`/`palette`
    dicts (used for the real scatter points) exactly - content never
    varies between calls, so ONE legend file is generated per
    scatter_plot() call, shared across every phenotype's own matrix,
    rather than one per phenotype:
      - green dot       : predicted phenotype (the 'phenotype' level -
        predicted vs. actual)
      - blue rectangle   : marker effect, non-QTL markers
      - orange triangle  : marker effect, markers flagged as QTL
    """
    handles = [
        mlines.Line2D([0], [0], marker='o', linestyle='None', markerfacecolor='g',
                      markeredgecolor='black', markersize=9, label='Predicted phenotype'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor='#377eb8', edgecolor='black',
                            linewidth=0.6, label='Marker effect'),
        mlines.Line2D([0], [0], marker='v', linestyle='None', markerfacecolor='#ff7f00',
                      markeredgecolor='black', markersize=9, label='Marker effect (QTL)'),
    ]
    legend_fig = plt.figure(figsize=(3.0, 1.6))
    legend_fig.legend(handles=handles, loc='center', fontsize=9, title='Legend',
                       title_fontsize=10, frameon=True)
    legend_fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(legend_fig)


def scatter_plot(MODEL, PHENOTYPE, predicted_result_test, effect, QTL, SCATTER_CONFIG, RESULT_NAME,
                  MARKER_INFO=None, QTL_WINDOW=0.0, QTL_WINDOW_MODE='all_in_window', PLOT_DPI=300):
    """Draw one scatter-plot matrix per phenotype, comparing every pair of
    selected models at both the predicted-phenotype and marker-effect
    levels, and (optionally) highlighting markers known/believed to be
    near a real QTL.

    ver4-4 R6: the QTL file can now be given in EITHER of two schemas,
    detected automatically by column count:

      - **Legacy (2 columns): 'phenotype', 'marker'.** Exact marker-NAME
        matching, byte-identical to ver4-3 - a marker is flagged 'QTL'
        only if its name appears verbatim in this file for this
        phenotype. `MARKER_INFO`/`QTL_WINDOW`/`QTL_WINDOW_MODE` are
        accepted but have no effect in this mode.
      - **Coordinate (>=5 columns): 'chromosome', 'start', 'end', 'name',
        'phenotype' [, 'colour']** (the `gene_info.csv` column set, so one
        file can serve both circos gene rings and scatter highlighting -
        ver4-4 Design Blueprint S2.6.2). A marker is flagged 'QTL' if it
        falls within `QTL_WINDOW` of a QTL region for this phenotype
        (`QTL_WINDOW_MODE='all_in_window'`) or is the single nearest
        marker to it (`QTL_WINDOW_MODE='nearest'`) - see
        `pipeline_utils.markers_in_windows()`, which this delegates to
        rather than re-implementing the interval-overlap test (I6).
        Requires `MARKER_INFO` (a marker_info.csv path - already a
        required config key for circos, so no new user burden).

    This was hidden (commented out in the GUI) in ver4-3 because exact-
    name matching essentially never fires in practice - a genotyped
    marker rarely coincides exactly with a reported QTL position - not
    because the underlying styling/highlighting machinery was broken.
    ver4-4 R6 fixes the ROOT CAUSE (exact-name-only matching) rather than
    only re-exposing the same, largely-inert, feature.

    Parameters
    ----------
    MODEL : list of str
        Model labels to include (matches columns already present in
        `predicted_result_test`/`effect`).
    PHENOTYPE : list of str
        Phenotypes to plot - one scatter-plot matrix PNG per entry.
    predicted_result_test : pandas.DataFrame
        `Prediction_result_test.csv`-shaped frame (or the in-memory
        equivalent).
    effect : pandas.DataFrame
        `Marker_effect.csv`-shaped frame (or the in-memory equivalent).
    QTL : str or None
        Path to the QTL file (either schema above), or `None` - `None`
        reproduces today's behaviour exactly (no QTL highlighting at
        all).
    SCATTER_CONFIG : dict
        `{'font_size': int, 'fig_size': int}`.
    RESULT_NAME : str
        Output folder name under `./Result/`.
    MARKER_INFO : str or None, default None
        Path to `marker_info.csv` (`chromosome, name, start, end`).
        Required (raises `ValueError` otherwise) only when `QTL` is in
        coordinate mode; ignored entirely for legacy-mode QTL files or
        when `QTL` is `None`.
    QTL_WINDOW : float, default 0.0
        Symmetric window width (same units as `MARKER_INFO`'s
        `start`/`end` columns) used for coordinate-mode QTL matching.
        `0.0` reproduces exact-overlap-only matching. Ignored in legacy
        mode.
    QTL_WINDOW_MODE : {'all_in_window', 'nearest'}, default
        'all_in_window'. Selects which `pipeline_utils.markers_in_windows`
        assignment strategy is used for coordinate-mode QTL matching.
        Ignored in legacy mode.
    PLOT_DPI : int, default 300
        Resolution (dots per inch) used for both the per-phenotype
        scatter-plot-matrix PNGs and the shared legend PNG. ver4-4 R7.h
        gave `metric_plot()`/`circos_plot()` this same caller-supplied
        `PLOT_DPI` knob (replacing their own previously hard-coded
        values); this brings `scatter_plot()`'s own previously
        hard-coded `dpi=300` (both the `sns.set_theme` rc values and the
        legend save) in line with them, so a single `cfg['PLOT_DPI']`
        controls every raster plot a run produces. `300` is kept as this
        function's own default so a caller that omits the argument keeps
        today's exact behaviour.

    Raises
    ------
    ValueError
        If the QTL file's column count matches neither accepted schema,
        or if it's in coordinate mode but `MARKER_INFO` is not provided.
    """

    #if 'Linear transformation' in list(predicted_result_test.columns):
    #     predicted_result_test = predicted_result_test.drop('Linear transformation', axis=1)
    #elif 'Nelder_Mead' in list(predicted_result_test.columns):
    #    predicted_result_test = predicted_result_test.drop('Nelder Mead', axis=1)
    #elif 'Bayesian optimisation' in list(predicted_result_test.columns):
    #    predicted_result_test = predicted_result_test.drop('Bayesian optimisation', axis=1)
    
    #if 'Linear transformation' in effect['type'].values:
    #     effect = effect[effect['type']!='Linear transformation'].reset_index(drop=True)
    #elif 'Nelder_Mead' in effect['type'].values:
    #     effect = effect[effect['type']!='Nelder_Mead'].reset_index(drop=True)
    #elif 'Bayesian optimisation' in effect['type'].values:
    #     effect = effect[effect['type']!='Bayesian optimisation'].reset_index(drop=True) 
    
    model_selected = MODEL.copy()
    if 'ensemble' in MODEL:
        model_selected.remove('ensemble')
        
    # Setting
    sns.set_theme(style="whitegrid", font_scale = SCATTER_CONFIG['font_size'], rc={"figure.dpi":PLOT_DPI, 'savefig.dpi':PLOT_DPI})
    markers = {"non-QTL": "s", "QTL": "v", "phenotype": "o"}
    figsize = SCATTER_CONFIG['fig_size']

    # Requirements.md item 3: one shared legend PNG per scatter_plot() call
    # (see _save_scatter_legend()'s own docstring for why this isn't
    # generated once per phenotype - the three entries never change).
    _save_scatter_legend('./Result/'+RESULT_NAME+'/Scatter_plot_legend.png', dpi=PLOT_DPI)
    
    # ver4-4 R6: dual-schema QTL file read - detected by column count,
    # exactly as the Design Blueprint (S2.6.2) specifies, rather than a
    # config flag the user has to set correctly themselves. Read and
    # schema-detected ONCE here, outside the phenotype loop below (the
    # same "don't re-read per iteration" discipline ver4-4 R7.b/f already
    # applied to circos_plot.py's own per-model/per-population re-reads).
    QTL_info = None
    _qtl_mode = None
    marker_info_df = None
    if QTL != None:
        QTL_raw = pd.read_csv(QTL)
        _n_qtl_cols = QTL_raw.shape[1]
        if _n_qtl_cols >= 5:
            _qtl_mode = 'coordinate'
            # Requirement 8 / I6: unify by position - 'chromosome', 'start',
            # 'end', 'name', 'phenotype' here are this file's STRUCTURAL
            # column headers (the fixed schema position), not the QTL/
            # marker names themselves (the VALUES, matched against real
            # marker names elsewhere via markers_in_windows() and left
            # completely untouched). Any 6th ('colour') column, if
            # present, is left exactly as-is (first_n=5 renames only the
            # first 5 columns).
            QTL_info = unify_columns_by_position(
                QTL_raw, ['chromosome', 'start', 'end', 'name', 'phenotype'],
                'QTL file', first_n=5)
            QTL_info.iloc[:,-3] = QTL_info.iloc[:,-1]
            QTL_info = QTL_info.iloc[:, :-2]
        elif _n_qtl_cols == 2:
            _qtl_mode = 'legacy'
            # Requirement 8: unify by position (phenotype, marker) -
            # 'marker' here is this file's marker-identifying column
            # HEADER (a fixed, structural field per the GUI's own
            # documented legacy schema: 'phenotype|marker name identified
            # as QTL'), not the marker names themselves (the VALUES,
            # matched against real marker names elsewhere and left
            # completely untouched).
            QTL_info = unify_columns_by_position(QTL_raw, ['phenotype', 'marker'], 'QTL file')
        else:
            raise ValueError(
                f"[scatter_plot] QTL file '{QTL}' has {_n_qtl_cols} column(s); expected either "
                f"exactly 2 ('phenotype', 'marker' - legacy exact marker-name matching) or at "
                f"least 5 ('chromosome', 'start', 'end', 'name', 'phenotype'[, 'colour'] - "
                f"ver4-4 R6 coordinate-mode window matching). Got columns: {list(QTL_raw.columns)}."
            )

        if _qtl_mode == 'coordinate':
            if not MARKER_INFO:
                raise ValueError(
                    "[scatter_plot] QTL file is in coordinate mode (>=5 columns: chromosome, "
                    "start, end, name, phenotype[, colour]), which requires MARKER_INFO (the "
                    "same marker_info.csv path already used for the circos gene/marker rings - "
                    "see the architecture doc's Appendix B) so QTL regions can be mapped to "
                    "nearby markers by genomic position. MARKER_INFO was not provided."
                )
            marker_info_df = pd.read_csv(MARKER_INFO)
            # Requirement 8: same reasoning as circos_plot.py's own
            # marker_info.csv load - 'chromosome', 'name', 'start', 'end'
            # are this file's structural column headers, not marker
            # identities.
            marker_info_df = unify_columns_by_position(
                marker_info_df, ['chromosome', 'name', 'start', 'end'], 'marker info file')
            print(
                f"[scatter_plot] QTL file detected as COORDINATE mode ({_n_qtl_cols} columns) - "
                f"markers within QTL_WINDOW={QTL_WINDOW} (QTL_WINDOW_MODE={QTL_WINDOW_MODE!r}) of "
                f"a QTL region will be flagged, rather than requiring an exact marker-name match."
            )
        else:
            print(
                "[scatter_plot] QTL file detected as LEGACY mode (2 columns: phenotype, marker) "
                "- exact marker-name matching, as in ver4-3. QTL_WINDOW/QTL_WINDOW_MODE have no "
                "effect in this mode."
            )
    
    # Generate a scatter plot per phenotype
    for k in range(len(PHENOTYPE)):
        
        # Change the format of the data for scatter plot matrix
        predicted_test_formatted = predicted_result_test[predicted_result_test['phenotype']==PHENOTYPE[k]].iloc[:,6:].reset_index(drop=True)
        effect_selected = effect[effect['phenotype']==PHENOTYPE[k]].reset_index(drop=True)
        
        effect_formatted = pd.DataFrame()
        for jj in range(len(MODEL)):
            if MODEL[jj] == 'ensemble':
                continue
            else:
                tmp = effect_selected[effect_selected['model']==MODEL[jj]].reset_index(drop=True)
                if tmp.shape[0] != 0:
                    tmp = tmp.iloc[:,5:].melt()
                    tmp.columns = ['marker',MODEL[jj]]
                    if effect_formatted.shape[0] == 0:
                         effect_formatted = tmp
                    else:
                        effect_formatted = pd.concat([effect_formatted,
                                                      tmp.iloc[:,1:]],axis=1)
        
        for kk in range(1,effect_formatted.shape[1]):
            effect_formatted.iloc[:,kk] = effect_formatted.iloc[:,kk].abs()
        
        # Add the information of level
        predicted_test_formatted['level'] = 'phenotype'
        effect_formatted['level'] = 'non-QTL'
        
        if QTL:
            QTL_info_selected = QTL_info[QTL_info['phenotype']==PHENOTYPE[k]]
            if _qtl_mode == 'legacy':
                qtl_markers = set(QTL_info_selected['marker'].tolist())
            else:
                # ver4-4 R6: coordinate-mode window matching. `available_markers`
                # is restricted to the markers actually present in THIS
                # phenotype's effect table (mirroring
                # _map_genes_to_markers's own "the genotype table actually
                # being modelled for this task" restriction) - not every
                # marker on this file's chromosome, and not markers from a
                # different phenotype/model combination.
                available_markers = effect_formatted['marker'].tolist()
                region_to_markers = markers_in_windows(
                    QTL_info_selected, marker_info_df, available_markers,
                    window=QTL_WINDOW, mode=QTL_WINDOW_MODE)
                qtl_markers = set()
                for _markers in region_to_markers:
                    qtl_markers.update(_markers)
            effect_formatted.loc[effect_formatted['marker'].isin(qtl_markers), 'level'] = 'QTL'
            # Requirement (diagnostic, matches the "log negative/informative
            # states unconditionally" convention this codebase already
            # follows elsewhere - architecture doc S17): how many markers
            # ended up flagged, out of how many considered, so a QTL_WINDOW
            # set far too large (flagging almost everything) or far too
            # small (flagging nothing) is immediately visible in the log,
            # not just guessable from the rendered plot.
            _n_flagged = int((effect_formatted['level'] == 'QTL').sum())
            print(
                f"[scatter_plot] phenotype={PHENOTYPE[k]!r}: {_n_flagged} of "
                f"{effect_formatted.shape[0]} marker(s) flagged as QTL "
                f"({QTL_info_selected.shape[0]} QTL region(s)/row(s) considered)."
            )
        
        # Combine both predicted phenotype and marker effect information
        data_scatter = pd.concat([predicted_test_formatted,
                                  effect_formatted.iloc[:,1:]]).fillna(0)
        
        #if 'ensemble' in list(data_scatter.columns):
        #    data_scatter = data_scatter.drop('ensemble',axis=1)
        
        # Set the matrix size
        fig, axes = plt.subplots(len(model_selected), len(model_selected), figsize=(figsize, figsize))
        
        # Determine which subplots show marker effects 
        lower = np.arange(0, (len(model_selected))*(len(model_selected))).reshape(len(model_selected),len(model_selected))
        lower = list(lower[np.tril_indices(len(model_selected), k = -1)])
        
        # Generate a subplot per model combination in both types
        for i in range(len(model_selected)):
            for j in range(len(model_selected)):
                if i == j:
                    continue
                elif (len(model_selected)*i)+j in lower:
                    extracted = data_scatter[data_scatter['level'] != 'phenotype']
                    sns.scatterplot(ax=axes[i, j], data=extracted, x=model_selected[j], y=model_selected[i],hue=extracted['level'],
                                    style=extracted['level'],
                                    markers=markers,
                                    palette={'non-QTL':'#377eb8','QTL':'#ff7f00',"phenotype":'g'})
                else:
                    extracted = data_scatter[data_scatter['level'] == 'phenotype']
                    sns.scatterplot(ax=axes[i, j], data=extracted, x=model_selected[j], y=model_selected[i],
                                    hue=extracted['level'],
                                    palette={'non-QTL':'#377eb8','QTL':'#ff7f00',"phenotype":'g'}) 
                try:
                    axes[i,j].get_legend().remove()
                except:
                    continue
        plt.tight_layout()
        plt.savefig('./Result/'+RESULT_NAME+'/Scatter_plot_'+PHENOTYPE[k]+'.png', dpi=PLOT_DPI)
        
