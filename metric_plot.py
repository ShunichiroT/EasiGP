import seaborn as sns
import matplotlib.pyplot as plt

from pipeline_utils import canonical_model_order

# Requirement: give the violin plots the same font/figure-size adjustment
# scatter_plot() already offers, for consistency between the two. Defaults
# match this function's previous hard-coded behaviour exactly (font_scale=1,
# FacetGrid's own default height=5), so any caller that doesn't pass
# METRIC_PLOT_CONFIG (e.g. an older/unreviewed script) keeps working
# unchanged.
_DEFAULT_METRIC_PLOT_CONFIG = {'font_size': 1, 'fig_size': 5}

# ver4-4 R7.h: PLOT_DPI replaces the previous hard-coded dpi=600 for both
# 'figure.dpi' and 'savefig.dpi' below. Default kept at 600 here (this
# function's own, legacy-preserving default if a caller omits the
# argument) - callers thread the actual PLOT_DPI config value through
# (default 300 there, per the blueprint's §4a reproducibility policy).


def metric_plot(record, MODEL, RESULT_NAME, SCENARIO, METRIC_PLOT_CONFIG=None, PLOT_DPI=600):

    # The naive ensemble ('ensemble', or 'ensemble__<algo>' when tuned
    # per-method) is computed once, in GP()'s finalisation step, strictly
    # AFTER the per-task loop that produces every weighted-ensemble
    # ('Nelder Mead' etc.) row - so any caller deriving MODEL straight
    # from Metric.csv's own row order (e.g. the HP_TUNE fallback both
    # headless runners use) would otherwise place it after every
    # weighted-ensemble method's violin instead of before, as intended.
    # Applied here too - not just at the call sites - so this function's
    # own `hue_order` is always correct regardless of how the caller built
    # MODEL. A no-op whenever MODEL doesn't contain both an ensemble-family
    # and a weighted-ensemble-family label - see canonical_model_order()'s
    # own docstring.
    MODEL = canonical_model_order(MODEL)

    if METRIC_PLOT_CONFIG is None:
        METRIC_PLOT_CONFIG = _DEFAULT_METRIC_PLOT_CONFIG
    font_size = METRIC_PLOT_CONFIG.get('font_size', _DEFAULT_METRIC_PLOT_CONFIG['font_size'])
    fig_size = METRIC_PLOT_CONFIG.get('fig_size', _DEFAULT_METRIC_PLOT_CONFIG['fig_size'])

    record = record.reset_index(drop=True)
    metrics = ['Pearson correlation', 'MSE']
    record['x'] = 'Model'
    
    if SCENARIO == 'between':
        record['population'] = record['population'].str.split('->', expand=True).iloc[:,-1]
    
    sns.set_theme(style="ticks",font_scale = font_size, rc={"figure.dpi":PLOT_DPI, 'savefig.dpi':PLOT_DPI})
    
    for i in range(len(metrics)):
        ax_share = False if metrics[i] == 'MSE' else True
        
        g = sns.FacetGrid(record, col="phenotype", row='population', sharey=ax_share, height=fig_size)
        
        for axis in g.axes.flat:
            axis.tick_params(labelleft=True) 
                    
        g.map(sns.violinplot,'x', metrics[i], 'model', palette='colorblind', hue_order=MODEL)
        #, hue_order=MODEL)    
        
        for axis in g.axes.flat:
            axis.set_ylabel(metrics[i])  
            axis.set_xlabel("")  
        
        plt.tight_layout()
        g.add_legend(loc='lower right')
        g.savefig('./Result/'+RESULT_NAME+'/'+metrics[i]+'.png') 
        
        #======#
        
        g = sns.FacetGrid(record, col="phenotype", sharey=ax_share, height=fig_size)
        
        for axis in g.axes.flat:
            axis.tick_params(labelleft=True) 
                    
        g.map(sns.violinplot,'x', metrics[i], 'model', palette='colorblind', hue_order=MODEL)
        #, hue_order=MODEL)    
        
        for axis in g.axes.flat:
            axis.set_ylabel(metrics[i])  
            axis.set_xlabel("")  
        
        plt.tight_layout()
        g.add_legend(loc='lower right')
        g.savefig('./Result/'+RESULT_NAME+'/'+metrics[i]+'_total.png') 