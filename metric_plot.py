import seaborn as sns
import matplotlib.pyplot as plt

# Requirement: give the violin plots the same font/figure-size adjustment
# scatter_plot() already offers, for consistency between the two. Defaults
# match this function's previous hard-coded behaviour exactly (font_scale=1,
# FacetGrid's own default height=5), so any caller that doesn't pass
# METRIC_PLOT_CONFIG (e.g. an older/unreviewed script) keeps working
# unchanged.
_DEFAULT_METRIC_PLOT_CONFIG = {'font_size': 1, 'fig_size': 5}


def metric_plot(record, MODEL, RESULT_NAME, SCENARIO, METRIC_PLOT_CONFIG=None):

    if METRIC_PLOT_CONFIG is None:
        METRIC_PLOT_CONFIG = _DEFAULT_METRIC_PLOT_CONFIG
    font_size = METRIC_PLOT_CONFIG.get('font_size', _DEFAULT_METRIC_PLOT_CONFIG['font_size'])
    fig_size = METRIC_PLOT_CONFIG.get('fig_size', _DEFAULT_METRIC_PLOT_CONFIG['fig_size'])

    record = record.reset_index(drop=True)
    metrics = ['Pearson correlation', 'MSE']
    record['x'] = 'Model'
    
    if SCENARIO == 'between':
        record['population'] = record['population'].str.split('->', expand=True).iloc[:,-1]
    
    sns.set_theme(style="ticks",font_scale = font_size, rc={"figure.dpi":600, 'savefig.dpi':600})
    
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