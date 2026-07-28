import seaborn as sns
import matplotlib.pyplot as plt

def metric_plot(record, MODEL, RESULT_NAME, SCENARIO):
    
    record = record.reset_index(drop=True)
    metrics = ['Pearson correlation', 'MSE']
    record['x'] = 'Model'
    
    if SCENARIO == 'between':
        record['population'] = record['population'].str.split('->', expand=True).iloc[:,-1]
    
    sns.set_theme(style="ticks",font_scale = 1, rc={"figure.dpi":600, 'savefig.dpi':600})
    
    for i in range(len(metrics)):
        ax_share = False if metrics[i] == 'MSE' else True
        
        g = sns.FacetGrid(record, col="phenotype", row='population', sharey=ax_share)
        
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
        
        g = sns.FacetGrid(record, col="phenotype", sharey=ax_share)
        
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