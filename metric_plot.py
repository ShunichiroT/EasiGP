import seaborn as sns
import matplotlib.pyplot as plt

def metric_plot(record, MODEL):
    
    metrics = ['Pearson correlation', 'MSE']
    record['models'] = 'models'
    sns.set_theme(style="ticks",font_scale = 1, rc={"figure.dpi":300, 'savefig.dpi':300})
    
    for i in range(len(metrics)):
        ax_share = False #if metrics[i] == 'MSE' else True
        
        g = sns.FacetGrid(record, col="phenotype", row='population', sharey=ax_share)
        
        for axis in g.axes.flat:
            axis.tick_params(labelleft=True) 
                    
        g.map(sns.violinplot,'models', metrics[i], 'type', palette='colorblind', hue_order=MODEL)    
        
        for axis in g.axes.flat:
            axis.set_ylabel(metrics[i])  
            axis.set_xlabel("")  
        
        plt.tight_layout()
        g.add_legend()
        g.savefig("./Result/"+metrics[i]+".png") 