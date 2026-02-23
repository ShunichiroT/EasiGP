import matplotlib as mpl
import matplotlib.pyplot as plt


def attention_distribution(attention, RESULT_NAME, bins):
    mpl.rcParams['figure.dpi'] = 600
    
    phenotypes = attention['phenotype'].unique()
    models = attention['model'].unique()
    
    for i in range(len(phenotypes)):
        for j in range(len(models)):
            attention_selected = attention.loc[(attention['phenotype']==phenotypes[i])&(attention['model']==models[j]),'value']
            
            plt.hist(attention_selected, color='blue', bins=bins)
            plt.xlabel('Attention weights')
            plt.ylabel('Frequency')
            plt.savefig('./Result/'+RESULT_NAME+'/Attention_histgram_'+phenotypes[i]+'_'+models[j]+'.png', bbox_inches='tight')