import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def scatter_plot(MODEL, PHENOTYPE, predicted_result_test, effect, QTL, SCATTER_CONFIG):
    
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
    sns.set_theme(style="whitegrid", font_scale = SCATTER_CONFIG['font_size'], rc={"figure.dpi":300, 'savefig.dpi':300})
    markers = {"non-QTL": "s", "QTL": "v", "phenotype": "o"}
    figsize = SCATTER_CONFIG['fig_size']
    
    # Read QTL data if needed
    if QTL != None:
        QTL_info = pd.read_csv(QTL+'.csv')
    
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
                tmp = effect_selected[effect_selected['type']==MODEL[jj]].reset_index(drop=True)
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
            effect_formatted.loc[effect_formatted['marker'].isin(QTL_info_selected['marker'].tolist()),'level']='QTL'   
        
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
        plt.savefig('./Result/Scatter_plot_'+PHENOTYPE[k]+'.png')
        