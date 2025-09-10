from pycirclize import Circos
import numpy as np
import pandas as pd


def quantile_conversion(effect, SCOPE, marker_info, PHENOTYPE, MODEL, end_adjust, POPULATION = None):
    
    # Convert genomic marker effects into ten level quantiles
    effect.iloc[:,5:] = effect.iloc[:,5:].abs()
    if SCOPE == 'overall':
        effect_grouped = effect.iloc[:,1:].groupby(['phenotype','type']).mean()
        effect_grouped = effect_grouped.reset_index(drop=False)
        REMOVE = []
        for iii in range(len(MODEL)):
            colour = 'red' if MODEL[iii] == 'ensemble' else 'blue'
            effect_selected = effect_grouped[(effect_grouped['type']==MODEL[iii]) & (effect_grouped['phenotype']==PHENOTYPE)].iloc[:,4:].T
            if effect_selected.shape[1] != 0:
                effect_selected_copy = effect_selected.copy()
                effect_selected_copy.iloc[:,0] = colour+'1'
                tmp = list(effect_selected[effect_selected>= np.quantile(effect_selected.to_numpy().flatten(), 0.1)].dropna().index)
                effect_selected_copy.loc[tmp,:] = colour+'2'
                tmp = list(effect_selected[effect_selected>= np.quantile(effect_selected.to_numpy().flatten(), 0.2)].dropna().index)
                effect_selected_copy.loc[tmp,:] = colour+'3'
                tmp = list(effect_selected[effect_selected>= np.quantile(effect_selected.to_numpy().flatten(), 0.3)].dropna().index)
                effect_selected_copy.loc[tmp,:] = colour+'4'
                tmp = list(effect_selected[effect_selected>= np.quantile(effect_selected.to_numpy().flatten(), 0.4)].dropna().index)
                effect_selected_copy.loc[tmp,:] = colour+'5'
                tmp = list(effect_selected[effect_selected>= np.quantile(effect_selected.to_numpy().flatten(), 0.5)].dropna().index)
                effect_selected_copy.loc[tmp,:] = colour+'6'
                tmp = list(effect_selected[effect_selected>= np.quantile(effect_selected.to_numpy().flatten(), 0.6)].dropna().index)
                effect_selected_copy.loc[tmp,:] = colour+'7'
                tmp = list(effect_selected[effect_selected>= np.quantile(effect_selected.to_numpy().flatten(), 0.7)].dropna().index)
                effect_selected_copy.loc[tmp,:] = colour+'8'
                tmp = list(effect_selected[effect_selected>= np.quantile(effect_selected.to_numpy().flatten(), 0.8)].dropna().index)
                effect_selected_copy.loc[tmp,:] = colour+'9'
                tmp = list(effect_selected[effect_selected>= np.quantile(effect_selected.to_numpy().flatten(), 0.9)].dropna().index)
                effect_selected_copy.loc[tmp,:] = colour+'10'
                
                effect_selected_copy.columns = ['colour']
                effect_selected_copy = effect_selected_copy.reset_index(drop=False)
                marker = pd.read_csv(marker_info+'.csv')
                merged = pd.merge(effect_selected_copy, marker, left_on=['index'], right_on=['name'])
                merged = merged.loc[:,['chromosome','start','end','index','colour']]
                merged['start'] = (merged['start'] - end_adjust).round().astype(int)
                merged.loc[merged['start'] < 0, 'start'] = 0
                merged['end'] = (merged['end'] + end_adjust).round().astype(int)
                merged['chromosome'] = 'chr' + merged['chromosome'].astype(int).astype(str)
                
                merged.to_csv('./Result/marker_effect_'+str(MODEL[iii])+'_'+PHENOTYPE+'.tsv', sep ='\t',index=False)
            else:
                REMOVE += [MODEL[iii]]
    
    elif SCOPE == 'population':
        effect_grouped = effect.groupby(['population','phenotype','type']).mean()
        effect_grouped = effect_grouped.reset_index(drop=False)
         
        REMOVE = []
        for iii in range(len(MODEL)):
            colour = 'red' if MODEL[iii] == 'ensemble' else 'blue'
            effect_selected = effect_grouped[(effect_grouped['type']==MODEL[iii]) & (effect_grouped['phenotype']==PHENOTYPE) &
                                             (effect_grouped['population']==POPULATION)].iloc[:,5:].T
            if effect_selected.shape[1] != 0:
                effect_selected_copy = effect_selected.copy()
                effect_selected_copy.iloc[:,0] = colour+'1'
                tmp = list(effect_selected[effect_selected>= np.quantile(effect_selected.to_numpy().flatten(), 0.1)].dropna().index)
                effect_selected_copy.loc[tmp,:] = colour+'2'
                tmp = list(effect_selected[effect_selected>= np.quantile(effect_selected.to_numpy().flatten(), 0.2)].dropna().index)
                effect_selected_copy.loc[tmp,:] = colour+'3'
                tmp = list(effect_selected[effect_selected>= np.quantile(effect_selected.to_numpy().flatten(), 0.3)].dropna().index)
                effect_selected_copy.loc[tmp,:] = colour+'4'
                tmp = list(effect_selected[effect_selected>= np.quantile(effect_selected.to_numpy().flatten(), 0.4)].dropna().index)
                effect_selected_copy.loc[tmp,:] = colour+'5'
                tmp = list(effect_selected[effect_selected>= np.quantile(effect_selected.to_numpy().flatten(), 0.5)].dropna().index)
                effect_selected_copy.loc[tmp,:] = colour+'6'
                tmp = list(effect_selected[effect_selected>= np.quantile(effect_selected.to_numpy().flatten(), 0.6)].dropna().index)
                effect_selected_copy.loc[tmp,:] = colour+'7'
                tmp = list(effect_selected[effect_selected>= np.quantile(effect_selected.to_numpy().flatten(), 0.7)].dropna().index)
                effect_selected_copy.loc[tmp,:] = colour+'8'
                tmp = list(effect_selected[effect_selected>= np.quantile(effect_selected.to_numpy().flatten(), 0.8)].dropna().index)
                effect_selected_copy.loc[tmp,:] = colour+'9'
                tmp = list(effect_selected[effect_selected>= np.quantile(effect_selected.to_numpy().flatten(), 0.9)].dropna().index)
                effect_selected_copy.loc[tmp,:] = colour+'10'
                
                effect_selected_copy.columns = ['colour']
                effect_selected_copy = effect_selected_copy.reset_index(drop=False)
                
                marker = pd.read_csv(marker_info+'.csv')
                merged = pd.merge(effect_selected_copy, marker, left_on=['index'], right_on=['name'])
                merged = merged.loc[:,['chromosome','start','end','index','colour']]
                merged['start'] = (merged['start'] - end_adjust).round().astype(int)
                merged.loc[merged['start'] < 0, 'start'] = 0
                merged['end'] = (merged['end'] + end_adjust).round().astype(int)
                merged['chromosome'] = 'chr' + merged['chromosome'].astype(int).astype(str)
                
                merged.to_csv('./Result/marker_effect_'+str(POPULATION)+'_'+str(MODEL[iii])+'_'+PHENOTYPE+'.tsv', sep ='\t',index=False)
            else:
                REMOVE += [MODEL[iii]]
                
    MODEL = [e for e in MODEL if e not in REMOVE]
    
    return MODEL

def interaction(interaction, SCOPE, marker_info, PHENOTYPE, circos_config, POPULATION=None):
    
    # Extract key gmarker-by-marker interaction patterns
    if SCOPE == 'overall' and interaction.shape[0]!=0:
        interaction = interaction.loc[:,['phenotype','from','to', 'value']]
        interaction = interaction[(interaction['from'] != 'factor') & (interaction['to'] != 'factor')]
        interaction_total = interaction.groupby(['phenotype','from','to'], as_index=False).mean()
        
        interaction_selected = interaction_total[interaction_total['phenotype'] == PHENOTYPE]
        interaction_selected = interaction_selected[interaction_selected['value'] >= np.quantile(interaction_selected['value'], circos_config['interaction_top'])].reset_index(drop=True)
        interaction_selected['value'] = interaction_selected['value'] / interaction_selected['value'].sum()
        
        loc_info = pd.read_csv(marker_info+'.csv')
        
        start = pd.merge(interaction_selected['from'], loc_info, 'inner', left_on='from', right_on='name')
        end = pd.merge(interaction_selected['to'], loc_info, 'inner', left_on='to', right_on='name')
        chrom_start = 'chr'+ start['chromosome'].astype(int).astype(str)
        chrom_end = 'chr'+ end['chromosome'].astype(int).astype(str)

        interaction_selected = pd.concat([chrom_start, start.loc[:,['start','end']],
                                   chrom_end, end.loc[:,['start','end']],
                                   interaction_selected['value']],axis=1)
        interaction_selected.columns = ['chromosome_from', 'start','end','chromosome_to','start','end','value']
        
    elif SCOPE == 'population' and interaction.shape[0]!=0:
        
        interaction = interaction.loc[:,['population','phenotype','from','to', 'value']]
        interaction = interaction[(interaction['from'] != 'factor') & (interaction['to'] != 'factor')]
        interaction_total = interaction.groupby(['population','phenotype','from','to'], as_index=True).mean().reset_index(drop=False)
        
        interaction_selected = interaction_total[interaction_total['phenotype'] == PHENOTYPE]
        interaction_selected = interaction_selected[interaction_selected['value'] >= np.quantile(interaction_selected['value'], circos_config['interaction_top'])].reset_index(drop=True)
        interaction_selected['value'] = interaction_selected['value'] / interaction_selected['value'].sum()
        
        loc_info = pd.read_csv(marker_info+'.csv')
        
        start = pd.merge(interaction_selected['from'], loc_info, 'inner', left_on='from', right_on='name')
        end = pd.merge(interaction_selected['to'], loc_info, 'inner', left_on='to', right_on='name')
        chrom_start = 'chr'+ start['chromosome'].astype(int).astype(str)
        chrom_end = 'chr'+ end['chromosome'].astype(int).astype(str)

        interaction_selected = pd.concat([chrom_start, start.loc[:,['start','end']],
                                   chrom_end, end.loc[:,['start','end']],
                                   interaction_selected['value']],axis=1)
        interaction_selected.columns = ['chromosome_from', 'start','end','chromosome_to','start','end','value']
     
    elif interaction.shape[0]==0:
        interaction_selected = interaction
        
    return interaction_selected

def plot(interactions, SCOPE, chrom_info, gene_info, PHENOTYPE, MODEL, circos_config, CYTOBAND_COLORMAP, POPULATION = None):
    
    # Generate a circos plot 
    if SCOPE == 'overall':
        cnt = 0
        circos = Circos.initialize_from_bed(chrom_info+".bed", space=circos_config['space'], start=circos_config['start'], end=circos_config['end'])
        
        # Add genomic marker effects
        for i in range(len(MODEL)):
             circos.add_cytoband_tracks((97-(3*cnt), 100-(3*cnt)), './Result/marker_effect_'+MODEL[i]+'_'+PHENOTYPE+'.tsv', track_name=MODEL[i], cytoband_cmap=CYTOBAND_COLORMAP)
             circos.text(MODEL[i], r=circos.tracks[-1].r_center-1, deg=0, size=8, color="black")
             cnt+=1
        
        # Add known gene regions
        for i in range(len(gene_info)):
            circos.add_cytoband_tracks((97-(3*cnt), 100-(3*cnt)), gene_info[i]+'.tsv', track_name=gene_info[i], cytoband_cmap=CYTOBAND_COLORMAP)
            circos.text(gene_info[i].split("/")[-1], r=circos.tracks[-1].r_center-1, deg=0, size=8, color="black")
            cnt+=1
         
        # Add ticks to the outermost ring
        for sector in circos.sectors:
            sector.text(sector.name, r=120, size=10)
            sector.get_track(MODEL[0]).xticks_by_interval(
                circos_config['scale'],
                label_size=circos_config['label_size'],
                label_orientation="vertical",
                label_formatter=lambda v: f"{v / circos_config['scale']:.0f}",
            )
            
        # Add marker-by-marker interactions
        if interactions.shape[0] != 0:
            for ii in range(interactions.shape[0]):
                region1 = (interactions.iloc[ii,0], interactions.iloc[ii,1], interactions.iloc[ii,2])
                region2 = (interactions.iloc[ii,3], interactions.iloc[ii,4], interactions.iloc[ii,5])
                if interactions.iloc[ii,0] != interactions.iloc[ii,3]:   #within chromosome or between chromosome
                    colour = 'blue'
                else:
                    colour = 'red'
                circos.link(region1, region2, lw=interactions.loc[ii,'value']/circos_config['link_width'], color=colour)
                
        # Store the circos plot
        fig = circos.plotfig()
        fig.savefig('./Result/circos_'+PHENOTYPE+'.png',dpi=600) 
            
    elif SCOPE == 'population':
        
        cnt = 0
        circos = Circos.initialize_from_bed(chrom_info+"_"+POPULATION+".bed", space=circos_config['space'], start=circos_config['start'], end=circos_config['end'])
        
        # Add genomic marker effects
        for i in range(len(MODEL)):
             circos.add_cytoband_tracks((97-(3*cnt), 100-(3*cnt)), './Result/marker_effect_'+str(POPULATION)+'_'+MODEL[i]+'_'+PHENOTYPE+'.tsv', track_name=MODEL[i], cytoband_cmap=CYTOBAND_COLORMAP)
             circos.text(MODEL[i], r=circos.tracks[-1].r_center-1, deg=0, size=8, color="black")
             cnt+=1
             
        # Add known gene regions
        for i in range(len(gene_info)):
            circos.add_cytoband_tracks((97-(3*cnt), 100-(3*cnt)), gene_info[i]+'_'+POPULATION+'.tsv', track_name=gene_info[i], cytoband_cmap=CYTOBAND_COLORMAP)
            circos.text(gene_info[i].split("/")[-1], r=circos.tracks[-1].r_center-1, deg=0, size=8, color="black")
            cnt+=1
         
        # Add ticks to the outermost ring
        for sector in circos.sectors:
            sector.text(sector.name, r=120, size=10)
            sector.get_track(MODEL[0]).xticks_by_interval(
                circos_config['scale'],
                label_size=circos_config['label_size'],
                label_orientation="vertical",
                label_formatter=lambda v: f"{v / circos_config['scale']:.0f}",
            )
         
        # Add interactions
        if interactions.shape[0] != 0:
            for ii in range(interactions.shape[0]):
                region1 = (interactions.iloc[ii,0], interactions.iloc[ii,1], interactions.iloc[ii,2])
                region2 = (interactions.iloc[ii,3], interactions.iloc[ii,4], interactions.iloc[ii,5])
                if interactions.iloc[ii,0] != interactions.iloc[ii,3]:   #within chromosome or between chromosome
                    colour = 'blue'
                else:
                    colour = 'red'
                circos.link(region1, region2, lw=interactions.loc[ii,'value']/circos_config['link_width'], color=colour)
     
        # Store the circos plot
        fig = circos.plotfig()
        fig.savefig('./Result/circos_'+str(POPULATION)+'_'+PHENOTYPE+'.png',dpi=600)  
    

def circos_plot(effect, SCOPE, interactions, marker_info, chrom_info, gene_info, PHENOTYPE, MODEL, circos_config, end_adjust, CYTOBAND_COLORMAP, POPULATION=None):
    
    if SCOPE == 'overall':    
        for i in range(len(PHENOTYPE)):
            MODEL = quantile_conversion(effect, SCOPE, marker_info, PHENOTYPE[i], MODEL, end_adjust)
            interaction_selected = interaction(interactions, SCOPE, marker_info, PHENOTYPE[i],circos_config)
            plot(interaction_selected, SCOPE, chrom_info[0], gene_info[i], PHENOTYPE[i], MODEL, circos_config, CYTOBAND_COLORMAP)
    
    elif SCOPE == 'population':
        for k in range(len(POPULATION)):
            for i in range(len(PHENOTYPE)):
                MODEL = quantile_conversion(effect, SCOPE, marker_info, PHENOTYPE[i], MODEL, end_adjust, POPULATION[k])
                interaction_selected = interaction(interactions, SCOPE, marker_info, PHENOTYPE[i], circos_config, POPULATION[k])
                plot(interaction_selected, SCOPE, chrom_info[k], gene_info[k][i], PHENOTYPE[i], MODEL, circos_config, CYTOBAND_COLORMAP, POPULATION[k])