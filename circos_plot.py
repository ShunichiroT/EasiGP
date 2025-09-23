from pycirclize import Circos
import numpy as np
import pandas as pd

def data_conversion(chrom_info, gene_info, PHENOTYPE):
    chromosome = pd.read_csv(chrom_info+'.csv')
    chromosome_population = pd.unique(chromosome['population'])
    
    for i in range(len(chromosome_population)):
        chromosome_selected = chromosome[chromosome['population']==chromosome_population[i]]
        chromosome_selected = chromosome_selected.drop(['population'],axis=1)
        chromosome_selected.to_csv('./Result/chrom_'+str(chromosome_population[i])+'.bed', sep='\t', index=False)
    
    gene = pd.read_csv(gene_info+'.csv')
    gene_population = pd.unique(gene['population'])
    gene_source = pd.unique(gene['source'])
    for i in range(len(PHENOTYPE)):
        for j in range(len(gene_population)):
            for k in range(len(gene_source)):
                gene_selected = gene[(gene['population']==gene_population[j]) & (gene['phenotype']==PHENOTYPE[i]) & (gene['source']==gene_source[k])]
                gene_selected = gene_selected.drop(['source','population','phenotype'],axis=1)
                gene_selected.to_csv('./Result/gene_info_'+str(PHENOTYPE[i])+'_'+str(gene_source[k])+'_'+str(chromosome_population[j])+'.tsv', sep='\t', index=False)
    
    pop_source = gene.loc[:,['population','source']].drop_duplicates()

    return pop_source

def quantile_conversion(effect, marker_info, chrom_info, PHENOTYPE, MODEL, end_adjust, POPULATION):
    
    chromosome = pd.read_csv('./Result/chrom_'+str(POPULATION)+'.bed', delimiter='\t')
    
    # Convert genomic marker effects into ten level quantiles
    effect.iloc[:,5:] = effect.iloc[:,5:].abs()
    if POPULATION == 'all':
        effect_grouped = effect.iloc[:,1:].groupby(['phenotype','type']).mean()
    else:
        effect_grouped = effect.groupby(['population','phenotype','type']).mean()
    effect_grouped = effect_grouped.reset_index(drop=False)
    REMOVE = []
    for iii in range(len(MODEL)):
        colour = 'red' if MODEL[iii] == 'ensemble' else 'blue'
        if POPULATION == 'all':
            effect_selected = effect_grouped[(effect_grouped['type']==MODEL[iii]) & (effect_grouped['phenotype']==PHENOTYPE)].iloc[:,4:].T
        else:
            effect_selected = effect_grouped[(effect_grouped['type']==MODEL[iii]) & (effect_grouped['phenotype']==PHENOTYPE) & (effect_grouped['population']==POPULATION)].iloc[:,4:].T
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
            
            chromosome_total = pd.unique(merged['chromosome'])
            
            for k in range(len(chromosome_total)):
                merged.loc[(merged['chromosome']==chromosome_total[k]) & 
                           (merged['end'] > chromosome.loc[chromosome['Chromosome']==chromosome_total[k],'End'].values[0]),'end'] = int(chromosome.loc[chromosome['Chromosome']==chromosome_total[k], 'End'].values)
            merged.to_csv('./Result/marker_effect_'+str(MODEL[iii])+'_'+str(PHENOTYPE)+'_'+str(POPULATION)+'.tsv', sep ='\t',index=False)
        else:
            REMOVE += [MODEL[iii]]
    
    MODEL = [e for e in MODEL if e not in REMOVE]
    
    return MODEL

def interaction(interaction, marker_info, PHENOTYPE, circos_config, POPULATION):
    
    # Extract key gmarker-by-marker interaction patterns
    if POPULATION == 'all' and interaction.shape[0]!=0:
        interaction = interaction.loc[:,['phenotype','from','to', 'value']]
    elif POPULATION != 'all' and interaction.shape[0]!=0:
        interaction = interaction.loc[:,['population','phenotype','from','to', 'value']]
        interaction = interaction[interaction['population']==POPULATION].reset_index(drop=False)
    else:
        return pd.DataFrame()
    
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

    return interaction_selected

def plot(interactions, chrom_info, gene_info, pop_source, PHENOTYPE, MODEL, circos_config, CYTOBAND_COLORMAP, POPULATION):
  
    cnt = 0
    circos = Circos.initialize_from_bed('./Result/chrom_'+str(POPULATION)+".bed", space=circos_config['space'], start=circos_config['start'], end=circos_config['end'])
    
    # Add genomic marker effects
    for i in range(len(MODEL)):
         circos.add_cytoband_tracks((97-(3*cnt), 100-(3*cnt)), './Result/marker_effect_'+MODEL[i]+'_'+PHENOTYPE+'_'+str(POPULATION)+'.tsv', track_name=MODEL[i], cytoband_cmap=CYTOBAND_COLORMAP)
         circos.text(MODEL[i], r=circos.tracks[-1].r_center-1, deg=0, size=8, color="black")
         cnt+=1
    
    # Add known gene regions
    gene_source = pd.unique(pop_source.loc[pop_source['population']==str(POPULATION),'source'])
    for i in range(len(gene_source)):    
        circos.add_cytoband_tracks((97-(3*cnt), 100-(3*cnt)), './Result/gene_info_'+str(PHENOTYPE)+'_'+str(gene_source[i])+'_'+str(POPULATION)+'.tsv', track_name=gene_source[i], cytoband_cmap=CYTOBAND_COLORMAP)
        circos.text(gene_source[i], r=circos.tracks[-1].r_center-1, deg=0, size=8, color="black")
        cnt+=1
    
    # Add ticks to the outermost ring
    for sector in circos.sectors:
        sector.text(sector.name, r=110, size=10)
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
    fig.savefig('./Result/circos_'+str(PHENOTYPE)+'_'+str(POPULATION)+'.png',dpi=600) 

def circos_plot(effect, interactions, marker_info, chrom_info, gene_info, POPULATION, PHENOTYPE, MODEL, circos_config, end_adjust, CYTOBAND_COLORMAP):

    pop_source =  data_conversion(chrom_info, gene_info, PHENOTYPE)
    POPULATION = ('all',) + tuple(POPULATION)

    for i in range(len(PHENOTYPE)):
        for j in range(len(POPULATION)):
            MODEL = quantile_conversion(effect, marker_info, chrom_info, PHENOTYPE[i], MODEL, end_adjust, POPULATION[j])
            interaction_selected = interaction(interactions, marker_info, PHENOTYPE[i], circos_config, POPULATION[j])
            plot(interaction_selected, chrom_info, gene_info, pop_source, PHENOTYPE[i], MODEL, circos_config, CYTOBAND_COLORMAP, POPULATION[j])