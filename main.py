from genomic_prediction import *
from scatter_plot import *
from circos_plot import *
  

# Change below
### ============= ###
# 1. Genomic prediction configuration

# Assign path for R if causing an error
R_PATH = None   #'C:\Program Files\R\R-4.4.0'

# Your target phenotypes
PHENOTYPE = ['days2anthesis', 'asi']  #or 'all' 

# Genomic prediction models to run
MODEL = ['rrBLUP', 'BayesB', 'RF', 'ensemble']  

# training set ratio
RATIO = [0.8]     

# Number of iterations with random sampling for training & test sets
ITER_NUM = 1

# File paths for your genotype & phenotype files
GENOTYPE_FILE_NAME = './Data/MaizeNAM/MaizeNAM_dataset_genotype_population_1' 
PHENOTYPE_FILE_NAME = './Data/MaizeNAM/MaizeNAM_dataset_phenotype_population_1' 

# Hyperparameters
# rrBLUP:                          iteration number, burn-in
# BayesB:                          iteration number, burn-in
# RKHS:                            iteration number, burn-in, number of samples for Shapley scores, return marker effect?
# RF:                              tree number, maximum features per tree, maximum samples per tree, number of samples for Shapley scores, return marker effect?
# SVR:                             kernel type, epsilon, regularisation, dimension for poly kernel, gamma, number of samples for Shapley scores, return marker effect?
# MLP:                             neuron numbers, dropout, learning rate, decay, epoch, batch size, number of samples for Shapley scores
# GAT_infinitesimal_node_level:    neuron numbers, dropout, learning rate, decay, epoch, batch size, number of heads, number of samples for Shapley scores, return marker effect?
# GAT_infinitesimal:               neuron numbers, dropout, learning rate, decay, epoch, batch size, number of heads, number of samples for Shapley scores, return marker effect?
# GAT_fully_connected:             neuron numbers, dropout, learning rate, decay, epoch, batch size, number of heads, number of samples for Shapley scores, return marker effect?
# GAT_prior_knowledge:             neuron numbers, dropout, learning rate, decay, epoch, batch size, number of heads, number of samples for Shapley scores, 
#                                  selection rate for edges from RF (e.g. 0.1= selection of the top 10% of the most important edges), return marker effect?
HPARAMETERS = {'rrBLUP': [10000, 2000],     
               'BayesB': [12000, 2000],    
               'RKHS': [12000, 2000, 30, False],   
               'RF': [1000, 1.0, None, 30, True],           
               'SVR':['rbf', 0.5, 1.0, 3, 'scale', 30, False],      
               'MLP':[30, 0, 0.001, 5e-4, 30, 8, 10],
               'GAT_infinitesimal_node_level':[20, 0.9, 0.005, 5e-4, 1, 8, 1, 30, True],
               'GAT_infinitesimal':[20, 0, 0.01, 5e-4, 40, 8, 1, 1, True],
               'GAT_fully_connected':[20, 0, 0.01, 5e-4, 40, 8, 1, 1, True],
               'GAT_prior_knowledge':[20, 0, 0.01, 5e-4, 40, 8, 1, 1, 0.2, True]} 

# 2.Scatter plot matrix configuration

# True if scatter plot is needed. False otherwise
SCATTER_CREATE = True 

# Write a file path if you want to add QTL information. None otherwise
QTL = None 

# Parameters
# font_size:  font size for scatter plot matrix
# fig_size:   size of scatter plot matrix
SCATTER_CONFIG = {'font_size':2,
                  'fig_size':30}

# 3. Circos plot configuration
# File path for the information of all markers
MARKER_INFO = './Data/MaizeNAM/marker_info'

# File path for the information of chromosomes
CHROMOSOME_INFO = './Data/MaizeNAM/chrom'

# File path for the gene information
GENE_INFO = './Data/MaizeNAM/gene_info'

# Parameters
# space:           space size between rings
# start:           start angle of a ring
# end:             end angle of a ring
# link_width:      the thickness of links
# interaction_top: select only the top N% of strongest links in the ratio form
# label_size:      size of font
# scale:           scale of circos plot  
CIRCOS_CONFIG = {'space':3,
                 'start':15,
                 'end':345,
                 'link_width':10,
                 'interaction_top':0.9999,
                 'label_size':6,
                 'scale':10000}

# adjust the end location of each marker for visualisation
end_adjust = 10

# colour palette for circos plot
CYTOBAND_COLORMAP = {   
       "gpos100": "#000000", # 0,0,0
       "gpos": "#000000",    # 0,0,0
       "gpos75": "#828282",  # 130,130,130
       "gpos66": "#A0A0A0",  # 160,160,160
       "gpos50": "#C8C8C8",  # 200,200,200
       "gpos33": "#D2D2D2",  # 210,210,210
       "gpos25": "#C8C8C8",  # 200,200,200
       "gvar": "#DCDCDC",    # 220,220,220
       "gneg": "#FFFFFF",    # 255,255,255
       "acen": "#D92F27",    # 217,47,39
       "stalk": "#647FA4",   # 100,127,164
       "green": "#47c462",
       "brown": "#e0a22f",
       "purple": "#a62bcc",
       "blue1": "#def2ff",
       "blue2": "#c2e5fc",
       "blue3": "#addeff",
       "blue4": "#99d6ff",
       "blue5": "#83ccfc",
       "blue6": "#68c1fc",
       "blue7": "#45b5ff",
       "blue8": "#14a0fc",
       "blue9": "#027ac9",
       "blue10": "#014f82",
       "red1": "#fce1a7",
       "red2": "#ffd780",
       "red3": "#ffc954",
       "red4": "#fcba2b",
       "red5": "#ffaf03",
       "red6": "#d99502",
       "red7": "#b57c02",
       "red8": "#8a5e01",
       "red9": "#874001",
       "red10": "#610901"
   }
### ============= ###

# Run genomic prediction models
metrics, predicted_result_train, predicted_result_test, effect, interactions, POPULATION, PHENOTYPE = \
    GP(GENOTYPE_FILE_NAME, PHENOTYPE_FILE_NAME, MODEL, PHENOTYPE, RATIO, ITER_NUM, HPARAMETERS, R_PATH)

# Generate scatter plot matrices if needed
if SCATTER_CREATE:
    scatter_plot(MODEL, PHENOTYPE, predicted_result_test, effect, QTL, SCATTER_CONFIG)

# Generate circos plots
circos_plot(effect, interactions, MARKER_INFO, CHROMOSOME_INFO, GENE_INFO, POPULATION, PHENOTYPE, MODEL, CIRCOS_CONFIG, end_adjust, CYTOBAND_COLORMAP) 
