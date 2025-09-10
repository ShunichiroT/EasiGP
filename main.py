from genomic_prediction import *
from circos_plot import *
  

# Change below
### ============= ###
# 1. Genomic prediction configuration

R_PATH = None  #None # Assign path for R if causing an error

TOTAL_PHENOTYPE = 1           # Total nummber of phenotype columns in your dataset
PHENOTYPE = ['DTA']           # Your target phenotypes. ['DTA', 'RBN', 'RL' ,'DTF3','FT16', 'FT10'] options for Arabidopsis

MODEL = ['rrBLUP', 'BayesB', 'RKHS', 'RF', 'SVR', 'MLP', 'GAT', 'ensemble'] # Genomic prediction models to run

POPULATION = ['W22TIL01']    # Your target population in your dataset
RATIO = [0.8, 0.65, 0.5]     # training set ratio
ITER_NUM = 1                 # Number of iterations with random sampling ror training & test sets
DATA_NAME = './Data/TeoNAM/W22TIL01_subset' # Name of your genotype & phenotype dataset # Geno_pheno_adjusted_GRM_ver2'

# Hyprparameters
# rrBLUP: iteration number, burin-in
# BayesB: iteration number, burin-in
# RKHS:   iteration number, buin-in, number of samples for Shapley scores, return marker effect?
# RF:     tree number, maximum fearures per tree, maximum samples per tree, number of samples for Shapley scores, return marker effect?
# SVR:    kernel type, epsilon, regularisation, dimension for poly kernel, gamma, number of samples for Shapley scores, return marker effect?
# MLP:    neuron numbers, dropout, learning rate, decay, epoch, batch size, number of samples for Shapley scores
# GAT:    neuron numbers, dropout, learning rate, decay, epoch, batch size, number of heads, number of samples for Shapley scores, return marker effect?
HPARAMETERS = {'rrBLUP': [10000, 2000],     
               'BayesB': [12000, 2000],    
               'RKHS': [12000, 2000, 30, False],   
               'RF': [1000, 1.0, None, 30, True],           
               'SVR':['rbf', 0.5, 1.0, 3, 'scale', 30, False],      
               'MLP':[30, 0, 0.001, 5e-4, 30, 8, 10],
               'GAT':[20, 0.9, 0.005, 5e-4, 1, 8, 1, 30, False]} 


# 2. Circos plot configuration
SCOPE = 'overall' #overall or population

# File name containing the information of all markers
MARKER_INFO = './Data/TeoNAM/marker_info'

# File name array containing the length of chromosome
CHROMOSOME_INFO = ['./Data/TeoNAM/chrom']

# File name(s) nested array containing key known gene locations. 
# Each subarray indicates key gene file(s) for phenotype(s) in each population
# Each subsubarray (if there are more than one target phenotypes) indicates key gene files for a particular phenotype
# The number of subarrays must be the same with the number of POPULATION
GENE_INFO = [['./Data/TeoNAM/Genes_leaf', './Data/TeoNAM/Genes_SAM']]*len(POPULATION)

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
                 'scale':1000000}

# adjust the end location of each marker for visualisation
end_adjust = 100000

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
metrics, predicted_result_train, predicted_result_test, effect, interactions = \
    GP(DATA_NAME, MODEL, PHENOTYPE, POPULATION, RATIO, ITER_NUM, HPARAMETERS, TOTAL_PHENOTYPE, R_PATH)

# Generate circos plots
circos_plot(effect, SCOPE, interactions, MARKER_INFO, CHROMOSOME_INFO, GENE_INFO, PHENOTYPE, MODEL, CIRCOS_CONFIG, end_adjust, CYTOBAND_COLORMAP, POPULATION) 
