## Description
This folder contains two folders containing example files to implement both genomic prediction and circos plot generation functions.
   - TeoNAM folder contains adjusted example data from the TeoNAM dataset (Chen et al.,2019), accompanied by key genomic marker information for flowering (Dong et al., 2012)
   - Arabidopsis folder contains adjusted example data from the 1001 Genomes Consortium (2016), Grimm et al. (2017) and Gibbs et al. (2025), accompanied by key genomic marker information for flowering and branching (Arabidopsis Information Resource (TAIR); https://www.arabidopsis.org/)

## Data structure
1. Genotype & phenotype data
  - TeoNAM folder: "TeoNAM_dataset.zip", "W22TIL01(_subset).csv", "W22TIL03.csv", "W22TIL11.csv", "W22TIL14.csv" and "W22TIL25.csv"
  - Arabidopsis folder: "Arabidopsis_dataset.csv"
  - Columns:
       - ID: identification code for each individual
       - Population: the name of the affiliated population
       - Genomic markers: SNP information in numerical format (0,1 or 2)
       - Phenotype: recorded phenotypes of each individual
 - Rows: records of each individual
   
2. Chromosome length data
   - TeoNAM folder: "chrom.bed", "chrom_W22TIL01.bed", "chrom_W22TIL03.bed", "chrom_W22TIL11.bed", "chrom_W22TIL14.bed" and "chromW22TIL25.bed"
   - Arabidopsis folder: "Arabidopsis_chrom.bed"
   - Columns:
      - Chromosome: chromosome number ("chr"+NUMBER)
      - Start: the beginning location of the chromosome (the value should be 0 for the standard use)
      - End: the end location of the chromosome
   - Rows:
      - Each chromosome
        
3. Marker information data
   - TeoNAM folder: "marker_info.csv"
   - Arabidopsis folder: "Arabidopsis_marker_info.zip"
   - Columns:
       - chromosome: belonging chromosome number (NUMBER)
       - Start: the beginning location of the marker
       - Middle: the midpoint location of the marker
       - End: the end location of the marker
   - Rows: markers

 4. Key gene marker data
   - TeoNAM folder: "QTL.tsv","QTL_W22TIL01.tsv","QTL_W22TIL03.tsv","QTL_W22TIL11.tsv","QTL_W22TIL14.tsv","QTL_W22TIL25.tsv", "Genes_leaf.tsv", "Genes_leaf_W22TIL01.tsv","Genes_leaf_W22TIL03.tsv","Genes_leaf_W22TIL11.tsv","Genes_leaf_W22TIL14.tsv","Genes_leaf_W22TIL25.tsv", "Genes_SAM.tsv", "Genes_SAM_W22TIL01.tsv","Genes_SAM_W22TIL03.tsv","Genes_SAM_W22TIL11.tsv","Genes_SAM_W22TIL14.tsv" and "Genes_SAM_W22TIL25.tsv"
   - Arabidopsis folder: "Arabidopsis_branching.tsv" and "Arabidopsis_flowering.tsv"
   - Columns:
     - chromosome: belonging chromosome number ("chr"+NUMBER)
     - Start: the beginning location of the marker
     - End: the end location of the marker
     - Name: name of the gene
     - Colour: the colour of the gene on a circos plot
 - Rows: genes
