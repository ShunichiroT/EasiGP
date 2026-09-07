## Description

This folder contains three example datasets you can use to try both the genomic prediction and circos plot generation functions of EasiGP before bringing in your own data.

| Folder | Source | Key gene regions included |
|---|---|---|
| `TeoNAM/` | Adjusted example data from the TeoNAM dataset (Chen et al., 2019) | Flowering time (Dong et al., 2012; Wisser et al., 2019) |
| `Arabidopsis/` | Adjusted example data from the 1001 Genomes Consortium (2016), Grimm et al. (2017) and Gibbs et al. (2025) | Flowering and branching (Arabidopsis Information Resource (TAIR); https://www.arabidopsis.org/) |
| `MaizeNAM/` | Adjusted example data from the MaizeNAM dataset (Buckler et al., 2009) | Flowering time (Dong et al., 2012; Wisser et al., 2019) |

Each folder is a complete, working set of the files described below, so you can point the GUI directly at one of them to see the whole pipeline run end to end.

---

## Data structure

EasiGP reads several small, plain files rather than one combined dataset. The table below is a quick reference; the sections after it explain each file in detail with an example. "Required" means the pipeline cannot run without it; "optional" means it's only needed for a specific add-on feature.

| # | File | Required? | Used for |
|---|---|---|---|
| 1a | Genotype CSV | Required (unless using 1b) | The marker data every model trains and predicts on |
| 1b | Genotype PLINK1 fileset (`.bed`/`.bim`/`.fam`) | Required (unless using 1a) | Alternative to 1a for large marker sets |
| 2 | Phenotype CSV | Required | The trait(s) being predicted |
| 3 | Marker information (`marker_info.csv`) | Required for circos plots | Places each marker on the genome |
| 4 | Chromosome information (`chrom.csv`) | Required for circos plots | Chromosome lengths, for drawing the plot's outer ring |
| 5 | Key gene region information (`gene_info.csv`) | Optional | Known reference gene regions to compare against inferred marker effects |
| 6a | Gene-interaction network (`network.json`) | Optional | The GAT biological prior-knowledge model's gene-gene network |
| 6b | Gene location lookup table | Optional | Genomic coordinates for the genes named in 6a, if not already in `gene_info.csv` |

Two structural rules apply across every tabular file below:

- **The first two columns of the genotype and phenotype files are matched by position, not by name.** Whatever you actually call them, EasiGP treats column 1 as `ID` and column 2 as `population` internally. Every column *after* that - marker names, trait names, gene names - is matched **by name** and is never renamed, since those names are used as join keys elsewhere in the pipeline. In short: get columns 1-2 in the right order; get every other column's *name* right.
- **Markers are always numerically coded `0`, `1`, or `2`** (the count of one allele at that locus), never letters or genotype strings.

### 1a. Genotype data (CSV)

| Column | Meaning |
|---|---|
| 1 (`ID`) | Identification code for each individual |
| 2 (`population`) | Name of the individual's population |
| 3 onward | One column per genomic marker, named however you like - these names are used as-is throughout the pipeline and in every output file |

Each row is one individual. Marker values are `0`/`1`/`2` as above.

```
ID,     population, SNP_001, SNP_002, SNP_003, ...
Ind_01, PopA,        0,       1,       2, ...
Ind_02, PopA,        1,       1,       0, ...
Ind_03, PopB,        2,       0,       1, ...
```

### 1b. Genotype data (PLINK1 binary fileset) - alternative to 1a

Instead of a CSV, you can point EasiGP at a standard PLINK1 binary fileset - three files sharing one name (the "stem"):

```
<stem>.bed   the marker calls themselves (binary)
<stem>.bim   one row per marker: chromosome, marker name, position, alleles
<stem>.fam   one row per individual (family/within-family ID, etc.)
```

Use this when you have a large marker set already in PLINK format, or want EasiGP's built-in LD pruning to run PLINK2 natively rather than converting first. A few things behave differently from the CSV path and are worth knowing:

- `ID` and `population` still always come from your **phenotype CSV** (file 2), never invented from the `.fam` file - the `.fam` file's individual-ID column is used only as the key to match individuals against the phenotype file.
- You'll still need marker information (file 3) for circos plotting; EasiGP does not read marker positions out of the `.bim` file for that purpose (only for the PLINK-side pruning/extraction steps).
- This is the more efficient option for a large marker set when you're only running the GAT biological prior-knowledge model, since EasiGP will extract just the markers each gene needs rather than loading everything into memory.

### 2. Phenotype data (CSV)

| Column | Meaning |
|---|---|
| 1 (`ID`) | Identification code for each individual - matched against the genotype file's `ID` column |
| 2 (`population`) | Name of the individual's population |
| 3 onward | One column per trait, named however you like - select which trait(s) to predict by name in the GUI |

Each row is one individual.

```
ID,     population, flowering_time, plant_height
Ind_01, PopA,       62.5,           178.2
Ind_02, PopA,       58.0,           165.4
Ind_03, PopB,       70.1,           190.0
```

### 3. Marker information (`marker_info.csv`)

Tells EasiGP where each marker sits on the genome, so marker effects can be placed on the circos plot. One row per marker; row order doesn't matter.

| Column | Meaning |
|---|---|
| `chromosome` | Chromosome number |
| `name` | Marker name - **must match a genotype column name exactly** |
| `start` | Start position of the marker |
| `end` | End position of the marker |

```
chromosome, name,    start,   end
1,          SNP_001, 102345,  102345
1,          SNP_002, 210987,  210987
2,          SNP_003, 45678,   45678
```

### 4. Chromosome information (`chrom.csv`)

Defines the length of each chromosome, used to draw the outer ring of the circos plot. One row per chromosome.

| Column | Meaning |
|---|---|
| `chromosome` | Chromosome number |
| `start` | Beginning of the chromosome - use `0` for standard use |
| `end` | End of the chromosome (its total length) |
| `population` | *(Optional)* Which population this row applies to - see note below |

```
chromosome, start, end,       population
1,          0,     301354135, all
2,          0,     237068873, all
```

**About the `population` column:** only include it if different populations in your data have chromosomes of different lengths (e.g. different reference genome builds per population). Write `all` for a row that applies across every population. If every population shares the same chromosome lengths, you can leave the `population` column out entirely - EasiGP will automatically apply the same rows to every population plus `all`.

### 5. Key gene region information (`gene_info.csv`) - optional

Known genes/regions from the literature that you want drawn on the circos plot alongside the models' *inferred* marker effects, for visual comparison. One row per gene.

| Column | Meaning |
|---|---|
| `chromosome` | Chromosome number |
| `start` | Beginning position of the gene |
| `end` | End position of the gene |
| `name` | Gene name |
| `colour` | Colour name from the palette offered in the Circos plot tab of the GUI |
| `source` | Where this gene region came from (e.g. a citation) - shown as the ring section's own label on the plot |
| `phenotype` | Which trait this gene is relevant to - must match a phenotype column name from file 2 |
| `population` | *(Optional)* Which population this row applies to - same rule as `chrom.csv` above: write `all`, or omit the column entirely if every population shares the same gene regions |

```
chromosome, start,  end,    name, colour, source,          phenotype,       population
1,          100000, 102000, FT1,  red,    Dong et al. 2012, flowering_time, all
```

### 6. Biological prior network files - optional, for the GAT biological prior-knowledge model

These two files together tell the GAT biological prior-knowledge model which *genes* interact with which, instead of asking it to learn marker-level structure from scratch. You can supply both by hand, or generate them automatically from the GUI's "Biological prior network" preprocessing step (the automatic route needs Claude access - see the main README's setup instructions).

#### 6a. Gene-interaction network (`network.json`)

A JSON file with `nodes` and `edges` sections, in either the plain schema below or [FLASH-P](https://flash-p.com/)'s more compact equivalent (`"ty":"G"` in place of `"type":"GENE"`, `"s"`/`"t"` in place of `"source"`/`"target"`, etc. - EasiGP reads both automatically):

```json
{
  "nodes": [
    {"id": "FT1", "type": "GENE"},
    {"id": "FT2", "type": "GENE"}
  ],
  "edges": [
    {"source": "FT1", "target": "FT2", "sign": 1, "mechanism": "upstream regulator"}
  ]
}
```

Only nodes typed `GENE` (or `"ty":"G"`) are used as genes - any other node type present in the file (e.g. a pathway or compound) is reported and ignored, never guessed at. If you're hand-curating this file, double check every `GENE`-typed node really is a single gene and not a pathway or gene family name - this is the single most common source of a gene silently missing from your network.

#### 6b. Gene location lookup table

Gives the genomic coordinates for the gene names used in `network.json`, if they aren't already covered by `gene_info.csv` (file 5). One row per gene (a multi-locus gene may have more than one row).

| Column | Meaning |
|---|---|
| `Gene_Name` | Must match a node `id` in `network.json` |
| `Chromosome` | Chromosome number **(note the capital "C" here - this file uses different capitalisation from `chrom.csv`/`gene_info.csv`/`marker_info.csv` above, so don't copy-paste a header between them)** |
| `Start_bp` and `End_bp`, *or* `Start_cM` and `End_cM` | Gene boundaries, in base pairs or centimorgans - pick one unit consistently for the whole file |
| `AGI_Locus_ID` | *(Optional)* Systematic locus identifier, for provenance/cross-checking only |
| `Source` | *(Optional)* Where this coordinate came from, for provenance/cross-checking only |

```
Gene_Name, Chromosome, Start_bp, End_bp, AGI_Locus_ID, Source
FT1,       1,          100000,   101500, AT1G00000,    TAIR
FT2,       1,          150200,   151800, AT1G00010,    TAIR
```

A gene whose coordinates can't be resolved trustworthily is reported and dropped by EasiGP, never guessed at or left at a default value - this applies whether the table above was hand-built or generated automatically.

---

## How the files fit together

```
Genotype  ──┐  joined on ID + population
Phenotype ──┘

Genotype marker columns ── matched by NAME ── marker_info.csv ("name")
                                                    │
                                     placed on chromosomes defined in
                                     chrom.csv, alongside optional
                                     reference regions from gene_info.csv

Only for the GAT biological prior-knowledge model:
  network.json (which genes interact)
        │  gene names matched against
        ▼
  gene_info.csv OR the gene location lookup table (6b) (where each gene is)
        │  gene's start/end window overlapped against
        ▼
  marker_info.csv / PLINK .bim (where each marker is)
        │
        ▼
  one feature vector per gene, built from every marker inside its window
```

In short: genotype and phenotype are joined on the individual; markers are placed on the genome using `marker_info.csv`; `chrom.csv` (and optionally `gene_info.csv`) control what the circos plot draws; and the two biological-prior files (6a/6b) are only needed if you're using the GAT biological prior-knowledge model, in which case they connect back to the same marker positions via each gene's genomic window.

---

## References

Buckler ES, Holland JB, Bradbury PJ, Acharya CB, Brown PJ, Browne C, Ersoz E, Flint-Garcia S, Garcia A, Glaubitz JC et al. 2009. The genetic architecture of maize flowering time. Science. 325:714–718.

Chen Q, Yang CJ, York AM, Xue W, Daskalska LL, DeValk CA, Krueger KW, Lawton SB, Spiegelberg BG, Schnell JM et al. 2019. Teonam: A nested association mapping population for domestication and agronomic trait analysis in maize. Genetics. 213:1065–1078.

Dong Z, Danilevskaya O, Abadie T, Messina C, Coles N, Cooper M. 2012. A gene regulatory network model for floral transition of the shoot apex in maize and its dynamic modelling. PLoS ONE.

Gibbs, Patrick M., Jefferson F. Paril, and Alexandre Fournier-Level. 2025. Trait genetic architecture and population structure determine model selection for genomic prediction in natural Arabidopsis thaliana populations. Genetics 229.3: iyaf003.

Grimm DG, Roqueiro D, Salomé PA, Kleeberger S, Greshake B, Zhu W, Liu C, Lippert C, Stegle O, Schölkopf B, Weigel D, Borgwardt KM. 2017. easyGWAS: A Cloud-Based Platform for Comparing the Results of Genome-Wide Association Studies. The Plant Cell. 29. 5-19.

The 1001 Genomes Consortium. 2016. 1,135 Genomes Reveal the Global Pattern of Polymorphism in Arabidopsis thaliana. Cell. 166(2). 481-491.

Wisser RJ, Fang Z, Holland JB, Teixeira JE, Dougherty J, Weldekidan T, de Leon N, Flint-Garcia S, Lauter N, Murray SC et al. 2019. The genomic basis for short-term evolution of environmental adaptation in maize. Genetics. 213:1479–1494.
