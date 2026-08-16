from pycirclize import Circos
from pycirclize.parser import Bed
from pycirclize import config as _pycirclize_config
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.legend_handler as mlegend_handler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
import tempfile
import uuid
import math

from pipeline_utils import unify_columns_by_position


def _visible_edge_color(fill_color, min_contrast=90.0, min_factor=0.15):
    """The border colour to use for a given fill colour.

    Requirement (does the border colour match the marker colour?):
    returns `fill_color` COMPLETELY UNCHANGED whenever it's already
    visible enough against a white plot background on its own - which,
    for EasiGP's own real marker-effect/gene-source palette
    (DEFAULT_CYTOBAND_COLORMAP, main_app.py), is the MAJORITY of colours
    (every mid-to-high quantile level, e.g. blue4 upward, red3 upward,
    and any other reasonably saturated custom colour someone adds).
    Only darkens - and only by the SMALLEST amount that actually fixes
    it - for colours too close to white to be visible as a border at all
    on their own (EasiGP's own deliberately pale LOWEST quantile levels,
    e.g. blue0 = #def2ff, red0 = #fce1a7, used to visually de-emphasise
    low-effect markers - see this file's data_conversion()/
    quantile_conversion() - plus plain white itself). Preserves hue in
    both cases: an untouched colour is obviously identical, and a
    darkened one keeps every RGB channel's relative ratio (scaling all
    three by the same factor never shifts which channel dominates), so
    even an adjusted border stays recognisably in the same colour family
    as its own fill - still bluish for a blue-family marker, still
    reddish for a red-family one - never replaced by an unrelated fixed
    colour like plain black, which would lose that information entirely.

    Uses matplotlib's own colour parser (matplotlib.colors.to_rgb),
    which accepts every colour format matplotlib itself understands -
    '#rrggbb', '#rgb', named CSS colours ('steelblue', 'orange', ...),
    etc. - so a NEWLY ADDED custom colour in ANY of those formats is
    handled exactly the same way, not just the specific '#rrggbb' shape
    EasiGP's own built-in palette happens to use. A colour matplotlib
    can't parse at all is returned unchanged (the same colour a caller
    would already have seen fail when trying to use it as a FILL, so
    this never needs to raise its own separate error over it).

    Parameters
    ----------
    min_contrast : the minimum acceptable distance (0-255 scale) between
        a colour's perceptual luminance and white - anything at or above
        this is left completely untouched; anything below is darkened by
        just enough to reach it (never further).
    min_factor : a floor on how much darkening is ever applied (0.15 =
        never darker than 15% of the original brightness), so a colour
        that's ALREADY very close to pure black (luminance near 0, which
        would otherwise need an enormous - or undefined - darkening
        factor to reach min_contrast) doesn't get pushed to solid black
        and lose its own hue entirely.
    """
    try:
        r, g, b = mcolors.to_rgb(fill_color)
    except ValueError:
        return fill_color

    luminance = 255.0 * (0.299 * r + 0.587 * g + 0.114 * b)
    contrast = 255.0 - luminance
    if contrast >= min_contrast or luminance <= 0:
        return fill_color

    factor = max(min_factor, min(1.0, (255.0 - min_contrast) / luminance))
    return mcolors.to_hex((r * factor, g * factor, b * factor))


def _add_cytoband_tracks_with_border(circos, r_lim, cytoband_file, track_name, cytoband_cmap,
                                      edge_width=0.4):
    """A drop-in replacement for Circos.add_cytoband_tracks() (pycirclize's
    own built-in method, used elsewhere in this file for the gene/model
    tracks) that ALSO draws a thin, visible border around every region -
    pycirclize's own version calls track.rect(start, end, fc=color) with
    no way to pass through an edge colour/width at all, filling the
    region only.

    Requirement (marker/gene regions too small to actually see on the
    plot): a filled region's rendered width scales with its DATA-SPACE
    size, so an unusually short marker/gene, or one on an unusually long
    chromosome, can end up thinner than a single pixel and simply not
    render at all - this is exactly what the GENE_ADJUST/END_ADJUST
    widening (see this file's data_conversion()) and the WINDOW-size
    suggestion (main_app.py) already fix at the DATA level, by making the
    region itself bigger before it's ever drawn. A border LINE WIDTH,
    however, is a fixed on-screen thickness (matplotlib's `lw`, in
    points) that does NOT scale down with the region's data-space size -
    so even a region that still ends up sub-pixel-thin after widening
    (e.g. an extreme genome scale, or a value smaller than intended) gets
    outlined by a border that stays visible regardless, as a second,
    complementary safety net on top of the data-level fix rather than a
    replacement for it.

    The border colour is DERIVED from each region's own intended fill
    colour via _visible_edge_color() (an EXACT match whenever the fill is
    already visible enough on its own - the majority of colours in
    practice - and only ever the smallest necessary darkening otherwise,
    never a single fixed colour for everything) - for a normal-sized
    (visible) region this usually means the border and fill are
    literally the same colour; for a sub-pixel-width one, this border IS
    the only thing that ends up visible at all, so it needs to still
    convey which quantile level / gene source that region actually
    represents, not a uniform, information-losing colour like plain
    black would.

    Mirrors pycirclize's own add_cytoband_tracks() implementation
    exactly otherwise (same per-sector track creation, same axis() call,
    same cytoband_cmap colour lookup) - see that method's source (in the
    installed pycirclize.circos module) for the reference this is kept
    in sync with. Tooltips (pycirclize's own hover-text feature, only
    relevant for interactive/HTML output) are deliberately omitted here,
    since this codebase only ever saves static PNGs.
    """
    if cytoband_cmap is None:
        cytoband_cmap = _pycirclize_config.CYTOBAND_COLORMAP
    cytoband_records = Bed(cytoband_file).records
    for sector in circos.sectors:
        track = sector.add_track(r_lim, name=track_name)
        track.axis()
        for rec in cytoband_records:
            if sector.name == rec.chr:
                color = cytoband_cmap.get(str(rec.score), 'white')
                # Requirement 3 (bugfix - ValueError: x=... is invalid
                # range of '...' sector): the widened start/end written by
                # quantile_conversion()/data_conversion() are ALREADY meant
                # to be clamped to their own chromosome's bounds at write
                # time (see those functions' own per-chromosome clamping
                # loops) - but that clamp depends on a SEPARATE lookup
                # against a chrom_info file read independently of what
                # circos itself was actually initialized with, and can
                # silently skip a chromosome entirely (falls through via
                # `continue`) if that lookup ever fails to find a matching
                # row (e.g. a chromosome present in marker/gene info but
                # missing - or named slightly differently - in the
                # chrom_info file). Clamping AGAIN here, directly against
                # THIS sector's own authoritative bounds (exactly what
                # circos was actually initialized with, so this can never
                # itself be wrong) guarantees pycirclize is never handed an
                # out-of-range coordinate, regardless of whether the
                # earlier, separate clamp ran into that edge case - a
                # defensive second layer, not a replacement for fixing the
                # data itself to be correctly widened in the first place.
                _start = max(sector.start, min(rec.start, sector.end))
                _end = max(sector.start, min(rec.end, sector.end))
                track.rect(_start, _end, fc=color, ec=_visible_edge_color(color), lw=edge_width)


def _load_combined_marker_info(marker_info, RESULT_NAME, PHENOTYPE):
    """Load marker_info.csv (SNP-level coordinates: chromosome, name, start,
    end) and transparently merge in any per-model GENE-level coordinate
    table(s) for this phenotype, so a single 'name' lookup resolves both SNP
    names (the usual case) and gene names (e.g.
    models/GAT_biological_prior_knowledge.py's own attention output, whose
    marker1/marker2 values are gene names, not SNP names, since its graph
    nodes are genes - see that file's RESULT_NAME/PHENOTYPE_NAME docstring
    entry for the side file this picks up:
    './Result/<RESULT_NAME>/<model>_gene_coordinates_<PHENOTYPE>.csv').

    Matched by filename pattern (`*_gene_coordinates_<PHENOTYPE>.csv`) so
    this stays usable for any future gene-node-based model that writes the
    same shape of side file, without circos_plot.py needing to hardcode a
    specific model's name. If no such file exists for this phenotype (the
    common case - every non-gene-level model), this is exactly equivalent
    to the plain `pd.read_csv(marker_info)` it replaces: zero behaviour
    change for SNP-only results.
    """
    marker = pd.read_csv(marker_info)
    # Requirement 8: unify by position (chromosome, name, start, end) -
    # 'name' here is the MARKER's identifying column HEADER (a fixed,
    # structural part of this file's schema), not the marker names
    # themselves (the VALUES in that column, which are of course left
    # completely untouched) - so this is safe to rename regardless of
    # what header text the file actually uses.
    marker = unify_columns_by_position(marker, ['chromosome', 'name', 'start', 'end'], 'marker info file')

    pattern = os.path.join('.', 'Result', RESULT_NAME, f'*_gene_coordinates_{PHENOTYPE}.csv')
    gene_coord_files = glob.glob(pattern)
    if not gene_coord_files:
        return marker

    gene_tables = [pd.read_csv(f) for f in gene_coord_files]
    combined = pd.concat([marker] + gene_tables, ignore_index=True)
    # A name should resolve to exactly one location; if a gene coordinate
    # table and marker_info somehow both define the same name, keep
    # marker_info's own entry (read first, so kept by keep='first').
    combined = combined.drop_duplicates(subset=['name'], keep='first')
    return combined


def _circos_intermediate_dir(RESULT_NAME):
    """Requirement 2: the chrom_*.bed / gene_info_*.tsv / marker_effect_*.tsv
    files this module writes and reads back are purely intermediate
    working files for pycirclize's own Bed-file-based track API - never
    meant to be inspected directly the way the REAL result files (
    Metric.csv, Marker_effect.csv, the final circos_*.png images
    themselves, etc.) are. Keeping them in this subfolder (created here,
    on demand) instead of dumped directly into
    './Result/<RESULT_NAME>/' alongside the real output keeps that
    top-level result folder readable - the final circos_*.png plots
    still save directly there, unaffected; only these intermediate
    per-model/per-population working files move."""
    path = os.path.join('.', 'Result', RESULT_NAME, 'circos_intermediate')
    os.makedirs(path, exist_ok=True)
    return path


def data_conversion(chrom_info, gene_info, PHENOTYPE, RESULT_NAME, gene_adjust=0):
    chromosome = pd.read_csv(chrom_info)
    # Requirement 8: unify by position - every column in this file is a
    # fixed structural field (chromosome ID, start, end, population),
    # never a user-chosen identifier used as a lookup key elsewhere, so
    # positional unification is always safe here.
    chromosome = unify_columns_by_position(
        chromosome, ['chromosome', 'start', 'end', 'population'], 'chromosome info file'
    )
    # Requirement (bugfix): explicit str cast, not left to pandas' own
    # CSV type inference - a 'population' column mixing numeric
    # populations with the literal string 'all' (which circos_plot()
    # itself always adds - see its own POPULATION = ('all',) +
    # tuple(POPULATION)) usually gets inferred as all-string by pandas
    # anyway, but that inference is CONTEXT-DEPENDENT on the exact file
    # contents (whether 'all' rows happen to be present, their exact
    # position, etc.) - relying on it left chromosome_population's
    # actual dtype (str vs int) genuinely unpredictable from one file to
    # the next, and the SAME risk applied separately to gene_population
    # below, for the SAME reason. Two independently-unpredictable dtypes
    # being compared/indexed against each other (see the fix a few lines
    # down) is exactly the kind of thing that can silently work in a
    # simple case and silently break in a different one.
    chromosome['population'] = chromosome['population'].astype(str)
    chromosome_population = pd.unique(chromosome['population'])
    
    chromosome['start'] =[int(round(chromosome.loc[k, 'start'])) for k in range(chromosome.shape[0])]
    chromosome['end'] =[int(round(chromosome.loc[k, 'end'])) for k in range(chromosome.shape[0])]
    chromosome['chromosome'] = chromosome['chromosome'].astype(str)
    
    for i in range(len(chromosome_population)):
        chromosome_selected = chromosome[chromosome['population']==chromosome_population[i]]
        chromosome_selected = chromosome_selected.drop(['population'],axis=1)
        chromosome_selected.to_csv(_circos_intermediate_dir(RESULT_NAME)+'/chrom_'+str(chromosome_population[i])+'.bed', sep='\t', index=False)
    
    if gene_info is not None:
        gene = pd.read_csv(gene_info)
        # Requirement 8: same reasoning as marker_info.csv above - 'name'
        # here is this file's gene-identifying column HEADER, a fixed
        # structural field, not the gene names themselves (which stay
        # untouched as VALUES).
        gene = unify_columns_by_position(
            gene, ['chromosome', 'start', 'end', 'name', 'colour', 'source', 'phenotype', 'population'],
            'gene info file'
        )
        # Requirement (bugfix): same explicit str cast as chromosome
        # above, and for the same reason - see that comment.
        gene['population'] = gene['population'].astype(str)

        # Requirement: gene regions are often too small to actually see
        # on the plot (a modest-sized gene against a whole chromosome's
        # rendered arc) - gene_adjust (in the same units as the
        # chromosome/gene info files, e.g. bp) widens each one the same
        # way END_ADJUST already widens marker regions: subtracted from
        # start, added to end, then clamped to stay within that gene's
        # OWN chromosome (never a hardcoded 0, and never another
        # chromosome's bounds) - see the per-chromosome loop just below.
        gene['chromosome'] = gene['chromosome'].astype(str)
        # Requirement (diagnostic): check the RAW, un-widened positions
        # against the chromosome file BEFORE gene_adjust is applied below -
        # see _warn_if_raw_position_exceeds_chromosome()'s own docstring.
        _warn_if_raw_position_exceeds_chromosome(gene, chromosome, 'gene info file')
        # Requirement (bugfix - GENE_ADJUST could overflow into a wildly
        # wrong number for a large coordinate): int(...) used to be
        # applied AFTER the subtraction/addition below, not before -
        # gene.loc[k, 'start'] is a numpy scalar (whatever dtype the
        # 'start' column happens to be, e.g. int32 in some pandas/numpy
        # version or code path), and numpy arithmetic between a
        # fixed-width integer type and a plain Python int STAYS in that
        # same fixed-width type rather than safely upcasting (confirmed
        # directly: np.int32(2_147_483_600) + 100 silently wraps around
        # to a large NEGATIVE number, with only a RuntimeWarning, not an
        # exception) - genomic coordinates for a large genome can
        # realistically approach or exceed that range. Converting each
        # coordinate to a plain Python int FIRST, before doing the
        # widening arithmetic, guarantees the arithmetic itself happens
        # in Python's own arbitrary-precision integers - which cannot
        # overflow at all, regardless of how large the numbers are -
        # rather than in whatever fixed-width numpy type the column
        # happened to be, whether that's ever actually int32 in practice
        # or not. gene_adjust itself is also explicitly rounded to a
        # plain int first, for the same reason (a non-integer widening
        # amount doesn't correspond to a real base-pair distance anyway).
        _gene_adjust_int = int(round(gene_adjust))
        gene['start'] = [int(gene.loc[k, 'start']) - _gene_adjust_int for k in range(gene.shape[0])]
        gene['end'] = [int(gene.loc[k, 'end']) + _gene_adjust_int for k in range(gene.shape[0])]

        chromosome_total = pd.unique(gene['chromosome'])
        _unmatched_chromosomes = [c for c in chromosome_total if chromosome.loc[chromosome['chromosome'] == c].shape[0] == 0]
        if _unmatched_chromosomes:
            # Requirement (diagnostic - a real, confirmed cause of gene
            # regions being invisible, found by checking a real user's
            # actual chrom_info/gene_info files directly): if a gene's
            # chromosome name never matches ANY row in the chromosome info
            # file, its widened start/end below never get clamped at all
            # (the loop simply skips it), and - more fundamentally -
            # plot()'s own rendering match (`sector.name == rec.chr`)
            # will never find that chromosome either, so the gene never
            # gets drawn at all, however large GENE_ADJUST is. A common,
            # easy-to-make cause: the two files using different naming
            # conventions for the same chromosomes (e.g. chrom_info using
            # 'A10' while gene_info uses '10A' for the exact same
            # chromosome - confirmed to happen in practice, not a
            # hypothetical). Printed as a clear, specific warning rather
            # than left to be silently invisible - this is exactly the
            # kind of problem that looks identical to 'GENE_ADJUST is too
            # small' from the rendered plot alone, but no amount of
            # widening fixes it.
            _all_chrom_names = sorted(pd.unique(chromosome['chromosome']).tolist())
            print(
                f"[circos] WARNING: {len(_unmatched_chromosomes)} chromosome name(s) in the gene info "
                f"file have NO match at all in the chromosome info file, so genes on them can never "
                f"be drawn (this is a common cause of a gene ring looking empty, regardless of "
                f"GENE_ADJUST): {_unmatched_chromosomes[:10]}{'...' if len(_unmatched_chromosomes) > 10 else ''}\n"
                f"[circos]   Chromosome names in the chromosome info file: "
                f"{_all_chrom_names[:10]}{'...' if len(_all_chrom_names) > 10 else ''}\n"
                f"[circos]   Check whether the two files are using different naming conventions for "
                f"the same chromosomes (e.g. 'A10' vs '10A')."
            )
        for k in range(len(chromosome_total)):
            _chrom_mask = gene['chromosome'] == chromosome_total[k]
            _chrom_row = chromosome.loc[chromosome['chromosome'] == chromosome_total[k]]
            if _chrom_row.shape[0] == 0:
                continue
            _chrom_start = int(_chrom_row['start'].values[0])
            _chrom_end = int(_chrom_row['end'].values[0])
            _clamp_region_to_chromosome(gene, _chrom_mask, _chrom_start, _chrom_end)

        gene_population = pd.unique(gene['population'])
        gene_source = pd.unique(gene['source'])
        print(f"[circos] data_conversion(): gene file's own distinct population values "
              f"(what gene rings will actually be written for): {list(gene_population)}")

        for i in range(len(PHENOTYPE)):
            for j in range(len(gene_population)):
                for k in range(len(gene_source)):
                    gene_selected = gene[(gene['population']==gene_population[j]) & (gene['phenotype']==PHENOTYPE[i]) & (gene['source']==gene_source[k])]
                    gene_selected = gene_selected.drop(['source','population','phenotype'],axis=1)
                    if gene_selected.shape[0] != 0:
                        # Requirement (bugfix): this MUST be
                        # gene_population[j] (the loop's own index
                        # variable, i.e. the population this specific
                        # gene_selected block actually belongs to) - it
                        # was chromosome_population[j] before, a
                        # DIFFERENT array from an entirely different
                        # file, coincidentally the same length/order in
                        # simple cases (masking the bug) but never
                        # guaranteed to be, and genuinely NOT
                        # guaranteed to be once chromosome_population's
                        # dtype could differ unpredictably from
                        # gene_population's (see the two explicit str
                        # casts added above) - a mismatch here silently
                        # writes a gene track under the WRONG
                        # population's filename, which plot() would
                        # then never find when it goes looking for the
                        # CORRECT population's own gene file, exactly
                        # matching a 'gene ring silently missing for
                        # some population' symptom.
                        gene_selected.to_csv(_circos_intermediate_dir(RESULT_NAME)+'/gene_info_'+str(PHENOTYPE[i])+'_'+str(gene_source[k])+'_'+str(gene_population[j])+'.tsv', sep='\t', index=False)
        
        pop_source = gene.loc[:,['phenotype','population','source']].drop_duplicates()
    
    else:
        pop_source = None
        
    return pop_source

def quantile_conversion(effect, marker_info, chrom_info, PHENOTYPE, MODEL, end_adjust, POPULATION, WINDOW, RESULT_NAME, ASCENDING):
    
    chromosome = pd.read_csv(_circos_intermediate_dir(RESULT_NAME)+'/chrom_'+str(POPULATION)+'.bed', delimiter='\t')
    chromosome['chromosome'] = chromosome['chromosome'].astype(str)
    
    if WINDOW != 0:
        division = []
        cnt = 0
        for n in range(int(chromosome['end'].max())):
            division += [WINDOW*cnt]
            cnt += 1
            if WINDOW*cnt > int(chromosome['end'].max()):
                break
    # Convert genomic marker effects into ten level quantiles
    effect.iloc[:,5:] = effect.iloc[:,5:].abs().astype(float)
    effect = effect.drop('ratio', axis=1)
    
    if POPULATION == 'all':
        effect_grouped = effect.iloc[:,1:].groupby(['phenotype','model']).mean()
        effect_grouped = effect_grouped.reset_index(drop=False)
    else:
        effect_grouped = effect.groupby(['population','phenotype','model']).mean()
        effect_grouped = effect_grouped.reset_index(drop=False)
        effect_grouped['population'] = effect_grouped['population'].astype(str)
    
    REMOVE = []
    
    for iii in range(len(MODEL)):
        colour = 'red' if MODEL[iii] in ['ensemble', 'Linear transformation', 'Nelder Mead', 'Bayesian optimisation'] else 'blue'
        if POPULATION == 'all':
            effect_selected = effect_grouped[(effect_grouped['model']==MODEL[iii]) & (effect_grouped['phenotype']==PHENOTYPE)].iloc[:,3:].T
        else:
            # Requirement 7 (bugfix, found while fixing the all-zero-
            # effect crash): iloc[:,3:] here only skips 3 columns
            # (assuming population/phenotype/model precede the marker
            # columns), but this branch's own groupby(['population',
            # 'phenotype','model']).mean().reset_index() actually
            # produces population, phenotype, model, sample, <markers...>
            # - the replicate number ('sample') sits between 'model' and
            # the first real marker column, so iloc[:,3:] was silently
            # including it as if it were itself a marker's effect value
            # (only ever discarded much later, once merged against
            # marker_info.csv - by which point it had already
            # contaminated the quantile-threshold computation every
            # marker's colour level is chosen from). iloc[:,4:] correctly
            # skips all 4 metadata columns. The POPULATION == 'all'
            # branch above groups by phenotype/model only (population
            # was already dropped before that groupby), so its own
            # reset_index() only ever re-adds 2 columns - its iloc[:,3:]
            # already lands on the first real marker column correctly,
            # and is intentionally left alone.
            effect_selected = effect_grouped[(effect_grouped['model']==MODEL[iii]) & (effect_grouped['phenotype']==PHENOTYPE) & (effect_grouped['population']==str(POPULATION))].iloc[:,4:].T
           
        if effect_selected.shape[1] != 0:
            # Requirement 7 (bugfix): if EVERY marker's effect for this
            # model is exactly 0, the quantile-threshold approach below
            # breaks down in two different ways depending on WINDOW:
            #   - WINDOW == 0: every np.quantile(...) breakpoint also
            #     collapses to 0, so every '>= threshold' comparison
            #     matches EVERY marker at EVERY level in turn, and the
            #     LAST assignment (colour9 - the highest/darkest level)
            #     silently overwrites all the earlier ones - exactly
            #     backwards from what an all-zero effect means.
            #   - WINDOW != 0: rows get filtered to effect > 0 further
            #     down BEFORE any quantile is computed - with every
            #     effect at exactly 0, that filter empties the
            #     DataFrame entirely, and np.quantile() on an empty
            #     array raises an exception outright.
            # Detected once, up front, for both branches: skip the
            # quantile computation entirely and assign colour0 (the
            # lowest/faintest level) to every marker directly instead -
            # correct either way, and never touches np.quantile() on
            # data that can't support it.
            #
            # Requirement (SECOND correction - confirmed against a real
            # crash report): this used to check '== 0' specifically -
            # but the actual filter a few lines below (WINDOW != 0
            # branch) is 'effect > 0', not 'effect != 0'. Those are NOT
            # the same condition: a genomic marker's effect on a trait
            # can legitimately be NEGATIVE (decreases the trait, not
            # just 'has no effect'), and a set of effects that are all
            # zero-or-negative (no marker strictly positive, but not
            # literally every single one exactly 0 either) still passes
            # '== 0).all()' as False - so the ORIGINAL fix's own
            # detection silently failed to catch this close cousin of
            # the exact case it was written for, and the SAME empty-
            # DataFrame crash it was meant to prevent still happened.
            # Checking '<= 0' instead matches the real filter condition
            # exactly, catching every input that filter would empty out
            # - not just the narrower all-exactly-zero case.
            _all_zero_effect = bool((effect_selected.to_numpy(dtype=float) <= 0).all())

            if WINDOW == 0:
                effect_selected_copy = effect_selected.copy().astype(object)
                
                if _all_zero_effect:
                    effect_selected_copy.iloc[:,0] = colour+'0'
                else:
                    effect_selected_copy.iloc[:,0] = colour+'0'
                    tmp = list(effect_selected[effect_selected>= np.quantile(effect_selected.to_numpy().flatten(), 0.1)].dropna().index)
                    effect_selected_copy.loc[tmp,:] = colour+'1'
                    tmp = list(effect_selected[effect_selected>= np.quantile(effect_selected.to_numpy().flatten(), 0.2)].dropna().index)
                    effect_selected_copy.loc[tmp,:] = colour+'2'
                    tmp = list(effect_selected[effect_selected>= np.quantile(effect_selected.to_numpy().flatten(), 0.3)].dropna().index)
                    effect_selected_copy.loc[tmp,:] = colour+'3'
                    tmp = list(effect_selected[effect_selected>= np.quantile(effect_selected.to_numpy().flatten(), 0.4)].dropna().index)
                    effect_selected_copy.loc[tmp,:] = colour+'4'
                    tmp = list(effect_selected[effect_selected>= np.quantile(effect_selected.to_numpy().flatten(), 0.5)].dropna().index)
                    effect_selected_copy.loc[tmp,:] = colour+'5'
                    tmp = list(effect_selected[effect_selected>= np.quantile(effect_selected.to_numpy().flatten(), 0.6)].dropna().index)
                    effect_selected_copy.loc[tmp,:] = colour+'6'
                    tmp = list(effect_selected[effect_selected>= np.quantile(effect_selected.to_numpy().flatten(), 0.7)].dropna().index)
                    effect_selected_copy.loc[tmp,:] = colour+'7'
                    tmp = list(effect_selected[effect_selected>= np.quantile(effect_selected.to_numpy().flatten(), 0.8)].dropna().index)
                    effect_selected_copy.loc[tmp,:] = colour+'8'
                    tmp = list(effect_selected[effect_selected>= np.quantile(effect_selected.to_numpy().flatten(), 0.9)].dropna().index)
                    effect_selected_copy.loc[tmp,:] = colour+'9'
                
                effect_selected_copy.columns = ['colour']
                if ASCENDING is not None:
                    effect_selected_copy = effect_selected_copy.sort_values(by=['colour'], ascending=ASCENDING)
                effect_selected_copy = effect_selected_copy.reset_index(drop=False)
                
                marker = _load_combined_marker_info(marker_info, RESULT_NAME, PHENOTYPE)
                merged = pd.merge(effect_selected_copy, marker, left_on=['index'], right_on=['name'])
                merged = merged.loc[:,['chromosome','start','end','index','colour']]
                # Requirement (bugfix - END_ADJUST could overflow into a
                # wildly wrong number for a large coordinate): explicit
                # .astype('int64') BEFORE the widening arithmetic below,
                # not left at whatever dtype 'start'/'end' happened to
                # arrive in from the merge above - a narrower fixed-width
                # integer type (e.g. int32) silently wraps around to a
                # large negative/wrong number once it overflows, rather
                # than raising an error (confirmed directly:
                # np.int32(2_147_483_600) + 100 wraps to a large negative
                # number, with only a RuntimeWarning) - genomic
                # coordinates for a large genome can realistically
                # approach or exceed int32's range. int64's own range
                # (+-~9.2 quintillion) is, by contrast, far beyond any
                # real genomic coordinate (even the largest known genome
                # is under 150 billion bp in total), so this is a safe,
                # practically-unlimited ceiling for this domain without
                # needing Python's fully arbitrary-precision int (which
                # isn't available for a vectorised pandas column the way
                # it is for the scalar per-row case in data_conversion()
                # above).
                merged['start'] = merged['start'].astype('int64')
                merged['end'] = merged['end'].astype('int64')
                _end_adjust_int = int(round(end_adjust))
                merged['start'] = merged['start'] - _end_adjust_int
                merged['end'] = merged['end'] + _end_adjust_int
                merged['chromosome'] = merged['chromosome'].astype(str)

                chromosome_total = pd.unique(merged['chromosome'])

                # Requirement (bugfix): clamp BOTH the widened start and
                # end to THIS SPECIFIC chromosome's own bounds - not a
                # hardcoded 0 for start (chromosomes don't always start
                # at coordinate 0 in every coordinate system a chrom_info
                # file might use), and not another chromosome's bounds by
                # mistake. Both sides are handled together, per
                # chromosome, so a widened marker region can never spill
                # outside where its own chromosome actually starts/ends.
                for k in range(len(chromosome_total)):
                    _chrom_mask = merged['chromosome'] == chromosome_total[k]
                    _chrom_row = chromosome.loc[chromosome['chromosome'] == chromosome_total[k]]
                    if _chrom_row.shape[0] == 0:
                        continue
                    _chrom_start = int(_chrom_row['start'].values[0])
                    _chrom_end = int(_chrom_row['end'].values[0])
                    _clamp_region_to_chromosome(merged, _chrom_mask, _chrom_start, _chrom_end)
                merged.to_csv(_circos_intermediate_dir(RESULT_NAME)+'/marker_effect_'+str(MODEL[iii])+'_'+str(PHENOTYPE)+'_'+str(POPULATION)+'.tsv', sep ='\t',index=False)
            else:
                effect_selected.columns = ['effect']
                effect_selected = effect_selected.reset_index(drop=False)
                marker = _load_combined_marker_info(marker_info, RESULT_NAME, PHENOTYPE)
                effect_selected = pd.merge(effect_selected, marker, left_on=['index'], right_on=['name'])
                effect_selected = effect_selected.loc[:,['chromosome','start','end','index','effect']]

                effect_selected['chromosome'] = effect_selected['chromosome'].astype(str)
                
                effect_selected['range'] = (effect_selected['start'] + effect_selected['end'])/2
                
                chromosome_total = pd.unique(effect_selected['chromosome'])
                
                for k in range(len(chromosome_total)):
                    effect_selected.loc[(effect_selected['chromosome']==chromosome_total[k]) & 
                               (effect_selected['range'] > chromosome.loc[chromosome['chromosome']==chromosome_total[k],'end'].values[0]),'range'] = int(chromosome.loc[chromosome['chromosome']==chromosome_total[k], 'end'].values[0])

                effect_selected = effect_selected.groupby(['chromosome',pd.cut((effect_selected['range']), bins=division)]).sum().drop(['start','end', 'range'],axis=1).reset_index(drop=False)
                effect_selected = effect_selected.rename(columns={'range':'interval'})
                effect_selected['start'] =[int(round(effect_selected['interval'][k].left)) for k in range(effect_selected['interval'].shape[0])]
                effect_selected['end'] =[int((effect_selected['interval'][k].right)) for k in range(effect_selected['interval'].shape[0])]
                
                effect_selected = effect_selected.drop('interval',axis=1)
                # Requirement (bugfix): clamp to each bin's own
                # chromosome start, not a hardcoded 0 - matching the same
                # fix already applied elsewhere in this file (chromosomes
                # don't always start at coordinate 0 in every coordinate
                # system a chrom_info file might use).
                for k in range(len(chromosome_total)):
                    _chrom_row = chromosome.loc[chromosome['chromosome'] == chromosome_total[k]]
                    if _chrom_row.shape[0] == 0:
                        continue
                    _chrom_start = int(_chrom_row['start'].values[0])
                    effect_selected.loc[
                        (effect_selected['chromosome'] == chromosome_total[k]) &
                        (effect_selected['start'] < _chrom_start), 'start'
                    ] = _chrom_start
                if not _all_zero_effect:
                    effect_selected = effect_selected[effect_selected['effect'] > 0].reset_index(drop=True)
                # else: keep every row (would otherwise be filtered down
                # to 0 rows, since every effect is exactly 0) - all get
                # assigned colour0 directly below instead.


                effect_selected_copy = effect_selected.copy()
                effect_selected_copy = effect_selected_copy.astype({effect_selected_copy.columns[2]: object})
                effect_selected_copy.iloc[:,2] = colour+'0'
                if not _all_zero_effect:
                    tmp = list(effect_selected[effect_selected['effect']>= np.quantile(effect_selected['effect'].to_numpy().flatten(), 0.1)].dropna().index)
                    effect_selected_copy.iloc[tmp,2] = colour+'1'
                    tmp = list(effect_selected[effect_selected['effect']>= np.quantile(effect_selected['effect'].to_numpy().flatten(), 0.2)].dropna().index)
                    effect_selected_copy.iloc[tmp,2] = colour+'2'
                    tmp = list(effect_selected[effect_selected['effect']>= np.quantile(effect_selected['effect'].to_numpy().flatten(), 0.3)].dropna().index)
                    effect_selected_copy.iloc[tmp,2] = colour+'3'
                    tmp = list(effect_selected[effect_selected['effect']>= np.quantile(effect_selected['effect'].to_numpy().flatten(), 0.4)].dropna().index)
                    effect_selected_copy.iloc[tmp,2] = colour+'4'
                    tmp = list(effect_selected[effect_selected['effect']>= np.quantile(effect_selected['effect'].to_numpy().flatten(), 0.5)].dropna().index)
                    effect_selected_copy.iloc[tmp,2] = colour+'5'
                    tmp = list(effect_selected[effect_selected['effect']>= np.quantile(effect_selected['effect'].to_numpy().flatten(), 0.6)].dropna().index)
                    effect_selected_copy.iloc[tmp,2] = colour+'6'
                    tmp = list(effect_selected[effect_selected['effect']>= np.quantile(effect_selected['effect'].to_numpy().flatten(), 0.7)].dropna().index)
                    effect_selected_copy.iloc[tmp,2] = colour+'7'
                    tmp = list(effect_selected[effect_selected['effect']>= np.quantile(effect_selected['effect'].to_numpy().flatten(), 0.8)].dropna().index)
                    effect_selected_copy.iloc[tmp,2] = colour+'8'
                    tmp = list(effect_selected[effect_selected['effect']>= np.quantile(effect_selected['effect'].to_numpy().flatten(), 0.9)].dropna().index)
                    effect_selected_copy.iloc[tmp,2] = colour+'9'
                
                merged = effect_selected_copy.loc[:,['chromosome', 'start', 'end', 'index', 'effect']]
                merged = merged.rename(columns={'effect':'colour'})
                
                # Requirement (bugfix): clamp BOTH start and end to each
                # bin's own chromosome, matching the same fix already
                # applied in the WINDOW==0 branch above and in
                # data_conversion() - this branch previously only ever
                # clamped 'end' here (never 'start'; that used a
                # hardcoded 0 further up instead of a chromosome's own
                # start - see effect_selected['start'] a few lines above)
                # and did so with a genuine crash-causing bug of its own:
                # int(...values) without an index on a (possibly multi-
                # element, or on newer numpy, even single-element) array
                # raises TypeError rather than converting - int(...
                # values[0]) is the correct scalar extraction, matching
                # every other clamp in this file.
                for k in range(len(chromosome_total)):
                    _chrom_mask = merged['chromosome'] == chromosome_total[k]
                    _chrom_row = chromosome.loc[chromosome['chromosome'] == chromosome_total[k]]
                    if _chrom_row.shape[0] == 0:
                        continue
                    _chrom_start = int(_chrom_row['start'].values[0])
                    _chrom_end = int(_chrom_row['end'].values[0])
                    _clamp_region_to_chromosome(merged, _chrom_mask, _chrom_start, _chrom_end)

                merged.to_csv(_circos_intermediate_dir(RESULT_NAME)+'/marker_effect_'+str(MODEL[iii])+'_'+str(PHENOTYPE)+'_'+str(POPULATION)+'.tsv', sep ='\t',index=False)
        else:
            REMOVE += [MODEL[iii]]
    
    MODEL = [e for e in MODEL if e not in REMOVE]
    
    return MODEL

def interaction(interaction, marker_info, PHENOTYPE, circos_config, POPULATION, RESULT_NAME, attention_original):
    
    interaction_selected_total = pd.DataFrame()
    model_selected = []    
    
    if interaction.shape[0]==0 and attention_original.shape[0]==0:
        return pd.DataFrame()
    else:
        if interaction.shape[0] != 0:
            model_selected += ['RF'] 
        
            # Extract key gmarker-by-marker interaction patterns
            if POPULATION == 'all' and interaction.shape[0]!=0:
                interaction = interaction.loc[:,['phenotype','marker1','marker2', 'value']]
            elif POPULATION != 'all' and interaction.shape[0]!=0:
                interaction = interaction.loc[:,['population','phenotype','marker1','marker2', 'value']]
                interaction = interaction[interaction['population'].astype(str)==str(POPULATION)].reset_index(drop=False)
            
            interaction = interaction[(interaction['marker1'] != 'factor') & (interaction['marker2'] != 'factor')]
            interaction = interaction.groupby(['phenotype','marker1','marker2'], as_index=False).mean(numeric_only=True)
            
            interaction_selected = interaction[interaction['phenotype'] == PHENOTYPE]
            interaction_selected = interaction_selected[interaction_selected['value'] >= np.quantile(interaction_selected['value'], (1-(circos_config['interaction_top']/100)))].reset_index(drop=True)
            interaction_selected['value'] = interaction_selected['value'] / interaction_selected['value'].sum()
            
            loc_info = _load_combined_marker_info(marker_info, RESULT_NAME, PHENOTYPE)
            
            start = pd.merge(interaction_selected['marker1'], loc_info, 'inner', left_on='marker1', right_on='name')
            end = pd.merge(interaction_selected['marker2'], loc_info, 'inner', left_on='marker2', right_on='name')
            
            start['start'] =[int(round(start.loc[k, 'start'])) for k in range(start.shape[0])]
            start['end'] =[int(round(start.loc[k, 'end'])) for k in range(start.shape[0])]
            end['start'] =[int(round(end.loc[k, 'start'])) for k in range(end.shape[0])]
            end['end'] =[int(round(end.loc[k, 'end'])) for k in range(end.shape[0])]
            
            chrom_start = start['chromosome'].astype(str)
            chrom_end = end['chromosome'].astype(str)
        
            interaction_selected = pd.concat([chrom_start, start.loc[:,['start','end']],
                                       chrom_end, end.loc[:,['start','end']],
                                       interaction_selected['value']],axis=1)
            interaction_selected.columns = ['chromosome_marker1', 'start','end','chromosome_marker2','start','end','value']
            interaction_selected['model'] = 'RF'

            interaction_selected_total = pd.concat([interaction_selected_total, interaction_selected])
        if attention_original.shape[0]!=0:
            models_GAT = attention_original['model'].unique().tolist()
            model_selected += models_GAT

            for i in range(len(models_GAT)):
                # Extract key gmarker-by-marker interaction patterns
                if POPULATION == 'all' and attention_original.shape[0]!=0:
                    attention = attention_original[attention_original['model']==models_GAT[i]].reset_index(drop=False)
                    attention = attention.loc[:,['phenotype','marker1','marker2', 'value']]
                elif POPULATION != 'all' and attention_original.shape[0]!=0:
                    attention = attention_original[attention_original['model']==models_GAT[i]].reset_index(drop=False)
                    attention = attention.loc[:,['population','phenotype','marker1','marker2', 'value']]
                    attention = attention[attention['population'].astype(str)==str(POPULATION)].reset_index(drop=False)
                attention = attention[(attention['marker1'] != 'factor') & (attention['marker2'] != 'factor')]
                attention = attention.groupby(['phenotype','marker1','marker2'], as_index=False).mean(numeric_only=True)
                
                attention = attention[attention['phenotype'] == PHENOTYPE]
                attention = attention[attention['value'] >= np.quantile(attention['value'], (1-(circos_config['interaction_top']/100)))].reset_index(drop=True)
                attention['value'] = attention['value'] / attention['value'].sum()
                
                loc_info = _load_combined_marker_info(marker_info, RESULT_NAME, PHENOTYPE)
                
                start = pd.merge(attention['marker1'], loc_info, 'inner', left_on='marker1', right_on='name')
                end = pd.merge(attention['marker2'], loc_info, 'inner', left_on='marker2', right_on='name')
                
                start['start'] =[int(round(start.loc[k, 'start'])) for k in range(start.shape[0])]
                start['end'] =[int(round(start.loc[k, 'end'])) for k in range(start.shape[0])]
                end['start'] =[int(round(end.loc[k, 'start'])) for k in range(end.shape[0])]
                end['end'] =[int(round(end.loc[k, 'end'])) for k in range(end.shape[0])]
                
                chrom_start = start['chromosome'].astype(str)
                chrom_end = end['chromosome'].astype(str)
            
                attention = pd.concat([chrom_start, start.loc[:,['start','end']],
                                           chrom_end, end.loc[:,['start','end']],
                                           attention['value']],axis=1)
                attention.columns = ['chromosome_marker1', 'start','end','chromosome_marker2','start','end','value']
                attention['model'] = models_GAT[i]
                interaction_selected_total = pd.concat([interaction_selected_total, attention])
            
    return interaction_selected_total

def _circos_axis_unit_label(circos_config):
    """Requirement 2: a human-readable label for what the tick numbers
    around the plot actually mean - shown once, at the centre of the
    plot (see plot()'s own circos.text(...,r=0,...) call), since
    xticks_by_interval()'s own label_formatter only ever prints the raw
    (scaled) number with no unit attached at all.

    Requirement (THIRD correction): the previous version showed
    circos_config['scale'] itself, exactly - technically accurate (a
    tick step of '1' really is 'scale' raw units), but confirmed
    directly against a real report to look wrong in practice: for
    scale=200 (chosen by the auto-suggestion formula's own 1/2/5 x 10^n
    'nice number' rule - 200 = 2 x 10^2), the reported IDEAL label was
    '100 cM', not '200 cM'; separately, a bp scale of 5,000,000 (5 x
    10^6) labelled as '5 Mb' was confirmed to look 'unnatural' - the
    ideal was apparently just 'Mb' (10^6), with no multiplier at all.

    Both examples agree on the same fix, which is also what was
    suggested directly: describe the axis using only the LARGEST POWER
    OF TEN that 'scale' doesn't exceed - i.e. keep only scale's own
    trailing zeros, drop its leading significant digit(s) (1, 2, or 5,
    from the nice-number rule). 200 -> 10^2 = 100. 5,000,000 -> 10^6 =
    1,000,000, which is EXACTLY 1 Mb, so no multiplier is needed at
    all - matching both reports at once. This describes the axis at a
    clean, round order of magnitude a reader can immediately relate to
    (matching how bp/kb/Mb/Gb are conventionally used as approximate
    magnitude indicators in genomics generally, not as an exact
    per-tick distance), rather than the exact, possibly-2x-or-5x 'nice
    number' scale value used for the real tick spacing underneath it -
    which is unchanged by this and still exactly what
    xticks_by_interval()/its label_formatter actually use for placing
    and numbering ticks; only how it's DESCRIBED in this one caption
    changes."""
    unit = circos_config.get('unit', 'bp')
    scale = circos_config.get('scale', 1) or 1
    power_of_ten = 10 ** math.floor(math.log10(scale)) if scale > 0 else 1

    if unit == 'cM':
        if power_of_ten == 1:
            return 'Position (cM)'
        return f'Position ({power_of_ten:g} cM)'

    if power_of_ten >= 1_000_000_000:
        unit_name, unit_size = 'Gb', 1_000_000_000
    elif power_of_ten >= 1_000_000:
        unit_name, unit_size = 'Mb', 1_000_000
    elif power_of_ten >= 1_000:
        unit_name, unit_size = 'kb', 1_000
    else:
        unit_name, unit_size = 'bp', 1
    multiplier = power_of_ten / unit_size
    if multiplier == 1:
        return f'Position ({unit_name})'
    return f'Position ({multiplier:g} {unit_name})'


class _GradientSwatchHandler(mlegend_handler.HandlerBase):
    """Requirement 1 (correction, matching the attached 'example.png'
    reference format): draws a smooth light-to-dark colour gradient
    rectangle for a legend entry, instead of matplotlib's usual single
    flat-colour swatch - the marker-effect quantile scale is a
    continuous range, not two discrete levels, so the legend should look
    like one now, exactly like the reference image's own 'Genomic marker
    effect' gradient bars."""
    def __init__(self, low_hex, high_hex, **kwargs):
        self.low_hex = low_hex
        self.high_hex = high_hex
        super().__init__(**kwargs)

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        n = 24
        low_rgb = np.array(mcolors.to_rgb(self.low_hex))
        high_rgb = np.array(mcolors.to_rgb(self.high_hex))
        patches = []
        for i in range(n):
            frac = i / (n - 1)
            color = tuple(low_rgb + (high_rgb - low_rgb) * frac)
            rect = mpatches.Rectangle(
                (xdescent + frac * width, ydescent), width / n * 1.08, height,
                facecolor=color, edgecolor='none', transform=trans,
            )
            patches.append(rect)
        return patches


def _build_circos_legend_handles(colours_used_by_hue, CYTOBAND_COLORMAP, gene_colours_used, has_inter_chr_link, has_intra_chr_link):
    """Requirement 13 (and its Requirement 1 correction): build the
    legend entries for the current plot - ONLY for colours/categories
    that actually appear on THIS specific rendering, never every colour
    the underlying scheme could theoretically produce.

    Returns (handles, handler_map) - `handler_map` pairs each gradient-bar
    handle with a _GradientSwatchHandler instance (see plot()'s own
    fig.legend(..., handler_map=handler_map) call, which is REQUIRED for
    the gradient rendering to actually happen - a plain fig.legend(handles)
    call would fall back to treating each handle as an ordinary flat-colour
    patch instead).

    `colours_used_by_hue` is a {hue: {colour strings actually found in
    that hue's marker_effect_*.tsv file(s), e.g. 'blue3'}} mapping - built
    by plot() itself from the SAME tsv file(s) it just rendered, so this
    reflects the real, current data rather than an assumption. For each
    hue actually present, this shows a GRADIENT bar spanning the LOWEST
    to HIGHEST quantile level ACTUALLY FOUND (not hardcoded 0-9) -
    labelled 'Marker effect (single model)' for blue and 'Marker effect
    (ensemble model)' for red, matching what these two hues actually
    represent (see quantile_conversion()'s own colour = 'red' if MODEL in
    [ensemble/meta-model combinations] else 'blue') - NOT a meaningless
    'low effect'/'high effect' per-swatch label, which is what this
    looked like before this correction.

    `gene_colours_used` is the set of DISTINCT 'colour' (pathway/category)
    values actually found in the gene tsv file(s) rendered this plot -
    NOT the 'source' values (e.g. 'leaf'/'SAM'/'QTL'/'wisser_et_al'),
    which are just RING LABELS identifying which data source a gene
    annotation came from, never colour-coded at all (gene_info.csv's
    'colour' column - a pathway/category name like 'photoperiod' - is
    what actually determines the fill colour, per data_conversion()'s own
    gene_selected.drop(['source','population','phenotype'],...) call,
    which leaves 'colour' as BED's 5th/'score' column,
    _add_cytoband_tracks_with_border's own colour lookup key). Using
    'source' here (the original bug) meant looking up 'leaf'/'SAM'/etc.
    in CYTOBAND_COLORMAP, which are never actual keys in it at all - every
    such legend swatch silently fell back to plain white.

    `has_inter_chr_link`/`has_intra_chr_link` are checked independently
    (not a single 'has_links' flag) - a plot could have EITHER only
    within-chromosome or only between-chromosome interactions among
    whatever's actually displayed, and showing both colours regardless
    would list one that never appears as a line anywhere on the plot."""
    handles = []
    handler_map = {}
    if CYTOBAND_COLORMAP is None:
        CYTOBAND_COLORMAP = _pycirclize_config.CYTOBAND_COLORMAP

    _hue_label = {'blue': 'Marker effect (single model)', 'red': 'Marker effect (ensemble model)'}
    for hue in sorted(colours_used_by_hue):
        _levels = []
        for c in colours_used_by_hue[hue]:
            if isinstance(c, str) and c.startswith(hue) and c[len(hue):].isdigit():
                _levels.append(int(c[len(hue):]))
        if not _levels:
            continue
        _low_level, _high_level = min(_levels), max(_levels)
        low_c = CYTOBAND_COLORMAP.get(f'{hue}{_low_level}', '#ffffff')
        high_c = CYTOBAND_COLORMAP.get(f'{hue}{_high_level}', low_c)
        _label = _hue_label.get(hue, f'Marker effect ({hue})')
        if _high_level == _low_level:
            # Every marker actually rendered landed on the exact same
            # quantile level (e.g. the all-zero-effect case) - a flat
            # swatch, not a (visually identical) gradient, is the
            # honest representation here.
            handles.append(mpatches.Patch(facecolor=low_c, edgecolor='black', linewidth=0.3, label=f'{_label} (uniform)'))
        else:
            _proxy = mpatches.Patch(facecolor=low_c, edgecolor='black', linewidth=0.3, label=f'{_label}  weaker \u2192 stronger')
            handles.append(_proxy)
            handler_map[_proxy] = _GradientSwatchHandler(low_c, high_c)

    for colour_value in gene_colours_used:
        color = CYTOBAND_COLORMAP.get(str(colour_value), '#ffffff')
        handles.append(mpatches.Patch(facecolor=color, edgecolor='black', linewidth=0.3, label=str(colour_value)))

    if has_inter_chr_link:
        handles.append(mlines.Line2D([0], [0], color='blue', lw=2, label='inter-chr link'))
    if has_intra_chr_link:
        handles.append(mlines.Line2D([0], [0], color='red', lw=2, label='intra-chr link'))

    return handles, handler_map


def _save_circos_legend(handles, handler_map, save_path):
    """Requirement 1 (correction - separate legend file): builds and saves
    a small, STANDALONE figure containing ONLY the legend - no circos plot
    at all - to its own PNG file.

    Replaces the earlier approach of widening the main plot's own figure
    and shifting its axes to carve out room for the legend inside the SAME
    image - a legend can end up needing anywhere from a couple of rows to
    dozens (Requirement 1's own fix reads every colour genuinely used in
    the data, so a richly-annotated gene file needs a correspondingly
    long one), and getting a one-shot margin estimate right for that whole
    range, on top of an unpredictable chromosome count/sizing, proved
    fragile in practice. A completely separate file sidesteps needing
    that estimate at all: there is nothing for the legend to ever overlap,
    on any plot, at any size or row count.

    The main plot's own figure IS still adjusted slightly (see plot()'s
    own comment on its unit-label placement) - but only by a small, FIXED
    amount for that one line of caption text, not a variable amount
    depending on legend content, which is what made this approach workable
    for the label but not for the legend itself.

    Sized to the ACTUAL number of legend rows about to be drawn - a short
    legend gets a small image, a long one (dozens of gene-pathway colours)
    gets a taller one - rather than a fixed guess either way. Does nothing
    (no file written) if there's nothing to show a legend for."""
    if not handles:
        return
    n_rows = len(handles)
    fig_height_in = max(1.2, 0.28 * n_rows + 0.6)
    legend_fig = plt.figure(figsize=(3.2, fig_height_in))
    legend_fig.legend(handles=handles, handler_map=handler_map, loc='center left',
                       fontsize=8, title='Legend', title_fontsize=9, frameon=True)
    legend_fig.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close(legend_fig)


def plot(interactions_original, chrom_info, gene_info, pop_source, PHENOTYPE, MODEL, circos_config, CYTOBAND_COLORMAP, POPULATION, RESULT_NAME):
    
    if interactions_original.shape[0] != 0:
        model_selected = interactions_original['model'].unique().tolist()
    else:
        model_selected = ['not_returned']

    for n in range(len(model_selected)):
        cnt = 0
        circos = Circos.initialize_from_bed(_circos_intermediate_dir(RESULT_NAME)+'/chrom_'+str(POPULATION)+".bed", space=circos_config['space'], start=circos_config['start'], end=circos_config['end'])
        
        # Add genomic marker effects
        _colours_used_by_hue = {}  # e.g. {'blue': {'blue0', 'blue3', 'blue7'}, ...}
        for i in range(len(MODEL)):
             _tsv_path = _circos_intermediate_dir(RESULT_NAME)+'/marker_effect_'+MODEL[i]+'_'+PHENOTYPE+'_'+str(POPULATION)+'.tsv'
             _add_cytoband_tracks_with_border(circos, (97-(3*cnt), 100-(3*cnt)), _tsv_path, track_name=MODEL[i], cytoband_cmap=CYTOBAND_COLORMAP)
             circos.text(MODEL[i], r=circos.tracks[-1].r_center-1, deg=0, size=8, color="black")
             cnt+=1
             # Requirement 13 (correction): the legend should only ever
             # show colours that ACTUALLY appear on this specific plot,
             # not every colour the quantile scheme could theoretically
             # produce - read the SAME tsv file just rendered above and
             # record which 'colour' values are genuinely present (e.g.
             # an all-zero-effect model - see the all-zero-effect fix
             # elsewhere in this file - would leave EVERY marker at
             # 'blue0', and a legend hardcoded to always show 'blue9' as
             # 'high effect' would then be showing a colour that never
             # actually appears anywhere on the ring).
             try:
                 _tsv_colours = pd.read_csv(_tsv_path, sep='\t')['colour'].unique().tolist()
             except Exception:
                 _tsv_colours = []
             _hue = 'red' if MODEL[i] in ['ensemble', 'Linear transformation', 'Nelder Mead', 'Bayesian optimisation'] else 'blue'
             _colours_used_by_hue.setdefault(_hue, set()).update(_tsv_colours)
        
        # Add known gene regions
        gene_colours_used = set()
        if gene_info is not None:
            gene_source = pd.unique(pop_source.loc[(pop_source['population'].astype(str)==str(POPULATION)) & (pop_source['phenotype']==str(PHENOTYPE)),'source'])
            for i in range(len(gene_source)):    
                _gene_tsv_path = _circos_intermediate_dir(RESULT_NAME)+'/gene_info_'+str(PHENOTYPE)+'_'+str(gene_source[i])+'_'+str(POPULATION)+'.tsv'
                _add_cytoband_tracks_with_border(circos, (97-(3*cnt), 100-(3*cnt)), _gene_tsv_path, track_name=gene_source[i], cytoband_cmap=CYTOBAND_COLORMAP)
                circos.text(gene_source[i], r=circos.tracks[-1].r_center-1, deg=0, size=8, color="black")
                cnt+=1
                # Requirement 1 (correction): 'source' (leaf/SAM/QTL/
                # wisser_et_al, etc.) is only ever a RING LABEL identifying
                # which data source a gene annotation came from - it's
                # never itself a colour-coded category, and was never a
                # key in CYTOBAND_COLORMAP at all (the previous version's
                # bug - every such legend swatch silently fell back to
                # white). The actual fill colour comes from gene_info.csv's
                # 'colour' column (a pathway/category name, e.g.
                # 'photoperiod') - data_conversion() writes it as this
                # file's 5th/'score' column once 'source' is dropped, so
                # read it back here the same way the marker-effect legend
                # fix already reads its own tsv's 'colour' column.
                try:
                    gene_colours_used.update(pd.read_csv(_gene_tsv_path, sep='\t')['colour'].unique().tolist())
                except Exception:
                    pass
                
        # Add ticks to the outermost ring
        for sector in circos.sectors:
            # Requirement 6: a bit more radial space between the tick
            # numbers and the chromosome name text above them - label_margin
            # (pycirclize's own gap between a tick and its number label,
            # default 0.5) pushes the tick numbers a little further out,
            # and the chromosome name's own radius is nudged out to match
            # (105 -> 108) so the two don't end up crowding each other -
            # modest increases on both sides rather than a single large
            # jump, so the gap grows without leaving an awkwardly empty
            # ring between them.
            #
            # Requirement 1 (bugfix - the actual root cause of labels still
            # overlapping ring/tick data on a many-chromosome genome, e.g.
            # a real 26-chromosome cotton assembly): the chromosome NAME
            # text below used to be size=10, hardcoded - completely
            # unaffected by circos_config['label_size'] (the GUI's own
            # 'Label font size' field, and everything the Requirement 1
            # start/end-angle fix was calibrated against). The chromosome
            # name is the WIDER of the two texts (e.g. 'A01', vs a tick's
            # 1-3 digit number), so it was always the real driver of
            # overlap on a genome with many, densely-packed sectors - and
            # a fixed size=10 never shrank no matter how small
            # 'Label font size' was suggested/set to, which is exactly why
            # a real render still showed overlap despite that suggestion
            # already being smaller for many chromosomes.
            #
            # Tick numbers get their OWN, separately-derived size instead
            # of directly sharing circos_config['label_size'] - short
            # numeric ticks don't need to shrink nearly as aggressively as
            # a multi-character chromosome name does to avoid the same
            # overlap, and forcing them down to the (usually smaller)
            # chromosome-name size made them harder to read for no
            # actual benefit (a real, reported regression). Floored at
            # 5pt and capped at 8pt regardless of how small the
            # chromosome name itself gets.
            # Requirement (chromosome name now matches tick size): the
            # chromosome name used to render at circos_config['label_size']
            # directly (the smaller of the two sizes, deliberately kept
            # small so it wouldn't overlap neighbouring ticks/data on a
            # many-chromosome genome) while tick numbers got their own,
            # separately-derived _tick_label_size (see the comment above)
            # - by explicit request, the chromosome name now renders at
            # THAT SAME, larger size instead, to read more consistently
            # with the ticks around it. Since the chromosome name is the
            # WIDER of the two texts to begin with (e.g. 'A01', vs a
            # tick's 1-3 digit number), this makes it the same size AND
            # still the wider text - so it remains the real driver of how
            # much seam-gap room is needed, now more so than before. See
            # main_app.py's own _circos_suggest_start_end_angle() - it
            # applies this SAME size transform to whatever label size it's
            # given before calculating the gap, specifically so the two
            # stay in sync and this size increase doesn't reintroduce the
            # overlap it was originally calibrated against.
            _tick_label_size = max(5.0, min(8.0, circos_config['label_size'] * 1.8))
            sector.text(sector.name, r=108, size=_tick_label_size)
            sector.get_track(MODEL[0]).xticks_by_interval(
                circos_config['scale'],
                label_size=_tick_label_size,
                label_orientation="vertical",
                label_margin=1.5,
                label_formatter=lambda v: f"{v / circos_config['scale']:.0f}",
            )
            
        # Add marker-by-marker interactions
        has_inter_chr_link_for_legend = False
        has_intra_chr_link_for_legend = False
        if interactions_original.shape[0] != 0:
            interactions = interactions_original[interactions_original['model']==model_selected[n]].reset_index(drop=True)
            # Requirement: link OPACITY (not width) encodes each link's
            # relative importance ('value') - a higher value means a more
            # solid/opaque (visually "thicker-looking") link, while every
            # link is drawn at the same, fixed line width. 'value' itself
            # is normalized to sum to 1 across whatever's currently
            # displayed (see the `value / value.sum()` step above), so its
            # absolute scale shifts with how many links are shown - using
            # it as alpha directly (or via a fixed multiplier, as the old
            # width-based version did) would make the same link look very
            # different across plots with different link counts. Instead,
            # min-max normalize WITHIN this plot's own displayed link set,
            # into [link_alpha_min, 1.0] - the strongest link on any given
            # plot is always fully opaque, the weakest is always still at
            # least link_alpha_min visible (never fully invisible), and
            # everything in between scales smoothly - regardless of how
            # many links happen to be shown.
            link_alpha_min = circos_config.get('link_alpha_min', 0.15)
            # Fixed for every link - not user-configurable, since only
            # opacity (above) is meant to vary by strength; a separate
            # width control would just uniformly scale every line at once
            # (no diagnostic value) and was removed from the GUI.
            link_lw = 1.5
            if interactions.shape[0] != 0:
                v_min, v_max = interactions['value'].min(), interactions['value'].max()
                if v_max > v_min:
                    interactions['_alpha'] = link_alpha_min + (interactions['value'] - v_min) / (v_max - v_min) * (1.0 - link_alpha_min)
                else:
                    # Every displayed link has the same value (including the
                    # common case of a single link) - nothing to contrast
                    # against, so draw it fully opaque rather than an
                    # arbitrary floor value.
                    interactions['_alpha'] = 1.0
            for ii in range(interactions.shape[0]):
                region1 = (interactions.iloc[ii,0], interactions.iloc[ii,1], interactions.iloc[ii,2])
                region2 = (interactions.iloc[ii,3], interactions.iloc[ii,4], interactions.iloc[ii,5])
                if interactions.iloc[ii,0] != interactions.iloc[ii,3]:   #within chromosome or between chromosome
                    colour = 'blue'
                    has_inter_chr_link_for_legend = True
                else:
                    colour = 'red'
                    has_intra_chr_link_for_legend = True
                circos.link(region1, region2, lw=link_lw, alpha=float(interactions.loc[ii,'_alpha']), color=colour)
                
        # Store the circos plot
        fig = circos.plotfig()
        # Requirement (bugfix - unit label overlapping a chromosome name):
        # a fixed-position fig.text(0.5, 0.02, ...) worked for a modest
        # chromosome count, but for a genome with MANY chromosomes (e.g.
        # a real 26-sub-genome cotton assembly, A01-A13/D01-D13) some
        # chromosome's own name label inevitably ends up positioned right
        # at the bottom of the circle, where the unit label was also
        # fixed - colliding with it (confirmed with a real render at that
        # scale during development). Rather than trying to predict WHICH
        # chromosome that will be (unknowable in general - it depends on
        # chromosome count, sizes, and the start/end angles), this opens
        # a small, fixed, DEDICATED margin below the circle instead -
        # unlike the legend (Requirement 13's own correction, a separate
        # file), a single line of caption text doesn't need a variable-
        # sized allowance, so growing the figure by a small fixed amount
        # and shifting the polar axes up within it by exactly that much
        # (so the circle's own on-page size is unchanged) reliably clears
        # space for it - verified empirically against a real 26-
        # chromosome render, not just assumed.
        _extra_height_in = 0.6
        _fig_width_in, _fig_height_in = fig.get_size_inches()
        _polar_ax = fig.axes[0]
        _old_pos = _polar_ax.get_position()
        _old_y0_in = _old_pos.y0 * _fig_height_in
        _old_height_in = _old_pos.height * _fig_height_in
        _new_fig_height_in = _fig_height_in + _extra_height_in
        fig.set_size_inches(_fig_width_in, _new_fig_height_in)
        _polar_ax.set_position([
            _old_pos.x0,
            (_old_y0_in + _extra_height_in) / _new_fig_height_in,
            _old_pos.width,
            _old_height_in / _new_fig_height_in,
        ])
        fig.text(0.5, 0.02, _circos_axis_unit_label(circos_config), ha='center', va='bottom',
                 fontsize=9, color='black')

        # Requirement 13 (correction - separate legend file): an earlier
        # version tried to fit the legend INTO this same image, widening
        # the figure and shrinking the polar axes' own reported bounding
        # box to make room - this looked correct in isolated testing, but
        # broke down on real plots: chromosome NAME labels are drawn well
        # outside the polar axes' own nominal r-max (see the sector.text(
        # ..., r=108, ...) call above - deliberately larger than the
        # marker-effect rings' own r=100 so labels sit clearly outside
        # them), so matplotlib's own ax.get_position() (a plain rectangle)
        # never actually captured where those labels really render on the
        # page - repositioning based on it could shrink the DECLARED axes
        # bounds without preventing labels drawn OUTSIDE those bounds from
        # still overlapping the legend, and (for a genuinely large,
        # densely-labelled plot) the plot itself could still collide with
        # a tall legend regardless of how the two were arranged on one
        # shared canvas.
        #
        # Saved as a completely separate PNG instead - this sidesteps the
        # whole problem rather than trying to precisely predict it: the
        # circos plot's own layout is now COMPLETELY UNCHANGED from
        # before the legend feature existed (this exact fig.text() call
        # and the plain, un-widened fig = circos.plotfig() above are both
        # back to their pre-legend form), and the legend lives in its own
        # image, sized only for its own content, with nothing else on the
        # page it could ever collide with. See _save_circos_legend().
        _legend_handles, _legend_handler_map = _build_circos_legend_handles(
            _colours_used_by_hue, CYTOBAND_COLORMAP, gene_colours_used,
            has_inter_chr_link_for_legend, has_intra_chr_link_for_legend,
        )
        if model_selected[n] == 'not_returned':
            _plot_path = './Result/'+RESULT_NAME+'/circos_'+str(PHENOTYPE)+'_'+str(POPULATION)+'.png'
            fig.savefig(_plot_path, dpi=600)
            _save_circos_legend(_legend_handles, _legend_handler_map,
                                 './Result/'+RESULT_NAME+'/circos_'+str(PHENOTYPE)+'_'+str(POPULATION)+'_legend.png')
        else:
            _plot_path = './Result/'+RESULT_NAME+'/circos_'+str(PHENOTYPE)+'_'+str(POPULATION)+'_interaction_'+str(model_selected[n])+'.png'
            fig.savefig(_plot_path, dpi=600)
            _save_circos_legend(_legend_handles, _legend_handler_map,
                                 './Result/'+RESULT_NAME+'/circos_'+str(PHENOTYPE)+'_'+str(POPULATION)+'_interaction_'+str(model_selected[n])+'_legend.png')

def _clamp_region_to_chromosome(df, mask, chrom_start, chrom_end):
    """Requirement (bugfix - a widened marker/gene position could come out
    excessively large or small, not matching a simple GENE_ADJUST/
    END_ADJUST addition or subtraction): every call site that widens a
    region and then clamps it to its own chromosome's bounds used to
    clamp 'start' only from BELOW (raise it if under chrom_start) and
    'end' only from ABOVE (lower it if over chrom_end) - correct on its
    own for the ordinary case, but it silently left 'start' completely
    unclamped whenever it was ALSO too large (above chrom_end), and
    likewise left 'end' unclamped whenever it was ALSO too small (below
    chrom_start).

    Confirmed against real data this actually happens: a gene's own
    reported position can already extend past its chromosome's stated
    end even BEFORE any widening is applied at all (e.g. a coordinate
    system mismatch between a gene_info file and its chrom_info file) -
    widening such a position pushes it even further out, 'end' gets
    correctly clamped down to chrom_end by the old one-sided check, but
    'start' - despite being even larger than the now-clamped 'end' -
    never gets touched, producing a nonsensical start > end region
    that's neither the widened value NOR a sensible clamp of it.

    The fix: clamp 'start' and 'end' fully independently, each to the
    complete [chrom_start, chrom_end] range (both directions, not just
    one) - this is provably safe (never introduces a start > end that
    wasn't already true beforehand): clamping is a monotonic operation,
    so if start <= end before clamping (always true here - a widening
    step only ever moves start down and end up from an originally valid
    start <= end, by the same amount in each direction), then
    clamp(start) <= clamp(end) afterward too, for any shared [lo, hi]
    bounds - regardless of by how much either one originally overshot.

    Modifies df in place (matching every call site's own prior in-place
    .loc[...] = ... usage); returns nothing."""
    df.loc[mask & (df['start'] < chrom_start), 'start'] = chrom_start
    df.loc[mask & (df['start'] > chrom_end), 'start'] = chrom_end
    df.loc[mask & (df['end'] > chrom_end), 'end'] = chrom_end
    df.loc[mask & (df['end'] < chrom_start), 'end'] = chrom_start


def _warn_if_raw_position_exceeds_chromosome(df, chromosome, file_description, result_name_for_log=None):
    """Requirement (diagnostic - companion to the clamping fix above):
    the clamp fix guarantees a widened region can never come out
    nonsensical (start > end) any more, but a region whose RAW,
    un-widened position ALREADY exceeded its own chromosome's stated
    bounds - confirmed to happen in practice, see
    _clamp_region_to_chromosome()'s own docstring - still gets silently
    squashed down to a single point at that chromosome's very edge,
    which is technically valid but doesn't reflect where the position
    actually was. Surfaced here as an explicit, specific warning (the
    same pattern as the chromosome-name-mismatch warning above) rather
    than left for a person to notice only as an oddly-clustered handful
    of markers/genes sitting right at a chromosome's edge with no
    obvious explanation. Call this BEFORE any widening is applied, on
    the raw start/end columns."""
    _n_bad = 0
    _examples = []
    for _chrom_name, _group in df.groupby('chromosome'):
        _chrom_row = chromosome.loc[chromosome['chromosome'] == _chrom_name]
        if _chrom_row.shape[0] == 0:
            continue
        _c_start = int(_chrom_row['start'].values[0])
        _c_end = int(_chrom_row['end'].values[0])
        _out_of_range = _group[(_group['start'] < _c_start) | (_group['start'] > _c_end) |
                                (_group['end'] < _c_start) | (_group['end'] > _c_end)]
        if _out_of_range.shape[0] > 0:
            _n_bad += _out_of_range.shape[0]
            for _, _row in _out_of_range.head(3 - len(_examples) if len(_examples) < 3 else 0).iterrows():
                _examples.append(f"{_chrom_name} {int(_row['start'])}-{int(_row['end'])} (chromosome is {_c_start}-{_c_end})")
    if _n_bad > 0:
        print(
            f"[circos] WARNING: {_n_bad} row(s) in the {file_description} have a position that already "
            f"falls outside their own chromosome's stated start/end - BEFORE any widening is applied. "
            f"These get clamped to sit exactly at their chromosome's edge instead of crashing or coming "
            f"out reversed, but that edge position won't reflect where the row actually is. This usually "
            f"means the {file_description} and the chromosome info file don't agree on that chromosome's "
            f"coordinate system (e.g. a different genome assembly version). Example(s): {_examples}"
        )


def _clean_population_label(pop):
    """Requirement (bugfix - gene ring silently missing for real
    populations, only ever present for 'all'): normalizes a population
    VALUE to a consistent, 'clean' string form, so the SAME population is
    never accidentally treated as two different ones depending on which
    code path it happened to pass through on its way here.

    Concretely: Parallel mode's own assemble() (main_app.py) computes
    population = pd.unique(metric['population']).tolist() from the
    combined Metric_*.csv across every batch - and pandas silently
    promotes an entire int column to float64 the moment even ONE value
    in it is missing/NaN anywhere across those concatenated batches (a
    well-known, common pandas behaviour, not specific to any one
    dataset - e.g. one incomplete/placeholder row in a single batch's
    own Metric_N.csv is enough). Once that happens, EVERY population
    value becomes e.g. 1.0 instead of 1, and str(1.0) == '1.0', not '1' -
    which then fails to match a gene/chromosome file correctly written
    under the 'clean' label '1' (by data_conversion(), from the
    broadcast file's own str-cast population column), or fails
    pop_source's own population-column lookup in plot(). The literal
    string 'all' is never affected by this (it was never a number to
    begin with), which is exactly why this symptom shows up as 'only
    ever works for the all population' - every real, numeric population
    silently fails the same string comparison in the same way.

    Converts a whole-number float (1.0, 2.0, ...) to its clean integer
    string ('1', '2', ...); anything else (already a clean int/str, or
    a non-numeric string like 'all') is returned as plain str(pop)
    unchanged."""
    if isinstance(pop, float) and pop.is_integer():
        return str(int(pop))
    return str(pop)


def _broadcast_population_info(path, target_populations, description):
    """Requirement 8: expands a chromosome/gene info file's rows across
    every population circos actually needs, when the person has said (via
    the 'Chromosome/gene lengths are the same for every population'
    checkbox) that they're all identical - lets them provide the
    coordinates ONCE instead of manually duplicating every row once per
    population plus once more for 'all' (which circos_plot() itself
    always adds on top of the real populations - see its own
    `POPULATION = ('all',) + tuple(POPULATION)`).

    Lives here in circos_plot.py (moved from main_app.py) rather than
    duplicated separately in every place that needs it - this exact
    kind of duplication (a near-identical copy living in the Parallel/
    Sequential HEADLESS scripts, run_step2_assemble.py/run_sequential.py,
    never updated when this function or the broadcast feature as a whole
    changed in main_app.py) was a real, confirmed cause of the broadcast
    step silently never running at all for headless/HPC runs - both of
    those scripts call circos_plot() directly with the raw, un-broadcast
    cfg['CHROMOSOME_INFO']/cfg['GENE_INFO'] paths, with no broadcast logic
    of their own. A single shared implementation, imported by every
    caller (main_app.py's own GUI code included) from this one place,
    means there is only ever one version of this logic to keep correct
    and in sync - the caller (see run_step2_assemble.py/run_sequential.py/
    main_app.py, all updated to call this the same way) is still
    responsible for checking whether broadcasting was actually requested
    at all (a config flag) and for the SCENARIO='between' population-label
    split beforehand (that needs SCENARIO, which isn't a parameter here).

    Any existing 'population' column is dropped first (its values don't
    matter in this mode - only that a value CAN be written back in,
    correctly, per target population) and a fresh one appended at the
    end, matching where circos_plot.py's own unify_columns_by_position()
    calls expect it (chromosome/gene info files both have 'population' as
    their LAST expected column).

    Writes a new, fully-expanded TEMPORARY CSV - never touches or
    overwrites the person's own original file - and returns its path;
    the caller uses this path for the rest of the circos-plotting
    pipeline, exactly as if the duplication had been done by hand.
    Raises ValueError (surfaced to the person, not silently swallowed)
    if the source file can't be read at all, since silently falling back
    to the original (non-broadcast) file here would produce a confusing
    'file not found'/schema error much later instead.
    """
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Could not read {description} at {path!r} to broadcast it across populations: {exc}")

    if 'population' in df.columns:
        df = df.drop(columns=['population'])

    frames = []
    for pop in target_populations:
        _block = df.copy()
        _block['population'] = pop
        frames.append(_block)
    expanded = pd.concat(frames, ignore_index=True)

    tmp_path = os.path.join(tempfile.gettempdir(), f"easigp_broadcast_{description}_{uuid.uuid4().hex[:8]}.csv")
    expanded.to_csv(tmp_path, index=False)
    return tmp_path


def circos_plot(effect, interactions, marker_info, chrom_info, gene_info, POPULATION, PHENOTYPE, circos_config, end_adjust, WINDOW, CYTOBAND_COLORMAP, RESULT_NAME, attention, SCENARIO, ASCENDING, gene_adjust=0):

    pop_source =  data_conversion(chrom_info, gene_info, PHENOTYPE, RESULT_NAME, gene_adjust=gene_adjust)
    
    if SCENARIO == 'between' and interactions.shape[0] != 0:
        interactions['population'] = interactions['population'].str.split('->', expand=True).iloc[:,-1]
    if SCENARIO == 'between' and interactions.shape[0] != 0:
        attention['population'] = attention['population'].str.split('->', expand=True).iloc[:,-1]
    if SCENARIO == 'between' and effect.shape[0] != 0:
        effect['population'] = effect['population'].str.split('->', expand=True).iloc[:,-1]
    if SCENARIO == 'between':
        # Requirement (bugfix - gene ring silently missing for real
        # populations, SCENARIO='between' specifically): interactions/
        # attention/effect above all get their own 'population' column
        # split on '->' (a 'between' scenario's own population values
        # are combined train->test labels, e.g. 'Historical->2014') and
        # only the test-population half kept - but POPULATION itself
        # (this function's own parameter, used directly for every
        # file-name/comparison in quantile_conversion()/interaction()/
        # plot() below, and by the broadcast step in main_app.py before
        # this function is even called) was never given the same
        # treatment. Left uncombined, POPULATION[j] stays e.g.
        # 'Historical->2014' - which never matches a gene/chromosome
        # file correctly written under the clean, split label '2014'
        # (by data_conversion(), from the broadcast file's own clean
        # population column) - exactly the same 'gene ring silently
        # missing, but not for the always-clean-string all' symptom as
        # the earlier float-promotion bug, just triggered by a
        # different upstream cause specific to this scenario.
        POPULATION = [p.split('->')[-1] if isinstance(p, str) and '->' in p else p for p in POPULATION]
    
    # Requirement (bugfix): normalize every population value to its
    # clean string form HERE, once, immediately - before it's used for
    # ANY file-naming or comparison downstream (quantile_conversion(),
    # interaction(), plot() all receive POPULATION[j] straight from
    # this tuple) - see _clean_population_label()'s own docstring for
    # why this specific normalization (float 1.0 -> '1', not '1.0') is
    # necessary, not just a defensive nicety.
    POPULATION = ('all',) + tuple(_clean_population_label(p) for p in POPULATION)
        
    MODEL = pd.unique(effect['model'])
    
    for i in range(len(PHENOTYPE)):
        for j in range(len(POPULATION)):
            MODEL = quantile_conversion(effect, marker_info, chrom_info, PHENOTYPE[i], MODEL, end_adjust, POPULATION[j], WINDOW, RESULT_NAME, ASCENDING)
            interaction_selected = interaction(interactions, marker_info, PHENOTYPE[i], circos_config, POPULATION[j],RESULT_NAME, attention)
            plot(interaction_selected, chrom_info, gene_info, pop_source, PHENOTYPE[i], MODEL, circos_config, CYTOBAND_COLORMAP, POPULATION[j],RESULT_NAME)
