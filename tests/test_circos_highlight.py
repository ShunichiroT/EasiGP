"""Tests for the circos region-highlight option (`circos_highlight.py`, its use in `circos_plot.py`,
and its config round trip through `gui_config_loader.py`).

Run: python -m pytest tests/test_circos_highlight.py -q. Fixtures are generated here; the render
tests are skipped when pyCirclize is not installed."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import circos_highlight as hl  # noqa: E402
import gui_config_loader as gcl  # noqa: E402


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
def test_parse_position_accepts_suffixes_and_rejects_garbage():
    assert hl.parse_position('12.5Mb') == 12_500_000
    assert hl.parse_position(' 300 kb ') == 300_000
    assert hl.parse_position('1,200,000') == 1_200_000
    assert hl.parse_position(2e6) == 2_000_000
    assert hl.parse_position('2e6') == 2_000_000
    assert hl.parse_position('') is None and hl.parse_position(None) is None and hl.parse_position(float('nan')) is None
    assert hl.parse_position('42.5', 'cM') == 42.5 and hl.parse_position('42.5cM', 'cM') == 42.5
    for bad in ('abc', '12 parsecs', '-5', '1..2'):
        with pytest.raises(ValueError):
            hl.parse_position(bad)
    with pytest.raises(ValueError):
        hl.parse_position('5Mb', 'cM')                       # a size suffix makes no sense for cM


def test_position_text_round_trips():
    for v in (None, 0, 1500, 300_000, 1_234_567, 12_500_000, 2_000_000_000, 10_000_000.5):
        text = hl.position_to_text(v)
        assert hl.parse_position(text) == (None if v is None else pytest.approx(float(v)))
    assert hl.position_to_text(12_500_000) == '12.5Mb'


def test_parse_region_spec():
    rows = hl.parse_region_spec('A03:12Mb-15.5Mb=QTL1 ; A05 ;\nD02:3000000-4000000\nChr1:5Mb-')
    assert [r['chrom'] for r in rows] == ['A03', 'A05', 'D02', 'Chr1']
    assert rows[0] == {'chrom': 'A03', 'start': '12Mb', 'end': '15.5Mb', 'label': 'QTL1', 'colour': None}
    assert rows[1]['start'] is None and rows[1]['end'] is None                  # whole chromosome
    assert rows[3]['start'] == '5Mb' and rows[3]['end'] is None                 # open-ended
    assert hl.parse_region_spec('  ;; \n') == []


def test_clean_regions_skips_blank_rows_and_reports_bad_ones():
    regions, problems = hl.clean_regions([
        {'chrom': 'Chr1', 'start': '1Mb', 'end': '2Mb', 'label': 'a', 'colour': 'orange'},
        {'chrom': None, 'start': None, 'end': None, 'label': None, 'colour': None},        # empty editor line
        {'chrom': float('nan'), 'start': '', 'end': '', 'label': '', 'colour': 'auto'},   # ditto
        {'chrom': 'Chr2', 'start': '5Mb', 'end': '1Mb'},                                  # start > end
        {'chrom': '', 'start': '1Mb'},                                                    # no chromosome
        {'chrom': 'Chr3', 'start': 'x'},                                                  # unreadable
        {'chrom': 'Chr4', 'colour': 'not-a-colour'},
    ])
    assert len(regions) == 1 and regions[0]['colour'] == '#E69F00' and regions[0]['start'] == 1_000_000
    assert len(problems) == 4 and 'start is after end' in problems[0]


def test_options_default_to_off_and_survive_junk():
    o = hl.highlight_options({})
    assert o['regions'] == [] and o['mode'] == 'emphasise' and o['match'] == 'either' and o['band'] and o['shade'] and o['label']
    assert hl.highlight_options(None)['regions'] == []
    o = hl.highlight_options({'highlight_regions': [{'chrom': 'A'}], 'highlight_mode': 'bogus', 'highlight_dim_opacity': 'x',
                              'highlight_lw_boost': 99, 'highlight_colour': 'nope', 'highlight_band': 'false'})
    assert o['mode'] == 'emphasise' and o['dim_opacity'] == hl.DEFAULT_DIM_OPACITY
    assert o['lw_boost'] == 6.0 and o['colour'] == hl.DEFAULT_ACCENT and o['band'] is False


# ---------------------------------------------------------------------------
# resolving + matching
# ---------------------------------------------------------------------------
RANGES = {'Chr01': (0.0, 100e6), 'Chr02': (0.0, 80e6), 'ChrX': (0.0, 50e6)}


def test_resolve_matches_names_loosely_and_clamps():
    regions, _ = hl.clean_regions([{'chrom': 'chr1', 'start': '90Mb', 'end': '120Mb', 'label': 'late'},
                                   {'chrom': '2'}, {'chrom': 'x'}, {'chrom': 'Chr9'},
                                   {'chrom': 'Chr01', 'start': '200Mb', 'end': '300Mb'}])
    res, problems = hl.resolve_regions(regions, RANGES)
    assert [r['chrom'] for r in res] == ['Chr01', 'Chr02', 'ChrX']
    assert res[0]['end'] == 100e6 and res[0]['start'] == 90e6                        # clamped to the chromosome
    assert res[1]['whole'] and res[1]['tag'] == 'R2'
    assert [r['idx'] for r in res] == [0, 1, 2]
    assert len(problems) == 2 and 'Chr9' in problems[0] and 'outside the chromosome' in problems[1]
    assert res[0]['legend'].startswith('late  (')


def test_colours_follow_the_user_list_not_the_valid_rows():
    regions, _ = hl.clean_regions([{'chrom': 'Chr9'}, {'chrom': 'Chr01'}, {'chrom': 'Chr02', 'colour': 'black'}])
    res, _ = hl.resolve_regions(regions, RANGES)
    assert res[0]['colour'] == hl.PALETTE[1]                                         # index 1 even though row 0 was dropped
    assert res[1]['colour'] == '#000000'
    single, _ = hl.resolve_regions(regions, RANGES, {'colour_by': 'single', 'colour': '#123456'})
    assert {r['colour'] for r in single} == {'#123456'}


def _res(*specs):
    regions, _ = hl.clean_regions(list(specs))
    return hl.resolve_regions(regions, RANGES)[0]


def test_classify_pairs_either_and_both():
    res = _res({'chrom': 'Chr01', 'start': '10Mb', 'end': '20Mb'}, {'chrom': 'Chr02'})
    c1 = ['Chr01', 'Chr01', 'Chr02', 'ChrX', 'Chr01']
    s1 = [12e6, 25e6, 5e6, 1e6, 19.9e6]
    e1 = [12e6 + 1, 25e6 + 1, 5e6 + 1, 1e6 + 1, 20.5e6]                               # last one straddles the end of region 0
    c2 = ['ChrX', 'ChrX', 'Chr01', 'ChrX', 'Chr02']
    s2 = [1e6, 1e6, 15e6, 2e6, 1e6]
    e2 = [1e6 + 1, 1e6 + 1, 15e6 + 1, 2e6 + 1, 1e6 + 1]
    hit, region = hl.classify_pairs(c1, s1, e1, c2, s2, e2, res, 'either')
    assert hit.tolist() == [True, False, True, False, True]
    assert region.tolist() == [0, -1, 0, -1, 0]                                       # row 2: Chr02 end and Chr01 end -> region 0 (listed first)
    hit, region = hl.classify_pairs(c1, s1, e1, c2, s2, e2, res, 'both')
    assert hit.tolist() == [False, False, True, False, True]
    assert region.tolist() == [-1, -1, 0, -1, 0]


def test_classify_members_for_hyperedges():
    res = _res({'chrom': 'Chr01', 'start': '10Mb', 'end': '20Mb'})
    members = [[('Chr01', 15e6), ('ChrX', 1e6), ('Chr02', 3e6)], [('ChrX', 1e6), ('Chr02', 3e6)], [('Chr01', 12e6), ('Chr01', 18e6)], []]
    hit, region = hl.classify_members(members, res, 'either')
    assert hit.tolist() == [True, False, True, False] and region.tolist() == [0, -1, 0, -1]
    hit, _ = hl.classify_members(members, res, 'both')
    assert hit.tolist() == [False, False, True, False]


def test_style_lifts_highlighted_and_fades_the_rest():
    opts = hl.highlight_options({})
    alpha = np.array([0.15, 0.5, 1.0, 0.15, 1.0])
    hit = np.array([True, True, True, False, False])
    a, lw = hl.style_alpha_lw(alpha, np.full(5, 2.0), hit, opts, 0.15)
    assert a[:3].min() == pytest.approx(hl.HIGHLIGHT_ALPHA_FLOOR) and a[2] == pytest.approx(1.0) and a[0] < a[1] < a[2]
    assert a[3] == pytest.approx(0.05) or a[3] >= 0.04                                 # faded, never invisible
    assert a[4] == pytest.approx(opts['dim_opacity'])
    assert lw[0] == pytest.approx(2.0 * opts['lw_boost']) and lw[3] < 2.0
    assert hl.draw_order(hit).tolist() == [3, 4, 0, 1, 2]                              # highlighted drawn last, order kept


def test_preview_figure_builds_without_pyplot():
    res = _res({'chrom': 'Chr01', 'start': '10Mb', 'end': '20Mb', 'label': 'QTL'}, {'chrom': 'ChrX'})
    fig = hl.preview_figure(RANGES, res)
    assert len(fig.axes) == 1


# ---------------------------------------------------------------------------
# key genes: suggest / add / remove by name or interval
# ---------------------------------------------------------------------------
GENE_ROWS = [{'name': 'GA1', 'chrom': 'chr1', 'start': 1e6, 'end': 1.2e6, 'phenotype': 'yield'},
             {'name': 'GA1', 'chrom': 'chr1', 'start': 1e6, 'end': 1.2e6, 'phenotype': 'height'},     # same gene, other phenotype
             {'name': 'B2', 'chrom': 'Chr03', 'start': 9e6, 'end': 5e6},                              # swapped ends
             {'name': 'Off', 'chrom': 'ChrZ', 'start': 1, 'end': 2},                                  # not on this genome
             {'name': '', 'chrom': 'Chr1', 'start': 1, 'end': 2}, {'name': 'Bad', 'chrom': 'Chr1', 'start': 'x', 'end': 2}]


GRANGES = {'Chr01': (0.0, 1e7), 'Chr02': (0.0, 80e6), 'Chr03': (0.0, 2e7)}


def test_genes_are_cleaned_and_suggested_in_genome_order():
    genes = hl.genes_from_rows(GENE_ROWS)
    assert [g['name'] for g in genes] == ['GA1', 'B2', 'Off'] and genes[1]['start'] == 5e6
    sug = hl.suggest_genes(genes, GRANGES)
    assert [(g['name'], g['chrom']) for g in sug] == [('GA1', 'Chr01'), ('B2', 'Chr03')]      # real chromosome names, 'Off' dropped
    assert hl.gene_label_text(sug[0]).startswith('GA1 - Chr01: 1 Mb')


def test_add_by_name_or_interval_with_flank():
    sug = hl.suggest_genes(hl.genes_from_rows(GENE_ROWS), GRANGES)
    rows, problems = hl.add_by_text('ga1, b2; Chr02:3Mb-4Mb=Q; Chr03; nope', sug, GRANGES, flank=100e3)
    assert [r['label'] for r in rows] == ['GA1', 'B2', 'Q', ''] and rows[0]['start'] == '900kb' and rows[0]['end'] == '1.3Mb'
    assert rows[2]['start'] == '3Mb' and rows[3]['chrom'] == 'Chr03' and rows[3]['start'] == ''
    assert len(problems) == 1 and 'nope' in problems[0]
    clipped, _ = hl.add_by_text('GA1', sug, GRANGES, flank=5e6)
    assert clipped[0]['start'] == '0'                                                         # flank never goes below 0


def test_remove_by_name_or_interval():
    rows = [{'chrom': 'Chr01', 'start': '900kb', 'end': '1.3Mb', 'label': 'GA1'},
            {'chrom': 'Chr02', 'start': '3Mb', 'end': '4Mb', 'label': 'Q'},
            {'chrom': 'Chr03', 'start': '', 'end': '', 'label': ''}]
    kept, n, un = hl.remove_by_text('ga1', rows, 'bp', GRANGES)
    assert n == 1 and [r['label'] for r in kept] == ['Q', ''] and un == []
    kept, n, un = hl.remove_by_text('Chr02:3.5Mb-9Mb; Chr03:1Mb-2Mb; nothing', rows, 'bp', GRANGES)   # overlap by interval; whole chromosome row
    assert n == 2 and [r['label'] for r in kept] == ['GA1'] and un == ['nothing']
    kept, n, _ = hl.remove_by_text('Chr01', rows, 'bp', GRANGES)                                       # bare chromosome = everything on it
    assert n == 1 and len(kept) == 2


def test_gene_picklist_and_table_stay_in_step():
    genes = hl.suggest_genes(hl.genes_from_rows(GENE_ROWS), GRANGES)                    # GA1 (Chr01), B2 (Chr03)
    manual = {'chrom': 'Chr02', 'start': '3Mb', 'end': '4Mb', 'label': 'QTL', 'colour': ''}
    rows = hl.sync_gene_rows([manual], genes, genes, 100e3)                             # pick both
    assert [r['label'] for r in rows] == ['QTL', 'GA1', 'B2'] and rows[1]['start'] == '900kb'
    assert hl.sync_gene_rows(rows, genes, genes) == rows                                # picking again changes nothing
    only_b2 = hl.sync_gene_rows(rows, genes, genes[1:], 100e3)                          # unpick GA1: its row goes, the QTL stays
    assert [r['label'] for r in only_b2] == ['QTL', 'B2']
    assert [g['name'] for g in hl.gene_rows_of(only_b2, genes).values()] == ['B2']      # the QTL is not a gene row
    widened = hl.apply_flank(only_b2, genes, 1e6)
    assert widened[1]['start'] == '4Mb' and widened[0] == manual                        # only gene rows are re-widened


def test_chromosome_picklist_and_interval_form():
    rows = [{'chrom': 'Chr02', 'start': '3Mb', 'end': '4Mb', 'label': '', 'colour': ''}]
    rows = hl.sync_chromosome_rows(rows, ['Chr01', 'Chr03'], GRANGES)
    assert hl.whole_chromosomes_of(rows, GRANGES) == ['Chr01', 'Chr03'] and len(rows) == 3
    rows = hl.sync_chromosome_rows(rows, ['Chr03'], GRANGES)                            # unpick Chr01: only that whole-chromosome row goes
    assert hl.whole_chromosomes_of(rows, GRANGES) == ['Chr03'] and rows[0]['start'] == '3Mb'
    row, err = hl.interval_row('Chr01', '12Mb', '', 'QTL1')
    assert err is None and row['start'] == '12Mb' and row['end'] == ''
    assert hl.interval_row('Chr01', '5Mb', '1Mb', '')[1] and hl.interval_row('', '', '', '')[1] and hl.interval_row('Chr01', 'zz', '', '')[1]


def test_text_box_separators():
    assert hl.split_tokens('GENE1,GENE2, GENE3\nA03:1,000-2,000=x; A05') == ['GENE1', 'GENE2', 'GENE3', 'A03:1,000-2,000=x', 'A05']
    kept, n, un = hl.remove_by_text('ga1, Chr02:3.5Mb-9Mb', [{'chrom': 'Chr01', 'label': 'GA1', 'start': '1Mb', 'end': '2Mb'},
                                                              {'chrom': 'Chr02', 'label': 'Q', 'start': '3Mb', 'end': '4Mb'}], 'bp', GRANGES)
    assert n == 2 and kept == [] and un == []


def test_only_relevant_option_defaults_off_and_reads_true_forms():
    assert hl.highlight_options({})['only_relevant'] is False
    assert hl.highlight_options({'highlight_only_relevant': 'yes'})['only_relevant'] is True
    assert hl.relevant_pairs([True, False, False], [False, True, False]).tolist() == [True, True, False]
    assert hl.relevant_pairs([True, True, False], [False, True, False], 'both').tolist() == [False, True, False]


def test_store_restriction_cuts_among_relevant_pairs_only():
    from interaction_reduce import store_from_frame
    df = pd.DataFrame({'population': 'P', 'model': 'RF', 'phenotype': 'y',
                       'marker1': ['a', 'b', 'c', 'd', 'a', 'e'], 'marker2': ['x', 'y', 'z', 'w', 'b', 'a'],
                       'value': [9.0, 8.0, 7.0, 6.0, 1.0, 0.5]})
    full, _ = store_from_frame(df).reduce_groups(select={'mode': 'count', 'top_count': 2})
    assert full[('all', 'RF', 'y')]['marker1'].tolist() == ['a', 'b']                         # without the filter: the two strongest
    spec = {'mask_for': lambda pheno, names: np.isin(names.astype(str), ['c', 'd']), 'match': 'either'}
    got, _ = store_from_frame(df).reduce_groups(select={'mode': 'percentage', 'top_percent': 50}, restrict=spec)
    assert got[('all', 'RF', 'y')]['marker1'].tolist() == ['c']                               # 2 relevant pairs -> top 50% = 1
    both, _ = store_from_frame(df).reduce_groups(select={'mode': 'count', 'top_count': 5},
                                                 restrict=dict(spec, match='both'))
    assert len(both[('all', 'RF', 'y')]) == 0                                                  # no pair has both ends in {c, d}


# ---------------------------------------------------------------------------
# config round trip through the GUI loader (absent keys = off, I11)
# ---------------------------------------------------------------------------
def test_gui_loader_restores_highlight_settings_and_defaults_when_absent():
    regions, _ = hl.config_regions_from_rows([{'chrom': 'Chr2', 'start': '10Mb', 'end': '20Mb', 'label': 'A', 'colour': 'sky blue'},
                                              {'chrom': 'Chr5'}])
    assert regions[0] == {'chrom': 'Chr2', 'start': 10_000_000, 'end': 20_000_000, 'label': 'A', 'colour': '#56B4E9'}
    cfg = {'CIRCOS_CONFIG': {'unit': 'bp', 'highlight_regions': regions, 'highlight_mode': 'only', 'highlight_match': 'both',
                             'highlight_dim_opacity': 0.2, 'highlight_only_relevant': True}}
    state, _ = gcl.engine_config_to_gui_state(cfg, 'sequential')
    assert state['circos_highlight_on'] is True
    assert state['circos_highlight_regions'][0] == {'chrom': 'Chr2', 'start': '10Mb', 'end': '20Mb', 'label': 'A', 'colour': 'sky blue'}
    assert state['circos_highlight_regions'][1]['start'] == ''
    assert state['circos_highlight_mode'] == 'only' and state['circos_highlight_match'] == 'both'
    assert state['circos_highlight_only_relevant'] is True
    assert state['circos_highlight_dim_opacity'] == 0.2 and state['circos_highlight_lw_boost'] == hl.DEFAULT_LW_BOOST
    # the table rows the GUI shows must read back to the same engine regions
    assert hl.config_regions_from_rows(state['circos_highlight_regions'], 'bp')[0] == regions
    old, _ = gcl.engine_config_to_gui_state({'CIRCOS_CONFIG': {'space': 1}}, 'sequential')
    assert old['circos_highlight_on'] is False and old['circos_highlight_regions'] == [] and old['circos_highlight_only_relevant'] is False


# ---------------------------------------------------------------------------
# rendering (needs pyCirclize)
# ---------------------------------------------------------------------------
pytest.importorskip('pycirclize')
import circos_plot as cp  # noqa: E402


def _dataset(tmp_path, n_chrom=4, per_chrom=40, seed=0):
    rng = np.random.default_rng(seed)
    chroms = [f'Chr{i}' for i in range(1, n_chrom + 1)]
    length = 40_000_000
    rows = []
    for c in chroms:
        for k, p in enumerate(np.sort(rng.integers(1, length, per_chrom))):
            rows.append((c, f'{c}_{k:03d}', int(p), int(p) + 1))
    marker = pd.DataFrame(rows, columns=['chromosome', 'name', 'start', 'end'])
    marker.to_csv(tmp_path / 'marker_info.csv', index=False)
    pd.DataFrame([(c, 0, length, pop) for c in chroms for pop in ('all', 'P1')],
                 columns=['chromosome', 'start', 'end', 'population']).to_csv(tmp_path / 'chrom.csv', index=False)
    names = marker['name'].tolist()
    effect = pd.DataFrame([['P1', 'yield', 'RF', 0.8, s] + list(rng.normal(0, 1, len(names)) * rng.gamma(1, 1, len(names)))
                           for s in (1, 2)], columns=['population', 'phenotype', 'model', 'ratio', 'sample'] + names)
    in_region = marker[(marker.chromosome == 'Chr2') & (marker.start > 10e6) & (marker.start < 20e6)]['name'].to_numpy()
    pairs = [tuple(rng.choice(names, 2, replace=False)) + (float(rng.gamma(1, 1)),) for _ in range(200)]
    pairs += [(rng.choice(in_region), rng.choice(names), float(rng.gamma(4, 2))) for _ in range(12)]
    inter = pd.DataFrame([['P1', 'RF', 'yield', a, b, v] for a, b, v in pairs],
                         columns=['population', 'model', 'phenotype', 'marker1', 'marker2', 'value'])
    cmap = {f'{h}{i}': matplotlib.colors.to_hex((1 - 0.8 * i / 9, 1 - 0.7 * i / 9, 1 - 0.2 * i / 9)) for h in ('blue', 'red') for i in range(10)}
    return effect, inter, cmap


def _config(**extra):
    cfg = {'space': 1.5, 'start': 15, 'end': 345, 'link_alpha_min': 0.15, 'interaction_top': 10.0,
           'interaction_top_mode': 'percentage', 'interaction_top_count': 75, 'label_size': 6, 'scale': 10_000_000,
           'unit': 'bp', 'ring_label_size': 8.0, 'ring_label_fit': 'shrink_then_truncate', 'ring_layout': 'fit',
           'figsize': None, 'seam_gap_max': 40.0, 'ring_label_max_chars': 0, 'link_sort': 'ascending',
           'link_scale': 'rank', 'link_lw_range': [0.6, 2.4], 'link_colour_mode': 'chrom_cvd'}
    cfg.update(extra)
    return cfg


@pytest.fixture()
def rendered(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    effect, inter, cmap = _dataset(tmp_path)

    def run(name, **extra):
        cp.circos_plot(effect.copy(), inter.copy(), 'marker_info.csv', 'chrom.csv', None, ['P1'], ['yield'], _config(**extra),
                       0, 0, cmap, name, pd.DataFrame(), 'within', None, plot_dpi=60)
        return tmp_path / 'Result' / name
    return run


REGION = [{'chrom': 'Chr2', 'start': 10_000_000, 'end': 20_000_000, 'label': 'QTL-A'}]


def test_no_regions_means_the_plain_file_names_and_no_extras(rendered):
    out = rendered('plain')
    assert (out / 'circos_yield_P1_interaction_RF.png').exists()
    assert not list(out.glob('*highlight*'))


def test_highlight_writes_suffixed_png_links_table_and_legend_rows(rendered):
    out = rendered('hl', highlight_regions=REGION)
    assert (out / 'circos_yield_P1_interaction_RF_highlight.png').exists()
    assert not (out / 'circos_yield_P1_interaction_RF.png').exists()
    links = pd.read_csv(out / 'circos_yield_P1_interaction_RF_highlight_links.tsv', sep='\t')
    assert len(links) >= 5 and set(links['region']) == {'QTL-A'}
    in_a = ((links['chromosome_1'] == 'Chr2') & links['start_1'].between(10e6, 20e6)) | \
           ((links['chromosome_2'] == 'Chr2') & links['start_2'].between(10e6, 20e6))
    assert in_a.all()                                                                  # every listed link really touches the region
    assert list(links['value']) == sorted(links['value'], reverse=True)
    handles, _ = cp._build_circos_legend_handles({}, None, set(), False, False,
                                                 highlight_regions={0: {'colour': '#E69F00', 'legend': 'QTL-A  (Chr2: 10 Mb – 20 Mb)'}},
                                                 highlight_intra=True, highlight_inter=True, highlight_dimmed=True)
    assert [h.get_label() for h in handles][0].startswith('Highlighted: QTL-A') and len(handles) == 4


def test_only_mode_keeps_just_the_highlighted_links_in_the_same_set(rendered):
    emphasise = pd.read_csv(rendered('e', highlight_regions=REGION) / 'circos_yield_P1_interaction_RF_highlight_links.tsv', sep='\t')
    only = pd.read_csv(rendered('o', highlight_regions=REGION, highlight_mode='only') / 'circos_yield_P1_interaction_RF_highlight_links.tsv', sep='\t')
    assert len(only) == len(emphasise)                                                 # the same links qualify either way
    assert (only['value'].to_numpy() > 0).all()


def test_unplaceable_region_falls_back_to_a_plain_plot(rendered, capsys):
    out = rendered('bad', highlight_regions=[{'chrom': 'ChrNope'}, {'chrom': 'Chr1', 'start': 900e6, 'end': 950e6}])
    assert (out / 'circos_yield_P1_interaction_RF.png').exists() and not list(out.glob('*highlight*'))
    assert 'could be placed' in capsys.readouterr().out


def test_prepare_links_orders_highlighted_last_and_filters_in_only_mode():
    res = _res({'chrom': 'Chr01', 'start': '10Mb', 'end': '20Mb'})
    df = pd.DataFrame({'a': ['Chr01', 'Chr02', 'Chr01', 'ChrX'], 'b': [15e6, 1e6, 50e6, 1e6], 'c': [15e6 + 1, 1e6 + 1, 50e6 + 1, 1e6 + 1],
                       'd': ['ChrX', 'ChrX', 'Chr02', 'Chr02'], 'e': [1.0, 1.0, 1.0, 1.0], 'f': [2.0, 2.0, 2.0, 2.0],
                       'value': [0.4, 0.3, 0.2, 0.1]})
    df.columns = ['chromosome_marker1', 'start', 'end', 'chromosome_marker2', 'start', 'end', 'value']
    cfg = {'link_alpha_min': 0.15, 'link_sort': 'none'}
    opts = hl.highlight_options({})
    out, lws, s, style = cp._prepare_links_with_highlight(df, cfg, 1.5, res, opts)
    assert int(style['hit'].sum()) == 1                                                 # only the Chr01 15 Mb end is inside the region
    assert style['hit'].tolist() == sorted(style['hit'].tolist())                       # all False first, then all True
    assert len(out) == 4 and len(lws) == 4 and len(s) == 4
    only, lws2, _, style2 = cp._prepare_links_with_highlight(df, cfg, 1.5, res, dict(opts, mode='only'))
    assert len(only) == int(style['hit'].sum()) == len(lws2) and style2['hit'].all() and style2['n_total'] == 4


def test_only_relevant_takes_the_top_percentage_over_the_relevant_links(rendered, tmp_path):
    plain = pd.read_csv(rendered('p', highlight_regions=REGION) / 'circos_yield_P1_interaction_RF_highlight_links.tsv', sep='\t')
    out = rendered('r', highlight_regions=REGION, highlight_only_relevant=True)
    links = pd.read_csv(out / 'circos_yield_P1_interaction_RF_highlight_links.tsv', sep='\t')
    # how many pairs of the data touch the region at all?
    effect, inter, _ = _dataset(tmp_path)
    marker = pd.read_csv(tmp_path / 'marker_info.csv').set_index('name')
    inside = {n for n, r in marker.iterrows() if r.chromosome == 'Chr2' and 10e6 <= r.start <= 20e6}
    n_rel = int(inter.apply(lambda r: r.marker1 in inside or r.marker2 in inside, axis=1).sum())
    assert len(links) == int(round(n_rel * 0.10)) > 0                                         # top 10% OF THE RELEVANT links
    assert len(links) < len(plain)                                                            # the unrestricted cut found more of them (it spends its quota on the whole table)
    assert set(links['region']) == {'QTL-A'}


def test_only_relevant_without_a_placeable_region_keeps_the_normal_selection(rendered, capsys):
    out = rendered('n', highlight_regions=[{'chrom': 'ChrNope'}], highlight_only_relevant=True)
    assert (out / 'circos_yield_P1_interaction_RF.png').exists()
    assert 'no region could be placed' in capsys.readouterr().out
