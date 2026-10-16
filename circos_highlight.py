"""circos_highlight.py - highlight circos interactions that come from chosen genome regions.

Pure helpers (NumPy + matplotlib colour parsing only; no pyCirclize, no Streamlit, no pandas)
shared by `circos_plot.py` (the renderer), `main_app.py` (the GUI table, validation and the
linear preview) and `gui_config_loader.py`. Keeping them here follows the same discipline as
`circos_geometry.py`: one authority, imported downward, trivially unit-testable.

What "highlight" means
----------------------
The user names one or more genome regions: a chromosome, or a start-end window on one. Every
link (k = 2 chord) or hub glyph (k >= 3 hyperedge) with an end inside a region is
"highlighted"; the rest are "other" links. `circos_plot.py` then, per `highlight_mode`:

  emphasise  (default)  other links are faded (grey, low opacity), highlighted links are drawn on
                        top, thicker, in their region's colour.
  only                  other links are not drawn at all.

and, independent of the mode, marks each region on the plot with a coloured band where its links
land, a faint wedge through the link area, and a short tag; the standalone legend lists them.

Config contract (`CIRCOS_CONFIG`; every key optional, absent = no highlighting at all, so an
older config renders exactly as before - I11):

  highlight_regions      list of {'chrom': str, 'start': pos|None, 'end': pos|None,
                         'label': str|None, 'colour': name|hex|None}. start/end None = open end
                         (both None = the whole chromosome). Positions are numbers in the plot's
                         own coordinate unit, or - for 'bp' - strings such as '12.5Mb' / '300kb'.
  highlight_mode         'emphasise' | 'only'                       (default 'emphasise')
  highlight_match        'either' | 'both'                          (default 'either')
                         either: an end of the link is in a region; both: both ends are in
                         highlighted regions (links *among* the highlighted regions).
  highlight_colour_by    'region' | 'single'                        (default 'region')
  highlight_colour       accent colour used when colour_by='single' (default orange)
  highlight_dim          'grey' | 'original'                        (default 'grey')
  highlight_dim_opacity  0.02..1, opacity multiplier of other links (default 0.35)
  highlight_lw_boost     1..6, width multiplier of highlighted links (default 1.5)
  highlight_band / highlight_shade / highlight_label   bool, all default True
  highlight_only_relevant  bool, default False. True: links that do not touch a highlighted
                         region are left out BEFORE the 'Top interactions' cut, so the top
                         percentage / count is taken over the relevant links only (with False the
                         cut is taken over every link and the highlight can only pick among the
                         survivors). The 'highlight_match' rule decides what "touches" means.
"""
from __future__ import annotations

import math
import re

import numpy as np

HIGHLIGHT_MODES = ('emphasise', 'only')
HIGHLIGHT_MATCHES = ('either', 'both')
HIGHLIGHT_COLOUR_BY = ('region', 'single')
HIGHLIGHT_DIM = ('grey', 'original')

# Named colours offered in the GUI table. Okabe-Ito (colour-blind safe), chosen so that none is
# close to the default 'other link' grey and the first few are far from each other; 'vermillion'
# is last because it is also the within-chromosome link colour of the CVD-safe link mode.
COLOUR_CHOICES = {
    'auto': None,
    'orange': '#E69F00',
    'bluish green': '#009E73',
    'sky blue': '#56B4E9',
    'reddish purple': '#CC79A7',
    'blue': '#0072B2',
    'vermillion': '#D55E00',
    'yellow': '#F0E442',
    'black': '#000000',
    'grey': '#7F7F7F',
}
PALETTE = tuple(COLOUR_CHOICES[n] for n in
                ('orange', 'bluish green', 'sky blue', 'reddish purple', 'blue', 'vermillion'))
DEFAULT_ACCENT = PALETTE[0]
DIM_GREY = '#8C8C8C'
NEUTRAL_MARK = '#3A3A3A'          # region marks of the 'combined' layout, where colour already means k
HIGHLIGHT_DASH = (0, (2.6, 1.5))      # tighter than 'dashed': highlighted links are drawn thick

DEFAULT_DIM_OPACITY = 0.35
DEFAULT_LW_BOOST = 1.5
HIGHLIGHT_ALPHA_FLOOR = 0.55      # the faintest highlighted link is still this opaque
DIM_LW_FACTOR = 0.75
MIN_BAND_DEG = 0.8                # a region narrower than this is widened on the page (not in the data)
TAG_MAX_CHARS = 8

_BLANK_STRINGS = ('', 'nan', 'none', 'na', 'null', '-')


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
def _blank(v):
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    try:                                   # pandas NA / NaT without importing pandas
        if v != v:                         # noqa: PLR0124 - NaN-like
            return True
    except Exception:
        pass
    return isinstance(v, str) and v.strip().lower() in _BLANK_STRINGS


is_blank = _blank          # public: the GUI table normaliser uses the same notion of 'empty cell'


_SUFFIX = {'': 1.0, 'bp': 1.0, 'b': 1.0, 'k': 1e3, 'kb': 1e3, 'm': 1e6, 'mb': 1e6, 'g': 1e9, 'gb': 1e9}
_POS_RE = re.compile(r'^([+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?)\s*([a-z]*)$')


def parse_position(value, unit='bp'):
    """A genomic position as a float, or None when blank. Accepts numbers and, for unit 'bp',
    strings with a size suffix ('12.5Mb', '300 kb', '1,200,000', '2e6'); for 'cM' a plain number
    or one ending in 'cM'. Raises ValueError on anything else or on a negative value."""
    if _blank(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        v = float(value)
    else:
        s = str(value).strip().lower().replace(',', '').replace('_', '')
        m = _POS_RE.match(s)
        if not m:
            raise ValueError(f"cannot read {value!r} as a position")
        num, suffix = m.groups()
        if str(unit) == 'cM':
            if suffix not in ('', 'cm'):
                raise ValueError(f"{value!r}: use a plain number for a cM position")
            mult = 1.0
        else:
            if suffix not in _SUFFIX:
                raise ValueError(f"{value!r}: unknown unit {suffix!r} (use bp, kb, Mb or Gb)")
            mult = _SUFFIX[suffix]
        v = float(num) * mult
    if v < 0 or math.isinf(v) or math.isnan(v):
        raise ValueError(f"{value!r} is not a valid position")
    return v


def normalise_chrom_key(name):
    """Case-insensitive chromosome key that ignores a 'chr'/'chromosome' prefix and leading zeros
    ('Chr01', 'chr1', '1' -> '1'; 'A01' -> 'a01'), so a typed '3' finds a sector called 'Chr03'."""
    s = str(name).strip().lower()
    for prefix in ('chromosome', 'chrom', 'chr'):
        if s.startswith(prefix) and len(s) > len(prefix):
            s = s[len(prefix):].lstrip('_- ')
            break
    return str(int(s)) if s.isdigit() else s


def parse_region_spec(text):
    """Parse a pasted/typed region list into raw row dicts (positions are left as written; run
    `clean_regions` to validate them). Items are separated by ';' or new lines:

        chrom[:start-end][=label]        e.g.   A03:12Mb-15.5Mb=QTL1 ; A05 ; D02:3000000-4000000

    Either end of the range may be left empty (open). Colours are not part of the text format."""
    rows = []
    for item in re.split(r'[;\n]+', str(text or '')):
        item = item.strip()
        if not item:
            continue
        label = None
        if '=' in item:
            item, label = item.split('=', 1)
            label = label.strip() or None
        start = end = None
        if ':' in item:
            item, rng = item.split(':', 1)
            parts = re.split(r'\s*(?:-|–|\.\.)\s*', rng.strip(), maxsplit=1)
            start = parts[0].strip() or None
            end = (parts[1].strip() or None) if len(parts) > 1 else start
        rows.append({'chrom': item.strip(), 'start': start, 'end': end, 'label': label, 'colour': None})
    return rows


def _colour_to_hex(value):
    """A colour name from COLOUR_CHOICES, a hex string or any matplotlib colour -> '#rrggbb'
    (None for blank/'auto'). Raises ValueError for an unknown colour."""
    if _blank(value) or str(value).strip().lower() == 'auto':
        return None
    s = str(value).strip()
    if s.lower() in COLOUR_CHOICES:
        return COLOUR_CHOICES[s.lower()]
    import matplotlib.colors as mcolors
    if not mcolors.is_color_like(s):
        raise ValueError(f"unknown colour {value!r}")
    return mcolors.to_hex(s)


def colour_label(hex_or_name):
    """The GUI name for a stored colour (reverse of COLOUR_CHOICES), else the value itself."""
    if _blank(hex_or_name):
        return 'auto'
    s = str(hex_or_name).strip()
    for name, hx in COLOUR_CHOICES.items():
        if hx is not None and hx.lower() == s.lower():
            return name
    return s if s.lower() in COLOUR_CHOICES else s


def clean_regions(raw, unit='bp'):
    """`(regions, problems)` from raw rows (GUI table / JSON config / `parse_region_spec`).

    A completely blank row (an empty table line) is skipped silently; any other malformed row is
    reported in `problems` (human-readable, 1-based row numbers) and left out. Each region is
    `{'chrom': str, 'start': float|None, 'end': float|None, 'label': str, 'colour': '#hex'|None}`.
    """
    regions, problems = [], []
    for i, row in enumerate(raw or []):
        if row is None:
            continue
        try:
            row = dict(row)
        except (TypeError, ValueError):
            problems.append(f"Row {i + 1}: not a region")
            continue
        chrom_raw = row.get('chrom', row.get('chromosome'))
        chrom = '' if _blank(chrom_raw) else str(chrom_raw).strip()
        s_raw, e_raw = row.get('start'), row.get('end')
        label = '' if _blank(row.get('label')) else str(row.get('label')).strip()
        colour_raw = row.get('colour')
        if not chrom and _blank(s_raw) and _blank(e_raw) and not label and (_blank(colour_raw)
                                                                           or str(colour_raw).strip().lower() == 'auto'):
            continue
        if not chrom:
            problems.append(f"Row {i + 1}: choose a chromosome")
            continue
        try:
            start, end = parse_position(s_raw, unit), parse_position(e_raw, unit)
        except ValueError as exc:
            problems.append(f"Row {i + 1} ({chrom}): {exc}")
            continue
        if start is not None and end is not None and start > end:
            problems.append(f"Row {i + 1} ({chrom}): start is after end")
            continue
        try:
            colour = _colour_to_hex(colour_raw)
        except ValueError as exc:
            problems.append(f"Row {i + 1} ({chrom}): {exc}")
            continue
        regions.append({'chrom': chrom, 'start': start, 'end': end, 'label': label, 'colour': colour})
    return regions, problems


def _flag(v, default):
    if v is None:
        return default
    if isinstance(v, str):
        return v.strip().lower() in ('1', 'true', 'yes', 'on')
    return bool(v)


def highlight_options(circos_config):
    """Validated highlight settings from `CIRCOS_CONFIG`. `['regions']` is empty - meaning "no
    highlighting, render exactly as before" - when the key is absent or holds no usable region;
    `['problems']` lists what was dropped. Unknown values of any other key fall back to the
    documented default."""
    c = circos_config or {}
    unit = str(c.get('unit', 'bp') or 'bp')
    regions, problems = clean_regions(c.get('highlight_regions'), unit)

    def pick(key, allowed, default):
        v = c.get(key)
        v = default if v is None else str(v).strip().lower()
        return v if v in allowed else default

    def num(key, default, lo, hi):
        try:
            v = float(c.get(key, default))
        except (TypeError, ValueError):
            return default
        return min(max(v, lo), hi) if not math.isnan(v) else default

    try:
        accent = _colour_to_hex(c.get('highlight_colour')) or DEFAULT_ACCENT
    except ValueError:
        accent = DEFAULT_ACCENT
    return {
        'regions': regions,
        'problems': problems,
        'unit': unit,
        'mode': pick('highlight_mode', HIGHLIGHT_MODES, 'emphasise'),
        'match': pick('highlight_match', HIGHLIGHT_MATCHES, 'either'),
        'colour_by': pick('highlight_colour_by', HIGHLIGHT_COLOUR_BY, 'region'),
        'colour': accent,
        'dim': pick('highlight_dim', HIGHLIGHT_DIM, 'grey'),
        'dim_opacity': num('highlight_dim_opacity', DEFAULT_DIM_OPACITY, 0.02, 1.0),
        'lw_boost': num('highlight_lw_boost', DEFAULT_LW_BOOST, 1.0, 6.0),
        'band': _flag(c.get('highlight_band'), True),
        'shade': _flag(c.get('highlight_shade'), True),
        'label': _flag(c.get('highlight_label'), True),
        'only_relevant': _flag(c.get('highlight_only_relevant'), False),
    }


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------
def format_position(x, unit='bp'):
    """Compact human position: 12500000 -> '12.5 Mb' (bp), 42 -> '42 cM'."""
    x = float(x)
    if str(unit) == 'cM':
        return f"{x:.4g} cM"
    for size, name in ((1e9, 'Gb'), (1e6, 'Mb'), (1e3, 'kb')):
        if abs(x) >= size:
            return f"{x / size:.4g} {name}"
    return f"{x:.0f} bp"


def _tag_of(label, idx):
    label = (label or '').strip()
    if not label:
        return f"R{idx + 1}"
    return label if len(label) <= TAG_MAX_CHARS else label[:TAG_MAX_CHARS - 1].rstrip() + '…'


def position_to_text(value, unit='bp'):
    """The editable text for a stored position: None -> '', 12500000 -> '12.5Mb', 300000 -> '300kb',
    1234567 -> '1234567' (exact round trip through `parse_position`), cM -> '42.5'."""
    if _blank(value):
        return ''
    v = float(value)
    if str(unit) == 'cM':
        return f"{v:.12g}"
    for size, name in ((1e9, 'Gb'), (1e6, 'Mb'), (1e3, 'kb')):
        if v >= size and abs(v / size - round(v / size, 6)) < 1e-9 and len(f"{v / size:g}") <= 8:
            text = f"{v / size:g}{name}"
            if abs(parse_position(text, unit) - v) < 1e-6:
                return text
    return f"{v:.0f}" if abs(v - round(v)) < 1e-9 else f"{v:.12g}"


def table_rows_from_regions(regions, unit='bp'):
    """GUI table rows (all text, colour as its display name) from config-style regions."""
    rows = []
    for r in regions or []:
        rows.append({'chrom': str(r.get('chrom', '')), 'start': position_to_text(r.get('start'), unit),
                     'end': position_to_text(r.get('end'), unit),
                     'label': '' if _blank(r.get('label')) else str(r.get('label')),
                     'colour': colour_label(r.get('colour'))})
    return rows


def config_regions_from_rows(rows, unit='bp'):
    """`(regions, problems)` for `CIRCOS_CONFIG['highlight_regions']` from GUI table rows: the
    cleaned regions in JSON-safe form (plain numbers, hex colours)."""
    cleaned, problems = clean_regions(rows, unit)
    out = []
    for r in cleaned:
        out.append({'chrom': r['chrom'],
                    'start': None if r['start'] is None else (int(r['start']) if float(r['start']).is_integer() else r['start']),
                    'end': None if r['end'] is None else (int(r['end']) if float(r['end']).is_integer() else r['end']),
                    'label': r['label'] or None, 'colour': r['colour']})
    return out, problems


# ---------------------------------------------------------------------------
# resolving regions against the chromosomes of one plot / one chromosome file
# ---------------------------------------------------------------------------
def resolve_regions(regions, chrom_ranges, opts=None, unit='bp'):
    """`(resolved, problems)`: `regions` (from `clean_regions`) matched to the real chromosome
    names in `chrom_ranges` ({name: (start, end)}, in plot order) and clamped to them.

    Each resolved region: `idx` (0-based position in the user's list - colours key off it, so they
    do not shuffle when another row is invalid), `chrom` (the real name), `start`, `end`, `whole`,
    `colour`, `label`, `tag` (short text for the plot), `desc` (e.g. 'A03: 12 Mb - 15.5 Mb') and
    `legend` (the legend row text). A region naming an unknown chromosome, or lying wholly
    outside it, is dropped with a message in `problems`."""
    opts = opts or {}
    unit = opts.get('unit', unit)
    single = opts.get('colour_by') == 'single'
    accent = opts.get('colour') or DEFAULT_ACCENT
    names = [str(n) for n in chrom_ranges]
    by_key = {}
    for n in names:
        by_key.setdefault(normalise_chrom_key(n), n)
    resolved, problems = [], []
    for i, reg in enumerate(regions or []):
        raw = str(reg['chrom'])
        name = raw if raw in chrom_ranges else by_key.get(normalise_chrom_key(raw))
        if name is None:
            shown = ', '.join(names[:8]) + (' ...' if len(names) > 8 else '')
            problems.append(f"Region {i + 1}: chromosome {raw!r} is not in the chromosome file (found: {shown})")
            continue
        lo, hi = chrom_ranges[name]
        start = lo if reg.get('start') is None else max(float(reg['start']), float(lo))
        end = hi if reg.get('end') is None else min(float(reg['end']), float(hi))
        if start > end:
            problems.append(f"Region {i + 1}: {raw} {format_position(reg.get('start') or 0, unit)}"
                            f"-{format_position(reg.get('end') or 0, unit)} lies outside the chromosome "
                            f"({format_position(lo, unit)}-{format_position(hi, unit)})")
            continue
        whole = reg.get('start') is None and reg.get('end') is None
        colour = accent if single else (reg.get('colour') or PALETTE[i % len(PALETTE)])
        label = reg.get('label') or ''
        tag = _tag_of(label, i)
        desc = f"{name} (whole chromosome)" if whole else f"{name}: {format_position(start, unit)} – {format_position(end, unit)}"
        resolved.append({
            'idx': i, 'chrom': name, 'start': start, 'end': end, 'whole': whole, 'colour': colour,
            'label': label, 'tag': tag, 'desc': desc,
            'legend': f"{label}  ({desc})" if label else f"{tag}  {desc}",
        })
    return resolved, problems


# ---------------------------------------------------------------------------
# which links / hyperedges are highlighted
# ---------------------------------------------------------------------------
def _end_hits(chrom, start, end, resolved):
    """Region position (index into `resolved`, lowest wins) each interval touches, else -1."""
    chrom = np.asarray(chrom).astype(str)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    out = np.full(chrom.shape[0], -1, dtype=int)
    for pos in range(len(resolved) - 1, -1, -1):
        r = resolved[pos]
        m = (chrom == r['chrom']) & (end >= r['start']) & (start <= r['end'])
        out[m] = pos
    return out


def classify_pairs(chrom1, start1, end1, chrom2, start2, end2, resolved, match='either'):
    """`(hit, region)` for k = 2 links: `hit` bool array, `region` int array (index into
    `resolved` of the first-listed region the link touches; -1 when not a hit)."""
    h1 = _end_hits(chrom1, start1, end1, resolved)
    h2 = _end_hits(chrom2, start2, end2, resolved)
    if match == 'both':
        hit = (h1 >= 0) & (h2 >= 0)
        region = np.minimum(h1, h2)
    else:
        hit = (h1 >= 0) | (h2 >= 0)
        region = np.where(h1 < 0, h2, np.where(h2 < 0, h1, np.minimum(h1, h2)))
    return hit, np.where(hit, region, -1)


def classify_members(members_list, resolved, match='either'):
    """Same for k >= 3 hyperedges; each element of `members_list` is a list of
    `(chromosome, position)`. 'either': any member in a region; 'both': every member is."""
    n = len(members_list)
    hit = np.zeros(n, dtype=bool)
    region = np.full(n, -1, dtype=int)
    for i, members in enumerate(members_list):
        if not members:
            continue
        pos = np.array([float(p) for _, p in members])
        hits = _end_hits([c for c, _ in members], pos, pos, resolved)
        ok = bool((hits >= 0).all()) if match == 'both' else bool((hits >= 0).any())
        if ok:
            hit[i] = True
            region[i] = int(hits[hits >= 0].min())
    return hit, region


# ---------------------------------------------------------------------------
# link styling
# ---------------------------------------------------------------------------
def draw_order(hit):
    """Stable order that draws every non-highlighted link first, so highlighted ones sit on top."""
    return np.argsort(np.asarray(hit, dtype=int), kind='stable')


def style_alpha_lw(alpha, lw, hit, opts, alpha_min):
    """`(alpha, lw)` arrays after highlighting. Highlighted links keep their strength ranking but
    are lifted into [HIGHLIGHT_ALPHA_FLOOR, 1] and thickened by `lw_boost`; the others are faded
    by `dim_opacity` (never fully invisible) and slightly thinned."""
    alpha = np.asarray(alpha, dtype=float)
    lw = np.asarray(lw, dtype=float)
    hit = np.asarray(hit, dtype=bool)
    a_min = float(alpha_min)
    if a_min >= 1.0:
        up = np.ones_like(alpha)
    else:
        up = HIGHLIGHT_ALPHA_FLOOR + np.clip((alpha - a_min) / (1.0 - a_min), 0.0, 1.0) * (1.0 - HIGHLIGHT_ALPHA_FLOOR)
    down = np.clip(alpha * opts['dim_opacity'], 0.04, 1.0)
    a_out = np.where(hit, up, down)
    lw_out = np.where(hit, lw * opts['lw_boost'], np.maximum(lw * DIM_LW_FACTOR, 0.25))
    return a_out, lw_out


# ---------------------------------------------------------------------------
# GUI preview (linear genome strip)
# ---------------------------------------------------------------------------
def preview_figure(chrom_ranges, resolved, unit='bp', width_in=7.4):
    """A small matplotlib Figure: every chromosome as a bar (length to scale) with the
    highlighted regions drawn in their colours and tagged. Built on `Figure` (not pyplot), so it
    leaves no global state behind in a long-lived Streamlit process."""
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle
    names = [str(n) for n in chrom_ranges]
    spans = [max(float(chrom_ranges[n][1]) - float(chrom_ranges[n][0]), 1.0) for n in chrom_ranges]
    total = sum(spans)
    gap = 0.012 * total
    offsets, x = {}, 0.0
    for n, sp in zip(names, spans):
        offsets[n] = x
        x += sp + gap
    xmax = max(x - gap, 1.0)
    rot = 90 if len(names) > 10 else 0
    fig = Figure(figsize=(width_in, 1.55 if rot else 1.35), dpi=110)
    ax = fig.add_axes([0.01, 0.30 if rot else 0.22, 0.98, 0.50])
    for n, sp in zip(names, spans):
        ax.add_patch(Rectangle((offsets[n], 0), sp, 1, facecolor='#e4e4e4', edgecolor='#555555', linewidth=0.5))
        ax.text(offsets[n] + sp / 2, -0.14, n, ha='center', va='top', fontsize=5.5 if rot else 7,
                rotation=rot, color='#222222')
    min_w = 0.004 * xmax
    for r in resolved:
        lo = float(chrom_ranges[r['chrom']][0])
        x0 = offsets[r['chrom']] + (r['start'] - lo)
        w = max(r['end'] - r['start'], min_w)
        ax.add_patch(Rectangle((x0, -0.10), w, 1.20, facecolor=r['colour'], edgecolor='black',
                               linewidth=0.5, alpha=0.95, zorder=3))
        ax.text(x0 + w / 2, 1.20, r['tag'], ha='center', va='bottom', fontsize=7, fontweight='bold', zorder=4)
    ax.set_xlim(-0.005 * xmax, 1.005 * xmax)
    ax.set_ylim(-0.12, 1.75)
    ax.axis('off')
    return fig


# ---------------------------------------------------------------------------
# restricting the 'Top interactions' selection to relevant links
# ---------------------------------------------------------------------------
def marker_membership(chrom, start, end, resolved):
    """Boolean array: does the interval `chrom:start-end` of each marker touch a resolved region?
    (The same overlap rule `classify_pairs` uses for a link end.)"""
    return _end_hits(chrom, start, end, resolved) >= 0


def relevant_pairs(in1, in2, match='either'):
    """Boolean mask of the links to keep from per-end membership flags: 'either' = at least one
    end is in a region, 'both' = both are."""
    in1 = np.asarray(in1, dtype=bool)
    in2 = np.asarray(in2, dtype=bool)
    return (in1 & in2) if match == 'both' else (in1 | in2)


# ---------------------------------------------------------------------------
# key genes: candidate regions from a gene table, and add / remove by name or interval
# ---------------------------------------------------------------------------
# items are separated by ';', new lines or tabs, and by a comma that is followed by a space or does
# not look like a thousands separator (so 'GENE1,GENE2' and 'A:1,000,000-2,000,000' both read right)
_TOKEN_SPLIT = re.compile(r'[;\n\r\t]+|,\s+|,(?!\d{3}(?:\D|$))')
_WORD_SPLIT = re.compile(r'[,\s]+')


def split_tokens(text):
    """Items of a free-text box: separated by ';' or new lines; an item without ':' (a gene name or a
    chromosome) may also be separated by commas / spaces. A ':' item is an interval and kept whole
    (its commas can be thousands separators)."""
    out = []
    for part in _TOKEN_SPLIT.split(str(text or '')):
        part = part.strip()
        if not part:
            continue
        if ':' in part:
            out.append(part)
        else:
            out += [w for w in _WORD_SPLIT.split(part) if w]
    return out


def genes_from_rows(rows):
    """Clean gene records from an iterable of dicts with `chrom`/`start`/`end`/`name` (and optional
    `source`, `phenotype`). Rows without a usable name or position are dropped; repeats of the same
    gene (the gene file lists a gene once per phenotype / population) are merged, keeping the
    first. Order of first appearance is kept."""
    seen, out = set(), []
    for r in rows or []:
        name = r.get('name')
        if _blank(name):
            continue
        try:
            s, e = float(r.get('start')), float(r.get('end'))
        except (TypeError, ValueError):
            continue
        if math.isnan(s) or math.isnan(e) or _blank(r.get('chrom')):
            continue
        if s > e:
            s, e = e, s
        key = (str(name).strip().lower(), str(r['chrom']).strip(), round(s, 3), round(e, 3))
        if key in seen:
            continue
        seen.add(key)
        out.append({'name': str(name).strip(), 'chrom': str(r['chrom']).strip(), 'start': s, 'end': e,
                    'source': '' if _blank(r.get('source')) else str(r.get('source')).strip(),
                    'phenotype': '' if _blank(r.get('phenotype')) else str(r.get('phenotype')).strip()})
    return out


def suggest_genes(genes, chrom_ranges=None):
    """The candidate list offered to the user: the genes that can actually be placed on this
    plot's chromosomes (when `chrom_ranges` is given), in genome order (chromosome order of the
    plot, then position). Each gets `chrom` set to the chromosome's real name."""
    if not chrom_ranges:
        return _first_per_name(genes)
    order = {}
    by_key = {}
    for i, n in enumerate(chrom_ranges):
        order[str(n)] = i
        by_key.setdefault(normalise_chrom_key(n), str(n))
    out = []
    for g in genes:
        name = g['chrom'] if g['chrom'] in order else by_key.get(normalise_chrom_key(g['chrom']))
        if name is None:
            continue
        lo, hi = chrom_ranges[name]
        if g['end'] < lo or g['start'] > hi:
            continue
        out.append(dict(g, chrom=name))
    out.sort(key=lambda g: (order[g['chrom']], g['start'], g['end'], g['name'].lower()))
    return _first_per_name(out)


def _first_per_name(genes):
    """One entry per (gene name, chromosome): the GUI identifies a picked gene by that pair."""
    seen, out = set(), []
    for g in genes:
        key = (g['name'].lower(), str(g['chrom']))
        if key not in seen:
            seen.add(key)
            out.append(g)
    return out


def gene_label_text(gene, unit='bp'):
    """'Zm00001d012345 - Chr3: 12.5 Mb - 12.6 Mb', the text used to pick a gene in the GUI."""
    return (f"{gene['name']} - {gene['chrom']}: {format_position(gene['start'], unit)} - "
            f"{format_position(gene['end'], unit)}")


def gene_to_row(gene, flank=0.0, unit='bp'):
    """One GUI table row (text cells) for a gene, widened by `flank` on each side (never below 0)."""
    flank = max(float(flank or 0.0), 0.0)
    return {'chrom': gene['chrom'],
            'start': position_to_text(max(gene['start'] - flank, 0.0), unit),
            'end': position_to_text(gene['end'] + flank, unit),
            'label': gene['name'], 'colour': ''}


def add_by_text(text, genes, chrom_ranges=None, flank=0.0, unit='bp'):
    """`(rows, problems)` for a free-text 'add' box. Each item is a gene name (matched
    case-insensitively against `genes`; the gene's own interval, plus `flank`, is used), else an
    interval / chromosome in the region syntax `chrom[:start-end][=label]`. Items that are neither
    are reported in `problems`; a gene name found at several places adds every place."""
    by_name = {}
    for g in genes or []:
        by_name.setdefault(g['name'].lower(), []).append(g)
    rows, problems = [], []
    for tok in split_tokens(text):
        hit = by_name.get(tok.lower())
        if hit:
            rows += [gene_to_row(g, flank, unit) for g in hit]
            continue
        spec = parse_region_spec(tok) if (':' in tok or '=' in tok or _is_chromosome(tok, chrom_ranges)) else []
        if spec:
            rows += [{'chrom': r['chrom'], 'start': r['start'] or '', 'end': r['end'] or '',
                      'label': r['label'] or '', 'colour': ''} for r in spec]
        else:
            problems.append(f"'{tok}' is not a gene in the gene file or a region (use chromosome:start-end)")
    return rows, problems


def _is_chromosome(tok, chrom_ranges):
    if not chrom_ranges:
        return False
    keys = {normalise_chrom_key(n) for n in chrom_ranges}
    return normalise_chrom_key(tok) in keys


def remove_by_text(text, rows, unit='bp', chrom_ranges=None):
    """`(kept_rows, n_removed, unmatched)`: drop table rows named by a free-text box. An item
    matches a row by LABEL (case-insensitive, e.g. a gene name) or - written as `chrom`,
    `chrom:start-end` - by INTERVAL: every row on that chromosome whose interval overlaps it (a
    bare chromosome name matches every row on it). Items that match nothing are returned."""
    rows = list(rows or [])
    drop = set()
    unmatched = []
    for tok in split_tokens(text):
        n0 = len(drop)
        low = tok.lower()
        for i, r in enumerate(rows):
            if not _blank(r.get('label')) and str(r['label']).strip().lower() == low:
                drop.add(i)
        spec = parse_region_spec(tok) if (':' in tok or _is_chromosome(tok, chrom_ranges)
                                         or any(normalise_chrom_key(r.get('chrom')) == normalise_chrom_key(tok)
                                                for r in rows if not _blank(r.get('chrom')))) else []
        for q in spec:
            try:
                qs = parse_position(q['start'], unit)
                qe = parse_position(q['end'], unit)
            except ValueError:
                continue
            qs = -math.inf if qs is None else qs
            qe = math.inf if qe is None else qe
            for i, r in enumerate(rows):
                if _blank(r.get('chrom')) or normalise_chrom_key(r['chrom']) != normalise_chrom_key(q['chrom']):
                    continue
                try:
                    rs = -math.inf if _blank(r.get('start')) else parse_position(r['start'], unit)
                    re_ = math.inf if _blank(r.get('end')) else parse_position(r['end'], unit)
                except ValueError:
                    continue
                if re_ >= qs and rs <= qe:
                    drop.add(i)
        if len(drop) == n0:
            unmatched.append(tok)
    kept = [r for i, r in enumerate(rows) if i not in drop]
    return kept, len(drop), unmatched


# ---------------------------------------------------------------------------
# keeping the region table and the "pick genes / chromosomes" widgets in step
# ---------------------------------------------------------------------------
def _gene_key(name, chrom):
    return (str(name).strip().lower(), normalise_chrom_key(chrom))


def gene_rows_of(rows, genes):
    """`{row index: gene}` for the table rows that ARE a gene from `genes` (the row's label is the
    gene's name and it sits on the gene's chromosome). Other rows (intervals, chromosomes, a QTL
    someone named) are not gene rows."""
    by_key = {_gene_key(g['name'], g['chrom']): g for g in genes}
    out = {}
    for i, r in enumerate(rows or []):
        if _blank(r.get('label')) or _blank(r.get('chrom')):
            continue
        g = by_key.get(_gene_key(r['label'], r['chrom']))
        if g is not None:
            out[i] = g
    return out


def sync_gene_rows(rows, genes, picked, flank=0.0, unit='bp'):
    """The table rows after the gene pick-list changed to `picked`: gene rows whose gene is no
    longer picked are removed, picked genes without a row get one (own interval widened by `flank`)
    appended; every other row, and the order, is untouched."""
    rows = [dict(r) for r in rows or []]
    want = {_gene_key(g['name'], g['chrom']): g for g in picked}
    have = gene_rows_of(rows, genes)
    drop = {i for i, g in have.items() if _gene_key(g['name'], g['chrom']) not in want}
    kept = [r for i, r in enumerate(rows) if i not in drop]
    present = {_gene_key(g['name'], g['chrom']) for i, g in have.items() if i not in drop}
    kept += [gene_to_row(g, flank, unit) for k, g in want.items() if k not in present]
    return kept


def apply_flank(rows, genes, flank, unit='bp'):
    """Recompute the interval of every gene row with a new `flank` (other rows untouched)."""
    rows = [dict(r) for r in rows or []]
    for i, g in gene_rows_of(rows, genes).items():
        new = gene_to_row(g, flank, unit)
        rows[i].update({'chrom': new['chrom'], 'start': new['start'], 'end': new['end']})
    return rows


def whole_chromosomes_of(rows, chrom_ranges=None):
    """Chromosome names (real names when `chrom_ranges` is given) of the whole-chromosome rows."""
    by_key = {normalise_chrom_key(n): str(n) for n in (chrom_ranges or {})}
    out = []
    for r in rows or []:
        if _blank(r.get('chrom')) or not _blank(r.get('start')) or not _blank(r.get('end')):
            continue
        name = by_key.get(normalise_chrom_key(r['chrom']), str(r['chrom']))
        if name not in out:
            out.append(name)
    return out


def sync_chromosome_rows(rows, picked, chrom_ranges=None):
    """Table rows after the 'whole chromosomes' pick-list changed to `picked`: whole-chromosome
    rows of unpicked chromosomes are removed, newly picked ones appended."""
    rows = [dict(r) for r in rows or []]
    want = [str(c) for c in picked or []]
    want_keys = {normalise_chrom_key(c) for c in want}
    kept = [r for r in rows
            if not (not _blank(r.get('chrom')) and _blank(r.get('start')) and _blank(r.get('end'))
                    and normalise_chrom_key(r['chrom']) not in want_keys)]
    have = {normalise_chrom_key(c) for c in whole_chromosomes_of(kept, chrom_ranges)}
    kept += [{'chrom': c, 'start': '', 'end': '', 'label': '', 'colour': ''} for c in want
             if normalise_chrom_key(c) not in have]
    return kept


def interval_row(chrom, start, end, label, unit='bp'):
    """`(row, error)` from the 'add an interval' form fields (text). Start/end may be empty."""
    if _blank(chrom):
        return None, "Choose a chromosome."
    try:
        s, e = parse_position(start, unit), parse_position(end, unit)
    except ValueError as exc:
        return None, str(exc)
    if s is not None and e is not None and s > e:
        return None, f"Start ({start}) is after end ({end})."
    return {'chrom': str(chrom).strip(), 'start': '' if _blank(start) else str(start).strip(),
            'end': '' if _blank(end) else str(end).strip(), 'label': '' if _blank(label) else str(label).strip(),
            'colour': ''}, None
