"""
Preprocess/gene_location_resolver.py
====================================

Zero-token tiers of gene-location curation.

Why this module exists
----------------------
`gene_location_agent.py` used to hand EVERY candidate gene of a network to a
Claude Code session. A Claude Code session pays for its whole system prompt,
every tool definition and every search result on every turn, so most of the
tokens went on gene symbols that a lookup table can answer for free. This module
answers as many genes as possible WITHOUT a model, in this order (cheapest
first), and returns only the remainder for the agent:

  tier 0  prefilter   node names that are obviously not single-locus genes
                      (``X_Pathway``, ``Strigolactones``, ...) are dropped
                      before anything is looked up. Rules are conservative: a
                      name is only dropped when it ends in a pathway/complex
                      word or equals a hormone-class name.
  tier 1  cache       a persistent per-species table of earlier answers
                      (resolved AND explicitly-unresolved), shared by every
                      network and run, so a gene is paid for at most once.
  tier 2  annotation  the species' own GFF3/GTF, streamed once, keeping only
                      the wanted symbols in memory. This is also the only tier
                      that is guaranteed to use the SAME genome build as the
                      user's markers, so it is the preferred source.
  tier 3  ensembl     one batched Ensembl REST call (``POST lookup/symbol``).
                      Optional; never raises (offline = skipped); the assembly
                      name returned by Ensembl is written to ``Source`` and can
                      be required to match ``expected_assembly``.

Every tier obeys the discipline of the rest of the feature: ambiguous symbols
(one symbol, several different loci/IDs), swapped/implausible coordinates and
chromosomes the marker file does not contain are NOT resolved here - they fall
through to the next tier and finally to the agent, which may drop them. Nothing
is ever guessed.

Only 'bp' coordinates can come from tiers 2-3 (annotations are in bp). For
unit='cM' only the cache and the agent are used.
"""

import csv
import gzip
import json
import os
import re
import time
import urllib.error
import urllib.request
import warnings
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

CACHE_COLUMNS = ['species', 'unit', 'gene', 'status', 'chromosome', 'start', 'end',
                 'locus_id', 'source', 'resolver', 'added']

# Longest plausible single gene. Anything larger is almost certainly a region
# or a unit mix-up (a cM value, a chromosome arm) and is not accepted by tiers
# 2-3. The agent tier applies the same bound as a WARNING only.
MAX_PLAUSIBLE_GENE_BP = 5_000_000

_COMMON_TO_ENSEMBL = {
    'maize': 'zea_mays', 'corn': 'zea_mays', 'zea mays': 'zea_mays',
    'arabidopsis': 'arabidopsis_thaliana', 'arabidopsis thaliana': 'arabidopsis_thaliana',
    'rice': 'oryza_sativa', 'oryza sativa': 'oryza_sativa',
    'wheat': 'triticum_aestivum', 'bread wheat': 'triticum_aestivum',
    'barley': 'hordeum_vulgare', 'sorghum': 'sorghum_bicolor',
    'soybean': 'glycine_max', 'soy': 'glycine_max', 'tomato': 'solanum_lycopersicum',
    'potato': 'solanum_tuberosum', 'canola': 'brassica_napus', 'rapeseed': 'brassica_napus',
    'pea': 'pisum_sativum', 'cotton': 'gossypium_raimondii', 'brachypodium': 'brachypodium_distachyon',
    'foxtail millet': 'setaria_italica', 'grapevine': 'vitis_vinifera', 'grape': 'vitis_vinifera',
}

# Hormone / metabolite classes that FLASH-P-style extraction sometimes types as
# "GENE" (see gene_network_prior.py's module docstring). Exact, case-insensitive.
_NON_GENE_EXACT = {
    'auxin', 'auxins', 'cytokinin', 'cytokinins', 'strigolactone', 'strigolactones',
    'gibberellin', 'gibberellins', 'brassinosteroid', 'brassinosteroids', 'ethylene',
    'jasmonate', 'jasmonates', 'jasmonic acid', 'abscisic acid', 'salicylic acid',
    'sucrose', 'sugar', 'sugars', 'light', 'shade', 'temperature', 'nitrogen',
    'phosphate', 'carotenoids', 'flavonoids', 'lignin', 'starch',
}
_NON_GENE_SUFFIX = re.compile(r'(?i)(^|[\s_\-])(pathway|pathways|complex|signalling|signaling|'
                              r'cascade|biosynthesis|metabolism|response)$')


def _norm(symbol):
    """Matching key: case-insensitive, punctuation-insensitive ('PIN-1' == 'pin1')."""
    return re.sub(r'[^0-9a-z]+', '', str(symbol).lower())


def _norm_chrom(value):
    return re.sub(r'(?i)^chr', '', str(value).strip())


def species_to_ensembl(species):
    """'Arabidopsis thaliana' -> 'arabidopsis_thaliana'; common names via a small table."""
    key = re.sub(r'\s+', ' ', str(species).strip().lower())
    if key in _COMMON_TO_ENSEMBL:
        return _COMMON_TO_ENSEMBL[key]
    return re.sub(r'[^a-z0-9]+', '_', key).strip('_')


def marker_chromosomes(marker_info_path):
    """Normalised set of chromosome labels in a marker_info/.bim-as-CSV file
    (first column only, so a 10^6-row file costs one column), or None when the
    file is missing/unreadable (then no chromosome check is applied)."""
    if not marker_info_path or not os.path.isfile(marker_info_path):
        return None
    try:
        import pandas as pd
        col = pd.read_csv(marker_info_path, usecols=[0], dtype=str, encoding='utf-8-sig').iloc[:, 0]
        return {_norm_chrom(v) for v in col.dropna().unique()}
    except Exception as exc:  # never let an optional sanity check break curation
        warnings.warn(f"[gene_location_resolver] could not read chromosomes from "
                      f"'{marker_info_path}' ({type(exc).__name__}: {exc}) - chromosome check skipped.")
        return None


def _plausible(start, end, chromosomes, chrom):
    """Reason string if the locus is NOT acceptable for tiers 2-3, else None."""
    try:
        start, end = float(start), float(end)
    except (TypeError, ValueError):
        return 'non-numeric coordinates'
    if end < start:
        return 'end < start'
    if end - start > MAX_PLAUSIBLE_GENE_BP:
        return f'implausibly long ({int(end - start)} bp)'
    if chromosomes is not None and _norm_chrom(chrom) not in chromosomes:
        return f"chromosome '{chrom}' not present in the marker file"
    return None


# ---------------------------------------------------------------------------
# tier 0: prefilter
# ---------------------------------------------------------------------------

def prefilter_candidates(candidate_ids, hints=None):
    """Split candidates into (to_resolve, skipped) where skipped maps
    id -> reason. Deliberately conservative - see the module docstring."""
    hints = hints or {}
    keep, skipped = [], {}
    for gid in candidate_ids:
        texts = [str(gid)] + ([str(hints[gid])] if hints.get(gid) else [])
        reason = None
        for t in texts:
            if t.strip().lower() in _NON_GENE_EXACT:
                reason = f"'{t}' is a hormone/metabolite/condition class, not a gene"
                break
            if _NON_GENE_SUFFIX.search(t.strip()):
                reason = f"'{t}' names a pathway/complex, not a single-locus gene"
                break
        if reason:
            skipped[gid] = reason
        else:
            keep.append(gid)
    return keep, skipped


# ---------------------------------------------------------------------------
# tier 1: persistent cache
# ---------------------------------------------------------------------------

class GeneLocationCache:
    """CSV-backed cache keyed by (species, unit, gene). Rows with status
    'resolved' carry a locus; rows with status 'unresolved' record that the
    agent explicitly could not verify the gene (so it is not paid for again
    unless the caller passes retry_unresolved=True)."""

    def __init__(self, path):
        self.path = path
        self.rows = {}  # (species_key, unit, gene) -> list of row dicts
        if path and os.path.isfile(path):
            try:
                with open(path, newline='', encoding='utf-8-sig') as fh:
                    for row in csv.DictReader(fh):
                        key = (_norm(row.get('species', '')), row.get('unit', 'bp'), row.get('gene', ''))
                        self.rows.setdefault(key, []).append(row)
            except Exception as exc:
                warnings.warn(f"[gene_location_resolver] cache '{path}' unreadable "
                              f"({type(exc).__name__}: {exc}) - starting with an empty cache.")
                self.rows = {}

    def get(self, species, unit, gene):
        return self.rows.get((_norm(species), unit, gene), [])

    def put(self, species, unit, gene, status, loci=(), resolver=''):
        key = (_norm(species), unit, gene)
        stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        if status == 'resolved':
            self.rows[key] = [{'species': species, 'unit': unit, 'gene': gene, 'status': 'resolved',
                               'chromosome': l['chromosome'], 'start': l['start'], 'end': l['end'],
                               'locus_id': l.get('locus_id', ''), 'source': l.get('source', ''),
                               'resolver': resolver, 'added': stamp} for l in loci]
        else:
            self.rows[key] = [{'species': species, 'unit': unit, 'gene': gene, 'status': 'unresolved',
                               'chromosome': '', 'start': '', 'end': '', 'locus_id': '',
                               'source': loci[0].get('source', '') if loci else '',
                               'resolver': resolver, 'added': stamp}]

    def save(self):
        if not self.path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or '.', exist_ok=True)
        tmp = self.path + '.tmp'
        with open(tmp, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=CACHE_COLUMNS)
            writer.writeheader()
            for rows in self.rows.values():
                writer.writerows(rows)
        os.replace(tmp, self.path)  # atomic: a crash never leaves a half-written cache


# ---------------------------------------------------------------------------
# tier 2: local GFF3 / GTF
# ---------------------------------------------------------------------------

_GTF_ATTR = re.compile(r'(\w+)\s+"([^"]*)"')


def _open_text(path):
    return gzip.open(path, 'rt', encoding='utf-8', errors='replace') if path.endswith('.gz') \
        else open(path, 'r', encoding='utf-8', errors='replace')


def _attr_symbols(attrs, is_gtf):
    """(primary_id, [all symbol-like strings]) from a GFF3/GTF attribute column."""
    if is_gtf:
        d = {k: v for k, v in _GTF_ATTR.findall(attrs)}
        syms = [d[k] for k in ('gene_name', 'gene_id', 'gene_synonym', 'Name', 'locus_tag') if k in d]
        return d.get('gene_id', ''), syms
    d = {}
    for part in attrs.split(';'):
        if '=' in part:
            k, v = part.split('=', 1)
            d[k.strip()] = v.strip()
    syms = []
    for k in ('Name', 'gene_name', 'symbol', 'gene', 'locus_tag', 'gene_id', 'ID'):
        if k in d:
            syms.append(d[k])
    for k in ('Alias', 'gene_synonym', 'synonym'):
        if k in d:
            syms += re.split(r'[,|]', d[k])
    pid = d.get('ID', d.get('gene_id', ''))
    return re.sub(r'^gene[:\-]', '', pid), [re.sub(r'^gene[:\-]', '', s) for s in syms if s]


def resolve_from_annotation(genes, annotation_path, chromosomes=None):
    """Stream `annotation_path` once; return (resolved, ambiguous) where
    resolved maps gene -> [locus dict] and ambiguous maps gene -> reason.

    Only the wanted symbols are kept, so memory is O(len(genes)), not O(genome).
    A symbol that matches more than one DIFFERENT gene record is ambiguous and
    is not resolved (the agent may disambiguate it from context)."""
    wanted = {_norm(g): g for g in genes}
    hits = {}  # gene -> {primary_id: locus}
    is_gtf = re.search(r'\.gtf(\.gz)?$', annotation_path, re.I) is not None
    source = os.path.basename(annotation_path)
    with _open_text(annotation_path) as fh:
        for line in fh:
            if not line or line[0] == '#':
                continue
            cols = line.rstrip('\n').split('\t')
            if len(cols) < 9 or cols[2] not in ('gene', 'ncRNA_gene', 'pseudogene', 'transposable_element_gene'):
                continue
            pid, symbols = _attr_symbols(cols[8], is_gtf)
            matched = {wanted[_norm(s)] for s in symbols if _norm(s) in wanted}
            if not matched:
                continue
            locus = {'chromosome': cols[0], 'start': int(cols[3]), 'end': int(cols[4]),
                     'locus_id': pid, 'source': source}
            for g in matched:
                hits.setdefault(g, {})[pid or f'{cols[0]}:{cols[3]}-{cols[4]}'] = locus
    resolved, ambiguous = {}, {}
    for g, by_id in hits.items():
        if len(by_id) > 1:
            ambiguous[g] = f"matches {len(by_id)} different gene records in {source}: {sorted(by_id)[:4]}"
            continue
        locus = next(iter(by_id.values()))
        problem = _plausible(locus['start'], locus['end'], chromosomes, locus['chromosome'])
        if problem:
            ambiguous[g] = f"{source}: {problem}"
            continue
        resolved[g] = [locus]
    return resolved, ambiguous


# ---------------------------------------------------------------------------
# tier 3: Ensembl REST (batched, optional, never raises)
# ---------------------------------------------------------------------------

ENSEMBL_REST = 'https://rest.ensembl.org'


def _http_post_json(url, payload, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), method='POST',
                                 headers={'Content-Type': 'application/json', 'Accept': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def resolve_from_ensembl(genes, species, chromosomes=None, expected_assembly=None,
                         timeout=30, batch=200, post=_http_post_json):
    """One POST per `batch` symbols. Returns (resolved, notes). Network errors
    only produce a note - the genes simply stay unresolved for the next tier.
    `post` is injectable so the tier can be tested offline."""
    name = species_to_ensembl(species)
    resolved, notes = {}, []
    genes = list(genes)
    for i in range(0, len(genes), batch):
        chunk = genes[i:i + batch]
        data = None
        for attempt in range(3):
            try:
                data = post(f'{ENSEMBL_REST}/lookup/symbol/{name}', {'symbols': chunk}, timeout)
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 429 and attempt < 2:
                    time.sleep(float(exc.headers.get('Retry-After', 1)) + 0.5)
                    continue
                notes.append(f"Ensembl lookup skipped ({exc.code} {exc.reason})")
                break
            except Exception as exc:  # offline, DNS, proxy refusal, bad JSON ...
                notes.append(f"Ensembl lookup skipped ({type(exc).__name__}: {exc})")
                break
        if not isinstance(data, dict):
            continue
        for sym, rec in data.items():
            if not isinstance(rec, dict) or rec.get('object_type', 'Gene') != 'Gene':
                continue
            assembly = str(rec.get('assembly_name', ''))
            if expected_assembly and expected_assembly.lower() not in assembly.lower():
                notes.append(f"{sym}: Ensembl assembly '{assembly}' != expected '{expected_assembly}' - not used")
                continue
            problem = _plausible(rec.get('start'), rec.get('end'), chromosomes, rec.get('seq_region_name'))
            if problem:
                notes.append(f"{sym}: Ensembl {problem} - not used")
                continue
            resolved[sym] = [{'chromosome': str(rec['seq_region_name']), 'start': int(rec['start']),
                              'end': int(rec['end']), 'locus_id': str(rec.get('id', '')),
                              'source': f"Ensembl Plants REST ({assembly or 'assembly n/a'}; "
                                        f"{rec.get('db_type', 'core')})"}]
    return resolved, notes


# ---------------------------------------------------------------------------
# authoritative coordinates BY LOCUS ID (used to verify what the agent returns)
# ---------------------------------------------------------------------------

def resolve_ids_authoritatively(locus_ids, species, annotation_path=None, use_ensembl=True,
                                chromosomes=None, expected_assembly=None, post=_http_post_json):
    """locus_id -> locus dict, from the local annotation first and Ensembl
    (`POST lookup/id`) second. The agent is good at the fuzzy step (symbol ->
    systematic ID) and unreliable at exact base-pair coordinates; an ID, unlike
    a symbol, is unambiguous, so this is the cheap, exact half of the job.
    IDs neither source knows are simply absent from the result."""
    ids = sorted({str(i).strip() for i in locus_ids if str(i).strip()})
    found = {}
    if not ids:
        return found
    if annotation_path and os.path.isfile(annotation_path):
        try:
            got, _ = resolve_from_annotation(ids, annotation_path, chromosomes)
            for key, loci in got.items():
                found[key] = loci[0]
        except Exception as exc:
            warnings.warn(f"[gene_location_resolver] annotation ID check failed "
                          f"({type(exc).__name__}: {exc})")
    rest = [i for i in ids if i not in found]
    if rest and use_ensembl:
        name = species_to_ensembl(species)
        for i in range(0, len(rest), 200):
            chunk = rest[i:i + 200]
            try:
                data = post(f'{ENSEMBL_REST}/lookup/id?species={name}', {'ids': chunk}, 30)
            except Exception:
                break
            for key, rec in (data or {}).items():
                if not isinstance(rec, dict):
                    continue
                assembly = str(rec.get('assembly_name', ''))
                if expected_assembly and expected_assembly.lower() not in assembly.lower():
                    continue
                if _plausible(rec.get('start'), rec.get('end'), chromosomes, rec.get('seq_region_name')):
                    continue
                found[key] = {'chromosome': str(rec['seq_region_name']), 'start': int(rec['start']),
                              'end': int(rec['end']), 'locus_id': key,
                              'source': f"Ensembl Plants REST ({assembly or 'assembly n/a'})"}
    return found


# ---------------------------------------------------------------------------
# orchestration of tiers 0-3
# ---------------------------------------------------------------------------

def resolve_without_model(candidate_ids, species, unit='bp', cache_path=None, annotation_path=None,
                          use_ensembl=True, expected_assembly=None, marker_info_path=None,
                          hints=None, retry_unresolved=False, ensembl_post=_http_post_json):
    """Run tiers 0-3. Returns a dict:

        resolved   : gene -> [locus dict]    (from cache / annotation / ensembl)
        residual   : [gene]                  (still needs the agent)
        skipped    : gene -> reason          (tier-0 non-genes, and cached 'unresolved')
        tiers      : {'cache': n, 'annotation': n, 'ensembl': n, 'prefilter': n, 'cached_unresolved': n}
        notes      : [str]                   (human-readable diagnostics)
        cache      : GeneLocationCache or None (caller saves it after the agent tier)
    """
    notes = []
    resolved, skipped = {}, {}
    tiers = {'prefilter': 0, 'cache': 0, 'cached_unresolved': 0, 'annotation': 0, 'ensembl': 0}

    todo, pre_skipped = prefilter_candidates(candidate_ids, hints)
    skipped.update(pre_skipped)
    tiers['prefilter'] = len(pre_skipped)

    cache = GeneLocationCache(cache_path) if cache_path else None
    if cache is not None:
        remaining = []
        for g in todo:
            rows = cache.get(species, unit, g)
            if rows and rows[0].get('status') == 'resolved':
                resolved[g] = [{'chromosome': r['chromosome'], 'start': float(r['start']),
                                'end': float(r['end']), 'locus_id': r.get('locus_id', ''),
                                'source': (r.get('source') or '') + ' [cached]'} for r in rows]
                tiers['cache'] += 1
            elif rows and rows[0].get('status') == 'unresolved' and not retry_unresolved:
                skipped[g] = 'previously could not be verified (cached; enable "retry" to ask again)'
                tiers['cached_unresolved'] += 1
            else:
                remaining.append(g)
        todo = remaining

    chromosomes = marker_chromosomes(marker_info_path)

    if todo and unit == 'bp' and annotation_path:
        if os.path.isfile(annotation_path):
            try:
                found, ambiguous = resolve_from_annotation(todo, annotation_path, chromosomes)
                for g, loci in found.items():
                    resolved[g] = loci
                    if cache is not None:
                        cache.put(species, unit, g, 'resolved', loci, 'annotation')
                tiers['annotation'] = len(found)
                notes += [f"{g}: {why} - left for the next tier" for g, why in ambiguous.items()]
                todo = [g for g in todo if g not in found]
            except Exception as exc:
                notes.append(f"annotation '{annotation_path}' could not be used ({type(exc).__name__}: {exc})")
        else:
            notes.append(f"annotation file '{annotation_path}' not found - skipped")

    if todo and unit == 'bp' and use_ensembl:
        found, enotes = resolve_from_ensembl(todo, species, chromosomes, expected_assembly, post=ensembl_post)
        notes += enotes
        for g, loci in found.items():
            resolved[g] = loci
            if cache is not None:
                cache.put(species, unit, g, 'resolved', loci, 'ensembl')
        tiers['ensembl'] = len(found)
        todo = [g for g in todo if g not in found]

    return {'resolved': resolved, 'residual': todo, 'skipped': skipped, 'tiers': tiers,
            'notes': notes, 'cache': cache, 'chromosomes': chromosomes}
