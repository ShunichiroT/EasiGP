"""tools/gene_location_resolver_check.py - offline self-test of the tiered gene-location curation.

Needs no network, no Claude account and no torch: it builds a tiny GFF3, mocks Ensembl and
replaces the `claude` binary with a stub that records its command line. Run from the EasiGP
source directory:   python tools/gene_location_resolver_check.py
Asserts: prefilter, annotation (ambiguity/chromosome guards), Ensembl (assembly/length guards,
offline tolerance), agent sees ONLY the residual genes in ONE lean call, agent coordinates are
corrected by locus ID, the cache makes a re-run cost zero agent sessions, and the unknown-flag
fallback retries once. Prints ALL OK.
"""
import os, sys
STUB = r'''#!/usr/bin/env python3
"""Stub for `claude -p`: records argv, answers like the lean agent would."""
import sys, json, os, re
argv = sys.argv[1:]
open(os.environ['FAKE_CLAUDE_LOG'], 'a').write(json.dumps(argv) + '\n')
if os.environ.get('FAKE_REJECT_FLAGS') and '--effort' in argv:
    sys.stderr.write("error: unknown option '--effort'\n"); sys.exit(1)
prompt = argv[argv.index('-p') + 1]
genes = [l.split(' [')[0] for l in prompt.split('Genes ([full name] when known):\n')[1].splitlines() if l.strip()]
table = {'BRC1': 'AT3G18550\t3\t6400000\t6402000\tTAIR10 search snippet',
         'D14': 'AT3G03990\t3\t1000000\t1003000\tTAIR10 search snippet',
         'MAX2': 'AT2G42620\t2\t17700000\t17703000\tTAIR10'}
lines = [f"{g}\t{table[g]}" if g in table else f"{g}\tNA" for g in genes]
out = {"result": "\n".join(lines), "total_cost_usd": 0.0123, "num_turns": 2,
       "usage": {"input_tokens": 3, "output_tokens": 120, "cache_creation_input_tokens": 4000, "cache_read_input_tokens": 0}}
print(json.dumps(out))
'''
import json, os, sys, tempfile, gzip
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import pandas as pd
from Preprocess import gene_location_resolver as R
from Preprocess import gene_location_agent as A

tmp = tempfile.mkdtemp()
gff = os.path.join(tmp, 'ann.gff3.gz')
with gzip.open(gff, 'wt') as f:
    f.write("##gff-version 3\n")
    f.write("1\tTAIR10\tgene\t100\t900\t.\t+\t.\tID=AT1G00100;Name=PIN1;Alias=PIN-FORMED1\n")
    f.write("1\tTAIR10\tmRNA\t100\t900\t.\t+\t.\tID=AT1G00100.1;Parent=AT1G00100;Name=PIN1\n")   # not a gene row
    f.write("2\tTAIR10\tgene\t5000\t6000\t.\t-\t.\tID=AT2G00200;Name=DUP\n")
    f.write("4\tTAIR10\tgene\t7000\t8000\t.\t-\t.\tID=AT4G00300;Name=DUP\n")                      # ambiguous
    f.write("NC_999\tx\tgene\t1\t50\t.\t+\t.\tID=ATXG1;Name=BADCHR\n")                              # chromosome not in markers
    f.write("3\tTAIR10\tgene\t6400000\t6402000\t.\t+\t.\tID=AT3G18550;Name=BRC1\n")                  # also verifies agent ID
    f.write("3\tTAIR10\tgene\t1000100\t1003100\t.\t+\t.\tID=AT3G03990;Name=OTHER\n")                 # agent said 1000000-1003000 -> within 1kb
    f.write("2\tTAIR10\tgene\t17700000\t17701000\t.\t+\t.\tID=AT2G42620;Name=OTHER2\n")             # agent said ...703000 -> corrected
markers = os.path.join(tmp, 'markers.csv')
pd.DataFrame({'chromosome': ['1','2','3','4'], 'name': list('abcd'), 'start': [1]*4, 'end': [2]*4}).to_csv(markers, index=False)

# --- tier 0 -----------------------------------------------------------------
keep, skip = R.prefilter_candidates(['MAX_Pathway', 'Strigolactones', 'BRC1', 'CCD7', 'Auxin signalling'])
assert keep == ['BRC1', 'CCD7'] and set(skip) == {'MAX_Pathway', 'Strigolactones', 'Auxin signalling'}, (keep, skip)

# --- tier 2 -----------------------------------------------------------------
res, amb = R.resolve_from_annotation(['PIN1', 'PIN-FORMED1', 'DUP', 'BADCHR', 'NOPE'], gff, chromosomes={'1','2','3','4'})
assert res['PIN1'][0]['start'] == 100 and res['PIN-FORMED1'][0]['locus_id'] == 'AT1G00100'
assert 'DUP' in amb and 'BADCHR' in amb and 'NOPE' not in res

# --- tier 3 (mock) ----------------------------------------------------------
def fake_post(url, payload, timeout):
    assert '/lookup/symbol/arabidopsis_thaliana' in url
    return {'CCD7': {'object_type': 'Gene', 'seq_region_name': '2', 'start': 100, 'end': 2000, 'id': 'AT2G44990',
                     'assembly_name': 'TAIR10', 'db_type': 'core'},
            'HUGE': {'object_type': 'Gene', 'seq_region_name': '2', 'start': 1, 'end': 9_000_000, 'id': 'X', 'assembly_name': 'TAIR10'}}
r, notes = R.resolve_from_ensembl(['CCD7', 'HUGE'], 'Arabidopsis thaliana', {'1','2','3','4'}, 'TAIR10', post=fake_post)
assert list(r) == ['CCD7'] and any('HUGE' in n for n in notes), (r, notes)
r2, notes2 = R.resolve_from_ensembl(['CCD7'], 'Arabidopsis thaliana', None, 'TAIR9', post=fake_post)
assert not r2 and 'TAIR9' in notes2[0]
def boom(*a, **k): raise OSError('proxy refused')
r3, n3 = R.resolve_from_ensembl(['CCD7'], 'Arabidopsis thaliana', post=boom)
assert not r3 and 'skipped' in n3[0]

# --- full flow with stub claude ---------------------------------------------
os.environ['FAKE_CLAUDE_LOG'] = os.path.join(tmp, 'calls.log')
stub = os.path.join(tmp, 'fake_claude.py')
open(stub, 'w').write(STUB)
os.chmod(stub, 0o755)
out = os.path.join(tmp, 'locs.csv')
genes = ['MAX_Pathway', 'PIN1', 'CCD7', 'BRC1', 'D14', 'MAX2', 'MYSTERY', 'DUP']
R._http_post_json = fake_post  # tier 3 via module default is bound at def time -> pass explicitly instead
import Preprocess.gene_location_resolver as RR
_orig = RR.resolve_without_model
def patched(*a, **k):
    k['ensembl_post'] = fake_post
    return _orig(*a, **k)
A.resolve_without_model = patched
_orig_ids = A.resolve_ids_authoritatively
A.resolve_ids_authoritatively = lambda *a, **k: _orig_ids(*a, **dict(k, post=fake_post))

table, resolved, unresolved = A.curate_gene_locations(
    genes, 'Arabidopsis thaliana', out, backend='claude_code', unit='bp', claude_binary=stub,
    annotation_path=gff, marker_info_path=markers, expected_assembly='TAIR10',
    candidate_hints={'BRC1': 'BRANCHED1'})
calls = [json.loads(l) for l in open(os.environ['FAKE_CLAUDE_LOG'])]
print('agent sessions:', len(calls))
assert len(calls) == 1
argv = calls[0]
for flag in ('--tools', '--system-prompt', '--disable-slash-commands', '--strict-mcp-config', '--no-session-persistence'):
    assert flag in argv, flag
assert argv[argv.index('--tools') + 1] == 'WebSearch'
prompt = argv[argv.index('-p') + 1]
asked = prompt.split('Genes ([full name] when known):\n')[1].split('\n')
print('genes shown to agent:', asked)
# MAX_Pathway prefiltered; PIN1 annotation; CCD7 ensembl; DUP ambiguous (annotation) -> agent; BRC1 annotation (AT3G18550 matches Name) -> free
assert 'PIN1' not in asked and 'MAX_Pathway' not in asked and 'CCD7' not in asked and 'BRC1' not in asked
assert set(asked) == {'D14', 'MAX2', 'MYSTERY', 'DUP'}, asked
df = pd.read_csv(out)
print(df[['Gene_Name', 'Start_bp', 'End_bp', 'Chromosome', 'AGI_Locus_ID', 'Resolver']].to_string(index=False))
m = {r.Gene_Name: r for r in df.itertuples()}
assert m['MAX2'].End_bp == 17701000 and m['MAX2'].Resolver == 'agent+verified'   # corrected by annotation-by-ID
assert m['D14'].Start_bp == 1000100                                              # within 1kb -> authoritative still used
assert 'MYSTERY' not in m and 'DUP' not in m and 'MAX_Pathway' not in m
assert sorted(resolved) == ['BRC1', 'CCD7', 'D14', 'MAX2', 'PIN1'], resolved
rep = A.get_last_curation_report()
print('tiers:', rep['tiers'], '| usage:', rep['agent_usage'])
assert rep['tiers']['annotation'] >= 2 and rep['tiers']['ensembl'] == 1 and rep['tiers']['prefilter'] == 1
assert any('MAX2' in n and 'corrected' in n for n in rep['notes'])

# --- second run: everything from cache, zero agent sessions ------------------
open(os.environ['FAKE_CLAUDE_LOG'], 'w').close()
A.curate_gene_locations(genes, 'Arabidopsis thaliana', out, backend='claude_code', unit='bp', claude_binary=stub,
                        annotation_path=gff, marker_info_path=markers, expected_assembly='TAIR10')
n2 = len([l for l in open(os.environ['FAKE_CLAUDE_LOG']) if l.strip()])
print('agent sessions on re-run:', n2)
assert n2 == 0
rep2 = A.get_last_curation_report(); print('tiers re-run:', rep2['tiers'])
assert rep2['tiers']['cached_unresolved'] >= 2     # MYSTERY / DUP remembered as unresolved

# --- unknown-flag fallback -----------------------------------------------------
os.environ['FAKE_REJECT_FLAGS'] = '1'
open(os.environ['FAKE_CLAUDE_LOG'], 'w').close()
A.curate_gene_locations(genes + ['EXTRA1'], 'Arabidopsis thaliana', os.path.join(tmp, 'o2.csv'), backend='claude_code',
                        unit='bp', claude_binary=stub, annotation_path=gff, use_cache=False, retry_unresolved=True)
calls = [json.loads(l) for l in open(os.environ['FAKE_CLAUDE_LOG'])]
print('fallback calls:', len(calls), ['--effort' in c for c in calls])
assert len(calls) == 2 and '--effort' in calls[0] and '--effort' not in calls[1]
print('ALL OK')
