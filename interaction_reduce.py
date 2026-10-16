"""Memory-bounded reduction of marker-pair interaction / attention tables.

Why this module exists
----------------------
`Interaction.csv` / `Attention.csv` hold one row per (population, model,
phenotype, ratio, sample, marker1, marker2).  When ``interaction_top`` is
``'all'`` (the default) every pair of the interaction shortlist is stored for
every replicate, so with several interaction-emitting models the number of
DISTINCT (population, model, phenotype, marker1, marker2) groups G reaches
hundreds of millions.  The circos step only ever draws the strongest few of
them per (population, model, phenotype), but the previous implementation

  * held all G groups as five *object* (Python-string) columns - roughly
    150-300 bytes per group instead of the 24 bytes of information in it;
  * then made several full copies of that table (``work``, ``_all_reduced``,
    ``_pop_reduced``, the per-key slices);
  * and pickled all of it into every worker process of the circos fan-out.

`PairSumStore` replaces that with a compact, streaming representation:

  * marker / model / phenotype / population names are dictionary-coded to
    small integers once; a group is (key id int32, marker1 int32, marker2
    int32, sum float64, count int32) = 24 bytes;
  * chunks are folded in with one hash-based ``factorize`` + ``bincount``;
  * the running total is compacted on a doubling schedule (amortised linear
    time) and, if it would ever exceed ``max_resident_rows``, spilled to flat
    binary files partitioned by key - so peak memory is bounded by the
    resident budget, not by G;
  * the final per-key mean, the "top N% / top M" selection that circos
    applies, and the string decoding all happen one (model, phenotype) unit
    at a time, and only the rows that will actually be drawn are ever turned
    back into strings.

The selection implemented here is a faithful, vectorised port of
``circos_plot._select_top_interactions()`` (including its rank-based top-N%,
nonzero cap, round-to-zero floor and 'count' mode) - see `select_top_indices`.

Pure NumPy / pandas, zero project imports (so batch_reader, circos_plot and
tests can all import it without circularity).
"""
from __future__ import annotations

import os
import shutil
import tempfile
import weakref

import numpy as np
import pandas as pd

# Rows (groups) the in-memory running total may hold before it is spilled to
# disk.  24 bytes/row resident, ~4-5x that transiently while compacting.
# Override with EASIGP_PAIR_REDUCE_MAX_ROWS (an integer number of rows).
_DEFAULT_MAX_RESIDENT_ROWS = 40_000_000
_MIN_COMPACT_TRIGGER = 4_000_000
# Rows per slice when ingesting an in-memory DataFrame.
_FRAME_CHUNK_ROWS = 2_000_000

_BITS = 21                      # fast composite-key path: ids < 2**21
_FAST_LIMIT = 1 << _BITS

_SPILL_DTYPE = np.dtype([('m1', '<i4'), ('m2', '<i4'), ('s', '<f8'), ('c', '<i4')])


def _max_resident_rows() -> int:
    try:
        v = int(float(os.environ.get('EASIGP_PAIR_REDUCE_MAX_ROWS', '')))
        if v > 0:
            return v
    except (TypeError, ValueError):
        pass
    return _DEFAULT_MAX_RESIDENT_ROWS


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
class _Codebook:
    """name <-> dense int code.  Names keep their native Python type."""
    __slots__ = ('names', 'index')

    def __init__(self):
        self.names = []
        self.index = {}

    def add(self, name) -> int:
        i = self.index.get(name)
        if i is None:
            i = len(self.names)
            self.index[name] = i
            self.names.append(name)
        return i

    def __len__(self):
        return len(self.names)


def _native(values):
    """Python-native list of a factorize()/categories result."""
    return pd.Index(values).tolist() if not isinstance(values, list) else values


def _encode(series, codebook, *, stringify=False, natives=None):
    """int64 array of global codes for `series` (-1 where the value is NaN).

    Works for categorical columns (cheap - uses their categories) and for
    plain object / numeric / string columns (hash-based factorize)."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        cats = _native(series.cat.categories)
        local = series.cat.codes.to_numpy().astype(np.int64, copy=False)
    else:
        local, uniques = pd.factorize(series)
        cats = _native(uniques)
        local = np.asarray(local, dtype=np.int64)
    raw_cats = cats
    if stringify:
        cats = [str(c) for c in cats]
    gmap = np.fromiter((codebook.add(c) for c in cats), dtype=np.int64, count=len(cats))
    if natives is not None:
        for code, raw in zip(gmap.tolist(), raw_cats):
            natives.setdefault(code, raw)
    if gmap.size == 0:
        return np.full(local.shape[0], -1, dtype=np.int64)
    out = gmap[np.clip(local, 0, None)]
    out[local < 0] = -1
    return out


def _lexical_rank(names) -> np.ndarray:
    """rank[code] = position of names[code] in ascending lexical order - the
    order a pandas ``groupby(..., sort=True)`` over the string column gives."""
    n = len(names)
    arr = np.empty(n, dtype=object)
    arr[:] = names
    try:
        order = np.argsort(arr, kind='stable')
    except TypeError:                      # mixed int / str marker names
        order = np.array(sorted(range(n), key=lambda i: str(names[i])), dtype=np.int64)
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n, dtype=np.int64)
    return rank


def _reduce_rows(kg, m1, m2, s, c):
    """Collapse rows sharing (kg, m1, m2): sums of `s` and `c`."""
    n = kg.shape[0]
    if n == 0:
        return kg, m1, m2, s, c
    kmax = int(kg.max()); mmax = int(max(m1.max(), m2.max()))
    if kmax < _FAST_LIMIT and mmax < _FAST_LIMIT:
        key = (kg.astype(np.int64) << (2 * _BITS)) | (m1.astype(np.int64) << _BITS) | m2.astype(np.int64)
        codes, uniq = pd.factorize(key)
        ng = uniq.shape[0]
        if ng == n:                         # already unique - nothing to merge
            return kg, m1, m2, s, c
        s_out = np.bincount(codes, weights=s, minlength=ng)
        c_out = np.bincount(codes, weights=c, minlength=ng)
        mask = (1 << _BITS) - 1
        return ((uniq >> (2 * _BITS)).astype(np.int32),
                ((uniq >> _BITS) & mask).astype(np.int32),
                (uniq & mask).astype(np.int32),
                s_out, np.rint(c_out).astype(np.int32))
    frame = pd.DataFrame({'k': kg, 'a': m1, 'b': m2, 's': s, 'c': c.astype(np.int64)})
    g = frame.groupby(['k', 'a', 'b'], sort=False, as_index=False)[['s', 'c']].sum()
    return (g['k'].to_numpy(np.int32), g['a'].to_numpy(np.int32), g['b'].to_numpy(np.int32),
            g['s'].to_numpy(np.float64), g['c'].to_numpy(np.int64).astype(np.int32))


def _mean(s, c):
    out = np.full(s.shape[0], np.nan, dtype=np.float64)
    np.divide(s, c, out=out, where=c > 0)
    return out


def select_top_indices(values, m1, m2, rank, *, mode='percentage', top_percent=None, top_count=None):
    """Indices (ordered strongest first) of the rows circos draws for ONE
    (population, model, phenotype) key - a vectorised port of
    ``circos_plot._select_top_interactions()``:

      * ``'count'``      : exactly the top ``top_count`` rows by value
        (fewer if fewer exist; NaN never selected);
      * ``'percentage'`` : the RANK-based top ``top_percent``% of ALL rows of
        the key (``round(n * pct / 100)``), floored at 1 when pct > 0 and a
        nonzero row exists, and capped at the number of rows with value > 0
        so unscored (zero) pairs are never padded in.

    Ties are broken by (marker1, marker2) lexical order, which is the row
    order the old code's sorted ``groupby`` frame had, so equal values select
    and order exactly as ``DataFrame.nlargest`` did.
    """
    n = values.shape[0]
    if mode == 'count':
        m = max(0, min(int(top_count or 0), n))
        cand = np.flatnonzero(~np.isnan(values))
        m = min(m, cand.size)
    else:
        nz = values > 0                                   # NaN -> False
        cand = np.flatnonzero(nz)
        pct = float(top_percent)
        m = int(round(n * (pct / 100)))
        if m == 0 and pct > 0 and cand.size > 0:
            m = 1
        m = max(0, min(m, cand.size))
    if m <= 0 or cand.size == 0:
        return np.empty(0, dtype=np.int64)
    v = values[cand]
    if m >= cand.size:
        chosen = cand
    else:
        thr = np.partition(v, cand.size - m)[cand.size - m]      # m-th largest
        gt = cand[v > thr]
        eq = cand[v == thr]
        need = m - gt.size
        if eq.size > need:
            eq = eq[np.lexsort((rank[m2[eq]], rank[m1[eq]]))[:need]]
        chosen = np.concatenate([gt, eq])
    order = np.lexsort((rank[m2[chosen]], rank[m1[chosen]], -values[chosen]))
    return chosen[order]


# --------------------------------------------------------------------------
# the store
# --------------------------------------------------------------------------
def _cleanup_dir(path):
    shutil.rmtree(path, ignore_errors=True)


class PairSumStore:
    """Streaming (sum, count) accumulator over (population, model, phenotype,
    marker1, marker2) groups.  See the module docstring.

    order_mode : 'appearance' (in-memory tables: models ordered by first
        appearance) or 'sorted' (the CSV streaming path: models ordered as the
        old ``groupby(sort=True)`` aggregate frame ordered them).
    """

    def __init__(self, *, order_mode='appearance', max_resident_rows=None):
        if order_mode not in ('appearance', 'sorted'):
            raise ValueError(f"order_mode must be 'appearance' or 'sorted', got {order_mode!r}")
        self.order_mode = order_mode
        self.max_resident_rows = int(max_resident_rows or _max_resident_rows())
        self._pop, self._model, self._pheno, self._marker = _Codebook(), _Codebook(), _Codebook(), _Codebook()
        self._pop_native = {}          # pop code -> value as read (for legacy sort order)
        self._kgrp_index = {}          # (pop, model, pheno) code tuple -> kgrp id
        self._kgrp_keys = []           # kgrp id -> code tuple
        self._seen_seq = {}            # (pop code, model code) -> first-seen sequence
        self._parts = []               # resident (kg, m1, m2, s, c) tuples
        self._resident_rows = 0
        self._compact_trigger = _MIN_COMPACT_TRIGGER
        self._spill_dir = None
        self._spill_finalizer = None
        self._spilled = False
        self._rows_in = 0

    # ---- DataFrame-like surface used by callers that only test emptiness ----
    @property
    def shape(self):
        return (0 if not self._kgrp_keys else 1, 5)

    def __len__(self):
        return 0 if not self._kgrp_keys else 1

    @property
    def empty(self):
        return not self._kgrp_keys

    # ---- ingestion -------------------------------------------------------
    def ingest(self, df, *, population_col='population'):
        """Fold one table into the running totals.  `df` needs 'population',
        'model', 'phenotype', 'marker1', 'marker2' and either 'value' (one raw
        observation per row) or 'value_sum' + 'value_count'.  Rows whose
        marker1/marker2 is the 'factor' sentinel, or any key is NaN, are
        skipped (exactly what the old code's row filter / groupby did)."""
        if df is None or df.shape[0] == 0:
            return self
        n = df.shape[0]
        for start in range(0, n, _FRAME_CHUNK_ROWS):
            self._ingest_chunk(df.iloc[start:start + _FRAME_CHUNK_ROWS] if n > _FRAME_CHUNK_ROWS else df,
                               population_col)
        return self

    def _ingest_chunk(self, chunk, population_col):
        has_pop = population_col in chunk.columns
        pop = _encode(chunk[population_col], self._pop, stringify=True, natives=self._pop_native) if has_pop \
            else np.full(chunk.shape[0], self._pop.add('all'), dtype=np.int64)
        model = _encode(chunk['model'], self._model)
        pheno = _encode(chunk['phenotype'], self._pheno)
        m1 = _encode(chunk['marker1'], self._marker)
        m2 = _encode(chunk['marker2'], self._marker)
        if 'value_sum' in chunk.columns and 'value_count' in chunk.columns:
            s = chunk['value_sum'].to_numpy(dtype=np.float64, na_value=0.0)
            c = chunk['value_count'].to_numpy(dtype=np.float64, na_value=0.0)
        else:
            raw = chunk['value'].to_numpy(dtype=np.float64, na_value=np.nan)
            ok = ~np.isnan(raw)
            s = np.where(ok, raw, 0.0)
            c = ok.astype(np.float64)

        keep = (pop >= 0) & (model >= 0) & (pheno >= 0) & (m1 >= 0) & (m2 >= 0)
        fac = self._marker.index.get('factor')
        if fac is not None:
            keep &= (m1 != fac) & (m2 != fac)
        if not keep.all():
            pop, model, pheno, m1, m2, s, c = (a[keep] for a in (pop, model, pheno, m1, m2, s, c))
        if pop.shape[0] == 0:
            return
        self._rows_in += int(pop.shape[0])

        # (pop, model, pheno) -> dense kgrp id; first-seen order of (pop, model)
        combo = (pop << 40) | (model << 20) | pheno if (len(self._pop) < (1 << 20) and len(self._model) < (1 << 20)
                                                           and len(self._pheno) < (1 << 20)) else None
        if combo is None:
            raise ValueError('too many distinct population/model/phenotype labels (>= 2**20)')
        codes, uniq = pd.factorize(combo)
        kmap = np.empty(uniq.shape[0], dtype=np.int32)
        for j, u in enumerate(uniq.tolist()):
            key = (u >> 40, (u >> 20) & 0xFFFFF, u & 0xFFFFF)
            kid = self._kgrp_index.get(key)
            if kid is None:
                kid = len(self._kgrp_keys)
                self._kgrp_index[key] = kid
                self._kgrp_keys.append(key)
            kmap[j] = kid
            pm = (key[0], key[1])
            if pm not in self._seen_seq:
                self._seen_seq[pm] = len(self._seen_seq)
        kg = kmap[codes]

        part = _reduce_rows(kg, m1.astype(np.int32), m2.astype(np.int32), s, c)
        self._parts.append(part)
        self._resident_rows += part[0].shape[0]
        if self._resident_rows >= self._compact_trigger:
            self._compact()

    # ---- compaction / spill ---------------------------------------------
    def _compact(self):
        if not self._parts:
            return
        if len(self._parts) == 1:
            merged = self._parts[0]
        else:
            merged = _reduce_rows(*(np.concatenate([p[i] for p in self._parts]) for i in range(5)))
        self._parts = [merged]
        self._resident_rows = merged[0].shape[0]
        if self._resident_rows > self.max_resident_rows:
            self._spill()
        else:
            self._compact_trigger = max(_MIN_COMPACT_TRIGGER, 2 * self._resident_rows)

    def _ensure_spill_dir(self):
        if self._spill_dir is None:
            base = os.environ.get('SCRATCH_TMP_DIR') or None
            if base:
                os.makedirs(base, exist_ok=True)
            self._spill_dir = tempfile.mkdtemp(prefix='easigp_pairs_', dir=base)
            self._spill_finalizer = weakref.finalize(self, _cleanup_dir, self._spill_dir)

    def _spill(self):
        """Move the resident running total to per-key flat files on disk."""
        kg, m1, m2, s, c = self._parts[0]
        self._ensure_spill_dir()
        order = np.argsort(kg, kind='stable')
        kg = kg[order]
        rec = np.empty(kg.shape[0], dtype=_SPILL_DTYPE)
        rec['m1'] = m1[order]; rec['m2'] = m2[order]; rec['s'] = s[order]; rec['c'] = c[order]
        bounds = np.flatnonzero(np.diff(kg)) + 1
        starts = np.concatenate([[0], bounds]); ends = np.concatenate([bounds, [kg.shape[0]]])
        for a, b in zip(starts.tolist(), ends.tolist()):
            with open(os.path.join(self._spill_dir, f'k{int(kg[a])}.bin'), 'ab') as fh:
                rec[a:b].tofile(fh)
        self._parts = []
        self._resident_rows = 0
        self._compact_trigger = _MIN_COMPACT_TRIGGER
        self._spilled = True

    def close(self):
        if self._spill_finalizer is not None and self._spill_finalizer.alive:
            self._spill_finalizer()
        self._spill_dir = None

    # ---- read-back -------------------------------------------------------
    def _finalize_resident(self):
        """Compact once more and index the resident rows by key id."""
        self._compact()
        if not self._parts:
            self._res = None
            return
        kg, m1, m2, s, c = self._parts[0]
        order = np.argsort(kg, kind='stable')
        kg = kg[order]
        self._res = (m1[order], m2[order], s[order], c[order])
        uniq, first = np.unique(kg, return_index=True)
        last = np.append(first[1:], kg.shape[0])
        self._res_slices = {int(k): (int(a), int(b)) for k, a, b in zip(uniq, first, last)}

    def _rows_for(self, kid):
        """(m1, m2, s, c) arrays of every row of key `kid`, unique per pair."""
        pieces = []
        if self._res is not None and kid in self._res_slices:
            a, b = self._res_slices[kid]
            pieces.append(tuple(x[a:b] for x in self._res))
        if self._spilled:
            path = os.path.join(self._spill_dir, f'k{kid}.bin')
            if os.path.exists(path):
                rec = np.fromfile(path, dtype=_SPILL_DTYPE)
                # the same pair can have been appended by several spill rounds
                _, a, b, s, c = _reduce_rows(np.zeros(rec.shape[0], np.int32), rec['m1'], rec['m2'], rec['s'], rec['c'])
                pieces.append((a, b, s, c))
        if not pieces:
            z = np.empty(0)
            return z.astype(np.int32), z.astype(np.int32), z.astype(np.float64), z.astype(np.int32)
        if len(pieces) == 1:
            return pieces[0]
        cat = [np.concatenate([p[i] for p in pieces]) for i in range(4)]
        zeros = np.zeros(cat[0].shape[0], dtype=np.int32)
        _, a, b, s, c = _reduce_rows(zeros, cat[0], cat[1], cat[2], cat[3])
        return a, b, s, c

    def _model_orders(self, pop_label_of):
        """(all_order, {pop_label: [model, ...]}) in the legacy order."""
        if self.order_mode == 'sorted':
            _nat = [self._pop_native.get(i, self._pop.names[i]) for i in range(len(self._pop))]
            try:                                   # numeric populations sort numerically, as the old frame did
                pops = sorted(range(len(_nat)), key=lambda i: _nat[i])
            except TypeError:
                pops = sorted(range(len(_nat)), key=lambda i: str(_nat[i]))
            models = self._model.names
            try:
                mrank = {i: r for r, i in enumerate(sorted(range(len(models)), key=lambda i: models[i]))}
            except TypeError:
                mrank = {i: r for r, i in enumerate(sorted(range(len(models)), key=lambda i: str(models[i])))}
            prank = {p: r for r, p in enumerate(pops)}
            nm = max(1, len(models))
            seq = {pm: prank[pm[0]] * nm + mrank[pm[1]] for pm in self._seen_seq}
        else:
            seq = dict(self._seen_seq)
        best_all, best_pop = {}, {}
        for (p, m), q in seq.items():
            if q < best_all.get(m, 1 << 62):
                best_all[m] = q
            lab = pop_label_of[p]
            d = best_pop.setdefault(lab, {})
            if q < d.get(m, 1 << 62):
                d[m] = q
        names = self._model.names
        all_order = [names[m] for m, _ in sorted(best_all.items(), key=lambda kv: kv[1])]
        per_pop = {lab: [names[m] for m, _ in sorted(d.items(), key=lambda kv: kv[1])]
                   for lab, d in best_pop.items()}
        return all_order, per_pop

    def reduce_groups(self, *, population_map=None, select=None, scope_model_order_by_population=True, restrict=None):
        """The per-key marker-pair tables circos draws from.

        population_map : optional callable str -> str applied to every
            population label (the 'between'-scenario ``'train->test'`` ->
            ``'test'`` rewrite); labels that collide are merged by summing
            their (sum, count) BEFORE the mean, exactly as before.
        select : None for the full per-key tables, or a dict
            ``{'mode': 'percentage'|'count', 'top_percent': .., 'top_count': ..}``
            to keep only the rows circos would draw (the tables then carry
            ``df.attrs['top_selected'] = True`` so the selection is not applied
            a second time).
        restrict : None (default - nothing changes) or a dict ``{'mask_for': callable,
            'match': 'either'|'both'}``. ``mask_for(phenotype_name, marker_names)`` returns a
            boolean array over ``marker_names`` (the store's marker dictionary): True = the marker
            lies in a region of interest. Only pairs with an end ('either') / both ends ('both') on
            such a marker are kept, BEFORE `select` ranks them - so 'top N%' / 'top M' is taken over
            the relevant pairs only. Used by circos' "only links involving the highlighted regions".

        Returns ``(groups, model_order)``: ``groups`` is
        ``{(population_key, model, phenotype): DataFrame[marker1, marker2, value]}``
        with ``population_key`` either 'all' or the (mapped) population label;
        ``model_order`` ``{population_key: [model, ...]}`` (only 'all' when
        `scope_model_order_by_population` is False) - the same shapes
        ``circos_plot._precompute_interaction_groups()`` always returned.
        """
        if self.empty:
            return {}, {}
        self._finalize_resident()
        pop_names = self._pop.names
        pop_label_of = {i: (population_map(p) if population_map else p) for i, p in enumerate(pop_names)}
        all_order, per_pop = self._model_orders(pop_label_of)
        model_order = {'all': all_order}
        if scope_model_order_by_population:
            model_order.update(per_pop)

        mnames = np.empty(len(self._marker), dtype=object)
        mnames[:] = self._marker.names
        rank = _lexical_rank(self._marker.names)

        # kgrp ids grouped by (model, phenotype) -> {mapped pop label: [kid, ...]}
        units = {}
        for kid, (p, m, f) in enumerate(self._kgrp_keys):
            units.setdefault((m, f), {}).setdefault(pop_label_of[p], []).append(kid)

        groups = {}
        member_cache = {}
        for (m, f), by_pop in units.items():
            model_name, pheno_name = self._model.names[m], self._pheno.names[f]
            member = None
            if restrict is not None:
                if f not in member_cache:
                    member_cache[f] = np.asarray(restrict['mask_for'](pheno_name, mnames), dtype=bool)
                member = (member_cache[f], restrict.get('match', 'either'))
            per_kid = {}
            for kids in by_pop.values():
                for kid in kids:
                    per_kid[kid] = self._rows_for(kid)
            # population-level keys
            for label, kids in by_pop.items():
                if len(kids) == 1:
                    a, b, s, c = per_kid[kids[0]]
                else:
                    cat = [np.concatenate([per_kid[k][i] for k in kids]) for i in range(4)]
                    _, a, b, s, c = _reduce_rows(np.zeros(cat[0].shape[0], np.int32), *cat)
                groups[(label, model_name, pheno_name)] = self._emit(a, b, s, c, mnames, rank, select, member)
            # cross-population 'all' key
            if len(per_kid) == 1:
                a, b, s, c = next(iter(per_kid.values()))
            else:
                cat = [np.concatenate([r[i] for r in per_kid.values()]) for i in range(4)]
                _, a, b, s, c = _reduce_rows(np.zeros(cat[0].shape[0], np.int32), *cat)
            groups[('all', model_name, pheno_name)] = self._emit(a, b, s, c, mnames, rank, select, member)
            per_kid.clear()
        return groups, model_order

    @staticmethod
    def _emit(m1, m2, s, c, mnames, rank, select, member=None):
        if member is not None:
            flags, match = member
            keep = (flags[m1] & flags[m2]) if match == 'both' else (flags[m1] | flags[m2])
            m1, m2, s, c = m1[keep], m2[keep], s[keep], c[keep]
        values = _mean(s, c)
        if select is None:
            order = np.lexsort((rank[m2], rank[m1]))
            frame = pd.DataFrame({'marker1': mnames[m1[order]], 'marker2': mnames[m2[order]],
                                  'value': values[order]})
            return frame
        idx = select_top_indices(values, m1, m2, rank, mode=select.get('mode', 'percentage'),
                                 top_percent=select.get('top_percent'), top_count=select.get('top_count'))
        frame = pd.DataFrame({'marker1': mnames[m1[idx]], 'marker2': mnames[m2[idx]], 'value': values[idx]})
        frame.attrs['top_selected'] = True
        return frame

    # ---- compatibility -----------------------------------------------------
    def to_frame(self):
        """Full DataFrame in the old ``aggregate_marker_pair_sums()`` shape
        (population, model, phenotype, marker1, marker2, value_sum,
        value_count) - NOT memory-bounded, for tests / small tables only."""
        if self.empty:
            return pd.DataFrame(columns=['population', 'model', 'phenotype', 'marker1', 'marker2',
                                         'value_sum', 'value_count'])
        self._finalize_resident()
        mnames = np.empty(len(self._marker), dtype=object); mnames[:] = self._marker.names
        out = []
        for kid, (p, m, f) in enumerate(self._kgrp_keys):
            a, b, s, c = self._rows_for(kid)
            out.append(pd.DataFrame({'population': self._pop.names[p], 'model': self._model.names[m],
                                     'phenotype': self._pheno.names[f], 'marker1': mnames[a],
                                     'marker2': mnames[b], 'value_sum': s, 'value_count': c}))
        return pd.concat(out, ignore_index=True)


def store_from_frame(df, *, order_mode='appearance', max_resident_rows=None):
    """`PairSumStore` from a raw (one row per observation) or pre-aggregated
    ('value_sum'/'value_count') interaction / attention DataFrame."""
    store = PairSumStore(order_mode=order_mode, max_resident_rows=max_resident_rows)
    if df is not None and df.shape[0]:
        store.ingest(df)
    return store
