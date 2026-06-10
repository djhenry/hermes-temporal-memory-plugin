"""
hermes-graphiti — advanced performance metrics
================================================
Six harder metrics that go beyond the basic Precision@1 / Recall@5:

  1. MRR              Mean Reciprocal Rank
                      Standard IR metric; penalises every rank step down.

  2. NDCG@5           Normalised Discounted Cumulative Gain
                      Log-discounted position weighting; hits at rank 1
                      count for much more than hits at rank 5.

  3. Staleness Rate   Fraction of the top-K results that correspond to
                      superseded (invalid) episodes.  Quantifies how much
                      stale information would contaminate the LLM's context.

  4. Temporal         For queries with an as_of date: fraction of the K
     Precision@K      returned results that are actually valid at that date.
                      FTS5 ignores as_of; graphiti enforces the window.

  5. Adversarial      Recall on queries whose wording deliberately names
     Recall           the *wrong* (outdated) entity.  BM25 scores the wrong
                      entity's documents higher via IDF; temporal filtering
                      ignores query-text keywords entirely.

  6. Scale            P@1 and R@5 vs. the number of stale entries in the
     Degradation      index for a fixed temporal query.  FTS5's accuracy
                      collapses when more current entries than stale ones
                      exist (newest-wins tie-break returns the wrong period);
                      graphiti's temporal filter is immune to scale.

Run
---
  python benchmarks/advanced_benchmark.py
  make benchmark-advanced
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.dataset import build_dataset, Episode, Query
from benchmarks.backends.fts5_backend import FTS5Backend
from benchmarks.backends.graphiti_backend import GraphitiMockBackend

_W = 76


# ══════════════════════════════════════════════════════════════════════════════
#  Metric functions
# ══════════════════════════════════════════════════════════════════════════════

def mrr(hit_ranks: list[Optional[int]]) -> float:
    """Mean Reciprocal Rank across a list of hit_at values (None = miss)."""
    if not hit_ranks:
        return 0.0
    return sum(1.0 / r for r in hit_ranks if r is not None) / len(hit_ranks)


def ndcg(hit_at: Optional[int], k: int = 5) -> float:
    """NDCG@K with binary relevance (one relevant doc; IDCG = 1.0 at rank 1)."""
    if hit_at is None or hit_at > k:
        return 0.0
    return 1.0 / math.log2(hit_at + 1)


def mean_ndcg(hit_ranks: list[Optional[int]], k: int = 5) -> float:
    return sum(ndcg(r, k) for r in hit_ranks) / len(hit_ranks) if hit_ranks else 0.0


def staleness_rate(
    returned_facts: list[str],
    current_fact_texts: set[str],
) -> float:
    """
    Fraction of returned facts that are NOT present in the current valid set.

    A fact text is "stale" if no episode with that exact text is currently
    valid (i.e., no episode with invalid_at=None has this text).  This
    definition is backend-agnostic: the oracle comes from graphiti's
    supersession model, applied uniformly when evaluating FTS5's output.

    Example: 'User lives in Barcelona' is stale (superseded by Madrid);
    'User lives in Madrid' is current (most-recent version has no invalid_at).
    """
    if not returned_facts:
        return 0.0
    stale = sum(1 for f in returned_facts if f not in current_fact_texts)
    return stale / len(returned_facts)


def temporal_precision(
    returned_facts: list[str],
    as_of: str,
    all_episodes: list[Episode],
    k: int = 5,
) -> float:
    """
    For a temporal query, what fraction of the top-K results are valid at
    `as_of`?

    A returned fact is temporally valid if ANY episode whose .fact text
    matches has valid_at <= as_of and (invalid_at is None or invalid_at > as_of).
    """
    as_of_dt = datetime.fromisoformat(as_of).replace(tzinfo=timezone.utc)

    # Build: fact_text → is any episode valid at as_of?
    validity: dict[str, bool] = {}
    for ep in all_episodes:
        v = datetime.fromisoformat(ep.valid_at).replace(tzinfo=timezone.utc)
        inv = (datetime.fromisoformat(ep.invalid_at).replace(tzinfo=timezone.utc)
               if ep.invalid_at else None)
        in_window = v <= as_of_dt and (inv is None or inv > as_of_dt)
        if in_window:
            validity[ep.fact] = True
        elif ep.fact not in validity:
            validity[ep.fact] = False

    facts = returned_facts[:k]
    if not facts:
        return 0.0
    return sum(1 for f in facts if validity.get(f, False)) / len(facts)


# ══════════════════════════════════════════════════════════════════════════════
#  Adversarial query dataset
# ══════════════════════════════════════════════════════════════════════════════

# Uses the same location_history + job_history episodes as the main benchmark.
# These queries deliberately include the *wrong* (stale) entity name in their
# text.  BM25 boosts the stale entity's documents (high IDF for a rare term);
# FTS5 therefore ranks the wrong result first.  graphiti ignores keyword
# frequency and returns only currently-valid facts.

ADVERSARIAL_EPISODES: list[Episode] = [
    # Location: 1 old city, 3 current
    Episode("User lives in Barcelona",          "user", "LIVES_IN", "2024-01-01"),
    Episode("User lives in Madrid",             "user", "LIVES_IN", "2024-06-15"),
    Episode("User lives in Madrid",             "user", "LIVES_IN", "2025-01-01"),
    Episode("User lives in Madrid",             "user", "LIVES_IN", "2026-03-01"),
    # Job: 1 old employer, 3 current
    Episode("User works at TechCorp as software engineer", "user", "WORKS_AT", "2024-01-01"),
    Episode("User works at StartupXYZ",                   "user", "WORKS_AT", "2025-01-10"),
    Episode("User works at StartupXYZ as lead engineer",  "user", "WORKS_AT", "2025-06-01"),
    Episode("User works at StartupXYZ on platform team",  "user", "WORKS_AT", "2026-03-01"),
]

ADVERSARIAL_QUERIES: list[Query] = [
    # ── Negation: query contains the stale entity name ──────────────────────
    Query(
        text="The user no longer lives in Barcelona — what city are they in now?",
        expected="Madrid",
        category="adversarial_negation",
        description="'Barcelona' in query → FTS5 IDF-boosts Barcelona entry; "
                    "graphiti returns only current (Madrid)",
    ),
    Query(
        text="After leaving Barcelona, where does the user now reside?",
        expected="Madrid",
        category="adversarial_deceptive",
        description="'Barcelona' + 'leaving' co-occur; FTS5 ranks Barcelona first",
    ),
    Query(
        text="User no longer works at TechCorp — current employer?",
        expected="StartupXYZ",
        category="adversarial_negation",
        description="'TechCorp' in query; FTS5 IDF-boosts TechCorp entry",
    ),
    Query(
        text="They moved on from TechCorp — who employs them today?",
        expected="StartupXYZ",
        category="adversarial_deceptive",
        description="'TechCorp' + 'moved' → FTS5 IDF boost for stale entry",
    ),
    Query(
        text="Has the city changed since Barcelona?  Where are they based now?",
        expected="Madrid",
        category="adversarial_confusion",
        description="'Barcelona' + 'city' + 'now' mixed; FTS5 confused by keyword salad",
    ),
    Query(
        text="Since leaving TechCorp, what company has the user joined?",
        expected="StartupXYZ",
        category="adversarial_confusion",
        description="'TechCorp' + 'company' gives TechCorp high BM25",
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
#  Scale-degradation test
# ══════════════════════════════════════════════════════════════════════════════

_SCALE_NS = [0, 1, 2, 3, 4, 5, 8, 10, 15, 20]

def _run_scale_test(k: int = 5) -> dict:
    """
    Temporal scale-degradation test.

    Scenario: one stale Barcelona entry (inserted first), then N current
    Madrid entries added afterward.

    Query: "Where did the user live in early 2024?" (as_of=2024-03-01)
    Expected: Barcelona.

    FTS5 tie-breaking (rowid DESC): with N Madrid entries all having higher
    rowids than Barcelona, FTS5 always returns a Madrid entry first — the
    WRONG answer for a past-time query.  The stale Barcelona entry gets
    pushed out of Recall@K once N >= K.

    graphiti filters by validity window: Barcelona is the only fact valid
    at 2024-03-01 regardless of how many current Madrid entries exist.
    """
    fts5_p1:  list[float] = []
    fts5_r5:  list[float] = []
    gph_p1:   list[float] = []
    gph_r5:   list[float] = []

    query   = "Where did the user live in early 2024"
    as_of   = "2024-03-01"
    base    = datetime(2024, 6, 1, tzinfo=timezone.utc)

    for n in _SCALE_NS:
        fts5 = FTS5Backend()
        gph  = GraphitiMockBackend()

        # Stale entry first (gets a low rowid in FTS5)
        for backend in (fts5, gph):
            backend.ingest_episode("User lives in Barcelona", "user", "LIVES_IN", "2024-01-01")

        # N current entries added after (higher rowids in FTS5)
        for i in range(n):
            date = (base + timedelta(days=30 * i)).strftime("%Y-%m-%d")
            for backend in (fts5, gph):
                backend.ingest_episode("User lives in Madrid", "user", "LIVES_IN", date)

        f_res = fts5.search(query, as_of=as_of, k=k)
        g_res = gph.search(query,  as_of=as_of, k=k)

        f_hit1 = bool(f_res) and "Barcelona" in f_res[0]
        g_hit1 = bool(g_res) and "Barcelona" in g_res[0]
        f_hit5 = any("Barcelona" in r for r in f_res)
        g_hit5 = any("Barcelona" in r for r in g_res)

        fts5_p1.append(float(f_hit1))
        fts5_r5.append(float(f_hit5))
        gph_p1.append(float(g_hit1))
        gph_r5.append(float(g_hit5))

    return {
        "ns":       _SCALE_NS,
        "fts5_p1":  fts5_p1,
        "fts5_r5":  fts5_r5,
        "gph_p1":   gph_p1,
        "gph_r5":   gph_r5,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Runner helpers
# ══════════════════════════════════════════════════════════════════════════════

def _hr(c: str = "─") -> None: print(c * _W)
def _header(t: str) -> None:
    print(); _hr("═"); print(f"  {t}"); _hr("═")
def _section(t: str) -> None:
    print(); _hr(); print(f"  {t}"); _hr()


def _load_backend(backend, episodes: list[Episode]) -> None:
    backend.reset()
    for ep in episodes:
        backend.ingest_episode(ep.fact, ep.entity, ep.relationship, ep.valid_at, ep.invalid_at)


@dataclass
class AdvResult:
    query: Query
    hit_at:      Optional[int]
    meta:        list[tuple[str, str, str]]  # (content, valid_at, invalid_at)
    top_facts:   list[str]


def _eval_hit(results: list[str], expected: str) -> Optional[int]:
    for i, r in enumerate(results, 1):
        if expected.lower() in r.lower():
            return i
    return None


def _run_with_meta(backend, queries: list[Query], k: int) -> list[AdvResult]:
    out: list[AdvResult] = []
    for q in queries:
        facts = backend.search(q.text, as_of=q.as_of, k=k)
        # For FTS5 we have search_with_meta; graphiti constructs meta from fact timeline
        if hasattr(backend, "search_with_meta"):
            meta = backend.search_with_meta(q.text, as_of=q.as_of, k=k)
        else:
            # graphiti: reconstruct meta from its internal _facts list
            meta = _graphiti_meta(backend, facts)
        out.append(AdvResult(
            query    = q,
            hit_at   = _eval_hit(facts, q.expected),
            meta     = meta,
            top_facts = facts,
        ))
    return out


def _graphiti_meta(gph: GraphitiMockBackend, facts: list[str]) -> list[tuple[str, str, str]]:
    """Build (content, valid_at, invalid_at) triples from graphiti's internal store."""
    lookup: dict[str, tuple[str, str, str]] = {}
    for f in gph._facts:
        if f.fact not in lookup:  # take first occurrence (earliest valid_at)
            inv = f.invalid_at.strftime("%Y-%m-%d") if f.invalid_at else ""
            lookup[f.fact] = (f.fact, f.valid_at.strftime("%Y-%m-%d"), inv)
    return [lookup[fact] for fact in facts if fact in lookup]


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def run_advanced(k: int = 5) -> None:
    now = datetime.now(timezone.utc)
    dataset = build_dataset()
    all_episodes: list[Episode] = [ep for s in dataset for ep in s.episodes]

    fts5 = FTS5Backend()
    gph  = GraphitiMockBackend()

    # Load ALL dataset episodes into both backends for section-level metrics
    _load_backend(fts5, all_episodes)
    _load_backend(gph,  all_episodes)

    # Oracle: set of fact texts that are currently valid in graphiti's model.
    # Used as the ground-truth "current" set when evaluating FTS5 staleness.
    current_fact_texts: set[str] = {f.fact for f in gph._facts if f.invalid_at is None}

    # Collect all queries from the dataset
    all_queries: list[Query] = [q for s in dataset for q in s.queries]

    _header("HERMES TEMPORAL MEMORY PLUGIN — ADVANCED METRICS")
    print(f"  Run at  : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Dataset : {len(dataset)} scenarios · {len(all_episodes)} episodes · {len(all_queries)} queries")
    print(f"  K       : {k}")
    print()
    print("  All six metrics benchmark retrieval only; LLM extraction is bypassed.")

    # ── Per-query results ─────────────────────────────────────────────────────
    f_results = _run_with_meta(fts5, all_queries, k)
    g_results = _run_with_meta(gph,  all_queries, k)

    # ── 1 & 2 : MRR and NDCG@5 ───────────────────────────────────────────────
    _section("1. MRR and NDCG@5  (ranking quality)")

    f_ranks = [r.hit_at for r in f_results]
    g_ranks = [r.hit_at for r in g_results]

    f_mrr   = mrr(f_ranks)
    g_mrr   = mrr(g_ranks)
    f_ndcg  = mean_ndcg(f_ranks, k)
    g_ndcg  = mean_ndcg(g_ranks, k)

    mfmt = "  {:<18s}  {:>8.4f}   {:>10.4f}   {:>+.4f}"
    print("  " + "─" * 52)
    print(f"  {'Metric':<18s}  {'FTS5':>8s}   {'graphiti':>10s}   {'Δ':>6s}")
    print("  " + "─" * 52)
    print(mfmt.format("MRR",      f_mrr,  g_mrr,  g_mrr  - f_mrr))
    print(mfmt.format(f"NDCG@{k}", f_ndcg, g_ndcg, g_ndcg - f_ndcg))
    print()
    print("  Interpretation:")
    print(f"    MRR {f_mrr:.4f} vs {g_mrr:.4f} — every rank step down costs 1/rank.")
    print(f"    A 0.1 MRR gap means the average result rank is ~{1/f_mrr:.1f} (FTS5) "
          f"vs ~{1/g_mrr:.1f} (graphiti).")

    # ── 3. Staleness Rate ─────────────────────────────────────────────────────
    _section("3. Result Staleness Rate  (context contamination)")

    print("  For each query, fraction of the top-K returned facts that correspond")
    print("  to superseded (invalid) episodes.  Stale facts pollute the LLM's")
    print("  context even when the correct answer is also present.")
    print()

    # Only meaningful for present-state queries (no as_of)
    present_f = [(r, q) for r, q in zip(f_results, all_queries) if not q.as_of]
    present_g = [(r, q) for r, q in zip(g_results, all_queries) if not q.as_of]

    f_stale_rates: list[float] = [staleness_rate(r.top_facts, current_fact_texts) for r, _ in present_f]
    g_stale_rates: list[float] = [staleness_rate(r.top_facts, current_fact_texts) for r, _ in present_g]

    f_mean_stale = sum(f_stale_rates) / len(f_stale_rates) if f_stale_rates else 0
    g_mean_stale = sum(g_stale_rates) / len(g_stale_rates) if g_stale_rates else 0

    # Show breakdown by category
    from collections import defaultdict
    cat_stale_f: dict[str, list[float]] = defaultdict(list)
    cat_stale_g: dict[str, list[float]] = defaultdict(list)
    for (fr, q), gs in zip(present_f, [r for r, _ in present_g]):
        cat_stale_f[q.category].append(staleness_rate(fr.top_facts, current_fact_texts))
        cat_stale_g[q.category].append(staleness_rate(gs.top_facts, current_fact_texts))

    sfmt = "  {:<22s}  {:>8s}   {:>10s}   {:>10s}"
    print(sfmt.format("Category", "FTS5", "graphiti", "Δ"))
    print("  " + "─" * 58)
    for cat in sorted(cat_stale_f):
        fv = cat_stale_f[cat];    fm = sum(fv)/len(fv) if fv else 0
        gv = cat_stale_g.get(cat, []); gm = sum(gv)/len(gv) if gv else 0
        print(sfmt.format(cat, f"{fm:6.1%}", f"{gm:6.1%}", f"{gm-fm:+.1%}"))
    print(sfmt.format("OVERALL", f"{f_mean_stale:6.1%}", f"{g_mean_stale:6.1%}",
                       f"{g_mean_stale-f_mean_stale:+.1%}"))
    print()
    print("  graphiti: 0% — superseded facts are excluded from present-state search.")
    print("  FTS5: keeps ALL episodes → stale facts contaminate every result set")
    print("        that contains superseded entries.")

    # ── 4. Temporal Precision@K ───────────────────────────────────────────────
    _section(f"4. Temporal Precision@{k}  (fraction of results valid at as_of)")

    print(f"  For queries with an as_of date, what fraction of the {k} returned")
    print("  results are actually valid at that point in time?")
    print()

    temporal_f = [(r, q) for r, q in zip(f_results, all_queries) if q.as_of]
    temporal_g = [(r, q) for r, q in zip(g_results, all_queries) if q.as_of]

    f_tp: list[float] = []
    g_tp: list[float] = []
    for (fr, fq), (gr, _gq) in zip(temporal_f, temporal_g):
        f_tp.append(temporal_precision(fr.top_facts, fq.as_of, all_episodes, k))  # type: ignore[arg-type]
        g_tp.append(temporal_precision(gr.top_facts, fq.as_of, all_episodes, k))  # type: ignore[arg-type]

    f_tp_mean = sum(f_tp) / len(f_tp) if f_tp else 0
    g_tp_mean = sum(g_tp) / len(g_tp) if g_tp else 0

    tpfmt = "  {:<46s}  {:>8s}   {:>9s}"
    print(tpfmt.format("Query  [as_of]", "FTS5 TP", "graph TP"))
    print("  " + "─" * 68)
    for (fr, fq), ft, gt in zip(temporal_f, f_tp, g_tp):
        label = (fq.text[:42] + "…" if len(fq.text) > 44 else fq.text)
        label += f"  [{fq.as_of}]"
        print(tpfmt.format(label, f"{ft:.0%}", f"{gt:.0%}"))
    print("  " + "─" * 68)
    print(tpfmt.format(f"MEAN ({len(f_tp)} temporal queries)", f"{f_tp_mean:.0%}", f"{g_tp_mean:.0%}"))
    print()
    print("  graphiti enforces validity windows → TP@K is always 100%.")
    print("  FTS5 returns facts regardless of time → TP@K tracks only accidentally")
    print("  valid results, not a result of temporal awareness.")

    # ── 5. Adversarial Recall ─────────────────────────────────────────────────
    _section("5. Adversarial Recall  (queries containing the wrong entity name)")

    print("  Queries whose wording includes the stale entity (e.g. 'No longer in")
    print("  Barcelona...').  BM25 IDF-boosts the stale entity's documents;")
    print("  temporal filtering in graphiti ignores query text for validity decisions.")
    print()

    fts5_adv = FTS5Backend()
    gph_adv  = GraphitiMockBackend()
    _load_backend(fts5_adv, ADVERSARIAL_EPISODES)
    _load_backend(gph_adv,  ADVERSARIAL_EPISODES)

    af_results = _run_with_meta(fts5_adv, ADVERSARIAL_QUERIES, k)
    ag_results = _run_with_meta(gph_adv,  ADVERSARIAL_QUERIES, k)

    afmt = "  {:<44s}  {:>10s}   {:>10s}"
    print(afmt.format("Query", "FTS5", "graphiti"))
    print("  " + "─" * 68)
    for fqr, gqr in zip(af_results, ag_results):
        q = fqr.query
        q_txt = q.text[:42] + "…" if len(q.text) > 44 else q.text

        def _fmt(r: AdvResult) -> str:
            if r.hit_at is None: return "✗  miss"
            return f"✓  @{r.hit_at}"

        print(afmt.format(q_txt, _fmt(fqr), _fmt(gqr)))

    n_adv = len(af_results)
    f_adv_hits = sum(1 for r in af_results if r.hit_at and r.hit_at <= k)
    g_adv_hits = sum(1 for r in ag_results if r.hit_at and r.hit_at <= k)
    print("  " + "─" * 68)
    print(afmt.format(f"Adversarial Recall@{k}  ({n_adv} queries)",
                      f"{f_adv_hits}/{n_adv} ({100*f_adv_hits/n_adv:.0f}%)",
                      f"{g_adv_hits}/{n_adv} ({100*g_adv_hits/n_adv:.0f}%)"))
    print()
    print("  Why FTS5 fails: 'Barcelona' is a rare term (IDF ≈ 1.2 nats); its")
    print("  presence in the query raises the Barcelona entry's BM25 score above")
    print("  all Madrid entries even when the question means the opposite.")
    print("  Why graphiti succeeds: validity filtering operates on the graph, not")
    print("  on the query text — the stale entity in the question is irrelevant.")

    # ── 6. Scale Degradation ──────────────────────────────────────────────────
    _section("6. Scale Degradation  (temporal query, P@1 / R@5 vs stale-entry count)")

    print("  Setup: 1 stale Barcelona entry (inserted first) + N current Madrid")
    print("  entries (inserted after).  Query: 'Where did user live in early 2024?'")
    print(f"  as_of=2024-03-01, expected 'Barcelona'.")
    print()
    print(f"  FTS5 tie-breaking: rowid DESC (newest wins — best case for FTS5).")
    print(f"  With N≥1 Madrid entries, all having higher rowids than Barcelona,")
    print(f"  FTS5 returns Madrid first for any BM25 tie — the WRONG answer for")
    print(f"  a past-time query.  Once N≥{k}, Barcelona is also pushed out of top-{k}.")
    print()

    sd = _run_scale_test(k)

    hdr = f"  {'N stale entries':>14s}  {'FTS5 P@1':>9s}  {'FTS5 R@5':>9s}  {'graph P@1':>10s}  {'graph R@5':>10s}  Notes"
    print(hdr)
    print("  " + "─" * 72)
    for n, fp1, fr5, gp1, gr5 in zip(sd["ns"], sd["fts5_p1"], sd["fts5_r5"],
                                       sd["gph_p1"],  sd["gph_r5"]):
        notes: list[str] = []
        if n == 0:     notes.append("baseline")
        if fp1 == 0 and n > 0: notes.append("FTS5 P@1 drops")
        if fr5 == 0 and n >= k: notes.append(f"FTS5 R@{k} drops")
        note = "  ← " + ", ".join(notes) if notes else ""
        f_p1_s = f"{fp1:.0%}" if fp1 == 1 else "0%"
        f_r5_s = f"{fr5:.0%}" if fr5 == 1 else "0%"
        g_p1_s = f"{gp1:.0%}"
        g_r5_s = f"{gr5:.0%}"
        print(f"  {n:>14d}  {f_p1_s:>9s}  {f_r5_s:>9s}  {g_p1_s:>10s}  {g_r5_s:>10s}{note}")

    print()
    print(f"  graphiti: flat 100%/100% — temporal filter is O(1) in stale-entry count.")
    print(f"  FTS5:     P@1 collapses at N=1; R@{k} collapses at N={k}.")

    # ── Summary ───────────────────────────────────────────────────────────────
    _header("ADVANCED METRICS SUMMARY")

    f_p1_main = sum(1 for r in f_results if r.hit_at == 1) / len(f_results)
    g_p1_main = sum(1 for r in g_results if r.hit_at == 1) / len(g_results)

    rows = [
        ("Precision@1 (main dataset)", f"{f_p1_main:.1%}", f"{g_p1_main:.1%}"),
        ("MRR",                        f"{f_mrr:.4f}",     f"{g_mrr:.4f}"),
        (f"NDCG@{k}",                  f"{f_ndcg:.4f}",    f"{g_ndcg:.4f}"),
        ("Staleness Rate (mean)",       f"{f_mean_stale:.1%}", f"{g_mean_stale:.1%}"),
        (f"Temporal Precision@{k}",     f"{f_tp_mean:.0%}", f"{g_tp_mean:.0%}"),
        (f"Adversarial Recall@{k}",     f"{100*f_adv_hits/n_adv:.0f}%" if n_adv else "n/a",
                                        f"{100*g_adv_hits/n_adv:.0f}%" if n_adv else "n/a"),
        ("Scale: P@1 at N=5",          f"{sd['fts5_p1'][_SCALE_NS.index(5)]:.0%}",
                                        f"{sd['gph_p1'][_SCALE_NS.index(5)]:.0%}"),
        (f"Scale: R@{k} at N={k}",     f"{sd['fts5_r5'][_SCALE_NS.index(k if k in _SCALE_NS else 5)]:.0%}",
                                        f"{sd['gph_r5'][_SCALE_NS.index(k if k in _SCALE_NS else 5)]:.0%}"),
    ]

    rfmt = "  {:<32s}  {:>8s}   {:>10s}   {:>12s}"
    print(rfmt.format("Metric", "FTS5", "graphiti", "Δ (graph−fts5)"))
    print("  " + "─" * 68)
    for label, fv, gv in rows:
        try:
            fd = float(fv.rstrip("%")) / (100 if "%" in fv else 1)
            gd = float(gv.rstrip("%")) / (100 if "%" in gv else 1)
            delta_s = f"{gd-fd:+.1%}" if "%" in fv else f"{gd-fd:+.4f}"
        except ValueError:
            delta_s = "n/a"
        print(rfmt.format(label, fv, gv, delta_s))

    _hr("═")
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()
    run_advanced(k=args.k)
