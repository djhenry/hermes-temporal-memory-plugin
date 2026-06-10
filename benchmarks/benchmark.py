"""
hermes-graphiti performance benchmark
======================================
Compares hermes-graphiti (temporal knowledge-graph memory) against
Hermes's built-in FTS5 SQLite memory on a curated recall dataset.

Metrics
-------
  Precision@1   Is the #1 result the correct answer?  Primary UX metric.
  Recall@3/5    Is the correct answer anywhere in top-3 / top-5?

What this tests
---------------
  current_stable   Stable facts; both backends should perform equally.
  current_updated  Fact has been superseded; correct answer is the NEW value.
                   — graphiti advantage: superseded facts excluded from search.
                     FTS5 keeps all history; old entries may rank above the new one.
  temporal         Point-in-time query (as_of date in the past).
                   — graphiti advantage: validity-window filter.
                     FTS5 ignores as_of; with more current entries than historical
                     ones, FTS5 ranks the WRONG (current) answer first.
  disambiguation   Same first name, different people.
  multi_hop        Answer requires combining ≥2 related facts.

What this does NOT test
-----------------------
  LLM extraction quality — pre-extracted facts are fed directly, isolating
  retrieval accuracy.  Real graphiti adds LLM-driven entity / relationship
  extraction on top of what this benchmark measures.

  Absolute latency at scale — both backends are in-memory mocks.  A live
  Neo4j / FalkorDB backend shows similar recall characteristics but higher
  absolute latency.  See docs/temporal-memory.md for production latency targets.

Run
---
  python benchmarks/benchmark.py            # standard
  python benchmarks/benchmark.py --verbose  # show per-query top results
  make benchmark
  make benchmark-verbose
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.dataset import build_dataset, Query, Scenario
from benchmarks.backends.fts5_backend import FTS5Backend
from benchmarks.backends.graphiti_backend import GraphitiMockBackend


# ── Result model ──────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    query: Query
    top_results: list[str]
    hit_at: Optional[int]   # 1-based rank of first hit, None = miss

    @property
    def hit1(self) -> bool:
        return self.hit_at == 1

    @property
    def hit3(self) -> bool:
        return self.hit_at is not None and self.hit_at <= 3

    @property
    def hit5(self) -> bool:
        return self.hit_at is not None and self.hit_at <= 5


@dataclass
class BackendResults:
    backend_name: str
    short_name: str
    scenario_results: list[list[QueryResult]] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


# ── Evaluation helpers ────────────────────────────────────────────────────────

def _evaluate(top_k: list[str], expected: str) -> Optional[int]:
    """Return 1-based rank of first result containing expected, else None."""
    exp_lower = expected.lower()
    for i, result in enumerate(top_k, start=1):
        if exp_lower in result.lower():
            return i
    return None


def _run_backend(backend, scenarios: list[Scenario], k: int = 5) -> BackendResults:
    br = BackendResults(backend_name=backend.name, short_name=backend.short_name)
    for scenario in scenarios:
        backend.reset()
        for ep in scenario.episodes:
            backend.ingest_episode(
                fact=ep.fact,
                entity=ep.entity,
                relationship=ep.relationship,
                valid_at=ep.valid_at,
                invalid_at=ep.invalid_at,
            )
        qresults: list[QueryResult] = []
        for q in scenario.queries:
            top = backend.search(q.text, as_of=q.as_of, k=k)
            qresults.append(QueryResult(query=q, top_results=top, hit_at=_evaluate(top, q.expected)))
        br.scenario_results.append(qresults)
    br.stats = backend.stats()
    return br


# ── Printing helpers ──────────────────────────────────────────────────────────

_W = 76

def _hr(char: str = "─") -> None:
    print(char * _W)

def _header(text: str) -> None:
    print()
    _hr("═")
    print(f"  {text}")
    _hr("═")

def _section(text: str) -> None:
    print()
    _hr()
    print(f"  {text}")
    _hr()

def _symbol(hit: bool) -> str:
    return "✓" if hit else "✗"

def _pct(hits: int, total: int) -> str:
    return f"{100*hits/total:5.1f}%" if total else "  n/a"

def _delta_pp(f: int, g: int, total: int) -> str:
    if not total:
        return "  n/a"
    d = 100 * (g - f) / total
    return f"{d:+.1f} pp"


# ── Main ──────────────────────────────────────────────────────────────────────

def run_benchmark(k: int = 5, verbose: bool = False) -> None:
    dataset = build_dataset()
    fts5 = FTS5Backend()
    gph  = GraphitiMockBackend()

    total_q  = sum(len(s.queries)  for s in dataset)
    total_ep = sum(len(s.episodes) for s in dataset)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")

    _header("HERMES TEMPORAL MEMORY PLUGIN — PERFORMANCE BENCHMARK")
    print(f"  Run at  : {ts}")
    print(f"  Dataset : {len(dataset)} scenarios · {total_ep} episodes · {total_q} queries")
    print(f"  K       : {k}  (Precision@1, Recall@3, Recall@{k} measured)")
    print()
    print(f"  Backends:")
    print(f"    baseline — {fts5.name}")
    print(f"    plugin   — {gph.name}")
    print()
    print("  Pre-extracted facts are fed directly to both backends.")
    print("  LLM extraction quality is NOT measured here.")

    fts5_br = _run_backend(fts5, dataset, k=k)
    gph_br  = _run_backend(gph,  dataset, k=k)

    # ── Per-scenario detail ───────────────────────────────────────────────────
    # cat_hits[cat][backend] = list of (hit1, hit5) pairs
    cat_p1:  dict[str, dict[str, list[bool]]] = {}   # Precision@1 per category
    cat_r5:  dict[str, dict[str, list[bool]]] = {}   # Recall@5 per category

    for si, scenario in enumerate(dataset):
        _section(f"Scenario {si+1}: {scenario.name}")
        print(f"  {scenario.description}")
        print(f"  Episodes: {len(scenario.episodes)}")
        print()

        f_qrs = fts5_br.scenario_results[si]
        g_qrs = gph_br.scenario_results[si]

        col_q = 44
        col_cat = 16
        col_r  = 14
        hdr = (f"  {'Query':<{col_q}}  {'Category':<{col_cat}}"
               f"  {'FTS5 (base)':>{col_r}}  {'graphiti':>{col_r}}")
        print(hdr)
        print("  " + "─" * (col_q + col_cat + col_r * 2 + 6))

        for fqr, gqr in zip(f_qrs, g_qrs):
            q = fqr.query
            q_txt = (q.text[:col_q-1] + "…") if len(q.text) > col_q else q.text

            cat_lbl = q.category
            if q.as_of:
                cat_lbl += f"*"  # asterisk means as_of set

            def _fmt(qr: QueryResult) -> str:
                if qr.hit_at is None:
                    return "✗  miss"
                return f"{_symbol(qr.hit5)}  @{qr.hit_at}"

            print(f"  {q_txt:<{col_q}}  {cat_lbl:<{col_cat}}"
                  f"  {_fmt(fqr):>{col_r}}  {_fmt(gqr):>{col_r}}")

            for backend_key, qr in [("fts5", fqr), ("graphiti", gqr)]:
                cat_p1.setdefault(q.category, {}).setdefault(backend_key, []).append(qr.hit1)
                cat_r5.setdefault(q.category, {}).setdefault(backend_key, []).append(qr.hit5)

            if verbose:
                print(f"    Expected : '{q.expected}'  as_of={q.as_of!r}")
                for backend_name, qr in [("FTS5    ", fqr), ("graphiti", gqr)]:
                    print(f"    {backend_name} top-{min(k,3)}:")
                    for i, r in enumerate(qr.top_results[:3], 1):
                        marker = " ◀ HIT" if q.expected.lower() in r.lower() else ""
                        print(f"       {i}. {r[:68]}{marker}")
                print()

    if verbose:
        print("  (* = temporal query with as_of date set)")

    # ── Summary ───────────────────────────────────────────────────────────────
    _header("SUMMARY")

    all_fts5 = [qr for srs in fts5_br.scenario_results for qr in srs]
    all_gph  = [qr for srs in gph_br.scenario_results  for qr in srs]
    total = len(all_fts5)

    f_p1 = sum(1 for qr in all_fts5 if qr.hit1)
    g_p1 = sum(1 for qr in all_gph  if qr.hit1)
    f_r3 = sum(1 for qr in all_fts5 if qr.hit3)
    g_r3 = sum(1 for qr in all_gph  if qr.hit3)
    f_r5 = sum(1 for qr in all_fts5 if qr.hit5)
    g_r5 = sum(1 for qr in all_gph  if qr.hit5)

    print(f"  Overall ({total} queries):")
    print()
    mfmt = "  {:<18s}  {:>8s}   {:>10s}   {:>12s}"
    print(mfmt.format("Metric", "FTS5", "graphiti", "Δ (graph−fts5)"))
    print("  " + "─" * 58)
    print(mfmt.format("Precision@1",  _pct(f_p1,total), _pct(g_p1,total), _delta_pp(f_p1,g_p1,total)))
    print(mfmt.format("Recall@3",     _pct(f_r3,total), _pct(g_r3,total), _delta_pp(f_r3,g_r3,total)))
    print(mfmt.format(f"Recall@{k}",  _pct(f_r5,total), _pct(g_r5,total), _delta_pp(f_r5,g_r5,total)))

    print()
    print("  By category  (Precision@1 / Recall@5):")
    print()
    cat_order = ["current_stable", "current_updated", "temporal", "disambiguation", "multi_hop"]
    all_cats = cat_order + [c for c in sorted(cat_p1) if c not in cat_order]

    cfmt = "  {:<22s}  {:>12s}   {:>14s}   {:>10s}   {}"
    print(cfmt.format("Category", "FTS5 @1/@5", "graphiti @1/@5", "Δ @1", "winner@1"))
    print("  " + "─" * 75)

    for cat in all_cats:
        if cat not in cat_p1:
            continue
        fp1 = cat_p1[cat].get("fts5", []);     fn1, ft = sum(fp1), len(fp1)
        gp1 = cat_p1[cat].get("graphiti", []); gn1, gt = sum(gp1), len(gp1)
        fr5 = cat_r5[cat].get("fts5", []);     fn5 = sum(fr5)
        gr5 = cat_r5[cat].get("graphiti", []); gn5 = sum(gr5)
        fts5_col = f"{_pct(fn1,ft)} / {_pct(fn5,ft)}"
        gph_col  = f"{_pct(gn1,gt)} / {_pct(gn5,gt)}"
        d  = (100*gn1/gt - 100*fn1/ft) if ft and gt else 0
        d_s = f"{d:+.1f} pp"
        winner = "graphiti" if d > 5 else ("fts5" if d < -5 else "tie")
        star = " *" if cat in ("temporal", "current_updated") else "  "
        print(cfmt.format(cat + star, fts5_col, gph_col, d_s, winner))

    print()
    print("  * = categories where graphiti has a structural advantage over FTS5")

    # ── Latency ───────────────────────────────────────────────────────────────
    _section("Latency  (in-memory mocks — algorithmic cost only)")

    fs = fts5_br.stats
    gs = gph_br.stats

    lfmt = "  {:<24s}  fts5: {:>8.1f} µs    graphiti: {:>8.1f} µs    ({:+.1f} µs)"
    def _lat(label: str, fk: str, gk: str) -> None:
        print(lfmt.format(label, fs[fk], gs[gk], gs[gk]-fs[fk]))

    _lat("Ingest (mean)",   "avg_ingest_us", "avg_ingest_us")
    _lat("Query  (mean)",   "avg_query_us",  "avg_query_us")
    _lat("Query  (p95)",    "p95_query_us",  "p95_query_us")

    print()
    print(f"  Facts stored:")
    print(f"    FTS5     : {fs['total_facts']} rows (append-only — all episodes retained)")
    gst = gph_br.stats
    print(f"    graphiti : {gst['total_facts']} total  "
          f"({gst['current_facts']} current · {gst['superseded_facts']} superseded/closed)")

    print()
    print("  Production targets (real backend, from docs/temporal-memory.md):")
    print("    prefetch  p95  < 1 s   (Neo4j, 10K episodes)")
    print("    sync_turn mean 0.5–2 s  (daemon thread, non-blocking)")
    print("    fact_history  < 200 ms")

    # ── Temporal tools demo ───────────────────────────────────────────────────
    _section("Temporal tools  (graphiti-only — no FTS5 equivalent)")

    demo = GraphitiMockBackend()
    from benchmarks.dataset import _location_history
    for ep in _location_history().episodes:
        demo.ingest_episode(ep.fact, ep.entity, ep.relationship, ep.valid_at, ep.invalid_at)

    print("  fact_history('user', 'LIVES_IN') — full validity timeline:")
    print()
    for entry in demo.fact_history("user", "LIVES_IN"):
        icon = "●" if entry["status"] == "current" else "○"
        print(f"    {icon} [{entry['valid_at']} → {entry['invalid_at']:>10s}]  {entry['fact']}")

    print()
    print("  temporal_search(as_of='2024-03-01')  — point-in-time snapshot:")
    for i, r in enumerate(demo.search("where does the user live", as_of="2024-03-01", k=3), 1):
        print(f"    {i}. {r}")

    print()
    print("  temporal_search(as_of=None)  — present state only:")
    for i, r in enumerate(demo.search("where does the user live", k=3), 1):
        print(f"    {i}. {r}")

    print()
    print("  FTS5 — same query, no temporal concept:")
    demo_fts5 = FTS5Backend()
    for ep in _location_history().episodes:
        demo_fts5.ingest_episode(ep.fact, ep.entity, ep.relationship, ep.valid_at, ep.invalid_at)
    fts5_results_demo = demo_fts5.search("where does the user live", k=4)
    for i, r in enumerate(fts5_results_demo, 1):
        note = "← STALE" if "Barcelona" in r else "← current"
        print(f"    {i}. {r}  {note}")

    # ── Key findings ──────────────────────────────────────────────────────────
    _section("Key findings")

    def _p1(cat: str, backend: str) -> tuple[int, int]:
        hits = cat_p1.get(cat, {}).get(backend, [])
        return sum(hits), len(hits)

    t_f1, t_ft = _p1("temporal",        "fts5")
    t_g1, t_gt = _p1("temporal",        "graphiti")
    c_f1, c_ft = _p1("current_updated", "fts5")
    c_g1, c_gt = _p1("current_updated", "graphiti")
    s_f1, s_ft = _p1("current_stable",  "fts5")
    s_g1, s_gt = _p1("current_stable",  "graphiti")

    print(f"  1. Temporal queries (as_of filtering):")
    print(f"       FTS5     ignores as_of — returns most-frequent entries regardless of date")
    print(f"                Precision@1 on temporal: {_pct(t_f1, t_ft)}")
    print(f"       graphiti filters by validity window — returns only facts valid at that date")
    print(f"                Precision@1 on temporal: {_pct(t_g1, t_gt)}")
    print()
    print(f"  2. Current-fact queries when history exists (supersession):")
    print(f"       FTS5     keeps ALL episodes — old facts compete with new ones for rank@1")
    print(f"                Precision@1 on current_updated: {_pct(c_f1, c_ft)}")
    print(f"       graphiti marks superseded facts invalid — only the newest is searchable")
    print(f"                Precision@1 on current_updated: {_pct(c_g1, c_gt)}")
    print()
    print(f"  3. Stable facts (no supersession):")
    print(f"       FTS5     Precision@1: {_pct(s_f1, s_ft)}   (no stale data to interfere)")
    print(f"       graphiti Precision@1: {_pct(s_g1, s_gt)}")
    print()
    print(f"  4. Temporal tools (graphiti-only):")
    print(f"       fact_history()   — complete per-entity validity timeline")
    print(f"       temporal_search(as_of=…) — point-in-time fact retrieval")
    print(f"       graph_browse()   — multi-hop neighbourhood traversal")
    print(f"       fact_correct()   — invalidate wrong facts with audit trail")
    print(f"       FTS5 provides no equivalent for any of the above.")

    _hr("═")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hermes-graphiti performance benchmark")
    parser.add_argument("-k", type=int, default=5, help="Top-K (default: 5)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show per-query top results")
    args = parser.parse_args()
    run_benchmark(k=args.k, verbose=args.verbose)
