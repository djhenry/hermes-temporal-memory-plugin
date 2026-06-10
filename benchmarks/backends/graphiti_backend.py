"""
Graphiti mock backend: simulates hermes-graphiti's retrieval behaviour.

Implements the same ingest/search interface as FTS5Backend but with:
  - Bi-temporal fact storage  — every fact has valid_at / invalid_at timestamps
  - Supersession              — ingesting a new fact for the same entity+relationship
                                marks the previous one invalid (sets invalid_at)
  - Temporal filtering        — search(as_of=…) returns only facts whose validity
                                window contains that date
  - Present-state queries     — without as_of, only facts with invalid_at=None are
                                eligible (no stale results)
  - BM25 ranking              — same scoring algorithm as FTS5 backend for fairness

This mock reproduces graphiti-core's query semantics faithfully without requiring
a running Neo4j / FalkorDB backend.  The LLM-extraction step is bypassed: tests
feed pre-extracted episodes directly, isolating retrieval accuracy.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Optional


# ── Tokenisation & BM25 ───────────────────────────────────────────────────────

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "where", "what", "when",
    "who", "how", "which", "that", "this", "there", "their", "they",
    "and", "or", "but", "not", "with", "at", "by", "for", "in", "of",
    "on", "to", "from", "into", "about", "as", "it", "its", "my",
    "me", "he", "she", "we", "i", "you", "his", "her", "your",
})

_BM25_K1 = 1.5
_BM25_B  = 0.75


def _stem(word: str) -> str:
    """Minimal English suffix stripping so 'enjoys'/'enjoyed' match 'enjoy'."""
    for suffix in ("ying", "ing", "ied", "ies", "ed", "es", "s"):
        stem = word[: len(word) - len(suffix)]
        if len(stem) >= 3:
            return stem
    return word


def _tokenize(text: str) -> list[str]:
    """Lower-case alpha-only tokens, stop-word filtered, length ≥ 3, lightly stemmed."""
    raw = re.findall(r"[a-zA-Z]+", text.lower())
    return [_stem(w) for w in raw if w not in _STOP_WORDS and len(w) >= 3]


def _bm25(query_tokens: list[str], doc_tokens: list[str], idf: dict[str, float], avg_dl: float) -> float:
    tf_map = Counter(doc_tokens)
    dl = len(doc_tokens)
    score = 0.0
    for term in query_tokens:
        tf = tf_map.get(term, 0)
        if tf == 0:
            continue
        norm_tf = (tf * (_BM25_K1 + 1)) / (tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / max(avg_dl, 1)))
        score += idf.get(term, 0.0) * norm_tf
    return score


def _compute_idf(query_tokens: list[str], corpus: list[list[str]]) -> dict[str, float]:
    n = len(corpus)
    result: dict[str, float] = {}
    for term in set(query_tokens):
        df = sum(1 for doc in corpus if term in doc)
        result[term] = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
    return result


# ── Temporal fact data model ───────────────────────────────────────────────────

def _parse_dt(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)


class _Fact:
    __slots__ = ("fact", "entity", "relationship", "valid_at", "invalid_at", "tokens")

    def __init__(
        self,
        fact: str,
        entity: str,
        relationship: str,
        valid_at: str,
        invalid_at: Optional[str],
    ) -> None:
        self.fact         = fact
        self.entity       = entity
        self.relationship = relationship
        self.valid_at     = _parse_dt(valid_at)
        self.invalid_at   = _parse_dt(invalid_at) if invalid_at else None
        self.tokens       = _tokenize(fact)


# ── Backend ───────────────────────────────────────────────────────────────────

class GraphitiMockBackend:
    """
    In-memory simulation of hermes-graphiti's retrieval layer.

    Key guarantees (mirroring graphiti-core behaviour):
      1. Supersession — a later fact for the same (entity, relationship) pair
         invalidates all older facts with no invalid_at.
      2. Present-state queries — only facts with invalid_at=None are searched.
      3. Temporal queries — only facts whose [valid_at, invalid_at) window
         contains the as_of date are searched.
      4. BM25 ranking — same scoring logic used in both backends.
    """

    name = "hermes-graphiti (temporal mock)"
    short_name = "graphiti"

    def __init__(self) -> None:
        self._facts: list[_Fact] = []
        self._ingest_times: list[float] = []
        self._query_times:  list[float] = []

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def ingest_episode(
        self,
        fact: str,
        entity: str,
        relationship: str,
        valid_at: str,
        invalid_at: Optional[str] = None,
    ) -> None:
        t0 = time.perf_counter()
        new_valid_at = _parse_dt(valid_at)

        # Supersession: close any open fact for the same (entity, relationship)
        # that is older than the incoming fact.
        for existing in self._facts:
            if (
                existing.entity       == entity
                and existing.relationship == relationship
                and existing.invalid_at   is None
                and existing.valid_at     <= new_valid_at
            ):
                existing.invalid_at = new_valid_at

        self._facts.append(_Fact(fact, entity, relationship, valid_at, invalid_at))
        self._ingest_times.append(time.perf_counter() - t0)

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        as_of: Optional[str] = None,
        k: int = 5,
    ) -> list[str]:
        """BM25 search over temporally eligible facts.

        Without as_of: searches only currently-valid facts (invalid_at is None).
        With as_of   : searches facts valid at that point in time.
        """
        t0 = time.perf_counter()

        if as_of:
            as_of_dt = _parse_dt(as_of)
            candidates = [
                f for f in self._facts
                if f.valid_at <= as_of_dt
                and (f.invalid_at is None or f.invalid_at > as_of_dt)
            ]
        else:
            # Present-state: only facts that have not been superseded
            candidates = [f for f in self._facts if f.invalid_at is None]

        if not candidates:
            self._query_times.append(time.perf_counter() - t0)
            return []

        q_tokens  = _tokenize(query)
        corpus    = [f.tokens for f in candidates]
        avg_dl    = sum(len(t) for t in corpus) / len(corpus)
        idf_map   = _compute_idf(q_tokens, corpus)

        scored = sorted(
            candidates,
            key=lambda f: _bm25(q_tokens, f.tokens, idf_map, avg_dl),
            reverse=True,
        )

        results = [f.fact for f in scored[:k]]
        self._query_times.append(time.perf_counter() - t0)
        return results

    # ── Temporal tools (graphiti-only) ─────────────────────────────────────────

    def fact_history(self, entity: str, relationship: Optional[str] = None) -> list[dict]:
        """Return the complete validity timeline for an entity (graphiti-only feature)."""
        relevant = [
            f for f in self._facts
            if f.entity == entity
            and (relationship is None or f.relationship == relationship)
        ]
        relevant.sort(key=lambda f: f.valid_at)
        return [
            {
                "fact":       f.fact,
                "valid_at":   f.valid_at.strftime("%Y-%m-%d"),
                "invalid_at": f.invalid_at.strftime("%Y-%m-%d") if f.invalid_at else "present",
                "status":     "superseded" if f.invalid_at else "current",
            }
            for f in relevant
        ]

    # ── Metrics ────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        total      = len(self._facts)
        current    = sum(1 for f in self._facts if f.invalid_at is None)
        superseded = total - current
        return {
            "total_facts":      total,
            "current_facts":    current,
            "superseded_facts": superseded,
            "avg_ingest_us":    _mean_us(self._ingest_times),
            "avg_query_us":     _mean_us(self._query_times),
            "p95_query_us":     _p95_us(self._query_times),
        }

    def reset(self) -> None:
        self._facts.clear()
        self._ingest_times.clear()
        self._query_times.clear()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _mean_us(times: list[float]) -> float:
    return (sum(times) / len(times) * 1_000_000) if times else 0.0


def _p95_us(times: list[float]) -> float:
    if not times:
        return 0.0
    s = sorted(times)
    return s[max(0, int(len(s) * 0.95) - 1)] * 1_000_000
