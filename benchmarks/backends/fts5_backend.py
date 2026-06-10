"""
FTS5 baseline: simulates Hermes's built-in SQLite FTS5 episodic memory.

Behaviour mirrors how MEMORY.md + SQLite FTS5 works in Hermes:
  - Facts are stored as plain-text rows in an FTS5 virtual table
  - APPEND-only: new facts are added but old ones are never deleted
    (the agent can explicitly remove entries, but passive memory accumulates)
  - Search returns top-k matches ranked by BM25
  - No temporal metadata — as_of is silently IGNORED

This means:
  - Superseded facts (e.g. old city) remain in the index and compete for rank
  - Temporal queries always search the full fact set regardless of when facts
    were recorded
  - All past and present entries for the same entity co-exist in the index
"""

from __future__ import annotations

import re
import sqlite3
import time
from typing import Optional


_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "where", "what", "when",
    "who", "how", "which", "that", "this", "there", "their", "they",
    "and", "or", "but", "not", "with", "at", "by", "for", "in", "of",
    "on", "to", "from", "into", "about", "as", "it", "its", "my",
    "me", "he", "she", "we", "i", "you", "his", "her", "your",
})


def _to_fts5_query(text: str) -> str:
    """Convert a natural-language question to an FTS5 OR-prefix query."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    keywords = [w for w in words if w not in _STOP_WORDS and len(w) >= 3]
    if not keywords:
        words_fallback = re.findall(r"[a-zA-Z]+", text.lower())
        keywords = [w for w in words_fallback if len(w) >= 3]
    return " OR ".join(f"{w}*" for w in keywords) if keywords else text


class FTS5Backend:
    """SQLite FTS5 memory simulation — Hermes built-in memory baseline."""

    name = "Hermes FTS5 (built-in baseline)"
    short_name = "fts5"

    def __init__(self) -> None:
        self._db = sqlite3.connect(":memory:", check_same_thread=False)
        self._db.execute("""
            CREATE VIRTUAL TABLE facts
            USING fts5(
                content,
                entity      UNINDEXED,
                relation    UNINDEXED,
                valid_at    UNINDEXED,
                invalid_at  UNINDEXED,
                tokenize    = 'unicode61'
            )
        """)
        self._db.commit()

        self._ingest_times: list[float] = []
        self._query_times: list[float] = []
        self._last_meta: list[tuple[str, str, str]] = []

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def ingest_episode(
        self,
        fact: str,
        entity: str,
        relationship: str,
        valid_at: str,
        invalid_at: Optional[str] = None,
    ) -> None:
        """Append-only insert — old facts are NEVER removed (MEMORY.md behaviour)."""
        t0 = time.perf_counter()
        self._db.execute(
            "INSERT INTO facts(content, entity, relation, valid_at, invalid_at) VALUES (?, ?, ?, ?, ?)",
            (fact, entity, relationship, valid_at, invalid_at or ""),
        )
        self._db.commit()
        self._ingest_times.append(time.perf_counter() - t0)

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        as_of: Optional[str] = None,   # deliberately IGNORED — FTS5 has no temporal concept
        k: int = 5,
    ) -> list[str]:
        """FTS5 BM25 search.  `as_of` is ignored — no temporal filtering exists.

        Tie-breaking uses rowid DESC (most-recently inserted fact wins) which
        matches the optimistic assumption that newer memories are added later.
        This is the BEST CASE for FTS5 on current-state queries, and the
        WORST CASE for temporal queries (always returns newest, never historical).
        """
        t0 = time.perf_counter()
        fts_query = _to_fts5_query(query)
        results: list[str] = []
        if fts_query:
            try:
                rows = self._db.execute(
                    f"SELECT content, valid_at, invalid_at FROM facts "
                    f"WHERE facts MATCH ? ORDER BY bm25(facts), rowid DESC LIMIT {k}",
                    (fts_query,),
                ).fetchall()
                results = [r[0] for r in rows]
                self._last_meta = [(r[0], r[1], r[2]) for r in rows]
            except sqlite3.OperationalError:
                rows = self._db.execute(
                    f"SELECT content, valid_at, invalid_at FROM facts ORDER BY rowid DESC LIMIT {k}"
                ).fetchall()
                results = [r[0] for r in rows]
                self._last_meta = [(r[0], r[1], r[2]) for r in rows]
        else:
            self._last_meta = []
        self._query_times.append(time.perf_counter() - t0)
        return results

    def search_with_meta(
        self,
        query: str,
        as_of: Optional[str] = None,
        k: int = 5,
    ) -> list[tuple[str, str, str]]:
        """Search and return (content, valid_at, invalid_at) triples."""
        self.search(query, as_of=as_of, k=k)
        return getattr(self, "_last_meta", [])

    # ── Metrics ────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        total = self._db.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        return {
            "total_facts": total,
            "avg_ingest_us": _mean_us(self._ingest_times),
            "avg_query_us":  _mean_us(self._query_times),
            "p95_query_us":  _p95_us(self._query_times),
        }

    def reset(self) -> None:
        self._db.execute("DELETE FROM facts")
        self._db.commit()
        self._ingest_times.clear()
        self._query_times.clear()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _mean_us(times: list[float]) -> float:
    return (sum(times) / len(times) * 1_000_000) if times else 0.0


def _p95_us(times: list[float]) -> float:
    if not times:
        return 0.0
    s = sorted(times)
    idx = max(0, int(len(s) * 0.95) - 1)
    return s[idx] * 1_000_000
