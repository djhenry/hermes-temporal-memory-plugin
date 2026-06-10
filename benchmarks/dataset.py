"""
Benchmark dataset: curated multi-turn conversations with pre-extracted facts and QA pairs.

Each scenario has:
  episodes  — facts an LLM extractor would produce from the conversation turns
              (pre-extracted to isolate retrieval from extraction quality)
  queries   — QA pairs with expected answer, category, and optional as_of date

Categories
----------
  current_stable   Facts that never change; both systems should ace these
  current_updated  The fact has been superseded; correct answer is the NEW value
                   — graphiti advantage: only the new fact is in the valid set,
                     FTS5 may rank older (incorrect) entries higher
  temporal         Point-in-time query with as_of set to a PAST date
                   — graphiti advantage: filters by validity window;
                     FTS5 has no temporal concept and ignores as_of
  disambiguation   Two entities with the same first name; different contexts
  multi_hop        Answer requires combining ≥2 connected facts

FTS5 failure conditions we deliberately engineer
-------------------------------------------------
  current_updated: old episodes outnumber new ones so BM25 ranks the wrong
                   (stale) entry first when the test has no temporal filter.
  temporal       : we make the CURRENT entry more frequent than the historical
                   one, so FTS5 (which ignores as_of) ranks the current —
                   and wrong — answer above the historically correct one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Episode:
    """A pre-extracted (entity, relationship, fact) triple with validity window."""
    fact: str
    entity: str
    relationship: str
    valid_at: str               # ISO-8601: when this fact became true
    invalid_at: Optional[str] = None  # ISO-8601: when superseded (None = still current)


@dataclass
class Query:
    """One benchmark question with ground-truth answer."""
    text: str
    expected: str               # Substring that must appear in a correct answer
    category: str               # see module docstring
    description: str
    as_of: Optional[str] = None  # ISO-8601 date for temporal queries


@dataclass
class Scenario:
    name: str
    description: str
    episodes: list[Episode]
    queries: list[Query]


# ── Scenario builders ──────────────────────────────────────────────────────────

def _location_history() -> Scenario:
    """
    User lived briefly in Barcelona (ONE episode), then moved to Madrid
    (THREE episodes — more frequent in the index).

    This arrangement makes FTS5 fail on temporal queries: when asked
    "where did the user live in early 2024?" (expected: Barcelona), FTS5
    ignores as_of and returns Madrid first because Madrid has more BM25-
    equal rows in insertion order.

    graphiti filters the validity window to the as_of date and correctly
    returns the single Barcelona fact that was valid then.
    """
    return Scenario(
        name="location_history",
        description="One historical city (Barcelona) vs three current entries (Madrid) "
                    "— FTS5 temporal failure engineered via frequency imbalance",
        episodes=[
            # ONE old entry — low frequency in FTS5
            Episode("User lives in Barcelona",          "user", "LIVES_IN", "2024-01-01"),
            # THREE current entries — dominates FTS5 ranking
            Episode("User lives in Madrid",             "user", "LIVES_IN", "2024-06-15"),
            Episode("User lives in Madrid",             "user", "LIVES_IN", "2025-01-01"),
            Episode("User lives in Madrid",             "user", "LIVES_IN", "2026-03-01"),
        ],
        queries=[
            Query(
                "Where does the user live?",
                "Madrid",
                "current_updated",
                "Current city — graphiti returns only valid fact; FTS5 returns "
                "Madrid first by frequency (happens to be correct here)",
            ),
            Query(
                "What city does the user currently live in?",
                "Madrid",
                "current_updated",
                "Explicit 'currently' — same supersession test",
            ),
            Query(
                "Where did the user live in early 2024?",
                "Barcelona",
                "temporal",
                "Past city before the June 2024 move — graphiti filters to "
                "as_of; FTS5 returns Madrid (3 entries) before Barcelona (1 entry)",
                as_of="2024-03-01",
            ),
            Query(
                "What city was the user living in before moving to Madrid?",
                "Barcelona",
                "temporal",
                "Point-in-time before the move — graphiti wins with as_of filter",
                as_of="2024-05-01",
            ),
        ],
    )


def _job_history() -> Scenario:
    """
    ONE historical TechCorp entry, THREE current StartupXYZ entries.
    Same frequency-imbalance trick: FTS5 returns StartupXYZ for the
    temporal query asking about TechCorp times.
    """
    return Scenario(
        name="job_history",
        description="One historical employer (TechCorp) vs three current entries (StartupXYZ)"
                    " — FTS5 temporal failure via frequency imbalance",
        episodes=[
            # ONE old entry
            Episode("User works at TechCorp as software engineer", "user", "WORKS_AT", "2024-01-01"),
            # THREE current entries
            Episode("User works at StartupXYZ",                   "user", "WORKS_AT", "2025-01-10"),
            Episode("User works at StartupXYZ as lead engineer",  "user", "WORKS_AT", "2025-06-01"),
            Episode("User works at StartupXYZ on platform team",  "user", "WORKS_AT", "2026-03-01"),
        ],
        queries=[
            Query(
                "Where does the user work?",
                "StartupXYZ",
                "current_updated",
                "Current employer — graphiti returns only valid; FTS5 returns "
                "StartupXYZ first by frequency (correct, but for wrong reasons)",
            ),
            Query(
                "What company does the user work at now?",
                "StartupXYZ",
                "current_updated",
                "Explicit 'now' — same test",
            ),
            Query(
                "Where did the user work in 2024?",
                "TechCorp",
                "temporal",
                "Historical employer — FTS5 returns StartupXYZ (3 entries) "
                "before TechCorp (1 entry); graphiti filters correctly to as_of",
                as_of="2024-06-01",
            ),
            Query(
                "What was the user's job before StartupXYZ?",
                "TechCorp",
                "temporal",
                "Point-in-time before the job switch — graphiti wins",
                as_of="2024-10-01",
            ),
        ],
    )


def _entity_disambiguation() -> Scenario:
    """Two people named Alice — tests whether context distinguishes them."""
    return Scenario(
        name="entity_disambiguation",
        description="Two 'Alice' contacts in different contexts — graph node disambiguation",
        episodes=[
            Episode("Alice Chen is user's colleague at StartupXYZ on the backend team",
                    "alice_chen",   "COLLEAGUE_OF", "2025-01-10"),
            Episode("Alice Chen works with user on the database migration project",
                    "alice_chen",   "WORKS_WITH",   "2025-02-01"),
            Episode("Alice Rivera is user's friend from the weekend running club",
                    "alice_rivera", "FRIEND_OF",    "2024-03-01"),
            Episode("Alice Rivera was introduced to user by Marcus at Retiro Park",
                    "alice_rivera", "MET_VIA",      "2024-03-01"),
        ],
        queries=[
            Query(
                "Who is Alice Chen?",
                "colleague",
                "disambiguation",
                "Work-context Alice — answer should mention work/colleague",
            ),
            Query(
                "Who is Alice Rivera?",
                "running",
                "disambiguation",
                "Social-context Alice — answer should mention running club",
            ),
            Query(
                "Which Alice is the user's work colleague?",
                "Chen",
                "disambiguation",
                "Disambiguation by professional context",
            ),
            Query(
                "Who introduced the user to the running club?",
                "Marcus",
                "multi_hop",
                "Multi-hop: Alice Rivera ← introduced by Marcus",
            ),
        ],
    )


def _stable_facts() -> Scenario:
    """Facts that never change — baseline both systems should ace."""
    return Scenario(
        name="stable_facts",
        description="Unchanging personal facts — equal-ground baseline",
        episodes=[
            Episode("User has a dog named Max",                "user", "HAS_PET",     "2024-01-01"),
            Episode("User speaks Spanish",                     "user", "SPEAKS",       "2024-06-01"),
            Episode("User enjoys hiking on weekends",          "user", "ENJOYS",       "2024-06-01"),
            Episode("User follows a vegetarian diet",          "user", "DIET",         "2024-01-01"),
            Episode("User's birthday is on March 15",          "user", "HAS_BIRTHDAY", "2024-01-01"),
        ],
        queries=[
            Query("What is the user's dog's name?",              "Max",         "current_stable", "Stable pet fact"),
            Query("Does the user speak any other languages?",    "Spanish",     "current_stable", "Stable language fact"),
            Query("What hobby does the user enjoy?",             "hiking",      "current_stable", "Stable hobby fact"),
            Query("What is the user's diet?",                    "vegetarian",  "current_stable", "Stable dietary preference"),
        ],
    )


def _relationship_network() -> Scenario:
    """Multi-hop connections through the running club network."""
    return Scenario(
        name="relationship_network",
        description="Social network facts — multi-hop retrieval across connected nodes",
        episodes=[
            Episode("Marcus coaches the Saturday running group at Retiro Park",
                    "marcus",       "COACHES",      "2024-03-01"),
            Episode("The running club meets every Saturday morning at Retiro Park",
                    "running_club", "MEETS_AT",     "2024-03-01"),
            Episode("User joined the running club in March 2024 through Marcus",
                    "user",         "JOINED",       "2024-03-01"),
            Episode("Sarah co-organises the running club together with Marcus",
                    "sarah",        "CO_ORGANIZES", "2024-04-01"),
            Episode("The running club has about thirty members",
                    "running_club", "HAS_MEMBERS",  "2024-06-01"),
        ],
        queries=[
            Query(
                "Who coaches the running group?",
                "Marcus",
                "current_stable",
                "Direct organiser fact",
            ),
            Query(
                "Where does the running club meet?",
                "Retiro",
                "current_stable",
                "Meeting location",
            ),
            Query(
                "Who co-organises the running club with Marcus?",
                "Sarah",
                "multi_hop",
                "Second organiser — connected via club relationship",
            ),
            Query(
                "When did the user join the running club?",
                "March 2024",
                "temporal",
                "Join date; both systems have the fact but phrasing matters",
            ),
        ],
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def build_dataset() -> list[Scenario]:
    return [
        _location_history(),
        _job_history(),
        _entity_disambiguation(),
        _stable_facts(),
        _relationship_network(),
    ]
