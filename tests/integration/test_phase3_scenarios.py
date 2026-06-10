"""Phase 3 integration tests: entity resolution, supersession, fact correction.

These tests verify correctness of the temporal memory behaviour using a mocked
Graphiti client. They document the expected behaviour for:

  - Fact supersession: a city move should surface the new city as current and
    keep the old one accessible via fact_history.
  - Entity disambiguation: two people sharing a first name should not be
    conflated in the index or in search results.
  - Contradiction logging: superseded edges trigger an INFO log line.
  - fact_correct: the tool creates a correction episode in the graph.

Live versions of these tests (against a real graph) belong in test-live and
are the acceptance criteria for Phase 3 KRs 2.2, 2.3, and 2.4.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from plugins.memory.graphiti.memory_index import FENCE_START


# ── Helpers ───────────────────────────────────────────────────────────────────

def _node(name, labels=None):
    return SimpleNamespace(
        name=name, labels=labels or ["Person"],
        created_at=datetime.now(timezone.utc),
    )


def _edge(fact, valid_at=None, invalid_at=None, name="REL"):
    return SimpleNamespace(
        fact=fact, name=name,
        valid_at=valid_at, invalid_at=invalid_at,
        source_node_uuid="u1", target_node_uuid="u2",
    )


def _episode_result(nodes=None, edges=None):
    r = MagicMock()
    r.nodes = nodes or []
    r.edges = edges or []
    return r


# Fixtures (hermes_home, mock_client, provider) are shared — see conftest.py.


# ── Supersession scenarios ────────────────────────────────────────────────────

class TestSupersessionScenario:
    """KR 2.2 — prefetch never returns a superseded fact as current."""

    def test_city_move_returns_only_new_city_from_prefetch(self, provider, mock_client):
        """After a move, prefetch should return Madrid, not Barcelona."""
        valid_from = datetime(2026, 3, 1, tzinfo=timezone.utc)
        mock_client.search = AsyncMock(return_value=[
            # Only the current fact — graphiti's supersession removed the old one
            _edge("User lives in Madrid", valid_at=valid_from),
        ])
        result = provider.prefetch("where do I live")
        assert result is not None
        assert "Madrid" in result
        assert "Barcelona" not in result

    def test_fact_history_shows_both_cities(self, provider, mock_client):
        """fact_history must show both cities with their validity windows."""
        v_old = datetime(2024, 1, 1, tzinfo=timezone.utc)
        v_new = datetime(2026, 3, 1, tzinfo=timezone.utc)
        mock_client.search = AsyncMock(return_value=[
            _edge("User lives in Madrid", valid_at=v_new, name="LIVES_IN"),
            _edge("User lived in Barcelona", valid_at=v_old,
                  invalid_at=datetime(2026, 3, 1, tzinfo=timezone.utc), name="LIVES_IN"),
        ])
        result = provider._fact_history("user", "LIVES_IN")
        assert "Barcelona" in result
        assert "Madrid" in result
        # Barcelona (older) must appear before Madrid (newer)
        assert result.index("Barcelona") < result.index("Madrid")

    def test_fact_history_shows_validity_windows(self, provider, mock_client):
        """Each entry in fact_history must show its validity window."""
        v_old = datetime(2024, 1, 1, tzinfo=timezone.utc)
        v_new = datetime(2026, 3, 1, tzinfo=timezone.utc)
        mock_client.search = AsyncMock(return_value=[
            _edge("User lived in Barcelona", valid_at=v_old,
                  invalid_at=datetime(2026, 3, 1, tzinfo=timezone.utc)),
            _edge("User lives in Madrid", valid_at=v_new),
        ])
        result = provider._fact_history("user")
        assert "2024-01" in result  # Barcelona start
        assert "2026-03" in result  # Move date

    def test_temporal_search_as_of_excludes_future_facts(self, provider, mock_client):
        """temporal_search(as_of=2025-01-01) must not return facts valid after that."""
        mock_client.search = AsyncMock(return_value=[
            _edge("User lives in Madrid",
                  valid_at=datetime(2026, 3, 1, tzinfo=timezone.utc)),
        ])
        result = provider._temporal_search("where do I live", as_of="2025-01-01")
        assert "Madrid" not in result

    def test_temporal_search_as_of_includes_valid_facts(self, provider, mock_client):
        """temporal_search(as_of=2025-01-01) returns facts valid at that date."""
        mock_client.search = AsyncMock(return_value=[
            _edge("User lives in Barcelona",
                  valid_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                  invalid_at=datetime(2026, 3, 1, tzinfo=timezone.utc)),
        ])
        result = provider._temporal_search("where do I live", as_of="2025-01-01")
        assert "Barcelona" in result

    def test_job_change_history_timeline(self, provider, mock_client):
        """Same supersession pattern applies to job changes."""
        mock_client.search = AsyncMock(return_value=[
            _edge("User works at Acme Corp as Engineer",
                  valid_at=datetime(2023, 6, 1, tzinfo=timezone.utc),
                  invalid_at=datetime(2025, 3, 1, tzinfo=timezone.utc), name="WORKS_AT"),
            _edge("User works at Acme Corp as Staff Engineer",
                  valid_at=datetime(2025, 3, 1, tzinfo=timezone.utc), name="WORKS_AT"),
        ])
        result = provider._fact_history("user", "WORKS_AT")
        assert "Engineer" in result
        assert "Staff Engineer" in result
        assert result.index("Engineer") < result.index("Staff Engineer")


# ── Entity disambiguation scenarios ──────────────────────────────────────────

class TestEntityDisambiguation:
    """KR 2.3 — same first name, different contexts → separate index entries."""

    def test_two_martas_appear_in_index(self, provider, mock_client, hermes_home):
        """Two Martas introduced in different contexts should both appear."""
        provider._ingest_turn(
            "Met Marta Kovač from the data science team today.",
            "Got it, I'll remember Marta Kovač is your data science colleague.",
        )
        mock_client.add_episode = AsyncMock(
            return_value=_episode_result(
                nodes=[_node("Marta Kovač"), _node("Marta Ruiz")]
            )
        )
        provider._ingest_turn(
            "Marta Ruiz introduced me to the running club.",
            "Got it, Marta Ruiz is your running club contact.",
        )
        # _ingest_turn runs synchronously, so the index write has completed.
        memory_md = hermes_home / "memories" / "MEMORY.md"
        content = memory_md.read_text()
        assert "Marta Kovač" in content
        assert "Marta Ruiz" in content

    def test_disambiguation_preserved_in_graph_browse(self, provider, mock_client):
        """graph_browse for each Marta returns distinct facts."""
        mock_client.search = AsyncMock(
            return_value=[_edge("Marta Kovač works in data science")]
        )
        result_kovac = provider._graph_browse("Marta Kovač")
        assert "Marta Kovač" in result_kovac or "data science" in result_kovac

        mock_client.search = AsyncMock(
            return_value=[_edge("Marta Ruiz is a running club member")]
        )
        result_ruiz = provider._graph_browse("Marta Ruiz")
        assert "running" in result_ruiz or "Marta Ruiz" in result_ruiz


# ── Contradiction logging ─────────────────────────────────────────────────────

class TestContradictionLogging:
    """Phase 3 — superseded facts are logged at INFO level."""

    def test_superseded_edge_is_logged(self, provider, mock_client, caplog):
        """When add_episode returns an edge with invalid_at, it should be logged."""
        superseded_edge = _edge(
            "User lived in Barcelona",
            valid_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            invalid_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
        )
        mock_client.add_episode = AsyncMock(
            return_value=_episode_result(edges=[superseded_edge])
        )

        with caplog.at_level(logging.INFO, logger="plugins.memory.graphiti"):
            provider._ingest_turn(
                "I just moved to Madrid.",
                "Got it, you're now in Madrid.",
            )

        assert any("Superseded" in r.message for r in caplog.records)
        assert any("Barcelona" in r.message for r in caplog.records)

    def test_non_superseded_edges_not_logged_as_superseded(self, provider, mock_client, caplog):
        """Normal new edges (no invalid_at) must NOT trigger a Superseded log."""
        new_edge = _edge("User lives in Madrid",
                         valid_at=datetime(2026, 3, 1, tzinfo=timezone.utc))
        mock_client.add_episode = AsyncMock(
            return_value=_episode_result(edges=[new_edge])
        )

        with caplog.at_level(logging.INFO, logger="plugins.memory.graphiti"):
            provider._ingest_turn("I live in Madrid.", "Noted.")

        assert not any("Superseded" in r.message for r in caplog.records)


# ── fact_correct tool ─────────────────────────────────────────────────────────

class TestFactCorrect:
    """fact_correct creates a correction episode and returns a confirmation."""

    def test_creates_correction_episode(self, provider, mock_client):
        mock_client.add_episode = AsyncMock(return_value=_episode_result())
        result = provider._fact_correct(
            entity="user",
            relationship="LIVES_IN",
            correction="The user has always lived in Madrid; the Barcelona entry was wrong.",
        )
        mock_client.add_episode.assert_called_once()
        call_kwargs = mock_client.add_episode.call_args.kwargs
        assert "user_correction" in call_kwargs.get("name", "")
        assert "correction" in call_kwargs.get("source_description", "")

    def test_confirmation_message_includes_entity(self, provider, mock_client):
        mock_client.add_episode = AsyncMock(return_value=_episode_result())
        result = provider._fact_correct(
            entity="Marta",
            relationship="WORKS_AT",
            correction="Marta works at BioTech, not Acme.",
        )
        assert "Marta" in result

    def test_fact_correct_registered_in_tool_schemas(self, provider):
        names = {s["name"] for s in provider.get_tool_schemas()}
        assert "fact_correct" in names

    def test_fact_correct_dispatched_via_handle_tool_call(self, provider, mock_client):
        mock_client.add_episode = AsyncMock(return_value=_episode_result())
        result = provider.handle_tool_call("fact_correct", {
            "entity": "user",
            "relationship": "LIVES_IN",
            "correction": "Actually always Madrid.",
        })
        assert "user" in result or "Recorded" in result


# ── Context-fencing depth ─────────────────────────────────────────────────────

class TestContextFencingDepth:
    """Injected context must never reach the graph, even through edge cases."""

    def test_multi_line_memory_context_fully_stripped(self, provider, mock_client):
        mock_client.add_episode = AsyncMock(return_value=_episode_result())
        user_msg = (
            "<memory-context>\n"
            "- [2026-03 → now] Lives in Madrid\n"
            "- [2024-01 → 2026-03] Lived in Barcelona\n"
            "</memory-context>\n"
            "When did I move to Madrid?"
        )
        provider._ingest_turn(user_msg, "You moved in March 2026.")
        body = mock_client.add_episode.call_args.kwargs["episode_body"]
        assert "Lives in Madrid" not in body
        assert "Barcelona" not in body
        assert "When did I move" in body

    def test_index_section_stripped_from_assistant_turn(self, provider, mock_client):
        mock_client.add_episode = AsyncMock(return_value=_episode_result())
        assistant_msg = (
            "Here is your answer.\n"
            f"{FENCE_START}\n## Index\nAlice, Bob\n{FENCE_START.replace('start', 'end')}\n"
            "End of response."
        )
        provider._ingest_turn("Hello", assistant_msg)
        body = mock_client.add_episode.call_args.kwargs["episode_body"]
        assert "## Index" not in body
        assert "Here is your answer." in body
