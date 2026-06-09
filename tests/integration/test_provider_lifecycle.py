"""Integration tests for GraphitiMemoryProvider.

Uses a mocked Graphiti client — no real Neo4j, SQLite, or LLM needed.
Tests cover the full hook lifecycle, context-fencing, MEMORY.md index
updates, and tool implementations.

Live tests (real SQLite backend + extraction LLM) are marked @pytest.mark.live
and skipped in CI. Run them with: make test-live
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from plugins.memory.graphiti import GraphitiMemoryProvider
from plugins.memory.graphiti.memory_index import FENCE_END, FENCE_START


# ── Helpers ───────────────────────────────────────────────────────────────────

def _node(name="Alice", labels=None):
    return SimpleNamespace(
        name=name,
        labels=labels or ["Person"],
        created_at=datetime.now(timezone.utc),
    )


def _edge(fact="test fact", valid_at=None, invalid_at=None):
    return SimpleNamespace(
        fact=fact, name="REL",
        valid_at=valid_at, invalid_at=invalid_at,
        source_node_uuid="u1", target_node_uuid="u2",
    )


def _episode_result(nodes=None, edges=None):
    r = MagicMock()
    r.nodes = nodes or []
    r.edges = edges or []
    return r


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def hermes_home(tmp_path):
    (tmp_path / "memories").mkdir()
    return tmp_path


@pytest.fixture
def mock_client():
    c = MagicMock()
    c.search = AsyncMock(return_value=[])
    c.add_episode = AsyncMock(return_value=_episode_result())
    c.nodes = MagicMock()
    c.nodes.entity = MagicMock()
    c.nodes.entity.get_by_group_ids = AsyncMock(return_value=[])
    c.edges = MagicMock()
    c.edges.entity = MagicMock()
    c.edges.entity.get_by_group_ids = AsyncMock(return_value=[])
    c.build_communities = AsyncMock(return_value=([], []))
    c.close = AsyncMock()
    return c


@pytest.fixture
def provider(mock_client, hermes_home, monkeypatch):
    monkeypatch.setenv("GRAPHITI_USE_KUZU", "1")
    with patch.object(GraphitiMemoryProvider, "_build_client", return_value=mock_client):
        p = GraphitiMemoryProvider()
        p.initialize("sess-001", identity="testuser", hermes_home=str(hermes_home))
    time.sleep(0.05)  # let seed thread settle
    return p


# ── Tests: initialization ─────────────────────────────────────────────────────

class TestInit:
    def test_group_id_uses_identity(self, mock_client, hermes_home, monkeypatch):
        monkeypatch.setenv("GRAPHITI_USE_KUZU", "1")
        with patch.object(GraphitiMemoryProvider, "_build_client", return_value=mock_client):
            p = GraphitiMemoryProvider()
            p.initialize("s1", identity="alice", hermes_home=str(hermes_home))
        assert p._group_id == "hermes-alice"

    def test_group_id_appends_platform_user(self, mock_client, hermes_home, monkeypatch):
        monkeypatch.setenv("GRAPHITI_USE_KUZU", "1")
        source = SimpleNamespace(user_id="tg-99")
        with patch.object(GraphitiMemoryProvider, "_build_client", return_value=mock_client):
            p = GraphitiMemoryProvider()
            p.initialize("s1", identity="alice", session_source=source,
                         hermes_home=str(hermes_home))
        assert p._group_id == "hermes-alice-tg-99"

    def test_is_available_true_when_kuzu_env_set(self, monkeypatch):
        monkeypatch.setenv("GRAPHITI_USE_KUZU", "1")
        assert GraphitiMemoryProvider().is_available() is True

    def test_is_available_false_when_nothing_configured(self, monkeypatch):
        monkeypatch.delenv("GRAPHITI_USE_KUZU", raising=False)
        monkeypatch.delenv("GRAPHITI_NEO4J_URI", raising=False)
        p = GraphitiMemoryProvider()
        p._cfg.backend = "neo4j"  # prevent sqlite-default from triggering
        assert p.is_available() is False


# ── Tests: prefetch ───────────────────────────────────────────────────────────

class TestPrefetch:
    def test_returns_none_when_no_results(self, provider):
        assert provider.prefetch("where do I live") is None

    def test_formats_open_validity_window(self, provider, mock_client):
        valid = datetime(2026, 3, 1, tzinfo=timezone.utc)
        mock_client.search = AsyncMock(return_value=[_edge("Lives in Madrid", valid_at=valid)])
        result = provider.prefetch("where do I live")
        assert result is not None
        assert "Lives in Madrid" in result
        assert "2026-03" in result
        assert "→ now" in result

    def test_formats_closed_validity_window(self, provider, mock_client):
        v_from = datetime(2024, 1, 1, tzinfo=timezone.utc)
        v_to = datetime(2026, 3, 1, tzinfo=timezone.utc)
        mock_client.search = AsyncMock(
            return_value=[_edge("Lived in Barcelona", valid_at=v_from, invalid_at=v_to)]
        )
        result = provider.prefetch("old city")
        assert "2024-01 → 2026-03" in result

    def test_wraps_result_in_memory_context_fence(self, provider, mock_client):
        mock_client.search = AsyncMock(return_value=[_edge("some fact")])
        result = provider.prefetch("anything")
        assert result.startswith("<memory-context>")
        assert result.endswith("</memory-context>")

    def test_returns_none_in_tools_only_mode(self, provider):
        provider._cfg.recall_mode = "tools"
        assert provider.prefetch("anything") is None


# ── Tests: sync_turn / episode ingestion ──────────────────────────────────────

class TestSyncTurnIngestion:
    def test_strips_memory_context_fence(self, provider, mock_client):
        mock_client.add_episode = AsyncMock(return_value=_episode_result())
        provider._ingest_turn(
            "<memory-context>[recalled fact]</memory-context> Hello there",
            "Reply",
        )
        body = mock_client.add_episode.call_args.kwargs["episode_body"]
        assert "<memory-context>" not in body
        assert "Hello there" in body

    def test_strips_index_fence(self, provider, mock_client):
        mock_client.add_episode = AsyncMock(return_value=_episode_result())
        provider._ingest_turn(
            f"Lead text {FENCE_START}INDEX{FENCE_END} tail text",
            "Reply",
        )
        body = mock_client.add_episode.call_args.kwargs["episode_body"]
        assert "INDEX" not in body
        assert "Lead text" in body
        assert "tail text" in body

    def test_includes_both_turns_in_episode_body(self, provider, mock_client):
        mock_client.add_episode = AsyncMock(return_value=_episode_result())
        provider._ingest_turn("User said hello", "Agent replied hi")
        body = mock_client.add_episode.call_args.kwargs["episode_body"]
        assert "User said hello" in body
        assert "Agent replied hi" in body

    def test_updates_index_with_newly_extracted_nodes(self, provider, mock_client, hermes_home):
        mock_client.add_episode = AsyncMock(
            return_value=_episode_result(nodes=[_node("Marta"), _node("Madrid", ["Place"])])
        )
        provider._ingest_turn("Met Marta in Madrid", "Got it")
        content = (hermes_home / "memories" / "MEMORY.md").read_text()
        assert "Marta" in content

    def test_handles_ingest_exception_without_raising(self, provider, mock_client):
        mock_client.add_episode = AsyncMock(side_effect=ConnectionError("neo4j down"))
        provider._ingest_turn("Hello", "Hi")  # must not raise

    def test_sync_turn_returns_immediately(self, provider):
        start = time.monotonic()
        provider.sync_turn("Hello", "Hi")
        assert time.monotonic() - start < 0.5

    def test_sync_turn_starts_daemon_thread(self, provider):
        provider.sync_turn("Hello", "Hi")
        assert provider._sync_thread is not None
        assert provider._sync_thread.daemon is True


# ── Tests: on_memory_write ────────────────────────────────────────────────────

class TestOnMemoryWrite:
    def test_mirrors_add_action_to_graph(self, provider, mock_client):
        mock_client.add_episode = AsyncMock(return_value=_episode_result())
        provider.on_memory_write("add", "memory", "Alice is my colleague")
        time.sleep(0.1)
        mock_client.add_episode.assert_called()

    def test_does_not_mirror_remove_action(self, provider, mock_client):
        mock_client.add_episode = AsyncMock(return_value=_episode_result())
        provider.on_memory_write("remove", "memory", "some fact")
        time.sleep(0.05)
        mock_client.add_episode.assert_not_called()

    def test_does_not_mirror_user_target(self, provider, mock_client):
        mock_client.add_episode = AsyncMock(return_value=_episode_result())
        provider.on_memory_write("add", "user", "prefers dark mode")
        time.sleep(0.05)
        # User-target is not mirrored to graph (it's profile, not episodic memory)
        mock_client.add_episode.assert_not_called()

    def test_add_triggers_index_update(self, provider, hermes_home):
        provider.on_memory_write("add", "memory", "Bob is my running partner")
        memory_md = hermes_home / "memories" / "MEMORY.md"
        assert memory_md.exists()


# ── Tests: system_prompt_block ────────────────────────────────────────────────

class TestSystemPromptBlock:
    def test_returns_instruction_when_index_enabled(self, provider):
        provider._cfg.enable_memory_index = True
        block = provider.system_prompt_block()
        assert block is not None
        assert "temporal_search" in block
        assert "graphiti" in block.lower()

    def test_returns_none_when_index_disabled(self, provider):
        provider._cfg.enable_memory_index = False
        assert provider.system_prompt_block() is None


# ── Tests: tool schemas & handlers ───────────────────────────────────────────

class TestTools:
    def test_get_tool_schemas_exposes_four_tools(self, provider):
        names = {s["name"] for s in provider.get_tool_schemas()}
        assert names == {"temporal_search", "fact_history", "graph_browse", "fact_correct"}

    def test_temporal_search_formats_results(self, provider, mock_client):
        valid = datetime(2026, 3, 1, tzinfo=timezone.utc)
        mock_client.search = AsyncMock(return_value=[_edge("Lives in Madrid", valid_at=valid)])
        result = provider._temporal_search("where do I live")
        assert "Lives in Madrid" in result
        assert "2026-03" in result

    def test_temporal_search_client_side_as_of_filter(self, provider, mock_client):
        future_valid = datetime(2027, 1, 1, tzinfo=timezone.utc)
        mock_client.search = AsyncMock(
            return_value=[_edge("Will move to Berlin", valid_at=future_valid)]
        )
        result = provider._temporal_search("city", as_of="2026-01-01")
        assert "Berlin" not in result

    def test_fact_history_orders_chronologically(self, provider, mock_client):
        mock_client.search = AsyncMock(return_value=[
            _edge("User lives in Madrid", valid_at=datetime(2026, 3, 1, tzinfo=timezone.utc)),
            _edge("User lived in Barcelona", valid_at=datetime(2024, 1, 1, tzinfo=timezone.utc)),
        ])
        result = provider._fact_history("user")
        assert result.index("Barcelona") < result.index("Madrid")

    def test_fact_history_returns_message_when_nothing_found(self, provider, mock_client):
        mock_client.search = AsyncMock(return_value=[])
        result = provider._fact_history("unknownentity")
        assert "No history" in result

    def test_handle_tool_call_unknown_returns_error(self, provider):
        result = provider.handle_tool_call("no_such_tool", {})
        assert "Unknown tool" in result


# ── Live tests ────────────────────────────────────────────────────────────────

@pytest.mark.live
class TestLiveGraphiti:
    """End-to-end round-trip against real Graphiti (Kuzu embedded backend).

    Prerequisites:
        pip install hermes-graphiti[kuzu]
        GRAPHITI_USE_KUZU=1
        OPENAI_API_KEY         — any OpenAI-compatible key, including OpenRouter
        OPENAI_BASE_URL        — https://openrouter.ai/api/v1 for OpenRouter
        GRAPHITI_LLM_MODEL     — optional; defaults to gpt-4o-mini

    Run: make test-live
    """

    def test_ingest_and_retrieve(self, tmp_path):
        import os

        pytest.importorskip("graphiti_core")
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        (tmp_path / "memories").mkdir()
        p = GraphitiMemoryProvider()
        p.initialize("live-session", identity="tester", hermes_home=str(tmp_path))
        time.sleep(0.2)

        # Call ingestion synchronously so the pytest timeout governs wait time,
        # not a fixed join. sync_turn uses a daemon thread that free-tier LLMs
        # can outlast; _ingest_turn blocks until all LLM calls complete.
        p._ingest_turn(
            "I just moved from Barcelona to Madrid.",
            "Noted — I'll remember you're in Madrid now.",
        )

        result = p.prefetch("where do I live")
        assert result is not None
        assert "Madrid" in result

        history = p._fact_history("user")
        assert "Madrid" in history

        p.shutdown()
