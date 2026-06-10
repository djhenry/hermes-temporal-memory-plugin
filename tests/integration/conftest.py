"""Shared fixtures for integration tests — mocked Graphiti client.

Both integration test modules previously defined their own copies of these
fixtures, which drifted (one mocked build_indices_and_constraints, the other
didn't; one still set the deprecated GRAPHITI_USE_KUZU). Single source here.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from plugins.memory.graphiti import GraphitiMemoryProvider


def _empty_episode_result():
    r = MagicMock()
    r.nodes = []
    r.edges = []
    return r


@pytest.fixture
def hermes_home(tmp_path):
    (tmp_path / "memories").mkdir()
    return tmp_path


@pytest.fixture
def mock_client():
    c = MagicMock()
    c.search = AsyncMock(return_value=[])
    c.add_episode = AsyncMock(return_value=_empty_episode_result())
    c.nodes = MagicMock()
    c.nodes.entity = MagicMock()
    c.nodes.entity.get_by_group_ids = AsyncMock(return_value=[])
    c.edges = MagicMock()
    c.edges.entity = MagicMock()
    c.edges.entity.get_by_group_ids = AsyncMock(return_value=[])
    c.build_communities = AsyncMock(return_value=([], []))
    c.build_indices_and_constraints = AsyncMock()
    c.close = AsyncMock()
    return c


@pytest.fixture
def provider(mock_client, hermes_home, monkeypatch):
    monkeypatch.setenv("GRAPHITI_USE_FALKORDB_LITE", "1")
    with patch.object(GraphitiMemoryProvider, "_build_client", return_value=mock_client):
        p = GraphitiMemoryProvider()
        p.initialize("sess-001", identity="testuser", hermes_home=str(hermes_home))
    time.sleep(0.05)  # let seed thread settle
    yield p
    p.shutdown()  # stop the persistent event loop thread between tests
