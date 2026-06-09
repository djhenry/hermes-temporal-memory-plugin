"""Tests for MemoryIndexManager — the MEMORY.md recall-trigger index."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from plugins.memory.graphiti.memory_index import (
    FENCE_END,
    FENCE_START,
    ENTRY_SEP,
    MemoryIndexManager,
)


def _node(name: str, labels: list[str] | None = None, uuid: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        labels=labels or ["Person"],
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        uuid=uuid,
    )


def _edge(fact: str, valid_at=None, invalid_at=None, src_uuid="u1", tgt_uuid="u2") -> SimpleNamespace:
    return SimpleNamespace(
        fact=fact,
        name="RELATED_TO",
        valid_at=valid_at,
        invalid_at=invalid_at,
        source_node_uuid=src_uuid,
        target_node_uuid=tgt_uuid,
    )


@pytest.fixture
def memory_dir(tmp_path: Path) -> Path:
    (tmp_path / "memories").mkdir()
    return tmp_path / "memories"


@pytest.fixture
def manager(memory_dir: Path) -> MemoryIndexManager:
    return MemoryIndexManager(memory_dir)


class TestFencedWrite:
    def test_creates_memory_md_when_absent(self, manager, memory_dir):
        manager.update_from_episode([_node("Alice")], [])
        content = (memory_dir / "MEMORY.md").read_text()
        assert FENCE_START in content
        assert FENCE_END in content
        assert "Alice" in content

    def test_appends_as_entry_delimiter(self, manager, memory_dir):
        existing = "My name is Bob."
        (memory_dir / "MEMORY.md").write_text(existing)
        manager.update_from_episode([_node("Alice")], [])
        content = (memory_dir / "MEMORY.md").read_text()
        assert existing in content
        assert ENTRY_SEP in content  # §-separator present
        assert FENCE_START in content

    def test_replaces_existing_fence_in_place(self, manager, memory_dir):
        initial = f"User fact.{ENTRY_SEP}{FENCE_START}\n## old\n{FENCE_END}"
        (memory_dir / "MEMORY.md").write_text(initial)
        manager.update_from_episode([_node("Carol")], [])
        content = (memory_dir / "MEMORY.md").read_text()
        assert "User fact." in content
        assert "Carol" in content
        assert content.count(FENCE_START) == 1, "must not duplicate the fence"

    def test_user_content_outside_fence_untouched(self, manager, memory_dir):
        user_content = "Important personal note that must not be deleted."
        (memory_dir / "MEMORY.md").write_text(user_content)
        manager.update_from_episode([_node("Dave")], [])
        content = (memory_dir / "MEMORY.md").read_text()
        assert user_content in content


class TestEntityGrouping:
    def test_people_label(self, manager, memory_dir):
        manager.update_from_episode([_node("Alice", ["Person"])], [])
        content = (memory_dir / "MEMORY.md").read_text()
        assert "**People:**" in content
        assert "Alice" in content

    def test_place_label(self, manager, memory_dir):
        manager.update_from_episode([_node("Madrid", ["Place"])], [])
        content = (memory_dir / "MEMORY.md").read_text()
        assert "**Places:**" in content

    def test_unknown_label_grouped_as_other(self, manager, memory_dir):
        manager.update_from_episode([_node("Heliox", ["CustomThing"])], [])
        content = (memory_dir / "MEMORY.md").read_text()
        assert "Heliox" in content

    def test_history_available_hint(self, manager, memory_dir):
        node = _node("Barcelona", ["Place"], uuid="barcelona-uuid")
        edge = _edge(
            "lived in Barcelona",
            invalid_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
            src_uuid="barcelona-uuid",
            tgt_uuid="user-uuid",
        )
        manager.update_from_episode([node], [edge])
        content = (memory_dir / "MEMORY.md").read_text()
        assert "Barcelona" in content
        assert "· history available" in content


class TestOnMemoryWrite:
    def test_add_action_adds_to_notes(self, manager, memory_dir):
        manager.on_memory_write("add", "memory", "Alice is my colleague")
        content = (memory_dir / "MEMORY.md").read_text()
        assert "Alice is my colleague" in content

    def test_remove_of_fence_re_seeds(self, manager, memory_dir):
        manager.update_from_episode([_node("Bob")], [])
        before = (memory_dir / "MEMORY.md").read_text()
        assert FENCE_START in before
        # Simulate LLM removing our fence entry
        manager.on_memory_write("remove", "memory", f"{FENCE_START}\n## ...\n{FENCE_END}")
        after = (memory_dir / "MEMORY.md").read_text()
        assert FENCE_START in after  # re-seeded

    def test_ignores_user_target_content(self, manager, memory_dir):
        manager.on_memory_write("add", "user", "User prefers dark mode")
        # Should not add user-target content to index
        content = (memory_dir / "MEMORY.md").read_text() if (memory_dir / "MEMORY.md").exists() else ""
        assert "User prefers dark mode" not in content

    def test_does_not_double_ingest_own_fence(self, manager, memory_dir):
        manager.on_memory_write("add", "memory", f"{FENCE_START}injected{FENCE_END}")
        # Must not add our own fence content to the Notes section
        content = (memory_dir / "MEMORY.md").read_text() if (memory_dir / "MEMORY.md").exists() else ""
        assert "Notes" not in content or "graphiti-index" not in content.split("Notes")[0]


class TestFullRefresh:
    def _community(self, name, summary="") -> SimpleNamespace:
        return SimpleNamespace(name=name, summary=summary)

    def test_full_refresh_replaces_entity_view_with_topics(self, manager, memory_dir):
        manager.update_from_episode([_node("Alice"), _node("Bob")], [])
        communities = [
            self._community("Work & Career", "Projects and professional contacts"),
            self._community("Personal Life", "Family, friends, and hobbies"),
        ]
        manager.full_refresh(communities)
        content = (memory_dir / "MEMORY.md").read_text()
        assert "Work & Career" in content
        assert "Personal Life" in content
        assert "**Topics:**" in content

    def test_empty_communities_falls_back_to_entity_view(self, manager, memory_dir):
        manager.update_from_episode([_node("Carol")], [])
        manager.full_refresh([])
        content = (memory_dir / "MEMORY.md").read_text()
        # Should still render with whatever is in _entities (cleared to empty)
        assert FENCE_START in content


class TestContextFencing:
    def test_fence_start_end_markers_present(self, manager, memory_dir):
        manager.update_from_episode([_node("Eve")], [])
        content = (memory_dir / "MEMORY.md").read_text()
        start = content.index(FENCE_START)
        end = content.index(FENCE_END)
        assert start < end

    def test_index_content_between_markers_only(self, manager, memory_dir):
        user_fact = "Secret user note."
        (memory_dir / "MEMORY.md").write_text(user_fact)
        manager.update_from_episode([_node("Frank")], [])
        content = (memory_dir / "MEMORY.md").read_text()
        # User fact must be outside the fence
        fence_start_idx = content.index(FENCE_START)
        assert user_fact in content[:fence_start_idx] or user_fact in content[content.index(FENCE_END):]


class TestSeedFromGraph:
    def test_seed_populates_index(self, manager, memory_dir):
        nodes = [_node("Grace", ["Person"]), _node("London", ["Place"])]
        edges = []
        manager.seed_from_graph(nodes, edges)
        content = (memory_dir / "MEMORY.md").read_text()
        assert "Grace" in content
        assert "London" in content
