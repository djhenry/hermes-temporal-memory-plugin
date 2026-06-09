"""
Manages the Temporal Memory Index section in MEMORY.md.

The index is a fenced, plugin-owned entry inside MEMORY.md that acts as a
shallow discovery surface — the agent sees it in every system prompt and
knows what topics are available in the deep temporal graph.

Entry format (a normal §-delimited MEMORY.md entry):

    <!-- graphiti-index:start -->
    ## Temporal Memory Index
    _Updated 2026-06-09 · query deeper: temporal_search, fact_history, graph_browse_

    **People:** Alice (colleague), Bob (running club, via Alice)
    **Places:** Madrid (2026-03 → now) · history available
    **Projects:** Helios (active), Atlas (archived)
    <!-- graphiti-index:end -->
"""

from __future__ import annotations

import os
import threading
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # avoid heavy graphiti imports at module load time

FENCE_START = "<!-- graphiti-index:start -->"
FENCE_END = "<!-- graphiti-index:end -->"
ENTRY_SEP = "\n§\n"
MAX_INDEX_TOKENS = 200  # rough char-based cap (~4 chars/token)
MAX_INDEX_CHARS = MAX_INDEX_TOKENS * 4


class MemoryIndexManager:
    """Maintains the recall-trigger index section in MEMORY.md."""

    def __init__(self, memory_dir: Path) -> None:
        self._memory_md = memory_dir / "MEMORY.md"
        self._lock = threading.Lock()
        # In-memory entity store: label → {name: last_seen_dt}
        self._entities: dict[str, dict[str, datetime]] = defaultdict(dict)
        # Track which entities have >1 validity window (history available)
        self._has_history: set[str] = set()

    # ------------------------------------------------------------------
    # Public API called from plugin hooks
    # ------------------------------------------------------------------

    def seed_from_graph(self, nodes: list, edges: list) -> None:
        """Populate index from existing graph entities on initialize.

        Args:
            nodes: list of EntityNode objects from get_by_group_ids()
            edges: list of EntityEdge objects (used to detect history)
        """
        for node in nodes:
            label = _primary_label(node)
            self._entities[label][node.name] = getattr(node, "created_at", datetime.now(timezone.utc))

        # Any entity with both valid_at and invalid_at on at least one edge has history
        for edge in edges:
            if getattr(edge, "invalid_at", None) is not None:
                self._has_history.add(getattr(edge, "source_node_uuid", ""))
                self._has_history.add(getattr(edge, "target_node_uuid", ""))

        self._flush()

    def update_from_episode(self, new_nodes: list, new_edges: list) -> None:
        """Merge newly extracted entities after a sync_turn ingestion.

        Called inside the sync_turn daemon thread — never blocks the agent.

        Args:
            new_nodes: AddEpisodeResults.nodes
            new_edges: AddEpisodeResults.edges
        """
        now = datetime.now(timezone.utc)
        for node in new_nodes:
            label = _primary_label(node)
            self._entities[label][node.name] = now

        for edge in new_edges:
            if getattr(edge, "invalid_at", None) is not None:
                self._has_history.add(getattr(edge, "source_node_uuid", ""))
                self._has_history.add(getattr(edge, "target_node_uuid", ""))

        self._flush()

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """React to the built-in memory tool writing to MEMORY.md.

        - add/replace: treat the content as a plain-text entity hint and add
          it as an "Other" entry so it surfaces in the index.
        - remove: if the removed content contains our fence markers, re-seed
          the index immediately so it doesn't disappear.
        """
        if action in ("add", "replace") and target == "memory" and content:
            # Don't re-process our own index entry
            if FENCE_START not in content:
                self._entities["Notes"][content[:80].strip()] = datetime.now(timezone.utc)
                self._flush()
        elif action == "remove" and FENCE_START in (content or ""):
            # LLM tried to delete our section — re-seed it
            self._flush()

    def full_refresh(self, communities: list) -> None:
        """Rebuild index using community summaries from build_communities().

        Replaces the per-entity view with richer topic-cluster names when
        available. Called on on_session_end.

        Args:
            communities: list of CommunityNode objects
        """
        if not communities:
            self._flush()
            return

        # Replace entity store with community-derived topics
        self._entities.clear()
        for community in communities:
            name = getattr(community, "name", None) or "Unknown"
            summary = getattr(community, "summary", "") or ""
            # Use first 60 chars of summary as context hint
            hint = summary[:60].rstrip()
            if hint and not hint.endswith((".", "…")):
                hint += "…"
            self._entities["Topics"][f"{name}" + (f" ({hint})" if hint else "")] = datetime.now(timezone.utc)

        self._flush()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flush(self) -> None:
        """Render current entity state and write to MEMORY.md."""
        content = self._render()
        with self._lock:
            self._write_entry(content)

    def _render(self) -> str:
        """Render entity store into the fenced index block."""
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        lines = [
            FENCE_START,
            "## Temporal Memory Index",
            f"_Updated {now_str} · query deeper: temporal_search, fact_history, graph_browse_",
            "",
        ]

        # Preferred label order; anything else falls to the end
        label_order = ["Person", "Place", "Organization", "Project", "Topics", "Notes"]
        display_names = {
            "Person": "People",
            "Place": "Places",
            "Organization": "Organizations",
            "Project": "Projects",
            "Topics": "Topics",
            "Notes": "Notes",
        }

        all_labels = label_order + [l for l in self._entities if l not in label_order]

        for label in all_labels:
            entities = self._entities.get(label)
            if not entities:
                continue
            # Sort by last-seen descending
            sorted_names = sorted(entities.items(), key=lambda kv: kv[1], reverse=True)
            parts = []
            for name, _ in sorted_names:
                suffix = " · history available" if name in self._has_history else ""
                parts.append(f"{name}{suffix}")

            display = display_names.get(label, label)
            lines.append(f"**{display}:** {', '.join(parts)}")

        lines.append(FENCE_END)
        body = "\n".join(lines)

        # Enforce token cap: drop least-recently-seen entities
        while len(body) > MAX_INDEX_CHARS and self._entities:
            # Remove the single oldest entity across all labels
            oldest_label = None
            oldest_name = None
            oldest_dt = datetime.max.replace(tzinfo=timezone.utc)
            for label, ents in self._entities.items():
                for name, dt in ents.items():
                    if dt < oldest_dt:
                        oldest_dt = dt
                        oldest_label = label
                        oldest_name = name
            if oldest_label and oldest_name:
                del self._entities[oldest_label][oldest_name]
                if not self._entities[oldest_label]:
                    del self._entities[oldest_label]
                body = self._render()  # re-render after trimming
            else:
                break

        return body

    def _read_entry(self) -> tuple[str, int, int] | None:
        """Find the fenced block in MEMORY.md.

        Returns (raw_text, start_char_offset, end_char_offset) or None.
        """
        if not self._memory_md.exists():
            return None
        text = self._memory_md.read_text(encoding="utf-8")
        start = text.find(FENCE_START)
        if start == -1:
            return None
        end = text.find(FENCE_END, start)
        if end == -1:
            return None
        end += len(FENCE_END)
        return text[start:end], start, end

    def _write_entry(self, new_content: str) -> None:
        """Atomic read-modify-write of the fenced index block in MEMORY.md.

        - If the fenced block exists: replace it in-place.
        - If not: append it as a new §-delimited entry.
        - Content outside the fence is never touched.
        """
        if self._memory_md.exists():
            text = self._memory_md.read_text(encoding="utf-8")
        else:
            text = ""

        start = text.find(FENCE_START)
        end_marker_pos = text.find(FENCE_END, max(start, 0))

        if start != -1 and end_marker_pos != -1:
            # Replace existing fenced block
            end = end_marker_pos + len(FENCE_END)
            new_text = text[:start] + new_content + text[end:]
        else:
            # Append as a new §-delimited entry
            separator = ENTRY_SEP if text.strip() else ""
            new_text = text + separator + new_content

        _atomic_write(self._memory_md, new_text)


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _primary_label(node) -> str:
    """Return the most specific label from a Graphiti EntityNode."""
    labels = getattr(node, "labels", None) or []
    # Skip generic graph labels
    skip = {"__Entity__", "Entity", "Node"}
    for label in labels:
        if label not in skip:
            return label
    return "Other"


def _atomic_write(path: Path, content: str) -> None:
    """Write content to path atomically via a temp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=".graphiti-index-")
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
