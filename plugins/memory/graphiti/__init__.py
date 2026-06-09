"""Graphiti temporal knowledge-graph memory provider for Hermes Agent.

Implements the MemoryProvider interface with:
- Hybrid graph+vector+BM25 retrieval via Graphiti (Zep)
- Bi-temporal fact storage (valid_at / invalid_at windows)
- MEMORY.md recall-trigger index: a plugin-owned section that surfaces
  what topics are available in deep memory without bloating the prompt
- Non-blocking ingestion (daemon thread)
- Context-fencing to prevent graph self-pollution
- Per-identity group namespacing (profile + platform user)
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import GraphitiConfig
from .memory_index import MemoryIndexManager

log = logging.getLogger(__name__)

# Tool JSON schemas -------------------------------------------------------

TEMPORAL_SEARCH_SCHEMA = {
    "name": "temporal_search",
    "description": (
        "Search the temporal knowledge graph for facts about entities, relationships, "
        "or events. Optionally restrict to facts that were valid at a specific point in time."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language search query"},
            "as_of": {
                "type": "string",
                "description": "ISO-8601 date (YYYY-MM-DD). When supplied, returns only facts valid at that date.",
                "nullable": True,
            },
        },
        "required": ["query"],
    },
}

FACT_HISTORY_SCHEMA = {
    "name": "fact_history",
    "description": (
        "Return the complete timeline for an entity or relationship — all versions "
        "with validity windows, ordered from oldest to newest."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "entity": {"type": "string", "description": "Entity name (e.g. 'user', 'Alice', 'Acme Corp')"},
            "relationship": {
                "type": "string",
                "description": "Optional relationship type to filter (e.g. 'LIVES_IN', 'WORKS_AT')",
                "nullable": True,
            },
        },
        "required": ["entity"],
    },
}

GRAPH_BROWSE_SCHEMA = {
    "name": "graph_browse",
    "description": (
        "Return the immediate neighborhood of an entity in the knowledge graph. "
        "Useful for multi-hop questions: who introduced me to X, what projects is Y on, etc."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "entity": {"type": "string", "description": "Entity name to look up"},
            "depth": {
                "type": "integer",
                "description": "How many hops to traverse (1–3, default 1)",
                "minimum": 1,
                "maximum": 3,
                "default": 1,
            },
        },
        "required": ["entity"],
    },
}

FACT_CORRECT_SCHEMA = {
    "name": "fact_correct",
    "description": (
        "Invalidate an incorrect fact and record the correction. Use when the user "
        "says a previously recorded fact was never true."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "entity": {"type": "string"},
            "relationship": {"type": "string"},
            "correction": {"type": "string", "description": "The correct fact to record instead"},
        },
        "required": ["entity", "relationship", "correction"],
    },
}

# Memory-context fence tag (stripped before graph ingestion) --------------
_CONTEXT_FENCE_TAG = "<memory-context>"
_CONTEXT_FENCE_END = "</memory-context>"
_INDEX_FENCE_START = "<!-- graphiti-index:start -->"
_INDEX_FENCE_END = "<!-- graphiti-index:end -->"

SYSTEM_PROMPT_BLOCK = (
    'The "Temporal Memory Index" section in MEMORY.md is automatically maintained '
    "by the graphiti memory plugin. It summarises what topics and entities are "
    "available in deep memory. Do not edit or delete it manually. "
    "To retrieve full details, timelines, or historical facts, use the "
    "temporal_search, fact_history, or graph_browse tools."
)


# -------------------------------------------------------------------------


class GraphitiMemoryProvider:
    """Hermes MemoryProvider backed by Graphiti temporal knowledge graph."""

    name = "graphiti"

    def __init__(self) -> None:
        self._client: Any = None
        self._group_id: str = ""
        self._cfg: GraphitiConfig = GraphitiConfig()
        self._index: MemoryIndexManager | None = None
        self._sync_thread: threading.Thread | None = None
        self._sync_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        # No network call — just check that at least one backend is configured.
        return bool(
            os.environ.get("GRAPHITI_NEO4J_URI")
            or os.environ.get("GRAPHITI_USE_SQLITE")
            or self._cfg.backend == "sqlite"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        cfg_dict = kwargs.get("config", {})
        if isinstance(cfg_dict, dict):
            self._cfg = GraphitiConfig(**cfg_dict)

        # Namespace: hermes-<profile>[-<platform_user_id>]
        identity = kwargs.get("identity") or os.environ.get("HERMES_PROFILE", "default")
        source = kwargs.get("session_source")
        user_suffix = f"-{source.user_id}" if source and getattr(source, "user_id", None) else ""
        self._group_id = f"hermes-{identity}{user_suffix}"

        self._client = self._build_client()

        # Build the MEMORY.md index manager
        if self._cfg.enable_memory_index:
            hermes_home = Path(
                kwargs.get("hermes_home") or os.environ.get("HERMES_HOME", Path.home() / ".hermes")
            )
            memory_dir = hermes_home / "memories"
            self._index = MemoryIndexManager(memory_dir)

            # Seed index from existing graph entities in a background thread
            # so initialize() itself never blocks on a network call.
            threading.Thread(
                target=self._seed_index_from_graph,
                daemon=True,
                name="graphiti-index-seed",
            ).start()

    def shutdown(self) -> None:
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5)
        if self._client:
            try:
                import asyncio
                asyncio.get_event_loop().run_until_complete(self._client.close())
            except Exception:
                pass

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def system_prompt_block(self) -> str | None:
        if not self._cfg.enable_memory_index:
            return None
        return SYSTEM_PROMPT_BLOCK

    # ------------------------------------------------------------------
    # Context retrieval
    # ------------------------------------------------------------------

    def prefetch(self, query: str) -> str | None:
        if self._cfg.recall_mode == "tools":
            return None
        try:
            results = _run_sync(self._client.search(
                query=query,
                group_ids=[self._group_id],
                num_results=10,
            ))
        except Exception as exc:
            log.warning("graphiti prefetch failed: %s", exc)
            return None

        edges = results if isinstance(results, list) else getattr(results, "edges", [])
        if not edges:
            return None

        lines = [f"{_CONTEXT_FENCE_TAG}[Temporal Memory — relevant facts]"]
        token_budget = self._cfg.max_recall_tokens
        for edge in edges:
            fact = getattr(edge, "fact", str(edge))
            valid_at = getattr(edge, "valid_at", None)
            invalid_at = getattr(edge, "invalid_at", None)

            if valid_at and invalid_at:
                window = f"[{_fmt_date(valid_at)} → {_fmt_date(invalid_at)}]"
            elif valid_at:
                window = f"[{_fmt_date(valid_at)} → now]"
            else:
                window = ""

            line = f"- {window} {fact}".strip()
            token_budget -= len(line) // 4
            if token_budget <= 0:
                break
            lines.append(line)

        lines.append(_CONTEXT_FENCE_END)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Turn ingestion
    # ------------------------------------------------------------------

    def sync_turn(self, user_content: str, assistant_content: str, messages: Any = None) -> None:
        # Join previous sync thread with short timeout before starting the next
        with self._sync_lock:
            if self._sync_thread and self._sync_thread.is_alive():
                self._sync_thread.join(timeout=2)

            self._sync_thread = threading.Thread(
                target=self._ingest_turn,
                args=(user_content, assistant_content),
                daemon=True,
                name="graphiti-sync",
            )
            self._sync_thread.start()

    # ------------------------------------------------------------------
    # Built-in memory tool mirror
    # ------------------------------------------------------------------

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        # Mirror to graph (tagged as source: memory_tool per Lesson 9)
        if action in ("add", "replace") and (content or "").strip():
            threading.Thread(
                target=self._mirror_memory_write,
                args=(action, target, content),
                daemon=True,
                name="graphiti-mirror",
            ).start()

        # Update the recall-trigger index
        if self._index:
            self._index.on_memory_write(action, target, content or "")

    # ------------------------------------------------------------------
    # Session boundary
    # ------------------------------------------------------------------

    def on_session_end(self) -> None:
        if not self._index or not self._client:
            return
        threading.Thread(
            target=self._full_index_refresh,
            daemon=True,
            name="graphiti-index-refresh",
        ).start()

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def get_tool_schemas(self) -> list[dict]:
        return [
            TEMPORAL_SEARCH_SCHEMA,
            FACT_HISTORY_SCHEMA,
            GRAPH_BROWSE_SCHEMA,
            FACT_CORRECT_SCHEMA,
        ]

    def handle_tool_call(self, name: str, args: dict) -> str:
        if name == "temporal_search":
            return self._temporal_search(**args)
        if name == "fact_history":
            return self._fact_history(**args)
        if name == "graph_browse":
            return self._graph_browse(**args)
        if name == "fact_correct":
            return self._fact_correct(**args)
        return f"Unknown tool: {name}"

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _temporal_search(self, query: str, as_of: str | None = None) -> str:
        try:
            results = _run_sync(self._client.search(
                query=query,
                group_ids=[self._group_id],
                num_results=15,
            ))
        except Exception as exc:
            return f"temporal_search error: {exc}"

        edges = results if isinstance(results, list) else getattr(results, "edges", [])
        if not edges:
            return "No facts found."

        lines = []
        for edge in edges:
            fact = getattr(edge, "fact", str(edge))
            valid_at = getattr(edge, "valid_at", None)
            invalid_at = getattr(edge, "invalid_at", None)
            window = ""
            if valid_at and invalid_at:
                window = f"[{_fmt_date(valid_at)} → {_fmt_date(invalid_at)}] "
            elif valid_at:
                window = f"[{_fmt_date(valid_at)} → now] "
            if as_of:
                # Filter client-side if the backend didn't
                as_of_dt = datetime.fromisoformat(as_of).replace(tzinfo=timezone.utc)
                if valid_at and valid_at > as_of_dt:
                    continue
                if invalid_at and invalid_at < as_of_dt:
                    continue
            lines.append(f"- {window}{fact}")

        return "\n".join(lines) if lines else "No facts valid at that time."

    def _fact_history(self, entity: str, relationship: str | None = None) -> str:
        try:
            # Search for all facts about this entity
            results = _run_sync(self._client.search(
                query=entity,
                group_ids=[self._group_id],
                num_results=50,
            ))
        except Exception as exc:
            return f"fact_history error: {exc}"

        edges = results if isinstance(results, list) else getattr(results, "edges", [])
        # Filter edges that mention the entity
        entity_lower = entity.lower()
        relevant = [
            e for e in edges
            if entity_lower in getattr(e, "fact", "").lower()
        ]
        if relationship:
            rel_lower = relationship.lower()
            relevant = [e for e in relevant if rel_lower in getattr(e, "name", "").lower()]

        if not relevant:
            return f"No history found for '{entity}'."

        # Sort by valid_at ascending
        relevant.sort(key=lambda e: getattr(e, "valid_at", None) or datetime.min.replace(tzinfo=timezone.utc))

        lines = [f"Timeline for '{entity}'" + (f" / {relationship}" if relationship else "") + ":"]
        for edge in relevant:
            fact = getattr(edge, "fact", str(edge))
            valid_at = getattr(edge, "valid_at", None)
            invalid_at = getattr(edge, "invalid_at", None)
            if valid_at and invalid_at:
                window = f"[{_fmt_date(valid_at)} → {_fmt_date(invalid_at)}]"
            elif valid_at:
                window = f"[{_fmt_date(valid_at)} → now]"
            else:
                window = "[unknown period]"
            lines.append(f"  {window} {fact}")

        return "\n".join(lines)

    def _graph_browse(self, entity: str, depth: int = 1) -> str:
        try:
            results = _run_sync(self._client.search(
                query=entity,
                group_ids=[self._group_id],
                num_results=20,
            ))
        except Exception as exc:
            return f"graph_browse error: {exc}"

        edges = results if isinstance(results, list) else getattr(results, "edges", [])
        if not edges:
            return f"No connections found for '{entity}'."

        lines = [f"Connections for '{entity}':"]
        for edge in edges[:20]:
            lines.append(f"  - {getattr(edge, 'fact', str(edge))}")
        return "\n".join(lines)

    def _fact_correct(self, entity: str, relationship: str, correction: str) -> str:
        try:
            _run_sync(self._client.add_episode(
                name="user_correction",
                episode_body=(
                    f"Correction: the fact '{relationship}' for '{entity}' was incorrect. "
                    f"The correct fact is: {correction}"
                ),
                source_description="user_correction",
                reference_time=datetime.now(timezone.utc),
                group_id=self._group_id,
            ))
            return f"Recorded correction for '{entity}' / {relationship}."
        except Exception as exc:
            return f"fact_correct error: {exc}"

    # ------------------------------------------------------------------
    # Background tasks
    # ------------------------------------------------------------------

    def _ingest_turn(self, user_content: str, assistant_content: str) -> None:
        # Context-fence: strip injected <memory-context> block and index section
        # before ingesting so the graph never re-ingests its own recalled output.
        clean_user = _strip_fences(user_content)
        clean_assistant = _strip_fences(assistant_content)

        episode_body = f"User: {clean_user}\nAssistant: {clean_assistant}"

        try:
            result = _run_sync(self._client.add_episode(
                name="conversation_turn",
                episode_body=episode_body,
                source_description="hermes_conversation",
                reference_time=datetime.now(timezone.utc),
                group_id=self._group_id,
            ))
            # Update recall-trigger index with newly extracted entities
            if self._index and result:
                new_nodes = getattr(result, "nodes", [])
                new_edges = getattr(result, "edges", [])
                if new_nodes or new_edges:
                    self._index.update_from_episode(new_nodes, new_edges)
        except Exception as exc:
            log.warning("graphiti sync_turn ingestion failed: %s", exc)

    def _mirror_memory_write(self, action: str, target: str, content: str) -> None:
        try:
            _run_sync(self._client.add_episode(
                name="memory_tool_write",
                episode_body=f"Memory tool {action} on {target}: {content}",
                source_description="memory_tool",
                reference_time=datetime.now(timezone.utc),
                group_id=self._group_id,
            ))
        except Exception as exc:
            log.warning("graphiti memory mirror failed: %s", exc)

    def _seed_index_from_graph(self) -> None:
        if not self._index or not self._client:
            return
        try:
            nodes = _run_sync(self._client.nodes.entity.get_by_group_ids(
                group_ids=[self._group_id],
                limit=200,
            ))
            edges = _run_sync(self._client.edges.entity.get_by_group_ids(
                group_ids=[self._group_id],
                limit=500,
            ))
            self._index.seed_from_graph(nodes or [], edges or [])
        except Exception as exc:
            log.warning("graphiti index seed failed: %s", exc)

    def _full_index_refresh(self) -> None:
        if not self._index or not self._client:
            return
        try:
            communities, _ = _run_sync(self._client.build_communities(
                group_ids=[self._group_id]
            ))
            self._index.full_refresh(communities or [])
        except Exception as exc:
            log.warning("graphiti index full refresh failed: %s", exc)

    # ------------------------------------------------------------------
    # Client factory
    # ------------------------------------------------------------------

    def _build_client(self) -> Any:
        from graphiti_core import Graphiti  # type: ignore[import]

        if self._cfg.backend == "sqlite" or os.environ.get("GRAPHITI_USE_SQLITE"):
            # SQLite backend — no Docker needed
            db_path = os.environ.get(
                "GRAPHITI_SQLITE_PATH",
                str(Path.home() / ".hermes" / "graphiti.db"),
            )
            return Graphiti(database_url=f"sqlite:///{db_path}")

        return Graphiti(
            uri=os.environ.get("GRAPHITI_NEO4J_URI", "bolt://localhost:7687"),
            user=os.environ.get("GRAPHITI_NEO4J_USER", "neo4j"),
            password=os.environ.get("GRAPHITI_NEO4J_PASSWORD", "password"),
        )


# ------------------------------------------------------------------
# Hermes plugin entry point
# ------------------------------------------------------------------

def register() -> GraphitiMemoryProvider:
    return GraphitiMemoryProvider()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _fmt_date(dt: datetime | None) -> str:
    if dt is None:
        return "?"
    return dt.strftime("%Y-%m")


def _strip_fences(text: str) -> str:
    """Remove injected context fences from text before graph ingestion."""
    import re

    # Strip <memory-context>...</memory-context>
    text = re.sub(
        r"<memory-context>.*?</memory-context>",
        "",
        text,
        flags=re.DOTALL,
    )
    # Strip <!-- graphiti-index:start -->...<!-- graphiti-index:end -->
    text = re.sub(
        r"<!--\s*graphiti-index:start\s*-->.*?<!--\s*graphiti-index:end\s*-->",
        "",
        text,
        flags=re.DOTALL,
    )
    return text.strip()


def _run_sync(coro: Any) -> Any:
    """Run an async coroutine from a sync context."""
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already in an async context — use a new thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)
