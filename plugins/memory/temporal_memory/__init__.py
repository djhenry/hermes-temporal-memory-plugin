"""Temporal knowledge-graph memory provider for Hermes Agent.

Implements the MemoryProvider interface with:
- Hybrid graph+vector+BM25 retrieval via a pluggable backend (ships with Graphiti/Zep)
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

from .config import TemporalMemoryConfig
from .memory_index import MemoryIndexManager

log = logging.getLogger(__name__)

try:
    from agent.memory_provider import MemoryProvider as _MemoryBase  # type: ignore[import]
except ImportError:
    _MemoryBase = object  # running outside Hermes (tests, standalone)

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
                "description": "Search depth hint (1=immediate facts, 2–3=broader context). Uses semantic search rather than strict BFS traversal.",
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
_INDEX_FENCE_START = "<!-- temporal-memory-index:start -->"
_INDEX_FENCE_END = "<!-- temporal-memory-index:end -->"

SYSTEM_PROMPT_BLOCK = (
    'The "Temporal Memory Index" section in MEMORY.md is automatically maintained '
    "by the temporal memory plugin. It summarises what topics and entities are "
    "available in deep memory. Do not edit or delete it manually. "
    "To retrieve full details, timelines, or historical facts, use the "
    "temporal_search, fact_history, or graph_browse tools."
)


# -------------------------------------------------------------------------


class TemporalMemoryProvider(_MemoryBase):
    """Hermes MemoryProvider backed by a temporal knowledge graph."""

    name = "temporal-memory"

    def __init__(self) -> None:
        self._client: Any = None
        self._group_id: str = ""
        self._cfg: TemporalMemoryConfig = TemporalMemoryConfig()
        self._index: MemoryIndexManager | None = None
        self._sync_thread: threading.Thread | None = None
        self._sync_lock = threading.Lock()
        # Persistent event loop for async client calls.
        # FalkorDB Lite's asyncio-based Redis client binds connections to the
        # event loop it's first used on. asyncio.run() creates-then-closes a
        # new loop on every call, so the second call finds the connection's
        # loop closed. A single long-lived loop avoids this entirely.
        self._async_loop: Any = None
        self._async_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Async loop management
    # ------------------------------------------------------------------

    def _start_async_loop(self) -> None:
        import asyncio
        self._async_loop = asyncio.new_event_loop()
        self._async_thread = threading.Thread(
            target=self._async_loop.run_forever,
            daemon=True,
            name="temporal-memory-async",
        )
        self._async_thread.start()

    def _run(self, coro: Any) -> Any:
        """Run a coroutine on the provider's persistent event loop."""
        import asyncio
        loop = self._async_loop
        if loop and loop.is_running():
            return asyncio.run_coroutine_threadsafe(coro, loop).result()
        if self._client is not None:
            # Loop is gone but client still exists: provider is shutting down or
            # has already shut down. Creating a new throwaway loop here would
            # break FalkorDB Lite's loop-bound connections. Close the coroutine
            # to prevent ResourceWarning and raise so callers can log/skip.
            coro.close()
            raise RuntimeError("temporal-memory async loop is not running (shutting down?)")
        return _run_sync(coro)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        # No network call — just check that at least one backend is configured.
        return bool(
            os.environ.get("GRAPHITI_NEO4J_URI")
            or os.environ.get("GRAPHITI_USE_KUZU")
            or os.environ.get("GRAPHITI_USE_FALKORDB_LITE")
            or self._cfg.backend in ("kuzu", "falkordblite")
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        log.info("temporal-memory: initialize() called for session %s", session_id)
        cfg_dict = kwargs.get("config", {})
        if not cfg_dict:
            # Hermes doesn't pass plugins.<name> config to initialize().
            # Read it from config.yaml directly.
            try:
                from hermes_cli.config import load_config, cfg_get
                full_cfg = load_config()
                cfg_dict = cfg_get(full_cfg, "plugins", "temporal-memory") or {}
            except Exception:
                cfg_dict = {}
        if isinstance(cfg_dict, dict):
            self._cfg = TemporalMemoryConfig(**cfg_dict)

        # Ensure OPENAI_API_KEY is in os.environ for Graphiti's embedder.
        # Hermes reads ~/.hermes/.env via load_env() but doesn't always
        # propagate values to os.environ. Graphiti's default OpenAI embedder
        # reads os.environ directly.
        if not os.environ.get("OPENAI_API_KEY"):
            try:
                from hermes_cli.config import load_env
                _key = load_env().get("OPENAI_API_KEY")
                if _key:
                    os.environ["OPENAI_API_KEY"] = _key
            except Exception:
                pass

        # Namespace: hermes-<profile>[-<platform_user_id>]
        identity = kwargs.get("identity") or os.environ.get("HERMES_PROFILE", "default")
        source = kwargs.get("session_source")
        user_suffix = f"-{source.user_id}" if source and getattr(source, "user_id", None) else ""
        self._group_id = f"hermes-{identity}{user_suffix}"

        self._start_async_loop()
        try:
            self._client = self._build_client()
        except Exception as exc:
            log.error("temporal-memory: _build_client() failed: %s", exc, exc_info=True)
            raise

        # KuzuDriver never sets _database (backend bug); patch it here so
        # the backend driver.s `group_id != driver._database` check doesn.t throw.
        _patch_kuzu_database(self._client, self._group_id)

        # Create graph indices/constraints (no-op for Kuzu; required for Neo4j/FalkorDB).
        try:
            self._run(self._client.build_indices_and_constraints())
        except Exception as exc:
            log.warning("temporal-memory build_indices_and_constraints failed (continuing): %s", exc)

        # Build the MEMORY.md index manager
        if self._cfg.enable_memory_index:
            hermes_home = Path(
                kwargs.get("hermes_home") or os.environ.get("HERMES_HOME", Path.home() / ".hermes")
            )
            memory_dir = hermes_home / "memories"
            self._index = MemoryIndexManager(memory_dir, max_tokens=self._cfg.max_index_tokens)
            log.debug("[temporal-memory] index manager created: dir=%s, max_tokens=%d", memory_dir, self._cfg.max_index_tokens)

            # Seed index from existing graph entities in a background thread
            # so initialize() itself never blocks on a network call.
            threading.Thread(
                target=self._seed_index_from_graph,
                daemon=True,
                name="temporal-memory-index-seed",
            ).start()
            log.debug("[temporal-memory] index seed thread started")
        else:
            log.debug("[temporal-memory] memory index disabled")

    def shutdown(self) -> None:
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5)
        if self._client:
            try:
                self._run(self._client.close())
            except Exception:
                pass
        # Detach the loop BEFORE stopping it: a concurrent or repeated
        # shutdown could otherwise submit close() to a loop that stops
        # before running it, blocking forever on the un-resolved future.
        loop, thread = self._async_loop, self._async_thread
        self._async_loop = None
        self._async_thread = None
        if loop and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if thread and thread.is_alive():
            thread.join(timeout=5)

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
            edges: list = self._search_edges(query, num_results=10)
        except Exception as exc:
            log.warning("temporal-memory prefetch failed: %s", exc)
            return None

        if not edges:
            return None

        lines = [f"{_CONTEXT_FENCE_TAG}[Temporal Memory — relevant facts]"]
        token_budget = self._cfg.max_recall_tokens
        for edge in edges:
            fact = getattr(edge, "fact", str(edge))
            valid_at = getattr(edge, "valid_at", None)
            invalid_at = getattr(edge, "invalid_at", None)
            window = _fmt_window(valid_at, invalid_at)
            line = f"- {window + ' ' if window else ''}{fact}"
            token_budget -= len(line) // 4
            if token_budget <= 0:
                break
            lines.append(line)

        lines.append(_CONTEXT_FENCE_END)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Turn ingestion
    # ------------------------------------------------------------------

    def sync_turn(self, user_content: str, assistant_content: str, messages: Any = None, session_id: str = "") -> None:
        log.info("[temporal-memory] sync_turn CALLED: session=%s, user_len=%d, assistant_len=%d", session_id, len(user_content), len(assistant_content))
        # Join previous sync thread with short timeout before starting the next
        with self._sync_lock:
            if self._sync_thread and self._sync_thread.is_alive():
                self._sync_thread.join(timeout=2)

            self._sync_thread = threading.Thread(
                target=self._ingest_turn,
                args=(user_content, assistant_content),
                daemon=True,
                name="temporal-memory-sync",
            )
            self._sync_thread.start()

    # ------------------------------------------------------------------
    # Built-in memory tool mirror
    # ------------------------------------------------------------------

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        # Mirror episodic memory writes to graph (Lesson 9).
        # USER.md writes are profile data, not episodic — skip them.
        if action in ("add", "replace") and target == "memory" and (content or "").strip():
            threading.Thread(
                target=self._mirror_memory_write,
                args=(action, target, content),
                daemon=True,
                name="temporal-memory-mirror",
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
            name="temporal-memory-index-refresh",
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
    # Internal search helper
    # ------------------------------------------------------------------

    def _search_edges(self, query: str, num_results: int = 10) -> list:
        """Run hybrid search; fall back to BM25-only when the embeddings
        endpoint is unavailable (e.g. OpenRouter proxy doesn't expose /embeddings)."""
        try:
            return self._run(self._client.search(
                query=query,
                group_ids=[self._group_id],
                num_results=num_results,
            ))
        except Exception as exc:
            err = str(exc).lower()
            # Only fall back to BM25 for connectivity / service-availability errors
            # (e.g. OpenRouter not exposing /embeddings). Re-raise auth errors,
            # schema errors, and programming bugs so they're not masked.
            if not any(w in err for w in (
                "connection", "unreachable", "unavailable", "timeout", "refused",
                "404", "not found", "not supported",
            )):
                raise
        # Embeddings endpoint unreachable — retry with BM25-only search config.
        try:
            from graphiti_core.search.search_config import (
                EdgeSearchConfig, EdgeSearchMethod, EdgeReranker, SearchConfig,
            )
            bm25_cfg = SearchConfig(edge_config=EdgeSearchConfig(
                search_methods=[EdgeSearchMethod.bm25],
                reranker=EdgeReranker.rrf,
            ))
            bm25_cfg.limit = num_results
            result = self._run(self._client.search_(
                query=query,
                group_ids=[self._group_id],
                config=bm25_cfg,
            ))
            return result.edges
        except Exception as exc:
            log.warning("temporal-memory BM25 fallback search failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _temporal_search(self, query: str, as_of: str | None = None) -> str:
        try:
            edges: list = self._search_edges(query, num_results=15)
        except Exception as exc:
            return f"temporal_search error: {exc}"

        if not edges:
            return "No facts found."

        as_of_dt = None
        if as_of:
            try:
                as_of_dt = datetime.fromisoformat(as_of).replace(tzinfo=timezone.utc)
            except ValueError:
                return f"temporal_search error: invalid as_of date '{as_of}' — use ISO-8601 (YYYY-MM-DD)"

        lines = []
        for edge in edges:
            fact = getattr(edge, "fact", str(edge))
            valid_at = getattr(edge, "valid_at", None)
            invalid_at = getattr(edge, "invalid_at", None)
            if as_of_dt:
                # Filter client-side if the backend didn't
                if valid_at and valid_at > as_of_dt:
                    continue
                if invalid_at and invalid_at < as_of_dt:
                    continue
            window = _fmt_window(valid_at, invalid_at)
            lines.append(f"- {window + ' ' if window else ''}{fact}")

        return "\n".join(lines) if lines else "No facts valid at that time."

    def _fact_history(self, entity: str, relationship: str | None = None) -> str:
        try:
            edges: list = self._search_edges(entity, num_results=50)
        except Exception as exc:
            return f"fact_history error: {exc}"

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
            window = _fmt_window(valid_at, invalid_at, unknown="[unknown period]")
            lines.append(f"  {window} {fact}")

        return "\n".join(lines)

    def _graph_browse(self, entity: str, depth: int = 1) -> str:
        try:
            edges: list = self._search_edges(entity, num_results=20)
        except Exception as exc:
            return f"graph_browse error: {exc}"

        if not edges:
            return f"No connections found for '{entity}'."

        lines = [f"Connections for '{entity}':"]
        for edge in edges[:20]:
            lines.append(f"  - {getattr(edge, 'fact', str(edge))}")
        return "\n".join(lines)

    def _fact_correct(self, entity: str, relationship: str, correction: str) -> str:
        try:
            self._run(self._client.add_episode(
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
            result = self._run(self._client.add_episode(
                name="conversation_turn",
                episode_body=episode_body,
                source_description="hermes_conversation",
                reference_time=datetime.now(timezone.utc),
                group_id=self._group_id,
            ))
            if result:
                new_nodes = getattr(result, "nodes", []) or []
                new_edges = getattr(result, "edges", []) or []

                log.debug(
                    "[temporal-memory] sync_turn: extracted %d nodes, %d edges",
                    len(new_nodes), len(new_edges),
                )
                for node in new_nodes:
                    log.debug(
                        "[temporal-memory] new node: name=%s labels=%s",
                        getattr(node, "name", "?"),
                        getattr(node, "labels", []),
                    )
                for edge in new_edges:
                    if getattr(edge, "invalid_at", None):
                        log.info(
                            "[temporal-memory] Superseded: %s",
                            getattr(edge, "fact", "unknown fact"),
                        )

                # Update recall-trigger index with newly extracted entities
                if self._index and (new_nodes or new_edges):
                    log.debug("[temporal-memory] updating index with %d nodes, %d edges", len(new_nodes), len(new_edges))
                    self._index.update_from_episode(new_nodes, new_edges)
                    log.info(
                        "[temporal-memory] index updated: +%d nodes, +%d edges (total entities: see MEMORY.md)",
                        len(new_nodes), len(new_edges),
                    )
                elif not self._index:
                    log.debug("[temporal-memory] index disabled (enable_memory_index=False)")
        except Exception as exc:
            log.warning("temporal-memory sync_turn ingestion failed: %s", exc)

    def _mirror_memory_write(self, action: str, target: str, content: str) -> None:
        try:
            self._run(self._client.add_episode(
                name="memory_tool_write",
                episode_body=f"Memory tool {action} on {target}: {content}",
                source_description="memory_tool",
                reference_time=datetime.now(timezone.utc),
                group_id=self._group_id,
            ))
        except Exception as exc:
            log.warning("temporal-memory memory mirror failed: %s", exc)

    def _seed_index_from_graph(self) -> None:
        if not self._index or not self._client:
            log.debug("[temporal-memory] seed_index: skipped (index=%s, client=%s)", bool(self._index), bool(self._client))
            return
        _NODE_LIMIT = 500
        _EDGE_LIMIT = 1000
        try:
            log.debug("[temporal-memory] seed_index: fetching nodes/edges for group %s", self._group_id)
            nodes = self._run(self._client.nodes.entity.get_by_group_ids(
                group_ids=[self._group_id],
                limit=_NODE_LIMIT,
            ))
            edges = self._run(self._client.edges.entity.get_by_group_ids(
                group_ids=[self._group_id],
                limit=_EDGE_LIMIT,
            ))
            node_count = len(nodes or [])
            edge_count = len(edges or [])
            log.debug("[temporal-memory] seed_index: fetched %d nodes, %d edges", node_count, edge_count)
            if nodes and len(nodes) >= _NODE_LIMIT:
                log.warning(
                    "temporal-memory index seed: fetched %d nodes (limit). Some entities may be "
                    "missing '· history available' hints. Increase limit or run full refresh.",
                    _NODE_LIMIT,
                )
            # Log entity names for debugging
            for node in (nodes or [])[:20]:
                labels = getattr(node, "labels", []) or []
                skip = {"__Entity__", "Entity", "Node"}
                label = next((l for l in labels if l not in skip), "Other")
                log.debug(
                    "[temporal-memory] seed entity: name=%s label=%s",
                    getattr(node, "name", "?"),
                    label,
                )
            self._index.seed_from_graph(nodes or [], edges or [])
            log.info("[temporal-memory] index seeded: %d entities, %d facts from graph", node_count, edge_count)
        except Exception as exc:
            log.warning("temporal-memory index seed failed: %s", exc)

    def _full_index_refresh(self) -> None:
        if not self._index or not self._client:
            log.debug("[temporal-memory] full_refresh: skipped (index=%s, client=%s)", bool(self._index), bool(self._client))
            return
        try:
            log.debug("[temporal-memory] full_refresh: building communities for group %s", self._group_id)
            communities, _ = self._run(self._client.build_communities(
                group_ids=[self._group_id]
            ))
            log.debug("[temporal-memory] full_refresh: built %d communities", len(communities or []))
            for comm in (communities or [])[:10]:
                log.debug(
                    "[temporal-memory] community: name=%s",
                    getattr(comm, "name", "?"),
                )
            self._index.full_refresh(communities or [])
            log.debug("[temporal-memory] full_refresh: index rebuilt from communities")
        except Exception as exc:
            log.warning("temporal-memory index full refresh failed: %s", exc)

    # ------------------------------------------------------------------
    # Client factory
    # ------------------------------------------------------------------

    def _build_client(self) -> Any:
        from graphiti_core import Graphiti  # type: ignore[import]

        llm_client = _build_llm_client(self._cfg.extraction)
        embedder = _build_embedder(self._cfg.embedder)
        backend = self._cfg.backend
        use_kuzu = os.environ.get("GRAPHITI_USE_KUZU") or backend == "kuzu"
        use_falkordblite = os.environ.get("GRAPHITI_USE_FALKORDB_LITE") or backend == "falkordblite"

        # Common kwargs for all backends
        common_kwargs: dict[str, Any] = {}
        if llm_client:
            common_kwargs["llm_client"] = llm_client
        if embedder:
            common_kwargs["embedder"] = embedder

        if use_kuzu:
            import warnings
            warnings.warn(
                "The Kuzu backend for graphiti-core is deprecated and will be removed in a "
                "future release. Migrate to Neo4j (Docker) or FalkorDB Lite (Python 3.12+).",
                DeprecationWarning,
                stacklevel=3,
            )
            from graphiti_core.driver.kuzu_driver import KuzuDriver  # type: ignore[import]
            db_path = os.environ.get(
                "GRAPHITI_KUZU_PATH",
                str(Path.home() / ".hermes" / "graphiti.kuzu"),
            )
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            kuzu_driver = KuzuDriver(db=db_path)
            _create_kuzu_fts_indices(kuzu_driver)
            common_kwargs["graph_driver"] = kuzu_driver
            return Graphiti(**common_kwargs)

        if use_falkordblite:
            # Requires Python 3.12+ and pip install graphiti-core[falkordblite]
            import sys
            if sys.version_info < (3, 12):
                raise RuntimeError(
                    f"FalkorDB Lite requires Python 3.12+, but the gateway is running "
                    f"Python {sys.version_info.major}.{sys.version_info.minor}. "
                    f"Recreate the gateway venv with Python 3.12+:\n"
                    f"  cd ~/.hermes/hermes-agent && uv venv --python python3.12\n"
                    f"  source .venv/bin/activate\n"
                    f"  pip install -e '.[all]' 'graphiti-core[falkordblite]'\n"
                    f"  hermes gateway restart"
                )
            from graphiti_core.driver.falkordb_driver import FalkorDriver  # type: ignore[import]
            try:
                from redislite.async_falkordb_client import AsyncFalkorDB  # type: ignore[import]
            except ImportError as exc:
                raise RuntimeError(
                    "FalkorDB Lite requires graphiti-core[falkordblite]. "
                    "Run: pip install 'graphiti-core[falkordblite]'"
                ) from exc
            db_path = os.environ.get(
                "GRAPHITI_FALKORDBLITE_PATH",
                str(Path.home() / ".hermes" / "graphiti.fdb"),
            )
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            falkor_client = AsyncFalkorDB(dbfilename=db_path)
            common_kwargs["graph_driver"] = FalkorDriver(falkor_db=falkor_client)
            return Graphiti(**common_kwargs)

        # Neo4j — default for production / multi-user / gateway deployments
        common_kwargs["uri"] = os.environ.get("GRAPHITI_NEO4J_URI", "bolt://localhost:7687")
        common_kwargs["user"] = os.environ.get("GRAPHITI_NEO4J_USER", "neo4j")
        common_kwargs["password"] = os.environ.get("GRAPHITI_NEO4J_PASSWORD", "password")
        return Graphiti(**common_kwargs)


    # ------------------------------------------------------------------
    # Setup wizard
    # ------------------------------------------------------------------

    def post_setup(self, hermes_home: str, config: dict | None = None) -> None:
        """Interactive setup wizard called by `hermes setup`.

        Prompts for backend choice and writes the minimum required env vars
        to ~/.hermes/.env so the plugin works on next launch.
        """
        home = Path(hermes_home)
        env_file = home / ".env"

        print("\n=== Temporal Memory Plugin Setup ===\n")
        print("Backend options:")
        print("  1. neo4j        — Neo4j via Docker (recommended for production)")
        print("  2. kuzu         — embedded, no Docker, Python 3.11+ (deprecated)")
        print("  3. falkordblite — embedded, no Docker, Python 3.12+")
        choice = input("\nChoose backend [1/2/3, default=1]: ").strip() or "1"

        lines: list[str] = []

        if choice == "2":
            lines.append("GRAPHITI_USE_KUZU=1")
            db_path = input(
                f"Kuzu DB path [default: {home / 'graphiti.kuzu'}]: "
            ).strip() or str(home / "graphiti.kuzu")
            lines.append(f"GRAPHITI_KUZU_PATH={db_path}")
            backend_name = "kuzu"
        elif choice == "3":
            lines.append("GRAPHITI_USE_FALKORDB_LITE=1")
            db_path = input(
                f"FalkorDB path [default: {home / 'graphiti.fdb'}]: "
            ).strip() or str(home / "graphiti.fdb")
            lines.append(f"GRAPHITI_FALKORDBLITE_PATH={db_path}")
            backend_name = "falkordblite"
        else:
            uri = input("Neo4j URI [default: bolt://localhost:7687]: ").strip() or "bolt://localhost:7687"
            user = input("Neo4j user [default: neo4j]: ").strip() or "neo4j"
            password = input("Neo4j password [default: password]: ").strip() or "password"
            lines += [
                f"GRAPHITI_NEO4J_URI={uri}",
                f"GRAPHITI_NEO4J_USER={user}",
                f"GRAPHITI_NEO4J_PASSWORD={password}",
            ]
            backend_name = "neo4j"

        print("\nExtraction LLM (used to extract entities from conversations):")
        print("  1. inherit  — reuse Hermes's active model (simplest)")
        print("  2. openai   — dedicated OpenAI key")
        print("  3. ollama   — local model, nothing leaves device")
        print("  4. other    — skip (configure manually in config.yaml)")
        llm_choice = input("\nChoose extraction LLM [1/2/3/4, default=1]: ").strip() or "1"

        if llm_choice == "2":
            key = input("OPENAI_API_KEY: ").strip()
            if key:
                lines.append(f"OPENAI_API_KEY={key}")
        elif llm_choice == "3":
            base_url = input("Ollama base URL [default: http://localhost:11434]: ").strip() or "http://localhost:11434"
            model = input("Ollama model [default: llama3.1:8b]: ").strip() or "llama3.1:8b"
            lines += [
                f"GRAPHITI_EXTRACTION_PROVIDER=ollama",
                f"GRAPHITI_EXTRACTION_MODEL={model}",
                f"GRAPHITI_EXTRACTION_BASE_URL={base_url}",
            ]

        # Append to .env (create if missing)
        existing = env_file.read_text() if env_file.exists() else ""
        if existing and not existing.endswith("\n"):
            existing += "\n"
        new_block = "\n# hermes-temporal-memory\n" + "\n".join(lines) + "\n"
        env_file.write_text(existing + new_block)

        print(f"\nWrote {len(lines)} variable(s) to {env_file}")
        print(f"Backend: {backend_name}")
        print("\nTo activate: set  memory.provider: temporal-memory  in ~/.hermes/config.yaml")
        print("or run:  hermes plugins enable temporal-memory\n")


# ------------------------------------------------------------------
# Hermes plugin entry point
# ------------------------------------------------------------------

def register(ctx: Any = None) -> TemporalMemoryProvider:
    """Entry point for Hermes plugin discovery.

    Hermes calls ``register(collector)`` where *collector* has a
    ``register_memory_provider`` method.  We support both the
    collector-based call and a no-arg call for standalone use.
    """
    provider = TemporalMemoryProvider()
    if ctx is not None and hasattr(ctx, "register_memory_provider"):
        ctx.register_memory_provider(provider)
    return provider


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _fmt_date(dt: datetime | None) -> str:
    if dt is None:
        return "?"
    return dt.strftime("%Y-%m")


def _fmt_window(
    valid_at: datetime | None,
    invalid_at: datetime | None,
    unknown: str = "",
) -> str:
    """Format a bi-temporal validity window as a bracketed string."""
    if valid_at and invalid_at:
        return f"[{_fmt_date(valid_at)} → {_fmt_date(invalid_at)}]"
    if valid_at:
        return f"[{_fmt_date(valid_at)} → now]"
    return unknown


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
    # Strip <!-- temporal-memory-index:start -->...<!-- temporal-memory-index:end -->
    text = re.sub(
        r"<!--\s*temporal-memory-index:start\s*-->.*?<!--\s*temporal-memory-index:end\s*-->",
        "",
        text,
        flags=re.DOTALL,
    )
    return text.strip()


def _patch_kuzu_database(client: Any, group_id: str) -> None:
    """Set _database on KuzuDriver if absent.

    the backend driver compares group_id to driver._database before deciding whether
    to clone the driver. KuzuDriver never assigns _database (backend bug),
    so the attribute access throws AttributeError. Setting it to the group_id
    makes the check a no-op for Kuzu, which is correct: Kuzu is single-file,
    not per-group, so no driver cloning is ever needed.
    """
    try:
        driver = client.driver
        if "Kuzu" in type(driver).__name__ and not hasattr(driver, "_database"):
            driver._database = group_id
    except Exception:
        pass


def _create_kuzu_fts_indices(kuzu_driver: Any) -> None:
    """Create FTS indices required by Kuzu search operations.

    KuzuDriver.setup_schema() creates node/edge tables but not the FTS indices.
    KuzuDriver.build_indices_and_constraints() is a no-op (backend bug).
    Without these indices, every search call fails with "table doesn't have an
    index with name edge_name_and_fact" (and similar for other tables).

    Queries mirror the backend graph_queries.get_fulltext_indices(KUZU).
    "Already exists" errors on subsequent runs are silently ignored.
    """
    try:
        from graphiti_core.graph_queries import get_fulltext_indices  # type: ignore[import]
        from graphiti_core.driver.driver import GraphProvider  # type: ignore[import]
        import kuzu as _kuzu  # type: ignore[import]

        conn = _kuzu.Connection(kuzu_driver.db)
        for query in get_fulltext_indices(GraphProvider.KUZU):
            try:
                conn.execute(query)
            except Exception as exc:
                if "already exist" not in str(exc).lower():
                    log.debug("Kuzu FTS index: %s", exc)
        conn.close()
    except Exception as exc:
        log.warning("Could not create Kuzu FTS indices: %s", exc)


def _build_llm_client(extraction: Any = None) -> Any | None:
    """Build a Graphiti LLMClient from config or env vars.

    Uses OpenAIGenericClient which works with any OpenAI-compatible endpoint
    (OpenRouter, LM Studio, Ollama, etc.) via standard json_schema/json_object
    response_format instead of OpenAI's native structured output API.
    """
    model = (
        (extraction and getattr(extraction, "model", None))
        or os.environ.get("GRAPHITI_LLM_MODEL")
        or os.environ.get("GRAPHITI_EXTRACTION_MODEL")
    )
    if not model:
        return None

    try:
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient  # type: ignore[import]
        from graphiti_core.llm_client.config import LLMConfig  # type: ignore[import]

        cfg = LLMConfig(model=model)
        base_url = (
            (extraction and getattr(extraction, "base_url", None))
            or os.environ.get("OPENAI_BASE_URL")
        )
        if base_url:
            cfg.base_url = base_url

        # Prefer an explicit api_key from extraction config (e.g. "lm-studio" for
        # local providers), then fall back to OPENAI_API_KEY from env or hermes .env.
        api_key = (extraction and getattr(extraction, "api_key", None)) or None
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            try:
                from hermes_cli.config import load_env
                api_key = load_env().get("OPENAI_API_KEY")
            except Exception:
                pass
        if api_key:
            cfg.api_key = api_key

        # structured_output_mode: json_schema for local providers (LM Studio, Ollama);
        # json_object for OpenRouter and other proxies that don't support json_schema.
        output_mode = (extraction and getattr(extraction, "structured_output_mode", None)) or "json_schema"
        return OpenAIGenericClient(cfg, structured_output_mode=output_mode)
    except Exception as exc:
        log.debug("Could not build custom LLM client (%s); using Graphiti default.", exc)
        return None


def _build_embedder(embedder_cfg: Any = None) -> Any | None:
    """Build a Graphiti EmbedderClient from config.

    If no embedder config is provided, falls back to Graphiti's default
    OpenAIEmbedder (reads OPENAI_API_KEY from env). When the extraction
    provider uses OpenRouter (which has no /embeddings endpoint), we
    must configure a separate embedding endpoint — otherwise the default
    embedder fails with 401.
    """
    model = (embedder_cfg and getattr(embedder_cfg, "model", None)) or None
    base_url = (embedder_cfg and getattr(embedder_cfg, "base_url", None)) or None
    api_key = (embedder_cfg and getattr(embedder_cfg, "api_key", None)) or None

    # If nothing configured, let Graphiti use its default
    if not model and not base_url:
        return None

    try:
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

        embedder_config = OpenAIEmbedderConfig()
        if model:
            embedder_config.embedding_model = model
        if base_url:
            embedder_config.base_url = base_url
        if api_key:
            embedder_config.api_key = api_key

        return OpenAIEmbedder(config=embedder_config)
    except Exception as exc:
        log.debug("Could not build custom embedder (%s); using Graphiti default.", exc)
        return None


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
