# Project Plan: Temporal Knowledge Graph Memory for Hermes Agent

**Approach:** Plugin (preferred) with a fork option for deeper integration  
**Target stack:** Graphiti + Neo4j (gateway/scale) or SQLite (single-user local) — both ship in one plugin
**Extraction LLM:** configurable via Hermes's standard provider/config pattern (cloud or local Ollama)
**Estimated effort:** 7–9 weeks solo, 4–5 weeks with two engineers (added scope: benchmarking, context-fencing, safety hardening)

---

## Settled Decisions

1. **Plugin, not fork.** Confirmed by decision 3 below — coexistence keeps us inside the `MemoryProvider` interface with zero core changes.
2. **Both backends ship in one plugin.** SQLite is the default for single-user local installs; Neo4j is the default for the messaging gateway and multi-user/scale deployments.
3. **Configurable extraction LLM, Hermes-style.** Cloud (OpenAI/Anthropic/Gemini/Groq), local Ollama, or `inherit` from Hermes's active model — selected via config/env, no code changes.
4. **Additive coexistence with built-in memory (Option B).** The graph is the additive external provider; MEMORY.md/USER.md stay always-on. No migration from another provider is assumed (fresh setup). Division of labor is defined below.

---

## Background & Problem Statement

Hermes Agent's current memory system is a two-tier design:

- **Working memory** — `MEMORY.md` / `USER.md`, injected into the system prompt at session start (~2,200 chars). The agent manages entries directly. No time dimension.
- **Session archive** — SQLite (`~/.hermes/state.db`) with FTS5 full-text search. Stores raw conversation messages; retrieved on demand via `session_search`.

This works well for "remember this fact forever" and "find what we discussed last week." It breaks down the same way vector RAG does for long-lived agents:

- Two people named "Marta" become ambiguous blobs in MEMORY.md.
- Outdated facts (old city, old job, old project status) co-exist with new ones; the agent picks whichever scores highest in FTS.
- Multi-hop reasoning ("who introduced me to the running club?") requires chaining facts that are never connected in the current flat structure.
- There is no concept of *when* a fact was true — no validity window, no supersession tracking.

The article "Building Agent Memory with Knowledge Graphs" (The Neural Maze, June 2026) describes exactly the right solution: a **temporal knowledge graph** that stores entities, relationships, and crucially, the time intervals during which each relationship was valid. Graphiti (from Zep) implements this with incremental episode ingestion, a bi-temporal model, and sub-second hybrid retrieval (vector + BM25 + graph traversal). It runs against Neo4j or a local SQLite-based backend.

---

## Decision: Plugin vs Fork

### Plugin (confirmed)

Hermes already has a first-class `MemoryProvider` plugin interface (`agent/memory_provider.py`). It defines five lifecycle hooks:

| Hook | When called | What we do |
|---|---|---|
| `initialize(session_id)` | Agent startup | Connect to Neo4j / Graphiti; create group namespace |
| `prefetch(query)` | Before each reply | Hybrid search the graph; inject relevant facts as context |
| `sync_turn(user, assistant, messages)` | After each reply | Ingest the turn as a Graphiti episode |
| `handle_tool_call(name, args)` | On memory tool call | Serve `temporal_search`, `graph_browse`, `fact_history` |
| `get_tool_schemas()` | At tool registration | Declare the tools above |

The plugin lives at `plugins/memory/graphiti/` and requires **zero changes to core files** — consistent with the AGENTS.md rule ("plugins MUST NOT modify core files"). A user enables it with `hermes plugins enable graphiti` and sets `memory.provider: graphiti` in `config.yaml`.

**Pros:** Ships as an installable package; doesn't diverge from upstream; other Hermes users can adopt it; upgrades to Hermes core are trivially merged.

**Cons:** Constrained to the `MemoryProvider` hook surface. If temporal context needs to reach the system prompt (not just the user-turn injection), we'd need to propose a new hook upstream rather than hardcode it.

### Fork (fallback)

Fork `NousResearch/hermes-agent`, add the Graphiti layer directly into `run_agent.py` and `prompt_builder.py`, and replace the MEMORY.md write-path with graph writes. Allows richer integration (temporal facts in the system prompt, temporal graph as the canonical store replacing MEMORY.md entirely) but creates a maintenance burden on every upstream release.

**Recommended only if:** the plugin hook surface proves insufficient after Phase 2 prototyping, or if this is intended as a permanent fork with a different distribution strategy.

---

## Lessons Learned from Existing Hermes Memory Providers

Hermes already ships eight external memory providers (Honcho, Mem0, Hindsight, Holographic, RetainDB, ByteRover, OpenViking, Supermemory), plus community efforts like agentmemory. Their public docs, source, and the original `MemoryProvider` design discussion (issue #3943) surface concrete, hard-won lessons. We fold these into the plan rather than rediscovering them.

1. **Recursive memory pollution is the classic footgun.** Supermemory strips recalled memories out of the captured turn before ingesting it, "to prevent recursive memory pollution." If we inject facts via `prefetch` and then ingest that same turn verbatim in `sync_turn`, the graph re-ingests its own output — entities and edges duplicate and reinforce themselves over time. **We must implement context fencing: tag injected context and strip it before ingestion.**

2. **Never inject into the system prompt.** The `MemoryProvider` interface was explicitly designed *without* a system-prompt injection method. Honcho's original approach bloated the prompt with stale, frozen-on-first-turn context and duplicated tool docs the model already gets from tool schemas. The agreed pattern is a sanitized, fenced `<memory-context>` block appended at the *user-message* level, refreshed every turn. **We inject only through `prefetch`'s return value; the fence is applied by the MemoryManager, but we treat all stored content as untrusted.**

3. **Adversarial content can escape the injection wrapper.** Because memory is written across sessions, a compromised or manipulated earlier session could store prompt-injection payloads that later surface as "facts." **Graphiti's extracted facts are untrusted input; we sanitize on the way out and never store raw tool output (file paths, command results) as episode bodies without scrubbing.**

4. **`sync_turn` must be non-blocking.** Every mature provider runs ingestion in a daemon thread and joins with a short timeout. This matters doubly for us: Graphiti ingestion costs an LLM extraction call (0.5–2s per episode). **The agent's reply must never wait on graph writes.**

5. **The built-in memory tool out-competes provider tools.** Hindsight's docs warn: disable Hermes's built-in `memory` tool, or the LLM keeps preferring it and the provider's tools go unused. **We need a clear story for how `temporal_search` / `fact_history` coexist with (or supersede) the built-in tool — likely a config flag.**

6. **`is_available()` must be cheap — no network I/O, no subprocess.** It's called during discovery. **Connection checks belong in `initialize`, not `is_available`.**

7. **Hooks are optional and best-effort; the manager normalizes nothing.** Providers expose their own mode switches (`recall_mode: hybrid | context | tools`) and their own budget knobs; the shared manager doesn't unify them. **We adopt the established `recall_mode` convention and own our own token budgeting rather than expecting the framework to do it.**

8. **Per-identity namespacing is expected.** Supermemory scopes containers with `{identity}` (e.g. `hermes-coder`); Honcho uses per-peer scoping. **Our `group_id` must key off the Hermes profile/identity, not just a session ID, so memories isolate per user and per profile.**

9. **Mirror built-in memory writes.** Active providers mirror MEMORY.md/USER.md writes into the external store so nothing is lost. **We do this on `sync_turn` and on initial import.**

10. **Graphiti-specific: throttle concurrency.** Graphiti defaults concurrency low to avoid LLM 429s and exposes `SEMAPHORE_LIMIT`. **We expose this as config and default conservatively, raising it only with a dedicated extraction model/key.**

11. **Benchmark against a real memory eval, not vibes.** agentmemory reports 95.2% R@5 on LongMemEval-S vs ~55% for keyword-only FTS. **We adopt LongMemEval-S as our headline retrieval benchmark so improvement over Hermes's built-in FTS5 is measurable, not asserted.**

---

## Objectives & Key Results

### Objective 1 — Ship a self-hostable temporal-memory plugin with zero core changes

- **KR1.1** Plugin implements all `MemoryProvider` lifecycle hooks (`initialize`, `prefetch`, `sync_turn`, `handle_tool_call`, `get_tool_schemas`, `on_memory_write`, `on_session_end`, `system_prompt_block`, `shutdown`) and loads via `hermes plugins enable graphiti`.
- **KR1.5** After a 5-turn conversation that introduces named entities, MEMORY.md contains a correct, non-empty Temporal Memory Index section; index is stripped from graph ingestion (context-fence test).
- **KR1.2** Zero modifications to any file outside `plugins/memory/graphiti/` (verified by diff against upstream).
- **KR1.3** Both backends functional: Neo4j (Docker) and SQLite (offline, no Docker).
- **KR1.4** `pip install hermes-graphiti` works on macOS and Linux; `hermes setup` offers it via `post_setup`.

### Objective 2 — Beat the built-in FTS5 memory on temporal and relational recall

- **KR2.1** ≥ 85% R@5 on LongMemEval-S (vs ~55% keyword-only baseline), measured on the same machine.
- **KR2.2** 100% correct on a curated suite of ≥ 20 supersession scenarios (city move, job change, status change): `prefetch` never returns a superseded fact as current.
- **KR2.3** ≥ 90% correct entity disambiguation on a suite of ≥ 15 same-first-name scenarios.
- **KR2.4** `fact_history` returns a complete, correctly-ordered timeline for 100% of tested entities.

### Objective 3 — Production-ready performance, safety, and cost transparency

- **KR3.1** `prefetch` p95 latency < 1s on the Neo4j backend with a 10K-episode graph.
- **KR3.2** `sync_turn` adds 0 ms to the user-visible reply latency (fully backgrounded), verified by timing.
- **KR3.3** Context-fencing test: 0 injected-context facts re-ingested across a 50-turn synthetic conversation.
- **KR3.4** Cold-start plugin load < 500 ms.
- **KR3.5** Per-turn extraction cost surfaced to the user (`hermes graphiti status` shows running token/$ cost); no silent spend.

### Objective 4 — Adoption and maintainability

- **KR4.1** Documentation complete: quickstart, bi-temporal explainer, config reference, privacy note on what leaves the device.
- **KR4.2** PR submitted to `NousResearch/hermes-agent` (plugin path) **or** published to PyPI + listed on `awesome-hermes-agent`.
- **KR4.3** Test coverage ≥ 80% on the plugin package; CI green on both backends.

---

## Phased Roadmap

### Phase 0 — Research & Environment (Week 1)

**Goals:** Understand the full Graphiti API; validate that Neo4j and the SQLite backend both work locally; confirm the MemoryProvider hook surface is sufficient.

Tasks:
- Clone `NousResearch/hermes-agent`; read `agent/memory_provider.py`, `agent/memory_manager.py`, and the Honcho plugin (`plugins/memory/honcho/`) as a reference implementation.
- Stand up Neo4j via Docker Compose (the Neural Maze article includes a working Compose file). Run the Graphiti quickstart; ingest a few episodes and run hybrid searches.
- Map each Graphiti operation (`client.add_episode`, `client.search`, `client.get_entity`) to the corresponding MemoryProvider hook.
- Identify any gaps: does `prefetch` fire early enough to inject context before the system prompt is frozen? (If not, flag for a hook proposal to Nous Research.)
- Decide on the local fallback: Graphiti supports a SQLite-backed store (`graphiti-core[sqlite]`) which removes the Neo4j Docker dependency for lightweight installs.

Deliverable: a one-page technical note confirming the approach or recommending the fork path.

---

### Phase 1 — Skeleton Plugin (Week 2)

**Goals:** A working plugin that Hermes loads, connects, and doesn't crash.

File layout:
```
plugins/memory/graphiti/
├── __init__.py        # GraphitiMemoryProvider(MemoryProvider) + register()
├── plugin.yaml        # name, description, hooks, min_hermes_version
├── cli.py             # register_cli(subparser) → hermes graphiti status/clear/export
├── config.py          # Pydantic model for config.yaml keys
└── README.md
```

`__init__.py` skeleton:

```python
import os
from agent.memory_provider import MemoryProvider
from graphiti_core import Graphiti

class GraphitiMemoryProvider(MemoryProvider):
    name = "graphiti"

    def is_available(self) -> bool:
        # No network call; just check config
        return bool(os.environ.get("GRAPHITI_NEO4J_URI") or
                    os.environ.get("GRAPHITI_USE_SQLITE"))

    def initialize(self, session_id: str, **kwargs) -> None:
        cfg = kwargs.get("config", {})
        # Namespace per profile AND per platform user (Lesson 8).
        # Single-user local: group_id = "hermes-<profile>".
        # Multi-user gateway: append the SessionSource user_id so two
        # Telegram users never share a graph. Falls back gracefully.
        identity = kwargs.get("identity") or os.environ.get("HERMES_PROFILE", "default")
        source = kwargs.get("session_source")  # platform, chat_id, user_id
        user_suffix = f"-{source.user_id}" if source and source.user_id else ""
        self._group_id = f"hermes-{identity}{user_suffix}"
        self._client = Graphiti(
            uri=os.environ.get("GRAPHITI_NEO4J_URI", "bolt://localhost:7687"),
            user=os.environ.get("GRAPHITI_NEO4J_USER", "neo4j"),
            password=os.environ.get("GRAPHITI_NEO4J_PASSWORD", "password"),
        )

    def prefetch(self, query: str) -> str | None:
        results = self._client.search(query, group_ids=[self._group_id], limit=10)
        if not results:
            return None
        lines = ["[Temporal Memory — relevant facts]"]
        for r in results:
            validity = f" (as of {r.valid_at})" if r.valid_at else ""
            lines.append(f"- {r.fact}{validity}")
        return "\n".join(lines)

    def sync_turn(self, user: str, assistant: str, messages=None) -> None:
        self._client.add_episode(
            name="conversation_turn",
            episode_body=f"User: {user}\nAssistant: {assistant}",
            group_id=self._group_id,
        )

    def get_tool_schemas(self) -> list[dict]:
        return [TEMPORAL_SEARCH_SCHEMA, FACT_HISTORY_SCHEMA, GRAPH_BROWSE_SCHEMA]

    def handle_tool_call(self, name: str, args: dict) -> str:
        if name == "temporal_search":
            return self._temporal_search(**args)
        if name == "fact_history":
            return self._fact_history(**args)
        if name == "graph_browse":
            return self._graph_browse(**args)
        return f"Unknown tool: {name}"
```

Deliverable: plugin loads via `hermes plugins enable graphiti`; `hermes graphiti status` shows connection state; no other functionality yet.

---

### Phase 2 — Core Memory Loop (Weeks 3–4)

**Goals:** Every conversation turn is ingested as a Graphiti episode; every new turn is prefetched with temporally-aware context.

Key implementation work:

**Episode ingestion (`sync_turn`)**
- **Run in a daemon thread (Lesson 4).** The agent reply must never wait on Graphiti's extraction LLM call. Join the previous sync thread with a short timeout before starting the next.
- **Context-fence before ingesting (Lesson 1).** Strip any `<memory-context>` block we injected this turn out of the episode body, so the graph never re-ingests its own recalled output. This is the single most important correctness step in the loop.
- **Scrub untrusted content (Lesson 3).** Do not store raw tool output (file paths, command results, pasted secrets) as episode bodies without scrubbing; treat everything that goes into the graph as potentially resurfacing in a future prompt.
- Include platform metadata (session source, timestamp) so Graphiti's bi-temporal model has a real `source_description` / `reference_time`.
- Batch short turns to avoid one LLM extraction call per message (Graphiti supports multi-message episode bodies). Expose `SEMAPHORE_LIMIT` as config and default it low to avoid LLM 429s (Lesson 10).
- Handle ingestion failures gracefully — log and continue; never block the agent response.

**Prefetch injection (`prefetch`)**
- Return only the fenced fact block; **never write to the system prompt (Lesson 2)** — the MemoryManager appends our return value at the user-message level and applies the sanitization fence.
- Format retrieved facts with their validity window: `"[2025-11 → 2026-03] Lives in Barcelona"` vs `"[2026-03 → now] Lives in Madrid"`. The agent can distinguish current from historical facts without any additional logic.
- Cap injected context at ~600 tokens to avoid crowding the prompt. Rank by recency × relevance score. Own this budget ourselves (Lesson 7).
- Support a `recall_mode` config knob (`hybrid | context | tools`) matching the established provider convention (Lesson 7), so users can choose automatic injection, tool-only retrieval, or both.
- Run prefetch in the background, non-blocking, before the prompt is composed.

**Tool: `temporal_search`**
- Arguments: `query: str`, `as_of: str | null` (ISO date).
- When `as_of` is supplied, search the graph for facts valid at that point in time.
- Returns structured results with entity, relationship, valid_from, valid_to.

**Tool: `fact_history`**
- Arguments: `entity: str`, `relationship: str | null`.
- Returns the full timeline for an entity or relationship — all versions with validity windows.
- Example: "Show me my job history" → returns every `WORKS_AT` edge the agent has recorded, ordered by valid_from.

**Tool: `graph_browse`**
- Arguments: `entity: str`, `depth: int` (1–3).
- Returns the neighborhood of an entity in the graph — useful for multi-hop questions the agent can answer by inspection.

**MEMORY.md coexistence & division of labor (Option B — recall-trigger index)**

The two memory layers have complementary, non-overlapping jobs:

- **Built-in MEMORY.md / USER.md** owns a tiny set of always-on, must-never-miss facts (the user's name, core preferences, current critical context) **plus a plugin-maintained Temporal Memory Index section** — a brief topic catalog that tells the agent what is retrievable from the deep graph.
- **Temporal graph** owns the rich, relational, time-varying web — people, places, projects, how they connect, and how they've changed. Retrieved per turn via `prefetch` and queried via the temporal tools.

**The recall-trigger index** is a fenced section the plugin owns inside MEMORY.md:

```
<!-- graphiti-index:start -->
## Temporal Memory Index
_Updated 2026-06-09 · query deeper: temporal_search, fact_history, graph_browse_

**People:** Marta Kovač (colleague), Marta Ruiz (running club, via Jaime)
**Places:** Madrid (2026-03 → now) · history available
**Projects:** Helios (active), Atlas (archived 2025-11)
<!-- graphiti-index:end -->
```

The agent sees this in every system prompt and knows which topics have richer history in the graph. `system_prompt_block()` adds a one-time instruction: *"Do not edit the Temporal Memory Index section; use temporal_search, fact_history, or graph_browse for depth."*

Implementation:
- On `initialize`, seed the index from existing graph entities via `get_by_group_ids()`.
- On `sync_turn`, after episode ingestion passes `AddEpisodeResults.nodes/edges` to `MemoryIndexManager.update_from_episode()` inside the existing daemon thread (no extra latency).
- On `on_memory_write` (new hook), mirror the write to the graph (`source: memory_tool`, Lesson 9) **and** call `MemoryIndexManager.on_memory_write()` to add a teaser stub for new entities. If the LLM tries to remove the index entry, re-seed it immediately.
- On `on_session_end` (new hook), call `MemoryIndexManager.full_refresh()` with community summaries from `build_communities()` for richer topic-cluster names.
- **Context fencing (KR3.3):** the index section (`<!-- graphiti-index:start/end -->`) is stripped from episode bodies before ingestion, alongside the `<memory-context>` block, so the graph never re-ingests its own recalled output.
- **Tool competition (Lesson 5):** resolved by collaboration, not suppression. `disable_builtin_memory_tool` remains an optional power-user escape hatch only. The index bridge makes the tools complementary — the LLM uses the built-in tool for atomic facts and the temporal tools for depth queries.

Deliverable: a conversation of 20 turns creates a visible knowledge graph in Neo4j Browser; prefetch returns temporally-correct facts; `temporal_search` and `fact_history` work from the CLI.

---

### Phase 3 — Entity Resolution & Contradiction Handling (Week 5)

**Goals:** The graph correctly handles ambiguous entities and superseded facts.

This is the hardest part, and largely handled by Graphiti's built-in extraction — but it needs configuration and testing:

**Entity disambiguation**
- Configure Graphiti's entity resolution to use the `group_id` namespace so "Marta the partner" and "Marta the colleague" are represented as distinct nodes if context differs.
- Test with a scripted conversation that introduces two people with the same name in different contexts; verify the graph holds two nodes with distinct relationship contexts.

**Fact supersession**
- Test the "I moved to Madrid" scenario: ingest `LIVES_IN Barcelona`, then ingest `LIVES_IN Madrid`. Verify that `prefetch("where do I live")` returns Madrid and that `fact_history("user", "LIVES_IN")` returns both with correct validity windows.
- Test job-change scenario; promotion scenario; relationship status change.

**Contradiction logging**
- When Graphiti marks an edge as superseded, log a brief note to the agent's console (`[graphiti] Updated: LIVES_IN Barcelona → Madrid`). This gives the user visibility into what the agent "learned."

**Edge cases**
- User explicitly corrects a fact ("actually, I never lived in Barcelona — that was my sister"). Implement a `fact_correct` tool that invalidates an edge and creates a replacement with a `source: user_correction` tag.

**Benchmarking (Lesson 11)**
- Stand up LongMemEval-S and run it against (a) Hermes built-in FTS5 and (b) the Graphiti plugin on the same hardware. Record R@5 for both — this is KR2.1 and the headline "is this worth the cost" number.
- Capture per-episode extraction latency and token cost during the run to validate KR3.1 and KR3.5.

Deliverable: entity resolution and supersession tests pass; `fact_history` shows clean timelines for tested scenarios; LongMemEval-S numbers recorded for both backends vs the FTS5 baseline.

---

### Phase 4 — Local / Offline Mode (Week 6)

**Goals:** The plugin works without Docker or a network connection, using Graphiti's SQLite backend.

Tasks:
- Test `graphiti-core[sqlite]` for all operations used in Phases 1–3.
- Add `GRAPHITI_USE_SQLITE=1` env var path in `initialize`; default DB path to `~/.hermes/graphiti.db`.
- Document the performance trade-off (SQLite is slower for large graphs; recommend Neo4j above ~50K episodes).
- Add `hermes graphiti migrate --to neo4j` CLI command that exports SQLite graph to Neo4j for users who outgrow the local store.

Deliverable: full plugin functionality with no external services; Docker only needed for Neo4j mode.

---

### Phase 5 — Packaging, Docs, and Release (Weeks 7–8)

**Goals:** Installable package, documentation, and community submission.

Tasks:
- Add `pyproject.toml` with entry point: `hermes.memory.providers = graphiti = plugins.memory.graphiti:register`.
- Write `README.md` with quickstart (Docker Compose snippet, env vars, one-command enable).
- Write `docs/temporal-memory.md` explaining the bi-temporal model in plain language for Hermes users.
- Add `setup_wizard` integration (`post_setup(hermes_home, config)`) so `hermes setup` offers Graphiti as a memory provider option alongside Honcho and Supermemory.
- Submit as a PR to `NousResearch/hermes-agent` (plugin path, no core changes) or publish as `hermes-graphiti` on PyPI.
- Optional: submit to `awesome-hermes-agent`.

Deliverable: `pip install hermes-graphiti` works; `hermes plugins enable graphiti` activates it; docs are complete.

---

## Architecture Diagram

```
User message
     │
     ▼
MemoryManager.prefetch(query)
     │
     ├──► GraphitiMemoryProvider.prefetch(query)
     │         │
     │         ▼
     │    Graphiti hybrid search
     │    (vector + BM25 + graph traversal)
     │         │
     │         ▼
     │    Temporally-filtered facts
     │    "[2026-03→now] Lives in Madrid"
     │         │
     ▼         ▼
AIAgent builds prompt  ◄──── injected as user-turn context
     │
     ▼
LLM generates reply
     │
     ▼
MemoryManager.sync_turn(user, assistant, messages)
     │
     └──► GraphitiMemoryProvider.sync_turn(...)
               │
               ▼
          Graphiti.add_episode(...)
               │
               ▼
          LLM extraction (async, non-blocking)
          Entity resolution + supersession
               │
               ▼
          Neo4j / SQLite graph updated
```

---

## Configuration

Following Hermes conventions: the setup wizard prompts only for the minimum (backend choice + API key), and everything else lives in a config reference (`$HERMES_HOME/graphiti.json`) plus `config.yaml`. Secrets go in `~/.hermes/.env`.

**Backend (both ship; default chosen by deployment):**
```yaml
# ~/.hermes/config.yaml
memory:
  provider: graphiti
plugins:
  graphiti:
    backend: sqlite          # default for single-user local; no Docker
    # backend: neo4j         # default for gateway / multi-user / scale
    recall_mode: hybrid      # hybrid | context | tools  (Lesson 7)
    disable_builtin_memory_tool: false   # Lesson 5
    semaphore_limit: 5       # Graphiti ingestion concurrency (Lesson 10)
    max_recall_tokens: 600
```
- **SQLite** is the default for single-user local installs — zero external services, DB at `~/.hermes/graphiti.db`.
- **Neo4j** is the default for the messaging gateway and any multi-user or large-graph deployment. `hermes graphiti migrate --to neo4j` promotes a local graph when a user outgrows SQLite.

**Extraction LLM (configurable, Hermes-style):**

Graphiti needs an LLM for entity/relationship extraction and a model for embeddings. Graphiti natively supports OpenAI, Anthropic, Gemini, Groq, and local models via Ollama — so we expose model selection the same way Hermes exposes its own model config (env + config, no code changes to switch):
```yaml
  graphiti:
    extraction:
      provider: ollama       # openai | anthropic | gemini | groq | ollama
      model: llama3.1:8b      # local extraction → nothing leaves the device
      base_url: http://localhost:11434
      # or reuse Hermes's active model:
      # provider: inherit
```
- `provider: inherit` reuses whatever model Hermes is already configured with (one less thing to set up).
- A dedicated **local** model (Ollama) keeps all conversation content on-device — the privacy-preserving default for users who need it.
- A dedicated **cheap cloud** model (e.g. a small fast model) can be set just for extraction, independent of the agent's main model, to control cost.
- The privacy note (KR4.1) documents exactly what is sent off-device under each setting.

---

## Dependencies

| Package | Purpose | License |
|---|---|---|
| `graphiti-core` | Temporal KG engine, episode ingestion, hybrid search | Apache 2.0 |
| `neo4j` (Python driver) | Neo4j connection (gateway/scale backend) | Apache 2.0 |
| `graphiti-core[sqlite]` | Local single-user backend, no Docker needed | Apache 2.0 |

Optional, for local extraction: an Ollama install (no Python dependency — reached over HTTP). No changes to `requirements.txt` in the core repo — all dependencies declared in the plugin's own `pyproject.toml`.

---

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Graphiti ingestion latency (0.5–2s per turn) blocks the UX | Medium | Run `sync_turn` in a daemon thread; agent reply is never delayed by graph writes (Lesson 4) |
| Recursive memory pollution — graph re-ingests its own injected context | High | Context-fence: strip the injected `<memory-context>` block before ingestion (Lesson 1); add the 50-turn regression test in KR3.3 |
| Adversarial/poisoned facts surface in a later prompt | Medium | Treat all stored content as untrusted; sanitize on output, scrub raw tool output before storing (Lesson 3) |
| Built-in `memory` tool out-competes our tools, leaving them unused | Medium | Recall-trigger index in MEMORY.md + `system_prompt_block()` instruction bridges both tools; `disable_builtin_memory_tool` flag retained as power-user escape hatch only (Lesson 5) |
| Graphiti entity extraction misses implicit relationships | Medium | Log missed extractions; expose `fact_correct` tool for user corrections |
| Neo4j Docker dependency is too heavy for typical Hermes users | High | SQLite mode (Phase 4) is the default; Neo4j is opt-in for power users |
| LLM 429 rate-limit errors during high-throughput ingestion | Medium | Expose and conservatively default `SEMAPHORE_LIMIT` (Lesson 10) |
| Sending conversation content to a cloud LLM for extraction is unacceptable for some users | Medium | Support local extraction via Ollama; document clearly what leaves the device (KR4.1) |
| Plugin hook surface insufficient for system-prompt-level context | Low | By design we inject at the user-message level (Lesson 2), which is sufficient; raise an upstream issue only if proven otherwise |
| One-provider-at-a-time limit conflicts with users already on Honcho/Hindsight | Medium | Document clearly; Graphiti subsumes most temporal use cases of those providers |

---

## Definition of Done

Quantitative targets live in the Objectives & Key Results above. The project is "done" when all KRs are met and the following acceptance demo passes end to end:

- A 30-turn conversation builds a knowledge graph visible node-by-node in Neo4j Browser.
- After an in-conversation city move, `prefetch("where do I live")` returns only the current city, and `fact_history("user", "LIVES_IN")` shows both cities with correct validity windows.
- Two people sharing a first name, introduced in different contexts, resolve to separate nodes.
- The same conversation runs identically on the SQLite backend with no Docker.
- `git diff` against upstream shows zero changes outside `plugins/memory/graphiti/`.
