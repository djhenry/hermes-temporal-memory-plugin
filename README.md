# hermes-graphiti

> [!WARNING]
> **Work in Progress — Not Functional**
>
> This plugin is under active development and is **not yet usable**. Core functionality is incomplete, APIs may change without notice, and the installation/quickstart instructions below do not work end-to-end. Do not use this in any project expecting working memory integration.
>
> Follow the repo for updates as development progresses.

Temporal knowledge-graph memory provider for [Hermes Agent](https://github.com/NousResearch/hermes-agent), powered by [Graphiti](https://github.com/getzep/graphiti) (Zep).

Stores entities, relationships, and bi-temporal validity windows. Every fact carries the date range during which it was true — so when you move cities or change jobs, the old fact is archived, not overwritten.

**Backends:** Neo4j (Docker, recommended for production) · Kuzu (embedded, no Docker, Python 3.11+) · FalkorDB Lite (embedded, no Docker, Python 3.12+)

---

## What it adds to Hermes

| Without graphiti | With graphiti |
|---|---|
| "Where do I live?" returns the latest entry in MEMORY.md | `prefetch` returns `[2026-03→now] Lives in Madrid` — temporally correct |
| Two "Marta"s collapse into one entry | Two separate graph nodes with distinct relationship contexts |
| Changing a fact overwrites the old one | Old fact archived with `invalid_at`; `fact_history` shows the full timeline |
| Multi-hop: "who introduced me to the running club?" requires hand-built notes | `graph_browse("running club")` traverses the relationship graph |

The plugin also maintains a **Temporal Memory Index** section inside MEMORY.md — a compact topic catalog that tells Hermes what's retrievable from the deep graph without crowding the prompt.

---

## Quickstart (Neo4j backend)

**1. Start Neo4j**

```bash
docker compose up -d
```

The included `docker-compose.yml` starts Neo4j on `bolt://localhost:7687`.

**2. Install**

```bash
pip install hermes-graphiti
# or, from source:
pip install -e ".[neo4j,dev]"
```

**3. Set environment variables**

```bash
# ~/.hermes/.env  (or export in your shell)
GRAPHITI_NEO4J_URI=bolt://localhost:7687
GRAPHITI_NEO4J_USER=neo4j
GRAPHITI_NEO4J_PASSWORD=password

# Extraction LLM — Graphiti needs this to extract entities from conversations
OPENAI_API_KEY=sk-...          # or ANTHROPIC_API_KEY, GROQ_API_KEY
```

**4. Enable the plugin**

```bash
hermes plugins enable graphiti
```

Or set it in `~/.hermes/config.yaml`:

```yaml
memory:
  provider: graphiti
plugins:
  graphiti:
    backend: neo4j
    recall_mode: hybrid
```

**5. Start Hermes — memory works automatically**

```
hermes
> I moved from Barcelona to Madrid last month.
> [graphiti] Superseded: User lived in Barcelona
```

---

## Quickstart (embedded / no Docker)

**Kuzu — Python 3.11+** (deprecated upstream, good for offline dev):

```bash
pip install hermes-graphiti[kuzu]
export GRAPHITI_USE_KUZU=1
hermes plugins enable graphiti
```

**FalkorDB Lite — Python 3.12+** (recommended embedded option):

```bash
pip install hermes-graphiti[falkordblite]
export GRAPHITI_USE_FALKORDB_LITE=1
hermes plugins enable graphiti
```

Both store the graph at `~/.hermes/graphiti.kuzu` / `~/.hermes/graphiti.fdb` with no external services.

---

## Tools exposed to the agent

| Tool | When to use |
|---|---|
| `temporal_search(query, as_of?)` | Retrieve facts valid at a specific date, or the most relevant current facts |
| `fact_history(entity, relationship?)` | Full timeline for an entity — all versions with validity windows |
| `graph_browse(entity, depth?)` | Multi-hop graph traversal — who introduced me to X, what is Y connected to |
| `fact_correct(entity, relationship, correction)` | Invalidate a wrong fact and record the correction |

---

## CLI

```bash
hermes graphiti status            # backend, group_id, db path or Neo4j URI
hermes graphiti status --verbose  # + live connectivity check + entity/fact counts
hermes graphiti export -o graph.json
hermes graphiti clear --yes
hermes graphiti migrate --to neo4j  # promote Kuzu graph to Neo4j
```

---

## Configuration reference

```yaml
plugins:
  graphiti:
    backend: neo4j              # neo4j | kuzu | falkordblite
    recall_mode: hybrid         # hybrid | context | tools
    disable_builtin_memory_tool: false
    semaphore_limit: 5          # Graphiti extraction concurrency
    max_recall_tokens: 600
    enable_memory_index: true   # MEMORY.md recall-trigger index
    max_index_tokens: 200
    extraction:
      provider: inherit         # inherit | openai | anthropic | gemini | groq | ollama
      model: null               # null = use Hermes's active model
      base_url: null            # required for ollama
```

**`recall_mode`**
- `hybrid` (default): automatic context injection via `prefetch` + temporal tools both available
- `context`: inject-only, no tools registered
- `tools`: tools-only, no automatic injection

**`extraction.provider: inherit`** reuses Hermes's active model — nothing extra to configure. A dedicated cheap/local model reduces cost and keeps extraction off the main model's context.

**Privacy:** with `provider: inherit` or any cloud provider, conversation content is sent to that provider's API for entity extraction. Use `provider: ollama` with a local model to keep all content on-device.

---

## Benchmarks

| Backend | R@5 LongMemEval-S | Notes |
|---|---|---|
| Hermes built-in FTS5 | ~55% | keyword-only baseline |
| hermes-graphiti (Neo4j) | target ≥ 85% | hybrid vector + BM25 + graph |

*(Live results recorded in Phase 3 benchmarking — see docs/temporal-memory.md)*

---

## Development

```bash
make install        # clones hermes-agent + installs plugin + dev deps
make test           # unit + integration (mocked backend, no LLM needed)
make test-live      # end-to-end against real Neo4j + extraction LLM
make lint
```

See [docs/temporal-memory.md](docs/temporal-memory.md) for the bi-temporal model explainer and [docs/project-plan.md](docs/project-plan.md) for the full implementation plan.

---

## License

MIT
