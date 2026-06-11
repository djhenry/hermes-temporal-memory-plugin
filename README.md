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

## Installation

The plugin requires **two** install steps: the Python package and registering it with Hermes's plugin directory.

### Step 1 — Python package

> **⚠️ FalkorDB Lite requires Python 3.12+ in the gateway venv.**
> The Hermes gateway runs in its own virtualenv (`~/.hermes/hermes-agent/venv`).
> If your system Python is 3.12+ but the gateway venv uses 3.11, FalkorDB Lite
> will fail to initialize. To check and fix:
> ```bash
> ~/.hermes/hermes-agent/venv/bin/python --version
> # If < 3.12, recreate the venv:
> cd ~/.hermes/hermes-agent
> hermes gateway stop
> uv venv --python python3.12
> source .venv/bin/activate
> uv pip install "graphiti-core[falkordblite]"
> hermes gateway start
> ```

Install the `graphiti-core` package with the appropriate backend extra into the
**gateway venv** (`~/.hermes/hermes-agent/venv`):

```bash
# FalkorDB Lite (embedded, no Docker, Python 3.12+):
uv pip install --python ~/.hermes/hermes-agent/venv/bin/python "graphiti-core[falkordblite]"

# Kuzu (embedded, no Docker, Python 3.11+):
# uv pip install --python ~/.hermes/hermes-agent/venv/bin/python "graphiti-core[kuzu]"

# Neo4j client only (you still need a Neo4j instance via Docker):
# uv pip install --python ~/.hermes/hermes-agent/venv/bin/python "graphiti-core[neo4j]"
```

> **Note:** The plugin itself is registered via symlink (Step 2). The Python
> package only needs to be installed into the gateway venv so the plugin can
> import `graphiti_core` at runtime.

### Step 2 — Register with Hermes

Hermes discovers plugins from `~/.hermes/plugins/`, not from Python entry points alone. Symlink the plugin module:

```bash
ln -s /path/to/hermes-temporal-memory-plugin/plugins/memory/graphiti \
      ~/.hermes/plugins/graphiti
```

The directory **must** contain `plugin.yaml` and `__init__.py` at its root — the symlink target is the `plugins/memory/graphiti/` subdirectory, not the repo root.

Verify it worked:

```bash
hermes memory status
# Should show:  Plugin: installed ✓  /  Status: available ✓
```

### Step 3 — Configure

Set `memory.provider: graphiti` in `~/.hermes/config.yaml`:

```yaml
memory:
  provider: graphiti

plugins:
  graphiti:
    backend: falkordblite        # or neo4j, kuzu
    recall_mode: hybrid          # hybrid | context | tools
    enable_memory_index: true
    extraction:
      provider: openai           # graphiti uses OpenAI-compatible API
      model: openrouter/auto     # cheap model for entity extraction
      base_url: https://openrouter.ai/api/v1
    embedder:
      model: text-embedding-nomic-embed-text-v1.5  # local embedding model
      base_url: http://localhost:1234/v1            # LM Studio endpoint
      api_key: lm-studio
```

**⚠️ Embedder is required when using OpenRouter.** OpenRouter doesn't expose an `/embeddings` endpoint, so Graphiti's default embedder (which targets OpenAI) fails with 401. You need a separate embedding endpoint — LM Studio, Ollama, or OpenAI all work. See the [embedder config](#embedder-configuration) section below.

Add the backend-specific env vars to `~/.hermes/.env`:

```bash
# ── FalkorDB Lite (embedded, recommended for local use) ──
# Only needed if you don't set backend: falkordblite in config.yaml:
# GRAPHITI_USE_FALKORDB_LITE=1

# ── Extraction LLM — entity extraction needs an OpenAI-compatible endpoint ──
# If using OpenRouter:
OPENAI_API_KEY=sk-or-v1-...     # same key as OPENROUTER_API_KEY

# If using OpenAI directly:
# OPENAI_API_KEY=sk-...

# If using a local Ollama model (all content stays on-device):
# GRAPHITI_EXTRACTION_PROVIDER=ollama
# GRAPHITI_EXTRACTION_MODEL=llama3.1:8b
# GRAPHITI_EXTRACTION_BASE_URL=http://localhost:11434
```

### Step 4 — Restart Hermes

```bash
hermes gateway restart   # or just start a new session
```

The graph is stored at `~/.hermes/graphiti.fdb` (FalkorDB Lite) or `~/.hermes/graphiti.kuzu` (Kuzu).

---

## Quick reference — backend choices

| Backend | Python | Docker | Command |
|---------|--------|--------|---------|
| **FalkorDB Lite** (recommended) | 3.12+ | No | `uv pip install --python ~/.hermes/hermes-agent/venv/bin/python "graphiti-core[falkordblite]"` |
| **Kuzu** (deprecated) | 3.11+ | No | `uv pip install --python ~/.hermes/hermes-agent/venv/bin/python "graphiti-core[kuzu]"` |
| **Neo4j** | any | Yes | `docker compose up -d` + set `GRAPHITI_NEO4J_URI/USER/PASSWORD` |

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
hermes memory status              # verify plugin is installed and available
hermes graphiti status            # backend, group_id, db path or Neo4j URI
hermes graphiti status --verbose  # + live connectivity check + entity/fact counts
hermes graphiti export -o graph.json
hermes graphiti clear --yes
hermes graphiti migrate --to neo4j  # promote Kuzu graph to Neo4j
```

> **Note:** `hermes graphiti` subcommands require a hermes version with plugin CLI integration. If unavailable, use the Python API directly or check `hermes memory status` for basic diagnostics.

---

## Configuration reference

```yaml
memory:
  provider: graphiti            # activates the graphiti memory provider

plugins:
  graphiti:
    backend: falkordblite       # neo4j | kuzu | falkordblite
    recall_mode: hybrid         # hybrid | context | tools
    disable_builtin_memory_tool: false
    semaphore_limit: 5          # Graphiti extraction concurrency
    max_recall_tokens: 600
    enable_memory_index: true   # MEMORY.md recall-trigger index
    max_index_tokens: 200
    extraction:
      provider: openai          # inherit | openai | ollama
      model: openrouter/auto    # model for entity extraction (OpenAI-compatible)
      base_url: https://openrouter.ai/api/v1  # endpoint URL
    embedder:
      model: text-embedding-nomic-embed-text-v1.5  # embedding model
      base_url: http://localhost:1234/v1            # OpenAI-compatible endpoint
      api_key: lm-studio                            # API key (or any string for local)
```

### Key settings

**`recall_mode`**
- `hybrid` (default): automatic context injection via `prefetch` + temporal tools both available
- `context`: inject-only, no tools registered
- `tools`: tools-only, no automatic injection

**`extraction`** — Graphiti needs an LLM to extract entities from conversations. The extraction client is OpenAI-compatible, so any endpoint that implements the `/chat/completions` API works (OpenAI, OpenRouter, Ollama, LM Studio, etc.).

| Provider | `base_url` | `model` | `OPENAI_API_KEY` |
|----------|-----------|---------|-------------------|
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` | `sk-...` |
| OpenRouter | `https://openrouter.ai/api/v1` | `openrouter/auto` | your OpenRouter key |
| Ollama (local) | `http://localhost:11434` | `llama3.1:8b` | any string |
| LM Studio | `http://localhost:1234/v1` | `ibm/granite-4-h-tiny` | `lm-studio` |

**Privacy:** entity extraction sends conversation content to the extraction LLM provider. Use `provider: ollama` with a local model to keep all content on-device.

### Embedder configuration

Graphiti uses an embedding model for vector similarity search. The embedder is OpenAI-compatible — any endpoint implementing `/embeddings` works.

**⚠️ Required when using OpenRouter** — OpenRouter has no `/embeddings` endpoint. Without a configured embedder, `temporal_search` fails with 401.

| Provider | `base_url` | `model` | `api_key` |
|----------|-----------|---------|-----------|
| LM Studio (local) | `http://localhost:1234/v1` | `text-embedding-nomic-embed-text-v1.5` | `lm-studio` |
| Ollama (local) | `http://localhost:11434` | `nomic-embed-text` | any string |
| OpenAI | `https://api.openai.com/v1` | `text-embedding-3-small` | your OpenAI key |

If no embedder config is provided, Graphiti falls back to its default (reads `OPENAI_API_KEY` from env and targets OpenAI's endpoint).

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

## Troubleshooting

### "Plugin: NOT installed ✗" after `hermes memory status`

The plugin module isn't in Hermes's discovery path. Ensure the symlink exists and points to the right directory:

```bash
ls -la ~/.hermes/plugins/graphiti/plugin.yaml   # must exist
# If missing, recreate:
ln -s /path/to/hermes-temporal-memory-plugin/plugins/memory/graphiti \
      ~/.hermes/plugins/graphiti
```

The symlink target **must** be `plugins/memory/graphiti/`, not the repo root. Hermes looks for `plugin.yaml` and `__init__.py` at the root of `~hermes/plugins/<name>/`.

### Config not applied (backend defaults to neo4j, extraction model missing)

Hermes doesn't pass `plugins.<name>` config to the provider's `initialize()` method. The plugin reads its config from `~/.hermes/config.yaml` directly during initialization. Ensure your config has:

```yaml
plugins:
  graphiti:
    backend: falkordblite
    extraction:
      model: openrouter/auto
      base_url: https://openrouter.ai/api/v1
```

### `is_available()` returns False

The backend isn't configured. Check:

```bash
hermes memory status   # should show "Status: available ✓"
```

If not, ensure you have **either** the env var **or** the config key set:
- FalkorDB Lite: `GRAPHITI_USE_FALKORDB_LITE=1` in `~/.hermes/.env` **or** `backend: falkordblite` under `plugins.graphiti` in `~/.hermes/config.yaml`
- Kuzu: `GRAPHITI_USE_KUZU=1` in `~/.hermes/.env` **or** `backend: kuzu` in config
- Neo4j: `GRAPHITI_NEO4J_URI=bolt://localhost:7687` in `~/.hermes/.env` **or** `backend: neo4j` in config

### Extraction errors ("connection refused", "timeout", "invalid API key")

The extraction LLM endpoint isn't reachable or `OPENAI_API_KEY` isn't loaded. Verify:
1. `OPENAI_API_KEY` is set in `~/.hermes/.env` (the plugin reads this via hermes's `load_env()`)
2. The `base_url` in config matches your provider
3. For Ollama: `ollama serve` is running

**Note:** Hermes reads `~/.hermes/.env` via `load_env()` but doesn't always propagate values to `os.environ`. The plugin handles this by reading `.env` directly during initialization — no manual `export` needed.

### `temporal_search` returns 401 / "Incorrect API key"

The embedder (not the extraction LLM) is failing. This happens when using OpenRouter, which has no `/embeddings` endpoint. Fix: configure the `embedder` section in config.yaml to use a local or separate embedding provider (LM Studio, Ollama, or OpenAI).

### "FalkorDB Lite requires Python 3.12+"

The gateway venv is running Python < 3.12. FalkorDB Lite (and its `redislite`
dependency) require Python 3.12+. The system Python may be 3.12+ while the
gateway venv is still on 3.11 — the venv has its own Python.

Fix: recreate the gateway venv with Python 3.12+:

```bash
~/.hermes/hermes-agent/venv/bin/python --version   # check current version
cd ~/.hermes/hermes-agent
hermes gateway stop
uv venv --python python3.12
source .venv/bin/activate
uv pip install "graphiti-core[falkordblite]"
hermes gateway start
```

### `sync_turn() got an unexpected keyword argument 'session_id'`

The plugin's `sync_turn` method didn't accept the `session_id` keyword argument
that the Hermes agent framework passes. This was fixed in the plugin source
(`plugins/memory/graphiti/__init__.py`, line 328) — ensure your checkout
includes this fix:

```python
def sync_turn(self, user_content: str, assistant_content: str, messages: Any = None, session_id: str = "") -> None:
```

After updating the plugin source, restart the gateway: `hermes gateway restart`.

### Graph not persisting between sessions

Check the DB path:
```bash
ls -la ~/.hermes/graphiti.fdb    # FalkorDB Lite
ls -la ~/.hermes/graphiti.kuzu   # Kuzu
```

If the file doesn't exist, the first `add_episode` call creates it automatically.

---

## License

MIT
