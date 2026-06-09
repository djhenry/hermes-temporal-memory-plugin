# Temporal Memory: How It Works

This document explains the bi-temporal model behind `hermes-graphiti` in plain language — no graph theory required.

---

## The problem with flat memory

Hermes's built-in MEMORY.md stores facts as simple bullet points:

```
- Lives in Barcelona
- Works at Acme Corp as Engineer
```

When something changes — you move to Madrid, get promoted — the old entry is replaced. The agent sees only the current state. There's no record of what was true before, no way to ask "wait, when did that change?", and no way to distinguish two people named "Marta."

This is fine for simple personal assistants. It breaks down for a long-lived agent that's been with you for months or years.

---

## What a temporal knowledge graph adds

Instead of bullet points, the graph stores **facts as edges between entities**:

```
[User] --LIVES_IN--> [Barcelona]   valid: 2024-01-01 → 2026-03-01
[User] --LIVES_IN--> [Madrid]      valid: 2026-03-01 → (now)
```

Every edge has a **validity window**: `valid_at` (when the fact became true) and `invalid_at` (when it stopped being true, or `null` if it's still current).

When you tell Hermes "I moved to Madrid," Graphiti:
1. Finds the existing `LIVES_IN Barcelona` edge
2. Sets its `invalid_at = today`
3. Creates a new `LIVES_IN Madrid` edge with `valid_at = today`

Nothing is deleted. The old fact is archived with a closed window. You can always ask for the full history.

---

## Bi-temporal vs. uni-temporal

You may hear the term "bi-temporal." It just means the graph tracks two time dimensions:

| Dimension | What it means |
|---|---|
| **valid time** (`valid_at` / `invalid_at`) | When the fact was true in the world |
| **transaction time** (`created_at`) | When the agent learned the fact |

For most uses, valid time is what matters: "where did I live in 2025?" But transaction time lets you audit what the agent knew at any given session, which is useful for debugging memory errors.

---

## How retrieval works

When you send a message, `prefetch` runs a **hybrid search** combining three signals:

1. **Vector similarity** — semantic embedding distance between your query and stored facts
2. **BM25 keyword match** — traditional full-text relevance
3. **Graph traversal** — edges connected to recently-mentioned entities get a recency boost

Results are filtered to facts with an open validity window (current state) unless you ask for history.

What gets injected into your prompt:

```
<memory-context>
- [2026-03 → now] Lives in Madrid
- [2026-03 → now] Works at Acme Corp as Staff Engineer
- [2025-06] Met Marta Kovač at team offsite
</memory-context>
```

The fence tags (`<memory-context>`) are stripped before the turn is ingested back into the graph, preventing recursive self-pollution.

---

## The Temporal Memory Index in MEMORY.md

MEMORY.md still exists and still contains your always-on personal context. The plugin adds a short, plugin-maintained section at the bottom:

```
<!-- graphiti-index:start -->
## Temporal Memory Index
_Updated 2026-06-09 · query deeper: temporal_search, fact_history, graph_browse_

**People:** Marta Kovač (colleague, data science), Marta Ruiz (running club, via Jaime)
**Work:** Staff Engineer at Acme Corp (since 2026-03) · history available
**Places:** Madrid (2026-03 → now) · Barcelona history available
**Projects:** Helios (active), Atlas (archived 2025-11)
<!-- graphiti-index:end -->
```

This section is:
- **Compact** (target ≤ 200 tokens) — it's a discovery surface, not a full dump
- **Auto-maintained** — the plugin updates it after each turn; you never edit it
- **Stripped before graph ingestion** — so the index never echoes back into the graph

The agent sees this in every system prompt and knows what's retrievable. It uses the temporal tools for depth.

---

## The temporal tools

### `temporal_search(query, as_of?)`

Standard deep-memory lookup. `as_of` lets you query historical state:

```
temporal_search("where did I live", as_of="2025-06-01")
→ [2024-01 → 2026-03] User lived in Barcelona
```

### `fact_history(entity, relationship?)`

Full timeline for an entity:

```
fact_history("user", "LIVES_IN")
→ [2024-01-01 → 2026-03-01] User lived in Barcelona
→ [2026-03-01 → now]        User lives in Madrid
```

### `graph_browse(entity, depth?)`

Neighborhood traversal:

```
graph_browse("running club")
→ Marta Ruiz (introduced via Jaime, 2025-09)
→ Sunday runs, Retiro Park
→ Jaime García (colleague, introduced me to club)
```

### `fact_correct(entity, relationship, correction)`

When a recorded fact was never true:

```
fact_correct(
  entity="user",
  relationship="LIVES_IN",
  correction="The Barcelona entry was wrong — that was my sister, not me."
)
```

Creates a correction episode tagged `source: user_correction` so the original extraction error is visible in the audit log.

---

## Entity disambiguation

Two people named "Marta" are stored as separate nodes as long as Graphiti's extraction sees distinguishing context:

```
[Marta Kovač] --WORKS_AT--> [Acme Corp data science team]
[Marta Ruiz]  --MEMBER_OF-> [Sunday running club]
```

`group_id` namespacing (per Hermes profile) ensures no cross-user contamination. If disambiguation fails and two facts get merged onto one node, `fact_correct` can split them.

---

## Supersession logging

When a fact is superseded, the plugin logs it at INFO level:

```
[graphiti] Superseded: User lived in Barcelona
```

This appears in Hermes's console output so you can see what the agent "learned" from the turn. It's not written back into the conversation.

---

## Performance expectations

| Operation | Typical latency | Notes |
|---|---|---|
| `prefetch` (Neo4j, 10K episodes) | < 1s p95 | Hybrid search is fast; most time is vector ANN |
| `sync_turn` ingestion | 0.5–2s | Runs in daemon thread; never delays your reply |
| `fact_history` | < 200ms | Simple edge lookup by entity name |
| Plugin cold-start | < 500ms | Connection + `build_indices_and_constraints()` |

Extraction cost: each turn costs one LLM call in the extraction model. With `provider: inherit` this adds to your main model's bill. With a dedicated cheap model (e.g. Haiku or a local Ollama model), extraction runs independently. `hermes graphiti status --verbose` shows running token/$ totals.

---

## Privacy

| Setting | What leaves your device |
|---|---|
| `extraction.provider: openai` | Conversation turns sent to OpenAI for entity extraction |
| `extraction.provider: anthropic` | Conversation turns sent to Anthropic |
| `extraction.provider: ollama` | Nothing — all extraction runs locally |
| `extraction.provider: inherit` | Same as your Hermes model provider |
| Graph itself (Neo4j, Docker) | Stays on your machine |
| Graph itself (Kuzu / FalkorDB Lite) | Stays on your machine |

No conversation content is sent to Zep/Graphiti's servers. Graphiti is an open-source library; the graph is yours.
