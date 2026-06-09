#!/usr/bin/env bash
# setup_env.sh — install graphiti-core and validate the live environment.
#
# Usage:
#   ./scripts/setup_env.sh [--neo4j | --kuzu]
#
# Defaults to --neo4j. Requires:
#   --neo4j:  Docker running, Neo4j up via `docker compose up -d`
#   --kuzu:   No Docker needed; graphiti-core[kuzu] is installed locally
#             (Kuzu backend is deprecated upstream — dev/offline only)
#
# An extraction LLM key is always required:
#   export OPENAI_API_KEY=sk-...    (or ANTHROPIC_API_KEY / GROQ_API_KEY)

set -euo pipefail

BACKEND="${1:---neo4j}"
PYTHON="${PYTHON:-python3}"
PIP="$PYTHON -m pip"

log() { echo "[setup_env] $*"; }
die() { echo "[setup_env] ERROR: $*" >&2; exit 1; }

# ── 1. Install graphiti-core ──────────────────────────────────────────────

log "Installing graphiti-core 0.29+ ..."
case "$BACKEND" in
  --neo4j)
    $PIP install "graphiti-core>=0.29" -q
    log "Backend: Neo4j (Docker). Start with: docker compose up -d"
    ;;
  --kuzu)
    $PIP install "graphiti-core[kuzu]>=0.29" -q
    log "Backend: Kuzu (embedded, no Docker). WARNING: deprecated upstream."
    export GRAPHITI_USE_KUZU=1
    ;;
  *)
    die "Unknown backend '$BACKEND'. Use --neo4j or --kuzu."
    ;;
esac

# ── 2. Check LLM key ────────────────────────────────────────────────────────

if [[ -z "${OPENAI_API_KEY:-}" && -z "${ANTHROPIC_API_KEY:-}" && -z "${GROQ_API_KEY:-}" ]]; then
  die "No LLM API key found. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GROQ_API_KEY."
fi
log "LLM key found."

# ── 3. Check Neo4j connectivity (Neo4j mode only) ──────────────────────────

if [[ "$BACKEND" == "--neo4j" ]]; then
  NEO4J_URI="${GRAPHITI_NEO4J_URI:-bolt://localhost:7687}"
  log "Checking Neo4j at $NEO4J_URI ..."
  $PYTHON - <<'PYCHECK'
import sys, os
try:
    from neo4j import GraphDatabase
    uri = os.environ.get("GRAPHITI_NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("GRAPHITI_NEO4J_USER", "neo4j")
    password = os.environ.get("GRAPHITI_NEO4J_PASSWORD", "password")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    driver.close()
    print("[setup_env] Neo4j reachable.")
except Exception as e:
    print(f"[setup_env] ERROR: Neo4j not reachable: {e}", file=sys.stderr)
    print("[setup_env] Start it with: docker compose up -d", file=sys.stderr)
    sys.exit(1)
PYCHECK
fi

# ── 4. Smoke-test graphiti-core round-trip ──────────────────────────────────

log "Running graphiti smoke test (ingest + search) ..."
$PYTHON - <<'PYSMOKE'
import asyncio, os, sys, tempfile

async def smoke():
    from graphiti_core import Graphiti
    from datetime import datetime, timezone

    backend = "kuzu" if os.environ.get("GRAPHITI_USE_KUZU") else "neo4j"

    if backend == "kuzu":
        from graphiti_core.driver.kuzu_driver import KuzuDriver
        with tempfile.TemporaryDirectory() as d:
            client = Graphiti(graph_driver=KuzuDriver(db=f"{d}/smoke.kuzu"))
            await _run(client)
    else:
        client = Graphiti(
            uri=os.environ.get("GRAPHITI_NEO4J_URI", "bolt://localhost:7687"),
            user=os.environ.get("GRAPHITI_NEO4J_USER", "neo4j"),
            password=os.environ.get("GRAPHITI_NEO4J_PASSWORD", "password"),
        )
        await _run(client)

async def _run(client):
    from datetime import datetime, timezone
    await client.build_indices_and_constraints()
    result = await client.add_episode(
        name="smoke_test",
        episode_body="Alice moved from Barcelona to Madrid in March 2026.",
        source_description="smoke_test",
        reference_time=datetime(2026, 3, 1, tzinfo=timezone.utc),
        group_id="hermes-smoke",
    )
    nodes = [n.name for n in (result.nodes or [])]
    edges = [e.fact for e in (result.edges or [])]
    print(f"[setup_env] Extracted nodes: {nodes}")
    print(f"[setup_env] Extracted edges: {edges}")

    hits = await client.search(
        query="where does Alice live",
        group_ids=["hermes-smoke"],
        num_results=5,
    )
    print(f"[setup_env] Search returned {len(hits)} result(s).")
    if not hits:
        print("[setup_env] WARNING: search returned no results — extraction may have failed.")
        sys.exit(1)
    print(f"[setup_env] First fact: {hits[0].fact}")
    print("[setup_env] Smoke test PASSED.")

asyncio.run(smoke())
PYSMOKE

log "Environment ready. Run live tests with: make test-live"
