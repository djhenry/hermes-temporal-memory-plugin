"""CLI commands: hermes temporal-memory status | clear | export | migrate"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path


def register_cli(subparser) -> None:
    sub = subparser.add_subparsers(dest="temporal_memory_cmd")

    status = sub.add_parser("status", help="Show connection state and running cost estimate")
    status.add_argument("--verbose", "-v", action="store_true")

    clear = sub.add_parser("clear", help="Delete all episodes for the current profile/group")
    clear.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")

    exp = sub.add_parser("export", help="Export the knowledge graph to JSON")
    exp.add_argument("--output", "-o", default="temporal-memory-export.json")

    mig = sub.add_parser("migrate", help="Migrate local Kuzu graph to Neo4j")
    mig.add_argument("--to", choices=["neo4j"], required=True)

    subparser.set_defaults(func=_dispatch)


def _dispatch(args) -> None:
    cmd = getattr(args, "temporal_memory_cmd", None) or "status"
    if cmd == "status":
        _cmd_status(args)
    elif cmd == "clear":
        _cmd_clear(args)
    elif cmd == "export":
        _cmd_export(args)
    elif cmd == "migrate":
        _cmd_migrate(args)
    else:
        print(f"Unknown command: {cmd}")


# ── status ────────────────────────────────────────────────────────────────────

def _cmd_status(args) -> None:
    profile = os.environ.get("HERMES_PROFILE", "default")
    group_id = f"hermes-{profile}"
    use_kuzu = bool(os.environ.get("GRAPHITI_USE_KUZU"))
    use_falkordblite = bool(os.environ.get("GRAPHITI_USE_FALKORDB_LITE"))
    if use_kuzu:
        backend = "kuzu"
    elif use_falkordblite:
        backend = "falkordblite"
    else:
        backend = os.environ.get("GRAPHITI_BACKEND", "neo4j")

    print("Temporal memory plugin")
    print(f"  backend  : {backend}")
    print(f"  group_id : {group_id}")

    if use_kuzu or backend == "kuzu":
        db_path = os.environ.get(
            "GRAPHITI_KUZU_PATH", str(Path.home() / ".hermes" / "graphiti.kuzu")
        )
        print(f"  db       : {db_path}")
        print(f"  db found : {'yes' if Path(db_path).exists() else 'no (not yet initialised)'}")
    elif use_falkordblite or backend == "falkordblite":
        db_path = os.environ.get(
            "GRAPHITI_FALKORDBLITE_PATH", str(Path.home() / ".hermes" / "graphiti.fdb")
        )
        print(f"  db       : {db_path}")
        print(f"  db found : {'yes' if Path(db_path).exists() else 'no (not yet initialised)'}")
    else:
        uri = os.environ.get("GRAPHITI_NEO4J_URI", "bolt://localhost:7687")
        print(f"  uri      : {uri}")
        if getattr(args, "verbose", False):
            _check_neo4j_connectivity(uri)

    if getattr(args, "verbose", False):
        _print_entity_counts(group_id, backend, use_kuzu)


def _check_neo4j_connectivity(uri: str) -> None:
    try:
        from neo4j import GraphDatabase  # type: ignore[import]

        user = os.environ.get("GRAPHITI_NEO4J_USER", "neo4j")
        password = os.environ.get("GRAPHITI_NEO4J_PASSWORD", "password")
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        driver.close()
        print("  neo4j    : reachable")
    except ImportError:
        print("  neo4j    : neo4j driver not installed (pip install neo4j)")
    except Exception as exc:
        print(f"  neo4j    : unreachable ({exc})")


def _print_entity_counts(group_id: str, backend: str, use_kuzu: bool) -> None:
    try:
        client = _make_client(backend, use_kuzu)

        async def _fetch():
            nodes = await client.nodes.entity.get_by_group_ids(group_ids=[group_id], limit=1000)
            edges = await client.edges.entity.get_by_group_ids(group_ids=[group_id], limit=1000)
            return nodes, edges

        nodes, edges = asyncio.run(_fetch())
        print(f"  entities : {len(nodes or [])}")
        print(f"  facts    : {len(edges or [])}")
    except Exception as exc:
        print(f"  counts   : unavailable ({exc})")


# ── clear ─────────────────────────────────────────────────────────────────────

def _cmd_clear(args) -> None:
    profile = os.environ.get("HERMES_PROFILE", "default")
    group_id = f"hermes-{profile}"

    if not getattr(args, "yes", False):
        confirm = input(f"Delete ALL graph data for group '{group_id}'? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return

    use_kuzu = bool(os.environ.get("GRAPHITI_USE_KUZU"))
    use_falkordblite = bool(os.environ.get("GRAPHITI_USE_FALKORDB_LITE"))
    if use_kuzu:
        backend = "kuzu"
    elif use_falkordblite:
        backend = "falkordblite"
    else:
        backend = os.environ.get("GRAPHITI_BACKEND", "neo4j")

    try:
        client = _make_client(backend, use_kuzu)
        asyncio.run(_delete_group(client, group_id))
        print(f"Cleared group '{group_id}'.")
    except Exception as exc:
        print(f"Error clearing graph: {exc}")


async def _delete_group(client, group_id: str) -> None:
    # execute_query is part of graphiti-core's GraphDriver interface.
    # Neo4j, Kuzu, and FalkorDB Lite all implement it with Cypher syntax.
    # FalkorDB Lite uses Redis-protocol Cypher — same query syntax applies.
    await client.driver.execute_query(
        "MATCH (n {group_id: $group_id}) DETACH DELETE n",
        group_id=group_id,
    )


# ── export ────────────────────────────────────────────────────────────────────

def _cmd_export(args) -> None:
    profile = os.environ.get("HERMES_PROFILE", "default")
    group_id = f"hermes-{profile}"
    output = args.output
    use_kuzu = bool(os.environ.get("GRAPHITI_USE_KUZU"))
    use_falkordblite = bool(os.environ.get("GRAPHITI_USE_FALKORDB_LITE"))
    if use_kuzu:
        backend = "kuzu"
    elif use_falkordblite:
        backend = "falkordblite"
    else:
        backend = os.environ.get("GRAPHITI_BACKEND", "neo4j")

    try:
        client = _make_client(backend, use_kuzu)
        data = asyncio.run(_export_group(client, group_id))
        with open(output, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        node_count = len(data.get("nodes", []))
        edge_count = len(data.get("edges", []))
        print(f"Exported {node_count} entities and {edge_count} facts → {output}")
    except Exception as exc:
        print(f"Error exporting graph: {exc}")


async def _export_group(client, group_id: str) -> dict:
    nodes = await client.nodes.entity.get_by_group_ids(group_ids=[group_id], limit=5000)
    edges = await client.edges.entity.get_by_group_ids(group_ids=[group_id], limit=10000)

    def _node_dict(n) -> dict:
        return {
            "uuid": getattr(n, "uuid", None),
            "name": getattr(n, "name", None),
            "labels": getattr(n, "labels", []),
            "summary": getattr(n, "summary", None),
            "created_at": getattr(n, "created_at", None),
        }

    def _edge_dict(e) -> dict:
        return {
            "uuid": getattr(e, "uuid", None),
            "name": getattr(e, "name", None),
            "fact": getattr(e, "fact", None),
            "source_node_uuid": getattr(e, "source_node_uuid", None),
            "target_node_uuid": getattr(e, "target_node_uuid", None),
            "valid_at": getattr(e, "valid_at", None),
            "invalid_at": getattr(e, "invalid_at", None),
        }

    return {
        "group_id": group_id,
        "nodes": [_node_dict(n) for n in (nodes or [])],
        "edges": [_edge_dict(e) for e in (edges or [])],
    }


# ── migrate ───────────────────────────────────────────────────────────────────

def _cmd_migrate(args) -> None:
    if args.to != "neo4j":
        print(f"Unsupported migration target: {args.to}")
        return

    kuzu_path = os.environ.get(
        "GRAPHITI_KUZU_PATH", str(Path.home() / ".hermes" / "graphiti.kuzu")
    )
    if not Path(kuzu_path).exists():
        print(f"No Kuzu graph found at {kuzu_path}. Nothing to migrate.")
        return

    neo4j_uri = os.environ.get("GRAPHITI_NEO4J_URI", "bolt://localhost:7687")
    profile = os.environ.get("HERMES_PROFILE", "default")
    group_id = f"hermes-{profile}"

    print(f"Migrating Kuzu graph at {kuzu_path} → Neo4j {neo4j_uri}")
    print(f"  group_id : {group_id}")

    confirm = input("Proceed? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    try:
        asyncio.run(_migrate_kuzu_to_neo4j(kuzu_path, neo4j_uri, group_id))
        print("Migration complete.")
    except Exception as exc:
        print(f"Migration failed: {exc}")


async def _migrate_kuzu_to_neo4j(kuzu_path: str, neo4j_uri: str, group_id: str) -> None:
    from graphiti_core import Graphiti  # type: ignore[import]
    from graphiti_core.driver.kuzu_driver import KuzuDriver  # type: ignore[import]

    # Read from Kuzu
    src = Graphiti(graph_driver=KuzuDriver(db=kuzu_path))
    nodes = await src.nodes.entity.get_by_group_ids(group_ids=[group_id], limit=5000)
    edges = await src.edges.entity.get_by_group_ids(group_ids=[group_id], limit=10000)

    print(f"  read {len(nodes or [])} entities, {len(edges or [])} facts from Kuzu")

    # Write to Neo4j
    neo4j_user = os.environ.get("GRAPHITI_NEO4J_USER", "neo4j")
    neo4j_password = os.environ.get("GRAPHITI_NEO4J_PASSWORD", "password")
    dst = Graphiti(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
    await dst.build_indices_and_constraints()

    # Re-ingest as synthetic episodes (preserves entity extraction and relationships)
    from datetime import datetime, timezone

    for edge in (edges or []):
        fact = getattr(edge, "fact", None)
        if not fact:
            continue
        await dst.add_episode(
            name="kuzu_migration",
            episode_body=fact,
            source_description="kuzu_migration",
            reference_time=getattr(edge, "valid_at", None) or datetime.now(timezone.utc),
            group_id=group_id,
        )

    print(f"  wrote {len(edges or [])} facts to Neo4j")


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_client(backend: str, use_kuzu: bool):
    """Create a Graphiti client from environment config (no LLM needed for read ops)."""
    from graphiti_core import Graphiti  # type: ignore[import]

    if use_kuzu or backend == "kuzu":
        from graphiti_core.driver.kuzu_driver import KuzuDriver  # type: ignore[import]

        db_path = os.environ.get(
            "GRAPHITI_KUZU_PATH", str(Path.home() / ".hermes" / "graphiti.kuzu")
        )
        return Graphiti(graph_driver=KuzuDriver(db=db_path))

    use_falkordblite = bool(os.environ.get("GRAPHITI_USE_FALKORDB_LITE") or backend == "falkordblite")
    if use_falkordblite:
        from graphiti_core.driver.falkordb_driver import FalkorDriver  # type: ignore[import]
        try:
            from redislite.async_falkordb_client import AsyncFalkorDB  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "FalkorDB Lite requires Python 3.12+ and graphiti-core[falkordblite]. "
                "Run: pip install 'graphiti-core[falkordblite]'"
            ) from exc
        db_path = os.environ.get(
            "GRAPHITI_FALKORDBLITE_PATH",
            str(Path.home() / ".hermes" / "graphiti.fdb"),
        )
        falkor_client = AsyncFalkorDB(dbfilename=db_path)
        return Graphiti(graph_driver=FalkorDriver(falkor_db=falkor_client))

    return Graphiti(
        uri=os.environ.get("GRAPHITI_NEO4J_URI", "bolt://localhost:7687"),
        user=os.environ.get("GRAPHITI_NEO4J_USER", "neo4j"),
        password=os.environ.get("GRAPHITI_NEO4J_PASSWORD", "password"),
    )
