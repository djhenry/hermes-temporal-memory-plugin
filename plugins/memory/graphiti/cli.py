"""CLI commands: hermes graphiti status | clear | export"""

from __future__ import annotations

import os
from pathlib import Path


def register_cli(subparsers) -> None:
    p = subparsers.add_parser("graphiti", help="Manage the Graphiti temporal memory plugin")
    sub = p.add_subparsers(dest="graphiti_cmd")

    sub.add_parser("status", help="Show connection state and running cost estimate")
    sub.add_parser("clear", help="Delete all episodes for the current profile/group")

    exp = sub.add_parser("export", help="Export the knowledge graph to JSON")
    exp.add_argument("--output", "-o", default="graphiti-export.json")

    mig = sub.add_parser("migrate", help="Migrate the local SQLite graph to Neo4j")
    mig.add_argument("--to", choices=["neo4j"], required=True)

    p.set_defaults(func=_dispatch)


def _dispatch(args) -> None:
    cmd = getattr(args, "graphiti_cmd", None) or "status"
    if cmd == "status":
        _cmd_status()
    elif cmd == "clear":
        _cmd_clear(args)
    elif cmd == "export":
        _cmd_export(args)
    elif cmd == "migrate":
        _cmd_migrate(args)
    else:
        print(f"Unknown command: {cmd}")


def _cmd_status() -> None:
    backend = "sqlite" if os.environ.get("GRAPHITI_USE_SQLITE") else "neo4j"
    uri = os.environ.get("GRAPHITI_NEO4J_URI", "bolt://localhost:7687")
    profile = os.environ.get("HERMES_PROFILE", "default")
    group_id = f"hermes-{profile}"

    print(f"Graphiti memory plugin")
    print(f"  backend  : {backend}")
    if backend == "neo4j":
        print(f"  uri      : {uri}")
    else:
        db_path = os.environ.get("GRAPHITI_SQLITE_PATH", str(Path.home() / ".hermes" / "graphiti.db"))
        print(f"  db       : {db_path}")
    print(f"  group_id : {group_id}")
    print()
    print("Run 'hermes graphiti status --verbose' for episode/token cost breakdown.")


def _cmd_clear(args) -> None:
    profile = os.environ.get("HERMES_PROFILE", "default")
    group_id = f"hermes-{profile}"
    confirm = input(f"Delete ALL graph data for group '{group_id}'? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return
    print(f"Clearing group '{group_id}'... (not yet implemented — connect client here)")


def _cmd_export(args) -> None:
    print(f"Exporting graph to {args.output}... (not yet implemented)")


def _cmd_migrate(args) -> None:
    print(f"Migrating SQLite graph to {args.to}... (not yet implemented)")
