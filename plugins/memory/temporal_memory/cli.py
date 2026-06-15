"""CLI commands: hermes temporal-memory status | clear | export | migrate | mood"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

# Inline mood model (avoid relative import — cli.py is loaded via
# spec_from_file_location which doesn't set up the parent package).
EMOTION_KEYS = [
    "joy", "trust", "fear", "surprise",
    "sadness", "disgust", "anger", "anticipation",
]

_DYADS = [
    (("joy", "trust"), "love"),
    (("trust", "fear"), "submission"),
    (("fear", "surprise"), "awe"),
    (("surprise", "sadness"), "disapproval"),
    (("sadness", "disgust"), "remorse"),
    (("disgust", "anger"), "contempt"),
    (("anger", "anticipation"), "aggressiveness"),
    (("anticipation", "joy"), "optimism"),
]

_OUTER_DYADS = [
    (("joy", "fear"), "guilt"),
    (("trust", "sadness"), "sentimentality"),
    (("fear", "disgust"), "shame"),
    (("surprise", "anger"), "outrage"),
    (("sadness", "anger"), "envy"),
    (("disgust", "anticipation"), "cynicism"),
    (("anger", "joy"), "pride"),
    (("anticipation", "trust"), "hope"),
]


def _default_emotions():
    return {
        "joy": 0.4, "trust": 0.6, "fear": 0.1, "surprise": 0.3,
        "sadness": 0.1, "disgust": 0.05, "anger": 0.05, "anticipation": 0.5,
    }


def _resolve_mood_label(emotions: dict, threshold: float = 0.55) -> str:
    max_emotion = max(emotions, key=lambda k: emotions.get(k, 0.0))
    if emotions.get(max_emotion, 0.0) >= threshold:
        return max_emotion
    for (e1, e2), label in _DYADS:
        if (emotions.get(e1, 0.0) + emotions.get(e2, 0.0)) / 2 >= threshold:
            return label
    for (e1, e2), label in _OUTER_DYADS:
        if (emotions.get(e1, 0.0) + emotions.get(e2, 0.0)) / 2 >= threshold * 0.85:
            return label
    pos = emotions.get("joy", 0.0) + emotions.get("trust", 0.0) + emotions.get("anticipation", 0.0)
    neg = emotions.get("sadness", 0.0) + emotions.get("anger", 0.0) + emotions.get("fear", 0.0) + emotions.get("disgust", 0.0)
    if pos > neg + 0.2:
        return "content"
    elif neg > pos + 0.2:
        return "subdued"
    return "calm"


def _emotion_summary(emotions: dict, label: str) -> str:
    sorted_e = sorted(emotions.items(), key=lambda x: -x[1])
    parts = []
    for name, val in sorted_e[:3]:
        if val > 0.7:
            parts.append(f"very {name}")
        elif val > 0.5:
            parts.append(f"quite {name}")
        elif val > 0.3:
            parts.append(f"mildly {name}")
        elif val > 0.15:
            parts.append(f"slightly {name}")
    if not parts:
        return "feeling neutral and balanced"
    if len(parts) == 1:
        return f"feeling {parts[0]}"
    elif len(parts) == 2:
        return f"feeling {parts[0]} and {parts[1]}"
    else:
        return f"feeling {parts[0]}, {parts[1]}, and {parts[2]}"

# Emoji mappings for mood labels
_MOOD_EMOJI = {
    "joy": "😊", "trust": "🤝", "fear": "😰", "surprise": "😲",
    "sadness": "😢", "disgust": "😒", "anger": "😠", "anticipation": "🤔",
    "love": "❤️", "awe": "😮", "optimism": "🌟", "contempt": "😤",
    "calm": "😌", "content": "🙂", "subdued": "😔", "hope": "🌈",
    "pride": "💪", "envy": "😒", "shame": "😳", "guilt": "😞",
    "outrage": "🤬", "cynicism": "😏", "sentimentality": "🥺",
    "remorse": "😔", "aggressiveness": "😡", "disapproval": "👎",
    "submission": "😶",
}


def _get_state_file() -> str:
    try:
        from hermes_cli.config import load_config, cfg_get
        full_cfg = load_config()
        cfg_dict = cfg_get(full_cfg, "plugins", "temporal-memory") or {}
        return os.path.expanduser(
            cfg_dict.get("mood_state_file", "~/.hermes/mood-state.json")
        )
    except Exception:
        return os.path.expanduser("~/.hermes/mood-state.json")


def _load_state():
    path = Path(_get_state_file())
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return None


def _save_state(data: dict) -> None:
    path = Path(_get_state_file())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _cmd_mood_status(args) -> None:
    state = _load_state()
    if state is None:
        print("No mood state found. Mood tracking may be disabled or no conversations yet.")
        return
    emotions = state.get("emotions", _default_emotions())
    label = _resolve_mood_label(emotions, 0.55)
    summary = _emotion_summary(emotions, label)
    emoji = _MOOD_EMOJI.get(label, "🤖")
    print(f"Mood: {label} {emoji}")
    print(f"Summary: {summary}")
    print()
    print("Emotion intensities:")
    for k in EMOTION_KEYS:
        val = emotions.get(k, 0.0)
        bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
        print(f"  {k:>14s} [{bar}] {val:.3f}")
    print()
    print(f"Session turns: {state.get('turns', 0)}")
    print(f"History entries: {len(state.get('history', []))}")


def _cmd_mood_set(args) -> None:
    state = _load_state()
    if state is None:
        state = {"emotions": _default_emotions(), "history": [], "turns": 0}
    if args.emotion not in EMOTION_KEYS:
        print(f"Error: Unknown emotion '{args.emotion}'. Valid: {', '.join(EMOTION_KEYS)}")
        return
    emotions = state.get("emotions", _default_emotions())
    old = emotions.get(args.emotion, 0.0)
    emotions[args.emotion] = max(0.0, min(1.0, float(args.intensity)))
    state["emotions"] = emotions
    _save_state(state)
    label = _resolve_mood_label(emotions, 0.55)
    emoji = _MOOD_EMOJI.get(label, "🤖")
    print(f"Set {args.emotion}: {old:.3f} -> {emotions[args.emotion]:.3f}")
    print(f"New dominant mood: {label} {emoji}")


def _cmd_mood_history(args) -> None:
    state = _load_state()
    if state is None:
        print("No mood history found.")
        return
    history = state.get("history", [])
    entries = history[-args.limit:]
    if not entries:
        print("No mood history yet.")
        return
    print(f"Last {len(entries)} mood entries:")
    print()
    for e in entries:
        print(
            f"  Turn {e.get('turn', '?'):>4} | {e.get('mood_label', '?'):<16s} | "
            f"joy={e['emotions']['joy']:.2f} "
            f"trust={e['emotions']['trust']:.2f} "
            f"fear={e['emotions']['fear']:.2f} "
            f"sad={e['emotions']['sadness']:.2f} "
            f"anger={e['emotions']['anger']:.2f}"
        )


def _cmd_mood_reset(args) -> None:
    state = {"emotions": _default_emotions(), "history": [], "turns": 0}
    _save_state(state)
    label = _resolve_mood_label(_default_emotions(), 0.55)
    emoji = _MOOD_EMOJI.get(label, "🤖")
    print(f"Mood reset to baseline. Dominant mood: {label} {emoji}")


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

    # mood subcommand with its own sub-subcommands
    mood_parser = sub.add_parser("mood", help="Check or adjust OWL's emotional state")
    mood_sub = mood_parser.add_subparsers(dest="mood_cmd")

    mood_sub.add_parser("status", help="Show current emotional state")

    mood_set = mood_sub.add_parser("set", help="Set an emotion intensity")
    mood_set.add_argument("emotion", choices=EMOTION_KEYS, help="Emotion to set")
    mood_set.add_argument("intensity", type=float, help="Intensity 0.0-1.0")

    mood_hist = mood_sub.add_parser("history", help="Show mood history")
    mood_hist.add_argument("--limit", "-n", type=int, default=20, help="Number of entries")

    mood_sub.add_parser("reset", help="Reset all emotions to baseline")

    mood_parser.set_defaults(func=_dispatch_mood)

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


def _dispatch_mood(args) -> None:
    cmd = getattr(args, "mood_cmd", None) or "status"
    if cmd == "status":
        _cmd_mood_status(args)
    elif cmd == "set":
        _cmd_mood_set(args)
    elif cmd == "history":
        _cmd_mood_history(args)
    elif cmd == "reset":
        _cmd_mood_reset(args)
    else:
        print(f"Unknown mood command: {cmd}")


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
