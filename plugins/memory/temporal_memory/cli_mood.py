"""CLI commands for the mood plugin: hermes mood status | set | history | reset"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .mood_lexicon import (
    EMOTION_KEYS,
    _default_emotions,
    _resolve_mood_label,
    _emotion_summary,
)

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
    """Resolve state file path from config or default."""
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
    """Load mood state from JSON file."""
    path = Path(_get_state_file())
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return None


def _save_state(data: dict) -> None:
    """Save mood state to JSON file."""
    path = Path(_get_state_file())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def cmd_status(args) -> None:
    """Show current emotional state."""
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


def cmd_set(args) -> None:
    """Set an emotion intensity."""
    state = _load_state()
    if state is None:
        state = {"emotions": _default_emotions(), "history": [], "turns": 0}

    emotion = args.emotion
    intensity = args.intensity

    if emotion not in EMOTION_KEYS:
        print(f"Error: Unknown emotion '{emotion}'. Valid: {', '.join(EMOTION_KEYS)}")
        return

    emotions = state.get("emotions", _default_emotions())
    old = emotions.get(emotion, 0.0)
    emotions[emotion] = max(0.0, min(1.0, float(intensity)))
    state["emotions"] = emotions
    _save_state(state)

    label = _resolve_mood_label(emotions, 0.55)
    emoji = _MOOD_EMOJI.get(label, "🤖")
    print(f"Set {emotion}: {old:.3f} -> {emotions[emotion]:.3f}")
    print(f"New dominant mood: {label} {emoji}")


def cmd_history(args) -> None:
    """Show recent mood history."""
    state = _load_state()
    if state is None:
        print("No mood history found.")
        return

    history = state.get("history", [])
    limit = args.limit
    entries = history[-limit:]

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


def cmd_reset(args) -> None:
    """Reset all emotions to baseline."""
    state = {
        "emotions": _default_emotions(),
        "history": [],
        "turns": 0,
    }
    _save_state(state)
    print("Mood reset to baseline.")
    label = _resolve_mood_label(_default_emotions(), 0.55)
    emoji = _MOOD_EMOJI.get(label, "🤖")
    print(f"  Dominant mood: {label} {emoji}")


def register_mood_cli(subparser) -> None:
    """Register mood CLI subcommands with Hermes."""
    sub = subparser.add_subparsers(dest="mood_command")

    # status
    sub.add_parser("status", help="Show current emotional state")

    # set
    set_parser = sub.add_parser("set", help="Set an emotion intensity")
    set_parser.add_argument("emotion", choices=EMOTION_KEYS, help="Emotion to set")
    set_parser.add_argument("intensity", type=float, help="Intensity 0.0-1.0")

    # history
    hist_parser = sub.add_parser("history", help="Show mood history")
    hist_parser.add_argument("--limit", "-n", type=int, default=20, help="Number of entries")

    # reset
    sub.add_parser("reset", help="Reset all emotions to baseline")

    subparser.set_defaults(
        func=lambda args: {
            "status": cmd_status,
            "set": cmd_set,
            "history": cmd_history,
            "reset": cmd_reset,
        }.get(args.mood_command, lambda a: subparser.print_help())(args)
    )
