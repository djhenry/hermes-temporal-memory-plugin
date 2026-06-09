"""
Shared test configuration.

Injects a minimal agent.memory_provider stub when hermes-agent is not
installed, so tests work in both fresh checkouts (unit-only) and full
installs (make install). When hermes-agent IS installed the real
MemoryProvider is used automatically.
"""

from __future__ import annotations

import sys
import types


# ── Hermes stub ──────────────────────────────────────────────────────────────

try:
    import agent.memory_provider  # noqa: F401
except ImportError:
    class _MemoryProvider:
        name: str = ""

        def is_available(self) -> bool: return True
        def initialize(self, session_id: str, **kwargs) -> None: pass
        def prefetch(self, query: str): return None
        def sync_turn(self, user: str, assistant: str, messages=None) -> None: pass
        def get_tool_schemas(self) -> list: return []
        def handle_tool_call(self, name: str, args: dict) -> str: return ""
        def shutdown(self) -> None: pass
        def system_prompt_block(self): return None
        def on_memory_write(self, action: str, target: str, content: str) -> None: pass
        def on_session_end(self) -> None: pass

    _agent_mod = types.ModuleType("agent")
    _mp_mod = types.ModuleType("agent.memory_provider")
    _mp_mod.MemoryProvider = _MemoryProvider  # type: ignore[attr-defined]
    _agent_mod.memory_provider = _mp_mod  # type: ignore[attr-defined]
    sys.modules["agent"] = _agent_mod
    sys.modules["agent.memory_provider"] = _mp_mod
