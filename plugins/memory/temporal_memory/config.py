"""Pydantic config model for the temporal memory provider plugin."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ExtractionConfig(BaseModel):
    provider: Literal["openai", "anthropic", "gemini", "groq", "ollama", "inherit"] = "inherit"
    model: str | None = None
    base_url: str | None = None  # used for ollama / LM Studio / any OpenAI-compatible endpoint
    api_key: str | None = None   # override OPENAI_API_KEY for local providers (e.g. "lm-studio")
    # json_object: wide proxy compat (OpenRouter). json_schema: local models (LM Studio, Ollama).
    structured_output_mode: Literal["json_schema", "json_object", "text"] = "json_schema"


class EmbedderConfig(BaseModel):
    """Embedding model config. Falls back to OPENAI_API_KEY + default OpenAI
    endpoint if not set. Set base_url to use a local provider (Ollama, LM Studio)."""
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None


class TemporalMemoryConfig(BaseModel):
    # neo4j: Docker/cloud, recommended for production and multi-user deployments.
    # kuzu: embedded, no Docker, single-user (deprecated upstream — use for dev/offline only).
    # falkordblite: embedded, no Docker, production-quality (requires Python 3.12+).
    backend: Literal["neo4j", "kuzu", "falkordblite"] = "neo4j"
    recall_mode: Literal["hybrid", "context", "tools"] = "hybrid"

    # Handled at the Hermes host level, not by the plugin. Setting this has no effect.
    disable_builtin_memory_tool: bool = False

    # Planned: cap concurrent backend extraction calls. Not yet implemented.
    semaphore_limit: int = Field(default=5, ge=1, le=50)
    max_recall_tokens: int = Field(default=600, ge=100, le=2000)

    # Index feature (new)
    enable_memory_index: bool = True
    max_index_tokens: int = Field(default=200, ge=50, le=500)

    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)

    # --- Mood tracking (Plutchik's Wheel of Emotions) ---
    enable_mood: bool = True
    mood_decay_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    mood_influence_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    mood_label_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    mood_state_file: str = "~/.hermes/mood-state.json"
    # Baseline emotion values (0.0-1.0):
    mood_baseline_joy: float = Field(default=0.4, ge=0.0, le=1.0)
    mood_baseline_trust: float = Field(default=0.6, ge=0.0, le=1.0)
    mood_baseline_fear: float = Field(default=0.1, ge=0.0, le=1.0)
    mood_baseline_surprise: float = Field(default=0.3, ge=0.0, le=1.0)
    mood_baseline_sadness: float = Field(default=0.1, ge=0.0, le=1.0)
    mood_baseline_disgust: float = Field(default=0.05, ge=0.0, le=1.0)
    mood_baseline_anger: float = Field(default=0.05, ge=0.0, le=1.0)
    mood_baseline_anticipation: float = Field(default=0.5, ge=0.0, le=1.0)
