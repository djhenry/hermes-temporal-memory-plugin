"""Pydantic config model for the Graphiti memory provider plugin."""

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


class GraphitiConfig(BaseModel):
    # neo4j: Docker/cloud, recommended for production and multi-user deployments.
    # kuzu: embedded, no Docker, single-user (deprecated upstream — use for dev/offline only).
    # falkordblite: embedded, no Docker, production-quality (requires Python 3.12+).
    backend: Literal["neo4j", "kuzu", "falkordblite"] = "neo4j"
    recall_mode: Literal["hybrid", "context", "tools"] = "hybrid"

    # Handled at the Hermes host level, not by the plugin. Setting this has no effect.
    disable_builtin_memory_tool: bool = False

    # Planned: cap concurrent Graphiti extraction calls. Not yet implemented.
    semaphore_limit: int = Field(default=5, ge=1, le=50)
    max_recall_tokens: int = Field(default=600, ge=100, le=2000)

    # Index feature (new)
    enable_memory_index: bool = True
    max_index_tokens: int = Field(default=200, ge=50, le=500)

    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
