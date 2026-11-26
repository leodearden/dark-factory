"""The ``narrow_merge_gate`` config section: whether the docs/comments merge gate may narrow, and which paths are prose.

Rationale and the command set: docs/narrow-merge-gate.md.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class NarrowMergeGateConfig(BaseModel):
    """The narrow merge gate's knobs; off, it changes nothing about any merge verify."""

    model_config = ConfigDict(extra='forbid')

    enabled: bool = Field(
        default=False,
        description='True lets a docs-only or comments-only merge run the narrow command set instead of the full gate.',
    )
    docs_paths: list[str] = Field(
        default_factory=list,
        description="Repo-relative globs of prose files; '*' stays within one path segment and '**' spans any depth.",
    )
