#!/usr/bin/env python3
"""Per-project legibility config schema (PRD §7.4).

``LegibilityConfig`` models ``<root>/docs/legibility/legibility.yaml`` —
the file that opts a project into the nightly trickle (ε) and periodic
census (η). Mirrors the ``shared.task_metadata`` BeforeDone/Milestone
pydantic pattern: nested ``BaseModel`` classes with ``extra='allow'`` for
forward-compat, field-level defaults for every §7.4 leaf value, and
``load_config`` doing a plain ``yaml.safe_load`` + model construction so a
malformed config raises ``pydantic.ValidationError`` (never silently
discarded or partially applied).

Task β of the confusion-reduction PRD (plans/confusion-reduction-prd.md
§5.2, contract §7.4). Self-contained — does not import task α's
``digest.py`` or task γ's ``codebook.py``.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Budgets(BaseModel):
    """``budgets`` block — the daily digest byte cap (PRD §5.2 point 2)."""

    model_config = ConfigDict(extra='allow')

    max_daily_digest_bytes: int = 300000


class Sampling(BaseModel):
    """``sampling`` block — stratified-sampler tuning (PRD §5.2 point 2)."""

    model_config = ConfigDict(extra='allow')

    top_fraction: float = 0.12
    per_stratum_min: int = 2


class NoveltySpike(BaseModel):
    """``census.novelty_spike`` — the (c) census-trigger condition (PRD §5.2 point 6)."""

    model_config = ConfigDict(extra='allow')

    count: int = 4
    window_hours: int = 72


class Saturation(BaseModel):
    """``census.saturation`` — the mining-stop condition (PRD §5.2 point 7)."""

    model_config = ConfigDict(extra='allow')

    dup_rate: float = 0.9
    consecutive_batches: int = 2


class Census(BaseModel):
    """``census`` block — census-trigger (ζ) and saturation tuning (PRD §5.2 points 6-7)."""

    model_config = ConfigDict(extra='allow')

    max_interval_days: int = 10
    tasks_landed_threshold: int = 120
    tasks_landed_min_days: int = 7
    novelty_spike: NoveltySpike = Field(default_factory=NoveltySpike)
    floor_days: int = 5
    saturation: Saturation = Field(default_factory=Saturation)


class Models(BaseModel):
    """``models`` block — the ratified static model-routing policy (PRD §5.2 "Model routing")."""

    model_config = ConfigDict(extra='allow')

    trickle: str = 'haiku'
    census_miner: str = 'sonnet'
    census_verify: str = 'sonnet'
    census_synthesis: str = 'fable'


class LegibilityConfig(BaseModel):
    """The full ``docs/legibility/legibility.yaml`` schema (PRD §7.4).

    ``project_id`` / ``project_root`` / ``escalation_port`` / ``cwd_prefixes``
    are required — there is no sensible project-agnostic default for any of
    them. Every nested block defaults to its own all-defaults instance, so a
    minimal YAML carrying only the four required fields still produces a
    fully-populated, typed config.
    """

    model_config = ConfigDict(extra='allow')

    project_id: str
    project_root: str
    escalation_port: int
    cwd_prefixes: list[str]
    budgets: Budgets = Field(default_factory=Budgets)
    sampling: Sampling = Field(default_factory=Sampling)
    census: Census = Field(default_factory=Census)
    models: Models = Field(default_factory=Models)


def load_config(path: Path | str) -> LegibilityConfig:
    """Load and validate a ``legibility.yaml`` file into a typed config.

    Raises ``pydantic.ValidationError`` on malformed input (missing
    required field, wrong type, ...) — propagated, never swallowed, so a
    broken config fails loudly at CLI startup rather than degrading into a
    partially-defaulted sampler run.
    """
    with open(path, encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}
    return LegibilityConfig(**raw)
