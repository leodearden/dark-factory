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

import logging
import os
import sys
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

LOG_LEVEL_ENV_VAR = 'LEGIBILITY_LOG_LEVEL'
LOG_FORMAT = '%(asctime)s %(levelname)s %(name)s: %(message)s'


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


class Timeouts(BaseModel):
    """``timeouts`` block — per-census-stage ``claude`` CLI subprocess
    budgets (seconds).

    The census runs three claude-CLI-backed stages with very different
    shapes, but originally handed every stage ``coder._invoke_cli``'s single
    module default of 120s (sized for one Haiku trickle-coding call). That
    default demonstrably killed the first dark_factory census: every Sonnet
    verify-vs-``main`` call (one per novel cluster, exploring current
    ``main`` via targeted reads) and the final large Fable synthesis call
    over all clusters timed out at 120s, so census-state.json never advanced.

    So each stage gets its own budget: ``census_mining_secs`` (120 — mining
    is the same shape as trickle coding, and the tiny headroom probe rides
    the same knob), ``census_verify_secs`` (900 — one Sonnet call per
    cluster exploring current ``main``), ``census_synthesis_secs`` (1800 —
    one large Fable call over all verified clusters). An omitted block loads
    with all three defaults, so a pre-existing legibility.yaml keeps working.
    """

    model_config = ConfigDict(extra='allow')

    census_mining_secs: int = 120
    census_verify_secs: int = 900
    census_synthesis_secs: int = 1800


class LegibilityConfig(BaseModel):
    """The full ``docs/legibility/legibility.yaml`` schema (PRD §7.4).

    ``project_id`` / ``project_root`` / ``escalation_port`` / ``cwd_prefixes``
    are required — there is no sensible project-agnostic default for any of
    them. Every nested block defaults to its own all-defaults instance, so a
    minimal YAML carrying only the four required fields still produces a
    fully-populated, typed config. ``agent_transcript_roots`` is an optional
    list of additional archive roots the miner enumerates alongside
    ``~/.claude/projects`` (the ``shared.transcript_archive`` fleet-transcript
    tree); it defaults to ``[]`` — the byte-parity baseline in which only
    ``~/.claude/projects`` is read.
    """

    model_config = ConfigDict(extra='allow')

    project_id: str
    project_root: str
    escalation_port: int
    cwd_prefixes: list[str]
    agent_transcript_roots: list[str] = Field(default_factory=list)
    budgets: Budgets = Field(default_factory=Budgets)
    sampling: Sampling = Field(default_factory=Sampling)
    census: Census = Field(default_factory=Census)
    models: Models = Field(default_factory=Models)
    timeouts: Timeouts = Field(default_factory=Timeouts)


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


def _resolve_level(value: str) -> int | None:
    """Resolve *value* to a logging level int, or None if it is not one.

    Accepts both spellings operators actually reach for: a level NAME
    (case-insensitive, surrounding whitespace tolerated) and a NUMERIC
    level. The numeric form matters because it is what ``logging``'s own
    API takes, so ``LEGIBILITY_LOG_LEVEL=10`` is a reasonable thing to
    write in a unit file — and ``getLevelName('10')`` returns the string
    ``'Level 10'``, so a name-only lookup would reject it as a typo.

    Returns None rather than raising: every caller here degrades on an
    unresolvable value, and :func:`configure_logging`'s contract is that
    nothing it is handed can take the nightly timer down.
    """
    text = value.strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    resolved = logging.getLevelName(text.upper())
    # getLevelName returns the int for a known name, else the string
    # 'Level <name>' — an int is the only success signal.
    return resolved if isinstance(resolved, int) else None


def configure_logging(default_level: str = 'INFO') -> None:
    """Point the legibility CLIs' log output at stderr, at a visible level.

    Shared by ``nightly.main()`` and ``census.main()`` so the env-var name,
    the default level and the log format are single-sourced across both
    entrypoints (INV-5 no-lockstep-duplication) rather than drifting in two
    inline copies — which is the drift class that produced this fix's
    occasion in the first place.

    WHY THIS EXISTS: nothing under ``scripts/legibility/`` used to call
    ``logging.basicConfig`` at all, so under the systemd ``ExecStart`` root
    stayed at its WARNING default and every INFO line both CLIs emit — the
    nightly sampler summary, the no-change-night line, the empty-codebook
    line — was discarded before it ever reached the journal. That is half of
    why the 14 budget-suppressed nights of 2026-07-16..29 were indis-
    tinguishable from genuine no-change nights.

    ``LEGIBILITY_LOG_LEVEL`` overrides *default_level* (matching the
    ``LEGIBILITY_SEARCH_ROOTS`` / ``LEGIBILITY_CLAUDE_BIN`` env convention
    already used across these modules). Either spelling operators reach for
    is accepted — a level NAME (``DEBUG``, case-insensitive) or a numeric
    level (``10``, which is what ``logging``'s own API takes). An
    unparseable value degrades to *default_level* and logs one warning — it
    must never raise, because a typo in a unit file's ``Environment=`` must
    not take the nightly timer down.

    NEITHER input path can raise, *default_level* included: an unknown
    *default_level* from a future caller falls back to ``INFO`` rather than
    reaching ``basicConfig`` as the string ``'Level FOO'`` and raising
    ValueError out of the helper whose whole job is to never do that.

    Call from ``main()``, NEVER at import: ``basicConfig`` is a documented
    no-op when root already has handlers, so an import-time call would be a
    silent no-op under pytest (whose logging plugin owns root) while an
    import-time call from a library context would hijack the importer's
    logging setup.
    """
    level = _resolve_level(default_level)
    if level is None:
        # Not the operator's doing, so it is not the operator's warning:
        # a bad *default_level* is a caller bug. Degrade to INFO — the
        # level this helper exists to guarantee — rather than raise.
        level = logging.INFO

    raw = os.environ.get(LOG_LEVEL_ENV_VAR)
    unparseable = None
    if raw is not None and raw.strip():
        candidate = _resolve_level(raw)
        if candidate is not None:
            level = candidate
        else:
            unparseable = raw

    logging.basicConfig(level=level, format=LOG_FORMAT, stream=sys.stderr)

    # Emitted AFTER basicConfig, deliberately. Module-level logging.warning()
    # calls basicConfig() ITSELF (with no level and the default format) when
    # root has no handlers, which would leave root at WARNING and turn the
    # explicit call above into a no-op — silently defeating the whole point of
    # this helper on precisely the runs that already have a misconfiguration.
    if unparseable is not None:
        # Reports the level actually in force, not *default_level*: those
        # differ when default_level was itself unresolvable, and a warning
        # that names a level the process is NOT running at is worse than no
        # warning at all.
        logging.getLogger('legibility.config').warning(
            '%s=%r is not a valid log level; falling back to %s. '
            'Valid values: CRITICAL, ERROR, WARNING, INFO, DEBUG, '
            'or a numeric level such as 10.',
            LOG_LEVEL_ENV_VAR, unparseable, logging.getLevelName(level),
        )
