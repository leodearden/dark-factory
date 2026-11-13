#!/usr/bin/env python3
"""Scheduled duplicate-cluster report: one report per project, diffed against its last.

Run weekly by ``scripts/fused-memory-duplicate-audit.timer``
(``docs/prds/memory-write-path-convergence.md`` leaf kappa; ``OPERATIONS.md``
section 12). For each configured project it runs
``scripts/audit_duplicate_memories.py`` once per category as a read-only
subprocess, persists each stdout plan, builds a report from those plans and
diffs it against the project's previous report: cluster lineage by member
overlap, regrowth split by member age, and per-category comparability.

The report measures regrowth. It is ``authoritative: false`` and files
nothing: gate filing is blocked on the detector's paraphrase recall, and its
wiring belongs to task 4916. ``recall_caveat`` in
``fused-memory/config/duplicate-audit.yaml`` states which bar each number
clears, and every report carries it.

The detector is consumed through its stdout contract only, never imported,
so a report can also be rebuilt from saved plan files (``--plan``) with no
store access.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / 'fused-memory' / 'config' / 'duplicate-audit.yaml'


class DuplicateAuditConfigError(ValueError):
    """The duplicate-audit config is missing, unparseable or not a valid declaration."""


class _Frozen(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)


class ParaphraseRecall(_Frozen):
    caught: int = Field(ge=0)
    of: int = Field(gt=0)


class Fitness(_Frozen):
    regrowth_measurement: Literal['supported', 'unsupported']
    gate_filing: Literal['blocked', 'cleared']


class RecallCaveat(_Frozen):
    """The measured detector recall every report carries, and the bars it clears."""

    measured_at: str = Field(min_length=1)
    fixture: str = Field(min_length=1)
    paraphrase_band_recall: dict[str, ParaphraseRecall]
    auc_true_vs_hard_negative: float
    source: str = Field(min_length=1)
    fitness: Fitness
    gate_filing_owner_task: int


class Cadence(_Frozen):
    on_calendar: str = Field(min_length=1)
    resweep_interval_days: float = Field(gt=0)


def _distinct_non_empty(values: tuple[str, ...]) -> tuple[str, ...]:
    if not values:
        raise ValueError('must name at least one entry')
    if len(set(values)) != len(values):
        raise ValueError(f'must not repeat an entry: {list(values)}')
    return values


class DuplicateAuditConfig(_Frozen):
    """Operator declarations; every key is required, because none has a defensible default."""

    projects: tuple[str, ...]
    categories: tuple[str, ...]
    scan_limit: int = Field(gt=0)
    detector_timeout_secs: float = Field(gt=0)
    cadence: Cadence
    machine_written_kinds: tuple[str, ...]
    recall_caveat: RecallCaveat

    _distinct_projects = field_validator('projects')(_distinct_non_empty)
    _distinct_categories = field_validator('categories')(_distinct_non_empty)


def load_config(path: str | Path) -> DuplicateAuditConfig:
    target = Path(path)
    try:
        return DuplicateAuditConfig.model_validate(
            yaml.safe_load(target.read_text(encoding='utf-8')),
        )
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise DuplicateAuditConfigError(f'{target}: {exc}') from exc
