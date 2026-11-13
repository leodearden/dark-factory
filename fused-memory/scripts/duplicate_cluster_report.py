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

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

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


# ---------------------------------------------------------------------------
# The report core: pure functions over detector plans (no I/O, no store)
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1
DETECTOR_LABEL = 'fused-memory/scripts/audit_duplicate_memories.py'
NO_KIND_KEY = '(no kind)'
"""The ``child_rankings`` key for children whose ``metadata.kind`` is absent."""

Plan = Mapping[str, Any]
LabelledPlan = tuple[str, Plan]
"""``(source_label, plan)``: the label names where the plan came from."""


class DuplicatePlanCategoryError(ValueError):
    """Two plans handed to one report sweep the same category."""


def swept_categories(plan: Plan) -> list[str]:
    """The categories *plan* swept: its ``scan_coverage`` keys.

    A legacy plan (the 2026-08-11 T0 baseline) carries no coverage, so it
    falls back to the categories its groups name.
    """
    coverage = plan.get('scan_coverage')
    if coverage is not None:
        return list(coverage)
    named = (g.get('category') for g in plan.get('near_duplicate_groups') or ())
    return list(dict.fromkeys(c for c in named if c))


def _category_entry(plan: Plan, category: str, source: str) -> dict[str, Any]:
    coverage = (plan.get('scan_coverage') or {}).get(category)
    return {
        'scan_coverage': dict(coverage) if coverage is not None else None,
        'coverage_known': coverage is not None,
        'lexical_threshold': plan.get('threshold'),
        'ann_cutoff': (plan.get('ann_threshold_by_category') or {}).get(category),
        'source': source,
    }


def _category_entries(plans: Sequence[LabelledPlan]) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for source, plan in plans:
        for category in swept_categories(plan):
            if category in entries:
                raise DuplicatePlanCategoryError(
                    f'category {category!r} is swept by both '
                    f'{entries[category]["source"]} and {source}',
                )
            entries[category] = _category_entry(plan, category, source)
    return entries


def cluster_entry(group: Mapping[str, Any], machine_written_kinds: Iterable[str]) -> dict[str, Any]:
    """One report row per detector group, with facts derived from its members.

    ``canonical_present`` and ``machine_written`` are ``None`` when the plan
    carries no member facts: unknown, not false.
    """
    member_ids = list(group.get('member_ids') or ())
    members = group.get('members')
    machine_kinds = frozenset(machine_written_kinds)
    return {
        'category': group.get('category'),
        'size': len(member_ids),
        'member_ids': member_ids,
        'survivor_id': group.get('survivor_id'),
        'found_by': list(group.get('found_by') or ()),
        'ann_max_score': group.get('ann_max_score'),
        'lexical_max_ratio': group.get('lexical_max_ratio'),
        'delete_candidate_ids': list(group.get('delete_candidate_ids') or ()),
        'retained_member_ids': list(group.get('retained_member_ids') or ()),
        'members': [dict(m) for m in members] if members is not None else None,
        'topics': sorted({str(m['topic']) for m in members or () if m.get('topic')}),
        'canonical_present': (
            any(m.get('canonical') is True for m in members) if members is not None else None
        ),
        'machine_written': (
            bool(members) and all(m.get('kind') in machine_kinds for m in members)
            if members is not None else None
        ),
    }


def merge_child_counts(plans: Iterable[Plan]) -> dict[str, list[dict[str, Any]]]:
    """Sum every plan's ``child_counts`` into ``{kind: [{'parent_id', 'count'}]}``.

    Rows are sorted by ``(-count, str(parent_id))``, so the sighting-count
    ranking is ``['sighting']``.
    """
    totals: dict[str, dict[Any, int]] = {}
    for plan in plans:
        for row in plan.get('child_counts') or ():
            kind = row.get('kind')
            by_parent = totals.setdefault(NO_KIND_KEY if kind is None else str(kind), {})
            by_parent[row['parent_id']] = by_parent.get(row['parent_id'], 0) + int(row['count'])
    return {
        kind: sorted(
            ({'parent_id': parent, 'count': count} for parent, count in by_parent.items()),
            key=lambda r: (-r['count'], str(r['parent_id'])),
        )
        for kind, by_parent in sorted(totals.items())
    }


def _summed_retained_carveout(plans: Iterable[Plan]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for plan in plans:
        for reason, count in (plan.get('retained_carveout') or {}).items():
            totals[reason] = totals.get(reason, 0) + int(count)
    return totals


def build_report(
    project_id: str,
    stamp: str,
    plans: Sequence[LabelledPlan],
    *,
    config: DuplicateAuditConfig,
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble one project's report from its per-category detector plans."""
    bare_plans = [plan for _source, plan in plans]
    return {
        'schema_version': SCHEMA_VERSION,
        'project_id': project_id,
        'run_stamp': stamp,
        'detector': DETECTOR_LABEL,
        'authoritative': False,
        'recall_caveat': config.recall_caveat.model_dump(mode='json'),
        'cadence': config.cadence.model_dump(mode='json'),
        'categories': _category_entries(plans),
        'clusters': [
            cluster_entry(group, config.machine_written_kinds)
            for plan in bare_plans
            for group in plan.get('near_duplicate_groups') or ()
        ],
        'topic_carveout_groups': [
            dict(g) for plan in bare_plans for g in plan.get('topic_carveout_groups') or ()
        ],
        'retained_carveout': _summed_retained_carveout(bare_plans),
        'child_rankings': merge_child_counts(bare_plans),
        'child_counts_known': all('child_counts' in plan for plan in bare_plans),
    }
