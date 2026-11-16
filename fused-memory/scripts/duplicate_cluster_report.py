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

Exit status: 0 when every report was written; 1 when a report could not be
written, a detector run failed (the report is still written, listing it under
``failed_categories``), or a project's previous report could not be read as
one (checked before that project is swept, which is then skipped); 2 for bad
input or config.

Usage
-----
  # Scheduled run (what the timer does): every configured project.
  python fused-memory/scripts/duplicate_cluster_report.py

  # Rebuild one project's report from saved plans, with no store access.
  python fused-memory/scripts/duplicate_cluster_report.py --project-id dark_factory \\
      --plan plan-a.json --plan plan-b.json --run-stamp 20260811T120000Z
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from shared.cli_boundary import LoudArgumentParser, run_cli
from shared.memory_eval_metrics import MetricSchemaError, run_stamp, stamp_datetime, validate_stamp
from shared.safe_io import atomic_write_text

from fused_memory.utils.created_at import parse_created_at

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / 'fused-memory' / 'config' / 'duplicate-audit.yaml'
DEFAULT_OUT_ROOT = REPO_ROOT / 'fused-memory' / 'data' / 'duplicate-cluster-reports'
AUDIT_SCRIPT = REPO_ROOT / 'fused-memory' / 'scripts' / 'audit_duplicate_memories.py'
_STDERR_TAIL_CHARS = 2000

EXIT_OK = 0
EXIT_RUN_FAILED = 1
"""The run did not finish cleanly: a report could not be written, a detector
run failed (that project's report is still written and lists it), or a
previous report could not be read (that project is not swept). Agrees with
shared.cli_boundary.EXIT_STDOUT_FAILED."""
EXIT_BAD_INPUT = 2
"""Bad invocation or declaration: a config error, a malformed --run-stamp, an
unreadable or non-JSON --plan file, two plans for one category, --plan without
--project-id, or an unconfigured --project-id. Agrees with argparse's own 2."""

logger = logging.getLogger('duplicate_cluster_report')


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
        'child_counts_known': 'child_counts' in plan,
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
        'retained_members': list(group.get('retained_members') or ()),
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
    """Sum every plan's ``child_counts`` into ``{kind: [{'parent_id', 'category', 'count'}]}``.

    One row per parent per category: counts from two categories come from two
    sweeps with their own coverage, so they are never merged. Rows are sorted
    by ``(-count, str(parent_id), str(category))``, so the sighting-count
    ranking is ``['sighting']``.
    """
    totals: dict[str, dict[tuple[Any, Any], int]] = {}
    for plan in plans:
        for row in plan.get('child_counts') or ():
            kind = row.get('kind')
            by_parent = totals.setdefault(NO_KIND_KEY if kind is None else str(kind), {})
            key = (row.get('category'), row['parent_id'])
            by_parent[key] = by_parent.get(key, 0) + int(row['count'])
    return {
        kind: sorted(
            (
                {'parent_id': parent, 'category': category, 'count': count}
                for (category, parent), count in by_parent.items()
            ),
            key=lambda r: (-r['count'], str(r['parent_id']), str(r['category'])),
        )
        for kind, by_parent in sorted(totals.items())
    }


def _summed_retained_carveout(plans: Iterable[Plan]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for plan in plans:
        for reason, count in (plan.get('retained_carveout') or {}).items():
            totals[reason] = totals.get(reason, 0) + int(count)
    return totals


class FailureKind(Enum):
    """Why a (project, category) detector run produced no plan."""

    STDERR_LOG_FAILED = 'stderr_log_failed'
    """Its stderr log could not be opened, so the detector was not run."""
    SPAWN_FAILED = 'spawn_failed'
    """The detector could not be started."""
    TIMED_OUT = 'timed_out'
    """It ran past ``detector_timeout_secs`` and was killed."""
    DETECTOR_EXIT = 'detector_exit'
    """It exited non-zero."""
    BAD_PLAN_OUTPUT = 'bad_plan_output'
    """It exited 0, but its stdout was not a JSON object."""
    PLAN_WRITE_FAILED = 'plan_write_failed'
    """Its plan could not be persisted."""


@dataclass(frozen=True)
class DetectorFailure:
    """One (project, category) detector run that produced no plan."""

    category: str
    error_type: FailureKind
    exit_code: int | None
    stderr_tail: str
    exception_type: str | None = None
    """The class name of the OSError behind a log, spawn or plan-write failure."""

    def as_dict(self) -> dict[str, Any]:
        return {
            'category': self.category, 'error_type': self.error_type.value,
            'exit_code': self.exit_code, 'stderr_tail': self.stderr_tail,
            'exception_type': self.exception_type,
        }


def build_report(
    project_id: str,
    stamp: str,
    plans: Sequence[LabelledPlan],
    *,
    config: DuplicateAuditConfig,
    previous: Mapping[str, Any] | None = None,
    failed_categories: Sequence[DetectorFailure] = (),
) -> dict[str, Any]:
    """Assemble one project's report from its per-category detector plans.

    With *previous* (that project's prior report) the report is also diffed
    against it; see ``diff_against_previous``. *failed_categories* names the
    categories whose detector run produced no plan.
    """
    bare_plans = [plan for _source, plan in plans]
    inventory = {
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
        'failed_categories': [failure.as_dict() for failure in failed_categories],
    }
    return {**inventory, **diff_against_previous(inventory, previous, config=config)}


# ---------------------------------------------------------------------------
# The consecutive-run regrowth diff: candidate-cluster lineage, not topics
# ---------------------------------------------------------------------------

class LineageStatus(Enum):
    BASELINE = 'baseline'
    """No previous report to diff against."""
    NEW = 'new'
    """Overlaps no previous cluster of its category."""
    UNCHANGED = 'unchanged'
    GREW = 'grew'
    """Gained at least one member (it may also have lost some: size_delta is the net)."""
    SHRANK = 'shrank'
    """Lost members and gained none."""
    SPLIT = 'split'
    """Overlaps a previous cluster whose key a larger overlap inherited."""


class MemberAge(Enum):
    RE_EMITTED = 're_emitted'
    """Created after the previous run: regrowth."""
    PREEXISTING = 'preexisting'
    """Created before the previous run: not regrowth (newly detected, or moved
    in from another cluster)."""
    UNKNOWN = 'unknown'


class Incomparable(Enum):
    PREVIOUS_MISSING_CATEGORY = 'previous_missing_category'
    COVERAGE_UNKNOWN = 'coverage_unknown'
    SCAN_TRUNCATED = 'scan_truncated'
    SCAN_LIMIT_CHANGED = 'scan_limit_changed'
    LEXICAL_THRESHOLD_CHANGED = 'lexical_threshold_changed'
    ANN_CUTOFF_CHANGED = 'ann_cutoff_changed'


@dataclass(frozen=True)
class Lineage:
    """One current cluster's relation to the previous report's clusters."""

    cluster_key: str
    status: LineageStatus
    previous_key: str | None = None
    split_from: str | None = None
    overlapped_previous_keys: tuple[str, ...] = ()
    added_member_ids: tuple[Any, ...] = ()
    departed_member_ids: tuple[Any, ...] = ()
    size_delta: int | None = None


def _min_member_key(member_ids: Iterable[Any]) -> str:
    return min(str(m) for m in member_ids)


def _overlaps(
    current: Sequence[Mapping[str, Any]], previous: Sequence[Mapping[str, Any]],
) -> dict[int, list[tuple[int, str]]]:
    """``{current index: [(overlap, previous key)]}``, within a category only, largest first.

    One pass over each side's members, through a (category, member) index of
    the previous clusters holding it.
    """
    holders: dict[tuple[Any, Any], list[str]] = {}
    for prior in previous:
        for member in set(prior['member_ids']):
            holders.setdefault((prior['category'], member), []).append(prior['cluster_key'])
    found: dict[int, list[tuple[int, str]]] = {}
    for index, cluster in enumerate(current):
        hits = Counter(
            key
            for member in set(cluster['member_ids'])
            for key in holders.get((cluster['category'], member), ())
        )
        found[index] = sorted(((n, key) for key, n in hits.items()), key=lambda t: (-t[0], t[1]))
    return found


def _inherit_greedily(overlaps: Mapping[int, list[tuple[int, str]]]) -> dict[int, str]:
    """Hand each previous key to at most one current cluster, largest overlap first."""
    ranked = sorted(
        ((n, key, index) for index, hits in overlaps.items() for n, key in hits),
        key=lambda t: (-t[0], t[1], t[2]),
    )
    inherited: dict[int, str] = {}
    taken: set[str] = set()
    for _n, key, index in ranked:
        if index not in inherited and key not in taken:
            inherited[index] = key
            taken.add(key)
    return inherited


def _fresh_key(member_ids: Iterable[Any], taken: set[str], stamp: str) -> str:
    """A newborn cluster's key: its minimum member id, stamped if an inherited key holds it."""
    key = _min_member_key(member_ids)
    return key if key not in taken else f'{key}@{stamp}'


def _compared_lineage(
    cluster: Mapping[str, Any], reference: Mapping[str, Any], *, key: str, split: bool,
    overlapped: tuple[str, ...],
) -> Lineage:
    """Compare *cluster* with the previous cluster it inherits from, or split off from.

    ``size_delta`` is the net change against *reference*.
    """
    current_ids = list(cluster['member_ids'])
    previous_ids = list(reference['member_ids'])
    current_set, previous_set = set(current_ids), set(previous_ids)
    added = tuple(m for m in current_ids if m not in previous_set)
    size_delta = len(current_ids) - len(previous_ids)
    if split:
        return Lineage(
            cluster_key=key, status=LineageStatus.SPLIT, split_from=reference['cluster_key'],
            overlapped_previous_keys=overlapped, added_member_ids=added, size_delta=size_delta,
        )
    departed = tuple(m for m in previous_ids if m not in current_set)
    status = (LineageStatus.GREW if added else
              LineageStatus.SHRANK if departed else LineageStatus.UNCHANGED)
    return Lineage(
        cluster_key=key, status=status, previous_key=reference['cluster_key'],
        overlapped_previous_keys=overlapped, added_member_ids=added,
        departed_member_ids=departed, size_delta=size_delta,
    )


def match_lineage(
    current: Sequence[Mapping[str, Any]],
    previous: Sequence[Mapping[str, Any]] | None,
    *,
    stamp: str,
) -> list[Lineage]:
    """Relate each current cluster to the previous report's clusters, by member overlap.

    Matching never crosses a category. Each previous key is inherited at most
    once, greedily by descending overlap with ties to the lower key, so a
    merge keeps the larger contributor's key and a split leaves the key on
    its larger piece. A cluster keeps the key it was born with (its minimum
    member id) for as long as the matches carry it forward.
    """
    if previous is None:
        return [
            Lineage(cluster_key=_min_member_key(c['member_ids']), status=LineageStatus.BASELINE)
            for c in current
        ]
    by_key = {prior['cluster_key']: prior for prior in previous}
    overlaps = _overlaps(current, previous)
    inherited = _inherit_greedily(overlaps)
    taken = set(inherited.values())
    lineages = []
    for index, cluster in enumerate(current):
        overlapped = tuple(sorted(key for _n, key in overlaps[index]))
        if index in inherited:
            lineages.append(_compared_lineage(
                cluster, by_key[inherited[index]], key=inherited[index], split=False,
                overlapped=overlapped,
            ))
            continue
        key = _fresh_key(cluster['member_ids'], taken, stamp)
        taken.add(key)
        if overlaps[index]:
            lineages.append(_compared_lineage(
                cluster, by_key[overlaps[index][0][1]], key=key, split=True,
                overlapped=overlapped,
            ))
        else:
            lineages.append(Lineage(
                cluster_key=key, status=LineageStatus.NEW,
                added_member_ids=tuple(cluster['member_ids']),
                size_delta=len(cluster['member_ids']),
            ))
    return lineages


def classify_added(
    added_ids: Iterable[Any],
    members: Sequence[Mapping[str, Any]] | None,
    previous_stamp: datetime,
) -> dict[str, int]:
    """Count *added_ids* by age against the previous run's stamp."""
    created = {m.get('id'): m.get('created_at') for m in members or ()}
    counts = dict.fromkeys((age.value for age in MemberAge), 0)
    for member_id in added_ids:
        moment = parse_created_at(created.get(member_id))
        age = (MemberAge.UNKNOWN if moment is None else
               MemberAge.RE_EMITTED if moment > previous_stamp else MemberAge.PREEXISTING)
        counts[age.value] += 1
    return counts


def _per_day(count: int, elapsed_days: float | None) -> float | None:
    return count / elapsed_days if elapsed_days else None


def _lineage_dict(
    lineage: Lineage,
    cluster: Mapping[str, Any],
    previous_stamp: datetime | None,
    elapsed_days: float | None,
) -> dict[str, Any]:
    by_age = (
        classify_added(lineage.added_member_ids, cluster.get('members'), previous_stamp)
        if previous_stamp is not None else dict.fromkeys((a.value for a in MemberAge), 0)
    )
    return {
        'status': lineage.status.value,
        'previous_key': lineage.previous_key,
        'split_from': lineage.split_from,
        'overlapped_previous_keys': list(lineage.overlapped_previous_keys),
        'added_member_ids': list(lineage.added_member_ids),
        'departed_member_ids': list(lineage.departed_member_ids),
        'size_delta': lineage.size_delta,
        'added_by_age': by_age,
        'rate_per_day': _per_day(by_age[MemberAge.RE_EMITTED.value], elapsed_days),
    }


def category_comparability(
    current: Mapping[str, Any], previous: Mapping[str, Any] | None,
) -> list[str]:
    """Why this category's diff may not compare like with like; empty means comparable.

    A reason discloses; it never suppresses the diff.
    """
    reasons: set[Incomparable] = set()
    if previous is None:
        reasons.add(Incomparable.PREVIOUS_MISSING_CATEGORY)
    for entry in (current,) if previous is None else (current, previous):
        if not entry['coverage_known']:
            reasons.add(Incomparable.COVERAGE_UNKNOWN)
        elif entry['scan_coverage'].get('truncated'):
            reasons.add(Incomparable.SCAN_TRUNCATED)
    if previous is not None:
        if (current['coverage_known'] and previous['coverage_known']
                and current['scan_coverage'].get('scan_limit')
                != previous['scan_coverage'].get('scan_limit')):
            reasons.add(Incomparable.SCAN_LIMIT_CHANGED)
        if current['lexical_threshold'] != previous['lexical_threshold']:
            reasons.add(Incomparable.LEXICAL_THRESHOLD_CHANGED)
        if current['ann_cutoff'] != previous['ann_cutoff']:
            reasons.add(Incomparable.ANN_CUTOFF_CHANGED)
    return sorted(reason.value for reason in reasons)


def _categories_with_known_child_counts(previous: Mapping[str, Any] | None) -> frozenset[str]:
    """The categories whose children *previous* counted: swept with the flag set."""
    if previous is None:
        return frozenset()
    return frozenset(
        name for name, entry in previous['categories'].items()
        if entry.get('child_counts_known') is True
    )


def child_deltas(
    current: Mapping[str, Sequence[Mapping[str, Any]]], previous: Mapping[str, Any] | None,
) -> dict[str, list[dict[str, Any]]]:
    """Each ranking row with its previous count and delta.

    Both are known only for a category the previous report counted (see
    ``_categories_with_known_child_counts``); a parent absent from such a
    category's previous ranking had a known zero. For any other category
    (no previous report, a failed or newly configured category, a legacy
    plan) both are ``None``: an unknown count is not a zero.
    """
    known = _categories_with_known_child_counts(previous)
    prior = {
        (kind, row.get('category'), row['parent_id']): row['count']
        for kind, rows in (previous['child_rankings'] if previous is not None else {}).items()
        for row in rows
    }

    def with_delta(kind: str, row: Mapping[str, Any]) -> dict[str, Any]:
        if row['category'] not in known:
            return {**row, 'previous_count': None, 'delta': None}
        previous_count = prior.get((kind, row['category'], row['parent_id']), 0)
        return {**row, 'previous_count': previous_count, 'delta': row['count'] - previous_count}

    return {kind: [with_delta(kind, row) for row in rows] for kind, rows in current.items()}


def _dissolved(
    lineages: Iterable[Lineage], previous: Sequence[Mapping[str, Any]], swept: Iterable[str],
) -> list[dict[str, Any]]:
    """Previous clusters no current cluster overlaps, in categories this run swept."""
    overlapped = {key for lineage in lineages for key in lineage.overlapped_previous_keys}
    swept_now = set(swept)
    return [
        {'cluster_key': c['cluster_key'], 'category': c['category'], 'size': len(c['member_ids'])}
        for c in previous
        if c['category'] in swept_now and c['cluster_key'] not in overlapped
    ]


def _regrowth_totals(
    clusters: Sequence[Mapping[str, Any]], elapsed_days: float | None,
) -> dict[str, Any]:
    statuses = [c['lineage']['status'] for c in clusters]
    re_emitted = sum(c['lineage']['added_by_age'][MemberAge.RE_EMITTED.value] for c in clusters)
    return {
        'appeared': statuses.count(LineageStatus.NEW.value),
        'grew': statuses.count(LineageStatus.GREW.value),
        'shrank': statuses.count(LineageStatus.SHRANK.value),
        'unchanged': statuses.count(LineageStatus.UNCHANGED.value),
        'split': statuses.count(LineageStatus.SPLIT.value),
        're_emitted_members': re_emitted,
        'rate_per_day': _per_day(re_emitted, elapsed_days),
    }


def diff_against_previous(
    inventory: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
    *,
    config: DuplicateAuditConfig,
) -> dict[str, Any]:
    """The report keys the diff adds or extends: clusters, categories, child_rankings, regrowth.

    ``regrowth`` totals each category under ``by_category``. Its headline
    totals leave out ``excluded_categories``: those the previous report did
    not sweep, whose every cluster would otherwise count as new.
    """
    stamp = inventory['run_stamp']
    previous_stamp = stamp_datetime(previous['run_stamp']) if previous is not None else None
    elapsed_days = (
        (stamp_datetime(stamp) - previous_stamp).total_seconds() / 86400.0
        if previous_stamp is not None else None
    )
    previous_clusters = previous['clusters'] if previous is not None else None
    lineages = match_lineage(inventory['clusters'], previous_clusters, stamp=stamp)
    clusters = [
        {**cluster, 'cluster_key': lineage.cluster_key,
         'lineage': _lineage_dict(lineage, cluster, previous_stamp, elapsed_days)}
        for cluster, lineage in zip(inventory['clusters'], lineages, strict=True)
    ]
    previous_categories = previous['categories'] if previous is not None else {}
    categories = {
        name: {**entry, 'incomparable_reasons': category_comparability(
            entry, previous_categories.get(name),
        )}
        for name, entry in inventory['categories'].items()
    }
    unswept_before = sorted(
        name for name, entry in categories.items()
        if Incomparable.PREVIOUS_MISSING_CATEGORY.value in entry['incomparable_reasons']
    )
    headline_categories = set(categories) - set(unswept_before)
    return {
        'clusters': clusters,
        'categories': categories,
        'child_rankings': child_deltas(inventory['child_rankings'], previous),
        'regrowth': {
            'previous_run_stamp': previous['run_stamp'] if previous is not None else None,
            'elapsed_days': elapsed_days,
            'previous_overdue': (
                elapsed_days > 2 * config.cadence.resweep_interval_days
                if elapsed_days is not None else None
            ),
            'dissolved': _dissolved(lineages, previous_clusters or (), inventory['categories']),
            'by_category': {
                name: _regrowth_totals([c for c in clusters if c['category'] == name], elapsed_days)
                for name in categories
            },
            'excluded_categories': unswept_before,
            **_regrowth_totals(
                [c for c in clusters if c['category'] in headline_categories], elapsed_days,
            ),
        },
    }


# ---------------------------------------------------------------------------
# Artifacts: <out_root>/<project_id>/{plan,report,summary}-<STAMP>.*
# ---------------------------------------------------------------------------

_REPORT_PREFIX = 'report-'
_REPORT_SUFFIX = '.json'
_TOP_REGROWTH_CLUSTERS = 10


def report_path(out_root: Path, project_id: str, stamp: str) -> Path:
    return Path(out_root) / project_id / f'{_REPORT_PREFIX}{stamp}{_REPORT_SUFFIX}'


def summary_path(out_root: Path, project_id: str, stamp: str) -> Path:
    return Path(out_root) / project_id / f'summary-{stamp}.txt'


def plan_path(out_root: Path, project_id: str, stamp: str, category: str) -> Path:
    return Path(out_root) / project_id / f'plan-{stamp}-{category}.json'


def stderr_log_path(out_root: Path, project_id: str, stamp: str, category: str) -> Path:
    return Path(out_root) / project_id / f'plan-{stamp}-{category}.stderr.log'


def _fitness_line(caveat: Mapping[str, Any]) -> str:
    fitness = caveat['fitness']
    return (
        f"regrowth measurement: {fitness['regrowth_measurement']}; "
        f"gate filing: {fitness['gate_filing']} (task {caveat['gate_filing_owner_task']})"
    )


def _previous_line(regrowth: Mapping[str, Any]) -> str:
    if regrowth['previous_run_stamp'] is None:
        return 'previous report: none (baseline)'
    overdue = ' -- OVERDUE' if regrowth['previous_overdue'] else ''
    return (f"previous report: {regrowth['previous_run_stamp']} "
            f"({regrowth['elapsed_days']:.1f} days earlier){overdue}")


def _regrowth_line(regrowth: Mapping[str, Any]) -> str:
    rate = regrowth['rate_per_day']
    per_day = f'{rate:.3f}/day' if rate is not None else 'no rate'
    excluded = regrowth['excluded_categories']
    exclusion = (
        f"; excludes {', '.join(excluded)} (not swept by the previous report)"
        if excluded and regrowth['previous_run_stamp'] is not None else ''
    )
    return (
        f"regrowth: appeared {regrowth['appeared']}, grew {regrowth['grew']}, "
        f"shrank {regrowth['shrank']}, unchanged {regrowth['unchanged']}, "
        f"split {regrowth['split']}, dissolved {len(regrowth['dissolved'])}; "
        f"re-emitted members {regrowth['re_emitted_members']} ({per_day}){exclusion}"
    )


def _category_lines(report: Mapping[str, Any]) -> list[str]:
    lines = ['categories:']
    for name, entry in report['categories'].items():
        count = sum(1 for c in report['clusters'] if c['category'] == name)
        reasons = entry['incomparable_reasons']
        verdict = 'incomparable: ' + ', '.join(reasons) if reasons else 'comparable'
        lines.append(f"  {name}: {count} cluster{'' if count == 1 else 's'}; {verdict}")
    return lines


def _top_regrowth_lines(clusters: Sequence[Mapping[str, Any]]) -> list[str]:
    grown = sorted(
        (c for c in clusters if c['lineage']['added_member_ids']),
        key=lambda c: (-len(c['lineage']['added_member_ids']), c['cluster_key']),
    )[:_TOP_REGROWTH_CLUSTERS]
    lines = ['top regrowth clusters (by added members):']
    for c in grown:
        lineage = c['lineage']
        lines.append(
            f"  {c['cluster_key']} [{c['category']}] {lineage['status']}: size {c['size']}, "
            f"+{len(lineage['added_member_ids'])} added "
            f"({lineage['added_by_age'][MemberAge.RE_EMITTED.value]} re-emitted)"
        )
    return lines if grown else [*lines, '  none']


def _failure_lines(failed: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = ['failed categories:']
    lines += [f"  {f['category']}: {f['error_type']} (exit {f['exit_code']})" for f in failed]
    return lines if failed else [*lines, '  none']


def render_summary(report: Mapping[str, Any]) -> str:
    """The human-readable digest of *report*; the JSON report is the record."""
    regrowth = report['regrowth']
    lines = [
        f"duplicate-cluster report: {report['project_id']} @ {report['run_stamp']}",
        f"authoritative: false; {_fitness_line(report['recall_caveat'])}",
        _previous_line(regrowth),
        _regrowth_line(regrowth),
        *_category_lines(report),
        *_top_regrowth_lines(report['clusters']),
        *_failure_lines(report['failed_categories']),
    ]
    return '\n'.join(lines) + '\n'


def write_report(
    out_root: Path, project_id: str, stamp: str, report: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Write the JSON report and its summary atomically; return both paths."""
    json_path = report_path(out_root, project_id, stamp)
    text_path = summary_path(out_root, project_id, stamp)
    atomic_write_text(json_path, json.dumps(report, indent=2, sort_keys=True) + '\n', mkdir=True)
    atomic_write_text(text_path, render_summary(report), mkdir=True)
    return json_path, text_path


def _report_stamp(path: Path) -> str | None:
    stamp = path.name[len(_REPORT_PREFIX):-len(_REPORT_SUFFIX)]
    try:
        return validate_stamp(stamp, what='report stamp')
    except MetricSchemaError:
        return None


class PreviousReportError(ValueError):
    """The project's previous report exists but is not a report the diff can read."""


class _PriorShape(BaseModel):
    model_config = ConfigDict(extra='allow', strict=True)


class _PriorCluster(_PriorShape):
    cluster_key: str
    category: str | None
    member_ids: list[str | int]


class _PriorCategory(_PriorShape):
    scan_coverage: dict[str, Any] | None
    coverage_known: bool
    lexical_threshold: float | None
    ann_cutoff: float | None
    child_counts_known: bool

    @model_validator(mode='after')
    def _coverage_is_known_exactly_when_recorded(self) -> _PriorCategory:
        if self.coverage_known != (self.scan_coverage is not None):
            raise ValueError('coverage_known disagrees with scan_coverage')
        return self


class _PriorChildRow(_PriorShape):
    parent_id: str | int
    category: str | None
    count: int


class _PriorReport(_PriorShape):
    """The part of a previous report that ``diff_against_previous`` reads."""

    project_id: str
    run_stamp: str
    clusters: list[_PriorCluster]
    categories: dict[str, _PriorCategory]
    child_rankings: dict[str, list[_PriorChildRow]]

    @field_validator('run_stamp')
    @classmethod
    def _a_run_stamp(cls, value: str) -> str:
        return validate_stamp(value, what='run_stamp')


def _read_report(path: Path) -> dict[str, Any]:
    try:
        prior = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise PreviousReportError(f'cannot read {path}: {exc}') from exc
    if not isinstance(prior, dict):
        raise PreviousReportError(f'{path} is not a JSON object')
    return prior


def find_previous_report(
    out_root: Path, project_id: str, before_stamp: str,
) -> tuple[str, dict[str, Any]] | None:
    """The project's latest report stamped strictly before *before_stamp*, or None.

    Raises PreviousReportError, naming the file, when that report cannot be
    read or lacks the shape the diff reads, so a bad history stops before a
    sweep rather than after it.
    """
    directory = Path(out_root) / project_id
    if not directory.is_dir():
        return None
    try:
        stamped = sorted(
            ((stamp, path)
             for path in directory.glob(f'{_REPORT_PREFIX}*{_REPORT_SUFFIX}')
             if (stamp := _report_stamp(path)) is not None and stamp < before_stamp),
            reverse=True,
        )
    except OSError as exc:
        raise PreviousReportError(f'cannot list {directory}: {exc}') from exc
    for stamp, path in stamped:
        prior = _read_report(path)
        if prior.get('project_id') == project_id:
            try:
                _PriorReport.model_validate(prior)
            except ValidationError as exc:
                raise PreviousReportError(f'{path} is not a report the diff can read: {exc}') from exc
            return stamp, prior
    return None


def _previous_report(out_root: Path, project_id: str, before_stamp: str) -> dict[str, Any] | None:
    """``find_previous_report``'s report alone."""
    found = find_previous_report(out_root, project_id, before_stamp)
    return found[1] if found is not None else None


# ---------------------------------------------------------------------------
# The CLI shell
# ---------------------------------------------------------------------------

class PlanInputError(ValueError):
    """A --plan file could not be read as a JSON plan object."""


def _load_plans(paths: Sequence[str]) -> list[LabelledPlan]:
    plans: list[LabelledPlan] = []
    for path in paths:
        try:
            plan = json.loads(Path(path).read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise PlanInputError(f'cannot read plan {path}: {exc}') from exc
        if not isinstance(plan, dict):
            raise PlanInputError(f'plan {path} is not a JSON object')
        plans.append((str(path), plan))
    return plans


def _publish(out_root: Path, project_id: str, stamp: str, report: Mapping[str, Any]) -> int:
    """Write *report*, then print its summary. A write failure is converted here,
    so ``run_cli``'s OSError arm only ever means stdout
    (shared/src/shared/cli_boundary.py::run_cli)."""
    try:
        json_path, _summary = write_report(out_root, project_id, stamp, report)
    except OSError as exc:
        logger.error('ABORT: cannot write the report for %s under %s: %s', project_id, out_root, exc)
        return EXIT_RUN_FAILED
    logger.info('wrote %s', json_path)
    print(render_summary(report), end='')
    return EXIT_OK


def _run_plan_mode(args: argparse.Namespace, config: DuplicateAuditConfig, stamp: str) -> int:
    if not args.project_id:
        logger.error('--plan needs --project-id: a report belongs to one project')
        return EXIT_BAD_INPUT
    try:
        previous = _previous_report(args.out_root, args.project_id, stamp)
    except PreviousReportError as exc:
        logger.error('ABORT: %s', exc)
        return EXIT_RUN_FAILED
    try:
        built = build_report(
            args.project_id, stamp, _load_plans(args.plan), config=config, previous=previous,
        )
    except (PlanInputError, DuplicatePlanCategoryError) as exc:
        logger.error('%s', exc)
        return EXIT_BAD_INPUT
    return _publish(args.out_root, args.project_id, stamp, built)


def detector_command(
    detector_argv: Sequence[str], project_id: str, category: str, config: DuplicateAuditConfig,
) -> list[str]:
    """One read-only, metrics-free detector run: never --apply, never a second e6 series."""
    return [
        *detector_argv, '--project-id', project_id, '--categories', category,
        '--scan-limit', str(config.scan_limit), '--no-metrics',
    ]


def _relay_log(log_path: Path) -> str:
    """Pass the finished run's stderr log on to this unit's journal; return its tail."""
    try:
        text = log_path.read_text(encoding='utf-8', errors='replace')
    except OSError as exc:
        return f'cannot read the stderr log {log_path}: {exc}'
    if text:
        sys.stderr.write(text)
        sys.stderr.flush()
    return text[-_STDERR_TAIL_CHARS:].strip()


def _json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def run_detector(
    project_id: str,
    category: str,
    *,
    config: DuplicateAuditConfig,
    detector_argv: Sequence[str],
    out_path: Path,
    log_path: Path,
) -> dict[str, Any] | DetectorFailure:
    """Run the detector for one category: its plan, persisted at *out_path*, or why there is none.

    The detector's stderr streams into *log_path* as it is written, so a run
    stopped mid-category keeps what it had logged; a finished run's log is
    then relayed to the journal.
    """
    def failed(kind: FailureKind, tail: str, exit_code: int | None = None,
               exc: OSError | None = None) -> DetectorFailure:
        return DetectorFailure(category, kind, exit_code, tail,
                               type(exc).__name__ if exc is not None else None)

    argv = detector_command(detector_argv, project_id, category, config)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open('w', encoding='utf-8')
    except OSError as exc:
        return failed(FailureKind.STDERR_LOG_FAILED, f'cannot open {log_path}: {exc}', exc=exc)
    with log:
        try:
            completed = subprocess.run(
                argv, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=log, text=True,
                timeout=config.detector_timeout_secs, check=False,
            )
        except subprocess.TimeoutExpired:
            completed = None
        except OSError as exc:
            return failed(FailureKind.SPAWN_FAILED, f'could not start {argv[0]}: {exc}', exc=exc)
    tail = _relay_log(log_path)
    if completed is None:
        return failed(FailureKind.TIMED_OUT, tail)
    if completed.returncode != 0:
        return failed(FailureKind.DETECTOR_EXIT, tail, completed.returncode)
    plan = _json_object(completed.stdout)
    if plan is None:
        return failed(FailureKind.BAD_PLAN_OUTPUT, tail, completed.returncode)
    try:
        atomic_write_text(out_path, completed.stdout, mkdir=True)
    except OSError as exc:
        return failed(FailureKind.PLAN_WRITE_FAILED, f'cannot write {out_path}: {exc}',
                      completed.returncode, exc)
    return plan


def _project_plans(
    project_id: str, stamp: str, *, config: DuplicateAuditConfig, out_root: Path,
    detector_argv: Sequence[str],
) -> tuple[list[LabelledPlan], list[DetectorFailure]]:
    plans: list[LabelledPlan] = []
    failures: list[DetectorFailure] = []
    for category in config.categories:
        out_path = plan_path(out_root, project_id, stamp, category)
        outcome = run_detector(
            project_id, category, config=config, detector_argv=detector_argv,
            out_path=out_path, log_path=stderr_log_path(out_root, project_id, stamp, category),
        )
        if isinstance(outcome, DetectorFailure):
            logger.error('detector run failed for %s/%s: %s (exit %s)',
                         project_id, category, outcome.error_type.value, outcome.exit_code)
            failures.append(outcome)
        else:
            plans.append((str(out_path), outcome))
    return plans, failures


def run_scheduled(
    config: DuplicateAuditConfig,
    *,
    out_root: Path,
    stamp: str,
    projects: Sequence[str],
    detector_argv: Sequence[str] = (sys.executable, str(AUDIT_SCRIPT)),
) -> int:
    """Sweep each project category by category, then write its report.

    Sequential by design: one detector run is the whole Qdrant load. A failed
    category is reported, never fatal to the others. A project whose previous
    report cannot be read is not swept at all, and never stops the projects
    after it.
    """
    worst = EXIT_OK
    for project_id in projects:
        try:
            previous = _previous_report(out_root, project_id, stamp)
        except PreviousReportError as exc:
            logger.error('ABORT: %s; %s was not swept', exc, project_id)
            worst = EXIT_RUN_FAILED
            continue
        plans, failures = _project_plans(
            project_id, stamp, config=config, out_root=out_root, detector_argv=detector_argv,
        )
        built = build_report(project_id, stamp, plans, config=config, previous=previous,
                             failed_categories=failures)
        published = _publish(out_root, project_id, stamp, built)
        worst = max(worst, published, EXIT_RUN_FAILED if failures else EXIT_OK)
    return worst


def _build_parser() -> LoudArgumentParser:
    parser = LoudArgumentParser(
        description='Build the duplicate-cluster report and diff it against the previous one.',
    )
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG_PATH,
                        help='duplicate-audit config (default: %(default)s)')
    parser.add_argument('--out-root', dest='out_root', type=Path, default=DEFAULT_OUT_ROOT,
                        help='artifact root; one directory per project (default: %(default)s)')
    parser.add_argument('--project-id', dest='project_id', default=None,
                        help='restrict the run to this project (required with --plan)')
    parser.add_argument('--plan', action='append', default=[],
                        help='build from this saved detector plan instead of running the '
                             'detector; repeat once per category')
    parser.add_argument('--run-stamp', dest='run_stamp', default=None,
                        help='run stamp, YYYYMMDDTHHMMSSZ (default: now)')
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    args = _build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        stamp = (validate_stamp(args.run_stamp, what='--run-stamp')
                 if args.run_stamp is not None else run_stamp())
    except (DuplicateAuditConfigError, MetricSchemaError) as exc:
        logger.error('%s', exc)
        return EXIT_BAD_INPUT
    if args.plan:
        return _run_plan_mode(args, config, stamp)
    if args.project_id is not None and args.project_id not in config.projects:
        logger.error('--project-id %r is not a configured project (%s: %s)',
                     args.project_id, args.config, ', '.join(config.projects))
        return EXIT_BAD_INPUT
    projects = (args.project_id,) if args.project_id is not None else config.projects
    return run_scheduled(config, out_root=args.out_root, stamp=stamp, projects=projects)


if __name__ == '__main__':
    sys.exit(run_cli(main))
