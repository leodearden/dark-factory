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
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from shared.memory_eval_metrics import stamp_datetime

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
    """Assemble one project's report from its per-category detector plans.

    With *previous* (that project's prior report) the report is also diffed
    against it; see ``diff_against_previous``.
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
        'child_counts_known': all('child_counts' in plan for plan in bare_plans),
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
    """Created before the previous run and newly detected: not regrowth."""
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
    """``{current index: [(overlap, previous key)]}``, within a category only, largest first."""
    by_category: dict[Any, list[Mapping[str, Any]]] = {}
    for cluster in previous:
        by_category.setdefault(cluster['category'], []).append(cluster)
    found: dict[int, list[tuple[int, str]]] = {}
    for index, cluster in enumerate(current):
        members = set(cluster['member_ids'])
        hits = [
            (len(members & set(prior['member_ids'])), prior['cluster_key'])
            for prior in by_category.get(cluster['category'], ())
        ]
        found[index] = sorted(((n, key) for n, key in hits if n), key=lambda t: (-t[0], t[1]))
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


def _created_moment(created_at: Any) -> datetime | None:
    if not isinstance(created_at, str) or not created_at:
        return None
    try:
        moment = datetime.fromisoformat(created_at)
    except ValueError:
        return None
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def classify_added(
    added_ids: Iterable[Any],
    members: Sequence[Mapping[str, Any]] | None,
    previous_stamp: datetime,
) -> dict[str, int]:
    """Count *added_ids* by age against the previous run's stamp."""
    created = {m.get('id'): m.get('created_at') for m in members or ()}
    counts = dict.fromkeys((age.value for age in MemberAge), 0)
    for member_id in added_ids:
        moment = _created_moment(created.get(member_id))
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
    overlapped = list(lineage.overlapped_previous_keys)
    return {
        'status': lineage.status.value,
        'previous_key': lineage.previous_key,
        'split_from': lineage.split_from,
        'merged_previous_keys': overlapped if len(overlapped) > 1 else [],
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


def child_deltas(
    current: Mapping[str, Sequence[Mapping[str, Any]]], previous: Mapping[str, Any] | None,
) -> dict[str, list[dict[str, Any]]]:
    """Each ranking row with its previous count and delta.

    Both are ``None`` when there is no previous report or it could not count
    children (a legacy plan): an unknown count is not a zero.
    """
    known = previous is not None and previous.get('child_counts_known') is True
    previous_rankings = previous['child_rankings'] if known and previous is not None else {}
    prior = {
        (kind, row['parent_id']): row['count']
        for kind, rows in previous_rankings.items()
        for row in rows
    }
    return {
        kind: [
            {
                **row,
                'previous_count': prior.get((kind, row['parent_id']), 0) if known else None,
                'delta': row['count'] - prior.get((kind, row['parent_id']), 0) if known else None,
            }
            for row in rows
        ]
        for kind, rows in current.items()
    }


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
    """The report keys the diff adds or extends: clusters, categories, child_rankings, regrowth."""
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
    return {
        'clusters': clusters,
        'categories': {
            name: {**entry, 'incomparable_reasons': category_comparability(
                entry, previous_categories.get(name),
            )}
            for name, entry in inventory['categories'].items()
        },
        'child_rankings': child_deltas(inventory['child_rankings'], previous),
        'regrowth': {
            'previous_run_stamp': previous['run_stamp'] if previous is not None else None,
            'elapsed_days': elapsed_days,
            'previous_overdue': (
                elapsed_days > 2 * config.cadence.resweep_interval_days
                if elapsed_days is not None else None
            ),
            'dissolved': _dissolved(lineages, previous_clusters or (), inventory['categories']),
            **_regrowth_totals(clusters, elapsed_days),
        },
    }
