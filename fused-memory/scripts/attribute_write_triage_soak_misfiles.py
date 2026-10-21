#!/usr/bin/env python3
"""Where the 2026-10-19 write-triage soak's misfiles came from, per project (task 7147).

Purpose
-------
The soak census measured contested misfiles at 0.42 on the solar projects
against 0.23 on dark_factory + reify. This script splits the soak's writes
along named origin axes (band, hoist, retarget, record class, cosine to the
filed parent), so a per-group misfile rate can be read off each cell.

Every axis that depends on routing CALLS the production routing:
``write_triage.decide_band`` for the band, and ``select_judge_candidates``
over the band winner for the judge slate. Slates are fed to them as
synthetic ``MemoryResult`` records carrying ``store_score``, ``kind`` and,
for a child, ``parent_id``. A second copy of a routing rule here would drift
from the live path without anything failing.

Inputs
------
An evidence directory holding the soak seats' joined rows:

- ``duplicates/records.jsonl``: one row per write, with ``nn_triage``, the
  triage-scope max cosine, on stored and attached writes alike;
- ``team/A/attaches_axes.json``: one row per triage-made attach, with its
  20-record slate, retarget, filed parent and adjudicated verdict.

Read-only rule
--------------
The evidence is read and never written. :func:`load_soak_writes` is the only
function that reads it; everything else is pure.

Usage
-----
Run from ``fused-memory/``::

    uv run python scripts/attribute_write_triage_soak_misfiles.py \
        --evidence-dir /home/leo/src/dark-factory/data/write-triage-soak-2026-10-19 \
        --project-report solar_challenge=calibration/write_triage_calibration_report_solar_challenge.json \
        --out calibration/write_triage_solar_soak_origins.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import types
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.models.enums import SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server.grouped_read import PARENT_ID_KEY
from fused_memory.server.write_triage import (
    OUTCOME_AMENDED,
    OUTCOME_CONTESTED,
    OUTCOME_JUDGE,
    OUTCOME_RESTATED,
    OUTCOME_STORED,
    TRIAGE_OUTCOMES,
    AttachGuard,
    AttachRefusal,
    BandDecision,
    decide_band,
)
from fused_memory.server.write_triage_judge import select_judge_candidates

_SCRIPTS = Path(__file__).resolve().parent


def _load_script(path: Path, mod_name: str) -> types.ModuleType:
    """Load a ``scripts/`` sibling by path, cached in ``sys.modules``."""
    cached = sys.modules.get(mod_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


if TYPE_CHECKING:
    import calibrate_write_triage as _calibrate
    import write_triage_stats as _stats
else:
    _calibrate = _load_script(_SCRIPTS / 'calibrate_write_triage.py', 'calibrate_write_triage')
    _stats = _load_script(_SCRIPTS / 'write_triage_stats.py', 'write_triage_stats')

RECORDS_FILE = Path('duplicates') / 'records.jsonl'
ATTACHES_FILE = Path('team') / 'A' / 'attaches_axes.json'
EVIDENCE_FILES = (RECORDS_FILE, ATTACHES_FILE)

_SOAK_WINDOW = 'soak'
_STANDALONE_CLASS = 'standalone'
_CHILD_CLASS = 'child'
_ATTACH_OUTCOMES = TRIAGE_OUTCOMES - {OUTCOME_STORED}
_REPORTED_OUTCOMES = (OUTCOME_CONTESTED, OUTCOME_AMENDED, OUTCOME_RESTATED)
_COMPLETION_SOURCE = 'orchestrator_completion'
NOT_ATTACHED = 'not_attached'
COUNTERFACTUAL_UNKNOWN = 'unknown'
PROJECT_ARM = 'project'
PROJECT_CATEGORY_ARM = 'project_category'

#: The soak census's own grouping: 'solar' is the group that measured 98/232.
GROUPS: Mapping[str, frozenset[str]] = MappingProxyType({
    'solar': frozenset({'solar_challenge', 'solar_challenge_platform', 'scoping_intake'}),
    'dark_factory_reify': frozenset({'dark_factory', 'reify'}),
})


class SoakEvidenceError(ValueError):
    """An evidence row this script cannot read honestly; names the row."""


class Verdict(StrEnum):
    """A soak adjudication verdict (``correctness/adjudications.jsonl``)."""

    CORRECT = 'correct'
    CORRECT_OVERLABEL_BENIGN = 'correct_overlabel_benign'
    MISFILE = 'misfile'
    CONTESTED_MISFILE = 'contested_misfile'
    FALSE_CONTESTED = 'false_contested'
    RIGHT_SUBJECT_WRONG_LABEL_HIDES_DETAIL = 'right_subject_wrong_label_hides_detail'
    RIGHT_SUBJECT_WRONG_LABEL_HIDES_DISPUTE = 'right_subject_wrong_label_hides_dispute'
    UNCLEAR = 'unclear'
    PENDING = 'pending'


_MISFILE_VERDICTS = frozenset({Verdict.CONTESTED_MISFILE, Verdict.MISFILE})
_UNRATED_VERDICTS = frozenset({Verdict.UNCLEAR, Verdict.PENDING})


class Exclusion(StrEnum):
    """Why a soak write row is not in the analysed population."""

    NO_TRIAGE_COSINE = 'no_triage_cosine'
    CALLER_DECLARED_CHILD = 'caller_declared_child'


class Band(StrEnum):
    """The band ``decide_band`` routes a write's max cosine to."""

    RESTATED = OUTCOME_RESTATED
    JUDGE = OUTCOME_JUDGE
    STORED = OUTCOME_STORED


class Hoist(StrEnum):
    """Whether an attach landed on the record that matched, or on its parent."""

    DIRECT = 'direct'
    VIA_CHILD = 'via_child'
    NOT_ATTACHED = NOT_ATTACHED


class Retarget(StrEnum):
    """Whether the judge moved the attach off the band winner."""

    MOVED = 'moved'
    KEPT = 'kept'
    NOT_ATTACHED = NOT_ATTACHED


class RecordClass(StrEnum):
    """What kind of writer produced the attached record."""

    COMPLETION_LEDGER = 'completion_ledger'
    TASK_STAMPED = 'task_stamped'
    UNTASKED = 'untasked'


class CosParentBucket(StrEnum):
    """The attached write's cosine to its filed parent, bucketed on the routing edges."""

    BELOW_T_LOW = 'below_t_low'
    T_LOW_TO_0_65 = 't_low_to_0.65'
    FROM_0_65_TO_0_75 = '0.65_to_0.75'
    FROM_0_75_TO_T_HIGH = '0.75_to_t_high'
    AT_OR_ABOVE_T_HIGH = 'at_or_above_t_high'
    NOT_ATTACHED = NOT_ATTACHED


@dataclass(frozen=True)
class SlateEntry:
    """One retrieved candidate: its cosine to the write and the canonical it groups under."""

    id: str
    cosine: float
    canonical_id: str
    kind: str | None


@dataclass(frozen=True)
class SoakWrite:
    """One triage-routed soak write; the attach fields stay empty for a stored write."""

    memory_id: str
    project: str
    category: str
    outcome: str
    max_cosine: float
    slate: tuple[SlateEntry, ...] = ()
    parent_id: str | None = None
    cos_parent: float | None = None
    retarget_named_id: str | None = None
    verdict: Verdict | None = None
    child_source: str | None = None
    child_task_id: str | None = None

    @property
    def attached(self) -> bool:
        return self.outcome != OUTCOME_STORED

    @property
    def rated(self) -> bool:
        """Did an adjudicator reach a verdict? ``unclear`` and ``pending`` did not."""
        return self.verdict is not None and self.verdict not in _UNRATED_VERDICTS

    @property
    def misfiled(self) -> bool:
        return self.verdict in _MISFILE_VERDICTS


@dataclass(frozen=True)
class SoakPopulation:
    """The analysed writes, and the ids of every soak row left out, by reason."""

    writes: tuple[SoakWrite, ...]
    excluded: Mapping[Exclusion, tuple[str, ...]]


@dataclass(frozen=True)
class Routing:
    """The write-path routing configuration a write is classified under."""

    t_high: float | None
    t_low: float | None
    judge_candidate_count: int


# --- loading (the only I/O) -------------------------------------------------


def load_soak_writes(evidence_dir: Path) -> SoakPopulation:
    """Read the soak evidence under *evidence_dir* into a parsed population."""
    records = [
        json.loads(line)
        for line in (evidence_dir / RECORDS_FILE).read_text().splitlines()
        if line.strip()
    ]
    attaches = json.loads((evidence_dir / ATTACHES_FILE).read_text())
    return _population(records, attaches)


def _population(records: Iterable[dict], attaches: Iterable[dict]) -> SoakPopulation:
    soak = [
        record for record in records
        if record['window'] == _SOAK_WINDOW and record['class'] in (_STANDALONE_CLASS, _CHILD_CLASS)
    ]
    attach_by_child = _attaches_by_child(soak, attaches)
    writes: list[SoakWrite] = []
    excluded: dict[Exclusion, list[str]] = {reason: [] for reason in Exclusion}
    for record in soak:
        reason = _exclusion_of(record, attach_by_child)
        if reason is None:
            writes.append(_soak_write(record, attach_by_child.get(record['memory_id'])))
        else:
            excluded[reason].append(record['memory_id'])
    return SoakPopulation(
        writes=tuple(writes),
        excluded=MappingProxyType({reason: tuple(ids) for reason, ids in excluded.items()}),
    )


def _attaches_by_child(soak: list[dict], attaches: Iterable[dict]) -> dict[str, dict]:
    child_ids = {record['memory_id'] for record in soak if record['class'] == _CHILD_CLASS}
    by_child: dict[str, dict] = {}
    for attach in attaches:
        if attach['child_id'] not in child_ids:
            raise SoakEvidenceError(
                f"attach row {attach['child_id']} has no soak child record in {RECORDS_FILE}",
            )
        by_child[attach['child_id']] = attach
    return by_child


def _exclusion_of(record: dict, attach_by_child: Mapping[str, dict]) -> Exclusion | None:
    if record.get('nn_triage') is None:
        return Exclusion.NO_TRIAGE_COSINE
    if record['class'] == _CHILD_CLASS and record['memory_id'] not in attach_by_child:
        return Exclusion.CALLER_DECLARED_CHILD
    return None


def _soak_write(record: dict, attach: dict | None) -> SoakWrite:
    if attach is None:
        return SoakWrite(
            memory_id=record['memory_id'], project=record['project'],
            category=record['category'], outcome=OUTCOME_STORED,
            max_cosine=float(record['nn_triage']),
        )
    return SoakWrite(
        memory_id=record['memory_id'],
        project=record['project'],
        category=record['category'],
        outcome=_attach_outcome(attach),
        max_cosine=float(record['nn_triage']),
        slate=tuple(_slate_entry(entry) for entry in attach.get('slate20') or ()),
        parent_id=attach['parent_id'],
        cos_parent=attach['cos_parent'],
        retarget_named_id=attach.get('rt_named'),
        verdict=_verdict(attach),
        child_source=attach['child_source'],
        child_task_id=attach['child_task_id'],
    )


def _attach_outcome(attach: dict) -> str:
    outcome = attach['outcome']
    if outcome not in _ATTACH_OUTCOMES:
        raise SoakEvidenceError(f"unknown attach outcome {outcome!r} on attach row {attach['child_id']}")
    return outcome


def _verdict(attach: dict) -> Verdict | None:
    raw = attach.get('verdict')
    if raw is None:
        return None
    try:
        return Verdict(raw)
    except ValueError:
        raise SoakEvidenceError(f"unknown verdict {raw!r} on attach row {attach['child_id']}") from None


def _slate_entry(entry: dict) -> SlateEntry:
    return SlateEntry(
        id=entry['id'], cosine=float(entry['cos']), canonical_id=entry['canon'], kind=entry['kind'],
    )


# --- production routing over synthetic results ------------------------------


def _as_result(entry: SlateEntry) -> MemoryResult:
    """*entry* as the search result production routing reads its cosine and parentage from."""
    metadata: dict[str, Any] = {'store_score': entry.cosine, 'kind': entry.kind}
    if entry.canonical_id != entry.id:
        metadata[PARENT_ID_KEY] = entry.canonical_id
    return MemoryResult(id=entry.id, content='', source_store=SourceStore.mem0, metadata=metadata)


def _decide(entries: Iterable[SlateEntry], routing: Routing) -> BandDecision:
    return decide_band(
        [_as_result(entry) for entry in entries], t_high=routing.t_high, t_low=routing.t_low,
    )


# --- origin axes ------------------------------------------------------------


def band_of(write: SoakWrite, routing: Routing) -> Band:
    """The band *write*'s max cosine routes to under *routing*."""
    probe = SlateEntry(id=write.memory_id, cosine=write.max_cosine, canonical_id=write.memory_id, kind=None)
    return Band(_decide((probe,), routing).outcome)


def hoist_of(write: SoakWrite, routing: Routing) -> Hoist:
    """Did *write* attach to the record that matched it, or was it hoisted to that record's parent?

    A retarget or a restate-band attach has one known source record: the
    judge's named id, or ``decide_band``'s winner. It was hoisted exactly when
    the soak filed the write under some other id. The filed parent is read off
    the soak rather than off the slate's ``canon``, which was re-read after
    the soak and can since have changed. A judged attach went via a child when
    its parent was not on the judge's slate.
    """
    if not write.attached:
        return Hoist.NOT_ATTACHED
    if write.retarget_named_id is not None or band_of(write, routing) is Band.RESTATED:
        return _hoist(write.parent_id != _attach_source(write, routing).id)
    judge_slate = select_judge_candidates(
        [_as_result(entry) for entry in write.slate],
        routing.judge_candidate_count,
        canonical_id=_decide(write.slate, routing).canonical_id,
    )
    return _hoist(all(result.id != write.parent_id for result in judge_slate))


def _hoist(via_child: bool) -> Hoist:
    return Hoist.VIA_CHILD if via_child else Hoist.DIRECT


@dataclass(frozen=True)
class _AttachSource:
    id: str | None
    cosine: float | None


def _attach_source(write: SoakWrite, routing: Routing) -> _AttachSource:
    """The slate record *write*'s attach derived from: the judge's named id, else the band winner.

    For a judged attach that the judge did not retarget this is an
    approximation: the judge may have named a sibling of the winner that
    hoists to the same canonical.
    """
    source_id = write.retarget_named_id
    if source_id is None:
        source_id = _decide(write.slate, routing).winner_id
    cosine = next((entry.cosine for entry in write.slate if entry.id == source_id), None)
    return _AttachSource(id=source_id, cosine=cosine)


def retarget_of(write: SoakWrite, routing: Routing) -> Retarget:
    """Did the judge move *write*'s attach off the band winner?"""
    if not write.attached:
        return Retarget.NOT_ATTACHED
    return Retarget.MOVED if write.retarget_named_id is not None else Retarget.KEPT


def record_class_of(write: SoakWrite, routing: Routing) -> RecordClass:
    """Completion-ledger, task-stamped or untasked writer of *write*."""
    if write.child_source == _COMPLETION_SOURCE:
        return RecordClass.COMPLETION_LEDGER
    return RecordClass.TASK_STAMPED if write.child_task_id else RecordClass.UNTASKED


def cos_parent_bucket_of(write: SoakWrite, routing: Routing) -> CosParentBucket:
    """*write*'s cosine to its filed parent, bucketed lower-inclusive on the routing edges."""
    if write.cos_parent is None:
        return CosParentBucket.NOT_ATTACHED
    floors = (
        (routing.t_high, CosParentBucket.AT_OR_ABOVE_T_HIGH),
        (0.75, CosParentBucket.FROM_0_75_TO_T_HIGH),
        (0.65, CosParentBucket.FROM_0_65_TO_0_75),
        (routing.t_low, CosParentBucket.T_LOW_TO_0_65),
    )
    return next(
        (bucket for floor, bucket in floors if floor is not None and write.cos_parent >= floor),
        CosParentBucket.BELOW_T_LOW,
    )


Axis = Callable[[SoakWrite, Routing], StrEnum]

ORIGIN_AXES: Mapping[str, Axis] = MappingProxyType({
    'band': band_of,
    'hoist': hoist_of,
    'retarget': retarget_of,
    'record_class': record_class_of,
    'cos_parent': cos_parent_bucket_of,
})


# --- tables -----------------------------------------------------------------


def shipped_write_path() -> tuple[Routing, AttachGuard]:
    """The routing and attach guard the shipped ``write_triage`` config puts in force."""
    triage = FusedMemoryConfig().write_triage
    routing = Routing(
        t_high=triage.t_high, t_low=triage.t_low, judge_candidate_count=triage.judge_candidate_count,
    )
    guard = AttachGuard(
        min_parent_cosine=triage.min_parent_cosine, hoist_max_cosine_gap=triage.hoist_max_cosine_gap,
    )
    return routing, guard


def groups_with_projects(
    groups: Mapping[str, frozenset[str]], writes: Iterable[SoakWrite],
) -> dict[str, frozenset[str]]:
    """*groups*, then one single-project group per project in *writes*."""
    projects = sorted({write.project for write in writes})
    shadowing = [name for name in groups if name in projects and groups[name] != {name}]
    if shadowing:
        raise ValueError(f'group names {shadowing} shadow a project of the same name')
    return {**groups, **{project: frozenset({project}) for project in projects}}


def k_of_n(k: int, n: int) -> dict[str, Any]:
    """*k* of *n* with its rate and Wilson 95% interval; both None for an empty sample."""
    return {'k': k, 'n': n, 'rate': _stats.rate(k, n), 'wilson95': _stats.wilson95(k, n)}


def _misfile_k_of_n(writes: Iterable[SoakWrite]) -> dict[str, Any]:
    rated = [write for write in writes if write.rated]
    return k_of_n(sum(write.misfiled for write in rated), len(rated))


def origin_tables(
    writes: Iterable[SoakWrite], groups: Mapping[str, frozenset[str]], routing: Routing,
) -> dict[str, Any]:
    """Per axis, group and cell: attaches and misfile k/n for each attach outcome."""
    attached = [write for write in writes if write.attached]
    return {
        axis: _axis_table([(classify(write, routing).value, write) for write in attached], groups)
        for axis, classify in ORIGIN_AXES.items()
    }


def _axis_table(
    classified: list[tuple[str, SoakWrite]], groups: Mapping[str, frozenset[str]],
) -> dict[str, Any]:
    return {
        name: _cell_table([(cell, write) for cell, write in classified if write.project in projects])
        for name, projects in groups.items()
    }


def _cell_table(classified: list[tuple[str, SoakWrite]]) -> dict[str, Any]:
    by_cell: defaultdict[str, list[SoakWrite]] = defaultdict(list)
    for cell, write in classified:
        by_cell[cell].append(write)
    return {
        cell: {
            outcome: _attach_summary([write for write in by_cell[cell] if write.outcome == outcome])
            for outcome in _REPORTED_OUTCOMES
        }
        for cell in sorted(by_cell)
    }


def _attach_summary(writes: list[SoakWrite]) -> dict[str, Any]:
    return {'attaches': len(writes), 'misfile': _misfile_k_of_n(writes)}


def census_fidelity(writes: Iterable[SoakWrite], groups: Mapping[str, frozenset[str]]) -> dict[str, Any]:
    """Contested misfile over rated contested attaches per group: the census headline's figure."""
    contested = [write for write in writes if write.outcome == OUTCOME_CONTESTED]
    return {
        name: _misfile_k_of_n(write for write in contested if write.project in projects)
        for name, projects in groups.items()
    }


def guard_residual(
    writes: Iterable[SoakWrite],
    guard: AttachGuard,
    routing: Routing,
    groups: Mapping[str, frozenset[str]],
) -> dict[str, Any]:
    """What *guard* would have refused of each group's attaches, and the misfile rate left over."""
    judged = [(write, _refusals(write, guard, routing)) for write in writes if write.attached]
    return {
        name: {
            outcome: _residual_summary([
                (write, refusals) for write, refusals in judged
                if write.project in projects and write.outcome == outcome
            ])
            for outcome in _REPORTED_OUTCOMES
        }
        for name, projects in groups.items()
    }


def _refusals(write: SoakWrite, guard: AttachGuard, routing: Routing) -> frozenset[AttachRefusal]:
    source = _attach_source(write, routing)
    return guard.refusals(
        source_cosine=source.cosine,
        target_cosine=write.cos_parent,
        hoisted=write.parent_id != source.id,
        judged=band_of(write, routing) is Band.JUDGE,
    )


def _residual_summary(judged: list[tuple[SoakWrite, frozenset[AttachRefusal]]]) -> dict[str, Any]:
    removed = [(write, refusals) for write, refusals in judged if refusals]
    rated_removed = [write for write, _ in removed if write.rated]
    reasons = Counter(reason.value for _, refusals in removed for reason in refusals)
    return {
        'attaches': len(judged),
        'removed': len(removed),
        'removed_by_reason': dict(sorted(reasons.items())),
        'rated_misfiles_removed': sum(write.misfiled for write in rated_removed),
        'rated_non_misfiles_removed': sum(not write.misfiled for write in rated_removed),
        'misfile_before': _misfile_k_of_n(write for write, _ in judged),
        'misfile_after': _misfile_k_of_n(write for write, refusals in judged if not refusals),
    }


# --- per-project band arms and the replay -----------------------------------


class EdgeSource(StrEnum):
    """Where a replay arm's band edge came from."""

    REPORT = 'report'
    POOLED_FALLBACK = 'pooled_fallback'


@dataclass(frozen=True)
class ArmEdge:
    value: float | None
    source: EdgeSource

    def as_json(self) -> dict[str, Any]:
        return {'value': self.value, 'source': self.source.value}


@dataclass(frozen=True)
class ArmEdges:
    t_high: ArmEdge
    t_low: ArmEdge

    def routing(self, judge_candidate_count: int) -> Routing:
        return Routing(
            t_high=self.t_high.value, t_low=self.t_low.value,
            judge_candidate_count=judge_candidate_count,
        )

    def as_json(self) -> dict[str, Any]:
        return {'t_high': self.t_high.as_json(), 't_low': self.t_low.as_json()}


@dataclass(frozen=True)
class BandArm:
    """One replay arm: band edges per write category, *default* for a category it does not name."""

    default: ArmEdges
    by_category: Mapping[str, ArmEdges]

    def edges_for(self, category: str) -> ArmEdges:
        return self.by_category.get(category, self.default)

    def as_json(self) -> dict[str, Any]:
        return {
            'default': self.default.as_json(),
            'by_category': {category: edges.as_json() for category, edges in self.by_category.items()},
        }


def project_band_arms(report: Mapping[str, Any], pooled: Routing) -> dict[str, BandArm]:
    """The project-level and per-category replay arms a calibration *report* proposes.

    An edge the report did not derive takes the *pooled* write-path edge: a
    refusal means no per-project evidence, never 'store everything'.
    """
    per_category = report.get('per_category') or {}
    return {
        PROJECT_ARM: BandArm(
            default=_resolve_edges(report.get('chosen_t_high'), report.get('chosen_t_low'), pooled),
            by_category=MappingProxyType({}),
        ),
        PROJECT_CATEGORY_ARM: BandArm(
            default=_resolve_edges(None, None, pooled),
            by_category=MappingProxyType({
                category: _resolve_edges(outcome.get('t_high'), outcome.get('t_low'), pooled)
                for category, outcome in per_category.items()
            }),
        ),
    }


def _resolve_edges(t_high: float | None, t_low: float | None, pooled: Routing) -> ArmEdges:
    return ArmEdges(t_high=_resolve_edge(t_high, pooled.t_high), t_low=_resolve_edge(t_low, pooled.t_low))


def _resolve_edge(derived: float | None, pooled: float | None) -> ArmEdge:
    if derived is None:
        return ArmEdge(value=pooled, source=EdgeSource.POOLED_FALLBACK)
    return ArmEdge(value=float(derived), source=EdgeSource.REPORT)


def replay_bands(
    writes: Iterable[SoakWrite], pooled: Routing, arms: Mapping[str, Mapping[str, BandArm]],
) -> dict[str, Any]:
    """Per project with arms, and per arm: how each write's band moves from *pooled* to the arm."""
    writes = list(writes)
    return {
        project: {
            name: _replay_arm([write for write in writes if write.project == project], pooled, arm)
            for name, arm in project_arms.items()
        }
        for project, project_arms in arms.items()
    }


_Move = tuple[Band, Band, SoakWrite]


def _replay_arm(writes: list[SoakWrite], pooled: Routing, arm: BandArm) -> dict[str, Any]:
    moves: list[_Move] = [
        (
            band_of(write, pooled),
            band_of(write, arm.edges_for(write.category).routing(pooled.judge_candidate_count)),
            write,
        )
        for write in writes
    ]
    return {
        'edges': arm.as_json(),
        'matrix': _transition_matrix(moves),
        'into_stored': _into_stored([
            write for source, target, write in moves if source != target and target is Band.STORED
        ]),
        'other_moves': _other_moves(moves),
    }


def _transition_matrix(moves: list[_Move]) -> dict[str, dict[str, int]]:
    counts = Counter((source, target) for source, target, _ in moves)
    return {source.value: {target.value: counts[(source, target)] for target in Band} for source in Band}


def _into_stored(movers: list[SoakWrite]) -> dict[str, int]:
    rated = [write for write in movers if write.rated]
    return {
        'movers': len(movers),
        'rated_misfiles_removed': sum(write.misfiled for write in rated),
        'rated_non_misfiles_lost': sum(not write.misfiled for write in rated),
        'unrated': len(movers) - len(rated),
    }


def _other_moves(moves: list[_Move]) -> list[dict[str, Any]]:
    by_move: defaultdict[tuple[Band, Band], list[SoakWrite]] = defaultdict(list)
    for source, target, write in moves:
        if source != target and target is not Band.STORED:
            by_move[(source, target)].append(write)
    return [
        _move_summary(source, target, by_move[(source, target)])
        for source in Band for target in Band if (source, target) in by_move
    ]


def _move_summary(source: Band, target: Band, movers: list[SoakWrite]) -> dict[str, Any]:
    verdicts = [write.verdict.value for write in movers if write.rated and write.verdict is not None]
    return {
        'from': source.value,
        'to': target.value,
        'movers': len(movers),
        'soak_outcomes': dict(sorted(Counter(write.outcome for write in movers).items())),
        'rated_verdicts': dict(sorted(Counter(verdicts).items())),
        'unrated': len(movers) - len(verdicts),
        'counterfactual_verdict': COUNTERFACTUAL_UNKNOWN,
    }


# --- the origins document and its CLI ---------------------------------------


def analyse(
    population: SoakPopulation,
    routing: Routing,
    guard: AttachGuard,
    reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Every table the origins document carries, from the parsed evidence alone."""
    groups = groups_with_projects(GROUPS, population.writes)
    arms = {project: project_band_arms(report, routing) for project, report in reports.items()}
    return {
        'census_fidelity': census_fidelity(population.writes, groups),
        'origin_tables': origin_tables(population.writes, groups, routing),
        'guard_residual': guard_residual(population.writes, guard, routing, groups),
        'replay': replay_bands(population.writes, routing, arms),
    }


def _provenance(
    evidence_dir: Path,
    report_paths: Mapping[str, Path],
    population: SoakPopulation,
    routing: Routing,
    guard: AttachGuard,
) -> dict[str, Any]:
    return {
        'evidence_dir': str(evidence_dir),
        'evidence_files': {
            str(name): hashlib.sha256((evidence_dir / name).read_bytes()).hexdigest()
            for name in EVIDENCE_FILES
        },
        'pooled_bands': {'t_high': routing.t_high, 't_low': routing.t_low},
        'judge_candidate_count': routing.judge_candidate_count,
        'attach_guard': {
            'min_parent_cosine': guard.min_parent_cosine,
            'hoist_max_cosine_gap': guard.hoist_max_cosine_gap,
        },
        'project_reports': {
            project: _calibrate.package_relative(path) for project, path in report_paths.items()
        },
        'groups': {name: sorted(projects) for name, projects in GROUPS.items()},
        'population': {
            'writes': len(population.writes),
            'attached': sum(write.attached for write in population.writes),
            'excluded': {
                reason.value: {'count': len(ids), 'memory_ids': list(ids)}
                for reason, ids in population.excluded.items()
            },
        },
    }


def _project_report_arg(value: str) -> tuple[str, Path]:
    project, separator, path = value.partition('=')
    if not (separator and project and path):
        raise argparse.ArgumentTypeError(f'expected <project>=<path>, got {value!r}')
    return project, Path(path)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Attribute the write-triage soak misfiles per project.')
    parser.add_argument('--evidence-dir', type=Path, required=True)
    parser.add_argument(
        '--project-report', type=_project_report_arg, action='append', default=[],
        metavar='PROJECT=PATH',
    )
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    projects = [project for project, _ in args.project_report]
    if len(set(projects)) != len(projects):
        parser.error(f'--project-report names a project more than once: {projects}')
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report_paths = dict(args.project_report)
    population = load_soak_writes(args.evidence_dir)
    routing, guard = shipped_write_path()
    reports = {project: json.loads(path.read_text()) for project, path in report_paths.items()}
    document = {
        'provenance': _provenance(args.evidence_dir, report_paths, population, routing, guard),
        **analyse(population, routing, guard, reports),
    }
    args.out.write_text(json.dumps(document, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
