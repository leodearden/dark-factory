"""Ten consecutive unknown-tier picks at one gate file one L1: the priority lookup is broken (task 9933)."""
from __future__ import annotations

from pathlib import Path

import pytest
from escalation.queue import EscalationQueue

from orchestrator.sentinel_l1 import SentinelL1, file_sentinel_l1, resolve_sentinel_l1
from orchestrator.task_priority import TaskPriority, TaskPriorityLookup
from orchestrator.unknown_tier_streak import (
    UNKNOWN_TIER_STREAK_LIMIT,
    PriorityGate,
    UnknownTierStreak,
)

_UNKNOWN = TaskPriority()
_UNRECOGNISED = TaskPriority(tier='urgent')
_KNOWN = TaskPriority(tier='high', pinned=False, base_tier='high')


def _wired(task_id: str) -> TaskPriority:
    return _UNKNOWN


class _Recorder:
    def __init__(self) -> None:
        self.filed: list[SentinelL1] = []
        self.resolved: list[str] = []

    def escalate(self, l1: SentinelL1) -> None:
        self.filed.append(l1)

    def resolve(self, sentinel: str, resolution: str) -> None:
        self.resolved.append(sentinel)

    def streak(
        self, gate: PriorityGate = PriorityGate.MERGE_POOL,
        lookup: TaskPriorityLookup | None = _wired,
    ) -> UnknownTierStreak:
        return UnknownTierStreak(gate, lookup, escalate=self.escalate, resolve=self.resolve)


def _pick(streak: UnknownTierStreak, priority: TaskPriority, times: int) -> None:
    for _ in range(times):
        streak.note_pick(priority)


def _raise(*args: object) -> None:
    raise RuntimeError('escalation queue is down')


def test_nine_unknown_picks_file_nothing() -> None:
    recorder = _Recorder()
    streak = recorder.streak()

    _pick(streak, _UNKNOWN, 5)
    _pick(streak, _UNRECOGNISED, UNKNOWN_TIER_STREAK_LIMIT - 6)

    assert recorder.filed == []


def test_the_tenth_unknown_pick_files_one_infra_l1_for_the_gate() -> None:
    recorder = _Recorder()
    streak = recorder.streak()

    _pick(streak, _UNKNOWN, UNKNOWN_TIER_STREAK_LIMIT)

    (l1,) = recorder.filed
    assert 'merge_pool' in l1.sentinel
    assert l1.category == 'infra_issue'


def test_each_gate_files_under_its_own_sentinel() -> None:
    recorder = _Recorder()

    _pick(recorder.streak(PriorityGate.MERGE_POOL), _UNKNOWN, UNKNOWN_TIER_STREAK_LIMIT)
    _pick(recorder.streak(PriorityGate.VERIFY_ADMISSION), _UNKNOWN, UNKNOWN_TIER_STREAK_LIMIT)

    pool, admission = recorder.filed
    assert pool.sentinel != admission.sentinel


def test_a_known_pick_resets_the_count() -> None:
    recorder = _Recorder()
    streak = recorder.streak()

    _pick(streak, _UNKNOWN, UNKNOWN_TIER_STREAK_LIMIT - 1)
    streak.note_pick(_KNOWN)
    _pick(streak, _UNKNOWN, UNKNOWN_TIER_STREAK_LIMIT - 1)

    assert recorder.filed == []


def test_the_first_known_pick_after_a_filing_resolves_it_once() -> None:
    recorder = _Recorder()
    streak = recorder.streak()
    _pick(streak, _UNKNOWN, UNKNOWN_TIER_STREAK_LIMIT)
    (l1,) = recorder.filed

    _pick(streak, _KNOWN, 2)

    assert recorder.resolved == [l1.sentinel]


def test_without_a_lookup_nothing_is_ever_filed() -> None:
    recorder = _Recorder()
    streak = recorder.streak(lookup=None)

    _pick(streak, _UNKNOWN, 5 * UNKNOWN_TIER_STREAK_LIMIT)

    assert recorder.filed == []


def test_a_raising_escalate_or_resolve_does_not_reach_the_pick() -> None:
    streak = UnknownTierStreak(PriorityGate.MERGE_POOL, _wired, escalate=_raise, resolve=_raise)

    _pick(streak, _UNKNOWN, UNKNOWN_TIER_STREAK_LIMIT + 1)
    streak.note_pick(_KNOWN)


@pytest.fixture
def queue(tmp_path: Path) -> EscalationQueue:
    return EscalationQueue(tmp_path / 'esc')


def test_a_long_streak_leaves_one_pending_l1(queue: EscalationQueue) -> None:
    filed: list[str] = []

    def escalate(l1: SentinelL1) -> None:
        filed.append(l1.sentinel)
        file_sentinel_l1(queue, l1)

    streak = UnknownTierStreak(
        PriorityGate.MERGE_POOL, _wired, escalate=escalate,
        resolve=lambda sentinel, resolution: resolve_sentinel_l1(queue, sentinel, resolution),
    )

    _pick(streak, _UNKNOWN, 25)

    assert len(queue.get_by_task(filed[0], status='pending')) == 1
