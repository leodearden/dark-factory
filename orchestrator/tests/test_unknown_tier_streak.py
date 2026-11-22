"""Ten consecutive unknown-tier picks at one gate file one L1: the priority lookup is broken (task 9933)."""
from __future__ import annotations

from pathlib import Path

import pytest
from escalation.queue import EscalationQueue

from orchestrator.sentinel_l1 import SentinelL1, file_sentinel_l1, resolve_sentinel_l1
from orchestrator.task_priority import TaskPriority
from orchestrator.unknown_tier_streak import (
    UNKNOWN_TIER_STREAK_LIMIT,
    PriorityGate,
    UnknownTierStreak,
)

_UNKNOWN = TaskPriority()
_UNRECOGNISED = TaskPriority(tier='urgent')
_KNOWN = TaskPriority(tier='high', pinned=False, base_tier='high')


class _Recorder:
    def __init__(self) -> None:
        self.filed: list[SentinelL1] = []
        self.resolved: list[str] = []

    def escalate(self, l1: SentinelL1) -> None:
        self.filed.append(l1)

    def resolve(self, sentinel: str, resolution: str) -> None:
        self.resolved.append(sentinel)

    def streak(
        self, gate: PriorityGate = PriorityGate.MERGE_POOL, *, lookup_wired: bool = True,
    ) -> UnknownTierStreak:
        return UnknownTierStreak(
            gate, lookup_wired=lookup_wired, escalate=self.escalate, resolve=self.resolve,
        )


class _RaisingRecorder(_Recorder):
    def escalate(self, l1: SentinelL1) -> None:
        super().escalate(l1)
        raise RuntimeError('escalation queue is down')

    def resolve(self, sentinel: str, resolution: str) -> None:
        super().resolve(sentinel, resolution)
        raise RuntimeError('escalation queue is down')


def _pick(streak: UnknownTierStreak, priority: TaskPriority, times: int) -> None:
    for _ in range(times):
        streak.note_pick(priority)


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
    assert l1.sentinel == '__unknown_tier_merge_pool__'
    assert l1.category == 'infra_issue'


def test_a_continuing_streak_files_again_only_at_each_multiple_of_the_limit() -> None:
    recorder = _Recorder()
    streak = recorder.streak()

    _pick(streak, _UNKNOWN, 3 * UNKNOWN_TIER_STREAK_LIMIT - 1)

    assert len(recorder.filed) == 2


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


def test_a_fresh_streak_resolves_what_a_previous_process_may_have_left_open() -> None:
    recorder = _Recorder()
    streak = recorder.streak(PriorityGate.VERIFY_ADMISSION)

    _pick(streak, _KNOWN, 2)

    assert recorder.resolved == ['__unknown_tier_verify_admission__']


def test_without_a_lookup_nothing_is_ever_filed() -> None:
    recorder = _Recorder()
    streak = recorder.streak(lookup_wired=False)

    _pick(streak, _UNKNOWN, 5 * UNKNOWN_TIER_STREAK_LIMIT)

    assert recorder.filed == []


def test_a_raising_escalate_or_resolve_does_not_reach_the_pick() -> None:
    recorder = _RaisingRecorder()
    streak = recorder.streak()

    _pick(streak, _UNKNOWN, UNKNOWN_TIER_STREAK_LIMIT)
    streak.note_pick(_KNOWN)

    assert (len(recorder.filed), len(recorder.resolved)) == (1, 1)


@pytest.fixture
def queue(tmp_path: Path) -> EscalationQueue:
    return EscalationQueue(tmp_path / 'esc')


def test_a_long_streak_leaves_one_pending_l1(queue: EscalationQueue) -> None:
    filed: list[str] = []

    def escalate(l1: SentinelL1) -> None:
        filed.append(l1.sentinel)
        file_sentinel_l1(queue, l1)

    streak = UnknownTierStreak(
        PriorityGate.MERGE_POOL, lookup_wired=True, escalate=escalate,
        resolve=lambda sentinel, resolution: resolve_sentinel_l1(queue, sentinel, resolution),
    )

    _pick(streak, _UNKNOWN, 25)

    assert len(queue.get_by_task(filed[0], status='pending')) == 1
