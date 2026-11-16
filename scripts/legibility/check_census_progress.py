#!/usr/bin/env python3
"""Predicate: did the censuses the census trigger FIRED for <project_id>
actually advance census-state? Exit 0 = yes, or too few fires to say;
non-zero = escalate.

Usage: check_census_progress.py <project_id> [max_unadvanced_fires]

  max_unadvanced_fires defaults to
  trickle_state.DEFAULT_MAX_UNADVANCED_CENSUS_FIRES.

It reads the trickle run-state record's ``census_fire_streak`` (written by
``trickle_state.record_run``), then census-state.json LIVE at the path that
streak recorded, so a census that advances clears the verdict at once.
"Has not moved" is exact equality with the ``last_census_at`` the trigger
evaluated against (``trickle_state.CensusObservation``). Never invokes git
or any subprocess.

Division of labour: ``check_trickle_liveness.sh`` owns "did the unit run",
``check_trickle_progress.py`` owns "did signal flow" and the record's
freshness, so this probe does not re-check freshness.
"""
from __future__ import annotations

import sys
from pathlib import Path

# EXEC'd directly as a bound predicate with bare python3 (no venv, no
# package context): see check_trickle_progress.py's identical shim.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import state_docs
import trickle_state

USAGE = 'usage: check_census_progress.py <project_id> [max_unadvanced_fires]'


def _parse_args(args: list[str]) -> tuple[str, int] | None:
    if len(args) not in (1, 2):
        return None
    try:
        threshold = (
            int(args[1]) if len(args) == 2
            else trickle_state.DEFAULT_MAX_UNADVANCED_CENSUS_FIRES
        )
    except ValueError:
        return None
    if threshold < 1:
        return None
    return args[0], threshold


def _fail(message: str) -> int:
    print(f'ERROR: {message}', file=sys.stderr)
    return 1


def _anchor_text(anchor: object) -> str:
    return 'never censused' if anchor is None else str(anchor)


def _unadvanced(
    project_id: str, streak: trickle_state.CensusFireStreak, threshold: int,
) -> int:
    return _fail(
        f'the census trigger for {project_id} FIRED on {streak.fires} '
        f'consecutive recorded trickle nights since {streak.since} '
        f'(threshold {threshold}), and last_census_at has not moved from '
        f'{_anchor_text(streak.last_census_at)} in '
        f'{streak.census_state_path}. Every FIRE launched, or tried to '
        f'launch, a paid census that did not advance census-state. A census '
        f'deferred for usage headroom counts toward this streak, so check '
        f"the census's own 'legibility census deferred' info escalations "
        f'first. Then read journalctl --user -u '
        f'legibility-trickle@{project_id} --since {streak.since} for the '
        f"census child's lines: 'census: deferred' is a headroom deferral; "
        f"'census: commit did not land' is the unlanded shape (task 5780: "
        f'census-state rolled back to HEAD, census exited 1); '
        f"'census: NO-FIRE for <project>' means census.py re-decided the "
        f'trigger and declined, and naming another project there is the '
        f'wrong-target shape of task 3269.'
    )


def _assess_census_state(
    project_id: str, streak: trickle_state.CensusFireStreak, threshold: int,
) -> int:
    path = Path(streak.census_state_path)
    status, current = trickle_state.read_last_census_at(path)

    if status == 'malformed':
        return _fail(
            f'census-state unreadable at {path}; cannot tell whether the '
            f'{streak.fires} censuses fired for {project_id} since '
            f'{streak.since} advanced it.'
        )
    if status == 'missing' and streak.last_census_at is not None:
        return _fail(
            f'census-state vanished from {path} since the census-fire streak '
            f'for {project_id} began at {streak.last_census_at} '
            f'({streak.fires} consecutive fires since {streak.since}, '
            f'threshold {threshold}); check that checkout.'
        )
    if current != streak.last_census_at:
        print(
            f'OK: census for {project_id} advanced from '
            f'{_anchor_text(streak.last_census_at)} to {current} since the '
            f'fire streak began {streak.since}; the next NO-FIRE night '
            f'resets it.'
        )
        return 0
    return _unadvanced(project_id, streak, threshold)


def main(argv: list[str]) -> int:
    parsed = _parse_args(argv[1:])
    if parsed is None:
        return _fail(
            f'{USAGE}\nmax_unadvanced_fires must be an integer >= 1'
        )
    project_id, threshold = parsed

    path = trickle_state.trickle_state_path(project_id)
    match trickle_state.load_state(path):
        case state_docs.Missing():
            return _fail(
                f'legibility trickle for {project_id} has never recorded a run '
                f'(no state file at {path}), so census progress cannot be '
                f"assessed; the recorder's health is check_trickle_progress.py's "
                f'verdict.'
            )
        case state_docs.Malformed(reason=reason):
            return _fail(
                f'legibility trickle state for {project_id} is unreadable/corrupt '
                f'at {path}: {reason}. Census progress cannot be assessed; see '
                f'check_trickle_progress.py.'
            )
        case state_docs.Ok(value=state):
            return _assess_streak(project_id, state, threshold)


def _assess_streak(
    project_id: str, state: trickle_state.TrickleState, threshold: int,
) -> int:
    streak = state.census_fire_streak
    if streak is None or streak.fires < threshold:
        print(
            f'OK: census trigger for {project_id}: census_fired='
            f'{state.census_fired}, '
            f'fires={0 if streak is None else streak.fires} '
            f'(threshold {threshold}).'
        )
        return 0
    return _assess_census_state(project_id, streak, threshold)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
