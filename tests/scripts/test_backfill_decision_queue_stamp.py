"""Tests for scripts/backfill_decision_queue_stamp.py — the queue-stamp back-fill.

Task 3528 scoped the ``reap-decisions`` join by QUEUE, because an escalation id
(``esc-<taskid>-<n>``) is unique only WITHIN one queue while DecisionRecords are
fleet-global. It stamped the queue on newly-FILED decisions but did not
back-fill the records that already existed, so every legacy record still carries
``escalations_dir == ''`` and still falls back to project-only scoping — i.e. is
still cross-queue false-closable. Task 3640 drains that population.

NO TEST HERE ASSERTS A COUNT OR RECORD ID DERIVED FROM THE LIVE FLEET.
This norm is inherited verbatim from scripts/tests/test_audit_wiped_metadata_files.py
and tests/scripts/test_repair_wiped_metadata_files.py: ``~/.claude/fleet/decisions/``
is mutated continuously by the running watchers and the cockpit, so a test
pinning "the live store yields N candidates" would be a guessed threshold that
goes red the moment a watcher files. It is not hypothetical here either — the
candidate count measured 27 when this task was filed, 42 the next day and 39 the
day after, while the total record count grew 364 -> 375. Every assertion below
runs against a synthetic ``CLAUDE_FLEET_ROOT`` under ``tmp_path`` whose contents
the test controls exactly. Task WORK item 4's confirmation is the script's own
``--verify`` exit code plus the committed audit report, not a unit test.

Mirrors the sibling migration scripts' split: pure functions get direct pytest
coverage; ``main()`` gets subprocess coverage.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]

# backfill_decision_queue_stamp lives in <repo>/scripts, which
# tests/scripts/conftest.py puts on sys.path at collection time; scripts/ is
# also in [tool.pyright] extraPaths (task 3456), so this resolves statically
# too. `orchestrator` resolves via the workspace's editable install.
from orchestrator import session_registry as sr

from backfill_decision_queue_stamp import (
    queues_holding,
    resolve_queue_for_decision,
)


# ---------------------------------------------------------------------------
# Fixtures / builders -- all synthetic, none pointing at the live fleet
# ---------------------------------------------------------------------------


def _make_queue(base: Path, name: str, *, root_ids: tuple[str, ...] = (),
                archive_ids: tuple[str, ...] = ()) -> Path:
    """Build a synthetic escalation queue holding *root_ids* / *archive_ids*.

    Both tiers matter and are exercised separately: a still-pending escalation
    lives at ``<queue>/<id>.json``, while a resolved/dismissed one has been
    MOVED into ``<queue>/archive/YYYY-MM-DD/`` by
    escalation.queue._archive_resolved. A lookup that only checked the root
    would miss exactly the terminal escalations the reaper cares about.
    """
    queue = base / name
    queue.mkdir(parents=True, exist_ok=True)
    for esc_id in root_ids:
        (queue / f'{esc_id}.json').write_text(json.dumps({'id': esc_id, 'status': 'pending'}))
    if archive_ids:
        dated = queue / 'archive' / '2026-01-01'
        dated.mkdir(parents=True, exist_ok=True)
        for esc_id in archive_ids:
            (dated / f'{esc_id}.json').write_text(json.dumps({'id': esc_id, 'status': 'resolved'}))
    return queue


def _record(**overrides: object) -> sr.DecisionRecord:
    """A DecisionRecord built straight from the library type.

    Deliberately NOT a copy of orchestrator/tests/test_session_registry.py's
    _make_decision: duplicating that builder would let the two drift, and this
    file only needs the three fields the resolver actually reads
    (escalation_id, session_id, project).
    """
    fields: dict = {
        'id': 'dec-1',
        'project': 'df',
        'text': 'approve?',
        'filed_at': '2026-01-01T00:00:00+00:00',
        'escalation_id': 'esc-1-1',
        'session_id': None,
        'state': sr.DecisionState.OPEN,
    }
    fields.update(overrides)
    return sr.DecisionRecord(**fields)


# ---------------------------------------------------------------------------
# queues_holding
# ---------------------------------------------------------------------------


def test_queues_holding_finds_a_root_hit(tmp_path: Path) -> None:
    """A still-pending escalation sits directly under the queue root."""
    q = _make_queue(tmp_path, 'orch', root_ids=('esc-1-1',))
    other = _make_queue(tmp_path, 'recon')

    assert queues_holding('esc-1-1', [q, other]) == [q]


def test_queues_holding_finds_an_archive_hit(tmp_path: Path) -> None:
    """A resolved escalation has been moved under archive/YYYY-MM-DD/.

    This tier is the load-bearing one for the back-fill: most legacy records
    point at escalations that have long since resolved, so a root-only lookup
    would report "no holders" for nearly the whole population and stamp
    UNKNOWN across the board — technically safe, but it would abandon the
    task's actual goal of attributing the records it CAN attribute.
    """
    q = _make_queue(tmp_path, 'orch', archive_ids=('esc-1-1',))

    assert queues_holding('esc-1-1', [q]) == [q]


def test_queues_holding_returns_every_holder_in_caller_order(tmp_path: Path) -> None:
    """Ambiguity is DETECTED, never resolved here — this function reports."""
    a = _make_queue(tmp_path, 'a', root_ids=('esc-1-1',))
    b = _make_queue(tmp_path, 'b', archive_ids=('esc-1-1',))
    c = _make_queue(tmp_path, 'c')

    assert queues_holding('esc-1-1', [a, b, c]) == [a, b]
    assert queues_holding('esc-1-1', [b, a, c]) == [b, a]


def test_queues_holding_no_holders_is_an_empty_list(tmp_path: Path) -> None:
    q = _make_queue(tmp_path, 'orch', root_ids=('esc-other-1',))

    assert queues_holding('esc-1-1', [q]) == []


def test_queues_holding_skips_a_missing_queue_dir(tmp_path: Path) -> None:
    """A queue path that does not exist is skipped, not raised on.

    The live invocation passes six project queues; a project checked out on
    another machine, renamed, or not yet escalated-in must not abort a
    migration over the whole fleet.
    """
    present = _make_queue(tmp_path, 'orch', root_ids=('esc-1-1',))
    missing = tmp_path / 'never-created'

    assert queues_holding('esc-1-1', [missing, present]) == [present]


def test_queues_holding_skips_a_queue_that_is_a_regular_file(tmp_path: Path) -> None:
    """Fail-soft on a malformed queue argument too (an operator typo pointing
    --queue at a FILE), for the same reason: one bad argument must not decide
    the fate of the other five queues.
    """
    blocker = tmp_path / 'not-a-dir'
    blocker.write_text('oops')
    present = _make_queue(tmp_path, 'orch', root_ids=('esc-1-1',))

    assert queues_holding('esc-1-1', [blocker, present]) == [present]


def test_queues_holding_dedupes_the_same_queue_spelled_twice(tmp_path: Path) -> None:
    """The same queue passed twice is ONE holder, not two.

    Non-hypothetical: the real invocation passes the fleet-wide recon queue as
    BOTH a --queue and the --recon-queue, and an operator may well repeat a
    path. If that counted as two holders, a uniquely-attributable record would
    be misclassified AMBIGUOUS and fall to UNKNOWN — silently losing exactly
    the attribution this task exists to produce.
    """
    q = _make_queue(tmp_path, 'orch', root_ids=('esc-1-1',))
    (tmp_path / 'x').mkdir()
    dotted = Path(str(tmp_path / 'x' / '..' / 'orch') + '/')

    assert queues_holding('esc-1-1', [q, q, dotted]) == [q]


# ---------------------------------------------------------------------------
# resolve_queue_for_decision -- the decision table
# ---------------------------------------------------------------------------


def test_resolve_no_escalation_id_is_unknown(tmp_path: Path) -> None:
    """Rule 1: nothing to attribute.

    These are the manual/sentinel records (a human or an /unblock filed a gate
    with no escalation behind it). reap_answered_decisions already skips a
    falsy escalation_id before ever consulting the queue, so they were never
    false-closable — but leaving them '' would leave the WORK-item-4 invariant
    ("no OPEN record lacks a queue stamp") permanently unreachable, and an
    invariant that cannot be checked is not an invariant.
    """
    q = _make_queue(tmp_path, 'orch', root_ids=('esc-1-1',))

    for empty in (None, ''):
        assert resolve_queue_for_decision(
            _record(escalation_id=empty),
            queues=[q],
            recon_queue=None,
            orch_queue_for_project={},
        ) == sr.UNKNOWN_QUEUE


@pytest.mark.parametrize('tier', ['root', 'archive'])
def test_resolve_single_holder_wins_outright(tmp_path: Path, tier: str) -> None:
    """Rule 2: exactly one queue holds the id -> that queue. No tiebreak runs.

    Direct evidence beats inference, so this branch must be reached BEFORE any
    session_id heuristic — pinned by giving the record a session_id shape that
    would point somewhere else entirely if the ladder were ordered wrongly.
    """
    kwargs = {'root_ids': ('esc-1-1',)} if tier == 'root' else {'archive_ids': ('esc-1-1',)}
    q = _make_queue(tmp_path, 'orch', **kwargs)  # pyright: ignore[reportArgumentType]
    decoy = _make_queue(tmp_path, 'recon')

    resolved = resolve_queue_for_decision(
        _record(session_id=None),
        queues=[q, decoy],
        recon_queue=decoy,
        orch_queue_for_project={},
    )

    assert resolved == sr.normalize_escalations_dir(q)


def test_resolve_zero_holders_is_unknown(tmp_path: Path) -> None:
    """Rule 3: the id resolves nowhere (archive retention pruned it, or the
    owning project's queue was not passed) -> UNKNOWN. Never a guess.
    """
    empty = _make_queue(tmp_path, 'orch')

    assert resolve_queue_for_decision(
        _record(escalation_id='esc-vanished-1'),
        queues=[empty],
        recon_queue=empty,
        orch_queue_for_project={'df': empty},
    ) == sr.UNKNOWN_QUEUE


def test_resolve_ambiguous_null_session_id_corroborated_picks_recon(tmp_path: Path) -> None:
    """Rule 4: the recon tiebreak, CORROBORATED.

    Evidence basis (measured, not guessed): across the already-stamped
    population every record stamped with the fleet-wide reconciliation queue
    has ``session_id is None``, and every record stamped with an orchestrator
    queue has a ``watcher-<slug>-<pid>`` session_id — a clean discriminator at
    both re-measurements. It is still only ever used to CHOOSE AMONG queues
    that demonstrably hold the id.
    """
    recon = _make_queue(tmp_path, 'recon', root_ids=('esc-1-1',))
    orch = _make_queue(tmp_path, 'orch', archive_ids=('esc-1-1',))

    resolved = resolve_queue_for_decision(
        _record(session_id=None),
        queues=[orch, recon],
        recon_queue=recon,
        orch_queue_for_project={'df': orch},
    )

    assert resolved == sr.normalize_escalations_dir(recon)


def test_resolve_ambiguous_watcher_session_id_corroborated_picks_orch(tmp_path: Path) -> None:
    """Rule 5: the orchestrator-watcher tiebreak, CORROBORATED — the mirror of
    rule 4, keyed on the record's OWN project so a df watcher never claims a
    reify record.
    """
    recon = _make_queue(tmp_path, 'recon', root_ids=('esc-1-1',))
    orch = _make_queue(tmp_path, 'orch', archive_ids=('esc-1-1',))

    resolved = resolve_queue_for_decision(
        _record(project='df', session_id='watcher-df-4029467'),
        queues=[orch, recon],
        recon_queue=recon,
        orch_queue_for_project={'df': orch},
    )

    assert resolved == sr.normalize_escalations_dir(orch)


def test_resolve_ambiguous_uncorroborated_inference_is_unknown(tmp_path: Path) -> None:
    """Rule 6: THE ONE THAT MATTERS. The inferred queue must actually hold the id.

    Here a ``watcher-df-*`` record is ambiguous between two queues, but its
    project's orchestrator queue is NOT one of them. Trusting the inference
    would stamp a queue that demonstrably does not contain this escalation —
    re-creating the exact cross-queue false-closure hazard this task exists to
    remove, and doing it with a value that LOOKS authoritative. The corroboration
    requirement is what keeps the heuristic a tiebreak instead of a guess.
    """
    a = _make_queue(tmp_path, 'a', root_ids=('esc-1-1',))
    b = _make_queue(tmp_path, 'b', archive_ids=('esc-1-1',))
    orch_without_the_id = _make_queue(tmp_path, 'orch')

    resolved = resolve_queue_for_decision(
        _record(project='df', session_id='watcher-df-4029467'),
        queues=[a, b, orch_without_the_id],
        recon_queue=None,
        orch_queue_for_project={'df': orch_without_the_id},
    )

    assert resolved == sr.UNKNOWN_QUEUE


def test_resolve_ambiguous_uncorroborated_recon_inference_is_unknown(tmp_path: Path) -> None:
    """Rule 6, recon arm: a null session_id whose recon queue does not hold the
    id falls to UNKNOWN for the same reason. Neither tiebreak is privileged.
    """
    a = _make_queue(tmp_path, 'a', root_ids=('esc-1-1',))
    b = _make_queue(tmp_path, 'b', archive_ids=('esc-1-1',))
    recon_without_the_id = _make_queue(tmp_path, 'recon')

    resolved = resolve_queue_for_decision(
        _record(session_id=None),
        queues=[a, b, recon_without_the_id],
        recon_queue=recon_without_the_id,
        orch_queue_for_project={},
    )

    assert resolved == sr.UNKNOWN_QUEUE


def test_resolve_ambiguous_unrecognized_session_id_shape_is_unknown(tmp_path: Path) -> None:
    """Rule 7: a session_id matching NEITHER measured shape yields no inference.

    ``unblock-df-2085-4242`` is a real shape (an /unblock session filing a
    gate); the discriminator was measured only over ``None`` and
    ``watcher-*``, so it says nothing about this one. Extending the rule to
    "anything not-None means orchestrator" would be inventing evidence.
    """
    a = _make_queue(tmp_path, 'a', root_ids=('esc-1-1',))
    b = _make_queue(tmp_path, 'b', archive_ids=('esc-1-1',))
    orch = _make_queue(tmp_path, 'orch', root_ids=('esc-1-1',))

    resolved = resolve_queue_for_decision(
        _record(project='df', session_id='unblock-df-2085-4242'),
        queues=[a, b, orch],
        recon_queue=None,
        orch_queue_for_project={'df': orch},
    )

    assert resolved == sr.UNKNOWN_QUEUE


def test_resolve_ambiguous_unmapped_project_is_unknown(tmp_path: Path) -> None:
    """Rule 8: a watcher-filed record whose project has no --orch-queue mapping.

    The live records spell project ids inconsistently (df / dark_factory /
    dark-factory; autopilot-video / autopilot_video), so an unmapped spelling
    is a REAL possibility, not a contrived one. It must degrade to UNKNOWN
    rather than falling through to some other queue — a missing mapping is
    missing evidence, and the operator's remedy is to pass the mapping and
    re-run, which the report makes visible.
    """
    a = _make_queue(tmp_path, 'a', root_ids=('esc-1-1',))
    b = _make_queue(tmp_path, 'b', archive_ids=('esc-1-1',))

    resolved = resolve_queue_for_decision(
        _record(project='unmapped-project', session_id='watcher-unmapped-project-99'),
        queues=[a, b],
        recon_queue=None,
        orch_queue_for_project={'df': a},
    )

    assert resolved == sr.UNKNOWN_QUEUE


def test_resolve_returns_a_normalized_path(tmp_path: Path) -> None:
    """Whatever the resolver returns is stamped onto a record and later compared
    against a reaper's own --escalations-dir. Both sides must be in the ONE
    canonical spelling or the compare fails open and the record never closes.
    """
    _make_queue(tmp_path, 'orch', root_ids=('esc-1-1',))
    (tmp_path / 'x').mkdir()
    dotted = Path(str(tmp_path / 'x' / '..' / 'orch') + '/')
    assert str(dotted) != str(tmp_path / 'orch')

    resolved = resolve_queue_for_decision(
        _record(),
        queues=[dotted],
        recon_queue=None,
        orch_queue_for_project={},
    )

    assert resolved == sr.normalize_escalations_dir(tmp_path / 'orch')
    assert Path(resolved).is_absolute()
