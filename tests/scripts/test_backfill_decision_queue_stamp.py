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
import os
import subprocess
import sys
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

SCRIPT = Path(__file__).resolve().parent.parent.parent / 'scripts' / 'backfill_decision_queue_stamp.py'


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


# ---------------------------------------------------------------------------
# main() -- subprocess coverage (the repo convention for CLI entry points)
# ---------------------------------------------------------------------------


def _run(fleet_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Drive main() out-of-process against a SYNTHETIC fleet root.

    CLAUDE_FLEET_ROOT is set in the CHILD env only, so a stray absolute-path
    bug in the script surfaces as a failed assertion here rather than as a
    write against the operator's real ~/.claude/fleet.
    """
    env = {**os.environ, 'CLAUDE_FLEET_ROOT': str(fleet_root)}
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
    )


def _fixture_fleet(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A synthetic fleet root + two queues, seeded with the full record matrix.

    Records, and why each is here:
      dec-unique      OPEN, unstamped, id held ONLY by orch      -> stamped orch
      dec-nothing     OPEN, unstamped, no escalation_id          -> stamped UNKNOWN
      dec-ambiguous   OPEN, unstamped, id in BOTH queues, session_id shape the
                      discriminator says nothing about           -> stamped UNKNOWN
      dec-answered    ANSWERED, unstamped                        -> untouched
      dec-dropped     DROPPED, unstamped                         -> untouched
      dec-prestamped  OPEN, stamped with a queue that does NOT hold its id
                                                                 -> untouched
    The last is the important one: this script must never "correct" an existing
    stamp. The watchers' stamps are first-hand evidence from the filer; these
    are inference, and inference must not overwrite evidence — even when the
    inference looks better.
    """
    fleet = tmp_path / 'fleet'
    orch = _make_queue(tmp_path, 'orch', root_ids=('esc-uni-1',), archive_ids=('esc-amb-1',))
    recon = _make_queue(tmp_path, 'recon', root_ids=('esc-amb-1',))

    for rec in (
        _record(id='dec-unique', escalation_id='esc-uni-1', session_id=None),
        _record(id='dec-nothing', escalation_id=None),
        _record(id='dec-ambiguous', escalation_id='esc-amb-1',
                session_id='unblock-df-2085-4242'),
        _record(id='dec-answered', escalation_id='esc-uni-1',
                state=sr.DecisionState.ANSWERED),
        _record(id='dec-dropped', escalation_id='esc-uni-1',
                state=sr.DecisionState.DROPPED),
        _record(id='dec-prestamped', escalation_id='esc-uni-1',
                escalations_dir=sr.normalize_escalations_dir(recon)),
    ):
        assert sr.write_decision(rec, root=fleet)
    return fleet, orch, recon


def _queue_args(orch: Path, recon: Path) -> list[str]:
    return [
        '--queue', str(orch),
        '--queue', str(recon),
        '--recon-queue', str(recon),
        '--orch-queue', f'df={orch}',
    ]


def _stamps(fleet: Path) -> dict[str, str]:
    return {d.id: d.escalations_dir for d in sr.list_decisions(root=fleet)}


def test_main_dry_run_is_the_default_and_writes_nothing(tmp_path: Path) -> None:
    """No --apply -> inert. The single most important property of a migration
    over live fleet state that no test may assert against: an operator must be
    able to see exactly what WOULD happen before anything happens.
    """
    fleet, orch, recon = _fixture_fleet(tmp_path)
    before = _stamps(fleet)

    proc = _run(fleet, *_queue_args(orch, recon))

    assert proc.returncode == 0, proc.stderr
    assert _stamps(fleet) == before
    # Every candidate gets a disposition line naming the id and the outcome.
    for decision_id in ('dec-unique', 'dec-nothing', 'dec-ambiguous'):
        assert decision_id in proc.stdout
    assert str(orch) in proc.stdout
    assert sr.UNKNOWN_QUEUE in proc.stdout


def test_main_apply_stamps_exactly_what_the_dry_run_predicted(tmp_path: Path) -> None:
    """--apply must be the dry run made real, not a second, different decision.

    A dry run whose output does not predict the apply is worse than no dry run
    at all: it manufactures confidence for a review that never happened. Both
    runs go against identically-seeded fixtures and the resolved values are
    compared.
    """
    fleet_a, orch_a, recon_a = _fixture_fleet(tmp_path / 'a')
    fleet_b, orch_b, recon_b = _fixture_fleet(tmp_path / 'b')

    dry = _run(fleet_a, *_queue_args(orch_a, recon_a))
    applied = _run(fleet_b, *_queue_args(orch_b, recon_b), '--apply')

    assert dry.returncode == 0, dry.stderr
    assert applied.returncode == 0, applied.stderr
    stamps = _stamps(fleet_b)
    assert stamps['dec-unique'] == sr.normalize_escalations_dir(orch_b)
    assert stamps['dec-nothing'] == sr.UNKNOWN_QUEUE
    assert stamps['dec-ambiguous'] == sr.UNKNOWN_QUEUE
    # The dry run named the same two outcomes for the same two ids.
    assert dry.stdout.replace(str(tmp_path / 'a'), 'X') == applied.stdout.replace(
        str(tmp_path / 'b'), 'X'
    )


def test_main_apply_never_touches_out_of_scope_records(tmp_path: Path) -> None:
    """Scope is OPEN + unstamped, and nothing else -- asserted byte-for-byte.

    ANSWERED/DROPPED records are not false-closable (the reaper skips any
    non-OPEN record before consulting the queue at all), so stamping them would
    be pure churn on immutable history. The pre-stamped record is the one that
    matters: its stamp is deliberately a queue that does NOT hold its id, so a
    script that "corrected" stamps would visibly change it here.
    """
    fleet, orch, recon = _fixture_fleet(tmp_path)
    paths = {
        decision_id: sr.decision_path_for_id(decision_id, root=fleet)
        for decision_id in ('dec-answered', 'dec-dropped', 'dec-prestamped')
    }
    before = {k: p.read_bytes() for k, p in paths.items()}

    proc = _run(fleet, *_queue_args(orch, recon), '--apply')

    assert proc.returncode == 0, proc.stderr
    assert {k: p.read_bytes() for k, p in paths.items()} == before


def test_main_apply_is_idempotent(tmp_path: Path) -> None:
    """A second --apply finds zero candidates and changes nothing.

    Re-runnability is a hard requirement, not a nicety: the operator runs this
    against live state, and the honest response to a partial or interrupted run
    must be "run it again", never "work out what it already did".
    """
    fleet, orch, recon = _fixture_fleet(tmp_path)

    first = _run(fleet, *_queue_args(orch, recon), '--apply')
    after_first = _stamps(fleet)
    second = _run(fleet, *_queue_args(orch, recon), '--apply')

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert _stamps(fleet) == after_first
    assert 'candidates: 0' in second.stdout


def test_main_verify_exits_nonzero_before_and_zero_after_apply(tmp_path: Path) -> None:
    """--verify is task WORK item 4's actual confirmation, in both directions.

    Pinning only the post-apply exit-0 would pass against a --verify that
    always returns 0 — a check that cannot fail is not a check. So the same
    fixture is verified before the back-fill (must FAIL, naming the residue)
    and after (must pass).
    """
    fleet, orch, recon = _fixture_fleet(tmp_path)

    before = _run(fleet, *_queue_args(orch, recon), '--verify')
    applied = _run(fleet, *_queue_args(orch, recon), '--apply')
    after = _run(fleet, *_queue_args(orch, recon), '--verify')

    assert before.returncode == 2, before.stdout
    assert 'dec-unique' in before.stdout
    assert applied.returncode == 0, applied.stderr
    assert after.returncode == 0, after.stdout


def test_main_prints_a_summary_with_counts_by_disposition(tmp_path: Path) -> None:
    """The summary is the audit report's raw material, so it must break the run
    down by REASON rather than reporting a single total -- 'stamped 39' hides
    whether the script attributed 39 records or gave up on 39.
    """
    fleet, orch, recon = _fixture_fleet(tmp_path)

    proc = _run(fleet, *_queue_args(orch, recon), '--apply')

    assert proc.returncode == 0, proc.stderr
    assert '---- summary ----' in proc.stdout
    assert 'unique-hit' in proc.stdout
    assert 'no-escalation-id' in proc.stdout
    assert 'ambiguous-uncorroborated' in proc.stdout


def test_main_reports_the_tiebreak_reasons_distinctly(tmp_path: Path) -> None:
    """A tiebreak-resolved record must be labelled as such, never merged into
    the unique-hit count. They are different grades of evidence: one is the id
    being in exactly one queue, the other is a measured regularity choosing
    among several. A reviewer reading the report has to be able to tell which
    records rest on which -- that distinction is the whole reason the report
    exists.
    """
    fleet = tmp_path / 'fleet'
    orch = _make_queue(tmp_path, 'orch', root_ids=('esc-t-1',))
    recon = _make_queue(tmp_path, 'recon', root_ids=('esc-t-1',))
    assert sr.write_decision(
        _record(id='dec-tb-recon', escalation_id='esc-t-1', session_id=None), root=fleet
    )
    assert sr.write_decision(
        _record(id='dec-tb-orch', escalation_id='esc-t-1', session_id='watcher-df-1'), root=fleet
    )

    proc = _run(fleet, *_queue_args(orch, recon), '--apply')

    assert proc.returncode == 0, proc.stderr
    assert 'tiebreak-recon' in proc.stdout
    assert 'tiebreak-orch' in proc.stdout
    stamps = _stamps(fleet)
    assert stamps['dec-tb-recon'] == sr.normalize_escalations_dir(recon)
    assert stamps['dec-tb-orch'] == sr.normalize_escalations_dir(orch)


def test_main_fail_soft_on_a_corrupt_decision_file(tmp_path: Path) -> None:
    """One unreadable record must not abort a fleet-wide migration.

    list_decisions already skips a corrupt body; this pins that the script
    inherits that rather than crashing partway through, which would leave the
    population half-stamped with no record of where it stopped.
    """
    fleet, orch, recon = _fixture_fleet(tmp_path)
    (sr.decisions_dir(root=fleet) / 'corrupt.json').write_text('{not valid json')

    proc = _run(fleet, *_queue_args(orch, recon), '--apply')

    assert proc.returncode == 0, proc.stderr
    assert _stamps(fleet)['dec-unique'] == sr.normalize_escalations_dir(orch)


def test_main_requires_at_least_one_queue(tmp_path: Path) -> None:
    """No usable queue argument is its OWN exit code, never a clean run.

    Without it, an operator who fat-fingers the flags gets exit 0 and a
    zero-candidate report that reads exactly like "everything is already
    stamped" -- the failure mode where a migration is believed to have run and
    did not.
    """
    fleet, _orch, _recon = _fixture_fleet(tmp_path)

    proc = _run(fleet, '--apply')

    assert proc.returncode != 0
    assert proc.returncode != 2  # not to be confused with --verify's residue code


def test_main_rejects_a_malformed_orch_queue_pair(tmp_path: Path) -> None:
    """--orch-queue must be PROJECT=PATH, and a malformed pair fails LOUDLY.

    Silently ignoring it would drop the rule-5 tiebreak for that project and
    quietly downgrade every one of its ambiguous records to UNKNOWN -- a
    strictly-worse outcome wearing a clean exit code.
    """
    fleet, orch, recon = _fixture_fleet(tmp_path)

    proc = _run(fleet, '--queue', str(orch), '--orch-queue', str(recon))

    assert proc.returncode != 0
    assert 'orch-queue' in (proc.stderr + proc.stdout)
