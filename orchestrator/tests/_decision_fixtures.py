"""Decision-record builders, CLI filing drivers and assertion helpers shared by the decision test files."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from orchestrator import decision_registry as dr
from orchestrator import session_registry as sr


def make_decision(**overrides: object) -> dr.DecisionRecord:
    """Build a fully-populated DecisionRecord for round-trip/identity tests.

    Mirrors _make_record's kwargs-builder idiom: every field is given a
    concrete, distinguishable value so a round-trip test can catch a field
    being dropped/mis-typed; ``overrides`` lets a test tweak just the
    field(s) it cares about.

    ONE DELIBERATE EXCEPTION: ``escalations_dir`` is absent from the map
    below, so the DecisionRecord dataclass default ('' -- unset/legacy)
    supplies it. It is the only field that is a GUARD INPUT to the reaper's
    axis-2 queue check (``_run_reap_decisions._status``), so a non-empty
    shared default here would make every make_decision()-built decision
    carry a queue FOREIGN to the tmp_path queue a reap test invokes the
    reaper with -- short-circuiting _status before read_escalation_status
    runs and collapsing the test into a vacuous ``assert state == OPEN``. Do
    not add it here; a test needing a real queue passes ``escalations_dir=``
    at its own call site.
    """
    fields: dict = {
        'id': 'dec-1',
        'project': 'df',
        'text': 'approve?',
        'filed_at': '2026-07-07T00:00:00+00:00',
        'session_id': 'unblock-df-2085-4242',
        'task_id': '2085',
        'escalation_id': 'esc-1',
        'options': ['yes', 'no'],
        'manual_boost': 2,
        'state': 'answered',
        'severity': 'critical',
        'record_slug': 'unblock-df-2085-uuid',
    }
    fields.update(overrides)
    return dr.DecisionRecord(**fields)


def qid(local_id: str, project: str = 'df') -> str:
    """The stored id write-decision files a CLI ``--id`` under (task 4835)."""
    return dr.qualify_decision_id(project, local_id)


def names_the_destination_token(message: str) -> bool:
    """True when *message* names ``solar_challenge`` as a token in its OWN
    right -- not merely as the tail of ``my_solar_challenge``.

    Exists because the obvious spelling of that assertion is VACUOUS:
    ``'solar_challenge' in msg`` is implied by ``'my_solar_challenge' in
    msg``, so a regression that dropped the destination token entirely --
    leaving the operator exactly where the silent zero-row no-op did --
    passes it. The word-boundary lookaround is what makes the check
    discriminating; a bare substring test is not.

    Deliberately boundary-based rather than quote-based (``"'solar_challenge'"``
    would also work today) so it survives a message that renders the token
    without ``!r`` quoting -- it pins the CLAIM, not the formatting.
    """
    return re.search(r'(?<!\w)solar_challenge(?!\w)', message) is not None


def claims_zero_matches(message: str) -> bool:
    """True when *message* asserts the passed token matched nothing.

    The one wording pin in this area, and a deliberate one: that claim is
    TRUE for ``reap-decisions`` (which matches) and FALSE for
    ``write-decision`` (which creates, and files a row under exactly that
    token one line later). Pinning it in both directions is what keeps the
    hint's verb-awareness from silently regressing to a single message.
    """
    return re.search(r'matches (no|zero)\b', message, re.IGNORECASE) is not None


def two_queues(tmp_path: Path) -> tuple[Path, Path]:
    """Build the observed two-queue collision on disk (task 3528).

    dark_factory runs TWO escalation queues over ONE ``esc-<taskid>-<n>`` id
    namespace: the orchestrator's ``data/escalations`` and the reconciliation
    watcher's ``data/reconciliation/escalations``. Here that is `orch` and
    `recon`, both holding an *unrelated* escalation that happens to share the
    id ``esc-3036-1`` -- RESOLVED (archived) in `orch`, still PENDING (queue
    root) in `recon`, exactly as observed when a blocking recon gate sat
    invisible in the cockpit for ~7 days.

    Reuses the layout pinned by test_read_escalation_status_reads_queue_root_file
    / ..._falls_back_to_archive: queue-root file = pending, dated
    ``archive/YYYY-MM-DD/`` file = resolved.
    """
    orch = tmp_path / 'orch'
    recon = tmp_path / 'recon'
    orch_archive = orch / 'archive' / '2026-07-26'
    orch_archive.mkdir(parents=True)
    recon.mkdir(parents=True)
    (orch_archive / 'esc-3036-1.json').write_text(json.dumps({'status': 'resolved'}))
    (recon / 'esc-3036-1.json').write_text(json.dumps({'status': 'pending'}))
    return orch, recon


def seed_mixed_spellings(tmp_path: Path) -> dict[str, dict[str, Any]]:
    """Seed the observed mixed-spelling population, one record per case.

    Mirrors the live 2026-08-06 census in miniature: three dark-factory
    spellings needing repair (one per DecisionState, so preservation is
    checked against a row that must NOT be reopened), one already-canonical
    dark-factory row, and one unrelated project. Returns each seeded record's
    ``to_dict()`` keyed by id, so a test can diff the whole dict before/after
    and catch a field a future change lets escape.
    """
    seeds = {
        'd-df': ('df', dr.DecisionState.OPEN),
        'd-hyphen': ('dark-factory', dr.DecisionState.ANSWERED),
        'd-mixed': ('Dark_Factory', dr.DecisionState.DROPPED),
        'd-canon': ('dark_factory', dr.DecisionState.OPEN),
        'd-reify': ('reify', dr.DecisionState.OPEN),
    }
    before: dict[str, dict[str, Any]] = {}
    for decision_id, (project, state) in seeds.items():
        record = make_decision(id=decision_id, project=project, state=state)
        dr.write_decision(record, root=tmp_path)
        before[decision_id] = record.to_dict()
    return before


EVIDENCE = 'gate 1 ruling_is_leos_own: held\nLeo 2026-09-20: esc-400-1 option A'


def file_decision(**kwargs: str | None) -> int:
    """Invoke the write-decision verb with kwargs as --flags, skipping Nones.

    Keeps the MODE-2 end-to-end cases readable: each is two filings that
    differ in only a few fields, and spelling both argv lists out in full
    buries the difference under boilerplate.
    """
    argv = ['write-decision']
    for name, value in kwargs.items():
        if value is not None:
            argv += [f'--{name.replace("_", "-")}', value]
    return sr.main(argv)


def file_from_a_linked_session(
    monkeypatch: pytest.MonkeyPatch, root: Path, pid: int = 7_527_001
) -> None:
    """Make the next write-decision file from a session linked to its own registry record.

    write-decision WARNs when it cannot link a filing to its filer's session
    record (task 7527), so a test pinning an otherwise-quiet filing path files
    from a linked session, as a spawned watcher does. The pid pointer counts
    only while the record names that pid as its owner, so that is the one
    field set beyond the mandatory ones.
    """
    record = sr.SessionRecord(
        session_slug='linked-watcher-record',
        status=sr.Status.LAUNCHING,
        start_ts='2026-07-07T00:00:00+00:00',
        claude_owner_pid=pid,
    )
    sr.write_record(record, root=root)
    sr.write_session_pointer(pid, 'linked-watcher-record', root=root)
    monkeypatch.setenv('CLAUDE_PID', str(pid))
