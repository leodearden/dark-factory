"""The ONE home for the pending-anchor-fold escalation filer skeleton (INV-5).

`middleware/_folded_escalation.file_folded_escalation` is the extracted body
that seven fused-memory filers had each been carrying verbatim: defensive
optional-`escalation`-package import, guarded queue construction, a
`get_by_task(anchor, status='pending')` fold, and a never-raise
`Escalation(...)` + `queue.submit(...)`.

WHAT MUST NOT COLLAPSE.  `anchor_task_id` is a REQUIRED keyword-only parameter
with NO default, and every caller keeps its own `_ANCHOR_TASK_ID` constant in
its own module.  A filer that dedupes against an anchor somebody else keeps
open never files again, and that silence is indistinguishable from health —
see `TestNoTwoFilersShareAnAnchor` for the measured incident.

NOT THE HOME FOR the `submit_or_dedupe` content-fingerprint family
(`scope_violation_escalator`, `mem0_update_storm_escalator`,
`entity_mint_storm_escalator`).  Those dedupe on a content fingerprint over a
cached per-project queue, not on a pending anchor; routing them through this
helper would be a behaviour change, not a refactor.
"""

from __future__ import annotations

import ast
import asyncio
import importlib
import importlib.util
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import pytest
from _ast_guard import parse_python_module
from shared.project_layout import escalation_queue_dir

from fused_memory.middleware import _folded_escalation
from fused_memory.middleware._folded_escalation import file_folded_escalation
from fused_memory.middleware.curation_deferral import DeferReason
from fused_memory.middleware.ticket_store import CurationBacklog
from fused_memory.services.write_journal import JournalGrowthSample

#: Applied PER CLASS rather than as a module-level ``pytestmark``, so the
#: pairwise anchor-collision regression at the foot of this file keeps running
#: in a minimal env. It compares module CONSTANTS and needs no queue — and an
#: anchor collision that only fails where the escalation package happens to be
#: installed is an alarm switched off exactly where nobody is looking.
_needs_escalation = pytest.mark.skipif(
    not _folded_escalation.HAS_ESCALATION,
    reason='escalation package unavailable (minimal env); the HAS_ESCALATION '
           'no-op arm is covered separately below',
)

_ANCHOR = 'folded-escalation-test'
_ROLE = 'fused-memory/folded-escalation-test'
_CATEGORY = 'risk_identified'

logger = logging.getLogger('fused_memory.tests.folded_escalation')


def _emit(tmp_path, **overrides) -> str | None:
    """Call the helper with every required keyword supplied."""
    kwargs = {
        'anchor_task_id': _ANCHOR,
        'agent_role': _ROLE,
        'category': _CATEGORY,
        'severity': 'blocking',
        'summary': 'a folded summary',
        'detail': 'a folded detail\nwith two lines',
        'suggested_action': 'do the thing',
        'logger': logger,
        'log_label': 'folded_escalation_test',
    }
    kwargs.update(overrides)
    root = kwargs.pop('project_root', str(tmp_path))
    return file_folded_escalation(root, **kwargs)


def _filed(tmp_path) -> list[dict]:
    queue_dir = tmp_path / 'data' / 'escalations'
    if not queue_dir.is_dir():
        return []
    return [json.loads(p.read_text()) for p in sorted(queue_dir.glob('esc-*.json'))]


@_needs_escalation
class TestTheFiledEscalation:
    """One escalation, into the caller-named project's OWN queue."""

    def test_files_exactly_one_escalation_and_returns_its_id(self, tmp_path):
        esc_id = _emit(tmp_path)

        assert isinstance(esc_id, str)
        filed = _filed(tmp_path)
        assert len(filed) == 1
        assert filed[0]['id'] == esc_id

    def test_lands_in_the_named_projects_own_queue(self, tmp_path):
        """`{project_root}/data/escalations` — never the server cwd, where no
        operator watches."""
        _emit(tmp_path)
        assert (tmp_path / 'data' / 'escalations').is_dir()

    def test_files_into_the_layout_escalation_queue_when_relocated(
        self, tmp_path, escalation_queue_relocation,
    ):
        esc_id = _emit(tmp_path)

        filed = sorted(escalation_queue_dir(tmp_path).glob('esc-*.json'))
        assert [json.loads(p.read_text())['id'] for p in filed] == [esc_id]
        assert not (tmp_path / escalation_queue_relocation.legacy_relpath).exists()

    def test_the_id_is_minted_off_the_anchor(self, tmp_path):
        esc_id = _emit(tmp_path)
        assert esc_id is not None
        assert esc_id.startswith(f'esc-{_ANCHOR}-'), (
            'the id must be minted by queue.make_id off the caller anchor, so '
            'the series is greppable and the dedupe lookup can find it'
        )

    def test_the_caller_fields_pass_through_unmodified(self, tmp_path):
        """The helper owns the SKELETON, not the content: every caller-supplied
        field lands on the record byte-for-byte."""
        esc_id = _emit(
            tmp_path,
            summary='a very specific summary',
            detail='line one\nline two\nline three',
            suggested_action='grep for write_triage: in the server log',
        )
        assert esc_id is not None
        record = _filed(tmp_path)[0]

        assert record['task_id'] == _ANCHOR
        assert record['agent_role'] == _ROLE
        assert record['category'] == _CATEGORY
        assert record['severity'] == 'blocking'
        assert record['summary'] == 'a very specific summary'
        assert record['detail'] == 'line one\nline two\nline three'
        assert record['suggested_action'] == (
            'grep for write_triage: in the server log'
        )

    def test_severity_is_the_callers_not_the_helpers(self, tmp_path):
        """Two of the seven callers file at `info`, five at `blocking`; the
        helper must never impose one."""
        esc_id = _emit(tmp_path, severity='info')
        assert esc_id is not None
        assert _filed(tmp_path)[0]['severity'] == 'info'

    def test_is_born_at_l1_by_default(self, tmp_path):
        """Every migrated filer is a background server process filing under a
        synthetic anchor that is never dispatched and therefore never has a
        steward, so an L0 entry would have no consumer at all."""
        esc_id = _emit(tmp_path)
        assert esc_id is not None
        assert _filed(tmp_path)[0]['level'] == 1

    def test_an_explicit_level_is_honoured(self, tmp_path):
        """`emit_markup_residue_escalation` carries a caller-supplied level
        (defaulting to 2), so the default must be overridable."""
        esc_id = _emit(tmp_path, level=2)
        assert esc_id is not None
        assert _filed(tmp_path)[0]['level'] == 2


@_needs_escalation
class TestTheAnchorIsRequired:
    """The one property the consolidation must not collapse."""

    def test_anchor_task_id_is_required_keyword_only(self, tmp_path):
        """No default: forgetting the anchor is a TypeError at call time, not a
        silently disabled alarm sharing somebody else's anchor."""
        with pytest.raises(TypeError):
            file_folded_escalation(  # type: ignore[call-arg]
                str(tmp_path),
                agent_role=_ROLE,
                category=_CATEGORY,
                severity='blocking',
                summary='s',
                detail='d',
                suggested_action='a',
                logger=logger,
                log_label='folded_escalation_test',
            )

    def test_the_anchor_cannot_be_passed_positionally(self, tmp_path):
        """Keyword-only, so a caller cannot drift the anchor into another
        parameter's slot by reordering."""
        with pytest.raises(TypeError):
            file_folded_escalation(str(tmp_path), _ANCHOR)  # type: ignore[misc]


@_needs_escalation
class TestDedupeFold:
    """A SUSTAINED storm folds into one pending parent, not one page per breach."""

    def test_a_second_call_while_the_first_is_pending_reuses_its_id(self, tmp_path):
        first = _emit(tmp_path)
        second = _emit(tmp_path, summary='the storm got worse')

        assert first is not None
        assert second == first
        assert len(_filed(tmp_path)) == 1, (
            'a storm breaches its threshold on EVERY subsequent event; filing '
            'one escalation per breach would bury the operator queue'
        )

    def test_dedupe_is_keyed_on_the_PENDING_anchor(self, tmp_path):
        """`queue.get_by_task(anchor, status='pending')` — a RESOLVED record
        under the same anchor must not suppress a fresh alarm, or the filer is
        one-shot for the queue's lifetime."""
        first = _emit(tmp_path)
        assert first is not None
        from escalation.queue import EscalationQueue

        EscalationQueue(tmp_path / 'data' / 'escalations').resolve(
            first, 'the underlying regression was fixed',
        )

        third = _emit(tmp_path)
        assert third is not None
        assert third != first
        # `resolve` archives the first out of the queue root, so the root now
        # holds exactly the fresh alarm — the lookup is scoped to PENDING, not
        # to "an escalation was ever filed under this anchor".
        assert [record['id'] for record in _filed(tmp_path)] == [third]

    def test_two_project_roots_do_not_fold_into_each_other(self, tmp_path):
        """Each project's alarm lands in its OWN queue: `reify` storming must
        not be silenced by an open `dark_factory` alarm."""
        root_a = tmp_path / 'a'
        root_b = tmp_path / 'b'
        root_a.mkdir()
        root_b.mkdir()

        id_a = _emit(root_a)
        id_b = _emit(root_b)

        assert id_a is not None
        assert id_b is not None
        assert len(_filed(root_a)) == 1
        assert len(_filed(root_b)) == 1

    def test_two_anchors_in_the_SAME_project_do_not_fold_into_each_other(
        self, tmp_path,
    ):
        """The anchor-squat regression, expressed at helper level.

        MEASURED INCIDENT: the L1 escalation watcher squatted the
        `markup-tripwire` anchor, so the tripwire filed NOTHING from
        2026-08-16 to 2026-08-19 while 41 rejections occurred — all 17
        records sat at `dedupe_count` 0, meaning the fold was not folding,
        the filer was simply never firing.  A filer deduping against an
        anchor someone else keeps open goes permanently silent, and that
        silence is indistinguishable from health.

        Two filers sharing one queue must therefore key their folds on
        DIFFERENT anchors.  `anchor_task_id` threads through BOTH
        `get_by_task` and `make_id`/`task_id` from one parameter precisely so
        it is structurally impossible to file under one anchor while deduping
        against another.
        """
        first = _emit(tmp_path, anchor_task_id='filer-one')
        second = _emit(tmp_path, anchor_task_id='filer-two')

        assert first is not None
        assert second is not None
        assert second != first
        filed = _filed(tmp_path)
        assert len(filed) == 2, (
            "an open alarm under one filer's anchor must not silence a "
            "different filer sharing the same project queue"
        )
        assert {record['task_id'] for record in filed} == {'filer-one', 'filer-two'}

    def test_dedupe_false_files_a_second_record_while_the_first_is_pending(
        self, tmp_path,
    ):
        """`emit_markup_residue_escalation` opts OUT: each of its records is
        the only surviving copy of a DIFFERENT caller payload, so folding two
        together would destroy the very data the record exists to preserve."""
        first = _emit(tmp_path, dedupe=False, detail='payload A')
        second = _emit(tmp_path, dedupe=False, detail='payload B')

        assert first is not None
        assert second is not None
        assert second != first
        filed = _filed(tmp_path)
        assert len(filed) == 2
        assert {record['detail'] for record in filed} == {'payload A', 'payload B'}

    def test_dedupe_false_never_calls_get_by_task_at_all(self, tmp_path, monkeypatch):
        """Not merely "ignores the result": the lookup is skipped, so a
        non-deduping filer cannot be broken by a queue-scan failure."""
        real_queue = _folded_escalation.EscalationQueue
        calls: list[tuple] = []

        class _RecordingQueue(real_queue):  # type: ignore[misc,valid-type]
            def get_by_task(self, *args, **kwargs):
                calls.append((args, kwargs))
                return super().get_by_task(*args, **kwargs)

        monkeypatch.setattr(_folded_escalation, 'EscalationQueue', _RecordingQueue)

        assert _emit(tmp_path, dedupe=False) is not None
        assert calls == []

        assert _emit(tmp_path, dedupe=True) is not None
        assert len(calls) == 1, 'the default path DOES consult the pending anchor'


@_needs_escalation
class TestNeverRaises:
    """Called from live write paths: a raise here fails a write because the
    COMPLAINT about the write failed.

    Every caller's own docstring already promises this; consolidating to one
    home means the promise is now kept in exactly one place.
    """

    def test_a_submit_failure_returns_none_and_logs(self, tmp_path, monkeypatch, caplog):
        class _BrokenQueue:
            def __init__(self, *_a, **_kw):
                pass

            def get_by_task(self, *_a, **_kw):
                return []

            def make_id(self, task_id):
                return f'esc-{task_id}-1'

            def submit(self, _esc):
                raise OSError('read-only filesystem')

        monkeypatch.setattr(_folded_escalation, 'EscalationQueue', _BrokenQueue)

        with caplog.at_level(logging.DEBUG):
            assert _emit(tmp_path) is None
        assert any(r.levelno == logging.ERROR for r in caplog.records), (
            'a swallowed failure must still be visible, and this arm is '
            '`logger.exception`, i.e. ERROR. Captured at DEBUG and pinned on '
            'levelno so a downgrade FAILS here rather than being hidden by the '
            'capture filter, which a bare `assert caplog.records` under '
            "`at_level('ERROR')` could not distinguish from silence."
        )

    def test_a_get_by_task_failure_falls_through_to_filing(self, tmp_path, monkeypatch):
        """A read failure must not BLOCK the alarm — better a possible
        duplicate than a silenced storm.  This arm is reached only when the
        queue directory is already misbehaving."""
        real_queue = _folded_escalation.EscalationQueue

        class _UnreadableQueue(real_queue):  # type: ignore[misc,valid-type]
            def get_by_task(self, *_a, **_kw):
                raise OSError('queue scan failed')

        monkeypatch.setattr(_folded_escalation, 'EscalationQueue', _UnreadableQueue)

        esc_id = _emit(tmp_path)
        assert isinstance(esc_id, str), (
            'the lookup guard must FALL THROUGH to filing, never return early'
        )
        assert len(_filed(tmp_path)) == 1

    def test_a_queue_construction_failure_returns_none(self, tmp_path, monkeypatch):
        """Constructing the queue creates its directory; a read-only or missing
        project_root must not turn an alarm into a crash on the write path."""
        def _explode(*_a, **_kw):
            raise OSError('cannot create queue dir')

        monkeypatch.setattr(_folded_escalation, 'EscalationQueue', _explode)

        assert _emit(tmp_path) is None

    def test_a_malformed_payload_degrades_to_no_escalation(self, tmp_path, monkeypatch):
        """`Escalation(...)` is constructed INSIDE the guard deliberately: a
        malformed payload must degrade to "no escalation", never to an
        exception out of the guard."""
        def _explode(*_a, **_kw):
            raise ValueError('malformed escalation payload')

        monkeypatch.setattr(_folded_escalation, 'Escalation', _explode)

        assert _emit(tmp_path) is None

    def test_without_the_escalation_package_it_no_ops(self, tmp_path, monkeypatch):
        """The minimal-env path: nothing filed, `None` returned. Every caller
        must behave identically whether or not the optional `escalation`
        workspace package is installed. The line it logs is pinned by
        TestTheEmittedLevels."""
        monkeypatch.setattr(_folded_escalation, 'HAS_ESCALATION', False)

        assert _emit(tmp_path) is None
        assert not (tmp_path / 'data' / 'escalations').exists()

    def test_a_none_project_root_returns_none_quietly(self, tmp_path, caplog):
        """`write_triage` takes `project_root: str | None`; None means there is
        no project queue to file into, which is not an error."""
        with caplog.at_level('DEBUG'):
            assert _emit(tmp_path, project_root=None) is None
        assert not (tmp_path / 'data' / 'escalations').exists()


@_needs_escalation
class TestFoldHookAndLogging:
    """`on_fold` exists so a caller can keep FOLD-TIME logging on ITS OWN
    logger — `emit_markup_storm_escalation` compares the folded burst's
    outcome against the open record's and logs at ERROR when they differ, and
    `tests/server/test_markup_tripwire.py` asserts that record's `.name` is
    `fused_memory.server.markup_tripwire`."""

    def test_on_fold_is_called_once_with_the_existing_escalation(self, tmp_path):
        first = _emit(tmp_path)
        assert first is not None

        seen: list = []
        second = _emit(tmp_path, on_fold=seen.append)

        assert second == first
        assert len(seen) == 1
        assert seen[0].id == first, (
            'the hook receives the EXISTING escalation object, so a caller can '
            'read fields off the open record (markup_storm reads its outcome)'
        )
        assert len(_filed(tmp_path)) == 1

    def test_on_fold_is_not_called_when_no_fold_occurs(self, tmp_path):
        seen: list = []
        assert _emit(tmp_path, on_fold=seen.append) is not None
        assert seen == []

    def test_an_on_fold_that_raises_does_not_propagate(self, tmp_path):
        """The hook is caller-supplied logging on a never-raise path; it must
        not become a new way to break the write path."""
        first = _emit(tmp_path)
        assert first is not None

        def _boom(_existing):
            raise RuntimeError('the fold hook blew up')

        second = _emit(tmp_path, on_fold=_boom)

        assert second == first, (
            'a broken hook must not cost the caller the fold result'
        )
        assert len(_filed(tmp_path)) == 1

    def test_without_on_fold_the_helper_logs_its_own_fold_line(self, tmp_path, caplog):
        first = _emit(tmp_path)
        assert first is not None

        with caplog.at_level('INFO'):
            second = _emit(tmp_path)

        assert second == first
        assert any(first in r.getMessage() for r in caplog.records), (
            'a fold is a suppression; it must stay visible in the log'
        )

    def test_log_label_and_context_appear_in_the_emitted_messages(
        self, tmp_path, caplog,
    ):
        """`log_label` carries the operator-facing grep token that existing
        detail text tells triagers to search for; `context` carries the
        caller's own description of the subject."""
        with caplog.at_level('DEBUG'):
            _emit(
                tmp_path,
                log_label='write_triage',
                context="fail-open storm 'unknown_key'",
            )

        messages = [r.getMessage() for r in caplog.records]
        assert any('write_triage' in m for m in messages)
        assert any("fail-open storm 'unknown_key'" in m for m in messages)

    def test_records_are_attributed_to_the_PASSED_IN_logger(self, tmp_path, caplog):
        """A helper logging under its OWN name would break both module
        attribution and every existing caplog filter keyed on the caller."""
        callers_logger = logging.getLogger('fused_memory.server.some_caller')

        with caplog.at_level('DEBUG'):
            _emit(tmp_path, logger=callers_logger)

        names = {r.name for r in caplog.records}
        assert 'fused_memory.server.some_caller' in names
        assert 'fused_memory.middleware._folded_escalation' not in names

    def test_context_reaches_the_no_op_arms_too(self, tmp_path, monkeypatch, caplog):
        """The quiet arms are where an operator most needs to know WHAT went
        unescalated, so `context` must not be dropped on them."""
        monkeypatch.setattr(_folded_escalation, 'HAS_ESCALATION', False)

        with caplog.at_level('DEBUG'):
            assert _emit(tmp_path, context='the subject that went unescalated') is None

        assert any(
            'the subject that went unescalated' in r.getMessage()
            for r in caplog.records
        )


class TestTheEmittedLevels:
    """A never-raise alarm path is exactly where a silent level downgrade is
    most costly, because the ONLY symptom is absence of output.

    Every other arm of this helper announces a downgrade somehow — a wrong
    `severity` shows up in the filed record, a wrong anchor shows up in the
    pairwise sweep. A level that drops below the operator's threshold produces
    no record, no file and no exception; the alarm simply stops being seen,
    and "no output" is indistinguishable from "nothing to report". That is the
    same silence this module's own docstring argues against, one layer down.

    So the levels are PINNED rather than left to whatever `logger.<method>`
    the body happens to call. The fold arm defaults to INFO and a caller
    replaces it wholesale via `on_fold`; the package-unavailable arm is
    ALWAYS WARNING, with no override.

    Asserted on `levelno` against the `logging.*` constants, never on
    `levelname` strings: a level is an ordered numeric fact, and comparing its
    rendered name would be a meaningful string standing in for structured data
    (heuristic 12).
    """

    def test_the_package_unavailable_arm_logs_at_warning(
        self, tmp_path, monkeypatch, caplog,
    ):
        """A caller reaches this arm only after deciding an alarm fires, so it
        records a LOST alarm. The fused-memory server logs at INFO
        (`fused-memory/src/fused_memory/server/main.py`, `logging.basicConfig`),
        so DEBUG here printed nothing; and `escalation` is only a dev-group
        dependency, so a `--no-dev` deploy loses it for every filer at once.
        `orchestrator/src/orchestrator/mcp/markup_sink.py::open_escalation_channel`
        logs the same condition at WARNING. The volume is bounded: callers
        reach the helper only on a breach, and the storm callers rate-limit
        their breaches per window."""
        monkeypatch.setattr(_folded_escalation, 'HAS_ESCALATION', False)

        with caplog.at_level(logging.DEBUG):
            assert _emit(tmp_path) is None

        assert [r.levelno for r in caplog.records] == [logging.WARNING]

    @_needs_escalation
    def test_the_default_fold_arm_is_info(self, tmp_path, caplog):
        """A fold is a SUPPRESSION, so it stays visible — but a storming
        project folds on every subsequent event, which is why the default sits
        at INFO rather than WARNING. A caller wanting otherwise supplies
        `on_fold` and logs on its own logger at its own level."""
        first = _emit(tmp_path)
        assert first is not None

        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            assert _emit(tmp_path) == first

        assert [r.levelno for r in caplog.records] == [logging.INFO]


class _Filer(NamedTuple):
    """One migrated filer: where its spy goes, how to fire it, what it forwards."""

    label: str
    #: Dotted path of the filer's OWN module, which is also the name of the
    #: logger it logs on. The spy is installed HERE, not on
    #: `_folded_escalation`, because that is where the name the filer body
    #: resolves actually lives — patching the helper's home would prove nothing
    #: about what this caller forwards.
    module: str
    fire: Callable[[Any, str], object]
    #: The escalation level the filed record must be born at.
    escalation_level: int


def _storm(module: Any, project_root: str) -> object:
    return module.emit_markup_storm_escalation(
        project_root,
        {'count': 3, 'threshold': 3, 'window_seconds': 3600.0,
         'outcome': 'rejected', 'project': '/project-a'},
    )


def _residue(module: Any, project_root: str) -> object:
    return module.emit_markup_residue_escalation(
        project_root, {'tool': 'add_memory', 'field': 'content'},
    )


def _journal_growth(ceiling: str) -> Callable[[Any, str], object]:
    def fire(module: Any, project_root: str) -> object:
        sample = JournalGrowthSample(
            file_bytes=2, free_bytes=0, rows_inserted=2,
            since=datetime(2026, 10, 8, tzinfo=UTC),
        )
        breach = module.JournalGrowthBreach(
            ceiling=module.JournalCeiling[ceiling], measured=2, limit=1,
        )
        return module.emit_journal_growth_escalation(project_root, breach, sample)

    return fire


def _unverified_claim(module: Any, project_root: str) -> object:
    return module.emit_unverified_claim_escalation(
        project_root,
        {'claims': [{'subject': 'task', 'ref': '5422', 'kind': 'task',
                     'project_id': 'dark_factory', 'status': 'mismatch',
                     'observed': 'in-progress', 'text': 'task 5422 has been applied'}]},
    )


class _OneCurationBacklogStore:
    """The reads a bare `TicketJanitor.tick` makes: no failed tickets, and one
    backlog that has survived a retry."""

    async def fetch_unescalated_failures(self, project_id=None, limit=100) -> list[dict]:
        return []

    async def summarise_awaiting_curation(self) -> list[CurationBacklog]:
        return [CurationBacklog(
            project_id='dark_factory', defer_reason=DeferReason.ALL_ACCOUNTS_CAPPED,
            count=1, oldest_created_at='2026-10-08T00:00:00+00:00', max_attempts=2,
        )]

    async def oldest_awaiting_curation(self, project_id, *, defer_reason, limit) -> list[dict]:
        return []


def _curation_backlog(module: Any, project_root: str) -> object:
    janitor = module.TicketJanitor(
        _OneCurationBacklogStore(), known_projects={'dark_factory': project_root},
    )
    return asyncio.run(janitor.tick())


#: THE SINGLE POINT OF TRUTH (heuristic 11) for the escalation level each
#: migrated filer's record is born at.
#:
#: That level is silent when it drifts — a record that waits for the orphan-L0
#: reaper looks like "nothing to report" — so it is fenced here rather than
#: left implicit in every call site. Changing what a filer forwards means
#: editing this table, which is the point: it cannot land unnoticed. The
#: lost-alarm log level is the same for every row, and
#: `test_a_lost_alarm_is_logged_at_warning_on_the_filers_own_logger` pins it
#: by firing each row for real.
_MIGRATED_FILERS: tuple[_Filer, ...] = (
    _Filer(
        'write_triage',
        'fused_memory.server.write_triage',
        lambda m, root: m.emit_triage_fail_open_storm_escalation(
            root, {'count': 5, 'window_seconds': 60},
        ),
        1,
    ),
    _Filer(
        'markup_tripwire storm',
        'fused_memory.server.markup_tripwire',
        _storm,
        1,
    ),
    _Filer(
        'markup_tripwire residue',
        'fused_memory.server.markup_tripwire',
        _residue,
        # The residue record's own `level`, falling back to L2: it holds a
        # payload only a human can repair.
        2,
    ),
    _Filer(
        'candidate_key_escalation',
        'fused_memory.middleware.candidate_key_escalation',
        lambda m, root: m.emit_residual_candidate_key_escalation(
            root,
            [{'tag': 't', 'candidate_key': 'k', 'task_ids': ['1', '2'],
              'count': 2, 'reason': 'mixed_status'}],
        ),
        1,
    ),
    _Filer(
        'referent_repair_storm_escalator',
        'fused_memory.middleware.referent_repair_storm_escalator',
        lambda m, root: m.emit_referent_repair_storm_escalation(
            root, project_id='dark_factory', streak=10, threshold=10,
            repairs=1, records=[],
        ),
        1,
    ),
    _Filer(
        'completion_claim_gate',
        'fused_memory.services.completion_claim_gate',
        _unverified_claim,
        # L0, the Escalation model's default, which this filer has always been
        # born at. Whether an 'info' record under a synthetic anchor should be
        # born at L1 like its siblings is a design question, not a refactor's.
        0,
    ),
    _Filer(
        'memory_metadata_census',
        'fused_memory.services.memory_metadata_census',
        lambda m, root: m.file_unknown_key_storm_escalation(
            root, project_id='dark_factory', agent_id='claude-x',
            keys=['weird_key'],
        ),
        1,
    ),
    _Filer(
        'journal_growth_alarm file_size',
        'fused_memory.services.journal_growth_alarm',
        _journal_growth('FILE_SIZE'),
        1,
    ),
    _Filer(
        'journal_growth_alarm insert_rate',
        'fused_memory.services.journal_growth_alarm',
        _journal_growth('INSERT_RATE'),
        1,
    ),
    _Filer(
        'classification_fallback_alarm',
        'fused_memory.services.classification_fallback_alarm',
        lambda m, root: m.emit_classification_fallback_storm_escalation(
            root, project_id='dark_factory',
            storm={'count': 5, 'threshold': 5, 'window_seconds': 3600.0,
                   'labels': ['llm_error']},
        ),
        1,
    ),
    _Filer(
        'link_heal_executor',
        'fused_memory.maintenance.link_heal_executor',
        lambda m, root: m.FoldedEscapeFiler(root)(
            m.Escape(anchor=m.BACKLOG_ANCHOR, summary='s', detail={}),
        ),
        1,
    ),
    _Filer(
        'ticket_janitor curation-backlog',
        'fused_memory.middleware.ticket_janitor',
        _curation_backlog,
        1,
    ),
)


#: Test-file-relative, so the census always reads the tree this test lives in,
#: never an inherited venv's editable install of some other checkout.
_SRC_ROOT = Path(__file__).parents[1] / 'src' / 'fused_memory'

_HELPER_MODULE = _folded_escalation.__name__


def _imported_targets(tree: ast.AST, package: str) -> set[str]:
    """The absolute dotted name of everything an import under *tree* binds.

    `import a.b` yields `a.b`, `from a import b` yields `a.b`, and a relative
    `from .a import b` inside package `p` yields `p.a.b`, resolved by the
    import system's own `importlib.util.resolve_name`.
    """
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            source = importlib.util.resolve_name(
                '.' * node.level + (node.module or ''), package,
            )
            targets.update(f'{source}.{alias.name}' for alias in node.names)
    return targets


def _imports_the_helper(tree: ast.AST, package: str) -> bool:
    """Whether *tree* imports the helper's module or anything from it, in any
    spelling: so a caller cannot leave the table by how it writes the import."""
    return any(
        target == _HELPER_MODULE or target.startswith(f'{_HELPER_MODULE}.')
        for target in _imported_targets(tree, package)
    )


def _modules_that_call_the_helper() -> set[str]:
    """The dotted name of every production module that imports the helper.

    Over PARSED imports, so prose that merely names the helper cannot count.
    """
    found: set[str] = set()
    for path in sorted(_SRC_ROOT.rglob('*.py')):
        package = path.parent.relative_to(_SRC_ROOT.parent).as_posix().replace('/', '.')
        if _imports_the_helper(parse_python_module(path), package):
            found.add(package if path.stem == '__init__' else f'{package}.{path.stem}')
    return found


def _forwarded_kwargs(filer: _Filer, tmp_path, monkeypatch) -> dict:
    """Fire *filer* against a spy and return what it forwarded to the helper."""
    module = importlib.import_module(filer.module)
    seen: dict = {}

    def _spy(_project_root, **kwargs):
        seen.update(kwargs)
        return 'esc-spied-1'

    monkeypatch.setattr(module, 'file_folded_escalation', _spy)
    filer.fire(module, str(tmp_path))

    assert seen, f'{filer.label} did not reach file_folded_escalation at all'
    return seen


class TestEveryFilerPinsItsLevels:
    """The house levels, their exceptions, and a fence against a new answer.

    Deliberately NOT decorated with `_needs_escalation`: the lost-alarm check
    runs the package-unavailable arm and the escalation-level check replaces
    `file_folded_escalation` with a spy, so no queue is ever built — and a
    level regression that only fails where the escalation package happens to
    be installed is an alarm switched off exactly where nobody is looking.

    The forwarded escalation level is read with `.get` and the helper's own
    default, so a row passes whether the caller omits the keyword or passes it
    explicitly: the property under test is the EFFECTIVE level, not the call
    spelling.
    """

    @pytest.mark.parametrize(
        'filer', _MIGRATED_FILERS, ids=[f.label for f in _MIGRATED_FILERS],
    )
    def test_a_lost_alarm_is_logged_at_warning_on_the_filers_own_logger(
        self, filer, tmp_path, monkeypatch, caplog,
    ):
        """Only the lines the helper itself emits are compared. A long-lived
        filer such as the ticket janitor logs once on construction, and driving
        an async filer starts an event loop that logs on `asyncio`; neither is
        the lost alarm."""
        monkeypatch.setattr(_folded_escalation, 'HAS_ESCALATION', False)
        module = importlib.import_module(filer.module)

        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            assert filer.fire(module, str(tmp_path)) is None

        records = [
            (r.name, r.levelno) for r in caplog.records
            if r.pathname == _folded_escalation.__file__
        ]
        assert records == [(filer.module, logging.WARNING)], (
            f'{filer.label} logged {records!r} for a lost alarm. Every filer '
            'logs on its own `__name__` logger, and a lost alarm below WARNING '
            "prints nothing at the fused-memory server's INFO threshold — see "
            'TestTheEmittedLevels.'
        )

    @pytest.mark.parametrize(
        'filer', _MIGRATED_FILERS, ids=[f.label for f in _MIGRATED_FILERS],
    )
    def test_the_forwarded_escalation_level_matches_the_table(
        self, filer, tmp_path, monkeypatch,
    ):
        seen = _forwarded_kwargs(filer, tmp_path, monkeypatch)

        forwarded = seen.get('level', 1)
        assert forwarded == filer.escalation_level, (
            f'{filer.label} files at level={forwarded!r}, table says '
            f'{filer.escalation_level!r}. An L0 record under a synthetic anchor '
            'has no steward and waits for the orphan-L0 reaper; an L1 record '
            'reaches the escalation watcher at once. If the change is '
            'deliberate, edit the table and say why in the row.'
        )

    def test_the_table_covers_every_module_that_calls_the_helper(self):
        """Anti-vacuity: a parametrized fence passes trivially for a filer that
        is simply absent from the table, so the table is checked against the
        source tree rather than against a second hand-written copy of itself."""
        found = _modules_that_call_the_helper()
        assert 'fused_memory.server.write_triage' in found, (
            f'the census found {sorted(found)!r}; it is reading the wrong tree '
            'or the import shape changed, and would pass vacuously'
        )

        tabled = {f.module for f in _MIGRATED_FILERS}
        assert found == tabled, (
            f'missing from _MIGRATED_FILERS: {sorted(found - tabled)!r}; in the '
            f'table but no longer calling the helper: {sorted(tabled - found)!r}. '
            'A new pending-anchor-fold filer must get a row, so the levels it '
            'forwards are fenced like every other filer\'s.'
        )

    @pytest.mark.parametrize(('source', 'package'), [
        ('from fused_memory.middleware._folded_escalation import file_folded_escalation',
         'fused_memory.server'),
        ('from fused_memory.middleware import _folded_escalation', 'fused_memory.server'),
        ('import fused_memory.middleware._folded_escalation', 'fused_memory.server'),
        ('import fused_memory.middleware._folded_escalation as fe', 'fused_memory.server'),
        ('from ._folded_escalation import file_folded_escalation', 'fused_memory.middleware'),
        ('from . import _folded_escalation', 'fused_memory.middleware'),
        ('from ..middleware._folded_escalation import file_folded_escalation',
         'fused_memory.services'),
    ])
    def test_the_census_counts_every_import_spelling(self, source, package):
        assert _imports_the_helper(ast.parse(source), package)

    @pytest.mark.parametrize(('source', 'package'), [
        ('from fused_memory.middleware import candidate_key_escalation', 'fused_memory.server'),
        ('from ._folded_escalation_neighbour import x', 'fused_memory.middleware'),
        ('from .middleware import _folded_escalation', 'fused_memory.services'),
    ])
    def test_the_census_ignores_a_lookalike(self, source, package):
        assert not _imports_the_helper(ast.parse(source), package)

    def test_the_table_names_its_exceptions(self):
        not_l1 = {
            f.label: f.escalation_level for f in _MIGRATED_FILERS
            if f.escalation_level != 1
        }
        assert not_l1 == {
            'markup_tripwire residue': 2, 'completion_claim_gate': 0,
        }, (
            'L1 is the house escalation level; residue (L2) and '
            f'completion_claim_gate (L0) are its exceptions. {not_l1!r} says '
            'another filer moved'
        )


# ---------------------------------------------------------------------------
# The anchor-collision regression.
#
# Deliberately NOT decorated with `_needs_escalation`: it reads module
# constants and needs no queue, and an alarm that only fires where the
# escalation package happens to be installed is switched off exactly where
# nobody is looking.
# ---------------------------------------------------------------------------


def _anchors_read_from_their_own_homes() -> dict[str, str]:
    """Every filer's anchor, each imported FROM ITS OWN HOME.

    Reading them from their homes rather than restating the literals here is
    the property that makes a colliding RENAME fail this test: a copy of the
    values in this file would keep passing while production went silent.
    """
    from fused_memory.maintenance.link_heal_executor import (  # noqa: PLC0415
        ADJUDICATOR_ANCHOR as LINK_HEAL_ADJUDICATOR_ANCHOR,
    )
    from fused_memory.maintenance.link_heal_executor import (
        BACKLOG_ANCHOR as LINK_HEAL_BACKLOG_ANCHOR,
    )
    from fused_memory.maintenance.link_heal_executor import (
        CORRECTS_SHARE_ANCHOR as LINK_HEAL_CORRECTS_SHARE_ANCHOR,
    )
    from fused_memory.maintenance.link_heal_executor import (
        MISFILE_SHARE_ANCHOR as LINK_HEAL_MISFILE_SHARE_ANCHOR,
    )
    from fused_memory.maintenance.link_heal_executor import (
        WRITE_FAILURE_ANCHOR as LINK_HEAL_WRITE_FAILURE_ANCHOR,
    )
    from fused_memory.middleware.candidate_key_escalation import (  # noqa: PLC0415
        _ANCHOR_TASK_ID as CANDIDATE_KEY_ANCHOR,
    )
    from fused_memory.middleware.curator_escalator import (
        _ANCHOR_TASK_ID as CURATOR_ANCHOR,
    )
    from fused_memory.middleware.entity_mint_storm_escalator import (  # noqa: PLC0415
        _ANCHOR_TASK_ID as ENTITY_MINT_ANCHOR,
    )
    from fused_memory.middleware.mem0_update_storm_escalator import (  # noqa: PLC0415
        _ANCHOR_TASK_ID as MEM0_UPDATE_ANCHOR,
    )
    from fused_memory.middleware.referent_repair_storm_escalator import (  # noqa: PLC0415
        _ANCHOR_TASK_ID as REFERENT_REPAIR_ANCHOR,
    )
    from fused_memory.middleware.scope_violation_escalator import (  # noqa: PLC0415
        _ANCHOR_TASK_ID as SCOPE_VIOLATION_ANCHOR,
    )
    from fused_memory.middleware.scope_violation_escalator import (
        _BUDGET_MISCONFIG_ANCHOR_TASK_ID as SCOPE_BUDGET_ANCHOR,
    )
    from fused_memory.middleware.scope_violation_escalator import (
        _OVERRIDE_ANCHOR_TASK_ID as SCOPE_OVERRIDE_ANCHOR,
    )
    from fused_memory.middleware.ticket_janitor import (
        _ANCHOR_TASK_ID as TICKET_JANITOR_ANCHOR,
    )
    from fused_memory.middleware.ticket_janitor import (
        curation_backlog_anchor,
    )
    from fused_memory.server.markup_guard import (  # noqa: PLC0415
        _RESIDUE_ANCHOR_TASK_ID as GUARD_RESIDUE_ANCHOR,
    )
    from fused_memory.server.markup_guard import (
        _STORM_ANCHOR_TASK_ID as GUARD_STORM_ANCHOR,
    )
    from fused_memory.server.markup_tripwire import (  # noqa: PLC0415
        _ANCHOR_TASK_ID as TRIPWIRE_ANCHOR,
    )
    from fused_memory.server.markup_tripwire import (
        _RESIDUE_ANCHOR_TASK_ID as TRIPWIRE_RESIDUE_ANCHOR,
    )
    from fused_memory.server.write_triage import (  # noqa: PLC0415
        _ANCHOR_TASK_ID as WRITE_TRIAGE_ANCHOR,
    )
    from fused_memory.services.classification_fallback_alarm import (  # noqa: PLC0415
        ANCHOR_TASK_ID as CLASSIFICATION_FALLBACK_ANCHOR,
    )
    from fused_memory.services.completion_claim_gate import (  # noqa: PLC0415
        _ANCHOR_PREFIX as UNVERIFIED_CLAIM_PREFIX,
    )
    from fused_memory.services.journal_growth_alarm import (  # noqa: PLC0415
        _FILE_SIZE_ANCHOR_TASK_ID as JOURNAL_SIZE_ANCHOR,
    )
    from fused_memory.services.journal_growth_alarm import (
        _INSERT_RATE_ANCHOR_TASK_ID as JOURNAL_RATE_ANCHOR,
    )
    from fused_memory.services.memory_metadata_census import (  # noqa: PLC0415
        _ANCHOR_TASK_ID as CENSUS_ANCHOR_BASE,
    )
    from fused_memory.services.memory_metadata_census import (
        writer_anchor_task_id,
    )

    return {
        # -- the filers that call `file_folded_escalation` ------------------
        'write_triage': WRITE_TRIAGE_ANCHOR,
        'markup_tripwire storm (the SQUATTED one)': TRIPWIRE_ANCHOR,
        'markup_tripwire residue': TRIPWIRE_RESIDUE_ANCHOR,
        'candidate_key_escalation': CANDIDATE_KEY_ANCHOR,
        'referent_repair_storm_escalator': REFERENT_REPAIR_ANCHOR,
        'completion_claim_gate prefix': UNVERIFIED_CLAIM_PREFIX,
        'memory_metadata_census base': CENSUS_ANCHOR_BASE,
        'memory_metadata_census sample writer': writer_anchor_task_id(
            'dark_factory', 'claude-x',
        ),
        'journal_growth_alarm file_size': JOURNAL_SIZE_ANCHOR,
        'journal_growth_alarm insert_rate': JOURNAL_RATE_ANCHOR,
        'classification_fallback_alarm': CLASSIFICATION_FALLBACK_ANCHOR,
        # -- markup_guard, which CALLS the tripwire's filers with anchors of
        #    its own rather than carrying a copy of the skeleton -------------
        'markup_guard storm': GUARD_STORM_ANCHOR,
        'markup_guard residue': GUARD_RESIDUE_ANCHOR,
        # -- the link-heal executor's run escapes (tasks 6181, 6184) ----------
        'link_heal backlog': LINK_HEAL_BACKLOG_ANCHOR,
        'link_heal write-failure': LINK_HEAL_WRITE_FAILURE_ANCHOR,
        'link_heal misfile-share': LINK_HEAL_MISFILE_SHARE_ANCHOR,
        'link_heal corrects-share': LINK_HEAL_CORRECTS_SHARE_ANCHOR,
        'link_heal adjudicator': LINK_HEAL_ADJUDICATOR_ANCHOR,
        # -- the ticket janitor's curation backlog, one anchor per defer reason
        **{
            f'ticket_janitor curation-backlog {reason}': curation_backlog_anchor(reason)
            for reason in DeferReason
        },
        # -- the non-member neighbours. They dedupe on content fingerprints
        #    via `submit_or_dedupe`, not on a pending anchor, so they are NOT
        #    migrating — but they write to the SAME queue, so they can still
        #    squat an anchor and must be in this sweep. ---------------------
        'mem0_update_storm_escalator': MEM0_UPDATE_ANCHOR,
        'entity_mint_storm_escalator': ENTITY_MINT_ANCHOR,
        'scope_violation_escalator': SCOPE_VIOLATION_ANCHOR,
        'scope_violation_escalator override': SCOPE_OVERRIDE_ANCHOR,
        'scope_violation_escalator budget-misconfig': SCOPE_BUDGET_ANCHOR,
        # -- the curator-subsystem filers. Plain `queue.submit`, and a SUBJECT
        #    anchor (the curator bucket) rather than a dedupe anchor. --------
        'curator_escalator': CURATOR_ANCHOR,
        'ticket_janitor curator-bucket': TICKET_JANITOR_ANCHOR,
    }


#: The pairs that are deliberately the same value, asserted equal rather than
#: excluded silently. `markup_guard` hands its residue records to
#: `markup_tripwire.emit_markup_residue_escalation`, so both spell the SAME
#: anchor for the SAME record kind. The curator pair is one SUBJECT anchor,
#: 'task-curator': the curator-subsystem bucket operators query by task id.
#: Neither of those filers dedupes on a pending anchor, so sharing cannot
#: suppress either one. Two homes for one value is a lockstep hazard in the
#: other direction — renaming one would split the series (or the curator
#: history) in half without any test noticing — so each pair is pinned EQUAL
#: below.
_DELIBERATE_ALIASES: tuple[tuple[str, str], ...] = (
    ('markup_tripwire residue', 'markup_guard residue'),
    ('curator_escalator', 'ticket_janitor curator-bucket'),
)


class TestNoTwoFilersShareAnAnchor:
    """A SQUATTED anchor is suppressed indefinitely, and reads as calm.

    Measured incident: the L1 escalation watcher files its own cluster
    records under the `markup-tripwire` anchor and SQUATS it — the tripwire
    filed nothing 2026-08-16..2026-08-19 while 41 rejections occurred, all
    17 records sitting at dedupe_count 0. A filer that dedupes against an
    anchor somebody else keeps open never files again, and the resulting
    silence is indistinguishable from health.

    That incident is why `emit_markup_storm_escalation` grew its
    `anchor_task_id` parameter (see its docstring), and it is why no filer
    may share an anchor with any other. Asserted against every filer's
    constants IMPORTED FROM THEIR OWN HOMES, so a future rename that collides
    is caught here rather than in production silence. PAIRWISE, so a
    collision between any two filers fails, not only one against the rest.
    """

    def test_every_pair_of_anchors_is_distinct(self) -> None:
        import itertools  # noqa: PLC0415

        anchors = _anchors_read_from_their_own_homes()
        aliased = {frozenset(pair) for pair in _DELIBERATE_ALIASES}

        for (label_a, value_a), (label_b, value_b) in itertools.combinations(
            anchors.items(), 2,
        ):
            if frozenset((label_a, label_b)) in aliased:
                continue
            assert value_a != value_b, (
                f'{label_a} and {label_b} both file under the anchor '
                f'{value_a!r}. A filer deduping against an anchor another '
                'party keeps open is suppressed indefinitely, and that '
                'silence reads as calm — give one of them an anchor of its '
                'own, or add the pair to _DELIBERATE_ALIASES if they really '
                'are one record kind with two spellings.'
            )

    def test_the_deliberate_aliases_stay_in_lockstep(self) -> None:
        """Two homes for one value is the hazard in the other direction:
        renaming one would split the series in half with nothing noticing."""
        anchors = _anchors_read_from_their_own_homes()

        for label_a, label_b in _DELIBERATE_ALIASES:
            assert anchors[label_a] == anchors[label_b], (
                f'{label_a} and {label_b} name the SAME record kind and must '
                f'stay equal: got {anchors[label_a]!r} and {anchors[label_b]!r}'
            )

    def test_the_sweep_covers_every_filer_on_the_queue(self) -> None:
        """Anti-vacuity: a pairwise sweep passes trivially if a filer is left
        out."""
        anchors = _anchors_read_from_their_own_homes()

        for required in (
            'write_triage',
            'markup_tripwire storm (the SQUATTED one)',
            'markup_tripwire residue',
            'candidate_key_escalation',
            'referent_repair_storm_escalator',
            'completion_claim_gate prefix',
            'memory_metadata_census base',
            'journal_growth_alarm file_size',
            'journal_growth_alarm insert_rate',
            'classification_fallback_alarm',
            'mem0_update_storm_escalator',
            'entity_mint_storm_escalator',
            'scope_violation_escalator',
            'link_heal backlog',
            'link_heal write-failure',
            'link_heal misfile-share',
            'link_heal corrects-share',
            'link_heal adjudicator',
            'curator_escalator',
            'ticket_janitor curator-bucket',
        ):
            assert required in anchors, (
                f'{required!r} files into the same queue but is absent from '
                'the anchor sweep'
            )
        assert len(anchors) >= 25, (
            f'the sweep shrank to {len(anchors)} entries; a filer was dropped '
            'rather than renamed'
        )


@_needs_escalation
class TestTheFiledAnchorAndTheLookupAreTheSame:
    """Filing under one anchor while deduping against another is the bug.

    It would produce a record nobody dedupes against (unbounded duplicates)
    or a lookup nobody files under (permanent suppression). Pinned by
    capturing the anchor the dedup read is called with and comparing it to
    the `task_id` that actually landed. Both are threaded from the single
    `anchor_task_id` parameter, so one test covers every filer that calls the
    helper.
    """

    def test_the_dedup_lookup_uses_the_anchor_that_is_filed(
        self, tmp_path, monkeypatch,
    ):
        seen: list = []
        real_get_by_task = _folded_escalation.EscalationQueue.get_by_task

        def _spy(self, task_id, status=None):
            seen.append(task_id)
            return real_get_by_task(self, task_id, status=status)

        monkeypatch.setattr(_folded_escalation.EscalationQueue, 'get_by_task', _spy)
        _emit(tmp_path, anchor_task_id='a-distinctive-anchor')

        filed = _filed(tmp_path)
        assert len(filed) == 1
        assert seen == [filed[0]['task_id']], (
            f'deduped against {seen!r} but filed under {filed[0]["task_id"]!r}'
        )
        assert seen == ['a-distinctive-anchor']
