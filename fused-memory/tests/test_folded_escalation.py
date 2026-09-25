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

import importlib
import json
import logging
from collections.abc import Callable
from typing import Any, NamedTuple

import pytest

from fused_memory.middleware import _folded_escalation
from fused_memory.middleware._folded_escalation import file_folded_escalation

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
_CATEGORY = 'folded_escalation_test'

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

    def test_without_the_escalation_package_it_no_ops(self, tmp_path, monkeypatch, caplog):
        """The minimal-env path: logged, nothing filed, `None` returned. Every
        caller must behave identically whether or not the optional `escalation`
        workspace package is installed."""
        monkeypatch.setattr(_folded_escalation, 'HAS_ESCALATION', False)

        with caplog.at_level('DEBUG'):
            result = _emit(tmp_path)

        assert result is None
        assert not (tmp_path / 'data' / 'escalations').exists()
        assert any(r.levelno == logging.DEBUG for r in caplog.records), (
            'a no-op alarm must still say so, at the helper DEFAULT level — '
            'see TestTheEmittedLevelIsACallerConcern for why that default is '
            'DEBUG and how a caller opts out of it'
        )

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


class TestTheEmittedLevelIsACallerConcern:
    """A never-raise alarm path is exactly where a silent level downgrade is
    most costly, because the ONLY symptom is absence of output.

    Every other arm of this helper announces a downgrade somehow — a wrong
    `severity` shows up in the filed record, a wrong anchor shows up in the
    pairwise sweep. A level that drops below the operator's threshold produces
    no record, no file and no exception; the alarm simply stops being seen,
    and "no output" is indistinguishable from "nothing to report". That is the
    same silence this module's own docstring argues against, one layer down.

    So the levels are PINNED rather than left to whatever `logger.<method>`
    the body happens to call. The package-unavailable arm defaults to DEBUG
    and a caller may raise it; the fold arm defaults to INFO and a caller
    replaces it wholesale via `on_fold`.

    Asserted on `levelno` against the `logging.*` constants, never on
    `levelname` strings: a level is an ordered numeric fact, and comparing its
    rendered name would be a meaningful string standing in for structured data
    (heuristic 12).
    """

    def test_the_package_unavailable_arm_defaults_to_debug(
        self, tmp_path, monkeypatch, caplog,
    ):
        """Six of the seven migrated filers treat an absent optional
        `escalation` package as a DEBUG detail, so that is the default — NOT a
        judgement that DEBUG is the right house level for a lost alarm."""
        monkeypatch.setattr(_folded_escalation, 'HAS_ESCALATION', False)

        with caplog.at_level(logging.DEBUG):
            assert _emit(tmp_path) is None

        assert [r.levelno for r in caplog.records] == [logging.DEBUG]

    def test_a_caller_can_raise_the_package_unavailable_arm(
        self, tmp_path, monkeypatch, caplog,
    ):
        """`referent_repair_storm_escalator` treats a missing package as a LOST
        ALARM rather than a detail: a repair storm is a sustained scanner or
        resolver regression, so in an env without the optional package the
        DEBUG default emits nothing at all at the default threshold."""
        monkeypatch.setattr(_folded_escalation, 'HAS_ESCALATION', False)

        with caplog.at_level(logging.DEBUG):
            result = _emit(tmp_path, no_escalation_level=logging.WARNING)

        assert [r.levelno for r in caplog.records] == [logging.WARNING]
        assert result is None, 'raising the level must not change the verdict'
        assert not (tmp_path / 'data' / 'escalations').exists(), (
            'raising the level must not start filing where the package is gone'
        )

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
    #: Dotted path of the filer's OWN module. The spy is installed HERE, not on
    #: `_folded_escalation`, because that is where the name the filer body
    #: resolves actually lives — patching the helper's home would prove nothing
    #: about what this caller forwards.
    module: str
    fire: Callable[[Any, str], object]
    #: The level this filer's package-unavailable arm must be emitted at.
    no_escalation_level: int
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


def _unverified_claim(module: Any, project_root: str) -> object:
    return module.emit_unverified_claim_escalation(
        project_root,
        {'claims': [{'subject': 'task', 'ref': '5422', 'kind': 'task',
                     'project_id': 'dark_factory', 'status': 'mismatch',
                     'observed': 'in-progress', 'text': 'task 5422 has been applied'}]},
    )


#: THE SINGLE POINT OF TRUTH (heuristic 11) for the two levels each migrated
#: filer forwards: the log level of its package-unavailable arm, and the
#: escalation level its record is born at.
#:
#: Both are silent when they drift — a log line that drops below threshold and
#: a record that waits for the orphan-L0 reaper both look like "nothing to
#: report" — so they are fenced here rather than left implicit in seven call
#: sites. Changing what a filer forwards means editing this table, which is the
#: point: it cannot land unnoticed.
_MIGRATED_FILERS: tuple[_Filer, ...] = (
    _Filer(
        'write_triage',
        'fused_memory.server.write_triage',
        lambda m, root: m.emit_triage_fail_open_storm_escalation(
            root, {'count': 5, 'window_seconds': 60},
        ),
        logging.DEBUG,
        1,
    ),
    _Filer(
        'markup_tripwire storm',
        'fused_memory.server.markup_tripwire',
        _storm,
        logging.DEBUG,
        1,
    ),
    _Filer(
        'markup_tripwire residue',
        'fused_memory.server.markup_tripwire',
        _residue,
        logging.DEBUG,
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
        logging.DEBUG,
        1,
    ),
    # THE ONE EXCEPTION, and the defect this table exists to fence. A repair
    # storm is a sustained scanner/resolver regression against a measured
    # ~0.22% base rate, so an env without the optional `escalation` package
    # emitting nothing at the default threshold is a LOST ALARM, not a detail.
    # Merge-base 6f9cddb0bb logged it at WARNING; consolidation flattened it to
    # DEBUG and 417 passing tests said nothing.
    _Filer(
        'referent_repair_storm_escalator',
        'fused_memory.middleware.referent_repair_storm_escalator',
        lambda m, root: m.emit_referent_repair_storm_escalation(
            root, project_id='dark_factory', streak=10, threshold=10,
            repairs=1, records=[],
        ),
        logging.WARNING,
        1,
    ),
    _Filer(
        'completion_claim_gate',
        'fused_memory.services.completion_claim_gate',
        _unverified_claim,
        logging.DEBUG,
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
        logging.DEBUG,
        1,
    ),
)


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


class TestEveryFilerPinsItsForwardedLevels:
    """The house levels, their exceptions, and a fence against a new answer.

    Deliberately NOT decorated with `_needs_escalation`: the spy replaces
    `file_folded_escalation` outright, so no queue is ever built — and a level
    regression that only fails where the escalation package happens to be
    installed is an alarm switched off exactly where nobody is looking.

    Each forwarded value is read with `.get` and the helper's own default, so
    a row passes whether the caller omits the keyword or passes it explicitly:
    the property under test is the EFFECTIVE level, not the call spelling.
    """

    @pytest.mark.parametrize(
        'filer', _MIGRATED_FILERS, ids=[f.label for f in _MIGRATED_FILERS],
    )
    def test_the_forwarded_no_escalation_level_matches_the_table(
        self, filer, tmp_path, monkeypatch,
    ):
        seen = _forwarded_kwargs(filer, tmp_path, monkeypatch)

        forwarded = seen.get('no_escalation_level', logging.DEBUG)
        assert forwarded == filer.no_escalation_level, (
            f'{filer.label} forwards no_escalation_level={forwarded!r}, table '
            f'says {filer.no_escalation_level!r}. If the change is deliberate, '
            'edit the table and say why in the row; if not, a never-raise alarm '
            'just changed how loudly it fails, and the only symptom would have '
            'been absence of output.'
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

    def test_the_table_covers_every_folded_filer_and_names_its_exceptions(self):
        """Anti-vacuity: a parametrized fence passes trivially for a filer that
        is simply absent from the table, which is how the predecessor
        anchor sweep missed two of them."""
        assert {f.label for f in _MIGRATED_FILERS} == {
            'write_triage',
            'markup_tripwire storm',
            'markup_tripwire residue',
            'candidate_key_escalation',
            'referent_repair_storm_escalator',
            'completion_claim_gate',
            'memory_metadata_census',
        }
        raised = {
            f.label for f in _MIGRATED_FILERS
            if f.no_escalation_level != logging.DEBUG
        }
        assert raised == {'referent_repair_storm_escalator'}, (
            'DEBUG is the house level and referent_repair is its ONE deliberate '
            f'exception; {raised!r} says a second answer arrived without the '
            'house-level question being settled'
        )
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
# The anchor-collision regression, generalised (task 4854).
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
    from fused_memory.middleware.candidate_key_escalation import (  # noqa: PLC0415
        _ANCHOR_TASK_ID as CANDIDATE_KEY_ANCHOR,
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
    from fused_memory.services.completion_claim_gate import (  # noqa: PLC0415
        _ANCHOR_PREFIX as UNVERIFIED_CLAIM_PREFIX,
    )
    from fused_memory.services.memory_metadata_census import (  # noqa: PLC0415
        _ANCHOR_TASK_ID as CENSUS_ANCHOR_BASE,
    )
    from fused_memory.services.memory_metadata_census import (
        writer_anchor_task_id,
    )

    return {
        # -- the seven filers task 4854 folded into `_folded_escalation` -----
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
        # -- markup_guard, which CALLS the tripwire's filers with anchors of
        #    its own rather than carrying a copy of the skeleton -------------
        'markup_guard storm': GUARD_STORM_ANCHOR,
        'markup_guard residue': GUARD_RESIDUE_ANCHOR,
        # -- the non-member neighbours. They dedupe on content fingerprints
        #    via `submit_or_dedupe`, not on a pending anchor, so they are NOT
        #    migrating — but they write to the SAME queue, so they can still
        #    squat an anchor and must be in this sweep. ---------------------
        'mem0_update_storm_escalator': MEM0_UPDATE_ANCHOR,
        'entity_mint_storm_escalator': ENTITY_MINT_ANCHOR,
        'scope_violation_escalator': SCOPE_VIOLATION_ANCHOR,
        'scope_violation_escalator override': SCOPE_OVERRIDE_ANCHOR,
        'scope_violation_escalator budget-misconfig': SCOPE_BUDGET_ANCHOR,
    }


#: The ONE pair that is deliberately the same value, asserted equal rather than
#: excluded silently. `markup_guard` hands its residue records to
#: `markup_tripwire.emit_markup_residue_escalation`, so both spell the SAME
#: anchor for the SAME record kind. Two homes for one value is a lockstep
#: hazard in the other direction — renaming one would split the series in half
#: without any test noticing — so the pair is pinned EQUAL below.
_DELIBERATE_ALIASES: tuple[tuple[str, str], ...] = (
    ('markup_tripwire residue', 'markup_guard residue'),
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
    is caught here rather than in production silence.

    GENERALISED FROM one-vs-seven to PAIRWISE by task 4854. The narrower
    predecessor lived in `tests/server/test_write_triage.py` and compared
    write_triage's anchor against its siblings only, so a collision between
    any two OTHER filers passed it unnoticed — and it named neither
    `completion_claim_gate` nor `memory_metadata_census`, the two filers that
    task discovered.
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
        out, which is exactly how the predecessor missed two of them."""
        anchors = _anchors_read_from_their_own_homes()

        for required in (
            'write_triage',
            'markup_tripwire storm (the SQUATTED one)',
            'markup_tripwire residue',
            'candidate_key_escalation',
            'referent_repair_storm_escalator',
            'completion_claim_gate prefix',
            'memory_metadata_census base',
            'mem0_update_storm_escalator',
            'entity_mint_storm_escalator',
            'scope_violation_escalator',
        ):
            assert required in anchors, (
                f'{required!r} files into the same queue but is absent from '
                'the anchor sweep'
            )
        assert len(anchors) >= 15, (
            f'the sweep shrank to {len(anchors)} entries; a filer was dropped '
            'rather than renamed'
        )


@_needs_escalation
class TestTheFiledAnchorAndTheLookupAreTheSame:
    """Filing under one anchor while deduping against another is the bug.

    It would produce a record nobody dedupes against (unbounded duplicates)
    or a lookup nobody files under (permanent suppression). Pinned by
    capturing the anchor the dedup read is called with and comparing it to
    the `task_id` that actually landed.

    MOVED HERE by task 4854 from `tests/server/test_write_triage.py`, where it
    could only ever prove the property for one caller. Both uses are now
    threaded from the SINGLE `anchor_task_id` parameter of
    `file_folded_escalation`, so this is where the property lives — and one
    test now covers every filer that calls it.
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
