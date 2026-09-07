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

import json
import logging

import pytest

from fused_memory.middleware import _folded_escalation
from fused_memory.middleware._folded_escalation import file_folded_escalation

pytestmark = pytest.mark.skipif(
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

        with caplog.at_level('ERROR'):
            assert _emit(tmp_path) is None
        assert caplog.records, 'a swallowed failure must still be visible'

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
        assert caplog.records, 'a no-op alarm must still say so'

    def test_a_none_project_root_returns_none_quietly(self, tmp_path, caplog):
        """`write_triage` takes `project_root: str | None`; None means there is
        no project queue to file into, which is not an error."""
        with caplog.at_level('DEBUG'):
            assert _emit(tmp_path, project_root=None) is None
        assert not (tmp_path / 'data' / 'escalations').exists()
