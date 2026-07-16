"""Tests for correct_found_on_main_backlog.py.

Loaded via importlib so the script (not on PYTHONPATH) can be tested without
sys.path pollution — mirrors the pattern in
test_audit_found_on_main_provenance.py / test_audit_duplicate_tasks.py.
"""
from __future__ import annotations

import importlib.util
import json
import types
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'correct_found_on_main_backlog.py'


def _load_module() -> types.ModuleType:
    """Load correct_found_on_main_backlog.py from its file path.

    The module is registered in sys.modules under its name so that
    @dataclass and other reflection-based decorators work correctly
    (they call sys.modules.get(cls.__module__)).
    """
    import sys  # noqa: PLC0415

    mod_name = 'correct_found_on_main_backlog'
    spec = importlib.util.spec_from_file_location(mod_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module  # required for @dataclass __module__ lookup
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


_mod = _load_module()
Correction = _mod.Correction
REOPEN_DISPOSITIONS = _mod.REOPEN_DISPOSITIONS
BENIGN_DISPOSITIONS = _mod.BENIGN_DISPOSITIONS
plan_corrections = _mod.plan_corrections
ACTION_REOPEN = _mod.ACTION_REOPEN
ACTION_ANNOTATE = _mod.ACTION_ANNOTATE
LABEL_REOPENED = _mod.LABEL_REOPENED
LABEL_REVIEWED_BENIGN = _mod.LABEL_REVIEWED_BENIGN
LABEL_PRESUMED_BENIGN_HISTORICAL = _mod.LABEL_PRESUMED_BENIGN_HISTORICAL
apply_corrections = _mod.apply_corrections


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _detail(
    task_id: str,
    verdict: str,
    *,
    commit: str = 'a' * 40,
    commit_subject: str = '',
    reasons: list[str] | None = None,
) -> dict:
    """Build a per-task detail dict shaped like build_audit_report's ``tasks`` entries."""
    return {
        'task_id': task_id,
        'verdict': verdict,
        'commit': commit,
        'commit_subject': commit_subject,
        'reasons': list(reasons) if reasons is not None else [],
    }


def _report(tasks: list[dict], ref: str = 'main') -> dict:
    """Build a report dict shaped like build_audit_report's return value."""
    return {'ref': ref, 'dry_run': True, 'total': len(tasks), 'tasks': tasks}


# ===========================================================================
# Step-1/2: plan_corrections — reviewed-disposition data sanity
# ===========================================================================

class TestReviewedDispositionData:
    """The reviewed disposition maps carry the two highest-signal task ids."""

    def test_1175_is_a_reopen_disposition(self):
        assert '1175' in REOPEN_DISPOSITIONS
        assert isinstance(REOPEN_DISPOSITIONS['1175'], str)
        assert REOPEN_DISPOSITIONS['1175']

    def test_2273_is_a_benign_disposition(self):
        assert '2273' in BENIGN_DISPOSITIONS
        assert isinstance(BENIGN_DISPOSITIONS['2273'], str)
        assert BENIGN_DISPOSITIONS['2273']


# ===========================================================================
# Step-1/2: plan_corrections — reopen routing
# ===========================================================================

class TestPlanCorrectionsReopen:
    """A task_id present in REOPEN_DISPOSITIONS routes to action=='reopen'."""

    def test_reopen_disposition_task_routes_to_reopen_action(self):
        reasons = ['declared file(s) missing from the ref HEAD: fused-memory/tests/x.py']
        report = _report([_detail('1175', 'reverted', reasons=reasons)])
        corrections = plan_corrections(report)
        assert len(corrections) == 1
        correction = corrections[0]
        assert correction.task_id == '1175'
        assert correction.action == ACTION_REOPEN
        assert correction.label == LABEL_REOPENED
        assert correction.ref == 'main'
        # reopen_reason carries the reviewed evidence, and the audit's own
        # reasons are preserved (not discarded) with the evidence appended.
        assert correction.reopen_reason == REOPEN_DISPOSITIONS['1175']
        assert correction.reasons[:-1] == reasons
        assert correction.reasons[-1] == REOPEN_DISPOSITIONS['1175']


# ===========================================================================
# Step-1/2: plan_corrections — benign routing
# ===========================================================================

class TestPlanCorrectionsBenign:
    """A task_id present in BENIGN_DISPOSITIONS routes to action=='annotate',
    label=='reviewed_benign'."""

    def test_benign_disposition_task_routes_to_annotate_reviewed_benign(self):
        reasons = ["none of the declared file(s) ['x.py'] appear in the cited commit's diff"]
        report = _report([_detail('2273', 'deliverable_absent', reasons=reasons)])
        corrections = plan_corrections(report)
        assert len(corrections) == 1
        correction = corrections[0]
        assert correction.task_id == '2273'
        assert correction.action == ACTION_ANNOTATE
        assert correction.label == LABEL_REVIEWED_BENIGN
        assert correction.reopen_reason is None
        assert correction.reasons[:-1] == reasons
        assert correction.reasons[-1] == BENIGN_DISPOSITIONS['2273']


# ===========================================================================
# Step-1/2: plan_corrections — default fallback (unlisted, non-ok verdicts)
# ===========================================================================

class TestPlanCorrectionsDefaultFallback:
    """Any unlisted non-ok verdict is annotated presumed_benign_historical,
    carrying the audit reasons forward untouched."""

    def test_unlisted_misattributed_task_annotated_presumed_benign_historical(self):
        reasons = ['commit message cites task(s) 77, not task 9999 — likely proof of a '
                   'different task, not this one']
        report = _report([_detail('9999', 'misattributed', reasons=reasons)])
        corrections = plan_corrections(report)
        assert len(corrections) == 1
        correction = corrections[0]
        assert correction.task_id == '9999'
        assert correction.action == ACTION_ANNOTATE
        assert correction.label == LABEL_PRESUMED_BENIGN_HISTORICAL
        assert correction.reopen_reason is None
        assert correction.reasons == reasons

    def test_unlisted_unverifiable_task_is_also_annotated_not_skipped(self):
        """unverifiable is NOT skipped here, unlike the audit tool's --apply
        (_FLAGGED_VERDICTS), per task 2667's explicit scope."""
        reasons = ['no declared files and the commit message does not cite this task — '
                   'nothing to verify the found_on_main claim against']
        report = _report([_detail('8888', 'unverifiable', reasons=reasons)])
        corrections = plan_corrections(report)
        assert len(corrections) == 1
        correction = corrections[0]
        assert correction.task_id == '8888'
        assert correction.action == ACTION_ANNOTATE
        assert correction.label == LABEL_PRESUMED_BENIGN_HISTORICAL
        assert correction.reasons == reasons

    def test_unlisted_commit_not_on_main_is_also_annotated(self):
        reasons = ['cited commit is not an ancestor of the audited ref']
        report = _report([_detail('7000', 'commit_not_on_main', reasons=reasons)])
        corrections = plan_corrections(report)
        assert len(corrections) == 1
        assert corrections[0].label == LABEL_PRESUMED_BENIGN_HISTORICAL


# ===========================================================================
# Step-1/2: plan_corrections — ok verdicts are skipped
# ===========================================================================

class TestPlanCorrectionsSkipsOk:
    """An 'ok' verdict produces no Correction at all."""

    def test_ok_verdict_produces_no_correction(self):
        report = _report([_detail('7777', 'ok', reasons=[])])
        assert plan_corrections(report) == []

    def test_mixed_report_only_ok_is_skipped(self):
        tasks = [
            _detail('1175', 'reverted', reasons=['x']),
            _detail('2273', 'deliverable_absent', reasons=['y']),
            _detail('9999', 'misattributed', reasons=['z']),
            _detail('7777', 'ok', reasons=[]),
        ]
        report = _report(tasks)
        corrections = plan_corrections(report)
        assert [c.task_id for c in corrections] == ['1175', '2273', '9999']


# ===========================================================================
# Step-1/2: plan_corrections — Correction dataclass shape
# ===========================================================================

class TestCorrectionDataclassShape:
    """Every Correction exposes the fields apply_corrections will consume."""

    def test_correction_has_expected_fields(self):
        report = _report([_detail('9999', 'misattributed', reasons=['z'])])
        [correction] = plan_corrections(report)
        assert isinstance(correction, Correction)
        assert correction.task_id == '9999'
        assert correction.action in (ACTION_REOPEN, ACTION_ANNOTATE)
        assert isinstance(correction.label, str)
        assert isinstance(correction.reasons, list)
        assert correction.reopen_reason is None or isinstance(correction.reopen_reason, str)
        assert isinstance(correction.ref, str)


# ===========================================================================
# Step-3+: apply_corrections — fixtures
# ===========================================================================

def _correction(
    task_id: str,
    action: str = ACTION_ANNOTATE,
    *,
    label: str = LABEL_PRESUMED_BENIGN_HISTORICAL,
    ref: str = 'main',
    reasons: list[str] | None = None,
    reopen_reason: str | None = None,
) -> Correction:
    """Build a Correction directly, bypassing plan_corrections, for
    apply_corrections-focused tests."""
    return Correction(
        task_id=task_id, action=action, label=label, ref=ref,
        reasons=list(reasons) if reasons is not None else ['some audit reason'],
        reopen_reason=reopen_reason,
    )


class FakeCorrectionsBackend:
    """Recording fake backend for apply_corrections tests.

    - ``update_task`` calls are recorded (metadata/tag included, so tests
      can decode the persisted x_provenance_audit payload); raises for any
      task_id in ``fail_update_for``.
    - ``set_task_status`` calls are recorded; returns
      ``status_results[task_id]`` when configured, else a synthesized
      success DTO shaped like the real backend's ``SetTaskStatusResult``
      (no ``'success'`` key on success — mirrors the real wire shape).
    - ``get_task`` calls are recorded; returns ``get_task_results[task_id]``
      when configured, else ``{'status': 'pending'}`` — the common
      happy-path shape for a post-reopen re-read.
    """

    def __init__(
        self,
        *,
        status_results: dict[str, dict] | None = None,
        get_task_results: dict[str, dict] | None = None,
        fail_update_for: set[str] | None = None,
    ):
        self.status_results = status_results or {}
        self.get_task_results = get_task_results or {}
        self.fail_update_for = fail_update_for or set()
        self.update_calls: list[dict] = []
        self.status_calls: list[dict] = []
        self.get_task_calls: list[dict] = []

    async def update_task(self, task_id, project_root, metadata=None, tag=None, **kwargs):
        if task_id in self.fail_update_for:
            raise RuntimeError(f'simulated update_task failure for {task_id}')
        self.update_calls.append({
            'task_id': task_id, 'project_root': project_root,
            'metadata': metadata, 'tag': tag, 'kwargs': kwargs,
        })
        return {'success': True}

    async def set_task_status(self, task_id, status, project_root, tag=None):
        self.status_calls.append({
            'task_id': task_id, 'status': status, 'project_root': project_root, 'tag': tag,
        })
        if task_id in self.status_results:
            return self.status_results[task_id]
        return {
            'message': f'Successfully updated 1 task(s) to "{status}"',
            'tasks': [{'taskId': task_id, 'oldStatus': 'done', 'newStatus': status}],
        }

    async def get_task(self, task_id, project_root, tag=None):
        self.get_task_calls.append({
            'task_id': task_id, 'project_root': project_root, 'tag': tag,
        })
        if task_id in self.get_task_results:
            return self.get_task_results[task_id]
        return {'status': 'pending'}


# ===========================================================================
# Step-3/4: apply_corrections — dry run
# ===========================================================================

@pytest.mark.asyncio
class TestApplyCorrectionsDryRun:
    """apply=False performs no backend calls and reports the intended plan."""

    async def test_dry_run_makes_no_backend_calls(self):
        corrections = [
            _correction('1175', ACTION_REOPEN, label=LABEL_REOPENED, reopen_reason='evidence'),
            _correction('9999', ACTION_ANNOTATE, label=LABEL_PRESUMED_BENIGN_HISTORICAL),
        ]
        backend = FakeCorrectionsBackend()
        summary = await apply_corrections(backend, '/proj', corrections, apply=False)
        assert backend.status_calls == []
        assert backend.update_calls == []
        assert backend.get_task_calls == []
        assert summary['dry_run'] is True

    async def test_dry_run_lists_intended_task_id_and_action_per_correction(self):
        corrections = [
            _correction('1175', ACTION_REOPEN, label=LABEL_REOPENED, reopen_reason='evidence'),
            _correction('9999', ACTION_ANNOTATE, label=LABEL_PRESUMED_BENIGN_HISTORICAL),
        ]
        backend = FakeCorrectionsBackend()
        summary = await apply_corrections(backend, '/proj', corrections, apply=False)
        planned = summary['planned']
        assert {(p['task_id'], p['action']) for p in planned} == {
            ('1175', ACTION_REOPEN), ('9999', ACTION_ANNOTATE),
        }


# ===========================================================================
# Step-5/6: apply_corrections — annotate apply path
# ===========================================================================

@pytest.mark.asyncio
class TestApplyCorrectionsAnnotate:
    """apply=True on an annotate correction writes a single, non-destructive
    x_provenance_audit merge — never a status write."""

    async def test_annotate_writes_single_non_destructive_update_task_call(self):
        correction = _correction(
            '9999', ACTION_ANNOTATE, label=LABEL_PRESUMED_BENIGN_HISTORICAL, ref='main',
            reasons=['commit message cites task(s) 77, not task 9999'],
        )
        backend = FakeCorrectionsBackend()
        summary = await apply_corrections(backend, '/proj', [correction], apply=True)

        # No status write for an annotate-only correction.
        assert backend.status_calls == []

        assert len(backend.update_calls) == 1
        call = backend.update_calls[0]
        assert call['task_id'] == '9999'
        assert call['project_root'] == '/proj'
        # metadata_mode left at its 'merge' default — never forced to
        # 'replace', which would otherwise wipe done_provenance/files.
        assert call['kwargs'] == {}

        payload = json.loads(call['metadata'])
        annotation = payload['x_provenance_audit']
        assert annotation['label'] == LABEL_PRESUMED_BENIGN_HISTORICAL
        assert annotation['reasons'] == correction.reasons
        assert annotation['ref'] == 'main'
        assert annotation.get('audited_at')

        assert summary['annotated'] == 1
        assert summary['needs_human_review'] == ['9999']


# ===========================================================================
# Step-7/8: apply_corrections — reopen happy path
# ===========================================================================

@pytest.mark.asyncio
class TestApplyCorrectionsReopenHappyPath:
    """apply=True on a reopen correction flips status to pending, verifies
    the flip persisted, and also writes an audit-trail annotation."""

    async def test_reopen_happy_path(self):
        correction = _correction(
            '1175', ACTION_REOPEN, label=LABEL_REOPENED, ref='main',
            reasons=['declared file(s) missing from the ref HEAD: fused-memory/tests/x.py'],
            reopen_reason=REOPEN_DISPOSITIONS['1175'],
        )
        # Defaults already shape the happy path: set_task_status returns a
        # success DTO, get_task's post-write re-read reports 'pending'.
        backend = FakeCorrectionsBackend()
        summary = await apply_corrections(backend, '/proj', [correction], apply=True)

        assert len(backend.status_calls) == 1
        status_call = backend.status_calls[0]
        assert status_call['task_id'] == '1175'
        assert status_call['status'] == 'pending'
        assert status_call['project_root'] == '/proj'

        assert len(backend.get_task_calls) == 1
        assert backend.get_task_calls[0]['task_id'] == '1175'

        assert summary['reopened'] == 1
        assert '1175' not in summary['reopen_failed']
        assert summary['errors'] == 0

        # An audit-trail annotation is also written on a successful reopen.
        assert len(backend.update_calls) == 1
        call = backend.update_calls[0]
        assert call['task_id'] == '1175'
        payload = json.loads(call['metadata'])
        annotation = payload['x_provenance_audit']
        assert annotation['label'] == LABEL_REOPENED
        assert annotation['reopen_reason'] == REOPEN_DISPOSITIONS['1175']


# ===========================================================================
# Step-9/10: apply_corrections — reopen silent-write-failure guard (task 1175
# regression: a "successful" status write that doesn't actually persist)
# ===========================================================================

@pytest.mark.asyncio
class TestApplyCorrectionsReopenNotPersisted:
    """A reopen whose flip did not verifiably persist must be recorded
    loudly (reopen_failed + errors), never silently trusted — this is the
    exact task-1175 regression this script exists to guard against. The
    batch must keep processing subsequent corrections (loud, not aborting)."""

    async def test_not_persisted_dto_is_recorded_reopen_failed_and_batch_continues(self):
        reopen_correction = _correction(
            '1175', ACTION_REOPEN, label=LABEL_REOPENED, ref='main',
            reasons=['declared file(s) missing from the ref HEAD: fused-memory/tests/x.py'],
            reopen_reason=REOPEN_DISPOSITIONS['1175'],
        )
        annotate_correction = _correction(
            '9999', ACTION_ANNOTATE, label=LABEL_PRESUMED_BENIGN_HISTORICAL,
        )
        backend = FakeCorrectionsBackend(
            status_results={
                '1175': {
                    'success': False,
                    'error': 'status_write_not_persisted',
                    'task_id': '1175',
                    'requested_status': 'pending',
                    'actual_status': 'done',
                },
            },
        )
        summary = await apply_corrections(
            backend, '/proj', [reopen_correction, annotate_correction], apply=True,
        )

        assert '1175' in summary['reopen_failed']
        assert summary['errors'] >= 1
        assert summary['reopened'] == 0

        # No audit-trail annotation is written for the not-persisted reopen
        # itself — loud failure, not a paper trail implying success.
        assert all(call['task_id'] != '1175' for call in backend.update_calls)

        # The batch is NOT aborted: the following annotate correction still
        # gets applied (loud-but-non-aborting).
        assert summary['annotated'] == 1
        assert any(call['task_id'] == '9999' for call in backend.update_calls)

    async def test_reread_still_shows_done_is_recorded_reopen_failed(self):
        """set_task_status reports success, but the independent get_task
        re-read still shows 'done' — the exact silent-non-persist shape
        task 1175 suffered under prior reconciliation reopens."""
        reopen_correction = _correction(
            '1175', ACTION_REOPEN, label=LABEL_REOPENED, ref='main',
            reopen_reason=REOPEN_DISPOSITIONS['1175'],
        )
        backend = FakeCorrectionsBackend(
            get_task_results={'1175': {'status': 'done'}},
        )
        summary = await apply_corrections(backend, '/proj', [reopen_correction], apply=True)

        assert len(backend.status_calls) == 1
        assert len(backend.get_task_calls) == 1

        assert '1175' in summary['reopen_failed']
        assert summary['errors'] >= 1
        assert summary['reopened'] == 0
        assert all(call['task_id'] != '1175' for call in backend.update_calls)


# ===========================================================================
# Step-11/12: apply_corrections — per-op error isolation
# ===========================================================================

@pytest.mark.asyncio
class TestApplyCorrectionsErrorIsolation:
    """One correction's backend failure must not abort the batch — mirrors
    apply_audit_annotations' per-op try/except isolation guarantee."""

    async def test_one_failing_update_task_does_not_abort_the_batch(self):
        failing_annotate = _correction(
            '1111', ACTION_ANNOTATE, label=LABEL_PRESUMED_BENIGN_HISTORICAL,
        )
        ok_annotate = _correction(
            '2222', ACTION_ANNOTATE, label=LABEL_PRESUMED_BENIGN_HISTORICAL,
        )
        ok_reopen = _correction(
            '1175', ACTION_REOPEN, label=LABEL_REOPENED,
            reopen_reason=REOPEN_DISPOSITIONS['1175'],
        )
        backend = FakeCorrectionsBackend(fail_update_for={'1111'})

        # apply_corrections itself must not raise, even though one of the
        # three corrections' update_task call raises RuntimeError.
        summary = await apply_corrections(
            backend, '/proj', [failing_annotate, ok_annotate, ok_reopen], apply=True,
        )

        # The raising correction is counted as an error...
        assert summary['errors'] >= 1
        # ...but the OTHER corrections still applied successfully.
        assert summary['annotated'] == 1
        assert summary['reopened'] == 1
        assert any(call['task_id'] == '2222' for call in backend.update_calls)
        assert all(call['task_id'] != '1111' for call in backend.update_calls)
