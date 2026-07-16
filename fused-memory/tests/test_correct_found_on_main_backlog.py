"""Tests for correct_found_on_main_backlog.py.

Loaded via importlib so the script (not on PYTHONPATH) can be tested without
sys.path pollution — mirrors the pattern in
test_audit_found_on_main_provenance.py / test_audit_duplicate_tasks.py.
"""
from __future__ import annotations

import importlib.util
import types
from pathlib import Path

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
