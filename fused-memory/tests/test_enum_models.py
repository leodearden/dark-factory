"""Tests for StrEnum model additions and frozenset immutability."""

import pytest
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Step 1 & 2: frozenset immutability for GRAPHITI_PRIMARY / MEM0_PRIMARY
# ---------------------------------------------------------------------------


class TestFrozensetConstants:
    """GRAPHITI_PRIMARY and MEM0_PRIMARY must be immutable frozensets."""

    def test_graphiti_primary_is_frozenset(self):
        from fused_memory.models.enums import GRAPHITI_PRIMARY

        assert isinstance(GRAPHITI_PRIMARY, frozenset)

    def test_mem0_primary_is_frozenset(self):
        from fused_memory.models.enums import MEM0_PRIMARY

        assert isinstance(MEM0_PRIMARY, frozenset)

    def test_graphiti_primary_rejects_add(self):
        from fused_memory.models.enums import GRAPHITI_PRIMARY

        with pytest.raises(AttributeError):
            GRAPHITI_PRIMARY.add(None)  # type: ignore[attr-defined]

    def test_mem0_primary_rejects_add(self):
        from fused_memory.models.enums import MEM0_PRIMARY

        with pytest.raises(AttributeError):
            MEM0_PRIMARY.add(None)  # type: ignore[attr-defined]

    def test_graphiti_primary_rejects_discard(self):
        from fused_memory.models.enums import GRAPHITI_PRIMARY

        with pytest.raises(AttributeError):
            GRAPHITI_PRIMARY.discard(None)  # type: ignore[attr-defined]

    def test_mem0_primary_rejects_discard(self):
        from fused_memory.models.enums import MEM0_PRIMARY

        with pytest.raises(AttributeError):
            MEM0_PRIMARY.discard(None)  # type: ignore[attr-defined]

    def test_graphiti_primary_membership(self):
        """Membership checks still work after frozenset conversion."""
        from fused_memory.models.enums import GRAPHITI_PRIMARY, MemoryCategory

        assert MemoryCategory.entities_and_relations in GRAPHITI_PRIMARY
        assert MemoryCategory.temporal_facts in GRAPHITI_PRIMARY
        assert MemoryCategory.decisions_and_rationale in GRAPHITI_PRIMARY
        assert MemoryCategory.preferences_and_norms not in GRAPHITI_PRIMARY

    def test_mem0_primary_membership(self):
        """Membership checks still work after frozenset conversion."""
        from fused_memory.models.enums import MEM0_PRIMARY, MemoryCategory

        assert MemoryCategory.preferences_and_norms in MEM0_PRIMARY
        assert MemoryCategory.procedural_knowledge in MEM0_PRIMARY
        assert MemoryCategory.observations_and_summaries in MEM0_PRIMARY
        assert MemoryCategory.entities_and_relations not in MEM0_PRIMARY


# ---------------------------------------------------------------------------
# Step 3: EpisodeStatus StrEnum
# ---------------------------------------------------------------------------


class TestEpisodeStatus:
    """EpisodeStatus StrEnum construction, equality, invalid values."""

    def test_construction_from_queued(self):
        from fused_memory.models.memory import EpisodeStatus

        s = EpisodeStatus('queued')
        assert s == EpisodeStatus.queued

    def test_construction_from_processed(self):
        from fused_memory.models.memory import EpisodeStatus

        s = EpisodeStatus('processed')
        assert s == EpisodeStatus.processed

    def test_construction_from_error(self):
        from fused_memory.models.memory import EpisodeStatus

        s = EpisodeStatus('error')
        assert s == EpisodeStatus.error

    def test_string_equality(self):
        from fused_memory.models.memory import EpisodeStatus

        assert EpisodeStatus.queued == 'queued'
        assert EpisodeStatus.processed == 'processed'
        assert EpisodeStatus.error == 'error'

    def test_is_str_subclass(self):
        from fused_memory.models.memory import EpisodeStatus

        assert isinstance(EpisodeStatus.queued, str)

    def test_invalid_value_raises(self):
        from fused_memory.models.memory import EpisodeStatus

        with pytest.raises(ValueError):
            EpisodeStatus('unknown')

    def test_all_values(self):
        from fused_memory.models.memory import EpisodeStatus

        values = {e.value for e in EpisodeStatus}
        assert values == {'queued', 'processed', 'error'}


# ---------------------------------------------------------------------------
# Step 4: AddEpisodeResponse pydantic coercion
# ---------------------------------------------------------------------------


class TestAddEpisodeResponseCoercion:
    """AddEpisodeResponse coerces bare strings to EpisodeStatus."""

    def test_bare_string_queued_coerces(self):
        from fused_memory.models.memory import AddEpisodeResponse, EpisodeStatus

        resp = AddEpisodeResponse(status='queued')  # type: ignore[arg-type]
        assert resp.status == EpisodeStatus.queued
        assert isinstance(resp.status, EpisodeStatus)

    def test_bare_string_processed_coerces(self):
        from fused_memory.models.memory import AddEpisodeResponse, EpisodeStatus

        resp = AddEpisodeResponse(status='processed')  # type: ignore[arg-type]
        assert resp.status == EpisodeStatus.processed

    def test_bare_string_error_coerces(self):
        from fused_memory.models.memory import AddEpisodeResponse, EpisodeStatus

        resp = AddEpisodeResponse(status='error')  # type: ignore[arg-type]
        assert resp.status == EpisodeStatus.error

    def test_enum_value_passes_through(self):
        from fused_memory.models.memory import AddEpisodeResponse, EpisodeStatus

        resp = AddEpisodeResponse(status=EpisodeStatus.queued)
        assert resp.status == EpisodeStatus.queued

    def test_invalid_status_raises_validation_error(self):
        from fused_memory.models.memory import AddEpisodeResponse

        with pytest.raises(ValidationError):
            AddEpisodeResponse(status='unknown_status')  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Step 6: Reconciliation StrEnums
# ---------------------------------------------------------------------------


class TestRunType:
    def test_values(self):
        from fused_memory.models.reconciliation import RunType

        assert set(RunType) == {RunType.full, RunType.targeted, RunType.remediation}

    def test_string_equality(self):
        from fused_memory.models.reconciliation import RunType

        assert RunType.full == 'full'
        assert RunType.targeted == 'targeted'
        assert RunType.remediation == 'remediation'

    def test_construction_from_string(self):
        from fused_memory.models.reconciliation import RunType

        assert RunType('full') == RunType.full
        assert RunType('targeted') == RunType.targeted
        assert RunType('remediation') == RunType.remediation

    def test_invalid_raises(self):
        from fused_memory.models.reconciliation import RunType

        with pytest.raises(ValueError):
            RunType('batch')


class TestRunStatus:
    def test_values(self):
        from fused_memory.models.reconciliation import RunStatus

        expected = {'running', 'completed', 'failed', 'rolled_back', 'circuit_breaker'}
        assert {e.value for e in RunStatus} == expected

    def test_string_equality(self):
        from fused_memory.models.reconciliation import RunStatus

        assert RunStatus.running == 'running'
        assert RunStatus.completed == 'completed'
        assert RunStatus.failed == 'failed'
        assert RunStatus.rolled_back == 'rolled_back'
        assert RunStatus.circuit_breaker == 'circuit_breaker'

    def test_invalid_raises(self):
        from fused_memory.models.reconciliation import RunStatus

        with pytest.raises(ValueError):
            RunStatus('unknown')


class TestVerdictSeverity:
    def test_values(self):
        from fused_memory.models.reconciliation import VerdictSeverity

        assert {e.value for e in VerdictSeverity} == {'ok', 'minor', 'moderate', 'serious'}

    def test_string_equality(self):
        from fused_memory.models.reconciliation import VerdictSeverity

        assert VerdictSeverity.ok == 'ok'
        assert VerdictSeverity.minor == 'minor'
        assert VerdictSeverity.moderate == 'moderate'
        assert VerdictSeverity.serious == 'serious'

    def test_invalid_raises(self):
        from fused_memory.models.reconciliation import VerdictSeverity

        with pytest.raises(ValueError):
            VerdictSeverity('critical')


class TestVerdictAction:
    def test_values(self):
        from fused_memory.models.reconciliation import VerdictAction

        assert {e.value for e in VerdictAction} == {'none', 'auto_fix', 'rollback', 'halt'}

    def test_string_equality(self):
        from fused_memory.models.reconciliation import VerdictAction

        assert VerdictAction.none == 'none'
        assert VerdictAction.auto_fix == 'auto_fix'
        assert VerdictAction.rollback == 'rollback'
        assert VerdictAction.halt == 'halt'

    def test_invalid_raises(self):
        from fused_memory.models.reconciliation import VerdictAction

        with pytest.raises(ValueError):
            VerdictAction('skip')


class TestVerificationVerdict:
    def test_values(self):
        from fused_memory.models.reconciliation import VerificationVerdict

        assert {e.value for e in VerificationVerdict} == {
            'confirmed',
            'contradicted',
            'inconclusive',
        }

    def test_string_equality(self):
        from fused_memory.models.reconciliation import VerificationVerdict

        assert VerificationVerdict.confirmed == 'confirmed'
        assert VerificationVerdict.contradicted == 'contradicted'
        assert VerificationVerdict.inconclusive == 'inconclusive'

    def test_invalid_raises(self):
        from fused_memory.models.reconciliation import VerificationVerdict

        with pytest.raises(ValueError):
            VerificationVerdict('maybe')


# ---------------------------------------------------------------------------
# Step 7: Reconciliation model pydantic coercion
# ---------------------------------------------------------------------------


class TestReconciliationRunCoercion:
    """ReconciliationRun coerces bare strings to RunType/RunStatus."""

    def _make_run(self, **kwargs):
        from datetime import datetime

        from fused_memory.models.reconciliation import ReconciliationRun

        defaults = {
            'id': 'run-1',
            'project_id': 'dark_factory',
            'run_type': 'full',
            'trigger_reason': 'test',
            'started_at': datetime(2026, 1, 1),
        }
        defaults.update(kwargs)
        return ReconciliationRun(**defaults)

    def test_run_type_coerces_from_string(self):
        from fused_memory.models.reconciliation import RunType

        run = self._make_run(run_type='full')
        assert run.run_type == RunType.full
        assert isinstance(run.run_type, RunType)

    def test_run_type_targeted_coerces(self):
        from fused_memory.models.reconciliation import RunType

        run = self._make_run(run_type='targeted')
        assert run.run_type == RunType.targeted

    def test_status_default_is_running(self):
        from fused_memory.models.reconciliation import RunStatus

        run = self._make_run()
        assert run.status == RunStatus.running

    def test_status_coerces_from_string(self):
        from fused_memory.models.reconciliation import RunStatus

        run = self._make_run(status='completed')
        assert run.status == RunStatus.completed
        assert isinstance(run.status, RunStatus)

    def test_run_type_remediation_coerces(self):
        from fused_memory.models.reconciliation import RunType

        run = self._make_run(run_type='remediation')
        assert run.run_type == RunType.remediation

    def test_invalid_run_type_raises(self):
        with pytest.raises(ValidationError):
            self._make_run(run_type='batch')

    def test_invalid_status_raises(self):
        with pytest.raises(ValidationError):
            self._make_run(status='unknown')


class TestJudgeVerdictCoercion:
    """JudgeVerdict coerces bare strings to VerdictSeverity/VerdictAction."""

    def _make_verdict(self, **kwargs):
        from datetime import datetime

        from fused_memory.models.reconciliation import JudgeVerdict

        defaults = {
            'run_id': 'run-1',
            'reviewed_at': datetime(2026, 1, 1),
            'severity': 'ok',
        }
        defaults.update(kwargs)
        return JudgeVerdict(**defaults)

    def test_severity_coerces_from_string(self):
        from fused_memory.models.reconciliation import VerdictSeverity

        v = self._make_verdict(severity='minor')
        assert v.severity == VerdictSeverity.minor
        assert isinstance(v.severity, VerdictSeverity)

    def test_action_taken_default_is_none(self):
        from fused_memory.models.reconciliation import VerdictAction

        v = self._make_verdict()
        assert v.action_taken == VerdictAction.none

    def test_action_taken_coerces_from_string(self):
        from fused_memory.models.reconciliation import VerdictAction

        v = self._make_verdict(action_taken='rollback')
        assert v.action_taken == VerdictAction.rollback
        assert isinstance(v.action_taken, VerdictAction)

    def test_invalid_severity_raises(self):
        with pytest.raises(ValidationError):
            self._make_verdict(severity='catastrophic')

    def test_invalid_action_taken_raises(self):
        with pytest.raises(ValidationError):
            self._make_verdict(action_taken='skip')


class TestVerificationResultCoercion:
    """VerificationResult coerces bare strings to VerificationVerdict."""

    def _make_result(self, **kwargs):
        from fused_memory.models.reconciliation import VerificationResult

        defaults = {'verdict': 'confirmed', 'confidence': 0.9}
        defaults.update(kwargs)
        return VerificationResult(**defaults)

    def test_verdict_coerces_from_string(self):
        from fused_memory.models.reconciliation import VerificationVerdict

        r = self._make_result(verdict='confirmed')
        assert r.verdict == VerificationVerdict.confirmed
        assert isinstance(r.verdict, VerificationVerdict)

    def test_verdict_contradicted_coerces(self):
        from fused_memory.models.reconciliation import VerificationVerdict

        r = self._make_result(verdict='contradicted')
        assert r.verdict == VerificationVerdict.contradicted

    def test_verdict_inconclusive_coerces(self):
        from fused_memory.models.reconciliation import VerificationVerdict

        r = self._make_result(verdict='inconclusive')
        assert r.verdict == VerificationVerdict.inconclusive

    def test_invalid_verdict_raises(self):
        with pytest.raises(ValidationError):
            self._make_result(verdict='maybe')


# ---------------------------------------------------------------------------
# Step 1: Watermark.project_id field_validator (tests written before impl)
# ---------------------------------------------------------------------------


class TestWatermarkProjectIdValidator:
    """Watermark.project_id is stripped and whitespace-only becomes '' sentinel."""

    def _make_watermark(self, project_id: str):
        from fused_memory.models.reconciliation import Watermark

        return Watermark(project_id=project_id)

    def test_valid_project_id_passes_through_unchanged(self):
        """A normal project_id with no surrounding whitespace is left as-is."""
        wm = self._make_watermark('dark_factory')
        assert wm.project_id == 'dark_factory'

    def test_whitespace_only_normalizes_to_empty_string(self):
        """Whitespace-only project_id is the bootstrap sentinel — normalizes to ''."""
        wm = self._make_watermark('   ')
        assert wm.project_id == ''

    def test_leading_trailing_whitespace_is_stripped(self):
        """Leading/trailing whitespace is stripped from project_id."""
        wm = self._make_watermark('  dark_factory  ')
        assert wm.project_id == 'dark_factory'

    def test_empty_string_passes_through_as_valid_sentinel(self):
        """Empty string is the valid 'no project' bootstrap sentinel."""
        wm = self._make_watermark('')
        assert wm.project_id == ''

    def test_tabs_and_newlines_normalize_to_empty(self):
        """Tab/newline-only project_id normalizes to empty string sentinel."""
        wm = self._make_watermark('\t\n')
        assert wm.project_id == ''


# ---------------------------------------------------------------------------
# Step 3: base.py guard contract tests (pass before and after simplification)
# ---------------------------------------------------------------------------


class TestBaseStageGuardContract:
    """BaseStage.run() guard handles the three expected cases correctly.

    These are contract tests — they must pass both before and after step 4
    simplifies the guard logic.  They document the intended behaviour rather
    than the implementation detail.
    """

    @pytest.fixture
    def mock_deps(self):
        from unittest.mock import AsyncMock

        from fused_memory.config.schema import ReconciliationConfig

        config = ReconciliationConfig(enabled=True, explore_codebase_root='/tmp/test')
        return {
            'memory_service': AsyncMock(),
            'taskmaster': AsyncMock(),
            'journal': AsyncMock(),
            'config': config,
        }

    @staticmethod
    def _patch_stage(stage):
        """Patch away the expensive I/O so only the guard logic is exercised."""
        from contextlib import contextmanager
        from unittest.mock import patch

        from fused_memory.reconciliation.cli_stage_runner import StageResult

        async def _fake_assemble(events, watermark, prior_reports):
            return 'payload'

        async def _fake_cli(**kwargs):
            return StageResult(success=True, report={'summary': 'ok'})

        @contextmanager
        def _ctx():
            with (
                patch.object(stage, 'assemble_payload', side_effect=_fake_assemble),
                patch(
                    'fused_memory.reconciliation.stages.base.run_stage_via_cli',
                    side_effect=_fake_cli,
                ),
            ):
                yield

        return _ctx()

    @pytest.mark.asyncio
    async def test_matching_watermark_project_id_proceeds(self, mock_deps):
        """When watermark.project_id matches stage.project_id, run() completes."""
        from fused_memory.models.reconciliation import StageId, Watermark
        from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator

        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        stage.project_id = 'dark_factory'

        with self._patch_stage(stage):
            result = await stage.run(
                events=[],
                watermark=Watermark(project_id='dark_factory'),
                prior_reports=[],
                run_id='guard-test-match',
            )
        assert result is not None
        assert result.stage == StageId.memory_consolidator

    @pytest.mark.asyncio
    async def test_empty_watermark_project_id_proceeds_with_debug_log(self, mock_deps, caplog):
        """When watermark.project_id is '', run() proceeds and emits a DEBUG log."""
        import logging

        from fused_memory.models.reconciliation import StageId, Watermark
        from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator

        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        stage.project_id = 'dark_factory'

        with (
            self._patch_stage(stage),
            caplog.at_level(logging.DEBUG, logger='fused_memory.reconciliation.stages.base'),
        ):
            result = await stage.run(
                events=[],
                watermark=Watermark(project_id=''),
                prior_reports=[],
                run_id='guard-test-empty',
            )
        assert result is not None
        # Guard must emit a debug message mentioning the skip/mismatch check
        assert any(
            'no project_id' in msg.lower() or 'skipping' in msg.lower()
            for msg in caplog.messages
        )

    @pytest.mark.asyncio
    async def test_mismatching_watermark_project_id_raises_value_error(self, mock_deps):
        """When watermark.project_id is set but differs from stage.project_id, ValueError is raised."""
        from fused_memory.models.reconciliation import StageId, Watermark
        from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator

        stage = MemoryConsolidator(StageId.memory_consolidator, **mock_deps)
        stage.project_id = 'project_a'

        with self._patch_stage(stage), pytest.raises(ValueError) as exc_info:
            await stage.run(
                events=[],
                watermark=Watermark(project_id='project_b'),
                prior_reports=[],
                run_id='guard-test-mismatch',
            )
        error_msg = str(exc_info.value)
        assert 'project_a' in error_msg
        assert 'project_b' in error_msg
