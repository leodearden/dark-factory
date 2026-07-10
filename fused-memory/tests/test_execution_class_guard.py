"""Unit tests for fused_memory.middleware.execution_class_guard.

Step 1 (RED → step-2 GREEN): invariant matrix for execution_class_error.
Step 3 (RED → step-4 GREEN): inject_execution_class normalization.
"""

from __future__ import annotations

import json

import pytest

from fused_memory.middleware.execution_class_guard import execution_class_error
from fused_memory.reconciliation.recon_self_model import EXECUTION_CLASSES

# ---------------------------------------------------------------------------
# Step-1: invariant matrix tests
# ---------------------------------------------------------------------------


class TestExecutionClassErrorInvariantMatrix:
    """recon-stage-only enforcement of metadata.execution_class."""

    def test_recon_stage_missing_execution_class_rejects(self):
        """recon-stage agent_id + metadata missing execution_class → reject,
        naming execution_class, with a hint key."""
        result = execution_class_error({}, 'recon-stage-task_knowledge_sync', '/proj')
        assert result is not None
        assert result.get('error_type') == 'ValidationError'
        assert 'hint' in result
        assert 'execution_class' in result['error']

    def test_recon_stage_unknown_execution_class_rejects(self):
        """recon-stage agent_id + metadata.execution_class='bogus' (unknown) → reject."""
        result = execution_class_error(
            {'execution_class': 'bogus'},
            'recon-stage-task_knowledge_sync',
            '/proj',
        )
        assert result is not None
        assert result.get('error_type') == 'ValidationError'
        assert 'execution_class' in result['error']

    @pytest.mark.parametrize('valid_class', EXECUTION_CLASSES)
    def test_recon_stage_valid_execution_class_accepts(self, valid_class):
        """recon-stage agent_id + each valid class (code_tdd/operational/decision) → accept (None)."""
        result = execution_class_error(
            {'execution_class': valid_class},
            'recon-stage-task_knowledge_sync',
            '/proj',
        )
        assert result is None

    def test_recon_stage_metadata_json_string_with_valid_class_accepts(self):
        """metadata as a JSON string carrying a valid class → accept."""
        meta_str = json.dumps({'execution_class': 'operational'})
        result = execution_class_error(meta_str, 'recon-stage-task_knowledge_sync', '/proj')
        assert result is None

    def test_recon_stage_metadata_json_string_without_class_rejects(self):
        """metadata as a JSON string carrying no class → reject."""
        meta_str = json.dumps({'foo': 'bar'})
        result = execution_class_error(meta_str, 'recon-stage-task_knowledge_sync', '/proj')
        assert result is not None
        assert result.get('error_type') == 'ValidationError'

    def test_non_recon_agent_no_execution_class_accepts(self):
        """NON-recon agent_id with NO execution_class → accept (no enforcement)."""
        result = execution_class_error({}, 'claude-interactive', '/proj')
        assert result is None

    def test_agent_id_none_no_execution_class_accepts(self):
        """agent_id=None + no execution_class → accept (no enforcement)."""
        result = execution_class_error({}, None, '/proj')
        assert result is None
