"""Default and green-tier reload disposition of ``verify_allow_no_tests_ran``, the task 7041 opt-in."""

from __future__ import annotations

import pytest

from orchestrator.config import RELOADABLE_FIELDS, OrchestratorConfig, diff_config


@pytest.mark.usefixtures('code_default_config')
class TestVerifyAllowNoTestsRanDefault:
    def test_defaults_to_failing_an_empty_test_slot(self):
        assert OrchestratorConfig().verify_allow_no_tests_ran is False


class TestVerifyAllowNoTestsRanReload:
    def test_field_is_reloadable(self):
        assert 'verify_allow_no_tests_ran' in RELOADABLE_FIELDS

    @pytest.mark.usefixtures('code_default_config')
    def test_edit_lands_in_applied_candidates_not_restart_required(self):
        live = OrchestratorConfig(verify_allow_no_tests_ran=False)
        fresh = OrchestratorConfig(verify_allow_no_tests_ran=True)

        diff = diff_config(live, fresh)

        assert 'verify_allow_no_tests_ran' in diff.applied_candidates
        assert 'verify_allow_no_tests_ran' not in diff.restart_required
