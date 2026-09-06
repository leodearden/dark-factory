"""Tests for the pure finding→task escalation helpers (task 4821 / 4764 arm 3).

Layout precedent: ``tests/test_predicate_contradiction.py`` — the module under
test is a PURE builder (no filesystem or network I/O), so every test here runs
on plain dicts with no harness, no journal, no event buffer and no tmp_path.
"""

from __future__ import annotations

import pytest

from fused_memory.reconciliation.finding_task_escalation import (
    resolve_finding_task_target,
)


class TestResolveFindingTaskTargetBareTaskId:
    """The bare ``finding['task_id']`` branch.

    ``flagged_items`` copies ``_Finding.task_id`` verbatim
    (``server/recon_report.py`` finding_dict projection) and Stage-3 findings are
    LLM-authored, so neither the presence nor the TYPE of this field is
    guaranteed.
    """

    def test_string_task_id_resolves(self):
        finding = {'task_id': '4458', 'category': 'memory_contradiction'}
        assert resolve_finding_task_target(finding, 'dark_factory') == '4458'

    def test_int_task_id_is_coerced_to_str(self):
        # Stage 3 is LLM-authored and flagged_items copies the field verbatim,
        # so an int can reach this function.  The escalation queue's task_id is
        # a str, so coercion must happen here rather than at the filer.
        finding = {'task_id': 4458}
        result = resolve_finding_task_target(finding, 'dark_factory')
        assert result == '4458'
        assert isinstance(result, str)

    def test_surrounding_whitespace_is_stripped(self):
        finding = {'task_id': '  4458 '}
        assert resolve_finding_task_target(finding, 'dark_factory') == '4458'

    @pytest.mark.parametrize('task_id', [None, '', '   '])
    def test_absent_or_blank_task_id_resolves_to_none(self, task_id):
        finding = {'task_id': task_id}
        assert resolve_finding_task_target(finding, 'dark_factory') is None

    def test_missing_task_id_key_resolves_to_none(self):
        assert resolve_finding_task_target({}, 'dark_factory') is None
