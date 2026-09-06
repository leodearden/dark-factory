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


def _citation(project_id: str, task_id: str, title: str = 'x') -> dict:
    """A `cited_tasks` entry in the shape `cite_task` writes it.

    ``{project_id, task_id, title}`` — see the citation construction in
    ``fused_memory/server/recon_report.py``.  Unlike the bare ``task_id`` field,
    a citation IS project-qualified, which is the whole basis of the
    same-project guard below.
    """
    return {'project_id': project_id, 'task_id': task_id, 'title': title}


class TestResolveFindingTaskTargetCitedTasksFallback:
    """The `cited_tasks` fallback, and the same-project guard on it."""

    def test_same_project_citation_resolves(self):
        finding = {
            'task_id': None,
            'cited_tasks': [_citation('dark_factory', '4458')],
        }
        assert resolve_finding_task_target(finding, 'dark_factory') == '4458'

    def test_foreign_project_citation_is_never_routed(self):
        # The SAME finding, resolved for a different project, must yield None.
        # Routing a foreign project's task id into this project's queue would
        # file a well-formed-looking escalation onto an unrelated task — the
        # false attribution the ladder is least able to detect.
        finding = {
            'task_id': None,
            'cited_tasks': [_citation('dark_factory', '4458')],
        }
        assert resolve_finding_task_target(finding, 'reify') is None

    def test_first_same_project_citation_wins_over_leading_foreign_one(self):
        finding = {
            'cited_tasks': [
                _citation('reify', '11'),
                _citation('dark_factory', '4458'),
                _citation('dark_factory', '9999'),
            ],
        }
        assert resolve_finding_task_target(finding, 'dark_factory') == '4458'

    def test_citation_task_id_is_coerced_and_stripped(self):
        finding = {'cited_tasks': [_citation('dark_factory', '  4458 ')]}
        assert resolve_finding_task_target(finding, 'dark_factory') == '4458'
        finding_int = {'cited_tasks': [{'project_id': 'dark_factory', 'task_id': 4458}]}
        assert resolve_finding_task_target(finding_int, 'dark_factory') == '4458'

    @pytest.mark.parametrize(
        'entry',
        [
            'not-a-mapping',
            42,
            None,
            {'project_id': 'dark_factory'},                       # no task_id
            {'task_id': '4458'},                                  # no project_id
            {'project_id': 'dark_factory', 'task_id': ''},        # empty task_id
            {'project_id': 'dark_factory', 'task_id': '   '},     # blank task_id
            {'project_id': None, 'task_id': '4458'},              # null project_id
            {'project_id': '', 'task_id': '4458'},                # empty project_id
        ],
    )
    def test_malformed_citation_entries_are_skipped_not_raised(self, entry):
        # Stage 3 is LLM-authored: a malformed citation must degrade to
        # "no target" rather than abort the remediation pass.
        assert resolve_finding_task_target({'cited_tasks': [entry]}, 'dark_factory') is None

    def test_malformed_entry_does_not_hide_a_later_valid_one(self):
        finding = {'cited_tasks': ['junk', _citation('dark_factory', '4458')]}
        assert resolve_finding_task_target(finding, 'dark_factory') == '4458'

    @pytest.mark.parametrize('cited', [None, [], 'not-a-list'])
    def test_absent_or_unusable_cited_tasks_resolves_to_none(self, cited):
        assert resolve_finding_task_target({'cited_tasks': cited}, 'dark_factory') is None


class TestResolveFindingTaskTargetPrecedence:
    """A bare ``task_id`` outranks ``cited_tasks``, and names NO project."""

    def test_bare_task_id_wins_over_conflicting_same_project_citation(self):
        finding = {
            'task_id': '4458',
            'cited_tasks': [_citation('dark_factory', '9999')],
        }
        assert resolve_finding_task_target(finding, 'dark_factory') == '4458'

    @pytest.mark.parametrize('project_id', ['dark_factory', 'reify', 'anything-at-all'])
    def test_bare_task_id_is_project_agnostic(self, project_id):
        """The projectless hazard, pinned behaviourally.

        Per the task-4185 operator ruling recorded in
        ``fused_memory/server/recon_report.py``, the in-run signature index is
        deliberately keyed on a PROJECTLESS signature: an
        ``add_finding(task_id='42', ...)`` call carries no project whatsoever.
        The bare field therefore CANNOT be project-matched — it is interpreted
        as belonging to the project being reconciled, whichever that is.  A
        citation, being project-qualified, is matched; this asymmetry is
        deliberate and is what the two classes above pin from either side.
        """
        assert resolve_finding_task_target({'task_id': '42'}, project_id) == '42'

    def test_blank_bare_task_id_falls_through_to_citation(self):
        finding = {
            'task_id': '   ',
            'cited_tasks': [_citation('dark_factory', '4458')],
        }
        assert resolve_finding_task_target(finding, 'dark_factory') == '4458'
