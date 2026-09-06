"""Tests for the pure finding→task escalation helpers (task 4821 / 4764 arm 3).

Layout precedent: ``tests/test_predicate_contradiction.py`` — the module under
test is a PURE builder (no filesystem or network I/O), so every test here runs
on plain dicts with no harness, no journal, no event buffer and no tmp_path.
"""

from __future__ import annotations

import pytest

from fused_memory.reconciliation.finding_task_escalation import (
    FINDING_TASK_ESCALATION_CATEGORY,
    build_finding_task_escalation_kwargs,
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


def _finding_with_task() -> dict:
    """A realistic actionable Stage-3 finding naming a task id."""
    return {
        'finding_id': 'f-001',
        'severity': 'serious',
        'category': 'memory_contradiction',
        'description': 'Operator ruled Option A on esc-4458-87; commit did the opposite',
        'suggested_action': 'Re-read the operator ruling before closing',
        'actionable': True,
        'task_id': '4458',
        'cited_tasks': [_citation('dark_factory', '4458', 'Reify 4458')],
    }


class TestBuildFindingTaskEscalationKwargs:
    """The pure payload builder."""

    def _build(self, finding=None, **over):
        kwargs = {
            'task_id': '4458',
            'project_id': 'dark_factory',
            'run_id': 'abcdef0123456789',
            'persistence': 4,
        }
        kwargs.update(over)
        return build_finding_task_escalation_kwargs(finding or _finding_with_task(), **kwargs)

    def test_carries_the_real_task_id_never_a_synthetic_recon_id(self):
        """The whole point of the arm: the REAL id reaches the orchestrator queue.

        `_escalate` files to the recon queue under a synthetic `recon-<run8>`
        task id, which is why a finding naming a task currently dead-ends.
        """
        payload = self._build()
        assert payload['task_id'] == '4458'
        assert not payload['task_id'].startswith('recon-')

    def test_fixed_routing_fields(self):
        payload = self._build()
        assert payload['agent_role'] == 'reconciliation-harness'
        assert payload['severity'] == 'info'
        assert payload['level'] == 1
        assert payload['category'] == FINDING_TASK_ESCALATION_CATEGORY
        assert FINDING_TASK_ESCALATION_CATEGORY == 'recon_task_finding'

    def test_summary_is_one_line_naming_the_task_and_finding_category(self):
        payload = self._build()
        summary = payload['summary']
        assert '\n' not in summary, f'summary must be one line, got: {summary!r}'
        assert '4458' in summary
        assert 'memory_contradiction' in summary

    def test_detail_is_json_carrying_the_full_finding_provenance(self):
        import json

        finding = _finding_with_task()
        payload = self._build(finding)
        detail = json.loads(payload['detail'])
        assert detail['finding_id'] == 'f-001'
        assert detail['category'] == 'memory_contradiction'
        assert detail['severity'] == 'serious'
        assert detail['description'] == finding['description']
        assert detail['suggested_action'] == finding['suggested_action']
        assert detail['run_id'] == 'abcdef0123456789'
        assert detail['project_id'] == 'dark_factory'
        assert detail['persistence'] == 4
        assert detail['cited_tasks'] == finding['cited_tasks']

    def test_detail_survives_non_json_serialisable_finding_values(self):
        import json
        from datetime import UTC, datetime

        finding = _finding_with_task()
        finding['description'] = datetime.now(UTC)  # type: ignore[assignment]
        payload = self._build(finding)
        json.loads(payload['detail'])  # must not raise

    def test_no_key_outside_the_escalation_dataclass_can_be_introduced(self):
        """CRITICAL GUARD (constraint (d) of task 4821).

        `Escalation.to_dict` is a bare `asdict` and `from_dict` filters to
        `__dataclass_fields__`, while `queue.resolve()` rewrites the file from
        `esc.to_json()`. Any non-dataclass key on disk is therefore DESTROYED on
        the first resolve -- the exact bug `_POLICY_ONLY_KEYS` /
        `BacklogPolicy._restore_policy_keys` exists to work around. Carrying all
        provenance in `detail` (a real field) means we never need that
        workaround, and this test is what keeps it that way.
        """
        from escalation.models import Escalation  # type: ignore[import-untyped]

        payload = self._build()
        allowed = set(Escalation.__dataclass_fields__) - {'id'}
        assert set(payload) <= allowed, (
            f'payload introduces non-Escalation keys: {sorted(set(payload) - allowed)}'
        )

    def test_id_is_not_supplied_only_the_filer_can_mint_it(self):
        # `id` must come from `queue.make_id(...)`, a durable per-key counter --
        # a pure function has no queue and must not guess a sequence number.
        assert 'id' not in self._build()

    def test_dedupe_fingerprint_is_not_set(self):
        # Nothing on the orchestrator queue folds on a fingerprint for this
        # category; cross-cycle dedupe is `has_open_l1`'s job. Setting one
        # risks unintended folding if a future submit_or_dedupe config ever
        # names the category.
        payload = self._build()
        assert payload.get('dedupe_fingerprint') is None

    def test_suggested_action_is_left_empty(self):
        # The correct disposition is exactly what the ladder exists to decide;
        # the finding's own suggested_action is carried in `detail`.
        assert self._build().get('suggested_action', '') == ''

    def test_builder_does_not_mutate_the_finding(self):
        finding = _finding_with_task()
        before = dict(finding)
        self._build(finding)
        assert finding == before

    def test_missing_finding_fields_degrade_rather_than_raise(self):
        import json

        payload = build_finding_task_escalation_kwargs(
            {}, task_id='4458', project_id='dark_factory', run_id='r', persistence=4,
        )
        assert payload['task_id'] == '4458'
        json.loads(payload['detail'])
