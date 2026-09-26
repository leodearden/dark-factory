"""Green-tier registration + defaults for
``merge_disjoint_skip_requires_verified_drift`` — the soundness gate on the
merge queue's disjoint-delta fast path (the 2026-09-22 whole-tree-drift
incident, where footprint-disjointness let the queue advance, and report
"merged to main successfully" for, a tree no verification had ever seen
green).

Fixtures are kept MODULE-LOCAL (not conftest.py) for the reason stated in
test_config_verify_admission_reload.py: a conftest.py edit trips verify.py's
has_conftest and forces the merge-time verify to run the full owning-package
suite instead of a scoped subset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from orchestrator.config import (
    RELOADABLE_FIELDS,
    OrchestratorConfig,
    apply_reload,
    diff_config,
)

if TYPE_CHECKING:
    from orchestrator.merge_types import MergeRequest


class TestDisjointSkipGateDefault:
    def test_defaults_to_requiring_verified_drift(self, monkeypatch, tmp_path):
        """Fail-safe by default: an extra verify costs minutes, a laundered
        red main cost nine hours."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        assert OrchestratorConfig().merge_disjoint_skip_requires_verified_drift is True


class TestDisjointSkipGateReloadDisposition:
    """GREEN tier, unlike its restart-only ``merge_verify_breadth`` neighbour:
    the knob only ever decides whether ONE more verify runs before an advance,
    never an in-flight merge's breadth — and a safety kill switch you can only
    pull by restarting the fleet is not a kill switch."""

    def test_field_is_reloadable(self):
        assert 'merge_disjoint_skip_requires_verified_drift' in RELOADABLE_FIELDS

    def test_breadth_neighbour_stays_restart_only(self):
        """Guard against a copy-paste that green-tiers the breadth knob too."""
        assert 'merge_verify_breadth' not in RELOADABLE_FIELDS

    def test_edit_lands_in_applied_candidates_not_restart_required(
        self, monkeypatch, tmp_path,
    ):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        live = OrchestratorConfig(merge_disjoint_skip_requires_verified_drift=True)
        fresh = OrchestratorConfig(merge_disjoint_skip_requires_verified_drift=False)
        diff = diff_config(live, fresh)
        assert 'merge_disjoint_skip_requires_verified_drift' in diff.applied_candidates
        assert 'merge_disjoint_skip_requires_verified_drift' not in diff.restart_required

    def test_apply_reload_applies_in_place(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        live = OrchestratorConfig(merge_disjoint_skip_requires_verified_drift=True)
        fresh = OrchestratorConfig(merge_disjoint_skip_requires_verified_drift=False)
        report = apply_reload(live, fresh)
        assert report['reloaded'] is True
        assert report['applied']['merge_disjoint_skip_requires_verified_drift'] == {
            'old': True, 'new': False,
        }
        assert live.merge_disjoint_skip_requires_verified_drift is False


class TestDisjointSkipBlockers:
    """Unit coverage for the predicate the gate consults."""

    def _req(self, config) -> MergeRequest:
        """A duck-typed stand-in: the predicate reads only task_id/config."""
        from types import SimpleNamespace
        return cast('MergeRequest', SimpleNamespace(task_id='t', config=config))

    def test_verified_drift_and_scoped_breadth_has_no_blockers(
        self, monkeypatch, tmp_path,
    ):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        from orchestrator.merge_gates import (
            _disjoint_skip_blockers,
            note_queue_verified_main_tip,
        )
        sha = 'f' * 40
        note_queue_verified_main_tip(sha)
        cfg = OrchestratorConfig(merge_verify_breadth='scoped')
        assert _disjoint_skip_blockers(self._req(cfg), rebased_onto=sha) == []

    def test_unverified_drift_blocks(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        from orchestrator.merge_gates import _disjoint_skip_blockers
        cfg = OrchestratorConfig(merge_verify_breadth='scoped')
        blockers = _disjoint_skip_blockers(self._req(cfg), rebased_onto='0' * 40)
        assert len(blockers) == 1 and 'unverified drift' in blockers[0]

    def test_kill_switch_restores_legacy_trust(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        from orchestrator.merge_gates import _disjoint_skip_blockers
        cfg = OrchestratorConfig(
            merge_verify_breadth='scoped',
            merge_disjoint_skip_requires_verified_drift=False,
        )
        assert _disjoint_skip_blockers(self._req(cfg), rebased_onto='0' * 40) == []

    def test_whole_tree_breadth_blocks_even_with_kill_switch_off(
        self, monkeypatch, tmp_path,
    ):
        """A project that declares a whole-tree merge gate has declared the
        skip unsound outright, not merely expensive — so the P2 kill switch
        does not reach this arm."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        from orchestrator.merge_gates import (
            _disjoint_skip_blockers,
            note_queue_verified_main_tip,
        )
        sha = 'e' * 40
        note_queue_verified_main_tip(sha)
        cfg = OrchestratorConfig(
            merge_verify_breadth='full',
            merge_disjoint_skip_requires_verified_drift=False,
        )
        blockers = _disjoint_skip_blockers(self._req(cfg), rebased_onto=sha)
        assert len(blockers) == 1 and 'whole-tree' in blockers[0]

    def test_missing_config_fails_safe(self):
        """Unknown premises answer 'distrust', matching the overlap probe's
        own fail-CLOSED policy."""
        from types import SimpleNamespace

        from orchestrator.merge_gates import _disjoint_skip_blockers
        req = cast('MergeRequest', SimpleNamespace(task_id='t'))
        assert _disjoint_skip_blockers(req, rebased_onto='0' * 40) != []
