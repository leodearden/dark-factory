"""Tests for escalation/authority.py — the identity-derived escalation authority.

PRD ``plans/task-status-authority-prd.md`` contract C8 / decision D7
(findings 10.1/10.2). ``ROLE_LEVEL_ALLOWLIST`` and ``PROMOTE_ALLOWED`` are the
data tables consulted by ``escalation.server.resolve_issue`` /
``promote_to_l2`` to derive level authority from the caller's
X-Escalation-Identity, replacing the general-case default-open hole left by
2041's header-opt-in capability guard.

These tests pin the tables' contents (not yet the server wiring, which is
step-4/step-6 of this task's plan) AND pin a cross-layer lockstep: the
canonical watcher identity string is duplicated (not imported) in
authority.py to preserve the escalation -> orchestrator layer direction (see
authority.py module docstring), so this test imports the REAL
``orchestrator.harness._WATCHER_ESCALATION_HEADERS`` constant and asserts it
stays in sync with the duplicated string.
"""

from __future__ import annotations

from escalation.authority import (
    L2_AUTO_CLOSE_ACTION,
    L2_AUTO_CLOSE_ALLOWLIST,
    L2_AUTO_CLOSE_DENY_CATEGORIES,
    L2_AUTO_CLOSE_DENY_ROLES,
    PROMOTE_ALLOWED,
    ROLE_LEVEL_ALLOWLIST,
    l2_auto_close_class,
)


class TestRoleLevelAllowlistShape:
    """ROLE_LEVEL_ALLOWLIST maps the deployed auto-watcher identity to {0,1}."""

    def test_watcher_identity_maps_to_levels_0_and_1(self) -> None:
        ceiling = ROLE_LEVEL_ALLOWLIST.get('orchestrator-escalation-watcher-auto')
        assert ceiling == frozenset({0, 1}), (
            f'Expected frozenset({{0, 1}}), got: {ceiling!r}'
        )

    def test_ceiling_values_are_frozensets(self) -> None:
        assert len(ROLE_LEVEL_ALLOWLIST) > 0, 'expected at least one mapped identity'
        for identity, ceiling in ROLE_LEVEL_ALLOWLIST.items():
            assert isinstance(ceiling, frozenset), (
                f'{identity!r} ceiling must be a frozenset, got: {type(ceiling)!r}'
            )


class TestPromoteAllowedShape:
    """PROMOTE_ALLOWED is a frozenset containing the deployed auto-watcher identity."""

    def test_contains_watcher_identity(self) -> None:
        assert 'orchestrator-escalation-watcher-auto' in PROMOTE_ALLOWED


class TestCrossLayerIdentityLockstep:
    """The duplicated identity string stays pinned to the REAL orchestrator constant.

    escalation is the lower fleet-wide package and must not module-level
    import orchestrator (see authority.py module docstring), so the
    canonical watcher identity is duplicated there. This test is the
    cross-layer pin: it imports the real
    ``orchestrator.harness._WATCHER_ESCALATION_HEADERS`` wire constant and
    asserts it is a key of ROLE_LEVEL_ALLOWLIST (mapped to exactly {0, 1} —
    the no-op guarantee: the deployed watcher's ceiling doesn't change) AND
    is a member of PROMOTE_ALLOWED (the deployed watcher may still promote).
    """

    def test_watcher_wire_identity_is_mapped_and_promote_allowed(self) -> None:
        from orchestrator.harness import _WATCHER_ESCALATION_HEADERS

        identity = _WATCHER_ESCALATION_HEADERS['X-Escalation-Identity']

        assert identity in ROLE_LEVEL_ALLOWLIST, (
            f'Real watcher identity {identity!r} must be a key of ROLE_LEVEL_ALLOWLIST'
        )
        assert ROLE_LEVEL_ALLOWLIST[identity] == frozenset({0, 1}), (
            f'Real watcher identity {identity!r} ceiling must be exactly {{0, 1}} '
            f'(no-op guarantee), got: {ROLE_LEVEL_ALLOWLIST[identity]!r}'
        )
        assert identity in PROMOTE_ALLOWED, (
            f'Real watcher identity {identity!r} must remain in PROMOTE_ALLOWED'
        )


class TestL2AutoCloseClass:
    """``l2_auto_close_class`` — the narrow above-ceiling ``close_only`` carve-out.

    Task 2630: identity ``orchestrator-escalation-watcher-auto`` may resolve a
    level-2 record with ``action='close_only'`` ONLY when the record matches
    one of three allowlisted classes AND the resolution text carries that
    class's required structural evidence. Everything else at L2 stays
    forbidden (returns ``None`` — the caller falls back to ``level_forbidden``).

    Uses the literal watcher identity string (mirroring the existing classes
    above, which use the literal string rather than importing the private
    ``_WATCHER_AUTO_IDENTITY``).
    """

    WATCHER = 'orchestrator-escalation-watcher-auto'

    # -- (1) class (a): superseded_main_sweep — BOTH evidence groups required --

    def test_superseded_main_sweep_both_evidence_groups_present(self) -> None:
        resolution = (
            'Superseded by newer main-sweep escalation esc-main-sweep-b2c3d4e5f6a7-1; '
            'main advanced and verifies clean; swept SHA of this record is-ancestor '
            'of the clean tip.'
        )
        result = l2_auto_close_class(
            identity=self.WATCHER, level=2, action='close_only',
            category='infra_issue', agent_role='orchestrator-main-sweep',
            resolution=resolution,
        )
        assert result == 'superseded_main_sweep', f'Expected match, got: {result!r}'

    def test_superseded_main_sweep_missing_sweep_id_no_match(self) -> None:
        """Ancestor/green proof present but no newer-sweep-escalation id -> no match."""
        resolution = 'main advanced and verifies clean; swept SHA is-ancestor of the clean tip.'
        result = l2_auto_close_class(
            identity=self.WATCHER, level=2, action='close_only',
            category='infra_issue', agent_role='orchestrator-main-sweep',
            resolution=resolution,
        )
        assert result is None, f'Expected no match (missing sweep-id evidence), got: {result!r}'

    def test_superseded_main_sweep_missing_ancestor_proof_no_match(self) -> None:
        """Newer-sweep-escalation id present but no ancestor/green proof -> no match."""
        resolution = 'Superseded by newer main-sweep escalation esc-main-sweep-b2c3d4e5f6a7-1.'
        result = l2_auto_close_class(
            identity=self.WATCHER, level=2, action='close_only',
            category='infra_issue', agent_role='orchestrator-main-sweep',
            resolution=resolution,
        )
        assert result is None, (
            f'Expected no match (missing ancestor-proof evidence), got: {result!r}'
        )

    # -- (2) class (b): self_cleared_infra — one live-probe key=value token --

    def test_self_cleared_infra_with_probe_token(self) -> None:
        resolution = 'live probe: curator paused=false — transient infra self-cleared.'
        result = l2_auto_close_class(
            identity=self.WATCHER, level=2, action='close_only',
            category='infra_issue', agent_role='some-agent',
            resolution=resolution,
        )
        assert result == 'self_cleared_infra', f'Expected match, got: {result!r}'

    def test_self_cleared_infra_without_probe_token_no_match(self) -> None:
        resolution = 'infra looks fine now.'
        result = l2_auto_close_class(
            identity=self.WATCHER, level=2, action='close_only',
            category='infra_issue', agent_role='some-agent',
            resolution=resolution,
        )
        assert result is None, f'Expected no match (no probe token), got: {result!r}'

    def test_self_cleared_infra_arbitrary_key_value_no_match(self) -> None:
        """An arbitrary 'word=word' token (not a known liveness-probe key)
        must NOT satisfy the evidence requirement — only a recognized probe
        key (paused/ActiveState/MainPID/ActiveEnterTimestamp) counts."""
        resolution = 'note=ok, config a=b was wrong but infra_issue closed anyway.'
        result = l2_auto_close_class(
            identity=self.WATCHER, level=2, action='close_only',
            category='infra_issue', agent_role='some-agent',
            resolution=resolution,
        )
        assert result is None, (
            f'Expected no match (arbitrary key=value is not a probe token), got: {result!r}'
        )

    def test_self_cleared_infra_with_active_state_and_mainpid_tokens(self) -> None:
        """Other allowlisted probe keys (ActiveState, MainPID) also count,
        including the '>' comparison harness.py itself emits (MainPID>0)."""
        resolution = 'Live unit healthy: MainPID>0, ActiveState=active.'
        result = l2_auto_close_class(
            identity=self.WATCHER, level=2, action='close_only',
            category='infra_issue', agent_role='some-agent',
            resolution=resolution,
        )
        assert result == 'self_cleared_infra', f'Expected match, got: {result!r}'

    # -- (3) class (c): stale_task_scoped — category-agnostic, status citation required --

    def test_stale_task_scoped_status_citation(self) -> None:
        resolution = 'Subject task status=done per get_task; escalation moot.'
        result = l2_auto_close_class(
            identity=self.WATCHER, level=2, action='close_only',
            category='task_failure', agent_role='some-agent',
            resolution=resolution,
        )
        assert result == 'stale_task_scoped', f'Expected match, got: {result!r}'

    def test_stale_task_scoped_rescoped_citation(self) -> None:
        resolution = 'Subject task was re-scoped per get_task; escalation moot.'
        result = l2_auto_close_class(
            identity=self.WATCHER, level=2, action='close_only',
            category='task_failure', agent_role='some-agent',
            resolution=resolution,
        )
        assert result == 'stale_task_scoped', f'Expected match, got: {result!r}'

    def test_stale_task_scoped_redispatched_citation(self) -> None:
        resolution = 'Subject task was re-dispatched per get_task; escalation moot.'
        result = l2_auto_close_class(
            identity=self.WATCHER, level=2, action='close_only',
            category='task_failure', agent_role='some-agent',
            resolution=resolution,
        )
        assert result == 'stale_task_scoped', f'Expected match, got: {result!r}'

    def test_stale_task_scoped_no_status_citation_no_match(self) -> None:
        resolution = 'Looks stale but no live status was checked.'
        result = l2_auto_close_class(
            identity=self.WATCHER, level=2, action='close_only',
            category='task_failure', agent_role='some-agent',
            resolution=resolution,
        )
        assert result is None, f'Expected no match (no status citation), got: {result!r}'

    def test_stale_task_scoped_casual_is_done_phrasing_no_match(self) -> None:
        """A free-standing 'is done' English phrase is NOT a live get_task
        citation — it must not satisfy the evidence requirement, even though
        it reads as plausible closure prose."""
        resolution = 'The work is done, closing this out.'
        result = l2_auto_close_class(
            identity=self.WATCHER, level=2, action='close_only',
            category='task_failure', agent_role='some-agent',
            resolution=resolution,
        )
        assert result is None, (
            f'Expected no match (casual "is done" phrasing is not a status citation), '
            f'got: {result!r}'
        )

    # -- (4) GATING: non-watcher identity / level != 2 / action != close_only --

    def test_non_watcher_identity_no_match_even_with_full_evidence(self) -> None:
        resolution = 'Subject task status=done per get_task; escalation moot.'
        result = l2_auto_close_class(
            identity='some-other-agent', level=2, action='close_only',
            category='task_failure', agent_role='some-agent',
            resolution=resolution,
        )
        assert result is None, f'Expected no match (non-watcher identity), got: {result!r}'

    def test_level_not_2_no_match(self) -> None:
        resolution = 'Subject task status=done per get_task; escalation moot.'
        for level in (0, 1, 3):
            result = l2_auto_close_class(
                identity=self.WATCHER, level=level, action='close_only',
                category='task_failure', agent_role='some-agent',
                resolution=resolution,
            )
            assert result is None, f'Expected no match at level={level}, got: {result!r}'

    def test_non_close_only_action_no_match(self) -> None:
        resolution = 'Subject task status=done per get_task; escalation moot.'
        for action in ('resume', 'restart', 'park', 'abandon'):
            result = l2_auto_close_class(
                identity=self.WATCHER, level=2, action=action,
                category='task_failure', agent_role='some-agent',
                resolution=resolution,
            )
            assert result is None, f'Expected no match for action={action!r}, got: {result!r}'

    # -- (5) DENYLIST-first: applied BEFORE the allowlist --

    def test_design_concern_denied_even_with_valid_class_c_evidence(self) -> None:
        resolution = 'Subject task status=done per get_task; escalation moot.'
        result = l2_auto_close_class(
            identity=self.WATCHER, level=2, action='close_only',
            category='design_concern', agent_role='some-agent',
            resolution=resolution,
        )
        assert result is None, f'Expected denylist to block design_concern, got: {result!r}'

    def test_milestone_gate_denied(self) -> None:
        resolution = 'Subject task status=done per get_task; escalation moot.'
        result = l2_auto_close_class(
            identity=self.WATCHER, level=2, action='close_only',
            category='milestone_gate', agent_role='some-agent',
            resolution=resolution,
        )
        assert result is None, f'Expected denylist to block milestone_gate, got: {result!r}'

    def test_milestone_check_failed_denied_even_for_non_deterministic_role(self) -> None:
        """'milestone_check_failed' is blocked by CATEGORY (not just the
        orchestrator-deterministic role denylist) — defense-in-depth so a
        future filing under a different role stays blocked too."""
        resolution = 'Subject task status=done per get_task; escalation moot.'
        result = l2_auto_close_class(
            identity=self.WATCHER, level=2, action='close_only',
            category='milestone_check_failed', agent_role='some-other-role',
            resolution=resolution,
        )
        assert result is None, (
            f'Expected denylist to block milestone_check_failed, got: {result!r}'
        )

    def test_orchestrator_deterministic_role_denied_even_for_infra_issue(self) -> None:
        resolution = 'live probe: curator paused=false — transient infra self-cleared.'
        result = l2_auto_close_class(
            identity=self.WATCHER, level=2, action='close_only',
            category='infra_issue', agent_role='orchestrator-deterministic',
            resolution=resolution,
        )
        assert result is None, (
            f'Expected denylist to block orchestrator-deterministic role, got: {result!r}'
        )

    # -- (6) DATA SHAPE --

    def test_action_constant_is_close_only(self) -> None:
        assert L2_AUTO_CLOSE_ACTION == 'close_only'

    def test_deny_categories_are_frozenset_containing_expected_members(self) -> None:
        assert isinstance(L2_AUTO_CLOSE_DENY_CATEGORIES, frozenset)
        assert 'design_concern' in L2_AUTO_CLOSE_DENY_CATEGORIES
        assert 'milestone_gate' in L2_AUTO_CLOSE_DENY_CATEGORIES
        assert 'milestone_check_failed' in L2_AUTO_CLOSE_DENY_CATEGORIES

    def test_deny_roles_are_frozenset_containing_expected_members(self) -> None:
        assert isinstance(L2_AUTO_CLOSE_DENY_ROLES, frozenset)
        assert 'orchestrator-deterministic' in L2_AUTO_CLOSE_DENY_ROLES

    def test_allowlist_contains_three_classes_in_specificity_order(self) -> None:
        names = [klass.name for klass in L2_AUTO_CLOSE_ALLOWLIST]
        assert names == ['superseded_main_sweep', 'self_cleared_infra', 'stale_task_scoped'], (
            f'Expected most-specific-first ordering, got: {names!r}'
        )
