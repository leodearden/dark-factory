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

import dataclasses
import importlib.util
import pathlib
import re
import types
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from escalation.authority import (
    CLAIM_TTL,
    COORDINATOR_IDENTITY,
    EXECUTOR_DENY_ROLES,
    EXECUTOR_IDENTITY,
    HOST_FILER_ROLES,
    L2_AUTO_CLOSE_ACTION,
    L2_AUTO_CLOSE_ALLOWLIST,
    L2_AUTO_CLOSE_DENY_CATEGORIES,
    L2_AUTO_CLOSE_DENY_ROLES,
    POLICY_TABLE,
    PROMOTE_ALLOWED,
    PROMOTE_SENTINEL_BOUND_ROLES,
    REVERSIBILITY_WAIVED_ACTIONS,
    REVERSIBLE_POLICY_ACTIONS,
    ROLE_LEVEL_ALLOWLIST,
    ClassSwitch,
    IllegalClassTransition,
    PolicyAction,
    PolicyMatch,
    PolicyRefusal,
    PolicyRow,
    RefusalCode,
    audit_escape_target,
    check_class_transition,
    claim_refusal,
    coordinator_denied,
    executor_policy_class,
    executor_refusal,
    l2_auto_close_class,
    policy_row,
    storm_escape_target,
)
from escalation.categories import EscalationCategory
from escalation.models import Escalation
from escalation.store_identity import QueueKind


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

    def test_runner_curator_category_is_in_the_deny_list(self) -> None:
        """The category the DeterministicRunner ACTUALLY files stays denied.

        Same anti-drift shape as the ``_WATCHER_AUTO_IDENTITY`` pin above, for
        the same documented reason: the string lives as a bare literal in
        ``orchestrator.deterministic_runner`` and again in this package's
        denylist, with nothing otherwise tying the two together — so a rename
        on either side would silently re-open the auto-close hole while every
        test stayed green.

        The orchestrator import is FUNCTION-LOCAL, exactly as
        ``test_watcher_wire_identity_is_mapped_and_promote_allowed`` does it:
        escalation is the lower fleet-wide package and must not module-level
        import orchestrator (see authority.py's module docstring).
        """
        from orchestrator.deterministic_runner import (
            CURATOR_ADJUDICATION_MISSING_CATEGORY,
        )

        assert CURATOR_ADJUDICATION_MISSING_CATEGORY in L2_AUTO_CLOSE_DENY_CATEGORIES, (
            f'The category the DeterministicRunner actually files '
            f'({CURATOR_ADJUDICATION_MISSING_CATEGORY!r}) must be denied auto-close'
        )

    def test_runner_milestone_categories_are_in_the_deny_list(self) -> None:
        """The runner's OTHER two born-at-L2 categories stay denied too.

        Reviewer amendment: 'milestone_gate' and 'milestone_check_failed' sit in
        the same denylist with the same exposure as the curator category above,
        so pinning only the curator one would leave the identical silent-rename
        hole open on its siblings — and leave a reader of authority.py with an
        unexplained asymmetry between members of one frozenset.

        Asserted in a SEPARATE test from the curator pin (not appended to it) so
        a milestone rename and a curator rename identify themselves distinctly
        in the failure report.
        """
        from orchestrator.deterministic_runner import (
            MILESTONE_CHECK_FAILED_CATEGORY,
            MILESTONE_GATE_CATEGORY,
        )

        for category in (MILESTONE_GATE_CATEGORY, MILESTONE_CHECK_FAILED_CATEGORY):
            assert category in L2_AUTO_CLOSE_DENY_CATEGORIES, (
                f'The category the DeterministicRunner actually files '
                f'({category!r}) must be denied auto-close'
            )

    def test_runner_stall_category_is_in_the_deny_list(self) -> None:
        """A re-arming predicate's stall escalation stays denied (PRD C1).

        Separate from the milestone pin above so a rename of the stall
        category identifies itself distinctly in the failure report.
        """
        from orchestrator.deterministic_runner import MILESTONE_CHECK_STALLED_CATEGORY

        assert MILESTONE_CHECK_STALLED_CATEGORY in L2_AUTO_CLOSE_DENY_CATEGORIES

    def test_runner_review_due_category_is_registered_and_denied(self) -> None:
        """The trigger chain's human gate stays denied (PRD C4), and the
        literal the runner files is the registered category, not a copy."""
        from orchestrator.deterministic_runner import REVIEW_DUE_CATEGORY

        assert REVIEW_DUE_CATEGORY in L2_AUTO_CLOSE_DENY_CATEGORIES
        assert REVIEW_DUE_CATEGORY in set(EscalationCategory)


class TestPromoteSentinelBoundRoles:
    """``PROMOTE_SENTINEL_BOUND_ROLES``: member roles whose clusters must be minted under a sentinel id.

    The set ranges over the ``agent_role`` of the MEMBERS being promoted, not
    over the promoting caller's identity, so it must never be confused with
    ``PROMOTE_ALLOWED`` (task 4541 RC#3).
    """

    def test_is_non_empty_frozenset(self) -> None:
        assert isinstance(PROMOTE_SENTINEL_BOUND_ROLES, frozenset)
        assert len(PROMOTE_SENTINEL_BOUND_ROLES) > 0

    def test_recovery_veto_streak_role_is_bound_in_lockstep(self) -> None:
        """The duplicated role string stays pinned to the REAL orchestrator constant.

        ``escalation`` must not module-level import ``orchestrator`` (see
        authority.py's module docstring), so the streak alarm's role string is
        DUPLICATED in ``PROMOTE_SENTINEL_BOUND_ROLES`` rather than imported.
        This function-local import is the only thing that keeps the two sides
        pinned: a rename on either side would otherwise silently re-open the
        esc-5469-11 real-id wrapper while every test stayed green.
        """
        from orchestrator.recovery_emission import RECOVERY_VETO_STREAK_ROLE

        assert RECOVERY_VETO_STREAK_ROLE in PROMOTE_SENTINEL_BOUND_ROLES, (
            f'The role the streak alarm actually files under '
            f'({RECOVERY_VETO_STREAK_ROLE!r}) must be sentinel-bound'
        )

    def test_offline_lane_role_is_bound_in_lockstep(self) -> None:
        """The duplicated lane role string stays pinned to the REAL orchestrator constant."""
        from orchestrator.offline_lane_escalations import LANE_ESCALATION_ROLE

        assert LANE_ESCALATION_ROLE in PROMOTE_SENTINEL_BOUND_ROLES, (
            f'The role the offline lane actually files under '
            f'({LANE_ESCALATION_ROLE!r}) must be sentinel-bound'
        )

    def test_zero_progress_requeue_role_is_bound_in_lockstep(self) -> None:
        """The duplicated zero-progress alarm role stays pinned to the REAL orchestrator constant."""
        from orchestrator.zero_progress_requeue import ZERO_PROGRESS_REQUEUE_ROLE

        assert ZERO_PROGRESS_REQUEUE_ROLE in PROMOTE_SENTINEL_BOUND_ROLES, (
            f'The role the zero-progress requeue alarm actually files under '
            f'({ZERO_PROGRESS_REQUEUE_ROLE!r}) must be sentinel-bound'
        )

    def test_disjoint_from_caller_identities(self) -> None:
        assert 'orchestrator-escalation-watcher-auto' not in PROMOTE_SENTINEL_BOUND_ROLES
        assert PROMOTE_SENTINEL_BOUND_ROLES.isdisjoint(PROMOTE_ALLOWED)

    def test_disjoint_from_the_auto_close_role_denylist(self) -> None:
        assert PROMOTE_SENTINEL_BOUND_ROLES.isdisjoint(L2_AUTO_CLOSE_DENY_ROLES)


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

    def test_stale_task_scoped_status_colon_citation(self) -> None:
        """The anchored 'status=' alternative also accepts the ':' spelling —
        untouched by the task-4192 tightening, pinned here as a regression guard."""
        resolution = 'probe: get_task 604 -> status: cancelled'
        result = l2_auto_close_class(
            identity=self.WATCHER, level=2, action='close_only',
            category='task_failure', agent_role='some-agent',
            resolution=resolution,
        )
        assert result == 'stale_task_scoped', f'Expected match, got: {result!r}'

    def test_stale_task_scoped_rescoped_citation(self) -> None:
        resolution = 'get_task 3821 -> re-scoped to task 3821; escalation moot.'
        result = l2_auto_close_class(
            identity=self.WATCHER, level=2, action='close_only',
            category='task_failure', agent_role='some-agent',
            resolution=resolution,
        )
        assert result == 'stale_task_scoped', f'Expected match, got: {result!r}'

    def test_stale_task_scoped_redispatched_citation(self) -> None:
        resolution = 'Subject task re-dispatched as run-abc123/3821 per get_task; escalation moot.'
        result = l2_auto_close_class(
            identity=self.WATCHER, level=2, action='close_only',
            category='task_failure', agent_role='some-agent',
            resolution=resolution,
        )
        assert result == 'stale_task_scoped', f'Expected match, got: {result!r}'

    @pytest.mark.parametrize(
        'resolution',
        (
            're-scoped into 3821',
            're-scoped as task 3821',
            'rescoped to 3821',
            're-dispatched as task 3821',
            'redispatched as run abc123',
            # '#'-prefixed / quoted spellings: how this repo idiomatically
            # writes a task reference ('Task #841'), so the watcher will reach
            # for them. Punctuation must not disqualify an otherwise-good
            # citation, or a legitimate close silently fails level_forbidden.
            're-scoped to task #3821',
            're-scoped to #3821',
            're-scoped to `3821`',
            're-scoped to "task 3821"',
            're-dispatched as run #abc123',
            're-dispatched as `run-abc123`',
        ),
    )
    def test_stale_task_scoped_cited_spelling_variants(self, resolution: str) -> None:
        """The accepted citation spellings: 're-scoped' takes to/into/as,
        're-dispatched' takes 'as' only; the 'task' word, the '-' between a
        run/task keyword and its identifier, and a '#'/quote/backtick wrapper
        around the identifier are all optional.

        Parametrized rather than looped so every spelling is reported
        independently — a loop would abort the matrix at the first failure.
        """
        result = l2_auto_close_class(
            identity=self.WATCHER, level=2, action='close_only',
            category='task_failure', agent_role='some-agent',
            resolution=resolution,
        )
        assert result == 'stale_task_scoped', (
            f'Expected cited variant {resolution!r} to match, got: {result!r}'
        )

    def test_stale_task_scoped_casual_rescoped_prose_no_match(self) -> None:
        """Casual 're-scoped' prose with no concrete task identifier is NOT a
        live get_task citation — it must not satisfy the evidence requirement."""
        resolution = 'The concern was rescoped last week.'
        result = l2_auto_close_class(
            identity=self.WATCHER, level=2, action='close_only',
            category='task_failure', agent_role='some-agent',
            resolution=resolution,
        )
        assert result is None, (
            f'Expected no match (casual "rescoped" prose cites no task), got: {result!r}'
        )

    def test_stale_task_scoped_casual_redispatched_prose_no_match(self) -> None:
        """Casual 're-dispatched' prose with no concrete run/task identifier is
        NOT a live get_task citation."""
        resolution = 're-dispatched it this morning, all good.'
        result = l2_auto_close_class(
            identity=self.WATCHER, level=2, action='close_only',
            category='task_failure', agent_role='some-agent',
            resolution=resolution,
        )
        assert result is None, (
            f'Expected no match (casual "re-dispatched" prose cites no run/task), got: {result!r}'
        )

    def test_stale_task_scoped_uncited_rescoped_no_match(self) -> None:
        """Naming get_task without quoting WHAT it returned is not evidence:
        this exact string was blessed before task 4192 and must no longer pass."""
        resolution = 'Subject task was re-scoped per get_task; escalation moot.'
        result = l2_auto_close_class(
            identity=self.WATCHER, level=2, action='close_only',
            category='task_failure', agent_role='some-agent',
            resolution=resolution,
        )
        assert result is None, (
            f'Expected no match (re-scoped with no task citation), got: {result!r}'
        )

    def test_stale_task_scoped_uncited_redispatched_no_match(self) -> None:
        """The re-dispatched counterpart of the above — also blessed before
        task 4192, also no longer sufficient."""
        resolution = 'Subject task was re-dispatched per get_task; escalation moot.'
        result = l2_auto_close_class(
            identity=self.WATCHER, level=2, action='close_only',
            category='task_failure', agent_role='some-agent',
            resolution=resolution,
        )
        assert result is None, (
            f'Expected no match (re-dispatched with no run/task citation), got: {result!r}'
        )

    def test_stale_task_scoped_redispatched_wrong_preposition_no_match(self) -> None:
        """The asymmetry is deliberate and load-bearing: 're-scoped' admits
        to/into/as but 're-dispatched' admits 'as' ONLY. Pinned here because
        the other re-dispatched negatives all fail for the missing-identifier
        reason, so widening the alternative to (?:to|into|as) would otherwise
        pass the whole suite while silently dropping the documented contract.
        """
        resolution = 'Subject task re-dispatched to task 3821 per get_task; escalation moot.'
        result = l2_auto_close_class(
            identity=self.WATCHER, level=2, action='close_only',
            category='task_failure', agent_role='some-agent',
            resolution=resolution,
        )
        assert result is None, (
            f'Expected no match ("re-dispatched TO" is not the admitted '
            f'preposition — only "as" is), got: {result!r}'
        )

    def test_stale_task_scoped_digitless_identifier_no_match(self) -> None:
        """The citation's identifier must contain a digit — a word-shaped
        'identifier' that is really just more prose is not a citation. The
        symmetric counterpart of the wrong-preposition guard above: here the
        preposition IS right and only the identifier fails."""
        resolution = 'Subject task re-scoped to task abc per get_task; escalation moot.'
        result = l2_auto_close_class(
            identity=self.WATCHER, level=2, action='close_only',
            category='task_failure', agent_role='some-agent',
            resolution=resolution,
        )
        assert result is None, (
            f'Expected no match (identifier carries no digit), got: {result!r}'
        )

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

    def test_design_concern_denied_even_with_cited_rescoped_evidence(self) -> None:
        """The denylist beats class (c)'s re-scoped citation too, not just its
        'status=' alternative — every other denylist test uses 'status=done',
        so nothing else pins this second door shut."""
        resolution = 're-scoped to task 3821'
        result = l2_auto_close_class(
            identity=self.WATCHER, level=2, action='close_only',
            category='design_concern', agent_role='some-agent',
            resolution=resolution,
        )
        assert result is None, (
            f'Expected denylist to block design_concern despite cited re-scoped '
            f'evidence, got: {result!r}'
        )

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

    def test_milestone_check_stalled_denied_even_for_non_deterministic_role(self) -> None:
        """'milestone_check_stalled' is blocked by CATEGORY alone, exactly as
        'milestone_check_failed' above: the stall of a re-arming predicate
        waits for a human to choose resume or abandon."""
        resolution = 'Subject task status=done per get_task; escalation moot.'
        result = l2_auto_close_class(
            identity=self.WATCHER, level=2, action='close_only',
            category='milestone_check_stalled', agent_role='some-other-role',
            resolution=resolution,
        )
        assert result is None, (
            f'Expected denylist to block milestone_check_stalled, got: {result!r}'
        )

    def test_curator_adjudication_missing_denied_even_for_non_deterministic_role(self) -> None:
        """'curator_adjudication_missing' is blocked by CATEGORY — the
        highest-stakes member of the denylist.

        It is the re-ask ``orchestrator.deterministic_runner`` files when a
        ``human_curator_gate`` task resumes with no
        ``human_curator_adjudicated_at`` stamp, so auto-closing it re-opens
        the task-3181 phantom-done hazard the category exists to stop
        (esc-3181-1 was auto-resolved by the watcher whose own resolution text
        said the curator work was "deliberately NOT executed"). Class (c)
        ``stale_task_scoped`` is both category- AND role-agnostic and accepts
        a bare ``status=done`` marker, so without this denylist entry a filing
        under any role other than ``orchestrator-deterministic`` sails
        straight through — defense-in-depth, exactly as for
        'milestone_check_failed' above."""
        resolution = 'Subject task status=done per get_task; escalation moot.'
        result = l2_auto_close_class(
            identity=self.WATCHER, level=2, action='close_only',
            category='curator_adjudication_missing', agent_role='some-other-role',
            resolution=resolution,
        )
        assert result is None, (
            f'Expected denylist to block curator_adjudication_missing, got: {result!r}'
        )

    def test_review_due_denied_even_for_non_deterministic_role(self) -> None:
        """'review_due' is the human gate of a trigger chain (PRD
        completion-driven-triggers C4): Leo decides whether to run the review,
        so the category alone blocks an auto-close under any role."""
        resolution = 'Subject task status=done per get_task; escalation moot.'
        result = l2_auto_close_class(
            identity=self.WATCHER, level=2, action='close_only',
            category='review_due', agent_role='some-other-role',
            resolution=resolution,
        )
        assert result is None, (
            f'Expected denylist to block review_due, got: {result!r}'
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
        assert 'curator_adjudication_missing' in L2_AUTO_CLOSE_DENY_CATEGORIES
        assert 'review_due' in L2_AUTO_CLOSE_DENY_CATEGORIES

    def test_deny_roles_are_frozenset_containing_expected_members(self) -> None:
        assert isinstance(L2_AUTO_CLOSE_DENY_ROLES, frozenset)
        assert 'orchestrator-deterministic' in L2_AUTO_CLOSE_DENY_ROLES

    def test_allowlist_contains_three_classes_in_specificity_order(self) -> None:
        names = [klass.policy_class for klass in L2_AUTO_CLOSE_ALLOWLIST]
        assert names == ['superseded_main_sweep', 'self_cleared_infra', 'stale_task_scoped'], (
            f'Expected most-specific-first ordering, got: {names!r}'
        )


_VALID_ROW = PolicyRow(
    policy_class='probe_class',
    match=PolicyMatch(),
    actions=frozenset({PolicyAction.CLOSE_ONLY}),
    evidence=(re.compile(r'probe', re.IGNORECASE),),
)


def _row(**overrides: Any) -> PolicyRow:
    """A valid PolicyRow with *overrides* applied, re-running the construction invariants."""
    return dataclasses.replace(_VALID_ROW, **overrides)


class TestPolicyRowInvariants:
    """A PolicyRow refuses to exist in a shape the executor gate could misread."""

    def test_a_valid_row_constructs(self) -> None:
        assert _row().policy_class == 'probe_class'

    @pytest.mark.parametrize('slug', ('Has-Caps', 'has space', 'dash-ed', ''))
    def test_slug_must_be_lower_snake(self, slug: str) -> None:
        with pytest.raises(ValueError, match=re.escape(repr(slug))):
            _row(policy_class=slug)

    def test_empty_actions_refused_naming_slug_and_value(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            _row(policy_class='empty_actions', actions=frozenset())
        assert 'empty_actions' in str(excinfo.value)
        assert 'frozenset()' in str(excinfo.value)

    def test_actions_without_a_resolve_action_refused(self) -> None:
        actions = frozenset({PolicyAction.FILE_TASK, PolicyAction.ADD_DEPENDENCY})
        with pytest.raises(ValueError) as excinfo:
            _row(policy_class='no_resolve', actions=actions)
        message = str(excinfo.value)
        assert 'no_resolve' in message
        assert 'file_task' in message and 'add_dependency' in message

    def test_empty_evidence_refused(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            _row(policy_class='no_evidence', evidence=())
        assert 'no_evidence' in str(excinfo.value)
        assert '()' in str(excinfo.value)

    def test_string_evidence_entry_refused(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            _row(policy_class='str_evidence', evidence=(r'raw-string',))
        assert 'str_evidence' in str(excinfo.value)
        assert 'raw-string' in str(excinfo.value)

    def test_reversible_iff_every_act_is_a_reversible_action(self) -> None:
        assert _row().reversible is True
        filing = _row(actions=frozenset({PolicyAction.FILE_TASK, PolicyAction.CLOSE_ONLY}))
        assert filing.reversible is True
        for irreversible in (PolicyAction.DONE_FLIP, PolicyAction.RECOVER_MAIN, PolicyAction.B3_APPLY):
            row = _row(actions=frozenset({irreversible, PolicyAction.CLOSE_ONLY}))
            assert row.reversible is False, irreversible

    def test_no_act_is_both_reversible_and_waived(self) -> None:
        """A waiver makes an act stampable, not reversible; an act in both sets
        would make every row allowing it read reversible."""
        overlap = REVERSIBLE_POLICY_ACTIONS & REVERSIBILITY_WAIVED_ACTIONS
        assert not overlap, sorted(map(str, overlap))

    def test_audit_full_until_is_20_for_file_writing_or_recover_main(self) -> None:
        assert _row().audit_full_until == 0
        assert _row(file_writing=True).audit_full_until == 20
        recovering = _row(actions=frozenset({PolicyAction.RECOVER_MAIN, PolicyAction.CLOSE_ONLY}))
        assert recovering.audit_full_until == 20

    def test_evidence_present_needs_every_pattern(self) -> None:
        row = _row(evidence=(
            re.compile(r'alpha', re.IGNORECASE),
            re.compile(r'beta', re.IGNORECASE),
        ))
        assert row.evidence_present('ALPHA then Beta')
        assert not row.evidence_present('alpha only')
        assert not row.evidence_present('')

    def test_missing_evidence_names_each_unmatched_pattern_in_row_order(self) -> None:
        row = _row(evidence=(
            re.compile(r'alpha', re.IGNORECASE),
            re.compile(r'beta', re.IGNORECASE),
        ))
        assert row.missing_evidence('ALPHA then Beta') == ()
        assert row.missing_evidence('beta only') == ('alpha',)
        assert row.missing_evidence('') == ('alpha', 'beta')

    def test_table_evidence_is_case_insensitive(self) -> None:
        row = policy_row('landing_provenance')
        assert row is not None
        assert row.evidence_present(
            'FOUND_ON_MAIN AT 1A2B3C4D; GIT MERGE-BASE --IS-ANCESTOR 1A2B3C4D MAIN RC=0'
        )


class TestPolicyMatch:
    """PolicyMatch reads only the record's category and agent_role."""

    def test_unconstrained_match_accepts_anything(self) -> None:
        match = PolicyMatch()
        assert match.matches('infra_issue', 'some-agent')
        assert match.matches('task_failure', '')

    def test_categories_must_contain_the_record_category(self) -> None:
        match = PolicyMatch(categories=frozenset({EscalationCategory.INFRA_ISSUE}))
        assert match.matches('infra_issue', 'any-role')
        assert not match.matches('task_failure', 'any-role')

    def test_agent_role_is_a_full_match(self) -> None:
        match = PolicyMatch(agent_role=re.compile(r'orchestrator-main-sweep'))
        assert match.matches('task_failure', 'orchestrator-main-sweep')
        assert not match.matches('task_failure', 'orchestrator-main-sweep-2')
        assert not match.matches('task_failure', 'x-orchestrator-main-sweep')

    def test_category_and_role_are_anded(self) -> None:
        match = PolicyMatch(
            categories=frozenset({EscalationCategory.RISK_IDENTIFIED}),
            agent_role=re.compile(r'orchestrator-recovery-veto-streak'),
        )
        assert match.matches('risk_identified', 'orchestrator-recovery-veto-streak')
        assert not match.matches('risk_identified', 'some-agent')
        assert not match.matches('infra_issue', 'orchestrator-recovery-veto-streak')


class TestPolicyTableRows:
    """POLICY_TABLE: the one enumeration of executor classes (PRD §5.1)."""

    _FOUND_ON_MAIN = 'found_on_main at 1a2b3c4d; git merge-base --is-ancestor 1a2b3c4d main rc=0'
    _FOUND_ON_MAIN_NO_SHA = 'found_on_main; git merge-base --is-ancestor main rc=0'
    _CLAIM_TRUE = 'get_task 4321 status=done; git merge-base --is-ancestor 1a2b3c4d main rc=0'
    _CLAIM_TRUE_NO_ANCESTRY = 'get_task 4321 status=done; 1a2b3c4d'
    _CLAIM_TRUE_NO_SHA = 'get_task 4321 status=done; git merge-base --is-ancestor main rc=0'

    def test_table_is_a_tuple_of_rows_with_unique_slugs(self) -> None:
        assert isinstance(POLICY_TABLE, tuple)
        assert all(isinstance(row, PolicyRow) for row in POLICY_TABLE)
        slugs = [row.policy_class for row in POLICY_TABLE]
        assert len(slugs) == len(set(slugs)), f'duplicate slugs in {slugs!r}'

    def test_every_class_starts_shadow(self) -> None:
        not_shadow = {
            row.policy_class: row.initial_state
            for row in POLICY_TABLE if row.initial_state is not ClassSwitch.SHADOW
        }
        assert not not_shadow, f'every class must start shadow (decision 7): {not_shadow!r}'

    def test_no_row_allows_park_restart_or_abandon(self) -> None:
        for row in POLICY_TABLE:
            assert {'park', 'restart', 'abandon'}.isdisjoint(row.actions), row.policy_class

    def test_unknown_slug_has_no_row(self) -> None:
        assert policy_row('nope') is None

    def test_every_row_is_reachable_by_its_slug(self) -> None:
        for row in POLICY_TABLE:
            assert policy_row(row.policy_class) is row

    def test_landing_provenance_row(self) -> None:
        row = policy_row('landing_provenance')
        assert row is not None
        assert row.match.matches('provenance_unattributed', 'some-agent')
        assert not row.match.matches('provenance_conflict', 'some-agent')
        assert {PolicyAction.DONE_FLIP, PolicyAction.CLOSE_ONLY} <= row.actions
        assert row.reversible is False
        assert row.evidence_present(self._FOUND_ON_MAIN)
        assert not row.evidence_present(self._FOUND_ON_MAIN_NO_SHA)

    def test_completion_claim_true_closes_the_gate_record(self) -> None:
        """Sitting item 522 (Leo, 2026-11-09): the TRUE arm closes; nothing is abandoned."""
        row = policy_row('completion_claim_true')
        assert row is not None
        assert row.match.matches('unverified_completion_claim', 'some-agent')
        assert not row.match.matches('task_failure', 'some-agent')
        assert row.actions == frozenset({PolicyAction.CLOSE_ONLY})
        assert row.evidence_present(self._CLAIM_TRUE)
        assert not row.evidence_present(self._CLAIM_TRUE_NO_ANCESTRY)
        assert not row.evidence_present(self._CLAIM_TRUE_NO_SHA)

    def test_found_on_main_matches_any_category(self) -> None:
        row = policy_row('found_on_main')
        assert row is not None
        for category in ('task_failure', 'infra_issue', 'provenance_unattributed'):
            assert row.match.matches(category, 'some-agent')
        assert row.evidence_present(self._FOUND_ON_MAIN)

    @pytest.mark.parametrize('slug', ('red_main_recovery', 'b3_low_risk_unblock'))
    def test_file_writing_rows_are_fully_audited(self, slug: str) -> None:
        row = policy_row(slug)
        assert row is not None
        assert row.file_writing is True
        assert row.audit_full_until == 20

    def test_only_the_recon_row_serves_the_reconciliation_queue(self) -> None:
        recon = policy_row('recon_run_self_cleared')
        assert recon is not None
        assert recon.queue_kind == 'reconciliation'
        others = {
            row.policy_class: row.queue_kind
            for row in POLICY_TABLE
            if row is not recon and row.queue_kind != 'project'
        }
        assert not others, f'only recon_run_self_cleared may serve the recon queue: {others!r}'

    def test_carve_out_rows_are_table_rows_in_specificity_order(self) -> None:
        assert [row.policy_class for row in L2_AUTO_CLOSE_ALLOWLIST] == [
            'superseded_main_sweep', 'self_cleared_infra', 'stale_task_scoped',
        ]
        for row in L2_AUTO_CLOSE_ALLOWLIST:
            assert any(row is table_row for table_row in POLICY_TABLE), row.policy_class


_NOW = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)
_FRESH = (_NOW - timedelta(minutes=10)).isoformat()
_STALE = (_NOW - timedelta(hours=7)).isoformat()
_FOUND_ON_MAIN_TEXT = (
    'found_on_main at 1a2b3c4d; git merge-base --is-ancestor 1a2b3c4d main rc=0'
)


def _record(
    *,
    category: str = 'provenance_unattributed',
    agent_role: str = 'implementer',
    policy_class: str | None = 'landing_provenance',
    claimed_at: str | None = _FRESH,
    claimed_by: str | None = EXECUTOR_IDENTITY,
) -> Escalation:
    return Escalation(
        id='esc-8422-1', task_id='8422', agent_role=agent_role, severity='blocking',
        category=category, summary='s', level=2, policy_class=policy_class,
        claimed_at=claimed_at, claimed_by=claimed_by,
    )


_LIVE = {'landing_provenance': ClassSwitch.LIVE}


def _executor(
    record: Escalation,
    *,
    action: str = 'close_only',
    resolution: str = _FOUND_ON_MAIN_TEXT,
    state: dict[str, ClassSwitch] | None = None,
    queue_kind: QueueKind = 'project',
) -> PolicyRefusal | None:
    return executor_refusal(
        record, action, resolution, _LIVE if state is None else state,
        now=_NOW, queue_kind=queue_kind,
    )


def _claim(
    record: Escalation,
    policy_class: str = 'landing_provenance',
    *,
    state: dict[str, ClassSwitch] | None = None,
    clean: bool = False,
    queue_kind: QueueKind = 'project',
) -> PolicyRefusal | None:
    return claim_refusal(
        record, policy_class, {} if state is None else state,
        now=_NOW, clean=clean, queue_kind=queue_kind,
    )


def _unclaimed(**overrides: str) -> Escalation:
    return _record(policy_class=None, claimed_at=None, claimed_by=None, **overrides)


_DASHBOARD_WATCHDOG_PATH = pathlib.Path(__file__).parents[2] / 'scripts' / 'dashboard-watchdog.py'


def _load_dashboard_watchdog() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location('dashboard_watchdog', _DASHBOARD_WATCHDOG_PATH)
    assert spec is not None and spec.loader is not None, _DASHBOARD_WATCHDOG_PATH
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestExecutorIdentity:
    def test_executor_identity_resolves_levels_0_to_2(self) -> None:
        assert EXECUTOR_IDENTITY == 'orchestrator-escalation-executor'
        assert ROLE_LEVEL_ALLOWLIST[EXECUTOR_IDENTITY] == frozenset({0, 1, 2})

    def test_coordinator_keeps_full_levels(self) -> None:
        assert COORDINATOR_IDENTITY == 'interactive:coordinator'
        assert COORDINATOR_IDENTITY not in ROLE_LEVEL_ALLOWLIST

    def test_both_identities_may_promote(self) -> None:
        assert EXECUTOR_IDENTITY in PROMOTE_ALLOWED
        assert COORDINATOR_IDENTITY in PROMOTE_ALLOWED

    def test_executor_deny_roles_extend_the_auto_close_deny_roles(self) -> None:
        watchdogs = {'orchestrator-watchdog', 'harness-dashboard-watchdog'}
        assert L2_AUTO_CLOSE_DENY_ROLES | watchdogs == EXECUTOR_DENY_ROLES

    def test_host_filer_roles_are_the_two_watchdogs_the_executor_denies(self) -> None:
        assert {'orchestrator-watchdog', 'harness-dashboard-watchdog'} == HOST_FILER_ROLES
        assert EXECUTOR_DENY_ROLES == L2_AUTO_CLOSE_DENY_ROLES | HOST_FILER_ROLES

    def test_dashboard_watchdog_role_is_denied_in_lockstep(self) -> None:
        """The deny entry names the role the dashboard watchdog REALLY files under.

        The alarm is filed by a hyphenated script, so its role string is
        duplicated here rather than imported; loading the script is the only
        thing that keeps the two pinned.
        """
        watchdog = _load_dashboard_watchdog()
        assert watchdog.ESCALATION_AGENT_ROLE in EXECUTOR_DENY_ROLES

    def test_real_dashboard_watchdog_alarm_is_neither_claimable_nor_closable(self) -> None:
        watchdog = _load_dashboard_watchdog()
        alarm = Escalation(
            id='esc-dashboard-watchdog-restart-ceiling-1',
            task_id=watchdog.ESCALATION_TASK_SENTINEL,
            agent_role=watchdog.ESCALATION_AGENT_ROLE,
            severity=watchdog.ESCALATION_SEVERITY,
            category='infra_issue', summary='restart ceiling hit', level=2,
        )
        claim = _claim(alarm, 'self_cleared_infra')
        assert claim is not None
        assert claim.code is RefusalCode.DENIED_RECORD

        claimed = dataclasses.replace(
            alarm, policy_class='self_cleared_infra',
            claimed_at=_FRESH, claimed_by=EXECUTOR_IDENTITY,
        )
        close = _executor(
            claimed, resolution='ActiveState=active',
            state={'self_cleared_infra': ClassSwitch.LIVE},
        )
        assert close is not None
        assert close.code is RefusalCode.DENIED_RECORD

    def test_claim_ttl_is_the_six_hour_triage_ack_window(self) -> None:
        assert timedelta(hours=6) == CLAIM_TTL

    def test_carve_out_stays_watcher_only(self) -> None:
        assert l2_auto_close_class(
            identity=EXECUTOR_IDENTITY, level=2, action='close_only',
            category='task_failure', agent_role='some-agent',
            resolution='Subject task status=done per get_task; escalation moot.',
        ) is None


class TestExecutorRefusal:
    """executor_refusal checks in PRD §5.1 order and names what failed."""

    def test_everything_valid_is_not_refused(self) -> None:
        record = _record()
        assert _executor(record) is None
        assert executor_policy_class(
            record, 'close_only', _FOUND_ON_MAIN_TEXT, _LIVE, now=_NOW, queue_kind='project',
        ) == 'landing_provenance'

    def test_deny_category_is_denied(self) -> None:
        refusal = _executor(_record(category='milestone_gate'))
        assert refusal is not None
        assert refusal.code is RefusalCode.DENIED_RECORD
        assert 'milestone_gate' in refusal.detail

    @pytest.mark.parametrize('agent_role', sorted(EXECUTOR_DENY_ROLES))
    def test_deny_role_is_denied(self, agent_role: str) -> None:
        refusal = _executor(_record(agent_role=agent_role))
        assert refusal is not None
        assert refusal.code is RefusalCode.DENIED_RECORD
        assert agent_role in refusal.detail

    def test_deny_role_outranks_a_missing_claim(self) -> None:
        refusal = _executor(_unclaimed(agent_role='orchestrator-watchdog'))
        assert refusal is not None
        assert refusal.code is RefusalCode.DENIED_RECORD

    @pytest.mark.parametrize(
        ('claimed_at', 'claimed_by'),
        (
            pytest.param(None, None, id='no-claim'),
            pytest.param(_STALE, EXECUTOR_IDENTITY, id='seven-hours-old'),
            pytest.param((_NOW - CLAIM_TTL).isoformat(), EXECUTOR_IDENTITY, id='exactly-ttl'),
            pytest.param(_FRESH, 'interactive', id='other-claimant'),
            pytest.param('not-a-timestamp', EXECUTOR_IDENTITY, id='unparsable'),
        ),
    )
    def test_claim_must_be_fresh_and_the_executors(
        self, claimed_at: str | None, claimed_by: str | None,
    ) -> None:
        refusal = _executor(_record(claimed_at=claimed_at, claimed_by=claimed_by))
        assert refusal is not None
        assert refusal.code is RefusalCode.NO_FRESH_CLAIM

    def test_unknown_class(self) -> None:
        refusal = _executor(_record(policy_class='nope'))
        assert refusal is not None
        assert refusal.code is RefusalCode.UNKNOWN_CLASS
        assert 'nope' in refusal.detail

    def test_row_mismatch_on_category(self) -> None:
        refusal = _executor(_record(category='task_failure'))
        assert refusal is not None
        assert refusal.code is RefusalCode.ROW_MISMATCH
        assert 'task_failure' in refusal.detail

    def test_recon_row_does_not_serve_a_project_queue(self) -> None:
        record = _record(category='recon_stale_run', policy_class='recon_run_self_cleared')
        refusal = _executor(record, state={'recon_run_self_cleared': ClassSwitch.LIVE})
        assert refusal is not None
        assert refusal.code is RefusalCode.ROW_MISMATCH
        assert 'project' in refusal.detail

    def test_project_row_does_not_serve_the_recon_queue(self) -> None:
        refusal = _executor(_record(), queue_kind='reconciliation')
        assert refusal is not None
        assert refusal.code is RefusalCode.ROW_MISMATCH
        assert 'reconciliation' in refusal.detail

    def test_action_must_be_allowed_by_the_row(self) -> None:
        refusal = _executor(_record(), action='resume')
        assert refusal is not None
        assert refusal.code is RefusalCode.ACTION_NOT_ALLOWED
        assert 'resume' in refusal.detail

    @pytest.mark.parametrize(
        'state',
        (
            pytest.param({'landing_provenance': ClassSwitch.SHADOW}, id='shadow'),
            pytest.param({'landing_provenance': ClassSwitch.OFF}, id='off'),
            pytest.param({}, id='absent-means-initial-shadow'),
        ),
    )
    def test_class_must_be_live(self, state: dict[str, ClassSwitch]) -> None:
        refusal = _executor(_record(), state=state)
        assert refusal is not None
        assert refusal.code is RefusalCode.CLASS_NOT_LIVE

    def test_evidence_missing_names_the_unmatched_pattern(self) -> None:
        text = 'found_on_main; git merge-base --is-ancestor main rc=0'
        refusal = _executor(_record(), resolution=text)
        assert refusal is not None
        assert refusal.code is RefusalCode.EVIDENCE_MISSING
        row = policy_row('landing_provenance')
        assert row is not None
        missing = row.missing_evidence(text)
        assert missing
        for pattern in missing:
            assert pattern in refusal.detail

    def test_any_refusal_projects_to_no_class(self) -> None:
        assert executor_policy_class(
            _record(), 'resume', _FOUND_ON_MAIN_TEXT, _LIVE, now=_NOW, queue_kind='project',
        ) is None


class TestClaimRefusal:
    def test_shadow_class_is_claimable(self) -> None:
        assert _claim(_unclaimed()) is None

    def test_deny_record(self) -> None:
        refusal = _claim(_unclaimed(agent_role='orchestrator-watchdog'))
        assert refusal is not None
        assert refusal.code is RefusalCode.DENIED_RECORD

    def test_unknown_class(self) -> None:
        refusal = _claim(_unclaimed(), 'nope')
        assert refusal is not None
        assert refusal.code is RefusalCode.UNKNOWN_CLASS

    def test_row_mismatch(self) -> None:
        refusal = _claim(_unclaimed(category='task_failure'))
        assert refusal is not None
        assert refusal.code is RefusalCode.ROW_MISMATCH

    def test_off_class_is_not_claimable(self) -> None:
        refusal = _claim(_unclaimed(), state={'landing_provenance': ClassSwitch.OFF})
        assert refusal is not None
        assert refusal.code is RefusalCode.CLASS_OFF

    def test_fresh_claim_blocks_a_second(self) -> None:
        refusal = _claim(_record())
        assert refusal is not None
        assert refusal.code is RefusalCode.ALREADY_CLAIMED

    def test_stale_claim_is_reclaimable(self) -> None:
        assert _claim(_record(claimed_at=_STALE)) is None

    def test_file_writing_row_needs_a_clean_project_root(self) -> None:
        record = _unclaimed(category='preexisting_main_break')
        refusal = _claim(record, 'red_main_recovery', clean=False)
        assert refusal is not None
        assert refusal.code is RefusalCode.PROJECT_ROOT_DIRTY
        assert _claim(record, 'red_main_recovery', clean=True) is None


class TestCoordinatorDenied:
    def test_fresh_executor_claim_denies_the_coordinator(self) -> None:
        assert coordinator_denied(_record(), _NOW) is True

    @pytest.mark.parametrize(
        ('claimed_at', 'claimed_by'),
        (
            pytest.param(_STALE, EXECUTOR_IDENTITY, id='seven-hours-old'),
            pytest.param((_NOW - CLAIM_TTL).isoformat(), EXECUTOR_IDENTITY, id='exactly-ttl'),
            pytest.param(None, None, id='no-claim'),
            pytest.param('not-a-timestamp', EXECUTOR_IDENTITY, id='unparsable'),
            pytest.param(_FRESH, 'interactive', id='non-executor-claimant'),
        ),
    )
    def test_anything_else_leaves_the_coordinator_free(
        self, claimed_at: str | None, claimed_by: str | None,
    ) -> None:
        record = _record(claimed_at=claimed_at, claimed_by=claimed_by)
        assert coordinator_denied(record, _NOW) is False


class TestClassTransitions:
    @pytest.mark.parametrize(
        ('current', 'target'),
        (
            (ClassSwitch.OFF, ClassSwitch.SHADOW),
            (ClassSwitch.SHADOW, ClassSwitch.LIVE),
            (ClassSwitch.LIVE, ClassSwitch.SHADOW),
        ),
    )
    def test_allowed_transitions(self, current: ClassSwitch, target: ClassSwitch) -> None:
        check_class_transition('landing_provenance', current, target)

    @pytest.mark.parametrize(
        ('current', 'target'),
        (
            (ClassSwitch.SHADOW, ClassSwitch.OFF),
            (ClassSwitch.LIVE, ClassSwitch.OFF),
            (ClassSwitch.OFF, ClassSwitch.LIVE),
        ),
    )
    def test_illegal_transitions_name_slug_current_and_target(
        self, current: ClassSwitch, target: ClassSwitch,
    ) -> None:
        with pytest.raises(IllegalClassTransition) as excinfo:
            check_class_transition('landing_provenance', current, target)
        assert isinstance(excinfo.value, ValueError)
        message = str(excinfo.value)
        assert 'landing_provenance' in message
        assert str(current) in message and str(target) in message

    def test_storm_escape_drops_a_live_class_over_its_cap_to_shadow(self) -> None:
        assert storm_escape_target(ClassSwitch.LIVE, 31, 30) is ClassSwitch.SHADOW

    @pytest.mark.parametrize(
        ('current', 'closes'),
        ((ClassSwitch.LIVE, 30), (ClassSwitch.SHADOW, 99), (ClassSwitch.OFF, 99)),
    )
    def test_storm_escape_leaves_everything_else(self, current: ClassSwitch, closes: int) -> None:
        assert storm_escape_target(current, closes, 30) is None


class TestAuditEscape:
    """The audit flip rule (PRD §3 decision 8): a live class whose audited closes run wrong drops to shadow."""

    @pytest.mark.parametrize(
        ('audited', 'wrong', 'expected'),
        (
            (3, 1, ClassSwitch.SHADOW),
            (3, 0, None),
            (20, 1, None),
            (20, 2, ClassSwitch.SHADOW),
            (40, 2, None),
        ),
        ids=('below-n-any-wrong', 'below-n-none-wrong', 'at-the-ceiling', 'above-the-ceiling', 'under-the-ceiling'),
    )
    def test_a_live_class(self, audited: int, wrong: int, expected: ClassSwitch | None) -> None:
        assert audit_escape_target(ClassSwitch.LIVE, audited, wrong) is expected

    @pytest.mark.parametrize('current', (ClassSwitch.SHADOW, ClassSwitch.OFF))
    @pytest.mark.parametrize(('audited', 'wrong'), ((3, 1), (20, 20), (0, 0)))
    def test_nothing_to_escape_unless_live(self, current: ClassSwitch, audited: int, wrong: int) -> None:
        assert audit_escape_target(current, audited, wrong) is None

    @pytest.mark.parametrize(('audited', 'wrong'), ((1, 2), (-1, 0), (3, -1)))
    def test_impossible_counts_are_refused_naming_both(self, audited: int, wrong: int) -> None:
        with pytest.raises(ValueError) as excinfo:
            audit_escape_target(ClassSwitch.LIVE, audited, wrong)
        message = str(excinfo.value)
        assert str(audited) in message and str(wrong) in message
