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

from escalation.authority import PROMOTE_ALLOWED, ROLE_LEVEL_ALLOWLIST


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
