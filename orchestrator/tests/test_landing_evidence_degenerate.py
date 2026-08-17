"""Tests for the degenerate-branch predicate extracted into landing_evidence.

Task 3103 moves ``harness._is_valid_sha_40`` and the body of
``Harness._branch_is_degenerate`` into ``orchestrator.landing_evidence`` so a
single implementation backs all consumers: the harness (×3 call sites) and
the escalation server's merge_status Tier-3.5 (×2 arms) + merge_request
fast-path.  This module pins:

(A) ``is_valid_sha_40``'s exact acceptance set,
(B) ``branch_is_degenerate``'s fail-open contract and short-circuit ordering,
(C) that ``Harness._branch_is_degenerate`` is a behaviour-preserving
    delegation to the extracted function, over one shared case table,
(D) the ``branch_tip_sha`` override: a caller that already resolved the ref
    judges degeneracy against that SAME observed tip, with no second
    ``git rev-parse``.

(C) is the regression pin that the extraction did not change semantics — the
whole point of the move is that there is exactly ONE implementation, so the
method and the function must agree on every case.
"""
from __future__ import annotations

import types
from typing import Any
from unittest.mock import AsyncMock

import pytest

from orchestrator.harness import Harness
from orchestrator.landing_evidence import branch_is_degenerate, is_valid_sha_40

BASE = 'a' * 40
OTHER = 'b' * 40


# ---------------------------------------------------------------------------
# (A) is_valid_sha_40 — mirrors the moved harness definition exactly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        ('0123456789abcdef' + '0' * 24, True),   # 40-char lowercase hex
        ('a' * 40, True),
        (None, False),
        ('', False),
        ('a' * 39, False),                        # too short
        ('a' * 41, False),                        # too long
        ('A' * 40, False),                        # uppercase hex rejected
        ('a' * 39 + 'g', False),                  # 40 chars, one non-hex
        ('a' * 39 + ' ', False),                  # 40 chars, whitespace
        (123, False),                             # non-str
        (b'a' * 40, False),                       # bytes, not str
        (['a' * 40], False),                      # non-str container
    ],
)
def test_is_valid_sha_40(value: object, expected: bool) -> None:
    assert is_valid_sha_40(value) is expected


# ---------------------------------------------------------------------------
# (B) branch_is_degenerate — shared case table, also driven by (C)
# ---------------------------------------------------------------------------

# (case id, metadata, resolve_branch_sha return, expected, git call expected)
_DEGENERATE_CASES: list[tuple[str, dict[str, Any], Any, bool, bool]] = [
    ('tip_equals_base_is_degenerate', {'branch_base_sha': BASE}, BASE, True, True),
    ('tip_advanced_past_base', {'branch_base_sha': BASE}, OTHER, False, True),
    ('base_key_absent', {}, BASE, False, False),
    ('base_none', {'branch_base_sha': None}, BASE, False, False),
    ('base_empty_string', {'branch_base_sha': ''}, BASE, False, False),
    ('base_too_short', {'branch_base_sha': 'a' * 39}, BASE, False, False),
    ('base_uppercase', {'branch_base_sha': 'A' * 40}, BASE, False, False),
    ('base_non_hex_char', {'branch_base_sha': 'a' * 39 + 'z'}, BASE, False, False),
    ('base_non_str', {'branch_base_sha': 12345}, BASE, False, False),
    ('ref_vanished_mid_sweep', {'branch_base_sha': BASE}, None, False, True),
]

_CASE_IDS = [c[0] for c in _DEGENERATE_CASES]


def _stub_git_ops(tip: str | None) -> types.SimpleNamespace:
    """A git_ops stand-in exposing ONLY ``resolve_branch_sha``.

    Deliberately a ``SimpleNamespace`` rather than a ``MagicMock``: the
    predicate must touch nothing else on git_ops, and any stray attribute
    access here raises AttributeError instead of silently returning a truthy
    auto-mock.  ``branch_is_degenerate`` duck-types its ``git_ops`` parameter
    (see the landing_evidence module docstring), so call sites below carry
    ``# type: ignore[arg-type]`` for the nominal ``GitOps`` annotation.
    """
    return types.SimpleNamespace(resolve_branch_sha=AsyncMock(return_value=tip))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('metadata', 'tip', 'expected', 'git_called'),
    [c[1:] for c in _DEGENERATE_CASES],
    ids=_CASE_IDS,
)
async def test_branch_is_degenerate(
    metadata: dict[str, Any], tip: str | None, expected: bool, git_called: bool,
) -> None:
    """Fail-open contract + short-circuit ordering.

    An absent or malformed ``branch_base_sha`` must return False WITHOUT
    touching git — the cheap metadata check comes first, exactly as in the
    original harness method, so a bogus value never reaches a git comparison.
    """
    git_ops = _stub_git_ops(tip)

    result = await branch_is_degenerate(git_ops, 'task/999', metadata)  # type: ignore[arg-type]

    assert result is expected
    if git_called:
        git_ops.resolve_branch_sha.assert_awaited_once_with('task/999')
    else:
        git_ops.resolve_branch_sha.assert_not_awaited()


# ---------------------------------------------------------------------------
# (C) Delegation parity — the extraction is behaviour-preserving
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('metadata', 'tip', 'expected', 'git_called'),
    [c[1:] for c in _DEGENERATE_CASES],
    ids=_CASE_IDS,
)
async def test_harness_method_matches_extracted_function(
    metadata: dict[str, Any], tip: str | None, expected: bool, git_called: bool,
) -> None:
    """``Harness._branch_is_degenerate`` must agree with the function.

    Invoked as an unbound method against a duck-typed self: the method's only
    ``self`` dependency is ``self.git_ops``, so a full Harness is neither
    needed nor desirable here.
    """
    method_git = _stub_git_ops(tip)
    fn_git = _stub_git_ops(tip)

    method_result = await Harness._branch_is_degenerate(
        types.SimpleNamespace(git_ops=method_git), 'task/999', metadata,  # type: ignore[arg-type]
    )
    fn_result = await branch_is_degenerate(fn_git, 'task/999', metadata)  # type: ignore[arg-type]

    assert method_result == fn_result
    assert method_result is expected
    assert method_git.resolve_branch_sha.await_count == (1 if git_called else 0)


# ---------------------------------------------------------------------------
# (D) branch_tip_sha override — one observed tip, no second ref read
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('supplied_tip', 'expected'),
    [
        (BASE, True),      # supplied tip == recorded base → degenerate
        (OTHER, False),    # supplied tip advanced past base → not degenerate
    ],
    ids=['supplied_tip_equals_base', 'supplied_tip_advanced'],
)
async def test_branch_tip_sha_is_used_verbatim_without_resolving(
    supplied_tip: str, expected: bool,
) -> None:
    """A supplied tip decides the verdict and suppresses resolve_branch_sha.

    The git stub is wired to return the OPPOSITE answer, so a regression that
    re-reads the ref flips the verdict rather than merely costing a
    subprocess.  That is the point of the parameter: the escalation call
    sites have already run an ancestry / patch-id check against a tip they
    observed, and a concurrent warm-lane reseed between the two reads would
    otherwise let the degeneracy verdict be computed against a different SHA
    than the evidence it gates.
    """
    git_ops = _stub_git_ops(OTHER if supplied_tip == BASE else BASE)

    result = await branch_is_degenerate(
        git_ops, 'task/999', {'branch_base_sha': BASE},  # type: ignore[arg-type]
        branch_tip_sha=supplied_tip,
    )

    assert result is expected
    git_ops.resolve_branch_sha.assert_not_awaited()


@pytest.mark.asyncio
async def test_branch_tip_sha_still_short_circuits_on_bad_base() -> None:
    """The metadata check keeps precedence over a supplied tip.

    An absent/malformed ``branch_base_sha`` fails OPEN (False) even when the
    caller hands in a tip, so the backward-compat direction is identical on
    both entry forms.
    """
    git_ops = _stub_git_ops(BASE)

    result = await branch_is_degenerate(
        git_ops, 'task/999', {},  # type: ignore[arg-type]
        branch_tip_sha=BASE,
    )

    assert result is False
    git_ops.resolve_branch_sha.assert_not_awaited()
