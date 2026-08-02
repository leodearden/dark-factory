"""Tests for the degenerate-branch predicate extracted into landing_evidence.

Task 3103 moves ``harness._is_valid_sha_40`` and the body of
``Harness._branch_is_degenerate`` into ``orchestrator.landing_evidence`` so a
single implementation backs all consumers: the harness (×3 call sites) and
the escalation server's merge_status Tier-3.5 (×2 arms) + merge_request
fast-path.  This module pins:

(A) ``is_valid_sha_40``'s exact acceptance set,
(B) ``branch_is_degenerate``'s fail-open contract and short-circuit ordering,
(C) that ``Harness._branch_is_degenerate`` is a behaviour-preserving
    delegation to the extracted function, over one shared case table.

(C) is the regression pin that the extraction did not change semantics — the
whole point of the move is that there is exactly ONE implementation, so the
method and the function must agree on every case.
"""
from __future__ import annotations

import types
from typing import Any
from unittest.mock import AsyncMock

import pytest

import orchestrator.harness as harness_mod
import orchestrator.landing_evidence as landing_evidence_mod
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


def test_harness_rebinds_the_extracted_is_valid_sha_40() -> None:
    """harness._is_valid_sha_40 must BE the extracted function, not a copy.

    The extraction exists to eliminate lockstep duplication; a future
    copy-paste regression that reintroduces a second implementation in
    harness.py would silently pass every behavioural test above but fail
    this identity assertion.
    """
    assert harness_mod._is_valid_sha_40 is landing_evidence_mod.is_valid_sha_40


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

    result = await branch_is_degenerate(git_ops, 'task/999', metadata)

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
        types.SimpleNamespace(git_ops=method_git), 'task/999', metadata,
    )
    fn_result = await branch_is_degenerate(fn_git, 'task/999', metadata)

    assert method_result == fn_result
    assert method_result is expected
    assert method_git.resolve_branch_sha.await_count == (1 if git_called else 0)
