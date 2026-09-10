"""Tests for the extracted git-authority landing tier.

THIS FILE COULD NOT HAVE EXISTED BEFORE THE EXTRACTION (task 4887, PRD
`docs/prds/landed-not-done-recovery.md` label ζ).  Every member it exercises
lived as a 4-space-indent sibling inside ``escalation.server.create_server``
and was reachable only by building a whole MCP server and driving the
``merge_status`` tool through it.  So the fact that nothing here constructs a
server IS the workstream's signal — ζ's stated intermediate signal is that
"the tier is importable outside ``create_server``'s scope", and an import at
module level plus direct calls is the most direct way to assert it.

The behavioural pins for the tier as it is reached THROUGH ``merge_status``
stay in ``test_merge_status_git_authority.py``; this file pins the extracted
module's own interface.
"""
from __future__ import annotations

import dataclasses
import enum
import logging
import types
from typing import Any
from unittest.mock import AsyncMock

import pytest
from shared.merge_state import MergeState

from escalation.git_authority import (
    GitAuthorityOutcome,
    TaskMetadataResult,
    found_on_main_response,
    task_metadata,
)


class TestFoundOnMainResponse:
    """The Tier-3.5 done/found_on_main renderer, called with no server."""

    def test_returns_the_exact_response_shape(self) -> None:
        """Whole-dict equality — no key may be added, dropped or renamed.

        A partial ``result['state'] == 'done'`` style assertion would let a
        silently-added key through, and the response shape is task 4831's to
        extend (`plans/merge-status-durable-non-landed-prd.md` D1/D2), not
        this task's.
        """
        assert found_on_main_response('req-1', 'a' * 40) == {
            'state': MergeState.done,
            'request_id': 'req-1',
            'generation': 1,
            'kind': 'found_on_main',
            'merge_sha': 'a' * 40,
            'outcome': 'found_on_main',
        }

    def test_accepts_a_none_request_id(self) -> None:
        """``request_id`` is ``str | None`` — merge_status may resolve by branch."""
        assert found_on_main_response(None, 'b' * 40) == {
            'state': MergeState.done,
            'request_id': None,
            'generation': 1,
            'kind': 'found_on_main',
            'merge_sha': 'b' * 40,
            'outcome': 'found_on_main',
        }

    def test_state_is_a_genuine_str(self) -> None:
        """The wire-compatibility property the shared vocabulary relies on.

        ``MergeState`` is a ``StrEnum``, so the emitted ``state`` is a real
        ``str`` instance and JSON-encodes as its plain spelling.  That is why
        the whole-dict pins above hold against plain-string expectations, and
        why moving the renderer out of server.py cannot change a byte on the
        wire.
        """
        state = found_on_main_response(None, 'c' * 40)['state']

        assert isinstance(state, str), f'state must be a genuine str, got {type(state)}'
        assert state == 'done'


class TestGitAuthorityOutcome:
    """The verdict vocabulary task 4831 (PRD label β) will switch on."""

    def test_is_a_str_enum(self) -> None:
        assert issubclass(GitAuthorityOutcome, enum.StrEnum)

    def test_has_exactly_three_members(self) -> None:
        """A fourth outcome must not appear without a reviewer seeing this move.

        ``landed_unconfirmed`` vs ``no_signal`` is this task's answer to "are
        'landed but unattributable' and 'we know nothing' the same
        proposition?" — they are NOT, and the distinction is materialized
        here, on the verdict, rather than on the ``merge_status`` MCP
        response, whose vocabulary belongs to task 4831.
        """
        assert {m.value for m in GitAuthorityOutcome} == {
            'found_on_main', 'landed_unconfirmed', 'no_signal',
        }

    @pytest.mark.parametrize(
        'member,spelling',
        [
            (GitAuthorityOutcome.found_on_main, 'found_on_main'),
            (GitAuthorityOutcome.landed_unconfirmed, 'landed_unconfirmed'),
            (GitAuthorityOutcome.no_signal, 'no_signal'),
        ],
    )
    def test_members_round_trip_as_their_plain_spelling(
        self, member: GitAuthorityOutcome, spelling: str
    ) -> None:
        assert member == spelling
        assert isinstance(member, str)


def _harness(
    *, task: Any = None, raises: bool = False, with_scheduler: bool = True,
) -> types.SimpleNamespace:
    """Build the minimal duck-typed harness ``task_metadata`` reads.

    The function touches nothing but ``harness.scheduler.get_task``, which is
    why it can be exercised with a two-attribute SimpleNamespace and no MCP
    server anywhere in sight.
    """
    harness = types.SimpleNamespace()
    if with_scheduler:
        harness.scheduler = types.SimpleNamespace(
            get_task=(
                AsyncMock(side_effect=RuntimeError('scheduler unreachable'))
                if raises else AsyncMock(return_value=task)
            ),
        )
    return harness


@pytest.mark.asyncio
class TestTaskMetadata:
    """The G7 fetch-outcome tri-state — a FAULT is not a FACT.

    Before the extraction this helper returned a bare ``{}`` on every failure
    mode, so a caller could not tell "the task genuinely declares no
    metadata" from "we could not read it".  The PRD's G7 requirement
    (``docs/prds/landed-not-done-recovery.md`` label ζ) is that the extracted
    API return the fetch outcome alongside the metadata, so leaf η's writer
    can treat "could not read metadata" as "cannot verify the degeneracy
    guard" rather than as "no degeneracy".
    """

    async def test_healthy_read_with_metadata(self) -> None:
        result = await task_metadata(
            _harness(task={'metadata': {'branch_base_sha': 'b' * 40}}),
            '4887', site='merge_status',
        )

        assert result.metadata == {'branch_base_sha': 'b' * 40}
        assert result.unavailable is False

    @pytest.mark.parametrize(
        'task', [{}, {'metadata': None}, {'metadata': {}}],
        ids=['key_absent', 'key_none', 'key_empty'],
    )
    async def test_healthy_read_without_metadata_is_not_unavailable(
        self, task: dict
    ) -> None:
        """The load-bearing half of the tri-state, with its twin below.

        The read SUCCEEDED and the answer is "this task declares nothing".
        That is a FACT, and it must not be reported with the same shape as
        the FAULT cases — otherwise ``{}`` is exactly as uninformative as it
        was before the extraction.
        """
        result = await task_metadata(_harness(task=task), '4887', site='merge_status')

        assert result.metadata == {}
        assert result.unavailable is False

    @pytest.mark.parametrize(
        'harness,label',
        [
            (None, 'harness absent'),
            (_harness(with_scheduler=False), 'no .scheduler attribute'),
            (_harness(raises=True), 'get_task raised'),
        ],
        ids=['no_harness', 'no_scheduler', 'get_task_raises'],
    )
    async def test_fault_modes_are_flagged_unavailable(
        self, harness: Any, label: str
    ) -> None:
        """The twin: ``{}`` here means "we could not read", not "nothing there".

        All three must ALSO honour the never-raises guarantee — a fault in
        the task store must degrade a single guard, not swallow the whole
        probe.
        """
        result = await task_metadata(harness, '4887', site='merge_status')

        assert result.metadata == {}, f'{label}: must fail open to an empty dict'
        assert result.unavailable is True, f'{label}: must be flagged unavailable'

    async def test_absent_task_is_a_fact_not_a_fault(self) -> None:
        """THE ONE GENUINELY ARGUABLE CELL — ``get_task`` answering None.

        Decided as ``unavailable is False``, i.e. "no such task" is a FACT.

        The flag's discriminant is READ SUCCESS, not guard-verifiability.
        That is forced by the cell above: a task record that is present but
        carries no metadata is pinned False, and it sits in exactly the same
        guard-verifiability position as this one (no ``branch_base_sha``
        either way).  So "cannot verify the degeneracy guard" cannot be what
        the flag means, and the only consistent reading left is fault vs.
        fact.  Here the scheduler was healthy and answered definitively.

        The counter-argument was weighed and rejected: one could call an
        absent record "unreadable" and flag it True on fail-safe grounds.
        But that would make ``unavailable is True`` mean two different things
        — a fault OR a legitimately-absent task — which is precisely the
        conflation this flag exists to remove, and it would leave η unable to
        tell them apart. The module docstring states the same rule.
        """
        result = await task_metadata(_harness(task=None), '4887', site='merge_status')

        assert result.metadata == {}
        assert result.unavailable is False

    async def test_degradation_is_logged_with_the_site_label(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Task 3103 review #3: without ``site`` a submit-path fault was logged
        as a merge_status failure, so an operator grepping for a submit-path
        degradation would not find it.
        """
        with caplog.at_level(logging.WARNING):
            await task_metadata(_harness(raises=True), '4887', site='merge_request')

        assert 'merge_request' in caplog.text, (
            f'the site label must be interpolated into the warning, got: {caplog.text!r}'
        )
        assert 'merge_status' not in caplog.text


class TestTaskMetadataResult:
    """The ``{}``-is-not-proof carrier itself."""

    def test_is_frozen(self) -> None:
        """Immutable so a consumer cannot rewrite the fetch outcome downstream."""
        result = TaskMetadataResult(metadata={})

        with pytest.raises(dataclasses.FrozenInstanceError):
            result.unavailable = True   # type: ignore[misc]

    def test_unavailable_defaults_to_false(self) -> None:
        """Mirrors ``TruthReport.escalation_store_unavailable``: defaulting keeps
        hand-construction ergonomic, so the common healthy case reads cleanly.
        """
        assert TaskMetadataResult(metadata={}).unavailable is False
