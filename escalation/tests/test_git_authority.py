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

import enum

import pytest
from shared.merge_state import MergeState

from escalation.git_authority import GitAuthorityOutcome, found_on_main_response


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
