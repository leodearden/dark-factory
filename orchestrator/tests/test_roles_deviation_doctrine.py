"""Guards for the plan-deviation grant, doctrine splice and escalate_info contract.

What this module guards (PRD ``plans/plan-deviation-recording-prd.md`` leaf γ):
which roles hold the deviation-recording tool, the IMPLEMENTER deviation
doctrine's splice and its render-from-source parity, and the splice of the one
escalate_info contract.

Every assertion is a containment, count or equality check against a NAMED live
constant: no regex over prose, and no scraping of backticked names out of a
prompt (INV-10 guards-exercise-behaviour; PRD D6). The standing rule and its
rationale live in ``orchestrator/tests/_role_splice_contract.py``.
"""

from __future__ import annotations

import pytest
from _role_splice_contract import SpliceContract, assert_nonempty, assert_opens_own_section
from shared.plan_deviation import DEVIATION_KINDS

from orchestrator.agents.deviation_doctrine import render_deviation_doctrine
from orchestrator.agents.roles import (
    _PLAN_DEVIATION_TOOLS,
    IMPLEMENTER_DEVIATION_DOCTRINE,
    ROLES,
)
from orchestrator.artifacts import TaskArtifacts
from orchestrator.mcp import plan_tools

#: The roles that execute a frozen plan and may record where its text was wrong.
_DEVIATION_TOOL_HOLDERS = frozenset({'implementer', 'debugger'})

_PLAN_TOOLS_PREFIX = 'mcp__plan-tools__'

#: The roles that carry the deviation doctrine. Editorial, not derived:
#: DEBUGGER holds the tool but not the doctrine (PRD D5).
_DOCTRINE_ROLES = frozenset({'implementer'})

# Absence only, never presence: a tombstone cannot fail on a rewording of the
# doctrine, only on the old sentence coming back (e.g. a bad conflict resolution).
_RETIRED_INSTRUCTION = 'note it and stop. Do NOT modify the plan'


def _bare_name(qualified: str) -> str:
    return qualified.rsplit('__', 1)[-1]


def _code_span(token: str) -> str:
    # The bare kind `other` is a substring of ordinary prose, so a bare-word
    # check would pass vacuously; the code span is the render's delimiter.
    return '`' + token + '`'


_DOCTRINE_CONTRACT = SpliceContract(
    constant_name='IMPLEMENTER_DEVIATION_DOCTRINE',
    constant=IMPLEMENTER_DEVIATION_DOCTRINE,
    roles=_DOCTRINE_ROLES,
    role_set_name='_DOCTRINE_ROLES',
    capability=lambda role: role.name in _DOCTRINE_ROLES,
    capability_description='the editorial judgement that this role executes frozen plan text',
)


class TestPlanDeviationToolGrant:
    """The allowlist half of PRD D14: who may call the deviation-recording tool."""

    def test_the_deviation_tool_list_is_nonempty(self):
        assert _PLAN_DEVIATION_TOOLS, (
            '_PLAN_DEVIATION_TOOLS is empty, so every containment check in this '
            'class passes vacuously. Restore its record_step_deviation entry.'
        )

    def test_implementer_and_debugger_hold_every_deviation_tool(self):
        missing = {
            name: sorted(set(_PLAN_DEVIATION_TOOLS) - set(ROLES[name].allowed_tools))
            for name in sorted(_DEVIATION_TOOL_HOLDERS)
        }
        offenders = {name: tools for name, tools in missing.items() if tools}
        assert offenders == {}, (
            f'Roles missing deviation tools from allowed_tools: {offenders}. '
            'Append *_PLAN_DEVIATION_TOOLS to their allowed_tools in roles.py.'
        )

    def test_no_other_role_holds_a_deviation_tool(self):
        offenders = {
            name: sorted(set(_PLAN_DEVIATION_TOOLS) & set(role.allowed_tools))
            for name, role in sorted(ROLES.items())
            if name not in _DEVIATION_TOOL_HOLDERS
            and set(_PLAN_DEVIATION_TOOLS) & set(role.allowed_tools)
        }
        assert offenders == {}, (
            f'Roles outside {sorted(_DEVIATION_TOOL_HOLDERS)} hold deviation '
            f'tools: {offenders}. Remove the grant. SIMPLE_TASK is the likely '
            'case: it holds _PLAN_STATUS_TOOLS but has no frozen plan and no '
            'doctrine (PRD D14), so the grant must not ride on _PLAN_STATUS_TOOLS.'
        )

    @pytest.mark.asyncio
    async def test_every_deviation_tool_is_registered_on_the_plan_tools_server(self, tmp_path):
        artifacts = TaskArtifacts(tmp_path / 'wt')
        artifacts.init('5753', 'T', 'd')
        server = plan_tools.create_server(artifacts)
        registry = {tool.name for tool in await server.list_tools()}

        foreign = sorted(
            qualified for qualified in _PLAN_DEVIATION_TOOLS
            if not qualified.startswith(_PLAN_TOOLS_PREFIX)
        )
        assert foreign == [], (
            f'_PLAN_DEVIATION_TOOLS entries not on the plan-tools server: {foreign}. '
            f'Every entry must start with {_PLAN_TOOLS_PREFIX!r}.'
        )
        unregistered = sorted(
            qualified for qualified in _PLAN_DEVIATION_TOOLS
            if _bare_name(qualified) not in registry
        )
        assert unregistered == [], (
            f'_PLAN_DEVIATION_TOOLS names tools plan_tools.create_server does not '
            f'register: {unregistered}. A granted name with no registered tool '
            'leaves the role with a dead grant; fix the name or register the tool.'
        )


class TestDoctrineRenderer:
    """The synthetic half: the render-from-source mechanism can fail."""

    def test_renders_the_tool_name_it_is_given(self):
        assert 'probe_record_tool' in render_deviation_doctrine('probe_record_tool')

    def test_renders_every_deviation_kind_as_a_code_span(self):
        rendered = render_deviation_doctrine('probe_record_tool')
        missing = [kind for kind in sorted(DEVIATION_KINDS) if _code_span(kind) not in rendered]
        assert missing == [], (
            f'render_deviation_doctrine omits deviation kinds: {missing}. It must '
            'render every member of shared.plan_deviation.DEVIATION_KINDS as a '
            'code span, so the prompt never offers a kind the tool refuses.'
        )

    def test_opens_its_own_section(self):
        assert_opens_own_section(
            'render_deviation_doctrine(...)',
            render_deviation_doctrine('probe_record_tool'),
            remedy=(
                'Spliced after IMPLEMENTER\'s "## Important" bullets, it would read '
                'as more of them. Start the render with a `## ` heading.'
            ),
        )


class TestImplementerDeviationDoctrine:
    """The wired half, over the live constants."""

    def test_doctrine_is_nonempty(self):
        assert_nonempty(
            'IMPLEMENTER_DEVIATION_DOCTRINE',
            IMPLEMENTER_DEVIATION_DOCTRINE,
            remedy='It must be render_deviation_doctrine\'s render for the granted tool.',
        )

    def test_the_constant_is_the_render_for_the_granted_tool(self):
        # What keeps the constant derived by construction: re-hand-writing it in
        # roles.py, even word for word today, goes red on the next leaf edit.
        rendered = render_deviation_doctrine(_bare_name(_PLAN_DEVIATION_TOOLS[0]))
        assert rendered == IMPLEMENTER_DEVIATION_DOCTRINE

    def test_doctrine_names_every_deviation_tool_by_bare_name(self):
        missing = [
            _bare_name(qualified) for qualified in _PLAN_DEVIATION_TOOLS
            if _bare_name(qualified) not in IMPLEMENTER_DEVIATION_DOCTRINE
        ]
        assert missing == [], (
            f'IMPLEMENTER_DEVIATION_DOCTRINE does not name deviation tools: {missing}. '
            'Render it from the _PLAN_DEVIATION_TOOLS entry the role is granted.'
        )

    def test_doctrine_names_every_deviation_kind(self):
        missing = [
            kind for kind in sorted(DEVIATION_KINDS)
            if _code_span(kind) not in IMPLEMENTER_DEVIATION_DOCTRINE
        ]
        assert missing == [], (
            f'IMPLEMENTER_DEVIATION_DOCTRINE omits deviation kinds: {missing}. '
            'Re-render it from shared.plan_deviation.DEVIATION_KINDS.'
        )

    # assert_role_set_matches_capability is not called: _DOCTRINE_ROLES is an
    # editorial set (DEBUGGER holds the tool but not the doctrine), not a derivation.

    def test_every_doctrine_role_carries_the_doctrine(self):
        _DOCTRINE_CONTRACT.assert_every_role_carries(
            remedy=(
                'Splice IMPLEMENTER_DEVIATION_DOCTRINE into IMPLEMENTER\'s prompt '
                'right after the "## Important" bullets.'
            ),
        )

    def test_the_doctrine_is_spliced_exactly_once(self):
        # absent_ok=True leaves a missing splice to fail only the containment
        # test above: one root cause, one failing test.
        _DOCTRINE_CONTRACT.assert_spliced_exactly_once(
            absent_ok=True,
            remedy='Keep exactly one IMPLEMENTER_DEVIATION_DOCTRINE term in IMPLEMENTER.',
        )

    def test_no_other_role_carries_the_doctrine(self):
        _DOCTRINE_CONTRACT.assert_no_other_role_carries(
            remedy=(
                'Remove the splice. DEBUGGER is excluded by PRD D5: it holds the '
                'deviation tool but does not execute frozen plan text.'
            ),
        )

    def test_the_retired_instruction_is_gone(self):
        assert _RETIRED_INSTRUCTION not in ROLES['implementer'].system_prompt, (
            'IMPLEMENTER\'s prompt carries the retired "note it and stop" '
            'instruction again, which contradicts the deviation doctrine. Delete it.'
        )
