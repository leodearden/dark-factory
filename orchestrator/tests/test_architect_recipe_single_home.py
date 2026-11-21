"""The architect's plan-tools recipe has one home (task 5532).

Every parameterised plan-tools call the recipe spells lives in
``orchestrator/src/orchestrator/agents/roles.py::ARCHITECT_PLAN_TOOLS_RECIPE``,
which ``ARCHITECT.system_prompt`` carries into every architect dispatch. A
dispatched architect reads that system prompt plus one briefing, so no briefing
workflow.py hands to ARCHITECT may re-spell a recipe signature. ``_ARCHITECT_DISPATCHES``
renders each such briefing builder with every optional section its signature
declares, and is held equal to the builders
``_workflow_architect_dispatch.architect_briefing_builders`` finds in workflow.py.
"""

from __future__ import annotations

import copy
import inspect
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

import pytest
from _briefing_helpers import briefing as briefing
from _role_splice_contract import SpliceContract, assert_nonempty, assert_opens_own_section
from _workflow_architect_dispatch import architect_briefing_builders
from shared.task_text_fingerprint import TaskTextDrift

from orchestrator.agents.briefing import BriefingAssembler
from orchestrator.agents.roles import ARCHITECT, ARCHITECT_PLAN_TOOLS_RECIPE
from orchestrator.escalation_rulings import Ruling
from orchestrator.plan_descope import DoneStepIntegrityVerdict
from orchestrator.review_lineage import RetiredReview

_RECIPE_CONTRACT = SpliceContract(
    constant_name='ARCHITECT_PLAN_TOOLS_RECIPE',
    constant=ARCHITECT_PLAN_TOOLS_RECIPE,
    roles=frozenset({ARCHITECT.name}),
    role_set_name='_RECIPE_CONTRACT.roles',
    capability=lambda role: role.name == ARCHITECT.name,
    capability_description='being the architect',
)

_RECIPE_TOOL_NAME = re.compile(r'`(\w+)\(')


def _recipe_tool_names() -> list[str]:
    """Every tool the plan-tools recipe spells as a backticked call."""
    names = sorted(set(_RECIPE_TOOL_NAME.findall(ARCHITECT_PLAN_TOOLS_RECIPE)))
    assert names, (
        'ARCHITECT_PLAN_TOOLS_RECIPE spells no backticked tool call, which makes '
        'every recipe-signature check below vacuous — either the recipe stopped '
        'spelling its tool calls or _RECIPE_TOOL_NAME no longer matches how it '
        'spells them.'
    )
    return names


def _recipe_signature_spellings(text: str) -> list[str]:
    """Every recipe tool call in *text* that spells a parameter list.

    Any parameter list counts, whatever its names, and so does an
    ``mcp__plan-tools__``-qualified spelling. ``tool(...)`` names the tool
    without restating its signature, and ``confirm_plan()`` has no
    parameters to drift.
    """
    return [
        match.group(0)
        for name in _recipe_tool_names()
        for match in re.finditer(re.escape(name) + r'\(([^)`]+)\)', text)
        if match.group(1) != '...'
    ]


_SPOT_TASK = {
    'id': '5532',
    'title': 'Recipe single-home task',
    'description': 'The task every recipe single-home dispatch briefs.',
    'metadata': {
        'dry_run_proposals': [
            {
                'proposal_text': 'Root cause is a stale cache entry; fix by invalidating on write.',
                'risk_label': 'medium',
                'files_referenced': ['orchestrator/x.py'],
                'timestamp': '2026-07-13T10:00:00+00:00',
                'investigated_at': '2026-07-13T10:00:00+00:00',
            },
        ],
    },
}

_SPOT_COMMITTED_WORK = [{'sha': 'a' * 40, 'subject': 'feat: an earlier attempt'}]

_SPOT_RULINGS = (
    Ruling(
        escalation_id='esc-5532-9', level=2, summary='keep or drop the write-through cache?',
        resolution='KEEP the write-through cache — operator ruling',
        resolved_at='2026-10-18T00:00:00+00:00', resolved_by='interactive',
        resolution_action='resume',
    ),
)

_SPOT_RETIRED_REVIEWS = (
    RetiredReview(
        branch='refs/heads/task/5532-20261112T000000Z',
        tip_sha='c' * 12,
        parking='/q/5532-20261112T000000Z',
        quarantined_at='2026-11-12T00:00:00+00:00',
        amendment_rounds_total=1,
        review_cycles_total=0,
        archive=Path('/meta/task-5532/retired-reviews/5532-20261112T000000Z'),
        suggestions=(
            {
                'reviewer': 'reviewer_comprehensive',
                'location': 'orchestrator/x.py:12',
                'category': 'correctness',
                'description': 'the cache invalidation misses the delete path',
                'suggested_fix': 'invalidate on delete as well as on write',
            },
        ),
    ),
)

_SPOT_INTEGRITY_REFUSAL = DoneStepIntegrityVerdict(
    undisposed_drops=(('s1', 'b' * 40),),
    introduced_duplicates=(),
    invalid_descope_records=(),
)

_SPOT_PLAN = {
    'files': ['a.py'],
    'steps': [{'id': 's1', 'type': 'test', 'description': 'x', 'status': 'pending'}],
}


class _ArchitectDispatch(NamedTuple):
    builder: str
    args: tuple[object, ...]
    options: Mapping[str, object]


def _dispatch(builder: str, *args: object, **options: object) -> _ArchitectDispatch:
    return _ArchitectDispatch(builder, args, MappingProxyType(options))


_ARCHITECT_DISPATCHES: Mapping[str, _ArchitectDispatch] = MappingProxyType({
    'fresh': _dispatch('build_architect_prompt', _SPOT_TASK),
    'fresh_with_optional_sections': _dispatch(
        'build_architect_prompt',
        _SPOT_TASK,
        include_prior_proposals=True,
        committed_work=_SPOT_COMMITTED_WORK,
        rulings=_SPOT_RULINGS,
        retired_reviews=_SPOT_RETIRED_REVIEWS,
    ),
    'revalidation': _dispatch('build_revalidation_prompt', _SPOT_TASK, _SPOT_PLAN, ['a.py']),
    'revalidation_with_optional_sections': _dispatch(
        'build_revalidation_prompt',
        _SPOT_TASK,
        _SPOT_PLAN,
        ['a.py'],
        text_drift=TaskTextDrift.AMENDED,
        integrity_refusal=_SPOT_INTEGRITY_REFUSAL,
        rulings=_SPOT_RULINGS,
        retired_reviews=_SPOT_RETIRED_REVIEWS,
    ),
    'completion': _dispatch('build_plan_completion_prompt', _SPOT_TASK, _SPOT_PLAN),
    'completion_with_optional_sections': _dispatch(
        'build_plan_completion_prompt', _SPOT_TASK, _SPOT_PLAN, rulings=_SPOT_RULINGS,
    ),
    'tightening': _dispatch('build_plan_tightening_prompt', _SPOT_TASK, _SPOT_PLAN, ['a.py']),
    'schema_repair': _dispatch('build_plan_schema_repair_prompt', _SPOT_TASK),
    'replan': _dispatch('build_replan_prompt', _SPOT_TASK, 'REVIEW-FEEDBACK'),
    'replan_with_optional_sections': _dispatch(
        'build_replan_prompt', _SPOT_TASK, 'REVIEW-FEEDBACK', rulings=_SPOT_RULINGS,
    ),
})


async def _brief(
    briefing: BriefingAssembler, dispatch: _ArchitectDispatch, options: Mapping[str, object],
) -> str:
    """Render *dispatch* with *options*, on a fresh copy of the fixtures every case shares."""
    args, kwargs = copy.deepcopy((dispatch.args, dict(options)))
    return await getattr(briefing, dispatch.builder)(*args, context='', **kwargs)


async def _dispatch_briefing(briefing: BriefingAssembler, name: str) -> str:
    dispatch = _ARCHITECT_DISPATCHES[name]
    return await _brief(briefing, dispatch, dispatch.options)


def _optional_sections(builder: str) -> set[str]:
    """The keyword-only parameters of *builder*; each feeds one optional briefing section."""
    parameters = inspect.signature(getattr(BriefingAssembler, builder)).parameters.values()
    return {param.name for param in parameters if param.kind is inspect.Parameter.KEYWORD_ONLY}


class TestArchitectPlanToolsRecipeSingleHome:
    """Every ARCHITECT dispatch spells each plan-tools recipe signature once.

    The dispatched prompt is ``ARCHITECT.system_prompt`` plus the briefing. Tests
    tie ``_ARCHITECT_DISPATCHES`` to workflow.py's ARCHITECT builders and to every
    optional section each builder's signature declares.
    """

    def test_recipe_is_a_nonempty_section_of_its_own(self) -> None:
        assert_nonempty(
            _RECIPE_CONTRACT.constant_name,
            ARCHITECT_PLAN_TOOLS_RECIPE,
            remedy='Restore the plan-tools call recipe in '
            'roles.py::ARCHITECT_PLAN_TOOLS_RECIPE.',
        )
        assert_opens_own_section(
            _RECIPE_CONTRACT.constant_name,
            ARCHITECT_PLAN_TOOLS_RECIPE,
            remedy='Keep roles.py::ARCHITECT_PLAN_TOOLS_RECIPE opening with its '
            '`## Your Output` heading.',
        )

    def test_recipe_reaches_no_role_but_the_architect(self) -> None:
        _RECIPE_CONTRACT.assert_no_other_role_carries(
            remedy="It is the architect's recipe and belongs in "
            'ARCHITECT.system_prompt alone; SIMPLE_TASK plans by '
            'roles.py::SIMPLE_TASK_WORKFLOW.',
        )

    def test_architect_carries_the_recipe_exactly_once(self) -> None:
        _RECIPE_CONTRACT.assert_spliced_exactly_once(
            remedy='ARCHITECT.system_prompt is the one home of '
            'roles.py::ARCHITECT_PLAN_TOOLS_RECIPE and splices it once.',
        )

    def test_system_prompt_spells_recipe_signatures_only_inside_the_recipe(self) -> None:
        in_recipe = _recipe_signature_spellings(ARCHITECT_PLAN_TOOLS_RECIPE)
        assert in_recipe, (
            '_recipe_signature_spellings finds no signature in '
            'ARCHITECT_PLAN_TOOLS_RECIPE, so it cannot see a re-spelled one '
            'anywhere else either.'
        )
        assert len(set(in_recipe)) == len(in_recipe), (
            f'ARCHITECT_PLAN_TOOLS_RECIPE spells a signature twice: {in_recipe}.'
        )
        outside = _recipe_signature_spellings(
            ARCHITECT.system_prompt.replace(ARCHITECT_PLAN_TOOLS_RECIPE, ''),
        )
        assert not outside, (
            f'ARCHITECT.system_prompt re-spells {outside} outside '
            'ARCHITECT_PLAN_TOOLS_RECIPE. Name the tool without its signature, '
            '`tool(...)`, or point at the recipe.'
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize('dispatch', sorted(_ARCHITECT_DISPATCHES))
    async def test_dispatch_briefing_spells_no_recipe_signature(
        self, briefing: BriefingAssembler, dispatch: str,
    ) -> None:
        respelled = _recipe_signature_spellings(await _dispatch_briefing(briefing, dispatch))
        assert not respelled, (
            f'The {dispatch!r} architect briefing re-spells these plan-tools '
            f'recipe signatures: {respelled}. The recipe has one home, '
            'roles.py::ARCHITECT_PLAN_TOOLS_RECIPE, which ARCHITECT.system_prompt '
            'carries into every architect dispatch. A briefing that re-spells a '
            'recipe signature, under any parameter names, shows the architect a '
            'second recipe that can drift from the first (task 5532). Name the '
            'tool without its signature, `tool(...)`, or point at the recipe.'
        )

    def test_dispatches_pass_every_optional_section_of_each_tabled_builder(self) -> None:
        passed: dict[str, set[str]] = {}
        for dispatch in _ARCHITECT_DISPATCHES.values():
            passed.setdefault(dispatch.builder, set()).update(dispatch.options)
        assert any(_optional_sections(builder) for builder in passed), (
            '_optional_sections finds no keyword-only parameter on any builder '
            '_ARCHITECT_DISPATCHES names, so this check sees no optional '
            'section; it no longer matches how the architect builders declare them.'
        )
        unpassed = {
            builder: sorted(missing)
            for builder, options in sorted(passed.items())
            if (missing := _optional_sections(builder) - options)
        }
        assert not unpassed, (
            f'No _ARCHITECT_DISPATCHES entry passes these optional sections of '
            f'the builders it names: {unpassed}. Add or extend a '
            "'<dispatch>_with_optional_sections' entry that passes them, with "
            'fixture text free of recipe signatures: a section no entry renders '
            'is a section the recipe-signature check never scans.'
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'dispatch', sorted(name for name, entry in _ARCHITECT_DISPATCHES.items() if entry.options),
    )
    async def test_each_optional_section_a_dispatch_passes_renders_into_its_briefing(
        self, briefing: BriefingAssembler, dispatch: str,
    ) -> None:
        entry = _ARCHITECT_DISPATCHES[dispatch]
        rendered = await _brief(briefing, entry, entry.options)
        inert = [
            option
            for option in sorted(entry.options)
            if rendered == await _brief(
                briefing,
                entry,
                {key: value for key, value in entry.options.items() if key != option},
            )
        ]
        assert not inert, (
            f'Dropping {inert} from the {dispatch!r} dispatch leaves its briefing '
            'byte-identical: those options reach no section, so the '
            'recipe-signature check is not scanning the sections it claims to.'
        )

    def test_dispatch_table_covers_every_workflow_architect_builder(self) -> None:
        dispatched = architect_briefing_builders()
        assert dispatched, (
            'No self.briefing.build_* builder reaches an ARCHITECT dispatch in '
            'workflow.py, so this check sees nothing; the scan in '
            '_workflow_architect_dispatch.py no longer matches how workflow.py '
            'dispatches the architect.'
        )
        tabled = {dispatch.builder for dispatch in _ARCHITECT_DISPATCHES.values()}
        assert tabled == dispatched, (
            f'workflow.py dispatches these ARCHITECT briefing builders that '
            f'_ARCHITECT_DISPATCHES lacks: {sorted(dispatched - tabled)}; and '
            f'the table holds these that workflow.py no longer dispatches to '
            f'ARCHITECT: {sorted(tabled - dispatched)}. If workflow.py did not '
            'change which builders brief the architect, it changed how it '
            'dispatches them: extend the scan in _workflow_architect_dispatch.py.'
        )
