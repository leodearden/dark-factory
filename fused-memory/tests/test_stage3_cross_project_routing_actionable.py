"""Contract test for the Stage 3 prompt's ``cross_project_routing`` example (task 4347).

Stage 3's "Cross-Project Routing Guard" section tells the agent to file a
``cross_project_routing`` finding when ``get_task`` returns a task stamped with
a different ``project_id`` than the project under reconciliation — i.e. the
harness was pointed at a wrong ``project_root`` and the data in hand belongs to
someone else.  That is a data-integrity signal, filed at ``severity='serious'``.

But the example omitted ``actionable``, and ``ReconReportState.add_finding``
resolves an omitted ``actionable`` to the COMPUTED default
``not (task_id is None or category.startswith('cross_project'))``.  The Stage 3
example trips both triggers (no ``task_id``, ``cross_project`` prefix), so it
resolved to ``False`` — which is the necessary condition for the task-1654
read-time suppression in ``get_assembled_report``.  A serious wrong-project_root
finding could therefore be dropped from ``flagged_items`` whenever its citations
traced entirely to Stage 1.  Stage 3 is never ``memory_consolidator``, the one
stage suppression is skipped for, so the exposure was live.

WHY THIS FILE PINS AN IDENTIFIER AND PINS NO SENTENCE
-----------------------------------------------------
``test_duplicate_finding_salvage_guidance.py`` records at length why a prose
substring pin inside a prompt literal is a documentation meta-test rather than a
behavioural contract: it is wrong in both directions — a faithful reword that
says the same thing differently goes red on a correct edit, while a garbled
paragraph that happens to retain the tokens goes green.

That reasoning governs SENTENCES.  It does not govern ``actionable=True``, which
is not wording but the INTERFACE an LLM writes to — the literal kwarg on the
``add_finding`` call the agent emits, and the single bit that decides whether the
finding survives ``_traces_exclusively_to_stage1``.  A prompt that spells it
differently does not read differently; it WRITES differently.  So the identifier
is a behavioural contract while the surrounding justification is not, and
accordingly **no assertion in this file pins any sentence**.  Step-2's prose may
be reworded freely.

The example's layout and its ``severity`` are deliberately not pinned: neither
decides whether the finding survives suppression.

WHY LAYER (b) IS GREEN FROM THE FIRST COMMIT, BY DESIGN
-------------------------------------------------------
Layer (a) alone would be a test that can agree with nothing but itself — it
greps a string literal for a token it also chose.  ``test_finding_provenance_
prompt_guidance.py`` guards against exactly that by cross-checking its pinned
keys against a real consumer, and its docstring names the failure it prevents: a
first shipped clause that was "a silent no-op" because it asserted one
vocabulary against nothing.

Layer (b) is that cross-check.  It round-trips a live ``ReconReportState`` to
demonstrate the mechanism the kwarg exists to defeat: the same Stage-3 finding
VANISHES from ``flagged_items`` with ``actionable`` omitted and SURVIVES with
``actionable=True``.  It characterizes behaviour that already exists, so it
passes from commit one — that asymmetry is deliberate and is not a missing RED.
Its job is to keep layer (a) load-bearing: if task-1654 suppression is ever
narrowed so Stage 3 is exempt, layer (b) goes red and tells the next reader the
prompt pin has lost its reason, instead of the pin quietly outliving its purpose.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from fused_memory.models.reconciliation import StageId
from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT
from fused_memory.server.recon_report import ReconReportState

CROSS_PROJECT_ROUTING = "category='cross_project_routing'"

_FENCE = '```'


def _routing_example_blocks() -> list[str]:
    """The fenced blocks of Stage 3's prompt that file a routing finding.

    Splitting on the fence marker alternates prose and fenced segments, so the
    odd-indexed ones are the fenced blocks; prose that mentions the category is
    never mistaken for an example.  Nothing here parses the call itself, so the
    prompt's layout is free to change.  ``STAGE3_SYSTEM_PROMPT`` is an
    f-string, so this reads the RENDERED prompt the agent sees.
    """
    fenced_blocks = STAGE3_SYSTEM_PROMPT.split(_FENCE)[1::2]
    return [block for block in fenced_blocks if CROSS_PROJECT_ROUTING in block]


# ---------------------------------------------------------------------------
# Layer (a) — interface pin on the Stage 3 prompt
# ---------------------------------------------------------------------------


class TestStage3CrossProjectRoutingExampleIsExplicitlyActionable:
    """The Stage 3 template names ``actionable`` explicitly, and names it True."""

    def test_prompt_carries_exactly_one_routing_example(self):
        """Non-emptiness guard so the assertions below cannot pass vacuously.

        Mirrors the ``.count(...) == 1`` reasoning in
        test_duplicate_finding_salvage_guidance.py: an emptied prompt constant
        or a renamed category would make every assertion below trivially true
        against nothing.  A second example is also a signal: the pins below
        were written for the wrong-project_root one.
        """
        blocks = _routing_example_blocks()
        assert len(blocks) == 1, (
            f'Expected exactly one fenced block in STAGE3_SYSTEM_PROMPT '
            f'carrying {CROSS_PROJECT_ROUTING}, found {len(blocks)}. Either the '
            'Cross-Project Routing Guard section of '
            'fused_memory/reconciliation/prompts/stage3.py lost its worked '
            'example, or a second example was added and the actionable pins '
            'below need to be re-aimed at it.'
        )

    def test_example_passes_actionable_true(self):
        """The kwarg that keeps a wrong-project_root finding out of suppression."""
        for block in _routing_example_blocks():
            assert 'actionable=True' in block, (
                'Stage 3 cross_project_routing example must pass actionable=True '
                'explicitly. Omitting it inherits the computed default '
                "not (task_id is None or category.startswith('cross_project')) — "
                'which is False here on BOTH triggers — making a serious '
                'wrong-project_root finding eligible for task-1654 read-time '
                'suppression in get_assembled_report whenever its citations trace '
                f'entirely to Stage 1. Offending block: {block}'
            )

    def test_example_does_not_pass_actionable_false(self):
        """Mirror-image regression guard.

        Stage 2 files cross_project_routing with actionable=False because those
        ARE informational routing notes.  A future editor "aligning" Stage 3 with
        that spelling would reintroduce exactly the defect this file closes.
        """
        for block in _routing_example_blocks():
            assert 'actionable=False' not in block, (
                'Stage 3 must NOT copy Stage 2 / autopilot_video.py\'s '
                'actionable=False. Same category, opposite actionability: Stage 2 '
                'files an informational "this belongs to another project" note, '
                'Stage 3 reports that the harness read another project\'s data. '
                f'Offending block: {block}'
            )


# ---------------------------------------------------------------------------
# Layer (b) — behavioural cross-check via a live ReconReportState round-trip
# ---------------------------------------------------------------------------


class TestStage3RoutingFindingSurvivesSuppressionOnlyWhenActionable:
    """Round-trip proof of the mechanism layer (a)'s kwarg defeats.

    Construction copied from
    tests/server/test_recon_report.py::TestGetAssembledReportNonActionableEchoSuppression
    so this differs from a proven-correct template only in the dimensions this
    task is about: Stage 3 (``integrity_check``) instead of Stage 2, and
    ``category='cross_project_routing'`` / ``severity='serious'`` instead of a
    generic echo.

    It uses ``StageId`` members rather than retyped literals so the test cannot
    drift from the enum ``get_assembled_report`` compares against.  Stage
    3's id is ``integrity_check`` — never ``memory_consolidator``, the sole stage
    suppression is skipped for — which is precisely why suppression is live here.
    """

    def _build_state(self) -> ReconReportState:
        task_interceptor = AsyncMock()
        task_interceptor.get_task = AsyncMock(return_value={
            'title': 'Task from reify project',
            'data': {},
        })
        state = ReconReportState(
            ttl_seconds=3600,
            clock=lambda: 0.0,
            task_interceptor=task_interceptor,
        )
        # cite_task records nothing for a project absent from known_projects.
        state.known_projects['reify'] = '/tmp/reify'
        return state

    async def _file_stage1_citation(self, state: ReconReportState, run_id: str) -> None:
        """Stage 1 files a cross-project finding citing reify/3803."""
        state.start_report(run_id, StageId.memory_consolidator, 'dark_factory')
        r = state.add_finding(
            run_id=run_id,
            severity='low',
            category='cross_project',
            description='Stage 1 cross-project finding about reify/3803',
            suggested_action='Check reify project',
            actionable=False,
            task_id=None,
            flag_type='cross_project',
        )
        assert 'error' not in r, f'Stage 1 add_finding failed: {r}'
        await state.cite_task(run_id=run_id, finding_id=r['finding_id'],
                              project_id='reify', task_id='3803')

    @pytest.mark.parametrize('actionable_kwargs, expect_present', [
        pytest.param({}, False, id='actionable-omitted'),
        pytest.param({'actionable': True}, True, id='actionable-true'),
    ])
    @pytest.mark.asyncio
    async def test_routing_finding_survives_only_when_explicitly_actionable(
        self, actionable_kwargs, expect_present,
    ):
        """One finding, one difference: ``actionable``, and it decides survival.

        Parametrized rather than written twice so the round-trip setup — the
        Stage 1 citation, the Stage 3 arguments, the shared cited task — cannot
        drift between the two halves of the contrast it exists to draw.

        The omitted case is the vanishing bug the Stage 3 prompt fix prevents: a
        ``severity='serious'`` wrong-project_root report, filed with the
        category and severity the template prescribes and carrying the
        ``cite_task`` the template requires, silently dropped from
        ``flagged_items`` at read time purely because ``actionable`` was left to
        its computed default.  (``flag_type='wrong_project_root'`` is the one
        argument here the template does not prescribe — see the comment below.)
        """
        state = self._build_state()
        run_id = f'r4347-{"explicit" if expect_present else "omitted"}'
        await self._file_stage1_citation(state, run_id)

        state.start_report(run_id, StageId.integrity_check, 'dark_factory')
        r = state.add_finding(
            run_id=run_id,
            severity='serious',
            category='cross_project_routing',
            description='get_task returned task from project reify, expected dark_factory',
            suggested_action='Re-run with the correct project_root',
            task_id=None,
            # flag_type differs from Stage 1's 'cross_project' to dodge the
            # in-run (task_id, flag_type) sig dedup for null-task_id findings.
            flag_type='wrong_project_root',
            # The ONLY dimension under test: omitted (pre-fix) vs explicit True.
            **actionable_kwargs,
        )
        assert 'error' not in r, f'Stage 3 add_finding failed: {r}'
        stage3_finding_id = r['finding_id']

        # The template instructs this citation; it is also what makes the
        # finding's identity set a subset of Stage 1's, i.e. suppressible.
        await state.cite_task(run_id=run_id, finding_id=stage3_finding_id,
                              project_id='reify', task_id='3803')

        assembled = state.get_assembled_report(run_id, StageId.integrity_check)
        assert assembled is not None, 'get_assembled_report returned None'
        rows = {f['finding_id']: f for f in assembled['flagged_items']}

        if not expect_present:
            assert stage3_finding_id not in rows, (
                'Expected the omitted-actionable Stage 3 routing finding to be '
                'suppressed — that suppression is the whole reason the prompt '
                f'must pass actionable=True. flagged_items: {list(rows)}'
            )
            return

        assert stage3_finding_id in rows, (
            'An explicit actionable=True must keep the Stage 3 routing finding '
            f'in flagged_items. flagged_items: {list(rows)}'
        )
        # Confirms the round-trip really exercised the cross_project_routing path:
        # a non-empty cited_tasks is what keeps _apply_cross_project_routing_guard
        # from downgrading the category to 'other'/'cross_project_info'.
        assert rows[stage3_finding_id]['category'] == 'cross_project_routing', (
            'Surviving row was downgraded, so this test was not exercising the '
            f"cross_project_routing path: {rows[stage3_finding_id]['category']!r}"
        )
