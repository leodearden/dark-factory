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


def _extract_add_finding_calls(prompt: str) -> list[str]:
    """Return the whitespace-collapsed argument text of every ``add_finding(...)``.

    The Stage 3 call spans three physical lines inside a fenced block, so this
    walks a paren-depth counter from each ``add_finding(`` to its matching close
    paren rather than matching a single line.  An unbalanced occurrence (one that
    never closes) is skipped rather than silently truncated.

    ``STAGE3_SYSTEM_PROMPT`` is an f-string, so its ``{{``/``}}`` escaping is
    already collapsed at import — this reads the RENDERED prompt the agent sees.
    """
    marker = 'add_finding('
    calls: list[str] = []
    idx = prompt.find(marker)
    while idx != -1:
        start = idx + len(marker)
        depth = 1
        pos = start
        while pos < len(prompt) and depth > 0:
            ch = prompt[pos]
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            pos += 1
        if depth == 0:
            calls.append(' '.join(prompt[start:pos - 1].split()))
        idx = prompt.find(marker, start)
    return calls


def _cross_project_routing_calls() -> list[str]:
    """The subset of Stage 3's ``add_finding`` calls filing a routing finding."""
    return [c for c in _extract_add_finding_calls(STAGE3_SYSTEM_PROMPT)
            if CROSS_PROJECT_ROUTING in c]


# ---------------------------------------------------------------------------
# Layer (a) — interface pin on the Stage 3 prompt
# ---------------------------------------------------------------------------


class TestStage3CrossProjectRoutingExampleIsExplicitlyActionable:
    """The Stage 3 template names ``actionable`` explicitly, and names it True."""

    def test_extraction_finds_at_least_one_call(self):
        """Non-emptiness guard so the assertions below cannot pass vacuously.

        Mirrors the ``.count(...) == 1`` reasoning in
        test_duplicate_finding_salvage_guidance.py: a broken extractor (or a
        prompt constant accidentally emptied) would make every "for each match"
        assertion trivially true against nothing.
        """
        calls = _cross_project_routing_calls()
        assert calls, (
            "No add_finding(...) call carrying "
            f"{CROSS_PROJECT_ROUTING} was extracted from STAGE3_SYSTEM_PROMPT. "
            'Either the Cross-Project Routing Guard section of '
            'fused_memory/reconciliation/prompts/stage3.py lost its worked '
            'example, or _extract_add_finding_calls no longer matches its shape '
            '— every other assertion in this class is vacuous until this passes.'
        )

    def test_every_call_passes_actionable_true(self):
        """The kwarg that keeps a wrong-project_root finding out of suppression."""
        for call in _cross_project_routing_calls():
            assert 'actionable=True' in call, (
                'Stage 3 cross_project_routing example must pass actionable=True '
                'explicitly. Omitting it inherits the computed default '
                "not (task_id is None or category.startswith('cross_project')) — "
                'which is False here on BOTH triggers — making a serious '
                'wrong-project_root finding eligible for task-1654 read-time '
                'suppression in get_assembled_report whenever its citations trace '
                f'entirely to Stage 1. Offending call: add_finding({call})'
            )

    def test_no_call_passes_actionable_false(self):
        """Mirror-image regression guard.

        Stage 2 files cross_project_routing with actionable=False because those
        ARE informational routing notes.  A future editor "aligning" Stage 3 with
        that spelling would reintroduce exactly the defect this file closes.
        """
        for call in _cross_project_routing_calls():
            assert 'actionable=False' not in call, (
                'Stage 3 must NOT copy Stage 2 / autopilot_video.py\'s '
                'actionable=False. Same category, opposite actionability: Stage 2 '
                'files an informational "this belongs to another project" note, '
                'Stage 3 reports that the harness read another project\'s data. '
                f'Offending call: add_finding({call})'
            )

    def test_every_call_is_still_the_serious_data_integrity_example(self):
        """Pins that the matched set is the wrong-project_root example.

        If a future cross_project_routing example is added that is legitimately
        informational, this goes red rather than letting the actionable=True pin
        above quietly attach itself to the wrong call.
        """
        for call in _cross_project_routing_calls():
            assert "severity='serious'" in call, (
                'The Stage 3 cross_project_routing example is a data-integrity '
                "signal filed at severity='serious'. A call matching "
                f'{CROSS_PROJECT_ROUTING} at another severity is a different '
                'example, and the actionable=True pin above may not apply to it. '
                f'Offending call: add_finding({call})'
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

    Both tests use ``StageId`` members rather than retyped literals so the test
    cannot drift from the enum ``get_assembled_report`` compares against.  Stage
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

    @pytest.mark.asyncio
    async def test_omitting_actionable_lets_the_routing_finding_vanish(self):
        """With ``actionable`` omitted the finding is ABSENT from flagged_items.

        This is the vanishing bug the Stage 3 prompt fix prevents: a
        severity='serious' wrong-project_root report, filed exactly as the
        template instructed, silently dropped at read time.
        """
        state = self._build_state()
        run_id = 'r4347-omitted'
        await self._file_stage1_citation(state, run_id)

        state.start_report(run_id, StageId.integrity_check, 'dark_factory')
        r = state.add_finding(
            run_id=run_id,
            severity='serious',
            category='cross_project_routing',
            description='get_task returned task from project reify, expected dark_factory',
            suggested_action='Re-run with the correct project_root',
            # actionable deliberately OMITTED — the pre-fix Stage 3 template.
            task_id=None,
            # flag_type differs from Stage 1's 'cross_project' to dodge the
            # in-run (task_id, flag_type) sig dedup for null-task_id findings.
            flag_type='wrong_project_root',
        )
        assert 'error' not in r, f'Stage 3 add_finding failed: {r}'
        stage3_finding_id = r['finding_id']

        await state.cite_task(run_id=run_id, finding_id=stage3_finding_id,
                              project_id='reify', task_id='3803')

        assembled = state.get_assembled_report(run_id, StageId.integrity_check)
        assert assembled is not None, 'get_assembled_report returned None'
        finding_ids = [f['finding_id'] for f in assembled['flagged_items']]
        assert stage3_finding_id not in finding_ids, (
            'Expected the omitted-actionable Stage 3 routing finding to be '
            'suppressed — that suppression is the whole reason the prompt must '
            f'pass actionable=True. flagged_items finding_ids: {finding_ids}'
        )

    @pytest.mark.asyncio
    async def test_explicit_actionable_true_keeps_the_routing_finding(self):
        """With ``actionable=True`` the identical finding is PRESENT."""
        state = self._build_state()
        run_id = 'r4347-explicit'
        await self._file_stage1_citation(state, run_id)

        state.start_report(run_id, StageId.integrity_check, 'dark_factory')
        r = state.add_finding(
            run_id=run_id,
            severity='serious',
            category='cross_project_routing',
            description='get_task returned task from project reify, expected dark_factory',
            suggested_action='Re-run with the correct project_root',
            actionable=True,   # the one difference from the test above
            task_id=None,
            flag_type='wrong_project_root',
        )
        assert 'error' not in r, f'Stage 3 add_finding failed: {r}'
        stage3_finding_id = r['finding_id']

        await state.cite_task(run_id=run_id, finding_id=stage3_finding_id,
                              project_id='reify', task_id='3803')

        assembled = state.get_assembled_report(run_id, StageId.integrity_check)
        assert assembled is not None, 'get_assembled_report returned None'
        rows = {f['finding_id']: f for f in assembled['flagged_items']}
        assert stage3_finding_id in rows, (
            'An explicit actionable=True must keep the Stage 3 routing finding in '
            f'flagged_items. flagged_items finding_ids: {list(rows)}'
        )
        # Confirms the round-trip really exercised the cross_project_routing path:
        # a non-empty cited_tasks is what keeps _apply_cross_project_routing_guard
        # from downgrading the category to 'other'/'cross_project_info'.
        assert rows[stage3_finding_id]['category'] == 'cross_project_routing', (
            'Surviving row was downgraded, so this test was not exercising the '
            f"cross_project_routing path: {rows[stage3_finding_id]['category']!r}"
        )
