"""Lazy, atomic repair of envelope-markup corruption in a live plan.json.

Task 3692 (PRD ``plans/toolcall-markup-containment-prd.md``, task EPSILON,
boundary row B12). The subject under test is ``orchestrator.mcp.plan_tools``'s
read-time repair surface; the detection/repair mechanism itself is owned by
``shared.toolcall_markup`` (task 3688) and is NOT re-implemented here.

## Sentinel-literal hazard — every fixture is BUILT, never written verbatim

This module describes MCP tool-call envelope markup, so it is exactly the file
that must not contain any of that markup literally. The rationale is the one
recorded at ``shared/src/shared/toolcall_markup.py`` lines 52-62: an agent
editing a file that contains a raw envelope literal has to emit that literal
INSIDE its own tool-call argument, which reproduces the very over-consumption
defect under test — the Write/Edit argument terminates early, truncating this
file and silently dropping that call's sibling arguments.

So every specimen below is assembled at import time from :func:`_close`,
:func:`_open_param` and :data:`_INVOKE_CLOSER`, which build their angle bracket
from ``chr(60)``. The result is byte-identical at runtime and never appears
verbatim in the file text. :func:`_assert_no_raw_sentinels` enforces that on the
module's OWN BYTES at import, so a future editor cannot quietly reintroduce one
(it is a check on this file's source text, not on any docstring's wording).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from shared.toolcall_markup import ENVELOPE_LITERALS

from orchestrator.artifacts import TaskArtifacts

# ---------------------------------------------------------------------------
# Sentinel BUILDERS — the only way markup enters this module.
# ---------------------------------------------------------------------------

#: The opening angle bracket, spelled so it never appears verbatim in the file.
_LT = chr(60)


def _close(name: str) -> str:
    """Build the closing tag for *name* (the mis-close shape the harness emits)."""
    return _LT + '/' + name + '>'


def _open_param(name: str) -> str:
    """Build the canonical opening tag for parameter *name*."""
    return _LT + 'parameter name="' + name + '">'


#: The bare invoke closer — the terminator that trails a last-parameter leak.
_INVOKE_CLOSER = _close('invoke')


def _assert_no_raw_sentinels() -> None:
    """Fail at IMPORT if this file's own bytes carry a raw envelope literal.

    Checked against ``shared.toolcall_markup.ENVELOPE_LITERALS`` (the single
    owner of the literal set, INV-5) plus the two structural prefixes every
    built specimen uses, so a builder output spelled out by hand is caught even
    when it is not itself one of the enumerated literals.
    """
    source = Path(__file__).read_text(encoding='utf-8')
    forbidden = (*ENVELOPE_LITERALS, _LT + '/', _LT + 'parameter ')
    for sequence in forbidden:
        if sequence in source:
            raise AssertionError(
                f'{Path(__file__).name} contains a RAW envelope sentinel '
                f'({sequence!r}). Build it from _close()/_open_param() instead '
                '— a verbatim literal here corrupts the tool call that writes '
                'this file. See the module docstring.'
            )


_assert_no_raw_sentinels()


# ---------------------------------------------------------------------------
# The four REAL specimen shapes, measured on the 28 corrupted live plans.
# ---------------------------------------------------------------------------

#: Clean prose used as the intended value of whichever field a specimen poisons.
#: Deliberately free of any markup so ``detect()`` on a repaired prefix is None.
_RATIONALE_PROSE = (
    'Reusing the shared detector keeps one enumeration of the literals, so the '
    'write-time guard and the read-time repair can never drift apart.'
)
_HOW_PROSE = (
    'Imported directly; the helper does the grammar work and owns every '
    'accept/refuse decision, so this module adds no literal enumeration.'
)
_DECISION_PROSE = (
    'Repair the plan lazily on read rather than sweeping the fleet, because a '
    'sweep would have to quiesce every running task first.'
)

#: TRAILING RESIDUE on ``design_decisions[].rationale`` — the DOMINANT live
#: shape (97 of 118 corrupted strings): the parameter was last in the call, so
#: nothing was absorbed and only the mis-close plus the invoke closer trail it.
TRAILING_RATIONALE = _RATIONALE_PROSE + _close('rationale') + '\n' + _INVOKE_CLOSER + '\n'

#: The same trailing-residue shape on ``reuse[].how`` (27 of the 97).
TRAILING_HOW = _HOW_PROSE + _close('how') + '\n' + _INVOKE_CLOSER + '\n'

#: ABSORBED SIBLING on ``design_decisions[].decision``: the parser mis-closed
#: ``decision`` and then swallowed the whole ``rationale`` parameter into it, so
#: the rationale a later reader sees is another field's text (PRD section 2.4).
#: The final opener is UNTERMINATED — its closer was consumed as the terminator.
ABSORBED_RATIONALE = (
    _DECISION_PROSE + _close('decision') + '\n' + _open_param('rationale') + _RATIONALE_PROSE
)

#: PROSE FALSE POSITIVE, measured live in worktree 2939 — a plan ABOUT this leak,
#: whose authored text legitimately QUOTES the sentinels in ordinary sentences.
#: ``detect()`` fires, ``repair()`` correctly declines, and a trailing-only
#: sanitize contract would mutilate it. This is the specimen that makes the
#: repair-or-leave-byte-identical contract load-bearing rather than stylistic.
PROSE_QUOTED = (
    'The harness closes the argument with ' + _close('description') + ' or with '
    + _close('parameter') + ', and then re-opens with ' + _open_param('x')
    + ' before the next value, which is how the sibling arguments get lost.'
)


# ---------------------------------------------------------------------------
# Fixtures and plan factory.
# ---------------------------------------------------------------------------


@pytest.fixture()
def plan_artifacts(tmp_path):
    """TaskArtifacts over a temp worktree — mirrors ``test_plan_tools_server``."""
    a = TaskArtifacts(tmp_path)
    a.init('test-1', 'Test task', 'A test')
    return a


def corrupt_plan(**overrides) -> dict:
    """Return a complete, VALID plan dict whose fields can then be poisoned.

    Every call builds a fresh, independent document (no shared mutable state),
    so a test may poison ``plan['design_decisions'][0]['rationale']`` in place.
    Keyword *overrides* replace whole top-level keys, which is how a test swaps
    in its own collection (e.g. four decisions instead of the default two).

    The default document is entirely CLEAN: nothing here trips ``detect()``, so
    any fact a test observes came from the field it poisoned.
    """
    plan: dict = {
        'task_id': 'test-1',
        'title': 'A test plan',
        'analysis': 'Clean analysis prose describing the approach.',
        'files': ['orchestrator/src/orchestrator/mcp/plan_tools.py'],
        'prerequisites': [
            {
                'id': 'pre-1',
                'description': 'Clean prerequisite prose.',
                'status': 'pending',
                'commit': None,
                'tests': [],
            },
        ],
        'steps': [
            {
                'id': 'step-1',
                'type': 'test',
                'description': 'Clean step prose for the first step.',
                'status': 'pending',
                'commit': None,
            },
            {
                'id': 'step-2',
                'type': 'impl',
                'description': 'Clean step prose for the second step.',
                'status': 'pending',
                'commit': None,
            },
        ],
        'design_decisions': [
            {'decision': _DECISION_PROSE, 'rationale': 'Clean rationale prose.'},
            {'decision': 'A second clean decision.', 'rationale': 'A second clean rationale.'},
        ],
        'reuse': [
            {
                'what': 'The shared detector',
                'where': 'shared/src/shared/toolcall_markup.py',
                'how': 'Clean reuse prose.',
            },
            {
                'what': 'The plan artifact reader',
                'where': 'orchestrator/src/orchestrator/artifacts.py',
                'how': 'A second clean reuse prose.',
            },
        ],
    }
    plan.update(overrides)
    return plan
