"""The rejected-call counter a plan-tools refusal stamps onto ``plan.json``.

Task 4597 (esc-4528-1). Under ``RepairPolicy.REJECT_WITH_REPAIR`` the guard
RAISES before the tool body runs, so a refused ``add_design_decision`` writes
nothing to ``plan.json``. The refusal reaches a ``ToolError``, one journal line
and — on the unrepairable path — a residue escalation. The plan itself records
NOTHING, so ``design_decisions: []`` is genuinely ambiguous between "the
architect never called" and "the architect called six times and was refused six
times". In task 4528 a reviewer could only tell the two apart because the
architect happened to hand-file an info note.

The subject under test is :mod:`orchestrator.mcp.plan_markup_stamp`: the second
consumer of the middleware's fact channel, which folds a guard-owned
``_markup_rejections`` bookkeeping block into the plan document so the gap is
self-describing to any later reader. The END-TO-END wiring (that plan-tools'
registered guard actually reaches it, and that the block survives a
``create_plan`` overwrite) is pinned in ``test_plan_tools_markup_guard.py``
against the real server. Nothing here re-derives detection, repair or policy,
which are owned by ``shared.toolcall_markup`` / ``shared.mcp_markup_middleware``
and pinned by their own tests.

## Async marker

Every async test carries an explicit ``@pytest.mark.asyncio``. orchestrator does
NOT set ``asyncio_mode = auto`` (shared does — do not copy that half of the
idiom from its middleware tests).

## Sentinel-literal hazard — every specimen is BUILT, never written verbatim

This module describes MCP tool-call envelope markup, so it is exactly the file
that must not contain any of it literally. The rationale is the one recorded at
``shared/src/shared/toolcall_markup.py`` lines 52-62 and repeated in
``test_plan_tools_markup_guard.py`` and ``test_markup_journal.py``: an agent
editing a file that holds a raw envelope literal has to emit that literal INSIDE
its own tool-call argument, which reproduces the very over-consumption defect
under test.

The discipline is LOAD-BEARING here beyond convention. This module's central
negative control proves the stamp copies no envelope literal out of a fact
record whose ``pattern`` and ``misclose`` ARE that literal — so the module has
to hold such specimens, which makes it precisely the file that must not contain
one verbatim.

So every specimen is assembled from :func:`_close` / :func:`_open_param`, which
build their angle bracket from ``chr(60)``, and :func:`_assert_no_raw_sentinels`
enforces that on this module's OWN BYTES at import — checked against
``shared.toolcall_markup.ENVELOPE_LITERALS``, the single owner of the literal
set (INV-5), plus the two structural prefixes.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from shared.mcp_markup_middleware import FACT_MARKUP_DETECTED
from shared.toolcall_markup import ENVELOPE_LITERALS, detect

from orchestrator.mcp import plan_markup_stamp

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


#: ``shared.toolcall_markup.ENVELOPE_LITERALS`` (the single owner of the literal
#: set, INV-5) plus the two structural prefixes every built specimen uses, so a
#: builder output spelled out by hand is caught even when it is not itself one
#: of the enumerated literals. Applied to this module's OWN BYTES at import, and
#: to the STAMPED BLOCK the sink writes — the same predicate, two artifacts.
_FORBIDDEN_SEQUENCES = (*ENVELOPE_LITERALS, _LT + '/', _LT + 'parameter ')


def _assert_no_raw_sentinels() -> None:
    """Fail at IMPORT if this file's own bytes carry a raw envelope literal."""
    source = Path(__file__).read_text(encoding='utf-8')
    for sequence in _FORBIDDEN_SEQUENCES:
        if sequence in source:
            raise AssertionError(
                f'{Path(__file__).name} contains a RAW envelope sentinel '
                f'({sequence!r}). Build it from _close()/_open_param() instead '
                '— a verbatim literal here corrupts the tool call that writes '
                'this file. See the module docstring.'
            )


_assert_no_raw_sentinels()


# ---------------------------------------------------------------------------
# The fact record — the shape ``MarkupGuardMiddleware._emit_fact`` builds.
# ---------------------------------------------------------------------------

#: The measured plan-tools leak: ``add_design_decision.decision`` mis-closed and
#: swallowing its ``rationale`` sibling. ``pattern`` and ``misclose`` are the
#: leaked markup itself, which is why they are BUILT rather than written.
_MISCLOSE = _close('decision')
_PATTERN = _open_param('rationale')


def make_fact(**overrides: Any) -> dict[str, Any]:
    """One ``markup_detected`` record, keyed exactly as the middleware keys it.

    Spelled here rather than imported — the same reason
    ``test_markup_journal.make_fact`` is — so a middleware key rename shows up
    as a failing assertion in the module that CONSUMES the record, instead of
    silently re-shaping the stamp's own contract.
    """
    fact = {
        'fact': FACT_MARKUP_DETECTED,
        'tool': 'add_design_decision',
        'param': 'decision',
        'pattern': _PATTERN,
        'misclose': _MISCLOSE,
        'outcome': 'rejected',
        'recovered_params': ['rationale'],
        # Structurally None on this boundary: ``_identity`` reads only
        # arguments named agent_id / project_root / project_id, and no
        # plan-tools tool declares any of the three.
        'agent_id': None,
        'project': None,
    }
    fact.update(overrides)
    return fact


class _Clock:
    """A hand-cranked time source, so a stamped ``ts`` is an assertable value."""

    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# ---------------------------------------------------------------------------
# build_event — the allowlist copy.
# ---------------------------------------------------------------------------


class TestBuildEventCopiesTheAllowlistAndNothingElse:
    """One fact record in, exactly four keys out.

    The four are the ones the task asks for: WHEN, WHICH TOOL, WHICH
    PARAMETER, and WHAT HAPPENED. Everything else the middleware emits is
    deliberately left behind — see the negative control below for the three
    measured reasons.
    """

    def test_the_event_carries_exactly_the_four_declared_keys(self):
        event = plan_markup_stamp.build_event(make_fact(), now=_Clock())

        assert set(event) == {'ts', 'tool', 'param', 'outcome'}
        assert set(event) == set(plan_markup_stamp.STAMP_EVENT_KEYS), (
            'STAMP_EVENT_KEYS is the DECLARED allowlist; the event must be '
            'built from it rather than happening to agree with it'
        )

    def test_the_three_identity_fields_are_copied_verbatim(self):
        event = plan_markup_stamp.build_event(
            make_fact(tool='add_reuse_item', param='how', outcome='unrepairable'),
            now=_Clock(),
        )

        assert event['tool'] == 'add_reuse_item'
        assert event['param'] == 'how'
        assert event['outcome'] == 'unrepairable', (
            "the fact channel's own vocabulary is recorded verbatim rather "
            'than re-derived into the caller-facing error_type spelling, whose '
            "rejected literal is private to the middleware's _reject"
        )

    def test_the_timestamp_comes_from_the_injected_clock_as_utc_iso(self):
        clock = _Clock(1_700_000_000.0)

        event = plan_markup_stamp.build_event(make_fact(), now=clock)

        assert event['ts'] == datetime.fromtimestamp(clock.now, tz=UTC).isoformat()
        assert event['ts'].endswith('+00:00'), 'UTC, explicitly offset — never naive'

    def test_a_long_field_is_capped_to_a_prefix(self):
        """A trimmed value is a PREFIX of what was sent, never a rewrite.

        ``tool`` and ``param`` come from the invoked tool's own registration and
        schema so they are short in practice, but the cap is what keeps that a
        GUARANTEE rather than an observation — this block rides inside a
        document embedded verbatim into four architect-facing prompts.
        """
        overlong = 'x' * (plan_markup_stamp.MARKUP_STAMP_MAX_FIELD_CHARS + 500)

        event = plan_markup_stamp.build_event(
            make_fact(tool=overlong, param=overlong), now=_Clock()
        )

        cap = plan_markup_stamp.MARKUP_STAMP_MAX_FIELD_CHARS
        assert len(event['tool']) == cap
        assert len(event['param']) == cap
        assert overlong.startswith(event['tool'])

    def test_a_fact_that_grows_a_key_cannot_leak_into_the_plan(self):
        """The copy is an ALLOWLIST, never a filtered copy of the record.

        A filtered copy inverts the default: a middleware that grows a field
        tomorrow would land it in ``plan.json`` by default, and the whole
        argument for what this block may hold rests on it holding nothing else.
        """
        event = plan_markup_stamp.build_event(
            make_fact(some_future_field='a value nobody has reviewed yet'),
            now=_Clock(),
        )

        assert 'some_future_field' not in event
        assert set(event) == {'ts', 'tool', 'param', 'outcome'}


class TestTheEventCarriesNoEnvelopeMarkup:
    """The load-bearing negative control.

    ``pattern`` and ``misclose`` ARE the leaked markup, and this is why they are
    left behind. Three independent measured reasons:

    1. ``plan.json`` is embedded verbatim into four architect-facing prompts
       (``briefing.py``'s revalidation and completion passes, ``workflow._replan``
       and the evals judge) and agents EDIT the document, so a literal stored
       there reproduces the exact over-consumption defect at the one artifact a
       reader is told to open. It is the same reason ``markup_journal`` escapes
       its own bytes and the committed specimen corpus escapes every literal.
    2. ``scripts/sweep_toolcall_markup.py`` walks dead-lane ``plan.json``
       RECURSIVELY, so a stored literal would be classified as fresh corruption
       and inflate the very census the sweep exists to report.
    3. The raw payload already has exactly one owner — the residue escalation,
       which is by contract the only surviving copy — and a second, weaker copy
       is the INV-5 duplication the containment PRD rules against.
    """

    def test_no_value_of_the_event_carries_markup(self):
        record = make_fact()
        # THE TWO SPECIMENS NEED TWO PREDICATES, because ``pattern`` and
        # ``misclose`` are different things (the middleware keeps them apart on
        # purpose, PRD 2.2). ``pattern`` is the ENUMERATED envelope literal
        # ``detect`` matched. ``misclose`` is the tag that actually drifted,
        # verbatim — a raw sentinel by this module's own import-time predicate,
        # but not one of ``ENVELOPE_LITERALS``, so ``detect`` does not see it.
        # Asserting ``detect`` on both would make the control silently vacuous
        # on the mis-close half.
        assert detect(record['pattern']) is not None, (
            'the specimen must actually be markup, or this control proves '
            'nothing'
        )
        assert any(seq in record['misclose'] for seq in _FORBIDDEN_SEQUENCES), (
            'the mis-close specimen must actually be a raw sentinel'
        )

        event = plan_markup_stamp.build_event(record, now=_Clock())

        for name, value in event.items():
            assert detect(value) is None, f'{name} carries envelope markup'
            assert not any(seq in str(value) for seq in _FORBIDDEN_SEQUENCES), (
                f'{name} carries a raw envelope sentinel'
            )

    def test_the_encoded_event_carries_markup_nowhere(self):
        """Asserted on the WHOLE encoding, not merely field by field.

        A per-field sweep would miss a literal that only exists once the block
        is serialised — which is the form every downstream reader actually
        encounters it in.
        """
        event = plan_markup_stamp.build_event(make_fact(), now=_Clock())

        encoded = json.dumps(event)
        assert detect(encoded) is None
        for sequence in _FORBIDDEN_SEQUENCES:
            assert sequence not in encoded, (
                f'the stamped event carries the raw sentinel {sequence!r}'
            )

    def test_the_caller_adjacent_fields_are_not_copied(self):
        """``recovered_params``, ``agent_id`` and ``project`` stay behind.

        Not because they are dangerous — they are names, and the journal keeps
        them — but because this block's whole justification is that it holds no
        caller-supplied bytes at all. ``recovered_params`` is derived from the
        caller's own payload; the other two are structurally ``None`` on this
        boundary and would be pure noise.
        """
        event = plan_markup_stamp.build_event(
            make_fact(agent_id='claude-task-4528-architect', project='dark_factory'),
            now=_Clock(),
        )

        assert 'recovered_params' not in event
        assert 'agent_id' not in event
        assert 'project' not in event
        assert 'pattern' not in event
        assert 'misclose' not in event
        assert 'fact' not in event
