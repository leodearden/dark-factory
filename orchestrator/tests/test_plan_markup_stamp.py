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

import pytest
from shared.mcp_markup_middleware import FACT_MARKUP_DETECTED
from shared.toolcall_markup import ENVELOPE_LITERALS, detect

from orchestrator.artifacts import TaskArtifacts
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

#: The prose the leak ABSORBED — the caller's own bytes. It appears in no fact
#: field this stamp copies, which is exactly what the artifact-level control
#: asserts: the refused payload has one owner, the residue escalation.
_RATIONALE_LEAK_PROSE = (
    'The two layers guard different populations: arguments being sent now, '
    'versus damage already stored in plan.json.'
)


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


# ---------------------------------------------------------------------------
# The block algebra — pure functions over plain dicts.
# ---------------------------------------------------------------------------


def _events(count: int, *, tool: str = 'add_design_decision') -> list[dict[str, Any]]:
    """*count* stamped events one second apart, so their order is assertable."""
    clock = _Clock()
    built = []
    for _ in range(count):
        built.append(plan_markup_stamp.build_event(make_fact(tool=tool), now=clock))
        clock.advance(1.0)
    return built


class TestBlockOfBuildsTheOneEventBlock:
    """The identity every merge starts from."""

    def test_a_single_event_block_is_fully_populated(self):
        (event,) = _events(1)

        block = plan_markup_stamp.block_of(event)

        assert block['count'] == 1
        assert block['by_tool'] == {'add_design_decision': 1}
        assert block['first_at'] == event['ts']
        assert block['last_at'] == event['ts']
        assert block['events'] == [event]
        assert block['note'] == plan_markup_stamp.STAMP_NOTE

    def test_nothing_was_cut_so_the_disclosure_key_is_absent(self):
        """PRESENCE means something was dropped — never merely that it ran."""
        (event,) = _events(1)

        assert 'events_truncated' not in plan_markup_stamp.block_of(event)


class TestMergeBlockAccumulates:
    """``merge_block(left, right)`` — the whole of the on-disk update."""

    def test_counts_sum_and_by_tool_sums_per_tool(self):
        first = plan_markup_stamp.block_of(_events(1, tool='add_plan_step')[0])
        second = plan_markup_stamp.block_of(_events(1, tool='add_design_decision')[0])
        third = plan_markup_stamp.block_of(_events(1, tool='add_design_decision')[0])

        merged = plan_markup_stamp.merge_block(
            plan_markup_stamp.merge_block(first, second), third
        )

        assert merged['count'] == 3
        assert merged['by_tool'] == {'add_plan_step': 1, 'add_design_decision': 2}

    def test_the_window_bounds_take_the_min_and_the_max(self):
        events = _events(3)
        blocks = [plan_markup_stamp.block_of(event) for event in events]

        # Merged OUT OF ORDER on purpose: the bounds are min/max over values,
        # never "whatever arrived first and last".
        merged = plan_markup_stamp.merge_block(
            plan_markup_stamp.merge_block(blocks[2], blocks[0]), blocks[1]
        )

        assert merged['first_at'] == events[0]['ts']
        assert merged['last_at'] == events[2]['ts']

    def test_a_null_bound_never_wins_a_min_or_a_max(self):
        """A bound is taken over the values that EXIST.

        A block whose ``first_at`` is missing or null must not pull the merged
        window's start to nothing — that would erase a bound this side actually
        knows.
        """
        (event,) = _events(1)
        known = plan_markup_stamp.block_of(event)
        blank = {'count': 1, 'by_tool': {'confirm_plan': 1}, 'first_at': None,
                 'last_at': None, 'events': []}

        merged = plan_markup_stamp.merge_block(blank, known)

        assert merged['first_at'] == event['ts']
        assert merged['last_at'] == event['ts']

    def test_the_note_is_always_rewritten_from_the_constant(self):
        """A stale or hand-edited note on disk is CORRECTED, not inherited.

        The constant stays the single owner of the wording; ``plan.json`` is
        agent-adjacent, so a note that drifted must not survive a merge.
        """
        stale = plan_markup_stamp.block_of(_events(1)[0])
        stale['note'] = 'Some earlier wording an agent edited by hand.'

        merged = plan_markup_stamp.merge_block(stale, plan_markup_stamp.block_of(_events(1)[0]))

        assert merged['note'] == plan_markup_stamp.STAMP_NOTE

    def test_merging_is_associative_over_a_sequence(self):
        """Left-folded and right-folded must agree, or the on-disk block drifts.

        The sink folds one incoming event into whatever is on disk, and
        ``_create_plan`` folds a whole buffered block into a carried-forward
        one. Those are different association orders over the same events.
        """
        blocks = [plan_markup_stamp.block_of(event) for event in _events(4)]

        left = blocks[0]
        for block in blocks[1:]:
            left = plan_markup_stamp.merge_block(left, block)
        right = blocks[-1]
        for block in reversed(blocks[:-1]):
            right = plan_markup_stamp.merge_block(block, right)

        assert left == right


class TestTheEventListIsCapped:
    """Bounded, because ``plan.json`` rides verbatim into four prompts."""

    def test_the_first_n_events_are_kept_and_the_rest_dropped(self):
        """FIRST N, not last: the list stops changing once it is full.

        A pathological leak then churns two integers instead of rewriting the
        whole block on every refusal.
        """
        cap = plan_markup_stamp.MARKUP_STAMP_MAX_EVENTS
        events = _events(cap + 5)

        merged = plan_markup_stamp.block_of(events[0])
        for event in events[1:]:
            merged = plan_markup_stamp.merge_block(
                merged, plan_markup_stamp.block_of(event)
            )

        assert merged['events'] == events[:cap]
        assert merged['count'] == cap + 5, 'the COUNT is never capped'

    def test_the_disclosure_key_appears_only_once_something_was_cut(self):
        cap = plan_markup_stamp.MARKUP_STAMP_MAX_EVENTS
        events = _events(cap + 1)

        at_cap = plan_markup_stamp.block_of(events[0])
        for event in events[1:cap]:
            at_cap = plan_markup_stamp.merge_block(
                at_cap, plan_markup_stamp.block_of(event)
            )
        assert len(at_cap['events']) == cap
        assert 'events_truncated' not in at_cap, 'full is not the same as cut'

        over_cap = plan_markup_stamp.merge_block(
            at_cap, plan_markup_stamp.block_of(events[cap])
        )

        assert over_cap['events_truncated'] is True
        assert over_cap['count'] - len(over_cap['events']) == 1, (
            'how many were dropped is already derivable from the two numbers '
            'present; a stored dropped-count would be a second accounting able '
            'to drift from them'
        )

    def test_by_tool_stays_complete_past_the_event_cap(self):
        """The aggregate is what actually answers the question.

        It is bounded by the TOOL SURFACE rather than by the leak, so it stays
        complete and small even when ``events`` is full.
        """
        cap = plan_markup_stamp.MARKUP_STAMP_MAX_EVENTS
        merged = plan_markup_stamp.block_of(_events(1, tool='add_design_decision')[0])
        for _ in range(cap + 5):
            merged = plan_markup_stamp.merge_block(
                merged, plan_markup_stamp.block_of(_events(1, tool='add_reuse_item')[0])
            )

        assert merged['by_tool'] == {'add_design_decision': 1, 'add_reuse_item': cap + 5}


class TestMergeIsTotalAgainstAMangledBlock:
    """``plan.json`` is agent-adjacent, so the block on disk may be anything.

    A merge runs INSIDE a decided refusal, so raising here would turn a working
    guard into an outage of its own. Every field degrades to its identity and
    the INCOMING event survives — a partially-recovered count beats a lost one.
    """

    def test_a_non_dict_block_merges_without_raising(self):
        incoming = plan_markup_stamp.block_of(_events(1)[0])

        for mangled in ('a string', 42, None, ['a', 'list']):
            merged = plan_markup_stamp.merge_block(mangled, incoming)

            assert merged['count'] == 1, f'{mangled!r} lost the incoming event'
            assert merged['by_tool'] == {'add_design_decision': 1}
            assert merged['events'] == incoming['events']

    def test_a_non_int_count_degrades_to_the_recoverable_total(self):
        incoming = plan_markup_stamp.block_of(_events(1)[0])
        mangled = {**plan_markup_stamp.block_of(_events(1)[0]), 'count': 'seven'}

        merged = plan_markup_stamp.merge_block(mangled, incoming)

        assert merged['count'] == 1, (
            'the unusable side contributes its identity; the incoming event is '
            'never the thing that gets dropped'
        )

    def test_a_non_dict_by_tool_degrades_without_losing_the_incoming_tally(self):
        incoming = plan_markup_stamp.block_of(_events(1)[0])
        mangled = {**plan_markup_stamp.block_of(_events(1)[0]), 'by_tool': 'nope'}

        merged = plan_markup_stamp.merge_block(mangled, incoming)

        assert merged['by_tool'] == {'add_design_decision': 1}

    def test_a_non_list_events_degrades_without_losing_the_incoming_event(self):
        incoming = plan_markup_stamp.block_of(_events(1)[0])
        mangled = {**plan_markup_stamp.block_of(_events(1)[0]), 'events': {'not': 'a list'}}

        merged = plan_markup_stamp.merge_block(mangled, incoming)

        assert merged['events'] == incoming['events']

    def test_a_by_tool_holding_junk_counts_still_merges(self):
        incoming = plan_markup_stamp.block_of(_events(1)[0])
        mangled = {
            **plan_markup_stamp.block_of(_events(1)[0]),
            'by_tool': {'add_design_decision': 'lots', 'add_plan_step': 2},
        }

        merged = plan_markup_stamp.merge_block(mangled, incoming)

        assert merged['by_tool']['add_design_decision'] == 1
        assert merged['by_tool']['add_plan_step'] == 2


class TestSummaryIsTheCompactView:
    """What ``_confirm_plan`` folds into the architect's LAST tool result."""

    def test_a_plan_with_a_block_summarises_to_count_and_by_tool(self):
        block = plan_markup_stamp.merge_block(
            plan_markup_stamp.block_of(_events(1, tool='add_plan_step')[0]),
            plan_markup_stamp.block_of(_events(1, tool='add_design_decision')[0]),
        )
        plan = {'steps': [], plan_markup_stamp.PLAN_MARKUP_REJECTIONS_KEY: block}

        summary = plan_markup_stamp.summary(plan)

        assert summary == {
            'count': 2,
            'by_tool': {'add_plan_step': 1, 'add_design_decision': 1},
        }

    def test_the_summary_carries_no_events_and_no_note(self):
        """A signal to the architect, not a second copy of the block."""
        block = plan_markup_stamp.block_of(_events(1)[0])
        plan = {plan_markup_stamp.PLAN_MARKUP_REJECTIONS_KEY: block}

        summary = plan_markup_stamp.summary(plan)

        assert summary is not None
        assert set(summary) == {'count', 'by_tool'}

    def test_a_plan_with_no_block_summarises_to_none(self):
        """Absent, so the omit-when-absent convention has something to omit."""
        assert plan_markup_stamp.summary({'steps': [], 'files': ['a.py']}) is None
        assert plan_markup_stamp.summary({}) is None

    def test_an_unusable_block_summarises_to_none(self):
        for mangled in ('a string', 42, None, ['a', 'list']):
            plan = {plan_markup_stamp.PLAN_MARKUP_REJECTIONS_KEY: mangled}

            assert plan_markup_stamp.summary(plan) is None, (
                f'{mangled!r} must not reach a tool response half-formed'
            )


# ---------------------------------------------------------------------------
# make_plan_stamp — the sink, over a real TaskArtifacts.
# ---------------------------------------------------------------------------


#: Every field an AGENT authored, plus the writer's own ``_schema_version``.
#: The narrowed contract is stated in these terms: a refusal leaves each of
#: them identical, and the block is the ONLY difference.
AUTHORED_KEYS = (
    'task_id', 'title', 'analysis', 'files', 'prerequisites', 'steps',
    'design_decisions', 'reuse', '_schema_version',
)


@pytest.fixture(autouse=True)
def _clear_pending_refusals():
    """Clear the pending buffer around every test in this module.

    It is PROCESS-global state, mirroring ``plan_tools._REPORTED_REFUSALS`` and
    the ``_clear_reported_refusals`` fixture that guards it — so one test's
    buffered refusal cannot leak into the next and inflate its count.
    """
    plan_markup_stamp.clear_pending()
    yield
    plan_markup_stamp.clear_pending()


@pytest.fixture()
def artifacts(tmp_path) -> TaskArtifacts:
    """TaskArtifacts over a temp worktree — mirrors ``test_plan_tools_server``."""
    a = TaskArtifacts(tmp_path)
    a.init('test-1', 'Test task', 'A test')
    return a


def seed_plan(artifacts: TaskArtifacts) -> dict[str, Any]:
    """A plan built through the REAL writer, as ``_create_plan`` builds one."""
    from orchestrator.mcp import plan_tools

    plan_tools._create_plan(
        artifacts,
        'test-1',
        'A clean plan',
        'Clean analysis prose describing the approach.',
        ['orchestrator/src/orchestrator/mcp/plan_tools.py'],
    )
    plan_tools._add_plan_step(artifacts, 'step-1', 'test', 'A clean step.')
    return artifacts.read_plan()


def plan_on_disk(artifacts: TaskArtifacts) -> dict[str, Any]:
    """``plan.json`` parsed straight off disk, bypassing any normalisation."""
    return json.loads(
        (artifacts.root / 'plan.json').read_text(encoding='utf-8')
    )


class TestThePlanStampRecordsTheRefusal:
    """The sink's happy path: one fact in, one accumulated block on disk."""

    @pytest.mark.asyncio
    async def test_one_fact_lands_as_a_one_event_block(self, artifacts: TaskArtifacts):
        seed_plan(artifacts)
        stamp = plan_markup_stamp.make_plan_stamp(artifacts=artifacts, now=_Clock())

        locator = await stamp(make_fact())

        block = plan_on_disk(artifacts)[plan_markup_stamp.PLAN_MARKUP_REJECTIONS_KEY]
        assert block['count'] == 1
        assert block['by_tool'] == {'add_design_decision': 1}
        assert len(block['events']) == 1
        assert block['events'][0]['param'] == 'decision'
        assert block['note'] == plan_markup_stamp.STAMP_NOTE
        assert locator == str(artifacts.root / 'plan.json'), (
            'the locator names the artifact, as the journal sink names its file'
        )

    @pytest.mark.asyncio
    async def test_every_authored_field_survives_untouched(
        self, artifacts: TaskArtifacts
    ):
        """THE NARROWED CONTRACT, asserted directly.

        The old pin was "a refused call leaves plan.json byte-identical". It
        was a proxy for a property about VALUES — the registration comment
        justifies the reject policy as preventing "a guessed-at document that
        every later reader inherits", and the middleware header says "no
        middleware-repaired value can ever reach plan.json". That property is
        preserved intact and is what this asserts: the block is the only
        difference, and it holds no caller-supplied bytes at all.
        """
        seed_plan(artifacts)
        before = plan_on_disk(artifacts)
        assert plan_markup_stamp.PLAN_MARKUP_REJECTIONS_KEY not in before
        stamp = plan_markup_stamp.make_plan_stamp(artifacts=artifacts, now=_Clock())

        await stamp(make_fact())

        after = plan_on_disk(artifacts)
        assert after.pop(plan_markup_stamp.PLAN_MARKUP_REJECTIONS_KEY)
        assert after == before, 'the block is the ONLY difference'
        for key in AUTHORED_KEYS:
            assert after[key] == before[key], f'{key} was disturbed'

    @pytest.mark.asyncio
    async def test_a_second_refusal_accumulates_rather_than_replacing(
        self, artifacts: TaskArtifacts
    ):
        seed_plan(artifacts)
        clock = _Clock()
        stamp = plan_markup_stamp.make_plan_stamp(artifacts=artifacts, now=clock)

        await stamp(make_fact())
        clock.advance(5.0)
        await stamp(make_fact(tool='add_reuse_item', param='how'))

        block = plan_on_disk(artifacts)[plan_markup_stamp.PLAN_MARKUP_REJECTIONS_KEY]
        assert block['count'] == 2
        assert block['by_tool'] == {'add_design_decision': 1, 'add_reuse_item': 1}
        assert len(block['events']) == 2
        assert block['first_at'] != block['last_at'], 'the window has two ends'

    @pytest.mark.asyncio
    async def test_the_stamped_block_carries_no_envelope_markup(
        self, artifacts: TaskArtifacts
    ):
        """The negative control, asserted on the ARTIFACT rather than the event.

        ``plan.json`` is the thing embedded verbatim into four architect-facing
        prompts and walked recursively by the dead-lane sweep, so the predicate
        that matters is the one applied to the file.
        """
        seed_plan(artifacts)
        stamp = plan_markup_stamp.make_plan_stamp(artifacts=artifacts, now=_Clock())

        await stamp(make_fact())

        text = (artifacts.root / 'plan.json').read_text(encoding='utf-8')
        for sequence in _FORBIDDEN_SEQUENCES:
            assert sequence not in text, (
                f'the stamped plan carries the raw sentinel {sequence!r}'
            )
        assert _RATIONALE_LEAK_PROSE not in text, (
            'the refused payload has exactly one owner, the residue escalation'
        )


class TestThePlanStampNeverRaises:
    """A sink outage costs visibility, never an outcome.

    The call is already DECIDED by the time this runs, so the whole channel is
    additive — the same never-raises contract ``markup_journal`` and
    ``markup_sink`` keep, and for the same reason.
    """

    @pytest.mark.asyncio
    async def test_a_failing_write_is_swallowed_and_the_plan_is_left_alone(
        self, artifacts: TaskArtifacts, monkeypatch
    ):
        seed_plan(artifacts)
        before = (artifacts.root / 'plan.json').read_bytes()

        def boom(plan):
            raise OSError('the plan is unwritable')

        monkeypatch.setattr(artifacts, 'write_plan', boom)
        stamp = plan_markup_stamp.make_plan_stamp(artifacts=artifacts, now=_Clock())

        assert await stamp(make_fact()) is None
        assert (artifacts.root / 'plan.json').read_bytes() == before

    @pytest.mark.asyncio
    async def test_a_failing_read_is_swallowed(
        self, artifacts: TaskArtifacts, monkeypatch
    ):
        seed_plan(artifacts)

        def boom():
            raise ValueError('plan.json is not parseable')

        monkeypatch.setattr(artifacts, 'read_plan', boom)
        stamp = plan_markup_stamp.make_plan_stamp(artifacts=artifacts, now=_Clock())

        assert await stamp(make_fact()) is None

    @pytest.mark.asyncio
    async def test_a_fact_missing_keys_still_stamps_something(
        self, artifacts: TaskArtifacts
    ):
        """A degraded record beats a dropped one.

        The middleware builds one complete record on every path, but this sink
        must not be the thing that turns a shape surprise into a lost event —
        that is the exact fail-soft the containment PRD exists to end.
        """
        seed_plan(artifacts)
        stamp = plan_markup_stamp.make_plan_stamp(artifacts=artifacts, now=_Clock())

        assert await stamp({}) is not None

        block = plan_on_disk(artifacts)[plan_markup_stamp.PLAN_MARKUP_REJECTIONS_KEY]
        assert block['count'] == 1, 'the event is recorded even when unnamed'

    @pytest.mark.asyncio
    async def test_a_non_dict_fact_never_propagates(self, artifacts: TaskArtifacts):
        seed_plan(artifacts)
        stamp = plan_markup_stamp.make_plan_stamp(artifacts=artifacts, now=_Clock())

        # Deliberately off-contract: the middleware always hands a dict, and
        # this pins that a shape surprise still cannot reach the caller.
        assert await stamp('not a record at all') is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The pending buffer — the one gap the eager stamp cannot close alone.
# ---------------------------------------------------------------------------


class TestARefusalBeforeAnyPlanExistsIsBuffered:
    """A refused ``create_plan`` has no document to stamp.

    That is the LOUDEST leak shape on this server — an architect bounced
    repeatedly before its plan even exists — so it is the one case the counter
    least affords to lose. The existing ``test_no_plan_is_written`` pins that
    no plan file appears, and the eager stamp must not violate it, so the event
    goes to process-global state and is drained by ``_create_plan``.
    """

    @pytest.mark.asyncio
    async def test_no_plan_file_is_created_by_the_stamp(self, artifacts: TaskArtifacts):
        """The stamp is a BOOKKEEPING channel, never a plan author.

        Writing a plan here would manufacture a document out of a refusal —
        one carrying no task_id, no title and no analysis — and every reader
        downstream would inherit it as the architect's own work.
        """
        stamp = plan_markup_stamp.make_plan_stamp(artifacts=artifacts, now=_Clock())

        await stamp(make_fact(tool='create_plan', param='title'))

        assert not (artifacts.root / 'plan.json').exists(), (
            'a refusal must not conjure a plan that no architect authored'
        )

    @pytest.mark.asyncio
    async def test_the_event_is_held_in_the_pending_buffer(
        self, artifacts: TaskArtifacts
    ):
        stamp = plan_markup_stamp.make_plan_stamp(artifacts=artifacts, now=_Clock())

        await stamp(make_fact(tool='create_plan', param='title'))

        pending = plan_markup_stamp.pending_block()
        assert pending is not None
        assert pending['count'] == 1
        assert pending['by_tool'] == {'create_plan': 1}

    @pytest.mark.asyncio
    async def test_repeated_refusals_accumulate_in_the_buffer(
        self, artifacts: TaskArtifacts
    ):
        clock = _Clock()
        stamp = plan_markup_stamp.make_plan_stamp(artifacts=artifacts, now=clock)

        for _ in range(3):
            await stamp(make_fact(tool='create_plan', param='analysis'))
            clock.advance(1.0)

        pending = plan_markup_stamp.pending_block()
        assert pending is not None
        assert pending['count'] == 3

    def test_an_empty_buffer_reports_nothing(self):
        assert plan_markup_stamp.pending_block() is None


class TestDrainPendingFoldsTheBufferIntoAPlan:
    """``_create_plan`` adopts the losses that preceded the document."""

    def test_draining_folds_the_buffer_and_returns_the_plan(self):
        plan_markup_stamp.note_pending(
            plan_markup_stamp.build_event(
                make_fact(tool='create_plan', param='title'), now=_Clock()
            )
        )
        plan = {'task_id': 'test-1', 'steps': []}

        drained = plan_markup_stamp.drain_pending(plan)

        block = drained[plan_markup_stamp.PLAN_MARKUP_REJECTIONS_KEY]
        assert block['count'] == 1
        assert block['by_tool'] == {'create_plan': 1}
        assert drained['task_id'] == 'test-1', 'the authored fields are untouched'

    def test_draining_clears_the_buffer_so_a_second_drain_cannot_double_count(self):
        plan_markup_stamp.note_pending(
            plan_markup_stamp.build_event(make_fact(tool='create_plan'), now=_Clock())
        )

        first = plan_markup_stamp.drain_pending({'steps': []})
        second = plan_markup_stamp.drain_pending({'steps': []})

        assert first[plan_markup_stamp.PLAN_MARKUP_REJECTIONS_KEY]['count'] == 1
        assert plan_markup_stamp.pending_block() is None
        assert plan_markup_stamp.PLAN_MARKUP_REJECTIONS_KEY not in second, (
            'a re-plan must not re-adopt refusals a prior plan already carries'
        )

    def test_draining_merges_into_a_block_the_plan_already_carries(self):
        """A carried-forward block and a buffered one are two sides of a merge.

        ``_create_plan`` overwrites ``plan.json`` wholesale, so it carries the
        existing block forward and drains the buffer into it — both must land.
        """
        carried = plan_markup_stamp.block_of(
            plan_markup_stamp.build_event(
                make_fact(tool='add_design_decision'), now=_Clock()
            )
        )
        plan_markup_stamp.note_pending(
            plan_markup_stamp.build_event(make_fact(tool='create_plan'), now=_Clock(2_000.0))
        )

        drained = plan_markup_stamp.drain_pending(
            {'steps': [], plan_markup_stamp.PLAN_MARKUP_REJECTIONS_KEY: carried}
        )

        block = drained[plan_markup_stamp.PLAN_MARKUP_REJECTIONS_KEY]
        assert block['count'] == 2
        assert block['by_tool'] == {'add_design_decision': 1, 'create_plan': 1}

    def test_draining_an_empty_buffer_leaves_the_plan_untouched(self):
        """THE OVERWHELMINGLY COMMON PATH: no refusal, so no key at all.

        Not present-and-zero. The block's PRESENCE is the whole signal, so a
        clean plan must stay byte-identical to what it is today — otherwise
        every plan in the fleet grows a key that means nothing.
        """
        plan = {'task_id': 'test-1', 'steps': [], 'design_decisions': []}

        drained = plan_markup_stamp.drain_pending(plan)

        assert plan_markup_stamp.PLAN_MARKUP_REJECTIONS_KEY not in drained
        assert drained == {'task_id': 'test-1', 'steps': [], 'design_decisions': []}

    def test_the_buffer_inherits_the_same_cap_as_the_on_disk_block(self):
        """It holds a FOLDED block, not a growing list.

        A long-leaking session that never creates a plan must not grow
        unbounded process state, so the buffer reuses the on-disk block's cap
        and merge algebra rather than inventing a second accounting.
        """
        cap = plan_markup_stamp.MARKUP_STAMP_MAX_EVENTS
        clock = _Clock()
        for _ in range(cap + 5):
            plan_markup_stamp.note_pending(
                plan_markup_stamp.build_event(make_fact(tool='create_plan'), now=clock)
            )
            clock.advance(1.0)

        pending = plan_markup_stamp.pending_block()

        assert pending is not None
        assert pending['count'] == cap + 5
        assert len(pending['events']) == cap
        assert pending['events_truncated'] is True
