"""The committed specimen corpus, replayed through the MIDDLEWARE.

PRD ``plans/toolcall-markup-containment-prd.md`` task beta, contract C2.

## Why this exists beside ``test_toolcall_markup_corpus.py``

That module replays the same fixture through :func:`shared.toolcall_markup.repair`
DIRECTLY. This one drives every specimen through a live FastMCP server with the
guard registered, which is a different claim: it pins that the BOUNDARY LAYER
preserves the repairer's guarantees end to end — through argument scanning,
through the live schema read, through pydantic's own validation of a real tool
signature, and through whichever of the three outcome paths the declared policy
selects. A repairer that is correct in isolation buys nothing if the middleware
around it hands a still-poisoned value downstream, forwards a refusal, or drops
a recovered parameter on the floor between ``repair()`` and the tool.

## What this module can and cannot prove — stated honestly

``expected_outcome`` and ``expected_recovered`` are MACHINE-GENERATED: the
extractor wrote them by calling the very repairer under test (see
``toolcall_markup_corpus_extract`` and PRD G6). Agreeing with them therefore
pins REGRESSION-STABILITY and DETERMINISM, not correctness. Correctness is
pinned by the hand-authored boundary rows in
``test_mcp_markup_middleware.py`` (B1-B10), against specimens whose right
answer was decided by a human.

The one assertion here that is NOT circular is
:class:`TestNoStillPoisonedValueEverEscapes`. It re-derives its verdict from
:func:`shared.toolcall_markup.detect` against what the tool ACTUALLY received
and what the rejection payload ACTUALLY offered, so it would catch the
middleware laundering a leak past its own guard even if every stored
expectation agreed.

Following that module's PRD G6 discipline, there are NO assertions on how many
specimens repair or refuse. Pinning those figures would make a genuinely BETTER
repairer look like a regression. The only counting here is COVERAGE — that
every record in the file was actually driven — because a replay that silently
shrinks to nothing would otherwise report green.

## Synthetic tools, generated per signature

Each record carries its own ``schema_params``, and FastMCP derives a tool's
parameter schema from the Python function signature — so a fixed toy tool
cannot express 14 different schemas. The tools below are therefore generated
with :func:`exec` from each distinct ``(tool, schema_params)`` signature
observed in the corpus. That is deliberate: the point is to exercise the live
``get_tool`` schema read against the REAL parameter sets these leaks were
emitted against, since ``schema_params`` is exactly what decides whether a
recovered name may be accepted (boundary row B8).

## Storms are suppressed here, on purpose

One middleware instance drives ~500 specimens, so the storm escape would fire
continuously and fold a ``storm`` key into most payloads. INV-4's storm
contract is pinned by boundary row B10 in the sibling module, against an
injected clock. Leaving it live here would turn a repair regression into a
storm-shaped failure without pinning anything the sibling does not already.

## Substrate facts, measured against fastmcp 3.2.2

Middleware is BYPASSED by ``tool.fn(...)``, ``await tool.run({...})`` and
``server._tool_manager.call_tool(...)`` — the repo's two established in-process
idioms — so only a ``Client``-driven call exercises it. See the sibling
module's docstring for the full list.

## Sentinel-literal hazard

This file contains NO envelope literal anywhere, by construction: every
specimen is read from the fixture, which stores ``<`` as ``\\u003c``. Should one
ever need to be written here, spell it ``\\x3c`` — writing ``<`` verbatim would
force an agent editing this file to emit that literal inside its own tool-call
envelope, reproducing the very defect these tests pin.
"""
from __future__ import annotations

import asyncio
import json
import keyword
import logging
from pathlib import Path
from typing import Any, NamedTuple

import pytest
import toolcall_markup_corpus_extract as extract
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError

from shared.mcp_markup_middleware import OUTCOMES, MarkupGuardMiddleware, RepairPolicy
from shared.toolcall_markup import detect

logger = logging.getLogger(__name__)

CORPUS_PATH = Path(__file__).resolve().parent / 'fixtures' / 'toolcall_markup_corpus.jsonl'

#: How many specimens the replay is allowed to skip as not expressible as a
#: live tool call. MEASURED, not aspirational: at the corpus committed by task
#: 3688 every single record round-trips through a synthesized tool, so the
#: honest ceiling is zero. It exists as a named constant rather than an inline
#: ``not skips`` so that a future corpus refresh which genuinely introduces an
#: inexpressible shape has to RAISE this number in a diff someone reviews,
#: instead of quietly shrinking the replay behind a green run.
MAX_SKIPPED = 0

#: Set beyond any plausible corpus size so no burst fires. See the module
#: docstring: INV-4 is boundary row B10's, in the sibling module.
_NO_STORM = 10**9

#: The corpus records a REPAIRER verdict (two values). The middleware records a
#: POLICY verdict (three), because the two tiers disagree about what to do with
#: a repair and agree exactly about a refusal. This is that mapping, and it is
#: the whole reason a middleware-layer replay says something the direct replay
#: does not.
_EXPECTED_OUTCOME = {
    (extract.OUTCOME_REPAIRED, RepairPolicy.FORWARD_REPAIR): 'repaired',
    (extract.OUTCOME_REPAIRED, RepairPolicy.REJECT_WITH_REPAIR): 'rejected',
    (extract.OUTCOME_UNREPAIRABLE, RepairPolicy.FORWARD_REPAIR): 'unrepairable',
    (extract.OUTCOME_UNREPAIRABLE, RepairPolicy.REJECT_WITH_REPAIR): 'unrepairable',
}

#: Filled into every supplied argument that is NOT the leaking one. Clean by
#: construction, so the middleware's first-hit argument scan can only ever land
#: on the record's own ``param`` — which is what makes a disagreement about the
#: repaired field a real finding rather than an artefact of this harness.
_FILLER = 'placeholder'


# ---------------------------------------------------------------------------
# Synthesizing a tool whose schema is exactly a record's ``schema_params``.
# ---------------------------------------------------------------------------


def _signature(record: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    return record['tool'], tuple(record['schema_params'])


def _sanitized(tool: str) -> str:
    """A Python-identifier spelling of an agent-facing tool name.

    The corpus records names as the agent sees them
    (``mcp__plan-tools__add_design_decision``), which carry hyphens. The name is
    cosmetic here — nothing asserts on it — but keeping it recognisable is what
    makes a failure report point at a real tool.
    """
    return ''.join(char if char.isalnum() or char == '_' else '_' for char in tool)


def _tool_names(corpus: list[dict[str, Any]]) -> dict[tuple[str, tuple[str, ...]], str]:
    """One registered tool name per distinct ``(tool, schema_params)`` signature.

    A tool observed with two different parameter sets across the archive gets
    two tools, because the schema is the thing under test: collapsing them would
    validate half the specimens against a schema they never arrived with.

    Only EXPRESSIBLE records contribute a signature. A record whose parameter
    names cannot be a Python signature is a counted, logged
    :class:`Skip` — but the tools are built for the whole corpus up front, before
    any per-record skip runs, so without this filter one inexpressible record
    would take the entire harness down at construction instead of routing
    through ``MAX_SKIPPED``.
    """
    names: dict[tuple[str, tuple[str, ...]], str] = {}
    seen: dict[str, int] = {}
    expressible = {
        _signature(record) for record in corpus if _why_not_expressible(record) is None
    }
    for signature in sorted(expressible):
        base = _sanitized(signature[0])
        index = seen.get(base, 0)
        seen[base] = index + 1
        names[signature] = f'{base}__s{index}'
    return names


def _synthetic_tool(name: str, params: tuple[str, ...], sink: Any) -> Any:
    """A function whose signature is exactly *params*, recording what it received.

    ``exec`` rather than ``functools.partial``/``**kwargs``, because FastMCP
    builds the tool's JSON Schema from the real signature — a ``**kwargs`` tool
    would advertise NO parameters, so ``schema_params`` would come back empty
    and every recovery would be refused as out-of-schema (boundary row B8),
    turning the whole replay into a vacuous pass.

    Every parameter is optional so a record's ``supplied`` subset is a legal
    call, and typed ``str | None`` so a raw specimen validates unchanged rather
    than being coerced on its way in.

    The "corpus identifiers only" property that makes the ``exec`` safe is
    ASSERTED here, not merely asserted ABOUT here. *name* is sanitized by
    :func:`_sanitized`, but the parameter names are interpolated straight into a
    ``def`` signature — and the corpus is machine-generated from a live
    transcript archive and explicitly expected to be refreshed. A refresh
    yielding a name that is not a Python identifier, or that is a keyword
    (``from``, ``class`` — both ``isidentifier()``-true and both a
    ``SyntaxError`` in a signature), must fail by NAMING that name rather than
    as an opaque error inside a fixture. :func:`_why_not_expressible` routes
    such records to the counted skip path first; this is the backstop that
    keeps the two in step.
    """
    unusable = [p for p in params if not p.isidentifier() or keyword.iskeyword(p)]
    assert not unusable, (
        f'{name} cannot be synthesized: {unusable} are not usable Python '
        'parameter names. _why_not_expressible must route these to a counted '
        'Skip before the harness reaches here.'
    )
    source = (
        f'def {name}(' + ', '.join(f'{p}: str | None = None' for p in params) + '):\n'
        '    sink({' + ', '.join(f'{p!r}: {p}' for p in params) + '})\n'
        "    return 'ok'\n"
    )
    namespace: dict[str, Any] = {'sink': sink}
    exec(source, namespace)  # noqa: S102 — see the docstring; every interpolated
    return namespace[name]                # name is asserted to be an identifier


# ---------------------------------------------------------------------------
# The sweep.
# ---------------------------------------------------------------------------


class Skip(NamedTuple):
    """A specimen the harness could not express as a live tool call."""

    tool_use_id: str
    reason: str


class Observation(NamedTuple):
    """Everything one specimen's trip through the guard is judged on."""

    tool_use_id: str
    policy: str
    param: str
    raw_value: str
    supplied: tuple[str, ...]
    expected_outcome: str
    expected_recovered: tuple[str, ...]
    #: The middleware's verdict, read from the forwarded result's ``meta`` or
    #: from the refusal payload — never inferred from which branch we took.
    outcome: str | None
    error_type: str | None
    recovered_params: tuple[str, ...]
    #: What the tool body ACTUALLY received. Empty means it never ran.
    delivered: tuple[dict[str, Any], ...]
    repaired_call: dict[str, Any] | None
    escalations: tuple[dict[str, Any], ...]
    facts: tuple[dict[str, Any], ...]


class Replay(NamedTuple):
    observations: tuple[Observation, ...]
    skips: tuple[Skip, ...]


def _why_not_expressible(record: dict[str, Any]) -> str | None:
    """Why this record cannot be driven as a live call, or ``None``.

    Every reason is a statement about the RECORD, never about the middleware:
    a specimen skipped because the guard mishandled it would be exactly the
    silently-shrinking replay ``MAX_SKIPPED`` exists to prevent.
    """
    names = [record['param'], *record['schema_params'], *record['supplied']]
    # Keywords are ``isidentifier()``-true and still a SyntaxError in a ``def``
    # signature, so ``from``/``class``/``lambda`` have to be excluded here or
    # they reach :func:`_synthetic_tool`'s exec.
    unusable = [
        name for name in names if not name.isidentifier() or keyword.iskeyword(name)
    ]
    if unusable:
        return f'parameter name(s) are not usable Python names: {sorted(set(unusable))}'
    if record['param'] not in record['supplied']:
        return 'the leaking param is not among the supplied arguments'
    if record['param'] not in record['schema_params']:
        return 'the leaking param is not a parameter of its own tool'
    return None


def _arguments_for(record: dict[str, Any]) -> dict[str, Any]:
    """The call as the leaking agent made it: the raw value at its own param."""
    return {
        name: (record['value'] if name == record['param'] else f'{_FILLER}-{name}')
        for name in record['supplied']
    }


class _Harness:
    """One server per policy, carrying a tool for every signature in the corpus."""

    def __init__(self, policy: RepairPolicy, corpus: list[dict[str, Any]]) -> None:
        self.policy = policy
        self.names = _tool_names(corpus)
        self.delivered: list[dict[str, Any]] = []
        self.facts: list[dict[str, Any]] = []
        self.escalations: list[dict[str, Any]] = []
        self.mcp = FastMCP(f'markup-corpus-{policy.value}')
        for signature, name in self.names.items():
            self.mcp.tool(_synthetic_tool(name, signature[1], self.delivered.append))
        self.mcp.add_middleware(
            MarkupGuardMiddleware(
                policy,
                fact_sink=self.facts.append,
                escalation_sink=self._escalate,
                storm_threshold=_NO_STORM,
            )
        )

    def _escalate(self, record: dict[str, Any]) -> str:
        self.escalations.append(record)
        return f'esc-corpus-{len(self.escalations)}'

    def reset(self) -> None:
        """Clear the per-specimen channels, so each observation is attributable."""
        self.delivered.clear()
        self.facts.clear()
        self.escalations.clear()


def _observe(record: dict[str, Any], harness: _Harness, result: Any, error: ToolError | None):
    """Fold one specimen's outcome into an :class:`Observation`."""
    repaired_call = None
    error_type = None
    if error is not None:
        payload = json.loads(str(error))
        outcome = payload.get('outcome')
        error_type = payload.get('error_type')
        recovered = payload.get('recovered_params') or []
        repaired_call = payload.get('repaired_call')
    else:
        warning = (result.meta or {}).get('markup_repair') or {}
        outcome = warning.get('outcome')
        recovered = warning.get('recovered_params') or []

    return Observation(
        tool_use_id=record['tool_use_id'],
        policy=harness.policy.value,
        param=record['param'],
        raw_value=record['value'],
        supplied=tuple(record['supplied']),
        expected_outcome=record['expected_outcome'],
        expected_recovered=tuple(sorted(record['expected_recovered'])),
        outcome=outcome,
        error_type=error_type,
        recovered_params=tuple(sorted(recovered)),
        delivered=tuple(harness.delivered),
        repaired_call=repaired_call,
        escalations=tuple(harness.escalations),
        facts=tuple(harness.facts),
    )


async def _sweep(policy: RepairPolicy, corpus: list[dict[str, Any]]) -> Replay:
    """Drive every specimen through one server, in corpus order."""
    harness = _Harness(policy, corpus)
    observations: list[Observation] = []
    skips: list[Skip] = []

    async with Client(harness.mcp) as client:
        for record in corpus:
            reason = _why_not_expressible(record)
            if reason is not None:
                # Counted AND logged: a skip nobody can see is precisely the
                # silent degradation this repo's ratchet exists to forbid.
                logger.warning(
                    'corpus replay skipped %s: %s', record['tool_use_id'], reason
                )
                skips.append(Skip(record['tool_use_id'], reason))
                continue

            harness.reset()
            name = harness.names[_signature(record)]
            try:
                result = await client.call_tool(name, _arguments_for(record))
            except ToolError as exc:
                observations.append(_observe(record, harness, None, exc))
            else:
                observations.append(_observe(record, harness, result, None))

    return Replay(tuple(observations), tuple(skips))


def _digest(replay: Replay) -> str:
    """A canonical serialization of a whole sweep, for the determinism pin."""
    return json.dumps(
        [observation._asdict() for observation in replay.observations],
        sort_keys=True,
        default=str,
    )


@pytest.fixture(scope='module')
def corpus() -> list[dict[str, Any]]:
    return extract.load_corpus(CORPUS_PATH)


@pytest.fixture(scope='module')
def replay(corpus) -> dict[RepairPolicy, Replay]:
    """One full sweep per declared tier, computed once for the whole module.

    Module scoped because the sweep is the expensive part (~1s per tier) and
    every assertion below is a pure read over its result. The fixture is SYNC
    and owns its own loop: pytest-asyncio's default function-scoped loop would
    otherwise close under a client held across tests.
    """
    return {policy: asyncio.run(_sweep(policy, corpus)) for policy in RepairPolicy}


def _all(replay: dict[RepairPolicy, Replay]) -> list[Observation]:
    return [obs for sweep in replay.values() for obs in sweep.observations]


# ---------------------------------------------------------------------------
# (a) Coverage — the replay actually ran, over everything.
# ---------------------------------------------------------------------------


class TestReplayCoverage:
    """A replay that quietly stops replaying is the failure mode to fear here.

    Deliberately NO assertion on how many specimens repair or refuse (PRD G6):
    a better repairer must be free to move that split without going red.
    """

    def test_the_corpus_is_not_empty(self, corpus):
        assert corpus

    def test_every_specimen_was_driven_under_every_tier(self, corpus, replay):
        for policy, sweep in replay.items():
            driven = {obs.tool_use_id for obs in sweep.observations}
            missing = {record['tool_use_id'] for record in corpus} - driven
            assert not missing, (
                f'{policy.name} replayed {len(driven)} of {len(corpus)} specimens; '
                f'{len(missing)} never ran, e.g. {sorted(missing)[:5]}'
            )

    def test_skips_stay_under_the_pinned_ceiling(self, replay):
        for policy, sweep in replay.items():
            assert len(sweep.skips) <= MAX_SKIPPED, (
                f'{policy.name} skipped {len(sweep.skips)} specimen(s), over the '
                f'ceiling of {MAX_SKIPPED}. Raise MAX_SKIPPED only with the '
                f'reasons in the diff: {list(sweep.skips)[:5]}'
            )


# ---------------------------------------------------------------------------
# (a) The middleware's verdict agrees with the stored expectation.
# ---------------------------------------------------------------------------


class TestOutcomesAgreeWithTheCorpus:
    def test_every_outcome_is_in_the_declared_vocabulary(self, replay):
        """INV-2's three values, read from the same constant the facts use."""
        for observation in _all(replay):
            assert observation.outcome in OUTCOMES, observation.tool_use_id

    def test_outcomes_match_the_stored_expectation(self, replay):
        disagreements = [
            (obs.tool_use_id, obs.policy, expected, obs.outcome)
            for policy, sweep in replay.items()
            for obs in sweep.observations
            for expected in [_EXPECTED_OUTCOME[(obs.expected_outcome, policy)]]
            if obs.outcome != expected
        ]
        assert not disagreements, (
            f'{len(disagreements)} specimen(s) disagree with the committed '
            'expectation at the middleware layer. If the repairer legitimately '
            'improved, regenerate the corpus in the SAME commit (PRD G6 '
            f'anticipates exactly this). First few: {disagreements[:5]}'
        )

    def test_an_unrepairable_specimen_never_reaches_the_tool(self, replay):
        """Boundary row B5, at corpus scale, under BOTH tiers.

        FORWARD_REPAIR forwards a REPAIR. A tier that also forwarded unparsed
        residue would deliver a call the caller never made while permanently
        dropping whatever arguments hide inside it.
        """
        leaked = [
            (obs.tool_use_id, obs.policy)
            for obs in _all(replay)
            if obs.outcome == 'unrepairable' and obs.delivered
        ]
        assert not leaked, leaked[:5]

    def test_an_unrepairable_specimen_keeps_its_payload_verbatim(self, replay):
        """INV-7: the refusal is non-destructive, for every specimen.

        The escalation is the ONLY surviving copy of a payload whose author is,
        by construction, an agent that cannot re-emit it identically.
        """
        lost = [
            (obs.tool_use_id, obs.policy)
            for obs in _all(replay)
            if obs.outcome == 'unrepairable'
            and [record['raw_value'] for record in obs.escalations] != [obs.raw_value]
        ]
        assert not lost, lost[:5]

    def test_a_rejected_specimen_never_reaches_the_tool(self, replay):
        """"Write nothing" is the whole of REJECT_WITH_REPAIR's promise."""
        written = [
            obs.tool_use_id
            for obs in replay[RepairPolicy.REJECT_WITH_REPAIR].observations
            if obs.delivered
        ]
        assert not written, written[:5]

    def test_a_repaired_specimen_reaches_the_tool_exactly_once(self, replay):
        wrong = [
            (obs.tool_use_id, len(obs.delivered))
            for obs in replay[RepairPolicy.FORWARD_REPAIR].observations
            if obs.outcome == 'repaired' and len(obs.delivered) != 1
        ]
        assert not wrong, wrong[:5]

    def test_every_specimen_emits_exactly_one_fact(self, replay):
        """INV-2: no outcome is invisible to a consumer that never reads a log."""
        wrong = [
            (obs.tool_use_id, obs.policy, [fact.get('outcome') for fact in obs.facts])
            for obs in _all(replay)
            if [fact.get('outcome') for fact in obs.facts] != [obs.outcome]
        ]
        assert not wrong, wrong[:5]


# ---------------------------------------------------------------------------
# (b) The recovered parameters are the ones the corpus names — and they LAND.
# ---------------------------------------------------------------------------


class TestRecoveredParameters:
    def test_recovered_names_match_the_stored_expectation(self, replay):
        disagreements = [
            (obs.tool_use_id, obs.policy, list(obs.expected_recovered), list(obs.recovered_params))
            for obs in _all(replay)
            if obs.outcome in ('repaired', 'rejected')
            and obs.recovered_params != obs.expected_recovered
        ]
        assert not disagreements, disagreements[:5]

    def test_a_recovered_parameter_actually_lands_on_the_tool(self, replay):
        """The point of FORWARD_REPAIR: reported is not the same as delivered.

        A middleware that computed the recovery, announced it in ``meta`` and
        then forwarded the original arguments would pass every name assertion
        above while dropping exactly the data this tier exists to save.
        """
        dropped = [
            (obs.tool_use_id, name)
            for obs in replay[RepairPolicy.FORWARD_REPAIR].observations
            if obs.outcome == 'repaired'
            for name in obs.expected_recovered
            if not obs.delivered or obs.delivered[0].get(name) is None
        ]
        assert not dropped, dropped[:5]

    def test_the_repaired_call_is_the_complete_argument_map(self, replay):
        """A retry must be mechanical: resubmit ``repaired_call`` verbatim.

        A payload carrying only the repaired field would make the caller
        reassemble the rest by hand, which is how the swallowed arguments get
        lost for good.
        """
        incomplete = [
            (obs.tool_use_id, sorted(expected - set(obs.repaired_call or {})))
            for obs in replay[RepairPolicy.REJECT_WITH_REPAIR].observations
            if obs.outcome == 'rejected'
            for expected in [set(obs.supplied) | set(obs.expected_recovered)]
            if expected - set(obs.repaired_call or {})
        ]
        assert not incomplete, incomplete[:5]


# ---------------------------------------------------------------------------
# (c) THE LOAD-BEARING ONE — non-circular with respect to the corpus.
# ---------------------------------------------------------------------------


class TestNoStillPoisonedValueEverEscapes:
    """No specimen leaves this guard carrying markup the guard would reject.

    Re-derived from :func:`detect` against what actually crossed the boundary,
    so it holds even if every stored expectation is wrong. Both halves matter:
    a value forwarded still-poisoned would re-trip fused-memory's write-time
    tripwire downstream — the guard laundering a leak into a deeper failure —
    and a ``repaired_call`` still carrying one would send the caller into an
    infinite mechanical-retry loop against its own rejection.
    """

    def test_no_value_delivered_to_a_tool_trips_detect(self, replay):
        poisoned = [
            (obs.tool_use_id, obs.policy, name)
            for obs in _all(replay)
            for call in obs.delivered
            for name, value in call.items()
            if isinstance(value, str) and detect(value) is not None
        ]
        assert not poisoned, poisoned[:5]

    def test_no_repaired_call_offered_to_a_caller_trips_detect(self, replay):
        poisoned = [
            (obs.tool_use_id, obs.policy, name)
            for obs in _all(replay)
            for name, value in (obs.repaired_call or {}).items()
            if isinstance(value, str) and detect(value) is not None
        ]
        assert not poisoned, poisoned[:5]


# ---------------------------------------------------------------------------
# (d) Determinism.
# ---------------------------------------------------------------------------


class TestReplayIsDeterministic:
    def test_two_sweeps_in_one_process_are_identical(self, corpus, replay):
        """Same fixture, same process, same server class — same verdicts.

        Guards the accidental-state failures a middleware is exposed to that a
        pure function is not: a counter, a cache or a schema lookup that makes
        the second run of a specimen differ from the first.
        """
        for policy, first in replay.items():
            again = asyncio.run(_sweep(policy, corpus))
            assert _digest(again) == _digest(first), policy.name
