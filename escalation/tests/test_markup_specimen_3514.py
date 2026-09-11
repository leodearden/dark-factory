"""The esc-3514 markup specimens: fixture integrity + the landed-containment verdict (task 3643).

Task 3643 is a VERIFICATION task. It asks whether task 3083 covered the
`claude-task-3514-implementer` envelope-markup leak, and whether the escalation
write boundary is guarded now. The measured answer, recorded in full in
`docs/escalation-markup-write-boundary.md`, is:

  * the memory-write path was covered and rejected all three `add_memory`
    calls (3141's tripwire), so nothing entered the corpus;
  * the leak nevertheless LANDED via `escalate_info`, a boundary 3083's
    Mem0/Graphiti-scoped tooling structurally could not reach;
  * that boundary IS guarded now (task 3690 registers `MarkupGuardMiddleware`
    on the escalation server) — so DETECTION is covered;
  * but RECOVERY is not, for this specimen class. `repair()` returns None for
    both records.

This module makes those claims executable rather than prose, replayed against
the two preserved records themselves — plus their clean sibling `esc-3514-2`,
preserved as a NEGATIVE control so that the one predicate a future sweep must
not use (`evidence == []`, which the clean record shares) fails a test rather
than merely contradicting a paragraph.

AUTHORING RULE, mandatory in this file — the same rule
`shared/tests/fixtures/toolcall_markup_corpus.README.md` rule 1 states.
NEVER TYPE A RAW ENVELOPE SENTINEL HERE. Every sentinel is built from
`chr(0x3C)` or a `\x3c` escape. Writing the raw literal would put it inside
this file's own authoring tool call, which reproduces the exact defect under
test: the harness parser over-consumes at the literal, truncates the argument
and silently drops every sibling argument of the same call.

The fixtures are located via `Path(__file__).parent`, never by reaching into
the live `data/escalations/` tree — that tree is gitignored, and both records
have already been moved once by the archiver.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import tempfile
from pathlib import Path

import pytest
from fastmcp.tools.function_tool import FunctionTool
from shared.toolcall_markup import detect, repair

from escalation.queue import EscalationQueue
from escalation.server import create_server

# The opening angle bracket, never typed as a literal. See the authoring rule.
LT = chr(0x3C)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "markup_specimens"

# `parsed_sha256` is sha256(json.dumps(obj, sort_keys=True)) — the parsed-VALUE
# digest, recorded in prerequisite pre-1 (commit 53a0f25839) against each SOURCE
# record under `data/escalations/archive/` before any re-encoding, and measured
# the same way for the control when it was preserved. It is what proves the
# escaping applied to the committed text is lossless: the file text changes, the
# parsed value does not.
#
# The source PATHS and source-byte digests are deliberately not repeated here.
# The fixture README's provenance table is their one home; a pin in code that no
# assertion reads is exactly what a later reader mistakes for a live guarantee.
# This module pins the parsed value, never the source bytes.
SPECIMENS: dict[str, dict[str, object]] = {
    "esc-3514-1.json": {
        "parsed_sha256": (
            "a4fbd90ff0371a88c711b82d63f3cbd6ba3bdcae217540cc74db486eb6023299"
        ),
        # The DIRECT producer record: the leaking session's own filing.
        "agent_role": "implementer",
        "detail_len": 2812,
        "suggested_action": "",
        "detail_carries_envelope_literals": True,
    },
    "esc-3514-3.json": {
        "parsed_sha256": (
            "aeaeb3dca1a345ebfe0afd3e9d97db51d760f754cd2100564a8ac022f6429c56"
        ),
        # The reaper's RE-FILING, which propagated the same corrupted `detail`.
        "agent_role": "harness-orphan-reaper",
        "detail_len": 2873,
        # Not empty here but a bare default — the same loss, differently shaped.
        "suggested_action": "manual_intervention",
        "detail_carries_envelope_literals": True,
    },
}

# THE NEGATIVE CONTROL, preserved for exactly the reason the two specimens were:
# it lived only in the gitignored `data/` tree, and the caveat it anchors — that
# an empty `evidence` list is NOT a corruption signal — was prose until the
# record itself was committed. Same task 3514, a different producer, and clean.
CONTROL_ID = "esc-3514-2.json"

CONTROL: dict[str, object] = {
    "parsed_sha256": (
        "0f8319e76220e3293fbdd10b1d6e820d0bbf2216507fdca5d2834b31db3533bd"
    ),
    "agent_role": "orchestrator",
    "detail_len": 3481,
    "suggested_action": "await_preexisting_main_hotfix",
    # Its `summary` does carry one literal opening bracket (the repr of an enum
    # member), so the committed text is still non-trivially escaped — but its
    # `detail`, the field the detector reads, carries none. That is the control.
    "detail_carries_envelope_literals": False,
}

FIXTURES: dict[str, dict[str, object]] = {**SPECIMENS, CONTROL_ID: CONTROL}

SPECIMEN_IDS = sorted(SPECIMENS)
FIXTURE_IDS = sorted(FIXTURES)


def _text(name: str) -> str:
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


def _load(name: str) -> dict:
    return json.loads(_text(name))


def _parsed_digest(obj: object) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Fixture integrity.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", FIXTURE_IDS)
def test_specimen_parses_and_matches_the_preserved_parsed_digest(name: str) -> None:
    """The committed fixture is value-identical to the record that was preserved.

    This is the losslessness pin for the `\\u003c` re-encoding: the file TEXT is
    rewritten, so its own sha256 changes, but the PARSED value must not move.
    A failure here means the re-emission dropped or altered evidence.
    """
    path = FIXTURE_DIR / name
    assert path.is_file(), f"specimen {name} is missing from {FIXTURE_DIR}"

    obj = _load(name)
    assert isinstance(obj, dict)

    assert _parsed_digest(obj) == FIXTURES[name]["parsed_sha256"], (
        f"{name}: parsed-value digest moved — the re-encoding was NOT lossless"
    )


@pytest.mark.parametrize("name", FIXTURE_IDS)
def test_committed_text_is_escaped_while_the_parsed_value_is_not(name: str) -> None:
    """THE ESCAPING INVARIANT.

    The committed file text carries no literal opening angle bracket, while the
    parsed value demonstrably still carries whatever it carried. That is the
    convention `shared/tests/fixtures/toolcall_markup_corpus.README.md`
    establishes: the escaped form is standard JSON so nothing is lost, and it is
    what makes the fixture safe for a future agent to hand-edit (PRD G6
    anticipates exactly that edit when the repairer improves).

    The second assertion is direction-aware. For the two specimens the literals
    in `detail` ARE the payload and must survive re-encoding; for the control
    their ABSENCE is the payload, since a control whose detail acquired markup
    would stop discriminating anything.
    """
    text = _text(name)
    assert LT not in text, (
        f"{name}: committed text contains a literal opening bracket; it must be "
        f"escaped as its JSON \\u003c form"
    )

    detail = _load(name)["detail"]
    expected = FIXTURES[name]["detail_carries_envelope_literals"]
    assert (LT in detail) == expected, (
        f"{name}: parsed detail carries envelope literals = {LT in detail}, "
        f"expected {expected}. For a specimen the corruption IS the payload and "
        f"must survive re-encoding; for the control the clean detail is."
    )


def test_readme_documents_the_specimens() -> None:
    """A fixture with no provenance record is a fixture a later pass deletes."""
    readme = FIXTURE_DIR / "README.md"
    assert readme.is_file(), f"{readme} is missing"
    assert readme.read_text(encoding="utf-8").strip(), f"{readme} is empty"


# ---------------------------------------------------------------------------
# The replayed call shape, DERIVED from the live tool.
# ---------------------------------------------------------------------------


_CALL_SHAPE_CACHE: tuple[frozenset[str], frozenset[str]] | None = None


async def _escalate_info_call_shape() -> tuple[frozenset[str], frozenset[str]]:
    """Return `(schema_params, supplied)` for replaying a specimen through `repair`.

    MEMOISED. The derivation stands up a temporary directory, a queue and a full
    server, and it is called once per parametrized invocation of two tests; the
    signature it reads cannot differ between them, so repeating the construction
    bought nothing. The cache is per-process, like the signature it holds.

    DERIVED, never hardcoded. A hardcoded parameter list would keep passing
    while modelling a call shape that no longer exists — `repair()` consults
    `schema_params` to decide whether a recovered name is real, so a silent
    drift there would weaken every assertion in this module without failing
    anything. Reading the names off the live tool means a rename fails loudly,
    here, with a message that says so.

    The server is built the established way — see
    `escalation/tests/test_markup_middleware_registration.py` — with the
    startup sweep off. Nothing is CALLED through it: this reads
    `escalate_info`'s signature only, so the middleware-bypass warning that
    file carries does not apply.

    `supplied` models the argument NAMES the corrupted call actually arrived
    with: the tool's required parameters plus `detail`, and DELIBERATELY NOT
    `suggested_action` or `evidence`. That omission is the whole point of the
    specimen — the harness dropped them from the arguments map, which is why
    they ended up absorbed into `detail` instead. It also matters mechanically:
    `repair()` refuses any candidate whose recovered names intersect
    `supplied`, so listing them here would return None for entirely the wrong
    reason and make the unrepairable assertion vacuous.
    """
    global _CALL_SHAPE_CACHE
    if _CALL_SHAPE_CACHE is not None:
        return _CALL_SHAPE_CACHE

    with tempfile.TemporaryDirectory() as tmp:
        server = create_server(
            EscalationQueue(Path(tmp) / "esc"), startup_sweep=False
        )
        tool = await server.get_tool("escalate_info")
        # `get_tool` is typed `Tool | None`, and the base `Tool` declares no
        # `.fn` — only the `FunctionTool` subclass does. Narrow before the
        # standard `tool.fn` unit-test read, the convention established by
        # `orchestrator/tests/test_workflow_e2e.py::test_steward_l1_reescalation_survives_the_wip_l0_sweep`.
        # The assertion is load-bearing beyond the type checker: if
        # `escalate_info` ever stops being registered, or is re-registered as
        # a non-function tool, the derivation below would otherwise fail with
        # an opaque AttributeError instead of naming the cause.
        assert isinstance(tool, FunctionTool), (
            "escalate_info is not registered on the escalation server as a "
            "FunctionTool; the replayed call shape cannot be derived"
        )
        parameters = inspect.signature(tool.fn).parameters

    schema_params = frozenset(parameters)
    required = frozenset(
        name
        for name, p in parameters.items()
        if p.default is inspect.Parameter.empty
    )

    # A rename must fail HERE, loudly, rather than quietly weakening the
    # specimen assertions below.
    missing = {"detail", "suggested_action", "evidence"} - schema_params
    assert not missing, (
        f"escalate_info no longer declares {sorted(missing)}; the replayed "
        f"call shape no longer models the call these specimens came from"
    )
    assert required, "escalate_info declares no required parameters"

    supplied = required | {"detail"}
    # The docstring's warning, made mechanical rather than left to a reader.
    # If `escalate_info` ever gives `suggested_action` or `evidence` a required
    # (default-less) parameter, it enters `required` and therefore `supplied`,
    # and repair() would start returning None because the recovered names
    # collide with a supplied one — so the unrepairable pin would keep passing
    # for a reason with nothing to do with the recorded verdict. Fail here,
    # where the cause can be named, instead of there.
    assert {"suggested_action", "evidence"}.isdisjoint(supplied), (
        "escalate_info now requires the swallowed params; the replayed call "
        "shape no longer models the specimen"
    )

    _CALL_SHAPE_CACHE = (schema_params, supplied)
    return _CALL_SHAPE_CACHE


# ---------------------------------------------------------------------------
# The landed-containment verdict.
# ---------------------------------------------------------------------------
#
# These four tests are this task's ANSWER made executable. They replay the two
# real records through the detector and repairer that actually shipped
# (`shared.toolcall_markup`, tasks 3688/3689), never through a local
# reimplementation — the verdict is a property of the landed code, which is the
# whole point of the exercise.

# The envelope literal the report was QUOTING: the `matched_pattern` the memory
# tripwire handed back to the leaking session, reproduced faithfully in its
# evidence. Built from chr(0x3C), never typed. It is the sole thing blocking
# recovery — see the scrub control below.
QUOTED_MATCHED_PATTERN = LT + "/content>"
INERT_PLACEHOLDER = "[content-closer]"

# The values the harness dropped from the arguments map, still legible as inert
# text in the tail of `detail`.
REAL_SUGGESTED_ACTION_PREFIX = "Attach these observations to DF task 3083"
EVIDENCE_HEAD_PIN = "HEAD=860abb2210110deec67355c12b235b8b38f50c77"
EVIDENCE_ENTRY_COUNT = 3
OBSERVATION_KEY = chr(34) + "observation" + chr(34)

# `suggested_action` values that carry no information. esc-3514-3 is why the
# predicate cannot be "empty": its stored value is `manual_intervention`, a
# plausible-looking default — the same loss wearing a disguise.
BARE_SUGGESTED_ACTIONS = {"", "manual_intervention"}


@pytest.mark.parametrize("name", SPECIMEN_IDS)
def test_the_landed_detector_fires_on_the_specimen(name: str) -> None:
    """(a) DETECTED — the shipped detector covers this payload.

    Task 3690 registers `MarkupGuardMiddleware` on the escalation server
    (`escalation/src/escalation/server.py`, `RepairPolicy.FORWARD_REPAIR`,
    `exempt_tools=frozenset()`), and exemptions match bare in-server tool
    names, so `escalate_info` is intercepted. This asserts the half of the
    guard that would fire: the value is recognised as leaked markup.
    """
    detail = _load(name)["detail"]
    assert detect(detail) is not None, (
        f"{name}: the landed detector no longer recognises this specimen"
    )


@pytest.mark.parametrize("name", SPECIMEN_IDS)
@pytest.mark.asyncio
async def test_the_specimen_is_unrepairable_as_stored(name: str) -> None:
    """(b) THE VERDICT PIN — recovery does NOT happen for this specimen class.

    Task 3643's own description anticipated that under FORWARD_REPAIR "a
    corrupted escalate_info LANDS with its suggested_action recovered instead
    of being lost". Measured against the real records that is FALSE: `repair()`
    returns None, which routes to `_refuse_unrepairable`, so the escalation
    does not file under its own task id at all and the payload survives only as
    a separate critical L2 residue record.

    IF THIS ASSERTION EVER FAILS, that is the intended signal, not a bug: a
    future repairer has improved past this shape. Revisit — and update —
    `docs/escalation-markup-write-boundary.md`, which records this verdict as
    the answer to the task's question. Do not delete the test.
    """
    schema_params, supplied = await _escalate_info_call_shape()
    detail = _load(name)["detail"]

    assert repair(detail, "detail", schema_params, supplied) is None, (
        f"{name}: repair() now SUCCEEDS where it measurably returned None. "
        f"The recorded verdict in docs/escalation-markup-write-boundary.md is "
        f"stale and must be revisited."
    )


@pytest.mark.parametrize("name", SPECIMEN_IDS)
@pytest.mark.asyncio
async def test_the_quoted_pattern_is_the_sole_blocker(name: str) -> None:
    """(c) THE SCRUB CONTROL — what makes (b) a finding rather than a smoke test.

    `repair()` rejects any candidate whose parsed tail contains a second
    mis-close. The swallowed `evidence` value here QUOTES an envelope literal,
    because the escalation was *reporting a markup leak* and faithfully
    reproduced the `matched_pattern` that tripped the tripwire. Replacing ONLY
    that one quoted literal with an inert placeholder — every other byte
    untouched, asserted below — flips `repair()` from None to recovering both
    swallowed siblings.

    Together (b) and (c) prove the quote is the ONLY blocker, which is the
    generalisable finding: an escalation REPORTING a markup leak is the one
    payload class the repairer structurally cannot recover, because a faithful
    report quotes the pattern. This independently confirms the "doubly
    corrupted" PRD boundary row B5 shape that
    `test_markup_middleware_registration.py` describes for `esc-3184-2`.
    """
    schema_params, supplied = await _escalate_info_call_shape()
    detail = _load(name)["detail"]

    assert detail.count(QUOTED_MATCHED_PATTERN) == 1, (
        f"{name}: expected exactly one quoted matched_pattern to scrub"
    )
    assert INERT_PLACEHOLDER not in detail

    scrubbed = detail.replace(QUOTED_MATCHED_PATTERN, INERT_PLACEHOLDER)
    # The control is only valid if nothing else moved.
    assert scrubbed.replace(INERT_PLACEHOLDER, QUOTED_MATCHED_PATTERN) == detail

    result = repair(scrubbed, "detail", schema_params, supplied)
    assert result is not None, (
        f"{name}: scrubbing the quoted pattern did NOT make the specimen "
        f"repairable, so something other than the quote is blocking recovery"
    )
    assert set(result.recovered) == {"evidence", "suggested_action"}

    recovered_action = result.recovered["suggested_action"]
    assert recovered_action.startswith(REAL_SUGGESTED_ACTION_PREFIX)

    recovered_evidence = result.recovered["evidence"]
    assert EVIDENCE_HEAD_PIN in recovered_evidence
    # Not `json.loads(recovered_evidence)`: esc-3514-3's reaper note trails the
    # JSON list, so the recovered string is not loadable whole. Measured.
    assert recovered_evidence.count(OBSERVATION_KEY) == EVIDENCE_ENTRY_COUNT

    # repair()'s no-silent-partial-repair contract, on this payload.
    assert detect(result.clean_value) is None


def _empty_evidence(record: dict) -> bool:
    """The NON-discriminating half of the caveat, as code rather than prose.

    Fires on the clean control too — which is the whole point, and is asserted
    in `test_the_control_record_is_clean_yet_shares_the_empty_evidence_list`.
    """
    return record["evidence"] == []


def _swallowed_sibling_signature(record: dict) -> bool:
    """The DISCRIMINATING pair from the fixture README, as code.

    (1) `detect()` fires on `detail`; AND (2) `suggested_action` is empty or a
    bare default WHILE the argument it lost is still legible inside `detail` —
    its own parameter NAME is there, absorbed along with the rest of the
    envelope. A clean record has neither leg.

    Task 3691's sweep wants this predicate, not the empty-evidence list. It is
    kept executable here so that "the empty list does not discriminate" is a
    test rather than a paragraph a sweep author may not read.
    """
    detail = record["detail"]
    return (
        detect(detail) is not None
        and record["suggested_action"] in BARE_SUGGESTED_ACTIONS
        and "suggested_action" in detail
    )


@pytest.mark.parametrize("name", SPECIMEN_IDS)
def test_the_corruption_signature_is_the_discriminating_pair(name: str) -> None:
    """(d) THE SIGNATURE — stated so a future sweep does not repeat a known error.

    The discriminating pair is `detect()` firing on `detail` PLUS a
    `suggested_action` that is empty or a bare default WHILE its real text sits
    inside `detail`.

    An empty `evidence` list is deliberately NOT treated as a signal on its
    own — the clean control record stores one too. That asymmetry is asserted
    directly in the control test below; here `evidence == []` is claimed only
    in CONJUNCTION with the entries being legible inside `detail`.
    """
    obj = _load(name)
    detail = obj["detail"]

    assert _swallowed_sibling_signature(obj), (
        f"{name}: the discriminating pair no longer holds on this record"
    )

    assert len(detail) == SPECIMENS[name]["detail_len"]
    assert obj["agent_role"] == SPECIMENS[name]["agent_role"]

    stored_action = obj["suggested_action"]
    assert stored_action == SPECIMENS[name]["suggested_action"]
    assert REAL_SUGGESTED_ACTION_PREFIX in detail, (
        f"{name}: the real suggested_action is no longer legible in detail, so "
        f"the loss is no longer demonstrable from the record alone"
    )

    assert _empty_evidence(obj)
    assert detail.count(OBSERVATION_KEY) == EVIDENCE_ENTRY_COUNT
    assert EVIDENCE_HEAD_PIN in detail


def test_the_control_record_is_clean_yet_shares_the_empty_evidence_list() -> None:
    """(e) THE CONTROL — the caveat as an executable pair, not a paragraph.

    `esc-3514-2` is the same task's sibling filing by a different producer
    (`agent_role=orchestrator`), and it is CLEAN: the landed detector does not
    fire on its `detail`, and its `suggested_action` of
    `await_preexisting_main_hotfix` is intact and informative. It nevertheless
    stores `evidence == []`, exactly as both corrupted records do — most
    escalations simply never pass evidence.

    A sweep keyed on the empty list alone would therefore fire on clean records.
    That warning was prose in three places and enforced nowhere, while the
    control itself lived only in the gitignored `data/` tree — as perishable as
    the two records this directory exists to rescue, and `esc-3514-3` had
    already been archived out from under an earlier plan. Preserving it and
    asserting both halves here is what makes the claim load-bearing: the naive
    predicate fires on all three records, the discriminating pair on the two
    corrupted ones only. Task 3691 is the sweep that needs the difference.
    """
    control = _load(CONTROL_ID)
    specimens = [_load(name) for name in SPECIMEN_IDS]

    assert control["agent_role"] == CONTROL["agent_role"]
    assert len(control["detail"]) == CONTROL["detail_len"]
    assert detect(control["detail"]) is None, (
        "the control record is no longer clean, so it discriminates nothing"
    )
    assert control["suggested_action"] == CONTROL["suggested_action"]
    assert control["suggested_action"] not in BARE_SUGGESTED_ACTIONS

    # The NON-discriminating half: true of the clean record and both corrupted.
    assert _empty_evidence(control), (
        "the control no longer stores an empty evidence list — it can no longer "
        "demonstrate that the empty list is not a corruption signal"
    )
    assert all(_empty_evidence(record) for record in specimens)

    # The DISCRIMINATING pair: false of the clean record, true of both corrupted.
    assert not _swallowed_sibling_signature(control), (
        "the discriminating pair now fires on a CLEAN record; a sweep built on "
        "it would produce false positives"
    )
    assert all(_swallowed_sibling_signature(record) for record in specimens)
