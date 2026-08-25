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
the two preserved records themselves.

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
import json
from pathlib import Path

import pytest
from shared.toolcall_markup import detect, repair

# The opening angle bracket, never typed as a literal. See the authoring rule.
LT = chr(0x3C)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "markup_specimens"

# Digests recorded in prerequisite pre-1 (commit 53a0f25839) against the SOURCE
# records under `data/escalations/archive/`, before any re-encoding.
#
# `parsed_sha256` is sha256(json.dumps(obj, sort_keys=True)) — the parsed-VALUE
# digest. It is what proves the escaping applied to the committed text
# is lossless: the file text changes, the parsed value does not.
SPECIMENS: dict[str, dict[str, object]] = {
    "esc-3514-1.json": {
        "source_path": (
            "data/escalations/archive/2026-08-03/esc-3514-1.json"
        ),
        "source_sha256": (
            "c0231182da2fab09e8ed2652688b646b5e7bcd77083a2461d2840748639070a7"
        ),
        "parsed_sha256": (
            "a4fbd90ff0371a88c711b82d63f3cbd6ba3bdcae217540cc74db486eb6023299"
        ),
        # The DIRECT producer record: the leaking session's own filing.
        "agent_role": "implementer",
        "detail_len": 2812,
        "suggested_action": "",
    },
    "esc-3514-3.json": {
        "source_path": (
            "data/escalations/archive/2026-08-08/esc-3514-3.json"
        ),
        "source_sha256": (
            "cab14ac7e97f097fa5a3e53fe835d63be1ee513cc4c685131656f320c4581664"
        ),
        "parsed_sha256": (
            "aeaeb3dca1a345ebfe0afd3e9d97db51d760f754cd2100564a8ac022f6429c56"
        ),
        # The reaper's RE-FILING, which propagated the same corrupted `detail`.
        "agent_role": "harness-orphan-reaper",
        "detail_len": 2873,
        # Not empty here but a bare default — the same loss, differently shaped.
        "suggested_action": "manual_intervention",
    },
}

SPECIMEN_IDS = sorted(SPECIMENS)


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


@pytest.mark.parametrize("name", SPECIMEN_IDS)
def test_specimen_parses_and_matches_the_preserved_parsed_digest(name: str) -> None:
    """The committed fixture is value-identical to the record pre-1 preserved.

    This is the losslessness pin for the `\\u003c` re-encoding: the file TEXT is
    rewritten, so its own sha256 changes, but the PARSED value must not move.
    A failure here means the re-emission dropped or altered evidence.
    """
    path = FIXTURE_DIR / name
    assert path.is_file(), f"specimen {name} is missing from {FIXTURE_DIR}"

    obj = _load(name)
    assert isinstance(obj, dict)

    assert _parsed_digest(obj) == SPECIMENS[name]["parsed_sha256"], (
        f"{name}: parsed-value digest moved — the re-encoding was NOT lossless"
    )


@pytest.mark.parametrize("name", SPECIMEN_IDS)
def test_committed_text_is_escaped_while_the_parsed_value_is_not(name: str) -> None:
    """THE ESCAPING INVARIANT.

    The committed file text carries no literal opening angle bracket, while the
    parsed `detail` demonstrably still does. That is the convention
    `shared/tests/fixtures/toolcall_markup_corpus.README.md` establishes: the
    escaped form is standard JSON so nothing is lost, and it is what makes the
    fixture safe for a future agent to hand-edit (PRD G6 anticipates exactly
    that edit when the repairer improves).
    """
    text = _text(name)
    assert LT not in text, (
        f"{name}: committed text contains a literal opening bracket; it must be "
        f"escaped as its JSON \\u003c form"
    )

    detail = _load(name)["detail"]
    assert LT in detail, (
        f"{name}: parsed detail no longer carries envelope literals — the "
        f"corruption IS the payload and must survive re-encoding"
    )


def test_readme_documents_the_specimens() -> None:
    """A fixture with no provenance record is a fixture a later pass deletes."""
    readme = FIXTURE_DIR / "README.md"
    assert readme.is_file(), f"{readme} is missing"
    assert readme.read_text(encoding="utf-8").strip(), f"{readme} is empty"


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
@pytest.mark.asyncio
async def test_the_landed_detector_fires_on_the_specimen(name: str) -> None:
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


@pytest.mark.parametrize("name", SPECIMEN_IDS)
def test_the_corruption_signature_is_the_discriminating_pair(name: str) -> None:
    """(d) THE SIGNATURE — stated so a future sweep does not repeat a known error.

    The discriminating pair is `detect()` firing on `detail` PLUS a
    `suggested_action` that is empty or a bare default WHILE its real text sits
    inside `detail`.

    An empty `evidence` list is deliberately NOT treated as a signal on its
    own. Sibling record `esc-3514-2` (same task, `agent_role=orchestrator`) is
    clean — zero matching markup patterns, `suggested_action` intact — and
    stores `evidence == []` too. Most escalations simply never pass evidence.
    It is asserted here only in CONJUNCTION with the entries being legible
    inside `detail`, which the control record has no counterpart for.
    """
    obj = _load(name)
    detail = obj["detail"]

    assert len(detail) == SPECIMENS[name]["detail_len"]
    assert obj["agent_role"] == SPECIMENS[name]["agent_role"]

    stored_action = obj["suggested_action"]
    assert stored_action == SPECIMENS[name]["suggested_action"]
    assert stored_action in BARE_SUGGESTED_ACTIONS
    assert REAL_SUGGESTED_ACTION_PREFIX in detail, (
        f"{name}: the real suggested_action is no longer legible in detail, so "
        f"the loss is no longer demonstrable from the record alone"
    )

    assert obj["evidence"] == []
    assert detail.count(OBSERVATION_KEY) == EVIDENCE_ENTRY_COUNT
    assert EVIDENCE_HEAD_PIN in detail
