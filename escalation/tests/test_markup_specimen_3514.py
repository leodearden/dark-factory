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
