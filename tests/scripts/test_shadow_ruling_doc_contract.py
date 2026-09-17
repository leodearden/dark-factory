"""Contract guards binding the shadow-ruling documents to the live module.

Task 5374. ``skills/escalation-watcher/SKILL.md`` shows the L2 session an
``x_shadow_ruling:`` payload to stamp, and ``docs/escalation-standing-policy.md``
enumerates the slug vocabularies that payload draws on. Both are hand-written
prose next to a real parser. A literal the skill presents but the parser rejects
costs a sample of the very measurement it exists to feed — silently, because a
rejected stamp is indistinguishable from a record nobody stamped.

SCOPE — STRUCTURE ONLY, NEVER PROSE. Adopting the task-4095 precedent by name
(``tests/scripts/test_unblock_skill_done_provenance_templates.py``): every word
of guidance in both documents stays free to be rewritten, reordered or deleted.
What is held is only that a payload literal the skill presents is one the parser
accepts, and that a slug the policy document enumerates is one the code knows.
Nothing about the vocabularies is hand-copied here — they are read off the same
frozensets the codec validates against.

COUNTER-EXAMPLES. The skill shows a REJECTED payload next to a good one, so a
reader can recognise the shape that will be thrown away. Put
``<!-- shadow-guard: negative -->`` on the same line as the fence opener to mark
one: the extractor sorts it into the negative set, where it is required to FAIL
the parser. An unmarked counter-example fails this guard, as it should.

MUST NOT SKIP. No ``pytest.importorskip``, no try/except-and-skip. A missing
SKILL.md, a missing policy document or an unimportable
``escalation.shadow_ruling`` must FAIL here. A guard that skips reports green on
exactly the breakage it was written to catch.
"""
from __future__ import annotations

import json
import pathlib
import re

import pytest
from escalation.shadow_ruling import (
    FIRST_TRANCHE_CLASSES,
    HUMAN_FOREVER_GATES,
    REVERSIBLE_ACTIONS,
    SHADOW_RULING_MARKER,
    ShadowRuling,
    parse_shadow_ruling,
)

# Resolved by walking up from THIS file, never from cwd: the suite is run from
# the repo root by its module config and from `tests/scripts/` by hand.
REPO_ROOT = pathlib.Path(__file__).parents[2]

SKILL = REPO_ROOT / "skills" / "escalation-watcher" / "SKILL.md"
POLICY = REPO_ROOT / "docs" / "escalation-standing-policy.md"

#: Marks a fenced literal the skill shows in order to be REJECTED.
_NEGATIVE_MARKER = "<!-- shadow-guard: negative -->"

#: The policy document's slug lists, keyed by the exact `## ` heading each sits
#: under, and paired with the frozenset each must equal. Both directions are
#: checked, so neither a documentation addition nor a code addition can drift
#: alone.
_SLUG_SECTIONS: tuple[tuple[str, frozenset[str]], ...] = (
    ("Reversible actions", REVERSIBLE_ACTIONS),
    ("Human-forever gates", HUMAN_FOREVER_GATES),
    ("First-tranche candidate classes", FIRST_TRANCHE_CLASSES),
)

#: A slug bullet: a list item that OPENS with a backticked lowercase token.
#: Leading-position only, so the surrounding prose in those sections may keep
#: backticking whatever it likes without joining the vocabulary.
_SLUG_BULLET = re.compile(r"^- `([a-z0-9_]+)`")

#: The heading of the skill's shadow-mode section. Its content is read from here
#: to the next top-level heading.
_SHADOW_SECTION = "## Shadow-mode standing-policy rulings (measurement only)"


def _read(path: pathlib.Path) -> str:
    assert path.is_file(), (
        f"{path.relative_to(REPO_ROOT)} is missing. This guard binds it to "
        f"escalation.shadow_ruling and must FAIL rather than skip when it is gone."
    )
    return path.read_text()


def _section(text: str, heading: str, level: str = "## ") -> str:
    """The body under *heading*, up to the next heading at the same level."""
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == heading.strip()]
    assert len(starts) == 1, f"expected exactly one {heading!r} heading, found {len(starts)}"
    start = starts[0] + 1
    end = next(
        (i for i in range(start, len(lines)) if lines[i].startswith(level)),
        len(lines),
    )
    return "\n".join(lines[start:end])


def _fenced_payload_literals(text: str) -> tuple[list[str], list[str]]:
    """Split fenced JSON literals in *text* into (positive, negative).

    A literal qualifies when the marker token appears on the fence opener or
    inside the block — that is what makes it a payload the skill is PRESENTING
    rather than an unrelated JSON example.
    """
    positive: list[str] = []
    negative: list[str] = []
    for opener, body in re.findall(r"^(```[^\n]*)\n(.*?)^```", text, re.DOTALL | re.MULTILINE):
        if SHADOW_RULING_MARKER not in opener and SHADOW_RULING_MARKER not in body:
            continue
        (negative if _NEGATIVE_MARKER in opener else positive).append(body)
    return positive, negative


def _marker_lines(block: str) -> list[str]:
    return [line.strip() for line in block.splitlines() if line.strip().startswith(
        SHADOW_RULING_MARKER
    )]


# ---------------------------------------------------------------------------
# The skill's payload literals, against the live parser
# ---------------------------------------------------------------------------


def test_every_presented_payload_literal_parses_into_a_valid_ruling():
    positive, _ = _fenced_payload_literals(_read(SKILL))
    for block in positive:
        lines = _marker_lines(block)
        assert lines, (
            f"a fenced literal mentions {SHADOW_RULING_MARKER} but carries no line "
            f"beginning with it, so a reader copying it would stamp something the "
            f"parser cannot see:\n{block}"
        )
        for line in lines:
            ruling = parse_shadow_ruling(line)
            assert isinstance(ruling, ShadowRuling), (
                f"the skill presents a payload the live parser rejects — a reader "
                f"following it would lose the sample silently:\n{line}"
            )


def test_at_least_one_payload_literal_was_extracted():
    """Non-vacuity: an extractor that silently matches nothing is worse than no
    extractor, because it reports green forever."""
    positive, _ = _fenced_payload_literals(_read(SKILL))
    assert positive, (
        f"no fenced {SHADOW_RULING_MARKER} literal found in "
        f"{SKILL.relative_to(REPO_ROOT)} — either the section was removed or the "
        f"extractor stopped matching it."
    )


def test_at_least_one_counter_example_is_really_rejected():
    """A counter-example that silently validates is worse than none: it teaches
    the reader that a broken shape is fine."""
    _, negative = _fenced_payload_literals(_read(SKILL))
    assert negative, (
        f"the skill marks no {_NEGATIVE_MARKER} literal, so nothing shows a reader "
        f"the shape that gets thrown away."
    )
    for block in negative:
        lines = _marker_lines(block)
        assert lines, f"a negative literal carries no marker line:\n{block}"
        for line in lines:
            assert parse_shadow_ruling(line) is None, (
                f"this literal is marked as a counter-example but the parser ACCEPTS "
                f"it:\n{line}"
            )


def test_payload_literals_are_json_objects_carrying_the_documented_keys():
    """The skill's fenced literals must be readable as JSON by a human too —
    a payload that only the parser's leniency saves is not a good example."""
    positive, _ = _fenced_payload_literals(_read(SKILL))
    for block in positive:
        for line in _marker_lines(block):
            payload = json.loads(line[len(SHADOW_RULING_MARKER):])
            assert set(payload) == {"class", "proposed_action", "evidence", "confidence"}


# ---------------------------------------------------------------------------
# The policy document's slug vocabularies, against the live frozensets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("heading", "expected"), _SLUG_SECTIONS, ids=lambda v: str(v)[:40])
def test_documented_slugs_equal_the_live_vocabulary(heading: str, expected: frozenset[str]):
    body = _section(_read(POLICY), f"## {heading}")
    documented = {m.group(1) for line in body.splitlines() if (m := _SLUG_BULLET.match(line))}

    assert documented == expected, (
        f"under '## {heading}', the policy document and escalation.shadow_ruling "
        f"have drifted.\n"
        f"  documented but unknown to the code: {sorted(documented - expected)}\n"
        f"  known to the code but undocumented: {sorted(expected - documented)}"
    )


# ---------------------------------------------------------------------------
# The skill's shadow section points at things that exist
# ---------------------------------------------------------------------------


def test_the_shadow_section_names_the_policy_document_and_it_exists():
    section = _section(_read(SKILL), _SHADOW_SECTION)
    relative = str(POLICY.relative_to(REPO_ROOT))
    assert relative in section, (
        f"the shadow section must name {relative} — it is the authority for the "
        f"classes it tells the session to stamp."
    )
    assert POLICY.is_file()


def test_the_shadow_section_quotes_a_runnable_weekly_count_command():
    section = _section(_read(SKILL), _SHADOW_SECTION)
    assert "python -m escalation.shadow_ruling" in section, (
        "the shadow section must quote the weekly-count command, or the "
        "measurement half has no operator."
    )
    assert "--queue-dir" in section

    # The module path inside the quoted command is real: importing this test
    # module already proved it, and this asserts the SPELLING in the doc is the
    # one that imports rather than a plausible neighbour.
    import escalation.shadow_ruling as live

    assert live.__name__ == "escalation.shadow_ruling"
