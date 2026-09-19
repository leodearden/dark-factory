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

import importlib
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
    main,
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

#: The arguments the skill's quoted weekly-count command passes, captured from
#: after the module spelling to the end of the (possibly `\`-continued) command.
#: Read off the doc and driven through the live `main()`, so the flag vocabulary
#: is checked in BOTH directions rather than pinned as a substring.
_QUOTED_ARGV = re.compile(
    r"python -m escalation\.shadow_ruling(?P<args>(?:[ \t]|\\\n|[-\w<>/.=:]+)*)"
)

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
    """Every line the block PRESENTS as a stamp, returned VERBATIM.

    Recognised by the stripped form, so an indented literal is still SEEN —
    but returned unstripped, so the live parser judges exactly the bytes a
    reader would copy. `parse_shadow_ruling` requires the marker at index 0
    (`line.startswith(...)`, no strip), so stripping here would normalise away
    the one difference that matters: an `x_shadow_ruling:` literal nested
    inside a list item or a numbered step would pass this guard and still be
    rejected in production, which is precisely the equivalence the guard
    exists to hold.
    """
    return [
        line for line in block.splitlines()
        if line.strip().startswith(SHADOW_RULING_MARKER)
    ]


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


def test_an_indented_literal_is_caught_rather_than_normalised_away():
    """Non-vacuity for `_marker_lines` returning the line VERBATIM.

    The guard's whole claim is that a literal the skill presents is one the
    parser accepts. Indentation is the one transformation that breaks that
    equivalence without changing a single payload byte, so this drives a block
    the extractor would have passed when it stripped: the line is still SEEN
    (a silently-empty extraction would be the worse failure) and the live
    parser still REJECTS it.
    """
    indented = "    " + ShadowRuling(
        ruling_class=sorted(FIRST_TRANCHE_CLASSES)[0],
        proposed_action=sorted(REVERSIBLE_ACTIONS)[0],
        evidence="probe output quoted verbatim",
        confidence=0.9,
    ).to_note_line()

    lines = _marker_lines(indented)

    assert lines == [indented], (
        "an indented literal must still be seen, or the guard reports green by "
        "extracting nothing"
    )
    assert parse_shadow_ruling(lines[0]) is None, (
        "the live parser requires the marker at index 0; if this now passes, "
        "_marker_lines has started normalising the presented bytes again"
    )
    assert parse_shadow_ruling(lines[0].strip()) is not None, (
        "sanity: the same payload unindented must parse, so the rejection "
        "above is about the indentation and not a malformed fixture"
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


# IDs are the headings alone, never the frozensets: a set's repr order varies
# with each process's hash seed, so deriving an id from one makes every xdist
# worker collect a differently-named test and the run dies in collection.
@pytest.mark.parametrize(
    ("heading", "expected"),
    _SLUG_SECTIONS,
    ids=[heading for heading, _ in _SLUG_SECTIONS],
)
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


def test_the_shadow_sections_quoted_command_actually_runs(tmp_path: pathlib.Path):
    """The strongest form of the check below: the doc's OWN argv, driven
    through the live `main()`.

    `assert "--queue-dir" in section` — what this replaces — was a one-way
    string pin on prose. It failed if the DOC dropped the flag and stayed green
    if the CLI RENAMED it, which is the direction that actually costs the
    operator a runnable command. Running the quoted arguments proves both
    directions at once: an unknown flag exits 2 out of argparse, and a required
    flag the doc forgot to quote does too.

    Placeholders (`<project_root>/...`) are substituted with a real empty
    directory, so this exercises the argument vocabulary and nothing else.

    THE LIMIT, measured rather than assumed: argparse accepts an unambiguous
    PREFIX of a long option, so renaming `--queue-dir` to `--queue-directory`
    still parses and still passes here. A rename that is not a prefix of the
    new spelling — the ordinary case — exits 2 and fails.
    """
    section = _section(_read(SKILL), _SHADOW_SECTION)
    quoted = _QUOTED_ARGV.search(section)
    assert quoted, (
        "the shadow section must quote the weekly-count command with its "
        "arguments, or the measurement half has no operator."
    )

    argv = [
        str(tmp_path) if token.startswith("<") else token
        for token in quoted.group("args").replace("\\\n", " ").split()
    ]
    assert argv, "the quoted command passes no arguments; --queue-dir is required"
    assert main(argv) == 0, (
        f"the command the shadow section tells the operator to run is not "
        f"runnable as quoted: {argv}"
    )


def test_the_shadow_section_quotes_a_runnable_weekly_count_command():
    """The module spelling is EXTRACTED from the doc and run against the
    import system, never compared to a literal written here.

    Asserting `escalation.shadow_ruling.__name__ == "escalation.shadow_ruling"`
    cannot fail — a module's `__name__` is its import path by construction — so
    it tested nothing the file-level import had not already proved, and a doc
    quoting `python -m escalation.shadow_ruling_v2` would have passed it while
    naming a module that does not exist.
    """
    section = _section(_read(SKILL), _SHADOW_SECTION)
    quoted = re.findall(r"python -m ([A-Za-z_][\w.]*)", section)
    assert quoted, (
        "the shadow section must quote the weekly-count command, or the "
        "measurement half has no operator."
    )

    for spelling in sorted(set(quoted)):
        module = importlib.import_module(spelling)
        assert callable(getattr(module, "main", None)), (
            f"the shadow section quotes `python -m {spelling}`, but that module "
            f"exposes no callable main() — the command it tells the operator to "
            f"run is not runnable."
        )
