"""Contract guards on the /unblock runbook's ``done_provenance`` instructions:
the payload literals validate against the LIVE ``DoneProvenance`` model, and
every command the doc offers as a SHA source is scoped to THIS task.

Task 4095. ``skills/unblock/SKILL.md`` ends by telling an agent to call
``set_task_status(status="done", done_provenance=...)``. A template that omits
the REQUIRED ``kind`` — or a ``found_on_main`` missing its ``commit``/``note``
— is hard-rejected at the write chokepoint on the last step of a SUCCESSFUL
unblock, after the merge has already landed, leaving the task stuck in exactly
the state /unblock exists to clear.

TWO INVARIANTS, TWO DISTINCT FAILURE MODES — do not fold them together:

1. PAYLOAD SHAPE (top half of this module). Is the JSON object one the server
   will accept at all?
2. SHA PROVENANCE (bottom half, under its own banner). Does the commit inside
   that object actually belong to THIS task? A payload can be perfectly
   well-formed and still cite an unrelated task's merge — a mode invariant 1
   is structurally blind to, and the one this very task's first draft
   introduced by deriving the SHA from ``git log --format=%H -1 main``.

SCOPE — STRUCTURE ONLY, NEVER PROSE, for both. Every word of guidance around
these literals and commands stays free to be rewritten, reordered or deleted;
what is held is only that a JSON object the doc presents as a payload is one
the server will accept, and that a git invocation it presents as a SHA source
takes a this-task argument. Nothing about the schema is hand-copied — the key
vocabulary and the requirements are read off the same pydantic model the
chokepoint (``fused-memory``'s ``_validate_done_provenance``) derives its own
from.

COUNTER-EXAMPLES. To show a REJECTED shape verbatim next to a good one, put
``<!-- provenance-guard: negative -->`` on the same line: the extractors skip
marked literals and marked commands, and the anchor cross-checks skip marked
anchors. Without that marker a valid-JSON counter-example — or a forbidden
command the doc names so an agent can recognise it — fails these guards, as it
should. The markers on the runbook's stated prohibitions are therefore
load-bearing, not decoration.

PLACEMENT. ``tests/scripts/`` is collected by its own module config so this
runs inside verify; ``skills/`` may hold no tests of its own
(``test_skills_module_config_decision.py``). Because this READS a real path
under ``skills/`` it is registered in that module's ``SKILLS_CONSUMING_TESTS``
and mirrored into ``dark-factory-orchestrator.yaml``, which must not drift
apart from it.

The measured-red transcript, the precedent argument for this test's shape and
the anti-vacuity design narrative live in task 4095's record and in the commit
that introduced this file.
"""
from __future__ import annotations

import json
import pathlib
import re
from typing import Any, NamedTuple

import pytest
from pydantic import ValidationError
from shared.task_metadata import DoneProvenance

REPO_ROOT = pathlib.Path(__file__).parents[2]

# Repo-relative docs whose done_provenance payload literals are held to the
# live model. A TUPLE, not a single constant, because the identical defect in
# `skills/orchestrate/SKILL.md` is already filed as task 4400 — whose own
# files_to_modify names THIS file, i.e. it is scoped to be one added line
# here. Seeded with just skills/unblock/SKILL.md because task 4095 names that
# file singular; widening it now would make this task's own guard red for a
# reason outside its scope.
_GUARDED_DOCS: tuple[str, ...] = ("skills/unblock/SKILL.md",)

# Keys the write chokepoint documents as accepted but the model does not
# DECLARE — they ride on `extra='allow'`. See the schema block in
# `fused-memory/src/fused_memory/middleware/task_interceptor.py`'s
# `_validate_done_provenance`: `transient_unit` (str) and `fire_delay_secs`
# (int) are read off the raw payload for kind='deterministic-deploy-scheduled'.
# Unioned into the recognised vocabulary below so a payload carrying them is
# EXTRACTED and checked rather than silently dropped by the key-subset filter.
_CHOKEPOINT_EXTRA_KEYS = frozenset({"transient_unit", "fire_delay_secs"})

# The recognised key vocabulary, DERIVED from the live model rather than
# hand-listed, so a field added to DoneProvenance is picked up here for free.
_PROVENANCE_FIELDS = frozenset(DoneProvenance.model_fields) | _CHOKEPOINT_EXTRA_KEYS

# Bounded lookahead for the balanced-brace walk: an unbalanced `{` in prose
# must abandon that start, never run away over the rest of the file.
_MAX_LITERAL_CHARS = 4000

# The textual form an author writes when threading a payload into the tool
# call. Used ONLY by the (c) coverage cross-check — the extractor itself is
# deliberately anchor-free, since the bare fallback literals this doc states
# mid-sentence carry no `done_provenance=` prefix at all.
_ANCHOR_RE = re.compile(r"done_provenance\s*=\s*\{")

# Opt-out marker for a literal the doc quotes as a shape the server REJECTS.
# A machine token in an HTML comment (invisible in rendered markdown), so it
# pins no prose and costs the author one inline annotation.
_NEGATIVE_MARKER = "provenance-guard: negative"


class ProvenanceLiteral(NamedTuple):
    """One JSON object literal that the doc presents as a provenance payload.

    ``source`` is stamped in at extraction time so a failure message can name
    its origin without the caller re-threading it alongside the literal.
    """

    source: str
    line: int
    text: str
    obj: dict[str, Any]
    start: int


def _line_at(text: str, offset: int) -> str:
    """The whole physical line of *text* containing *offset*."""
    start = text.rfind("\n", 0, offset) + 1
    end = text.find("\n", offset)
    return text[start:] if end == -1 else text[start:end]


def _is_negated(text: str, offset: int) -> bool:
    """True if the line at *offset* opts out via `_NEGATIVE_MARKER`."""
    return _NEGATIVE_MARKER in _line_at(text, offset)


def _provenance_literals(text: str, *, source: str) -> list[ProvenanceLiteral]:
    """Every JSON object literal in *text* that is shaped like a provenance blob.

    Walks every ``{`` in the document, matching braces with string-awareness
    (so a ``}`` inside a JSON string cannot close the object early) under a
    bounded lookahead. Embedded newlines and markdown indentation are collapsed
    to single spaces so a literal wrapped across two source lines is still
    parsed as one object. Non-dicts, empty dicts, parse failures and literals
    on a `_NEGATIVE_MARKER` line are discarded. Each surviving literal is
    stamped with *source* so a failure downstream can name where it came from.

    THE KEY-SUBSET FILTER BELOW IS LOAD-BEARING — DO NOT DROP IT AS REDUNDANT
    "the model would reject it anyway" TIDYING. ``DoneProvenance.model_config``
    sets ``extra='allow'``, so the model does NOT reject unknown keys: handed
    the unrelated ``{"error": "Merge queue not available — orchestrator not
    running"}`` literal this same runbook quotes as a tool RESPONSE, it would
    reject it for MISSING KIND — indistinguishably from a real defect, and the
    guard would report a false positive on a literal that has nothing to do
    with provenance. Restricting to objects whose keys are a SUBSET of the
    recognised vocabulary is what makes this guard sound.

    This function is TOTAL — it never raises, so it can be unit-tested against
    fixture markdown that legitimately yields nothing. Loud failure on an empty
    or under-covered extraction is the callers' job, in the two non-vacuity
    tests below.
    """
    out: list[ProvenanceLiteral] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        end = _match_brace(text, i)
        if end is None:
            i += 1
            continue
        flat = re.sub(r"\s*\n\s*", " ", text[i:end]).strip()
        try:
            obj = json.loads(flat)
        except (ValueError, TypeError):
            i += 1
            continue
        if isinstance(obj, dict) and obj and set(obj) <= _PROVENANCE_FIELDS:
            if _is_negated(text, i):
                # Quoted deliberately as a shape the server rejects. Skip past
                # it so nothing inside is re-reported either.
                i = end
                continue
            out.append(
                ProvenanceLiteral(
                    source=source,
                    line=text.count("\n", 0, i) + 1,
                    text=flat,
                    obj=obj,
                    start=i,
                )
            )
            # Skip past the accepted literal so a nested object is not also
            # reported as a second, overlapping hit.
            i = end
        else:
            i += 1
    return out


def _match_brace(text: str, start: int) -> int | None:
    """Offset just past the ``}`` closing the ``{`` at *start*, or None."""
    depth = 0
    in_str = False
    escaped = False
    limit = min(len(text), start + _MAX_LITERAL_CHARS)
    for j in range(start, limit):
        char = text[j]
        if in_str:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_str = False
            continue
        if char == '"':
            in_str = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return j + 1
    return None


def _unmarked_anchors(text: str) -> list[tuple[int, int]]:
    """``(line, brace_offset)`` for each `done_provenance={` not opted out."""
    return [
        (text.count("\n", 0, m.end() - 1) + 1, m.end() - 1)
        for m in _ANCHOR_RE.finditer(text)
        if not _is_negated(text, m.start())
    ]


def _doc_text(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


def _literals_in_doc(rel_path: str) -> list[ProvenanceLiteral]:
    """Provenance literals in a real guarded doc, read off disk."""
    return _provenance_literals(_doc_text(rel_path), source=rel_path)


def _collect() -> dict[str, list[ProvenanceLiteral]]:
    """Per-doc literals, tolerating a missing file so test (b) can name it."""
    found: dict[str, list[ProvenanceLiteral]] = {}
    for rel in _GUARDED_DOCS:
        path = REPO_ROOT / rel
        found[rel] = _literals_in_doc(rel) if path.is_file() else []
    return found


_DOC_LITERALS = _collect()

_PARAMS = [
    pytest.param(lit, id=f"{rel}:{lit.line}")
    for rel, lits in _DOC_LITERALS.items()
    for lit in lits
]

_REQUIRED_SHAPES = (
    "`kind` is REQUIRED on every done_provenance payload; "
    "kind='merged' additionally requires `commit`; "
    "kind='found_on_main' additionally requires BOTH `commit` AND `note` "
    "(note alone is no longer accepted — post-3092 phantom-done hardening). "
    "There is no note-only provenance: if you cannot cite a commit, the merge "
    "did not land. If this literal is deliberately quoting a shape the server "
    f"REJECTS, mark its line `<!-- {_NEGATIVE_MARKER} -->`."
)


@pytest.mark.parametrize("literal", _PARAMS)
def test_done_provenance_literal_is_accepted_by_the_live_model(
    literal: ProvenanceLiteral,
) -> None:
    """A payload template the runbook tells an agent to send must validate.

    Failing here means an agent that copies this template verbatim gets its
    final `set_task_status(status="done", ...)` hard-rejected at the
    fused-memory chokepoint, after the merge has already landed.
    """
    try:
        DoneProvenance.model_validate(literal.obj)
    except ValidationError as exc:
        pytest.fail(
            f"{literal.source}:{literal.line} presents a done_provenance payload the "
            f"server will REJECT (task 4095): {literal.text}\n"
            f"{_REQUIRED_SHAPES}\n"
            f"Live DoneProvenance (shared/src/shared/task_metadata.py) says:\n"
            f"{exc}"
        )


def test_every_guarded_doc_exists_and_yields_literals() -> None:
    """(b) NON-VACUITY: the parameterized guard above must have real params.

    Spelled `len(...) > 0` rather than as a truthiness test, per the pyright
    `reportAssertAlwaysTrue` note in test_skills_module_config_decision.py.
    """
    assert len(_GUARDED_DOCS) > 0, (
        "the guarded-doc inventory is EMPTY (task 4095) — this module would "
        "pin nothing at all."
    )
    missing = [rel for rel in _GUARDED_DOCS if not (REPO_ROOT / rel).is_file()]
    assert not missing, (
        f"guarded runbooks have MOVED or been DELETED (task 4095): {missing}. "
        f"Do NOT just drop them from _GUARDED_DOCS — if a runbook moved, "
        f"repoint the tuple; if it was deleted, its done_provenance templates "
        f"went with it and that should be confirmed deliberately."
    )
    barren = [rel for rel in _GUARDED_DOCS if len(_DOC_LITERALS[rel]) == 0]
    assert not barren, (
        f"no done_provenance payload literal was extracted from {barren} "
        f"(task 4095) — the parameterized guard would pass VACUOUSLY. Either "
        f"the runbook genuinely stopped presenting payload templates, or the "
        f"extractor no longer recognises the form they are written in."
    )


@pytest.mark.parametrize("rel_path", _GUARDED_DOCS)
def test_every_textual_anchor_is_recognised_by_the_extractor(rel_path: str) -> None:
    """(c) COVERAGE: every `done_provenance={` the doc writes must be scanned.

    Catches a future edit that reintroduces a payload in a shape the balanced
    brace walker misses. Asserts anchor COVERAGE only — never a count, and
    never total equality. The extractor legitimately finds MORE literals than
    there are anchors, because this runbook also states bare fallback payloads
    mid-sentence with no `done_provenance=` prefix (exactly the sites an
    anchor-only regex would have missed, and exactly where two of task 4095's
    five defects lived); and a doc that presented every payload some other way
    — a table, a variable threaded into the call — is a legitimate rewrite this
    check must not block. Non-vacuity is test (b)'s job, from extracted
    literals, which is the stronger signal anyway.

    It counts a machine token, not wording, so it pins no prose.
    """
    text = _doc_text(rel_path)
    starts = {lit.start for lit in _DOC_LITERALS[rel_path]}
    missed = [line for line, offset in _unmarked_anchors(text) if offset not in starts]
    assert not missed, (
        f"{rel_path} writes a `done_provenance={{...}}` payload at line(s) "
        f"{missed} that the structural extractor did NOT recognise (task "
        f"4095). The literal is therefore UNGUARDED — it could be rejected by "
        f"the server and nothing here would notice. Fix the payload's form, "
        f"or extend _provenance_literals to cover it."
    )


# ---------------------------------------------------------------------------
# (a) Extractor unit tests, against HAND-WRITTEN fixture markdown rather than
# the real doc — so the extractor's behaviour stays pinned no matter what the
# runbook happens to contain, and this module keeps biting once the ratchet
# above is green by construction.
# ---------------------------------------------------------------------------

_FIXTURE_SOURCE = "<fixture>"


def _fixture(text: str) -> list[ProvenanceLiteral]:
    return _provenance_literals(text, source=_FIXTURE_SOURCE)


def test_extractor_catches_kindless_attached_payload() -> None:
    """The headline defect shape: attached to `done_provenance=`, no `kind`."""
    found = _fixture(
        'Then call `set_task_status(..., done_provenance={"commit": "<sha>"})`.\n'
    )
    assert [lit.obj for lit in found] == [{"commit": "<sha>"}]
    assert found[0].line == 1
    with pytest.raises(ValidationError):
        DoneProvenance.model_validate(found[0].obj)


def test_extractor_catches_bare_note_only_fallback_in_prose() -> None:
    """A fallback stated mid-sentence, with NO `done_provenance=` prefix.

    This is precisely why the extractor is structural rather than anchored on
    that token: an anchor-only regex would miss this site entirely, and two of
    task 4095's five defects lived here.
    """
    found = _fixture(
        'If the search comes back empty, fall back to `{"note": "<explanation>"}`.\n'
    )
    assert [lit.obj for lit in found] == [{"note": "<explanation>"}]
    with pytest.raises(ValidationError):
        DoneProvenance.model_validate(found[0].obj)


def test_extractor_accepts_wellformed_merged_payload() -> None:
    found = _fixture('Pass `{"kind": "merged", "commit": "<sha>"}` for that case.\n')
    assert [lit.obj for lit in found] == [{"kind": "merged", "commit": "<sha>"}]
    # Must not raise — this is the shape the fix converges on.
    DoneProvenance.model_validate(found[0].obj)


def test_extractor_joins_a_literal_wrapped_across_two_lines() -> None:
    """Markdown wraps long templates; the payload is still one object."""
    found = _fixture(
        "  - proceed to step 8 with `done_provenance={\"kind\": \"found_on_main\",\n"
        '    "commit": "<sha>", "note": "absorbed into train <train_id>"}`.\n'
    )
    assert [lit.obj for lit in found] == [
        {
            "kind": "found_on_main",
            "commit": "<sha>",
            "note": "absorbed into train <train_id>",
        }
    ]
    assert "\n" not in found[0].text
    assert found[0].line == 1
    DoneProvenance.model_validate(found[0].obj)


def test_extractor_ignores_a_non_provenance_object() -> None:
    """The key-subset filter, pinned.

    `DoneProvenance.model_config` sets `extra='allow'`, so handing this literal
    to the model would reject it for MISSING KIND — a false positive on a tool
    RESPONSE the runbook quotes, which has nothing to do with provenance.
    """
    assert DoneProvenance.model_config.get("extra") == "allow"
    found = _fixture(
        '- `{"error": "Merge queue not available — orchestrator not running"}` '
        "-> orchestrator is down.\n"
    )
    assert found == []


def test_extractor_ignores_non_objects_and_unparseable_braces() -> None:
    """Prose braces, empty objects and JSON arrays are not payloads."""
    found = _fixture(
        "Run `git log --format=%H` and note the `{` in this unbalanced sentence.\n"
        "An empty `{}` is not a payload, nor is a shell `${VAR}` expansion.\n"
    )
    assert found == []


def test_extractor_keeps_chokepoint_extra_keys_that_the_model_does_not_declare() -> None:
    """`transient_unit`/`fire_delay_secs` ride on `extra='allow'`.

    They are documented as accepted by `_validate_done_provenance` but are not
    declared fields, so a key-subset filter built from `model_fields` ALONE
    would silently drop this payload — an unanchored one would then be checked
    by nothing at all.
    """
    assert not _CHOKEPOINT_EXTRA_KEYS & frozenset(DoneProvenance.model_fields), (
        "these keys became real model fields — drop them from "
        "_CHOKEPOINT_EXTRA_KEYS rather than carrying a stale union."
    )
    found = _fixture(
        'Stamp `{"kind": "deterministic-deploy-scheduled", "unit": "orch.service", '
        '"transient_unit": "restart-orch.service", "fire_delay_secs": 30}`.\n'
    )
    assert [lit.obj for lit in found] == [
        {
            "kind": "deterministic-deploy-scheduled",
            "unit": "orch.service",
            "transient_unit": "restart-orch.service",
            "fire_delay_secs": 30,
        }
    ]
    DoneProvenance.model_validate(found[0].obj)


def test_extractor_is_string_aware_about_braces_and_escaped_quotes() -> None:
    """A `}` or an escaped `"` inside a JSON string must not close the object.

    This is the entire reason `_match_brace` tracks `in_str`/`escaped`; without
    it the literal below truncates at the `}` in the note and silently drops.
    """
    found = _fixture(
        'Pass `{"kind": "merged", "commit": "<sha>", '
        '"note": "closes the {stuck} state; queue said \\"landed\\""}`.\n'
    )
    assert [lit.obj for lit in found] == [
        {
            "kind": "merged",
            "commit": "<sha>",
            "note": 'closes the {stuck} state; queue said "landed"',
        }
    ]
    DoneProvenance.model_validate(found[0].obj)


def test_match_brace_abandons_a_run_longer_than_the_bounded_lookahead() -> None:
    """The lookahead bound, pinned at both sides of the boundary.

    An unbalanced `{` in prose must abandon that start rather than walk the
    rest of the file; a literal comfortably inside the bound must still match.
    """
    assert _match_brace("{" + "x" * 10 + "}", 0) == 12
    runaway = "{" + "x" * (_MAX_LITERAL_CHARS + 100) + "}"
    assert _match_brace(runaway, 0) is None
    assert _fixture(runaway) == []


def test_extractor_reports_a_nested_payload_exactly_once() -> None:
    """A payload nested in a larger object is reported once, as the payload.

    Pins the `i = end` skip: the enclosing objects fail the key-subset filter
    and are walked past, and the accepted inner literal is not re-scanned into
    a second overlapping hit.
    """
    text = (
        'The task record holds `{"metadata": {"done_provenance": '
        '{"kind": "merged", "commit": "<sha>"}}}` after the write.\n'
    )
    found = _fixture(text)
    assert [lit.obj for lit in found] == [{"kind": "merged", "commit": "<sha>"}]
    assert found[0].start == text.index('{"kind"')


def test_extractor_skips_a_literal_marked_as_a_negative_example() -> None:
    """A quoted counter-example opts out with an inline marker.

    Without the escape the guard silently forbids showing the rejected shape
    verbatim next to the good one — a normal and useful doc improvement.
    """
    bad = 'A bare `{"commit": "<sha>"}` is hard-rejected at the chokepoint.'
    # Unmarked, the same literal IS extracted — so the marker, not the shape,
    # is what suppresses it below.
    assert [lit.obj for lit in _fixture(bad + "\n")] == [{"commit": "<sha>"}]
    assert _fixture(f"{bad} <!-- {_NEGATIVE_MARKER} -->\n") == []


def test_anchor_crosscheck_skips_a_marked_negative_anchor() -> None:
    """A marked counter-example must not read as an unrecognised payload.

    The extractor drops it by design, so the (c) coverage check has to drop its
    anchor too — otherwise the escape would trade a false payload failure for a
    false coverage failure.
    """
    line = 'Never write `done_provenance={"commit": "<sha>"}`'
    assert len(_unmarked_anchors(f"{line}.\n")) == 1
    assert _unmarked_anchors(f"{line} <!-- {_NEGATIVE_MARKER} -->\n") == []


# ---------------------------------------------------------------------------
# SECOND INVARIANT — every provenance SHA DERIVATION in the runbook is
# TASK-SCOPED.
#
# The guard above validates the payload's SHAPE. It is structurally BLIND to
# where the `<sha>` inside it came from: a payload can carry a perfectly
# well-formed kind/commit/note and still cite an UNRELATED task's merge
# commit. That is the second, distinct way this runbook can produce a wrong
# `done` — and it is the way task 4095's own first draft broke it, by telling
# the agent to derive the landing SHA from `git log --format=%H -1 main`, i.e.
# main's CURRENT HEAD. On a live merge queue that is this task's merge commit
# only when this task's merge happens to be the newest commit on main.
#
# Demonstrated in this repo on the real landed branch task/4559:
#   exact-subject marker search                    -> 73b5f4555b  (TRUE merge)
#   git log --format=%H -1 main                    -> f62767929b  (UNRELATED)
#   git rev-list --ancestry-path --merges ..|tail  -> 73b5f4555b  (TRUE merge)
#
# Nothing downstream catches the misattribution: the server's only
# `done_provenance` backstop is `git merge-base --is-ancestor <sha> main`,
# which every recent commit on main passes — the exact reason this doc's own
# prohibitions (the rc=128 bullet, the polled-done bullet) already give. It
# lands as the `misattributed` / commit fan-out class that
# `fused-memory/scripts/check_found_on_main_spurious_rate.py` and
# `audit_found_on_main_provenance.py` gate on only AFTER the fact.
#
# SCOPE — GIT COMMAND TOKENS ONLY, NEVER PROSE, exactly as above. What is held
# is that a git invocation the doc presents as a SHA source takes a THIS-TASK
# argument. Every word of surrounding guidance stays free to be rewritten.
# ---------------------------------------------------------------------------

# Markdown wraps a long command three different ways in this one runbook, and
# a continuation-only join catches only two of them:
#   * bare `\` continuation inside a fenced block;
#   * `\` continuation inside `#`-prefixed comment lines in a fenced block;
#   * a prose SOFT-WRAP with NO `\` at all, inside an inline-backtick span.
# A first prototype that joined only on `\` silently under-reported 6 of the 7
# real sites, missing the soft-wrapped one. So this FLATTENS — newline +
# markdown indent + optional `#` prefix + optional trailing `\` collapse to a
# single space — mirroring the collapse `_provenance_literals` already does
# for multi-line payloads. The `[ \t]+` after the newline is what keeps some
# structure: a line starting at column 0 is never joined to the one above.
_JOIN_RE = re.compile(r"[ \t]*\\?\n[ \t]+(?:#[ \t]*)?")

# A git invocation that EMITS a commit SHA: an explicit `--format=%H`, or a
# bare `git rev-list` (which prints SHAs by default). Bounded repetition, and
# `` ` ``/newline excluded, so a match can never run away across the flattened
# document and swallow unrelated text into "the command".
_SHA_DERIVATION_RE = re.compile(
    r"git\s+(?:log|rev-list|rev-parse)\b[^`\n]{0,400}?--format=%H"
    r"|git\s+rev-list\b[^`\n|;]{0,200}"
)

# The two forms that make a derivation THIS task's. Either the exact-subject
# merge-marker search (`--grep=` carrying the `task/<...>` subject placeholder,
# the shape `GitOps.find_merge_marker` uses), or a `task/<...>..main` revision
# range. Both take a task argument; `-1 main` does not.
_SCOPED_GREP_RE = re.compile(r"--grep=[\"']?[^\"'`]{0,80}?task/<[A-Za-z0-9_]+>")
_TASK_REV_RANGE_RE = re.compile(r"task/<[A-Za-z0-9_]+>\.\.\w+")

# Used ONLY by the coverage cross-check below, never by the extractor.
_FORMAT_ANCHOR_RE = re.compile(r"--format=%H")

_REQUIRED_SCOPING = (
    "A provenance SHA source must name THIS task. Use either the exact-subject "
    'merge-marker search — `git log main --fixed-strings --grep="Merge '
    'task/<TASK_ID> into main" --max-count=1 --format=%H` — or an ancestry-path '
    "range — `git rev-list --ancestry-path --merges task/<TASK_ID>..main | tail "
    "-1`. An unscoped `git log --format=%H -1 main` takes no task argument: it "
    "yields main's CURRENT HEAD, which is this task's merge commit only when "
    "this task's merge is the newest commit on main. The server's only backstop "
    "(`git merge-base --is-ancestor <sha> main`) passes for every recent commit "
    "on main, so nothing downstream would catch the misattribution. If this "
    "command is deliberately quoted as one the runbook FORBIDS, mark its line "
    f"`<!-- {_NEGATIVE_MARKER} -->`."
)


class ShaDerivation(NamedTuple):
    """One git invocation the doc presents as a source of a commit SHA."""

    source: str
    line: int
    command: str
    start: int
    end: int


def _flatten(text: str) -> tuple[str, list[int]]:
    """*text* with continuations/soft-wraps collapsed, plus an offset map.

    Returns ``(flat, offsets)`` where ``offsets[i]`` is the offset in the
    ORIGINAL *text* of ``flat[i]`` — so a match found in the flattened text can
    be reported at its true line and cross-checked against textual anchors at
    their true offsets. Each inserted join space maps back to the newline run
    it replaced.
    """
    parts: list[str] = []
    offsets: list[int] = []
    pos = 0
    for m in _JOIN_RE.finditer(text):
        parts.append(text[pos : m.start()])
        offsets.extend(range(pos, m.start()))
        parts.append(" ")
        offsets.append(m.start())
        pos = m.end()
    parts.append(text[pos:])
    offsets.extend(range(pos, len(text)))
    return "".join(parts), offsets


def _is_task_scoped(command: str) -> bool:
    """True if *command* takes a THIS-TASK argument."""
    return bool(_SCOPED_GREP_RE.search(command) or _TASK_REV_RANGE_RE.search(command))


def _sha_derivations(text: str, *, source: str) -> list[ShaDerivation]:
    """Every git invocation in *text* that emits a commit SHA.

    Reported at the line of the command's LAST character — i.e. the line
    carrying `--format=%H` — so a wrapped command is named where its SHA-
    emitting token actually is, and so the anchor cross-check below lines up
    with it exactly.

    A derivation on a `_NEGATIVE_MARKER` line is skipped. THAT ESCAPE IS
    LOAD-BEARING HERE, not a nicety: the runbook states the forbidden
    `git log --format=%H -1 main` verbatim so an agent can recognise it, and
    without the marker this guard would flag the doc's own prohibition as a
    violation. Do not strip the markers from those lines.

    TOTAL — never raises, so it is unit-testable against fixture markdown that
    legitimately yields nothing. Loud failure on an empty or under-covered
    extraction is the callers' job, in the two non-vacuity tests below.
    """
    flat, offsets = _flatten(text)
    out: list[ShaDerivation] = []
    for m in _SHA_DERIVATION_RE.finditer(flat):
        start = offsets[m.start()]
        end = offsets[m.end() - 1]
        if _is_negated(text, start) or _is_negated(text, end):
            continue
        out.append(
            ShaDerivation(
                source=source,
                line=text.count("\n", 0, end) + 1,
                command=m.group(0).strip(),
                start=start,
                end=end,
            )
        )
    return out


def _unmarked_format_anchors(text: str) -> list[tuple[int, int]]:
    """``(line, offset)`` for each textual `--format=%H` not opted out."""
    return [
        (text.count("\n", 0, m.start()) + 1, m.start())
        for m in _FORMAT_ANCHOR_RE.finditer(text)
        if not _is_negated(text, m.start())
    ]


def _derivations_in_doc(rel_path: str) -> list[ShaDerivation]:
    return _sha_derivations(_doc_text(rel_path), source=rel_path)


def _collect_derivations() -> dict[str, list[ShaDerivation]]:
    """Per-doc derivations, tolerating a missing file so the guard above names it."""
    found: dict[str, list[ShaDerivation]] = {}
    for rel in _GUARDED_DOCS:
        path = REPO_ROOT / rel
        found[rel] = _derivations_in_doc(rel) if path.is_file() else []
    return found


_DOC_DERIVATIONS = _collect_derivations()

_DERIVATION_PARAMS = [
    pytest.param(deriv, id=f"{rel}:{deriv.line}")
    for rel, derivs in _DOC_DERIVATIONS.items()
    for deriv in derivs
]


@pytest.mark.parametrize("derivation", _DERIVATION_PARAMS)
def test_provenance_sha_derivation_is_task_scoped(derivation: ShaDerivation) -> None:
    """A command the runbook offers as a SHA source must name THIS task.

    Failing here means an agent that copies this command verbatim stamps some
    other task's commit as its own landing SHA — a payload that is perfectly
    well-formed, passes the server's ancestry backstop, and is wrong.
    """
    assert _is_task_scoped(derivation.command), (
        f"{derivation.source}:{derivation.line} derives a provenance SHA with a "
        f"command that is NOT scoped to this task (task 4095): "
        f"{derivation.command}\n{_REQUIRED_SCOPING}"
    )


def test_every_guarded_doc_yields_sha_derivations() -> None:
    """NON-VACUITY: the parameterized scoping guard must have real params."""
    barren = [rel for rel in _GUARDED_DOCS if len(_DOC_DERIVATIONS[rel]) == 0]
    assert not barren, (
        f"no provenance SHA derivation was extracted from {barren} (task 4095) "
        f"— the scoping guard would pass VACUOUSLY. Either the runbook genuinely "
        f"stopped telling an agent how to derive a landing SHA, or the extractor "
        f"no longer recognises the form those commands are written in."
    )


@pytest.mark.parametrize("rel_path", _GUARDED_DOCS)
def test_every_format_anchor_is_recognised_by_the_command_scan(rel_path: str) -> None:
    """COVERAGE: every textual `--format=%H` must sit inside a scanned command.

    This is what stops a future rewrite from hiding a SHA derivation in a shape
    the scanner misses — a wrapping style the flattener does not collapse, a
    command spelled across a table cell — and thereby smuggling an unscoped
    derivation past the guard above. It counts a machine token, not wording, so
    it pins no prose.
    """
    text = _doc_text(rel_path)
    spans = [(d.start, d.end) for d in _DOC_DERIVATIONS[rel_path]]
    missed = [
        line
        for line, offset in _unmarked_format_anchors(text)
        if not any(start <= offset <= end for start, end in spans)
    ]
    assert not missed, (
        f"{rel_path} emits a commit SHA with `--format=%H` at line(s) {missed} "
        f"that the command scan did NOT recognise (task 4095). That derivation "
        f"is therefore UNGUARDED — it could be unscoped and stamp an unrelated "
        f"task's merge commit, and nothing here would notice. Fix the command's "
        f"form, or extend _sha_derivations to cover it."
    )


# ---------------------------------------------------------------------------
# Extractor unit tests for the derivation scan, against HAND-WRITTEN fixture
# markdown — same house requirement as the payload extractor's above, and for
# the same reason: once step 4's doc fix lands, the ratchet is green by
# construction, so its behaviour has to be pinned independently of whatever
# the runbook happens to contain.
# ---------------------------------------------------------------------------


def _fixture_derivations(text: str) -> list[ShaDerivation]:
    return _sha_derivations(text, source=_FIXTURE_SOURCE)


def test_derivation_scan_catches_unscoped_main_head() -> None:
    """The headline defect: main's CURRENT HEAD, with no task argument."""
    found = _fixture_derivations(
        "# rc=0 (on main): proceed to step 8 with the landing SHA from "
        "`git log --format=%H -1 main`:\n"
    )
    assert [d.command for d in found] == ["git log --format=%H"]
    assert found[0].line == 1
    assert not _is_task_scoped(found[0].command)


def test_derivation_scan_accepts_the_exact_subject_marker_search() -> None:
    """The correct primary derivation, taken straight from the rc=128 arm."""
    found = _fixture_derivations(
        'Run `git log main --fixed-strings --grep="Merge task/<TASK_ID> into '
        'main" --max-count=1 --format=%H`.\n'
    )
    assert len(found) == 1
    assert _is_task_scoped(found[0].command)


def test_derivation_scan_accepts_an_ancestry_path_rev_range() -> None:
    """The fallback derivation: a `task/<...>..main` range is task-scoped too.

    Empirically validated on the real landed branch task/4559, where this and
    the marker search both return the true merge commit while an unscoped
    `git log --format=%H -1 main` returns an unrelated later commit.
    """
    found = _fixture_derivations(
        "#   git rev-list --ancestry-path --merges task/<TASK_ID>..main | tail -1\n"
    )
    assert len(found) == 1
    assert found[0].command.startswith("git rev-list --ancestry-path")
    assert _is_task_scoped(found[0].command)


def test_derivation_scan_joins_a_backslash_continuation_in_a_fenced_block() -> None:
    """Wrapping style 1 of 3 — bare `\\` continuation."""
    found = _fixture_derivations(
        '     git log main --fixed-strings --grep="Merge task/<TASK_ID> into main" \\\n'
        "         --max-count=1 --format=%H\n"
    )
    assert [d.command for d in found] == [
        'git log main --fixed-strings --grep="Merge task/<TASK_ID> into main" '
        "--max-count=1 --format=%H"
    ]
    # Reported at the `--format=%H` line, not the line the command opened on —
    # this is the offset map, pinned.
    assert found[0].line == 2
    assert _is_task_scoped(found[0].command)


def test_derivation_scan_joins_a_commented_continuation_in_a_fenced_block() -> None:
    """Wrapping style 2 of 3 — `\\` continuation across `#`-prefixed lines."""
    found = _fixture_derivations(
        '  #     git log main --fixed-strings --grep="Merge task/<TASK_ID> into main" \\\n'
        "  #         --max-count=1 --format=%H\n"
    )
    assert [d.command for d in found] == [
        'git log main --fixed-strings --grep="Merge task/<TASK_ID> into main" '
        "--max-count=1 --format=%H"
    ]
    assert found[0].line == 2
    assert _is_task_scoped(found[0].command)


def test_derivation_scan_joins_a_prose_soft_wrap_with_no_continuation() -> None:
    """Wrapping style 3 of 3 — a prose SOFT-WRAP with no `\\` at all.

    THE REASON `_flatten` FLATTENS RATHER THAN JOINING ON `\\`. A
    continuation-only join misses this shape entirely; a prototype that did so
    under-reported the real runbook's derivations by one, and the missed site
    would have been silently unguarded.
    """
    found = _fixture_derivations(
        "    **(a)** the tip's merge marker on main — `git log main --fixed-strings\n"
        '    --grep="Merge task/<TIP_ID> into main" --max-count=1 --format=%H` (with a\n'
    )
    assert [d.command for d in found] == [
        'git log main --fixed-strings --grep="Merge task/<TIP_ID> into main" '
        "--max-count=1 --format=%H"
    ]
    assert found[0].line == 2
    assert _is_task_scoped(found[0].command)


def test_derivation_scan_skips_a_command_marked_as_a_negative_example() -> None:
    """The runbook must be able to NAME the forbidden command.

    Step 4 states `git log --format=%H -1 main` verbatim so an agent recognises
    it on sight. Without this escape the guard would flag the doc's own
    prohibition as a violation — so the marker on that line is load-bearing,
    not decoration.
    """
    line = "**Never derive provenance from `git log --format=%H -1 main`**"
    # Unmarked, the same command IS extracted — the marker, not the wording, is
    # what suppresses it below.
    assert len(_fixture_derivations(line + "\n")) == 1
    assert _fixture_derivations(f"{line} <!-- {_NEGATIVE_MARKER} -->\n") == []
    # ...and its anchor is dropped from the coverage cross-check too, so the
    # escape does not trade a false scoping failure for a false coverage one.
    assert _unmarked_format_anchors(f"{line} <!-- {_NEGATIVE_MARKER} -->\n") == []


def test_derivation_scan_ignores_git_commands_that_emit_no_sha() -> None:
    """`--oneline`/`--merges` listings and `git merge-base` are not SHA sources.

    They are quoted in this runbook precisely as things NOT to derive a SHA
    from; matching them here would flag prose that already says the right
    thing.
    """
    assert (
        _fixture_derivations(
            "Do not eyeball `git log main --oneline | head -5` and pick a SHA, and "
            "do not substitute an unfiltered `git log main --merges | head -5`.\n"
            "The backstop is only `git merge-base --is-ancestor <sha> main`.\n"
        )
        == []
    )


def test_flatten_offsets_map_every_character_back_to_the_original() -> None:
    """The offset map, pinned end to end.

    A drift here would silently misreport every line number and quietly break
    the anchor cross-check's containment test.
    """
    text = "  a b \\\n    c\nd\n"
    flat, offsets = _flatten(text)
    assert flat == "  a b c\nd\n"
    assert len(offsets) == len(flat)
    assert all(text[offsets[i]] == flat[i] for i in range(len(flat)) if flat[i] != " ")
