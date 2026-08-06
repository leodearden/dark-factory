"""Mirror contract: CONTRIBUTING.md's Lint bullet must restate the live ``lint_command``.

Task 3558. CONTRIBUTING.md documented a ``uv run ruff check`` command that the
repo-root ``dark-factory-orchestrator.yaml`` had outgrown TWICE — task 3397 added
``sampler`` and ``cockpit``, task 3485 added ``conftest.py``,
``df_pytest_isolation.py`` and ``skills`` — while the doc stayed at its original
five targets. The failure mode is not cosmetic: a contributor who runs the
documented command gets a clean local green over a strict SUBSET of what the
merge gate checks, then eats a red verify on a branch they believed was ready.

This is the DOCS half of the ``lint_command`` contract — "does the prose still
say what the config says?". ``test_root_lint_covers_nonmember_py.py`` (task 3485)
is the FILES half — "does the config still target every file that needs it?".
Neither implies the other: the doc can faithfully mirror a command that misses a
file, and the command can cover every file while the doc describes a stale one.

WHY A MACHINE CHECK, WHEN A POINTER ALREADY EXISTS. CONTRIBUTING.md already tells
the reader to "treat ``dark-factory-orchestrator.yaml``'s ``test_command`` /
``lint_command`` / ``type_check_command`` as the source of truth if these drift."
Both drifts landed with that sentence already present, so it is measured — not
predicted — that prose pointing at prose does not hold. The literal is kept (its
value to a contributor is being copy-pasteable; ``lint_command`` sits under ~130
lines of comment in the yaml) and the tie is made mechanical instead.

MEASURED RED at base HEAD ``ca8dfb6a6e``: the documented command named 5 targets,
the live ruff leg named 10, and the missing set was exactly
``['sampler', 'cockpit', 'conftest.py', 'df_pytest_isolation.py', 'skills']``.

PLACEMENT IS LOAD-BEARING, NOT STYLISTIC. This file lives in ``tests/scripts/``
because that directory carries its own module config, so the guard actually runs
under FULL_SUITE and merge-role ``merge_verify_breadth: full``. A drift guard
that never ran on merge full-verify would be vacuous in exactly the way the gap
it closes is — the same rationale ``test_root_lint_covers_nonmember_py.py`` and
``test_scripts_module_config.py`` record for themselves.

SCOPE. This guard asserts equality between two committed *executable command
strings* — a config field and its documented mirror. It is NOT a prose or
docstring test: it makes no assertion about surrounding wording, headings or
explanatory text, and must not be extended to do so. It is the same class as
``TestSplitAndChainSegmentsLiveConfigDrift`` (orchestrator/tests/
test_verify_cmd.py) and ``test_orchestrator_restart_config_drift.py``.

EXPLICITLY OUT OF SCOPE. ``_ROOT_LINT_COMMAND`` (orchestrator/tests/
test_verify_cmd.py, test_verify_plan.py) and ``_FLEET_LINT_COMMAND_OPAQUE``
(orchestrator/tests/test_verify_scope_kappa.py) are DELIBERATELY decoupled
five-member shape fixtures exercising chain-splitting and scope-narrowing
behaviour, not live lint coverage. Re-copying the live string into them would
silently change what shape each fixture tests, converting a passing behavioural
test into a different, unreviewed one. A test fixture that happens to contain a
similar string is not a mirror of the config and must not be pinned as one. The
mirror obligation applies to CONTRIBUTING.md only, which is prose addressed to
humans.

SYNTHETIC FIXTURES ON PURPOSE. Every extractor case below runs against a
hand-written markdown string, never against the real CONTRIBUTING.md, so those
cases stay stable under any future edit to that file's content. Only the live
drift assertion reads the real file — a different invariant, deliberately.

The fixtures spell the ``lint-command-mirror`` marker literals out in full rather
than interpolating the constants below. That is deliberate: a rename of a
constant must not be able to silently keep a broken parser agreeing with its own
fixtures.
"""
from __future__ import annotations

import pathlib
import re
import shlex

import pytest
import yaml

from orchestrator import verify_cmd

REPO_ROOT = pathlib.Path(__file__).parents[2]

# The canonical root config filename — what the dashboard's escalation-URL
# discovery (_discover_escalation_urls) keys on. Loading it by this exact path
# means the guard fails loudly if the config is renamed or lint_command
# disappears, rather than quietly guarding nothing.
DF_CONFIG_PATH = REPO_ROOT / "dark-factory-orchestrator.yaml"
CONTRIBUTING_PATH = REPO_ROOT / "CONTRIBUTING.md"

# The HTML-comment marker anchoring extraction, placed immediately around the
# Lint bullet. An explicit marker rather than a regex on the `- **Lint**:` label
# or on "the span matching `uv run ruff check`", for two reasons.
#
# MECHANICAL: CONTRIBUTING.md carries a SECOND inline-code span beginning
# `uv run ruff check` in its quality-gates list — `uv run ruff check <touched
# packages>` — which is deliberately generic and must never be pinned. Any
# "first/only matching span" extractor is order-dependent and would silently
# re-target on a harmless doc reorder or relabel.
#
# CAUSAL: the root cause this task closes is "nothing ties the prose to the
# config". The marker puts that tie AT THE EDIT SITE — a human editing the line
# sees which yaml field it mirrors and which test pins it. HTML comments render
# invisibly, so the cost is two lines of source noise.
MIRROR_BEGIN = "lint-command-mirror:begin"
MIRROR_END = "lint-command-mirror:end"

_HTML_COMMENT_CLOSE = "-->"
_HTML_COMMENT_OPEN = "<!--"

# Non-greedy single-backtick inline-code span. Applied only to the marker
# payload, never to the whole document.
_INLINE_CODE = re.compile(r"`([^`]+)`")


def _documented_lint_command(markdown_text: str) -> str:
    """The single inline-code command delimited by the mirror markers.

    Every failure below is a loud ``AssertionError`` naming the marker literal
    and CONTRIBUTING.md, never a ``''``/``None`` return. That is the vacuity
    hazard and the whole point: an extractor that silently yields nothing turns
    the drift assertion green while pinning nothing at all — strictly worse than
    no guard, because the check still reports success.

    The payload is sliced from the END of the begin comment (its ``-->``) to the
    START of the end comment (its ``<!--``), NOT from the marker literals
    themselves. The begin comment's own prose contains inline-code spans — the
    real marker cites ``ruff check`` and ``lint_command`` — so slicing on the
    literal would return comment prose, and would return a plausible-looking
    string while doing it.

    Returns the command ``strip()``ed and otherwise verbatim. No further
    normalisation: the exact-equality assertion downstream depends on not
    silently canonicalising away a real difference.
    """
    begin_count = markdown_text.count(MIRROR_BEGIN)
    assert begin_count == 1, (
        f"expected exactly one {MIRROR_BEGIN!r} marker, found {begin_count} "
        f"(task 3558). This marker delimits the Lint bullet in CONTRIBUTING.md "
        f"that mirrors the `ruff check` leg of dark-factory-orchestrator.yaml's "
        f"lint_command. If it was deleted, restore it around that bullet; if it "
        f"was duplicated, one of the two mirrors is unpinned and free to drift."
    )
    end_count = markdown_text.count(MIRROR_END)
    assert end_count == 1, (
        f"expected exactly one {MIRROR_END!r} marker to close {MIRROR_BEGIN!r}, "
        f"found {end_count} (task 3558) — restore the closing marker below the "
        f"Lint bullet in CONTRIBUTING.md"
    )

    begin_at = markdown_text.index(MIRROR_BEGIN)
    end_at = markdown_text.index(MIRROR_END)
    assert begin_at < end_at, (
        f"{MIRROR_END!r} precedes {MIRROR_BEGIN!r} in CONTRIBUTING.md (task "
        f"3558) — the markers are inverted, so no payload can be delimited"
    )

    comment_close = markdown_text.find(_HTML_COMMENT_CLOSE, begin_at, end_at)
    assert comment_close != -1, (
        f"the {MIRROR_BEGIN!r} marker's HTML comment is never closed with "
        f"{_HTML_COMMENT_CLOSE!r} before {MIRROR_END!r} (task 3558) — "
        f"CONTRIBUTING.md's marker block is malformed"
    )
    payload_start = comment_close + len(_HTML_COMMENT_CLOSE)

    payload_end = markdown_text.rfind(_HTML_COMMENT_OPEN, payload_start, end_at)
    assert payload_end != -1, (
        f"the {MIRROR_END!r} marker is not opened by {_HTML_COMMENT_OPEN!r} "
        f"(task 3558) — CONTRIBUTING.md's marker block is malformed"
    )

    payload = markdown_text[payload_start:payload_end]
    spans: list[str] = _INLINE_CODE.findall(payload)
    assert len(spans) == 1, (
        f"expected exactly one `backticked` command between {MIRROR_BEGIN!r} "
        f"and {MIRROR_END!r}, found {len(spans)}: {spans!r} (task 3558). The "
        f"marker block in CONTRIBUTING.md must wrap the Lint bullet and nothing "
        f"else — additional prose with inline code belongs outside it."
    )

    command = spans[0].strip()
    assert command, (
        f"the command between {MIRROR_BEGIN!r} and {MIRROR_END!r} in "
        f"CONTRIBUTING.md is empty (task 3558)"
    )
    return command

# (a) Happy path. Modelled on the real CONTRIBUTING.md block: a fenced ```bash
# example ABOVE the marker (backticks outside the slice), and — the measured
# hazard — TWO inline-code spans inside the begin comment's own prose. The real
# marker committed in pre-1 really does carry `ruff check` and `lint_command` in
# its explanatory text, so an extractor that scanned the raw slice for "the one
# backtick span" would pick up comment prose, not the command.
_HAPPY_DOC = """\
- **Tests** run per-package with `pytest`, e.g.:
  ```bash
  cd orchestrator && uv run pytest tests/ --timeout=300
  ```
<!-- lint-command-mirror:begin
     Mirrors the `ruff check` leg of `lint_command` in
     dark-factory-orchestrator.yaml. Pinned by
     tests/scripts/test_contributing_lint_command_drift.py. -->
- **Lint**: `uv run ruff check alpha beta gamma.py`
<!-- lint-command-mirror:end -->
- **Type-check** (pyright), run from each configured package directory.
"""

_HAPPY_COMMAND = "uv run ruff check alpha beta gamma.py"

# (b) No marker at all — someone deleted it, or renamed the file's section.
# Note this doc still CONTAINS a plausible-looking lint bullet: the extractor
# must not fall back to "find something that looks right".
_NO_MARKER_DOC = """\
- **Lint**: `uv run ruff check alpha beta`

Then, before submitting:

2. `uv run ruff check <touched packages>`.
"""

# (c) Two begin markers — e.g. a section duplicated in a bad merge. Picking the
# first silently pins one of two mirrors and lets the other rot unwatched.
_DUPLICATE_MARKER_DOC = """\
<!-- lint-command-mirror:begin -->
- **Lint**: `uv run ruff check alpha beta`
<!-- lint-command-mirror:end -->

## Some later section

<!-- lint-command-mirror:begin -->
- **Lint**: `uv run ruff check delta epsilon`
<!-- lint-command-mirror:end -->
"""

# (d) Decoy immunity. The generic `uv run ruff check <touched packages>` span
# appears BOTH before and after the marker block, so neither a "first span" nor
# a "last span" heuristic can pass this by accident. This is a MEASURED hazard,
# not a hypothetical: CONTRIBUTING.md really carries that generic span in its
# quality-gates list, and it must stay generic and stay unpinned.
_DECOY_DOC = """\
1. `uv run ruff check <touched packages>` — before you start.

<!-- lint-command-mirror:begin
     Mirrors the `ruff check` leg of `lint_command`. -->
- **Lint**: `uv run ruff check alpha beta gamma.py`
<!-- lint-command-mirror:end -->

2. `uv run ruff check <touched packages>` — and again before submitting.
"""


def test_documented_lint_command_extracts_the_marked_span() -> None:
    """(a) The marked inline-code span is returned, markdown and backticks stripped."""
    assert _documented_lint_command(_HAPPY_DOC) == _HAPPY_COMMAND


def test_documented_lint_command_ignores_backticks_in_the_marker_prose() -> None:
    """(a') The begin comment's own inline-code spans are not the command.

    Split out from (a) because it is the specific failure an extractor that
    slices on the marker LITERAL (rather than on the end of the begin comment)
    exhibits: it would return ``ruff check`` — the first span of the comment's
    explanatory prose — which is a plausible-looking string, so the mistake
    would not announce itself.
    """
    extracted = _documented_lint_command(_HAPPY_DOC)
    assert extracted == _HAPPY_COMMAND
    assert extracted not in ("ruff check", "lint_command")
    assert extracted.startswith("uv run ruff check ")


def test_documented_lint_command_raises_when_the_marker_is_missing() -> None:
    """(b) A missing marker FAILS LOUDLY — never '' or None.

    This is the vacuity hazard, and it is the whole reason the guard exists: an
    extractor that silently returns nothing turns every downstream assertion
    green while pinning nothing at all, which is strictly worse than having no
    guard, because the check still reports success.
    """
    with pytest.raises(AssertionError) as excinfo:
        _documented_lint_command(_NO_MARKER_DOC)

    # The message must tell a human who deleted the marker exactly what to
    # restore and where — the marker literal and the file it belongs in.
    message = str(excinfo.value)
    assert "lint-command-mirror:begin" in message
    assert "CONTRIBUTING.md" in message


def test_documented_lint_command_raises_on_a_duplicated_marker() -> None:
    """(c) Two marker blocks FAIL LOUDLY rather than silently picking the first.

    Silently taking the first would leave the second mirror unpinned and free to
    drift — the exact failure this task closes, reintroduced one level down.
    """
    with pytest.raises(AssertionError) as excinfo:
        _documented_lint_command(_DUPLICATE_MARKER_DOC)

    assert "lint-command-mirror:begin" in str(excinfo.value)


def test_documented_lint_command_is_immune_to_the_generic_ruff_decoy() -> None:
    """(d) Only the MARKED span is returned, never the generic instruction.

    CONTRIBUTING.md's quality-gates list carries a deliberately generic
    ``uv run ruff check <touched packages>``. Pinning that to the live config
    would be wrong twice over: it would fail immediately, and "fixing" it would
    destroy a correct, audience-appropriate instruction.
    """
    assert _documented_lint_command(_DECOY_DOC) == _HAPPY_COMMAND


# ---------------------------------------------------------------------------
# The live drift assertion. Everything below reads BOTH committed artifacts
# fresh on every run and never stores its own snapshot of the command — a
# snapshot would just relocate the drift problem into this file.
# ---------------------------------------------------------------------------

_LIVE_LABEL = "the root lint_command's ruff leg (dark-factory-orchestrator.yaml)"
_DOC_LABEL = "the documented Lint bullet (CONTRIBUTING.md)"


def _root_lint_command() -> str:
    return yaml.safe_load(DF_CONFIG_PATH.read_text(encoding="utf-8"))["lint_command"]


def _ruff_segment(cmd: str, label: str) -> str:
    """The ``&&``-chained segment of *cmd* that actually invokes ``ruff check``.

    Uses the production splitter ``verify_cmd.split_top_level_and`` (quote-aware)
    rather than a naive ``str.split('&&')`` — matching ``_ruff_segment`` in
    ``test_root_lint_covers_nonmember_py.py`` and ``test_scripts_module_config.py``.
    Duplicated rather than imported from either sibling: that is this repo's
    stated norm for this helper, and cross-importing a leading-underscore private
    would couple two guards' failure modes for ~10 lines. The genuinely shared
    dependency — the production splitter — IS imported by all of them.

    Extracting the ruff segment FIRST is what keeps the target comparison honest:
    tokenising the whole live chain would read ``&&``, ``python3`` and the
    magicmock checker's own directory arguments as ruff lint targets.

    Works unchanged on the single-segment documented command, so both sides of
    the mirror are tokenised by exactly one code path.
    """
    segments = verify_cmd.split_top_level_and(cmd)
    ruff_segments = [s for s in segments if "ruff check" in s]
    assert len(ruff_segments) == 1, (
        f"expected exactly one `ruff check` segment in {label}, got "
        f"{ruff_segments!r} (task 3558); full command: {cmd!r}"
    )
    return ruff_segments[0].strip()


def _ruff_targets(cmd: str, label: str) -> list[str]:
    """The positional path arguments the ``ruff check`` segment of *cmd* lints.

    Returns whole path TOKENS. Callers must compare against them by exact element
    and never substring-match the raw command — the contract documented on
    ``_lint_leg_targets`` in ``test_fallback_verify_config.py`` and restated on
    ``_ruff_targets`` in ``test_root_lint_covers_nonmember_py.py``.

    That rule is backed by a MEASURED counterexample, not a hypothetical:
    ``'shared' in cmd`` is ALREADY TRUE of the live ``lint_command`` via the
    ``check_bare_magicmock_config.py`` TAIL leg's ``shared/tests`` argument. The
    ruff leg and the tail leg have DISJOINT target lists, so a substring test
    would pass vacuously for a path the ruff leg never checks.
    """
    tokens = shlex.split(_ruff_segment(cmd, label))
    assert "check" in tokens, f"no ruff `check` subcommand in {label}: {cmd!r} (task 3558)"
    tail = tokens[tokens.index("check") + 1:]
    return [t for t in tail if not t.startswith("-")]


def test_contributing_lint_bullet_mirrors_the_live_lint_command() -> None:
    """CONTRIBUTING.md's marked Lint bullet must equal the live ruff leg.

    Reads BOTH sides live from the committed artifacts. The documented string is
    compared against the extracted ruff SEGMENT of ``lint_command``, not against
    the whole ``&&`` chain: the live command's tail leg
    (``python3 fused-memory/scripts/check_bare_magicmock_config.py <test dirs>``)
    has a DISJOINT target list and is a gate-internal style check, not something
    a contributor runs as "the lint command". Forcing the doc to carry it would
    be wrong for its audience.

    MEASURED RED at base HEAD ``ca8dfb6a6e``: (b) reported missing
    ``['sampler', 'cockpit', 'conftest.py', 'df_pytest_isolation.py', 'skills']``
    — 5 documented targets against 10 live — and (c) reported the string
    mismatch. (a) and (d) passed already, which is correct: they are vacuity and
    typo backstops, not the RED.
    """
    live_cmd = _root_lint_command()
    live_segment = _ruff_segment(live_cmd, _LIVE_LABEL)
    live_targets = _ruff_targets(live_cmd, _LIVE_LABEL)
    documented = _documented_lint_command(CONTRIBUTING_PATH.read_text(encoding="utf-8"))

    # (a) NON-VACUITY, both sides. Neither an empty target list nor an empty
    # extraction may let this invariant pass by checking nothing at all.
    assert live_targets, (
        f"{_LIVE_LABEL} had no positional targets at all (task 3558) — this "
        f"mirror invariant would pass vacuously; command: {live_cmd!r}"
    )
    assert documented, (
        f"{_DOC_LABEL} extracted as empty (task 3558) — this mirror invariant "
        f"would pass vacuously"
    )
    assert shlex.split(documented)[:4] == ["uv", "run", "ruff", "check"], (
        f"{_DOC_LABEL} does not lead with `uv run ruff check` (task 3558): "
        f"{documented!r}. The marker must wrap the copy-pasteable lint command; "
        f"if the bullet was repurposed, move the marker to the command it names."
    )

    documented_targets = _ruff_targets(documented, _DOC_LABEL)

    # (b) SEMANTIC — whole-token set equality, element-wise, NEVER substring
    # (see _ruff_targets for the measured reason).
    missing = [t for t in live_targets if t not in documented_targets]
    extra = [t for t in documented_targets if t not in live_targets]
    assert not missing and not extra, (
        f"CONTRIBUTING.md's Lint bullet has drifted from "
        f"dark-factory-orchestrator.yaml's lint_command (task 3558).\n"
        f"  MISSING from the doc (linted by the gate, not by the documented "
        f"command): {missing}\n"
        f"  EXTRA in the doc (documented but not linted by the gate): {extra}\n"
        f"  documented: {documented_targets}\n"
        f"  live:       {live_targets}\n"
        f"UNDER-COVERAGE is the failure this guard exists to stop: a contributor "
        f"who runs the documented command gets a clean LOCAL GREEN over a strict "
        f"subset of what the gate checks, then eats a RED merge verify on a "
        f"branch they believed was ready. If you widened the yaml head, update "
        f"the bullet inside the `lint-command-mirror` marker in CONTRIBUTING.md "
        f"to match."
    )

    # (c) EXACT. Catches flag and spelling drift a set comparison alone would
    # miss — e.g. a future `--select` added to the live head.
    assert documented == live_segment, (
        f"CONTRIBUTING.md's Lint bullet is not verbatim equal to the ruff leg of "
        f"lint_command (task 3558), even though their target SETS agree — so the "
        f"difference is in flags, spelling or spacing.\n"
        f"  documented: {documented!r}\n"
        f"  live:       {live_segment!r}\n"
        f"Copy the live segment into the `lint-command-mirror` marker verbatim. "
        f"Note the doc deliberately mirrors the ruff leg ONLY, never the "
        f"`&& python3 ...check_bare_magicmock_config.py` tail."
    )

    # (d) STALE-TARGET / typo guard. A bogus documented target is invisible to
    # (b) and (c) once the yaml carries the same typo, and it would instruct a
    # contributor to run a command that exits non-zero on a path that is not
    # there. (Mirrors assertion (c) of
    # test_root_lint_command_targets_every_root_level_and_skills_py.)
    for target in documented_targets:
        assert (REPO_ROOT / target).exists(), (
            f"{_DOC_LABEL} names {target!r}, which does not exist under "
            f"{REPO_ROOT} (task 3558) — the doc would instruct a contributor to "
            f"run a command that exits non-zero; targets: {documented_targets}"
        )
