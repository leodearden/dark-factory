"""Mirror contract: CONTRIBUTING.md's Lint bullet restates the live ``lint_command``.

Task 3558. CONTRIBUTING.md documents the ``ruff check`` leg of
``dark-factory-orchestrator.yaml``'s ``lint_command`` verbatim, inside a
``lint-command-mirror`` HTML-comment marker. This guard pins the two together.

WHY A MACHINE CHECK. The doc drifted twice while the config grew — task 3397
added ``sampler`` and ``cockpit``, task 3485 added ``conftest.py``,
``df_pytest_isolation.py`` and ``skills``. MEASURED RED at base HEAD
``ca8dfb6a6e``: 5 documented targets against 10 live, missing exactly
``['sampler', 'cockpit', 'conftest.py', 'df_pytest_isolation.py', 'skills']``.
The failure mode is not cosmetic — a contributor who runs the documented command
gets a clean local green over a strict SUBSET of what the merge gate checks, then
eats a red verify. Both drifts landed while CONTRIBUTING.md ALREADY told the
reader to treat the yaml as the source of truth, so it is measured, not
predicted, that prose pointing at prose does not hold.

The doc mirrors the ruff leg ONLY: ``lint_command``'s tail leg
(``check_bare_magicmock_config.py`` over each package's ``tests/``) has a
disjoint target list and is a gate-internal style check. CONTRIBUTING.md names it
outside the marker, unpinned.

This is the DOCS half of the ``lint_command`` contract ("does the prose still say
what the config says?"); ``test_root_lint_covers_nonmember_py.py`` (task 3485) is
the FILES half ("does the config still target every file that needs it?").
Neither implies the other.

PLACEMENT IS LOAD-BEARING. ``tests/scripts/`` carries its own module config, so
this guard actually runs under FULL_SUITE and merge-role
``merge_verify_breadth: full``.

OUT OF SCOPE. ``_ROOT_LINT_COMMAND`` (orchestrator/tests/test_verify_cmd.py,
test_verify_plan.py) and ``_FLEET_LINT_COMMAND_OPAQUE``
(orchestrator/tests/test_verify_scope_kappa.py) are deliberately decoupled
five-member SHAPE fixtures, not mirrors of the config; pinning them would
silently change what they test. This guard covers CONTRIBUTING.md only, and
asserts equality between two committed command STRINGS — it makes no assertion
about surrounding prose, and must not be extended to.
"""
from __future__ import annotations

import pathlib
import re
import shlex

import pytest
import yaml

from orchestrator import verify_cmd

REPO_ROOT = pathlib.Path(__file__).parents[2]

# The canonical root config filename. Loading it by this exact path means the
# guard fails loudly if the config is renamed or lint_command disappears, rather
# than quietly guarding nothing.
DF_CONFIG_PATH = REPO_ROOT / "dark-factory-orchestrator.yaml"
CONTRIBUTING_PATH = REPO_ROOT / "CONTRIBUTING.md"

# The HTML-comment marker anchoring extraction, wrapped around the Lint bullet.
# An explicit marker rather than "the span matching `uv run ruff check`" because
# CONTRIBUTING.md carries a SECOND such span in its quality-gates list —
# `uv run ruff check <touched packages>` — which is deliberately generic and must
# never be pinned; any "first matching span" extractor would silently re-target
# on a doc reorder. The marker also puts the tie AT THE EDIT SITE, which is the
# root cause this task closes: nothing tied the prose to the config.
MIRROR_BEGIN = "lint-command-mirror:begin"
MIRROR_END = "lint-command-mirror:end"

# The bullet's own shape, anchored on its label. Anchoring on the label (rather
# than on "a backtick span in the marker slice") is what keeps the begin
# comment's own inline-code prose — the real marker cites `ruff check` and
# `lint_command` — out of the extraction.
_MARKED_LINT_COMMAND = re.compile(r"- \*\*Lint\*\*: `([^`]+)`")


def _documented_lint_command(markdown_text: str) -> str:
    """The inline-code command on the Lint bullet delimited by the mirror markers.

    Every failure is a loud ``AssertionError`` naming the marker literal and
    CONTRIBUTING.md, never a ``''``/``None`` return. That is the vacuity hazard
    and the whole point: an extractor that silently yields nothing turns the drift
    assertion green while pinning nothing — strictly worse than no guard, because
    the check still reports success.

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
        f"expected exactly one {MIRROR_END!r} marker to close {MIRROR_BEGIN!r} in "
        f"CONTRIBUTING.md, found {end_count} (task 3558) — restore the closing "
        f"marker below the Lint bullet"
    )

    # Inverted markers yield an empty slice, so the next assertion catches that
    # too, loudly and with the same remedy.
    marked = markdown_text[markdown_text.index(MIRROR_BEGIN):markdown_text.index(MIRROR_END)]
    spans: list[str] = _MARKED_LINT_COMMAND.findall(marked)
    assert len(spans) == 1, (
        f"expected exactly one ``- **Lint**: `<command>``` bullet between "
        f"{MIRROR_BEGIN!r} and {MIRROR_END!r} in CONTRIBUTING.md, found "
        f"{len(spans)}: {spans!r} (task 3558). The marker must wrap that bullet "
        f"and nothing else; if the bullet was relabelled or the markers were "
        f"inverted, move the marker back around the copy-pasteable lint command."
    )

    command = spans[0].strip()
    assert command, (
        f"the command between {MIRROR_BEGIN!r} and {MIRROR_END!r} in "
        f"CONTRIBUTING.md is empty (task 3558)"
    )
    return command


# Extractor fixtures are hand-written markdown, never the real CONTRIBUTING.md,
# so they stay stable under any future edit to that file's content. They spell
# the marker literals out in full rather than interpolating the constants above:
# a rename must not be able to silently keep a broken parser agreeing with its
# own fixtures.

# (a) Happy path, modelled on the real block: a fenced ```bash example ABOVE the
# marker, and — the measured hazard — TWO inline-code spans inside the begin
# comment's own prose, which an extractor keyed on backticks alone would return.
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

# (b) No marker at all — deleted, or the section renamed. The doc still CONTAINS
# a plausible-looking lint bullet: the extractor must not fall back to "find
# something that looks right".
_NO_MARKER_DOC = """\
- **Lint**: `uv run ruff check alpha beta`

Then, before submitting:

2. `uv run ruff check <touched packages>`.
"""

# (c) Two marker blocks — e.g. a section duplicated in a bad merge. Picking the
# first silently pins one mirror and lets the other rot unwatched.
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
# appears BOTH before and after the marker block, so neither a "first span" nor a
# "last span" heuristic passes by accident. A MEASURED hazard, not hypothetical:
# CONTRIBUTING.md really carries that span, and it must stay generic and unpinned.
_DECOY_DOC = """\
1. `uv run ruff check <touched packages>` — before you start.

<!-- lint-command-mirror:begin
     Mirrors the `ruff check` leg of `lint_command`. -->
- **Lint**: `uv run ruff check alpha beta gamma.py`
<!-- lint-command-mirror:end -->

2. `uv run ruff check <touched packages>` — and again before submitting.
"""


def test_documented_lint_command_extracts_the_marked_span() -> None:
    """(a) Only the marked bullet's command is returned, backticks stripped.

    Also the specific failure of an extractor keyed on backticks alone: it would
    return ``ruff check`` — the first span of the begin comment's explanatory
    prose — which is a plausible-looking string, so the mistake would not
    announce itself.
    """
    assert _documented_lint_command(_HAPPY_DOC) == _HAPPY_COMMAND


@pytest.mark.parametrize(
    ("markdown_text", "case"),
    [
        (_NO_MARKER_DOC, "missing"),
        (_DUPLICATE_MARKER_DOC, "duplicated"),
    ],
)
def test_documented_lint_command_fails_loudly_on_a_broken_marker(
    markdown_text: str, case: str
) -> None:
    """(b, c) A missing or duplicated marker RAISES — never '' or None.

    Missing is the vacuity hazard: an extractor that silently returns nothing
    turns every downstream assertion green while pinning nothing at all.
    Duplicated is the same failure one level down: silently taking the first
    leaves the second mirror unpinned and free to drift. The message must tell a
    human what to restore and where.
    """
    with pytest.raises(AssertionError) as excinfo:
        _documented_lint_command(markdown_text)

    message = str(excinfo.value)
    assert MIRROR_BEGIN in message, case
    assert "CONTRIBUTING.md" in message, case


def test_documented_lint_command_is_immune_to_the_generic_ruff_decoy() -> None:
    """(d) The generic ``<touched packages>`` instruction is never extracted.

    Pinning it to the live config would be wrong twice over: it would fail
    immediately, and "fixing" it would destroy a correct, audience-appropriate
    instruction.
    """
    assert _documented_lint_command(_DECOY_DOC) == _HAPPY_COMMAND


# ---------------------------------------------------------------------------
# The live drift assertion. Everything below reads BOTH committed artifacts
# fresh on every run and never stores its own snapshot of the command — a
# snapshot would just relocate the drift problem into this file.
# ---------------------------------------------------------------------------

_LIVE_LABEL = "the root lint_command's ruff leg (dark-factory-orchestrator.yaml)"
_DOC_LABEL = "the documented Lint bullet (CONTRIBUTING.md)"

# Ruff flags that consume the FOLLOWING token as their value. Without this,
# `--select E,F` would read `E,F` as a lint TARGET and assertion (d) below would
# fail with "names 'E,F', which does not exist" — a red merge verify with a
# misleading diagnosis, on a change that did not break the mirror at all.
# `--flag=value` spellings need no entry: shlex keeps them as one token, and the
# leading `-` already excludes them.
_RUFF_FLAGS_TAKING_A_VALUE = frozenset(
    {
        "--config",
        "--exclude",
        "--extend-exclude",
        "--extend-ignore",
        "--extend-select",
        "--ignore",
        "--line-length",
        "--output-format",
        "--per-file-ignores",
        "--select",
        "--target-version",
    }
)


def _root_lint_command() -> str:
    return yaml.safe_load(DF_CONFIG_PATH.read_text(encoding="utf-8"))["lint_command"]


def _ruff_segment(cmd: str, label: str) -> str:
    """The ``&&``-chained segment of *cmd* that actually invokes ``ruff check``.

    Uses the production splitter ``verify_cmd.split_top_level_and`` (quote-aware)
    rather than a naive ``str.split('&&')``, matching ``_ruff_segment`` in
    ``test_root_lint_covers_nonmember_py.py`` and ``test_scripts_module_config.py``.

    Extracting the ruff segment FIRST is what keeps the target comparison honest:
    tokenising the whole live chain would read ``&&``, ``python3`` and the
    magicmock checker's own directory arguments as ruff lint targets. Works
    unchanged on the single-segment documented command, so both sides of the
    mirror are tokenised by exactly one code path.
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

    targets: list[str] = []
    consume_next = False
    for token in tokens[tokens.index("check") + 1:]:
        if consume_next:
            consume_next = False
            continue
        if token.startswith("-"):
            consume_next = token in _RUFF_FLAGS_TAKING_A_VALUE
            continue
        targets.append(token)
    return targets


def test_ruff_targets_reads_flag_values_as_flags_not_paths() -> None:
    """A space-separated flag VALUE is a flag's value, not a lint target.

    Guards the diagnosis quality of the live assertions below: without this, a
    future ``--select E,F`` on the ruff head would surface as "the doc names
    'E,F', which does not exist under <repo root>" — pointing a reader at a
    nonexistent path problem instead of at the (correctly mirrored) flag.
    """
    targets = _ruff_targets(
        "uv run ruff check --select E,F --line-length 100 --fix alpha beta.py",
        "a synthetic command",
    )
    assert targets == ["alpha", "beta.py"]


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
