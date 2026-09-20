"""Behavioral contract: CLAUDE.md's checkout-root anchor recipe actually works.

Task 5690. Confusion-codebook entry ``entry-cand-20260916-18``, sighting
2026-09-16 (session ``bc5a9c30``, task 3895 implementer): a ``python3 -``
heredoc died with ``FileNotFoundError: [Errno 2] No such file or directory:
'test_bake_off_storage_shape.py'``.

THE MECHANISM IS NOT "THE AGENT FORGOT A DIRECTORY". Per
``plans/confusion-census-2026-09-20.md`` §1.4, derived from the archived
transcript: the harness tracks a Bash working directory that PERSISTS across
calls and is mutated by any earlier ``cd``; two adjacent calls in that one
session carried DIFFERENT tracked cwds (``.worktrees/3895`` and
``.../fused-memory/tests``). "The effective cwd is therefore not readable from
the model's command text, which is the condition under which a bare filename is
a guess." The same shape recurs across roughly twenty codebook entries, so the
class is well past one-off.

WHY A MACHINE CHECK FOR A DOCUMENTATION FIX. ``docs/legibility/
design-invariants.md`` INV-10 ``guards-exercise-behaviour`` ranks "EXECUTE the
documented thing" tier 1 of its ladder and names
``tests/scripts/test_package_source_lookup_convention.py`` (task 3959) as the
house instance; a substring or regex over prose is explicitly "a finding, not a
guard". Prose only reaches agents who read it — this file goes one step past a
string mirror by RUNNING the command CLAUDE.md hands them, against fixtures it
seeds itself. A recipe that no longer works fails here instead of failing an
agent mid-task.

WHAT THIS FILE DELIBERATELY DOES NOT DO. It asserts nothing about prose,
wording, headings or ordering. Fully rewording ``### Anchoring ad-hoc paths``
while keeping the marked commands intact must leave this green; breaking either
command must turn it red.

THE EXTRACTOR IS RE-IMPLEMENTED RATHER THAN IMPORTED from its two siblings in
this directory, per the no-cross-import-between-guards convention recorded in
``tests/scripts/conftest.py`` — task 3959 copied it from task 3558, and task
5330 from 3959, for the same reason.

PLACEMENT IS LOAD-BEARING. ``tests/scripts/`` carries its own module config, so
this guard runs under FULL_SUITE and merge-role ``merge_verify_breadth: full``.
"""
from __future__ import annotations

import pathlib
import shlex
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).parents[2]

CLAUDE_MD_PATH = REPO_ROOT / "CLAUDE.md"

ANCHOR_MARKER = "checkout-root-anchor"
ANCHOR_LABEL = "Checkout root"

_ANCHOR_ARGV = ("git", "rev-parse", "--show-toplevel")

_RUN_TIMEOUT_SECS = 60

_GIT_IDENTITY = (
    "-c", "user.email=guard@example.invalid",
    "-c", "user.name=Guard",
    "-c", "commit.gpgsign=false",
)


def _marked_command(markdown_text, marker, bullet_label):
    """The inline-code command on the *bullet_label* bullet delimited by *marker*.

    Every failure is a loud ``AssertionError`` naming the marker literal and
    CLAUDE.md, never a ``''``/``None`` return: an extractor that silently
    yields nothing turns every execution assertion below vacuously green while
    running nothing at all, which is strictly worse than no guard because the
    check still reports success. The span is anchored on the LABEL prefix
    rather than on backticks, because keying on backticks alone would extract
    the begin comment's own explanatory inline code — a plausible-looking
    string, so the mistake would not announce itself.
    """
    begin = f"{marker}:begin"
    end = f"{marker}:end"

    begin_count = markdown_text.count(begin)
    assert begin_count == 1, (
        f"expected exactly one {begin!r} marker, found {begin_count} (task 5690). "
        f"It delimits a copy-pasteable command in CLAUDE.md's `### Anchoring "
        f"ad-hoc paths` section. If it was deleted, restore it around that "
        f"bullet; if it was duplicated, one of the two copies is unpinned and "
        f"free to rot into a command that no longer runs."
    )
    end_count = markdown_text.count(end)
    assert end_count == 1, (
        f"expected exactly one {end!r} marker to close {begin!r} in CLAUDE.md, "
        f"found {end_count} (task 5690) — restore the closing marker below the "
        f"bullet it wraps"
    )

    # Inverted markers yield an empty slice, so the next assertion catches that
    # too, loudly and with the same remedy.
    marked = markdown_text[markdown_text.index(begin):markdown_text.index(end)]
    prefix = f"- **{bullet_label}**: `"
    spans = [
        segment.split("`", 1)[0]
        for segment in marked.split(prefix)[1:]
        if "`" in segment
    ]
    assert len(spans) == 1, (
        f"expected exactly one ``{prefix}<command>``` bullet between {begin!r} "
        f"and {end!r} in CLAUDE.md, found {len(spans)}: {spans!r} (task 5690). "
        f"The marker must wrap that bullet and nothing else; if the bullet was "
        f"relabelled or the markers were inverted, move the marker back around "
        f"the copy-pasteable command."
    )

    command = spans[0].strip()
    assert command, (
        f"the command between {begin!r} and {end!r} in CLAUDE.md is empty (task 5690)"
    )
    return command


_HAPPY_DOC = """\
Your cwd is not derivable from your command text. Ask git; never assume you are
still where you last were.

<!-- checkout-root-anchor:begin
     EXECUTED verbatim by
     tests/scripts/test_checkout_root_anchor_convention.py from a subdirectory
     of a temp checkout, asserting it answers with the checkout ROOT and not
     with cwd. Edit it into `pwd` or a hard-coded path and that guard goes
     red — which is the point. -->
- **Checkout root**: `git rev-parse --show-toplevel`
<!-- checkout-root-anchor:end -->

It refuses loudly rather than handing back a path that is silently wrong.
"""

_HAPPY_COMMAND = "git rev-parse --show-toplevel"

_NO_MARKER_DOC = """\
- **Checkout root**: `git rev-parse --show-toplevel`

Somewhere else entirely, an unmarked mention of `git worktree list`.
"""

_DUPLICATE_MARKER_DOC = """\
<!-- checkout-root-anchor:begin -->
- **Checkout root**: `git rev-parse --show-toplevel`
<!-- checkout-root-anchor:end -->

## Some later section

<!-- checkout-root-anchor:begin -->
- **Checkout root**: `pwd`
<!-- checkout-root-anchor:end -->
"""

_DECOY_DOC = """\
The Bash working directory persists across calls and an earlier `cd` moved it.

<!-- checkout-root-anchor:begin
     Mirrors nothing; it is EXECUTED as written by
     tests/scripts/test_checkout_root_anchor_convention.py. Contains prose
     inline code such as `pwd` and `git worktree list | head -1` that a
     backtick-keyed extractor would grab first. -->
- **Checkout root**: `git rev-parse --show-toplevel`
<!-- checkout-root-anchor:end -->

Afterwards, every relative path is measured from the root it printed.
"""


def test_marked_command_extracts_the_marked_span():
    """Only the marked bullet's command is returned, backticks stripped."""
    assert _marked_command(_HAPPY_DOC, ANCHOR_MARKER, ANCHOR_LABEL) == _HAPPY_COMMAND


@pytest.mark.parametrize(
    ("markdown_text", "case"),
    [
        (_NO_MARKER_DOC, "missing"),
        (_DUPLICATE_MARKER_DOC, "duplicated"),
    ],
)
def test_marked_command_fails_loudly_on_a_broken_marker(markdown_text, case):
    """A missing or duplicated marker RAISES — never '' or None.

    Missing is the vacuity hazard: an extractor that silently returns nothing
    turns every downstream execution assertion green while executing nothing at
    all. Duplicated is the same failure one level down — silently taking the
    first leaves the second copy unexecuted and free to rot.
    """
    with pytest.raises(AssertionError) as excinfo:
        _marked_command(markdown_text, ANCHOR_MARKER, ANCHOR_LABEL)

    message = str(excinfo.value)
    assert ANCHOR_MARKER in message, case
    assert "CLAUDE.md" in message, case


def test_marked_command_is_immune_to_inline_code_in_the_begin_comment():
    """Prose inline code inside the marked slice is never extracted.

    An extractor keyed on "the first backtick span after the begin marker"
    would return ``pwd`` here — a plausible-looking string, so the mistake
    would survive review and then be executed as a command, silently answering
    the cwd question this recipe exists to stop an agent from asking.
    """
    assert _marked_command(_DECOY_DOC, ANCHOR_MARKER, ANCHOR_LABEL) == _HAPPY_COMMAND


def _live_anchor_argv():
    """The live CLAUDE.md's anchor command, shape-checked and tokenised.

    The SHAPE assertion is load-bearing: a recipe degraded into ``pwd``,
    ``echo $PWD`` or a hard-coded absolute path would answer the WRONG question
    (where am I, rather than where is the checkout) and would sail through an
    exit-0 execution check.
    """
    command = _marked_command(
        CLAUDE_MD_PATH.read_text(encoding="utf-8"),
        ANCHOR_MARKER,
        ANCHOR_LABEL,
    )
    argv = tuple(shlex.split(command))
    assert argv == _ANCHOR_ARGV, (
        f"the command inside the {ANCHOR_MARKER!r} marker in CLAUDE.md is not "
        f"the checkout-root query (task 5690): {command!r} tokenises to {argv!r}, "
        f"expected {list(_ANCHOR_ARGV)!r}. The recipe has to answer 'where is "
        f"the checkout root', independent of the directory it runs in — `pwd`, "
        f"`echo $PWD` and a hard-coded path all answer a different question and "
        f"would still exit 0."
    )
    return list(argv)


def test_live_claude_md_documents_the_checkout_root_anchor():
    """The live CLAUDE.md carries the marker, and it wraps the right command.

    Named separately from the execution checks below so that a recipe degraded
    into ``pwd`` reports itself as a SHAPE failure rather than only as three
    execution failures whose common cause a reader has to reconstruct.
    """
    assert _live_anchor_argv() == list(_ANCHOR_ARGV)


def _run(argv, cwd):
    """*argv* under *cwd*, with a timeout that fails the test rather than hanging."""
    try:
        return subprocess.run(
            argv,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=_RUN_TIMEOUT_SECS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"the {ANCHOR_MARKER!r} command documented in CLAUDE.md did not finish "
            f"within {_RUN_TIMEOUT_SECS}s (task 5690); argv: {argv!r}, cwd: {cwd}"
        )


def _git(root, *args):
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        timeout=_RUN_TIMEOUT_SECS,
        check=False,
    )
    assert completed.returncode == 0, (
        f"fixture setup failed: git -C {root} {' '.join(args)} exited "
        f"{completed.returncode}: {completed.stderr.strip()!r}"
    )
    return completed


def _seeded_checkout(parent, name="checkout"):
    """A git checkout at *parent*/*name* holding one commit, and its nested subdir.

    ``df_pytest_isolation``'s session-scoped autouse ``GIT_CEILING_DIRECTORIES``
    fixture pins the running basetemp as git's discovery ceiling, so none of
    these calls can walk up into a live checkout.
    """
    root = parent / name
    (root / "pkg" / "tests").mkdir(parents=True)
    _git(root.parent, "init", "-q", "-b", "main", str(root))
    (root / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git(root, "add", "seed.txt")
    _git(root, *_GIT_IDENTITY, "commit", "-q", "-m", "seed")
    return root, root / "pkg" / "tests"


def test_documented_anchor_answers_with_the_checkout_root_not_cwd(tmp_path):
    """The anchor is cwd-INDEPENDENT — the property the whole recipe rests on.

    Run from a nested subdirectory, it must still print the checkout root. Both
    sides are ``resolve()``d because a symlinked tmpdir (macOS ``/private/var``,
    and ``/tmp`` symlinks generally) would otherwise produce a spurious failure.
    """
    argv = _live_anchor_argv()
    root, nested = _seeded_checkout(tmp_path)

    completed = _run(argv, nested)

    assert completed.returncode == 0, (
        f"the {ANCHOR_MARKER!r} command documented in CLAUDE.md exited "
        f"{completed.returncode} inside a checkout (task 5690) — agents are being "
        f"handed a recipe that does not work.\n"
        f" argv: {argv!r}\n cwd: {nested}\n"
        f" stdout: {completed.stdout.strip()!r}\n"
        f" stderr: {completed.stderr.strip()!r}"
    )
    printed = pathlib.Path(completed.stdout.strip()).resolve()
    assert printed == root.resolve(), (
        f"the {ANCHOR_MARKER!r} command printed {printed} when run from {nested}, "
        f"expected the checkout root {root.resolve()} (task 5690)"
    )
    assert printed != nested.resolve(), (
        f"the {ANCHOR_MARKER!r} command echoed its own cwd {nested.resolve()} "
        f"(task 5690) — it is documented as cwd-independent, which is the entire "
        f"reason it can anchor a path an agent cannot otherwise place"
    )


def test_documented_anchor_answers_with_the_linked_worktrees_own_root(tmp_path):
    """Inside a linked worktree it answers with THAT worktree, not the main checkout.

    This is the property the factory actually stands on: a task agent's files
    live in its own worktree, so the worktree root is the answer it wants. The
    main-checkout derivation is the different question, owned by
    ``skills/do/SKILL.md``.
    """
    argv = _live_anchor_argv()
    main_root, _ = _seeded_checkout(tmp_path, name="main")
    worktree_root = tmp_path / "wt"
    _git(main_root, "worktree", "add", "-q", "-b", "task-5690", str(worktree_root), "main")
    nested = worktree_root / "pkg" / "tests"
    nested.mkdir(parents=True)

    completed = _run(argv, nested)

    assert completed.returncode == 0, (
        f"the {ANCHOR_MARKER!r} command exited {completed.returncode} inside a "
        f"linked worktree (task 5690); stderr: {completed.stderr.strip()!r}"
    )
    printed = pathlib.Path(completed.stdout.strip()).resolve()
    assert printed == worktree_root.resolve(), (
        f"the {ANCHOR_MARKER!r} command printed {printed} from {nested}, expected "
        f"the linked worktree's own root {worktree_root.resolve()} (task 5690)"
    )
    assert printed != main_root.resolve(), (
        f"the {ANCHOR_MARKER!r} command answered with the MAIN checkout "
        f"{main_root.resolve()} from inside a linked worktree (task 5690) — "
        f"CLAUDE.md documents it as the checkout you are STANDING IN, and a task "
        f"agent's files live in its worktree, not in main"
    )


def test_documented_anchor_refuses_loudly_outside_any_checkout(tmp_path):
    """Outside a checkout it fails loudly instead of handing back a wrong path.

    A recipe that guessed here would be worse than no recipe: the caller would
    anchor every subsequent path against a directory that is not a checkout and
    never learn of it.
    """
    argv = _live_anchor_argv()
    outside = tmp_path / "outside"
    outside.mkdir()

    completed = _run(argv, outside)

    assert completed.returncode != 0, (
        f"the {ANCHOR_MARKER!r} command exited 0 from {outside}, which is inside no "
        f"checkout (task 5690) — it printed {completed.stdout.strip()!r}, and a "
        f"silently-wrong root is the failure mode this recipe exists to replace"
    )
    assert completed.stdout.strip() == "", (
        f"the {ANCHOR_MARKER!r} command failed but still printed "
        f"{completed.stdout.strip()!r} on stdout (task 5690) — a caller "
        f"substituting it into a path would consume that as the root"
    )
    assert completed.stderr.strip() != "", (
        f"the {ANCHOR_MARKER!r} command failed silently outside a checkout "
        f"(task 5690) — the refusal has to say so"
    )
