"""Executes the two marked commands in CLAUDE.md's ``### Anchoring ad-hoc paths``.

Task 5690, codebook entry ``entry-cand-20260916-18``; a tier-1 INV-10
``guards-exercise-behaviour`` guard. Rewording that subsection must leave this
green, and breaking either marked command must turn it red.
"""
from __future__ import annotations

import pathlib
import re
import shlex
import subprocess

import pytest
import verify_command_invariants as vci

REPO_ROOT = pathlib.Path(__file__).parents[2]

CLAUDE_MD_PATH = REPO_ROOT / "CLAUDE.md"

ANCHOR_MARKER = "checkout-root-anchor"
ANCHOR_LABEL = "Checkout root"

PROBE_MARKER = "anchored-probe-idiom"
PROBE_LABEL = "Anchor an ad-hoc probe"

_ANCHOR_ARGV = ("git", "rev-parse", "--show-toplevel")

_ANCHOR_PREFIX = 'cd "$(git rev-parse --show-toplevel)" &&'

_PROBE_ARGV0 = ("python", "python3")

_FIXTURE_FIRST_LINE = "# fixture-root-marker-5690 do-not-match-by-accident"

_RUN_TIMEOUT_SECS = 60

_GIT_IDENTITY = (
    "-c", "user.email=guard@example.invalid",
    "-c", "user.name=Guard",
    "-c", "commit.gpgsign=false",
)


def _marked_command(markdown_text, marker, bullet_label):
    """The command on the *bullet_label* bullet between *marker*'s begin/end comments.

    ``verify_command_invariants.marked_span`` owns the loud marker checks. This
    guard supplies only the pattern, keyed on the bullet's label rather than on
    backticks so that inline code in the begin comment is never extracted.
    """
    bullet = re.compile(r"- \*\*" + re.escape(bullet_label) + r"\*\*: `([^`]+)`")
    return vci.marked_span(
        markdown_text,
        begin=f"{marker}:begin",
        end=f"{marker}:end",
        pattern=bullet,
        what=f"`- **{bullet_label}**: <command>` bullet",
        source="CLAUDE.md",
        label=f"the {bullet_label!r} command",
        task="5690",
    ).strip()


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
    assert _marked_command(_HAPPY_DOC, ANCHOR_MARKER, ANCHOR_LABEL) == _HAPPY_COMMAND


@pytest.mark.parametrize(
    ("markdown_text", "case"),
    [
        (_NO_MARKER_DOC, "missing"),
        (_DUPLICATE_MARKER_DOC, "duplicated"),
    ],
)
def test_marked_command_fails_loudly_on_a_broken_marker(markdown_text, case):
    """It raises: an empty return would make every execution check below vacuous."""
    with pytest.raises(AssertionError) as excinfo:
        _marked_command(markdown_text, ANCHOR_MARKER, ANCHOR_LABEL)

    message = str(excinfo.value)
    assert ANCHOR_MARKER in message, case
    assert "CLAUDE.md" in message, case


def test_marked_command_is_immune_to_inline_code_in_the_begin_comment():
    assert _marked_command(_DECOY_DOC, ANCHOR_MARKER, ANCHOR_LABEL) == _HAPPY_COMMAND


def _live_anchor_argv():
    command = _marked_command(
        CLAUDE_MD_PATH.read_text(encoding="utf-8"), ANCHOR_MARKER, ANCHOR_LABEL
    )
    argv = tuple(shlex.split(command))
    assert argv == _ANCHOR_ARGV, (
        f"{ANCHOR_MARKER!r} in CLAUDE.md must be the checkout-root query "
        f"{shlex.join(_ANCHOR_ARGV)!r}; found {command!r}"
    )
    return list(argv)


def test_live_claude_md_documents_the_checkout_root_anchor():
    """Reports a recipe degraded into ``pwd`` or a fixed path as one shape failure."""
    assert _live_anchor_argv() == list(_ANCHOR_ARGV)


def _run(argv, cwd, marker=ANCHOR_MARKER):
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
            f"{marker!r} command from CLAUDE.md did not finish within "
            f"{_RUN_TIMEOUT_SECS}s; argv {argv!r}, cwd {cwd}"
        )


def _describe(completed):
    return (
        f"exit {completed.returncode}, stdout {completed.stdout.strip()!r}, "
        f"stderr {completed.stderr.strip()!r}"
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
        f"fixture setup: git -C {root} {' '.join(args)}: {_describe(completed)}"
    )
    return completed


def _seeded_checkout(parent, name="checkout"):
    """A checkout at *parent*/*name* holding one commit, and its ``pkg/tests`` subdir."""
    root = parent / name
    (root / "pkg" / "tests").mkdir(parents=True)
    _git(root.parent, "init", "-q", "-b", "main", str(root))
    (root / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git(root, "add", "seed.txt")
    _git(root, *_GIT_IDENTITY, "commit", "-q", "-m", "seed")
    return root, root / "pkg" / "tests"


def test_documented_anchor_answers_with_the_checkout_root_not_cwd(tmp_path):
    """Both sides are ``resolve()``d, so a symlinked tmpdir cannot fail it spuriously."""
    argv = _live_anchor_argv()
    root, nested = _seeded_checkout(tmp_path)

    completed = _run(argv, nested)

    assert completed.returncode == 0, (
        f"{ANCHOR_MARKER!r} run from {nested}: {_describe(completed)}"
    )
    printed = pathlib.Path(completed.stdout.strip()).resolve()
    assert printed == root.resolve(), (
        f"{ANCHOR_MARKER!r} printed {printed} from {nested}; "
        f"expected the checkout root {root.resolve()}"
    )
    assert printed != nested.resolve(), f"{ANCHOR_MARKER!r} echoed its cwd {printed}"


def test_documented_anchor_answers_with_the_linked_worktrees_own_root(tmp_path):
    argv = _live_anchor_argv()
    main_root, _ = _seeded_checkout(tmp_path, name="main")
    worktree_root = tmp_path / "wt"
    _git(main_root, "worktree", "add", "-q", "-b", "task-5690", str(worktree_root), "main")
    nested = worktree_root / "pkg" / "tests"
    nested.mkdir(parents=True)

    completed = _run(argv, nested)

    assert completed.returncode == 0, (
        f"{ANCHOR_MARKER!r} run from {nested}: {_describe(completed)}"
    )
    printed = pathlib.Path(completed.stdout.strip()).resolve()
    assert printed == worktree_root.resolve(), (
        f"{ANCHOR_MARKER!r} printed {printed} from {nested}; "
        f"expected the linked worktree's own root {worktree_root.resolve()}"
    )
    assert printed != main_root.resolve(), (
        f"{ANCHOR_MARKER!r} printed the main checkout {printed} from a linked worktree"
    )


def test_documented_anchor_refuses_loudly_outside_any_checkout(tmp_path):
    """No real checkout can answer: ``df_pytest_isolation`` caps git discovery at basetemp."""
    argv = _live_anchor_argv()
    outside = tmp_path / "outside"
    outside.mkdir()

    completed = _run(argv, outside)

    assert completed.returncode != 0, (
        f"{ANCHOR_MARKER!r} succeeded outside any checkout: {_describe(completed)}"
    )
    assert completed.stdout.strip() == "", (
        f"{ANCHOR_MARKER!r} failed but still printed a root: {_describe(completed)}"
    )
    assert completed.stderr.strip() != "", (
        f"{ANCHOR_MARKER!r} failed silently: {_describe(completed)}"
    )


def _live_probe_command():
    """A string rather than argv: ``$(...)`` and ``&&`` need a shell."""
    command = _marked_command(
        CLAUDE_MD_PATH.read_text(encoding="utf-8"), PROBE_MARKER, PROBE_LABEL
    )
    assert command.startswith(_ANCHOR_PREFIX), (
        f"{PROBE_MARKER!r} in CLAUDE.md must start with the anchor "
        f"{_ANCHOR_PREFIX!r}; found {command!r}"
    )

    remainder = command[len(_ANCHOR_PREFIX):].strip()
    argv = shlex.split(remainder)
    assert len(argv) == 3 and argv[0] in _PROBE_ARGV0 and argv[1] == "-c", (
        f"{PROBE_MARKER!r} in CLAUDE.md must anchor a `python3 -c` probe; "
        f"found {remainder!r}"
    )
    return command


def test_live_claude_md_documents_the_anchored_probe_idiom():
    """Reports an unanchored or hard-coded probe as one shape failure."""
    assert _live_probe_command().startswith(_ANCHOR_PREFIX)


def _seeded_checkout_with_root_claude_md(tmp_path):
    """A checkout whose ROOT alone holds a CLAUDE.md, so only an anchored read finds one."""
    root, nested = _seeded_checkout(tmp_path)
    (root / "CLAUDE.md").write_text(f"{_FIXTURE_FIRST_LINE}\nsecond line\n", encoding="utf-8")
    assert not (nested / "CLAUDE.md").exists(), (
        f"fixture invariant: {nested} must hold no CLAUDE.md of its own"
    )
    return root, nested


def test_documented_probe_idiom_reads_relative_to_the_checkout_root(tmp_path):
    command = _live_probe_command()
    _root, nested = _seeded_checkout_with_root_claude_md(tmp_path)

    completed = _run(["bash", "-c", command], nested, marker=PROBE_MARKER)

    assert completed.returncode == 0, (
        f"{PROBE_MARKER!r} run from {nested}: {_describe(completed)}"
    )
    assert _FIXTURE_FIRST_LINE in completed.stdout, (
        f"{PROBE_MARKER!r} did not print the root CLAUDE.md's first line "
        f"{_FIXTURE_FIRST_LINE!r}: {_describe(completed)}"
    )


def test_stripping_the_documented_anchor_reproduces_the_sightings_error(tmp_path):
    """The control: the same command minus its prefix fails the way the sighting did."""
    command = _live_probe_command()
    unanchored = command[len(_ANCHOR_PREFIX):].strip()
    _root, nested = _seeded_checkout_with_root_claude_md(tmp_path)

    completed = _run(["bash", "-c", unanchored], nested, marker=PROBE_MARKER)

    assert completed.returncode != 0, (
        f"{PROBE_MARKER!r} without its anchor still succeeded from {nested}: "
        f"{_describe(completed)}"
    )
    assert "FileNotFoundError" in completed.stderr, (
        f"{PROBE_MARKER!r} without its anchor failed, but not with the sighting's "
        f"FileNotFoundError: {_describe(completed)}"
    )
