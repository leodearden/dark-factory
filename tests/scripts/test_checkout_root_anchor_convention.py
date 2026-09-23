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

PLACEMENT IS LOAD-BEARING. ``tests/scripts/`` carries its own module config, so
this guard runs under FULL_SUITE and merge-role ``merge_verify_breadth: full``.
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

# Written into the fixture checkout's ROOT CLAUDE.md, and into no other file:
# the anchored probe can only print it by resolving its relative path against
# the checkout root rather than against the subdirectory it was launched from.
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


def _run(argv, cwd, marker=ANCHOR_MARKER):
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
            f"the {marker!r} command documented in CLAUDE.md did not finish "
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


# ── The anchored-probe idiom ──────────────────────────────────────────────────
#
# Knowing the checkout root does not by itself close the sighting. The call that
# failed was a `python3 - <<'PY'` heredoc opening a file by BARE NAME, and a root
# printed in some earlier turn does not reach inside that heredoc. So CLAUDE.md
# has to hand over the composite — anchor AND probe in one command — and this
# half of the file pins that the documented prefix is what makes the difference,
# by running the same command with it and without it.


def _live_probe_command():
    """The live CLAUDE.md's anchored-probe command, shape-checked.

    Returned as a STRING, not argv: `$(...)` and `&&` need a shell. Same
    extractor as the anchor bullet above — a second marker, not a second
    matcher, so there is nothing new to prove correct.
    """
    command = _marked_command(
        CLAUDE_MD_PATH.read_text(encoding="utf-8"),
        PROBE_MARKER,
        PROBE_LABEL,
    )
    assert command.startswith(_ANCHOR_PREFIX), (
        f"the command inside the {PROBE_MARKER!r} marker in CLAUDE.md does not "
        f"start with {_ANCHOR_PREFIX!r} (task 5690): {command!r}. The idiom's "
        f"whole content is that the anchor travels WITH the probe — an absolute "
        f"path hard-coded in its place works only on the machine it was written "
        f"on, and an unanchored probe is the sighting this subsection closes."
    )

    remainder = command[len(_ANCHOR_PREFIX):].strip()
    argv = shlex.split(remainder)
    assert len(argv) == 3 and argv[0] in _PROBE_ARGV0 and argv[1] == "-c", (
        f"the {PROBE_MARKER!r} command in CLAUDE.md does not anchor a `python3 -c` "
        f"probe (task 5690): {remainder!r} tokenises to {argv!r}. The sighting "
        f"failed inside a python heredoc, so the idiom has to stay demonstrated "
        f"in that language — a `cat` or `head` in its place would not show that "
        f"paths INSIDE the interpreter are what the prefix fixes."
    )
    return command


def test_live_claude_md_documents_the_anchored_probe_idiom():
    """The live CLAUDE.md carries the second marker, wrapping an anchored probe.

    Named separately from the execution checks below for the same reason as its
    counterpart above: an idiom degraded into an unanchored probe or a
    hard-coded absolute path should report itself as a SHAPE failure.
    """
    assert _live_probe_command().startswith(_ANCHOR_PREFIX)


def _seeded_checkout_with_root_claude_md(tmp_path):
    """A checkout whose ROOT holds a marked CLAUDE.md, and a nested subdir without one.

    The subdirectory deliberately holds NO ``CLAUDE.md``: if it did, the
    stripped-prefix control below would pass for the wrong reason — reading the
    subdirectory's copy — and would certify nothing about the anchor.
    """
    root, nested = _seeded_checkout(tmp_path)
    (root / "CLAUDE.md").write_text(f"{_FIXTURE_FIRST_LINE}\nsecond line\n", encoding="utf-8")
    assert not (nested / "CLAUDE.md").exists(), (
        f"fixture invariant broken: {nested} must hold no CLAUDE.md of its own, "
        f"otherwise the stripped-prefix control cannot distinguish an anchored "
        f"read from a cwd-relative one (task 5690)"
    )
    return root, nested


def test_documented_probe_idiom_reads_relative_to_the_checkout_root(tmp_path):
    """Run from a subdirectory, the documented probe still opens a ROOT-relative path."""
    command = _live_probe_command()
    _root, nested = _seeded_checkout_with_root_claude_md(tmp_path)

    completed = _run(["bash", "-c", command], nested, marker=PROBE_MARKER)

    assert completed.returncode == 0, (
        f"the {PROBE_MARKER!r} command documented in CLAUDE.md exited "
        f"{completed.returncode} when run from {nested} (task 5690) — agents are "
        f"being handed an idiom that does not work.\n"
        f" command: {command!r}\n"
        f" stdout: {completed.stdout.strip()!r}\n"
        f" stderr: {completed.stderr.strip()!r}"
    )
    assert _FIXTURE_FIRST_LINE in completed.stdout, (
        f"the {PROBE_MARKER!r} command exited 0 from {nested} without printing "
        f"anything derived from the checkout root's CLAUDE.md (task 5690): "
        f"stdout {completed.stdout.strip()!r}. It is documented as putting every "
        f"path inside it on repo-relative footing, so it has to demonstrably READ "
        f"that file rather than merely succeed."
    )


def test_stripping_the_documented_anchor_reproduces_the_sightings_error(tmp_path):
    """Without the prefix, the SAME command fails the way the sighting did.

    This is what makes the pair a behavioural claim rather than two commands
    that happen to exit 0: the failure is derived from the live CLAUDE.md's own
    text, so it shows the documented prefix — not something incidental about
    the fixture — is what removes the ``FileNotFoundError``.
    """
    command = _live_probe_command()
    unanchored = command[len(_ANCHOR_PREFIX):].strip()
    _root, nested = _seeded_checkout_with_root_claude_md(tmp_path)

    completed = _run(["bash", "-c", unanchored], nested, marker=PROBE_MARKER)

    assert completed.returncode != 0, (
        f"the {PROBE_MARKER!r} command stripped of {_ANCHOR_PREFIX!r} still exited "
        f"0 from {nested} (task 5690), printing {completed.stdout.strip()!r}. The "
        f"anchor is then not load-bearing and the bullet is documenting a prefix "
        f"that buys nothing — either the probe stopped opening a repo-relative "
        f"path, or this fixture's subdirectory acquired a file it must not have."
    )
    assert "FileNotFoundError" in completed.stderr, (
        f"the unanchored {PROBE_MARKER!r} command failed from {nested} with "
        f"something other than the sighting's error (task 5690): "
        f"{completed.stderr.strip()!r}. This subsection exists to stop "
        f"`FileNotFoundError: [Errno 2] No such file or directory`, so the control "
        f"has to reproduce exactly that rather than any nonzero exit."
    )
