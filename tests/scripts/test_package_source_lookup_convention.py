"""Behavioral contract: CLAUDE.md's package-lookup recipes actually resolve a package.

Task 3959. Codebook entry ``entry-cand-20260809-16`` (severity medium, open): an
implement-phase agent in ``.worktrees/3871`` needed uvicorn's default
``timeout_keep_alive`` and ran ``grep -rn 'timeout_keep_alive' $(find / -path
'*/uvicorn/config.py' -not -path '/proc/*')``. Exit 143 — killed at the 2-minute
Bash default, zero information returned. The codebook's stated root cause is "No
surfaced convention for where the venv/site-packages tree lives". A sibling
sighting is the same gap in the other direction: an agent probing a GUESSED path,
``.worktrees/3609/.venv/lib/python3*/site-packages/mem0/memory/``, which did not
exist.

WHY A MACHINE CHECK, for a fix that is mostly documentation. The same reasoning
``test_ruff_format_policy.py`` (task 3441) records: prose only reaches agents who
read it, and silence generates repeat work. This guard goes one step past a
string mirror — it EXECUTES the command CLAUDE.md hands agents, so the recipe is
certified to resolve a real package in this environment rather than merely
certified to still be spelled the same way. There is no config value to mirror
here (unlike task 3558's ``lint_command``), so execution is the only way to tie
the doc to reality.

MEASUREMENTS, stated once here so they are not restated per test. Taken in the
task worktree at HEAD ``446ba24fbc``:

  - ``python -c 'import uvicorn, os; print(os.path.dirname(uvicorn.__file__))'``
    -> ``<venv>/lib/python3.13/site-packages/uvicorn`` in 0.396s (0.274s on a
    warm re-measure). The ``find /`` scan it replaces blew through the 120000ms
    Bash default and returned NOTHING. ~300x, which is why raising the Bash
    timeout is the wrong fix for that sighting.
  - The venv path is NOT derivable from cwd, so it must be asked of the
    interpreter rather than guessed: the main checkout's ``.venv`` is CPython
    3.13.9, several cold-verified ``.worktrees/*/.venv`` are CPython 3.14.0, and
    this task's own worktree had no ``.venv`` at all.
  - A first-party workspace member is installed EDITABLE, so it resolves into a
    checkout's ``src/`` tree rather than into ``site-packages``:
    ``import shared`` -> ``<checkout>/shared/src/shared/__init__.py`` in 0.081s.
    That inverse is the whole reason there are two markers and two tests; a
    single assertion covering both would have to be a disjunction that certifies
    neither.

WHY THE PROVENANCE CHECK PROBES ``shared`` AND NOT ``orchestrator``. Measured in
this task's worktree, not assumed. This suite's own ``test_command`` is
``uv run --project shared pytest ...``; when that command has to CREATE the
worktree ``.venv`` it installs shared's closure only — 87 packages, with
``shared`` itself editable and no ``orchestrator``. A subprocess probing
``orchestrator`` there does not fail loudly: the repo root is on ``sys.path``
for a ``python -c``, so the ``orchestrator/`` DIRECTORY is picked up as a
NAMESPACE package and ``orchestrator.__file__`` prints ``None`` — the exact
shadowing hazard the root ``conftest.py`` docstring exists to prevent, which it
handles for the in-process case only (a subprocess inherits none of its
``sys.path`` work). ``shared`` is the one member this command guarantees is
installed as a regular editable package, so probing it keeps a red here meaning
"the documented recipe is wrong" rather than "this venv was provisioned
narrowly". The ``exists()`` assertion below still catches the namespace case
loudly if it ever arises.

WHY ``sys.executable``, AND WHY ``timeout=60`` INSTEAD OF A WALL-CLOCK ASSERT.
Two separate hazards. (1) ``verify._target_subprocess_env`` — cited by SYMBOL,
since its line has already moved once (3473 -> 3609) between this plan's two
revisions — strips ``_VENV_ISOLATION_KEYS`` including ``VIRTUAL_ENV`` and calls
``_strip_venv_bin_from_path`` so a target resolves its own ``.venv``. A test that
shelled out to a bare ``python`` off PATH could therefore hit a system
interpreter with no uvicorn installed and go FALSELY red under verify;
``sys.executable`` is the interpreter already running pytest, correct in the main
checkout (3.13.9) and in a cold-verified worktree (3.14.0) alike. (2) The
invariant worth pinning is "this is a scoped query, not a whole-filesystem scan".
A float duration assert would be flaky on a host ``tests/scripts/
orchestrator.yaml`` already records at a ~39% run-to-run spread, whereas a 60s
subprocess timeout against a measured 0.4s runtime is a ~150x margin and a
``TimeoutExpired`` is an unambiguous, self-explaining red. 60s also sits inside
this suite's own ``--timeout=300`` per-test ceiling, so the subprocess guard
fires first and names the real cause.

WHAT THIS FILE DELIBERATELY DOES NOT DO. It asserts nothing about prose, wording,
headings or ordering in CLAUDE.md or OPERATIONS.md, and must not be extended to:
pinning wording would go red on any future rewording, which is not a defect.
Reword the surrounding section freely — only degrading a marked COMMAND into
something broken, or into something that is not an interpreter query, fails here.

PLACEMENT IS LOAD-BEARING. ``tests/scripts/`` carries its own module config, so
this guard actually runs under FULL_SUITE and merge-role
``merge_verify_breadth: full`` — the same reason recorded on
``test_ruff_format_policy.py`` and ``test_contributing_lint_command_drift.py``.
"""
from __future__ import annotations

import pathlib
import re
import shlex
import subprocess
import sys
import tomllib

import pytest

REPO_ROOT = pathlib.Path(__file__).parents[2]

CLAUDE_MD_PATH = REPO_ROOT / "CLAUDE.md"

PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"

PACKAGE_SOURCE_MARKER = "package-source-lookup"
PACKAGE_SOURCE_LABEL = "Third-party package source"

IMPORT_PROVENANCE_MARKER = "import-provenance-check"
IMPORT_PROVENANCE_LABEL = "First-party import provenance"

# `import alpha, os` / `import alpha` — the module the documented -c body probes.
_IMPORTED_MODULE = re.compile(r"\bimport\s+([A-Za-z_][A-Za-z0-9_]*)")

# The documented recipe must be an INTERPRETER QUERY. This is the shape that
# distinguishes it from the `find /` scan the task exists to retire.
_PYTHON_ARGV0 = ("python", "python3")

# ~150x the measured 0.396s. Not a performance assertion — a scan detector.
_QUERY_TIMEOUT_SECS = 60


def _marked_command(markdown_text, marker, bullet_label):
    """The inline-code command on the *bullet_label* bullet delimited by *marker*.

    Every failure is a loud ``AssertionError`` naming the marker literal and
    CLAUDE.md, never a ``''``/``None`` return. That is the vacuity hazard and the
    whole point: an extractor that silently yields nothing turns every downstream
    assertion green while certifying nothing — strictly worse than no guard,
    because the check still reports success. Copied from
    ``_documented_lint_command`` in ``test_contributing_lint_command_drift.py``
    (task 3558), including its label-anchored span regex: keying on backticks
    alone would extract the begin comment's own explanatory inline code, which is
    a plausible-looking string, so the mistake would not announce itself.
    """
    begin = f"{marker}:begin"
    end = f"{marker}:end"

    begin_count = markdown_text.count(begin)
    assert begin_count == 1, (
        f"expected exactly one {begin!r} marker, found {begin_count} (task 3959). "
        f"This marker delimits the copy-pasteable lookup command in CLAUDE.md's "
        f"`### Locating installed code` section. If it was deleted, restore it "
        f"around that bullet; if it was duplicated, one of the two copies is "
        f"unpinned and free to rot into a command that no longer works."
    )
    end_count = markdown_text.count(end)
    assert end_count == 1, (
        f"expected exactly one {end!r} marker to close {begin!r} in CLAUDE.md, "
        f"found {end_count} (task 3959) — restore the closing marker below the "
        f"bullet it wraps"
    )

    # Inverted markers yield an empty slice, so the next assertion catches that
    # too, loudly and with the same remedy.
    marked = markdown_text[markdown_text.index(begin):markdown_text.index(end)]
    pattern = re.compile(r"- \*\*" + re.escape(bullet_label) + r"\*\*: `([^`]+)`")
    spans = pattern.findall(marked)
    assert len(spans) == 1, (
        f"expected exactly one ``- **{bullet_label}**: `<command>``` bullet "
        f"between {begin!r} and {end!r} in CLAUDE.md, found {len(spans)}: "
        f"{spans!r} (task 3959). The marker must wrap that bullet and nothing "
        f"else; if the bullet was relabelled or the markers were inverted, move "
        f"the marker back around the copy-pasteable command."
    )

    command = spans[0].strip()
    assert command, (
        f"the command between {begin!r} and {end!r} in CLAUDE.md is empty "
        f"(task 3959)"
    )
    return command


# Hand-written fixtures. The marker literals are spelled out IN FULL rather than
# interpolated from the constants above, per the precedent's stated reason: "a
# rename must not be able to silently keep a broken parser agreeing with its own
# fixtures." They are never the real CLAUDE.md, so they stay stable under any
# future edit to it.
_HAPPY_DOC = """\
Ask the interpreter; never guess a path under `.venv` or `site-packages`.

<!-- package-source-lookup:begin
     EXECUTED by tests/scripts/test_package_source_lookup_convention.py. Edit
     this into anything that is not a `python -c` query and that guard goes
     red. -->
- **Third-party package source**: `python -c 'import alpha, os; print(os.path.dirname(alpha.__file__))'`
<!-- package-source-lookup:end -->

Never `find /` for an installed package.
"""

_HAPPY_COMMAND = "python -c 'import alpha, os; print(os.path.dirname(alpha.__file__))'"

_NO_MARKER_DOC = """\
- **Third-party package source**: `python -c 'import alpha, os; print(os.path.dirname(alpha.__file__))'`

Somewhere else entirely, an unmarked mention of `site-packages`.
"""

_DUPLICATE_MARKER_DOC = """\
<!-- package-source-lookup:begin -->
- **Third-party package source**: `python -c 'import alpha; print(alpha.__file__)'`
<!-- package-source-lookup:end -->

## Some later section

<!-- package-source-lookup:begin -->
- **Third-party package source**: `python -c 'import beta; print(beta.__file__)'`
<!-- package-source-lookup:end -->
"""

_DECOY_DOC = """\
Do not run `find / -path '*/alpha/config.py'` — it hits the Bash timeout.

<!-- package-source-lookup:begin
     Mirrors nothing; it is EXECUTED as written by
     tests/scripts/test_package_source_lookup_convention.py. Contains prose
     inline code such as `site-packages` and `python -c 'import os'` that a
     backtick-keyed extractor would grab first. -->
- **Third-party package source**: `python -c 'import alpha, os; print(os.path.dirname(alpha.__file__))'`
<!-- package-source-lookup:end -->

Afterwards, `grep -rn 'timeout_keep_alive' <that dir>/config.py`.
"""


def test_marked_command_extracts_the_marked_span():
    """(a) Only the marked bullet's command is returned, backticks stripped."""
    assert (
        _marked_command(_HAPPY_DOC, PACKAGE_SOURCE_MARKER, PACKAGE_SOURCE_LABEL)
        == _HAPPY_COMMAND
    )


@pytest.mark.parametrize(
    ("markdown_text", "case"),
    [
        (_NO_MARKER_DOC, "missing"),
        (_DUPLICATE_MARKER_DOC, "duplicated"),
    ],
)
def test_marked_command_fails_loudly_on_a_broken_marker(markdown_text, case):
    """(b, c) A missing or duplicated marker RAISES — never '' or None.

    Missing is the vacuity hazard: an extractor that silently returns nothing
    turns every downstream assertion green while executing nothing at all.
    Duplicated is the same failure one level down: silently taking the first
    leaves the second copy unexecuted and free to rot. The message must tell a
    human what to restore and where.
    """
    with pytest.raises(AssertionError) as excinfo:
        _marked_command(markdown_text, PACKAGE_SOURCE_MARKER, PACKAGE_SOURCE_LABEL)

    message = str(excinfo.value)
    assert PACKAGE_SOURCE_MARKER in message, case
    assert "CLAUDE.md" in message, case


def test_marked_command_is_immune_to_inline_code_in_the_begin_comment():
    """(d) Prose inline code inside the marked slice is never extracted.

    An extractor keyed on "the first backtick span after the begin marker" would
    return ``site-packages`` here — a plausible-looking string, so the mistake
    would survive review and then be executed as a command.
    """
    assert (
        _marked_command(_DECOY_DOC, PACKAGE_SOURCE_MARKER, PACKAGE_SOURCE_LABEL)
        == _HAPPY_COMMAND
    )


def _interpreter_query_body(command, marker):
    """The ``-c`` body of *command*, after pinning that it IS an interpreter query.

    This is the load-bearing SHAPE assertion: a ``find``-shaped recipe, a
    hardcoded ``<venv>/lib/...`` path, or a shell pipeline fails here rather than
    being executed. It is what stops the documented convention from silently
    degrading back into the whole-filesystem scan this task retired.
    """
    argv = shlex.split(command)
    assert len(argv) == 3 and argv[0] in _PYTHON_ARGV0 and argv[1] == "-c", (
        f"the command inside the {marker!r} marker in CLAUDE.md is not an "
        f"interpreter query (task 3959): {command!r} tokenises to {argv!r}, "
        f"expected exactly [python|python3, '-c', <body>]. The whole point of "
        f"this convention is that locating installed code is a ~0.4s import "
        f"query, NOT a filesystem scan — the `find /` it replaced hit the "
        f"120000ms Bash default and returned nothing (exit 143)."
    )
    body = argv[2].strip()
    assert body, f"the {marker!r} command in CLAUDE.md has an empty -c body (task 3959)"
    return body


def _run_documented_query(body, marker):
    """Execute *body* and return its stripped stdout, or fail loudly.

    ``sys.executable``, NEVER a bare ``python`` off PATH: ``verify.
    _target_subprocess_env`` strips ``VIRTUAL_ENV`` and the venv ``bin`` dir from
    PATH so a target resolves its own ``.venv``, so a PATH-resolved ``python``
    could be a system interpreter with none of these packages installed and would
    go falsely red under verify. See the module docstring.
    """
    try:
        completed = subprocess.run(
            [sys.executable, "-c", body],
            capture_output=True,
            text=True,
            timeout=_QUERY_TIMEOUT_SECS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        # The scoping gate. A scoped import query returns in well under a second;
        # anything that runs for a minute has degraded into a filesystem scan,
        # which is the exact failure this convention exists to prevent.
        pytest.fail(
            f"the {marker!r} command documented in CLAUDE.md did not finish "
            f"within {_QUERY_TIMEOUT_SECS}s (task 3959), against a measured "
            f"0.396s. That is the signature of a filesystem scan rather than an "
            f"interpreter query; body: {body!r}"
        )

    assert completed.returncode == 0, (
        f"the {marker!r} command documented in CLAUDE.md exited "
        f"{completed.returncode} (task 3959) — agents are being handed a recipe "
        f"that does not work.\n"
        f" interpreter: {sys.executable}\n"
        f" body: {body!r}\n"
        f" stdout: {completed.stdout.strip()!r}\n"
        f" stderr: {completed.stderr.strip()!r}"
    )
    resolved = completed.stdout.strip()
    assert resolved, (
        f"the {marker!r} command documented in CLAUDE.md exited 0 but printed "
        f"nothing (task 3959) — it would leave an agent with no answer at all; "
        f"body: {body!r}"
    )
    return resolved


def test_documented_package_source_lookup_resolves_a_third_party_package():
    """CLAUDE.md's third-party recipe must actually resolve into ``site-packages``.

    Executed, not string-compared: this certifies the recipe WORKS in this
    environment, which is the claim CLAUDE.md is making to every agent that reads
    it. MEASURED at task 3959: resolves ``<venv>/lib/python3.13/site-packages/
    uvicorn`` in 0.396s, against a ``find /`` that returned nothing in 120s.
    """
    command = _marked_command(
        CLAUDE_MD_PATH.read_text(encoding="utf-8"),
        PACKAGE_SOURCE_MARKER,
        PACKAGE_SOURCE_LABEL,
    )
    body = _interpreter_query_body(command, PACKAGE_SOURCE_MARKER)
    resolved = _run_documented_query(body, PACKAGE_SOURCE_MARKER)

    resolved_path = pathlib.Path(resolved)
    assert resolved_path.is_dir(), (
        f"the {PACKAGE_SOURCE_MARKER!r} command documented in CLAUDE.md printed "
        f"{resolved!r}, which is not an existing directory (task 3959) — an "
        f"agent following it would `cd`/grep into nothing"
    )
    assert "site-packages" in resolved_path.parts, (
        f"the {PACKAGE_SOURCE_MARKER!r} command documented in CLAUDE.md resolved "
        f"to {resolved!r}, which is not under a `site-packages` directory (task "
        f"3959). This marker documents where THIRD-PARTY package source lives; "
        f"if the probed package became first-party (installed editable, so it "
        f"resolves into a checkout's src/ tree), the recipe no longer "
        f"demonstrates what the surrounding section claims — probe a genuinely "
        f"third-party package instead."
    )


def _workspace_member_modules():
    """Importable module names of the ``[tool.uv.workspace].members`` directories.

    Read live from the root ``pyproject.toml`` rather than hardcoded, so that a
    workspace-member rename cannot silently un-test the provenance check: the
    cross-check below goes loudly red naming the live member set instead.

    Normalises directory name -> module name with ``.replace("-", "_")``, since
    ``fused-memory`` imports as ``fused_memory``. Both the member list and every
    normalised name are asserted non-empty — a discovery that silently matches
    nothing turns every downstream assertion green while checking nothing at all,
    which is strictly worse than no guard because the check still reports
    success.

    ``tomllib`` is stdlib on 3.11+ and this repo requires ``>=3.11``, so no
    dependency is added to a suite that runs under ``uv run --project shared``.
    """
    data = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    members = data["tool"]["uv"]["workspace"]["members"]
    assert members, (
        f"[tool.uv.workspace].members in {PYPROJECT_PATH} is empty (task 3959) — "
        f"the first-party cross-check below would pass vacuously"
    )

    modules = {member.rstrip("/").rsplit("/", 1)[-1].replace("-", "_") for member in members}
    assert all(modules), (
        f"a workspace member in {PYPROJECT_PATH} normalised to an empty module "
        f"name (task 3959): members={members!r}"
    )
    return modules


def test_documented_import_provenance_check_resolves_to_a_checkout_source_tree():
    """CLAUDE.md's first-party recipe must resolve into a checkout's ``src/`` tree.

    This is the DISCRIMINATING half, and the reason the two markers are separate:
    a first-party workspace member is installed EDITABLE, so it lands in a
    checkout's source tree, which is the exact inverse of the third-party
    assertion above. That is also the concrete mechanism behind the OPERATIONS.md
    Troubleshooting symptom "a task blocks at VERIFY with ``AttributeError`` for
    code it just wrote" — from a worktree shell that inherited main's
    ``VIRTUAL_ENV``, a first-party import gives you MAIN's source, not the
    worktree edits.

    DELIBERATELY NOT ASSERTED: that the resolved path is under ``REPO_ROOT``. In
    a cold-verified worktree running its own ``.venv`` the editable install
    correctly points at THAT worktree; from an un-synced worktree it correctly
    resolves to the main checkout. Both are valid, and pinning either would go
    red in the other environment.
    """
    command = _marked_command(
        CLAUDE_MD_PATH.read_text(encoding="utf-8"),
        IMPORT_PROVENANCE_MARKER,
        IMPORT_PROVENANCE_LABEL,
    )
    body = _interpreter_query_body(command, IMPORT_PROVENANCE_MARKER)

    # The probe must be FIRST-PARTY, cross-checked against the live workspace
    # table. Probing a third-party package here would make the site-packages
    # assertion below fail for the right-looking wrong reason.
    imported = _IMPORTED_MODULE.findall(body)
    assert imported, (
        f"could not find an `import <module>` in the {IMPORT_PROVENANCE_MARKER!r} "
        f"command documented in CLAUDE.md (task 3959): {body!r}"
    )
    probed = imported[0]
    members = _workspace_member_modules()
    assert probed in members, (
        f"the {IMPORT_PROVENANCE_MARKER!r} command documented in CLAUDE.md probes "
        f"{probed!r}, which is not a workspace member (task 3959). Live members, "
        f"normalised to module names: {sorted(members)}. This marker documents "
        f"FIRST-PARTY import provenance — editable installs resolving into a "
        f"checkout's src/ tree — so it must probe a workspace member. If a member "
        f"was renamed, update the command in CLAUDE.md to match."
    )

    resolved = _run_documented_query(body, IMPORT_PROVENANCE_MARKER)
    resolved_path = pathlib.Path(resolved)
    assert resolved_path.exists(), (
        f"the {IMPORT_PROVENANCE_MARKER!r} command documented in CLAUDE.md printed "
        f"{resolved!r}, which does not exist (task 3959). A printed 'None' means "
        f"{probed!r} was NOT installed in {sys.executable}'s environment and the "
        f"repo-root directory of the same name was picked up as a NAMESPACE "
        f"package instead — provision the venv (`uv sync --all-packages`) rather "
        f"than weakening this assertion."
    )
    assert "site-packages" not in resolved_path.parts, (
        f"the {IMPORT_PROVENANCE_MARKER!r} command documented in CLAUDE.md resolved "
        f"{probed!r} to {resolved!r}, which is UNDER site-packages (task 3959). "
        f"This marker exists to demonstrate the opposite of the "
        f"{PACKAGE_SOURCE_MARKER!r} one: a first-party workspace member is "
        f"installed editable, so it must resolve into a checkout's source tree. A "
        f"copied-in (non-editable) install would silently give agents main's "
        f"stale code, which is the failure this check surfaces."
    )
    assert "src" in resolved_path.parts, (
        f"the {IMPORT_PROVENANCE_MARKER!r} command documented in CLAUDE.md resolved "
        f"{probed!r} to {resolved!r}, which has no `src/` segment (task 3959) — "
        f"this repo's members all live at <pkg>/src/<pkg>/, so the resolved path "
        f"does not look like a checkout source tree at all. Which tree it actually "
        f"resolved to is named above; read it before changing anything."
    )
